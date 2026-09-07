"""train.py -- 训练入口骨架（main 模块维护；D1-D4 首期用途）。

用途
----
打通「config → build_network → run_sample」链路，作为 G0 调试与单样本冒烟的
命令行入口（H-STDN 实施计划 D4 阶段）：

    1. 加载 configs/default.yaml（hstdn.configs，§4 超参唯一契约）；
    2. ``to_core_cfg()`` -> core.layout.NetConfig；
    3. ``build_network()`` 构建 M1 网络（NetworkBundle，构造时内嵌结构性自检）；
    4. data.synthetic 生成合成 10 类输入（D3/D14 布局），逐样本 latency
       编码（M2）后 ``run_sample``（M3）驱动储备池；
    5. 输出率/沉默/可塑性诊断（复用 hstdn.exp.diagnostics）；
    6. 保存 NetworkBundle checkpoint（npz，含配置指纹与运行元数据）。

算法纪律：本模块**只做编排不做算法实现**；LIF/STDP/homeostasis/norm 全部
调用 hstdn.core 接口（core 为权威实现）。可复现：固定随机种子同时驱动
网络构建（build rng）与每 epoch 数据生成（seed + epoch 派生），运行参数与
配置指纹全部落盘。

M6 调度器接入状态（已实现）
--------------------------
core.scheduler（M6，core 模块交付）已实现完整 G1 协议状态机
``run_g1_protocol``（ADAPT→CALIBRATE→COLLECT→READOUT→EVAL，含轮次回滚 /
eta 衰减）；本入口以 ``--g1`` 消费它（见 :func:`run_g1_training`）。默认
（无 --g1）仍为 D4 冒烟 flat 循环 :func:`_training_epochs`（只跑单样本
链路，不消费 protocol 节）。协议超参唯一来源为 default.yaml ``protocol``
节（由 :func:`_protocol_kwargs_from_yaml` 翻译，禁止散落魔法数字）。

命令行（项目根 E:\\neuron3d）
    python -m hstdn.train --epochs 1 --samples 10          # D4 冒烟
    python -m hstdn.train --g1 --classes 4 --train-samples 60 ^
                         --test-samples 20 --seeds 1       # G1 协议训练
    python -m hstdn.train --help
    python -m hstdn.train --selfcheck        # 指纹 + checkpoint 编解码自检
退出码：0 = 成功；1 = 运行期失败（异常上抛并打印）；2 = 参数错误。

checkpoint npz 格式（hstdn-network-bundle-v1；G1 附加读出数组）
    bundle 全部数组字段 + 两条 0-d 字符串记录：
      ``__meta__``：meta JSON（format/mode/fingerprint/run_fingerprint/
                    run_args/config_json/protocol/stage_sequence/
                    rolled_back/best_acc/各阶段读数/derived/...）；
      ``__cfg__``：NetConfig 全部字段 JSON（含派生字段；加载时以
                   ``with_derived()`` 复验一致性）。
    G1 checkpoint（meta.mode == "g1"）另存读出权重 W（(d, C)）与偏置 b
    （(C,)），供 eval.py 独立评估复用；config 指纹与运行参数指纹保证
    可复现命名（同一配置 + 同一运行参数覆盖写）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from hstdn.configs import load_config, to_core_cfg
from hstdn.core.kernel import run_sample
from hstdn.core.layout import F8, I8, NetConfig
from hstdn.core.network import NetworkBundle, build_network, print_report
from hstdn.data.synthetic import N_CLASSES, synthetic_batch
from hstdn.exp import diagnostics as diag

__all__ = [
    "CKPT_FORMAT",
    "config_fingerprint",
    "save_checkpoint",
    "load_checkpoint",
    "load_g1_checkpoint",
    "sample_aggregate_panel",
    "run_training",
    "run_g1_training",
    "main",
    "build_parser",
]

#: checkpoint 格式标记（npz 内 meta["format"] 校验，防误读/跨版本漂移）
CKPT_FORMAT = "hstdn-network-bundle-v1"

#: 每 epoch 数据派生 seed 的步长（素数；seed 流 = base + epoch * STEP）
_EPOCH_SEED_STEP = 7919

#: 运行期默认值（非 §4 超参；超参唯一来源是 default.yaml）
_DEF_EPOCHS = 1
_DEF_SAMPLES = 10        # 每 epoch 样本总数（冒烟规模）
_DEF_NOISE = 0.02        # 训练强度噪声 std（高斯，截断 [0,1]）
_DEF_SEED = 0
_DEF_OUT_DIR = "checkpoints"


def _json_safe(obj: Any) -> Any:
    """把 numpy 标量/数组递归转换为原生 JSON 类型（np.savez 元数据需要）。"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    raise TypeError(f"_json_safe: 不支持的类型 {type(obj).__name__}: {obj!r}")


def config_fingerprint(cfg: Mapping[str, Any]) -> str:
    """配置指纹：sha256(规范化 JSON) —— checkpoint 与评估方的对齐契约。

    Args:
        cfg: hstdn.configs.load_config 返回的顶层映射。

    Returns:
        64 位小写 hex 摘要（排序键 + 紧凑分隔的确定性序列化）。
    """
    canonical = json.dumps(
        _json_safe(cfg), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_fingerprint(run_args: Mapping[str, Any]) -> str:
    """运行参数指纹（seed/样本数/开关等）—— 与 config 指纹区分。"""
    canonical = json.dumps(
        _json_safe(run_args), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# checkpoint 编解码（npz；train 保存 / eval 加载共用同一 schema）
# ---------------------------------------------------------------------------


def save_checkpoint(bundle: NetworkBundle, meta: Mapping[str, Any],
                    path: str | Path,
                    *, extra: Optional[Mapping[str, np.ndarray]] = None) -> Path:
    """把 NetworkBundle（+可选 extra 数组）与元数据写入 npz checkpoint。

    Args:
        bundle: 训练结束的网络（全部数组字段落盘）。
        meta: JSON 可序列化元数据；至少含 format/fingerprint/run_fingerprint。
        path: 目标文件路径（父目录须已存在或由调用方创建）。
        extra: 可选附加数组（如 G1 读出权重 W/b）；键不得与 bundle 字段或
            ``__`` 保留键冲突（断言保护）。

    Returns:
        实际写入的 Path。

    Raises:
        AssertionError: bundle 数组与 cfg 规模不一致 / extra 键冲突。
    """
    out = Path(path)
    _assert_bundle_shapes(bundle)

    arrays: Dict[str, np.ndarray] = {}
    for f in fields(NetworkBundle):
        if f.name == "cfg":
            continue
        arrays[f.name] = np.asarray(getattr(bundle, f.name))  # 数组或 0-d 标量
    for key, val in (extra or {}).items():
        if key in arrays or key.startswith("__"):
            raise AssertionError(
                f"extra 数组键 {key!r} 与 bundle 字段/保留键冲突"
            )
        arrays[key] = np.asarray(val)

    cfg_json = json.dumps({
        "fields": _json_safe(
            {f.name: getattr(bundle.cfg, f.name) for f in fields(NetConfig)}
        ),
    }, sort_keys=True)
    meta_json = json.dumps(_json_safe(dict(meta)), sort_keys=True)
    np.savez(out, **arrays, __meta__=np.asarray(meta_json),
             __cfg__=np.asarray(cfg_json))
    return out


def _assert_bundle_shapes(bundle: NetworkBundle) -> None:
    """bundle 数组 ↔ cfg 规模的一致性闸（checkpoint 完整性 G0 类断言）。

    Raises:
        AssertionError: 任一形状不符（附统计数值的可读消息）。
    """
    cfg = bundle.cfg
    n_total, n_pool = cfg.n_total, cfg.n_pool
    if bundle.ring.shape != (cfg.wheel_l, n_pool):
        raise AssertionError(
            f"ring shape {bundle.ring.shape} != wheel ({cfg.wheel_l}, {n_pool})"
        )
    if bundle.trace.shape != (n_total,):
        raise AssertionError(
            f"trace shape {bundle.trace.shape} != (N_TOTAL={n_total},)"
        )
    if bundle.csr_ptr.shape != (n_total + 1,):
        raise AssertionError(
            f"csr_ptr shape {bundle.csr_ptr.shape} != (N_TOTAL+1={n_total + 1},)"
        )
    e = int(bundle.csr_w.size)
    if e != int(bundle.csr_ptr[-1]) or e != int(bundle.csc_ptr[-1]):
        raise AssertionError(
            f"edge count mismatch: csr_w={e}, csr_ptr[-1]={bundle.csr_ptr[-1]}, "
            f"csc_ptr[-1]={bundle.csc_ptr[-1]}"
        )
    if bundle.csc_ptr.shape != (n_pool + 1,):
        raise AssertionError(
            f"csc_ptr shape {bundle.csc_ptr.shape} != (N_POOL+1={n_pool + 1},)"
        )
    for name, arr in (("csr_dst", bundle.csr_dst), ("csr_src", bundle.csr_src)):
        if arr.size:
            if int(arr.min()) < 0 or int(arr.max()) >= n_total:
                raise AssertionError(
                    f"{name} 越出全局 ID 域 [0, {n_total}): "
                    f"实际 [{int(arr.min())}, {int(arr.max())}]"
                )
    if bundle.csr_dst.size and int(bundle.csr_dst.min()) < cfg.n_in:
        raise AssertionError(
            f"csr_dst 违反池目标契约：min={int(bundle.csr_dst.min())} "
            f"< N_IN={cfg.n_in}"
        )


def load_checkpoint(path: str | Path) -> tuple[NetworkBundle, Dict[str, Any]]:
    """从 npz 加载 NetworkBundle 与 meta（train.save_checkpoint 的逆操作）。

    Args:
        path: checkpoint npz 路径。

    Returns:
        (bundle, meta)：bundle 的 cfg 由落盘字段重建并经 with_derived 复验；
        meta 为字典（含 format/fingerprint/run_args/config_json）。

    Raises:
        AssertionError: 格式标记不符 / 数组与重建 cfg 规模不一致。
        FileNotFoundError: 文件不存在。
    """
    bundle, meta, _extra = _read_checkpoint(path)
    return bundle, meta


def load_g1_checkpoint(path: str | Path
                       ) -> tuple[NetworkBundle, Dict[str, Any], Dict[str, Any]]:
    """加载 G1 协议 checkpoint：bundle + meta + 读出权重等附加数组。

    Args:
        path: train --g1 产出的 npz 路径（meta.mode == "g1"）。

    Returns:
        (bundle, meta, extras)：extras 含 W（(d, C)）、b（(C,)）读出参数
        及其它非 bundle 数组。

    Raises:
        AssertionError: 不是 G1 checkpoint（mode 不符或 W/b 缺失）。
    """
    bundle, meta, extra = _read_checkpoint(path)
    if meta.get("mode") != "g1":
        raise AssertionError(
            f"{Path(path).name} 不是 G1 协议 checkpoint（meta.mode="
            f"{meta.get('mode')!r}，期望 'g1'；请先用 train --g1 训练）"
        )
    missing = [k for k in ("W", "b") if k not in extra]
    if missing:
        raise AssertionError(
            f"G1 checkpoint 缺少读出数组 {missing}（{Path(path).name}）"
        )
    return bundle, meta, extra


def _read_checkpoint(path: str | Path
                     ) -> tuple[NetworkBundle, Dict[str, Any], Dict[str, Any]]:
    """底层 npz 读取：bundle 字段按名重建，其余数组键归入 extras。"""
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        meta_raw = z["__meta__"].item()
        cfg_raw = z["__cfg__"].item()
        if not isinstance(meta_raw, str) or not isinstance(cfg_raw, str):
            raise AssertionError(
                f"{p.name} 不是 hstdn checkpoint（__meta__/__cfg__ 缺失）"
            )
        meta = json.loads(meta_raw)
        if meta.get("format") != CKPT_FORMAT:
            raise AssertionError(
                f"checkpoint 格式标记不符：{meta.get('format')!r} != "
                f"{CKPT_FORMAT!r}（{p.name}）"
            )
        cfg_fields = json.loads(cfg_raw)["fields"]
        cfg = NetConfig(**cfg_fields).with_derived()

        bundle_fields = {f.name for f in fields(NetworkBundle)
                         if f.name != "cfg"}
        kw: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for name in z.files:
            if name in ("__meta__", "__cfg__"):
                continue
            arr = z[name]
            if name in bundle_fields:
                kw[name] = int(arr) if arr.ndim == 0 else np.asarray(arr)
            else:
                extra[name] = np.asarray(arr)
        bundle = NetworkBundle(cfg=cfg, **kw)
    _assert_bundle_shapes(bundle)
    return bundle, meta, extra
# ---------------------------------------------------------------------------
# 训练编排（只做调度；算法一律走 core）
# ---------------------------------------------------------------------------


def sample_aggregate_panel(counts_mat: Any, window_s: float) -> Dict[str, float]:
    """跨样本聚合面板：率/沉默（诚实处理均值，不做整数截断）。

    Args:
        counts_mat: (S, N) 每样本池发放计数矩阵（int 或 float 均可）。
        window_s: 统计窗秒数（契约 0.2，T=200 ms）。

    Returns:
        dict：n_samples / total_spikes / mean_rate_hz（全池全体样本均值）/
        mean_active_rate_hz（逐样本活跃均值再平均）/ silence_ratio（逐样本
        沉默比再平均）/ window_s。
    """
    c = np.asarray(counts_mat, dtype=np.float64)
    if c.ndim != 2:
        raise AssertionError(
            f"counts_mat 需为 (S, N) 矩阵，实际 shape={c.shape}"
        )
    s, n = c.shape
    rates_all = c.sum() / (float(s) * float(n) * window_s)
    active_rate = np.where(c > 0.0, c / window_s, 0.0)
    denom = np.count_nonzero(c > 0.0, axis=1)
    active_mean_per_s = np.divide(
        active_rate.sum(axis=1), denom,
        out=np.zeros(s, dtype=np.float64), where=denom > 0,
    )
    sil = (c == 0.0).mean(axis=1)
    return {
        "n_samples": float(s),
        "n_neurons": float(n),
        "total_spikes": float(c.sum()),
        "mean_rate_hz": float(rates_all),
        "mean_active_rate_hz": float(active_mean_per_s.mean()) if s else 0.0,
        "silence_ratio": float(sil.mean()) if s else 0.0,
        "window_s": float(window_s),
    }


def _encode_frames(frames: Any, cfg: NetConfig) -> List[Dict[int, Any]]:
    """逐帧 latency 编码（M2 输入契约；帧行主序展平 -> BucketMap 列表）。"""
    from hstdn.core.encoder import latency_encode
    return [latency_encode(np.asarray(fr, dtype=F8).ravel(), cfg=cfg)
            for fr in frames]


def _training_epochs(bundle: NetworkBundle, netcfg: NetConfig, *, epochs: int,
                     samples: int, n_per_class: int, noise: float,
                     graded: bool, base_seed: int, stdp_on: bool,
                     homeo_on: bool, norm_on: bool, quiet: bool
                     ) -> List[Dict[str, Any]]:
    """均匀 flat 训练循环（仅默认/D4 冒烟路径使用）。

    M6 调度器接入状态：G1 完整协议（ADAPT→CALIBRATE→COLLECT→READOUT→
    EVAL）已由 core.scheduler.run_g1_protocol 实现并经
    :func:`run_g1_training`（--g1）接入；本 flat 循环不消费 default.yaml
    ``protocol`` 节，只为 D4 config→run_sample 链路冒烟保留（协议超参唯一
    来源仍是 default.yaml protocol 节）。

    Args:
        bundle/netcfg: 网络与其配置（bundle 就地演化：csr_w/theta/rate_ema）。
        epochs: 训练轮数（每轮数据独立生成：seed = base_seed + epoch*STEP）。
        samples: 每轮使用的样本总数（从按类分块的批次头部截取）。
        n_per_class: 每轮每类样本数（>= ceil(samples / N_CLASSES)）。
        noise/graded: 合成数据噪声与渐变参数（data.synthetic）。
        base_seed: 数据派生种子基准。
        stdp_on/homeo_on/norm_on: run_sample 三开关。
        quiet: 关闭逐样本打印。

    Returns:
        每 epoch 的聚合面板 dict 列表。
    """
    epochs_summary: List[Dict[str, Any]] = []
    win = float(netcfg.window_s)
    T = int(round(win * 1000.0))
    for ep in range(int(epochs)):
        ep_seed = int(base_seed) + ep * _EPOCH_SEED_STEP
        batch = synthetic_batch(int(n_per_class), noise=float(noise),
                                graded=bool(graded), seed=ep_seed)
        # 从按类分块批次头部截取前 samples 个样本（确定性）
        n_use = min(int(samples), len(batch))
        frames = batch.frames[:n_use]
        labels = batch.labels[:n_use]
        buckets = _encode_frames(frames, netcfg)

        counts_mat = np.zeros((n_use, netcfg.n_pool), dtype=I8)
        if not quiet:
            print(f"--- epoch {ep + 1}/{epochs} "
                  f"(seed={ep_seed}, samples={n_use}, "
                  f"stdp={stdp_on}, homeo={homeo_on}, norm={norm_on}) ---")
        for i in range(n_use):
            stats = run_sample(bundle, buckets[i], T=T,
                               stdp_on=stdp_on, homeo_on=homeo_on,
                               norm_on=norm_on)
            counts_mat[i] = stats["spike_counts"]
            if not quiet:
                n_spk = int(stats["n_spikes_total"])
                print(f"  sample {i + 1:>3}/{n_use} class {int(labels[i])} "
                      f"spikes {n_spk:>5}")

        panel = sample_aggregate_panel(counts_mat, window_s=win)
        panel["epoch"] = float(ep + 1)
        panel["n_samples"] = float(n_use)
        epochs_summary.append(panel)
        if not quiet:
            diag.print_panel(panel, title=f"epoch {ep + 1} aggregate")
            diag.print_panel(diag.plasticity_panel(bundle),
                             title="plasticity after epoch")
    return epochs_summary


# ---------------------------------------------------------------------------
# G1 协议训练（--g1；消费 core.scheduler.run_g1_protocol —— M6 已实现）
# ---------------------------------------------------------------------------

#: G1 默认运行参数（非协议超参；协议超参唯一来源 = default.yaml protocol 节）
_G1_DEF_POOL = 200            # G1 门禁 light 规模（exp.ablation 同款约定；
                              # §4 完整 800 池请显式 --pool 800，注意 L0 耗时）
_G1_DEF_CLASSES = N_CLASSES   # 合成 10 类（data.synthetic 契约）
_G1_DEF_TRAIN_SAMPLES = 120   # 默认训练样本总数（10 类 × 12/类）
_G1_DEF_TEST_SAMPLES = 100    # 默认协议 EVAL 测试样本总数（10 类 × 10/类）
_G1_DEF_SEEDS = (0, 1)        # 默认双种子（消融惯例 >= 2 种子）
#: 数据派生种子偏移（与 exp/ablation.make_data_fns 完全同款，保证可比）
_G1_TRAIN_SEED_OFFSET = 1000
_G1_TEST_SEED_OFFSET = 2000

#: default.yaml protocol 节键 → core.scheduler ProtocolConfig 字段（唯一翻译）
_PROTOCOL_KEY_MAP = {
    "adapt_epochs": "n_adapt_epochs",
    "calibrate_samples": "calibrate_samples",
    "calibrate_max_iter": "calibrate_max_iter",
    "extra_loops": "extra_loops",
}
_PROTOCOL_BAND_KEY = "calibrate_band"


def _parse_seeds(text: str) -> List[int]:
    """种子列表解析（逗号/分号分隔；供 argparse type= 使用）。"""
    parts = [p for p in str(text).replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("empty seeds list")
    return [int(p) for p in parts]


def _protocol_kwargs_from_yaml(cfg_yaml: Mapping[str, Any]) -> Dict[str, Any]:
    """把 default.yaml ``protocol`` 节翻译成 ProtocolConfig 键值。

    映射：adapt_epochs→n_adapt_epochs；calibrate_samples / calibrate_max_iter /
    extra_loops 同名直取；calibrate_band=[lo, hi]→calib_rate_lo/calib_rate_hi；
    样本时长 t_ms 取 lif.T（§4 契约，单位 ms）。本节未出现的其余 M6 旋钮归
    core.scheduler ProtocolConfig 的 §4 默认（防漂移）；未知键显式报错。

    Raises:
        AssertionError: protocol 节非 mapping / 含未知键 / band 形状非法。
    """
    proto = cfg_yaml.get("protocol")
    if proto is None:
        return {}
    if not isinstance(proto, Mapping):
        raise AssertionError(
            f"config protocol 节必须是 mapping，got {type(proto).__name__}"
        )
    known = set(_PROTOCOL_KEY_MAP) | {_PROTOCOL_BAND_KEY}
    unknown = sorted(set(proto) - known)
    if unknown:
        raise AssertionError(
            "config protocol 节含 main 未消费的键 "
            f"{unknown}（已消费 {sorted(known)}；其余 M6 旋钮由 "
            "core.scheduler ProtocolConfig 默认承载）"
        )
    out: Dict[str, Any] = {}
    for yk, pk in _PROTOCOL_KEY_MAP.items():
        if yk in proto:
            out[pk] = int(proto[yk])
    band = proto.get(_PROTOCOL_BAND_KEY)
    if band is not None:
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise AssertionError(f"calibrate_band 需为 [lo, hi]，got {band!r}")
        out["calib_rate_lo"] = float(band[0])
        out["calib_rate_hi"] = float(band[1])
    lif = cfg_yaml.get("lif") or {}
    out["t_ms"] = int(lif.get("T", 200)) if lif else 200
    return out


def _g1_data_fns(n_classes: int, train_tpc: int, test_tpc: int, noise: float,
                 graded: bool, seed: int) -> tuple[Callable, Callable]:
    """零参 train/test data_fn（(frame, label)；约定同 exp.ablation）。

    synthetic_batch 固定产出 data.synthetic 的 N_CLASSES=10 类分块批次；
    本函数只消费前 ``n_classes`` 个类块（class 升序、类内顺序稳定），使
    每类恰好 train_tpc/test_tpc 个样本 —— 总样本数 = n_classes × tpc。
    批次种子派生：train = 1000 + seed、test = 2000 + seed。每次调用返回
    新迭代器，帧以拷贝给出 —— 全程确定性、不重开随机流。

    Raises:
        ValueError: n_classes 超出 data.synthetic N_CLASSES（调用前已校验）。
    """
    if not 1 <= int(n_classes) <= N_CLASSES:
        raise ValueError(
            f"n_classes ∈ [1, {N_CLASSES}]，got {n_classes}"
        )
    tr = synthetic_batch(int(train_tpc), noise=float(noise),
                         graded=bool(graded),
                         seed=_G1_TRAIN_SEED_OFFSET + int(seed))
    te = synthetic_batch(int(test_tpc), noise=float(noise),
                         graded=bool(graded),
                         seed=_G1_TEST_SEED_OFFSET + int(seed))

    def train_fn() -> Iterable[tuple]:
        for k in range(int(n_classes)):
            base = k * int(train_tpc)
            for i in range(int(train_tpc)):
                yield tr.frames[base + i].copy(), int(k)

    def test_fn() -> Iterable[tuple]:
        for k in range(int(n_classes)):
            base = k * int(test_tpc)
            for i in range(int(test_tpc)):
                yield te.frames[base + i].copy(), int(k)

    return train_fn, test_fn


def _derive_readout(bundle: NetworkBundle, train_fn: Callable,
                    test_fn: Callable, *, seed: int,
                    feat_mode: str = "v0", feat_n_bins: int = 5
                    ) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """协议结束后确定性重导出读出权重 W/b 与自洽测试精度。

    run_g1_protocol 的每轮读出权重不对外暴露（G1Report 仅回传轮次统计）；
    本函数以**公开 core API**（run_sample / build_features /
    train_linear_readout / predict_linear_readout / accuracy）按相同确定性
    流程（同数据流 + ReadoutConfig(seed)）重放 COLLECT→READOUT→EVAL，
    得到与协议轮一致（发生回滚时 = best 轮）的 W/b。属编排而非算法实现。

    Returns:
        (W, b, info)：W (d, C)、b (C,)；info 含 derived_acc / n_train /
        feat_dim / n_class / readout_best_val_acc 等。
    """
    from hstdn.core.encoder import latency_encode
    from hstdn.core.features import FeaturesConfig, build_features
    from hstdn.core.readout import (
        ReadoutConfig, accuracy, predict_linear_readout, train_linear_readout,
    )
    cfg = bundle.cfg
    t_ms = int(round(float(cfg.window_s) * 1000.0))
    fcfg = FeaturesConfig(mode=feat_mode, n_bins=feat_n_bins)

    rows: List[np.ndarray] = []
    labs: List[int] = []
    for frame, lab in train_fn():
        buckets = latency_encode(np.asarray(frame, dtype=F8).ravel(), cfg=cfg)
        st = run_sample(bundle, buckets, T=t_ms, stdp_on=False,
                        homeo_on=False, norm_on=False)
        rows.append(st["spike_counts"].astype(F8))
        labs.append(int(lab))
    if not rows:
        raise RuntimeError("G1 derive COLLECT: train data_fn 无样本")
    counts = np.stack(rows, axis=0)
    y = np.asarray(labs, dtype=I8)
    X = build_features(counts, fcfg)
    W, b, hist = train_linear_readout(X, y, None, None,
                                      ReadoutConfig(seed=int(seed)))

    rows2: List[np.ndarray] = []
    labs2: List[int] = []
    for frame, lab in test_fn():
        buckets = latency_encode(np.asarray(frame, dtype=F8).ravel(), cfg=cfg)
        st = run_sample(bundle, buckets, T=t_ms, stdp_on=False,
                        homeo_on=False, norm_on=False)
        rows2.append(st["spike_counts"].astype(F8))
        labs2.append(int(lab))
    Xt = build_features(np.stack(rows2, axis=0), fcfg)
    acc = accuracy(np.asarray(labs2, dtype=I8),
                   predict_linear_readout(Xt, W, b))
    return W, b, {
        "n_train": int(X.shape[0]),
        "n_test": int(Xt.shape[0]),
        "feat_dim": int(X.shape[1]),
        "n_class": int(np.unique(y).size),
        "readout_best_val_acc": float(hist["best_val_acc"]),
        "readout_best_iter": int(hist["best_iter"]),
        "derived_acc": float(acc),
    }


def _g1_seed_report_lines(rep: Any, seed: int, wall_s: float) -> List[str]:
    """把单次 run_g1_protocol 的 G1Report 格式化为阶段报告行（纯格式化）。"""
    lines = [
        f"G1 seed {seed}: ok={rep.ok} rounds={rep.n_rounds} "
        f"best_acc={rep.best_acc * 100:.1f}% rolled_back={rep.rolled_back} "
        f"wall={wall_s:.1f}s",
        f"  stages: {' -> '.join(rep.stage_sequence)}",
    ]
    for idx, a in enumerate(rep.adapt_rounds):
        lines.append(
            f"  ADAPT[{idx}]: rate={a['mean_rate_hz']:.2f}Hz "
            f"silence={a['silence_frac'] * 100:.1f}% "
            f"passed={a['passed']} forced={a['forced']} "
            f"epochs={a['n_epochs_total']}"
        )
    for idx, c in enumerate(rep.calibrate_rounds):
        lines.append(
            f"  CALIBRATE[{idx}]: ok={c['ok']} iters={c['iterations']} "
            f"rate_hist={[round(x, 2) for x in c['mean_rate_hz_history']]}"
        )
    if rep.readout:
        r = rep.readout
        lines.append(
            f"  READOUT: best_val_acc={r.get('best_val_acc', 0.0) * 100:.1f}% "
            f"best_iter={r.get('best_iter')} n_train={r.get('n_train')} "
            f"feat_dim={r.get('d')} n_class={r.get('n_class')}"
        )
    for e in rep.eval_rounds:
        lines.append(
            f"  EVAL: acc={e['acc'] * 100:.1f}% "
            f"rate={e['mean_rate_hz']:.2f}Hz "
            f"silence={e['silence_frac'] * 100:.1f}% "
            f"capped={e['capped_ratio'] * 100:.1f}% "
            f"cos_tr_te={e['cos_train_test']:.3f} n_test={e['n_test']}"
        )
    for note in rep.notes:
        lines.append(f"  [note] {note}")
    return lines


def run_g1_training(args: argparse.Namespace) -> Dict[str, Any]:
    """G1 协议训练编排：逐 seed 建网 → run_g1_protocol → 重导出 → 落盘。

    种子/数据/读出约定复用 exp.ablation（build seed = 数据种子基准；训练/
    测试数据派生 1000+/2000+seed；ReadoutConfig(seed)），与 G1 门禁
    （exp.gates --g1 / exp.ablation）同一对照口径可比。每 seed 完成一轮
    协议即落盘一个 checkpoint（meta 指纹含 seed/协议参数/config 指纹/
    阶段 stage_sequence/是否回滚 rolled_back）。

    Raises:
        ValueError/AssertionError: 参数非法或契约被违反（含统计数值）。
    """
    cfg_yaml = load_config(args.config)
    netcfg = to_core_cfg(cfg_yaml)              # 超参唯一契约的权威翻译
    fp_cfg = config_fingerprint(cfg_yaml)

    n_classes = int(args.classes or _G1_DEF_CLASSES)
    if not 2 <= n_classes <= N_CLASSES:
        raise ValueError(f"--classes ∈ [2, {N_CLASSES}]，got {n_classes}")
    train_total = int(args.train_samples or _G1_DEF_TRAIN_SAMPLES)
    test_total = int(args.test_samples or _G1_DEF_TEST_SAMPLES)
    if train_total < n_classes or test_total < n_classes:
        raise ValueError(
            f"--train-samples/--test-samples 需保证每类 >= 1 个样本"
            f"（>= {n_classes}），got {train_total}/{test_total}"
        )
    noise = float(args.noise)
    graded = bool(args.graded)
    pool = int(args.pool or _G1_DEF_POOL)
    if pool < 1:
        raise ValueError(f"--pool 必须 >= 1，got {pool}")
    if args.seeds:
        seeds = tuple(int(s) for s in args.seeds)
    elif int(args.seed) != _DEF_SEED:
        seeds = (int(args.seed),)
    else:
        seeds = _G1_DEF_SEEDS

    train_tpc = int((train_total + n_classes - 1) // n_classes)
    test_tpc = int((test_total + n_classes - 1) // n_classes)
    pcfg_kw = _protocol_kwargs_from_yaml(cfg_yaml)

    t0 = time.perf_counter()
    checkpoints: List[str] = []
    rows: List[Dict[str, Any]] = []
    accs: List[float] = []
    for s in seeds:
        net = replace(netcfg, n_pool=pool, seed=int(s)).with_derived()
        bundle = build_network(net)
        print(f"=== hstdn.train --g1 ===  seed {s}  (pool={pool}, "
              f"classes={n_classes}, train={n_classes * train_tpc}, "
              f"test={n_classes * test_tpc}, noise={noise}) ===")
        train_fn, test_fn = _g1_data_fns(n_classes, train_tpc, test_tpc,
                                         noise, graded, int(s))
        from hstdn.core.scheduler import ProtocolConfig, run_g1_protocol
        pc = ProtocolConfig(seed=int(s), **pcfg_kw)
        t_s = time.perf_counter()
        rep = run_g1_protocol(bundle, train_fn, test_fn, pc, seed=int(s))
        W, b, dacc = _derive_readout(bundle, train_fn, test_fn, seed=int(s),
                                     feat_mode=pc.feat_mode,
                                     feat_n_bins=pc.feat_n_bins)
        wall = time.perf_counter() - t_s
        print("\n".join(_g1_seed_report_lines(rep, int(s), wall)))
        acc_eq = abs(float(dacc["derived_acc"]) - float(rep.best_acc)) <= 1e-9
        print(f"  derived readout: acc={dacc['derived_acc'] * 100:.1f}% "
              f"(protocol best {rep.best_acc * 100:.1f}%, "
              f"consistent={acc_eq})")

        # ---- meta 指纹：seed / 协议参数 / config 指纹 / 阶段 / 是否回滚 ----
        meta_run_args = {
            "mode": "g1",
            "seed": int(s),
            "pool": pool,
            "classes": n_classes,
            "train_total": int(n_classes * train_tpc),
            "test_total": int(n_classes * test_tpc),
            "noise": noise,
            "graded": graded,
        }
        fp_run = _run_fingerprint(
            {"run_args": meta_run_args, "protocol": pcfg_kw,
             "fingerprint": fp_cfg}
        )
        meta = {
            "format": CKPT_FORMAT,
            "mode": "g1",
            "fingerprint": fp_cfg,
            "run_fingerprint": fp_run,
            "config_json": _json_safe(cfg_yaml),
            "run_args": meta_run_args,
            "protocol": _json_safe(pcfg_kw),
            "protocol_yaml": _json_safe(cfg_yaml.get("protocol", {})),
            "stage_sequence": list(rep.stage_sequence),
            "rolled_back": bool(rep.rolled_back),
            "n_rounds": int(rep.n_rounds),
            "best_acc": float(rep.best_acc),
            "notes": list(rep.notes),
            "adapt_rounds": _json_safe(rep.adapt_rounds),
            "calibrate_rounds": _json_safe(rep.calibrate_rounds),
            "eval_rounds": _json_safe(rep.eval_rounds),
            "collect_n": int(rep.collect_n),
            "feat_dim": int(rep.feat_dim),
            "n_class": int(rep.n_class),
            "readout": _json_safe(rep.readout),
            "eta_scale_final": float(rep.eta_scale_final),
            "derived": _json_safe(dacc),
            "n_in": int(net.n_in),
            "n_pool": int(net.n_pool),
            "n_edges": int(bundle.n_edges),
            "elapsed_s": wall,
        }
        rows.append({
            "seed": int(s),
            "acc": float(rep.best_acc),
            "derived_acc": float(dacc["derived_acc"]),
            "wall_s": wall,
            "rolled_back": bool(rep.rolled_back),
            "n_rounds": int(rep.n_rounds),
        })
        accs.append(float(rep.best_acc))
        if not args.no_save:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"g1_train_s{int(s)}_{fp_run[:16]}.npz"
            path = save_checkpoint(bundle, meta, out_dir / fname,
                                   extra={"W": W, "b": b})
            checkpoints.append(str(path))
            print(f"g1 checkpoint saved: {path}")

    summary = diag.multi_seed_summary(accs)
    print(
        f"G1 protocol train summary: seeds={list(seeds)} "
        f"acc={summary['mean_pct']:.1f}% ± {summary['std_pct']:.1f}% "
        f"(n={int(summary['n'])}, min {summary['min'] * 100:.1f}% / "
        f"max {summary['max'] * 100:.1f}%)"
    )
    print(f"total wall: {time.perf_counter() - t0:.1f} s")
    return {"seeds": rows, "acc_summary": summary,
            "checkpoints": checkpoints}


def build_parser() -> argparse.ArgumentParser:
    """训练入口命令行参数（超参一律来自 yaml，此处只收运行参数）。"""
    p = argparse.ArgumentParser(
        prog="hstdn.train",
        description="H-STDN 训练入口骨架（config→build_network→run_sample），"
                    "D1-D4 首期用途：链路打通 / G0 调试 / 单样本冒烟。",
    )
    p.add_argument("--config", default=None, metavar="PATH",
                   help="yaml 配置路径（缺省：configs/default.yaml）")
    p.add_argument("--epochs", type=int, default=_DEF_EPOCHS,
                   help=f"训练轮数（默认 {_DEF_EPOCHS}）")
    p.add_argument("--samples", type=int, default=_DEF_SAMPLES,
                   help=f"每 epoch 样本总数（默认 {_DEF_SAMPLES}；自动按 "
                        f"{N_CLASSES} 类分块生成并截取头部）")
    p.add_argument("--noise", type=float, default=_DEF_NOISE,
                   help=f"训练强度噪声 std（默认 {_DEF_NOISE}；0 = 无噪）")
    p.add_argument("--no-graded", dest="graded", action="store_false",
                   help="掩码单元强度不渐变（flat，全 GLYPH_HI）")
    p.set_defaults(graded=True)
    p.add_argument("--seed", type=int, default=_DEF_SEED,
                   help=f"随机种子（build + 数据派生，默认 {_DEF_SEED}）")
    p.add_argument("--stdp", dest="stdp_on", action=argparse.BooleanOptionalAction,
                   default=True, help="STDP 开关（默认开）")
    p.add_argument("--homeo", dest="homeo_on",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="homeostasis 开关（默认开）")
    p.add_argument("--norm", dest="norm_on", action=argparse.BooleanOptionalAction,
                   default=True, help="competitive norm 开关（默认开）")
    p.add_argument("--out-dir", default=_DEF_OUT_DIR, metavar="DIR",
                   help=f"checkpoint 输出目录（默认 {_DEF_OUT_DIR}，被 gitignore）")
    p.add_argument("--tag", default=None, metavar="TAG",
                   help="可选文件名后缀（train_<tag>_<run_fp>.npz）")
    p.add_argument("--no-save", action="store_true",
                   help="不落盘 checkpoint（纯链路冒烟）")
    p.add_argument("--quiet", action="store_true", help="仅打印 epoch 汇总")
    p.add_argument("--selfcheck", action="store_true",
                   help="运行指纹 + checkpoint 编解码自检后退出")
    g = p.add_argument_group("G1 protocol mode (--g1)")
    g.add_argument("--g1", action="store_true",
                   help="运行 M6 完整协议：core.scheduler.run_g1_protocol "
                        "（ADAPT→CALIBRATE→COLLECT→READOUT→EVAL）")
    g.add_argument("--classes", type=int, default=None, metavar="K",
                   help=f"G1 类别数（默认 {_G1_DEF_CLASSES}）")
    g.add_argument("--train-samples", type=int, default=None, metavar="N",
                   help="G1 每 seed 训练样本总数（默认 "
                        f"{_G1_DEF_TRAIN_SAMPLES} = 10 类 × 12/类）")
    g.add_argument("--test-samples", type=int, default=None, metavar="N",
                   help="G1 每 seed 协议 EVAL 测试样本总数（默认 "
                        f"{_G1_DEF_TEST_SAMPLES}）")
    g.add_argument("--seeds", type=_parse_seeds, default=None, metavar="S",
                   help="G1 种子列表（如 0,1,2 或 1）；缺省回落 --seed，"
                        "再缺省则用默认双种子 0,1")
    g.add_argument("--pool", type=int, default=None, metavar="N",
                   help=f"G1 池规模（默认 {_G1_DEF_POOL} = G1 门禁 light "
                        "规模；§4 800 池请 --pool 800，注意 L0 耗时）")
    return p
def _selfcheck() -> int:
    """指纹确定性与 checkpoint 编解码往返自检（G0 类断言，独立可执行）。

    Returns:
        0 = 通过；失败抛 AssertionError（带统计的可读消息）。
    """
    import tempfile

    print("hstdn.train selfcheck:")
    fp1 = config_fingerprint(load_config())
    fp2 = config_fingerprint(load_config())
    assert fp1 == fp2, f"config 指纹不稳定：{fp1} != {fp2}"
    print(f"  [PASS] config fingerprint deterministic: {fp1[:16]}...")

    micro = NetConfig(n_in=9, n_pool=16, n_input_cols=3, seed=0).with_derived()
    b0 = build_network(micro, rng=np.random.default_rng(0))
    meta = {
        "format": CKPT_FORMAT,
        "fingerprint": fp1,
        "run_fingerprint": "0" * 64,
        "run_args": {"seed": 0, "epochs": 1, "samples": 4},
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt_roundtrip.npz"
        save_checkpoint(b0, meta, path)
        b1, meta1 = load_checkpoint(path)
    for f in fields(NetworkBundle):
        if f.name == "cfg":
            continue
        a0 = np.asarray(getattr(b0, f.name))
        a1 = np.asarray(getattr(b1, f.name))
        assert a0.shape == a1.shape and np.array_equal(a0, a1), (
            f"roundtrip 字段 {f.name} 不一致: shape {a0.shape} vs {a1.shape}, "
            f"dtype {a0.dtype} vs {a1.dtype}"
        )
    c0 = {f.name: getattr(b0.cfg, f.name) for f in fields(NetConfig)}
    c1 = {f.name: getattr(b1.cfg, f.name) for f in fields(NetConfig)}
    assert c0 == c1, "roundtrip NetConfig 字段不一致"
    assert meta1["format"] == CKPT_FORMAT and meta1["fingerprint"] == fp1
    print("  [PASS] checkpoint npz roundtrip (arrays + NetConfig + meta)")
    print("selfcheck PASSED")
    return 0


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    """训练编排主体：配置 → 建网 → flat 训练 → 诊断 → checkpoint。

    Args:
        args: build_parser().parse_args 的结果。

    Returns:
        汇总 dict（epochs_summary / fingerprint / checkpoint 路径等），
        供调用方（含 __main__）打印与退出。

    Raises:
        ValueError/AssertionError: 参数非法或契约被违反（含统计数值）。
    """
    if int(args.epochs) < 1:
        raise ValueError(f"--epochs 必须 >= 1，got {args.epochs}")
    if int(args.samples) < 1:
        raise ValueError(f"--samples 必须 >= 1，got {args.samples}")
    if float(args.noise) < 0.0:
        raise ValueError(f"--noise 必须 >= 0，got {args.noise}")

    t0 = time.perf_counter()
    cfg_yaml = load_config(args.config)
    netcfg = to_core_cfg(cfg_yaml)              # 超参唯一契约的权威翻译
    fp_cfg = config_fingerprint(cfg_yaml)

    # 数据规模：ceil(samples / N_CLASSES) 保证类覆盖，头部截取前 samples 个
    n_per_class = int((int(args.samples) + N_CLASSES - 1) // N_CLASSES)

    run_args = {
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "samples": int(args.samples),
        "n_per_class": n_per_class,
        "noise": float(args.noise),
        "graded": bool(args.graded),
        "stdp_on": bool(args.stdp_on),
        "homeo_on": bool(args.homeo_on),
        "norm_on": bool(args.norm_on),
    }
    fp_run = _run_fingerprint(run_args)

    print(f"=== hstdn.train ===  (config fingerprint {fp_cfg[:16]}... , "
          f"run fingerprint {fp_run[:16]}...)")
    bundle = build_network(netcfg, rng=np.random.default_rng(int(args.seed)))
    print_report(bundle)

    epochs_summary = _training_epochs(
        bundle, netcfg,
        epochs=int(args.epochs), samples=int(args.samples),
        n_per_class=n_per_class, noise=float(args.noise),
        graded=bool(args.graded), base_seed=int(args.seed),
        stdp_on=bool(args.stdp_on), homeo_on=bool(args.homeo_on),
        norm_on=bool(args.norm_on), quiet=bool(args.quiet),
    )

    # ---- 最终汇总（全部 epoch）----
    total_spikes = float(sum(e["total_spikes"] for e in epochs_summary))
    n_epochs = len(epochs_summary)
    # 均值发放率分母 = 全部 epoch 的样本总数（逐样本率再平均，与聚合面板一致）
    n_eval_samples = max(float(sum(e["n_samples"] for e in epochs_summary)), 1.0)
    final_panel = {
        "epochs": float(n_epochs),
        "total_spikes": total_spikes,
        "mean_rate_hz": total_spikes / (
            n_eval_samples * float(netcfg.n_pool) * float(netcfg.window_s)),
        "silence_ratio": float(np.mean(
            [e["silence_ratio"] for e in epochs_summary])),
        "theta_range": f"[{float(bundle.theta.min()):.3f}, "
                       f"{float(bundle.theta.max()):.3f}]",
        "rate_ema_mean_hz": float(bundle.rate_ema.mean()),
    }
    diag.print_panel(final_panel, title="train final (pool state)")

    meta = {
        "format": CKPT_FORMAT,
        "fingerprint": fp_cfg,
        "run_fingerprint": fp_run,
        "config_json": _json_safe(cfg_yaml),
        "run_args": run_args,
        "n_in": int(netcfg.n_in),
        "n_pool": int(netcfg.n_pool),
        "n_edges": int(bundle.n_edges),
        "elapsed_s": float(time.perf_counter() - t0),
    }

    summary: Dict[str, Any] = {
        "epochs_summary": epochs_summary,
        "fingerprint": fp_cfg,
        "run_fingerprint": fp_run,
        "checkpoint": None,
    }
    if not args.no_save:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"train_{fp_run[:16]}.npz"
        if args.tag:
            fname = f"train_{args.tag}_{fp_run[:16]}.npz"
        path = save_checkpoint(bundle, meta, out_dir / fname)
        summary["checkpoint"] = str(path)
        print(f"checkpoint saved: {path}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口（退出码：0 成功 / 1 运行期失败 / 2 参数错误）。"""
    args = build_parser().parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    try:
        if args.g1:
            run_g1_training(args)
        else:
            run_training(args)
    except (ValueError, AssertionError, FileNotFoundError,
            NotImplementedError, RuntimeError) as exc:
        print(f"[hstdn.train] FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())