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

M6 调度器接入点（预留；不得误读为已实现）
----------------------------------------
default.yaml ``protocol`` 节（adapt_epochs / calibrate_* / extra_loops）由
未来的 core.scheduler（M6）消费。G1 之前本骨架退化为均匀 flat 训练循环，
接入点见 :func:`_training_epochs` 内注释；scheduler 落地前不宣称协议调度
已实现（无 pass 占位纪律，先例：core.kernel.run_sample_l1）。

命令行（项目根 E:\\neuron3d）
    python -m hstdn.train --epochs 1 --samples 10
    python -m hstdn.train --help
    python -m hstdn.train --selfcheck        # 指纹 + checkpoint 编解码自检
退出码：0 = 成功；1 = 运行期失败（异常上抛并打印）；2 = 参数错误。

checkpoint npz 格式（hstdn-network-bundle-v1）
    bundle 全部数组字段 + 两条 0-d 字符串记录：
      ``__meta__``：meta JSON（format/fingerprint/run_fingerprint/run_args/
                    config_json/...）；
      ``__cfg__``：NetConfig 全部字段 JSON（含派生字段；加载时以
                   ``with_derived()`` 复验一致性）。
    fingerprint = sha256(config 规范化 JSON)；run_fingerprint 另含运行参数，
    保证「同一配置 + 同一运行参数」的文件名确定（可复现覆盖写）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
    "sample_aggregate_panel",
    "run_training",
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
                    path: str | Path) -> Path:
    """把 NetworkBundle 与元数据写入 npz checkpoint。

    Args:
        bundle: 训练结束的网络（全部数组字段落盘）。
        meta: JSON 可序列化元数据；至少含 format/fingerprint/run_fingerprint。
        path: 目标文件路径（父目录须已存在或由调用方创建）。

    Returns:
        实际写入的 Path。

    Raises:
        AssertionError: bundle 数组与 cfg 规模不一致（落盘前最后一道闸）。
    """
    out = Path(path)
    _assert_bundle_shapes(bundle)

    arrays: Dict[str, np.ndarray] = {}
    for f in fields(NetworkBundle):
        if f.name == "cfg":
            continue
        arrays[f.name] = np.asarray(getattr(bundle, f.name))  # 数组或 0-d 标量

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

        kw: Dict[str, Any] = {}
        for f in fields(NetworkBundle):
            if f.name == "cfg":
                continue
            arr = z[f.name]
            kw[f.name] = int(arr) if arr.ndim == 0 else np.asarray(arr)
        bundle = NetworkBundle(cfg=cfg, **kw)
    _assert_bundle_shapes(bundle)
    return bundle, meta
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
    """均匀 flat 训练循环（D4 骨架）。

    M6 调度器接入点（TODO/G1）：default.yaml ``protocol`` 节定义的适应轮 /
    增益标定 / 附加环协议将由 core.scheduler 在此处逐 epoch 接管；scheduler
    落地前不得宣称协议调度已实现，本循环仅为打通链路而保留。

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
        run_training(args)
    except (ValueError, AssertionError, FileNotFoundError) as exc:
        print(f"[hstdn.train] FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())