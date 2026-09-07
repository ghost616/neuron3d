"""eval.py -- 评估入口骨架（main 模块维护；D1-D4 首期用途）。

用途
----
加载 train.py 产出的 checkpoint（npz，hstdn-network-bundle-v1）+ 配置，在
独立生成的合成测试集上以**全 off 冻结**模式（stdp_on/homeo_on/norm_on 均
为 False）逐样本 ``run_sample``，输出率/沉默/cos 汇总：

    1. ``train.load_checkpoint`` 重建 NetworkBundle（cfg 落盘复验）；
    2. 若显式给出 --config，将其指纹与 checkpoint 内配置指纹比对
       （不一致即失败 —— 评估必须与被评估训练配置对齐）；
    3. 测试集：data.synthetic 独立派生 seed（test_seed = 1000003 + seed，
       与训练数据流不重叠）生成，逐样本 latency 编码后冻结 run_sample；
    4. 汇总：跨样本率/沉默面板（复用 train.sample_aggregate_panel）、
       类均值发放特征两两余弦（复用 exp.diagnostics.class_cosine_matrix）。

算法纪律：本模块**只做编排不做算法实现**；冻结模拟与全部诊断量都来自
hstdn.core / hstdn.exp 接口。cos 仅为「类均值特征可分离性」骨架读数，
G1 的读出/特征可比性统计由 exp/data 模块按里程碑补充。

v3.3 冻结 #6（默认评估 = LSM 档冻结只读；STDP 实验臂 ckpt 可读）
----------------------------------------------------------------
默认路径（D4 模式与 --g1 协议评估）均以**冻结只读**评估任意 train 产物：
stdp/homeo/norm 恒 False，不修改权重/阈值 —— 与 LSM 生产线默认口径一致。
checkpoint meta 带 ``profile``（train 落盘：lsm / stdp-min）时如实打印；
profile=stdp-min（stdp_min.yaml 实验臂档）的 checkpoint 可由同一只读路径
评估（读出权重 W/b 在 g1 checkpoint 内）。--config 指纹比对失败时提示按
profile 选用匹配档文件。

G1 协议评估模式（--g1）
----------------------
加载 train.py --g1 产出的协议 checkpoint（meta.mode="g1"，含读出权重 W/b），
在**独立测试集**（固定派生 test_seed = 5000 + 训练 seed + --seed 附加）上
全 off 冻结推理：core.features.build_features → core.readout 预测 → 输出
测试精度 + 率/沉默/cos；报告格式与 G1 门禁/ablation（exp.gates --g1 /
exp.ablation）对齐。协议训练阶段的 EVAL 精度（2000+seed 数据流）记录于
checkpoint meta（best_acc / eval_rounds）；独立评估使用不同数据流，精度
可能与之不同（两者都打印）。

命令行（项目根 E:\\neuron3d）
    python -m hstdn.eval --checkpoint checkpoints/train_<fp>.npz --samples 10
    python -m hstdn.eval --g1 --checkpoint checkpoints/g1_train_s0_<fp>.npz --samples 20
    python -m hstdn.eval --help
退出码：0 = 成功；1 = 运行期失败（异常上抛并打印）；2 = 参数错误。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from hstdn.configs import load_config
from hstdn.core.encoder import latency_encode
from hstdn.core.kernel import run_sample
from hstdn.core.layout import F8, I8
from hstdn.data.synthetic import N_CLASSES, synthetic_batch
from hstdn.exp import diagnostics as diag
from hstdn.train import (
    config_fingerprint,
    load_checkpoint,
    load_g1_checkpoint,
    sample_aggregate_panel,
)

__all__ = ["build_parser", "class_mean_features", "run_evaluation",
           "run_g1_evaluation", "main"]

#: 测试集派生 seed 偏移（素数；与训练数据流错开，保证评估集独立）
_TEST_SEED_OFFSET = 1000003

#: 评估运行期默认值（超参唯一来源是 default.yaml；此处仅测试运行参数）
_DEF_SAMPLES = 10
_DEF_NOISE = 0.02
_DEF_SEED = 0


def build_parser() -> argparse.ArgumentParser:
    """评估入口命令行参数。"""
    p = argparse.ArgumentParser(
        prog="hstdn.eval",
        description="H-STDN 评估入口（v3.3 冻结 #6：默认评估 = LSM 档冻结只读；"
                    "支持 lsm 与 stdp_min 档 checkpoint）：checkpoint + 配置 -> "
                    "冻结测试集 run_sample -> 率/沉默/cos（--g1 另含测试精度）。",
    )
    p.add_argument("--checkpoint", required=True, metavar="PATH",
                   help="train.py 产出的 npz checkpoint 路径（必填；LSM 与 "
                        "stdp-min 档产物均可——评估恒为冻结只读）")
    p.add_argument("--config", default=None, metavar="PATH",
                   help="yaml 配置路径（可选；给出时与 checkpoint 配置指纹比对）")
    p.add_argument("--samples", type=int, default=_DEF_SAMPLES,
                   help=f"测试样本总数（默认 {_DEF_SAMPLES}；按类分块并截取头部）")
    p.add_argument("--noise", type=float, default=_DEF_NOISE,
                   help=f"测试强度噪声 std（默认 {_DEF_NOISE}；0 = 无噪）")
    p.add_argument("--no-graded", dest="graded", action="store_false",
                   help="掩码单元强度不渐变（flat，全 GLYPH_HI）")
    p.set_defaults(graded=True)
    p.add_argument("--seed", type=int, default=_DEF_SEED,
                   help="测试派生种子基准（实际 = 1000003 + seed，默认 0）")
    p.add_argument("--quiet", action="store_true", help="仅打印汇总面板")
    g = p.add_argument_group("G1 protocol eval (--g1)")
    g.add_argument("--g1", action="store_true",
                   help="G1 模式：加载 train --g1 的协议 checkpoint（lsm 或 "
                        "stdp-min profile 均可），独立测试集冻结推理输出测试"
                        "精度（缺省为 D4 率/沉默/cos 模式）")
    return p


def _print_profile(meta: Dict[str, Any],
                   run_args: Optional[Mapping[str, Any]] = None) -> None:
    """打印 checkpoint 运行档（meta.profile；兼容旧 ckpt 从 run_args 取）。"""
    profile = meta.get("profile") or ((run_args or {}).get("profile"))
    if profile == "stdp-min":
        print("  profile: stdp-min  [STDP 实验臂 stdp_min 档 checkpoint；"
              "本评估为冻结只读]")
    elif profile:
        print(f"  profile: {profile}")
    else:
        print("  profile: n/a (pre-#6 checkpoint；评估仍为冻结只读)")


def _verify_config(meta: Dict[str, Any], config_path: Optional[str]) -> None:
    """--config 与 checkpoint 配置指纹对齐校验（评估有效性闸）。

    Args:
        meta: checkpoint 元数据（含 fingerprint/config_json）。
        config_path: 用户显式给出的 yaml 路径；None 时信任落盘配置。

    Raises:
        AssertionError: 指纹不一致（附两侧指纹前缀/路径与 profile 提示）。
    """
    if config_path is None:
        return
    fp_checkpoint = str(meta.get("fingerprint", ""))
    fp_file = config_fingerprint(load_config(config_path))
    if fp_file != fp_checkpoint:
        profile = meta.get("profile")
        hint = (
            f"checkpoint profile={profile}；如为 stdp-min 实验臂请用 "
            "hstdn/configs/stdp_min.yaml，或省略 --config 信任落盘配置"
            if profile else "或省略 --config 信任落盘配置"
        )
        raise AssertionError(
            f"配置指纹与 checkpoint 不一致：file({config_path}) "
            f"{fp_file[:16]}... != checkpoint {fp_checkpoint[:16]}...；"
            f"评估必须与被评估的训练配置对齐（{hint}）"
        )


def class_mean_features(counts_mat: Any, labels: Any,
                        n_classes: int = N_CLASSES
                        ) -> tuple[np.ndarray, np.ndarray]:
    """类均值发放特征矩阵与各类样本数。

    Args:
        counts_mat: (S, N_POOL) 每样本池发放计数。
        labels: (S,) 类别标签（0..n_classes-1）。
        n_classes: 类别数（合成 10 类）。

    Returns:
        (class_means, n_seen)：
            class_means: (K, N_POOL) f8，未见类的行全 0；
            n_seen: (K,) i8 每类样本数。
    """
    c = np.asarray(counts_mat, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)
    if c.ndim != 2 or lab.shape != (c.shape[0],):
        raise AssertionError(
            f"counts_mat ({c.shape}) 与 labels ({lab.shape}) 不一致"
        )
    k = int(n_classes)
    means = np.zeros((k, c.shape[1]), dtype=np.float64)
    n_seen = np.zeros(k, dtype=np.int64)
    for cls in range(k):
        sel = lab == cls
        n_seen[cls] = int(sel.sum())
        if sel.any():
            means[cls] = c[sel].mean(axis=0)
    return means, n_seen
def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """评估编排主体：加载 checkpoint → 冻结测试 run_sample → 汇总面板。

    Args:
        args: build_parser().parse_args 的结果。

    Returns:
        汇总 dict（面板 / 类间 cos 统计 / 元数据），供 __main__ 与调用方使用。

    Raises:
        ValueError/AssertionError: 参数非法或契约被违反（含统计数值）。
    """
    if int(args.samples) < 1:
        raise ValueError(f"--samples 必须 >= 1，got {args.samples}")
    if float(args.noise) < 0.0:
        raise ValueError(f"--noise 必须 >= 0，got {args.noise}")

    t0 = time.perf_counter()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt}")
    bundle, meta = load_checkpoint(ckpt)
    netcfg = bundle.cfg

    _verify_config(meta, args.config)
    print(f"=== hstdn.eval ===  checkpoint: {ckpt}")
    print(f"  format={meta.get('format')}  config_fp="
          f"{str(meta.get('fingerprint', ''))[:16]}...  run_fp="
          f"{str(meta.get('run_fingerprint', ''))[:16]}...")
    run_args = meta.get("run_args") or {}
    if run_args:
        print(f"  trained with: {run_args}")
    _print_profile(meta, run_args)

    # ---- 冻结评估（全 off）：权重/阈值不再演化，仅读池响应 ----
    win = float(netcfg.window_s)
    T = int(round(win * 1000.0))
    n_per_class = int((int(args.samples) + N_CLASSES - 1) // N_CLASSES)
    test_seed = _TEST_SEED_OFFSET + int(args.seed)
    batch = synthetic_batch(int(n_per_class), noise=float(args.noise),
                            graded=bool(args.graded), seed=test_seed)
    n_use = min(int(args.samples), len(batch))
    frames = batch.frames[:n_use]
    labels = batch.labels[:n_use]

    counts_mat = np.zeros((n_use, netcfg.n_pool), dtype=I8)
    if not args.quiet:
        print(f"--- eval test set (frozen; seed={test_seed}, "
              f"samples={n_use}, stdp/homeo/norm all OFF) ---")
    for i in range(n_use):
        buckets = latency_encode(np.asarray(frames[i], dtype=F8).ravel(),
                                 cfg=netcfg)
        stats = run_sample(bundle, buckets, T=T,
                           stdp_on=False, homeo_on=False, norm_on=False)
        counts_mat[i] = stats["spike_counts"]
        if not args.quiet:
            print(f"  test {i + 1:>3}/{n_use} class {int(labels[i])} "
                  f"spikes {int(stats['n_spikes_total']):>5}")

    agg = sample_aggregate_panel(counts_mat, window_s=win)
    diag.print_panel(agg, title="eval aggregate (frozen)")

    # ---- 类间余弦：类均值发放特征两两相似度（骨架读数，G1 特征里程碑跟进）----
    class_means, n_seen = class_mean_features(counts_mat, labels)
    seen = np.flatnonzero(n_seen > 0)
    cos: Dict[str, float] = {"n_classes_seen": float(seen.size)}
    if seen.size >= 2:
        m = diag.class_cosine_matrix(class_means[seen])
        off = m[np.triu_indices(seen.size, k=1)]
        cos["cos_mean_offdiag"] = float(off.mean()) if off.size else 0.0
        cos["cos_max_offdiag"] = float(off.max()) if off.size else 0.0
        cos["cos_min_offdiag"] = float(off.min()) if off.size else 0.0
    diag.print_panel(cos, title="class cos (class-mean features)")
    if not args.quiet:
        print("per-class n_seen:", n_seen.tolist())

    summary: Dict[str, Any] = {
        "checkpoint": str(ckpt),
        "fingerprint": meta.get("fingerprint"),
        "run_fingerprint": meta.get("run_fingerprint"),
        "n_test_samples": float(n_use),
        "aggregate": agg,
        "class_cos": cos,
        "elapsed_s": float(time.perf_counter() - t0),
    }
    print(f"eval done in {summary['elapsed_s']:.2f} s")
    return summary


#: 独立评估测试 seed 偏移（素数；与训练协议 EVAL 的 2000+seed 及 D4 评估的
#: 1000003+seed 均错开 —— 独立测试集固定派生，见模块 docstring）
_G1_INDEP_TEST_OFFSET = 5000
#: 训练协议 EVAL 阶段测试 seed 基准 = train._G1_TEST_SEED_OFFSET（仅供报告；
#: 权威值在 hstdn/train.py，此处不复算数据流）
_G1_TRAIN_EVAL_SEED_BASE = 2000


def run_g1_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """G1 评估编排：协议 checkpoint → 独立测试集冻结推理 → 精度/率/沉默/cos。

    数据流：test_seed = 5000 + checkpoint 训练 seed + args.seed；类别数/噪声/
    graded/样本时长取 checkpoint meta（train --g1 落盘的 run_args / protocol），
    与训练协议 EVAL（2000+seed）数据流独立。读出参数 W/b 取自 checkpoint。

    Raises:
        ValueError/AssertionError: 参数非法、非 G1 checkpoint 或契约违反。
    """
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt}")
    bundle, meta, extra = load_g1_checkpoint(ckpt)      # 非 g1 会给出指引
    _verify_config(meta, args.config)
    W = extra["W"]
    b = extra["b"]
    netcfg = bundle.cfg
    run_args = meta.get("run_args") or {}
    if run_args.get("mode") != "g1":
        raise AssertionError(f"{ckpt.name} meta.run_args.mode != 'g1'")
    seed = int(run_args.get("seed", 0))
    n_classes = int(run_args.get("classes", N_CLASSES))
    noise = float(run_args.get("noise", _DEF_NOISE))
    graded = bool(run_args.get("graded", True))

    test_total = int(args.samples or run_args.get("test_total", 100))
    if test_total < n_classes:
        raise ValueError(
            f"--samples 需保证每类 >= 1 个测试样本（>= {n_classes}），"
            f"got {test_total}"
        )
    test_seed = _G1_INDEP_TEST_OFFSET + seed + int(args.seed)
    n_per_class = int((test_total + n_classes - 1) // n_classes)
    batch = synthetic_batch(n_per_class, noise=noise, graded=graded,
                            seed=test_seed)
    n_use = min(test_total, len(batch))
    frames = batch.frames[:n_use]
    labels = batch.labels[:n_use]

    t0 = time.perf_counter()
    win = float(netcfg.window_s)
    T = int(round(win * 1000.0))
    counts_mat = np.zeros((n_use, netcfg.n_pool), dtype=I8)
    if not args.quiet:
        print(f"--- G1 eval (independent test; seed={test_seed}, "
              f"samples={n_use}, frozen all-OFF) ---")
    for i in range(n_use):
        buckets = latency_encode(np.asarray(frames[i], dtype=F8).ravel(),
                                 cfg=netcfg)
        st = run_sample(bundle, buckets, T=T,
                        stdp_on=False, homeo_on=False, norm_on=False)
        counts_mat[i] = st["spike_counts"]
        if not args.quiet:
            print(f"  test {i + 1:>3}/{n_use} class {int(labels[i])} "
                  f"spikes {int(st['n_spikes_total']):>5}")

    # ---- 读出推理（checkpoint 内 W/b；features 与训练协议同口径 v0）----
    from hstdn.core.features import FeaturesConfig, build_features
    from hstdn.core.readout import accuracy, predict_linear_readout
    X = build_features(counts_mat, FeaturesConfig())
    pred = predict_linear_readout(X, W, b)
    acc = accuracy(np.asarray(labels, dtype=I8), pred)

    # ---- 率/沉默 + 类间 cos（与 D4 模式同源面板）----
    agg = sample_aggregate_panel(counts_mat, window_s=win)
    class_means, n_seen = class_mean_features(counts_mat, labels,
                                              n_classes=n_classes)
    seen = np.flatnonzero(n_seen > 0)
    cos: Dict[str, float] = {"n_classes_seen": float(seen.size)}
    if seen.size >= 2:
        m = diag.class_cosine_matrix(class_means[seen])
        off = m[np.triu_indices(seen.size, k=1)]
        cos["cos_mean_offdiag"] = float(off.mean()) if off.size else 0.0
        cos["cos_max_offdiag"] = float(off.max()) if off.size else 0.0
        cos["cos_min_offdiag"] = float(off.min()) if off.size else 0.0

    # ---- 报告（格式与 ablation/gates 种子行口径对齐）----
    best_acc = float(meta.get("best_acc", 0.0))
    g1_profile = meta.get("profile") or run_args.get("profile")
    profile_line = (f"  profile: {g1_profile}  [stdp_min 实验臂档，评估只读]"
                    if g1_profile == "stdp-min"
                    else f"  profile: {g1_profile or 'n/a (pre-#6)'}")
    lines = [
        f"G1 eval: checkpoint={ckpt.name}",
        profile_line,
        f"  seed {seed} (independent test_seed={test_seed}): "
        f"acc={acc * 100:.1f}% n={n_use}",
        f"  rate={agg['mean_rate_hz']:.2f}Hz silence="
        f"{agg['silence_ratio'] * 100:.1f}% cos_offdiag_mean="
        f"{cos.get('cos_mean_offdiag', 0.0):.3f}",
        f"  protocol-train best_acc={best_acc * 100:.1f}% "
        f"(test_seed={_G1_TRAIN_EVAL_SEED_BASE + seed}); "
        f"stages={' -> '.join(meta.get('stage_sequence') or [])}; "
        f"rolled_back={meta.get('rolled_back', False)}",
    ]
    print("\n".join(lines))

    summary: Dict[str, Any] = {
        "checkpoint": str(ckpt),
        "profile": meta.get("profile") or run_args.get("profile"),
        "seed": seed,
        "test_seed": test_seed,
        "n_test_samples": float(n_use),
        "acc": float(acc),
        "protocol_best_acc": best_acc,
        "aggregate": agg,
        "class_cos": cos,
        "rolled_back": bool(meta.get("rolled_back", False)),
        "elapsed_s": float(time.perf_counter() - t0),
    }
    print(f"g1 eval done in {summary['elapsed_s']:.2f} s")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口（退出码：0 成功 / 1 运行期失败 / 2 参数错误）。"""
    args = build_parser().parse_args(argv)
    try:
        if args.g1:
            run_g1_evaluation(args)
        else:
            run_evaluation(args)
    except (ValueError, AssertionError, FileNotFoundError,
            NotImplementedError, RuntimeError) as exc:
        print(f"[hstdn.eval] FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())