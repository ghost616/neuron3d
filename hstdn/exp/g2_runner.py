"""g2_runner.py -- G2 双臂 runner（exp 模块；v3.3 冻结清单 #7；R1 第一判合法性前提）。

设计（文档 §6 G2 执行规格 Step 1；双臂除开关组外逐字段一致）：
    LSM 臂  对照组：随机固定池、无学习（run_g1_protocol 以
            ``adapt_enabled=False`` 走 CALIBRATE+COLLECT+READOUT+EVAL，
            main 默认 LSM 语义）；
    STDP 臂 实验组：``hstdn/configs/stdp_min.yaml`` 档（pool_learn=False
            冻结池内 E→E 可塑性，Diehl-Cook 前馈）+ ADAPT 三开关开。
两臂共享同一确定性网络 cfg（同 seed）、同一数据切分（同 seed、逐字段一致，
唯一差异 = ADAPT 三开关组）；共享同一 run_g1_protocol 协议入口与
COLLECT/READOUT/EVAL 路径（不重复实现，遵循 core scheduler 约定）。

执行规格：
- 强制双臂同 seed 配对，默认 5 seeds（G2 Step 1：G1 n=3 在 ±4.2pp 方差下
  无功效 → 升到 5）；mini 模式 2 seeds 供冒烟。
- 逐 seed 配对差 d_s = acc_stdp_s - acc_lsm_s：输出 mean±std。
- **预注册判据**（一经注册不事后移动）：
    R1_LSM_GATE_MIN = 0.70    LSM 臂均值 acc ≥70%（绝对门禁，不满足则该
                              seed 集无法公平检验 R1 第一判，标 lsm-gate-fail）；
    R1 第一判（paired superiority）：mean(d) > 0 且 mean(d) > 1×std(d)。
- 处理有效性（评审 §2 迁移性预注册规则）：STDP 臂 ADAPT 后自动读取
  drift_frac/gini（复用 exp.diagnostics.read_drift_stats）：
    mean drift_frac ≥5% → 处理有效（valid），判据照常执行；
    <5%           → 判 inconclusive-treatment，按预注册一次性 η 校准
      （eta_ltp/eta_ltd × G2_ETA_CALIB_FACTOR=2.0）执行**一次**并重跑；
      校准机会全局仅此一次（防再调调滑坡）；重跑后仍 <5% → 终判
      inconclusive-treatment（不再校准）。

数据（MNIST）：G2 全量要求真实 MNIST 1k 子集（data.mnist.load_mnist_subset
——**纯 NumPy IDX 下载，无 torch 依赖**：gzip + urllib 从 ossci S3 镜像自动
获取并缓存到 data/mnist，网络可达时无需手工准备）。mini/冒烟默认经 auto：
缓存命中或网络可达 → 真实 MNIST；**仅完全离线且无缓存**时才回退到合成
MNIST 布局等价输入（10x10 单元网格 = N_IN 100、10 类、类别条件强度帧），
并在输出中如实标注 data_source=synth/mnist；--data mnist 强制真实子集
（无缓存且无网络时报可读指引，**不静默回退合成伪装真实数据**）。

CLI：python -m hstdn.exp.g2_runner [--mini|--full] [--seeds 0 1 2 3 4]
      [--data auto|synth|mnist] [--json]
gates.py 经 run_g2() 注册轻量（mini auto）与全量（--full）语义。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from hstdn.core.layout import NetConfig
from hstdn.core.network import build_network
from hstdn.core.scheduler import ProtocolConfig, G1Report, run_g1_protocol
from hstdn.data.synthetic import N_CLASSES, N_GRID, synthetic_frame
from hstdn.configs import load_config, profile_path, to_core_cfg
from hstdn.exp import diagnostics as diag

__all__ = [
    "R1_LSM_GATE_MIN", "R1_DIFF_STD_FACTOR", "DRIFT_VALID_MIN",
    "G2_ETA_CALIB_FACTOR", "G2_DEFAULT_SEEDS", "G2_MINI_SEEDS",
    "make_g2_dataset", "make_g2_data_fns", "g2_net_cfg", "run_arm",
    "run_g2_comparison", "report_lines", "main",
]

# ---------------------------------------------------------------------------
# 预注册常量（R1 判据与迁移性规则；一经注册不事后移动）
# ---------------------------------------------------------------------------

#: LSM 臂绝对门禁：均值 acc >= 70%（低于则 seed 集无法公平检验 R1 第一判）
R1_LSM_GATE_MIN: float = 0.70
#: R1 第一判：配对差 mean(d) 须 > 1×std(d)（G2 执行规格 Step 1 功效判据）
R1_DIFF_STD_FACTOR: float = 1.0
#: 处理有效性漂移门：STDP 臂 ADAPT 后 mean drift_frac >= 5% 才判结论有效
DRIFT_VALID_MIN: float = 0.05
#: 一次性 η 校准因子（仅当 drift < DRIFT_VALID_MIN 时执行一次）
G2_ETA_CALIB_FACTOR: float = 2.0
#: G2 正式 5 seeds（G1 n=3 无功效 → Step 1 升到 5）
G2_DEFAULT_SEEDS: tuple = (0, 1, 2, 3, 4)
#: mini 冒烟 2 seeds
G2_MINI_SEEDS: tuple = (0, 1)


def g2_net_cfg(seed: int, *, n_pool: int = 150,
               eta_scale: float = 1.0) -> NetConfig:
    """STDP/LSM 两臂共享的确定性网络 cfg（stdp_min 档；n_in=100 网格）。

    Args:
        seed: 网络种子（双臂同 seed 强制配对）。
        n_pool: 池规模（G2 门禁规模默认 150，与 FULL 档一致）。
        eta_scale: 一次性 η 校准乘子（eta_ltp/eta_ltd ×= eta_scale）。
    """
    from dataclasses import replace
    base = to_core_cfg(load_config(profile_path("stdp_min")))
    cfg = replace(base, n_pool=n_pool, seed=seed,
                  pool_learn=False).with_derived()
    if eta_scale != 1.0:
        cfg = replace(cfg, eta_ltp=cfg.eta_ltp * eta_scale,
                      eta_ltd=cfg.eta_ltd * eta_scale).with_derived()
    return cfg


def mnist_cache_available(root: str = "data/mnist") -> bool:
    """真实 MNIST 是否已缓存（data 模块下载过的 4 个 IDX gz 文件齐全）。"""
    from hstdn.data.mnist import MNIST_IDX_FILES
    from pathlib import Path
    base = Path(root)
    files = []
    for split, (img, lbl, _n) in MNIST_IDX_FILES.items():
        files += [base / img, base / lbl]
    return all(p.is_file() and p.stat().st_size > 0 for p in files)


def make_g2_dataset(source: str, *, n_train_per_class: int,
                    n_test_per_class: int, seed: int, n_pool: int = 150,
                    root: str = "data/mnist") -> Dict[str, Any]:
    """生成/加载 G2 数据集（train/test 各 n_per_class×10 个 (10,10) 强度帧）。

    Args:
        source: 'synth'=合成 MNIST 布局等价帧（10x10 单元、类别条件强度；
            仅完全离线冒烟/回退用）；'mnist'=真实 MNIST（data.mnist **纯
            NumPy IDX 下载**：无 torch 依赖，网络可达时自动获取/缓存）；
            'auto'=缓存命中或网络可达 → 真实 MNIST；仅完全离线无缓存才回退
            synth 并如实标注（绝不静默用合成冒充真实）。
        n_train_per_class / n_test_per_class: 每类训练/测试样本数。
        seed: 数据切分种子（双臂共享同一数据集）。
        n_pool: 透传（仅用于占位一致性，当前数据与池规模无关）。
        root: MNIST IDX 落盘目录（data/mnist，默认相对项目根）。

    Returns:
        dict：train/test（frames (N,10,10) f8, labels (N,) i8），
        data_source（'mnist'/'synth'），note。
    """
    if source == "auto":
        if mnist_cache_available(root):
            # 缓存命中：离线也可直接用真实 MNIST
            return _make_mnist_split(n_train_per_class, n_test_per_class,
                                     seed, root)
        try:
            # 无缓存：尝试网络自动获取（纯 NumPy IDX，无 torch 依赖）
            return _make_mnist_split(n_train_per_class, n_test_per_class,
                                     seed, root)
        except (OSError, ValueError) as exc:
            out = _make_synth_split(n_train_per_class, n_test_per_class, seed)
            out["data_source"] = "synth"
            out["note"] = (f"真实 MNIST 不可用（无缓存/子集不足或网络获取"
                           f"失败）——mini/冒烟回退合成 MNIST 布局等价帧，"
                           f"仅完全离线场景。错误：{exc}")
            return out
    if source == "mnist":
        # 强制真实：无缓存且网络不可达时给出可读错误（不静默回退合成）
        return _make_mnist_split(n_train_per_class, n_test_per_class,
                                 seed, root)
    if source == "synth":
        out = _make_synth_split(n_train_per_class, n_test_per_class, seed)
        out["data_source"] = "synth"
        return out
    raise ValueError(f"unknown data source {source!r} (auto/mnist/synth)")


def _make_synth_split(n_tr: int, n_te: int, seed: int) -> Dict[str, Any]:
    """合成 MNIST 布局等价切分：每类 n 个加噪 glyph 帧（(10,10)，id 行主序）。

    noise=0.18（与 ablation light 同噪声体制）：在 v3.3 严格 ADAPT 门禁下
    双臂可稳定通过门禁、产生完整配对（低噪声 0.06 体制会触发 STDP 臂
    HARD STOP——如实记录为合成回退的已知特征）。
    """
    rng = np.random.default_rng(9000 + seed)
    def _frames(n):
        labels, frames = [], []
        for k in range(N_CLASSES):
            for _ in range(n):
                fr = synthetic_frame(k, noise=0.18,
                                     rng=rng)  # 类别条件强度（≥GLYPH_LO）
                frames.append(fr)
                labels.append(k)
        return np.stack(frames).astype(np.float64), \
            np.asarray(labels, dtype=np.int64)
    tr_f, tr_l = _frames(n_tr)
    te_f, te_l = _frames(n_te)
    return {"train": (tr_f, tr_l), "test": (te_f, te_l),
            "data_source": "synth", "note": "synthetic MNIST-layout (10x10) "
            "equivalent frames, noise 0.18 (G2 offline-only fallback)"}


def _make_mnist_split(n_tr: int, n_te: int, seed: int,
                      root: str) -> Dict[str, Any]:
    """真实 MNIST：data.mnist 纯 NumPy IDX 加载（无 torch 依赖）+ 10x10 池化。

    网络/数据可用性错误直接透传（data.mnist 自带可读 ValueError 与缓存
    损坏指引）；无缓存且无网络时由上层（auto/强制 mnist）给出可读提示。
    类平衡：从随机子集的**安全超集**（每类需要量 × 8）里按类取前 n 个
    （确定性：子集抽取 seed 由 load_mnist_subset 决定；类序优先），
    避免随机子集按类欠采样。
    """
    from hstdn.data.mnist import load_mnist_subset
    oversample = 8
    tr = load_mnist_subset(N_CLASSES * n_tr * oversample, train=True,
                           root=root, seed=seed)
    te = load_mnist_subset(N_CLASSES * n_te * oversample, train=False,
                           root=root, seed=1000 + seed)

    def _bucket(sub, n):
        frames, labels = [], []
        for k in range(N_CLASSES):
            idx = np.flatnonzero(sub.labels == k)[:n]
            if idx.size < n:
                raise ValueError(
                    f"MNIST class {k} has only {idx.size} samples in the "
                    f"{sub.labels.size}-sample subset; need {n}（安全超集 "
                    f"仍不足：请扩大子集或检查缓存）"
                )
            frames.append(sub.images[idx])
            labels.append(np.full(n, k, dtype=np.int64))
        pooled = np.stack([f for fr in frames for f in fr])  # (N,28,28)
        from hstdn.data.mnist import pool_mnist_to_units
        pooled = pool_mnist_to_units(pooled.reshape(-1, 28, 28), out=10)
        return pooled, np.concatenate(labels)
    tr_p, tr_l = _bucket(tr, n_tr)
    te_p, te_l = _bucket(te, n_te)
    return {"train": (tr_p, tr_l), "test": (te_p, te_l),
            "data_source": "mnist",
            "note": "real MNIST D14 10x10 pooled subset "
                    "(data.mnist pure-NumPy IDX; network auto-fetch)"}


def make_g2_data_fns(data: Dict[str, Any]):
    """零参可调用 train/test data_fn（每次返回新迭代器 (frame, label)）。"""
    tr_f, tr_l = data["train"]
    te_f, te_l = data["test"]

    def train_fn():
        for fr, lb in zip(tr_f, tr_l):
            yield fr.copy(), int(lb)

    def test_fn():
        for fr, lb in zip(te_f, te_l):
            yield fr.copy(), int(lb)
    return train_fn, test_fn


def _protocol(seed: int, *, switches_on: bool, t_ms: int = 200) -> ProtocolConfig:
    """双臂共享协议参数（唯一差异 adapt_enabled = switches_on）。"""
    return ProtocolConfig(
        t_ms=t_ms,
        n_adapt_epochs=1,
        adapt_enabled=switches_on,
        adapt_diag_samples=3,
        adapt_gate_extra_max=1,
        calibrate_samples=5,
        calibrate_max_iter=2,
        extra_loops=0,
        seed=seed,
    )

# ---------------------------------------------------------------------------
# 单臂运行（共享 run_g1_protocol 与 exp 诊断读数）
# ---------------------------------------------------------------------------


def run_arm(data: Dict[str, Any], seed: int, *, arm: str,
            switches_on: bool, eta_scale: float = 1.0) -> Dict[str, Any]:
    """跑一个 seed 的一个臂（LSM/STDP）；返回行字典（含归因读数）。

    Args:
        data: make_g2_dataset 输出（两臂共享同一数据集）。
        seed: 种子（网络 cfg 与数据同 seed 配对）。
        arm: 'lsm'/'stdp' 标签。
        switches_on: True=STDP 臂（ADAPT 三开关开）；False=LSM 臂。
        eta_scale: 一次性 η 校准乘子（仅校准重跑时 != 1）。

    Returns:
        dict：seed/arm/acc/eval_rate/silence/calib_ok/drift_frac/gini/
        gini_ratio/budget_dev/r7_alert/aborted/notes/wall_s。
    """
    cfg = g2_net_cfg(seed, eta_scale=eta_scale)
    bundle = build_network(cfg)
    snap = diag.capture_drift_snapshot(bundle)
    train_fn, test_fn = make_g2_data_fns(data)
    pc = _protocol(seed, switches_on=switches_on)
    import time
    t0 = time.perf_counter()
    rep: G1Report = run_g1_protocol(bundle, train_fn, test_fn, pc, seed=seed)
    wall = time.perf_counter() - t0
    ev = rep.eval_rounds[-1] if rep.eval_rounds else {}
    drift = diag.read_drift_stats(bundle, snap)
    r7 = diag.assess_r7(bundle, snap)
    return {
        "seed": int(seed),
        "arm": arm,
        "acc": float(rep.best_acc),
        "eval_rate_hz": float(ev.get("mean_rate_hz", 0.0)),
        "silence_frac": float(ev.get("silence_frac", 1.0)),
        "calib_ok": bool(rep.calibrate_rounds
                         and rep.calibrate_rounds[-1].get("ok", False)),
        "drift_frac": float(drift["drift_frac"]),
        "gini": float(drift["gini"]),
        "gini_ratio": float(drift["gini_ratio"]),
        "budget_dev": float(drift["budget_dev"]),
        "r7_alert": bool(r7["alert"]),
        "aborted": bool(not rep.ok or not rep.eval_rounds),
        "n_rounds": int(rep.n_rounds),
        "wall_s": float(wall),
        "notes": list(rep.notes),
    }


# ---------------------------------------------------------------------------
# 双臂对比 + 预注册判据判定（含一次性 η 校准 / 迁移性规则）
# ---------------------------------------------------------------------------


def run_g2_comparison(full: bool = False, *, seeds: Optional[Sequence[int]] = None,
                      data_source: str = "auto",
                      n_train_per_class: int = 12,
                      n_test_per_class: int = 10,
                      n_pool: int = 150) -> Dict[str, Any]:
    """运行 G2 双臂配对对比并输出预注册判据字段。

    Args:
        full: True=正式（5 seeds、真实 MNIST——data.mnist 纯 NumPy IDX，缓存
            命中或网络可达即自动获取；G2 全量数据要求见模块 docstring）；
            False=mini 冒烟（2 seeds，小样本，仅完全离线无缓存时回退 synth）。
        seeds: 种子列表（None → full: G2_DEFAULT_SEEDS / mini: G2_MINI_SEEDS）。
        data_source: auto/mnist/synth（auto：缓存或网络 → 真实；仅完全离线
            无缓存才回退 synth 并标注——绝不静默冒充真实数据）。
        n_train_per_class/n_test_per_class/n_pool: 数据与网络规模。

    Returns:
        dict：data、rows（lsm/stdp 逐 seed 行）、paired_diffs、汇总、
        criteria（预注册判据字段与判定）、treatment 标签、calib_used 等。
    """
    seeds = tuple(seeds) if seeds is not None else (
        G2_DEFAULT_SEEDS if full else G2_MINI_SEEDS)
    n_tr = n_train_per_class if not full else 24   # full 每类训练样本
    n_te = n_test_per_class if not full else 20
    if full and data_source == "auto":
        data_source = "mnist"      # 正式 G2 强制真实 MNIST（无缓存无网络报错）
    data = make_g2_dataset(data_source, n_train_per_class=n_tr,
                           n_test_per_class=n_te, seed=int(seeds[0]),
                           n_pool=n_pool)
    lsm_rows: List[Dict[str, Any]] = []
    stdp_rows: List[Dict[str, Any]] = []
    calib_used = False
    for sd in seeds:
        lsm_rows.append(run_arm(data, int(sd), arm="lsm", switches_on=False))
        stdp_rows.append(run_arm(data, int(sd), arm="stdp", switches_on=True))
    stdp_mean_drift = float(np.mean([r["drift_frac"] for r in stdp_rows]))
    valid = bool(stdp_mean_drift >= DRIFT_VALID_MIN)
    treatment = "valid" if valid else "inconclusive-treatment"
    if not valid:
        # 预注册一次性 η 校准（全局仅此一次），随后重跑 STDP 臂
        calib_used = True
        stdp_rows = []
        for sd in seeds:
            stdp_rows.append(run_arm(data, int(sd), arm="stdp",
                                     switches_on=True,
                                     eta_scale=G2_ETA_CALIB_FACTOR))
        stdp_mean_drift2 = float(np.mean([r["drift_frac"] for r in stdp_rows]))
        valid2 = bool(stdp_mean_drift2 >= DRIFT_VALID_MIN)
        treatment = ("valid" if valid2 else "inconclusive-treatment")
        calib_note = ("eta x%.1f calibration applied once; drift %.1f%% -> "
                      "%.1f%%" % (G2_ETA_CALIB_FACTOR,
                                  stdp_mean_drift * 100.0,
                                  stdp_mean_drift2 * 100.0))
    else:
        calib_note = "no calibration needed (drift >= 5%)"
    # --- 配对差（同 seed 强制配对；仅统计双臂均未 ABORT 的配对）---
    diffs = [stdp_rows[i]["acc"] - lsm_rows[i]["acc"]
             for i in range(len(seeds))
             if not stdp_rows[i]["aborted"] and not lsm_rows[i]["aborted"]]
    n_paired = len(diffs)
    mean_d = float(np.mean(diffs)) if diffs else 0.0
    std_d = (float(np.std(diffs, ddof=1)) if len(diffs) >= 2
             else (float(np.std(diffs)) if len(diffs) == 1 else 0.0))
    lsm_ok = [r["acc"] for r in lsm_rows if not r["aborted"]]
    stdp_ok = [r["acc"] for r in stdp_rows if not r["aborted"]]
    lsm_mean = float(np.mean(lsm_ok)) if lsm_ok else 0.0
    stdp_mean = float(np.mean(stdp_ok)) if stdp_ok else 0.0
    # LSM 绝对门禁只看 LSM 臂本身（不掺入配对完成度）
    lsm_gate = bool(lsm_mean >= R1_LSM_GATE_MIN)
    paired_complete = bool(n_paired == len(seeds) and len(seeds) > 0)
    mean_gt0 = bool(mean_d > 0.0)
    mean_gt_std = bool(std_d > 0.0 and mean_d > R1_DIFF_STD_FACTOR * std_d)
    criterion_pass = bool(valid and paired_complete and lsm_gate
                          and mean_gt0 and mean_gt_std)
    criteria = {
        "lsm_gate_min": R1_LSM_GATE_MIN,
        "lsm_mean": lsm_mean,
        "lsm_gate_pass": lsm_gate,
        "stdp_mean": stdp_mean,
        "paired_diff_mean": mean_d,
        "paired_diff_std": std_d,
        "n_paired": n_paired,
        "n_seeds": len(seeds),
        "paired_complete": paired_complete,
        "mean_gt_zero": mean_gt0,
        "mean_gt_std": mean_gt_std,
        "criterion_pass": criterion_pass,
    }
    return {
        "full": bool(full),
        "seeds": list(seeds),
        "n_train_per_class": n_tr,
        "n_test_per_class": n_te,
        "n_pool": n_pool,
        "data_source": data.get("data_source"),
        "data_note": data.get("note", ""),
        "lsm_rows": lsm_rows,
        "stdp_rows": stdp_rows,
        "paired_diffs": diffs,
        "summaries": {"lsm_mean": lsm_mean, "stdp_mean": stdp_mean,
                      "diff_mean": mean_d, "diff_std": std_d},
        "criteria": criteria,
        "treatment": treatment,
        "drift_pre_calib": stdp_mean_drift,
        "calib_used": calib_used,
        "calib_note": calib_note,
    }


def report_lines(out: Dict[str, Any]) -> List[str]:
    """把 G2 运行结果格式化为可打印行。"""
    lines = [
        "=" * 78,
        "G2 runner (%s mode, seeds=%s, data=%s, n_tr/class=%d, n_te/class=%d, "
        "pool=%d)" % ("full" if out["full"] else "mini", out["seeds"],
                      out["data_source"], out["n_train_per_class"],
                      out["n_test_per_class"], out["n_pool"]),
        "  LSM arm | STDP arm | per-seed paired diff:",
    ]
    for l, s in zip(out["lsm_rows"], out["stdp_rows"]):
        lines.append(
            f"    seed {l['seed']}: lsm={l['acc'] * 100:5.1f}% "
            f"(sil {l['silence_frac'] * 100:4.1f}%"
            + (", ABORT" if l["aborted"] else "")
            + ") | stdp="
            + (f"{s['acc'] * 100:5.1f}%" if not s["aborted"] else " ABORT ")
            + f" (sil {s['silence_frac'] * 100:4.1f}%"
            + (", ABORT" if s["aborted"] else "")
            + f") | d={s['acc'] - l['acc']:+.3f}"
        )
    c = out["criteria"]
    n_abort = sum(1 for r in out["stdp_rows"] if r["aborted"]) + \
        sum(1 for r in out["lsm_rows"] if r["aborted"])
    lines.append(
        f"  paired diff: mean={c['paired_diff_mean'] * 100:+.2f}pp ± "
        f"{c['paired_diff_std'] * 100:.2f}pp "
        f"(paired {c['n_paired']}/{c['n_seeds']}; ABORTED seeds={n_abort})"
    )
    lines.append(
        f"  criteria (pre-registered): LSM gate acc>={c['lsm_gate_min'] * 100:.0f}%"
        f" -> {c['lsm_mean'] * 100:.1f}% ({'PASS' if c['lsm_gate_pass'] else 'FAIL'}); "
        f"paired_complete={c['paired_complete']}; mean(diff)>0: "
        f"{c['mean_gt_zero']}; mean(diff)>1xstd(diff): {c['mean_gt_std']}"
    )
    lines.append(
        f"  treatment: {out['treatment']} (drift_pre_calib="
        f"{out['drift_pre_calib'] * 100:.2f}%, threshold 5%; "
        f"calib_used={out['calib_used']}; {out['calib_note']})"
    )
    lines.append(
        f"  R1 first criterion: {'PASS' if c['criterion_pass'] else 'NOT MET'} "
        f"(requires valid treatment + LSM gate + mean>0 & mean>1xstd)"
    )
    if out.get("data_note"):
        lines.append(f"  data: {out['data_note']}")
    return lines

def _parse_seeds(text: str) -> List[int]:
    parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("empty seed list")
    return [int(p) for p in parts]


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    ap = argparse.ArgumentParser(
        prog="python -m hstdn.exp.g2_runner",
        description="G2 双臂 runner（LSM 对照 vs STDP；R1 第一判预注册）",
    )
    ap.add_argument("--mini", action="store_true", default=True,
                    help="mini 冒烟：2 seeds、小样本（默认）")
    ap.add_argument("--full", action="store_true",
                    help="正式 G2：5 seeds、真实 MNIST 子集（data.mnist 纯 "
                         "NumPy IDX，缓存命中或网络可达即自动获取）")
    ap.add_argument("--seeds", type=_parse_seeds, default=None,
                    help="种子列表，如 0,1,2,3,4（覆盖默认）")
    ap.add_argument("--data", choices=("auto", "mnist", "synth"),
                    default="auto",
                    help="auto=缓存/网络优先、仅完全离线回退 synth；"
                         "mnist=强制真实（无缓存无网络报错）；synth=合成")
    ap.add_argument("--json", action="store_true", help="附加 JSON 汇总")
    args = ap.parse_args(list(argv) if argv is not None else None)
    if args.mini and args.full:
        args.mini = False
    try:
        out = run_g2_comparison(full=args.full, seeds=args.seeds,
                                data_source=args.data)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] G2 run failed: {exc}")
        return 2
    print("\n".join(report_lines(out)))
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
