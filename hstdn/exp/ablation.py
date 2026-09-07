"""ablation.py -- G1 随机储备池对照消融（R1；exp 模块，文档 §6 G1 / §8）。

对比设计（文档 §11 冻结条件：两组唯一差异 = 三开关；v3.3 起由
``core.scheduler.ProtocolConfig.adapt_enabled`` 承载：False = LSM 臂（无
ADAPT 阶段 / #2 门禁），True = 实验臂（ADAPT 阶段三开关逐样本开）；其余
协议旋钮、数据与种子完全一致）：
    实验组（exp/expg 臂）: adapt_enabled=True —— ADAPT 三开关全开适应储备
        池，再 CALIBRATE+READOUT；网络档位由 Spec.pool_learn 决定：默认
        False（v3.3 FIX-A 冻结池内 E→E 可塑性，stdp_min 档），True 为已退役
        legacy 全可学习条件（H6 / HSTDN-EXP-2026-001，仅显式保留）；
    对照组（ctrl / LSM 臂）: adapt_enabled=False —— 随机**固定**储备池（不
        学习），仅 CALIBRATE 校准 + READOUT（经典 LSM，文档 §6 G1 回退方案；
        v3.3 下 CALIBRATE 失败同样 HARD STOP，不软跳过）。
两组都复用 core.scheduler.run_g1_protocol 作为唯一协议入口。

v3.3 冻结 #6（G1-STDP 调查结案；main 侧配套见 hstdn/train.py --g1 [--profile]）：
    - 默认运行路径全部基于**新配置（FIX-A 冻结 + LSM）**：run_ablation 的 exp
      臂 = stdp_min 档（Spec.pool_learn=False 冻结池内 E→E 可塑性）+ 三开关开；
      ctrl = LSM（adapt_enabled=False，随机固定储备池）。全可学习 legacy 条件
      （exp, pool_learn=True 且三开关全开；H6 已发表证据链 HSTDN-EXP-2026-001）
      **不再默认可运行** —— 仅经 run_expg_comparison(legacy=True) / CLI
      ``--legacy-exp-all-learn`` 显式调用，代码路径保留但默认不可达。
    - ``--expg``（判定臂：pool_learn=False + 三开关开）默认只与 ctrl(LSM)
      组成**双臂**对比；加 ``--legacy-exp-all-learn`` 才并入 legacy 全可学习臂
      （三臂表）。

运行：
    python -m hstdn.exp.ablation            # light（exp=stdp_min 档 vs ctrl LSM）
    python -m hstdn.exp.ablation --full     # 完整 G1（10 类、≥3 种子）
    python -m hstdn.exp.ablation --expg [--full]          # expg 判定（双臂）
    python -m hstdn.exp.ablation --expg --legacy-exp-all-learn [--full]
                                                  # 三臂（含 H6 legacy 臂）
    python -m hstdn.exp.ablation --seeds 0 1 2 --classes 10 --pool 800 ...
    python -m hstdn.exp.ablation --json     # 附加 JSON 汇总输出

规模说明（诚实注明）：G0-11 基准显示 L0 NumPy 内核在 §4 800 池配置上逐样本
发放路径成本为数百 ms 量级（高活性样本），10 类多种子全协议跑 800 池在 L0
阶段不现实（L1 Numba 动机之一）；故 gate/默认消融运行在**门禁规模**
（light n_pool=200、full n_pool=150，均可 --pool 调回 800）。两组同规模
同配置比较，结论不受池数取向影响其相对性。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from hstdn.core.layout import NetConfig
from hstdn.core.network import build_network
from hstdn.core.scheduler import ProtocolConfig, G1Report, run_g1_protocol
from hstdn.data.synthetic import N_CLASSES, synthetic_batch
from hstdn.configs import load_config, to_core_cfg
from hstdn.exp import diagnostics as diag

__all__ = [
    "LIGHT_SPEC", "FULL_SPEC", "Spec", "SeedResult", "GroupResult",
    "AblationResult", "default_spec", "make_protocol", "make_data_fns",
    "run_one_seed", "run_group", "run_ablation", "evaluate_full_criterion",
    "report_lines", "main",
    "EXPG_CRITERIA", "expg_metrics", "evaluate_expg", "expg_report_lines",
    "run_expg_comparison", "print_expg_comparison",
]

# ---------------------------------------------------------------------------
# 默认规格（两模式；均可被 CLI / 调用方覆盖）
# ---------------------------------------------------------------------------

#: 轻量模式：4 类、双种子、池 200 —— 机制快速验证（无官方判据断言，
#: 仅验证协议跑通并打印组间对比，官方判据提示用 --full）。
#: v3.3 FIX-A：默认 pool_learn=False（冻结池内可塑性，STDP 臂唯一合法档，
#: 见 hstdn/configs/stdp_min.yaml）。
LIGHT_SPEC: Dict[str, Any] = dict(
    mode="light",
    n_pool=200,
    n_classes=4,
    train_per_class=8,
    test_per_class=6,
    noise=0.18,
    seeds=(0, 1),
    adapt_epochs=1,
    extra_loops=0,
    adapt_gate_extra_max=0,
    calibrate_samples=5,
    calibrate_max_iter=2,
    t_ms=200,
    pool_learn=False,
)

#: 完整 G1：10 类、≥3 种子 —— 官方判据（§6）：实验组 ≥60% 且显著高于对照组。
FULL_SPEC: Dict[str, Any] = dict(
    mode="full",
    n_pool=150,
    n_classes=10,
    train_per_class=12,
    test_per_class=10,
    noise=0.16,
    seeds=(0, 1, 2),
    adapt_epochs=2,
    extra_loops=0,
    adapt_gate_extra_max=0,
    calibrate_samples=5,
    calibrate_max_iter=2,
    t_ms=200,
    pool_learn=False,
)

# ---------------------------------------------------------------------------
# 规格与结果容器
# ---------------------------------------------------------------------------

_SPEC_KEYS = frozenset({
    "mode", "n_pool", "n_classes", "train_per_class", "test_per_class",
    "noise", "seeds", "adapt_epochs", "extra_loops", "adapt_gate_extra_max",
    "calibrate_samples", "calibrate_max_iter", "t_ms", "pool_learn",
})


@dataclass(frozen=True)
class Spec:
    """一次消融运行的完整规格（mode: light/full 或自定义）。

    pool_learn: False=冻结池内 E→E 可塑性（默认，v3.3 FIX-A / exp-g /
        Diehl-Cook 结构，仅输入→池可学习）；True=池内可塑性开启（legacy
        全可学习条件，H6 / HSTDN-EXP-2026-001 —— 已退役，仅显式保留）。
    """

    n_pool: int
    n_classes: int
    train_per_class: int
    test_per_class: int
    noise: float
    seeds: tuple
    adapt_epochs: int
    extra_loops: int
    adapt_gate_extra_max: int
    calibrate_samples: int
    calibrate_max_iter: int
    t_ms: int
    pool_learn: bool = False
    mode: str = "custom"

    def to_dict(self) -> Dict[str, Any]:
        out = {k: getattr(self, k) for k in _SPEC_KEYS if hasattr(self, k)}
        out["seeds"] = list(self.seeds)
        return out


def default_spec(spec: Optional[Mapping[str, Any]] = None) -> Spec:
    """把 spec（None | 'light'/'full' | mapping）归一化为冻结 Spec。

    Mapping 以对应模式的默认值为底做部分覆盖（LIGHT_SPEC/FULL_SPEC）。
    """
    if spec is None:
        base = dict(LIGHT_SPEC)
    elif isinstance(spec, str):
        if spec == "light":
            base = dict(LIGHT_SPEC)
        elif spec == "full":
            base = dict(FULL_SPEC)
        else:
            raise ValueError(f"unknown spec key {spec!r} ('light'/'full')")
    else:
        unknown = set(spec) - set(LIGHT_SPEC)
        if unknown:
            raise ValueError(f"unknown ablation spec keys: {sorted(unknown)}")
        mode = spec.get("mode", "light")
        base = dict(FULL_SPEC if mode == "full" else LIGHT_SPEC)
        base.update(spec)
    mode = base.get("mode", "custom")
    return Spec(
        mode=str(mode),
        n_pool=int(base["n_pool"]),
        n_classes=int(base["n_classes"]),
        train_per_class=int(base["train_per_class"]),
        test_per_class=int(base["test_per_class"]),
        noise=float(base["noise"]),
        seeds=tuple(int(s) for s in base["seeds"]),
        adapt_epochs=int(base["adapt_epochs"]),
        extra_loops=int(base["extra_loops"]),
        adapt_gate_extra_max=int(base["adapt_gate_extra_max"]),
        calibrate_samples=int(base["calibrate_samples"]),
        calibrate_max_iter=int(base["calibrate_max_iter"]),
        t_ms=int(base["t_ms"]),
        pool_learn=bool(base.get("pool_learn", False)),  # #6：默认冻结
    )


@dataclass
class SeedResult:
    """单种子单组结果（acc + 健康度 + 校准/w_ratio 归因诊断）。"""

    seed: int
    switches_on: bool
    acc: float
    wall_s: float
    report_ok: bool
    n_rounds: int
    eval_rate_hz: float
    silence_frac: float
    capped_ratio: float
    calib_ok: bool = False
    w_ratio: float = 1.0          # 池→池 E 源权重 末/初 均值比（冻结=1）
    # --- v3.3 归因诊断（read_drift_stats 自动记账；#4 度量拆分）---
    drift_frac: float = 0.0        # |Δw|>0.1·|w_init| 可学习边占比（学习度量）
    input_drift_frac: float = 0.0  # 输入边漂移占比（input_w_drift 主项）
    gini: float = 0.0              # within-neuron Gini（当前均值）
    gini_ratio: float = 1.0        # ADAPT 后/初值比（选择性指数）
    gini_std: float = 0.0          # per-neuron Gini 样本 std（正确 per-neuron std）
    input_w_min_sat: float = 0.0   # 输入边 w<=w_lo+ε 占比（R7 触发器）
    budget_dev: float = 0.0        # 劫持度量（FIX-A 后恒 ≈0）
    r7_alert: bool = False         # R7 警报（死数值钉死 / gini_ratio 超限）
    aborted: bool = False          # 协议硬停（v3.3 HARD STOP，未产出 EVAL）
    notes: List[str] = field(default_factory=list)


@dataclass
class GroupResult:
    """一个实验条件（组）的多种子汇总。"""

    name: str
    switches_on: bool
    seed_results: List[SeedResult]
    summary: Dict[str, float]


@dataclass
class AblationResult:
    """消融运行总结果（两组的汇总 + 组间对比 + 判据评估）。"""

    spec: Dict[str, Any]
    exp: GroupResult
    ctrl: GroupResult
    compare: Dict[str, float]
    criterion: Dict[str, Any]          # 仅在 full 模式评估官方判据
    wall_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": dict(self.spec),
            "exp": {"seed_accs": [s.acc for s in self.exp.seed_results],
                    "summary": dict(self.exp.summary)},
            "ctrl": {"seed_accs": [s.acc for s in self.ctrl.seed_results],
                     "summary": dict(self.ctrl.summary)},
            "compare": dict(self.compare),
            "criterion": dict(self.criterion),
            "wall_s": self.wall_s,
        }


# ---------------------------------------------------------------------------
# 协议与数据
# ---------------------------------------------------------------------------


def _net_cfg(spec: Spec, seed: int) -> NetConfig:
    """§4 契约网络（default.yaml）按规格改池规模/种子/pool_learn。

    pool_learn=False（exp-g / Diehl-Cook 回归）会冻结池内 E→E 可塑性
    （core 布局：learn = input | (E-source & pool_learn)）。
    """
    from dataclasses import replace
    base = to_core_cfg(load_config())
    return replace(base, n_pool=spec.n_pool, seed=seed,
                   pool_learn=spec.pool_learn).with_derived()


def make_protocol(spec: Spec, seed: int, switches_on: bool) -> ProtocolConfig:
    """构造协议配置（实验组与对照组共用同一协议，仅开关/臂标志不同）。

    Args:
        spec: 规格。
        seed: 种子。
        switches_on: True=实验组（STDP 臂）：ADAPT 阶段开启 stdp/homeo/norm
            三开关（v3.3 core 经 adapt_enabled=True 进入 ADAPT）；
            False=对照组（LSM 臂）：v3.3 core ``adapt_enabled=False`` ——
            无 ADAPT 阶段、无 ADAPT 门禁（随机固定储备池 + CALIBRATE +
            READOUT 经典 LSM；三开关全程全 False）。其余旋钮完全一致。
    """
    return ProtocolConfig(
        t_ms=spec.t_ms,
        n_adapt_epochs=spec.adapt_epochs,
        adapt_enabled=switches_on,
        adapt_diag_samples=3,
        adapt_gate_extra_max=spec.adapt_gate_extra_max,
        calibrate_samples=spec.calibrate_samples,
        calibrate_max_iter=spec.calibrate_max_iter,
        extra_loops=spec.extra_loops if switches_on else 0,
        seed=seed,
    )

def make_data_fns(spec: Spec, seed: int):
    """零参可调用 train/test data_fn（每次返回新迭代器 (frame, label)）。

    训练/测试批次噪声同一 seed 派生（+1000/+2000），**与组别无关** ——
    实验组与对照组喂完全相同的数据，唯一差异是三开关。
    """
    n_classes = spec.n_classes
    tr = synthetic_batch(spec.train_per_class, noise=spec.noise,
                         seed=1000 + seed)
    te = synthetic_batch(spec.test_per_class, noise=spec.noise,
                         seed=2000 + seed)

    def train_fn():
        for k in range(n_classes):
            for i in range(spec.train_per_class):
                yield tr.frames[k * spec.train_per_class + i].copy(), int(k)

    def test_fn():
        for k in range(n_classes):
            for i in range(spec.test_per_class):
                yield te.frames[k * spec.test_per_class + i].copy(), int(k)

    return train_fn, test_fn


def run_one_seed(spec: Spec, seed: int, switches_on: bool) -> SeedResult:
    """单组单种子：建网（seed 确定性）→ run_g1_protocol（唯一协议入口）。

    Args:
        spec: 规格。
        seed: 种子（网络、数据、读出共用）。
        switches_on: True=实验组（ADAPT 三开关开）；False=对照组（全关）。

    v3.3 自动记账（#4 制度化）：协议前 capture_drift_snapshot，协议后
    read_drift_stats + assess_r7 写入 SeedResult —— drift_frac / Gini /
    gini_ratio / input_w_min_sat / budget_dev / r7_alert；calib_ok（CALIBRATE
    是否落带）；w_ratio（E 源池→池权重末/初比；冻结/ctrl 行 dev=0 = budget_dev
    记账语义）。
    """
    from hstdn.exp import diagnostics as diag
    cfg = _net_cfg(spec, seed)
    bundle = build_network(cfg)
    snap = diag.capture_drift_snapshot(bundle)
    train_fn, test_fn = make_data_fns(spec, seed)
    pc = make_protocol(spec, seed, switches_on)
    t0 = time.perf_counter()
    rep: G1Report = run_g1_protocol(bundle, train_fn, test_fn, pc, seed=seed)
    wall = time.perf_counter() - t0
    ev = rep.eval_rounds[-1] if rep.eval_rounds else {}
    drift = diag.read_drift_stats(bundle, snap)
    r7 = diag.assess_r7(bundle, snap)
    calib_ok = bool(rep.calibrate_rounds
                    and rep.calibrate_rounds[-1].get("ok", False))
    aborted = bool(not rep.ok or not rep.eval_rounds)
    return SeedResult(
        seed=seed,
        switches_on=switches_on,
        acc=float(rep.best_acc),
        wall_s=float(wall),
        report_ok=bool(rep.ok),
        n_rounds=int(rep.n_rounds),
        eval_rate_hz=float(ev.get("mean_rate_hz", 0.0)),
        silence_frac=float(ev.get("silence_frac", 1.0)),
        capped_ratio=float(ev.get("capped_ratio", 0.0)),
        calib_ok=calib_ok,
        w_ratio=float(drift["w_ratio"]),
        drift_frac=float(drift["drift_frac"]),
        input_drift_frac=float(drift["input_drift_frac"]),
        gini=float(drift["gini"]),
        gini_ratio=float(drift["gini_ratio"]),
        gini_std=float(drift["gini_std"]),
        input_w_min_sat=float(drift["input_w_min_sat"]),
        budget_dev=float(drift["budget_dev"]),
        r7_alert=bool(r7["alert"]),
        aborted=aborted,
        notes=list(rep.notes),
    )


def run_group(name: str, switches_on: bool, spec: Spec
              ) -> GroupResult:
    """跑一组（多种子）并汇总 mean±std。"""
    rows: List[SeedResult] = []
    for seed in spec.seeds:
        rows.append(run_one_seed(spec, seed, switches_on))
    accs = [r.acc for r in rows]
    summary = diag.multi_seed_summary(accs)
    return GroupResult(name=name, switches_on=switches_on,
                       seed_results=rows, summary=summary)

# ---------------------------------------------------------------------------
# 组间对比与官方判据
# ---------------------------------------------------------------------------


def run_ablation(spec: Optional[Mapping[str, Any]] = None,
                 ) -> AblationResult:
    """运行完整消融（实验组 + 对照组），返回汇总 + 对比 +（full 时）判据。

    Args:
        spec: None/字符串(light/full)/mapping（default_spec 语义）。

    Returns:
        AblationResult：exp/ctrl 组汇总、compare（均值差/分离界）与 criterion。
    """
    s = default_spec(spec)
    t0 = time.perf_counter()
    exp = run_group("exp", True, s)
    ctrl = run_group("ctrl", False, s)
    wall = time.perf_counter() - t0
    compare = diag.compare_groups(exp.summary, ctrl.summary)
    # 逐种子配对比：同种子 exp 是否全部高于 ctrl（保守显著优势）
    exp_by_seed = {r.seed: r.acc for r in exp.seed_results}
    ctrl_by_seed = {r.seed: r.acc for r in ctrl.seed_results}
    common = sorted(set(exp_by_seed) & set(ctrl_by_seed))
    paired = [exp_by_seed[k] - ctrl_by_seed[k] for k in common]
    dominance = {
        "n_paired": len(common),
        "paired_diffs": paired,
        "all_seeds_exp_gt_ctrl": bool(paired) and all(d > 0.0 for d in paired),
    }
    criterion: Dict[str, Any] = {"mode": s.mode, "evaluated": False}
    if s.mode == "full":
        criterion = evaluate_full_criterion(exp.summary, ctrl.summary,
                                            dominance=dominance)
    return AblationResult(spec=s.to_dict(), exp=exp, ctrl=ctrl,
                          compare=compare, criterion=criterion, wall_s=wall)


def evaluate_full_criterion(exp_summary: Mapping[str, float],
                            ctrl_summary: Mapping[str, float],
                            dominance: Optional[Mapping[str, Any]] = None,
                            *, threshold: float = 0.60,
                            gap_min: float = 0.05) -> Dict[str, Any]:
    """官方完整 G1 判据（文档 §6；纯函数，不 assert）：

    1) 实验组均值 acc >= threshold（默认 0.60 = 60%）；
    2) 显著高于对照组：均值差 >= gap_min（默认 5pp）且每个配种子上
       实验组都高于对照组（dominance.all_seeds_exp_gt_ctrl；同种子
       严格优势，无种子数/方差假设的保守判据）。

    Returns:
        dict：passed、逐子条件布尔与数值（供门禁断言与报告复用）。
    """
    mean_diff = float(exp_summary["mean"]) - float(ctrl_summary["mean"])
    all_win = bool((dominance or {}).get("all_seeds_exp_gt_ctrl", False))
    ge_th = bool(float(exp_summary["mean"]) >= threshold)
    ge_gap = bool(mean_diff >= gap_min)
    out: Dict[str, Any] = {
        "mode": "full",
        "evaluated": True,
        "threshold": threshold,
        "gap_min": gap_min,
        "exp_mean": float(exp_summary["mean"]),
        "ctrl_mean": float(ctrl_summary["mean"]),
        "mean_diff": mean_diff,
        "all_seeds_exp_gt_ctrl": all_win,
        "n_paired": int((dominance or {}).get("n_paired", 0)),
        "exp_ge_threshold": ge_th,
        "mean_diff_ge_gap": ge_gap,
        "passed": bool(ge_th and ge_gap and all_win),
        "explain": "",
    }
    reasons = [
        f"exp_mean={float(exp_summary['mean']) * 100:.1f}% "
        f"{'>=' if ge_th else '<'} {threshold * 100:.0f}%",
        f"mean_diff={mean_diff * 100:+.1f}pp "
        f"{'>=' if ge_gap else '<'} {gap_min * 100:.0f}pp",
        f"all-seed exp>ctrl: {all_win}",
    ]
    out["explain"] = "; ".join(reasons)
    return out


def report_lines(result: AblationResult) -> List[str]:
    """把消融结果格式化为可打印行（纯格式化）。"""
    s = result.spec
    lines = [
        f"Ablation spec: mode={s['mode']} n_pool={s['n_pool']} "
        f"classes={s['n_classes']} train/class={s['train_per_class']} "
        f"test/class={s['test_per_class']} noise={s['noise']} "
        f"seeds={s['seeds']} adapt_epochs={s['adapt_epochs']} "
        f"pool_learn={s.get('pool_learn', False)}",
    ]
    if not bool(s.get("pool_learn", False)):
        lines.append(
            "  arm note: FIX-A 冻结（pool_learn=False）—— exp 臂 = STDP-min"
            " 条件（stdp_min 档）+ 三开关开；ctrl = LSM（默认新配置路径，"
            "v3.3 冻结 #6）"
        )
    else:
        lines.append(
            "  [legacy] pool_learn=True 全可学习条件：H6 已退役路径，仅显式"
            "保留（HSTDN-EXP-2026-001；默认不可达）"
        )
    lines.append(diag.format_g1_ablation(
        result.exp.summary, result.ctrl.summary, result.compare))
    for grp in (result.exp, result.ctrl):
        for r in grp.seed_results:
            lines.append(
                f"    seed {r.seed} ({grp.name}): "
                + ("[ABORT] " if r.aborted else "")
                + f"acc={r.acc * 100:.1f}% "
                f"eval_rate={r.eval_rate_hz:.1f}Hz "
                f"silence={r.silence_frac * 100:.0f}% "
                f"rounds={r.n_rounds} "
                f"drift={r.drift_frac:.3f} gini_r={r.gini_ratio:.3f} "
                f"bdev={r.budget_dev:.1e} r7={int(r.r7_alert)} "
                f"wall={r.wall_s:.0f}s"
            )
    if result.criterion.get("evaluated"):
        c = result.criterion
        lines.append(f"G1 full criterion: passed={c['passed']} — {c['explain']}")
    lines.append(f"total wall: {result.wall_s:.0f}s")
    return lines


# ---------------------------------------------------------------------------
# exp-g 判定（冻结池内可塑性 → Diehl-Cook 结构；归因 Wave 1b / §3 M4）
# ---------------------------------------------------------------------------

#: exp-g 健康判据阈值：acc>=80%、静默<5%、CALIBRATE 全种子 ok、w_ratio≈1
EXPG_CRITERIA = dict(acc_min=0.80, silence_max=0.05, w_ratio_tol=1e-6)


def expg_metrics(grp: GroupResult) -> Dict[str, Any]:
    """exp-g 组汇总指标（acc/静默/校准/w_ratio + v3.3 归因漂移汇总）。"""
    accs = [r.acc for r in grp.seed_results]
    sils = [r.silence_frac for r in grp.seed_results]
    ratios = [r.w_ratio for r in grp.seed_results]
    calib = [bool(r.calib_ok) for r in grp.seed_results]
    return {
        "acc_mean": float(np.mean(accs)),
        "accs": [float(a) for a in accs],
        "silence_mean": float(np.mean(sils)),
        "silence_max": float(np.max(sils)) if sils else 0.0,
        "sils": [float(s) for s in sils],
        "calib_ok": calib,
        "calib_all_ok": bool(calib) and all(calib),
        "w_ratios": [float(x) for x in ratios],
        "w_ratio_dev_max": float(np.max(np.abs(np.asarray(ratios) - 1.0)))
        if ratios else 0.0,
        # --- v3.3 归因汇总（#4 自动记账读数）---
        "drift_mean": float(np.mean([r.drift_frac for r in grp.seed_results])),
        "input_drift_mean": float(np.mean(
            [r.input_drift_frac for r in grp.seed_results])),
        "gini_mean": float(np.mean([r.gini for r in grp.seed_results])),
        "gini_ratio_mean": float(np.mean(
            [r.gini_ratio for r in grp.seed_results])),
        "budget_dev_max": float(np.max(
            [r.budget_dev for r in grp.seed_results]) or 0.0),
        "r7_alerts": int(sum(1 for r in grp.seed_results if r.r7_alert)),
    }


def evaluate_expg(metrics: Mapping[str, Any],
                  criteria: Optional[Mapping[str, float]] = None
                  ) -> Dict[str, Any]:
    """exp-g 健康判据（纯函数，不 assert）：
    acc 均值 >=80%、静默均值 <5%、CALIBRATE 全种子 ok（3/3）、
    w_ratio（池→池 E 源权重 末/初）≈1（冻结生效，容差 1e-6）。
    """
    c = dict(EXPG_CRITERIA)
    if criteria:
        c.update(criteria)
    acc_ok = bool(metrics["acc_mean"] >= c["acc_min"])
    sil_ok = bool(metrics["silence_mean"] < c["silence_max"])
    cal_ok = bool(metrics["calib_all_ok"])
    wr_ok = bool(metrics["w_ratio_dev_max"] <= c["w_ratio_tol"])
    out: Dict[str, Any] = {
        "acc_ok": acc_ok,
        "silence_ok": sil_ok,
        "calib_ok": cal_ok,
        "w_ratio_ok": wr_ok,
        "healthy": bool(acc_ok and sil_ok and cal_ok and wr_ok),
        "explain": (
            f"acc_mean={metrics['acc_mean'] * 100:.1f}% "
            f"({'ok' if acc_ok else '<'} {c['acc_min'] * 100:.0f}%); "
            f"silence_mean={metrics['silence_mean'] * 100:.1f}% "
            f"({'ok' if sil_ok else '>='} {c['silence_max'] * 100:.0f}%); "
            f"calib {sum(metrics['calib_ok'])}/{len(metrics['calib_ok'])} "
            f"({'ok' if cal_ok else 'FAIL'}); "
            f"w_ratio dev max={metrics['w_ratio_dev_max']:.2e} "
            f"({'ok' if wr_ok else 'FAIL'})"
        ),
    }
    return out


def expg_report_lines(expg: GroupResult, metrics: Mapping[str, Any],
                      verdict: Mapping[str, Any], wall_s: float) -> List[str]:
    """exp-g 判定表（per-seed + 判据 + 健康结论）。"""
    lines = [
        "=== exp-g（Diehl-Cook 回归：pool_learn=False 冻结池内 E→E 可塑性，"
        "协议三开关全开 ===",
        f"seeds={[r.seed for r in expg.seed_results]}",
    ]
    for r in expg.seed_results:
        lines.append(
            f"    seed {r.seed}: " + ("[ABORT] " if r.aborted else "")
            + f"acc={r.acc * 100:.1f}% "
            f"rate={r.eval_rate_hz:.1f}Hz silence={r.silence_frac * 100:.1f}% "
            f"calib_ok={r.calib_ok} w_ratio={r.w_ratio:.9f} "
            f"drift={r.drift_frac:.3f} gini_r={r.gini_ratio:.3f} "
            f"r7={int(r.r7_alert)} wall={r.wall_s:.0f}s"
        )
    lines.append(
        f"metrics: acc_mean={metrics['acc_mean'] * 100:.1f}% | "
        f"silence_mean={metrics['silence_mean'] * 100:.1f}% | "
        f"calib {sum(metrics['calib_ok'])}/{len(metrics['calib_ok'])} ok | "
        f"w_ratio dev max={metrics['w_ratio_dev_max']:.2e}"
    )
    lines.append(f"criterion: healthy={verdict['healthy']} — {verdict['explain']}")
    lines.append(f"total wall: {wall_s:.0f}s")
    return lines


def run_expg_comparison(full: bool = False, *, legacy: bool = False
                        ) -> Dict[str, Any]:
    """exp-g 判定入口（light/full 两模式；v3.3 归因 Wave 1b / 冻结 #6）：

    - expg（判定臂）：pool_learn=False（冻结池内可塑性 → Diehl-Cook 结构，
      stdp_min 档）+ 三开关全开；
    - ctrl（LSM 臂）：随机固定储备池（adapt_enabled=False，三开关全关）。
    **默认双臂**（新配置 FIX-A 冻结 + LSM）；当 ``legacy=True``（CLI
    ``--legacy-exp-all-learn``）时才并入 exp（legacy 对照臂）：pool_learn=
    True（池内 E→E 可塑性，v3.3 前复现条件）+ 三开关全开 —— 该全可学习
    条件属 H6 已发表证据链 HSTDN-EXP-2026-001，代码路径保留但**默认不可
    达**（退役）。各臂同种子同数据；protocol/其余配置一致。
    - 输出：expg 判据表（silence<5% / CALIBRATE 全 ok / w_ratio≈1 /
      acc>=80%）+ v3.3 归因读数（drift/gini/budget_dev/r7）+ 臂对比。

    Args:
        full: True=FULL 规模（10 类/noise0.16/池150/adapt2/3 种子）；
            False=light 规模（4 类/池200 双种子，冒烟）。
        legacy: True=显式并入 H6 legacy 全可学习臂（三臂表；默认 False）。

    Returns:
        dict：expg（GroupResult）、metrics、verdict、臂对比摘要、wall；
        ``legacy_exp_on`` 标记 legacy 臂是否并入。
    """
    mode = "full" if full else "light"
    t0 = time.perf_counter()
    expg_spec = default_spec({"mode": mode, "pool_learn": False})  # 判定臂
    ctrl_spec = default_spec(mode)                                 # LSM 臂
    expg = run_group("expg", True, expg_spec)
    ctrl = run_group("ctrl", False, ctrl_spec)
    metrics = expg_metrics(expg)
    verdict = evaluate_expg(metrics)
    compare = {
        "expg": _group_compare(expg),
        "ctrl": _group_compare(ctrl),
    }
    out: Dict[str, Any] = {
        "mode": mode,
        "spec": dict(expg_spec.to_dict()),
        "expg": {"seed_results": [vars(r) for r in expg.seed_results]},
        "legacy_exp_on": bool(legacy),
        "metrics": metrics,
        "verdict": verdict,
        "compare": compare,
        "wall_s": 0.0,
    }
    if legacy:
        exp_spec = default_spec({"mode": mode, "pool_learn": True})  # H6 legacy
        exp = run_group("exp", True, exp_spec)
        out["compare"] = {"exp": _group_compare(exp), **compare}
        out["exp"] = {"seed_results": [vars(r) for r in exp.seed_results]}
    out["wall_s"] = float(time.perf_counter() - t0)
    return out


def _group_compare(grp: GroupResult) -> Dict[str, float]:
    """组级对比摘要（acc±std + 静默 + v3.3 归因读数）。"""
    rows = grp.seed_results
    return {
        "acc_mean": grp.summary["mean"],
        "acc_std": grp.summary["std"],
        "silence_mean": float(np.mean([r.silence_frac for r in rows])),
        "drift_mean": float(np.mean([r.drift_frac for r in rows])),
        "gini_mean": float(np.mean([r.gini for r in rows])),
        "gini_ratio_mean": float(np.mean([r.gini_ratio for r in rows])),
        "budget_dev_max": float(np.max([r.budget_dev for r in rows]) or 0.0),
        "r7_alerts": float(sum(1 for r in rows if r.r7_alert)),
        "aborted": float(sum(1 for r in rows if r.aborted)),
    }


def print_expg_comparison(out: Dict[str, Any]) -> None:
    """打印 exp-g 判定结果（per-seed 判据表 + v3.3 归因读数）+ 臂对比。

    默认双臂（expg 判定臂 vs ctrl LSM）；``legacy_exp_on=True`` 时另含 H6
    legacy 全可学习臂（--legacy-exp-all-learn，退役保留路径）。
    """
    print("=" * 78)
    arms = " [+legacy exp 全可学习臂]" if out.get("legacy_exp_on") else ""
    print(f"exp-g comparison (mode={out['mode']}, pool_learn=False){arms}")
    spec = out["spec"]
    print(f"spec: n_pool={spec['n_pool']} classes={spec['n_classes']} "
          f"train/class={spec['train_per_class']} "
          f"test/class={spec['test_per_class']} noise={spec['noise']} "
          f"seeds={spec['seeds']} adapt_epochs={spec['adapt_epochs']}")
    if not out.get("legacy_exp_on"):
        print("  (双臂：expg 判定臂 vs ctrl LSM；H6 legacy 全可学习臂经 "
              "--legacy-exp-all-learn 显式启用——已退役默认不可达)")
    c = out["compare"]
    print("  group | acc         | silence | drift | gini    | gini_ratio | "
          "budget_dev | r7")
    for name in c:
        cc = c[name]
        print(f"  {name:5s} | {cc['acc_mean'] * 100:5.1f}±"
              f"{cc['acc_std'] * 100:4.1f}% | "
              f"{cc['silence_mean'] * 100:5.1f}% | "
              f"{cc['drift_mean']:.3f} | {cc['gini_mean']:.4f} | "
              f"{cc['gini_ratio_mean']:.3f}    | "
              f"{cc['budget_dev_max']:.1e}   | {int(cc['r7_alerts'])}"
              + (f"   [{int(cc['aborted'])}/{len(out['expg']['seed_results'])}"
                 f" ABORTED]" if cc["aborted"] else ""))
    print("  expg per-seed (v3.3 自动记账):")
    for r in out["expg"]["seed_results"]:
        print(f"    seed {r['seed']}: "
              + ("[ABORT] " if r["aborted"] else "")
              + f"acc={r['acc'] * 100:.1f}% "
              f"rate={r['eval_rate_hz']:.1f}Hz silence={r['silence_frac'] * 100:.1f}% "
              f"calib_ok={r['calib_ok']} "
              f"w_ratio_dev={abs(r['w_ratio'] - 1):.2e} "
              f"budget_dev={r['budget_dev']:.1e} "
              f"drift={r['drift_frac']:.3f} gini={r['gini']:.4f} "
              f"gini_ratio={r['gini_ratio']:.3f} r7={int(r['r7_alert'])} "
              f"wall={r['wall_s']:.0f}s")
    if out.get("legacy_exp_on") and "exp" in out:
        print("  exp (legacy all-learn, pool_learn=True) per-seed:")
        for r in out["exp"]["seed_results"]:
            print(f"    seed {r['seed']}: "
                  + ("[ABORT] " if r["aborted"] else "")
                  + f"acc={r['acc'] * 100:.1f}% "
                  f"rate={r['eval_rate_hz']:.1f}Hz "
                  f"silence={r['silence_frac'] * 100:.1f}% "
                  f"calib_ok={r['calib_ok']} "
                  f"drift={r['drift_frac']:.3f} gini_r={r['gini_ratio']:.3f} "
                  f"r7={int(r['r7_alert'])} wall={r['wall_s']:.0f}s")
    v = out["verdict"]
    print(f"exp-g criterion: healthy={v['healthy']} — {v['explain']}")
    print(f"total wall: {out['wall_s']:.0f}s")


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
        prog="python -m hstdn.exp.ablation",
        description="G1 随机储备池消融（exp vs 经典 LSM 对照；R1）"
                    "+ exp-g 判定（冻结池内可塑性，Diehl-Cook 回归）",
    )
    ap.add_argument("--full", action="store_true",
                    help="完整规模（10 类、≥3 种子，官方判据）")
    ap.add_argument("--expg", "--freeze-pool", dest="expg",
                    action="store_true",
                    help="exp-g 判定：pool_learn=False 冻结池内 E→E 可塑性"
                         "（Diehl-Cook 结构），协议三开关全开；输出判据表"
                         "（silence<5%/CALIBRATE 全 ok/w_ratio≈1/acc>=80%）"
                         "及与 LSM ctrl 对比（默认双臂；默认 light，--full "
                         "组合）")
    ap.add_argument("--legacy-exp-all-learn", dest="legacy_all_learn",
                    action="store_true",
                    help="[退役路径] 把 H6 legacy 全可学习臂（pool_learn=True "
                         "+ 三开关全开，HSTDN-EXP-2026-001）并入 exp-g 三臂"
                         "对比（默认不可达；需配合 --expg）")
    ap.add_argument("--classes", type=int, default=None)
    ap.add_argument("--train", type=int, default=None,
                    help="每类训练样本数")
    ap.add_argument("--test", type=int, default=None)
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--pool", type=int, default=None,
                    help="池规模（默认 light=200/full=150；可设 800）")
    ap.add_argument("--seeds", type=_parse_seeds, default=None,
                    help="种子列表，如 0,1,2")
    ap.add_argument("--adapt-epochs", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="附加 JSON 汇总")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.expg or args.legacy_all_learn:
        out = run_expg_comparison(full=args.full,
                                  legacy=args.legacy_all_learn)
        print_expg_comparison(out)
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False,
                             default=float))
        return 0

    overrides = {
        k: v for k, v in {
            "mode": "full" if args.full else None,
            "n_classes": args.classes,
            "train_per_class": args.train,
            "test_per_class": args.test,
            "noise": args.noise,
            "n_pool": args.pool,
            "adapt_epochs": args.adapt_epochs,
        }.items() if v is not None
    }
    if args.seeds is not None:
        overrides["seeds"] = tuple(args.seeds)
    spec = default_spec(overrides if overrides else ("full" if args.full
                                                     else "light"))
    # 命令行 --full 与覆盖合并：spec 已由 default_spec 归一化
    result = run_ablation(spec.to_dict())
    print("\n".join(report_lines(result)))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
