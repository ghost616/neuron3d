"""ablation.py -- G1 随机储备池对照消融（R1；exp 模块，文档 §6 G1 / §8）。

对比设计（文档 §11 冻结条件：两组唯一差异变量 = stdp_on/homeo_on/norm_on
三开关）：
    实验组  : 三开关全开 —— 经 core.scheduler.run_g1_protocol 的 ADAPT 阶段
              （stdp/homeo/norm 逐样本开）适应储备池，再 CALIBRATE+READOUT；
    对照组  : 三开关全关 —— 随机**固定**储备池（不学习），仅 CALIBRATE 校准 +
              READOUT，即经典 LSM（液体状态机）范式（文档 §6 G1 回退方案）。
实现上两组都复用 core.scheduler.run_g1_protocol 作为唯一协议入口；实验组以
``n_adapt_epochs>=1`` 表达"ADAPT 阶段开启三开关"，对照组以
``n_adapt_epochs=0``（任何阶段都不开三开关、不触发 eta 衰减回环）表达——协议
其余旋钮（calibrate/collect/readout/eval 阈值与样本数）完全一致，数据与种子
也一致，故唯一差异即为三开关（对照组的 n_adapt_epochs=0 仅是把三开关置 False
的等价表述，见 run_g1_protocol 内部：ADAPT 是唯一开启开关的阶段）。

运行：
    python -m hstdn.exp.ablation            # light（机制快速验证）
    python -m hstdn.exp.ablation --full     # 完整 G1（10 类、≥3 种子）
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
]

# ---------------------------------------------------------------------------
# 默认规格（两模式；均可被 CLI / 调用方覆盖）
# ---------------------------------------------------------------------------

#: 轻量模式：4 类、双种子、池 200 —— 机制快速验证（无官方判据断言，
#: 仅验证协议跑通并打印组间对比，官方判据提示用 --full）。
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
)

# ---------------------------------------------------------------------------
# 规格与结果容器
# ---------------------------------------------------------------------------

_SPEC_KEYS = frozenset({
    "mode", "n_pool", "n_classes", "train_per_class", "test_per_class",
    "noise", "seeds", "adapt_epochs", "extra_loops", "adapt_gate_extra_max",
    "calibrate_samples", "calibrate_max_iter", "t_ms",
})


@dataclass(frozen=True)
class Spec:
    """一次消融运行的完整规格（mode: light/full 或自定义）。"""

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
    )


@dataclass
class SeedResult:
    """单种子单组结果（acc + 健康度 + 报告摘要）。"""

    seed: int
    switches_on: bool
    acc: float
    wall_s: float
    report_ok: bool
    n_rounds: int
    eval_rate_hz: float
    silence_frac: float
    capped_ratio: float
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
    """§4 契约网络（default.yaml）按规格改池规模/种子（两组共用同一 cfg）。"""
    from dataclasses import replace
    base = to_core_cfg(load_config())
    return replace(base, n_pool=spec.n_pool, seed=seed).with_derived()


def make_protocol(spec: Spec, seed: int, switches_on: bool) -> ProtocolConfig:
    """构造协议配置（实验组与对照组共用同一协议，仅开关不同）。

    Args:
        spec: 规格。
        seed: 种子。
        switches_on: True=实验组：ADAPT 阶段开启 stdp/homeo/norm 三开关
            （n_adapt_epochs>=1，run_g1_protocol 内唯一开启三开关的阶段）；
            False=对照组：三开关全 False（n_adapt_epochs=0，等价于随机固定
            储备池 + CALIBRATE + READOUT 的经典 LSM）。其余全部旋钮一致。
    """
    return ProtocolConfig(
        t_ms=spec.t_ms,
        n_adapt_epochs=spec.adapt_epochs if switches_on else 0,
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
    """
    cfg = _net_cfg(spec, seed)
    bundle = build_network(cfg)
    train_fn, test_fn = make_data_fns(spec, seed)
    pc = make_protocol(spec, seed, switches_on)
    t0 = time.perf_counter()
    rep: G1Report = run_g1_protocol(bundle, train_fn, test_fn, pc, seed=seed)
    wall = time.perf_counter() - t0
    ev = rep.eval_rounds[-1] if rep.eval_rounds else {}
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
        f"seeds={s['seeds']} adapt_epochs={s['adapt_epochs']}",
    ]
    lines.append(diag.format_g1_ablation(
        result.exp.summary, result.ctrl.summary, result.compare))
    for grp in (result.exp, result.ctrl):
        for r in grp.seed_results:
            lines.append(
                f"    seed {r.seed} ({grp.name}): acc={r.acc * 100:.1f}% "
                f"eval_rate={r.eval_rate_hz:.1f}Hz "
                f"silence={r.silence_frac * 100:.0f}% "
                f"rounds={r.n_rounds} wall={r.wall_s:.0f}s"
            )
    if result.criterion.get("evaluated"):
        c = result.criterion
        lines.append(f"G1 full criterion: passed={c['passed']} — {c['explain']}")
    lines.append(f"total wall: {result.wall_s:.0f}s")
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
        prog="python -m hstdn.exp.ablation",
        description="G1 随机储备池消融（exp vs 经典 LSM 对照；R1）",
    )
    ap.add_argument("--full", action="store_true",
                    help="完整 G1（10 类、≥3 种子，官方判据）")
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
