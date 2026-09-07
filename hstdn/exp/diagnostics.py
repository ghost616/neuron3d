"""diagnostics.py -- 诊断面板函数（G1 与 gates 调试共用；exp 模块交付）。

定位：不承载业务断言，只做**读数/汇总**，供 G1 实验评估与 G0 门禁调试复用。
core 依赖一律惰性导入（本模块自身保持轻量、可独立 import）。

覆盖的诊断量（对应开发计划）：
    率/沉默：firing_rates_hz / silence_ratio / rate_summary
    类间余弦：cos_similarity / class_cosine_matrix（类均值特征向量两两 cos）
    sqrt+L2 特征可比性：feature_transform(mode="sqrt_l2") / feature_cos_similarity
    （对 Poisson 类计数向量做方差稳定化 sqrt 再 L2 归一化，余弦可比）
    顶界突触占比：capped_synapse_ratio（learnable 中 w >= w_hi 份额）
    输入→池 w 均值：input_to_pool_w_mean
    延迟直方图：delay_histogram（对齐 1..delay_max）
    汇总面板：plasticity_panel / network_panel / sample_panel / print_panel

约定：本模块不 import core 顶层符号；bundle 相关函数内部
``from hstdn.core...`` 惰性导入，core 变更接口时诊断层只需同步内联调用点。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "firing_rates_hz", "silence_ratio", "rate_summary",
    "cos_similarity", "class_cosine_matrix",
    "feature_transform", "feature_cos_similarity",
    "capped_synapse_ratio", "input_to_pool_w_mean",
    "plasticity_panel", "delay_histogram", "network_panel",
    "sample_panel", "print_panel",
    "multi_seed_summary", "compare_groups", "format_g1_ablation",
    # v3.3 诊断升级（#3/#4/#9）
    "per_neuron_gini", "gini_within_neuron",
    "rate_percentiles", "effective_dimension", "bimodality_coefficient",
    "rate_distribution_panel",
    "capture_drift_snapshot", "read_drift_stats", "assess_r7",
]


# ---------------------------------------------------------------------------
# 发放率 / 沉默比例
# ---------------------------------------------------------------------------


def firing_rates_hz(spike_counts: Any, window_s: float = 0.2) -> np.ndarray:
    """逐神经元发放率（Hz）= counts / window_s。

    Args:
        spike_counts: (N,) 整型每样本发放计数（pool-local）。
        window_s: 统计窗（秒），默认 0.2（契约 T=200ms）。

    Returns:
        (N,) float64 发放率数组。
    """
    return np.asarray(spike_counts, dtype=np.float64) / float(window_s)


def silence_ratio(spike_counts: Any) -> float:
    """沉默（整窗零发放）神经元占比 ∈ [0, 1]；空输入返回 0.0。"""
    c = np.asarray(spike_counts)
    if c.size == 0:
        return 0.0
    return float(np.count_nonzero(c == 0)) / float(c.size)


def rate_summary(spike_counts: Any, window_s: float = 0.2) -> Dict[str, float]:
    """发放率面板摘要：总数/活跃数/沉默比/全池均值/活跃均值/峰值。"""
    c = np.asarray(spike_counts, dtype=np.int64)
    rates = c / float(window_s)
    n = int(c.size)
    n_active = int(np.count_nonzero(c > 0))
    out: Dict[str, float] = {
        "n_neurons": float(n),
        "n_active": float(n_active),
        "n_silent": float(n - n_active),
        "silence_ratio": float(n - n_active) / float(n) if n else 0.0,
        "total_spikes": float(c.sum()),
        "mean_rate_hz": float(rates.mean()) if n else 0.0,
        "max_rate_hz": float(rates.max()) if n else 0.0,
        "window_s": float(window_s),
    }
    if n_active:
        out["mean_active_rate_hz"] = float(rates[c > 0].mean())
    else:
        out["mean_active_rate_hz"] = 0.0
    return out


# ---------------------------------------------------------------------------
# 类间余弦（类均值特征向量两两相似度）
# ---------------------------------------------------------------------------


def cos_similarity(a: Any, b: Any) -> float:
    """两个非零向量的余弦相似度（零向量视为与任何向量相似度为 0）。"""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx == 0.0 or ny == 0.0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def class_cosine_matrix(class_means: Any) -> np.ndarray:
    """类均值特征矩阵的逐对余弦矩阵。

    Args:
        class_means: (K, D) —— 每行是一个类的平均特征向量（K 类）。

    Returns:
        (K, K) 余弦矩阵；同类别对角为 1.0（非零均值），值域 [-1, 1]。
    """
    m = np.asarray(class_means, dtype=np.float64)
    if m.ndim != 2:
        raise AssertionError(
            f"class_means 需为 (K, D) 二维数组，实际 shape={m.shape}"
        )
    norms = np.linalg.norm(m, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = m / norms[:, None]
    u[~np.isfinite(u)] = 0.0
    return u @ u.T


# ---------------------------------------------------------------------------
# sqrt+L2 特征可比性（v0 特征 / 方差稳定化）
# ---------------------------------------------------------------------------


def feature_transform(features: Any, mode: str = "sqrt_l2",
                      mask: Optional[Sequence[int]] = None) -> np.ndarray:
    """把 (N, D) 特征行归一化为可比的读出输入。

    Args:
        features: (N, D) 特征矩阵（如发放计数 / 直方图计数，非负）。
        mode: 当前实现 "sqrt_l2"：逐元素 sqrt（Poisson 类计数的方差稳定化）
            后按行 L2 归一化；未知 mode 抛 ValueError（无 pass 占位）。
        mask: 可选特征列掩码（如 features.feature_mask=pool 只取池单元列）。

    Returns:
        (N, D') 归一化特征（D' = mask 大小或 D）。
    """
    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 2:
        raise AssertionError(
            f"features 需为 (N, D) 二维数组，实际 shape={f.shape}"
        )
    if mask is not None:
        f = f[:, np.asarray(mask, dtype=np.intp)]
    if mode == "sqrt_l2":
        y = np.sqrt(np.maximum(f, 0.0))
        norms = np.linalg.norm(y, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return y / norms
    raise ValueError(f"feature_transform: 未知 mode={mode!r}（支持 'sqrt_l2'）")


def feature_cos_similarity(a: Any, b: Any, mode: str = "sqrt_l2") -> float:
    """两行特征的 sqrt+L2 可比余弦（计数尺度不同也可比）。"""
    x = feature_transform(np.atleast_2d(a), mode=mode)
    y = feature_transform(np.atleast_2d(b), mode=mode)
    return cos_similarity(x[0], y[0])


# ---------------------------------------------------------------------------
# bundle 诊断（core 惰性导入）
# ---------------------------------------------------------------------------


def capped_synapse_ratio(bundle: Any) -> float:
    """顶界突触占比：learnable 中 w >= w_hi 的份额（委托 core 原语）。"""
    from hstdn.core.plasticity import capped_ratio
    return capped_ratio(bundle)


def input_to_pool_w_mean(bundle: Any) -> float:
    """输入→池 w 均值（委托 core 原语）。"""
    from hstdn.core.plasticity import input_w_mean
    return input_w_mean(bundle)


def plasticity_panel(bundle: Any) -> Dict[str, float]:
    """可塑性诊断面板：capped_ratio / input_w_mean / w 界 / 可学习边数。"""
    from hstdn.core.plasticity import plasticity_diagnostics
    d = plasticity_diagnostics(bundle)
    return {
        "capped_ratio": float(d["capped_ratio"]),
        "input_w_mean": float(d["input_w_mean"]),
        "learn_w_min": float(d["learn_w_min"]),
        "learn_w_max": float(d["learn_w_max"]),
        "w_lo": float(d["w_lo"]),
        "w_hi": float(d["w_hi"]),
        "n_learnable": float(d["n_learnable"]),
    }


def delay_histogram(bundle: Any) -> List[int]:
    """传导延迟直方图（对齐 delay_min..delay_max；core 网络报告同源）。"""
    from hstdn.core.network import structural_report
    return list(structural_report(bundle)["delay_hist"])


def network_panel(bundle: Any) -> Dict[str, Any]:
    """网络结构面板：structural_report 摘要 + 延迟直方图。"""
    from hstdn.core.network import structural_report
    r = structural_report(bundle)
    return {
        "n_in": int(r["n_in"]),
        "n_pool": int(r["n_pool"]),
        "n_edges_in": int(r["n_edges_in"]),
        "n_edges_pool": int(r["n_edges_pool"]),
        "n_e_pool": int(r["n_e_pool"]),
        "n_i_pool": int(r["n_i_pool"]),
        "outdeg_max": int(r["outdeg_max"]),
        "e_src_outdeg_max": int(r["e_src_outdeg_max"]),
        "pools_zero_indegree": int(r["pools_zero_indegree"]),
        "pools_isolated": int(r["pools_isolated"]),
        "input_w_mean": float(r["input_w_mean"]),
        "e_w_mean": float(r["e_w_mean"]),
        "delay_hist": list(r["delay_hist"]),
        "wheel_l": int(r["wheel_l"]),
    }


def sample_panel(bundle: Any, stats: Optional[Dict[str, Any]] = None,
                 window_s: Optional[float] = None) -> Dict[str, Any]:
    """样本级面板：发放率/沉默 + （可选）结构诊断。

    Args:
        bundle: NetworkBundle（读取 bundle.counts 与 bundle.cfg）。
        stats: run_sample 返回的 dict（含 spike_counts）；缺省时读取 bundle。
        window_s: 统计窗秒数，缺省取 bundle.cfg.window_s。

    Returns:
        dict：rate_summary 各键 + capped_ratio / input_w_mean。
    """
    cfg = bundle.cfg
    win = float(window_s) if window_s is not None else float(cfg.window_s)
    if stats is not None and stats.get("spike_counts") is not None:
        counts = np.asarray(stats["spike_counts"])
    else:
        counts = np.asarray(bundle.counts)
    panel: Dict[str, Any] = dict(rate_summary(counts, window_s=win))
    panel["capped_ratio"] = capped_synapse_ratio(bundle)
    panel["input_w_mean"] = input_to_pool_w_mean(bundle)
    return panel


def print_panel(panel: Dict[str, Any], title: str = "diagnostics") -> None:
    """打印诊断面板（键 → 值，易读对齐）。"""
    print(f"=== {title} ===")
    for k, v in panel.items():
        if isinstance(v, (list, tuple)):
            print(f"  {k:24s}: {list(v)}")
        elif isinstance(v, float):
            print(f"  {k:24s}: {v:.6g}")
        else:
            print(f"  {k:24s}: {v}")


# ---------------------------------------------------------------------------
# G1 评估辅助（多种子汇总 / 组间比较 / 报告格式化 —— 纯函数，不承载断言）
# ---------------------------------------------------------------------------


def multi_seed_summary(accuracies: Sequence[float],
                       ddof: Optional[int] = None) -> Dict[str, float]:
    """多种子精度汇总：mean ± std（选择 ddof）。

    Args:
        accuracies: 各种子测试精度（∈ [0,1]）。
        ddof: 标准差自由度；None 时 n>=2 用样本 std（ddof=1）、n==1 用 0。

    Returns:
        dict：mean/std/min/max/n/mean_pct/std_pct（百分比视图）。
    """
    a = np.asarray(accuracies, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"accuracies must be a non-empty 1-D sequence, got {a}")
    if ddof is None:
        ddof = 1 if a.size >= 2 else 0
    if a.size <= ddof:
        ddof = 0
    std = float(a.std(ddof=ddof))
    return {
        "n": float(a.size),
        "mean": float(a.mean()),
        "std": float(std),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean_pct": float(a.mean()) * 100.0,
        "std_pct": float(std) * 100.0,
    }


def compare_groups(exp_summary: Mapping[str, Any],
                   ctrl_summary: Mapping[str, Any]) -> Dict[str, float]:
    """实验组 vs 对照组均值/逐种子差汇总（纯统计，不含断言）。

    Args:
        exp_summary / ctrl_summary: multi_seed_summary 输出。

    Returns:
        dict：mean_diff / mean_diff_pct / 以及最大单种子对照差 max_seed_gap
        （= min(exp)-max(ctrl) 视差保守界，种子数相同时有效）。
    """
    mean_diff = float(exp_summary["mean"]) - float(ctrl_summary["mean"])
    out: Dict[str, float] = {
        "mean_diff": mean_diff,
        "mean_diff_pct": mean_diff * 100.0,
        "exp_mean_pct": float(exp_summary["mean_pct"]),
        "ctrl_mean_pct": float(ctrl_summary["mean_pct"]),
    }
    if int(exp_summary["n"]) >= 1 and int(ctrl_summary["n"]) >= 1:
        # 保守分离界：实验组最差种子仍优于对照组最好种子
        out["min_exp_gt_max_ctrl"] = float(exp_summary["min"]) - \
            float(ctrl_summary["max"])
    return out


def format_g1_ablation(exp_summary: Mapping[str, Any],
                       ctrl_summary: Mapping[str, Any],
                       compare: Mapping[str, float],
                       *, group_labels=("exp(STDP+homeo+norm)", "ctrl(random)")) -> str:
    """格式化 G1 消融对比表（纯格式化；mean±std 百分比视图）。"""
    e, c = group_labels
    lines = [
        f"G1 ablation ({e} vs {c}):",
        f"  {e:26s}: {exp_summary['mean_pct']:.1f}% ± "
        f"{exp_summary['std_pct']:.1f}%  (n={int(exp_summary['n'])}, "
        f"min {exp_summary['min'] * 100:.1f}% / max {exp_summary['max'] * 100:.1f}%)",
        f"  {c:26s}: {ctrl_summary['mean_pct']:.1f}% ± "
        f"{ctrl_summary['std_pct']:.1f}%  (n={int(ctrl_summary['n'])}, "
        f"min {ctrl_summary['min'] * 100:.1f}% / max {ctrl_summary['max'] * 100:.1f}%)",
        f"  mean diff                : {compare['mean_diff_pct']:+.1f} pp",
    ]
    if "min_exp_gt_max_ctrl" in compare:
        lines.append(
            f"  min(exp) - max(ctrl)    : "
            f"{compare['min_exp_gt_max_ctrl'] * 100:+.1f} pp"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# v3.3 诊断升级（冻结清单 #3/#4/#9；§7 风险登记册 / §6 诊断；纯函数）
#
# 选择性指数：within-neuron Gini（每神经元可学习入突触权重 Gini）+ gini_ratio
# （ADAPT 后/初值比）—— 原型经 manipulation check 验证（drift_frac≈28.7%、
# gini Δ≈0），转正为正式诊断量（G3 终审机制级前哨）。
# 度量拆分（#4）：budget_dev（劫持度量——池内突触并入预算时的预算偏差，
#   FIX-A 后恒 ≈0：in_learn/in_sum0 仅覆盖输入边）与 input_w_drift（学习度量：
#   drift_frac（|Δw|>0.1·|w_init| 的占比）+ within-neuron Gini）。
# R7 触发器（#4，死数值钉死）：输入突触 w_min 饱和占比 >20% 或 gini_ratio>1.3。
# 率分布（#3）：P50/P95 分位、有效维度（非沉默且未过热（rate<3×target））、
#   双峰系数；均值率降级为参考项（键保留但标注 note）。
# 卫生（#9）：无 std_ratio 类死汇总——per-neuron std 一律按定义正确计算
#   （如 gini_std = per-neuron Gini 的样本 std，见 read_drift_stats）。
# ---------------------------------------------------------------------------


def _gini_1d(values: Any) -> float:
    """单条样本的 Gini 系数（非负向量；n<2 时返回 0.0；公式稳定）。

    G = (2 * Σ_{i=1..n} i * x_sorted_i) / (n * Σx) - (n+1)/n （x 升序）。
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    x = x[x >= 0.0]
    n = int(x.size)
    if n == 0:
        return 0.0
    s = float(x.sum())
    if s <= 0.0:
        return 0.0
    x = np.sort(x)
    csum = float(np.dot(np.arange(1, n + 1), x))
    return max(0.0, (2.0 * csum) / (float(n) * s) - (float(n) + 1.0) / float(n))


def _learnable_incoming_groups(bundle: Any):
    """(pool_ids, weights)：可学习入突触（bundle.in_learn 行，正权重）分组。

    惰性导入 core 类型以保持模块轻量。
    """
    cfg = bundle.cfg
    ptr = bundle.in_learn_ptr
    idx = bundle.in_learn_idx
    pool_ids = np.repeat(np.arange(int(cfg.n_pool), dtype=np.int64),
                         np.diff(ptr))
    w = np.asarray(bundle.csr_w[idx], dtype=np.float64)
    return pool_ids, w


def per_neuron_gini(bundle: Any) -> np.ndarray:
    """逐神经元 within-neuron Gini（可学习入突触权重；无/单突触神经元 = 0）。"""
    pool_ids, w = _learnable_incoming_groups(bundle)
    n_pool = int(bundle.cfg.n_pool)
    out = np.zeros(n_pool, dtype=np.float64)
    counts = np.bincount(pool_ids, minlength=n_pool)
    # 逐神经元向量化 Gini（按池分组：对每池排序+公式）
    order = np.lexsort((w, pool_ids))          # 按池分组内升序
    w = w[order]
    pool_ids = pool_ids[order]
    bounds = np.cumsum(counts)
    lo = np.concatenate([[0], bounds[:-1]])
    has = counts >= 2
    for p in np.flatnonzero(has):
        seg = w[lo[p]:bounds[p]]
        n = int(counts[p])
        s = float(seg.sum())
        if s > 0.0:
            csum = float(np.dot(np.arange(1, n + 1), seg))
            out[p] = max(0.0, (2.0 * csum) / (float(n) * s)
                         - (float(n) + 1.0) / float(n))
    return out


def gini_within_neuron(bundle: Any) -> float:
    """within-neuron Gini（每神经元入突触权重 Gini）的全池均值。

    返回 float ∈ [0, 1]（G3 终审机制级前哨；正式诊断量）。
    """
    vals = per_neuron_gini(bundle)
    return float(vals.mean())


def _rate_arr(spike_counts: Any, window_s: float) -> np.ndarray:
    return np.asarray(spike_counts, dtype=np.float64) / float(window_s)


def rate_percentiles(spike_counts: Any, window_s: float = 0.2,
                     qs=(50.0, 95.0)) -> Dict[str, float]:
    """发放率分位数面板（默认 P50/P95）。"""
    r = _rate_arr(spike_counts, window_s)
    if r.size == 0:
        return {}
    q = np.percentile(r, list(qs))
    return {f"p{int(qi)}_rate_hz": float(v)
            for qi, v in zip(qs, q)}


def effective_dimension(spike_counts: Any, window_s: float = 0.2,
                        overheat_factor: float = 3.0,
                        target_rate_hz: Optional[float] = None) -> Dict[str, float]:
    """有效维度：非沉默且未过热（rate < overheat_factor×target）的神经元。

    target_rate_hz 缺省按 8 Hz 契约；返回 n_effective 与占比 effective_ratio
    （相对总神经元数）。
    """
    r = _rate_arr(spike_counts, window_s)
    n = int(r.size)
    target = 8.0 if target_rate_hz is None else float(target_rate_hz)
    if n == 0:
        return {"n_effective": 0.0, "effective_ratio": 0.0,
                "overheat_hz": float(overheat_factor) * target}
    hot = r >= overheat_factor * target
    eff = (r > 0.0) & (~hot)
    return {"n_effective": float(np.count_nonzero(eff)),
            "effective_ratio": float(np.count_nonzero(eff)) / float(n),
            "overheat_hz": float(overheat_factor) * target}


def bimodality_coefficient(spike_counts: Any, window_s: float = 0.2) -> float:
    """发放率分布双峰系数 BC=(skew²+1)/(kurt+3(n−1)²/((n−2)(n−3)))。

    BC>0.555 提示双峰；样本 n<4 时返回 0.0（信息不足）。纯 NumPy 矩。
    """
    r = _rate_arr(spike_counts, window_s)
    n = int(r.size)
    if n < 4:
        return 0.0
    mu = float(r.mean())
    if mu == 0.0:
        return 0.0
    m2 = float(np.mean((r - mu) ** 2))
    if m2 <= 0.0:
        return 0.0
    m3 = float(np.mean((r - mu) ** 3))
    m4 = float(np.mean((r - mu) ** 4))
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2.0) - 3.0
    denom = kurt + 3.0 * (float(n) - 1.0) ** 2 / (
        float(n - 2) * float(n - 3))
    if denom <= 0.0:
        return 1.0
    return float(min(1.0, (skew ** 2 + 1.0) / denom))


def rate_distribution_panel(spike_counts: Any, window_s: float = 0.2,
                            target_rate_hz: Optional[float] = None,
                            overheat_factor: float = 3.0) -> Dict[str, float]:
    """率分布升级面板（#3）：P50/P95、有效维度、双峰系数。

    均值率在此面板降级为参考项（键名加 _ref 并保留 mean_rate_hz_ref）。
    """
    r = _rate_arr(spike_counts, window_s)
    n = int(r.size)
    out: Dict[str, float] = rate_percentiles(spike_counts, window_s)
    out.update(effective_dimension(spike_counts, window_s,
                                   overheat_factor, target_rate_hz))
    out["bimodality"] = bimodality_coefficient(spike_counts, window_s)
    out["silence_ratio"] = silence_ratio(spike_counts)
    out["mean_rate_hz_ref"] = float(r.mean()) if n else 0.0   # 参考项
    out["mean_rate_hz_ref_note"] = 0.0   # 占位：键名即标注（参考项非主判据）
    return out


def capture_drift_snapshot(bundle: Any) -> Dict[str, Any]:
    """训练/ADAPT 前权重快照（run_one_seed / runner 复用读接口）。

    Args:
        bundle: NetworkBundle（协议运行前调用）。

    Returns:
        dict：pos（可学习边 CSR 位置）、w0（初始权重拷贝）、src_gids、
        w_in0（输入边子集位置与权重）、e_mask0（池→池 E 源边位置）、
        gini0（within-neuron Gini 初值均值）、per_pool_sum0（in_sum0 拷贝）、
        w_lo（契约权重下界）、pool_learn、pool_is_E。
    """
    cfg = bundle.cfg
    pos = np.flatnonzero(bundle.csr_learn)
    return {
        "pos": pos.copy(),
        "w0": np.asarray(bundle.csr_w[pos], dtype=np.float64).copy(),
        "src_gids": np.asarray(bundle.csr_src[pos], dtype=np.int64).copy(),
        "w_in0": (pos[bundle.csr_src[pos] < cfg.n_in].copy(),
                  np.asarray(
                      bundle.csr_w[pos[bundle.csr_src[pos] < cfg.n_in]],
                      dtype=np.float64).copy()),
        "e_mask0": _e_source_positions(bundle, pos),
        "gini0": gini_within_neuron(bundle),
        "per_pool_sum0": np.asarray(bundle.in_sum0, dtype=np.float64).copy(),
        "w_lo": float(cfg.w_lo),
        "pool_learn": bool(cfg.pool_learn),
        "n_pool": int(cfg.n_pool),
    }


def _e_source_positions(bundle: Any, pos: np.ndarray) -> np.ndarray:
    """pos 中属于池→池 E 源边的 CSR 位置数组。"""
    cfg = bundle.cfg
    src = bundle.csr_src[pos]
    is_pool = src >= cfg.n_in
    e = np.zeros(pos.size, dtype=bool)
    if is_pool.any():
        pool_ids = src[is_pool] - cfg.n_in
        e[is_pool] = bundle.pool_is_E[pool_ids]
    return pos[e]


def read_drift_stats(bundle: Any, snapshot: Mapping[str, Any]) -> Dict[str, float]:
    """协议运行后读取漂移/选择性统计（与 capture_drift_snapshot 配对）。

    Args:
        bundle: NetworkBundle（协议运行后，状态即终态）。
        snapshot: capture_drift_snapshot 输出。

    Returns（全部纯读数，不含断言）：
        drift_frac        |Δw| > 0.1·|w_init| 的可学习边占比（学习度量）
        input_drift_frac  同上但仅输入边（input_w_drift 的分布级主项）
        gini             within-neuron Gini（当前均值）
        gini_ratio       gini / gini0（ADAPT 后/初值比；gini0=0 时为 1.0）
        gini_std         per-neuron Gini 的样本 std（正确 per-neuron std，
                         替代历史 std_ratio 死汇总行的正确实现）
        w_ratio          池→池 E 源权重均值 末/初 比（w_ratio dev 记账）
        input_w_min_sat  输入边 w <= w_lo+1e-12 的占比（R7 触发器）
        budget_dev       劫持度量：逐池可学习入突触 Σw 对 in_sum0 的
                         最大相对偏差（FIX-A 后恒 ≈0）
    """
    cfg = bundle.cfg
    pos = np.asarray(snapshot["pos"], dtype=np.intp)
    w0 = np.asarray(snapshot["w0"], dtype=np.float64)
    w1 = np.asarray(bundle.csr_w[pos], dtype=np.float64)
    dw = np.abs(w1 - w0)
    thr = 0.1 * np.abs(w0)
    drift = float(np.count_nonzero(dw > thr) / max(1, int(w0.size)))
    gini1 = gini_within_neuron(bundle)
    gini0 = float(snapshot.get("gini0", 0.0))
    gini_ratio = float(gini1 / gini0) if gini0 > 0.0 else 1.0
    gini_std = float(per_neuron_gini(bundle).std())
    # 输入边占比（输入→池 学习度量）
    src = np.asarray(snapshot["src_gids"], dtype=np.int64)
    in_m = src < cfg.n_in
    input_drift = (float(np.count_nonzero(dw[in_m] > thr[in_m]) /
                         max(1, int(np.count_nonzero(in_m))))
                   if np.any(in_m) else 0.0)
    w_lo = float(snapshot.get("w_lo", cfg.w_lo))
    in_pos = np.asarray(snapshot["w_in0"][0], dtype=np.intp)
    if in_pos.size:
        w_in = np.asarray(bundle.csr_w[in_pos], dtype=np.float64)
        input_w_min_sat = float(np.count_nonzero(w_in <= w_lo + 1e-12)
                                / float(in_pos.size))
    else:
        input_w_min_sat = 0.0
    # 池→池 E 源 w_ratio（记账统一：冻结/ctrl 行恒 ≈1 = budget_dev 语义）
    e_pos = np.asarray(snapshot.get("e_mask0", np.zeros(0, dtype=np.intp)),
                       dtype=np.intp)
    if e_pos.size and snapshot["w0"].size:
        e0 = w0[np.isin(pos, e_pos)]
        e1 = w1[np.isin(pos, e_pos)]
        w_ratio = float(e1.mean() / e0.mean()) if e0.size and e0.mean() > 0 \
            else 1.0
    else:
        w_ratio = 1.0
    # budget_dev（FIX-A 劫持度量：逐池相对预算偏差）
    ptr = bundle.in_learn_ptr
    idx = bundle.in_learn_idx
    if idx.size:
        pool_ids = np.repeat(np.arange(int(cfg.n_pool), dtype=np.int64),
                             np.diff(ptr))
        s = np.bincount(pool_ids, weights=bundle.csr_w[idx],
                        minlength=int(cfg.n_pool))
        sums0 = np.asarray(snapshot["per_pool_sum0"], dtype=np.float64)
        dev = np.abs(s - sums0) / np.maximum(sums0, 1e-12)
        budget_dev = float(np.max(dev)) if dev.size else 0.0
    else:
        budget_dev = 0.0
    return {
        "drift_frac": drift,
        "input_drift_frac": input_drift,
        "gini": gini1,
        "gini_ratio": gini_ratio,
        "gini_std": gini_std,
        "w_ratio": w_ratio,
        "w_ratio_dev": abs(w_ratio - 1.0),
        "input_w_min_sat": input_w_min_sat,
        "budget_dev": budget_dev,
    }


def assess_r7(bundle: Any, snapshot: Mapping[str, Any],
              sat_threshold: float = 0.20,
              gini_ratio_threshold: float = 1.3) -> Dict[str, Any]:
    """R7 触发器（#4 正式替换旧均值触发器 0.5×初值——FIX-A 归一化钉死下失明）：

    输入突触 w_min 饱和占比 >20%（死数值钉死）或 gini_ratio >1.3 → 警报。

    Returns:
        dict：input_w_min_sat / gini_ratio / alert（bool）/ explain。
    """
    st = read_drift_stats(bundle, snapshot)
    sat = float(st["input_w_min_sat"])
    gr = float(st["gini_ratio"])
    alert = bool(sat > sat_threshold or gr > gini_ratio_threshold)
    reasons = []
    if sat > sat_threshold:
        reasons.append(f"input_w_min_sat={sat * 100:.1f}% > "
                       f"{sat_threshold * 100:.0f}%")
    if gr > gini_ratio_threshold:
        reasons.append(f"gini_ratio={gr:.3f} > {gini_ratio_threshold:.1f}")
    return {
        "input_w_min_sat": sat,
        "gini_ratio": gr,
        "thresholds": {"sat": sat_threshold, "gini_ratio": gini_ratio_threshold},
        "alert": alert,
        "explain": "; ".join(reasons) if reasons else "no R7 trigger",
    }
