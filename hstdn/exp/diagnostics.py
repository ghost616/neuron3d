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
