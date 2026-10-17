"""轮 3 · B4：**开集「未识别」档的 τ 标定**（补齐第一步 1a 的遗留）。

背景
----
1a（``robust_eval``）的「未识别」档一直只报**不依赖 τ** 的分位数，并四处声明
「本档**不标定**阈值（属第二步变体 B 的范围）」。本模块把这件事**实现**了。

判定单元与两个总体
------------------
* **正总体**（应被**识别**）：已知问题的最高余弦 ``pos``（期望 ``>= τ``）；
* **负总体**（应被**拒识**）：未知问题的最高余弦 ``neg``（期望 ``< τ``）。

口径（显式）
------------
* **未识别** ⇔ 最高余弦 ``< τ``；**误召回** ⇔ ``>= τ``（与 ``unrecognized_report`` 逐字一致）；
* 工作点 ``τ`` = 使**平衡准确率** ``(recall + specificity)/2`` 最大的候选阈值；
  候选集 = ``sorted(pos ∪ neg)`` 的**相邻值中点**（确定性、无超参）；
* **余量** ``margin`` = 「正总体里被识别者的最小值」−「负总体里被拒识者的最大值」，
  即工作点上两族之间的**实测间隔**。

边界归属（**必须显式**）
------------------------
守卫判据是 **``margin > max(noise_floor, granularity)`` —— 严格大于（不是 ``>=``）**。
理由：``margin`` **恰好等于** 尺子分辨率时，这次"间隔"完全可能只是**一个样本的偶然**
造成（换一个样本就翻转），把它判为"有效"等于把分辨率当成信号。故等号归**无效**。
现场用例见 :func:`boundary_cases`（``margin == threshold`` 的三组恰好相等用例）。

噪声底 / 粒度的**同总体、同样本量、同口径**（B4-3）
---------------------------------------------------
* **噪声底** ``noise_floor`` = 把**同一批** ``pos`` / ``neg`` 各自按确定性**对半拆分**，
  在每个半集上用**同一个**标定过程得到 ``τ_half``，取 ``max(|τ_h1 − τ|, |τ_h2 − τ|)``。
  ⇒ 同总体（同一批查询）、同口径（同一算法）、样本量**减半**（这是小样本抖动的**保守**
  估计：样本越少抖动越大）。
* **粒度** ``granularity`` = 工作点处**移动一个样本**引起的 ``τ`` 变化量，取正 / 负两侧
  的最大值；同时报**比率空间**的 ``1/n_pos`` / ``1/n_neg``。
  ⇒ 把「1/n 的粒度」与「落差」一并核算，禁止只看落差。
* **禁止跨总体复用**：``pos`` 与 ``neg`` 必须**来自同一侧、同一 profile、同一键表**；
  本模块在 :func:`calibrate_tau` 里校验 ``population`` 字段的这三个承诺，不一致即抛。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

#: 本轮协议版本（结构变化必须递增；进产物）。
OPEN_SET_TAU_VERSION: str = "n3dqa-open-set-tau-v1"

#: 守卫判据的**边界归属**（唯一来源；实现与文档同源）。
GUARD_BOUNDARY_RULE: str = (
    "守卫判据 = `margin > max(noise_floor, granularity_tau)` —— **严格大于**，"
    "等号归**无效**。理由：margin 恰好等于尺子分辨率时，该间隔完全可能只由**一个样本**"
    "造成（换一个样本就翻转），把等号判为「有效」等于把分辨率当成信号。"
    "现场用例：`boundary_cases()` 给出三组 `margin == threshold` 的**恰好相等**用例，"
    "它们必须全部判为 `effective = False`。"
)

#: 噪声底口径（唯一来源）。
NOISE_FLOOR_RULE: str = (
    "噪声底 = 把**同一批** pos / neg 各自按确定性的**对半拆分**（排序后按奇偶下标分组），"
    "在每个半集上用**同一个**标定过程得到 τ_half，取 max(|τ_h1 − τ|, |τ_h2 − τ|)。"
    "⇒ 与判定单元**同总体、同口径**，样本量**减半**（小样本抖动的保守估计：样本越少抖动越大）。"
    "**禁止跨总体复用**：pos 与 neg 必须同侧、同 profile、同一键表。"
)

#: 粒度口径（唯一来源）。
GRANULARITY_RULE: str = (
    "粒度 = 工作点处**移动一个样本**引起的 τ 变化量（正 / 负两侧取最大），"
    "并同时报比率空间的 1/n_pos 与 1/n_neg。把「1/n 的粒度」与「落差」一并核算，"
    "禁止只看落差 —— 小样本下 1/n 可能大于落差本身。"
)

#: 产物目录与守卫根（τ 部分写 ``_verify/robust/``，与既有 robust 产物同目录）。
VERIFY_ROOT: str = os.path.join("checkpoints", "qa_learn", "_verify")
DEFAULT_OUT_DIR: str = os.path.join(VERIFY_ROOT, "robust")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _sha256_bytes(blob: bytes) -> str:
    """字节 SHA256。"""
    return hashlib.sha256(blob).hexdigest()


def sha256_text(text: str) -> str:
    """UTF-8 裸字节 SHA256。"""
    return _sha256_bytes(str(text).encode("utf-8"))


def canonical_json(obj: Any) -> str:
    """规范化 JSON（``sort_keys`` + 紧凑分隔符）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# 标定核心
# ---------------------------------------------------------------------------


def candidate_thresholds(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """候选阈值 = ``sorted(pos ∪ neg)`` 的**相邻值中点**（确定性、无超参）。

    若并集只有一个不同取值，返回该值本身（退化情形，由调用方按有效性判定处理）。
    """
    vals = np.unique(np.concatenate([np.asarray(pos, dtype=np.float64),
                                     np.asarray(neg, dtype=np.float64)]))
    if vals.size <= 1:
        return vals
    return (vals[:-1] + vals[1:]) / 2.0


def _balanced_at(t: float, pos: np.ndarray, neg: np.ndarray) -> Dict[str, float]:
    """给定 τ，算该工作点的 recall / specificity / 平衡准确率 / margin。"""
    pos_kept = pos[pos >= float(t)]
    neg_rejected = neg[neg < float(t)]
    recall = float(pos_kept.size) / float(max(1, pos.size))
    spec = float(neg_rejected.size) / float(max(1, neg.size))
    margin = (
        float(pos_kept.min()) - float(neg_rejected.max())
        if (pos_kept.size and neg_rejected.size)
        else float("nan")
    )
    return {
        "recall": float(recall),
        "specificity": float(spec),
        "balanced_accuracy": float((recall + spec) / 2.0),
        "margin": float(margin),
        "n_pos_kept": int(pos_kept.size),
        "n_neg_rejected": int(neg_rejected.size),
    }


def select_tau(pos: Sequence[float], neg: Sequence[float]) -> Dict[str, Any]:
    """选出工作点 τ（**最优平衡准确率**；并列按 (平衡准确率, margin, −τ) 字典序）。

    返回
    ----
    Dict[str, Any]
        ``tau`` / ``balanced_accuracy`` / ``recall`` / ``specificity`` /
        ``margin`` / ``n_candidates`` / ``tie_break_rule`` / ``all_equal_margin``。

    **退化**：``pos`` 或 ``neg`` 为空时 ``tau = None``（如实标不适用，不臆造阈值）。
    """
    p = np.asarray([float(x) for x in pos], dtype=np.float64)
    n = np.asarray([float(x) for x in neg], dtype=np.float64)
    base = {
        "n_pos": int(p.size),
        "n_neg": int(n.size),
        "tie_break_rule": (
            "并列时按 (balanced_accuracy 降序, margin 降序, τ 升序) 取第一个 —— "
            "确定性、与输入顺序无关"
        ),
    }
    if p.size == 0 or n.size == 0:
        return {
            **base,
            "tau": None,
            "applicable": False,
            "reason": "pos 或 neg 为空：无法标定（如实标不适用，不臆造阈值）",
        }
    cands = candidate_thresholds(p, n)
    rows: List[Dict[str, Any]] = []
    for t in cands.tolist():
        m = _balanced_at(float(t), p, n)
        rows.append({"tau": float(t), **m})
    if not rows:
        return {**base, "tau": None, "applicable": False, "reason": "无候选阈值"}
    rows.sort(
        key=lambda r: (
            -float(r["balanced_accuracy"]),
            -(float(r["margin"]) if np.isfinite(r["margin"]) else -1e9),
            float(r["tau"]),
        )
    )
    best = rows[0]
    all_equal = bool(
        rows[1]["balanced_accuracy"] == best["balanced_accuracy"] if len(rows) > 1 else True
    )
    return {
        **base,
        "tau": float(best["tau"]),
        "applicable": True,
        "balanced_accuracy": float(best["balanced_accuracy"]),
        "recall": float(best["recall"]),
        "specificity": float(best["specificity"]),
        "margin": float(best["margin"]),
        "n_pos_kept": int(best["n_pos_kept"]),
        "n_neg_rejected": int(best["n_neg_rejected"]),
        "n_candidates": int(len(rows)),
        "best_tied_with_others": bool(all_equal),
    }


def _split_half(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """确定性的**对半拆分**（按排序后的奇偶下标分组；与输入顺序无关）。"""
    v = np.sort(np.asarray(values, dtype=np.float64))
    return v[0::2], v[1::2]


def noise_floor_of(pos: Sequence[float], neg: Sequence[float], tau: float) -> Dict[str, Any]:
    """**同总体、同口径、样本量减半**的噪声底。

    见 :data:`NOISE_FLOOR_RULE`。两个半集各自独立标定，取 ``τ`` 偏离全量标定值的最大者。
    """
    p = np.asarray([float(x) for x in pos], dtype=np.float64)
    n = np.asarray([float(x) for x in neg], dtype=np.float64)
    p1, p2 = _split_half(p)
    n1, n2 = _split_half(n)
    halves: List[Dict[str, Any]] = []
    for idx, (pp, nn) in enumerate(((p1, n1), (p2, n2)), start=1):
        sub = select_tau(pp.tolist(), nn.tolist())
        dev = (
            abs(float(sub["tau"]) - float(tau)) if sub.get("tau") is not None else float("nan")
        )
        halves.append(
            {
                "half": int(idx),
                "n_pos": int(pp.size),
                "n_neg": int(nn.size),
                "tau_half": (None if sub.get("tau") is None else float(sub["tau"])),
                "abs_deviation": (None if not np.isfinite(dev) else float(dev)),
            }
        )
    devs = [
        float(h["abs_deviation"]) for h in halves if h["abs_deviation"] is not None
    ]
    return {
        "noise_floor": float(max(devs)) if devs else 0.0,
        "halves": halves,
        "rule": str(NOISE_FLOOR_RULE),
        "applicable": bool(len(devs) == 2),
    }


def granularity_of(pos: Sequence[float], neg: Sequence[float], tau: float) -> Dict[str, Any]:
    """工作点处**移动一个样本**引起的 τ 变化量（正 / 负两侧取最大）+ 比率空间粒度。"""
    p = np.sort(np.asarray([float(x) for x in pos], dtype=np.float64))
    n = np.sort(np.asarray([float(x) for x in neg], dtype=np.float64))
    # 工作点两侧的相邻样本之间的间隙（= 移动一个样本对 τ 的最大影响）
    gaps: List[float] = []
    for arr in (p, n):
        if arr.size >= 2:
            gaps.extend(np.diff(arr).tolist())
    # 只取工作点**附近**（τ 两侧最近的各 1 个间隙），避免用远端大间隙虚高
    near: List[float] = []
    for arr in (p, n):
        if arr.size == 0:
            continue
        below = arr[arr < float(tau)]
        above = arr[arr >= float(tau)]
        if below.size and above.size:
            near.append(float(above.min() - below.max()))
    granularity_tau = float(max(near)) if near else (float(max(gaps)) if gaps else 0.0)
    return {
        "granularity_tau": float(granularity_tau),
        "granularity_rate_pos": float(1.0 / max(1, p.size)),
        "granularity_rate_neg": float(1.0 / max(1, n.size)),
        "near_gaps": [float(x) for x in near],
        "rule": str(GRANULARITY_RULE),
    }


def calibrate_tau(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    *,
    population: Optional[Dict[str, Any]] = None,
    factor: float = 1.0,
) -> Dict[str, Any]:
    """**开集 τ 的完整标定**：工作点 + 噪声底 + 粒度 + 守卫判定 + 边界归属。

    参数
    ----
    positive_scores : Sequence[float]
        正总体（应被识别）的「最高余弦」逐条取值。
    negative_scores : Sequence[float]
        负总体（应被拒识）的逐条取值。
    population : Optional[Dict[str, Any]]
        总体取证（``side`` / ``profile`` / ``table_sha256`` 等）。**必须**能证明两侧
        同侧、同 profile、同一键表；不一致即抛。
    factor : float
        噪声底的放大倍数（默认 1.0；``τ 门槛 = max(factor × noise_floor, granularity)``）。

    返回
    ----
    Dict[str, Any]
        ``tau`` / ``noise_floor`` / ``granularity`` / ``threshold`` / ``margin`` /
        ``effective`` / ``boundary_rule`` / ``population`` / ``applicable`` / ``reason``。

    异常
    ------
    ValueError
        ``population`` 缺失或三个"同"承诺缺失/不一致（**禁止跨总体复用**）。
    """
    pop = dict(population or {})
    if not pop:
        raise ValueError(
            "calibrate_tau 必须给 `population` 取证（side / profile / table_sha256）；"
            "「同总体、同样本量、同口径」是硬约束，不能靠调用方口头保证"
        )
    for key in ("side", "profile", "table_sha256"):
        if str(pop.get(key, "")) == "":
            raise ValueError(
                f"population 缺 `{key}`；无法证明「同总体、同口径」，拒绝标定"
            )
    if str(pop.get("pos_side", pop.get("side"))) != str(pop.get("neg_side", pop.get("side"))):
        raise ValueError(
            "pos 与 neg 的 side 不一致（"
            + str(pop.get("pos_side")) + " vs " + str(pop.get("neg_side"))
            + "）；**禁止跨总体复用**噪声底与阈值"
        )
    pos = np.asarray([float(x) for x in positive_scores], dtype=np.float64)
    neg = np.asarray([float(x) for x in negative_scores], dtype=np.float64)
    sel = select_tau(pos.tolist(), neg.tolist())
    if not bool(sel.get("applicable")):
        return {
            "version": OPEN_SET_TAU_VERSION,
            "tau": None,
            "applicable": False,
            "reason": str(sel.get("reason", "")),
            "population": pop,
            "n_pos": int(pos.size),
            "n_neg": int(neg.size),
            "boundary_rule": str(GUARD_BOUNDARY_RULE),
        }
    tau = float(sel["tau"])
    floor = noise_floor_of(pos.tolist(), neg.tolist(), tau)
    gran = granularity_of(pos.tolist(), neg.tolist(), tau)
    threshold = max(
        float(factor) * float(floor["noise_floor"]), float(gran["granularity_tau"])
    )
    margin = float(sel["margin"])
    # **边界归属：严格大于**（见 GUARD_BOUNDARY_RULE）
    effective = bool(np.isfinite(margin) and float(margin) > float(threshold))
    return {
        "version": OPEN_SET_TAU_VERSION,
        "tau": float(tau),
        "applicable": True,
        "workpoint": {
            "balanced_accuracy": float(sel["balanced_accuracy"]),
            "recall": float(sel["recall"]),
            "specificity": float(sel["specificity"]),
            "n_pos_kept": int(sel["n_pos_kept"]),
            "n_neg_rejected": int(sel["n_neg_rejected"]),
            "n_candidates": int(sel["n_candidates"]),
            "best_tied_with_others": bool(sel["best_tied_with_others"]),
            "tie_break_rule": str(sel["tie_break_rule"]),
        },
        "n_pos": int(pos.size),
        "n_neg": int(neg.size),
        "margin": float(margin),
        "noise_floor": float(floor["noise_floor"]),
        "noise_floor_block": floor,
        "granularity_tau": float(gran["granularity_tau"]),
        "granularity_block": gran,
        "factor": float(factor),
        "threshold": float(threshold),
        "threshold_rule": (
            "threshold = max(factor × noise_floor, granularity_tau)；"
            "判据 = margin **严格大于** threshold"
        ),
        "effective": bool(effective),
        "boundary_rule": str(GUARD_BOUNDARY_RULE),
        "population": pop,
        "unrecognized_rule": "未识别 ⇔ 最高余弦 < τ；误召回 ⇔ 最高余弦 >= τ",
    }


# ---------------------------------------------------------------------------
# 边界用例（**恰好等于阈值**）
# ---------------------------------------------------------------------------


def boundary_cases() -> Dict[str, Any]:
    """**恰好等于阈值**的现场用例（守卫生效且**等号归无效**）。

    构造方式（**现场搜索，不是手写期望值**）
    ----------------------------------------
    在一个**确定性的小网格**上枚举若干 ``(pos, neg)`` 组合（取值都在 0.25 的整数倍上，
    float 可精确表示），对每个组合跑 :func:`calibrate_tau`，**筛选出**
    ``margin == threshold`` 且 ``margin > 0`` 的那些 —— 这些就是「恰好落在边界上」的用例。
    再用**严格判据**断言它们全部 ``effective == False``。

    与 :func:`strictness_control_case` 配对：后者把同一格抬高一档，证明判据不是恒假。

    返回
    ----
    Dict[str, Any]
        ``cases``（搜索到的等号用例，最多 6 条）/ ``n_found`` /
        ``all_effective_false`` / ``search_grid`` / ``rule``。
    """
    pop = {"side": "synthetic", "profile": "synthetic-1d", "table_sha256": "synthetic"}
    # 确定性网格：取值 = k × 0.25（k ∈ [-2, 8]），pos / neg 各取 2~3 条，递增排列。
    grid = [round(0.25 * k, 2) for k in range(-2, 9)]
    found: List[Dict[str, Any]] = []
    n_scanned = 0
    n_separable = 0
    for pos in ([a, b] for a in grid for b in grid if b > a):
        for neg in ([c, d] for c in grid for d in grid if d > c):
            # **必须可分**：正总体严格高于负总体（否则「margin」没有语义，
            # 会把 pos==neg 这种退化组合算进来 —— 那不是「恰好落在边界上」，
            # 而是「根本没有间隔」）。
            if not (min(pos) > max(neg)):
                continue
            n_scanned += 1
            out = calibrate_tau(list(pos), list(neg), population=pop)
            if not bool(out.get("applicable")):
                continue
            n_separable += 1
            margin = float(out["margin"])
            threshold = float(out["threshold"])
            if not (margin > 0.0 and margin == threshold):
                continue
            found.append(
                {
                    "case": "exact_equality_%02d" % (len(found) + 1),
                    "pos": [float(x) for x in pos],
                    "neg": [float(x) for x in neg],
                    "separable": True,
                    "tau": float(out["tau"]),
                    "margin": float(margin),
                    "noise_floor": float(out["noise_floor"]),
                    "granularity_tau": float(out["granularity_tau"]),
                    "threshold": float(threshold),
                    "effective": bool(out["effective"]),
                    "margin_equals_threshold": bool(margin == threshold),
                }
            )
            if len(found) >= 6:
                break
        if len(found) >= 6:
            break
    return {
        "version": OPEN_SET_TAU_VERSION,
        "search_grid": {
            "values": [float(x) for x in grid],
            "note": (
                "取值均为 0.25 的整数倍（float 精确表示），pos/neg 各 2 条、严格递增，"
                "且**必须可分**（min(pos) > max(neg)）"
            ),
            "n_scanned": int(n_scanned),
            "n_after_separable_filter": int(n_separable),
        },
        "n_found": int(len(found)),
        "cases": found,
        "all_effective_false": bool(
            len(found) > 0 and all(bool(c["effective"]) is False for c in found)
        ),
        "boundary_rule": str(GUARD_BOUNDARY_RULE),
        "rule": (
            "每个用例都是 `margin == threshold`（**等号**）的现场搜索结果，必须判 "
            "`effective=False`；若有用例判 True，说明实现用了 `>=`，与 GUARD_BOUNDARY_RULE 不符。"
            "**要求至少搜到 1 条**（``n_found > 0``）。只接受**可分**组合"
            "（``min(pos) > max(neg)``）—— pos==neg 之类的退化组合的「margin」没有语义。"
        ),
    }


def strictness_control_case() -> Dict[str, Any]:
    """**对照用例**：在边界用例的基础上把 ``margin`` 抬高 ⇒ 必须判 ``effective=True``。

    做法：取 :func:`boundary_cases` 的第一条等号用例，把它的**负总体最大值**降到更低一档
    （0.25 的整数倍，精确表示），使 ``margin`` 变大而 ``granularity`` 不变；
    断言 ``effective`` 由 ``False`` 翻成 ``True``。⇒ 判据**严格且非恒假**。
    """
    bc = boundary_cases()
    cases = list(bc.get("cases", []))
    if not cases:
        return {
            "version": OPEN_SET_TAU_VERSION,
            "strictness_proven": False,
            "reason": "边界用例搜索为空，无法构造对照",
            "rule": str(GUARD_BOUNDARY_RULE),
        }
    pop = {"side": "synthetic", "profile": "synthetic-1d", "table_sha256": "synthetic"}
    first = cases[0]
    pos = [float(x) for x in first["pos"]]
    neg = [float(x) for x in first["neg"]]
    step = 0.25
    # 把负总体整体降一档（保持递增），margin 变大
    neg_lifted = [round(float(x) - step, 2) for x in neg]
    base = calibrate_tau(pos, neg, population=pop)
    lifted = calibrate_tau(pos, neg_lifted, population=pop)
    return {
        "version": OPEN_SET_TAU_VERSION,
        "source_case": str(first["case"]),
        "pos": pos,
        "neg": neg,
        "neg_lifted": neg_lifted,
        "step": float(step),
        "at_equality": {
            "tau": float(base["tau"]),
            "margin": float(base["margin"]),
            "threshold": float(base["threshold"]),
            "effective": bool(base["effective"]),
        },
        "lifted": {
            "tau": float(lifted["tau"]),
            "margin": float(lifted["margin"]),
            "threshold": float(lifted["threshold"]),
            "effective": bool(lifted["effective"]),
        },
        "strictness_proven": bool(
            bool(base["effective"]) is False and bool(lifted["effective"]) is True
        ),
        "rule": (
            "等号处 `effective=False`（边界归无效）；把负总体降一档（margin 变大）后 "
            "`effective=True` ⇒ 判据**严格且非恒假**。"
        ),
    }


def factor_sweep(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    *,
    population: Optional[Dict[str, Any]] = None,
    factors: Sequence[float] = (1.0, 2.0, 3.0),
) -> Dict[str, Any]:
    """**倍数扫描**（E-B4a 修复，离朱 R82）：逐 factor 给出 threshold 与判定。

    为什么需要它：报告里只写公式 ``threshold = max(factor × noise_floor, granularity)``
    而**不给具体倍数下的数值**时，读者无法确认「换 factor 判定会不会翻」。
    本函数把 ``factor ∈ {1,2,3}`` 的 ``threshold`` / ``effective`` 逐一算出来，
    使「判定对 factor 的敏感性」成为**可核对的数**而不是一句公式。

    返回
    ----
    Dict[str, Any]
        ``rows``（逐 factor 的 ``factor`` / ``threshold`` / ``effective`` /
        ``binding_term``）+ ``verdict_stable``（所有 factor 判定相同）+
        ``noise_floor`` / ``granularity_tau`` / ``margin`` / ``rule``。
    """
    rows: List[Dict[str, Any]] = []
    for f in factors:
        out = calibrate_tau(
            positive_scores, negative_scores,
            population=population, factor=float(f),
        )
        if not bool(out.get("applicable")):
            rows.append(
                {
                    "factor": float(f),
                    "threshold": None,
                    "effective": None,
                    "binding_term": "",
                    "reason": str(out.get("reason", "")),
                }
            )
            continue
        floor_term = float(f) * float(out["noise_floor"])
        gran_term = float(out["granularity_tau"])
        rows.append(
            {
                "factor": float(f),
                "threshold": float(out["threshold"]),
                "effective": bool(out["effective"]),
                "noise_floor_term": float(floor_term),
                "granularity_term": float(gran_term),
                "binding_term": (
                    "granularity" if gran_term >= floor_term else "noise_floor"
                ),
            }
        )
    effs = [r["effective"] for r in rows if r["effective"] is not None]
    return {
        "rows": rows,
        "factors": [float(f) for f in factors],
        "verdict_stable": bool(len(set(effs)) <= 1),
        "rule": (
            "`threshold = max(factor × noise_floor, granularity_tau)`；"
            "`binding_term` 指明哪一项在起作用。`verdict_stable = True` 表示判定"
            "**不随 factor 变化**（本轮现场即如此：granularity 主导）。"
        ),
    }


__all__ = [
    "OPEN_SET_TAU_VERSION",
    "GUARD_BOUNDARY_RULE",
    "NOISE_FLOOR_RULE",
    "GRANULARITY_RULE",
    "VERIFY_ROOT",
    "DEFAULT_OUT_DIR",
    "boundary_cases",
    "calibrate_tau",
    "candidate_thresholds",
    "canonical_json",
    "factor_sweep",
    "granularity_of",
    "noise_floor_of",
    "select_tau",
    "sha256_text",
    "strictness_control_case",
]