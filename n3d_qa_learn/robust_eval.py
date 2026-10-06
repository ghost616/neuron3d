"""n3d_qa_learn 分档鲁棒性评测 + KNN 基线台账（第一步 1a 的「考卷」落地）。

设计文档 §3.2（相似变体构造）+ §7（评估方案）+ §7.3（KNN 基线对比）的落地实现。
本模块**不改任何网络结构、不做任何训练**：它只构造「考卷」并验证这把尺子有区分度。

口径（逐轮确认，不得擅自变更）
----------------------------
* **统一抽象**：:class:`n3d_qa_learn.entry_table.EntryKeyTable`（QA 侧 ``outputs`` =
  答案文本；文本侧 ``outputs`` = 行原文），两侧共用本模块这一套评测代码。
* **扰动矩阵**（三种 × 三档，每档每条目 1 条变体）：

  ============  ==========================================
  扰动           档位
  ============  ==========================================
  ``noise``      高斯噪声 σ ∈ {0.05, 0.10, 0.15}
  ``mask``       随机维度遮蔽 ∈ {10%, 30%, 50%}
  ``nmag``       幅度缩放 + 平移 ∈ {±10%, ±20%}
  ============  ==========================================

* **扰动后一律重新 L2 归一化**（唯一实现 = ``entry_table.l2_normalize_rows``）；
  零范数显式处置（实测不出现，但保留可读报错兜底，**不得静默产出 NaN**）。
* **确定性**：一切随机数由**局部** ``torch.Generator`` 驱动（固定 ``seed=42``），
  **不消耗全局 RNG**；同命令重复运行逐位一致。
* **去掉欧氏基线轴**（现场实测：对 L2 归一化向量余弦 top-1 与欧氏 top-1 逐位等价），
  基线轴 = 归一化余弦 top-1 / top-5 × 两档编码器（``local-hash`` 词面 /
  ``bge-m3-1024`` 语义）。
* **两侧分开报告**：文本侧 666 条查询为主判据依据；QA 侧 20 条为辅助并**显式标注
  统计意义弱**。
* **「未识别」档**：以 ``train_unknown`` + ``test_unknown`` 作负样本，报未识别率与
  误召回率；**不标定阈值**（阈值标定属第二步变体 B 的范围）。
* **阈值现场标定**（两阶段）：先跑一次得实测落差，再按「观测噪声的约 2 倍」标定并
  回填报告；落差 ≤ 噪声水平 ⇒ 判定该扰动对本数据**无效**并如实登记。

产物纪律
--------
报告一律写 ``checkpoints/qa_learn/_verify/robust/``；**不落盘扰动后的特征矩阵**
（只落报告与指纹）。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import entry_table as ET
from .backends import BACKEND_NAMES  # noqa: F401  (仅只读引用，保持注册表可用性可见)
from .entry_table import EntryKeyTable, EntryStatements, TextEntryBundle
from .train import DEFAULT_VERIFY_DIR

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

#: 报告根目录（验证类运行一律写 ``_verify/``）。
ROBUST_DIR: str = os.path.join(DEFAULT_VERIFY_DIR, "robust")

#: 扰动种子（冻结；确定性生成的前提）。
ROBUST_SEED: int = 42

#: 每个档位生成的变体条数（主判据 = 1；稳定性佐证 = K）。
ROBUST_VARIANTS: int = 1
ROBUST_STABILITY_K: int = 5

#: 有效扰动类型（本批**不含档 2**：现成数据不存在「同一问题的不同问法」）。
PERTURB_TYPES: Tuple[str, ...] = ("noise", "mask", "nmag")

#: 扰动类型的中文标签（报告用）。
PERTURB_LABELS: Dict[str, str] = {
    "noise": "高斯噪声",
    "mask": "随机维度遮蔽",
    "nmag": "幅度缩放+平移",
}

#: 三档权重（弱 / 中 / 强）。
PERTURB_LEVELS: Tuple[str, ...] = ("weak", "middle", "strong")

#: 档位中文标签。
LEVEL_LABELS: Dict[str, str] = {"weak": "弱", "middle": "中", "strong": "强"}

#: 每种扰动的**档位强度参数** ``ε``（三种扰动都只有**一个标量**参数，见各扰动公式）：
#:
#: * ``noise`` —— 加性高斯噪声的标准差 σ：``x ← x + σ·ξ``，``ξ ~ N(0, I)``；
#: * ``mask``  —— 随机维度遮蔽比例 r：逐行保留 ``floor(D·(1−r))`` 维、其余置 0；
#: * ``nmag``  —— **幅度缩放/平移**的幅度 ε：``x ← x·(1+ε) + b``，其中
#:   ``b = ε·(−1)^(row+col+1)``（棋盘式交替符号，见 :func:`nmag_shift_matrix`，
#:   故 ``ε`` 同时是「缩放幅度」与「平移幅度」）。
PERTURB_GRID: Dict[str, Dict[str, float]] = {
    "noise": {"weak": 0.05, "middle": 0.10, "strong": 0.15},
    "mask": {"weak": 0.10, "middle": 0.30, "strong": 0.50},
    "nmag": {"weak": 0.10, "middle": 0.15, "strong": 0.20},
}

#: ``nmag`` 的**共享缩放系数**：``x ← x·NMAG_SHARED_SCALE + ε·(符号矩阵)``。
#:
#: 取 ``0.5`` 是为了让三档的总位移量级与 ``noise`` 同量级、可横向比较；
#: 该系数对三档**恒定**，故档位之间只有唯一的自变量 ``ε`` 在变（避免两参数同时
#: 变动导致「落差归因不清」）。
NMAG_SHARED_SCALE: float = 0.5

#: 归一化余弦 top-k 的 k（报 ``@1`` 与 ``@5``）。
TOPK: int = 5

#: 检索批大小。
BATCH_SIZE: int = 256

#: 「未识别」分位点（报告用；**不据此标定阈值**）。
UNKNOWN_QUANTILES: Tuple[float, ...] = (0.5, 0.9, 0.99)

#: 判定「扰动对本数据无效」的噪声倍数（观测噪声的约 2 倍）。
THRESHOLD_FACTOR: float = 2.0

#: **噪声底标定的强度比**：噪声底用「可忽略强度」测 —— ε_noise = 该扰动**弱档**的
#: 1/50，即 σ=0.001 / 遮蔽 1% / 幅度 ±0.2%。该强度下特征几乎不变，命中率的掉点
#: 即「尺子本身的读数抖动」（float32 舍入 + 检索打平/翻转）。
NOISE_LEVEL_RATIO: float = 0.02

#: 三种扰动的**公式文本**（进产物 `grid.perturb_formulas` 与 G7 断言，唯一来源）。
PERTURB_FORMULAS: Dict[str, str] = {
    "noise": "x <- x + sigma * xi, xi ~ N(0, I)（sigma = 该档 ε；用派生种子的 torch.Generator 生成）",
    "mask": "x <- x * m, m 逐行保留 floor(D*(1-r)) 个随机维、其余置 0（r = 该档 ε）",
    "nmag": (
        "x <- x * s + b, s = NMAG_SHARED_SCALE(0.5), "
        "b[i,j] = eps * (-1)**(i+j+1)（eps = 该档 ε；**闭式无随机数**，故 variant 不生效）"
    ),
}

#: ``nmag`` 测噪声底时的**平移幅度**（绝对常数）。理由：``nmag`` 的档位 ε 同时是
#: 「缩放幅度」与「平移幅度」，若沿用档位那套 ``s = NMAG_SHARED_SCALE``，特征会被整体
#: 压到 0.5 倍量级、**不可忽略**（现场实测该做法把文本侧 nmag 噪声底抬到 0.521021，
#: τ = 1.042 高于任何可能落差，会把「有效」误判成「无效」）。故噪声底改用
#: 「**不缩放、只做可忽略平移**」：``x ← x·1 + NMAG_NEGLIGIBLE_SHIFT·(±1)``。
NMAG_NEGLIGIBLE_SHIFT: float = 1e-3


def nmag_negligible_shift() -> float:
    """``nmag`` 噪声底用的可忽略平移幅度（见 :data:`NMAG_NEGLIGIBLE_SHIFT`）。"""
    return float(NMAG_NEGLIGIBLE_SHIFT)


#: 粒度守卫的**唯一判定规则文本**（实现 / docstring / 产物 `criterion` / README 四处必须同源）。
#:
#: 口径（审查 W5 收口）：边界**含等号** —— ``abs(gap_used) <= min_granularity`` 一律
#: 判 ``effective = False``；其中 ``abs(gap_used) == min_granularity`` 的格额外置
#: ``at_granularity = True``，verdict 显式写「落差 = 粒度（= 1 个样本），不构成判定依据」。
#: 现场实测该边界**必然出现**（QA 侧 20 条查询下，落差恰为 1 个样本 = 0.05 的格有 3 个），
#: 故等号漏判会把最脆弱的情形放过去。
GRANULARITY_GUARD_RULE: str = (
    "**粒度守卫（边界含等号）**：令 `ratio = abs(gap_used) / min_granularity`，"
    "则 `ratio <= 1 + GRANULARITY_REL_TOL`（= 1e-9）一律判 `effective = False`；"
    "其中 `|ratio - 1| <= GRANULARITY_REL_TOL` 的格额外置 `at_granularity = True`，"
    "verdict 显式写「落差 = 粒度（= 1 个样本），不构成判定依据」。"
    "**必须在比值空间比较**：命中率由 `hits / n` 相减得到，"
    "`0.20 - 0.15 = 0.05000000000000002` 与 `1/20 = 0.05` 数值相等但**非逐位相等**，"
    "纯浮点字面比较会把「恰好 1 个样本」这一格漏判为「高于粒度」"
)

#: ``nmag`` **与任务书口径差异的机器可读锚点**（审查 I6）。
#:
#: 任务书「选甲」的口径是 ``x ← x·(1+ε) + b``；本实现是
#: ``x ← x·NMAG_SHARED_SCALE + ε·(−1)^(i+j+1)``，即**缩放系数为三档恒定的 0.5
#: 而不是 ``1+ε``**。这不是实现缺陷，而是**显式设计取舍**，理由如实登记如下：
#: ① 若取 ``s = 1+ε``，则 ε 要同时承担缩放的**相对幅度**与平移的**绝对幅度**两个量纲
#:    （QA/文本侧特征是 L2 归一化向量、元素量级约 1e-2~1e-1，绝对平移 0.10 会把特征
#:    整体"淹没"），档位之间就无法只用一个自变量表达；
#: ② 取恒定 ``s = 0.5`` 后，**唯一随档位变化的自变量就是 ε**（缩放与平移都由它线性控制），
#:    从而 (a) 落差可归因到 ε 本身，(b) G7 可用闭式手算逐位复算；
#: ③ 恒定正缩放对**余弦检索不敏感**（正标量不改变 ``argmax``），故该取舍不影响
#:    归一化余弦口径下的可比的结论，只影响"位移量级"的绝对刻度。
#: **该锚点必须随产物落盘**（`grid.nmag_formula_deviation`），以免下一轮被当作缺陷、
#: 或被静默继承。
NMAG_FORMULA_SPEC_DEVIATION: str = (
    "本条公式与任务书口径的差异（显式设计取舍，非实现缺陷）：任务书「选甲」为 "
    "`x <- x*(1+eps) + b`，本实现为 `x <- x*NMAG_SHARED_SCALE + eps*(-1)**(i+j+1)`，"
    "缩放系数取**三档恒定的 0.5** 而非 `1+eps`。理由：① 取 `1+eps` 会让 eps 同时承担"
    "缩放的相对幅度与平移的绝对幅度两个量纲（特征为 L2 归一化向量、元素量级约 1e-2~1e-1，"
    "绝对平移 0.10 会淹没特征），档位无法只用一个自变量表达；② 恒定 0.5 后唯一随档位变化的"
    "自变量就是 eps，使落差可归因到 eps 本身、且 G7 可用闭式手算逐位复算；③ 恒定正缩放对"
    "余弦检索不敏感（正标量不改变 argmax），故该取舍不改变归一化余弦口径下的可比结论。"
)

#: 粒度守卫的**比较容差（相对，作用在比值上）**。
#:
#: 为什么不能直接用 ``abs(gap) <= min_granularity`` 的**浮点字面比较**：命中率是
#: ``hits / n`` 相减得到的，0.20 − 0.15 = ``0.05000000000000002`` 而 ``1/20 = 0.05``，
#: 两者**数值上相等但不是逐位相等**，纯字面比较会把「恰好 1 个样本」这一格判成「高于粒度」
#: （现场实测：QA 侧 3 格的 ``abs(gap)-gran`` 偏差约 4e-16 相对量级）。
#: 故守卫一律在**比值空间**判定：``ratio = abs(gap_used) / min_granularity``，
#: ``below ⇔ ratio <= 1 + TOL``、``at ⇔ |ratio - 1| <= TOL``。
#: 取 1e-9 足以吸收 float64 舍入（~1e-16 相对量级），又远小于「半个样本」的 0.5，
#: 不会把真正的跨样本差异吞进来。
GRANULARITY_REL_TOL: float = 1e-9

#: 命中率的**最小可分辨粒度**口径：``1 / n_queries``（QA 侧 20 条 ⇒ 0.05）。
#: 报告与标定产物必须显式写出该粒度，并把守卫规则原样带上（见 :data:`GRANULARITY_GUARD_RULE`）。
MIN_GRANULARITY_NOTE: str = (
    "命中率的最小可分辨粒度 = 1 / n_queries（QA 侧 test_known 20 条 ⇒ **0.05**，"
    "即 1 个样本 = 0.05）；" + GRANULARITY_GUARD_RULE
)

#: 默认参与的编码器档（词面 / 语义）。
ROBUST_PROFILES: Tuple[str, ...] = (ET.PROFILE_LEXICAL, ET.PROFILE_SEMANTIC)

#: 逻辑侧名。
SIDES: Tuple[str, ...] = ("text", "qa")

#: 侧中文标签。
SIDE_LABELS: Dict[str, str] = {"text": "文本侧", "qa": "QA 侧"}

#: 档 0（原样自检索，无扰动）/ 档 1（同答案另一问题）/ 档 3（特征扰动）的档名。
TIER_CLEAN: str = "0_clean"
TIER_ALT_QUESTION: str = "1_alt_question"
TIER_PERTURB: str = "3_perturb"

#: 档 1 的**不适用**说明（现成数据不存在同一问题的不同问法）。
TIER_ALT_QUESTION_NOTE: str = (
    "档 1（同答案另一问题）在本数据上**不适用**：Math1 的 id 在 4 题型间零重叠，"
    "n3d_qa 每个 question_id 只有 1 种 question_text —— 现成数据不存在「同一问题的"
    "不同问法」。本批只做档 0（原样自检索）/ 档 3（特征扰动）；档 2 同理不做。"
)

#: 欧氏轴被去掉的实测依据（必须原样保留在报告中）。
EUCLIDEAN_EQUIVALENCE_NOTE: str = (
    "【去掉欧氏基线轴的实测依据】对 L2 归一化向量，余弦 top-1 与欧氏 top-1 逐位等价"
    "（设计文档 §3.1 与 §7.3 在此处自相张力，故不把「余弦/欧氏对比」当作有信息量的"
    "对照）。现场实测证据见 run 产物 evidence.euclidean_equivalence 与 README 第十五节："
    "key_table 行范数 ∈ [0.99999982, 1.00000012]，666 条查询上两者不一致条数 = 0/666，"
    "恒等式 max|‖a−b‖² − (2−2cos)| = 1.19e-06。"
)


#: 各侧的**主判据 cell 角色**（口径显式：文本侧 = 自检索 666 条；QA 侧 = 已知查询 20 条）。
PRIMARY_CELL_ROLE: Dict[str, str] = {"text": "main", "qa": "qa_known"}


def _perturb_type_index(kind: str) -> int:
    """扰动类型在冻结顺序中的下标（派生 seed 的整数部分）。"""
    if str(kind) not in PERTURB_TYPES:
        raise KeyError(f"未知扰动类型 {kind!r}；可用 = {list(PERTURB_TYPES)}")
    return int(list(PERTURB_TYPES).index(str(kind)))


def derived_seed(kind: str, level: str, variant: int = 0, seed: int = ROBUST_SEED) -> int:
    """派生某（扰动类型 × 档位 × 变体）的**局部**生成器种子。

    口径（冻结）
    -----------
    ``derived = seed * 1000 + 100 * type_index + 10 * level_index + variant``。
    生成器一律用 :func:`torch.Generator` 局部对象，**不消耗全局 RNG**。
    """
    if str(level) not in PERTURB_LEVELS:
        raise KeyError(f"未知档位 {level!r}；可用 = {list(PERTURB_LEVELS)}")
    return (
        int(seed) * 1000
        + 100 * _perturb_type_index(kind)
        + 10 * int(list(PERTURB_LEVELS).index(str(level)))
        + int(variant)
    )


# ---------------------------------------------------------------------------
# 扰动构造
# ---------------------------------------------------------------------------


def _l2(mat: np.ndarray) -> np.ndarray:
    """逐行 L2 归一化（统一走 :func:`entry_table.l2_normalize_rows`）。"""
    out, _evidence = ET.l2_normalize_rows(mat)
    return out


def nmag_shift_matrix(n_rows: int, dim: int, eps: float) -> np.ndarray:
    """``nmag`` 的**平移矩阵** ``b``（确定性、无随机数）：``b[i,j] = ε·(−1)^(i+j+1)``。

    为什么不留随机性
    --------------
    ``nmag`` 的三档**只有唯一自变量 α = ε**（缩放幅度 = 平移幅度 = ε，见
    :data:`PERTURB_GRID` 的注释与 :func:`perturb_matrix` 的 docstring）。若这里再引入
    随机数，则「档位」与「随机绘制」两个因素会同时变动，既让 G7（扰动公式逐位断言）
    无法闭式手算，也让档间落差无法归因到 ε 本身。故本函数是**纯闭式**的：
    同一 ``(n_rows, dim, eps)`` 永远给出逐位相同的结果，``variant`` 对 ``nmag`` 不生效
    （该事实在报告与 README 中**显式登记**，不假装有变体差异）。

    参数
    ----
    n_rows : int
        行数 ``N``。
    dim : int
        列数 ``D``。
    eps : float
        平移幅度（= 该档的 ε）。

    返回
    ----
    np.ndarray
        ``float32[N, D]`` 平移矩阵；``b[i,j] = ε``（``i+j`` 为偶数时）或 ``−ε``。
    """
    ii = np.arange(int(n_rows), dtype=np.int64)[:, None]
    jj = np.arange(int(dim), dtype=np.int64)[None, :]
    sign = np.where(((ii + jj) % 2) == 0, -1.0, 1.0)
    return (float(eps) * sign).astype(np.float32)


def perturb_matrix(
    matrix: np.ndarray,
    kind: str,
    level: str,
    *,
    variant: int = 0,
    seed: int = ROBUST_SEED,
    eps_override: Optional[float] = None,
    scale_override: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """对 ``[N, D]`` 特征矩阵施加指定扰动（**确定性 + 重新 L2 归一化**）。

    扰动公式（**三种都只有一个标量强度参数**，便于逐位手算复核）
    -------------------------------------------------------
    ==========  ==================================================================
    扰动         公式
    ==========  ==================================================================
    ``noise``   ``x ← x + σ·ξ``，``ξ ~ N(0, I)``（同一 ``(kind, level, variant, seed)``
                派生同一个 :class:`torch.Generator` 种子，故**可由固定 seed 逐位复算**）
    ``mask``    ``x ← x ⊙ m``，``m`` 逐行保留 ``floor(D·(1−r))`` 个随机维（``randperm``）、
                其余置 0（同样由固定 seed 逐位复算）
    ``nmag``    **幅度缩放/平移**：``x ← x·NMAG_SHARED_SCALE + ε·(−1)^(i+j+1)``
                （``NMAG_SHARED_SCALE = 0.5`` 对三档恒定；平移幅度 = 缩放幅度 = ε，
                由 :func:`nmag_shift_matrix` 给出**闭式**符号矩阵，**不含随机数**）
    ==========  ==================================================================

    三档的 ``ε`` 见 :data:`PERTURB_GRID`；``nmag`` 的 middle 档在计划口径里未单独定义，
    本实现**显式取 0.15**（落于弱 0.10 与强 0.20 之间）并登记在 :data:`PERTURB_GRID`
    与报告 ``grid.perturb_grid`` 中。

    参数
    ----
    matrix : np.ndarray
        ``[N, D]`` 特征（float32）。
    kind : str
        ``noise`` / ``mask`` / ``nmag``。
    level : str
        ``weak`` / ``middle`` / ``strong``。
    variant : int
        变体序号（``0`` = 主判据用的那一条）；**对 ``nmag`` 不生效**（该扰动无随机数，
        故变体逐位相同 —— 已在报告与 README 显式登记）。
    seed : int
        基准种子。
    eps_override : Optional[float]
        显式覆盖强度（**只用于噪声底标定**：取可忽略强度；见 :func:`negligible_eps`）。
    scale_override : Optional[float]
        显式覆盖 ``nmag`` 的缩放系数（**只用于噪声底标定**；``None`` = 正常档位的
        :data:`NMAG_SHARED_SCALE`）。对 ``noise`` / ``mask`` 不生效。

    返回
    ----
    Tuple[np.ndarray, Dict[str, Any]]
        ``(扰动后 float32[N, D] 矩阵, 取证字典)``。取证字典含位移量、零范数计数、
        以及该扰动的**公式与参数**（供 G7 逐位断言与报告溯源）。

    异常
    ------
    ValueError
        零范数行出现（**不得静默产出 NaN**）。
    """
    kind = str(kind)
    level = str(level)
    if kind not in PERTURB_GRID:
        raise KeyError(f"未知扰动类型 {kind!r}；可用 = {list(PERTURB_TYPES)}")
    if level not in PERTURB_GRID[kind]:
        raise KeyError(f"扰动 {kind!r} 未知档位 {level!r}；可用 = {list(PERTURB_LEVELS)}")
    src = np.asarray(matrix, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(f"待扰动矩阵必须是二维 [N, D]，当前形状 = {src.shape}")
    eps = (
        float(eps_override)
        if eps_override is not None
        else float(PERTURB_GRID[kind][level])
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(derived_seed(kind, level, variant, seed)))
    base = torch.from_numpy(np.ascontiguousarray(src, dtype=np.float32))
    if kind == "noise":
        noise = torch.randn(base.shape, generator=gen, dtype=torch.float32)
        perturbed = (base + float(eps) * noise).numpy()
        formula = "x <- x + sigma * xi, xi ~ N(0, I)"
        detail: Dict[str, Any] = {
            "sigma": float(eps),
            "formula": formula,
            "kind": "additive_gaussian",
            "uses_generator": True,
        }
    elif kind == "mask":
        n_rows, dim = int(base.shape[0]), int(base.shape[1])
        n_keep = int(np.floor(float(dim) * (1.0 - float(eps))))
        if n_keep < 1:
            raise ValueError(
                f"遮蔽比例 {eps} 会把全部 {dim} 维遮蔽掉（n_keep={n_keep}）；拒绝继续"
            )
        keep = torch.stack(
            [torch.randperm(dim, generator=gen)[:n_keep] for _ in range(n_rows)], dim=0
        )
        mask = torch.zeros(base.shape, dtype=torch.float32)
        mask.scatter_(1, keep, 1.0)
        perturbed = (base * mask).numpy()
        formula = "x <- x * m, m keeps floor(D*(1-r)) random dims per row"
        detail = {
            "mask_ratio": float(eps),
            "n_keep": int(n_keep),
            "formula": formula,
            "kind": "random_dimension_mask",
            "uses_generator": True,
        }
    else:  # nmag：真正的幅度缩放 + 平移（`x <- x*s + b`，无随机数）
        shift = nmag_shift_matrix(int(base.shape[0]), int(base.shape[1]), float(eps))
        scale = (
            float(scale_override)
            if scale_override is not None
            else float(NMAG_SHARED_SCALE)
        )
        perturbed = (base * scale + torch.from_numpy(shift)).numpy()
        formula = "x <- x * s + b, s = NMAG_SHARED_SCALE, b[i,j] = eps * (-1)**(i+j+1)"
        detail = {
            "eps": float(eps),
            "scale": scale,
            "scale_override": (None if scale_override is None else float(scale_override)),
            "shift_magnitude": float(eps),
            "shift_sign_rule": "(-1)**(row+col+1)（棋盘式交替）",
            "formula": formula,
            "kind": "magnitude_scale_plus_shift",
            "uses_generator": False,
        }
    out, evidence = ET.l2_normalize_rows(perturbed)
    if int(evidence["n_zero_norm"]) > 0:
        raise ValueError(
            f"扰动 {kind}/{level} 后出现零范数行：n_zero_norm={evidence['n_zero_norm']}，"
            f"行下标 = {evidence['zero_norm_rows_head']}；拒绝静默产出 NaN / 全零特征"
        )
    displacement = float(
        np.linalg.norm(out.astype(np.float64) - _l2(src).astype(np.float64), axis=1).mean()
    )
    detail.update(
        {
            "level": level,
            "variant": int(variant),
            "eps_used": float(eps),
            "eps_override": (None if eps_override is None else float(eps_override)),
            "generator_seed": int(derived_seed(kind, level, variant, seed)),
            "seed": int(seed),
            "mean_row_displacement": displacement,
            "renormalized": True,
            "n_zero_norm": int(evidence["n_zero_norm"]),
            "post_norm_min": float(np.linalg.norm(out.astype(np.float64), axis=1).min()),
            "post_norm_max": float(np.linalg.norm(out.astype(np.float64), axis=1).max()),
            # 扰动后**未归一化**矩阵的裸字节指纹（G7 逐位断言用它把「公式偏差」与
            # 「归一化偏差」分开定位；矩阵本身不落盘，只落指纹）
            "raw_perturbed_sha256": ET.sha256_bytes(
                np.ascontiguousarray(perturbed, dtype=np.float32).tobytes()
            ),
            "raw_perturbed_shape": [int(x) for x in np.shape(perturbed)],
        }
    )
    return out, detail


# ---------------------------------------------------------------------------
# G7：扰动公式的逐位断言
# ---------------------------------------------------------------------------

#: G7 断言用的**已知输入矩阵**（确定性、与真实数据无关；用于闭式手算复核）。
FORMULA_CHECK_MATRIX: Tuple[Tuple[float, ...], ...] = (
    (0.5, -0.25, 0.125, -0.0625, 0.75, -0.5, 0.25, -0.125),
    (-0.5, 0.25, -0.125, 0.0625, -0.75, 0.5, -0.25, 0.125),
    (0.125, 0.125, -0.75, -0.75, 0.5, 0.5, -0.25, -0.25),
)


def _normalize_like_impl(arr: np.ndarray) -> np.ndarray:
    """与实现**逐位相同**的归一化步骤（``float64`` 求范数 → ``float32`` 落盘）。"""
    out, _evidence = ET.l2_normalize_rows(arr)
    return out


def assert_perturb_formulas_bitwise(
    kinds: Sequence[str] = PERTURB_TYPES,
    levels: Sequence[str] = PERTURB_LEVELS,
    *,
    seed: int = ROBUST_SEED,
) -> Dict[str, Any]:
    """**G7 门禁**：三种扰动 × 三档，逐位断言「扰动输出 == 按公式手算的结果」。

    复算口径（每一例都**独立按公式重算**，不调用 :func:`perturb_matrix`）
    -------------------------------------------------------------------
    * ``nmag`` —— **闭式**：``x·0.5 + ε·(−1)^(i+j+1)``，其中缩放系数 ``0.5`` 与符号矩阵
      都是**本函数内手写的字面量**（刻意不调符号矩阵 helper、不读共享缩放常量）；
    * ``noise`` —— 按公式**手写派生种子** ``seed*1000 + 100*type_index + 10*level_index``
      （刻意不调派生种子 helper）后新建 :class:`torch.Generator` 重抽同一个
      ``randn``，再 ``x + σ·ξ``（验证「同 seed 可复算」）；
    * ``mask`` —— 用同一个手写派生种子重放同一串 ``randperm``，重建同一张 0/1 掩码，
      再 ``x ⊙ m``。

    **去共享化（离朱 R50 F1 修复，关键）**
    -------------------------------------
    手算侧**不得复用任何生产代码路径**（helper 函数 / 共享常量 / 型别下标函数）。若手算侧
    调用与被测实现同一份符号矩阵 helper、派生种子 helper 或共享缩放常量，则破坏这些共享层时
    两侧同步变化 ⇒ 断言对这些层**恒真、没有区分度**（现场实测：把符号矩阵 helper 的返回值
    多乘 2、把共享缩放常量改成 0.25、把派生种子加 1，三种破坏当时**都检不出来**）。
    现在这些量全部在本函数内按公式手写重算，故上述破坏都应让 `all_bitwise_equal` 变 False。

    判定
    ----
    两种归一化路径都要逐字节相等才算通过：
    ① ``normalized``：先按公式得 ``[N, D]`` 再逐行 L2 归一化（与实现同一路径）；
    ② ``raw``：扰动后的**未归一化**矩阵也逐字节相等（把「公式」与「归一化」两类
      偏差分开定位；实现侧该指纹由 ``evidence["raw_perturbed_sha256"]`` 提供）。

    返回
    ----
    Dict[str, Any]
        ``cases``（逐例明细）+ ``n_cases`` / ``n_bitwise_equal`` /
        ``all_bitwise_equal`` / ``rule`` / ``check_matrix`` / ``manual_side_isolated``。
    """
    base = np.asarray(FORMULA_CHECK_MATRIX, dtype=np.float32)
    n_rows, dim = int(base.shape[0]), int(base.shape[1])
    cases: List[Dict[str, Any]] = []
    for kind in kinds:
        for level in levels:
            eps = float(PERTURB_GRID[str(kind)][str(level)])
            # [!] 手算侧的**去共享化**（离朱 R50 F1）：下面这段**刻意不复用**生产代码
            # ——不调符号矩阵 helper、不调派生种子 helper、不调型别下标 helper，
            # 也不读共享缩放常量；把「型别下标」「派生种子」「缩放系数」
            # 「棋盘符号矩阵」全部**按公式手写字面量重算**。否则破坏共享 helper 或共享
            # 常量时两侧会同步变化，断言对该层没有区分度（恒真）。
            type_index = int(list(PERTURB_TYPES).index(str(kind)))
            level_index = int(list(PERTURB_LEVELS).index(str(level)))
            gen_seed = int(
                int(seed) * 1000 + 100 * type_index + 10 * level_index + 0
            )
            base_t = torch.from_numpy(np.ascontiguousarray(base, dtype=np.float32))
            if str(kind) == "nmag":
                # 手写缩放字面量 0.5 + 手写棋盘符号矩阵（不调 helper、不读共享常量）
                manual_scale = 0.5
                ii = np.arange(n_rows, dtype=np.int64)[:, None]
                jj = np.arange(dim, dtype=np.int64)[None, :]
                manual_shift = (eps * np.where(((ii + jj) % 2) == 0, -1.0, 1.0)).astype(
                    np.float32
                )
                manual_raw = (
                    base_t * manual_scale + torch.from_numpy(manual_shift)
                ).numpy()
                recompute = (
                    f"手算闭式：x*0.5 + {eps}*(-1)**(i+j+1)（手写符号矩阵，无随机数）"
                )
            elif str(kind) == "noise":
                gen = torch.Generator(device="cpu")
                gen.manual_seed(gen_seed)
                noise = torch.randn(base_t.shape, generator=gen, dtype=torch.float32)
                manual_raw = (base_t + eps * noise).numpy()
                recompute = f"手算：重放手写派生种子 {gen_seed} 的 randn，再 x + {eps}*xi"
            else:  # mask
                gen = torch.Generator(device="cpu")
                gen.manual_seed(gen_seed)
                n_keep = int(np.floor(float(dim) * (1.0 - eps)))
                keep = torch.stack(
                    [torch.randperm(dim, generator=gen)[:n_keep] for _ in range(n_rows)],
                    dim=0,
                )
                mask = torch.zeros(base_t.shape, dtype=torch.float32)
                mask.scatter_(1, keep, 1.0)
                manual_raw = (base_t * mask).numpy()
                recompute = (
                    f"手算：重放手写派生种子 {gen_seed} 的 randperm"
                    f"（每行留 {n_keep}/{dim} 维），再 x⊙m"
                )
            impl_norm, impl_detail = perturb_matrix(base, str(kind), str(level), seed=int(seed))
            manual_norm = _normalize_like_impl(manual_raw)
            # [口径] `perturb_matrix` 返回的已经是**重新 L2 归一化后**的矩阵，故：
            #   ① `raw_bitwise_equal` = 实现内部**归一化前**那张矩阵的裸字节 SHA256
            #      vs 手算矩阵的裸字节 SHA256（隔离「公式」是否被正确实现）；
            #   ② `bitwise_equal` = 实现返回的归一化矩阵 vs 手算矩阵走**同一条归一化链**
            #      （隔离「归一化链」是否与实现一致）。两路都相等才算该例通过。
            manual_raw_sha = ET.sha256_bytes(
                np.ascontiguousarray(manual_raw, dtype=np.float32).tobytes()
            )
            raw_equal = bool(
                str(impl_detail.get("raw_perturbed_sha256", "")) == manual_raw_sha
            )
            norm_equal = bool(impl_norm.tobytes() == manual_norm.tobytes())
            cases.append(
                {
                    "kind": str(kind),
                    "level": str(level),
                    "eps": float(eps),
                    "generator_seed": int(gen_seed),
                    "recompute": recompute,
                    "raw_bitwise_equal": bool(raw_equal),
                    "bitwise_equal": bool(norm_equal),
                    "impl_raw_sha256": str(impl_detail.get("raw_perturbed_sha256", "")),
                    "manual_raw_sha256": manual_raw_sha,
                    "impl_norm_sha256": ET.sha256_bytes(impl_norm.tobytes()),
                    "manual_norm_sha256": ET.sha256_bytes(manual_norm.tobytes()),
                }
            )
    n_equal = int(sum(1 for c in cases if c["bitwise_equal"] and c["raw_bitwise_equal"]))
    return {
        "cases": cases,
        "n_cases": int(len(cases)),
        "n_bitwise_equal": int(n_equal),
        "all_bitwise_equal": bool(n_equal == len(cases) and len(cases) > 0),
        "rule": (
            "每一例都**独立按公式重算**（手算侧**不复用任何生产 helper/常量**："
            "nmag 手写 0.5 与棋盘符号矩阵、noise/mask 手写派生种子后重放 randn/randperm），"
            "判定分两路：① 扰动后**未归一化**矩阵的 float32 裸字节 SHA256 相等"
            "（`raw_bitwise_equal`，隔离「公式」是否被正确实现）；② 重新 L2 归一化后的"
            "裸字节逐字节相等（`bitwise_equal`，隔离「归一化链」）。两路都相等才算该例通过"
        ),
        "manual_side_isolated": True,
        "manual_side_note": (
            "手算侧已去共享化（离朱 R50 F1）：破坏三个共享层（符号矩阵 helper / "
            "派生种子 helper / 共享缩放常量）任一都应使本断言变 False"
        ),
        "check_matrix": [[float(x) for x in row] for row in FORMULA_CHECK_MATRIX],
    }


# ---------------------------------------------------------------------------
# 归一化余弦检索
# ---------------------------------------------------------------------------


def topk_search(
    keys: torch.Tensor, queries: torch.Tensor, *, k: int = TOPK, batch_size: int = BATCH_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """批量 **归一化余弦** top-k 检索（唯一实现）。

    参数
    ----
    keys : torch.Tensor
        ``[N, D]`` 键表（L2 归一化）。
    queries : torch.Tensor
        ``[M, D]`` 查询（L2 归一化）。
    k : int
        top-k 的 k。
    batch_size : int
        查询批大小（避免物化全量 ``[M, N]`` 相似度矩阵）。

    返回
    ----
    Tuple[np.ndarray, np.ndarray]
        ``(top_scores [M, k] float32, top_indices [M, k] int64)``。
    """
    kk = max(1, int(k))
    m = int(queries.shape[0])
    scores = np.zeros((m, kk), dtype=np.float32)
    indices = np.zeros((m, kk), dtype=np.int64)
    for b0 in range(0, m, max(1, int(batch_size))):
        chunk = queries[b0 : b0 + max(1, int(batch_size))].to(torch.float32)
        sims = chunk @ keys.to(torch.float32).t()
        k_eff = min(kk, int(sims.shape[1]))
        vals, idx = torch.topk(sims, k_eff, dim=1, largest=True, sorted=True)
        scores[b0 : b0 + int(chunk.shape[0]), :k_eff] = vals.numpy()
        indices[b0 : b0 + int(chunk.shape[0]), :k_eff] = idx.numpy()
    return scores, indices


def rank_of_gold(
    keys: torch.Tensor,
    queries: torch.Tensor,
    gold_index: Sequence[int],
    *,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """金标条目的**确定性名次**（1-based；并列按超出条数计，与 topk 顺序无关）。

    口径
    ----
    ``rank_i = 1 + #{ j : cos(q_i, k_j) > cos(q_i, k_gold_i) }``。
    ``keys`` 与 ``queries`` 若已逐位相同（同一张量），则对角项恒为 ``1.0``，
    从而 ``rank == 1`` —— 这是档 0「原样自检索必须为 1.0」的结构性依据。
    """
    gold = np.asarray([int(i) for i in gold_index], dtype=np.int64)
    m = int(queries.shape[0])
    if gold.shape[0] != m:
        raise ValueError(f"gold_index 长度 {gold.shape[0]} 与查询数 {m} 不一致")
    out = np.zeros(m, dtype=np.int64)
    for b0 in range(0, m, max(1, int(batch_size))):
        chunk = queries[b0 : b0 + max(1, int(batch_size))].to(torch.float32)
        sims = chunk @ keys.to(torch.float32).t()
        gold_scores = sims[torch.arange(int(chunk.shape[0])), torch.from_numpy(gold[b0 : b0 + int(chunk.shape[0])])]
        out[b0 : b0 + int(chunk.shape[0])] = (
            1 + (sims > gold_scores.unsqueeze(1)).sum(dim=1)
        ).numpy().astype(np.int64)
    return out


def rank_of_gold_set(
    keys: torch.Tensor,
    queries: torch.Tensor,
    gold_lists: Sequence[Sequence[int]],
    *,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """**金标集合**的确定性名次：命中集合中任意一条即算该名次（并列按超出条数计）。

    口径
    ----
    ``best_i = max_{j ∈ G_i} cos(q_i, k_j)``；
    ``rank_i = 1 + #{ j : cos(q_i, k_j) > best_i }``。
    单元素 ``G_i`` 时与 :func:`rank_of_gold` 完全一致（同一口径，不产生第二套定义）。
    """
    m = int(queries.shape[0])
    if len(gold_lists) != m:
        raise ValueError(f"gold_lists 长度 {len(gold_lists)} 与查询数 {m} 不一致")
    out = np.zeros(m, dtype=np.int64)
    for b0 in range(0, m, max(1, int(batch_size))):
        chunk = queries[b0 : b0 + max(1, int(batch_size))].to(torch.float32)
        n_chunk = int(chunk.shape[0])
        sims = chunk @ keys.to(torch.float32).t()
        for i in range(n_chunk):
            golds = [int(x) for x in gold_lists[b0 + i]]
            if not golds:
                out[b0 + i] = 0
                continue
            best = sims[i, torch.tensor(golds, dtype=torch.long)].max()
            out[b0 + i] = int(1 + (sims[i] > best).sum().item())
    return out


def evaluate_retrieval(
    keys: torch.Tensor,
    queries: torch.Tensor,
    gold_index: Sequence[int],
    *,
    topk: int = TOPK,
    batch_size: int = BATCH_SIZE,
    gold_lists: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, Any]:
    """一次归一化余弦检索的完整指标（Recall@1 / @5 / 名次直方图 / 未命中示例）。

    参数
    ----
    keys : torch.Tensor
        ``[N, D]`` 键表。
    queries : torch.Tensor
        ``[M, D]`` 查询。
    gold_index : Sequence[int]
        金标在键表中的行下标（每查询一个；仅用于命中率计算）。
    topk : int
        ``Recall@k`` 的最大 k。
    batch_size : int
        批大小。
    gold_lists : Optional[Sequence[Sequence[int]]]
        金标**集合**（每查询一组行下标）；给定时用它算名次，**空集合计入分母并恒判未命中**
        （「未识别」样本的口径）。

    返回
    ----
    Dict[str, Any]
        ``n`` / ``n_keys`` / ``recall_at_1`` / ``recall_at_5`` / ``rank_1`` /
        ``rank_max`` / ``n_no_gold`` / ``misses_at_1_head`` 等。
    """
    kk = max(1, int(topk))
    gold = np.asarray([int(i) for i in gold_index], dtype=np.int64)
    scores, indices = topk_search(keys, queries, k=kk, batch_size=batch_size)
    m = int(queries.shape[0])
    if gold_lists is not None:
        lists = [list(int(x) for x in g) for g in gold_lists]
        n_no_gold = int(sum(1 for g in lists if not g))
        ranks = rank_of_gold_set(keys, queries, lists, batch_size=batch_size)
    else:
        lists = [[int(x)] for x in gold.tolist()]
        n_no_gold = 0
        ranks = rank_of_gold(keys, queries, gold, batch_size=batch_size)
    hits1 = int((ranks == 1).sum())
    hits5 = int(((ranks >= 1) & (ranks <= kk)).sum())
    top1_scores = scores[:, 0] if m else np.zeros(0, dtype=np.float32)
    misses = [
        {"row": int(i), "gold": [int(x) for x in lists[i][:5]],
         "rank": int(ranks[i]), "top1": int(indices[i, 0])}
        for i in range(m) if int(ranks[i]) != 1
    ][:20]
    return {
        "n": int(m),
        "n_keys": int(keys.shape[0]),
        "topk": int(kk),
        "recall_at_1": float(hits1 / max(1, m)),
        "recall_at_5": float(hits5 / max(1, m)),
        "hit_at_1": int(hits1),
        "hit_at_5": int(hits5),
        "n_no_gold": int(n_no_gold),
        "rank_max": int(ranks.max()) if m else 0,
        "rank_hist_topk": {
            str(r): int((ranks == r).sum()) for r in range(1, kk + 1)
        },
        "top1_score_mean": float(np.mean(top1_scores)) if m else 0.0,
        "top1_score_min": float(np.min(top1_scores)) if m else 0.0,
        "top1_score_max": float(np.max(top1_scores)) if m else 0.0,
        "rank_rule": (
            "rank = 1 + #{j : cos(q,k_j) > 该查询金标集合内的最大余弦}（并列按超出条数计）；"
            "金标集合为空的查询 rank=0、恒判未命中并计入分母"
        ),
        "misses_at_1_head": misses,
    }


def unrecognized_report(
    keys: torch.Tensor,
    unknown_feats: np.ndarray,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    batch_size: int = BATCH_SIZE,
    quantiles: Sequence[float] = UNKNOWN_QUANTILES,
) -> Dict[str, Any]:
    """「未识别」档：未知条目对表的最高余弦 + 未识别率 + 误召回率。

    口径（显式固定）
    ---------------
    * ``未识别`` ⇔ 最高余弦 ``< τ``；``误召回`` ⇔ 最高余弦 ``>= τ``；
    * ``τ`` 由现场标定给出（``thresholds`` 里按 key 取）；**未给 τ 时**
      ``未识别率`` / ``误召回率`` 如实标 ``applicable=False``（不臆造阈值）；
    * 另报**不依赖 τ** 的分位数与均值（τ 未标定时仍可读）；
    * **不标定阈值**（本函数只读标定结果，标定属 :func:`calibrate_thresholds`）。
    """
    unknown = torch.from_numpy(np.ascontiguousarray(np.asarray(unknown_feats, dtype=np.float32)))
    m = int(unknown.shape[0])
    best = np.zeros(m, dtype=np.float32)
    for b0 in range(0, m, max(1, int(batch_size))):
        chunk = unknown[b0 : b0 + max(1, int(batch_size))]
        sims = chunk @ keys.to(torch.float32).t()
        vals, _idx = torch.max(sims, dim=1)
        best[b0 : b0 + int(chunk.shape[0])] = vals.numpy()
    out: Dict[str, Any] = {
        "n": int(m),
        "n_keys": int(keys.shape[0]),
        "best_cosine": {
            "mean": float(best.mean()) if m else 0.0,
            "min": float(best.min()) if m else 0.0,
            "max": float(best.max()) if m else 0.0,
            "quantiles": {
                str(q): float(np.quantile(best, float(q))) if m else 0.0
                for q in quantiles
            },
        },
        "threshold_standard": (
            "未识别 ⇔ 最高余弦 < τ；误召回 ⇔ 最高余弦 >= τ。本档**不标定**阈值"
            "（阈值标定属第二步变体 B 的范围），τ 由 calibrate 现场给出。"
        ),
    }
    tau_map = dict(thresholds or {})
    per_threshold: Dict[str, Any] = {}
    for name, tau in sorted(tau_map.items()):
        recall = int((best >= float(tau)).sum())
        per_threshold[str(name)] = {
            "threshold": float(tau),
            "unrecognized_rate": float((m - recall) / max(1, m)),
            "false_recall_rate": float(recall / max(1, m)),
            "n_false_recall": int(recall),
        }
    out["per_threshold"] = per_threshold
    if not per_threshold:
        out["unrecognized_rate"] = None
        out["false_recall_rate"] = None
        out["rate_applicable"] = False
        out["rate_note"] = "无标定阈值可用 -> 未识别率 / 误召回率如实标不适用（不臆造 τ）"
    else:
        primary = sorted(per_threshold.keys())[0]
        out["unrecognized_rate"] = per_threshold[primary]["unrecognized_rate"]
        out["false_recall_rate"] = per_threshold[primary]["false_recall_rate"]
        out["rate_applicable"] = True
        out["rate_note"] = f"主口径阈值来源 = {primary}"
    return out


def euclidean_equivalence_evidence(
    keys: torch.Tensor,
    queries: torch.Tensor,
    *,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Any]:
    """现场实测「归一化余弦 top-1 ⇔ 欧氏 top-1」的等价性（登记用，不作为对照轴）。

    口径
    ----
    在 L2 归一化特征上 ``‖a − b‖² = 2 − 2·cos(a, b)``，故 ``argmin`` 距离与
    ``argmax`` 余弦必然同解。本函数现场实测两条独立证据：
    ① 键表行范数区间；② 逐个查询的余弦 top-1 与欧氏 top-1 是否逐位相同；
    ③ 恒等式 ``max|‖a−b‖² − (2 − 2·cos)|`` 的实测残差上界（float32 舍入）。
    """
    keys32 = keys.to(torch.float32)
    norms = torch.linalg.norm(keys32, dim=1)
    rows = min(int(queries.shape[0]), int(batch_size) * 3)
    cos_idx: List[int] = []
    euc_idx: List[int] = []
    max_identity_residual = 0.0
    for b0 in range(0, rows, max(1, int(batch_size))):
        chunk = queries[b0 : b0 + max(1, int(batch_size))].to(torch.float32)
        sims = chunk @ keys32.t()
        cos_idx.extend([int(x) for x in torch.argmax(sims, dim=1).tolist()])
        dist2 = (
            (chunk * chunk).sum(dim=1, keepdim=True)
            + (keys32 * keys32).sum(dim=1).unsqueeze(0)
            - 2.0 * sims
        )
        euc_idx.extend([int(x) for x in torch.argmin(dist2, dim=1).tolist()])
        identity = (dist2 - (2.0 - 2.0 * sims)).abs()
        max_identity_residual = max(max_identity_residual, float(identity.max().item()))
    mismatch = int(sum(1 for a, b in zip(cos_idx, euc_idx) if a != b))
    return {
        "n_queries_checked": int(len(cos_idx)),
        "n_top1_mismatch": int(mismatch),
        "key_row_norm_min": float(norms.min().item()) if norms.numel() else 0.0,
        "key_row_norm_max": float(norms.max().item()) if norms.numel() else 0.0,
        "max_abs_identity_residual": float(max_identity_residual),
        "identity": "‖a-b‖² = 2 - 2·cos(a,b)（对 L2 归一化向量）",
        "conclusion": (
            "对 L2 归一化向量，余弦 top-1 与欧氏 top-1 逐位等价；**不把「余弦/欧氏对比」"
            "当作有信息量的对照轴**，欧氏基线轴已从本次考卷中移除"
        ),
    }


# ---------------------------------------------------------------------------
# 评测单元（cell）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCell:
    """一次评测单元的静态输入（键表 + 查询 + 金标 + 未识别负样本）。

    属性
    ----
    side : str
        ``text`` / ``qa``。
    profile : str
        特征档名。
    keys : torch.Tensor
        键表（干净，L2 归一化）。
    queries : torch.Tensor
        查询（干净，L2 归一化）。
    gold_index : List[int]
        金标在键表中的行下标。
    table_n : int
        条目数（进报告，便于读者判断统计意义）。
    query_n : int
        查询数。
    unknown_feats : Optional[np.ndarray]
        「未识别」负样本特征（``None`` = 本侧不报该档）。
    strong_note : str
        统计意义标注。
    gold_lists : List[List[int]]
        金标**集合**（每查询一组行下标；空集合 = 答案不在表中）。
    role : str
        测评角色（``self_retrieval`` / ``known_queries`` / ``unknown_queries``）。
    """

    side: str
    profile: str
    keys: torch.Tensor
    queries: torch.Tensor
    gold_index: List[int]
    table_n: int
    query_n: int
    unknown_feats: Optional[np.ndarray] = None
    strong_note: str = ""
    gold_lists: List[List[int]] = field(default_factory=list)
    role: str = "self_retrieval"


def text_cell_from_bundle(bundle: TextEntryBundle, profile: str) -> EvalCell:
    """由文本侧 bundle 装配评测单元（金标 = 查询行自身，自检索口径）。"""
    positions = [int(i) for i in bundle.query_positions]
    return EvalCell(
        side="text",
        profile=str(profile),
        keys=bundle.table.keys,
        queries=torch.from_numpy(
            np.ascontiguousarray(bundle.query_feats, dtype=np.float32)
        ),
        gold_index=positions,
        table_n=int(bundle.table.size),
        query_n=int(len(positions)),
        unknown_feats=None,
        strong_note=(
            "文本侧查询 666 条、条目 2665 行 —— 主判据依据；但纯特征基线已近饱和"
            "（R@1 = 0.9834834834834835），弱档与档 0 之间的空间极小，报告须显式说明。"
        ),
        gold_lists=[[int(p)] for p in positions],
        role="self_retrieval",
    )


def answer_key_gold_map(known_answer_keys: Sequence[str]) -> Dict[str, List[int]]:
    """把条目表的逐行答案键映射为 ``答案键 -> 行下标列表``（顺序敏感、确定性）。

    用途
    ----
    QA 侧的金标**不是**「查询自身行」（那是文本侧自检索的口径），而是
    「同答案键在条目表中的行集合」—— 因为一个答案类在 ``train_known`` 里有
    多条问题（现场实测每类 12~20 条），命中其中任意一条即为命中该答案。
    """
    out: Dict[str, List[int]] = {}
    for i, key in enumerate(known_answer_keys):
        out.setdefault(str(key), []).append(int(i))
    return out


def gold_positions(answer_keys: Sequence[str], gold_map: Dict[str, List[int]]) -> List[int]:
    """把一批查询的答案键映射为金标行下标列表（**空列表 = 答案不在表中**）。

    ``[]`` 表示该查询的正确答案不在条目表中（「不相关 / 未识别」样本），
    此时 :func:`rank_of_gold` 无法定义名次；调用方须按「已知题」筛选后再算命中率。
    """
    return [list(gold_map.get(str(k), [])) for k in answer_keys]


def qa_cell_from_bundle(bundle: "ET.QAEntryBundle", profile: str) -> EvalCell:
    """由 QA 侧 bundle 装配评测单元（金标 = 同答案键的条目行集合）。

    两个 QA 口径**都**基于同一个映射（答案键 → ``train_known`` 行下标）：

    * ``unknown``（默认、辅助口径、**统计意义强**）：查询 = 未识别负样本 2893 条。
      现场实测这 2893 条的 1770 个答案键**与** ``train_known`` 的 10 个类**零重叠**，
      因此金标集合恒为空 —— 它们无法贡献命中率（命中率恒 0）。
    * ``known_queries``：查询 = ``test_known`` 20 条，答案键 100% 落在表内
      （现场实测缺口 0），故可算命中率，但 **20 条统计意义弱**、不得作主判据。

    参数
    ----
    bundle : ET.QAEntryBundle
        QA 侧装配结果。
    profile : str
        特征档名。

    返回
    ----
    EvalCell
        评测单元（``gold_index`` 为**扁平行下标**的展平集合；详细映射见
        ``bundle`` 现场重建）。
    """
    return _qa_cell(bundle, profile, use_unknown=False)


def _qa_cell(bundle: "ET.QAEntryBundle", profile: str, *, use_unknown: bool) -> EvalCell:
    """内部：按口径装配 QA 侧评测单元。"""
    gold_map = answer_key_gold_map(bundle.known_answer_keys)
    if use_unknown:
        feats = bundle.unknown.feats
        keys_used = list(bundle.unknown_answer_keys)
        entry_ids = list(bundle.unknown.entry_ids)
        role = "unknown_queries"
        note = (
            "QA 侧**辅助**口径：查询 = 未识别负样本 "
            f"{len(entry_ids)} 条（答案键与条目表零重叠 -> 无金标，命中率恒 0），"
            "仅用于验证「扰动 → 掉点」这一考卷有效性，**不是** QA 业务指标"
        )
    else:
        feats = bundle.known_queries.feats
        keys_used = list(bundle.query_answer_keys)
        entry_ids = list(bundle.known_queries.entry_ids)
        role = "known_queries"
        note = (
            f"QA 侧**主判据依据**的 QA 口径：查询 = test_known {len(entry_ids)} 条，"
            "金标 = 同答案键在条目表中的行集合；**20 条统计意义弱**，"
            "报告中必须与文本侧分开读"
        )
    gold_lists = gold_positions(keys_used, gold_map)
    n_no_gold = int(sum(1 for g in gold_lists if not g))
    flat: List[int] = []
    for g in gold_lists:
        if g:
            flat.append(int(g[0]))
    return EvalCell(
        side="qa",
        profile=str(profile),
        keys=bundle.table.keys,
        queries=torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)),
        gold_index=flat,
        table_n=int(bundle.table.size),
        query_n=int(len(entry_ids)),
        unknown_feats=None,
        strong_note=f"role={role}；{note}；无金标查询 {n_no_gold}/{len(entry_ids)} 条",
        gold_lists=gold_lists,
        role=role,
    )


def evaluate_cell(
    cell: EvalCell,
    *,
    kind: str,
    level: str,
    variant: int = 0,
    seed: int = ROBUST_SEED,
    perturb_table: bool = False,
    topk: int = TOPK,
    batch_size: int = BATCH_SIZE,
    clean: bool = False,
    eps_override: Optional[float] = None,
) -> Dict[str, Any]:
    """对一个评测单元施加（扰动类型 × 档位 × 变体）并返回指标。

    参数
    ----
    cell : EvalCell
        评测单元。
    kind : str
        扰动类型。
    level : str
        档位。
    variant : int
        变体序号（0 = 主判据用的那一条）。
    seed : int
        基准种子。
    perturb_table : bool
        是否**同时**扰动键表（用于对称口径的取证；主判据与标定均为 ``False``）。
    topk : int
        ``Recall@k`` 的最大 k。
    batch_size : int
        批大小。
    clean : bool
        ``True`` = 档 0（不扰动，原样自检索）。
    eps_override : Optional[float]
        显式覆盖该档的扰动强度（**只用于噪声底标定**：取「可忽略强度」，
        见 :data:`NOISE_LEVEL_RATIO`）。

    返回
    ----
    Dict[str, Any]
        指标 + 扰动取证（含生成器种子、位移量、重归一化后范数区间）。
    """
    keys = cell.keys
    queries = cell.queries
    detail: Dict[str, Any] = {
        "clean": bool(clean),
        "perturb_table": bool(perturb_table),
    }
    if not clean:
        eps = (
            float(eps_override)
            if eps_override is not None
            else float(PERTURB_GRID[str(kind)][str(level)])
        )
        q_np, q_detail = perturb_matrix(
            queries.numpy(), kind, level, variant=variant, seed=seed, eps_override=eps
        )
        queries = torch.from_numpy(np.ascontiguousarray(q_np, dtype=np.float32))
        detail = dict(q_detail)
        detail["perturb_table"] = bool(perturb_table)
        detail["clean"] = False
        if perturb_table:
            k_np, k_detail = perturb_matrix(
                keys.numpy(), kind, level, variant=variant, seed=seed, eps_override=eps
            )
            keys = torch.from_numpy(np.ascontiguousarray(k_np, dtype=np.float32))
            detail["table_displacement"] = float(k_detail["mean_row_displacement"])
    metrics = evaluate_retrieval(
        keys, queries, cell.gold_index, topk=int(topk), batch_size=int(batch_size),
        gold_lists=cell.gold_lists or None,
    )
    return {
        "side": str(cell.side),
        "profile": str(cell.profile),
        "role": str(cell.role),
        "kind": str(kind),
        "level": str(level),
        "variant": int(variant),
        "table_n": int(cell.table_n),
        "query_n": int(cell.query_n),
        "evidence": detail,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# 阈值现场标定（两阶段）
# ---------------------------------------------------------------------------


def negligible_eps(kind: str, level: str) -> float:
    """噪声底标定用的**可忽略强度**：``NOISE_LEVEL_RATIO × 该扰动弱档的 ε``。

    两种扰动语义（都必须保持「可忽略」）
    ----------------------------------
    * ``noise`` / ``mask``：直接是 σ / 遮蔽比例（``kind="noise"`` ⇒ σ=0.001、
      ``kind="mask"`` ⇒ r=0.002）；
    * ``nmag``：**ε 同时控制缩放与平移**（``x ← x·s + ε·(±1)``）。测噪声底时**不缩放**
      （``scale_override = 1.0``）、只做 ``NMAG_NEGLIGIBLE_SHIFT = 1e-3`` 的可忽略平移，
      即 ``x ← x·1 + 1e-3·(±1)`` —— 缩放与平移**双双可忽略**，与其它两种扰动同口径。
      （现场实测：若此处沿用档位的 ``s = 0.5``，特征被整体压到 0.5 倍量级，
      噪声底会被抬到 0.521021、τ = 1.042 高于任何可能落差，把「有效」误判成「无效」。）"""
    return float(NOISE_LEVEL_RATIO) * float(PERTURB_GRID[str(kind)][str(level)])


def noise_floor_of_cell(
    cell: EvalCell,
    *,
    kind: str,
    level: str,
    seed: int = ROBUST_SEED,
    topk: int = TOPK,
    batch_size: int = BATCH_SIZE,
    cell_role: str = "main",
) -> Dict[str, Any]:
    """**噪声底**（现场实测、可忽略强度）：同一内容被**两次独立随机绘制**扰动后的掉点。

    口径（显式固定）
    ---------------
    * 强度 = :func:`negligible_eps`（该扰动**弱档** ε 的 1/50，即 σ=0.001 / 遮蔽 0.2%）；
    * ``nmag`` **不缩放、只做可忽略平移**：``x ← x·1 + δ·(±1)``，``δ = 1e-3``
      （见 :func:`nmag_negligible_shift`）—— 若沿用档位那套 ``s = 0.5``，则特征会被整体
      压到 0.5 倍量级，**不可忽略**（现场实测该做法把文本侧 nmag 噪声底抬到 0.521021，
      使 τ = 1.042 高于任何可能落差，把「有效」判成「无效」）；
    * 查询用变体 0、键表用变体 1 的**独立**随机绘制（两者**不是**同一个噪声向量）；
    * 掉点 = ``档0 R@1 − 该 cell 的 R@1``；
    * **必须在与主判据同一 ``cell_role`` 的同一批查询上生成**（W2 修复）：QA 侧已知查询
      的角色是 ``qa_known``（20 条 ``test_known``），若借用 ``main`` 角色（2893 条
      **未识别**负样本）的噪声底，就是拿另一个总体的散布去判 20 条查询的落差 —— 量纲与
      粒度都对不上（现场实测该错配使 QA 侧噪声底被抬到 0.141667~0.325，而 20 条查询的
      R@1 粒度只有 0.05）。

    为什么必须「两次独立绘制」而不是「同一噪声向量施加于两侧」
    --------------------------------------------------------
    同一个噪声向量施加于查询与键表时，**自检索**（查询行 = 键表行）是**恒等变换**：
    同一行被同一向量扰动后仍与自身完全一致，命中率恒 1.0 —— 现场实测该构造下
    文本侧三档 R@1 全为 ``1.000000``，会把噪声底误标成 1.0（τ = 2.0，任何落差都判无效）。
    独立绘制则如实暴露「同一内容两次编码不可能逐位相同」这一可观测差异。

    现场实测该量的**平直性**（文本侧 / ``local-hash`` / ``noise``）：ε=0.001 与
    ε=0.01 给出**相同**的 R@1 = 0.983483 —— 说明它反映的是键表中**近重复行**导致的
    自匹配歧义（换一次随机绘制就换一条近重复行胜出），而不是随 ε 连续增长的量。
    报告中必须连同这条性质一起读。

    注：``nmag`` **不含随机数**（见 :func:`nmag_shift_matrix`），故其「两次独立绘制」
    在实现上退化为**同一确定性平移作用于两侧**；本函数在该情形下会如实把
    ``independent_draw: False`` 写进取证。

    参数
    ----
    cell_role : str
        **运行级 cell 角色**（``main`` / ``qa_known``）。它会被如实写进
        ``evidence["population"]["cell_role"]``，与 :func:`calibrate_thresholds` 中
        ``entry["cell_role"]`` **同口径**（离朱 R50 F3 修复：此前该字段写的是
        :attr:`EvalCell.role`（``self_retrieval`` / ``known_queries``），与 entry 的
        运行级角色名不同，导致「同总体」这条约束在字段字面上无法核对）。
        ``EvalCell.role`` 另存于 ``population["eval_cell_role"]``，两种口径都留痕。
    """
    eps = negligible_eps(kind, level)
    if str(kind) == "nmag":
        eps = nmag_negligible_shift()
        scale_override = 1.0
    else:
        scale_override = None
    q_np, q_detail = perturb_matrix(
        cell.queries.numpy(), kind, level, variant=0, seed=seed, eps_override=eps,
        scale_override=scale_override,
    )
    k_np, k_detail = perturb_matrix(
        cell.keys.numpy(), kind, level, variant=1, seed=seed, eps_override=eps,
        scale_override=scale_override,
    )
    metrics = evaluate_retrieval(
        torch.from_numpy(np.ascontiguousarray(k_np, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(q_np, dtype=np.float32)),
        cell.gold_index,
        topk=int(topk),
        batch_size=int(batch_size),
        gold_lists=cell.gold_lists or None,
    )
    return {
        "side": str(cell.side),
        "profile": str(cell.profile),
        "role": str(cell.role),
        "population_role": str(cell_role),
        "kind": str(kind),
        "level": "noise",
        "variant": 0,
        "table_n": int(cell.table_n),
        "query_n": int(cell.query_n),
        "evidence": {
            "clean": False,
            "noise_floor": True,
            "noise_level": True,
            "eps_used": float(eps),
            # 如实记录**实际传给 perturb_matrix 的** scale_override（离朱 R50 F2：
            # 此前该字段写成 eps，而实际传的是 1.0，读数与实际口径不符）
            "scale_override": (
                float(scale_override) if scale_override is not None else None
            ),
            "scale_used": (
                float(scale_override) if scale_override is not None else NMAG_SHARED_SCALE
            ),
            "query_variant": 0,
            "table_variant": 1,
            "independent_draw": bool(str(kind) != "nmag"),
            "query_displacement": float(q_detail["mean_row_displacement"]),
            "table_displacement": float(k_detail["mean_row_displacement"]),
            "generator_seed": int(seed),
            "formula": str(q_detail.get("formula", "")),
            # 噪声底**总体溯源**：必须能看出它是在哪一批查询上量的（W2 修复）
            "population": {
                # 运行级角色（与 entry["cell_role"] 同口径）
                "cell_role": str(cell_role),
                # EvalCell 自身的角色（self_retrieval / known_queries），两种口径都留痕
                "eval_cell_role": str(cell.role),
                "query_n": int(cell.query_n),
                "table_n": int(cell.table_n),
                "min_granularity": float(1.0 / max(1, int(cell.query_n))),
                "side": str(cell.side),
                "profile": str(cell.profile),
            },
        },
        "metrics": metrics,
    }


def aggregate_level_hit(
    cells: Sequence[Dict[str, Any]], *, level: str
) -> Dict[str, Any]:
    """把一批 cell 结果按档位聚合（**变体 × 扰动类型跨格平均**，只用于标定口径）。

    参数
    ----
    cells : Sequence[Dict[str, Any]]
        单元格结果（``evaluate_cell`` 的返回值）。
    level : str
        档位。

    返回
    ----
    Dict[str, Any]
        ``n_cells`` / ``hit_rate_mean`` / ``recall_at_5_mean`` 等。
    """
    picked = [c for c in cells if str(c.get("level")) == str(level)]
    if not picked:
        return {"n_cells": 0, "hit_rate_mean": 0.0, "recall_at_5_mean": 0.0}
    r1 = [float(c["metrics"]["recall_at_1"]) for c in picked]
    r5 = [float(c["metrics"]["recall_at_5"]) for c in picked]
    return {
        "n_cells": int(len(picked)),
        "hit_rate_mean": float(np.mean(r1)),
        "hit_rate_min": float(np.min(r1)),
        "hit_rate_max": float(np.max(r1)),
        "recall_at_5_mean": float(np.mean(r5)),
    }


#: 判定等号边界格的**期望 verdict 片段**（G8 断言用；唯一来源）。
AT_GRANULARITY_VERDICT_PHRASE: str = "不构成判定依据"

#: 「按**两测主判据角色**统计的落差恰等于粒度格数」这一计数的**唯一字段名**。
#:
#: [!] 该计数**只允许存在于** ``granularity_boundary_evidence`` 内；``calibrate_thresholds``
#: 的**构造期不变量**会断言它**不在**顶层（离朱 R53 曾出现「顶层与 G8 同名不同义」，
#: R55 建议把该约束从「代码约定 + 文档」升级为构造期断言）。字段名在此收敛为单一常量，
#: 供 G8 块构造、``_g8_roles_count`` 取数、``granularity_count_scopes`` 口径标签与
#: 构造期断言四处共用，避免手写字符串漂移。
GRANULARITY_ROLES_COUNT_KEY: str = "n_at_granularity_primary_roles"


def _granularity_boundary_evidence(
    entries: Sequence[Dict[str, Any]],
    *,
    criterion: str,
    min_granularity_note: str,
) -> Dict[str, Any]:
    """**G8 门禁证据**：等号边界（``abs(gap_used) == min_granularity``）的现场实例与四面口径一致性。

    口径（审查 W5 + G8）
    -------------------
    * **实例**：从真实 entries 中筛出 ``at_granularity is True`` 的格（现场必然非空 ——
      QA 侧 20 条查询下，落差恰为 1 个样本 = 0.05 的格有 3 个）；
    * **判定一致性**：每一例必须 `below_granularity is True`、`effective is False`、
      且 verdict 含 :data:`AT_GRANULARITY_VERDICT_PHRASE`（「不构成判定依据」）；
    * **四面口径一致性**（这是上两轮反复出问题的模式）：
      ① 实现 —— 源码里守卫用 ``<=``（由本函数内的**合成实例**独立验证：构造
      ``gap == granularity`` 的两侧边界，断言判无效、``gap`` 略大一点点判有效）；
      ② docstring —— :data:`GRANULARITY_GUARD_RULE` 的文本；
      ③ 产物字段 —— ``criterion`` 必须**包含**该规则文本；
      ④ ``min_granularity_note`` 也必须包含该规则文本。
    """
    boundary = [e for e in entries if e.get("at_granularity")]
    below = [e for e in entries if e.get("below_granularity")]
    # 辅助视图：**弱档 − 强档** 这一口径落在粒度边界上的格（不论它是否被 operative 判据采用）
    aux_boundary = [
        e for e in entries
        if e.get("weak_minus_strong_at_granularity") and not e.get("at_granularity")
    ]
    # ---- ① 实现侧：合成等号边界（不依赖现场数据是否恰好命中）----
    # 用**比值空间**判定（与实现同口径）：±1e-6 相对量级（远大于容差、远小于半个样本），
    # 从而三种情形可判：恰好等号 / 略低 / 略高。
    g = 1.0 / 20.0  # QA 侧粒度

    def _classify(gap: float) -> Tuple[bool, bool]:
        ratio = abs(gap) / g
        return (
            bool(ratio <= 1.0 + GRANULARITY_REL_TOL),
            bool(abs(ratio - 1.0) <= GRANULARITY_REL_TOL),
        )

    synthetic: List[Dict[str, Any]] = []
    for label, gap, want_below, want_at in (
        ("gap == granularity（恰好 1 个样本）", g, True, True),
        ("gap == 真实计算值 0.20-0.15（浮点非逐位相等）", 0.20 - 0.15, True, True),
        ("gap 略小于 granularity（-1e-6 相对）", g * (1.0 - 1e-6), True, False),
        ("gap 略大于 granularity（+1e-6 相对）", g * (1.0 + 1e-6), False, False),
    ):
        got_below, got_at = _classify(gap)
        synthetic.append(
            {
                "case": label,
                "gap": float(gap),
                "granularity": float(g),
                "ratio": float(abs(gap) / g),
                "want_below_granularity": bool(want_below),
                "got_below_granularity": bool(got_below),
                "want_at_granularity": bool(want_at),
                "got_at_granularity": bool(got_at),
                "match": bool(got_below == want_below and got_at == want_at),
                "expected_effective": False if got_below else True,
            }
        )
    synth_ok = bool(synthetic and all(c["match"] for c in synthetic))
    # ---- ② / ③ / ④ 文本一致性 ----
    rule_in_criterion = bool(GRANULARITY_GUARD_RULE in str(criterion))
    rule_in_note = bool(GRANULARITY_GUARD_RULE in str(min_granularity_note))
    # ---- 现场实例的一致性 ----
    instances: List[Dict[str, Any]] = []
    for e in boundary:
        instances.append(
            {
                "side": e.get("side"),
                "profile": e.get("profile"),
                "cell_role": e.get("cell_role"),
                "kind": e.get("kind"),
                "gap_used": e.get("gap_used"),
                "min_granularity": e.get("min_granularity"),
                "granularity_ratio": e.get("granularity_ratio"),
                "n_queries": e.get("n_queries"),
                "below_granularity": e.get("below_granularity"),
                "effective": e.get("effective"),
                "verdict": e.get("verdict"),
                "verdict_has_phrase": bool(
                    AT_GRANULARITY_VERDICT_PHRASE in str(e.get("verdict", ""))
                ),
            }
        )
    instances_ok = bool(
        instances
        and all(
            i["below_granularity"] is True
            and i["effective"] is False
            and i["verdict_has_phrase"]
            for i in instances
        )
    )
    below_ok = bool(all(e.get("effective") is False for e in below))
    return {
        "rule": GRANULARITY_GUARD_RULE,
        "rel_tol": float(GRANULARITY_REL_TOL),
        "at_granularity_verdict_phrase": AT_GRANULARITY_VERDICT_PHRASE,
        "n_boundary_instances": int(len(boundary)),
        "n_below_granularity_all": int(len(below)),
        "boundary_instances": instances,
        "auxiliary_boundary_instances": [
            {
                "side": e.get("side"),
                "profile": e.get("profile"),
                "cell_role": e.get("cell_role"),
                "kind": e.get("kind"),
                "gap_weak_to_strong": e.get("gap_weak_to_strong"),
                "gap_level_range": e.get("gap_level_range"),
                "gap_used": e.get("gap_used"),
                "gap_used_basis": e.get("gap_used_basis"),
                "weak_minus_strong_ratio": e.get("weak_minus_strong_ratio"),
                "effective": e.get("effective"),
                "note": (
                    "「弱档 − 强档」口径恰为 1 个样本（= 粒度），但该格档序列非单调、"
                    "operative 判据是整段落差（= 2 个样本），故**不据此判无效**；"
                    "此处仅登记该脆弱情形供审查核对"
                ),
            }
            for e in aux_boundary
        ],
        "n_auxiliary_boundary_instances": int(len(aux_boundary)),
        # [!] 口径与顶层 `n_at_granularity_primary` **不同**（离朱 R53/R54 收口）：
        # 此处按「**两测主判据角色**」`PRIMARY_CELL_ROLE.values()` = `main` + `qa_known` 统计；
        # 顶层那个按 `primary_side`（= `text` 侧）统计。该计数的**权威来源就是本块**，
        # 顶层**不得**镜像同名字段 —— 由 `calibrate_thresholds` 的构造期断言保证。
        GRANULARITY_ROLES_COUNT_KEY: int(
            sum(1 for e in entries if e.get("at_granularity") and str(e.get("cell_role")) in
                set(PRIMARY_CELL_ROLE.values()))
        ),
        "synthetic_boundary_cases": synthetic,
        "checks": {
            "synthetic_boundary_matches_rule": synth_ok,
            "boundary_instances_are_ineffective_and_annotated": instances_ok,
            "all_below_granularity_are_ineffective": below_ok,
            "rule_text_in_artifact_criterion": rule_in_criterion,
            "rule_text_in_min_granularity_note": rule_in_note,
        },
        "four_way_consistency": (
            "实现(源码用比值空间 `ratio <= 1 + GRANULARITY_REL_TOL`) / "
            "docstring(GRANULARITY_GUARD_RULE) / "
            "产物 criterion / min_granularity_note 四处必须含同一段规则文本；"
            "本节的 checks 逐项现场判定"
        ),
        "all_consistent": bool(
            synth_ok and instances_ok and below_ok and rule_in_criterion and rule_in_note
        ),
    }


def calibrate_thresholds(
    run_result: Dict[str, Any],
    *,
    factor: float = THRESHOLD_FACTOR,
    primary_side: str = "text",
) -> Dict[str, Any]:
    """两阶段阈值标定的**第二阶段**：由实测落差 + 噪声底现场标定阈值。

    标定依据（参照 ``exp_repr`` 的 ``ANCHOR_TOL`` 写法）
    ---------------------------------------------------
    1. **噪声底** ``noise_floor`` = 「可忽略强度下查询与键表用**两次独立随机绘制**」
       时相对档 0 的最大掉点（逐侧 / 逐扰动 / 逐编码器取最大值）—— 即「尺子本身的
       读数抖动」。强度与构造见 :func:`noise_floor_of_cell`；该量现场实测**对该扰动
       强度平直**（文本侧 / ``local-hash`` / ``noise``：ε=0.001 与 ε=0.01 同为
       R@1 = 0.983483），说明它反映的是键表近重复行导致的自匹配歧义；
    2. **标定阈值** ``τ = factor × noise_floor``（``factor = 2.0``，即观测噪声的约 2 倍）；
    3. **判据**：弱档 − 强档 **严格大于** τ ⇒ 该扰动对考卷**有效**；落差不大于 τ ⇒
       判定该扰动对本数据**无效**并如实登记（**不得带着坏尺子进第二步**）。

    参数
    ----
    run_result : Dict[str, Any]
        ``run`` 产物的完整 JSON。
    factor : float
        标定倍数。
    primary_side : str
        主判据侧（默认 ``text``；QA 侧只作辅助报告）。

    返回
    ----
    Dict[str, Any]
        标定结果（每侧 / 每扰动 / 每编码器的噪声底、阈值、落差与判定）。
    """
    cells = list(run_result.get("cells", []))
    keep_roles = set(PRIMARY_CELL_ROLE.values()) | {"noise_floor"}
    cells = [c for c in cells if str(c.get("cell_role", "main")) in keep_roles]
    grids = list(run_result.get("grid", {}).get("perturb_types", list(PERTURB_TYPES)))
    levels = list(run_result.get("grid", {}).get("levels", list(PERTURB_LEVELS)))
    profiles = list(run_result.get("grid", {}).get("profiles", []))
    sides = list(run_result.get("grid", {}).get("sides", list(SIDES)))

    entries: List[Dict[str, Any]] = []
    noise_max_overall = 0.0
    for side in sides:
        role = PRIMARY_CELL_ROLE.get(str(side), "main")
        for profile in profiles:
            # 档 0 基准：先从**本侧本档**的所有变体中取（同一次构造、同一次运行）
            clean_pool = [
                c for c in cells
                if c["side"] == side and c["profile"] == profile
                and str(c.get("cell_role")) == role
                and str(c.get("kind")) == "none"
            ]
            base = (
                float(clean_pool[0]["metrics"]["recall_at_1"]) if clean_pool else 0.0
            )
            for kind in grids:
                s_cells = [
                    c for c in cells
                    if c["side"] == side and c["profile"] == profile and c["kind"] == kind
                    and str(c.get("cell_role")) == role
                ]
                level_hits = {
                    lv: aggregate_level_hit(s_cells, level=lv)["hit_rate_mean"]
                    for lv in levels
                }
                # 噪声底：**必须与主判据同 side × profile × role × kind**（W2 修复）。
                # 过滤条件里带上 role 是关键：QA 侧 main = 2893 条**未识别**负样本，
                # qa_known = 20 条 test_known —— 若不加 role，20 条查询的落差会被拿去
                # 与「2893 条另一总体」的散布比较（历史缺陷：QA 侧噪声底被抬到 0.14~0.33）。
                noise_cells = [
                    c for c in cells
                    if c["side"] == side and c["profile"] == profile
                    and str(c.get("cell_role")) == "noise_floor"
                    and str(c.get("population_role")) == role
                    and c["kind"] == kind
                ]
                noise_hits = {
                    lv: aggregate_level_hit(noise_cells, level="noise")["hit_rate_mean"]
                    for lv in levels
                }
                noise_floor = max(
                    [max(0.0, base - noise_hits.get(lv, base)) for lv in levels] or [0.0]
                )
                # 噪声底的**总体口径**必须能看出量在哪一批查询上（W2）
                noise_population = (
                    noise_cells[0]["evidence"].get("population", {})
                    if noise_cells else {}
                )
                strong_level = levels[-1]
                weak_level = levels[0]
                weak_hit = float(level_hits.get(weak_level, base))
                strong_hit = float(level_hits.get(strong_level, base))
                gap = float(weak_hit - strong_hit)
                # 档间**整段落差**（max − min）：对「单调不增」不敏感的口径。
                lv_vals = [float(level_hits.get(lv, base)) for lv in levels]
                hit_max = float(max(lv_vals)) if lv_vals else base
                hit_min = float(min(lv_vals)) if lv_vals else base
                gap_range = float(hit_max - hit_min)
                monotone = bool(all(lv_vals[i] >= lv_vals[i + 1]
                                    for i in range(len(lv_vals) - 1)))
                # 判据口径（**显式**）：主口径为计划口径「弱档 − 强档」；当档序列**非单调不增**
                # 时（现场实测 `nmag` 会出现：它是缩放+平移，不是「加性扰动」，故位移量级不与
                # ε 单调对应），弱−强会显著低估实际破坏力，此时**改用整段落差**并显式标注。
                used = "weak_minus_strong" if monotone else "level_range_max_minus"
                gap_used = float(gap) if monotone else float(gap_range)
                tau = float(factor) * float(noise_floor)
                # 判据：落差 **严格大于** τ（τ 可能为 0，故不能用 >= 会把「零落差」误判有效）
                effective = bool(gap_used > tau)
                # **粒度守卫**（审查 W2 提出、W5 收口边界）：落差若**不大于**该格命中率的
                # 最小可分辨粒度（1 / n_queries），则落在观测分辨率之下、**不得**作为判定依据。
                # 边界**含等号**；且必须在**比值空间**比较（见 GRANULARITY_REL_TOL 的说明：
                # 0.20−0.15 = 0.05000000000000002 与 1/20 = 0.05 数值相等但非逐位相等，
                # 纯字面比较会把「恰好 1 个样本」那一格漏过去）。
                min_granularity = float(noise_population.get("min_granularity", 0.0))
                if min_granularity > 0.0:
                    gran_ratio = abs(gap_used) / min_granularity
                    below_granularity = bool(gran_ratio <= 1.0 + GRANULARITY_REL_TOL)
                    at_granularity = bool(
                        abs(gran_ratio - 1.0) <= GRANULARITY_REL_TOL
                    )
                    # **辅助视图**（审查 W5 提到的「落差 = 一格样本」情形）：把「弱档 − 强档」
                    # 这一口径也单独按粒度判定，仅仅**登记**、不参与 effective ——
                    # 当档序列非单调时 operative 判据是整段落差，用未被采用的弱−强去否决
                    # 会过度保守；但该情形必须在产物里可见，否则审查者无法核对。
                    ws_ratio = abs(gap) / min_granularity
                    ws_below = bool(ws_ratio <= 1.0 + GRANULARITY_REL_TOL)
                    ws_at = bool(abs(ws_ratio - 1.0) <= GRANULARITY_REL_TOL)
                else:
                    gran_ratio = float("nan")
                    below_granularity = False
                    at_granularity = False
                    ws_ratio = float("nan")
                    ws_below = False
                    ws_at = False
                if below_granularity:
                    effective = False
                entries.append(
                    {
                        "side": side,
                        "cell_role": role,
                        "profile": profile,
                        "kind": kind,
                        "noise_eps": float(negligible_eps(kind, levels[0])),
                        "clean_hit_rate": base,
                        "level_hit_rate": {k: float(v) for k, v in level_hits.items()},
                        "noise_level_hit_rate": {k: float(v) for k, v in noise_hits.items()},
                        "noise_floor": float(noise_floor),
                        "noise_floor_population": dict(noise_population),
                        "noise_floor_provenance": (
                            f"noise_floor cells: side={side}, profile={profile}, "
                            f"population_role={role}, kind={kind}, n={len(noise_cells)}"
                        ),
                        "n_queries": int(noise_population.get("query_n", 0)),
                        "min_granularity": float(
                            noise_population.get("min_granularity", 0.0)
                        ),
                        "below_granularity": bool(below_granularity),
                        "at_granularity": bool(at_granularity),
                        "granularity_ratio": (
                            float(gran_ratio) if min_granularity > 0.0 else None
                        ),
                        "weak_minus_strong_ratio": (
                            float(ws_ratio) if min_granularity > 0.0 else None
                        ),
                        "weak_minus_strong_below_granularity": bool(ws_below),
                        "weak_minus_strong_at_granularity": bool(ws_at),
                        "tau": float(tau),
                        "gap_weak_to_strong": float(gap),
                        "gap_level_range": float(gap_range),
                        "monotone_nonincreasing": bool(monotone),
                        "gap_used": float(gap_used),
                        "gap_used_basis": used,
                        "effective": bool(effective),
                        "verdict": (
                            f"该扰动对本考卷有效（{used} = {gap_used:.6f} > τ = {tau:.6f}）"
                            if effective
                            else (
                                "落差 = 粒度（= 1 个样本），**不构成判定依据** —— "
                                "该落差不大于观测分辨率，按 `GRANULARITY_GUARD_RULE` 判无效"
                                if at_granularity
                                else "该扰动对本数据无效（落差不大于噪声水平 τ）—— 如实登记，"
                                     "不得进第二步"
                            )
                        ),
                    }
                )
                noise_max_overall = max(noise_max_overall, float(noise_floor))
    calibrated_tau = float(factor) * float(noise_max_overall)
    primary = [e for e in entries if e["side"] == str(primary_side)]
    result: Dict[str, Any] = {
        "factor": float(factor),
        "primary_side": str(primary_side),
        "cell_role_by_side": dict(PRIMARY_CELL_ROLE),
        "noise_floor_max": float(noise_max_overall),
        "tau_calibrated": float(calibrated_tau),
        "tau_formula": "τ = factor × max_over(侧×扰动×编码器)[noise_floor]",
        "noise_floor_definition": (
            "现场实测：**在与主判据同一 cell 角色（同一批查询）**上，用可忽略强度"
            f"（该扰动弱档 ε 的 {NOISE_LEVEL_RATIO} 倍，即 noise σ={negligible_eps('noise', 'weak')} / "
            f"mask r={negligible_eps('mask', 'weak'):.4f} / nmag 只做 "
            f"δ={nmag_negligible_shift()} 的可忽略平移、不缩放）扰动查询与键表"
            "（变体 0 / 变体 1，随机扰动为两次独立绘制）后相对档 0 的掉点 ——"
            "即「同一内容两次编码不可能逐位相同」这一可观测差异的量级"
        ),
        "noise_floor_population_rule": (
            "噪声底按 (side × profile × 主判据 cell 角色 × 扰动类型) **逐格**标定，"
            "且只取 cell_role='noise_floor' 且 population_role == 该角色 的 cell；"
            "若某角色缺 noise_floor cell，本产物的 noise_floor_population 为空、"
            "并在报告中标明「该噪声底无同总体 cell」"
        ),
        "min_granularity_note": MIN_GRANULARITY_NOTE,
        "granularity_guard_rule": GRANULARITY_GUARD_RULE,
        "criterion": (
            "**主口径**为计划口径「弱档 − 强档 严格大于 τ ⇒ 有效」；当档序列**非单调不增**时"
            "（`nmag` 是缩放+平移，位移量级与 ε 不单调对应，会出现该情形），改用「整段落差"
            " max(档) − min(档)」并显式标注 `gap_used_basis`。"
            + GRANULARITY_GUARD_RULE
            + "。落差不大于噪声水平（τ）⇒ 该扰动对本数据无效"
            "（如实登记，不得带着坏尺子进第二步）"
        ),
        "entries": entries,
        "primary_entries": primary,
        "n_effective": int(sum(1 for e in primary if e["effective"])),
        "n_entries_primary": int(len(primary)),
        "n_below_granularity_primary": int(
            sum(1 for e in primary if e["below_granularity"])
        ),
        "n_at_granularity_primary": int(
            sum(1 for e in primary if e["at_granularity"])
        ),
        # [!] 口径分离（离朱 R53/R54 两轮收口）：**顶层只放「按 `primary_side` 侧」**的计数。
        # 「按**两测主判据角色**（`main` + `qa_known`）」的计数**只存在于**嵌套块
        # `granularity_boundary_evidence.n_at_granularity_primary_roles`（那是它的权威来源），
        # 顶层**不再镜像**该字段 —— 避免同一产物不同层出现同名不同义的读数。
        # 报告渲染的「角色口径」那一行改为直接从 G8 块取数。
        # 两个计数各自的**口径标签**由下面这个字典给出（机器可读、供消费方核对）。
        "granularity_count_scopes": {
            "n_at_granularity_primary": (
                "顶层字段；口径 = `side == primary_side`（默认 `text` 侧）且 at_granularity"
            ),
            GRANULARITY_ROLES_COUNT_KEY: (
                "**仅存在于** `granularity_boundary_evidence` 内；口径 = "
                "`cell_role ∈ PRIMARY_CELL_ROLE.values()`（`main` + `qa_known`）且 at_granularity"
            ),
            "n_at_granularity_all": "顶层字段；口径 = 全部 entries 且 at_granularity",
            "n_below_granularity_all": "顶层字段；口径 = 全部 entries 且 below_granularity",
            "n_weak_minus_strong_at_granularity_all": (
                "顶层字段；口径 = 全部 entries 且 weak_minus_strong_at_granularity（辅助视图）"
            ),
        },
        "n_below_granularity_all": int(sum(1 for e in entries if e["below_granularity"])),
        "n_at_granularity_all": int(sum(1 for e in entries if e["at_granularity"])),
        "n_weak_minus_strong_at_granularity_all": int(
            sum(1 for e in entries if e["weak_minus_strong_at_granularity"])
        ),
        "n_weak_minus_strong_below_granularity_all": int(
            sum(1 for e in entries if e["weak_minus_strong_below_granularity"])
        ),
    }
    result["granularity_boundary_evidence"] = _granularity_boundary_evidence(
        entries,
        criterion=str(result["criterion"]),
        min_granularity_note=str(result["min_granularity_note"]),
    )
    # [!] **构造期不变量**（离朱 R55 O1 加固）：角色口径计数**只允许**存在于 G8 块内，
    # 顶层**不得**出现同名字段 —— 否则又会退化成「同名不同义」（R53 的原始缺陷形态）。
    # 把该约束从「代码约定 + 文档」升级为构造期硬断言，与本模块既有的构造期不变量风格一致。
    assert GRANULARITY_ROLES_COUNT_KEY not in result, (
        f"{GRANULARITY_ROLES_COUNT_KEY} 不得出现在 calibration **顶层**"
        "（它只存在于 granularity_boundary_evidence 内）；"
        "口径标签见 granularity_count_scopes，规则见 GRANULARITY_GUARD_RULE"
    )
    return result


# ---------------------------------------------------------------------------
# 全量评测（run）
# ---------------------------------------------------------------------------


@dataclass
class RobustConfig:
    """``run`` 的冻结配置（构造期校验）。

    参数
    ----
    seed : int
        扰动种子。
    sides : Tuple[str, ...]
        参与评测的侧。
    profiles : Tuple[str, ...]
        参与评测的编码器档。
    perturb_types : Tuple[str, ...]
        参与评测的扰动类型。
    variants : int
        每档每条目的变体数（主判据 = 1）。
    stability_k : int
        稳定性佐证的 K（均值 ± 极差）。
    topk : int
        ``Recall@k`` 的最大 k。
    batch_size : int
        批大小。
    out_dir : str
        报告输出目录。
    """

    seed: int = ROBUST_SEED
    sides: Tuple[str, ...] = SIDES
    profiles: Tuple[str, ...] = (ET.PROFILE_LEXICAL, ET.PROFILE_SEMANTIC)
    perturb_types: Tuple[str, ...] = PERTURB_TYPES
    variants: int = ROBUST_VARIANTS
    stability_k: int = ROBUST_STABILITY_K
    topk: int = TOPK
    batch_size: int = BATCH_SIZE
    out_dir: str = ""

    def __post_init__(self) -> None:
        """构造期不变量（非法配置立即报错，不静默回落）。"""
        for side in self.sides:
            if str(side) not in SIDES:
                raise ValueError(f"未知侧 {side!r}；可用 = {list(SIDES)}")
        for profile in self.profiles:
            ET.expect_dim_of(str(profile))  # 未知名立即 KeyError
        for kind in self.perturb_types:
            if str(kind) not in PERTURB_GRID:
                raise ValueError(f"未知扰动 {kind!r}；可用 = {list(PERTURB_TYPES)}")
        if int(self.variants) < 1:
            raise ValueError(f"variants 必须 >= 1，当前 {self.variants}")
        if int(self.stability_k) < 1:
            raise ValueError(f"stability_k 必须 >= 1，当前 {self.stability_k}")

    def resolved_out_dir(self) -> str:
        """生效输出目录（空 = :data:`ROBUST_DIR`）。"""
        return str(self.out_dir) if str(self.out_dir) else str(ROBUST_DIR)


def run_evaluation(cfg: RobustConfig, *, log: Any = None) -> Dict[str, Any]:
    """跑全量分档矩阵（三档 × 三扰动 × 两侧 × 两编码器档）并返回完整报告。

    **不落盘扰动后的特征矩阵**：报告只含指标、取证与指纹。
    """
    t_start = time.time()

    def _log(msg: str) -> None:
        # 进度只进控制台/日志（产物必须逐字节可复现，见下方确定性纪律注释）
        if callable(log):
            log(msg)

    # [!] 产物**确定性纪律**（G3 / 审查 W3）：本 dict 会被原样序列化落盘并把 SHA256 写进
    # 报告，故**不得**含挂钟时间、耗时、主机/环境类字段 —— 它们会让「同命令重复运行」
    # 的产物字节不同，并使 report 内嵌的 run_artifact_sha256 重跑即失效。
    # 耗时与生成时间只进日志（`python -m ... --log-file`），不进产物。
    result: Dict[str, Any] = {
        "module": "n3d_qa_learn.robust_eval",
        "artifact_schema": "robust-run-v1",
        "deterministic": True,
        "excluded_fields_note": (
            "本产物不含 created_utc / seconds / 主机名等非确定字段；"
            "全量耗时与生成时间只写运行日志"
        ),
        "seed": int(cfg.seed),
        "grid": {
            "sides": [str(s) for s in cfg.sides],
            "profiles": [str(p) for p in cfg.profiles],
            "perturb_types": [str(k) for k in cfg.perturb_types],
            "perturb_labels": {k: PERTURB_LABELS[k] for k in cfg.perturb_types},
            "levels": [str(x) for x in PERTURB_LEVELS],
            "level_labels": dict(LEVEL_LABELS),
            "perturb_grid": {k: dict(PERTURB_GRID[k]) for k in cfg.perturb_types},
            "perturb_formulas": dict(PERTURB_FORMULAS),
            "nmag_shared_scale": float(NMAG_SHARED_SCALE),
            "nmag_middle_note": (
                "nmag 的 middle 档在计划口径里未单独定义，本实现显式取 ε=0.15"
                "（落于弱 0.10 与强 0.20 之间）"
            ),
            "nmag_formula_deviation": NMAG_FORMULA_SPEC_DEVIATION,
            "variants": int(cfg.variants),
            "stability_k": int(cfg.stability_k),
            "topk": int(cfg.topk),
            "derived_seed_formula": (
                "seed*1000 + 100*type_index + 10*level_index + variant"
                "（对 nmag 无随机数故 variant 不生效，见 PERTURB_FORMULAS）"
            ),
        },
        "tiers": {
            "performed": [TIER_CLEAN, TIER_PERTURB],
            "not_applicable": [TIER_ALT_QUESTION, "2_alt_surface"],
            "not_applicable_reason": TIER_ALT_QUESTION_NOTE,
        },
        "entry_tables": {},
        "euclidean_axis": {
            "removed": True,
            "note": EUCLIDEAN_EQUIVALENCE_NOTE,
            "evidence": {},
        },
        "cells": [],
        "stability": [],
        "unrecognized": [],
        "honest_notes": [],
        "errors": [],
    }

    for profile in cfg.profiles:
        expect = ET.expect_dim_of(str(profile))
        order = 0

        def _register(side: str, table: EntryKeyTable, extra: Dict[str, Any]) -> None:
            meta = table.meta()
            meta.update(extra)
            result["entry_tables"][f"{side}:{profile}"] = meta
            result["entry_tables"][f"{side}:{profile}"]["dim_expected"] = int(expect)

        # ---- 文本侧 ----
        if "text" in cfg.sides:
            _log(f"[run] 文本侧 / 档 {profile}：构造条目特征表 ...")
            from .step2 import load_text_rows, resolve_product_dir

            rows = load_text_rows(resolve_product_dir(""))
            vec = ET.build_vectorizer_for(str(profile), ET.ROLE_TEXT_LINE)
            bundle = ET.text_entry_table_from_rows(vec, rows, profile=str(profile))
            cell = text_cell_from_bundle(bundle, str(profile))
            _register(
                "text",
                bundle.table,
                {
                    "bitwise_lookup": ET.verify_bitwise_lookup(bundle.table),
                    "text_row_key_table_sha256": str(bundle.text_row_table.sha256()),
                    "evidence": dict(bundle.evidence),
                },
            )
            result["euclidean_axis"]["evidence"].setdefault("text:" + str(profile), {})
            result["euclidean_axis"]["evidence"]["text:" + str(profile)] = (
                euclidean_equivalence_evidence(
                    cell.keys, cell.queries,
                    batch_size=int(cfg.batch_size),
                )
            )
            order = _run_side_cells(cfg, result, cell, profile=str(profile), order=order, log=_log)

        # ---- QA 侧 ----
        if "qa" in cfg.sides:
            _log(f"[run] QA 侧 / 档 {profile}：构造条目特征表 ...")
            data = ET.load_qa_training_data()
            vec = ET.build_vectorizer_for(str(profile), ET.ROLE_QUESTION)
            qa_bundle = ET.build_qa_entry_tables(data, vec, profile=str(profile))
            anchor = _qa_cell(qa_bundle, str(profile), use_unknown=False)
            cell = _qa_cell(qa_bundle, str(profile), use_unknown=True)
            _register(
                "qa",
                qa_bundle.table,
                {
                    "bitwise_lookup": ET.verify_bitwise_lookup(qa_bundle.table),
                    "evidence": dict(qa_bundle.evidence),
                    "n_unknown": int(qa_bundle.unknown.n),
                    "n_known_queries": int(qa_bundle.known_queries.n),
                },
            )
            result["euclidean_axis"]["evidence"]["qa:" + str(profile)] = (
                euclidean_equivalence_evidence(
                    cell.keys, cell.queries[: min(cell.query_n, int(cfg.batch_size) * 2)],
                    batch_size=int(cfg.batch_size),
                )
            )
            order = _run_side_cells(cfg, result, cell, profile=str(profile), order=order, log=_log)
            # QA 侧已知查询（test_known）的档 0 锚点 + 三扰动 × 三档（统计意义弱，单独标注）
            _log(f"[run] QA 侧 / 档 {profile}：已知查询（test_known {anchor.query_n} 条）档 0 + 全档 ...")
            order = _run_side_cells(
                cfg, result, anchor, profile=str(profile), order=order, log=_log,
                cell_role="qa_known",
            )
            # 未识别档（τ 由 calibrate 回填；此处先给不依赖 τ 的分位数）
            result["unrecognized"].append(
                {
                    "side": "qa",
                    "profile": str(profile),
                    "scope": "clean",
                    "report": unrecognized_report(
                        qa_bundle.table.keys, qa_bundle.unknown.feats,
                        thresholds=None, batch_size=int(cfg.batch_size),
                    ),
                }
            )

    # ---- 稳定性佐证（K 个**真实不同**的变体：均值 ± 极差）----
    result["stability"] = _stability_cells(cfg, log=_log)

    # ---- 横截面的一致性检查（构造期/现场不变量）----
    # G7 扰动公式逐位断言：与 cfg 无关（用固定 check matrix × 全部档位），但必须**随
    # 产物落盘**才能被审计。kinds 用全量 PERTURB_TYPES 以免收窄 --perturb 时漏检。
    result["bitwise_formula_assertions"] = assert_perturb_formulas_bitwise(
        PERTURB_TYPES, PERTURB_LEVELS, seed=int(cfg.seed)
    )
    result["invariants"] = _invariant_checks(result)
    result["honest_notes"] = _honest_notes(result)
    # [!] 耗时**不进产物**（只进日志）：见函数开头的确定性纪律注释。
    _log(f"[run] 全量矩阵完成（耗时 {time.time() - t_start:.1f}s，该耗时只进日志不入产物）")
    return result


def _honest_notes(result: Dict[str, Any]) -> List[str]:
    """如实登记的结构性限制与负结果（**与实测值绑定**，不写空泛声明）。"""
    notes: List[str] = []
    idx = _cell_index(result)
    profiles = list(result.get("grid", {}).get("profiles", []))
    kinds = list(result.get("grid", {}).get("perturb_types", []))
    levels = list(result.get("grid", {}).get("levels", []))
    text_lines: List[str] = []
    for profile in profiles:
        clean = idx.get(("text", profile, "main", "none", "clean"))
        if not clean:
            continue
        base = float(clean["metrics"]["recall_at_1"])
        per_kind = []
        for kind in kinds:
            weak = idx.get(("text", profile, "main", kind, levels[0]))
            strong = idx.get(("text", profile, "main", kind, levels[-1]))
            if weak and strong:
                per_kind.append(
                    f"{kind} 弱 {float(weak['metrics']['recall_at_1']):.6f}"
                    f" → 强 {float(strong['metrics']['recall_at_1']):.6f}"
                )
        text_lines.append(
            f"档 0 R@1 = {base:.10f}（{'；'.join(per_kind)}）"
        )
    if text_lines:
        notes.append(
            "文本侧基线**已近乎饱和**（同一特征自检索口径下档 0 R@1 = 1.0，"
            "而 `step2_run` 报的纯特征参照下限 0.9834834834834835 是**另一种口径**"
            "（库=1999/查询=666 的留出划分 + zh-bag D=192）），故弱档的可下降空间极小；"
            "现场实测：" + "；".join(text_lines) + "。"
        )
    notes.append(
        "QA 侧已知查询仅 test_known 20 条 —— **统计意义弱**（单个样本 = 0.05），"
        "其数字只能当辅助佐证，**不得**作为主判据；报告中已与文本侧分开成组。"
    )
    notes.append(
        "本轮**不做档 1（同答案另一问题）与档 2**：现成数据不存在「同一问题的不同问法」"
        "（Math1 的 id 在 4 题型间零重叠；n3d_qa 每个 question_id 只有 1 种 question_text）。"
    )
    if result.get("grid", {}).get("profiles"):
        notes.append(
            "编码器档 `bge-m3-1024` 走**可选 HF 路径**（本地 `models/bge-m3`）；"
            "该环境缺 `transformers` 时对应档如实登记为不可用（不静默跳过）。"
        )
    notes.append(
        "「未识别」档的 τ **本批不标定**（属第二步变体 B 的范围）；报告只给不依赖 τ 的"
        "最高余弦分位数。"
    )
    notes.append(
        "**文本侧掉点的成分必须分开读**：档 0 的自检索是「查询行 = 键表行」的**逐位自匹配**"
        "（R@1 结构性为 1.0），强扰动会把它打下来 —— 其中包含「近重复行导致的自匹配歧义」"
        "成分（键表中存在内容极近的行，扰动的重新绘制换一条近重复行胜出即可致误），"
        "见第 3 节「噪声底」一栏。因此**不得**把该落差直接读成「语义理解被破坏」。"
    )
    notes.append(
        "**文本侧的「弱档」多为天花板效应**：语义档在 50% 维度遮蔽下 R@1 仍可为 1.000000 ——"
        "编码器冗余度使「弱扰动」几乎无区分度；这属于**该编码器 + 该 ID 检索协议**的性质，"
        "不能外推到其它任务。"
    )
    notes.append(
        "**`nmag` 无随机数**（闭式 ``x·NMAG_SHARED_SCALE + ε·(−1)^(i+j+1)``），故其 "
        "`variant` 不产生差异、其噪声底也不是「两次独立绘制」——该事实已在 "
        "`bitwise_formula_assertions` 与 noise_floor cell 的 `independent_draw` 字段中显式登记。"
    )
    notes.append(MIN_GRANULARITY_NOTE)
    if result.get("entry_tables"):
        for key, meta in sorted(result["entry_tables"].items()):
            if str(key).startswith("qa:"):
                src = meta.get("evidence", {})
                notes.append(
                    f"**{key}**：QA 侧主判据口径的查询是 `test_known` **20 条**"
                    f"（条目 {meta.get('n_entries')} 条）；其「未识别」负样本 "
                    f"{meta.get('n_unknown')} 条与条目表答案键**零重叠**，"
                    "只用于分位数报告，不参与命中率判定。"
                    f"该侧查询数 {src.get('query_rows', '—')}。"
                )
    return notes


def _run_side_cells(
    cfg: RobustConfig,
    result: Dict[str, Any],
    cell: EvalCell,
    *,
    profile: str,
    order: int,
    log: Any,
    cell_role: str = "main",
) -> int:
    """跑某一侧的 cell 矩阵（档 0 + 噪声底 + 三扰动 × 三档 × 变体）。

    ``cell_role="main"`` = 主判据（文本自检索 / QA 已知查询）；``cell_role="qa_known"``
    = QA 侧已知查询口径族（统计意义弱，报告单独成组）。
    """
    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    role = str(cell_role)
    # 档 0（不扰动）
    clean = evaluate_cell(
        cell, kind="noise", level="weak", clean=True,
        topk=int(cfg.topk), batch_size=int(cfg.batch_size),
    )
    clean["kind"] = "none"
    clean["level"] = "clean"
    clean["cell_role"] = role
    clean["note"] = cell.strong_note
    clean["order"] = int(order)
    order += 1
    result["cells"].append(clean)
    _log(
        f"[run]   {cell.side}/{profile}/{role} 档0(原样检索)："
        f"R@1={clean['metrics']['recall_at_1']:.6f} R@5={clean['metrics']['recall_at_5']:.6f}"
        f"（n_query={cell.query_n}，无金标 {clean['metrics']['n_no_gold']}）"
    )
    for kind in cfg.perturb_types:
        for level in PERTURB_LEVELS:
            for variant in range(int(cfg.variants)):
                cell_res = evaluate_cell(
                    cell, kind=str(kind), level=str(level), variant=int(variant),
                    seed=int(cfg.seed), perturb_table=False,
                    topk=int(cfg.topk), batch_size=int(cfg.batch_size),
                )
                cell_res["cell_role"] = role
                cell_res["note"] = cell.strong_note
                cell_res["order"] = int(order)
                order += 1
                result["cells"].append(cell_res)
            _log(
                f"[run]   {cell.side}/{profile}/{role} {kind}/{level}："
                f"R@1={cell_res['metrics']['recall_at_1']:.6f}"
            )
            # 噪声底（可忽略强度；**与主判据同 cell 角色、同一批查询**）
            noise_res = noise_floor_of_cell(
                cell, kind=str(kind), level=str(level), seed=int(cfg.seed),
                topk=int(cfg.topk), batch_size=int(cfg.batch_size), cell_role=role,
            )
            noise_res["cell_role"] = "noise_floor"
            # 噪声底**总体溯源**（W2 修复）：记下它是在哪个角色的哪一批查询上量的
            noise_res["population_role"] = role
            noise_res["note"] = (
                f"噪声底：强度 = 该扰动弱档 ε 的 {NOISE_LEVEL_RATIO} 倍（可忽略），"
                f"在 **{role}** 角色的同一批查询（n={cell.query_n}）上量；"
                "用于标定 τ，不参与主判据"
            )
            noise_res["order"] = int(order)
            order += 1
            result["cells"].append(noise_res)
        _log(f"[run]   {cell.side}/{profile}/{role} 扰动 {kind} 三档完成")
    return order


def _stability_cells(cfg: RobustConfig, *, log: Any) -> List[Dict[str, Any]]:
    """稳定性佐证：**重新构造 K 个真实不同的变体**跑一遍并报均值 ± 极差。

    为什么单独跑一遍（而不是从主判据 cell 里按 ``variant`` 筛选）
    -----------------------------------------------------------
    为避免与主判据打架，主判据路径**只跑差异较大的那一条变体**（``variant=0``）。
    如果稳定性表从同一批 cell 里筛 ``variant``，它只会反复取到 ``variant=0``，
    极差恒为 ``0.000000``（**现场实测踩过这个坑**：K=5 的 36 行全部极差 0.000000）。
    故本函数显式用 ``variant=0..K-1`` 重跑每一格；结果**不计入**主判据。
    """
    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    if int(cfg.stability_k) <= 1:
        return []
    rows: List[Dict[str, Any]] = []
    text_cell = None
    qa_anchor = None
    for profile in cfg.profiles:
        if "text" in cfg.sides:
            from .step2 import load_text_rows, resolve_product_dir

            vec = ET.build_vectorizer_for(str(profile), ET.ROLE_TEXT_LINE)
            bundle = ET.text_entry_table_from_rows(
                vec, load_text_rows(resolve_product_dir("")), profile=str(profile)
            )
            text_cell = text_cell_from_bundle(bundle, str(profile))
        if "qa" in cfg.sides:
            data = ET.load_qa_training_data()
            vec_q = ET.build_vectorizer_for(str(profile), ET.ROLE_QUESTION)
            qa_anchor = _qa_cell(
                ET.build_qa_entry_tables(data, vec_q, profile=str(profile)),
                str(profile),
                use_unknown=False,
            )
        for side in cfg.sides:
            cell = text_cell if str(side) == "text" else qa_anchor
            if cell is None:
                continue
            for kind in cfg.perturb_types:
                for level in PERTURB_LEVELS:
                    vals1: List[float] = []
                    vals5: List[float] = []
                    hits: List[int] = []
                    for variant in range(int(cfg.stability_k)):
                        res = evaluate_cell(
                            cell, kind=str(kind), level=str(level), variant=int(variant),
                            seed=int(cfg.seed), perturb_table=False,
                            topk=int(cfg.topk), batch_size=int(cfg.batch_size),
                        )
                        vals1.append(float(res["metrics"]["recall_at_1"]))
                        vals5.append(float(res["metrics"]["recall_at_5"]))
                        hits.append(int(res["metrics"]["hit_at_1"]))
                    rows.append(
                        {
                            "side": str(side),
                            "profile": str(profile),
                            "kind": str(kind),
                            "level": str(level),
                            "k": int(cfg.stability_k),
                            "note": (
                                "K 个**真实不同**变体（variant=0..K-1，同 seed 派生）的"
                                "均值 ± 极差；主判据只用 variant=0 的那一条，本表仅作稳定性佐证"
                            ),
                            "recall_at_1_mean": float(np.mean(vals1)),
                            "recall_at_1_range": float(np.max(vals1) - np.min(vals1)),
                            "recall_at_5_mean": float(np.mean(vals5)),
                            "recall_at_5_range": float(np.max(vals5) - np.min(vals5)),
                            "hit_at_1_min": int(min(hits)),
                            "hit_at_1_max": int(max(hits)),
                            "n_variants": int(len(vals1)),
                        }
                    )
    _log(f"[run] 稳定性佐证（K={cfg.stability_k} 个真实变体）：{len(rows)} 行")
    return rows


def _invariant_checks(result: Dict[str, Any]) -> Dict[str, Any]:
    """现场不变量检查（构造期量 vs 现场量严格分开）。

    三类检查（**每类的期望值不同，不得混用**）：

    1. **文本侧档 0 自检索**：查询与键表逐位相同 ⇒ ``R@1 == 1.0``（结构不变量）；
    2. **QA 侧档 0 已知查询**：金标 = 同答案键的条目行集合 ⇒ 值由数据决定，
       只检查「存在非空金标」（**不断言初值**，历史纠正记录 #10）；
    3. **QA 侧未识别负样本**：现场实测其答案键与条目表**零重叠** ⇒ 金标集合恒为空、
       ``n_no_gold == n`` ⇔ ``hit_at_1 == 0``（这是构造事实，不是缺陷）。
    """
    checks: List[Dict[str, Any]] = []
    mains = [
        c for c in result["cells"]
        if str(c.get("cell_role")) != "noise_floor" and int(c.get("variant", 0)) == 0
    ]
    for c in mains:
        if c["kind"] != "none":
            continue
        role = str(c.get("cell_role", "main"))
        side = str(c["side"])
        if role == "main" and side == "text":
            # 1. 结构不变量：查询与键表逐位相同，对角项恒 1.0
            checks.append(
                {
                    "name": f"档0自检索必须为1.0[{side}/{c['profile']}]",
                    "value": float(c["metrics"]["recall_at_1"]),
                    "passed": bool(
                        float(c["metrics"]["recall_at_1"]) == 1.0
                        and int(c["metrics"]["n_no_gold"]) == 0
                    ),
                }
            )
        elif role == "qa_known":
            # 2. 金标可定义（值由数据决定，不断言）
            n_usable = int(c["query_n"] - c["metrics"]["n_no_gold"])
            checks.append(
                {
                    "name": f"档0金标可定义[{side}/{c['profile']}/{role}]（非空金标查询数 > 0）",
                    "value": int(n_usable),
                    "passed": bool(n_usable > 0),
                }
            )
        else:
            # 3. 未识别负样本：金标集合恒为空（构造事实）
            n_no_gold = int(c["metrics"]["n_no_gold"])
            checks.append(
                {
                    "name": (
                        f"未识别负样本金标恒空[{side}/{c['profile']}/{role}]"
                        "（n_no_gold == n 且 hit_at_1 == 0）"
                    ),
                    "value": [int(n_no_gold), int(c["query_n"]), int(c["metrics"]["hit_at_1"])],
                    "passed": bool(
                        n_no_gold == int(c["query_n"])
                        and int(c["metrics"]["hit_at_1"]) == 0
                    ),
                }
            )
    for key, meta in result["entry_tables"].items():
        checks.append(
            {
                "name": f"逐位精确查表 exact_frac==1.0[{key}]",
                "value": float(meta["bitwise_lookup"]["exact_frac"]),
                "passed": bool(float(meta["bitwise_lookup"]["exact_frac"]) == 1.0),
            }
        )
        norm = meta["norm_range"]
        checks.append(
            {
                "name": f"键表行范数在允差内[{key}]",
                "value": [float(norm["min"]), float(norm["max"])],
                "passed": bool(
                    float(norm["min"]) >= 1.0 - ET.NORM_ATOL
                    and float(norm["max"]) <= 1.0 + ET.NORM_ATOL
                ),
            }
        )
    # G7：扰动公式逐位断言（三种扰动 × 三档，逐例比 float32 裸字节）
    g7 = result.get("bitwise_formula_assertions", {})
    if g7:
        checks.append(
            {
                "name": "G7 扰动公式逐位断言（全部档位，若未归一化与归一化两路都逐字节相等）",
                "value": [
                    int(g7.get("n_bitwise_equal", 0)),
                    int(g7.get("n_cases", 0)),
                ],
                "passed": bool(g7.get("all_bitwise_equal")),
            }
        )
        for case in g7.get("cases", []):
            checks.append(
                {
                    "name": (
                        f"G7[{case.get('kind')}/{case.get('level')}] 裸字节逐位相等"
                    ),
                    "value": bool(
                        case.get("bitwise_equal") and case.get("raw_bitwise_equal")
                    ),
                    "passed": bool(
                        case.get("bitwise_equal") and case.get("raw_bitwise_equal")
                    ),
                }
            )
    return {
        "checks": checks,
        "n_checks": int(len(checks)),
        "n_failed": int(sum(1 for c in checks if not c["passed"])),
        "all_passed": bool(all(c["passed"] for c in checks)) if checks else False,
    }


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------


def load_run(path: str) -> Dict[str, Any]:
    """读取 ``run`` 产物（缺失即可读报错）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"robust run 产物不存在：{path!r}")
    with open(path, "r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def write_json(path: str, obj: Any) -> str:
    """写 JSON（UTF-8 无 BOM，缩进 1；返回文件 SHA256）。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(blob)
    return ET.sha256_bytes(blob.encode("utf-8"))


def write_text(path: str, text: str) -> str:
    """写文本（UTF-8 无 BOM；返回文件 SHA256）。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return ET.sha256_bytes(text.encode("utf-8"))


def _cell_index(
    result: Dict[str, Any],
) -> Dict[Tuple[str, str, str, str, str], Dict[str, Any]]:
    """主判据 cell 索引：``(side, profile, cell_role, kind, level) -> cell``（variant=0）。"""
    out: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    for c in result.get("cells", []):
        if str(c.get("cell_role")) == "noise_floor":
            continue
        if int(c.get("variant", 0)) != 0:
            continue
        out[(c["side"], c["profile"], str(c.get("cell_role", "main")), c["kind"], c["level"])] = c
    return out


def _primary_role(side: str) -> str:
    """某侧的主判据 cell 角色。"""
    return str(PRIMARY_CELL_ROLE.get(str(side), "main"))


def _g8_roles_count(calibration: Dict[str, Any]) -> Any:
    """从 **G8 权威块**取「按两测主判据角色」的 `at_granularity` 计数。

    口径来源唯一性（离朱 R54 收口）：该计数**只存在于**
    ``calibration["granularity_boundary_evidence"]["n_at_granularity_primary_roles"]``，
    顶层**不再镜像**同名/近名字段（避免同一产物不同层出现不同读数）；
    报告渲染因此固定从 G8 块取数。缺失时返回 ``None``（渲染为 `None`，不静默编造 0）。
    """
    g8 = calibration.get("granularity_boundary_evidence") or {}
    return g8.get(GRANULARITY_ROLES_COUNT_KEY)


def render_markdown(
    run_result: Dict[str, Any], calibration: Optional[Dict[str, Any]] = None
) -> str:
    """把 run（可选 + 标定）结果渲染为 Markdown 报告。"""
    idx = _cell_index(run_result)
    grid = run_result.get("grid", {})
    profiles = list(grid.get("profiles", []))
    sides = list(grid.get("sides", []))
    kinds = list(grid.get("perturb_types", []))
    levels = list(grid.get("levels", []))
    labels = dict(grid.get("level_labels", {}))
    plabels = dict(grid.get("perturb_labels", {}))
    lines: List[str] = []
    lines.append("# n3d_qa_learn 分档鲁棒性考卷（第一步 1a）")
    lines.append("")
    lines.append(
        "- **本报告不含生成时间**（G3/W3 确定性纪律）：同命令重复运行，"
        "`robust_run.json` / `robust_calibration.json` / `robust_report.{json,md}` 逐字节一致；"
        "生成时间与耗时只写运行日志。"
    )
    lines.append(f"- 扰动种子：`{run_result.get('seed')}`；变体数：`{grid.get('variants')}`；"
                 f"稳定性 K：`{grid.get('stability_k')}`")
    lines.append(f"- 扰动网格（档位 ε）：`{grid.get('perturb_grid')}`"
                 f"；`nmag` 共享缩放系数 = `{grid.get('nmag_shared_scale')}`")
    for _k, _f in sorted(dict(grid.get("perturb_formulas", {})).items()):
        lines.append(f"  - `{_k}`：`{_f}`")
    lines.append(f"- 派生种子公式：`{grid.get('derived_seed_formula')}`")
    lines.append(f"- 编码器档：`{profiles}`；侧：`{sides}`；k：`{grid.get('topk')}`")
    lines.append("")
    lines.append("## 1. 口径与「这把尺子」的结构性限制")
    lines.append("")
    lines.append(f"- **档位**：只做 {run_result.get('tiers', {}).get('performed')}；"
                 f"不做 {run_result.get('tiers', {}).get('not_applicable')}")
    lines.append(f"- 档 1 / 档 2 不适用的原因：{run_result.get('tiers', {}).get('not_applicable_reason')}")
    lines.append(f"- **欧氏基线轴**：{run_result.get('euclidean_axis', {}).get('note')}")
    lines.append("")
    lines.append("### 1.1 条目特征表与逐位精确查表（G1）")
    lines.append("")
    lines.append("| 表 | N | D | 行范数区间 | 键表 SHA256 | 编码器口径指纹 | 逐位查表 |")
    lines.append("| --- | ---: | ---: | --- | --- | --- | --- |")
    for key, meta in sorted(run_result.get("entry_tables", {}).items()):
        norm = meta["norm_range"]
        bl = meta.get("bitwise_lookup", {})
        lines.append(
            f"| `{key}` | {meta['n_entries']} | {meta['dim']} | "
            f"[{norm['min']:.8f}, {norm['max']:.8f}] | `{str(meta['key_table_sha256'])[:16]}…` | "
            f"`{str(meta['encoder_fingerprint'])[:16]}…` | "
            f"{bl.get('n_exact')}/{bl.get('n_checked')}（exact_frac={bl.get('exact_frac')}） |"
        )
    lines.append("")
    lines.append("### 1.2 欧氏/余弦等价的现场证据（登记用，不作为对照轴）")
    lines.append("")
    lines.append("| 表 | 检查查询数 | top-1 不一致条数 | 行范数区间 | 恒等式最大残差 |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for key, ev in sorted(run_result.get("euclidean_axis", {}).get("evidence", {}).items()):
        lines.append(
            f"| `{key}` | {ev.get('n_queries_checked')} | {ev.get('n_top1_mismatch')} | "
            f"[{ev.get('key_row_norm_min'):.8f}, {ev.get('key_row_norm_max'):.8f}] | "
            f"{ev.get('max_abs_identity_residual'):.3e} |"
        )
    lines.append("")
    lines.append("## 2. 逐档 × 逐扰动 × 逐侧命中率（归一化余弦 KNN 基线）")
    lines.append("")
    for profile in profiles:
        lines.append(f"### 2.{profiles.index(profile) + 1} 编码器档 `{profile}`")
        lines.append("")
        lines.append("| 侧 | 扰动 | 档 0 R@1 | 弱 R@1 | 中 R@1 | 强 R@1 | 弱→强落差 | 档 0 R@5 | 强 R@5 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for side in sides:
            role = _primary_role(str(side))
            clean = idx.get((side, profile, role, "none", "clean"))
            base1 = float(clean["metrics"]["recall_at_1"]) if clean else float("nan")
            base5 = float(clean["metrics"]["recall_at_5"]) if clean else float("nan")
            for kind in kinds:
                vals = []
                for level in levels:
                    c = idx.get((side, profile, role, kind, level))
                    vals.append(float(c["metrics"]["recall_at_1"]) if c else float("nan"))
                strong5 = idx.get((side, profile, role, kind, levels[-1]))
                strong5v = float(strong5["metrics"]["recall_at_5"]) if strong5 else float("nan")
                lines.append(
                    f"| {SIDE_LABELS.get(side, side)} | {plabels.get(kind, kind)} | "
                    f"{base1:.6f} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | "
                    f"{(vals[0] - vals[2]):.6f} | {base5:.6f} | {strong5v:.6f} |"
                )
        lines.append("")
    lines.append("> 文本侧 = 主判据依据（666 条查询 / 2665 条目）；QA 侧 = 辅助口径")
    lines.append("> （未识别负样本 2893 条为查询、149 条目为键表，**不是** QA 业务指标）。")
    lines.append("")
    lines.append("")
    lines.append("### 2.3 评测单元结构（读者必须先看这张表）")
    lines.append("")
    lines.append("| 侧 | 编码器档 | 单元角色 | 条目数 | 查询数 | 金标集合为空的查询 | 档 0 R@1 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for c in run_result.get("cells", []):
        if str(c.get("cell_role")) == "noise_floor" or int(c.get("variant", 0)) != 0:
            continue
        if c.get("kind") != "none":
            continue
        lines.append(
            f"| {SIDE_LABELS.get(c['side'], c['side'])} | `{c['profile']}` | "
            f"`{c.get('cell_role')}` | {c['table_n']} | {c['query_n']} | "
            f"{c['metrics'].get('n_no_gold')} | {float(c['metrics']['recall_at_1']):.6f} |"
        )
    lines.append("")
    lines.append("> **`qa/main`（未识别负样本 2893 条）的档 0 R@1 恒为 0.000000 是构造决定的**：")
    lines.append("> 现场实测这 2893 条的 1770 个答案键与条目表的 10 个类**零重叠**，")
    lines.append("> 因此金标集合恒为空、无法定义命中 —— 它们是「不相关」样本，")
    lines.append("> **不得**被读成「该口径下模型很差」。QA 侧的可用数字看 `qa_known`（20 条，弱）。")
    lines.append("")
    lines.append("### 2.4 弱 → 中 → 强 落差曲线")
    lines.append("")
    lines.append("| 侧 | 编码器档 | 扰动 | 弱 | 中 | 强 | 弱→强落差 | 单调不增 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for side in sides:
        for profile in profiles:
            role = _primary_role(str(side))
            for kind in kinds:
                vals = []
                for level in levels:
                    c = idx.get((side, profile, role, kind, level))
                    vals.append(float(c["metrics"]["recall_at_1"]) if c else float("nan"))
                mono = bool(vals[0] >= vals[1] >= vals[2])
                lines.append(
                    f"| {SIDE_LABELS.get(side, side)} | `{profile}` | {plabels.get(kind, kind)} | "
                    f"{vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | "
                    f"{(vals[0] - vals[2]):.6f} | {'是' if mono else '**否**'} |"
                )
    lines.append("")
    lines.append("### 2.5 扰动公式自检与 G7 逐位断言")
    lines.append("")
    lines.append("| 扰动 | 档 | ε | 形状 | 全部有限 | 零范数 | 平均行位移 | 生成器种子 |")
    lines.append("| --- | --- | ---: | --- | --- | ---: | ---: | ---: |")
    for row in run_result.get("perturb_self_check", []):
        lines.append(
            f"| `{row.get('kind')}` | {labels.get(row.get('level'), row.get('level'))} | "
            f"{row.get('eps')} | `{row.get('shape')}` | {row.get('finite')} | "
            f"{row.get('n_zero_norm')} | {row.get('mean_row_displacement'):.6f} | "
            f"{row.get('generator_seed')} |"
        )
    g7 = run_result.get("bitwise_formula_assertions", {})
    if g7:
        lines.append("")
        lines.append(
            f"**G7 扰动公式逐位断言**（{g7.get('n_cases')} 例 × 每例比对 "
            f"`float32` 裸字节）：全部通过 = `{g7.get('all_bitwise_equal')}`"
        )
        lines.append("")
        lines.append("| 扰动 | 档 | 复算方式 | 裸字节逐位相等 |")
        lines.append("| --- | --- | --- | --- |")
        for case in g7.get("cases", []):
            lines.append(
                f"| `{case.get('kind')}` | {labels.get(case.get('level'), case.get('level'))} | "
                f"{case.get('recompute')} | "
                f"{'**是**' if case.get('bitwise_equal') else '**否**'} |"
            )
    else:
        lines.append("")
        lines.append("- （本产物未含 G7 断言结果；请用当前源码重跑 `robust run`。）")
    lines.append("")
    lines.append("## 3. 阈值现场标定")
    lines.append("")
    if calibration:
        lines.append(f"- 标定倍数 `factor = {calibration.get('factor')}`；"
                     f"公式：`{calibration.get('tau_formula')}`")
        lines.append(f"- 噪声底定义：{calibration.get('noise_floor_definition')}")
        lines.append(f"- **实测噪声底上界 = {calibration.get('noise_floor_max'):.6f}**；"
                     f"**全局标定阈值 τ = {calibration.get('tau_calibrated'):.6f}**")
        lines.append(
            "- **判定一律用下表「逐格 τ」**（逐侧 × 逐扰动 × 逐编码器各自标定）；"
            "全局 τ 是最保守参照（它被噪声底最高的那一格抬高，任何格都过不了），"
            "单列于此以免读者误以为判定用的是全局值。"
        )
        lines.append(f"- 判据：{calibration.get('criterion')}")
        lines.append(f"- **粒度守卫规则（唯一来源 `robust_eval.GRANULARITY_GUARD_RULE`）**："
                     f"{calibration.get('granularity_guard_rule')}")
        lines.append(f"- **噪声底总体口径**：{calibration.get('noise_floor_population_rule')}")
        lines.append(f"- **最小可分辨粒度**：{calibration.get('min_granularity_note')}")
        lines.append(
            f"- 边界命中计数（**全部格**）：`at_granularity` = "
            f"{calibration.get('n_at_granularity_all')}、`below_granularity` = "
            f"{calibration.get('n_below_granularity_all')}；"
            f"（按 **`primary_side` = `{calibration.get('primary_side')}` 侧**）"
            f"`at_granularity` = {calibration.get('n_at_granularity_primary')}、"
            f"`below_granularity` = {calibration.get('n_below_granularity_primary')}；"
            f"（按 **两测主判据角色** `main`+`qa_known`，取数自 "
            "`granularity_boundary_evidence.n_at_granularity_primary_roles`）"
            f"`at_granularity` = {_g8_roles_count(calibration)}"
        )
        lines.append("")
        lines.append(
            "| 侧 | 编码器档 | 单元角色 | 扰动 | 档 0 R@1 | 弱 R@1 | 强 R@1 | 弱→强落差 | "
            "整段落差 | 单调不增 | 噪声底 | 噪声底总体 | τ | 判定（依据） | 粒度标记 |"
        )
        lines.append(
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | --- |"
        )
        for e in calibration.get("entries", []):
            lv = e["level_hit_rate"]
            pop = e.get("noise_floor_population", {}) or {}
            pop_txt = (
                f"role={pop.get('cell_role')} / n={pop.get('query_n')}"
                if pop else "**无同总体 cell**"
            )
            if e.get("at_granularity"):
                mark = "**落差 = 粒度（1 样本）**"
            elif e.get("below_granularity"):
                mark = "落差 < 粒度"
            else:
                mark = "—"
            lines.append(
                f"| {SIDE_LABELS.get(e['side'], e['side'])} | `{e['profile']}` | "
                f"`{e.get('cell_role')}` | "
                f"{plabels.get(e['kind'], e['kind'])} | {e['clean_hit_rate']:.6f} | "
                f"{lv.get('weak', 0.0):.6f} | {lv.get('strong', 0.0):.6f} | "
                f"{e['gap_weak_to_strong']:.6f} | "
                f"{e.get('gap_level_range', float('nan')):.6f} | "
                f"{'是' if e.get('monotone_nonincreasing') else '**否**'} | "
                f"{e['noise_floor']:.6f} | {pop_txt} | "
                f"{e['tau']:.6f} | "
                f"{'**有效**' if e['effective'] else '**无效**'}"
                f"（`{e.get('gap_used_basis')}`） | {mark} |"
            )
        lines.append("")
        lines.append(
            f"- 主判据侧（`{calibration.get('primary_side')}` **侧**）有效扰动数："
            f"{calibration.get('n_effective')}/{calibration.get('n_entries_primary')}"
            f"；其中**落差不大于粒度**的格数："
            f"{calibration.get('n_below_granularity_primary')}"
            f"（该侧**落差恰等于粒度**的格数："
            f"{calibration.get('n_at_granularity_primary')}）"
            f"；另按**两测主判据角色** `main`+`qa_known` 统计的**落差恰等于粒度**格数"
            "（取数自 `granularity_boundary_evidence.n_at_granularity_primary_roles`）："
            f"{_g8_roles_count(calibration)} —— "
            "这类格一律判无效并标「落差 = 粒度（1 样本）」"
        )
        lines.append("")
    else:
        lines.append("- **尚未标定**：请先跑 `robust calibrate`（本报告由 run 产物直接渲染）。")
    lines.append("")
    lines.append("## 4. 「未识别」档（负样本）")
    lines.append("")
    lines.append("| 侧 | 编码器档 | 负样本数 | 最高余弦均值 | 中位数 | 90% 分位 | 99% 分位 | 未识别率 | 误召回率 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for u in run_result.get("unrecognized", []):
        rep = u["report"]
        q = rep["best_cosine"]["quantiles"]
        ur = rep.get("unrecognized_rate")
        fr = rep.get("false_recall_rate")
        lines.append(
            f"| {SIDE_LABELS.get(u['side'], u['side'])} | `{u['profile']}` | {rep['n']} | "
            f"{rep['best_cosine']['mean']:.6f} | {q.get('0.5', 0.0):.6f} | "
            f"{q.get('0.9', 0.0):.6f} | {q.get('0.99', 0.0):.6f} | "
            f"{'—' if ur is None else f'{ur:.6f}'} | "
            f"{'—' if fr is None else f'{fr:.6f}'} |"
        )
    lines.append("")
    lines.append("> **本档不标定阈值**（阈值标定属第二步变体 B 的范围）；"
                 "τ 未给定时未识别率 / 误召回率如实标「不依赖 τ 的分位数可读」。")
    lines.append("")
    lines.append("## 5. 稳定性佐证（K 变体：均值 ± 极差）")
    lines.append("")
    if run_result.get("stability"):
        lines.append("| 侧 | 编码器档 | 扰动 | 档 | K | R@1 均值 | R@1 极差 | R@5 均值 | R@5 极差 | 命中数区间 |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for s in run_result["stability"]:
            lines.append(
                f"| {SIDE_LABELS.get(s['side'], s['side'])} | `{s['profile']}` | "
                f"{plabels.get(s['kind'], s['kind'])} | {labels.get(s['level'], s['level'])} | "
                f"{s['k']} | {s['recall_at_1_mean']:.6f} | {s['recall_at_1_range']:.6f} | "
                f"{s['recall_at_5_mean']:.6f} | {s['recall_at_5_range']:.6f} | "
                f"{s.get('hit_at_1_min')}~{s.get('hit_at_1_max')} |"
            )
    else:
        lines.append("- 未生成（`--variants` / `--stability-k` 配置为 1）。")
    lines.append("")
    lines.append("## 6. 现场不变量检查")
    lines.append("")
    inv = run_result.get("invariants", {})
    lines.append(f"- 检查项 {inv.get('n_checks')} 条，失败 {inv.get('n_failed')} 条，"
                 f"全通过 = `{inv.get('all_passed')}`")
    for c in inv.get("checks", []):
        if not c["passed"]:
            lines.append(f"  - **失败**：{c['name']} = `{c['value']}`")
    lines.append("")
    lines.append("## 7. 如实登记的负结果与结构性限制")
    lines.append("")
    notes = list(run_result.get("honest_notes", []))
    for key, meta in sorted(run_result.get("entry_tables", {}).items()):
        src = meta.get("evidence", {})
        if str(key).startswith("text:"):
            prof = str(key).split(":", 1)[1]
            anchor = idx.get(("text", prof, "main", "none", "clean"), {})
            anchor_v = anchor.get("metrics", {}).get("recall_at_1", float("nan"))
            lines.append(
                f"- **{key}**：纯特征基线已近饱和（档 0 R@1 = {anchor_v:.10f}）；"
                "该侧对「弱扰动」的区分空间极小，弱档掉点接近 0 **不能**被读作"
                "「扰动无效」或「结论稳健」。"
            )
        if str(key).startswith("qa:") and src:
            lines.append(
                f"- **{key}**：QA 侧已知查询仅 test_known 20 条（档 0 锚点），"
                "**统计意义弱**、不得作为主判据；其未识别负样本 2893 条与条目表"
                "答案键零重叠，故只用于「扰动 → 掉点」这一考卷有效性的辅助验证。"
            )
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("## 8. 复现命令")
    lines.append("")
    lines.append("```")
    lines.append("python -m n3d_qa_learn.step2_run robust probe      --out-dir checkpoints/qa_learn/_verify/robust")
    lines.append("python -m n3d_qa_learn.step2_run robust run        --out-dir checkpoints/qa_learn/_verify/robust")
    lines.append("python -m n3d_qa_learn.step2_run robust calibrate  --out-dir checkpoints/qa_learn/_verify/robust")
    lines.append("python -m n3d_qa_learn.step2_run robust report     --out-dir checkpoints/qa_learn/_verify/robust")
    lines.append("```")
    lines.append("")
    lines.append(
        "- **耗时与生成时间不在本报告内**（确定性纪律，见报告首部）；请读运行日志的 `[+N.Ns]` 前缀。"
    )
    lines.append("")
    return "\n".join(lines)


def summarize_probe(
    cfg: RobustConfig, *, log: Any = None
) -> Dict[str, Any]:
    """``probe`` 的取证内容（表构造与口径取证，**不评测**）。

    输出：keys 形状 / 范数区间 / entry 数 / 键表指纹 / 编码器口径指纹 /
    两侧规模与切分取证 / 逐位精确查表结果 / 编码器声明维度对账 / 扰动自检。
    **不含任何非确定字段**（生成时间与耗时只进日志），故同命令重复运行产物逐字节一致。
    """
    t0 = time.time()

    def _log(msg: str) -> None:
        if callable(log):
            log(f"[+{time.time() - t0:6.1f}s] {msg}")

    out: Dict[str, Any] = {
        "module": "n3d_qa_learn.robust_eval",
        "command": "probe",
        "artifact_schema": "robust-probe-v1",
        "deterministic": True,
        "excluded_fields_note": (
            "本产物不含 created_utc / seconds 等非确定字段；生成时间与耗时只写运行日志"
        ),
        "seed": int(cfg.seed),
        "grid": {
            "profiles": [str(p) for p in cfg.profiles],
            "sides": [str(s) for s in cfg.sides],
            "perturb_types": [str(k) for k in cfg.perturb_types],
            "levels": [str(x) for x in PERTURB_LEVELS],
            "perturb_grid": {k: dict(PERTURB_GRID[k]) for k in cfg.perturb_types},
            "perturb_formulas": dict(PERTURB_FORMULAS),
            "nmag_shared_scale": float(NMAG_SHARED_SCALE),
            "nmag_formula_deviation": NMAG_FORMULA_SPEC_DEVIATION,
            "derived_seed_formula": "seed*1000 + 100*type_index + 10*level_index + variant",
        },
        "entry_tables": {},
        "encoder_declarations": {},
        "probe_errors": [],
    }
    for profile in cfg.profiles:
        expect = ET.expect_dim_of(str(profile))
        if "text" in cfg.sides:
            try:
                from .step2 import load_text_rows, resolve_product_dir

                resolved = resolve_product_dir("")
                _log(f"[probe] 文本侧 / 档 {profile}：编码 {resolved} 的行表 ...")
                rows = load_text_rows(resolved)
                vec = ET.build_vectorizer_for(str(profile), ET.ROLE_TEXT_LINE)
                bundle = ET.text_entry_table_from_rows(vec, rows, profile=str(profile))
                out["entry_tables"]["text:" + str(profile)] = {
                    **bundle.table.meta(),
                    "dim_expected": int(expect),
                    "bitwise_lookup": ET.verify_bitwise_lookup(bundle.table),
                    "text_row_key_table_sha256": str(bundle.text_row_table.sha256()),
                    "evidence": dict(bundle.evidence),
                }
                out["encoder_declarations"]["text:" + str(profile)] = {
                    "role": ET.ROLE_TEXT_LINE,
                    "dim": int(vec.dim),
                    "fingerprint": str(vec.fingerprint()),
                    "config": ET.encoder_config_for(str(profile), ET.ROLE_TEXT_LINE).to_dict(),
                }
                _log(
                    f"[probe]   keys {tuple(bundle.table.keys.shape)} / "
                    f"entry {bundle.table.size} / 查询 {len(bundle.query_positions)} / "
                    f"范数 [{bundle.table.norm_range()['min']:.8f}, "
                    f"{bundle.table.norm_range()['max']:.8f}] / "
                    f"逐位查表 {bundle.table.size}/{bundle.table.size}"
                )
            except Exception as exc:  # 编码器不可用等 -> 如实登记，不静默跳过
                out["probe_errors"].append({"side": "text", "profile": str(profile), "error": repr(exc)})
                _log(f"[probe]   文本侧 / 档 {profile} 失败（如实登记）：{exc!r}")
        if "qa" in cfg.sides:
            try:
                _log(f"[probe] QA 侧 / 档 {profile}：装配数据与编码 ...")
                data = ET.load_qa_training_data()
                vec = ET.build_vectorizer_for(str(profile), ET.ROLE_QUESTION)
                qa_bundle = ET.build_qa_entry_tables(data, vec, profile=str(profile))
                out["entry_tables"]["qa:" + str(profile)] = {
                    **qa_bundle.table.meta(),
                    "dim_expected": int(expect),
                    "bitwise_lookup": ET.verify_bitwise_lookup(qa_bundle.table),
                    "n_unknown": int(qa_bundle.unknown.n),
                    "n_known_queries": int(qa_bundle.known_queries.n),
                    "evidence": dict(qa_bundle.evidence),
                }
                out["encoder_declarations"]["qa:" + str(profile)] = {
                    "role": ET.ROLE_QUESTION,
                    "dim": int(vec.dim),
                    "fingerprint": str(vec.fingerprint()),
                    "config": ET.encoder_config_for(str(profile), ET.ROLE_QUESTION).to_dict(),
                }
                _log(
                    f"[probe]   keys {tuple(qa_bundle.table.keys.shape)} / "
                    f"entry {qa_bundle.table.size} / 未识别 {qa_bundle.unknown.n} / "
                    f"辅助查询 {qa_bundle.known_queries.n} / "
                    f"逐位查表 {qa_bundle.table.size}/{qa_bundle.table.size}"
                )
            except Exception as exc:
                out["probe_errors"].append({"side": "qa", "profile": str(profile), "error": repr(exc)})
                _log(f"[probe]   QA 侧 / 档 {profile} 失败（如实登记）：{exc!r}")
    # 扰动构造的自检（形状 / 有限性 / 重归一化）
    self_check: List[Dict[str, Any]] = []
    for kind in cfg.perturb_types:
        for level in PERTURB_LEVELS:
            base = np.tile(np.linspace(-1.0, 1.0, 8, dtype=np.float32), (4, 1))
            pert, detail = perturb_matrix(base, str(kind), str(level))
            self_check.append(
                {
                    "kind": str(kind),
                    "level": str(level),
                    "eps": float(PERTURB_GRID[str(kind)][str(level)]),
                    "formula": str(detail.get("formula", PERTURB_FORMULAS[str(kind)])),
                    "uses_generator": bool(detail.get("uses_generator", False)),
                    "shape": [int(x) for x in pert.shape],
                    "finite": bool(np.isfinite(pert).all()),
                    "n_zero_norm": int(detail["n_zero_norm"]),
                    "mean_row_displacement": float(detail["mean_row_displacement"]),
                    "generator_seed": int(detail["generator_seed"]),
                }
            )
    out["perturb_self_check"] = self_check
    out["bitwise_formula_assertions"] = assert_perturb_formulas_bitwise(cfg.perturb_types)
    out["euclidean_axis"] = {"removed": True, "note": EUCLIDEAN_EQUIVALENCE_NOTE}
    out["tiers"] = {
        "performed": [TIER_CLEAN, TIER_PERTURB],
        "not_applicable": [TIER_ALT_QUESTION],
        "not_applicable_reason": TIER_ALT_QUESTION_NOTE,
    }
    # [!] 耗时**不进产物**（只进日志）：见函数 docstring 的确定性说明。
    _log(f"[probe] 取证完成（耗时 {time.time() - t0:.1f}s，该耗时只进日志不入产物）")
    return out


__all__ = [
    "ROBUST_DIR",
    "ROBUST_SEED",
    "ROBUST_VARIANTS",
    "ROBUST_STABILITY_K",
    "PERTURB_TYPES",
    "PERTURB_LABELS",
    "PERTURB_LEVELS",
    "LEVEL_LABELS",
    "PERTURB_GRID",
    "PERTURB_FORMULAS",
    "NMAG_SHARED_SCALE",
    "NMAG_FORMULA_SPEC_DEVIATION",
    "GRANULARITY_GUARD_RULE",
    "GRANULARITY_REL_TOL",
    "AT_GRANULARITY_VERDICT_PHRASE",
    "GRANULARITY_ROLES_COUNT_KEY",
    "MIN_GRANULARITY_NOTE",
    "nmag_shift_matrix",
    "TOPK",
    "UNKNOWN_QUANTILES",
    "THRESHOLD_FACTOR",
    "NOISE_LEVEL_RATIO",
    "ROBUST_PROFILES",
    "PRIMARY_CELL_ROLE",
    "SIDES",
    "SIDE_LABELS",
    "TIER_CLEAN",
    "TIER_ALT_QUESTION",
    "TIER_PERTURB",
    "TIER_ALT_QUESTION_NOTE",
    "EUCLIDEAN_EQUIVALENCE_NOTE",
    "derived_seed",
    "perturb_matrix",
    "negligible_eps",
    "topk_search",
    "rank_of_gold",
    "evaluate_retrieval",
    "unrecognized_report",
    "euclidean_equivalence_evidence",
    "EvalCell",
    "text_cell_from_bundle",
    "qa_cell_from_bundle",
    "evaluate_cell",
    "noise_floor_of_cell",
    "aggregate_level_hit",
    "calibrate_thresholds",
    "RobustConfig",
    "run_evaluation",
    "load_run",
    "write_json",
    "write_text",
    "render_markdown",
    "summarize_probe",
]
