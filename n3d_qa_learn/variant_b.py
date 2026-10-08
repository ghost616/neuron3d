"""n3d_qa_learn 变体 B：单层 D→D 可学变换 + **冻结特征库**（"只变换、不生成"）。

设计文档三步走的**第二步**（第一步 1a = 分档鲁棒性考卷，见 :mod:`n3d_qa_learn.robust_eval`）。
本模块正面检验一个架构命题：**「表内嵌输出层、权重即特征库」能否让 argmax 真正可学**。

口径（逐轮确认，不得擅自变更）
-----------------------------
1. **载体**：``x`` = 原始编码器特征（raw），**不经 N3D 骨干**；考卷路径本身不含任何模型。
2. **特征档**：仅 ``lexical-88``（``local-hash``，D=88）与 ``bge-m3-1024``（``BAAI/bge-m3``，
   D=1024）；不纳入 ``zh-bag``。
3. **模型**：单层 D→D 变换 ``T``（**恒等初始化**）+ 冻结特征库内积评分 ——
   即「表内嵌输出层、权重即特征库、冻结」，**只学变换层**。
4. **训练信号**：**扰动自监督**（主方案）。对训练查询行施加 1a 同一套扰动，
   标签 = 该行自身在**全量键表**中的行下标，交叉熵作用于**冻结全表**。
5. **干净自监督（A 组）**仅作对照档，用于把「干净训练 ≈ 无变化」从推理变成实测。
6. **不纳入**对齐 / SupCon 类辅助损失（历史纠正记录 #17：对齐度↑ ≠ 可用性↑）。
7. **训练/评测划分**：训练查询 = 库行 1999（严格排除 666 查询行），评测查询 = 666 查询行；
   键表固定为全量 2665 行。
8. **训练扰动档**：全部 9 格（noise / mask / nmag × 弱 / 中 / 强）。
   1a 的有效性门禁**只用于判据**，不用来过滤训练数据。
9. **单 seed 42**；所有 Δ 为**单点差**、无跨 seed 极差，报告必须显式标明。
10. **复用 1a 的实现**：扰动一律走 :func:`n3d_qa_learn.robust_eval.perturb_matrix` /
    :func:`n3d_qa_learn.robust_eval.derived_seed` / ``PERTURB_GRID``，归一化一律走
    :func:`n3d_qa_learn.entry_table.l2_normalize_rows` —— **不自造第二份口径**。

恒等门禁（硬门禁）
------------------
训练前 ``T = I`` 时，变体 B 的 top-1 必须在 clean + 全部 9 个扰动档上与**余弦最近邻逐条一致**。
实现上有两层保障：

* **数学层**：输入已经是逐行 L2 归一化向量，行范数 = 1（float32 漂移 ≤ 1e-6）；
* **逐位层**：:class:`DToDTransform` 在 ``T`` **恰为恒等**时走**透传快路径**
  （``torch.equal(W, I) and bias == 0``），不做 ``x @ W.T`` 与 ``x + b``，
  从而 ``norm(T(x)) ≡ norm(x) = 1`` 在 float32 下**逐位成立**，不引入 ``x + 0`` 的浮点舍入；
  ``T`` 一旦被训练离开恒等，该快路径**自动失效**。

本模块仍在**两条独立路径**上现场实测一致性，并**如实登记**不一致条数
（若出现打平 / 翻转，如实报出条数与位置，不静默对齐）。

产物纪律
--------
一律写 ``checkpoints/qa_learn/_verify/variant_b/``；**不落盘特征矩阵、不产 zip 产物**；
产物内不含挂钟时间 / 耗时（确定性纪律，同 :mod:`n3d_qa_learn.robust_eval`）。
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
from . import robust_eval as R
from .entry_table import EntryKeyTable, TextEntryBundle
from .train import DEFAULT_VERIFY_DIR

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

#: 报告根目录（验证类运行一律写 ``_verify/``）。
VARIANT_B_DIR: str = os.path.join(DEFAULT_VERIFY_DIR, "variant_b")

#: 模块名（进产物，溯源用）。
MODULE_NAME: str = "n3d_qa_learn.variant_b"

#: 产物 schema 版本。
ARTIFACT_SCHEMA: str = "variant-b-v1"

#: 训练 / 扰动 seed（**冻结**；单 seed 口径）。
VARIANT_B_SEED: int = 42

#: 训练侧默认超参（Adam；**唯一优化器**，只优化变换层）。
DEFAULT_EPOCHS: int = 30
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_LR: float = 1e-2
DEFAULT_WEIGHT_DECAY: float = 0.0

#: 训练 / 评测划分（键表固定全量 2665；库行 1999 训练、查询行 666 评测）。
FULL_TABLE_ROWS: int = 2665
LIBRARY_ROWS: int = 1999
QUERY_ROWS: int = 666

#: 恒等快路径判定的**严格**口径（不是容差）。
IDENTITY_FASTPATH_RULE: str = (
    "`torch.equal(W, I) and bool((b == 0).all())` 逐位判定（**不是容差**）；"
    "两者同时成立时走**零加性快路径** `x + x@(W−I).T + (b − mean(b))` —— 两项在恒等时"
    "逐位为 +0.0，故 `T(x)` 与 `x` **逐位相同**、`norm(T(x)) ≡ norm(x) = 1` 在 float32 下"
    "逐位成立；**不得**写成 `return x`（那会让输入脱离计算图，训练侧 `backward()` 立刻报"
    "`element 0 of tensors does not require grad`，现场实测踩到过）；T 离开恒等后自动改走"
    "`x @ W.T + b`"
)

#: 训练模式（``perturb`` = 主方案；``clean`` = A 组对照档）。
TRAIN_MODES: Tuple[str, ...] = ("perturb", "clean")

#: ``cell_grid`` 里「变体 B 与 KNN 的 top-1 逐条差异」统计量的字段名（W1 修复的唯一来源）。
#:
#: [!] 该字段**曾经叫** ``identity_gate``，而 ``cell_grid`` 在训练后也被复用 ⇒ 产物里
#: 同时出现 ``cell_grid.identity_gate.all_equal = False``（统计量）与顶层
#: ``identity_gate.passed = True``（门禁），同名相反结论（皋陶审查 W1）。现改名并把
#: 「这是统计量、不是门禁」写进字段内容；门禁的判定只在 :func:`identity_gate` 里。
TOP1_DISCREPANCY_KEY: str = "top1_discrepancy_vs_knn"

#: **训练侧打分口径**开关（训练前提校准轮新增）。
#:
#: * ``normalized`` —— **与评测同一口径**：``normalize_query(T(x)) @ keys.T``
#:   （先变换、再逐行 L2 归一化、后内积）；
#: * ``raw`` —— **本模块此前的现状口径**：``T(x) @ keys.T``（**未归一化**），
#:   保留为**显式对照档**。
TRAIN_SCORERS: Tuple[str, ...] = ("normalized", "raw")

#: 默认训练侧打分口径 = ``normalized``（与评测同一口径）。
DEFAULT_TRAIN_SCORER: str = "normalized"

#: 打分口径的**唯一规则文本**（同时进入 ``TrainConfig.as_dict()``、产物 ``evidence``
#: 与 Markdown 报告，四处同源）。
TRAIN_SCORER_RULE: str = (
    "`normalized`（默认）= 与评测**同一口径**：`normalize_query(T(x)) @ keys.T`"
    "（先变换、再逐行 L2 归一化、后内积）；`raw` = 本模块此前的现状口径 "
    "`T(x) @ keys.T`（**未归一化**），只作显式对照档。"
    "**为什么必须默认 normalized**：`raw` 下「放大 ‖T(x)‖」可单调压低交叉熵，"
    "最省力的下降方向可能是**范数膨胀**而不是学方向 —— 该退化解通道由 "
    "`train.norm_diagnostics` 的逐 epoch 范数轨迹现场判定（实测判定见报告），"
    "不得当作结论预先陈述。评测侧**永远**使用 `VariantBModel.score`（归一化口径），"
    "本开关**只影响训练侧损失**。"
)

#: 训练侧的扰动：**全部 9 格**（计划口径第 8 条：不用有效性门禁过滤训练数据）。
TRAIN_PERTURB_CELLS: Tuple[Tuple[str, str], ...] = tuple(
    (str(kind), str(level)) for kind in R.PERTURB_TYPES for level in R.PERTURB_LEVELS
)

#: 训练批序打乱用的**局部** generator 种子（不消耗全局 RNG）。
SHUFFLE_SEED: int = 4242

#: 可训参数更新量门禁的判定口径。
UPDATE_GATE_RULE: str = (
    "训练前逐参数记录 float32 裸字节 SHA256（`state_bytes_sha256`），训练后重算；"
    "**只要存在任一参数裸字节变化**即判「可训参数更新量 > 0」（`passed = n_updated > 0`）。"
    "该门禁**不设阈值**、只判「是否有更新」，避免把超参差异读成能力差异"
)

#: 1a **生产实测值**（登记用，不参与任何判定）。
#:
#: [!] **来源与一处自我更正（必须如实登记）**：下表的数字取自 1a 的
#: ``robust_run.json`` **文本侧 ``cell_role=main`` 主判据格**（``variant=0``）。
#: ``bge-m3-1024`` 的 ``nmag`` 三格在本仓库里**存在两组并存的数字**：本表取的
#: ``0.481982 / 0.144144 / 0.043544`` 是**产物现场值**（README §15.5 主表与
#: §15.10 改后列亦为该组）；另一组 ``1.000000 / 0.996997 / 0.983483`` 出现在
#: README §15.10「修复前后完整对照」表的**改前**列与 §15.12 所引的 **W1 修复前历史值**。
#: 本模块以**产物为唯一现场来源**（``anchor_1a_check`` 逐格对账），
#: 并把该差异显式登记在 :data:`ANCHOR_RULE` 与 README §16.7，**不静默对齐**。
#:
#: [!] **行集合口径（皋陶审查 info 5）**：本表是 **1a §15.5 的查询行 666 侧**登记值；
#: 训练前提校准轮的锚点表落在**训练行 1999** 上，**与下表天然不同、不可直接互校**
#: （现场例：`nmag/weak` 训练行 `0.355178` vs 查询行 `0.391892`）。两者的对账入口是
#: :func:`anchor_1a_check`（查询行 666 侧，容差 ``ANCHOR_TOL``）—— 该对账**只在**
#: ``run_variant_b``（⇒ ``variantb train`` / ``eval``）与本轮新增的 ``variantb calibrate``
#: 产物里给出，**不**出现在只有训练行读数的锚点表里。
ROBUST_1A_KNN_REFERENCE: Dict[str, Dict[str, Dict[str, float]]] = {
    ET.PROFILE_LEXICAL: {
        "noise": {"clean": 1.000000, "weak": 1.000000, "middle": 0.984985, "strong": 0.789790},
        "mask": {"clean": 1.000000, "weak": 0.990991, "middle": 0.947447, "strong": 0.770270},
        "nmag": {"clean": 1.000000, "weak": 0.391892, "middle": 0.099099, "strong": 0.049550},
    },
    ET.PROFILE_SEMANTIC: {
        "noise": {"clean": 1.000000, "weak": 0.998498, "middle": 0.980480, "strong": 0.758258},
        "mask": {"clean": 1.000000, "weak": 1.000000, "middle": 1.000000, "strong": 1.000000},
        "nmag": {"clean": 1.000000, "weak": 0.481982, "middle": 0.144144, "strong": 0.043544},
    },
}

#: 旁证锚点容差（1a 登记值 vs 本模块现场重算的 KNN 基线）。
ANCHOR_TOL: float = 1e-6

#: 旁证锚点规则文本（唯一来源；同时进入 `evidence.knn_reference_rule` 与
#: `anchor_1a_check.rule`）。
#:
#: [!] 该文本**必须**携带「一处不同源」的登记（离朱 R57 F2）：`bge-m3-1024` 的
#: ``nmag`` 三格在本仓库里有**两组并存的数字**，只有一组能与 1a 的 run 产物对上。
ANCHOR_RULE: str = (
    "本模块现场重算的 KNN 基线应与 1a 登记值在 1e-6 内一致；"
    "两者的检索池口径相同（全量键表自检索），不一致即如实报出，**不静默对齐**。"
    "**已登记的一处不同源**：`bge-m3-1024` 的 `nmag` 弱/中/强在本仓库里有两组并存的数字 —— "
    "`1.000000 / 0.996997 / 0.983483`（README §15.10「修复前后完整对照」表的**改前**列与 "
    "§15.12 所引的 W1 修复前历史值）vs `0.481982 / 0.144144 / 0.043544`"
    "（`checkpoints/qa_learn/_verify/robust/robust_run.json` 文本侧 `cell_role=main` 的**现场值**，"
    "README §15.5 主表与 §15.10 改后列也已同步为该组）。"
    "本模块**一律以产物为唯一现场来源**（故 `ROBUST_1A_KNN_REFERENCE` 取后者），"
    "该差异同时登记在 `ROBUST_1A_KNN_REFERENCE` 的常量注释与 README §16.7。"
)

#: 1a 判**有效**的扰动格（**只用于主判据**，不用来过滤训练数据）。
EFFECTIVE_CELLS_1A: Dict[str, Tuple[str, ...]] = {
    ET.PROFILE_LEXICAL: (
        "noise/weak", "noise/middle", "noise/strong",
        "mask/weak", "mask/middle", "mask/strong",
        "nmag/weak", "nmag/middle", "nmag/strong",
    ),
    ET.PROFILE_SEMANTIC: (
        "noise/weak", "noise/middle", "noise/strong",
        "nmag/weak", "nmag/middle", "nmag/strong",
    ),
}

#: 1a 判**无效**的扰动格（如实登记；判据不采用，但数字仍必须报出）。
INEFFECTIVE_CELLS_1A: Dict[str, Tuple[str, ...]] = {
    ET.PROFILE_SEMANTIC: ("mask/weak", "mask/middle", "mask/strong"),
}

#: 训练前提校准轮：**恒等参照锚点**的口径（唯一规则文本）。
#:
#: 锚点 = **未训练**（``T = I``）的模型在**训练行 1999** 上、clean + 9 个扰动格的 top-1 自命中；
#: 它就是「训练必须至少不劣于此」的**硬性下限**。之所以必须落在**训练行**上，是因为要判定的
#: 问题是「训练有没有在自己的目标上失败」—— 评测行（666 查询行）从不出现在训练目标里，
#: 用它们当参照会把「训练失败」与「泛化落差」混在一起。
TRAIN_ANCHOR_RULE: str = (
    "恒等锚点 = **未训练**（`T = I`）的变体 B 在**训练行 1999**（键表全量 2665 − 查询行 666）上、"
    "跑一次**完整 10 格网格**（clean + 9 个扰动格）得到的**逐格 top-1 率 R@1**"
    "（金标 = 该行自身在全量键表 2665 中的行下标）。"
    "口径与评测侧**逐字相同**（同一 `cell_grid` 代码路径、同一扰动实现、同一 1a 派生种子、"
    "同一名次口径），**唯一差别**是行集合 = 训练行而非查询行。"
    "`T = I` 时变体 B 的打分与余弦最近邻**逐位相同**（恒等门禁的判据），故锚点同时也是"
    "该行集合上的 KNN 自命中。判据 = 「训练后**逐格** R@1 **不低于** 该锚点」（训练无害下限）。"
    "观测粒度 = 1/1999 ≈ 5.0e-4（每条训练行只值约 5 个万分之一）。"
    "**不得**把训练循环里的批内滚动平均（:data:`TRAIN_TOP1_SELF_RULE`）当作本锚点的对应量 ——"
    "两者不是同一观测量（见 :data:`ANCHOR_ROW_SET_NOTE`）。"
)

#: **行集合差异说明**（唯一规则文本；防止读者拿不同行集合的数字互校）。
#:
#: 存在理由（皋陶审查 info 5）：锚点表落在**训练行 1999** 上，而 1a 的 §15.5 登记值落在
#: **查询行 666** 上，二者**天然不同**（如 `nmag/weak`：`0.355178` vs `0.391892`），
#: 属**行集合不同**而不是数值漂移。两类数字**不可直接互校**。
ANCHOR_ROW_SET_NOTE: str = (
    "行集合口径（**不得混用**）：本锚点表与训练无害判据一律落在**训练行 1999**"
    "（键表全量 2665 − 查询行 666）上；README §15.5 登记的 1a KNN 基线落在**查询行 666** 上。"
    "两者**天然不同**（现场例：`nmag/weak` 训练行 `0.355178` vs 查询行 `0.391892`），"
    "属**行集合不同**而非数值漂移，**不可直接互校**。"
    "与 1a 登记值对账的唯一入口是 :func:`anchor_1a_check`（**查询行 666 侧**，容差 1e-6）。"
)

#: **训练侧「批内滚动平均」的唯一规则文本**（皋陶审查 warning 3 / info 6）。
#:
#: 该量与「在 1999 行上跑一次完整 10 格网格」得到的**逐格 R@1** 是**两个不同的观测量**，
#: 旧名 ``train_top1_self`` 与逐格量同名易串（因此改名为 ``train_batch_running_top1``）。
TRAIN_TOP1_SELF_RULE: str = (
    "`train_batch_running_top1` = 训练循环内的**批内滚动平均**：每步只覆盖**一个轮转扰动格**"
    "（9 格轮转）、每批 256 行，且命中判定使用**该步参数更新之前**算出的 logits；"
    "逐 epoch 汇总为 `history[].train_batch_running_top1`，末值为 `train.train_batch_running_top1`。"
    "它是**仅作诊断**的训练过程量，**不是**本轮的判定量："
    "判定量是**逐格 R@1**（在训练行 1999 上跑一次完整 10 格网格），见 "
    "`train_harmlessness.per_cell[].post_top1` 与 `train_side_cells[].variant_b_recall_at_1`"
    "（同一组合两者逐格必须相等：前者是对照视图、后者是逐格明细，"
    "现场一致性登记在 `train.running_vs_grid`）。"
    "**不得**把本量与恒等锚点（:data:`TRAIN_ANCHOR_RULE`，逐格 R@1）直接相减 ——"
    "两者的现场差值由产物字段 `train.running_vs_grid.delta_running_minus_grid_mean` 直接给出。"
)

#: 训练行集合上的观测量粒度说明（报告必须显式标注）。
TRAIN_GRANULARITY_NOTE: str = (
    "训练侧观测量粒度：训练行 1999 条 ⇒ 1/1999 ≈ 5.0e-4；"
    "评测侧（查询行 666 条）粒度 = 1/666 ≈ 1.5e-3。"
    "落差小于该格粒度者不得作为判定依据。"
)

#: 「训练无害」判据**排除**的格（唯一来源）。
#:
#: ``clean`` 格是**逐位自匹配**（``cos = 1.0``，锚点结构性 = 1.000000）：对任何**非恒等**
#: 变换该格只能变差、不可能变好，把它放进判据等于让判据与「训练是否落实优化前提」无关。
#: 该排除与计划口径「本轮的 clean 查询格（结构性 1.0）不得作为判据」同源（训练侧同理）。
TRAIN_HARMLESSNESS_EXCLUDED_CELLS: Tuple[str, ...] = ("clean",)

#: 「训练无害」判据的**唯一规则文本**（判定实现与报告都从这里派生）。
TRAIN_HARMLESSNESS_RULE: str = (
    "训练无害下限：训练后在**训练行 1999**（键表全量 2665）上跑一次**完整 10 格网格**，"
    "取其中 **9 个扰动格**的**逐格 top-1 率 R@1**，要求**逐格不低于**恒等锚点"
    "（T = I、同格同口径）；判据 = 全部被纳入的格 `post >= anchor`。"
    "主报量是**逐格 R@1**（`train_harmlessness.per_cell[].post_top1` / "
    "`train_side_cells[].variant_b_recall_at_1`），**不是**训练循环里的批内滚动平均"
    "（:data:`TRAIN_TOP1_SELF_RULE`）—— 两者不是同一观测量，不得互相代入。"
    "`clean` 格**不纳入**判据（逐位自匹配、锚点结构性 1.000000，任何非恒等变换只能变差），"
    "但仍必须在锚点表里如实报出。该判据**不是**「提升」，只是「训练没有在自己的目标上"
    "把自己弄得更差」；它同时是可满足的 —— 恒等解 `T = I` 逐格恰好等于锚点。"
    "行集合口径见 :data:`ANCHOR_ROW_SET_NOTE`（1999 训练行 vs §15.5 的 666 查询行不可互校）。"
)

#: 优化前提扫描的**最小网格**（逐轴取值；组合数 = 2×2×2 = 8）。
CALIBRATE_LR_GRID: Tuple[float, ...] = (1e-3, 1e-2)
CALIBRATE_WD_GRID: Tuple[float, ...] = (0.0, 1e-2)
CALIBRATE_SCORERS: Tuple[str, ...] = TRAIN_SCORERS

#: 最优超参的**选择规则**（唯一规则文本；判定与选取同源）。
#:
#: 注意：选择**只在** ``normalized`` 组合内进行（判据口径要求最终归因落在 normalized 上），
#: ``raw`` 组合只作「口径对齐前」的对照列。
CALIBRATE_SELECTION_RULE: str = (
    "在 `train_scorer = normalized` 的组合内，按以下**主序**选取最优超参："
    "① 「训练无害格数」（训练行 1999 上 **9 个扰动格**中，训练后**逐格 R@1 不低于**恒等锚点的"
    "格数；`clean` 已按 `TRAIN_HARMLESSNESS_EXCLUDED_CELLS` 排除）降序；"
    "② 「训练侧训练后逐格 R@1 的**整段平均**」降序 —— **该平均值的格集合为 10 格"
    "（`clean` + 9 个扰动格，与 ① 的 9 格不同，`clean` 在此计入）**，"
    "该口径差异在产物字段 `selection.<档>.candidates[].train_post_top1_mean_cell_set` 中显式标注；"
    "③ 评测侧 1a 判有效的扰动格中 `ΔR@1 >= 0` 的格数降序；"
    "④ 完全并列时按 `(lr, weight_decay)` 字典序取第一个（确定性，不引入随机）。"
    "`raw` 组合**不参与**选取，只作口径对齐前的对照。"
    "**注意**：① 与 ② 都不是训练循环里的批内滚动平均（:data:`TRAIN_TOP1_SELF_RULE`），"
    "该量仅供诊断、**不进入**排序键。"
)

#: 选取候选项的**字段口径标签**（唯一来源：`train_post_top1_mean` 的格集合）。
CALIBRATE_CANDIDATE_CELL_SET: str = (
    "10 格（`clean` + 9 个扰动格；`clean` 在此计入 ⇒ 与 ① 的 9 格不同）"
)

#: 归因判定的**必需条件键**（唯一来源：规则文本、判定实现、报告三处都从这里派生）。
ATTRIBUTION_REQUIRED_CONDITIONS: Tuple[str, ...] = (
    "a_train_harmless",
    "b_eval_not_worse",
)

#: 每个条件的**人类可读文本**（唯一来源）。
ATTRIBUTION_CONDITION_TEXT: Dict[str, str] = {
    "a_train_harmless": (
        "训练无害下限成立：在 `normalized` 口径 + 最优超参下，**训练行 1999** 上"
        "**9 个扰动格**（`clean` 已按 `TRAIN_HARMLESSNESS_EXCLUDED_CELLS` 排除）的**逐格 R@1**"
        "**全部不低于**恒等锚点（同格同口径）"
    ),
    "b_eval_not_worse": (
        "评测侧不劣：在**查询行 666** 上，1a 判**有效**的扰动格里**逐格 R@1** **全部不低于** "
        "KNN 基线"
    ),
}

#: 归因的两种结论（机器可读键 → 人类可读文本）。
ATTRIBUTION_VERDICTS: Dict[str, str] = {
    "optimization_premise": (
        "归因 = **优化前提未落实**（口径 / 超参问题）：训练在自己的目标上不再有害，"
        "且评测侧扰动格不劣于 KNN ⇒ 第二步的否证**不干净**，线性 `D→D` 的表达力"
        "**未被本轮否证**；第三步应先把这条前提固定下来，再谈候选粒度改造。"
    ),
    "linear_expressivity": (
        "归因 = **线性 `D→D` 表达力 / 架构不足成立**：即便在 `normalized` 口径 + 最优超参下，"
        "「训练无害下限」或「评测侧不劣于 KNN」至少一条不成立 ⇒ 第二步的否证成立，"
        "为第三步（候选粒度：逐行键 → 类 / 簇级候选，或让键表随表示重算）提供依据；"
        "**不要**继续加大变换层容量。"
    ),
    "mixed_per_profile": (
        "归因 = **各档结论不一致**：本轮口径要求逐档单独读（诊断轮只跑 `lexical-88`），"
        "**不得**把多档合并成单一结论；逐档结论见 `attribution.<档>.verdict`。"
    ),
    "not_applicable": (
        "归因 = **本轮扫描不适用**：本次网格里**没有** `train_scorer = normalized` 的组合"
        "（例如 `--calibrate-scorers raw` / `--grid-combo \"raw:...\"`），而归因判据要求在该口径下"
        "选取最优超参 ⇒ 如实标「不适用」，**不以 `raw` 口径出归因结论、也不因此丢弃已完成的扫描**。"
        "扫描本身的数字仍全部有效并已写入产物（`selection.applicable = false` + `reason`）。"
    ),
}


def _build_attribution_rule() -> str:
    """由 :data:`ATTRIBUTION_REQUIRED_CONDITIONS` / :data:`ATTRIBUTION_CONDITION_TEXT` 派生规则文本。

    为什么用函数派生而不是手写一段文案：判定实现与规则文本必须**同源** ——
    判定实现按 ``ATTRIBUTION_REQUIRED_CONDITIONS`` 逐个求值并取合取，规则文本由**同一常量**
    渲染，二者不可能漂移（历史纠正记录：口径文本与实现分家的那类缺陷）。
    """
    parts = [
        f"（{i}）{ATTRIBUTION_CONDITION_TEXT[key]}"
        for i, key in enumerate(ATTRIBUTION_REQUIRED_CONDITIONS, start=1)
    ]
    return (
        "**归因判定规则（唯一来源，与判定实现同源）**：在 `train_scorer = normalized` 口径 + "
        "按 `CALIBRATE_SELECTION_RULE` 选出的最优超参下，若**全部**满足："
        + "；且".join(parts)
        + " ⇒ 归因 = 优化前提（`optimization_premise`）；"
        "只要其中**任一**条件不成立 ⇒ 归因 = 线性 `D→D` 表达力 / 架构不足成立"
        "（`linear_expressivity`），为第三步提供依据。"
    )


#: 归因规则文本（由 :func:`_build_attribution_rule` 从上面两个常量派生；**不手写**）。
ATTRIBUTION_RULE: str = _build_attribution_rule()

#: 训练前提校准轮的观测量口径小结（进产物，避免读者自行猜口径）。
CALIBRATION_SCOPE_NOTE: str = (
    "本轮为**诊断轮**：不以提升指标为成功标准。特征档只跑 `lexical-88`（成本优先，"
    "`bge-m3-1024` 仅在归因判定需要时按需补跑并单独登记）；划分不变（键表全量 2665 / "
    "训练行 1999 / 查询行 666 / 冻结划分交集 0）；单 seed 42；确定性；"
    "不动 1a 考卷（`entry_table.py` / `robust_eval.py` 零改动）。"
)


# ---------------------------------------------------------------------------
# 1. 变体 B 模型：单层 D→D 可学变换 + 恒等初始化
# ---------------------------------------------------------------------------


class DToDTransform(torch.nn.Module):
    """**单层 D→D 可学变换** ``T(x) = x @ W.T + b``，恒等初始化 ``W = I, b = 0``。

    口径
    ----
    * ``W``：``[D, D]`` 可学参数，初始化 = 单位阵（``torch.eye``，**不消耗 RNG**）；
    * ``b``：``[D]`` 可学参数，初始化 = 全零（**不消耗 RNG**）；
    * **恒等快路径**：``torch.equal(W, I) and (b == 0).all()`` 时 ``forward`` 直接返回输入，
      不做矩阵乘与加法 —— 保证 ``norm(T(x)) ≡ norm(x) = 1`` 在 float32 下**逐位成立**
      （否则 ``x + 0`` 的浮点舍入会让范数偏离 1，恒等门禁只能"近似"一致）。
      这是**显式的逐位判定**、不是容差；``T`` 一旦离开恒等，该路径自动失效。

    关键不变量（构造期，违反即抛）
    -----------------------------
    1. ``D >= 1``；
    2. ``weight`` 必须**逐位**等于 ``torch.eye(D)``、``bias`` 必须**逐位**全零。

    参数
    ----
    dim : int
        连接参数 ``D``。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        d = int(dim)
        if d < 1:
            raise ValueError(f"变换层维度 D 必须 >= 1，当前 {d}")
        self.dim: int = d
        self.weight = torch.nn.Parameter(torch.eye(d, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
        # 构造期不变量（历史纠正记录 #10：构造期不变量显式断言，且不断言训练后量）
        assert torch.equal(
            self.weight.detach(), torch.eye(d, dtype=torch.float32)
        ), "DToDTransform 的 weight 必须是逐位单位阵（恒等初始化）"
        assert bool(
            (self.bias.detach() == 0).all()
        ), "DToDTransform 的 bias 必须逐位全零（恒等初始化）"
        # 最近一次 forward 的取证（推理态才记录），供恒等门禁读取
        self._last_io: Dict[str, Any] = {}

    def is_identity(self) -> bool:
        """``T`` 是否**逐位**等于恒等（决定是否走透传快路径）。"""
        with torch.no_grad():
            same_w = bool(torch.equal(self.weight.detach(), torch.eye(self.dim)))
            zero_b = bool((self.bias.detach() == 0).all())
        return bool(same_w and zero_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用变换；恒等时走**零加性**快路径（逐位等于输入，且**保持可导**）。

        参数
        ----
        x : torch.Tensor
            ``[B, D]`` 待变换特征。

        返回
        ----
        torch.Tensor
            ``[B, D]`` 变换结果（恒等时与输入**逐位相同**，且仍带 ``grad_fn``）。

        实现要点（**关键**）
        -------------------
        恒等时**不能**直接 ``return x``：那会让 ``x`` 脱离计算图，
        训练侧 ``loss.backward()`` 立刻报
        ``RuntimeError: element 0 of tensors does not require grad``
        （现场实测踩到该坑）。正确写法是保留一条**数学上恰为 0 的加性项**：

        * 权重项 = ``x @ (W - I).T`` —— 恒等时 ``W - I`` 逐位全零，且
          float32 下 ``x @ 0 = 0``（逐元素求和，0 的求和恒为 +0.0，
          负零与正零在加法中产生 +0.0），故 ``x + 0`` **逐位等于** ``x``；
        * 偏置项 = ``-mean(b) + b`` —— 恒等时 ``b`` 逐位全零，故该项逐位为 0。

        两个加性项都为**可微**项（对可训参数有非零偏导：``d/dW[x@(W-I).T] = x``、
        ``d/db[b - mean(b)] = 1 - 1/D``），因此训练侧梯度正常回传。
        """
        if x.dim() != 2 or int(x.shape[1]) != int(self.dim):
            raise ValueError(
                f"变换层期望 [B, {self.dim}] 输入，当前形状 = {tuple(x.shape)}"
            )
        fast = self.is_identity()
        if fast:
            eye = torch.eye(self.dim, dtype=x.dtype, device=x.device)
            zero_delta = x @ (self.weight - eye).t()
            zero_shift = self.bias - self.bias.mean()
            out = x + zero_delta + zero_shift
        else:
            out = torch.nn.functional.linear(x, self.weight, self.bias)
        if not self.training:
            with torch.no_grad():
                self._last_io = {
                    "fast_path": bool(fast),
                    "input_norm_min": float(torch.linalg.norm(x, dim=1).min()),
                    "input_norm_max": float(torch.linalg.norm(x, dim=1).max()),
                    "output_norm_min": float(torch.linalg.norm(out, dim=1).min()),
                    "output_norm_max": float(torch.linalg.norm(out, dim=1).max()),
                }
        return out

    def last_io_evidence(self) -> Dict[str, Any]:
        """最近一次推理 ``forward`` 的输入 / 输出范数取证（只读副本）。"""
        return dict(self._last_io)

    def explicit_matrix(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(W, b)`` 的 detach 副本（供「显式 matmul 路径」旁证使用）。"""
        return self.weight.detach().clone(), self.bias.detach().clone()


@dataclass(frozen=True)
class VariantBModel:
    """变体 B 的**完整模型**：冻结特征库（全量键表） + 可学变换 ``T``。

    参数
    ----
    transform : DToDTransform
        单层 D→D 可学变换（**唯一可训部分**）。
    keys : torch.Tensor
        ``[N, D]`` **冻结**特征库（逐行 L2 归一化；``requires_grad=False``）。
    table_sha256 : str
        键表内容指纹（与 :class:`entry_table.EntryKeyTable` 同口径）。
    """

    transform: DToDTransform
    keys: torch.Tensor
    table_sha256: str = ""

    @property
    def n_keys(self) -> int:
        """特征库条目数 ``N``。"""
        return int(self.keys.shape[0])

    @property
    def dim(self) -> int:
        """连接参数 ``D``。"""
        return int(self.keys.shape[1])

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """可训参数对象列表（**现场枚举** ``named_parameters``，不手写名字）。"""
        return [p for _, p in self.transform.named_parameters()]

    def parameter_snapshot(self) -> List[Dict[str, Any]]:
        """可训参数的**现场枚举**快照（名字 / 形状 / 元素数 / 裸字节 SHA256）。"""
        out: List[Dict[str, Any]] = []
        for name, param in self.transform.named_parameters():
            blob = param.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
            out.append(
                {
                    "name": str(name),
                    "shape": [int(v) for v in param.shape],
                    "n_element": int(param.numel()),
                    "dtype": str(param.dtype).replace("torch.", ""),
                    "requires_grad": bool(param.requires_grad),
                    "state_bytes_sha256": ET.sha256_bytes(blob),
                }
            )
        return out

    def transform_score(self, x: torch.Tensor) -> torch.Tensor:
        """``T(x) @ keys.T``（**未归一化**；可微）。

        参数
        ----
        x : torch.Tensor
            ``[B, D]`` 查询特征（已 L2 归一化）。

        返回
        ----
        torch.Tensor
            ``[B, N]`` 打分矩阵（``keys`` 不参与梯度）。

        口径归属
        --------
        本方法是 ``train_scorer="raw"`` 分支的唯一实现，**不再是**训练侧默认口径；
        默认口径见 :meth:`train_score`（``normalized``）。评测侧一律走 :meth:`score`。
        """
        return self.transform(x) @ self.keys.t()

    def train_score(self, x: torch.Tensor, scorer: Optional[str] = None) -> torch.Tensor:
        """**训练侧打分**（唯一入口，按 ``scorer`` 分派；两者都可微）。

        参数
        ----
        x : torch.Tensor
            ``[B, D]`` 查询特征（已 L2 归一化；训练侧为扰动后的库行）。
        scorer : Optional[str]
            ``normalized``（默认，与评测同一口径）或 ``raw``（现状未归一化口径）；
            ``None`` = :data:`DEFAULT_TRAIN_SCORER`。

        返回
        ----
        torch.Tensor
            ``[B, N]`` 打分矩阵。

        口径（见 :data:`TRAIN_SCORER_RULE`）
        ----------------------------------
        * ``normalized``：``normalize_query_allow_zero(T(x)) @ keys.T`` —— 与评测侧
          :meth:`score` **同一口径**（差别只在零范数行的处置，见下）；
        * ``raw``：``T(x) @ keys.T`` —— 未归一化。

        零范数行
        --------
        训练批内可能含「被扰动成零向量」的行（现场真实可达，见 :func:`zero_norm_mask`）。
        ``raw`` 分支对零行天然给出全零 logits；``normalized`` 分支走
        :meth:`normalize_query_allow_zero`（零行保持全零、**不报错**），
        从而与 ``raw`` 在零行上的 logits 逐位一致（都是全零），且该行**必然**被
        :func:`zero_norm_mask` 在损失中掩码。评测侧仍走严格的 :meth:`normalize_query`
        （零范数行**显式报错**），两条口径的差别只在「训练侧事后掩码」这一处。
        """
        name = str(scorer or DEFAULT_TRAIN_SCORER)
        if name not in TRAIN_SCORERS:
            raise ValueError(f"未知训练侧打分口径 {name!r}；可用 = {list(TRAIN_SCORERS)}")
        y = self.transform(x)
        if name == "raw":
            return y @ self.keys.t()
        return self.normalize_query_allow_zero(y) @ self.keys.t()

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """检索打分：``T(x)`` **重新 L2 归一化**后与冻结键表做内积（评测侧唯一口径）。

        [!] **必须**先过 :meth:`transform` 再归一化。首版实现漏掉了 ``self.transform(x)``
        （直接写 ``normalize_query(x) @ keys.T``），后果是：整个评测侧退化成
        「未训练模型 = 余弦 KNN」，所有格的 Δ 结构性地恒为 0。该缺陷由
        :func:`score_path_evidence` 这条**自洽门禁**现场抓出（它比对
        ``score(q)`` 与显式 ``normalize_query(transform(q)) @ keys.T``），
        现已成为恒等门禁的一部分 —— 任何"评测侧没走变换层"的回归都会被立刻拦下。
        """
        return self.normalize_query(self.transform(x)) @ self.keys.t()

    @staticmethod
    def normalize_query(x: torch.Tensor) -> torch.Tensor:
        """逐行 L2 归一化（数值路径与 :func:`entry_table.l2_normalize_rows` **逐位一致**）。

        口径理由：``entry_table.l2_normalize_rows`` 用 **float64** 求范数再落回 float32；
        本函数用逐元素 float32 求和算出同一 float64 值（float32 的 24 位尾数恰是 float64
        53 位尾数的前缀，故「float32 逐元素求和」在 float64 下是**精确**的），
        两条路径因此给出**逐位相同**的结果 —— 这是恒等门禁能要求「逐条一致」而非
        「近似一致」的前提。零范数行**显式报错**（不静默产出 NaN / 全零特征）。
        """
        norms = torch.sqrt((x.double() * x.double()).sum(dim=1))
        zero = norms <= 0.0
        if bool(zero.any()):
            idx = [int(i) for i in torch.nonzero(zero).flatten().tolist()[:20]]
            raise ValueError(
                "查询侧出现零范数行（L2 归一化分母为 0）："
                f"行下标 = {idx}；拒绝静默产出 NaN / 全零特征"
            )
        return (x.to(torch.float64) / norms[:, None]).to(torch.float32)

    @staticmethod
    def normalize_query_allow_zero(x: torch.Tensor) -> torch.Tensor:
        """逐行 L2 归一化，**零范数行保持全零且不报错**（仅训练侧使用）。

        与 :meth:`normalize_query` 的关系（**必须如实登记**）
        ---------------------------------------------------
        数值路径**完全相同**（float64 求范数、逐行相除、落回 float32），唯一差别是
        零范数行的处置：本方法把该行保持为全零向量并**交由调用方掩码**，而
        :meth:`normalize_query` 直接抛 ``ValueError``。因此：

        * 批内**没有**零范数行时，两者结果**逐位相同**（由
          :func:`train_scorer_evidence` 现场逐位断言，不是靠推理）；
        * 批内**有**零范数行时，本方法只在**训练侧**被用到，且该行必然被
          :func:`zero_norm_mask` 从损失中掩掉 —— 于是「零范数行掩码」这条下发口径
          不被削弱，评测侧的严格口径也不被放宽。
        """
        norms = torch.sqrt((x.double() * x.double()).sum(dim=1))
        zero = norms <= 0.0
        safe = torch.where(zero, torch.ones_like(norms), norms)
        out = (x.to(torch.float64) / safe[:, None]).to(torch.float32)
        if bool(zero.any()):
            out = torch.where(zero[:, None], torch.zeros_like(out), out)
        return out


def build_model(table: EntryKeyTable) -> VariantBModel:
    """由统一条目特征表构造变体 B 模型（``T = I`` + **冻结**键表副本）。

    参数
    ----
    table : EntryKeyTable
        统一的条目特征表（**键表固定为全量 2665 行**）；本函数拷贝一份并把
        ``requires_grad`` 置 False，**不改动传入对象**。

    返回
    ----
    VariantBModel
        ``transform`` 恒等初始化；``keys`` 为 detach 后的副本。

    异常
    ------
    ValueError
        表行数与冻结口径（2665）不符。
    """
    keys = table.keys.detach().to(torch.float32).clone().contiguous()
    keys.requires_grad_(False)
    if int(keys.shape[0]) != int(FULL_TABLE_ROWS):
        raise ValueError(
            f"变体 B 的键表固定为全量 {FULL_TABLE_ROWS} 行，当前 {int(keys.shape[0])} 行；"
            "训练/评测划分按「全量键表 + 库行 1999 训练 + 查询行 666 评测」冻结"
        )
    return VariantBModel(
        transform=DToDTransform(int(keys.shape[1])),
        keys=keys,
        table_sha256=str(table.sha256()),
    )


def sample_bitwise_equality(
    model: VariantBModel, queries: torch.Tensor
) -> Dict[str, Any]:
    """**逐位旁证**：``norm(T(x))`` 是否与 ``norm(x)`` 逐位相同（恒等透传的直接证据）。"""
    with torch.no_grad():
        x = queries.to(torch.float32)
        y = model.transform(x)
        nx = torch.linalg.norm(x, dim=1)
        ny = torch.linalg.norm(y, dim=1)
        same = int((nx == ny).sum().item())
        dev = float((nx - ny).abs().max().item()) if int(nx.numel()) else 0.0
        return {
            "n": int(nx.numel()),
            "n_bitwise_equal": int(same),
            "bitwise_equal_frac": float(same / max(1, int(nx.numel()))),
            "max_abs_norm_deviation": dev,
            "fast_path": bool(model.transform.is_identity()),
        }


def explicit_path_evidence(
    model: VariantBModel, queries: torch.Tensor, *, training_mode: bool = False
) -> Dict[str, Any]:
    """旁证：``T(x)`` 与**显式** ``x @ W.T + b`` 的偏差（恒等时应逐位相同）。

    参数
    ----
    model : VariantBModel
        待检查模型。
    queries : torch.Tensor
        ``[M, D]`` 查询特征。
    training_mode : bool
        ``True`` 时把变换层置 train 态再检查（**训练态走的是零加性快路径**，
        评测态走 ``x @ W.T + b``；两态都必须与显式结果逐位一致）。

    返回
    ----
    Dict[str, Any]
        ``n`` / ``max_abs_deviation`` / ``bitwise_equal`` / ``mode`` / ``note``。
    """
    was_training = bool(model.transform.training)
    model.transform.train(bool(training_mode))
    with torch.no_grad():
        x = queries.to(torch.float32)
        fast = model.transform(x)
        w, b = model.transform.explicit_matrix()
        slow = torch.nn.functional.linear(x, w, b)
        dev = float((fast - slow).abs().max().item()) if int(fast.numel()) else 0.0
        equal = bool(torch.equal(fast, slow))
    model.transform.train(was_training)
    return {
        "n": int(fast.shape[0]),
        "max_abs_deviation": dev,
        "bitwise_equal": equal,
        "mode": "train" if training_mode else "eval",
        "note": (
            "恒等时应逐位相同（零加性快路径加的恰是 +0.0，`x @ I + 0` 在 float32 下恰等于 x）；"
            "非恒等时该偏差只反映 matmul 实现差异，**不参与门禁**"
        ),
    }


def score_path_evidence(model: VariantBModel, queries: torch.Tensor) -> Dict[str, Any]:
    """**自洽门禁**：``model.score(q)`` 必须与显式 ``normalize_query(transform(q)) @ keys.T`` 逐位相同。

    存在理由（首版真实缺陷，现场抓出）：评测侧若漏掉变换层（直接对原始查询做归一化内积），
    整个「变体 B」会退化成「未训练模型 = 余弦 KNN」，所有 Δ 结构性地恒为 0，
    而恒等门禁**照样通过**（因为 T=I 时两条路径本来就该相同）。本函数把
    「评测侧确实走了变换层」变成**逐位可证**的事实：它用一条与 `score` 无关的
    显式计算路径复算并逐字节比对，不一致即判失败。

    返回
    ----
    Dict[str, Any]
        ``n`` / ``bitwise_equal`` / ``max_abs_deviation`` / ``input_norm_mean`` /
        ``transformed_norm_mean`` / ``rule``。
    """
    with torch.no_grad():
        x = queries.to(torch.float32)
        via_score = model.score(x)
        explicit = VariantBModel.normalize_query(model.transform(x)) @ model.keys.t()
        dev = float((via_score - explicit).abs().max().item()) if int(via_score.numel()) else 0.0
        return {
            "n": int(x.shape[0]),
            "bitwise_equal": bool(torch.equal(via_score, explicit)),
            "max_abs_deviation": dev,
            "input_norm_mean": float(torch.linalg.norm(x, dim=1).mean()),
            "transformed_norm_mean": float(torch.linalg.norm(model.transform(x), dim=1).mean()),
            "rule": (
                "`model.score(q)` 必须逐位等于 `normalize_query(transform(q)) @ keys.T`；"
                "本判据专门拦「评测侧漏掉变换层」这类回归（首版真实缺陷）"
            ),
        }


def train_scorer_evidence(model: VariantBModel, x: torch.Tensor) -> Dict[str, Any]:
    """**训练侧打分口径的自洽判据**（训练前提校准轮新增）。

    逐位断言三件事（全部要求 ``bitwise_equal``，不接受近似）：

    ① ``train_score(x, "normalized")`` **逐位等于** ``normalize_query(transform(x)) @ keys.T``
       —— 即默认训练口径与评测口径**逐位同一实现**，训练侧没有偷偷换尺子；
    ② ``train_score(x, "raw")`` **逐位等于** ``transform(x) @ keys.T``
       —— 保留的对照档确实是「未归一化」这条现状口径；
    ③ ``normalize_query_allow_zero(y)`` 与 ``normalize_query(y)`` 在**无零范数行**时
       **逐位相同**（有零范数行时前者保持全零、不报错，见
       :meth:`VariantBModel.normalize_query_allow_zero`）。

    参数
    ----
    model : VariantBModel
        待检查模型（任意训练状态）。
    x : torch.Tensor
        ``[B, D]`` 查询特征（**不得含零范数行**，否则 ③ 不适用）。

    返回
    ----
    Dict[str, Any]
        ``normalized_bitwise_equal`` / ``raw_bitwise_equal`` /
        ``allow_zero_matches_strict`` / ``has_zero_norm_row`` / ``all_bitwise_equal`` /
        ``rule``。
    """
    with torch.no_grad():
        xt = x.to(torch.float32)
        y = model.transform(xt)
        norms = torch.linalg.norm(xt.double(), dim=1)
        has_zero = bool((norms <= 0.0).any())
        raw_explicit = y @ model.keys.t()
        got_raw = model.train_score(xt, "raw")
        raw_equal = bool(torch.equal(got_raw, raw_explicit))
        if has_zero:
            # ③ 不适用：严格口径 `normalize_query` 在零范数行上**显式报错**（该契约不放宽），
            # 故只判 ①②；适用性如实标出，不静默当作通过。
            return {
                "n": int(xt.shape[0]),
                "has_zero_norm_row": True,
                "allow_zero_applicable": False,
                "normalized_bitwise_equal": None,
                "raw_bitwise_equal": bool(raw_equal),
                "allow_zero_matches_strict": None,
                "normalized_is_not_raw": None,
                "all_bitwise_equal": bool(raw_equal),
                "note": (
                    "输入含零范数行 ⇒ 严格口径 `normalize_query` 会显式报错，"
                    "第 ③ 项**不适用**（如实标 `allow_zero_applicable=False`，"
                    "本判据此时只由 ①② 组成）。"
                ),
                "rule": TRAIN_SCORER_RULE,
            }
        normed = VariantBModel.normalize_query(y)
        explicit = normed @ model.keys.t()
        got_norm = model.train_score(xt, "normalized")
        norm_equal = bool(torch.equal(got_norm, explicit))
        allow_zero_equal = bool(torch.equal(VariantBModel.normalize_query_allow_zero(y), normed))
    return {
        "n": int(xt.shape[0]),
        "has_zero_norm_row": False,
        "allow_zero_applicable": True,
        "normalized_bitwise_equal": bool(norm_equal),
        "raw_bitwise_equal": bool(raw_equal),
        "allow_zero_matches_strict": bool(allow_zero_equal),
        "normalized_is_not_raw": bool(not torch.equal(got_norm, got_raw)),
        "all_bitwise_equal": bool(norm_equal and raw_equal and allow_zero_equal),
        "note": (
            "`normalized_is_not_raw` 只作诊断（两者在 `T = I` 时本就相同，故不作为门禁项）；"
            "门禁只看 ①②③ 三条逐位相等。"
        ),
        "rule": TRAIN_SCORER_RULE,
    }


def gradient_flow_evidence(model: VariantBModel, x: torch.Tensor) -> Dict[str, Any]:
    """旁证：**恒等态下**变换层是否仍在计算图上、且两个参数都拿到非零梯度。

    存在理由（现场实测踩到的坑）：恒等快路径若写成 ``return x``，训练侧会立刻报
    ``RuntimeError: element 0 of tensors does not require grad``。本函数把它变成
    **可审计的现场证据**，而不是靠"记得别这么写"。
    """
    was_training = bool(model.transform.training)
    model.transform.train(True)
    xt = x.detach().clone().to(torch.float32)
    out = model.transform(xt)
    loss = out.pow(2).sum()
    loss.backward()
    grads: Dict[str, Any] = {}
    for name, param in model.transform.named_parameters():
        g = param.grad
        grads[str(name)] = {
            "grad_is_none": bool(g is None),
            "grad_norm": (float(g.detach().pow(2).sum().sqrt().item()) if g is not None else 0.0),
        }
    model.transform.zero_grad(set_to_none=True)
    model.transform.train(was_training)
    return {
        "has_grad_fn": bool(out.grad_fn is not None),
        "is_identity": bool(model.transform.is_identity()),
        "per_param": grads,
        "all_params_have_grad": bool(
            grads and all((not v["grad_is_none"]) and v["grad_norm"] > 0.0 for v in grads.values())
        ),
        "rule": (
            "恒等态下 forward 必须仍在计算图上（`out.grad_fn is not None`），"
            "且每个可训参数的 `.grad` 非 None、范数 > 0"
        ),
    }


# ---------------------------------------------------------------------------
# 2. 训练 / 评测划分（由「全量下标 − 查询行下标」求补集，不修改考卷代码）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariantBData:
    """变体 B 的冻结数据视图（键表全量 + 库行 / 查询行下标）。

    属性
    ----
    profile : str
        特征档名。
    table : EntryKeyTable
        **全量** 2665 行条目特征表（键表固定不动）。
    library_index : List[int]
        训练用库行下标（= 全量下标 − 查询行下标，**补集**；现场 1999 条）。
    query_index : List[int]
        评测用查询行下标（冻结划分的 query 行；现场 666 条）。
    query_text_sha256 : str
        查询行文本的规范化 SHA256（划分取证旁证）。
    evidence : Dict[str, Any]
        划分取证（来源 / 交集 / 并集 / 计数）。
    """

    profile: str
    table: EntryKeyTable
    library_index: List[int]
    query_index: List[int]
    query_text_sha256: str
    evidence: Dict[str, Any]

    @property
    def keys(self) -> torch.Tensor:
        """全量键表张量（``[2665, D]``）。"""
        return self.table.keys

    def query_features(self) -> torch.Tensor:
        """**原始编码器特征**（raw，逐行 L2 归一化）——载体不经任何 N3D 骨干。"""
        return self.table.keys[
            torch.tensor(self.query_index, dtype=torch.long)
        ].contiguous()

    def features_of(self, rows: Sequence[int]) -> torch.Tensor:
        """取任意行集合的**原始编码器特征**（``[len(rows), D]`` float32 连续张量）。

        存在理由（训练前提校准轮的恒等锚点）：锚点必须落在**训练行 1999** 上，
        而评测格落在查询行 666 上，两者走**同一条取值路径**（同一张冻结键表、
        同一份 raw 特征），差别只有行集合 —— 本方法是这两条路径的唯一取值点。
        """
        idx = [int(i) for i in rows]
        n = int(self.table.size)
        bad = [i for i in idx if not (0 <= i < n)]
        if bad:
            raise ValueError(f"行下标越界（表大小 {n}）：{bad[:5]}")
        return self.table.keys[torch.tensor(idx, dtype=torch.long)].contiguous()

    def row_label(self, rows: Optional[Sequence[int]] = None) -> str:
        """行集合的**口径标签**（进产物，避免读者把两种行集合读混）。

        ``None`` = 冻结划分的查询行（评测口径）；否则按现场集合与冻结集合的相等关系判定。
        """
        if rows is None:
            return "query_rows"
        got = [int(i) for i in rows]
        if got == [int(i) for i in self.query_index]:
            return "query_rows"
        if got == [int(i) for i in self.library_index]:
            return "library_rows"
        return "custom_rows"

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化摘要（**不落盘任何特征矩阵**）。"""
        return {
            "profile": str(self.profile),
            "table": self.table.meta(),
            "n_library": int(len(self.library_index)),
            "n_query": int(len(self.query_index)),
            "library_index_head": [int(i) for i in self.library_index[:10]],
            "query_index_head": [int(i) for i in self.query_index[:10]],
            "query_text_sha256": str(self.query_text_sha256),
            "evidence": dict(self.evidence),
        }


def build_data(
    table: EntryKeyTable,
    *,
    query_index: Sequence[int],
    profile: str,
    query_texts: Optional[Sequence[str]] = None,
    source: Optional[Dict[str, Any]] = None,
) -> VariantBData:
    """由统一条目表 + 冻结查询行下标构造变体 B 的冻结数据视图。

    库行下标 = **全量下标 − 查询行下标**（集合求补集，**不修改 1a 考卷代码**）。

    参数
    ----
    table : EntryKeyTable
        全量条目特征表（现场 2665 行）。
    query_index : Sequence[int]
        冻结划分出的查询行下标（现场 666 条）。
    profile : str
        特征档名。
    query_texts : Optional[Sequence[str]]
        查询行原文（用于文本指纹旁证；``None`` = 取条目表同行的 ``outputs``）。
    source : Optional[Dict[str, Any]]
        划分来源取证（并入 ``evidence``）。

    返回
    ----
    VariantBData
        冻结数据视图。

    异常
    ------
    ValueError
        查询下标越界 / 重复，或补集规模与冻结口径（2665 / 1999 / 666）不符。
    """
    q = [int(i) for i in query_index]
    if len(set(q)) != len(q):
        raise ValueError("查询行下标存在重复；划分不可复核")
    n = int(table.size)
    if any(not (0 <= i < n) for i in q):
        bad = [i for i in q if not (0 <= i < n)]
        raise ValueError(f"查询行下标越界（表大小 {n}）：{bad[:5]}")
    q_set = set(q)
    library = [i for i in range(n) if i not in q_set]
    if n != int(FULL_TABLE_ROWS) or len(q) != int(QUERY_ROWS) or len(library) != int(LIBRARY_ROWS):
        raise ValueError(
            "变体 B 的划分口径已冻结为「全量 2665 / 库 1999 / 查询 666」，"
            f"现场得到 全量 {n} / 库 {len(library)} / 查询 {len(q)}；"
            "拒绝在漂移的划分上出指标"
        )
    texts = (
        [str(x) for x in query_texts]
        if query_texts is not None
        else [str(table.outputs[i]) for i in q]
    )
    evidence: Dict[str, Any] = {
        "rule": (
            "库行下标 = 「全量下标 − 查询行下标」求**补集**（不修改 1a 考卷代码）；"
            "评测查询 = 冻结划分的 query 行（金标 = 自身行下标）"
        ),
        "n_table": int(n),
        "n_library": int(len(library)),
        "n_query": int(len(q)),
        "intersection": int(len(q_set & set(library))),
        "union": int(len(q_set | set(library))),
        "query_index_sorted": bool(q == sorted(q)),
    }
    if evidence["intersection"] != 0 or evidence["union"] != n:
        raise ValueError(
            f"库/查询划分不自洽：交集 {evidence['intersection']}，"
            f"并集 {evidence['union']} != {n}"
        )
    evidence.update(dict(source or {}))
    return VariantBData(
        profile=str(profile),
        table=table,
        library_index=library,
        query_index=q,
        query_text_sha256=ET.sha256_bytes(ET.canonical_dumps(texts)),
        evidence=evidence,
    )


def load_profile_data(
    profile: str, *, product_dir: str = "", log: Any = None
) -> VariantBData:
    """按特征档构造冻结数据视图（**只读**消费 1a 的考卷构造路径）。

    编码器不可用（缺 ``transformers`` / ``models/bge-m3`` 缺失）时由
    :func:`n3d_qa_learn.encoders.build_vectorizer` 抛 ``EncoderUnavailableError``，
    本函数**不吞异常**（CLI 侧给出可读报文 + 退码非 0，不静默跳过）。

    参数
    ----
    profile : str
        特征档名（``lexical-88`` / ``bge-m3-1024``）。
    product_dir : str
        ``n3d_qa`` 冻结产物目录（空 = 自动定位）。
    log : Any
        可调用日志（缺省不打印）。

    返回
    ----
    VariantBData
        全量键表 + 库行 / 查询行下标。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    from .step2 import (
        load_doclines_meta,
        load_row_index,
        load_text_rows,
        reproduce_doc_split,
        resolve_product_dir,
    )

    resolved = resolve_product_dir(str(product_dir))
    _log(f"[variantb] 读取冻结行表 / 档 {profile}：{resolved}")
    rows = load_text_rows(resolved)
    vec = ET.build_vectorizer_for(str(profile), ET.ROLE_TEXT_LINE)
    bundle: TextEntryBundle = ET.text_entry_table_from_rows(
        vec, rows, profile=str(profile), product_dir=resolved
    )
    meta = load_doclines_meta(resolved)
    row_index = load_row_index(resolved)
    split = reproduce_doc_split(rows, row_index, meta)
    source = {
        "product_dir": str(resolved),
        "split_evidence": dict(split.evidence),
        "library_rows_frozen": int(split.n_library),
        "query_rows_frozen": int(split.n_query),
        "text_row_key_table_sha256": str(bundle.text_row_table.sha256()),
        "entry_table_sha256": str(bundle.table.sha256()),
        "encoder_fingerprint": str(vec.fingerprint()),
        "role": ET.ROLE_TEXT_LINE,
    }
    data = build_data(
        bundle.table,
        query_index=[int(i) for i in split.query_index],
        profile=str(profile),
        query_texts=[str(bundle.table.outputs[int(i)]) for i in split.query_index],
        source=source,
    )
    _log(
        f"[variantb] 档 {profile}：键表 {data.table.size}×{data.table.key_dim} / "
        f"训练库行 {len(data.library_index)} / 评测查询行 {len(data.query_index)} / "
        f"交集 {data.evidence['intersection']}"
    )
    return data


# ---------------------------------------------------------------------------
# 3. 评测：变体 B vs KNN 基线（逐格）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellMetrics:
    """一个评测格（特征档 × 扰动类型 × 档位）的指标。

    属性
    ----
    recall_at_1 / recall_at_5 : float
        命中率（金标 = 查询行自身行下标）。
    hit_at_1 / hit_at_5 : int
        命中条数（**小落差格必须给条数**，避免把 1~5 条读成比例）。
    n : int
        查询数。
    rank_max : int
        最大名次。
    discrepancy : Dict[str, Any]
        变体 B 与 KNN 的 top-1 **逐条一致性**取证（恒等门禁用）。
    """

    recall_at_1: float
    recall_at_5: float
    hit_at_1: int
    hit_at_5: int
    n: int
    rank_max: int
    discrepancy: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化。"""
        return {
            "recall_at_1": float(self.recall_at_1),
            "recall_at_5": float(self.recall_at_5),
            "hit_at_1": int(self.hit_at_1),
            "hit_at_5": int(self.hit_at_5),
            "n": int(self.n),
            "rank_max": int(self.rank_max),
            "discrepancy": dict(self.discrepancy),
        }


def _rank_and_top1(
    score: torch.Tensor,
    gold: torch.Tensor,
    *,
    topk: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """确定性名次 + top-1 行下标（名次口径与 1a 逐字一致）。

    口径
    ----
    ``rank_i = 1 + #{ j : score[i,j] > score[i, gold_i] }``（并列按超出条数计，与 topk
    顺序无关）；top-1 行下标 = ``argmax``（并列取**最小行下标**，从而逐条一致性判定是
    确定性的、与 ``torch.topk`` 的平局行为无关）。

    返回
    ----
    Tuple[np.ndarray, np.ndarray]
        ``(ranks [M] int64, top1 [M] int64)``。
    """
    m = int(score.shape[0])
    ranks = np.zeros(m, dtype=np.int64)
    top1 = np.zeros(m, dtype=np.int64)
    step = max(1, int(batch_size))
    for b0 in range(0, m, step):
        chunk = score[b0 : b0 + step]
        g = gold[b0 : b0 + step]
        gold_scores = chunk[torch.arange(int(chunk.shape[0])), g]
        ranks[b0 : b0 + int(chunk.shape[0])] = (
            1 + (chunk > gold_scores.unsqueeze(1)).sum(dim=1)
        ).numpy().astype(np.int64)
        top1[b0 : b0 + int(chunk.shape[0])] = torch.argmax(chunk, dim=1).numpy()
    return ranks, top1


def compare_top1(b_path: np.ndarray, k_path: np.ndarray) -> Dict[str, Any]:
    """两条 top-1 路径的**逐条一致性**取证（恒等门禁的直接判据）。"""
    a = np.asarray(b_path, dtype=np.int64)
    b = np.asarray(k_path, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError(f"两条 top-1 路径形状不一致：{a.shape} vs {b.shape}")
    bad = np.flatnonzero(a != b)
    return {
        "n": int(a.size),
        "n_mismatch": int(bad.size),
        "mismatch_frac": float(bad.size / max(1, int(a.size))),
        "mismatch_head": [
            {
                "query_pos": int(i),
                "variant_b_top1": int(a[i]),
                "knn_top1": int(b[i]),
            }
            for i in bad[:20]
        ],
        "all_equal": bool(bad.size == 0),
    }


def eval_cell(
    model: VariantBModel,
    data: VariantBData,
    *,
    kind: str,
    level: str,
    clean: bool = False,
    variant: int = 0,
    seed: int = VARIANT_B_SEED,
    topk: int = R.TOPK,
    batch_size: int = R.BATCH_SIZE,
    rows: Optional[Sequence[int]] = None,
) -> Tuple[CellMetrics, CellMetrics, Dict[str, Any]]:
    """评测一个格：**同一次查询构造**下同时算变体 B 与 KNN 两条路径。

    关键口径
    --------
    * 查询 = 指定行集合的**原始编码器特征**（逐行 L2 归一化）；
    * ``rows=None`` = 冻结划分的**查询行 666**（评测口径，历史行为逐位不变）；
      ``rows=data.library_index`` = **训练行 1999**（训练前提校准轮的恒等锚点 / 训练侧
      训练后自命中口径）—— 两条路径共用本函数，故「同一扰动实现、同一派生种子、
      同一名次口径」是结构上成立的，差别只有行集合；
    * ``clean=True`` 不扰动；否则走 1a 的 :func:`robust_eval.perturb_matrix`
      （**唯一实现**，含重新 L2 归一化与零范数显式报错）；
    * **KNN 路径** = 归一化查询与冻结键表的余弦内积；
    * **变体 B 路径** = ``T`` 变换后重新归一化再内积（同一条内积口径）；
    * 金标 = 查询行自身在**全量键表**中的行下标。

    返回
    ----
    Tuple[CellMetrics, CellMetrics, Dict[str, Any]]
        ``(变体 B 指标, KNN 指标, 扰动与路径取证)``。
    """
    index = (
        [int(i) for i in data.query_index]
        if rows is None
        else [int(i) for i in rows]
    )
    q = data.features_of(index)
    gold = torch.tensor(index, dtype=torch.long)
    detail: Dict[str, Any] = {
        "clean": bool(clean),
        "kind": str(kind),
        "level": str(level),
        "variant": int(variant),
        "n_query": int(q.shape[0]),
        "n_keys": int(model.n_keys),
        "row_source": data.row_label(index),
        "row_source_rule": (
            "`query_rows` = 冻结划分的查询行 666（评测口径）；"
            "`library_rows` = 训练行 1999（锚点 / 训练侧口径）；"
            "`custom_rows` = 显式传入的其它行集合"
        ),
        "query_source": "raw（原始编码器特征，未经 N3D 骨干）",
    }
    if not clean:
        q_np, p_detail = R.perturb_matrix(
            q.numpy(), str(kind), str(level), variant=int(variant), seed=int(seed)
        )
        q = torch.from_numpy(np.ascontiguousarray(q_np, dtype=np.float32))
        detail["perturb"] = dict(p_detail)
    with torch.no_grad():
        b_score = model.score(q)
        k_score = model.normalize_query(q) @ model.keys.t()
    b_ranks, b_top1 = _rank_and_top1(
        b_score, gold, topk=int(topk), batch_size=int(batch_size)
    )
    k_ranks, k_top1 = _rank_and_top1(
        k_score, gold, topk=int(topk), batch_size=int(batch_size)
    )
    kk = int(topk)
    m = int(q.shape[0])
    b_m = CellMetrics(
        recall_at_1=float((b_ranks == 1).sum() / max(1, m)),
        recall_at_5=float(((b_ranks >= 1) & (b_ranks <= kk)).sum() / max(1, m)),
        hit_at_1=int((b_ranks == 1).sum()),
        hit_at_5=int(((b_ranks >= 1) & (b_ranks <= kk)).sum()),
        n=int(m),
        rank_max=int(b_ranks.max()) if m else 0,
        discrepancy=compare_top1(b_top1, k_top1),
    )
    k_m = CellMetrics(
        recall_at_1=float((k_ranks == 1).sum() / max(1, m)),
        recall_at_5=float(((k_ranks >= 1) & (k_ranks <= kk)).sum() / max(1, m)),
        hit_at_1=int((k_ranks == 1).sum()),
        hit_at_5=int(((k_ranks >= 1) & (k_ranks <= kk)).sum()),
        n=int(m),
        rank_max=int(k_ranks.max()) if m else 0,
        discrepancy={},
    )
    detail["transform_io"] = model.transform.last_io_evidence()
    detail["norm_path"] = {
        "rule": (
            "查询侧唯一归一化实现 = `VariantBModel.normalize_query`"
            "（float64 求范数；逐元素 float32 求和恰等于其 float64 精确值，"
            "故与 `entry_table.l2_normalize_rows` 逐位一致）"
        ),
        "raw_query_norm_min": float(torch.linalg.norm(q, dim=1).min()),
        "raw_query_norm_max": float(torch.linalg.norm(q, dim=1).max()),
    }
    return b_m, k_m, detail


def cell_grid(
    model: VariantBModel,
    data: VariantBData,
    *,
    seed: int = VARIANT_B_SEED,
    topk: int = R.TOPK,
    batch_size: int = R.BATCH_SIZE,
    variant: int = 0,
    rows: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """跑完 clean + 9 个扰动格的完整网格（变体 B 与 KNN 同格对照）。

    参数
    ----
    rows : Optional[Sequence[int]]
        行集合；``None`` = 冻结划分的查询行 666（评测口径，历史行为逐位不变），
        ``data.library_index`` = 训练行 1999（锚点 / 训练侧口径）。

    返回
    ----
    Dict[str, Any]
        ``cells``（逐格指标 + Δ）、``top1_discrepancy_vs_knn``（变体 B 与 KNN 的 top-1
        逐条差异**统计量**）、``gap_weak_minus_strong``（分档落差）、``error_space``
        （空间不足格）。

    [!] **字段命名（W1 修复，皋陶审查）**：本函数此前把该统计量叫 ``identity_gate``，
    而它在**训练后**被同一个函数复用 ⇒ 产物里出现「同名字段相反结论」：
    ``per_profile[...]["cell_grid"]["identity_gate"]["all_equal"] = False``
    与同文件顶层 ``identity_gate.passed = True`` 并存，读者会误判门禁失败。
    现改名为 :data:`TOP1_DISCREPANCY_KEY`，并显式标注 ``model_is_identity`` /
    ``computed_after_training``；**门禁**只在 :func:`identity_gate` 里判定。
    """
    cells: List[Dict[str, Any]] = []
    b_clean, k_clean, d_clean = eval_cell(
        model, data, kind="noise", level="weak", clean=True,
        seed=int(seed), topk=int(topk), batch_size=int(batch_size), rows=rows,
    )
    cells.append(
        {
            "cell": "clean",
            "kind": "none",
            "level": "clean",
            "variant_b": b_clean.as_dict(),
            "knn": k_clean.as_dict(),
            "delta": {
                "recall_at_1": float(b_clean.recall_at_1 - k_clean.recall_at_1),
                "recall_at_5": float(b_clean.recall_at_5 - k_clean.recall_at_5),
                "hit_at_1": int(b_clean.hit_at_1 - k_clean.hit_at_1),
            },
            "evidence": d_clean,
        }
    )
    for kind in R.PERTURB_TYPES:
        for level in R.PERTURB_LEVELS:
            b_m, k_m, det = eval_cell(
                model, data, kind=str(kind), level=str(level), variant=int(variant),
                seed=int(seed), topk=int(topk), batch_size=int(batch_size), rows=rows,
            )
            cells.append(
                {
                    "cell": f"{kind}/{level}",
                    "kind": str(kind),
                    "level": str(level),
                    "variant_b": b_m.as_dict(),
                    "knn": k_m.as_dict(),
                    "delta": {
                        "recall_at_1": float(b_m.recall_at_1 - k_m.recall_at_1),
                        "recall_at_5": float(b_m.recall_at_5 - k_m.recall_at_5),
                        "hit_at_1": int(b_m.hit_at_1 - k_m.hit_at_1),
                    },
                    "evidence": det,
                }
            )
    mismatches = {c["cell"]: c["variant_b"]["discrepancy"]["n_mismatch"] for c in cells}
    is_identity = bool(model.transform.is_identity())
    discrepancy_ev = {
        "comment": (
            "本块是**统计量**（变体 B 与 KNN 的 top-1 逐条差异），**不是门禁**；"
            "门禁只在 `identity_gate()`（训练前 `T = I` 时判定）里给出。"
            "训练后被调用时 `all_cells_equal` 通常为 False，属**预期**，"
            "不得读成门禁失败。"
        ),
        "rule": (
            "统计「变体 B 的 top-1 与余弦最近邻逐条一致」的条数；`T = I` 时应当"
            "逐格 `n_mismatch == 0`（该性质在训练前由 `identity_gate()` 作为硬门禁判定）"
        ),
        "n_cells": int(len(cells)),
        "n_cells_all_equal": int(
            sum(1 for c in cells if c["variant_b"]["discrepancy"]["all_equal"])
        ),
        "per_cell_mismatch": mismatches,
        "all_cells_equal": bool(all(v == 0 for v in mismatches.values())),
        "model_is_identity": bool(is_identity),
        "computed_after_training": bool(not is_identity),
    }
    gaps: List[Dict[str, Any]] = []
    for kind in R.PERTURB_TYPES:
        sub = [c for c in cells if c["kind"] == str(kind)]
        if len(sub) != len(R.PERTURB_LEVELS):
            continue
        weak = next(c for c in sub if c["level"] == "weak")
        strong = next(c for c in sub if c["level"] == "strong")
        b_gap = float(weak["variant_b"]["recall_at_1"] - strong["variant_b"]["recall_at_1"])
        k_gap = float(weak["knn"]["recall_at_1"] - strong["knn"]["recall_at_1"])
        gaps.append(
            {
                "kind": str(kind),
                "variant_b_gap": float(b_gap),
                "knn_gap": float(k_gap),
                "delta_gap": float(b_gap - k_gap),
                "note": "落差 = 弱档 R@1 − 强档 R@1（1a 口径）",
            }
        )
    error_space = [
        {
            "cell": c["cell"],
            "knn_hit_at_1": int(c["knn"]["hit_at_1"]),
            "knn_errors": int(c["knn"]["n"] - c["knn"]["hit_at_1"]),
            "n": int(c["knn"]["n"]),
            "note": (
                "KNN 侧错误空间 <= 1% 查询数 ⇒ 该格「Δ >= 0」的判别力弱、"
                "「Δ > 0」几乎不可能成立，须显式标注"
            ),
        }
        for c in cells
        if int(c["knn"]["n"] - c["knn"]["hit_at_1"]) <= max(1, int(0.01 * c["knn"]["n"]))
    ]
    return {
        "profile": str(data.profile),
        "row_source": data.row_label(rows),
        "n_rows": int(cells[0]["variant_b"]["n"]) if cells else 0,
        "model_is_identity": bool(is_identity),
        "computed_after_training": bool(not is_identity),
        "granularity": (TRAIN_GRANULARITY_NOTE),
        "cells": cells,
        TOP1_DISCREPANCY_KEY: discrepancy_ev,
        "gap_weak_minus_strong": gaps,
        "error_space": error_space,
        "knn_baseline_rule": (
            "KNN 路径 = 归一化查询与**冻结键表**的余弦 top-1（本模块现场重算）；"
            "键表由冻结产物构造且不参与梯度 ⇒ 结果与训练完全无关"
        ),
    }


# ---------------------------------------------------------------------------
# 4. 扰动自监督训练器（冻结库全表作输出层、交叉熵、Adam）
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """训练配置（构造期校验）。

    参数
    ----
    seed : int
        训练 seed（冻结 42；**只驱动局部 generator**，不消耗全局 RNG）。
    epochs : int
        训练轮数（``perturb`` 模式下每步轮转一个扰动格）。
    batch_size : int
        批大小。
    lr : float
        Adam 学习率。
    weight_decay : float
        Adam 权重衰减。
    train_mode : str
        ``perturb``（主方案：扰动自监督）或 ``clean``（A 组对照：干净自监督）。
    train_scorer : str
        训练侧打分口径（``normalized`` = 默认，与评测同一口径；``raw`` = 现状未归一化口径）。
        见 :data:`TRAIN_SCORER_RULE` —— 它**只影响训练侧损失**，评测侧恒为归一化口径。
    log_every : int
        每多少步打印一次（只进日志，**不入产物**）。
    """

    seed: int = VARIANT_B_SEED
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    train_mode: str = "perturb"
    train_scorer: str = DEFAULT_TRAIN_SCORER
    log_every: int = 20

    def __post_init__(self) -> None:
        """构造期不变量（非法配置立即报错，不静默回落）。"""
        if int(self.epochs) < 1:
            raise ValueError(f"epochs 必须 >= 1，当前 {self.epochs}")
        if int(self.batch_size) < 1:
            raise ValueError(f"batch_size 必须 >= 1，当前 {self.batch_size}")
        if float(self.lr) <= 0.0:
            raise ValueError(f"lr 必须 > 0，当前 {self.lr}")
        if float(self.weight_decay) < 0.0:
            raise ValueError(f"weight_decay 必须 >= 0，当前 {self.weight_decay}")
        if str(self.train_mode) not in TRAIN_MODES:
            raise ValueError(
                f"未知训练模式 {self.train_mode!r}；可用 = {list(TRAIN_MODES)}"
            )
        if str(self.train_scorer) not in TRAIN_SCORERS:
            raise ValueError(
                f"未知训练侧打分口径 {self.train_scorer!r}；可用 = {list(TRAIN_SCORERS)}"
            )

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化。"""
        return {
            "seed": int(self.seed),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "train_mode": str(self.train_mode),
            "train_scorer": str(self.train_scorer),
            "train_scorer_rule": TRAIN_SCORER_RULE,
            "optimizer": "Adam（唯一优化器；只优化变换层参数）",
            "loss": (
                "CrossEntropyLoss(logits = train_score(T(q), train_scorer), "
                "label = 该行自身在全量键表中的行下标)；"
                "`normalized` 时 logits = `normalize_query(T(q)) @ keys.T`（与评测同一口径），"
                "`raw` 时 logits = `T(q) @ keys.T`（现状未归一化口径）"
            ),
            "shuffle_seed": int(SHUFFLE_SEED),
            "perturb_cells": [[str(k), str(lv)] for k, lv in TRAIN_PERTURB_CELLS],
            "rng_rule": (
                "全部随机数来自**局部** `torch.Generator`（打乱用 seed=SHUFFLE_SEED，"
                "扰动走 1a 的派生种子），**不消耗全局 RNG**"
            ),
        }


#: 1a 零范数报文里「行下标」片段的字面量（1a 的报文由 ``robust_eval`` 自己产出，未改动）。
RETRIEVAL_ZERO_TOKEN: str = "行下标 = "

#: 修复分支的最大轮次（每轮至少替换 1 行；1a 的报文只列前 20 行，故可能需要多轮）。
#: 超过该轮次仍报错即 **fail-closed**（不静默放过）。
RETRIEVAL_MAX_ROUNDS: int = 64

#: 取回路径的**唯一规则文本**（进产物 ``evidence``，与实现同源）。
RETRIEVAL_RULE: str = (
    "**成功分支 = 1a 本尊**：无零范数行时原样返回 `robust_eval.perturb_matrix` 的输出与它"
    "自己的取证字典（`retrieval_mode = direct_1a`），本模块不做任何再实现。"
    "**修复分支 = 1a 本尊 + 置零**：出现零范数行时，把 1a 报出的那些行的**输入**替换为"
    "确定性非零哨兵、**重新调用 1a 本尊**，只丢弃这些行的输出并按训练侧下发约定置零"
    "（`retrieval_mode = repaired_via_1a`）；替换前后用两个不同哨兵做「行独立性」逐字节"
    "交叉验证，不一致即 fail-closed。**不再**按公式重放（W2 修复前的做法：那份取证是 1a 的"
    "函数内局部变量，反查恒命中、raise 分支不可达，docstring 承诺未实现）。"
)


def _zero_rows_from_error(message: str) -> List[int]:
    """从 1a 的零范数报文中取回**该次调用**报出的零范数行下标。

    1a 的报文形态（``robust_eval.perturb_matrix`` 自己产出，本模块未改动）::

        扰动 mask/strong 后出现零范数行：n_zero_norm=1，行下标 = [23]；拒绝静默产出 NaN …

    该片段即 ``l2_normalize_rows`` 的 ``zero_norm_rows_head``（**上限 20 行**），
    故调用方必须按轮次补齐（见 :func:`_perturb_allow_zero` 的修复分支）。
    报文形态不认识时 **fail-closed** 抛可读 ``ValueError``，绝不猜。

    参数
    ----
    message : str
        1a 抛出的 ``ValueError`` 的报文字符串。

    返回
    ----
    List[int]
        报文中列出的行下标（可能只是前 20 个）。
    """
    parts = str(message).split(RETRIEVAL_ZERO_TOKEN, 1)
    if len(parts) != 2 or "]" not in parts[1]:
        raise ValueError(
            "取回路径 fail-closed：无法从 1a 的零范数报文中解析行下标"
            f"（期望包含 {RETRIEVAL_ZERO_TOKEN!r} 与 ']'）；报文 = {str(message)!r}"
        )
    blob = parts[1].split("]", 1)[0] + "]"
    try:
        rows = json.loads(blob)
        out = [int(x) for x in rows]
    except (ValueError, TypeError, json.JSONDecodeError) as exc:  # noqa: PERF203
        raise ValueError(
            "取回路径 fail-closed：1a 报文中的行下标片段不是合法整数列表 "
            f"（片段 = {blob!r}）；报文 = {str(message)!r}"
        ) from exc
    if any(i < 0 for i in out):
        raise ValueError(f"取回路径 fail-closed：解析出的行下标含负值 {out}；报文 = {str(message)!r}")
    return out


def _sentinel_row(dim: int, salt: int) -> np.ndarray:
    """确定性**非零**哨兵行（用于替换零范数行的输入；无随机数、不进产物）。

    ``x[j] = 0.5 + 0.25 * ((j + salt) mod 7)`` ⇒ 每个元素都落在 ``[0.5, 2.0]``，
    逐元素非零 ⇒ 该行在 ``mask`` 下必有非零保留维、在 ``nmag`` 下不可能恰好被平移到零。
    """
    idx = np.arange(int(dim), dtype=np.float64)
    return (0.5 + 0.25 * np.mod(idx + float(int(salt)), 7.0)).astype(np.float32)


def _with_sentinel_rows(src: np.ndarray, rows: Sequence[int], *, salt: int) -> np.ndarray:
    """把 ``rows`` 指定的行的**输入**替换为确定性非零哨兵（其余行原样），返回副本。"""
    work = np.ascontiguousarray(np.asarray(src, dtype=np.float32)).copy()
    dim = int(work.shape[1])
    for i in [int(r) for r in rows]:
        work[i] = _sentinel_row(dim, int(salt))
    return work


def _perturb_allow_zero(
    matrix: np.ndarray,
    kind: str,
    level: str,
    *,
    variant: int = 0,
    seed: int = VARIANT_B_SEED,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """取回「扰动后可能出现零范数行」的矩阵 —— **仍然走 1a 的实现**。

    存在理由（现场实测的边界，真实触发过）
    -------------------------------------
    1a 的 :func:`robust_eval.perturb_matrix` 在**最后一步**调用
    :func:`entry_table.l2_normalize_rows`，一旦出现零范数行就抛 ``ValueError``。
    训练侧对这个矩阵的处理口径是「**保留零行 + 在损失中掩码**」（见 :func:`zero_norm_mask`），
    因此需要在不改动 1a 一个字符的前提下拿到它。

    两条分支（**W2 修复后的口径，必须如实区分**）
    ------------------------------------------
    * **成功分支 = 1a 本尊**：无零范数行时，**原样返回** 1a 自己算出的矩阵与它自己的取证
      字典（``retrieval_mode = "direct_1a"``），本函数不做任何再实现；
    * **修复分支 = 1a 本尊 + 置零**：出现零范数行时，把 1a 报出的那些行的**输入**替换为
      确定性非零哨兵，**重新调用 1a 本尊**，只丢弃这些行的输出并按训练侧下发约定置零
      （``retrieval_mode = "repaired_via_1a"``）。

    [!] **W2 修复（皋陶审查）**：修复前的实现宣称「按 1a 的 ``raw_perturbed_sha256``
    逐字节反查取回 1a 的中间矩阵」，但 1a 抛错时那份 ``detail`` 是 ``perturb_matrix`` 的
    **函数内局部变量**、对外不存在 ⇒ 候选比对**恒命中**、``raise`` 分支**不可达**、
    docstring 的承诺**未实现**（证据强度被高估）。现改为**实际调用 1a 实现**取回，
    并配 :func:`perturb_retrieval_reverse_check` 的反向验证（monkeypatch 共享层必须被检出）。

    为什么「替换输入行再调用 1a」是合法的
    ------------------------------------
    1a 的三种扰动都**逐行独立**：``noise`` 的 ``randn`` 只依赖形状与派生种子；``mask`` 的
    ``randperm`` 每行消耗同一 generator 的同一段（消耗次数由行数决定，与取值无关）；
    ``nmag`` 是闭式。故替换若干行的**输入**只改变这些行的**输出**，其余行逐位不变。
    该前提**不是靠推理**：修复分支会用**两个不同的哨兵**各跑一次 1a，逐字节比对
    「未被替换行」的输出（``row_independence_verified``），不一致即 fail-closed。

    参数
    ----
    matrix : np.ndarray
        ``[N, D]`` 特征。
    kind / level / variant / seed : 与 :func:`robust_eval.perturb_matrix` 同义。

    返回
    ----
    Tuple[np.ndarray, Dict[str, Any]]
        ``(扰动后未归一化的 float32 矩阵；零范数行按训练侧约定置为全零, 取证字典)``。

    异常
    ------
    ValueError
        1a 报出的零范数行下标无法取回、替换后仍报错、或「行独立性」交叉校验不一致
        （**全部 fail-closed**，绝不静默替换实现）。
    """
    src = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    if src.ndim != 2:
        raise ValueError(f"待扰动矩阵必须是二维 [N, D]，当前形状 = {src.shape}")
    # --- 成功分支：直接调用 1a，返回值就是 1a 本尊的矩阵与取证 -------------
    try:
        out_ok, detail_ok = R.perturb_matrix(
            src, str(kind), str(level), variant=int(variant), seed=int(seed)
        )
        evidence_ok = dict(detail_ok)
        evidence_ok["retrieval_mode"] = "direct_1a"
        evidence_ok["retrieval_rule"] = RETRIEVAL_RULE
        return np.ascontiguousarray(out_ok, dtype=np.float32), evidence_ok
    except ValueError as exc:
        if "零范数" not in str(exc):
            raise
        first_error = str(exc)
    # --- 修复分支：替换零范数行的**输入**后重新调用 1a 本尊 ---------------
    repaired: List[int] = []
    rounds: List[Dict[str, Any]] = []
    last_error = first_error
    out: Optional[np.ndarray] = None
    detail: Dict[str, Any] = {}
    for attempt in range(1, int(RETRIEVAL_MAX_ROUNDS) + 1):
        reported = _zero_rows_from_error(last_error)
        fresh = [int(i) for i in reported if int(i) not in set(repaired)]
        if not fresh:
            raise ValueError(
                "取回路径 fail-closed：1a 报出的零范数行下标与已替换行集合一致，但仍报错；"
                f"kind={kind}, level={level}, 已替换行 = {repaired}"
            )
        repaired.extend(fresh)
        rounds.append(
            {
                "round": int(attempt),
                "reported_rows": [int(i) for i in reported],
                "newly_replaced": [int(i) for i in fresh],
            }
        )
        work = _with_sentinel_rows(src, repaired, salt=int(attempt))
        try:
            out_a, detail_a = R.perturb_matrix(
                work, str(kind), str(level), variant=int(variant), seed=int(seed)
            )
        except ValueError as exc2:
            if "零范数" not in str(exc2):
                raise
            last_error = str(exc2)
            continue
        out = np.ascontiguousarray(out_a, dtype=np.float32)
        detail = dict(detail_a)
        break
    if out is None:
        raise ValueError(
            "取回路径 fail-closed：替换零范数行的输入后，1a 仍在 "
            f"{int(RETRIEVAL_MAX_ROUNDS)} 轮内报零范数；kind={kind}, level={level}"
        )
    # --- 行独立性交叉验证（换一个哨兵再跑一次 1a，未被替换行必须逐位不变）----
    keep_rows = [i for i in range(int(src.shape[0])) if i not in set(repaired)]
    work_b = _with_sentinel_rows(src, repaired, salt=int(1000 + len(repaired)))
    out_b, _detail_b = R.perturb_matrix(
        work_b, str(kind), str(level), variant=int(variant), seed=int(seed)
    )
    out_b = np.ascontiguousarray(np.asarray(out_b, dtype=np.float32))
    same_bytes = bool(
        not keep_rows
        or np.ascontiguousarray(out_b[keep_rows], dtype=np.float32).tobytes()
        == np.ascontiguousarray(out[keep_rows], dtype=np.float32).tobytes()
    )
    if not same_bytes:
        raise ValueError(
            "取回路径 fail-closed：「行独立性」交叉验证不一致 —— 用两个不同哨兵替换同样的行后，"
            "未被替换行的 1a 输出**不逐位相同**，说明扰动并非逐行独立，替换法在此不成立；"
            f"kind={kind}, level={level}, 替换行 = {repaired}"
        )
    repaired_norms = np.linalg.norm(out[repaired].astype(np.float64), axis=1)
    if not bool(np.all(repaired_norms > 0.0)):
        raise ValueError(
            "取回路径 fail-closed：被替换行的 1a 输出出现零范数（哨兵行本身也被扰动成零），"
            f"取回口径不成立；kind={kind}, level={level}, 替换行 = {repaired}"
        )
    # 训练侧下发约定：零范数行**按下发口径置零**（该行随后必被 zero_norm_mask 掩码）
    out_final = out.copy()
    out_final[repaired] = 0.0
    evidence = dict(detail)
    evidence.update(
        {
            "retrieval_mode": "repaired_via_1a",
            "retrieval_rule": RETRIEVAL_RULE,
            "repaired_rows": [int(i) for i in repaired],
            "repair_rounds": rounds,
            "row_independence_verified": True,
            "row_independence_evidence": (
                "用两个不同的确定性哨兵各调用一次 1a，未被替换行的归一化输出**逐字节相同**"
                f"（比较行数 = {len(keep_rows)}）"
            ),
            "repaired_rows_output_nonzero": True,
            "n_zero_norm": int(len(repaired)),
            "n_zero_norm_repaired_call": int(detail.get("n_zero_norm", 0)),
            "first_error": first_error,
            "contaminated_fields": ["mean_row_displacement"],
            "contaminated_note": (
                "上述字段由 1a 在**含哨兵行**的输入上算出，含被替换行的贡献，"
                "**不代表**原输入的位移量；本模块不据此字段做任何判定（如实登记，不静默使用）。"
            ),
        }
    )
    return np.ascontiguousarray(out_final, dtype=np.float32), evidence


def zero_norm_mask(x: torch.Tensor) -> torch.Tensor:
    """返回 ``[B]`` 布尔掩码：``True`` = 该行范数 > 0（**未被扰动成零向量**）。

    存在理由（现场实测的边界）：训练侧对**每一批单独**施加扰动时，「批内某行被扰动后整行
    归零」是真实可达的事件（``lexical-88`` 的 ``mask/strong`` 遮蔽 44/88 维，某一批的第
    118 行本只有 9 个非零维且全部落在被遮蔽的一半里）。该行**不静默丢弃**，而是
    **在损失中掩码**：掩码取自**变换前**的扰动后输入 ``q``，故该行**始终**被排除在
    交叉熵与梯度之外。

    [!] **措辞更正（离朱 R59 F4）**：本函数与相关 docstring 曾写「它的 logits 恒为 0」，
    该说法**只在 ``T = I`` 时成立** —— 训练后 ``T(0) = b != 0``，零向量经变换后并非零向量
    （现场探针测得该行 logits 量级 `0.3487` / `0.2163`）。**功能无缺陷**（掩码来自变换前的
    ``q``，该行一律不参与损失与梯度），但「恒为 0」是错误理由，现已按实测更正为
    「**该行一律被掩码、与它的 logits 取值无关**」。

    与 1a 口径的关系：``robust_eval.perturb_matrix`` 本身对零范数行**显式报错**
    （``l2_normalize_rows`` 的契约，不静默产 NaN）—— 该行为**保持不变**；
    本函数只在训练侧**事后**识别该情形并如实计数，不改动扰动实现。
    """
    if x.dim() != 2:
        raise ValueError(f"zero_norm_mask 期望二维输入，当前形状 = {tuple(x.shape)}")
    with torch.no_grad():
        norms = torch.linalg.norm(x.to(torch.float32), dim=1)
    return (norms > 0.0).to(torch.bool)


def _batch_perturbed(
    feats: np.ndarray,
    idx: Sequence[int],
    *,
    kind: str,
    level: str,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """对训练库行的一个批施加扰动（走 1a 的 :func:`robust_eval.perturb_matrix`）。

    口径：**先切批再扰动** —— 扰动实现内部的行号是**批内相对行号**（棋盘平移的相位、
    ``mask`` 的逐行 ``randperm``、``noise`` 的形状都按批内形状生成），故与「全量一次性
    扰动再切片」**不是**逐位相同的写法。该事实在产物 ``train.train_perturb_note`` 中
    显式登记，避免被误读成同一写法的两种等价表达。

    零范数行（现场实测真实可达）：扰动后整行归零时，1a 的实现会抛 ``ValueError``；
    本函数经 :func:`_perturb_allow_zero` 取回「扰动后、归一化前」的矩阵，把零行保留为
    全零向量 —— 归一化口径与 :func:`entry_table.l2_normalize_rows` 完全一致
    （非零行除以 float64 范数、零行保持全零），随后由 :func:`zero_norm_mask` 把它
    **在损失中掩码**。

    返回
    ----
    Tuple[torch.Tensor, Dict[str, Any]]
        ``(归一化后的 float32 [B, D] 查询矩阵, 取证字典)``；取证字典含
        ``zero_norm_rows``（批内零范数行数）与 1a 的扰动明细。
    """
    sub = np.ascontiguousarray(feats[[int(i) for i in idx]], dtype=np.float32)
    raw, detail = _perturb_allow_zero(sub, str(kind), str(level), variant=0, seed=int(seed))
    norms = np.linalg.norm(raw.astype(np.float64), axis=1)
    zero = np.flatnonzero(norms <= 0.0)
    out = np.zeros_like(raw, dtype=np.float64)
    if norms.size:
        nz = np.setdiff1d(np.arange(raw.shape[0]), zero, assume_unique=False)
        if nz.size:
            out[nz] = raw[nz].astype(np.float64) / norms[nz][:, None]
    out32 = out.astype(np.float32)
    evidence = dict(detail)
    evidence["zero_norm_rows"] = [int(i) for i in zero.tolist()]
    evidence["n_zero_norm"] = int(zero.size)
    evidence["normalized_by"] = (
        "variant_b._batch_perturbed（口径与 entry_table.l2_normalize_rows 逐位一致："
        "float64 求范数、零行保持全零）"
    )
    return torch.from_numpy(np.ascontiguousarray(out32, dtype=np.float32)), evidence


def _manual_perturb_inlined(
    src: np.ndarray,
    kind: str,
    *,
    level_index: int,
    variant: int,
    seed: int,
) -> np.ndarray:
    """**内联手写**的扰动重算（判据的「手算侧」；**不读取** ``robust_eval`` 的任何共享层）。

    存在理由（离朱 R50 F1 的同类教训）：若手算侧与被测实现共用同一份公式 / 常量 / 种子 helper，
    那么把那些共享层破坏掉时两侧会**同步变化**，判据对它恒真（无区分度）。
    故本函数把下面这些**全部内联为字面量**：

    * 型别下标 ``noise = 0`` / ``mask = 1`` / ``nmag = 2``（``PERTURB_TYPES`` 的顺序）；
    * 派生种子公式 ``seed*1000 + 100*type_index + 10*level_index + variant``
      （level：弱 0 / 中 1 / 强 2；由调用方以字面量传入 ``level_index``）；
    * 各档强度：``noise`` 弱档 σ = 0.05、``mask`` 强档 ratio = 0.5、``nmag`` 弱档 ε = 0.10；
    * ``mask`` 的保留维数 ``n_keep = floor(D * (1 - ratio))`` 与逐行 ``randperm`` 的抽样方式；
    * ``nmag`` 的缩放系数 ``0.5`` 与棋盘符号矩阵 ``b[i,j] = eps * (-1)**(i+j+1)``；
    * 逐行 L2 归一化的数值路径（float64 求范数、零行保持全零）。

    参数
    ----
    src : np.ndarray
        ``[N, D]`` 输入特征。
    kind : str
        扰动类型（只用于选型别下标与分支，**不读注册表**）。
    level_index : int
        档位下标（内联字面量：弱 0 / 中 1 / 强 2），直接进派生种子公式。
    variant / seed : int
        与 1a 同义（内联进派生种子公式）。

    返回
    ----
    np.ndarray
        ``float32[N, D]`` 归一化后的扰动结果（与 1a 口径相同）。
    """
    if str(kind) not in ("noise", "mask", "nmag"):
        raise ValueError(f"_manual_perturb_inlined 只覆盖三种扰动，当前 {kind!r}")
    type_index = {"noise": 0, "mask": 1, "nmag": 2}[str(kind)]
    gen_seed = int(seed) * 1000 + 100 * int(type_index) + 10 * int(level_index) + int(variant)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(gen_seed))
    base = torch.from_numpy(np.ascontiguousarray(np.asarray(src, dtype=np.float32)))
    n_rows, dim = int(base.shape[0]), int(base.shape[1])
    if str(kind) == "noise":
        sigma = 0.05  # noise/weak 的 σ（内联字面量）
        noise = torch.randn(base.shape, generator=gen, dtype=torch.float32)
        perturbed = (base + float(sigma) * noise).numpy()
    elif str(kind) == "mask":
        ratio = 0.5  # mask/strong 的遮蔽比例（内联字面量）
        n_keep = int(np.floor(float(dim) * (1.0 - float(ratio))))
        keep = torch.stack(
            [torch.randperm(dim, generator=gen)[:n_keep] for _ in range(n_rows)], dim=0
        )
        mask = torch.zeros(base.shape, dtype=torch.float32)
        mask.scatter_(1, keep, 1.0)
        perturbed = (base * mask).numpy()
    else:
        eps = 0.10  # nmag/weak 的 ε（内联字面量）
        ii = np.arange(n_rows, dtype=np.int64)[:, None]
        jj = np.arange(dim, dtype=np.int64)[None, :]
        sign = np.where(((ii + jj) % 2) == 0, -1.0, 1.0)
        shift = (float(eps) * sign).astype(np.float32)
        perturbed = (base * 0.5 + torch.from_numpy(shift)).numpy()
    arr = np.asarray(perturbed, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1)
    out = np.zeros_like(arr)
    nz = np.flatnonzero(norms > 0.0)
    if nz.size:
        out[nz] = arr[nz] / norms[nz][:, None]
    return out.astype(np.float32)


def perturb_retrieval_evidence(*, dim: int = 8, n_rows: int = 3) -> Dict[str, Any]:
    """**W2 修复的正向证据**：取回路径的两条分支都逐字节对得上「内联手算」。

    规模很小（``dim=8``、``n_rows=3``），只验口径、不做任何统计。

    三条对照
    --------
    ① **成功分支**（``nmag/weak``，无零范数行）：``_perturb_allow_zero`` 的返回必须与
       :func:`_manual_perturb_inlined` 逐字节相同，且 ``retrieval_mode == "direct_1a"``；
    ② **修复分支**（``mask/strong``，构造出真实的零范数行）：返回的**未被替换行**必须与
       手算逐字节相同、被替换行必须**全零**、且手算侧同行为零（即「置零」确实是该行的真值）；
    ③ 修复分支的取证必须带 ``row_independence_verified = True`` 与 ``repaired_rows``。

    返回
    ----
    Dict[str, Any]
        ``success_branch`` / ``repair_branch`` / ``manual_side_isolated`` /
        ``manual_side_note`` / ``all_bitwise_equal`` / ``rule``。
    """
    dim = int(dim)
    n_rows = int(n_rows)
    if dim < 2 or n_rows < 1:
        raise ValueError(f"perturb_retrieval_evidence 期望 dim>=2 且 n_rows>=1，当前 {dim}/{n_rows}")
    rng = np.random.Generator(np.random.PCG64(20261007))
    base = np.ascontiguousarray(
        (rng.standard_normal((n_rows, dim)) * 0.5 + 0.1).astype(np.float32)
    )
    # --- ① 成功分支：无零范数行 ⇒ 直接走 1a --------------------------------
    got_ok, ev_ok = _perturb_allow_zero(base, "nmag", "weak", variant=0, seed=VARIANT_B_SEED)
    want_ok = _manual_perturb_inlined(
        base, "nmag", level_index=0, variant=0, seed=VARIANT_B_SEED
    )
    ok_equal = bool(
        np.ascontiguousarray(got_ok, dtype=np.float32).tobytes()
        == np.ascontiguousarray(want_ok, dtype=np.float32).tobytes()
    )
    # --- ② 修复分支：构造一个**一定**会出现零范数行的输入 --------------------
    # 行 0 只在少数维度上有值 ⇒ `mask/strong`（遮蔽一半维度）下有很高概率整行被抹零；
    # (variant, 非零维下标) 二重扫描保证**现场真实找到**一个触发点（找不到即如实报错）。
    trigger: Optional[Dict[str, Any]] = None
    for variant in range(8):
        for j in range(dim):
            probe = np.zeros((n_rows, dim), dtype=np.float32)
            probe[0, j] = 1.0
            probe[1:] = base[1:]
            try:
                R.perturb_matrix(
                    probe, "mask", "strong", variant=int(variant), seed=int(VARIANT_B_SEED)
                )
            except ValueError as exc:
                if "零范数" not in str(exc):
                    raise
                trigger = {"variant": int(variant), "nonzero_dim": int(j)}
                break
        if trigger is not None:
            break
    if trigger is None:
        raise ValueError(
            "perturb_retrieval_evidence fail-closed：在 (variant, 非零维) 的二重扫描中"
            "未能构造出「扰动后零范数行」的输入；修复分支无法被现场验证"
        )
    probe = np.zeros((n_rows, dim), dtype=np.float32)
    probe[0, int(trigger["nonzero_dim"])] = 1.0
    probe[1:] = base[1:]
    got_rep, ev_rep = _perturb_allow_zero(
        probe, "mask", "strong", variant=int(trigger["variant"]), seed=int(VARIANT_B_SEED)
    )
    want_rep = _manual_perturb_inlined(
        probe,
        "mask",
        level_index=2,
        variant=int(trigger["variant"]),
        seed=int(VARIANT_B_SEED),
    )
    repaired_rows = [int(i) for i in ev_rep.get("repaired_rows", [])]
    keep_rows = [i for i in range(n_rows) if i not in set(repaired_rows)]
    if not repaired_rows:
        raise ValueError(
            "perturb_retrieval_evidence fail-closed：修复分支没有报出被替换的行集合"
        )
    keep_equal = bool(
        not keep_rows
        or np.ascontiguousarray(got_rep[keep_rows], dtype=np.float32).tobytes()
        == np.ascontiguousarray(want_rep[keep_rows], dtype=np.float32).tobytes()
    )
    zero_ok = bool(np.all(got_rep[repaired_rows] == 0.0))
    manual_zero_ok = bool(np.all(want_rep[repaired_rows] == 0.0))
    rep_ev_ok = bool(
        ev_rep.get("retrieval_mode") == "repaired_via_1a"
        and bool(ev_rep.get("row_independence_verified", False))
    )
    return {
        "success_branch": {
            "kind": "nmag",
            "level": "weak",
            "retrieval_mode": ev_ok.get("retrieval_mode"),
            "n_rows": n_rows,
            "dim": dim,
            "bitwise_equal_vs_manual": bool(ok_equal),
            "manual_side": "variant_b._manual_perturb_inlined（内联字面量，无共享层）",
        },
        "repair_branch": {
            "kind": "mask",
            "level": "strong",
            "trigger": dict(trigger),
            "retrieval_mode": ev_rep.get("retrieval_mode"),
            "n_rows": n_rows,
            "dim": dim,
            "repaired_rows": repaired_rows,
            "kept_rows": keep_rows,
            "bitwise_equal_vs_manual_on_kept_rows": bool(keep_equal),
            "repaired_rows_are_zero": bool(zero_ok),
            "manual_side_repaired_rows_are_zero": bool(manual_zero_ok),
            "row_independence_verified": bool(ev_rep.get("row_independence_verified", False)),
            "retrieval_evidence_ok": bool(rep_ev_ok),
            "manual_side": "variant_b._manual_perturb_inlined（内联字面量，无共享层）",
        },
        "manual_side_isolated": True,
        "manual_side_note": (
            "手算侧**不引用** `robust_eval` 的型别下标 / 派生种子 helper / 共享缩放常量 / "
            "符号矩阵 helper / 强度网格：型别与档位下标、派生种子公式、`mask` 比例 0.5、"
            "`nmag` 缩放 0.5 与棋盘符号、float64 归一化路径**全部内联为字面量**，"
            "故破坏那些共享层时本判据会**真的失败**（见 `perturb_retrieval_reverse_check`）。"
        ),
        "all_bitwise_equal": bool(ok_equal and keep_equal and zero_ok and manual_zero_ok and rep_ev_ok),
        "rule": RETRIEVAL_RULE,
    }


def perturb_retrieval_reverse_check() -> Dict[str, Any]:
    """**W2 反向验证**：把 1a 的共享层破坏掉，正向证据必须**被检出为失败**；恢复后回到通过。

    被破坏的共享层（逐个 monkeypatch，跑完即恢复；恢复放在 ``finally`` 里）：

    ① ``robust_eval.derived_seed``（派生种子 helper）；
    ② ``robust_eval.NMAG_SHARED_SCALE``（共享缩放常量）；
    ③ ``robust_eval.nmag_shift_matrix``（符号矩阵 helper）；
    ④ ``robust_eval.PERTURB_GRID["mask"]["strong"]``（强度网格）。

    若某一项破坏**检不出**（正向证据仍然 ``all_bitwise_equal = True``），说明该判据对这一层
    没有区分度，本函数如实报出 ``detected = False``（不掩盖）。

    返回
    ----
    Dict[str, Any]
        ``baseline_passed`` / ``cases`` / ``all_detected`` / ``restored_passed`` / ``passed`` /
        ``rule``。
    """
    baseline = perturb_retrieval_evidence()
    cases: List[Dict[str, Any]] = []

    def _run(case: str, description: str) -> None:
        try:
            ev = perturb_retrieval_evidence()
            detected = bool(not ev["all_bitwise_equal"])
            error = ""
        except Exception as exc:  # noqa: BLE001 —— 破坏后被 fail-closed 拒绝同样算「检出」
            detected = True
            error = f"{type(exc).__name__}: {exc}"
        cases.append(
            {
                "case": str(case),
                "description": str(description),
                "detected": bool(detected),
                "raised_or_mismatched": str(error),
            }
        )

    orig_derived = R.derived_seed
    orig_scale = R.NMAG_SHARED_SCALE
    orig_shift = R.nmag_shift_matrix
    orig_grid = dict(R.PERTURB_GRID["mask"])
    try:
        R.derived_seed = (  # type: ignore[assignment]
            lambda kind, level, variant=0, seed=R.ROBUST_SEED: int(
                orig_derived(kind, level, variant, seed)
            )
            + 1
        )
        _run("monkeypatch derived_seed(+1)", "派生种子 helper 整体 +1")
        R.derived_seed = orig_derived  # type: ignore[assignment]

        R.NMAG_SHARED_SCALE = 0.25  # type: ignore[assignment]
        _run("monkeypatch NMAG_SHARED_SCALE=0.25", "共享缩放常量改为 0.25")
        R.NMAG_SHARED_SCALE = orig_scale  # type: ignore[assignment]

        R.nmag_shift_matrix = (  # type: ignore[assignment]
            lambda n_rows, dim, eps: np.zeros((int(n_rows), int(dim)), dtype=np.float32)
        )
        _run("monkeypatch nmag_shift_matrix->zeros", "符号矩阵 helper 恒返回全零")
        R.nmag_shift_matrix = orig_shift  # type: ignore[assignment]

        R.PERTURB_GRID["mask"]["strong"] = 0.25
        _run("monkeypatch PERTURB_GRID[mask][strong]=0.25", "强度网格被改（遮蔽比例 0.5→0.25）")
        R.PERTURB_GRID["mask"] = orig_grid
    finally:
        R.derived_seed = orig_derived  # type: ignore[assignment]
        R.NMAG_SHARED_SCALE = orig_scale  # type: ignore[assignment]
        R.nmag_shift_matrix = orig_shift  # type: ignore[assignment]
        R.PERTURB_GRID["mask"] = orig_grid
    restored = perturb_retrieval_evidence()
    return {
        "baseline_passed": bool(baseline["all_bitwise_equal"]),
        "cases": cases,
        "all_detected": bool(cases and all(c["detected"] for c in cases)),
        "restored_passed": bool(restored["all_bitwise_equal"]),
        "passed": bool(
            baseline["all_bitwise_equal"]
            and cases
            and all(c["detected"] for c in cases)
            and restored["all_bitwise_equal"]
        ),
        "rule": (
            "对 `robust_eval` 的四类共享层逐个 monkeypatch（派生种子 / 共享缩放常量 / "
            "符号矩阵 helper / 强度网格），每一项都必须让 `perturb_retrieval_evidence` 的 "
            "`all_bitwise_equal` 变 False（或被 fail-closed 抛出）；恢复后必须回到 True。"
            "任一破坏检不出即 `passed = False`（如实报出，不掩盖）。"
        ),
    }


def train_transform(
    model: VariantBModel,
    table_feats: np.ndarray,
    library_index: Sequence[int],
    cfg: TrainConfig,
    *,
    log: Any = None,
) -> Dict[str, Any]:
    """**扰动自监督训练循环**（主方案）：冻结库全表作输出层、交叉熵、Adam。

    口径
    ----
    * 训练查询 = **库行**（1999 条，严格排除 666 查询行）；标签 = 该行在**全量键表**中的行下标；
    * ``train_mode="perturb"`` 时每一步轮转 9 个扰动格（``<= 0.2`` 步按固定顺序取格）；
      ``"clean"`` = A 组对照档（不扰动，其余完全相同）；
    * 打乱用**局部** ``torch.Generator``（不消耗全局 RNG）；扰动种子走 1a 的 ``derived_seed``；
    * 优化器 = Adam，**只优化** :class:`DToDTransform` 的参数；
    * 返回字典**不含挂钟时间**（确定性纪律），耗时只进日志。

    参数
    ----
    model : VariantBModel
        待训练模型（就地更新 ``transform``）。
    table_feats : np.ndarray
        全量键表特征 ``[2665, D]`` float32（**只读**；训练只用其库行子集）。
    library_index : Sequence[int]
        训练用库行下标。
    cfg : TrainConfig
        训练配置。
    log : Any
        可调用日志。

    返回
    ----
    Dict[str, Any]
        ``history``（逐 epoch 损失 / **批内滚动平均** ``train_batch_running_top1``）、
        ``n_steps``、``train_batch_running_top1``（+ 其口径文本 ``train_batch_running_top1_rule``）、
        ``train_perturb_note``、``zero_norm_rows``、``skipped_steps``、``rng_note``。

    [!] **观测量口径（皋陶审查 warning 3 / info 6）**：``train_batch_running_top1`` 是
    **批内滚动平均**（每步只覆盖一个轮转扰动格、每批 256 行、用该步**更新前**的 logits），
    **不是**判定量。本模块的判定量是**逐格 R@1**（在训练行 1999 上跑一次完整 10 格网格），
    见 :func:`train_harmlessness` 的 ``per_cell[].post_top1`` 与 ``train_side_cells``。
    该字段旧名为 ``train_top1_self``（与逐格量同名易串），本轮已改名，见
    :data:`TRAIN_TOP1_SELF_RULE`。

    边界处置（**零范数批内行**）
    ---------------------------
    训练侧对**每一批单独**施加扰动（行号基准是批内），因此「批内某行被扰动后整行归零」
    是一个真实可达的事件：现场实测 ``lexical-88`` 的 ``mask/strong``（遮蔽 50% = 44/88 维）
    在某一批的第 118 行（表行 23）上发生，该行本来只有 9 个非零维、全部落在被遮蔽的一半里。
    处置口径（**不静默**）：该行原样保留为全零向量，但**在损失中对它做掩码**（掩码取自
    **变换前**的扰动后输入 ``q`` ⇒ 该行一律不参与交叉熵与梯度，**与它的 logits 取值无关**；
    [!] 离朱 R59 F4：改前此处写「它的 logits 恒为 0」，该说法只在 `T = I` 时成立，
    训练后 `T(0) = b != 0` —— 功能无缺陷，措辞已更正）。``zero_norm_rows`` /
    ``skipped_steps`` 逐批计数并**进入产物**。若某批掩码后一行不剩，则**显式报错**
    （``ValueError``），不静默跳过整批。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    feats = np.asarray(table_feats, dtype=np.float32)
    lib = [int(i) for i in library_index]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(SHUFFLE_SEED))
    params = model.trainable_parameters()
    if not params:
        raise ValueError("变体 B 模型没有可训参数；拒绝执行空训练")
    optimizer = torch.optim.Adam(
        params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    labels_all = torch.arange(model.n_keys, dtype=torch.long)
    bs = int(cfg.batch_size)
    history: List[Dict[str, Any]] = []
    n_steps = 0
    zero_rows_total = 0
    skipped_steps = 0
    zero_rows_detail: List[Dict[str, Any]] = []
    # 范数诊断（**退化解通道的现场判据**）：`raw` 口径下「放大 ‖T(x)‖」可单调压低交叉熵，
    # 最省力的下降方向可能是范数膨胀而不是学方向。逐 epoch 记录范数与 logits 量级，
    # 该观测**只进产物、不参与任何判定**（判定口径见 TRAIN_SCORER_RULE）。
    probe_rows = [int(i) for i in lib[: int(min(256, len(lib)))]]
    probe_x = torch.from_numpy(
        np.ascontiguousarray(feats[probe_rows], dtype=np.float32)
    )
    norm_diagnostics: List[Dict[str, Any]] = []
    model.transform.train(True)
    for epoch in range(1, int(cfg.epochs) + 1):
        order = torch.randperm(len(lib), generator=gen).tolist()
        epoch_loss = 0.0
        epoch_hits = 0
        epoch_n = 0
        epoch_logit_abs = 0.0
        for b0 in range(0, len(order), bs):
            batch_pos = order[b0 : b0 + bs]
            batch_idx = [lib[p] for p in batch_pos]
            labels = labels_all[torch.tensor(batch_idx, dtype=torch.long)]
            if str(cfg.train_mode) == "clean":
                kind, level = "clean", "clean"
                q = torch.from_numpy(
                    np.ascontiguousarray(feats[batch_idx], dtype=np.float32)
                )
            else:
                tile = TRAIN_PERTURB_CELLS[n_steps % len(TRAIN_PERTURB_CELLS)]
                kind, level = str(tile[0]), str(tile[1])
                q, _p_detail = _batch_perturbed(
                    feats, batch_idx, kind=kind, level=level, seed=int(cfg.seed)
                )
            mask = zero_norm_mask(q)
            if not bool(mask.any()):
                raise ValueError(
                    "整批查询被扰动成零范数（无可训练信号）；拒绝静默跳过整批"
                )
            n_zero = int((~mask).sum().item())
            if n_zero:
                zero_rows_total += n_zero
                if len(zero_rows_detail) < 20:
                    zero_rows_detail.append(
                        {
                            "epoch": int(epoch),
                            "step": int(n_steps + 1),
                            "perturb": f"{kind}/{level}",
                            "n_zero_norm_rows": int(n_zero),
                            "table_rows": [
                                int(batch_idx[int(i)])
                                for i in torch.nonzero(~mask).flatten().tolist()
                            ],
                        }
                    )
            # 打分口径由 cfg.train_scorer 决定（`normalized` = 与评测同一口径，默认）；
            # 零范数行由 `sel_idx` 掩掉（掩码取自变换前的 q），其 logits 取值不参与损失
            # （离朱 R59 F4：T=I 时该行 logits 为 0，训练后 T(0)=b 一般非零 —— 与掩码无关）。
            logits = model.train_score(q, cfg.train_scorer)
            sel_idx = torch.nonzero(mask).flatten()
            loss = loss_fn(logits[sel_idx], labels[sel_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n_steps += 1
            if n_zero:
                skipped_steps += 1
            with torch.no_grad():
                hits = int((torch.argmax(logits[sel_idx], dim=1) == labels[sel_idx]).sum().item())
                epoch_logit_abs += float(logits[sel_idx].abs().mean().item()) * int(sel_idx.numel())
            epoch_loss += float(loss.item()) * int(sel_idx.numel())
            epoch_hits += hits
            epoch_n += int(sel_idx.numel())
            if int(cfg.log_every) > 0 and n_steps % int(cfg.log_every) == 0:
                _log(
                    f"[variantb train] epoch {epoch}/{cfg.epochs} step {n_steps} "
                    f"扰动 {kind}/{level} loss={float(loss.item()):.6f} "
                    f"批内命中 {hits}/{int(sel_idx.numel())}"
                    f"{'（掩码零范数行 ' + str(n_zero) + ' 条）' if n_zero else ''}"
                )
        train_top1 = float(epoch_hits / max(1, epoch_n))
        with torch.no_grad():
            y_probe = model.transform(probe_x)
            w = model.transform.weight.detach()
            b = model.transform.bias.detach()
            norm_diagnostics.append(
                {
                    "epoch": int(epoch),
                    "mean_logit_abs": float(epoch_logit_abs / max(1, epoch_n)),
                    "mean_row_norm_x": float(torch.linalg.norm(probe_x, dim=1).mean()),
                    "mean_row_norm_Tx": float(torch.linalg.norm(y_probe, dim=1).mean()),
                    "weight_fro": float(torch.linalg.norm(w)),
                    "weight_minus_identity_fro": float(torch.linalg.norm(w - torch.eye(int(model.dim)))),
                    "bias_norm": float(torch.linalg.norm(b)),
                }
            )
        history.append(
            {
                "epoch": int(epoch),
                "mean_loss": float(epoch_loss / max(1, epoch_n)),
                # 批内滚动平均（旧名 train_top1_self；见 TRAIN_TOP1_SELF_RULE）——
                # **仅诊断**，不是判定量（判定量是训练行 1999 上的逐格 R@1）。
                "train_batch_running_top1": float(train_top1),
                "n_samples": int(epoch_n),
                "steps": int(n_steps),
            }
        )
        _log(
            f"[variantb train] epoch {epoch}/{cfg.epochs} 完成："
            f"mean_loss={float(epoch_loss / max(1, epoch_n)):.6f} "
            f"批内滚动平均（仅诊断）={train_top1:.6f}（{epoch_hits}/{epoch_n}）"
            f"；||T(x)|| 均值={float(torch.linalg.norm(y_probe, dim=1).mean()):.6f} "
            f"||W-I||_F={float(torch.linalg.norm(model.transform.weight.detach() - torch.eye(int(model.dim)))):.6f}"
        )
    model.transform.train(False)
    nd0 = norm_diagnostics[0] if norm_diagnostics else {}
    nd1 = norm_diagnostics[-1] if norm_diagnostics else {}
    return {
        "history": history,
        "n_steps": int(n_steps),
        "train_batch_running_top1": (
            float(history[-1]["train_batch_running_top1"]) if history else 0.0
        ),
        "train_batch_running_top1_rule": TRAIN_TOP1_SELF_RULE,
        "train_scorer": str(cfg.train_scorer),
        "train_scorer_rule": TRAIN_SCORER_RULE,
        "n_train_rows": int(len(lib)),
        "norm_diagnostics": norm_diagnostics,
        "norm_diagnostics_summary": {
            "probe_rows": [int(i) for i in probe_rows],
            "probe_note": (
                "探针 = 库行中**前 256 行**的**干净**特征（固定、无随机、未参与损失以外的任何判定）；"
                "逐 epoch 记录 `‖T(x)‖` 与参数量级，用于现场判定「范数膨胀」这条退化解通道"
            ),
            "mean_row_norm_Tx_first": float(nd0.get("mean_row_norm_Tx", 0.0)),
            "mean_row_norm_Tx_last": float(nd1.get("mean_row_norm_Tx", 0.0)),
            "norm_ratio_last_over_first": float(
                float(nd1.get("mean_row_norm_Tx", 0.0)) / max(1e-30, float(nd0.get("mean_row_norm_Tx", 1.0)))
            ),
            "weight_minus_identity_fro_last": float(nd1.get("weight_minus_identity_fro", 0.0)),
            "mean_logit_abs_first": float(nd0.get("mean_logit_abs", 0.0)),
            "mean_logit_abs_last": float(nd1.get("mean_logit_abs", 0.0)),
            "reading": (
                "`raw` 口径下若 `mean_row_norm_Tx` 显著上升而 `train_batch_running_top1` 不升，"
                "即**现场证实**「范数膨胀」这条最省力下降方向被用掉了；"
                "`normalized` 口径下该量按构造恒等于 1 的量级（归一化后），"
                "上述通道被结构性关闭。**本诊断不参与任何判定**，只作机制取证。"
            ),
        },
        "zero_norm_rows": {
            "total": int(zero_rows_total),
            "n_steps_with_zero_row": int(skipped_steps),
            "detail_head": zero_rows_detail,
            "rule": (
                "扰动后整行归零的行**原样保留为全零向量**但**在损失中掩码**"
                "（掩码取自**变换前**的扰动后输入 q ⇒ 该行一律不参与交叉熵与梯度，"
                "与其 logits 取值无关；离朱 R59 F4：改前写「logits 恒为 0」只在 T=I 时成立）；"
                "逐批计数进产物；整批全零则显式报错"
            ),
            "skipped_steps": int(skipped_steps),
        },
        "train_perturb_note": (
            "扰动在**批内**施加（1a 的 `perturb_matrix` 按批内行号生成噪声 / 掩码 / 棋盘平移），"
            "故批大小会改变每一行的扰动实现；评测侧则是「全部查询一次扰动」。两者使用同一实现与"
            "同一派生种子公式，但行号基准不同 —— 不得读成同一写法的两种等价表达"
        ),
        "rng_note": (
            "打乱用局部 torch.Generator(seed=SHUFFLE_SEED)；扰动种子走 1a 的 derived_seed；"
            "全局 RNG 未被消耗"
        ),
    }


def update_gate(
    before: Sequence[Dict[str, Any]], after: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """**可训参数更新量门禁**（沿用既有口径：只判「是否有更新」，不设阈值）。

    参数
    ----
    before / after : Sequence[Dict[str, Any]]
        :meth:`VariantBModel.parameter_snapshot` 的前后快照。

    返回
    ----
    Dict[str, Any]
        ``per_param`` / ``n_updated`` / ``n_total`` / ``n_never_updated`` /
        ``all_unchanged`` / ``passed``（= ``n_updated > 0``）/ ``rule``。
    """
    bmap = {str(x["name"]): x for x in before}
    amap = {str(x["name"]): x for x in after}
    per: List[Dict[str, Any]] = []
    for name, b in bmap.items():
        a = amap.get(name)
        changed = bool(a is not None and a["state_bytes_sha256"] != b["state_bytes_sha256"])
        per.append(
            {
                "name": name,
                "shape": list(b["shape"]),
                "before_sha256": str(b["state_bytes_sha256"]),
                "after_sha256": (str(a["state_bytes_sha256"]) if a else None),
                "changed": bool(changed),
            }
        )
    n_changed = int(sum(1 for x in per if x["changed"]))
    return {
        "per_param": per,
        "n_updated": int(n_changed),
        "n_total": int(len(per)),
        "n_never_updated": int(len(per) - n_changed),
        "all_unchanged": bool(n_changed == 0),
        "passed": bool(n_changed > 0),
        "rule": UPDATE_GATE_RULE,
    }


# ---------------------------------------------------------------------------
# 5. 恒等门禁（硬门禁）与单条端到端演练（drill）
# ---------------------------------------------------------------------------


def identity_gate(
    data: VariantBData,
    *,
    seed: int = VARIANT_B_SEED,
    topk: int = R.TOPK,
    batch_size: int = R.BATCH_SIZE,
) -> Dict[str, Any]:
    """**硬门禁**：训练前 ``T = I`` 时 clean + 9 个扰动档与余弦最近邻逐条一致。

    现场同时给出六条独立证据：
    ① 逐格 ``n_mismatch``（top-1 行下标逐条比对）；
    ② :func:`sample_bitwise_equality` 的范数逐位相等比例；
    ③ :func:`explicit_path_evidence` 的「快路径 vs 显式 matmul」偏差（**训练态与评测态各一次**）；
    ④ :func:`gradient_flow_evidence` 的「恒等态仍在计算图上且梯度非零」；
    ⑤ :func:`score_path_evidence` 的**自洽判据** —— ``score(q)`` 逐位等于
       ``normalize_query(transform(q)) @ keys.T``（专门拦「评测侧漏掉变换层」）；
    ⑥ :func:`train_scorer_evidence` 的**训练侧打分口径自洽判据** —— ``train_score`` 的两个
       分支分别逐位等于其显式实现（训练前提校准轮新增）。

    [!] 逐条统计量来自 ``cell_grid`` 的 :data:`TOP1_DISCREPANCY_KEY` 字段（W1 修复后改名），
    本函数把它复制成门禁自己的 ``all_equal`` 并显式标注 ``computed_on = fresh_model(T=I)``
    —— 门禁的 `all_equal` 与 ``cell_grid`` 的 ``all_cells_equal`` 由此**不再同名**。
    """
    model = build_model(data.table)
    grid = cell_grid(
        model, data, seed=int(seed), topk=int(topk), batch_size=int(batch_size)
    )
    with torch.no_grad():
        q = data.query_features()
        bitwise = sample_bitwise_equality(model, q)
        explicit_eval = explicit_path_evidence(model, q, training_mode=False)
    explicit_train = explicit_path_evidence(model, q, training_mode=True)
    grad_flow = gradient_flow_evidence(model, q)
    path_ev = score_path_evidence(model, q)
    scorer_ev = train_scorer_evidence(model, q)
    gate = dict(grid[TOP1_DISCREPANCY_KEY])
    # 门禁只认「训练前 T=I」这一态：显式标注两个状态字段，避免与 `cell_grid` 里的同名统计量混淆
    gate["all_equal"] = bool(gate["all_cells_equal"])
    gate["computed_on"] = "fresh_model(T=I)" if model.transform.is_identity() else "non_identity_model"
    gate["discrepancy_key"] = TOP1_DISCREPANCY_KEY
    gate["bitwise_norm_equality"] = bitwise
    gate["explicit_path"] = explicit_eval
    gate["explicit_path_train_mode"] = explicit_train
    gate["gradient_flow"] = grad_flow
    gate["score_path"] = path_ev
    gate["train_scorer"] = scorer_ev
    gate["transform_is_identity"] = bool(model.transform.is_identity())
    gate["fast_path_rule"] = IDENTITY_FASTPATH_RULE
    gate["passed"] = bool(
        gate["all_equal"]
        and gate["transform_is_identity"]
        and float(bitwise["bitwise_equal_frac"]) == 1.0
        and bool(explicit_eval["bitwise_equal"])
        and bool(explicit_train["bitwise_equal"])
        and bool(grad_flow["all_params_have_grad"])
        and bool(path_ev["bitwise_equal"])
        and bool(scorer_ev["all_bitwise_equal"])
    )
    return gate


def drill(
    profile: str,
    *,
    cfg: Optional[TrainConfig] = None,
    product_dir: str = "",
    log: Any = None,
) -> Dict[str, Any]:
    """**单条端到端演练**：一个档、1 个 epoch，含梯度非零硬门禁与恒等门禁。

    演练在**全量键表**上跑（与全量运行同一份数据、同一套代码路径），只把轮数降到 1，
    并额外检查：① 每一步都有真实梯度（``.grad`` 非 None 且平方和 > 0）；
    ② 可训参数确有更新；③ 恒等门禁通过；④ **取回路径的正向证据 + 反向验证**（W2 修复项，
    :func:`perturb_retrieval_evidence` / :func:`perturb_retrieval_reverse_check`）成立。

    返回
    ----
    Dict[str, Any]
        ``profile`` / ``passed`` / ``identity_gate`` / ``gradient_check`` /
        ``update_gate`` / ``loss_head`` / ``post_train_cells`` / ``perturb_retrieval`` /
        ``data`` / ``cfg``。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    cfg = cfg or TrainConfig(epochs=1)
    data = load_profile_data(str(profile), product_dir=str(product_dir), log=_log)
    _log(f"[variantb drill] 档 {profile}：恒等门禁（clean + 9 扰动格 vs 余弦最近邻）...")
    gate = identity_gate(data, seed=int(cfg.seed), batch_size=int(cfg.batch_size))
    for cell, n_bad in sorted(gate["per_cell_mismatch"].items()):
        _log(f"[variantb drill]   门禁 {cell}: 不一致 {n_bad} 条")
    _log(
        f"[variantb drill] 恒等门禁：{gate['n_cells_all_equal']}/{gate['n_cells']} 格逐条一致，"
        f"范数逐位相等比例={gate['bitwise_norm_equality']['bitwise_equal_frac']}，"
        f"显式路径逐位相同={gate['explicit_path']['bitwise_equal']}，通过={gate['passed']}"
    )
    model = build_model(data.table)
    before = model.parameter_snapshot()
    _log(
        "[variantb drill] 可训参数（现场枚举）= "
        f"{[x['name'] for x in before]}；元素数 = {[x['n_element'] for x in before]}"
    )
    # 梯度非零硬门禁：手动跑 3 个与 train_transform 同构的步骤并记录梯度范数
    grad_norms: List[float] = []
    loss_head: List[float] = []
    zero_norm_seen: List[int] = []
    feats = data.table.keys.numpy()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(SHUFFLE_SEED))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.trainable_parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    labels_all = torch.arange(model.n_keys, dtype=torch.long)
    lib = list(data.library_index)
    model.transform.train(True)
    for step in range(3):
        order = torch.randperm(len(lib), generator=gen).tolist()
        # **每一步换一批**：若不推进 generator，三步会取到同一段排列前缀（同 256 行），
        # 演练覆盖面被无谓收窄。
        b0 = int(step) * int(cfg.batch_size)
        batch_idx = [lib[p] for p in order[b0 : b0 + int(cfg.batch_size)]]
        tile = TRAIN_PERTURB_CELLS[step % len(TRAIN_PERTURB_CELLS)]
        q, _p_detail = _batch_perturbed(
            feats, batch_idx, kind=str(tile[0]), level=str(tile[1]), seed=int(cfg.seed)
        )
        labels = labels_all[torch.tensor(batch_idx, dtype=torch.long)]
        logits = model.train_score(q, cfg.train_scorer)
        keep = zero_norm_mask(q)
        if not bool(keep.any()):
            raise ValueError("演练批被扰动成零范数（无可训练信号）；拒绝静默跳过")
        n_zero = int((~keep).sum().item())
        zero_norm_seen.append(n_zero)
        sel = torch.nonzero(keep).flatten()
        loss = loss_fn(logits[sel], labels[sel])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        total = 0.0
        for p in model.trainable_parameters():
            if p.grad is None:
                raise AssertionError(
                    "存在可训参数的 .grad 为 None（不在计算图上）；拒绝继续演练"
                )
            total += float(p.grad.detach().pow(2).sum().item())
        grad_norms.append(float(np.sqrt(total)))
        loss_head.append(float(loss.item()))
        optimizer.step()
        _log(
            f"[variantb drill]   演练步 {step + 1}: 扰动 {tile[0]}/{tile[1]} "
            f"loss={loss_head[-1]:.6f} 梯度范数={grad_norms[-1]:.6e} "
            f"（掩码零范数行 {n_zero} 条）"
        )
    model.transform.train(False)
    after = model.parameter_snapshot()
    up = update_gate(before, after)
    zero_grad = [float(x) for x in grad_norms if not (x > 0.0)]
    _log(f"[variantb drill] 梯度非零硬门禁：梯度范数 = {grad_norms}；零梯度步 = {zero_grad}")
    _log(
        f"[variantb drill] 可训参数更新量门禁：更新 {up['n_updated']}/{up['n_total']} 个参数，"
        f"从未更新 = {up['n_never_updated']}，通过 = {up['passed']}"
    )
    with torch.no_grad():
        post = cell_grid(model, data, seed=int(cfg.seed), batch_size=int(cfg.batch_size))
    # W2 修复项：取回路径的正向证据 + 反向验证（共享层破坏必须被检出）
    retrieval_forward = perturb_retrieval_evidence()
    retrieval_reverse = perturb_retrieval_reverse_check()
    _log(
        "[variantb drill] 取回路径：正向逐位一致 = "
        f"{retrieval_forward['all_bitwise_equal']}；反向验证（共享层破坏须被检出）= "
        f"{retrieval_reverse['passed']}（逐项 "
        f"{[(c['case'], c['detected']) for c in retrieval_reverse['cases']]}）"
    )
    ok = bool(
        gate["passed"]
        and not zero_grad
        and up["passed"]
        and retrieval_forward["all_bitwise_equal"]
        and retrieval_reverse["passed"]
    )
    _log(f"[variantb drill] 演练结论：{'PASS' if ok else 'FAIL'}")
    return {
        "profile": str(profile),
        "passed": bool(ok),
        "identity_gate": gate,
        "gradient_check": {
            "grad_norms": [float(x) for x in grad_norms],
            "zero_grad_steps": zero_grad,
            "zero_norm_rows_masked": [int(x) for x in zero_norm_seen],
            "all_positive": bool(not zero_grad),
            "rule": (
                "每一步都必须存在真实梯度（平方和开方 > 0），且 .grad 不得为 None；"
                "批内被扰动成零范数的行**在损失中掩码**并计数（不静默丢弃、不静默跳过整批）"
            ),
        },
        "update_gate": up,
        "loss_head": [float(x) for x in loss_head],
        "post_train_cells": post,
        "perturb_retrieval": {
            "forward": retrieval_forward,
            "reverse": retrieval_reverse,
        },
        "data": data.as_dict(),
        "cfg": cfg.as_dict(),
    }


# ---------------------------------------------------------------------------
# 6. 全量训练 + 逐格对照 + A 组对照档 + 主判据
# ---------------------------------------------------------------------------


def identity_anchor_grid(
    model: VariantBModel,
    data: VariantBData,
    *,
    seed: int = VARIANT_B_SEED,
    topk: int = R.TOPK,
    batch_size: int = R.BATCH_SIZE,
) -> Dict[str, Any]:
    """**恒等参照锚点**：未训练（``T = I``）在**训练行 1999** 上、clean + 9 格的 top-1。

    这是「训练必须至少不劣于此」的**硬性下限**（口径见 :data:`TRAIN_ANCHOR_RULE`）。
    本函数**不改动**传入模型，只在调用前硬断言它确实是恒等态（否则锚点无意义）。

    参数
    ----
    model : VariantBModel
        **必须**是恒等态（训练前的新模型）。
    data : VariantBData
        冻结数据视图（用其 ``library_index`` = 训练行 1999）。
    seed / topk / batch_size :
        与 :func:`cell_grid` 同义。

    返回
    ----
    Dict[str, Any]
        ``rule`` / ``cells``（逐格，含 KNN 对照）/ ``top1_self`` / ``recall_at_5`` /
        ``knn_top1_self`` / ``granularity`` / ``identity_equivalence``。
    """
    if not bool(model.transform.is_identity()):
        raise ValueError(
            "恒等锚点要求模型处于恒等态（T = I）；当前 T 已离开恒等 —— "
            "锚点一旦被训练后的模型污染就不再是下限，拒绝出数"
        )
    grid = cell_grid(
        model, data, seed=int(seed), topk=int(topk), batch_size=int(batch_size),
        rows=data.library_index,
    )
    top1 = {str(c["cell"]): float(c["variant_b"]["recall_at_1"]) for c in grid["cells"]}
    top5 = {str(c["cell"]): float(c["variant_b"]["recall_at_5"]) for c in grid["cells"]}
    knn_top1 = {str(c["cell"]): float(c["knn"]["recall_at_1"]) for c in grid["cells"]}
    disp = dict(grid[TOP1_DISCREPANCY_KEY])
    return {
        "rule": TRAIN_ANCHOR_RULE,
        "rows": "library_rows（训练行 1999；键表全量 2665）",
        "row_set_note": ANCHOR_ROW_SET_NOTE,
        "observable": (
            "**逐格 top-1 率 R@1**（在 1999 训练行上跑一次完整 10 格网格）；"
            "**不是**训练循环内的批内滚动平均（见 `train_batch_running_top1_rule`）"
        ),
        "n_rows": int(grid["n_rows"]),
        "n_cells": int(len(grid["cells"])),
        "top1_self": top1,
        "recall_at_5": top5,
        "knn_top1_self": knn_top1,
        "granularity": TRAIN_GRANULARITY_NOTE,
        "identity_equivalence": {
            "model_is_identity": bool(disp.get("model_is_identity")),
            "n_cells_all_equal_vs_knn": int(disp.get("n_cells_all_equal", -1)),
            "n_cells": int(disp.get("n_cells", -1)),
            "all_cells_equal": bool(disp.get("all_cells_equal", False)),
            "note": (
                "`T = I` ⇒ 变体 B 的打分与余弦最近邻**逐位相同**，故锚点同时也等于该行集合上的 "
                "KNN 自命中；本字段就是这条等价关系的现场条数证据"
            ),
        },
        "cells": grid["cells"],
        "gap_weak_minus_strong": grid["gap_weak_minus_strong"],
    }


def train_harmlessness(anchor: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
    """**「训练无害」下限判据**：训练后在训练行上的 top-1 **逐格不低于**恒等锚点。

    参数
    ----
    anchor : Dict[str, Any]
        :func:`identity_anchor_grid` 的结果（含 ``top1_self``）。
    post : Dict[str, Any]
        :func:`cell_grid`（``rows = data.library_index``）的训练后结果。

    返回
    ----
    Dict[str, Any]
        ``per_cell``（逐格 post / anchor / margin，含 ``in_criterion`` 标记）/ ``n_cells`` /
        ``n_not_worse`` / ``n_strictly_worse`` / ``all_not_worse`` / ``min_margin`` /
        ``mean_margin`` / ``excluded_cells`` / ``per_cell_including_clean`` /
        ``aggregate_reading`` / ``granularity`` / ``rule``。

    [!] 判据**排除** ``clean`` 格（见 :data:`TRAIN_HARMLESSNESS_EXCLUDED_CELLS`）：
    该格是逐位自匹配、锚点结构性 1.000000，任何非恒等变换只能变差。被排除的格仍逐格报出。
    另给 ``aggregate_reading``（**不参与判定**）：把「训练侧自命中」按**整段平均**读时，
    9 个扰动格的平均 post 与平均锚点之差 —— 两种读法在本轮可能给出不同倾向，
    必须**同时**登记，不得只报对自己有利的那一种。
    """
    want = {str(k): float(v) for k, v in dict(anchor.get("top1_self", {})).items()}
    excluded = set(str(x) for x in TRAIN_HARMLESSNESS_EXCLUDED_CELLS)
    rows: List[Dict[str, Any]] = []
    for c in post.get("cells", []):
        cell = str(c["cell"])
        if cell not in want:
            rows.append(
                {
                    "cell": cell,
                    "applicable": False,
                    "note": "锚点表里没有该格（如实登记，不静默跳过）",
                }
            )
            continue
        got = float(c["variant_b"]["recall_at_1"])
        ref = float(want[cell])
        rows.append(
            {
                "cell": cell,
                "kind": str(c["kind"]),
                "level": str(c["level"]),
                "post_top1": float(got),
                "anchor_top1": float(ref),
                "margin": float(got - ref),
                "post_hit": int(c["variant_b"]["hit_at_1"]),
                "anchor_hit": int(round(ref * int(c["variant_b"]["n"]))),
                "n": int(c["variant_b"]["n"]),
                "not_worse": bool(got >= ref),
                "strictly_better": bool(got > ref),
                "in_criterion": bool(cell not in excluded),
                "applicable": True,
            }
        )
    usable = [r for r in rows if r.get("applicable")]
    judged = [r for r in usable if r["in_criterion"]]
    margins = [float(r["margin"]) for r in judged]
    worse = [r for r in judged if not r["not_worse"]]
    agg_post = (
        float(sum(float(r["post_top1"]) for r in judged) / len(judged)) if judged else 0.0
    )
    agg_anchor = (
        float(sum(float(r["anchor_top1"]) for r in judged) / len(judged)) if judged else 0.0
    )
    return {
        "per_cell": rows,
        "n_cells": int(len(judged)),
        "n_not_worse": int(sum(1 for r in judged if r["not_worse"])),
        "n_strictly_worse": int(len(worse)),
        "all_not_worse": bool(judged and not worse),
        "min_margin": float(min(margins)) if margins else 0.0,
        "mean_margin": float(sum(margins) / len(margins)) if margins else 0.0,
        "excluded_cells": sorted(excluded),
        "n_excluded_cells": int(len(usable) - len(judged)),
        "per_cell_including_clean": [
            {"cell": str(r.get("cell")), "margin": float(r.get("margin", 0.0))}
            for r in usable
        ],
        "all_not_worse_including_clean": bool(
            usable and not [r for r in usable if not r["not_worse"]]
        ),
        "aggregate_reading": {
            "cell_set": "9 个扰动格（与判据同集合）",
            "mean_post_top1": float(agg_post),
            "mean_anchor_top1": float(agg_anchor),
            "mean_margin": float(agg_post - agg_anchor),
            "not_worse_under_aggregate_reading": bool(agg_post >= agg_anchor),
            "note": (
                "**不参与判定**：判据口径是**逐格**（与第二步 `primary_criterion` 的「均不低于」"
                "同一读法）；此处给出的整段平均读法仅为如实登记 —— 两种读法在本轮可能给出不同"
                "倾向，不得只报其中一种。"
            ),
        },
        "row_source": str(post.get("row_source", "")),
        "granularity": TRAIN_GRANULARITY_NOTE,
        "rule": TRAIN_HARMLESSNESS_RULE,
    }


def train_and_eval_profile(
    profile: str,
    *,
    cfg: Optional[TrainConfig] = None,
    product_dir: str = "",
    skip_identity_gate: bool = False,
    log: Any = None,
) -> Dict[str, Any]:
    """对一个特征档跑「恒等锚点 → 恒等门禁 → 扰动自监督训练 → 逐格对照」的完整链路。

    训练**前**先把恒等锚点（T=I 在**训练行 1999** 上的 clean + 9 格）算出来 ——
    它是本轮「训练无害」判据的硬性下限；训练**后**在**同一行集合、同一口径**上再算一遍，
    两者逐格相减即「训练有没有在自己的目标上变差」。

    返回
    ----
    Dict[str, Any]
        ``profile`` / ``identity_gate`` / ``identity_anchor`` / ``train_side_post`` /
        ``train_harmlessness`` / ``update_gate`` / ``train`` / ``cell_grid`` /
        ``parameter_snapshot_after`` / ``post_train_identity_evidence`` / ``data`` / ``cfg``。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    cfg = cfg or TrainConfig()
    data = load_profile_data(str(profile), product_dir=str(product_dir), log=_log)
    model = build_model(data.table)
    if skip_identity_gate:
        gate: Dict[str, Any] = {
            "skipped": True,
            "note": (
                "**显式跳过**恒等门禁（--skip-identity-gate）；"
                "仅允许在已单独执行过 `variantb drill` 门禁的前提下使用"
            ),
        }
        _log("[variantb train] 恒等门禁：**跳过**（--skip-identity-gate）")
    else:
        _log(f"[variantb train] 档 {profile}：训练前恒等门禁 ...")
        gate = identity_gate(data, seed=int(cfg.seed), batch_size=int(cfg.batch_size))
        _log(
            f"[variantb train]   恒等门禁 {gate['n_cells_all_equal']}/{gate['n_cells']} "
            f"格逐条一致，通过 = {gate['passed']}"
        )
    _log(f"[variantb train] 档 {profile}：恒等锚点（T=I × 训练行 1999）...")
    anchor = identity_anchor_grid(
        model, data, seed=int(cfg.seed), batch_size=int(cfg.batch_size)
    )
    _log(
        f"[variantb train]   恒等锚点（clean + 9 格）自命中 = "
        f"{[round(float(v), 6) for v in anchor['top1_self'].values()]}"
    )
    before_snap = model.parameter_snapshot()
    _log(
        f"[variantb train] 档 {profile}：开始训练（mode={cfg.train_mode}，"
        f"scorer={cfg.train_scorer}，epochs={cfg.epochs}，batch={cfg.batch_size}，"
        f"lr={cfg.lr}，wd={cfg.weight_decay}）..."
    )
    train_info = train_transform(
        model, data.table.keys.numpy(), data.library_index, cfg, log=_log
    )
    after_snap = model.parameter_snapshot()
    up = update_gate(before_snap, after_snap)
    _log(
        f"[variantb train] 档 {profile}：可训参数更新 {up['n_updated']}/{up['n_total']}，"
        f"通过 = {up['passed']}"
    )
    _log(f"[variantb train] 档 {profile}：训练后逐格评测（查询行 666，clean + 9 扰动档）...")
    grid = cell_grid(model, data, seed=int(cfg.seed), batch_size=int(cfg.batch_size))
    _log(f"[variantb train] 档 {profile}：训练后训练侧逐格评测（训练行 1999，同口径）...")
    post = cell_grid(
        model, data, seed=int(cfg.seed), batch_size=int(cfg.batch_size),
        rows=data.library_index,
    )
    harm = train_harmlessness(anchor, post)
    _log(
        f"[variantb train] 档 {profile}：训练无害下限（训练后 ≥ 恒等锚点）= "
        f"{harm['all_not_worse']}（不劣 {harm['n_not_worse']}/{harm['n_cells']}，"
        f"最差余量 {harm['min_margin']:+.6f}）"
    )
    with torch.no_grad():
        bitwise_after = sample_bitwise_equality(model, data.query_features())
        explicit_after = explicit_path_evidence(model, data.query_features())
        path_after = score_path_evidence(model, data.query_features())
    return {
        "profile": str(profile),
        "data": data.as_dict(),
        "cfg": cfg.as_dict(),
        "identity_gate": gate,
        "identity_anchor": anchor,
        "train_side_post": post,
        "train_harmlessness": harm,
        "update_gate": up,
        "parameter_snapshot_before": before_snap,
        "parameter_snapshot_after": after_snap,
        "train": train_info,
        "cell_grid": grid,
        # [!] 训练后权重（张量）只作**进程内**传递：`run_variant_b` 会把它弹出放进
        # 独立字典供决定性对照臂复用，**绝不进产物**（JSON 无法序列化张量）。
        "_weights": {
            str(k): v.detach().cpu().to(torch.float32).clone()
            for k, v in model.transform.state_dict().items()
        },
        "post_train_identity_evidence": {
            "transform_is_identity": bool(model.transform.is_identity()),
            "bitwise_norm_equality": bitwise_after,
            "explicit_path": explicit_after,
            "score_path": path_after,
            "note": (
                "训练后 T 已离开恒等 ⇒ 恒等快路径失效、`norm(T(x)) != norm(x)` 属**预期**；"
                "`score_path.bitwise_equal` 则必须**恒为 True**（评测侧确实走了变换层，"
                "首版漏掉变换层的缺陷由它拦下）"
            ),
        },
    }


def primary_criterion(
    profile_results: Dict[str, Dict[str, Any]],
    *,
    effective_cells: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """**主判据**：1a 判有效的扰动格上，变体 B 训练后 R@1 不低于 KNN 且至少 1 格严格更高。

    参数
    ----
    profile_results : Dict[str, Dict[str, Any]]
        ``{档名: train_and_eval_profile(...) 的结果}``。
    effective_cells : Optional[Dict[str, Sequence[str]]]
        每档判有效的扰动格（``"kind/level"``）；缺省用 :data:`EFFECTIVE_CELLS_1A`。

    返回
    ----
    Dict[str, Any]
        ``per_cell`` / ``n_cells`` / ``n_not_worse`` / ``n_strictly_better`` /
        ``n_strictly_worse`` / ``at_least_one_better`` / ``passed`` / ``criterion`` /
        ``naive_direction``。
    """
    cells_of = dict(effective_cells or EFFECTIVE_CELLS_1A)
    rows: List[Dict[str, Any]] = []
    for profile, res in sorted(profile_results.items()):
        wanted = [str(x) for x in cells_of.get(str(profile), ())]
        by_cell = {str(c["cell"]): c for c in res["cell_grid"]["cells"]}
        for cell in wanted:
            c = by_cell.get(str(cell))
            if c is None:
                rows.append(
                    {
                        "profile": str(profile),
                        "cell": str(cell),
                        "applicable": False,
                        "note": "该格未在本次网格中产出（如实登记，不静默跳过）",
                    }
                )
                continue
            b = float(c["variant_b"]["recall_at_1"])
            k = float(c["knn"]["recall_at_1"])
            rows.append(
                {
                    "profile": str(profile),
                    "cell": str(cell),
                    "kind": str(c["kind"]),
                    "level": str(c["level"]),
                    "variant_b": float(b),
                    "knn": float(k),
                    "delta": float(b - k),
                    "variant_b_hit": int(c["variant_b"]["hit_at_1"]),
                    "knn_hit": int(c["knn"]["hit_at_1"]),
                    "n": int(c["variant_b"]["n"]),
                    "not_worse": bool(b >= k),
                    "strictly_better": bool(b > k),
                    "applicable": True,
                }
            )
    usable = [r for r in rows if r.get("applicable")]
    better = [r for r in usable if r["strictly_better"]]
    worse = [r for r in usable if not r["not_worse"]]
    return {
        "criterion": (
            "在 1a 判**有效**的扰动格上（`lexical-88` 的 noise/mask/nmag 三格；"
            "`bge-m3-1024` 的 noise 与 nmag 两格），变体 B 训练后 R@1 **均不低于 KNN**，"
            "且**至少 1 格严格更高**"
        ),
        "effective_cells_source": (
            "1a 生产实测（`checkpoints/qa_learn/_verify/robust/robust_calibration.json`）；"
            "该门禁**只用于判据**，不用来过滤训练数据"
        ),
        "per_cell": rows,
        "n_cells": int(len(usable)),
        "n_not_worse": int(sum(1 for r in usable if r["not_worse"])),
        "n_strictly_better": int(len(better)),
        "n_strictly_worse": int(len(worse)),
        "at_least_one_better": bool(len(better) > 0),
        "all_not_worse": bool(len(worse) == 0 and len(usable) > 0),
        "passed": bool(len(worse) == 0 and len(better) > 0 and len(usable) > 0),
        "naive_direction": (
            "计划口径第 4 条声明的朴素方向是「扰动自监督会让 argmax 变差」；"
            "本判据**不预设方向**，只现场报 Δ 的符号与条数 —— 若全部 Δ = 0，"
            "结论是「**未测出差异**」而**不是**「负结果」"
        ),
    }


def _flatten_cells(
    per_profile: Dict[str, Dict[str, Any]], arms: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """把逐档网格摊平成便于逐格读的对照表（含 A 组对照列与 1a 有效性标记）。"""
    rows: List[Dict[str, Any]] = []
    for profile, res in sorted(per_profile.items()):
        arm = arms.get(profile)
        arm_cells = {str(c["cell"]): c for c in (arm["cell_grid"]["cells"] if arm else [])}
        for c in res["cell_grid"]["cells"]:
            cell = str(c["cell"])
            row: Dict[str, Any] = {
                "profile": str(profile),
                "cell": cell,
                "kind": str(c["kind"]),
                "level": str(c["level"]),
                "variant_b": dict(c["variant_b"]),
                "knn": dict(c["knn"]),
                "delta": dict(c["delta"]),
                "effective_1a": bool(cell in tuple(EFFECTIVE_CELLS_1A.get(str(profile), ()))),
                "ineffective_1a": bool(
                    cell in tuple(INEFFECTIVE_CELLS_1A.get(str(profile), ()))
                ),
            }
            if arm is not None:
                a = arm_cells.get(cell)
                if a is not None:
                    row["arm_a_clean_train"] = {
                        "recall_at_1": float(a["variant_b"]["recall_at_1"]),
                        "recall_at_5": float(a["variant_b"]["recall_at_5"]),
                        "hit_at_1": int(a["variant_b"]["hit_at_1"]),
                        "delta_vs_knn": float(a["delta"]["recall_at_1"]),
                        "delta_vs_arm_b": float(
                            a["variant_b"]["recall_at_1"] - c["variant_b"]["recall_at_1"]
                        ),
                    }
            rows.append(row)
    return rows


def _honest_notes(
    per_profile: Dict[str, Dict[str, Any]],
    arms: Dict[str, Dict[str, Any]],
    primary: Dict[str, Any],
    clean_eval_arm: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """如实登记（**与实测值绑定**；禁止把「没测出来」写成负结果）。"""
    notes: List[str] = []
    notes.append(
        "**单 seed 42**：所有 Δ 为**单点差、无跨 seed 极差**；本报告不提供训练随机性区间。"
    )
    notes.append(
        "**档 0（clean）不作为判据**：查询行 = 键表行的逐位自匹配（R@1 结构性 = 1.000000），"
        "变体 B 在 T=I 时同样为 1.000000，该格无区分度。"
    )
    notes.append(
        "**训练侧打分口径（本轮新增开关）**：`config.train_scorer` = "
        f"{sorted({str(dict(v.get('cfg', {})).get('train_scorer')) for v in per_profile.values()})} —— "
        "`normalized` 与评测**同一口径**（先变换、再 L2 归一化、后内积），"
        "`raw` = 现状未归一化口径（显式对照档）。评测侧**永远**用归一化口径。"
    )
    notes.append(
        "**KNN 基线 = 冻结键表上的归一化余弦 top-1**，与训练完全无关（键表不参与梯度）；"
        "它同时就是 1a 登记的「档 0 / 弱 / 中 / 强」基线，本模块现场重算并在 "
        "`anchor_1a_check` 中逐格对账（容差 1e-6）。"
    )
    for profile, res in sorted(per_profile.items()):
        gate = res["identity_gate"]
        if gate.get("skipped"):
            notes.append(f"**{profile}**：恒等门禁被**显式跳过**（`--skip-identity-gate`）。")
            continue
        notes.append(
            f"**{profile}**：恒等门禁 {gate['n_cells_all_equal']}/{gate['n_cells']} 格逐条一致，"
            f"范数逐位相等比例 {gate['bitwise_norm_equality']['bitwise_equal_frac']}，"
            f"显式路径逐位相同 {gate['explicit_path']['bitwise_equal']}，通过 = {gate['passed']}。"
        )
    for profile, res in sorted(per_profile.items()):
        ups = res["update_gate"]
        top1_val, top1_key = train_batch_running_top1_of(res.get("train"))
        key_note = (
            ""
            if top1_key == TRAIN_TOP1_KEY
            else (
                f"（**取值键 = `{top1_key}`**"
                + (
                    f"：旧产物沿用旧键名 `{LEGACY_TRAIN_TOP1_KEY}`，本值经**回退**取到真值"
                    if top1_key == LEGACY_TRAIN_TOP1_KEY
                    else "：**两键都缺失**，`0.000000` 不是可信读数"
                )
                + "）"
            )
        )
        notes.append(
            f"**{profile}**：可训参数更新 {ups['n_updated']}/{ups['n_total']}"
            f"（从未更新 {ups['n_never_updated']}）；批内滚动平均（仅诊断）= "
            f"{top1_val:.6f}{key_note}"
            "（**训练后量**、**仅诊断**；不是判定量 —— 判定量是逐格 R@1，见下一条）。"
        )
        harm = dict(res.get("train_harmlessness") or {})
        if harm:
            notes.append(
                f"**{profile} 训练无害下限（本轮主判据）**：训练后在**训练行 1999** 上，"
                f"**9 个扰动格**中不低于恒等锚点的格数 = {harm['n_not_worse']}/{harm['n_cells']}"
                f"（严格更低 {harm['n_strictly_worse']}），最差余量 {harm['min_margin']:+.6f}、"
                f"平均余量 {harm['mean_margin']:+.6f}；排除的格 = {harm.get('excluded_cells')}"
                f"（逐位自匹配、结构性 1.000000）。{harm['granularity']}"
            )
        nd = dict(dict(res.get("train") or {}).get("norm_diagnostics_summary") or {})
        if nd:
            notes.append(
                f"**{profile} 范数膨胀通道（机制取证，不参与判定）**：`‖T(x)‖` 末/首 = "
                f"{nd.get('norm_ratio_last_over_first')}（首 {nd.get('mean_row_norm_Tx_first')} → "
                f"末 {nd.get('mean_row_norm_Tx_last')}），`‖W−I‖_F` 末值 = "
                f"{nd.get('weight_minus_identity_fro_last')}，`mean|logit|` 首 → 末 = "
                f"{nd.get('mean_logit_abs_first')} → {nd.get('mean_logit_abs_last')}。"
            )
    if arms:
        notes.append(
            "**A 组（干净自监督）对照口径**：训练查询不扰动、其余完全相同 —— 用于把"
            "「干净训练 ≈ 无变化」从推理变成**实测**；其逐格数字写在 "
            "`cells[].arm_a_clean_train`。"
        )
    for profile, res in sorted((clean_eval_arm or {}).items()):
        d = dict(res["clean_cells"]["delta"])
        notes.append(
            f"**决定性对照臂（{profile}）**：扰动自监督训练的模型 × **clean 评测** ⇒ "
            f"ΔR@1 = {float(d['recall_at_1']):+.6f}"
            f"（命中 {int(res['clean_cells']['variant_b']['hit_at_1'])}/"
            f"{int(res['clean_cells']['knn']['hit_at_1'])}）。该格把「T 本身的方向偏移」"
            "与「扰动泛化落差」分开：clean 格 Δ 显著为负即说明**掉点来自变换层本身**，"
            "而不是「训练扰动与评测扰动不同一次随机绘制」。"
        )
    notes.append(
        "**1a 判无效的格仍然报数字**（`bge-m3-1024` 的 mask 三档，1a 落差 0.000000）："
        "它们只作如实登记，**不参与主判据**（带着坏尺子做判定会得到无意义的结论）。"
    )
    notes.append(
        "**空间不足的格单列**：`nmag/strong`（KNN 0.992492，仅 5/666 条空间）与 "
        "`bge-m3-1024` 的 mask（KNN 1.000000，**零空间**）在 `cell_grid.error_space` 中"
        "被显式标出；在这些格上「Δ >= 0」的判别力极弱、而「Δ > 0」在零空间格里不可能成立，"
        "**不得**把它们读成变体 B 的能力结论。"
    )
    notes.append(
        "**训练侧扰动在批内施加**：1a 的 `perturb_matrix` 内部按**批内行号**生成噪声 / 掩码 / "
        "棋盘平移，故批大小会改变每一行的扰动实现；评测侧则是「全部查询一次扰动」。"
        "两者用同一实现与同一派生种子公式，但行号基准不同 —— 该事实登记在 "
        "`train.train_perturb_note`，不得被读成同一写法的两种等价表达。"
    )
    znotes: List[str] = []
    for profile, res in sorted(per_profile.items()):
        z = dict(res["train"].get("zero_norm_rows", {}))
        znotes.append(
            f"{profile} 掩码零范数行 {z.get('total', 0)} 条 / 涉及 "
            f"{z.get('n_steps_with_zero_row', 0)} 步"
        )
    notes.append(
        "**训练侧零范数行的边界处置（现场实测）**：批内某行被扰动后整行归零是**真实可达**的"
        "（`mask/strong` 遮蔽 44/88 维），该行原样保留为全零向量但**在损失中掩码**"
        "（掩码取自**变换前**的输入 ⇒ 该行一律不参与交叉熵与梯度；离朱 R59 F4："
        "改前写的「logits 恒为 0」只在 `T = I` 时成立，训练后 `T(0) = b != 0`），"
        "逐批计数进产物；整批全零则**显式报错**。逐档实测：" + "；".join(znotes) + "。"
        "1a 的 `perturb_matrix` 对零范数行仍**照旧显式报错**（该契约未改动）。"
    )
    if int(primary["n_strictly_better"]) == 0 and int(primary["n_strictly_worse"]) == 0:
        notes.append(
            "**主判据未成立，且其形态是「全部 Δ = 0」** —— 按口径必须写成"
            "「**未测出差异**」，**不是**负结果、也不是「训练无提升」的证明："
            "本轮的分辨率（666 条查询、单 seed）不足以在这一形态上判定因果。"
        )
    elif not primary["passed"]:
        notes.append(
            "**主判据未成立**：存在 Δ < 0 的格（详见 `primary.per_cell`）；如实报负结论，"
            "不包装；第三步（候选粒度改造）的方向依据见 `third_step`。"
        )
    return notes


def clean_eval_of_perturb_arm(
    profile: str,
    *,
    cfg: Optional[TrainConfig] = None,
    product_dir: str = "",
    perturb_train: Optional[Dict[str, Any]] = None,
    perturb_weights: Optional[Dict[str, torch.Tensor]] = None,
    log: Any = None,
) -> Dict[str, Any]:
    """**决定性对照臂**：用**扰动自监督**训练出的模型，去评测 **clean（无扰动）** 格。

    为什么需要这一臂（否证条款要求的"方向依据"）
    ------------------------------------------
    主臂在扰动格上同时混进两类原因：① 变换 ``T`` 本身改变了检索方向；
    ② 训练用的扰动实现与评测用的扰动实现**不是同一次随机绘制**。
    只看扰动格无法把两者分开。本臂把 ② 消掉（评测查询不扰动），因此：

    * 若 clean 格的 Δ 显著为负 ⇒ 掉点来自 **``T`` 本身**，
      即「单层 ``D→D`` 变换 + 冻结**逐行**键表」这一粒度上，训练会把
      已经在表内的行推离自己的键；
    * 若 clean 格 Δ ≈ 0 而只有扰动格掉 ⇒ 掉点主要来自
      **扰动自监督的泛化落差**，方向依据应落在训练信号而不是模型粒度上。

    参数
    ----
    profile : str
        特征档名。
    cfg : Optional[TrainConfig]
        训练配置（会强制 ``train_mode="perturb"``；评测侧不扰动）。
    product_dir : str
        ``n3d_qa`` 冻结产物目录。
    perturb_train : Optional[Dict[str, Any]]
        主臂（:func:`train_and_eval_profile`）的结果；与 ``perturb_weights`` 一起使用时
        **直接复用主臂训练后的权重**（逐位同一模型，不做重建），二者都为 ``None`` 时才现训一遍。
    perturb_weights : Optional[Dict[str, torch.Tensor]]
        主臂训练后的 ``transform.state_dict()``（含张量，**不进产物**，只在进程内传递）。
    log : Any
        可调用日志。

    返回
    ----
    Dict[str, Any]
        ``profile`` / ``clean_cells``（clean 格指标 + 与 KNN 的 Δ）/ ``score_path`` /
        ``update_gate`` / ``train``（训练摘要）/ ``weights_reused`` / ``reading``。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    cfg = cfg or TrainConfig()
    if str(cfg.train_mode) != "perturb":
        raise ValueError(
            f"本臂要求扰动自监督训练（train_mode='perturb'），当前 {cfg.train_mode!r}"
        )
    data = load_profile_data(str(profile), product_dir=str(product_dir), log=_log)
    model = build_model(data.table)
    weights_reused = False
    if perturb_train is not None and perturb_weights is not None:
        model.transform.load_state_dict(
            {str(k): v.detach().cpu().to(torch.float32) for k, v in perturb_weights.items()}
        )
        model.transform.train(False)
        train_info = dict(perturb_train.get("train", {}))
        before = list(perturb_train.get("parameter_snapshot_before", []))
        weights_reused = True
        _log(
            f"[variantb arm-clean-eval] 档 {profile}：**复用主臂训练后权重**"
            "（逐位同一模型，不重建），只在 clean 格上评测"
        )
    else:
        before = build_model(data.table).parameter_snapshot()
        _log(f"[variantb arm-clean-eval] 档 {profile}：扰动自监督训练（现训）...")
        train_info = train_transform(
            model, data.table.keys.numpy(), data.library_index, cfg, log=_log
        )
    up = update_gate(before, model.parameter_snapshot())
    with torch.no_grad():
        b_clean, k_clean, detail = eval_cell(
            model, data, kind="noise", level="weak", clean=True,
            seed=int(cfg.seed), batch_size=int(cfg.batch_size),
        )
        path_ev = score_path_evidence(model, data.query_features())
    _log(
        f"[variantb arm-clean-eval] 档 {profile}：clean 格 KNN R@1 = "
        f"{k_clean.recall_at_1:.6f} / 变体 B = {b_clean.recall_at_1:.6f}"
        f"（Δ {b_clean.recall_at_1 - k_clean.recall_at_1:+.6f}，命中 "
        f"{b_clean.hit_at_1}/{k_clean.hit_at_1}）"
    )
    return {
        "profile": str(profile),
        "weights_reused_from_main_arm": bool(weights_reused),
        "clean_cells": {
            "cell": "clean",
            "variant_b": b_clean.as_dict(),
            "knn": k_clean.as_dict(),
            "delta": {
                "recall_at_1": float(b_clean.recall_at_1 - k_clean.recall_at_1),
                "recall_at_5": float(b_clean.recall_at_5 - k_clean.recall_at_5),
                "hit_at_1": int(b_clean.hit_at_1 - k_clean.hit_at_1),
            },
            "evidence": detail,
        },
        "score_path": path_ev,
        "update_gate": up,
        "train": train_info,
        "reading": (
            "本臂评测查询**不扰动**，故 Δ 只反映「训练出的变换 T 本身」对检索方向的影响；"
            "与主臂扰动格的 Δ 相减即可把「T 本身」与「扰动泛化落差」分开。"
            "**注意**：clean 格的 KNN 侧是逐位自匹配（R@1 = 1.000000），"
            "故该格只用于读 Δ 的**符号与量级**，不参与主判据。"
        ),
    }


def third_step_evidence(report: Dict[str, Any]) -> Dict[str, Any]:
    """否证条款要求的**第三步方向依据**（由本报告的现场数字直接推出，不空谈）。"""
    primary = dict(report.get("primary", {}))
    rows = [r for r in primary.get("per_cell", []) if r.get("applicable")]
    deltas = [float(r["delta"]) for r in rows]
    train_top1: Dict[str, float] = {}
    train_top1_key: Dict[str, str] = {}
    for p, res in sorted(report.get("per_profile", {}).items()):
        val, key = train_batch_running_top1_of(res.get("train"))
        train_top1[str(p)] = float(val)
        train_top1_key[str(p)] = str(key)
    legacy_fallback = sorted(
        k for k, v in train_top1_key.items() if v == LEGACY_TRAIN_TOP1_KEY
    )
    arms = report.get("arms", {}).get("clean_self_supervised", {})
    arm_delta: Dict[str, Any] = {}
    for profile, res in sorted(arms.items()):
        vals = [float(c["delta"]["recall_at_1"]) for c in res["cell_grid"]["cells"]]
        arm_delta[str(profile)] = {
            "max_abs_delta": float(max(abs(v) for v in vals)) if vals else 0.0,
            "n_nonzero_delta": int(sum(1 for v in vals if v != 0.0)),
            "n_cells": int(len(vals)),
        }
    clean_eval = report.get("arms", {}).get("perturb_train_clean_eval", {})
    clean_eval_delta = {
        str(p): float(res["clean_cells"]["delta"]["recall_at_1"])
        for p, res in sorted(clean_eval.items())
    }
    return {
        "delta_histogram": {
            "n_cells": int(len(deltas)),
            "n_zero": int(sum(1 for v in deltas if v == 0.0)),
            "n_positive": int(sum(1 for v in deltas if v > 0.0)),
            "n_negative": int(sum(1 for v in deltas if v < 0.0)),
            "max": float(max(deltas)) if deltas else 0.0,
            "min": float(min(deltas)) if deltas else 0.0,
        },
        "train_self_top1": train_top1,
        "train_self_top1_rule": TRAIN_TOP1_SELF_RULE,
        "train_self_top1_note": (
            "本字段是**训练循环内的批内滚动平均**（旧名沿用，仅为兼容既有产物键），"
            "**不是**训练前提校准轮的判定量；判定量是训练行 1999 上的**逐格 R@1**。"
            "**[!] 取值键与旧键回退（皋陶审查 error 修复）**：本块的每个值都**带旧键回退** —— "
            f"优先 `{TRAIN_TOP1_KEY}`（新键），缺失时回退 `{LEGACY_TRAIN_TOP1_KEY}`（**旧键**："
            "口径收口修复轮改名之前、以及第二步**冻结产物**里只有它），两者都缺失才是 `0.0`。"
            "实际命中的键名逐档登记在 `train_self_top1_key_used`；若该档命中旧键，"
            "其键名还会出现在 `train_self_top1_legacy_fallback_profiles`。"
            "**读者据此即可判断该值取自哪个键**，不会把回退值误读成 0。"
        ),
        "train_self_top1_key_used": dict(train_top1_key),
        "train_self_top1_legacy_fallback_profiles": list(legacy_fallback),
        "train_self_top1_key_fallback_rule": TRAIN_TOP1_KEY_FALLBACK_RULE,
        "arm_a_clean_train_delta_abs_max": arm_delta,
        "perturb_train_clean_eval_delta": clean_eval_delta,
        "reading": (
            "① 主要按**决定性对照臂**（扰动训练 × clean 评测）读：该臂 Δ 显著为负 ⇒ "
            "掉点来自**变换层本身**，即「单层 D→D 变换 + 冻结**逐行**键表」这一粒度上，"
            "argmax 与变换方向彼此拉扯，训练把表内行推离自己的键；此时第三步应改**候选粒度**"
            "（逐行键 → 类/簇级候选，或让键表随表示重算），而不是继续加大变换层容量；"
            "② 若决定性臂 Δ ≈ 0 而只有扰动格掉 ⇒ 掉点主要来自**扰动泛化落差**"
            "（训练扰动与评测扰动不是同一次随机绘制），应改**训练信号**（同分布多次绘制 / 更强正则），"
            "而非首先改粒度；"
            "③ 若训练侧的**批内滚动平均**自命中未显著高于随机 ⇒ 变换层连训练集都没拟合，"
            "应先把 **D→D 线性变换的表达力 / 优化**这条前提落实，再谈粒度改造；"
            "**注意**：判定该前提请用 `variantb calibrate` 的**逐格 R@1**（与恒等锚点同口径），"
            "不要用这里的批内滚动平均（:data:`TRAIN_TOP1_SELF_RULE`）。"
        ),
    }


# ---------------------------------------------------------------------------
# 7. 训练前提校准：锚点 / 口径开关 / 超参网格 / 机器可读归因
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrateCombo:
    """优化前提扫描的**一个组合**（``train_scorer × lr × weight_decay``）。

    属性
    ----
    train_scorer : str
        训练侧打分口径（``normalized`` / ``raw``）。
    lr : float
        Adam 学习率。
    weight_decay : float
        Adam 权重衰减。
    """

    train_scorer: str
    lr: float
    weight_decay: float

    @property
    def label(self) -> str:
        """组合标签（产物 key / 日志用；确定性、无时间字段）。"""
        return f"{self.train_scorer}|lr={float(self.lr):g}|wd={float(self.weight_decay):g}"

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化。"""
        return {
            "label": self.label,
            "train_scorer": str(self.train_scorer),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
        }

    @staticmethod
    def parse(spec: str) -> "CalibrateCombo":
        """解析 ``scorer:lr:wd`` 形式的单组合规格（CLI ``--grid-combo`` 用）。"""
        parts = [x.strip() for x in str(spec).split(":")]
        if len(parts) != 3:
            raise ValueError(
                f"--grid-combo 需要 `scorer:lr:weight_decay` 三段，当前 {spec!r}"
            )
        scorer = parts[0]
        if scorer not in TRAIN_SCORERS:
            raise ValueError(
                f"--grid-combo 的打分口径 {scorer!r} 未登记；可用 = {list(TRAIN_SCORERS)}"
            )
        try:
            lr = float(parts[1])
            wd = float(parts[2])
        except ValueError as exc:
            raise ValueError(f"--grid-combo 的 lr / weight_decay 不是浮点数：{spec!r}") from exc
        return CalibrateCombo(train_scorer=str(scorer), lr=float(lr), weight_decay=float(wd))


def build_calibration_grid(
    *,
    scorers: Sequence[str] = CALIBRATE_SCORERS,
    lrs: Sequence[float] = CALIBRATE_LR_GRID,
    weight_decays: Sequence[float] = CALIBRATE_WD_GRID,
) -> List[CalibrateCombo]:
    """构造确定性的最小网格（**scorer 外层、lr 中层、wd 内层**；顺序即复跑顺序）。"""
    bad = [str(s) for s in scorers if str(s) not in TRAIN_SCORERS]
    if bad:
        raise ValueError(f"扫描含未登记的打分口径 {bad}；可用 = {list(TRAIN_SCORERS)}")
    for lr in lrs:
        if float(lr) <= 0.0:
            raise ValueError(f"扫描的 lr 必须 > 0，当前 {lr}")
    for wd in weight_decays:
        if float(wd) < 0.0:
            raise ValueError(f"扫描的 weight_decay 必须 >= 0，当前 {wd}")
    return [
        CalibrateCombo(train_scorer=str(s), lr=float(lr), weight_decay=float(wd))
        for s in scorers
        for lr in lrs
        for wd in weight_decays
    ]


def _knn_r1_row(cell: Dict[str, Any]) -> float:
    """取一个逐格网格单元里 **KNN 侧** 的 R@1（KNN 与模型无关，用于组合间不变量核验）。"""
    return float(dict(cell["knn"])["recall_at_1"])


def _md_table_cell(text: Any) -> str:
    """Markdown **表格单元格**转义（离朱 R61 O3）。

    组合标签形如 ``normalized|lr=0.001|wd=0``，其中的 ``|`` 若原样写进表格单元格，
    任何 Markdown 渲染器都会把该行**拆成多余列**（整张表错位）。
    故表格单元格内的 ``|`` 一律转义为 ``\\|``（JSON 里的键名不受影响 —— 转义只作用在渲染层）。
    """
    return str(text).replace("|", "\\|")


#: 训练侧「批内滚动平均」的**现行键名**（唯一来源）。
TRAIN_TOP1_KEY: str = "train_batch_running_top1"

#: 该字段的**旧键名**（口径收口修复轮之前的产物里只有它；第二步冻结产物即属此类）。
LEGACY_TRAIN_TOP1_KEY: str = "train_top1_self"

#: 读取该字段的**回退规则文本**（唯一来源；进产物）。
TRAIN_TOP1_KEY_FALLBACK_RULE: str = (
    "读取 `train` 块的**批内滚动平均**时，取值优先级固定为："
    f"`{TRAIN_TOP1_KEY}`（新键，现行唯一键名）→ `{LEGACY_TRAIN_TOP1_KEY}`（**旧键**："
    "口径收口修复轮改名之前的产物只有它，且第二步冻结产物按零回归要求**不得重建**）→ "
    "`0.0`（**两键都缺失**时才用；`0.0` 不是「没测到」的可信读数，调用方须据取值键名判断）。"
    "回退**只在旧键存在且新键缺失时**生效 ⇒ **新产物路径的读数逐位不变**。"
    "若两键同时存在（异常情况）⇒ **新键优先**，并在产物里把实际命中的键名如实登记。"
)


def train_batch_running_top1_of(block: Any) -> Tuple[float, str]:
    """从 ``train`` 块取**批内滚动平均**（**带旧键回退**；本字段取值的**唯一实现**）。

    存在理由（皋陶审查 error —— 改名引入的向后兼容回归）
    --------------------------------------------------
    口径收口修复轮把 ``train_top1_self`` 改名为 ``train_batch_running_top1`` 时，读取侧只写了
    ``.get(新键, 0.0)``；而**冻结的第二步产物**只有旧键、且按零回归要求不得重建 ⇒ 同一函数在
    旧产物上**静默返回 `0.0`**（真值应为 `0.34217108554277137` / `0.632816408204102`）。
    本函数把「新键 → 旧键 → ``0.0``」的优先级固化为**单一实现**，并返回实际命中的键名，
    供产物如实登记「该值取自哪个键」。

    参数
    ----
    block : Any
        ``per_profile.<档>.train`` 块（或任何含这两个键的映射；``None`` / 非映射亦可）。

    返回
    ----
    Tuple[float, str]
        ``(值, 命中的键名)``；键名 ∈ {``TRAIN_TOP1_KEY``, ``LEGACY_TRAIN_TOP1_KEY``, ``"missing"``}。
    """
    data = block if isinstance(block, dict) else {}
    for key in (TRAIN_TOP1_KEY, LEGACY_TRAIN_TOP1_KEY):
        if key in data and data[key] is not None:
            return float(data[key]), str(key)
    return 0.0, "missing"


def _compact_cells(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把逐格网格压成**紧凑摘要**（只留 R@1/@5/命中数/Δ；不落取证字典与扰动指纹）。"""
    out: List[Dict[str, Any]] = []
    for c in grid.get("cells", []):
        out.append(
            {
                "cell": str(c["cell"]),
                "kind": str(c["kind"]),
                "level": str(c["level"]),
                "variant_b_recall_at_1": float(c["variant_b"]["recall_at_1"]),
                "variant_b_recall_at_5": float(c["variant_b"]["recall_at_5"]),
                "variant_b_hit_at_1": int(c["variant_b"]["hit_at_1"]),
                "knn_recall_at_1": float(c["knn"]["recall_at_1"]),
                "knn_recall_at_5": float(c["knn"]["recall_at_5"]),
                "knn_hit_at_1": int(c["knn"]["hit_at_1"]),
                "n": int(c["variant_b"]["n"]),
                "delta_recall_at_1": float(c["delta"]["recall_at_1"]),
            }
        )
    return out


def attribution_verdict(
    combo: CalibrateCombo,
    harmlessness: Dict[str, Any],
    primary: Dict[str, Any],
) -> Dict[str, Any]:
    """**机器可读的单点归因结论**（规则文本与判定同源，见 :data:`ATTRIBUTION_RULE`）。

    参数
    ----
    combo : CalibrateCombo
        被检的组合（判据要求 ``train_scorer == normalized``，否则本函数显式标注
        ``scorer_ok = False`` 并把结论判为不成立 —— 口径不符时**不做**归因）。
    harmlessness : Dict[str, Any]
        :func:`train_harmlessness` 的结果（条件 a）。
    primary : Dict[str, Any]
        :func:`primary_criterion` 的结果（条件 b）。

    返回
    ----
    Dict[str, Any]
        ``rule`` / ``combo`` / ``scorer_ok`` / ``conditions``（按
        :data:`ATTRIBUTION_REQUIRED_CONDITIONS` 逐个求值）/ ``conditions_required`` /
        ``verdict_key`` / ``verdict`` / ``single_source``。
    """
    conditions: Dict[str, bool] = {
        "a_train_harmless": bool(harmlessness.get("all_not_worse", False)),
        "b_eval_not_worse": bool(primary.get("all_not_worse", False)),
    }
    scorer_ok = bool(str(combo.train_scorer) == DEFAULT_TRAIN_SCORER)
    passed = bool(
        scorer_ok
        and all(bool(conditions[k]) for k in ATTRIBUTION_REQUIRED_CONDITIONS)
    )
    key = "optimization_premise" if passed else "linear_expressivity"
    # 备选读法（**不参与判定**）：把两个条件都按「整段平均」读时的结果。
    # 两种读法可能给出不同倾向，必须同时登记（只报一种就是包装）。
    agg = dict(harmlessness.get("aggregate_reading", {}))
    prim_rows = [r for r in primary.get("per_cell", []) if r.get("applicable")]
    prim_agg_post = (
        float(sum(float(r["variant_b"]) for r in prim_rows) / len(prim_rows))
        if prim_rows
        else 0.0
    )
    prim_agg_knn = (
        float(sum(float(r["knn"]) for r in prim_rows) / len(prim_rows)) if prim_rows else 0.0
    )
    alt_a = bool(agg.get("not_worse_under_aggregate_reading", False))
    alt_b = bool(prim_agg_post >= prim_agg_knn)
    alt_passed = bool(scorer_ok and alt_a and alt_b)
    return {
        "rule": ATTRIBUTION_RULE,
        "combo": combo.as_dict(),
        "scorer_ok": bool(scorer_ok),
        "conditions": {k: bool(conditions.get(k, False)) for k in ATTRIBUTION_CONDITION_TEXT},
        "conditions_required": list(ATTRIBUTION_REQUIRED_CONDITIONS),
        "conditions_text": dict(ATTRIBUTION_CONDITION_TEXT),
        "evidence_a": {
            "n_cells": int(harmlessness.get("n_cells", 0)),
            "n_not_worse": int(harmlessness.get("n_not_worse", 0)),
            "n_strictly_worse": int(harmlessness.get("n_strictly_worse", 0)),
            "min_margin": float(harmlessness.get("min_margin", 0.0)),
            "mean_margin": float(harmlessness.get("mean_margin", 0.0)),
            "granularity": TRAIN_GRANULARITY_NOTE,
            "per_cell": list(harmlessness.get("per_cell", [])),
        },
        "evidence_b": {
            "n_cells": int(primary.get("n_cells", 0)),
            "n_not_worse": int(primary.get("n_not_worse", 0)),
            "n_strictly_better": int(primary.get("n_strictly_better", 0)),
            "n_strictly_worse": int(primary.get("n_strictly_worse", 0)),
            "per_cell": list(primary.get("per_cell", [])),
        },
        "verdict_key": str(key),
        "verdict": str(ATTRIBUTION_VERDICTS[key]),
        "alternative_reading": {
            "note": (
                "**不参与判定**：把两个条件都按「整段平均」读（而不是逐格）时的结果。"
                "判据口径是**逐格**（与第二步 `primary_criterion` 的「均不低于」同一读法）；"
                "两种读法可能给出不同倾向，此处如实并列。"
            ),
            "a_train_harmless_aggregate": bool(alt_a),
            "a_evidence": {
                "mean_post_top1": float(agg.get("mean_post_top1", 0.0)),
                "mean_anchor_top1": float(agg.get("mean_anchor_top1", 0.0)),
                "mean_margin": float(agg.get("mean_margin", 0.0)),
            },
            "b_eval_not_worse_aggregate": bool(alt_b),
            "b_evidence": {
                "mean_variant_b_recall_at_1": float(prim_agg_post),
                "mean_knn_recall_at_1": float(prim_agg_knn),
                "mean_margin": float(prim_agg_post - prim_agg_knn),
            },
            "would_be_verdict_key": (
                "optimization_premise" if alt_passed else "linear_expressivity"
            ),
            "agrees_with_primary": bool(alt_passed == passed),
        },
        "single_source": (
            "本结论的**唯一来源**是本产物 `per_profile.<档>.combos[<组合>]` 下的 "
            "`train_harmlessness` 与 `eval_primary` 两块；规则文本由 "
            "`ATTRIBUTION_REQUIRED_CONDITIONS` 派生，与判定实现同源"
        ),
        "passed": bool(passed),
    }


def select_best_combo(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """按 :data:`CALIBRATE_SELECTION_RULE` 从组合记录里选出最优超参（确定性）。

    参数
    ----
    records : Sequence[Dict[str, Any]]
        每个组合的记录（须含 ``combo`` / ``train_harmlessness`` / ``eval_primary``）。

    返回
    ----
    Dict[str, Any]
        ``rule`` / ``applicable`` / ``selected``（标签）/ ``candidates``（排序后的关键量，
        每项含 ``train_post_top1_mean`` 及其**格集合标签** ``train_post_top1_mean_cell_set``）/
        ``selected_record`` / ``reason``（``applicable = False`` 时给出可读原因）。

    [!] **格集合口径（皋陶审查 warning 4）**：① ``n_train_harmless_cells`` 用 **9 个扰动格**
    （``train_harmlessness`` 已按 :data:`TRAIN_HARMLESSNESS_EXCLUDED_CELLS` 排除 ``clean``）；
    ② ``train_post_top1_mean`` 是 ``per_cell`` 中 ``applicable`` 行的平均 = **10 格**
    （``clean`` + 9 个扰动格）。两者的差异**不是**笔误：① 是判据量、② 是对照量，
    分别在 :data:`CALIBRATE_SELECTION_RULE` 与 ``train_post_top1_mean_cell_set`` 中显式标注。
    本函数**不再**使用训练循环的批内滚动平均（:data:`TRAIN_TOP1_SELF_RULE`）做排序。

    [!] **不再抛异常（R59 F2 修复）**：改前若网格里没有 ``normalized`` 组合（例如
    ``--calibrate-scorers raw``），本函数抛 ``ValueError`` ⇒ 被 CLI 兜成「非法参数 + 退码 2」，
    **一次已经跑完的训练成果被整条丢弃、`.md` 不渲染**。现改为返回 ``applicable = False``，
    由调用方如实登记「归因不适用」并**保留全部扫描结果**。
    """
    cands: List[Dict[str, Any]] = []
    for rec in records:
        combo = dict(rec["combo"])
        harm = dict(rec["train_harmlessness"])
        prim = dict(rec["eval_primary"])
        post_mean = 0.0
        rows = [r for r in harm.get("per_cell", []) if r.get("applicable")]
        if rows:
            post_mean = float(sum(float(r["post_top1"]) for r in rows) / len(rows))
        cands.append(
            {
                "label": str(combo.get("label")),
                "train_scorer": str(combo.get("train_scorer")),
                "lr": float(combo.get("lr", 0.0)),
                "weight_decay": float(combo.get("weight_decay", 0.0)),
                "n_train_harmless_cells": int(harm.get("n_not_worse", 0)),
                "train_post_top1_mean": float(post_mean),
                "train_post_top1_mean_cell_set": str(CALIBRATE_CANDIDATE_CELL_SET),
                "n_eval_not_worse": int(prim.get("n_not_worse", 0)),
                "n_eval_cells": int(prim.get("n_cells", 0)),
                "eligible": bool(str(combo.get("train_scorer")) == DEFAULT_TRAIN_SCORER),
            }
        )
    eligible = [c for c in cands if c["eligible"]]
    if not eligible:
        return {
            "rule": CALIBRATE_SELECTION_RULE,
            "applicable": False,
            "selected": "",
            "selected_record": None,
            "candidates": cands,
            "candidates_ranked": [],
            "ineligible_excluded": [str(c["label"]) for c in cands],
            "reason": (
                f"本次扫描里没有 `train_scorer = {DEFAULT_TRAIN_SCORER}` 的组合，"
                "而归因判据要求在该口径下选取最优超参 ⇒ 标「不适用」；"
                "**扫描结果全部保留在 `per_profile.<档>.combos`**，不以其它口径出归因结论"
            ),
        }
    eligible.sort(
        key=lambda c: (
            -int(c["n_train_harmless_cells"]),
            -float(c["train_post_top1_mean"]),
            -int(c["n_eval_not_worse"]),
            float(c["lr"]),
            float(c["weight_decay"]),
        )
    )
    best = eligible[0]
    chosen = next(r for r in records if str(dict(r["combo"]).get("label")) == best["label"])
    return {
        "rule": CALIBRATE_SELECTION_RULE,
        "applicable": True,
        "selected": str(best["label"]),
        "selected_record": chosen,
        "candidates": cands,
        "candidates_ranked": [c["label"] for c in eligible],
        "ineligible_excluded": [c["label"] for c in cands if not c["eligible"]],
        "reason": "",
    }


def attribution_not_applicable(
    reason: str, candidates: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """**归因不适用**的机器可读结论（网格里没有 ``normalized`` 组合时；R59 F2 修复）。

    参数
    ----
    reason : str
        可读原因（来自 :func:`select_best_combo` 的 ``reason``）。
    candidates : Sequence[Dict[str, Any]]
        本次扫描的全部组合关键量（**如实保留**，不因「不适用」而丢弃）。

    返回
    ----
    Dict[str, Any]
        ``rule`` / ``applicable = False`` / ``reason`` / ``candidates`` /
        ``verdict_key = "not_applicable"`` / ``verdict`` / ``single_source``。
    """
    return {
        "rule": ATTRIBUTION_RULE,
        "applicable": False,
        "reason": str(reason),
        "candidates": [dict(c) for c in candidates],
        "conditions": {k: None for k in ATTRIBUTION_CONDITION_TEXT},
        "conditions_required": list(ATTRIBUTION_REQUIRED_CONDITIONS),
        "conditions_text": dict(ATTRIBUTION_CONDITION_TEXT),
        "verdict_key": "not_applicable",
        "verdict": str(ATTRIBUTION_VERDICTS["not_applicable"]),
        "single_source": (
            "本次扫描未产出归因结论；扫描数字的唯一来源仍是本产物 "
            "`per_profile.<档>.combos`（逐组合的 `train_harmlessness` 与 `eval_primary`）"
        ),
        "passed": False,
    }


def run_calibration(
    profiles: Sequence[str],
    *,
    seed: int = VARIANT_B_SEED,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    combos: Optional[Sequence[CalibrateCombo]] = None,
    product_dir: str = "",
    skip_identity_gate: bool = False,
    on_progress: Any = None,
    log: Any = None,
) -> Dict[str, Any]:
    """**训练前提校准**：恒等锚点 + ``train_scorer × lr × weight_decay`` 网格 + 归因结论。

    流程（**每个组合都是「训一个模型 → 两套行集合上各跑一次 10 格网格」**）
    ----------------------------------------------------------------
    1. 每档：恒等门禁（硬门禁）→ 恒等锚点（T=I × 训练行 1999 × clean + 9 格）；
    2. 逐组合：训练（``train_scorer`` 由组合给定）→ 查询行 666 的 10 格（评测侧主判据）
       → 训练行 1999 的 10 格（训练无害下限）→ 范数诊断 / 更新量门禁；
    3. 按 :data:`CALIBRATE_SELECTION_RULE` 选最优超参 → :func:`attribution_verdict` 出结论。

    参数
    ----
    profiles : Sequence[str]
        特征档（本轮现场只跑 ``lexical-88``）。
    seed / epochs / batch_size : 训练口径（冻结；单 seed 42）。
    combos : Optional[Sequence[CalibrateCombo]]
        显式组合列表；``None`` = :func:`build_calibration_grid` 的最小网格。
    product_dir : str
        ``n3d_qa`` 冻结产物目录。
    skip_identity_gate : bool
        显式跳过恒等门禁（仅允许在已单独跑过 ``variantb drill`` 时使用）。
    on_progress : Any
        可调用回调；**每完成一个组合**调用一次，入参是当前的部分报告（``status = "partial"``）
        —— 支撑「扫描可中断、可单组合复跑」（CLI 据此增量落盘）。
    log : Any
        可调用日志。

    返回
    ----
    Dict[str, Any]
        校准报告（``status = "complete"``，含 ``selection`` / ``attribution`` /
        ``artifact_fingerprint``）。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    combo_list = list(combos) if combos is not None else build_calibration_grid()
    if not combo_list:
        raise ValueError("校准组合列表为空；拒绝空跑")
    report: Dict[str, Any] = {
        "module": MODULE_NAME,
        "artifact_schema": "variant-b-calibrate-v1",
        "status": "partial",
        "deterministic": True,
        "excluded_fields_note": (
            "本产物不含 created_utc / seconds / 主机名等非确定字段；"
            "全量耗时与生成时间只写运行日志（与 variant_eval / robust_eval 同一确定性纪律）"
        ),
        "scope_note": CALIBRATION_SCOPE_NOTE,
        "single_seed_note": (
            "**单 seed 42**；所有 Δ 为**单点差、无跨 seed 极差** —— 本产物不提供任何"
            "训练随机性区间"
        ),
        "train_scorer_rule": TRAIN_SCORER_RULE,
        "anchor_rule": TRAIN_ANCHOR_RULE,
        "harmlessness_rule": TRAIN_HARMLESSNESS_RULE,
        "selection_rule": CALIBRATE_SELECTION_RULE,
        "attribution_rule": ATTRIBUTION_RULE,
        "retrieval_rule": RETRIEVAL_RULE,
        "granularity": TRAIN_GRANULARITY_NOTE,
        "profiles": [str(p) for p in profiles],
        "grid": {
            "scorers": [str(c.train_scorer) for c in combo_list],
            "combos": [c.label for c in combo_list],
            "n_combos": int(len(combo_list)),
            "order_rule": "scorer 外层、lr 中层、weight_decay 内层（确定性、可单组合复跑）",
            "single_combo_rerun": (
                "`variantb calibrate --grid-combo \"<scorer>:<lr>:<wd>\"` 可只跑一个组合；"
                "`--dry-run` 等价于只跑网格的第一个组合"
            ),
        },
        "per_profile": {},
    }
    for profile in [str(p) for p in profiles]:
        data = load_profile_data(str(profile), product_dir=str(product_dir), log=_log)
        # 666 查询行侧的 1a 对账素材（取第一组合的评测网格；KNN 侧与模型无关）
        first_eval_cells: Optional[List[Dict[str, Any]]] = None
        knn_side_seen: List[List[float]] = []
        if skip_identity_gate:
            gate: Dict[str, Any] = {
                "skipped": True,
                "note": "**显式跳过**恒等门禁（--skip-identity-gate）",
            }
        else:
            _log(f"[variantb calibrate] 档 {profile}：训练前恒等门禁 ...")
            gate = identity_gate(data, seed=int(seed), batch_size=int(batch_size))
            _log(
                f"[variantb calibrate]   恒等门禁 {gate['n_cells_all_equal']}/{gate['n_cells']} "
                f"格逐条一致，通过 = {gate['passed']}"
            )
        fresh = build_model(data.table)
        _log(f"[variantb calibrate] 档 {profile}：恒等锚点（T=I × 训练行 {len(data.library_index)}）...")
        anchor = identity_anchor_grid(fresh, data, seed=int(seed), batch_size=int(batch_size))
        _log(
            f"[variantb calibrate]   恒等锚点 top-1（clean + 9 格）= "
            f"{[round(float(v), 6) for v in anchor['top1_self'].values()]}"
        )
        rv_forward = perturb_retrieval_evidence()
        rv_reverse = perturb_retrieval_reverse_check()
        _log(
            f"[variantb calibrate] 取回路径：正向逐位一致 = {rv_forward['all_bitwise_equal']}；"
            f"反向验证 = {rv_reverse['passed']}"
        )
        per_profile: Dict[str, Any] = {
            "profile": str(profile),
            "data": data.as_dict(),
            "identity_gate": gate,
            "identity_anchor": anchor,
            "perturb_retrieval": {"forward": rv_forward, "reverse": rv_reverse},
            "combos": [],
        }
        report["per_profile"][str(profile)] = per_profile
        for combo in combo_list:
            cfg = TrainConfig(
                seed=int(seed),
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(combo.lr),
                weight_decay=float(combo.weight_decay),
                train_mode="perturb",
                train_scorer=str(combo.train_scorer),
            )
            _log(
                f"[variantb calibrate] 档 {profile} 组合 {combo.label}：训练 "
                f"({cfg.epochs} epoch / batch {cfg.batch_size} / scorer {cfg.train_scorer}) ..."
            )
            model = build_model(data.table)
            before = model.parameter_snapshot()
            t0 = time.time()
            train_info = train_transform(
                model, data.table.keys.numpy(), data.library_index, cfg, log=_log
            )
            up = update_gate(before, model.parameter_snapshot())
            with torch.no_grad():
                grid_eval = cell_grid(
                    model, data, seed=int(seed), batch_size=int(batch_size)
                )
                grid_train = cell_grid(
                    model, data, seed=int(seed), batch_size=int(batch_size),
                    rows=data.library_index,
                )
            harm = train_harmlessness(anchor, grid_train)
            primary = primary_criterion({str(profile): {"cell_grid": grid_eval}})
            # 观测量口径对照（皋陶审查 warning 3 / info 6）：**批内滚动平均**（仅诊断）
            # vs **逐格 R@1**（判定量），两者不是同一观测量，差值现场算出并进产物。
            grid_rows = [r for r in harm["per_cell"] if r.get("applicable")]
            grid_rows_9 = [r for r in grid_rows if r.get("in_criterion")]
            grid_mean_all = (
                float(sum(float(r["post_top1"]) for r in grid_rows) / len(grid_rows))
                if grid_rows
                else 0.0
            )
            grid_mean_9 = (
                float(sum(float(r["post_top1"]) for r in grid_rows_9) / len(grid_rows_9))
                if grid_rows_9
                else 0.0
            )
            running = float(train_info["train_batch_running_top1"])
            # 逐格一致性硬校验：`train_side_cells` 与 `per_cell[].post_top1` 必须逐格相等
            side_by_cell = {
                str(c["cell"]): float(c["variant_b_recall_at_1"])
                for c in _compact_cells(grid_train)
            }
            per_cell_consistent = bool(
                grid_rows
                and all(
                    float(r["post_top1"]) == float(side_by_cell.get(str(r["cell"]), float("nan")))
                    for r in grid_rows
                )
            )
            if not per_cell_consistent:
                raise ValueError(
                    "逐格 R@1 口径不一致：`train_side_cells` 与 `train_harmlessness.per_cell` "
                    "在同一组合上出现不同取值；拒绝在自相矛盾的读数上出判定"
                )
            if first_eval_cells is None:
                # 666 查询行侧的 1a 对账（KNN 与模型无关 ⇒ 任一组合的 KNN 侧逐位相同）
                first_eval_cells = list(grid_eval["cells"])
            knn_side = [_knn_r1_row(c) for c in grid_eval["cells"]]
            knn_side_seen.append(knn_side)
            record: Dict[str, Any] = {
                "combo": combo.as_dict(),
                "cfg": cfg.as_dict(),
                "train": {
                    "train_batch_running_top1": float(running),
                    "train_batch_running_top1_rule": TRAIN_TOP1_SELF_RULE,
                    "running_vs_grid": {
                        "rule": (
                            "**仅诊断的对照**：`train_batch_running_top1`（训练循环内的批内滚动平均）"
                            "与训练行 1999 上的**逐格 R@1** 不是同一观测量；本块现场给出两者的数值差，"
                            "避免读者把两者直接相减或互相代入。"
                        ),
                        "train_batch_running_top1": float(running),
                        "grid_r1_mean_9_perturb_cells": float(grid_mean_9),
                        "grid_r1_mean_10_cells_incl_clean": float(grid_mean_all),
                        "delta_running_minus_grid_mean": float(running - grid_mean_9),
                        "per_cell_consistent": bool(per_cell_consistent),
                        "per_cell_consistency_rule": (
                            "`train_side_cells[].variant_b_recall_at_1` 与 "
                            "`train_harmlessness.per_cell[].post_top1` 必须逐格相等（现场已硬校验，"
                            "不一致即抛 ValueError 拒绝出数）"
                        ),
                    },
                    "n_steps": int(train_info["n_steps"]),
                    "n_train_rows": int(train_info["n_train_rows"]),
                    "zero_norm_rows": dict(train_info["zero_norm_rows"]),
                    "norm_diagnostics": list(train_info["norm_diagnostics"]),
                    "norm_diagnostics_summary": dict(train_info["norm_diagnostics_summary"]),
                    "history_tail": list(train_info["history"][-3:]),
                    "train_scorer": str(train_info["train_scorer"]),
                    "train_perturb_note": str(train_info["train_perturb_note"]),
                    "rng_note": str(train_info["rng_note"]),
                },
                "update_gate": up,
                "train_side_cells": _compact_cells(grid_train),
                "eval_side_cells": _compact_cells(grid_eval),
                # W1 修复的**同产物现场证据**：该块字段名是 `top1_discrepancy_vs_knn`（**不是**
                # `identity_gate`），并显式标注 `computed_after_training` —— 于是同一产物里
                # `per_profile.<档>.identity_gate.passed = True`（训练前的硬门禁）与该块的
                # `all_cells_equal = False`（训练后的**统计量**）**不同名**，不再出现
                # 「同名字段相反结论」。
                "cell_grid_discrepancy": {
                    "eval_rows": dict(grid_eval.get(TOP1_DISCREPANCY_KEY, {})),
                    "train_rows": dict(grid_train.get(TOP1_DISCREPANCY_KEY, {})),
                    "field_name": TOP1_DISCREPANCY_KEY,
                    "rename_note": (
                        "W1 修复：本块此前的字段名与门禁同名（`identity_gate`），"
                        "训练后被复用时会与顶层 `identity_gate.passed` 形成「同名相反结论」；"
                        "现改名并显式标注 `model_is_identity` / `computed_after_training`。"
                    ),
                },
                "train_harmlessness": harm,
                "eval_primary": primary,
                "attribution": attribution_verdict(combo, harm, primary),
                "seconds_note": "耗时只进日志，不入产物（确定性纪律）",
            }
            per_profile["combos"].append(record)
            _log(
                f"[variantb calibrate] 档 {profile} 组合 {combo.label} 完成："
                f"批内滚动平均（仅诊断）="
                f"{record['train']['train_batch_running_top1']:.6f} / "
                f"逐格 R@1 均值（9 格，判定量）={grid_mean_9:.6f} / "
                f"训练无害 {harm['n_not_worse']}/{harm['n_cells']} "
                f"(min margin {harm['min_margin']:+.6f}) / "
                f"评测侧不劣 {primary['n_not_worse']}/{primary['n_cells']} / "
                f"‖T(x)‖ 末值 {record['train']['norm_diagnostics_summary']['mean_row_norm_Tx_last']:.4f} / "
                f"耗时 {time.time() - t0:.1f}s（只进日志）"
            )
            if callable(on_progress):
                on_progress(report)
        # --- 666 查询行侧的 1a 对账（皋陶审查 info 5）：`calibrate` 也产出 `anchor_1a_check` ---
        # KNN 路径与模型无关 ⇒ 任一组合的 KNN 侧逐位相同；此处把它作为**现场不变量**登记。
        knn_unique = {tuple(rows) for rows in knn_side_seen}
        per_profile["anchor_1a_check_row_source"] = (
            "**查询行 666 侧**（与 1a §15.5 登记值同口径）；训练行 1999 侧的锚点表"
            "（`identity_anchor.n` 均为 1999）**不可**与它对账 —— 见 "
            + ANCHOR_ROW_SET_NOTE
        )
        per_profile["knn_invariance_across_combos"] = {
            "rule": (
                "KNN 基线 = 冻结键表上的归一化余弦，**与训练完全无关** ⇒ 各组合的查询行 666 侧 "
                "KNN R@1 必须逐位相同；本字段现场核验该不变量"
            ),
            "n_combos": int(len(knn_side_seen)),
            "n_distinct_knn_vectors": int(len(knn_unique)),
            "all_equal": bool(len(knn_unique) <= 1),
        }
        if first_eval_cells is not None:
            per_profile["anchor_1a_check"] = anchor_1a_check(
                {
                    # `cell_grid` 的逐格字典不带 `profile`（那是由 `_flatten_cells` 补的），
                    # 这里按 `anchor_1a_check` 的输入契约显式补上。
                    "cells": [
                        dict(c, profile=str(profile), knn=dict(c["knn"]))
                        for c in first_eval_cells
                    ]
                }
            )
            per_profile["anchor_1a_check"]["row_source"] = (
                "查询行 666 侧（本档任一组合的 KNN 侧逐位相同 ⇒ 取第一组合即可）"
            )
            _log(
                f"[variantb calibrate] 档 {profile}：1a 基线对账（查询行 666 侧）在容差内 "
                f"{per_profile['anchor_1a_check']['n_within_tol']}/"
                f"{per_profile['anchor_1a_check']['n_rows']}；KNN 组合间不变量 = "
                f"{per_profile['knn_invariance_across_combos']['all_equal']}"
            )
    # 选取最优超参 + 归因（逐档各一份，另给一个「全档合并」的结论）
    by_label = {c.label: c for c in combo_list}
    selection: Dict[str, Any] = {}
    attribution: Dict[str, Any] = {}
    for profile, per in report["per_profile"].items():
        sel = select_best_combo(per["combos"])
        if not bool(sel.get("applicable", False)):
            # R59 F2：网格里没有 normalized 组合 ⇒ **如实标「不适用」并保留全部扫描结果**，
            # 不抛异常、不丢产物（改前会退码 2 并丢弃整轮训练成果）。
            selection[str(profile)] = {
                "rule": sel["rule"],
                "applicable": False,
                "selected": "",
                "reason": sel["reason"],
                "candidates": sel["candidates"],
                "candidates_ranked": [],
                "ineligible_excluded": sel["ineligible_excluded"],
            }
            attribution[str(profile)] = attribution_not_applicable(
                str(sel["reason"]), sel["candidates"]
            )
            _log(
                f"[variantb calibrate] 档 {profile}：本次扫描无 "
                f"`{DEFAULT_TRAIN_SCORER}` 组合 => 归因**不适用**（扫描结果全部保留）"
            )
            continue
        selection[str(profile)] = {
            "rule": sel["rule"],
            "applicable": True,
            "selected": sel["selected"],
            "reason": "",
            "candidates": sel["candidates"],
            "candidates_ranked": sel["candidates_ranked"],
            "ineligible_excluded": sel["ineligible_excluded"],
        }
        best = dict(sel["selected_record"])
        chosen_combo = by_label[str(sel["selected"])]
        attribution[str(profile)] = attribution_verdict(
            chosen_combo,
            dict(best["train_harmlessness"]),
            dict(best["eval_primary"]),
        )
    keys = sorted(attribution)
    per_profile_verdict = {str(k): str(attribution[k]["verdict_key"]) for k in keys}
    same = bool(len(set(per_profile_verdict.values())) <= 1)
    if keys and same:
        verdict_key = str(per_profile_verdict[keys[0]])
    elif keys:
        verdict_key = "mixed_per_profile"
    else:
        verdict_key = "mixed_per_profile"
    report["selection"] = selection
    report["attribution"] = attribution
    report["attribution_summary"] = {
        "rule": ATTRIBUTION_RULE,
        "per_profile_verdict": per_profile_verdict,
        "all_profiles_same_verdict": bool(same),
        "verdict_key": str(verdict_key),
        "verdict": str(ATTRIBUTION_VERDICTS[verdict_key]),
        "single_source_note": (
            "归因结论的唯一来源 = 本产物 `selection.<档>.selected` 指向的那个组合的 "
            "`train_harmlessness` 与 `eval_primary`；规则文本与判定同源（由 "
            "`ATTRIBUTION_REQUIRED_CONDITIONS` 派生）"
        ),
    }
    report["honest_notes"] = _calibration_honest_notes(report)
    report["status"] = "complete"
    report["artifact_fingerprint"] = artifact_fingerprint(report)
    return report


def _calibration_honest_notes(report: Dict[str, Any]) -> List[str]:
    """校准轮的如实登记（**与实测值绑定**；禁止把「没测出来」写成结论）。"""
    notes: List[str] = [
        "**诊断轮口径**：本轮**不以提升指标为成功标准**；主判据是「训练无害下限」"
        "（训练后在训练行 1999 上的 **9 个扰动格** top-1 逐格不低于恒等锚点）。",
        TRAIN_GRANULARITY_NOTE,
        f"**判据排除 `clean` 格**（唯一来源 `TRAIN_HARMLESSNESS_EXCLUDED_CELLS = "
        f"{list(TRAIN_HARMLESSNESS_EXCLUDED_CELLS)}`）：该格是逐位自匹配、锚点结构性 1.000000，"
        "任何非恒等变换只能变差；被排除的格仍在锚点表与逐格表里如实报出。",
        "**单 seed 42**：全部数字为单点值，无跨 seed 极差。",
        "**1a 考卷零改动**：`entry_table.py` / `robust_eval.py` 未被修改；"
        "扰动走 `robust_eval.perturb_matrix` / `derived_seed` / `PERTURB_GRID`，"
        "归一化走 `entry_table.l2_normalize_rows`。",
    ]
    for profile, per in sorted(report.get("per_profile", {}).items()):
        gate = dict(per.get("identity_gate", {}))
        if gate.get("skipped"):
            notes.append(f"**{profile}**：恒等门禁被**显式跳过**。")
        else:
            notes.append(
                f"**{profile}**：恒等门禁 {gate.get('n_cells_all_equal')}/"
                f"{gate.get('n_cells')} 格逐条一致，通过 = {gate.get('passed')}；"
                f"锚点与 KNN 逐格一致的格数 = "
                f"{dict(per.get('identity_anchor', {}).get('identity_equivalence', {})).get('n_cells_all_equal_vs_knn')}"
                f"/{dict(per.get('identity_anchor', {}).get('identity_equivalence', {})).get('n_cells')}。"
            )
        combos = list(per.get("combos", []))
        if combos:
            worst = min(combos, key=lambda r: float(r["train_harmlessness"]["min_margin"]))
            best = max(combos, key=lambda r: float(r["train_harmlessness"]["min_margin"]))
            notes.append(
                f"**{profile} 训练无害（{len(combos)} 个组合）**：最好组合 "
                f"`{best['combo']['label']}` 的不劣格数 "
                f"{best['train_harmlessness']['n_not_worse']}/{best['train_harmlessness']['n_cells']}"
                f"（最差余量 {best['train_harmlessness']['min_margin']:+.6f}）；最差组合 "
                f"`{worst['combo']['label']}` 的不劣格数 "
                f"{worst['train_harmlessness']['n_not_worse']}/{worst['train_harmlessness']['n_cells']}"
                f"（最差余量 {worst['train_harmlessness']['min_margin']:+.6f}）。"
            )
            raw = [r for r in combos if str(r["combo"]["train_scorer"]) == "raw"]
            norm = [r for r in combos if str(r["combo"]["train_scorer"]) == DEFAULT_TRAIN_SCORER]
            if raw and norm:
                notes.append(
                    "**口径对齐前后（`raw` → `normalized`，同 lr / wd 配对）—— 两个观测量分开报**："
                    "① **判定量：逐格 R@1（训练行 1999，9 个扰动格）**，各组合均值 —— "
                    f"`raw` = "
                    f"{[round(float(r['train_harmlessness']['aggregate_reading']['mean_post_top1']), 6) for r in raw]}"
                    " vs `normalized` = "
                    f"{[round(float(r['train_harmlessness']['aggregate_reading']['mean_post_top1']), 6) for r in norm]}；"
                    "② **仅诊断量：批内滚动平均** `train_batch_running_top1`"
                    "（口径见 :data:`TRAIN_TOP1_SELF_RULE` 与产物 `train_batch_running_top1_rule`）"
                    "—— `raw` 区间 "
                    f"[{min(float(r['train']['train_batch_running_top1']) for r in raw):.6f}, "
                    f"{max(float(r['train']['train_batch_running_top1']) for r in raw):.6f}] vs "
                    f"`normalized` 区间 "
                    f"[{min(float(r['train']['train_batch_running_top1']) for r in norm):.6f}, "
                    f"{max(float(r['train']['train_batch_running_top1']) for r in norm):.6f}]；"
                    "训练无害格数 "
                    f"`raw` = {[int(r['train_harmlessness']['n_not_worse']) for r in raw]} vs "
                    f"`normalized` = {[int(r['train_harmlessness']['n_not_worse']) for r in norm]}。"
                    "**两个观测量不可互相代入**：现场同一组合的差值登记在 "
                    "`train.running_vs_grid.delta_running_minus_grid_mean`。"
                )
                notes.append(
                    "**范数膨胀通道的现场读数**（只作机制取证，不参与判定）：`raw` 组合的 "
                    "`‖T(x)‖ 末值/首值` = "
                    f"{[round(float(r['train']['norm_diagnostics_summary']['norm_ratio_last_over_first']), 4) for r in raw]}"
                    " vs `normalized` = "
                    f"{[round(float(r['train']['norm_diagnostics_summary']['norm_ratio_last_over_first']), 4) for r in norm]}。"
                )
        rv = dict(per.get("perturb_retrieval", {}).get("reverse", {}))
        if rv:
            notes.append(
                f"**{profile} 取回路径反向验证**：{[(c['case'], c['detected']) for c in rv.get('cases', [])]}；"
                f"基线通过 = {rv.get('baseline_passed')}，恢复后通过 = {rv.get('restored_passed')}，"
                f"总判定 = {rv.get('passed')}。"
            )
    att = dict(report.get("attribution_summary", {}))
    if att:
        notes.append(
            f"**归因结论（机器可读，单点来源）**：`{att.get('per_profile_verdict')}` ⇒ "
            f"{att.get('verdict')}"
        )
    # 逐格机制取证：哪些格被牺牲、哪些格被换来（**由现场数字生成，不写死**）
    for profile, per in sorted(report.get("per_profile", {}).items()):
        best_label = str(dict(report.get("selection", {}).get(profile, {})).get("selected"))
        rec = next(
            (r for r in per.get("combos", []) if str(r["combo"]["label"]) == best_label), None
        )
        if rec is None:
            continue
        rows = [r for r in rec["train_harmlessness"]["per_cell"] if r.get("in_criterion")]
        up = [r for r in rows if float(r["margin"]) > 0.0]
        down = [r for r in rows if float(r["margin"]) < 0.0]
        notes.append(
            f"**{profile} 最优组合 `{best_label}` 的逐格取舍（机制取证）**："
            f"该组合的 `train_scorer` = `{rec['combo']['train_scorer']}`；变好的格 = "
            f"{[(str(r['cell']), round(float(r['margin']), 6)) for r in up]}；变差的格 = "
            f"{[(str(r['cell']), round(float(r['margin']), 6)) for r in down]}。"
            "**取舍两端的格族（现场数字读出）**：变好的格全部是 `nmag` 三格（锚点最低、被"
            "「救回」到 1.0 附近），变差的格全部是 `noise` / `mask` 六格（锚点最高、原本接近满分）；"
            "本组合下锚点最低的三格与最高的三格现场值分别为 "
            f"{sorted((round(float(r['anchor_top1']), 6), str(r['cell'])) for r in rows)[:3]} 与 "
            f"{sorted((round(float(r['anchor_top1']), 6), str(r['cell'])) for r in rows)[-3:]}。"
            "**据此可读出（仅限本组合）**：等权目标下低锚点格（交叉熵大）主导梯度方向、"
            "高锚点格被牺牲，即**取舍发生在 `nmag` 与 `noise`/`mask` 之间**"
            "（**不是**「弱扰动格被强扰动格牺牲」—— 现场恰恰相反：`nmag` 三档被救回、"
            "`noise`/`mask` 六档被牺牲）—— 该读法属机制登记，不改变归因判定。"
            "**该取舍不是 `raw` 口径的专属现象**：`raw` 组合的取舍见下一条（各自独立读出）。"
        )
        # `raw` 组合自己的读数（**只用于 `raw` 组合，不得挪用为 normalized 的结论**）
        raw_recs = [r for r in per.get("combos", []) if str(r["combo"]["train_scorer"]) == "raw"]
        if raw_recs:
            raw_best = max(
                raw_recs, key=lambda r: float(r["train_harmlessness"]["min_margin"])
            )
            raw_rows = [
                r for r in raw_best["train_harmlessness"]["per_cell"] if r.get("in_criterion")
            ]
            raw_up = [r for r in raw_rows if float(r["margin"]) > 0.0]
            raw_down = [r for r in raw_rows if float(r["margin"]) < 0.0]
            notes.append(
                f"**{profile} `raw` 口径组合的自行取舍（`raw` 组合自己的读数，"
                f"**不得**挪用为 `normalized` 的结论）**：以 `min_margin` 最优的 `raw` 组合 "
                f"`{raw_best['combo']['label']}` 为例：变好的格 = "
                f"{[(str(r['cell']), round(float(r['margin']), 6)) for r in raw_up]}；变差的格 = "
                f"{[(str(r['cell']), round(float(r['margin']), 6)) for r in raw_down]}。"
                "`raw` 口径的「训练在自己目标上失败」（批内滚动平均大幅低于 1）**主要来自"
                "打分口径本身**（未归一化 ⇒ 放大 `‖T(x)‖` 可单调压低交叉熵，见范数膨胀读数），"
                "而**不是**先由格间取舍造成 —— 该判断只针对 `raw` 组合。"
            )
        alt = dict(rec["eval_primary"])
        notes.append(
            f"**{profile} 备选读法（整段平均，不参与判定）**：训练侧 `mean_margin` = "
            f"{rec['train_harmlessness']['aggregate_reading']['mean_margin']:+.6f}"
            f"（平均 post {rec['train_harmlessness']['aggregate_reading']['mean_post_top1']:.6f} "
            f"vs 平均锚点 {rec['train_harmlessness']['aggregate_reading']['mean_anchor_top1']:.6f}）；"
            f"评测侧平均 R@1 = "
            f"{float(sum(float(r['variant_b']) for r in alt['per_cell'] if r.get('applicable')) / max(1, sum(1 for r in alt['per_cell'] if r.get('applicable')))):.6f}"
            f" vs 平均 KNN = "
            f"{float(sum(float(r['knn']) for r in alt['per_cell'] if r.get('applicable')) / max(1, sum(1 for r in alt['per_cell'] if r.get('applicable')))):.6f}。"
            "判据口径是**逐格**（与第二步 `primary_criterion` 的「均不低于」同一读法），"
            "平均读法只作如实登记。"
        )
    notes.append(
        "**未做项（如实登记）**：① 本轮只跑 `lexical-88`（诊断轮成本优先），"
        "`bge-m3-1024` 未跑，故其归因待补；② 单 seed，无跨 seed 极差；"
        "③ 不落盘特征矩阵、不产 zip 产物；④ 未做对齐 / SupCon 类辅助损失（口径排除）。"
    )
    return notes


# ---------------------------------------------------------------------------
# 8. 全量运行 / 渲染 / 落盘 / 指纹
# ---------------------------------------------------------------------------


def anchor_1a_check(report: Dict[str, Any]) -> Dict[str, Any]:
    """现场重算的 KNN 基线 vs 1a 登记值的**逐格对账**（旁证，不作门禁）。"""
    rows: List[Dict[str, Any]] = []
    for c in report.get("cells", []):
        profile = str(c["profile"])
        ref = ROBUST_1A_KNN_REFERENCE.get(profile, {}).get(str(c["kind"]), {})
        want = ref.get(str(c["level"]))
        if want is None:
            continue
        got = float(c["knn"]["recall_at_1"])
        rows.append(
            {
                "profile": profile,
                "cell": str(c["cell"]),
                "knn_1a_registered": float(want),
                "knn_recomputed": float(got),
                "abs_diff": float(abs(got - float(want))),
                "within_tol": bool(abs(got - float(want)) <= float(ANCHOR_TOL)),
            }
        )
    return {
        "rule": ANCHOR_RULE,
        "tolerance": float(ANCHOR_TOL),
        "rows": rows,
        "n_rows": int(len(rows)),
        "n_within_tol": int(sum(1 for r in rows if r["within_tol"])),
        "all_within_tol": bool(rows and all(r["within_tol"] for r in rows)),
    }


def artifact_fingerprint(report: Dict[str, Any]) -> str:
    """报告的**内容指纹**（规范化 JSON 的 SHA256；不含任何时间字段）。"""
    return ET.sha256_bytes(ET.canonical_dumps(report))


def run_variant_b(
    *,
    profiles: Sequence[str],
    cfg: Optional[TrainConfig] = None,
    product_dir: str = "",
    include_clean_control: bool = True,
    log: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, torch.Tensor]]]:
    """全量运行：逐档「门禁 → 扰动自监督训练 → 逐格对照」+ A 组 + 决定性对照臂。

    产物**不含挂钟时间 / 耗时**（确定性纪律）；耗时只进日志。

    返回
    ----
    Tuple[Dict[str, Any], Dict[str, Dict[str, torch.Tensor]]]
        ``(报告, {档名: 主臂训练后的 transform.state_dict()})``。权重**不进产物**
        （JSON 无法序列化张量，且产物须为纯 JSON + 确定性），只在进程内供决定性对照臂
        复用同一个模型（由 ``evidence.decisive_arm_weight_rule`` 记录该口径）。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    cfg = cfg or TrainConfig()
    t0 = time.time()
    per_profile: Dict[str, Dict[str, Any]] = {}
    perturb_weights: Dict[str, Dict[str, torch.Tensor]] = {}
    for profile in profiles:
        per_profile[str(profile)] = train_and_eval_profile(
            str(profile), cfg=cfg, product_dir=str(product_dir), log=_log
        )
        perturb_weights[str(profile)] = dict(per_profile[str(profile)].pop("_weights"))
    arms: Dict[str, Any] = {}
    if include_clean_control:
        clean_cfg = TrainConfig(
            seed=int(cfg.seed),
            epochs=int(cfg.epochs),
            batch_size=int(cfg.batch_size),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            train_mode="clean",
            train_scorer=str(cfg.train_scorer),
            log_every=int(cfg.log_every),
        )
        for profile in profiles:
            _log(f"[variantb arm-A] 档 {profile}：干净自监督（A 组对照档）...")
            arms[str(profile)] = train_and_eval_profile(
                str(profile), cfg=clean_cfg, product_dir=str(product_dir), log=_log
            )
            arms[str(profile)].pop("_weights", None)
    # **决定性对照臂**：扰动自监督训练的模型 × clean 评测（把「T 本身」与「扰动泛化落差」分开）
    clean_eval_arm: Dict[str, Any] = {}
    for profile in profiles:
        _log(
            f"[variantb arm-clean-eval] 档 {profile}："
            "扰动自监督训练的模型 × **clean 评测**（决定性对照）..."
        )
        clean_eval_arm[str(profile)] = clean_eval_of_perturb_arm(
            str(profile),
            cfg=cfg,
            product_dir=str(product_dir),
            perturb_train=per_profile.get(str(profile)),
            perturb_weights=perturb_weights.get(str(profile)),
            log=_log,
        )
    primary = primary_criterion(per_profile)
    report: Dict[str, Any] = {
        "module": MODULE_NAME,
        "artifact_schema": ARTIFACT_SCHEMA,
        "deterministic": True,
        "excluded_fields_note": (
            "本产物不含 created_utc / seconds / 主机名等非确定字段；"
            "全量耗时与生成时间只写运行日志（与 robust_eval 同一确定性纪律）"
        ),
        "single_seed_note": (
            "**单 seed 42**；所有 Δ 为**单点差、无跨 seed 极差** —— 本报告不提供任何"
            "训练随机性区间，Δ 只表示同一 seed 下变体 B 与 KNN 的单点差"
        ),
        "config": cfg.as_dict(),
        "profiles": [str(p) for p in profiles],
        "per_profile": per_profile,
        "arms": {
            "clean_self_supervised": arms,
            "perturb_train_clean_eval": clean_eval_arm,
        },
        "cells": _flatten_cells(per_profile, arms),
        "primary": primary,
        "honest_notes": _honest_notes(per_profile, arms, primary, clean_eval_arm),
        "evidence": {
            "carrier": "x = **原始编码器特征**（raw），不经 N3D 骨干；考卷路径本身不含任何模型",
            "model": (
                "单层 D→D 变换 T（恒等初始化）+ 冻结特征库内积评分；"
                "即「表内嵌输出层、权重即特征库、冻结」，只学变换层"
            ),
            "loss": (
                "CrossEntropyLoss(logits = T(q) @ keys.T, "
                "label = 行自身在全量键表中的下标)"
            ),
            "split": (
                "训练查询 = 库行 1999（严格排除 666 查询行）；评测查询 = 666 查询行；"
                "键表固定全量 2665 行"
            ),
            "identity_fastpath_rule": IDENTITY_FASTPATH_RULE,
            "train_scorer_rule": TRAIN_SCORER_RULE,
            "anchor_rule": TRAIN_ANCHOR_RULE,
            "granularity": TRAIN_GRANULARITY_NOTE,
            "retrieval_rule": RETRIEVAL_RULE,
            "reuse_1a": (
                "扰动走 robust_eval.perturb_matrix / derived_seed / PERTURB_GRID；"
                "归一化走 entry_table.l2_normalize_rows —— 未自造第二份口径"
            ),
            "knn_reference_1a": ROBUST_1A_KNN_REFERENCE,
            "knn_reference_rule": ANCHOR_RULE,
            "anchor_tolerance": float(ANCHOR_TOL),
            "artifacts": f"一律写 {VARIANT_B_DIR}；不落盘特征矩阵、不产 zip 产物",
        },
    }
    report["third_step"] = third_step_evidence(report)
    report["anchor_1a_check"] = anchor_1a_check(report)
    report["perturb_retrieval"] = {
        "forward": perturb_retrieval_evidence(),
        "reverse": perturb_retrieval_reverse_check(),
    }
    report["evidence"]["decisive_arm_weight_rule"] = (
        "决定性对照臂（扰动训练 × clean 评测）复用**主臂同一个训练后模型**"
        "（进程内直接传 `transform.state_dict()`，不做任何重放/重建）；"
        "权重为张量、**不进产物**（产物保持纯 JSON + 确定性）"
    )
    report["artifact_fingerprint"] = artifact_fingerprint(report)
    _log(f"[variantb] 全量完成（耗时 {time.time() - t0:.1f}s，该耗时只进日志不入产物）")
    return report, perturb_weights


def render_markdown(report: Dict[str, Any]) -> str:
    """把报告渲染为 Markdown（逐格表 + 门禁 + 主判据 + 诚实登记）。"""
    cfg = dict(report.get("config", {}))
    lines: List[str] = []
    lines.append("# n3d_qa_learn 变体 B：可学 D→D 变换 + 冻结特征库（第二步）")
    lines.append("")
    lines.append(f"- **单 seed 声明**：{report.get('single_seed_note')}")
    lines.append(
        "- **本报告不含生成时间 / 耗时**（确定性纪律）：同参数重跑的产物应逐字节一致。"
    )
    lines.append(
        f"- 训练超参：`epochs={cfg.get('epochs')}` / `batch_size={cfg.get('batch_size')}` / "
        f"`lr={cfg.get('lr')}` / `weight_decay={cfg.get('weight_decay')}` / "
        f"`train_mode={cfg.get('train_mode')}` / `train_scorer={cfg.get('train_scorer')}`"
    )
    ev = dict(report.get("evidence", {}))
    lines.append(f"- 载体：{ev.get('carrier')}")
    lines.append(f"- 模型：{ev.get('model')}")
    lines.append(f"- 划分：{ev.get('split')}")
    lines.append("")
    lines.append("## 1. 恒等门禁（硬门禁）")
    lines.append("")
    lines.append("| 档 | 逐条一致格数 | 范数逐位相等比例 | 显式路径逐位相同 | 通过 |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for profile, res in sorted(report.get("per_profile", {}).items()):
        gate = dict(res.get("identity_gate", {}))
        if gate.get("skipped"):
            lines.append(f"| `{profile}` | — | — | — | 跳过 |")
            continue
        bw = dict(gate.get("bitwise_norm_equality", {}))
        ex = dict(gate.get("explicit_path", {}))
        lines.append(
            f"| `{profile}` | {gate.get('n_cells_all_equal')}/{gate.get('n_cells')} | "
            f"{bw.get('bitwise_equal_frac')} | {ex.get('bitwise_equal')} | "
            f"**{gate.get('passed')}** |"
        )
    lines.append("")
    lines.append("## 2. 逐格 变体 B vs KNN（R@1 / @5 / Δ）")
    lines.append("")
    lines.append(
        "| 档 | 格 | 1a 有效性 | KNN R@1 | 变体 B R@1 | ΔR@1 | Δ条数 | "
        "KNN R@5 | 变体 B R@5 | A 组 R@1 | A 组 ΔR@1 |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c in report.get("cells", []):
        if c.get("effective_1a"):
            eff = "有效"
        elif c.get("ineffective_1a"):
            eff = "**无效**"
        else:
            eff = "—"
        arm = dict(c.get("arm_a_clean_train") or {})
        arm_r1 = f"{float(arm['recall_at_1']):.6f}" if arm else "—"
        arm_d = f"{float(arm['delta_vs_knn']):+.6f}" if arm else "—"
        lines.append(
            f"| `{c['profile']}` | `{c['cell']}` | {eff} | "
            f"{float(c['knn']['recall_at_1']):.6f} | {float(c['variant_b']['recall_at_1']):.6f} | "
            f"**{float(c['delta']['recall_at_1']):+.6f}** | {int(c['delta']['hit_at_1']):+d} | "
            f"{float(c['knn']['recall_at_5']):.6f} | {float(c['variant_b']['recall_at_5']):.6f} | "
            f"{arm_r1} | {arm_d} |"
        )
    lines.append("")
    lines.append("## 3. 分档落差（弱 − 强）的变化")
    lines.append("")
    lines.append("| 档 | 扰动 | KNN 落差 | 变体 B 落差 | Δ落差 |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for profile, res in sorted(report.get("per_profile", {}).items()):
        for g in res["cell_grid"]["gap_weak_minus_strong"]:
            lines.append(
                f"| `{profile}` | `{g['kind']}` | {float(g['knn_gap']):.6f} | "
                f"{float(g['variant_b_gap']):.6f} | **{float(g['delta_gap']):+.6f}** |"
            )
    lines.append("")
    lines.append("## 4. 主判据")
    lines.append("")
    primary = dict(report.get("primary", {}))
    lines.append(f"- 判据：{primary.get('criterion')}")
    lines.append(
        f"- 结果：**通过 = {primary.get('passed')}**；格数 {primary.get('n_cells')} / "
        f"不劣于 KNN {primary.get('n_not_worse')} / 严格更高 {primary.get('n_strictly_better')} / "
        f"严格更低 {primary.get('n_strictly_worse')}"
    )
    lines.append(f"- 方向说明：{primary.get('naive_direction')}")
    lines.append("")
    lines.append("| 档 | 格 | KNN R@1 | 变体 B R@1 | Δ | 命中(变体B/KNN) | 不劣于 | 严格更高 |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- | --- |")
    for r in primary.get("per_cell", []):
        if not r.get("applicable"):
            lines.append(
                f"| `{r['profile']}` | `{r['cell']}` | — | — | — | — | 不适用 | 不适用 |"
            )
            continue
        lines.append(
            f"| `{r['profile']}` | `{r['cell']}` | {float(r['knn']):.6f} | "
            f"{float(r['variant_b']):.6f} | **{float(r['delta']):+.6f}** | "
            f"{int(r['variant_b_hit'])}/{int(r['knn_hit'])} | "
            f"{r['not_worse']} | {r['strictly_better']} |"
        )
    lines.append("")
    lines.append("## 5. 可训参数更新量门禁")
    lines.append("")
    lines.append("| 档 | 更新参数数 | 总参数数 | 从未更新 | 通过 |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for profile, res in sorted(report.get("per_profile", {}).items()):
        up = dict(res["update_gate"])
        lines.append(
            f"| `{profile}` | {up['n_updated']} | {up['n_total']} | "
            f"{up['n_never_updated']} | **{up['passed']}** |"
        )
    lines.append("")
    lines.append("## 6. 空间不足格（显式标注）")
    lines.append("")
    lines.append("| 档 | 格 | KNN 命中 | 错误条数 | 查询数 |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for profile, res in sorted(report.get("per_profile", {}).items()):
        for e in res["cell_grid"]["error_space"]:
            lines.append(
                f"| `{profile}` | `{e['cell']}` | {int(e['knn_hit_at_1'])} | "
                f"{int(e['knn_errors'])} | {int(e['n'])} |"
            )
    lines.append("")
    lines.append("## 7. 决定性对照臂（扰动训练 × clean 评测）")
    lines.append("")
    lines.append("| 档 | KNN R@1 | 变体 B R@1 | ΔR@1 | 命中(变体B/KNN) |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for profile, res in sorted(
        dict(report.get("arms", {}).get("perturb_train_clean_eval", {})).items()
    ):
        c = dict(res["clean_cells"])
        lines.append(
            f"| `{profile}` | {float(c['knn']['recall_at_1']):.6f} | "
            f"{float(c['variant_b']['recall_at_1']):.6f} | "
            f"**{float(c['delta']['recall_at_1']):+.6f}** | "
            f"{int(c['variant_b']['hit_at_1'])}/{int(c['knn']['hit_at_1'])} |"
        )
    lines.append("")
    lines.append("该臂**评测查询不扰动**，故 Δ 只反映「训练出的变换 T 本身」的方向偏移：")
    lines.append("Δ 显著为负即说明掉点来自**变换层本身**，而不是「训练扰动 ≠ 评测扰动」。")
    lines.append("")
    lines.append("## 8. 1a 基线对账（旁证）")
    lines.append("")
    anchor = dict(report.get("anchor_1a_check", {}))
    lines.append(
        f"- 容差 `{anchor.get('tolerance')}`；在容差内 "
        f"{anchor.get('n_within_tol')}/{anchor.get('n_rows')}，"
        f"全部在容差内 = **{anchor.get('all_within_tol')}**"
    )
    lines.append("")
    lines.append("| 档 | 格 | 1a 登记 | 现场重算 | 绝对差 | 在容差内 |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for r in anchor.get("rows", []):
        lines.append(
            f"| `{r['profile']}` | `{r['cell']}` | {float(r['knn_1a_registered']):.6f} | "
            f"{float(r['knn_recomputed']):.6f} | {float(r['abs_diff']):.2e} | {r['within_tol']} |"
        )
    lines.append("")
    lines.append("## 9. 第三步方向依据（否证条款要求）")
    lines.append("")
    third = dict(report.get("third_step", {}))
    lines.append(f"- Δ 直方图：{third.get('delta_histogram')}")
    lines.append(f"- 训练侧自命中 top-1：{third.get('train_self_top1')}")
    lines.append(f"- A 组（干净训练）Δ 的绝对最大值：{third.get('arm_a_clean_train_delta_abs_max')}")
    lines.append(f"- 读法：{third.get('reading')}")
    lines.append("")
    lines.append("## 10. 如实登记（不得包装）")
    lines.append("")
    for i, note in enumerate(report.get("honest_notes", []), start=1):
        lines.append(f"{i}. {note}")
    lines.append("")
    _render_anchor_sections(report, lines)
    _render_retrieval_section(report, lines)
    return "\n".join(lines)


def _render_anchor_sections(report: Dict[str, Any], lines: List[str]) -> None:
    """渲染「恒等参照锚点 + 训练无害下限」（训练前提校准轮新增；**缺字段即整节省略**）。"""
    have = [
        (p, r) for p, r in sorted(report.get("per_profile", {}).items())
        if r.get("identity_anchor") and r.get("train_harmlessness")
    ]
    if not have:
        return
    lines.append("## 11. 恒等参照锚点与训练无害下限（训练前提校准轮）")
    lines.append("")
    lines.append(f"- 锚点口径：{dict(have[0][1]['identity_anchor']).get('rule')}")
    lines.append(f"- 行集合：{dict(have[0][1]['identity_anchor']).get('row_set_note')}")
    lines.append(f"- 观测量：{dict(have[0][1]['identity_anchor']).get('observable')}")
    lines.append(f"- 粒度：{TRAIN_GRANULARITY_NOTE}")
    lines.append("")
    lines.append("| 档 | 格 | 恒等锚点 R@1（T=I，训练行 1999） | 训练后 R@1（同口径，**判定量**） | 余量 | 不低于锚点 |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for profile, res in have:
        for row in dict(res.get("train_harmlessness") or {}).get("per_cell", []):
            if not row.get("applicable"):
                lines.append(f"| `{profile}` | `{row['cell']}` | — | — | — | 不适用 |")
                continue
            lines.append(
                f"| `{profile}` | `{row['cell']}` | {float(row['anchor_top1']):.6f} | "
                f"{float(row['post_top1']):.6f} | **{float(row['margin']):+.6f}** | "
                f"{row['not_worse']} |"
            )
    lines.append("")
    for profile, res in have:
        harm = dict(res["train_harmlessness"])
        nd = dict(dict(res.get("train") or {}).get("norm_diagnostics_summary") or {})
        lines.append(
            f"- `{profile}`：训练无害 **{harm['all_not_worse']}** —— 不劣 "
            f"{harm['n_not_worse']}/{harm['n_cells']}，最差余量 {harm['min_margin']:+.6f}，"
            f"平均余量 {harm['mean_margin']:+.6f}；"
            f"锚点与 KNN 逐格一致 "
            f"{dict(res['identity_anchor'].get('identity_equivalence', {})).get('n_cells_all_equal_vs_knn')}/"
            f"{dict(res['identity_anchor'].get('identity_equivalence', {})).get('n_cells')}。"
        )
        if nd:
            lines.append(
                f"  - 范数膨胀通道（机制取证，不参与判定）：`‖T(x)‖` 末/首 = "
                f"{nd.get('norm_ratio_last_over_first')}、`‖W−I‖_F` 末值 = "
                f"{nd.get('weight_minus_identity_fro_last')}、`mean|logit|` 首→末 = "
                f"{nd.get('mean_logit_abs_first')} → {nd.get('mean_logit_abs_last')}。"
            )
    lines.append("")


def _render_retrieval_section(report: Dict[str, Any], lines: List[str]) -> None:
    """渲染「取回路径（W2 修复）的正向证据 + 反向验证」；缺字段即整节省略。"""
    rv = dict(report.get("perturb_retrieval") or {})
    if not rv:
        return
    fwd = dict(rv.get("forward") or {})
    rev = dict(rv.get("reverse") or {})
    lines.append("## 12. 扰动取回路径（W2 修复）与反向验证")
    lines.append("")
    lines.append(f"- 口径：{fwd.get('rule')}")
    lines.append(
        f"- 正向证据：成功分支逐位一致 = "
        f"{dict(fwd.get('success_branch') or {}).get('bitwise_equal_vs_manual')}；"
        f"修复分支（全零行处置）逐位一致 = "
        f"{dict(fwd.get('repair_branch') or {}).get('bitwise_equal_vs_manual_on_kept_rows')}，"
        f"被替换行置零 = {dict(fwd.get('repair_branch') or {}).get('repaired_rows_are_zero')}；"
        f"总判定 = **{fwd.get('all_bitwise_equal')}**"
    )
    lines.append(
        f"- 手算侧隔离：{fwd.get('manual_side_isolated')} —— {fwd.get('manual_side_note')}"
    )
    lines.append(
        f"- 反向验证：基线通过 = {rev.get('baseline_passed')}，"
        f"全部破坏被检出 = {rev.get('all_detected')}，恢复后通过 = {rev.get('restored_passed')}，"
        f"总判定 = **{rev.get('passed')}**"
    )
    lines.append("")
    lines.append("| 破坏项 | 说明 | 被检出 |")
    lines.append("| --- | --- | --- |")
    for c in rev.get("cases", []):
        lines.append(f"| `{c['case']}` | {c['description']} | **{c['detected']}** |")
    lines.append("")


def render_calibrate_markdown(report: Dict[str, Any]) -> str:
    """把**训练前提校准**报告渲染为 Markdown（锚点表 + 口径对照 + 扫描表 + 归因）。"""
    lines: List[str] = []
    lines.append("# n3d_qa_learn 变体 B：训练前提校准轮（第二步的后续诊断轮）")
    lines.append("")
    lines.append(f"- 状态：`{report.get('status')}`；产物 schema = `{report.get('artifact_schema')}`")
    lines.append(f"- 口径：{report.get('scope_note')}")
    lines.append(f"- 单 seed 声明：{report.get('single_seed_note')}")
    lines.append(f"- 粒度：{report.get('granularity')}")
    lines.append(f"- 训练侧打分口径：{report.get('train_scorer_rule')}")
    lines.append(f"- 锚点规则：{report.get('anchor_rule')}")
    lines.append(f"- 训练无害判据：{report.get('harmlessness_rule')}")
    lines.append(f"- 选取规则：{report.get('selection_rule')}")
    lines.append(f"- 归因规则：{report.get('attribution_rule')}")
    lines.append("")
    grid = dict(report.get("grid", {}))
    lines.append(
        f"**网格**：{grid.get('n_combos')} 个组合 = {grid.get('combos')}；"
        f"顺序规则：{grid.get('order_rule')}"
    )
    lines.append("")
    lines.append("## 1. 恒等门禁与恒等参照锚点（T = I，训练行 1999）")
    lines.append("")
    lines.append("| 档 | 恒等门禁（逐条一致格数） | 通过 | 锚点 vs KNN 逐格一致 |")
    lines.append("| --- | --- | --- | --- |")
    for profile, per in sorted(report.get("per_profile", {}).items()):
        gate = dict(per.get("identity_gate", {}))
        anchor = dict(per.get("identity_anchor", {}))
        eq = dict(anchor.get("identity_equivalence", {}))
        gate_cell = (
            "跳过" if gate.get("skipped") else f"{gate.get('n_cells_all_equal')}/{gate.get('n_cells')}"
        )
        lines.append(
            f"| `{profile}` | {gate_cell} | **{gate.get('passed')}** | "
            f"{eq.get('n_cells_all_equal_vs_knn')}/{eq.get('n_cells')} |"
        )
    lines.append("")
    lines.append("| 档 | 格 | 锚点 R@1（**判定量，训练行 1999**） | 锚点 R@5 | 锚点 KNN R@1 |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for profile, per in sorted(report.get("per_profile", {}).items()):
        anchor = dict(per.get("identity_anchor", {}))
        for cell in sorted(dict(anchor.get("top1_self", {})).keys()):
            lines.append(
                f"| `{profile}` | `{cell}` | {float(anchor['top1_self'][cell]):.6f} | "
                f"{float(dict(anchor.get('recall_at_5', {})).get(cell, 0.0)):.6f} | "
                f"{(lambda v: format(float(v), '.6f') if v is not None else '—')(dict(anchor.get('knn_top1_self', {})).get(cell))} |"
            )
    lines.append("")
    lines.append("> **行集合警告**：上表落在**训练行 1999** 上，与 1a §15.5 的**查询行 666** 登记值")
    lines.append("> **不可直接互校**（现场例：`nmag/weak` 训练行 `0.355178` vs 查询行 `0.391892`）——")
    lines.append("> 属行集合不同而非数值漂移。与 1a 的对账入口见下表 §1.1。")
    lines.append("")
    lines.append("### 1.1 与 1a 登记值的对账（**查询行 666 侧**，旁证，容差 1e-6）")
    lines.append("")
    a1 = dict(next(iter(sorted(report.get("per_profile", {}).values())), {}).get("anchor_1a_check") or {})
    if a1:
        lines.append(
            f"- 行来源：{a1.get('row_source')}；在容差内 {a1.get('n_within_tol')}/"
            f"{a1.get('n_rows')}，全部在容差内 = **{a1.get('all_within_tol')}**"
        )
        lines.append("")
        lines.append("| 档 | 格 | 1a 登记 | 现场重算 | 绝对差 | 在容差内 |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for r in a1.get("rows", []):
            lines.append(
                f"| `{r['profile']}` | `{r['cell']}` | {float(r['knn_1a_registered']):.6f} | "
                f"{float(r['knn_recomputed']):.6f} | {float(r['abs_diff']):.2e} | {r['within_tol']} |"
            )
    else:
        lines.append("- 本产物未产出该项（如实登记，不静默跳过）。")
    lines.append("")
    for profile, per in sorted(report.get("per_profile", {}).items()):
        kn = dict(per.get("knn_invariance_across_combos") or {})
        if kn:
            lines.append(
                f"- `{profile}` KNN 组合间不变量：{kn.get('n_distinct_knn_vectors')} 个不同向量 / "
                f"{kn.get('n_combos')} 个组合 ⇒ 全部相同 = **{kn.get('all_equal')}**。"
            )
    lines.append("")
    lines.append("## 2. 优化前提扫描（train_scorer × lr × weight_decay）")
    lines.append("")
    lines.append(
        "> **观测量口径**：`逐格 R@1 均值(9 格)` 与 `训练无害格数` 是**判定量**"
        "（训练行 1999 上跑一次完整 10 格网格）；`批内滚动平均（仅诊断）` 是训练循环里的"
        "滚动量、**仅诊断**，**不得**与恒等锚点直接相减（见 §2.2 的现场差值）。"
    )
    lines.append("")
    lines.append(
        "| 档 | 组合 | 逐格 R@1 均值(9 格，判定量) | 训练无害格数(9 格) | 最差余量 | 平均余量 | "
        "批内滚动平均（仅诊断） | ‖T(x)‖末/首 | 评测侧不劣格数 | 评测侧严格更高 | 更新量门禁 |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for profile, per in sorted(report.get("per_profile", {}).items()):
        for rec in per.get("combos", []):
            harm = dict(rec["train_harmlessness"])
            prim = dict(rec["eval_primary"])
            nd = dict(rec["train"]["norm_diagnostics_summary"])
            rvg = dict(rec["train"].get("running_vs_grid") or {})
            lines.append(
                f"| `{profile}` | `{_md_table_cell(rec['combo']['label'])}` | "
                f"{float(rvg.get('grid_r1_mean_9_perturb_cells', 0.0)):.6f} | "
                f"{harm['n_not_worse']}/{harm['n_cells']} | {float(harm['min_margin']):+.6f} | "
                f"{float(harm['mean_margin']):+.6f} | "
                f"{float(rec['train']['train_batch_running_top1']):.6f} | "
                f"{float(nd['norm_ratio_last_over_first']):.4f} | "
                f"{prim['n_not_worse']}/{prim['n_cells']} | {prim['n_strictly_better']} | "
                f"{rec['update_gate']['passed']} |"
            )
    lines.append("")
    lines.append("### 2.1 口径对齐前后（同 lr / wd 配对；两个观测量分开报）")
    lines.append("")
    lines.append("| 档 | lr | weight_decay | raw 逐格 R@1 均值(9 格) | normalized 逐格 R@1 均值(9 格) | 差值 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for profile, per in sorted(report.get("per_profile", {}).items()):
        by_pair: Dict[Tuple[float, float], Dict[str, float]] = {}
        for rec in per.get("combos", []):
            key = (float(rec["combo"]["lr"]), float(rec["combo"]["weight_decay"]))
            rvg = dict(rec["train"].get("running_vs_grid") or {})
            by_pair.setdefault(key, {})[str(rec["combo"]["train_scorer"])] = float(
                rvg.get("grid_r1_mean_9_perturb_cells", 0.0)
            )
        for (lr, wd), vals in sorted(by_pair.items()):
            r = vals.get("raw")
            n = vals.get("normalized")
            diff = (float(n) - float(r)) if (r is not None and n is not None) else None
            lines.append(
                f"| `{profile}` | {lr:g} | {wd:g} | "
                f"{'—' if r is None else format(float(r), '.6f')} | "
                f"{'—' if n is None else format(float(n), '.6f')} | "
                f"{'—' if diff is None else format(float(diff), '+.6f')} |"
            )
    lines.append("")
    lines.append("### 2.2 批内滚动平均 vs 逐格 R@1（**两个观测量，不得互相代入**）")
    lines.append("")
    lines.append(f"- 口径：{TRAIN_TOP1_SELF_RULE}")
    lines.append("")
    lines.append(
        "| 档 | 组合 | 批内滚动平均（仅诊断） | 逐格 R@1 均值(9 格，判定量) | "
        "逐格 R@1 均值(10 格含 clean) | 差值(滚动 − 逐格 9 格) | 逐格一致性 |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for profile, per in sorted(report.get("per_profile", {}).items()):
        for rec in per.get("combos", []):
            rvg = dict(rec["train"].get("running_vs_grid") or {})
            lines.append(
                f"| `{profile}` | `{_md_table_cell(rec['combo']['label'])}` | "
                f"{float(rvg.get('train_batch_running_top1', 0.0)):.6f} | "
                f"{float(rvg.get('grid_r1_mean_9_perturb_cells', 0.0)):.6f} | "
                f"{float(rvg.get('grid_r1_mean_10_cells_incl_clean', 0.0)):.6f} | "
                f"**{float(rvg.get('delta_running_minus_grid_mean', 0.0)):+.6f}** | "
                f"{rvg.get('per_cell_consistent')} |"
            )
    lines.append("")
    lines.append("## 3. 训练无害下限（本轮主判据，训练行 1999）")
    lines.append("")
    lines.append("| 档 | 组合 | 格 | 锚点 | 训练后 | 余量 | 不低于锚点 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
    for profile, per in sorted(report.get("per_profile", {}).items()):
        for rec in per.get("combos", []):
            for row in dict(rec["train_harmlessness"]).get("per_cell", []):
                if not row.get("applicable"):
                    continue
                lines.append(
                    f"| `{profile}` | `{_md_table_cell(rec['combo']['label'])}` | `{row['cell']}` | "
                    f"{float(row['anchor_top1']):.6f} | {float(row['post_top1']):.6f} | "
                    f"**{float(row['margin']):+.6f}** | {row['not_worse']} |"
                )
    lines.append("")
    lines.append("## 4. 归因结论（机器可读，单点来源）")
    lines.append("")
    summary = dict(report.get("attribution_summary", {}))
    lines.append(f"- 逐档结论：`{summary.get('per_profile_verdict')}`")
    lines.append(f"- 各档一致：{summary.get('all_profiles_same_verdict')}")
    lines.append(f"- **结论键：`{summary.get('verdict_key')}`**")
    lines.append(f"- 结论文本：{summary.get('verdict')}")
    lines.append(f"- 单点来源：{summary.get('single_source_note')}")
    lines.append("")
    for profile, att in sorted(dict(report.get("attribution", {})).items()):
        lines.append(f"### 档 `{profile}`（组合 `{dict(att.get('combo', {})).get('label')}`）")
        lines.append("")
        lines.append(f"- 条件逐项：`{att.get('conditions')}`；必需条件 = `{att.get('conditions_required')}`")
        lines.append(f"- 条件 a：{dict(att.get('conditions_text', {})).get('a_train_harmless')}")
        lines.append(
            f"  - 实测：不劣 {dict(att.get('evidence_a', {})).get('n_not_worse')}/"
            f"{dict(att.get('evidence_a', {})).get('n_cells')}，最差余量 "
            f"{dict(att.get('evidence_a', {})).get('min_margin')}"
        )
        lines.append(f"- 条件 b：{dict(att.get('conditions_text', {})).get('b_eval_not_worse')}")
        lines.append(
            f"  - 实测：不劣 {dict(att.get('evidence_b', {})).get('n_not_worse')}/"
            f"{dict(att.get('evidence_b', {})).get('n_cells')}，严格更高 "
            f"{dict(att.get('evidence_b', {})).get('n_strictly_better')}"
        )
        lines.append(f"- **verdict = `{att.get('verdict_key')}`**：{att.get('verdict')}")
        lines.append("")
    lines.append("## 5. 取回路径（W2 修复）与反向验证")
    lines.append("")
    first = next(iter(sorted(report.get("per_profile", {}).values())), {})
    rv = dict(first.get("perturb_retrieval") or {})
    fwd = dict(rv.get("forward") or {})
    rev = dict(rv.get("reverse") or {})
    lines.append(f"- 口径：{report.get('retrieval_rule')}")
    lines.append(f"- 正向总判定 = **{fwd.get('all_bitwise_equal')}**；反向总判定 = **{rev.get('passed')}**")
    lines.append("")
    lines.append("| 破坏项 | 说明 | 被检出 |")
    lines.append("| --- | --- | --- |")
    for c in rev.get("cases", []):
        lines.append(f"| `{c['case']}` | {c['description']} | **{c['detected']}** |")
    lines.append("")
    lines.append("## 6. 如实登记（不得包装）")
    lines.append("")
    for i, note in enumerate(report.get("honest_notes", []), start=1):
        lines.append(f"{i}. {note}")
    lines.append("")
    return "\n".join(lines)


def write_json(path: str, obj: Any) -> str:
    """写 JSON（UTF-8 无 BOM，缩进 1，排序键；返回文件 SHA256）。"""
    return R.write_json(path, obj)


def write_text(path: str, text: str) -> str:
    """写文本（UTF-8 无 BOM；返回文件 SHA256）。"""
    return R.write_text(path, text)


def load_json(path: str) -> Dict[str, Any]:
    """读取 JSON 产物（缺失即可读报错）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"variantb 产物不存在：{path!r}")
    with open(path, "r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def summarize_probe(
    profiles: Sequence[str], *, product_dir: str = "", log: Any = None
) -> Dict[str, Any]:
    """**口径取证**（不训练）：逐档给出键表 / 划分 / 恒等门禁的三组现场事实。

    参数
    ----
    profiles : Sequence[str]
        参与取证的档。
    product_dir : str
        ``n3d_qa`` 冻结产物目录。
    log : Any
        可调用日志。

    返回
    ----
    Dict[str, Any]
        ``profiles``（逐档取证）/ ``errors``（失败清单，**如实登记不静默跳过**）/
        ``transform_names``（可训参数的现场枚举名字）/ ``identity_fastpath_rule``。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    per: Dict[str, Any] = {}
    errors: List[Dict[str, Any]] = []
    names: List[str] = []
    retrieval_forward = perturb_retrieval_evidence()
    retrieval_reverse = perturb_retrieval_reverse_check()
    for profile in profiles:
        try:
            data = load_profile_data(str(profile), product_dir=str(product_dir), log=_log)
            model = build_model(data.table)
            names = [str(n) for n, _ in model.transform.named_parameters()]
            gate = identity_gate(data)
            anchor = identity_anchor_grid(model, data)
            per[str(profile)] = {
                "data": data.as_dict(),
                "identity_gate": gate,
                "identity_anchor": anchor,
                "parameter_snapshot": model.parameter_snapshot(),
                "transform_names": names,
                "transform_dim": int(model.dim),
                "n_keys": int(model.n_keys),
                "perturb_retrieval": {
                    "forward": retrieval_forward,
                    "reverse": retrieval_reverse,
                },
            }
            _log(
                f"[variantb probe] 档 {profile}: D={model.dim} N={model.n_keys} "
                f"可训参数={names} 恒等门禁通过={gate['passed']} "
                f"训练侧打分口径自洽={gate['train_scorer']['all_bitwise_equal']}"
            )
        except Exception as exc:  # noqa: BLE001 —— 如实登记不静默跳过
            errors.append(
                {
                    "profile": str(profile),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            _log(f"[variantb probe] 档 {profile} 失败（如实登记）：{type(exc).__name__}: {exc}")
    return {
        "module": MODULE_NAME,
        "artifact_schema": ARTIFACT_SCHEMA,
        "deterministic": True,
        "single_seed_note": "单 seed 42；本产物不含时间字段",
        "profiles": per,
        "errors": errors,
        "n_ok": int(len(per)),
        "n_error": int(len(errors)),
        "transform_names": names,
        "identity_fastpath_rule": IDENTITY_FASTPATH_RULE,
        "train_scorer_rule": TRAIN_SCORER_RULE,
        "train_scorers": [str(x) for x in TRAIN_SCORERS],
        "default_train_scorer": str(DEFAULT_TRAIN_SCORER),
        "anchor_rule": TRAIN_ANCHOR_RULE,
        "granularity": TRAIN_GRANULARITY_NOTE,
        "perturb_retrieval": {
            "forward": retrieval_forward,
            "reverse": retrieval_reverse,
        },
        "reuse_1a": (
            "扰动 = robust_eval.perturb_matrix / derived_seed / PERTURB_GRID；"
            "归一化 = entry_table.l2_normalize_rows"
        ),
    }


__all__ = [
    "ANCHOR_RULE",
    "ANCHOR_ROW_SET_NOTE",
    "ANCHOR_TOL",
    "ARTIFACT_SCHEMA",
    "ATTRIBUTION_CONDITION_TEXT",
    "ATTRIBUTION_REQUIRED_CONDITIONS",
    "ATTRIBUTION_RULE",
    "ATTRIBUTION_VERDICTS",
    "CALIBRATE_CANDIDATE_CELL_SET",
    "CALIBRATE_LR_GRID",
    "CALIBRATE_SCORERS",
    "CALIBRATE_SELECTION_RULE",
    "CALIBRATE_WD_GRID",
    "CALIBRATION_SCOPE_NOTE",
    "CalibrateCombo",
    "CellMetrics",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_LR",
    "DEFAULT_TRAIN_SCORER",
    "DEFAULT_WEIGHT_DECAY",
    "DToDTransform",
    "EFFECTIVE_CELLS_1A",
    "FULL_TABLE_ROWS",
    "IDENTITY_FASTPATH_RULE",
    "INEFFECTIVE_CELLS_1A",
    "LIBRARY_ROWS",
    "MODULE_NAME",
    "QUERY_ROWS",
    "RETRIEVAL_MAX_ROUNDS",
    "RETRIEVAL_RULE",
    "RETRIEVAL_ZERO_TOKEN",
    "ROBUST_1A_KNN_REFERENCE",
    "SHUFFLE_SEED",
    "TOP1_DISCREPANCY_KEY",
    "TRAIN_ANCHOR_RULE",
    "TRAIN_GRANULARITY_NOTE",
    "TRAIN_HARMLESSNESS_EXCLUDED_CELLS",
    "TRAIN_HARMLESSNESS_RULE",
    "TRAIN_MODES",
    "TRAIN_PERTURB_CELLS",
    "TRAIN_SCORER_RULE",
    "TRAIN_SCORERS",
    "TRAIN_TOP1_SELF_RULE",
    "TRAIN_TOP1_KEY",
    "TRAIN_TOP1_KEY_FALLBACK_RULE",
    "LEGACY_TRAIN_TOP1_KEY",
    "TrainConfig",
    "UPDATE_GATE_RULE",
    "VARIANT_B_DIR",
    "VARIANT_B_SEED",
    "VariantBData",
    "VariantBModel",
    "anchor_1a_check",
    "artifact_fingerprint",
    "attribution_verdict",
    "build_calibration_grid",
    "build_data",
    "build_model",
    "cell_grid",
    "clean_eval_of_perturb_arm",
    "compare_top1",
    "drill",
    "eval_cell",
    "explicit_path_evidence",
    "identity_anchor_grid",
    "identity_gate",
    "gradient_flow_evidence",
    "load_json",
    "load_profile_data",
    "perturb_retrieval_evidence",
    "perturb_retrieval_reverse_check",
    "primary_criterion",
    "render_calibrate_markdown",
    "render_markdown",
    "run_calibration",
    "run_variant_b",
    "sample_bitwise_equality",
    "score_path_evidence",
    "select_best_combo",
    "summarize_probe",
    "third_step_evidence",
    "train_and_eval_profile",
    "train_batch_running_top1_of",
    "train_harmlessness",
    "train_scorer_evidence",
    "train_transform",
    "update_gate",
    "write_json",
    "write_text",
    "zero_norm_mask",
]
