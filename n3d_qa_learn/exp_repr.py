"""n3d_qa_learn 的「表示训练」对照实验（exp_repr）。

实验目的
--------
把一句**结构事实**变成可量化的结论：

    现有默认配置（``train_head=False`` + ``answer_table_mode="centroid"`` +
    ``head_input_mode="raw"``）下，唯一可训练参数是标量 ``logit_scale``；
    而 logits 对**正标量**的正比例缩放不改变 ``argmax``，所以
    「冻结特征 + 冻结的逐类质心答案表」下的 top-1 完全由几何决定，
    训练不改变任何预测。

本模块在 **n3d_qa_learn** 模块内提供一组对照实验，量化「表示训练」本身对
类内/类间可分性（``gap`` / ``sigma``）的影响，并定位/修复 ``train_head=True``
的「一律拒绝」退化解（现场登记：``macro 0.00`` / ``refusal_rate 1.00``）。

被测维度（逐维单独测，每次只换一项）
----------------------------------
* **a 表示冻结范围**：``logit_only``（基线，仅 ``logit_scale``）/
  ``head``（+``q_head``）/ ``head_backbone``（+``q_head``+骨干）；
* **b 头输入口径**：``raw`` / ``concat`` / ``n3d``（取值域取自
  :data:`n3d_qa_learn.heads.HEAD_INPUT_MODES`）；
* **c 目标修法**：``ce``（现行交叉熵）/ ``no_irr_centroid``（「不相关」类不参与质心吸引）/
  ``staged``（分阶段：先训表示再定标量）/ ``supcon``（监督对比损失，同答案问题为正样本）。

矩阵（共 8 组）与每个维度的基准组严格对应「每次只换一项」：

========================  ==================  ================  =================
组名                      freeze_scope        head_input_mode   loss_mode
========================  ==================  ================  =================
``A1_baseline``           ``logit_only``      ``raw``           ``ce``
``A2_head``               ``head``            ``raw``           ``ce``
``A3_head_backbone``      ``head_backbone``   ``raw``           ``ce``
``B1_concat``             ``logit_only``      ``concat``        ``ce``
``B2_n3d``                ``logit_only``      ``n3d``           ``ce``
``C1_no_irr_centroid``    ``head``            ``raw``           ``no_irr_centroid``
``C2_staged``             ``head``            ``raw``           ``staged``
``C3_supcon``             ``head``            ``raw``           ``supcon``
========================  ==================  ================  =================

* 维度 a 的基准 = ``A1``，变化项 = ``A2`` / ``A3``；
* 维度 b 的基准 = ``A1``，变化项 = ``B1`` / ``B2``；
* 维度 c 的基准 = ``A2``（「打开 q 头训练」这一**退化态**本身），
  变化项 = ``C1`` / ``C2`` / ``C3``。修法只有在表示可训时才有意义，
  故该维度的基准必须落在 ``freeze_scope="head"`` 上。

固定切分与可复现
----------------
* **切分 seed 与训练 seed 分离**：切分由 :attr:`TrainConfig.split_seed` 驱动
  （``-1`` = 沿用 ``seed``，即历史行为），训练只由 ``seed`` 驱动
  （``torch.Generator().manual_seed``）。所有对照组共用**同一份切分**，
  并逐位断言 ``train_known`` / ``test_known``（以及两个 ``*_unknown``）的
  qid 有序序列相同；不一致即该对照判无效。
* 监督对比损失的正负样本构造**显式固定**：同一答案类的问题互为正样本、
  不同答案类互为负样本、批内构造、不采样；温度是冻结常量。

零回归
------
* 本模块**不写任何正式产物**：所有输出只落
  ``checkpoints/qa_learn/_verify/exp_repr/``（见 :data:`EXP_REPR_DIR`）；
* 训练一律不落盘（不生成 zip 产物），因此 ``checkpoints/qa_learn/`` 顶层与
  ``checkpoints/triviaqa/`` 不被触碰。
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import step2 as S2
from .backends import (
    GEO_FIELD_CHOICES,
    SHAPE_CHOICES,
    STRUCTURE_DEFAULTS,
    BackendRegistry,
    BackendStructure,
)
from .data import DEFAULT_QA_CACHE_DIR, DEFAULT_TEXT_DIR
from .encoders import (
    DEFAULT_MODEL_DIR,
    ENCODER_BGE_M3,
    ENCODER_LOCAL_HASH,
    ROLE_QUESTION,
    ROLE_TEXT_LINE,
    EncoderConfig,
    build_vectorizer,
    canonical_dumps,
    sha256_bytes,
)
from .evaluate import evaluate_refusal, evaluate_step1
from .heads import (
    ALIGN_MODES,
    ALIGN_TARGETS,
    HEAD_INPUT_MODES,
    PROJ_INIT_MODES,
    SUPCON_TEMPERATURE,
    N3DQA,
    N3DQAConfig,
    supcon_loss,
)
from .train import (
    DEFAULT_VERIFY_DIR,
    TrainConfig,
    TrainingData,
    _build_model,
    build_training_data,
    label_index_early,
    set_deterministic_seed,
)

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

#: 实验产物目录（**验证类**目录；绝不写正式产物目录）。
EXP_REPR_DIR: str = os.path.join(DEFAULT_VERIFY_DIR, "exp_repr")

#: 表示冻结范围（功能点 4a；取值即口径，禁止改名）。
FREEZE_SCOPES: Tuple[str, ...] = ("logit_only", "head", "head_backbone")

#: 目标修法（功能点 4c；取值即口径，禁止改名）。
LOSS_MODES: Tuple[str, ...] = ("ce", "no_irr_centroid", "staged", "supcon")

#: 监督对比损失的温度（冻结常量）。**唯一实现与常量落点在** ``heads``
#: （``exp_repr.supcon_loss`` / ``exp_repr.SUPCON_TEMPERATURE`` 只是再导出），
#: 这样机制 B 的对齐损失与 ``loss_mode="supcon"`` 必然是同一段代码。
#: 取值 :data:`n3d_qa_learn.heads.SUPCON_TEMPERATURE` = 0.07。

#: 结构开关（本轮新增）的取值域：直接引用 ``backends`` 的注册点，避免两处各写一套枚举。
SHAPES: Tuple[str, ...] = SHAPE_CHOICES

#: 本轮实际测的几何权重场档（``class_tied`` / ``mlp`` 上游构造期显式拒绝，不进矩阵）。
GEO_FIELDS: Tuple[str, ...] = ("none", "additive")

#: 对齐机制与投影头初始化的取值域（引用 ``heads`` 的注册点）。
ALIGN_MODE_CHOICES: Tuple[str, ...] = ALIGN_MODES
PROJ_INIT_CHOICES: Tuple[str, ...] = PROJ_INIT_MODES

#: 分阶段修法的两段 epoch 数（stage1 + stage2 = 40 = 基线 epoch 数，训练预算对齐）。
STAGE1_EPOCHS: int = 30
STAGE2_EPOCHS: int = 10

#: 基线档口径（与 ``TrainConfig`` 默认值一致，**逐字冻结**）。
BASE_MAX_CLASSES: int = 10
BASE_MIN_QUESTIONS: int = 8
BASE_TEST_EVERY: int = 3
BASE_TEST_PER_CLASS: int = 2
BASE_UNKNOWN_TRAIN_CAP: int = 500
BASE_EPOCHS: int = 40
BASE_BATCH_SIZE: int = 64
BASE_LR: float = 5e-3
BASE_BACKBONE_LR: float = 5e-4
BASE_LOGIT_SCALE_INIT: float = 20.0
BASE_SPLIT_SEED: int = 42
BASE_TRAIN_SEED: int = 42

#: 风后现场实测的**对照锚点**（执行时必须自己重算并逐项核对）。
ANCHOR: Dict[str, float] = {
    "within_mean": 0.8149,
    "within_sigma": 0.0433,
    "cross_mean": 0.7777,
    "cross_sigma": 0.0468,
    "gap": 0.0372,
    "gap_over_sigma": 0.8591,
    "nn1_top1_raw": 0.3000,
}

#: 锚点容差（**现场标定**）：取「本实现口径下的观测偏差」的约 1.3~2 倍向上取整，
#: 且不小于该量记录到 4 位小数时的半个末位（0.00005）；1-NN 是确定性口径，取 1e-9。
#: 标定过程与残差归因见 README。
ANCHOR_TOL: Dict[str, float] = {
    "within_mean": 0.0020,
    "within_sigma": 0.0020,
    "cross_mean": 0.0020,
    "cross_sigma": 0.0005,
    "gap": 0.0020,
    "gap_over_sigma": 0.0500,
    "nn1_top1_raw": 1e-09,
}

# ---------------------------------------------------------------------------
# 组定义与矩阵
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReprGroup:
    """一个对照组的标识（由三个**原有**被测维度 + 本轮新增的**结构/对齐**维度共同确定）。

    属性
    ----
    name : str
        组名（报告与终端输出中的键）。
    freeze_scope : str
        表示冻结范围，取值见 :data:`FREEZE_SCOPES`。
    head_input_mode : str
        ``q`` 头输入口径，取值见 :data:`n3d_qa_learn.heads.HEAD_INPUT_MODES`。
    loss_mode : str
        目标修法，取值见 :data:`LOSS_MODES`。
    shape / cyl_aspect / fc_dim / N / y_in / y_out / geo_field
        **后端结构开关**（本轮新增）。默认值 = :data:`backends.STRUCTURE_DEFAULTS`
        = 既有 8 组的口径，因此**原样保留既有 8 组时逐位可复现**。
    align_mode / align_lambda / proj_dim
        **对齐机制开关**（本轮新增）。默认 = 关闭（``off`` / ``0.0`` / ``0``），
        与既有 8 组逐位一致。
    """

    name: str
    freeze_scope: str
    head_input_mode: str
    loss_mode: str
    # ---- 结构开关（默认 = 既有口径）----
    shape: str = "sphere"
    cyl_aspect: float = 1.0
    fc_dim: int = 0
    N: int = 64
    y_in: int = 4
    y_out: int = 4
    geo_field: str = "none"
    # ---- 对齐开关（默认 = 关闭）----
    align_mode: str = "off"
    align_lambda: float = 0.0
    proj_dim: int = 0

    def __post_init__(self) -> None:
        if self.freeze_scope not in FREEZE_SCOPES:
            raise ValueError(
                f"freeze_scope 仅允许 {list(FREEZE_SCOPES)}，当前 {self.freeze_scope!r}"
            )
        if self.head_input_mode not in HEAD_INPUT_MODES:
            raise ValueError(
                f"head_input_mode 仅允许 {list(HEAD_INPUT_MODES)}，"
                f"当前 {self.head_input_mode!r}"
            )
        if self.loss_mode not in LOSS_MODES:
            raise ValueError(
                f"loss_mode 仅允许 {list(LOSS_MODES)}，当前 {self.loss_mode!r}"
            )
        if self.shape not in SHAPES:
            raise ValueError(f"shape 仅允许 {list(SHAPES)}，当前 {self.shape!r}")
        if self.geo_field not in GEO_FIELDS:
            raise ValueError(
                f"geo_field 仅允许 {list(GEO_FIELDS)}（本轮矩阵口径），"
                f"当前 {self.geo_field!r}"
            )
        if self.align_mode not in ALIGN_MODE_CHOICES:
            raise ValueError(
                f"align_mode 仅允许 {list(ALIGN_MODE_CHOICES)}，当前 {self.align_mode!r}"
            )

    def structure(self) -> BackendStructure:
        """本组的结构开关（装配成 :class:`BackendStructure`，供代理层透传）。"""
        return BackendStructure(
            shape=str(self.shape),
            cyl_aspect=float(self.cyl_aspect),
            fc_dim=int(self.fc_dim),
            N=int(self.N),
            y_in=int(self.y_in),
            y_out=int(self.y_out),
            geo_field=str(self.geo_field),
        )

    def structure_is_default(self) -> bool:
        """结构是否全默认（= 既有 8 组口径）。"""
        return bool(self.structure().is_default())

    def align_is_off(self) -> bool:
        """对齐是否关闭（= 既有 8 组口径）。"""
        return bool(self.align_mode == "off")

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化。"""
        return {
            "name": self.name,
            "freeze_scope": self.freeze_scope,
            "head_input_mode": self.head_input_mode,
            "loss_mode": self.loss_mode,
            "structure": self.structure().as_dict(),
            "align_mode": str(self.align_mode),
            "align_lambda": float(self.align_lambda),
            "proj_dim": int(self.proj_dim),
        }


#: 对照组矩阵（**既有 8 组 + 本轮 10 组**，顺序即报告顺序）。
#:
#: * 前 8 组 = 原样保留、**逐位复现**（既有口径：结构全默认 + 对齐关闭）；
#: * 后 10 组 = 本轮新增：**公共基线** ``S0_base``（``concat`` + ``head_backbone`` +
#:   结构默认 + 对齐关闭）+ **逐轴单换**（形状 / ``fc_dim`` / ``N`` / ``geo_field`` /
#:   对齐 B / 对齐 C×2），加上**载体对照** ``B1_concat`` / ``B2_n3d``（复用既有组，
#:   它们的 ``freeze_scope`` 是 ``logit_only``，正是"载体"这一轴的对照）。
#:   ⑦ 特征档（``lexical-88`` / ``bge-m3-1024``）**不是矩阵组**，而是
#:   :func:`run_comparison` 的 ``profiles`` 轴（同一组在两档下各跑一次）。
MATRIX: Tuple[ReprGroup, ...] = (
    # ---- 既有 8 组（逐位复现；不得改动其字段）----
    ReprGroup("A1_baseline", "logit_only", "raw", "ce"),
    ReprGroup("A2_head", "head", "raw", "ce"),
    ReprGroup("A3_head_backbone", "head_backbone", "raw", "ce"),
    ReprGroup("B1_concat", "logit_only", "concat", "ce"),
    ReprGroup("B2_n3d", "logit_only", "n3d", "ce"),
    ReprGroup("C1_no_irr_centroid", "head", "raw", "no_irr_centroid"),
    ReprGroup("C2_staged", "head", "raw", "staged"),
    ReprGroup("C3_supcon", "head", "raw", "supcon"),
    # ---- 本轮新增：公共基线 ----
    ReprGroup("S0_base", "head_backbone", "concat", "ce"),
    # ---- 本轮新增：逐轴单换（每次只换一项，其余等于 S0_base）----
    ReprGroup("S1_shape_cube", "head_backbone", "concat", "ce", shape="cube"),
    ReprGroup("S2_shape_cylinder", "head_backbone", "concat", "ce",
              shape="cylinder", cyl_aspect=1.0),
    ReprGroup("S3_fc_follow", "head_backbone", "concat", "ce", fc_dim=-1),
    ReprGroup("S4_fc_128", "head_backbone", "concat", "ce", fc_dim=128),
    ReprGroup("S5_N256", "head_backbone", "concat", "ce", N=256),
    ReprGroup("S6_geo_additive", "head_backbone", "concat", "ce", geo_field="additive"),
    ReprGroup("S7_align_B", "head_backbone", "concat", "ce",
              align_mode="proj_supcon", align_lambda=1.0, proj_dim=-1),
    ReprGroup("S8_align_C_l01", "head_backbone", "concat", "ce",
              align_mode="distill", align_lambda=0.1),
    ReprGroup("S9_align_C_l10", "head_backbone", "concat", "ce",
              align_mode="distill", align_lambda=1.0),
)

#: 基线组名（锚点对标与「换特征 vs 打开训练」归因的参照系）—— **既有口径，不得改名**。
BASELINE_GROUP: str = "A1_baseline"

#: 本轮「结构 / 对齐」各轴的**公共基线组名**（逐轴单换的参照系）。
STRUCT_BASELINE_GROUP: str = "S0_base"

#: 三个**原有**被测维度的结构描述（基准组 + 变化项 + 被改变的字段）。
DIMENSIONS: Tuple[Dict[str, Any], ...] = (
    {
        "dim": "a_trainable_scope",
        "varying_field": "freeze_scope",
        "base": "A1_baseline",
        "groups": ["A1_baseline", "A2_head", "A3_head_backbone"],
    },
    {
        "dim": "b_head_input_mode",
        "varying_field": "head_input_mode",
        "base": "A1_baseline",
        "groups": ["A1_baseline", "B1_concat", "B2_n3d"],
    },
    {
        "dim": "c_loss_mode",
        "varying_field": "loss_mode",
        "base": "A2_head",
        "groups": ["A2_head", "C1_no_irr_centroid", "C2_staged", "C3_supcon"],
    },
    # ---- 本轮新增的四条轴（基准 = S0_base；逐轴单换，禁止全交叉）----
    {
        "dim": "d_shape",
        "varying_field": "shape (+cyl_aspect)",
        "base": STRUCT_BASELINE_GROUP,
        "groups": ["S0_base", "S1_shape_cube", "S2_shape_cylinder"],
    },
    {
        "dim": "e_fc_dim",
        "varying_field": "fc_dim",
        "base": STRUCT_BASELINE_GROUP,
        "groups": ["S0_base", "S3_fc_follow", "S4_fc_128"],
    },
    {
        "dim": "f_capacity_N",
        "varying_field": "N",
        "base": STRUCT_BASELINE_GROUP,
        "groups": ["S0_base", "S5_N256"],
    },
    {
        "dim": "g_geo_field",
        "varying_field": "geo_field",
        "base": STRUCT_BASELINE_GROUP,
        "groups": ["S0_base", "S6_geo_additive"],
    },
    {
        "dim": "h_align",
        "varying_field": "align_mode (+align_lambda/proj_dim)",
        "base": STRUCT_BASELINE_GROUP,
        "groups": ["S0_base", "S7_align_B", "S8_align_C_l01", "S9_align_C_l10"],
    },
    {
        "dim": "i_carrier",
        "varying_field": "freeze_scope (载体组：是否打开表示训练)",
        "base": STRUCT_BASELINE_GROUP,
        "groups": ["S0_base", "B1_concat", "B2_n3d"],
    },
)

#: ``ungrouped_trainable_names`` 在现场允许出现的**完整**名单（显式列出，不放宽为"大部分"）。
#: 本轮结论：**无需扩充**。投影头（机制 B）走 ``model.alignment_parameters()``，已在
#: :func:`make_optimizer` / :func:`ungrouped_trainable_names` 里显式并入 ``head`` 组，
#: 因此它**不在**本名单内；本名单仍然只有 ``concat`` 口径下恒不进优化器的 ``mix_logit``。
ALLOWED_UNGROUPED_TRAINABLE: Tuple[str, ...] = ("head.mix_logit",)


def group_by_name(name: str) -> ReprGroup:
    """按组名取矩阵中的组定义（未知名立即报错，不静默回落）。"""
    for group in MATRIX:
        if group.name == name:
            return group
    raise KeyError(f"未知组名 {name!r}；可用组名 = {[g.name for g in MATRIX]}")


#: 「载体组」的取值 -> ``(freeze_scope, head_input_mode)``（本轮确认口径的显式注册点）。
#: 载体组回答的是"q 的载体是什么 + 表示训练是否打开"这一**组合**问题：
#: ``base`` = 公共基线（``head_backbone`` + ``concat``，表示训练打开）；
#: ``B1_concat`` = 只训 ``logit_scale`` 且载体为 ``concat``（= 既有 B1 组口径）；
#: ``B2_n3d`` = 只训 ``logit_scale`` 且载体为 ``n3d``（= 既有 B2 组口径）。
CARRIER_GROUPS: Dict[str, Tuple[str, str]] = {
    "base": ("head_backbone", "concat"),
    "B1_concat": ("logit_only", "concat"),
    "B2_n3d": ("logit_only", "n3d"),
}

#: 自定义组可覆盖的轴字段（顺序即命名顺序；**逐轴单换**的合法入口）。
AXIS_FIELDS: Tuple[str, ...] = (
    "carrier", "shape", "cyl_aspect", "fc_dim", "N", "y_in", "y_out",
    "geo_field", "align_mode", "align_lambda", "proj_dim",
)


def custom_group(
    *,
    base: str = STRUCT_BASELINE_GROUP,
    carrier: Optional[str] = None,
    shape: Optional[str] = None,
    cyl_aspect: Optional[float] = None,
    fc_dim: Optional[int] = None,
    N: Optional[int] = None,
    y_in: Optional[int] = None,
    y_out: Optional[int] = None,
    geo_field: Optional[str] = None,
    align_mode: Optional[str] = None,
    align_lambda: Optional[float] = None,
    proj_dim: Optional[int] = None,
) -> ReprGroup:
    """从某个矩阵组出发，按 CLI 给定的轴覆盖构造一个**自定义组**（逐轴单换）。

    口径
    ----
    * 未给出的轴一律沿用 ``base``（默认 :data:`STRUCT_BASELINE_GROUP` = 公共基线），
      因此这是"只换指定轴"的合法入口，不会引入隐式全交叉；
    * 对齐三件套有**联动默认**：显式给了 ``align_mode != "off"`` 而未给
      ``align_lambda`` / ``proj_dim`` 时，按机制取默认（B -> λ=1.0、proj_dim=-1；
      C -> λ=0.1、proj_dim=0）；
    * 组名由实际生效的覆盖**确定性**生成（``X_<field>=<value>__...``），
      便于报告与产物对账；
    * 结果仍是 :class:`ReprGroup`，构造期不变量（枚举合法性、对齐开关组合）照常生效。

    参数
    ----
    base : str
        起始组名（必须是 :data:`MATRIX` 里的组）。
    其余 : 各轴的覆盖值；``None`` = 不覆盖。

    返回
    ----
    ReprGroup
        自定义组定义。

    异常
    ------
    KeyError
        ``base`` 不是已知组名。
    ValueError
        覆盖值本身非法（由 :class:`ReprGroup` / 对齐联动规则抛出）。
    """
    origin = group_by_name(str(base))
    fields: Dict[str, Any] = {
        "carrier": carrier,
        "shape": shape,
        "cyl_aspect": cyl_aspect,
        "fc_dim": fc_dim,
        "N": N,
        "y_in": y_in,
        "y_out": y_out,
        "geo_field": geo_field,
        "align_mode": align_mode,
        "align_lambda": align_lambda,
        "proj_dim": proj_dim,
    }
    used = {k: v for k, v in fields.items() if v is not None}
    if not used:
        return origin
    kwargs: Dict[str, Any] = {
        "freeze_scope": origin.freeze_scope,
        "head_input_mode": origin.head_input_mode,
        "loss_mode": origin.loss_mode,
        "shape": origin.shape,
        "cyl_aspect": origin.cyl_aspect,
        "fc_dim": origin.fc_dim,
        "N": origin.N,
        "y_in": origin.y_in,
        "y_out": origin.y_out,
        "geo_field": origin.geo_field,
        "align_mode": origin.align_mode,
        "align_lambda": origin.align_lambda,
        "proj_dim": origin.proj_dim,
    }
    if "carrier" in used:
        key = str(used["carrier"])
        if key not in CARRIER_GROUPS:
            raise ValueError(
                f"carrier 仅允许 {sorted(CARRIER_GROUPS)}（载体组注册表），当前 {key!r}"
            )
        scope, mode = CARRIER_GROUPS[key]
        kwargs["freeze_scope"] = scope
        kwargs["head_input_mode"] = mode
    if "shape" in used:
        kwargs["shape"] = str(used["shape"])
    if "cyl_aspect" in used:
        kwargs["cyl_aspect"] = float(used["cyl_aspect"])
    if "fc_dim" in used:
        kwargs["fc_dim"] = int(used["fc_dim"])
    if "N" in used:
        kwargs["N"] = int(used["N"])
    if "y_in" in used:
        kwargs["y_in"] = int(used["y_in"])
    if "y_out" in used:
        kwargs["y_out"] = int(used["y_out"])
    if "geo_field" in used:
        kwargs["geo_field"] = str(used["geo_field"])
    if "align_mode" in used:
        amode = str(used["align_mode"])
        kwargs["align_mode"] = amode
        if amode == "off":
            kwargs["align_lambda"] = 0.0
            kwargs["proj_dim"] = 0
        elif amode == "proj_supcon":
            kwargs["align_lambda"] = float(used.get("align_lambda", 1.0))
            kwargs["proj_dim"] = int(used.get("proj_dim", -1))
        else:  # distill
            kwargs["align_lambda"] = float(used.get("align_lambda", 0.1))
            kwargs["proj_dim"] = int(used.get("proj_dim", 0))
    if "align_lambda" in used and "align_mode" not in used:
        kwargs["align_lambda"] = float(used["align_lambda"])
    if "proj_dim" in used and "align_mode" not in used:
        kwargs["proj_dim"] = int(used["proj_dim"])
    name = "X_" + "__".join(f"{k}={used[k]}" for k in AXIS_FIELDS if k in used)
    return ReprGroup(name=name, **kwargs)


# ---------------------------------------------------------------------------
# 特征档（词面 D=88 vs 语义 bge-m3 D=1024）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureProfile:
    """一个**特征档**：步骤 1（题面）与步骤 2（文本行）各自的编码器口径。

    属性
    ----
    name : str
        档名（报告键）。
    label : str
        中文标签（报告可读性）。
    question : EncoderConfig
        步骤 1 题面口径（决定模型连接参数 ``D``）。
    text_line : EncoderConfig
        步骤 2 文本行口径（**必须与 ``question`` 同维**，否则同一条 q 通路不可用）。
    expect_dim : int
        该档的 ``D``（现场与 ``declared_dim`` 对账）。

    口径声明（**非平凡**）
    --------------------
    ``step2.py`` 的默认文本行口径是 ``zh-bag``（``D=192``），而本实验的模型由
    **步骤 1** 的 ``D`` 决定；为了让「同一条 q 通路」在步骤 2 上可用，
    文本行一律用**与步骤 1 同族**的编码器（词面档 ``local-hash`` D=88 /
    语义档 ``bge-m3`` ``role=text_line`` D=1024）。这一点在报告中显式标注，
    不与 ``step2_run`` 的默认口径混为一谈。
    """

    name: str
    label: str
    question: EncoderConfig
    text_line: EncoderConfig
    expect_dim: int

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化（含两端口径的 ``to_dict``）。"""
        return {
            "name": str(self.name),
            "label": str(self.label),
            "expect_dim": int(self.expect_dim),
            "question": self.question.to_dict(),
            "text_line": self.text_line.to_dict(),
        }


#: 词面档（现状口径，**基线**）。
PROFILE_LEXICAL: str = "lexical-88"
#: 语义档（新特征，``BAAI/bge-m3``，``D = hidden_size = 1024``）。
PROFILE_SEMANTIC: str = "bge-m3-1024"
#: 默认档（= 现状口径；不显式选档时行为与历史一致）。
DEFAULT_PROFILE: str = PROFILE_LEXICAL

#: 特征档注册表（**唯一注册点**）。
FEATURE_PROFILES: Dict[str, FeatureProfile] = {
    PROFILE_LEXICAL: FeatureProfile(
        name=PROFILE_LEXICAL,
        label="词面口径（blake2b 哈希词袋 + 8 长度特征 + L2）",
        question=EncoderConfig(name=ENCODER_LOCAL_HASH, role=ROLE_QUESTION),
        text_line=EncoderConfig(name=ENCODER_LOCAL_HASH, role=ROLE_TEXT_LINE),
        expect_dim=88,
    ),
    PROFILE_SEMANTIC: FeatureProfile(
        name=PROFILE_SEMANTIC,
        label="语义口径（BAAI/bge-m3，手工 CLS pooling + L2）",
        question=EncoderConfig(
            name=ENCODER_BGE_M3, role=ROLE_QUESTION, source=DEFAULT_MODEL_DIR
        ),
        text_line=EncoderConfig(
            name=ENCODER_BGE_M3, role=ROLE_TEXT_LINE, source=DEFAULT_MODEL_DIR
        ),
        expect_dim=1024,
    ),
}


def profile_by_name(name: str) -> FeatureProfile:
    """按档名取特征档（未知名立即报错，不静默回落）。"""
    key = str(name) if str(name) else DEFAULT_PROFILE
    if key not in FEATURE_PROFILES:
        raise KeyError(
            f"未知特征档 {key!r}；可用档 = {sorted(FEATURE_PROFILES.keys())}"
        )
    return FEATURE_PROFILES[key]


def split_qids_digest(qids: Dict[str, List[str]]) -> Dict[str, Any]:
    """四子集 qid 序列的**sha256 摘要**（报告里不必重复存整表，又能逐位对账）。"""
    return {
        "per_split_sha256": {
            k: sha256_bytes(canonical_dumps([str(x) for x in v]))
            for k, v in sorted(qids.items())
        },
        "n_per_split": {k: int(len(v)) for k, v in sorted(qids.items())},
    }


def split_qids_equal(a: Dict[str, List[str]], b: Dict[str, List[str]]) -> bool:
    """四子集 qid 有序序列是否**逐位相同**（跨特征档的切分一致性判据）。"""
    for key in ("train_known", "test_known", "train_unknown", "test_unknown"):
        if list(a.get(key, [])) != list(b.get(key, [])):
            return False
    return True


def dimension_cost(
    dim: int,
    *,
    n_train_known: int,
    n_train_unknown: int,
    n_classes: int,
    backend: str = "n3d_shape",
    structure: Optional[BackendStructure] = None,
) -> Dict[str, Any]:
    """连接参数 ``D`` 与**结构开关**的**代价实测**（现场构造，不凭记忆写）。

    报告内容
    --------
    * 骨干逐参数形状与元素数 —— **一律现场枚举 ``named_parameters()``**，
      不假设存在 ``W_in`` / ``W_out``（现场实测：``fc_dim != 0`` 时骨干参数名被替换为
      ``fc_in_weight`` / ``fc_in_bias`` / ``proj_weight`` / ``fc_out_weight`` /
      ``fc_out_bias`` / ``head_weight`` / ``head_bias``，``W_in`` / ``W_out`` **消失**）；
    * ``q`` 头逐参数（``q_head.weight`` / ``q_head.bias`` / ``logit_scale``）与合计；
    * ``answer_table`` 是 **buffer**（``centroid`` 口径下冻结，不入优化器）的字节数；
    * 总参数量、每样本参数比（分母 = ``train_known + train_unknown``）；
    * **两种情形的可训参数量**：
      「冻结嵌入 + 质心答案表」（``freeze_scope="logit_only"``，只训 ``logit_scale``）
      与「打开表示训练」（``head`` / ``head_backbone``）。

    参数
    ----
    dim : int
        连接参数 ``D``。
    n_train_known / n_train_unknown : int
        训练侧样本数（每样本参数比的分母）。
    n_classes : int
        答案类别数 ``C``。
    backend : str
        后端名（默认 ``n3d_shape``，与矩阵口径一致）。
    structure : Optional[BackendStructure]
        结构开关（``None`` = 全默认档）。

    返回
    ----
    Dict[str, Any]
        代价实测字典。``backbone_shapes`` / ``backbone_parameters`` 是**现场枚举**结果，
        **不保证**含 ``W_in`` / ``W_out`` 键；消费侧必须用 ``.get`` 且能把缺键渲染成
        ``n/a``（本模块的 Markdown 渲染已如此）。
    """
    reg = BackendRegistry(int(dim), structure=structure)
    adapter = reg.register(str(backend))
    backbone = adapter.model
    backbone_shapes = {n: [int(x) for x in p.shape] for n, p in backbone.named_parameters()}
    backbone_numel = {n: int(p.numel()) for n, p in backbone.named_parameters()}
    backbone_total = int(sum(backbone_numel.values()))

    head_model = N3DQA(
        adapter,
        int(n_classes),
        N3DQAConfig(dim=int(dim), output_mode="index", head_input_mode="raw",
                    logit_scale_init=BASE_LOGIT_SCALE_INIT, learn_logit_scale=True,
                    answer_table_mode="centroid"),
    )
    head_numel = {n: int(p.numel()) for n, p in head_model.named_parameters()}
    head_shapes = {n: [int(x) for x in p.shape] for n, p in head_model.named_parameters()}
    head_total = int(sum(head_numel.values()))
    table = head_model.answer_table
    answer_table = {
        "shape": [int(x) for x in table.shape],
        "numel": int(table.numel()),
        "bytes_float32": int(table.numel() * 4),
        "trainable": bool(isinstance(table, nn.Parameter)),
        "role": "buffer（centroid 口径：由训练样本逐类质心确定性写入，冻结）",
    }
    n_samples = int(n_train_known) + int(n_train_unknown)
    # 「打开表示训练」的可训集合**现场枚举**（不硬编码 q_head.weight 等名字），
    # 只统计 head 侧（N3DQA 自己的参数）；骨干侧另计。
    head_trainable = {
        n: int(p.numel()) for n, p in head_model.named_parameters()
    }
    frozen_trainable = int(head_trainable.get("logit_scale", 0))
    opened_trainable = int(sum(head_trainable.values()))
    return {
        "dim": int(dim),
        "backend": str(backend),
        "structure": (
            BackendStructure() if structure is None else structure
        ).as_dict(),
        "topology": {k: float(v) for k, v in adapter.topology_stats().items()},
        "backbone_parameter_names": sorted(backbone_numel.keys()),
        "backbone_parameters": backbone_numel,
        "backbone_shapes": backbone_shapes,
        "backbone_total": int(backbone_total),
        "head_parameter_names": sorted(head_numel.keys()),
        "head_parameters": head_numel,
        "head_shapes": head_shapes,
        "head_total": int(head_total),
        "total_parameters": int(backbone_total + head_total),
        "answer_table_buffer": answer_table,
        "n_train_samples": int(n_samples),
        "params_per_sample": (
            float(backbone_total + head_total) / float(n_samples) if n_samples > 0 else 0.0
        ),
        "trainable_frozen_embedding": {
            "scope": "logit_only（冻结嵌入 + 质心答案表）",
            "n_trainable": int(frozen_trainable),
            "names": ["head.logit_scale"],
            "note": ("q 头与骨干都冻结；answer_table 是 buffer（不在优化器内）；"
                     "「训练」只改一个正标量尺度，而正标量缩放不改变 argmax"),
        },
        "trainable_open_representation": {
            "scope": "head / head_backbone（打开表示训练）",
            "n_trainable_head": int(opened_trainable),
            "n_trainable_head_backbone": int(opened_trainable + backbone_total),
            # 现场枚举（不假设一定有 q_head.weight / q_head.bias）
            "names_head": sorted(f"head.{k}" for k in head_trainable),
            "names_head_backbone_extra": sorted(backbone_numel.keys()),
            "note": ("head 档训 q 头；head_backbone 档额外声明骨干可训，"
                     "但在 head_input_mode='raw' 下骨干结构性不在计算图上（见 README 12.8）"),
        },
    }


# ---------------------------------------------------------------------------
# 步骤 2 自检索（同一 q 通路）—— 主判据的第二半
# ---------------------------------------------------------------------------


@dataclass
class Step2Bundle:
    """一个特征档的**步骤 2 自检索**共用件（与模型无关的部分，每档只建一次）。

    属性
    ----
    profile : str
        特征档名。
    product_dir : str
        ``n3d_qa`` 冻结产物目录（只读）。
    vectorizer : Any
        文本行向量化器（与步骤 1 同族、**同维**）。
    rows : List[Any]
        文本行（``n3d_qa`` 冻结行表全量，只读）。
    split : Any
        库/查询划分（**只作划分复核与查询集来源**）。
    pool_index : List[int]
        **检索池**下标。口径与 ``step2_run eval`` **逐字一致**：``n3d_qa`` 产物的
        ``label_rule`` 是「candidate library row IS the query row」，故检索池 =
        冻结行表全量（否则「命中自身行」不可定义）；冻结划分出的库/查询互斥性另行复核。
    query_index : List[int]
        查询集下标（= 冻结划分出的 query 行 ∩ 检索池）。
    key_table : Any
        冻结候选键表（由**该档特征**在检索池上构造）。
    det_baseline : Dict[str, Any]
        **不经 N3D** 的纯特征余弦检索 ``Recall@1/@5``（与模型无关，故每档只算一次）；
        这是「换特征」最干净的可比量。
    evidence : Dict[str, Any]
        取证（行数、池/查询规模、口径指纹、D）。
    """

    profile: str
    product_dir: str
    vectorizer: Any
    rows: List[Any]
    split: Any
    pool_index: List[int]
    query_index: List[int]
    key_table: Any
    det_baseline: Dict[str, Any]
    evidence: Dict[str, Any]


def build_step2_bundle(
    profile: FeatureProfile,
    *,
    product_dir: str = "",
    topk: int = 5,
    pool_cap: int = 0,
    batch_size: int = 256,
) -> Step2Bundle:
    """构造某特征档的步骤 2 自检索共用件（**只读** ``n3d_qa`` 冻结产物）。

    参数
    ----
    profile : FeatureProfile
        特征档（用它的 ``text_line`` 口径）。
    product_dir : str
        ``n3d_qa`` 冻结产物目录；空串 = ``step2.resolve_product_dir`` 自动定位。
    topk : int
        ``Recall@k`` 的最大 k（同时报 ``@1``）。
    pool_cap : int
        检索池行数上限（``0`` = 全量；限批演练用，**报告里显式登记是否限批**）。
    batch_size : int
        编码批大小。

    返回
    ----
    Step2Bundle
        共用件。

    异常
    ------
    ValueError
        向量化维度与声明 ``expect_dim`` 不一致（拒绝静默错配）。
    """
    resolved = S2.resolve_product_dir(str(product_dir))
    rows = S2.load_text_rows(resolved)
    row_index = S2.load_row_index(resolved)
    meta = S2.load_doclines_meta(resolved)
    vectorizer = build_vectorizer(profile.text_line)
    if int(vectorizer.dim) != int(profile.expect_dim):
        raise ValueError(
            f"特征档 {profile.name!r} 的文本行向量化维度 {vectorizer.dim} "
            f"与声明 expect_dim={profile.expect_dim} 不一致（拒绝静默错配）"
        )
    split = S2.reproduce_doc_split(rows, row_index, meta)
    pool_index = list(range(len(rows)))
    limited = False
    if int(pool_cap) > 0 and len(pool_index) > int(pool_cap):
        pool_index = pool_index[: int(pool_cap)]
        limited = True
    pool_set = set(pool_index)
    # 查询集 = 冻结划分出的查询行 ∩ 检索池（限批时可能被截掉一部分）
    query_index = [int(i) for i in split.query_index if int(i) in pool_set]
    key_table = S2.build_key_table(vectorizer, rows, pool_index)
    det = S2.deterministic_baseline(
        vectorizer, rows, query_index, pool_index, key_table,
        topk=int(topk), batch_size=int(batch_size),
    )
    evidence = {
        "product_dir": resolved,
        "n_rows": int(len(rows)),
        "n_pool": int(len(pool_index)),
        "n_query": int(len(query_index)),
        "pool_cap": int(pool_cap),
        "pool_limited": bool(limited),
        "frozen_split": {
            "library_rows": int(split.n_library),
            "query_rows": int(split.n_query),
            "intersection": int(split.evidence["intersection"]),
            "union": int(split.evidence["union"]),
        },
        "retrieval_pool_rule": (
            "检索池 = 冻结行表全量（n3d_qa label_rule: candidate library row IS the "
            "query row）；查询集 = 冻结划分出的 query 行 ∩ 检索池"
        ),
        "dim": int(vectorizer.dim),
        "encoder_fingerprint": str(vectorizer.fingerprint()),
        "role": str(profile.text_line.role),
        "max_length": int(profile.text_line.resolved_max_length()),
        "split_seed": int(split.seed),
        "key_table_sha256": str(key_table.sha256()),
        "deterministic_baseline": dict(det),
    }
    return Step2Bundle(
        profile=str(profile.name), product_dir=resolved, vectorizer=vectorizer,
        rows=list(rows), split=split, pool_index=pool_index, query_index=query_index,
        key_table=key_table, det_baseline=dict(det), evidence=evidence,
    )


@torch.no_grad()
def step2_recall_of_model(
    model: N3DQA,
    bundle: Step2Bundle,
    *,
    topk: int = 5,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """把**训练好的模型**接到步骤 2：``q = q_head(enc(text_line))`` → 库内检索。

    口径（与 :mod:`n3d_qa_learn.step2` 的自检索**同构**）
    ------------------------------------------------
    * **检索池 = 冻结行表全量**（``n3d_qa`` 的 ``label_rule``：「candidate library row
      IS the query row」），查询集 = 冻结划分出的 query 行 ∩ 检索池；这与
      ``step2_run eval`` 的口径**逐字一致**（不是「库=1999 行子集」那一种）；
    * 查询行经**同一条 q 通路**在池中检索，命中自身行即正确；
    * 另报**不经 N3D** 的纯特征检索（``bundle.det_baseline``）作为参照下限。

    参数
    ----
    model : N3DQA
        训练好的模型（``model.config.dim`` 必须等于 ``bundle.vectorizer.dim``）。
    bundle : Step2Bundle
        步骤 2 共用件。
    topk / batch_size : int
        运行参数。

    返回
    ----
    Dict[str, Any]
        ``q_path``（经 N3D）与 ``deterministic``（不经 N3D）两组 ``Recall@1/@5``。
    """
    if int(model.config.dim) != int(bundle.vectorizer.dim):
        raise ValueError(
            f"步骤 2 的向量化维度 {bundle.vectorizer.dim} 与模型 q 维度 "
            f"{model.config.dim} 不一致；拒绝在错配的特征空间上检索"
        )
    matcher = S2.TextRowMatcher(model, bundle.vectorizer, "index", bundle.key_table)
    q_idx = [int(i) for i in bundle.query_index]
    if _pool_matches_table(bundle):
        # 非限批：直接复用 step2 的母实现（口径逐字一致）
        qpath = S2.self_retrieval(
            matcher, bundle.rows, q_idx, list(bundle.pool_index), bundle.key_table,
            topk=int(topk), batch_size=int(batch_size),
        )
    else:
        # 限批演练：口径相同，只把「库下标来源」换成键表自身（见函数 docstring）
        qpath = _self_retrieval_limited(
            matcher, bundle, q_idx, topk=int(topk), batch_size=int(batch_size)
        )
    return {
        "q_path": {k: qpath[k] for k in
                   ("n", "n_library", "topk", "recall_at_1", "recall_at_5",
                    "rank_1", "rank_miss_topk")},
        "q_path_misses_head": list(qpath.get("misses_head", []))[:10],
        "deterministic": dict(bundle.det_baseline),
        "dim": int(bundle.vectorizer.dim),
        "protocol": (
            "检索池 = n3d_qa 冻结行表全量；查询集 = 冻结划分的 query 行；查询行经同一 "
            "q 通路检索，命中自身行即正确；deterministic 为不经 N3D 的同池同查询参照下限"
        ),
    }


def _pool_matches_table(bundle: Step2Bundle) -> bool:
    """检索池下标序列是否与键表行序一致（限批时不一致，需走专用路径）。"""
    pool_ids = [str(bundle.rows[int(i)].row_id) for i in bundle.pool_index]
    return pool_ids == [str(x) for x in bundle.key_table.line_ids]


def _self_retrieval_limited(
    matcher: Any,
    bundle: Step2Bundle,
    query_index: Sequence[int],
    *,
    topk: int = 5,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """限批检索池上的自检索（口径与 :func:`step2.self_retrieval` 相同，只换池下标来源）。

    存在理由：``step2.self_retrieval`` 会显式断言「库行顺序 == 键表行 id 顺序」，
    限批（``pool_cap > 0``）时该断言必然失败；本函数用**键表自身的行 id 顺序**
    作为池顺序，保持其余口径逐字不变。
    """
    q_idx = [int(i) for i in query_index]
    pos_of = {str(lid): i for i, lid in enumerate(bundle.key_table.line_ids)}
    kk = max(1, int(topk))
    hits1 = 0
    hits5 = 0
    misses: List[Dict[str, Any]] = []
    ranks: List[int] = []
    for b0 in range(0, len(q_idx), int(batch_size)):
        chunk = q_idx[b0: b0 + int(batch_size)]
        texts = [bundle.rows[i].text for i in chunk]
        logits = matcher.logits(texts)
        order = torch.argsort(logits, dim=1, descending=True)[:, :kk]
        for i, row_i in enumerate(chunk):
            gold = str(bundle.rows[row_i].row_id)
            gold_pos = pos_of.get(gold)
            if gold_pos is None:
                raise ValueError(f"查询行 {gold!r} 不在限批库中；拒绝在缺金标的库上算命中")
            rank_list = [int(x) for x in order[i].tolist()]
            rank = (rank_list.index(gold_pos) + 1) if gold_pos in rank_list else -1
            ranks.append(int(rank))
            hits1 += int(rank == 1)
            hits5 += int(1 <= rank <= kk)
            if rank != 1 and len(misses) < 10:
                misses.append({"row_id": gold, "rank": int(rank)})
    n = len(q_idx)
    return {
        "n": int(n),
        "n_library": int(bundle.key_table.size),
        "topk": int(kk),
        "recall_at_1": float(hits1 / max(1, n)),
        "recall_at_5": float(hits5 / max(1, n)),
        "rank_1": int(hits1),
        "rank_miss_topk": int(sum(1 for r in ranks if r < 0)),
        "misses_head": misses,
    }


# ---------------------------------------------------------------------------
# 基础工具（续）
# ---------------------------------------------------------------------------


def _device() -> torch.device:
    """训练/评估设备（有 GPU 用 GPU，否则 CPU）。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _feats(vectorizer: Any, records: Sequence[Any], device: torch.device) -> torch.Tensor:
    """把问题文本编码为 ``[N, D]`` 张量（与推理端共用同一向量化口径）。"""
    return torch.tensor(
        [vectorizer.encode(r.question) for r in records],
        dtype=torch.float32,
        device=device,
    )


def _mean_std(values: torch.Tensor) -> Tuple[float, float]:
    """``(均值, 样本标准差 ddof=1)``（显式实现，不依赖 torch 的 ``unbiased`` 弃用面）。"""
    n = int(values.numel())
    mean = float(values.mean().item())
    if n < 2:
        return mean, 0.0
    var = float(((values - values.mean()) ** 2).sum().item()) / float(n - 1)
    return mean, math.sqrt(var)


def split_qids(data: TrainingData) -> Dict[str, List[str]]:
    """四子集的 **qid 有序序列**（切分一致性断言的比较对象）。"""
    return {
        "train_known": [r.qid for r in data.splits.train_known],
        "train_unknown": [r.qid for r in data.splits.train_unknown],
        "test_known": [r.qid for r in data.splits.test_known],
        "test_unknown": [r.qid for r in data.splits.test_unknown],
    }


def assert_same_split(reference: Dict[str, List[str]], other: Dict[str, List[str]],
                      tag: str) -> None:
    """断言两份切分的 qid 有序序列**逐位相同**（不一致即该对照判无效）。

    参数
    ----
    reference : Dict[str, List[str]]
        基准切分（首个被跑出来的组）。
    other : Dict[str, List[str]]
        待比较切分。
    tag : str
        组名（报错报文里给出上下文）。

    异常
    ------
    AssertionError
        任一子集的 qid 序列不逐位相同；报文显式列出**子集名**与**首个不同位置**。
    """
    for key in ("train_known", "test_known", "train_unknown", "test_unknown"):
        a, b = reference[key], other[key]
        if a == b:
            continue
        detail = f"长度 基准 {len(a)} vs 本组 {len(b)}"
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                detail = f"首个不同位置 i={i}：基准 {a[i]!r} vs 本组 {b[i]!r}"
                break
        raise AssertionError(
            f"[{tag}] 切分一致性断言失败：{key} 的 qid 序列与基准不逐位相同"
            f"（{detail}）；该对照判无效"
        )


# ---------------------------------------------------------------------------
# 配置装配
# ---------------------------------------------------------------------------


def build_group_config(
    group: ReprGroup,
    *,
    split_seed: int = BASE_SPLIT_SEED,
    train_seed: int = BASE_TRAIN_SEED,
    epochs: int = BASE_EPOCHS,
    batch_size: int = BASE_BATCH_SIZE,
    lr: float = BASE_LR,
    backbone_lr: float = BASE_BACKBONE_LR,
    logit_scale_init: float = BASE_LOGIT_SCALE_INIT,
    unknown_train_cap: int = BASE_UNKNOWN_TRAIN_CAP,
    max_classes: int = BASE_MAX_CLASSES,
    min_questions: int = BASE_MIN_QUESTIONS,
    test_every: int = BASE_TEST_EVERY,
    test_per_class: int = BASE_TEST_PER_CLASS,
    qa_cache_dir: str = "",
    text_dir: str = "",
) -> TrainConfig:
    """把「组定义 + 实验超参」装配成 :class:`TrainConfig`。

    连接参数（特征维 ``D``）的唯一来源仍是代理层 ``backends``，本函数只负责把
    组定义映射为 ``TrainConfig`` 的字段，**不承载训练编排**。

    参数
    ----
    group : ReprGroup
        组定义。
    split_seed : int
        **切分种子**（与训练种子分离；``-1`` = 沿用 ``train_seed``，即历史行为）。
    train_seed : int
        训练种子。
    其余为训练 / 数据装配超参。

    返回
    ----
    TrainConfig
        配置（``answer_table_mode`` 恒为 ``centroid``，故答案表是 buffer 而非参数）。
    """
    return TrainConfig(
        backend="n3d_shape",
        output_mode="index",
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=0.0,
        seed=int(train_seed),
        split_seed=int(split_seed),
        max_classes=int(max_classes),
        min_questions=int(min_questions),
        test_every=int(test_every),
        test_per_class=int(test_per_class),
        unknown_class_weight=1.0,
        unknown_train_cap=int(unknown_train_cap),
        label_smoothing=0.0,
        max_train_samples=0,
        text_threshold=0.28,
        qa_cache_dir=str(qa_cache_dir) if qa_cache_dir else DEFAULT_QA_CACHE_DIR,
        text_dir=str(text_dir) if text_dir else DEFAULT_TEXT_DIR,
        train_backbone=(group.freeze_scope == "head_backbone"),
        backbone_lr=float(backbone_lr),
        head_input_mode=str(group.head_input_mode),
        logit_scale_init=float(logit_scale_init),
        learn_logit_scale=True,
        answer_table_mode="centroid",
        train_head=(group.freeze_scope in ("head", "head_backbone")),
        # ---- 结构开关（本轮新增；默认 = 既有口径，逐位不变）----
        structure_shape=str(group.shape),
        structure_cyl_aspect=float(group.cyl_aspect),
        structure_fc_dim=int(group.fc_dim),
        structure_N=int(group.N),
        structure_y_in=int(group.y_in),
        structure_y_out=int(group.y_out),
        structure_geo_field=str(group.geo_field),
        # ---- 对齐开关（本轮新增；默认关闭，逐位不变）----
        align_mode=str(group.align_mode),
        align_lambda=float(group.align_lambda),
        proj_dim=int(group.proj_dim),
    )


# ---------------------------------------------------------------------------
# 损失与冻结口径
# ---------------------------------------------------------------------------
# 口径说明（本轮上移，审查收口）：监督对比损失的**唯一实现**落在
# `n3d_qa_learn.heads.supcon_loss`（§模块 docstring 的「对齐机制」），本模块在文件头
# `from .heads import supcon_loss` 做再导出 —— 因此 `exp_repr.supcon_loss` 与
# `heads.supcon_loss` 是**同一个函数对象**，机制 B 与 `loss_mode="supcon"` 不可能漂移。
# 默认参数 `temperature=SUPCON_TEMPERATURE`（0.07）随实现一起上移，取值不变。


def batch_loss(
    model: N3DQA,
    kind: str,
    logits: torch.Tensor,
    q: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    n_classes: int,
    *,
    feats: Optional[torch.Tensor] = None,
    n3d_branch: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """按口径分派一个 batch 的损失（**含对齐附加项**），返回含分项的诊断。

    ``kind`` 取值与语义
    ------------------
    * ``"ce"``：现行交叉熵（含逐样本权重），**与 :meth:`N3DQA.cross_entropy` 同路径**；
    * ``"ce_masked"``：「不相关」样本以 ``keep = targets != C`` 显式排除，
      不产生把它们拉向「不相关」键的梯度（配合答案表末位置零，见
      :func:`setup_answer_table`）；
    * ``"supcon"``：监督对比损失（见 :func:`supcon_loss`）。

    对齐附加项（本轮新增）
    --------------------
    当 ``model.config.align_mode != "off"`` 时，总损失 = ``基础损失 + λ · align``，
    其中 ``align`` 由 :meth:`N3DQA.alignment_loss` 给出（机制 B = supcon /
    机制 C = 显式蒸馏两项）。**关闭对齐时路径逐字符不变**：不额外前向、不加任何项。

    参数
    ----
    feats : Optional[torch.Tensor]
        ``[B, D]`` 本 batch 的编码器原始特征。开启对齐时**必需**（蒸馏目标 (i)(ii)
        都在这个空间里算）。
    n3d_branch : Optional[torch.Tensor]
        ``[B, D]`` 已算好的 :meth:`N3DQA.n3d_branch` 输出（与 ``query`` 共用同一次
        骨干前向，避免两倍算力）。

    返回
    ----
    Dict[str, Any]
        ``{"loss", "base", "align", "align_detail", "parts"}``：
        ``loss`` 是本 batch 实际反传的标量（``None`` = 本 batch 无可做功样本，
        调用侧按「跳过该 batch」处置并计数）；
        ``base`` / ``align`` 是分项标量值（``float`` 或 ``None``）；
        ``parts`` 是各分项的字典（供报告逐项登记）。
    """
    align_detail: Optional[Dict[str, Any]] = None
    base: Optional[torch.Tensor]
    if kind == "ce":
        base = model.cross_entropy(logits, targets, sample_weight=weight)
    elif kind == "ce_masked":
        per = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=float(model.config.label_smoothing),
        )
        keep = targets != int(n_classes)
        if int(keep.sum().item()) == 0:
            base = None
        else:
            base = (per[keep] * weight[keep]).sum() / weight[keep].sum().clamp_min(1e-12)
    elif kind == "supcon":
        base = supcon_loss(q, targets, int(n_classes))
    else:
        raise ValueError(f"未知损失口径 {kind!r}")

    parts: Dict[str, Any] = {
        "base": None if base is None else float(base.detach().item()),
        "align": None,
        "align_lambda": float(model.config.align_lambda),
        "align_mode": str(model.config.align_mode),
        "align_applicable": False,
        "align_supcon": None,
        "align_distill_self": None,
        "align_distill_centroid": None,
        "align_n_used": 0,
        "align_n_excluded": 0,
        "align_skipped_reason": None,
        "distill_self_weight": None,
        "distill_centroid_weight": None,
    }
    if str(model.config.align_mode) != "off":
        if feats is None:
            raise ValueError(
                "align_mode 非 off 时必须提供 feats（对齐目标 (i)(ii) 都在编码器特征空间里算）"
            )
        align_detail = model.alignment_loss(
            feats, targets, int(n_classes), branch=n3d_branch
        )
        parts.update({
            "align": (None if align_detail["loss"] is None
                      else float(align_detail["loss"].detach().item())),
            "align_applicable": bool(align_detail["applicable"]),
            "align_supcon": align_detail["supcon"],
            "align_distill_self": align_detail["distill_self"],
            "align_distill_centroid": align_detail["distill_centroid"],
            "align_n_used": int(align_detail["n_used"]),
            "align_n_excluded": int(align_detail["n_excluded"]),
            "align_skipped_reason": align_detail["skipped_reason"],
            # 两个对齐目标在总损失里的**实际权重**（都取 0.5 的分项系数 × λ）：
            # 显式写出来，避免报告把它们与「λ 本身」混为一谈。
            "distill_self_weight": (
                0.5 * float(model.config.align_lambda)
                if align_detail["distill_self"] is not None else None
            ),
            "distill_centroid_weight": (
                0.5 * float(model.config.align_lambda)
                if align_detail["distill_centroid"] is not None else None
            ),
        })

    total: Optional[torch.Tensor]
    if base is None and (align_detail is None or align_detail["loss"] is None):
        total = None
    elif base is None:
        total = align_detail["weighted"]  # type: ignore[index]
    elif align_detail is None or align_detail["loss"] is None:
        total = base
    else:
        total = base + align_detail["weighted"]
    parts["total"] = None if total is None else float(total.detach().item())
    return {
        "loss": total,
        "base": None if base is None else float(base.detach().item()),
        "align": parts["align"],
        "align_detail": align_detail,
        "parts": parts,
    }


def setup_answer_table(model: N3DQA, data: TrainingData,
                       loss_mode: str, device: torch.device) -> Dict[str, Any]:
    """按 ``centroid`` 口径写入答案表，并施加 ``no_irr_centroid`` 的定标改动。

    参数
    ----
    model : N3DQA
        模型（答案表在该模型上就地写入）。
    data : TrainingData
        数据装配结果（提供 ``train_known`` / ``train_unknown`` 与向量化器）。
    loss_mode : str
        目标修法；``"no_irr_centroid"`` 时对末位施加额外处置。
    device : torch.device
        设备。

    **口径说明（审查收口，皋陶 info）**：本函数**不接收** ``TrainConfig``。
    历史实现曾带一个 ``cfg`` 形参但在函数体内从未使用（属"看似依赖训练配置"的
    误导性签名），现按审查意见**删除该形参**并同步唯一调用点
    （``run_group``，现场枚举全仓仅此 1 处）。答案表的取值只由
    ``data`` 与 ``loss_mode`` 决定，与任何训练超参无关。

    口径
    ----
    * 与 :func:`n3d_qa_learn.train.run_training` **完全一致**：答案表由
      ``train_known + train_unknown`` 的**原始文本特征**逐类质心确定性写入、
      L2 归一化、固化为 buffer（末位 ``C`` = 「不相关」类质心）；
    * ``loss_mode == "no_irr_centroid"`` 时，把末位（「不相关」键）**置零** ——
      即「不相关」类不再以一个真实质心充当吸引子。这是该修法的**第一个动作**；
      第二个动作在 ``batch_loss`` 的 ``ce_masked`` 里（不相关样本不参与损失）。

    返回
    ----
    Dict[str, Any]
        构造期（训练前）的答案表诊断量。
    """
    C = int(data.corpus.n_classes)
    all_train = list(data.splits.train_known) + list(data.splits.train_unknown)
    feats = _feats(data.vectorizer, all_train, device)
    targets = torch.tensor(
        [label_index_early(r, data) for r in all_train],
        dtype=torch.long,
        device=device,
    )
    model.set_answer_table_from_centroids(feats, targets, C)
    norm_before = float(model.answer_table[C].detach().norm().item())
    zeroed = False
    if loss_mode == "no_irr_centroid":
        with torch.no_grad():
            model.answer_table[C].zero_()
        zeroed = True
    return {
        "irrelevant_row_index": int(C),
        "irrelevant_row_zeroed": bool(zeroed),
        "irrelevant_row_norm_if_not_zeroed": norm_before,
        "irrelevant_row_norm_before_train": float(
            model.answer_table[C].detach().norm().item()
        ),
    }


def apply_freeze(model: N3DQA, scope: str, train_repr: bool, train_scale: bool) -> None:
    """设置 ``requires_grad``（与 :func:`n3d_qa_learn.train.run_training` 的冻结口径同构）。

    参数
    ----
    model : N3DQA
        模型。
    scope : str
        表示冻结范围（``logit_only`` / ``head`` / ``head_backbone``）。
    train_repr : bool
        本阶段是否训练表示（``q`` 头；``head_backbone`` 档同时含骨干）。
    train_scale : bool
        本阶段是否训练 ``logit_scale``。
    """
    train_head = bool(train_repr) and scope in ("head", "head_backbone")
    train_backbone = bool(train_repr) and scope == "head_backbone"
    for p in model.q_head.parameters():
        p.requires_grad_(train_head)
    # 投影头（机制 B）**随 head / head_backbone 档可训**：它属于"头侧"表示参数，
    # 其全部梯度都来自对齐损失（见 `batch_loss`），不会被别的路径污染。
    for p in model.alignment_parameters():
        p.requires_grad_(train_head)
    for p in model.adapter.model.parameters():
        p.requires_grad_(train_backbone)
    model.logit_scale.requires_grad_(bool(train_scale))
    model.adapter.model.train(mode=bool(train_backbone))


def make_optimizer(
    model: N3DQA, cfg: TrainConfig
) -> Tuple[torch.optim.Optimizer, List[str]]:
    """按当前 ``requires_grad`` 构造优化器，并**现场枚举**进入优化器的参数名。

    参数集合与 ``run_training`` 同口径：``q_head`` 的参数、（``free`` 口径下的）
    答案表参数、``logit_scale``、以及（可训时的）骨干参数。

    **如实登记的偏差点**：``concat`` 口径下的 ``mix_logit`` 是 ``nn.Parameter``
    且默认 ``requires_grad=True``，但 ``run_training`` 的 ``head_params`` 只取
    ``model.q_head.parameters()``，故它**从不进入优化器**（恒为初值）。本实验
    与之一致地把它排除在优化器外，并在 :func:`ungrouped_trainable_names` 中显式报出。

    返回
    ----
    Tuple[torch.optim.Optimizer, List[str]]
        （优化器，优化器内参数名列表）。
    """
    named: List[Tuple[str, nn.Parameter]] = []
    # [!] 参数名必须**以 ``model.named_parameters()`` 的现场命名为唯一来源**，
    #     再按 ``id`` 映射回 ``q_head`` 的参数。历史缺陷：直接取
    #     ``model.q_head.named_parameters()`` 得到的是相对名（``weight`` / ``bias``），
    #     前缀后成为 ``head.weight``，与 :func:`param_deltas` 的键
    #     （``head.q_head.weight``）**对不上**，导致门禁把真实更新过的参数误判为
    #     "未更新"、并把它们计入 ``missing_from_snapshot``。
    name_of = {id(p): n for n, p in model.named_parameters()}
    for _n, p in model.q_head.named_parameters():
        if p.requires_grad:
            named.append((f"head.{name_of[id(p)]}", p))
    # 投影头（机制 B；`proj.weight` / `proj.bias`）同属 head 组。名字一律取自
    # `model.named_parameters()` 的**现场命名**（因此天然是 `head.proj.weight` 形态，
    # 与 `param_deltas` 的键口径一致 —— 历史纠正记录 #9）。
    proj = getattr(model, "proj", None)
    if proj is not None:
        for _n, p in proj.named_parameters():
            if p.requires_grad:
                named.append((f"head.{name_of[id(p)]}", p))
    if isinstance(getattr(model, "answer_table", None), nn.Parameter) and bool(
        model.answer_table.requires_grad
    ):
        named.append((f"head.{name_of[id(model.answer_table)]}", model.answer_table))
    if bool(model.logit_scale.requires_grad):
        named.append((f"head.{name_of[id(model.logit_scale)]}", model.logit_scale))
    backbone = [
        (f"backbone.{n}", p)
        for n, p in model.adapter.model.named_parameters()
        if p.requires_grad
    ]
    groups: List[Dict[str, Any]] = [{"params": [p for _, p in named], "lr": float(cfg.lr)}]
    if backbone:
        groups.append({"params": [p for _, p in backbone], "lr": float(cfg.backbone_lr)})
    if not any(g["params"] for g in groups):
        raise RuntimeError(
            "本阶段没有任何可训练参数（检查 freeze_scope 与阶段划分）；"
            "拒绝构造空优化器"
        )
    opt_cls = torch.optim.AdamW if float(cfg.weight_decay) > 0.0 else torch.optim.Adam
    optimizer = opt_cls(groups, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    return optimizer, [n for n, _ in named] + [n for n, _ in backbone]


def ungrouped_trainable_names(model: N3DQA) -> List[str]:
    """现场枚举「``requires_grad=True`` 但不进入优化器」的参数名（诊断用）。

    该集合在现场**应当**恰好是 :data:`ALLOWED_UNGROUPED_TRAINABLE` 的子集
    （当前实现里只有 ``concat`` 口径下的 ``head.mix_logit``）。集合类断言必须
    显式列出允许的额外项，故这里返回现场枚举结果、并由调用侧与该常量比对，
    而不是写"恰好等于空集"。

    口径：``grouped`` 由**优化器实际接纳的参数集合**定义（``q_head`` 的全部参数、
    答案表（若为 Parameter）、``logit_scale``、以及骨干的全部参数），按 ``id`` 比对；
    名字一律取自 ``model.named_parameters()`` / 骨干 ``named_parameters()`` 的现场命名。
    """
    grouped = {id(p) for p in model.q_head.parameters()}
    grouped.add(id(model.logit_scale))
    # 投影头（机制 B）进入优化器，故属于 grouped；未启用时 `alignment_parameters()` 为空。
    grouped.update(id(p) for p in model.alignment_parameters())
    if isinstance(getattr(model, "answer_table", None), nn.Parameter):
        grouped.add(id(model.answer_table))
    out: List[str] = []
    for n, p in model.named_parameters():
        if p.requires_grad and id(p) not in grouped:
            out.append(f"head.{n}")
    backbone_grouped = {id(p) for p in model.adapter.model.parameters()}
    for n, p in model.adapter.model.named_parameters():
        if p.requires_grad and id(p) not in backbone_grouped:
            out.append(f"backbone.{n}")
    return out


# ---------------------------------------------------------------------------
# 参数位移统计与门禁
# ---------------------------------------------------------------------------


def param_deltas(
    model: N3DQA,
    head_state0: Dict[str, torch.Tensor],
    backbone_state0: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    """逐参数计算训练前后位移（``L2`` 与最大绝对位移）。

    参数
    ----
    model : N3DQA
        训练后的模型。
    head_state0 : Dict[str, torch.Tensor]
        训练前 ``model.state_dict()`` 的深拷贝。
    backbone_state0 : Dict[str, torch.Tensor]
        训练前 ``model.adapter.model.state_dict()`` 的深拷贝
        （``N3DQA`` 不把骨干持有为子模块，故必须单独快照）。

    返回
    ----
    Dict[str, Dict[str, float]]
        参数名（``head.`` / ``backbone.`` 前缀）-> ``{"l2": ..., "max_abs": ...}``。

    **前缀口径的前提（审查收口，皋陶 info —— 前提必须显式登记）**
    ------------------------------------------------------------
    ``head.`` / ``backbone.`` 这套前缀**只在「``N3DQA`` 不把骨干持有为 ``nn.Module``
    子模块」这一现状下成立**：本模块的 ``N3DQA.adapter`` 是普通 Python 对象，
    骨干权重位于 ``model.adapter.model``，**不在** ``model.state_dict()`` /
    ``model.named_parameters()`` 里，因此两次遍历（``model.state_dict()`` 与
    ``model.adapter.model.state_dict()``）的键集合**天然不相交**，加前缀后不会重名，
    且与 :func:`make_optimizer` 现场枚举出的 ``grouped`` 名字口径一致。
    **若日后 ``adapter.model`` 被登记为 ``N3DQA`` 的子模块**，则骨干参数会同时出现在
    ``model.state_dict()`` 里，两次遍历的键会**重名**、``head.`` 前缀会把骨干参数误标为
    head 侧，且 ``make_optimizer`` / ``ungrouped_trainable_names`` / ``update_gate``
    的名字口径会失配（门禁会把真实更新的参数判为 ``missing_from_snapshot``）。
    届时必须同步改口径（例如改用 ``named_parameters(remove_duplicate=True)`` 或直接以
    ``id`` 为键），**不得**继续沿用当前前缀。
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, tensor in model.state_dict().items():
        old = head_state0.get(name)
        if old is None:
            continue
        diff = tensor.detach().cpu().to(torch.float32) - old.to(torch.float32)
        out[f"head.{name}"] = {
            "l2": float(diff.norm().item()),
            "max_abs": float(diff.abs().max().item()),
        }
    for name, tensor in model.adapter.model.state_dict().items():
        old = backbone_state0.get(name)
        if old is None:
            continue
        diff = tensor.detach().cpu().to(torch.float32) - old.to(torch.float32)
        out[f"backbone.{name}"] = {
            "l2": float(diff.norm().item()),
            "max_abs": float(diff.abs().max().item()),
        }
    return out


def update_gate(deltas: Dict[str, Dict[str, float]],
                trainable_names: Sequence[str]) -> Dict[str, Any]:
    """门禁：**可训参数的更新量 > 0**（否则该组结果判无效）。

    参数
    ----
    deltas : Dict[str, Dict[str, float]]
        逐参数位移（:func:`param_deltas` 的输出）。
    trainable_names : Sequence[str]
        本组各阶段实际进入优化器的参数名（并集，现场枚举）。

    返回
    ----
    Dict[str, Any]
        ``{"passed", "trainable_params", "updated_params", "total_update_l2",
        "max_update_abs", "missing_from_snapshot"}``。
    """
    total = 0.0
    max_abs = 0.0
    updated: List[str] = []
    never_updated: List[str] = []
    missing: List[str] = []
    for name in sorted(set(trainable_names)):
        item = deltas.get(name)
        if item is None:
            missing.append(name)
            continue
        total += float(item["l2"]) ** 2
        max_abs = max(max_abs, float(item["max_abs"]))
        if float(item["max_abs"]) > 0.0:
            updated.append(name)
        else:
            never_updated.append(name)
    return {
        "passed": bool(total > 0.0 and updated and not missing),
        "names_consistent": bool(not missing),
        "trainable_params": sorted(set(trainable_names)),
        "n_trainable_params": int(len(set(trainable_names))),
        "updated_params": updated,
        "n_updated_params": int(len(updated)),
        # 「声明可训却**全程零更新**」的参数名：门禁只要求"更新量 > 0"（按需求口径），
        # 但该名单必须显式报出 —— 它就是「该参数根本不在计算图上」的直接证据
        # （实测：``raw`` 口径下骨干参数恒零梯度，A3 组 4 个骨干参数全部落在这里）。
        "never_updated_params": never_updated,
        "total_update_l2": float(math.sqrt(total)),
        "max_update_abs": float(max_abs),
        "missing_from_snapshot": missing,
    }

# ---------------------------------------------------------------------------
# 表示几何指标
# ---------------------------------------------------------------------------


@torch.no_grad()
def representation_geometry(model: N3DQA, data: TrainingData,
                            device: torch.device) -> Dict[str, Any]:
    """表示层的类内/类间可分性（``gap`` / ``sigma``）与 1-NN 指标。

    口径（**显式固定**；转导诊断量，只作诊断，不参与任何模型选择）
    -------------------------------------------------------------
    * 评估池 = ``train_known + test_known``（本档 149 + 20 = 169 条）——
      目的是度量「已知答案集合整体在表示空间中的几何」，故质心由**同一池**逐类算出；
      该口径用到池内标签，是**转导**的，因此只当诊断量、不当判据；
    * 表示 ``rep`` = :meth:`N3DQA.query` 的输出（L2 归一化，恒为 ``D`` 维）；
    * ``within`` = 每个样本与其**自身类质心**的余弦；
      ``cross`` = 与**其余每个类质心**余弦的均值；
    * ``gap = mean(within) - mean(cross)``；``sigma = std(within, ddof=1)``；
    * ``nn1_top1_raw`` = **确定性原始特征**上的 1-NN top-1（``train_known`` ->
      ``test_known``）—— 锚点口径（风后实测 ``0.3000``）；
    * ``nn1_top1_repr`` = **表示空间**上的同口径 1-NN；
    * ``nn1_top1_centroid_repr`` = 表示空间上「最近类质心」判据的 top-1
      （质心取自 ``train_known``，候选空间**不含**「不相关」）。

    参数
    ----
    model : N3DQA
        模型（内部切 ``eval()``）。
    data : TrainingData
        数据装配结果。
    device : torch.device
        设备。

    返回
    ----
    Dict[str, Any]
        见 docstring 的指标清单。
    """
    model.eval()
    C = int(data.corpus.n_classes)
    idx = data.corpus.key_to_index()
    pool = list(data.splits.train_known) + list(data.splits.test_known)
    pool_feats = _feats(data.vectorizer, pool, device)
    pool_lab = torch.tensor([idx[r.answer_key] for r in pool], dtype=torch.long,
                            device=device)
    rep = model.query(pool_feats)

    cent = torch.zeros(C, int(rep.shape[1]), device=device)
    for c in range(C):
        cent[c] = F.normalize(rep[pool_lab == c].mean(dim=0), dim=0)
    scores = rep @ cent.transpose(0, 1)
    rows = torch.arange(len(pool_lab), device=device)
    gold = scores[rows, pool_lab]
    mask = torch.ones_like(scores, dtype=torch.bool, device=device)
    mask[rows, pool_lab] = False
    cross = (scores * mask).sum(dim=1) / mask.sum(dim=1)
    within_mean, within_sigma = _mean_std(gold)
    cross_mean, cross_sigma = _mean_std(cross)
    gap = within_mean - cross_mean

    def _top1(train_rep: torch.Tensor, train_lab: torch.Tensor,
              test_rep: torch.Tensor, test_lab: torch.Tensor) -> float:
        pred = (test_rep @ train_rep.transpose(0, 1)).argmax(dim=1)
        # 用 float64 求命中比例：该量是确定性的 ``命中数 / n``，用 float32 求和会引入
        # 1e-8 量级的舍入残差（实测 0.30000001192092896），锚点对账要求逐位一致。
        return float((train_lab[pred] == test_lab).to(torch.float64).mean().item())

    tr_recs, te_recs = list(data.splits.train_known), list(data.splits.test_known)
    tr_lab = torch.tensor([idx[r.answer_key] for r in tr_recs], dtype=torch.long,
                          device=device)
    te_lab = torch.tensor([idx[r.answer_key] for r in te_recs], dtype=torch.long,
                          device=device)
    raw_tr = F.normalize(_feats(data.vectorizer, tr_recs, device), dim=1)
    raw_te = F.normalize(_feats(data.vectorizer, te_recs, device), dim=1)
    repr_tr = model.query(raw_tr)
    repr_te = model.query(raw_te)
    cent_tr = torch.stack(
        [F.normalize(repr_tr[tr_lab == c].mean(dim=0), dim=0) for c in range(C)]
    )
    centroid_pred = (repr_te @ cent_tr.transpose(0, 1)).argmax(dim=1)

    return {
        "pool_size": int(len(pool)),
        "within_mean": float(within_mean),
        "within_sigma": float(within_sigma),
        "cross_mean": float(cross_mean),
        "cross_sigma": float(cross_sigma),
        "gap": float(gap),
        "gap_over_sigma": float(gap / within_sigma) if within_sigma > 0 else float("nan"),
        "nn1_top1_raw": _top1(raw_tr, tr_lab, raw_te, te_lab),
        "nn1_top1_repr": _top1(repr_tr, tr_lab, repr_te, te_lab),
        "nn1_top1_centroid_repr": float(
            (centroid_pred == te_lab).to(torch.float64).mean().item()
        ),
    }


# ---------------------------------------------------------------------------
# 对齐度与有效秩（本轮新增的两个诊断量）
# ---------------------------------------------------------------------------

#: 有效秩的**容差口径**（显式固定，不得随运行变化）：奇异值 ``σ_i`` 计入"有效"当且仅当
#: ``σ_i > σ_max * EFFECTIVE_RANK_REL_TOL``。取 ``1e-6`` 是 float32 下"数值零"与
#: "真实小奇异值"的经验分界（float32 的相对精度约 ``1.2e-7``，取 1e-6 留一个数量级余量）。
EFFECTIVE_RANK_REL_TOL: float = 1e-6


@torch.no_grad()
def effective_rank(
    matrix: torch.Tensor, *, rel_tol: float = EFFECTIVE_RANK_REL_TOL
) -> Dict[str, Any]:
    """矩阵的**有效秩 / 参与比 / 占比**（口径在此显式固定，禁止各处各写一套）。

    口径
    ----
    * 输入 ``matrix`` 为 ``[n, D]`` 的**原始**表示矩阵（**不中心化**、不归一化；
      中心化与否会改变奇异谱，故必须固定下来并在报告里写明）；
    * 对 ``matrix`` 做 SVD，取奇异值 ``σ``（降序，长度 ``min(n, D)``）；
    * ``rel_tol`` 容差口径：``σ_i > σ_max * rel_tol`` 的个数即**有效秩**
      （``σ_max`` 为最大奇异值；``σ_max == 0`` 时有效秩记 ``0``）；
      **默认 ``rel_tol = 1e-6``**，见 :data:`EFFECTIVE_RANK_REL_TOL`；
    * **参与比** = ``(Σσ)² / Σσ²``（谱的"有效维数"，对小幅奇异值敏感）；
    * **占 R^D 的百分比** = ``有效秩 / D``（D = ``matrix.shape[1]``）；
    * **占样本数 n 的百分比** = ``有效秩 / n``（``fc_dim=0`` 时读出是纯线性
      ``f = W_out @ h``，``rank`` 天然 ``<= min(n, N)``，该比值用于识别"被样本数卡住"）。

    参数
    ----
    matrix : torch.Tensor
        ``[n, D]`` 表示矩阵（``n >= 2``）。
    rel_tol : float
        相对容差（``> 0``）。

    返回
    ----
    Dict[str, Any]
        ``{"n", "dim", "rank", "rel_tol", "rank_ratio_dim", "rank_ratio_n",
        "participation_ratio", "sigma_max", "sigma_sum", "sigma_sq_sum",
        "sigma_head"}``；``sigma_head`` 为前 8 个奇异值（报告可读性）。
    """
    if matrix.dim() != 2:
        raise ValueError(f"effective_rank 需要 2D [n, D] 矩阵，当前 {tuple(matrix.shape)}")
    if float(rel_tol) <= 0.0:
        raise ValueError(f"rel_tol 必须 > 0，当前 {rel_tol}")
    x = matrix.detach().to(torch.float32).cpu()
    n, dim = int(x.shape[0]), int(x.shape[1])
    if n < 2:
        raise ValueError(f"effective_rank 需要 n >= 2 个样本，当前 n={n}")
    sigma = torch.linalg.svdvals(x)
    sigma_max = float(sigma[0].item()) if int(sigma.numel()) > 0 else 0.0
    if sigma_max <= 0.0:
        rank = 0
    else:
        rank = int((sigma > sigma_max * float(rel_tol)).sum().item())
    sigma_sum = float(sigma.sum().item())
    sigma_sq_sum = float((sigma ** 2).sum().item())
    return {
        "n": n,
        "dim": dim,
        "rank": rank,
        "rel_tol": float(rel_tol),
        "rank_ratio_dim": float(rank) / float(dim) if dim > 0 else 0.0,
        "rank_ratio_n": float(rank) / float(n) if n > 0 else 0.0,
        "participation_ratio": (
            float(sigma_sum ** 2 / sigma_sq_sum) if sigma_sq_sum > 0.0 else 0.0
        ),
        "sigma_max": sigma_max,
        "sigma_sum": sigma_sum,
        "sigma_sq_sum": sigma_sq_sum,
        "sigma_head": [float(v) for v in sigma[:8].tolist()],
    }


@torch.no_grad()
def alignment_degree(model: N3DQA, data: TrainingData,
                     device: torch.device) -> Dict[str, Any]:
    """**对齐度**：N3D 支路表示与「答案表各类质心」的平均余弦（逐类 + 总体）。

    口径（显式固定，只作诊断，不参与任何模型选择）
    --------------------------------------------
    * 评估池 = ``train_known + test_known``（与 :func:`representation_geometry` **同池**，
      便于两处数字互相对账）；
    * 表示 ``rep`` = :meth:`N3DQA.n3d_branch` 的输出（``concat`` 口径下已过无仿射
      ``LayerNorm``，机制 B 下再经投影头 -> 即"投影后"），逐行 L2 归一化；
    * **各类质心** = 模型自己的 ``answer_table``（``centroid`` 口径下由训练样本的**原始
      编码器特征**逐类质心确定性写入、L2 归一化、冻结为 buffer），逐行 L2 归一化后取
      前 ``C`` 行（答案类）；末位「不相关」行**单独报**，不并入总体均值；
    * 另外报告 4.1 要求的两个**对齐目标量**（都取"表示 vs 目标"的余弦均值）：
      目标 (i) ``self_encoder_feature`` = 本样本的编码器原始特征；
      目标 (ii) ``class_centroid`` = 本样本所属类**由训练样本特征算出的**质心。

    参数
    ----
    model : N3DQA
        模型（内部切 ``eval()``）。
    data : TrainingData
        数据装配结果。
    device : torch.device
        设备。

    返回
    ----
    Dict[str, Any]
        ``{"applicable", "reason", "pool_size", "dim", "projection_head",
        "cos_class_centroid", "cos_irrelevant_key", "cos_self_encoder_feature",
        "cos_own_class_centroid", "align_targets_reported", "index_convention"}``。
        ``applicable=False``（``head_input_mode="raw"``）时只返回 ``applicable`` /
        ``reason``：raw 档下 ``adapter.features`` **从不被调用**（现场登记的结构事实），
        此时"N3D 输出的对齐度"无定义，如实记 ``None`` 而不是拿 raw 特征冒充。
    """
    if str(model.config.head_input_mode) == "raw":
        return {
            "applicable": False,
            "reason": (
                "head_input_mode='raw'：adapter.features 从不被调用（结构性事实），"
                "不存在 N3D 输出，故对齐度无定义"
            ),
            "align_targets_reported": list(ALIGN_TARGETS),
        }
    model.eval()
    C = int(data.corpus.n_classes)
    idx = data.corpus.key_to_index()
    pool = list(data.splits.train_known) + list(data.splits.test_known)
    pool_feats = _feats(data.vectorizer, pool, device)
    lab = torch.tensor(
        [idx[r.answer_key] for r in pool], dtype=torch.long, device=device
    )
    rep = F.normalize(model.n3d_branch(pool_feats), dim=1)
    keys = F.normalize(model.answer_table.detach(), dim=1)  # [C+1, D]
    scores = rep @ keys[:C].transpose(0, 1)                 # [n, C]
    per_class: Dict[str, float] = {}
    for c in range(C):
        mask = lab == c
        if int(mask.sum().item()) == 0:
            continue
        per_class[str(data.corpus.answer_keys[c])] = float(scores[mask, c].mean().item())
    own = scores[torch.arange(int(lab.shape[0]), device=device), lab]
    irr = float((rep @ keys[C]).mean().item()) if int(keys.shape[0]) > C else None
    raw = F.normalize(pool_feats, dim=1)
    cent = torch.zeros(C, int(raw.shape[1]), device=device)
    for c in range(C):
        mask = lab == c
        if int(mask.sum().item()) == 0:
            continue
        cent[c] = F.normalize(raw[mask].mean(dim=0), dim=0)
    return {
        "applicable": True,
        "reason": None,
        "pool_size": int(len(pool)),
        "dim": int(rep.shape[1]),
        "projection_head": bool(model.proj is not None),
        "index_convention": (
            "各类质心取 answer_table 的前 C 行（答案类，逐类质心、冻结 buffer）；"
            "末位「不相关」行单独报，不并入总体均值"
        ),
        "cos_class_centroid": {
            "per_class": per_class,
            "mean": float(own.mean().item()),
            "min": float(own.min().item()),
            "max": float(own.max().item()),
        },
        "cos_irrelevant_key": irr,
        "cos_self_encoder_feature": float(
            (rep * raw).sum(dim=1).mean().item()
        ),
        "cos_own_class_centroid": float(
            (rep * cent[lab]).sum(dim=1).mean().item()
        ),
        "align_targets_reported": list(ALIGN_TARGETS),
    }


@torch.no_grad()
def spectral_diagnostics(model: N3DQA, data: TrainingData,
                         device: torch.device) -> Dict[str, Any]:
    """**有效秩 / 参与比 / 占比**：对 N3D 输出矩阵（与"投影后"表示）做 SVD。

    口径（显式固定）
    ---------------
    * 评估池 = ``train_known + test_known``（与 :func:`representation_geometry` /
      :func:`alignment_degree` **同池**）；
    * 两套矩阵**都报**：
      ``readout`` = :meth:`BackendAdapter.features` 的**原始 N3D 读出**（未 LayerNorm、
      未投影；现场实测的"纯线性读出"口径），
      ``aligned`` = :meth:`N3DQA.n3d_branch` 的输出（``concat`` 档已过无仿射 LayerNorm，
      机制 B 下再经投影头）；
    * 容差口径见 :func:`effective_rank`（``σ > σ_max · 1e-6``），**不中心化**；
    * ``head_input_mode="raw"`` 时不适用（raw 档下 ``adapter.features`` 不被调用），
      返回 ``applicable=False`` 与可读原因，**不用 raw 文本特征冒充 N3D 输出**。

    参数
    ----
    model, data, device : 同 :func:`alignment_degree`。

    返回
    ----
    Dict[str, Any]
        ``{"applicable", "reason", "pool_size", "dim", "readout", "aligned",
        "rel_tol", "note"}``；``readout`` / ``aligned`` 为 :func:`effective_rank` 的输出。
    """
    if str(model.config.head_input_mode) == "raw":
        return {
            "applicable": False,
            "reason": (
                "head_input_mode='raw'：adapter.features 从不被调用（结构性事实），"
                "不存在 N3D 输出矩阵，故有效秩无定义"
            ),
            "rel_tol": float(EFFECTIVE_RANK_REL_TOL),
        }
    model.eval()
    pool = list(data.splits.train_known) + list(data.splits.test_known)
    pool_feats = _feats(data.vectorizer, pool, device)
    readout = model.adapter.features(pool_feats)
    aligned = model.n3d_branch(pool_feats, readout=readout)
    return {
        "applicable": True,
        "reason": None,
        "pool_size": int(len(pool)),
        "dim": int(readout.shape[1]),
        "projection_head": bool(model.proj is not None),
        "rel_tol": float(EFFECTIVE_RANK_REL_TOL),
        "readout": effective_rank(readout),
        "aligned": effective_rank(aligned),
        "note": (
            "readout = adapter.features 的原始 N3D 读出（未 LayerNorm / 未投影）；"
            "aligned = n3d_branch（concat 档含无仿射 LayerNorm，机制 B 再含投影头）；"
            "两套都不中心化，容差 σ > σ_max·1e-6"
        ),
    }


# ---------------------------------------------------------------------------
# 逐 epoch 退化诊断
# ---------------------------------------------------------------------------


@torch.no_grad()
def epoch_diagnostics(
    model: N3DQA,
    data: TrainingData,
    ref_irrelevant: torch.Tensor,
    device: torch.device,
    *,
    loss_mean: float,
    n_batches: int,
    n_skipped: int,
    phase: str,
    epoch: int,
) -> Dict[str, Any]:
    """逐 epoch 记录「``q`` 与「不相关」键的余弦」轨迹（根因诊断，**不凭推测**）。

    记录量
    ------
    * ``cos_q_irr_key``：``q`` 与**当前答案表末位**（「不相关」键）余弦的均值；
      末位为零向量（``no_irr_centroid`` 口径）时该量**无定义**，如实置 ``None``；
    * ``cos_q_irr_ref``：``q`` 与**冻结参考方向**余弦的均值。参考方向 =
      训练侧 ``train_unknown`` 原始特征的归一化质心（训练前算出、全程冻结），
      因此该量在**所有组、所有 epoch** 都有定义，是根因假设
      「把所有 q 推向不相关质心」的稳定检验量。它与 ``cos_q_irr_key`` **同源但不是
      逐位相等**：后者的键取自答案表末位（同一批 ``train_unknown`` 的质心，由
      ``set_answer_table_from_centroids`` 写入 buffer），差异仅来自 float32 下求均值与
      归一化的运算顺序（现场实测 A2 的 40 个 epoch：恰好逐位相等 6 个、其余 34 个最大
      绝对差 ``1.788e-07``、方向恒为 ``key > ref``，4 位小数下 40/40 同值）；
    * ``train_refusal_frac``：训练样本上 ``argmax == C`` 的比例（退化度）；
    * ``train_top1_known``：训练侧 known 样本上 ``argmax == 金标`` 的比例；
    * ``gap`` / ``gap_over_sigma``：与 :func:`representation_geometry` 同池口径。

    参数
    ----
    model, data, ref_irrelevant, device : 同 :func:`run_group`。
    loss_mean, n_batches, n_skipped, phase, epoch : 本 epoch 的训练统计。

    返回
    ----
    Dict[str, Any]
        一行 epoch 诊断记录。
    """
    model.eval()
    C = int(data.corpus.n_classes)
    idx = data.corpus.key_to_index()
    items = list(data.splits.train_known) + list(data.splits.train_unknown)
    feats = _feats(data.vectorizer, items, device)
    labels = torch.tensor(
        [idx.get(r.answer_key, C) for r in items], dtype=torch.long, device=device
    )
    q = model.query(feats)
    logits = model.logits(feats, None, q=q)
    pred = logits.argmax(dim=1)
    known = labels != C
    irr_row = model.answer_table[C].detach()
    if float(irr_row.norm().item()) > 1e-12:
        cos_key: Optional[float] = float(
            (q @ F.normalize(irr_row, dim=0)).mean().item()
        )
    else:
        cos_key = None
    cos_ref = float((q @ ref_irrelevant).mean().item())
    # 与**金标类质心**的余弦（known 样本；由 logits / logit_scale 反推，无需额外前向）。
    # 该量与 ``cos_q_irr_key`` 一起定位「不相关」列胜出的几何原因。
    scale = float(model.logit_scale.detach().item())
    cos_gold: Optional[float] = None
    if bool(known.any()) and scale != 0.0:
        known_idx = torch.nonzero(known, as_tuple=False).squeeze(1)
        cos_gold = float(
            (logits[known_idx, labels[known_idx]] / scale).mean().item()
        )
    geo = representation_geometry(model, data, device)
    model.train()
    return {
        "phase": str(phase),
        "epoch": int(epoch),
        "loss": float(loss_mean),
        "batches": int(n_batches),
        "skipped_batches": int(n_skipped),
        "cos_q_irr_key": cos_key,
        "cos_q_irr_ref": cos_ref,
        "cos_q_gold_known": cos_gold,
        "train_refusal_frac": float((pred == C).to(torch.float32).mean().item()),
        "train_top1_known": float(
            ((pred == labels) & known).sum().item() / max(1, int(known.sum().item()))
        ),
        "gap": float(geo["gap"]),
        "gap_over_sigma": float(geo["gap_over_sigma"]),
        "logit_scale": float(model.logit_scale.detach().item()),
    }


def phase_plan(loss_mode: str, freeze_scope: str, epochs: int,
               stage1: int, stage2: int) -> List[Tuple[str, int, bool, bool, str]]:
    """把「修法 + 冻结范围 + epoch 预算」展开为阶段序列。

    每项 = ``(阶段名, epoch 数, train_repr, train_scale, 损失口径)``。

    * ``ce`` / ``no_irr_centroid``：单阶段（epoch 数 = ``epochs``）；
    * ``staged``：``repr`` 段（训表示、冻结标量）-> ``scale`` 段（冻表示、只训标量）；
    * ``supcon``：``repr`` 段用监督对比损失训表示 -> ``scale`` 段用交叉熵定标量。

    ``staged`` / ``supcon`` 的 ``repr`` 段 + ``scale`` 段总 epoch = ``stage1 + stage2``
    = 40，与 ``ce`` 档的 ``epochs`` 对齐（训练预算可比）。当冻结范围是
    ``logit_only``（无可训表示）时，表示段被显式跳过（无参数可训，阶段本身无定义）。
    """
    has_repr = freeze_scope in ("head", "head_backbone")
    if loss_mode in ("staged", "supcon"):
        head_loss = "ce" if loss_mode == "staged" else "supcon"
        plan: List[Tuple[str, int, bool, bool, str]] = []
        if has_repr:
            plan.append(("repr", int(stage1), True, False, head_loss))
        plan.append(("scale", int(stage2), False, True, "ce"))
        return plan
    kind = "ce" if loss_mode == "ce" else "ce_masked"
    return [("main", int(epochs), has_repr, True, kind)]

# ---------------------------------------------------------------------------
# 单组实验
# ---------------------------------------------------------------------------


def _sample_weights(targets: Sequence[int], data: TrainingData, cfg: TrainConfig,
                    device: torch.device) -> torch.Tensor:
    """逐样本权重（与 ``train._batch_tensors`` 同口径）。

    口径：``target == C``（「不相关」）取 ``cfg.unknown_class_weight``；
    其余取 ``1 / sqrt(该答案类在训练侧的样本数)`` —— 类别越小样本权重越大，
    抑制"大类吃掉全部注意力"。权重逐 batch 现场重算（不预先固化，避免与切分耦合）。
    """
    counts = data.class_frequencies()
    C = int(data.corpus.n_classes)
    weights: List[float] = []
    for t in targets:
        if int(t) == C:
            weights.append(float(cfg.unknown_class_weight))
        else:
            weights.append(1.0 / (max(1.0, float(counts[int(t)])) ** 0.5))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_group(
    group: ReprGroup,
    *,
    split_seed: int = BASE_SPLIT_SEED,
    train_seed: int = BASE_TRAIN_SEED,
    epochs: int = BASE_EPOCHS,
    batch_size: int = BASE_BATCH_SIZE,
    lr: float = BASE_LR,
    backbone_lr: float = BASE_BACKBONE_LR,
    logit_scale_init: float = BASE_LOGIT_SCALE_INIT,
    unknown_train_cap: int = BASE_UNKNOWN_TRAIN_CAP,
    max_classes: int = BASE_MAX_CLASSES,
    min_questions: int = BASE_MIN_QUESTIONS,
    test_every: int = BASE_TEST_EVERY,
    test_per_class: int = BASE_TEST_PER_CLASS,
    stage1_epochs: int = STAGE1_EPOCHS,
    stage2_epochs: int = STAGE2_EPOCHS,
    qa_cache_dir: str = "",
    text_dir: str = "",
    profile: str = DEFAULT_PROFILE,
    step2_bundle: Optional[Step2Bundle] = None,
    step2_topk: int = 5,
    step2_batch_size: int = 128,
) -> Dict[str, Any]:
    """跑一个对照组（固定切分 + 指定训练 seed），返回完整结果字典。

    训练循环与 :func:`n3d_qa_learn.train.run_training` **同构**：同样的数据装配、
    同样的 ``centroid`` 答案表口径、同样的冻结口径、同样的
    ``torch.randperm(generator)`` 顺序、同样的逐样本权重交叉熵。差别只在：
    本函数支持多阶段/多种损失，并且**不落盘任何产物**（等价 ``save=False``）。

    参数
    ----
    group : ReprGroup
        组定义。
    其余为实验超参（默认值即基线锚点档）。
    profile : str
        **特征档**名（见 :data:`FEATURE_PROFILES`）。默认 = 词面档（现状口径，
        行为与历史逐位一致）；``"bge-m3-1024"`` 即把步骤 1 的特征换成
        ``BAAI/bge-m3``（``D = hidden_size = 1024``）。
    step2_bundle : Optional[Step2Bundle]
        步骤 2 自检索共用件；给出时本组额外计算
        「同一 q 通路」的 ``Recall@1/@5``（主判据的第二半）。
    step2_topk / step2_batch_size : int
        步骤 2 的运行参数。

    返回
    ----
    Dict[str, Any]
        含 ``group`` / ``profile`` / ``config`` / ``split`` / ``split_qids`` /
        ``metric_step1`` / ``refusal`` / ``geo`` / ``step2`` / ``history`` /
        ``gate`` / ``shifts`` / ``param_deltas``。
    """
    prof = profile_by_name(profile)
    cfg = build_group_config(
        group,
        split_seed=split_seed,
        train_seed=train_seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        backbone_lr=backbone_lr,
        logit_scale_init=logit_scale_init,
        unknown_train_cap=unknown_train_cap,
        max_classes=max_classes,
        min_questions=min_questions,
        test_every=test_every,
        test_per_class=test_per_class,
        qa_cache_dir=qa_cache_dir,
        text_dir=text_dir,
    )
    device = _device()
    set_deterministic_seed(int(train_seed))
    data = build_training_data(cfg, encoder_config=prof.question)
    if int(data.vectorizer.dim) != int(prof.expect_dim):
        raise ValueError(
            f"特征档 {prof.name!r} 的步骤 1 实测维度 {data.vectorizer.dim} "
            f"与声明 expect_dim={prof.expect_dim} 不一致（拒绝静默错配）"
        )
    model, _adapter, _registry = _build_model(data, cfg)
    model.to(device)
    model.train()
    model.adapter.model.train(mode=bool(cfg.train_backbone))

    C = int(data.corpus.n_classes)
    idx = data.corpus.key_to_index()
    pool = list(data.splits.train_known) + list(data.splits.test_known)
    pool_feats = _feats(data.vectorizer, pool, device)
    train_items: List[Tuple[Any, int]] = [
        (r, idx[r.answer_key]) for r in data.splits.train_known
    ]
    train_items.extend((r, C) for r in data.splits.train_unknown)

    table_info = setup_answer_table(model, data, group.loss_mode, device)

    # ---- 类别不平衡的**可观测口径**：训练侧逐样本权重下的损失质量占比 ----
    counts = data.class_frequencies()
    known_mass = sum(math.sqrt(float(c)) for c in counts if float(c) > 0.0)
    unknown_mass = float(cfg.unknown_class_weight) * float(len(data.splits.train_unknown))
    weight_mass = {
        "known_samples": int(len(data.splits.train_known)),
        "unknown_samples": int(len(data.splits.train_unknown)),
        "known_weight_mass": float(known_mass),
        "unknown_weight_mass": float(unknown_mass),
        "known_share": float(known_mass / max(1e-12, known_mass + unknown_mass)),
        "unknown_share": float(unknown_mass / max(1e-12, known_mass + unknown_mass)),
        "class_counts_train": {
            k: int(v) for k, v in zip(
                data.corpus.answer_keys, [int(x) for x in counts]
            )
        },
    }

    # ---- 训练前快照（训练后量一律与快照比较，绝不断言初值） ----
    head_state0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    backbone_state0 = {
        k: v.detach().clone() for k, v in model.adapter.model.state_dict().items()
    }
    table0 = model.answer_table.detach().clone()
    with torch.no_grad():
        q0 = model.query(pool_feats)
        # 「训练是否改变预测」的**直接证据**：训练前的 argmax（与训练后逐位比较）
        pred0 = model.logits(pool_feats, None, q=q0).argmax(dim=1)
    logit_scale0 = float(model.logit_scale.detach().item())

    # ---- 冻结参考方向：「不相关」类的原始特征质心（训练前算出、全程冻结） ----
    if data.splits.train_unknown:
        unk_feats = _feats(data.vectorizer, data.splits.train_unknown, device)
        ref_irrelevant = F.normalize(unk_feats.mean(dim=0), dim=0)
    else:
        ref_irrelevant = torch.zeros(int(data.vectorizer.dim), device=device)

    plan = phase_plan(group.loss_mode, group.freeze_scope, int(epochs),
                      int(stage1_epochs), int(stage2_epochs))
    gen = torch.Generator().manual_seed(int(train_seed))
    history: List[Dict[str, Any]] = []
    trainable_names: List[str] = []
    ungrouped: List[str] = []
    epoch_no = 0
    # ---- 对齐项的批级累计（G7 用：对齐损失有限性 + 跳过计数可见）----
    align_on = bool(str(cfg.align_mode) != "off")
    align_stats: Dict[str, Any] = {
        "mode": str(cfg.align_mode),
        "lambda": float(cfg.align_lambda),
        "batches_total": 0,
        "batches_with_align": 0,
        "batches_align_skipped": 0,      # 对齐项本 batch 不适用（如批内无同类正样本）
        "batches_loss_skipped": 0,       # 基础损失本 batch 不适用（整个 batch 被跳过）
        "align_sum": 0.0,
        "align_max": None,
        "align_min": None,
        "align_nonfinite": 0,
        "supcon_sum": 0.0,
        "distill_self_sum": 0.0,
        "distill_centroid_sum": 0.0,
        "distill_terms": 0,
        "skipped_reasons": {},
    }
    for phase_name, n_epochs, train_repr, train_scale, loss_kind in plan:
        apply_freeze(model, group.freeze_scope, train_repr, train_scale)
        optimizer, names = make_optimizer(model, cfg)
        trainable_names.extend(names)
        ungrouped = ungrouped_trainable_names(model)
        for _ in range(int(n_epochs)):
            epoch_no += 1
            order = torch.randperm(len(train_items), generator=gen).tolist()
            total_loss = 0.0
            n_batches = 0
            n_skipped = 0
            for b0 in range(0, len(order), int(cfg.batch_size)):
                idxs = order[b0 : b0 + int(cfg.batch_size)]
                batch = [train_items[i] for i in idxs]
                records = [it[0] for it in batch]
                targets = [it[1] for it in batch]
                feats = _feats(data.vectorizer, records, device)
                tgt = torch.tensor(targets, dtype=torch.long, device=device)
                weight = _sample_weights(targets, data, cfg, device)
                optimizer.zero_grad(set_to_none=True)
                # 对齐开启时先算一次 N3D 支路表示，供 `query` 与对齐损失**共用**
                # （避免两次骨干前向；关闭对齐时该变量恒为 None，路径逐字符不变）。
                branch = model.n3d_branch(feats) if align_on else None
                q = model.query(feats) if branch is None else model.query(
                    feats, n3d_branch=branch
                )
                logits = model.logits(feats, None, q=q)
                outcome = batch_loss(
                    model, loss_kind, logits, q, tgt, weight, C,
                    feats=feats if align_on else None,
                    n3d_branch=branch,
                )
                parts = outcome["parts"]
                align_stats["batches_total"] += 1
                if align_on:
                    if bool(parts["align_applicable"]):
                        align_stats["batches_with_align"] += 1
                        value = float(parts["align"])
                        if not math.isfinite(value):
                            align_stats["align_nonfinite"] += 1
                        align_stats["align_sum"] += value
                        align_stats["align_max"] = (
                            value if align_stats["align_max"] is None
                            else max(float(align_stats["align_max"]), value)
                        )
                        align_stats["align_min"] = (
                            value if align_stats["align_min"] is None
                            else min(float(align_stats["align_min"]), value)
                        )
                        if parts["align_supcon"] is not None:
                            align_stats["supcon_sum"] += float(parts["align_supcon"])
                        if parts["align_distill_self"] is not None:
                            align_stats["distill_self_sum"] += float(
                                parts["align_distill_self"]
                            )
                            align_stats["distill_centroid_sum"] += float(
                                parts["align_distill_centroid"]
                            )
                            align_stats["distill_terms"] += 1
                    else:
                        align_stats["batches_align_skipped"] += 1
                        reason = str(parts["align_skipped_reason"])
                        align_stats["skipped_reasons"][reason] = (
                            int(align_stats["skipped_reasons"].get(reason, 0)) + 1
                        )
                loss = outcome["loss"]
                if loss is None:
                    n_skipped += 1
                    align_stats["batches_loss_skipped"] += 1
                    continue
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(
                        f"训练损失非有限值（组={group.name}, 阶段={phase_name}, "
                        f"epoch={epoch_no}, batch={n_batches}）："
                        f"loss={float(loss.detach().item())}"
                    )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().item())
                n_batches += 1
            if n_batches == 0:
                raise RuntimeError(
                    f"组 {group.name} 阶段 {phase_name} 没有任何 batch 被训练"
                )
            history.append(
                epoch_diagnostics(
                    model, data, ref_irrelevant, device,
                    loss_mean=total_loss / float(n_batches),
                    n_batches=n_batches, n_skipped=n_skipped,
                    phase=phase_name, epoch=epoch_no,
                )
            )
    # 对齐项的批级均值（只对「适用」的 batch 求均值，分母显式登记，避免把跳过当 0）
    n_align = int(align_stats["batches_with_align"])
    align_stats["align_mean"] = (
        float(align_stats["align_sum"]) / float(n_align) if n_align > 0 else None
    )
    align_stats["supcon_mean"] = (
        float(align_stats["supcon_sum"]) / float(n_align) if n_align > 0 else None
    )
    n_distill = int(align_stats["distill_terms"])
    align_stats["distill_self_mean"] = (
        float(align_stats["distill_self_sum"]) / float(n_distill) if n_distill else None
    )
    align_stats["distill_centroid_mean"] = (
        float(align_stats["distill_centroid_sum"]) / float(n_distill)
        if n_distill else None
    )
    # G7：开启对齐时，**实际参与过**的对齐项必须全部有限；否则该组判无效（抛错）
    if align_on and n_align > 0 and int(align_stats["align_nonfinite"]) > 0:
        raise RuntimeError(
            f"组 {group.name} 的对齐损失出现非有限值 "
            f"（{align_stats['align_nonfinite']} / {n_align} 个 batch）；该组判无效"
        )
    if align_on and n_align == 0:
        raise RuntimeError(
            f"组 {group.name} 开启了对齐（align_mode={cfg.align_mode}）但"
            f"**没有任何 batch** 的对齐项适用（跳过原因计数 = "
            f"{align_stats['skipped_reasons']}）；该组判无效"
        )

    model.eval()
    deltas = param_deltas(model, head_state0, backbone_state0)
    gate = update_gate(deltas, trainable_names)
    # 构造期不变量（不是训练后量）：现场枚举出的「有梯度但不进优化器」的参数名，
    # 必须落在**显式列出的**允许名单内（当前实现只允许 concat 口径的 mix_logit）。
    # 该断言防的正是"名字口径不一致导致的假阳性/假阴性"这一类缺陷。
    unexpected = sorted(set(ungrouped) - set(ALLOWED_UNGROUPED_TRAINABLE))
    if unexpected:
        raise RuntimeError(
            f"组 {group.name} 出现未登记的「requires_grad 但不在优化器内」参数："
            f"{unexpected}；允许名单 = {list(ALLOWED_UNGROUPED_TRAINABLE)}"
        )

    with torch.no_grad():
        q1 = model.query(pool_feats)
        pred1 = model.logits(pool_feats, None, q=q1).argmax(dim=1)
    argmax_changed = int((pred0 != pred1).sum().item())
    table1 = model.answer_table.detach().clone()
    step1 = evaluate_step1(model, data, k=3, device=str(device))
    refusal = evaluate_refusal(model, data, device=str(device))
    geo = representation_geometry(model, data, device)
    # ---- 本轮新增的两个诊断量（4.1 对齐度 / 4.2 有效秩·参与比）----
    align_deg = alignment_degree(model, data, device)
    spectrum = spectral_diagnostics(model, data, device)
    # ---- 结构代价（逐组现场枚举；不假设存在 W_in / W_out）----
    backbone_named = {n: int(p.numel()) for n, p in model.adapter.model.named_parameters()}
    structure_stats = {
        "structure": group.structure().as_dict(),
        "structure_is_default": bool(group.structure_is_default()),
        "topology": {k: float(v) for k, v in model.adapter.topology_stats().items()},
        "backbone_parameter_names": sorted(backbone_named.keys()),
        "backbone_parameters": backbone_named,
        "backbone_total": int(sum(backbone_named.values())),
        "head_total": int(sum(p.numel() for p in model.parameters())),
        "answer_table_shape": [int(x) for x in model.answer_table.shape],
    }
    step2_metric: Optional[Dict[str, Any]] = None
    if step2_bundle is not None:
        step2_metric = step2_recall_of_model(
            model, step2_bundle, topk=int(step2_topk), batch_size=int(step2_batch_size)
        )

    return {
        "group": group.as_dict(),
        "profile": prof.name,
        "profile_detail": prof.as_dict(),
        "config": {
            "split_seed": int(split_seed),
            "train_seed": int(train_seed),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "backbone_lr": float(backbone_lr),
            "logit_scale_init": float(logit_scale_init),
            "unknown_train_cap": int(unknown_train_cap),
            "max_classes": int(max_classes),
            "min_questions": int(min_questions),
            "test_every": int(test_every),
            "test_per_class": int(test_per_class),
            "stage1_epochs": int(stage1_epochs),
            "stage2_epochs": int(stage2_epochs),
            "supcon_temperature": float(SUPCON_TEMPERATURE),
            "device": str(device),
            # ---- 结构 / 对齐开关（本轮新增；默认档值即既有口径）----
            "structure": group.structure().as_dict(),
            "align_mode": str(group.align_mode),
            "align_lambda": float(group.align_lambda),
            "proj_dim": int(group.proj_dim),
            "proj_init": str(cfg.proj_init),
        },
        "split": data.splits.summary(),
        "split_qids": split_qids(data),
        "dim": int(data.vectorizer.dim),
        "n_classes": int(C),
        "split_seed_effective": int(
            cfg.seed if int(cfg.split_seed) < 0 else cfg.split_seed
        ),
        "train_seed_effective": int(cfg.seed),
        "answer_table": table_info,
        "weight_mass": weight_mass,
        "phase_plan": [
            {"phase": p[0], "epochs": int(p[1]), "train_repr": bool(p[2]),
             "train_scale": bool(p[3]), "loss_kind": p[4]}
            for p in plan
        ],
        "metric_step1": step1.as_dict(),
        "refusal": {k: float(v) for k, v in refusal.items()},
        "geo": geo,
        # ---- 本轮新增：对齐度 / 有效秩·参与比 / 对齐项批级统计 / 结构代价 ----
        "alignment_degree": align_deg,
        "spectrum": spectrum,
        "align_stats": align_stats,
        "structure_stats": structure_stats,
        "step2": step2_metric,
        "history": history,
        "gate": gate,
        "ungrouped_trainable_names": ungrouped,
        "shifts": {
            "q_shift_mean_l2": float((q1 - q0).norm(dim=1).mean().item()),
            "q_shift_max_l2": float((q1 - q0).norm(dim=1).max().item()),
            "answer_table_shift_l2": float((table1 - table0).norm().item()),
            "answer_table_row_C_norm_after": float(table1[C].norm().item()),
            "argmax_changed_count": argmax_changed,
            "argmax_pool_size": int(len(pool)),
            "argmax_changed_frac": float(argmax_changed) / float(len(pool)),
            "logit_scale_before": logit_scale0,
            "logit_scale_after": float(model.logit_scale.detach().item()),
            "logit_scale_delta": float(model.logit_scale.detach().item() - logit_scale0),
        },
        "param_deltas": deltas,
    }

# ---------------------------------------------------------------------------
# 实验编排与报告
# ---------------------------------------------------------------------------


def anchor_comparison(geo: Dict[str, Any]) -> Dict[str, Any]:
    """把基线组的几何指标与锚点逐项对账（容差显式给出，**现场标定**）。

    参数
    ----
    geo : Dict[str, Any]
        基线组的 :func:`representation_geometry` 输出。

    返回
    ----
    Dict[str, Any]
        ``{"rows": [...], "all_within_tolerance": bool}``；每行含
        ``metric / anchor / measured / delta / tolerance / within_tolerance``。
    """
    rows: List[Dict[str, Any]] = []
    for key in ("within_mean", "within_sigma", "cross_mean", "cross_sigma",
                "gap", "gap_over_sigma", "nn1_top1_raw"):
        measured = float(geo[key])
        anchor = float(ANCHOR[key])
        tol = float(ANCHOR_TOL[key])
        rows.append({
            "metric": key,
            "anchor": anchor,
            "measured": measured,
            "delta": measured - anchor,
            "tolerance": tol,
            "within_tolerance": bool(abs(measured - anchor) <= tol),
        })
    return {
        "rows": rows,
        "all_within_tolerance": bool(all(r["within_tolerance"] for r in rows)),
    }


def construction_probe(group: ReprGroup, dim: int) -> Dict[str, Any]:
    """**单组可构造性预检**（正式跑之前现场构造一次，失败原因原样登记）。

    检查内容（全部走**只读**上游 ``config`` / ``model`` 与代理层的
    :func:`recommended_config`，不使用任何记忆中的能力边界）：

    1. 结构开关经代理层装配成上游 ``Config`` 是否成功（非法组合由上游构造期抛错）；
    2. 上游 ``ThreeDNeuronSpace`` 是否可构造（连通性下限 / 尺寸窗口 / FCC 点数等校验）；
    3. ``N3DQAConfig`` 的对齐开关组合是否合法（``heads`` 的构造期不变量）。

    参数
    ----
    group : ReprGroup
        组定义。
    dim : int
        连接参数 ``D``。

    返回
    ----
    Dict[str, Any]
        ``{"group", "ok", "exc_type", "reason", "topology", "parameters",
        "backbone_parameter_names", "align_mode", "structure"}``；
        ``ok=False`` 时 ``exc_type`` / ``reason`` 必给出（可读原因）。
    """
    from .backends import BACKEND_NAMES, recommended_config

    payload: Dict[str, Any] = {
        "group": str(group.name),
        "ok": False,
        "exc_type": None,
        "reason": None,
        "structure": group.structure().as_dict(),
        "align_mode": str(group.align_mode),
        "align_lambda": float(group.align_lambda),
        "proj_dim": int(group.proj_dim),
        "dim": int(dim),
        "backend": "n3d_shape",
        "topology": None,
        "parameters": None,
        "backbone_parameter_names": None,
    }
    try:
        if "n3d_shape" not in BACKEND_NAMES:
            raise RuntimeError("后端 n3d_shape 未登记（BACKEND_NAMES 现场枚举异常）")
        cfg = recommended_config("n3d_shape", int(dim), group.structure())
        from n3d_shape.model import ThreeDNeuronSpace as _ShapeModel

        backbone = _ShapeModel(cfg)
        # 拓扑量**直接在现场构造出来的骨架上取**（不再二次构造，避免重复建图）
        axis = backbone.neuron_pos[:, int(backbone.flow_axis_index)]
        payload["topology"] = {
            "E": float(int(getattr(backbone, "num_edges"))),
            "K": float(torch.unique(axis).numel()),
            "S_in": float(int(getattr(backbone, "num_in_scope"))),
            "S_out": float(int(getattr(backbone, "num_out_scope"))),
        }
        payload["parameters"] = int(sum(p.numel() for p in backbone.parameters()))
        payload["backbone_parameter_names"] = sorted(
            n for n, _ in backbone.named_parameters()
        )
        # 对齐开关的构造期不变量（heads 侧）；不构造完整 N3DQA，避免多花一次拓扑构造
        N3DQAConfig(
            dim=int(dim),
            output_mode="index",
            head_input_mode=str(group.head_input_mode),
            answer_table_mode="centroid",
            align_mode=str(group.align_mode),
            align_lambda=float(group.align_lambda),
            proj_dim=int(group.proj_dim),
        )
        payload["ok"] = True
        return payload
    except Exception as exc:  # noqa: BLE001 - 失败原因必须原样登记，不得吞掉
        payload["exc_type"] = type(exc).__name__
        payload["reason"] = str(exc)
        return payload


def precheck_constructibility(
    groups: Sequence[ReprGroup], dim: int
) -> Dict[str, Any]:
    """**全组可构造性预检**：逐组现场构造，返回可跑集合与失败清单。

    口径（对应「构造失败 -> 判该组无效并跳过 + 报告显式登记失败原因」）
    --------------------------------------------------------------
    * **不硬失败终止**：任一组构造失败只把该组挡在 ``runnable`` 之外；
    * **绝不以成功状态落账**：失败组的信息（``exc_type`` + 可读 ``reason``）原样放进
      ``failures``，并由调用侧（``run_experiment`` / CLI）在报告与退出码中显式体现。

    参数
    ----
    groups : Sequence[ReprGroup]
        待预检的组。
    dim : int
        连接参数 ``D``。

    返回
    ----
    Dict[str, Any]
        ``{"dim", "probed", "runnable", "failures", "all_runnable"}``；
        ``failures`` 每项 = :func:`construction_probe` 的 ``ok=False`` 输出。
    """
    probed = [construction_probe(g, int(dim)) for g in groups]
    failures = [p for p in probed if not p["ok"]]
    return {
        "dim": int(dim),
        "probed": probed,
        "runnable": [p["group"] for p in probed if p["ok"]],
        "failures": failures,
        "all_runnable": bool(not failures),
        "rule": (
            "构造失败 -> 判该组无效并跳过（不硬失败终止），失败原因（异常类型 + 可读原因）"
            "在 construction_failures 中显式登记，绝不以成功状态落账"
        ),
    }


def run_experiment(
    *,
    group_names: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[ReprGroup]] = None,
    progress: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """顺序跑完整矩阵（或指定子集），执行**可构造性预检**与**切分一致性断言**。

    参数
    ----
    group_names : Optional[Sequence[str]]
        只跑这些组（``None`` = :data:`MATRIX` 的全部组）。
    groups : Optional[Sequence[ReprGroup]]
        **显式组定义**（本轮新增；用于 CLI 的"逐轴单换"自定义组）。
        给出时**优先于** ``group_names``。
    progress : Optional[Callable[[str], None]]
        进度回调（只接收一行文本；本模块自身不打印任何东西）。
    kwargs
        透传给 :func:`run_group` 的超参（含 ``profile`` / ``step2_bundle``）。

    返回
    ----
    Dict[str, Any]
        报告字典（含 ``profile`` / ``groups`` / ``construction_failures`` /
        ``split_identical`` / ``anchor_comparison`` / ``cost`` /
        ``axis_attribution``）。
    """
    if groups is not None:
        wanted = list(groups)
    else:
        names = list(group_names) if group_names else [g.name for g in MATRIX]
        wanted = [group_by_name(n) for n in names]
    prof = profile_by_name(str(kwargs.get("profile", DEFAULT_PROFILE)))
    # ---- 前置门禁：可构造性预检（G6；失败组跳过并显式登记）----
    precheck = precheck_constructibility(wanted, int(prof.expect_dim))
    if progress is not None:
        progress(
            f"[precheck] 可构造性预检：{len(precheck['runnable'])} / {len(wanted)} 组可构造"
            f"（失败 {len(precheck['failures'])} 组）"
        )
        for item in precheck["failures"]:
            progress(
                f"[precheck] 构造失败 -> 跳过：{item['group']} "
                f"({item['exc_type']}: {str(item['reason'])[:160]})"
            )
    runnable = [g for g in wanted if g.name in set(precheck["runnable"])]
    results: List[Dict[str, Any]] = []
    reference: Optional[Dict[str, List[str]]] = None
    ref_name = ""
    for group in runnable:
        if progress is not None:
            progress(
                f"[start] {group.name} (profile={prof.name}, "
                f"freeze_scope={group.freeze_scope}, "
                f"head_input_mode={group.head_input_mode}, loss_mode={group.loss_mode}, "
                f"shape={group.shape}, fc_dim={group.fc_dim}, N={group.N}, "
                f"geo_field={group.geo_field}, align_mode={group.align_mode}"
                + (f", λ={group.align_lambda}" if group.align_mode != "off" else "")
                + ")"
            )
        t0 = time.time()
        res = run_group(group, **kwargs)
        res["seconds"] = float(time.time() - t0)
        qids = res["split_qids"]
        if reference is None:
            reference = qids
            ref_name = group.name
        else:
            assert_same_split(reference, qids, group.name)
        results.append(res)
        if progress is not None:
            gate = res["gate"]
            s2 = res.get("step2") or {}
            s2_txt = (
                f" step2R@1={(s2.get('q_path') or {}).get('recall_at_1'):.4f}"
                if s2 else ""
            )
            ad = res.get("alignment_degree") or {}
            er = ((res.get("spectrum") or {}).get("aligned") or {}).get("rank")
            progress(
                f"[done ] {group.name} gap/σ={res['geo']['gap_over_sigma']:.4f} "
                f"macro={res['metric_step1']['macro_acc']:.4f}{s2_txt} "
                f"refusal={res['refusal']['refusal_rate']:.4f} "
                f"align_cos={(ad.get('cos_class_centroid') or {}).get('mean')} "
                f"eff_rank={er} "
                f"gate={'PASS' if gate['passed'] else 'FAIL'} "
                f"({res['seconds']:.1f}s)"
            )

    baseline = next((r for r in results if r["group"]["name"] == BASELINE_GROUP), None)
    cost: Optional[Dict[str, Any]] = None
    if baseline is not None:
        cost = dimension_cost(
            int(baseline["dim"]),
            n_train_known=int(baseline["split"]["train_known"]),
            n_train_unknown=int(baseline["split"]["train_unknown"]),
            n_classes=int(baseline["n_classes"]),
        )
    by_name = {r["group"]["name"]: r for r in results}
    return {
        "experiment": "n3d_qa_learn/exp_repr",
        "profile": prof.as_dict(),
        "matrix": [g.as_dict() for g in MATRIX],
        "dimensions": [dict(d) for d in DIMENSIONS],
        "anchor": dict(ANCHOR),
        "anchor_tolerance": dict(ANCHOR_TOL),
        "groups": results,
        "single_seed": {
            "split_seed": int(kwargs.get("split_seed", BASE_SPLIT_SEED)),
            "train_seed": int(kwargs.get("train_seed", BASE_TRAIN_SEED)),
            "statement": (
                "所有 Δ 均为单点差（同一切分、同一训练 seed 下两组之差），"
                "**不含跨 seed 训练随机性区间、无跨 seed 极差**"
            ),
        },
        "construction_precheck": {
            "dim": int(precheck["dim"]),
            "n_requested": int(len(wanted)),
            "n_runnable": int(len(precheck["runnable"])),
            "runnable": list(precheck["runnable"]),
            "rule": precheck["rule"],
        },
        "construction_failures": list(precheck["failures"]),
        "split_reference_group": str(ref_name),
        "split_identical": True,
        "split_digest": split_qids_digest(reference) if reference else {},
        "anchor_comparison": anchor_comparison(baseline["geo"]) if baseline else None,
        "cost": cost,
        "axis_attribution": axis_attribution(by_name),
        "step2_protocol": (
            dict(kwargs["step2_bundle"].evidence)
            if kwargs.get("step2_bundle") is not None else None
        ),
        "seconds_total": float(sum(float(r.get("seconds", 0.0)) for r in results)),
    }


def summarize(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把报告压成终端友好的逐组摘要行。"""
    out: List[Dict[str, Any]] = []
    for res in report["groups"]:
        geo = res["geo"]
        g = res["group"]
        st = g.get("structure") or {}
        out.append({
            "group": g["name"],
            "freeze_scope": g["freeze_scope"],
            "head_input_mode": g["head_input_mode"],
            "loss_mode": g["loss_mode"],
            "shape": st.get("shape"),
            "fc_dim": st.get("fc_dim"),
            "N": st.get("N"),
            "geo_field": st.get("geo_field"),
            "align_mode": g.get("align_mode"),
            "align_lambda": g.get("align_lambda"),
            "gap": round(float(geo["gap"]), 4),
            "sigma": round(float(geo["within_sigma"]), 4),
            "gap_over_sigma": round(float(geo["gap_over_sigma"]), 4),
            "nn1_top1_raw": round(float(geo["nn1_top1_raw"]), 4),
            "nn1_top1_repr": round(float(geo["nn1_top1_repr"]), 4),
            "macro_acc": round(float(res["metric_step1"]["macro_acc"]), 4),
            "top1_acc": round(float(res["metric_step1"]["top1_acc"]), 4),
            "refusal_rate": round(float(res["refusal"]["refusal_rate"]), 4),
            "q_shift_mean_l2": float(res["shifts"]["q_shift_mean_l2"]),
            "answer_table_shift_l2": float(res["shifts"]["answer_table_shift_l2"]),
            "argmax_changed_frac": float(res["shifts"]["argmax_changed_frac"]),
            "gate_passed": bool(res["gate"]["passed"]),
            # ---- 本轮新增诊断量的摘要列 ----
            "align_mode": g.get("align_mode"),
            "align_lambda": g.get("align_lambda"),
            "align_cos_mean": ((res.get("alignment_degree") or {}).get(
                "cos_class_centroid") or {}).get("mean"),
            "eff_rank_aligned": ((res.get("spectrum") or {}).get("aligned") or {}).get("rank"),
            "eff_rank_readout": ((res.get("spectrum") or {}).get("readout") or {}).get("rank"),
            "participation_ratio_aligned": ((res.get("spectrum") or {}).get(
                "aligned") or {}).get("participation_ratio"),
            "backbone_total": (res.get("structure_stats") or {}).get("backbone_total"),
        })
    return out


def _metric_row(res: Dict[str, Any]) -> Dict[str, Any]:
    """从一组结果里抽出**主判据与另报量**（缺项如实记 ``None``，不填 0 冒充）。"""
    s2 = res.get("step2") or None
    qpath = (s2 or {}).get("q_path") or {}
    det = (s2 or {}).get("deterministic") or {}
    ad = res.get("alignment_degree") or {}
    spec = res.get("spectrum") or {}
    aligned = spec.get("aligned") or {}
    readout = spec.get("readout") or {}
    cent = ad.get("cos_class_centroid") or {}
    return {
        "macro_acc": float(res["metric_step1"]["macro_acc"]),
        "top1_acc": float(res["metric_step1"]["top1_acc"]),
        "step2_recall_at_1": (float(qpath["recall_at_1"]) if "recall_at_1" in qpath else None),
        "step2_recall_at_5": (float(qpath["recall_at_5"]) if "recall_at_5" in qpath else None),
        "step2_det_recall_at_1": (float(det["recall_at_1"]) if "recall_at_1" in det else None),
        "step2_det_recall_at_5": (float(det["recall_at_5"]) if "recall_at_5" in det else None),
        "gap": float(res["geo"]["gap"]),
        "within_sigma": float(res["geo"]["within_sigma"]),
        "gap_over_sigma": float(res["geo"]["gap_over_sigma"]),
        "nn1_top1_raw": float(res["geo"]["nn1_top1_raw"]),
        "nn1_top1_repr": float(res["geo"]["nn1_top1_repr"]),
        "refusal_rate": float(res["refusal"]["refusal_rate"]),
        "argmax_changed_frac": float(res["shifts"]["argmax_changed_frac"]),
        "gate_passed": bool(res["gate"]["passed"]),
        "seconds": float(res.get("seconds", 0.0)),
        # ---- 本轮新增的四个诊断量（对齐度 / 有效秩 / 参与比；raw 档如实记 None）----
        "align_cos_mean": (float(cent["mean"]) if "mean" in cent else None),
        "align_cos_self_encoder_feature": (
            float(ad["cos_self_encoder_feature"])
            if ad.get("cos_self_encoder_feature") is not None else None
        ),
        "align_cos_own_class_centroid": (
            float(ad["cos_own_class_centroid"])
            if ad.get("cos_own_class_centroid") is not None else None
        ),
        "eff_rank_readout": (int(readout["rank"]) if "rank" in readout else None),
        "eff_rank_aligned": (int(aligned["rank"]) if "rank" in aligned else None),
        "participation_ratio_aligned": (
            float(aligned["participation_ratio"]) if "participation_ratio" in aligned else None
        ),
        "rank_ratio_dim_aligned": (
            float(aligned["rank_ratio_dim"]) if "rank_ratio_dim" in aligned else None
        ),
        "spectrum_applicable": bool(spec.get("applicable", False)),
    }


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """``b - a``（任一为 ``None`` 时返回 ``None``，不把缺失当 0）。"""
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _rel(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """相对增量 ``(b - a) / |a|``（``a`` 缺失或为 0 时返回 ``None``，**不编造分母**）。"""
    if a is None or b is None or float(a) == 0.0:
        return None
    return (float(b) - float(a)) / abs(float(a))


def run_comparison(
    *,
    profiles: Sequence[str] = (PROFILE_LEXICAL, PROFILE_SEMANTIC),
    group_names: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[ReprGroup]] = None,
    step2_product_dir: str = "",
    step2_pool_cap: int = 0,
    step2_topk: int = 5,
    progress: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """**同切分**下逐组对照两个特征档（词面 ``D=88`` vs ``bge-m3`` ``D=1024``）。

    口径（**必须显式**）
    ------------------
    * 两个档共用同一 ``split_seed`` 与同一 ``train_seed``，因而共用同一份切分；
      脚本对两档的四子集 qid 有序序列做 :func:`assert_same_split`（G1）；
    * 步骤 2 的文本行口径**与该档步骤 1 同族同维**（词面档 ``local-hash`` /
      语义档 ``bge-m3`` ``role=text_line``），以便「同一条 q 通路」可用；
      该口径与 ``step2_run`` 默认的 ``zh-bag``（``D=192``）**不同**，报告中显式标注；
    * 每个档的步骤 2 共用件（文本行向量化 + 冻结键表 + 纯特征参照下限）**只建一次**
      并复用于该档全部组。

    参数
    ----
    profiles : Sequence[str]
        要对照的档（默认两档）。
    group_names : Optional[Sequence[str]]
        只跑这些组（``None`` = 全部 8 组）。
    step2_product_dir : str
        ``n3d_qa`` 冻结产物目录（空 = 自动定位）。
    step2_pool_cap : int
        检索池行数上限（``0`` = 全量）。
    step2_topk : int
        ``Recall@k`` 的 k。
    progress : Optional[Callable[[str], None]]
        进度回调。
    kwargs
        其余透传给 :func:`run_group`。

    返回
    ----
    Dict[str, Any]
        ``{"experiment", "profiles", "runs", "split", "comparison",
        "attribution", "verdict", "cost", "seconds"}``。
    """
    profs = [profile_by_name(p) for p in profiles]
    if len(profs) < 2:
        raise ValueError("run_comparison 至少需要两个特征档")
    runs: Dict[str, Dict[str, Any]] = {}
    bundles: Dict[str, Step2Bundle] = {}
    for prof in profs:
        if progress is not None:
            progress(f"[profile] {prof.name} 构造步骤 2 共用件（{prof.label}）")
        bundle = build_step2_bundle(
            prof, product_dir=str(step2_product_dir), topk=int(step2_topk),
            pool_cap=int(step2_pool_cap),
        )
        bundles[prof.name] = bundle
        if progress is not None:
            progress(
                f"[profile] {prof.name} 步骤 2：池 {bundle.evidence['n_pool']} / "
                f"查询 {bundle.evidence['n_query']} / D={bundle.evidence['dim']} / "
                f"纯特征 Recall@1={bundle.det_baseline['recall_at_1']:.4f}"
            )
        runs[prof.name] = run_experiment(
            group_names=group_names, groups=groups, progress=progress,
            profile=prof.name, step2_bundle=bundle, **kwargs,
        )

    # ---- G1：跨档切分一致性 -------------------------------------------
    ref_name = profs[0].name
    ref_groups = {r["group"]["name"]: r["split_qids"] for r in runs[ref_name]["groups"]}
    cross_checks: List[Dict[str, Any]] = []
    for prof in profs[1:]:
        for res in runs[prof.name]["groups"]:
            gname = res["group"]["name"]
            if gname not in ref_groups:
                continue
            same = split_qids_equal(ref_groups[gname], res["split_qids"])
            cross_checks.append({
                "group": gname, "reference_profile": ref_name, "profile": prof.name,
                "identical": bool(same),
            })
            if not same:
                # 跨档切分不一致 -> 该对照判无效（与组内断言同口径）
                raise AssertionError(
                    f"[{prof.name}/{gname}] 跨特征档切分一致性断言失败：四子集 qid "
                    f"有序序列与 {ref_name}/{gname} 不逐位相同；该对照判无效"
                )

    # ---- 逐组对照 ------------------------------------------------------
    def _by_group(run: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {r["group"]["name"]: r for r in run["groups"]}

    rows: List[Dict[str, Any]] = []
    by_profile = {p.name: _by_group(runs[p.name]) for p in profs}
    names = [g["group"]["name"] for g in runs[ref_name]["groups"]]
    for gname in names:
        base = by_profile[ref_name].get(gname)
        if base is None:
            continue
        b_row = _metric_row(base)
        entry: Dict[str, Any] = {
            "group": gname,
            "freeze_scope": base["group"]["freeze_scope"],
            "head_input_mode": base["group"]["head_input_mode"],
            "loss_mode": base["group"]["loss_mode"],
            "reference_profile": ref_name,
            "reference": b_row,
            "others": {},
        }
        both_better_all = True
        for prof in profs[1:]:
            other = by_profile[prof.name].get(gname)
            if other is None:
                continue
            o_row = _metric_row(other)
            d_macro = _delta(b_row["macro_acc"], o_row["macro_acc"])
            d_recall = _delta(b_row["step2_recall_at_1"], o_row["step2_recall_at_1"])
            d_det = _delta(b_row["step2_det_recall_at_1"], o_row["step2_det_recall_at_1"])
            better_macro = bool(d_macro is not None and d_macro > 0.0)
            better_recall = bool(d_recall is not None and d_recall > 0.0)
            entry["others"][prof.name] = {
                "metrics": o_row,
                "delta_macro_acc": d_macro,
                "delta_step2_recall_at_1": d_recall,
                "delta_step2_det_recall_at_1": d_det,
                "delta_gap_over_sigma": _delta(b_row["gap_over_sigma"], o_row["gap_over_sigma"]),
                "delta_refusal_rate": _delta(b_row["refusal_rate"], o_row["refusal_rate"]),
                "delta_seconds": _delta(b_row["seconds"], o_row["seconds"]),
                "macro_better": better_macro,
                "step2_recall_better": better_recall,
                "step2_det_recall_better": bool(d_det is not None and d_det > 0.0),
                "both_better": bool(better_macro and better_recall),
                "gate_passed": bool(o_row["gate_passed"]),
            }
            both_better_all = both_better_all and bool(better_macro and better_recall)
        entry["all_profiles_both_better"] = bool(both_better_all)
        rows.append(entry)

    # ---- 归因分解（**换特征** 与 **打开表示训练** 不得合并）-------------
    attribution = _attribution(by_profile, profs)

    # ---- 逐轴归因（本轮新增：形状 / FC / N / geo / 对齐 / 载体 六节分开，不合并）----
    axis_by_profile = {
        p.name: runs[p.name]["axis_attribution"] for p in profs
    }
    # 每条轴变体在**每一档**是否都相对本档的公共基线同时更优（主判据口径）
    axis_rows: List[Dict[str, Any]] = []
    axis_names = list((axis_by_profile[ref_name].get("axes") or {}).keys())
    for dim_name in axis_names:
        ref_axis = (axis_by_profile[ref_name]["axes"] or {}).get(dim_name) or {}
        for ref_row in ref_axis.get("rows", []):
            gname = str(ref_row["group"])
            per_profile = {ref_name: {
                "all_both_better": bool(ref_row["both_better"]),
                "delta_macro_acc": ref_row["delta_macro_acc"],
                "delta_macro_acc_rel": ref_row["delta_macro_acc_rel"],
                "delta_step2_recall_at_1": ref_row["delta_step2_recall_at_1"],
                "delta_step2_recall_at_1_rel": ref_row["delta_step2_recall_at_1_rel"],
                "delta_step2_det_recall_at_1": ref_row["delta_step2_det_recall_at_1"],
                "delta_gap_over_sigma": ref_row["delta_gap_over_sigma"],
                "delta_refusal_rate": ref_row["delta_refusal_rate"],
                "delta_align_cos_mean": ref_row["delta_align_cos_mean"],
                "delta_eff_rank_aligned": ref_row["delta_eff_rank_aligned"],
                "present": True,
            }}
            missing: List[str] = []
            for prof in profs[1:]:
                ax = (axis_by_profile[prof.name].get("axes") or {}).get(dim_name) or {}
                hit = next((r for r in ax.get("rows", []) if r["group"] == gname), None)
                if hit is None:
                    missing.append(prof.name)
                    per_profile[prof.name] = {"present": False}
                    continue
                per_profile[prof.name] = {
                    "all_both_better": bool(hit["both_better"]),
                    "delta_macro_acc": hit["delta_macro_acc"],
                    "delta_macro_acc_rel": hit["delta_macro_acc_rel"],
                    "delta_step2_recall_at_1": hit["delta_step2_recall_at_1"],
                    "delta_step2_recall_at_1_rel": hit["delta_step2_recall_at_1_rel"],
                    "delta_step2_det_recall_at_1": hit["delta_step2_det_recall_at_1"],
                    "delta_gap_over_sigma": hit["delta_gap_over_sigma"],
                    "delta_refusal_rate": hit["delta_refusal_rate"],
                    "delta_align_cos_mean": hit["delta_align_cos_mean"],
                    "delta_eff_rank_aligned": hit["delta_eff_rank_aligned"],
                    "present": True,
                }
            axis_rows.append({
                "axis": dim_name,
                "varying_field": ref_axis.get("varying_field"),
                "group": gname,
                "per_profile": per_profile,
                "missing_profiles": missing,
                "all_profiles_both_better": bool(
                    not missing
                    and all(v.get("all_both_better") for v in per_profile.values())
                ),
            })
    axis_attribution_report = {
        "sep": (
            "逐轴归因：形状 / FC / 容量 N / 几何权重场 / 对齐 / 载体 六节分开报，"
            "**禁止合并归因**；每节都相对同一档内的公共基线 "
            f"{STRUCT_BASELINE_GROUP}（逐轴单换，非全交叉）"
        ),
        "baseline": STRUCT_BASELINE_GROUP,
        "per_profile": axis_by_profile,
        "rows": axis_rows,
        "criterion": (
            "主判据 = 同档内 步骤 1 macro 与 步骤 2 自检索 Recall@1 **同时**优于公共基线"
            f"（{STRUCT_BASELINE_GROUP}）；以相对 Δ 为主，绝对量作可用性附加判定"
        ),
        "groups_axis_both_better_all_profiles": sorted(
            {r["group"] for r in axis_rows if r["all_profiles_both_better"]}
        ),
        "groups_axis_not_both_better": sorted(
            {r["group"] for r in axis_rows if not r["all_profiles_both_better"]}
        ),
    }
    # 构造失败清单（逐档聚合；**不允许**被吞掉）
    construction_failures = {
        p.name: list(runs[p.name].get("construction_failures") or []) for p in profs
    }

    verdict = {
        "criterion": "步骤 1 macro 与步骤 2 自检索 Recall@1 **同时**高于参照档",
        "reference_profile": ref_name,
        "n_groups": int(len(rows)),
        "groups_both_better": [r["group"] for r in rows if r["all_profiles_both_better"]],
        "groups_not_both_better": [r["group"] for r in rows
                                   if not r["all_profiles_both_better"]],
        "any_group_both_better": bool(any(r["all_profiles_both_better"] for r in rows)),
        "all_groups_both_better": bool(rows) and all(
            r["all_profiles_both_better"] for r in rows
        ),
        "note": (
            "本判据是**逐组**的联合条件；同时给出纯特征（不经 N3D）Recall@1 的"
            "同组增量 delta_step2_det_recall_at_1，用于把「特征好坏」与「q 通路好坏」分开看"
        ),
    }

    return {
        "experiment": "n3d_qa_learn/exp_repr/feature-comparison",
        "profiles": [p.as_dict() for p in profs],
        "runs": runs,
        "step2_bundles": {k: dict(v.evidence) for k, v in bundles.items()},
        "split": {
            "cross_profile_checks": cross_checks,
            "identical_across_profiles": bool(all(c["identical"] for c in cross_checks))
            if cross_checks else True,
            "digest": {p.name: runs[p.name]["split_digest"] for p in profs},
        },
        "comparison": rows,
        "attribution": attribution,
        "axis_attribution": axis_attribution_report,
        "construction_failures": construction_failures,
        "verdict": verdict,
        "cost": {p.name: runs[p.name]["cost"] for p in profs},
        "seconds": {
            "per_profile_total": {p.name: float(runs[p.name]["seconds_total"]) for p in profs},
            "per_group": {
                p.name: {r["group"]["name"]: float(r["seconds"]) for r in runs[p.name]["groups"]}
                for p in profs
            },
            "grand_total": float(sum(runs[p.name]["seconds_total"] for p in profs)),
        },
        "config": {
            "split_seed": int(kwargs.get("split_seed", BASE_SPLIT_SEED)),
            "train_seed": int(kwargs.get("train_seed", BASE_TRAIN_SEED)),
            "epochs": int(kwargs.get("epochs", BASE_EPOCHS)),
            "batch_size": int(kwargs.get("batch_size", BASE_BATCH_SIZE)),
        },
        "single_seed": {
            "split_seed": int(kwargs.get("split_seed", BASE_SPLIT_SEED)),
            "train_seed": int(kwargs.get("train_seed", BASE_TRAIN_SEED)),
            "statement": (
                "所有 Δ 均为单点差（同一切分、同一训练 seed 下两组之差），"
                "**不含跨 seed 训练随机性区间、无跨 seed 极差**"
            ),
        },
    }


def axis_attribution(by_group: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """把本轮**五条轴**的贡献逐条分开量化（**禁止合并归因**）。

    口径
    ----
    * 参照系 = :data:`STRUCT_BASELINE_GROUP`（``S0_base``：``concat`` +
      ``head_backbone`` + 结构默认 + 对齐关闭）；
    * 每条轴 = :data:`DIMENSIONS` 里本轮的 ``d_shape`` / ``e_fc_dim`` /
      ``f_capacity_N`` / ``g_geo_field`` / ``h_align`` / ``i_carrier`` 之一，
      **逐轴单换**（每次只动该轴字段，其余等于 ``S0_base``）；
    * 每条变体给出与 ``S0_base`` 的 Δ（macro / 步骤 2 自检索 R@1 / 纯特征 detR@1 /
      ``gap/σ`` / refusal / 对齐度 / 有效秩 / 骨干参数量），``None`` 表示缺项
      （如实记 ``None``，**不填 0 冒充**）。

    参数
    ----
    by_group : Dict[str, Dict[str, Any]]
        组名 -> 该组结果。

    返回
    ----
    Dict[str, Any]
        ``{"sep", "baseline", "axes": {...}, "sections": ["形状", "FC", "N", "geo",
        "对齐", "载体"]}``。
    """
    base = by_group.get(STRUCT_BASELINE_GROUP)
    out: Dict[str, Any] = {
        "sep": (
            "形状 / FC / 容量 N / 几何权重场 / 对齐 / 载体 —— 六节分开报，"
            "**禁止合并归因**；每节都相对同一个公共基线 "
            f"{STRUCT_BASELINE_GROUP}（逐轴单换，非全交叉）"
        ),
        "baseline": str(STRUCT_BASELINE_GROUP),
        "axes": {},
    }
    if base is None:
        out["note"] = f"公共基线组 {STRUCT_BASELINE_GROUP} 不在本次运行内，无法给出逐轴 Δ"
        return out
    b_row = _metric_row(base)
    for dim in DIMENSIONS:
        dim_name = str(dim["dim"])
        if str(dim["base"]) != STRUCT_BASELINE_GROUP:
            continue
        rows: List[Dict[str, Any]] = []
        for gname in dim["groups"]:
            if gname == STRUCT_BASELINE_GROUP:
                continue
            other = by_group.get(str(gname))
            if other is None:
                continue
            o_row = _metric_row(other)
            rows.append({
                "group": str(gname),
                "varying_field": str(dim["varying_field"]),
                "structure": other["group"].get("structure"),
                "align_mode": other["group"].get("align_mode"),
                "align_lambda": other["group"].get("align_lambda"),
                "delta_macro_acc": _delta(b_row["macro_acc"], o_row["macro_acc"]),
                # 「相对 Δ 为主」：主判据以相对增量为主，绝对量只作可用性附加判定
                "delta_macro_acc_rel": _rel(
                    b_row["macro_acc"], o_row["macro_acc"]
                ),
                "delta_step2_recall_at_1": _delta(
                    b_row["step2_recall_at_1"], o_row["step2_recall_at_1"]
                ),
                "delta_step2_recall_at_1_rel": _rel(
                    b_row["step2_recall_at_1"], o_row["step2_recall_at_1"]
                ),
                "delta_step2_det_recall_at_1": _delta(
                    b_row["step2_det_recall_at_1"], o_row["step2_det_recall_at_1"]
                ),
                "delta_gap_over_sigma": _delta(
                    b_row["gap_over_sigma"], o_row["gap_over_sigma"]
                ),
                "delta_refusal_rate": _delta(
                    b_row["refusal_rate"], o_row["refusal_rate"]
                ),
                "delta_align_cos_mean": _delta(
                    b_row["align_cos_mean"], o_row["align_cos_mean"]
                ),
                "delta_eff_rank_aligned": _delta(
                    b_row["eff_rank_aligned"], o_row["eff_rank_aligned"]
                ),
                "delta_eff_rank_readout": _delta(
                    b_row["eff_rank_readout"], o_row["eff_rank_readout"]
                ),
                "delta_participation_ratio_aligned": _delta(
                    b_row["participation_ratio_aligned"],
                    o_row["participation_ratio_aligned"],
                ),
                "backbone_total": (other.get("structure_stats") or {}).get("backbone_total"),
                "macro_better": bool(
                    (d := _delta(b_row["macro_acc"], o_row["macro_acc"])) is not None and d > 0.0
                ),
                "step2_recall_better": bool(
                    (d := _delta(b_row["step2_recall_at_1"], o_row["step2_recall_at_1"]))
                    is not None and d > 0.0
                ),
                "both_better": bool(
                    (_delta(b_row["macro_acc"], o_row["macro_acc"]) or 0.0) > 0.0
                    and (_delta(b_row["step2_recall_at_1"], o_row["step2_recall_at_1"])
                         or 0.0) > 0.0
                ),
                "gate_passed": bool(o_row["gate_passed"]),
                "base_metrics": b_row,
                "variant_metrics": o_row,
            })
        out["axes"][dim_name] = {
            "varying_field": str(dim["varying_field"]),
            "base": str(dim["base"]),
            "groups": list(dim["groups"]),
            "rows": rows,
        }
    out["sections"] = ["形状", "FC", "容量 N", "几何权重场", "对齐", "载体"]
    return out


def _attribution(
    by_profile: Dict[str, Dict[str, Dict[str, Any]]],
    profs: Sequence[FeatureProfile],
) -> Dict[str, Any]:
    """把「**换特征**」与「**打开表示训练**」两种贡献**分开**量化（不得合并归因）。

    口径
    ----
    * **只换特征**：同一组配置下，参照档 vs 其它档（``A1_baseline`` 组即
      ``logit_only/raw/ce``，可训参数只有 ``logit_scale``，故该差异**只**来自特征）；
    * **只打开表示训练**：同一档内，``A1_baseline`` vs ``A2_head`` / ``A3_head_backbone``
      （特征不变，只把 ``q`` 头/骨干放进优化器）；
    * **两者叠加**：其它档的 ``A2``/``A3`` vs 参照档的 ``A1``。

    返回
    ----
    Dict[str, Any]
        三个分节 + 一句口径声明。
    """
    ref = profs[0].name
    out: Dict[str, Any] = {
        "sep": "换特征 / 打开表示训练 / 两者叠加 —— 三节分开报，**禁止合并归因**",
        "feature_only": [],
        "training_only": [],
        "combined": [],
    }
    a1, a2, a3 = "A1_baseline", "A2_head", "A3_head_backbone"

    # 只换特征：A1_baseline 同组、跨档
    if a1 in by_profile.get(ref, {}):
        base = _metric_row(by_profile[ref][a1])
        for prof in profs[1:]:
            other = by_profile.get(prof.name, {}).get(a1)
            if other is None:
                continue
            o = _metric_row(other)
            out["feature_only"].append({
                "group": a1,
                "reference_profile": ref,
                "profile": prof.name,
                "macro_acc": {"reference": base["macro_acc"], "other": o["macro_acc"],
                              "delta": _delta(base["macro_acc"], o["macro_acc"])},
                "step2_recall_at_1": {
                    "reference": base["step2_recall_at_1"], "other": o["step2_recall_at_1"],
                    "delta": _delta(base["step2_recall_at_1"], o["step2_recall_at_1"])},
                "step2_det_recall_at_1": {
                    "reference": base["step2_det_recall_at_1"],
                    "other": o["step2_det_recall_at_1"],
                    "delta": _delta(base["step2_det_recall_at_1"], o["step2_det_recall_at_1"])},
                "gap_over_sigma": {"reference": base["gap_over_sigma"],
                                   "other": o["gap_over_sigma"],
                                   "delta": _delta(base["gap_over_sigma"],
                                                   o["gap_over_sigma"])},
                "refusal_rate": {"reference": base["refusal_rate"], "other": o["refusal_rate"],
                                 "delta": _delta(base["refusal_rate"], o["refusal_rate"])},
                "note": "A1_baseline 只训 logit_scale（正标量，不改 argmax），故该差异只来自特征",
            })

    # 只打开表示训练：同档内 A1 -> A2 / A3
    for prof in profs:
        packs = by_profile.get(prof.name, {})
        if a1 not in packs:
            continue
        base = _metric_row(packs[a1])
        for gname in (a2, a3):
            if gname not in packs:
                continue
            o = _metric_row(packs[gname])
            out["training_only"].append({
                "profile": prof.name,
                "from_group": a1,
                "to_group": gname,
                "macro_acc": {"reference": base["macro_acc"], "other": o["macro_acc"],
                              "delta": _delta(base["macro_acc"], o["macro_acc"])},
                "step2_recall_at_1": {
                    "reference": base["step2_recall_at_1"], "other": o["step2_recall_at_1"],
                    "delta": _delta(base["step2_recall_at_1"], o["step2_recall_at_1"])},
                "gap_over_sigma": {"reference": base["gap_over_sigma"],
                                   "other": o["gap_over_sigma"],
                                   "delta": _delta(base["gap_over_sigma"],
                                                   o["gap_over_sigma"])},
                "refusal_rate": {"reference": base["refusal_rate"], "other": o["refusal_rate"],
                                 "delta": _delta(base["refusal_rate"], o["refusal_rate"])},
                "note": "特征不变，只把 q 头（/ 骨干）放进优化器",
            })

    # 两者叠加：其它档的 A2/A3 vs 参照档 A1
    if a1 in by_profile.get(ref, {}):
        base = _metric_row(by_profile[ref][a1])
        for prof in profs[1:]:
            for gname in (a2, a3):
                other = by_profile.get(prof.name, {}).get(gname)
                if other is None:
                    continue
                o = _metric_row(other)
                out["combined"].append({
                    "reference": f"{ref}/{a1}",
                    "profile": prof.name,
                    "group": gname,
                    "macro_acc": {"reference": base["macro_acc"], "other": o["macro_acc"],
                                  "delta": _delta(base["macro_acc"], o["macro_acc"])},
                    "step2_recall_at_1": {
                        "reference": base["step2_recall_at_1"],
                        "other": o["step2_recall_at_1"],
                        "delta": _delta(base["step2_recall_at_1"],
                                        o["step2_recall_at_1"])},
                    "gap_over_sigma": {"reference": base["gap_over_sigma"],
                                       "other": o["gap_over_sigma"],
                                       "delta": _delta(base["gap_over_sigma"],
                                                       o["gap_over_sigma"])},
                    "refusal_rate": {"reference": base["refusal_rate"],
                                     "other": o["refusal_rate"],
                                     "delta": _delta(base["refusal_rate"], o["refusal_rate"])},
                })
    return out


# ---------------------------------------------------------------------------
# 报告（对照版）
# ---------------------------------------------------------------------------


def render_comparison_markdown(report: Dict[str, Any]) -> str:
    """把**特征档对照**报告渲染为 Markdown。"""
    lines: List[str] = []
    lines.append("# n3d_qa_learn 特征档对照实验（exp_repr × 词面 D=88 vs bge-m3 D=1024）")
    lines.append("")
    lines.append("> 全部数字由 `python -m n3d_qa_learn.exp_repr_run compare` 现场产出；")
    lines.append("> 产物只落 `checkpoints/qa_learn/_verify/exp_repr/`，训练路径不落盘任何 zip。")
    lines.append("")
    cfg = report.get("config", {})
    lines.append("## 固定口径")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("| --- | --- |")
    for key in ("split_seed", "train_seed", "epochs", "batch_size"):
        lines.append(f"| {key} | {cfg.get(key)} |")
    lines.append("")
    for prof in report.get("profiles", []):
        lines.append(
            f"- **{prof['name']}**（{prof['label']}）D={prof['expect_dim']}；"
            f"文本行口径 = `{prof['text_line'].get('name')}` "
            f"max_length={prof['text_line'].get('max_length')}"
        )
    lines.append("")
    sp = report.get("split", {})
    lines.append(
        "G1 切分一致性：组内断言**全部通过**（`run_experiment` 内 `assert_same_split`）；"
        "跨档断言**" + ("全部通过" if sp.get("identical_across_profiles") else "未通过")
        + f"**（{len(sp.get('cross_profile_checks', []))} 项逐组比对）"
    )
    lines.append("")

    lines.append("## 逐组对照（主判据：macro 与 步骤 2 自检索 Recall@1 同时更优）")
    lines.append("")
    prof_names = [p["name"] for p in report.get("profiles", [])]
    hdr = "| 组 | freeze_scope | head_input_mode | loss_mode | shape | fc_dim | N | geo | 对齐(λ) |"
    for p in prof_names:
        hdr += f" macro({p}) | R@1({p}) | detR@1({p}) | refusal({p}) | gap/σ({p}) | s({p}) |"
    hdr += " Δmacro | ΔR@1 | ΔdetR@1 | 同时更优 |"
    lines.append(hdr)
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |" + \
          " --- |" * (6 * len(prof_names)) + " --- | --- | --- | --- |"
    lines.append(sep)
    ref = report["verdict"]["reference_profile"]
    run_by_profile = report.get("runs") or {}
    for row in report.get("comparison", []):
        m = row["reference"]
        meta = _structure_cell((run_by_profile.get(ref, {}) or {}).get("groups"), row["group"])
        line = (f"| {row['group']} | {row['freeze_scope']} | {row['head_input_mode']} | "
                f"{row['loss_mode']} | {meta} |")
        line += (f" {m['macro_acc']:.4f} | {_fmt(m['step2_recall_at_1'])} | "
                 f"{_fmt(m['step2_det_recall_at_1'])} | {m['refusal_rate']:.4f} | "
                 f"{m['gap_over_sigma']:.4f} | {m['seconds']:.1f} |")
        for pname in prof_names[1:]:
            o = (row.get("others") or {}).get(pname)
            if o is None:
                line += " — | — | — | — | — | — |"
                continue
            om = o["metrics"]
            line += (f" {om['macro_acc']:.4f} | {_fmt(om['step2_recall_at_1'])} | "
                     f"{_fmt(om['step2_det_recall_at_1'])} | {om['refusal_rate']:.4f} | "
                     f"{om['gap_over_sigma']:.4f} | {om['seconds']:.1f} |")
            line += (f" {_fmt_d(o['delta_macro_acc'])} | {_fmt_d(o['delta_step2_recall_at_1'])} | "
                     f"{_fmt_d(o['delta_step2_det_recall_at_1'])} | "
                     f"{'是' if o['both_better'] else '否'} |")
        lines.append(line)
    lines.append("")
    v = report.get("verdict", {})
    lines.append(
        f"**主判据结论**：同时更优的组 = `{v.get('groups_both_better')}`；"
        f"未同时更优的组 = `{v.get('groups_not_both_better')}`；"
        f"`any_group_both_better = {v.get('any_group_both_better')}`；"
        f"`all_groups_both_better = {v.get('all_groups_both_better')}`。"
    )
    lines.append("")

    # ---- 构造失败清单（绝不吞掉；没有失败时也显式写"0 项"）----
    lines.append("## 可构造性预检与失败清单（G6）")
    lines.append("")
    lines.append(
        "预检规则：构造失败 -> 判该组无效并跳过（不硬失败终止），"
        "失败原因（异常类型 + 可读原因）在 `construction_failures` 中显式登记，"
        "**绝不以成功状态落账**"
    )
    lines.append("")
    lines.append("| 档 | 请求组数 | 可构造 | 构造失败 |")
    lines.append("| --- | --- | --- | --- |")
    for pname, fails in (report.get("construction_failures") or {}).items():
        run = (report.get("runs") or {}).get(pname) or {}
        pre = run.get("construction_precheck") or {}
        lines.append(
            f"| {pname} | {pre.get('n_requested')} | {pre.get('n_runnable')} | "
            f"{len(fails)} |"
        )
    lines.append("")
    any_fail = False
    for pname, fails in (report.get("construction_failures") or {}).items():
        for item in fails:
            any_fail = True
            lines.append(
                f"* **{pname} / {item['group']}**：`{item['exc_type']}` —— "
                f"{str(item['reason'])[:400]}"
            )
    if not any_fail:
        lines.append("* 本批**无**构造失败组（0 项）。")
    lines.append("")

    # ---- 逐轴归因（形状 / FC / N / geo / 对齐 / 载体，六节分开）----
    ax = report.get("axis_attribution") or {}
    lines.append("## 逐轴归因（**形状 / FC / 容量 N / 几何权重场 / 对齐 / 载体** 六节分开，禁止合并）")
    lines.append("")
    lines.append(str(ax.get("sep", "")))
    lines.append("")
    lines.append("| 轴 | 组 | 档 | Δmacro | Δmacro(相对) | ΔR@1 | ΔR@1(相对) | ΔdetR@1 | Δgap/σ | Δrefusal | Δ对齐度 | Δ有效秩 | 同时更优 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in ax.get("rows", []):
        for pname, cell in (r.get("per_profile") or {}).items():
            if not cell.get("present"):
                lines.append(f"| {r['axis']} | {r['group']} | {pname} | — | — | — | — | — | — | — | — | — | 缺失 |")
                continue
            lines.append(
                f"| {r['axis']} | {r['group']} | {pname} | "
                f"{_fmt_d(cell['delta_macro_acc'])} | {_fmt_pct(cell['delta_macro_acc_rel'])} | "
                f"{_fmt_d(cell['delta_step2_recall_at_1'])} | "
                f"{_fmt_pct(cell['delta_step2_recall_at_1_rel'])} | "
                f"{_fmt_d(cell.get('delta_step2_det_recall_at_1'))} | "
                f"{_fmt_d(cell.get('delta_gap_over_sigma'))} | "
                f"{_fmt_d(cell.get('delta_refusal_rate'))} | "
                f"{_fmt_d(cell.get('delta_align_cos_mean'))} | "
                f"{_fmt_d(cell.get('delta_eff_rank_aligned'))} | "
                f"{'是' if cell['all_both_better'] else '否'} |"
            )
    lines.append("")
    lines.append(
        "逐轴主判据（各档都相对公共基线同时更优）："
        f"通过 = `{ax.get('groups_axis_both_better_all_profiles')}`；"
        f"未通过 = `{ax.get('groups_axis_not_both_better')}`。"
    )
    lines.append("")

    # ---- 对齐度 / 有效秩（本轮新增的两个诊断量）----
    lines.append("## 对齐度与有效秩（G4：两个新诊断量，逐档给出）")
    lines.append("")
    lines.append(
        "对齐度口径：N3D 支路表示（机制 B 下为**投影后**）与答案表各类质心的平均余弦；"
        "有效秩口径：``σ > σ_max·1e-6`` 的奇异值个数（不中心化），池 = train_known + test_known。"
    )
    lines.append("")
    lines.append("| 档 | 组 | 对齐度(各类质心均值) | 对齐目标(i) 自特征 | 对齐目标(ii) 类质心 | 有效秩(readout) | 有效秩(aligned) | 参与比(aligned) | 占 R^D | 适用 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for pname, run in (report.get("runs") or {}).items():
        for res in run.get("groups", []):
            ad = res.get("alignment_degree") or {}
            spec = res.get("spectrum") or {}
            al = spec.get("aligned") or {}
            ro = spec.get("readout") or {}
            if not ad.get("applicable"):
                lines.append(
                    f"| {pname} | {res['group']['name']} | n/a | n/a | n/a | n/a | n/a | "
                    f"n/a | n/a | 否（{str(ad.get('reason'))[:40]}） |"
                )
                continue
            lines.append(
                f"| {pname} | {res['group']['name']} | "
                f"{(ad.get('cos_class_centroid') or {}).get('mean', float('nan')):.4f} | "
                f"{ad['cos_self_encoder_feature']:.4f} | {ad['cos_own_class_centroid']:.4f} | "
                f"{ro.get('rank')} | {al.get('rank')} | "
                f"{al.get('participation_ratio', float('nan')):.2f} | "
                f"{al.get('rank_ratio_dim', float('nan')) * 100:.2f}% | 是 |"
            )
    lines.append("")

    lines.append("## 归因分解（**换特征** / **打开表示训练** / 两者叠加，不合并）")
    lines.append("")
    attr = report.get("attribution", {})
    lines.append("### 只换特征（`A1_baseline`：只训 logit_scale，正标量不改 argmax）")
    lines.append("")
    lines.append("| 组 | 参照档 | 其它档 | Δmacro | ΔR@1(自检索) | ΔdetR@1(纯特征) | Δgap/σ | Δrefusal |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in attr.get("feature_only", []):
        lines.append(
            f"| {r['group']} | {r['reference_profile']} | {r['profile']} | "
            f"{_fmt_d(r['macro_acc']['delta'])} | {_fmt_d(r['step2_recall_at_1']['delta'])} | "
            f"{_fmt_d(r['step2_det_recall_at_1']['delta'])} | "
            f"{_fmt_d(r['gap_over_sigma']['delta'])} | {_fmt_d(r['refusal_rate']['delta'])} |"
        )
    lines.append("")
    lines.append("### 只打开表示训练（特征不变）")
    lines.append("")
    lines.append("| 档 | 从 | 到 | Δmacro | ΔR@1 | Δgap/σ | Δrefusal |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in attr.get("training_only", []):
        lines.append(
            f"| {r['profile']} | {r['from_group']} | {r['to_group']} | "
            f"{_fmt_d(r['macro_acc']['delta'])} | {_fmt_d(r['step2_recall_at_1']['delta'])} | "
            f"{_fmt_d(r['gap_over_sigma']['delta'])} | {_fmt_d(r['refusal_rate']['delta'])} |"
        )
    lines.append("")
    lines.append("### 两者叠加（其它档 A2/A3 vs 参照档 A1）")
    lines.append("")
    lines.append("| 参照 | 档 | 组 | Δmacro | ΔR@1 | Δgap/σ | Δrefusal |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in attr.get("combined", []):
        lines.append(
            f"| {r['reference']} | {r['profile']} | {r['group']} | "
            f"{_fmt_d(r['macro_acc']['delta'])} | {_fmt_d(r['step2_recall_at_1']['delta'])} | "
            f"{_fmt_d(r['gap_over_sigma']['delta'])} | {_fmt_d(r['refusal_rate']['delta'])} |"
        )
    lines.append("")

    lines.append("## D 的代价（现场构造实测）")
    lines.append("")
    lines.append("| 档 | D | 骨干合计 | `W_in` 形状/元素 | `W_out` 形状/元素 | q 头合计 | 总参数 | 每样本参数 | 冻结嵌入可训 | 打开表示可训 | answer_table(buffer) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for pname, cost in (report.get("cost") or {}).items():
        if not cost:
            continue
        # [!] `fc_dim != 0` 时骨干参数名被替换（W_in / W_out 消失），故一律 `.get` +
        #     显式渲染 `n/a`，不得假设键一定存在。
        wi = cost["backbone_shapes"].get("W_in")
        wo = cost["backbone_shapes"].get("W_out")
        wi_n = cost["backbone_parameters"].get("W_in")
        wo_n = cost["backbone_parameters"].get("W_out")
        lines.append(
            f"| {pname} | {cost['dim']} | {cost['backbone_total']} | "
            f"{'n/a' if wi is None else f'{wi} / {wi_n}'} | "
            f"{'n/a' if wo is None else f'{wo} / {wo_n}'} | {cost['head_total']} | "
            f"{cost['total_parameters']} | {cost['params_per_sample']:.1f} | "
            f"{cost['trainable_frozen_embedding']['n_trainable']} | "
            f"{cost['trainable_open_representation']['n_trainable_head']} | "
            f"{cost['answer_table_buffer']['shape']} |"
        )
    lines.append("")
    lines.append("现场枚举的骨干参数名（**不假设存在 `W_in` / `W_out`**）：")
    lines.append("")
    lines.append("| 档 | 骨干参数名（现场枚举） |")
    lines.append("| --- | --- |")
    for pname, cost in (report.get("cost") or {}).items():
        if not cost:
            continue
        lines.append(f"| {pname} | `{cost.get('backbone_parameter_names')}` |")
    lines.append("")

    sec = report.get("seconds", {})
    lines.append("## CPU 耗时（秒）")
    lines.append("")
    lines.append("| 档 | 合计 | 逐组 |")
    lines.append("| --- | --- | --- |")
    for pname, total in (sec.get("per_profile_total") or {}).items():
        per = ", ".join(f"{k}={v:.1f}" for k, v in (sec.get("per_group") or {}).get(pname, {}).items())
        lines.append(f"| {pname} | {total:.1f} | {per} |")
    lines.append(f"| **总计** | {sec.get('grand_total', 0.0):.1f} | — |")
    lines.append("")

    # ---- 单 seed 声明（离朱 R48 D1 修复：对照版此前漏渲染本节，与单档版对齐）----
    lines.append("## 单 seed 声明（口径）")
    lines.append("")
    ss = report.get("single_seed") or {}
    lines.append(
        f"本轮固定 `split_seed={ss.get('split_seed', BASE_SPLIT_SEED)}`、"
        f"`train_seed={ss.get('train_seed', BASE_TRAIN_SEED)}`；"
        "报告中**所有 Δ 均为单点差**（同一切分、同一训练 seed 下两组之差），"
        "**不含跨 seed 的训练随机性区间**，**无跨 seed 极差**。"
        "跨 seed 的均值 ± 极差不属本批口径，不得由本报告推断。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    """可空浮点的表格渲染（缺失显式写 ``n/a``，不写 0）。"""
    return "n/a" if value is None else f"{float(value):.4f}"


def _structure_cell(groups: Any, name: str) -> str:
    """从某档的组结果里取出该组的「结构 + 对齐」单元格文本（缺失写 ``n/a``）。"""
    if not groups:
        return "n/a | n/a | n/a | n/a | n/a"
    for res in groups:
        if res["group"]["name"] != name:
            continue
        g = res["group"]
        st = g.get("structure") or {}
        align = str(g.get("align_mode", "off"))
        if align != "off":
            align = f"{align}(λ={g.get('align_lambda')})"
        return (
            f"{st.get('shape')} | {st.get('fc_dim')} | {st.get('N')} | "
            f"{st.get('geo_field')} | {align}"
        )
    return "n/a | n/a | n/a | n/a | n/a"


def _fmt_d(value: Any) -> str:
    """可空增量的表格渲染（带符号）。"""
    return "n/a" if value is None else f"{float(value):+.4f}"


def _fmt_pct(value: Any) -> str:
    """可空**相对增量**的表格渲染（按百分比带符号；缺失写 ``n/a``，不填 0）。"""
    return "n/a" if value is None else f"{float(value) * 100.0:+.1f}%"


def write_comparison_report(report: Dict[str, Any], out_dir: str = "") -> Dict[str, str]:
    """把对照报告写为 UTF-8（无 BOM）的 JSON 与 Markdown（**只写验证目录**）。"""
    target = out_dir or EXP_REPR_DIR
    os.makedirs(target, exist_ok=True)
    json_path = os.path.join(target, "exp_repr_compare.json")
    md_path = os.path.join(target, "exp_repr_compare.md")
    with open(json_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    with open(md_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_comparison_markdown(report))
    return {"json": json_path, "markdown": md_path}


def load_comparison_report(path: str = "") -> Dict[str, Any]:
    """读取已落盘的**对照**报告 JSON。"""
    target = path or os.path.join(EXP_REPR_DIR, "exp_repr_compare.json")
    if not os.path.isfile(target):
        raise FileNotFoundError(
            f"对照报告不存在：{target!r}；请先执行 "
            "`python -m n3d_qa_learn.exp_repr_run compare`"
        )
    with open(target, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# 报告（单档版）
# ---------------------------------------------------------------------------


def render_markdown(report: Dict[str, Any]) -> str:
    """把报告字典渲染为 Markdown（表格 + 锚点对账 + 逐 epoch 轨迹 + 门禁明细）。"""
    lines: List[str] = []
    lines.append("# n3d_qa_learn 表示训练对照实验（exp_repr）")
    lines.append("")
    lines.append("> 全部数字由 `python -m n3d_qa_learn.exp_repr_run run` 现场产出；")
    lines.append("> 产物只落 `checkpoints/qa_learn/_verify/exp_repr/`，正式产物目录零改动。")
    lines.append("")
    cfg = report["groups"][0]["config"] if report["groups"] else {}
    if cfg:
        lines.append("## 固定口径")
        lines.append("")
        lines.append("| 项 | 值 |")
        lines.append("| --- | --- |")
        for key in ("split_seed", "train_seed", "epochs", "batch_size", "lr",
                    "backbone_lr", "logit_scale_init", "unknown_train_cap",
                    "max_classes", "min_questions", "test_every", "test_per_class",
                    "stage1_epochs", "stage2_epochs", "supcon_temperature", "device"):
            lines.append(f"| {key} | {cfg.get(key)} |")
        lines.append("")
        lines.append(
            "切分一致性断言：**"
            + ("通过" if report.get("split_identical") else "未通过")
            + "**（基准组 " + str(report.get("split_reference_group"))
            + "；`train_known` / `test_known` / `train_unknown` / `test_unknown` "
            "的 qid 有序序列逐位相同）"
        )
        lines.append("")
    lines.append("## 全矩阵结果")
    lines.append("")
    lines.append("| 组 | freeze_scope | head_input_mode | loss_mode | shape | fc_dim | N | geo | "
                 "对齐(λ) | gap | σ(within) | "
                 "gap/σ | 1-NN(raw) | 1-NN(repr) | macro | top1 | refusal | q 位移 | "
                 "答案表位移 | argmax 变化率 | 门禁 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
                 "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for res in report["groups"]:
        g = res["group"]
        st = g.get("structure") or {}
        align = str(g.get("align_mode", "off"))
        if align != "off":
            align = f"{align}(λ={g.get('align_lambda')})"
        geo = res["geo"]
        m = res["metric_step1"]
        sh = res["shifts"]
        gate = "PASS" if res["gate"]["passed"] else "**FAIL**"
        lines.append(
            f"| {g['name']} | {g['freeze_scope']} | {g['head_input_mode']} | "
            f"{g['loss_mode']} | {st.get('shape')} | {st.get('fc_dim')} | {st.get('N')} | "
            f"{st.get('geo_field')} | {align} | "
            f"{geo['gap']:.4f} | {geo['within_sigma']:.4f} | "
            f"{geo['gap_over_sigma']:.4f} | {geo['nn1_top1_raw']:.4f} | "
            f"{geo['nn1_top1_repr']:.4f} | {m['macro_acc']:.4f} | {m['top1_acc']:.4f} | "
            f"{res['refusal']['refusal_rate']:.4f} | {sh['q_shift_mean_l2']:.4e} | "
            f"{sh['answer_table_shift_l2']:.4e} | {sh['argmax_changed_frac']:.4f} | "
            f"{gate} |"
        )
    lines.append("")
    comp = report.get("anchor_comparison")
    if comp:
        lines.append("## 基线组锚点对账")
        lines.append("")
        lines.append("| 指标 | 锚点 | 本实现实测 | 偏差 | 容差 | 是否在容差内 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in comp["rows"]:
            lines.append(
                f"| {row['metric']} | {row['anchor']:.4f} | {row['measured']:.4f} | "
                f"{row['delta']:+.4f} | {row['tolerance']:.4f} | "
                f"{'是' if row['within_tolerance'] else '**否**'} |"
            )
        lines.append("")
        lines.append("全部落在容差内：**"
                     + ("是" if comp["all_within_tolerance"] else "否") + "**")
        lines.append("")
    lines.append("## 逐 epoch 轨迹摘要（首 / 末 epoch）")
    lines.append("")
    lines.append("| 组 | epoch | phase | loss | cos(q, 不相关键) | cos(q, 金标类) | "
                 "cos(q, 冻结参考方向) | 训练侧拒绝占比 | 训练侧 known top1 | gap/σ |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for res in report["groups"]:
        hist = res["history"]
        if not hist:
            continue
        for row in (hist[0], hist[-1]):
            ck = row["cos_q_irr_key"]
            ck_s = "n/a" if ck is None else f"{ck:+.4f}"
            cg = row.get("cos_q_gold_known")
            cg_s = "n/a" if cg is None else f"{cg:+.4f}"
            lines.append(
                f"| {res['group']['name']} | {row['epoch']} | {row['phase']} | "
                f"{row['loss']:.4f} | {ck_s} | {cg_s} | {row['cos_q_irr_ref']:+.4f} | "
                f"{row['train_refusal_frac']:.4f} | {row['train_top1_known']:.4f} | "
                f"{row['gap_over_sigma']:.4f} |"
            )
    lines.append("")
    lines.append("## 门禁明细（可训参数更新量 > 0）")
    lines.append("")
    for res in report["groups"]:
        gate = res["gate"]
        lines.append(
            f"* **{res['group']['name']}**：可训参数 {gate['n_trainable_params']} 个，"
            f"实际发生更新 {gate['n_updated_params']} 个，总更新量 L2 = "
            f"{gate['total_update_l2']:.6e}，最大单参数位移 = "
            f"{gate['max_update_abs']:.6e} → {'PASS' if gate['passed'] else '**FAIL**'}"
        )
        lines.append(f"  * 可训参数名（现场枚举）：`{gate['trainable_params']}`")
        lines.append(
            "  * **声明可训但全程零更新**（现场枚举）："
            f"`{gate['never_updated_params']}`"
        )
        lines.append(
            "  * requires_grad 但未进入优化器（现场枚举）："
            f"`{res['ungrouped_trainable_names']}`"
        )
        ast = res.get("align_stats") or {}
        if str(ast.get("mode", "off")) != "off":
            lines.append(
                f"  * 对齐项（G7）：mode={ast.get('mode')}，λ={ast.get('lambda')}，"
                f"适用 batch {ast.get('batches_with_align')} / 跳过 "
                f"{ast.get('batches_align_skipped')} / 总 batch {ast.get('batches_total')}，"
                f"align 均值={ast.get('align_mean')}、最大={ast.get('align_max')}、"
                f"最小={ast.get('align_min')}、非有限计数={ast.get('align_nonfinite')}"
            )
            if ast.get("skipped_reasons"):
                lines.append(f"  * 对齐跳过原因计数：`{ast.get('skipped_reasons')}`")
    lines.append("")

    # ---- 可构造性预检 / 失败清单（G6）----
    lines.append("## 可构造性预检与失败清单（G6）")
    lines.append("")
    pre = report.get("construction_precheck") or {}
    lines.append(
        f"请求 {pre.get('n_requested')} 组，可构造 `{pre.get('n_runnable')}` 组；"
        f"预检规则：{pre.get('rule')}"
    )
    lines.append("")
    fails = report.get("construction_failures") or []
    if fails:
        lines.append("| 组 | 异常类型 | 可读原因 |")
        lines.append("| --- | --- | --- |")
        for item in fails:
            lines.append(
                f"| {item['group']} | `{item['exc_type']}` | {str(item['reason'])[:400]} |"
            )
    else:
        lines.append("* 本批**无**构造失败组（0 项）。")
    lines.append("")

    # ---- 逐轴归因（形状 / FC / N / geo / 对齐 / 载体，六节分开）----
    ax = report.get("axis_attribution") or {}
    lines.append("## 逐轴归因（**形状 / FC / 容量 N / 几何权重场 / 对齐 / 载体** 六节分开，禁止合并）")
    lines.append("")
    lines.append(str(ax.get("sep", "")))
    lines.append("")
    lines.append("| 轴 | 组 | Δmacro | Δmacro(相对) | ΔR@1 | ΔR@1(相对) | ΔdetR@1 | Δgap/σ | Δrefusal | Δ对齐度 | Δ有效秩 | 同时更优 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for dim_name, pack in (ax.get("axes") or {}).items():
        for r in pack.get("rows", []):
            lines.append(
                f"| {dim_name} | {r['group']} | "
                f"{_fmt_d(r['delta_macro_acc'])} | {_fmt_pct(r.get('delta_macro_acc_rel'))} | "
                f"{_fmt_d(r['delta_step2_recall_at_1'])} | "
                f"{_fmt_pct(r.get('delta_step2_recall_at_1_rel'))} | "
                f"{_fmt_d(r.get('delta_step2_det_recall_at_1'))} | "
                f"{_fmt_d(r.get('delta_gap_over_sigma'))} | "
                f"{_fmt_d(r.get('delta_refusal_rate'))} | "
                f"{_fmt_d(r.get('delta_align_cos_mean'))} | "
                f"{_fmt_d(r.get('delta_eff_rank_aligned'))} | "
                f"{'是' if r.get('both_better') else '否'} |"
            )
    lines.append("")

    # ---- 对齐度 / 有效秩（新诊断量）----
    lines.append("## 对齐度与有效秩（G4：两个新诊断量）")
    lines.append("")
    lines.append(
        "对齐度 = N3D 支路表示（机制 B 下为**投影后**）与答案表各类质心的平均余弦；"
        "有效秩 = ``σ > σ_max·1e-6`` 的奇异值个数（**不中心化**），"
        "池 = `train_known + test_known`。"
    )
    lines.append("")
    lines.append("| 组 | 对齐度 | 目标(i) 自特征 | 目标(ii) 类质心 | 有效秩(readout) | 有效秩(aligned) | 参与比(aligned) | 占 R^D | 适用 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for res in report["groups"]:
        ad = res.get("alignment_degree") or {}
        spec = res.get("spectrum") or {}
        al = spec.get("aligned") or {}
        ro = spec.get("readout") or {}
        if not ad.get("applicable"):
            lines.append(
                f"| {res['group']['name']} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | "
                f"否（{str(ad.get('reason'))[:44]}） |"
            )
            continue
        lines.append(
            f"| {res['group']['name']} | "
            f"{(ad.get('cos_class_centroid') or {}).get('mean', float('nan')):.4f} | "
            f"{ad['cos_self_encoder_feature']:.4f} | {ad['cos_own_class_centroid']:.4f} | "
            f"{ro.get('rank')} | {al.get('rank')} | "
            f"{al.get('participation_ratio', float('nan')):.2f} | "
            f"{al.get('rank_ratio_dim', float('nan')) * 100:.2f}% | 是 |"
        )
    lines.append("")

    # ---- 单 seed 声明（口径必须显式）----
    lines.append("## 单 seed 声明（口径）")
    lines.append("")
    lines.append(
        f"本轮固定 `split_seed={report.get('single_seed', {}).get('split_seed', BASE_SPLIT_SEED)}`、"
        f"`train_seed={report.get('single_seed', {}).get('train_seed', BASE_TRAIN_SEED)}`；"
        "报告中**所有 Δ 均为单点差**（同一切分、同一训练 seed 下两组之差），"
        "**不含跨 seed 的训练随机性区间**，**无跨 seed 极差**。"
        "跨 seed 的均值 ± 极差不属本批口径，不得由本报告推断。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_report(report: Dict[str, Any], out_dir: str = "") -> Dict[str, str]:
    """把报告写为 UTF-8（无 BOM）的 JSON 与 Markdown（**只写验证目录**）。

    参数
    ----
    report : Dict[str, Any]
        报告字典。
    out_dir : str
        输出目录；空串时用 :data:`EXP_REPR_DIR`。

    返回
    ----
    Dict[str, str]
        ``{"json": ..., "markdown": ...}`` 路径。
    """
    target = out_dir or EXP_REPR_DIR
    os.makedirs(target, exist_ok=True)
    json_path = os.path.join(target, "exp_repr_report.json")
    md_path = os.path.join(target, "exp_repr_report.md")
    with open(json_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    with open(md_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(report))
    return {"json": json_path, "markdown": md_path}


def load_report(path: str = "") -> Dict[str, Any]:
    """读取已落盘的报告 JSON（``summary`` 子命令用）。"""
    target = path or os.path.join(EXP_REPR_DIR, "exp_repr_report.json")
    if not os.path.isfile(target):
        raise FileNotFoundError(
            f"报告不存在：{target!r}；请先执行 `python -m n3d_qa_learn.exp_repr_run run`"
        )
    with open(target, "r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "EXP_REPR_DIR",
    "FREEZE_SCOPES",
    "LOSS_MODES",
    "SHAPES",
    "GEO_FIELDS",
    "ALIGN_MODE_CHOICES",
    "PROJ_INIT_CHOICES",
    "ALIGN_TARGETS",
    "STRUCTURE_DEFAULTS",
    "BackendStructure",
    "SUPCON_TEMPERATURE",
    "STAGE1_EPOCHS",
    "STAGE2_EPOCHS",
    "BASE_MAX_CLASSES",
    "BASE_MIN_QUESTIONS",
    "BASE_TEST_EVERY",
    "BASE_TEST_PER_CLASS",
    "BASE_UNKNOWN_TRAIN_CAP",
    "BASE_EPOCHS",
    "BASE_BATCH_SIZE",
    "BASE_LR",
    "BASE_BACKBONE_LR",
    "BASE_LOGIT_SCALE_INIT",
    "BASE_SPLIT_SEED",
    "BASE_TRAIN_SEED",
    "ANCHOR",
    "ANCHOR_TOL",
    "MATRIX",
    "DIMENSIONS",
    "BASELINE_GROUP",
    "STRUCT_BASELINE_GROUP",
    "ALLOWED_UNGROUPED_TRAINABLE",
    "ReprGroup",
    "group_by_name",
    "CARRIER_GROUPS",
    "AXIS_FIELDS",
    "custom_group",
    "FeatureProfile",
    "FEATURE_PROFILES",
    "PROFILE_LEXICAL",
    "PROFILE_SEMANTIC",
    "DEFAULT_PROFILE",
    "profile_by_name",
    "split_qids_digest",
    "split_qids_equal",
    "dimension_cost",
    "Step2Bundle",
    "build_step2_bundle",
    "step2_recall_of_model",
    "split_qids",
    "assert_same_split",
    "build_group_config",
    "supcon_loss",
    "batch_loss",
    "setup_answer_table",
    "apply_freeze",
    "make_optimizer",
    "ungrouped_trainable_names",
    "param_deltas",
    "update_gate",
    "representation_geometry",
    "effective_rank",
    "EFFECTIVE_RANK_REL_TOL",
    "alignment_degree",
    "spectral_diagnostics",
    "construction_probe",
    "precheck_constructibility",
    "axis_attribution",
    "epoch_diagnostics",
    "phase_plan",
    "run_group",
    "run_experiment",
    "run_comparison",
    "anchor_comparison",
    "summarize",
    "render_markdown",
    "write_report",
    "load_report",
    "render_comparison_markdown",
    "write_comparison_report",
    "load_comparison_report",
]
