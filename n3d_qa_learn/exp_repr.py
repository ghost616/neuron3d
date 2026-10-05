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
from .backends import BackendRegistry
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
from .heads import HEAD_INPUT_MODES, N3DQA, N3DQAConfig
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

#: 监督对比损失的温度（冻结常量）。
SUPCON_TEMPERATURE: float = 0.07

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
    """一个对照组的标识（恰好由三个被测维度的取值确定）。

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
    """

    name: str
    freeze_scope: str
    head_input_mode: str
    loss_mode: str

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

    def as_dict(self) -> Dict[str, str]:
        """JSON 化。"""
        return {
            "name": self.name,
            "freeze_scope": self.freeze_scope,
            "head_input_mode": self.head_input_mode,
            "loss_mode": self.loss_mode,
        }


#: 对照组矩阵（8 组，逐维单独测；顺序即报告顺序）。
MATRIX: Tuple[ReprGroup, ...] = (
    ReprGroup("A1_baseline", "logit_only", "raw", "ce"),
    ReprGroup("A2_head", "head", "raw", "ce"),
    ReprGroup("A3_head_backbone", "head_backbone", "raw", "ce"),
    ReprGroup("B1_concat", "logit_only", "concat", "ce"),
    ReprGroup("B2_n3d", "logit_only", "n3d", "ce"),
    ReprGroup("C1_no_irr_centroid", "head", "raw", "no_irr_centroid"),
    ReprGroup("C2_staged", "head", "raw", "staged"),
    ReprGroup("C3_supcon", "head", "raw", "supcon"),
)

#: 基线组名（锚点对标与「换特征 vs 打开训练」归因的参照系）。
BASELINE_GROUP: str = "A1_baseline"

#: 三个被测维度的结构描述（基准组 + 变化项 + 被改变的字段）。
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
)

#: ``ungrouped_trainable_names`` 在现场允许出现的**完整**名单（显式列出，不放宽为"大部分"）。
ALLOWED_UNGROUPED_TRAINABLE: Tuple[str, ...] = ("head.mix_logit",)


def group_by_name(name: str) -> ReprGroup:
    """按组名取矩阵中的组定义（未知名立即报错，不静默回落）。"""
    for group in MATRIX:
        if group.name == name:
            return group
    raise KeyError(f"未知组名 {name!r}；可用组名 = {[g.name for g in MATRIX]}")


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
) -> Dict[str, Any]:
    """连接参数 ``D`` 的**代价实测**（现场构造，不凭记忆写）。

    报告内容
    --------
    * 骨干逐参数形状与元素数（``W_in`` / ``W_out`` / ``edge_weight`` / ``neuron_bias``）
      与骨干合计；
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

    返回
    ----
    Dict[str, Any]
        代价实测字典。
    """
    reg = BackendRegistry(int(dim))
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
    frozen_trainable = int(head_numel.get("logit_scale", 0))
    opened_trainable = int(
        head_numel.get("logit_scale", 0)
        + head_numel.get("q_head.weight", 0)
        + head_numel.get("q_head.bias", 0)
    )
    return {
        "dim": int(dim),
        "backend": str(backend),
        "topology": {k: float(v) for k, v in adapter.topology_stats().items()},
        "backbone_parameters": backbone_numel,
        "backbone_shapes": backbone_shapes,
        "backbone_total": int(backbone_total),
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
            "names_head": ["head.logit_scale", "head.q_head.bias", "head.q_head.weight"],
            "names_head_backbone_extra": sorted(backbone_numel.keys()),
            "note": ("head 档训 q 头；head_backbone 档额外声明骨干可训，"
                     "但在 head_input_mode='raw' 下骨干结构性不在计算图上（见 12.8）"),
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
    )


# ---------------------------------------------------------------------------
# 损失与冻结口径
# ---------------------------------------------------------------------------


def supcon_loss(
    q: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    temperature: float = SUPCON_TEMPERATURE,
) -> Optional[torch.Tensor]:
    """监督对比损失（同答案问题为正样本；**批内构造、不采样**）。

    口径（显式固定，不随运行变化）
    -----------------------------
    * 锚点集合 ``I`` = 批内**答案表内**样本中「至少有一个同类正样本」的那些样本；
      「不相关」样本（``target == n_classes``）**不参与**（它们没有正样本）；
    * 正样本集合 ``P(i)`` = 批内与 ``i`` 同答案类、且不等于 ``i`` 的样本；
    * 负样本 = 批内其余样本（含其它类别），归一化后按 ``q @ q^T / temperature`` 打分；
    * 损失 = ``-1/|I| * sum_i [ 1/|P(i)| * sum_{p in P(i)} log softmax_i(p) ]``。

    参数
    ----
    q : torch.Tensor
        ``[B, D]`` 的 **L2 归一化**查询（:meth:`N3DQA.query` 的输出）。
    targets : torch.Tensor
        ``[B]`` 目标下标；等于 ``n_classes`` 的是「不相关」样本。
    n_classes : int
        答案类别数 ``C``。
    temperature : float
        温度（``> 0``）。

    返回
    ----
    Optional[torch.Tensor]
        标量损失；批内可用锚点不足（``|I| == 0``）时返回 ``None``
        （调用侧按「跳过该 batch」处置，并在诊断里计数可见）。
    """
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature 必须 > 0，当前 {temperature}")
    keep = targets != int(n_classes)
    z = q[keep]
    y = targets[keep]
    if int(z.shape[0]) < 2:
        return None
    eye = torch.eye(int(z.shape[0]), dtype=torch.bool, device=z.device)
    sim = (z @ z.transpose(0, 1)) / float(temperature)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & (~eye)
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if int(valid.sum().item()) == 0:
        return None
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # [!] 必须用 ``where`` 而不是 ``* pos``：``-inf * False`` 在 PyTorch 里是 ``nan``
    #     （对角线被填成 ``-inf``，乘 0 仍得 nan），会污染整个 batch 的损失。
    log_prob = torch.where(pos, log_prob, torch.zeros_like(log_prob))
    per_anchor = log_prob.sum(dim=1) / pos_count.clamp_min(1)
    return -per_anchor[valid].mean()


def batch_loss(
    model: N3DQA,
    kind: str,
    logits: torch.Tensor,
    q: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    n_classes: int,
) -> Optional[torch.Tensor]:
    """按口径分派一个 batch 的损失。

    ``kind`` 取值与语义
    ------------------
    * ``"ce"``：现行交叉熵（含逐样本权重），**与 :meth:`N3DQA.cross_entropy` 同路径**；
    * ``"ce_masked"``：「不相关」样本以 ``keep = targets != C`` 显式排除，
      不产生把它们拉向「不相关」键的梯度（配合答案表末位置零，见
      :func:`setup_answer_table`）；
    * ``"supcon"``：监督对比损失（见 :func:`supcon_loss`）。

    返回
    ----
    Optional[torch.Tensor]
        标量损失；本 batch 无可做功样本时返回 ``None``。
    """
    if kind == "ce":
        return model.cross_entropy(logits, targets, sample_weight=weight)
    if kind == "ce_masked":
        per = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=float(model.config.label_smoothing),
        )
        keep = targets != int(n_classes)
        if int(keep.sum().item()) == 0:
            return None
        per = per[keep]
        w = weight[keep]
        return (per * w).sum() / w.sum().clamp_min(1e-12)
    if kind == "supcon":
        return supcon_loss(q, targets, int(n_classes))
    raise ValueError(f"未知损失口径 {kind!r}")


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
                q = model.query(feats)
                logits = model.logits(feats, None, q=q)
                loss = batch_loss(model, loss_kind, logits, q, tgt, weight, C)
                if loss is None:
                    n_skipped += 1
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


def run_experiment(
    *,
    group_names: Optional[Sequence[str]] = None,
    progress: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """顺序跑完整矩阵（或指定子集），并执行**切分一致性断言**。

    参数
    ----
    group_names : Optional[Sequence[str]]
        只跑这些组（``None`` = 全部 8 组）。
    progress : Optional[Callable[[str], None]]
        进度回调（只接收一行文本；本模块自身不打印任何东西）。
    kwargs
        透传给 :func:`run_group` 的超参（含 ``profile`` / ``step2_bundle``）。

    返回
    ----
    Dict[str, Any]
        报告字典（含 ``profile`` / ``groups`` / ``split_identical`` /
        ``anchor_comparison`` / ``cost``）。
    """
    names = list(group_names) if group_names else [g.name for g in MATRIX]
    wanted = [group_by_name(n) for n in names]
    prof = profile_by_name(str(kwargs.get("profile", DEFAULT_PROFILE)))
    results: List[Dict[str, Any]] = []
    reference: Optional[Dict[str, List[str]]] = None
    ref_name = ""
    for group in wanted:
        if progress is not None:
            progress(
                f"[start] {group.name} (profile={prof.name}, "
                f"freeze_scope={group.freeze_scope}, "
                f"head_input_mode={group.head_input_mode}, loss_mode={group.loss_mode})"
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
            progress(
                f"[done ] {group.name} gap/σ={res['geo']['gap_over_sigma']:.4f} "
                f"macro={res['metric_step1']['macro_acc']:.4f}{s2_txt} "
                f"refusal={res['refusal']['refusal_rate']:.4f} "
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
    return {
        "experiment": "n3d_qa_learn/exp_repr",
        "profile": prof.as_dict(),
        "matrix": [g.as_dict() for g in MATRIX],
        "dimensions": [dict(d) for d in DIMENSIONS],
        "anchor": dict(ANCHOR),
        "anchor_tolerance": dict(ANCHOR_TOL),
        "groups": results,
        "split_reference_group": str(ref_name),
        "split_identical": True,
        "split_digest": split_qids_digest(reference) if reference else {},
        "anchor_comparison": anchor_comparison(baseline["geo"]) if baseline else None,
        "cost": cost,
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
        out.append({
            "group": res["group"]["name"],
            "freeze_scope": res["group"]["freeze_scope"],
            "head_input_mode": res["group"]["head_input_mode"],
            "loss_mode": res["group"]["loss_mode"],
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
        })
    return out


def _metric_row(res: Dict[str, Any]) -> Dict[str, Any]:
    """从一组结果里抽出**主判据与另报量**（缺项如实记 ``None``，不填 0 冒充）。"""
    s2 = res.get("step2") or None
    qpath = (s2 or {}).get("q_path") or {}
    det = (s2 or {}).get("deterministic") or {}
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
    }


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """``b - a``（任一为 ``None`` 时返回 ``None``，不把缺失当 0）。"""
    if a is None or b is None:
        return None
    return float(b) - float(a)


def run_comparison(
    *,
    profiles: Sequence[str] = (PROFILE_LEXICAL, PROFILE_SEMANTIC),
    group_names: Optional[Sequence[str]] = None,
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
            group_names=group_names, progress=progress,
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
    }


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
    hdr = "| 组 | freeze_scope | head_input_mode | loss_mode |"
    for p in prof_names:
        hdr += f" macro({p}) | R@1({p}) | detR@1({p}) | refusal({p}) | gap/σ({p}) | s({p}) |"
    hdr += " Δmacro | ΔR@1 | ΔdetR@1 | 同时更优 |"
    lines.append(hdr)
    sep = "| --- | --- | --- | --- |" + " --- |" * (6 * len(prof_names)) + " --- | --- | --- | --- |"
    lines.append(sep)
    ref = report["verdict"]["reference_profile"]
    for row in report.get("comparison", []):
        m = row["reference"]
        line = (f"| {row['group']} | {row['freeze_scope']} | {row['head_input_mode']} | "
                f"{row['loss_mode']} |")
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
        wi = cost["backbone_shapes"].get("W_in")
        wo = cost["backbone_shapes"].get("W_out")
        lines.append(
            f"| {pname} | {cost['dim']} | {cost['backbone_total']} | "
            f"{wi} / {cost['backbone_parameters'].get('W_in')} | "
            f"{wo} / {cost['backbone_parameters'].get('W_out')} | {cost['head_total']} | "
            f"{cost['total_parameters']} | {cost['params_per_sample']:.1f} | "
            f"{cost['trainable_frozen_embedding']['n_trainable']} | "
            f"{cost['trainable_open_representation']['n_trainable_head']} | "
            f"{cost['answer_table_buffer']['shape']} |"
        )
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
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    """可空浮点的表格渲染（缺失显式写 ``n/a``，不写 0）。"""
    return "n/a" if value is None else f"{float(value):.4f}"


def _fmt_d(value: Any) -> str:
    """可空增量的表格渲染（带符号）。"""
    return "n/a" if value is None else f"{float(value):+.4f}"


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
    lines.append("| 组 | freeze_scope | head_input_mode | loss_mode | gap | σ(within) | "
                 "gap/σ | 1-NN(raw) | 1-NN(repr) | macro | top1 | refusal | q 位移 | "
                 "答案表位移 | argmax 变化率 | 门禁 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
                 "--- | --- | --- | --- | --- |")
    for res in report["groups"]:
        g = res["group"]
        geo = res["geo"]
        m = res["metric_step1"]
        sh = res["shifts"]
        gate = "PASS" if res["gate"]["passed"] else "**FAIL**"
        lines.append(
            f"| {g['name']} | {g['freeze_scope']} | {g['head_input_mode']} | "
            f"{g['loss_mode']} | {geo['gap']:.4f} | {geo['within_sigma']:.4f} | "
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
    "ALLOWED_UNGROUPED_TRAINABLE",
    "ReprGroup",
    "group_by_name",
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
