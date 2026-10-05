"""n3d_qa_learn 的自建训练循环、产物落盘与加载守卫。

职责
----
* 组装 ``连接契约代理层 -> 后端模型 -> D 维 q 头 -> 候选打分`` 的训练链路；
* 交叉熵训练（含显式「不相关」类的样本权重处置）；
* 产物落盘（**自写 zip + 固定时间戳**，零新依赖）到 ``checkpoints/qa_learn/``；
* **加载守卫**：校验答案表指纹与向量化口径指纹，不一致立即报错。

产物（``checkpoints/qa_learn/*.pt.zip``，自写 zip，成员固定）
-----------------------------------------------------------
====================================  ==================================================
成员                                  内容
====================================  ==================================================
``meta.json``                         全部标量/指纹字段（见 ``build_meta``）
``model_state_dict.pt``               ``torch.save`` 的 ``state_dict``
``answer_table.pt``                   仅 ``index`` 模式：``A in R^[C+1, D]``
====================================  ==================================================

``meta.json`` 的 **D 落点**（功能点 3 的硬要求）：``dim`` / ``backend.input_dim`` /
``answer_table.shape[1]`` 三处都必须等于 ``D``。

产出确定性
----------
* ``torch.save`` 写入 ``io.BytesIO`` 后取字节，zip 条目的 ``date_time`` 固定为
  ``ZIP_EPOCH``（1980-01-01），因此同参数重复运行的产物**逐字节一致**。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .backends import BackendAdapter, BackendRegistry, build_registry
from .data import (  # noqa: F401 - QACorpus 等类型在签名中使用
    DEFAULT_QA_CACHE_DIR,
    DEFAULT_TEXT_DIR,
    QACorpus,
    QARecord,
    SplitSpec,
    TextRecord,
    build_answer_space,
    load_qa_records,
    load_text_lines,
    make_splits,
)
from .encoders import (
    ROLE_QUESTION,
    EncoderConfig,
    build_vectorizer,
    declared_dim,
    vectorizer_from_meta,
)
from .features import TextVectorizer, VectorizerConfig
from .heads import N3DQA, N3DQAConfig

#: 产物目录（正式）。
DEFAULT_ARTIFACT_DIR: str = os.path.join("checkpoints", "qa_learn")

#: 验证类运行目录。
DEFAULT_VERIFY_DIR: str = os.path.join(DEFAULT_ARTIFACT_DIR, "_verify")

#: zip 条目固定时间戳（1980-01-01，zip 格式的最小合法值）。
ZIP_EPOCH: Tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)

#: 产物格式版本（结构变化必须递增）。
ARTIFACT_FORMAT_VERSION: str = "n3dqa-art-v1"

#: `rebuild_model` 重建模型时填给 `logit_scale` 的**固定初值常量**。
#: 真实（训练后）尺度由 `load_state_dict` 从 state_dict 恢复（该字段是 Parameter，必在其中）；
#: 这里**不得**再引用 `meta["model"]["logit_scale"]`——那是训练后数值，用它当构造初值属语义错位。
REBUILD_LOGIT_SCALE_INIT: float = 20.0


# ---------------------------------------------------------------------------
# 训练配置
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """训练与数据装配配置（构造期不变量在 ``__post_init__`` 校验）。

    参数
    ----
    backend : str
        N3D 后端名。
    output_mode : str
        ``index`` / ``pointer``。
    epochs / batch_size / lr / weight_decay : 训练超参。
    seed : int
        单一随机种子（数据顺序 / 参数初始化已由局部 generator 隔离）。
    split_seed : int
        **切分种子**（``-1`` = 沿用 ``seed``，即历史行为，逐位不变）。置为 ``>= 0`` 时，
        数据切分只由它驱动、训练仍由 ``seed`` 驱动 —— 这是「固定切分 + 多训练 seed」
        这类对照实验得以表达的前提（否则切分变化会污染对照）。
    max_classes : int
        答案表类别数上限。
    min_questions : int
        类别进入答案表所需的最少问题数。
    test_every : int
        每个已入表类别的第 ``test_every`` 条问题进测试侧。
    unknown_class_weight : float
        「不相关」训练样本的交叉熵权重（去偏，``1.0`` = 不加权）。
    label_smoothing : float
        交叉熵标签平滑。
    max_train_samples : int
        训练样本上限（``0`` = 不限制）；用于限批验证类运行。
    text_threshold : float
        步骤 2 的路由余弦阈值。
    qa_cache_dir / text_dir : 数据来源目录。
    train_backbone : bool
        是否训练后端 N3D 权重（**默认 ``False`` = 冻结**）。置 ``False`` 时后端是固定
        特征提取器；置 ``True`` 时后端与头一起训练。保留该开关是为了让"后端是否真的在学"
        这件事**可被实验区分**（实测：打开后端训练在 ``C<=10`` 档把 ``macro`` 从 ``0.30``
        降到 ``0.08`` 量级 —— 10k 量级后端参数在 140 个训练样本上必然过拟合）。
    backbone_lr : float
        后端权重的学习率（与 ``lr`` 分开，避免 1e3 量级样本上后端被大步长打散）。
    head_input_mode : str
        喂给 ``q`` 头的连接口径（``raw`` / ``concat`` / ``n3d``，见
        :class:`n3d_qa_learn.heads.N3DQAConfig`）。默认 ``"raw"``（现场实测最优）。
    logit_scale_init : float
        logits 尺度（逆温度）的初值。
    learn_logit_scale : bool
        是否让 logit 尺度参与学习（**默认 ``True``**）。该标量**不改变 argmax**，
        因此"训练它"只改变打分的锐度与拒绝率，不会改变步骤 1 的 top-1 判据 ——
        这正是"冻结 q 头但仍有真实可训练参数、真实跑训练循环"所需要的口径。
    answer_table_mode : str
        答案表来源口径（``centroid`` 默认 / ``free``）。
    """

    train_backbone: bool = False
    train_head: bool = False
    backbone_lr: float = 5e-4
    head_input_mode: str = "raw"
    logit_scale_init: float = 20.0
    learn_logit_scale: bool = True
    answer_table_mode: str = "centroid"

    backend: str = "n3d_shape"
    output_mode: str = "index"
    epochs: int = 40
    batch_size: int = 64
    lr: float = 5e-3
    weight_decay: float = 0.0
    seed: int = 42
    split_seed: int = -1
    max_classes: int = 8
    min_questions: int = 8
    test_every: int = 3
    test_per_class: int = 2
    unknown_class_weight: float = 1.0
    unknown_train_cap: int = 500
    label_smoothing: float = 0.0
    max_train_samples: int = 0
    text_threshold: float = 0.28
    qa_cache_dir: str = DEFAULT_QA_CACHE_DIR
    text_dir: str = DEFAULT_TEXT_DIR

    def __post_init__(self) -> None:
        if int(self.epochs) < 1:
            raise ValueError(f"epochs 必须 >= 1，当前 {self.epochs}")
        if int(self.batch_size) < 1:
            raise ValueError(f"batch_size 必须 >= 1，当前 {self.batch_size}")
        if not (float(self.lr) > 0.0):
            raise ValueError(f"lr 必须 > 0，当前 {self.lr}")
        if float(self.weight_decay) < 0.0:
            raise ValueError(f"weight_decay 必须 >= 0，当前 {self.weight_decay}")
        if int(self.split_seed) < -1:
            raise ValueError(
                f"split_seed 必须 >= -1（-1 = 沿用 seed），当前 {self.split_seed}"
            )
        if int(self.min_questions) < 2:
            raise ValueError(f"min_questions 必须 >= 2，当前 {self.min_questions}")
        if int(self.test_every) < 2:
            raise ValueError(f"test_every 必须 >= 2，当前 {self.test_every}")
        if int(self.test_per_class) < 0:
            raise ValueError(
                f"test_per_class 必须 >= 0（0 = 不限制），当前 {self.test_per_class}"
            )
        if not (float(self.unknown_class_weight) > 0.0):
            raise ValueError(
                f"unknown_class_weight 必须 > 0，当前 {self.unknown_class_weight}"
            )
        if int(self.max_train_samples) < 0:
            raise ValueError(
                f"max_train_samples 必须 >= 0，当前 {self.max_train_samples}"
            )
        if self.output_mode not in ("index", "pointer"):
            raise ValueError(f"output_mode 仅允许 index / pointer，当前 {self.output_mode!r}")
        if not (float(self.backbone_lr) > 0.0):
            raise ValueError(f"backbone_lr 必须 > 0，当前 {self.backbone_lr}")

    def to_dict(self) -> Dict[str, Any]:
        """JSON 化（写进产物 meta）。"""
        return {
            "backend": str(self.backend),
            "output_mode": str(self.output_mode),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "seed": int(self.seed),
            "split_seed": int(self.split_seed),
            "max_classes": int(self.max_classes),
            "min_questions": int(self.min_questions),
            "test_every": int(self.test_every),
            "test_per_class": int(self.test_per_class),
            "unknown_class_weight": float(self.unknown_class_weight),
            "unknown_train_cap": int(self.unknown_train_cap),
            "label_smoothing": float(self.label_smoothing),
            "max_train_samples": int(self.max_train_samples),
            "text_threshold": float(self.text_threshold),
            "qa_cache_dir": str(self.qa_cache_dir),
            "text_dir": str(self.text_dir),
            "train_backbone": bool(self.train_backbone),
            "backbone_lr": float(self.backbone_lr),
            "head_input_mode": str(self.head_input_mode),
            "logit_scale_init": float(self.logit_scale_init),
            "learn_logit_scale": bool(self.learn_logit_scale),
            "answer_table_mode": str(self.answer_table_mode),
        }


def set_deterministic_seed(seed: int) -> None:
    """设置全局随机种子（python / numpy / torch）。

    参数
    ----
    seed : int
        随机种子。
    """
    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))


# ---------------------------------------------------------------------------
# 数据组装
# ---------------------------------------------------------------------------


@dataclass
class TrainingData:
    """训练用数据（全部由现场数据装配得到，不含任何硬编码样本）。"""

    corpus: QACorpus
    splits: SplitSpec
    text_lines: List[TextRecord]
    qa_files: List[str]
    vectorizer: TextVectorizer

    def class_frequencies(self) -> List[float]:
        """训练侧（known 子集）逐类样本数（与答案表下标对齐）。"""
        counts = [0.0] * self.corpus.n_classes
        index = self.corpus.key_to_index()
        for rec in self.splits.train_known:
            counts[index[rec.answer_key]] += 1.0
        return counts


def split_seed_of(cfg: TrainConfig) -> int:
    """返回**生效的切分种子**（``split_seed < 0`` 时沿用 ``seed``，即历史行为）。

    参数
    ----
    cfg : TrainConfig
        配置。

    返回
    ----
    int
        数据切分（``data._deterministic_order`` 与未知类配额截断）使用的种子。
    """
    return int(cfg.seed) if int(cfg.split_seed) < 0 else int(cfg.split_seed)


def build_training_data(
    cfg: TrainConfig,
    encoder_name: str = "",
    encoder_config: Optional[EncoderConfig] = None,
) -> TrainingData:
    """按配置装配训练数据（读 QA 缓存 -> 答案表 -> 切分 -> 文本行）。

    参数
    ----
    cfg : TrainConfig
        配置。
    encoder_name : str
        **可插拔特征实现的入口**（注册表键；空串 = 角色默认实现
        ``ROLE_DEFAULT_ENCODER["question"]``，即现状的 ``local-hash`` 口径）。
        传 ``"bge-m3"`` 即切到 HF 编码器适配器（D = ``hidden_size``）。
    encoder_config : Optional[EncoderConfig]
        完整的编码器配置（**优先于** ``encoder_name``）；需要指定 ``source`` /
        ``max_length`` / ``expect_dim`` 等时用它（如 ``source="models/bge-m3"``
        指向本地权重目录，避免走 HF 缓存或联网）。

    返回
    ----
    TrainingData
        数据装配结果。
    """
    records, qa_files = load_qa_records(cache_dir=cfg.qa_cache_dir)
    corpus = build_answer_space(
        records, min_questions=int(cfg.min_questions), max_classes=int(cfg.max_classes)
    )
    splits = make_splits(
        corpus,
        test_every=int(cfg.test_every),
        seed=split_seed_of(cfg),
        unknown_train_cap=int(cfg.unknown_train_cap),
        test_per_class=int(cfg.test_per_class),
    )
    if not splits.train_known:
        raise RuntimeError(
            "训练侧（known）为空：请放宽 min_questions / max_classes，或换用样本更丰富的 QA 缓存"
        )
    # 向量化器经**可插拔编码器注册表**构造；空串 = 角色默认实现（与历史逐位一致）
    enc_cfg = (
        encoder_config
        if encoder_config is not None
        else EncoderConfig(name=str(encoder_name), role=ROLE_QUESTION)
    )
    vectorizer = build_vectorizer(enc_cfg)
    return TrainingData(
        corpus=corpus,
        splits=splits,
        text_lines=load_text_lines(cfg.text_dir),
        qa_files=qa_files,
        vectorizer=vectorizer,
    )


def _build_model(data: TrainingData, cfg: TrainConfig) -> Tuple[N3DQA, BackendAdapter, BackendRegistry]:
    """构造注册表并取指定后端，再组装 ``N3DQA`` 模型（代理层不参与超参调优）。"""
    registry = build_registry(data.vectorizer.config)
    if cfg.backend not in registry.available:
        raise RuntimeError(
            f"后端 {cfg.backend!r} 不可用；已登记 = {registry.available}，"
            f"不可用 = {registry.unavailable}"
        )
    adapter = registry.get(cfg.backend)
    model = N3DQA(
        adapter,
        data.corpus.n_classes,
        N3DQAConfig(
            dim=data.vectorizer.dim,
            output_mode=cfg.output_mode,
            head_input_mode=str(cfg.head_input_mode),
            label_smoothing=float(cfg.label_smoothing),
            logit_scale_init=float(cfg.logit_scale_init),
            learn_logit_scale=bool(cfg.learn_logit_scale),
            answer_table_mode=str(cfg.answer_table_mode),
        ),
    )
    return model, adapter, registry


# ---------------------------------------------------------------------------
# 一个 epoch
# ---------------------------------------------------------------------------


def _answer_vectors(records: Sequence[QARecord], vectorizer: TextVectorizer) -> torch.Tensor:
    """把问题文本编码为 ``[B, D]`` 张量。"""
    feats = [vectorizer.encode(rec.question) for rec in records]
    return torch.tensor(feats, dtype=torch.float32)


def _pointer_keys(
    answer_keys: Sequence[str],
    answer_display: Dict[str, str],
    vectorizer: TextVectorizer,
    batch_size: int,
) -> torch.Tensor:
    """构造指针模式的候选键表 ``K in R^[B, C+1, D]``。

    口径
    ----
    * 候选集合 = **全局答案表 + 末位「不相关」**（与索引模式同宽，便于两种模式直接对比）；
    * 键文本 = 答案的**展示文本**（如 ``"Switzerland"``）；末位「不相关」的键文本是
      固定常量 :data:`IRRELEVANT_KEY_TEXT`；
    * 键由 :class:`TextVectorizer` 确定性编码得到 —— 因此指针模式**没有任何固定候选参数表**。
    """
    lines: List[str] = [answer_display.get(k, k) for k in answer_keys]
    lines.append(IRRELEVANT_KEY_TEXT)
    encoded = [vectorizer.encode(t) for t in lines]
    keys = torch.tensor(encoded, dtype=torch.float32)  # [C+1, D]
    return keys.unsqueeze(0).expand(int(batch_size), -1, -1).contiguous()


#: 指针模式下「不相关」候选位的固定键文本（口径常量）。
IRRELEVANT_KEY_TEXT: str = "unrelated"


def _batch_tensors(
    records: Sequence[QARecord],
    targets: Sequence[int],
    data: TrainingData,
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把一个 batch 编码成 ``(features, targets, sample_weight)``。

    样本权重口径
    -----------
    * 「不相关」样本（``target == C``）：权重 = ``cfg.unknown_class_weight``；
    * 具体答案样本：权重 = ``1 / sqrt(该类别在训练侧的样本数)`` —— 类别越少样本权重越大，
      抑制"大类吃掉全部注意力"。权重逐 batch 重算（不预先固化，避免与切分耦合）。
    * ``pointer`` 模式的目标是**候选位次**，与 ``index`` 模式的候选下标在"全局答案表 + 末位
      不相关"口径下**恰好同值**（``candidate_width`` 相同），故这里无需分支。
    """
    feats = _answer_vectors(records, data.vectorizer).to(device)
    tgt = torch.tensor(list(targets), dtype=torch.long, device=device)
    counts = data.class_frequencies()
    weights: List[float] = []
    for t in targets:
        if int(t) == int(data.corpus.n_classes):
            weights.append(float(cfg.unknown_class_weight))
        else:
            c = max(1.0, float(counts[int(t)]))
            weights.append(1.0 / (c ** 0.5))
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return feats, tgt, w


# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """一次训练的结果（供报告与验收断言使用）。"""

    meta: Dict[str, Any]
    history: List[Dict[str, float]]
    model: N3DQA
    data: TrainingData
    device: str
    artifact_bytes_sha256: str
    artifact_path: str

    def final_loss(self) -> float:
        """最后一轮训练损失（训练后量，不作为构造期不变量断言）。"""
        return float(self.history[-1]["loss"]) if self.history else float("nan")


def label_index_early(rec: QARecord, data: TrainingData) -> int:
    """返回样本的候选下标：答案表内 -> 类别下标；答案表外 -> 「不相关」位（``C``）。"""
    index = data.corpus.key_to_index()
    return int(index.get(rec.answer_key, data.corpus.n_classes))


def run_training(
    cfg: TrainConfig,
    *,
    max_batches: int = 0,
    artifact_path: str = "",
    save: bool = True,
    encoder_name: str = "",
    encoder_config: Optional[EncoderConfig] = None,
) -> TrainingResult:
    """执行训练（含可选限批模式）并落盘产物。

    参数
    ----
    cfg : TrainConfig
        训练配置。
    max_batches : int
        每个 epoch 的最大 batch 数（``0`` = 全量）；用于验证类运行。
    artifact_path : str
        产物路径；空串时按 ``DEFAULT_ARTIFACT_DIR`` + 指纹名生成。
    save : bool
        是否落盘。
    encoder_name : str
        可插拔特征实现的注册表键（空串 = 角色默认实现，与历史行为逐位一致）。
    encoder_config : Optional[EncoderConfig]
        完整的编码器配置（**优先于** ``encoder_name``）。

    返回
    ----
    TrainingResult
        训练结果（含 meta、history、模型与产物字节 SHA256）。
    """
    t0 = time.time()
    set_deterministic_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_training_data(
        cfg, encoder_name=str(encoder_name), encoder_config=encoder_config
    )
    model, adapter, _registry = _build_model(data, cfg)
    model = model.to(device)
    model.train()

    # ---- 答案表（centroid 口径）：由训练样本的**原始文本特征**质心确定性写入 ----
    label_index = data.corpus.key_to_index()
    if cfg.output_mode == "index" and model.config.answer_table_mode == "centroid":
        all_train = list(data.splits.train_known) + list(data.splits.train_unknown)
        feats_all = _answer_vectors(all_train, data.vectorizer).to(device)
        tgt_all = torch.tensor(
            [label_index_early(r, data) for r in all_train],
            dtype=torch.long,
            device=device,
        )
        model.set_answer_table_from_centroids(
            feats_all, tgt_all, int(data.corpus.n_classes)
        )
        del feats_all, tgt_all

    n_candidates = int(model.output_dim)
    # ---- 训练样本：known（金标 = 答案表下标） + unknown（金标 = 末位「不相关」） ----
    train_items: List[Tuple[QARecord, int]] = [
        (rec, label_index[rec.answer_key]) for rec in data.splits.train_known
    ]
    train_items.extend(
        (rec, int(data.corpus.n_classes)) for rec in data.splits.train_unknown
    )
    if int(cfg.max_train_samples) > 0:
        # 确定性截断：先按 (是否为不相关, qid) 排序，再取前 N（限批验证类运行使用）
        train_items.sort(key=lambda it: (it[1] != data.corpus.n_classes, it[0].qid))
        train_items = train_items[: int(cfg.max_train_samples)]

    model.adapter.model.train(mode=bool(cfg.train_backbone))
    for p in model.adapter.model.parameters():
        p.requires_grad_(bool(cfg.train_backbone))
    if not bool(cfg.train_head):
        # 冻结 q 头（见 TrainConfig.train_head 的实测理由）；logit 尺度按配置单独处置
        for p in model.q_head.parameters():
            p.requires_grad_(False)
        model.logit_scale.requires_grad_(bool(cfg.learn_logit_scale))
    head_params = [p for p in model.q_head.parameters() if p.requires_grad]
    # 答案表只在 free 口径下才是可学习参数（centroid 口径下是绑定 buffer，不入优化器）
    if isinstance(getattr(model, "answer_table", None), nn.Parameter):
        head_params.append(model.answer_table)
    if model.logit_scale.requires_grad:
        head_params.append(model.logit_scale)
    groups: List[Dict[str, Any]] = [
        {"params": head_params, "lr": float(cfg.lr)},
    ]
    backbone_params = [p for p in model.adapter.model.parameters() if p.requires_grad]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(cfg.backbone_lr)})
    if not any(g["params"] for g in groups):
        raise RuntimeError(
            "没有任何可训练参数（q 头 / 答案表 / logit 尺度 / 后端全部被冻结）；"
            "请打开 --train-head 或 --train-backbone"
        )
    opt_cls = torch.optim.AdamW if float(cfg.weight_decay) > 0.0 else torch.optim.Adam
    optimizer: torch.optim.Optimizer = opt_cls(
        groups, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )

    gen = torch.Generator().manual_seed(int(cfg.seed))
    history: List[Dict[str, float]] = []
    for epoch in range(1, int(cfg.epochs) + 1):
        order = torch.randperm(len(train_items), generator=gen).tolist()
        total_loss = 0.0
        n_batches = 0
        for b0 in range(0, len(order), int(cfg.batch_size)):
            if int(max_batches) > 0 and n_batches >= int(max_batches):
                break
            idxs = order[b0 : b0 + int(cfg.batch_size)]
            batch = [train_items[i] for i in idxs]
            records = [it[0] for it in batch]
            targets = [it[1] for it in batch]
            feats, tgt, weight = _batch_tensors(
                records, targets, data, cfg, device
            )
            keys = (
                _pointer_keys(
                    data.corpus.answer_keys,
                    data.corpus.answer_display,
                    data.vectorizer,
                    len(records),
                ).to(device)
                if cfg.output_mode == "pointer"
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model.logits(feats, keys)
            loss = model.cross_entropy(logits, tgt, sample_weight=weight)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"训练损失非有限值（epoch={epoch}, batch={n_batches}）：loss={loss.item()}"
                )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
            n_batches += 1
        if n_batches == 0:
            raise RuntimeError("没有任何 batch 被训练（检查 max_batches / 数据规模）")
        history.append(
            {"epoch": int(epoch), "loss": float(total_loss / float(n_batches)),
             "batches": int(n_batches)}
        )

    model.eval()
    model.adapter.model.requires_grad_(False)
    state = _state_to_bytes(model.state_dict())
    state_sha = hashlib.sha256(state).hexdigest()
    meta = build_meta(cfg, data, adapter, model, history, state_sha, device=str(device))

    if not artifact_path:
        artifact_path = os.path.join(
            DEFAULT_ARTIFACT_DIR, artifact_name(cfg, data.vectorizer.dim)
        )
    blob = build_artifact_bytes(meta, state, model)
    sha = hashlib.sha256(blob).hexdigest()
    if save:
        os.makedirs(os.path.dirname(os.path.abspath(artifact_path)), exist_ok=True)
        with open(artifact_path, "wb") as handle:
            handle.write(blob)
    meta["artifact_sha256"] = sha

    return TrainingResult(
        meta=meta,
        history=history,
        model=model,
        data=data,
        device=str(device),
        artifact_bytes_sha256=sha,
        artifact_path=artifact_path if save else "",
    )


def _state_to_bytes(state: Dict[str, Any]) -> bytes:
    """``state_dict`` -> 字节（``torch.save`` 到内存，不落临时文件）。"""
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# meta / 产物
# ---------------------------------------------------------------------------


def build_meta(
    cfg: TrainConfig,
    data: TrainingData,
    adapter: BackendAdapter,
    model: N3DQA,
    history: Sequence[Dict[str, float]],
    state_sha256: str,
    device: str,
) -> Dict[str, Any]:
    """构造产物 meta（**答案表口径与向量化口径指纹必须在此**）。

    参数
    ----
    cfg : TrainConfig
        训练配置。
    data : TrainingData
        数据装配结果。
    adapter : BackendAdapter
        后端适配器。
    model : N3DQA
        训练好的模型。
    history : Sequence[Dict[str, float]]
        epoch 历史。
    state_sha256 : str
        ``state_dict`` 字节 SHA256。
    device : str
        训练设备。

    返回
    ----
    Dict[str, Any]
        可 JSON 化的 meta。
    """
    describe = model.describe()
    meta: Dict[str, Any] = {
        "module": "n3d_qa_learn",
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "dim": int(data.vectorizer.dim),
        "n_answers": int(data.corpus.n_classes),
        "answer_keys": list(data.corpus.answer_keys),
        "answer_display": dict(data.corpus.answer_display),
        "irrelevant_index": int(data.corpus.n_classes),
        "output_mode": str(cfg.output_mode),
        "backend": adapter.describe(),
        "model": describe,
        "vectorizer_config": data.vectorizer.config.to_dict(),
        "vectorizer_fingerprint": data.vectorizer.fingerprint(),
        "answer_table_sha256": answer_table_sha256(model, data.corpus.answer_keys,
                                                  data.corpus.answer_display),
        "answer_table_exists": bool(cfg.output_mode == "index"),
        "state_dict_sha256": str(state_sha256),
        "train_config": cfg.to_dict(),
        "split": data.splits.summary(),
        "class_counts_train": {
            k: int(v) for k, v in zip(
                data.corpus.answer_keys,
                [int(x) for x in data.class_frequencies()],
            )
        },
        "qa_files": [os.path.basename(p) for p in data.qa_files],
        "text_dir": str(cfg.text_dir),
        "n_text_lines": int(len(data.text_lines)),
        "history": [dict(h) for h in history],
        "train_backbone": bool(cfg.train_backbone),
        "backbone_parameter_count": int(
            sum(p.numel() for p in adapter.model.parameters())
        ),
        "head_parameter_count": int(
            sum(p.numel() for p in model.parameters()) - sum(
                p.numel() for p in adapter.model.parameters()
            )
        ),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return meta


def answer_table_sha256(
    model: N3DQA,
    answer_keys: Sequence[str],
    answer_display: Dict[str, str],
) -> str:
    """答案表**内容与口径指纹**（SHA256）。

    覆盖内容：答案键列表（顺序敏感）+ 展示文本 + 「不相关」位 + （若存在）候选键张量字节。

    参数
    ----
    model : N3DQA
        模型。
    answer_keys : Sequence[str]
        答案表键（顺序敏感）。
    answer_display : Dict[str, str]
        展示文本。

    返回
    ----
    str
        64 位十六进制小写串。
    """
    payload: Dict[str, Any] = {
        "answer_keys": list(answer_keys),
        "answer_display": [answer_display.get(k, k) for k in answer_keys],
        "irrelevant_index": int(len(answer_keys)),
        "has_answer_table": bool(hasattr(model, "answer_table")),
    }
    if hasattr(model, "answer_table"):
        table = model.answer_table.detach().to(torch.float32).cpu().contiguous()
        payload["table_shape"] = list(table.shape)
        payload["table_bytes_sha256"] = hashlib.sha256(
            table.numpy().tobytes()
        ).hexdigest()
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def artifact_name(cfg: TrainConfig, dim: int) -> str:
    """产物文件名（含后端 / 模式 / 维度 / 种子指纹，**防撞名**）。

    当 ``split_seed`` 被**显式给出**（``>= 0``）时额外附加 ``_sp<split_seed>`` 后缀；
    默认档（``-1``）的文件名与历史**逐字不变**，避免既有产物被改名或覆盖。
    """
    name = (
        f"qa_{cfg.backend}_{cfg.output_mode}_D{dim}"
        f"_C{cfg.max_classes}_mq{cfg.min_questions}_s{cfg.seed}.pt.zip"
    )
    if int(cfg.split_seed) >= 0:
        name = f"{name[: -len('.pt.zip')]}_sp{int(cfg.split_seed)}.pt.zip"
    return name


def build_artifact_bytes(meta: Dict[str, Any], state_bytes: bytes, model: N3DQA) -> bytes:
    """自写 zip：``meta.json`` + ``model_state_dict.pt`` + （index 模式）``answer_table.pt``。

    参数
    ----
    meta : Dict[str, Any]
        可 JSON 化 meta。
    state_bytes : bytes
        ``torch.save(state_dict)`` 的字节。
    model : N3DQA
        模型（用于取 ``answer_table``）。

    返回
    ----
    bytes
        zip 字节（固定时间戳，跨运行确定性）。
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("meta.json", date_time=ZIP_EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(
            info,
            json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=1),
        )

        info = zipfile.ZipInfo("model_state_dict.pt", date_time=ZIP_EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, state_bytes)

        if hasattr(model, "answer_table"):
            table_bytes = _state_to_bytes(
                {"answer_table": model.answer_table.detach().cpu()}
            )
            info = zipfile.ZipInfo("answer_table.pt", date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, table_bytes)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 加载守卫
# ---------------------------------------------------------------------------


def load_artifact(path: str, *, verify: bool = True) -> Dict[str, Any]:
    """加载产物并执行**两道守卫**。

    守卫
    ----
    1. **答案表指纹**：``meta["answer_table_sha256"]`` 必须与现场由答案键/展示文本
       （以及候选键张量字节）重算的值一致；
    2. **向量化口径指纹**：``meta["vectorizer_fingerprint"]`` 必须与现场口径一致
       （由 :func:`n3d_qa_learn.features.vectorizer_from_meta` 校验）。

    参数
    ----
    path : str
        产物路径。
    verify : bool
        是否执行守卫（``False`` 仅供诊断对照，**生产路径必须为 True**）。

    返回
    ----
    Dict[str, Any]
        ``{"meta", "state_dict", "answer_table"}``。

    异常
    ------
    FileNotFoundError
        产物不存在。
    ValueError
        任一道守卫失败（报文给出期望值与实测值）。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"产物不存在：{path!r}")
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        required = {"meta.json", "model_state_dict.pt"}
        missing = required - names
        if missing:
            raise ValueError(
                f"产物缺少必需成员 {sorted(missing)}；实际成员 = {sorted(names)}"
            )
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        state = torch.load(io.BytesIO(zf.read("model_state_dict.pt")),
                           map_location="cpu", weights_only=False)
        table = None
        if "answer_table.pt" in names:
            table = torch.load(io.BytesIO(zf.read("answer_table.pt")),
                               map_location="cpu", weights_only=False)["answer_table"]
    if not verify:
        return {"meta": meta, "state_dict": state, "answer_table": table}

    # ---- 守卫 1：答案表内容与口径指纹 ----
    payload: Dict[str, Any] = {
        "answer_keys": list(meta["answer_keys"]),
        "answer_display": [
            meta["answer_display"].get(k, k) for k in meta["answer_keys"]
        ],
        "irrelevant_index": int(meta["irrelevant_index"]),
        "has_answer_table": bool(meta.get("answer_table_exists", False)),
    }
    if table is not None:
        t = table.to(torch.float32).cpu().contiguous()
        payload["table_shape"] = list(t.shape)
        payload["table_bytes_sha256"] = hashlib.sha256(t.numpy().tobytes()).hexdigest()
    recomputed = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected = str(meta.get("answer_table_sha256", ""))
    if recomputed != expected:
        raise ValueError(
            "答案表指纹校验失败：产物 meta 记录 "
            f"{expected[:16]}...，现场重算 {recomputed[:16]}...；"
            "答案表内容（答案键 / 展示文本 / 「不相关」位 / 候选键张量）已被改动，拒绝加载"
        )

    # ---- 守卫 2：向量化口径指纹 ----
    # 走**按口径分派**的重建入口：hash 家族委派 features 的历史路径（逐位不变），
    # HF 家族按编码器声明重建（revision + 权重 SHA256 任一变化都会改指纹）。
    vectorizer_from_meta(meta)
    return {"meta": meta, "state_dict": state, "answer_table": table}


def rebuild_model(meta: Dict[str, Any], state_dict: Dict[str, Any]) -> N3DQA:
    """按产物 meta 重建 ``N3DQA`` 并载入 ``state_dict``（严格模式）。

    参数
    ----
    meta : Dict[str, Any]
        产物 meta。
    state_dict : Dict[str, Any]
        模型状态字典。

    返回
    ----
    N3DQA
        重建并载入权重的模型（``eval()``）。

    异常
    ------
    RuntimeError
        状态字典与重建模型结构不一致（``strict=True``）。
    """
    vectorizer = vectorizer_from_meta(meta)
    registry = build_registry(vectorizer.config)
    backend = str(meta["backend"]["backend"])
    adapter = registry.get(backend)
    model = N3DQA(
        adapter,
        int(meta["n_answers"]),
        N3DQAConfig(
            dim=int(meta["dim"]),
            output_mode=str(meta["output_mode"]),
            head_input_mode=str(meta["model"].get("head_input_mode", "raw")),
            label_smoothing=float(meta["model"].get("label_smoothing", 0.0)),
            # [!] 这里只填**重建用的固定初值常量**：真实（训练后）尺度由紧随其后的
            #     load_state_dict 从 state_dict 恢复（logit_scale 是 Parameter，必在其中）。
            #     历史缺陷：原先写 float(meta["model"].get("logit_scale", 20.0))，把
            #     describe() 记录的**训练后数值**当成构造初值用，语义错位。
            logit_scale_init=REBUILD_LOGIT_SCALE_INIT,
            learn_logit_scale=False,
            answer_table_mode=str(meta["model"].get("answer_table_mode", "centroid")),
        ),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


__all__ = [
    "label_index_early",
    "DEFAULT_ARTIFACT_DIR",
    "DEFAULT_VERIFY_DIR",
    "ZIP_EPOCH",
    "ARTIFACT_FORMAT_VERSION",
    "IRRELEVANT_KEY_TEXT",
    "TrainConfig",
    "TrainingData",
    "TrainingResult",
    "set_deterministic_seed",
    "split_seed_of",
    "build_training_data",
    "run_training",
    "build_meta",
    "answer_table_sha256",
    "artifact_name",
    "build_artifact_bytes",
    "load_artifact",
    "rebuild_model",
]