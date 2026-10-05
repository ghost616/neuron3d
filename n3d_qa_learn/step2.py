"""n3d_qa_learn 步骤 2：文本数据集匹配（两级路由的第二级）+ 分项验收协议。

职责
----
把**文本数据集匹配**接到 N3D 的 ``q`` 通路上，并给出可审计的分项验收：

1. **候选来源按全局开关二选一**（与 :mod:`n3d_qa_learn.heads` 的 ``OUTPUT_MODES`` 同一开关）：

   * ``index``（索引式）—— 候选键表 ``K in R^[L, D]`` 由**库行文本的确定性特征**构造、
     **冻结不参与梯度**，并固化进本模块自己的产物 zip；
   * ``pointer``（指针式）—— 候选键表 ``K in R^[B, L, D]`` 由**输入提供**，
     产物内不存任何固定候选表。

   两模式共用**同一个训练出的 ``q``**：``q = q_head(adapter.features(x))``，
   打分 ``logits = q @ K^T`` -> ``softmax`` -> ``top-1`` 行原文。

2. **自检索留出法**（步骤 2 主判据）：库行与查询行来自 ``n3d_qa`` 冻结产物的**互斥划分**
   （``intersection == 0``）；查询行经**同一条 q 通路**在库中检索，命中自身行即正确。
   报告 ``Recall@1`` 与 ``Recall@5``。

3. **参照下限**（如实报告，不作为门槛）：不经 N3D 的**纯确定性特征检索** ``Recall@1/@5``。

4. **双模式一致率**：同一批输入上两模式各跑一次，报告 ``top-1`` 一致率（只报实测值）。

5. **分项 5 项指标**（按任务分别产出）：① 步骤 1 答案准确率；② 步骤 2 ``Recall@1``；
   ③ 路由正确率（命中 / 未命中判定）；④ 无匹配类的精确率与召回；⑤ 端到端最终答案正确率。

6. **跨 seed 报告**：``seed in {42, 43, 44}``，按任务与分项分别报 ``均值 + 极差``。

上游边界（零改动）
------------------
* **只读** import ``n3d_qa.zh_features``（中文确定性文本特征）与
  ``n3d_shape`` / ``n3d_sphere`` / ``n3d_proto`` 的 ``config`` / ``model``（经
  :mod:`n3d_qa_learn.backends` 的代理层）；
* 复用本模块既有的 :mod:`n3d_qa_learn.heads` / :mod:`n3d_qa_learn.route` /
  :mod:`n3d_qa_learn.evaluate` / :mod:`n3d_qa_learn.train` 的**公开接口**，不修改它们；
* 产物一律写 ``checkpoints/qa_learn/step2/``，验证类运行写
  ``checkpoints/qa_learn/_verify/step2/``。

「不相关（未命中）」的构造口径（显式、可复现）
--------------------------------------------
``n3d_qa`` 冻结产物只导出**已入表答案**的正样本（现场实测：``judge`` 移除 0 题、
``choice`` 移除 15 题、``blank`` 移除 3060 题、``solve`` 移除 21864 题，被移除的题目
**不随产物导出**），因此不能从产物里直接取到「正确答案不在表内」的样本。
本模块采用**留出类（held-out classes）**这一显式开集协议：按固定 seed 从统一答案表
（199 类）中留出一定比例的类 ``H``，模型的可学候选空间 = ``A - H``；
答案不在 ``H`` 的题目 -> **已学（命中）**；答案在 ``H`` 的题目 -> **不相关**。
``H`` 的类清单与 SHA256 写入报告，**冻结**（不随运行重抽）。
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from n3d_qa.zh_features import ZhFeatureConfig, hash_bag_vector, l2_normalize_block, zh_normalize

from .backends import BACKEND_NAMES, BackendAdapter, BackendRegistry
from .data import QACorpus, QARecord, SplitSpec, TextRecord
from .encoders import (
    ENCODER_ZH_BAG,
    ROLE_TEXT_LINE,
    EncodeResult,
    EncoderConfig,
    EmbeddingCache,
    build_vectorizer,
)
from .evaluate import evaluate_refusal, evaluate_step1
from .heads import OUTPUT_MODES, N3DQA, N3DQAConfig
from .route import NO_MATCH_TEXT, QuestionRouter
from .train import IRRELEVANT_KEY_TEXT, ZIP_EPOCH, TrainingData, set_deterministic_seed

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

#: 步骤 2 产物格式版本（结构变化必须递增）。
STEP2_ARTIFACT_VERSION: str = "n3dqa-step2-art-v1"

#: 步骤 2 正式产物目录。
STEP2_DIR: str = os.path.join("checkpoints", "qa_learn", "step2")

#: 步骤 2 验证类运行目录。
STEP2_VERIFY_DIR: str = os.path.join("checkpoints", "qa_learn", "_verify", "step2")

#: ``n3d_qa`` 冻结产物的候选目录（按顺序探测；**只读**）。
N3D_QA_PRODUCT_DIRS: Tuple[str, ...] = (
    os.path.join("checkpoints", "qa_learn", "dataset"),
    os.path.join("checkpoints", "qa_learn", "_verify", "drill4"),
)

#: 四个中文任务（``n3d_qa`` 产物的 ``task`` 字段取值）。
TASKS: Tuple[str, ...] = ("judge", "choice", "blank", "solve", "triviaqa")

#: 中文文本特征口径（与 ``n3d_qa`` 产物 ``doclines_rows.jsonl`` 同 salt / 同归一化；
#: 本模块只取**字符 n-gram 哈希词袋块**：桶数 64/阶、阶数 (1,2,3) -> D = 192）。
N_GRAM_ORDERS: Tuple[int, ...] = (1, 2, 3)
BUCKETS_PER_ORDER: int = 64

#: 连接参数 D（= buckets_per_order * len(n_gram_orders)）。
FEATURE_DIM: int = int(BUCKETS_PER_ORDER) * len(N_GRAM_ORDERS)

#: Recall@k 的 k。
TOPK: int = 5

#: 跨 seed 报告的种子集合（训练侧 seed）。
TRAIN_SEEDS: Tuple[int, ...] = (42, 43, 44)

#: 留出类比例（开集协议）。
HELD_OUT_RATIO: float = 0.2

#: 留出类抽取的固定 seed（冻结口径）。
HELD_OUT_SEED: int = 20261010

#: 特征重算与冻结产物的容差（float32 舍入；全量实测最大偏差见报告）。
FEATURE_ATOL: float = 1e-6

#: 默认后端。
DEFAULT_BACKEND: str = "n3d_shape"

#: 步骤 2 默认的 q 头输入口径（**必须显式固定，不得吃上游默认值**）。
#:
#: 取值语义（随上游 ``heads.py`` 的 ``head_input_mode``）：
#: * ``"raw"``   —— 只用原始文本特征，**N3D 骨干不在计算图上**（骨干参数永远拿不到梯度）；
#: * ``"n3d"``   —— 只用 N3D 读出；
#: * ``"concat"``—— 原始特征与 N3D 读出的可学习凸组合（``mix_logit`` 控权），两路都在图上。
#:
#: 本模块取 ``"concat"``：它既真实经过 N3D 骨干（保住「接入 N3D 的 q 通路」这一语义），
#: 又让**全部可学习参数都参与前向**，从而单条演练的「梯度非零」硬门禁是严格的。
DEFAULT_HEAD_INPUT_MODE: str = "concat"
# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def sha256_bytes(blob: bytes) -> str:
    """字节流的 SHA256（小写十六进制）。"""
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """文件的 SHA256（流式读取，不整文件入内存）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_dumps(obj: Any) -> bytes:
    """规范化 JSON 字节（排序键 + 紧凑分隔符），用于指纹计算。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _rng(seed: int) -> np.random.Generator:
    """确定性 numpy 生成器（局部，不消耗全局 RNG）。"""
    return np.random.default_rng(int(seed))


# ---------------------------------------------------------------------------
# 文本行确定性向量化（复用 n3d_qa 的中文特征口径）
# ---------------------------------------------------------------------------


def zh_feature_config(buckets_per_order: int = BUCKETS_PER_ORDER) -> ZhFeatureConfig:
    """返回本模块使用的冻结中文特征口径（与 n3d_qa 产物同 salt / 同归一化）。"""
    return ZhFeatureConfig(
        n_gram_orders=N_GRAM_ORDERS, buckets_per_order=int(buckets_per_order)
    )


class ZhBagVectorizer:
    """D 维确定性文本向量化器（复用 ``n3d_qa.zh_features`` 的哈希词袋）。

    与 :class:`n3d_qa_learn.features.TextVectorizer` 的**接口**保持一致
    （``dim`` / ``fingerprint`` / ``encode`` / ``encode_batch``），以便本模块既有的
    :mod:`n3d_qa_learn.evaluate` 评估协议可以**零改动**复用。

    关键不变量
    ----------
    * 不消耗任何随机数（纯 blake2b 哈希 + 计数）；
    * 同一文本重复调用**逐位一致**；
    * 维度恒为 D = ``buckets_per_order * len(n_gram_orders)``。
    """

    def __init__(self, config: Optional[ZhFeatureConfig] = None) -> None:
        self.config: ZhFeatureConfig = config if config is not None else zh_feature_config()

    @property
    def dim(self) -> int:
        """特征维度 D（不含 n3d_qa 的 6 列 pair 附加列）。"""
        return int(self.config.bag_dim)

    def fingerprint(self) -> str:
        """口径指纹：n3d_qa 特征规格的 ``spec_hash``。"""
        return str(self.config.spec_hash())

    def encode(self, text: str) -> List[float]:
        """把一段文本编码为 D 维 L2 归一化词袋（``list[float]``）。"""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """批量编码（逐条确定性；空文本 -> 全零向量，不产生 NaN）。"""
        out: List[List[float]] = []
        for text in texts:
            vec = hash_bag_vector(str(text), self.config)
            l2_normalize_block(vec)
            out.append([float(x) for x in vec.tolist()])
        return out

    def encode_matrix(self, texts: Sequence[str]) -> np.ndarray:
        """批量编码为 ``float32[L, D]``（供冻结键表构造与参照下限使用）。"""
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            row = hash_bag_vector(str(text), self.config)
            mat[i] = l2_normalize_block(row)
        return mat

    def encode_with_stats(self, text: str) -> EncodeResult:
        """单条编码 + 统计（**统一接口的补齐**）。

        口径（**如实**）：n-gram 哈希词袋**没有词元上限**，因此
        ``n_tokens`` / ``n_truncated`` 恒为 ``0``；该家族在注册表里声明
        ``truncation_stats=False``，故「超长 -> n_truncated > 0」这条边界对它是
        **不适用**而不是失败（见 :func:`n3d_qa_learn.encoders.summarize_selftest`）。

        存在理由：统一接口要求 ``encode_with_stats`` 可用 —— 缺失会让
        ``encoders_run drill / boundary`` 在 ``zh-bag`` 上抛
        ``AttributeError: 'ZhBagVectorizer' object has no attribute 'encode_with_stats'``。
        """
        if not isinstance(text, str):
            raise TypeError(f"encode 需要 str，当前类型 {type(text).__name__}")
        return EncodeResult(vector=self.encode(text), n_tokens=0, n_truncated=0, cached=False)


# ---------------------------------------------------------------------------
# 步骤 2 的可插拔编码器入口 + 重建后的验证口径
# ---------------------------------------------------------------------------


def build_step2_vectorizer(cfg: Optional[EncoderConfig] = None) -> Any:
    """构造步骤 2 的文本行向量化器（**可插拔编码器注册表的步骤 2 入口**）。

    参数
    ----
    cfg : Optional[EncoderConfig]
        编码器选择配置；``None`` = ``zh-bag`` + ``role="text_line"``（现状口径，
        与历史逐位一致）。切 HF 编码器需显式给 ``EncoderConfig(name="bge-m3",
    role=ROLE_TEXT_LINE)``（文本行 ``max_length = 8192``）。

    返回
    ----
    Any
        满足统一接口（``dim`` / ``fingerprint`` / ``encode`` / ``encode_batch`` /
        ``encode_matrix``）的向量化器。
    """
    config = cfg if cfg is not None else EncoderConfig(
        name=ENCODER_ZH_BAG, role=ROLE_TEXT_LINE
    )
    return build_vectorizer(config)


def verify_vectorizer_contract(
    vectorizer: Any,
    texts: Sequence[str],
    *,
    registry_name: str = "",
    role: str = ROLE_TEXT_LINE,
    declared_fingerprint: str = "",
    declared_source: str = "",
    canonical: Any = None,
) -> Dict[str, Any]:
    """步骤 2 的**验证口径（第 2 轮重建，替代「与 n3d_qa 冻结产物逐位比对」）**。

    三条检查
    --------
    ① ``fingerprint()`` == 注册表 / 落盘声明 —— 证明步骤 2 用的就是注册表登记的口径
       （修订 / 权重 / 池化 / ``max_length`` 任一变化都会改变指纹）；
    ② 同文本重复编码**逐位一致** —— 证明确定性（无随机、无跨调用状态）；
    ③ 与**落盘缓存**逐位比对 —— 证明「跨进程复用同一缓存」不改变任何一位。

    口径声明（**如实**）
    ------------------
    * 旧的 `verify_features_against_product`（与 ``n3d_qa`` 冻结产物 ``feature`` 字段
      逐元素比对）**降级为 hash 家族的旁证**，不再是本接口的验证口径 —— 它对
      HF 编码器无法成立（两者根本不在同一特征空间）；
    * ③ 只在向量化器挂了落盘缓存时适用（``applicable=False`` 时不计入 ``passed``）。

    参数
    ----
    vectorizer : Any
        被测向量化器。
    texts : Sequence[str]
        比对用文本（至少 1 条）。
    registry_name : str
        注册表键（用于在未给声明时现场构造基准）。
    role : str
        角色。
    declared_fingerprint : str
        落盘声明里的指纹；非空时作为 ① 的基准。
    declared_source : str
        声明来源（取证用）。
    canonical : Any
        已构造好的基准向量化器（避免重复加载模型）；为空则按注册表现场构造。

    返回
    ----
    Dict[str, Any]
        ``{check1, check2, check3, passed, ...}``。
    """
    items = [str(t) for t in texts]
    if not items:
        raise ValueError("verify_vectorizer_contract 至少需要 1 条文本")

    # --- ① 指纹 == 注册表 / 落盘声明 ------------------------------------
    actual_fp = str(vectorizer.fingerprint())
    expected_fp = str(declared_fingerprint)
    basis = "落盘声明"
    if not expected_fp:
        if canonical is None:
            canonical = build_vectorizer(
                EncoderConfig(name=str(registry_name) or ENCODER_ZH_BAG, role=str(role))
            )
        expected_fp = str(canonical.fingerprint())
        basis = "注册表现场构造"
    check1 = bool(expected_fp) and (actual_fp == expected_fp)

    # --- ② 同文本重复编码逐位一致 ---------------------------------------
    # 判据**只用 float32 裸字节**（修复 N1）：缓存命中会把条目里的 float32 还原成
    # Python float，与首次实算的 float64 值在末位可能不同，若再叠一层 list 相等
    # 判断会产生「字节相同但判失败」的假阴性。逐位一致的语义就是字节一致。
    first = [vectorizer.encode(t) for t in items]
    second = [vectorizer.encode(t) for t in items]
    first_bytes = b"".join(np.asarray(v, dtype="<f4").tobytes() for v in first)
    second_bytes = b"".join(np.asarray(v, dtype="<f4").tobytes() for v in second)
    check2 = bool(first_bytes == second_bytes)

    # --- ③ 与落盘缓存逐位比对 -------------------------------------------
    attached = getattr(vectorizer, "cache", None)
    cache = attached if isinstance(attached, EmbeddingCache) else None
    per_text: List[Dict[str, Any]] = []
    if cache is not None and hasattr(vectorizer, "cache_key"):
        for text, vec in zip(items, first):
            key = str(vectorizer.cache_key(text))
            blob = cache.read_bytes(key)
            encoded = np.asarray(vec, dtype="<f4").tobytes()
            per_text.append(
                {
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "cache_key": key,
                    "cache_written": bool(blob is not None),
                    "bitwise_equal": bool(blob is not None and blob == encoded),
                    "bytes": int(len(encoded)),
                }
            )
    applicable3 = bool(per_text)
    check3 = bool(applicable3 and all(e["bitwise_equal"] for e in per_text))

    return {
        "registry_name": str(registry_name),
        "role": str(role),
        "dim": int(vectorizer.dim),
        "n_texts": int(len(items)),
        "check1_fingerprint_matches_declaration": {
            "passed": bool(check1),
            "basis": str(basis),
            "declared_source": str(declared_source),
            "declared_fingerprint": str(expected_fp),
            "actual_fingerprint": str(actual_fp),
        },
        "check2_repeat_encode_bitwise_equal": {
            "passed": bool(check2),
            "n_bytes": int(len(first_bytes)),
            "first_sha256": hashlib.sha256(first_bytes).hexdigest(),
            "second_sha256": hashlib.sha256(second_bytes).hexdigest(),
        },
        "check3_cache_bitwise_equal": {
            "passed": bool(check3),
            "applicable": bool(applicable3),
            "note": (
                "未挂载落盘缓存 -> 本项不适用（不计入 passed）"
                if not applicable3 else "逐条与落盘缓存裸字节比对"
            ),
            "per_text": per_text,
        },
        "passed": bool(check1 and check2 and (check3 if applicable3 else True)),
    }


# ---------------------------------------------------------------------------
# n3d_qa 冻结产物读取（只读）
# ---------------------------------------------------------------------------


def resolve_product_dir(explicit: str = "") -> str:
    """定位 ``n3d_qa`` 冻结产物目录。

    参数
    ----
    explicit : str
        显式目录；非空时必须是**含 doclines.npz** 的目录。

    返回
    ----
    str
        产物目录。

    异常
    ------
    FileNotFoundError
        显式目录不存在 / 候选目录都缺产物时抛出（报文列出候选）。
    """
    if explicit:
        if not os.path.isdir(explicit):
            raise FileNotFoundError(f"产物目录不存在：{explicit!r}")
        if not os.path.isfile(os.path.join(explicit, "doclines.npz")):
            raise FileNotFoundError(
                f"目录 {explicit!r} 下缺少 doclines.npz；实际内容 = "
                f"{sorted(os.listdir(explicit))[:20]}"
            )
        return explicit
    for candidate in N3D_QA_PRODUCT_DIRS:
        if os.path.isfile(os.path.join(candidate, "doclines.npz")):
            return candidate
    raise FileNotFoundError(
        f"未找到 n3d_qa 冻结产物（doclines.npz）；候选目录 = {list(N3D_QA_PRODUCT_DIRS)}"
    )


@dataclass(frozen=True)
class TextRow:
    """文本数据集的一行（原文 + 冻结特征 + 行 id）。"""

    row_id: str
    file: str
    line_no: int
    text: str
    feature: np.ndarray


def load_text_rows(product_dir: str) -> List[TextRow]:
    """读取 ``doclines_rows.jsonl``（n3d_qa 冻结产物；只读）。"""
    path = os.path.join(product_dir, "doclines_rows.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少文本行表：{path!r}")
    rows: List[TextRow] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                TextRow(
                    row_id=str(obj["row_id"]),
                    file=str(obj["file"]),
                    line_no=int(obj["line_no"]),
                    text=str(obj["text"]),
                    feature=np.asarray(obj["feature"], dtype=np.float32),
                )
            )
    if not rows:
        raise RuntimeError(f"文本行表为空：{path!r}")
    return rows


def load_row_index(product_dir: str) -> List[Dict[str, Any]]:
    """读取 ``doclines_row_index.jsonl``（用于**独立复核**库/查询划分）。"""
    path = os.path.join(product_dir, "doclines_row_index.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少行索引表：{path!r}")
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_doclines_meta(product_dir: str) -> Dict[str, Any]:
    """读取 ``doclines.npz`` 的 meta（含库/查询划分口径）。"""
    path = os.path.join(product_dir, "doclines.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少 doclines.npz：{path!r}")
    with np.load(path, allow_pickle=True) as data:
        return json.loads(str(data["meta"]))

@dataclass(frozen=True)
class DocSplit:
    """自检索留出法的**库 / 查询**划分（构造期不变量）。"""

    library_index: np.ndarray
    query_index: np.ndarray
    seed: int
    evidence: Dict[str, Any]

    @property
    def n_library(self) -> int:
        """库行数。"""
        return int(self.library_index.size)

    @property
    def n_query(self) -> int:
        """查询行数。"""
        return int(self.query_index.size)


def reproduce_doc_split(
    rows: Sequence[TextRow],
    row_index: Sequence[Dict[str, Any]],
    meta: Dict[str, Any],
) -> DocSplit:
    """复现 n3d_qa 冻结的库 / 查询划分，并用产物自身的行索引**独立复核**。

    复现规则（来自产物 meta 的 ``split`` 与 ``determinism`` 两节，逐字）：
    ``perm = np.random.default_rng(seed).permutation(rows_total)``；
    前 ``query_rows`` 个为查询行，其余为库行。

    独立复核（**不依赖复现结果**）：``doclines_row_index.jsonl`` 中
    ``positives >= 1`` 的行集合必须与复现出的查询行集合**完全相等**。

    参数
    ----
    rows : Sequence[TextRow]
        全部文本行（须按产物顺序）。
    row_index : Sequence[Dict[str, Any]]
        产物行索引表。
    meta : Dict[str, Any]
        ``doclines.npz`` 的 meta。

    返回
    ----
    DocSplit
        库 / 查询下标与取证信息。

    异常
    ------
    ValueError
        复核失败（交集非空 / 并集不等于全量 / 与行索引表不一致）时抛出。
    """
    split = dict(meta["split"])
    seed = int(split["seed"])
    n_query = int(split["query_rows"])
    total = int(len(rows))
    perm = _rng(seed).permutation(total)
    query_index = np.sort(perm[:n_query])
    library_index = np.sort(perm[n_query:])

    q_ids = {rows[int(i)].row_id for i in query_index}
    l_ids = {rows[int(i)].row_id for i in library_index}
    idx_query_ids = {
        str(entry["row_id"]) for entry in row_index if int(entry.get("positives", 0)) >= 1
    }
    evidence: Dict[str, Any] = {
        "reproduce_rule": "np.random.default_rng(split.seed).permutation(rows_total)",
        "seed": int(seed),
        "n_query": int(n_query),
        "n_library": int(library_index.size),
        "intersection": int(len(q_ids & l_ids)),
        "union": int(len(q_ids | l_ids)),
        "rows_total": int(total),
        "index_positives_ge_1": int(len(idx_query_ids)),
        "reproduced_equals_index": bool(q_ids == idx_query_ids),
        "meta_expected": {
            "library_rows": int(split["library_rows"]),
            "query_rows": int(split["query_rows"]),
            "intersection": int(split["intersection"]),
            "union_size": int(split["union_size"]),
        },
    }
    if evidence["intersection"] != 0:
        raise ValueError(
            f"库与查询集合交集非空（{evidence['intersection']}）；"
            "拒绝在一个有泄漏的划分上做自检索"
        )
    if evidence["union"] != total:
        raise ValueError(f"库并查询 = {evidence['union']}，不等于全量 {total}；划分不完整")
    if not evidence["reproduced_equals_index"]:
        raise ValueError(
            "复现出的查询行集合与产物行索引表（positives>=1）不一致："
            f"复现 {len(q_ids)} 行 vs 索引表 {len(idx_query_ids)} 行，"
            f"交集 {len(q_ids & idx_query_ids)} 行；拒绝在无法复核的划分上出指标"
        )
    if int(library_index.size) != int(split["library_rows"]):
        raise ValueError(
            f"复现库行数 {library_index.size} 与产物 meta 声明 {split['library_rows']} 不一致"
        )
    return DocSplit(
        library_index=library_index, query_index=query_index, seed=seed, evidence=evidence
    )


def verify_features_against_product(
    rows: Sequence[TextRow], config: ZhFeatureConfig, sample: int = 0
) -> Dict[str, Any]:
    """用产物自身的 ``feature`` 字段复核本模块的文本特征实现。

    产物的 ``feature`` 是 ``build_feature_vector(text, text)`` 的**自配对**口径：词袋块为
    ``L2(hash_bag(text + "\\n" + text))``（``zh_normalize`` 丢弃空白，故等价于把文本接两遍）。
    本模块的**行向量**取同口径下的**单份文本**词袋 ``L2(hash_bag(text))``。两者的差别仅在
    接缝处的少量 n-gram，因此本函数以**自配对口径**做重算比对，从而对
    「哈希盐 / 桶数 / 阶数 / L2 归一化」四件事给出**产物驱动**的等价证明。

    **口径地位（第 2 轮重建后，如实声明）**
    ------------------------------------
    本函数**只是 hash 家族的旁证**，**不再是步骤 2 的验证口径**。步骤 2 的验证口径已
    重建为 :func:`verify_vectorizer_contract` 的三条（指纹对账 / 重复编码逐位一致 /
    与落盘缓存逐位比对）；本函数的「与 ``n3d_qa`` 冻结产物逐元素比对」对可插拔接口
    （尤其 HF 编码器）根本无从成立 —— 两者不在同一特征空间。

    参数
    ----
    rows : Sequence[TextRow]
        文本行（含冻结 ``feature``）。
    config : ZhFeatureConfig
        现行口径。
    sample : int
        只检查前 ``sample`` 行（``0`` = 全量）。

    返回
    ----
    Dict[str, Any]
        取证信息：检查行数、最大绝对偏差、超容差行数、容差。
    """
    use = list(rows) if int(sample) <= 0 else list(rows)[: int(sample)]
    dim = int(config.bag_dim)
    worst = 0.0
    over = 0
    for row in use:
        joined = zh_normalize(row.text) + "\n" + zh_normalize(row.text)
        recomputed = hash_bag_vector(joined, config)
        l2_normalize_block(recomputed)
        frozen = np.asarray(row.feature, dtype=np.float32)[:dim]
        dev = float(np.abs(recomputed - frozen).max()) if frozen.size else 0.0
        worst = max(worst, dev)
        if dev > FEATURE_ATOL:
            over += 1
    return {
        "checked_rows": int(len(use)),
        "rows_total": int(len(rows)),
        "bag_dim": int(dim),
        "atol": float(FEATURE_ATOL),
        "max_abs_deviation": float(worst),
        "rows_over_atol": int(over),
        "spec_hash": str(config.spec_hash()),
    }

# ---------------------------------------------------------------------------
# QA 任务数据（统一答案表 + 留出类开集协议）
# ---------------------------------------------------------------------------


def load_answer_table(product_dir: str) -> Tuple[List[str], Dict[str, str], List[Dict[str, Any]]]:
    """读取 ``answer_table.jsonl``（统一答案表）。

    返回
    ----
    Tuple[List[str], Dict[str, str], List[Dict[str, Any]]]
        ``(答案键按 index 升序, 键->展示文本, 原始条目)``。
    """
    path = os.path.join(product_dir, "answer_table.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少统一答案表：{path!r}")
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda e: int(e["index"]))
    keys = [str(e["key"]) for e in entries]
    display = {str(e["key"]): str(e.get("display", e["key"])) for e in entries}
    if len(set(keys)) != len(keys):
        raise ValueError("统一答案表存在重复键；拒绝在非唯一键上建候选空间")
    return keys, display, entries


def load_task_pairs(product_dir: str, task: str) -> List[Dict[str, Any]]:
    """读取某任务的 ``*_pairs.jsonl``（含正负候选对与标签）。"""
    path = os.path.join(product_dir, f"n3d_qa_{task}_pairs.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少任务产物：{path!r}")
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    if not out:
        raise RuntimeError(f"任务 {task!r} 的候选对为空：{path!r}")
    return out


def select_held_out_classes(
    answer_keys: Sequence[str], ratio: float = HELD_OUT_RATIO, seed: int = HELD_OUT_SEED
) -> Tuple[List[str], List[str]]:
    """按固定 seed 从统一答案表留出一部分类（开集协议，**冻结**）。

    参数
    ----
    answer_keys : Sequence[str]
        统一答案表（顺序敏感）。
    ratio : float
        留出比例（``0 < ratio < 1``）。
    seed : int
        抽取 seed。

    返回
    ----
    Tuple[List[str], List[str]]
        ``(保留类 A - H, 留出类 H)``，两者均保持原表顺序。
    """
    keys = list(answer_keys)
    n_total = len(keys)
    if not (0.0 < float(ratio) < 1.0):
        raise ValueError(f"留出比例必须落在 (0, 1)，当前 {ratio}")
    n_hold = max(1, int(round(n_total * float(ratio))))
    if n_hold >= n_total:
        raise ValueError(f"留出类数 {n_hold} 不得覆盖整张表（{n_total} 类）；请减小比例")
    perm = _rng(seed).permutation(n_total)
    hold_idx = set(int(i) for i in perm[:n_hold])
    kept = [k for i, k in enumerate(keys) if i not in hold_idx]
    held = [k for i, k in enumerate(keys) if i in hold_idx]
    return kept, held


@dataclass
class TaskData:
    """一个任务的装配结果（模型候选空间为**全局统一**的保留类表）。"""

    task: str
    corpus: QACorpus
    splits: SplitSpec
    held_out_keys: List[str]
    n_positives: int
    n_questions: int
    n_unknown_questions: int
    n_known_questions: int

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化摘要。"""
        return {
            "task": str(self.task),
            "n_classes_kept": int(self.corpus.n_classes),
            "n_held_out": int(len(self.held_out_keys)),
            "n_positives": int(self.n_positives),
            "n_questions": int(self.n_questions),
            "n_known_questions": int(self.n_known_questions),
            "n_unknown_questions": int(self.n_unknown_questions),
            **{f"split_{k}": int(v) for k, v in self.splits.summary().items()},
        }


def build_task_data(
    product_dir: str,
    task: str,
    kept_keys: Sequence[str],
    held_keys: Sequence[str],
    answer_display: Dict[str, str],
    *,
    test_every: int = 3,
    seed: int = 42,
) -> TaskData:
    """把某任务的候选对装配成 ``QACorpus`` / ``SplitSpec``（已知 + 不相关）。

    口径
    ----
    * 金标答案在保留类 -> **已知**；金标答案在留出类 -> **不相关**；
    * 同一 ``question_id`` 的多条正样本（多正确答案）合并为一条记录（取排序后首个键）；
    * 测试侧按 ``question_id`` 的确定性置换每 ``test_every`` 条取 1 条，避免小类任务
      （judge 仅 2 类）的测试集退化。

    参数
    ----
    product_dir : str
        产物目录。
    task : str
        任务名。
    kept_keys : Sequence[str]
        保留类（= 模型候选空间，顺序即类别下标）。
    held_keys : Sequence[str]
        留出类。
    answer_display : Dict[str, str]
        键 -> 展示文本。
    test_every : int
        每 N 个问题取 1 个进测试侧（``>= 2``）。
    seed : int
        确定性排序 seed。

    返回
    ----
    TaskData
        装配结果。
    """
    if int(test_every) < 2:
        raise ValueError(f"test_every 必须 >= 2，当前 {test_every}")
    kept_set = set(kept_keys)
    held_set = set(held_keys)
    pairs = load_task_pairs(product_dir, task)

    gold: Dict[str, set] = {}
    qtext: Dict[str, str] = {}
    for obj in pairs:
        qid = str(obj["question_id"])
        qtext.setdefault(qid, str(obj["question_text"]))
        if int(obj["label"]) != 1:
            continue
        key = str(obj["candidate_answer_key"])
        if key in kept_set or key in held_set:
            gold.setdefault(qid, set()).add(key)
    if not gold:
        raise RuntimeError(f"任务 {task!r} 没有可用的正样本（候选对里 label==1 为空）")

    qids = sorted(gold.keys())
    order = _rng(seed).permutation(len(qids))
    test_positions = set(range(0, len(qids), int(test_every)))
    test_qids = {qids[int(order[i])] for i in range(len(qids)) if i in test_positions}
    train_qids = [q for q in qids if q not in test_qids]

    def _make(qid: str) -> QARecord:
        key = sorted(gold[qid])[0]
        return QARecord(
            qid=str(qid),
            question=str(qtext[qid]),
            answer_key=str(key),
            answer_display=str(answer_display.get(key, key)),
            source=str(task),
        )

    train_known: List[QARecord] = []
    train_unknown: List[QARecord] = []
    test_known: List[QARecord] = []
    test_unknown: List[QARecord] = []
    for qid in train_qids:
        rec = _make(qid)
        (train_unknown if rec.answer_key in held_set else train_known).append(rec)
    for qid in sorted(test_qids):
        rec = _make(qid)
        (test_unknown if rec.answer_key in held_set else test_known).append(rec)

    corpus = QACorpus(
        records=sorted(train_known + train_unknown + test_known + test_unknown,
                       key=lambda r: r.qid),
        answer_keys=list(kept_keys),
        answer_display={k: str(answer_display.get(k, k)) for k in kept_keys},
        class_counts={},
    )
    splits = SplitSpec(
        train_known=train_known,
        train_unknown=train_unknown,
        test_known=test_known,
        test_unknown=test_unknown,
    )
    n_pos = sum(1 for obj in pairs if int(obj["label"]) == 1)
    return TaskData(
        task=str(task),
        corpus=corpus,
        splits=splits,
        held_out_keys=list(held_keys),
        n_positives=int(n_pos),
        n_questions=int(len(qids)),
        n_unknown_questions=int(len(train_unknown) + len(test_unknown)),
        n_known_questions=int(len(train_known) + len(test_known)),
    )

def merge_task_data(
    items: Sequence[TaskData],
    text_rows: Sequence[TextRow],
    vectorizer: ZhBagVectorizer,
) -> TrainingData:
    """把多个任务的装配结果合并为**单一** ``TrainingData``（统一答案空间）。

    参数
    ----
    items : Sequence[TaskData]
        各任务装配结果（必须共享同一保留类表）。
    text_rows : Sequence[TextRow]
        文本行（仅用于满足 ``TrainingData`` 的字段契约；不参与 QA 训练）。
    vectorizer : ZhBagVectorizer
        与模型同源的向量化器。

    返回
    ----
    TrainingData
        合并后的训练数据视图（``records`` 的 ``source`` 即任务名，便于按任务过滤）。
    """
    if not items:
        raise ValueError("至少需要一个任务")
    base = list(items[0].corpus.answer_keys)
    for item in items:
        if list(item.corpus.answer_keys) != base:
            raise ValueError(
                f"任务 {item.task!r} 的候选空间与首个任务不一致；统一答案空间要求同一张表"
            )
    corpus = QACorpus(
        records=sorted((r for item in items for r in item.corpus.records),
                       key=lambda r: r.qid),
        answer_keys=list(base),
        answer_display=dict(items[0].corpus.answer_display),
        class_counts={},
    )
    splits = SplitSpec(
        train_known=[r for item in items for r in item.splits.train_known],
        train_unknown=[r for item in items for r in item.splits.train_unknown],
        test_known=[r for item in items for r in item.splits.test_known],
        test_unknown=[r for item in items for r in item.splits.test_unknown],
    )
    return TrainingData(
        corpus=corpus,
        splits=splits,
        text_lines=[
            TextRecord(line_id=r.row_id, text=r.text, source_file=r.file)
            for r in text_rows
        ],
        qa_files=[f"n3d_qa_{item.task}_pairs.jsonl" for item in items],
        vectorizer=vectorizer,  # type: ignore[arg-type]
    )


def filter_by_task(data: TrainingData, task: str) -> TrainingData:
    """按任务过滤出一个 ``TrainingData`` 视图（答案表与类别下标**保持不变**）。"""

    def _keep(records: Sequence[QARecord]) -> List[QARecord]:
        return [r for r in records if str(r.source) == str(task)]

    return TrainingData(
        corpus=QACorpus(
            records=_keep(data.corpus.records),
            answer_keys=list(data.corpus.answer_keys),
            answer_display=dict(data.corpus.answer_display),
            class_counts={},
        ),
        splits=SplitSpec(
            train_known=_keep(data.splits.train_known),
            train_unknown=_keep(data.splits.train_unknown),
            test_known=_keep(data.splits.test_known),
            test_unknown=_keep(data.splits.test_unknown),
        ),
        text_lines=list(data.text_lines),
        qa_files=list(data.qa_files),
        vectorizer=data.vectorizer,
    )


# ---------------------------------------------------------------------------
# 模型构造 / 训练（本模块自建训练循环）
# ---------------------------------------------------------------------------


def build_adapter(backend: str, dim: int = FEATURE_DIM) -> BackendAdapter:
    """经**连接契约代理层**构造后端适配器（D 是唯一连接参数）。"""
    if backend not in BACKEND_NAMES:
        raise KeyError(f"未登记的后端 {backend!r}；合法集合 = {list(BACKEND_NAMES)}")
    registry = BackendRegistry(int(dim))
    registry.register(str(backend))
    return registry.get(str(backend))


def build_model(
    backend: str,
    output_mode: str,
    n_answers: int,
    dim: int = FEATURE_DIM,
    label_smoothing: float = 0.0,
    head_input_mode: str = DEFAULT_HEAD_INPUT_MODE,
) -> N3DQA:
    """构造 ``N3DQA``（D 维 q 头 + 按全局开关的候选打分实现）。

    参数
    ----
    head_input_mode : str
        q 头的输入口径，**显式固定**（不吃上游默认值）：``"raw"`` 会让 N3D 骨干
        脱离计算图（全部骨干参数恒零梯度），步骤 2 需要真实经过 N3D，故默认 ``"concat"``。
        若上游 ``N3DQAConfig`` 没有该字段则忽略（向后兼容）。
    """
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"output_mode 仅允许 {list(OUTPUT_MODES)}，当前 {output_mode!r}")
    kwargs: Dict[str, Any] = {}
    fields = set(getattr(N3DQAConfig, "__dataclass_fields__", {}))
    if "head_input_mode" in fields:
        kwargs["head_input_mode"] = str(head_input_mode)
    return N3DQA(
        build_adapter(backend, dim),
        int(n_answers),
        N3DQAConfig(dim=int(dim), output_mode=str(output_mode),
                    label_smoothing=float(label_smoothing), **kwargs),
    )


def head_config_of(model: N3DQA) -> Dict[str, Any]:
    """取出 q 头相关的**构造口径**（写进产物 meta，供加载侧原样重建模型）。"""
    cfg = model.config
    out: Dict[str, Any] = {"output_mode": str(cfg.output_mode), "dim": int(cfg.dim)}
    for name in ("head_input_mode", "normalize_query", "answer_table_mode",
                 "label_smoothing", "logit_scale_init", "learn_logit_scale",
                 "mix_logit_init"):
        if hasattr(cfg, name):
            out[str(name)] = getattr(cfg, name)
    return out


def module_file_manifest(module_dir: str = "", extra_dirs: Sequence[str] = ()) -> Dict[str, Any]:
    """本模块与所依赖上游源码文件的 SHA256 指纹（**源码溯源**）。

    为什么必须有：上游 ``heads.py`` 在本轮执行期间被改写多次，仅靠产物 SHA256 无法
    回答「这份数字是哪一版上游跑出来的」。把源码指纹写进产物与报告，才能事后对账。

    参数
    ----
    module_dir : str
        模块源码目录；空串取本文件所在目录。
    extra_dirs : Sequence[str]
        额外要计入指纹的目录（如上游 ``n3d_qa``）。

    返回
    ----
    Dict[str, Any]
        ``{"files": [{"path","bytes","sha256"}], "n_files": N}``。
    """
    here = module_dir or os.path.dirname(os.path.abspath(__file__))
    dirs = [here] + [d for d in extra_dirs if d]
    files: List[Dict[str, Any]] = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(d, name)
            if not os.path.isfile(path):
                continue
            files.append(
                {
                    "path": os.path.relpath(path).replace("\\", "/"),
                    "bytes": int(os.path.getsize(path)),
                    "sha256": sha256_file(path),
                }
            )
    return {"files": files, "n_files": int(len(files))}


def pointer_keys_from_answers(
    answer_keys: Sequence[str],
    answer_display: Dict[str, str],
    vectorizer: ZhBagVectorizer,
    batch: int,
) -> torch.Tensor:
    """构造指针模式的候选键表 ``[B, C+1, D]``（末位 = 不相关，键文本与
    :data:`n3d_qa_learn.train.IRRELEVANT_KEY_TEXT` 同口径）。"""
    texts = [str(answer_display.get(k, k)) for k in answer_keys] + [IRRELEVANT_KEY_TEXT]
    mat = vectorizer.encode_matrix(texts)
    keys = torch.tensor(mat, dtype=torch.float32)
    return keys.unsqueeze(0).expand(int(batch), -1, -1).contiguous()


@dataclass
class TrainOutcome:
    """一次训练的产物（模型 + 历史 + 取证信息）。"""

    model: N3DQA
    history: List[Dict[str, float]]
    seconds: float
    n_train_items: int
    n_epochs: int
    backbone_trained: bool


def train_model(
    data: TrainingData,
    *,
    backend: str = DEFAULT_BACKEND,
    output_mode: str = "index",
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 5e-3,
    backbone_lr: float = 5e-4,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    train_backbone: bool = True,
    max_train_samples: int = 0,
    max_batches: int = 0,
    head_input_mode: str = DEFAULT_HEAD_INPUT_MODE,
    device: str = "cpu",
) -> TrainOutcome:
    """自建训练循环（交叉熵 + 显式「不相关」类）。

    训练样本 = ``train_known``（金标 = 类别下标） 并 ``train_unknown``（金标 = 末位 C）。
    优化器分组：q 头 / 候选键表用 ``lr``，后端 N3D 权重用 ``backbone_lr``。

    参数
    ----
    data : TrainingData
        装配结果（``corpus.answer_keys`` 即候选空间）。
    backend / output_mode / seed / epochs / batch_size / lr : 训练配置。
    backbone_lr : float
        后端 N3D 权重的学习率（与 ``lr`` 分开）。
    weight_decay / label_smoothing / train_backbone : 训练开关。
    max_train_samples : int
        训练样本上限（``0`` = 全量；限批演练用）。
    max_batches : int
        每 epoch 的批数上限（``0`` = 全量；单条端到端演练用）。
    device : str
        计算设备。

    返回
    ----
    TrainOutcome
        训练结果。
    """
    set_deterministic_seed(int(seed))
    dev = torch.device(device)
    vectorizer: ZhBagVectorizer = data.vectorizer  # type: ignore[assignment]
    model = build_model(backend, output_mode, data.corpus.n_classes,
                        dim=int(vectorizer.dim), label_smoothing=float(label_smoothing),
                        head_input_mode=str(head_input_mode))
    model = model.to(dev)

    label_index = data.corpus.key_to_index()
    items: List[Tuple[QARecord, int]] = [
        (rec, int(label_index[rec.answer_key])) for rec in data.splits.train_known
    ]
    items.extend((rec, int(data.corpus.n_classes)) for rec in data.splits.train_unknown)
    if int(max_train_samples) > 0:
        items.sort(key=lambda it: (it[1] != data.corpus.n_classes, it[0].qid))
        items = items[: int(max_train_samples)]
    if not items:
        raise RuntimeError("训练样本为空（train_known 与 train_unknown 都为空）")

    # ---- 上游 head 口径适配（answer_table_mode）----
    # `free`：候选键表是可学习参数（无需初始化）；
    # `centroid`：候选键表是**缓冲区**，必须由训练侧逐类质心确定性写入 ——
    # 否则键表恒为全零、logits 恒为 0、softmax 均匀、**所有参数梯度恒为 0**
    # （现场实测：drill 的梯度取证会报出全部 8 个参数零梯度）。
    if (output_mode == "index"
            and getattr(model.config, "answer_table_mode", "free") != "free"
            and hasattr(
            model, "set_answer_table_from_centroids"
        )
    ):
        feats_init = torch.tensor(
            vectorizer.encode_batch([r.question for r in data.splits.train_known]),
            dtype=torch.float32,
        )
        tgt_init = torch.tensor(
            [int(label_index[r.answer_key]) for r in data.splits.train_known],
            dtype=torch.long,
        )
        if int(feats_init.numel()) > 0:
            model.set_answer_table_from_centroids(
                feats_init, tgt_init, int(data.corpus.n_classes)
            )

    model.train()
    model.adapter.model.train(mode=bool(train_backbone))
    for p in model.adapter.model.parameters():
        p.requires_grad_(bool(train_backbone))

    head_params = list(model.q_head.parameters())
    if hasattr(model, "answer_table"):
        head_params.append(model.answer_table)
    # 两模式共用的可学习 logits 尺度（字段名随上游 heads.py 演进：
    # 新版为 logit_scale，历史名为 pointer_inv_temp；此处按存在性收集，避免硬绑字段名）
    for _scale_name in ("logit_scale", "pointer_inv_temp"):
        if hasattr(model, _scale_name):
            head_params.append(getattr(model, _scale_name))
    groups: List[Dict[str, Any]] = [{"params": head_params, "lr": float(lr)}]
    backbone_params = [p for p in model.adapter.model.parameters() if p.requires_grad]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(backbone_lr)})
    opt_cls = torch.optim.AdamW if float(weight_decay) > 0.0 else torch.optim.Adam
    optimizer: torch.optim.Optimizer = opt_cls(
        groups, lr=float(lr), weight_decay=float(weight_decay)
    )

    gen = torch.Generator().manual_seed(int(seed))
    history: List[Dict[str, float]] = []
    t0 = time.time()
    for epoch in range(1, int(epochs) + 1):
        perm = torch.randperm(len(items), generator=gen).tolist()
        n_batches = 0
        loss_sum = 0.0
        correct = 0
        seen = 0
        for b0 in range(0, len(perm), int(batch_size)):
            if int(max_batches) > 0 and n_batches >= int(max_batches):
                break
            chunk = [items[i] for i in perm[b0 : b0 + int(batch_size)]]
            feats = torch.tensor(
                vectorizer.encode_batch([rec.question for rec, _ in chunk]),
                dtype=torch.float32,
            ).to(dev)
            targets = torch.tensor([t for _, t in chunk], dtype=torch.long, device=dev)
            keys = None
            if output_mode == "pointer":
                keys = pointer_keys_from_answers(
                    data.corpus.answer_keys, data.corpus.answer_display,
                    vectorizer, len(chunk),
                ).to(dev)
            logits = model.logits(feats, keys)
            loss = model.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item())
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += len(chunk)
            n_batches += 1
        history.append(
            {
                "epoch": int(epoch),
                "loss": float(loss_sum / max(1, n_batches)),
                "train_acc": float(correct / max(1, seen)),
                "batches": int(n_batches),
            }
        )
    seconds = time.time() - t0
    model.eval()
    return TrainOutcome(
        model=model,
        history=history,
        seconds=float(seconds),
        n_train_items=int(len(items)),
        n_epochs=int(epochs),
        backbone_trained=bool(train_backbone),
    )

# ---------------------------------------------------------------------------
# 步骤 2：候选键表 + 冻结产物
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextRowKeyTable:
    """索引式候选键表（**冻结**，不参与梯度）。"""

    keys: torch.Tensor          # [L, D] float32
    line_ids: List[str]
    texts: List[str]

    @property
    def size(self) -> int:
        """库行数 L。"""
        return int(self.keys.shape[0])

    @property
    def dim(self) -> int:
        """连接参数 D。"""
        return int(self.keys.shape[1])

    def sha256(self) -> str:
        """键表字节指纹（float32 连续内存的 SHA256）。"""
        t = self.keys.detach().to(torch.float32).cpu().contiguous()
        return sha256_bytes(t.numpy().tobytes())

    def line_ids_sha256(self) -> str:
        """行 id 列表指纹（顺序敏感）。"""
        return sha256_bytes(canonical_dumps(list(self.line_ids)))


def build_key_table(
    vectorizer: ZhBagVectorizer, rows: Sequence[TextRow], index: Sequence[int]
) -> TextRowKeyTable:
    """按库行下标构造冻结键表（由**文本行确定性特征**构造，不参与梯度）。"""
    idx = [int(i) for i in index]
    mat = vectorizer.encode_matrix([rows[i].text for i in idx])
    return TextRowKeyTable(
        keys=torch.tensor(mat, dtype=torch.float32),
        line_ids=[rows[i].row_id for i in idx],
        texts=[rows[i].text for i in idx],
    )


def save_key_table(path: str, table: TextRowKeyTable, meta: Dict[str, Any]) -> str:
    """把冻结键表落成自写 zip（固定时间戳；返回文件 SHA256）。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("meta.json", date_time=ZIP_EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=1))
        tbl = io.BytesIO()
        torch.save(
            {
                "keys": table.keys.detach().to(torch.float32).cpu().contiguous(),
                "line_ids": list(table.line_ids),
            },
            tbl,
        )
        info = zipfile.ZipInfo("key_table.pt", date_time=ZIP_EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, tbl.getvalue())
    blob = buffer.getvalue()
    with open(path, "wb") as handle:
        handle.write(blob)
    return sha256_bytes(blob)


def load_key_table(path: str, *, verify: bool = True) -> Tuple[TextRowKeyTable, Dict[str, Any]]:
    """加载冻结键表并执行**两道守卫**。

    守卫
    ----
    1. 键表字节 SHA256 必须与 meta 记录一致；
    2. 特征口径 ``spec_hash`` 必须与现行口径一致，且键表维数须等于该口径的 bag_dim。

    异常
    ------
    FileNotFoundError
        产物不存在。
    ValueError
        任一道守卫失败。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"步骤 2 产物不存在：{path!r}")
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        missing = {"meta.json", "key_table.pt"} - names
        if missing:
            raise ValueError(f"产物缺少必需成员 {sorted(missing)}；实际成员 = {sorted(names)}")
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        blob = torch.load(io.BytesIO(zf.read("key_table.pt")),
                          map_location="cpu", weights_only=False)
    table = TextRowKeyTable(
        keys=blob["keys"].to(torch.float32).contiguous(),
        line_ids=[str(x) for x in blob["line_ids"]],
        texts=[],
    )
    if not verify:
        return table, meta
    expected = str(meta.get("key_table_sha256", ""))
    actual = table.sha256()
    if actual != expected:
        raise ValueError(
            "键表指纹校验失败：产物 meta 记录 "
            f"{expected[:16]}...，现场重算 {actual[:16]}...；候选键表已被改动，拒绝加载"
        )
    current = zh_feature_config().spec_hash()
    if str(meta.get("feature_spec_hash", "")) != current:
        raise ValueError(
            "特征口径指纹不一致：产物记录 "
            f"{str(meta.get('feature_spec_hash'))[:16]}...，现行实现 {current[:16]}...；"
            "拒绝在错配的特征空间上做检索"
        )
    if int(table.dim) != int(zh_feature_config().bag_dim):
        raise ValueError(
            f"键表维数 {table.dim} 与现行口径 D={zh_feature_config().bag_dim} 不一致"
        )
    if len(table.line_ids) != table.size:
        raise ValueError(f"行 id 数 {len(table.line_ids)} 与键表行数 {table.size} 不一致")
    return table, meta


def step2_artifact_name(backend: str, output_mode: str, dim: int, rows: int, seed: int) -> str:
    """步骤 2 产物文件名（含后端 / 模式 / 维度 / 库规模 / 种子指纹，防撞名）。"""
    return f"qa_step2_textrows_{backend}_{output_mode}_D{dim}_L{rows}_s{seed}.pt.zip"


# ---------------------------------------------------------------------------
# 匹配（两模式共用同一 q）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    """一次文本行匹配的结果。"""

    line_id: str
    text: str
    score: float
    topk_line_ids: List[str]
    topk_scores: List[float]
    mode: str


class TextRowMatcher:
    """步骤 2 匹配器：q 来自 N3D 模型，候选键按全局开关取两种来源。

    参数
    ----
    model : N3DQA
        训练好的模型（两模式共用同一 q）。
    vectorizer : ZhBagVectorizer
        与模型同源的特征口径。
    mode : str
        ``index``（候选键来自构造期冻结的键表）或 ``pointer``（候选键由输入提供）。
    key_table : Optional[TextRowKeyTable]
        ``index`` 模式**必须**给出；``pointer`` 模式必须为 ``None``。

    关键不变量（构造期）
    ------------------
    * ``index`` 模式的键表是普通张量（``requires_grad`` 恒为 ``False``，冻结不参与梯度）；
    * ``pointer`` 模式**不持有**任何候选键（``self.key_table is None``）。
    """

    def __init__(
        self,
        model: N3DQA,
        vectorizer: ZhBagVectorizer,
        mode: str,
        key_table: Optional[TextRowKeyTable] = None,
    ) -> None:
        if mode not in OUTPUT_MODES:
            raise ValueError(f"mode 仅允许 {list(OUTPUT_MODES)}，当前 {mode!r}")
        if int(vectorizer.dim) != int(model.config.dim):
            raise ValueError(
                f"向量化口径 D={vectorizer.dim} 与模型 q 的维度 D={model.config.dim} 不一致"
            )
        if mode == "index" and key_table is None:
            raise ValueError("index 模式必须提供冻结键表（产物内固化的候选键）")
        if mode == "pointer" and key_table is not None:
            raise ValueError("pointer 模式的候选键由输入提供，不接受构造期键表")
        self.model = model
        self.vectorizer = vectorizer
        self.mode = str(mode)
        self.key_table = key_table

    def features_of(self, text: str) -> torch.Tensor:
        """单条文本的 D 维确定性特征 ``[1, D]``。"""
        return torch.tensor([self.vectorizer.encode(text)], dtype=torch.float32)

    @torch.no_grad()
    def q_of(self, texts: Sequence[str]) -> torch.Tensor:
        """``[B, D]`` 查询向量（**同一条 q 通路**：后端特征 -> q 头）。"""
        feats = torch.tensor(self.vectorizer.encode_batch(list(texts)), dtype=torch.float32)
        return self.model.query(feats)

    @torch.no_grad()
    def logits(self, texts: Sequence[str],
               keys: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[B, L]`` logits（按模式取候选来源）。"""
        q = self.q_of(texts)
        if self.mode == "index":
            if keys is not None:
                raise ValueError("index 模式的候选键来自产物，不接受输入 keys")
            assert self.key_table is not None
            k = self.key_table.keys
            return q @ k.transpose(0, 1)
        if keys is None:
            raise ValueError("pointer 模式必须提供 keys=[B, L, D]")
        return torch.bmm(q.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)

    @torch.no_grad()
    def match(self, texts: Sequence[str],
              keys: Optional[torch.Tensor] = None,
              key_table: Optional[TextRowKeyTable] = None,
              k: int = TOPK) -> List[MatchResult]:
        """返回逐条 top-k 行（softmax 在候选维上做，取 top-1 行原文）。"""
        table = key_table if key_table is not None else self.key_table
        if table is None:
            raise ValueError("pointer 模式必须给出 key_table 以便回填行原文")
        logits = self.logits(list(texts), keys)
        probs = F.softmax(logits, dim=1)
        kk = max(1, min(int(k), int(logits.shape[1])))
        top = torch.topk(probs, k=kk, dim=1)
        out: List[MatchResult] = []
        for i in range(int(logits.shape[0])):
            idx = [int(x) for x in top.indices[i].tolist()]
            scores = [float(x) for x in top.values[i].tolist()]
            out.append(
                MatchResult(
                    line_id=table.line_ids[idx[0]],
                    text=table.texts[idx[0]] if table.texts else "",
                    score=scores[0],
                    topk_line_ids=[table.line_ids[j] for j in idx],
                    topk_scores=scores,
                    mode=self.mode,
                )
            )
        return out

# ---------------------------------------------------------------------------
# 自检索留出法 / 参照下限 / 双模式一致性
# ---------------------------------------------------------------------------


def _rank_of(gold_pos: int, rank_list: Sequence[int]) -> int:
    """金标在 top-k 位次表中的名次（1-based）；不在其中返回 -1。"""
    return (list(rank_list).index(int(gold_pos)) + 1) if int(gold_pos) in list(rank_list) else -1


def self_retrieval(
    matcher: TextRowMatcher,
    rows: Sequence[TextRow],
    query_index: Sequence[int],
    library_index: Sequence[int],
    key_table: TextRowKeyTable,
    *,
    topk: int = TOPK,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """自检索留出法：查询行经**同一条 q 通路**在库中检索，命中自身行即正确。

    参数
    ----
    matcher : TextRowMatcher
        匹配器（index 模式用产物键表；pointer 模式由本函数喂入同一键表）。
    rows : Sequence[TextRow]
        全部文本行。
    query_index : Sequence[int]
        查询行在 ``rows`` 中的下标。
    library_index : Sequence[int]
        库行在 ``rows`` 中的下标（与 ``key_table.line_ids`` **同序**）。
    key_table : TextRowKeyTable
        库行键表。
    topk : int
        ``Recall@k`` 的最大 k。
    batch_size : int
        批大小。

    返回
    ----
    Dict[str, Any]
        ``recall_at_1`` / ``recall_at_5`` / ``n`` / ``n_library`` / ``rank_hist`` 等。
    """
    q_idx = [int(i) for i in query_index]
    if len(q_idx) != len(set(q_idx)):
        raise ValueError("查询行下标存在重复；自检索要求查询行互异")
    pos_of = {lid: i for i, lid in enumerate(key_table.line_ids)}
    lib_ids = [rows[i].row_id for i in library_index]
    if lib_ids != list(key_table.line_ids):
        raise ValueError("库行顺序与键表行 id 顺序不一致；拒绝在错位候选上算命中")

    kk = max(1, int(topk))
    hits1 = 0
    hits5 = 0
    ranks: List[int] = []
    misses: List[Dict[str, Any]] = []
    for b0 in range(0, len(q_idx), int(batch_size)):
        chunk = q_idx[b0 : b0 + int(batch_size)]
        texts = [rows[i].text for i in chunk]
        keys = None
        if matcher.mode == "pointer":
            keys = key_table.keys.unsqueeze(0).expand(len(chunk), -1, -1).contiguous()
        logits = matcher.logits(texts, keys)
        order = torch.argsort(logits, dim=1, descending=True)[:, :kk]
        for i, row_i in enumerate(chunk):
            gold = str(rows[row_i].row_id)
            gold_pos = pos_of.get(gold)
            if gold_pos is None:
                raise ValueError(
                    f"查询行 {gold!r} 不在库中；自检索要求查询行与库行来自同一全量行表"
                )
            rank_list = [int(x) for x in order[i].tolist()]
            rank = _rank_of(gold_pos, rank_list)
            ranks.append(int(rank))
            hits1 += int(rank == 1)
            hits5 += int(1 <= rank <= kk)
            if rank != 1 and len(misses) < 20:
                misses.append(
                    {
                        "row_id": gold,
                        "file": rows[row_i].file,
                        "line_no": int(rows[row_i].line_no),
                        "rank": int(rank),
                        "top1_line_id": key_table.line_ids[rank_list[0]],
                    }
                )
    n = len(q_idx)
    return {
        "n": int(n),
        "n_library": int(key_table.size),
        "topk": int(kk),
        "recall_at_1": float(hits1 / max(1, n)),
        "recall_at_5": float(hits5 / max(1, n)),
        "rank_1": int(sum(1 for r in ranks if r == 1)),
        "rank_miss_topk": int(sum(1 for r in ranks if r < 0)),
        "rank_hist": {str(i): int(sum(1 for r in ranks if r == i))
                      for i in range(1, kk + 1)},
        "misses_head": misses,
    }


def deterministic_baseline(
    vectorizer: ZhBagVectorizer,
    rows: Sequence[TextRow],
    query_index: Sequence[int],
    library_index: Sequence[int],
    key_table: TextRowKeyTable,
    *,
    topk: int = TOPK,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """参照下限：**不经 N3D** 的纯确定性特征检索（余弦 top-k）。

    口径与步骤 2 的自检索**完全同源**（同一库键表、同一查询集），唯一差别是不经过
    后端与 q 头 —— 因此两者的差即为 N3D q 在步骤 2 上的增益。
    """
    q_idx = [int(i) for i in query_index]
    lib_mat = key_table.keys.numpy()
    pos_of = {lid: i for i, lid in enumerate(key_table.line_ids)}
    kk = max(1, int(topk))
    hits1 = 0
    hits5 = 0
    for b0 in range(0, len(q_idx), int(batch_size)):
        chunk = q_idx[b0 : b0 + int(batch_size)]
        qmat = np.asarray(
            vectorizer.encode_matrix([rows[i].text for i in chunk]), dtype=np.float32
        )
        sims = qmat @ lib_mat.T
        order = np.argsort(-sims, axis=1)[:, :kk]
        for i, row_i in enumerate(chunk):
            gold_pos = pos_of.get(str(rows[row_i].row_id))
            if gold_pos is None:
                raise ValueError(f"查询行 {rows[row_i].row_id!r} 不在库中")
            rank = _rank_of(gold_pos, [int(x) for x in order[i].tolist()])
            hits1 += int(rank == 1)
            hits5 += int(1 <= rank <= kk)
    n = len(q_idx)
    return {
        "n": int(n),
        "n_library": int(key_table.size),
        "recall_at_1": float(hits1 / max(1, n)),
        "recall_at_5": float(hits5 / max(1, n)),
        "note": "纯确定性特征检索（不经 N3D）；同库同查询集，只作参照下限，不作门槛",
    }


def dual_mode_consistency(
    matcher_index: TextRowMatcher,
    matcher_pointer: TextRowMatcher,
    rows: Sequence[TextRow],
    query_index: Sequence[int],
    key_table: TextRowKeyTable,
) -> Dict[str, Any]:
    """双模式一致性：同一批输入上两模式各跑一次，报告 top-1 一致率。

    同时给出**结构性取证**：两模式 logits 是否逐位相同（``torch.equal``）与最大绝对偏差。
    """
    texts = [rows[int(i)].text for i in query_index]
    lg_idx = matcher_index.logits(texts)
    keys = key_table.keys.unsqueeze(0).expand(len(texts), -1, -1).contiguous()
    lg_ptr = matcher_pointer.logits(texts, keys)
    same_bits = bool(torch.equal(lg_idx, lg_ptr))
    max_dev = float((lg_idx - lg_ptr).abs().max().item())
    top_i = torch.argmax(lg_idx, dim=1)
    top_p = torch.argmax(lg_ptr, dim=1)
    agree = int((top_i == top_p).sum().item())
    n = int(len(texts))
    return {
        "n": int(n),
        "top1_agreement": float(agree / max(1, n)),
        "n_agree": int(agree),
        "logits_bitwise_equal": bool(same_bits),
        "logits_max_abs_diff": float(max_dev),
        "index_key_table_requires_grad": bool(key_table.keys.requires_grad),
        "note": (
            "两模式在同一候选集上做的是同一次 q @ K^T；一致率是实测值，"
            "结构性取证见 logits_bitwise_equal / logits_max_abs_diff"
        ),
    }

# ---------------------------------------------------------------------------
# 分项 5 项指标（按任务）
# ---------------------------------------------------------------------------


def routing_metrics(
    router: QuestionRouter,
    model: N3DQA,
    data: TrainingData,
    vectorizer: ZhBagVectorizer,
    *,
    device: str = "cpu",
    batch_size: int = 128,
) -> Dict[str, Any]:
    """③ 路由正确率（命中 / 未命中判定）与 ⑤ 端到端最终答案正确率。

    口径（**显式**）
    ----------------
    两个**不同**的量，必须分开报（历史缺陷：docstring 写严格口径、实现只算宽松口径，
    导致 ⑤ 与 ③ 恒等且无人察觉）：

    * **③ 路由正确率（``decision_accuracy``，宽松）**：命中 = 路由结果 ``source == "qa"``；
      真值：金标答案在模型候选空间 -> 应命中，金标在留出类 -> 应未命中。
      只看「有没有走到步骤 1 的答案分支」，**不校验答案是否正确**。
    * **⑤ 端到端最终答案正确率（``e2e_accuracy``，严格）**：已知题要求
      「``source == "qa"`` **且** 返回的答案文本等于金标展示文本」；
      不相关题要求「最终 ``source == "none"``（两级都未命中，返回无匹配）」。
      故 ⑤ ≤ ③ 恒成立；两者相等当且仅当所有被命中的已知题答案都对。
    * **步骤 2 到达诊断（``step2_reached_*``）**：统计有多少测试样本真的落到了
      文本行匹配分支（``source == "text"``）。若该计数为 0，说明本轮**步骤 2 的文本匹配
      在任何被测指标里都没被走到**（路由退化的直接证据），必须如实登记而不是当作通过。

    参数
    ----
    router : QuestionRouter
        两级路由器（步骤 1 走模型 logits，步骤 2 走文本行余弦）。
    model : N3DQA
        步骤 1 模型。
    data : TrainingData
        **单任务视图**（``test_known`` / ``test_unknown`` 均已按任务过滤）。
    vectorizer : ZhBagVectorizer
        向量化器。
    device / batch_size : 运行参数。

    返回
    ----
    Dict[str, Any]
        命中率 / 未命中率 / 判定正确率 / 端到端正确率与分项计数。
    """
    dev = torch.device(device)
    known = list(data.splits.test_known)
    unknown = list(data.splits.test_unknown)

    def _logits(records: Sequence[QARecord]) -> List[List[float]]:
        out: List[List[float]] = []
        for b0 in range(0, len(records), int(batch_size)):
            chunk = records[b0 : b0 + int(batch_size)]
            feats = torch.tensor(
                vectorizer.encode_batch([r.question for r in chunk]), dtype=torch.float32
            ).to(dev)
            keys = None
            if model.config.output_mode == "pointer":
                keys = pointer_keys_from_answers(
                    data.corpus.answer_keys, data.corpus.answer_display,
                    vectorizer, len(chunk),
                ).to(dev)
            with torch.no_grad():
                lg = model.logits(feats, keys)
            out.extend([list(x) for x in lg.detach().cpu().tolist()])
        return out

    hit_known = 0
    e2e_known = 0
    step2_reached_known = 0
    for rec, lg in zip(known, _logits(known)):
        res = router.route(rec.question, lg)
        hit = res.source == "qa"
        hit_known += int(hit)
        # 严格 ⑤：命中还不够，返回的答案文本必须等于金标展示文本
        e2e_known += int(hit and str(res.answer) == str(rec.answer_display))
        step2_reached_known += int(res.source == "text")
    rejected_unknown = 0
    step2_reached_unknown = 0
    for rec, lg in zip(unknown, _logits(unknown)):
        res = router.route(rec.question, lg)
        rejected_unknown += int(res.source == "none")
        step2_reached_unknown += int(res.source == "text")
    correct_decision = hit_known + rejected_unknown
    n_known = len(known)
    n_unknown = len(unknown)
    n_total = n_known + n_unknown
    return {
        "n_known": int(n_known),
        "n_unknown": int(n_unknown),
        "hit_rate_on_known": float(hit_known / max(1, n_known)),
        "reject_rate_on_unknown": float(rejected_unknown / max(1, n_unknown)),
        "decision_accuracy": float(correct_decision / max(1, n_total)),
        # --- 严格 ⑤（端到端最终答案正确率）：已知题须「命中且答案文本正确」 ---
        "e2e_accuracy": float((e2e_known + rejected_unknown) / max(1, n_total)),
        "e2e_on_known": float(e2e_known / max(1, n_known)),
        "e2e_on_unknown": float(rejected_unknown / max(1, n_unknown)),
        # --- 宽松口径（仅判定是否走到步骤 1 答案分支），保留以便对照 ⑤ ≤ ③ ---
        "decision_accuracy_loose": float(correct_decision / max(1, n_total)),
        "e2e_loose_on_known": float(hit_known / max(1, n_known)),
        # --- 步骤 2 到达诊断（0 即「文本匹配根本没被走到」） ---
        "step2_reached_known": int(step2_reached_known),
        "step2_reached_unknown": int(step2_reached_unknown),
        "step2_reached_total": int(step2_reached_known + step2_reached_unknown),
        "no_match_text": str(NO_MATCH_TEXT),
    }


def task_metrics(
    task: str,
    model: N3DQA,
    task_view: TrainingData,
    router: QuestionRouter,
    vectorizer: ZhBagVectorizer,
    step2_global: Dict[str, Any],
    *,
    device: str = "cpu",
    batch_size: int = 128,
) -> Dict[str, Any]:
    """按任务产出分项 5 项指标（①-⑤）。

    ② 步骤 2 的 ``Recall@1`` 由**文本行自检索**给出，与任务无关（文本行库不随任务变化），
    故此处填入全局值并在字段 ``step2_recall_at_1_is_task_independent`` 中显式标记。
    """
    s1 = evaluate_step1(model, task_view, k=3, batch_size=batch_size, device=device)
    ref = evaluate_refusal(model, task_view, batch_size=batch_size, device=device)
    rt = routing_metrics(router, model, task_view, vectorizer,
                         device=device, batch_size=batch_size)
    return {
        "task": str(task),
        "step1_answer_accuracy": float(s1.top1_acc),
        "step1_top3_accuracy": float(s1.topk_acc),
        "step1_majority_baseline": float(s1.majority),
        "step1_random_baseline": float(s1.random_baseline),
        "step1_macro_accuracy": float(s1.macro_acc),
        "step2_recall_at_1": float(step2_global["recall_at_1"]),
        "step2_recall_at_1_is_task_independent": True,
        "routing_decision_accuracy": float(rt["decision_accuracy"]),
        "routing_hit_rate_on_known": float(rt["hit_rate_on_known"]),
        "routing_reject_rate_on_unknown": float(rt["reject_rate_on_unknown"]),
        "refusal_precision": float(ref["precision"]),
        "refusal_recall": float(ref["recall"]),
        "refusal_f1": float(ref["f1"]),
        "e2e_answer_accuracy": float(rt["e2e_accuracy"]),
        "e2e_accuracy_loose": float(rt["decision_accuracy"]),
        "step2_reached_total": int(rt["step2_reached_total"]),
        "counts": {"n_test_known": int(rt["n_known"]), "n_test_unknown": int(rt["n_unknown"])},
    }


# ---------------------------------------------------------------------------
# 跨 seed 报告
# ---------------------------------------------------------------------------


def _aggregate(values: Sequence[float]) -> Dict[str, float]:
    """均值 / 极差 / 最小 / 最大（极差 = max - min）。"""
    arr = [float(v) for v in values]
    if not arr:
        return {"mean": float("nan"), "range": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(sum(arr) / len(arr)),
        "range": float(max(arr) - min(arr)),
        "min": float(min(arr)),
        "max": float(max(arr)),
        "n": int(len(arr)),
    }


#: 参与跨 seed 聚合的分项字段。
AGGREGATED_FIELDS: Tuple[str, ...] = (
    "step1_answer_accuracy",
    "step2_recall_at_1",
    "routing_decision_accuracy",
    "refusal_precision",
    "refusal_recall",
    "refusal_f1",
    "e2e_answer_accuracy",
    "step2_reached_total",
)


def cross_seed_summary(per_seed: Dict[int, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """按任务 x 分项聚合跨 seed 的 ``均值 +- 极差``。

    参数
    ----
    per_seed : Dict[int, Dict[str, Dict[str, Any]]]
        ``{seed: {task: metrics}}``。

    返回
    ----
    Dict[str, Any]
        ``{"seeds": [...], "tasks": {task: {field: {mean, range, min, max, n}}}}``。
    """
    seeds = sorted(int(s) for s in per_seed)
    tasks: List[str] = []
    for seed in seeds:
        for task in per_seed[seed]:
            if task not in tasks:
                tasks.append(task)
    out: Dict[str, Any] = {"seeds": [int(s) for s in seeds], "tasks": {}}
    for task in tasks:
        blob: Dict[str, Any] = {}
        for fld in AGGREGATED_FIELDS:
            vals = [
                float(per_seed[seed][task][fld])
                for seed in seeds
                if task in per_seed[seed] and fld in per_seed[seed][task]
            ]
            blob[fld] = _aggregate(vals)
        out["tasks"][task] = blob
    return out


def format_cross_seed_markdown(summary: Dict[str, Any]) -> str:
    """把跨 seed 摘要渲染成 Markdown 表（均值 +- 极差）。"""
    lines: List[str] = []
    lines.append(f"跨 seed 集合：{summary.get('seeds', [])}（训练侧 seed；拓扑 seed 见「前提」节）")
    lines.append("")
    lines.append("| 任务 | 分项 | 均值 | 极差 | 最小 | 最大 | n |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for task, blob in summary.get("tasks", {}).items():
        for fld in AGGREGATED_FIELDS:
            if fld not in blob:
                continue
            agg = blob[fld]
            lines.append(
                f"| {task} | {fld} | {agg['mean']:.4f} | {agg['range']:.4f} | "
                f"{agg['min']:.4f} | {agg['max']:.4f} | {agg['n']} |"
            )
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# 报告落盘
# ---------------------------------------------------------------------------


def write_report(out_dir: str, report: Dict[str, Any]) -> Dict[str, str]:
    """把报告落盘为 ``step2_report.json`` 与 ``step2_report.md``（返回路径与 SHA256）。

    确定性：JSON 用规范化序列化计算指纹；``created_utc`` 与 ``report_paths`` 属**溯源字段**，
    均**不参与**指标指纹 —— ``report_paths`` 里含有指纹自身，若参与即形成自指循环
    （历史缺陷：指纹只排除 ``created_utc``，导致回填 ``report_paths`` 后指纹变化，
    且 replay 读到的报告恒无该键、``report_metrics_sha256`` 恒为空串）。
    """
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "step2_report.json")
    md_path = os.path.join(out_dir, "step2_report.md")
    blob = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1)
    with open(json_path, "w", encoding="utf-8") as handle:
        handle.write(blob)
    md = render_report_markdown(report)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(md)
    metrics_blob = canonical_dumps(
        {k: v for k, v in report.items()
         if k not in ("created_utc", "report_paths")}
    )
    return {
        "json": json_path,
        "markdown": md_path,
        "json_sha256": sha256_bytes(blob.encode("utf-8")),
        "markdown_sha256": sha256_bytes(md.encode("utf-8")),
        "metrics_sha256": sha256_bytes(metrics_blob),
    }


def render_report_markdown(report: Dict[str, Any]) -> str:
    """把报告渲染为 Markdown（前提 / 分项表 / 跨 seed 表 / 如实登记项）。"""
    L: List[str] = []
    L.append("# n3d_qa_learn 步骤 2 分项验收报告")
    L.append("")
    L.append(f"- 生成时间（UTC，溯源字段）：{report.get('created_utc')}")
    L.append(f"- n3d_qa 冻结产物目录（只读）：`{report.get('product_dir')}`")
    L.append(f"- 连接参数 D：{report.get('dim')}；后端：`{report.get('backend')}`")
    L.append(f"- 输出模式：{report.get('modes')}")
    L.append("")
    L.append("## 前提（拓扑 seed）")
    L.append("")
    L.append(str(report.get("topology_premise", "")))
    L.append("")
    L.append("## 开集协议（不相关的构造口径）")
    L.append("")
    held = report.get("held_out", {})
    L.append(f"- 统一答案表类数：{held.get('n_total_classes')}")
    L.append(f"- 留出类数：{held.get('n_held_out')}"
             f"（比例 {held.get('ratio')}，seed {held.get('seed')}）")
    L.append(f"- 留出类清单 SHA256：`{held.get('held_out_keys_sha256')}`")
    L.append(f"- 保留类数（= 模型候选空间 C）：{held.get('n_kept')}")
    L.append("")
    L.append("## 文本行自检索留出法（步骤 2 主判据）")
    L.append("")
    for mode, blob in (report.get("self_retrieval") or {}).items():
        L.append(f"### {mode}")
        L.append("")
        L.append(f"- 库行数 L={blob.get('n_library')}，查询行数 N={blob.get('n')}")
        L.append(f"- **Recall@1 = {blob.get('recall_at_1'):.6f}**")
        L.append(f"- **Recall@5 = {blob.get('recall_at_5'):.6f}**")
        L.append(f"- rank 直方图：{blob.get('rank_hist')}")
        L.append("")
    base = report.get("baseline") or {}
    L.append("### 参照下限（纯确定性特征检索，不经 N3D；不作门槛）")
    L.append("")
    L.append(f"- Recall@1 = {base.get('recall_at_1'):.6f}")
    L.append(f"- Recall@5 = {base.get('recall_at_5'):.6f}")
    L.append(f"- N3D q 相对参照下限的 Recall@1 增益 = {report.get('step2_gain_over_baseline')}")
    L.append("")
    L.append("### 双模式一致性（只报实测值）")
    L.append("")
    dm = report.get("dual_mode") or {}
    # 扁平汇总键（由 cmd_eval 写入；此处按存在性展示，绝不渲染成 None）
    flat_keys = ("top1_agreement", "logits_bitwise_equal", "logits_max_abs_diff", "n_seeds")
    if any(k in dm for k in flat_keys):
        L.append(f"- 步骤 2 候选（同一模型双模式）top-1 一致率 = {dm.get('top1_agreement')}"
                 f"（seeds={dm.get('n_seeds')}）")
        L.append(f"  - logits 逐位相同：{dm.get('logits_bitwise_equal')}；"
                 f"最大绝对偏差：{dm.get('logits_max_abs_diff')}")
    L.append("")
    L.append("| 用例 | top-1 一致率 | 一致数 / 总数 | logits 逐位相同 | logits 最大绝对偏差 |")
    L.append("| --- | --- | --- | --- | --- |")
    for name in sorted(dm.keys()):
        blob = dm[name]
        if not isinstance(blob, dict):
            continue
        if "top1_agreement" not in blob:
            continue
        L.append(
            f"| `{name}` | {blob.get('top1_agreement')} | "
            f"{blob.get('n_agree')} / {blob.get('n')} | "
            f"{blob.get('logits_bitwise_equal')} | {blob.get('logits_max_abs_diff')} |"
        )
    step1_blob = dm.get("step1") or {}
    for name in sorted(step1_blob.keys()):
        blob = step1_blob[name]
        L.append(f"- 步骤 1 跨模式（`{name}`）top-1 一致率 = {blob.get('top1_agreement')}"
                 f"（n={blob.get('n')}，一致 {blob.get('n_agree')}）")
    L.append("")
    L.append("## 分项 5 项指标（按任务 x 按 seed）")
    L.append("")
    L.append("| seed | 任务 | 1 步骤1答案准确率 | 2 步骤2 Recall@1 | 3 路由正确率 |"
             " 4 无匹配 P | 4 无匹配 R | 5 端到端正确率 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for seed, tasks in sorted((report.get("per_seed") or {}).items(),
                              key=lambda kv: int(kv[0])):
        for task, m in tasks.items():
            L.append(
                f"| {seed} | {task} | {m['step1_answer_accuracy']:.4f} | "
                f"{m['step2_recall_at_1']:.4f} | {m['routing_decision_accuracy']:.4f} | "
                f"{m['refusal_precision']:.4f} | {m['refusal_recall']:.4f} | "
                f"{m['e2e_answer_accuracy']:.4f} |"
            )
    L.append("")
    L.append("## 跨 seed（均值 +- 极差）")
    L.append("")
    L.append(format_cross_seed_markdown(report.get("cross_seed", {})))
    L.append("")
    L.append("## 产物与取证")
    L.append("")
    for item in report.get("artifacts", []):
        L.append(f"- `{item['path']}`（{item['bytes']} 字节，SHA256 `{item['sha256']}`）")
    L.append("")
    L.append("## 如实登记（未达标项 / 不可复现项 / 边界）")
    L.append("")
    for note in report.get("honest_notes", []):
        L.append(f"- {note}")
    L.append("")
    return "\n".join(L)


__all__ = [
    "STEP2_ARTIFACT_VERSION",
    "STEP2_DIR",
    "STEP2_VERIFY_DIR",
    "N3D_QA_PRODUCT_DIRS",
    "TASKS",
    "N_GRAM_ORDERS",
    "BUCKETS_PER_ORDER",
    "FEATURE_DIM",
    "TOPK",
    "TRAIN_SEEDS",
    "HELD_OUT_RATIO",
    "HELD_OUT_SEED",
    "FEATURE_ATOL",
    "DEFAULT_BACKEND",
    "AGGREGATED_FIELDS",
    "ZhBagVectorizer",
    "zh_feature_config",
    "TextRow",
    "DocSplit",
    "TaskData",
    "TrainOutcome",
    "MatchResult",
    "TextRowKeyTable",
    "TextRowMatcher",
    "resolve_product_dir",
    "load_text_rows",
    "load_row_index",
    "load_doclines_meta",
    "reproduce_doc_split",
    "verify_features_against_product",
    "load_answer_table",
    "load_task_pairs",
    "select_held_out_classes",
    "build_task_data",
    "merge_task_data",
    "filter_by_task",
    "build_adapter",
    "build_model",
    "pointer_keys_from_answers",
    "train_model",
    "build_key_table",
    "save_key_table",
    "load_key_table",
    "step2_artifact_name",
    "self_retrieval",
    "deterministic_baseline",
    "dual_mode_consistency",
    "routing_metrics",
    "task_metrics",
    "cross_seed_summary",
    "format_cross_seed_markdown",
    "write_report",
    "render_report_markdown",
    "sha256_bytes",
    "sha256_file",
    "canonical_dumps",
    "task_restricted_keys",
    "select_held_out_per_task",
    "product_file_manifest",
    "DEFAULT_HEAD_INPUT_MODE",
    "head_config_of",
    "module_file_manifest",
]

# ---------------------------------------------------------------------------
# 留出类（开集协议）——按任务分层的口径（**冻结**）
# ---------------------------------------------------------------------------


def task_restricted_keys(
    product_dir: str, task: str, answer_keys: Sequence[str]
) -> List[str]:
    """某任务实际使用的答案键（= 该任务 ``label==1`` 候选里落在统一答案表中的键）。

    参数
    ----
    product_dir : str
        产物目录。
    task : str
        任务名。
    answer_keys : Sequence[str]
        统一答案表。

    返回
    ----
    List[str]
        按统一答案表顺序排列的受限键集合。
    """
    used = {
        str(obj["candidate_answer_key"])
        for obj in load_task_pairs(product_dir, task)
        if int(obj["label"]) == 1
    }
    return [k for k in answer_keys if k in used]


def select_held_out_per_task(
    product_dir: str,
    answer_keys: Sequence[str],
    tasks: Sequence[str] = TASKS,
    ratio: float = HELD_OUT_RATIO,
    seed: int = HELD_OUT_SEED,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """**按任务分层**地留出答案类（全局留出会让 2 类的 judge 任务整体失守）。

    口径
    ----
    对每个任务 t：``R_t`` = 该任务的受限键集合；留出
    ``ceil(ratio * |R_t|)`` 个类（``|R_t| >= 2`` 时至少 1 个），
    由 ``default_rng(seed + 任务序号)`` 的确定性置换给出。
    全局留出集合 ``H`` = 各任务留出类的并集；候选空间 = ``answer_keys - H``。

    参数
    ----
    product_dir : str
        产物目录。
    answer_keys : Sequence[str]
        统一答案表。
    tasks : Sequence[str]
        任务列表。
    ratio : float
        每任务留出比例。
    seed : int
        抽取 seed（逐任务用 ``seed + i``）。

    返回
    ----
    Tuple[List[str], List[str], Dict[str, Any]]
        ``(保留类, 全局留出类 H, 取证信息)``。
    """
    keys = list(answer_keys)
    held: List[str] = []
    detail: Dict[str, Any] = {}
    for i, task in enumerate(tasks):
        r_t = task_restricted_keys(product_dir, task, keys)
        n_hold = int(math.ceil(float(ratio) * len(r_t)))
        if len(r_t) >= 2:
            n_hold = max(1, min(n_hold, len(r_t) - 1))
        else:
            n_hold = 0
        perm = _rng(int(seed) + i).permutation(len(r_t)) if r_t else np.zeros(0, dtype=int)
        picked = [r_t[int(j)] for j in perm[:n_hold]]
        for k in picked:
            if k not in held:
                held.append(k)
        detail[str(task)] = {
            "n_restricted": int(len(r_t)),
            "n_held_out": int(len(picked)),
            "seed": int(seed) + i,
            "held_out_keys": list(picked),
        }
    held_set = set(held)
    kept = [k for k in keys if k not in held_set]
    evidence = {
        "ratio": float(ratio),
        "seed": int(seed),
        "per_task": detail,
        "held_out_union": list(held),
        "n_held_out_union": int(len(held)),
        "n_kept": int(len(kept)),
    }
    return kept, held, evidence

def product_file_manifest(product_dir: str, tasks: Sequence[str] = TASKS) -> Dict[str, Any]:
    """列出本次运行**实际消费**的 n3d_qa 产物文件及其 SHA256（溯源字段）。

    参数
    ----
    product_dir : str
        产物目录。
    tasks : Sequence[str]
        任务列表（决定读取哪些 ``*_pairs.jsonl``）。

    返回
    ----
    Dict[str, Any]
        ``{"dir": ..., "files": [{"name","bytes","sha256"}]}``；缺失文件显式登记。
    """
    names = ["doclines.npz", "doclines_rows.jsonl", "doclines_row_index.jsonl",
             "answer_table.jsonl", "manifest.json"]
    names += [f"n3d_qa_{t}.npz" for t in tasks]
    names += [f"n3d_qa_{t}_pairs.jsonl" for t in tasks]
    files: List[Dict[str, Any]] = []
    missing: List[str] = []
    for name in names:
        path = os.path.join(product_dir, name)
        if not os.path.isfile(path):
            missing.append(str(name))
            continue
        files.append(
            {
                "name": str(name),
                "bytes": int(os.path.getsize(path)),
                "sha256": sha256_file(path),
            }
        )
    return {
        "dir": str(product_dir).replace("\\", "/"),
        "files": files,
        "missing": missing,
        "n_files": int(len(files)),
    }
