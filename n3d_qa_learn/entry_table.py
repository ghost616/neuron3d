"""n3d_qa_learn 统一「条目特征表」抽象（EntryKeyTable）+ 逐位精确查表（第一步 1a）。

设计文档三步走的**第一步 1a**：在动任何网络结构之前，先把「考卷」本身做成可审计、
可逐位复算的对象。本模块只管**一条不变量**：一张键表 = 若干条目，每条目有
**一组 float32 L2 归一化特征** + **一个条目 id** + **一个绑定输出**（QA 侧 = 答案展示
文本，文本侧 = 行原文），两侧共用同一套评测代码。

统一抽象
--------
.. code-block:: text

    EntryKeyTable
      keys     [N, D] float32，逐行 **L2 归一化**（范数 ∈ [1-1e-6, 1+1e-6]）
      entry_ids[N]    条目 id（唯一、顺序敏感）
      outputs  [N]    绑定输出（QA = 答案展示文本；文本 = 行原文）
      meta            dim / 归一化口径 / 构造来源 / 键表 SHA256 / 编码器口径指纹

两侧构造器
----------
* **QA 侧** :func:`build_qa_entry_tables`：只读 ``n3d_qa`` 冻结 QA 缓存（经
  :mod:`n3d_qa_learn.data`）复现切分，取 ``train_known`` 建表（``outputs`` = 答案展示
  文本）；``train_unknown`` + ``test_unknown`` 作为「未识别」负样本；特征经
  :func:`n3d_qa_learn.encoders.build_vectorizer`（**复用现有口径，不自造**）。
* **文本侧** :func:`text_entry_table_from_rows` / :func:`entry_table_from_text_row_table`：
  **包装**现有 :class:`n3d_qa_learn.step2.TextRowKeyTable`（键表字节与 ``sha256()``
  原样透传，**不改动其落盘格式**），补上统一的 ``entry_ids`` / ``outputs`` 视图。

逐位精确查表（G1 门禁）
-----------------------
:meth:`EntryKeyTable.lookup` 返回该行的 **float32 裸字节**；:func:`verify_bitwise_lookup`
断言该字节与表中对应行的裸字节**逐字节相等**（不是"接近"）。报告里的
``bitwise_lookup`` 一节即由它产出。

上游边界（零改动）
------------------
本模块**只读** import ``n3d_qa_learn`` 内既有模块与 ``n3d_qa`` 的冻结产物；
不修改任何上游模块源码 / 产物 / ``current_spec.md``；零新增依赖（torch / numpy + 标准库）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import DEFAULT_QA_CACHE_DIR, DEFAULT_TEXT_DIR, QARecord
from .encoders import (
    DEFAULT_MODEL_DIR,
    ENCODER_BGE_M3,
    ENCODER_LOCAL_HASH,
    ROLE_QUESTION,
    ROLE_TEXT_LINE,
    EncoderConfig,
    build_vectorizer,
)
from .step2 import (
    TextRow,
    TextRowKeyTable,
    build_key_table,
    load_doclines_meta,
    load_row_index,
    reproduce_doc_split,
    resolve_product_dir,
)
from .train import TrainConfig, TrainingData, build_training_data

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

#: 条目特征表的**协议版本**（结构变化必须递增；进指纹）。
ENTRY_TABLE_VERSION: str = "n3dqa-entry-table-v1"

#: 归一化口径的规范字符串（进指纹；改口径即改指纹）。
ENTRY_L2_NORM_SPEC: str = "l2_rowwise_then_float32"

#: 行范数允差（float32 下的 L2 归一化漂移；现场实测文本侧 2665 行落在
#: [0.99999982, 1.00000012]，故 1e-6 足够且不掩盖真实未归一化）。
NORM_ATOL: float = 1e-6

#: 键表内容指纹 schema（进指纹，防止两种表撞同一指纹）。
KEY_FINGERPRINT_SCHEMA: str = "entry-key-table-v1"

#: QA 侧建表的来源切分（**冻结**，与 ``exp_repr`` 基线档同口径）。
QA_KNOWN_SPLIT: str = "train_known"

#: QA 侧切分口径（与 ``exp_repr.BASE_*`` 逐项一致：C=10 / min_questions=8 /
#: test_every=3 / test_per_class=2 / unknown_train_cap=500 / split_seed=42）。
QA_MAX_CLASSES: int = 10
QA_MIN_QUESTIONS: int = 8
QA_TEST_EVERY: int = 3
QA_TEST_PER_CLASS: int = 2
QA_UNKNOWN_TRAIN_CAP: int = 500
QA_SPLIT_SEED: int = 42

#: 特征档名（与 ``n3d_qa_learn.exp_repr.FEATURE_PROFILES`` 的键逐字一致）。
PROFILE_LEXICAL: str = "lexical-88"
PROFILE_SEMANTIC: str = "bge-m3-1024"

#: 特征档注册表（**本模块唯一注册点**；每档同时声明题面口径与文本行口径，
#: 两者**必须同维**，否则两侧不可共用同一套评测代码）。
PAIR_PROFILES: Dict[str, Dict[str, EncoderConfig]] = {
    PROFILE_LEXICAL: {
        ROLE_QUESTION: EncoderConfig(name=ENCODER_LOCAL_HASH, role=ROLE_QUESTION),
        ROLE_TEXT_LINE: EncoderConfig(name=ENCODER_LOCAL_HASH, role=ROLE_TEXT_LINE),
    },
    PROFILE_SEMANTIC: {
        ROLE_QUESTION: EncoderConfig(
            name=ENCODER_BGE_M3, role=ROLE_QUESTION, source=DEFAULT_MODEL_DIR
        ),
        ROLE_TEXT_LINE: EncoderConfig(
            name=ENCODER_BGE_M3, role=ROLE_TEXT_LINE, source=DEFAULT_MODEL_DIR
        ),
    },
}

#: 档名 -> 声明维度（``local-hash``：hash_dim 80 + 8 长度特征 = 88；
#: ``bge-m3``：``hidden_size`` = 1024）。
PROFILE_EXPECT_DIM: Dict[str, int] = {
    PROFILE_LEXICAL: 88,
    PROFILE_SEMANTIC: 1024,
}

#: QA 侧未知样本的条目 id 前缀（避免与 known 条目 id 撞名）。
UNKNOWN_ID_PREFIX: str = "unknown:"


def profile_names() -> List[str]:
    """已登记的特征档名（排序后，确定性）。"""
    return sorted(PAIR_PROFILES.keys())


def encoder_config_for(profile: str, role: str) -> EncoderConfig:
    """取某档某角色（``question`` / ``text_line``）的编码器配置。

    参数
    ----
    profile : str
        特征档名（见 :func:`profile_names`）。
    role : str
        ``question``（QA 侧题面）或 ``text_line``（文本侧行）。

    返回
    ----
    EncoderConfig
        编码器配置（未知名立即报错，不静默回落）。
    """
    if str(profile) not in PAIR_PROFILES:
        raise KeyError(f"未知特征档 {profile!r}；可用档 = {profile_names()}")
    table = PAIR_PROFILES[str(profile)]
    if str(role) not in table:
        raise KeyError(
            f"特征档 {profile!r} 不支持角色 {role!r}；可用角色 = {sorted(table.keys())}"
        )
    return table[str(role)]


def expect_dim_of(profile: str) -> int:
    """某档的声明维度（未知名立即报错）。"""
    if str(profile) not in PROFILE_EXPECT_DIM:
        raise KeyError(f"未知特征档 {profile!r}；可用档 = {profile_names()}")
    return int(PROFILE_EXPECT_DIM[str(profile)])


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def canonical_dumps(obj: Any) -> bytes:
    """规范化 JSON 字节（排序键 + 紧凑分隔符），用于一切指纹计算。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(blob: bytes) -> str:
    """字节流的 SHA256（小写十六进制）。"""
    return hashlib.sha256(blob).hexdigest()


def as_float32_matrix(keys: Any) -> torch.Tensor:
    """把任意输入转成 **C 连续 float32 [N, D]** 张量（不做归一化、不做范数校验）。"""
    tensor = torch.as_tensor(keys, dtype=torch.float32)
    if tensor.dim() != 2:
        raise ValueError(f"键表必须是二维 [N, D]，当前形状 = {tuple(tensor.shape)}")
    return tensor.contiguous()


def l2_normalize_rows(matrix: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """逐行 L2 归一化（**扰动后一律重新归一化**的唯一实现）。

    参数
    ----
    matrix : Any
        任意 ``[N, D]`` 输入（以 float64 计算范数，再落回 float32）。

    返回
    ----
    Tuple[np.ndarray, Dict[str, Any]]
        ``(float32[N, D] 归一化结果, 取证字典)``。取证字典含零范数行数与其下标
        （**零范数显式处置**：不静默产出 NaN，由调用方决定报错或如实登记）。

    异常
    ------
    ValueError
        输入不是二维数组。
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"待归一化矩阵必须是二维 [N, D]，当前形状 = {arr.shape}")
    norms = np.linalg.norm(arr, axis=1)
    zero = np.flatnonzero(norms <= 0.0)
    out = np.zeros_like(arr)
    if zero.size:
        # 零范数行保持全零（不做 1/0，不产生 NaN），行号暴露给调用方显式处置
        nz = np.setdiff1d(np.arange(arr.shape[0]), zero, assume_unique=False)
        if nz.size:
            out[nz] = arr[nz] / norms[nz][:, None]
    else:
        out = arr / norms[:, None]
    evidence = {
        "n_rows": int(arr.shape[0]),
        "n_zero_norm": int(zero.size),
        "zero_norm_rows_head": [int(i) for i in zero[:20].tolist()],
        "norm_min": float(norms.min()) if norms.size else 0.0,
        "norm_max": float(norms.max()) if norms.size else 0.0,
    }
    return out.astype(np.float32), evidence


def key_table_fingerprint(
    keys: torch.Tensor, entry_ids: Sequence[str], outputs: Sequence[str], dim: int
) -> str:
    """键表**内容指纹**：schema + 形状 + dtype + 裸字节 + 条目 id + 绑定输出 + 维度。

    说明
    ----
    用裸字节（而非数值摘要）是为了让指纹与「逐位精确查表」这条门禁同一口径：
    指纹一致 ⇔ 裸字节一致。
    """
    tensor = as_float32_matrix(keys)
    payload = {
        "schema": KEY_FINGERPRINT_SCHEMA,
        "version": ENTRY_TABLE_VERSION,
        "shape": [int(x) for x in tensor.shape],
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "dim": int(dim),
        "keys_bytes": tensor.numpy().tobytes().hex(),
        "entry_ids": [str(x) for x in entry_ids],
        "outputs": [str(x) for x in outputs],
    }
    return sha256_bytes(canonical_dumps(payload))


def verify_bitwise_lookup(
    table: "EntryKeyTable", indices: Optional[Sequence[int]] = None
) -> Dict[str, Any]:
    """**逐位精确查表**门禁：返回的 float32 裸字节必须与表中对应行逐字节相等。

    参数
    ----
    table : EntryKeyTable
        待验证的表。
    indices : Optional[Sequence[int]]
        要抽查的行下标；``None`` = 全表每一行都查。

    返回
    ----
    Dict[str, Any]
        ``n_checked`` / ``n_exact`` / ``exact_frac`` / ``mismatch_head``（不一致行的
        下标与两侧 SHA256）。**主判据是 ``exact_frac == 1.0``。**
    """
    order = list(range(int(table.size))) if indices is None else [int(i) for i in indices]
    mismatch: List[Dict[str, Any]] = []
    n_exact = 0
    for i in order:
        raw, _out = table.lookup(int(i))
        expected = table.row_bytes(int(i))
        if raw == expected:
            n_exact += 1
        elif len(mismatch) < 20:
            mismatch.append(
                {
                    "index": int(i),
                    "lookup_sha256": sha256_bytes(raw),
                    "table_sha256": sha256_bytes(expected),
                }
            )
    n = len(order)
    return {
        "n_checked": int(n),
        "n_exact": int(n_exact),
        "exact_frac": float(n_exact / max(1, n)),
        "mismatch_head": mismatch,
        "rule": "lookup(i) 的 float32 裸字节必须与 keys[i] 的裸字节逐字节相等（不是近似）",
    }


# ---------------------------------------------------------------------------
# 统一条目特征表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryKeyTable:
    """统一「条目特征表」：一行 = 一个条目（QA 问题 / 文本行）的 L2 归一化特征。

    参数
    ----
    keys : torch.Tensor
        ``[N, D]`` float32，逐行 L2 归一化（构造期校验）。
    entry_ids : Sequence[str]
        条目 id（唯一、顺序敏感）。
    outputs : Sequence[str]
        绑定输出：QA 侧 = 答案展示文本；文本侧 = 行原文。
    kind : str
        构造来源标记（``qa`` / ``text``），进元数据与指纹。
    dim : int
        连接参数 ``D``（必须等于 ``keys.shape[1]``）。
    norm_spec : str
        归一化口径字符串（默认 :data:`ENTRY_L2_NORM_SPEC`）。
    encoder_profile : str
        编码器档名（进元数据）。
    encoder_fingerprint : str
        编码器口径指纹（``vectorizer.fingerprint()``；进元数据与守卫）。
    source : Dict[str, Any]
        构造来源明细（切分口径 / 产物路径 / 库行数等，JSON 可序列化）。
    verify_norm : bool
        构造期是否校验行范数（默认 ``True``；**禁止**在正常路径上关掉）。

    关键不变量（构造期，违反即抛）
    -----------------------------
    1. ``keys`` 二维、``dtype == torch.float32``、C 连续；
    2. 行范数 ∈ ``[1 - NORM_ATOL, 1 + NORM_ATOL]``（``verify_norm=True`` 时）；
    3. ``len(entry_ids) == len(outputs) == N``，且 ``entry_ids`` 无重复；
    4. ``keys.shape[1] == dim``。
    """

    keys: torch.Tensor
    entry_ids: List[str]
    outputs: List[str]
    kind: str = ""
    dim: int = 0
    norm_spec: str = ENTRY_L2_NORM_SPEC
    encoder_profile: str = ""
    encoder_fingerprint: str = ""
    source: Dict[str, Any] = field(default_factory=dict)
    verify_norm: bool = True

    def __post_init__(self) -> None:
        """构造期校验（历史纠正记录 #10：构造期不变量与训练后量严格分开）。"""
        tensor = self.keys
        if not torch.is_tensor(tensor):
            raise TypeError(f"keys 必须是 torch.Tensor，当前 {type(tensor).__name__}")
        if tensor.dim() != 2:
            raise ValueError(f"keys 必须是二维 [N, D]，当前形状 = {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32:
            raise ValueError(
                "keys 的 dtype 必须是 torch.float32（逐位精确查表的前提），"
                f"当前 {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError("keys 必须是 C 连续张量（裸字节口径要求内存布局唯一）")
        n_rows, n_cols = int(tensor.shape[0]), int(tensor.shape[1])
        if int(self.dim) != n_cols:
            raise ValueError(f"dim={self.dim} 与 keys.shape[1]={n_cols} 不一致")
        if len(self.entry_ids) != n_rows:
            raise ValueError(
                f"entry_ids 数 {len(self.entry_ids)} 与键表行数 {n_rows} 不一致"
            )
        if len(self.outputs) != n_rows:
            raise ValueError(f"outputs 数 {len(self.outputs)} 与键表行数 {n_rows} 不一致")
        if len(set(self.entry_ids)) != len(self.entry_ids):
            raise ValueError("entry_ids 存在重复；条目 id 必须唯一（顺序敏感）")
        if self.verify_norm and n_rows:
            norms = torch.linalg.norm(tensor, dim=1)
            lo = float(norms.min().item())
            hi = float(norms.max().item())
            if not (lo >= 1.0 - NORM_ATOL and hi <= 1.0 + NORM_ATOL):
                raise ValueError(
                    f"键表行范数落在 [{lo}, {hi}]，超出 L2 归一化允差 "
                    f"[{1.0 - NORM_ATOL}, {1.0 + NORM_ATOL}]；"
                    "拒绝在未归一化的特征上做余弦检索"
                )

    @property
    def size(self) -> int:
        """条目数 ``N``。"""
        return int(self.keys.shape[0])

    @property
    def key_dim(self) -> int:
        """连接参数 ``D``（与 :attr:`dim` 必须一致，构造期已校验）。"""
        return int(self.keys.shape[1])

    def row_bytes(self, index: int) -> bytes:
        """表中第 ``index`` 行的 float32 裸字节。"""
        i = int(index)
        if not (0 <= i < self.size):
            raise IndexError(f"行下标 {i} 越界（表大小 {self.size}）")
        return self.keys[i].detach().cpu().contiguous().numpy().tobytes()

    def lookup(self, index: int) -> Tuple[bytes, str]:
        """**逐位精确查表**：返回 ``(该行 float32 裸字节, 绑定输出)``。

        该返回值与 :meth:`row_bytes` 必须逐字节相等（由 :func:`verify_bitwise_lookup`
        断言，是本轮 G1 门禁的直接依据）。
        """
        i = int(index)
        if not (0 <= i < self.size):
            raise IndexError(f"行下标 {i} 越界（表大小 {self.size}）")
        return self.row_bytes(i), str(self.outputs[i])

    def entry_id(self, index: int) -> str:
        """表中第 ``index`` 行的条目 id。"""
        return str(self.entry_ids[int(index)])

    def sha256(self) -> str:
        """键表内容指纹（与 :func:`key_table_fingerprint` 同一口径）。"""
        return key_table_fingerprint(self.keys, self.entry_ids, self.outputs, int(self.dim))

    def norm_range(self) -> Dict[str, float]:
        """现场实测的行范数区间（取证用）。"""
        if not self.size:
            return {"min": 0.0, "max": 0.0}
        norms = torch.linalg.norm(self.keys, dim=1)
        return {"min": float(norms.min().item()), "max": float(norms.max().item())}

    def meta(self) -> Dict[str, Any]:
        """元数据（dim / 归一化口径 / 构造来源 / 键表 SHA256 / 编码器口径指纹）。"""
        return {
            "version": ENTRY_TABLE_VERSION,
            "kind": str(self.kind),
            "n_entries": int(self.size),
            "dim": int(self.dim),
            "norm_spec": str(self.norm_spec),
            "dtype": str(self.keys.dtype).replace("torch.", ""),
            "contiguous": bool(self.keys.is_contiguous()),
            "norm_range": self.norm_range(),
            "key_table_sha256": self.sha256(),
            "encoder_profile": str(self.encoder_profile),
            "encoder_fingerprint": str(self.encoder_fingerprint),
            "source": dict(self.source),
        }


@dataclass(frozen=True)
class EntryStatements:
    """一批**待编码文本**的特征码头（矩阵 + 条目 id + 绑定输出）。

    与 :class:`EntryKeyTable` 的差别：这里**不校验** L2 归一化（扰动后的矩阵由扰动器
    负责重新归一化），它只是「文本 → 特征」的载体。
    """

    feats: np.ndarray
    entry_ids: List[str]
    outputs: List[str]
    kind: str = ""
    dim: int = 0
    text_sha256: str = ""

    @property
    def n(self) -> int:
        """条目数。"""
        return int(self.feats.shape[0])

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化摘要（**不落盘特征矩阵**，只落形状与指纹）。"""
        return {
            "kind": str(self.kind),
            "n": int(self.n),
            "dim": int(self.dim),
            "shape": [int(x) for x in self.feats.shape],
            "dtype": str(self.feats.dtype),
            "text_sha256": str(self.text_sha256),
        }


def encode_statements(
    vectorizer: Any, texts: Sequence[str], *, batch_size: int = 64
) -> np.ndarray:
    """把一批文本编码为 ``float32[N, D]``（**逐行 L2 归一化**）。

    两条路径都走同一口径：先取编码器输出，若**未归一化**（范数不在允差内）再补做
    逐行 L2 归一化 —— 避免二次归一化引入不必要的舍入漂移。
    """
    n = len(texts)
    dim = int(vectorizer.dim)
    out = np.zeros((n, dim), dtype=np.float32)
    step = max(1, int(batch_size))
    for b0 in range(0, n, step):
        chunk = [str(t) for t in texts[b0 : b0 + step]]
        if hasattr(vectorizer, "encode_matrix"):
            arr = np.asarray(vectorizer.encode_matrix(chunk), dtype=np.float32)
            if arr.shape != (len(chunk), dim):
                raise ValueError(
                    f"编码器 encode_matrix 返回形状 {arr.shape} 与声明 "
                    f"(n={len(chunk)}, D={dim}) 不一致"
                )
        else:
            arr = np.asarray([vectorizer.encode(t) for t in chunk], dtype=np.float32)
            if arr.shape != (len(chunk), dim):
                raise ValueError(
                    f"编码器 encode 返回形状 {arr.shape} 与声明 "
                    f"(n={len(chunk)}, D={dim}) 不一致"
                )
        norms = np.linalg.norm(arr.astype(np.float64), axis=1)
        need = np.flatnonzero(np.abs(norms - 1.0) > NORM_ATOL)
        if need.size:
            arr = arr.copy()
            safe = np.where(norms > 0.0, norms, 1.0)
            arr[need] = (
                arr[need].astype(np.float64) / safe[need][:, None]
            ).astype(np.float32)
        out[b0 : b0 + len(chunk)] = arr
    return out


def build_key_table_from_statements(
    feats: np.ndarray,
    entry_ids: Sequence[str],
    outputs: Sequence[str],
    *,
    kind: str,
    dim: int,
    encoder_profile: str = "",
    encoder_fingerprint: str = "",
    source: Optional[Dict[str, Any]] = None,
    verify_norm: bool = True,
) -> EntryKeyTable:
    """由特征矩阵构造 :class:`EntryKeyTable`（内做一次逐行 L2 归一化）。

    逐行 L2 归一化的**唯一实现**是 :func:`l2_normalize_rows`；零范数行**不静默**
    通过 —— 本函数显式报错（扰动侧另有「先归一化再登记」的处置）。
    """
    keys, evidence = l2_normalize_rows(feats)
    if int(evidence["n_zero_norm"]) > 0:
        raise ValueError(
            "构造键表时出现零范数行："
            f"n_zero_norm={evidence['n_zero_norm']}，"
            f"行下标 = {evidence['zero_norm_rows_head']}；"
            "零范数行会让余弦检索退化为全 0 打分，拒绝静默继续"
        )
    return EntryKeyTable(
        keys=torch.from_numpy(np.ascontiguousarray(keys, dtype=np.float32)),
        entry_ids=[str(x) for x in entry_ids],
        outputs=[str(x) for x in outputs],
        kind=str(kind),
        dim=int(dim),
        encoder_profile=str(encoder_profile),
        encoder_fingerprint=str(encoder_fingerprint),
        source=dict(source or {}),
        verify_norm=bool(verify_norm),
    )


# ---------------------------------------------------------------------------
# QA 侧构造器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QAEntryBundle:
    """QA 侧的条目表 + 未识别负样本 + 辅助查询集（一次装配、多处共用）。

    属性
    ----
    table : EntryKeyTable
        ``train_known`` 条目表（``outputs`` = 答案展示文本）。
    unknown : EntryStatements
        「未识别」负样本的待编码语句（``train_unknown`` + ``test_unknown``）。
    known_queries : EntryStatements
        QA 侧辅助查询集（``test_known``；**统计意义弱**，报告须显式标注）。
    dim : int
        连接参数 ``D``。
    evidence : Dict[str, Any]
        切分与规模的取证信息。
    """

    table : EntryKeyTable
    unknown : EntryStatements
    known_queries : EntryStatements
    dim : int
    evidence : Dict[str, Any]
    known_answer_keys: List[str] = field(default_factory=list)
    unknown_answer_keys: List[str] = field(default_factory=list)
    query_answer_keys: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化摘要。"""
        return {
            "table": self.table.meta(),
            "unknown": self.unknown.as_dict(),
            "known_queries": self.known_queries.as_dict(),
            "evidence": dict(self.evidence),
            "known_answer_keys": list(self.known_answer_keys),
            "n_known_answer_keys": int(len(self.known_answer_keys)),
            "n_unknown_answer_keys": int(len(self.unknown_answer_keys)),
            "n_query_answer_keys": int(len(self.query_answer_keys)),
        }


def load_qa_training_data(
    *,
    qa_cache_dir: str = "",
    text_dir: str = "",
    split_seed: int = QA_SPLIT_SEED,
    max_classes: int = QA_MAX_CLASSES,
    min_questions: int = QA_MIN_QUESTIONS,
    test_every: int = QA_TEST_EVERY,
    test_per_class: int = QA_TEST_PER_CLASS,
    unknown_train_cap: int = QA_UNKNOWN_TRAIN_CAP,
) -> TrainingData:
    """只读装配 QA 侧数据（**复用既有 ``data`` / ``train`` 路径，不自造切分**）。

    口径与 ``exp_repr`` 基线档逐项一致（C=10 / min_questions=8 / test_every=3 /
    test_per_class=2 / unknown_train_cap=500 / split_seed=42），故现场得到
    ``train_known 149 / test_known 20 / train_unknown 500 / test_unknown 2393``。
    """
    cfg = TrainConfig(
        seed=int(split_seed),
        split_seed=int(split_seed),
        max_classes=int(max_classes),
        min_questions=int(min_questions),
        test_every=int(test_every),
        test_per_class=int(test_per_class),
        unknown_train_cap=int(unknown_train_cap),
        qa_cache_dir=str(qa_cache_dir) if qa_cache_dir else DEFAULT_QA_CACHE_DIR,
        text_dir=str(text_dir) if text_dir else DEFAULT_TEXT_DIR,
    )
    # 向量化器由调用方经 encoders.build_vectorizer 构造（本轮不做任何训练）；
    # 这里传词面档题面配置，只为让 build_training_data 的编码器装配口径唯一。
    return build_training_data(
        cfg, encoder_config=encoder_config_for(PROFILE_LEXICAL, ROLE_QUESTION)
    )


def qa_record_texts(records: Sequence[QARecord]) -> List[str]:
    """取一批 QA 记录的问题文本（问答对的**问题面**）。"""
    return [str(r.question) for r in records]


def question_is_blank(text: str) -> bool:
    """题面在**归一化后是否为空**。

    口径来源（审查 I4 更正措辞）：**沿用 ``n3d_qa_learn.features.normalize_text`` 的口径**
    （NFKC 折叠 + casefold + 空白折叠），归一化后为空串即视为「空问题 / 仅空白」
    —— 按契约**不进入分母**。

    **不沿用** ``data.check_question``：该函数的 docstring 明确是「超长 -> 立即报错；
    **空 / 空白放行**」，它对空问题**放行、不做过滤**（它只是内部 import 了同一个
    ``features.normalize_text`` 做归一化），故不是本判定的口径来源。
    """
    from .features import normalize_text

    return str(normalize_text(str(text))) == ""


def _drop_blank_questions(
    records: Sequence[QARecord],
) -> Tuple[List[QARecord], Dict[str, Any]]:
    """过滤「空问题 / 仅空白」记录并返回取证（审查 I2）。

    返回
    ----
    Tuple[List[QARecord], Dict[str, Any]]
        ``(保留的记录, 取证字典)``。取证含 ``n_dropped`` 与 ``dropped_ids``（最多 20 个）。
    """
    kept: List[QARecord] = []
    dropped: List[str] = []
    for rec in records:
        if question_is_blank(rec.question):
            dropped.append(str(rec.qid))
        else:
            kept.append(rec)
    return kept, {
        "n_in": int(len(records)),
        "n_kept": int(len(kept)),
        "n_dropped": int(len(dropped)),
        "dropped_ids_head": dropped[:20],
        "rule": (
            "题面经 features.normalize_text 归一化后为空即视为「空问题 / 仅空白」，"
            "**不计入分母**（不进入键表、不进入查询集）"
        ),
    }


def build_qa_entry_tables(
    data: TrainingData,
    vectorizer: Any,
    *,
    profile: str,
    batch_size: int = 64,
) -> QAEntryBundle:
    """由 QA 数据装配「条目特征表 + 未识别负样本 + 辅助查询集」。

    参数
    ----
    data : TrainingData
        :func:`load_qa_training_data` 得到的切分（**同一份**切分，不得重抽）。
    vectorizer : Any
        题面编码器（``role=question``；经 ``encoders.build_vectorizer`` 构造）。
    profile : str
        特征档名（进元数据）。
    batch_size : int
        编码批大小。

    返回
    ----
    QAEntryBundle
        条目表（``train_known``）、未识别负样本（``train_unknown + test_unknown``）、
        辅助查询集（``test_known``）与取证信息。

    异常
    ------
    ValueError
        编码器实测维度与所在档声明维度不一致（拒绝静默错配）。
    """
    want = expect_dim_of(profile)
    if int(vectorizer.dim) != int(want):
        raise ValueError(
            f"特征档 {profile!r} 的题面编码器实测维度 {vectorizer.dim} "
            f"与声明 {want} 不一致（拒绝静默错配）"
        )
    # 「空问题 / 仅空白」契约（审查 I2）：沿用 ``data.check_question`` 的口径 ——
    # 归一化后为空的题面**不进入分母**。``local-hash`` 对空串返回全零向量，若不过滤，
    # 它会先被 ``l2_normalize_rows`` 归零、再由 ``build_key_table_from_statements``
    # 抛 ``ValueError``（表现为「构造失败」而不是「不计入分母」，与契约不符）。
    # 这里显式过滤并**登记计数与 id**（本批真实数据为 0 条，但口径必须实现）。
    known_raw = list(data.splits.train_known)
    unknown_raw = list(data.splits.train_unknown) + list(data.splits.test_unknown)
    queries_raw = list(data.splits.test_known)
    known, known_blank = _drop_blank_questions(known_raw)
    unknown, unknown_blank = _drop_blank_questions(unknown_raw)
    queries, query_blank = _drop_blank_questions(queries_raw)

    known_feats = encode_statements(
        vectorizer, qa_record_texts(known), batch_size=batch_size
    )
    query_texts = qa_record_texts(queries)
    query_feats = encode_statements(vectorizer, query_texts, batch_size=batch_size)
    unknown_texts = qa_record_texts(unknown)
    unknown_feats = encode_statements(vectorizer, unknown_texts, batch_size=batch_size)

    table = build_key_table_from_statements(
        known_feats,
        [str(r.qid) for r in known],
        [str(r.answer_display) for r in known],
        kind="qa",
        dim=int(vectorizer.dim),
        encoder_profile=str(profile),
        encoder_fingerprint=str(vectorizer.fingerprint()),
        source={
            "split": QA_KNOWN_SPLIT,
            "role": ROLE_QUESTION,
            "split_seed": int(QA_SPLIT_SEED),
            "split_summary": data.splits.summary(),
            "answer_keys_sha256": sha256_bytes(
                canonical_dumps(list(data.corpus.answer_keys))
            ),
            "n_answer_classes": int(data.corpus.n_classes),
        },
    )
    unknown_stmt = EntryStatements(
        feats=unknown_feats,
        entry_ids=[f"{UNKNOWN_ID_PREFIX}{r.qid}" for r in unknown],
        outputs=[str(r.answer_display) for r in unknown],
        kind="qa_unknown",
        dim=int(vectorizer.dim),
        text_sha256=sha256_bytes(canonical_dumps(unknown_texts)),
    )
    query_stmt = EntryStatements(
        feats=query_feats,
        entry_ids=[str(r.qid) for r in queries],
        outputs=[str(r.answer_display) for r in queries],
        kind="qa_query",
        dim=int(vectorizer.dim),
        text_sha256=sha256_bytes(canonical_dumps(query_texts)),
    )
    evidence = {
        "split_summary": data.splits.summary(),
        "n_answer_classes": int(data.corpus.n_classes),
        "answer_keys_sha256": sha256_bytes(
            canonical_dumps(list(data.corpus.answer_keys))
        ),
        "qa_files": [str(x) for x in data.qa_files],
        "encoder_profile": str(profile),
        "encoder_fingerprint": str(vectorizer.fingerprint()),
        "dim": int(vectorizer.dim),
        "role": ROLE_QUESTION,
        "blank_question_filter": {
            "known": known_blank,
            "unknown": unknown_blank,
            "queries": query_blank,
        },
        "split_source": (
            "data.build_answer_space + data.make_splits（与 exp_repr 基线档同口径："
            f"max_classes={QA_MAX_CLASSES} / min_questions={QA_MIN_QUESTIONS} / "
            f"test_every={QA_TEST_EVERY} / test_per_class={QA_TEST_PER_CLASS} / "
            f"unknown_train_cap={QA_UNKNOWN_TRAIN_CAP} / split_seed={QA_SPLIT_SEED}）"
        ),
    }
    return QAEntryBundle(
        table=table,
        unknown=unknown_stmt,
        known_queries=query_stmt,
        dim=int(vectorizer.dim),
        evidence=evidence,
        known_answer_keys=[str(r.answer_key) for r in known],
        unknown_answer_keys=[str(r.answer_key) for r in unknown],
        query_answer_keys=[str(r.answer_key) for r in queries],
    )


# ---------------------------------------------------------------------------
# 文本侧构造器（包装既有 TextRowKeyTable，不改其落盘格式）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextEntryBundle:
    """文本侧的条目表 + 查询集 + 既有冻结键表（**原样透传**）。

    属性
    ----
    table : EntryKeyTable
        统一视图（键表字节与 ``text_row_table.sha256()`` 同源）。
    text_row_table : TextRowKeyTable
        **既有**步骤 2 冻结键表对象（落盘格式与 SHA256 口令逐位不变）。
    query_feats : np.ndarray
        查询行特征（冻结划分的 query 行；逐行 L2 归一化）。
    query_entry_ids : List[str]
        查询行 id（与 ``query_feats`` 同行序）。
    query_positions : List[int]
        查询行在条目表中的行下标（自检索的金标位置）。
    evidence : Dict[str, Any]
        划分与规模取证。
    """

    table: EntryKeyTable
    text_row_table: TextRowKeyTable
    query_feats: np.ndarray
    query_entry_ids: List[str]
    query_positions: List[int]
    evidence: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化摘要（**不落盘特征矩阵**）。"""
        return {
            "table": self.table.meta(),
            "text_row_key_table_sha256": str(self.text_row_table.sha256()),
            "text_row_key_table_shape": [int(x) for x in self.text_row_table.keys.shape],
            "n_query": int(len(self.query_entry_ids)),
            "query_feats_shape": [int(x) for x in self.query_feats.shape],
            "evidence": dict(self.evidence),
        }


def text_entry_table_from_rows(
    vectorizer: Any,
    rows: Sequence[TextRow],
    *,
    profile: str,
    product_dir: str = "",
    verify_against_row_table: bool = True,
) -> TextEntryBundle:
    """包装既有 :class:`TextRowKeyTable` 为统一 :class:`EntryKeyTable`（文本侧）。

    关键约束（**不得违反**）
    ----------------------
    1. **不重造键表**：键表一律经既有的 :func:`n3d_qa_learn.step2.build_key_table`
       构造，本函数只做视图包装；包装后 ``EntryKeyTable.sha256()`` 以 ``keys``
       裸字节为准，与 ``TextRowKeyTable.sha256()`` 同源；
    2. **不改落盘格式**：本函数不写任何文件；
    3. 查询行取冻结划分的 ``query_index``，金标 = 查询行在条目表中的行下标
       （自检索：命中自身行即正确）。

    参数
    ----
    vectorizer : Any
        文本行编码器（``role=text_line``）。
    rows : Sequence[TextRow]
        ``n3d_qa`` 冻结产物行表（全量）。
    profile : str
        特征档名。
    product_dir : str
        产物目录（空串 = ``step2.resolve_product_dir`` 自动定位）。
    verify_against_row_table : bool
        是否现场断言「包装视图的键表字节」与既有 ``TextRowKeyTable`` 第 0 行一致。

    返回
    ----
    TextEntryBundle
        统一视图 + 查询集 + 取证。
    """
    want = expect_dim_of(profile)
    if int(vectorizer.dim) != int(want):
        raise ValueError(
            f"特征档 {profile!r} 的文本行编码器实测维度 {vectorizer.dim} "
            f"与声明 {want} 不一致（拒绝静默错配）"
        )
    resolved = resolve_product_dir(str(product_dir))
    library_index = list(range(len(rows)))
    text_row_table = build_key_table(vectorizer, rows, library_index)
    # 包装路径**只有一处**（:func:`entry_table_from_text_row_table`），避免两份口径
    table = entry_table_from_text_row_table(
        text_row_table,
        profile=str(profile),
        encoder_fingerprint=str(vectorizer.fingerprint()),
        source={
            "role": ROLE_TEXT_LINE,
            "product_dir": str(resolved),
            "library_rows": int(len(library_index)),
            "text_row_key_table_sha256": str(text_row_table.sha256()),
            "retrieval_pool_rule": (
                "检索池 = 冻结行表全量（n3d_qa label_rule：candidate library row IS the "
                "query row）；查询集 = 冻结划分出的 query 行 ∩ 检索池"
            ),
        },
    )
    if verify_against_row_table:
        if table.size != text_row_table.size:
            raise ValueError(
                f"包装视图行数 {table.size} 与既有 TextRowKeyTable 行数 "
                f"{text_row_table.size} 不一致"
            )
        ref = text_row_table.keys[0].detach().cpu().contiguous().numpy().tobytes()
        if table.row_bytes(0) != ref:
            raise ValueError("包装视图第 0 行的裸字节与既有 TextRowKeyTable 不一致")

    meta = load_doclines_meta(resolved)
    row_index = load_row_index(resolved)
    split = reproduce_doc_split(rows, row_index, meta)
    pos_of = {rid: i for i, rid in enumerate(table.entry_ids)}
    query_index = [int(i) for i in split.query_index]
    missing = [int(i) for i in query_index if str(rows[int(i)].row_id) not in pos_of]
    if missing:
        raise ValueError(
            f"有 {len(missing)} 条查询行不在条目表中（前几个下标 = {missing[:5]}）；"
            "自检索要求查询行与条目表来自同一全量行表"
        )
    query_feats = encode_statements(
        vectorizer, [str(rows[int(i)].text) for i in query_index]
    )
    evidence = {
        "product_dir": str(resolved),
        "n_rows": int(len(rows)),
        "library_rows": int(split.n_library),
        "query_rows": int(split.n_query),
        "intersection": int(split.evidence["intersection"]),
        "union": int(split.evidence["union"]),
        "reproduced_equals_index": bool(split.evidence["reproduced_equals_index"]),
        "split_seed": int(split.seed),
        "dim": int(vectorizer.dim),
        "encoder_profile": str(profile),
        "encoder_fingerprint": str(vectorizer.fingerprint()),
        "role": ROLE_TEXT_LINE,
        "text_row_key_table_sha256": str(text_row_table.sha256()),
    }
    return TextEntryBundle(
        table=table,
        text_row_table=text_row_table,
        query_feats=query_feats,
        query_entry_ids=[str(rows[int(i)].row_id) for i in query_index],
        query_positions=[int(pos_of[str(rows[int(i)].row_id)]) for i in query_index],
        evidence=evidence,
    )


def entry_table_from_text_row_table(
    text_row_table: TextRowKeyTable,
    *,
    profile: str = "",
    encoder_fingerprint: str = "",
    source: Optional[Dict[str, Any]] = None,
) -> EntryKeyTable:
    """把既有 :class:`TextRowKeyTable` **只读**包装成 :class:`EntryKeyTable`。

    这是「文本侧复用现有 TextRowKeyTable，不重复造」这条口径的唯一包装点：
    键表字节、行序、``sha256()`` 全部原样透传，落盘格式**零改动**。
    """
    keys = as_float32_matrix(text_row_table.keys)
    return EntryKeyTable(
        keys=keys,
        entry_ids=[str(x) for x in text_row_table.line_ids],
        outputs=[str(x) for x in text_row_table.texts],
        kind="text",
        dim=int(keys.shape[1]),
        encoder_profile=str(profile),
        encoder_fingerprint=str(encoder_fingerprint),
        source=dict(source or {}),
    )


def fingerprint_guard(
    table: EntryKeyTable,
    *,
    expected_key_table_sha256: str = "",
    expected_encoder_fingerprint: str = "",
) -> Dict[str, Any]:
    """**指纹守卫**：键表 SHA256 + 编码器口径指纹，任一不一致立即报错。

    参数
    ----
    table : EntryKeyTable
        待校验的表。
    expected_key_table_sha256 : str
        期望的键表字节指纹（空串 = 不校验该项）。
    expected_encoder_fingerprint : str
        期望的编码器口径指纹（空串 = 不校验该项）。

    返回
    ----
    Dict[str, Any]
        两个指纹的现场值与校验结果。

    异常
    ------
    ValueError
        任一非空期望值与现场值不一致。
    """
    actual_key = table.sha256()
    actual_enc = str(table.encoder_fingerprint)
    if expected_key_table_sha256 and str(expected_key_table_sha256) != actual_key:
        raise ValueError(
            "键表指纹校验失败：期望 "
            f"{str(expected_key_table_sha256)[:16]}...，现场 {actual_key[:16]}...；"
            "候选键表已被改动，拒绝加载"
        )
    if expected_encoder_fingerprint and str(expected_encoder_fingerprint) != actual_enc:
        raise ValueError(
            "编码器口径指纹不一致：期望 "
            f"{str(expected_encoder_fingerprint)[:16]}...，现场 {actual_enc[:16]}...；"
            "拒绝在错配的特征空间上做检索"
        )
    return {
        "key_table_sha256": actual_key,
        "encoder_fingerprint": actual_enc,
        "key_table_sha256_matched": bool(
            not expected_key_table_sha256 or str(expected_key_table_sha256) == actual_key
        ),
        "encoder_fingerprint_matched": bool(
            not expected_encoder_fingerprint or str(expected_encoder_fingerprint) == actual_enc
        ),
    }


def build_vectorizer_for(profile: str, role: str) -> Any:
    """按档 + 角色构造编码器（**复用 ``encoders.build_vectorizer``，不自造**）。"""
    return build_vectorizer(encoder_config_for(str(profile), str(role)))


__all__ = [
    "ENTRY_TABLE_VERSION",
    "ENTRY_L2_NORM_SPEC",
    "NORM_ATOL",
    "KEY_FINGERPRINT_SCHEMA",
    "QA_KNOWN_SPLIT",
    "QA_MAX_CLASSES",
    "QA_MIN_QUESTIONS",
    "QA_TEST_EVERY",
    "QA_TEST_PER_CLASS",
    "QA_UNKNOWN_TRAIN_CAP",
    "QA_SPLIT_SEED",
    "PROFILE_LEXICAL",
    "PROFILE_SEMANTIC",
    "PROFILE_EXPECT_DIM",
    "PAIR_PROFILES",
    "UNKNOWN_ID_PREFIX",
    "EntryKeyTable",
    "EntryStatements",
    "QAEntryBundle",
    "TextEntryBundle",
    "as_float32_matrix",
    "build_key_table_from_statements",
    "build_qa_entry_tables",
    "build_vectorizer_for",
    "canonical_dumps",
    "encode_statements",
    "encoder_config_for",
    "entry_table_from_text_row_table",
    "expect_dim_of",
    "fingerprint_guard",
    "key_table_fingerprint",
    "l2_normalize_rows",
    "load_qa_training_data",
    "profile_names",
    "qa_record_texts",
    "question_is_blank",
    "sha256_bytes",
    "text_entry_table_from_rows",
    "verify_bitwise_lookup",
]
