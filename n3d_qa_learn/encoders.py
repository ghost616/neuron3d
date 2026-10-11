"""可插拔特征生成接口（编码器注册表）+ HF 编码器适配器 + 嵌入缓存。

职责
----
把「文本 -> ``D`` 维特征」这件事从**具体实现**里抽出来，做成一个**可插拔接口 +
注册表**：

* **接口**（结构化协议，与既有向量化器**逐字同形**，故上层评估 / 训练 / 推理
  代码零改动即可换实现）：

  ==================  ==================================================
  成员                语义
  ==================  ==================================================
  ``dim``             连接参数 ``D``（property）
  ``fingerprint()``   口径指纹（含模型 revision 与权重 SHA256）
  ``encode(text)``    单条 -> ``List[float]``，长度 = ``dim``
  ``encode_batch(t)`` 批量 -> ``List[List[float]]``（顺序保持）
  ``encode_matrix(t)``批量 -> ``np.float32[L, D]``（步骤 2 另有该成员）
  ``encode_with_stats``单条 -> :class:`EncodeResult`（含截断统计）
  ==================  ==================================================

* **注册表**（:data:`ENCODER_REGISTRY`）：登记已知模型（``bge-m3`` 为首个 HF
  默认项），登记时**声明 ``expect_dim``**；构造期实测维度与声明不符**立即报错**
  （拒绝静默错配）。

* **实现三家族**：

  * ``local-hash``（:class:`n3d_qa_learn.features.TextVectorizer`）—— 步骤 1 现状口径；
  * ``zh-bag``（:class:`n3d_qa_learn.step2.ZhBagVectorizer`）—— 步骤 2 现状口径；
  * ``bge-m3``（:class:`HFTextEncoder`）—— 通用 HF 编码器适配器，**手工 CLS pooling
    + L2 归一化**，不引 sentence-transformers / FlagEmbedding。

* **嵌入缓存**（:class:`EmbeddingCache`，:data:`DEFAULT_EMB_CACHE_DIR`）：键 =
  ``sha256(canonical_json(schema, model_id, revision, pooling, max_length, normalize,
  weight_sha256, text_sha256))`` —— **8 个分量一个都不能少**（唯一权威实现见
  :meth:`EmbeddingCache.key_for`；本节只是摘要，冲突时以实现为准）：
  ``schema`` = :data:`ENCODER_SCHEMA_VERSION`、``model_id`` = 注册表声明名、
  ``revision`` = 冻结 revision、``pooling`` / ``normalize`` = 编码口径、
  ``max_length`` = 该角色生效截断长度、``weight_sha256`` = 权重文件指纹、
  ``text_sha256`` = 文本 UTF-8 裸字节 SHA256。
  逐条落盘（**现行 = 分片布局** ``emb/<key[:2]>/<key>.bin|.json``）、可跨进程复用。

编码口径
--------
* **CLS pooling + L2 归一化**（``pooling="cls"``，``normalize="l2"``）；
* **题面**（role ``question``）``max_length = 512``；**文本行**（role ``text_line``）
  ``max_length = 8192``；两者是**不同口径**，指纹与缓存键都不同。

可复算
------
* 模型 revision **固定**（不得浮动 ``main``）；
* 权重文件 SHA256 现场计算（带 ``(path, bytes, mtime_ns)`` 记忆化，跨进程复用）；
* 二者**全部折进** :meth:`HFTextEncoder.fingerprint`。

边界处置（与 :mod:`n3d_qa_learn.features` 契约对齐）
--------------------------------------------------
* 空 / 仅空白文本 -> **零向量**（不因 CLS 特殊位而变成非零）；
* 超长 -> **截断**且 ``n_truncated`` 统计可见（``encode_with_stats``）；
* 模型缺失 / 权重不全 / 无网络 -> **可读报错**（:class:`EncoderUnavailableError`）；
* hidden size ≠ 注册声明 -> **构造期报错**（:class:`EncoderDimMismatchError`）。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

#: 编码器口径版本号；任何改变特征取值 / 指纹语义的改动都必须递增。
ENCODER_SCHEMA_VERSION: str = "n3dqa-encoder-v1"

#: 角色：题面（步骤 1 的问题文本 / 指针候选键文本）。
ROLE_QUESTION: str = "question"
#: 角色：文本行（步骤 2 的库行与查询行文本）。
ROLE_TEXT_LINE: str = "text_line"

#: 合法角色集合。
ROLES: Tuple[str, ...] = (ROLE_QUESTION, ROLE_TEXT_LINE)

#: 角色的 ``max_length`` 冻结口径（现场计划口径：题面 512、文本行 8192）。
ROLE_MAX_LENGTH: Dict[str, int] = {ROLE_QUESTION: 512, ROLE_TEXT_LINE: 8192}

#: 嵌入缓存根目录（相对仓库根；跨进程复用）。
DEFAULT_EMB_CACHE_DIR: str = os.path.join("checkpoints", "qa_learn", "_cache", "emb")

#: 嵌入缓存的**目录分片深度**（取缓存键的前 N 个十六进制字符做子目录名）。
#:
#: 取值理由（现场实测，见 ``_verify/emb_cache/`` 的 scale_bench / micro_bench）：
#: 条目全部平铺在根目录时，根目录条目数达到 **67,592**（= 33,795 条目 × 2 文件）后，
#: 单次按名查找的实测成本从 **0.34~0.76 ms**（40 / 100 / 2,000 / 8,000 条目目录）
#: 跃升到 **58~83 ms**（exFAT 目录按名查找是**线性扫描**，与目录条目数强相关）；
#: 一次缓存**命中**因此要付 2 次 stat + 2 次 open ≈ **177~239 ms**，
#: 使「命中的代价」与「重算的代价」同量级 ⇒ 缓存形同虚设。
#: 深度 2 把 33,795 条目摊到 **256** 个子目录（每片约 132 条），查找成本回到亚毫秒级。
#: 该常量**只影响条目的落盘位置**，不影响缓存键、不影响任何向量取值、
#: 也不参与口径指纹（``fingerprint()`` 仍原样委派）。
EMB_CACHE_SHARD_DEPTH: int = 2

#: 默认模型下载目录（相对仓库根）。
DEFAULT_MODEL_DIR: str = os.path.join("models", "bge-m3")

#: 镜像端点（现场实测：``huggingface.co`` 直连超时，镜像可达）。
HF_MIRROR_ENDPOINT: str = "https://hf-mirror.com"

#: 归一化口径名（冻结）。
NORMALIZE_MODE: str = "l2"

#: bge-m3 的固定 revision（**不得用浮动 main**）。
BGE_M3_REVISION: str = "5617a9f61b028005a4858fdac845db406aefb181"

#: 「本地 hash 词袋」注册名（步骤 1 现状口径）。
ENCODER_LOCAL_HASH: str = "local-hash"
#: 「中文词袋」注册名（步骤 2 现状口径）。
ENCODER_ZH_BAG: str = "zh-bag"
#: bge-m3 注册名（首个 HF 默认项）。
ENCODER_BGE_M3: str = "bge-m3"

#: 各角色的**默认实现**：保持现状（确定性词袋），HF 编码器须显式选择。
ROLE_DEFAULT_ENCODER: Dict[str, str] = {
    ROLE_QUESTION: ENCODER_LOCAL_HASH,
    ROLE_TEXT_LINE: ENCODER_ZH_BAG,
}


class EncoderError(RuntimeError):
    """编码器错误基类（报文一律带上下文，禁止静默错配）。"""


class EncoderUnavailableError(EncoderError):
    """模型缺失 / 权重不全 / 依赖缺失 / 无网络（**可读报错**）。"""


class EncoderDimMismatchError(EncoderError):
    """实际维度与注册表声明不一致（**构造期报错**）。"""


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def canonical_dumps(obj: Any) -> bytes:
    """规范化 JSON 字节（排序键 + 紧凑分隔符），用于一切指纹计算。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(blob: bytes) -> str:
    """字节流 SHA256（小写十六进制）。"""
    return hashlib.sha256(blob).hexdigest()


def sha256_text(text: str) -> str:
    """文本 SHA256（UTF-8 编码）。"""
    return sha256_bytes(str(text).encode("utf-8"))


def file_sha256(path: str, chunk: int = 1 << 22) -> str:
    """文件 SHA256（流式，不整文件入内存）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _memo_path(cache_dir: str) -> str:
    """权重 SHA256 记忆化文件路径（放缓存根下，与条目文件同住）。"""
    return os.path.join(str(cache_dir), "_weights_sha256.json")


def weight_sha256_memoized(path: str, cache_dir: str) -> Tuple[str, bool]:
    """权重文件 SHA256（记忆化：以 ``(abspath, bytes, mtime_ns)`` 作有效性判据）。

    参数
    ----
    path : str
        权重文件路径。
    cache_dir : str
        记忆化文件所在目录（不存在则创建）。

    返回
    ----
    Tuple[str, bool]
        ``(sha256, from_memo)``；记忆化只跳过**重复哈希**，不改变判据 ——
    文件大小或 mtime 变化即重算。
    """
    abs_path = os.path.abspath(path)
    stat = os.stat(abs_path)
    key = f"{abs_path}|{int(stat.st_size)}|{int(stat.st_mtime_ns)}"
    memo_file = _memo_path(cache_dir)
    memo: Dict[str, str] = {}
    if os.path.isfile(memo_file):
        try:
            with open(memo_file, "r", encoding="utf-8") as handle:
                memo = dict(json.load(handle))
        except (OSError, ValueError):
            memo = {}
    if key in memo:
        return str(memo[key]), True
    digest = file_sha256(abs_path)
    memo = {key: digest}
    try:
        os.makedirs(os.path.dirname(memo_file) or ".", exist_ok=True)
        with open(memo_file, "w", encoding="utf-8") as handle:
            json.dump(memo, handle, ensure_ascii=False, indent=1)
    except OSError:
        pass  # 记忆化只是加速器；写不进去不影响正确性
    return digest, False


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化；**零向量行保持零向量**（不做除零）。"""
    arr = np.asarray(mat, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    scale = np.where(norm > 0.0, norm, 1.0)
    return (arr / scale).astype(np.float32)


def vector_bytes(vector: Sequence[float]) -> bytes:
    """向量 -> ``float32`` 小端裸字节（一切「逐位比对」的判定载体）。"""
    return np.asarray(list(vector), dtype="<f4").tobytes()


def vector_l2_norm(vector: Sequence[float]) -> float:
    """向量的 L2 范数（零向量判定用，不做除零）。"""
    return float(math.sqrt(sum(float(x) * float(x) for x in vector)))


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderSpec:
    """一个编码器实现的静态描述（不含任何模型实例）。

    属性
    ----
    name : str
        注册名（注册表键）。
    kind : str
        ``"hash"``（确定性词袋家族）或 ``"hf"``（HF 编码器适配器）。
    expect_dim : int
        **注册时声明的维度**：``hf`` 家族 = ``hidden_size``；``hash`` 家族 =
        默认口径维度。``hf`` 家族构造期实测不符**立即报错**。
    pooling : str
        池化口径（``"cls"`` / ``"hash-bag"``）。
    model_id : str
        HF 仓库 id 或本地路径（``hash`` 家族为空串）。
    revision : str
        **固定** revision（提交号）；浮动 ``main`` 一律拒绝。
    weight_file : str
        权重文件名（``hash`` 家族为空串）。
    mirror_endpoint : str
        下载源镜像端点（如实登记，供事后复核「这份权重是哪来的」）。
    interface : str
        该实现暴露的接口成员（自解释，供文档与报告列示）。
    note : str
        口径备注。
    """

    name: str
    kind: str
    expect_dim: int
    pooling: str
    model_id: str = ""
    revision: str = ""
    weight_file: str = ""
    mirror_endpoint: str = ""
    interface: str = "dim/fingerprint/encode/encode_batch/encode_with_stats"
    #: 该实现是否**声明**词元截断统计（``n_tokens`` / ``n_truncated``）。
    #:
    #: ``False`` 的实现（如 ``zh-bag``：n-gram 哈希词袋没有词元上限）其统计项
    #: 恒为 0，因而「超长 -> n_truncated > 0」这条边界对它是**不适用**而不是失败。
    truncation_stats: bool = False
    #: 该实现是否**需要外部模型文件**才能构造（只有它才可能「模型缺失」）。
    requires_model_files: bool = False
    note: str = ""


#: 编码器注册表（**唯一注册点**；``bge-m3`` 为首个 HF 默认项）。
ENCODER_REGISTRY: Dict[str, EncoderSpec] = {
    ENCODER_LOCAL_HASH: EncoderSpec(
        name=ENCODER_LOCAL_HASH,
        kind="hash",
        expect_dim=88,  # DEFAULT_HASH_DIM(80) + len(LENGTH_FEATURE_SPECS)(8)
        pooling="hash-bag",
        interface="dim/fingerprint/encode/encode_batch/encode_with_stats",
        truncation_stats=True,  # features.MAX_TOKENS = 256，有真实截断统计
        requires_model_files=False,
        note="步骤 1 现状口径：blake2b 哈希词袋 + 长度特征 + L2；无外部依赖。",
    ),
    ENCODER_ZH_BAG: EncoderSpec(
        name=ENCODER_ZH_BAG,
        kind="hash",
        expect_dim=192,  # BUCKETS_PER_ORDER(64) * len(N_GRAM_ORDERS)(3)
        pooling="hash-bag",
        interface="dim/fingerprint/encode/encode_batch/encode_matrix/encode_with_stats",
        truncation_stats=False,  # n-gram 哈希词袋无词元上限，不声明截断统计
        requires_model_files=False,
        note="步骤 2 现状口径：n3d_qa 字符 n-gram 哈希词袋 + L2；无外部依赖。",
    ),
    ENCODER_BGE_M3: EncoderSpec(
        name=ENCODER_BGE_M3,
        kind="hf",
        expect_dim=1024,  # 现场核实 hidden_size = 1024
        pooling="cls",    # 1_Pooling/config.json -> pooling_mode_cls_token: true
        model_id="BAAI/bge-m3",
        revision=BGE_M3_REVISION,
        weight_file="pytorch_model.bin",  # 该仓库无 safetensors 变体
        mirror_endpoint=HF_MIRROR_ENDPOINT,
        truncation_stats=True,   # max_length 截断，n_truncated 可观测
        requires_model_files=True,
        note=(
            "通用 HF 编码器适配器；手工 CLS pooling + L2，"
            "不引 sentence-transformers / FlagEmbedding。"
        ),
    ),
}


def list_encoders() -> List[str]:
    """已登记编码器名（升序）。"""
    return sorted(ENCODER_REGISTRY.keys())


def get_spec(name: str) -> EncoderSpec:
    """取注册表条目；未登记时立即报错（报文列出合法集合）。"""
    key = str(name)
    if key not in ENCODER_REGISTRY:
        raise KeyError(f"未登记的编码器 {key!r}；合法集合 = {list_encoders()}")
    return ENCODER_REGISTRY[key]


def register_encoder(spec: EncoderSpec) -> EncoderSpec:
    """登记一个新编码器（幂等：同名同内容直接返回；同名不同内容报错）。"""
    if not isinstance(spec, EncoderSpec):
        raise TypeError(f"register_encoder 需要 EncoderSpec，当前 {type(spec).__name__}")
    if int(spec.expect_dim) < 1:
        raise ValueError(f"EncoderSpec.expect_dim 必须 >= 1，当前 {spec.expect_dim}")
    if str(spec.kind) not in ("hash", "hf"):
        raise ValueError(f"EncoderSpec.kind 仅允许 'hash' / 'hf'，当前 {spec.kind!r}")
    old = ENCODER_REGISTRY.get(spec.name)
    if old is not None and old != spec:
        raise ValueError(
            f"编码器 {spec.name!r} 已登记且内容不同；拒绝覆盖（注册表是唯一注册点）"
        )
    ENCODER_REGISTRY[spec.name] = spec
    return spec


def assert_declared_dim(spec: EncoderSpec, actual_dim: int, *, strict: bool) -> None:
    """把实测维度与注册表声明对账。

    参数
    ----
    spec : EncoderSpec
        注册表条目。
    actual_dim : int
        实测维度。
    strict : bool
        ``True`` -> 不符立即抛 :class:`EncoderDimMismatchError`；
        ``False`` -> 只在报文里登记差异（``hash`` 家族的桶数可配置，故非严格）。

    异常
    ------
    EncoderDimMismatchError
        ``strict`` 为真且维度不符。
    """
    if int(actual_dim) == int(spec.expect_dim):
        return
    message = (
        f"编码器 {spec.name!r} 的实测维度 {actual_dim} 与注册表声明 "
        f"expect_dim={spec.expect_dim} 不一致（拒绝静默错配）"
    )
    if strict:
        raise EncoderDimMismatchError(message)


# ---------------------------------------------------------------------------
# 编码结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodeResult:
    """单条文本的编码结果（含截断与缓存可观测性）。

    属性
    ----
    vector : List[float]
        长度 = ``dim`` 的 L2 归一化向量；空 / 仅空白文本为零向量。
    n_tokens : int
        参与编码的**词元总数**（截断前，含特殊位）。
    n_truncated : int
        因 ``max_length`` 被丢弃的词元数（``>= 0``）。
    cached : bool
        本次是否由**落盘缓存**命中（跨进程复用）。
    """

    vector: List[float]
    n_tokens: int
    n_truncated: int = 0
    cached: bool = False


# ---------------------------------------------------------------------------
# 嵌入缓存
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """逐条文本的嵌入缓存（落盘、可跨进程复用）。

    目录布局（:data:`DEFAULT_EMB_CACHE_DIR`）**现行 = 分片**（:data:`EMB_CACHE_SHARD_DEPTH`
    = 2）：::

        emb/<key[:2]>/<key>.bin     float32 小端裸字节（长度 = dim * 4）
        emb/<key[:2]>/<key>.json    条目元数据（含 model/revision/pooling/max_length/
                                    text_sha256/dim/sha256/n_tokens/n_truncated/created_utc）
        emb/_weights_sha256.json    权重 SHA256 记忆化（加速器，不参与口径）

    目录布局（**历史 = 扁平**，仍可**只读**复用）::

        emb/<key>.bin / emb/<key>.json

    **为何要分片（性能，取值零变化）**：根目录平铺时条目数达 **67,592**
    （= 33,795 条目 × 2 文件）后，单次按名查找实测 **58~83 ms**（40 / 100 / 2,000 /
    8,000 条目目录只有 0.34~0.76 ms）—— exFAT 的按名查找是**线性扫描**，成本随
    目录条目数增长；于是一次**命中**要付 2 stat + 2 open ≈ **177~239 ms**，
    与重算同量级 ⇒ 缓存给不出速度。分片后每片约 132 条，查找回到亚毫秒级。
    分片**只改条目落盘位置**：缓存键、条目字节、向量取值、口径指纹全部不变。

    键（``key = sha256(canonical_json(...))``）由下列字段组成：``schema`` /
    ``model_id`` / ``revision`` / ``pooling`` / ``max_length`` / ``normalize`` /
    ``weight_sha256`` / ``text_sha256``。

    关键不变量
    ----------
    * 同一 ``(口径, 文本)`` 的键唯一；不同口径（如 512 / 8192）**不共用**条目；
    * 条目以**裸 float32 字节**存放，故「与落盘缓存逐位比对」是字节级判定；
    * 缓存只影响**速度**，不参与任何口径指纹；
    * 分片与扁平两种布局**读路径都认**（扁平为只读回退，读到即**尽力迁移**到分片），
      写路径**一律写分片** —— 故旧条目不会因布局变更而失效（不产生重算）。

    进程内 memo（性能，取值不变）
    ----------------------------
    :meth:`get` 命中一次后把 ``(向量, 元数据)`` 留在**本实例**的 memo 里 ——
    训练循环每个 batch 都要取同一批文本，若每次都走「两次 ``open`` + SHA256 校验」
    会让磁盘 I/O 盖过算力（``D=1024`` 时尤其明显）。memo **只改速度**：memo 与磁盘
    返回的是同一份字节解出的同一数组；:meth:`read_bytes` 仍然**只读磁盘**，
    以保证「与落盘缓存逐位比对」这条验证口径不受 memo 影响。
    """

    def __init__(
        self,
        root: str = DEFAULT_EMB_CACHE_DIR,
        shard_depth: int = EMB_CACHE_SHARD_DEPTH,
        auto_migrate: bool = False,
    ) -> None:
        self.root = str(root)
        if int(shard_depth) < 0:
            raise ValueError(f"shard_depth 必须 >= 0，当前 {shard_depth}")
        if int(shard_depth) > 8:
            raise ValueError(
                f"shard_depth 过大（{shard_depth}）：子目录名取自缓存键的十六进制前缀，"
                "最多 8 位以免缓存目录层级失控"
            )
        self.shard_depth = int(shard_depth)
        #: 读到**历史扁平**条目时是否顺手迁移到分片。
        #: 默认 ``False`` —— 现场实测「读 + 改名」≈ **0.7 s/键**（改名会**改动那个大目录**，
        #: 使它按名查找的线性扫描缓存反复失效），会让一次全量 probe 从分钟级涨到小时级；
        #: 一次性维护请走显式的 :meth:`migrate_flat_keys`（每个键只做一次文件操作）。
        self.auto_migrate = bool(auto_migrate)
        self._memo: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
        self._memo_hits = 0
        self._disk_hits = 0
        #: 历史扁平布局下仍存在的键（**一次 ``os.listdir`` 惰性建立**）。
        #: 为空时读路径完全不走扁平回退 —— 迁移排空后不再付扁平目录的线性扫描成本。
        self._flat_keys: Optional[Set[str]] = None
        self._flat_hits = 0
        self._migrated = 0
        self._migrate_failed = 0

    def key_for(
        self,
        *,
        model_id: str,
        revision: str,
        pooling: str,
        max_length: int,
        normalize: str,
        weight_sha256: str,
        text: str,
    ) -> str:
        """按冻结字段序列算出缓存键（64 位十六进制）。

        构造期校验（修复 N3）：``max_length >= 1``、``model_id`` / ``pooling`` /
        ``normalize`` 非空 —— 键字段缺项会让不同口径**共用**同一条目，属静默错配，
        故直接报错而不是照算。``revision`` **允许为空**：``hash`` 家族没有 revision
        概念；而 ``hf`` 家族在 ``HFTextEncoder.__init__`` 已拒绝空 revision，
        故由它产生的键必然带 revision。
        """
        if int(max_length) < 1:
            raise ValueError(f"缓存键要求 max_length >= 1，当前 {max_length}")
        for field, value in (("model_id", model_id), ("pooling", pooling),
                             ("normalize", normalize)):
            if not str(value):
                raise ValueError(
                    f"缓存键要求 {field} 非空（空值会让不同口径共用同一条目）；当前为空"
                )
        payload = {
            "schema": ENCODER_SCHEMA_VERSION,
            "model_id": str(model_id),
            "revision": str(revision),
            "pooling": str(pooling),
            "max_length": int(max_length),
            "normalize": str(normalize),
            "weight_sha256": str(weight_sha256),
            "text_sha256": sha256_text(text),
        }
        return sha256_bytes(canonical_dumps(payload))

    # -- 路径（分片 = 写路径 / 首查路径；扁平 = 只读回退）-------------------
    def _shard_dir(self, key: str) -> str:
        """条目所属的分片子目录（``shard_depth=0`` 时退化为根目录 = 历史扁平口径）。"""
        if self.shard_depth <= 0:
            return self.root
        return os.path.join(self.root, str(key)[: self.shard_depth])

    def _entry_paths(self, key: str) -> Tuple[str, str]:
        """``(向量文件, 元数据文件)`` 的**现行（分片）**路径。"""
        shard = self._shard_dir(key)
        return (
            os.path.join(shard, f"{key}.bin"),
            os.path.join(shard, f"{key}.json"),
        )

    def _flat_entry_paths(self, key: str) -> Tuple[str, str]:
        """``(向量文件, 元数据文件)`` 的**历史（扁平）**路径（只读回退用）。"""
        return (
            os.path.join(self.root, f"{key}.bin"),
            os.path.join(self.root, f"{key}.json"),
        )

    def flat_keys(self) -> Set[str]:
        """历史扁平布局下仍存在的键集合（惰性一次 ``os.listdir``；取证用）。

        只是**加速器**：即使该集合过期，读路径也只会多算一次或回退实算，
        不会返回错值。
        """
        if self._flat_keys is None:
            try:
                names = os.listdir(self.root)
            except OSError:
                names = []
            self._flat_keys = {n[: -len(".bin")] for n in names if n.endswith(".bin")}
        return self._flat_keys

    def _read_pair(
        self, bin_path: str, meta_path: str
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """读一对 ``(bin, json)`` 并做完整性校验；任一步不满足返回 ``None``。

        校验口径（与改动前**逐字相同**）：① 两文件都存在；② 元数据可解析为 JSON 对象；
        ③ ``dim >= 1`` 且 ``len(blob) == dim * 4``；④ ``sha256(blob)`` 等于元数据
        ``sha256``。任一不满足 = 损坏 / 失配 ⇒ 返回 ``None``（**不抛异常**），
        调用方回退实算。

        **性能口径**：① 不用 ``os.path.isfile`` 预检查 —— 直接 ``open`` 并捕获
        ``OSError``（含 ``FileNotFoundError``）。在扁平大目录上，一次「按名查找」的实测
        成本是 **45 ms**（存在）~ **130 ms**（不存在），故把 4 次查找压到 2 次
        是这一步唯一的性能取舍；语义与「先 isfile 再 open」**逐字等价**（不存在的分支
        一样返回 ``None``，不抛异常）。
        """
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            with open(bin_path, "rb") as handle:
                blob = handle.read()
        except (OSError, ValueError):
            return None
        # 元数据必须是 **JSON 对象**：``[]`` / ``42`` / ``true`` / ``1.5`` 都是「合法 JSON
        # 但不是对象」，``dict(...)`` 会抛 ``TypeError`` —— 而契约要求损坏条目
        # **返回 None 而不是抛出**（历史遗留缺陷，本轮修复；12 类损坏逐例对照中其余
        # 12 类的行为与改动前逐字相同）。
        if not isinstance(raw, dict):
            return None
        meta = dict(raw)
        dim = int(meta.get("dim", 0))
        if dim < 1 or len(blob) != dim * 4:
            return None
        if sha256_bytes(blob) != str(meta.get("sha256", "")):
            return None
        vec = np.frombuffer(blob, dtype="<f4").astype(np.float32)
        return vec, meta

    def _migrate_to_shard(self, key: str, bin_path: str, meta_path: str) -> bool:
        """把读到的一条**扁平**条目尽力迁移到**分片**位置（性能；取值不变）。

        顺序 = **先 bin 后 json**（与 :meth:`put` 的提交序一致）；任一步失败只记数，
        不抛异常 —— 迁移失败最多让下一次再读一次扁平路径，绝不会改变取值。
        """
        if self.shard_depth <= 0:
            return False
        target_bin, target_meta = self._entry_paths(key)
        try:
            os.makedirs(os.path.dirname(target_bin), exist_ok=True)
        except OSError:
            self._migrate_failed += 1
            return False
        ok = True
        for src, dst in ((bin_path, target_bin), (meta_path, target_meta)):
            if os.path.isfile(dst):
                # 目标已存在（他进程已迁 / 已有更新的写入）：删掉扁平副本即可
                try:
                    os.remove(src)
                except OSError:
                    self._migrate_failed += 1
                continue
            try:
                os.replace(src, dst)
            except OSError:
                ok = False
                self._migrate_failed += 1
        if key in self.flat_keys():
            self._flat_keys.discard(key)  # type: ignore[union-attr]
        if ok:
            self._migrated += 1
        return ok

    # -- 读 / 写 -----------------------------------------------------------
    def get(self, key: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """读条目（**先查进程内 memo，再查分片，最后只读回退扁平**）；损坏即 ``None``。

        memo 与磁盘返回的是同一份字节解出的同一数组，故取值不受影响；
        ``read_bytes`` 不经过 memo，保证「与落盘缓存逐位比对」这条口径独立。
        布局回退**只影响从哪里读**：校验口径、返回的数组与元数据与改动前逐位相同。
        """
        hit = self._memo.get(key)
        if hit is not None:
            self._memo_hits += 1
            return hit
        bin_path, meta_path = self._entry_paths(key)
        legacy = False
        read = self._read_pair(bin_path, meta_path)
        if read is None and self.shard_depth > 0 and key in self.flat_keys():
            legacy = True
            bin_path, meta_path = self._flat_entry_paths(key)
            read = self._read_pair(bin_path, meta_path)
        if read is None:
            return None
        vec, meta = read
        self._memo[key] = (vec, meta)
        self._disk_hits += 1
        if legacy:
            self._flat_hits += 1
            if self.auto_migrate:
                self._migrate_to_shard(key, bin_path, meta_path)
        return vec, meta

    def put(
        self,
        key: str,
        vector: Sequence[float],
        meta: Dict[str, Any],
    ) -> str:
        """写条目（**一律写分片位置**），返回向量文件路径。

        写入是**先写向量再写元数据**：元数据充当提交标记，故不会读到半成品。
        tmp 与目标同目录（同卷）以保证 ``os.replace`` 的原子性。
        """
        bin_path, meta_path = self._entry_paths(key)
        shard = os.path.dirname(bin_path)
        os.makedirs(shard, exist_ok=True)
        blob = np.asarray(list(vector), dtype="<f4").tobytes()
        record = dict(meta)
        record.update(
            {
                "key": str(key),
                "dim": int(len(vector)),
                "bytes": int(len(blob)),
                "sha256": sha256_bytes(blob),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        tmp_bin = bin_path + ".tmp"
        with open(tmp_bin, "wb") as handle:
            handle.write(blob)
        os.replace(tmp_bin, bin_path)
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp_meta, meta_path)
        # 同步清理同名扁平副本（若存在）：避免扁平目录长期不排空
        if self.shard_depth > 0 and key in self.flat_keys():
            flat_bin, flat_meta = self._flat_entry_paths(key)
            for path in (flat_bin, flat_meta):
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            self._flat_keys.discard(key)  # type: ignore[union-attr]
        # 写入后同步 memo（同进程内的后续读取不必再落磁盘）
        self._memo[key] = (np.asarray(list(vector), dtype=np.float32), record)
        return bin_path

    def read_bytes(self, key: str) -> Optional[bytes]:
        """直接读条目的**裸字节**（「与落盘缓存逐位比对」用；**不经过 memo**）。

        分片位置优先，缺失时只读回退历史扁平位置（同样用 ``open`` + ``OSError``
        而不是 ``isfile`` 预检查，见 :meth:`_read_pair`）。

        **口径边界（不得误读）**：本方法**只看文件在不在**，**不校验内容** ——
        零字节 / 截断 / ``sha256`` 失配的 ``.bin`` 同样会被**原样读出**
        （这正是「逐位比对」这条口径要的语义：调用方拿裸字节自己去比）。
        只有「两处都没有该 ``.bin``」才返回 ``None``。内容校验只发生在
        :meth:`get` / :meth:`_read_pair`（损坏 ⇒ 返回 ``None``）。
        """
        for path in (self._entry_paths(key)[0], self._flat_entry_paths(key)[0]):
            try:
                with open(path, "rb") as handle:
                    return handle.read()
            except OSError:
                continue
        return None

    def read_meta(self, key: str) -> Optional[Dict[str, Any]]:
        """读条目的**元数据**（分片位置优先，缺失时只读回退历史扁平位置）。

        与 :meth:`read_bytes` 的区别：这里**解析 JSON 并要求「必须是对象」**
        （与 :meth:`_read_pair` 同口径），但**不校验向量内容** ——
        供取证侧回答「这条条目是谁写的、什么时候写的」。

        返回
        ----
        Optional[Dict[str, Any]]
            元数据字典；两处都不存在 / 不可解析 / 不是 JSON 对象时返回 ``None``
            （**不抛异常**）。

        存在理由（布局变更的连带面）：写路径自本轮起一律写分片位置，任何
        「按历史扁平布局直接拼 ``<cache_dir>/<key>.json``」的读取方在分片条目上都会
        **永远读空**；本方法把这个读取口径**收敛到 ``EmbeddingCache`` 内部**，
        调用方（如 ``encoders_run`` 的 G2 取证）不必再知道布局。
        """
        for path in (self._entry_paths(key)[1], self._flat_entry_paths(key)[1]):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    raw = json.load(handle)
            except (OSError, ValueError):
                continue
            return dict(raw) if isinstance(raw, dict) else None
        return None

    def migrate_flat_keys(
        self,
        keys: Optional[Iterable[str]] = None,
        *,
        mode: str = "rename",
        sink: Optional[Any] = None,
        chunk_report: int = 2000,
    ) -> Dict[str, int]:
        """**批量**把历史扁平条目尽力迁移到分片位置（一次性维护操作；只改名）。

        为什么要它：每一处「扁平目录里的按名查找」实测要 **45 ms**（存在）/ **130 ms**
        （不存在），而分片目录里只要亚毫秒；``get`` 的**惰性迁移**要先把条目读一遍
        （2 次查找）再改名（2 次查找），批量预迁移**只做改名**（2 次查找），
        对「全量工作负载」这类一次性成本约为惰性路径的一半。

        参数
        ----
        keys : Optional[Iterable[str]]
            要迁移的键；``None`` = 迁移当前已知的全部扁面条目（:meth:`flat_keys`）。
        mode : str
            ``"rename"``（默认，``os.replace``，扁平目录**会被改动** —— 现场实测
            改名后该目录的按名查找变慢，整批可达 **0.34 s/键**）或
            ``"copy"``（``open``+写，**只读**扁平目录 —— 实测 **0.11 s/键**，
            更快，但扁平副本会留下，成为不再被读取的死重）。
        sink : Optional[Any]
            可选的回调 ``sink(统计字典)``，每 ``chunk_report`` 条报一次进度。
        chunk_report : int
            进度上报间隔。

        返回
        ----
        Dict[str, int]
            ``{"attempted", "migrated", "failed"}``；**只搬运字节、不解析、不校验**，
            故取值不可能被改变（读出时照旧走 :meth:`_read_pair` 的完整校验）。

        **计数口径（逐分支完整对齐，避免歧义）**
        ========================================

        * ``attempted`` —— 以**键**为单位：进入循环的每个键都 +1（无论成功与否）。
        * ``migrated`` —— 以**键**为单位：该键 **≥1 个文件**搬运成功即 +1。
        * ``failed`` —— 以**文件**为单位，**逐个文件独立计数**：``.bin`` 与 ``.json``
          各自失败各记 1。因此：
          - 一个键的两个文件都缺（或都搬运失败）⇒ ``failed += 2``，
            故 ``failed`` **可能大于** ``attempted``；
          - 一个键只失败一半 ⇒ ``migrated += 1`` **且** ``failed += 1``（同一键同时计入两者）。
        * 目标位置**已存在**时：``mode="rename"`` / ``"copy"`` 都是**直接覆盖**
          （``os.replace`` / ``open(wb)``），**不计 failed**；这与 :meth:`_migrate_to_shard`
          （``auto_migrate=True`` 的惰性路径）不同 —— 后者遇到目标已存在会改为
          ``os.remove(src)`` 删除扁平副本（删除失败才 +1 到实例级的
          ``_migrate_failed`` 计数器，**不进本方法的返回值**）。
        * ``shard_depth == 0`` 时整个循环直接 ``break``：三个计数恒为 0（安全 no-op）。
        """
        if str(mode) not in ("rename", "copy"):
            raise ValueError(f"mode 仅允许 'rename' / 'copy'，当前 {mode!r}")
        todo = list(self.flat_keys() if keys is None else keys)
        attempted = 0
        migrated = 0
        failed = 0
        for index, key in enumerate(todo, start=1):
            if self.shard_depth <= 0:
                break
            bin_path, meta_path = self._flat_entry_paths(key)
            target_bin, target_meta = self._entry_paths(key)
            try:
                os.makedirs(os.path.dirname(target_bin), exist_ok=True)
            except OSError:
                failed += 1
                continue
            moved_any = False
            for src, dst in ((bin_path, target_bin), (meta_path, target_meta)):
                try:
                    if str(mode) == "copy":
                        with open(src, "rb") as handle:
                            blob = handle.read()
                        with open(dst, "wb") as handle:
                            handle.write(blob)
                    else:
                        os.replace(src, dst)
                    moved_any = True
                except OSError:
                    failed += 1
            attempted += 1
            if moved_any:
                migrated += 1
                # ``_flat_keys`` 是**惰性**索引（初值 None，只有 flat_keys() 会建立）：
                # 显式传 keys=[...] 时它可能还没建立，此时**不得**假定它存在
                # （本轮 P0 缺陷：AttributeError: 'NoneType' object has no attribute 'discard'）。
                if self._flat_keys is not None:
                    self._flat_keys.discard(key)
                self._migrated += 1
            else:
                self._migrate_failed += 1
            if sink is not None and chunk_report > 0 and index % int(chunk_report) == 0:
                sink({"attempted": attempted, "migrated": migrated, "failed": failed,
                      "index": index, "total": len(todo)})
        return {"attempted": attempted, "migrated": migrated, "failed": failed}

    def memo_stats(self) -> Dict[str, Any]:
        """进程内 memo / 布局迁移统计（取证用：证明这些机制只加速、不改值）。"""
        return {
            "memo_entries": int(len(self._memo)),
            "memo_hits": int(self._memo_hits),
            "disk_hits": int(self._disk_hits),
            "flat_hits": int(self._flat_hits),
            "migrated": int(self._migrated),
            "migrate_failed": int(self._migrate_failed),
            "shard_depth": int(self.shard_depth),
            "note": "memo 只加速；read_bytes 仍只读磁盘，逐位比对口径不受影响",
        }

    def stats(self) -> Dict[str, Any]:
        """缓存统计：条目数 / 总字节 / 根目录 / 两种布局各自的条目数。

        分片后逐片统计（每片目录很小），故统计本身也从「秒级」回到「毫秒级」。
        """
        n_entries = 0
        total = 0
        if not os.path.isdir(self.root):
            return {"root": self.root, "n_entries": 0, "bytes": 0,
                    "n_sharded_entries": 0, "n_flat_entries": 0,
                    "shard_depth": int(self.shard_depth)}
        dirs = [self.root]
        if self.shard_depth > 0:
            try:
                dirs += [
                    os.path.join(self.root, name)
                    for name in os.listdir(self.root)
                    if os.path.isdir(os.path.join(self.root, name))
                ]
            except OSError:
                pass
        n_sharded = 0
        n_flat = 0
        for index, directory in enumerate(dirs):
            try:
                bins = [n for n in os.listdir(directory) if n.endswith(".bin")]
            except OSError:
                continue
            n_entries += len(bins)
            total += sum(os.path.getsize(os.path.join(directory, n)) for n in bins)
            if index == 0:
                n_flat += len(bins)
            else:
                n_sharded += len(bins)
        return {"root": self.root, "n_entries": int(n_entries), "bytes": int(total),
                "n_sharded_entries": int(n_sharded), "n_flat_entries": int(n_flat),
                "shard_depth": int(self.shard_depth)}


# ---------------------------------------------------------------------------
# HF 编码器适配器
# ---------------------------------------------------------------------------


class HFTextEncoder:
    """通用 HF 编码器适配器：传模型名或本地路径即可用（CLS pooling + L2）。

    参数
    ----
    spec : EncoderSpec
        注册表条目（``kind`` 必须为 ``"hf"``）。
    source : str
        模型来源：本地目录（优先）或 HF 仓库 id；空串时回退 ``spec.model_id``。
    revision : str
        固定 revision；空串时回退 ``spec.revision``（浮动 ``main`` 一律拒绝）。
    role : str
        ``"question"``（``max_length = 512``）或 ``"text_line"``（``8192``）。
    max_length : int
        显式覆盖截断长度（``0`` = 用角色冻结口径）。
    expect_dim : int
        显式覆盖注册声明维度（``0`` = 用 ``spec.expect_dim``）；**仅供维度错配自检**。
    cache_dir : str
        嵌入缓存根目录。
    use_cache : bool
        是否启用落盘缓存。
    batch_size : int
        未命中缓存时的前向批大小。
    device : str
        计算设备（``cpu`` 优先）。
    local_files_only : bool
        只允许本地文件（无网络场景；缺失即**可读报错**）。
    verify_weights : bool
        是否计算权重 SHA256（关闭时指纹记 ``unverified``，不静默冒充已核实）。

    关键不变量（构造期）
    ------------------
    * ``config.hidden_size`` 必须等于声明维度，否则 :class:`EncoderDimMismatchError`；
    * ``pooling`` 必须为 ``"cls"``，且本地口径文件的 ``pooling_mode_cls_token``
      必须为 ``True``；
    * 维度恒为 ``hidden_size``；``fingerprint()`` 含 revision 与权重 SHA256。
    """

    def __init__(
        self,
        spec: EncoderSpec,
        source: str = "",
        *,
        revision: str = "",
        role: str = ROLE_QUESTION,
        max_length: int = 0,
        expect_dim: int = 0,
        cache_dir: str = DEFAULT_EMB_CACHE_DIR,
        use_cache: bool = True,
        batch_size: int = 8,
        device: str = "cpu",
        local_files_only: bool = False,
        verify_weights: bool = True,
    ) -> None:
        if not isinstance(spec, EncoderSpec):
            raise TypeError(f"HFTextEncoder 需要 EncoderSpec，当前 {type(spec).__name__}")
        if str(spec.kind) != "hf":
            raise EncoderError(
                f"HFTextEncoder 只接受 kind='hf' 的条目，当前 {spec.name!r} 的 kind="
                f"{spec.kind!r}（hash 家族请走 build_vectorizer 的分派）"
            )
        if str(role) not in ROLES:
            raise ValueError(f"role 仅允许 {list(ROLES)}，当前 {role!r}")
        self.spec = spec
        self.role = str(role)
        self.max_length = int(max_length) if int(max_length) > 0 else int(
            ROLE_MAX_LENGTH[self.role]
        )
        if self.max_length < 8:
            raise ValueError(f"max_length 必须 >= 8，当前 {self.max_length}")
        self.cache_dir = str(cache_dir)
        self.use_cache = bool(use_cache)
        self.batch_size = max(1, int(batch_size))
        self.device = str(device)
        self.local_files_only = bool(local_files_only)
        self.verify_weights = bool(verify_weights)
        self.expect_dim = int(expect_dim) if int(expect_dim) > 0 else int(spec.expect_dim)

        self.source = str(source) if str(source) else str(spec.model_id)
        if not self.source:
            raise EncoderUnavailableError(
                f"编码器 {spec.name!r} 既未给出 source 也未在注册表声明 model_id"
            )
        self.revision = str(revision) if str(revision) else str(spec.revision)
        if not self.revision:
            raise EncoderError(
                f"编码器 {spec.name!r} 未给出固定 revision；浮动版本会破坏可复算性，拒绝构造"
            )
        self.cache = EmbeddingCache(self.cache_dir)
        self._config_obj: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._weight_path = ""
        self._weight_sha256 = ""
        self._weight_from_memo = False
        self._hidden_size = 0
        self._max_position_embeddings = 0
        self._pooling_declared: Dict[str, Any] = {}
        self._hits = 0
        self._misses = 0
        # 构造期即加载 + 校验（维度错配必须在构造期报错，不留到首次前向）
        self._load()

    # -- 构造期加载 -------------------------------------------------------
    def _missing_local_files(self, path: str) -> List[str]:
        """本地目录下缺失的必需文件（可读报错用）。"""
        required = ["config.json"]
        if not (
            os.path.isfile(os.path.join(path, "tokenizer.json"))
            or os.path.isfile(os.path.join(path, "sentencepiece.bpe.model"))
        ):
            required.append("tokenizer.json|sentencepiece.bpe.model")
        if not any(
            os.path.isfile(os.path.join(path, name))
            for name in (str(self.spec.weight_file) or "pytorch_model.bin",
                         "model.safetensors", "pytorch_model.bin")
        ):
            required.append(str(self.spec.weight_file) or "pytorch_model.bin")
        return [name for name in required if not os.path.isfile(os.path.join(path, name))]

    def _load(self) -> None:
        """加载 config / tokenizer / 权重，并在构造期完成全部口径校验。"""
        is_local = os.path.isdir(self.source)
        if self.local_files_only and not is_local:
            raise EncoderUnavailableError(
                f"local_files_only=True 但 source={self.source!r} 不是本地目录；"
                f"请先下载到 {DEFAULT_MODEL_DIR!r} 或去掉 local_files_only"
            )
        if not is_local and os.sep in self.source and not os.path.exists(self.source):
            raise EncoderUnavailableError(
                f"本地模型目录不存在：{self.source!r}；"
                f"请先执行 encoders_run fetch（目标 {DEFAULT_MODEL_DIR!r}）"
            )
        if is_local:
            missing = self._missing_local_files(self.source)
            if missing:
                raise EncoderUnavailableError(
                    f"本地模型目录 {self.source!r} 缺少必需文件 {missing}；"
                    f"实际内容 = {sorted(os.listdir(self.source))[:20]}"
                )
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - 依赖缺失必须可读报错
            raise EncoderUnavailableError(
                "缺少 transformers 依赖：请执行 "
                "`.venv\\Scripts\\python.exe -m pip install transformers`"
                f"（依赖清单见 n3d_qa_learn/requirements-qa.txt）；原始错误：{exc}"
            ) from exc

        try:
            cfg = AutoConfig.from_pretrained(
                self.source, revision=self.revision,
                local_files_only=self.local_files_only, trust_remote_code=False,
            )
        except Exception as exc:  # noqa: BLE001 - 网络 / 缓存缺失必须可读报错
            raise EncoderUnavailableError(
                f"加载模型 config 失败：source={self.source!r} revision={self.revision!r}；"
                f"若为镜像下载请设置 HF_ENDPOINT={HF_MIRROR_ENDPOINT}。"
                f"原始错误：{type(exc).__name__}: {exc}"
            ) from exc
        self._config_obj = cfg
        self._hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
        if self._hidden_size < 1:
            raise EncoderUnavailableError(
                f"模型 config 未给出 hidden_size：model_type={getattr(cfg, 'model_type', '?')}"
            )
        self._max_position_embeddings = int(getattr(cfg, "max_position_embeddings", 0) or 0)
        # ---- 构造期维度校验（拒绝静默错配）----
        assert_declared_dim(self.spec, self._hidden_size, strict=False)
        if self._hidden_size != self.expect_dim:
            raise EncoderDimMismatchError(
                f"编码器 {self.spec.name!r} 的 hidden_size={self._hidden_size} 与声明维度 "
                f"expect_dim={self.expect_dim} 不一致（构造期拒绝，防止静默错配特征空间）"
            )
        if str(self.spec.pooling) != "cls":
            raise EncoderError(
                f"本适配器只实现 CLS pooling，注册表声明 pooling={self.spec.pooling!r}"
            )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.source, revision=self.revision,
                local_files_only=self.local_files_only, trust_remote_code=False,
            )
            model = AutoModel.from_pretrained(
                self.source, revision=self.revision,
                local_files_only=self.local_files_only, trust_remote_code=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise EncoderUnavailableError(
                f"加载 tokenizer / 权重失败：source={self.source!r} "
                f"revision={self.revision!r}；原始错误：{type(exc).__name__}: {exc}"
            ) from exc
        model.eval()
        self._model = model.to(self.device)

        self._pooling_declared = self._read_pooling_config()
        if self._pooling_declared:
            if not bool(self._pooling_declared.get("pooling_mode_cls_token", False)):
                raise EncoderError(
                    "1_Pooling/config.json 未声明 CLS pooling（"
                    f"pooling_mode_cls_token={self._pooling_declared.get('pooling_mode_cls_token')!r}）；"
                    "本适配器只做手工 CLS pooling，拒绝按未声明口径出数"
                )
        self._resolve_weight_sha256()

    def _read_pooling_config(self) -> Dict[str, Any]:
        """读本地 ``1_Pooling/config.json``（**只读**，用于现场核实 CLS 口径）。"""
        if not os.path.isdir(self.source):
            return {}
        path = os.path.join(self.source, "1_Pooling", "config.json")
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return dict(json.load(handle))
        except (OSError, ValueError):
            return {}

    def _resolve_weight_sha256(self) -> None:
        """定位权重文件并计算 SHA256（可关闭；关闭时指纹显式记 ``unverified``）。"""
        if not self.verify_weights:
            self._weight_sha256 = "unverified(verify_weights=False)"
            return
        candidates = [str(self.spec.weight_file) or "pytorch_model.bin", "model.safetensors",
                      "pytorch_model.bin"]
        if os.path.isdir(self.source):
            for name in candidates:
                path = os.path.join(self.source, name)
                if os.path.isfile(path):
                    self._weight_path = path
                    break
            if not self._weight_path:
                raise EncoderUnavailableError(
                    f"本地目录 {self.source!r} 下未找到权重文件（候选 {candidates}）；"
                    "拒绝在没有权重指纹的情况下出数"
                )
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:  # pragma: no cover
                raise EncoderUnavailableError(
                    f"缺少 huggingface_hub 依赖（无法定位权重文件）：{exc}"
                ) from exc
            name = str(self.spec.weight_file) or "pytorch_model.bin"
            try:
                self._weight_path = str(
                    hf_hub_download(
                        repo_id=self.source, filename=name, revision=self.revision,
                        local_files_only=self.local_files_only,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                raise EncoderUnavailableError(
                    f"定位权重文件 {name!r} 失败：repo={self.source!r} "
                    f"revision={self.revision!r}；原始错误：{type(exc).__name__}: {exc}"
                ) from exc
        self._weight_sha256, self._weight_from_memo = weight_sha256_memoized(
            self._weight_path, self.cache_dir
        )

    # -- 接口：dim / fingerprint -----------------------------------------
    @property
    def dim(self) -> int:
        """连接参数 ``D = hidden_size``。"""
        return int(self._hidden_size)

    @property
    def hidden_size(self) -> int:
        """现场读到的 ``hidden_size``（构造期已校验等于声明维度）。"""
        return int(self._hidden_size)

    @property
    def max_position_embeddings(self) -> int:
        """现场读到的 ``max_position_embeddings``。"""
        return int(self._max_position_embeddings)

    @property
    def pooling(self) -> str:
        """池化口径（本适配器恒为 ``"cls"``）。"""
        return "cls"

    @property
    def weight_sha256(self) -> str:
        """权重文件 SHA256（未核实状态下显式记 ``unverified``）。"""
        return str(self._weight_sha256)

    def declaration(self) -> Dict[str, Any]:
        """口径声明（写进产物 meta / 报告；``to_dict`` 与指纹的唯一来源）。

        **只含编码口径**（模型 / 修订 / 池化 / 长度 / 归一化 / 权重指纹 / 维度）——
        缓存开关与缓存目录**不在其中**：它们是速度设置，不是特征口径，放进指纹会让
        「换缓存目录 = 换特征空间」这种荒谬结论成立。
        """
        return {
            "kind": "hf",
            "schema_version": str(ENCODER_SCHEMA_VERSION),
            "name": str(self.spec.name),
            "model_id": str(self.spec.model_id),
            "source": str(self.source),
            "revision": str(self.revision),
            "pooling": str(self.pooling),
            "normalize": str(NORMALIZE_MODE),
            "max_length": int(self.max_length),
            "role": str(self.role),
            "hidden_size": int(self._hidden_size),
            "max_position_embeddings": int(self._max_position_embeddings),
            "weight_file": os.path.basename(self._weight_path) if self._weight_path else "",
            "weight_sha256": str(self._weight_sha256),
            "mirror_endpoint": str(self.spec.mirror_endpoint),
            "dim": int(self.dim),
        }

    def fingerprint(self) -> str:
        """口径指纹：SHA256 over 规范化声明（含**固定 revision** 与**权重 SHA256**）。"""
        return sha256_bytes(canonical_dumps(self.declaration()))

    @property
    def config(self) -> "EncoderConfigView":
        """与 ``VectorizerConfig`` 取用面同形的口径视图（供 ``build_registry`` 取 ``dim``）。"""
        return EncoderConfigView(self.declaration())

    def describe(self) -> Dict[str, Any]:
        """自描述（供报告与 README 现场枚举，字段与 :meth:`declaration` 同源）。"""
        out = self.declaration()
        out["interface"] = str(self.spec.interface)
        out["weight_path"] = str(self._weight_path)
        out["weight_sha256_from_memo"] = bool(self._weight_from_memo)
        out["use_cache"] = bool(self.use_cache)
        out["cache_dir"] = str(self.cache_dir)
        out["pooling_declared"] = dict(self._pooling_declared)
        out["device"] = str(self.device)
        return out

    # -- 接口：编码 -------------------------------------------------------
    def _cache_key(self, text: str) -> str:
        """某文本在当前口径下的缓存键（内部别名）。"""
        return self.cache_key(text)

    def cache_key(self, text: str) -> str:
        """某文本在当前口径下的缓存键（公开：供取证侧对账落盘条目）。"""
        return self.cache.key_for(
            model_id=str(self.spec.model_id),
            revision=str(self.revision),
            pooling=str(self.pooling),
            max_length=int(self.max_length),
            normalize=str(NORMALIZE_MODE),
            weight_sha256=str(self._weight_sha256),
            text=str(text),
        )

    def cache_stats(self) -> Dict[str, Any]:
        """本进程的缓存命中 / 未命中计数（取证用）。"""
        return {"hits": int(self._hits), "misses": int(self._misses),
                "cache": self.cache.stats()}

    @staticmethod
    def _is_blank(text: str) -> bool:
        """空 / 仅空白判定（契约：该情形编码必须是零向量）。"""
        return str(text).strip() == ""

    def _forward(self, texts: Sequence[str], max_length: int) -> np.ndarray:
        """CLS pooling + L2 的原始前向（**不含缓存**）。"""
        import torch

        items = [str(t) for t in texts]
        blanks = [self._is_blank(t) for t in items]
        non_blank = [t for t, blank in zip(items, blanks) if not blank]
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        if not non_blank:
            return out
        encoded = self._tokenizer(
            non_blank, padding=True, truncation=True, max_length=int(max_length),
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            model_out = self._model(**encoded)
        # 手工 CLS pooling：取序列首位（XLM-R 的 <s>），不引 sentence-transformers
        cls = model_out.last_hidden_state[:, 0, :].to(torch.float32).cpu().numpy()
        cls = l2_normalize_rows(cls)
        cursor = 0
        for i, blank in enumerate(blanks):
            if blank:
                continue  # 空 / 仅空白文本 -> 零向量（契约）
            out[i] = cls[cursor]
            cursor += 1
        return out

    def encode_batch_with_stats(self, texts: Sequence[str]) -> List[EncodeResult]:
        """批量编码并返回含截断统计的结果（缓存命中时统计取自条目元数据）。"""
        items = [str(t) for t in texts]
        n = len(items)
        results: List[Optional[EncodeResult]] = [None] * n
        keys = [self._cache_key(t) if self.use_cache else "" for t in items]
        miss: List[int] = []
        for i, key in enumerate(keys):
            hit = self.cache.get(key) if self.use_cache else None
            if hit is None:
                miss.append(i)
                continue
            vec, meta = hit
            self._hits += 1
            results[i] = EncodeResult(
                vector=[float(x) for x in vec.tolist()],
                n_tokens=int(meta.get("n_tokens", 0)),
                n_truncated=int(meta.get("n_truncated", 0)),
                cached=True,
            )
        for b0 in range(0, len(miss), self.batch_size):
            chunk = miss[b0: b0 + self.batch_size]
            chunk_texts = [items[i] for i in chunk]
            self._misses += len(chunk)
            mat = self._forward(chunk_texts, self.max_length)
            for row, i in enumerate(chunk):
                text = items[i]
                if self._is_blank(text):
                    n_tokens, n_trunc = 0, 0
                else:
                    n_tokens = len(
                        self._tokenizer(text, truncation=False, verbose=False)["input_ids"]
                    )
                    n_trunc = max(0, int(n_tokens) - int(self.max_length))
                vec = [float(x) for x in mat[row].tolist()]
                results[i] = EncodeResult(
                    vector=vec, n_tokens=int(n_tokens), n_truncated=int(n_trunc), cached=False
                )
                if self.use_cache:
                    self.cache.put(
                        keys[i],
                        vec,
                        {
                            "model_id": str(self.spec.model_id),
                            "revision": str(self.revision),
                            "pooling": str(self.pooling),
                            "normalize": str(NORMALIZE_MODE),
                            "max_length": int(self.max_length),
                            "role": str(self.role),
                            "text_sha256": sha256_text(text),
                            "weight_sha256": str(self._weight_sha256),
                            "n_tokens": int(n_tokens),
                            "n_truncated": int(n_trunc),
                        },
                    )
        return [r for r in results if r is not None]

    def encode_with_stats(self, text: str) -> EncodeResult:
        """单条编码 + 截断统计（边界处置断言入口）。"""
        if not isinstance(text, str):
            raise TypeError(f"encode 需要 str，当前类型 {type(text).__name__}")
        return self.encode_batch_with_stats([text])[0]

    def encode(self, text: str) -> List[float]:
        """单条文本 -> ``dim`` 维 L2 归一化向量。"""
        return self.encode_with_stats(text).vector

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """批量编码（顺序保持）。"""
        return [r.vector for r in self.encode_batch_with_stats(texts)]

    def encode_matrix(self, texts: Sequence[str]) -> np.ndarray:
        """批量编码为 ``float32[L, D]``（步骤 2 的键表 / 参照下限使用）。"""
        items = [str(t) for t in texts]
        mat = np.asarray(self.encode_batch(items), dtype=np.float32)
        return mat.reshape((len(items), self.dim))


class EncoderConfigView:
    """编码器口径视图（与 ``VectorizerConfig`` 的**取用面**同形）。

    存在理由：上游 ``backends.build_registry(data.vectorizer.config)`` 只读 ``.dim``，
    ``train.py`` 的产物 meta 只读 ``.to_dict()``。本视图让 HF 编码器在这两处**零改动**
    接入，而不必让上游认识 HF 概念。
    """

    def __init__(self, declaration: Dict[str, Any]) -> None:
        self._declaration = dict(declaration)

    @property
    def dim(self) -> int:
        """连接参数 ``D``。"""
        return int(self._declaration["dim"])

    def to_dict(self) -> Dict[str, Any]:
        """口径字典（写进产物 meta 的 ``vectorizer_config``）。"""
        return dict(self._declaration)

    def fingerprint(self) -> str:
        """口径指纹（与所属编码器一致）。"""
        return sha256_bytes(canonical_dumps(self._declaration))


# ---------------------------------------------------------------------------
# 缓存包装（对 hash 家族亦可挂缓存，使「与落盘缓存逐位比对」对两家族都成立）
# ---------------------------------------------------------------------------


class CachedVectorizer:
    """给任意编码器加一层**逐条落盘缓存**（透明代理，不改变口径指纹）。

    参数
    ----
    inner : Any
        被包装的编码器（``dim`` / ``fingerprint`` / ``encode`` / ``encode_batch``）。
    cache : EmbeddingCache
        缓存实例。
    model_id / revision / pooling / max_length : 缓存键字段。

    关键不变量
    ----------
    * :meth:`fingerprint` **原样委派**给 ``inner``（缓存不参与口径）；
    * ``cache_key`` 只由 ``(口径字段, 文本哈希)`` 决定。
    """

    def __init__(
        self,
        inner: Any,
        cache: EmbeddingCache,
        *,
        model_id: str,
        revision: str,
        pooling: str,
        max_length: int,
        normalize: str = NORMALIZE_MODE,
        weight_sha256: str = "",
    ) -> None:
        self.inner = inner
        self.cache = cache
        self._key_fields: Dict[str, Any] = {
            "model_id": str(model_id),
            "revision": str(revision),
            "pooling": str(pooling),
            "max_length": int(max_length),
            "normalize": str(normalize),
            "weight_sha256": str(weight_sha256),
        }
        self._hits = 0
        self._misses = 0

    def __getattr__(self, item: str) -> Any:
        """未定义成员一律委派给 ``inner``（含 ``config`` / ``token_count``）。"""
        return getattr(self.inner, item)

    @property
    def dim(self) -> int:
        """连接参数 ``D``（委派）。"""
        return int(self.inner.dim)

    @property
    def cache_dir(self) -> str:
        """缓存根目录（显式暴露，便于取证侧定位落盘条目）。"""
        return str(self.cache.root)

    def fingerprint(self) -> str:
        """口径指纹（委派，**不含缓存信息**）。"""
        return str(self.inner.fingerprint())

    def cache_key(self, text: str) -> str:
        """该文本在当前口径下的缓存键。"""
        return self.cache.key_for(text=str(text), **self._key_fields)

    def encode_with_stats(self, text: str) -> EncodeResult:
        """单条编码（先查缓存）。

        说明
        ----
        ``hash`` 家族里只有 ``local-hash`` 暴露 ``encode_with_stats``；``zh-bag``
        没有该成员，此时回退 ``encode`` 并把 ``n_tokens`` 记为 0（**如实口径**：
        该家族不声明截断统计，不做臆造）。
        """
        if not isinstance(text, str):
            raise TypeError(f"encode 需要 str，当前类型 {type(text).__name__}")
        key = self.cache_key(text)
        hit = self.cache.get(key)
        if hit is not None:
            vec, meta = hit
            self._hits += 1
            return EncodeResult(
                vector=[float(x) for x in vec.tolist()],
                n_tokens=int(meta.get("n_tokens", 0)),
                n_truncated=int(meta.get("n_truncated", 0)),
                cached=True,
            )
        stats_fn = getattr(self.inner, "encode_with_stats", None)
        if callable(stats_fn):
            inner_res = stats_fn(text)
            vec = [float(x) for x in inner_res.vector]
            n_tokens = int(getattr(inner_res, "n_tokens", 0))
            n_trunc = int(getattr(inner_res, "n_truncated", 0))
        else:
            vec = [float(x) for x in self.inner.encode(text)]
            n_tokens = int(getattr(self.inner, "token_count", lambda _t: 0)(text))
            n_trunc = 0
        self._misses += 1
        self.cache.put(
            key,
            vec,
            {
                **{k: v for k, v in self._key_fields.items()},
                "text_sha256": sha256_text(text),
                "n_tokens": int(n_tokens),
                "n_truncated": int(n_trunc),
            },
        )
        return EncodeResult(
            vector=vec, n_tokens=int(n_tokens), n_truncated=int(n_trunc), cached=False
        )

    def encode(self, text: str) -> List[float]:
        """单条文本 -> ``dim`` 维向量。"""
        return self.encode_with_stats(text).vector

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """批量编码（顺序保持）。"""
        return [self.encode_with_stats(str(t)).vector for t in texts]

    def encode_matrix(self, texts: Sequence[str]) -> np.ndarray:
        """批量 -> ``float32[L, D]``。"""
        items = [str(t) for t in texts]
        mat = np.asarray(self.encode_batch(items), dtype=np.float32)
        return mat.reshape((len(items), self.dim))

    def cache_stats(self) -> Dict[str, Any]:
        """本次进程内的命中 / 未命中计数（取证用）。"""
        return {"hits": int(self._hits), "misses": int(self._misses),
                "cache": self.cache.stats()}


# ---------------------------------------------------------------------------
# 配置与工厂
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderConfig:
    """编码器选择配置（**唯一的选择实现入口**）。

    参数
    ----
    name : str
        注册表键；空串 = 按角色取默认实现（现状口径）。
    role : str
        ``"question"`` 或 ``"text_line"``（决定 ``max_length`` 与默认实现）。
    hash_dim : int
        仅 ``hash`` 家族生效的词袋桶数（``local-hash``）。
    use_length_features : bool
        仅 ``local-hash`` 生效：是否附加长度特征。
    max_length : int
        显式覆盖截断长度（``0`` = 用角色冻结口径）。
    source : str
        模型来源（本地目录或 HF 仓库 id）；空串 = 用注册表 ``model_id``。
    revision : str
        固定 revision；空串 = 用注册表声明。
    expect_dim : int
        显式覆盖声明维度（``0`` = 用注册表 ``expect_dim``）；**仅供错配自检**。
    cache_dir : str
        嵌入缓存根目录。
    use_cache : Optional[bool]
        ``None`` = 自动（``hf`` 家族默认开启，``hash`` 家族默认关闭）。
    batch_size : int
        前向批大小。
    device : str
        计算设备。
    local_files_only : bool
        只允许本地文件。
    verify_weights : bool
        是否计算权重 SHA256。
    """

    name: str = ""
    role: str = ROLE_QUESTION
    hash_dim: int = 80
    use_length_features: bool = True
    max_length: int = 0
    source: str = ""
    revision: str = ""
    expect_dim: int = 0
    cache_dir: str = DEFAULT_EMB_CACHE_DIR
    use_cache: Optional[bool] = None
    batch_size: int = 8
    device: str = "cpu"
    local_files_only: bool = False
    verify_weights: bool = True

    def __post_init__(self) -> None:
        """构造期校验（修复 N2）：``role`` 必须是 ``ROLES`` 之一。

        未校验时 ``resolved_name`` 会**静默回退**到 ``question`` 的默认实现、
        ``resolved_max_length`` 会抛裸 ``KeyError`` —— 两种行为都不该出现。
        """
        if str(self.role) not in ROLES:
            raise ValueError(
                f"EncoderConfig.role 仅允许 {list(ROLES)}，当前 {self.role!r}"
                "（未知角色会静默回退默认实现，故构造期即拒绝）"
            )
        if int(self.hash_dim) < 1:
            raise ValueError(f"EncoderConfig.hash_dim 必须 >= 1，当前 {self.hash_dim}")
        if int(self.max_length) < 0:
            raise ValueError(f"EncoderConfig.max_length 必须 >= 0（0 = 角色口径），当前 {self.max_length}")

    def resolved_name(self) -> str:
        """生效的注册名（空串 -> 角色默认实现）。"""
        return str(self.name) if str(self.name) else str(ROLE_DEFAULT_ENCODER[str(self.role)])

    def resolved_max_length(self) -> int:
        """生效的截断长度。"""
        return int(self.max_length) if int(self.max_length) > 0 else int(
            ROLE_MAX_LENGTH[str(self.role)]
        )

    def use_cache_effective(self) -> bool:
        """生效的缓存开关（``None`` -> ``hf`` 家族开启，``hash`` 家族关闭）。"""
        if self.use_cache is not None:
            return bool(self.use_cache)
        return get_spec(self.resolved_name()).kind == "hf"

    def to_dict(self) -> Dict[str, Any]:
        """可 JSON 化字典（写进产物 meta / 报告）。"""
        return {
            "name": str(self.resolved_name()),
            "role": str(self.role),
            "hash_dim": int(self.hash_dim),
            "use_length_features": bool(self.use_length_features),
            "max_length": int(self.resolved_max_length()),
            "source": str(self.source),
            "revision": str(self.revision),
            "expect_dim": int(self.expect_dim),
            "cache_dir": str(self.cache_dir),
            "use_cache": bool(self.use_cache_effective()),
            "batch_size": int(self.batch_size),
            "device": str(self.device),
            "local_files_only": bool(self.local_files_only),
            "verify_weights": bool(self.verify_weights),
        }


def declared_dim(cfg: Optional[EncoderConfig] = None) -> int:
    """**不加载模型**地取声明维度 ``D``（供 CLI 的 ``--dim`` 类计算）。

    对 ``hf`` 家族返回注册表声明的 ``expect_dim``；该值在真正构造时会被
    ``hidden_size`` 现场校验，不符即构造期报错。
    """
    config = cfg if cfg is not None else EncoderConfig()
    spec = get_spec(config.resolved_name())
    if int(config.expect_dim) > 0:
        return int(config.expect_dim)
    if spec.kind == "hash" and str(spec.name) == ENCODER_LOCAL_HASH:
        from .features import LENGTH_FEATURE_SPECS, MAX_FEATURE_DIM

        n_len = len(LENGTH_FEATURE_SPECS) if config.use_length_features else 0
        dim = int(config.hash_dim) + int(n_len)
        if dim > int(MAX_FEATURE_DIM):
            raise ValueError(
                f"hash 家族生效维度 D={dim} 超过上限 {MAX_FEATURE_DIM}"
                f"（历史纠正记录 #12：样本 1e3 量级时词袋维度取 10^1~10^2）"
            )
        return int(dim)
    return int(spec.expect_dim)


def _build_hash_encoder(cfg: EncoderConfig) -> Any:
    """构造 ``hash`` 家族实现（现状口径，逐位不变）。

    **``source`` 的处置（修复 D3）**：``hash`` 家族不使用任何模型文件，因此
    ``source`` **非空即视为配置错误**并立即报错 —— 指向不存在目录时抛
    :class:`EncoderUnavailableError`（与 HF 分支同形，使「模型缺失 -> 可读报错」
    对两家族都成立），指向**存在**的目录时抛 :class:`EncoderError`
    （明确拒绝而不是静默忽略，避免「以为切了模型其实没切」）。
    """
    name = cfg.resolved_name()
    src = str(cfg.source)
    if src:
        if not os.path.isdir(src):
            raise EncoderUnavailableError(
                f"hash 家族编码器 {name!r} 不使用模型文件，但 source={src!r} 不存在；"
                f"若本意是选择 HF 编码器，请改用 --encoder {ENCODER_BGE_M3}"
                f"（本地目标 {DEFAULT_MODEL_DIR!r}）"
            )
        raise EncoderError(
            f"hash 家族编码器 {name!r} 不使用模型目录（source={src!r} 存在但无意义）；"
            "请清空 source，或改用 HF 编码器"
        )
    if name == ENCODER_LOCAL_HASH:
        from .features import TextVectorizer, VectorizerConfig

        return TextVectorizer(
            VectorizerConfig(
                hash_dim=int(cfg.hash_dim), use_length_features=bool(cfg.use_length_features)
            )
        )
    if name == ENCODER_ZH_BAG:
        from .step2 import ZhBagVectorizer  # 惰性 import：打破 step2 <-> train 环

        return ZhBagVectorizer()
    raise KeyError(f"未登记的 hash 家族编码器 {name!r}；合法集合 = {list_encoders()}")


def build_vectorizer(cfg: Optional[EncoderConfig] = None) -> Any:
    """按配置构造编码器（**选择实现的唯一入口**）。

    参数
    ----
    cfg : Optional[EncoderConfig]
        选择配置；``None`` = 角色默认实现（``question`` -> ``local-hash``）。

    返回
    ----
    Any
        满足统一接口的对象；``use_cache_effective()`` 为真时是 :class:`CachedVectorizer`
        包装（指纹仍原样委派，故口径不受缓存影响）。

    异常
    ------
    KeyError
        未登记的编码器名。
    EncoderDimMismatchError
        实测维度与声明不符（``hf`` 家族构造期即抛）。
    EncoderUnavailableError
        模型缺失 / 权重不全 / 缺依赖 / 无网络。
    """
    config = cfg if cfg is not None else EncoderConfig()
    spec = get_spec(config.resolved_name())
    want_cache = bool(config.use_cache_effective())
    if spec.kind == "hf":
        encoder: Any = HFTextEncoder(
            spec,
            source=str(config.source),
            revision=str(config.revision),
            role=str(config.role),
            max_length=int(config.max_length),
            expect_dim=int(config.expect_dim),
            cache_dir=str(config.cache_dir),
            use_cache=want_cache,
            batch_size=int(config.batch_size),
            device=str(config.device),
            local_files_only=bool(config.local_files_only),
            verify_weights=bool(config.verify_weights),
        )
    else:
        encoder = _build_hash_encoder(config)
        assert_declared_dim(spec, int(encoder.dim), strict=False)
        if int(config.expect_dim) > 0 and int(encoder.dim) != int(config.expect_dim):
            raise EncoderDimMismatchError(
                f"编码器 {spec.name!r} 的实测维度 {encoder.dim} 与显式声明 "
                f"expect_dim={config.expect_dim} 不一致（构造期拒绝）"
            )
    # 缓存包装只在实现**自身不提供** cache_key 时套上（避免同一份缓存被写两次）
    if want_cache and not callable(getattr(encoder, "cache_key", None)):
        encoder = CachedVectorizer(
            encoder,
            EmbeddingCache(str(config.cache_dir)),
            model_id=str(spec.model_id) or str(spec.name),
            revision=str(config.revision) or str(spec.revision),
            pooling=str(spec.pooling),
            max_length=int(config.resolved_max_length()),
            weight_sha256=str(getattr(encoder, "weight_sha256", "")),
        )
    return encoder


def encoder_from_declaration(
    declaration: Dict[str, Any], expected_fingerprint: str = ""
) -> Any:
    """由产物 meta 里的口径声明**重建**编码器，并校验指纹。

    参数
    ----
    declaration : Dict[str, Any]
        编码器声明（:meth:`HFTextEncoder.declaration` 的落盘形式）。
    expected_fingerprint : str
        产物 meta 记录的指纹；非空时必须与重建结果一致（不一致立即报错）。

    返回
    ----
    Any
        重建的编码器（``hf`` 家族带缓存包装）。

    异常
    ------
    ValueError
        声明缺少必需键 / 指纹不一致。
    EncoderUnavailableError
        模型不可用。
    """
    required = ("kind", "name", "model_id", "revision", "pooling", "max_length", "dim")
    missing = [k for k in required if k not in declaration]
    if missing:
        raise ValueError(
            f"编码器声明缺少必需键 {missing}；实际键集合 = {sorted(declaration.keys())}"
        )
    if str(declaration["kind"]) != "hf":
        raise ValueError(
            f"encoder_from_declaration 只处理 kind='hf'，当前 {declaration['kind']!r}"
        )
    spec = get_spec(str(declaration["name"]))
    encoder = build_vectorizer(
        EncoderConfig(
            name=str(declaration["name"]),
            role=str(declaration.get("role", ROLE_QUESTION)),
            max_length=int(declaration["max_length"]),
            source=str(declaration.get("source", "")),
            revision=str(declaration["revision"]),
            expect_dim=int(declaration["dim"]),
            cache_dir=str(declaration.get("cache_dir", DEFAULT_EMB_CACHE_DIR)),
            use_cache=bool(declaration.get("use_cache", True)),
        )
    )
    actual = str(encoder.fingerprint())
    if expected_fingerprint and actual != expected_fingerprint:
        raise ValueError(
            "编码器口径指纹不一致：产物 meta 记录 "
            f"{expected_fingerprint[:16]}...，现场重建算出 {actual[:16]}...；"
            "拒绝在错配的特征空间上加载（修订 / 权重 / 池化 / 长度任一变化都会触发）"
        )
    if int(encoder.dim) != int(spec.expect_dim) and int(declaration["dim"]) == int(spec.expect_dim):
        raise ValueError(
            f"重建维度 {encoder.dim} 与注册表声明 {spec.expect_dim} 不一致"
        )
    return encoder


def vectorizer_from_meta(meta: Dict[str, Any]) -> Any:
    """由产物 meta 重建向量化器（**按口径分派**：HF 走注册表，其余走历史路径）。

    参数
    ----
    meta : Dict[str, Any]
        产物 meta；``vectorizer_config.kind == "hf"`` 时按编码器声明重建，
        否则原样委派 :func:`n3d_qa_learn.features.vectorizer_from_meta`（逐位不变）。
    """
    blob = dict(meta.get("vectorizer_config") or {})
    if str(blob.get("kind", "")) == "hf":
        return encoder_from_declaration(blob, str(meta.get("vectorizer_fingerprint", "")))
    from .features import vectorizer_from_meta as _legacy_from_meta

    return _legacy_from_meta(meta)


# ---------------------------------------------------------------------------
# 边界处置自检（G4）
# ---------------------------------------------------------------------------


def encode_with_stats_of(encoder: Any, text: str) -> EncodeResult:
    """统一取「编码 + 统计」：实现提供 ``encode_with_stats`` 就用它，否则回退 ``encode``。

    存在理由（修复 D1）：``zh-bag`` 早期只暴露 ``encode`` / ``encode_batch`` /
    ``encode_matrix``，直接调 ``encode_with_stats`` 会抛
    ``AttributeError: 'ZhBagVectorizer' object has no attribute 'encode_with_stats'``。
    本函数让**任何**满足最小接口的实现都能被自检 / 演练路径安全消费；截断统计缺失时
    如实记 0（不臆造），并由 :data:`EncoderSpec.truncation_stats` 决定该边界是
    「不适用」还是「失败」。
    """
    if not isinstance(text, str):
        raise TypeError(f"encode 需要 str，当前类型 {type(text).__name__}")
    fn = getattr(encoder, "encode_with_stats", None)
    if callable(fn):
        return fn(text)
    vec = [float(x) for x in encoder.encode(text)]
    return EncodeResult(
        vector=vec, n_tokens=int(getattr(encoder, "token_count", lambda _t: 0)(text)),
        n_truncated=0, cached=False,
    )


def summarize_selftest(cases: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """汇总边界自检结果：``n_cases`` / ``n_applicable`` / ``n_inapplicable`` / ``n_passed``。

    **口径（修复 D2）**：``applicable=False`` 的用例**不计入** ``passed``，也**不**算失败
    （如实登记「该边界对本实现不适用」，而不是把「不适用」伪装成「通过」或「失败」）。
    """
    applicable = [c for c in cases if bool(c.get("applicable", True))]
    return {
        "n_cases": int(len(cases)),
        "n_applicable": int(len(applicable)),
        "n_inapplicable": int(len(cases) - len(applicable)),
        "n_passed": int(sum(1 for c in applicable if c.get("passed"))),
    }


def failed_selftest_cases(cases: Sequence[Dict[str, Any]]) -> List[str]:
    """返回**适用但未通过**的用例名（G4 的失败判据）。"""
    return [str(c["case"]) for c in cases
            if bool(c.get("applicable", True)) and not c.get("passed")]


def boundary_selftest(
    cfg: Optional[EncoderConfig] = None,
    *,
    text: str = "N3D 问答学习框架的空 / 超长 / 缺失 / 维度错配自检文本",
) -> List[Dict[str, Any]]:
    """边界处置自检：空文本 / 超长 / 模型缺失 / 维度不符（适用项全 PASS 才可放全量）。

    参数
    ----
    cfg : Optional[EncoderConfig]
        被测编码器配置（``None`` = 角色默认实现）。
    text : str
        正常文本样例。

    返回
    ----
    List[Dict[str, Any]]
        逐项 ``{"case", "applicable", "passed", "detail"}``；**断言不写死初值**，
        只断言契约；``applicable=False`` 表示该边界对本实现不适用（见
        :func:`summarize_selftest` 的口径）。
    """
    config = cfg if cfg is not None else EncoderConfig()
    spec = get_spec(config.resolved_name())
    cases: List[Dict[str, Any]] = []
    names = ("empty_text_zero_vector", "overlong_text_truncated",
             "missing_model_readable_error", "dim_mismatch_construction_error")

    # --- 0) 编码器构造（失败也如实登记 4 项，不伪装成通过）--------------
    try:
        enc = build_vectorizer(config)
    except Exception as exc:  # noqa: BLE001
        reason = f"编码器不可用，无法执行：{type(exc).__name__}: {exc}"
        return [
            {"case": name, "applicable": True, "passed": False, "detail": reason}
            for name in names
        ]

    # --- 1) 空 / 仅空白 -> 零向量（对一切实现都适用）--------------------
    blank_ok = True
    blank_detail: Dict[str, Any] = {}
    for probe in ("", "   ", "\t\n "):
        res = encode_with_stats_of(enc, probe)
        norm = vector_l2_norm(res.vector)
        blank_detail[repr(probe)] = {"l2_norm": float(norm), "n_tokens": int(res.n_tokens),
                                     "dim": int(len(res.vector))}
        blank_ok = blank_ok and (norm == 0.0) and (len(res.vector) == int(enc.dim))
    cases.append({
        "case": "empty_text_zero_vector",
        "applicable": True,
        "passed": bool(blank_ok),
        "detail": {"dim": int(enc.dim), "probes": blank_detail,
                   "rule": "空/仅空白 -> 零向量（L2 范数恒为 0.0）"},
    })

    # --- 2) 超长 -> 截断且 n_truncated > 0 ------------------------------
    # `applicable` 由注册表声明决定：不声明截断统计的实现（zh-bag：n-gram 词袋无词元
    # 上限）该边界**不适用**，如实登记而不是判失败。
    over_applicable = bool(spec.truncation_stats)
    long_text = ("长文本截断自检 " * 4000)[: int(config.resolved_max_length()) * 4]
    if over_applicable:
        extra = encode_with_stats_of(enc, long_text)
        normal = encode_with_stats_of(enc, text)
        over_ok = int(extra.n_truncated) > 0 and int(normal.n_truncated) == 0
        over_detail: Dict[str, Any] = {
            "max_length": int(config.resolved_max_length()),
            "long_n_tokens": int(extra.n_tokens),
            "long_n_truncated": int(extra.n_truncated),
            "normal_n_tokens": int(normal.n_tokens),
            "normal_n_truncated": int(normal.n_truncated),
            "rule": "超长 -> 截断且 n_truncated > 0；正常长度 n_truncated == 0",
        }
    else:
        over_ok = False
        over_detail = {
            "rule": "该实现不声明词元截断统计（spec.truncation_stats=False），本边界不适用",
            "max_length": int(config.resolved_max_length()),
            "n_tokens": int(encode_with_stats_of(enc, text).n_tokens),
        }
    cases.append({
        "case": "overlong_text_truncated",
        "applicable": bool(over_applicable),
        "passed": bool(over_ok),
        "detail": over_detail,
    })

    # --- 3) 模型缺失 -> 可读报错（两家族都适用于「source 指向不存在目录」）----
    missing_dir = os.path.join(config.cache_dir, "_selftest_missing_model")
    missing_ok = False
    missing_detail: Dict[str, Any] = {}
    try:
        build_vectorizer(EncoderConfig(
            name=config.resolved_name(), role=str(config.role), source=missing_dir,
            revision=str(config.revision), cache_dir=str(config.cache_dir),
            use_cache=False,
        ))
        missing_detail["raised"] = False
    except EncoderUnavailableError as exc:
        missing_ok = True
        missing_detail = {"raised": True, "type": type(exc).__name__, "message": str(exc)[:400]}
    except Exception as exc:  # noqa: BLE001 - 非可读错误视为不通过
        missing_detail = {"raised": True, "type": type(exc).__name__, "message": str(exc)[:400]}
    cases.append({
        "case": "missing_model_readable_error",
        "applicable": True,
        "passed": bool(missing_ok),
        "detail": {**missing_detail, "probe_source": missing_dir,
                   "requires_model_files": bool(spec.requires_model_files),
                   "rule": "模型缺失 -> EncoderUnavailableError（可读报文）"},
    })

    # --- 4) 维度不符 -> 构造期报错 -------------------------------------
    wrong_dim = int(spec.expect_dim) + 1
    dim_ok = False
    dim_detail: Dict[str, Any] = {}
    try:
        build_vectorizer(EncoderConfig(
            name=config.resolved_name(), role=str(config.role), source=str(config.source),
            revision=str(config.revision), expect_dim=int(wrong_dim),
            cache_dir=str(config.cache_dir), use_cache=False,
            verify_weights=bool(config.verify_weights),
        ))
        dim_detail["raised"] = False
    except EncoderDimMismatchError as exc:
        dim_ok = True
        dim_detail = {"raised": True, "type": type(exc).__name__, "message": str(exc)[:400]}
    except Exception as exc:  # noqa: BLE001
        dim_detail = {"raised": True, "type": type(exc).__name__, "message": str(exc)[:400]}
    # ``actual`` 的**口径必须显式**：本轮之前它取的是 ``spec.expect_dim``
    # （= **注册表声明维度**，不是实测值），字段名容易被误读成「现场实测的 hidden_size」。
    # 修法（采纳「显式标注口径」方案，不动字段名以免打断既有断言）：保留 ``actual``，
    # 新增 ``actual_measured``（从构造期报文里解析出的**实测 hidden_size**，解析不到为
    # ``None``）与 ``actual_source`` 说明文本。
    measured: Optional[int] = None
    message = str(dim_detail.get("message", ""))
    marker = "hidden_size="
    if marker in message:
        tail = message.split(marker, 1)[1]
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            measured = int(digits)
    cases.append({
        "case": "dim_mismatch_construction_error",
        "applicable": True,
        "passed": bool(dim_ok),
        "detail": {**dim_detail, "declared": int(wrong_dim), "actual": int(spec.expect_dim),
                   "actual_measured": measured,
                   "actual_source": (
                       "actual = 注册表声明维度 spec.expect_dim（**不是实测值**）；"
                       "actual_measured = 从构造期报文解析出的实测 hidden_size"
                       "（解析不到为 null）"
                   ),
                   "rule": "hidden_size != 声明维度 -> 构造期 EncoderDimMismatchError"},
    })
    return cases


__all__ = [
    "ENCODER_SCHEMA_VERSION",
    "ROLE_QUESTION",
    "ROLE_TEXT_LINE",
    "ROLES",
    "ROLE_MAX_LENGTH",
    "DEFAULT_EMB_CACHE_DIR",
    "EMB_CACHE_SHARD_DEPTH",
    "DEFAULT_MODEL_DIR",
    "HF_MIRROR_ENDPOINT",
    "NORMALIZE_MODE",
    "BGE_M3_REVISION",
    "ENCODER_LOCAL_HASH",
    "ENCODER_ZH_BAG",
    "ENCODER_BGE_M3",
    "ROLE_DEFAULT_ENCODER",
    "EncoderError",
    "EncoderUnavailableError",
    "EncoderDimMismatchError",
    "EncoderSpec",
    "ENCODER_REGISTRY",
    "list_encoders",
    "get_spec",
    "register_encoder",
    "assert_declared_dim",
    "EncodeResult",
    "EmbeddingCache",
    "HFTextEncoder",
    "EncoderConfigView",
    "CachedVectorizer",
    "EncoderConfig",
    "declared_dim",
    "build_vectorizer",
    "encoder_from_declaration",
    "vectorizer_from_meta",
    "boundary_selftest",
    "encode_with_stats_of",
    "summarize_selftest",
    "failed_selftest_cases",
    "canonical_dumps",
    "sha256_bytes",
    "sha256_text",
    "file_sha256",
    "weight_sha256_memoized",
    "l2_normalize_rows",
    "vector_bytes",
    "vector_l2_norm",
]
