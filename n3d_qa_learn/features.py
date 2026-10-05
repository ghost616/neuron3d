"""n3d_qa_learn 的确定性文本向量化器（连接契约的数据侧一半）。

职责
----
把「问题文本」与「文本行」映射到同一个 ``D`` 维特征空间，供：

* N3D 后端模型的 ``input_dim = D`` 输入；
* 指针 Softmax 实现的候选键表 ``K in R^[L, D]``；
* 步骤 2（文本数据集匹配）的词袋余弦打分。

三条硬纪律
----------
1. **零新依赖**：只用 Python 标准库（``unicodedata`` / ``hashlib`` / ``json`` / ``math``）；
2. **严格确定性**：不使用内置 ``hash()``（PYTHONHASHSEED 随机化），一律用 ``blake2b``
   摘要取整数桶；同一文本在同一 ``VectorizerConfig`` 下**逐位一致**；
3. **口径指纹**：向量化口径（归一化链 / 哈希盐 / 维度 / 词元范围 / 长度特征口径）的
   SHA256 指纹写入产物 meta；加载产物时比对，不一致立即报错。

特征布局（合计 ``D = hash_dim + n_length_features`` 维）
----------------------------------------------------
* 前 ``hash_dim`` 维：**哈希词袋桶**。词元 = 拉丁词 + CJK 单字 + 字符 bigram，
  每个词元按其 ``(scope, 词元)`` 摘要映射到一个桶并累加计数；
* 后 ``n_length_features`` 维：**确定性长度/结构特征**，逐维 ``tanh`` 饱和后线性映射到
  ``[0, 1]``（口径常量冻结在 ``LENGTH_FEATURE_SPECS`` 中并参与指纹）；
* 最后对整向量做 **L2 归一化**（零向量保持零向量，打分侧显式返回 0 分）。

``D`` 的规模纪律（历史纠正记录 #12）
---------------------------------
``D`` 必须与训练样本量匹配：样本数在 1e3 量级时，词袋维度取 10^1~10^2。
本模块默认 ``D = 88``（``hash_dim = 80`` + 8 个长度特征），并强制 ``D <= 128``。
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Sequence

# ---------------------------------------------------------------------------
# 冻结常量（改动即改口径，必须同步指纹）
# ---------------------------------------------------------------------------

#: 向量化口径版本号；任何改变特征取值的改动都必须递增。
FEATURE_SCHEMA_VERSION: str = "n3dqa-feat-v1"

#: 哈希盐（冻结）：参与每个词元的桶映射，改动即全量特征变化。
HASH_SALT: str = "n3d_qa_learn:bow:v1"

#: 单条文本参与词袋的词元上限（超出部分按「超长截断」口径丢弃并在统计中可见）。
MAX_TOKENS: int = 256

#: 长度特征的冻结口径：``(名称, 饱和尺度)``；饱和值 = ``tanh(raw / scale)``。
LENGTH_FEATURE_SPECS = (
    ("n_tokens", 32.0),
    ("n_chars", 128.0),
    ("digit_ratio", 0.5),
    ("upper_ratio", 0.5),
    ("question_mark", 1.0),
    ("has_cjk", 1.0),
    ("n_cjk_chars", 32.0),
    ("mean_token_len", 8.0),
)

#: 维度上限（历史纠正记录 #12：样本 1e3 量级时词袋维度不得放大）。
MAX_FEATURE_DIM: int = 128

#: 默认词袋桶数。
DEFAULT_HASH_DIM: int = 80

#: 词袋计数的**次线性缩放**口径：``count -> 1 + log(count)``（冻结常量，参与指纹）。
#:
#: **实测口径（本模块现场测量，不得凭直觉反转）**：在本模块的数据规模与切分下，
#: **原始计数优于次线性缩放** —— 步骤 1 主测试集的 1-NN 宏平均准确率实测
#: 原始计数 ``0.3677`` vs 次线性 ``0.3146``（``C=8`` 档）、``0.3342`` vs ``0.3017``
#: （``C=10`` 档）。故默认关闭，开启仅作对照实验用。
SUBLINEAR_TF: bool = False


def _is_cjk(ch: str) -> bool:
    """判断单个字符是否属于 CJK 统一表意文字区间（中文问题逐字切分用）。"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


def _nfkc(text: str) -> str:
    """NFKC 归一化（不含 casefold），供大小写敏感统计使用。"""
    if not isinstance(text, str):
        raise TypeError(f"需要 str，当前类型 {type(text).__name__}")
    return unicodedata.normalize("NFKC", text)


def normalize_text(text: str) -> str:
    """归一化链（冻结口径）：NFKC -> casefold -> 空白折叠。

    参数
    ----
    text : str
        原始文本。

    返回
    ----
    str
        归一化文本（空白折叠为单空格，首尾去空白）。
    """
    normalized = _nfkc(text)
    normalized = normalized.casefold()
    return " ".join(normalized.split())


def tokenize(text: str) -> List[str]:
    """词元化（冻结口径）：空白/标点切分 + CJK 逐字 + 字符 bigram。

    参数
    ----
    text : str
        原始文本（内部先归一化，故对已归一化输入幂等）。

    返回
    ----
    List[str]
        词元列表，形如 ``w:<拉丁词>`` / ``c:<CJK 单字>`` / ``b:<字1><字2>``。
    """
    norm = normalize_text(text)
    tokens: List[str] = []
    for chunk in norm.split(" "):
        if not chunk:
            continue
        # 逐字符分类：CJK 逐字成词元；连续的非 CJK 字母数字聚成一个词
        buf: List[str] = []
        for ch in chunk:
            if _is_cjk(ch):
                if buf:
                    tokens.append("w:" + "".join(buf))
                    buf.clear()
                tokens.append("c:" + ch)
            elif ch.isalnum():
                buf.append(ch)
            else:
                if buf:
                    tokens.append("w:" + "".join(buf))
                    buf.clear()
        if buf:
            tokens.append("w:" + "".join(buf))
        # 字符 bigram（对整段，含 CJK），提供词形级泛化信号
        chars = [c for c in chunk if not c.isspace()]
        for i in range(len(chars) - 1):
            tokens.append("b:" + chars[i] + chars[i + 1])
    return tokens


def _bucket(token: str, scope: str, hash_dim: int) -> int:
    """把 ``(scope, 词元)`` 确定性映射到 ``[0, hash_dim)`` 的桶下标。

    参数
    ----
    token : str
        词元。
    scope : str
        作用域标记（问题侧固定 ``"q"``，文本行侧固定 ``"t"``）。
    hash_dim : int
        桶数。

    返回
    ----
    int
        桶下标。使用 ``blake2b``（非内置 ``hash``），故跨进程 / 跨运行**确定性**。
    """
    payload = f"{HASH_SALT}\x00{scope}\x00{token}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % int(hash_dim)


def length_features(text: str) -> List[float]:
    """按 ``LENGTH_FEATURE_SPECS`` 计算长度/结构特征（逐维 ``tanh`` 饱和到 ``[0, 1]``）。

    口径（冻结）
    -----------
    * ``upper_ratio`` 统计的是**原始字符串**中大写字母（``str.isupper()``）的占比。
      该维**有意不做 casefold**：它承载"全大写强调"这一书写结构（若用归一化后的文本
      计算，该维恒为 0、无区分力）。因此本维是唯一一个对大小写敏感的维度，特征空间
      整体上**不是**大小写不变；
    * **空文本 / 仅空白文本一律返回全 0 向量**（契约：空输入的编码必须是零向量，
      不得因长度特征的常数项而变成非零向量）。

    参数
    ----
    text : str
        原始文本。

    返回
    ----
    List[float]
        长度 = ``len(LENGTH_FEATURE_SPECS)``，各值落在 ``[0, 1]``。
    """
    norm = normalize_text(text)
    if norm == "":
        return [0.0] * len(LENGTH_FEATURE_SPECS)
    toks = tokenize(text)
    n_tokens = float(len(toks))
    n_chars = float(len(norm))
    n_digits = float(sum(1 for c in norm if c.isdigit()))
    n_upper = float(sum(1 for c in text if c.isupper()))
    n_qmark = float(sum(1 for c in norm if c in "?\uff1f"))
    n_cjk = float(sum(1 for c in norm if _is_cjk(c)))
    raw = {
        "n_tokens": n_tokens,
        "n_chars": n_chars,
        "digit_ratio": (n_digits / n_chars) if n_chars > 0 else 0.0,
        "upper_ratio": (n_upper / float(len(text))) if text else 0.0,
        "question_mark": 1.0 if n_qmark > 0 else 0.0,
        "has_cjk": 1.0 if n_cjk > 0 else 0.0,
        "n_cjk_chars": n_cjk,
        "mean_token_len": (
            sum(len(t) for t in toks) / n_tokens if n_tokens > 0 else 0.0
        ),
    }
    out: List[float] = []
    for name, scale in LENGTH_FEATURE_SPECS:
        value = math.tanh(float(raw[name]) / float(scale))
        out.append((value + 1.0) / 2.0)  # 映射到 [0, 1]
    return out

# ---------------------------------------------------------------------------
# 向量化配置与指纹
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorizerConfig:
    """向量化器的**完整**口径（构造期不变量）。

    参数
    ----
    hash_dim : int
        哈希词袋桶数（``>= 1``）。
    use_length_features : bool
        是否附加 ``LENGTH_FEATURE_SPECS`` 长度特征（关闭会改变 ``D``，故参与指纹）。

    关键不变量
    ----------
    * ``1 <= hash_dim <= MAX_FEATURE_DIM``；
    * 生效维度 ``dim = hash_dim + (8 if use_length_features else 0) <= MAX_FEATURE_DIM``。
    """

    hash_dim: int = DEFAULT_HASH_DIM
    use_length_features: bool = True
    #: 冻结的口径版本字段（非构造参数，仅用于指纹自解释）
    schema_version: str = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.hash_dim) < 1:
            raise ValueError(f"VectorizerConfig.hash_dim 必须 >= 1，当前 {self.hash_dim}")
        if int(self.hash_dim) > MAX_FEATURE_DIM:
            raise ValueError(
                f"VectorizerConfig.hash_dim 必须 <= {MAX_FEATURE_DIM}"
                f"（历史纠正记录 #12：样本 1e3 量级时词袋维度取 10^1~10^2），"
                f"当前 {self.hash_dim}"
            )
        if int(self.dim) > MAX_FEATURE_DIM:
            raise ValueError(
                f"VectorizerConfig 生效维度 {self.dim} 超过上限 {MAX_FEATURE_DIM}"
                f"（历史纠正记录 #12），请减小 hash_dim 或关闭长度特征"
            )

    @property
    def n_length_features(self) -> int:
        """生效的长度特征列数。"""
        return len(LENGTH_FEATURE_SPECS) if self.use_length_features else 0

    @property
    def dim(self) -> int:
        """生效特征维度 ``D = hash_dim + n_length_features``。"""
        return int(self.hash_dim) + int(self.n_length_features)

    def to_dict(self) -> Dict[str, object]:
        """序列化为可 JSON 化字典（键顺序固定，供指纹与产物 meta 使用）。"""
        return {
            "schema_version": str(FEATURE_SCHEMA_VERSION),
            "hash_salt": str(HASH_SALT),
            "hash_dim": int(self.hash_dim),
            "use_length_features": bool(self.use_length_features),
            "n_length_features": int(self.n_length_features),
            "length_feature_specs": [[n, float(s)] for n, s in LENGTH_FEATURE_SPECS],
            "max_tokens": int(MAX_TOKENS),
            "normalize_chain": ["NFKC", "casefold", "collapse_ws"],
            "normalize": "l2",
            "sublinear_tf": bool(SUBLINEAR_TF),
            "dim": int(self.dim),
        }

    def fingerprint(self) -> str:
        """向量化**口径指纹**（SHA256 over 规范化 JSON）。

        返回
        ----
        str
            64 位十六进制小写串。加载产物时与 ``meta["vectorizer_fingerprint"]``
            比对，不一致（含维度 / 盐 / 归一化链 / 长度特征口径任一变化）立即报错。
        """
        blob = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


@dataclass
class VectorizeResult:
    """单条文本的向量化结果（含截断可观测性）。

    属性
    ----
    vector : List[float]
        长度 = ``VectorizerConfig.dim``。
    n_tokens : int
        参与词袋的词元总数（截断前）。
    n_truncated : int
        因 ``MAX_TOKENS`` 被丢弃的词元数。
    """

    vector: List[float]
    n_tokens: int
    n_truncated: int = 0


class TextVectorizer:
    """确定性文本向量化器：把文本映射到 ``D`` 维 L2 归一化稠密向量。

    参数
    ----
    config : VectorizerConfig
        向量化口径。

    关键不变量
    ----------
    * 同一 ``(config, text)`` 的输出**逐位一致**（无随机数、无内置 ``hash()``）；
    * 输出维度恒为 ``config.dim``；
    * 零向量（空文本 / 仅空白）保持为零向量（不做除零）。
    """

    def __init__(self, config: VectorizerConfig = VectorizerConfig()) -> None:
        self.config = config

    @property
    def dim(self) -> int:
        """特征维度 ``D``。"""
        return int(self.config.dim)

    def fingerprint(self) -> str:
        """口径指纹（委派 ``VectorizerConfig.fingerprint``）。"""
        return self.config.fingerprint()

    def encode_with_stats(self, text: str) -> VectorizeResult:
        """向量化并返回截断统计（供边界处置断言使用）。

        参数
        ----
        text : str
            文本（允许为空串）。

        返回
        ----
        VectorizeResult
            向量与词元统计。
        """
        if not isinstance(text, str):
            raise TypeError(f"encode 需要 str，当前类型 {type(text).__name__}")
        toks = tokenize(text)
        n_total = len(toks)
        kept = toks[:MAX_TOKENS]
        n_trunc = n_total - len(kept)

        # 步骤 1：哈希词袋桶（累加计数 -> 可选次线性缩放）
        counts: Dict[int, float] = {}
        for tok in kept:
            b = _bucket(tok, "q", int(self.config.hash_dim))
            counts[b] = counts.get(b, 0.0) + 1.0
        vector = [0.0] * self.dim
        if SUBLINEAR_TF:
            for b, c in counts.items():
                vector[b] = 1.0 + math.log(c)
        else:
            for b, c in counts.items():
                vector[b] = c
        # 步骤 2：长度/结构特征（可选）。契约：空文本 / 仅空白文本的编码必须是**零向量**，
        # 故词元为空时整向量保持全 0（不叠加任何长度特征常数项）。
        if self.config.use_length_features and kept:
            lf = length_features(text)
            offset = int(self.config.hash_dim)
            for i, value in enumerate(lf):
                vector[offset + i] = float(value)

        # 步骤 3：L2 归一化（零向量保持零向量）
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0.0:
            vector = [v / norm for v in vector]
        return VectorizeResult(vector=vector, n_tokens=n_total, n_truncated=n_trunc)

    def encode(self, text: str) -> List[float]:
        """向量化为 ``D`` 维列表（见 ``encode_with_stats``）。"""
        return self.encode_with_stats(text).vector

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """批量向量化（顺序保持）。"""
        return [self.encode(t) for t in texts]

    def token_count(self, text: str) -> int:
        """返回词元数（不含截断）。"""
        return len(tokenize(text))


# ---------------------------------------------------------------------------
# 向量打分（步骤 2 的文本匹配与指针实现的候选取键共用）
# ---------------------------------------------------------------------------


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """L2 归一化向量的余弦相似度；任一为零向量时返回 ``0.0``。

    参数
    ----
    a, b : Sequence[float]
        同维向量。

    返回
    ----
    float
        ``[-1, 1]``；零向量参与时恒为 ``0.0``（显式定义，不做除零）。
    """
    if len(a) != len(b):
        raise ValueError(f"cosine 要求同维，当前 {len(a)} vs {len(b)}")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = 0.0
    for x, y in zip(a, b):
        dot += x * y
    return float(dot / (na * nb))


def vectorizer_from_meta(meta: Dict[str, object]) -> TextVectorizer:
    """从产物 meta 重建 ``TextVectorizer`` 并做**口径指纹校验**。

    参数
    ----
    meta : Dict[str, object]
        产物 meta，必须含 ``vectorizer_config`` 与 ``vectorizer_fingerprint``。

    返回
    ----
    TextVectorizer
        重建的向量化器。

    异常
    ------
    KeyError
        meta 缺少必需键（报文列出实际键集合）。
    ValueError
        指纹不一致（报文给出期望值与实测值）。
    """
    required = ("vectorizer_config", "vectorizer_fingerprint")
    missing = [k for k in required if k not in meta]
    if missing:
        raise KeyError(
            f"产物 meta 缺少向量化口径键 {missing}；实际键集合 = {sorted(meta.keys())}"
        )
    cfg_blob = dict(meta["vectorizer_config"])  # type: ignore[arg-type]
    cfg = VectorizerConfig(
        hash_dim=int(cfg_blob["hash_dim"]),
        use_length_features=bool(cfg_blob["use_length_features"]),
    )
    actual = cfg.fingerprint()
    expected = str(meta["vectorizer_fingerprint"])
    if actual != expected:
        raise ValueError(
            "向量化口径指纹不一致：产物 meta 记录 "
            f"{expected[:16]}...，当前实现算出 {actual[:16]}...；"
            "说明产物是用另一套向量化口径训练的，拒绝加载（防止静默错配特征空间）"
        )
    return TextVectorizer(cfg)


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "HASH_SALT",
    "MAX_TOKENS",
    "LENGTH_FEATURE_SPECS",
    "MAX_FEATURE_DIM",
    "DEFAULT_HASH_DIM",
    "SUBLINEAR_TF",
    "normalize_text",
    "tokenize",
    "length_features",
    "VectorizerConfig",
    "VectorizeResult",
    "TextVectorizer",
    "cosine",
    "vectorizer_from_meta",
]