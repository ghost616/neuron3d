"""n3d_qa.build_dataset —— 通用 QA 数据集处理模块：QA 数据集 -> N3D 数组格式的确定性构建。

定位
----
本模块是**通用 QA 数据集处理模块**（任意问答数据集 -> N3D 数组格式）；**当前内置 TriviaQA**
（证据段落二分类）作为**参考实现**：下述 split / 映射 / 特征口径均为 TriviaQA 口径，
后续 QA 数据集按 adapter 挂入（输入解析 / 文档定位 / 答案合并 / split 表），
共用特征层与 npz 落盘层。

职责
----
把 ``data/triviaqa/OpenDataLab___TriviaQA/raw/triviaqa-rc.tar.gz`` 中的 QA 问答对与
evidence 文档，转成 ``X[M, D] float32`` / ``y[M] int64`` 的 npz（``D = feature_dim(hash_dim, no_bag, features)``：
缺省 ``--features base`` 且 ``--hash-dim 64`` -> ``64 + 6 = 70``；``--no-bag`` 为 6；
``--features rich`` 在附加特征块上追加 4 列 IDF/TF-IDF 特征（附加块 6 -> 10，故
``rich + h64 = 74``、``rich + no-bag = 10``）；原计划口径 1030 已实测不达标、仅作失败对照留档），
供 ``n3d_shape`` 以 ``--dataset npz`` 直接训练。本模块**自包含**：不导入、不修改
n3d_proto / n3d_sphere / n3d_shape / n3d_viz / framework 的任何代码。

split 口径（2026-10-04 扩容）
----------------------------
* ``wiki`` / ``web``：**verified** 子集（``qa/verified-wikipedia-dev.json`` 318 题 /
  ``qa/verified-web-dev.json`` 407 题），产物名 ``n3d_triviaqa_verified_{wiki,web}_dev*.npz``；
* ``wiki-dev`` / ``web-dev``：**非 verified 的全量 dev**（``qa/wikipedia-dev.json`` /
  ``qa/web-dev.json``），按 ``QuestionId`` 剔除 verified 子集后构建
  （剔除依据是同一 QA JSON 家族内 verified 成员的 ``QuestionId`` 集合，集合在 meta 中留档为
  ``verified_exclusion``），产物名 ``n3d_triviaqa_{wiki,web}_dev*.npz``。

任务定义（证据段落二分类）
--------------------------
* 样本 = (Question, Evidence 文档) 对；
* 正样本 ``label=1``：文档来自该题的 ``EntityPages ∪ SearchResults``；
* 负样本 ``label=0``：由**固定 seed**从**同 split 内其他题的文档**中采样，1:1 均衡。

归档访问口径（不整包解压）
--------------------------
归档实测：2 665 779 500 字节（压缩）/ 7 341 073 957 字节（解压）/ 487 254 个成员。
本模块**只用** ``tarfile.open(..., "r|gz")`` 顺序流式扫描 + ``extractfile`` 按需抽取
目标成员，**绝不调用 ``extractall``**、绝不整包解压。

[!] **实测事实（决定了遍数口径）**：``qa/*.json`` 位于归档**末尾**（成员序号
487248 / 487251，总数 487254），即 evidence 全部排在 QA 之前。因此"先知道要抽哪些
evidence"**不可能在同一遍流式扫描内完成**（gzip 流不可回退）。故：

* **冷构建 = 2 遍流式扫描**：第 1 遍只抽 QA JSON 并落缓存，第 2 遍只抽目标 evidence；
* **缓存命中（含 E1 幂等复跑）= 1 遍**。

两遍都只抽取目标成员，都不解压整包；实际遍数写入产物 meta 的 ``archive_passes``。

确定性口径
----------
* 所有随机性来自 ``np.random.default_rng(negative_seed)``，**不触碰全局 RNG**；
* 哈希词袋用 ``blake2b`` 确定性哈希（同样不消耗 RNG）；
* npz 由本模块自写的**确定性 zip**（固定时间戳 / 固定外部属性）落盘，
  故同参数重复构建**逐字节一致**（SHA256 相同）；meta 中不含任何时间戳/环境指纹。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

__all__ = [
    "ARCHIVE_SHA256",
    "FEATURE_DIM",
    "HASH_DIM",
    "EXTRA_DIM",
    "FEATURE_SET_CHOICES",
    "FEATURES_DEFAULT",
    "RICH_EXTRA_DIM",
    "IDF_SMOOTH_OFFSET",
    "HASH_TOKEN_ID_BYTES",
    "SPLIT_SCALE",
    "QA_JSON_MEMBERS",
    "VERIFIED_PARENT_SPLIT",
    "DEFAULT_NEGATIVE_SEED",
    "DEFAULT_PACK_FLUSH_BYTES",
    "IdfTable",
    "DocPack",
    "DocPackBuilder",
    "DocPackAccessor",
    "BagCache",
    "extra_dim_for",
    "split_slug",
    "hash_token_id",
    "numeric_answer_fraction",
    "rich_columns_from_pair",
    "preload_pairs_split",
    "ArchiveIntegrityError",
    "BuildContractError",
    "QaRecord",
    "Sample",
    "build_split",
    "archive_tag",
    "same_archive",
    "feature_columns",
    "feature_dim",
    "compile_answer_patterns",
    "doc_contains_answer",
    "hash_bag",
    "load_archive_qa",
    "merge_answer_strings",
    "parse_qa_json",
    "save_npz_deterministic",
    "sha256_file",
    "stream_extract_members",
    "tokenize",
    "verify_archive",
    "main",
]

# ======================================================================
# 常量（全部为**现场实测**口径，写死进代码；改动即需重新实测）
# ======================================================================
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

DEFAULT_ARCHIVE: str = os.path.join(
    PROJECT_ROOT, "data", "triviaqa", "OpenDataLab___TriviaQA", "raw", "triviaqa-rc.tar.gz"
)
# 归档 SHA256（构建前必须校验，不符即退码 2）
ARCHIVE_SHA256: str = "ef94fac6db0541e5bb5b27020d067a8b13b1c1ffc52717e836832e02aaed87b9"
ARCHIVE_SIZE_BYTES: int = 2665779500
ARCHIVE_MEMBER_COUNT: int = 487254
ARCHIVE_UNCOMPRESSED_BYTES: int = 7341073957

# split -> 归档内 QA JSON 成员名（全部**现场枚举**自归档，禁止凭记忆手写；
# verified 两名实测成员序号 487248 / 487251，dev 两名见缓存 ``members`` 段与 README §2）
QA_JSON_MEMBERS: Dict[str, str] = {
    "wiki": "qa/verified-wikipedia-dev.json",
    "web": "qa/verified-web-dev.json",
    "wiki-dev": "qa/wikipedia-dev.json",
    "web-dev": "qa/web-dev.json",
}
# 实测问题条数（用于"解析后条数"这一层的一致性校验）；**只登记已实测过的 split**，
# 未登记的 split 跳过该层校验（见 parse_qa_json 的 expect_n 判空）。
QA_JSON_QUESTIONS: Dict[str, int] = {"wiki": 318, "web": 407}
# 注册在案、可能被流式扫描的**全部** QA 成员（缺 / 多都会让 QA 缓存成员索引校验失败并自动重扫）
QA_JSON_MEMBERS_REGISTERED: Tuple[str, ...] = tuple(QA_JSON_MEMBERS[k] for k in ("wiki", "web", "wiki-dev", "web-dev"))

# verified 子集的来源 split（扩容时用于剔除重合 QuestionId；实测 verified 的 QuestionId
# 全部出现在同名 dev 家族内，故同一 split 家族内的 verified 记录可直接剔除）
VERIFIED_PARENT_SPLIT: Dict[str, str] = {"wiki-dev": "wiki", "web-dev": "web"}

# 文档类别 -> 归档内目录前缀（实测：evidence/web 413173 个成员、evidence/wikipedia 74070 个）
EVIDENCE_DIR: Dict[str, str] = {"wikipedia": "evidence/wikipedia", "web": "evidence/web"}
DOC_KINDS: Tuple[str, ...] = ("wikipedia", "web")

# 产物目录（产物一律 checkpoints/triviaqa/；验证类产物落 _verify/，缓存落 _cache/）
DEFAULT_OUT_DIR: str = os.path.join(PROJECT_ROOT, "checkpoints", "triviaqa")
VERIFY_OUT_DIR: str = os.path.join(DEFAULT_OUT_DIR, "_verify")
CACHE_DIR: str = os.path.join(DEFAULT_OUT_DIR, "_cache")

# 产物名模板（**按 split 的 verified 属性**决定是否带 verified 标签；verified 两名保持原样，
# 保证既有 4 个正式产物的文件名与字节一字未动）
# [!] verified split 的 slug 是 ``wiki``/``web``，模板补 ``_dev``；
#     非 verified（dev）split 的 slug 已经是 ``wiki_dev``/``web_dev``，**不再补** ``_dev``
#     ——否则会出现 ``n3d_triviaqa_wiki_dev_dev_h64.npz`` 这种重复后缀。
OUT_NAME_TEMPLATE: str = "n3d_triviaqa_verified_{slug}_dev.npz"
OUT_NAME_TEMPLATE_PLAIN: str = "n3d_triviaqa_{slug}.npz"


def split_slug(split: str) -> str:
    """split 名 -> 文件名中的短标识（``-`` 统一替换为 ``_``，避免文件名歧义）。

    参数
    ----
    split : str
        canonical split 名（``wiki`` / ``web`` / ``wiki-dev`` / ``web-dev``）。

    返回
    ----
    str
        文件名用的短标识（``wiki-dev`` -> ``wiki_dev``）。
    """
    return str(split).replace("-", "_")


def out_name_template(split: str) -> str:
    """split -> 产物文件名**前缀模板**（verified 子集带 ``verified`` 标签，非 verified 不带）。

    [!] 返回的是"前缀 + {slug}"模板（``n3d_triviaqa_verified_{slug}`` / ``n3d_triviaqa_{slug}``）：
    split 的 ``_dev`` 后缀由 :func:`split_slug` 带进来，绝不能在这里再补一次
    （否则 dev split 会得到 ``..._wiki_dev_dev_...`` 这种重复后缀）。

    参数
    ----
    split : str
        ``wiki`` / ``web`` / ``wiki-dev`` / ``web-dev``。

    返回
    ----
    str
        含 ``{slug}`` 占位符的文件名前缀模板。
    """
    return OUT_NAME_TEMPLATE if str(split) not in VERIFIED_PARENT_SPLIT else OUT_NAME_TEMPLATE_PLAIN

# 特征维数：D = hash_dim（哈希词袋）+ 6（覆盖度/Jaccard/长度/数字答案标记）
#
# [!] 修订（风后 R28）：`--hash-dim` 缺省由 1024 降为 **64**（产出 D = 64 + 6 = 70）。
#     原因（实测；口径说明见本模块 change_history 与 HASH_DIM_PLAN_ORIGINAL 的注释）：
#     n3d_shape 数据层「每 5 取 1」切分后训练样本仅 1024 条，
#     原口径 1024 维词袋块「特征维数 ≈ 样本数」，该块退化为噪声（单块 test_acc 35.94%，
#     numpy 逻辑回归 5 折 CV 仅 wiki 0.3445 / web 0.2866，比随机还差），把端到端 test_acc
#     从 91.02%（去掉该块）拖到 59.38%。降维后同口径 20 epoch 即达标。
HASH_DIM: int = 64
# 原计划口径的哈希维度（仅作**失败对照**留档与 README/验证口径说明用；不再是缺省产出）
HASH_DIM_PLAN_ORIGINAL: int = 1024
# 缺省（base）附加特征列数：既有 6 列，**只改名不改值/不改序**，保证 base 产物逐字节不变
EXTRA_DIM: int = 6
FEATURE_DIM: int = HASH_DIM + EXTRA_DIM

# ----------------------------------------------------------------------
# 特征集合（--features）：base = 既有 6 列附加特征；rich = 6 列 + 4 列 IDF/TF-IDF 特征
# ----------------------------------------------------------------------
FEATURE_SET_CHOICES: Tuple[str, ...] = ("base", "rich")
FEATURES_DEFAULT: str = "base"
# rich 相对 base 追加的列数（附加块 6 -> 10）
RICH_EXTRA_DIM: int = 4
# IDF 语料 = 本次构建该 split 的全部文档池（见 build_meta 的 documents 定义），平滑式：
#     idf(token) = log((1 + M) / (1 + df(token))) + 1
# 采用**满权重 +1 的平滑式**（等价于把它当作"稀有度加权"而非"抑制高频"），
# 使 qcov_idf / dcov_idf 在 IDF 全为常数时**退化为 base 的 q_to_d_coverage / d_to_q_coverage**
# （该退化关系是 E4 跨特征集合一致性的依据）。确定性、不消耗全局 RNG。
IDF_SMOOTH_OFFSET: float = 1.0

# 哈希词袋的确定性口径（blake2b，不消耗全局 RNG）
HASH_SALT: bytes = b"n3d-triviaqa-bow-v1\x00"
HASH_DIGEST_SIZE: int = 8

# 负样本采样种子（固定；写进 meta 供回读比对）
SPLIT_SCALE: Dict[str, str] = {
    "wiki": "inmem",
    "web": "inmem",
    "wiki-dev": "compact",
    "web-dev": "compact",
}
# DocPack 落盘分片阈值（控制峰值内存）
DEFAULT_PACK_FLUSH_BYTES: int = 128 << 20

DEFAULT_NEGATIVE_SEED: int = 20261003
# 短数字答案的"额外保护"长度阈值：len <= 该值的纯数字答案禁止与相邻数字/小数点连写
SHORT_NUMERIC_MAX_LEN: int = 3

TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMERIC_ONLY_RE = re.compile(r"^[0-9]+$")
WORD_HEAD_RE = re.compile(r"\w")
WORD_TAIL_RE = re.compile(r"\w$")

# 契约容差
L2_NORM_ABS_TOL: float = 1e-5
BALANCE_TOL: float = 0.01
TEXT_ENCODING: str = "utf-8"

# 全部 canonical split（``all`` 只覆盖既有 verified 两名：它保持"缺省产出 = 既有正式产物"语义）
SPLIT_CHOICES_CANONICAL: Tuple[str, ...] = ("wiki", "web", "wiki-dev", "web-dev")
SPLIT_CHOICES: Tuple[str, ...] = ("wiki", "web", "wiki-dev", "web-dev", "all")
# ``--split all`` 展开为哪些 split（既有两个 verified split；沿用旧行为，向后兼容）
SPLIT_ALL_EXPANSION: Tuple[str, ...] = ("wiki", "web")

# 特征列定义（写入 meta 的 feature_columns；端点**闭区间**、按 start 升序、无缝覆盖 [0, D-1]）
def _check_features(features: str) -> str:
    """校验特征集合名（口径写死，非法值直接报错，不静默回退）。

    参数
    ----
    features : str
        特征集合名（应为 :data:`FEATURE_SET_CHOICES` 之一）。

    返回
    ----
    str
        规范化后的特征集合名。

    异常
    ------
    BuildContractError
        取值不在 :data:`FEATURE_SET_CHOICES` 内时抛出。
    """
    val = str(features)
    if val not in FEATURE_SET_CHOICES:
        raise BuildContractError(
            f"未知的 features 取值 {val!r}；可选 {list(FEATURE_SET_CHOICES)}"
        )
    return val


def feature_columns(
    hash_dim: int = HASH_DIM, no_bag: bool = False, features: str = FEATURES_DEFAULT
) -> Tuple[Dict[str, Any], ...]:
    """按**实际生效口径**生成逐列特征定义（缺省 ``hash_dim = 64`` / ``features = base`` -> ``D = 70``）。

    参数
    ----
    hash_dim : int
        哈希词袋块维数（缺省写死值 :data:`HASH_DIM` = 64）。
    no_bag : bool
        为真时**不产出词袋块**（``--no-bag``），列定义只含附加特征（``base`` 为 6 列）。
    features : str
        ``base``（既有 6 列附加特征）或 ``rich``（6 列 + 4 列 IDF/TF-IDF 特征 ->
        附加块 10 列，``D = hash_dim + 10``）。

    返回
    ----
    Tuple[Dict[str, Any], ...]
        每列（或块）的 ``start`` / ``end`` / ``name`` / ``definition``；
        无缝覆盖 ``[0, feature_dim(hash_dim, no_bag, features) - 1]``。
    """
    feat = _check_features(features)
    base = 0 if bool(no_bag) else int(hash_dim)
    bag_cols: Tuple[Dict[str, Any], ...] = ()
    if not bool(no_bag):
        bag_cols = (
            {
                "start": 0,
                "end": base - 1,
                "name": f"bow_hash{base}",
                "definition": (
                    "哈希词袋：把 (Question + '\\n' + Evidence) 拼接后按 [a-z0-9]+ 分词（先小写），"
                    f"每个 token 经 blake2b(salt+token, digest_size=8) 取大端整数后 mod {base} 落桶并计数；"
                    f"该 {base} 维块整体做 L2 归一化（范数为 0 时保持全 0）。不消耗全局 RNG。"
                ),
            },
        )
    extra = _extra_columns(base)
    if feat == "rich":
        return bag_cols + extra + _rich_columns(base + EXTRA_DIM)
    return bag_cols + extra


def _extra_columns(base: int) -> Tuple[Dict[str, Any], ...]:
    """生成 6 列**基础**附加特征（覆盖度/Jaccard/长度/数字答案标记）的列定义。

    [!] ``base`` 特征集合下这 6 列的名称、顺序与定义**一字未改**，故既有 base 产物逐字节不变。

    参数
    ----
    base : int
        附加特征块的起始列号（``no_bag`` 时为 0，否则为 ``hash_dim``）。

    返回
    ----
    Tuple[Dict[str, Any], ...]
        6 条列定义，列号 ``base .. base + EXTRA_DIM - 1``。
    """
    return (
        {
            "start": base,
            "end": base,
            "name": "q_to_d_coverage",
            "definition": "|T(q) ∩ T(d)| / |T(q)|（token 集合口径；|T(q)| = 0 时取 0）",
        },
        {
            "start": base + 1,
            "end": base + 1,
            "name": "d_to_q_coverage",
            "definition": "|T(q) ∩ T(d)| / |T(d)|（token 集合口径；|T(d)| = 0 时取 0）",
        },
        {
            "start": base + 2,
            "end": base + 2,
            "name": "jaccard",
            "definition": "|T(q) ∩ T(d)| / |T(q) ∪ T(d)|（token 集合口径；并集为空时取 0）",
        },
        {
            "start": base + 3,
            "end": base + 3,
            "name": "log1p_doc_tokens",
            "definition": "log1p(文档 token 数)（**含重数**的 token 序列长度）",
        },
        {
            "start": base + 4,
            "end": base + 4,
            "name": "log1p_question_tokens",
            "definition": "log1p(问题 token 数)（含重数的 token 序列长度）",
        },
        {
            "start": base + 5,
            "end": base + 5,
            "name": "numeric_answer_flag",
            "definition": (
                "数字类答案标记：该题合并后的答案集合中存在 strip 后匹配 ^[0-9]+$ 的答案则 1.0，"
                "否则 0.0（逐题常量，同一 QuestionId 的全部行取值相同）"
            ),
        },
    )


# 缺省口径（hash_dim = HASH_DIM = 64）的列定义；``--hash-dim N`` / ``--no-bag`` 生效时
# 用 feature_columns(N, no_bag) 现场生成（两者必须同源，见 E0 回读比对）
FEATURE_COLUMNS: Tuple[Dict[str, Any], ...] = feature_columns(HASH_DIM)


def _rich_columns(base: int) -> Tuple[Dict[str, Any], ...]:
    """生成 4 列 **rich** 特征（IDF 加权覆盖度 / TF-IDF 余弦 / 数值型答案占比）的列定义。

    IDF 语料口径在一个 split 内是**常量**：``M`` = 该 split 的文档池大小
    （``meta["counts"]["pool_documents"]``），``df(token)`` = 含该 token 的池内文档数，
    ``idf = log((1 + M) / (1 + df)) + IDF_SMOOTH_OFFSET``。

    参数
    ----
    base : int
        rich 特征块的起始列号（base 口径下为 ``hash_dim + EXTRA_DIM``）。

    返回
    ----
    Tuple[Dict[str, Any], ...]
        4 条列定义，列号 ``base .. base + RICH_EXTRA_DIM - 1``。
    """
    return (
        {
            "start": base,
            "end": base,
            "name": "qcov_idf",
            "definition": (
                "IDF 加权问题->文档覆盖率：sum_{t in T(q) ∩ T(d)} idf(t) / sum_{t in T(q)} idf(t)"
                "（token 集合口径；分母为 0 时取 0）"
            ),
        },
        {
            "start": base + 1,
            "end": base + 1,
            "name": "dcov_idf",
            "definition": (
                "IDF 加权文档->问题覆盖率：sum_{t in T(q) ∩ T(d)} idf(t) / sum_{t in T(d)} idf(t)"
                "（token 集合口径；分母为 0 时取 0）"
            ),
        },
        {
            "start": base + 2,
            "end": base + 2,
            "name": "tfidf_cos",
            "definition": (
                "问题与文档的 TF-IDF 余弦相似度：tf = 该 token 在文本中的出现次数（含重数），"
                "权重 w(t) = tf(t) * idf(t)；分子 = sum_{t in T(q) ∩ T(d)} w_q(t) * w_d(t)，"
                "分母 = ||w_q||_2 * ||w_d||_2（任一范数为 0 时取 0）"
            ),
        },
        {
            "start": base + 3,
            "end": base + 3,
            "name": "ans_isnum_frac",
            "definition": (
                "答案中数值型 token 占比：答案集合中 strip 后匹配 ^[0-9]+$ 的条数 / 答案集合总条数"
                "（逐题常量）；与既有 numeric_answer_flag 口径不同（该列给比例、既有列给 0/1）"
            ),
        },
    )


def extra_dim_for(features: str = FEATURES_DEFAULT) -> int:
    """特征集合 -> 附加特征块列数（``base`` 为 6，``rich`` 为 10）。

    参数
    ----
    features : str
        ``base`` / ``rich``。

    返回
    ----
    int
        附加特征块列数。
    """
    return int(EXTRA_DIM) + (int(RICH_EXTRA_DIM) if _check_features(features) == "rich" else 0)


def feature_dim(
    hash_dim: int = HASH_DIM, no_bag: bool = False, features: str = FEATURES_DEFAULT
) -> int:
    """特征总维数 ``D = hash_dim + extra_dim_for(features)``。

    缺省 ``64 + 6 = 70``（``base``）；``no_bag`` 时为 6（``base``）/ 10（``rich``）；
    ``rich`` 且不 ``no_bag`` 时为 ``hash_dim + 10``（缺省 ``74``）。

    参数
    ----
    hash_dim : int
        哈希词袋块维数（``no_bag=True`` 时忽略）。
    no_bag : bool
        为真时不计词袋块（``--no-bag``）。
    features : str
        ``base`` / ``rich``。

    返回
    ----
    int
        总维数 ``D``。
    """
    return (0 if bool(no_bag) else int(hash_dim)) + extra_dim_for(features)


# 归档访问层（顺序流式；不整包解压）


class ArchiveIntegrityError(RuntimeError):
    """归档缺失或 SHA256 校验不符（CLI 层映射为退出码 2）。"""


class BuildContractError(RuntimeError):
    """构建过程中违反了写死的口径（字段缺失 / 映射未命中 / 平衡断言失败等）。"""


# ======================================================================


# ======================================================================
def sha256_file(path: str, chunk_bytes: int = 1 << 22) -> str:
    """计算文件整包 SHA256（分块读取，内存占用与文件大小无关）。

    参数
    ----
    path : str
        文件路径。
    chunk_bytes : int
        每次读取的字节数（默认 4 MiB）。

    返回
    ----
    str
        小写十六进制 SHA256。

    异常
    ------
    FileNotFoundError
        文件不存在时抛出。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(int(chunk_bytes)), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_archive(archive_path: str, expected_sha256: str = ARCHIVE_SHA256) -> str:
    """构建前的归档校验（**写死口径**）：存在性 + 大小 + SHA256。

    参数
    ----
    archive_path : str
        归档路径。
    expected_sha256 : str
        期望的 SHA256（默认即写死的 ``ARCHIVE_SHA256``）。

    返回
    ----
    str
        实测 SHA256（等于 ``expected_sha256``）。

    异常
    ------
    ArchiveIntegrityError
        文件缺失、大小不符或 SHA256 不符时抛出（报文含实测值与期望值）。
    """
    if not os.path.isfile(archive_path):
        raise ArchiveIntegrityError(f"归档不存在：{archive_path}")
    got_size = os.path.getsize(archive_path)
    if int(got_size) != int(ARCHIVE_SIZE_BYTES):
        raise ArchiveIntegrityError(
            f"归档大小不符：实测 {got_size} 字节，期望 {ARCHIVE_SIZE_BYTES} 字节（{archive_path}）"
        )
    got = sha256_file(archive_path)
    if got != str(expected_sha256):
        raise ArchiveIntegrityError(
            f"归档 SHA256 不符：实测 {got}，期望 {expected_sha256}（{archive_path}）"
        )
    return got


def stream_extract_members(
    archive_path: str,
    wanted: Set[str],
    on_member: Optional[Callable[..., None]] = None,
    progress_every: int = 0,
) -> Tuple[Dict[str, bytes], List[str]]:
    """**一遍**顺序流式扫描归档，只抽取 ``wanted`` 中的成员。

    实现要点
    --------
    * 用 ``tarfile.open(archive_path, "r|gz")`` 的**流式**模式（只能顺序前进，
      与"不整包解压"的纪律一致）；遍历 member 时非目标成员由 tarfile 内部
      seek 跳过（数据被丢弃，不落盘、不进内存）；
    * **不调用 ``extractall``**，不把归档解压到磁盘；
    * 目标成员若给了 ``on_member`` 回调则交给回调消费（可用于边读边算、不驻留内存），
      否则累积到返回字典。

    参数
    ----
    archive_path : str
        归档路径。
    wanted : Set[str]
        目标成员名集合（与 ``member.name`` 逐字比较）。
    on_member : Optional[Callable[..., None]]
        目标成员的消费回调 ``(member_name, data, member_index)``；``None`` 时数据进返回字典。
    progress_every : int
        每读取该数量的归档成员打印一行进度（``0`` = 不打印）。

    返回
    ----
    Tuple[Dict[str, bytes], List[str]]
        ``(命中成员名 -> 内容, 未命中成员名升序列表)``；用了 ``on_member`` 时字典为空。

    异常
    ------
    FileNotFoundError
        归档不存在时抛出。
    """
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"归档不存在：{archive_path}")
    found: Dict[str, bytes] = {}
    seen: Set[str] = set()
    n_members = 0
    with tarfile.open(archive_path, "r|gz") as tf:
        for member in tf:
            n_members += 1
            if int(progress_every) > 0 and n_members % int(progress_every) == 0:
                print(
                    f"[n3d_qa]   流式扫描进度：{n_members} 成员，已命中 {len(seen)}/"
                    f"{len(wanted)}",
                    flush=True,
                )
            if not member.isfile():
                continue
            name = member.name
            if name not in wanted:
                continue
            fh = tf.extractfile(member)
            if fh is None:  # pragma: no cover - isfile() 已过滤
                continue
            data = fh.read()
            seen.add(name)
            if on_member is not None:
                on_member(name, data, n_members)
            else:
                found[name] = data
    missing = sorted(str(x) for x in (set(wanted) - seen))
    return found, missing


def _qa_cache_path(archive_sha256: str, split: str) -> str:
    """QA JSON 的缓存路径（键 = 归档 SHA256 前缀 + split，保证换归档自动失效）。"""
    return os.path.join(CACHE_DIR, f"{archive_sha256[:16]}_{split}_qa.json")


def preload_pairs_split(
    split: str,
    records: Sequence[QaRecord],
    pool: Set[str],
    stream_docs: Callable[[Callable[[str, bytes], None]], Tuple[List[str], int]],
    features: str,
    pack_builder: Optional[DocPackBuilder],
    missing: Set[str],
) -> Dict[str, Any]:
    """compact 路径的**两遍流式预计算**（IDF 语料 / 逐文档聚合 / 答案命中 / 紧凑表示）。

    为什么必须两遍
    --------------
    IDF 依赖**语料全局**的 df 统计，而"每篇文档的 IDF 加权聚合量"又依赖 IDF 表本身；
    若把文档全文驻留内存即可省掉一遍，但那正是 compact 路径要避免的（token 规模巨大的
    split 会把常驻内存推到 GB 级）。故：第 1 遍只数 df（顺带建答案倒排索引），第 2 遍再算
    逐文档聚合量 + 答案命中 + 紧凑表示。

    答案命中为什么用**倒排索引 + 候选集**
    ------------------------------------
    负样本是从文档池里采的，可能把某文档配给**任何一个**题目，故 (题目, 文档) 的答案命中
    需要全量可用（候选对规模 = |题目| x |文档池|，web-dev 上约 1.6e8 对）。直接对全部对
    跑正则不可行，故：
    1. 由题目侧建"**答案预筛 token -> 题目**"索引（题目数 x 每题预筛 token 数，规模很小）；
    2. 第 1 遍对每篇文档的**唯一 token 集合**查该索引，得到"该文档的候选题目"；
    3. 第 2 遍只对这些候选对跑 :func:`doc_contains_answer`，其余对直接记 0。
    预筛 token 能与文档 token 相交是"答案出现在文档里"的**必要条件**（见
    :func:`answer_token_set`），故该剪枝**不改变判定结果**；最后仍逐 (题目, 文档) 断言命中表完整。

    口径
    ----
    * 语料 = **本次构建该 split 的文档池**（``pool``，即纳入题目的正样本文档并集）；
    * ``df`` 按**唯一 token 标识**去重计数（同一文档同一 token 只计一次）；
    * ``compact`` 路径下文档**不驻留全文**：第 2 遍只留"唯一 token 标识 + 重数"的紧凑表示，
      答案命中在被抽取的同一遍内现算；
    * 不消耗全局 RNG；负样本采样在 :func:`_collect_samples_precomputed` 内另行进行。

    参数
    ----
    split : str
        split 名（仅用于报文）。
    records : Sequence[QaRecord]
        本次构建的记录（用于建答案预筛索引与答案命中判定）。
    pool : Set[str]
        文档池成员名集合。
    stream_docs : Callable[[Callable[[str, bytes], None]], Tuple[List[str], int]]
        单遍流式扫描函数：``on_member(member_name, raw_bytes, member_index)``；
        返回 ``(未命中列表, 命中数)``。
    features : str
        ``base`` / ``rich``（rich 才需要 IDF 表与逐文档 IDF 聚合量）。
    pack_builder : Optional[DocPackBuilder]
        紧凑表示构造器（compact 路径必给；``None`` 时报错而不是静默少产出）。
    missing : Set[str]
        出参：归档中缺失的文档成员名。

    返回
    ----
    Dict[str, Any]
        ``idf`` / ``packed`` / ``doc_features`` / ``doc_ids`` / ``answer_hits`` /
        ``numeric_answer_fraction`` / ``corpus_documents`` / ``token_counts`` / ``doc_bytes`` /
        ``idf_vocab_size`` / ``df_pass_documents`` / ``hit_candidate_pairs`` / ``raw_bytes_read``。

    异常
    ------
    BuildContractError
        紧凑表示构造器缺失、两遍扫描的文档集合不一致，或答案命中表不完整时抛出。
    """
    feat = _check_features(features)
    if pack_builder is None:
        raise BuildContractError("preload_pairs_split 需要 pack_builder（compact 路径专用）")
    qids = sorted({str(r.question_id) for r in records})
    rec_by_qid: Dict[str, QaRecord] = {str(r.question_id): r for r in records}
    # 倒排索引（**答案首词 -> 题目**）：只有"该文档出现了答案的首个 token"才可能是命中，
    # 于是"文档 token 集合"（几百到几千个）即可把 1e8 量级候选压到极小规模。
    # 倒排索引（**答案 token -> 题目**）：命中必然要求答案文本出现在文档里，故文档必然含该答案的
    # 某个 token（必要条件，折叠安全由 _variant_tokens 保证）。索引取答案的**全部 token**而非首词
    # ——候选只增不减，宁多不漏；配合 _second 里的**字符级必要条件**再收一次。
    pref_to_qs: Dict[int, List[str]] = {}
    for q in qids:
        rec = rec_by_qid[q]
        toks: Set[str] = set()
        for lit in rec.answer_literals:
            toks.update(_variant_tokens(lit))
        for a in rec.answer_values:
            s = str(a).strip()
            if NUMERIC_ONLY_RE.match(s):
                toks.add(s)
        for t in toks:
            pref_to_qs.setdefault(int(hash_token_id(t)), []).append(q)
    # 非 ASCII 答案的题目（token/字面预筛对其**不可证伪**，故只按字符级条件做候选）
    ascii_safe: Set[str] = {q for q in qids if answer_is_ascii(rec_by_qid[q].answer_values)}
    # 每题的**字符级必要条件集合**（折叠安全，见 answer_char_set）
    q_char_set: Dict[str, frozenset] = {
        q: answer_char_set(rec_by_qid[q].answer_values, rec_by_qid[q].answer_literals) for q in qids
    }
    doc_bytes: Dict[str, int] = {m: 0 for m in pool}
    df: Dict[int, int] = {}
    doc_candidates: Dict[str, Set[str]] = {}
    seen_docs = 0
    missing.clear()

    t_pass1 = 0.0
    t_pass2 = 0.0
    n_regex_calls = 0

    def _first(name: str, data: bytes, _index: int = -1) -> None:
        """第 1 遍：数 df（唯一 token 标识去重）+ 求该文档的候选题目集合。"""
        nonlocal seen_docs, t_pass1
        _ts = time.perf_counter()
        seen_docs += 1
        doc_bytes[name] = len(data)
        text = data.decode(TEXT_ENCODING, errors="replace")
        toks = tokenize(text)
        uniq = set()
        cand: Set[str] = set()
        for t in toks:
            uniq.add(hash_token_id(t))
        # 候选 1：文档 token（含折叠变体）查答案 token 索引
        for t in _variant_tokens(text):
            hit_qs = pref_to_qs.get(int(hash_token_id(t)))
            if hit_qs is not None:
                cand.update(hit_qs)
        for t in _token_letter_forms(text):
            for q in pref_to_qs.get(int(hash_token_id(t)), ()):
                cand.add(q)
        # 候选 2（**折叠安全的必要条件**）：文档字符集与答案字符集必须相交，否则必然不命中。
        #   * 非 ASCII 答案的题目：**只认**这一判据（token/字面预筛对它们不可证伪）；
        #   * 纯 ASCII 答案的题目：token 索引已经给出候选，这一判据再收一次（同为保证）。
        dchars = doc_char_set(text)
        keep: Set[str] = set()
        for q in cand:
            if not dchars.isdisjoint(q_char_set.get(q, frozenset())):
                keep.add(q)
        for q in qids:
            if q in ascii_safe:
                continue
            if not dchars.isdisjoint(q_char_set.get(q, frozenset())):
                keep.add(q)
        for key in uniq:
            df[key] = df.get(key, 0) + 1
        doc_candidates[name] = keep
        t_pass1 += time.perf_counter() - _ts

    m1, n1 = stream_docs(_first)
    missing.update(m1)
    actual_docs = int(n1)
    idf = IdfTable(df, max(actual_docs, 1))
    df.clear()
    del df

    pack = DocPackBuilder(pack_builder.out_path) if pack_builder is not None else None
    doc_features: Dict[str, Tuple[float, float, int]] = {}
    token_counts: Dict[str, int] = {}
    hits: Dict[Tuple[str, str], int] = {}
    n_candidates = 0
    total_raw = 0

    def _second(name: str, data: bytes, _index: int = -1) -> None:
        """第 2 遍：逐文档 token 计数 + IDF 聚合量 + 候选集内答案命中 + 紧凑表示。"""
        nonlocal total_raw, n_candidates, t_pass2, n_regex_calls
        _ts = time.perf_counter()
        total_raw += len(data)
        text = data.decode(TEXT_ENCODING, errors="replace")
        toks = tokenize(text)
        token_counts[name] = len(toks)
        cnt = _term_counts([hash_token_id(t) for t in toks])
        uq_arr = np.asarray(sorted(cnt.keys()), dtype=np.uint64)
        cnt_arr = np.asarray([cnt[int(k)] for k in uq_arr.tolist()], dtype=np.int64)
        if pack is not None:
            pack.add(uq_arr, cnt_arr)
        if feat == "rich":
            norm = _tfidf_norm(cnt, idf)
            doc_features[name] = (
                float(idf.lookup(uq_arr).sum()),
                float(norm),
                int(len(toks)),
            )
        else:
            doc_features[name] = (0.0, 0.0, int(len(toks)))
        cand = doc_candidates.get(name, set())
        n_candidates += len(cand)
        if cand:
            for q in cand:
                r = rec_by_qid[q]
                # 候选已由字符级必要条件给出（折叠安全）：这里直接正则判定真值
                n_regex_calls += 1
                hits[(q, name)] = 1 if doc_contains_answer(text, r.answer_patterns) else 0
        t_pass2 += time.perf_counter() - _ts

    m2, n2 = stream_docs(_second)
    missing.update(m2)
    if int(n2) != int(actual_docs):
        raise BuildContractError(
            f"split={split} 两遍流式扫描的文档数不一致：第 1 遍 {actual_docs}，第 2 遍 {n2}"
        )
    # [!] 先在"补 0 之前"断言**候选覆盖完整**（原先的 len(hits) 比较在补 0 之后做，恒真 = 空转）
    computed = set(hits)
    expected_pairs = {(q, m) for q in qids for m in token_counts}
    if not computed.issubset(expected_pairs):
        raise BuildContractError(
            f"split={split} 答案命中表出现越界键：{sorted(computed - expected_pairs)[:5]}"
        )
    for name in token_counts:
        for q in qids:
            if (q, name) not in hits:
                hits[(q, name)] = 0
    expect_pairs = len(records) * len(token_counts)
    if len(hits) != expect_pairs:
        raise BuildContractError(
            f"split={split} 答案命中表不完整：{len(hits)} != {expect_pairs}（题目 x 池内文档）"
        )
    # [!] 阶段耗时与正则调用数一并回传：D2 规模曲线的瓶颈定位依据（不凭印象判断快慢）。
    profile = {
        "pass1_s": round(float(t_pass1), 2),
        "pass2_s": round(float(t_pass2), 2),
        "regex_calls": int(n_regex_calls),
    }
    packed = pack.finalize() if pack is not None else None
    # [!] doc_ids 只需"成员名 -> 打包序号"（与 pool 的升序一致）：这里显式按成员名升序生成，
    #     而不是依赖 token_counts 的插入序（插入序 = 归档流式顺序，恰与升序一致，但不该依赖它）；
    #     随后释放 token_counts（web-dev 量级下它是上百 MB 的临时表，且下游只用 doc_features）。
    doc_ids = {m: i for i, m in enumerate(sorted(doc_features.keys()))}
    token_counts.clear()
    del token_counts
    return {
        "idf": idf,
        "packed": packed,
        "doc_features": doc_features,
        "doc_ids": doc_ids,
        "answer_hits": hits,
        "numeric_answer_fraction": {
            r.question_id: numeric_answer_fraction(r.answer_values) for r in records
        },
        "corpus_documents": int(actual_docs),
        "doc_bytes": doc_bytes,
        "idf_vocab_size": int(idf.size),
        "df_pass_documents": int(seen_docs),
        "hit_candidate_pairs": int(n_candidates),
        "raw_bytes_read": int(total_raw),
        "profile": profile,
    }


def _qa_cache_members_valid(
    payload: Any, needed: Sequence[str], members: Dict[str, str]
) -> bool:
    """校验 QA 缓存里的"成员 -> 成员序号"是否与当前配置一致（**未登记的一律判失效**）。

    参数
    ----
    payload : Any
        缓存文件解析出的对象（应为 dict）。
    needed : Sequence[str]
        本次请求**必须已落档**的 QA 成员名（成员路径）。
    members : Dict[str, str]
        当前配置的成员->成员序号映射。

    返回
    ----
    bool
        为真表示缓存可复用（键齐备、``order`` 等于计划顺序、``members`` 与配置逐项一致，
        且每条记录的 ``member`` 与实际配置相同）。
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("members"), dict):
        return False
    cached_members = payload["members"]
    try:
        cached_entries = {str(k): v for k, v in cached_members.items()}
    except (KeyError, TypeError):
        return False
    # [!] 判定口径：缓存里**只含已登记成员**，且**覆盖本次请求的成员**。不能要求
    #     "缓存键集合 == 全部 4 个登记成员"——CLI 的 --split 一次只请求 1~2 个 split，
    #     那样该判定恒假、每次构建都白白多扫一遍 2.6 GB 归档（实测恒 2 遍）。
    # [!] 命名空间必须一致：``members`` 的键是**成员路径**（``QA_JSON_MEMBERS.values()``），
    #     不是 split 名。早期实现拿 split 名去比成员路径 -> 两集合恒不相交 -> 判定恒假 ->
    #     每次构建都白扫一遍 2.6 GB 归档（实测恒 2 遍）。
    registered = {str(v) for v in members.values()}
    if not set(cached_entries).issubset(registered):
        return False
    if not set(str(x) for x in needed).issubset(set(cached_entries)):
        return False
    for key in needed:
        key = str(key)
        if key not in cached_entries:
            continue
        entry = cached_entries[key]
        if not isinstance(entry, dict):
            return False
        if str(entry.get("member")) != key:
            return False
        if not isinstance(entry.get("index"), int):
            return False
    return True


def _qa_cache_paths(archive_sha256: str, splits: Sequence[str]) -> Dict[str, str]:
    """`split -> 该 split 的 QA 原始字节缓存路径`（键 = 归档 SHA256 前缀 + split）。"""
    return {split: _qa_cache_path(archive_sha256, split) for split in splits}


def load_archive_qa(
    archive_path: str,
    archive_sha256: str,
    splits: Sequence[str],
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Tuple[Dict[str, bytes], int, List[str], Dict[str, str]]:
    """取得各 split 的 QA JSON 原始字节（**第 1 遍流式扫描**，带缓存 + 成员序号留档）。

    口径
    ----
    * 缓存文件落在 ``checkpoints/triviaqa/_cache/<sha16>_<split>_qa.json``（**内容 = 该 split
      的 QA JSON 原始字节**，与历史版本完全一致，保证既有 base 产物逐字节不变）；
    * 成员名 -> 成员序号（归档内序号）留在 ``<sha16>_qa_members.json``：
      历史版本没有这个文件，本函数会**自动补扫一遍**把它补上（此后暖构建仍是 0 遍扫描）；
    * 一次调用把**所有**待取 split 的 QA JSON 在同一遍扫描里抽出（不为每个 split 各扫一遍）；
    * 成员序号只写进**归档访问层的缓存文件**，不写进产物 meta（保证产物字节与缓存状态无关）。

    参数
    ----
    archive_path : str
        归档路径。
    archive_sha256 : str
        归档实测 SHA256（缓存键）。
    splits : Sequence[str]
        需要的 split 列表（``wiki`` / ``web`` / ``wiki-dev`` / ``web-dev``）。
    use_cache : bool
        是否允许读/写缓存。
    refresh_cache : bool
        为真则忽略已有缓存、强制重新抽取并覆盖缓存。

    返回
    ----
    Tuple[Dict[str, bytes], int, List[str], Dict[str, str]]
        ``(split -> QA JSON 原始字节, 实际扫描遍数, 缓存状态说明, 成员名 -> 成员序号)``。

    异常
    ------
    BuildContractError
        归档中找不到该 split 的 QA JSON 成员时抛出。
    """
    # ``need`` = 本次请求必须落档的成员；``members`` = "成员路径 -> 归档成员序号"（1-based）。
    need = sorted({QA_JSON_MEMBERS[str(s)] for s in splits})
    out: Dict[str, bytes] = {}
    status: List[str] = []
    todo: List[str] = []
    for split in splits:
        path = _qa_cache_path(archive_sha256, str(split))
        if use_cache and not refresh_cache and os.path.isfile(path):
            with open(path, "rb") as fh:
                out[str(split)] = fh.read()
            status.append(f"{split}:缓存命中({path})")
        else:
            todo.append(str(split))
    members: Dict[str, int] = {}
    members_path = os.path.join(CACHE_DIR, f"{archive_sha256[:16]}_qa_members.json")
    payload: Any = None
    if use_cache and not refresh_cache and os.path.isfile(members_path):
        try:
            with open(members_path, "r", encoding=TEXT_ENCODING) as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            payload = None
    # 已有的成员档（用于合并写回，避免把历史成员档丢掉）
    cached_entries: Dict[str, Any] = {}
    if isinstance(payload, dict) and isinstance(payload.get("members"), dict):
        cached_entries = {str(k): v for k, v in payload["members"].items()}
    members_ok = _qa_cache_members_valid(payload, need, QA_JSON_MEMBERS)
    if members_ok:
        for name, item in cached_entries.items():
            members[name] = int(item["index"])
    elif not todo:
        # 字节缓存命中但成员档缺失/不含本次请求的成员：补扫一遍归档（只补成员档，不改字节缓存）
        todo = sorted({str(s) for s in splits})
        status.append("成员序号缓存缺失/不含请求成员 -> 补扫一遍归档以留档成员序号")
    passes = 0
    if todo:
        wanted = {QA_JSON_MEMBERS[s] for s in todo} | {QA_JSON_MEMBERS[s] for s in splits}
        print(
            f"[n3d_qa] 第 1 遍流式扫描：抽取 QA JSON {sorted(wanted)}（不解压整包）",
            flush=True,
        )
        got: Dict[str, bytes] = {}
        scanned: Dict[str, int] = {}

        def _on_qa(name: str, data: bytes, index: int = -1) -> None:
            """收集 QA JSON 字节并留档其归档成员序号（1-based）。"""
            got[name] = data
            scanned[name] = int(index)

        _, missing = stream_extract_members(
            archive_path, wanted, on_member=_on_qa, progress_every=100000
        )
        passes = 1
        if missing:
            raise BuildContractError(
                f"归档中找不到 QA JSON 成员：{missing}（期望 {sorted(wanted)}）"
            )
        # 合并：历史成员档（若有效）+ 本次实扫到的成员序号
        if members_ok:
            for name, item in cached_entries.items():
                members[str(name)] = int(item["index"])
        members.update(scanned)
        if use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(members_path, "w", encoding=TEXT_ENCODING) as fh:
                json.dump(
                    {
                        "archive_sha256": archive_sha256,
                        "members": {
                            # 语义统一为"成员路径 -> 归档成员序号"（缓存命中分支也按此读回）
                            name: {"member": name, "index": int(members[name])}
                            for name in sorted(members)
                        },
                    },
                    fh,
                    ensure_ascii=False,
                    indent=1,
                )
        for split in todo:
            raw = got[QA_JSON_MEMBERS[split]]
            out[split] = raw
            if use_cache:
                with open(_qa_cache_path(archive_sha256, split), "wb") as fh:
                    fh.write(raw)
                status.append(f"{split}:现场抽取并写入缓存({len(raw)} 字节)")
            else:
                status.append(f"{split}:现场抽取({len(raw)} 字节)")
    if any(m not in members for m in need):
        raise BuildContractError(
            f"QA 成员序号不完整（缓存与配置不一致）：缺 {[m for m in need if m not in members]}；"
            f"请加 --refresh-cache 重跑"
        )
    for split in splits:
        if str(split) not in out:
            raise BuildContractError(f"QA 字节缓存缺失：split={split}")
    return out, passes, status, members


def load_archive_evidence(
    archive_path: str,
    wanted: Set[str],
) -> Tuple[Dict[str, bytes], List[str]]:
    """**第 2 遍流式扫描**：把目标 evidence 文档抽进内存（按需抽取，不解压整包）。

    参数
    ----
    archive_path : str
        归档路径。
    wanted : Set[str]
        目标成员名集合（全部待构建 split 的并集）。

    返回
    ----
    Tuple[Dict[str, bytes], List[str]]
        ``(成员名 -> 原始字节, 未命中成员名升序列表)``。
    """
    print(f"[n3d_qa] 第 2 遍流式扫描：抽取 evidence {len(wanted)} 个成员", flush=True)
    return stream_extract_members(archive_path, wanted, progress_every=100000)


# ======================================================================
# QA 解析与答案判定
# ======================================================================
@dataclass(frozen=True)
class QaRecord:
    """一条 QA 问答对的**构建视图**（只保留构建数据集必需的字段）。

    属性
    ----
    question_id : str
        题目 ID（``Data[].QuestionId``，现场实测形如 ``tc_1250``）。
    question : str
        问题文本（``Data[].Question``）。
    answer_values : Tuple[str, ...]
        **合并后**的答案字符串（保序去重）：``Answer.Value`` / ``Answer.NormalizedValue`` /
        ``Answer.Aliases`` / ``Answer.NormalizedAliases`` 四个来源。
        ``Answer.HumanAnswers`` **按口径不参与合并**（仅手工核对用）。
    answer_patterns : Tuple[re.Pattern, ...]
        由 ``answer_values`` 编译出的匹配模式（大小写不敏感 + 词边界 + 短数字额外保护）。
    answer_prefetch : frozenset
        **预筛** token 集合（由 :func:`answer_token_set` 得到，冻结集合以便直接求交）；
        仅用于跳过"必然不命中"的题目-文档对，判定仍由 ``answer_patterns`` 完成。
    answer_chars : frozenset
        答案的**字符级必要条件集合**（见 :func:`answer_char_set`）：文档字符集与它不相交时
        可以**安全**判定不命中（折叠安全，覆盖 ``ſ/ı/İ/ς/K`` 等 Unicode 陷阱）。
    answer_literals : Tuple[str, ...]
        答案的**小写**字面串（去重、保序、去掉纯空白）；用于"文档小写文本包含字面"这一
        微秒级预筛，进一步剪掉不可能命中的候选对（判定结果不受影响）。
    doc_members : Tuple[str, ...]
        正样本文档在归档中的成员名（``EntityPages ∪ SearchResults`` 映射后，升序去重）。
    numeric_answer_flag : float
        列 1029 的取值：答案集合中存在纯数字答案（strip 后 ``^[0-9]+$``）则 1.0，否则 0.0。

    关键不变量
    ----------
    * ``answer_values`` 非空（空答案无法判定，直接拒绝该题）；
    * ``answer_patterns`` 与 ``answer_values`` 等长（空串答案不产生模式）；
    * ``doc_members`` 升序且无重复。
    """

    question_id: str
    question: str
    answer_values: Tuple[str, ...]
    answer_patterns: Tuple[re.Pattern, ...]
    doc_members: Tuple[str, ...]
    numeric_answer_flag: float
    answer_prefetch: frozenset = frozenset()
    answer_literals: Tuple[str, ...] = ()
    answer_chars: frozenset = frozenset()


@dataclass(frozen=True)
class Sample:
    """一个 (Question, Evidence 文档) 样本的 provenance 记录。

    属性
    ----
    question_id : str
        题目 ID。
    doc_member : str
        文档在归档中的成员名（``evidence/wikipedia/<basename>`` 或 ``evidence/web/<相对路径>``）。
    doc_kind : str
        文档类别：``wikipedia`` / ``web``。
    label : int
        1 = 正样本（该题证据），0 = 负样本（同 split 内其他题的文档）。
    answer_hit : int
        该题答案在本样本文档中的命中情况（1/0）；**只作为 provenance 诊断写入 meta，
        不参与任何特征列**（列 1029 只由问题侧答案集合决定）。
    """

    question_id: str
    doc_member: str
    doc_kind: str
    label: int
    answer_hit: int


def _require_dict_field(obj: Any, key: str, where: str) -> Any:
    """从对象里取字段；缺失即报错（报文列出该对象的全部键，便于核对归档版本）。

    参数
    ----
    obj : Any
        待取值的对象（通常为 dict）。
    key : str
        字段名。
    where : str
        出错报文里的定位描述（如 ``Data[17]``）。

    返回
    ----
    Any
        字段值（可为 None）。

    异常
    ------
    BuildContractError
        ``obj`` 不是 dict，或缺少 ``key`` 时抛出。
    """
    if not isinstance(obj, dict):
        raise BuildContractError(
            f"{where} 不是对象（dict），而是 {type(obj).__name__}"
        )
    if key not in obj:
        raise BuildContractError(
            f"{where} 缺少字段 {key!r}；该对象实际含键 {sorted(obj.keys())}"
        )
    return obj[key]


def _require_str(obj: Any, key: str, where: str) -> str:
    """取一个**非空字符串**字段；类型不符或空串即报错。"""
    val = _require_dict_field(obj, key, where)
    if not isinstance(val, str) or not val.strip():
        raise BuildContractError(
            f"{where}.{key} 必须是非空字符串，实测 {type(val).__name__}={val!r}"
        )
    return val


def _wikipage_member(filename: str) -> str:
    """``EntityPages[].Filename`` -> 归档成员名 ``evidence/wikipedia/<basename>``。

    口径（现场实测 640 命中 / 0 未命中）：维基侧 evidence 是**扁平目录**，
    故对 Filename 取 basename 后拼 ``evidence/wikipedia/``。
    """
    base = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    if not base:
        raise BuildContractError(f"EntityPages[].Filename 为空：{filename!r}")
    return f"{EVIDENCE_DIR['wikipedia']}/{base}"


def _webpage_member(filename: str) -> str:
    """``SearchResults[].Filename`` -> 归档成员名 ``evidence/web/<该相对路径>``。

    口径（现场实测形如 ``158/158_2486.txt``）：web 侧 evidence 是**数字子目录**，
    故保留 Filename 的相对路径结构。
    """
    rel = str(filename).replace("\\", "/").strip("/")
    if not rel or rel.startswith("../") or "/../" in rel:
        raise BuildContractError(f"SearchResults[].Filename 不是合法相对路径：{filename!r}")
    return f"{EVIDENCE_DIR['web']}/{rel}"


def merge_answer_strings(answer_obj: Any, where: str) -> Tuple[str, ...]:
    """合并一条题目的答案字符串（**口径写死**）。

    来源（保序去重）::

        Answer.Value, Answer.NormalizedValue, Answer.Aliases[], Answer.NormalizedAliases[]

    ``Answer.HumanAnswers`` 按口径**不参与合并**（避免引入人工答案带来的口径漂移；
    它仍留在归档里供人工核对）。实测键集合还有 ``Type`` / ``MatchedWikiEntityName`` /
    ``NormalizedMatchedWikiEntityName``，同样不参与合并。

    参数
    ----
    answer_obj : Any
        ``Data[].Answer`` 对象。
    where : str
        出错报文里的定位描述。

    返回
    ----
    Tuple[str, ...]
        保序去重后的答案字符串（至少 1 条）。

    异常
    ------
    BuildContractError
        ``Answer`` 缺字段、类型不符，或合并后为空时抛出。
    """
    out: List[str] = []
    seen: Set[str] = set()

    def _push(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            if value not in seen:
                seen.add(value)
                out.append(value)

    for key in ("Value", "NormalizedValue"):
        _push(_require_dict_field(answer_obj, key, where))
    for key in ("Aliases", "NormalizedAliases"):
        val = _require_dict_field(answer_obj, key, where)
        if val is None:
            continue
        if not isinstance(val, (list, tuple)):
            raise BuildContractError(
                f"{where}.{key} 必须是数组，实测 {type(val).__name__}={val!r}"
            )
        for item in val:
            _push(item)
    if not out:
        raise BuildContractError(f"{where} 合并后没有任何非空答案字符串")
    return tuple(out)


def compile_answer_patterns(answer_values: Sequence[str]) -> Tuple[re.Pattern, ...]:
    """把答案字符串编译成**大小写不敏感 + 词边界**的匹配模式。

    口径
    ----
    * 词边界按答案首/末字符是否为词字符决定加不加 ``\\b``（对 ``C++`` 这类以非词字符
      结尾的答案，尾部 ``\\b`` 会导致**永远匹配不上**，故按需省略）；
    * **短数字答案额外保护**：``strip`` 后匹配 ``^[0-9]+$`` 且长度 ``<= SHORT_NUMERIC_MAX_LEN``
      （实测阈值 3）的答案，改用 ``(?<![\\w.])...(?![\\w.])`` —— 除词边界外**额外禁止与
      小数点连写**，从而防 ``3`` 命中 ``3.14``（``\\b`` 单独用时会命中）。

    参数
    ----
    answer_values : Sequence[str]
        答案字符串（含空串时跳过，不产生模式）。

    返回
    ----
    Tuple[re.Pattern, ...]
        编译好的模式元组（可能为空，如全部答案为空串）。
    """
    sources: List[str] = []
    for raw in answer_values:
        s = str(raw).strip()
        if not s:
            continue
        esc = re.escape(s)
        if NUMERIC_ONLY_RE.match(s) and len(s) <= int(SHORT_NUMERIC_MAX_LEN):
            sources.append(r"(?<![\w.])" + esc + r"(?![\w.])")
            continue
        prefix = r"\b" if WORD_HEAD_RE.match(s) else ""
        suffix = r"\b" if WORD_TAIL_RE.search(s) else ""
        sources.append(prefix + esc + suffix)
    if not sources:
        return ()
    # [!] 合并成**一条**等价正则：每条分支用 (?...) 分组以保证 `|` 的作用域只在本分支内，
    #     于是"任一分支命中"与"逐条 search 命中"逐位等价，但每篇文档只调用一次正则引擎。
    #     （这是 compact 路径能在 1e8 量级候选对上跑完的关键；判定结果不变。）
    return (re.compile("|".join("(?:" + s + ")" for s in sources), re.IGNORECASE),)


def answer_is_ascii(answer_values: Sequence[str]) -> bool:
    """答案是否**全为 ASCII**（决定能否用 token/字面预筛作为"否决条件"）。

    ``TOKEN_RE = [a-z0-9]+`` 只认 ASCII 小写，且 Python 的 lower/casefold 与
    ``re.IGNORECASE`` 的简单小写折叠在 ``ſ/ı/İ/ς/K`` 等字符上并不一致：故**非 ASCII 答案**
    一律不能靠 token/字面预筛否决（实测 3 组残余反例），只认字符级必要条件
    （:func:`doc_char_set` / :func:`answer_char_set`，对折叠做闭包，可证伪）。
    """
    for raw in answer_values:
        if not str(raw).isascii():
            return False
    return True


def answer_char_set(
    answer_values: Sequence[str], extra_literals: Sequence[str] = ()
) -> frozenset:
    """答案的**字符级必要条件集合**（保证"正则命中 => 集合相交"，与折叠实现无关）。

    取答案文本的全部字符，并按"大小写不敏感匹配"补上各字符的候选形态（见 :func:`_char_variants`）。
    若 ``re.IGNORECASE`` 的正则在某文档命中，则命中的那段文本与答案**逐字符对应**，故文档中
    必然出现答案的某个字符（或其候选形态）—— 因此"文档字符集 ∩ 答案字符集 = ∅"时可**安全**
    判定不命中。该判据不依赖 token 能否被抽出来（对 ``ı`` / ``ς`` 这类 ``[a-z0-9]`` 抽不出
    token 的字符同样成立），因此比 token 级预筛更稳健。

    参数
    ----
    answer_values : Sequence[str]
        合并后的答案字符串。
    extra_literals : Sequence[str]
        额外的字面（通常是已小写化的答案），一并纳入字符集。

    返回
    ----
    frozenset
        字符集合（含各字符的候选形态）。
    """
    chars: Set[str] = set()
    for raw in tuple(answer_values) + tuple(extra_literals):
        s = str(raw).strip()
        if not s:
            continue
        for ch in s:
            chars.update(_char_variants(ch))
    # [!] 对**多字符折叠结果**做闭包：如 ``'İ'.lower() == 'i̇'``（长度 2，第二字符是组合点），
    #     只收单字符形态会让 ``{'i'}`` 与 ``{'İ'}`` 判为不相交（实测陷阱）。把 lower/casefold/upper
    #     产出的每个字符也纳入集合即可闭合（该集合只用于剪枝，宁大勿小）。
    for ch in tuple(chars):
        for form in fold_forms(ch):
            chars.update(form)
    chars.discard("")
    return frozenset(chars)


def doc_char_set(text: str) -> frozenset:
    """文档的字符集合（与 :func:`answer_char_set` 同规则：含各字符的候选形态 + 折叠闭包）。"""
    chars: Set[str] = set()
    for ch in str(text):
        chars.update(_char_variants(ch))
    for ch in tuple(chars):
        for form in fold_forms(ch):
            chars.update(form)
    chars.discard("")
    return frozenset(chars)


def _variant_tokens(text: str) -> Set[str]:
    """文本的"逐字符候选形态展开"token 集合（token 级预筛用，折叠安全）。

    对每个字符取其候选形态（见 :func:`_char_variants`）按原位置展开成变体串，再抽 token；
    另并入常规分词产物。文档与答案两侧都用本函数，可显著提高"同一 token 的不同折叠形态"
    的命中率（例如文档 ``'ſ'`` 会同时产出 ``ſ`` 与 ``s``）。

    [!] 该集合只用于**加速**：折叠完备性由 :func:`answer_char_set` / :func:`doc_char_set` 的
    字符级必要条件保证，故本函数"多收不漏"即可。

    参数
    ----
    text : str
        原始文本。

    返回
    ----
    Set[str]
        token 集合（含各候选形态）。
    """
    # [!] 只做"整体形态"（lower/casefold/upper/原样）分词 + 单 token 级的字符变体展开：
    #     绝不逐字符笛卡尔展开整篇文档 —— 组合数会爆炸（实测触发 RecursionError）。
    #     完备性由字符级必要条件（answer_char_set / doc_char_set）负责，这里只求"多收不漏"。
    out: Set[str] = set()
    for form in fold_forms(text):
        for tok in TOKEN_RE.findall(form):
            out.add(tok)
    out.update(_ascii_variant_tokens(text))
    return out


def _token_letter_forms(text: str) -> Set[str]:
    """]下限分词器"的逐字符简单折叠 token（大写字母也算），用于折叠陷阱下的安全匹配。

    ``TOKEN_RE = [a-z0-9]+`` **只认小写**，故 ``'I'`` / ``'İ'`` 这类抽不出 token；本函数用
    "字母或数字"逐字符切分后再做逐字符简单折叠（见 :func:`fold_forms`），使 ``'I'`` / ``'İ'``
    都能归一到 ``'i'``，从而与文档侧同口径产物可比。
    """
    out: Set[str] = set()
    buf: List[str] = []
    for ch in str(text):
        if ch.isalnum():
            buf.append(ch)
        elif buf:
            out.add(_simple_fold("".join(buf)))
            buf = []
    if buf:
        out.add(_simple_fold("".join(buf)))
    out.discard("")
    return out


def _simple_fold(s: str) -> str:
    """逐字符简单折叠：每字符取单字符候选（lower/upper/casefold 中长度为 1 者）。"""
    parts: List[str] = []
    for ch in str(s):
        best = None
        for v in sorted(_char_variants(ch)):
            if len(v) == 1:
                best = v
        parts.append(best if best is not None else ch)
    return "".join(parts)


def _ascii_variant_tokens(token: str) -> Set[str]:
    """短 token 的"逐字符候选形态展开"产物（仅用于 token 级加速，规模很小）。"""
    if len(token) > 24:
        return set()
    opts = [sorted(_char_variants(ch)) for ch in token]
    if not opts or any(len(o) == 0 for o in opts):
        return set()
    out: Set[str] = set()

    def _walk(idx: int, parts: List[str]) -> None:
        if idx == len(opts):
            cand = "".join(parts)
            if TOKEN_RE.fullmatch(cand):
                out.add(cand)
            return
        for v in opts[idx]:
            parts.append(v)
            _walk(idx + 1, parts)
            parts.pop()

    _walk(0, [])
    return out


def _char_variants(ch: str) -> frozenset:
    """单个字符在"大小写不敏感匹配"下的**全部候选形态**（字符级，确定性、有缓存）。

    只取 ``lower`` / ``casefold`` / ``upper`` **三者**的并集（实现即为此，见下方函数体）。
    例如 ``'ſ'`` 三者给出 ``{'ſ','S','s'}``、``'ı'`` 给出 ``{'ı','I'}``，可覆盖 ``ſ/s``、``ı/i``、
    ``İ/i``、``ς/Σ``、``K/K`` 等常见大小写折叠陷阱。

    [!] 早期 docstring 曾写"另取 ``str.upper()`` 的逐字符备选映射（CPython 在 ``str.upper()`` 里
    暴露 ``_upper``）"，该表述**已删除**：实现从来只做三者并集；且对 **0..0x10FFFF 全部 1,114,112
    个码位**逐一扫描证实，``_upper`` 备选映射相对"三者并集"的**额外贡献为 0**，故该分支被彻底移除。

    参数
    ----
    ch : str
        单个字符。

    返回
    ----
    frozenset
        该字符在大小写不敏感匹配下可能等价的所有单字符形态。
    """
    cached = _CHAR_VARIANT_CACHE.get(ch)
    if cached is not None:
        return cached
    # [!] 只取 lower / upper / casefold 三种形态：**全部 Unicode 码位**逐个实测表明，
    #     CPython 的 ``str.upper()`` 已经覆盖了 alt-cased 字符（如 ``'ı'.upper()=='I'``、
    #     ``'ſ'.upper()=='S'``），故不再需要（也**不再使用**）解释器内部的逐字符备选映射
    #     —— 早期实现曾对 ``str._upper`` 字段做 ``eval``，但全码位扫描显示它 0 贡献，已删除。
    out: Set[str] = {ch, ch.lower(), ch.upper(), ch.casefold()}
    result = frozenset(x for x in out if x)
    _CHAR_VARIANT_CACHE[ch] = result
    return result


_CHAR_VARIANT_CACHE: Dict[str, frozenset] = {}


def canon_char(ch: str) -> str:
    """字符的**规范代表**（大小写等价类里挑一个确定性代表），用于 Aho-Corasick 折叠匹配。

    构造
    ----
    以 :func:`_char_variants` 为边做并查集闭包；**并把各折叠形态的"每个字符"也并入同一类**
    （例如 ``'İ'.lower() == 'i̇'`` 长度为 2，把 ``'i'`` 与组合点都并入，才能让 ``'İ'`` 与 ``'i'``
    同类）。代表字符的选法：优先取 ASCII 且小写者，其次 ASCII，最后按码位最小者。

    已证实的事实
    ------------
    * ``canon_char`` 的输出**恒为 1 个字符**（逐字符 1:1 映射 -> 折叠后文本与原文本位置一一对应，
      词边界仍可按原文判定）。该性质已对 **0..0x10FFFF 全部 1,114,112 个码位**逐一验证：非 1:1
      的码位数 = **0**。

    [!] 本函数**不宣称**任何"必要条件"性质（旧断言已删除，理由均为**实测**）：

    1. **"``re.IGNORECASE`` 等价 => 同类"不成立**：实测存在"``canon_char`` 不同但 ``re.IGNORECASE``
       单字符等价"的反例（本轮定向扫描确认 4 对，均涉 U+0345 COMBINING GREEK YPOGEGRAMMENI：
       ``U+0345`` ↔ ``U+1FBE`` / ``U+0399`` / ``U+03B9``）—— 这类字符的**多字符折叠形态**不闭合于
       1:1 映射。故"规范折叠后字面相等"只能当作**充分条件**（用于过报候选），**不能**当作必要条件
       （用于剪枝）。上述反例均为**希腊文组合记号**，不影响已验证的实务用例（英文字母/数字/常见标点）。
    2. **主导漏判机制在词边界编码，不在折叠层**：右边界被实现为"紧跟在核心之后的虚拟末字符"，
       而该虚拟字符只在**整段文本末尾**出现一次，故**只有核心恰好结束于文本末尾时才会被报出**。
       实测（``ans='cat'``，两侧词边界启用）：``'the cat'`` -> ``scan=[(0, 4, 7)]``（命中）、
       ``'cat'`` -> ``[(0, 0, 3)]``（命中）；而 ``'cat '`` / ``'cat-'`` / ``'a cat here'`` 虽正则
       真值均为 True，``scan`` **全为 []**（漏判）。同理 ``ans='s'``：``'x s'`` / ``'s'`` 命中，
       ``'x s y'`` / ``'s y'`` 漏判。
    """
    cached = _CANON_CACHE.get(ch)
    if cached is not None:
        return cached
    parent: Dict[str, str] = {ch: ch}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        if a not in parent:
            parent[a] = a
        if b not in parent:
            parent[b] = b
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    # [!] 必须做**不动点闭包**：变体关系要迭代展开（'ı' -> 'I' -> 'i'），
    #     只展开一跳会漏（实测 'ı' 与 'i' 被判为不同类 -> AC 漏判）。
    work = [ch]
    seen = {ch}
    while work:
        cur = work.pop()
        for v in _char_variants(cur):
            _union(ch, v)
            for c2 in (v, *tuple(v)):
                if c2 not in seen:
                    seen.add(c2)
                    work.append(c2)
    cls = [c for c in parent if _find(c) == _find(ch)]
    ascii_cands = sorted(c for c in cls if c.isascii() and len(c) == 1)
    lower_cands = [c for c in ascii_cands if c.islower() or not c.isalpha()]
    rep_ch = (lower_cands or ascii_cands or sorted(cls))[0]
    for c in cls:
        _CANON_CACHE[c] = rep_ch
    return rep_ch


_CANON_CACHE: Dict[str, str] = {}


def canon_fold(text: str) -> str:
    """把整段文本折叠为规范形态（**逐字符 1:1**，长度与位置保持不变）。"""
    return "".join(canon_char(ch) for ch in str(text))


def fold_forms(text: str) -> Tuple[str, ...]:
    """文本用于预筛的**归一化形态集合**（至少含 ``lower()`` 与 ``casefold()``）。

    必要性保证：若 ``re.IGNORECASE`` 的正则在文档位置 ``i`` 命中 ``lit``，则按"逐字符取
    候选形态"的规则，``lit`` 的每个字符都有一种形态与文档对应字符的某形态相等；把这些形态
    分别拼成串即可在文档的某个归一化形态里找到 ``lit`` 的某个归一化形态（子串级）。故
    预筛判否时正则必然不命中 —— 这就是 :func:`_answer_prefilter_pass` / :func:`answer_char_set`
    能"安全剪枝"的依据（含 ``ſ/ı/İ/ς/K`` 等 Unicode 折叠陷阱）。

    参数
    ----
    text : str
        原始文本。

    返回
    ----
    Tuple[str, ...]
        归一化形态（去重保序）。
    """
    forms: List[str] = []
    for form in (str(text), str(text).lower(), str(text).casefold(), str(text).upper()):
        if form not in forms:
            forms.append(form)
    # [!] 再补一个"**逐字符简单折叠**"形态：每个字符取其单字符候选（lower/upper/casefold 里
    #     长度仍为 1 的结果）。这一步专治"整体 lower/casefold 会变成多字符"的折叠陷阱 ——
    #     如 ``'İ'.lower() == 'i̇'``（长度 2）、``'İ'.casefold() == 'i̇'``，于是 ``'i'`` 在
    #     常规三形态里**永远找不到**；逐字符简单折叠会给出单字符 ``'i'``，让字面层/ token 层
    #     在 ``İ ↔ i`` 这类对上不再误剪（实测 R32 的 D12 残余反例即此）。
    simple = "".join(
        next(
            (
                v
                for v in ("".join(sorted(_char_variants(ch))))
                if len(v) == 1
            ),
            ch,
        )
        for ch in str(text)
    )
    if simple not in forms:
        forms.append(simple)
    return tuple(forms)


def char_variant_tokens(text: str) -> Set[str]:
    """文本的"逐字符候选形态展开"产物 token 集合（用于 token 级预筛，折叠安全）。

    对每个字符取其候选形态（见 :func:`_char_variants`），再取所有形态的 token 并集。
    对**文档**与**答案**两侧都用本函数生成 token 集合，则可保证：正则命中 => 两侧 token
    集合相交（``ſ`` 侧会同时产出 ``ſ`` 与 ``s``，``ı`` 侧会同时产出 ``ı`` 与 ``I``/``i``）。

    参数
    ----
    text : str
        原始文本。

    返回
    ----
    Set[str]
        token 集合（含各候选形态）。
    """
    out: Set[str] = set()
    for ch in text:
        for v in _char_variants(ch):
            if TOKEN_RE.fullmatch(v):
                out.add(v)
    for form in fold_forms(text):
        out.update(TOKEN_RE.findall(form))
    return out


def answer_token_set(answer_values: Sequence[str]) -> Set[str]:
    """答案集合的**廉价预筛** token 集合（只用于剪枝，不作判定）。

    口径：对每个非空答案，取 (a) ``lower`` / ``casefold`` 等形态下 ``[a-z0-9]+`` 分词的
    **前 8 个 token**（含 1 字符 token，如答案 ``'e'`` -> ``{'e'}``；每个形态各取 8 个），
    (b) :func:`_token_letter_forms` 的"下限分词"产物（**无数量上限**，用于大写/折叠陷阱字符），
    以及 (c) 若答案 strip 后匹配 ``^[0-9]+$`` 则把该数字串也放进集合。
    故集合大小不设硬上界——它只用于剪枝加速，**折叠完备性由字符级判据负责**（见
    :func:`answer_char_set` / :func:`_answer_prefilter_pass`）。

    [!] 必须**保留 1 字符 token** 且**不得丢弃**：预筛是"答案出现在文档里"的必要条件，
    丢掉短 token 会把"文档里恰好有独立的 e"这类对误剪（实测让 wiki 的
    ``negative_answer_hit`` 从 31 变 30）。

    为什么不用 ``tokenize``：``doc_contains_answer`` 是**大小写不敏感**的，故不能用小写化后的
    文档集合去否决一个大写答案（会漏判）；这里的集合只包含**必然以原样出现在答案里**的小写
    token 与数字串，文档侧按同一规则抽取——命中该集合是"答案可能出现在文档里"的**必要条件**，
    故可安全用于剪枝（剪掉的必然不命中）。

    参数
    ----
    answer_values : Sequence[str]
        合并后的答案字符串。

    返回
    ----
    Set[str]
        预筛 token 集合（可能为空，表示无法预筛、必须走正则）。
    """
    out: Set[str] = set()
    for raw in answer_values:
        s = str(raw).strip()
        if not s:
            continue
        # [!] 必须**保留 1 字符 token**（如答案 'e'）：预筛是"答案出现在文档里"的必要条件，
        #     丢掉短 token 会把 "文档里恰好有独立的 e" 这类误剪（实测正是它让 wiki 的
        #     negative_answer_hit 从 31 变 30）。多出的短 token 只让索引略大，不影响判定。
        # [!] 取**全部候选形态**的 token（Unicode 折叠安全，见 fold_forms / _variant_tokens）。
        for form in fold_forms(s):
            out.update(TOKEN_RE.findall(form)[:8])
        out.update(_token_letter_forms(s))
        if NUMERIC_ONLY_RE.match(s):
            out.add(s)
    return out


def doc_contains_answer(doc_text: str, patterns: Sequence[re.Pattern]) -> bool:
    """文档是否含该题答案（任一模式命中即为真）。

    参数
    ----
    doc_text : str
        文档全文。
    patterns : Sequence[re.Pattern]
        由 :func:`compile_answer_patterns` 编译的模式。

    返回
    ----
    bool
        是否命中。
    """
    for p in patterns:
        if p.search(doc_text) is not None:
            return True
    return False


def parse_qa_json(raw: bytes, split: str) -> Tuple[List[QaRecord], Dict[str, Any]]:
    """解析一个 split 的 QA JSON，产出 **按 QuestionId 升序**排列的构建视图。

    归档版本事实（现场枚举）
    ------------------------
    顶层键：``Data`` / ``Domain`` / ``Split`` / ``VerifiedEval`` / ``Version``；
    ``Data[].`` 键：``Answer`` / ``EntityPages`` / ``Question`` / ``QuestionId`` /
    ``QuestionPartOfVerifiedEval`` / ``QuestionSource`` / ``QuestionVerifiedEvalAttempt``
    （web split 另有 ``SearchResults``）。

    参数
    ----
    raw : bytes
        QA JSON 原始字节。
    split : str
        ``wiki`` / ``web``（仅用于报文与条数校验）。

    返回
    ----
    Tuple[List[QaRecord], Dict[str, Any]]
        ``(记录列表（QuestionId 升序）, 头部信息摘要)``。

    异常
    ------
    BuildContractError
        JSON 非法 / 顶层缺键 / 题目条数与实测口径不符 / 单题字段缺失时抛出。
    """
    try:
        obj = json.loads(raw.decode(TEXT_ENCODING))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildContractError(
            f"{QA_JSON_MEMBERS.get(split, split)} 不是合法 JSON：{exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise BuildContractError(
            f"{QA_JSON_MEMBERS.get(split, split)} 顶层必须是对象，实测 {type(obj).__name__}"
        )
    for key in ("Data", "Domain", "Split", "Version"):
        _require_dict_field(obj, key, f"{split} 顶层对象")
    data = obj["Data"]
    if not isinstance(data, list):
        raise BuildContractError(f"{split} 的 Data 必须是数组，实测 {type(data).__name__}")
    expect_n = QA_JSON_QUESTIONS.get(split)
    if expect_n is not None and len(data) != int(expect_n):
        raise BuildContractError(
            f"{split} 的题目条数与实测口径不符：实测 {len(data)} 条，写死常量 {int(expect_n)} 条"
        )

    records: List[QaRecord] = []
    n_no_doc = 0
    for i, q in enumerate(data):
        where = f"Data[{i}]"
        qid = _require_str(q, "QuestionId", where)
        question = _require_str(q, "Question", where)
        answers = merge_answer_strings(
            _require_dict_field(q, "Answer", where), f"{where}.Answer"
        )
        members: List[str] = []
        entity_pages = _require_dict_field(q, "EntityPages", where) or []
        if not isinstance(entity_pages, (list, tuple)):
            raise BuildContractError(
                f"{where}.EntityPages 必须是数组，实测 {type(entity_pages).__name__}"
            )
        for j, ep in enumerate(entity_pages):
            members.append(_wikipage_member(_require_str(ep, "Filename", f"{where}.EntityPages[{j}]")))
        search_results = q.get("SearchResults") or []
        if not isinstance(search_results, (list, tuple)):
            raise BuildContractError(
                f"{where}.SearchResults 必须是数组，实测 {type(search_results).__name__}"
            )
        for j, sr in enumerate(search_results):
            members.append(
                _webpage_member(_require_str(sr, "Filename", f"{where}.SearchResults[{j}]"))
            )
        if not members:
            n_no_doc += 1
        numeric_flag = 1.0 if any(NUMERIC_ONLY_RE.match(a.strip()) for a in answers) else 0.0
        records.append(
            QaRecord(
                question_id=qid,
                question=question,
                answer_values=answers,
                answer_patterns=compile_answer_patterns(answers),
                doc_members=tuple(sorted(set(members))),
                numeric_answer_flag=numeric_flag,
                answer_prefetch=frozenset(answer_token_set(answers)),
                answer_literals=tuple(
                    dict.fromkeys(
                        str(a).strip().lower() for a in answers if str(a).strip()
                    )
                ),
                answer_chars=answer_char_set(answers),
            )
        )
    # 确定性口径：全部下游逻辑（负采样池、行序、截断）都基于 QuestionId 升序
    records.sort(key=lambda r: r.question_id)
    ids = [r.question_id for r in records]
    if len(set(ids)) != len(ids):
        dup = sorted({x for x in ids if ids.count(x) > 1})
        raise BuildContractError(f"{split} 的 QuestionId 存在重复：{dup[:10]}")
    head = {
        "qa_member": QA_JSON_MEMBERS.get(split, ""),
        "domain": str(obj.get("Domain")),
        "qa_split": str(obj.get("Split")),
        "version": float(obj.get("Version", 0.0)),
        "verified_eval": bool(obj.get("VerifiedEval", False)),
        "questions": len(records),
        "questions_without_evidence_ref": int(n_no_doc),
    }
    return records, head


# ======================================================================
# 稀疏文本特征层（哈希词袋 + 覆盖度/长度/答案类型）
# ======================================================================
def tokenize(text: str) -> List[str]:
    """分词（**口径写死**）：先整体小写，再按 ``[a-z0-9]+`` 抽 token（含重数）。

    该口径对非 ASCII 字符一律丢弃（与"证据文本里存在 U+FFFD 替换字符"这一实测现象无关，
    因为替换字符本就不落在 ``[a-z0-9]`` 内）。

    参数
    ----
    text : str
        待分词文本。

    返回
    ----
    List[str]
        token 序列（保留重数与出现顺序，供 ``log1p(token 数)`` 使用）。
    """
    return TOKEN_RE.findall(str(text).lower())


def hash_bucket(token: str, hash_dim: int = HASH_DIM) -> int:
    """单个 token 的确定性哈希桶（``blake2b(salt + token) mod hash_dim``）。

    参数
    ----
    token : str
        token 字符串。
    hash_dim : int
        桶数（缺省写死值 :data:`HASH_DIM` = 1024）。

    返回
    ----
    int
        ``[0, hash_dim)`` 内的桶号。
    """
    digest = hashlib.blake2b(
        HASH_SALT + token.encode(TEXT_ENCODING), digest_size=int(HASH_DIGEST_SIZE)
    ).digest()
    return int.from_bytes(digest, "big") % int(hash_dim)


def hash_bag(tokens: Sequence[str], hash_dim: int = HASH_DIM) -> np.ndarray:
    """把 token 序列打成 **hash_dim 维哈希词袋计数**（不做归一化）。

    实现要点
    --------
    * 逐 token 做 blake2b（确定性、不消耗全局 RNG），用 ``np.bincount`` 一次性计数；
    * 返回 ``float32``，取值即该桶内的 token **出现次数**（重数口径）；
    * 词袋是**可加**的：``bag(q + '\\n' + d) == bag(q) + bag(d)``（``\\n`` 不产生 token），
      因此文档侧词袋可按文档缓存复用。

    参数
    ----
    tokens : Sequence[str]
        token 序列。
    hash_dim : int
        桶数（缺省写死值 :data:`HASH_DIM` = 1024）。

    返回
    ----
    np.ndarray
        形状 ``[hash_dim]`` 的 float32 计数向量。
    """
    # [!] compact 路径下文档 token 是**整数标识数组**（不是 str）：此时"落桶"按标识取模，
    #     与 str 路径的 ``hash_bucket`` 语义不同但**同源**——标识本身就是
    #     ``blake2b(salt+token, digest_size=8)`` 的 64bit 整数，故 ``id % hash_dim`` 与
    #     ``hash_bucket(token, hash_dim)`` 对同一 token 给出同一个桶。两条路径因此等价。
    if isinstance(tokens, np.ndarray) and tokens.dtype.kind in "iu":
        if tokens.size == 0:
            return np.zeros(int(hash_dim), dtype=np.float32)
        buckets = np.asarray(tokens, dtype=np.uint64) % np.uint64(int(hash_dim))
        counts = np.bincount(buckets.astype(np.int64), minlength=int(hash_dim))
        return counts.astype(np.float32)
    if not tokens:
        return np.zeros(int(hash_dim), dtype=np.float32)
    buckets = np.fromiter(
        (hash_bucket(t, hash_dim) for t in tokens), dtype=np.int64, count=len(tokens)
    )
    counts = np.bincount(buckets, minlength=int(hash_dim))
    return counts.astype(np.float32)


def l2_normalize_block(vec: np.ndarray) -> np.ndarray:
    """对哈希词袋块做 L2 归一化（范数为 0 时原样返回全 0，不产生 NaN）。

    参数
    ----
    vec : np.ndarray
        一维 ``float32`` 向量（原地归一化并返回）。

    返回
    ----
    np.ndarray
        归一化后的同一数组。
    """
    norm = float(np.sqrt(np.dot(vec, vec)))
    if norm > 0.0:
        vec /= np.float32(norm)
    return vec


# ======================================================================
# rich 特征层（IDF 加权覆盖度 / TF-IDF 余弦 / 数值型答案占比）
#
# 设计口径（确定性、不消耗全局 RNG、不整包解压）
# ----------------------------------------------------------------
# * IDF 语料 = **一次构建内该 split 的全部文档池**（`build_meta` 的 documents 定义），
#   平滑式 `idf(token) = log((1 + M) / (1 + df(token))) + 1`；
# * token 在 rich 层被映射为 **8 字节 blake2b 摘要的整数值**（``hash_token_id``）：
#   8 字节 = 64 bit -> 恰好可存进定长 ``int64``/``uint64`` 数组（``np.fromiter``/``searchsorted``
#   全程向量化）；vocab 规模 1e7 时按生日问题估计的碰撞对数约 `n^2 / 2^65 ≈ 2.7e-6` 对，
#   影响面 = 把两个极稀有 token 当成同一个（只轻微改变 IDF 权重），远小于特征本身的分辨率。
#   （12 字节摘要的 96 bit 整数放不进 int64 定点数组，会让 token 计数与 IDF 查询退化成纯 Python，
#     故取 8 字节 + 定长数组的工程折中；该口径写在 meta.features.idf.token_key 里可回读。）
# * 所有 per-document 聚合量（`idf_sum` / `idf_norm` / `token_count`）与逐对量
#   （交集 IDF 和、TF-IDF 点积）都由 **同一个 token→计数表** 派生，
#   与 in-memory 路径（直接用 token 字符串集合）在数学上同源。
# ======================================================================
HASH_TOKEN_ID_BYTES: int = 8

# 单个文本进入 in-memory 缓存（tokens/sets/counts）的 token 数上限；超过则只保留集合与计数，
# 供 token 规模巨大（web-dev）时兜底（不影响特征取值，只影响缓存驻留量）。
CACHE_TOKEN_LIST_LIMIT: int = 2000000


def hash_token_id(token: str) -> int:
    """token -> **8 字节** blake2b 摘要的整数值（rich 层的 token 标识）。

    参数
    ----
    token : str
        分词后的 token（``[a-z0-9]+``，已小写）。

    返回
    ----
    int
        ``int.from_bytes(blake2b(salt + token, digest_size=HASH_TOKEN_ID_BYTES).digest(), "big")``，
        取值范围 ``[0, 2**64)``（``HASH_TOKEN_ID_BYTES = 8``：恰好可放进定长 ``uint64`` 数组，
        使 df 统计与 IDF 查询全程向量化；碰撞量级见模块头部的口径说明）。
    """
    digest = hashlib.blake2b(
        HASH_SALT + token.encode(TEXT_ENCODING), digest_size=int(HASH_TOKEN_ID_BYTES)
    ).digest()
    return int.from_bytes(digest, "big")


def _term_counts(token_ids: Sequence[int]) -> Dict[int, int]:
    """token 序列 -> ``{token_id: 出现次数}``（保留重数，供 TF-IDF 使用）。

    参数
    ----
    token_ids : Sequence[int]
        token 标识序列（含重数）。

    返回
    ----
    Dict[int, int]
        计数表。
    """
    counts: Dict[int, int] = {}
    for t in token_ids:
        counts[t] = counts.get(t, 0) + 1
    return counts


def _token_counts_from_cache(cache: "TextFeatureCache", key: str) -> Dict[int, int]:
    """取某个已缓存文本的 ``{token_id: 次数}`` 计数表（缺失则报错）。

    参数
    ----
    cache : TextFeatureCache
        文本缓存（必须已 :meth:`TextFeatureCache.ensure` 过 ``key``）。
    key : str
        缓存键。

    返回
    ----
    Dict[int, int]
        计数表。

    异常
    ------
    BuildContractError
        ``key`` 不在缓存中时抛出。
    """
    if key not in cache.counts:
        raise BuildContractError(f"缓存中没有 {key!r} 的 token 计数（未 ensure 或已被淘汰）")
    return cache.counts[key]


class IdfTable:
    """IDF 表：以 **token 标识整数**为键的确定性倒文档频率查询表。

    不变量
    ------
    * ``token_ids`` 升序且唯一；``idf`` 与之等长；
    * ``idf[i] = log((1 + n_docs) / (1 + df(token_ids[i]))) + IDF_SMOOTH_OFFSET > 0``；
    * 查询未登记 token 时返回 :data:`IDF_SMOOTH_OFFSET`（即 df = 0 时的取值），
      故 IDF 永远为正、可用作加权权重（不会出现权重为 0 导致覆盖率恒 0 的退化）。
    """

    def __init__(self, df: Dict[int, int], n_docs: int) -> None:
        """由 ``{token_id: df}`` 与文档总数构造 IDF 表（键升序，确定性）。

        参数
        ----
        df : Dict[int, int]
            ``{token_id: 含该 token 的文档数}``。
        n_docs : int
            语料文档数 ``M``（> 0）。
        """
        if int(n_docs) <= 0:
            raise BuildContractError(f"IDF 语料的文档数必须 > 0，实测 {n_docs}")
        self.n_docs = int(n_docs)
        keys = sorted(df.keys())
        self.token_ids = np.asarray(keys, dtype=np.uint64)
        if keys:
            dfs = np.asarray([int(df[k]) for k in keys], dtype=np.float64)
            self.idf = np.log((1.0 + float(n_docs)) / (1.0 + dfs)) + float(IDF_SMOOTH_OFFSET)
        else:
            self.idf = np.zeros(0, dtype=np.float64)
        # 缺省权重 = df 取 0 时的 IDF（保证查询未登记 token 时权重仍为正）
        self._default_idf: float = float(np.log(1.0 + float(n_docs)) + float(IDF_SMOOTH_OFFSET))

    @property
    def size(self) -> int:
        """语料中登记的**不同 token 数**（vocab 规模）。"""
        return int(self.token_ids.size)

    def default_idf(self) -> float:
        """未登记 token 的 IDF（= df 取 0 的取值）。"""
        return float(self._default_idf)

    def lookup(self, keys: np.ndarray) -> np.ndarray:
        """批量查询 IDF（未登记的键取 :meth:`default_idf`）。

        参数
        ----
        keys : np.ndarray
            一维 int64 的 token 标识数组。

        返回
        ----
        np.ndarray
            与 ``keys`` 等长的 float64 IDF 数组。
        """
        if keys.size == 0:
            return np.zeros(0, dtype=np.float64)
        k = np.asarray(keys, dtype=np.uint64)
        pos = np.searchsorted(self.token_ids, k)
        pos_clipped = np.clip(pos, 0, max(int(self.token_ids.size) - 1, 0))
        hit = (self.token_ids.size > 0) & (pos < int(self.token_ids.size))
        out = np.full(keys.shape, float(self._default_idf), dtype=np.float64)
        if np.any(hit):
            hit = hit & (self.token_ids[pos_clipped] == k)
        if np.any(hit):
            out[hit] = self.idf[pos_clipped[hit]]
        return out


class DocPack:
    """``compact`` 规模 split 的**紧凑文档表示**（唯一 token 标识 + 重数，落内存映射文件）。

    布局（写入 ``<out>_docpack.npy.npz``）
    ------------------------------------
    * ``uniq_ids``：全部文档的**唯一** token 标识按文档顺序拼接（int64，每文档内升序）；
    * ``uniq_counts``：同上位置处的 token 重数（int64，正整数），与 ``uniq_ids`` 一一对应；
    * ``offsets`` / ``sizes``：第 i 个文档在 ``uniq_ids`` 中的起点与长度。

    ``nbytes`` / ``ntokens`` 为总量统计（供 meta 与日志留档）。
    """

    def __init__(
        self,
        uniq_ids: np.ndarray,
        uniq_counts: np.ndarray,
        offsets: np.ndarray,
        sizes: np.ndarray,
    ) -> None:
        """由四段数组直接构造（``uniq_ids`` 与 ``uniq_counts`` 等长）。"""
        self.uniq_ids = uniq_ids
        self.uniq_counts = uniq_counts
        self.offsets = offsets
        self.sizes = sizes
        self.nbytes = int(sum(int(a.nbytes) for a in (uniq_ids, uniq_counts, offsets, sizes)))
        if self.uniq_ids.dtype != np.uint64:
            raise BuildContractError(
                f"DocPack 的 token 标识必须是 uint64（避免 64bit 标识被截断为负数），实测 {self.uniq_ids.dtype}"
            )
        self.ntokens = int(np.asarray(uniq_counts, dtype=np.int64).sum())

    @staticmethod
    def load(path: str) -> "DocPack":
        """从 npz 读回（用 ``mmap_mode="r"`` 打开成员，不整包进内存）。

        参数
        ----
        path : str
            ``*_docpack.npy.npz`` 路径。

        返回
        ----
        DocPack
            读回的对象（数组为内存映射视图）。
        """
        with np.load(path, allow_pickle=False, mmap_mode="r") as z:
            return DocPack(
                np.asarray(z["uniq_ids"]),
                np.asarray(z["uniq_counts"]),
                np.asarray(z["offsets"]),
                np.asarray(z["sizes"]),
            )

    def document(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """取第 ``index`` 个文档的 ``(唯一 token 标识, 对应重数)``（独立拷贝，文档内升序）。

        参数
        ----
        index : int
            文档序号（与 ``meta["documents"]`` 同序）。

        返回
        ----
        Tuple[np.ndarray, np.ndarray]
            ``(token_ids, counts)``；该文档无 token 时返回两个空数组。
        """
        off = int(self.offsets[int(index)])
        size = int(self.sizes[int(index)])
        if size <= 0:
            return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64)
        return (
            np.array(self.uniq_ids[off : off + size], dtype=np.uint64),
            np.array(self.uniq_counts[off : off + size], dtype=np.int64),
        )


class DocPackBuilder:
    """``DocPack`` 的增量构造器（按文档顺序追加，最后一次性落盘）。

    为控制内存，每次 ``flush`` 只把**已追加**的 token 数组拼接后写入一个分片文件，
    返回落地字节数；``finalize`` 再把分片合并成一个 npz（合并用 numpy 的数组拼接，
    不经过 Python 对象层）。
    """

    def __init__(self, out_path: str, flush_bytes: int = 128 << 20) -> None:
        """初始化构造器。

        参数
        ----
        out_path : str
            目标 ``*_docpack.npy.npz`` 路径。
        flush_bytes : int
            单个分片的字节上限（超过即 flush，控制峰值内存）。
        """
        self.out_path = out_path
        self.flush_bytes = int(flush_bytes)
        self._chunks: List[Tuple[np.ndarray, np.ndarray]] = []
        self._nbytes = 0
        self._shards: List[str] = []
        self._offsets: List[int] = []
        self._sizes: List[int] = []
        self._tok_total = 0

    def add(self, token_ids: np.ndarray, counts: np.ndarray) -> None:
        """追加一个文档（``counts`` 为该文档内各 token 的重数）。

        参数
        ----
        token_ids : np.ndarray
            该文档的 token 标识（含重数）。
        counts : np.ndarray
            与 ``token_ids`` 等长。
        """
        # [!] 标识是 64bit 无符号摘要（可能 >= 2**63）：全程保持 uint64，
        #     绝不能 cast 成 int64（那会把一半标识变成负数，后续 uint64 转换直接溢出）。
        t = np.asarray(token_ids, dtype=np.uint64)
        c = np.asarray(counts, dtype=np.int64)
        if t.shape != c.shape:
            raise BuildContractError(f"DocPack 追加时形状不一致：{t.shape} vs {c.shape}")
        self._offsets.append(int(self._tok_total))
        self._sizes.append(int(t.size))
        self._tok_total += int(t.size)
        self._chunks.append((t, c))
        self._nbytes += int(t.nbytes + c.nbytes)
        if self._nbytes >= self.flush_bytes:
            self.flush()

    def flush(self) -> int:
        """把已累积的文档拼接后写入一个分片文件，返回落地字节数（0 表示无内容）。"""
        if not self._chunks:
            return 0
        tok = np.concatenate([c[0] for c in self._chunks])
        cnt = np.concatenate([c[1] for c in self._chunks])
        self._chunks = []
        self._nbytes = 0
        shard = self.out_path + f".part{len(self._shards)}.npy"
        os.makedirs(os.path.dirname(os.path.abspath(shard)), exist_ok=True)
        np.save(shard, tok, allow_pickle=False)
        np.save(shard + ".counts.npy", cnt, allow_pickle=False)
        self._shards.append(shard)
        return int(tok.nbytes + cnt.nbytes)

    def finalize(self) -> DocPack:
        """合并全部分片并落盘为一个 npz，返回 :class:`DocPack`。

        返回
        ----
        DocPack
            最终对象；分片文件在合并后删除。
        """
        self.flush()
        def _cat(name: str, dtype: Any) -> np.ndarray:
            """把各分片按顺序拼成一整块（**按落盘 dtype 分配**，绝不做有损转换）。

            [!] 必须显式指定 dtype：``token_ids`` 是 uint64，若按 int64 分配会把 ≥2^63 的
            标识变成负数（DocPack 的 uint64 断言随即抛错，多分片全量构建必崩）。
            """
            arrs = [np.load(s + name, mmap_mode="r", allow_pickle=False) for s in self._shards]
            if not arrs:
                return np.zeros(0, dtype=dtype)
            if len(arrs) == 1:
                return np.array(arrs[0], dtype=dtype)
            out = np.empty(sum(int(a.size) for a in arrs), dtype=dtype)
            cursor = 0
            for a in arrs:
                out[cursor : cursor + int(a.size)] = a
                cursor += int(a.size)
            return out

        tok = _cat("", np.uint64)
        cnt = _cat(".counts.npy", np.int64)
        offsets = np.asarray(self._offsets, dtype=np.int64)
        sizes = np.asarray(self._sizes, dtype=np.int64)
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        tmp = self.out_path + ".tmp"
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for name, arr in (
                ("uniq_ids.npy", tok),
                ("uniq_counts.npy", cnt),
                ("offsets.npy", offsets),
                ("sizes.npy", sizes),
            ):
                buf = io.BytesIO()
                np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
                info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 0
                info.external_attr = 0
                zf.writestr(info, buf.getvalue())
        os.replace(tmp, self.out_path)
        for s in self._shards:
            for suffix in ("", ".counts.npy"):
                try:
                    os.remove(s + suffix)
                except OSError:  # pragma: no cover - 分片可能已被手工清理
                    pass
        self._shards = []
        return DocPack(tok, cnt, offsets, sizes)


class DocPackAccessor:
    """``DocPack`` 的**按文档缓存**访问器（解包后的紧凑表示，LRU 上限控制驻留量）。"""

    def __init__(self, pack: DocPack, max_docs: int = 8) -> None:
        """初始化访问器。

        参数
        ----
        pack : DocPack
            紧凑文档表示。
        max_docs : int
            最多驻留多少个已解包文档（超出按插入序淘汰）。
        """
        self.pack = pack
        self.max_docs = max(1, int(max_docs))
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def get(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """取第 ``index`` 个文档的 ``(token_ids, counts)``（带缓存）。

        参数
        ----
        index : int
            文档序号。

        返回
        ----
        Tuple[np.ndarray, np.ndarray]
            解包后的 ``(token_ids, counts)``。
        """
        key = int(index)
        item = self._cache.get(key)
        if item is None:
            item = self.pack.document(key)
            if len(self._cache) >= self.max_docs:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = item
        return item


class BagCache:
    """哈希词袋块的**有界缓存**（避免同一文档的多行重复哈希）。

    不变量：缓存的数组**只读**，调用方必须对取出的数组做 ``copy`` 后再原地归一化
    （否则缓存值会被就地改写）。哈希为 blake2b 确定性运算，与 RNG 无关。
    """

    def __init__(self, hash_dim: int, max_items: int = 64) -> None:
        """初始化缓存。

        参数
        ----
        hash_dim : int
            词袋块维数（0 表示不产出词袋块，此时本缓存不生效）。
        max_items : int
            最多缓存多少个文档的词袋块。
        """
        self.hash_dim = int(hash_dim)
        self.max_items = max(1, int(max_items))
        self._cache: Dict[Tuple[Any, int, int, str], np.ndarray] = {}

    def get(self, cache_key: Any, tokens: Sequence[str]) -> np.ndarray:
        """取某个文本的词袋块（命中则直接返回缓存数组，**只读**，调用方不得就地改写）。

        参数
        ----
        cache_key : Any
            调用方给的唯一标识（文档用文档序号、问题用问题缓存键 ``q:<QuestionId>``）。
        tokens : Sequence[str]
            该文本的 token 序列（含重数）。

        返回
        ----
        np.ndarray
            形状 ``[hash_dim]`` 的 float32 计数向量（未归一化）。
        """
        n_tok = int(len(tokens))
        if n_tok > 0:
            first = tokens[0]
            last = tokens[-1]
            key: Tuple[Any, int, int, Any] = (cache_key, n_tok, int(id(first)), last)
        else:
            key = (cache_key, 0, 0, "")
        arr = self._cache.get(key)
        if arr is None:
            arr = hash_bag(tokens, self.hash_dim)
            if len(self._cache) >= self.max_items:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = arr
        return arr


def _idf_weights(keys: Sequence[int], idf: IdfTable) -> Tuple[float, float, float]:
    """一组 token 标识的 ``(idf 之和, idf 平方和, 该组自身的 L2 范数)``。

    参数
    ----
    keys : Sequence[int]
        token 标识（**唯一**即可，重数在 TF-IDF 处另行加权）。
    idf : IdfTable
        IDF 表。

    返回
    ----
    Tuple[float, float, float]
        ``(sum(idf), sum(idf^2), sqrt(sum(idf^2)))``；空组返回 ``(0, 0, 0)``。
    """
    if not keys:
        return 0.0, 0.0, 0.0
    w = idf.lookup(np.asarray(keys, dtype=np.uint64))
    s = float(w.sum())
    q = float(np.dot(w, w))
    return s, q, float(np.sqrt(q))


def _idf_intersection_sum(
    q_counts: Dict[int, int], d_counts: Dict[int, int], idf: IdfTable
) -> float:
    """两个 token→计数表的**交集 IDF 和**（按升序 token 标识确定性累加）。

    参数
    ----
    q_counts : Dict[int, int]
        问题侧计数表。
    d_counts : Dict[int, int]
        文档侧计数表。
    idf : IdfTable
        IDF 表。

    返回
    ----
    float
        ``sum_{t in 交集} idf(t)``（交集为空时 0.0）。
    """
    if len(d_counts) < len(q_counts):
        small, big = d_counts, q_counts
    else:
        small, big = q_counts, d_counts
    keys = sorted(k for k in small if k in big)
    if not keys:
        return 0.0
    return float(idf.lookup(np.asarray(keys, dtype=np.uint64)).sum())


def _tfidf_dot(
    q_counts: Dict[int, int], d_counts: Dict[int, int], idf: IdfTable
) -> float:
    """TF-IDF 点积 ``sum_{t in 交集} (tf_q * idf) * (tf_d * idf)``（升序累加，确定性）。

    参数
    ----
    q_counts : Dict[int, int]
        问题侧 计数表。
    d_counts : Dict[int, int]
        文档侧计数表。
    idf : IdfTable
        IDF 表。

    返回
    ----
    float
        点积（交集为空时 0.0）。
    """
    if len(d_counts) < len(q_counts):
        small, big = d_counts, q_counts
    else:
        small, big = q_counts, d_counts
    keys = sorted(k for k in small if k in big)
    if not keys:
        return 0.0
    arr = np.asarray(keys, dtype=np.uint64)
    w = idf.lookup(arr)
    prod = np.asarray([int(small[k]) * int(big[k]) for k in keys], dtype=np.float64)
    return float(np.dot(prod, w * w))


def _tfidf_norm(counts: Dict[int, int], idf: IdfTable) -> float:
    """TF-IDF 向量的 L2 范数 ``sqrt(sum_t (tf(t) * idf(t))^2)``。

    参数
    ----
    counts : Dict[int, int]
        token→计数表。
    idf : IdfTable
        IDF 表。

    返回
    ----
    float
        范数（表为空时 0.0）。
    """
    if not counts:
        return 0.0
    keys = sorted(counts.keys())
    w = idf.lookup(np.asarray(keys, dtype=np.uint64))
    tf = np.asarray([int(counts[k]) for k in keys], dtype=np.float64)
    return float(np.sqrt(np.dot((tf * w), (tf * w))))


def numeric_answer_fraction(answer_values: Sequence[str]) -> float:
    """答案中数值型 token 占比（``^[0-9]+$`` 的条数 / 答案集合总条数）。

    参数
    ----
    answer_values : Sequence[str]
        合并后的答案字符串。

    返回
    ----
    float
        占比；答案集合为空时 0.0。
    """
    total = 0
    numerics = 0
    for raw in answer_values:
        total += 1
        if NUMERIC_ONLY_RE.match(str(raw).strip()):
            numerics += 1
    if total <= 0:
        return 0.0
    return float(numerics) / float(total)


def rich_columns_from_pair(
    q_counts: Dict[int, int],
    d_counts: Dict[int, int],
    q_idf_sum: float,
    d_idf_sum: float,
    q_idf_norm: float,
    d_idf_norm: float,
    idf: IdfTable,
    numeric_answer_fraction_value: float,
    out: np.ndarray,
    offset: int,
) -> None:
    """把 4 列 rich 特征写入 ``out[offset : offset + RICH_EXTRA_DIM]``（就地写）。

    列顺序与 :func:`_rich_columns` 一致：
    ``qcov_idf`` / ``dcov_idf`` / ``tfidf_cos`` / ``ans_isnum_frac``。

    参数
    ----
    q_counts : Dict[int, int]
        问题侧计数表（TF-IDF 用）。
    d_counts : Dict[int, int]
        文档侧计数表（TF-IDF 用）。
    q_idf_sum : float
        问题侧 idf 和 ``sum_{t in T(q)} idf(t)``。
    d_idf_sum : float
        文档侧 idf 和。
    q_idf_norm : float
        问题侧 IDF 权重向量范数。
    d_idf_norm : float
        文档侧 IDF 权重向量范数。
    idf : IdfTable
        IDF 表。
    numeric_answer_fraction_value : float
        末列 ``ans_isnum_frac`` 的取值（逐题常量）。
    out : np.ndarray
        目标行向量。
    offset : int
        rich 块在该行中的起始列号。

    返回
    ----
    None
    """
    inter_sum = _idf_intersection_sum(q_counts, d_counts, idf)
    out[offset] = float(inter_sum / q_idf_sum) if q_idf_sum > 0.0 else 0.0
    out[offset + 1] = float(inter_sum / d_idf_sum) if d_idf_sum > 0.0 else 0.0
    denom = float(q_idf_norm) * float(d_idf_norm)
    if denom > 0.0:
        out[offset + 2] = float(_tfidf_dot(q_counts, d_counts, idf) / denom)
    else:
        out[offset + 2] = 0.0
    out[offset + 3] = float(numeric_answer_fraction_value)

class TextFeatureCache:
    """文本特征的 **memoization** 缓存（唯一的性能优化点，不改变任何计算结果）。

    不变量
    ------
    ``tokens[key]`` / ``sets[key]``（以及 ``no_bag=False`` 时的 ``bags[key]``）由**同一份文本**派生；
    缓存不含随机性、不影响行序与特征取值（同输入恒等输出）。
    """

    def __init__(
        self, hash_dim: int = HASH_DIM, no_bag: bool = False, features: str = FEATURES_DEFAULT
    ) -> None:
        """初始化缓存（``hash_dim``/``no_bag``/``features`` 必须与构建口径一致）。

        参数
        ----
        hash_dim : int
            词袋维数（``no_bag=True`` 时忽略）。
        no_bag : bool
            为真时**完全跳过**词袋计算（既省时也保证不会误用词袋列）。
        features : str
            特征集合名（``base`` / ``rich``）；与 :func:`build_feature_vector` 的取值必须一致。
        """
        self.features = _check_features(features)
        self.no_bag = bool(no_bag)
        self.hash_dim = 0 if self.no_bag else int(hash_dim)
        self.tokens: Dict[str, List[str]] = {}
        self.sets: Dict[str, Set[str]] = {}
        self.counts: Dict[str, Dict[int, int]] = {}
        self.token_counts: Dict[str, int] = {}
        self.bags: Dict[str, np.ndarray] = {}

    def ensure(self, key: str, text: str) -> None:
        """首次见到 ``key`` 时计算并驻留其 token / token 集合 / 词袋计数。

        参数
        ----
        key : str
            缓存键（问题用 ``q:<QuestionId>``，文档用成员名）。
        text : str
            对应文本。

        返回
        ----
        None
        """
        if key in self.token_counts:
            return
        toks = tokenize(text)
        if len(toks) <= int(CACHE_TOKEN_LIST_LIMIT):
            self.tokens[key] = toks
        self.token_counts[key] = len(toks)
        self.sets[key] = set(toks)
        # rich 层要算 TF-IDF，需要保留重数 -> 存整数计数表（键为 token 标识）
        self.counts[key] = _term_counts([hash_token_id(t) for t in toks])


def build_feature_vector(
    cache: TextFeatureCache,
    q_key: str,
    q_text: str,
    d_key: str,
    d_text: str,
    numeric_answer_flag: float,
    hash_dim: int = HASH_DIM,
    no_bag: bool = False,
    features: str = FEATURES_DEFAULT,
    idf: Optional[IdfTable] = None,
    numeric_answer_fraction_value: float = 0.0,
    doc_index: Optional[int] = None,
    bag_cache: Optional[BagCache] = None,
    q_set: Optional[Set[str]] = None,
    q_counts: Optional[Dict[int, int]] = None,
    d_tokens: Optional[Sequence[str]] = None,
    d_set: Optional[Set[str]] = None,
    d_counts: Optional[Dict[int, int]] = None,
    q_idf_sum: float = 0.0,
    q_idf_norm: float = 0.0,
    d_idf_sum: float = 0.0,
    d_idf_norm: float = 0.0,
    doc_n_tokens: Optional[int] = None,
) -> np.ndarray:
    """构造一行 ``X``（长度 ``feature_dim(hash_dim, no_bag, features)`` 的 float32 向量）。

    列定义由 :func:`feature_columns` 按同一口径生成（与写入 meta 的内容同源，单一事实来源）。

    [!] ``base`` 特征集合下本函数的**取值与加法顺序与历史版本逐位一致**（词袋块由
    :func:`hash_bag` 临时构造并原地归一化；逐对特征按同一 token 集合口径计算），
    故既有 base 产物逐字节不变。``rich`` 才追加 4 列 IDF/TF-IDF 特征。

    参数
    ----
    cache : TextFeatureCache
        文本特征缓存（问题侧始终从缓存取；文档侧在 in-memory 路径也从缓存取）。
    q_key : str
        问题缓存键。
    q_text : str
        问题文本。
    d_key : str
        文档缓存键（归档成员名）。
    d_text : str
        文档文本（compact 路径下发空串，仅占位）。
    numeric_answer_flag : float
        ``numeric_answer_flag`` 列取值（逐题常量）。
    hash_dim : int
        哈希词袋块维数（缺省写死值 :data:`HASH_DIM` = 64）。
    no_bag : bool
        为真时**不产出词袋块**，向量只含附加特征块（``base`` 为 6 列）。
    features : str
        ``base`` / ``rich``（``rich`` 在附加块后追加 4 列 IDF/TF-IDF 特征，需要 ``idf``）。
    idf : Optional[IdfTable]
        ``rich`` 时必须给出的 IDF 表。
    numeric_answer_fraction_value : float
        rich 末列 ``ans_isnum_frac`` 的取值（逐题常量）。
    doc_index : Optional[int]
        文档序号（提供且本行确实用到词袋块时走 :class:`BagCache`，避免同文档多行重复哈希）。
    bag_cache : Optional[BagCache]
        词袋块缓存（``doc_index`` 非空且 ``d_tokens`` 可用时生效）。
    q_set : Optional[Set[str]]
        问题 token 集合；``None`` 时从缓存取（compact 路径下 token 为整数标识）。
    q_counts : Optional[Dict[int, int]]
        问题侧 token 计数表（``rich`` 必需）。
    d_tokens : Optional[Sequence[str]]
        文档 token 序列（in-memory 路径给出；compact 路径给出紧凑标识数组）。
    d_set : Optional[Set[str]]
        文档 token 集合（``base`` 的覆盖率/Jaccard 用；compact 路径传空集合，按计数表键求）。
    d_counts : Optional[Dict[int, int]]
        文档侧 token 计数表（``rich`` 必需）。
    q_idf_sum / q_idf_norm / d_idf_sum / d_idf_norm : float
        rich 的预计算聚合量（问题/文档侧的 ``sum idf`` 与 IDF/TF-IDF 范数）。
    doc_n_tokens : Optional[int]
        文档 token 总数（含重数），用于 ``log1p_doc_tokens``；``None`` 时回退查文本缓存
        （compact 路径的文档不进缓存，必须由调用方给出，否则该列会恒为 0）。

    返回
    ----
    np.ndarray
        形状 ``[feature_dim(hash_dim, no_bag, features)]`` 的 float32 行向量。

    异常
     -----
    BuildContractError
        口径与缓存不一致、``rich`` 却缺 ``idf``，或 token 规模超过缓存容量时抛出。
    """
    base = 0 if bool(no_bag) else int(hash_dim)
    if base != int(cache.hash_dim) or bool(no_bag) != bool(cache.no_bag):
        raise BuildContractError(
            f"口径与文本缓存不一致：build_feature_vector 收到 (hash_dim={base}, no_bag={no_bag})，"
            f"缓存为 (hash_dim={cache.hash_dim}, no_bag={cache.no_bag})"
        )
    feat = _check_features(features)
    if feat != str(cache.features):
        raise BuildContractError(
            f"特征集合与文本缓存不一致：build_feature_vector 收到 features={feat}，"
            f"缓存为 {cache.features}"
        )
    if feat == "rich" and idf is None:
        raise BuildContractError("rich 特征必须给出 idf（IDF 表）")
    cache.ensure(q_key, q_text)
    if not cache.no_bag:
        cache.ensure(d_key, d_text)
    if q_set is None:
        q_set = cache.sets[q_key]
    if d_set is None:
        d_set = cache.sets.get(d_key, set())
    row = np.zeros(feature_dim(base, bool(no_bag), feat), dtype=np.float32)
    q_tokens = cache.tokens[q_key]
    # ---- 列 0..base-1（仅 no_bag=False 时存在）：哈希词袋（问题词袋 + 文档词袋 -> L2 归一化）----
    if base > 0:
        d_toks = d_tokens if d_tokens is not None else cache.tokens[d_key]
        if bag_cache is not None and doc_index is not None:
            # 问题侧与文档侧各自按**自己的 cache_key** 取词袋（问题用 q_key、文档用 d_key），
            # 二者不得共用同一个键（早期实现用固定占位 key 复用问题词袋 -> 见 BagCache 的注记）
            row[:base] = bag_cache.get(q_key, q_tokens) + bag_cache.get(d_key, d_toks)
        else:
            row[:base] = hash_bag(q_tokens, base) + hash_bag(d_toks, base)
        l2_normalize_block(row[:base])
    # ---- 列 base+0..base+2：token 集合口径的覆盖率与 Jaccard（no_bag 时 base=0）----
    # [!] compact 路径两侧的 token 表示必须**同源**：文档侧是整数标识（hash_token_id），
    #     问题侧若用字符串集合则交集恒空 -> 覆盖率/Jaccard 三列恒为 0（静默错，实测已复现）。
    #     故当文档侧给的是整数标识时，两侧都改用"计数表键集合"（同一标识空间）。
    # 判据：文档侧是否处在"整数标识空间"（compact 路径的 d_tokens 是整数数组；in-memory 是 str 列表）
    doc_int_space = d_tokens is not None and not isinstance(d_tokens, (list, tuple))
    if doc_int_space:
        # 两侧都换成整数标识空间：问题侧用计数表（权威）或现算 hash_token_id
        q_keys = set(q_counts.keys()) if q_counts else {hash_token_id(t) for t in q_tokens}
        d_keys = set(d_counts.keys()) if d_counts else set(int(x) for x in d_tokens.tolist())
    else:
        q_keys, d_keys = q_set, (d_set or set())
    inter = len(q_keys & d_keys)
    union = len(q_keys | d_keys)
    row[base] = float(inter) / float(len(q_keys)) if q_keys else 0.0
    row[base + 1] = float(inter) / float(len(d_keys)) if d_keys else 0.0
    row[base + 2] = float(inter) / float(union) if union else 0.0
    # ---- 列 base+3..base+4：长度（含重数的 token 数）----
    # [!] 文档 token 数优先取调用方给的 ``doc_n_tokens``（compact 路径文档不进缓存，
    #     若只查 ``cache.token_counts`` 会静默取 0，该列变成死列）；缓存仅作兜底。
    if doc_n_tokens is None:
        doc_n_tokens = cache.token_counts.get(d_key, 0)
    row[base + 3] = float(np.log1p(int(doc_n_tokens)))
    row[base + 4] = float(np.log1p(cache.token_counts.get(q_key, len(q_tokens))))
    # ---- 列 base+5：数字类答案标记（问题侧常量）----
    row[base + 5] = float(numeric_answer_flag)
    # ---- 列 base+6..base+9（仅 rich）：IDF 加权覆盖度 / TF-IDF 余弦 / 数值答案占比 ----
    if feat == "rich":
        if q_counts is None:
            q_counts = _token_counts_from_cache(cache, q_key)
        if d_counts is None:
            raise BuildContractError("rich 特征需要文档侧 token 计数表（d_counts）")
        if q_idf_sum <= 0.0:
            q_idf_sum = _idf_weights(sorted(q_counts.keys()), idf)[0]
        if q_idf_norm <= 0.0:
            q_idf_norm = _tfidf_norm(q_counts, idf)
        if d_idf_sum <= 0.0:
            d_idf_sum = _idf_weights(sorted(d_counts.keys()), idf)[0]
        if d_idf_norm <= 0.0:
            d_idf_norm = _tfidf_norm(d_counts, idf)
        rich_columns_from_pair(
            q_counts,
            d_counts,
            float(q_idf_sum),
            float(d_idf_sum),
            float(q_idf_norm),
            float(d_idf_norm),
            idf,
            float(numeric_answer_fraction_value),
            row,
            base + EXTRA_DIM,
        )
    return row

# ======================================================================
# 样本采样与组装
# ======================================================================
def _sample_negatives(
    rng: np.random.Generator,
    pool: Sequence[str],
    own: Set[str],
    k: int,
    question_id: str,
) -> List[str]:
    """为一个题目采样 ``k`` 个负样本文档（**同 split 内其他题的文档**）。

    口径
    ----
    * 候选池 = 本次构建纳入的全部正样本文档的并集（按名升序，确定性）；
    * 排除该题**自己**的文档（``own``），从而保证"同一 QuestionId 的文档不会同时出现在
      正负两侧"；
    * 无放回采样（同一题的 ``k`` 个负样本互不相同）；
    * 采样只用传入的 ``rng``（独立 Generator），不触碰全局 RNG。

    参数
    ----
    rng : np.random.Generator
        独立随机数生成器（由固定 seed 播种）。
    pool : Sequence[str]
        文档候选池（升序）。
    own : Set[str]
        该题自己的正样本文档集合。
    k : int
        需要的负样本数（等于该题的正样本数，保证 1:1）。
    question_id : str
        题目 ID（仅用于报文）。

    返回
    ----
    List[str]
        采到的负样本文档成员名（按采样顺序）。

    异常
    ------
    BuildContractError
        候选池（去掉自身文档后）不足 ``k`` 个时抛出。
    """
    allowed = [d for d in pool if d not in own]
    if len(allowed) < int(k):
        raise BuildContractError(
            f"题目 {question_id!r} 需要 {k} 个负样本，但候选池去掉自身后只剩 "
            f"{len(allowed)} 个（{len(pool)} - {len(own)}）；"
            f"若这是 --max-questions 演练构建，请增大题目数（候选池 = 纳入题目的文档并集，"
            f"过小则无法为该题凑出 1:1 负样本；全量构建不受此限）"
        )
    idx = rng.choice(len(allowed), size=int(k), replace=False)
    return [allowed[int(i)] for i in idx]


def _build_split_idf(
    records: Sequence[QaRecord], doc_texts: Dict[str, str]
) -> IdfTable:
    """in-memory 路径的 IDF 表构造（语料 = 本次构建的文档池，口径与 compact 路径一致）。

    参数
    ----
    records : Sequence[QaRecord]
        本次构建的记录（用于取文档池）。
    doc_texts : Dict[str, str]
        文档成员名 -> 文档全文。

    返回
    ----
    IdfTable
        ``df`` 按**唯一 token 标识**去重计数，``M`` = 文档池大小。

    异常
    ------
    BuildContractError
        文档池为空时抛出。
    """
    pool = sorted({m for r in records for m in r.doc_members})
    if not pool:
        raise BuildContractError("IDF 语料的文档池为空（无任何 evidence 引用）")
    df: Dict[int, int] = {}
    for member in pool:
        ids = np.fromiter(
            (hash_token_id(t) for t in tokenize(doc_texts[member])),
            dtype=np.uint64,
            count=-1,
        )
        for key in np.unique(ids).tolist():
            src_key = key if isinstance(key, int) else int(key)
            df[src_key] = df.get(src_key, 0) + 1
    return IdfTable(df, len(pool))


def _fill_rows_inplace(
    X: np.ndarray,
    y: np.ndarray,
    samples: Sequence[Sample],
    records_by_id: Dict[str, QaRecord],
    doc_index: Dict[str, int],
    fields: Dict[str, Any],
    cache: TextFeatureCache,
    features: str,
    idf: Optional[IdfTable],
    hash_dim: int,
    no_bag: bool,
) -> None:
    """按行填 ``X`` / ``y``（in-memory 与 compact 两条路径共用同一段填行逻辑）。

    参数
    ----
    X : np.ndarray
        ``[M, D]`` float32 待填特征矩阵。
    y : np.ndarray
        ``[M]`` int64 待填标签。
    samples : Sequence[Sample]
        严格正负交替的逐样本 provenance。
    records_by_id : Dict[str, QaRecord]
        QuestionId -> 记录。
    doc_index : Dict[str, int]
        文档成员名 -> 文档序号。
    fields : Dict[str, Any]
        两种路径各自的取数闭包：``q_stats`` / ``doc_stats`` / ``ans_frac`` / ``bag_cache`` /
        ``doc_view``（compact 另有 ``packed``），由 :func:`build_split` 按路径装配。
    cache : TextFeatureCache
        in-memory 文本缓存（compact 路径只用它做**问题侧**取数）。
    features : str
        ``base`` / ``rich``。
    idf : Optional[IdfTable]
        rich 的 IDF 表（base 时为 ``None``）。
    hash_dim : int
        词袋块维数（``no_bag`` 时为 0 语义）。
    no_bag : bool
        是否不产出词袋块。

    返回
    ----
    None
    """
    _doc_view = fields.get("doc_view", lambda m: m)
    for i, s in enumerate(samples):
        r = records_by_id[s.question_id]
        q_key = f"q:{r.question_id}"
        q_set, q_counts, q_sum, q_norm = fields["q_stats"](q_key)
        d_tokens, d_set, d_counts, d_sum, d_norm = fields["doc_stats"](s.doc_member, q_set)
        doc_n_tokens = fields["doc_n_tokens"](s.doc_member) if "doc_n_tokens" in fields else None
        X[i] = build_feature_vector(
            cache,
            q_key,
            r.question,
            s.doc_member,
            _doc_view(s.doc_member),
            r.numeric_answer_flag,
            int(hash_dim),
            bool(no_bag),
            features,
            idf,
            float(fields["ans_frac"].get(s.question_id, 0.0)),
            int(doc_index[s.doc_member]),
            fields["bag_cache"] if not bool(no_bag) else None,
            q_set=q_set,
            q_counts=q_counts,
            d_tokens=d_tokens,
            d_set=d_set,
            d_counts=d_counts,
            q_idf_sum=q_sum,
            q_idf_norm=q_norm,
            d_idf_sum=d_sum,
            d_idf_norm=d_norm,
            doc_n_tokens=doc_n_tokens,
        )
        y[i] = int(s.label)


class WordBoundaryAhoCorasick:
    r"""**词边界感知**的 Aho-Corasick 自动机（纯 Python + 列表，不新增三方依赖）。

    用途
    ----
    在 compact 路径上把"每文档 x 每题目"的正则循环，换成"每文档**一次线性扫描**"：扫一遍
    文档即可得到"**可能命中**"的题目集合；真值仍由 :func:`doc_contains_answer` 用既有模式判定。

    [!] **本类未被接入构建路径**（AC 用法**未采用**）：设计意图是让 AC 只做保守剪枝
    （即保证"绝不漏判"），但该性质**未被实测支持** —— `_verify/_diag/test_ac_equivalence.py`
    在**默认参数**下实测 **5367 对中漏判 1240 对**（canon 折叠口径；数值随 `--samples/--fuzz`
    变化，故只报"默认参数"下的实测；主因是右边界编码只在文本末尾生效，见 :func:`canon_char` 的说明），
    且吞吐实测亦慢于正则（15345 µs/对 vs 5230 µs/对）。按"若等价性无法证明，宁可不替换"的原则，
    compact 路径继续使用 :func:`doc_contains_answer`；本类与对照脚本仅作**可复核留档**保留
    （**未接入构建路径**）。

    词边界如何编码
    --------------
    答案判定带词边界（首/末为非 ``\w`` 字符时省略该侧；短数字答案额外禁止与相邻数字或
    小数点连写）。裸字面 AC 会把 ``cat`` 命中 ``concatenate`` 这类对算成候选。本实现把
    模式拆成"**左边界编码 + 核心 + 右边界编码**"三段：

    * 左边界只允许"字符串开头"或"前一个字符不是 ``\w``"（短数字答案另允许"前一个字符不是
      ``.``"）——于是"前面是字母"的位置**根本不会进入核心**，天然排除 ``xcat``；
    * 右边界只允许"字符串结尾"或"后一个字符不是 ``\w``"（短数字答案另允许"后一个字符不是
      ``.``"）——排除 ``catx``；右边界用"后继字符"分支实现，故命中位置就是核心末字符。

    不变量
    ------
    * ``self.out[pos]`` 只含"在 ``pos`` 处结束**且两侧边界条件成立**"的模式下标；
    * 命中位置 ``pos`` = 核心末字符位置，起始位置 = ``pos + 1 - base_len[pidx]``；
    * 全程确定性、不消耗 RNG；``fail`` 用 BFS（``collections.deque``）计算。

    [!] 与逐条正则的等价性由 `_verify/_diag/test_ac_equivalence.py` 逐对验证（含词边界与
    Unicode 折叠陷阱的 fuzz），且**真值始终由正则给出**，因此不存在语义漂移空间。
    """

    def __init__(
        self,
        trie: Dict[str, int],
        fail: List[int],
        out: List[Tuple[int, ...]],
        base_len: List[int],
        run_len: List[int],
        mode: Tuple[str, str, str],
        start_len: List[int],
    ) -> None:
        """内部构造结果（对外请用 :meth:`build`）。"""
        self.trie = trie
        self.fail = fail
        self.out = out
        self.base_len = base_len
        self.run_len = run_len
        self.mode = mode
        self.start_len = start_len

    def __init__(
        self,
        trie: Dict[str, int],
        fail: List[int],
        out: List[Tuple[int, ...]],
        base_len: List[int],
        lead_len: List[int],
        word_chars: set,
        start_only: str,
        start_num: str,
        end_only: str,
        dot_lead: str,
    ) -> None:
        """内部构造结果（对外请用 :meth:`build`）。"""
        self.trie = trie
        self.fail = fail
        self.out = out
        self.base_len = base_len
        self.lead_len = lead_len
        self.word_chars = word_chars
        self.start_only = start_only
        self.start_num = start_num
        self.end_only = end_only
        self.dot_lead = dot_lead

    @classmethod
    def build(cls, patterns: Sequence[Tuple[str, bool, bool, bool]]) -> "WordBoundaryAhoCorasick":
        """由 ``(核心, 左侧启用词边界, 右侧启用词边界, 是否短数字保护)`` 构造自动机。

        构造方式
        --------
        每个模式最多产生 **3 个变体**（各自在 trie 里是唯一路径，绝不互相覆盖）：

        * 变体 A：``START_ALL + 核心 + END``（左边界 = "串开头或前一字符非 ``\\w``"，含短数字口径）
        * 变体 B：``核心 + END``（左边界 = "串开头 **且** 前一字符非 ``\\w`` 且非 ``.``"，
          用于短数字答案：由扫描时**只把 A 允许的前导字符替换成"不可达哨兵"**实现，见下）
        * 变体 C：``START_ALL + 核心``（右边界 = "串结尾或后一字符非 ``\\w``"）

        短数字答案的三条约束（前非 ``\\w``、前非 ``.``、后非 ``\\w``、后非 ``.``）在
        :meth:`boundary_ok` 中按**原文**逐条复核；trie 只承担"字面 + 方向"的必要条件。

        参数
        ----
        patterns : Sequence[Tuple[str, bool, bool, bool]]
            模式集合；``核心`` 为空者被跳过（仍保留下标占位以对齐模式列表）。

        返回
        ----
        WordBoundaryAhoCorasick
            构造好的自动机。
        """
        # 虚拟字符（控制字符；真实文本经 tokenize/语料抽取不可能含它们）：
        #   lead_ch  —— 模式前导，只可能匹配"文本当前位置是串开头"或"前一字符非 \w"
        #   end_ch   —— 模式后继，只可能匹配"文本当前位置是串结尾"或"后一字符非 \w"
        lead_ch, end_ch = "\x01", "\x02"
        trie: Dict[str, int] = {}
        base_len: List[int] = []
        lead_len: List[int] = []
        out_lists: List[List[int]] = [[]]
        n_states = 1

        def _new_state() -> int:
            nonlocal n_states
            st = n_states
            n_states += 1
            out_lists.append([])
            return st

        def _child(state: int, ch: str) -> int:
            key = f"{state}\x00{ch}"
            nxt = trie.get(key)
            if nxt is None:
                nxt = _new_state()
                trie[key] = nxt
            return nxt

        for pidx, (core, b_left, b_right, _numeric) in enumerate(patterns):
            base_len.append(len(core))
            if not core:
                lead_len.append(0)
                continue
            variants: List[Tuple[bool, bool]] = []
            if b_left:
                variants.append((True, b_right))
                if b_right:
                    variants.append((False, True))
            else:
                variants.append((False, b_right))
            for has_lead, has_end in variants:
                st = 0
                if has_lead:
                    st = _child(st, lead_ch)
                for ch in core:
                    st = _child(st, ch)
                if has_end:
                    st = _child(st, end_ch)
                out_lists[st].append(pidx)
            lead_len.append(1 if b_left else 0)

        from collections import deque

        fail: List[int] = [0] * n_states
        merged: List[Tuple[int, ...]] = [tuple(dict.fromkeys(x)) for x in out_lists]
        children: List[Dict[str, int]] = [dict() for _ in range(n_states)]
        dq: "deque[int]" = deque()
        for key, nxt in trie.items():
            st_s, ch = key.split("\x00", 1)
            st = int(st_s)
            children[st][ch] = nxt
            if st == 0:
                fail[nxt] = 0
                dq.append(nxt)
        while dq:
            u = dq.popleft()
            for ch, v in children[u].items():
                f = fail[u]
                while f and ch not in children[f]:
                    f = fail[f]
                tgt = children[f].get(ch, 0)
                fail[v] = 0 if tgt == v else tgt
                merged[v] = tuple(dict.fromkeys(merged[v] + merged[fail[v]]))
                dq.append(v)
        return cls(
            trie, fail, merged, base_len, lead_len, set(_word_chars_table()),
            lead_ch, lead_ch, end_ch, end_ch,
        )

    def _find(self, text: str) -> List[int]:
        """在 ``text``（**已带首尾虚拟字符**）上扫一遍，返回展平的命中三元组。"""
        trie = self.trie
        fail = self.fail
        out = self.out
        base_len = self.base_len
        state = 0
        hits: List[int] = []
        for idx, ch in enumerate(text):
            key = f"{state}\x00{ch}"
            nxt = trie.get(key)
            while nxt is None and state:
                state = fail[state]
                nxt = trie.get(f"{state}\x00{ch}")
            state = nxt if nxt is not None else 0
            if out[state]:
                for pidx in out[state]:
                    n = base_len[pidx]
                    if n <= 0:
                        continue
                    end_in_text = idx - 1
                    core_start = end_in_text - n + 1
                    core_end = end_in_text + 1
                    if core_start < 0 or core_end > len(text) - 1:
                        continue
                    hits.append(pidx)
                    hits.append(core_start)
                    hits.append(core_end)
        return hits

    def scan(self, original: str) -> List[Tuple[int, int, int]]:
        """扫一遍 ``original``，返回 ``[(pidx, core_start, core_end), ...]``（**边界已复核**）。

        参数
        ----
        original : str
            原始文本。内部会在首尾各补一个虚拟字符（``lead_ch`` / ``end_ch``），
            故返回的下标可直接用于在 ``original`` 上复核边界。

        返回
        ----
        List[Tuple[int, int, int]]
            命中列表：``pidx`` = 模式下标、``core_start``/``core_end`` = 核心在半开区间
            ``[core_start, core_end)``（即答案字面在文本中的位置）。
        """
        lead_ch, end_ch = self.start_only, self.end_only
        data = lead_ch + original + end_ch
        flat = self._find(data)
        # 下标平移：data 的下标 - 1 = original 的下标
        res: List[Tuple[int, int, int]] = []
        for k in range(0, len(flat), 3):
            pidx, cs, ce = flat[k], flat[k + 1], flat[k + 2]
            s = cs - 1
            e = ce - 1
            if 0 <= s < e <= len(original):
                res.append((pidx, s, e))
        return res

    def boundary_ok(self, original: str, core_start: int, core_end: int, numeric: bool) -> bool:
        """复核命中处的**左右边界**（与 ``(?<!...)`` / ``(?!...)`` 逐字对应）。

        参数
        ----
        original : str
            原文（边界字符按原文判定，避免折叠改变边界）。
        core_start / core_end : int
            核心在 ``original`` 中的半开区间。
        numeric : bool
            是否短数字答案（额外禁止与数字 / 小数点连写）。

        返回
        ----
        bool
            边界是否成立。
        """
        words = self.word_chars
        left = original[core_start - 1] if core_start > 0 else None
        right = original[core_end] if core_end < len(original) else None
        if left is not None:
            if left in words or (numeric and left == "."):
                return False
        if right is not None:
            if right in words or (numeric and right == "."):
                return False
        return True


def _word_chars_table() -> set:
    """现场枚举 ``\\w`` 覆盖的字符（**不凭记忆手写**）：对 0..0x10FFFF 逐码位用
    :data:`WORD_HEAD_RE` 实测。结果缓存，只算一次。"""
    global _WORD_CHARS_CACHE
    if _WORD_CHARS_CACHE is None:
        cache: Set[str] = set()
        for cp in range(0x110000):
            ch = chr(cp)
            if WORD_HEAD_RE.match(ch):
                cache.add(ch)
        _WORD_CHARS_CACHE = cache
    return _WORD_CHARS_CACHE


_WORD_CHARS_CACHE: Optional[Set[str]] = None


def _answer_prefilter_pass(r: "QaRecord", member: str, doc_chars: Dict[str, frozenset]) -> bool:
    """in-memory 采样路径的**答案命中预筛**（折叠安全）。

    口径（**单层：字符级必要条件**）：
    文档字符集必须与答案字符集相交，否则正则必然不命中。字符集经"大小写变体 + 折叠闭包"
    构造（见 :func:`answer_char_set` / :func:`doc_char_set` / :func:`fold_forms`），
    实测覆盖 ``ſ/ı/İ/ς/K/ς`` 等全部已知折叠陷阱。

    [!] 早期版本还叠了 token / 字面两层"加速"，但那两层在折叠陷阱下会**否决真命中**
    （实测 ``ans='i'`` vs ``doc='ı'``：两层都判否、正则判真），故已全部移除；现只有这一层，
    且它是**可靠的必要条件**（R33 以 450 组对抗集 + 3000 组 fuzz 复核：0 反例）。

    参数
    ----
    r : QaRecord
        题目记录。
    member : str
        文档成员名（缓存键）。
    doc_chars : Dict[str, frozenset]
        文档字符集缓存（由 :func:`doc_char_set` 生成）。

    返回
    ----
    bool
        为真表示"可能命中"（仍需正则判定）；为假可安全判定不命中。
    """
    # [!] 只用**字符级必要条件**：token / 字面两层在 Unicode 折叠陷阱下会**否决真命中**
    #     （实测 ``'i'`` vs ``'ı'``：两层都判否、正则判真），故不再作为否决条件。
    #     字符级判据对折叠做了等价类闭包（见 :func:`answer_char_set` / :func:`doc_char_set`），是可靠的。
    return not doc_chars[member].isdisjoint(r.answer_chars)


def _collect_samples(
    split: str,
    records: Sequence[QaRecord],
    pool: Sequence[str],
    doc_texts: Dict[str, str],
    negative_seed: int,
    negative_ratio: float,
) -> Tuple[List[Sample], Dict[str, Any]]:
    """采样正负样本并排成**严格正负交替**的行序（in-memory 与 compact 路径共用）。

    参数
    ----
    split : str
        split 名（仅用于统计与报文）。
    records : Sequence[QaRecord]
        已按 ``QuestionId`` 升序、且 doc_members 已过滤掉缺失文档的记录。
    pool : Sequence[str]
        文档候选池（升序）。
    doc_texts : Dict[str, str]
        文档成员名 -> 文档全文（**只读**，用于正样本的答案命中判定）。
    negative_seed : int
        负样本采样种子。
    negative_ratio : float
        负样本比例（写死 1.0）。

    返回
    ----
    Tuple[List[Sample], Dict[str, Any]]
        ``(行序样本, 计数统计)``。
    """
    if float(negative_ratio) != 1.0:
        raise BuildContractError(
            f"negative_ratio 口径写死为 1.0（1:1 均衡），当前 {negative_ratio}"
        )
    rng = np.random.default_rng(int(negative_seed))
    records_by_id_a = {r.question_id: r for r in records}
    # 预筛集合（文档侧按同一规则抽一次，供该文档参与的所有题目复用）
    doc_chars: Dict[str, frozenset] = {}
    for member in pool:
        doc_chars[member] = doc_char_set(doc_texts[member])
    per_question: Dict[str, Tuple[List[Sample], List[Sample]]] = {}
    n_pos = n_neg = 0
    n_hit_pos = n_hit_neg = 0
    for r in records:
        pos: List[Sample] = []
        for member in r.doc_members:
            text = doc_texts[member]
            hit = (
                1
                if _answer_prefilter_pass(r, member, doc_chars)
                and doc_contains_answer(text, r.answer_patterns)
                else 0
            )
            pos.append(
                Sample(
                    question_id=r.question_id,
                    doc_member=member,
                    doc_kind=_doc_kind(member),
                    label=1,
                    answer_hit=hit,
                )
            )
            n_pos += 1
            n_hit_pos += hit
        k = int(round(len(pos) * float(negative_ratio)))
        neg: List[Sample] = []
        for member in _sample_negatives(rng, pool, set(r.doc_members), k, r.question_id):
            text = doc_texts[member]
            hit = (
                1
                if _answer_prefilter_pass(r, member, doc_chars)
                and doc_contains_answer(text, r.answer_patterns)
                else 0
            )
            neg.append(
                Sample(
                    question_id=r.question_id,
                    doc_member=member,
                    doc_kind=_doc_kind(member),
                    label=0,
                    answer_hit=hit,
                )
            )
            n_neg += 1
            n_hit_neg += hit
        per_question[r.question_id] = (pos, neg)
    _ = records_by_id_a
    # ---- 严格正负交替的行序（见 build_split 的 docstring）----
    samples: List[Sample] = []
    max_k = max((len(p) for p, _ in per_question.values()), default=0)
    for i in range(max_k):
        for r in records:  # records 已按 QuestionId 升序
            pos, neg = per_question[r.question_id]
            if i < len(pos):
                samples.append(pos[i])
                samples.append(neg[i])
    if len(samples) != n_pos + n_neg:
        raise BuildContractError(
            f"行序组装后样本数不符：{len(samples)} != {n_pos} + {n_neg}"
        )
    for i, s in enumerate(samples):
        expect = 1 if (i % 2 == 0) else 0
        if s.label != expect:
            raise BuildContractError(
                f"行序未做到严格正负交替：第 {i} 行 label={s.label}，期望 {expect}"
            )
    stats: Dict[str, Any] = {
        "positive": int(n_pos),
        "negative": int(n_neg),
        "positive_fraction": float(n_pos) / float(max(n_pos + n_neg, 1)),
        "questions": len(records),
        "questions_without_evidence_ref": sum(1 for r in records if not r.doc_members),
        "pool_documents": len(pool),
        "positive_answer_hit": int(n_hit_pos),
        "negative_answer_hit": int(n_hit_neg),
    }
    return samples, stats


def build_split(
    split: str,
    records: Sequence[QaRecord],
    doc_texts: Optional[Dict[str, str]],
    negative_seed: int = DEFAULT_NEGATIVE_SEED,
    negative_ratio: float = 1.0,
    hash_dim: int = HASH_DIM,
    no_bag: bool = False,
    features: str = FEATURES_DEFAULT,
    idf: Optional[IdfTable] = None,
    preload: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Sample], Dict[str, Any]]:
    """把一个 split 的 QA 记录组装成 ``X`` / ``y``（含负样本采样与行序口径）。

    行序口径（**关键**：``n3d_shape`` 的数据层用"每 5 个样本取 1 个"的**交错**切分，
    故行序直接决定训练/测试集的类别比例）
    ----------------------------------------------------------------
    按"轮次 i 从 0 到最大正样本数-1、每轮按 QuestionId 升序遍历题目，逐题依次发出
    第 i 个正样本与第 i 个负样本"排布 —— 全表**严格正负交替**（索引偶数为正、奇数为负），
    于是任何"每 k 个取 1 个"的交错切分都得到近似均衡的两份。

    参数
    ----
    split : str
        ``wiki`` / ``web``（仅用于 meta）。
    records : Sequence[QaRecord]
        QuestionId 升序的 QA 记录。
    doc_texts : Dict[str, str]
        文档成员名 -> 文档全文。
    negative_seed : int
        负样本采样种子（固定口径）。
    negative_ratio : float
        负样本比例（写死 1.0 = 1:1；非 1.0 时按 ``round(k * ratio)`` 取整）。
    hash_dim : int
        哈希词袋块维数（缺省写死值 :data:`HASH_DIM` = 64；``--hash-dim`` 生效时随之变化）。
    no_bag : bool
        为真时不产出词袋块（``--no-bag``，``D = 6``）。

    返回
    ----
    Tuple[np.ndarray, np.ndarray, List[Sample], Dict[str, Any]]
        ``(X float32 [M, D], y int64 [M], 逐样本 provenance, 构建统计)``。

    异常
    ------
    BuildContractError
        文档缺失 / 负样本池不足 / 标签不平衡 / 特征出现 NaN 时抛出。
    """
    feat = _check_features(features)
    if feat == "rich" and idf is None:
        raise BuildContractError("features='rich' 必须给出 idf（IDF 表）")
    if float(negative_ratio) != 1.0:
        raise BuildContractError(
            f"negative_ratio 口径写死为 1.0（1:1 均衡），当前 {negative_ratio}"
        )
    # [!] 离朱 R26 缺陷 D3：本函数是 `__all__` 里的对外 API，"行序口径"依赖 QuestionId 升序。
    #     早期版本**信任调用方顺序**，被乱序调用时会静默产出不同的行序（仍严格交替但不一致）。
    #     现改为入口处**显式按 QuestionId 升序排序**（sort 稳定，故合法调用方产物逐位不变）。
    records = sorted(records, key=lambda r: r.question_id)
    ids = [r.question_id for r in records]
    if len(set(ids)) != len(ids):
        raise BuildContractError(f"split={split!r} 的 records 含重复 QuestionId（排序后仍有重复）")
    pool = sorted({m for r in records for m in r.doc_members})
    if not pool:
        raise BuildContractError(f"split={split!r} 的文档候选池为空（无任何 evidence 引用）")
    doc_index = {m: i for i, m in enumerate(pool)}
    # [!] 这里**不预置 questions**：``counts`` 段的键序必须与历史产物逐键一致
    #     （``questions`` 由 _collect_samples / _collect_samples_precomputed 在正确位置写入）。
    stats: Dict[str, Any] = {}
    kind_counts_pre: Optional[Dict[str, int]] = None

    if doc_texts is not None:
        # ---------------- in-memory 路径（既有口径；base 产物逐字节不变） ----------------
        for member in pool:
            if member not in doc_texts:
                raise BuildContractError(f"split={split!r} 缺少文档内容：{member}")
        cache = TextFeatureCache(hash_dim=int(hash_dim), no_bag=bool(no_bag), features=feat)
        for r in records:
            cache.ensure(f"q:{r.question_id}", r.question)
        for member in pool:
            cache.ensure(member, doc_texts[member])
        samples, cstats = _collect_samples(
            split, records, pool, doc_texts, negative_seed, negative_ratio
        )
        stats.update(cstats)
        ans_frac = {r.question_id: numeric_answer_fraction(r.answer_values) for r in records}
        fields: Dict[str, Any] = {
            "ans_frac": ans_frac,
            "bag_cache": None if bool(no_bag) else BagCache(int(hash_dim)),
            "doc_view": lambda m, _dt=doc_texts: _dt[m],
            "q_stats": lambda k: _cache_qstats(cache, k, idf, feat),
            "doc_stats": lambda m, qs: _cache_dstats(cache, m, idf, feat),
            "doc_n_tokens": lambda m: int(cache.token_counts.get(m, 0)),
        }
    else:
        # ---------------- compact 路径（wiki-dev / web-dev 等 token 规模巨大的 split） ----
        if preload is None:
            raise BuildContractError("compact 路径（doc_texts=None）必须给出 preload")
        packed: DocPack = preload["packed"]
        doc_ids: Dict[str, int] = preload["doc_ids"]
        records = [
            QaRecord(
                question_id=r.question_id,
                question=r.question,
                answer_values=r.answer_values,
                answer_patterns=r.answer_patterns,
                doc_members=tuple(m for m in r.doc_members if m in doc_ids),
                numeric_answer_flag=r.numeric_answer_flag,
                answer_prefetch=r.answer_prefetch,
                answer_literals=r.answer_literals,
                answer_chars=r.answer_chars,
            )
            for r in records
        ]
        pool = sorted({m for r in records for m in r.doc_members})
        doc_index = {m: i for i, m in enumerate(pool)}
        hits: Dict[Tuple[str, str], int] = preload["answer_hits"]
        samples, cstats = _collect_samples_precomputed(
            split, records, pool, hits, negative_seed, negative_ratio
        )
        stats.update(cstats)
        kind_counts_pre = cstats["doc_kind_counts"]
        num_frac: Dict[str, float] = preload["numeric_answer_fraction"]
        ans_frac = {r.question_id: float(num_frac.get(r.question_id, 0.0)) for r in records}
        cache = TextFeatureCache(hash_dim=int(hash_dim), no_bag=bool(no_bag), features=feat)
        for r in records:
            cache.ensure(f"q:{r.question_id}", r.question)
        accessor = DocPackAccessor(packed)
        d_view_cache: Dict[str, Dict[str, Any]] = {}

        def _compact_view(member: str) -> Dict[str, Any]:
            view = d_view_cache.get(member)
            if view is None:
                uq, cnt = accessor.get(doc_index[member])
                counts = {int(k): int(v) for k, v in zip(uq.tolist(), cnt.tolist())}
                view = {
                    "tokens": uq,
                    "counts": counts,
                    "qsum": float(idf.lookup(uq).sum()),
                    "norm": _tfidf_norm(counts, idf),
                    "n_tokens": int(cnt.sum()),
                }
                if len(d_view_cache) >= 512:
                    d_view_cache.clear()
                d_view_cache[member] = view
            return view

        fields = {
            "ans_frac": ans_frac,
            "bag_cache": None if bool(no_bag) else BagCache(int(hash_dim)),
            "doc_view": lambda m: "",
            "packed": packed,
            "q_stats": lambda k: _cache_qstats(cache, k, idf, feat),
            "doc_stats": lambda m, qs: _compact_dstats(m, qs, idf, feat, _compact_view),
            # compact 的文档不进文本缓存 -> token 数由 preload 的逐文档聚合量给出
            "doc_n_tokens": lambda m: int(preload["doc_features"].get(m, (0.0, 0.0, 0))[2]),
        }

    X = np.zeros((len(samples), feature_dim(int(hash_dim), bool(no_bag), feat)), dtype=np.float32)
    y = np.zeros(len(samples), dtype=np.int64)
    records_by_id = {r.question_id: r for r in records}
    _fill_rows_inplace(
        X,
        y,
        samples,
        records_by_id,
        doc_index,
        fields,
        cache,
        feat,
        idf,
        int(hash_dim),
        bool(no_bag),
    )

    # ---- 产物契约断言（违反即失败，绝不把坏数据留给下游）----
    if not np.all(np.isfinite(X)):
        bad = np.argwhere(~np.isfinite(X))[0]
        raise BuildContractError(
            f"split={split!r} 的特征出现 NaN/Inf：第 {int(bad[0])} 行第 {int(bad[1])} 列"
        )
    if int(y.sum()) != int((y == 0).sum()) or int(y.sum()) * 2 != int(y.size):
        raise BuildContractError(
            f"split={split!r} 标签不平衡：正 {int(y.sum())} / 负 {int((y == 0).sum())}"
        )
    frac_pos = float(y.sum()) / float(y.size)
    if abs(frac_pos - 0.5) > float(BALANCE_TOL):
        raise BuildContractError(
            f"split={split!r} 正样本占比 {frac_pos:.6f} 偏离 0.5 超过容差 {BALANCE_TOL}"
        )
    bag_cols = 0 if bool(no_bag) else int(hash_dim)
    if bag_cols > 0:
        hash_norms = np.sqrt((X[:, :bag_cols].astype(np.float64) ** 2).sum(axis=1))
        n_zero_norm = int(np.count_nonzero(hash_norms <= 0.0))
        n_bad_norm = int(np.count_nonzero(np.abs(hash_norms - 1.0) > float(L2_NORM_ABS_TOL)))
        if n_bad_norm or n_zero_norm:
            raise BuildContractError(
                f"split={split!r} 的哈希词袋块 L2 归一化不成立：零范数行 {n_zero_norm}，"
                f"范数偏离 1 的行 {n_bad_norm}"
            )
    else:
        hash_norms = np.zeros(int(X.shape[0]), dtype=np.float64)
        n_zero_norm = int(X.shape[0])
        n_bad_norm = 0
    stats.update(
        {
            "positive_fraction": frac_pos,
            "pool_documents": len(pool),
            "bag_columns": bag_cols,
            "hash_block_l2_min": (float(hash_norms.min()) if bag_cols > 0 else None),
            "hash_block_l2_max": (float(hash_norms.max()) if bag_cols > 0 else None),
            "hash_block_zero_norm_rows": (n_zero_norm if bag_cols > 0 else None),
            "doc_kind_counts": (
                kind_counts_pre
                if kind_counts_pre is not None
                else {k: sum(1 for s in samples if s.doc_kind == k) for k in DOC_KINDS}
            ),
            "feature_min": float(X.min()),
            "feature_max": float(X.max()),
        }
    )
    # [!] rich / compact 的额外统计写在**独立子段**（不塞进 counts）：``counts`` 段的键序必须与
    #     历史产物一致（base 逐字节不变），而 rich / dev 是新产物，另有子段更清晰、也不污染口径。
    extra: Dict[str, Any] = {}
    if feat == "rich":
        # 列起点 = hash_dim + 基础附加列数（no_bag 时 hash_dim 语义为 0 -> 起点 6）
        col_base = (0 if bool(no_bag) else int(hash_dim)) + int(EXTRA_DIM)
        sep_idx = col_base
        sep = X[:, col_base : col_base + 1].astype(np.float64)
        rich_cols = [c["name"] for c in feature_columns(int(hash_dim), bool(no_bag), feat)]
        extra["rich"] = {
            "column_start": int(col_base),
            "columns": rich_cols[-int(RICH_EXTRA_DIM) :],
            "qcov_idf_mean": float(sep.mean()),
            "qcov_idf_max": float(sep.max()),
        }
        if idf is not None:
            extra["idf"] = {
                "vocab_size": int(idf.size),
                "corpus_documents": int(idf.n_docs),
                "formula": "idf(token) = log((1 + M) / (1 + df(token))) + " + str(IDF_SMOOTH_OFFSET),
            }
    if preload is not None:
        extra["compact"] = {
            "corpus_documents": int(preload["corpus_documents"]),
            "idf_vocab_size": int(preload["idf_vocab_size"]),
            "docpack_bytes": int(preload["packed"].nbytes),
            "doc_token_total": int(preload["packed"].ntokens),
        }
    if extra:
        stats["extra"] = extra
    return X, y, samples, stats


def _cache_qstats(
    cache: TextFeatureCache, q_key: str, idf: Optional[IdfTable], features: str
) -> Tuple[Set[str], Dict[int, int], float, float]:
    """问题侧取数（in-memory / compact 共用）：``(token 集合, 计数表, idf 和, 范数)``。"""
    q_set = cache.sets[q_key]
    if features != "rich":
        return q_set, {}, 0.0, 0.0
    counts = _token_counts_from_cache(cache, q_key)
    s, _q, _idf_norm = _idf_weights(sorted(counts.keys()), idf)
    # [!] 列定义的分母是 **TF-IDF 向量的 L2 范数**（w = tf * idf），不是纯 IDF 范数：
    #     用纯 IDF 范数会让"余弦"偏大甚至 >1（实测 42.3% 的行 >1）。两条路径统一用 _tfidf_norm。
    return q_set, counts, s, _tfidf_norm(counts, idf)


def _cache_dstats(
    cache: TextFeatureCache, member: str, idf: Optional[IdfTable], features: str
) -> Tuple[Sequence[str], Set[str], Dict[int, int], float, float]:
    """文档侧取数（in-memory 路径）：``(token 序列, 集合, 计数表, idf 和, 范数)``。"""
    d_tokens = cache.tokens[member]
    d_set = cache.sets[member]
    if features != "rich":
        return d_tokens, d_set, {}, 0.0, 0.0
    counts = _token_counts_from_cache(cache, member)
    s, _q, _idf_norm = _idf_weights(sorted(counts.keys()), idf)
    return d_tokens, d_set, counts, s, _tfidf_norm(counts, idf)


def _compact_dstats(
    member: str,
    q_set: Set[str],
    idf: Optional[IdfTable],
    features: str,
    view_fn: Callable[[str], Dict[str, Any]],
) -> Tuple[Sequence[str], Set[str], Dict[int, int], float, float]:
    """文档侧取数（compact 路径）：从紧凑表示解包并算 per-doc 聚合量。

    ``compact`` 路径下 token 为整数标识（``hash_token_id``），故第二个返回值是**空集合**：
    逐对特征（覆盖率 / Jaccard）按计数表的键集合求，与整数标识口径一致。
    """
    # [!] 先判 features 再取 view：base 口径不需要 IDF 聚合量，若先取 view 会在
    #     ``idf=None`` 时对 ``idf.lookup`` 报 AttributeError（且是纯浪费）。
    # [!] compact 的文档 token 是**整数标识**：集合必须由标识数组构造（``set(arr.tolist())``），
    #     不能返回空集合——空集合会让 build_feature_vector 回落"计数表键集合"分支，而 base 口径
    #     下根本没有计数表，于是 q_to_d_coverage / d_to_q_coverage / jaccard 三列**恒为 0**（静默错）。
    if features != "rich":
        uq = view_fn(member)["tokens"]
        return uq, set(uq.tolist()), None, 0.0, 0.0
    view = view_fn(member)
    uq = view["tokens"]
    return uq, set(uq.tolist()), view["counts"], float(view["qsum"]), float(view["norm"])


def _collect_samples_precomputed(
    split: str,
    records: Sequence[QaRecord],
    pool: Sequence[str],
    answer_hits: Dict[Tuple[str, str], int],
    negative_seed: int,
    negative_ratio: float,
) -> Tuple[List[Sample], Dict[str, Any]]:
    """compact 路径的采样（答案命中由**流式扫描时**预计算，见 :func:`preload_pairs_split`）。

    与 :func:`_collect_samples` 的差异只有一处：答案命中不再从文档全文现算，而是查
    ``answer_hits[(QuestionId, 文档成员名)]``（**同一判定函数在同一遍扫描中算出的同一结果**）。
    """
    if float(negative_ratio) != 1.0:
        raise BuildContractError(
            f"negative_ratio 口径写死为 1.0（1:1 均衡），当前 {negative_ratio}"
        )
    rng = np.random.default_rng(int(negative_seed))
    docs = sorted({m for r in records for m in r.doc_members})
    kind_counts = {k: sum(1 for m in docs if _doc_kind(m) == k) for k in DOC_KINDS}
    per_question: Dict[str, Tuple[List[Sample], List[Sample]]] = {}
    n_pos = n_neg = 0
    n_hit_pos = n_hit_neg = 0
    for r in records:
        pos: List[Sample] = []
        for member in r.doc_members:
            hit = int(answer_hits[(r.question_id, member)])
            pos.append(Sample(r.question_id, member, _doc_kind(member), 1, hit))
            n_pos += 1
            n_hit_pos += hit
        k = int(round(len(pos) * float(negative_ratio)))
        neg: List[Sample] = []
        for member in _sample_negatives(rng, pool, set(r.doc_members), k, r.question_id):
            hit = int(answer_hits[(r.question_id, member)])
            neg.append(Sample(r.question_id, member, _doc_kind(member), 0, hit))
            n_neg += 1
            n_hit_neg += hit
        per_question[r.question_id] = (pos, neg)
    samples: List[Sample] = []
    max_k = max((len(p) for p, _ in per_question.values()), default=0)
    for i in range(max_k):
        for r in records:
            pos, neg = per_question[r.question_id]
            if i < len(pos):
                samples.append(pos[i])
                samples.append(neg[i])
    if len(samples) != n_pos + n_neg:
        raise BuildContractError(f"行序组装后样本数不符：{len(samples)} != {n_pos} + {n_neg}")
    for i, s in enumerate(samples):
        expect = 1 if (i % 2 == 0) else 0
        if s.label != expect:
            raise BuildContractError(
                f"行序未做到严格正负交替：第 {i} 行 label={s.label}，期望 {expect}"
            )
    stats: Dict[str, Any] = {
        "positive": int(n_pos),
        "negative": int(n_neg),
        "positive_fraction": float(n_pos) / float(max(n_pos + n_neg, 1)),
        "questions": len(records),
        "questions_without_evidence_ref": sum(1 for r in records if not r.doc_members),
        "pool_documents": len(pool),
        "positive_answer_hit": int(n_hit_pos),
        "negative_answer_hit": int(n_hit_neg),
        "doc_kind_counts": kind_counts,
    }
    return samples, stats


def _doc_kind(member: str) -> str:
    """按归档成员名前缀判断文档类别（``wikipedia`` / ``web``）。

    参数
    ----
    member : str
        归档成员名。

    返回
    ----
    str
        文档类别；未知前缀时报错（口径写死，不静默归类）。

    异常
    ------
    BuildContractError
        成员名前缀不在写死的两类目录下时抛出。
    """
    for kind in DOC_KINDS:
        if member.startswith(EVIDENCE_DIR[kind] + "/"):
            return kind
    raise BuildContractError(f"未知的文档目录前缀：{member}")


# ======================================================================
# 产物落盘（确定性 zip：同参数重复构建逐字节一致）
# ======================================================================
def save_npz_deterministic(path: str, X: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> None:
    """把 ``X`` / ``y`` / ``meta`` 写成**逐字节确定**的 ``.npz``。

    为什么要自写 zip
    ----------------
    ``np.savez`` 会把**当前时间**写进 zip 成员的时间戳，导致同数据两次落盘字节不同
    （E1「连跑两次 SHA256 相同」会直接失败）。本函数改为：

    * 用 ``zipfile.ZipFile`` 手写成员，全部成员的 ``date_time`` 固定为
      ``(1980, 1, 1, 0, 0, 0)``、``create_system=0``、``external_attr`` **尝试置 0**；
      [!] 离朱 R26 缺陷 D2：CPython 3.12 的 ``zipfile`` 在 ``external_attr == 0`` 时会把它
      改写为 ``0o600 << 16``（实测中心目录三成员均为 ``0x01800000``，与 ``np.savez`` 原生口径
      一致）——该字段的最终取值由解释器决定、不影响逐字节确定性（同数据两次落盘 SHA256 相同）；
    * 每个成员的载荷由 ``np.lib.format.write_array(..., allow_pickle=False)`` 生成
      （纯确定性，不含时间戳/环境信息）；
    * 先写临时文件再 ``os.replace``，避免半成品覆盖正式产物。

    键名与契约
    ----------
    * ``X`` -> ``X.npy``：float32 ``[M, D]``（C 连续，保证 header 里 ``fortran_order=False``）；
    * ``y`` -> ``y.npy``：int64 ``[M]``；
    * ``meta`` -> ``meta.npy``：0 维 Unicode 数组，内容为**紧凑 JSON 字符串**
      （``allow_pickle=False`` 也能读；``n3d_shape.data.load_npz_arrays`` 只挑 ``X``/``y``，
      多余键不影响其契约）。

    参数
    ----
    path : str
        目标 ``.npz`` 路径（父目录自动创建）。
    X : np.ndarray
        特征矩阵。
    y : np.ndarray
        标签向量。
    meta : Dict[str, Any]
        元信息（必须可 JSON 序列化；**不得含时间戳等非确定性字段**）。

    返回
    ----
    None
    """
    meta_json = json.dumps(meta, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
    members: Tuple[Tuple[str, np.ndarray], ...] = (
        ("X.npy", np.ascontiguousarray(X, dtype=np.float32)),
        ("y.npy", np.ascontiguousarray(y, dtype=np.int64)),
        ("meta.npy", np.array(meta_json)),
    )
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, arr in members:
            buf = io.BytesIO()
            np.lib.format.write_array(buf, arr, allow_pickle=False)
            info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            zf.writestr(info, buf.getvalue())
    os.replace(tmp, path)


def read_npz_meta(path: str) -> Dict[str, Any]:
    """回读 ``.npz`` 里的 ``meta``（供 :mod:`n3d_qa.verify_dataset` 校验口径）。

    参数
    ----
    path : str
        ``.npz`` 路径。

    返回
    ----
    Dict[str, Any]
        解析后的 meta 字典。

    异常
    ------
    FileNotFoundError
        文件不存在时抛出。
    BuildContractError
        缺 ``meta`` 键或 meta 不是合法 JSON 时抛出。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"npz 不存在：{path}")
    with np.load(path, allow_pickle=False) as npz:
        if "meta" not in npz.files:
            raise BuildContractError(f"{path} 缺少 meta 键（实际键：{list(npz.files)}）")
        raw = npz["meta"]
    text = str(raw.item()) if getattr(raw, "shape", ()) == () else str(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - 自产自销，正常不会发生
        raise BuildContractError(f"{path} 的 meta 不是合法 JSON：{exc}") from exc


def _features_meta(
    hash_dim: int,
    no_bag: bool,
    features: str = FEATURES_DEFAULT,
    stats: Optional[Dict[str, Any]] = None,
    split_idf: Optional[IdfTable] = None,
) -> Dict[str, Any]:
    """按**实际生效口径**组装 meta 的 ``features`` 段（供 E0/E2/E5 回读）。

    参数
    ----
    hash_dim : int
        哈希词袋块维数（``no_bag=True`` 时忽略）。
    no_bag : bool
        是否未产出词袋块。
    features : str
        特征集合名（``base`` / ``rich``）；``rich`` 时附带 IDF 语料口径与 TF-IDF 定义。
    stats : Optional[Dict[str, Any]]
        构建统计（``rich`` 时用于回填 IDF 语料的文档数与词表规模；``base`` 时忽略）。

    返回
    ----
    Dict[str, Any]
        ``feature_dim`` / ``hash_dim`` / ``extra_dim`` / ``no_bag`` / 逐列定义；
        ``no_bag=False`` 时才附带哈希算法、摘要长度、盐与落桶规则
        （``no_bag=True`` 时这些字段**不写**，避免"写了却没产出该块"的口径歧义）。
    """
    feat = _check_features(features)
    base = 0 if bool(no_bag) else int(hash_dim)
    pool_documents = int((stats or {}).get("pool_documents", 0))
    vocab_size = (stats or {}).get("idf_vocab_size")
    if vocab_size is None and split_idf is not None:
        vocab_size = int(split_idf.size)
    out: Dict[str, Any] = {
        "feature_dim": feature_dim(base, bool(no_bag), feat),
        "dtype": "float32",
        "hash_dim": base,
        "extra_dim": extra_dim_for(feat),
        "no_bag": bool(no_bag),
        "token_regex": TOKEN_RE.pattern,
        "tokenizer": "整体小写后按 [a-z0-9]+ 抽取（保留重数）",
        "columns": [dict(c) for c in feature_columns(base, bool(no_bag), feat)],
    }
    if base > 0:
        out.update(
            {
                "hash_algo": "blake2b",
                "hash_digest_size": int(HASH_DIGEST_SIZE),
                "hash_salt_hex": HASH_SALT.hex(),
                "hash_bucket_rule": (
                    "int.from_bytes(blake2b(salt+token).digest(), 'big') % " + str(base)
                ),
                "hash_block_l2_normalized": True,
            }
        )
    if feat == "rich":
        out["feature_block"] = "rich"
        out["base_extra_dim"] = int(EXTRA_DIM)
        out["rich_extra_dim"] = int(RICH_EXTRA_DIM)
        out["idf"] = {
            "formula": (
                "idf(token) = log((1 + M) / (1 + df(token))) + " + str(IDF_SMOOTH_OFFSET)
            ),
            "corpus": "本次构建该 split 的文档池（= meta.counts.pool_documents 条）",
            "M": int(pool_documents),
            "df_scope": "含该 token 的池内文档数（同一文档内按唯一 token 标识去重）",
            "token_key": (
                "blake2b(salt + token, digest_size=" + str(HASH_TOKEN_ID_BYTES) + ") 的大端整数标识"
            ),
            "unknown_token_idf": "未登记 token 取 df=0 时的取值（保证 IDF 恒正）",
            "vocab_size": (int(vocab_size) if vocab_size is not None else None),
            "smooth_offset": float(IDF_SMOOTH_OFFSET),
        }
        out["tfidf"] = {
            "tf": "该 token 在文本中的出现次数（含重数）",
            "weight": "w(t) = tf(t) * idf(t)",
            "cosine": "dot(w_q, w_d) / (||w_q||_2 * ||w_d||_2)（任一范数为 0 时取 0）",
        }
        out["determinism"] = "IDF/TF-IDF 全部为确定性整数统计，不消耗全局 RNG"
    return out


def build_meta(
    split: str,
    head: Dict[str, Any],
    records: Sequence[QaRecord],
    samples: Sequence[Sample],
    stats: Dict[str, Any],
    archive_path: str,
    archive_sha256: str,
    negative_seed: int,
    max_questions: int,
    out_name: str,
    hash_dim: int = HASH_DIM,
    no_bag: bool = False,
    features: str = FEATURES_DEFAULT,
    split_idf: Optional[IdfTable] = None,
) -> Dict[str, Any]:
    """组装写入产物的 meta（**构建口径的单一事实来源**，全部字段可回读比对）。

    参数
    ----
    split : str
        ``wiki`` / ``web``。
    head : Dict[str, Any]
        :func:`parse_qa_json` 返回的头部信息。
    records : Sequence[QaRecord]
        QA 记录。
    samples : Sequence[Sample]
        逐样本 provenance。
    stats : Dict[str, Any]
        :func:`build_split` 返回的构建统计。
    archive_path : str
        归档路径。
    archive_sha256 : str
        归档实测 SHA256。
    negative_seed : int
        负样本采样种子。
    max_questions : int
        单 split 题目数上限（``0`` = 全量）。
    out_name : str
        产物文件名（不含目录）。
    hash_dim : int
        哈希词袋块维数（写入 meta 的 ``features.hash_dim``，供验证时回读比对；
        ``no_bag=True`` 时该字段写 0）。
    no_bag : bool
        是否未产出词袋块（写入 meta 的 ``features.no_bag``）。
    features : str
        特征集合名（``base`` / ``rich``；写入 meta 的 ``features`` 段）。

    返回
    ----
    Dict[str, Any]
        meta 字典（无任何时间戳 / 环境指纹，保证产物逐字节可复现）。
    """
    documents = sorted({s.doc_member for s in samples})
    doc_index = {d: i for i, d in enumerate(documents)}
    q_index = {r.question_id: i for i, r in enumerate(records)}
    q_rows = [0] * len(records)
    for s in samples:
        if s.label == 1:
            q_rows[q_index[s.question_id]] += 1
    rel_archive = os.path.relpath(archive_path, PROJECT_ROOT).replace("\\", "/")
    return {
        # [!] meta["module"] 是**产物字节的一部分**：已发布的 8 个正式产物 meta 里写的都是
        #     "n3d_triviaqa"，把它改成 "n3d_qa" 会让产物 SHA256 全变（硬闸门禁止）。
        #     故该字符串**冻结不动**，它标注的是 TriviaQA 参考实现的产物归属。
        "module": "n3d_triviaqa",
        "out_name": out_name,
        "format_version": 1,
        "task": "evidence_passage_binary_classification",
        "source": {
            "split": split,
            "qa_member": head["qa_member"],
            "domain": head["domain"],
            "qa_split": head["qa_split"],
            "version": head["version"],
            "verified_eval": head["verified_eval"],
            # [!] dev split 的剔除口径（题数差 + 来源成员序号）必须可回读，否则"按 QuestionId
            #     剔除 verified 子集"这件事在产物里无据可查。缺该键时不写（保持 base 产物字节不变）。
            **(
                {"verified_exclusion": head["verified_exclusion"]}
                if head.get("verified_exclusion")
                else {}
            ),
            "archive": rel_archive,
            "archive_sha256": archive_sha256,
            "archive_size_bytes": ARCHIVE_SIZE_BYTES,
            "archive_member_count": ARCHIVE_MEMBER_COUNT,
            "archive_uncompressed_bytes": ARCHIVE_UNCOMPRESSED_BYTES,
            # [!] 这里**只写口径、不写遍数**：实际扫描遍数取决于 QA JSON 缓存是否命中
            #     （冷 2 遍 / 暖 1 遍），若把遍数写进 meta，产物字节就会随缓存状态而变，
            #     "同参数重复构建逐字节一致"将不再无条件成立。遍数改为只印在运行日志里。
            "archive_access": (
                "tarfile mode='r|gz' 顺序流式扫描 + extractfile 按需抽取（从不 extractall、"
                "从不整包解压）；QA JSON 优先走 _cache/<sha16>_<split>_qa.json 缓存，"
                "实际扫描遍数见运行日志"
            ),
        },
        "mapping_rules": {
            "wikipedia": "EntityPages[].Filename -> evidence/wikipedia/<basename>（扁平目录）",
            "web": "SearchResults[].Filename -> evidence/web/<该相对路径>（数字子目录）",
        },
        "label_rule": {
            "positive": "label=1：文档来自该题的 EntityPages ∪ SearchResults（provenance 判定）",
            "negative": "label=0：由固定 seed 从同 split 内其他题的文档中无放回采样",
            "negative_ratio": "1:1（正负各 50%）",
            "negative_seed": int(negative_seed),
            "negative_pool": "本 split 全部正样本文档的并集（升序），排除该题自身文档",
            "row_order": (
                "轮次 i 从 0 到 max_k-1、每轮按 QuestionId 升序遍历题目，逐题依次发出"
                "第 i 个正样本与第 i 个负样本 -> 全表严格正负交替（偶数为正、奇数为负）"
            ),
        },
        "answer_match_rule": {
            "case_insensitive": True,
            "word_boundary": "\\b（首/末为非词字符时相应省略，避免 C++ 这类答案永不命中）",
            "sources_merged": ["Answer.Value", "Answer.NormalizedValue", "Answer.Aliases", "Answer.NormalizedAliases"],
            "sources_excluded": ["Answer.HumanAnswers"],
            "short_numeric_max_len": int(SHORT_NUMERIC_MAX_LEN),
            "short_numeric_protection": "(?<![\\w.])<ans>(?![\\w.])：额外禁止与相邻数字/小数点连写（防 3 命中 3.14）",
        },
        "features": _features_meta(int(hash_dim), bool(no_bag), features, stats, split_idf),
        "counts": dict(stats),
        "questions": [
            {
                "question_id": r.question_id,
                "text": r.question,
                "n_positive_docs": len(r.doc_members),
                "numeric_answer_flag": float(r.numeric_answer_flag),
            }
            for r in records
        ],
        "documents": documents,
        "samples": [
            [q_index[s.question_id], doc_index[s.doc_member], int(s.label), int(s.answer_hit)]
            for s in samples
        ],
        "max_questions": int(max_questions),
        # [!] 离朱 R26 缺陷 D4：把易误读的计数字段定义写进 meta（口径单自解释）
        "notes": {
            "questions_without_evidence_ref": (
                "题目计数口径：``EntityPages`` 与 ``SearchResults`` **两类证据引用都为空**的题目条数"
                "（只空一类不计入）"
            ),
            "positive_answer_hit_/_negative_answer_hit": (
                "答案命中只作 provenance 诊断写入 meta，**不参与任何特征列**"
                "（列 numeric_answer_flag 只由问题侧答案集合决定），故不构成标签泄漏"
            ),
            "documents": "本 split 负样本候选池 = 纳入题目的正样本文档并集（升序去重）",
        },
    }


# ======================================================================
# CLI
# ======================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析命令行参数。

    参数
    ----
    argv : Optional[Sequence[str]]
        参数列表（``None`` 表示用 ``sys.argv``）。

    返回
    ----
    argparse.Namespace
        含 split / archive / out_dir / max_questions / negative_seed /
        no_cache / refresh_cache / allow_missing_evidence / progress 字段。
    """
    parser = argparse.ArgumentParser(
        description=(
            "TriviaQA 证据段落二分类数据集构建（输出 X[M,D] float32 / y[M] int64 的 npz；"
            f"D 缺省 {HASH_DIM + EXTRA_DIM}，--no-bag 为 {EXTRA_DIM}，原计划口径 "
            f"{HASH_DIM_PLAN_ORIGINAL + EXTRA_DIM} 已作失败对照留档）"
        )
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=list(SPLIT_CHOICES),
        help=(
            "构建哪个 split：wiki / web（verified 子集，产物名带 verified）/ wiki-dev / web-dev"
            "（非 verified 全量 dev，按 QuestionId 剔除 verified 子集后构建）/ all（缺省，"
            "只展开既有两个 verified split）"
        ),
    )
    parser.add_argument(
        "--archive",
        type=str,
        default=DEFAULT_ARCHIVE,
        help=f"triviaqa-rc.tar.gz 路径（缺省 {os.path.relpath(DEFAULT_ARCHIVE, PROJECT_ROOT)}）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=(
            "产物目录（缺省 checkpoints/triviaqa/；--max-questions > 0 的演练构建缺省写 "
            "checkpoints/triviaqa/_verify/，不覆盖正式产物）"
        ),
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help=(
            "单 split 只用前 K 题（按 QuestionId 升序；0 = 全量）。用于"
            "「先单条端到端演练，再放全量」。注意：K 会同时缩小负样本候选池（口径写进 meta），"
            "K 过小（如 wiki 取 2 题）会出现「候选池不足以为某题凑 1:1 负样本」而报错退码 2"
        ),
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=None,
        help=(
            f"哈希词袋块维数（缺省 = {HASH_DIM} -> D = {HASH_DIM} + 6 = {FEATURE_DIM}，即缺省产物口径）。"
            f"原计划口径 {HASH_DIM_PLAN_ORIGINAL}（D={HASH_DIM_PLAN_ORIGINAL + EXTRA_DIM}）已实测不达标"
            "（1024 维词袋块 vs 1024 条训练样本，该块退化为噪声），仅作失败对照保留能力。"
            "非缺省值落 _verify/ 且文件名带 _hN；下游训练必须用 --input-dim (hash_dim + 6)。"
            "与 --no-bag 互斥"
        ),
    )
    parser.add_argument(
        "--no-bag",
        dest="no_bag",
        action="store_true",
        help=(
            "不产出哈希词袋块，只用 6 列附加特征（q->d 覆盖率 / d->q 覆盖率 / Jaccard / "
            "log1p 文档 token 数 / log1p 问题 token 数 / 数字答案标记），产出 D=6，"
            "文件名带 _nobag。与 --hash-dim 互斥"
        ),
    )
    parser.add_argument(
        "--negative-seed",
        type=int,
        default=DEFAULT_NEGATIVE_SEED,
        help=f"负样本采样种子（缺省写死值 {DEFAULT_NEGATIVE_SEED}）",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=FEATURES_DEFAULT,
        choices=list(FEATURE_SET_CHOICES),
        help=(
            "特征集合：base（缺省，既有 6 列附加特征）/ rich（再追加 4 列 IDF/TF-IDF 特征："
            "qcov_idf / dcov_idf / tfidf_cos / ans_isnum_frac）。rich 属新增实验产物，缺省落 "
            "checkpoints/triviaqa/_verify/ 且文件名带 _rich，不覆盖既有正式产物"
        ),
    )
    parser.add_argument("--no-cache", action="store_true", help="不使用 QA JSON 缓存（每遍都现场抽取）")
    parser.add_argument("--refresh-cache", action="store_true", help="忽略已有缓存并重写缓存")
    parser.add_argument(
        "--allow-missing-evidence",
        action="store_true",
        help="归档中缺失的 evidence 文档按「丢弃该文档」处理（缺省为严格：直接报错退码 2）",
    )
    return parser.parse_args(argv)


def same_archive(a: str, b: str = DEFAULT_ARCHIVE) -> bool:
    """判断两个归档路径是否指向同一个文件（大小写与分隔符不敏感）。

    参数
    ----
    a : str
        归档路径一。
    b : str
        归档路径二（缺省 = 写死的计划归档路径）。

    返回
    ----
    bool
        归一化后是否相同（``abspath`` + ``normcase``）。
    """
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def archive_tag(archive_path: str) -> str:
    """非缺省归档的**短标识**（用于产物文件名后缀，杜绝同名互覆）。

    参数
    ----
    archive_path : str
        归档路径。

    返回
    ----
    str
        ``abspath`` 的 SHA256 前 8 位十六进制；缺省归档返回空串。
    """
    if same_archive(archive_path):
        return ""
    digest = hashlib.sha256(
        os.path.normcase(os.path.abspath(archive_path)).encode(TEXT_ENCODING)
    ).hexdigest()
    return digest[:8]


def resolve_out_dir(args: argparse.Namespace) -> str:
    """决定产物目录（**非计划口径 / 非计划数据源的构建一律不得落进正式目录**）。

    口径
    ----
    * 显式 ``--out-dir`` 优先；
    * 出现下列任一情形时缺省落 ``checkpoints/triviaqa/_verify/``，**绝不覆盖正式产物**：
      ``--max-questions > 0``（演练构建）、``--hash-dim != HASH_DIM``（非计划维度）、
      **``--archive`` 不是写死的计划归档**（[!] 离朱 R26 缺陷 D1：早期版本漏了这一条，
      用自定义归档跑实验会以正式文件名静默覆盖正式产物；现已补上归档守卫）；
    * 其余落 ``checkpoints/triviaqa/``。

    参数
    ----
    args : argparse.Namespace
        命令行参数。

    返回
    ----
    str
        产物目录的绝对路径。
    """
    if args.out_dir:
        return os.path.abspath(args.out_dir)
    no_bag = bool(getattr(args, "no_bag", False))
    explicit_dim = getattr(args, "hash_dim", None)
    dim = int(HASH_DIM) if explicit_dim is None else int(explicit_dim)
    feat = str(getattr(args, "features", FEATURES_DEFAULT))
    if (
        int(args.max_questions) > 0
        or (not no_bag and dim != int(HASH_DIM))
        or feat != FEATURES_DEFAULT
        or not same_archive(str(args.archive))
    ):
        return VERIFY_OUT_DIR
    return DEFAULT_OUT_DIR


def out_name_for(
    split: str,
    max_questions: int,
    hash_dim: int = HASH_DIM,
    archive_path: str = DEFAULT_ARCHIVE,
    no_bag: bool = False,
    features: str = FEATURES_DEFAULT,
) -> str:
    """产物文件名（**口径标签恒定在名内**，避免不同口径产物同名互覆）。

    后缀口径（可叠加）
    ------------------
    * ``_hN``：产出词袋块且维数为 ``N``（缺省口径 ``N = HASH_DIM = 64`` 也**显式带上**，
      即 D=70 版为 ``..._dev_h64.npz``）；
    * ``_nobag``：``--no-bag``（不产出词袋块，base 为 D=6）；
    * ``_qK``：``--max-questions K``（演练构建）；
    * ``_arch<8 位 hex>``：``--archive`` 非计划归档。

    参数
    ----
    split : str
        ``wiki`` / ``web``。
    max_questions : int
        题目数上限（``0`` = 全量）。
    hash_dim : int
        ``no_bag=False`` 时的词袋块维数。
    archive_path : str
        归档路径（等于计划归档时不加 ``_arch`` 后缀）。
    no_bag : bool
        是否不产出词袋块。
    features : str
        特征集合名（``base`` 不加标签、``rich`` 加 ``_rich``）。

    返回
    ----
    str
        文件名（不含目录）。
    """
    name = out_name_template(split).format(slug=split_slug(split))
    suffix = "_nobag" if bool(no_bag) else f"_h{int(hash_dim)}"
    if _check_features(features) == "rich":
        suffix += "_rich"
    if int(max_questions) > 0:
        suffix += f"_q{int(max_questions)}"
    tag = archive_tag(archive_path)
    if tag:
        suffix += f"_arch{tag}"
    return name[: -len(".npz")] + suffix + ".npz"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口：构建指定 split 的 npz。

    参数
    ----
    argv : Optional[Sequence[str]]
        参数列表（``None`` 表示 ``sys.argv``）。

    返回
    ----
    int
        退出码：``0`` 成功；``2`` 归档校验失败 / 映射未命中 / 契约断言失败。

    异常
    ------
    SystemExit
        argparse 在参数非法时抛出（退出码 2）。
    """
    args = parse_args(argv)
    no_bag = bool(args.no_bag)
    features = str(getattr(args, "features", FEATURES_DEFAULT))
    if features not in FEATURE_SET_CHOICES:
        print(
            f"[n3d_qa] [FAIL] 未知 --features={features}；可选 {list(FEATURE_SET_CHOICES)}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    if no_bag and args.hash_dim is not None:
        print(
            "[n3d_qa] [FAIL] --no-bag 与 --hash-dim 互斥（--no-bag 已不产出词袋块，"
            f"当前同时给了 --hash-dim {args.hash_dim}）",
            file=sys.stderr,
            flush=True,
        )
        return 2
    hash_dim = int(HASH_DIM) if args.hash_dim is None else int(args.hash_dim)
    if not no_bag and hash_dim <= 0:
        print(f"[n3d_qa] [FAIL] --hash-dim 必须 > 0，当前 {hash_dim}", file=sys.stderr, flush=True)
        return 2
    splits: Tuple[str, ...] = (
        tuple(SPLIT_ALL_EXPANSION) if args.split == "all" else (str(args.split),)
    )
    t0 = time.time()
    print(
        f"[n3d_qa] 特征口径：features={features} / "
        + ("--no-bag（无词袋块）" if no_bag else f"hash_dim={hash_dim}")
        + f" -> D = {feature_dim(hash_dim, no_bag, features)}"
        + (
            "（缺省口径）"
            if (not no_bag and hash_dim == int(HASH_DIM) and features == FEATURES_DEFAULT)
            else ""
        ),
        flush=True,
    )
    if (not no_bag) and hash_dim != int(HASH_DIM):
        print(
            f"[n3d_qa] [WARN] --hash-dim={hash_dim} 非缺省口径 {HASH_DIM}"
            f"（D = {hash_dim} + {EXTRA_DIM} = {feature_dim(hash_dim)}）；"
            f"产物落 _verify/，下游训练需 --input-dim {feature_dim(hash_dim)}",
            flush=True,
        )

    try:
        archive_sha = verify_archive(args.archive)
    except ArchiveIntegrityError as exc:
        print(f"[n3d_qa] [FAIL] 归档校验失败：{exc}", file=sys.stderr, flush=True)
        return 2
    print(
        f"[n3d_qa] 归档校验通过：SHA256={archive_sha}（{ARCHIVE_SIZE_BYTES} 字节，"
        f"{ARCHIVE_MEMBER_COUNT} 个成员）",
        flush=True,
    )

    try:
        raw_qa, passes_a, cache_status, qa_members = load_archive_qa(
            args.archive,
            archive_sha,
            splits,
            use_cache=not bool(args.no_cache),
            refresh_cache=bool(args.refresh_cache),
        )
        for s in cache_status:
            print(f"[n3d_qa]   QA 来源 {s}", flush=True)
        parsed: Dict[str, Tuple[List[QaRecord], Dict[str, Any]]] = {}
        for split in splits:
            parsed[split] = parse_qa_json(raw_qa[split], split)
            head = parsed[split][1]
            print(
                f"[n3d_qa]   {split}: {head['questions']} 题"
                f"（无 evidence 引用的题目 {head['questions_without_evidence_ref']} 条）",
                flush=True,
            )
        if int(args.max_questions) > 0:
            parsed = {
                s: (
                    parsed[s][0][: int(args.max_questions)],
                    parsed[s][1],
                )
                for s in splits
            }
            print(
                f"[n3d_qa]   --max-questions={int(args.max_questions)}：每个 split 取前 "
                f"{int(args.max_questions)} 题（按 QuestionId 升序）",
                flush=True,
            )
        wanted: Set[str] = set()
        wanted_inmem: Set[str] = set()
        for s in splits:
            wanted |= {m for r in parsed[s][0] for m in r.doc_members}
            if SPLIT_SCALE[s] == "inmem":
                wanted_inmem |= {m for r in parsed[s][0] for m in r.doc_members}
        evidence: Dict[str, bytes] = {}
        missing: List[str] = []
        if wanted_inmem:
            evidence, missing = load_archive_evidence(args.archive, wanted_inmem)
        n_wiki_hit = sum(1 for m in evidence if m.startswith(EVIDENCE_DIR["wikipedia"] + "/"))
        n_web_hit = sum(1 for m in evidence if m.startswith(EVIDENCE_DIR["web"] + "/"))
        print(
            f"[n3d_qa]   evidence 命中 {len(evidence)}/{len(wanted)}"
            f"（wikipedia {n_wiki_hit} / web {n_web_hit}），未命中 {len(missing)}",
            flush=True,
        )
        if missing and wanted_inmem:
            if not bool(args.allow_missing_evidence):
                print(
                    f"[n3d_qa] [FAIL] 归档中缺少 {len(missing)} 个 evidence 文档，"
                    f"前 10 个：{missing[:10]}（如确认可容忍，加 --allow-missing-evidence）",
                    file=sys.stderr,
                    flush=True,
                )
                return 2
            drop = set(missing)
            print(
                f"[n3d_qa] [WARN] --allow-missing-evidence：丢弃 {len(drop)} 个缺失文档",
                flush=True,
            )
            parsed = {
                s: (
                    [
                        QaRecord(
                            question_id=r.question_id,
                            question=r.question,
                            answer_values=r.answer_values,
                            answer_patterns=r.answer_patterns,
                            doc_members=tuple(m for m in r.doc_members if m not in drop),
                            numeric_answer_flag=r.numeric_answer_flag,
                            answer_prefetch=r.answer_prefetch,
                            answer_literals=r.answer_literals,
                            answer_chars=r.answer_chars,
                        )
                        for r in parsed[s][0]
                    ],
                    parsed[s][1],
                )
                for s in splits
            }
        doc_texts = {m: b.decode(TEXT_ENCODING, errors="replace") for m, b in evidence.items()}
        passes = int(passes_a) + (1 if wanted else 0)
        out_dir = resolve_out_dir(args)
        for split in splits:
            records, head = parsed[split]
            if split in VERIFIED_PARENT_SPLIT:
                # 扩容口径：dev split 按 QuestionId 剔除 verified 子集（同族 verified split 的集合）
                parent = VERIFIED_PARENT_SPLIT[split]
                if parent in parsed:
                    parent_records = parsed[parent][0]
                else:
                    parent_raw, _pp, _ps, parent_members = load_archive_qa(
                        args.archive,
                        archive_sha,
                        [parent],
                        use_cache=not bool(args.no_cache),
                    )
                    # [!] 父 split 的成员序号就在这次调用的返回值里：早期实现把它丢弃，
                    #     导致 qa_member_indices 的父序号恒为 0（真实值应为归档成员序号）。
                    qa_members.update(parent_members)
                    parent_records, _ph = parse_qa_json(parent_raw[parent], parent)
                exclude = {r.question_id for r in parent_records}
                before = len(records)
                records = [r for r in records if r.question_id not in exclude]
                head["verified_exclusion"] = {
                    "excluded_by": parent,
                    "qa_member_of_excluded": QA_JSON_MEMBERS[parent],
                    "excluded_questions": int(before - len(records)),
                    "qa_member_indices": {
                        QA_JSON_MEMBERS[parent]: int(qa_members.get(QA_JSON_MEMBERS[parent], 0)),
                        QA_JSON_MEMBERS[split]: int(qa_members.get(QA_JSON_MEMBERS[split], 0)),
                    },
                }
                print(
                    f"[n3d_qa]   {split}: 剔除 verified 子集 {before - len(records)} 题"
                    f"（{parent} 的 QuestionId 集合 {len(exclude)} 个），余 {len(records)} 题",
                    flush=True,
                )
            if SPLIT_SCALE[split] == "compact":
                # ---------------- compact 路径：两遍流式扫描（不驻留文档全文） ----------------
                raw_pool = {m for r in records for m in r.doc_members}
                pack_path = os.path.join(out_dir, out_name_for(
                    split, int(args.max_questions), hash_dim, args.archive, no_bag, features
                )[:-len(".npz")] + "_docpack.npy.npz")
                builder = DocPackBuilder(pack_path, flush_bytes=int(DEFAULT_PACK_FLUSH_BYTES))
                missing_docs: Set[str] = set()
                print(
                    f"[n3d_qa]   {split}: compact 路径（两遍流式扫描，不驻留文档全文），"
                    f"候选文档 {len(raw_pool)} 个",
                    flush=True,
                )

                def _stream(on_member, _ap=args.archive, _pool=raw_pool):
                    """单遍流式扫描：只抽文档池成员（按需抽取，不解压整包）。"""
                    print(
                        f"[n3d_qa]     stream {len(_pool)} members from archive",
                        flush=True,
                    )
                    found, miss = stream_extract_members(
                        _ap, _pool, on_member=on_member, progress_every=200000
                    )
                    del found
                    return miss, int(len(_pool)) - int(len(miss))

                t_pl = time.time()
                preload = preload_pairs_split(
                    split,
                    records,
                    raw_pool,
                    _stream,
                    features,
                    builder,
                    missing_docs,
                )
                _pf = preload["profile"]
                print(
                    f"[n3d_qa]   {split}: compact 预计算完成（用时 {time.time() - t_pl:.1f} s）："
                    f"语料文档 {preload['corpus_documents']}，IDF 词表 {preload['idf_vocab_size']}，"
                    f"紧凑表示 {preload['packed'].nbytes / 1048576.0:.1f} MiB"
                    f"（文档 token 总数 {preload['packed'].ntokens}）",
                    flush=True,
                )
                print(
                    f"[n3d_qa]     阶段耗时：第 1 遍（df+候选）{_pf['pass1_s']:.1f} s，"
                    f"第 2 遍（token+IDF+答案命中+打包）{_pf['pass2_s']:.1f} s；"
                    f"答案命中正则调用 {_pf['regex_calls']} 次",
                    flush=True,
                )
                if missing_docs:
                    if not bool(args.allow_missing_evidence):
                        print(
                            f"[n3d_qa] [FAIL] 归档中缺少 {len(missing_docs)} 个 evidence 文档，"
                            f"前 10 个：{sorted(missing_docs)[:10]}"
                            f"（如确认可容忍，加 --allow-missing-evidence）",
                            file=sys.stderr,
                            flush=True,
                        )
                        return 2
                    print(
                        f"[n3d_qa] [WARN] --allow-missing-evidence：丢弃 "
                        f"{len(missing_docs)} 个缺失文档及其样本对",
                        flush=True,
                    )
                X, y, samples, stats = build_split(
                    split,
                    records,
                    None,
                    negative_seed=int(args.negative_seed),
                    negative_ratio=1.0,
                    hash_dim=hash_dim,
                    no_bag=no_bag,
                    features=features,
                    idf=preload["idf"],
                    preload=preload,
                )
                # [!] compact 的统计只保留一处：``build_split`` 已把同一批量写进
                #     ``stats["extra"]["compact"]``，此处不再重复写 ``stats["preload"]``
                #     （两份同值字段只会让 meta 膨胀并形成"改一处忘一处"的口径漂移风险）。
                split_idf = preload["idf"]
                if int(args.max_questions) > 0:
                    keep = {r.question_id for r in records[: int(args.max_questions)]}
                    records = [r for r in records if r.question_id in keep]
            else:
                # ---------------- in-memory 路径（既有口径；base 产物逐字节不变） ----------------
                # [!] rich 且不走 compact 的 split（例如带 --max-questions 的演练构建）在
                #     调用前由 ``_build_split_idf`` 用**文档全文**现场建同口径的 IDF 表：
                #     语料 = 本次构建的文档池，与 compact 路径的语料口径一致。
                split_idf = (
                    _build_split_idf(records, doc_texts)
                    if (features == "rich" and SPLIT_SCALE[split] != "compact")
                    else None
                )
                X, y, samples, stats = build_split(
                    split,
                    records,
                    doc_texts,
                    negative_seed=int(args.negative_seed),
                    negative_ratio=1.0,
                    hash_dim=hash_dim,
                    no_bag=no_bag,
                    features=features,
                    idf=split_idf,
                )
            out_name = out_name_for(
                split, int(args.max_questions), hash_dim, args.archive, no_bag, features
            )
            out_path = os.path.join(out_dir, out_name)
            meta = build_meta(
                split,
                head,
                records,
                samples,
                stats,
                args.archive,
                archive_sha,
                int(args.negative_seed),
                int(args.max_questions),
                out_name,
                hash_dim,
                no_bag,
                features,
                split_idf,
            )
            save_npz_deterministic(out_path, X, y, meta)
            sha = sha256_file(out_path)
            print(
                f"[n3d_qa] [OK] {split}: X={X.shape}（D={int(X.shape[1])}）{X.dtype} / "
                f"y={y.shape} {y.dtype} "
                f"正 {stats['positive']} 负 {stats['negative']} "
                f"（文档池 {stats['pool_documents']}，正样本含答案 {stats['positive_answer_hit']}/"
                f"{stats['positive']}，负样本含答案 {stats['negative_answer_hit']}/{stats['negative']}）",
                flush=True,
            )
            print(
                f"[n3d_qa]      产物 {out_path}（SHA256={sha}，流式扫描 {passes} 遍）",
                flush=True,
            )
    except (BuildContractError, FileNotFoundError) as exc:
        print(f"[n3d_qa] [FAIL] {exc}", file=sys.stderr, flush=True)
        return 2
    print(f"[n3d_qa] 完成，用时 {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
