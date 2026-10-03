"""n3d_triviaqa.build_dataset —— TriviaQA「证据段落二分类」数据集的确定性构建。

职责
----
把 ``data/triviaqa/OpenDataLab___TriviaQA/raw/triviaqa-rc.tar.gz`` 中的 QA 问答对与
evidence 文档，转成 ``X[M, D] float32`` / ``y[M] int64`` 的 npz（``D = feature_dim(hash_dim, no_bag)``：
缺省 ``64 + 6 = 70``，``--no-bag`` 为 6；原计划口径 1030 已实测不达标、仅作失败对照留档），
供 ``n3d_shape`` 以 ``--dataset npz`` 直接训练。本模块**自包含**：不导入、不修改
n3d_proto / n3d_sphere / n3d_shape / n3d_viz / framework 的任何代码。

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
    "DEFAULT_NEGATIVE_SEED",
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

# split -> 归档内 QA JSON 成员名（实测成员序号 487248 / 487251）
QA_JSON_MEMBERS: Dict[str, str] = {
    "wiki": "qa/verified-wikipedia-dev.json",
    "web": "qa/verified-web-dev.json",
}
# 实测问题条数（用于"解析后条数"这一层的一致性校验）
QA_JSON_QUESTIONS: Dict[str, int] = {"wiki": 318, "web": 407}

# 文档类别 -> 归档内目录前缀（实测：evidence/web 413173 个成员、evidence/wikipedia 74070 个）
EVIDENCE_DIR: Dict[str, str] = {"wikipedia": "evidence/wikipedia", "web": "evidence/web"}
DOC_KINDS: Tuple[str, ...] = ("wikipedia", "web")

# 产物目录（产物一律 checkpoints/triviaqa/；验证类产物落 _verify/，缓存落 _cache/）
DEFAULT_OUT_DIR: str = os.path.join(PROJECT_ROOT, "checkpoints", "triviaqa")
VERIFY_OUT_DIR: str = os.path.join(DEFAULT_OUT_DIR, "_verify")
CACHE_DIR: str = os.path.join(DEFAULT_OUT_DIR, "_cache")

OUT_NAME_TEMPLATE: str = "n3d_triviaqa_verified_{split}_dev.npz"

# 特征维数：D = hash_dim（哈希词袋）+ 6（覆盖度/Jaccard/长度/数字答案标记）
#
# [!] 修订（风后 R28）：`--hash-dim` 缺省由 1024 降为 **64**（产出 D = 64 + 6 = 70）。
#     原因（实测，详见 README §7.6）：n3d_shape 数据层「每 5 取 1」切分后训练样本仅 1024 条，
#     原口径 1024 维词袋块「特征维数 ≈ 样本数」，该块退化为噪声（单块 test_acc 35.94%，
#     numpy 逻辑回归 5 折 CV 仅 wiki 0.3445 / web 0.2866，比随机还差），把端到端 test_acc
#     从 91.02%（去掉该块）拖到 59.38%。降维后同口径 20 epoch 即达标。
HASH_DIM: int = 64
# 原计划口径的哈希维度（仅作**失败对照**留档与 README/验证口径说明用；不再是缺省产出）
HASH_DIM_PLAN_ORIGINAL: int = 1024
EXTRA_DIM: int = 6
FEATURE_DIM: int = HASH_DIM + EXTRA_DIM

# 哈希词袋的确定性口径（blake2b，不消耗全局 RNG）
HASH_SALT: bytes = b"n3d-triviaqa-bow-v1\x00"
HASH_DIGEST_SIZE: int = 8

# 负样本采样种子（固定；写进 meta 供回读比对）
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

SPLIT_CHOICES: Tuple[str, ...] = ("wiki", "web", "all")

# 特征列定义（写入 meta 的 feature_columns；端点**闭区间**、按 start 升序、无缝覆盖 [0, D-1]）
def feature_columns(
    hash_dim: int = HASH_DIM, no_bag: bool = False
) -> Tuple[Dict[str, Any], ...]:
    """按**实际生效口径**生成逐列特征定义（缺省 ``hash_dim = 64`` -> ``D = 70``）。

    参数
    ----
    hash_dim : int
        哈希词袋块维数（缺省写死值 :data:`HASH_DIM` = 64）。
    no_bag : bool
        为真时**不产出词袋块**（``--no-bag``），列定义只含 6 列附加特征，总维数 ``D = 6``。

    返回
    ----
    Tuple[Dict[str, Any], ...]
        每列（或块）的 ``start`` / ``end`` / ``name`` / ``definition``；
        无缝覆盖 ``[0, feature_dim(hash_dim, no_bag) - 1]``。
    """
    base = 0 if bool(no_bag) else int(hash_dim)
    if bool(no_bag):
        return _extra_columns(base)
    return (
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
    ) + _extra_columns(base)


def _extra_columns(base: int) -> Tuple[Dict[str, Any], ...]:
    """生成 6 列附加特征（覆盖度/Jaccard/长度/数字答案标记）的列定义。

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


def feature_dim(hash_dim: int = HASH_DIM, no_bag: bool = False) -> int:
    """特征总维数 ``D = hash_dim + EXTRA_DIM``（缺省 ``64 + 6 = 70``；``no_bag`` 时为 6）。

    参数
    ----
    hash_dim : int
        哈希词袋块维数（``no_bag=True`` 时忽略）。
    no_bag : bool
        为真时不计词袋块（``--no-bag``）。

    返回
    ----
    int
        总维数 ``D``。
    """
    return (0 if bool(no_bag) else int(hash_dim)) + int(EXTRA_DIM)


class ArchiveIntegrityError(RuntimeError):
    """归档缺失或 SHA256 校验不符（CLI 层映射为退出码 2）。"""


class BuildContractError(RuntimeError):
    """构建过程中违反了写死的口径（字段缺失 / 映射未命中 / 平衡断言失败等）。"""


# ======================================================================
# 归档访问层（顺序流式；不整包解压）
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
    on_member: Optional[Callable[[str, bytes], None]] = None,
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
    on_member : Optional[Callable[[str, bytes], None]]
        目标成员的消费回调 ``(member_name, data)``；``None`` 时数据进返回字典。
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
                    f"[n3d_triviaqa]   流式扫描进度：{n_members} 成员，已命中 {len(seen)}/"
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
                on_member(name, data)
            else:
                found[name] = data
    missing = sorted(str(x) for x in (set(wanted) - seen))
    return found, missing


def _qa_cache_path(archive_sha256: str, split: str) -> str:
    """QA JSON 的缓存路径（键 = 归档 SHA256 前缀 + split，保证换归档自动失效）。"""
    return os.path.join(CACHE_DIR, f"{archive_sha256[:16]}_{split}_qa.json")


def load_archive_qa(
    archive_path: str,
    archive_sha256: str,
    splits: Sequence[str],
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Tuple[Dict[str, bytes], int, List[str]]:
    """取得各 split 的 QA JSON 原始字节（**第 1 遍流式扫描**，带缓存）。

    口径
    ----
    * 缓存文件落在 ``checkpoints/triviaqa/_cache/<sha16>_<split>_qa.json``；
      键含归档 SHA256，故"换归档"不会误用旧缓存；
    * 一次调用把**所有**待取 split 的 QA JSON 在同一遍扫描里抽出（不为每个 split 各扫一遍）；
    * 全部命中缓存时**不扫描**归档（返回的遍数为 0）。

    参数
    ----
    archive_path : str
        归档路径。
    archive_sha256 : str
        归档实测 SHA256（缓存键）。
    splits : Sequence[str]
        需要的 split 列表（``wiki`` / ``web``）。
    use_cache : bool
        是否允许读/写缓存。
    refresh_cache : bool
        为真则忽略已有缓存、强制重新抽取并覆盖缓存。

    返回
    ----
    Tuple[Dict[str, bytes], int, List[str]]
        ``(split -> QA JSON 原始字节, 实际扫描遍数, 缓存状态说明列表)``。

    异常
    ------
    BuildContractError
        归档中找不到该 split 的 QA JSON 成员时抛出。
    """
    out: Dict[str, bytes] = {}
    status: List[str] = []
    todo: List[str] = []
    for split in splits:
        path = _qa_cache_path(archive_sha256, split)
        if use_cache and not refresh_cache and os.path.isfile(path):
            with open(path, "rb") as fh:
                out[split] = fh.read()
            status.append(f"{split}:缓存命中({path})")
        else:
            todo.append(split)
    passes = 0
    if todo:
        wanted = {QA_JSON_MEMBERS[s] for s in todo}
        print(
            f"[n3d_triviaqa] 第 1 遍流式扫描：抽取 QA JSON {sorted(wanted)}（不解压整包）",
            flush=True,
        )
        got, missing = stream_extract_members(archive_path, wanted, progress_every=100000)
        passes = 1
        if missing:
            raise BuildContractError(
                f"归档中找不到 QA JSON 成员：{missing}（期望 {sorted(wanted)}）"
            )
        for split in todo:
            raw = got[QA_JSON_MEMBERS[split]]
            out[split] = raw
            if use_cache:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(_qa_cache_path(archive_sha256, split), "wb") as fh:
                    fh.write(raw)
                status.append(f"{split}:现场抽取并写入缓存({len(raw)} 字节)")
            else:
                status.append(f"{split}:现场抽取({len(raw)} 字节)")
    return out, passes, status


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
    print(f"[n3d_triviaqa] 第 2 遍流式扫描：抽取 evidence {len(wanted)} 个成员", flush=True)
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
    pats: List[re.Pattern] = []
    for raw in answer_values:
        s = str(raw).strip()
        if not s:
            continue
        esc = re.escape(s)
        if NUMERIC_ONLY_RE.match(s) and len(s) <= int(SHORT_NUMERIC_MAX_LEN):
            pats.append(re.compile(r"(?<![\w.])" + esc + r"(?![\w.])", re.IGNORECASE))
            continue
        prefix = r"\b" if WORD_HEAD_RE.match(s) else ""
        suffix = r"\b" if WORD_TAIL_RE.search(s) else ""
        pats.append(re.compile(prefix + esc + suffix, re.IGNORECASE))
    return tuple(pats)


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


class TextFeatureCache:
    """文本特征的 **memoization** 缓存（唯一的性能优化点，不改变任何计算结果）。

    不变量
    ------
    ``tokens[key]`` / ``sets[key]``（以及 ``no_bag=False`` 时的 ``bags[key]``）由**同一份文本**派生；
    缓存不含随机性、不影响行序与特征取值（同输入恒等输出）。
    """

    def __init__(self, hash_dim: int = HASH_DIM, no_bag: bool = False) -> None:
        """初始化缓存（``hash_dim``/``no_bag`` 决定是否产出词袋块，必须与构建口径一致）。

        参数
        ----
        hash_dim : int
            词袋维数（``no_bag=True`` 时忽略）。
        no_bag : bool
            为真时**完全跳过**词袋计算（既省时也保证不会误用词袋列）。
        """
        self.no_bag = bool(no_bag)
        self.hash_dim = 0 if self.no_bag else int(hash_dim)
        self.tokens: Dict[str, List[str]] = {}
        self.sets: Dict[str, Set[str]] = {}
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
        if key in self.tokens:
            return
        toks = tokenize(text)
        self.tokens[key] = toks
        self.sets[key] = set(toks)
        if not self.no_bag:
            self.bags[key] = hash_bag(toks, self.hash_dim)


def build_feature_vector(
    cache: TextFeatureCache,
    q_key: str,
    q_text: str,
    d_key: str,
    d_text: str,
    numeric_answer_flag: float,
    hash_dim: int = HASH_DIM,
    no_bag: bool = False,
) -> np.ndarray:
    """构造一行 ``X``（长度 ``feature_dim(hash_dim, no_bag)`` 的 float32 向量）。

    列定义由 :func:`feature_columns` 按同一 ``hash_dim`` 生成（与写入 meta 的内容同源，
    单一事实来源）。

    参数
    ----
    cache : TextFeatureCache
        文本特征缓存。
    q_key : str
        问题缓存键。
    q_text : str
        问题文本。
    d_key : str
        文档缓存键（归档成员名）。
    d_text : str
        文档文本。
    numeric_answer_flag : float
        末列（``numeric_answer_flag``）的取值（逐题常量）。
    hash_dim : int
        哈希词袋块维数（缺省写死值 :data:`HASH_DIM` = 64）。
    no_bag : bool
        为真时**不产出词袋块**，向量只含 6 列附加特征（``D = 6``）。

    返回
    ----
    np.ndarray
        形状 ``[feature_dim(hash_dim, no_bag)]`` 的 float32 行向量。
    """
    base = 0 if bool(no_bag) else int(hash_dim)
    if base != int(cache.hash_dim) or bool(no_bag) != bool(cache.no_bag):
        raise BuildContractError(
            f"口径与文本缓存不一致：build_feature_vector 收到 (hash_dim={base}, no_bag={no_bag})，"
            f"缓存为 (hash_dim={cache.hash_dim}, no_bag={cache.no_bag})"
        )
    cache.ensure(q_key, q_text)
    cache.ensure(d_key, d_text)
    row = np.zeros(feature_dim(base, bool(no_bag)), dtype=np.float32)
    # ---- 列 0..base-1（仅 no_bag=False 时存在）：哈希词袋（问题词袋 + 文档词袋 -> L2 归一化）----
    if base > 0:
        row[:base] = cache.bags[q_key] + cache.bags[d_key]
        l2_normalize_block(row[:base])
    # ---- 列 base+0..base+2：token 集合口径的覆盖率与 Jaccard（no_bag 时 base=0）----
    q_set = cache.sets[q_key]
    d_set = cache.sets[d_key]
    inter = len(q_set & d_set)
    union = len(q_set | d_set)
    row[base] = float(inter) / float(len(q_set)) if q_set else 0.0
    row[base + 1] = float(inter) / float(len(d_set)) if d_set else 0.0
    row[base + 2] = float(inter) / float(union) if union else 0.0
    # ---- 列 base+3..base+4：长度（含重数的 token 数）----
    row[base + 3] = float(np.log1p(len(cache.tokens[d_key])))
    row[base + 4] = float(np.log1p(len(cache.tokens[q_key])))
    # ---- 列 base+5：数字类答案标记（问题侧常量）----
    row[base + 5] = float(numeric_answer_flag)
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


def build_split(
    split: str,
    records: Sequence[QaRecord],
    doc_texts: Dict[str, str],
    negative_seed: int = DEFAULT_NEGATIVE_SEED,
    negative_ratio: float = 1.0,
    hash_dim: int = HASH_DIM,
    no_bag: bool = False,
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
    for member in pool:
        if member not in doc_texts:
            raise BuildContractError(f"split={split!r} 缺少文档内容：{member}")


    rng = np.random.default_rng(int(negative_seed))
    cache = TextFeatureCache(hash_dim=int(hash_dim), no_bag=bool(no_bag))
    for r in records:
        cache.ensure(f"q:{r.question_id}", r.question)

    per_question: Dict[str, Tuple[List[Sample], List[Sample]]] = {}
    n_pos = n_neg = 0
    n_hit_pos = n_hit_neg = 0
    for r in records:
        pos: List[Sample] = []
        for member in r.doc_members:
            text = doc_texts[member]
            hit = 1 if doc_contains_answer(text, r.answer_patterns) else 0
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
            hit = 1 if doc_contains_answer(text, r.answer_patterns) else 0
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

    # ---- 严格正负交替的行序（见 docstring）----
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

    records_by_id = {r.question_id: r for r in records}
    X = np.zeros((len(samples), feature_dim(int(hash_dim), bool(no_bag))), dtype=np.float32)
    y = np.zeros(len(samples), dtype=np.int64)
    for i, s in enumerate(samples):
        r = records_by_id[s.question_id]
        X[i] = build_feature_vector(
            cache,
            f"q:{r.question_id}",
            r.question,
            s.doc_member,
            doc_texts[s.doc_member],
            r.numeric_answer_flag,
            int(hash_dim),
            bool(no_bag),
        )
        y[i] = int(s.label)

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
    stats: Dict[str, Any] = {
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "positive_fraction": frac_pos,
        "questions": len(records),
        "questions_without_evidence_ref": sum(1 for r in records if not r.doc_members),
        "pool_documents": len(pool),
        "positive_answer_hit": int(n_hit_pos),
        "negative_answer_hit": int(n_hit_neg),
        "bag_columns": bag_cols,
        "hash_block_l2_min": (float(hash_norms.min()) if bag_cols > 0 else None),
        "hash_block_l2_max": (float(hash_norms.max()) if bag_cols > 0 else None),
        "hash_block_zero_norm_rows": (n_zero_norm if bag_cols > 0 else None),
        "doc_kind_counts": {
            k: sum(1 for s in samples if s.doc_kind == k) for k in DOC_KINDS
        },
        "feature_min": float(X.min()),
        "feature_max": float(X.max()),
    }
    return X, y, samples, stats


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
    """回读 ``.npz`` 里的 ``meta``（供 :mod:`n3d_triviaqa.verify_dataset` 校验口径）。

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


def _features_meta(hash_dim: int, no_bag: bool) -> Dict[str, Any]:
    """按**实际生效口径**组装 meta 的 ``features`` 段（供 E0/E2/E5 回读）。

    参数
    ----
    hash_dim : int
        哈希词袋块维数（``no_bag=True`` 时忽略）。
    no_bag : bool
        是否未产出词袋块。

    返回
    ----
    Dict[str, Any]
        ``feature_dim`` / ``hash_dim`` / ``extra_dim`` / ``no_bag`` / 逐列定义；
        ``no_bag=False`` 时才附带哈希算法、摘要长度、盐与落桶规则
        （``no_bag=True`` 时这些字段**不写**，避免"写了却没产出该块"的口径歧义）。
    """
    base = 0 if bool(no_bag) else int(hash_dim)
    out: Dict[str, Any] = {
        "feature_dim": feature_dim(base, bool(no_bag)),
        "dtype": "float32",
        "hash_dim": base,
        "extra_dim": int(EXTRA_DIM),
        "no_bag": bool(no_bag),
        "token_regex": TOKEN_RE.pattern,
        "tokenizer": "整体小写后按 [a-z0-9]+ 抽取（保留重数）",
        "columns": [dict(c) for c in feature_columns(base, bool(no_bag))],
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
        "features": _features_meta(int(hash_dim), bool(no_bag)),
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
        help="构建哪个 split：wiki（q/verified-wikipedia-dev）/ web / all（缺省 all）",
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
    if (
        int(args.max_questions) > 0
        or (not no_bag and dim != int(HASH_DIM))
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
) -> str:
    """产物文件名（**口径标签恒定在名内**，避免不同口径产物同名互覆）。

    后缀口径（可叠加）
    ------------------
    * ``_hN``：产出词袋块且维数为 ``N``（缺省口径 ``N = HASH_DIM = 64`` 也**显式带上**，
      即 D=70 版为 ``..._dev_h64.npz``）；
    * ``_nobag``：``--no-bag``（不产出词袋块，D=6）；
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

    返回
    ----
    str
        文件名（不含目录）。
    """
    name = OUT_NAME_TEMPLATE.format(split=split)
    suffix = "_nobag" if bool(no_bag) else f"_h{int(hash_dim)}"
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
    if no_bag and args.hash_dim is not None:
        print(
            "[n3d_triviaqa] [FAIL] --no-bag 与 --hash-dim 互斥（--no-bag 已不产出词袋块，"
            f"当前同时给了 --hash-dim {args.hash_dim}）",
            file=sys.stderr,
            flush=True,
        )
        return 2
    hash_dim = int(HASH_DIM) if args.hash_dim is None else int(args.hash_dim)
    if not no_bag and hash_dim <= 0:
        print(f"[n3d_triviaqa] [FAIL] --hash-dim 必须 > 0，当前 {hash_dim}", file=sys.stderr, flush=True)
        return 2
    splits: Tuple[str, ...] = tuple(SPLIT_CHOICES[:2]) if args.split == "all" else (str(args.split),)
    t0 = time.time()
    print(
        f"[n3d_triviaqa] 特征口径："
        + ("--no-bag（无词袋块）" if no_bag else f"hash_dim={hash_dim}")
        + f" -> D = {feature_dim(hash_dim, no_bag)}"
        + ("（缺省口径）" if (not no_bag and hash_dim == int(HASH_DIM)) else ""),
        flush=True,
    )
    if (not no_bag) and hash_dim != int(HASH_DIM):
        print(
            f"[n3d_triviaqa] [WARN] --hash-dim={hash_dim} 非缺省口径 {HASH_DIM}"
            f"（D = {hash_dim} + {EXTRA_DIM} = {feature_dim(hash_dim)}）；"
            f"产物落 _verify/，下游训练需 --input-dim {feature_dim(hash_dim)}",
            flush=True,
        )

    try:
        archive_sha = verify_archive(args.archive)
    except ArchiveIntegrityError as exc:
        print(f"[n3d_triviaqa] [FAIL] 归档校验失败：{exc}", file=sys.stderr, flush=True)
        return 2
    print(
        f"[n3d_triviaqa] 归档校验通过：SHA256={archive_sha}（{ARCHIVE_SIZE_BYTES} 字节，"
        f"{ARCHIVE_MEMBER_COUNT} 个成员）",
        flush=True,
    )

    try:
        raw_qa, passes_a, cache_status = load_archive_qa(
            args.archive,
            archive_sha,
            splits,
            use_cache=not bool(args.no_cache),
            refresh_cache=bool(args.refresh_cache),
        )
        for s in cache_status:
            print(f"[n3d_triviaqa]   QA 来源 {s}", flush=True)
        parsed: Dict[str, Tuple[List[QaRecord], Dict[str, Any]]] = {}
        for split in splits:
            parsed[split] = parse_qa_json(raw_qa[split], split)
            head = parsed[split][1]
            print(
                f"[n3d_triviaqa]   {split}: {head['questions']} 题"
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
                f"[n3d_triviaqa]   --max-questions={int(args.max_questions)}：每个 split 取前 "
                f"{int(args.max_questions)} 题（按 QuestionId 升序）",
                flush=True,
            )
        wanted: Set[str] = set()
        for s in splits:
            wanted |= {m for r in parsed[s][0] for m in r.doc_members}
        evidence, missing = load_archive_evidence(args.archive, wanted)
        n_wiki_hit = sum(1 for m in evidence if m.startswith(EVIDENCE_DIR["wikipedia"] + "/"))
        n_web_hit = sum(1 for m in evidence if m.startswith(EVIDENCE_DIR["web"] + "/"))
        print(
            f"[n3d_triviaqa]   evidence 命中 {len(evidence)}/{len(wanted)}"
            f"（wikipedia {n_wiki_hit} / web {n_web_hit}），未命中 {len(missing)}",
            flush=True,
        )
        if missing:
            if not bool(args.allow_missing_evidence):
                print(
                    f"[n3d_triviaqa] [FAIL] 归档中缺少 {len(missing)} 个 evidence 文档，"
                    f"前 10 个：{missing[:10]}（如确认可容忍，加 --allow-missing-evidence）",
                    file=sys.stderr,
                    flush=True,
                )
                return 2
            drop = set(missing)
            print(
                f"[n3d_triviaqa] [WARN] --allow-missing-evidence：丢弃 {len(drop)} 个缺失文档",
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
            X, y, samples, stats = build_split(
                split,
                records,
                doc_texts,
                negative_seed=int(args.negative_seed),
                negative_ratio=1.0,
                hash_dim=hash_dim,
                no_bag=no_bag,
            )
            out_name = out_name_for(split, int(args.max_questions), hash_dim, args.archive, no_bag)
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
            )
            save_npz_deterministic(out_path, X, y, meta)
            sha = sha256_file(out_path)
            print(
                f"[n3d_triviaqa] [OK] {split}: X={X.shape}（D={int(X.shape[1])}）{X.dtype} / "
                f"y={y.shape} {y.dtype} "
                f"正 {stats['positive']} 负 {stats['negative']} "
                f"（文档池 {stats['pool_documents']}，正样本含答案 {stats['positive_answer_hit']}/"
                f"{stats['positive']}，负样本含答案 {stats['negative_answer_hit']}/{stats['negative']}）",
                flush=True,
            )
            print(
                f"[n3d_triviaqa]      产物 {out_path}（SHA256={sha}，流式扫描 {passes} 遍）",
                flush=True,
            )
    except (BuildContractError, FileNotFoundError) as exc:
        print(f"[n3d_triviaqa] [FAIL] {exc}", file=sys.stderr, flush=True)
        return 2
    print(f"[n3d_triviaqa] 完成，用时 {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
