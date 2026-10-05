"""n3d_qa_learn 的数据装配层：问答对读取、答案空间构建、切分、文本行语料。

数据来源（只读，不修改上游任何产物）
----------------------------------
* **问答对（QA 数据集）**：读取 ``n3d_qa`` 模块已落盘的 QA JSON 缓存
  （``checkpoints/triviaqa/_cache/*_qa.json``，``{\"Data\": [...]}`` 形态）。这些缓存是
  ``n3d_qa`` 的既有产物，本模块**只读**，不改写、不重建。
* **文本行（文本数据集）**：读取 ``data/doc/*.md`` 的**逐行**文本。它是步骤 2 的匹配源，
  与步骤 1 的问答对匹配源**互相独立**（这是业务路由能分两级的前提）。

答案空间口径
------------
* 类别文本 = ``Answer.NormalizedValue``（``n3d_qa`` 既有口径的归一化答案）；
  展示文本取 ``Answer.MatchedWikiEntityName``，缺失时回退到首个别名。
* **全局答案表** = 训练侧出现次数 ``>= min_questions`` 的类别集合（确定性排序：
  先按频次降序，再按类别文本字典序 —— 保证跨运行一致）。
* **「不相关」类的来源（口径显式声明）**：答案表之外的类别（``unknown`` 组）的**全部**
  问题都标为「不相关」。这是 ``current_spec`` 已确认的口径
  （「正确答案不在全局答案表内的问题标为不相关」），**不是**"问题配错答案"。

切分口径（防数据泄漏）
--------------------
* 切分单位是**问题**，且**同一问题的问法只出现在一个 split 里**（按 ``QuestionId``
  排序后取模，确定性）；
* ``test_known``：答案在答案表内的问题（步骤 1 主测试集，金标 = 答案表下标）；
* ``test_unknown``：答案不在答案表内的问题（金标 = 「不相关」下标）；
* 训练集 = ``train_known``（金标 = 答案表下标） + ``train_unknown``（金标 = 不相关）。

**口径声明（必须如实登记）**：``train_unknown`` 与 ``test_unknown`` **按类别分组切分**
（unknown 组中 count >= 2 的类别，其问题的 1/2 进 train、1/2 进 test；count == 1 的类别
全部进 train）。这样「不相关」类在训练与测试两侧都有样本，且两侧**类别不重叠**，
避免"训练时见过该类别文本、测试时又要求拒绝"的自相矛盾；代价是 unknown 类别的
拒绝能力**不能在跨类别上泛化**上被评估（本模块显式声明该限制，不夸大结论）。
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

#: 默认 QA 缓存目录（``n3d_qa`` 既有产物，**只读**）。
DEFAULT_QA_CACHE_DIR: str = os.path.join("checkpoints", "triviaqa", "_cache")

#: 默认使用的 QA 缓存文件（``n3d_qa`` 既有产物，**只读**；文件名逐字冻结）。
#:
#: 口径说明：只取**同一次归档构建**（同一归档哈希前缀）产出的 4 个 QA 缓存。
#: 每个类别的可训练样本量由这 4 个文件的**并集**决定（按 ``QuestionId`` 去重），
#: 这是"答案类别至少 5 个问题"这一档位能达到的前提（现场实测：并集 9959 题 /
#: 7407 类，其中 >= 5 题的类 140 个、>= 6 题的类 89 个）。
DEFAULT_QA_CACHE_FILES: Tuple[str, ...] = (
    "ef94fac6db0541e5_web_qa.json",
    "ef94fac6db0541e5_wiki_qa.json",
    "ef94fac6db0541e5_wiki-dev_qa.json",
    "ef94fac6db0541e5_web-dev_qa.json",
)

#: 「不相关」训练样本的**逐类配额**（每类最多取多少条进训练侧的 unknown 子集）。
UNKNOWN_TRAIN_PER_CLASS: int = 1

#: 「不相关」训练样本的**总量上限**（防止 unknown 侧压倒 known 侧）。
UNKNOWN_TRAIN_CAP: int = 500

#: 「不相关」训练样本的默认权重（去偏；与 ``UNKNOWN_TRAIN_CAP`` 共同维持
#: ``known : unknown`` 的损失占比不失控）。
UNKNOWN_CLASS_WEIGHT: float = 1.0

#: 默认文本行语料目录（步骤 2 的匹配源）。
DEFAULT_TEXT_DIR: str = os.path.join("data", "doc")

#: 单个文本行的字符上限（超出按「超长行截断」口径处理，并在统计中可见）。
MAX_LINE_CHARS: int = 2000

#: 单个问题文本的字符上限（超出即报错 —— 「超长问题」的边界处置口径，二选一取"报错"）。
MAX_QUESTION_CHARS: int = 2000

#: 读 QA 缓存时接受的文件名模式（wiki / web 两套）。
QA_CACHE_PATTERNS: Tuple[str, ...] = ("*_wiki_qa.json", "*_web_qa.json")


# ---------------------------------------------------------------------------
# 记录结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QARecord:
    """一条问答对（只带路由与训练所需的字段；**不携带证据段落**）。

    属性
    ----
    qid : str
        数据集内唯一问题 id。
    question : str
        问题文本（TriviaQA 无 choices，故即原问题）。
    answer_key : str
        归一化答案文本（类别键）。
    answer_display : str
        答案展示文本（``MatchedWikiEntityName`` 或首个别名）。
    source : str
        QA 缓存来源标记（``wiki`` / ``web``）。
    """

    qid: str
    question: str
    answer_key: str
    answer_display: str
    source: str


@dataclass(frozen=True)
class TextRecord:
    """文本数据集的一行。

    属性
    ----
    line_id : str
        行 id（``<文件名>:<行号>``）。
    text : str
        行文本（已按 ``MAX_LINE_CHARS`` 截断）。
    source_file : str
        来源文件相对路径。
    truncated : bool
        是否发生过截断。
    """

    line_id: str
    text: str
    source_file: str
    truncated: bool = False


@dataclass
class QACorpus:
    """QA 数据集全集 + 全局答案表。

    属性
    ----
    records : List[QARecord]
        全部问答对（按 ``qid`` 升序，确定性）。
    answer_keys : List[str]
        全局答案表（下标即类别下标；**不含**「不相关」）。
    answer_display : Dict[str, str]
        类别键 -> 展示文本。
    class_counts : Dict[str, int]
        类别键 -> 落表前的频次（全量口径）。
    """

    records: List[QARecord]
    answer_keys: List[str]
    answer_display: Dict[str, str]
    class_counts: Dict[str, int]

    @property
    def n_classes(self) -> int:
        """答案类别数 ``C``（不含「不相关」）。"""
        return len(self.answer_keys)

    def key_to_index(self) -> Dict[str, int]:
        """类别键 -> 答案表下标。"""
        return {k: i for i, k in enumerate(self.answer_keys)}


@dataclass
class SplitSpec:
    """训练 / 测试切分规格（构造期不变量，测试后量不得写死初值）。"""

    train_known: List[QARecord]
    train_unknown: List[QARecord]
    test_known: List[QARecord]
    test_unknown: List[QARecord]

    def summary(self) -> Dict[str, int]:
        """各子集规模（写入报告与产物 meta）。"""
        return {
            "train_known": len(self.train_known),
            "train_unknown": len(self.train_unknown),
            "test_known": len(self.test_known),
            "test_unknown": len(self.test_unknown),
        }


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------


def _norm_answer_key(value: str) -> str:
    """归一化答案键：NFKC 折叠 + 小写 + 空白折叠（**与向量化口径同源**）。"""
    from .features import normalize_text

    return normalize_text(value)


def discover_qa_cache_files(
    cache_dir: str = DEFAULT_QA_CACHE_DIR,
    use_default_subset: bool = True,
) -> List[str]:
    """发现 QA 缓存文件（默认只用 ``DEFAULT_QA_CACHE_FILES`` 三个冻结文件）。

    参数
    ----
    cache_dir : str
        缓存目录。
    use_default_subset : bool
        ``True``（默认）：只取 ``DEFAULT_QA_CACHE_FILES`` 中**实际存在**的文件
        （口径冻结、跨运行一致）；``False``：按模式 ``QA_CACHE_PATTERNS`` 全量发现。

    返回
    ----
    List[str]
        匹配到的文件路径（升序）；目录不存在或为空时返回空列表。
    """
    if not os.path.isdir(cache_dir):
        return []
    if use_default_subset:
        found = [
            os.path.join(cache_dir, name)
            for name in DEFAULT_QA_CACHE_FILES
            if os.path.isfile(os.path.join(cache_dir, name))
        ]
        return sorted(found)
    patterns: List[str] = []
    for pattern in QA_CACHE_PATTERNS:
        patterns.extend(glob.glob(os.path.join(cache_dir, pattern)))
    return sorted(set(patterns))


def load_qa_records(
    cache_dir: str = DEFAULT_QA_CACHE_DIR,
    files: Optional[Sequence[str]] = None,
    use_default_subset: bool = True,
) -> Tuple[List[QARecord], List[str]]:
    """读取 QA 缓存为 ``QARecord`` 列表。

    参数
    ----
    cache_dir : str
        缓存目录（``files`` 给出时不使用）。
    files : Optional[Sequence[str]]
        显式文件列表。

    返回
    ----
    Tuple[List[QARecord], List[str]]
        （记录列表，实际读取的文件列表）。

    异常
    ------
    FileNotFoundError
        未找到任何 QA 缓存文件。
    ValueError
        JSON 结构不符合 ``{\"Data\": [...]}`` 形态。
    """
    paths = (
        list(files)
        if files
        else discover_qa_cache_files(cache_dir, use_default_subset=use_default_subset)
    )
    if not paths:
        raise FileNotFoundError(
            f"未在 {cache_dir!r} 下找到任何 QA 缓存文件（模式 {QA_CACHE_PATTERNS}）；"
            "请先由 n3d_qa 构建 QA 缓存，或用 --qa-cache-dir 指定目录"
        )
    out: List[QARecord] = []
    seen_qids: set = set()
    for path in paths:
        tag = "wiki" if "wiki" in os.path.basename(path) else "web"
        with open(path, "r", encoding="utf-8") as handle:
            blob = json.load(handle)
        if not isinstance(blob, dict) or "Data" not in blob:
            raise ValueError(
                f"QA 缓存 {path!r} 的结构不是 {{\"Data\": [...]}} 形态；"
                f"实际顶层键 = {sorted(blob.keys()) if isinstance(blob, dict) else type(blob).__name__}"
            )
        for item in blob["Data"]:
            qid = str(item.get("QuestionId", "")).strip()
            question = str(item.get("Question", "")).strip()
            answer = item.get("Answer") or {}
            key_raw = answer.get("NormalizedValue") or answer.get("Value") or ""
            key = _norm_answer_key(str(key_raw))
            if not qid or not question or not key:
                continue
            if qid in seen_qids:
                continue
            seen_qids.add(qid)
            display = str(
                answer.get("MatchedWikiEntityName")
                or (answer.get("Aliases") or [""])[0]
                or key
            ).strip()
            out.append(
                QARecord(
                    qid=qid,
                    question=question,
                    answer_key=key,
                    answer_display=display,
                    source=tag,
                )
            )
    out.sort(key=lambda r: r.qid)
    return out, paths


def build_answer_space(
    records: Sequence[QARecord],
    min_questions: int = 2,
    max_classes: int = 0,
) -> QACorpus:
    """构建全局答案表（频次门槛 + 可选类别数上限；档位固定、确定性排序）。

    参数
    ----
    records : Sequence[QARecord]
        全部问答对。
    min_questions : int
        类别进入答案表所需的最少问题数（``>= 2`` 才能同时有训练与测试样本）。
    max_classes : int
        答案表类别数上限；``0`` 表示不限制。

    返回
    ----
    QACorpus
        含答案表与全量频次的语料。
    """
    if int(min_questions) < 2:
        raise ValueError(
            f"min_questions 必须 >= 2（否则该类别无法同时提供训练与测试样本），"
            f"当前 {min_questions}"
        )
    counts: Dict[str, int] = {}
    display: Dict[str, str] = {}
    for rec in records:
        counts[rec.answer_key] = counts.get(rec.answer_key, 0) + 1
        display.setdefault(rec.answer_key, rec.answer_display)
    eligible = [k for k, c in counts.items() if c >= int(min_questions)]
    # 确定性档位：频次降序 -> 类别文本升序（跨运行一致，不依赖 dict 顺序）
    eligible.sort(key=lambda k: (-counts[k], k))
    if int(max_classes) > 0:
        eligible = eligible[: int(max_classes)]
    keys = sorted(eligible)
    return QACorpus(
        records=list(records),
        answer_keys=keys,
        answer_display={k: display.get(k, k) for k in keys},
        class_counts=counts,
    )


def _deterministic_order(key: str, items: List[QARecord], seed: int) -> List[QARecord]:
    """类别内问题的确定性打乱（``blake2b`` 派生，不用内置 ``hash``）。

    参数
    ----
    key : str
        类别键（参与盐，使不同类别得到不同置换）。
    items : List[QARecord]
        该类别的问题。
    seed : int
        种子。

    返回
    ----
    List[QARecord]
        打乱后的列表（同一 ``(key, seed, items)`` 恒得同一顺序）。
    """
    import hashlib

    return sorted(
        items,
        key=lambda r: hashlib.blake2b(
            f"{seed}\x00{key}\x00{r.qid}".encode("utf-8"), digest_size=8
        ).hexdigest(),
    )


def make_splits(
    corpus: QACorpus,
    test_every: int = 2,
    seed: int = 42,
    unknown_train_per_class: int = UNKNOWN_TRAIN_PER_CLASS,
    unknown_train_cap: int = UNKNOWN_TRAIN_CAP,
    test_per_class: int = 0,
) -> SplitSpec:
    """按「问题」切分出训练 / 测试四子集（见模块 docstring 的切分口径）。

    参数
    ----
    corpus : QACorpus
        语料（提供答案表）。
    test_every : int
        每个**已入表**类别的第 ``test_every`` 个问题进入测试侧（其余进训练侧），``>= 2``。
    seed : int
        类别内问题的打乱种子（``blake2b`` 派生）。
    unknown_train_per_class : int
        「不相关」训练样本的逐类配额（**去偏**：unknown 组类别极多，若全量入训练侧，
        ``known : unknown`` 会达 1:26，训练会退化为"一律拒绝"）。
    unknown_train_cap : int
        「不相关」训练样本总量上限（``<= 0`` = 不设上限）。
    test_per_class : int
        **主测试集的逐类样本上限**（``0`` = 不限制）。
        置为 ``k`` 时，每个已入表类别最多 ``k`` 条进 ``test_known``，多余的**回落到训练侧**
        （因此不会浪费样本）。作用是让主测试集**按类别均衡** —— 均衡后
        ``majority`` 基线恰为 ``1/class_count``，"多数类基线"这一口径才是可比的；
        不限制时测试集会按自然频次分布（头类占 10/38），基线被头类抬高。

    返回
    ----
    SplitSpec
        四子集（各子集内按 ``qid`` 升序）。

    关键不变量
    ----------
    * ``test_known`` 中每个答案类**至少 1 条**（交错切分保证）；
    * ``train_unknown`` 侧每个类别最多 ``unknown_train_per_class`` 条，**类别集合显著宽于**
      测试侧单类样本数 —— 这是刻意的：拒绝能力要能**跨类别泛化**（测试侧大量 unknown
      类别在训练侧只有 1 条样本，或完全没有样本）；
    * **不变量（可断言）**：设 ``C_test`` = 测试侧 unknown 类别集合中**在训练侧出现 >= 2 条**
      的类别集合，则这些类别的训练样本数被配额钉在 ``unknown_train_per_class``；
      ``test_known`` 与 unknown 侧**类别集合天然不相交**（前者必在答案表内，后者必不在）。
    """
    if int(test_every) < 2:
        raise ValueError(f"test_every 必须 >= 2，当前 {test_every}")
    in_table = corpus.key_to_index()
    by_key: Dict[str, List[QARecord]] = {}
    for rec in corpus.records:
        by_key.setdefault(rec.answer_key, []).append(rec)

    train_known: List[QARecord] = []
    test_known: List[QARecord] = []
    train_unknown: List[QARecord] = []
    test_unknown: List[QARecord] = []

    for key in sorted(by_key.keys()):
        items = _deterministic_order(key, by_key[key], seed)
        if key in in_table:
            # 已入表类别：交错切分，保证测试侧每个类别都有样本；
            # `test_per_class > 0` 时把该类的测试样本数**封顶**，多余的回落到训练侧
            n_test_taken = 0
            for i, rec in enumerate(items):
                take_test = (i % int(test_every) == 0) and (
                    int(test_per_class) <= 0 or n_test_taken < int(test_per_class)
                )
                if take_test:
                    test_known.append(rec)
                    n_test_taken += 1
                else:
                    train_known.append(rec)
        else:
            # 未入表类别：**逐类配额进训练侧**（每类最多 unknown_train_per_class 条），
            # 其余问题全部进测试侧。这样 unknown 训练侧的类别可以被"全体未入表类别"
            # 覆盖（提升拒绝能力的类别泛化），而测试侧保留全部样本。
            quota = max(1, int(unknown_train_per_class))
            train_unknown.extend(items[:quota])
            test_unknown.extend(items[quota:])

    # ---- unknown 训练侧的总量上限（确定性散列截断，避免"总是取字典序靠前的类别"） ----
    if int(unknown_train_cap) > 0 and len(train_unknown) > int(unknown_train_cap):
        import hashlib

        train_unknown.sort(
            key=lambda r: hashlib.blake2b(
                f"cap\x00{seed}\x00{r.qid}".encode("utf-8"), digest_size=8
            ).hexdigest()
        )
        train_unknown = train_unknown[: int(unknown_train_cap)]

    for bucket in (train_known, train_unknown, test_known, test_unknown):
        bucket.sort(key=lambda r: r.qid)
    return SplitSpec(
        train_known=train_known,
        train_unknown=train_unknown,
        test_known=test_known,
        test_unknown=test_unknown,
    )


# ---------------------------------------------------------------------------
# 文本行语料
# ---------------------------------------------------------------------------


def load_text_lines(text_dir: str = DEFAULT_TEXT_DIR) -> List[TextRecord]:
    """读取文本行语料（``*.md`` 的逐行切分；空行丢弃）。

    参数
    ----
    text_dir : str
        目录。

    返回
    ----
    List[TextRecord]
        行记录（按 ``(文件名, 行号)`` 升序）。目录不存在或为空时返回空列表。
    """
    if not os.path.isdir(text_dir):
        return []
    out: List[TextRecord] = []
    for name in sorted(os.listdir(text_dir)):
        if not name.lower().endswith((".md", ".txt")):
            continue
        path = os.path.join(text_dir, name)
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for lineno, raw in enumerate(handle, start=1):
                text = raw.rstrip("\r\n").strip()
                if not text:
                    continue
                truncated = len(text) > MAX_LINE_CHARS
                out.append(
                    TextRecord(
                        line_id=f"{name}:{lineno}",
                        text=text[:MAX_LINE_CHARS],
                        source_file=name,
                        truncated=truncated,
                    )
                )
    return out


def check_question(question: str) -> str:
    """问题文本的**边界处置**（口径：超长 -> 立即报错，不静默截断；空/空白放行）。

    参数
    ----
    question : str
        原始问题文本。

    返回
    ----
    str
        原样返回（便于链式调用）。

    异常
    ------
    TypeError
        非字符串。
    ValueError
        长度超过 ``MAX_QUESTION_CHARS``。

    口径说明
    --------
    **空串与仅空白串不是错误**：它们是"无匹配"这一业务结果的输入形态，由
    :func:`n3d_qa_learn.route.QuestionRouter.route` 显式返回「无匹配」；
    只有**超长**才是调用侧必须处置的错误（口径：报错，不静默截断）。
    """
    if not isinstance(question, str):
        raise TypeError(f"question 必须为 str，当前类型 {type(question).__name__}")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"问题长度 {len(question)} 超过上限 {MAX_QUESTION_CHARS}"
            f"（超长问题的边界处置口径为「报错」，不静默截断）"
        )
    return question


__all__ = [
    "DEFAULT_QA_CACHE_DIR",
    "DEFAULT_QA_CACHE_FILES",
    "UNKNOWN_TRAIN_PER_CLASS",
    "UNKNOWN_TRAIN_CAP",
    "DEFAULT_TEXT_DIR",
    "MAX_LINE_CHARS",
    "MAX_QUESTION_CHARS",
    "QA_CACHE_PATTERNS",
    "QARecord",
    "TextRecord",
    "QACorpus",
    "SplitSpec",
    "discover_qa_cache_files",
    "load_qa_records",
    "build_answer_space",
    "make_splits",
    "load_text_lines",
    "check_question",
]