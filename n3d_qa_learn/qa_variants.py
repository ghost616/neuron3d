"""轮 3 · B3：**QA 侧变体生成**（复用第三步的 4 支变体生成器，绑定输出 = 答案）。

与文本侧改造的本质差异（**必须显式**）
--------------------------------------
* **文本侧**（第三步 ``text_variants``）：查询 = 某一行的变体表述，**绑定输出 = 该行原文**
  （「多表述 -> 原文」）。
* **QA 侧**（本模块）：查询 = 某一**问题**的变体表述，**绑定输出 = 该问题绑定的答案展示文本**
  （「多表述问题 -> 答案」）。
⇒ **绑定输出绝不是问题原文**（:data:`BINDING_OUTPUT_RULE`）。这是两条改造路线唯一的、
也是最关键的语义差异：若把问题原文当输出，QA 侧就退化成「问题自匹配」，与「问题 -> 答案」
的业务目标无关。

严格整组切分（B3-3）
--------------------
切分单位是 ``question_id``，且**同一问题的全部变体必须同侧**。本模块用与轮 1
**完全相同**的哈希比例抽稀（:func:`n3d_qa_learn.qa_scale.question_hash`）在
``question_id`` 上定侧，因此「整组同侧」是**构造性**成立；再由
:func:`verify_group_split` 逐项复核三条不变量（行不交且并集全、问题集不交且并集全、
每问题整组同侧），并对**人为破坏**的切分给出反例（证明断言非恒真）。

产物纪律
--------
产物一律写 ``checkpoints/qa_learn/_verify/qa_variants/``（**不落盘裸特征矩阵、不产 zip、
无挂钟字段**）；**不改动**文本行侧既有产出路径与其 12 件产物。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .qa_scale import (
    DEFAULT_HOLDOUT_PCT,
    MATCH_TASKS,
    SOLVE_TASKS,
    FULL_CLASSES,
    ScaleCorpus,
    build_scale_corpus,
    make_scale_split,
    question_hash,
    question_side,
)

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

#: 本轮协议版本（结构变化必须递增；进产物）。
QA_VARIANTS_VERSION: str = "n3dqa-qa-variants-v1"

#: **绑定输出规则**（B3-2 的核心声明，唯一来源）。
BINDING_OUTPUT_RULE: str = (
    "QA 侧变体的**绑定输出 = 该问题绑定的答案展示文本**（`answer_display`），"
    "**不是**问题原文。这是与文本侧改造（多表述 -> 原文）的本质差异："
    "文本侧的绑定输出是**行原文**，QA 侧的绑定输出是**答案**。"
    "若把问题原文当输出，QA 侧就退化成「问题自匹配」，与「问题 -> 答案」的业务目标无关。"
)

#: **整组切分规则**（B3-3 的核心声明，唯一来源）。
GROUP_SPLIT_RULE: str = (
    "切分单位 = `question_id`；**同一问题的全部变体必须落在同一侧**（训练或测试），"
    "禁止跨侧泄漏。定侧用与轮 1 完全相同的哈希比例抽稀："
    "h = int(sha256('qgroup\\0seed\\0task\\0question_id'), 8B) % 100；h < holdout_pct 进测试侧。"
    "因为定侧**只看 question_id**、不看变体，故「整组同侧」是**构造性**成立；"
    "本模块仍用 `verify_group_split` 逐项复核三条不变量，并给出「人为破坏切分」的反例。"
)

#: 三条切分不变量的**现场枚举**（唯一来源；报告与产物都用它）。
GROUP_SPLIT_INVARIANTS: Tuple[str, str, str] = (
    "rows_disjoint_and_union_complete",
    "questions_disjoint_and_union_complete",
    "every_question_wholly_on_one_side",
)

#: 变体是否「信息减少」的声明（不得把抽取式摘要说成 paraphrase）。
EXTRACTIVE_NOTE: str = (
    "本模块的 4 支变体全部是**抽取式 / 词面级**变换（原文 / 关键词 / 摘要 / 自适应），"
    "**不是**语义 paraphrase。对某些问题，变体相对原文是**信息减少**而非「换种说法」——"
    "现场逐支统计「与原文逐字节相同」的比例并如实登记，不得隐藏。"
)

#: QA 侧语料级候选词表的**文档频次下限**（**与文本侧不同，必须显式声明**）。
#:
#: 文本侧（第三步）用 ``text_variants.KEYWORD_MIN_DOC_FREQ = 2``，其语料是 2665 行、
#: 唯一候选词 ~10464 个。QA 侧语料是 **12727 个问题**、唯一候选词 **163690** 个，
#: 取 ``min_doc_freq = 2`` 时保留词表 **49988** 个 —— 每个词都要走一次 bge-m3 前向，
#: 现场外推约需 **8~9 小时**（实测：50 题 206.5 s、200 题 505.3 s，即约 2.5 s/题），
#: 在单轮执行预算内不可完成。
#:
#: 故 QA 侧**显式提高**到 ``20``（现场保留词表 **2849** 个），并把这个偏离**写进产物**
#: （``generator.vocab_min_doc_freq`` / ``vocab_caliber_deviation``）。这是**口径偏离**，
#: 不是「同一个口径」——影响是关键词支更保守（只取全 QA 语料里出现于 >= 20 个问题的词）。
#: **它不改变**：变体生成器实现、关键词 top-k、摘要上限、自适应分档、以及「绑定输出 = 答案」。
QA_VOCAB_MIN_DOC_FREQ: int = 20

#: 该偏离的**口径声明**（进产物；四处同源的唯一文本）。
VOCAB_CALIBER_DEVIATION: str = (
    "QA 侧语料级候选词表用 min_doc_freq = 20（文本侧第三步为 2）。"
    "理由：QA 语料 12727 个问题的唯一候选词 163690 个，min_doc_freq=2 保留 49988 个词，"
    "每词需一次 bge-m3 前向，现场外推 8~9 小时（实测 50 题 206.5 s / 200 题 505.3 s）；"
    "min_doc_freq=20 保留 2849 个词，在单轮预算内可全量覆盖 12727 个问题。"
    "**这是显式口径偏离**：关键词支会更保守（只取全 QA 语料出现于 >= 20 个问题的词）；"
    "不改变生成器实现、top-k、摘要上限、自适应分档与「绑定输出 = 答案」。"
)

#: 产物目录与守卫根。
VERIFY_ROOT: str = os.path.join("checkpoints", "qa_learn", "_verify")
DEFAULT_OUT_DIR: str = os.path.join(VERIFY_ROOT, "qa_variants")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _sha256_bytes(blob: bytes) -> str:
    """字节 SHA256。"""
    return hashlib.sha256(blob).hexdigest()


def sha256_text(text: str) -> str:
    """UTF-8 裸字节 SHA256。"""
    return _sha256_bytes(str(text).encode("utf-8"))


def sha256_file(path: str) -> str:
    """文件 SHA256（分块读）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """规范化 JSON（``sort_keys`` + 紧凑分隔符）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# QA 侧查询单元（问题 + 绑定输出）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QAQuestionUnit:
    """一个 QA 侧查询单元（**问题**，不是行）。

    参数
    ----
    question_id : str
        问题 id（切分单位）。
    task : str
        任务名（manifest ``tasks`` 之一）。
    question_text : str
        问题原文（变体生成的**输入**）。
    answer_display : str
        该问题绑定的**答案展示文本**（变体的**绑定输出**；B3-2）。
    gold_class_indices : List[int]
        该问题的正确类下标（全局，升序）。
    row_indices : List[int]
        该问题在语料里的行下标（升序）。
    """

    question_id: str
    task: str
    question_text: str
    answer_display: str
    gold_class_indices: List[int]
    row_indices: List[int]


def load_question_texts(product_dir: str = "", *, limit: int = 0) -> Dict[str, str]:
    """读取 ``question_id -> 问题原文``（只读冻结产物）。

    为什么需要它：``n3d_qa_all_pairs.jsonl`` 里有 ``question_text``，但它**按行重复**
    （同一问题多行）。本函数按 ``question_id``去重，并**校验**同一问题在不同行上的
    ``question_text`` 完全一致（不一致即抛 —— 否则「一个问题一个变体」的前提不成立）。
    """
    from .qa_scale import ALL_PAIRS_FILE, DEFAULT_PRODUCT_DIR, load_all_pairs

    root = str(product_dir) if product_dir else str(DEFAULT_PRODUCT_DIR)
    path = os.path.join(root, ALL_PAIRS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少问答对文件：{path!r}")
    out: Dict[str, str] = {}
    conflicts: List[Dict[str, str]] = []
    seen = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj["question_id"])
            text = str(obj["question_text"])
            prev = out.get(qid)
            if prev is None:
                out[qid] = text
                seen += 1
            elif prev != text:
                if len(conflicts) < 10:
                    conflicts.append({"question_id": qid})
            if int(limit) > 0 and seen >= int(limit):
                break
    if conflicts:
        raise ValueError(
            f"有问题的 `question_text` 在不同行上不一致（前几个 = {conflicts}）；"
            "「一个问题一个变体」的前提不成立，拒绝继续"
        )
    return out


def build_question_units(
    corpus: ScaleCorpus,
    *,
    question_texts: Dict[str, str],
    limit_questions: int = 0,
) -> List[QAQuestionUnit]:
    """由扩容语料构建 QA 侧查询单元（每个 ``question_id`` 一个单元）。

    绑定输出取 ``answer_display``：用该问题的**第一个**正确类下标去查
    ``corpus.answer_display``（多金标问题时按类下标升序取第一个，并在单元里保留
    完整 ``gold_class_indices`` 供评测使用）。
    """
    units: List[QAQuestionUnit] = []
    qids = sorted(corpus.question_gold.keys())
    if int(limit_questions) > 0:
        qids = qids[: int(limit_questions)]
    missing: List[str] = []
    for qid in qids:
        gold = [int(x) for x in corpus.question_gold[str(qid)]]
        if not gold:
            raise ValueError(f"问题 {qid!r} 没有正确类；拒绝为无金标问题生成变体")
        text = question_texts.get(str(qid))
        if text is None:
            missing.append(str(qid))
            continue
        first = int(gold[0])
        pos = corpus.class_pos.get(first)
        if pos is None:
            raise ValueError(
                f"问题 {qid!r} 的正确类下标 {first} 不在本轮口径的保留类内；"
                "这会让「绑定输出」失去定义，拒绝继续"
            )
        units.append(
            QAQuestionUnit(
                question_id=str(qid),
                task=str(corpus.question_task[str(qid)]),
                question_text=str(text),
                answer_display=str(corpus.answer_display[int(pos)]),
                gold_class_indices=gold,
                row_indices=[int(x) for x in corpus.question_rows[str(qid)]],
            )
        )
    if missing:
        raise ValueError(
            f"有 {len(missing)} 个问题在冻结产物里找不到 `question_text`"
            f"（前几个 = {missing[:5]}）；拒绝在缺输入的问题上生成变体"
        )
    return units


# ---------------------------------------------------------------------------
# 变体生成（复用 text_variants 的 4 支生成器）
# ---------------------------------------------------------------------------


def generate_qa_variants(
    units: Sequence[QAQuestionUnit],
    generator: Any,
) -> List[Dict[str, str]]:
    """对每个问题生成 4 支变体（**只换输入语料，不换生成口径**）。

    参数
    ----
    units : Sequence[QAQuestionUnit]
        查询单元（按 ``question_id`` 升序）。
    generator : Any
        :class:`n3d_qa_learn.text_variants.VariantGenerator` 实例
        （**复用**：生成口径、冻结常量、回退规则全部沿用，不另写一套）。

    返回
    ----
    List[Dict[str, str]]
        与 ``units`` 同序；每项是 ``variant_name -> 变体文本``。
    """
    texts = [str(u.question_text) for u in units]
    out = generator.variants_batch(texts)
    if len(out) != len(texts):
        raise ValueError(
            f"生成器返回 {len(out)} 条变体，与输入问题数 {len(texts)} 不一致"
        )
    return out


def variant_records(
    units: Sequence[QAQuestionUnit],
    variants: Sequence[Dict[str, str]],
    *,
    variant_order: Sequence[str],
    holdout_pct: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """把「单元 + 变体」展开为**逐条变体记录**（一行 = 一个问题的某一支变体）。

    每条记录含：
    ``row_key``（``question_id|variant`` 的唯一键）、``question_id`` / ``task`` /
    ``variant`` / ``variant_text`` / ``bounding_output``（**答案展示文本**）/ ``side`` /
    ``text_sha256`` / ``output_sha256``。

    ``side`` 由 :func:`question_side` 在 ``question_id`` 上定 —— **同问题的 4 条记录必然同侧**
    （这是整组切分的直接后果，由 :func:`verify_group_split` 复核）。
    """
    order = [str(v) for v in variant_order]
    out: List[Dict[str, Any]] = []
    for unit, item in zip(units, variants):
        side = question_side(
            str(unit.question_id), str(unit.task), int(seed), int(holdout_pct)
        )
        for name in order:
            text = str(item[str(name)])
            out.append(
                {
                    "row_key": str(unit.question_id) + "|" + str(name),
                    "question_id": str(unit.question_id),
                    "task": str(unit.task),
                    "variant": str(name),
                    "variant_text": text,
                    "bounding_output": str(unit.answer_display),
                    "side": str(side),
                    "text_sha256": sha256_text(text),
                    "output_sha256": sha256_text(str(unit.answer_display)),
                    "is_anchor": bool(str(name) == str(order[0])),
                }
            )
    return out


# ---------------------------------------------------------------------------
# 整组切分的三条不变量 + 反例
# ---------------------------------------------------------------------------


def verify_group_split(
    records: Sequence[Dict[str, Any]],
    *,
    expected_row_keys: Optional[Sequence[str]] = None,
    expected_question_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """复核「按 ``question_id`` 整组切分」的三条不变量。

    三条不变量（:data:`GROUP_SPLIT_INVARIANTS`）
    -------------------------------------------
    1. ``rows_disjoint_and_union_complete``：训练侧与测试侧的行键集合**不交**，
       且并集**等于**权威全集（一条不多、一条不少）；
    2. ``questions_disjoint_and_union_complete``：两侧的问题集合**不交**，
       且并集等于权威问题全集；
    3. ``every_question_wholly_on_one_side``：每个问题的全部记录
       **只出现在一侧**（这是「整组切分」的定义）。

    **权威全集（``expected_row_keys`` / ``expected_question_ids``）不是可选项装饰**
    ------------------------------------------------------------------------------
    若把「全集」也从**被检查的这批记录**里现算，那「并集完整」就退化成
    「集合等于它自己」——**恒真**，删掉一行也查不出来（现场实测过：反例
    ``drop_one_row`` 不翻面）。故本函数要求调用方传入**权威全集**
    （真实运行时 = 语料侧现场枚举的全集；反例取证时 = **破坏前**的全集）。
    未传时退回「自比」并**显式标注** ``authority = self_referential``
    （该模式**不能**发现丢行，产物里必须可见）。

    返回
    ----
    Dict[str, Any]
        ``invariants``（三项布尔 + 计数）、``all_hold``、``authority``、
        ``violations_head``、``rule``。
    """
    rows = [str(r["row_key"]) for r in records]
    train_rows = [str(r["row_key"]) for r in records if str(r["side"]) == "train"]
    test_rows = [str(r["row_key"]) for r in records if str(r["side"]) == "test"]
    side_of: Dict[str, set] = {}
    for r in records:
        side_of.setdefault(str(r["question_id"]), set()).add(str(r["side"]))
    train_q = sorted(q for q, s in side_of.items() if s == {"train"})
    test_q = sorted(q for q, s in side_of.items() if s == {"test"})
    mixed_q = sorted(q for q, s in side_of.items() if len(s) > 1)
    unknown_side = sorted(
        str(r["row_key"]) for r in records if str(r["side"]) not in ("train", "test")
    )
    train_set = set(train_rows)
    test_set = set(test_rows)
    inter = sorted(train_set & test_set)
    union = train_set | test_set
    # **权威全集**：显式传入优先；否则退回自比并标注（自比无法发现丢行）
    if expected_row_keys is not None:
        all_rows = {str(x) for x in expected_row_keys}
        authority = "explicit"
    else:
        all_rows = set(rows)
        authority = "self_referential"
    if expected_question_ids is not None:
        all_q = {str(x) for x in expected_question_ids}
        q_authority = "explicit"
    else:
        all_q = set(side_of.keys())
        q_authority = "self_referential"
    inv = {
        "rows_disjoint_and_union_complete": {
            "passed": bool(len(inter) == 0 and union == all_rows),
            "n_train_rows": int(len(train_rows)),
            "n_test_rows": int(len(test_rows)),
            "n_rows_total": int(len(rows)),
            "n_authority_rows": int(len(all_rows)),
            "n_intersection": int(len(inter)),
            "n_union": int(len(union)),
            "intersection_head": inter[:10],
            "union_complete": bool(union == all_rows),
            "missing_from_union_head": sorted(all_rows - union)[:10],
            "extra_in_union_head": sorted(union - all_rows)[:10],
            "unknown_side_head": unknown_side[:10],
            "authority": str(authority),
        },
        "questions_disjoint_and_union_complete": {
            "passed": bool(
                len(set(train_q) & set(test_q)) == 0
                and set(train_q) | set(test_q) == all_q
            ),
            "n_train_questions": int(len(train_q)),
            "n_test_questions": int(len(test_q)),
            "n_questions_total": int(len(side_of)),
            "n_authority_questions": int(len(all_q)),
            "n_intersection": int(len(set(train_q) & set(test_q))),
            "union_complete": bool(set(train_q) | set(test_q) == all_q),
            "missing_from_union_head": sorted(all_q - (set(train_q) | set(test_q)))[:10],
            "authority": str(q_authority),
        },
        "every_question_wholly_on_one_side": {
            "passed": bool(len(mixed_q) == 0),
            "n_mixed_questions": int(len(mixed_q)),
            "mixed_questions_head": mixed_q[:10],
        },
    }
    return {
        "invariants": inv,
        "invariant_order": list(GROUP_SPLIT_INVARIANTS),
        "all_hold": bool(all(bool(v["passed"]) for v in inv.values())),
        "authority": {"rows": str(authority), "questions": str(q_authority)},
        "violations_head": {k: v for k, v in inv.items() if not bool(v["passed"])},
        "rule": str(GROUP_SPLIT_RULE),
    }


def group_split_counter_examples(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """**反例取证**：人为破坏切分后，三条不变量必须**真的失败**（证明非恒真）。

    构造三个破坏（都只在**内存副本**上做，**不落盘、不改真实产物**）：

    * ``flip_one_variant_half_question``：把某问题的**一支**变体翻到另一侧
      ⇒ 第 3 条（整组同侧）必须由 True 变 False；
    * ``duplicate_one_row``：把一行复制到另一侧（行键相同、侧不同）
      ⇒ 第 1 条（行不交且并集全）必须由 True 变 False；
    * ``drop_one_row``：删掉一行 ⇒ 第 1 条的「并集完整」必须由 True 变 False。

    返回
    ----
    Dict[str, Any]
        ``baseline_all_hold``（真实记录的三条是否全成立）与 ``cases``（每个反例的
        ``broken_invariant`` / ``before_passed`` / ``after_passed`` /
        ``assertion_fires``）。

    关键实现细节（**这条反例暴露过的真缺陷**）
    ------------------------------------------
    「并集完整」这条不变量**必须对照权威全集**（破坏**之前**的全集），而不能从被检查的
    这批记录里现算 —— 否则「删掉一行」时全集也少一行，两者仍相等 ⇒ 断言**恒真**、反例
    不翻面（现场实测：``drop_one_row`` 未触发）。故这里的每个反例都显式把
    ``expected_row_keys`` / ``expected_question_ids`` 传成**原始**全集。
    """
    all_row_keys = [str(r["row_key"]) for r in records]
    all_question_ids = sorted({str(r["question_id"]) for r in records})
    baseline = verify_group_split(
        records,
        expected_row_keys=all_row_keys,
        expected_question_ids=all_question_ids,
    )
    cases: List[Dict[str, Any]] = []

    def _check(mutated: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """**始终**对照原始全集（不是 mutated 自己的全集）。"""
        return verify_group_split(
            mutated,
            expected_row_keys=all_row_keys,
            expected_question_ids=all_question_ids,
        )

    def _record(
        name: str,
        invariant: str,
        before: bool,
        after: bool,
        detail: Dict[str, Any],
    ) -> None:
        cases.append(
            {
                "case": str(name),
                "broken_invariant": str(invariant),
                "before_passed": bool(before),
                "after_passed": bool(after),
                "assertion_fires": bool(before and not after),
                "detail": dict(detail),
            }
        )

    # --- 反例 1：把某问题的一支变体翻到另一侧 -------------------------------
    if records:
        first_q = str(records[0]["question_id"])
        q_records = [r for r in records if str(r["question_id"]) == first_q]
        if len(q_records) >= 2:
            victim_key = str(q_records[0]["row_key"])
            victim_side = str(q_records[0]["side"])
            other = "test" if victim_side == "train" else "train"
            mutated = [
                (dict(r) if str(r["row_key"]) != victim_key else {**r, "side": other})
                for r in records
            ]
            after = _check(mutated)
            _record(
                "flip_one_variant_half_question",
                "every_question_wholly_on_one_side",
                bool(
                    baseline["invariants"]["every_question_wholly_on_one_side"]["passed"]
                ),
                bool(
                    after["invariants"]["every_question_wholly_on_one_side"]["passed"]
                ),
                {
                    "question_id": first_q,
                    "flipped_row_key": victim_key,
                    "from_side": victim_side,
                    "to_side": other,
                    "n_variants_of_question": int(len(q_records)),
                },
            )
    # --- 反例 2：把一行复制到另一侧（行键重复、侧不同） ---------------------
    if records:
        victim = records[0]
        other = "test" if str(victim["side"]) == "train" else "train"
        mutated = list(records) + [{**victim, "side": other}]
        after = _check(mutated)
        _record(
            "duplicate_one_row",
            "rows_disjoint_and_union_complete",
            bool(baseline["invariants"]["rows_disjoint_and_union_complete"]["passed"]),
            bool(after["invariants"]["rows_disjoint_and_union_complete"]["passed"]),
            {"row_key": str(victim["row_key"]), "duplicated_to_side": other},
        )
    # --- 反例 3：删掉一行（并集不完整） -------------------------------------
    if records:
        mutated = list(records[:-1])
        after = _check(mutated)
        _record(
            "drop_one_row",
            "rows_disjoint_and_union_complete",
            bool(baseline["invariants"]["rows_disjoint_and_union_complete"]["passed"]),
            bool(after["invariants"]["rows_disjoint_and_union_complete"]["passed"]),
            {
                "dropped_row_key": str(records[-1]["row_key"]),
                "authority_rule": "对照**原始**全集（不是 mutated 自己的全集）",
            },
        )
    # --- 反例 4：删掉某问题的一整组（问题并集不完整） -----------------------
    if records:
        victim_q = str(records[0]["question_id"])
        mutated = [r for r in records if str(r["question_id"]) != victim_q]
        after = _check(mutated)
        _record(
            "drop_one_question_group",
            "questions_disjoint_and_union_complete",
            bool(
                baseline["invariants"]["questions_disjoint_and_union_complete"]["passed"]
            ),
            bool(
                after["invariants"]["questions_disjoint_and_union_complete"]["passed"]
            ),
            {"dropped_question_id": victim_q},
        )
    return {
        "baseline_all_hold": bool(baseline["all_hold"]),
        "baseline_authority": dict(baseline["authority"]),
        "cases": cases,
        "all_counter_examples_fire": bool(
            all(bool(c["assertion_fires"]) for c in cases)
        ),
        "rule": (
            "每个反例都必须在**内存副本**上把对应不变量由 True 翻成 False；"
            "任一反例未能翻面即说明该不变量是**恒真断言**，必须重写。"
            "**每个反例都对照「破坏前的原始全集」**（见 `_check`），"
            "否则「删一行」这类反例会被自比的全集掩盖。真实产物**不**被这些反例改动。"
        ),
    }


# ---------------------------------------------------------------------------
# 特征差异取证（B3-4）
# ---------------------------------------------------------------------------


def feature_difference_report(
    anchor_texts: Sequence[str],
    variant_texts: Sequence[str],
    *,
    vectorizer: Any,
    variant_name: str,
    anchor_name: str = "",
    fallback_count: int = 0,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """逐支给出与**原文支**的特征差异统计（B3-4）。

    口径（**可量化、可复算**）
    --------------------------
    * ``n``：本支记录数；
    * ``n_text_identical``：与原文支**逐字节相同**的条数（**如实报告，不得隐藏**）；
    * ``n_text_different``：文本不同的条数；
    * ``n_feature_bitwise_identical``：与原文支的 float32 **裸字节**逐字节相同的条数
      （文本不同但特征相同 = 「只改文本标签、特征没变」，必须被这个计数抓住）；
    * ``cosine``：逐条余弦（原文支 vs 本支）的 min / mean / max / 分位；
    * ``n_cosine_below_one``：余弦 < 1 的条数；
    * ``max_abs_diff``：逐条最大绝对差的最大值；
    * ``verdict``：``feature_really_changes``（存在 at least one 条余弦 < 1 或有裸字节差异）
      与否 —— **不得**只看文本不同就断言特征变了。

    两条输入必须是**同序、等长**的。
    """
    a = [str(t) for t in anchor_texts]
    b = [str(t) for t in variant_texts]
    if len(a) != len(b):
        raise ValueError(f"原文支 {len(a)} 条与变体支 {len(b)} 条长度不一致")
    from .entry_table import encode_statements

    fa = encode_statements(vectorizer, a, batch_size=int(batch_size))
    fb = encode_statements(vectorizer, b, batch_size=int(batch_size))
    n = int(len(a))
    same_text = 0
    same_bytes = 0
    cos: List[float] = []
    maxd: List[float] = []
    diff_text_head: List[Dict[str, Any]] = []
    for i in range(n):
        if a[i] == b[i]:
            same_text += 1
        ba = np.ascontiguousarray(fa[i], dtype=np.float32).tobytes()
        bb = np.ascontiguousarray(fb[i], dtype=np.float32).tobytes()
        if ba == bb:
            same_bytes += 1
        va = fa[i].astype(np.float64)
        vb = fb[i].astype(np.float64)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        c = float(np.dot(va, vb) / (na * nb)) if (na > 0 and nb > 0) else 0.0
        cos.append(c)
        maxd.append(float(np.max(np.abs(va - vb))) if va.size else 0.0)
        if a[i] != b[i] and len(diff_text_head) < 5:
            diff_text_head.append(
                {
                    "position": int(i),
                    "anchor_sha256": sha256_text(a[i]),
                    "variant_sha256": sha256_text(b[i]),
                    "cosine": float(c),
                    "anchor_chars": int(len(a[i])),
                    "variant_chars": int(len(b[i])),
                }
            )
    arr = np.asarray(cos, dtype=np.float64) if cos else np.zeros(1, dtype=np.float64)
    below_one = int(np.sum(arr < 1.0 - 1e-12))
    n_diff_text = int(n - same_text)
    return {
        "variant": str(variant_name),
        "n": int(n),
        "n_text_identical": int(same_text),
        "n_text_different": int(n_diff_text),
        "text_identical_rate": float(same_text / max(1, n)),
        "n_feature_bitwise_identical": int(same_bytes),
        "feature_bitwise_identical_rate": float(same_bytes / max(1, n)),
        "cosine": {
            "min": float(arr.min()) if cos else 0.0,
            "mean": float(arr.mean()) if cos else 0.0,
            "max": float(arr.max()) if cos else 0.0,
            "p05": float(np.quantile(arr, 0.05)) if cos else 0.0,
            "p50": float(np.quantile(arr, 0.50)) if cos else 0.0,
        },
        "n_cosine_below_one": int(below_one),
        "max_abs_diff": float(max(maxd)) if maxd else 0.0,
        "diff_text_head": diff_text_head,
        "feature_really_changes": bool(below_one > 0 or int(n - same_bytes) > 0),
        "verdict_rule": (
            "`feature_really_changes` = 「存在至少一条余弦 < 1」或「存在至少一条裸字节不同」。"
            "**只看文本不同不足以断言特征变了** —— 必须由本口径判定；"
            "`n_feature_bitwise_identical` 专门抓「文本变了但特征没变」的情形。"
        ),
        # **E-B3a 修复（离朱 R82）**：显式的「可评估性」块。原文支与原文自带
        # **恒等**，其与自身的差异恒为 0/余弦恒为 1 ⇒ **没有可评估的差异**
        # （`n_evaluable = 0`）；非锚点支的可评估条数 = 与原文**不同**的条数。
        # 这个块让「哪几支真的有差异可比」在产物里**可机读**，不必靠读余弦分布去猜。
        "evaluability": {
            "is_anchor": bool(str(variant_name) == str(anchor_name)),
            "n": int(n),
            "n_evaluable": int(0 if str(variant_name) == str(anchor_name) else n_diff_text),
            "n_not_evaluable": int(
                n if str(variant_name) == str(anchor_name) else same_text
            ),
            # **E-B3b 修复（离朱 R83）**：`n_evaluable_expected` 必须来自**独立来源**
            # （生成器自报的 `fallback_to_original` 计数），**不能**写成 `n` 或
            # `n - n_text_identical`：
            #   * 写成 `n` ⇒ 只要有一条回退到原文就会 `consistent=False`（现场摘要
            #     396 vs 400 即此错，且与汇总 `all_non_anchor_consistent=True` 自相矛盾）；
            #   * 写成 `n - n_text_identical` ⇒ 与 `n_evaluable` 的定义式恒等 ⇒ **恒真**。
            # 正确口径：`expected = n - fallback_to_original`（生成器侧计数）对
            # `n_evaluable = n - n_text_identical`（文本比对侧）—— 两个**不同来源**才可互证。
            "n_evaluable_expected": int(
                0
                if str(variant_name) == str(anchor_name)
                else max(0, int(n) - int(fallback_count))
            ),
            "n_evaluable_expected_source": (
                "anchor_definition"
                if str(variant_name) == str(anchor_name)
                else "generator_counters.fallback_to_original"
            ),
            "fallback_to_original": int(fallback_count),
            "evaluable_consistent": bool(
                int(0 if str(variant_name) == str(anchor_name) else n_diff_text)
                == int(
                    0
                    if str(variant_name) == str(anchor_name)
                    else max(0, int(n) - int(fallback_count))
                )
            ),
            "identical_positions_note": (
                "与原文支**逐字节相同**的条数（"
                + str(int(same_text))
                + " 条）没有「与原文的差异」可比 ⇒ 不计入 n_evaluable；"
                "它们的余弦恰为 1、裸字节相同，是**生成器回退**的结果，已如实登记。"
            ),
            "rule": (
                "`n_evaluable` = 与原文支**文本不同**的条数（只有这些条才存在"
                "「变体 vs 原文」的差异可比）。原文支自身 `n_evaluable = 0`（恒等）；"
                "非锚点支的 `n_evaluable` 必须等于 "
                "`n - n_evaluable_expected_source 报的回退数`"
                "（`n_evaluable_expected = n - fallback_to_original`）—— "
                "两侧来源不同（文本比对 vs 生成器计数），故该互证**不是恒真**。"
            ),
        },
        "info_loss_note": str(EXTRACTIVE_NOTE),
    }


__all__ = [
    "QA_VARIANTS_VERSION",
    "BINDING_OUTPUT_RULE",
    "GROUP_SPLIT_RULE",
    "GROUP_SPLIT_INVARIANTS",
    "EXTRACTIVE_NOTE",
    "VERIFY_ROOT",
    "DEFAULT_OUT_DIR",
    "QAQuestionUnit",
    "build_question_units",
    "canonical_json",
    "feature_difference_report",
    "generate_qa_variants",
    "group_split_counter_examples",
    "load_question_texts",
    "sha256_file",
    "sha256_text",
    "variant_records",
    "verify_group_split",
]