"""轮 1：QA 侧扩 C / 按比例留出 / 解题类任务拆分的纯新增 CLI。

驱动方式：``python -m n3d_qa_learn.qa_scale_run {probe|eval|report}``

子命令
------
* ``probe`` —— 数据取证（**不评测**）：现场枚举冻结产物的字段名与任务清单、统一答案表
  规模、问题/行/类别计数、以及**解题类拆分的依据**（judge / choice 的答案类数只有 2 / 4）。
* ``eval`` —— 扩 C 对照评测（**不训练**）：构建 C=10 重建与 C=298 全量两套语料，各做
  「问题组级按比例留出」切分，给出规模 / 分布 / 逐类覆盖 / 分任务分侧 / 分任务分组统计
  / 三条判据的适用性与边界归属 / 两个确定性参照基线（constant_majority 与 row_argmax）。
  ``--repro-dir`` 给定时额外产出**双跑逐字节一致**证据。
* ``report`` —— 渲染 ``qa_scale_report.json`` + ``qa_scale_report.md``。

产物纪律（**硬约束**）
----------------------
* 只写 ``checkpoints/qa_learn/_verify/qa_scale/``（``--out-dir`` 也只允许写在该根之下）；
* **不落盘裸特征矩阵**（不写 ``.npy`` / ``.npz`` / ``.pt``）、**不产 zip**、
  **不写任何挂钟 / 耗时字段**（故同参重跑逐字节一致）。

控制台编码纪律与 :mod:`n3d_qa_learn.merge_table_run` 同源：日志字面量只用 ASCII + 汉字，
``_log`` 做 ``backslashreplace`` 兜底；本模块**不 import 任何上游 backend**，故不存在
``n3d_shape.train`` 导入期改 ``sys.stdout`` 编码的问题（该风险已在 merge_table_run 登记）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 日志（ASCII 安全 + backslashreplace 兜底）
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """打印一行进度（带刷新；对不可表示字符做 ``backslashreplace`` 兜底）。"""
    text = str(msg)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = str(getattr(sys.stdout, "encoding", "") or "ascii")
        safe = text.encode(enc, errors="backslashreplace").decode(enc, errors="replace")
        print(safe, flush=True)


#: 本模块的 CLI 子命令（顺序敏感）。
QA_SCALE_COMMANDS: Tuple[str, ...] = ("probe", "eval", "report")

#: 双跑证据文件名（**唯一**来源）。
DETERMINISM_EVIDENCE_FILE: str = "determinism_check.json"

#: 证据文件自身的排除口径（**必须显式**，见轮 C 离朱 R74 D2 的教训）。
DETERMINISM_SCOPE: str = (
    "证据 = 主产物目录与对照目录**共有的非排除文件**逐文件比对 SHA256 与字节数；"
    "排除项 ① 证据文件自身（determinism_check.json，两目录都过滤 —— 否则 a_sha256 是"
    "「上一次运行留在盘上的自己」，且全新对照目录上首跑必然判 False）；② 后置可重渲件"
    "（qa_scale_report.json / .md，由 report 在证据之后写出）。单侧独有文件"
    "（a_only_files / b_only_files）**不纳入** determinism_ok 布尔值，但必须显式登记。"
)

#: 后置可重渲件（不进证据块）。
RERENDERABLE_FILES: Tuple[str, ...] = ("qa_scale_report.json", "qa_scale_report.md")


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def _write_text(path: str, text: str) -> Tuple[str, int]:
    """写 UTF-8 无 BOM 文本（LF）；返回 ``(sha256, bytes)``。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    blob = str(text).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(blob)
    return hashlib.sha256(blob).hexdigest(), int(len(blob))


def _write_json(path: str, obj: Any) -> Tuple[str, int]:
    """写 JSON（UTF-8 无 BOM，``sort_keys`` + 缩进 1 + 末尾换行）。"""
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    return _write_text(path, text)


def _read_json(path: str) -> Any:
    """读 JSON；缺文件给出**可读报错**。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "缺少输入产物 " + repr(path) + "；请先跑前置子命令（probe / eval）"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: str) -> str:
    """文件 SHA256（分块读）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _guard_out_dir(out_dir: str) -> str:
    """守卫产物目录：**只允许**写在 ``checkpoints/qa_learn/_verify/`` 之下。"""
    import n3d_qa_learn.qa_scale as QS

    text = str(out_dir) if out_dir else QS.DEFAULT_OUT_DIR
    root_abs = os.path.abspath(QS.VERIFY_ROOT)
    norm_abs = os.path.abspath(text)
    prefix = root_abs + os.sep
    if not (norm_abs == root_abs or norm_abs.startswith(prefix)):
        raise ValueError(
            "产物目录只允许写在 "
            + repr(QS.VERIFY_ROOT.replace(chr(92), "/")) + " 之下，当前 = "
            + repr(norm_abs.replace(chr(92), "/"))
        )
    return text


def _resolve_product_dir(explicit: str) -> str:
    """定位 ``n3d_qa`` 冻结产物目录（只读）。"""
    import n3d_qa_learn.qa_scale as QS

    if explicit:
        if os.path.isfile(os.path.join(explicit, QS.ALL_PAIRS_FILE)):
            return str(explicit)
        raise FileNotFoundError(
            "目录 " + repr(explicit) + " 下缺少 " + QS.ALL_PAIRS_FILE
        )
    if os.path.isfile(os.path.join(QS.DEFAULT_PRODUCT_DIR, QS.ALL_PAIRS_FILE)):
        return str(QS.DEFAULT_PRODUCT_DIR)
    raise FileNotFoundError(
        "未找到 n3d_qa 冻结产物（" + QS.ALL_PAIRS_FILE + "）；候选 = "
        + repr(QS.DEFAULT_PRODUCT_DIR)
    )


# ---------------------------------------------------------------------------
# 基线（C=10 现状口径）的**现场**复算 + 既有产物的只读取证
# ---------------------------------------------------------------------------

#: 现状（C=10）口径的**现场复算**参数来源说明。
BASELINE_SOURCE_RULE: str = (
    "现状基线（C=10 / 149 训练 / 20 测试 / 每类 2 条测试上限）由**既有的 QA 缓存口径**"
    "（data.build_answer_space + data.make_splits，参数与 entry_table.QA_* / exp_repr.BASE_* "
    "逐项一致）产生，其**现场数字**只从既有落盘产物/源码常量读取并逐项登记，"
    "**不凭记忆写**。注意：该基线的总体是 QA JSON 缓存（与冻结产物的统一答案表 298 类"
    "**不是同一个总体**），故它与「本轮冻结产物口径下的 C=10 重建」是两条**互不代替**的"
    "对照线，报告必须分别列出、不得混为一谈。"
)


def baseline_registered_values(module_dir: str = "n3d_qa_learn") -> Dict[str, Any]:
    """从**源码常量**现场读取 C=10 基线口径（只读；不做任何缓存扫描）。"""
    import n3d_qa_learn.entry_table as ET

    return {
        "source": "entry_table 的 QA_* 常量（现场 import 读取）",
        "QA_MAX_CLASSES": int(ET.QA_MAX_CLASSES),
        "QA_MIN_QUESTIONS": int(ET.QA_MIN_QUESTIONS),
        "QA_TEST_EVERY": int(ET.QA_TEST_EVERY),
        "QA_TEST_PER_CLASS": int(ET.QA_TEST_PER_CLASS),
        "QA_UNKNOWN_TRAIN_CAP": int(ET.QA_UNKNOWN_TRAIN_CAP),
        "QA_SPLIT_SEED": int(ET.QA_SPLIT_SEED),
        "QA_KNOWN_SPLIT": str(ET.QA_KNOWN_SPLIT),
        "rule": BASELINE_SOURCE_RULE,
    }


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    """数据取证（**不评测**）：字段名、任务清单、答案表规模、解题类拆分依据。"""
    import n3d_qa_learn.qa_scale as QS

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    product_dir = _resolve_product_dir(str(getattr(args, "product_dir", "") or ""))
    table_fields = QS.answer_table_fields(product_dir)
    pair_fields = QS.pair_record_fields(product_dir)
    manifest = QS.load_manifest(product_dir)
    corpus_full = QS.build_scale_corpus(product_dir, max_classes=0)
    # **W3 修复**：任务清单现场一致性校验（三处来源逐项相同 + 与 ALL_TASKS 比集合）
    task_list_check = QS.validate_task_list(product_dir)
    at = manifest["answer_table"]
    per_task_pairs: Dict[str, int] = {}
    for pair in corpus_full.pairs:
        per_task_pairs[str(pair.task)] = per_task_pairs.get(str(pair.task), 0) + 1
    # 解题类拆分依据：per-task 的答案类数（现场统计）
    per_task_classes: Dict[str, int] = {}
    per_task_questions: Dict[str, int] = {}
    per_task_positives: Dict[str, int] = {}
    for task in QS.ALL_TASKS:
        qs = [q for q, t in corpus_full.question_task.items() if str(t) == str(task)]
        classes: set = set()
        pos = 0
        for q in qs:
            for cls in corpus_full.question_gold[str(q)]:
                classes.add(int(cls))
        for pair in corpus_full.pairs:
            if str(pair.task) == str(task) and int(pair.label) == 1:
                pos += 1
        per_task_classes[str(task)] = int(len(classes))
        per_task_questions[str(task)] = int(len(qs))
        per_task_positives[str(task)] = int(pos)
    doc: Dict[str, Any] = {
        "version": QS.QA_SCALE_VERSION,
        "product_dir": str(product_dir),
        "product_files_sha256": {
            "answer_table.jsonl": QS.sha256_file(
                os.path.join(product_dir, QS.ANSWER_TABLE_FILE)
            ),
            "n3d_qa_all_pairs.jsonl": QS.sha256_file(
                os.path.join(product_dir, QS.ALL_PAIRS_FILE)
            ),
            "manifest.json": QS.sha256_file(os.path.join(product_dir, QS.MANIFEST_FILE)),
        },
        "field_enumeration": {
            "answer_table": table_fields,
            "pairs": pair_fields,
        },
        "manifest_claims": {
            "tasks": [str(t) for t in manifest["tasks"]],
            "answer_table.classes": int(at["classes"]),
            "answer_table.questions_seen": int(at["questions_seen"]),
            "answer_table.questions_in_table": int(at["questions_in_table"]),
            "answer_table.alias_count": int(at["alias_count"]),
            "answer_table.file": str(at.get("file", "")),
            "answer_table.class_list_ref": str(at.get("class_list_ref", "")),
            "answer_table_fields_present": sorted(str(k) for k in at.keys()),
            "answer_table.selection_rules": {
                str(k): str(v) for k, v in dict(at["selection_rules"]).items()
            },
            "files": [
                {"name": str(f["name"]), "bytes": int(f["bytes"]), "sha256": str(f["sha256"])}
                for f in manifest["files"]
            ],
        },
        "measured": {
            "answer_table_rows": int(table_fields["n_rows"]),
            "n_pairs": int(corpus_full.n_pairs),
            "n_questions": int(corpus_full.n_questions),
            "n_classes": int(corpus_full.n_classes),
            "per_task_pairs": {str(k): int(v) for k, v in sorted(per_task_pairs.items())},
            "per_task_questions": {str(k): int(v) for k, v in sorted(per_task_questions.items())},
            "per_task_positives": {str(k): int(v) for k, v in sorted(per_task_positives.items())},
            "per_task_answer_classes": {
                str(k): int(v) for k, v in sorted(per_task_classes.items())
            },
            "total_positive_rows": int(sum(per_task_positives.values())),
            "class_row_counts_distribution": QS.class_size_distribution(
                corpus_full.class_row_counts
            ),
            "class_question_counts_distribution": QS.class_size_distribution(
                corpus_full.class_question_counts
            ),
        },
        "task_stratification": {
            "tasks_manifest": [str(t) for t in QS.ALL_TASKS],
            "solve_tasks": [str(t) for t in QS.SOLVE_TASKS],
            "match_tasks": [str(t) for t in QS.MATCH_TASKS],
            "rule": str(QS.TASK_STRATIFICATION_RULE),
            "evidence": {
                "solve_tasks_answer_classes": {
                    str(t): int(per_task_classes.get(str(t), 0))
                    for t in QS.SOLVE_TASKS
                },
                "match_tasks_answer_classes": {
                    str(t): int(per_task_classes.get(str(t), 0))
                    for t in QS.MATCH_TASKS
                },
            },
        },
        "task_list_consistency": task_list_check,
        "baseline_registered": baseline_registered_values(),
        "ascii_safe_log_note": (
            "日志字面量只用 ASCII + 汉字；GBK 控制台下本模块不 import 任何上游 backend，"
            "故不存在导入期改 sys.stdout 编码的问题"
        ),
    }
    sha, size = _write_json(os.path.join(out_dir, "qa_scale_probe.json"), doc)
    _log(
        "[probe] 答案表 %d 行 / 类别 %d / 问题 %d / 行 %d；任务清单 = %s"
        % (
            int(table_fields["n_rows"]),
            int(corpus_full.n_classes),
            int(corpus_full.n_questions),
            int(corpus_full.n_pairs),
            str(doc["manifest_claims"]["tasks"]),
        )
    )
    _log(
        "[probe] 每任务答案类数 = "
        + str(doc["measured"]["per_task_answer_classes"])
        + "（解题任务 "
        + str(doc["task_stratification"]["evidence"]["solve_tasks_answer_classes"])
        + "）"
    )
    _log("[probe] qa_scale_probe.json = " + str(size) + " B " + sha[:16])
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def assert_merged_distribution_consistent(
    *,
    unit: str,
    train: Sequence[int],
    test: Sequence[int],
    merged: Sequence[int],
    expected_total: Optional[int] = None,
) -> Dict[str, Any]:
    """**E1 现场断言**：合并口径必须逐类等于「训练侧 + 测试侧」（逐类逐项）。

    为什么需要它（皋陶 error 的教训）
    ---------------------------------
    ``merged_by_row`` 曾经把「训练侧**问题数**」与「训练侧**正样本行数**」相加
    （``zip(q_all, r_all)``）—— 那是一个**既不是问题数也不是行数**的混合量，
    而且它的 ``total`` 看起来只是一个偏大的数字，**不会与任何现有断言冲突**。
    只靠「三个 total 自洽」是不够的：混合量的 total 同样可以"自洽"，只要
    分别与错误的分量比较。故本断言做的是**逐类**比较（``merged[i] == train[i] +
    test[i]``，全部 i 都要成立），并把不一致的**前若干个类下标与两侧取值**打出来。

    **反例证据（必须可复现，否则断言就是恒真）**：把 ``merged`` 换成
    ``[a + b for a, b in zip(train, test)][:-1] + [wrong]`` 任意改一个元素、
    或把 ``merged`` 换成「训练侧问题数 + 训练侧行数」那类错位数组，
    本函数都必须抛 ``AssertionError``。现场用
    ``merged_by_row`` 的历史污染值（C=10 total 16480）做过反例验证。

    参数
    ----
    unit : str
        单位标签（``'question'`` / ``'row'``），只进返回的取证字典。
    train / test / merged : Sequence[int]
        逐类计数（三者长度必须相同）。
    expected_total : Optional[int]
        若给定，则额外校验 ``sum(merged) == expected_total``。

    返回
    ----
    Dict[str, Any]
        取证字典（``unit`` / ``n_classes`` / ``total_train`` / ``total_test`` /
        ``total_merged`` / ``all_classes_consistent``）。

    异常
    ------
    AssertionError
        任一类的 ``merged != train + test``、或三者长度不一致、或 total 不符。
    """
    a = [int(x) for x in train]
    b = [int(x) for x in test]
    m = [int(x) for x in merged]
    if not (len(a) == len(b) == len(m)):
        raise AssertionError(
            "合并口径断言失败：三者长度不一致 "
            f"(train={len(a)} / test={len(b)} / merged={len(m)})，unit={unit!r}"
        )
    want = [int(x) + int(y) for x, y in zip(a, b)]
    bad = [i for i in range(len(m)) if int(m[i]) != int(want[i])]
    if bad:
        head = [
            {"class_index": int(i), "train": int(a[i]), "test": int(b[i]),
             "expected_merged": int(want[i]), "actual_merged": int(m[i])}
            for i in bad[:10]
        ]
        raise AssertionError(
            f"合并口径断言失败（unit={unit!r}）：有 {len(bad)} / {len(m)} 个类不满足 "
            f"merged == train + test；前几个 = {head}；"
            "（历史污染的混合量 total 会明显偏大，且逐类不等）"
        )
    total_merged = int(sum(m))
    if expected_total is not None and total_merged != int(expected_total):
        raise AssertionError(
            f"合并口径断言失败（unit={unit!r}）：sum(merged) = {total_merged} "
            f"!= expected_total = {int(expected_total)}"
        )
    return {
        "unit": str(unit),
        "n_classes": int(len(m)),
        "total_train": int(sum(a)),
        "total_test": int(sum(b)),
        "total_merged": int(total_merged),
        "all_classes_consistent": True,
        "rule": (
            "merged[i] == train[i] + test[i]（**逐类逐项**，不容许只对 total 自洽）；"
            "反例（任改一个元素 / 用错位数组）必须抛 AssertionError。"
        ),
    }


def _scale_block(
    *,
    label: str,
    corpus: Any,
    holdout_pct: int,
    seed: int,
) -> Dict[str, Any]:
    """单个口径（C=10 重建 / C=298 全量）的完整规模与切分证据。"""
    import n3d_qa_learn.qa_scale as QS

    split = QS.make_scale_split(corpus, holdout_pct=int(holdout_pct), seed=int(seed))
    q_all = QS.side_class_counts(corpus, split, side="train", unit="question")
    q_test = QS.side_class_counts(corpus, split, side="test", unit="question")
    r_all = QS.side_class_counts(corpus, split, side="train", unit="row")
    r_test = QS.side_class_counts(corpus, split, side="test", unit="row")
    merged_q = [int(a) + int(b) for a, b in zip(q_all, q_test)]
    # **E1 修复（皋陶 error）**：`merged_r` 必须与 `merged_q` **对称** —— 都取
    # 「训练侧 + 测试侧」的**同一单位**。修复前写的是 `zip(q_all, r_all)`
    # （把「训练侧问题数」与「训练侧正样本行数」相加），得到的是一个既不是问题数、
    # 也不是行数的混合量（现场污染值：C=10 total 16480、C=298 total 21698）。
    merged_r = [int(a) + int(b) for a, b in zip(r_all, r_test)]
    baseline = QS.deterministic_baseline_metrics(corpus, split, side="test")
    baseline_train = QS.deterministic_baseline_metrics(corpus, split, side="train")
    const_macro = float(baseline["constant_majority"]["macro"])
    # **E1 的现场断言 + total 自洽取证**（逐类逐项，不容许只对 total 自洽）
    consistency = {
        "question": assert_merged_distribution_consistent(
            unit="question", train=q_all, test=q_test, merged=merged_q
        ),
        "row": assert_merged_distribution_consistent(
            unit="row", train=r_all, test=r_test, merged=merged_r
        ),
    }
    consistency_rule = (
        "merged[i] == train[i] + test[i]（**逐类逐项**）；question 与 row 两个单位各自独立校验。"
        "历史污染（merged_by_row = zip(q_all, r_all)）会在此抛 AssertionError —— "
        "该断言**不是恒真**：任改一个元素或用错位数组都会失败（见 §E1 反例）。"
    )
    block: Dict[str, Any] = {
        "label": str(label),
        "n_classes": int(corpus.n_classes),
        "n_pairs": int(corpus.n_pairs),
        "n_questions": int(corpus.n_questions),
        "answer_keys_head": [str(x) for x in corpus.answer_keys[:5]],
        "seed": int(seed),
        "holdout_pct": int(holdout_pct),
        "split": {
            **split.summary(),
            "actual_test_question_ratio": (
                float(len(split.test_questions)) / float(corpus.n_questions)
            ),
            "actual_test_row_ratio": (
                float(len(split.test_rows)) / float(corpus.n_pairs)
            ),
            "target_ratio": float(holdout_pct) / 100.0,
            "rule": str(QS.HOLDOUT_RULE),
        },
        "class_distribution": {
            "unit_note": (
                "两套口径分开报：`by_question` 以**问题数**计、`by_row` 以**正样本行数**计；"
                "二者不可互换（一个问题可对多个类、一个类可被多行命中）"
            ),
            "merged_by_question": QS.class_size_distribution(merged_q),
            "merged_by_row": QS.class_size_distribution(merged_r),
            "frozen_table_count": QS.class_size_distribution(corpus.class_row_counts),
            "train_by_question": QS.class_size_distribution(q_all),
            "test_by_question": QS.class_size_distribution(q_test),
            "train_by_row": QS.class_size_distribution(r_all),
            "test_by_row": QS.class_size_distribution(r_test),
            "merged_consistency": consistency,
            "merged_consistency_rule": consistency_rule,
            "merged_total_self_consistency": {
                "question": {
                    "merged_total": int(consistency["question"]["total_merged"]),
                    "train_total": int(consistency["question"]["total_train"]),
                    "test_total": int(consistency["question"]["total_test"]),
                    "train_plus_test": int(
                        consistency["question"]["total_train"]
                        + consistency["question"]["total_test"]
                    ),
                },
                "row": {
                    "merged_total": int(consistency["row"]["total_merged"]),
                    "train_total": int(consistency["row"]["total_train"]),
                    "test_total": int(consistency["row"]["total_test"]),
                    "train_plus_test": int(
                        consistency["row"]["total_train"] + consistency["row"]["total_test"]
                    ),
                },
                "rule": "merged_total == train_total + test_total（两个单位各自成立）",
            },
        },
        "class_coverage": {
            "train_by_question": QS.class_coverage(q_all),
            "test_by_question": QS.class_coverage(q_test),
            "train_by_row": QS.class_coverage(r_all),
            "test_by_row": QS.class_coverage(r_test),
        },
        "small_class_actual_samples": {
            "rule": (
                "小样本类在留出后的**实际**样本数：按「合并后问题数」升序取前 20 个类，"
                "逐个给出该类的合并问题数 / 训练问题数 / 测试问题数 / 测试正样本行数。"
                "**不**强制每类至少 1 条测试样本（那会破坏比例性），故测试侧为 0 的类必须"
                "在此**逐个出现**，不得用汇总值掩盖。"
            ),
            "head": [
                {
                    "class_index": int(i),
                    "class_key": str(corpus.answer_keys[int(i)]),
                    "merged_questions": int(merged_q[int(i)]),
                    "train_questions": int(q_all[int(i)]),
                    "test_questions": int(q_test[int(i)]),
                    "test_positive_rows": int(r_test[int(i)]),
                    "frozen_table_count": int(corpus.class_row_counts[int(i)]),
                }
                for i in sorted(
                    range(int(corpus.n_classes)),
                    key=lambda j: (int(merged_q[j]), str(corpus.answer_keys[j])),
                )[:20]
            ],
        },
        "task_side_stats": {
            "train": QS.task_side_stats(corpus, split, side="train"),
            "test": QS.task_side_stats(corpus, split, side="test"),
        },
        "grouped_stats": {
            "train": QS.grouped_stats(corpus, split, side="train"),
            "test": QS.grouped_stats(corpus, split, side="test"),
        },
        "deterministic_baselines": {
            "test": baseline,
            "train": baseline_train,
            "criterion_table_using_test_baseline": QS.baseline_criterion_table(
                const_macro, int(corpus.n_classes)
            ),
        },
    }
    # 分组（匹配 / 解题）分别再算一遍确定性基线，供分侧报告
    for group, tasks in (("match", QS.MATCH_TASKS), ("solve", QS.SOLVE_TASKS)):
        block["grouped_stats"][str(group)] = {
            "test": QS.deterministic_baseline_metrics(
                corpus, split, side="test", tasks=tasks
            ),
            "train": QS.deterministic_baseline_metrics(
                corpus, split, side="train", tasks=tasks
            ),
        }
    # **W1 修复（皋陶 warning）**：判据必须**按作用域**给出 —— 全量类口径的
    # constant_majority 与「解题任务集（2/4 类）」「匹配任务集（153/78/100 类）」
    # **不是同一量纲**，前者绝不能被套用到后者的分侧 macro 上。故这里为每个作用域
    # 各算一份同作用域基线 + 同作用域阈值（+10pp / +5pp）。
    scoped_tables: Dict[str, Any] = {
        "all_classes": {
            "task_set": [],
            "n_classes": int(corpus.n_classes),
            "baseline_macro_test": float(
                baseline["constant_majority"]["macro"]
            ),
            "criterion_table": QS.baseline_criterion_table(
                float(baseline["constant_majority"]["macro"]), int(corpus.n_classes)
            ),
        }
    }
    for group, tasks in (("solve", QS.SOLVE_TASKS), ("match", QS.MATCH_TASKS)):
        gm = float(
            block["grouped_stats"][str(group)]["test"]["constant_majority"]["macro"]
        )
        gn = int(
            block["grouped_stats"][str(group)]["test"]["constant_majority"][
                "n_classes_in_scope"
            ]
        )
        scoped_tables[str(group) + "_tasks"] = {
            "task_set": [str(t) for t in tasks],
            "n_classes": int(gn),
            "baseline_macro_test": float(gm),
            "criterion_table": QS.baseline_criterion_table(float(gm), int(gn)),
        }
    block["deterministic_baselines"]["criterion_tables_by_scope"] = scoped_tables
    block["deterministic_baselines"]["criterion_scope"] = QS.criterion_scope_declaration(
        all_classes=int(corpus.n_classes),
        solve_macro=float(
            block["grouped_stats"]["solve"]["test"]["constant_majority"]["macro"]
        ),
        match_macro=float(
            block["grouped_stats"]["match"]["test"]["constant_majority"]["macro"]
        ),
    )
    return block


def cmd_eval(args: argparse.Namespace) -> int:
    """扩 C 对照评测（**不训练**）+ 三条判据的适用性 + 确定性参照基线。"""
    import n3d_qa_learn.qa_scale as QS

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    repro_dir = str(getattr(args, "repro_dir", "") or "")
    product_dir = _resolve_product_dir(str(getattr(args, "product_dir", "") or ""))
    holdout_pct = int(getattr(args, "holdout_pct", QS.DEFAULT_HOLDOUT_PCT))
    seed = int(getattr(args, "seed", 42) or 42)
    full_c = int(getattr(args, "full_classes", 0) or 0)
    base_c = int(getattr(args, "baseline_classes", 0) or QS.BASELINE_MAX_CLASSES)
    manifest = QS.load_manifest(product_dir)
    declared = int(manifest["answer_table"]["classes"])
    if full_c <= 0:
        full_c = declared
    corpus_full = QS.build_scale_corpus(product_dir, max_classes=0)
    if int(corpus_full.n_classes) != int(declared):
        raise ValueError(
            f"统一答案表现场类别数 {corpus_full.n_classes} 与 manifest 声明 {declared} "
            "不一致；拒绝在自相矛盾的类数上出数"
        )
    if int(full_c) != int(declared):
        raise ValueError(
            f"--full-classes={full_c} 与 manifest 声明的全量类数 {declared} 不一致；"
            "本轮「全量」口径由 manifest 钉死，不接受运行期缩小"
        )
    corpus_base = QS.build_scale_corpus(product_dir, max_classes=int(base_c))
    # **W3 修复**：eval 路径同样现场校验任务清单（fail-closed；不允许绕过 probe）
    task_list_check = QS.validate_task_list(product_dir)
    _log(
        "[eval] 冻结产物 = "
        + str(product_dir)
        + "；答案表类数 = " + str(corpus_full.n_classes)
        + "；问题 = " + str(corpus_full.n_questions)
        + "；行 = " + str(corpus_full.n_pairs)
    )
    holdout_ev = QS.proportional_holdout_evidence(int(seed), int(holdout_pct))
    blocks = {
        "baseline_c%d" % int(base_c): _scale_block(
            label="baseline_c%d" % int(base_c),
            corpus=corpus_base,
            holdout_pct=int(holdout_pct),
            seed=int(seed),
        ),
        "full_c%d" % int(full_c): _scale_block(
            label="full_c%d" % int(full_c),
            corpus=corpus_full,
            holdout_pct=int(holdout_pct),
            seed=int(seed),
        ),
    }
    base_block = blocks["baseline_c%d" % int(base_c)]
    full_block = blocks["full_c%d" % int(full_c)]
    comparison = {
        "rule": (
            "两套口径**使用同一选择规则、同一留出比例、同一 seed**，唯一变量是**类别数上限**"
            "（C=10 重建 vs C=298 全量）⇒ 倍数与分布变化可直接归因于类别数上限。"
        ),
        "n_classes": {
            "baseline": int(base_block["n_classes"]),
            "full": int(full_block["n_classes"]),
            "multiple": float(full_block["n_classes"]) / float(base_block["n_classes"]),
        },
        "n_questions": {
            "baseline": int(base_block["n_questions"]),
            "full": int(full_block["n_questions"]),
            "multiple": float(full_block["n_questions"])
            / float(base_block["n_questions"]),
        },
        "n_pairs": {
            "baseline": int(base_block["n_pairs"]),
            "full": int(full_block["n_pairs"]),
            "multiple": float(full_block["n_pairs"]) / float(base_block["n_pairs"]),
        },
        "n_train_questions": {
            "baseline": int(base_block["split"]["n_train_questions"]),
            "full": int(full_block["split"]["n_train_questions"]),
            "multiple": float(full_block["split"]["n_train_questions"])
            / float(base_block["split"]["n_train_questions"]),
        },
        "n_test_questions": {
            "baseline": int(base_block["split"]["n_test_questions"]),
            "full": int(full_block["split"]["n_test_questions"]),
            "multiple": float(full_block["split"]["n_test_questions"])
            / float(base_block["split"]["n_test_questions"]),
        },
        "n_train_rows": {
            "baseline": int(base_block["split"]["n_train_rows"]),
            "full": int(full_block["split"]["n_train_rows"]),
            "multiple": float(full_block["split"]["n_train_rows"])
            / float(base_block["split"]["n_train_rows"]),
        },
        "n_test_rows": {
            "baseline": int(base_block["split"]["n_test_rows"]),
            "full": int(full_block["split"]["n_test_rows"]),
            "multiple": float(full_block["split"]["n_test_rows"])
            / float(base_block["split"]["n_test_rows"]),
        },
        "merged_distribution_by_question": {
            "baseline": dict(base_block["class_distribution"]["merged_by_question"]),
            "full": dict(full_block["class_distribution"]["merged_by_question"]),
        },
        "category_coverage_test": {
            "baseline": dict(base_block["class_coverage"]["test_by_question"]),
            "full": dict(full_block["class_coverage"]["test_by_question"]),
        },
        "deterministic_baseline_macro_test": {
            "baseline": float(
                base_block["deterministic_baselines"]["test"]["constant_majority"]["macro"]
            ),
            "full": float(
                full_block["deterministic_baselines"]["test"]["constant_majority"]["macro"]
            ),
            "note": (
                "确定性参照基线（一律答多数类）的 macro。**不作为「模型能力」**，"
                "只作为「扩 C 后 macro 判据的分母/门槛是否仍可判」的现场锚点。"
            ),
        },
    }
    task_split = {
        "rule": str(QS.TASK_STRATIFICATION_RULE),
        "solve_tasks": [str(t) for t in QS.SOLVE_TASKS],
        "match_tasks": [str(t) for t in QS.MATCH_TASKS],
        "per_task_evidence_full": {
            str(t): {
                "train": dict(full_block["task_side_stats"]["train"][str(t)]),
                "test": dict(full_block["task_side_stats"]["test"][str(t)]),
            }
            for t in QS.ALL_TASKS
        },
        "per_task_evidence_baseline": {
            str(t): {
                "train": dict(base_block["task_side_stats"]["train"][str(t)]),
                "test": dict(base_block["task_side_stats"]["test"][str(t)]),
            }
            for t in QS.ALL_TASKS
        },
        "impact_on_merged_metric": {
            "rule": (
                "拆分对**合入口径**指标的影响必须显式给出：同一份切分下分别算"
                "「全部任务」/「匹配任务集」/「解题任务集」三个范围的确定性基线 macro，"
                "并给出「拆出后各自报告的宏平均」与「不拆分时的宏平均」的差。"
            ),
            "full": {
                "all_tasks": dict(full_block["grouped_stats"]["test"]),
                "baselines_all": dict(
                    full_block["deterministic_baselines"]["test"]["constant_majority"]
                ),
                "baselines_match": dict(
                    full_block["grouped_stats"]["match"]["test"]["constant_majority"]
                ),
                "baselines_solve": dict(
                    full_block["grouped_stats"]["solve"]["test"]["constant_majority"]
                ),
            },
            "baseline": {
                "all_tasks": dict(base_block["grouped_stats"]["test"]),
                "baselines_all": dict(
                    base_block["deterministic_baselines"]["test"]["constant_majority"]
                ),
                "baselines_match": dict(
                    base_block["grouped_stats"]["match"]["test"]["constant_majority"]
                ),
                "baselines_solve": dict(
                    base_block["grouped_stats"]["solve"]["test"]["constant_majority"]
                ),
            },
        },
    }
    doc: Dict[str, Any] = {
        "version": QS.QA_SCALE_VERSION,
        "effective_scope": {
            "max_classes": int(full_c),
            "holdout_pct": int(holdout_pct),
            "seed": int(seed),
            "single_seed": True,
            "statement": (
                "所有结论的生效口径 = max_classes=%d（全量）+ %d%% 按比例留出 + 单 seed(%d)"
                % (int(full_c), int(holdout_pct), int(seed))
            ),
        },
        "product_dir": str(product_dir),
        "product_files_sha256": {
            "answer_table.jsonl": QS.sha256_file(
                os.path.join(product_dir, QS.ANSWER_TABLE_FILE)
            ),
            "n3d_qa_all_pairs.jsonl": QS.sha256_file(
                os.path.join(product_dir, QS.ALL_PAIRS_FILE)
            ),
            "manifest.json": QS.sha256_file(os.path.join(product_dir, QS.MANIFEST_FILE)),
        },
        "frozen_constants": {
            "QA_SCALE_VERSION": str(QS.QA_SCALE_VERSION),
            "FULL_CLASSES": int(QS.FULL_CLASSES),
            "BASELINE_MAX_CLASSES": int(QS.BASELINE_MAX_CLASSES),
            "BASELINE_MIN_QUESTIONS": int(QS.BASELINE_MIN_QUESTIONS),
            "BASELINE_TEST_EVERY": int(QS.BASELINE_TEST_EVERY),
            "BASELINE_TEST_PER_CLASS": int(QS.BASELINE_TEST_PER_CLASS),
            "DEFAULT_HOLDOUT_PCT": int(QS.DEFAULT_HOLDOUT_PCT),
            "HOLDOUT_PCT_RANGE": [int(x) for x in QS.HOLDOUT_PCT_RANGE],
            "ALL_TASKS": [str(t) for t in QS.ALL_TASKS],
            "SOLVE_TASKS": [str(t) for t in QS.SOLVE_TASKS],
            "MATCH_TASKS": [str(t) for t in QS.MATCH_TASKS],
            "CLASS_SIZE_BINS": [
                [str(n), int(lo), (None if hi is None else int(hi))]
                for n, lo, hi in QS.CLASS_SIZE_BINS
            ],
            "HOLDOUT_RULE": str(QS.HOLDOUT_RULE),
            "CRITERION_RULE": str(QS.CRITERION_RULE),
            "TASK_STRATIFICATION_RULE": str(QS.TASK_STRATIFICATION_RULE),
        },
        "holdout_boundary_evidence": holdout_ev,
        "criterion_applicability": {
            "full": QS.macro_criterion_applicability(int(corpus_full.n_classes)),
            "baseline": QS.macro_criterion_applicability(int(corpus_base.n_classes)),
        },
        "baseline_registered_legacy": baseline_registered_values(),
        "task_list_consistency": task_list_check,
        "scales": blocks,
        "comparison": comparison,
        "task_split": task_split,
    }
    sha, size = _write_json(os.path.join(out_dir, "qa_scale_eval.json"), doc)
    _log(
        "[eval] C=10 重建：类 %d / 问题 %d / 行 %d / 训练问题 %d / 测试问题 %d"
        % (
            int(base_block["n_classes"]),
            int(base_block["n_questions"]),
            int(base_block["n_pairs"]),
            int(base_block["split"]["n_train_questions"]),
            int(base_block["split"]["n_test_questions"]),
        )
    )
    _log(
        "[eval] C=%d 全量：类 %d / 问题 %d / 行 %d / 训练问题 %d / 测试问题 %d"
        % (
            int(full_block["n_classes"]),
            int(full_block["n_classes"]),
            int(full_block["n_questions"]),
            int(full_block["n_pairs"]),
            int(full_block["split"]["n_train_questions"]),
            int(full_block["split"]["n_test_questions"]),
        )
    )
    _log(
        "[eval] 倍数：问题 x%.3f / 行 x%.3f / 训练问题 x%.3f / 测试问题 x%.3f"
        % (
            float(comparison["n_questions"]["multiple"]),
            float(comparison["n_pairs"]["multiple"]),
            float(comparison["n_train_questions"]["multiple"]),
            float(comparison["n_test_questions"]["multiple"]),
        )
    )
    _log("[eval] qa_scale_eval.json = " + str(size) + " B " + sha[:16])
    if repro_dir:
        det = determinism_block(str(out_dir), str(repro_dir))
        _write_json(os.path.join(out_dir, DETERMINISM_EVIDENCE_FILE), det)
        _log(
            "[eval] 双跑一致性：共有文件 %d，逐字节相同 = %s，不相等文件 = %s"
            % (
                int(det["n_common_files"]),
                str(det["determinism_ok"]),
                str(det["not_equal_files"]),
            )
        )
    return 0


# ---------------------------------------------------------------------------
# 双跑逐字节一致证据
# ---------------------------------------------------------------------------


def determinism_block(out_dir: str, repro_dir: str) -> Dict[str, Any]:
    """双跑逐字节一致证据（**排除证据文件自身与后置可重渲件**）。"""
    excluded = {str(DETERMINISM_EVIDENCE_FILE)} | {str(x) for x in RERENDERABLE_FILES}
    a_all = sorted(
        f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))
    )
    b_all = sorted(
        f for f in os.listdir(repro_dir) if os.path.isfile(os.path.join(repro_dir, f))
    )
    a_files = [f for f in a_all if f not in excluded]
    b_files = [f for f in b_all if f not in excluded]
    files: List[Dict[str, Any]] = []
    for name in sorted(set(a_files) & set(b_files)):
        pa = os.path.join(out_dir, name)
        pb = os.path.join(repro_dir, name)
        sa, aa = _sha256_file(pa), int(os.path.getsize(pa))
        sb, ab = _sha256_file(pb), int(os.path.getsize(pb))
        files.append(
            {
                "name": str(name),
                "a_sha256": str(sa),
                "b_sha256": str(sb),
                "a_bytes": int(aa),
                "b_bytes": int(ab),
                "bitwise_equal": bool(sa == sb and aa == ab),
            }
        )
    a_only = [f for f in a_files if f not in b_files]
    b_only = [f for f in b_files if f not in a_files]
    not_equal = [f["name"] for f in files if not bool(f["bitwise_equal"])]

    def _norm(p: str) -> str:
        return os.path.abspath(str(p)).replace("\\", "/").lower()

    return {
        "a_dir": str(out_dir),
        "b_dir": str(repro_dir),
        "a_dir_normalized": _norm(out_dir),
        "b_dir_normalized": _norm(repro_dir),
        "paths_normalized_equal": bool(_norm(out_dir) == _norm(repro_dir)),
        "path_dependence_note": (
            "`a_dir` / `b_dir` 保留**命令原样串**（故 Windows 下 \\ 与 / 的字形差异会被保留）；"
            "另给 `*_normalized`（绝对路径 + 统一正斜杠 + 小写）与 `paths_normalized_equal` "
            "供机器判定。**这两个原样串会进产物内容 ⇒ 凡内嵌它们的产物（本文件与 report）"
            "在「绝对路径不同」的运行之间不逐字节相同；同路径同参重跑才逐字节相同** —— "
            "这是「不落盘挂钟字段」之外的**唯一**已知非确定性来源，已显式登记。"
        ),
        "files": files,
        "n_common_files": int(len(files)),
        "n_files_before_exclusion": {"a": int(len(a_all)), "b": int(len(b_all))},
        "excluded_files": sorted(excluded),
        "a_only_files": a_only,
        "b_only_files": b_only,
        "not_equal_files": not_equal,
        "determinism_ok": bool(len(files) > 0 and not not_equal),
        "determinism_ok_semantics": (
            "determinism_ok = 「两目录**共有的非排除文件**全部逐字节相等」且集合非空；"
            "单侧独有文件不纳入该布尔值（真实工作序下对照目录首跑时天然缺少证据文件自身），"
            "但必须显式登记在 a_only_files / b_only_files。"
        ),
        "determinism_scope": DETERMINISM_SCOPE,
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def render_markdown(rep: Dict[str, Any]) -> str:
    """渲染 Markdown 报告（**确定性**：无时间戳、无耗时）。"""
    ev = rep["eval"]
    cmp_ = ev["comparison"]
    lines: List[str] = []
    lines.append("# 轮 1：QA 侧扩 C 至 298 + 按比例留出 + 解题类任务拆分")
    lines.append("")
    lines.append("## 1. 生效口径")
    lines.append("")
    lines.append("> " + str(ev["effective_scope"]["statement"]))
    lines.append("")
    lines.append("- 冻结产物目录：`" + str(ev["product_dir"]) + "`")
    for name, sha in sorted(ev["product_files_sha256"].items()):
        lines.append("  - `" + str(name) + "` SHA256 = `" + str(sha) + "`")
    lines.append("- **本轮不训练任何模型**：只给数据口径、规模、判据适用性与**确定性参照基线**")
    lines.append("")
    lines.append("## 2. 扩 C 前后对照（同一选择规则 / 同一留出比例 / 同一 seed）")
    lines.append("")
    lines.append("| 指标 | C=" + str(cmp_["n_classes"]["baseline"]) + "（重建基线） | C="
                 + str(cmp_["n_classes"]["full"]) + "（全量） | 倍数 |")
    lines.append("| --- | --- | --- | --- |")
    for key, label in (
        ("n_classes", "类别数 C"),
        ("n_questions", "问题数（切分单位）"),
        ("n_pairs", "问答对行数"),
        ("n_train_questions", "训练问题数"),
        ("n_test_questions", "测试问题数"),
        ("n_train_rows", "训练行数"),
        ("n_test_rows", "测试行数"),
    ):
        row = cmp_[str(key)]
        lines.append(
            "| " + str(label) + " | " + str(row["baseline"]) + " | " + str(row["full"])
            + " | x" + ("%.3f" % float(row["multiple"])) + " |"
        )
    lines.append("")
    lines.append("- " + str(cmp_["rule"]))
    lines.append("")
    lines.append("### 2.1 每类样本数分布对照（按问题数计）")
    lines.append("")
    lines.append("| 统计量 | C=" + str(cmp_["n_classes"]["baseline"]) + " | C="
                 + str(cmp_["n_classes"]["full"]) + " |")
    lines.append("| --- | --- | --- |")
    for key in ("n", "min", "p25", "median", "p75", "max", "mean", "zero_count", "total"):
        lines.append(
            "| `" + str(key) + "` | "
            + str(cmp_["merged_distribution_by_question"]["baseline"].get(key))
            + " | " + str(cmp_["merged_distribution_by_question"]["full"].get(key)) + " |"
        )
    lines.append("")
    lines.append("分档直方图（`" + str(ev["frozen_constants"]["CLASS_SIZE_BINS"]) + "`）：")
    lines.append("")
    lines.append("| 档位 | C=" + str(cmp_["n_classes"]["baseline"]) + " | C="
                 + str(cmp_["n_classes"]["full"]) + " |")
    lines.append("| --- | --- | --- |")
    for name in [str(x[0]) for x in ev["frozen_constants"]["CLASS_SIZE_BINS"]]:
        lines.append(
            "| `" + str(name) + "` | "
            + str(cmp_["merged_distribution_by_question"]["baseline"]["histogram"].get(name))
            + " | "
            + str(cmp_["merged_distribution_by_question"]["full"]["histogram"].get(name))
            + " |"
        )
    lines.append("")
    lines.append("### 2.2 类别覆盖（测试侧，按问题数）")
    lines.append("")
    lines.append("| 口径 | 类数 | 有样本类数 | 无样本类数 | 无样本类（前 20） |")
    lines.append("| --- | --- | --- | --- | --- |")
    for label, cov in (
        ("C=" + str(cmp_["n_classes"]["baseline"]),
         cmp_["category_coverage_test"]["baseline"]),
        ("C=" + str(cmp_["n_classes"]["full"]),
         cmp_["category_coverage_test"]["full"]),
    ):
        lines.append(
            "| " + str(label) + " | " + str(cov["n_classes"]) + " | "
            + str(cov["n_classes_with_samples"]) + " | "
            + str(cov["n_classes_without_samples"]) + " | `"
            + str(cov["classes_without_samples_head"]) + "` |"
        )
    lines.append("")
    lines.append("- " + str(cmp_["category_coverage_test"]["full"]["rule"]))
    lines.append("")
    lines.append("## 3. 按比例留出（边界归属 = 小于）")
    lines.append("")
    hb = ev["holdout_boundary_evidence"]
    lines.append("- 口径：" + str(hb["rule"]))
    lines.append("- 隐含命中概率：" + str(hb["implied_test_probability"]))
    lines.append("- 边界邻域现场判定：")
    lines.append("")
    lines.append("| h | 本实现（`<`） | 若改成 `<=` | 两者可区分 |")
    lines.append("| --- | --- | --- | --- |")
    for row in hb["boundary_probe"]:
        lines.append(
            "| " + str(row["h"]) + " | `" + str(row["implemented_rule_lt"]) + "` | `"
            + str(row["alternative_rule_le"]) + "` | "
            + str(row["rules_distinguishable_at_this_point"]) + " |"
        )
    lines.append("")
    lines.append("- 为什么不是 `<=`：" + str(hb["why_not_le"]))
    lines.append("- 退化端：" + str(hb["degenerate_cases"]))
    lines.append("")
    lines.append("## 4. 判据退化修正（10/C 的适用性）")
    lines.append("")
    lines.append("| 口径 | C | 阈值 10/C | applicable | 理由 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for label, app in (
        ("C=" + str(cmp_["n_classes"]["baseline"]), ev["criterion_applicability"]["baseline"]),
        ("C=" + str(cmp_["n_classes"]["full"]), ev["criterion_applicability"]["full"]),
    ):
        lines.append(
            "| " + str(label) + " | " + str(app["n_classes"]) + " | "
            + ("%.6f" % float(app["threshold"])) + " | **" + str(app["applicable"]) + "** | "
            + str(app["reason"]) + " |"
        )
    lines.append("")
    lines.append("- 边界归属：" + str(ev["criterion_applicability"]["baseline"]["boundary_rule"]))
    lines.append("- " + str(ev["frozen_constants"]["CRITERION_RULE"]))
    lines.append("")
    lines.append("### 4.1 三条判据的现场判定表（以 C=298 全量测试侧基线为准）")
    lines.append("")
    full_scale = ev["scales"]["full_c" + str(cmp_["n_classes"]["full"])]
    ctab = full_scale["deterministic_baselines"]["criterion_table_using_test_baseline"]
    lines.append("| 判据 | 表达式 | 阈值 | applicable | 边界 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for key in ("criterion_random_x10", "criterion_majority_plus_10pp",
                "criterion_majority_plus_5pp"):
        row = ctab[str(key)]
        lines.append(
            "| `" + str(key) + "` | `" + str(row["expression"]) + "` | "
            + ("%.6f" % float(row["threshold"])) + " | **" + str(row["applicable"]) + "** | "
            + str(row.get("boundary_rule", "")) + " |"
        )
    lines.append("")
    sat = ctab["criterion_majority_plus_10pp"]["baseline_saturation"]
    lines.append("- 多数类基线可行域核查：" + str(sat))
    lines.append("")
    # **P2 修复（轮 3 收口）**：把 W1 承诺的**作用域表**渲染进报告（此前只落 JSON 产物），
    # 并把 `cross_scope_rule` 一并渲染 —— 使「README / 报告 / 产物」三处同源。
    lines.append("### 4.2 判据基线的**作用域**表（`criterion_tables_by_scope`，禁止跨作用域套用）")
    lines.append("")
    lines.append("| 作用域 | 任务集 | 类数 | 基线 macro（测试侧） | `10/C` 阈值 | applicable | `+10pp` 阈值 | `+5pp` 阈值 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    scope_order = ("all_classes", "match_tasks", "solve_tasks")
    for scope in scope_order:
        cell = full_scale["deterministic_baselines"]["criterion_tables_by_scope"].get(
            str(scope)
        )
        if not cell:
            continue
        ct = cell["criterion_table"]
        tasks_txt = "（全部任务）" if not cell["task_set"] else str(cell["task_set"])
        lines.append(
            "| `" + str(scope) + "` | " + str(tasks_txt) + " | " + str(cell["n_classes"])
            + " | " + ("%.6f" % float(cell["baseline_macro_test"]))
            + " | " + ("%.6f" % float(ct["criterion_random_x10"]["threshold"]))
            + " | **" + str(ct["criterion_random_x10"]["applicable"]) + "**"
            + " | " + ("%.6f" % float(ct["criterion_majority_plus_10pp"]["threshold"]))
            + " | " + ("%.6f" % float(ct["criterion_majority_plus_5pp"]["threshold"]))
            + " |"
        )
    lines.append("")
    scope_decl = full_scale["deterministic_baselines"]["criterion_scope"]
    lines.append("- **`cross_scope_rule`（跨作用域规则，禁止跨格套用）**："
                 + str(scope_decl["cross_scope_rule"]))
    lines.append("- `why`（为什么不能套用，量纲）：" + str(scope_decl["why"]))
    lines.append("- `scopes`（各作用域界定）：" + str(scope_decl["scopes"]))
    lines.append("")
    # **P2 修复**：渲染 E1 的 `merged_consistency` 取证块（此前只在 JSON 产物里）。
    lines.append("### 4.3 E1 取证：合并口径的**逐类**自洽（`merged_consistency`）")
    lines.append("")
    lines.append("| 单位 | 类数 | 训练侧合计 | 测试侧合计 | 合并侧合计 | 训练+测试 | 逐类自洽 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for label in ("baseline", "full"):
        scale = ev["scales"][
            ("baseline_c%d" if label == "baseline" else "full_c%d")
            % (cmp_["n_classes"][label])
        ]
        cd = scale["class_distribution"]
        for unit in ("question", "row"):
            cons = cd["merged_consistency"][str(unit)]
            sc = cd["merged_total_self_consistency"][str(unit)]
            lines.append(
                "| " + str(scale["label"]) + " / `" + str(unit) + "` | "
                + str(cons["n_classes"]) + " | " + str(cons["total_train"]) + " | "
                + str(cons["total_test"]) + " | " + str(cons["total_merged"]) + " | "
                + str(sc["train_plus_test"]) + " | **"
                + str(cons["all_classes_consistent"]) + "** |"
            )
    lines.append("")
    lines.append("- 断言口径：" + str(
        full_scale["class_distribution"]["merged_consistency_rule"]
    ))
    lines.append("")
    # **P2 修复**：渲染 W3 的 `task_list_consistency` 取证块（此前只在 JSON 产物里）。
    lines.append("### 4.4 W3 取证：任务清单现场一致性（`task_list_consistency`）")
    lines.append("")
    tlc = ev["task_list_consistency"]
    lines.append("- `declared` = `" + str(tlc["declared"]) + "`")
    lines.append("- `manifest_tasks` = `" + str(tlc["manifest_tasks"]) + "`")
    lines.append("- `npz_tasks` = `" + str(tlc["npz_tasks"]) + "`")
    lines.append("- `pairs_tasks` = `" + str(tlc["pairs_tasks"]) + "`")
    lines.append("- `missing_from_declared` = `" + str(tlc["missing_from_declared"])
                 + "`；`extra_in_declared` = `" + str(tlc["extra_in_declared"])
                 + "`；`consistent` = **" + str(tlc["consistent"]) + "**")
    lines.append("- `per_task_rows` = `" + str(tlc["per_task_rows"]) + "`")
    lines.append("- 允许的额外项：" + str(tlc["allowed_extra"]))
    lines.append("")
    lines.append("## 5. 确定性参照基线（**不训练**）")
    lines.append("")
    lines.append("| 口径 | 侧 | 行数 | 问题数 | predictor | micro | macro | 范围内类数 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for label in ("baseline", "full"):
        scale = ev["scales"][
            ("baseline_c%d" if label == "baseline" else "full_c%d")
            % (cmp_["n_classes"][label])
        ]
        for side in ("train", "test"):
            blk = scale["deterministic_baselines"][str(side)]
            for pred in ("constant_majority", "row_argmax"):
                row = blk[str(pred)]
                lines.append(
                    "| " + str(scale["label"]) + " | " + str(side) + " | "
                    + str(blk["n_rows"]) + " | " + str(blk["n_questions"]) + " | `"
                    + str(row["predictor"]) + "` | " + ("%.6f" % float(row["micro"]))
                    + " | " + ("%.6f" % float(row["macro"])) + " | "
                    + str(row["n_classes_in_scope"]) + " |"
                )
    lines.append("")
    lines.append("## 6. 解题类任务拆分")
    lines.append("")
    ts = ev["task_split"]
    lines.append("- 依据：" + str(ts["rule"]))
    lines.append("- 解题任务集 = `" + str(ts["solve_tasks"]) + "`；匹配任务集 = `"
                 + str(ts["match_tasks"]) + "`")
    lines.append("")
    lines.append("### 6.1 拆出前后类数与样本量变化（C=298 全量）")
    lines.append("")
    lines.append("| 任务 | 侧 | 行数 | 问题数 | 答案类数 | 正样本行数 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for task in ts["solve_tasks"] + ts["match_tasks"]:
        for side in ("train", "test"):
            row = ts["per_task_evidence_full"][str(task)][str(side)]
            lines.append(
                "| `" + str(task) + "` | " + str(side) + " | " + str(row["n_rows"]) + " | "
                + str(row["n_questions"]) + " | " + str(row["n_answer_classes"]) + " | "
                + str(row["n_positive_rows"]) + " |"
            )
    lines.append("")
    lines.append("### 6.2 分组汇总（解题任务集 / 匹配任务集）")
    lines.append("")
    lines.append("| 分组 | 侧 | 行数 | 问题数 | 正样本行数 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for group, label in (("solve", "解题任务集"), ("match", "匹配任务集")):
        for side in ("train", "test"):
            row = full_scale["grouped_stats"][str(side)][str(group)]
            lines.append(
                "| " + str(label) + " | " + str(side) + " | " + str(row["n_rows"]) + " | "
                + str(row["n_questions"]) + " | " + str(row["n_positive_rows"]) + " |"
            )
    lines.append("")
    lines.append("### 6.3 拆分对**合入口径**指标的影响（确定性基线 macro，测试侧）")
    lines.append("")
    lines.append("| 口径 | 全部任务 | 匹配任务集 | 解题任务集 |")
    lines.append("| --- | --- | --- | --- |")
    for label in ("baseline", "full"):
        key = "full" if label == "full" else "baseline"
        imp = ts["impact_on_merged_metric"][str(key)]
        lines.append(
            "| " + str(ev["scales"][("full_c%d" if label == "full" else "baseline_c%d")
                                   % cmp_["n_classes"][label]]["label"]) + " | "
            + ("%.6f" % float(imp["baselines_all"]["macro"])) + " | "
            + ("%.6f" % float(imp["baselines_match"]["macro"])) + " | "
            + ("%.6f" % float(imp["baselines_solve"]["macro"])) + " |"
        )
    lines.append("")
    lines.append("- " + str(ts["impact_on_merged_metric"]["rule"]))
    lines.append("")
    lines.append("## 7. 确定性证据")
    lines.append("")
    det = rep.get("determinism")
    if det:
        lines.append("- 双跑目录：`" + str(det["a_dir"]) + "` vs `" + str(det["b_dir"]) + "`")
        lines.append("- 共有文件 " + str(det["n_common_files"]) + "，逐字节相同 = **"
                     + str(det["determinism_ok"]) + "**，不相等文件 = `"
                     + str(det["not_equal_files"]) + "`")
        lines.append("- 排除项：`" + str(det["excluded_files"]) + "`")
        lines.append("- 判定语义：" + str(det["determinism_ok_semantics"]))
        lines.append("")
        lines.append("| 文件 | a 字节 | b 字节 | 逐字节相同 |")
        lines.append("| --- | --- | --- | --- |")
        for item in det["files"]:
            lines.append(
                "| `" + str(item["name"]) + "` | " + str(item["a_bytes"]) + " | "
                + str(item["b_bytes"]) + " | " + str(item["bitwise_equal"]) + " |"
            )
    else:
        lines.append("- 本次 `report` 未携带双跑证据（`eval --repro-dir` 才产出）")
    lines.append("")
    lines.append("## 8. 现状基线（C=10 既有口径）的登记")
    lines.append("")
    bl = ev["baseline_registered_legacy"]
    lines.append("- 来源：" + str(bl["source"]))
    for key in sorted(bl.keys()):
        if str(key) in ("source", "rule"):
            continue
        lines.append("  - `" + str(key) + "` = " + str(bl[str(key)]))
    lines.append("- " + str(bl["rule"]))
    lines.append("")
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    """渲染 ``qa_scale_report.{json,md}``（数据全部来自已落盘产物）。"""
    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    probe = _read_json(os.path.join(out_dir, "qa_scale_probe.json"))
    ev = _read_json(os.path.join(out_dir, "qa_scale_eval.json"))
    rep: Dict[str, Any] = {
        "version": str(ev["version"]),
        "probe": probe,
        "eval": ev,
        "artifact_fingerprints": {},
    }
    det_path = os.path.join(out_dir, DETERMINISM_EVIDENCE_FILE)
    if os.path.isfile(det_path):
        rep["determinism"] = _read_json(det_path)
    markdown = render_markdown(rep)
    md_sha, md_size = _write_text(os.path.join(out_dir, "qa_scale_report.md"), markdown)
    fps: Dict[str, Any] = {}
    for name in ("qa_scale_probe.json", "qa_scale_eval.json"):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            fps[str(name)] = {
                "bytes": int(os.path.getsize(p)),
                "sha256": _sha256_file(p),
            }
    if os.path.isfile(det_path):
        fps[str(DETERMINISM_EVIDENCE_FILE)] = {
            "bytes": int(os.path.getsize(det_path)),
            "sha256": _sha256_file(det_path),
        }
    fps["qa_scale_report.md"] = {"bytes": int(md_size), "sha256": str(md_sha)}
    rep["artifact_fingerprints"] = fps
    json_sha, json_size = _write_json(os.path.join(out_dir, "qa_scale_report.json"), rep)
    _log(
        "[report] qa_scale_report.json = " + str(json_size) + " B " + json_sha[:16]
        + "；qa_scale_report.md = " + str(md_size) + " B " + md_sha[:16]
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_qa_scale(args: argparse.Namespace) -> int:
    """``qa_scale`` 子命令分派（错误边界与 :mod:`merge_table_run` 同口径）。"""
    if int(getattr(sys.flags, "optimize", 0)) != 0:
        _log(
            "[qa_scale] 拒绝在优化模式下运行：sys.flags.optimize = "
            + str(int(getattr(sys.flags, "optimize", 0)))
            + "。本模块的前置校验与边界证据是硬约束，优化模式会丢弃 __debug__ 分支与"
            "docstring，使口径不可审计；请用 python（不带 -O）重跑。"
        )
        return 1
    sub = str(getattr(args, "qa_scale_cmd", "") or "probe")
    handlers = {"probe": cmd_probe, "eval": cmd_eval, "report": cmd_report}
    if sub not in handlers:
        _log("[qa_scale] 未知子命令 " + repr(sub) + "；可用 = " + str(sorted(handlers)))
        return 2
    try:
        return int(handlers[sub](args))
    except UnicodeEncodeError as exc:
        _log(
            "[qa_scale] 输出编码错误：" + type(exc).__name__ + ": " + str(exc)
            + "。这不是参数错误；可用 PYTHONIOENCODING=utf-8 重跑。"
        )
        return 1
    except FileNotFoundError as exc:
        _log("[qa_scale] 输入产物缺失：" + str(exc))
        return 1
    except (ValueError, KeyError) as exc:
        _log("[qa_scale] 非法参数 / 契约校验失败：" + str(exc))
        return 2
    except RuntimeError as exc:
        _log("[qa_scale] 运行失败（依赖 / 产物不可用）：" + str(exc))
        return 1


class _Tee:
    """把 stdout 同时写到控制台与 **UTF-8 无 BOM** 日志文件。"""

    def __init__(self, primary: Any, handle: Any) -> None:
        self._primary = primary
        self._handle = handle

    def write(self, text: str) -> int:
        """两侧同时写。"""
        self._primary.write(text)
        self._handle.write(text)
        return len(text)

    def flush(self) -> None:
        """两侧同时 flush。"""
        self._primary.flush()
        self._handle.flush()

    def __getattr__(self, item: str) -> Any:
        """其余属性透传。"""
        return getattr(self._primary, item)


def _add_qa_scale_arguments(p: argparse.ArgumentParser) -> None:
    """把 ``qa_scale`` 的参数表挂到给定子解析器上（**唯一**定义点）。"""
    p.add_argument("--product-dir", default="", help="n3d_qa 冻结产物目录（只读）")
    p.add_argument("--out-dir", default="", help="产物目录（默认 checkpoints/qa_learn/_verify/qa_scale）")
    p.add_argument("--log-file", default="", help="同时写 UTF-8(无 BOM) 日志文件")
    p.add_argument("--seed", type=int, default=42, help="留出抽稀 seed（单 seed）")
    p.add_argument("--holdout-pct", type=int, default=20,
                   help="测试集留出比例（百分比整数，0..100；默认 20）")
    p.add_argument("--baseline-classes", type=int, default=10,
                   help="基线口径的类别数上限（默认 10 = 现状 C=10）")
    p.add_argument("--full-classes", type=int, default=0,
                   help="全量口径的类别数（0 = 用 manifest 声明的答案表类数；非 0 时必须与之一致）")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    import n3d_qa_learn.qa_scale as QS

    parser = argparse.ArgumentParser(
        prog="n3d_qa_learn.qa_scale_run",
        description=(
            "轮 1：QA 侧扩 C（C=10 -> C=298）+ 按比例留出（默认 20%）+ 解题类任务拆分；"
            "不训练、只出数据口径与判据适用性。产物一律写 "
            + QS.DEFAULT_OUT_DIR.replace("\\", "/") + " 之下。"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    helps = {
        "probe": "数据取证：字段名 / 任务清单 / 答案表规模 / 解题类拆分依据（不评测）",
        "eval": "扩 C 对照评测：规模 / 分布 / 覆盖 / 判据适用性 / 确定性参照基线（不训练）",
        "report": "渲染 qa_scale_report.{json,md}",
    }
    for name in QA_SCALE_COMMANDS:
        p = sub.add_parser(name, help=helps[str(name)])
        _add_qa_scale_arguments(p)
        if str(name) == "eval":
            p.add_argument("--repro-dir", default="",
                           help="对照产物目录：给定时产出双跑逐字节一致证据")
        p.set_defaults(func=cmd_qa_scale, qa_scale_cmd=str(name))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口（``python -m n3d_qa_learn.qa_scale_run <cmd>``）。"""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    log_file = str(getattr(args, "log_file", "") or "")
    handle = None
    original = sys.stdout
    if log_file:
        parent = os.path.dirname(os.path.abspath(log_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        handle = open(log_file, "w", encoding="utf-8", newline="\n")
        sys.stdout = _Tee(original, handle)
    try:
        return int(args.func(args))
    finally:
        if handle is not None:
            sys.stdout = original
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
