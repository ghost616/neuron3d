"""轮 8：答案表**同义归并**的零训练对照 CLI（``probe`` / ``eval`` / ``report``）。

驱动方式：``python -m n3d_qa_learn.answer_merge_run {probe|eval|report}``

子命令与产物（**每个子命令只负责一件产物**，便于逐文件对账）
------------------------------------------------------------
* ``probe`` —— 由冻结答案表构建归并计划、跑 **R3 误并反例的现场断言**、落
  ``merge_map.jsonl``（逐类映射 + 逐簇成员与规则依据）。
* ``eval`` —— **零训练对照**：同 seed / 同 20% 按比例留出 / 同 ``(-count, key)`` 排序键，
  对比归并前（C=298）与归并后（C=N'）的规模、每类样本数分布、测试侧覆盖、以及
  **确定性参照基线**（``constant_majority`` 与 ``row_argmax``）；给出**结构决策门**，
  落 ``before_after.json``。
* ``report`` —— 渲染 ``audit.md``（人工复核表，含 **high-risk 单独章节**）；给了
  ``--repro-dir`` 时额外落 ``determinism_check.json``（双跑逐字节一致证据）。

**本轮不训练任何模型、不加载任何模型、不产出 zip、不落盘裸特征矩阵、无挂钟字段。**

产物纪律（**硬约束**）
---------------------
* 只写 ``checkpoints/qa_learn/_verify/answer_merge/``（``--out-dir`` 也只允许写在该根之下）；
* 既有产物一律不动（``_verify/qa_scale`` 的 9 件、顶层 ``qa_*.pt.zip`` 等）；
* 上游 ``n3d_qa`` 冻结产物**只读**。

证据口径（**不自指、不含时间戳**）
----------------------------------
``determinism_check.json`` 比对的是**其它三个产物**（``merge_map.jsonl`` /
``before_after.json`` / ``audit.md``）的 SHA256 与字节数，**它自身**被排除在比对集合之外
⇒ 不存在自指哈希回路；全部产物不含挂钟字段 ⇒ **同路径同参重跑逐字节一致**。
唯一的路径依赖是证据文件里的 ``a_dir`` / ``b_dir`` 原样串（与轮 1 同源、已登记）。

控制台编码纪律与 :mod:`n3d_qa_learn.qa_scale_run` 同源：日志字面量只用 ASCII + 汉字，
``_log`` 做 ``backslashreplace`` 兜底；本模块**不 import 任何上游 backend**。
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
ANSWER_MERGE_COMMANDS: Tuple[str, ...] = ("probe", "eval", "report")

#: 产物文件名（**唯一来源**）。
MERGE_MAP_FILE: str = "merge_map.jsonl"
AUDIT_FILE: str = "audit.md"
BEFORE_AFTER_FILE: str = "before_after.json"
DETERMINISM_EVIDENCE_FILE: str = "determinism_check.json"

#: 双跑证据**实际比对**的产物（不含证据文件自身）。
COMPARED_FILES: Tuple[str, ...] = (MERGE_MAP_FILE, BEFORE_AFTER_FILE, AUDIT_FILE)

#: 默认口径（与轮 1 逐项一致：单 seed 42 + 20% 按比例留出 + 全量类别）。
DEFAULT_SEED: int = 42
DEFAULT_HOLDOUT_PCT: int = 20

#: 结构决策门的阈值（**进产物**，改口径必须同时改这里与文档）。
GATE_MEDIAN_MIN_SAMPLES: int = 10
GATE_SIGNIFICANT_RELATIVE_DROP: float = 0.10

#: 决策门口径文本（**唯一来源**；报告与产物同源）。
GATE_RULE: str = (
    "结构门（**只判结构可行性，不判可学性**）：① 归并后「每类样本数中位数 >= %d」"
    "（主口径单位 = 训练侧问题数，另附冻结表 count / 合并问题数 / 训练侧行数三档）；"
    "② 测试侧零样本类数**显著下降**（判据 = after < before 且相对降幅 >= %.0f%% 且"
    "绝对减少 >= 1 类）。两条同时成立才记 structural_gate_passed = True。"
    "**即使为 True 也不等于「已可学」**：归并只消标签噪声、不增加样本总量，"
    "本轮零训练、单 seed、未做任何模型评测。"
    % (GATE_MEDIAN_MIN_SAMPLES, GATE_SIGNIFICANT_RELATIVE_DROP * 100.0)
)


# ---------------------------------------------------------------------------
# 通用小工具（与 qa_scale_run 同口径）
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


def _write_jsonl(path: str, records: Sequence[Dict[str, Any]]) -> Tuple[str, int]:
    """写 JSONL（UTF-8 无 BOM；每行 ``sort_keys`` 紧凑 + LF）。"""
    lines = [json.dumps(rec, ensure_ascii=False, sort_keys=True) for rec in records]
    return _write_text(path, "\n".join(lines) + "\n")


def _read_json(path: str) -> Any:
    """读 JSON；缺文件给出**可读报错**。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "缺少输入产物 " + repr(path) + "；请先跑前置子命令（probe / eval）"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """读 JSONL；缺文件给出**可读报错**。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "缺少输入产物 " + repr(path) + "；请先跑前置子命令 probe"
        )
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


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


def out_root() -> str:
    """本模块**唯一**允许写盘的根目录（轮 9 · W3 修复：从 ``_verify`` 收窄到它自己的子目录）。"""
    import n3d_qa_learn.qa_scale as QS

    return os.path.join(str(QS.VERIFY_ROOT), "answer_merge")


def _guard_out_dir(out_dir: str) -> str:
    """守卫产物目录：**只允许**写在 ``checkpoints/qa_learn/_verify/answer_merge/`` 之下。

    **轮 9 · W3 修复**：旧守卫只要求落在 ``checkpoints/qa_learn/_verify/`` 之下 —— 于是
    ``--out-dir checkpoints/qa_learn/_verify/qa_scale`` 会被放行并**覆盖别轮的交付产物**。
    修复后根目录收窄为本模块自己的 ``answer_merge/``（**二者择一里选「收窄根目录」**，
    理由：还要能重跑本模块自己的产物，选「拒绝覆盖已存在文件」会把正常重跑也堵掉）。

    越界一律抛 ``ValueError``（CLI 侧映射为**退码 2**），且调用方**先校验再建目录**。
    """
    import n3d_qa_learn.qa_scale as QS

    root = out_root()
    text = str(out_dir) if out_dir else root
    root_abs = os.path.abspath(root)
    norm_abs = os.path.abspath(text)
    prefix = root_abs + os.sep
    if not (norm_abs == root_abs or norm_abs.startswith(prefix)):
        raise ValueError(
            "产物目录只允许写在 "
            + repr(root.replace(chr(92), "/"))
            + " 之下（轮 9 · W3 收窄根目录：不再允许写 "
            + repr(str(QS.VERIFY_ROOT).replace(chr(92), "/"))
            + " 的其它子目录，避免覆盖别轮交付产物），当前 = "
            + repr(norm_abs.replace(chr(92), "/"))
        )
    return text


def _guard_log_file(log_file: str) -> str:
    """守卫日志文件（轮 9 · W3 修复：旧版**完全没有守卫**，可写到任意路径）。

    规则与产物目录**完全同等**：非空时必须落在 :func:`out_root` 之下；越界抛 ``ValueError``
    （CLI 侧 **退码 2**，且**不创建任何目录**）。
    """
    text = str(log_file)
    if not text:
        return text
    root_abs = os.path.abspath(out_root())
    norm_abs = os.path.abspath(text)
    prefix = root_abs + os.sep
    if not norm_abs.startswith(prefix):
        raise ValueError(
            "日志文件只允许写在 "
            + repr(out_root().replace(chr(92), "/"))
            + " 之下（轮 9 · W3：日志与产物同等守卫），当前 = "
            + repr(norm_abs.replace(chr(92), "/"))
        )
    return text


def guard_evidence() -> Dict[str, Any]:
    """**W3 的现场取证块**：逐路径给出「旧守卫会放行 / 新守卫拒绝」的对照（纯计算、不建目录）。

    为什么这样取证：不能真的用旧代码去写 ``_verify/qa_scale``（会覆盖别轮交付产物）。
    故这里复刻旧判据（``startswith(VERIFY_ROOT)``）与新判据（``startswith(out_root())``），
    对同一批候选路径逐条求值 —— 两者在每个越界用例上的**取值不同**即证明收窄生效。
    """
    import n3d_qa_learn.qa_scale as QS

    verify_root = os.path.abspath(str(QS.VERIFY_ROOT))
    root = os.path.abspath(out_root())
    candidates = [
        os.path.join(root),
        os.path.join(root, "_repro"),
        os.path.join(verify_root, "qa_scale"),
        os.path.join(verify_root, "step2"),
        os.path.join(verify_root, "qa_scoped"),
        os.path.join(str(QS.DEFAULT_OUT_DIR)),
        os.path.join("checkpoints", "qa_learn"),
    ]
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        norm = os.path.abspath(candidate)
        old_ok = norm == verify_root or norm.startswith(verify_root + os.sep)
        new_ok = norm == root or norm.startswith(root + os.sep)
        rows.append(
            {
                "path": norm.replace("\\", "/"),
                "old_guard_would_accept": bool(old_ok),
                "new_guard_accepts": bool(new_ok),
                "changed": bool(old_ok != new_ok),
            }
        )
    return {
        "out_root": root.replace("\\", "/"),
        "old_root": verify_root.replace("\\", "/"),
        "cases": rows,
        "n_changed": int(sum(1 for r in rows if bool(r["changed"]))),
        "rule": (
            "W3 收窄根目录：旧守卫（`_verify/` 任意子目录）会放行 `_verify/qa_scale` 等**别轮交付目录**，"
            "新守卫（`_verify/answer_merge/`）一律拒绝；`changed=True` 的用例即证据。"
            "另外 `--log-file` 旧版**完全无守卫**、现在与产物目录同等（越界退码 2、不建目录）。"
        ),
    }


def _resolve_product_dir(explicit: str) -> str:
    """定位 ``n3d_qa`` 冻结产物目录（只读）。"""
    import n3d_qa_learn.qa_scale as QS

    if explicit:
        if os.path.isfile(os.path.join(explicit, QS.ANSWER_TABLE_FILE)):
            return str(explicit)
        raise FileNotFoundError(
            "目录 " + repr(explicit) + " 下缺少 " + QS.ANSWER_TABLE_FILE
        )
    if os.path.isfile(os.path.join(QS.DEFAULT_PRODUCT_DIR, QS.ANSWER_TABLE_FILE)):
        return str(QS.DEFAULT_PRODUCT_DIR)
    raise FileNotFoundError(
        "未找到 n3d_qa 冻结产物（" + QS.ANSWER_TABLE_FILE + "）；候选 = "
        + repr(QS.DEFAULT_PRODUCT_DIR)
    )


# ---------------------------------------------------------------------------
# 现场断言（**非恒真**：故意喂错量必须抛）
# ---------------------------------------------------------------------------


def all_side_class_counts(corpus: Any, *, unit: str) -> List[int]:
    """**独立来源**的「全侧」逐类计数（**不使用 split**；轮 9 · W1 修复的核心）。

    与 :func:`n3d_qa_learn.qa_scale.side_class_counts` 的差别只有一点、但正是关键点：
    本函数把 corpus 里**全部**问题（或全部行）按类计数，**完全不看** ``split`` 的训练侧 /
    测试侧划分。因此它给出的 ``all_counts`` 与「``train`` 侧 + ``test`` 侧」是**两个独立来源**，
    逐类比对才有意义（旧实现用 ``zip(train, test)`` 相加再与自身比较 ⇒ **恒真**）。

    参数
    ----
    corpus : Any
        :class:`n3d_qa_learn.qa_scale.ScaleCorpus`。
    unit : str
        ``'question'``（按问题计：一个问题金标命中该类即 +1，同一问题只计一次）或
        ``'row'``（按正样本行计：``label == 1`` 的行逐行 +1）。

    返回
    ----
    List[int]
        长度 = ``corpus.n_classes`` 的逐类计数。

    异常
    ------
    ValueError
        ``unit`` 非法。
    """
    key = str(unit)
    if key not in ("question", "row"):
        raise ValueError("unit 只能是 'question' / 'row'，当前 %r" % (str(unit),))
    counts = [0] * int(corpus.n_classes)
    if key == "question":
        for gold in corpus.question_gold.values():
            for cls in gold:
                pos = corpus.class_pos.get(int(cls))
                if pos is not None:
                    counts[int(pos)] += 1
        return counts
    for pair in corpus.pairs:
        if int(pair.label) != 1:
            continue
        pos = corpus.class_pos.get(int(pair.candidate_index))
        if pos is not None:
            counts[int(pos)] += 1
    return counts


def assert_side_sum_equals_all(
    *,
    unit: str,
    side_a: Sequence[int],
    side_b: Sequence[int],
    all_counts: Sequence[int],
) -> Dict[str, Any]:
    """**逐类**校验 ``side_a[i] + side_b[i] == all_counts[i]``（轮 9 · W1 修复）。

    为什么这不是恒真断言（旧版的缺陷）
    ----------------------------------
    旧版把 ``merged`` 写成 ``[a + b for a, b in zip(train, test)]`` 再断言
    ``merged[i] == train[i] + test[i]`` —— 左边由右边算出，**结构上恒真**，永远抓不到
    「某一侧的统计口径错了」这类缺陷。修复后 ``all_counts`` 由
    :func:`all_side_class_counts` **独立**统计（完全不看 split），两侧与全侧来自**两条
    不同的计算路径**，任一侧错算都会在这里翻面。

    异常
    ------
    AssertionError
        三者长度不一致，或任一类不满足 ``side_a + side_b == all_counts``。
    """
    a = [int(x) for x in side_a]
    b = [int(x) for x in side_b]
    m = [int(x) for x in all_counts]
    if not (len(a) == len(b) == len(m)):
        raise AssertionError(
            "分布自洽断言失败：长度不一致（side_a=%d / side_b=%d / all=%d，unit=%s）"
            % (len(a), len(b), len(m), str(unit))
        )
    bad = [i for i in range(len(m)) if int(a[i]) + int(b[i]) != int(m[i])]
    if bad:
        head = [
            {
                "class_index": int(i),
                "side_a": int(a[i]),
                "side_b": int(b[i]),
                "side_sum": int(a[i]) + int(b[i]),
                "all_side_independent": int(m[i]),
            }
            for i in bad[:10]
        ]
        raise AssertionError(
            "分布自洽断言失败（unit=%s）：%d / %d 个类不满足 side_a + side_b == all_side；"
            "前几个 = %s" % (str(unit), len(bad), len(m), str(head))
        )
    return {
        "unit": str(unit),
        "n_classes": int(len(m)),
        "total_side_a": int(sum(a)),
        "total_side_b": int(sum(b)),
        "total_all_side_independent": int(sum(m)),
        "all_classes_consistent": True,
        "sources": {
            "side_a": "n3d_qa_learn.qa_scale.side_class_counts(side='train')",
            "side_b": "n3d_qa_learn.qa_scale.side_class_counts(side='test')",
            "all_counts": "answer_merge_run.all_side_class_counts（**不看 split**，独立统计）",
        },
        "rule": (
            "side_a[i] + side_b[i] == all_counts[i]（逐类逐项），其中 all_counts 来自**独立**"
            "统计路径（直接对全部问题 / 全部行按类计数，不使用 split）⇒ 该断言**不是**"
            "恒真：任一侧少算/多算一个类都会抛 AssertionError（现场 counterexample 见同块）。"
        ),
    }


def tautological_side_sum_check(
    *,
    unit: str,
    side_a: Sequence[int],
    side_b: Sequence[int],
    all_counts: Sequence[int],
) -> Dict[str, Any]:
    """**故意恒真**的断言（轮 9 离朱 R1 修复用）：把 ``all_counts`` 现场算成 ``side_a + side_b``
    再比对 —— 这正是旧 W1 的缺陷模式。

    它**只**用于证明 :func:`side_sum_counterexample` 具备判别力：把恒真断言注入进去，
    ``raised`` 必须为 False；否则说明该自检根本识别不了「恒真」这件事。
    """
    rebuilt = [int(a) + int(b) for a, b in zip(side_a, side_b)]
    return assert_side_sum_equals_all(
        unit=str(unit), side_a=side_a, side_b=side_b, all_counts=rebuilt
    )


def side_sum_counterexample(
    *,
    unit: str,
    side_a: Sequence[int],
    side_b: Sequence[int],
    all_counts: Sequence[int],
    assertion: Optional[Any] = None,
) -> Dict[str, Any]:
    """**非恒真的现场证明**：把某一侧的某个类改动 +1 后，断言必须抛错（轮 9 · W1 配套）。

    **轮 9 离朱 R1 修复**：新增 ``assertion`` 注入参数。旧版把断言写死，于是「若 raised=False
    说明断言恒真」这句话**无法被现场检验**（写死的断言永远会抛 ⇒ 守卫分支不可达）。
    现在可以注入 :func:`tautological_side_sum_check`（恒真断言）来证明
    「恒真 ⇒ raised=False 被识别」，从而让调用点的守卫**可判别**。

    参数
    ----
    unit / side_a / side_b / all_counts :
        与 :func:`assert_side_sum_equals_all` 同义。
    assertion : Optional[Any]
        要检验的断言函数（默认 :func:`assert_side_sum_equals_all`）。

    返回
    ----
    Dict[str, Any]
        ``applicable``（空序列时 False —— **不再是 IndexError**）、``raised``、
        ``message_head``、``corrupted_class_index``、``delta``、``assertion_name``、``rule``。
    """
    checker = assertion if assertion is not None else assert_side_sum_equals_all
    name = str(getattr(checker, "__name__", "<injected>"))
    base = [int(x) for x in side_b]
    totals = [int(x) for x in all_counts]
    if not base or not totals:
        return {
            "unit": str(unit),
            "applicable": False,
            "raised": False,
            "message_head": "",
            "corrupted_class_index": -1,
            "delta": 0,
            "assertion_name": name,
            "rule": "序列为空 ⇒ 反例不适用（登记为 applicable=False，不抛 IndexError）",
        }
    picked = next((i for i, v in enumerate(totals) if int(v) > 0), 0)
    picked = int(min(picked, len(base) - 1))
    corrupted = list(base)
    corrupted[int(picked)] = int(corrupted[int(picked)]) + 1
    raised = False
    message = ""
    try:
        checker(
            unit=str(unit), side_a=side_a, side_b=corrupted, all_counts=all_counts
        )
    except AssertionError as exc:
        raised = True
        message = str(exc)[:120]
    return {
        "unit": str(unit),
        "applicable": True,
        "corrupted_class_index": int(picked),
        "delta": 1,
        "assertion_name": name,
        "raised": bool(raised),
        "message_head": str(message),
        "rule": (
            "把 side_b 的某一个类 +1 后再跑待检验的断言：**真实断言**下必须 raised=True；"
            "**注入恒真断言**（`tautological_side_sum_check`）下必须 raised=False —— 两个方向"
            "都成立才说明本自检有判别力（只测一个方向等于自说自话）。"
        ),
    }


def assert_same_split(before: Any, after: Any) -> Dict[str, Any]:
    """断言归并前后**切分逐位相同**（否则前后对照不是同一个对照）。

    切分只依赖 ``(seed, task, question_id)``，与答案表无关 ⇒ 该断言在正确实现下恒成立；
    但一旦将来有人把「类数」混进定侧公式，这里会立刻翻面（故**不是**装饰性断言）。
    """
    problems: List[str] = []
    for name in ("train_questions", "test_questions", "train_rows", "test_rows"):
        if list(getattr(before, name)) != list(getattr(after, name)):
            problems.append(str(name))
    if problems:
        raise AssertionError(
            "归并前后切分不一致（字段 = " + str(problems) + "）；"
            "前后对照要求同一批问题同侧，拒绝在错位切分上出数"
        )
    return {
        "split_identical": True,
        "checked_fields": ["train_questions", "test_questions", "train_rows", "test_rows"],
        "n_train_questions": int(len(before.train_questions)),
        "n_test_questions": int(len(before.test_questions)),
        "seed": int(before.seed),
        "holdout_pct": int(before.holdout_pct),
        "rule": (
            "定侧 = sha256('qgroup\\0seed\\0task\\0question_id') % 100 < holdout_pct，"
            "只依赖 (seed, task, question_id) 与答案表无关，故归并前后必须逐位相同。"
        ),
    }


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    """构建归并计划 + 跑 R3 反例断言 + 落 ``merge_map.jsonl``（**不评测、不训练**）。"""
    import n3d_qa_learn.answer_merge as AM

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    product_dir = _resolve_product_dir(str(getattr(args, "product_dir", "") or ""))
    rows = AM.load_table_rows(product_dir)
    plan = AM.build_merge_plan(rows)
    probes = AM.assert_no_merge_pairs(plan)
    relaxed = AM.diagnostic_relaxed_merges(
        [(str(m["key"]), int(c.canonical_index)) for c in plan.clusters for m in c.members]
    )
    records = AM.class_records(plan) + AM.cluster_records(plan)
    sha, size = _write_jsonl(os.path.join(out_dir, MERGE_MAP_FILE), records)
    summary = plan.summary()
    _log(
        "[probe] 归并计划：%d -> %d 类（减 %d）；多成员簇 %d 个；high-risk 簇 %d 个"
        % (
            int(summary["n_original"]),
            int(summary["n_canonical"]),
            int(summary["n_reductions"]),
            int(summary["n_multi_member_clusters"]),
            int(summary["n_high_risk_clusters"]),
        )
    )
    _log(
        "[probe] 规则命中（按受影响类数）："
        + str(summary["rule_counts"])
        + "；R2 命中 %d 条；否定阻断 %d 对"
        % (
            int(summary["n_synonym_applications"]),
            int(summary["n_blocked_merges"]),
        )
    )
    _log(
        "[probe] R3 误并反例：%d 条全部保持独立（前 3 条为验收硬要求）"
        % int(probes["n_probes"])
    )
    for probe in probes["probes"]:
        _log(
            "[probe]   %r(%d) -> canonical %d  vs  %r(%d) -> canonical %d  distinct=%s"
            % (
                probe["key_a"],
                int(probe["index_a"]),
                int(probe["canonical_a"]),
                probe["key_b"],
                int(probe["index_b"]),
                int(probe["canonical_b"]),
                bool(probe["distinct"]),
            )
        )
    _log(
        "[probe] 放宽口径诊断（不参与归并）：额外并 %d 类 -> 预测 C=%d；其中违反 R3 的组 %d 个"
        % (
            int(relaxed["n_extra_merges"]),
            int(relaxed["predicted_n_canonical"]),
            int(relaxed["n_groups_violating_r3"]),
        )
    )
    _log("[probe] " + MERGE_MAP_FILE + " = " + str(size) + " B " + sha[:16])
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def _corpus_block(corpus: Any, split: Any, *, label: str) -> Dict[str, Any]:
    """单个口径（归并前 / 归并后）的规模、分布、覆盖与确定性参照基线。"""
    import n3d_qa_learn.qa_scale as QS

    q_train = QS.side_class_counts(corpus, split, side="train", unit="question")
    q_test = QS.side_class_counts(corpus, split, side="test", unit="question")
    r_train = QS.side_class_counts(corpus, split, side="train", unit="row")
    r_test = QS.side_class_counts(corpus, split, side="test", unit="row")
    # **轮 9 · W1 修复**：全侧计数改为**独立统计**（不看 split），不再由 zip(train, test) 相加。
    merged_q = all_side_class_counts(corpus, unit="question")
    merged_r = all_side_class_counts(corpus, unit="row")
    consistency = {
        "question": assert_side_sum_equals_all(
            unit="question", side_a=q_train, side_b=q_test, all_counts=merged_q
        ),
        "row": assert_side_sum_equals_all(
            unit="row", side_a=r_train, side_b=r_test, all_counts=merged_r
        ),
    }
    # **轮 9 · W1 修复（含离朱 R1 复核）**：反例分**两个方向**：
    #   ① 真实断言 ⇒ 必须 raised=True（证明断言会翻面）；
    #   ② 注入**恒真断言** ⇒ 必须 raised=False（证明这个自检**能识别**恒真，
    #      否则调用点的守卫就是不可达死代码）。
    consistency["question"]["counterexample"] = side_sum_counterexample(
        unit="question", side_a=q_train, side_b=q_test, all_counts=merged_q
    )
    consistency["question"]["counterexample"]["vacuous_assertion_probe"] = (
        side_sum_counterexample(
            unit="question",
            side_a=q_train,
            side_b=q_test,
            all_counts=merged_q,
            assertion=tautological_side_sum_check,
        )
    )
    consistency["row"]["counterexample"] = side_sum_counterexample(
        unit="row", side_a=r_train, side_b=r_test, all_counts=merged_r
    )
    consistency["row"]["counterexample"]["vacuous_assertion_probe"] = (
        side_sum_counterexample(
            unit="row",
            side_a=r_train,
            side_b=r_test,
            all_counts=merged_r,
            assertion=tautological_side_sum_check,
        )
    )
    for unit_key in ("question", "row"):
        cell = consistency[unit_key]["counterexample"]
        if not bool(cell["applicable"]) or not bool(cell["raised"]):
            raise AssertionError(
                "W1 自检失败（" + str(unit_key) + "）：把一侧某类 +1 后真实断言**没有**抛错 "
                "⇒ 断言恒真，拒绝继续"
            )
        vac = cell["vacuous_assertion_probe"]
        if not bool(vac["applicable"]) or bool(vac["raised"]):
            raise AssertionError(
                "W1 自检失败（" + str(unit_key) + "）：注入**恒真断言**后反例仍抛错/不适用 "
                "⇒ 该自检无法识别恒真（守卫不可达），拒绝继续"
            )
        cell["discriminating"] = True
        cell["discriminating_rule"] = (
            "两个方向都成立才记 discriminating=True：真实断言下 raised=True，"
            "恒真断言（`tautological_side_sum_check`）下 raised=False。"
            "只测一个方向等于自说自话（离朱 R1 指出的旧问题）。"
        )
    baseline_test = QS.deterministic_baseline_metrics(corpus, split, side="test")
    baseline_train = QS.deterministic_baseline_metrics(corpus, split, side="train")
    return {
        "label": str(label),
        "n_classes": int(corpus.n_classes),
        "n_pairs": int(corpus.n_pairs),
        "n_questions": int(corpus.n_questions),
        "answer_keys_head": [str(x) for x in corpus.answer_keys[:5]],
        "split": {
            **split.summary(),
            "actual_test_question_ratio": (
                float(len(split.test_questions)) / float(corpus.n_questions)
            ),
            "actual_test_row_ratio": float(len(split.test_rows)) / float(corpus.n_pairs),
            "target_ratio": float(split.holdout_pct) / 100.0,
            "seed": int(split.seed),
            "holdout_pct": int(split.holdout_pct),
        },
        "class_size": {
            "unit_note": (
                "四个单位分开报：`frozen_table_count` = 冻结表 count（正样本条数）；"
                "`merged_by_question` = **全侧**问题数（由 `all_side_class_counts` 独立统计，"
                "**不看 split**；轮 9 · W1 修复后不再由 train+test 相加得出）；"
                "`train_by_question` = 训练侧问题数（**决策门主口径**）；"
                "`train_by_row` = 训练侧正样本行数。四者不可互换。"
            ),
            "frozen_table_count": QS.class_size_distribution(corpus.class_row_counts),
            "merged_by_question": QS.class_size_distribution(merged_q),
            "train_by_question": QS.class_size_distribution(q_train),
            "test_by_question": QS.class_size_distribution(q_test),
            "merged_by_row": QS.class_size_distribution(merged_r),
            "train_by_row": QS.class_size_distribution(r_train),
            "test_by_row": QS.class_size_distribution(r_test),
            "train_plus_test_consistency": consistency,
        },
        "class_coverage": {
            "train_by_question": QS.class_coverage(q_train),
            "test_by_question": QS.class_coverage(q_test),
            "train_by_row": QS.class_coverage(r_train),
            "test_by_row": QS.class_coverage(r_test),
        },
        "deterministic_baselines": {
            "test": baseline_test,
            "train": baseline_train,
            "rule": (
                "**零训练参照基线**（不训练、不用模型）：`constant_majority` = 一律答本侧"
                "正样本行数最多的类；`row_argmax` = 一律答本次统计行集合里正样本行数最多的类。"
                "micro = 行级命中率；macro = 逐类准确率的算术平均（分母 = 本范围内实际"
                "出现的类数）。两者都**不得**当成模型能力。"
            ),
        },
    }


def _median_of(block: Dict[str, Any], unit: str) -> float:
    """从 ``class_size`` 块取某单位的中位数（**唯一取值口径**）。"""
    return float(block["class_size"][str(unit)]["median"])


def _decision_gate(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """**结构决策门**：中位数是否 >= 10 + 测试零样本类是否显著下降。

    **只作结构判断**：本函数不训练、不评估任何模型，结论只说明「数据形状是否还挡着
    这件事」，**不得**被读成「已可学」。
    """
    units = (
        "frozen_table_count",
        "merged_by_question",
        "train_by_question",
        "train_by_row",
    )
    medians: Dict[str, Any] = {}
    for unit in units:
        med_before = _median_of(before, unit)
        med_after = _median_of(after, unit)
        medians[str(unit)] = {
            "before": float(med_before),
            "after": float(med_after),
            "ge_%d_before" % int(GATE_MEDIAN_MIN_SAMPLES): bool(
                med_before >= float(GATE_MEDIAN_MIN_SAMPLES)
            ),
            "ge_%d_after" % int(GATE_MEDIAN_MIN_SAMPLES): bool(
                med_after >= float(GATE_MEDIAN_MIN_SAMPLES)
            ),
        }
    primary_unit = "train_by_question"
    primary = medians[str(primary_unit)]
    primary_key = "ge_%d_after" % int(GATE_MEDIAN_MIN_SAMPLES)
    zero_before = int(
        before["class_coverage"]["test_by_question"]["n_classes_without_samples"]
    )
    zero_after = int(
        after["class_coverage"]["test_by_question"]["n_classes_without_samples"]
    )
    absolute_drop = int(zero_before - zero_after)
    relative_drop = (
        float(absolute_drop) / float(zero_before) if int(zero_before) > 0 else 0.0
    )
    significant = bool(
        zero_after < zero_before
        and float(relative_drop) >= float(GATE_SIGNIFICANT_RELATIVE_DROP)
        and int(absolute_drop) >= 1
    )
    median_ok = bool(primary[primary_key])
    # 结构改善的**可见量**（只在训练侧问题数单位上算；不改变门的判定）
    hist_before = before["class_size"]["train_by_question"]["histogram"]
    hist_after = after["class_size"]["train_by_question"]["histogram"]
    small_class_shifts = {
        str(name): {
            "before": int(hist_before.get(str(name), 0)),
            "after": int(hist_after.get(str(name), 0)),
            "delta": int(hist_after.get(str(name), 0)) - int(hist_before.get(str(name), 0)),
        }
        for name in sorted(set(hist_before.keys()) | set(hist_after.keys()))
    }
    le_1_before = int(hist_before.get("1", 0))
    le_1_after = int(hist_after.get("1", 0))
    le_4_before = int(hist_before.get("1", 0)) + int(hist_before.get("2-4", 0))
    le_4_after = int(hist_after.get("1", 0)) + int(hist_after.get("2-4", 0))
    structurally_improved = bool(
        le_1_after < le_1_before or le_4_after < le_4_before
    )
    if median_ok and significant:
        verdict = (
            "结构门通过（中位数 >= %d 且测试零样本类显著下降）—— **仍不等于可学**："
            "归并只消标签噪声、不增加样本总量，本轮零训练、单 seed。"
            % int(GATE_MEDIAN_MIN_SAMPLES)
        )
    else:
        median_txt = (
            "每类样本数中位数仍 < %d（主口径单位 = %s，归并后 = %.1f）"
            % (
                int(GATE_MEDIAN_MIN_SAMPLES),
                str(primary_unit),
                float(primary["after"]),
            )
            if not median_ok
            else "每类样本数中位数已达标"
        )
        zero_txt = (
            "测试零样本类未显著下降（%d -> %d）" % (zero_before, zero_after)
            if not significant
            else "测试零样本类显著下降（%d -> %d）" % (zero_before, zero_after)
        )
        verdict = (
            "结构门未通过："
            + median_txt
            + "；"
            + zero_txt
            + "。**结论只到结构层**：QA 侧仍不可学需靠上游放量（P2），不是再改模型。"
        )
    return {
        "gate_rule": GATE_RULE,
        "median_min_samples": int(GATE_MEDIAN_MIN_SAMPLES),
        "median_by_unit": medians,
        "median_primary_unit": str(primary_unit),
        "median_primary": {
            "before": float(primary["before"]),
            "after": float(primary["after"]),
            primary_key: bool(median_ok),
        },
        "zero_test_classes": {
            "unit": "question",
            "scope": "test_by_question",
            "before": int(zero_before),
            "after": int(zero_after),
            "absolute_drop": int(absolute_drop),
            "relative_drop": float(relative_drop),
            "significant_threshold_relative_drop": float(
                GATE_SIGNIFICANT_RELATIVE_DROP
            ),
            "significant": bool(significant),
            "rule": (
                "显著下降 ⇔ after < before 且相对降幅 >= %.0f%% 且绝对减少 >= 1 类；"
                "三条件同时成立才记 significant=True（禁止只写「下降」）。"
                % (GATE_SIGNIFICANT_RELATIVE_DROP * 100.0)
            ),
        },
        "structural_gate_passed": bool(median_ok and significant),
        "small_class_shifts": {
            "unit": "train_by_question",
            "bins": small_class_shifts,
            "n_classes_with_1_train_question": {
                "before": int(le_1_before),
                "after": int(le_1_after),
                "delta": int(le_1_after - le_1_before),
            },
            "n_classes_with_at_most_4_train_questions": {
                "before": int(le_4_before),
                "after": int(le_4_after),
                "delta": int(le_4_after - le_4_before),
            },
            "structurally_improved": bool(structurally_improved),
            "note": (
                "**改善可见但不足以过门**：这些量只说明归并把最小的类「往上挪了一档」，"
                "不改变「每类中位数仍 < %d」这个结构事实。" % int(GATE_MEDIAN_MIN_SAMPLES)
            ),
        },
        "verdict": verdict,
        "not_claimed": [
            "可学性（本轮零训练、无任何模型指标）",
            "模型能力（未训练、未评测）",
            "跨 seed 稳定性（单 seed 42）",
        ],
    }


def cmd_eval(args: argparse.Namespace) -> int:
    """**零训练对照**：归并前后同切分对比 + 确定性参照基线 + 结构决策门。"""
    import n3d_qa_learn.answer_merge as AM
    import n3d_qa_learn.qa_scale as QS

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    product_dir = _resolve_product_dir(str(getattr(args, "product_dir", "") or ""))
    seed = int(getattr(args, "seed", DEFAULT_SEED) or DEFAULT_SEED)
    holdout_pct = int(getattr(args, "holdout_pct", DEFAULT_HOLDOUT_PCT))
    max_classes = int(getattr(args, "max_classes", 0) or 0)
    rows = AM.load_table_rows(product_dir)
    plan = AM.build_merge_plan(rows)
    probes = AM.assert_no_merge_pairs(plan)
    # 与盘上 ``merge_map.jsonl`` 对账（存在才校验）：证明 eval 用的映射与 probe 落盘的一致
    merge_map_path = os.path.join(out_dir, MERGE_MAP_FILE)
    consistency: Dict[str, Any] = {
        "merge_map_file": MERGE_MAP_FILE,
        "applicable": False,
        "rule": (
            "eval 自行重建归并计划（确定性），并与 probe 落盘的 merge_map.jsonl 逐类对账；"
            "文件不存在时 applicable=False（不静默通过）。"
        ),
    }
    if os.path.isfile(merge_map_path):
        disk = {
            int(rec["answer_index"]): int(rec["canonical_index"])
            for rec in _read_jsonl(merge_map_path)
            if str(rec.get("record_type", "")) == "class"
        }
        plan_map = plan.as_mapping()
        mismatched = [
            int(i)
            for i in sorted(set(disk.keys()) | set(plan_map.keys()))
            if int(disk.get(i, -1)) != int(plan_map.get(i, -2))
        ]
        if mismatched:
            raise AssertionError(
                "eval 重建的归并映射与盘上 merge_map.jsonl 不一致（前几个类 = "
                + str(mismatched[:10])
                + "）；拒绝在自相矛盾的映射上出数"
            )
        consistency.update(
            {
                "applicable": True,
                "n_classes_checked": int(len(plan_map)),
                "mismatched_classes": [],
                "fingerprint": str(plan.fingerprint()),
            }
        )
    corpus_before = QS.build_scale_corpus(product_dir, max_classes=int(max_classes))
    corpus_after = QS.build_scale_corpus(
        product_dir,
        max_classes=int(max_classes),
        merge_map=plan.as_mapping(),
        merge_labels=plan.canonical_labels(),
    )
    split_before = QS.make_scale_split(
        corpus_before, holdout_pct=int(holdout_pct), seed=int(seed)
    )
    split_after = QS.make_scale_split(
        corpus_after, holdout_pct=int(holdout_pct), seed=int(seed)
    )
    split_check = assert_same_split(split_before, split_after)
    block_before = _corpus_block(
        corpus_before, split_before, label="before_merge_c%d" % int(corpus_before.n_classes)
    )
    block_after = _corpus_block(
        corpus_after, split_after, label="after_merge_c%d" % int(corpus_after.n_classes)
    )
    comparison = {
        "rule": (
            "两套口径**使用同一 seed、同一 20%% 按比例留出、同一 (-count, key) 排序键**，"
            "唯一变量是**是否归并**（C=298 vs C=N'）⇒ 差异可直接归因于归并。"
        ),
        "n_classes": {
            "before": int(block_before["n_classes"]),
            "after": int(block_after["n_classes"]),
            "reduction": int(block_before["n_classes"] - block_after["n_classes"]),
        },
        "n_questions": {
            "before": int(block_before["n_questions"]),
            "after": int(block_after["n_questions"]),
            "delta": int(block_after["n_questions"] - block_before["n_questions"]),
        },
        "n_pairs": {
            "before": int(block_before["n_pairs"]),
            "after": int(block_after["n_pairs"]),
            "delta": int(block_after["n_pairs"] - block_before["n_pairs"]),
        },
        "n_train_questions": {
            "before": int(block_before["split"]["n_train_questions"]),
            "after": int(block_after["split"]["n_train_questions"]),
            "delta": int(
                block_after["split"]["n_train_questions"]
                - block_before["split"]["n_train_questions"]
            ),
        },
        "n_test_questions": {
            "before": int(block_before["split"]["n_test_questions"]),
            "after": int(block_after["split"]["n_test_questions"]),
            "delta": int(
                block_after["split"]["n_test_questions"]
                - block_before["split"]["n_test_questions"]
            ),
        },
        "n_train_rows": {
            "before": int(block_before["split"]["n_train_rows"]),
            "after": int(block_after["split"]["n_train_rows"]),
            "delta": int(
                block_after["split"]["n_train_rows"]
                - block_before["split"]["n_train_rows"]
            ),
        },
        "n_test_rows": {
            "before": int(block_before["split"]["n_test_rows"]),
            "after": int(block_after["split"]["n_test_rows"]),
            "delta": int(
                block_after["split"]["n_test_rows"] - block_before["split"]["n_test_rows"]
            ),
        },
        "baseline_macro_test": {
            "before": float(
                block_before["deterministic_baselines"]["test"]["constant_majority"]["macro"]
            ),
            "after": float(
                block_after["deterministic_baselines"]["test"]["constant_majority"]["macro"]
            ),
            "note": (
                "确定性参照基线的 macro。C 变小 ⇒ 均匀基线 1/C 变大 ⇒ 该值**必然**上升，"
                "**不得**读成「模型变好」。"
            ),
        },
    }
    gate = _decision_gate(block_before, block_after)
    merge_src: Dict[str, Any] = dict(corpus_after.source.get("merge", {}))
    doc: Dict[str, Any] = {
        "version": str(AM.ANSWER_MERGE_VERSION),
        "effective_scope": {
            "seed": int(seed),
            "holdout_pct": int(holdout_pct),
            "max_classes": int(max_classes),
            "single_seed": True,
            "statement": (
                "生效口径 = 全量类别（max_classes=%d）+ %d%% 按比例留出 + 单 seed(%d)；"
                "**零训练**（不加载任何模型）"
                % (int(max_classes), int(holdout_pct), int(seed))
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
            "manifest.json": QS.sha256_file(
                os.path.join(product_dir, QS.MANIFEST_FILE)
            ),
            "n3d_qa_all.npz": QS.sha256_file(
                os.path.join(product_dir, QS.ALL_NPZ_FILE)
            ),
        },
        "merge_summary": plan.summary(),
        "merge_map_consistency": consistency,
        "no_merge_probes": probes,
        "out_dir_and_log_guard_evidence": guard_evidence(),
        "split_consistency": split_check,
        "ordering_rule": (
            "canonical 下标按 canonical key 升序；若 max_classes > 0，两套口径**都用**"
            "轮 1 的同一排序键 (-count, key) 截断（before 用冻结表 count，after 用 canonical "
            "聚合 count）；本轮 max_classes=0（全量），故截断不生效。"
        ),
        "before": block_before,
        "after": block_after,
        "comparison": comparison,
        "decision_gate": gate,
        "honest_notes": [
            "归并只消标签噪声、**不增加样本总量**：问题数与行数与归并前完全相同。",
            "确定性参照基线（constant_majority / row_argmax）**不是**模型能力；本轮不训练。",
            "单 seed（42）、单留出比例（20%）：不含跨 seed 极差。",
            "归并后类数不再是 298 ⇒ 轮 1 的「C=298」结论不再直接适用（口径变更已登记）。",
            (
                "归并会把「同一问题的两个变体类金标」折叠（现场见 corpus.source.merge."
                "n_gold_pairs_collapsed）；行**不去重**，故 unit='row' 的逐类计数含重复，"
                "比较以 unit='question' 为主。"
            ),
            (
                "**上游产物既有性质（不是本轮造成的）**：冻结问答对里 (question_id, "
                "candidate) 并不唯一 —— 现场 label=0 重复行 "
                + str(int(merge_src.get("n_duplicate_rows_before_merge", 0)))
                + " 条（按 label 分解 = "
                + str(dict(merge_src.get("n_duplicate_rows_before_merge_by_label", {})))
                + "），金标侧（label=1）唯一性现场判定 gold_rows_unique_premise = "
                + str(bool(merge_src.get("gold_rows_unique_premise", False)))
                + "；故「金标对折叠数」"
                + str(int(merge_src.get("n_gold_pairs_collapsed", 0)))
                + " 可以**只**归因于归并。"
            ),
        ],
    }
    sha, size = _write_json(os.path.join(out_dir, BEFORE_AFTER_FILE), doc)
    _log(
        "[eval] 归并前 C=%d / 归并后 C=%d（减 %d）；问题 %d / 行 %d（不变）"
        % (
            int(block_before["n_classes"]),
            int(block_after["n_classes"]),
            int(block_before["n_classes"] - block_after["n_classes"]),
            int(block_before["n_questions"]),
            int(block_before["n_pairs"]),
        )
    )
    _log(
        "[eval] 训练问题 %d / 测试问题 %d / 训练行 %d / 测试行 %d（两套口径同一切分）"
        % (
            int(split_check["n_train_questions"]),
            int(split_check["n_test_questions"]),
            int(block_before["split"]["n_train_rows"]),
            int(block_before["split"]["n_test_rows"]),
        )
    )
    _log(
        "[eval] 每类样本数中位数（主口径 %s）：%.1f -> %.1f；测试零样本类 %d -> %d"
        % (
            str(gate["median_primary_unit"]),
            float(gate["median_primary"]["before"]),
            float(gate["median_primary"]["after"]),
            int(gate["zero_test_classes"]["before"]),
            int(gate["zero_test_classes"]["after"]),
        )
    )
    _log(
        "[eval] 零训练基线（测试侧 constant_majority）macro：%.6f -> %.6f"
        % (
            float(comparison["baseline_macro_test"]["before"]),
            float(comparison["baseline_macro_test"]["after"]),
        )
    )
    _log(
        "[eval] 结构门 structural_gate_passed=%s（只判结构，不判可学）"
        % str(gate["structural_gate_passed"])
    )
    _log("[eval] " + BEFORE_AFTER_FILE + " = " + str(size) + " B " + sha[:16])
    return 0


# ---------------------------------------------------------------------------
# 双跑逐字节一致证据
# ---------------------------------------------------------------------------


def determinism_block(out_dir: str, repro_dir: str) -> Dict[str, Any]:
    """双跑逐字节一致证据（**排除证据文件自身**；比对三个内容产物）。"""
    excluded = {str(DETERMINISM_EVIDENCE_FILE)}
    expected = [str(x) for x in COMPARED_FILES if str(x) not in excluded]
    files: List[Dict[str, Any]] = []
    missing: List[str] = []
    for name in expected:
        pa = os.path.join(out_dir, str(name))
        pb = os.path.join(repro_dir, str(name))
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            missing.append(str(name))
            continue
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
    a_files = sorted(
        f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))
    )
    b_files = sorted(
        f for f in os.listdir(repro_dir) if os.path.isfile(os.path.join(repro_dir, f))
    )
    a_only = [f for f in a_files if f not in b_files]
    b_only = [f for f in b_files if f not in a_files]
    not_equal = [f["name"] for f in files if not bool(f["bitwise_equal"])]

    def _norm(path: str) -> str:
        return os.path.abspath(str(path)).replace("\\", "/").lower()

    return {
        "a_dir": str(out_dir),
        "b_dir": str(repro_dir),
        "a_dir_normalized": _norm(out_dir),
        "b_dir_normalized": _norm(repro_dir),
        "paths_normalized_equal": bool(_norm(out_dir) == _norm(repro_dir)),
        "path_dependence_note": (
            "`a_dir` / `b_dir` 保留命令原样串（Windows 下 \\ 与 / 的字形差异会被保留）；"
            "另给 `*_normalized` 供机器判定。**这两个原样串会进本文件内容** ⇒ 本文件在"
            "「绝对路径不同」的运行之间不逐字节相同；同路径同参重跑才逐字节相同。"
            "这是「不落盘挂钟字段」之外的**唯一**已知非确定性来源，已显式登记。"
        ),
        "files": files,
        "n_common_files": int(len(files)),
        "n_expected_files": int(len(expected)),
        "missing_files": missing,
        "compared_files": expected,
        "n_files_before_exclusion": {"a": int(len(a_files)), "b": int(len(b_files))},
        "excluded_files": sorted(excluded),
        "a_only_files": a_only,
        "b_only_files": b_only,
        "not_equal_files": not_equal,
        "determinism_ok": bool(len(files) == len(expected) and not not_equal),
        "determinism_ok_semantics": (
            "**轮 9 · W2 修复**：determinism_ok = 「三件内容产物**齐备**"
            "（len(files) == len(compared_files)）**且**逐个逐字节相等」；缺任何一件即 False，"
            "并把缺失项登记到 `missing_files`（旧实现只要 >= 1 件可比就为 True）。"
            "证据文件自身被排除（否则 a_sha256 会是「上一次运行留在盘上的自己」）；"
            "单侧独有文件不纳入布尔值，但必须显式登记在 a_only_files / b_only_files。"
        ),
        "no_self_reference": (
            "本文件记录的是 merge_map.jsonl / before_after.json / audit.md 的哈希，"
            "这三个文件的内容**不含**本文件哈希 ⇒ 无自指循环；本文件自身亦被排除在比对集合外。"
        ),
    }


def determinism_missing_file_probe(out_dir: str, repro_dir: str) -> Dict[str, Any]:
    """**非恒真的现场证明**（轮 9 · W2 配套）：删掉 b 侧一件产物后 ``determinism_ok`` 必须为 False。

    做法：把两个目录的三件内容产物复制到**临时目录**（TEMP 下、用完即删），再删掉临时 b 侧
    的一件，跑同一个 :func:`determinism_block`。**临时路径不进任何产物**（本块只记文件名与
    布尔值）⇒ 证据文件本身仍无挂钟字段、无额外路径。
    """
    import shutil
    import tempfile

    expected = [
        str(x) for x in COMPARED_FILES if str(x) != str(DETERMINISM_EVIDENCE_FILE)
    ]
    removed = str(expected[-1]) if expected else ""
    with tempfile.TemporaryDirectory(prefix="am_det_probe_") as tmp:
        left = os.path.join(tmp, "a")
        right = os.path.join(tmp, "b")
        os.makedirs(left)
        os.makedirs(right)
        copied = 0
        for name in expected:
            for base, source_dir in ((left, out_dir), (right, repro_dir)):
                src = os.path.join(str(source_dir), str(name))
                if os.path.isfile(src):
                    shutil.copyfile(src, os.path.join(base, str(name)))
                    copied += 1
        complete_block = determinism_block(left, right)
        target = os.path.join(right, removed)
        removed_ok = os.path.isfile(target)
        if removed_ok:
            os.remove(target)
        broken_block = determinism_block(left, right)
    return {
        "removed_file": str(removed),
        "removed_in_temp_b_side": bool(removed_ok),
        "n_files_copied": int(copied),
        "determinism_ok_when_complete": bool(complete_block["determinism_ok"]),
        "determinism_ok_when_missing": bool(broken_block["determinism_ok"]),
        "missing_files_when_missing": [str(x) for x in broken_block["missing_files"]],
        "rule": (
            "同一份代码、同一批文件：齐备时 `determinism_ok=True`；删掉一件后必须 "
            "`determinism_ok=False` 且 `missing_files` 恰为被删文件（若仍为 True，说明 "
            "W2 修复无效 ⇒ 恒真）。"
        ),
        "temp_paths_not_persisted": True,
    }


# ---------------------------------------------------------------------------
# report（audit.md 渲染）
# ---------------------------------------------------------------------------


def _cluster_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 ``merge_map.jsonl`` 取簇记录（按 canonical 下标升序）。"""
    rows = [r for r in records if str(r.get("record_type", "")) == "cluster"]
    rows.sort(key=lambda r: int(r["canonical_index"]))
    return rows


def _md(text: Any) -> str:
    """纯文本上下文里的 Markdown 转义（反引号 / 竖线），避免破坏表格与代码段。"""
    return str(text).replace("`", "\\`").replace("|", "\\|")


def _code(text: Any) -> str:
    """行内代码段（内容含反引号时改用双反引号定界，避免未闭合代码段）。"""
    s = str(text)
    if "`" in s:
        return "`` " + s + " ``"
    return "`" + s + "`"


def _class_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 ``merge_map.jsonl`` 取逐类映射记录（按原始下标升序）。"""
    rows = [r for r in records if str(r.get("record_type", "")) == "class"]
    rows.sort(key=lambda r: int(r["answer_index"]))
    return rows


def render_audit(
    records: Sequence[Dict[str, Any]],
    before_after: Dict[str, Any],
) -> str:
    """渲染 ``audit.md``（**确定性**：无时间戳、无耗时、无绝对路径）。"""
    import n3d_qa_learn.answer_merge as AM
    import n3d_qa_learn.qa_scale as QS

    docs = AM.rules_documentation()
    clusters = _cluster_rows(records)
    classes = _class_rows(records)
    multi = [c for c in clusters if int(c["cluster_size"]) >= 2]
    high_risk = [c for c in clusters if bool(c["high_risk"])]
    summary = before_after["merge_summary"]
    gate = before_after["decision_gate"]
    before = before_after["before"]
    after = before_after["after"]
    cmp_ = before_after["comparison"]
    gate_median_key = "ge_%d_after" % int(gate["median_min_samples"])
    lines: List[str] = []
    lines.append("# 轮 8：答案表同义归并 · 人工复核表（audit）")
    lines.append("")
    lines.append("> 本表由 `python -m n3d_qa_learn.answer_merge_run report` 从")
    lines.append("> `merge_map.jsonl` + `before_after.json` **确定性渲染**（无时间戳、无耗时、")
    lines.append("> 无绝对路径）。规则文本的唯一来源是 `answer_merge.RULE_TEXTS` /")
    lines.append("> `SYNONYM_RULES` / `SYNONYM_EXCLUSIONS`（本表只引用，不另写一套）。")
    lines.append("")
    lines.append("## 1. 口径与产物")
    lines.append("")
    lines.append("- 数据来源（**只读**）：`" + str(before_after["product_dir"]) + "`")
    for name, sha in sorted(before_after["product_files_sha256"].items()):
        lines.append("  - `" + str(name) + "` SHA256 = `" + str(sha) + "`")
    lines.append("- 生效口径：" + str(before_after["effective_scope"]["statement"]))
    lines.append("- 产物：`merge_map.jsonl`（逐类映射 + 逐簇依据）、`audit.md`（本表）、")
    lines.append("  `before_after.json`（前后对照 + 决策门）、`determinism_check.json`（双跑证据）")
    lines.append("- **不产 zip、不落盘裸特征矩阵、无挂钟字段**；既有产物一律未动")
    lines.append("")
    lines.append("### 1.1 `merge_map.jsonl` 的两种记录")
    lines.append("")
    lines.append("| `record_type` | 字段 | 用途 |")
    lines.append("| --- | --- | --- |")
    lines.append(
        "| `class` | `answer_index` / `answer_key` / `answer_display` / `answer_count` / "
        "`canonical_index` / `canonical_key` / `cluster_size` / `rule_ids` / `high_risk` | "
        "**逐类映射**（原 index/key -> canonical），逐条可追溯 |"
    )
    lines.append(
        "| `cluster` | `cluster_id` / `canonical_key` / `canonical_display` / `total_count` / "
        "`cluster_size` / `rule_ids` / `high_risk` / `high_risk_reasons` / "
        "`display_similarity_min` / `members[]`（含每个成员的 `normalized` / `rule_ids` / "
        "`details`） | **逐簇成员与依据**（规则编号 + 逐条 detail） |"
    )
    lines.append("")
    lines.append("## 2. 规则登记（与实现同源）")
    lines.append("")
    for rule_id in sorted(docs["rule_texts"].keys()):
        lines.append("- **`" + str(rule_id) + "`**：" + _md(docs["rule_texts"][rule_id]))
    lines.append("")
    lines.append("- 装饰性标点集合 = " + _code(docs["decoration_chars"]))
    lines.append("- 运算符集合 = " + _code(docs["operator_chars"]))
    lines.append("- 否定词元 = `" + str(docs["negation_tokens"]) + "`")
    lines.append(
        "- 括号剥离最大迭代 = "
        + str(docs["max_paren_strip"])
        + "；high-risk 阈值 = 簇内成员 > "
        + str(docs["high_risk_max_cluster_size"])
        + " 或 display 相似度下界 < "
        + ("%.3f" % float(docs["high_risk_min_display_similarity"]))
    )
    lines.append("")
    lines.append("### 2.1 R2 显式同义表（每条带依据）")
    lines.append("")
    lines.append("| 条目 | aliases（归一化后） | canonical | 依据 | 本语料命中类数 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for entry in docs["synonym_rules"]:
        hit = int(summary["rule_counts"].get(str(entry["id"]), 0))
        lines.append(
            "| `" + str(entry["id"]) + "` | `" + str(entry["aliases"]) + "` | `"
            + str(entry["canonical"]) + "` | " + str(entry["basis"]) + " | "
            + str(hit) + " |"
        )
    lines.append("")
    lines.append(
        "现场 R2 命中条数 = **"
        + str(int(summary["n_synonym_applications"]))
        + "**（如实登记：本语料的 298 个类里没有任何一个归一化后等于同义表的 alias，"
        "故 R2 **只作守门**；它仍逐条落审计，以便后续语料出现 `us` / `usa` 时行为可追溯）。"
    )
    lines.append("")
    lines.append("### 2.2 未采纳的同义候选（禁止随意扩张的留痕）")
    lines.append("")
    lines.append("| 候选 alias | 若并入 | 否决理由 |")
    lines.append("| --- | --- | --- |")
    for entry in docs["synonym_exclusions"]:
        lines.append(
            "| `" + str(entry["alias"]) + "` | `" + str(entry["would_merge_into"]) + "` | "
            + str(entry["reason"]) + " |"
        )
    lines.append("")
    lines.append("## 3. 归并结果总览")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | --- |")
    lines.append("| 原始类数 | " + str(int(summary["n_original"])) + " |")
    lines.append("| 归并后类数 | **" + str(int(summary["n_canonical"])) + "** |")
    lines.append("| 减少类数 | " + str(int(summary["n_reductions"])) + " |")
    lines.append("| 减少比例 | " + ("%.4f" % float(summary["reduction_ratio"])) + " |")
    lines.append("| 多成员簇数 | " + str(int(summary["n_multi_member_clusters"])) + " |")
    lines.append(
        "| 落在多成员簇里的类数 | "
        + str(int(summary["n_classes_in_multi_member_clusters"]))
        + " |"
    )
    lines.append("| high-risk 簇数 | " + str(int(summary["n_high_risk_clusters"])) + " |")
    lines.append("| 否定差异阻断对数 | " + str(int(summary["n_blocked_merges"])) + " |")
    lines.append(
        "| 计划指纹（`MergePlan.fingerprint`） | `" + str(summary["fingerprint"]) + "` |"
    )
    lines.append("")
    lines.append("规则命中（**按受影响的原始类数**）：")
    lines.append("")
    lines.append("| 规则 | 受影响类数 |")
    lines.append("| --- | --- |")
    for rule_id, count in sorted(summary["rule_counts"].items()):
        lines.append("| `" + str(rule_id) + "` | " + str(int(count)) + " |")
    lines.append("")
    lines.append(
        "## 4. 多成员簇明细（逐簇逐成员：原 index / key / display / count / 命中规则）"
    )
    lines.append("")
    lines.append(
        "| 簇 canonical | canonical key | 簇规模 | count 合计 | high-risk | display 相似度下界 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for cluster in multi:
        lines.append(
            "| " + str(int(cluster["canonical_index"])) + " | `"
            + str(cluster["canonical_key"]) + "` | " + str(int(cluster["cluster_size"]))
            + " | " + str(int(cluster["total_count"])) + " | "
            + str(bool(cluster["high_risk"])) + " | "
            + ("%.4f" % float(cluster["display_similarity_min"])) + " |"
        )
    lines.append("")
    for cluster in multi:
        lines.append(
            "### 簇 "
            + str(int(cluster["canonical_index"]))
            + " · `"
            + str(cluster["canonical_key"])
            + "`"
        )
        lines.append("")
        lines.append(
            "- 代表 display = `"
            + str(cluster["canonical_display"])
            + "`；count 合计 = "
            + str(int(cluster["total_count"]))
            + "；规则 = `"
            + str(cluster["rule_ids"])
            + "`；high-risk = **"
            + str(bool(cluster["high_risk"]))
            + "**"
            + (
                "（原因 = `" + str(cluster["high_risk_reasons"]) + "`）"
                if cluster["high_risk_reasons"]
                else ""
            )
        )
        lines.append("")
        lines.append("| 原 index | 原 key | 原 display | count | 成员命中规则 | 归一化结果 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for member in cluster["members"]:
            lines.append(
                "| " + str(int(member["index"])) + " | `" + str(member["key"]) + "` | `"
                + str(member["display"]) + "` | " + str(int(member["count"])) + " | `"
                + str(member["rule_ids"]) + "` | `" + str(member["normalized"]) + "` |"
            )
        lines.append("")
        lines.append("逐条依据（`details`）：")
        lines.append("")
        for member in cluster["members"]:
            for detail in member["details"]:
                lines.append("  - [" + str(int(member["index"])) + "] " + _md(detail))
        lines.append("")
    lines.append("## 5. **high-risk 人工复核清单**（R4 安全阀）")
    lines.append("")
    if not high_risk:
        lines.append("（本次无 high-risk 簇）")
    else:
        lines.append(
            "阈值：簇内成员 > "
            + str(docs["high_risk_max_cluster_size"])
            + " 或簇内 display 相似度下界 < "
            + ("%.3f" % float(docs["high_risk_min_display_similarity"]))
            + "。**这些簇仍然被合并**，但必须人工复核后才能用于训练口径。"
        )
        lines.append("")
        lines.append(
            "| 簇 canonical | canonical key | 规模 | 原因 | display 相似度下界 | "
            "成员（原 index: key / display） |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for cluster in high_risk:
            members_txt = "；".join(
                str(int(m["index"])) + ": `" + str(m["key"]) + "` / `" + str(m["display"]) + "`"
                for m in cluster["members"]
            )
            lines.append(
                "| " + str(int(cluster["canonical_index"])) + " | `"
                + str(cluster["canonical_key"]) + "` | " + str(int(cluster["cluster_size"]))
                + " | `" + str(cluster["high_risk_reasons"]) + "` | "
                + ("%.4f" % float(cluster["display_similarity_min"])) + " | "
                + members_txt + " |"
            )
        lines.append("")
        lines.append("**复核要点（逐条给问题，不预判结论）**：")
        lines.append("")
        for cluster in high_risk:
            lines.append(
                "- 簇 `"
                + str(cluster["canonical_key"])
                + "`：成员 "
                + str([str(m["key"]) for m in cluster["members"]])
                + " —— 请确认它们**确实是同一实体**（而不是同一词干下的不同实体）；"
                "若有异议，该簇应从映射中拆开并重跑 `eval`。"
            )
        lines.append("")
    lines.append("## 6. R3 禁并项与**现场反例断言**")
    lines.append("")
    probes = before_after["no_merge_probes"]
    lines.append("- 口径：" + str(probes["rule"]))
    lines.append("- 允许的额外项：" + str(probes["allowed_extra"]))
    lines.append("")
    lines.append(
        "| 答案 A | index | canonical | 答案 B | index | canonical | 保持独立 | 为什么必须独立 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for probe in probes["probes"]:
        lines.append(
            "| `" + str(probe["key_a"]) + "` | " + str(int(probe["index_a"])) + " | "
            + str(int(probe["canonical_a"])) + " | `" + str(probe["key_b"]) + "` | "
            + str(int(probe["index_b"])) + " | " + str(int(probe["canonical_b"])) + " | **"
            + str(bool(probe["distinct"])) + "** | " + str(probe["reason"]) + " |"
        )
    lines.append("")
    lines.append("## 7. 保守口径下的「漏并」清单（**诊断量，不参与归并**）")
    lines.append("")
    relaxed = AM.diagnostic_relaxed_merges(
        [(str(rec["answer_key"]), int(rec["canonical_index"])) for rec in classes]
    )
    lines.append("- " + str(relaxed["rule"]))
    lines.append(
        "- 现场：若再叠加「变音符折叠 + 去空格」，会**额外并掉 "
        + str(int(relaxed["n_extra_merges"]))
        + " 类**（预测 C = "
        + str(int(relaxed["predicted_n_canonical"]))
        + "）；其中 **"
        + str(int(relaxed["n_groups_violating_r3"]))
        + " 组违反 R3**。"
    )
    lines.append(
        "- 剔除违反 R3 的组后：额外并 "
        + str(int(relaxed["n_extra_merges_excluding_r3_violations"]))
        + " 类（预测 C = "
        + str(int(relaxed["predicted_n_canonical_excluding_r3_violations"]))
        + "）。"
    )
    lines.append("")
    if relaxed["groups"]:
        lines.append("| 放宽后的键 | 会合并的当前簇 | 额外并掉的类数 | 成员 key | 违反 R3 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for group in relaxed["groups"]:
            lines.append(
                "| `" + str(group["relaxed_key"]) + "` | `"
                + str(group["canonical_indices"]) + "` | "
                + str(int(group["extra_merges"])) + " | `"
                + str(group["member_keys"]) + "` | "
                + ("**是**" if group["violates_r3"] else "否") + " |"
            )
        lines.append("")
    lines.append(
        "**为什么本轮是 "
        + str(int(summary["n_canonical"]))
        + " 而不是任务书举例的 277**：R1 明确只做「小写 / 空白 / 装饰性标点 / 包裹式括号」，"
        "**不含**变音符折叠、也不做「去全部空白」，且 R3 禁止对含数字或运算符的答案删符号 —— "
        "上表的放宽口径会把 `1,0` 与 `10` 并成一组（标 **是** 的那行），这是**不可采纳**的误并。"
        "故 "
        + str(int(summary["n_canonical"]))
        + " 是「按本轮钉死规则能安全达到的数」，差额是**刻意保留的保守性**，不是遗漏。"
    )
    lines.append("")
    lines.append("## 8. 归并前后对照（零训练）")
    lines.append("")
    lines.append("| 指标 | 归并前 | 归并后 | 差 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        "| 类别数 C | " + str(int(cmp_["n_classes"]["before"])) + " | **"
        + str(int(cmp_["n_classes"]["after"])) + "** | -"
        + str(int(cmp_["n_classes"]["reduction"])) + " |"
    )
    for key, label in (
        ("n_questions", "问题数"),
        ("n_pairs", "问答对行数"),
        ("n_train_questions", "训练问题数"),
        ("n_test_questions", "测试问题数"),
        ("n_train_rows", "训练行数"),
        ("n_test_rows", "测试行数"),
    ):
        row = cmp_[str(key)]
        lines.append(
            "| " + str(label) + " | " + str(int(row["before"])) + " | "
            + str(int(row["after"])) + " | " + str(int(row["delta"])) + " |"
        )
    lines.append("")
    lines.append("### 8.1 每类样本数分布（五个单位分开报，**不可互换**）")
    lines.append("")
    lines.append(
        "| 单位 | 口径 | min（前->后） | median（前->后） | max（前->后） | 零样本类数（前->后） |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    unit_notes = {
        "frozen_table_count": "冻结表 count（正样本条数）",
        "merged_by_question": "训练侧+测试侧问题数",
        "train_by_question": "训练侧问题数（**决策门主口径**）",
        "train_by_row": "训练侧正样本行数",
        "test_by_question": "测试侧问题数",
    }
    for unit in (
        "frozen_table_count",
        "merged_by_question",
        "train_by_question",
        "train_by_row",
        "test_by_question",
    ):
        b = before["class_size"][str(unit)]
        a = after["class_size"][str(unit)]
        lines.append(
            "| `" + str(unit) + "` | " + str(unit_notes[str(unit)]) + " | "
            + str(int(b["min"])) + " -> " + str(int(a["min"])) + " | "
            + ("%.1f" % float(b["median"])) + " -> " + ("%.1f" % float(a["median"])) + " | "
            + str(int(b["max"])) + " -> " + str(int(a["max"])) + " | "
            + str(int(b["zero_count"])) + " -> " + str(int(a["zero_count"])) + " |"
        )
    lines.append("")
    lines.append(
        "分档直方图（档位顺序 = `qa_scale.CLASS_SIZE_BINS`，单位 = 训练侧问题数）："
    )
    lines.append("")
    lines.append("| 档位 | 归并前 | 归并后 | 差 |")
    lines.append("| --- | --- | --- | --- |")
    for name in [str(b[0]) for b in QS.CLASS_SIZE_BINS]:
        before_count = int(before["class_size"]["train_by_question"]["histogram"].get(name, 0))
        after_count = int(after["class_size"]["train_by_question"]["histogram"].get(name, 0))
        lines.append(
            "| `" + str(name) + "` | " + str(before_count) + " | " + str(after_count)
            + " | " + str(after_count - before_count) + " |"
        )
    lines.append("")
    lines.append("### 8.2 类别覆盖")
    lines.append("")
    lines.append("| 单位 | 类数（前->后） | 有样本类数（前->后） | 零样本类数（前->后） |")
    lines.append("| --- | --- | --- | --- |")
    for unit in ("test_by_question", "test_by_row", "train_by_question"):
        b = before["class_coverage"][str(unit)]
        a = after["class_coverage"][str(unit)]
        lines.append(
            "| `" + str(unit) + "` | " + str(int(b["n_classes"])) + " -> "
            + str(int(a["n_classes"])) + " | "
            + str(int(b["n_classes_with_samples"])) + " -> "
            + str(int(a["n_classes_with_samples"])) + " | "
            + str(int(b["n_classes_without_samples"])) + " -> "
            + str(int(a["n_classes_without_samples"])) + " |"
        )
    lines.append("")
    lines.append("### 8.3 零训练参照基线（**不是模型能力**）")
    lines.append("")
    lines.append("| 口径 | 侧 | predictor | micro | macro | 范围内类数 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for label, block in (("归并前", before), ("归并后", after)):
        for side in ("train", "test"):
            for pred in ("constant_majority", "row_argmax"):
                row = block["deterministic_baselines"][str(side)][str(pred)]
                lines.append(
                    "| " + str(label) + " | " + str(side) + " | `" + str(row["predictor"])
                    + "` | " + ("%.6f" % float(row["micro"])) + " | "
                    + ("%.6f" % float(row["macro"])) + " | "
                    + str(int(row["n_classes_in_scope"])) + " |"
                )
    lines.append("")
    lines.append("- " + str(before["deterministic_baselines"]["rule"]))
    lines.append(
        "- 测试侧 `constant_majority` 的 macro：前 "
        + ("%.6f" % float(cmp_["baseline_macro_test"]["before"]))
        + " -> 后 "
        + ("%.6f" % float(cmp_["baseline_macro_test"]["after"]))
        + "。" + str(cmp_["baseline_macro_test"]["note"])
    )
    lines.append("")
    lines.append("## 9. 结构决策门（**只判结构，不判可学**）")
    lines.append("")
    lines.append("- " + str(gate["gate_rule"]))
    lines.append("")
    lines.append(
        "| 单位 | 中位数（前） | 中位数（后） | 后 >= "
        + str(int(gate["median_min_samples"]))
        + " |"
    )
    lines.append("| --- | --- | --- | --- |")
    for unit in sorted(gate["median_by_unit"].keys()):
        cell = gate["median_by_unit"][str(unit)]
        lines.append(
            "| `" + str(unit) + "` | " + ("%.1f" % float(cell["before"])) + " | "
            + ("%.1f" % float(cell["after"])) + " | "
            + str(bool(cell[gate_median_key])) + " |"
        )
    lines.append("")
    zero_cell = gate["zero_test_classes"]
    lines.append(
        "- 测试侧零样本类：**"
        + str(int(zero_cell["before"]))
        + " -> "
        + str(int(zero_cell["after"]))
        + "**（绝对减少 "
        + str(int(zero_cell["absolute_drop"]))
        + " 类，相对降幅 "
        + ("%.4f" % float(zero_cell["relative_drop"]))
        + "）；显著 = **"
        + str(bool(zero_cell["significant"]))
        + "**"
    )
    lines.append("- " + str(zero_cell["rule"]))
    shifts = gate["small_class_shifts"]
    lines.append(
        "- **结构改善的可见量**（单位 = 训练侧问题数）：只有 1 条训练问题的类 "
        + str(int(shifts["n_classes_with_1_train_question"]["before"]))
        + " -> "
        + str(int(shifts["n_classes_with_1_train_question"]["after"]))
        + "；训练问题 <= 4 条的类 "
        + str(int(shifts["n_classes_with_at_most_4_train_questions"]["before"]))
        + " -> "
        + str(int(shifts["n_classes_with_at_most_4_train_questions"]["after"]))
        + "（`structurally_improved` = "
        + str(bool(shifts["structurally_improved"]))
        + "）"
    )
    lines.append("- " + str(shifts["note"]))
    lines.append(
        "- `structural_gate_passed` = **"
        + str(bool(gate["structural_gate_passed"]))
        + "**（主口径单位 = `"
        + str(gate["median_primary_unit"])
        + "`）"
    )
    lines.append("- 判定：" + str(gate["verdict"]))
    lines.append("- **未声明**：" + str(gate["not_claimed"]))
    lines.append("")
    lines.append("## 10. 口径变更声明与局限（如实登记）")
    lines.append("")
    lines.append(
        "1. **口径变更**：归并后类数 = **"
        + str(int(summary["n_canonical"]))
        + "**，不再是 298 ⇒ 轮 1（`qa_scale`）的全部「C=298」结论**不再直接适用**；"
        "引用时必须以本轮 `before_after.json` 的 `after` 块为准。"
    )
    lines.append(
        "2. 归并映射是**离线、可审计、默认关闭**的：`qa_scale.build_scale_corpus` 的 "
        "`merge_map=None` 时行为与轮 1 逐位不变（用重跑产物逐字节比对验证）。"
    )
    lines.append("3. 其余如实登记项（逐条来自 `before_after.json.honest_notes`）：")
    lines.append("")
    for note in before_after["honest_notes"]:
        lines.append("   - " + _md(note))
    lines.append("")
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    """渲染 ``audit.md``；给 ``--repro-dir`` 时落 ``determinism_check.json``。"""
    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    repro_dir = str(getattr(args, "repro_dir", "") or "")
    records = _read_jsonl(os.path.join(out_dir, MERGE_MAP_FILE))
    before_after = _read_json(os.path.join(out_dir, BEFORE_AFTER_FILE))
    markdown = render_audit(records, before_after)
    md_sha, md_size = _write_text(os.path.join(out_dir, AUDIT_FILE), markdown)
    _log("[report] " + AUDIT_FILE + " = " + str(md_size) + " B " + md_sha[:16])
    if repro_dir:
        det = determinism_block(str(out_dir), str(repro_dir))
        # **轮 9 · W2**：证据块内附「删一件 ⇒ False」的非恒真现场证明
        det["missing_file_counterexample"] = determinism_missing_file_probe(
            str(out_dir), str(repro_dir)
        )
        _write_json(os.path.join(out_dir, DETERMINISM_EVIDENCE_FILE), det)
        _log(
            "[report] 双跑一致性：应比对 %d 件、实比对 %d 件，缺失 = %s，逐字节相同 = %s，"
            "不相等 = %s"
            % (
                int(det["n_expected_files"]),
                int(det["n_common_files"]),
                str(det["missing_files"]),
                str(det["determinism_ok"]),
                str(det["not_equal_files"]),
            )
        )
        probe = det["missing_file_counterexample"]
        _log(
            "[report] W2 自检：齐备时 ok=%s；删掉 %s 后 ok=%s（缺失登记 = %s）"
            % (
                str(probe["determinism_ok_when_complete"]),
                str(probe["removed_file"]),
                str(probe["determinism_ok_when_missing"]),
                str(probe["missing_files_when_missing"]),
            )
        )
        if bool(probe["determinism_ok_when_missing"]):
            raise AssertionError(
                "W2 自检失败：删掉一件产物后 determinism_ok 仍为 True ⇒ 修复无效，拒绝出数"
            )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_answer_merge(args: argparse.Namespace) -> int:
    """``answer_merge`` 子命令分派（错误边界与 :mod:`qa_scale_run` 同口径）。"""
    if int(getattr(sys.flags, "optimize", 0)) != 0:
        _log(
            "[answer_merge] 拒绝在优化模式下运行：sys.flags.optimize = "
            + str(int(getattr(sys.flags, "optimize", 0)))
            + "。本模块的前置校验与边界证据是硬约束，优化模式会丢弃 __debug__ 分支，"
            "使口径不可审计；请用 python（不带 -O）重跑。"
        )
        return 1
    sub = str(getattr(args, "answer_merge_cmd", "") or "probe")
    handlers = {"probe": cmd_probe, "eval": cmd_eval, "report": cmd_report}
    if sub not in handlers:
        _log("[answer_merge] 未知子命令 " + repr(sub) + "；可用 = " + str(sorted(handlers)))
        return 2
    try:
        return int(handlers[sub](args))
    except UnicodeEncodeError as exc:
        _log(
            "[answer_merge] 输出编码错误："
            + type(exc).__name__
            + ": "
            + str(exc)
            + "。这不是参数错误；可用 PYTHONIOENCODING=utf-8 重跑。"
        )
        return 1
    except FileNotFoundError as exc:
        _log("[answer_merge] 输入产物缺失：" + str(exc))
        return 1
    except (ValueError, KeyError) as exc:
        _log("[answer_merge] 非法参数 / 契约校验失败：" + str(exc))
        return 2
    except AssertionError as exc:
        _log("[answer_merge] 现场断言失败：" + str(exc))
        return 1
    except RuntimeError as exc:
        _log("[answer_merge] 运行失败（依赖 / 产物不可用）：" + str(exc))
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


def _add_answer_merge_arguments(p: argparse.ArgumentParser) -> None:
    """把 ``answer_merge`` 的参数表挂到给定子解析器上（**唯一**定义点）。"""
    p.add_argument("--product-dir", default="", help="n3d_qa 冻结产物目录（只读）")
    p.add_argument(
        "--out-dir",
        default="",
        help="产物目录（默认 checkpoints/qa_learn/_verify/answer_merge）",
    )
    p.add_argument("--log-file", default="", help="同时写 UTF-8(无 BOM) 日志文件")
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="留出抽稀 seed（单 seed）"
    )
    p.add_argument(
        "--holdout-pct",
        type=int,
        default=DEFAULT_HOLDOUT_PCT,
        help="测试集留出比例（百分比整数，0..100；默认 20）",
    )
    p.add_argument(
        "--max-classes",
        type=int,
        default=0,
        help="类别数上限（0 = 全量；两套口径都按同一 (-count, key) 排序键截断）",
    )


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    import n3d_qa_learn.qa_scale as QS

    parser = argparse.ArgumentParser(
        prog="n3d_qa_learn.answer_merge_run",
        description=(
            "轮 8：答案表同义归并的零训练对照（probe / eval / report）；"
            "不训练、不用模型；产物一律写 "
            + os.path.join(QS.VERIFY_ROOT, "answer_merge").replace("\\", "/")
            + " 之下。"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    helps = {
        "probe": "构建归并计划 + R3 反例现场断言 + 落 merge_map.jsonl",
        "eval": "归并前后零训练对照 + 确定性参照基线 + 结构决策门（落 before_after.json）",
        "report": "渲染 audit.md（人工复核表，含 high-risk）；可产出双跑证据",
    }
    for name in ANSWER_MERGE_COMMANDS:
        p = sub.add_parser(name, help=helps[str(name)])
        _add_answer_merge_arguments(p)
        if str(name) == "report":
            p.add_argument(
                "--repro-dir",
                default="",
                help="对照产物目录：给定时产出双跑逐字节一致证据",
            )
        p.set_defaults(func=cmd_answer_merge, answer_merge_cmd=str(name))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口（``python -m n3d_qa_learn.answer_merge_run <cmd>``）。

    **轮 9 · W3**：``--out-dir`` 与 ``--log-file`` 在**建任何目录之前**先过守卫；
    越界一律打印可读报文并**退码 2**，且不创建目录、不写日志文件。
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    log_file = str(getattr(args, "log_file", "") or "")
    out_dir = str(getattr(args, "out_dir", "") or "")
    try:
        _guard_out_dir(out_dir)
        _guard_log_file(log_file)
    except ValueError as exc:
        _log("[answer_merge] 非法参数 / 契约校验失败：" + str(exc))
        return 2
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
