"""轮 9：A 口径类集收缩 + 零训练原型基线 的 CLI（``probe`` / ``eval``）。

驱动方式：``python -m n3d_qa_learn.qa_scoped_run {probe|eval}``

子命令与产物（**一个子命令只负责一组产物**）
--------------------------------------------
* ``probe`` —— 只做**语料取证**：按 A 口径（``min_class_questions`` 题/类）重建语料与切分，
  复核**四条**切分不变量，落 ``qa_scoped_corpus.json``（语料摘要 / 类别分布 / 覆盖 / 留出比例）。
* ``eval`` —— **零训练原型基线**（类质心余弦 argmax）+ 同口径多数类基线 + **决策门**
  （含轮 10 · W1 的**分层敏感性表**），落 ``qa_scoped_prototype.json`` 与
  ``qa_scoped_gate.json``；给 ``--repro-dir`` 时额外落
  ``determinism_check.json``（双跑逐字节一致证据）。

**不训练任何模型**：原型只是训练侧问题特征的质心；产物**不落盘裸特征矩阵**、不产 zip、
无挂钟字段。

守卫与证据口径（沿用轮 9 修好的 W2/W3 口径）
-------------------------------------------
* 写盘根目录**只有** ``checkpoints/qa_learn/_verify/qa_scoped/``；``--out-dir`` / ``--log-file``
  越界一律**退码 2** 且**不创建目录**；
* ``determinism_ok`` 要求**三件内容产物齐备**且逐字节相等（缺任何一件即 False，并把缺失项
  登记到 ``missing_files``），并附「删一件 ⇒ False」的非恒真现场证明。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 日志（ASCII + 汉字；backslashreplace 兜底）
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


#: CLI 子命令（顺序敏感）。
QA_SCOPED_COMMANDS: Tuple[str, ...] = ("probe", "eval")

#: 产物文件名（**唯一来源**）。
CORPUS_FILE: str = "qa_scoped_corpus.json"
PROTOTYPE_FILE: str = "qa_scoped_prototype.json"
GATE_FILE: str = "qa_scoped_gate.json"
DETERMINISM_EVIDENCE_FILE: str = "determinism_check.json"

#: 双跑证据实际比对的**三件内容产物**（不含证据文件自身）。
COMPARED_FILES: Tuple[str, ...] = (CORPUS_FILE, PROTOTYPE_FILE, GATE_FILE)


# ---------------------------------------------------------------------------
# 小工具
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
            "缺少输入产物 " + repr(path) + "；请先跑前置子命令（probe）"
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


def out_root() -> str:
    """本模块**唯一**允许写盘的根目录（``checkpoints/qa_learn/_verify/qa_scoped``）。"""
    import n3d_qa_learn.qa_scoped as S

    return str(S.DEFAULT_OUT_DIR)


def _guard_out_dir(out_dir: str) -> str:
    """守卫产物目录：**只允许**写在 ``checkpoints/qa_learn/_verify/qa_scoped/`` 之下。

    越界抛 ``ValueError``（CLI 侧映射为**退码 2**），且调用方**先校验再建目录**。
    """
    text = str(out_dir) if out_dir else out_root()
    root_abs = os.path.abspath(out_root())
    norm_abs = os.path.abspath(text)
    if not (norm_abs == root_abs or norm_abs.startswith(root_abs + os.sep)):
        raise ValueError(
            "产物目录只允许写在 "
            + repr(out_root().replace(chr(92), "/"))
            + " 之下（不写 _verify 的其它子目录，避免覆盖别轮交付产物），当前 = "
            + repr(norm_abs.replace(chr(92), "/"))
        )
    return text


def _guard_log_file(log_file: str) -> str:
    """守卫日志文件（与产物目录**同等**规则；越界抛 ``ValueError``，不创建目录）。"""
    text = str(log_file)
    if not text:
        return text
    root_abs = os.path.abspath(out_root())
    norm_abs = os.path.abspath(text)
    if not norm_abs.startswith(root_abs + os.sep):
        raise ValueError(
            "日志文件只允许写在 "
            + repr(out_root().replace(chr(92), "/"))
            + " 之下（日志与产物同等守卫），当前 = "
            + repr(norm_abs.replace(chr(92), "/"))
        )
    return text


def _resolve_product_dir(explicit: str) -> str:
    """定位 ``n3d_qa`` 冻结产物目录（只读）。"""
    import n3d_qa_learn.qa_scoped as S

    if explicit:
        if os.path.isfile(os.path.join(explicit, S.ANSWER_TABLE_FILE)):
            return str(explicit)
        raise FileNotFoundError(
            "目录 " + repr(explicit) + " 下缺少 " + S.ANSWER_TABLE_FILE
        )
    if os.path.isfile(os.path.join(S.DEFAULT_PRODUCT_DIR, S.ANSWER_TABLE_FILE)):
        return str(S.DEFAULT_PRODUCT_DIR)
    raise FileNotFoundError(
        "未找到 n3d_qa 冻结产物（" + S.ANSWER_TABLE_FILE + "）；候选 = "
        + repr(S.DEFAULT_PRODUCT_DIR)
    )


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    """语料取证（**不评测**）：A 口径语料 + 切分 + 四条不变量，落 ``qa_scoped_corpus.json``。"""
    import n3d_qa_learn.qa_scoped as S

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    product_dir = _resolve_product_dir(str(getattr(args, "product_dir", "") or ""))
    min_q = int(getattr(args, "min_class_questions", S.DEFAULT_MIN_CLASS_QUESTIONS))
    holdout = int(getattr(args, "holdout_pct", S.DEFAULT_HOLDOUT_PCT))
    seed = int(getattr(args, "seed", S.DEFAULT_SEED))
    built = S.build_scoped_corpus(
        product_dir, min_class_questions=min_q, holdout_pct=holdout, seed=seed
    )
    doc = built["document"]
    sha, size = _write_json(os.path.join(out_dir, CORPUS_FILE), doc)
    corpus = built["corpus"]
    scope = doc["corpus"]["class_scope"]
    _log(
        "[probe] A 口径：门槛 %d 题/类 -> 类 %d（%d 起）/ 问题 %d / 行 %d"
        % (
            int(min_q),
            int(corpus.n_classes),
            int(scope["n_classes_before"]),
            int(corpus.n_questions),
            int(corpus.n_pairs),
        )
    )
    _log(
        "[probe] 每类问题数（门槛口径，pre-scope）：min=%d median=%.1f max=%d；"
        "问题-类关联合计=%d；覆盖到的不同问题=%d"
        % (
            int(scope["kept_class_question_counts"]["min"]),
            float(scope["kept_class_question_counts"]["median"]),
            int(scope["kept_class_question_counts"]["max"]),
            int(scope["n_question_class_incidences_kept"]),
            int(scope["n_questions_covered_any_kept"]),
        )
    )
    _log(
        "[probe] 切分：训练问题 %d / 测试问题 %d / 训练行 %d / 测试行 %d；"
        "测试问题占比 %.6f；四条不变量 = %s"
        % (
            int(doc["split"]["n_train_questions"]),
            int(doc["split"]["n_test_questions"]),
            int(doc["split"]["n_train_rows"]),
            int(doc["split"]["n_test_rows"]),
            float(doc["split"]["actual_test_question_ratio"]),
            str(
                [
                    doc["split"]["invariants"]["rows_disjoint_and_union_complete"],
                    doc["split"]["invariants"]["questions_disjoint_and_union_complete"],
                    doc["split"]["invariants"]["every_question_wholly_on_one_side"],
                    doc["split"]["invariants"]["question_side_matches_row_side"],
                ]
            ),
        )
    )
    _log("[probe] " + CORPUS_FILE + " = " + str(size) + " B " + sha[:16])
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    """零训练原型基线 + 同口径多数类基线 + 决策门（落两件产物；可附双跑证据）。"""
    import n3d_qa_learn.qa_scoped as S
    import n3d_qa_learn.qa_scale as QS

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    repro_dir = str(getattr(args, "repro_dir", "") or "")
    product_dir = _resolve_product_dir(str(getattr(args, "product_dir", "") or ""))
    min_q = int(getattr(args, "min_class_questions", S.DEFAULT_MIN_CLASS_QUESTIONS))
    holdout = int(getattr(args, "holdout_pct", S.DEFAULT_HOLDOUT_PCT))
    seed = int(getattr(args, "seed", S.DEFAULT_SEED))
    profile = str(getattr(args, "profile", S.DEFAULT_PROFILE) or S.DEFAULT_PROFILE)
    built = S.build_scoped_corpus(
        product_dir, min_class_questions=min_q, holdout_pct=holdout, seed=seed
    )
    corpus_doc_path = os.path.join(out_dir, CORPUS_FILE)
    consistency: Dict[str, Any] = {
        "corpus_file": CORPUS_FILE,
        "applicable": False,
        "rule": (
            "eval 自行重建语料并复核切分不变量；若盘上已有 probe 的 qa_scoped_corpus.json，"
            "则逐字段对账（不一致即抛）。"
        ),
    }
    if os.path.isfile(corpus_doc_path):
        disk = _read_json(corpus_doc_path)
        rebuilt = built["document"]
        for key in ("corpus", "split", "side_class_counts", "coverage", "caliber"):
            if disk.get(str(key)) != rebuilt.get(str(key)):
                raise AssertionError(
                    "eval 重建的语料与盘上 " + CORPUS_FILE + " 的字段 " + repr(key) + " 不一致；"
                    "拒绝在自相矛盾的语料上出数"
                )
        consistency.update({"applicable": True, "checked_fields": [
            "caliber", "corpus", "split", "side_class_counts", "coverage"
        ]})
    evaluated = S.evaluate_prototype(
        corpus=built["corpus"], split=built["split"], product_dir=product_dir, profile=profile
    )
    prototype_doc = evaluated["prototype_document"]
    gate_doc = evaluated["gate_document"]
    prototype_doc["corpus_consistency"] = consistency
    prototype_doc["product_dir"] = str(product_dir)
    prototype_doc["product_files_sha256"] = built["document"]["product_files_sha256"]
    sha_p, size_p = _write_json(os.path.join(out_dir, PROTOTYPE_FILE), prototype_doc)
    sha_g, size_g = _write_json(os.path.join(out_dir, GATE_FILE), gate_doc)
    proto = prototype_doc["metrics"]["prototype"]
    base = prototype_doc["metrics"]["majority_same_caliber"]
    _log(
        "[eval] 原型基线（问题级）：micro=%.6f macro=%.6f；多数类基线：micro=%.6f macro=%.6f"
        % (
            float(proto["micro"]),
            float(proto["macro"]),
            float(base["micro"]),
            float(base["macro"]),
        )
    )
    _log(
        "[eval] 决策门：delta_macro=%+.6f >= %.2f ? -> %s"
        % (float(gate_doc["delta_macro"]), float(gate_doc["margin"]), str(gate_doc["passed"]))
    )
    stratified = dict(gate_doc.get("stratified_sensitivity", {}))
    for row in stratified.get("rows", []):
        _log(
            "[eval] 分层敏感性 min_test>=%d：类 %d / 原型 macro %.6f / 多数类 macro %.6f / "
            "delta %+.6f（%s）"
            % (
                int(row["min_test_questions"]),
                int(row["n_classes"]),
                float(row["prototype_macro"]),
                float(row["majority_macro"]),
                float(row["delta_macro"]),
                "达到门槛" if bool(row["reaches_margin"]) else "不足门槛",
            )
        )
    _log(
        "[eval] 零 recall 类 %d / 有测试问题的类 %d；特征档 %s（D=%d，零范数 %d）"
        % (
            int(proto["n_zero_recall_classes"]),
            int(proto["n_classes_with_test_questions"]),
            str(prototype_doc["feature"]["profile"]),
            int(prototype_doc["feature"]["dim"]),
            int(prototype_doc["feature"]["n_zero_norm"]),
        )
    )
    _log("[eval] " + PROTOTYPE_FILE + " = " + str(size_p) + " B " + sha_p[:16])
    _log("[eval] " + GATE_FILE + " = " + str(size_g) + " B " + sha_g[:16])
    if repro_dir:
        det = determinism_block(str(out_dir), str(repro_dir))
        det["missing_file_counterexample"] = determinism_missing_file_probe(
            str(out_dir), str(repro_dir)
        )
        _write_json(os.path.join(out_dir, DETERMINISM_EVIDENCE_FILE), det)
        _log(
            "[eval] 双跑一致性：应比对 %d 件、实比对 %d 件，缺失 = %s，逐字节相同 = %s，不相等 = %s"
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
            "[eval] W2 口径自检：齐备时 ok=%s；删掉 %s 后 ok=%s（缺失登记 = %s）"
            % (
                str(probe["determinism_ok_when_complete"]),
                str(probe["removed_file"]),
                str(probe["determinism_ok_when_missing"]),
                str(probe["missing_files_when_missing"]),
            )
        )
        if not bool(probe["determinism_ok_when_complete"]):
            raise AssertionError(
                "双跑自检失败：**齐备态**（三件内容产物都复制进临时目录）determinism_ok 应为 "
                "True，实际为 False ⇒ 判据在默认路径上就不可用（两侧同时缺件会被误判），"
                "拒绝出数"
            )
        if bool(probe["determinism_ok_when_missing"]):
            raise AssertionError(
                "双跑自检失败：删掉一件产物后 determinism_ok 仍为 True ⇒ 判据恒真，拒绝出数"
            )
    return 0


# ---------------------------------------------------------------------------
# 双跑逐字节一致证据（与轮 9 修好的 answer_merge_run 同口径：三件齐备才算过）
# ---------------------------------------------------------------------------


def determinism_block(out_dir: str, repro_dir: str) -> Dict[str, Any]:
    """双跑证据：**三件内容产物齐备且逐字节相等**才算 ``determinism_ok``。"""
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
            "`a_dir` / `b_dir` 保留命令原样串；**它们会进本文件内容** ⇒ 本文件在"
            "「绝对路径不同」的运行之间不逐字节相同，同路径同参重跑才逐字节相同"
            "（与轮 1 / 轮 8 登记同源）。三件内容产物本身不含任何路径串。"
        ),
        "files": files,
        "n_common_files": int(len(files)),
        "n_expected_files": int(len(expected)),
        "missing_files": missing,
        "compared_files": expected,
        "excluded_files": sorted(excluded),
        "a_only_files": a_only,
        "b_only_files": b_only,
        "not_equal_files": not_equal,
        "determinism_ok": bool(len(files) == len(expected) and not not_equal),
        "determinism_ok_semantics": (
            "determinism_ok = 「三件内容产物**齐备**（len(files) == len(compared_files)）"
            "**且**逐个逐字节相等」；缺任何一件即 False 并登记到 `missing_files`。"
        ),
        "no_self_reference": (
            "本文件记录的是三件内容产物的哈希，而它们的内容**不含**本文件哈希 ⇒ 无自指循环；"
            "本文件自身亦被排除在比对集合外。"
        ),
    }


def determinism_missing_file_probe(out_dir: str, repro_dir: str) -> Dict[str, Any]:
    """**非恒真的现场证明**：删掉 b 侧一件产物后 ``determinism_ok`` 必须为 False。"""
    import shutil
    import tempfile

    expected = [
        str(x) for x in COMPARED_FILES if str(x) != str(DETERMINISM_EVIDENCE_FILE)
    ]
    removed = str(expected[-1]) if expected else ""
    with tempfile.TemporaryDirectory(prefix="qs_det_probe_") as tmp:
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
            "同一份代码、同一批文件：齐备时 True；删掉一件后必须 False 且 `missing_files` "
            "恰为被删文件。临时目录在 TEMP 下、用完即删，**路径不进产物**。"
        ),
        "temp_paths_not_persisted": True,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_qa_scoped(args: argparse.Namespace) -> int:
    """``qa_scoped`` 子命令分派（错误边界与 :mod:`qa_scale_run` 同口径）。"""
    if int(getattr(sys.flags, "optimize", 0)) != 0:
        _log(
            "[qa_scoped] 拒绝在优化模式下运行：sys.flags.optimize = "
            + str(int(getattr(sys.flags, "optimize", 0)))
            + "；其前置校验与边界证据是硬约束，请用不带 -O 的 python 重跑。"
        )
        return 1
    sub = str(getattr(args, "qa_scoped_cmd", "") or "probe")
    handlers = {"probe": cmd_probe, "eval": cmd_eval}
    if sub not in handlers:
        _log("[qa_scoped] 未知子命令 " + repr(sub) + "；可用 = " + str(sorted(handlers)))
        return 2
    try:
        return int(handlers[sub](args))
    except UnicodeEncodeError as exc:
        _log(
            "[qa_scoped] 输出编码错误："
            + type(exc).__name__
            + ": "
            + str(exc)
            + "。可用 PYTHONIOENCODING=utf-8 重跑。"
        )
        return 1
    except FileNotFoundError as exc:
        _log("[qa_scoped] 输入产物缺失：" + str(exc))
        return 1
    except (ValueError, KeyError) as exc:
        _log("[qa_scoped] 非法参数 / 契约校验失败：" + str(exc))
        return 2
    except AssertionError as exc:
        _log("[qa_scoped] 现场断言失败：" + str(exc))
        return 1
    except RuntimeError as exc:
        _log("[qa_scoped] 运行失败（依赖 / 产物不可用）：" + str(exc))
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


def _add_qa_scoped_arguments(p: argparse.ArgumentParser) -> None:
    """把 ``qa_scoped`` 的参数表挂到给定子解析器上（**唯一**定义点）。"""
    import n3d_qa_learn.qa_scoped as S

    p.add_argument("--product-dir", default="", help="n3d_qa 冻结产物目录（只读）")
    p.add_argument(
        "--out-dir",
        default="",
        help="产物目录（默认 checkpoints/qa_learn/_verify/qa_scoped）",
    )
    p.add_argument("--log-file", default="", help="同时写 UTF-8(无 BOM) 日志文件")
    p.add_argument(
        "--min-class-questions",
        type=int,
        default=int(S.DEFAULT_MIN_CLASS_QUESTIONS),
        help="A 口径类集门槛：每类**问题数**下限（默认 10）",
    )
    p.add_argument(
        "--holdout-pct",
        type=int,
        default=int(S.DEFAULT_HOLDOUT_PCT),
        help="测试集留出比例（百分比整数，默认 20）",
    )
    p.add_argument(
        "--seed", type=int, default=int(S.DEFAULT_SEED), help="留出抽稀 seed（单 seed）"
    )
    p.add_argument(
        "--profile",
        default=str(S.DEFAULT_PROFILE),
        help="原型基线的特征档（默认 lexical-88；换档须整口径重跑）",
    )


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    import n3d_qa_learn.qa_scoped as S

    parser = argparse.ArgumentParser(
        prog="n3d_qa_learn.qa_scoped_run",
        description=(
            "轮 9：A 口径类集收缩（每类问题数 >= 门槛）+ 零训练原型质心基线 + 决策门；"
            "不训练任何模型；产物一律写 "
            + str(S.DEFAULT_OUT_DIR).replace(os.sep, "/")
            + " 之下。"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    helps = {
        "probe": "语料取证：A 口径语料 + 切分 + 四条不变量（落 qa_scoped_corpus.json）",
        "eval": "零训练原型基线 + 同口径多数类基线 + 决策门 + 分层敏感性表（落两件产物，可附双跑证据）",
    }
    for name in QA_SCOPED_COMMANDS:
        p = sub.add_parser(name, help=helps[str(name)])
        _add_qa_scoped_arguments(p)
        if str(name) == "eval":
            p.add_argument(
                "--repro-dir",
                default="",
                help="对照产物目录：给定时产出双跑逐字节一致证据",
            )
        p.set_defaults(func=cmd_qa_scoped, qa_scoped_cmd=str(name))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口（``python -m n3d_qa_learn.qa_scoped_run <cmd>``）。"""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    log_file = str(getattr(args, "log_file", "") or "")
    out_dir = str(getattr(args, "out_dir", "") or "")
    try:
        _guard_out_dir(out_dir)
        _guard_log_file(log_file)
    except ValueError as exc:
        _log("[qa_scoped] 非法参数 / 契约校验失败：" + str(exc))
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
