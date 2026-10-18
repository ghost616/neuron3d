"""轮 3 · B3 的纯新增 CLI：``python -m n3d_qa_learn.qa_variants_run {probe|eval|report}``。

子命令
------
* ``probe`` —— 规模与依赖取证（**不生成变体**）：问题单元数、绑定输出抽样、
  语料级候选词表构建（写进**本目录**，不碰文本侧词表）、生成器装配取证。
* ``eval`` —— 全量生成 4 支变体 + 逐支特征差异统计（**不训练、不评测模型**）
  + 三条整组切分不变量 + **反例取证** + 双跑一致性证据。
* ``report`` —— 渲染 ``qa_variants_report.{json,md}``。

产物纪律：只写 ``checkpoints/qa_learn/_verify/qa_variants/``；
**不落盘裸特征矩阵、不产 zip、无挂钟字段**；**不改动**文本侧既有 12 件产物。
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
    """打印一行进度（带刷新；不可表示字符做 ``backslashreplace`` 兜底）。"""
    text = str(msg)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = str(getattr(sys.stdout, "encoding", "") or "ascii")
        safe = text.encode(enc, errors="backslashreplace").decode(enc, errors="replace")
        print(safe, flush=True)


#: CLI 子命令（顺序敏感）。
QA_VARIANTS_COMMANDS: Tuple[str, ...] = ("probe", "eval", "report")

#: 双跑证据文件名（唯一来源）。
DETERMINISM_EVIDENCE_FILE: str = "determinism_check.json"

#: 证据排除口径（与 qa_scale / merge_table 同源纪律）。
DETERMINISM_SCOPE: str = (
    "证据 = 主产物目录与对照目录**共有的非排除文件**逐文件比对 SHA256 与字节数；"
    "排除 ① 证据文件自身（两目录都过滤，避免自指哈希链与「全新对照目录首跑判 False」）；"
    "② 后置可重渲件（本报告 json/md，由 report 在证据之后写出）。"
    "单侧独有文件不纳入 determinism_ok 布尔值，但必须显式登记。"
)

#: 后置可重渲件。
RERENDERABLE_FILES: Tuple[str, ...] = (
    "qa_variants_report.json",
    "qa_variants_report.md",
)


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


def _write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> Tuple[str, int]:
    """写 JSONL（逐行 ``sort_keys`` + 紧凑分隔符；LF；末尾换行）。"""
    lines = [
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    return _write_text(path, "\n".join(lines) + "\n")


def _read_json(path: str) -> Any:
    """读 JSON；缺文件给出可读报错。"""
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


def _normalize_dir(path: str) -> str:
    """目录路径的机器可比规范化（绝对路径 + 统一正斜杠 + 小写）。"""
    return os.path.abspath(str(path)).replace("\\", "/").lower()


def _guard_out_dir(out_dir: str) -> str:
    """产物目录守卫：**只允许**写在 ``checkpoints/qa_learn/_verify/`` 之下。"""
    import n3d_qa_learn.qa_variants as QV

    text = str(out_dir) if out_dir else QV.DEFAULT_OUT_DIR
    root_abs = os.path.abspath(QV.VERIFY_ROOT)
    norm_abs = os.path.abspath(text)
    prefix = root_abs + os.sep
    if not (norm_abs == root_abs or norm_abs.startswith(prefix)):
        raise ValueError(
            "产物目录只允许写在 "
            + repr(QV.VERIFY_ROOT.replace(chr(92), "/")) + " 之下，当前 = "
            + repr(norm_abs.replace(chr(92), "/"))
        )
    return text


def _resolve_product_dir(explicit: str) -> str:
    """定位 ``n3d_qa`` 冻结产物目录（只读）。"""
    import n3d_qa_learn.qa_scale as QS

    if explicit:
        if os.path.isfile(os.path.join(explicit, QS.ALL_PAIRS_FILE)):
            return str(explicit)
        raise FileNotFoundError("目录 " + repr(explicit) + " 下缺少 " + QS.ALL_PAIRS_FILE)
    if os.path.isfile(os.path.join(QS.DEFAULT_PRODUCT_DIR, QS.ALL_PAIRS_FILE)):
        return str(QS.DEFAULT_PRODUCT_DIR)
    raise FileNotFoundError(
        "未找到 n3d_qa 冻结产物（" + QS.ALL_PAIRS_FILE + "）；候选 = "
        + repr(QS.DEFAULT_PRODUCT_DIR)
    )


# ---------------------------------------------------------------------------
# 生成器装配（复用 text_variants，**不改其产出路径**）
# ---------------------------------------------------------------------------


def _build_generator(
    texts: Sequence[str],
    out_dir: str,
    *,
    min_doc_freq: int = 0,
    extractor: str = "",
) -> Tuple[Any, Dict[str, Any]]:
    """装配 4 支变体生成器（**复用** ``text_variants.VariantGenerator``）。

    关键约束
    --------
    1. 语料级候选词表在**本产物目录**构建（``qa_variants/keyword_vocab.json``），
       **不碰**文本侧的 ``keyword_vocab.json``（那 12 件产物必须逐字节不变）；
       **仅 KeyBERT 支**落该词表；TF-IDF 支不落词表文件（改落 ``keyword_idf.json``）；
    2. 关键词后端沿用 ``text_variants.build_keyword_backend``：KeyBERT 支用仓库自有
       bge-m3（``repo-encoder``）；**TF-IDF 支**用 jieba 词面统计（``jieba-tfidf``），
       不构造编码器、不需要 keybert；
    3. 生成口径（top-k / 摘要上限 / 回退规则）全部沿用第三步的冻结常量；
    4. **词表文档频次下限**是唯一的显式偏离：默认取
       :data:`n3d_qa_learn.qa_variants.QA_VOCAB_MIN_DOC_FREQ`（= 20，理由见该常量），
       并在产物里登记 ``vocab_caliber_deviation``（TF-IDF 支为 ``min_doc_freq``）；
    5. **抽取器可插拔**（轮 5）：``extractor="keybert"``（默认）走历史 KeyBERT 路径，
       证据的**键与取值与历史逐字相同**（零回归的代码侧保证）；
       ``extractor="tfidf"`` 走 jieba TF-IDF（语料侧 idf 统计落
       ``keyword_idf.json``，证据多 ``keyword_extractor`` / ``extractor_evidence`` 两块）；
    6. **TF-IDF 支的语料 fail-closed（轮 6 / W2）**：idf 必须在本侧语料上统计，
       故 ``texts`` **不能为空**（空则 ValueError，可读报文）；语料同时显式传给
       ``VariantGenerator(keyword_corpus=...)``，生成器侧不再允许「退化到批次拟合」。
    """
    import n3d_qa_learn.qa_variants as QV
    import n3d_qa_learn.text_variants as TV

    mdf = int(min_doc_freq) if int(min_doc_freq) > 0 else int(QV.QA_VOCAB_MIN_DOC_FREQ)
    extractor_name = str(extractor or TV.KEYWORD_EXTRACTOR_DEFAULT).strip().lower()
    if extractor_name not in TV.KEYWORD_EXTRACTORS:
        raise ValueError(
            "未知关键词抽取器 " + repr(extractor) + "；允许值 = "
            + str(list(TV.KEYWORD_EXTRACTORS))
        )
    vocab_path = os.path.join(str(out_dir), "keyword_vocab.json")
    if extractor_name == TV.KEYWORD_EXTRACTOR_TFIDF:
        # --- TF-IDF 支（**词面统计**）：语料侧 idf 只在一处统计，落 keyword_idf.json ---
        corpus = [str(t) for t in texts]
        if not corpus:
            raise ValueError(
                "TF-IDF 支的 idf 语料为空：必须在**本侧语料**（QA 侧 = 全部问题文本）上"
                "统计 idf，不得用空语料生成（N 与 df 会全部退化）"
            )
        backend = TV.build_keyword_backend(
            extractor=TV.KEYWORD_EXTRACTOR_TFIDF,
            keyword_top_k=int(TV.KEYWORD_TOP_K),
            min_doc_freq=int(mdf),
        )
        fit_ev = backend["model_object"].fit(corpus)
        fit_ev["corpus_scope"] = "explicit_keyword_corpus"
        fit_ev["corpus_n_texts"] = int(len(corpus))
        fit_ev["corpus_scope_note"] = (
            "QA 侧 idf 语料 = 本次生成的全部问题文本（" + str(len(corpus)) + " 条）；"
            "``--limit-questions`` 会让 N 随之缩小，属**口径偏离**并在此显式登记"
        )
        idf_path = os.path.join(str(out_dir), "keyword_idf.json")
        _write_json(
            idf_path,
            {
                "schema_version": str(TV.TEXT_VARIANTS_SCHEMA_VERSION),
                "backend_kind": str(backend.get("kind", "")),
                "extractor": str(extractor_name),
                "corpus": "qa_questions（本模块自建；**不**复用文本行侧词表）",
                "idf_formula": str(TV.TFIDF_IDF_FORMULA),
                "df_rule": str(TV.TFIDF_DF_RULE),
                "score_rule": str(TV.TFIDF_SCORE_RULE),
                "candidate_rule": str(TV.TFIDF_CANDIDATE_RULE),
                "dependency_note": str(TV.TFIDF_DEPENDENCY_NOTE),
                "min_doc_freq": int(mdf),
                "evidence": fit_ev,
            },
        )
        generator = TV.VariantGenerator(
            keyword_backend=backend, vocab_path="", keyword_corpus=corpus
        )
        evidence: Dict[str, Any] = {
            "backend_kind": str(backend.get("kind", "")),
            "backend_note": str(backend.get("note", "")),
            "word_max_length": int(backend.get("word_max_length", 0)),
            "keyword_extractor": str(extractor_name),
            "extractor_evidence": fit_ev,
            "extractor_evidence_path": str(idf_path),
            "extractor_evidence_sha256": _sha256_file(idf_path),
            "vocab_min_doc_freq": int(mdf),
            "text_side_min_doc_freq": int(TV.KEYWORD_MIN_DOC_FREQ),
            "idf_corpus": "qa_questions（自建；不复用文本行侧词表）",
            "vocab_caliber_deviation": str(QV.VOCAB_CALIBER_DEVIATION),
            "variant_order": list(TV.VARIANT_ORDER),
            "keyword_top_k": int(generator.keyword_top_k),
            "summary_max_sentences": int(generator.summary_max_sentences),
            "summary_max_chars": int(generator.summary_max_chars),
            "generator_manifest": generator.definition_manifest(),
            "reused_module": "n3d_qa_learn.text_variants.VariantGenerator",
            "reuse_rule": (
                "**复用**第三步的生成器实现与冻结常量；只换输入语料（QA 问题文本）、"
                "抽取器（tfidf；词面统计口径见 `extractor_evidence`）与**语料级 df 下限**"
                "（" + str(mdf) + "，见 `vocab_caliber_deviation`）。"
                "**不改动**文本行侧产出路径与其 12 件既有产物。"
            ),
        }
        return generator, evidence
    # --- KeyBERT 支（**默认档**；证据键与历史逐字相同 ⇒ 产物逐字节不变）-----
    backend = TV.build_keyword_backend(
        allow_sentence_transformers_fallback=True,
        extractor=TV.KEYWORD_EXTRACTOR_KEYBERT,
    )
    vocab_evidence = TV.build_keyword_vocabulary(list(texts), min_doc_freq=int(mdf), write=False)
    parent = os.path.dirname(os.path.abspath(vocab_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "schema_version": str(TV.TEXT_VARIANTS_SCHEMA_VERSION),
                "min_doc_freq": int(mdf),
                "n_rows": int(vocab_evidence["n_rows"]),
                "n_pairs": int(vocab_evidence["n_pairs"]),
                "n_distinct": int(vocab_evidence["n_distinct"]),
                "n_words": int(vocab_evidence["n_words"]),
                "words": list(vocab_evidence["words"]),
                "corpus": "qa_questions（本模块自建；**不**复用文本行侧词表）",
                "caliber_deviation": str(QV.VOCAB_CALIBER_DEVIATION),
            },
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=1,
        )
        handle.write("\n")
    generator = TV.VariantGenerator(keyword_backend=backend, vocab_path=vocab_path)
    evidence: Dict[str, Any] = {
        "backend_kind": str(backend.get("kind", "")),
        "backend_note": str(backend.get("note", "")),
        "word_max_length": int(backend.get("word_max_length", 0)),
        "vocab_path": str(vocab_path),
        "vocab_sha256": _sha256_file(vocab_path),
        "vocab_min_doc_freq": int(mdf),
        "text_side_min_doc_freq": int(TV.KEYWORD_MIN_DOC_FREQ),
        "vocab_n_words": int(vocab_evidence["n_words"]),
        "vocab_n_distinct": int(vocab_evidence["n_distinct"]),
        "vocab_corpus": "qa_questions（自建；不复用文本行侧词表）",
        "vocab_caliber_deviation": str(QV.VOCAB_CALIBER_DEVIATION),
        "variant_order": list(TV.VARIANT_ORDER),
        "keyword_top_k": int(generator.keyword_top_k),
        "summary_max_sentences": int(generator.summary_max_sentences),
        "summary_max_chars": int(generator.summary_max_chars),
        "generator_manifest": generator.definition_manifest(),
        "reused_module": "n3d_qa_learn.text_variants.VariantGenerator",
        "reuse_rule": (
            "**复用**第三步的生成器实现与冻结常量；只换输入语料（QA 问题文本）、"
            "词表落点（本产物目录）与**词表文档频次下限**（20，显式偏离，见"
            "`vocab_caliber_deviation`）。**不改动**文本行侧产出路径与其 12 件既有产物。"
        ),
    }
    return generator, evidence


def _attach_extractor_stats(
    generator: Any, gen_ev: Dict[str, Any], *, generated: bool
) -> None:
    """把 **TF-IDF 打分侧统计**（候选词与命中统计）挂进生成器取证。

    仅 ``extractor = tfidf`` 生效（KeyBERT 档**不改**任何证据键 ⇒ 零回归）。
    ``generated=False``（probe 不生成变体）时只登记「未产生」，**不臆造 0 值**。
    """
    import n3d_qa_learn.text_variants as TV

    if str(gen_ev.get("keyword_extractor", "")) != TV.KEYWORD_EXTRACTOR_TFIDF:
        return
    evidence = gen_ev.get("extractor_evidence")
    if not isinstance(evidence, dict):
        return
    if not generated:
        evidence["scoring"] = {
            "applicable": False,
            "note": "probe 只做规模与依赖取证、**不生成变体** ⇒ 打分侧（命中）统计在 eval 产物里",
        }
        return
    backend = getattr(generator, "keyword_backend", None)
    extractor = backend.get("model_object") if isinstance(backend, dict) else None
    snapshot = extractor.stats_snapshot() if extractor is not None else {}
    snapshot["applicable"] = True
    # 两个**来源不同**的计数并列给出，便于互证（不是恒等式）。
    snapshot["generator_counters"] = {
        str(k): dict(v) for k, v in sorted(getattr(generator, "counters", {}).items())
    }
    evidence["scoring"] = snapshot


def _gen_evidence(
    texts: Sequence[str],
    out_dir: str,
    args: argparse.Namespace,
    *,
    generated: bool,
) -> Dict[str, Any]:
    """装配生成器并把 ``--keyword-extractor`` 贯通（**唯一**取参点）。

    返回值即产物里的 ``generator`` 取证块（KeyBERT 档与历史逐字相同）。
    """
    generator, evidence = _build_generator(
        texts, out_dir, extractor=str(getattr(args, "keyword_extractor", "") or "")
    )
    _attach_extractor_stats(generator, evidence, generated=bool(generated))
    return evidence


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    """规模与依赖取证（**不生成变体**）。"""
    import n3d_qa_learn.qa_scale as QS
    import n3d_qa_learn.qa_variants as QV

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    product_dir = _resolve_product_dir(str(getattr(args, "product_dir", "") or ""))
    holdout_pct = int(getattr(args, "holdout_pct", QS.DEFAULT_HOLDOUT_PCT))
    seed = int(getattr(args, "seed", 42) or 42)
    limit_q = int(getattr(args, "limit_questions", 0) or 0)
    corpus = QS.build_scale_corpus(product_dir, max_classes=0)
    texts = QV.load_question_texts(product_dir)
    units = QV.build_question_units(
        corpus, question_texts=texts, limit_questions=int(limit_q)
    )
    per_task: Dict[str, int] = {}
    for u in units:
        per_task[str(u.task)] = per_task.get(str(u.task), 0) + 1
    # 绑定输出的**抽样**取证（证明它确实是答案而不是问题原文）
    sample = [
        {
            "question_id": str(u.question_id),
            "question_text_head": str(u.question_text)[:80],
            "answer_display": str(u.answer_display),
            "binding_is_question_text": bool(str(u.answer_display) == str(u.question_text)),
            "gold_class_indices": list(u.gold_class_indices),
        }
        for u in units[:5]
    ]
    n_binding_equals_question = int(
        sum(1 for u in units if str(u.answer_display) == str(u.question_text))
    )
    doc: Dict[str, Any] = {
        "version": str(QV.QA_VARIANTS_VERSION),
        "product_dir": str(product_dir),
        "product_files_sha256": {
            "n3d_qa_all_pairs.jsonl": QV.sha256_file(
                os.path.join(product_dir, QS.ALL_PAIRS_FILE)
            ),
            "answer_table.jsonl": QV.sha256_file(
                os.path.join(product_dir, QS.ANSWER_TABLE_FILE)
            ),
        },
        "scope": {
            "max_classes": int(corpus.n_classes),
            "holdout_pct": int(holdout_pct),
            "seed": int(seed),
            "limit_questions": int(limit_q),
            "statement": (
                "沿用轮 1 已生效口径：max_classes=298（全量）+ 按比例留出 "
                + str(holdout_pct) + "% + 问题组级切分 + 单 seed(" + str(seed) + ")"
            ),
        },
        "measured": {
            "n_questions": int(len(units)),
            "n_classes": int(corpus.n_classes),
            "per_task_questions": {str(k): int(v) for k, v in sorted(per_task.items())},
            "n_binding_equals_question_text": int(n_binding_equals_question),
        },
        "binding_output": {
            "rule": str(QV.BINDING_OUTPUT_RULE),
            "sample": sample,
            "n_binding_equals_question_text": int(n_binding_equals_question),
            "verdict": (
                "绑定输出 = 答案展示文本（与问题原文相等的条数为 "
                + str(n_binding_equals_question)
                + "）；**不是**问题原文"
            ),
        },
        "extractive_note": str(QV.EXTRACTIVE_NOTE),
        "group_split_rule": str(QV.GROUP_SPLIT_RULE),
        "generator": _gen_evidence(
            [u.question_text for u in units], out_dir, args, generated=False
        ),
    }
    sha, size = _write_json(os.path.join(out_dir, "qa_variants_probe.json"), doc)
    _log(
        "[probe] 问题单元 %d 个（类 %d）；逐任务 = %s；绑定输出==问题原文的条数 = %d"
        % (
            int(len(units)),
            int(corpus.n_classes),
            str(doc["measured"]["per_task_questions"]),
            int(n_binding_equals_question),
        )
    )
    _log("[probe] qa_variants_probe.json = " + str(size) + " B " + sha[:16])
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def _variant_side_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """按侧 × 变体统计记录数（纯计数，用于报告分档）。"""
    out: Dict[str, Dict[str, int]] = {}
    for r in records:
        side = str(r["side"])
        var = str(r["variant"])
        out.setdefault(side, {})
        out[side][var] = int(out[side].get(var, 0)) + 1
    return {
        "by_side": {s: dict(sorted(v.items())) for s, v in sorted(out.items())},
        "n_total": int(len(records)),
    }


def _evaluability_summary(
    diff: Dict[str, Any], *, order: Sequence[str], gen_counters: Dict[str, Any]
) -> Dict[str, Any]:
    """**可评估性汇总**（E-B3a 修复，离朱 R82）：逐支 `n_evaluable` + 与回退计数互证。

    口径：`n_evaluable` = 与原文支**文本不同**的条数 —— 只有这些条才存在
    「变体 vs 原文」的差异可比。原文支自身恒等 ⇒ `n_evaluable = 0`。
    非锚点支若 `n_evaluable < n`，说明有回退到原文的条数，必须与
    ``generator_counters[variant]["fallback_to_original"]`` **逐项相等**（互证）。
    """
    per: Dict[str, Any] = {}
    for name in order:
        d = diff[str(name)]
        ev = d["evaluability"]
        fb = int((gen_counters.get(str(name)) or {}).get("fallback_to_original", 0))
        is_anchor = bool(ev["is_anchor"])
        # 锚点行没有「回退」概念（它自己就是原文）⇒ 互证字段标 ``None``（不适用），
        # **不得**写成 False —— 那会被误读成「互证失败」。
        matches = (
            None if is_anchor else bool(int(d["n_text_identical"]) == int(fb))
        )
        per[str(name)] = {
            "is_anchor": bool(is_anchor),
            "n": int(d["n"]),
            "n_evaluable": int(ev["n_evaluable"]),
            "n_not_evaluable": int(ev["n_not_evaluable"]),
            "n_text_identical": int(d["n_text_identical"]),
            "fallback_to_original": int(fb),
            "identical_equals_fallback": matches,
            "identical_equals_fallback_note": (
                "锚点支不适用（它自身就是原文，不存在「回退」）；"
                "非锚点支 = 与原文逐字节相同的条数是否等于生成器的 fallback_to_original"
                if is_anchor
                else "非锚点支：与原文逐字节相同的条数 vs 生成器 fallback_to_original"
            ),
        }
    return {
        "per_variant": per,
        "anchor_variant": str(order[0]) if order else "",
        "all_non_anchor_consistent": bool(
            all(
                (v["identical_equals_fallback"] is True)
                for v in per.values()
                if not bool(v["is_anchor"])
            )
        ),
        "rule": (
            "`n_evaluable` = 与原文支**文本不同**的条数（只有这些条存在差异可比）。"
            "原文支自身 `n_evaluable = 0`（恒等）；非锚点支的 "
            "`n_text_identical` 必须等于生成器的 `fallback_to_original`（互证）；"
            "锚点支该互证字段为 `null`（不适用，不是失败）。"
        ),
    }


def cmd_eval(args: argparse.Namespace) -> int:
    """全量生成 4 支变体 + 特征差异统计 + 整组切分不变量 + 反例取证。"""
    import n3d_qa_learn.qa_scale as QS
    import n3d_qa_learn.qa_variants as QV
    import n3d_qa_learn.text_variants as TV

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    repro_dir = str(getattr(args, "repro_dir", "") or "")
    product_dir = _resolve_product_dir(str(getattr(args, "product_dir", "") or ""))
    holdout_pct = int(getattr(args, "holdout_pct", QS.DEFAULT_HOLDOUT_PCT))
    seed = int(getattr(args, "seed", 42) or 42)
    limit_q = int(getattr(args, "limit_questions", 0) or 0)
    corpus = QS.build_scale_corpus(product_dir, max_classes=0)
    texts = QV.load_question_texts(product_dir)
    units = QV.build_question_units(
        corpus, question_texts=texts, limit_questions=int(limit_q)
    )
    _log("[eval] 问题单元 %d 个；开始生成 4 支变体" % int(len(units)))
    generator, gen_ev = _build_generator(
        [u.question_text for u in units],
        out_dir,
        extractor=str(getattr(args, "keyword_extractor", "") or ""),
    )
    variants = QV.generate_qa_variants(units, generator)
    # 生成完成后才把 TF-IDF 打分侧（命中）统计挂进取证（仅 tfidf 档生效）。
    _attach_extractor_stats(generator, gen_ev, generated=True)
    order = [str(v) for v in TV.VARIANT_ORDER]
    records = QV.variant_records(
        units, variants, variant_order=order, holdout_pct=int(holdout_pct), seed=int(seed)
    )
    _log("[eval] 生成完成：%d 条变体记录（%d 问题 x %d 支）" % (
        int(len(records)), int(len(units)), int(len(order))))
    # --- 整组切分：三条不变量 + 反例 ------------------------------------
    # **权威全集**来自语料侧现场枚举（不是从记录里现算）—— 否则「并集完整」会退化成
    # 「集合等于它自己」（恒真），删行查不出来。
    expected_rows = [
        str(u.question_id) + "|" + str(name) for u in units for name in order
    ]
    expected_questions = [str(u.question_id) for u in units]
    split_ev = QV.verify_group_split(
        records,
        expected_row_keys=expected_rows,
        expected_question_ids=expected_questions,
    )
    counter = QV.group_split_counter_examples(records)
    # **必须在特征差异循环之前取计数**：循环里要用 `fallback_to_original` 作为
    # `n_evaluable_expected` 的**独立来源**（E-B3b 修复）。
    gen_counters = dict(generator.counters)
    # --- 特征差异：逐支 vs 原文支 ---------------------------------------
    vec = _qa_vectorizer()
    anchor_texts = [str(r["variant_text"]) for r in records if str(r["variant"]) == order[0]]
    diff: Dict[str, Any] = {}
    for name in order:
        texts_of = [
            str(r["variant_text"]) for r in records if str(r["variant"]) == str(name)
        ]
        diff[str(name)] = QV.feature_difference_report(
            anchor_texts,
            texts_of,
            vectorizer=vec,
            variant_name=str(name),
            anchor_name=str(order[0]),
            # **E-B3b 修复（离朱 R83）**：把生成器自报的回退数作为 `n_evaluable` 的
            # **独立来源**期望值（`expected = n - fallback_to_original`）；否则
            # 「期望 = n」会让摘要支（4 条回退）自相矛盾。
            fallback_count=int(
                (gen_counters.get(str(name)) or {}).get("fallback_to_original", 0)
            ),
        )
        _log(
            "[eval] 变体 %-6s：文本同原文 %d/%d，特征裸字节同 %d/%d，余弦 min=%.6f mean=%.6f，"
            "余弦<1 的条数=%d"
            % (
                str(name),
                int(diff[str(name)]["n_text_identical"]),
                int(diff[str(name)]["n"]),
                int(diff[str(name)]["n_feature_bitwise_identical"]),
                int(diff[str(name)]["n"]),
                float(diff[str(name)]["cosine"]["min"]),
                float(diff[str(name)]["cosine"]["mean"]),
                int(diff[str(name)]["n_cosine_below_one"]),
            )
        )
    doc: Dict[str, Any] = {
        "version": str(QV.QA_VARIANTS_VERSION),
        "product_dir": str(product_dir),
        "scope": {
            "max_classes": int(corpus.n_classes),
            "holdout_pct": int(holdout_pct),
            "seed": int(seed),
            "limit_questions": int(limit_q),
        },
        "binding_output": {
            "rule": str(QV.BINDING_OUTPUT_RULE),
            "n_binding_equals_question_text": int(
                sum(1 for u in units if str(u.answer_display) == str(u.question_text))
            ),
        },
        "sizes": {
            "n_questions": int(len(units)),
            "n_variant_records": int(len(records)),
            "n_variants": int(len(order)),
            "variant_order": list(order),
            **_variant_side_stats(records),
        },
        "group_split": {
            "rule": str(QV.GROUP_SPLIT_RULE),
            "invariant_order": list(QV.GROUP_SPLIT_INVARIANTS),
            "verification": split_ev,
            "counter_examples": counter,
        },
        "feature_difference": {
            "vectorizer": {
                "profile": "lexical-88",
                "role": "question",
                "dim": int(vec.dim),
                "fingerprint": str(vec.fingerprint()),
            },
            "per_variant": diff,
            # **E-B3a 修复（离朱 R82）**：可评估性汇总块 —— 原文支 n_evaluable=0、
            # 三支非锚点各 = n（若有回退则 < n，并与 generator_counters 互证）。
            "evaluability": _evaluability_summary(diff, order=order, gen_counters=gen_counters),
            "rule": (
                "以**原文支**为参照逐支比对；余弦与最大绝对差都基于**同一**向量化器"
                "（lexical-88 / role=question）现场编码，不落盘特征矩阵。"
            ),
        },
        "generator": gen_ev,
        "generator_counters": gen_counters,
        "extractive_note": str(QV.EXTRACTIVE_NOTE),
    }
    sha, size = _write_json(os.path.join(out_dir, "qa_variants_eval.json"), doc)
    _write_jsonl(os.path.join(out_dir, "qa_variants_records.jsonl"), records)
    _log(
        "[eval] 整组切分三条不变量 all_hold=%s；反例全部翻面=%s"
        % (str(split_ev["all_hold"]), str(counter["all_counter_examples_fire"]))
    )
    _log("[eval] qa_variants_eval.json = " + str(size) + " B " + sha[:16])
    if repro_dir:
        det = determinism_block(str(out_dir), str(repro_dir))
        _write_json(os.path.join(out_dir, DETERMINISM_EVIDENCE_FILE), det)
        _log(
            "[eval] 双跑一致性：共有文件 %d，逐字节相同 = %s，不相等 = %s"
            % (
                int(det["n_common_files"]),
                str(det["determinism_ok"]),
                str(det["not_equal_files"]),
            )
        )
    return 0


def _qa_vectorizer() -> Any:
    """QA 侧题面向量化器（**复用** ``entry_table.build_vectorizer_for``）。"""
    import n3d_qa_learn.entry_table as ET

    return ET.build_vectorizer_for("lexical-88", "question")


# ---------------------------------------------------------------------------
# 双跑证据
# ---------------------------------------------------------------------------


def determinism_block(out_dir: str, repro_dir: str) -> Dict[str, Any]:
    """双跑逐字节一致证据（排除证据自身与后置可重渲件）。"""
    excluded = {str(DETERMINISM_EVIDENCE_FILE)} | {str(x) for x in RERENDERABLE_FILES}
    a_all = sorted(f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f)))
    b_all = sorted(f for f in os.listdir(repro_dir) if os.path.isfile(os.path.join(repro_dir, f)))
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
    a_norm = _normalize_dir(out_dir)
    b_norm = _normalize_dir(repro_dir)
    return {
        "a_dir": str(out_dir),
        "b_dir": str(repro_dir),
        "a_dir_normalized": str(a_norm),
        "b_dir_normalized": str(b_norm),
        "paths_normalized_equal": bool(a_norm == b_norm),
        "path_dependence_note": (
            "`a_dir` / `b_dir` 保留**命令原样串**；另给 `*_normalized` 与 "
            "`paths_normalized_equal` 供机器判定。内嵌原样串的产物在「绝对路径不同」的"
            "运行之间不逐字节相同；同路径同参重跑才逐字节相同。"
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
            "determinism_ok = 「两目录共有的非排除文件全部逐字节相等」且集合非空；"
            "单侧独有文件不纳入该布尔值但必须显式登记。"
        ),
        "determinism_scope": DETERMINISM_SCOPE,
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def render_markdown(rep: Dict[str, Any]) -> str:
    """渲染 Markdown 报告（确定性：无时间戳、无耗时）。"""
    ev = rep["eval"]
    lines: List[str] = []
    lines.append("# 轮 3 · B3：QA 侧变体生成（绑定输出 = 答案）")
    lines.append("")
    lines.append("## 1. 生效口径")
    lines.append("")
    sc = ev["scope"]
    lines.append(
        "> max_classes=%d（全量）+ 按比例留出 %d%% + 问题组级切分 + 单 seed(%d)"
        % (int(sc["max_classes"]), int(sc["holdout_pct"]), int(sc["seed"]))
    )
    lines.append("")
    lines.append("- **本轮不训练任何模型**：只生成数据（变体）并给可量化的特征差异统计。")
    lines.append("- 复用 `n3d_qa_learn.text_variants.VariantGenerator`；**不改动**文本行侧"
                 "产出路径与其 12 件既有产物。")
    lines.append("")
    lines.append("## 2. 绑定输出 = 答案（**不是**问题原文）")
    lines.append("")
    lines.append("> " + str(ev["binding_output"]["rule"]))
    lines.append("")
    lines.append("- 现场：**绑定输出与问题原文逐字节相等的条数 = "
                 + str(ev["binding_output"]["n_binding_equals_question_text"]) + "**"
                 "（若该数与问题数相等，说明绑定输出退化成问题原文 —— 本口径当场失败）。")
    lines.append("")
    lines.append("## 3. 规模")
    lines.append("")
    sz = ev["sizes"]
    lines.append("- 问题单元 **" + str(sz["n_questions"]) + "** 个 x **"
                 + str(sz["n_variants"]) + "** 支 = **" + str(sz["n_variant_records"])
                 + "** 条变体记录；变体顺序 = `" + str(sz["variant_order"]) + "`")
    lines.append("")
    lines.append("| 侧 | " + " | ".join("`" + str(v) + "`" for v in sz["variant_order"]) + " |")
    lines.append("| --- " * (len(sz["variant_order"]) + 1) + "|")
    for side, counts in sorted(sz["by_side"].items()):
        lines.append(
            "| `" + str(side) + "` | "
            + " | ".join(str(int(counts.get(str(v), 0))) for v in sz["variant_order"])
            + " |"
        )
    lines.append("")
    lines.append("## 4. 整组切分（三条不变量 + 反例取证）")
    lines.append("")
    gs = ev["group_split"]
    lines.append("> " + str(gs["rule"]))
    lines.append("")
    ver = gs["verification"]
    lines.append("| 不变量 | 通过 | 现场计数 |")
    lines.append("| --- | --- | --- |")
    inv = ver["invariants"]
    r_inv = inv["rows_disjoint_and_union_complete"]
    lines.append(
        "| `rows_disjoint_and_union_complete` | **" + str(r_inv["passed"]) + "** | "
        "训练行 " + str(r_inv["n_train_rows"]) + " / 测试行 " + str(r_inv["n_test_rows"])
        + " / 合计 " + str(r_inv["n_rows_total"]) + "；交集 " + str(r_inv["n_intersection"])
        + "；并集 " + str(r_inv["n_union"]) + "；并集完整 " + str(r_inv["union_complete"])
        + " |"
    )
    q_inv = inv["questions_disjoint_and_union_complete"]
    lines.append(
        "| `questions_disjoint_and_union_complete` | **" + str(q_inv["passed"]) + "** | "
        "训练问题 " + str(q_inv["n_train_questions"]) + " / 测试问题 "
        + str(q_inv["n_test_questions"]) + " / 合计 " + str(q_inv["n_questions_total"])
        + "；交集 " + str(q_inv["n_intersection"]) + " |"
    )
    w_inv = inv["every_question_wholly_on_one_side"]
    lines.append(
        "| `every_question_wholly_on_one_side` | **" + str(w_inv["passed"]) + "** | "
        "跨侧问题数 " + str(w_inv["n_mixed_questions"]) + " |"
    )
    lines.append("")
    lines.append("- 三项全成立：**" + str(ver["all_hold"]) + "**")
    lines.append("")
    lines.append("### 4.1 反例取证（证明不变量**非恒真**）")
    lines.append("")
    lines.append("| 反例 | 目标不变量 | 破坏前 | 破坏后 | 断言如期触发 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for case in gs["counter_examples"]["cases"]:
        lines.append(
            "| `" + str(case["case"]) + "` | `" + str(case["broken_invariant"]) + "` | "
            + str(case["before_passed"]) + " | " + str(case["after_passed"]) + " | **"
            + str(case["assertion_fires"]) + "** |"
        )
    lines.append("")
    lines.append("- 全部反例如期触发：**"
                 + str(gs["counter_examples"]["all_counter_examples_fire"]) + "**")
    lines.append("- " + str(gs["counter_examples"]["rule"]))
    lines.append("")
    lines.append("## 5. 变体后**特征确实变化**（逐支 vs 原文支）")
    lines.append("")
    fd = ev["feature_difference"]
    lines.append("- 向量化器：" + str(fd["vectorizer"]))
    lines.append("")
    lines.append("| 变体 | n | 文本同原文 | 特征裸字节同原文 | 余弦 min | 余弦 mean | 余弦 p50 | 余弦<1 的条数 | 判定 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in sz["variant_order"]:
        d = fd["per_variant"][str(name)]
        lines.append(
            "| `" + str(name) + "` | " + str(d["n"]) + " | "
            + str(d["n_text_identical"]) + " / " + str(d["n"]) + " | "
            + str(d["n_feature_bitwise_identical"]) + " / " + str(d["n"]) + " | "
            + ("%.6f" % float(d["cosine"]["min"])) + " | "
            + ("%.6f" % float(d["cosine"]["mean"])) + " | "
            + ("%.6f" % float(d["cosine"]["p50"])) + " | "
            + str(d["n_cosine_below_one"]) + " | **"
            + str(d["feature_really_changes"]) + "** |"
        )
    lines.append("")
    lines.append("### 5.1 可评估性（`evaluability`：原文支 n_evaluable=0，非锚点支应 = n）")
    lines.append("")
    ev_sum = fd.get("evaluability", {})
    lines.append("| 变体 | 是否锚点 | n | `n_evaluable` | `n_evaluable_expected` | `n_not_evaluable` | 文本同原文 | 生成器 `fallback_to_original` | `evaluable_consistent` |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in sz["variant_order"]:
        row = (ev_sum.get("per_variant") or {}).get(str(name))
        if not row:
            continue
        pv = (fd.get("per_variant") or {}).get(str(name), {})
        pev = pv.get("evaluability", {})
        lines.append(
            "| `" + str(name) + "` | " + str(row["is_anchor"]) + " | " + str(row["n"])
            + " | **" + str(row["n_evaluable"]) + "** | "
            + str(pev.get("n_evaluable_expected", "—")) + " | "
            + str(row["n_not_evaluable"]) + " | "
            + str(row["n_text_identical"]) + " | "
            + str(row["fallback_to_original"]) + " | "
            + ("**" + str(pev.get("evaluable_consistent")) + "**")
            + " |"
        )
    lines.append("")
    lines.append("- 期望值来源：非锚点支 `n_evaluable_expected = n - fallback_to_original`"
                 "（**生成器侧计数**），而 `n_evaluable` 来自**文本比对** ⇒ 两个来源不同，"
                 "该互证**不是恒真**。")
    lines.append("- 非锚点支「与原文相同条数 == 生成器回退数」全部成立：**"
                 + str(ev_sum.get("all_non_anchor_consistent")) + "**")
    lines.append("- " + str(ev_sum.get("rule", "")))
    lines.append("")
    lines.append("- " + str(fd["rule"]))
    lines.append("")
    lines.append("**如实登记：哪些变体与原文一致**（不得隐藏）——")
    for name in sz["variant_order"]:
        d = fd["per_variant"][str(name)]
        lines.append(
            "  - `" + str(name) + "`：与原文支**逐字节相同** "
            + str(d["n_text_identical"]) + "/" + str(d["n"])
            + "（比例 " + ("%.6f" % float(d["text_identical_rate"])) + "）；"
            "特征裸字节相同 " + str(d["n_feature_bitwise_identical"]) + "/" + str(d["n"])
        )
    lines.append("")
    lines.append("- " + str(ev["extractive_note"]))
    lines.append("")
    lines.append("## 6. 生成器取证")
    lines.append("")
    ge = ev["generator"]
    lines.append("- 复用实现：" + str(ge["reused_module"]))
    lines.append("- 后端：" + str(ge["backend_kind"]) + "；`word_max_length` = "
                 + str(ge["word_max_length"]))
    if ge.get("vocab_path"):
        lines.append("- 词表：" + str(ge["vocab_path"]) + "（" + str(ge["vocab_n_words"])
                     + " 词 / 候选 " + str(ge["vocab_n_distinct"]) + "）；SHA256 = `"
                     + str(ge["vocab_sha256"]) + "`")
    else:
        lines.append("- 词表：**不适用**（`tfidf` 档不落候选词表文件；df 下限直接在本侧语料上"
                     "统计，取证见 6.1 的 `keyword_idf.json`）")
    lines.append("- " + str(ge["reuse_rule"]))
    lines.append("- 逐支计数（generated / fallback_to_original / empty_source）：`"
                 + str(ev["generator_counters"]) + "`")
    lines.append("")
    import n3d_qa_learn.text_variants as TV

    if str(ge.get("keyword_extractor", "")) == TV.KEYWORD_EXTRACTOR_TFIDF:
        xev = ge.get("extractor_evidence") or {}
        scoring_ev = xev.get("scoring") or {}
        lines.append("### 6.1 TF-IDF 口径取证（**词面统计**，与 KeyBERT 的语义抽取不等价）")
        lines.append("")
        lines.append("- 抽取器：**`tfidf`**（backend kind = `" + str(ge["backend_kind"])
                     + "`）；依赖口径：" + str(xev.get("dependency_note", "")))
        lines.append("- **IDF 公式**：`" + str(xev.get("idf_formula", "")) + "`")
        lines.append("- **df 口径**：" + str(xev.get("df_rule", "")))
        lines.append("- **tf / 打分 / 排序**：" + str(xev.get("score_rule", "")))
        lines.append("- **候选集**：" + str(xev.get("candidate_rule", "")))
        lines.append("- **N（语料文档数）** = " + str(xev.get("n_docs"))
                     + "（语料口径 = `" + str(xev.get("corpus_scope", "")) + "`）；"
                     "df 下限 `min_doc_freq` = " + str(xev.get("min_doc_freq")))
        lines.append("- 候选词统计：过滤前候选词条数（逐行累加）**"
                     + str(xev.get("n_candidate_pairs")) + "**；唯一候选词 **"
                     + str(xev.get("n_distinct_candidates")) + "**；过 df 下限保留 **"
                     + str(xev.get("n_kept_candidates")) + "**；IDF 区间 ["
                     + ("%.6f" % float(xev.get("idf_min", 0.0))) + ", "
                     + ("%.6f" % float(xev.get("idf_max", 0.0))) + "]（均值 "
                     + ("%.6f" % float(xev.get("idf_mean", 0.0))) + "）")
        lines.append("- 语料指纹 `corpus_sha256` = `" + str(xev.get("corpus_sha256", ""))
                     + "`；取证文件 `" + str(ge.get("extractor_evidence_path", ""))
                     + "`（SHA256 = `" + str(ge.get("extractor_evidence_sha256", "")) + "`）")
        if bool(scoring_ev.get("applicable", False)):
            cnt = scoring_ev.get("counters") or {}
            lines.append("- **命中统计**（打分侧，逐行累加）：参与打分行数 "
                         + str(cnt.get("n_scored_lines")) + "；候选词条数 "
                         + str(cnt.get("n_candidates_total")) + "；过 df 下限保留 "
                         + str(cnt.get("n_candidates_kept_total")) + "；实际输出的词数 "
                         + str(cnt.get("n_terms_emitted")) + "；空候选行 "
                         + str(cnt.get("n_empty_candidate_lines")) + "；被上限截断行 "
                         + str(cnt.get("n_candidates_truncated_lines")))
            lines.append("- 逐行均值：候选 " + ("%.6f" % float(scoring_ev.get("mean_candidates_per_line", 0.0)))
                         + " / 保留 " + ("%.6f" % float(scoring_ev.get("mean_kept_candidates_per_line", 0.0)))
                         + " / 输出 " + ("%.6f" % float(scoring_ev.get("mean_terms_per_line", 0.0)))
                         + "；空候选率 " + ("%.6f" % float(scoring_ev.get("empty_candidate_rate", 0.0)))
                         + "；截断率 " + ("%.6f" % float(scoring_ev.get("truncated_rate", 0.0))))
        else:
            lines.append("- 命中统计：本次运行未产生打分侧统计"
                         "（`probe` 不生成变体；命中统计在 `eval` 产物里给出）")
        # **I6**：口径说明**只输出一次**（此前 else 分支已经打印过一次 note，
        # 下面又无条件再打印一次 ⇒ probe 渲染出的报告里同一句出现两遍）。
        lines.append("- " + str(scoring_ev.get("note", "")))
        lines.append("")
    lines.append("## 7. 确定性证据")
    lines.append("")
    det = rep.get("determinism")
    if det:
        lines.append("- 双跑目录：`" + str(det["a_dir"]) + "` vs `" + str(det["b_dir"]) + "`")
        lines.append("- 共有文件 " + str(det["n_common_files"]) + "，逐字节相同 = **"
                     + str(det["determinism_ok"]) + "**，不相等 = `"
                     + str(det["not_equal_files"]) + "`")
        lines.append("- 排除项：`" + str(det["excluded_files"]) + "`")
        lines.append("- `paths_normalized_equal` = " + str(det["paths_normalized_equal"]))
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
    lines.append("## 8. 如实登记的局限")
    lines.append("")
    lines.append("1. 4 支变体全部是**抽取式 / 词面级**变换，**不是**语义 paraphrase；"
                 "对部分问题是**信息减少**（见第 5 节的逐支一致率）。")
    lines.append("2. 本轮**不训练**、不做检索评测：只交付数据与可量化的特征差异统计。")
    lines.append("3. 单 seed（" + str(sc["seed"]) + "）；跨 seed 未测。")
    lines.append("4. 切分口径沿用轮 1；本轮不重复报告宏观指标。")
    if str((ev.get("generator") or {}).get("keyword_extractor", "")) == TV.KEYWORD_EXTRACTOR_TFIDF:
        lines.append("5. **关键词支为 `tfidf`（词面统计）**：idf 只用 tf 与语料 df，"
                     "与 KeyBERT 的语义嵌入余弦**不等价**；本产物只能与**同抽取器**的"
                     "产物比较，**不得**把 tfidf 档的数字代入 `_verify/qa_variants/`"
                     "（keybert 档）的结论，也不宣称 tfidf 更优 —— 本轮目标是**跑通全量口径**。")
    lines.append("")
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    """渲染 ``qa_variants_report.{json,md}``。"""
    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    probe = _read_json(os.path.join(out_dir, "qa_variants_probe.json"))
    ev = _read_json(os.path.join(out_dir, "qa_variants_eval.json"))
    rep: Dict[str, Any] = {"version": str(ev["version"]), "probe": probe, "eval": ev}
    det_path = os.path.join(out_dir, DETERMINISM_EVIDENCE_FILE)
    if os.path.isfile(det_path):
        rep["determinism"] = _read_json(det_path)
    markdown = render_markdown(rep)
    md_sha, md_size = _write_text(os.path.join(out_dir, "qa_variants_report.md"), markdown)
    fps: Dict[str, Any] = {}
    for name in (
        "qa_variants_probe.json",
        "qa_variants_eval.json",
        "qa_variants_records.jsonl",
        "keyword_vocab.json",
        "keyword_idf.json",
    ):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            fps[str(name)] = {"bytes": int(os.path.getsize(p)), "sha256": _sha256_file(p)}
    if os.path.isfile(det_path):
        fps[str(DETERMINISM_EVIDENCE_FILE)] = {
            "bytes": int(os.path.getsize(det_path)),
            "sha256": _sha256_file(det_path),
        }
    fps["qa_variants_report.md"] = {"bytes": int(md_size), "sha256": str(md_sha)}
    rep["artifact_fingerprints"] = fps
    json_sha, json_size = _write_json(os.path.join(out_dir, "qa_variants_report.json"), rep)
    _log(
        "[report] qa_variants_report.json = " + str(json_size) + " B " + json_sha[:16]
        + "；qa_variants_report.md = " + str(md_size) + " B " + md_sha[:16]
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_qa_variants(args: argparse.Namespace) -> int:
    """子命令分派（错误边界与既有 CLI 同口径）。"""
    if int(getattr(sys.flags, "optimize", 0)) != 0:
        _log(
            "[qa_variants] 拒绝在优化模式下运行：sys.flags.optimize = "
            + str(int(getattr(sys.flags, "optimize", 0)))
            + "。前置校验与反例取证是硬约束，优化模式会丢弃 __debug__ 分支与 docstring。"
        )
        return 1
    sub = str(getattr(args, "qa_variants_cmd", "") or "probe")
    handlers = {"probe": cmd_probe, "eval": cmd_eval, "report": cmd_report}
    if sub not in handlers:
        _log("[qa_variants] 未知子命令 " + repr(sub) + "；可用 = " + str(sorted(handlers)))
        return 2
    try:
        return int(handlers[sub](args))
    except UnicodeEncodeError as exc:
        _log("[qa_variants] 输出编码错误：" + type(exc).__name__ + ": " + str(exc))
        return 1
    except FileNotFoundError as exc:
        _log("[qa_variants] 输入产物缺失：" + str(exc))
        return 1
    except (ValueError, KeyError) as exc:
        _log("[qa_variants] 非法参数 / 契约校验失败：" + str(exc))
        return 2
    except RuntimeError as exc:
        _log("[qa_variants] 运行失败（依赖 / 产物不可用）：" + str(exc))
        return 1


class _Tee:
    """把 stdout 同时写到控制台与 UTF-8 无 BOM 日志文件。"""

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

    def __getattr__(self, item: Any) -> Any:
        """其余属性透传。"""
        return getattr(self._primary, item)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    import n3d_qa_learn.qa_variants as QV

    parser = argparse.ArgumentParser(
        prog="n3d_qa_learn.qa_variants_run",
        description=(
            "轮 3 · B3：QA 侧变体生成（复用 text_variants 的 4 支生成器；"
            "绑定输出 = 答案；按 question_id 整组切分）。产物一律写 "
            + QV.DEFAULT_OUT_DIR.replace("\\", "/") + " 之下。"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    helps = {
        "probe": "规模与依赖取证（不生成变体）",
        "eval": "全量生成 4 支变体 + 特征差异 + 整组切分不变量与反例 + 双跑证据",
        "report": "渲染 qa_variants_report.{json,md}",
    }
    for name in QA_VARIANTS_COMMANDS:
        p = sub.add_parser(name, help=helps[str(name)])
        p.add_argument("--product-dir", default="", help="n3d_qa 冻结产物目录（只读）")
        p.add_argument("--out-dir", default="", help="产物目录（默认 "
                       + QV.DEFAULT_OUT_DIR + "）")
        p.add_argument("--log-file", default="", help="同时写 UTF-8(无 BOM) 日志文件")
        p.add_argument("--seed", type=int, default=42, help="留出抽稀 seed（单 seed）")
        p.add_argument("--holdout-pct", type=int, default=20,
                       help="测试集留出比例（百分比整数，默认 20）")
        p.add_argument("--limit-questions", type=int, default=0,
                       help="小批演练：只取前 N 个问题（0 = 全量）")
        if str(name) in ("probe", "eval"):
            # **W3**：该开关**只有 probe / eval 消费**（二者会装配生成器）；
            # `report` 只读已落盘产物、不生成变体 ⇒ 不挂该开关，
            # 传了由 argparse 报 unrecognized arguments 并退码 2（不静默忽略）。
            p.add_argument("--keyword-extractor", default="",
                           help="关键词抽取器（keybert=默认 / tfidf=新增 jieba TF-IDF；"
                                "空 = 默认 keybert，走历史路径、产物逐字节不变）。"
                                "**仅 probe / eval 接受**（report 不生成变体）")
        if str(name) == "eval":
            p.add_argument("--repro-dir", default="",
                           help="对照产物目录：给定时产出双跑逐字节一致证据")
        p.set_defaults(func=cmd_qa_variants, qa_variants_cmd=str(name))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口（``python -m n3d_qa_learn.qa_variants_run <cmd>``）。"""
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
