""":mod:`n3d_qa_learn.text_variants_run` —— 第三步 Phase 1 的纯新增 CLI。

驱动方式：`python -m n3d_qa_learn.text_variants_run {probe|gate|eval|report}`

子命令
------
* `probe` —— 数据与依赖取证（**不训练**）：行表规模 / `char_len` 分布 / 库查询划分 /
  4 支变体构造 + KeyBERT 关键词支小批演练 / 编码器近邻 / QA 侧同答案多问题成组取证；
* `gate` —— 特征差异**硬门禁**（逐位 + cos 分布 + 短行重合比例）+ 难负例清单；
* `eval` —— **零训练**纯余弦检索评测（4 支 x 2 种负例设定 -> `R@1` / `R@5`）
  + 机器可读判定；`--repro-dir` 时额外产出**双跑逐字节一致**证据；
* `report` —— 渲染 `text_variants_report.{json,md}`。

产物一律写 `checkpoints/qa_learn/_verify/text_variants/`（`--out-dir` 可覆盖，
但**只允许**写在 `checkpoints/qa_learn/_verify/` 之下）；不落盘特征矩阵、不产 zip、
不写任何挂钟字段（故同参重跑**逐字节一致**）。

控制台编码纪律（沿用 `step2_run` 的 R59 F1 修复）
--------------------------------------------------
本机 `sys.stdout.encoding = cp936`（GBK），而 GBK **没有** `U+2212`（真减号）/
`U+21D2`（双箭头）等字符。本模块的**新增**日志字面量一律只用 ASCII + 汉字；
`_log` 另做 `backslashreplace` 兜底（不改任何数值与产物）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 日志（ASCII 安全 + backslashreplace 兜底）
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """打印一行进度（带刷新）。

    两道修复与 `step2_run._log` 同源（**缺一不可**）：① 字面量层只用 ASCII + 汉字；
    ② 本函数对不可表示字符做 `backslashreplace` 转义后再输出（兜底，不改数值与产物）。
    """
    text = str(msg)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = str(getattr(sys.stdout, "encoding", "") or "ascii")
        safe = text.encode(enc, errors="backslashreplace").decode(enc, errors="replace")
        print(safe, flush=True)


#: 日志字面量的编码纪律（与 `step2_run._ASCII_SAFE_LOG_NOTE` 同一纪律）。
ASCII_SAFE_LOG_NOTE: str = (
    "运行时日志字面量必须能被 Windows 控制台默认编码 cp936（GBK）表示："
    "U+2212（真减号）与 U+21D2（双箭头）等字符不在 GBK 内，出现即让 print 抛 "
    "UnicodeEncodeError（它继承 ValueError，会被 CLI 兜住并误报成非法参数）。"
    "本模块一律用 ASCII 替代：减号用 -、蕴含箭头用 =>。"
    "docstring / 注释 / 产物 JSON（UTF-8）不受此限。"
)

# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

#: 本轮**唯一**允许的产物根目录。
VERIFY_ROOT: str = os.path.join("checkpoints", "qa_learn", "_verify")

#: 默认产物目录。
DEFAULT_OUT_DIR: str = os.path.join(VERIFY_ROOT, "text_variants")

#: `n3d_qa` 冻结产物候选目录（只读）。
PRODUCT_CANDIDATES: Tuple[str, ...] = (
    os.path.join("checkpoints", "qa_learn", "dataset"),
    os.path.join("checkpoints", "qa_learn", "_verify", "drill4"),
)

#: QA 侧同答案多问题取证所用的问答对文件（**只读**；judge 任务，字段最少）。
QA_PAIRS_FILE: str = "n3d_qa_judge_pairs.jsonl"

#: 问答对的答案键字段名（现场核实：`candidate_answer_key`）。
QA_ANSWER_KEY_FIELD: str = "candidate_answer_key"

#: 问答对的问题 id 字段名（现场核实：`question_id`）。
QA_QUESTION_ID_FIELD: str = "question_id"


def _sha256_text(text: str) -> str:
    """UTF-8 裸字节 SHA256。"""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _write_text(path: str, text: str) -> Tuple[str, int]:
    """写 UTF-8 无 BOM 文本（`newline=` 强制 LF）；返回 `(sha256, bytes)`。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    blob = str(text).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(blob)
    return hashlib.sha256(blob).hexdigest(), int(len(blob))


def _write_json(path: str, obj: Any) -> Tuple[str, int]:
    """写 JSON（UTF-8 无 BOM，`sort_keys` + 缩进 1 + 末尾换行）；返回 `(sha256, bytes)`。"""
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    return _write_text(path, text)


def _read_json(path: str) -> Any:
    """读 JSON；缺文件给出**可读报错**。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "缺少输入产物 " + repr(path) + "；请先跑前置子命令（probe / gate / eval）"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _guard_out_dir(out_dir: str) -> str:
    """守卫产物目录：**只允许**写在 `checkpoints/qa_learn/_verify/` 之下。

    口径（**修正后**）：先把目标与根**都归一为绝对路径**再比前缀；因此
    **绝对路径与相对路径同等可用**（此前只接受相对路径，会把合法的绝对路径误判为越界）。
    """
    text = str(out_dir) if out_dir else DEFAULT_OUT_DIR
    root_abs = os.path.abspath(VERIFY_ROOT)
    norm_abs = os.path.abspath(text)
    prefix = root_abs + os.sep
    if not (norm_abs == root_abs or norm_abs.startswith(prefix)):
        raise ValueError(
            "产物目录只允许写在 "
            + repr(VERIFY_ROOT.replace(chr(92), "/")) + " 之下，当前 = "
            + repr(norm_abs.replace(chr(92), "/"))
        )
    return text


def _resolve_product_dir(explicit: str) -> str:
    """定位 `n3d_qa` 冻结产物目录（只读）。"""
    if explicit:
        if os.path.isfile(os.path.join(explicit, "doclines_rows.jsonl")):
            return str(explicit)
        raise FileNotFoundError(
            "目录 " + repr(explicit) + " 下缺少 doclines_rows.jsonl"
        )
    for candidate in PRODUCT_CANDIDATES:
        if os.path.isfile(os.path.join(candidate, "doclines_rows.jsonl")):
            return str(candidate)
    raise FileNotFoundError(
        "未找到 n3d_qa 冻结产物（doclines_rows.jsonl）；候选 = " + str(list(PRODUCT_CANDIDATES))
    )
# ---------------------------------------------------------------------------
# 共用装配（生成器 / 编码器 / 行表）
# ---------------------------------------------------------------------------


def _build_generator(
    args: argparse.Namespace, *, vocab_path: str = ""
) -> Tuple[Any, Dict[str, Any]]:
    """构造变体生成器（含关键词后端可用性取证）。

    返回
    ----
    Tuple[Any, Dict[str, Any]]
        `(generator, backend_evidence)`。
    """
    import n3d_qa_learn.text_variants as TV

    fallback = not bool(getattr(args, "no_st_fallback", False))
    try:
        backend = TV.build_keyword_backend(
            allow_sentence_transformers_fallback=fallback,
            cache_dir=str(getattr(args, "enc_cache_dir", "") or ""),
        )
    except Exception as exc:
        raise RuntimeError(
            "关键词支编码后端不可用（可读报错，不静默跳过）："
            + type(exc).__name__ + ": " + str(exc)
        ) from exc
    evidence: Dict[str, Any] = {
        "kind": str(backend.get("kind", "")),
        "note": str(backend.get("note", "")),
        "encoder_dim": (
            int(backend["encoder"].dim) if backend.get("encoder") is not None else 0
        ),
    }
    if backend.get("encoder") is not None:
        try:
            evidence["encoder_fingerprint"] = str(backend["encoder"].fingerprint())
        except Exception:
            evidence["encoder_fingerprint"] = ""
    # **fix 4（R68）**：把编码器**实际生效**的 max_length 透传进取证
    # （此前 `build_keyword_backend` 返回的 `word_max_length` 在重拼 evidence 时被丢弃，
    #   故产物里只有常量注释、没有可核对的实测字段）。
    evidence["word_max_length"] = int(backend.get("word_max_length", 0))
    generator = TV.VariantGenerator(
        keyword_top_k=int(getattr(args, "keyword_top_k", TV.KEYWORD_TOP_K) or TV.KEYWORD_TOP_K),
        summary_max_sentences=int(
            getattr(args, "summary_max_sentences", TV.SUMMARY_MAX_SENTENCES)
            or TV.SUMMARY_MAX_SENTENCES
        ),
        summary_max_chars=int(
            getattr(args, "summary_max_chars", TV.SUMMARY_MAX_CHARS)
            or TV.SUMMARY_MAX_CHARS
        ),
        keyword_backend=backend,
        vocab_path=str(vocab_path),
    )
    return generator, evidence


def _load_rows(args: argparse.Namespace) -> Tuple[str, List[Any]]:
    """读取文本行（只读；`--limit` 截断到前 N 行用于小批演练）。"""
    import n3d_qa_learn.text_variants as TV

    product_dir = _resolve_product_dir(str(getattr(args, "product_dir", "") or ""))
    rows = TV.load_text_rows(product_dir)
    limit = int(getattr(args, "limit", 0) or 0)
    if limit > 0:
        rows = list(rows[:limit])
    return product_dir, rows


def _deps_evidence() -> Dict[str, Any]:
    """依赖版本取证（**只记录，不判定**）。"""
    out: Dict[str, Any] = {"python": sys.version.split()[0], "packages": {}}
    for name in ("keybert", "jieba", "sklearn", "numpy", "torch", "transformers"):
        try:
            module = __import__(name)
            out["packages"][name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            out["packages"][name] = "UNAVAILABLE: " + type(exc).__name__
    out["ascii_safe_log_note"] = ASCII_SAFE_LOG_NOTE
    return out


def _runtime_env_evidence() -> Dict[str, Any]:
    """**运行时环境快照**（含离线开关与控制台编码）。

    存在理由（R67）：`variants_meta.json` 是**内容产物**，其逐字节一致性是本轮的声明之一；
    而这些环境变量会随**运行方式**（是否设 `HF_HUB_OFFLINE`、控制台编码）变化，
    故它们不得进入内容产物，而应只在**路径/环境相关**的 `probe_summary.json` 里登记。
    """
    return {
        "offline_env": {
            key: str(os.environ.get(key, ""))
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_ENDPOINT",
                        "HF_HUB_DISABLE_XET", "PYTHONIOENCODING")
        },
        "stdout_encoding": str(getattr(sys.stdout, "encoding", "") or ""),
        "note": "环境相关：**不**进入内容产物指纹（避免跨进程假性不一致）",
    }


def _product_rows_sha256(product_dir: str) -> str:
    """`doclines_rows.jsonl` 的 SHA256（只读；用于溯源）。"""
    import n3d_qa_learn.text_variants as TV

    return TV.file_sha256(os.path.join(product_dir, "doclines_rows.jsonl"))


# ---------------------------------------------------------------------------
# probe：数据与依赖取证（不训练）
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    """`probe`：数据侧 / 依赖 / 变体构造 / 难负例 / QA 成组**全量取证**（**不训练**）。

    写 `variants.jsonl` 与 `variants_meta.json`（变体产物冻结落盘 + SHA256 指纹）。
    """
    import n3d_qa_learn.text_variants as TV

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    os.makedirs(out_dir, exist_ok=True)
    product_dir, rows = _load_rows(args)
    _log("[probe] 产物目录 = " + product_dir + "；行数 = " + str(len(rows)))

    deps = _deps_evidence()
    _log("[probe] 依赖 = " + json.dumps(deps["packages"], ensure_ascii=False))

    row_index = TV.load_row_index(product_dir)
    rows_ev = TV.rows_evidence(rows)
    split_ev = TV.split_evidence(rows, row_index)
    _log(
        "[probe] char_len min/p25/med/p75/max = "
        + str(rows_ev["char_len_distribution"])
        + "；查询行 = " + str(split_ev["n_query_rows"])
        + "；库行 = " + str(split_ev["n_library_rows"])
    )

    # 语料级候选词表（warning 12）：**从全量行表构建**并落进本次产物目录，
    # 既让词表成为可审计产物，也避免 `--limit` 演练用子集词表污染全量跑次。
    full_rows = TV.load_text_rows(product_dir)
    vocab_info = TV.build_keyword_vocabulary(
        [str(r.text) for r in full_rows], write=False
    )
    vocab_path = os.path.join(out_dir, "keyword_vocab.json")
    _write_json(vocab_path, vocab_info)
    _log("[probe] 语料级候选词表 -> keyword_vocab.json（从 " + str(len(full_rows))
         + " 行全量构建；n_words = " + str(vocab_info["n_words"])
         + "；fingerprint = " + str(vocab_info["fingerprint"])[:16] + "...）")

    generator, backend_ev = _build_generator(args, vocab_path=vocab_path)
    # **fix 4（R68）**：现场实测关键词编码器的截断情形（最长若干行文本），
    # 把 `truncation_observed` 落进 `keyword_backend` 取证（随 variants_meta.json
    # 与 probe_summary.json 一并落盘）。`encoder` 只存在于生成器内部的后端字典里，
    # 不进 evidence（不可 JSON 化）。
    _kw_encoder = None
    _kw_backend = getattr(generator, "keyword_backend", None)
    if isinstance(_kw_backend, dict):
        _kw_encoder = _kw_backend.get("encoder")
    if _kw_encoder is not None:
        backend_ev["truncation_observed"] = _truncation_evidence(_kw_encoder, rows)
    _log("[probe] 关键词后端 kind = " + backend_ev["kind"] + "；dim = "
         + str(backend_ev["encoder_dim"]))

    # 覆盖度（warning 7）：本次评估的行数 / 源总行数 / limit，落进 variants_meta.json。
    source_total = len(TV.load_text_rows(product_dir))
    coverage = {
        "n_rows": int(len(rows)),
        "source_total_rows": int(source_total),
        "limit": int(getattr(args, "limit", 0) or 0),
        "coverage_ratio": float(len(rows) / max(1, source_total)),
        "is_full_corpus": bool(len(rows) == source_total),
        "note": "n_rows = 本次生成变体的行数；source_total_rows = n3d_qa 冻结文本行总数",
    }
    _log("[probe] 覆盖度 = " + json.dumps(coverage, ensure_ascii=False))

    records = TV.build_variant_records(
        rows, generator, limit=0,
        on_progress=lambda done, total: _log(
            "[probe] 变体生成 " + str(done) + "/" + str(total)
        ),
    )
    files = TV.variant_product_files(
        records, generator,
        source_rows_sha256=_product_rows_sha256(product_dir),
        product_dir=product_dir,
        extra={
            "dependencies": deps,
            "keyword_backend": backend_ev,
            "coverage": coverage,
        },
    )
    for name in ("variants.jsonl", "variants_meta.json"):
        digest, nbytes = _write_text(os.path.join(out_dir, name), files[name]["text"])
        _log("[probe] 写 " + name + " (" + str(nbytes) + " B, sha256=" + digest[:16] + "...)")

    fp = TV.variant_fingerprint(records)
    _log("[probe] 逐支「与原文逐字不同」= " + json.dumps(
        {k: v["distinct_from_original"] for k, v in fp["per_variant"].items()},
        ensure_ascii=False,
    ))

    encoder = None
    if bool(getattr(args, "no_encoder_neighbors", False)):
        _log("[probe] 显式跳过编码器近邻（--no-encoder-neighbors）："
             "不产 encoder_neighbor 类，已在产物中如实登记")
    else:
        try:
            encoder = _neighbor_encoder(args)
        except Exception as exc:
            _log("[probe] 编码器近邻不可用（如实登记，不静默跳过）："
                 + type(exc).__name__ + ": " + str(exc))
    if len(rows) < 2:
        _log("[probe] 行数 < 2：无法抽样随机负例，难负例清单**处置为跳过**"
             "（显式登记，不静默产空清单）")
        skipped = {
            "skipped": True,
            "reason": "random_negative_sets 需要至少 2 行；当前行数 = " + str(len(rows)),
            "counts": {},
            "fingerprint": {},
        }
        _write_json(os.path.join(out_dir, "negatives.json"), skipped)
        qa_ev = _qa_grouping_evidence(product_dir, int(getattr(args, "qa_limit", 0) or 0))
        _write_json(os.path.join(out_dir, "qa_grouping.json"), qa_ev)
        summary = {
            "command": "probe",
            "out_dir": out_dir,
            "product_dir": product_dir,
            "n_rows": int(len(rows)),
            "limit": int(getattr(args, "limit", 0) or 0),
            "dependencies": deps,
            "runtime_env": _runtime_env_evidence(),
            "keyword_backend": backend_ev,
            "rows_evidence": rows_ev,
            "split_evidence": split_ev,
            "variant_fingerprint": fp,
            "variant_files": {
                name: {"sha256": files[name]["sha256"], "bytes": files[name]["bytes"]}
                for name in sorted(files)
            },
            "negative_counts": {},
            "negative_fingerprint": {},
            "negatives_skipped": skipped["reason"],
            "qa_grouping": qa_ev["summary"],
            "qa_grouping_fingerprint": qa_ev["summary"]["fingerprint"],
        }
        digest, nbytes = _write_json(os.path.join(out_dir, "probe_summary.json"), summary)
        _log("[probe] 汇总 -> probe_summary.json (" + str(nbytes) + " B, sha256="
             + digest[:16] + "...)")
        _log("[probe] done (tiny-batch)")
        return 0

    negatives = TV.build_negative_manifest(
        rows, include_encoder_neighbors=encoder is not None, encoder=encoder,
        random_count=int(getattr(args, "random_negatives", TV.RANDOM_NEGATIVE_COUNT)
                         or TV.RANDOM_NEGATIVE_COUNT),
        # 0 = 用冻结常量默认（**不得**写成 `or 0`：那会让常量默认失效，见 patch I）
        hard_cap=int(getattr(args, "hard_cap", 0) or TV.HARD_NEGATIVE_CAP),
        encoder_topk=int(getattr(args, "encoder_topk", 0) or TV.ENCODER_NEIGHBOR_TOPK),
        on_progress=lambda done, total: _log(
            "[probe] 编码器近邻 " + str(done) + "/" + str(total)
        ),
    )
    nbytes = _write_json(os.path.join(out_dir, "negatives.json"), negatives)[1]
    _log("[probe] 写 negatives.json (" + str(nbytes) + " B)；计数 = "
         + json.dumps({k: v["n_pairs"] for k, v in negatives["counts"].items()},
                      ensure_ascii=False))

    qa_ev = _qa_grouping_evidence(product_dir, int(getattr(args, "qa_limit", 0) or 0))
    _write_json(os.path.join(out_dir, "qa_grouping.json"), qa_ev)
    _log("[probe] 写 qa_grouping.json；答案组数 = " + str(qa_ev["summary"]["n_groups"]))

    summary = {
        "command": "probe",
        "out_dir": out_dir,
        "product_dir": product_dir,
        "n_rows": int(len(rows)),
        "limit": int(getattr(args, "limit", 0) or 0),
        "dependencies": deps,
        "runtime_env": _runtime_env_evidence(),
        "keyword_backend": backend_ev,
        "rows_evidence": rows_ev,
        "split_evidence": split_ev,
        "variant_fingerprint": fp,
        "variant_files": {
            name: {"sha256": files[name]["sha256"], "bytes": files[name]["bytes"]}
            for name in sorted(files)
        },
        "negative_counts": negatives["counts"],
        "negative_fingerprint": negatives["fingerprint"],
        "qa_grouping": qa_ev["summary"],
        "qa_grouping_fingerprint": qa_ev["summary"]["fingerprint"],
    }
    digest, nbytes = _write_json(os.path.join(out_dir, "probe_summary.json"), summary)
    _log("[probe] 汇总 -> probe_summary.json (" + str(nbytes) + " B, sha256="
         + digest[:16] + "...)")
    _log("[probe] done")
    return 0
def _neighbor_encoder(args: argparse.Namespace) -> Any:
    """构造**编码器近邻**用的编码器（仓库自有 `models/bge-m3`；缺失即抛可读错）。"""
    from n3d_qa_learn.encoders import ENCODER_BGE_M3, EncoderConfig, build_vectorizer
    import n3d_qa_learn.text_variants as TV

    kwargs: Dict[str, Any] = {
        "name": str(getattr(args, "neighbor_encoder", "") or ENCODER_BGE_M3),
        "role": TV.ENCODER_NEIGHBOR_ROLE,
        "source": str(getattr(args, "neighbor_source", "") or TV.ENCODER_NEIGHBOR_SOURCE),
    }
    cache_dir = str(getattr(args, "enc_cache_dir", "") or "")
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return build_vectorizer(EncoderConfig(**kwargs))


def _qa_grouping_evidence(product_dir: str, limit: int) -> Dict[str, Any]:
    """QA 侧「同答案多问题」成组取证（**只读**一条问答对文件；本轮不接损失）。"""
    import n3d_qa_learn.text_variants as TV

    path = os.path.join(product_dir, QA_PAIRS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError("缺少问答对文件 " + repr(path))
    pairs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    per_answer, summary = TV.candidate_group_map(
        pairs,
        answer_key_field=QA_ANSWER_KEY_FIELD,
        question_id_field=QA_QUESTION_ID_FIELD,
        limit=int(limit),
    )
    return {
        "source_file": path,
        "source_file_sha256": TV.file_sha256(path),
        "answer_key_field": QA_ANSWER_KEY_FIELD,
        "question_id_field": QA_QUESTION_ID_FIELD,
        "summary": summary,
        "top_groups": [
            {"answer_key": k, "n_questions": v["n_questions"]}
            for k, v in sorted(
                per_answer.items(), key=lambda kv: (-kv[1]["n_questions"], kv[0])
            )[:10]
        ],
        "per_answer_note": (
            "逐答案明细（qids）体积过大，产物只落规模与指纹；"
            "完整明细可由 candidate_group_map() 现场复算（确定性）"
        ),
        "no_loss_note": TV.QA_GROUPING_NO_LOSS_NOTE,
    }


# ---------------------------------------------------------------------------
# gate：特征差异硬门禁 + 难负例清单
# ---------------------------------------------------------------------------


def _truncation_evidence(encoder: Any, rows: Sequence[Any]) -> Dict[str, Any]:
    """现场实测关键词编码器的截断情形（fix 4）。

    取**最长的若干条行文本**送入 ``encode_batch_with_stats``，读回每条的
    ``n_tokens`` / ``n_truncated``（缓存命中时取自条目元数据）。该指标用于独立核对
    「``KEYWORD_ENCODER_MAX_LENGTH = 512`` 不发生截断」这一主张。

    参数
    ----
    encoder : Any
        具备 ``max_length`` 与 ``encode_batch_with_stats`` 的编码器（``HFTextEncoder``）。
    rows : Sequence[Any]
        文本行（取 ``.text``）。

    返回
    ----
    Dict[str, Any]
        ``{max_length, n_probe_texts, longest_char_len, n_tokens_max,
        n_truncated, verdict, rule}``；编码失败时附 ``error`` 且 ``n_truncated = None``
        （如实登记，不静默丢失）。
    """
    texts = [str(r.text) for r in rows]
    longest = sorted(texts, key=len)[-min(8, len(texts)):] if texts else []
    out: Dict[str, Any] = {
        "max_length": int(getattr(encoder, "max_length", 0)),
        "n_probe_texts": int(len(longest)),
        "longest_char_len": int(max((len(t) for t in texts), default=0)),
        "rule": (
            "\u5bf9\u6700\u957f\u7684 8 \u6761\u884c\u6587\u672c\u5b9e\u6d4b n_tokens / n_truncated\uff1b"
            "n_truncated = max(0, n_tokens - max_length)"
        ),
    }
    try:
        stats = list(encoder.encode_batch_with_stats(longest))
    except Exception as exc:
        out["error"] = type(exc).__name__ + ": " + str(exc)
        out["n_truncated"] = None
        out["verdict"] = "unavailable"
        return out
    n_tok = [int(getattr(x, "n_tokens", 0)) for x in stats]
    n_trunc = [int(getattr(x, "n_truncated", 0)) for x in stats]
    out["n_tokens_max"] = int(max(n_tok, default=0))
    out["n_truncated"] = int(sum(n_trunc))
    out["verdict"] = "no_truncation" if out["n_truncated"] == 0 else "truncated"
    return out


def _load_coverage(out_dir: str) -> Dict[str, Any]:
    """从 ``variants_meta.json`` 读取**覆盖度**（probe 写入）。

    存在理由（warning 7）：单看 ``feature_gate.json`` / ``retrieval_eval.json``
    无法判断本次评估是否只跑了子集，会被误读为全量结果。
    """
    path = os.path.join(out_dir, "variants_meta.json")
    if not os.path.isfile(path):
        return {"n_rows": None, "source_total_rows": None, "limit": None,
                "note": "缺 variants_meta.json：无法声明覆盖度"}
    with open(path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    cov = dict(meta.get("coverage", {}) or {})
    cov.setdefault("n_rows", meta.get("n_rows"))
    return cov


def _coverage_log(coverage: Dict[str, Any]) -> str:
    """一行可读的覆盖度描述。"""
    n = coverage.get("n_rows")
    total = coverage.get("source_total_rows")
    lim = coverage.get("limit")
    if total is None:
        return "覆盖度：" + str(n) + " 行（源总行数未声明）"
    pct = (100.0 * float(n) / float(total)) if total else 0.0
    return ("覆盖度：" + str(n) + " / " + str(total) + " 行（"
            + ("%.1f" % pct) + "%；limit=" + str(lim) + "）")


def _require_negatives_for(out_dir: str, cmd_name: str) -> Optional[int]:
    """统一「缺 negatives.json」的处置（warning 11）：返回 None = 可继续，否则返回退码。"""
    path = os.path.join(out_dir, "negatives.json")
    if os.path.isfile(path):
        return None
    _log("[" + str(cmd_name) + "] 缺少 " + repr(path)
         + "：请先跑 probe（它会写出负例清单）。")
    return 1


def _load_records(out_dir: str) -> List[Dict[str, Any]]:
    """读回 `variants.jsonl`（规范化 JSONL；缺文件即可读报错）。"""
    path = os.path.join(out_dir, "variants.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError("缺少 " + repr(path) + "；请先跑 probe")
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    if not out:
        raise RuntimeError("variants.jsonl 为空：" + repr(path))
    return out


def cmd_gate(args: argparse.Namespace) -> int:
    """`gate`：**特征差异硬门禁**（逐位 + 按行型分档 + cos 分布）+ 难负例清单复核。"""
    import n3d_qa_learn.text_variants as TV

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    records = _load_records(out_dir)
    _log("[gate] 读入变体记录 " + str(len(records)) + " 行")

    vectorizer = TV.ZhBagVectorizer()
    t0 = time.time()
    coverage = _load_coverage(out_dir)
    _log("[gate] " + _coverage_log(coverage))
    gate = TV.feature_gate(records, vectorizer, coverage=coverage)
    gate["identity_definition"] = (
        "逐位相同 = float32 特征向量任一分量不等取反（np.array_equal）；"
        "零范数行（空文本）余弦记为 0.0，不产生 NaN"
    )
    nbytes = _write_json(os.path.join(out_dir, "feature_gate.json"), gate)[1]
    for name in TV.NON_ANCHOR_VARIANTS:
        ent = gate["per_variant"][name]
        _log(
            "[gate] " + str(name) + "：与原文支逐位相同 = " + str(ent["n_identical"])
            + " / " + str(ent["n_rows"]) + "（比例 " + repr(ent["identity_ratio"])
            + "）；cos 分布 = " + json.dumps(ent["cos_distribution"])
        )
    reg = gate["short_tier_identity_registration"]
    _log(
        "[gate] 短行档重合如实登记：tier=" + str(reg["tier"]) + " 行数="
        + str(reg["n_rows_in_tier"]) + " 相同=" + str(reg["n_identical_in_tier"])
        + " 档内比例=" + repr(reg["identity_ratio_in_tier"])
        + " 全量比例=" + repr(reg["identity_ratio_overall"])
    )
    _log("[gate] 写 feature_gate.json (" + str(nbytes) + " B)；耗时 "
         + ("%.1f" % (time.time() - t0)) + "s")

    npath = os.path.join(out_dir, "negatives.json")
    neg_info: Dict[str, Any] = {}
    if os.path.isfile(npath):
        manifest = _read_json(npath)
        neg_info = {
            "counts": manifest.get("counts", {}),
            "fingerprint": manifest.get("fingerprint", {}),
            "reused": True,
            "encoder_neighbor_skipped": bool(manifest.get("encoder_neighbor_skipped")),
            "encoder_neighbor_skip_reason": str(
                manifest.get("encoder_neighbor_skip_reason", "")
            ),
        }
        _log("[gate] 复核 negatives.json（沿用 probe 冻结清单，不重抽）；"
             "encoder_neighbor_skipped = "
             + str(neg_info["encoder_neighbor_skipped"]))
    else:
        # **统一口径（warning 11）**：缺 negatives.json 时 gate 与 eval 一致 -> 可读报错 + 退码 1。
        _log("[gate] 缺少 " + repr(npath)
             + "：请先跑 probe（gate 需要它复核难负例清单）。")
        return 1
    hard_gate_passed = bool(gate["hard_gate"]["passed"])
    reg = gate.get("identity_registration", {})
    _log(
        "[gate] 硬门禁（" + str(gate["hard_gate"]["gated_variants"]) + "）通过 = "
        + str(gate["hard_gate"]["passed"])
        + "；不可约简行 = " + str(gate["hard_gate"]["irreducible_rows"])
        + "；成员变体 = " + str(reg.get("registered_variants"))
        + "；任一重合行 = " + str(reg.get("n_rows_with_any_identical"))
        + " / " + str(gate["n_rows"]) + "（比例 " + repr(reg.get("identity_ratio_overall")) + "）"
    )
    summary = {
        "command": "gate",
        "out_dir": out_dir,
        "n_rows": int(len(records)),
        "feature_gate": gate,
        "negatives": neg_info,
        "hard_gate_passed": hard_gate_passed,
        "granularity_guard_rule": TV.GRANULARITY_GUARD_RULE,
    }
    digest, nbytes = _write_json(os.path.join(out_dir, "gate_summary.json"), summary)
    _log("[gate] 硬门禁通过 = " + str(hard_gate_passed) + "；汇总 -> gate_summary.json ("
         + str(nbytes) + " B, sha256=" + digest[:16] + "...)")
    if not hard_gate_passed:
        _log("[gate] FAIL：至少一支与原文支特征逐位相同（不允许）")
        return 1
    _log("[gate] done")
    return 0
# ---------------------------------------------------------------------------
# eval：零训练纯余弦检索评测 + 判定（+ 可选双跑确定性证据）
# ---------------------------------------------------------------------------


def _load_or_rebuild_negatives(
    args: argparse.Namespace,
    out_dir: str,
    rows: Sequence[Any],
) -> Dict[str, Any]:
    """读取（或现场重建）难负例清单；**不重抽**：已有产物一律沿用。"""
    import n3d_qa_learn.text_variants as TV

    path = os.path.join(out_dir, "negatives.json")
    rebuild = bool(getattr(args, "rebuild_negatives", False))
    if os.path.isfile(path) and not rebuild:
        _log("[eval] 沿用已冻结难负例清单 negatives.json")
        return _read_json(path)
    if not os.path.isfile(path) and not rebuild:
        # **三者一致（info 11）**：缺文件时 gate / report / eval 行为必须相同 ——
        # 可读报错 + 退码 1；**只有显式 `--rebuild-negatives` 才现场重建**（且会写盘）。
        raise FileNotFoundError(
            "缺少 " + repr(path) + "：请先跑 probe（它会写出负例清单）；"
            "如确需现场重建，请显式加 --rebuild-negatives（该开关会写盘 negatives.json）。"
        )
    _log("[eval] 现场重建难负例清单（显式 --rebuild-negatives）")
    encoder = None
    if bool(getattr(args, "no_encoder_neighbors", False)):
        _log("[eval] 显式跳过编码器近邻（--no-encoder-neighbors）")
    else:
        try:
            encoder = _neighbor_encoder(args)
        except Exception as exc:
            _log("[eval] 编码器近邻不可用（如实登记）：" + type(exc).__name__ + ": " + str(exc))
    manifest = TV.build_negative_manifest(
        rows, include_encoder_neighbors=encoder is not None, encoder=encoder,
        random_count=int(getattr(args, "random_negatives", TV.RANDOM_NEGATIVE_COUNT)
                         or TV.RANDOM_NEGATIVE_COUNT),
        # 0 = 用冻结常量默认（**不得**写成 `or 0`：那会让常量默认失效，见 patch I）
        hard_cap=int(getattr(args, "hard_cap", 0) or TV.HARD_NEGATIVE_CAP),
        encoder_topk=int(getattr(args, "encoder_topk", 0) or TV.ENCODER_NEIGHBOR_TOPK),
        on_progress=lambda done, total: _log(
            "[eval] 编码器近邻 " + str(done) + "/" + str(total)
        ),
    )
    _write_json(path, manifest)
    return manifest


def _query_split_ids(product_dir: str, records: Sequence[Dict[str, Any]]) -> List[str]:
    """n3d_qa 冻结划分的**查询行**（与变体记录取交，保持产物顺序）。"""
    import n3d_qa_learn.text_variants as TV

    row_index = TV.load_row_index(product_dir)
    query = {
        str(rec.get("row_id", ""))
        for rec in row_index
        if int(rec.get("rows_for_this_query", 0)) > 0
    }
    return [str(r["row_id"]) for r in records if str(r["row_id"]) in query]


#: **内嵌了本次运行 out_dir 的运行报告**（两目录下天然不同，不计入确定性失败）。
PATH_DEPENDENT_FILES: Tuple[str, ...] = (
    "probe_summary.json",
    "gate_summary.json",
    "retrieval_eval.json",
    "determinism_check.json",
)


def _determinism_block(out_dir: str, repro_dir: str) -> Dict[str, Any]:
    """**双跑逐字节一致**证据：逐文件比对主产物目录与对照目录。"""
    import n3d_qa_learn.text_variants as TV

    names: List[str] = []
    for base in (out_dir, repro_dir):
        if os.path.isdir(base):
            for name in os.listdir(base):
                if name not in names and os.path.isfile(os.path.join(base, name)):
                    names.append(name)
    files: Dict[str, Any] = {}
    only_a: List[str] = []
    only_b: List[str] = []
    n_common = 0
    for name in sorted(names):
        a = os.path.join(out_dir, name)
        b = os.path.join(repro_dir, name)
        ea = os.path.isfile(a)
        eb = os.path.isfile(b)
        ba = open(a, "rb").read() if ea else b""
        bb = open(b, "rb").read() if eb else b""
        if ea and eb:
            n_common += 1
        elif ea:
            only_a.append(name)
        else:
            only_b.append(name)
        files[name] = {
            "a_path": a,
            "b_path": b,
            "a_exists": bool(ea),
            "b_exists": bool(eb),
            "common": bool(ea and eb),
            "a_bytes": int(len(ba)),
            "b_bytes": int(len(bb)),
            "a_sha256": hashlib.sha256(ba).hexdigest() if ea else "",
            "b_sha256": hashlib.sha256(bb).hexdigest() if eb else "",
            "bitwise_equal": bool(ea and eb and ba == bb),
        }
    common_ok = all(v["bitwise_equal"] for v in files.values() if v["common"])
    # **R68 收口**：区分「全部共有文件」与「**内容产物**（排除内嵌路径的运行报告）」两者，
    # 避免把 path_dependent_files 的天然不同误读成确定性失败。
    content_common = [
        k for k, v in files.items() if v["common"] and k not in PATH_DEPENDENT_FILES
    ]
    content_ok = all(files[k]["bitwise_equal"] for k in content_common)
    return {
        "out_dir": out_dir,
        "repro_dir": repro_dir,
        "files": files,
        "n_common_files": int(n_common),
        "n_content_common_files": int(len(content_common)),
        "content_common_files": sorted(content_common),
        "content_files_bitwise_equal": bool(content_common and content_ok),
        "only_in_out_dir": sorted(only_a),
        "only_in_repro_dir": sorted(only_b),
        "all_bitwise_equal": bool(n_common > 0 and common_ok),
        "rule": (
            "两目录**共有**的文件必须逐字节相等（存在性 + 字节数 + SHA256 + 裸字节）；"
            "单边文件**如实登记**（only_in_out_dir / only_in_repro_dir）、"
            "不计入失败；两次运行必须是**不同进程**。"
        ),
        "path_dependent_files": list(PATH_DEPENDENT_FILES),
        "path_dependence_note": (
            "上列三件**内嵌了本次运行的 out_dir 字符串**，故两个不同目录下跑出来"
            "的它们**天然不同**；它们是**运行报告**，不是本轮要求的冻结变体产物。"
            "本轮要求的确定性声明只针对**变体产物**（variants.jsonl 及其指纹）与"
            "其余内容产物（variants_meta.json / feature_gate.json / negatives.json / qa_grouping.json），"
            "它们在两个独立进程下**逐字节相同**。"
        ),
    }


def cmd_eval(args: argparse.Namespace) -> int:
    """`eval`：**零训练**纯余弦检索评测（4 支 x 2 种负例设定）+ 判定结论。"""
    import n3d_qa_learn.text_variants as TV

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    os.makedirs(out_dir, exist_ok=True)
    records = _load_records(out_dir)
    product_dir = _resolve_product_dir(str(getattr(args, "product_dir", "") or ""))
    rows = _rows_from_records(records)
    coverage = _load_coverage(out_dir)
    _log("[eval] 记录 " + str(len(records)) + " 行；" + _coverage_log(coverage))

    manifest = _load_or_rebuild_negatives(args, out_dir, rows)
    if bool(manifest.get("skipped")):
        _log(
            "[eval] 负数清单为**跳过标记**：" + str(manifest.get("reason", ""))
            + "；请用 >= 2 行跑 probe（或先跑完整 probe）后再 eval。"
        )
        return 1
    vectorizer = TV.ZhBagVectorizer()
    qsplit = _query_split_ids(product_dir, records)
    _log("[eval] 冻结查询行 ∩ 变体记录 = " + str(len(qsplit)))

    t0 = time.time()
    ev = TV.zero_training_eval(
        records, vectorizer, manifest,
        query_split_ids=qsplit,
        topk=int(getattr(args, "topk", 5) or 5),
        coverage=coverage,
    )
    main = ev["settings"]["full_pool_all_rows"]
    anchor_r1 = float(main["variants"][TV.VARIANT_ANCHOR]["recall_at_1"])
    by_variant = {
        name: float(main["variants"][name]["recall_at_1"])
        for name in TV.NON_ANCHOR_VARIANTS
    }
    verdict = TV.render_verdict(anchor_r1, by_variant)
    _log("[eval] 设定 A（全量池，不消费 negatives）R@1 = " + json.dumps(
        {n: main["variants"][n]["recall_at_1"] for n in TV.VARIANT_ORDER},
        ensure_ascii=False,
    ))
    hard = ev["settings"]["small_pool_hard"]
    rnd = ev["settings"]["small_pool_random"]
    _log("[eval] 设定 B（难负例小池）R@1 = " + json.dumps(
        {n: hard["variants"][n]["recall_at_1"] for n in TV.VARIANT_ORDER},
        ensure_ascii=False,
    ))
    _log("[eval] 设定 C（随机负例小池）R@1 = " + json.dumps(
        {n: rnd["variants"][n]["recall_at_1"] for n in TV.VARIANT_ORDER},
        ensure_ascii=False,
    ))
    _log("[eval] 难负例对照 B-C 同支 R@1 差 = " + json.dumps(
        {n: ev["hard_vs_random_delta"][n]["delta_hard_minus_random"]
         for n in TV.VARIANT_ORDER},
        ensure_ascii=False,
    ))
    _log("[eval] verdict_key = " + verdict["verdict_key"] + "；逐支 = "
         + json.dumps(verdict["verdict_keys"], ensure_ascii=False))

    payload = {
        "command": "eval",
        "out_dir": out_dir,
        "product_dir": product_dir,
        "n_rows": int(len(records)),
        "retrieval_eval": ev,
        "verdict": verdict,
        "negative_counts": manifest.get("counts", {}),
        "negative_fingerprint": manifest.get("fingerprint", {}),
        "coverage": coverage,
        "measured_scope": ev.get("measured_scope", {}),
        "retrieval_pool_size": int(len(records)),
        "random_baseline": float(1.0 / max(1, len(records))),
        "granularity": float(1.0 / max(1, len(records))),
        "declared_constants": {
            "RETRIEVAL_POOL_SIZE": int(TV.RETRIEVAL_POOL_SIZE),
            "RANDOM_BASELINE": float(TV.RANDOM_BASELINE),
            "GRANULARITY": float(TV.GRANULARITY),
            "note": "全语料口径常量（供判定规则文本）；本次运行的实测基线见上两项",
        },
        "granularity_guard_rule": TV.GRANULARITY_GUARD_RULE,
        "verdict_rule": TV.VERDICT_RULE,
        "extractive_not_paraphrase_note": TV.EXTRACTIVE_NOT_PARAPHRASE_NOTE,
    }
    nbytes = _write_json(os.path.join(out_dir, "retrieval_eval.json"), payload)[1]
    _log("[eval] 写 retrieval_eval.json (" + str(nbytes) + " B)；评测耗时 "
         + ("%.1f" % (time.time() - t0)) + "s（**只进日志、不进产物**）")

    repro_dir = str(getattr(args, "repro_dir", "") or "")
    if repro_dir:
        repro_dir = _guard_out_dir(repro_dir)
        crit = os.path.join(repro_dir, "variants.jsonl")
        if not os.path.isfile(crit):
            _log(
                "[eval] 对照目录缺少 " + repr(crit)
                + "：--repro-dir 必须指向一个**已跑过 probe** 的目录"
                "（即含 variants.jsonl 的第二份跑次）；"
                "请先在该目录执行 probe+gate+eval，或去掉 --repro-dir。"
            )
            return 1
        det = _determinism_block(out_dir, repro_dir)
        det_path = os.path.join(out_dir, TV.DETERMINISM_EVIDENCE_FILE)
        _write_json(det_path, det)
        _log("[eval] 双跑逐字节一致 = " + str(det["all_bitwise_equal"])
             + "；证据 -> " + TV.DETERMINISM_EVIDENCE_FILE)
        content = [
            n for n, v in (det.get("files", {}) or {}).items()
            if v.get("common") and n not in (det.get("path_dependent_files") or [])
        ]
        content_equal = all(det["files"][n]["bitwise_equal"] for n in content) if content else False
        _log("[eval] 内容产物逐字节一致 = " + str(content_equal)
             + "；" + str(len(content)) + " 件；路径相关含报告 = "
             + str(det.get("path_dependent_files")))
        if not content_equal:
            _log("[eval] FAIL：两次运行的**内容产物**未逐字节一致")
            return 1
    _log("[eval] done")
    return 0
def _rows_from_records(records: Sequence[Dict[str, Any]]) -> List[Any]:
    """由变体记录重建 `TextRow`（**只补 text/file/line_no/row_id**，不读特征）。

    存在理由：`eval` 的检索池必须与 `variants.jsonl` **逐行对齐**，
    因此行对象一律由已冻结的记录派生，而不是再去磁盘上按 `--limit` 重读一遍。
    """
    from n3d_qa_learn.step2 import TextRow

    out: List[Any] = []
    for rec in records:
        out.append(
            TextRow(
                row_id=str(rec["row_id"]),
                file=str(rec.get("file", "")),
                line_no=int(rec.get("line_no", 0)),
                text=str(rec["variants"]["原文"]),
                feature=None,
            )
        )
    return out


# ---------------------------------------------------------------------------
# report：渲染 JSON + Markdown
# ---------------------------------------------------------------------------


def _small_pool_precondition(ev: Dict[str, Any]) -> Dict[str, Any]:
    """小池设定（B/C）的显式前置条件与现场一致性（fix 8）。

    B - C 只在两侧**逐查询行的池大小严格相等**时才只是「负例内容」之差；否则它是
    「池大小 + 内容」的混合差。本函数只**读** ``retrieval_eval.json`` 里已有的
    ``measured_scope.per_setting``，**不向该产物新增任何字段**（保持其指纹不变）。

    参数
    ----
    ev : Dict[str, Any]
        ``retrieval_eval.json`` 的内容。

    返回
    ----
    Dict[str, Any]
        前置条件文本 + 现场池大小三元组 + 两项一致性布尔。
    """
    from .text_variants import SMALL_POOL_PRECONDITION_RULE

    ms = (ev.get("retrieval_eval", {}) or {}).get("measured_scope", {}) or {}
    pst = ms.get("per_setting", {}) or {}
    b = pst.get("small_pool_hard", {}) or {}
    c = pst.get("small_pool_random", {}) or {}

    def _trip(d: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        return (d.get("pool_size_min"), d.get("pool_size_median"), d.get("pool_size_max"))

    return {
        "rule": SMALL_POOL_PRECONDITION_RULE,
        "sufficient_condition": (
            "\u6bcf\u4e2a\u67e5\u8be2\u884c\u5747\u6709 >= NEGATIVE_POOL_SIZE \u6761\u540c\u7c7b\u8d1f\u4f8b\uff0c"
            "\u4e14\u4e24\u8bbe\u5b9a\u7684\u67e5\u8be2\u884c\u96c6\u76f8\u540c"
        ),
        "observed_pool_size_triplet_hard": list(_trip(b)),
        "observed_pool_size_triplet_random": list(_trip(c)),
        "pool_size_triplet_equal": bool(_trip(b) == _trip(c)),
        "query_count_equal": bool(b.get("n_queries") == c.get("n_queries")),
        "asserted_in": "zero_training_eval\uff08fail-closed\uff09",
    }


def cmd_report(args: argparse.Namespace) -> int:
    """`report`：把 probe / gate / eval 的产物合成 `text_variants_report.{json,md}`。"""
    import n3d_qa_learn.text_variants as TV

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    probe = _read_json(os.path.join(out_dir, "probe_summary.json"))
    gate = _read_json(os.path.join(out_dir, "gate_summary.json"))
    ev = _read_json(os.path.join(out_dir, "retrieval_eval.json"))
    qa = _read_json(os.path.join(out_dir, "qa_grouping.json"))
    code = _require_negatives_for(out_dir, "report")
    if code is not None:
        return int(code)

    det_path = os.path.join(out_dir, TV.DETERMINISM_EVIDENCE_FILE)
    det = _read_json(det_path) if os.path.isfile(det_path) else {}

    honest: List[str] = [
        TV.EXTRACTIVE_NOT_PARAPHRASE_NOTE,
        "关键词支与摘要支所需依赖（keybert / jieba）登记在 "
        "n3d_qa_learn/requirements-qa.txt，根 requirements.txt 未改动。",
        "本轮**零训练**：全程不上任何模型、不训练；评测为纯余弦检索。",
        "单 seed（" + str(TV.TEXT_VARIANTS_SEED) + "）、确定性；"
        "所有随机过程使用局部 RNG（random.Random / numpy），不消耗全局 RNG。",
        # **R68 收口修正**：原文案称「检索池全程恒为全量行、两种负例设定」，
        # 与三设定设计（B/C 是**逐查询行的小池**）**直接矛盾**，故改写为与实现一致的描述。
        "检索共**三种设定**（A `full_pool_all_rows` / B `small_pool_hard` / "
        "C `small_pool_random`）：**仅 A 的检索池 = 本次评估的全部行**"
        "（库支恒为「原文支」）；**B / C 是逐查询行的小池**"
        "（池 = { 自身 } ∪ 同类负例，本轮池大小恒为 17）。"
        "难负例对照 = **B - C**（两侧逐行池大小严格相等，故只差负例内容）；"
        "**A 不消费 negatives.json**，不得当作难负例对照基础。"
        "小池设定的前置条件由 `zero_training_eval` 在评测前 `assert` 校验。",
        "变体文本一律是**确定性**的（KeyBERT 关闭 MMR、候选词由 jieba 固定分词），"
        "故同参重跑产物逐字节一致。",
    ]
    # **R68 收口**：“证据缺位”类文字**只在证据确实缺位时**才出现（条件 = 该文件不存在）；
    # 并把该条件本身写进报告（`determinism_evidence`），使“为何出现 / 为何不出现”可审计。
    det_evidence = {
        "file": str(TV.DETERMINISM_EVIDENCE_FILE),
        "path": str(det_path),
        "present": bool(det),
        "note_present": bool(not det),
        "n_common_files": int(det.get("n_common_files", 0)) if det else 0,
        # **R68 收口**：确定性声明只针对**内容产物**；内嵌 out_dir 的运行报告
        # （path_dependent_files）天然不同，不得混入"逐字节一致"的判定。
        "n_content_common_files": int(det.get("n_content_common_files", 0)) if det else 0,
        "content_files_bitwise_equal": (
            bool(det.get("content_files_bitwise_equal", False)) if det else False
        ),
        "all_common_files_bitwise_equal": (
            bool(det.get("all_bitwise_equal", False)) if det else False
        ),
        "path_dependent_files": (
            list(det.get("path_dependent_files", [])) if det else []
        ),
    }
    if not det:
        honest.append(
            "本轮**未提供** " + str(TV.DETERMINISM_EVIDENCE_FILE) + "（--repro-dir 双跑证据），"
            "**双跑逐字节一致**证据缺位（须以独立进程跑两次并加 --repro-dir 才能产出）。"
        )
    else:
        honest.append(
            "双跑证据**已具备**：" + str(TV.DETERMINISM_EVIDENCE_FILE)
            + " 共比对 " + str(int(det.get("n_common_files", 0))) + " 件共有文件，"
            "其中**内容产物 " + str(det_evidence["n_content_common_files"])
            + " 件逐字节一致 = " + str(det_evidence["content_files_bitwise_equal"]) + "**；"
            "另 " + str(len(det_evidence["path_dependent_files"]))
            + " 件内嵌 out_dir 的运行报告登记为 path_dependent_files（天然不同，"
            "不计入确定性判定）。"
        )
    if str(probe.get("keyword_backend", {}).get("kind", "")) == "sentence-transformers":
        honest.append(
            "关键词后端**回退到了 sentence-transformers**（仓库自有编码器路径不可用）；"
            "已在 variants_meta.json 的 keyword_backend.note 中如实登记失败原因。"
        )

    report: Dict[str, Any] = {
        "schema_version": TV.TEXT_VARIANTS_SCHEMA_VERSION,
        "variant_order": list(TV.VARIANT_ORDER),
        "non_anchor_variants": list(TV.NON_ANCHOR_VARIANTS),
        "retrieval_pool_size": int(
            ev.get("retrieval_eval", {}).get("measured_scope", {}).get("n_rows", 0)
        ),
        "retrieval_pool_size_full_corpus_constant": int(TV.RETRIEVAL_POOL_SIZE),
        "random_baseline": float(
            ev.get("retrieval_eval", {}).get("measured_scope", {})
            .get("random_baseline_measured_full_pool", 0.0)
        ),
        "granularity": float(
            ev.get("retrieval_eval", {}).get("measured_scope", {})
            .get("granularity_measured", 0.0)
        ),
        "declared_constants": ev.get("declared_constants", {}),
        "coverage": ev.get("coverage", {}),
        "measured_scope": ev.get("measured_scope", {}),
        # **fix 8（R68）**：小池设定的显式前置条件 + 现场一致性（数据全部取自现有字段）
        "small_pool_precondition": _small_pool_precondition(ev),
        "determinism_evidence": det_evidence,
        "verdict_rule": TV.VERDICT_RULE,
        "granularity_guard_rule": TV.GRANULARITY_GUARD_RULE,
        "variant_definitions": probe.get("variant_fingerprint", {}),
        "variant_product": probe.get("variant_fingerprint", {}),
        "variant_files": probe.get("variant_files", {}),
        "dependencies": probe.get("dependencies", {}),
        "keyword_backend": probe.get("keyword_backend", {}),
        "rows_evidence": probe.get("rows_evidence", {}),
        "split_evidence": {
            k: v for k, v in (probe.get("split_evidence", {}) or {}).items()
            if k not in ("query_row_ids", "library_row_ids")
        },
        "feature_gate": gate.get("feature_gate", {}),
        "negatives": {
            "counts": gate.get("negatives", {}).get("counts", {}),
            "fingerprint": gate.get("negatives", {}).get("fingerprint", {}),
        },
        "retrieval_eval": ev.get("retrieval_eval", {}),
        "verdict": ev.get("verdict", {}),
        "qa_grouping": {"summary": qa.get("summary", {}), "top_groups": qa.get("top_groups", [])},
        "determinism": det,
        "honest_notes": honest,
        "ascii_safe_log_note": ASCII_SAFE_LOG_NOTE,
    }
    jd, jb = _write_json(os.path.join(out_dir, "text_variants_report.json"), report)
    md_text = TV.render_markdown(report)
    md, mb = _write_text(os.path.join(out_dir, "text_variants_report.md"), md_text)
    _log("[report] text_variants_report.json (" + str(jb) + " B, sha256=" + jd[:16] + "...)")
    _log("[report] text_variants_report.md (" + str(mb) + " B, sha256=" + md[:16] + "...)")
    _log("[report] verdict_key = " + str(report["verdict"].get("verdict_key", "")))
    _log("[report] done")
    return 0


# ---------------------------------------------------------------------------
# 分派
# ---------------------------------------------------------------------------

#: 子命令 -> 处理函数。
TEXT_VARIANTS_COMMANDS: Tuple[Tuple[str, str], ...] = (
    ("probe", "数据/依赖/变体构造/难负例/QA 成组取证（不训练）"),
    ("gate", "特征差异硬门禁 + 难负例清单复核"),
    ("eval", "零训练纯余弦检索评测（4 支 x 2 种负例设定）+ 判定"),
    ("report", "渲染 text_variants_report.{json,md}"),
)


def cmd_textvariants(args: argparse.Namespace) -> int:
    """`textvariants` 子命令分派：probe / gate / eval / report。

    **CLI 错误边界**：运行期的 `ValueError` / `KeyError` 转成**单行可读报文 +
    退码 2**；`FileNotFoundError` / `RuntimeError`（依赖缺失、模型不可用）转成
    可读报文 + 退码 1。`UnicodeEncodeError` 必须先于 `ValueError` 捕获
    （它**继承** `ValueError`），否则控制台编码故障会被误报成「非法参数」。
    """
    sub = str(getattr(args, "textvariants_cmd", "") or "probe")
    handlers = {
        "probe": cmd_probe,
        "gate": cmd_gate,
        "eval": cmd_eval,
        "report": cmd_report,
    }
    if sub not in handlers:
        _log("[textvariants] 未知子命令 " + repr(sub) + "；可用 = " + str(sorted(handlers)))
        return 2
    try:
        return int(handlers[sub](args))
    except UnicodeEncodeError as exc:
        _log(
            "[textvariants] 输出编码错误（日志含当前控制台编码无法表示的字符）："
            + type(exc).__name__ + ": " + str(exc)
            + "。这不是参数错误；可用 PYTHONIOENCODING=utf-8 重跑。"
        )
        return 1
    except FileNotFoundError as exc:
        _log("[textvariants] 输入产物缺失：" + str(exc))
        return 1
    except (ValueError, KeyError) as exc:
        _log("[textvariants] 非法参数：" + str(exc))
        return 2
    except RuntimeError as exc:
        _log("[textvariants] 运行失败（依赖 / 模型 / 产物不可用）：" + str(exc))
        return 1


class _Tee:
    """把 stdout 同时写到控制台与 **UTF-8 无 BOM** 日志文件（与 step2_run 同口径）。"""

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
        """其余属性透传（编码探测等）。"""
        return getattr(self._primary, item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口（`python -m n3d_qa_learn.text_variants_run <cmd>`）。"""
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


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="n3d_qa_learn.text_variants_run",
        description=(
            "n3d_qa_learn 第三步 Phase 1：文本侧数据改造（4 支变体 + 难负例 + 特征差异硬门禁）"
            " + 零训练纯余弦检索评测。产物一律写 "
            + DEFAULT_OUT_DIR.replace("\\", "/") + " 之下。"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in TEXT_VARIANTS_COMMANDS:
        p = sub.add_parser(name, help=help_text)
        _add_textvariants_arguments(p, subcommand=name)
        p.set_defaults(func=cmd_textvariants, textvariants_cmd=name)
    return parser


def _add_textvariants_arguments(
    p: argparse.ArgumentParser, *, subcommand: str = ""
) -> None:
    """把 ``textvariants`` 的**完整参数表**挂到给定子解析器上（**唯一**定义点）。

    存在理由：本子命令有两条入口
    （``text_variants_run`` 独立入口与 ``step2_run textvariants``），
    参数表若各写一份必然漂移。这里只写一次，两处都调它。

    参数
    ----
    p : argparse.ArgumentParser
        目标子解析器。
    subcommand : str
        子命令名；仅在 ``eval`` 上额外挂 ``--topk`` / ``--repro-dir`` /
        ``--rebuild-negatives``（其余子命令**不**接受这三个开关）。
    """
    p.add_argument("--product-dir", default="", help="n3d_qa 冻结产物目录（只读）")
    p.add_argument("--out-dir", default="", help="产物目录（默认 " + DEFAULT_OUT_DIR + "）")
    p.add_argument("--log-file", default="", help="同时写 UTF-8(无 BOM) 日志文件")
    p.add_argument("--limit", type=int, default=0,
                   help="小批演练：只取前 N 行（0 = 全量）")
    p.add_argument("--keyword-top-k", type=int, default=0,
                   help="关键词支 top-k（0 = 冻结默认 3）")
    p.add_argument("--summary-max-sentences", type=int, default=0,
                   help="摘要支句数上限（0 = 冻结默认；**仅 probe 生效**）")
    p.add_argument("--summary-max-chars", type=int, default=0,
                   help="摘要支字数上限（0 = 冻结默认；**仅 probe 生效**）")
    p.add_argument("--enc-cache-dir", default="", help="嵌入缓存目录（空 = 默认）")
    p.add_argument("--no-st-fallback", action="store_true",
                   help="禁止回退到 sentence-transformers（仓库自有编码器不可用即报错）")
    p.add_argument("--no-encoder-neighbors", action="store_true",
                   help="跳过「编码器近邻」负例构造（现场实测该步耗时最大；跳过即"
                        "不产 `encoder_neighbor` 类，须在报告中如实登记）")
    p.add_argument("--neighbor-encoder", default="", help="编码器近邻用的编码器（空 = bge-m3）")
    p.add_argument("--neighbor-source", default="", help="编码器近邻模型来源（空 = models/bge-m3）")
    p.add_argument("--encoder-topk", type=int, default=0,
                   help="编码器近邻 top-n（0 = 冻结默认 3）")
    p.add_argument("--random-negatives", type=int, default=0,
                   help="随机负例条数（0 = 冻结默认 1）")
    p.add_argument("--hard-cap", type=int, default=0,
                   help="每类难负例上限（0 = 用冻结默认 HARD_NEGATIVE_CAP）")
    p.add_argument("--qa-limit", type=int, default=0,
                   help="QA 成组取证只取前 N 条问答对（0 = 全量）")
    if str(subcommand) == "eval":
        p.add_argument("--topk", type=int, default=5, help="Recall@k 的 k")
        p.add_argument("--repro-dir", default="",
                       help="对照产物目录：给定时产出**双跑逐字节一致**证据")
        p.add_argument("--rebuild-negatives", action="store_true",
                       help="强制重抽难负例清单（默认沿用已冻结清单）")


if __name__ == "__main__":
    raise SystemExit(main())
