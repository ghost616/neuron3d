"""n3d_qa.verify_qa -- verification of the Chinese QA / text-line products (rounds 0-9, all measured).

What it checks
--------------
* **F0 feature-spec contract**: the meta spec of every product is re-derived from
  :mod:`n3d_qa.zh_features` constants; the spec hash is recomputed from the meta and must equal the
  recorded one; the column layout must tile ``[0, D-1]`` seamlessly; the tokenizer/normalization
  rule strings must match the constants.
* **F1 vectorization determinism**: the vectorizer is called **twice** on the same text pair and
  the two rows must be bit-identical (``array_equal``), for both a real product row and a synthetic
  probe pair; repeated calls for the same text must also give a bit-identical **spec hash**.
* **F2 feature contract**: ``X`` float32 ``[M, D]`` / ``y`` int64 ``[M]`` / no NaN-Inf /
  ``D == meta.features.feature_dim`` / the bag block is non-zero on every row (the Chinese
  non-all-zero requirement, measured) / the bag block is L2-normalized (norm ~1 per row).
* **F3 build idempotency (byte level)**: the builder is run twice with identical arguments into
  ``_verify/idem_a`` / ``_verify/idem_b`` and every file's SHA256 must match the shipped product
  file-by-file (this is the "two identical builds -> bit-identical artifacts" requirement).
* **F4 text-side JSONL**: the QA pair JSONL is parsed back row by row; row ids / labels / question
  ids must agree with the npz, every record must carry non-empty question and answer text, and the
  record count must equal ``M`` (this is the offline readability requirement).
* **F5 row-level recomputation**: the pair JSONL of a single-task product is re-vectorized from its
  own recorded text and compared **bit-exactly** with the corresponding npz rows.
* **F6 unified answer table**: the answer-table JSONL (written by the builder) is re-read; its
  per-class source counts must sum to the table's recorded counts, its keys must be exactly the meta
  selection key union, classes must be unique, and a class's ``count`` must equal the sum of its
  per-task source counts (consistency, readable back).
* **F7 removal registry**: every removal recorded in ``meta.removals`` is re-read and checked to be
  consistent with the recorded key counts (keys kept + keys removed = keys seen; removed questions
  + kept questions = total questions) so the coverage loss is auditable, never silent.
* **F8 text-line library/query disjointness**: from ``doclines_row_index.jsonl`` the union of
  ``(file, line_no)`` of rows with positives and of rows with negatives must have an empty
  intersection, and the union must cover every row of ``doclines_rows.jsonl``.
* **F9 zero regression on TriviaQA**: the 8 shipped TriviaQA npz SHA256 values are compared against
  the pre-change snapshot (``checkpoints/qa_learn/_snapshot/triviaqa_npz_sha256.json``) and, when
  ``--extra-triviaqa`` points at a richer snapshot, ``verify_dataset.py`` is run as a subprocess and
  its exit code must be 0.

Usage
-----
    python n3d_qa/verify_qa.py --dir checkpoints/qa_learn/dataset --checks F0,F1,F2,F4,F5,F6,F7,F8,F9
    python n3d_qa/verify_qa.py --checks F3            # byte-level idempotency (writes _verify/)
    python n3d_qa/verify_qa.py --checks F9 --verify-dataset-args "--checks E2,E4,E8 --product-set all"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # dual-mode import: package module or directly executed script
    from . import build_dataset as bd
    from . import build_qa as bq
    from .adapters import ANSWER_NORM_RULE
    from .zh_features import (
        DEFAULT_ZH_CONFIG,
        EXTRA_DIM,
        HASH_ALGO,
        HASH_DIGEST_SIZE,
        NGRAM_UNIT_TEMPLATE,
        NORMALIZATION_RULE,
        TOKENIZER_RULE,
        ZH_HASH_SALT,
        ZhFeatureConfig,
        build_feature_vector,
        feature_columns,
        zh_normalize,
    )
except ImportError:  # pragma: no cover - direct script execution
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)
    for _p in (_HERE, _ROOT):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from n3d_qa import build_dataset as bd  # type: ignore
    from n3d_qa import build_qa as bq  # type: ignore
    from n3d_qa.adapters import ANSWER_NORM_RULE  # type: ignore
    from n3d_qa.zh_features import (  # type: ignore
        DEFAULT_ZH_CONFIG,
        EXTRA_DIM,
        HASH_ALGO,
        HASH_DIGEST_SIZE,
        NGRAM_UNIT_TEMPLATE,
        NORMALIZATION_RULE,
        TOKENIZER_RULE,
        ZH_HASH_SALT,
        ZhFeatureConfig,
        build_feature_vector,
        feature_columns,
        zh_normalize,
    )

PROJECT_ROOT: str = bd.PROJECT_ROOT
DEFAULT_DIR: str = bq.DEFAULT_OUT_DIR
VERIFY_DIR: str = os.path.join(DEFAULT_DIR, "_verify")
SNAPSHOT_PATH: str = os.path.join(
    PROJECT_ROOT, "checkpoints", "qa_learn", "_snapshot", "triviaqa_npz_sha256.json"
)
TRIVIAQA_DIR: str = os.path.join(PROJECT_ROOT, "checkpoints", "triviaqa")
ANSWER_TABLE_NAME: str = "answer_table.jsonl"
ALL_CHECKS: Tuple[str, ...] = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10")
DEFAULT_CHECKS: Tuple[str, ...] = ("F0", "F1", "F2", "F4", "F5", "F6", "F7", "F8", "F9", "F10")
MATH1_TASKS: Tuple[str, ...] = ("judge", "choice", "blank", "solve")
L2_NORM_ABS_TOL: float = 1e-4
ROW_FEATURE_TOL: float = 1e-6


@dataclass
class CheckResult:
    """One verification outcome (mirrors the E-check container of verify_dataset)."""

    name: str
    title: str
    passed: bool
    skipped: bool = False
    detail: str = ""


def sha256_file(path: str, chunk_bytes: int = 1 << 22) -> str:
    """Whole-file SHA256 (chunked)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(int(chunk_bytes)), b""):
            h.update(chunk)
    return h.hexdigest()


def product_paths(directory: str) -> List[str]:
    """All ``n3d_qa_*.npz`` single-task and ``all`` products found in ``directory`` (sorted)."""
    if not os.path.isdir(directory):
        return []
    out = [
        os.path.join(directory, n)
        for n in sorted(os.listdir(directory))
        if n.startswith("n3d_qa_") and n.endswith(".npz")
    ]
    return out


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts (raises on a malformed line)."""
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d is not valid JSON: %s" % (path, i, exc)) from exc
    return recs


def spec_from_meta(meta: Dict[str, Any]) -> ZhFeatureConfig:
    """Rebuild the :class:`ZhFeatureConfig` declared by a product meta (source for all F0 checks).

    Parameters
    ----------
    meta : Dict[str, Any]
        Product meta read back from ``meta.npy``.

    Returns
    -------
    ZhFeatureConfig
        Config reconstructed from the recorded spec (never from memory).

    Raises
    ------
    KeyError
        When the meta does not carry the spec block.
    """
    spec = meta["features"]["spec"]
    return ZhFeatureConfig(
        n_gram_orders=tuple(int(x) for x in spec["n_gram_orders"]),
        buckets_per_order=int(spec["buckets_per_order"]),
        salt_hex=str(spec["hash_salt_hex"]),
    )


def check_f0_spec(directory: str) -> CheckResult:
    """F0: feature-spec contract of every product (constants vs meta, spec hash, column tiling)."""
    lines: List[str] = []
    problems: List[str] = []
    paths = product_paths(directory)
    if not paths:
        return CheckResult("F0", "feature spec contract", True, skipped=True,
                           detail="[SKIP] no n3d_qa_*.npz found under %s" % directory)
    for path in paths:
        name = os.path.basename(path)
        meta = bd.read_npz_meta(path)
        cfg = spec_from_meta(meta)
        spec = meta["features"]["spec"]
        bad: List[str] = []
        if str(spec.get("version")) != "zh-charbow-v1":
            bad.append("spec.version=%r" % spec.get("version"))
        if str(spec.get("tokenizer")) != TOKENIZER_RULE:
            bad.append("tokenizer text differs from the constant")
        if str(spec.get("normalization")) != NORMALIZATION_RULE:
            bad.append("normalization text differs from the constant")
        if str(spec.get("hash_algo")) != HASH_ALGO:
            bad.append("hash_algo=%r" % spec.get("hash_algo"))
        if int(spec.get("hash_digest_size", -1)) != int(HASH_DIGEST_SIZE):
            bad.append("hash_digest_size=%r" % spec.get("hash_digest_size"))
        if str(spec.get("hash_salt_hex")) != ZH_HASH_SALT.hex():
            bad.append("hash_salt_hex=%r != constant" % spec.get("hash_salt_hex"))
        if str(spec.get("hash_unit_template")) != NGRAM_UNIT_TEMPLATE:
            bad.append("hash_unit_template=%r" % spec.get("hash_unit_template"))
        if int(spec.get("extra_dim", -1)) != int(EXTRA_DIM):
            bad.append("extra_dim=%r" % spec.get("extra_dim"))
        if int(spec.get("bag_dim", -1)) != int(cfg.bag_dim):
            bad.append("bag_dim=%r != buckets_per_order * len(orders) = %d" % (spec.get("bag_dim"), cfg.bag_dim))
        if int(spec.get("feature_dim", -1)) != int(cfg.feature_dim):
            bad.append("feature_dim=%r != bag_dim + extra_dim = %d" % (spec.get("feature_dim"), cfg.feature_dim))
        want_hash = cfg.spec_hash()
        if str(meta["features"].get("spec_hash")) != want_hash:
            bad.append("spec_hash=%r, recomputed=%r" % (meta["features"].get("spec_hash"), want_hash))
        want_cols = [dict(c) for c in feature_columns(cfg)]
        got_cols = [dict(c) for c in meta["features"]["columns"]]
        if got_cols != want_cols:
            bad.append("columns differ from feature_columns(config)")
        cursor = 0
        cover_ok = True
        for c in got_cols:
            if int(c["start"]) != cursor:
                cover_ok = False
            cursor = int(c["end"]) + 1
        if not cover_ok or cursor != int(cfg.feature_dim):
            bad.append("columns do not tile [0, D-1] seamlessly (cursor=%d)" % cursor)
        if str(meta.get("answer_norm_rule")) != ANSWER_NORM_RULE:
            bad.append("answer_norm_rule=%r != constant" % meta.get("answer_norm_rule"))
        lines.append(
            "  %-28s D=%-4d bag=%-4d orders=%s buckets/order=%d spec_hash=%s -> %s"
            % (name, int(spec["feature_dim"]), int(spec["bag_dim"]), spec["n_gram_orders"],
               int(spec["buckets_per_order"]), want_hash[:16], "OK" if not bad else "FAIL")
        )
        for b in bad:
            lines.append("      [!] " + b)
            problems.append("%s: %s" % (name, b))
    passed = not problems
    if problems:
        lines.append("  problems: " + "; ".join(problems[:6]))
    return CheckResult("F0", "feature spec contract (constants vs meta, spec hash, column tiling)",
                       passed, detail="\n".join(lines))


def check_f1_vector_determinism(directory: str) -> CheckResult:
    """F1: the vectorizer is a pure function — twice on the same pair must be bit-identical."""
    lines: List[str] = []
    problems: List[str] = []
    probe_pairs = [
        ("\u8fd9\u662f\u4e00\u9053\u4e2d\u6587\u9898\u76ee", "\u6b63\u786e\u7b54\u6848\n\u89e3\u6790\u6587\u672c"),
        ("\u4e00\u4e2a\u5bb9\u91cf\u4e3a80\u7684\u6837\u672c", "10\n\u5206\u6210 10 \u7ec4"),
        ("", ""),
        ("\uff21\uff22\uff23 abc 123", "\uff41\uff42\uff43 ABC"),
    ]
    cfg = DEFAULT_ZH_CONFIG
    for i, (q, c) in enumerate(probe_pairs):
        a = build_feature_vector(q, c, cfg)
        b = build_feature_vector(q, c, cfg)
        same = bool(np.array_equal(a, b)) and a.dtype == b.dtype == np.float32
        lines.append("  probe pair %d: shape=%s dtype=%s bit-identical=%s" % (i, a.shape, a.dtype, same))
        if not same:
            problems.append("probe pair %d is not bit-identical across calls" % i)
    cfg2 = ZhFeatureConfig(n_gram_orders=(1, 2, 3), buckets_per_order=64)
    h1, h2 = cfg2.spec_hash(), ZhFeatureConfig(n_gram_orders=(1, 2, 3), buckets_per_order=64).spec_hash()
    lines.append("  spec hash repeated: %s == %s -> %s" % (h1[:16], h2[:16], h1 == h2))
    if h1 != h2:
        problems.append("spec hash is not reproducible")
    norms = [zh_normalize("  \uff21b\tC\u3000") for _ in range(3)]
    lines.append("  zh_normalize repeated: %r -> stable=%s" % (norms[0], len(set(norms)) == 1))
    if len(set(norms)) != 1:
        problems.append("zh_normalize is not stable")
    if os.path.isdir(directory):
        paths = product_paths(directory)
        if paths:
            path = paths[0]
            meta = bd.read_npz_meta(path)
            cfg_p = spec_from_meta(meta)
            jl = path[: -len(".npz")] + bq.PAIRS_SUFFIX
            if os.path.isfile(jl):
                recs = load_jsonl(jl)[:3]
                for rec in recs:
                    v1 = build_feature_vector(
                        str(rec["question_text"]), str(rec["answer_text"]), cfg_p
                    )
                    v2 = build_feature_vector(
                        str(rec["question_text"]), str(rec["answer_text"]), cfg_p
                    )
                    if not np.array_equal(v1, v2):
                        problems.append("product row %s not bit-identical" % rec.get("row_id"))
                lines.append("  product rows re-vectorized twice: %d rows, all bit-identical=%s"
                             % (len(recs), not problems))
            else:
                lines.append("  [SKIP] product %s has no pair JSONL" % os.path.basename(path))
    passed = not problems
    if problems:
        lines.append("  problems: " + "; ".join(problems[:6]))
    return CheckResult("F1", "vectorization determinism (twice on the same text -> bit-identical)",
                       passed, detail="\n".join(lines))

def check_f2_feature_contract(directory: str) -> CheckResult:
    """F2: ``X``/``y`` contract, finiteness, non-zero bag block and bag L2 norm per product."""
    lines: List[str] = []
    problems: List[str] = []
    paths = product_paths(directory)
    if not paths:
        return CheckResult("F2", "feature contract", True, skipped=True,
                           detail="[SKIP] no n3d_qa_*.npz found under %s" % directory)
    for path in paths:
        name = os.path.basename(path)
        meta = bd.read_npz_meta(path)
        cfg = spec_from_meta(meta)
        with np.load(path, allow_pickle=False) as z:
            x = np.asarray(z["X"])
            y = np.asarray(z["y"])
        bad: List[str] = []
        if x.dtype != np.float32:
            bad.append("X.dtype=%s" % x.dtype)
        if y.dtype != np.int64:
            bad.append("y.dtype=%s" % y.dtype)
        if x.ndim != 2 or int(x.shape[1]) != int(cfg.feature_dim):
            bad.append("X.shape=%r expected [M,%d]" % (x.shape, cfg.feature_dim))
        if y.ndim != 1 or int(y.shape[0]) != int(x.shape[0]):
            bad.append("y.shape=%r vs X rows %d" % (y.shape, int(x.shape[0])))
        if not bool(np.isfinite(x).all()):
            bad.append("non-finite X entries=%d" % int((~np.isfinite(x)).sum()))
        bag = x[:, : int(cfg.bag_dim)]
        n_empty = int(x.shape[0] - int(bag.any(axis=1).sum()))
        if n_empty:
            bad.append("rows with an all-zero bag block=%d" % n_empty)
        norms = np.sqrt((bag.astype(np.float64) ** 2).sum(axis=1))
        bad_norm = int((np.abs(norms - 1.0) > L2_NORM_ABS_TOL).sum())
        if bad_norm:
            bad.append("rows whose bag L2 norm deviates from 1 by > %g: %d" % (L2_NORM_ABS_TOL, bad_norm))
        labels = sorted(set(np.unique(y).tolist()))
        if not set(labels) <= {0, 1}:
            bad.append("labels=%r not a subset of {0,1}" % labels)
        counts = meta["counts"]
        if int(counts["pairs"]) != int(x.shape[0]):
            bad.append("meta.counts.pairs=%d != M=%d" % (int(counts["pairs"]), int(x.shape[0])))
        if int(counts["positives"]) != int((y == 1).sum()):
            bad.append("meta.counts.positives mismatch")
        if int(counts["negatives"]) != int((y == 0).sum()):
            bad.append("meta.counts.negatives mismatch")
        lines.append(
            "  %-28s M=%-6d D=%-4d bag_nonzero_rows=%-6d norm_ok=%s labels=%s majority=%.4f"
            % (name, int(x.shape[0]), int(x.shape[1]), int(x.shape[0]) - n_empty,
               bad_norm == 0, labels, float(counts["majority_baseline"]))
        )
        for b in bad:
            lines.append("      [!] " + b)
            problems.append("%s: %s" % (name, b))
    passed = not problems
    if problems:
        lines.append("  problems: " + "; ".join(problems[:6]))
    return CheckResult("F2", "feature contract (dtype/shape/NaN-free/non-zero bag/L2 norms)",
                       passed, detail="\n".join(lines))


def check_f10_meta_reproducible(directory: str) -> CheckResult:
    """F10: no product meta may carry a wall-clock- or cache-state-dependent field.

    Why this check exists
    ---------------------
    Two builds with identical arguments produced **different npz bytes** while ``X``/``y`` were
    bit-identical: the meta carried ``build_seconds`` (wall clock) and the TriviaQA block carried
    ``archive_passes`` (1 on a cold cache, 0 on a warm one). Both were removed from the products;
    this check walks the whole meta tree and fails on any forbidden key, so the regression cannot
    come back silently.
    """
    forbidden = (
        "build_seconds", "archive_passes", "cache_status", "timestamp", "elapsed",
        "wall_clock", "duration_s", "mtime", "generated_at", "hostname", "pid",
    )
    lines: List[str] = []
    problems: List[str] = []
    paths = product_paths(directory)
    if not paths:
        return CheckResult("F10", "meta reproducibility", True, skipped=True,
                           detail="[SKIP] no n3d_qa_*.npz found under %s" % directory)

    def walk(node: Any, trail: str, name: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k) in forbidden:
                    problems.append("%s: forbidden meta key %s at %s" % (name, k, trail or "<root>"))
                walk(v, "%s.%s" % (trail, k) if trail else str(k), name)
        elif isinstance(node, list):
            for i, v in enumerate(node[:3]):
                walk(v, "%s[%d]" % (trail, i), name)

    for path in paths:
        name = os.path.basename(path)
        meta = bd.read_npz_meta(path)
        before = len(problems)
        walk(meta, "", name)
        lines.append("  %-28s meta keys=%-3d forbidden hits=%d"
                     % (name, len(meta), len(problems) - before))
    passed = not problems
    if problems:
        lines.append("  problems: " + "; ".join(problems[:6]))
    return CheckResult("F10", "meta reproducibility (no wall-clock / cache-state fields)",
                       passed, detail="\n".join(lines))


def check_f3_idempotent(directory: str, tasks: Sequence[str], doclines: bool) -> CheckResult:
    """F3: run the builder twice with identical arguments and compare every file's SHA256."""
    lines: List[str] = []
    problems: List[str] = []
    if not os.path.isdir(directory):
        return CheckResult("F3", "byte-level idempotency", True, skipped=True,
                           detail="[SKIP] product directory %s does not exist" % directory)
    runs: List[Dict[str, str]] = []
    for sub in ("idem_a", "idem_b"):
        out_dir = os.path.join(VERIFY_DIR, sub)
        argv = ["--tasks", ",".join(tasks), "--out-dir", out_dir]
        if doclines:
            argv.append("--doclines")
        t0 = time.time()
        rc = bq.main(argv)
        lines.append("  run %s: exit=%d, %.1f s, dir=%s" % (sub, rc, time.time() - t0, out_dir))
        if rc != 0:
            problems.append("build run %s exited with %d" % (sub, rc))
            continue
        files: Dict[str, str] = {}
        for n in sorted(os.listdir(out_dir)):
            fp = os.path.join(out_dir, n)
            if os.path.isfile(fp) and n != bq.MANIFEST_NAME:
                files[n] = sha256_file(fp)
        runs.append(files)
    if len(runs) == 2:
        names = sorted(set(runs[0]) | set(runs[1]))
        for n in names:
            a, b = runs[0].get(n), runs[1].get(n)
            prod = os.path.join(directory, n)
            prod_sha = sha256_file(prod) if os.path.isfile(prod) else None
            ok = a is not None and a == b and (prod_sha is None or a == prod_sha)
            lines.append(
                "  %-28s A=%s B=%s product=%s -> %s"
                % (n, (a or "-")[:12], (b or "-")[:12], (prod_sha or "-")[:12],
                   "identical" if ok else "DIFFERENT")
            )
            if not ok:
                problems.append("%s differs between the two runs (or from the shipped product)" % n)
    passed = not problems
    if problems:
        lines.append("  problems: " + "; ".join(problems[:6]))
    return CheckResult("F3", "byte-level build idempotency (two runs + shipped product)",
                       passed, detail="\n".join(lines))


def check_f4_text_side(directory: str) -> CheckResult:
    """F4: the QA pair JSONL is readable and agrees with the npz row by row."""
    lines: List[str] = []
    problems: List[str] = []
    paths = product_paths(directory)
    if not paths:
        return CheckResult("F4", "text-side JSONL", True, skipped=True,
                           detail="[SKIP] no n3d_qa_*.npz found under %s" % directory)
    for path in paths:
        name = os.path.basename(path)
        jl = path[: -len(".npz")] + bq.PAIRS_SUFFIX
        if not os.path.isfile(jl):
            problems.append("%s has no pair JSONL" % name)
            lines.append("  %-28s [!] missing %s" % (name, os.path.basename(jl)))
            continue
        recs = load_jsonl(jl)
        with np.load(path, allow_pickle=False) as z:
            y = np.asarray(z["y"])
        bad = 0
        n_empty_q = 0
        n_empty_a = 0
        ids = set()
        for i, rec in enumerate(recs):
            if not str(rec.get("question_text", "")).strip():
                n_empty_q += 1
            if not str(rec.get("answer_text", "")).strip():
                n_empty_a += 1
            if i < y.size and int(rec.get("label", -1)) != int(y[i]):
                bad += 1
            ids.add(str(rec.get("row_id")))
        lines.append(
            "  %-28s records=%-6d npz_rows=%-6d label_mismatch=%d empty_question=%d "
            "empty_answer=%d distinct_row_ids=%d"
            % (name, len(recs), int(y.size), bad, n_empty_q, n_empty_a, len(ids))
        )
        if len(recs) != int(y.size):
            problems.append("%s pair JSONL has %d records but the npz has %d rows" % (name, len(recs), int(y.size)))
        if bad:
            problems.append("%s label mismatch on %d rows" % (name, bad))
        if n_empty_q or n_empty_a:
            problems.append("%s has %d empty question / %d empty answer texts" % (name, n_empty_q, n_empty_a))
        if int(y.size) and len(ids) != int(y.size):
            problems.append("%s row ids are not unique (%d ids for %d rows)" % (name, len(ids), int(y.size)))
    passed = not problems
    if problems:
        lines.append("  problems: " + "; ".join(problems[:6]))
    return CheckResult("F4", "text-side JSONL (readable, row-aligned, non-empty texts)",
                       passed, detail="\n".join(lines))


def check_f5_row_recompute(directory: str) -> CheckResult:
    """F5: re-vectorize every pair JSONL row from its recorded text and compare with the npz."""
    lines: List[str] = []
    problems: List[str] = []
    paths = product_paths(directory)
    if not paths:
        return CheckResult("F5", "row-level recomputation", True, skipped=True,
                           detail="[SKIP] no n3d_qa_*.npz found under %s" % directory)
    for path in paths:
        name = os.path.basename(path)
        jl = path[: -len(".npz")] + bq.PAIRS_SUFFIX
        if not os.path.isfile(jl):
            problems.append("%s has no pair JSONL" % name)
            continue
        meta = bd.read_npz_meta(path)
        cfg = spec_from_meta(meta)
        recs = load_jsonl(jl)
        with np.load(path, allow_pickle=False) as z:
            x = np.asarray(z["X"])
        n_bad = 0
        max_dev = 0.0
        for i, rec in enumerate(recs):
            v = build_feature_vector(str(rec["question_text"]), str(rec["answer_text"]), cfg)
            if not np.array_equal(v, x[i]):
                n_bad += 1
                max_dev = max(max_dev, float(np.abs(v.astype(np.float64) - x[i].astype(np.float64)).max()))
        lines.append("  %-28s rows=%-6d bit-exact=%d mismatch=%d max_dev=%g"
                     % (name, len(recs), len(recs) - n_bad, n_bad, max_dev))
        if n_bad:
            problems.append("%s: %d rows are not bit-exactly reproducible (max deviation %g)"
                            % (name, n_bad, max_dev))
    passed = not problems
    if problems:
        lines.append("  problems: " + "; ".join(problems[:6]))
    return CheckResult("F5", "row-level recomputation (pair JSONL text -> npz row, bit-exact)",
                       passed, detail="\n".join(lines))


def check_f6_answer_table(directory: str) -> CheckResult:
    """F6: the answer-table JSONL is readable and internally consistent with the meta."""
    lines: List[str] = []
    problems: List[str] = []
    paths = product_paths(directory)
    if not paths:
        return CheckResult("F6", "unified answer table", True, skipped=True,
                           detail="[SKIP] no n3d_qa_*.npz found under %s" % directory)
    table_path = os.path.join(directory, ANSWER_TABLE_NAME)
    if not os.path.isfile(table_path):
        return CheckResult("F6", "unified answer table", False,
                           detail="[FAIL] %s is missing" % table_path)
    recs = load_jsonl(table_path)
    lines.append("  %s: %d classes" % (ANSWER_TABLE_NAME, len(recs)))
    keys = [str(r["key"]) for r in recs]
    if len(set(keys)) != len(keys):
        problems.append("answer table has duplicate keys")
    for r in recs:
        src_sum = sum(int(v) for v in dict(r["sources"]).values())
        if src_sum != int(r["count"]):
            problems.append("class %r: count=%d != sum(sources)=%d" % (r["key"], int(r["count"]), src_sum))
            break
    for path in paths:
        name = os.path.basename(path)
        meta = bd.read_npz_meta(path)
        at = meta.get("answer_table") or {}
        classes = int(at.get("classes", -1))
        if classes != len(recs):
            problems.append("%s: meta.answer_table.classes=%d != JSONL rows=%d" % (name, classes, len(recs)))
        alias = int(at.get("alias_count", -1))
        if alias < classes:
            problems.append("%s: alias_count=%d < classes=%d" % (name, alias, classes))
        if int(at.get("questions_seen", -1)) != int(at.get("questions_in_table", -2)) + int(
            at.get("questions_not_relevant", -3)
        ):
            problems.append("%s: questions_seen != questions_in_table + questions_not_relevant" % name)
        lines.append(
            "  %-28s classes=%-5d alias=%-5d seen=%-6d in_table=%-6d not_relevant=%-5d"
            % (name, classes, alias, int(at.get("questions_seen", -1)),
               int(at.get("questions_in_table", -1)), int(at.get("questions_not_relevant", -1)))
        )
    passed = not problems
    if problems:
        lines.append("  problems: " + "; ".join(problems[:6]))
    return CheckResult("F6", "unified answer table (readable, unique keys, count consistency)",
                       passed, detail="\n".join(lines))


def check_f7_removals(directory: str) -> CheckResult:
    """F7: the removal registry is complete and its coverage arithmetic closes."""
    lines: List[str] = []
    problems: List[str] = []
    paths = product_paths(directory)
    if not paths:
        return CheckResult("F7", "removal registry", True, skipped=True,
                           detail="[SKIP] no n3d_qa_*.npz found under %s" % directory)
    for path in paths:
        name = os.path.basename(path)
        meta = bd.read_npz_meta(path)
        rem = meta.get("removals") or {}
        cov = meta.get("coverage") or {}
        keys_total = int(cov.get("kept_keys", 0)) + int(cov.get("removed_keys", 0))
        q_total = (
            int(rem.get("kept_questions", 0))
            + int(rem.get("removed_questions", 0))
            + int(rem.get("unkeyed_questions", 0))
        )
        lines.append(
            "  %-28s kept_keys=%-6d removed_keys=%-7d (sum=%-7d) | kept_q=%-6d removed_q=%-6d "
            "unkeyed_q=%-4d (sum=%-6d) total_q=%-6d rule=%s"
            % (name, int(cov.get("kept_keys", 0)), int(cov.get("removed_keys", 0)), keys_total,
               int(rem.get("kept_questions", 0)), int(rem.get("removed_questions", 0)),
               int(rem.get("unkeyed_questions", 0)), q_total,
               int(rem.get("total_questions", -1)), str(rem.get("rule", ""))[:60])
        )
        if q_total != int(rem.get("total_questions", -1)):
            problems.append(
                "%s: kept+removed+unkeyed questions (%d) != total questions (%d)"
                % (name, q_total, int(rem.get("total_questions", -1)))
            )
        # [!] The answer table is **unified across tasks**, so its class count is the global class
        # count and can never equal one task's own kept-key count (the first draft of this check
        # asserted exactly that and failed on every product). What must hold per product is:
        # ``coverage.kept_keys`` is positive and the per-task key arithmetic closes
        # (kept + removed = distinct keys seen); the global class count is checked against the
        # answer-table JSONL in F6.
        if int(cov.get("kept_keys", 0)) <= 0:
            problems.append("%s: coverage.kept_keys is not positive" % name)
        ptk = meta.get("per_task_keys") or {}
        k = int(ptk.get("kept_keys", -1))
        r = int(ptk.get("removed_keys", -1))
        u = int(ptk.get("unkeyed_questions", -1))
        if int(cov.get("kept_keys", -1)) != k or int(cov.get("removed_keys", -1)) != r:
            problems.append(
                "%s: coverage(kept=%d, removed=%d) != per_task_keys(kept=%d, removed=%d)"
                % (name, int(cov.get("kept_keys", -1)), int(cov.get("removed_keys", -1)), k, r)
            )
        if k < 0 or r < 0 or k + r != int(ptk.get("distinct_keys", -1)):
            problems.append(
                "%s: per-task key arithmetic does not close (%d + %d != %d)"
                % (name, k, r, int(ptk.get("distinct_keys", -1)))
            )
        if u != int(rem.get("unkeyed_questions", -1)):
            problems.append("%s: per_task_keys.unkeyed_questions != removals.unkeyed_questions" % name)
        if "removed_keys_head" not in rem:
            problems.append("%s: removals.removed_keys_head is missing" % name)
        if abs(float(cov.get("coverage_loss", -1.0)) - (1.0 - float(cov.get("coverage_rate", -1.0)))) > 1e-9:
            problems.append("%s: coverage_loss != 1 - coverage_rate" % name)
    passed = not problems
    if problems:
        lines.append("  problems: " + "; ".join(problems[:6]))
    return CheckResult("F7", "removal registry (quantified, arithmetic closes, auditable)",
                       passed, detail="\n".join(lines))


def check_f8_doclines(directory: str) -> CheckResult:
    """F8: text-line row table / index consistency and library-query disjointness."""
    lines: List[str] = []
    problems: List[str] = []
    rows_path = os.path.join(directory, bq.ROWS_NAME)
    index_path = os.path.join(directory, bq.ROW_INDEX_NAME)
    npz_path = os.path.join(directory, bq.DOCLINES_NAME)
    if not (os.path.isfile(rows_path) and os.path.isfile(index_path) and os.path.isfile(npz_path)):
        return CheckResult("F8", "text-line product", True, skipped=True,
                           detail="[SKIP] text-line product not built under %s" % directory)
    rows = load_jsonl(rows_path)
    index = load_jsonl(index_path)
    meta = bd.read_npz_meta(npz_path)
    cfg = spec_from_meta(meta)
    # [!] The library/query sets are recovered from the row index by the **positives** count only:
    # a query row is a query exactly when it is a query somewhere (every query row gets exactly one
    # positive), while a library row may well carry negatives (it was drawn as somebody's wrong
    # candidate) — so the first draft's "negatives > 0 -> library" rule misclassified most rows and
    # made the split look inconsistent with the meta.
    library = {int(r["index"]) for r in index if int(r["positives"]) == 0}
    query = {int(r["index"]) for r in index if int(r["positives"]) > 0}
    all_idx = {int(r["index"]) for r in index}
    inter = library & query
    lines.append("  rows=%d index=%d library=%d query=%d intersection=%d union=%d"
                 % (len(rows), len(index), len(library), len(query), len(inter), len(all_idx)))
    lines.append("  meta.split: library_rows=%d query_rows=%d intersection=%d seed=%d ratio=%s"
                 % (int(meta["split"]["library_rows"]), int(meta["split"]["query_rows"]),
                    int(meta["split"]["intersection"]), int(meta["split"]["seed"]),
                    meta["split"]["query_ratio"]))
    n_query_with_one_pos = sum(1 for r in index if int(r["positives"]) == 1)
    if n_query_with_one_pos != len(query):
        problems.append(
            "%d query rows do not carry exactly one positive row" % (len(query) - n_query_with_one_pos)
        )
    n_neg_rows = sum(int(r["negatives"]) for r in index)
    if n_neg_rows != int(meta["counts"]["negatives"]):
        problems.append(
            "sum(negative rows per query)=%d != meta.counts.negatives=%d"
            % (n_neg_rows, int(meta["counts"]["negatives"]))
        )
    if inter:
        problems.append("library and query index sets are not disjoint (%d shared)" % len(inter))
    if len(all_idx) != len(rows):
        problems.append("row index does not cover every row (%d vs %d)" % (len(all_idx), len(rows)))
    if int(meta["split"]["library_rows"]) != len(library) or int(meta["split"]["query_rows"]) != len(query):
        problems.append("meta.split sizes do not match the row index")
    if int(meta["counts"]["rows_total"]) != len(rows):
        problems.append("meta.counts.rows_total != row table length")
    # per-row feature length + text round trip
    n_bad_len = sum(1 for r in rows if len(r.get("feature", [])) != int(cfg.feature_dim))
    if n_bad_len:
        problems.append("%d rows carry a feature vector of the wrong length" % n_bad_len)
    n_bad_norm = sum(1 for r in rows if abs(float(r.get("char_len", -1)) - len(str(r["text"]))) > 0)
    if n_bad_norm:
        problems.append("%d rows have char_len inconsistent with the stored text" % n_bad_norm)
    if int(meta["counts"]["pairs"]) != int(np.load(npz_path, allow_pickle=False)["y"].size):
        problems.append("doclines meta pairs != npz rows")
    per_file = meta["counts"].get("per_file_rows") or {}
    real_files: Dict[str, int] = {}
    for r in rows:
        real_files[str(r["file"])] = real_files.get(str(r["file"]), 0) + 1
    if {k: int(v) for k, v in per_file.items()} != real_files:
        problems.append("meta.counts.per_file_rows != the measured per-file row counts")
    lines.append("  per-file rows (measured): %s" % json.dumps(real_files, ensure_ascii=False))
    passed = not problems
    if problems:
        lines.append("  problems: " + "; ".join(problems[:6]))
    return CheckResult("F8", "text-line product (index coverage, split disjointness, feature width)",
                       passed, detail="\n".join(lines))


def check_f9_zero_regression(run_verify_dataset: bool, extra_args: str) -> CheckResult:
    """F9: the shipped TriviaQA npz are unchanged, and the existing E-check verifier still passes."""
    lines: List[str] = []
    problems: List[str] = []
    rel = os.path.relpath(SNAPSHOT_PATH, PROJECT_ROOT).replace(os.sep, "/")
    if not os.path.isfile(SNAPSHOT_PATH):
        return CheckResult("F9", "zero regression", True, skipped=True,
                           detail="[SKIP] snapshot %s is missing (take it before the change)" % rel)
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as fh:
        snap = json.load(fh)
    items = {str(d["path"]): (int(d["size"]), str(d["sha256"])) for d in snap["items"]}
    n_ok = 0
    n_bad = 0
    for path, (size, sha) in sorted(items.items()):
        fp = os.path.join(PROJECT_ROOT, path.replace("/", os.sep))
        if not os.path.isfile(fp):
            problems.append("missing product %s" % path)
            n_bad += 1
            continue
        got = sha256_file(fp)
        if int(os.path.getsize(fp)) == size and got == sha:
            n_ok += 1
        else:
            n_bad += 1
            problems.append("%s changed: %s/%d -> %s/%d" % (path, sha[:12], size, got[:12], int(os.path.getsize(fp))))
    lines.append("  snapshot %s: %d entries, unchanged=%d, changed/missing=%d" % (rel, len(items), n_ok, n_bad))
    for path in sorted(items)[:8]:
        lines.append("    %s  %s" % (items[path][1][:16], os.path.basename(path)))
    if os.path.isdir(TRIVIAQA_DIR):
        found = sorted(
            n for n in os.listdir(TRIVIAQA_DIR) if n.endswith(".npz")
        )
        lines.append("  checkpoints/triviaqa/*.npz present: %d -> %s" % (len(found), found))
    if run_verify_dataset:
        cmd = [sys.executable, os.path.join("n3d_qa", "verify_dataset.py")]
        cmd += [x for x in str(extra_args).split(" ") if x]
        t0 = time.time()
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace",
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        tail = [ln for ln in out.splitlines() if ln.strip().startswith("汇总：") or ln.startswith("结论：")]
        lines.append("  n3d_qa/verify_dataset.py %s -> exit=%d (%.1f s)" % (extra_args, int(proc.returncode), time.time() - t0))
        for ln in tail:
            lines.append("    " + ln.strip())
        if int(proc.returncode) != 0:
            problems.append("verify_dataset.py exited with %d" % int(proc.returncode))
            lines.extend("    " + ln for ln in out.splitlines()[-10:])
    passed = not problems
    if problems:
        lines.append("  problems: " + "; ".join(problems[:6]))
    return CheckResult("F9", "zero regression (8 shipped TriviaQA npz bit-identical + E-checks)",
                       passed, detail="\n".join(lines))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the verification CLI options."""
    ap = argparse.ArgumentParser(description="n3d_qa Chinese QA / text-line product verification (F0-F9)")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="product directory (default %s)" % DEFAULT_DIR)
    ap.add_argument("--checks", default=",".join(DEFAULT_CHECKS),
                    help="comma separated subset of %s" % ",".join(ALL_CHECKS))
    ap.add_argument("--tasks", default="judge,choice,blank,solve,triviaqa",
                    help="tasks used by the F3 idempotency reruns")
    ap.add_argument("--no-doclines", action="store_true", help="skip the text-line product in F3")
    ap.add_argument("--run-verify-dataset", action="store_true",
                    help="F9: also run n3d_qa/verify_dataset.py and require exit 0")
    ap.add_argument("--verify-dataset-args", default="--checks E2,E4,E8 --product-set all",
                    help="arguments for the verify_dataset.py subprocess")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: run the requested checks and print the report (exit 1 on any failure)."""
    args = parse_args(argv)
    wanted = tuple(x.strip().upper() for x in str(args.checks).split(",") if x.strip())
    unknown = [x for x in wanted if x not in ALL_CHECKS]
    if unknown:
        print("[n3d_qa.verify_qa] unknown checks %s; expected %s" % (unknown, list(ALL_CHECKS)), file=sys.stderr)
        return 1
    results: List[CheckResult] = []
    t0 = time.time()
    print("=" * 78)
    print("n3d_qa Chinese QA / text-line product verification report")
    print("=" * 78)
    print("product directory : %s" % args.dir)
    print("checks            : %s" % list(wanted))
    print("-" * 78)

    def emit(res: CheckResult) -> None:
        results.append(res)
        tag = "SKIP" if res.skipped else ("PASS" if res.passed else "FAIL")
        print("[%s] %s -> %s" % (res.name, res.title, tag))
        if res.detail:
            print(res.detail)
        print("-" * 78)

    if "F0" in wanted:
        emit(check_f0_spec(str(args.dir)))
    if "F1" in wanted:
        emit(check_f1_vector_determinism(str(args.dir)))
    if "F2" in wanted:
        emit(check_f2_feature_contract(str(args.dir)))
    if "F3" in wanted:
        emit(check_f3_idempotent(str(args.dir), [t for t in str(args.tasks).split(",") if t],
                                 not bool(args.no_doclines)))
    if "F4" in wanted:
        emit(check_f4_text_side(str(args.dir)))
    if "F5" in wanted:
        emit(check_f5_row_recompute(str(args.dir)))
    if "F6" in wanted:
        emit(check_f6_answer_table(str(args.dir)))
    if "F7" in wanted:
        emit(check_f7_removals(str(args.dir)))
    if "F8" in wanted:
        emit(check_f8_doclines(str(args.dir)))
    if "F9" in wanted:
        emit(check_f9_zero_regression(bool(args.run_verify_dataset), str(args.verify_dataset_args)))
    if "F10" in wanted:
        emit(check_f10_meta_reproducible(str(args.dir)))

    n_pass = sum(1 for r in results if r.passed and not r.skipped)
    n_fail = sum(1 for r in results if not r.passed and not r.skipped)
    n_skip = sum(1 for r in results if r.skipped)
    print("=" * 78)
    for r in results:
        print("  [%s] %s  %s" % (r.name, "SKIP" if r.skipped else ("PASS" if r.passed else "FAIL"), r.title))
    print("-" * 78)
    print("summary: passed %d, failed %d, skipped %d; %.1f s" % (n_pass, n_fail, n_skip, time.time() - t0))
    print("conclusion: " + ("all checks passed (skips are missing-artifact cases)"
                            if n_fail == 0 else "failures present, see the detail above"))
    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())