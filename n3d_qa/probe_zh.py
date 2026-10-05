"""n3d_qa.probe_zh -- Chinese feature discrimination probe (the hard gate before a full build).

What it answers
---------------
"Are the Chinese character n-gram features non-degenerate and discriminative on real Chinese data?"
It reports, for every evaluation set, the measured share of all-zero rows and the 5-fold CV
readings, then applies a gate. It never claims more than it measures.

Evaluation sets
---------------
* ``lexical`` (**gated**): the text-line semantic of the product, on real ``data/doc`` lines — each
  held-out **query** line is paired with the positive candidate = that line itself and one negative
  candidate drawn from the held-out **library** (a different line, sometimes with identical text).
  This is genuine lexical matching over 2 665 real Chinese lines (the query/library split uses the
  product's own seed), so it measures exactly what a character n-gram bag can and must capture.
* ``math1judge`` (**reported, not gated**): the QA match row semantics of the Math1 product — every
  boolean question against its correct answer plus wrong answers drawn from the same table.
  **Measured finding (this is the honest reading, not a polishing):** this set sits near chance.
  Math1 questions are self-contained problems whose answers are *computed*, so the correct answer
  carries no lexical overlap with the question text; with only two answer classes (correct / wrong)
  there is nothing for a lexical bag to learn. The reading is reported so nobody mistakes the Math1
  QA match product for a lexically solvable task.
* ``doclines`` (**gated together with lexical**): the text-line product itself
  (:func:`n3d_qa.build_qa.assemble_doclines`), i.e. the library/query held-out match rows that are
  actually written to ``doclines.npz``.

Metrics
-------
5-fold CV with the **same convention as the existing E7 check**: the fold permutation and the
L2-regularized full-batch logistic regression are imported verbatim from
:mod:`n3d_qa.verify_dataset` (``_cv_scores`` / ``_fit_logistic``), so "same convention as E7" is
enforced by code rather than asserted in prose.

Gate
----
``--gate`` (default on) fails the run when any gated set shows

* an all-zero bag row share above ``--max-empty-frac`` (default 0.01), or
* a bag-only CV AUC below ``--min-auc`` (default 0.75).

Usage
-----
    python n3d_qa/probe_zh.py
    python n3d_qa/probe_zh.py --sets lexical,doclines,math1judge
    python n3d_qa/probe_zh.py --sweep --sweep-buckets 32,64,100,128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # dual-mode import: package module or directly executed script
    from .adapters import (
        build_answer_table,
        build_qa_pairs,
        load_doc_lines,
        load_math1_task,
        split_library_query,
        task_class_mask,
    )
    from .build_qa import assemble_doclines
    from .zh_features import (
        TEXT_BUCKETS_PER_ORDER,
        ZhFeatureConfig,
        build_feature_vector,
        feature_columns,
        ngram_universe,
    )
    from .verify_dataset import _cv_scores
except ImportError:  # pragma: no cover - direct script execution
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)
    for _p in (_HERE, _ROOT):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from n3d_qa.adapters import (  # type: ignore
        build_answer_table,
        build_qa_pairs,
        load_doc_lines,
        load_math1_task,
        split_library_query,
        task_class_mask,
    )
    from n3d_qa.build_qa import assemble_doclines  # type: ignore
    from n3d_qa.zh_features import (  # type: ignore
        TEXT_BUCKETS_PER_ORDER,
        ZhFeatureConfig,
        build_feature_vector,
        feature_columns,
        ngram_universe,
    )
    from n3d_qa.verify_dataset import _cv_scores  # type: ignore

SET_CHOICES: Tuple[str, ...] = ("lexical", "doclines", "math1judge")
GATED_SETS: Tuple[str, ...] = ("lexical", "doclines")
DEFAULT_SEED: int = 20261005
DEFAULT_NEG_PER_QUERY: int = 1
DEFAULT_MATH1_NEG: int = 2
DEFAULT_MAX_EMPTY_FRAC: float = 0.01
DEFAULT_MIN_AUC: float = 0.75


class ProbeError(RuntimeError):
    """Raised when the probe cannot run (bad option, unusable data)."""


def majority_baseline(y: np.ndarray) -> float:
    """Accuracy of the trivial "always predict the majority class" rule (0.0 for empty labels)."""
    if y.size == 0:
        return 0.0
    n_pos = int((y == 1).sum())
    return float(max(n_pos, int(y.size) - n_pos)) / float(y.size)


def vectorize_pairs(
    pairs: Sequence[Tuple[str, str]], config: ZhFeatureConfig
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Vectorize ``(query, candidate)`` text pairs and collect the measured bag diagnostics.

    Parameters
    ----------
    pairs : Sequence[Tuple[str, str]]
        Text pairs in row order.
    config : ZhFeatureConfig
        Effective feature spec.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        ``(X float32[M, D], stats)`` with ``stats`` holding ``rows``, ``bag_dim``,
        ``feature_dim``, ``empty_bag_rows``, ``empty_bag_fraction``, ``bag_nonzero_rows``,
        ``mean_bag_l1``, ``distinct_query_texts``, ``distinct_candidate_texts``,
        ``ngram_emissions``, ``out_of_ascii_cjk_chars``.
    """
    x = np.zeros((len(pairs), int(config.feature_dim)), dtype=np.float32)
    counters: Dict[str, int] = {}
    q_cache: Dict[str, frozenset] = {}
    c_cache: Dict[str, frozenset] = {}
    for i, (q, c) in enumerate(pairs):
        qu = q_cache.get(q)
        if qu is None:
            qu = ngram_universe(q, config)
            q_cache[q] = qu
        cu = c_cache.get(c)
        if cu is None:
            cu = ngram_universe(c, config)
            c_cache[c] = cu
        x[i, :] = build_feature_vector(q, c, config, q_units=qu, c_units=cu, counters=counters)
    bag = x[:, : int(config.bag_dim)]
    nonzero = bag.any(axis=1)
    n_empty = int(len(pairs) - int(nonzero.sum()))
    stats = {
        "rows": int(len(pairs)),
        "bag_dim": int(config.bag_dim),
        "feature_dim": int(config.feature_dim),
        "empty_bag_rows": n_empty,
        "empty_bag_fraction": float(n_empty) / float(max(len(pairs), 1)),
        "bag_nonzero_rows": int(nonzero.sum()),
        "mean_bag_l1": float(np.abs(bag).sum(axis=1).mean()) if len(pairs) else 0.0,
        "distinct_query_texts": int(len(q_cache)),
        "distinct_candidate_texts": int(len(c_cache)),
        "ngram_emissions": int(counters.get("ngrams", 0)),
        "out_of_ascii_cjk_chars": int(counters.get("oov_chars", 0)),
    }
    return x, stats


def build_eval_set(
    name: str,
    config: ZhFeatureConfig,
    seed: int,
    neg_per_query: int,
    math1_neg: int,
) -> Tuple[List[Tuple[str, str]], np.ndarray, Dict[str, Any]]:
    """Build one evaluation set: text pairs, labels and provenance.

    Parameters
    ----------
    name : str
        ``lexical`` / ``doclines`` / ``math1judge``.
    config : ZhFeatureConfig
        Effective spec (the text-line sets use it to build the product).
    seed : int
        Seed of every draw (split and negatives use ``seed`` / ``seed + 1``).
    neg_per_query : int
        Negatives per query row for the ``lexical`` set.
    math1_neg : int
        Negatives per question for the ``math1judge`` set.

    Returns
    -------
    Tuple[List[Tuple[str, str]], np.ndarray, Dict[str, Any]]
        ``(pairs, y, info)``.

    Raises
    ------
    ProbeError
        Unknown set name.
    """
    if name == "lexical":
        rows = load_doc_lines()
        library, query = split_library_query(rows, 0.25, int(seed))
        rng = np.random.default_rng(int(seed) + 1)
        pairs: List[Tuple[str, str]] = []
        labels: List[int] = []
        for qi in query:
            pairs.append((rows[qi].text, rows[qi].text))
            labels.append(1)
            for _k in range(int(neg_per_query)):
                ci = int(library[int(rng.integers(0, len(library)))])
                pairs.append((rows[qi].text, rows[ci].text))
                labels.append(0)
        return pairs, np.asarray(labels, dtype=np.int64), {
            "set": name,
            "semantics": (
                "text-line lexical matching: positive = the query line itself, negative = a line "
                "drawn from the held-out library side"
            ),
            "rows_source": len(rows),
            "library_rows": int(len(library)),
            "query_rows": int(len(query)),
            "split_seed": int(seed),
            "negative_seed": int(seed) + 1,
        }
    if name == "doclines":
        rows = load_doc_lines()
        library, query = split_library_query(rows, 0.25, int(seed))
        res = assemble_doclines(rows, library, query, config, int(neg_per_query), int(seed), 0.25)
        pairs = []
        for i in range(int(res.X.shape[0])):
            pairs.append(("", ""))
        return pairs, np.asarray(res.y, dtype=np.int64), {
            "set": name,
            "semantics": "the product text-line rows themselves (query line vs library candidate)",
            "matrix": res.X,
            "rows_source": len(rows),
            "library_rows": int(len(library)),
            "query_rows": int(len(query)),
            "split_seed": int(seed),
            "counters": res.meta["counts"],
        }
    if name == "math1judge":
        items = load_math1_task("judge")
        table = build_answer_table({"judge": items}, {"judge": "probe: the two boolean classes"})
        restrict = task_class_mask(items, table)
        pair_set = build_qa_pairs(
            "judge", items, table, int(seed), int(math1_neg), restrict=restrict,
            feature_config=config,
        )
        by_qid = {it.qid: it for it in items}
        pairs = []
        for q_row, c_row in zip(pair_set.questions, pair_set.candidates):
            it = by_qid[str(q_row["qid"])]
            pairs.append((str(it.question), str(c_row["display"]) + "\n" + str(it.explanation)))
        return pairs, np.asarray(pair_set.labels, dtype=np.int64), {
            "set": name,
            "semantics": (
                "QA match rows on Math1 boolean questions: positive = correct answer, negative = a "
                "wrong class drawn from the same 2-class table"
            ),
            "counters": pair_set.counters,
            "answer_table_classes": int(table.size),
        }
    raise ProbeError("unknown set %r; expected one of %s" % (name, list(SET_CHOICES)))


def cv_report(x: np.ndarray, y: np.ndarray, config: ZhFeatureConfig) -> Dict[str, Any]:
    """5-fold CV readings for the bag-only block, the extra block and the full row.

    Parameters
    ----------
    x : np.ndarray
        ``float32[M, D]`` features.
    y : np.ndarray
        ``int64[M]`` labels.
    config : ZhFeatureConfig
        Effective spec (defines the bag block boundary).

    Returns
    -------
    Dict[str, Any]
        ``rows`` / ``positives`` / ``negatives`` / ``majority`` plus ``bag_only``,
        ``extra_only`` and ``full`` entries ``{acc, auc, d}`` and the derived gains.
    """
    xd = np.asarray(x, dtype=np.float64)
    bag_dim = int(config.bag_dim)
    acc_bag, auc_bag = _cv_scores(xd[:, :bag_dim], y)
    acc_extra, auc_extra = _cv_scores(xd[:, bag_dim:], y)
    acc_full, auc_full = _cv_scores(xd, y)
    base = majority_baseline(y)
    return {
        "rows": int(y.size),
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "majority": float(base),
        "bag_only": {"acc": float(acc_bag), "auc": float(auc_bag), "d": int(bag_dim)},
        "extra_only": {"acc": float(acc_extra), "auc": float(auc_extra), "d": int(x.shape[1] - bag_dim)},
        "full": {"acc": float(acc_full), "auc": float(auc_full), "d": int(x.shape[1])},
        "gain_bag_over_majority": float(acc_bag) - float(base),
        "gain_full_over_majority": float(acc_full) - float(base),
        "gain_full_over_bag": float(acc_full) - float(acc_bag),
    }


def print_report(name: str, config: ZhFeatureConfig, stats: Dict[str, Any], report: Dict[str, Any]) -> None:
    """Print one evaluation-set report (every number measured in this run)."""
    spec = config.spec_dict()
    print("=" * 100)
    print("[probe] set=%s | spec_hash=%s | orders=%s buckets/order=%d bag_dim=%d D=%d"
          % (name, config.spec_hash()[:16], spec["n_gram_orders"], spec["buckets_per_order"],
             stats["bag_dim"], stats["feature_dim"]))
    print("[probe] rows=%d (pos=%d neg=%d) majority_baseline=%.4f"
          % (report["rows"], report["positives"], report["negatives"], report["majority"]))
    print("[probe] bag non-zero rows=%d/%d -> empty_bag_rows=%d (%.6f) | mean |bag|_1=%.4f | "
          "n-gram emissions=%d | distinct query texts=%d"
          % (stats["bag_nonzero_rows"], stats["rows"], stats["empty_bag_rows"],
             stats["empty_bag_fraction"], stats["mean_bag_l1"], stats["ngram_emissions"],
             stats["distinct_query_texts"]))
    print("[probe] CV bag-only  d=%-4d acc=%.4f auc=%.4f" % (report["bag_only"]["d"],
          report["bag_only"]["acc"], report["bag_only"]["auc"]))
    print("[probe] CV extra-only d=%-4d acc=%.4f auc=%.4f" % (report["extra_only"]["d"],
          report["extra_only"]["acc"], report["extra_only"]["auc"]))
    print("[probe] CV full       d=%-4d acc=%.4f auc=%.4f" % (report["full"]["d"],
          report["full"]["acc"], report["full"]["auc"]))
    print("[probe] gain over majority: bag %+.4f, full %+.4f | full over bag %+.4f"
          % (report["gain_bag_over_majority"], report["gain_full_over_majority"],
             report["gain_full_over_bag"]))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the probe CLI options."""
    ap = argparse.ArgumentParser(description="n3d_qa Chinese feature discrimination probe")
    ap.add_argument("--sets", default="lexical,doclines,math1judge",
                    help="comma separated subset of %s" % ",".join(SET_CHOICES))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed of the draws")
    ap.add_argument("--neg-per-query", type=int, default=DEFAULT_NEG_PER_QUERY,
                    help="negatives per query row (text-line sets)")
    ap.add_argument("--math1-neg", type=int, default=DEFAULT_MATH1_NEG,
                    help="negatives per question in the math1judge set")
    ap.add_argument("--buckets-per-order", type=int, default=None, help="override buckets per order")
    ap.add_argument("--ngram-orders", default="1,2,3", help="character n-gram orders")
    ap.add_argument("--sweep", action="store_true", help="sweep --sweep-buckets on the first gated set")
    ap.add_argument("--sweep-buckets", default="32,64,100,128", help="bucket counts of the sweep")
    ap.add_argument("--max-empty-frac", type=float, default=DEFAULT_MAX_EMPTY_FRAC,
                    help="gate: max share of all-zero bag rows")
    ap.add_argument("--min-auc", type=float, default=DEFAULT_MIN_AUC,
                    help="gate: min bag-only CV AUC")
    ap.add_argument("--no-gate", action="store_true", help="always exit 0 (report-only mode)")
    return ap.parse_args(argv)


def _config_for(args: argparse.Namespace, base: bool) -> Tuple[ZhFeatureConfig, ZhFeatureConfig]:
    """Return ``(text_side_config, qa_side_config)`` from the CLI options."""
    orders = tuple(int(x) for x in str(args.ngram_orders).replace(" ", "").split(",") if x)
    if args.buckets_per_order is not None:
        cfg = ZhFeatureConfig(n_gram_orders=orders, buckets_per_order=int(args.buckets_per_order))
        return cfg, cfg
    return (
        ZhFeatureConfig(n_gram_orders=orders, buckets_per_order=int(TEXT_BUCKETS_PER_ORDER)),
        ZhFeatureConfig(n_gram_orders=orders),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point of the probe.

    Returns
    -------
    int
        0 when the gate passes (or ``--no-gate``), 3 when a gated set fails the gate.
    """
    args = parse_args(argv)
    text_cfg, qa_cfg = _config_for(args, True)
    print("[probe] text-line spec: %s" % json.dumps(text_cfg.spec_dict(), ensure_ascii=False))
    print("[probe] text-line columns: %s"
          % json.dumps([c["name"] for c in feature_columns(text_cfg)], ensure_ascii=False))
    print("[probe] qa-side spec: %s" % json.dumps(qa_cfg.spec_dict(), ensure_ascii=False))
    names = [s.strip() for s in str(args.sets).split(",") if s.strip()]
    for n in names:
        if n not in SET_CHOICES:
            print("[probe] unknown set %r; expected one of %s" % (n, list(SET_CHOICES)), file=sys.stderr)
            return 2
    t0 = time.time()
    results: Dict[str, Dict[str, Any]] = {}
    for name in names:
        cfg = qa_cfg if name == "math1judge" else text_cfg
        print("[probe] building set %s (spec_hash=%s) ..." % (name, cfg.spec_hash()[:16]), flush=True)
        pairs, y, info = build_eval_set(
            name, cfg, int(args.seed), int(args.neg_per_query), int(args.math1_neg)
        )
        if "matrix" in info:
            x = np.asarray(info.pop("matrix"), dtype=np.float32)
            stats: Dict[str, Any] = {
                "rows": int(x.shape[0]),
                "bag_dim": int(cfg.bag_dim),
                "feature_dim": int(cfg.feature_dim),
                "empty_bag_rows": int(x.shape[0] - int((x[:, : cfg.bag_dim].any(axis=1)).sum())),
                "empty_bag_fraction": float(x.shape[0] - int((x[:, : cfg.bag_dim].any(axis=1)).sum()))
                / float(max(x.shape[0], 1)),
                "bag_nonzero_rows": int((x[:, : cfg.bag_dim].any(axis=1)).sum()),
                "mean_bag_l1": float(np.abs(x[:, : cfg.bag_dim]).sum(axis=1).mean()),
                "distinct_query_texts": -1,
                "distinct_candidate_texts": -1,
                "ngram_emissions": -1,
                "out_of_ascii_cjk_chars": -1,
            }
        else:
            x, stats = vectorize_pairs(pairs, cfg)
        report = cv_report(x, y, cfg)
        results[name] = {"stats": stats, "report": report, "info": info}
        print_report(name, cfg, stats, report)
        if info.get("semantics"):
            print("[probe] semantics: %s" % info["semantics"])
    if args.sweep:
        sweep_name = next((n for n in names if n in GATED_SETS), names[0])
        first = [k for k in results if k == sweep_name]
        orders = tuple(int(x) for x in str(args.ngram_orders).replace(" ", "").split(",") if x)
        print("=" * 100)
        print("[probe] bucket sweep on set=%s (report only, not a gate)" % sweep_name)
        for b in [int(x) for x in str(args.sweep_buckets).replace(" ", "").split(",") if x]:
            cfg = ZhFeatureConfig(n_gram_orders=orders, buckets_per_order=int(b))
            pairs, y, info = build_eval_set(
                sweep_name, cfg, int(args.seed), int(args.neg_per_query), int(args.math1_neg)
            )
            x, st = vectorize_pairs(pairs, cfg)
            rep = cv_report(x, y, cfg)
            results["sweep_%s_%d" % (sweep_name, b)] = {"stats": st, "report": rep, "info": {}}
            print("[probe]   buckets/order=%-4d D=%-4d bag d=%-4d empty_rows=%-3d | bag auc=%.4f acc=%.4f "
                  "| full auc=%.4f acc=%.4f | extra auc=%.4f"
                  % (b, cfg.feature_dim, cfg.bag_dim, st["empty_bag_rows"], rep["bag_only"]["auc"],
                     rep["bag_only"]["acc"], rep["full"]["auc"], rep["full"]["acc"],
                     rep["extra_only"]["auc"]))
    problems: List[str] = []
    gated = [n for n in names if n in GATED_SETS]
    for name in gated:
        r = results[name]
        if float(r["stats"]["empty_bag_fraction"]) > float(args.max_empty_frac):
            problems.append(
                "set %s: all-zero bag rows %d (%.6f) exceed the limit %.4f"
                % (name, r["stats"]["empty_bag_rows"], r["stats"]["empty_bag_fraction"],
                   args.max_empty_frac)
            )
        if float(r["report"]["bag_only"]["auc"]) < float(args.min_auc):
            problems.append(
                "set %s: bag-only CV AUC %.4f is below %.4f"
                % (name, r["report"]["bag_only"]["auc"], args.min_auc)
            )
    print("=" * 100)
    print("[probe] gated sets=%s | max_empty_frac=%.4f min_auc=%.4f | reported-only sets=%s"
          % (gated, args.max_empty_frac, args.min_auc, [n for n in names if n not in GATED_SETS]))
    if problems:
        for p in problems:
            print("[probe] [!] GATE FAIL: %s" % p)
        if args.no_gate:
            print("[probe] --no-gate: reporting the failure without failing the run (exit 0)")
            return 0
        print("[probe] gate FAILED after %.1f s -> stop, do not run a full build" % (time.time() - t0))
        return 3
    print("[probe] gate PASSED after %.1f s: the Chinese feature bag is non-zero and discriminative"
          % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())