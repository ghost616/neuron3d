"""n3d_qa.build_qa -- QA match dataset product (unified answer table + npz + text-side JSONL).

What it builds
--------------
One **QA match** product per task under ``checkpoints/qa_learn/dataset/``:

* ``judge``  : Math1 boolean questions (2 classes);
* ``choice`` : Math1 multiple-choice questions (classes A-D; single-letter answers, classes with
  fewer than ``MIN_CLASS_SAMPLES`` source questions are registered as removals, not silently
  dropped);
* ``blank``  : Math1 fill-in-the-blank questions, answers whose normalized frequency is
  ``>= MIN_REPEAT``;
* ``solve``  : Math1 solution questions, same ``>= MIN_REPEAT`` rule;
* ``triviaqa``: TriviaQA archive splits, the top ``TRIVIAQA_TOPN`` most frequent answer keys;
* ``all``    : the row/dimension-wise concatenation of the five tasks above.

Plus the **text-line** product of ``data/doc`` (``doclines``): a row table with fields
``[row_id, file, line_no, text, feature...]`` and a library/query held-out split.

Files written per product (all deterministic; see :func:`n3d_qa.build_dataset.save_npz_deterministic`):

* ``n3d_qa_<task>.npz``      : ``X`` float32 ``[M, D]`` / ``y`` int64 ``[M]`` / ``meta``;
* ``n3d_qa_<task>_pairs.jsonl``: question text + answer text side (one JSON object per row);
* ``doclines_rows.jsonl`` / ``doclines_row_index.jsonl`` / ``doclines.npz``;
* ``manifest.json``          : ``sha256`` + size of every written file (the byte-identity gate).

Determinism
-----------
No global RNG anywhere: the feature spec is hashing based, negative draws use
``np.random.default_rng(seed)`` with an explicit seed, the npz writer fixes zip timestamps, and
the JSONL writer is line based (LF, UTF-8, no ``ensure_ascii``). Same arguments -> bit-identical
files (verified by two consecutive runs plus SHA256 comparison).

Usage
-----
    python n3d_qa/build_qa.py --tasks judge --neg-per-question 1 --out-dir <dir> --drill
    python n3d_qa/build_qa.py --tasks all --triviaqa-splits wiki,web
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Import both as a package module and as a directly executed script (the TriviaQA verification
# module uses the same dual-mode pattern; the added modules follow it for the same reason: the
# drill / verify commands in the README run these files by path).
try:  # pragma: no cover - exercised by the CLI entry points
    from . import build_dataset as bd
    from .adapters import (
        ANSWER_NORM_RULE,
        DOC_DIR,
        AdapterError,
        AnswerTable,
        DocLine,
        QaItem,
        QaPairSet,
        TASK_QTYPE,
        build_answer_table,
        build_qa_pairs,
        load_doc_lines,
        load_math1_task,
        normalize_answer_key,
        sample_id,
        split_library_query,
        task_class_mask,
    )
    from .zh_features import (
        TEXT_BUCKETS_PER_ORDER,
        ZhFeatureConfig,
        build_feature_vector,
        feature_columns,
        ngram_universe,
        zh_normalize,
    )
except ImportError:  # pragma: no cover - direct script execution
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import build_dataset as bd
    from adapters import (
        ANSWER_NORM_RULE,
        DOC_DIR,
        AdapterError,
        AnswerTable,
        DocLine,
        QaItem,
        QaPairSet,
        TASK_QTYPE,
        build_answer_table,
        build_qa_pairs,
        load_doc_lines,
        load_math1_task,
        normalize_answer_key,
        sample_id,
        split_library_query,
        task_class_mask,
    )
    from zh_features import (
        TEXT_BUCKETS_PER_ORDER,
        ZhFeatureConfig,
        build_feature_vector,
        feature_columns,
        ngram_universe,
        zh_normalize,
    )

# --------------------------------------------------------------------------------------------
# specification constants (every one of them is recorded in the product meta)
# --------------------------------------------------------------------------------------------
DEFAULT_OUT_DIR: str = os.path.join(bd.PROJECT_ROOT, "checkpoints", "qa_learn", "dataset")
MANIFEST_NAME: str = "manifest.json"
PAIRS_SUFFIX: str = "_pairs.jsonl"
ROWS_NAME: str = "doclines_rows.jsonl"
ROW_INDEX_NAME: str = "doclines_row_index.jsonl"
DOCLINES_NAME: str = "doclines.npz"
ALL_TASK: str = "all"
TASK_CHOICES: Tuple[str, ...] = ("judge", "choice", "blank", "solve", "triviaqa", ALL_TASK)
ANSWER_TABLE_NAME: str = "answer_table.jsonl"

# Math1 selections
MIN_CLASS_SAMPLES: int = 5
MIN_REPEAT: int = 5
CHOICE_LETTERS: Tuple[str, ...] = ("a", "b", "c", "d")
MATH1_NEG_PER_QUESTION: Dict[str, int] = {"judge": 2, "choice": 3, "blank": 3, "solve": 3}

# TriviaQA selection
TRIVIAQA_TOPN: int = 100
TRIVIAQA_SPLITS_DEFAULT: Tuple[str, ...] = ("wiki", "web")
TRIVIAQA_NEG_PER_QUESTION: int = 3

# Text-line product
DOCLINES_QUERY_RATIO: float = 0.25
DOCLINES_NEG_PER_QUERY: int = 1

# Explicit seeds (no global RNG is ever touched)
QA_NEG_SEED: int = 20261005
DOC_SPLIT_SEED: int = 20261005
DOC_NEG_SEED: int = 20261005

TEXT_ENCODING: str = "utf-8"


class BuildQaError(RuntimeError):
    """Raised when the QA product contract is violated."""


# --------------------------------------------------------------------------------------------
# deterministic writers
# --------------------------------------------------------------------------------------------
def sha256_file(path: str, chunk_bytes: int = 1 << 22) -> str:
    """Whole-file SHA256 (chunked; memory use independent of file size)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(int(chunk_bytes)), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl_deterministic(path: str, records: Sequence[Dict[str, Any]]) -> None:
    """Write records as deterministic JSONL: LF line ends, UTF-8, no ASCII escaping, no timestamp.

    Parameters
    ----------
    path : str
        Target path (parent directories are created).
    records : Sequence[Dict[str, Any]]
        One JSON object per row; key order follows the dict insertion order of the caller (so the
        caller is the single source of truth for field order).

    Returns
    -------
    None
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=TEXT_ENCODING, newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
    os.replace(tmp, path)


def build_feature_matrix(
    pairs: Sequence[Tuple[str, str]], config: ZhFeatureConfig
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Vectorize ``(query, candidate)`` text pairs into ``float32[M, D]``.

    Parameters
    ----------
    pairs : Sequence[Tuple[str, str]]
        Text pairs in row order.
    config : ZhFeatureConfig
        Effective feature spec.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        ``(X, stats)`` where ``stats`` carries measured counters: ``rows``, ``empty_bag_rows``,
        ``distinct_query_texts``, ``distinct_candidate_texts``, ``ngram_emissions``,
        ``out_of_ascii_cjk_chars``.
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
    stats = {
        "rows": int(len(pairs)),
        "empty_bag_rows": int(counters.get("empty_rows", 0)),
        "distinct_query_texts": int(len(q_cache)),
        "distinct_candidate_texts": int(len(c_cache)),
        "ngram_emissions": int(counters.get("ngrams", 0)),
        "out_of_ascii_cjk_chars": int(counters.get("oov_chars", 0)),
    }
    return x, stats


# --------------------------------------------------------------------------------------------
# TriviaQA adapter (reuses the frozen TriviaQA archive layer read-only)
# --------------------------------------------------------------------------------------------
def load_triviaqa_items(
    splits: Sequence[str], topn: int = TRIVIAQA_TOPN, archive: str = ""
) -> Tuple[Dict[str, List[QaItem]], Dict[str, Any]]:
    """Load TriviaQA splits as QA items (answer-key selection rule recorded in the meta).

    The **family selection rule** is: count how many questions of the requested splits have the
    answer key in their merged answer set, then keep the ``topn`` most frequent keys (ties broken
    by key). The rule text is recorded verbatim in the meta.

    Parameters
    ----------
    splits : Sequence[str]
        Archive splits (``wiki`` / ``web`` / ``wiki-dev`` / ``web-dev``).
    topn : int
        Number of most frequent answer keys to keep.
    archive : str
        Archive path (defaults to the frozen TriviaQA archive).

    Returns
    -------
    Tuple[Dict[str, List[QaItem]], Dict[str, Any]]
        ``(split -> items sorted by qid, selection info)``.

    Raises
    ------
    BuildQaError
        When the archive is missing/invalid, or a requested split has no QA member.
    """
    arch = str(archive) if archive else bd.DEFAULT_ARCHIVE
    archive_sha = bd.verify_archive(arch)
    want = tuple(str(s) for s in splits)
    raw_qa, passes, status, _members = bd.load_archive_qa(arch, archive_sha, want)
    counts: Dict[str, int] = {}
    per_split_items: Dict[str, List[QaItem]] = {}
    for split in want:
        records, _head = bd.parse_qa_json(raw_qa[split], split)
        items: List[QaItem] = []
        for r in records:
            keys = set()
            for a in r.answer_values:
                # [!] The key must be the **canonical** key of the unified answer table
                # (normalize_answer_key: NFKC -> strip ALL whitespace -> casefold). Using a
                # whitespace-stripped-only key here produced keys like "likeaprayer" while the table
                # holds '"likeaprayer"'/'like a prayer'-normalized classes, so almost no TriviaQA
                # question found its answer in the table (measured: 27 of 725 questions paired, a
                # coverage of 3.7% that was an artifact of the key space, not of the data).
                key = normalize_answer_key(a)
                if key:
                    keys.add(key)
            if not keys:
                continue
            items.append(
                QaItem(
                    qid="%s:%s" % (split, r.question_id),
                    subject="triviaqa:%s" % split,
                    qtype="triviaqa",
                    question=str(r.question),
                    answer_values=tuple(str(a) for a in r.answer_values),
                    explanation="",
                    n_choices=0,
                )
            )
            for key in keys:
                counts[key] = counts.get(key, 0) + 1
        per_split_items[split] = items
    ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    keep = tuple(str(k) for k, _c in ranked[: int(topn)])
    info = {
        "archive_sha256": archive_sha,
        "archive_passes": int(passes),
        "cache_status": list(status),
        "splits": list(want),
        "selection_rule": (
            "answer keys ranked by the number of source questions that contain them across the "
            "requested splits, descending, ties broken by key ascending; the top %d keys are kept"
            % int(topn)
        ),
        "topn": int(topn),
        "distinct_answer_keys": int(len(counts)),
        "kept_keys": list(keep),
        "kept_key_counts": [int(counts[k]) for k in keep],
        "questions_per_split": {s: int(len(per_split_items[s])) for s in want},
    }
    return per_split_items, info


# --------------------------------------------------------------------------------------------
# answer-table selection rules (per task)
# --------------------------------------------------------------------------------------------
@dataclass
class TaskSelection:
    """Per-task answer-table selection outcome.

    Attributes
    ----------
    task : str
        Task name.
    rule : str
        Human-readable rule text (recorded in the meta).
    kept_keys : Tuple[str, ...]
        Answer keys that entered the unified table, ascending.
    removed : Tuple[Tuple[str, int], ...]
        ``(key, count)`` pairs **quantified and registered** as removals (never silently dropped).
    removed_samples : int
        Number of source questions affected by the removals (a question counts once).
    kept_samples : int
        Number of source questions whose answer set has at least one kept key.
    total_samples : int
        Number of source questions considered for this task.
    unkeyed_samples : int
        Source questions every one of whose answer values normalizes to the empty key (counted
        separately so that ``kept + removed + unkeyed == total`` closes exactly).
    distinct_keys : int
        Number of distinct canonical answer keys seen in this task (``kept + removed``).
    """

    task: str
    rule: str
    kept_keys: Tuple[str, ...]
    removed: Tuple[Tuple[str, int], ...]
    removed_samples: int
    kept_samples: int
    total_samples: int
    unkeyed_samples: int = 0
    distinct_keys: int = 0


def _task_key_counts(items: Sequence[QaItem]) -> Dict[str, int]:
    """``canonical answer key -> number of source questions`` for one task (ranking basis).

    The key is the **canonical** form produced by :func:`n3d_qa.adapters.normalize_answer_key`
    (NFKC -> strip all whitespace -> casefold), i.e. exactly the key space of the unified answer
    table, so a selection computed here always matches what
    :func:`n3d_qa.adapters.build_answer_table` builds (the first draft used a second, raw-strip
    key variant here, which could disagree with the table on full-width / spaced answers).
    """
    counts: Dict[str, int] = {}
    for it in items:
        keys = {normalize_answer_key(a) for a in it.answer_values}
        keys.discard("")
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
    return counts


def _item_keys(raw: str) -> List[str]:
    """Canonical key of one raw answer value as a single-element list (or empty when blank)."""
    key = normalize_answer_key(raw)
    return [key] if key else []


def select_task_keys(
    task: str,
    items: Sequence[QaItem],
    triviaqa_info: Optional[Dict[str, Any]] = None,
) -> TaskSelection:
    """Select the answer keys of one task that enter the unified table, registering removals.

    Parameters
    ----------
    task : str
        ``judge`` / ``choice`` / ``blank`` / ``solve`` / ``triviaqa``.
    items : Sequence[QaItem]
        Source records of the task.
    triviaqa_info : Optional[Dict[str, Any]]
        Selection info from :func:`load_triviaqa_items` (required for the ``triviaqa`` task).

    Returns
    -------
    TaskSelection
        Outcome including every removal with its measured size.

    Raises
    ------
    BuildQaError
        Unknown task, or missing TriviaQA selection info.
    """
    counts = _task_key_counts(items)
    ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    if task == "judge":
        rule = "all answer keys (the boolean classes)"
        kept = tuple(k for k, _c in ranked)
    elif task == "choice":
        rule = (
            "answer keys that are a single ASCII letter a-d; multi-letter answers (e.g. 'bc') and "
            "letters beyond d (e.g. 'e'/'f') are registered as removals"
        )
        kept = tuple(k for k, _c in ranked if k in CHOICE_LETTERS)
    elif task in ("blank", "solve"):
        rule = (
            "answer keys whose normalized frequency over the task is >= %d; keys below 5 are "
            "registered as removals" % int(MIN_REPEAT)
        )
        kept = tuple(k for k, c in ranked if int(c) >= int(MIN_REPEAT))
    elif task == "triviaqa":
        if not triviaqa_info:
            raise BuildQaError("triviaqa task needs the selection info of load_triviaqa_items")
        rule = str(triviaqa_info["selection_rule"])
        kept = tuple(str(k) for k in triviaqa_info["kept_keys"])
    else:
        raise BuildQaError("unknown task %r" % (task,))
    kept_set = set(kept)
    removed = tuple((k, int(c)) for k, c in ranked if k not in kept_set)
    n_kept = 0
    n_removed = 0
    n_unkeyed = 0
    for it in items:
        keys = {k for a in it.answer_values for k in _item_keys(a)}
        if keys & kept_set:
            n_kept += 1
        elif keys:
            n_removed += 1
        else:
            # no usable key at all (every answer value normalizes to the empty string): counted
            # separately from the removals, and exposed in the meta so that
            # kept + removed + unkeyed == total holds exactly and the coverage arithmetic closes.
            n_unkeyed += 1
    return TaskSelection(
        task=str(task),
        rule=str(rule),
        kept_keys=tuple(sorted(kept)),
        removed=removed,
        removed_samples=int(n_removed),
        kept_samples=int(n_kept),
        total_samples=int(len(items)),
        unkeyed_samples=int(n_unkeyed),
        distinct_keys=int(len(ranked)),
    )


def filter_items_by_keys(
    items: Sequence[QaItem], selection: TaskSelection, restrict_to_family: bool
) -> List[QaItem]:
    """Restrict records to those answerable from the selected keys, and trim their answer values.

    [!] Two distinct call sites need two distinct behaviours, so ``restrict_to_family`` selects the
    mode (the first draft conflated them and let **unkept** answer values leak into the unified
    answer table, which the builder caught by asserting that the table's key set equals the selected
    key union):

    * ``restrict_to_family=True`` — the **pair-building** mode: records with no kept answer value
      are dropped (they form the "not relevant" bucket, whose size is reported as coverage loss),
      and the surviving records keep only their kept answer values;
    * ``restrict_to_family=False`` — the **table-building** mode: every record is kept (so the
      coverage counters stay meaningful) but answer values whose key was not selected are trimmed
      away, so the unified answer table can only contain selected keys.

    Parameters
    ----------
    items : Sequence[QaItem]
        Source records.
    selection : TaskSelection
        Selection outcome (``kept_keys`` defines the allowed key set).
    restrict_to_family : bool
        Mode selector, see above.

    Returns
    -------
    List[QaItem]
        Filtered / trimmed records, same order as the input.
    """
    kept = set(selection.kept_keys)
    out: List[QaItem] = []
    for it in items:
        values = tuple(a for a in it.answer_values if set(_item_keys(a)) & kept)
        if not values:
            if restrict_to_family:
                continue
            # table-building mode: the record stays (it is counted as "not relevant"), but none of
            # its answer values may enter the table
            out.append(
                QaItem(
                    qid=it.qid, subject=it.subject, qtype=it.qtype, question=it.question,
                    answer_values=(), explanation=it.explanation, n_choices=it.n_choices,
                )
            )
            continue
        out.append(
            QaItem(
                qid=it.qid,
                subject=it.subject,
                qtype=it.qtype,
                question=it.question,
                answer_values=values,
                explanation=it.explanation,
                n_choices=it.n_choices,
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# product assembly
# --------------------------------------------------------------------------------------------
@dataclass
class TaskProduct:
    """A fully assembled QA task product (features + provenance + meta)."""

    task: str
    X: np.ndarray
    y: np.ndarray
    pairs: QaPairSet
    question_texts: List[str]
    answer_texts: List[str]
    meta: Dict[str, Any]


def _candidate_text(items_by_qid: Dict[str, QaItem], cand: Dict[str, Any]) -> str:
    """Candidate answer text of one row = drawn answer surface form + the question's explanation."""
    display = str(cand["display"])
    it = items_by_qid.get(str(cand.get("qid", "")))
    expl = str(it.explanation) if it is not None else ""
    return display + "\n" + expl


def assemble_task_product(
    task: str,
    items: Sequence[QaItem],
    table: AnswerTable,
    pair_set: QaPairSet,
    config: ZhFeatureConfig,
    neg_per_question: int,
    extra_meta: Dict[str, Any],
) -> TaskProduct:
    """Vectorize one task's pair set and assemble its meta.

    Parameters
    ----------
    task : str
        Task name.
    items : Sequence[QaItem]
        Records **after** key filtering (their explanations feed the answer text).
    table : AnswerTable
        Unified answer table.
    pair_set : QaPairSet
        Pair rows in emission order.
    config : ZhFeatureConfig
        Effective feature spec.
    neg_per_question : int
        Negatives per question actually used.
    extra_meta : Dict[str, Any]
        Task-specific meta (selection rule, removals, TriviaQA info, coverage loss).

    Returns
    -------
    TaskProduct
        Product with ``X`` / ``y`` / meta ready for the writers.

    Notes
    -----
    ``X``, ``y``, the pair provenance and this meta are **all** reproducible: nothing
    wall-clock- or cache-dependent is written into the product. In particular
    ``build_seconds`` is reported by the CLI but deliberately kept **out** of the meta, and the
    TriviaQA archive-pass count (a cache-state fact, 1 on a cold cache and 0 on a warm one) is not
    written either — both used to make two otherwise identical builds produce different npz bytes.
    """
    by_qid = {it.qid: it for it in items}
    question_texts: List[str] = []
    answer_texts: List[str] = []
    for q_row, c_row in zip(pair_set.questions, pair_set.candidates):
        it = by_qid.get(str(q_row["qid"]))
        question_texts.append(str(it.question) if it is not None else "")
        answer_texts.append(_candidate_text(by_qid, {**c_row, "qid": q_row["qid"]}))
    x, feat_stats = build_feature_matrix(list(zip(question_texts, answer_texts)), config)
    y = np.asarray(pair_set.labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    counters = dict(pair_set.counters)
    meta: Dict[str, Any] = {
        "module": "n3d_qa",
        "product": "qa_match",
        "product_kind": "npz",
        "dataset": ("%s@math1" % task) if task in TASK_QTYPE else str(task),
        "task": str(task),
        "label_rule": (
            "label=1 iff the candidate answer class is one of the question's correct answer "
            "classes (alias-resolved against the unified answer table); label=0 = a wrong class "
            "drawn from the same table"
        ),
        "question_text_rule": extra_meta.get("question_text_rule"),
        "answer_text_rule": extra_meta.get("answer_text_rule"),
        "answer_norm_rule": ANSWER_NORM_RULE,
        "features": {
            "spec": config.spec_dict(),
            "spec_hash": config.spec_hash(),
            "columns": [dict(c) for c in feature_columns(config)],
            "feature_dim": int(config.feature_dim),
            "dtype": "float32",
        },
        "selection": {
            "answer_table_rule": extra_meta.get("selection_rule"),
            "negatives_per_question": int(neg_per_question),
            "negative_draw_rule": counters.get("negative_draw_rule"),
            "restricted_classes": int(counters.get("restricted_classes", table.size)),
            "seeds": {
                "negative": int(counters.get("negative_seed", 0)),
                "doc_split": int(DOC_SPLIT_SEED),
                "doc_negative": int(DOC_NEG_SEED),
            },
        },
        "answer_table": {
            "classes": int(table.size),
            "questions_seen": int(table.n_questions_seen),
            "questions_in_table": int(table.n_questions_in_table),
            "questions_not_relevant": int(table.n_questions_seen - table.n_questions_in_table),
            "multi_value_questions": int(table.n_multi_value_questions),
            "alias_count": int(len(table.alias_index)),
            "class_list_full": extra_meta.get("class_list_ref"),
        },
        "counts": {
            "pairs": int(len(y)),
            "positives": n_pos,
            "negatives": n_neg,
            "positive_fraction": float(n_pos) / float(max(len(y), 1)),
            "majority_baseline": float(max(n_pos, n_neg)) / float(max(len(y), 1)),
            "questions": int(counters.get("questions_total", 0)),
            "questions_paired": int(counters.get("questions_paired", 0)),
            "distinct_question_ids": int(counters.get("distinct_question_ids", 0)),
            "distinct_candidate_keys": int(counters.get("distinct_candidate_keys", 0)),
        },
        "feature_stats": feat_stats,
        "coverage": extra_meta.get("coverage"),
        "removals": extra_meta.get("removals"),
        "source": extra_meta.get("source"),
        "determinism": (
            "all randomness derives from the explicit seeds above; features are blake2b hashing "
            "based; the npz writer fixes zip timestamps -> identical arguments give bit-identical "
            "files"
        ),
    }
    return TaskProduct(
        task=str(task), X=x, y=y, pairs=pair_set,
        question_texts=question_texts, answer_texts=answer_texts, meta=meta,
    )


def pairs_to_jsonl(product: TaskProduct) -> List[Dict[str, Any]]:
    """Serialize one product's rows as the text-side QA pair JSONL (question + answer text).

    Parameters
    ----------
    product : TaskProduct
        Assembled product.

    Returns
    -------
    List[Dict[str, Any]]
        One dict per row with the frozen field order
        ``row_id`` / ``task`` / ``question_id`` / ``question_text`` / ``answer_text`` / ``label`` /
        ``candidate_answer_index`` / ``candidate_answer_key``.
    """
    recs: List[Dict[str, Any]] = []
    for i, (q_row, c_row) in enumerate(zip(product.pairs.questions, product.pairs.candidates)):
        recs.append(
            {
                "row_id": str(c_row["id"]),
                "pair_id": str(sample_id("p", product.task, q_row["id"], c_row["id"])),
                "task": str(product.task),
                "question_id": str(q_row["qid"]),
                "question_text": product.question_texts[i],
                "answer_text": product.answer_texts[i],
                "label": int(product.y[i]),
                "candidate_answer_index": int(c_row["answer_index"]),
                "candidate_answer_key": str(c_row["key"]),
                "answer_table_size": int(product.meta["answer_table"]["classes"]),
            }
        )
    return recs


def write_product(out_dir: str, product: TaskProduct, tag: str) -> List[str]:
    """Write one product (npz + pair JSONL) and return the written file paths.

    Parameters
    ----------
    out_dir : str
        Output directory.
    product : TaskProduct
        Assembled product.
    tag : str
        File-name tag (task name, or ``all`` for the concatenated product).

    Returns
    -------
    List[str]
        Written file paths (npz first, then JSONL).
    """
    npz_path = os.path.join(out_dir, "n3d_qa_%s.npz" % tag)
    pairs_path = os.path.join(out_dir, "n3d_qa_%s%s" % (tag, PAIRS_SUFFIX))
    bd.save_npz_deterministic(npz_path, product.X, product.y, product.meta)
    write_jsonl_deterministic(pairs_path, pairs_to_jsonl(product))
    return [npz_path, pairs_path]


def concat_products(products: Sequence[TaskProduct], config: ZhFeatureConfig) -> TaskProduct:
    """Concatenate several products row-wise into the ``all`` product (dims must match).

    Parameters
    ----------
    products : Sequence[TaskProduct]
        Products in the desired task order.
    config : ZhFeatureConfig
        Effective spec (used to assert the common width).

    Returns
    -------
    TaskProduct
        Concatenated product with per-row ``task_of_row`` / provenance and summed counts.

    Raises
    ------
    BuildQaError
        When a product's width differs from the spec's.

    Notes
    -----
    ``X`` / ``y`` of the concatenated product are **derived** values: they carry no new
    information, so every per-task number in the report is read from the single-task products.
    """
    if not products:
        raise BuildQaError("concat_products needs at least one product")
    dim = int(config.feature_dim)
    for p in products:
        if int(p.X.shape[1]) != dim:
            raise BuildQaError(
                "product %s has width %d, expected %d" % (p.task, int(p.X.shape[1]), dim)
            )
    x = np.concatenate([p.X for p in products], axis=0)
    y = np.concatenate([p.y for p in products], axis=0)
    questions: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    q_texts: List[str] = []
    a_texts: List[str] = []
    task_of_row: List[str] = []
    for p in products:
        for i, (q_row, c_row) in enumerate(zip(p.pairs.questions, p.pairs.candidates)):
            questions.append(dict(q_row))
            candidates.append(dict(c_row))
            q_texts.append(p.question_texts[i])
            a_texts.append(p.answer_texts[i])
            task_of_row.append(str(p.task))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    per_task = {
        str(p.task): {
            "pairs": int(p.y.size),
            "positives": int((p.y == 1).sum()),
            "negatives": int((p.y == 0).sum()),
            "questions": int(p.meta["counts"]["questions"]),
            "positive_fraction": float(p.meta["counts"]["positive_fraction"]),
            "majority_baseline": float(p.meta["counts"]["majority_baseline"]),
        }
        for p in products
    }
    meta = {
        "module": "n3d_qa",
        "product": "qa_match_all",
        "product_kind": "npz",
        "dataset": "all_tasks",
        "task": ALL_TASK,
        "tasks": [str(p.task) for p in products],
        "task_of_row": task_of_row,
        "label_rule": "concatenation of the per-task label rules (see each single-task product)",
        "question_text_rule": products[0].meta["question_text_rule"],
        "answer_text_rule": products[0].meta["answer_text_rule"],
        "answer_norm_rule": ANSWER_NORM_RULE,
        "features": {
            "spec": config.spec_dict(),
            "spec_hash": config.spec_hash(),
            "columns": [dict(c) for c in feature_columns(config)],
            "feature_dim": dim,
            "dtype": "float32",
        },
        "counts": {
            "pairs": int(y.size),
            "positives": n_pos,
            "negatives": n_neg,
            "positive_fraction": float(n_pos) / float(max(y.size, 1)),
            "majority_baseline": float(max(n_pos, n_neg)) / float(max(y.size, 1)),
            "per_task": per_task,
        },
        # The concatenated product carries the same answer-table summary as the single-task ones
        # (values are the ones measured for this build; the per-class list itself lives in
        # answer_table.jsonl). Kept as a shallow copy of the first task's block on purpose: every
        # single-task product already records its own, richer block.
        "answer_table": dict(products[0].meta.get("answer_table") or {}),
        "per_task_keys": dict(products[0].meta.get("per_task_keys") or {}),
        "removals": products[0].meta.get("removals"),
        "coverage": products[0].meta.get("coverage"),
        "source": {"kind": "concatenation", "tasks": [str(p.task) for p in products]},
        "determinism": "row-wise concatenation in the declared task order; no RNG is used here",
    }
    return TaskProduct(
        task=ALL_TASK, X=x, y=y,
        pairs=QaPairSet(
            task=ALL_TASK, questions=tuple(questions), candidates=tuple(candidates),
            labels=y, counters={"task": ALL_TASK, "pairs": int(y.size)},
        ),
        question_texts=q_texts, answer_texts=a_texts, meta=meta,
    )

# --------------------------------------------------------------------------------------------
# text-line product (data/doc)
# --------------------------------------------------------------------------------------------
@dataclass
class DoclinesResult:
    """The text-line product (row table + library/query split + npz)."""

    X: np.ndarray
    y: np.ndarray
    rows: List[DocLine]
    library: Tuple[int, ...]
    query: Tuple[int, ...]
    row_records: List[Dict[str, Any]]
    index_records: List[Dict[str, Any]]
    meta: Dict[str, Any]


def assemble_doclines(
    rows: Sequence[DocLine],
    library: Sequence[int],
    query: Sequence[int],
    config: ZhFeatureConfig,
    neg_per_query: int,
    split_seed: int,
    split_ratio: float,
) -> DoclinesResult:
    """Assemble the text-line product: row features + the library/query held-out match pairs.

    Row semantics
    -------------
    * ``X`` / ``y`` / ``meta`` npz rows are ``(query line, candidate library line)`` pairs;
      ``label=1`` iff the candidate **is the same row** (compared by row id, so an identical text
      appearing on another line is a legitimate negative and can never make the label ambiguous);
    * one positive per query row, plus ``neg_per_query`` negatives drawn deterministically from the
      library with ``np.random.default_rng(negative_seed)`` (candidate indices sampled uniformly,
      a draw equal to the correct row is dropped and no substitute is drawn -> the emitted row
      count is measured, never assumed);
    * ``doclines_rows.jsonl`` lists every row with its full feature vector (row id / file / line
      number / original text / features), ``doclines_row_index.jsonl`` is the row index.

    Parameters
    ----------
    rows : Sequence[DocLine]
        All text rows (deterministic order).
    library : Sequence[int]
        Held-out library indices (ascending).
    query : Sequence[int]
        Held-out query indices (ascending).
    config : ZhFeatureConfig
        Effective feature spec.
    neg_per_query : int
        Negatives per query row.
    split_seed : int
        Seed recorded for the library/query split.
    split_ratio : float
        Query fraction recorded for the split.

    Returns
    -------
    DoclinesResult
        Assembled product.

    Raises
    ------
    BuildQaError
        When the library is empty (no negative candidate could be drawn).
    """
    lib = tuple(int(i) for i in library)
    qry = tuple(int(i) for i in query)
    if not lib:
        raise BuildQaError("the library side is empty; cannot draw a negative candidate")
    rng = np.random.default_rng(int(split_seed) + 1)
    pairs: List[Tuple[int, int, int]] = []
    n_dropped = 0
    for qi in qry:
        pairs.append((int(qi), int(qi), 1))
        draws = rng.integers(0, len(lib), size=int(neg_per_query)).tolist()
        for p in draws:
            ci = int(lib[int(p)])
            if ci == int(qi):
                n_dropped += 1
                continue
            pairs.append((int(qi), ci, 0))
    x = np.zeros((len(pairs), int(config.feature_dim)), dtype=np.float32)
    y = np.zeros(len(pairs), dtype=np.int64)
    counters: Dict[str, int] = {}
    q_cache: Dict[str, frozenset] = {}
    c_cache: Dict[str, frozenset] = {}
    for i, (qi, ci, lab) in enumerate(pairs):
        q_text, c_text = str(rows[qi].text), str(rows[ci].text)
        qu = q_cache.get(q_text)
        if qu is None:
            qu = ngram_universe(q_text, config)
            q_cache[q_text] = qu
        cu = c_cache.get(c_text)
        if cu is None:
            cu = ngram_universe(c_text, config)
            c_cache[c_text] = cu
        x[i, :] = build_feature_vector(q_text, c_text, config, q_units=qu, c_units=cu, counters=counters)
        y[i] = int(lab)
    # Row table (one JSONL record per row). [!] The row features are the row **against itself**,
    # not a slice of the pair matrix: the pair matrix holds (query, candidate) rows, so indexing it
    # by row number mixes the two and even runs out of bounds (first draft bug, caught by the
    # single-record drill). The self-pair vector is the canonical "text feature" of the row.
    row_x = np.zeros((len(rows), int(config.feature_dim)), dtype=np.float32)
    for i, row in enumerate(rows):
        row_x[i, :] = build_feature_vector(
            str(row.text), str(row.text), config, counters=counters
        )
    rounded = np.round(row_x, 6)
    row_records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        row_records.append(
            {
                "row_id": str(row.row_id),
                "file": str(row.file_name),
                "line_no": int(row.line_no),
                "text": str(row.text),
                "text_norm": zh_normalize(row.text),
                "char_len": int(len(row.text)),
                "feature_dim": int(config.feature_dim),
                "feature": [float(v) for v in rounded[i].tolist()],
            }
        )
    index_records: List[Dict[str, Any]] = []
    cases: Dict[int, List[Tuple[int, int]]] = {}
    for qi, ci, lab in pairs:
        cases.setdefault(int(qi), []).append((int(ci), int(lab)))
    for i, row in enumerate(rows):
        tr = cases.get(i, [])
        index_records.append(
            {
                "row_id": str(row.row_id),
                "index": int(i),
                "file": str(row.file_name),
                "line_no": int(row.line_no),
                "char_len": int(len(row.text)),
                "rows_for_this_query": len(tr),
                "positives": int(sum(1 for _c, l in tr if l == 1)),
                "negatives": int(sum(1 for _c, l in tr if l == 0)),
            }
        )
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    meta: Dict[str, Any] = {
        "module": "n3d_qa",
        "product": "doclines",
        "product_kind": "npz",
        "dataset": "data/doc",
        "task": "doclines",
        "label_rule": (
            "label=1 iff the candidate library row IS the query row (compared by row id, not by "
            "text: identical text on another line stays a legitimate negative)"
        ),
        "question_text_rule": (
            "query side = the original non-empty line text, located by row_id = blake2b(file, "
            "line_no); line numbers are 1-based physical lines"
        ),
        "answer_text_rule": "candidate side = the original non-empty line text of the library row",
        "line_definition": (
            "a feature unit is a line whose str.strip() is non-empty; the physical line number "
            "(1-based) and the row id are both stored in doclines_rows.jsonl / "
            "doclines_row_index.jsonl"
        ),
        "answer_norm_rule": ANSWER_NORM_RULE,
        "line_split_rule": (
            "non-empty line after str.strip(); physical line numbers kept 1-based; rows ordered by "
            "(file name, line number)"
        ),
        "features": {
            "spec": config.spec_dict(),
            "spec_hash": config.spec_hash(),
            "columns": [dict(c) for c in feature_columns(config)],
            "feature_dim": int(config.feature_dim),
            "dtype": "float32",
        },
        "split": {
            "library_rows": int(len(lib)),
            "query_rows": int(len(qry)),
            "query_ratio": float(split_ratio),
            "seed": int(split_seed),
            "negative_seed": int(split_seed) + 1,
            "negatives_per_query": int(neg_per_query),
            "intersection": int(len(set(lib) & set(qry))),
            "union_size": int(len(set(lib) | set(qry))),
        },
        "counts": {
            "rows_total": int(len(rows)),
            "pairs": int(len(y)),
            "positives": n_pos,
            "negatives": n_neg,
            "positive_fraction": float(n_pos) / float(max(len(y), 1)),
            "majority_baseline": float(max(n_pos, n_neg)) / float(max(len(y), 1)),
            "dropped_self_draws": int(n_dropped),
            "per_file_rows": {
                name: int(sum(1 for r in rows if r.file_name == name))
                for name in sorted({r.file_name for r in rows})
            },
        },
        "feature_stats": {
            "distinct_query_texts": int(len(q_cache)),
            "distinct_candidate_texts": int(len(c_cache)),
            "empty_bag_rows": int(counters.get("empty_rows", 0)),
            "ngram_emissions": int(counters.get("ngrams", 0)),
            "out_of_ascii_cjk_chars": int(counters.get("oov_chars", 0)),
        },
        "source": {
            "doc_dir": os.path.relpath(DOC_DIR, bd.PROJECT_ROOT).replace(os.sep, "/"),
            "files": sorted({r.file_name for r in rows}),
            "per_file_nonempty_lines": {
                name: int(sum(1 for r in rows if r.file_name == name))
                for name in sorted({r.file_name for r in rows})
            },
            "row_id_rule": "blake2b(\"line\\x1f<file>\\x1f<line_no>\", digest_size=8) as 16 hex chars",
        },
        "determinism": (
            "split permutation from np.random.default_rng(seed); negatives from "
            "np.random.default_rng(seed+1); features are blake2b hashing based; the npz writer "
            "fixes zip timestamps"
        ),
    }
    return DoclinesResult(
        X=x, y=y, rows=list(rows), library=lib, query=qry,
        row_records=row_records, index_records=index_records, meta=meta,
    )


def write_doclines(out_dir: str, result: DoclinesResult) -> List[str]:
    """Write the text-line product (npz + row table JSONL + row index JSONL); return the paths."""
    npz_path = os.path.join(out_dir, DOCLINES_NAME)
    rows_path = os.path.join(out_dir, ROWS_NAME)
    index_path = os.path.join(out_dir, ROW_INDEX_NAME)
    bd.save_npz_deterministic(npz_path, result.X, result.y, result.meta)
    write_jsonl_deterministic(rows_path, result.row_records)
    write_jsonl_deterministic(index_path, result.index_records)
    return [npz_path, rows_path, index_path]


# --------------------------------------------------------------------------------------------
# top level build
# --------------------------------------------------------------------------------------------
def load_task_source(
    task: str,
    triviaqa_splits: Sequence[str],
    triviaqa_topn: int,
    archive: str,
    max_records: int,
) -> Tuple[List[QaItem], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Load the raw source records of one task plus its selection info.

    Parameters
    ----------
    task : str
        ``judge`` / ``choice`` / ``blank`` / ``solve`` / ``triviaqa``.
    triviaqa_splits : Sequence[str]
        Archive splits for the ``triviaqa`` task.
    triviaqa_topn : int
        Top-N answer keys for the ``triviaqa`` task.
    archive : str
        Archive path (empty = the frozen default).
    max_records : int
        Optional cap per task (``0`` = no cap); applied after the deterministic sort.

    Returns
    -------
    Tuple[List[QaItem], Optional[Dict[str, Any]], Dict[str, Any]]
        ``(items, triviaqa selection info or None, source description)``.

    Raises
    ------
    BuildQaError
        Unknown task name.
    """
    if task in TASK_QTYPE:
        items = load_math1_task(task, max_records=max_records)
        source = {
            "kind": "math1",
            "qtype": TASK_QTYPE[task],
            "subjects": ["\u79bb\u6563\u6570\u5b66", "\u9ad8\u7b49\u6570\u5b66"],
            "records_loaded": int(len(items)),
            "max_records": int(max_records),
        }
        return items, None, source
    if task == "triviaqa":
        per_split, info = load_triviaqa_items(triviaqa_splits, triviaqa_topn, archive)
        items = [it for s in sorted(per_split) for it in per_split[s]]
        if int(max_records) > 0:
            items = items[: int(max_records)]
        source = {
            "kind": "triviaqa",
            "splits": list(triviaqa_splits),
            "archive_sha256": info.get("archive_sha256"),
            "records_loaded": int(len(items)),
            "max_records": int(max_records),
        }
        return items, info, source
    raise BuildQaError("unknown task %r" % (task,))


def build_task_product(
    task: str,
    items: Sequence[QaItem],
    table: AnswerTable,
    selection: TaskSelection,
    triviaqa_info: Optional[Dict[str, Any]],
    neg_per_question: int,
    feature_config: ZhFeatureConfig,
    seed: int,
    source: Dict[str, Any],
    class_list: Sequence[Dict[str, Any]],
    coverage: Dict[str, Any],
) -> TaskProduct:
    """Build one QA task product (filter records, build pairs, vectorize, assemble meta)."""
    t0 = time.time()
    kept_items = filter_items_by_keys(items, selection, restrict_to_family=True)
    restrict = task_class_mask(kept_items, table)
    pair_set = build_qa_pairs(
        task, kept_items, table, seed, neg_per_question,
        restrict=restrict, feature_config=feature_config,
    )
    extra_meta: Dict[str, Any] = {
        "question_text_rule": "question field + all choice options (joined by a newline) for math1; the Question field for triviaqa",
        "answer_text_rule": "the candidate answer surface form + the question's explanation field (joined by a newline)",
        "selection_rule": selection.rule,
        "class_list_ref": "meta.answer_table.class_list",
        "coverage": coverage,
        "removals": {
            "rule": selection.rule,
            "removed_key_count": int(len(selection.removed)),
            "removed_keys_head": [
                {"key": str(k), "questions": int(c)} for k, c in selection.removed[:50]
            ],
            "removed_questions": int(selection.removed_samples),
            "kept_questions": int(selection.kept_samples),
            "unkeyed_questions": int(selection.unkeyed_samples),
            "total_questions": int(selection.total_samples),
            "distinct_keys": int(selection.distinct_keys),
        },
        "source": source,
    }
    prod = assemble_task_product(
        task, kept_items, table, pair_set, feature_config, neg_per_question, extra_meta,
    )
    prod.meta["answer_table"]["class_list"] = [dict(c) for c in class_list]
    prod.meta["answer_table"]["class_list_ref"] = (
        "the full class list is written to answer_table.jsonl next to the product"
    )
    # Per-task key arithmetic, recorded so the verification can close it without re-deriving the
    # selection (the unified table's class count is a global number and cannot be compared to one
    # task's kept-key count).
    prod.meta["per_task_keys"] = {
        "kept_keys": int(len(selection.kept_keys)),
        "removed_keys": int(len(selection.removed)),
        "distinct_keys": int(selection.distinct_keys),
        "unkeyed_questions": int(selection.unkeyed_samples),
        "selection_rule": str(selection.rule),
    }
    if task in TASK_QTYPE:
        stem = task
    else:
        # top-N is read from the selection info (the number actually used), never re-derived here
        topn_used = int(triviaqa_info["topn"]) if triviaqa_info else int(TRIVIAQA_TOPN)
        stem = "triviaqa_%s_top%d" % (
            "-".join(sorted(triviaqa_info["splits"])) if triviaqa_info else "x", topn_used
        )
    prod.meta["dataset"] = ("math1:%s" % task) if task in TASK_QTYPE else stem
    if triviaqa_info is not None:
        # [!] Only the reproducible part of the TriviaQA selection info goes into the product meta:
        # ``archive_passes`` and ``cache_status`` describe the **cache state of this run** (1 pass
        # cold / 0 warm), so writing them would make two identical builds differ; they are reported
        # by the CLI instead.
        prod.meta["triviaqa"] = {
            k: v for k, v in triviaqa_info.items()
            if k not in ("cache_status", "archive_passes")
        }
    return prod


def validate_task_product(prod: TaskProduct, table: AnswerTable) -> List[str]:
    """Assert the product contract; return the list of violated invariants (empty = OK).

    Checked invariants
    ------------------
    * shape: ``X[M, D]`` float32, ``y[M]`` int64, ``D == meta.features.feature_dim``;
    * no NaN/Inf in ``X``;
    * the bag block is **not** all-zero for the whole matrix and at least one row has a non-zero
      bag (the blocking issue this product exists to fix);
    * labels are a subset of ``{0, 1}`` and both classes are present;
    * every ``label=1`` row's candidate class is one of the question's correct classes and every
      ``label=0`` row's candidate class is not (recomputed here from the table, not trusted);
    * meta counts match the arrays.

    Parameters
    ----------
    prod : TaskProduct
        Product to check.
    table : AnswerTable
        Unified answer table.

    Returns
    -------
    List[str]
        Violated invariants as readable messages.
    """
    problems: List[str] = []
    x, y = prod.X, prod.y
    dim = int(prod.meta["features"]["feature_dim"])
    if x.dtype != np.float32:
        problems.append("X.dtype=%s, expected float32" % x.dtype)
    if y.dtype != np.int64:
        problems.append("y.dtype=%s, expected int64" % y.dtype)
    if x.ndim != 2 or int(x.shape[1]) != dim:
        problems.append("X.shape=%r, expected [M, %d]" % (x.shape, dim))
    if y.ndim != 1 or int(y.shape[0]) != int(x.shape[0]):
        problems.append("y.shape=%r does not match X rows %d" % (y.shape, int(x.shape[0])))
    if not bool(np.isfinite(x).all()):
        problems.append("X contains NaN/Inf: %d non-finite" % int((~np.isfinite(x)).sum()))
    bag_dim = int(prod.meta["features"]["spec"]["bag_dim"])
    if not bool(x[:, :bag_dim].any()):
        problems.append("the character n-gram bag block is all-zero for every row")
    if int((x[:, :bag_dim].any(axis=1)).sum()) != int(x.shape[0]):
        problems.append(
            "rows with an all-zero bag block: %d"
            % int(x.shape[0] - (x[:, :bag_dim].any(axis=1)).sum())
        )
    labels = set(np.unique(y).tolist())
    if not labels <= {0, 1}:
        problems.append("labels %r are not a subset of {0, 1}" % sorted(labels))
    if labels != {0, 1}:
        problems.append("only one label class present: %r" % sorted(labels))
    n_bad = 0
    n_checked = 0
    for i, (q_row, c_row) in enumerate(zip(prod.pairs.questions, prod.pairs.candidates)):
        correct = tuple(int(j) for j in q_row.get("correct_indices", ()))
        if not correct:
            continue
        n_checked += 1
        is_correct = int(c_row["answer_index"]) in set(correct)
        if bool(int(y[i]) == 1) != bool(is_correct):
            n_bad += 1
    if n_checked != int(y.size):
        problems.append(
            "label rule not verifiable for %d of %d rows (missing correct_indices)"
            % (int(y.size) - n_checked, int(y.size))
        )
    if n_bad:
        problems.append("label/correct-answer mismatches: %d" % n_bad)
    if int(prod.meta["counts"]["pairs"]) != int(y.size):
        problems.append("meta.counts.pairs=%d != M=%d" % (int(prod.meta["counts"]["pairs"]), int(y.size)))
    if int(prod.meta["counts"]["positives"]) != int((y == 1).sum()):
        problems.append("meta.counts.positives mismatch")
    if int(prod.meta["counts"]["negatives"]) != int((y == 0).sum()):
        problems.append("meta.counts.negatives mismatch")
    return problems

@dataclass
class BuildQaResult:
    """Outcome of one full QA/dataset build (in-memory; the caller prints and/or returns it)."""

    task_products: List[TaskProduct]
    all_product: Optional[TaskProduct]
    doclines: Optional[DoclinesResult]
    answer_table: AnswerTable
    selections: Dict[str, TaskSelection]
    files: List[str]
    seconds: Dict[str, float]
    problems: List[str]


def build_all(
    tasks: Sequence[str],
    out_dir: str,
    config: ZhFeatureConfig,
    neg_per_question: int,
    triviaqa_splits: Sequence[str],
    triviaqa_topn: int,
    archive: str,
    with_doclines: bool,
    doc_query_ratio: float,
    doc_neg_per_query: int,
    max_records: int,
    seed: int,
    dry_run: bool = False,
    doclines_config: Optional[ZhFeatureConfig] = None,
) -> BuildQaResult:
    """Build the requested QA task products (+ the ``all`` product and the text-line product).

    Parameters
    ----------
    tasks : Sequence[str]
        Task names to build (``judge`` / ``choice`` / ``blank`` / ``solve`` / ``triviaqa``); the
        ``all`` product is derived automatically when more than one task is built.
    out_dir : str
        Output directory (created when missing).
    config : ZhFeatureConfig
        Effective Chinese feature spec.
    neg_per_question : int
        Negatives per question for the QA tasks (``-1`` selects the per-task defaults).
    triviaqa_splits : Sequence[str]
        Archive splits for the ``triviaqa`` task.
    triviaqa_topn : int
        Top-N answer keys for the ``triviaqa`` task.
    archive : str
        Archive path (empty = frozen default).
    with_doclines : bool
        Also build the text-line product.
    doc_query_ratio : float
        Query fraction of the text-line split.
    doc_neg_per_query : int
        Negative library rows per query row.
    max_records : int
        Optional per-task record cap (``0`` = no cap).
    seed : int
        Base seed for the negative draws (per-task offsets keep the tasks independent).
    dry_run : bool
        When true, everything is computed but nothing is written (the single-record drill path).
    doclines_config : Optional[ZhFeatureConfig]
        Feature spec of the text-line product (its own, smaller bucket count by default); recorded
        separately in the manifest because the two products use different specs.

    Returns
    -------
    BuildQaResult
        Products, written file list, timings and any contract problems found.
    """
    if doclines_config is None:
        doclines_config = ZhFeatureConfig(
            n_gram_orders=config.n_gram_orders, buckets_per_order=int(TEXT_BUCKETS_PER_ORDER)
        )
    seconds: Dict[str, float] = {}
    t_start = time.time()
    wanted = [str(t) for t in tasks]
    if ALL_TASK in wanted:
        wanted = [t for t in TASK_CHOICES if t not in (ALL_TASK,)]
    t0 = time.time()
    raw: Dict[str, List[QaItem]] = {}
    triviaqa_infos: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, Dict[str, Any]] = {}
    for task in wanted:
        items, info, source = load_task_source(
            task, triviaqa_splits, triviaqa_topn, archive, max_records
        )
        raw[task] = items
        sources[task] = source
        if info is not None:
            triviaqa_infos[task] = info
    seconds["load_sources"] = round(time.time() - t0, 2)
    # per-task selections -> one unified answer table over the *selected* keys only
    selections: Dict[str, TaskSelection] = {}
    selected_items: Dict[str, List[QaItem]] = {}
    for task in wanted:
        sel = select_task_keys(task, raw[task], triviaqa_infos.get(task))
        selections[task] = sel
        selected_items[task] = filter_items_by_keys(raw[task], sel, restrict_to_family=False)
    table = build_answer_table({t: selected_items[t] for t in wanted}, {t: selections[t].rule for t in wanted})
    # the table must cover exactly the selected key union
    selected_keys = {k for t in wanted for k in selections[t].kept_keys}
    if set(table.index_by_key) != selected_keys:
        missing = sorted(selected_keys - set(table.index_by_key))[:5]
        extra = sorted(set(table.index_by_key) - selected_keys)[:5]
        raise BuildQaError(
            "answer table does not match the selected key union (missing %s, extra %s)" % (missing, extra)
        )
    class_list = [
        {"index": i, "key": c.key, "display": c.display, "count": int(c.count),
         "sources": {k: int(v) for k, v in c.sources}, "raw_variants": int(c.raw_variants)}
        for i, c in enumerate(table.classes)
    ]
    for task in wanted:
        sel = selections[task]
        n_kept = int(sel.kept_samples)
        if task in TASK_QTYPE:
            default_neg = MATH1_NEG_PER_QUESTION.get(task, 3)
        else:
            default_neg = TRIVIAQA_NEG_PER_QUESTION
        used_neg = int(default_neg if int(neg_per_question) < 0 else neg_per_question)
        coverage = {
            "questions_total": int(sel.total_samples),
            "questions_kept": n_kept,
            "questions_removed": int(sel.total_samples - n_kept),
            "coverage_rate": float(n_kept) / float(max(sel.total_samples, 1)),
            "coverage_loss": 1.0 - float(n_kept) / float(max(sel.total_samples, 1)),
            "kept_keys": int(len(sel.kept_keys)),
            "removed_keys": int(len(sel.removed)),
        }
        selections[task] = sel
        sources[task]["neg_per_question"] = used_neg
    products: List[TaskProduct] = []
    problems: List[str] = []
    t0 = time.time()
    for i, task in enumerate(wanted):
        if task in TASK_QTYPE:
            default_neg = MATH1_NEG_PER_QUESTION.get(task, 3)
        else:
            default_neg = TRIVIAQA_NEG_PER_QUESTION
        used_neg = int(default_neg if int(neg_per_question) < 0 else neg_per_question)
        sel = selections[task]
        coverage = {
            "questions_total": int(sel.total_samples),
            "questions_kept": int(sel.kept_samples),
            "questions_removed": int(sel.total_samples - sel.kept_samples),
            "coverage_rate": float(sel.kept_samples) / float(max(sel.total_samples, 1)),
            "coverage_loss": 1.0 - float(sel.kept_samples) / float(max(sel.total_samples, 1)),
            "kept_keys": int(len(sel.kept_keys)),
            "removed_keys": int(len(sel.removed)),
        }
        prod = build_task_product(
            task, raw[task], table, sel, triviaqa_infos.get(task), used_neg, config,
            int(seed) + i, sources[task], class_list, coverage,
        )
        problems.extend("%s: %s" % (task, p) for p in validate_task_product(prod, table))
        # cross-check the two independent "how many questions are answerable" counters: the
        # selection side reported coverage.questions_kept, the pair side counted questions_paired
        n_paired = int(prod.meta["counts"]["questions_paired"])
        if n_paired != int(sel.kept_samples):
            problems.append(
                "%s: questions_paired=%d != selection.kept_samples=%d"
                % (task, n_paired, int(sel.kept_samples))
            )
        products.append(prod)
    seconds["build_tasks"] = round(time.time() - t0, 2)
    all_product: Optional[TaskProduct] = None
    if len(products) > 1:
        all_product = concat_products(products, config)
        if int(all_product.X.shape[0]) != sum(int(p.X.shape[0]) for p in products):
            problems.append("the 'all' product row count does not equal the sum of the tasks")
    doclines: Optional[DoclinesResult] = None
    if with_doclines:
        t0 = time.time()
        rows = load_doc_lines(DOC_DIR, max_lines_per_file=0)
        library, query = split_library_query(rows, float(doc_query_ratio), int(seed))
        doclines = assemble_doclines(
            rows, library, query, doclines_config, int(doc_neg_per_query), int(seed),
            float(doc_query_ratio),
        )
        seconds["build_doclines"] = round(time.time() - t0, 2)
    files: List[str] = []
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        # The unified answer table is written once, as its own readable JSONL: question meta only
        # carries the class count / alias count / key union, the full class list lives here so it
        # can be read back and audited (F6 re-reads this file).
        answer_table_path = os.path.join(out_dir, ANSWER_TABLE_NAME)
        write_jsonl_deterministic(
            answer_table_path,
            [
                {
                    "index": i,
                    "key": c.key,
                    "display": c.display,
                    "count": int(c.count),
                    "sources": {k: int(v) for k, v in c.sources},
                    "raw_variants": int(c.raw_variants),
                }
                for i, c in enumerate(table.classes)
            ],
        )
        files.append(answer_table_path)
        for prod in products:
            files.extend(write_product(out_dir, prod, prod.task))
        if all_product is not None:
            files.extend(write_product(out_dir, all_product, ALL_TASK))
        if doclines is not None:
            files.extend(write_doclines(out_dir, doclines))
        manifest = {
            "module": "n3d_qa",
            "generator": "n3d_qa/build_qa.py",
            "tasks": wanted,
            "doclines": bool(with_doclines),
            "feature_spec_hash": config.spec_hash(),
            "feature_spec": config.spec_dict(),
            "feature_dim": int(config.feature_dim),
            "doclines_feature_spec_hash": doclines_config.spec_hash(),
            "doclines_feature_spec": doclines_config.spec_dict(),
            "answer_table": {
                "classes": int(table.size),
                "alias_count": int(len(table.alias_index)),
                "questions_seen": int(table.n_questions_seen),
                "questions_in_table": int(table.n_questions_in_table),
                "file": ANSWER_TABLE_NAME,
                "selection_rules": {t: selections[t].rule for t in wanted},
            },
            "files": [
                {
                    "name": os.path.basename(p),
                    "bytes": int(os.path.getsize(p)),
                    "sha256": sha256_file(p),
                }
                for p in sorted(files)
            ],
        }
        manifest_path = os.path.join(out_dir, MANIFEST_NAME)
        with open(manifest_path, "w", encoding=TEXT_ENCODING, newline="\n") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=1, sort_keys=False)
            fh.write("\n")
        files.append(manifest_path)
    seconds["total"] = round(time.time() - t_start, 2)
    return BuildQaResult(
        task_products=products, all_product=all_product, doclines=doclines,
        answer_table=table, selections=selections, files=files, seconds=seconds,
        problems=problems,
    )


def build_zh_config(args: argparse.Namespace, text_side: bool = False) -> ZhFeatureConfig:
    """Translate CLI options into a :class:`ZhFeatureConfig`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI (``--ngram-orders`` / ``--buckets-per-order`` / ``--text-buckets-per-order``).
    text_side : bool
        When true, use the text-line bucket count (its own, smaller sample count).

    Returns
    -------
    ZhFeatureConfig
        Effective spec.
    """
    orders = tuple(int(x) for x in str(args.ngram_orders).replace(" ", "").split(",") if x)
    if text_side and args.text_buckets_per_order is not None:
        return ZhFeatureConfig(n_gram_orders=orders, buckets_per_order=int(args.text_buckets_per_order))
    if text_side and args.buckets_per_order is None:
        base = ZhFeatureConfig(n_gram_orders=orders, buckets_per_order=int(TEXT_BUCKETS_PER_ORDER))
        return ZhFeatureConfig(n_gram_orders=orders, buckets_per_order=int(TEXT_BUCKETS_PER_ORDER))
    if args.buckets_per_order is None:
        return ZhFeatureConfig(n_gram_orders=orders)
    return ZhFeatureConfig(n_gram_orders=orders, buckets_per_order=int(args.buckets_per_order))


def build_doclines_config(args: argparse.Namespace) -> ZhFeatureConfig:
    """Effective spec of the text-line product (defaults to :data:`TEXT_BUCKETS_PER_ORDER`).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    ZhFeatureConfig
        Spec used by the ``doclines`` product, i.e. with its own (smaller) bucket count unless
        ``--buckets-per-order`` was given explicitly.
    """
    orders = tuple(int(x) for x in str(args.ngram_orders).replace(" ", "").split(",") if x)
    if args.buckets_per_order is not None:
        return ZhFeatureConfig(n_gram_orders=orders, buckets_per_order=int(args.buckets_per_order))
    return ZhFeatureConfig(n_gram_orders=orders, buckets_per_order=int(TEXT_BUCKETS_PER_ORDER))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the CLI options of :mod:`n3d_qa.build_qa`."""
    ap = argparse.ArgumentParser(
        description="Build the n3d_qa QA match product and the data/doc text-line product"
    )
    ap.add_argument("--tasks", default="judge",
                    help="comma separated tasks: %s" % ",".join(TASK_CHOICES))
    ap.add_argument("--neg-per-question", type=int, default=-1,
                    help="negatives per question (-1 = per-task defaults: judge 2, others 3)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="output directory")
    ap.add_argument("--ngram-orders", default="1,2,3", help="character n-gram orders, e.g. 1,2,3")
    ap.add_argument("--buckets-per-order", type=int, default=None,
                    help="buckets per n-gram order (default: 100 QA side / 64 text side)")
    ap.add_argument("--text-buckets-per-order", type=int, default=None,
                    help="deprecated alias of --buckets-per-order, kept for the drill command")
    ap.add_argument("--triviaqa-splits", default="wiki,web", help="archive splits for the triviaqa task")
    ap.add_argument("--triviaqa-topn", type=int, default=TRIVIAQA_TOPN, help="top-N answer keys")
    ap.add_argument("--archive", default="", help="triviaqa archive path (default: frozen archive)")
    ap.add_argument("--doclines", action="store_true", help="also build the data/doc text-line product")
    ap.add_argument("--no-doclines", action="store_true", help="skip the text-line product")
    ap.add_argument("--doc-query-ratio", type=float, default=DOCLINES_QUERY_RATIO,
                    help="query fraction of the text-line split")
    ap.add_argument("--doc-neg-per-query", type=int, default=DOCLINES_NEG_PER_QUERY,
                    help="negative library rows per query row")
    ap.add_argument("--max-records", type=int, default=0, help="cap records per task (0 = no cap)")
    ap.add_argument("--seed", type=int, default=QA_NEG_SEED, help="base seed of the negative draws")
    ap.add_argument("--drill", action="store_true",
                    help="single-record drill: 1 task, few records, nothing written unless --out-dir")
    ap.add_argument("--dry-run", action="store_true", help="compute everything, write nothing")
    return ap.parse_args(argv)


def print_report(result: BuildQaResult, config: ZhFeatureConfig, doclines_config: ZhFeatureConfig) -> None:
    """Print the measured build report (every number comes from this run, none from memory)."""
    print("[n3d_qa.build_qa] feature spec hash = %s" % config.spec_hash())
    print("[n3d_qa.build_qa] spec = %s" % json.dumps(config.spec_dict(), ensure_ascii=False))
    print("[n3d_qa.build_qa] answer table: classes=%d, alias=%d, questions seen=%d, in table=%d, "
          "not relevant=%d, multi-value=%d"
          % (result.answer_table.size, len(result.answer_table.alias_index),
             result.answer_table.n_questions_seen, result.answer_table.n_questions_in_table,
             result.answer_table.n_questions_seen - result.answer_table.n_questions_in_table,
             result.answer_table.n_multi_value_questions))
    for task, sel in result.selections.items():
        print("[n3d_qa.build_qa] task %-9s selection: keys kept=%d removed=%d | questions=%d kept=%d "
              "removed=%d coverage=%.4f"
              % (task, len(sel.kept_keys), len(sel.removed), sel.total_samples, sel.kept_samples,
                 sel.total_samples - sel.kept_samples,
                 float(sel.kept_samples) / float(max(sel.total_samples, 1))))
    for prod in result.task_products:
        c = prod.meta["counts"]
        print("[n3d_qa.build_qa] task %-9s product: M=%d D=%d pos=%d neg=%d pos_frac=%.4f "
              "majority=%.4f questions=%d paired=%d cand_keys=%d empty_bag_rows=%d"
              % (prod.task, int(prod.X.shape[0]), int(prod.X.shape[1]), c["positives"], c["negatives"],
                 c["positive_fraction"], c["majority_baseline"], c["questions"], c["questions_paired"],
                 c["distinct_candidate_keys"], prod.meta["feature_stats"]["empty_bag_rows"]))
    if result.all_product is not None:
        c = result.all_product.meta["counts"]
        print("[n3d_qa.build_qa] all product: M=%d D=%d pos=%d neg=%d majority=%.4f"
              % (int(result.all_product.X.shape[0]), int(result.all_product.X.shape[1]),
                 c["positives"], c["negatives"], c["majority_baseline"]))
    if result.doclines is not None:
        c = result.doclines.meta["counts"]
        s = result.doclines.meta["split"]
        print("[n3d_qa.build_qa] doclines: rows=%d library=%d query=%d intersection=%d pairs=%d "
              "pos=%d neg=%d majority=%.4f spec_hash=%s"
              % (c["rows_total"], s["library_rows"], s["query_rows"], s["intersection"],
                 c["pairs"], c["positives"], c["negatives"], c["majority_baseline"],
                 doclines_config.spec_hash()))
    print("[n3d_qa.build_qa] seconds = %s" % json.dumps(result.seconds))
    for p in result.files:
        print("[n3d_qa.build_qa] wrote %s (%d bytes)" % (p, os.path.getsize(p)))
    if result.problems:
        print("[n3d_qa.build_qa] [!] contract problems:")
        for p in result.problems:
            print("    " + p)
    else:
        print("[n3d_qa.build_qa] contract checks: OK (0 problems)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point of the QA/text-line build.

    Returns
    -------
    int
        0 on success, 1 when a contract problem was found, 2 when the source data is unusable.
    """
    args = parse_args(argv)
    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    for t in tasks:
        if t not in TASK_CHOICES:
            print("[n3d_qa.build_qa] unknown task %r; expected %s" % (t, list(TASK_CHOICES)), file=sys.stderr)
            return 2
    if args.drill:
        if args.max_records <= 0:
            args.max_records = 8
        if not args.doclines and not args.no_doclines:
            args.doclines = True
    config = build_zh_config(args)
    doclines_config = build_doclines_config(args)
    try:
        result = build_all(
            tasks=tasks,
            out_dir=str(args.out_dir),
            config=config,
            neg_per_question=int(args.neg_per_question),
            triviaqa_splits=[s.strip() for s in str(args.triviaqa_splits).split(",") if s.strip()],
            triviaqa_topn=int(args.triviaqa_topn),
            archive=str(args.archive),
            with_doclines=bool(args.doclines) and not bool(args.no_doclines),
            doc_query_ratio=float(args.doc_query_ratio),
            doc_neg_per_query=int(args.doc_neg_per_query),
            max_records=int(args.max_records),
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            doclines_config=doclines_config,
        )
    except (AdapterError, BuildQaError, bd.ArchiveIntegrityError, bd.BuildContractError) as exc:
        print("[n3d_qa.build_qa] build failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2
    print_report(result, config, doclines_config)
    return 1 if result.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())