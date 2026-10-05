"""n3d_qa.adapters -- dataset adapters of the generic QA module (Math1 QA + doc text lines).

Positioning
-----------
Per the module convention ("new datasets plug in as adapters: input parsing / document location /
answer merging / split table; they share the feature layer and the npz writer and must not leak
their own format assumptions into the generic layer"), this file holds the *dataset-specific*
input parsing only:

* **Math1** (``Kupasai___HighQualityEducationCoTDataset-Math1``): eight ``.jsonl`` files under
  ``data/kupasai/math1/.../data/{<subject>}/{<subject>_<qtype>}.jsonl`` with fields
  ``id / subject / qtype / question / choices / answer(list) / explanation / sampling_results``.
* **data/doc** (``a.md`` .. ``d.md``): non-empty lines are the feature units.

Everything else (Chinese character n-gram features, determinism, npz writing) lives in
:mod:`n3d_qa.zh_features` and :mod:`n3d_qa.build_dataset` and is shared with the TriviaQA path.

Determinism
-----------
No global RNG: negative sampling uses ``np.random.default_rng(seed)`` with an explicit seed, and
every derived id is a blake2b hash of the row fields (so ids are stable across runs and
machine-independent). Nothing here writes files.
"""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # dual-mode import: package module or directly executed script
    from .zh_features import DEFAULT_ZH_CONFIG, ZhFeatureConfig
except ImportError:  # pragma: no cover - direct script execution
    from zh_features import DEFAULT_ZH_CONFIG, ZhFeatureConfig  # type: ignore

PROJECT_ROOT: str = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
MATH1_DIR: str = os.path.join(
    PROJECT_ROOT, "data", "kupasai", "math1",
    "Kupasai___HighQualityEducationCoTDataset-Math1", "data",
)
DOC_DIR: str = os.path.join(PROJECT_ROOT, "data", "doc")

# Window form of the field names, kept ASCII so this source file stays byte-stable everywhere.
SUBJECTS: Tuple[str, ...] = ("\u79bb\u6563\u6570\u5b66", "\u9ad8\u7b49\u6570\u5b66")
QTYPE_JUDGE: str = "\u5224\u65ad\u9898"
QTYPE_BLANK: str = "\u586b\u7a7a\u9898"
QTYPE_SOLVE: str = "\u89e3\u7b54\u9898"
QTYPE_CHOICE: str = "\u9009\u62e9\u9898"
QTYPES: Tuple[str, ...] = (QTYPE_JUDGE, QTYPE_BLANK, QTYPE_SOLVE, QTYPE_CHOICE)
MATH1_TASKS: Tuple[str, ...] = ("judge", "choice", "blank", "solve")
TASK_QTYPE: Dict[str, str] = {
    "judge": QTYPE_JUDGE, "choice": QTYPE_CHOICE, "blank": QTYPE_BLANK, "solve": QTYPE_SOLVE,
}

# Answer normalization rule (explicit, asserted in verify) ------------------------------------
# 1) NFKC (full-width -> half-width, compatibility folding) 2) strip ALL Unicode whitespace
# 3) casefold. Measured effect on the raw data (see tests/README): the number of distinct
# answer strings shrinks, e.g. the blank-question file 339 -> 332 raw pair counts and the
# solve-question file 564 -> 558 in the discrete-math file.
ANSWER_NORM_RULE: str = "NFKC -> strip all Unicode whitespace -> casefold()"


class AdapterError(RuntimeError):
    """Raised when a dataset file violates its on-site verified format."""


# --------------------------------------------------------------------------------------------
# generic small helpers
# --------------------------------------------------------------------------------------------
def normalize_answer_key(value: Any) -> str:
    """Normalize one answer string to its canonical key (see :data:`ANSWER_NORM_RULE`).

    Parameters
    ----------
    value : Any
        Raw answer token (coerced with ``str``).

    Returns
    -------
    str
        Canonical key: ``"".join(unicodedata.normalize("NFKC", str(value)).split()).casefold()``.
        The empty string is returned for a whitespace-only input; callers must drop it.
    """
    return "".join(unicodedata.normalize("NFKC", str(value)).split()).casefold()


def answer_key_variants(value: Any) -> List[str]:
    """The canonical key of one raw answer string as a one-element list (the single key space).

    [!] This function used to return **two** keys per value (the canonical form plus a
    "whitespace-stripped, casefolded only" form). That second form made the alias index of the
    unified answer table contain keys that the per-task selection rules never see
    (``'"like a prayer"'`` vs ``'likeaprayer'``), so the builder's key-union assertion failed on the
    first full run. The rule is now single-valued on purpose: **one canonical key per raw value**,
    so selection, table construction, coverage and label decisions all share the same key space.

    Parameters
    ----------
    value : Any
        Raw answer token.

    Returns
    -------
    List[str]
        ``[normalize_answer_key(value)]``, or ``[]`` when that key is empty.
    """
    key = normalize_answer_key(value)
    return [key] if key else []


def sample_id(*parts: Any) -> str:
    """Deterministic 16-hex-char sample id from the given parts (blake2b, no RNG).

    Parameters
    ----------
    *parts : Any
        Id parts, joined with ``"\\x1f"`` and encoded as UTF-8.

    Returns
    -------
    str
        ``blake2b(digest_size=8)`` big-endian hex; stable across runs and machines.
    """
    blob = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def jsonl_escape(text: str) -> str:
    """Escape a string for the deterministic JSONL writer (LF stays LF, no ``ensure_ascii``).

    Parameters
    ----------
    text : str
        Raw string.

    Returns
    -------
    str
        JSON-encoded string literal **without** the surrounding quotes; ``\\n`` inside the text
        becomes the two-character escape ``\\n`` so one record is always exactly one line.
    """
    return json.dumps(str(text), ensure_ascii=False)[1:-1]


# --------------------------------------------------------------------------------------------
# Math1 parsing
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class QaItem:
    """One Math1 record in its build view (answer list joined, choices folded into the question).

    Attributes
    ----------
    qid : str
        Source record id (``id`` field).
    subject : str
        Source subject directory (discrete math / higher math).
    qtype : str
        Source question type (judge / blank / solve / choice).
    question : str
        ``question`` + all ``choices`` joined by ``"\\n"`` (the spec's "question text").
    answer_values : Tuple[str, ...]
        Raw ``answer`` list entries (order preserved, duplicates dropped).
    explanation : str
        ``explanation`` text (the spec's "answer text" is ``answer`` + ``explanation``).
    n_choices : int
        Explicit option count (0 when ``choices`` is null).
    """

    qid: str
    subject: str
    qtype: str
    question: str
    answer_values: Tuple[str, ...]
    explanation: str
    n_choices: int

    @property
    def question_text(self) -> str:
        """Question text as consumed by the feature layer (``question`` + merged ``choices``)."""
        return self.question

    @property
    def answer_text(self) -> str:
        """Answer text as consumed by the feature layer (``answer`` joined + ``explanation``)."""
        joined = "\n".join(str(a) for a in self.answer_values)
        return joined + "\n" + str(self.explanation)


def math1_path(subject: str, qtype: str) -> str:
    """Path of one Math1 jsonl file (``<subject>_<qtype>.jsonl`` inside the subject directory).

    Parameters
    ----------
    subject : str
        Subject directory name.
    qtype : str
        Question-type name.

    Returns
    -------
    str
        Absolute path.
    """
    return os.path.join(MATH1_DIR, str(subject), "%s_%s.jsonl" % (str(subject), str(qtype)))


def parse_math1_line(line: str, where: str) -> Dict[str, Any]:
    """Parse and validate one Math1 jsonl line (missing / mistyped fields raise).

    Parameters
    ----------
    line : str
        One raw line (already stripped).
    where : str
        Location text used in error messages.

    Returns
    -------
    Dict[str, Any]
        The parsed object.

    Raises
    ------
    AdapterError
        When the JSON is invalid, the top level is not an object, or a required field is missing
        or of the wrong type.
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise AdapterError("%s is not valid JSON: %s" % (where, exc)) from exc
    if not isinstance(obj, dict):
        raise AdapterError("%s must be an object, got %s" % (where, type(obj).__name__))
    for key in ("id", "subject", "qtype", "question", "answer", "explanation"):
        if key not in obj:
            raise AdapterError(
                "%s is missing field %r; actual keys %s" % (where, key, sorted(obj.keys()))
            )
    if not isinstance(obj["answer"], list) or not obj["answer"]:
        raise AdapterError("%s.answer must be a non-empty list, got %r" % (where, obj["answer"]))
    if not isinstance(obj["question"], str) or not obj["question"].strip():
        raise AdapterError("%s.question must be a non-empty string" % where)
    choices = obj.get("choices")
    if choices is not None and not isinstance(choices, list):
        raise AdapterError("%s.choices must be a list or null, got %s" % (where, type(choices).__name__))
    return obj


def load_math1_task(
    task: str, max_records: int = 0, subjects: Sequence[str] = SUBJECTS
) -> List[QaItem]:
    """Load one Math1 task (a question type across both subjects), deterministically ordered.

    Parameters
    ----------
    task : str
        One of :data:`MATH1_TASKS` (``judge`` / ``choice`` / ``blank`` / ``solve``).
    max_records : int
        Optional cap on the number of merged records (``0`` = no cap); applied **after** the
        deterministic sort, so a capped run is a prefix of the full run.
    subjects : Sequence[str]
        Subject directories to read, in order.

    Returns
    -------
    List[QaItem]
        Records sorted by ``(subject, qid)``.

    Raises
    ------
    AdapterError
        Unknown task name, or a source file missing.
    """
    if str(task) not in TASK_QTYPE:
        raise AdapterError("unknown math1 task %r; expected one of %s" % (task, list(MATH1_TASKS)))
    qtype = TASK_QTYPE[str(task)]
    items: List[QaItem] = []
    for subject in subjects:
        path = math1_path(subject, qtype)
        if not os.path.isfile(path):
            raise AdapterError("math1 source file missing: %s" % path)
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                where = "%s:%d" % (os.path.relpath(path, PROJECT_ROOT), lineno)
                obj = parse_math1_line(line, where)
                if str(obj["qtype"]) != qtype:
                    raise AdapterError(
                        "%s declares qtype %r but the file is %r" % (where, obj["qtype"], qtype)
                    )
                if str(obj["subject"]) != str(subject):
                    raise AdapterError(
                        "%s declares subject %r but the directory is %r"
                        % (where, obj["subject"], subject)
                    )
                choices = obj.get("choices") or []
                qtext = str(obj["question"])
                if choices:
                    qtext = qtext + "\n" + "\n".join(str(c) for c in choices)
                answers = tuple(dict.fromkeys(str(a) for a in obj["answer"]))
                items.append(
                    QaItem(
                        qid=str(obj["id"]),
                        subject=str(subject),
                        qtype=qtype,
                        question=qtext,
                        answer_values=answers,
                        explanation=str(obj["explanation"]),
                        n_choices=len(choices),
                    )
                )
    items.sort(key=lambda it: (it.subject, it.qid))
    if int(max_records) > 0:
        items = items[: int(max_records)]
    return items


# --------------------------------------------------------------------------------------------
# unified answer table (cross-task merge + dedup + per-task counts)
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AnswerClass:
    """One class of the unified answer table.

    Attributes
    ----------
    key : str
        Canonical answer key (:func:`normalize_answer_key`).
    display : str
        Representative raw surface form (the most frequent raw string mapping onto ``key``).
    count : int
        Number of source questions whose answer set maps onto ``key`` (counting each question
        once, across all contributing tasks).
    sources : Tuple[Tuple[str, int], ...]
        ``(task, count)`` pairs, sorted by task name.
    raw_variants : int
        Number of distinct raw surface strings that map onto ``key``.
    """

    key: str
    display: str
    count: int
    sources: Tuple[Tuple[str, int], ...]
    raw_variants: int


@dataclass
class AnswerTable:
    """The unified answer table plus the aliases needed to keep the multi-value answer paths.

    Attributes
    ----------
    classes : Tuple[AnswerClass, ...]
        Classes sorted by ``(-count, key)``.
    index_by_key : Dict[str, int]
        ``key -> row index in classes``.
    alias_index : Dict[str, int]
        ``alias key -> row index``; an alias may be a single value of a multi-value answer.
    n_questions_seen : int
        Number of source questions that entered the merge.
    n_questions_in_table : int
        Number of source questions whose answer set has at least one key present in the table
        (the complement is the "not relevant" bucket of the product).
    n_multi_value_questions : int
        Source questions with more than one answer value.
    """

    classes: Tuple[AnswerClass, ...]
    index_by_key: Dict[str, int]
    alias_index: Dict[str, int]
    n_questions_seen: int
    n_questions_in_table: int
    n_multi_value_questions: int

    @property
    def size(self) -> int:
        """Number of classes in the table."""
        return len(self.classes)


def _task_answer_counts(items: Sequence[QaItem]) -> Dict[str, int]:
    """Per-task ``answer key -> number of source questions`` counter.

    Parameters
    ----------
    items : Sequence[QaItem]
        Source records of that task.

    Returns
    -------
    Dict[str, int]
        Number of questions whose answer set contains the key (a question counts once per key).
    """
    counts: Dict[str, int] = {}
    for it in items:
        keys = {k for a in it.answer_values for k in answer_key_variants(a)}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
    return counts


def _raw_display_counts(items: Sequence[QaItem]) -> Dict[str, Dict[str, int]]:
    """Per-key counter of raw surface strings (used to pick a representative display form)."""
    out: Dict[str, Dict[str, int]] = {}
    for it in items:
        for a in it.answer_values:
            key = normalize_answer_key(a)
            if not key:
                continue
            out.setdefault(key, {})
            raw = str(a).strip()
            out[key][raw] = out[key].get(raw, 0) + 1
    return out


def build_answer_table(
    task_items: Dict[str, Sequence[QaItem]], task_selection: Dict[str, str]
) -> AnswerTable:
    """Merge the per-task answer selections into the unified, deduplicated answer table.

    Parameters
    ----------
    task_items : Dict[str, Sequence[QaItem]]
        ``task -> records`` for every contributing task.
    task_selection : Dict[str, str]
        ``task -> selection rule``; the rule text is recorded verbatim in the meta. The rules used
        by the QA build are:

        * ``judge``: all answer values (the two boolean classes);
        * ``choice``: answer values that are a single ASCII letter A-D (multi-letter and E+ are
          registered as removals);
        * ``blank`` / ``solve``: answer keys whose normalized frequency across the task is
          ``>= min_repeat`` (5);
        * ``triviaqa``: the top-N most frequent answer keys of the archive splits.

    Returns
    -------
    AnswerTable
        Merged table; ``count`` is the number of source questions per class summed over tasks.

    Notes
    -----
    Aliases matter: a source question may carry several answer values (synonyms) that map onto
    *different* keys, so coverage is computed per key-set intersection, and a raw candidate
    drawn from the alias index resolves to its class.
    """
    per_task: Dict[str, Dict[str, int]] = {}
    for task in sorted(task_items):
        per_task[task] = _task_answer_counts(task_items[task])
    # union of keys, per-task counts summed (a question counts once per task)
    total: Dict[str, int] = {}
    sources: Dict[str, Dict[str, int]] = {}
    for task, counts in per_task.items():
        for key, cnt in counts.items():
            total[key] = total.get(key, 0) + int(cnt)
            sources.setdefault(key, {})[task] = int(cnt)
    display: Dict[str, Dict[str, int]] = {}
    for task in sorted(task_items):
        for key, raws in _raw_display_counts(task_items[task]).items():
            bucket = display.setdefault(key, {})
            for raw, cnt in raws.items():
                bucket[raw] = bucket.get(raw, 0) + int(cnt)
    classes = tuple(
        AnswerClass(
            key=key,
            display=(
                sorted(display.get(key, {"": 0}).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                if display.get(key)
                else key
            ),
            count=int(total[key]),
            sources=tuple(sorted(sources[key].items())),
            raw_variants=len(display.get(key, {})),
        )
        for key in sorted(total, key=lambda k: (-total[k], k))
    )
    index_by_key = {c.key: i for i, c in enumerate(classes)}
    alias_index: Dict[str, int] = {}
    for i, c in enumerate(classes):
        alias_index[c.key] = i
    # multi-value answers: register every single value as an alias of its class
    n_multi = 0
    seen_q = 0
    for task in sorted(task_items):
        for it in task_items[task]:
            seen_q += 1
            if len(it.answer_values) > 1:
                n_multi += 1
            for a in it.answer_values:
                for k in answer_key_variants(a):
                    if k in index_by_key:
                        alias_index.setdefault(k, index_by_key[k])
    # coverage: a question is "in table" iff at least one of its answer keys is a table class
    n_in = 0
    for task in sorted(task_items):
        for it in task_items[task]:
            keys = {normalize_answer_key(a) for a in it.answer_values}
            if any(k in index_by_key for k in keys):
                n_in += 1
    return AnswerTable(
        classes=classes,
        index_by_key=index_by_key,
        alias_index=alias_index,
        n_questions_seen=seen_q,
        n_questions_in_table=n_in,
        n_multi_value_questions=n_multi,
    )


def question_answer_indices(it: QaItem, alias_index: Dict[str, int]) -> Tuple[int, ...]:
    """Table row indices of a record's correct answers (alias-resolved, ascending, deduped).

    Parameters
    ----------
    it : QaItem
        Source record.
    alias_index : Dict[str, int]
        Alias-to-class index of the unified table.

    Returns
    -------
    Tuple[int, ...]
        Ascending tuple of distinct class rows; empty when the record is "not relevant" (its
        correct answer is absent from the unified table).
    """
    idxs = set()
    for a in it.answer_values:
        for k in answer_key_variants(a):
            j = alias_index.get(k)
            if j is not None:
                idxs.add(int(j))
    return tuple(sorted(idxs))


def task_class_mask(items: Sequence[QaItem], table: AnswerTable) -> Tuple[int, ...]:
    """Class rows that at least one record of this task actually answers (ascending).

    Parameters
    ----------
    items : Sequence[QaItem]
        Source records of one task.
    table : AnswerTable
        Unified answer table.

    Returns
    -------
    Tuple[int, ...]
        Class rows reachable from this task's correct answers; used to restrict the negative draw
        to the task's own answer family (label correctness does not depend on the restriction).
    """
    rows = set()
    for it in items:
        rows.update(question_answer_indices(it, table.alias_index))
    return tuple(sorted(rows))


# --------------------------------------------------------------------------------------------
# negative sampling for the QA match task
# --------------------------------------------------------------------------------------------
@dataclass
class QaPairSet:
    """Deterministic (query, candidate) pair set of one QA task.

    Attributes
    ----------
    task : str
        Task name (``judge`` / ``choice`` / ``blank`` / ``solve`` / ``triviaqa``).
    questions : Tuple[Dict[str, Any], ...]
        Per-pair question provenance (``qid`` / ``subject`` / ``qtype`` / ``n_choices`` /
        ``n_correct`` / ``id``).
    candidates : Tuple[Dict[str, Any], ...]
        Per-pair candidate provenance (``answer_index`` (class row) / ``key`` / ``display`` /
        ``label`` / ``alias_draw`` (kept for schema stability; equals the candidate key under the
        class-level draw rule) / ``id``).
    labels : np.ndarray
        ``int64[M]``; ``1`` = candidate is one of the question's correct answers.
    counters : Dict[str, Any]
        Measured counters: pairs, positives, negatives, questions, questions with 0 table
        answers ("not relevant"), irrelevant pair rows, negative draws, distinct question ids,
        distinct candidate keys, negative seed, negatives per question.
    """

    task: str
    questions: Tuple[Dict[str, Any], ...]
    candidates: Tuple[Dict[str, Any], ...]
    labels: np.ndarray
    counters: Dict[str, Any]


def class_sampling_weights(table: AnswerTable) -> np.ndarray:
    """Per-class draw weights of the negative sampler: normalized ``count`` (source-question count).

    Why class-level weighted draws
    ------------------------------
    The label rule must be stated, so the negative candidate is defined as *a wrong answer of the
    unified table drawn with probability proportional to its own margin over the source questions*.
    This is a well-defined distribution (frequent wrong answers are more likely) and, unlike
    drawing raw aliases and dropping collisions, it can always produce **exactly** the requested
    number of negatives (the first implementation drew raw aliases uniformly and had to drop the
    collisions, which for the two-class boolean task silently dropped about half of the draws).

    Parameters
    ----------
    table : AnswerTable
        Unified answer table.

    Returns
    -------
    np.ndarray
        ``float64[len(classes)]`` summing to 1.0 (uniform fallback when all counts are zero).

    Raises
    ------
    AdapterError
        When the answer table has no class at all.
    """
    counts = np.asarray([int(c.count) for c in table.classes], dtype=np.float64)
    if counts.size == 0:
        raise AdapterError("cannot build negative-sampling weights from an empty answer table")
    total = float(counts.sum())
    if total <= 0.0:
        return np.full(counts.size, 1.0 / counts.size, dtype=np.float64)
    return counts / total


def sample_wrong_answers(
    table: AnswerTable,
    n: int,
    rng: np.random.Generator,
    exclude: Sequence[int] = (),
    weights: Optional[np.ndarray] = None,
    restrict: Optional[Sequence[int]] = None,
) -> List[int]:
    """Draw ``n`` wrong candidate class rows (class-level, frequency weighted, correct ones banned).

    Parameters
    ----------
    table : AnswerTable
        Unified answer table.
    n : int
        Number of negatives to draw.
    rng : np.random.Generator
        Explicit generator (no global RNG is ever touched).
    exclude : Sequence[int]
        Class rows that must not be drawn (the question's own correct answers).
    weights : Optional[np.ndarray]
        Precomputed weights (see :func:`class_sampling_weights`); recomputed when omitted.
    restrict : Optional[Sequence[int]]
        When given, only these class rows may be drawn (weights of all other classes are zeroed
        and the distribution renormalized). Used by the math1 tasks to keep wrong candidates
        inside their own task family; label correctness never depends on it.

    Returns
    -------
    List[int]
        Exactly ``n`` drawn class rows (fewer only when the table offers no wrong class).

    Raises
    ------
    AdapterError
        When ``weights`` does not match the table size.
    """
    n_neg = int(n)
    if n_neg <= 0 or table.size == 0:
        return []
    w = class_sampling_weights(table) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (table.size,):
        raise AdapterError(
            "weights shape %r does not match the answer table size %d" % (w.shape, table.size)
        )
    banned = set(int(x) for x in exclude)
    w = w.copy()
    for j in banned:
        if 0 <= int(j) < w.size:
            w[int(j)] = 0.0
    if restrict is not None:
        allowed = set(int(x) for x in restrict)
        for j in range(w.size):
            if j not in allowed:
                w[j] = 0.0
    total = float(w.sum())
    if total <= 0.0:
        # every class is banned (only reachable for a single-class table): draw uniformly over the
        # remaining index set, which is empty in that corner case
        allowed = [i for i in range(table.size) if i not in banned]
        if not allowed:
            return []
        w = np.zeros(table.size, dtype=np.float64)
        for i in allowed:
            w[i] = 1.0
        total = float(w.sum())
    w = w / total
    return [int(x) for x in rng.choice(table.size, size=n_neg, replace=True, p=w).tolist()]


def build_qa_pairs(
    task: str,
    items: Sequence[QaItem],
    table: AnswerTable,
    seed: int,
    neg_per_question: int,
    restrict: Optional[Sequence[int]] = None,
    feature_config: Optional[ZhFeatureConfig] = None,
) -> QaPairSet:
    """Build the deterministic ``(question, candidate answer)`` pair set of one QA task.

    Label rule
    ----------
    * one **positive** row per (question, correct answer value in the table);
      the candidate text is that answer value **plus the question's explanation**;
    * ``neg_per_question`` **negative** rows per question, whose candidates are wrong answer
      classes drawn by :func:`sample_wrong_answers` (class-level, frequency weighted, the
      question's own correct classes banned -> the requested count is always met exactly);
      their candidate text is the drawn class's representative surface form plus the question's
      explanation;
    * a question whose correct answer is absent from the unified table contributes **no row at
      all** (the "not relevant" bucket); its count is reported, never silently dropped.

    Parameters
    ----------
    task : str
        Task name (recorded in the meta).
    items : Sequence[QaItem]
        Source records in deterministic order.
    table : AnswerTable
        Unified answer table.
    seed : int
        Seed of ``np.random.default_rng`` used for the negative draws.
    neg_per_question : int
        Negative rows per question (``0`` = positives only).
    restrict : Optional[Sequence[int]]
        Optional whitelist of class rows for the negative draw (task-family restriction).
    feature_config : Optional[ZhFeatureConfig]
        Effective Chinese feature spec, recorded in ``counters`` together with its spec hash; when
        ``None`` the default spec is recorded.

    Returns
    -------
    QaPairSet
        Pairs in the exact order they are emitted (question-major, positive first); ``counters``
        carries every measured quantity reported downstream.
    """
    rng = np.random.default_rng(int(seed))
    questions: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    labels: List[int] = []
    n_no_table = 0
    n_multi = 0
    n_neg_total = 0
    neg_weights = class_sampling_weights(table)
    key_set = set()
    for it in items:
        correct = question_answer_indices(it, table.alias_index)
        if len(it.answer_values) > 1:
            n_multi += 1
        if not correct:
            n_no_table += 1
            continue
        q_row = {
            "qid": it.qid,
            "subject": it.subject,
            "qtype": it.qtype,
            "n_choices": int(it.n_choices),
            "n_correct": int(len(correct)),
            "correct_indices": [int(j) for j in correct],
            "id": sample_id("q", task, it.subject, it.qid),
        }
        for j in correct:
            questions.append(q_row)
            candidates.append(
                {
                    "answer_index": int(j),
                    "key": table.classes[int(j)].key,
                    "display": table.classes[int(j)].display,
                    "label": 1,
                    "alias_draw": "",
                    "id": sample_id("c", task, it.qid, "pos", int(j)),
                }
            )
            labels.append(1)
            key_set.add(int(j))
        n_neg = int(neg_per_question)
        if n_neg > 0:
            chosen = sample_wrong_answers(
                table, n_neg, rng, exclude=correct, weights=neg_weights, restrict=restrict
            )
            n_neg_total += len(chosen)
            for k, j in enumerate(chosen):
                questions.append(q_row)
                candidates.append(
                    {
                        "answer_index": int(j),
                        "key": table.classes[int(j)].key,
                        "display": table.classes[int(j)].display,
                        "label": 0,
                        "alias_draw": table.classes[int(j)].key,
                        "id": sample_id("c", task, it.qid, "neg", int(k), int(j)),
                    }
                )
                labels.append(0)
                key_set.add(int(j))
    counters = {
        "task": str(task),
        "pairs": int(len(labels)),
        "positives": int(sum(labels)),
        "negatives": int(len(labels) - sum(labels)),
        "questions_total": int(len(items)),
        "questions_paired": int(len(items) - n_no_table),
        "questions_not_relevant": int(n_no_table),
        "questions_multi_value": int(n_multi),
        "irrelevant_pair_rows": int(n_no_table * (1 + int(neg_per_question))),
        "negative_draws": int(n_neg_total),
        "negative_draw_rule": (
            "class-level weighted draw: p(class) = class.count / sum(count) over the unified "
            "answer table, the question's own correct classes excluded (banned)"
        ),
        "distinct_question_ids": int(len({q["qid"] for q in questions})),
        "distinct_candidate_keys": int(len(key_set)),
        "negative_seed": int(seed),
        "neg_per_question": int(neg_per_question),
    }
    spec = feature_config if feature_config is not None else DEFAULT_ZH_CONFIG
    counters["feature_spec_hash"] = spec.spec_hash()
    counters["feature_spec"] = spec.spec_dict()
    counters["restricted_classes"] = (int(len(restrict)) if restrict is not None else int(table.size))
    return QaPairSet(
        task=str(task),
        questions=tuple(questions),
        candidates=tuple(candidates),
        labels=np.asarray(labels, dtype=np.int64),
        counters=counters,
    )


# --------------------------------------------------------------------------------------------
# text-line adapter (data/doc)
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DocLine:
    """One feature unit of the text-line product: a non-empty line of one ``.md`` file.

    Attributes
    ----------
    row_id : str
        Deterministic row id (``sample_id("line", file, lineno)``).
    file_name : str
        Source file name (``a.md`` .. ``d.md``).
    line_no : int
        1-based physical line number inside the source file.
    text : str
        Original line text (trailing/leading whitespace stripped).
    """

    row_id: str
    file_name: str
    line_no: int
    text: str


def load_doc_lines(doc_dir: str = DOC_DIR, max_lines_per_file: int = 0) -> List[DocLine]:
    """Split ``*.md`` files into non-empty lines (the feature units), deterministically ordered.

    Parameters
    ----------
    doc_dir : str
        Directory holding the ``.md`` files.
    max_lines_per_file : int
        Optional per-file cap (``0`` = no cap); applied after sorting by line number, so a capped
        run is a prefix of the full run.

    Returns
    -------
    List[DocLine]
        Rows sorted by ``(file_name, line_no)``.

    Raises
    ------
    AdapterError
        When the directory holds no ``.md`` file.
    """
    names = sorted(n for n in os.listdir(doc_dir) if n.lower().endswith(".md"))
    if not names:
        raise AdapterError("no .md file found under %s" % doc_dir)
    rows: List[DocLine] = []
    for name in names:
        path = os.path.join(doc_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        kept = 0
        for lineno, raw in enumerate(content.split("\n"), start=1):
            text = raw.strip()
            if not text:
                continue
            rows.append(
                DocLine(
                    row_id=sample_id("line", name, lineno),
                    file_name=name,
                    line_no=int(lineno),
                    text=text,
                )
            )
            kept += 1
            if int(max_lines_per_file) > 0 and kept >= int(max_lines_per_file):
                break
    rows.sort(key=lambda r: (r.file_name, r.line_no))
    return rows


def split_library_query(
    rows: Sequence[DocLine], query_ratio: float, seed: int
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Split the text rows into the held-out "library" / "query" sets with an explicit seed.

    Parameters
    ----------
    rows : Sequence[DocLine]
        All text rows in deterministic order.
    query_ratio : float
        Fraction of rows assigned to the query set (``0 < ratio < 1``).
    seed : int
        Seed of ``np.random.default_rng`` (permutation only; no global RNG).

    Returns
    -------
    Tuple[Tuple[int, ...], Tuple[int, ...]]
        ``(library_indices, query_indices)``, each ascending, disjoint, and covering every row.

    Raises
    ------
    AdapterError
        When ``query_ratio`` is outside ``(0, 1)`` or the split would leave a side empty.
    """
    n = len(rows)
    ratio = float(query_ratio)
    if not (0.0 < ratio < 1.0):
        raise AdapterError("query_ratio must lie in (0, 1), got %r" % (query_ratio,))
    n_query = int(round(n * ratio))
    n_query = max(1, min(n - 1, n_query))
    if n <= 1:
        raise AdapterError("at least 2 rows are needed to split, got %d" % n)
    order = np.random.default_rng(int(seed)).permutation(n)
    query = np.sort(order[:n_query])
    library = np.sort(order[n_query:])
    return tuple(int(x) for x in library.tolist()), tuple(int(x) for x in query.tolist())