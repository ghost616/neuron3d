"""n3d_qa.zh_features -- Chinese text feature specification (character-level 1/2/3-gram hashed bag).

Positioning
-----------
New **specification layer** of the generic QA dataset module; it runs alongside the existing
English (TriviaQA) specification and does NOT touch it:

* the existing English bag uses ``build_dataset.TOKEN_RE = [a-z0-9]+`` -> every non-ASCII
  character is dropped, so Chinese text yields an **all-zero** feature bag (the blocking issue
  this module removes);
* this module tokenizes by **character** and emits character 1/2/3-grams, folds the whole
  normalized text with ``str.casefold()`` first (so full-width/half-width and case differences
  collapse deterministically, and ASCII words keep working), and reuses the *same* deterministic
  primitives as the English spec: ``hashlib.blake2b`` bucket hashing and the L2 normalization
  convention "divide by the L2 norm; keep the zero vector as-is when the norm is 0".

Contract
--------
* No global RNG is consumed anywhere; every draw is driven by an explicit seed.
* Same input text + same :class:`ZhFeatureConfig` -> bit-identical vector (no caching involved
  in the value, caches only memoize).
* Every specification parameter (n-gram orders, buckets per order, salt, normalization, tokenizer
  rule) is recorded in the product meta and folded into a single ``spec_hash``.
* Zero new dependencies (stdlib + numpy only).

Dimension discipline
--------------------
Per the standing correction (#12: when the sample count is of order 1e3 the bag bucket count
must be 10^1~10^2), the default here is ``n_gram_orders=(1,2,3)`` with ``buckets_per_order=100``
-> ``bag_dim = 300`` and ``D = 300 + 6 = 306`` for the QA side (about 1e4 pair rows);
the text-line side uses ``buckets_per_order=64`` -> ``D = 198`` (2665 rows).
The actually used values are read back from meta, never hardcoded downstream.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---- deterministic hashing specification (same algorithm as the English spec) ----------------
# A dedicated salt namespace: the Chinese bag must never collide with the English bag buckets.
ZH_HASH_SALT: bytes = b"n3d-qa-zh-charbow-v1\x00"
HASH_ALGO: str = "blake2b"
HASH_DIGEST_SIZE: int = 8
# The unit string fed to blake2b is f"{order}:{ngram}"; the colon is the field separator.
NGRAM_UNIT_TEMPLATE: str = "{order}:{ngram}"

# ---- normalization specification ------------------------------------------------------------
NORMALIZATION_RULE: str = (
    "NFKC -> drop all Unicode whitespace -> casefold() (per character, length preserving on the "
    "canonical plane); pure-ASCII bytes are all < 0x80, so a UTF-8 product is byte-stable on any "
    "platform"
)
TOKENIZER_RULE: str = (
    "character level: take the normalized character sequence (whitespace already dropped), emit "
    "every contiguous n-gram for n in n_gram_orders"
)

# ---- default specification ------------------------------------------------------------------
DEFAULT_N_GRAM_ORDERS: Tuple[int, ...] = (1, 2, 3)
DEFAULT_BUCKETS_PER_ORDER: int = 100
TEXT_BUCKETS_PER_ORDER: int = 64
EXTRA_DIM: int = 6

# ---- extra-column names (shared by the QA side and the text-line side) ----------------------
COL_QCOV: str = "q_to_c_coverage"
COL_CCOV: str = "c_to_q_coverage"
COL_JACCARD: str = "jaccard"
COL_LOGC: str = "log1p_candidate_chars"
COL_LOGQ: str = "log1p_query_chars"
COL_LENRATIO: str = "length_ratio"

EXTRA_COLUMN_NAMES: Tuple[str, ...] = (
    COL_QCOV, COL_CCOV, COL_JACCARD, COL_LOGC, COL_LOGQ, COL_LENRATIO,
)


class ZhFeatureError(RuntimeError):
    """Raised when the Chinese feature specification is violated (bad config / shape mismatch)."""


def zh_normalize(text: str) -> str:
    """Normalize text per :data:`NORMALIZATION_RULE` (NFKC -> strip all whitespace -> casefold).

    Parameters
    ----------
    text : str
        Raw text (any Unicode).

    Returns
    -------
    str
        Normalized character sequence; whitespace-free.

    Notes
    -----
    ``str.split()`` splits on every Unicode whitespace character (including U+3000 IDEOGRAPHIC
    SPACE), which is exactly the "drop all whitespace" rule wanted here. ``casefold()`` is applied
    **after** NFKC so that compatibility forms (e.g. U+FF21 FULLWIDTH LATIN CAPITAL A) collapse
    before case folding; the resulting text is pure ASCII plus CJK, hence byte-stable in UTF-8.
    """
    return "".join(unicodedata.normalize("NFKC", str(text)).split()).casefold()


def char_ngrams(text: str, orders: Sequence[int] = DEFAULT_N_GRAM_ORDERS) -> List[Tuple[int, str]]:
    """Enumerate character n-grams of the normalized text as ``(order, ngram)`` units.

    Parameters
    ----------
    text : str
        Raw text; normalized internally (do not pre-normalize, it is idempotent anyway).
    orders : Sequence[int]
        n-gram orders to emit (e.g. ``(1, 2, 3)``).

    Returns
    -------
    List[Tuple[int, str]]
        One entry per occurrence (repetitions kept, so term frequency is preserved).

    Examples
    --------
    ``char_ngrams("ab", (1, 2))`` -> ``[(1, "a"), (1, "b"), (2, "ab")]`` for normalized "ab".
    """
    s = zh_normalize(text)
    out: List[Tuple[int, str]] = []
    length = len(s)
    for n in orders:
        n = int(n)
        if n <= 0:
            raise ZhFeatureError("n-gram order must be >= 1, got %r" % (n,))
        if n > length:
            continue
        for i in range(0, length - n + 1):
            out.append((n, s[i:i + n]))
    return out


def unit_hash(unit: str) -> int:
    """Deterministic 64-bit integer hash of one hashing unit (``blake2b(salt + unit)``).

    Parameters
    ----------
    unit : str
        Hashing unit, i.e. ``f"{order}:{ngram}"`` (see :data:`NGRAM_UNIT_TEMPLATE`).

    Returns
    -------
    int
        ``int.from_bytes(blake2b(salt + unit.encode("utf-8"), digest_size=8).digest(), "big")``
        in ``[0, 2**64)``; determinism is guaranteed by blake2b, and no RNG is consumed.
    """
    digest = hashlib.blake2b(
        ZH_HASH_SALT + unit.encode("utf-8"), digest_size=int(HASH_DIGEST_SIZE)
    ).digest()
    return int.from_bytes(digest, "big")


def unit_bucket(unit: str, buckets: int) -> int:
    """Bucket index of a hashing unit: ``unit_hash(unit) % buckets``.

    Parameters
    ----------
    unit : str
        Hashing unit (see :func:`unit_hash`).
    buckets : int
        Bucket count of the block (must be >= 1).

    Returns
    -------
    int
        ``[0, buckets)``.
    """
    n = int(buckets)
    if n < 1:
        raise ZhFeatureError("buckets must be >= 1, got %r" % (n,))
    return int(unit_hash(unit) % n)


@dataclass(frozen=True)
class ZhFeatureConfig:
    """Chinese text feature specification (frozen; every field is recorded in the product meta).

    Attributes
    ----------
    n_gram_orders : Tuple[int, ...]
        n-gram orders, ascending (default ``(1, 2, 3)``).
    buckets_per_order : int
        Bucket count **per order**; the bag block is laid out order by order, so
        ``bag_dim = buckets_per_order * len(n_gram_orders)``.
    salt_hex : str
        blake2b salt as hex (default :data:`ZH_HASH_SALT`).
    normalization : str
        Normalization rule text (:data:`NORMALIZATION_RULE`).
    tokenizer : str
        Tokenizer rule text (:data:`TOKENIZER_RULE`).
    """

    n_gram_orders: Tuple[int, ...] = DEFAULT_N_GRAM_ORDERS
    buckets_per_order: int = DEFAULT_BUCKETS_PER_ORDER
    salt_hex: str = field(default_factory=lambda: ZH_HASH_SALT.hex())
    normalization: str = NORMALIZATION_RULE
    tokenizer: str = TOKENIZER_RULE

    def __post_init__(self) -> None:
        orders = tuple(int(x) for x in self.n_gram_orders)
        if not orders or any(x < 1 for x in orders):
            raise ZhFeatureError("n_gram_orders must be a non-empty list of ints >= 1: %r" % (orders,))
        if len(set(orders)) != len(orders):
            raise ZhFeatureError("n_gram_orders must not repeat: %r" % (orders,))
        if int(self.buckets_per_order) < 1:
            raise ZhFeatureError("buckets_per_order must be >= 1: %r" % (self.buckets_per_order,))
        object.__setattr__(self, "n_gram_orders", orders)
        object.__setattr__(self, "buckets_per_order", int(self.buckets_per_order))

    @property
    def bag_dim(self) -> int:
        """Width of the hashed character n-gram block (``buckets_per_order * len(orders)``)."""
        return int(self.buckets_per_order) * len(self.n_gram_orders)

    @property
    def feature_dim(self) -> int:
        """Total feature width ``D = bag_dim + EXTRA_DIM``."""
        return int(self.bag_dim) + int(EXTRA_DIM)

    def order_offsets(self) -> Tuple[int, ...]:
        """Start column of each order block (ascending, first is 0)."""
        return tuple(int(i) * int(self.buckets_per_order) for i in range(len(self.n_gram_orders)))

    def spec_dict(self) -> Dict[str, Any]:
        """The specification as a plain dict (goes verbatim into the product meta)."""
        return {
            "version": "zh-charbow-v1",
            "language": "zh",
            "tokenizer": self.tokenizer,
            "normalization": self.normalization,
            "n_gram_orders": [int(x) for x in self.n_gram_orders],
            "buckets_per_order": int(self.buckets_per_order),
            "bag_dim": int(self.bag_dim),
            "extra_dim": int(EXTRA_DIM),
            "feature_dim": int(self.feature_dim),
            "hash_algo": HASH_ALGO,
            "hash_digest_size": int(HASH_DIGEST_SIZE),
            "hash_salt_hex": str(self.salt_hex),
            "hash_unit_template": NGRAM_UNIT_TEMPLATE,
            "hash_bucket_rule": (
                "int.from_bytes(blake2b(salt + f'{order}:{ngram}').digest(), 'big') "
                "% buckets_per_order, block offset = order_index * buckets_per_order"
            ),
            "block_normalization": "L2 (divide by the block L2 norm; zero norm -> keep all-zero)",
            "l2_normalized": True,
            "bag_dim_rule": "buckets_per_order * len(n_gram_orders)",
            "feature_dim_rule": "bag_dim + 6",
        }

    def spec_hash(self) -> str:
        """SHA-256 of the canonical JSON of :meth:`spec_dict` (the "spec hash" in the meta).

        Returns
        -------
        str
            Lowercase hex digest; changes iff any specification parameter changes.
        """
        blob = json.dumps(
            self.spec_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


DEFAULT_ZH_CONFIG: ZhFeatureConfig = ZhFeatureConfig()


def extra_columns(base: int) -> Tuple[Dict[str, Any], ...]:
    """Per-column definitions of the 6 shared extra features, starting at column ``base``.

    Parameters
    ----------
    base : int
        First column index of the extra block (``bag_dim`` when the bag is present).

    Returns
    -------
    Tuple[Dict[str, Any], ...]
        Six ``{"start", "end", "name", "definition"}`` entries covering ``base .. base+5``.
    """
    b = int(base)
    return (
        {
            "start": b,
            "end": b,
            "name": COL_QCOV,
            "definition": "|G(q) & G(c)| / |G(q)| (char n-gram universe of both sides; 0 if |G(q)|=0)",
        },
        {
            "start": b + 1,
            "end": b + 1,
            "name": COL_CCOV,
            "definition": "|G(q) & G(c)| / |G(c)| (char n-gram universe of both sides; 0 if |G(c)|=0)",
        },
        {
            "start": b + 2,
            "end": b + 2,
            "name": COL_JACCARD,
            "definition": "|G(q) & G(c)| / |G(q) | G(c)| (char n-gram universe; 0 if the union is empty)",
        },
        {
            "start": b + 3,
            "end": b + 3,
            "name": COL_LOGC,
            "definition": "log1p(number of characters of the candidate side after normalization)",
        },
        {
            "start": b + 4,
            "end": b + 4,
            "name": COL_LOGQ,
            "definition": "log1p(number of characters of the query side after normalization)",
        },
        {
            "start": b + 5,
            "end": b + 5,
            "name": COL_LENRATIO,
            "definition": "len(candidate) / len(query) (0.0 when the query side is empty)",
        },
    )


def feature_columns(config: ZhFeatureConfig = DEFAULT_ZH_CONFIG) -> Tuple[Dict[str, Any], ...]:
    """Full column layout: one entry per n-gram order block + the 6 extra columns.

    Parameters
    ----------
    config : ZhFeatureConfig
        Effective specification.

    Returns
    -------
    Tuple[Dict[str, Any], ...]
        ``len(n_gram_orders) + 6`` entries covering ``[0, config.feature_dim - 1]`` seamlessly.
    """
    cols: List[Dict[str, Any]] = []
    for idx, order in enumerate(config.n_gram_orders):
        start = int(idx) * int(config.buckets_per_order)
        end = start + int(config.buckets_per_order) - 1
        cols.append(
            {
                "start": start,
                "end": end,
                "name": "zh_char%dgram_hash%d" % (int(order), int(config.buckets_per_order)),
                "definition": (
                    "character %d-gram hashed bag: take the normalized character sequence, emit every "
                    "contiguous %d-gram, hash it as blake2b(salt + '%d:{ngram}') mod %d, count, then L2-"
                    "normalize this block (0 norm -> all-zero). No global RNG." % (
                        int(order), int(order), int(order), int(config.buckets_per_order)
                    )
                ),
            }
        )
    return tuple(cols) + extra_columns(config.bag_dim)


def hash_bag_vector(
    text: str, config: ZhFeatureConfig = DEFAULT_ZH_CONFIG, counters: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """Hashed character n-gram **count** vector of one text (no normalization).

    Parameters
    ----------
    text : str
        Raw text.
    config : ZhFeatureConfig
        Effective specification.
    counters : Optional[Dict[str, int]]
        Optional diagnostics sink: ``"ngrams"`` total emissions and ``"oov_chars"`` number of
        characters outside ASCII/CJK (informational only).

    Returns
    -------
    np.ndarray
        ``float32[bag_dim]`` term frequencies. Additivity note: unlike the whitespace-tokenized
        English bag, character n-grams **straddle concatenation boundaries**, so
        ``bag(q + c) != bag(q) + bag(c)``; the vector is therefore always computed from the
        concatenated text, never assembled from cached halves.
    """
    vec = np.zeros(int(config.bag_dim), dtype=np.float32)
    units = char_ngrams(text, config.n_gram_orders)
    if counters is not None:
        counters["ngrams"] = int(counters.get("ngrams", 0)) + len(units)
        counters["oov_chars"] = int(counters.get("oov_chars", 0)) + sum(
            1 for ch in zh_normalize(text) if not (ch.isascii() or "\u4e00" <= ch <= "\u9fff")
        )
    if not units:
        return vec
    orders = list(config.n_gram_orders)
    buckets = int(config.buckets_per_order)
    idx = np.fromiter(
        (
            orders.index(int(o)) * buckets + unit_bucket(NGRAM_UNIT_TEMPLATE.format(order=int(o), ngram=g), buckets)
            for o, g in units
        ),
        dtype=np.int64,
        count=len(units),
    )
    counts = np.bincount(idx, minlength=int(config.bag_dim))
    return counts.astype(np.float32)


def ngram_universe(text: str, config: ZhFeatureConfig = DEFAULT_ZH_CONFIG) -> frozenset:
    """Set of hashing units (``(order, ngram)``) of one text — the "G(.)" universe of the extra columns.

    Parameters
    ----------
    text : str
        Raw text.
    config : ZhFeatureConfig
        Effective specification (only the orders matter here).

    Returns
    -------
    frozenset
        Unique ``(order, ngram)`` units.
    """
    return frozenset(char_ngrams(text, config.n_gram_orders))


def l2_normalize_block(vec: np.ndarray) -> np.ndarray:
    """In-place L2 normalization of a block (norm 0 keeps the zero vector, so no NaN appears).

    Parameters
    ----------
    vec : np.ndarray
        One-dimensional float32 slice of the feature row (normalized in place and returned).

    Returns
    -------
    np.ndarray
        The same array object.
    """
    norm = float(np.sqrt(np.dot(vec, vec)))
    if norm > 0.0:
        vec /= np.float32(norm)
    return vec


def build_feature_vector(
    query: str,
    candidate: str,
    config: ZhFeatureConfig = DEFAULT_ZH_CONFIG,
    q_units: Optional[frozenset] = None,
    c_units: Optional[frozenset] = None,
    counters: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Build one feature row ``[D] float32`` for a ``(query, candidate)`` text pair.

    Parameters
    ----------
    query : str
        Query side text (question text / query line).
    candidate : str
        Candidate side text (answer text / library line).
    config : ZhFeatureConfig
        Effective specification.
    q_units, c_units : Optional[frozenset]
        Pre-computed n-gram universes (recomputed when omitted).
    counters : Optional[Dict[str, int]]
        Optional diagnostics sink (see :func:`hash_bag_vector`), plus ``"empty_rows"`` when the
        whole bag block came out all-zero.

    Returns
    -------
    np.ndarray
        ``float32[config.feature_dim]``: concatenated normalized bag block followed by the 6 extra
        columns. Deterministic and allocation-local (no shared mutable state, no RNG).
    """
    joined = zh_normalize(query) + "\n" + zh_normalize(candidate)
    row = np.zeros(int(config.feature_dim), dtype=np.float32)
    bag = hash_bag_vector(joined, config, counters)
    if counters is not None and not bool(bag.any()):
        counters["empty_rows"] = int(counters.get("empty_rows", 0)) + 1
    row[: int(config.bag_dim)] = l2_normalize_block(bag)
    if q_units is None:
        q_units = ngram_universe(query, config)
    if c_units is None:
        c_units = ngram_universe(candidate, config)
    inter = len(q_units & c_units)
    union = len(q_units | c_units)
    base = int(config.bag_dim)
    row[base + 0] = float(inter) / float(len(q_units)) if q_units else 0.0
    row[base + 1] = float(inter) / float(len(c_units)) if c_units else 0.0
    row[base + 2] = float(inter) / float(union) if union else 0.0
    n_q = len(zh_normalize(query))
    n_c = len(zh_normalize(candidate))
    row[base + 3] = float(np.log1p(n_c))
    row[base + 4] = float(np.log1p(n_q))
    row[base + 5] = float(n_c) / float(n_q) if n_q > 0 else 0.0
    return row