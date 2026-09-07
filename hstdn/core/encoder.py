"""encoder.py -- M2 input encoding (latency / MNIST pool / DVS interface).

- ``latency_encode``: first-spike latency coding of one frame
  ``t = enc_t0 + (1 - I) * enc_t_span`` for intensities I > enc_i_thr
  (default: t = 2 + (1-I)*40, only I > 0.12 fire).  The event list is sorted
  by spike time before bucketing (FIX-B8) and grouped into
  ``Dict[int, (ids, strengths)]`` where ids are **input global ids** (the
  state array layer is never touched here) and strengths default to 1.0.
- ``mnist_adaptive_pool``: pure-NumPy adaptive average pooling 28x28 -> 10x10
  = 100 units (D14; mirrors torch ``adaptive_avg_pool2d`` semantics).
- ``mnist_encode``: pools a 28x28 MNIST patch and latency-encodes the 100
  pooled units onto input ids 0..99 (row-major).
- ``dvs_patch_aggregate``: DVS patch aggregation to N_IN=512 input units
  (D14) -- interface only; implementation belongs to the DVS milestone and
  currently raises NotImplementedError (no pass placeholders, L1 rule).

Poisson encoding is disabled by default (``cfg.poisson_on == False``) and is
the scheduler's responsibility; this module stays deterministic.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from hstdn.core.layout import I8, F8, NetConfig, cfg_from_mapping

__all__ = ["latency_encode", "mnist_adaptive_pool", "mnist_encode",
           "dvs_patch_aggregate"]

BucketMap = Dict[int, Tuple[np.ndarray, np.ndarray]]


def latency_encode(intensities, *, cfg=None,
                   ids_offset: int = 0) -> BucketMap:
    """Latency-code a frame of input intensities into spike buckets.

    An input unit i fires once at absolute time
    ``t = enc_t0 + (1 - I_i) * enc_t_span`` (rounded to integer ms) iff
    ``I_i > enc_i_thr`` (strict; default 0.12).  Units at/under the threshold
    never fire.  Times are clipped to ``>= enc_t0`` (already guaranteed) and
    sorted before bucketing (FIX-B8); ids inside a bucket are ascending.

    Args:
        intensities: 1-D float array of unit intensities in [0, 1]
            (values > 1 or < 0 are clipped to the range first).
        cfg: NetConfig or mapping (defaults to module defaults).
        ids_offset: input global-id offset of the first unit (default 0).

    Returns:
        Dict[int, (ids, strengths)]: t_step -> (int64 input global ids sorted
        ascending, float64 strengths of 1.0).  Empty dict when no unit fires.

    Raises:
        AssertionError: intensities is not 1-D.
    """
    conf = cfg if isinstance(cfg, NetConfig) else cfg_from_mapping(cfg)
    arr = np.asarray(intensities, dtype=F8)
    if arr.ndim != 1:
        raise AssertionError(
            f"latency_encode expects 1-D intensities, got shape {arr.shape}"
        )
    arr = np.clip(arr, 0.0, 1.0)
    fire = arr > conf.enc_i_thr
    if not fire.any():
        return {}
    unit_ids = np.flatnonzero(fire).astype(I8) + int(ids_offset)
    times = np.rint(
        conf.enc_t0 + (1.0 - arr[fire]) * conf.enc_t_span
    ).astype(I8)
    # FIX-B8: sort the firing units by spike time (then by id for stability)
    order = np.lexsort((unit_ids, times))
    unit_ids = unit_ids[order]
    times = times[order]
    strengths = np.ones(unit_ids.size, dtype=F8)
    buckets: BucketMap = {}
    for t in np.unique(times):
        sel = unit_ids[times == t]
        buckets[int(t)] = (sel, strengths[: sel.size])
    return buckets


def _adaptive_bounds(n_in: int, n_out: int) -> np.ndarray:
    """Region boundaries of adaptive average pooling (torch semantics).

    Output unit j averages input rows ``[start_j, end_j)`` where
    ``start_j = floor(j * n_in / n_out)`` and
    ``end_j = ceil((j+1) * n_in / n_out)``.

    Args:
        n_in: input side length (e.g. 28).
        n_out: output side length (e.g. 10).

    Returns:
        (n_out, 2) int64 array with columns (start, end).
    """
    idx = np.arange(n_out, dtype=np.float64)
    starts = np.floor(idx * n_in / n_out).astype(np.int64)
    ends = np.ceil((idx + 1.0) * n_in / n_out).astype(np.int64)
    return np.stack([starts, ends], axis=1)


def mnist_adaptive_pool(image, out: int = 10) -> np.ndarray:
    """Adaptive average pooling of a square image to (out, out) (D14).

    Args:
        image: (S, S) numeric array (e.g. 28x28 MNIST patch, any range).
        out: output side length (10 -> 10x10 = 100 units, D14).

    Returns:
        (out, out) float64 pooled means; unit row-major id = r*out + c.

    Raises:
        AssertionError: image is not 2-D square or out is not positive.
    """
    img = np.asarray(image)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise AssertionError(
            f"mnist_adaptive_pool expects a square image, got shape {img.shape}"
        )
    if out <= 0:
        raise AssertionError(f"out must be positive, got {out}")
    side = img.shape[0]
    rows = _adaptive_bounds(side, out)
    cols = _adaptive_bounds(side, out)
    pooled = np.zeros((out, out), dtype=F8)
    for r in range(out):
        r0, r1 = int(rows[r, 0]), int(rows[r, 1])
        for c in range(out):
            c0, c1 = int(cols[c, 0]), int(cols[c, 1])
            pooled[r, c] = float(np.mean(img[r0:r1, c0:c1]))
    return pooled


def mnist_encode(image, out: int = 10, *, cfg=None) -> BucketMap:
    """Full MNIST single-patch encoder: pool 28x28 -> (out,out) then encode.

    Args:
        image: (S, S) input patch (MNIST 28x28, pixel values in [0, 1] after
            normalization; any range is clipped per unit inside
            ``latency_encode``).
        out: pooled side length (10 -> 100 units, D14); the 100 units map
            row-major onto input global ids 0..N_MNIST-1 with N_MNIST=out**2.
        cfg: NetConfig or mapping for the latency parameters.

    Returns:
        BucketMap of the pooled units (ids 0..out**2-1).

    Raises:
        ValueError: if the caller's network input count (cfg.n_in) does not
            match out**2 (MNIST input-size contract, D14).
    """
    conf = cfg if isinstance(cfg, NetConfig) else cfg_from_mapping(cfg)
    if conf.n_in != out * out:
        raise ValueError(
            f"MNIST mapping requires cfg.n_in == out**2 == {out * out}, "
            f"got cfg.n_in={conf.n_in}"
        )
    pooled = mnist_adaptive_pool(image, out=out)
    return latency_encode(pooled.ravel(), cfg=conf, ids_offset=0)


def dvs_patch_aggregate(events, *, cfg=None) -> BucketMap:
    """DVS patch aggregation to N_IN=512 input units (D14) -- interface only.

    The DVS-Gesture milestone (G3) owns the actual aggregation; this function
    is declared here so the M2 interface is stable and importable.  Calling it
    before that milestone raises NotImplementedError (no pass placeholder).

    Args:
        events: DVS event structure (schema defined by the DVS milestone).
        cfg: NetConfig or mapping (must satisfy cfg.n_in == 512 at G3).

    Returns:
        BucketMap (G3 contract).

    Raises:
        NotImplementedError: always at this milestone.
    """
    raise NotImplementedError(
        "dvs_patch_aggregate: DVS patch aggregation (D14, N_IN=512) belongs "
        "to the G3/DVS milestone and is not implemented in the D1-D4 scope"
    )