"""features.py -- M5 feature extraction from pool spike counts (L0 NumPy).

Feeding the L0 readout (readout.py) inside the G1 protocol (scheduler.py).

Modes (design doc M5 / features section; D9):
- ``v0`` (default, ``features.mode=v0``): per sample
  ``X = sqrt(counts[:, feature_mask])`` followed by row-wise L2
  normalization.  ``feature_mask`` defaults to the whole pool population
  (``features.feature_mask=pool``, D9).  Output (n_samples, |mask|).
- ``v1``: time-binned features, implemented but OFF by default (enabled after
  G2).  Input spike counts per bin: (n_samples, n_bins, N_POOL) or a single
  sample (n_bins, N_POOL); per bin ``sqrt`` applied, then flattened to
  (n_samples, n_bins*N_POOL) (row L2 is NOT applied -- kept as documented;
  callers may normalize).  ``n_bins=5`` (features.n_bins).
- ``v2``: surrogate-gradient variant -- interface notes only, raises
  NotImplementedError until the L1/V2 milestone (G3+).

Pure functions, deterministic.  Zero rows (no spikes at all) are kept as zero
vectors after L2 normalization (never NaN).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Union

import numpy as np

from hstdn.core.layout import I8, F8, assert_pool_local_ids

__all__ = ["FeaturesConfig", "features_from_cfg", "build_features",
           "l2_row_normalize"]

MaskLike = Union[str, np.ndarray, list]


@dataclass(frozen=True)
class FeaturesConfig:
    """Feature-extraction hyper-parameters (design doc features section).

    Fields:
        mode: 'v0' (default) | 'v1' | 'v2' (v2 raises NotImplementedError).
        feature_mask: 'pool'/'all' (default) or a 1-D array of pool-local ids
            / boolean mask of length N_POOL.
        n_bins: number of time bins for v1 (default 5).
    """

    mode: str = "v0"
    feature_mask: Any = "pool"   # 'pool' | 'all' | 1-D ids/bool (N_POOL,)
    n_bins: int = 5


def features_from_cfg(cfg: Union[FeaturesConfig, Mapping[str, Any], None]
                      ) -> FeaturesConfig:
    """Normalize a features config (None | FeaturesConfig | mapping).

    Args:
        cfg: features section; unknown keys raise ValueError (typo guard).

    Returns:
        A frozen FeaturesConfig.
    """
    if cfg is None or isinstance(cfg, FeaturesConfig):
        return cfg if cfg is not None else FeaturesConfig()
    known = set(FeaturesConfig.__dataclass_fields__)
    extra = set(cfg) - known
    if extra:
        raise ValueError(f"unknown features config keys: {sorted(extra)}")
    return replace(FeaturesConfig(), **{k: v for k, v in cfg.items() if k in known})


def _resolve_mask(mask: Any, n_pool: int) -> np.ndarray:
    """Resolve feature_mask to a boolean column selector of length n_pool.

    'pool'/'all'/None -> all True.  Arrays are interpreted as pool-local ids
    (int, asserted in [0, n_pool)) or as an explicit bool mask of length
    n_pool.
    """
    if mask is None or (isinstance(mask, str) and mask in ("pool", "all")):
        return np.ones(n_pool, dtype=bool)
    arr = np.asarray(mask)
    if arr.dtype == bool:
        if arr.shape != (n_pool,):
            raise AssertionError(
                f"boolean feature_mask must have shape ({n_pool},), got {arr.shape}"
            )
        return arr
    ids = arr.astype(I8)
    assert_pool_local_ids(ids, n_pool=n_pool, name="feature_mask")
    sel = np.zeros(n_pool, dtype=bool)
    sel[ids] = True
    return sel


def l2_row_normalize(x: np.ndarray, *, eps: float = 0.0) -> np.ndarray:
    """Row-wise L2 normalization; zero rows stay zero (never NaN).

    Args:
        x: (n, d) float array.
        eps: additive floor under the norm (default 0).

    Returns:
        (n, d) float64; each row divided by its L2 norm (norm > eps) else kept.
    """
    x = np.asarray(x, dtype=F8)
    if x.ndim != 2:
        raise AssertionError(f"l2_row_normalize expects 2-D input, got {x.shape}")
    norms = np.sqrt(np.einsum("ij,ij->i", x, x))
    out = x.copy()
    nz = norms > eps
    if nz.any():
        out[nz] = x[nz] / norms[nz, None]
    return out


def build_features(counts, cfg: Union[FeaturesConfig, Mapping[str, Any], None] = None
                   ) -> np.ndarray:
    """Build the per-sample feature matrix from pool spike counts.

    Args:
        counts:
            - v0: (n_samples, N_POOL) or (N_POOL,) counts (int/float >= 0).
            - v1: (n_samples, n_bins, N_POOL) or single (n_bins, N_POOL)
              spike counts per time bin (last axis must be N_POOL).
        cfg: None | FeaturesConfig | mapping (features section).

    Returns:
        float64 feature matrix:
            - v0: (n_samples, feat_dim) with feat_dim = |feature_mask|,
              sqrt + row L2 normalization;
            - v1: (n_samples, n_bins*N_POOL), sqrt per bin, flattened.

    Raises:
        NotImplementedError: mode 'v2' (interface only until L1/V2).
        AssertionError / ValueError: bad shapes, ids, or bin counts.
    """
    conf = features_from_cfg(cfg)
    if conf.mode == "v2":
        raise NotImplementedError(
            "features v2 (surrogate-gradient) is an interface note only; it "
            "belongs to the L1/V2 milestone (G3+)"
        )
    if conf.mode == "v1":
        return _build_v1(counts, conf)
    if conf.mode != "v0":
        raise ValueError(f"unknown features.mode {conf.mode!r} (v0|v1|v2)")
    return _build_v0(counts, conf)


def _counts2d(counts: Any, n_pool: int, name: str = "counts") -> np.ndarray:
    """Normalize 1-D/2-D counts to (n_samples, N_POOL) and validate domain."""
    c = np.asarray(counts)
    if c.ndim == 1:
        if c.shape[0] != n_pool:
            raise AssertionError(
                f"{name}: 1-D input must have length N_POOL={n_pool}, "
                f"got {c.shape[0]}"
            )
        c = c.reshape(1, -1)
    elif c.ndim == 2:
        if c.shape[1] != n_pool:
            raise AssertionError(
                f"{name}: 2-D input must have shape (n, N_POOL={n_pool}), "
                f"got {c.shape}"
            )
    else:
        raise AssertionError(
            f"{name}: expected 1-D/2-D (n, N_POOL), got ndim={c.ndim}"
        )
    if np.any(c < 0):
        raise AssertionError(f"{name} contains negative spike counts")
    return np.asarray(c, dtype=F8)


def _build_v0(counts: Any, conf: FeaturesConfig) -> np.ndarray:
    # N_POOL is the last axis of the input counts (1-D/2-D accepted)
    arr = np.asarray(counts)
    if arr.ndim not in (1, 2):
        raise AssertionError(
            f"v0 counts must be (n, N_POOL) or (N_POOL,), got ndim={arr.ndim}"
        )
    n_pool = int(arr.shape[-1])
    c = _counts2d(arr, n_pool)
    sel = _resolve_mask(conf.feature_mask, n_pool)
    x = np.sqrt(np.clip(c, 0.0, None))[:, sel]
    return l2_row_normalize(x)


def _build_v1(counts: Any, conf: FeaturesConfig) -> np.ndarray:
    c = np.asarray(counts)
    if c.ndim == 2:
        n_bins, n_pool = c.shape
        c = c.reshape(1, n_bins, n_pool)
    elif c.ndim == 3:
        n_bins = c.shape[1]
        n_pool = c.shape[2]
    else:
        raise AssertionError(
            f"v1 counts must be (n, n_bins, N_POOL) or (n_bins, N_POOL), "
            f"got ndim={c.ndim}"
        )
    if n_bins != conf.n_bins:
        raise ValueError(
            f"v1 input bin count {n_bins} != features.n_bins={conf.n_bins}"
        )
    if np.any(c < 0):
        raise AssertionError("v1 counts contain negative values")
    x = np.sqrt(np.clip(np.asarray(c, dtype=F8), 0.0, None))
    return x.reshape(x.shape[0], -1)