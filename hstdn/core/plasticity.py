"""plasticity.py -- M4 STDP, homeostasis and competitive normalization (L0).

Implements the D10/ADR-001/ADR-003 plasticity machinery on top of the
authoritative CSR weight array ``bundle.csr_w`` (CSC entries carry their CSR
flat position ``csc_csr_pos`` so CSC-side updates land on the same array):

- ``input_channel_stdp``: input pre-spike channel (ADR-001/D10) -- pre-side LTD
  over **all** outgoing synapses of the firing input, then ``trace[in_id]=1``.
- ``pre_ltd``: pool pre-spike LTD on the CSR side (D12) --
  ``w *= (1 - ETA_LTD * trace[dst])`` restricted to learnable edges; uses the
  global trace index of the postsynaptic neuron.
- ``post_ltp``: pool post-spike LTP on the CSC side --
  ``csr_w += ETA_LTP * trace[csc_src]`` restricted to learnable edges, then
  ``clip(w, w_lo, w_hi)`` (I-source synapses are frozen, D11).
- ``homeo_update``: per-neuron homeostasis at sample end --
  ``rate_ema = 0.9*ema + 0.1*(counts/window_s)``;
  ``theta = clip(theta + 0.05*(ema - target), theta_lo, theta_hi)``.
- ``competitive_norm``: per pool rescale of **learnable incoming** weights so
  their sum returns to the initial ``in_sum0``, then clip.
- diagnostics: ``capped_ratio`` (share of learnable synapses at the top clip
  bound) and ``input_w_mean`` (mean input->pool weight).

ID contract: LTD/LTP read the trace at its global index (trace is storage
layer, indexed by global id); weights live in CSR order; the CSC side maps to
CSR positions via ``csc_csr_pos``.  State arrays are pool-local.
"""

from __future__ import annotations

import numpy as np

from hstdn.core.layout import I8, F8, assert_input_gids

__all__ = [
    "input_channel_stdp", "pre_ltd", "post_ltp",
    "homeo_update", "competitive_norm",
    "capped_ratio", "input_w_mean", "plasticity_diagnostics",
]


def _row_slice(ptr: np.ndarray, gid: int):
    """Slice of CSR array rows for source ``gid`` (ptr is (N+1,))."""
    return slice(int(ptr[gid]), int(ptr[gid + 1]))


# ---------------------------------------------------------------------------
# input channel STDP (ADR-001 / D10)
# ---------------------------------------------------------------------------


def input_channel_stdp(bundle, in_ids) -> None:
    """Input pre-spike channel: LTD on all outgoing synapses + trace set.

    Called at the same simulation step the input fires (ADR-001).  Classical
    order: pre-side LTD first (using the postsynaptic traces accumulated so
    far -- pool spikes of the *same* step are processed later, D12), then set
    ``trace[in_id] = 1`` so that pool post-spikes later in the same step
    potentiate through this trace.

    Args:
        bundle: NetworkBundle.
        in_ids: input global ids firing this step (0..N_IN-1); duplicates are
            collapsed (an input fires at most once per step).

    Returns:
        None (bundle.csr_w and bundle.trace are updated in place).
    """
    cfg = bundle.cfg
    ids = np.unique(np.asarray(in_ids, dtype=I8).ravel())
    if ids.size == 0:
        return
    assert_input_gids(ids, n_in=cfg.n_in, name="in_ids")
    eta = cfg.eta_ltd
    ptr = bundle.csr_ptr
    for gid in ids:
        sl = _row_slice(ptr, int(gid))
        if sl.start == sl.stop:
            continue
        row = np.arange(sl.start, sl.stop, dtype=I8)
        learn = bundle.csr_learn[row]
        if not learn.any():
            continue
        idx_l = row[learn]
        dst_tr = bundle.trace[bundle.csr_dst[idx_l]]
        # multiplicative LTD, clipped back into [w_lo, w_hi] (explicit indexed
        # write-back: fancy indexing returns a copy)
        bundle.csr_w[idx_l] = np.clip(
            bundle.csr_w[idx_l] * (1.0 - eta * dst_tr), cfg.w_lo, cfg.w_hi
        )
    bundle.trace[ids] = 1.0


# ---------------------------------------------------------------------------
# pool pre-spike LTD (CSR side) / pool post-spike LTP (CSC side)
# ---------------------------------------------------------------------------


def pre_ltd(bundle, spk_gid: int) -> None:
    """Pool pre-spike LTD on the CSR side (D12).

    ``w *= (1 - ETA_LTD * trace[dst])`` over the learnable outgoing synapses
    of the spiking pool, using the **global** trace index of each
    postsynaptic pool; then clip back into ``[w_lo, w_hi]``.

    Args:
        bundle: NetworkBundle.
        spk_gid: pool neuron global id (>= N_IN) that fires now.

    Returns:
        None (bundle.csr_w updated in place).
    """
    cfg = bundle.cfg
    gid = int(spk_gid)
    if gid < cfg.n_in:
        raise AssertionError(
            f"pre_ltd requires a pool source (gid >= N_IN={cfg.n_in}), got {gid}"
        )
    sl = _row_slice(bundle.csr_ptr, gid)
    if sl.start == sl.stop:
        return
    row = np.arange(sl.start, sl.stop, dtype=I8)
    learn = bundle.csr_learn[row]
    if not learn.any():
        return
    idx_l = row[learn]
    dst_tr = bundle.trace[bundle.csr_dst[idx_l]]
    bundle.csr_w[idx_l] = np.clip(
        bundle.csr_w[idx_l] * (1.0 - cfg.eta_ltd * dst_tr),
        cfg.w_lo, cfg.w_hi,
    )


def post_ltp(bundle, spk_local: int) -> None:
    """Pool post-spike LTP on the CSC side (D12).

    For the spiking pool target ``spk_local``: over its **learnable** incoming
    synapses, ``csr_w[csr_pos] += ETA_LTP * trace[csc_src]`` then
    ``clip(w, w_lo, w_hi)``.  Inhibitory (I-source) incoming synapses are
    frozen and never touched (D11).

    Args:
        bundle: NetworkBundle.
        spk_local: pool-local id of the spiking target (0..N_POOL-1).

    Returns:
        None (bundle.csr_w updated in place through csc_csr_pos).
    """
    cfg = bundle.cfg
    local = int(spk_local)
    if not (0 <= local < cfg.n_pool):
        raise AssertionError(
            f"post_ltp requires a pool-local id in [0, {cfg.n_pool}), got {local}"
        )
    sl = _row_slice(bundle.csc_ptr, local)
    if sl.start == sl.stop:
        return
    pos = bundle.csc_csr_pos[sl]           # CSR positions of this row
    learn = bundle.csr_learn[pos]
    if not learn.any():
        return
    srcs = bundle.csc_src[sl][learn]       # global source ids (trace index)
    idx = pos[learn]
    bundle.csr_w[idx] = np.clip(
        bundle.csr_w[idx] + cfg.eta_ltp * bundle.trace[srcs],
        cfg.w_lo, cfg.w_hi,
    )


# ---------------------------------------------------------------------------
# homeostasis & competitive normalization (sample end)
# ---------------------------------------------------------------------------


def homeo_update(bundle) -> None:
    """Per-neuron homeostasis after a sample (rate_ema & theta adaptation).

    ``rate = counts / window_s`` (counts = spikes collected during the sample;
    window_s in seconds, default 0.2 for T=200 ms);
    ``rate_ema = (1 - alpha) * rate_ema + alpha * rate`` (alpha=0.1);
    ``theta = clip(theta + homeo_theta_lr * (rate_ema - target),
    theta_lo, theta_hi)`` (target 8 Hz).

    Args:
        bundle: NetworkBundle (uses bundle.counts; updates rate_ema & theta).

    Returns:
        None (in place).
    """
    cfg = bundle.cfg
    rate = bundle.counts.astype(F8) / cfg.window_s
    bundle.rate_ema *= 1.0 - cfg.homeo_ema_alpha
    bundle.rate_ema += cfg.homeo_ema_alpha * rate
    bundle.theta = np.clip(
        bundle.theta + cfg.homeo_theta_lr * (bundle.rate_ema - cfg.homeo_rate_target),
        cfg.theta_lo, cfg.theta_hi,
    )


def competitive_norm(bundle) -> None:
    """Per-pool competitive normalization over learnable incoming synapses.

    For each pool p: let s = current sum of its learnable incoming weights;
    every such weight is rescaled by ``in_sum0[p] / s`` (restoring the initial
    per-pool drive, D10/competitive normalization), then clipped into
    ``[w_lo, w_hi]``.  Fully vectorized over the compact learn rows.

    Args:
        bundle: NetworkBundle.

    Returns:
        None (bundle.csr_w updated in place).
    """
    cfg = bundle.cfg
    ptr = bundle.in_learn_ptr
    idx = bundle.in_learn_idx
    w = bundle.csr_w
    if idx.size == 0:
        return
    pool_ids = np.repeat(np.arange(cfg.n_pool, dtype=I8), np.diff(ptr))
    s = np.bincount(pool_ids, weights=w[idx], minlength=cfg.n_pool)
    scale = np.divide(
        bundle.in_sum0, s, out=np.ones(cfg.n_pool, dtype=F8),
        where=s > 0,
    )
    w[idx] = np.clip(w[idx] * scale[pool_ids], cfg.w_lo, cfg.w_hi)


# ---------------------------------------------------------------------------
# diagnostics (顶界突触占比 / 输入→池 w 均值; G0 #5/#8/#9 related)
# ---------------------------------------------------------------------------


def capped_ratio(bundle) -> float:
    """Share of learnable synapses sitting at the top clip bound ``w_hi``.

    Args:
        bundle: NetworkBundle.

    Returns:
        float in [0, 1] (0.0 if no learnable synapse exists).
    """
    cfg = bundle.cfg
    learn = bundle.csr_learn
    if not learn.any():
        return 0.0
    top = bundle.csr_w[learn] >= cfg.w_hi - 1e-12
    return float(np.count_nonzero(top)) / float(learn.sum())


def input_w_mean(bundle) -> float:
    """Mean weight of the input->pool synapses (input sources only).

    Args:
        bundle: NetworkBundle.

    Returns:
        float mean (0.0 when there is no input edge).
    """
    cfg = bundle.cfg
    mask = bundle.csr_src < cfg.n_in
    if not mask.any():
        return 0.0
    return float(bundle.csr_w[mask].mean())


def plasticity_diagnostics(bundle) -> dict:
    """Package plasticity diagnostics into a dict.

    Keys: capped_ratio (learnable synapses at w_hi), input_w_mean (input->
    pool mean weight), w_min/w_max over learnable synapses, n_learnable.
    """
    cfg = bundle.cfg
    learn_w = bundle.csr_w[bundle.csr_learn]
    return {
        "capped_ratio": capped_ratio(bundle),
        "input_w_mean": input_w_mean(bundle),
        "learn_w_min": float(learn_w.min()) if learn_w.size else 0.0,
        "learn_w_max": float(learn_w.max()) if learn_w.size else 0.0,
        "w_hi": float(cfg.w_hi),
        "w_lo": float(cfg.w_lo),
        "n_learnable": int(bundle.csr_learn.sum()),
    }