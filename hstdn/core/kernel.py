"""kernel.py -- M3 L0 time-wheel LIF simulation kernel (pure NumPy).

``run_sample(bundle, input_buckets, T=200, stdp_on/homeo_on/norm_on=False)``
advances one sample of T steps over the pool state of ``bundle``.

Strict per-step ordering invariant (design doc D2/M3):
  entry  reset contract B11: V / refr / counts / first_spike / trace / ring
         are cleared; theta / rate_ema are preserved across samples.
  ①  V *= decayV
  ②  V += ring[slot] and clear that slot (slot = t % L)
  ③  input injection: np.add.at scatters each firing input over its outgoing
         synapses into future wheel slots (delays >= 1) with strength s;
         then input-channel STDP (ADR-001) when stdp_on
  ④  firing decision: (V >= theta) & (t > refr) -- strict greater (R1)
         same-time pairs follow the classical order per spiking pool,
         ascending pool-local id: pre_ltd -> set trace(=1) -> post_ltp
         (ADR-003/D12), applied when stdp_on
  ⑤  delivery of pool spikes into future slots + V = 0 + refr = t + REFR
  ⑥  trace decay: trace *= decayT (when stdp_on)
  end homeo_update (rate_ema/theta) and competitive_norm when requested.

Time wheel: L = cfg.wheel_l = delay_max + 1 (build asserts max_delay < L);
ring has shape (L, n_pool) and column indices are pool-local ids.

This stage is the L0 NumPy kernel only; L1 (Numba) branches are not started
and raise NotImplementedError where a caller asks for them -- there are no
pass placeholders.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from hstdn.core.layout import I8, F8, assert_input_gids
from hstdn.core.network import NetworkBundle
from hstdn.core.plasticity import (
    competitive_norm,
    homeo_update,
    input_channel_stdp,
    post_ltp,
    pre_ltd,
)

__all__ = ["run_sample", "reset_state", "run_sample_l1"]


def reset_state(bundle: NetworkBundle) -> None:
    """Entry reset contract (B11): clear per-sample state, keep theta/rate_ema.

    Clears V, refr, counts, first_spike, trace and the whole time-wheel ring
    to their neutral values.  theta and rate_ema are intentionally preserved
    (they carry the homeostatic memory across samples).

    Args:
        bundle: NetworkBundle (mutated in place).
    """
    bundle.V[:] = 0.0
    bundle.refr[:] = -1
    bundle.counts[:] = 0
    bundle.first_spike[:] = -1
    bundle.trace[:] = 0.0
    bundle.ring[:] = 0.0


def _deliver_source_rows(bundle: NetworkBundle, gids: np.ndarray,
                         strengths: np.ndarray, t: int,
                         ring_flat: np.ndarray) -> None:
    """Scatter the outgoing-synapse contributions of ``gids`` into the wheel.

    L0 sparse-local Python loop over the firing sources (each source is a
    contiguous CSR row); inside each row everything is vectorized NumPy.
    Contributions land at absolute time ``t + delay`` (>= t+1), slot
    ``(t + delay) % L``, column = pool-local destination; weight is scaled by
    the source strength (input events carry strength 1.0 by default).

    Args:
        bundle: NetworkBundle.
        gids: 1-D int64 source global ids.
        strengths: 1-D f8 event strengths, same length as gids.
        t: current absolute time.
        ring_flat: flat view of bundle.ring (slot-major, size L*n_pool).
    """
    cfg = bundle.cfg
    n_pool = cfg.n_pool
    L = cfg.wheel_l
    ptr, delay, dst, w = (bundle.csr_ptr, bundle.csr_delay,
                          bundle.csr_dst_local, bundle.csr_w)
    for gid, s in zip(gids, strengths):
        g = int(gid)
        s0, s1 = int(ptr[g]), int(ptr[g + 1])
        if s0 == s1:
            continue
        slot = (t + delay[s0:s1]) % L
        flat = slot * n_pool + dst[s0:s1]
        np.add.at(ring_flat, flat, w[s0:s1] * s)


def run_sample(bundle: NetworkBundle, input_buckets: Optional[Dict] = None,
               T: int = 200, stdp_on: bool = False,
               homeo_on: bool = False, norm_on: bool = False) -> dict:
    """Run one sample of T ms through the time-wheel LIF kernel.

    Args:
        bundle: NetworkBundle (state mutated in place; B11 reset at entry).
        input_buckets: Dict[int, (ids, strengths)] of input events per
            absolute step (ids = input global ids, strengths = f8 multipliers,
            default 1.0).  None/empty means no input.
        T: sample length in ms (default 200, matches cfg.window_s).
        stdp_on: enable input-channel + pool pre/post STDP during the steps.
        homeo_on: run per-neuron homeostasis at sample end.
        norm_on: run competitive normalization at sample end.

    Returns:
        dict with sample statistics:
            n_spikes_total (int), spike_counts (i8 copy of bundle.counts),
            first_spike (i8 copy), n_input_events (int).
        The bundle's theta/rate_ema may also have been updated (homeo_on).

    Raises:
        ValueError: non-positive T.
        AssertionError: invalid bucket ids / times.
    """
    cfg = bundle.cfg
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if input_buckets is None:
        input_buckets = {}
    reset_state(bundle)
    ring_flat = bundle.ring.reshape(-1)
    decay_v = cfg.decay_v
    decay_t = cfg.decay_t
    n_pool = cfg.n_pool
    L = cfg.wheel_l
    assert bundle.ring.shape == (L, n_pool), (
        f"ring shape {bundle.ring.shape} != wheel ({L}, {n_pool})"
    )
    # pre-validate buckets (cheap, one pass)
    n_input_events = 0
    for t, (ids, _str) in input_buckets.items():
        if int(t) < 0 or int(t) >= T:
            raise AssertionError(
                f"input bucket time {t} outside [0, {T})"
            )
        ids = np.asarray(ids)
        if ids.size:
            assert_input_gids(ids, n_in=cfg.n_in, name=f"input_buckets[{t}].ids")
            n_input_events += int(ids.size)

    ptr = bundle.csr_ptr
    for t in range(T):
        slot = t % L
        # ①  membrane leak
        bundle.V *= decay_v
        # ②  ring arrival at this slot, then clear the slot
        ring_slot = bundle.ring[slot]
        bundle.V += ring_slot
        ring_slot[:] = 0.0

        # ③  input injection + input-channel STDP (ADR-001)
        ev = input_buckets.get(t)
        if ev is not None:
            ids, strengths = ev
            ids = np.asarray(ids, dtype=I8)
            strengths = np.asarray(strengths, dtype=F8)
            if ids.size:
                _deliver_source_rows(bundle, ids, strengths, t, ring_flat)
                if stdp_on:
                    input_channel_stdp(bundle, ids)  # LTD + set trace[in]=1

        # ④  firing decision (strict refractory: t > refr, R1)
        spk = np.flatnonzero((bundle.V >= bundle.theta) & (t > bundle.refr))
        if spk.size and stdp_on:
            # classical same-time order per spike, ascending local id (D12):
            # pre_ltd -> set trace -> post_ltp
            for local in spk:
                gid = int(local) + cfg.n_in
                pre_ltd(bundle, gid)            # CSR-side LTD (pre spike)
                bundle.trace[gid] = 1.0         # set own trace
                post_ltp(bundle, int(local))    # CSC-side LTP (post spike)
        if spk.size:
            # ⑤  delivery + V reset + refractory
            gids = spk.astype(I8) + cfg.n_in
            ones = np.ones(spk.size, dtype=F8)
            _deliver_source_rows(bundle, gids, ones, t, ring_flat)
            bundle.V[spk] = 0.0
            bundle.refr[spk] = t + cfg.refr
            bundle.counts[spk] += 1
            never = bundle.first_spike[spk] < 0
            if never.any():
                bundle.first_spike[spk[never]] = t
        # ⑥  trace decay
        if stdp_on:
            bundle.trace *= decay_t

    # ---- sample end: homeostasis & competitive normalization ----
    if homeo_on:
        homeo_update(bundle)
    if norm_on:
        competitive_norm(bundle)
    return {
        "n_spikes_total": int(bundle.counts.sum()),
        "spike_counts": bundle.counts.copy(),
        "first_spike": bundle.first_spike.copy(),
        "n_input_events": n_input_events,
    }


def run_sample_l1(*args, **kwargs):
    """L1 Numba kernel entry -- not implemented at this stage (no pass).

    Raises:
        NotImplementedError: always, until the L1 milestone.
    """
    raise NotImplementedError(
        "run_sample_l1: the L1 Numba time-wheel kernel belongs to a later "
        "milestone; the D1-D4 scope ships the L0 NumPy kernel only"
    )