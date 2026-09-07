"""network.py -- M1 network construction (slab wiring, CSR/CSC, gain report).

``build_network(cfg) -> NetworkBundle`` builds the full network following the
D11/D13 layout decisions:

- input neurons on the z=0 plane (square grid + jitter);
- pool neurons uniformly sampled in the slab [0,1]x[0,1]x[0,0.3] (D13);
- 80/20 E/I tags on pools (D11);
- input->pool: per pool the k=16 nearest inputs within R_IN=0.4 (3-D);
- pool->pool: per source the k=12 nearest pools within R_POOL (self excluded);
  E-source weights ~U(0.18,0.42)*theta0 (learnable); I-source weights
  -1.2*exp(-d/0.12) (non-learnable, |w| >= E and frozen);
- delay = clip(round(dist/VEL), 1, 15); time wheel L = delay_max + 1 = 16.

Memory layout (design doc Sec.2.2/2.3): SoA states + CSR (per-source rows,
sorted stably by source then destination global id) + CSC (per-target rows,
built with the B3 fix: np.add.at(csc_ptr[1:], csr_dst-N_IN, 1) then cumsum,
asserting csr_dst >= N_IN) + per-pool learnable-incoming compact rows
(in_learn_ptr / in_learn_idx / in_sum0) + the time-wheel ring.

Edge storage uses global IDs (CSR dst, CSC src, trace index); state arrays
are pool-local; the conversion dst_local = dst - N_IN happens only at the
delivery site.  build returns a NetworkBundle and a structural self-check
report (isolated neurons = 0, zero indegree = 0, E-source outdegree <= 12,
delay histogram, gain calibration numbers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Union

import numpy as np

from hstdn.core.layout import (
    I8,
    F8,
    NetConfig,
    assert_pool_gids,
    cfg_from_mapping,
)
from hstdn.core.spatial_hash import SpatialHash3D

__all__ = ["NetworkBundle", "build_network", "structural_report",
           "print_report"]

_CfgLike = Union[NetConfig, Mapping[str, Any], None]


def _coerce_cfg(cfg: _CfgLike) -> NetConfig:
    """Normalize ``cfg`` (None | NetConfig | mapping) to a derived NetConfig."""
    if cfg is None:
        return cfg_from_mapping(None)
    if isinstance(cfg, NetConfig):
        if cfg.wheel_l <= 0:  # not derived yet
            return cfg.with_derived()
        return cfg
    return cfg_from_mapping(cfg)


@dataclass
class NetworkBundle:
    """SoA network state bundle (design doc Sec.2.2/2.3 layout).

    Storage-layer fields use global IDs; state fields are pool-local.

    Structural fields:
        cfg: frozen NetConfig used at build time.
        input_xyz: (N_IN,3) f8 input positions (z=0 plane).
        pool_xyz: (N_POOL,3) f8 pool positions (slab).
        pool_is_E: (N_POOL,) bool E/I tags (80/20, D11).
        n_edges_in / n_edges_pool: int edge counts by source class.

    CSR (per-source rows; sources = all global ids ascending, incl. inputs):
        csr_ptr: (N_TOTAL+1,) i8 row boundaries.
        csr_src_g: (N_TOTAL,) i8 row source global id (arange(N_TOTAL)).
        csr_dst: (E,) i8 destination global ids (all >= N_IN).
        csr_dst_local: (E,) i8 destination pool-local ids (delivery cache).
        csr_w: (E,) f8 signed weights (input/E > 0, I < 0).
        csr_delay: (E,) i8 per-edge delay in [delay_min, delay_max].
        csr_learn: (E,) bool learnable mask (input & E-source edges).
        csr_src: (E,) i8 per-edge source global id (CSR order).

    CSC (per-target rows; targets are pool-local ids 0..N_POOL-1):
        csc_ptr: (N_POOL+1,) i8 row boundaries.
        csc_src: (E,) i8 per-edge source global id (CSC order).
        csc_csr_pos: (E,) i8 CSR flat position of each CSC entry (lets
            CSC-side plasticity update the authoritative csr_w).

    Learnable incoming rows (per pool, compact, learn=True only):
        in_learn_ptr: (N_POOL+1,) i8 boundaries over in_learn_idx.
        in_learn_idx: (K,) i8 CSR flat positions of learnable incoming edges,
            target-major order.
        in_sum0: (N_POOL,) f8 initial sum of learnable incoming weights
            (per-pool competitive-normalization target).

    State (pool-local unless noted; entry reset contract B11):
        V: (N_POOL,) f8 membrane potential (reset per sample).
        theta: (N_POOL,) f8 threshold (kept across samples).
        refr: (N_POOL,) i8 refractory deadline (reset per sample).
        counts: (N_POOL,) i8 spike counter of the current sample.
        first_spike: (N_POOL,) i8 first spike time or -1 (current sample).
        rate_ema: (N_POOL,) f8 homeostatic EMA rate (kept across samples).
        trace: (N_TOTAL,) f8 STDP trace; index = global id (storage layer).
        ring: (WHEEL_L, N_POOL) f8 time wheel (slot-major; pool-local cols).
    """

    cfg: NetConfig

    # geometry / types
    input_xyz: np.ndarray          # (N_IN, 3) f8
    pool_xyz: np.ndarray           # (N_POOL, 3) f8
    pool_is_E: np.ndarray          # (N_POOL,) bool
    n_edges_in: int
    n_edges_pool: int

    # CSR (authoritative edge storage; per-source rows, stable sorted)
    csr_ptr: np.ndarray            # (N_TOTAL+1,) i8
    csr_src_g: np.ndarray          # (N_TOTAL,) i8
    csr_dst: np.ndarray            # (E,) i8 (global, >= N_IN)
    csr_dst_local: np.ndarray      # (E,) i8 (pool local)
    csr_w: np.ndarray              # (E,) f8
    csr_delay: np.ndarray          # (E,) i8
    csr_learn: np.ndarray          # (E,) bool
    csr_src: np.ndarray            # (E,) i8 (per-edge source global id)

    # CSC (per-target rows; pool-local targets)
    csc_ptr: np.ndarray            # (N_POOL+1,) i8
    csc_src: np.ndarray            # (E,) i8
    csc_csr_pos: np.ndarray        # (E,) i8

    # learnable incoming rows per pool
    in_learn_ptr: np.ndarray       # (N_POOL+1,) i8
    in_learn_idx: np.ndarray       # (K,) i8 (CSR flat positions)
    in_sum0: np.ndarray            # (N_POOL,) f8

    # state
    V: np.ndarray = field(repr=False)             # (N_POOL,) f8
    theta: np.ndarray = field(repr=False)         # (N_POOL,) f8
    refr: np.ndarray = field(repr=False)          # (N_POOL,) i8
    counts: np.ndarray = field(repr=False)        # (N_POOL,) i8
    first_spike: np.ndarray = field(repr=False)   # (N_POOL,) i8
    rate_ema: np.ndarray = field(repr=False)      # (N_POOL,) f8
    trace: np.ndarray = field(repr=False)         # (N_TOTAL,) f8
    ring: np.ndarray = field(repr=False)          # (WHEEL_L, N_POOL) f8

    @property
    def n_in(self) -> int:
        """Input neuron count."""
        return self.cfg.n_in

    @property
    def n_pool(self) -> int:
        """Pool neuron count."""
        return self.cfg.n_pool

    @property
    def n_edges(self) -> int:
        """Total synapse count."""
        return int(self.csr_w.size)


# ---------------------------------------------------------------------------
# position sampling & connectivity
# ---------------------------------------------------------------------------


def _sample_input_positions(cfg: NetConfig, rng: np.random.Generator) -> np.ndarray:
    """Input positions on the z=0 plane: square grid + jitter (D13).

    Grid cell side = 1 / n_input_cols; each neuron sits at the cell centre
    plus a uniform jitter of +/- (input_jitter * cell), clipped into [0,1]^2.

    Raises:
        ValueError: if n_in != n_input_cols**2 (grid layout contract).
    """
    side = cfg.n_input_cols
    if side * side != cfg.n_in:
        raise ValueError(
            "grid input layout requires n_in == n_input_cols**2, got "
            f"n_in={cfg.n_in}, n_input_cols={side}"
        )
    cell = 1.0 / side
    centers = (np.arange(side, dtype=F8) + 0.5) * cell
    xs, ys = np.meshgrid(centers, centers, indexing="ij")
    xy = np.stack([xs.ravel(), ys.ravel()], axis=1)
    jit = cfg.input_jitter * cell
    xy = xy + rng.uniform(-jit, jit, size=xy.shape)
    xy = np.clip(xy, 0.0, 1.0)
    pos = np.zeros((cfg.n_in, 3), dtype=F8)
    pos[:, :2] = xy
    return pos


def _sample_pool_positions(cfg: NetConfig, rng: np.random.Generator) -> np.ndarray:
    """Pool positions: uniform in slab [0,1]x[0,1]x[0,pool_z_hi] (D13)."""
    return np.asarray(
        rng.uniform(
            low=[0.0, 0.0, 0.0],
            high=[1.0, 1.0, cfg.pool_z_hi],
            size=(cfg.n_pool, 3),
        ),
        dtype=F8,
    )


def _tag_ei(cfg: NetConfig, rng: np.random.Generator) -> np.ndarray:
    """80/20 E/I pool tags: exactly cfg.n_e_pool E pools (seeded random)."""
    is_E = np.zeros(cfg.n_pool, dtype=bool)
    is_E[rng.choice(cfg.n_pool, size=cfg.n_e_pool, replace=False)] = True
    return is_E


def _pool_to_pool_edges(cfg: NetConfig, pool_xyz: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-source k=12 nearest pool neighbours within R_POOL (D11, self out).

    Args:
        cfg: network config.
        pool_xyz: (N_POOL,3) pool positions.

    Returns:
        (src_local, dst_local, dist): i8/i8/f8 arrays of pool-local endpoints
        and 3-D distances (arbitrary order; sorted at the CSR stage).
    """
    hasher = SpatialHash3D(pool_xyz, cfg.r_pool)
    srcs: list[int] = []
    dsts: list[int] = []
    dists: list[float] = []
    for i in range(cfg.n_pool):
        idx, d = hasher.nearest_k(pool_xyz[i], cfg.k_pool, cfg.r_pool,
                                  exclude_self=True, self_id=i)
        if idx.size:
            srcs.extend([i] * int(idx.size))
            dsts.extend(int(j) for j in idx)
            dists.extend(float(x) for x in d)
    return (np.asarray(srcs, dtype=I8), np.asarray(dsts, dtype=I8),
            np.asarray(dists, dtype=F8))


def _input_to_pool_edges(cfg: NetConfig, input_xyz: np.ndarray,
                         pool_xyz: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per pool: k=16 nearest inputs within R_IN=0.4 (3-D distance).

    Args:
        cfg: network config.
        input_xyz: (N_IN,3) input positions.
        pool_xyz: (N_POOL,3) pool positions.

    Returns:
        (src_in, dst_local, dist): i8/i8/f8 arrays; src_in are input global
        ids (0..N_IN-1), dst_local are pool-local ids.
    """
    hasher = SpatialHash3D(input_xyz, cfg.r_pool)
    srcs: list[int] = []
    dsts: list[int] = []
    dists: list[float] = []
    for i in range(cfg.n_pool):
        idx, d = hasher.nearest_k(pool_xyz[i], cfg.k_in, cfg.r_in,
                                  exclude_self=False)
        if idx.size:
            srcs.extend(int(j) for j in idx)
            dsts.extend([i] * int(idx.size))
            dists.extend(float(x) for x in d)
    return (np.asarray(srcs, dtype=I8), np.asarray(dsts, dtype=I8),
            np.asarray(dists, dtype=F8))
# ---------------------------------------------------------------------------
# CSR/CSC construction
# ---------------------------------------------------------------------------


def _build_csr(cfg: NetConfig, src_g: np.ndarray, dst_local: np.ndarray,
               dist: np.ndarray, w_class_in: np.ndarray, w_class_e: np.ndarray,
               rng: np.random.Generator
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray, np.ndarray]:
    """Stable-sort edges into CSR rows; sample weights & delays per class.

    Sorts by (source global id asc, destination global id asc).  Every edge
    source lies in [0, N_TOTAL); every destination is a pool (>= N_IN).

    Weight classes (per edge):
        - input source: U(w_in_lo, w_in_hi) * theta0          (learnable)
        - E-pool source: U(w_ee_lo, w_ee_hi) * theta0         (learnable)
        - I-pool source: -w_i_amp * exp(-dist / w_i_tau)      (frozen)
    delay = clip(round(dist / VEL), delay_min, delay_max).

    Returns:
        (csr_ptr, csr_src, csr_dst_local, csr_w, csr_delay, csr_learn, order);
        ``order`` maps the input edge rows to their CSR flat positions.
    """
    dst_g = dst_local + cfg.n_in
    if src_g.size and int(src_g.min()) < 0:
        raise AssertionError(f"negative edge source: min={int(src_g.min())}")
    if src_g.size and int(src_g.max()) >= cfg.n_total:
        raise AssertionError(
            f"edge source out of global domain: max={int(src_g.max())} "
            f">= N_TOTAL={cfg.n_total}"
        )
    if dst_g.size and int(dst_g.min()) < cfg.n_in:
        raise AssertionError(
            "B3/CSC guard: non-pool destination found "
            f"(min csr_dst = {int(dst_g.min())} < N_IN = {cfg.n_in}); every "
            "synapse must target a pool neuron"
        )

    n = int(src_g.size)
    order = np.lexsort((dst_g, src_g))  # primary: source, secondary: dst
    csr_src = src_g[order]

    # --- weights (drawn after the stable sort keeps rng order deterministic;
    #     class arrays are precomputed on the unsorted edge rows) ---
    w = np.empty(n, dtype=F8)
    if n:
        w[w_class_in] = rng.uniform(cfg.w_in_lo, cfg.w_in_hi,
                                    size=int(w_class_in.sum())) * cfg.theta0
        w[w_class_e] = rng.uniform(cfg.w_ee_lo, cfg.w_ee_hi,
                                   size=int(w_class_e.sum())) * cfg.theta0
        w[w_class_in | w_class_e] = np.clip(
            w[w_class_in | w_class_e], cfg.w_lo, cfg.w_hi
        )
        i_mask = ~(w_class_in | w_class_e)
        w[i_mask] = -cfg.w_i_amp * np.exp(-dist[i_mask] / cfg.w_i_tau)

    delay = np.clip(
        np.rint(dist / cfg.vel).astype(I8), cfg.delay_min, cfg.delay_max
    )
    learn = w_class_in | w_class_e  # I-source synapses never learn (D11)

    csr_dst_local = dst_local[order]
    counts = np.bincount(csr_src, minlength=cfg.n_total).astype(I8)
    csr_ptr = np.zeros(cfg.n_total + 1, dtype=I8)
    csr_ptr[1:] = np.cumsum(counts)
    return (csr_ptr, csr_src, csr_dst_local, w[order], delay[order],
            learn[order], order)


def _build_csc(cfg: NetConfig, csr_dst_local: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-target CSC rows with the B3 fix.

    ``csc_ptr`` boundaries are computed via ``np.add.at(csc_ptr[1:],
    csr_dst_local, 1)`` followed by cumsum (B3 correction), and
    ``order_csc = np.argsort(csr_dst_local, kind='stable')`` gives the
    target-major permutation (inner order = source-major CSR order).

    Returns:
        (csc_ptr, order_csc, dst_csc); order_csc[j] is the CSR flat position
        of the j-th CSC entry; dst_csc[j] its pool-local target.
    """
    csc_ptr = np.zeros(cfg.n_pool + 1, dtype=I8)
    np.add.at(csc_ptr[1:], csr_dst_local, 1)
    np.cumsum(csc_ptr, out=csc_ptr)
    order_csc = np.argsort(csr_dst_local, kind="stable").astype(I8)
    dst_csc = csr_dst_local[order_csc]
    return csc_ptr, order_csc, dst_csc


def _build_learn_rows(cfg: NetConfig, csr_learn: np.ndarray,
                      order_csc: np.ndarray, dst_csc: np.ndarray,
                      csr_w: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compact per-pool learnable-incoming rows (learn=True synapses only).

    Args:
        cfg: network config.
        csr_learn: (E,) learnable mask in CSR order.
        order_csc: stable CSR -> CSC permutation (see _build_csc).
        dst_csc: (E,) pool-local targets in CSC order.
        csr_w: (E,) weights in CSR order.

    Returns:
        (in_learn_ptr, in_learn_idx, in_sum0):
            - in_learn_ptr (N_POOL+1,) i8 boundaries;
            - in_learn_idx (K,) i8 CSR flat positions of the learnable edges,
              target-major order;
            - in_sum0 (N_POOL,) f8 initial per-pool sum of learnable incoming
              weights (normalization target).
    """
    learn_csc = csr_learn[order_csc]          # learnable mask in CSC order
    dst_learn = dst_csc[learn_csc]            # targets of learnable edges
    in_learn_idx = order_csc[learn_csc]       # CSR positions, target-major
    counts = np.bincount(dst_learn, minlength=cfg.n_pool).astype(I8)
    in_learn_ptr = np.zeros(cfg.n_pool + 1, dtype=I8)
    in_learn_ptr[1:] = np.cumsum(counts)
    in_sum0 = np.bincount(
        dst_learn, weights=csr_w[in_learn_idx], minlength=cfg.n_pool
    ).astype(F8)
    return in_learn_ptr, in_learn_idx, in_sum0


# ---------------------------------------------------------------------------
# public construction
# ---------------------------------------------------------------------------


def build_network(cfg: _CfgLike = None,
                  rng: Optional[np.random.Generator] = None) -> NetworkBundle:
    """Build the full network (M1) and run the structural self-checks.

    Args:
        cfg: None (defaults) | NetConfig | mapping of NetConfig fields.
        rng: optional seeded Generator; defaults to default_rng(cfg.seed).

    Returns:
        NetworkBundle with CSR/CSC/in_learn rows, SoA states and the time
        wheel ring; all structural invariants asserted before returning.

    Raises:
        AssertionError: on any violated structural invariant (readable message
            with the statistics).  ValueError on invalid config.
    """
    cfg = _coerce_cfg(cfg)
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    # --- geometry & types ---
    input_xyz = _sample_input_positions(cfg, rng)
    pool_xyz = _sample_pool_positions(cfg, rng)
    pool_is_E = _tag_ei(cfg, rng)

    # --- connectivity ---
    p_src, p_dst, p_dist = _pool_to_pool_edges(cfg, pool_xyz)
    i_src, i_dst, i_dist = _input_to_pool_edges(cfg, input_xyz, pool_xyz)
    n_in_edges = int(i_src.size)
    n_pp_edges = int(p_src.size)

    # combined edge rows (unsorted)
    src_g = np.concatenate([i_src, p_src + cfg.n_in]).astype(I8)
    dst_local = np.concatenate([i_dst, p_dst]).astype(I8)
    dist = np.concatenate([i_dist, p_dist]).astype(F8)
    # weight classes on the unsorted rows
    cls_in = src_g < cfg.n_in
    cls_e = (~cls_in) & pool_is_E[(src_g - cfg.n_in).astype(np.intp)]
    cls_i = (~cls_in) & (~cls_e)
    if not (cls_in.sum() + cls_e.sum() + cls_i.sum() == src_g.size):
        raise AssertionError("edge classification is not exhaustive")

    csr_ptr, csr_src, csr_dst_local, csr_w, csr_delay, csr_learn, _ = \
        _build_csr(cfg, src_g, dst_local, dist, cls_in, cls_e, rng)
    del cls_i, cls_in, cls_e, src_g, dst_local, dist  # reduce live memory

    csr_dst = csr_dst_local + cfg.n_in
    csr_src_g = np.arange(cfg.n_total, dtype=I8)
    if csr_dst.size and int(csr_dst.min()) < cfg.n_in:
        raise AssertionError(
            "csr_dst violates the pool-target contract: min = "
            f"{int(csr_dst.min())} < N_IN = {cfg.n_in}"
        )
    assert_pool_gids(csr_dst, n_in=cfg.n_in, n_pool=cfg.n_pool, name="csr_dst")

    csc_ptr, order_csc, dst_csc = _build_csc(cfg, csr_dst_local)
    csc_src = csr_src[order_csc]
    csc_csr_pos = order_csc

    in_learn_ptr, in_learn_idx, in_sum0 = _build_learn_rows(
        cfg, csr_learn, order_csc, dst_csc, csr_w
    )

    # --- SoA states ---
    theta = np.full(cfg.n_pool, cfg.theta0, dtype=F8)
    bundle = NetworkBundle(
        cfg=cfg,
        input_xyz=input_xyz,
        pool_xyz=pool_xyz,
        pool_is_E=pool_is_E,
        n_edges_in=n_in_edges,
        n_edges_pool=n_pp_edges,
        csr_ptr=csr_ptr,
        csr_src_g=csr_src_g,
        csr_dst=csr_dst,
        csr_dst_local=csr_dst_local,
        csr_w=csr_w,
        csr_delay=csr_delay,
        csr_learn=csr_learn,
        csr_src=csr_src,
        csc_ptr=csc_ptr,
        csc_src=csc_src,
        csc_csr_pos=csc_csr_pos,
        in_learn_ptr=in_learn_ptr,
        in_learn_idx=in_learn_idx,
        in_sum0=in_sum0,
        V=np.zeros(cfg.n_pool, dtype=F8),
        theta=theta,
        refr=np.full(cfg.n_pool, -1, dtype=I8),
        counts=np.zeros(cfg.n_pool, dtype=I8),
        first_spike=np.full(cfg.n_pool, -1, dtype=I8),
        rate_ema=np.zeros(cfg.n_pool, dtype=F8),
        trace=np.zeros(cfg.n_total, dtype=F8),
        ring=np.zeros((cfg.wheel_l, cfg.n_pool), dtype=F8),
    )
    _assert_structure(bundle)
    return bundle


def _assert_structure(bundle: NetworkBundle) -> None:
    """Hard structural invariants (G0 #10/#12/#13 family) with statistics."""
    cfg = bundle.cfg
    n_pool = cfg.n_pool

    # CSR pointer sanity
    if not (bundle.csr_ptr.shape == (cfg.n_total + 1,)
            and bundle.csr_ptr[-1] == bundle.csr_w.size):
        raise AssertionError(
            f"csr_ptr inconsistent: shape={bundle.csr_ptr.shape}, "
            f"ptr[-1]={bundle.csr_ptr[-1]}, n_edges={bundle.csr_w.size}"
        )
    if not (bundle.csc_ptr.shape == (n_pool + 1,)
            and bundle.csc_ptr[-1] == bundle.csr_w.size):
        raise AssertionError(
            f"csc_ptr inconsistent: shape={bundle.csc_ptr.shape}, "
            f"ptr[-1]={bundle.csc_ptr[-1]}, n_edges={bundle.csr_w.size}"
        )
    if not (bundle.in_learn_ptr.shape == (n_pool + 1,)
            and bundle.in_learn_ptr[-1] == bundle.in_learn_idx.size):
        raise AssertionError(
            "in_learn_ptr inconsistent: "
            f"shape={bundle.in_learn_ptr.shape}, "
            f"ptr[-1]={bundle.in_learn_ptr[-1]}, idx={bundle.in_learn_idx.size}"
        )

    indeg = np.bincount(bundle.csr_dst_local, minlength=n_pool)
    outdeg = np.diff(bundle.csr_ptr)
    pool_outdeg = outdeg[cfg.n_in:cfg.n_total]

    zero_in = int(np.count_nonzero(indeg == 0))
    if zero_in:
        raise AssertionError(
            f"structural check failed: {zero_in}/{n_pool} pools have zero "
            "in-degree (wiring must leave no orphan target)"
        )
    # isolated = zero indegree AND zero outdegree
    isolated = int(np.count_nonzero((indeg == 0) & (pool_outdeg == 0)))
    if isolated:
        raise AssertionError(
            f"structural check failed: {isolated}/{n_pool} isolated pools"
        )

    e_rows = bundle.pool_is_E  # per pool local
    e_out = pool_outdeg[e_rows]
    if e_out.size and int(e_out.max()) > cfg.k_pool:
        raise AssertionError(
            f"E-source outdegree exceeds k_pool={cfg.k_pool}: "
            f"max={int(e_out.max())}"
        )
    if bundle.csr_delay.size:
        dmin, dmax = int(bundle.csr_delay.min()), int(bundle.csr_delay.max())
        if dmin < cfg.delay_min or dmax > cfg.delay_max:
            raise AssertionError(
                f"delay outside [{cfg.delay_min}, {cfg.delay_max}]: "
                f"actual [{dmin}, {dmax}]"
            )
        if dmax >= cfg.wheel_l:
            raise AssertionError(
                f"max_delay={dmax} >= wheel L={cfg.wheel_l} (violates "
                "max_delay < L wheel contract)"
            )
    if bundle.in_sum0.size and float(bundle.in_sum0.min()) <= 0:
        raise AssertionError(
            "some pool has a non-positive learnable in-sum (normalization "
            f"target): min in_sum0 = {float(bundle.in_sum0.min()):.4f}"
        )
    # learnable rows must be non-empty per pool (each pool has >= 16 learnable
    # input synapses)
    per_pool_learn = np.diff(bundle.in_learn_ptr)
    if int(per_pool_learn.min()) < 1:
        raise AssertionError(
            "some pool has no learnable incoming synapse: "
            f"min per-pool learn count = {int(per_pool_learn.min())}"
        )


def structural_report(bundle: NetworkBundle) -> dict:
    """Compile the structural/gain self-check report (dict of scalars).

    Contents (all keys used by the G0 self-check script):
        n_edges_total / n_edges_in / n_edges_pool, n_e_pool / n_i_pool,
        min/max per-source outdegree, pools_zero_indegree=0,
        pools_isolated=0, e_src_outdeg_max, delay histogram (list aligned to
        delays 1..delay_max), input_w_mean / e_w_mean / i_amp_min /
        i_amp_min_ge_e_mean (bool), gain_in_theta0 (mean input w / theta0),
        gain_e_theta0 (mean E w / theta0), theta0.
    """
    cfg = bundle.cfg
    outdeg = np.diff(bundle.csr_ptr)
    pool_outdeg = outdeg[cfg.n_in:cfg.n_total]
    indeg = np.bincount(bundle.csr_dst_local, minlength=cfg.n_pool)
    e_rows = bundle.pool_is_E
    hist = np.bincount(
        bundle.csr_delay, minlength=cfg.delay_max + 2
    )[cfg.delay_min:cfg.delay_max + 1].tolist()

    src_is_input = bundle.csr_src < cfg.n_in
    src_is_E = (~src_is_input) & bundle.pool_is_E[
        (bundle.csr_src - cfg.n_in).astype(np.intp)
    ]
    w_in = bundle.csr_w[src_is_input & bundle.csr_learn]
    w_e = bundle.csr_w[src_is_E]
    w_i = bundle.csr_w[~src_is_input & ~src_is_E]
    amp_i_min = float(np.abs(w_i).min()) if w_i.size else 0.0
    e_mean = float(w_e.mean()) if w_e.size else 0.0
    return {
        "n_pool": cfg.n_pool,
        "n_in": cfg.n_in,
        "n_edges_total": int(bundle.csr_w.size),
        "n_edges_in": int(bundle.n_edges_in),
        "n_edges_pool": int(bundle.n_edges_pool),
        "n_e_pool": int(cfg.n_e_pool),
        "n_i_pool": int(cfg.n_i_pool),
        "outdeg_min": int(pool_outdeg.min()) if pool_outdeg.size else 0,
        "outdeg_max": int(pool_outdeg.max()) if pool_outdeg.size else 0,
        "pools_zero_indegree": int(np.count_nonzero(indeg == 0)),
        "pools_isolated": int(np.count_nonzero(
            (indeg == 0) & (pool_outdeg == 0))),
        "e_src_outdeg_max": int(pool_outdeg[e_rows].max())
        if e_rows.any() else 0,
        "delay_hist": hist,
        "input_w_mean": float(w_in.mean()) if w_in.size else 0.0,
        "e_w_mean": e_mean,
        "i_amp_min": amp_i_min,
        "i_amp_min_ge_e_mean": bool(amp_i_min >= e_mean),
        "gain_in_theta0": (float(w_in.mean()) / cfg.theta0) if w_in.size else 0.0,
        "gain_e_theta0": e_mean / cfg.theta0 if w_e.size else 0.0,
        "theta0": float(cfg.theta0),
        "wheel_l": int(cfg.wheel_l),
    }


def print_report(bundle: NetworkBundle) -> None:
    """Print the structural self-check report (human-readable)."""
    r = structural_report(bundle)
    print("=== network structural report ===")
    print(f"  pools       : {r['n_pool']} (E={r['n_e_pool']}, I={r['n_i_pool']})")
    print(f"  inputs      : {r['n_in']}")
    print(f"  edges       : total={r['n_edges_total']} "
          f"(in={r['n_edges_in']}, pool={r['n_edges_pool']})")
    print(f"  pool outdeg : [{r['outdeg_min']}, {r['outdeg_max']}] "
          f"(E-source max {r['e_src_outdeg_max']} <= k_pool)")
    print(f"  zero indegree: {r['pools_zero_indegree']}   "
          f"isolated: {r['pools_isolated']}")
    print(f"  delay hist  : {r['delay_hist']}")
    print(f"  gain        : input mean={r['input_w_mean']:.3f} "
          f"({r['gain_in_theta0']:.2f}*theta0), E mean={r['e_w_mean']:.3f} "
          f"({r['gain_e_theta0']:.2f}*theta0)")
    print(f"  I amplitude : min |w|={r['i_amp_min']:.3f} >= E mean "
          f"{r['e_w_mean']:.3f}: {r['i_amp_min_ge_e_mean']}")
    print(f"  wheel       : L={r['wheel_l']} (theta0={r['theta0']})")