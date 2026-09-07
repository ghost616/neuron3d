"""selfcheck.py -- minimal core self-check (G0-annotated), D1-D4 scope.

The authoritative gate file is ``hstdn/exp/gates.py`` (exp module, not yet
available at this milestone).  Until it lands, this script provides the core
module's own minimal assertions, each annotated with the G0 item family it
serves (numbers follow the project gate plan):

    #4     layout ID contract & conversions; #13 shapes/dtypes/constants
    #10    SpatialHash3D query correctness
    #12    network wiring invariants (report: isolated=0, zero-in=0, E outdeg
          <= 12, delay histogram; CSR/CSC/B3 consistency; gain calibration)
    #1/#2/#3/#6/#7/#14/#15
           kernel: entry reset (B11), strict per-step order ①-⑥, strict
          refractory (R1), time wheel (max_delay < L), delivery timing,
          input-channel STDP (ADR-001), same-time classical order (D12)
    #5/#8/#9
           plasticity: input_channel_stdp / pre_ltd / post_ltp, homeo,
          competitive norm, weight bounds, capped-ratio / input-w diagnostics
    encoder (M2) sanity (not a numbered gate yet; feeds later gates)

Run: ``python -m hstdn.core.selfcheck`` from the project root (E:\\neuron3d).
Exit code 0 on full pass; any failure raises AssertionError with statistics.
"""

from __future__ import annotations

import sys

import numpy as np

from hstdn.core.layout import (
    I8, F8, NetConfig, DEFAULT_CFG, cfg_from_mapping,
    pool_local_of, pool_gid_of, WHEEL_L, N_IN, N_POOL, N_TOTAL,
)
from hstdn.core.spatial_hash import SpatialHash3D
from hstdn.core.network import NetworkBundle, build_network, structural_report
from hstdn.core.kernel import run_sample, reset_state
from hstdn.core import encoder
from hstdn.core import plasticity as plast

_PASS = 0


def _ok(name: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  [PASS] {name}")


# ---------------------------------------------------------------------------
# toy bundle for deterministic kernel / plasticity micro-tests
# ---------------------------------------------------------------------------
# sources: inputs 0..3, pools 4..11 (local 0..7).  Edges:
#   e0 input0(gid 0)   -> pool0 (local0) delay 1  w +1.00 learn
#   e1 input1(gid 1)   -> pool2 (local2) delay 15 w +1.00 learn
#   e2 pool0 (gid 4)   -> pool1 (local1) delay 2  w +0.30 learn (E source)
#   e3 pool1 (gid 5)   -> pool2 (local2) delay 1  w -0.50 frozen (I source)
# pool1 is the single I pool of the toy net; theta0 = 1.0.


def _toy_cfg() -> NetConfig:
    return NetConfig(n_in=4, n_pool=8, n_input_cols=2, n_e_pool=7,
                     n_i_pool=1, seed=0).with_derived()


def _make_toy_bundle() -> NetworkBundle:
    cfg = _toy_cfg()
    edges = [
        # (src_g, dst_local, delay, w, learn)
        (0, 0, 1, 1.00, True),
        (1, 2, 15, 1.00, True),
        (4, 1, 2, 0.30, True),
        (5, 2, 1, -0.50, False),
    ]
    src = np.asarray([e[0] for e in edges], dtype=I8)
    dst = np.asarray([e[1] for e in edges], dtype=I8)
    delay = np.asarray([e[2] for e in edges], dtype=I8)
    w = np.asarray([e[3] for e in edges], dtype=F8)
    learn = np.asarray([e[4] for e in edges], dtype=bool)
    # CSR: stable sort by (source asc, destination asc)
    order = np.lexsort((dst, src))
    csr_src = src[order]
    csr_dst_local = dst[order]
    csr_w = w[order]
    csr_delay = delay[order]
    csr_learn = learn[order]
    counts = np.bincount(csr_src, minlength=cfg.n_total).astype(I8)
    csr_ptr = np.zeros(cfg.n_total + 1, dtype=I8)
    csr_ptr[1:] = np.cumsum(counts)
    # CSC (B3 fix) with stable target argsort
    csc_ptr = np.zeros(cfg.n_pool + 1, dtype=I8)
    np.add.at(csc_ptr[1:], csr_dst_local, 1)
    np.cumsum(csc_ptr, out=csc_ptr)
    order_csc = np.argsort(csr_dst_local, kind="stable").astype(I8)
    dst_csc = csr_dst_local[order_csc]
    # learnable rows per pool
    learn_csc = csr_learn[order_csc]
    dst_learn = dst_csc[learn_csc]
    in_learn_idx = order_csc[learn_csc]
    lcnt = np.bincount(dst_learn, minlength=cfg.n_pool).astype(I8)
    in_learn_ptr = np.zeros(cfg.n_pool + 1, dtype=I8)
    in_learn_ptr[1:] = np.cumsum(lcnt)
    in_sum0 = np.bincount(dst_learn, weights=csr_w[in_learn_idx],
                          minlength=cfg.n_pool).astype(F8)
    pool_is_E = np.ones(cfg.n_pool, dtype=bool)
    pool_is_E[1] = False  # pool1 = I
    bundle = NetworkBundle(
        cfg=cfg,
        input_xyz=np.zeros((cfg.n_in, 3), dtype=F8),
        pool_xyz=np.zeros((cfg.n_pool, 3), dtype=F8),
        pool_is_E=pool_is_E,
        n_edges_in=2,
        n_edges_pool=2,
        csr_ptr=csr_ptr,
        csr_src_g=np.arange(cfg.n_total, dtype=I8),
        csr_dst=csr_dst_local + cfg.n_in,
        csr_dst_local=csr_dst_local,
        csr_w=csr_w,
        csr_delay=csr_delay,
        csr_learn=csr_learn,
        csr_src=csr_src,
        csc_ptr=csc_ptr,
        csc_src=csr_src[order_csc],
        csc_csr_pos=order_csc,
        in_learn_ptr=in_learn_ptr,
        in_learn_idx=in_learn_idx,
        in_sum0=in_sum0,
        V=np.zeros(cfg.n_pool, dtype=F8),
        theta=np.full(cfg.n_pool, cfg.theta0, dtype=F8),
        refr=np.full(cfg.n_pool, -1, dtype=I8),
        counts=np.zeros(cfg.n_pool, dtype=I8),
        first_spike=np.full(cfg.n_pool, -1, dtype=I8),
        rate_ema=np.zeros(cfg.n_pool, dtype=F8),
        trace=np.zeros(cfg.n_total, dtype=F8),
        ring=np.zeros((cfg.wheel_l, cfg.n_pool), dtype=F8),
    )
    return bundle


def _ev(*ids: int):
    return {0: (np.asarray(list(ids), dtype=I8), np.ones(len(ids), dtype=F8))}


# ---------------------------------------------------------------------------
# layout (G0 #4 / #13)
# ---------------------------------------------------------------------------


def check_layout() -> None:
    print("layout (G0 #4 / #13):")
    # constants coherence
    assert WHEEL_L == DEFAULT_CFG.wheel_l == DEFAULT_CFG.delay_max + 1
    assert N_TOTAL == N_IN + N_POOL
    assert pool_local_of(N_IN) == 0
    assert pool_local_of(N_IN + N_POOL - 1) == N_POOL - 1
    assert pool_gid_of(0) == N_IN
    raised = False
    try:
        pool_local_of(0)  # input gid 0 is not a pool
    except AssertionError:
        raised = True
    assert raised, "pool_local_of must reject input global ids"
    raised = False
    try:
        cfg_from_mapping({"n_pool_typo": 10})
    except ValueError:
        raised = True
    assert raised, "unknown config keys must raise"
    c2 = cfg_from_mapping({"n_pool": 100})
    assert c2.n_pool == 100 and c2.n_total == c2.n_in + 100
    assert c2.wheel_l == DEFAULT_CFG.wheel_l
    # dtype constants
    assert np.dtype(F8) == np.float64 and np.dtype(I8) == np.int64
    _ok("constants & ID conversion/domain contracts")


def check_shapes() -> None:
    print("shape/dtype helpers (G0 #13):")
    from hstdn.core.layout import (
        assert_pool_local_shape1, assert_global_trace_shape, assert_ring_shape,
        assert_f8, assert_i8,
    )
    assert_pool_local_shape1(np.zeros(N_POOL))
    assert_global_trace_shape(np.zeros(N_TOTAL))
    assert_ring_shape(np.zeros((WHEEL_L, N_POOL)))
    assert_f8(np.zeros(3, dtype=np.float64))
    assert_i8(np.zeros(3, dtype=np.int64))
    for fn, bad in [
        (assert_pool_local_shape1, np.zeros(N_POOL + 1)),
        (assert_global_trace_shape, np.zeros(N_IN)),
        (assert_ring_shape, np.zeros((WHEEL_L, N_POOL + 1))),
        (assert_f8, np.zeros(3, dtype=np.float32)),
        (assert_i8, np.zeros(3, dtype=np.int32)),
    ]:
        try:
            fn(bad)
        except AssertionError:
            continue
        raise AssertionError(f"{fn.__name__} must reject the bad input")
    _ok("shape/dtype invariant helpers")


# ---------------------------------------------------------------------------
# spatial hash (G0 #10)
# ---------------------------------------------------------------------------


def check_spatial_hash() -> None:
    print("spatial_hash (G0 #10):")
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.0, 1.0, size=(2000, 3))
    hs = SpatialHash3D(pts, cell_size=0.15)
    q = rng.uniform(0.0, 1.0, size=(100, 3))
    for i in range(100):
        got = hs.query_radius(q[i], 0.2)
        d = np.linalg.norm(pts - q[i], axis=1)
        expect = np.flatnonzero(d <= 0.2)
        expect = expect[np.argsort(d[expect], kind="stable")]
        assert np.array_equal(got, expect), f"query {i} mismatch"
        gotk, dk = hs.nearest_k(q[i], 7, 0.2)
        if expect.size:
            assert np.array_equal(gotk, expect[: min(7, expect.size)])
            assert np.allclose(dk, d[gotk], atol=1e-12)
    # exclude-self behaviour
    q0 = pts[3] + 1e-9
    got = hs.query_radius(q0, 1e-6, exclude_self=True, self_id=3)
    assert 3 not in got
    _ok("radius & k-nearest queries match brute force (O(1) cells)")


# ---------------------------------------------------------------------------
# network (G0 #12 / #13 / #10)
# ---------------------------------------------------------------------------


def check_network() -> None:
    print("network (G0 #12 / #13 / #10):")
    bundle = build_network()  # default config, seeded & deterministic
    r = structural_report(bundle)
    assert r["pools_zero_indegree"] == 0
    assert r["pools_isolated"] == 0
    assert r["e_src_outdeg_max"] <= DEFAULT_CFG.k_pool
    assert r["n_edges_total"] == r["n_edges_in"] + r["n_edges_pool"]
    assert r["n_e_pool"] == DEFAULT_CFG.n_e_pool
    assert r["n_i_pool"] == DEFAULT_CFG.n_i_pool
    assert len(r["delay_hist"]) == DEFAULT_CFG.delay_max
    assert sum(r["delay_hist"]) == r["n_edges_total"]
    # gain calibration targets
    assert 0.90 <= r["gain_in_theta0"] <= 1.30
    assert 0.18 <= r["gain_e_theta0"] <= 0.42
    assert r["i_amp_min_ge_e_mean"] is True
    # structural cross checks
    cfg = bundle.cfg
    assert bundle.csc_ptr[-1] == bundle.csr_w.size
    assert np.array_equal(bundle.csc_src, bundle.csr_src[bundle.csc_csr_pos])
    cnt = np.bincount(bundle.csr_dst_local, minlength=cfg.n_pool)
    assert np.array_equal(np.diff(bundle.csc_ptr), cnt)
    lcnt = np.bincount(bundle.csr_dst_local[bundle.csr_learn],
                       minlength=cfg.n_pool)
    assert np.array_equal(np.diff(bundle.in_learn_ptr), lcnt)
    s0 = np.bincount(bundle.csr_dst_local[bundle.csr_learn],
                     weights=bundle.csr_w[bundle.csr_learn],
                     minlength=cfg.n_pool)
    assert np.allclose(s0, bundle.in_sum0, atol=1e-12)
    assert bundle.csr_delay.min() >= 1
    assert bundle.csr_delay.max() < cfg.wheel_l
    src_in = bundle.csr_src < cfg.n_in
    src_E = (~src_in) & bundle.pool_is_E[(bundle.csr_src - cfg.n_in).astype(int)]
    assert np.array_equal(bundle.csr_learn, src_in | src_E)
    assert np.all(bundle.csr_w[src_in | src_E] > 0)
    assert np.all(bundle.csr_w[~src_in & ~src_E] < 0)
    _ok("wiring invariants, CSR/CSC/B3 consistency and gain report")
    return bundle


# ---------------------------------------------------------------------------
# kernel micro tests (G0 #1/#2/#3/#6/#7/#14/#15 family)
# ---------------------------------------------------------------------------


def check_kernel_basic() -> None:
    print("kernel micro (G0 #1/#2/#3/#6/#7/#14/#15 family):")
    # delivery timing: input0(t=0, delay 1) -> pool0 fires exactly at t=1
    b = _make_toy_bundle()
    st = run_sample(b, _ev(0), T=20)
    assert st["n_spikes_total"] == 1
    assert b.counts[0] == 1 and b.first_spike[0] == 1
    assert b.counts[1] == 0 and b.counts[2] == 0
    # B11: entry reset each sample
    b2 = _make_toy_bundle()
    run_sample(b2, {}, T=10)
    assert not np.any(b2.V) and not np.any(b2.ring) and not np.any(b2.trace)
    assert np.all(b2.refr == -1) and not np.any(b2.counts)
    assert np.all(b2.first_spike == -1)
    assert np.all(b2.theta == 1.0)  # preserved default
    # B11: theta/rate_ema preserved across samples
    b3 = _make_toy_bundle()
    run_sample(b3, _ev(0, 0), T=20)  # same input at t=0 twice -> strengths 2
    theta_before = b3.theta.copy()
    ema_before = b3.rate_ema.copy()
    run_sample(b3, {}, T=5)
    assert np.array_equal(b3.theta, theta_before)
    assert np.array_equal(b3.rate_ema, ema_before)
    _ok("delivery timing at delay + entry reset contract (B11)")


def check_kernel_refractory() -> None:
    print("kernel refractory (G0 #1 R1):")
    # arrivals at t=1,3,5; the t=3 arrival is blocked by refr=1+REFR=3
    # (strict t > refr) -> pool0 fires exactly twice
    b = _make_toy_bundle()
    buckets = {
        0: (np.asarray([0], dtype=I8), np.ones(1, dtype=F8)),
        2: (np.asarray([0], dtype=I8), np.ones(1, dtype=F8)),
        4: (np.asarray([0], dtype=I8), np.ones(1, dtype=F8)),
    }
    st = run_sample(b, buckets, T=20)
    assert st["n_spikes_total"] == 2
    assert b.counts[0] == 2 and b.first_spike[0] == 1
    _ok("strict refractory: blocked at t = fire+REFR, allowed at t > refr")


def check_kernel_wheel_far_delay() -> None:
    print("kernel wheel far delay (G0 #2/#3):")
    # input1 -> pool2 with delay 15 (max < wheel L=16): fires at t=15
    b = _make_toy_bundle()
    st = run_sample(b, _ev(1), T=20)
    assert st["n_spikes_total"] == 1
    assert b.counts[2] == 1 and b.first_spike[2] == 15
    _ok("wheel slot wrap with max delay < L")


def check_kernel_stdp() -> None:
    print("kernel STDP ordering (G0 #6/#7/#14/#15, ADR-001/D12):")
    b = _make_toy_bundle()
    w0 = b.csr_w[0].copy()   # e0 input0->pool0
    w1 = b.csr_w[1].copy()   # e1 input1->pool2
    w3 = b.csr_w[3].copy()   # e3 frozen I edge
    run_sample(b, {0: (np.asarray([0, 1], dtype=I8),
                       np.ones(2, dtype=F8))}, T=25, stdp_on=True)
    dt = b.cfg.decay_t
    eta = b.cfg.eta_ltp
    # pool0 fires at t=1 -> LTP on e0 with trace[0]=decay_t^1 (input trace set
    # at t=0 then decayed once at the end of t=0)
    assert np.isclose(b.csr_w[0], w0 + eta * dt)
    # pool2 fires at t=15 -> LTP on e1 with trace[1]=decay_t^15
    assert np.isclose(b.csr_w[1], w1 + eta * dt ** 15)
    # frozen I edge untouched and still negative
    assert b.csr_w[3] == w3 and b.csr_w[3] < 0
    assert b.counts[0] == 1 and b.counts[2] == 1
    _ok("input-channel trace precedes same-time post_ltp (pre->post potentiation)")


# ---------------------------------------------------------------------------
# plasticity (G0 #5/#8/#9)
# ---------------------------------------------------------------------------


def check_plasticity_units() -> None:
    print("plasticity units (G0 #5/#8/#9):")
    # --- post_ltp via CSC (learnable only, I frozen) ---
    b = _make_toy_bundle()
    src0 = int(b.csc_src[b.csc_ptr[0]])  # input0 gid
    b.trace[src0] = 1.0
    w_before = b.csr_w.copy()
    plast.post_ltp(b, 0)
    assert np.isclose(b.csr_w[0], w_before[0] + b.cfg.eta_ltp)
    # --- pre_ltd via CSR on an E pool ---
    b = _make_toy_bundle()
    b.trace[5] = 1.0  # pool1 (dst of e2) recently fired
    plast.pre_ltd(b, 4)  # pool0 fires as pre
    assert np.isclose(b.csr_w[2], 0.30 * (1 - b.cfg.eta_ltd))
    # --- input_channel_stdp (ADR-001): LTD all outgoing + set trace ---
    b = _make_toy_bundle()
    b.trace[6] = 1.0  # pool2 (dst of e1) recently fired
    plast.input_channel_stdp(b, np.asarray([1], dtype=I8))
    assert np.isclose(b.csr_w[1], 1.0 * (1 - b.cfg.eta_ltd))
    assert b.trace[1] == 1.0
    # I-source pre_ltd is a no-op (its row is frozen)
    b = _make_toy_bundle()
    b.trace[6] = 1.0
    plast.pre_ltd(b, 5)  # pool1 (I) fires: row e3 not learnable
    assert b.csr_w[3] == -0.5
    # --- homeostasis formula ---
    b = _make_toy_bundle()
    b.counts[0] = 40  # 40 spikes / 0.2 s = 200 Hz
    plast.homeo_update(b)
    assert np.isclose(b.rate_ema[0], 0.1 * 200.0)
    assert np.isclose(b.theta[0], np.clip(1.0 + 0.05 * (20.0 - 8.0), 0.5, 20.0))
    assert b.theta[0] == 1.6
    # --- competitive normalization restores in_sum0 per pool, I frozen ---
    b = _make_toy_bundle()
    b.csr_w[0] = 1.5
    b.csr_w[2] = 0.45
    plast.competitive_norm(b)
    assert np.isclose(b.csr_w[0], 1.0, atol=1e-9)   # pool0 target 1.0
    assert np.isclose(b.csr_w[2], 0.3, atol=1e-9)   # pool1 target 0.3
    assert b.csr_w[3] == -0.5                        # I frozen
    for p in range(b.n_pool):
        seg = b.in_learn_idx[b.in_learn_ptr[p]:b.in_learn_ptr[p + 1]]
        if seg.size:
            assert np.isclose(b.csr_w[seg].sum(), b.in_sum0[p], atol=1e-9)
    _ok("input_channel_stdp / pre_ltd / post_ltp / homeo / norm unit checks")
    # --- diagnostics ---
    b = _make_toy_bundle()
    b.csr_w[0] = 1.5
    b.csr_w[1] = 1.5
    d = plast.plasticity_diagnostics(b)
    assert 0.0 <= d["capped_ratio"] <= 1.0
    assert np.isclose(d["input_w_mean"], (1.5 + 1.5) / 2.0)
    assert d["n_learnable"] == 3
    assert d["capped_ratio"] == 2.0 / 3.0  # e0,e1 at cap; e2 (0.3) is not
    _ok("capped-ratio & input->pool weight diagnostics")


# ---------------------------------------------------------------------------
# encoder (M2 sanity)
# ---------------------------------------------------------------------------


def check_encoder() -> None:
    print("encoder (M2):")
    inten = np.asarray([1.0, 0.5, 0.0, 0.1, 0.13], dtype=F8)
    bks = encoder.latency_encode(inten)
    # I=1 -> t=2; I=0.5 -> t=22; I=0.13 -> t=37 (threshold strict > 0.12)
    assert sorted(bks.keys()) == [2, 22, 37]
    assert [int(x) for x in bks[2][0]] == [0]
    assert [int(x) for x in bks[22][0]] == [1]
    assert [int(x) for x in bks[37][0]] == [4]
    assert int(np.unique(np.concatenate([v[0] for v in bks.values()])).size) == 3
    total = sum(int(v[0].size) for v in bks.values())
    assert total == 3
    # empty frame
    assert encoder.latency_encode(np.zeros(10)) == {}
    # MNIST adaptive pooling 28x28 -> 10x10
    img = np.arange(28 * 28, dtype=F8).reshape(28, 28)
    pooled = encoder.mnist_adaptive_pool(img, out=10)
    assert pooled.shape == (10, 10)
    # first pooled cell averages rows 0..2, cols 0..2 (torch semantics)
    expect00 = float(np.mean(img[0:3, 0:3]))
    assert np.isclose(pooled[0, 0], expect00)
    # row-major ids on a N_IN=100 cfg
    mcfg = cfg_from_mapping({"n_in": 100, "n_input_cols": 10})
    mb = encoder.mnist_encode(img, out=10, cfg=mcfg)
    ids_all = np.concatenate([v[0] for v in mb.values()])
    assert int(ids_all.min()) >= 0 and int(ids_all.max()) <= 99
    # DVS interface raises until the G3 milestone
    try:
        encoder.dvs_patch_aggregate(None)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("dvs_patch_aggregate must raise NotImplementedError")
    _ok("latency buckets (FIX-B8), MNIST 10x10 pool mapping, DVS interface")


# ---------------------------------------------------------------------------
# default-bundle dynamics smoke (B11 across samples; stdp/homeo/norm run)
# ---------------------------------------------------------------------------


def check_dynamics(bundle: NetworkBundle) -> None:
    print("dynamics smoke (default bundle):")
    rng = np.random.default_rng(3)
    # two random latency frames -> buckets (ids = flat input grid positions)
    def frame_buckets() -> dict:
        img = rng.uniform(0.0, 1.0, size=(N_IN,))
        return encoder.latency_encode(img)
    b1 = frame_buckets()
    # determinism: two plasticity-off runs over the same frame are identical
    st1 = run_sample(bundle, b1, T=100)
    st1b = run_sample(bundle, b1, T=100)
    assert st1b["n_spikes_total"] == st1["n_spikes_total"]
    assert np.array_equal(st1b["spike_counts"], st1["spike_counts"])
    theta_no_homeo = bundle.theta.copy()
    # plasticity + homeo + norm sample
    st2 = run_sample(bundle, frame_buckets(), T=100, stdp_on=True,
                     homeo_on=True, norm_on=True)
    assert st2["n_spikes_total"] >= 0
    assert np.all(bundle.theta >= bundle.cfg.theta_lo - 1e-9)
    assert np.all(bundle.theta <= bundle.cfg.theta_hi + 1e-9)
    # B11: theta stayed untouched while homeo was off, then moved once on
    assert np.allclose(theta_no_homeo, DEFAULT_CFG.theta0)
    assert np.any(np.abs(bundle.theta - theta_no_homeo) > 1e-9)
    d = plast.plasticity_diagnostics(bundle)
    assert 0.0 <= d["capped_ratio"] <= 1.0
    assert np.all(bundle.csr_w[bundle.csr_learn] >= bundle.cfg.w_lo - 1e-9)
    assert np.all(bundle.csr_w[bundle.csr_learn] <= bundle.cfg.w_hi + 1e-9)
    _ok("deterministic re-runs; stdp/homeo/norm bounds respected")


def main() -> int:
    global _PASS
    check_layout()
    check_shapes()
    check_spatial_hash()
    bundle = check_network()
    check_kernel_basic()
    check_kernel_refractory()
    check_kernel_wheel_far_delay()
    check_kernel_stdp()
    check_plasticity_units()
    check_encoder()
    check_dynamics(bundle)
    print(f"\ncore selfcheck PASSED: {_PASS} assertion groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())