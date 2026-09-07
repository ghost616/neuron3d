"""lizhu_extra_core_tests.py -- supplementary unit tests for hstdn.core (D1-D4).

Complement to ``python -m hstdn.core.selfcheck``.  Covers reverse / boundary /
edge dimensions the built-in self-check does not exercise:

- layout: config validation errors (non-positive n_in/n_pool, delay_min<1,
  delay_max<delay_min, bad theta0), partial overrides & derived fields,
  ID-domain boundary rejections, dtype/shape rejections incl. 2-D.
- spatial_hash: empty point set, bad cell_size/NaN/malformed points,
  exclude_self without self_id, k=0, radius boundary d==r, fewer-than-k,
  ties broken by index, deterministic rebuilds.
- network: deterministic rebuilds with a fixed seed (small config), change
  under a different seed, small-config structural report, gain bounds.
- kernel: T<=0 ValueError, out-of-range bucket times, out-of-domain input
  gids, run_sample_l1 NotImplementedError, no-STDP leaves weights intact,
  determinism of two independent runs, extra refractory boundary case.
- plasticity: pre_ltd rejects input gids, post_ltp rejects out-of-range
  locals, weight clip at w_lo/w_hi bounds, homeo clip at theta_lo/theta_hi,
  empty competitive_norm, diagnostics on toy bundles.
- encoder: threshold strictness at I=0.12, clip of out-of-range intensities,
  ids_offset, per-bucket ascending ids, mnist adaptive pooling cell boundary
  checks (floor/ceil semantics), mnist_encode N_IN mismatch ValueError,
  DVS NotImplementedError.

Run from the package root E:\neuron3d with PYTHONPATH set to it:
Exit 0 on full pass.
"""

from __future__ import annotations

import sys

import numpy as np

from hstdn.core import selfcheck as sc  # reuse toy-bundle builders (test-only)
from hstdn.core import encoder, plasticity as plast
from hstdn.core.kernel import run_sample, run_sample_l1
from hstdn.core.layout import (
    I8, F8, NetConfig, DEFAULT_CFG, cfg_from_mapping,
    is_input_gid, is_pool_gid,
    pool_local_of, pool_gid_of, WHEEL_L, N_IN, N_POOL, N_TOTAL,
    assert_input_gids, assert_pool_gids, assert_pool_local_ids,
    assert_pool_local_shape1, assert_global_trace_shape, assert_ring_shape,
    assert_f8, assert_i8,
)
from hstdn.core.spatial_hash import SpatialHash3D
from hstdn.core.network import build_network, structural_report

_PASS = 0


def _ok(name: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  [PASS] {name}")


def _raises(exc, fn, name: str) -> None:
    """Assert fn() raises exc; PASS otherwise FAIL (AssertionError)."""
    try:
        fn()
    except exc:
        _ok(name)
        return
    raise AssertionError(f"{name}: expected {exc.__name__} to be raised")


def _run(name: str, fn) -> None:
    """Run one named check group; a raised failure aborts the script."""
    print(name)
    fn()


# ---------------------------------------------------------------------------
# layout -- reverse & boundary (G0 #4 / #13)
# ---------------------------------------------------------------------------


def check_layout_extra() -> None:
    # --- with_derived validation (reverse) ---
    _raises(ValueError, lambda: NetConfig(n_in=-5).with_derived(),
            "NetConfig(n_in=-5) rejected")
    _raises(ValueError, lambda: NetConfig(n_pool=0).with_derived(),
            "NetConfig(n_pool=0) rejected")
    _raises(ValueError, lambda: NetConfig(delay_min=0).with_derived(),
            "NetConfig(delay_min=0) rejected")
    _raises(ValueError, lambda: NetConfig(delay_min=8, delay_max=5).with_derived(),
            "delay_max<delay_min rejected")
    _raises(ValueError, lambda: NetConfig(theta0=0.0).with_derived(),
            "NetConfig(theta0=0) rejected")
    _raises(ValueError, lambda: NetConfig(theta0=float("nan")).with_derived(),
            "NetConfig(theta0=nan) rejected")
    _raises(ValueError, lambda: NetConfig(theta0=float("inf")).with_derived(),
            "NetConfig(theta0=inf) rejected")
    # --- cfg_from_mapping: full defaults / partial overrides / derived ---
    assert cfg_from_mapping(None) is DEFAULT_CFG
    c = cfg_from_mapping({"n_in": 100, "n_pool": 1000, "delay_max": 9})
    assert c.n_total == 1100 and c.wheel_l == 10
    assert c.n_e_pool == int(round(1000 * 0.8)) and c.n_i_pool == 1000 - c.n_e_pool
    assert np.isclose(c.decay_v, np.exp(-1.0 / 20.0))
    assert np.isclose(c.decay_t, np.exp(-1.0 / 20.0))
    _raises(ValueError, lambda: cfg_from_mapping({"n_poolzz": 3}),
            "cfg_from_mapping unknown key rejected")
    # --- ID domain boundary rejections ---
    for bad_gid in (-1, N_IN - 1, N_TOTAL):
        _raises(AssertionError, lambda g=bad_gid: pool_local_of(g),
                f"pool_local_of({bad_gid}) rejected")
    _raises(AssertionError, lambda: pool_gid_of(-1), "pool_gid_of(-1) rejected")
    _raises(AssertionError, lambda: pool_gid_of(N_POOL), "pool_gid_of(N_POOL) rejected")
    assert pool_gid_of(N_POOL - 1) == N_TOTAL - 1
    assert pool_local_of(N_IN + 1234) == 1234
    assert pool_gid_of(pool_local_of(N_IN + N_POOL - 1)) == N_IN + N_POOL - 1
    # --- boolean domain predicates at the boundaries ---
    assert is_input_gid(0) and is_input_gid(N_IN - 1)
    assert not is_input_gid(N_IN) and not is_input_gid(-1)
    assert is_pool_gid(N_IN) and is_pool_gid(N_IN + N_POOL - 1)
    assert not is_pool_gid(N_IN - 1) and not is_pool_gid(N_IN + N_POOL)
    # --- array domain guards: mixed / out-of-domain / wrong dtype ---
    _raises(AssertionError,
            lambda: assert_input_gids(np.asarray([0, N_IN], dtype=I8)),
            "assert_input_gids rejects a pool gid")
    _raises(AssertionError,
            lambda: assert_pool_gids(np.asarray([N_IN, 0], dtype=I8)),
            "assert_pool_gids rejects an input gid")
    _raises(AssertionError,
            lambda: assert_pool_local_ids(np.asarray([0, N_POOL], dtype=I8)),
            "assert_pool_local_ids rejects id==N_POOL")
    _raises(AssertionError,
            lambda: assert_input_gids(np.asarray([1.5])),
            "assert_input_gids rejects non-integer dtype")
    assert_input_gids(np.asarray([], dtype=I8))          # empty ok
    assert_pool_local_ids(np.asarray([], dtype=I8))
    # --- shape/dtype guards incl. 2-D ---
    _raises(AssertionError,
            lambda: assert_pool_local_shape1(np.zeros((N_POOL, 1))),
            "pool shape helper rejects 2-D")
    _raises(AssertionError,
            lambda: assert_global_trace_shape(np.zeros(N_TOTAL - 1)),
            "trace shape helper rejects short array")
    _raises(AssertionError,
            lambda: assert_ring_shape(np.zeros((WHEEL_L, N_POOL, 1))),
            "ring shape helper rejects 3-D")
    _raises(AssertionError, lambda: assert_f8(np.zeros(2, dtype=np.int64)),
            "assert_f8 rejects int dtype")
    _raises(AssertionError, lambda: assert_i8(np.zeros(2, dtype=np.float64)),
            "assert_i8 rejects float dtype")
    _ok("layout reverse/boundary group")

# ---------------------------------------------------------------------------
# spatial_hash -- edge cases (G0 #10)
# ---------------------------------------------------------------------------


def check_hash_extra() -> None:
    # empty point set
    hs0 = SpatialHash3D(np.zeros((0, 3)), cell_size=0.1)
    assert hs0.query_radius([0.0, 0.0, 0.0], 1.0).size == 0
    gi, di = hs0.nearest_k([0.0, 0.0, 0.0], 3, 1.0)
    assert gi.size == 0 and di.size == 0
    _ok("empty point set queries return empty")
    # malformed construction
    _raises(ValueError, lambda: SpatialHash3D(np.zeros((5, 2)), 0.1),
            "non-(N,3) points rejected")
    _raises(ValueError, lambda: SpatialHash3D(np.zeros((4, 3)), 0.0),
            "cell_size<=0 rejected")
    _raises(ValueError,
            lambda: SpatialHash3D(np.full((4, 3), np.nan), 0.1),
            "NaN points rejected")
    _ok("construction guard group")
    # exclude_self without self_id
    pts = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=F8)
    hs = SpatialHash3D(pts, cell_size=0.1)
    _raises(ValueError, lambda: hs.query_radius([0.0, 0.0, 0.0], 0.5,
                                                exclude_self=True),
            "exclude_self without self_id rejected")
    # radius boundary: point at exactly d == r must be included
    got = hs.query_radius([0.0, 0.0, 0.0], 0.2)
    assert int(got.size) == 2
    got = hs.query_radius([0.0, 0.0, 0.0], 0.2 - 1e-9)
    assert int(got.size) == 1 and int(got[0]) == 0
    # fewer than k available -> return what exists
    gi, di = hs.nearest_k([0.0, 0.0, 0.0], 7, 0.5)
    assert int(gi.size) == 2 and int(di.size) == 2
    gi, di = hs.nearest_k([0.0, 0.0, 0.0], 0, 0.5)
    assert gi.size == 0 and di.size == 0
    _ok("radius boundary / fewer-than-k / k=0")
    # tie breaking by ascending index (equal distance)
    pts2 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                     [0.5, 0.0, 0.0]], dtype=F8)
    hs2 = SpatialHash3D(pts2, cell_size=0.5)
    got = hs2.query_radius([0.0, 0.0, 0.0], 1.2)
    d = np.linalg.norm(pts2[got], axis=1)
    assert np.all(np.diff(d) >= -1e-12)
    # closest point idx4 (d=0.5) first, then the four exact ties
    # (d=1.0) sorted by ascending index: ids 0,1,2,3
    assert int(got[0]) == 4
    assert np.array_equal(got[1:], np.arange(4, dtype=I8))
    # exclude_self drops its own coincident point
    pts3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=F8)
    hs3 = SpatialHash3D(pts3, cell_size=0.5)
    g = hs3.query_radius(pts3[0] + 1e-9, 1.0, exclude_self=True, self_id=0)
    assert int(g.size) == 1 and int(g[0]) == 1
    gk, _dk = hs3.nearest_k(pts3[0] + 1e-9, 1, 1.0,
                            exclude_self=True, self_id=0)
    assert int(gk.size) == 1 and int(gk[0]) == 1
    _ok("tie order & exclude-self")
    # determinism: two identical builds give identical query results
    rng = np.random.default_rng(11)
    A = rng.uniform(0, 1, size=(500, 3))
    hA = SpatialHash3D(A, 0.1)
    hB = SpatialHash3D(A.copy(), 0.1)
    q = rng.uniform(0, 1, size=20)
    for i in range(0, 20, 2):
        assert np.array_equal(hA.query_radius(q[i:i + 1], 0.3),
                              hB.query_radius(q[i:i + 1], 0.3))
    _ok("deterministic queries across identical builds")


# ---------------------------------------------------------------------------
# network -- determinism & small-config structural report (G0 #12/#13)
# ---------------------------------------------------------------------------


_SMALL = {"n_in": 100, "n_input_cols": 10, "n_pool": 300, "seed": 11}


def check_network_extra() -> None:
    cfg = cfg_from_mapping(_SMALL)
    b1 = build_network(cfg)
    b2 = build_network(cfg)  # same seed -> byte-identical network
    for f in ("input_xyz", "pool_xyz", "pool_is_E", "csr_ptr", "csr_dst_local",
              "csr_w", "csr_delay", "csr_learn", "csr_src", "csc_ptr",
              "csc_src", "csc_csr_pos", "in_learn_ptr", "in_learn_idx",
              "in_sum0"):
        assert np.array_equal(getattr(b1, f), getattr(b2, f)), f"{f} differs"
    _ok("deterministic rebuild under fixed seed")
    b3 = build_network(cfg_from_mapping({**_SMALL, "seed": 12}))
    assert not np.array_equal(b1.pool_xyz, b3.pool_xyz), "seed must change layout"
    _ok("different seed changes the layout")
    # small-config structural report mirrors the default-config expectations
    r = structural_report(b1)
    assert r["n_pool"] == 300 and r["n_in"] == 100
    assert r["pools_zero_indegree"] == 0 and r["pools_isolated"] == 0
    assert r["n_e_pool"] == int(round(300 * 0.8))
    assert r["e_src_outdeg_max"] <= cfg.k_pool
    assert sum(r["delay_hist"]) == r["n_edges_total"]
    assert r["n_edges_total"] == r["n_edges_in"] + r["n_edges_pool"]
    assert 0.90 <= r["gain_in_theta0"] <= 1.30
    assert 0.18 <= r["gain_e_theta0"] <= 0.42
    assert r["i_amp_min_ge_e_mean"] is True
    # weight sign per class; I edges frozen
    src_in = b1.csr_src < cfg.n_in
    src_E = (~src_in) & b1.pool_is_E[(b1.csr_src - cfg.n_in).astype(int)]
    assert np.all(b1.csr_w[src_in | src_E] > 0)
    assert np.all(b1.csr_w[~src_in & ~src_E] < 0)
    assert np.array_equal(b1.csr_learn, src_in | src_E)
    _ok("small-config structural report & class invariants")
    # invalid layout: n_in != n_input_cols**2
    _raises(ValueError, lambda: build_network({"n_in": 10, "n_input_cols": 4}),
            "build_network rejects non-square input grid")


# ---------------------------------------------------------------------------
# kernel -- reverse / boundary (G0 #1/#2/#3 family)
# ---------------------------------------------------------------------------


def check_kernel_extra() -> None:
    # T must be positive
    b = sc._make_toy_bundle()
    _raises(ValueError, lambda: run_sample(b, {}, T=0), "T=0 rejected")
    _raises(ValueError, lambda: run_sample(b, {}, T=-3), "T<0 rejected")
    # bucket times must lie in [0, T)
    b = sc._make_toy_bundle()
    _raises(AssertionError, lambda: run_sample(b, {5: (np.asarray([0], dtype=I8),
                                                     np.ones(1))}, T=5),
            "bucket time == T rejected")
    _raises(AssertionError, lambda: run_sample(b, {-1: (np.asarray([0], dtype=I8),
                                                       np.ones(1))}, T=10),
            "negative bucket time rejected")
    # input ids must be input global ids
    _raises(AssertionError, lambda: run_sample(b, {0: (np.asarray([4], dtype=I8),
                                                      np.ones(1))}, T=10),
            "input gid == N_IN rejected")
    _raises(AssertionError, lambda: run_sample(b, {0: (np.asarray([-2], dtype=I8),
                                                      np.ones(1))}, T=10),
            "negative input gid rejected")
    # L1 entry raises NotImplementedError
    _raises(NotImplementedError, lambda: run_sample_l1(b, {}, T=10),
            "run_sample_l1 NotImplementedError")
    # no-STDP sample leaves weights / theta untouched
    b = sc._make_toy_bundle()
    w0 = b.csr_w.copy()
    th0 = b.theta.copy()
    st = run_sample(b, sc._ev(0), T=20, stdp_on=False, homeo_on=False,
                    norm_on=False)
    assert np.array_equal(b.csr_w, w0)
    assert np.array_equal(b.theta, th0)
    assert st["n_spikes_total"] == 1
    _ok("weights/theta untouched without plasticity")
    # determinism across two independent runs on fresh bundles
    bA, bB = sc._make_toy_bundle(), sc._make_toy_bundle()
    ev = sc._ev(0, 1)
    stA = run_sample(bA, ev, T=25, stdp_on=True)
    stB = run_sample(bB, ev, T=25, stdp_on=True)
    assert stA["n_spikes_total"] == stB["n_spikes_total"]
    assert np.array_equal(stA["spike_counts"], stB["spike_counts"])
    assert np.array_equal(bA.csr_w, bB.csr_w)
    assert np.array_equal(bA.ring, bB.ring)
    _ok("kernel determinism (stdp path)")
    # refractory boundary: blocked at t=fire+REFR, allowed at t=fire+REFR+1
    # toy: input0 -> pool0 delay 1, w=+1, theta0=1 -> fire at t=1, refr=3.
    # an arrival at t=3 (input at t=2) is blocked; V persists; an arrival
    # at t=4 (input at t=3) crosses theta at t=4 -> second fire.
    b = sc._make_toy_bundle()
    buckets = {0: (np.asarray([0], dtype=I8), np.ones(1)),
               2: (np.asarray([0], dtype=I8), np.ones(1)),
               3: (np.asarray([0], dtype=I8), np.ones(1))}
    st = run_sample(b, buckets, T=20)
    assert st["n_spikes_total"] == 2, f"got {st['n_spikes_total']}"
    assert b.counts[0] == 2 and b.first_spike[0] == 1
    _ok("refractory boundary: blocked at t=refr, allowed at t>refr")

# ---------------------------------------------------------------------------
# plasticity -- reverse / clip boundaries (G0 #5/#8/#9)
# ---------------------------------------------------------------------------


def check_plasticity_extra() -> None:
    # pre_ltd must reject an input (non-pool) source gid
    b = sc._make_toy_bundle()
    _raises(AssertionError, lambda: plast.pre_ltd(b, 2),
            "pre_ltd rejects input gid")
    # post_ltp must reject out-of-domain pool-local ids
    b = sc._make_toy_bundle()
    _raises(AssertionError, lambda: plast.post_ltp(b, -1),
            "post_ltp rejects negative local")
    _raises(AssertionError, lambda: plast.post_ltp(b, 100),
            "post_ltp rejects local >= N_POOL")
    # input_channel_stdp rejects non-input gids
    b = sc._make_toy_bundle()
    _raises(AssertionError, lambda: plast.input_channel_stdp(b, [4]),
            "input_channel_stdp rejects pool gid")
    # post_ltp clips at w_hi (toy edge0: input0->pool0, learnable)
    b = sc._make_toy_bundle()
    b.trace[0] = 1.0
    b.csr_w[0] = b.cfg.w_hi - 1e-3
    plast.post_ltp(b, 0)
    assert np.isclose(b.csr_w[0], b.cfg.w_hi, atol=1e-12), b.csr_w[0]
    _ok("post_ltp clips at w_hi")
    # pre_ltd multiplicative LTD clips at w_lo (edge2 pool0->pool1, learnable)
    b = sc._make_toy_bundle()
    b.trace[5] = 1.0               # pool1 (dst of e2) fired recently
    b.csr_w[2] = 0.0201
    plast.pre_ltd(b, 4)            # pool0 fires as pre
    assert np.isclose(b.csr_w[2], b.cfg.w_lo, atol=1e-12), b.csr_w[2]
    _ok("pre_ltd clips at w_lo")
    # homeo: theta hit theta_hi when the firing rate is extreme
    b = sc._make_toy_bundle()
    b.counts[0] = 200000
    plast.homeo_update(b)
    assert b.theta[0] == b.cfg.theta_hi
    assert np.isclose(b.rate_ema[0], 0.1 * 200000 / 0.2)
    _ok("homeo clips theta at theta_hi")
    # homeo: theta clips at theta_lo when rate collapses below target
    b = sc._make_toy_bundle()
    b.theta[0] = 0.51
    plast.homeo_update(b)          # counts == 0 -> ema -> 0
    assert b.theta[0] == b.cfg.theta_lo
    _ok("homeo clips theta at theta_lo")
    # competitive_norm no-ops on an empty learnable index
    b = sc._make_toy_bundle()
    b.in_learn_idx = np.zeros(0, dtype=I8)
    b.in_learn_ptr[:] = 0
    w_before = b.csr_w.copy()
    plast.competitive_norm(b)
    assert np.array_equal(b.csr_w, w_before)
    _ok("competitive_norm empty-index no-op")
    # input_w_mean over the toy's two input edges == 1.0
    b = sc._make_toy_bundle()
    assert np.isclose(plast.input_w_mean(b), 1.0)
    d = plast.plasticity_diagnostics(b)
    assert d["n_learnable"] == 3
    assert 0.0 <= d["capped_ratio"] <= 1.0
    _ok("diagnostics on the toy bundle")


# ---------------------------------------------------------------------------
# encoder -- threshold / boundary / reverse (M2)
# ---------------------------------------------------------------------------


def check_encoder_extra() -> None:
    # strict threshold: I == 0.12 must NOT fire, just above must
    bks = encoder.latency_encode(np.asarray([0.12], dtype=F8))
    assert bks == {}
    bks = encoder.latency_encode(np.asarray([0.12 + 1e-9], dtype=F8))
    assert sorted(bks.keys()) == [37]      # t = 2 + 0.88*40 = 37.2 -> 37
    # out-of-range intensities are clipped: I=2 -> t=2; I=-5 -> silent
    bks = encoder.latency_encode(np.asarray([2.0, 0.5], dtype=F8))
    assert sorted(bks.keys()) == [2, 22]
    bks = encoder.latency_encode(np.asarray([-1.0, 0.5], dtype=F8))
    assert sorted(bks.keys()) == [22]
    # ids_offset shifts the input gids but keeps the schedule
    bks = encoder.latency_encode(np.asarray([1.0, 0.5], dtype=F8),
                                 ids_offset=100)
    assert sorted(bks.keys()) == [2, 22]
    assert [int(x) for x in bks[2][0]] == [100]
    assert [int(x) for x in bks[22][0]] == [101]
    # equal intensities land in one bucket with ascending ids
    bks = encoder.latency_encode(np.asarray([0.5, 0.5], dtype=F8))
    assert sorted(bks.keys()) == [22]
    assert [int(x) for x in bks[22][0]] == [0, 1]
    _ok("latency threshold strictness, clipping, offset, per-bucket order")
    # intensities must be 1-D
    _raises(AssertionError,
            lambda: encoder.latency_encode(np.zeros((2, 3))),
            "latency_encode rejects 2-D intensities")
    # adaptive pooling: torch floor/ceil region semantics at every boundary
    img = np.arange(28 * 28, dtype=F8).reshape(28, 28)
    out = encoder.mnist_adaptive_pool(img, out=10)
    idx = np.arange(10, dtype=F8)
    starts = np.floor(idx * 28 / 10).astype(int)
    ends = np.ceil((idx + 1.0) * 28 / 10).astype(int)
    for r in (0, 4, 9):
        for c in (0, 5, 9):
            exp = float(np.mean(img[starts[r]:ends[r], starts[c]:ends[c]]))
            assert np.isclose(out[r, c], exp), (r, c)
    # all-ones image pools to all-ones
    ones = np.ones((28, 28))
    assert np.allclose(encoder.mnist_adaptive_pool(ones, out=10), 1.0)
    # 1x1 -> 1x1
    assert np.allclose(encoder.mnist_adaptive_pool(np.array([[7.0]]), out=1),
                       [[7.0]])
    _ok("mnist adaptive pooling floor/ceil cell boundaries")
    # pooling rejects non-square input / non-positive out
    _raises(AssertionError,
            lambda: encoder.mnist_adaptive_pool(np.zeros((2, 3))),
            "mnist_adaptive_pool rejects non-square image")
    _raises(AssertionError,
            lambda: encoder.mnist_adaptive_pool(np.zeros((4, 4)), out=0),
            "mnist_adaptive_pool rejects out<=0")
    # mnist_encode requires cfg.n_in == out**2
    _raises(ValueError,
            lambda: encoder.mnist_encode(img, out=10, cfg=cfg_from_mapping(
                {"n_in": 64, "n_input_cols": 8})),
            "mnist_encode rejects cfg.n_in != out**2")
    # full MNIST encode on a 100-input config: white patch -> 100 ids at t=2
    mcfg = cfg_from_mapping({"n_in": 100, "n_input_cols": 10})
    mb = encoder.mnist_encode(np.ones((28, 28)), out=10, cfg=mcfg)
    assert sorted(mb.keys()) == [2]
    ids = mb[2][0]
    assert ids.size == 100
    assert np.array_equal(ids, np.arange(100, dtype=I8))
    _ok("mnist_encode white patch -> all 100 ids at t=2")
    # DVS interface raises NotImplementedError (G3 milestone)
    _raises(NotImplementedError, lambda: encoder.dvs_patch_aggregate(None),
            "dvs_patch_aggregate NotImplementedError")


def main() -> int:
    _run("layout extra (reverse/boundary, G0 #4/#13):", check_layout_extra)
    _run("spatial_hash extra (G0 #10):", check_hash_extra)
    _run("network extra (G0 #12/#13):", check_network_extra)
    _run("kernel extra (G0 #1/#2/#3):", check_kernel_extra)
    _run("plasticity extra (G0 #5/#8/#9):", check_plasticity_extra)
    _run("encoder extra (M2):", check_encoder_extra)
    print(f"\nextra core tests PASSED: {_PASS} assertion groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
