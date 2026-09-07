"""lizhu_extra_m56_tests.py -- supplementary unit tests for hstdn.core M5/M6.

Complement to ``python -m hstdn.core.selfcheck`` (15 groups).  Covers the
reverse / boundary / edge dimensions of the three new components:

- features.py (M5): config defaults & passthrough, unknown-key rejection,
  V0 1-D/2-D inputs, sqrt+L2 equivalence with manual math, all-zero rows stay
  zero (no NaN), feature_mask 'pool'/'all'/id-list/bool variants and invalid
  masks, negative-count / bad-ndim rejection, V1 bin flatten & n_bins
  mismatch, V2 NotImplementedError, unknown-mode ValueError.
- readout.py (M5): config validation, W/b shapes, history keys, explicit vs
  auto 10% split paths, fewer-than-2-classes / non-dense / shifted-dense /
  shape / dtype / NaN rejections, label-range guard for explicit validation,
  determinism under a fixed seed, predict/proba/accuracy semantics, L1/L2
  NotImplementedError stubs, linear-separable convergence.
- scheduler.py (M6): protocol config validation, full small-net G1 flow with
  report structure + JSON-safe to_dict + theta bounds (CALIBRATE FIX-B2),
  multi-round loop with eta-halving consistency (bundle.cfg.eta_ltp vs
  eta_scale_final), rollback consistency (best_acc == max eval acc),
  force-advance with notes when the ADAPT gate is never met (white-box
  _measure patch) and when CALIBRATE fails, RuntimeError on empty data at
  every stage, AssertionError on unsupported frame shapes.

Run from E:\neuron3d (PYTHONPATH set to it):
    python .lizhu_env/lizhu_tests/lizhu_extra_m56_tests.py
Exit 0 on full pass.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from hstdn.core.layout import I8, F8, NetConfig
from hstdn.core.network import build_network
from hstdn.core.features import (
    FeaturesConfig, features_from_cfg, build_features, l2_row_normalize,
)
from hstdn.core import readout as rd
from hstdn.core import scheduler as sched
from hstdn.core.scheduler import (
    ProtocolConfig, protocol_from_cfg, G1Report, run_g1_protocol,
)

_PASS = 0


def _ok(name: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  [PASS] {name}")


def _raises(exc, fn, name: str) -> None:
    try:
        fn()
    except exc:
        _ok(name)
        return
    raise AssertionError(f"{name}: expected {exc.__name__}")


def _run(name: str, fn) -> None:
    print(name)
    fn()


# ---------------------------------------------------------------------------
# features (M5)
# ---------------------------------------------------------------------------


def check_features_extra() -> None:
    # --- defaults / config normalization ---
    d = FeaturesConfig()
    assert d.mode == "v0" and d.feature_mask == "pool" and d.n_bins == 5
    f0 = features_from_cfg(None)
    assert isinstance(f0, FeaturesConfig) and f0.mode == "v0"
    assert features_from_cfg(d) is d          # dataclass passthrough
    m = features_from_cfg({"mode": "v1", "n_bins": 3})
    assert m.mode == "v1" and m.n_bins == 3 and m.feature_mask == "pool"
    _raises(ValueError, lambda: features_from_cfg({"mode": "v0", "bins": 2}),
            "features_from_cfg rejects unknown keys")
    _ok("FeaturesConfig defaults / normalization / unknown-key")
    # --- V0 basic math ---
    rng = np.random.default_rng(21)
    C = rng.integers(0, 9, size=(5, 12)).astype(F8)
    X = build_features(C)
    assert X.shape == (5, 12) and X.dtype == F8
    S = np.sqrt(C)
    norms = np.sqrt(np.einsum("ij,ij->i", S, S))
    for i in range(5):
        assert np.allclose(X[i], S[i] / norms[i], atol=1e-12), i
    # zero (silent) row stays zero, no NaN
    Cz = C.copy()
    Cz[2, :] = 0.0
    Xz = build_features(Cz)
    assert np.linalg.norm(Xz[2]) == 0.0 and np.isfinite(Xz).all()
    # 1-D (N_POOL,) input -> (1, N_POOL)
    X1 = build_features(C[0])
    assert X1.shape == (1, 12) and np.allclose(X1[0], S[0] / norms[0])
    _ok("V0 sqrt+L2 math, zero-row handling, 1-D input")
    # --- feature_mask variants ---
    Xall = build_features(C, {"feature_mask": "all"})
    assert np.array_equal(Xall, X)
    ids = np.array([0, 2, 4], dtype=I8)
    Xid = build_features(C, {"feature_mask": ids})
    assert Xid.shape == (5, 3)
    Si = S[:, [0, 2, 4]]
    ni = np.sqrt(np.einsum("ij,ij->i", Si, Si))
    for i in range(5):
        assert np.allclose(Xid[i], Si[i] / ni[i], atol=1e-12), i
    # list form accepted
    Xlst = build_features(C, {"feature_mask": [1, 3]})
    assert Xlst.shape == (5, 2)
    # explicit bool mask
    bm = np.arange(12) % 2 == 0
    Xb = build_features(C, {"feature_mask": bm})
    assert Xb.shape == (5, 6)
    Sb = S[:, bm]
    nb = np.sqrt(np.einsum("ij,ij->i", Sb, Sb))
    assert np.allclose(Xb[1], Sb[1] / nb[1], atol=1e-12)
    _ok("feature_mask pool/all/id-list/bool variants")
    # --- invalid masks / counts / ndim ---
    _raises(AssertionError,
            lambda: build_features(C, {"feature_mask": np.ones(11, bool)}),
            "bool mask of wrong length rejected")
    _raises(AssertionError,
            lambda: build_features(C, {"feature_mask": np.array([0, 12])}),
            "out-of-range pool-local id mask rejected")
    _raises(AssertionError,
            lambda: build_features(C, {"feature_mask": np.array([-1, 2])}),
            "negative mask id rejected")
    _raises(ValueError,
            lambda: build_features(C, {"feature_mask": "nope"}),
            "unknown string mask rejected")
    Cn = C.copy()
    Cn[0, 1] = -1.0
    _raises(AssertionError, lambda: build_features(Cn),
            "negative spike counts rejected (v0)")
    _raises(AssertionError, lambda: build_features(np.zeros((2, 3, 12))),
            "3-D counts rejected (v0)")
    _raises(AssertionError, lambda: l2_row_normalize(np.zeros(4)),
            "l2_row_normalize rejects 1-D")
    _ok("invalid mask / negative counts / bad ndim rejection")
    # --- V1 ---
    C3 = rng.integers(0, 9, size=(2, 5, 12)).astype(F8)
    Xv1 = build_features(C3, {"mode": "v1", "n_bins": 5})
    assert Xv1.shape == (2, 60)
    assert np.allclose(Xv1[0], np.sqrt(C3[0]).ravel())
    # single sample (n_bins, N_POOL)
    Xs = build_features(C3[0], {"mode": "v1", "n_bins": 5})
    assert Xs.shape == (1, 60)
    # bin count mismatch -> ValueError
    Cbad = rng.integers(0, 9, size=(2, 4, 12)).astype(F8)
    _raises(ValueError, lambda: build_features(Cbad, {"mode": "v1",
                                                      "n_bins": 5}),
            "v1 n_bins mismatch rejected")
    _raises(AssertionError, lambda: build_features(C3[0, 0], {"mode": "v1",
                                                              "n_bins": 5}),
            "v1 1-D input rejected")
    _raises(AssertionError,
            lambda: build_features(np.zeros((2, 5, 12, 1)), {"mode": "v1",
                                                             "n_bins": 5}),
            "v1 4-D input rejected")
    Cneg = C3.copy()
    Cneg[0, 0, 0] = -1
    _raises(AssertionError, lambda: build_features(Cneg, {"mode": "v1",
                                                          "n_bins": 5}),
            "v1 negative counts rejected")
    _ok("V1 bin flatten / n_bins mismatch / ndim guards")
    # --- modes ---
    _raises(NotImplementedError, lambda: build_features(C, {"mode": "v2"}),
            "features v2 NotImplementedError")
    _raises(ValueError, lambda: build_features(C, {"mode": "v9"}),
            "unknown features mode ValueError")
    _ok("v2 NotImplementedError / unknown mode ValueError")

# ---------------------------------------------------------------------------
# readout (M5)
# ---------------------------------------------------------------------------


def check_readout_extra() -> None:
    d = rd.ReadoutConfig()
    assert d.lr == 0.5 and d.iters == 300 and d.l2_lambda == 1e-4
    assert d.batch == 64 and d.val_frac == 0.1 and d.w_scale == 1e-3
    assert rd.readout_from_cfg(None) == d
    assert rd.readout_from_cfg(d) is d
    mc = rd.readout_from_cfg({"iters": 5, "lr": 0.1})
    assert mc.iters == 5 and mc.lr == 0.1 and mc.batch == 64
    _raises(ValueError, lambda: rd.readout_from_cfg({"iterz": 3}),
            "readout_from_cfg rejects unknown keys")
    _ok("ReadoutConfig defaults / normalization / unknown-key")
    # --- separable convergence, explicit validation split ---
    rng = np.random.default_rng(1)
    xc = np.concatenate([rng.normal(-2.0, 0.4, (150, 16)),
                         rng.normal(2.0, 0.4, (150, 16))])
    yc = np.concatenate([np.zeros(150, dtype=I8), np.ones(150, dtype=I8)])
    perm = rng.permutation(300)
    Xtr, ytr = xc[perm[:240]], yc[perm[:240]]
    Xva, yva = xc[perm[240:]], yc[perm[240:]]
    W, b, hist = rd.train_linear_readout(Xtr, ytr, Xva, yva)
    assert W.shape == (16, 2) and b.shape == (2,)
    acc = rd.accuracy(yva, rd.predict_linear_readout(Xva, W, b))
    assert acc > 0.95, f"separable convergence failed: {acc:.3f}"
    for k in ("train_acc", "val_acc", "loss", "best_iter", "best_val_acc",
              "final_val_acc", "n_train", "n_val", "d", "n_class", "lr",
              "l2_lambda", "batch"):
        assert k in hist, k
    assert len(hist["train_acc"]) == 300 and len(hist["loss"]) == 300
    assert 0.0 <= hist["best_val_acc"] <= 1.0
    assert hist["n_train"] == 240 and hist["n_val"] == 60
    assert hist["d"] == 16 and hist["n_class"] == 2
    # auto 10% split path
    W2, b2, hist2 = rd.train_linear_readout(Xtr, ytr)
    assert hist2["n_val"] >= 1
    assert hist2["n_train"] + hist2["n_val"] == 240
    _ok("separable convergence + explicit/auto validation paths + history")
    # --- determinism under a fixed seed ---
    cfgd = {"iters": 6, "batch": 32, "seed": 3}
    Wa, ba, ha = rd.train_linear_readout(Xtr, ytr, Xva, yva, cfgd)
    Wb, bb, hb = rd.train_linear_readout(Xtr, ytr, Xva, yva, cfgd)
    assert np.array_equal(Wa, Wb) and np.array_equal(ba, bb)
    assert ha["loss"] == hb["loss"]
    _ok("readout training deterministic under fixed seed")
    # --- reverse / boundary rejections ---
    y1 = np.zeros(50, dtype=I8)
    _raises(AssertionError, lambda: rd.train_linear_readout(
        np.zeros((50, 4)), y1),
        "fewer than 2 classes rejected")
    y_sparse = np.where(np.arange(60) % 2 == 0, 0, 2).astype(I8)
    _raises(AssertionError, lambda: rd.train_linear_readout(
        np.zeros((60, 4)), y_sparse),
        "non-dense labels rejected")
    y_shift = (np.arange(60) % 3 + 1).astype(I8)   # 1,2,3 not starting at 0
    _raises(AssertionError, lambda: rd.train_linear_readout(
        np.zeros((60, 4)), y_shift),
        "shifted-dense labels rejected")
    _raises(AssertionError, lambda: rd.train_linear_readout(
        np.zeros(60), (np.arange(60) % 2).astype(I8)),
        "1-D X rejected")
    _raises(AssertionError, lambda: rd.train_linear_readout(
        np.zeros((60, 4)), np.zeros((30, 1))),
        "2-D y rejected")
    _raises(AssertionError, lambda: rd.train_linear_readout(
        np.zeros((60, 4)), np.arange(30, dtype=I8)),
        "y length mismatch rejected")
    _raises(AssertionError, lambda: rd.train_linear_readout(
        np.zeros((60, 4)), np.arange(60, dtype=F8)),
        "float labels rejected")
    Xnan = np.zeros((60, 4))
    Xnan[0, 0] = np.nan
    _raises(AssertionError, lambda: rd.train_linear_readout(
        Xnan, (np.arange(60) % 2).astype(I8)),
        "NaN X rejected")
    # explicit validation labels outside the class range
    _raises(AssertionError, lambda: rd.train_linear_readout(
        Xtr, ytr, Xva, np.full(60, 5, dtype=I8)),
        "validation labels outside class range rejected")
    _ok("readout reverse/boundary rejections")
    # --- predict / proba / accuracy semantics ---
    P = rd.proba_linear_readout(Xva, W, b)
    assert P.shape == (60, 2)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
    pred = rd.predict_linear_readout(Xva, W, b)
    assert pred.dtype == I8 and np.array_equal(pred, P.argmax(axis=1))
    assert 0.0 <= rd.accuracy(yva, pred) <= 1.0
    assert rd.accuracy(np.zeros(0, dtype=I8), np.zeros(0, dtype=I8)) == 0.0
    _raises(AssertionError, lambda: rd.proba_linear_readout(
        np.zeros((5, 3)), W, b),
        "proba rejects W/X dimension mismatch")
    _raises(AssertionError, lambda: rd.accuracy(yva, pred[:-1]),
            "accuracy rejects length mismatch")
    _ok("predict/proba/accuracy semantics")
    # --- L1/L2 stubs ---
    _raises(NotImplementedError, lambda: rd.train_pytorch_readout(None),
            "train_pytorch_readout NotImplementedError")
    _raises(NotImplementedError, lambda: rd.train_v2_readout(None),
            "train_v2_readout NotImplementedError")
    _ok("readout L1/L2 interface stubs raise")


# ---------------------------------------------------------------------------
# scheduler (M6) helpers
# ---------------------------------------------------------------------------

SCHED_NET = NetConfig(n_in=64, n_pool=200, n_input_cols=8).with_derived()
PC_BASE = dict(t_ms=60, n_adapt_epochs=1, adapt_diag_samples=2,
               adapt_gate_extra_max=2, calibrate_samples=3,
               calibrate_max_iter=2)


def _make_train_test(n_tr: int, n_te: int, seed: int):
    """Deterministic (per seed) frame generators, 3 classes."""
    rng = np.random.default_rng(seed)

    def base_frames():
        # one fresh deterministic stream per data_fn invocation;
        # round-robin over classes so every block of 3 has one of each
        local = np.random.default_rng(seed + 1000)
        for _ in range(64):
            for c in range(3):
                base = 0.35 + 0.15 * c
                yield (np.clip(base + local.normal(0.0, 0.2, (8, 8)),
                               0.0, 1.0), c)

    def train_fn():
        it = iter(base_frames())
        n = 0
        for frame, c in it:
            if n >= n_tr * 3:
                break
            yield frame, c
            n += 1

    def test_fn():
        it = iter(base_frames())
        n = 0
        for frame, c in it:
            if n >= n_te * 3:
                break
            yield frame, c
            n += 1

    return train_fn, test_fn

def _build_bundle():
    return build_network(SCHED_NET)


def _protocol_cfg(**kw) -> ProtocolConfig:
    base = dict(PC_BASE)
    base.update(kw)
    return ProtocolConfig(**base)


def check_scheduler_config() -> None:
    d = ProtocolConfig()
    assert d.t_ms == 200 and d.n_adapt_epochs == 2
    assert d.calibrate_samples == 10 and d.calibrate_max_iter == 2
    assert d.extra_loops == 2 and d.eta_scale == 0.5
    assert protocol_from_cfg(None) == d
    assert protocol_from_cfg(d) is d
    pc = protocol_from_cfg({"t_ms": 60, "extra_loops": 0})
    assert pc.t_ms == 60 and pc.extra_loops == 0
    _raises(ValueError, lambda: protocol_from_cfg({"t_m": 60}),
            "protocol_from_cfg rejects unknown keys")
    _ok("ProtocolConfig defaults / normalization / unknown-key")


def check_scheduler_flow() -> None:
    bundle = _build_bundle()
    train_fn, test_fn = _make_train_test(8, 4, seed=5)
    pc = _protocol_cfg(extra_loops=0, seed=0)
    rep = run_g1_protocol(bundle, train_fn, test_fn, pc, seed=1)
    assert isinstance(rep, G1Report)
    assert rep.ok and rep.n_rounds == 1
    seq = rep.stage_sequence
    assert seq[:3] == ["ADAPT", "CALIBRATE", "COLLECT"]
    assert "READOUT" in seq and "EVAL" in seq
    assert rep.collect_n == 24
    assert rep.feat_dim == 200 and rep.n_class == 3
    assert rep.readout["n_class"] == 3
    assert rep.eval_rounds and 0.0 <= rep.eval_rounds[-1]["acc"] <= 1.0
    ev = rep.eval_rounds[-1]
    assert ev["mean_rate_hz"] >= 0.0
    assert 0.0 <= ev["silence_frac"] <= 1.0
    assert 0.0 <= ev["capped_ratio"] <= 1.0
    assert -1.0 <= ev["cos_train_test"] <= 1.0
    assert 0.0 <= rep.best_acc <= 1.0
    # CALIBRATE FIX-B2 keeps theta inside [0.5, 20]
    assert np.all(bundle.theta >= 0.5 - 1e-9)
    assert np.all(bundle.theta <= 20.0 + 1e-9)
    # single round -> no eta halving
    assert rep.eta_scale_final == 1.0
    assert np.isclose(bundle.cfg.eta_ltp, 0.01)
    assert np.isclose(bundle.cfg.homeo_theta_lr, 0.05)
    # report dict is JSON-safe
    d = rep.to_dict()
    for k in ("ok", "n_rounds", "stage_sequence", "adapt_rounds",
              "calibrate_rounds", "eval_rounds", "collect_n", "feat_dim",
              "n_class", "readout", "eta_scale_final", "rolled_back",
              "best_acc", "notes"):
        assert k in d
    json.dumps(d)  # must not raise
    _ok("G1 full flow: report structure, theta bounds, JSON-safe to_dict")


def check_scheduler_eta_rounds() -> None:
    for seed in (5, 6, 7):
        bundle = _build_bundle()
        train_fn, test_fn = _make_train_test(8, 4, seed=seed)
        pc = _protocol_cfg(extra_loops=1, seed=seed)
        rep = run_g1_protocol(bundle, train_fn, test_fn, pc, seed=seed)
        assert rep.ok
        assert 1 <= rep.n_rounds <= 1 + pc.extra_loops
        # halvings = number of ADAPT re-entries after the first round
        adapt_stages = [s for s in rep.stage_sequence
                        if s.startswith("ADAPT")]
        halvings = len(adapt_stages) - 1
        assert halvings >= 0 and halvings <= 1
        expect_scale = 0.5 ** halvings
        assert np.isclose(rep.eta_scale_final, expect_scale), \
            (rep.eta_scale_final, expect_scale, rep.stage_sequence)
        # eta on bundle.cfg must have been multiplied by eta_scale^halvings
        assert np.isclose(bundle.cfg.eta_ltp, 0.01 * expect_scale)
        assert np.isclose(bundle.cfg.eta_ltd, 0.01 * expect_scale)
        assert np.isclose(bundle.cfg.homeo_theta_lr, 0.05 * expect_scale)
        # monotonic best_acc == max of recorded eval accuracies
        accs = [e["acc"] for e in rep.eval_rounds]
        assert accs
        assert np.isclose(rep.best_acc, max(accs), atol=1e-12)
        if rep.rolled_back:
            assert accs[-1] < rep.best_acc - 1e-12
        assert np.all(bundle.theta >= 0.5 - 1e-9)
        assert np.all(bundle.theta <= 20.0 + 1e-9)
    _ok("multi-round eta halving semantics + rollback consistency (3 seeds)")


def check_scheduler_force_advance() -> None:
    """ADAPT gate never met -> force-advance with recorded notes, no raise."""
    bundle = _build_bundle()
    train_fn, test_fn = _make_train_test(8, 4, seed=9)
    pc = _protocol_cfg(extra_loops=0, seed=0,
                       adapt_gate_extra_max=1)
    orig_measure = sched._measure

    def fake_measure(data_fn, bundle, n, t_ms):
        return {"mean_rate_hz": 0.0, "silence_frac": 1.0, "n_used": n,
                "rate_per_pool_hz": np.zeros(bundle.n_pool, dtype=F8)}

    sched._measure = fake_measure
    try:
        rep = run_g1_protocol(bundle, train_fn, test_fn, pc, seed=1)
    finally:
        sched._measure = orig_measure
    assert rep.ok
    assert rep.adapt_rounds[0]["forced"] is True
    assert any("ADAPT gate unmet" in n for n in rep.notes), rep.notes
    assert np.all(bundle.theta >= 0.5 - 1e-9)
    assert np.all(bundle.theta <= 20.0 + 1e-9)
    _ok("ADAPT gate unmet: force-advance + notes, theta bounds kept")


def check_scheduler_empty_data() -> None:
    # stage-level: each helper raises RuntimeError on an empty iterator
    bundle = _build_bundle()
    pc = _protocol_cfg(extra_loops=0)
    empty = lambda: iter(())
    _raises(RuntimeError, lambda: sched._measure(empty, bundle, 5, 60),
            "_measure raises on empty data_fn")
    _raises(RuntimeError, lambda: sched._run_adapt_epoch(bundle, empty, 60),
            "adapt epoch raises on empty data_fn")
    _raises(RuntimeError, lambda: sched._collect(empty, bundle, pc),
            "COLLECT raises on empty data_fn")
    _raises(RuntimeError,
            lambda: sched._eval_round(bundle, empty, pc,
                                      np.zeros((1, 1)), np.zeros(1),
                                      np.zeros((1, 1))),
            "EVAL raises on empty data_fn")
    # protocol-level: empty train data -> RuntimeError (ADAPT stage)
    bundle2 = _build_bundle()
    _, test_fn = _make_train_test(8, 4, seed=3)
    _raises(RuntimeError,
            lambda: run_g1_protocol(bundle2, empty, test_fn,
                                    _protocol_cfg(extra_loops=0), seed=1),
            "run_g1_protocol raises RuntimeError on empty train data")
    # protocol-level: empty test data -> RuntimeError (EVAL stage)
    bundle3 = _build_bundle()
    train_fn, _ = _make_train_test(8, 4, seed=3)
    _raises(RuntimeError,
            lambda: run_g1_protocol(bundle3, train_fn, empty,
                                    _protocol_cfg(extra_loops=0), seed=1),
            "run_g1_protocol raises RuntimeError on empty test data")
    _ok("empty data_fn raises RuntimeError at every stage")


def check_scheduler_frames() -> None:
    bundle = _build_bundle()
    # private encoder guard
    good = sched._encode_frame(np.zeros(64), bundle)
    assert isinstance(good, dict)
    good2 = sched._encode_frame(np.zeros((8, 8)), bundle)
    assert isinstance(good2, dict)
    _raises(AssertionError, lambda: sched._encode_frame(np.zeros(63), bundle),
            "1-D frame with wrong length rejected")
    _raises(AssertionError,
            lambda: sched._encode_frame(np.zeros((4, 4, 4)), bundle),
            "3-D frame rejected (element count == N_IN)")
    _raises(AssertionError,
            lambda: sched._encode_frame(np.zeros((8, 7)), bundle),
            "2-D frame with wrong element count rejected")
    # protocol-level: unsupported frames surface as AssertionError
    bundle2 = _build_bundle()

    def bad_train():
        for c in range(3):
            yield np.zeros((4, 4, 4)), c

    _, test_fn = _make_train_test(8, 4, seed=4)
    _raises(AssertionError,
            lambda: run_g1_protocol(bundle2, bad_train, test_fn,
                                    _protocol_cfg(extra_loops=0), seed=1),
            "run_g1_protocol surfaces AssertionError on unsupported frames")
    _ok("unsupported frame shapes raise AssertionError")


def main() -> int:
    _run("features extra (M5):", check_features_extra)
    _run("readout extra (M5):", check_readout_extra)
    _run("scheduler config (M6):", check_scheduler_config)
    _run("scheduler full flow (M6):", check_scheduler_flow)
    _run("scheduler eta/rounds (M6):", check_scheduler_eta_rounds)
    _run("scheduler force-advance (M6):", check_scheduler_force_advance)
    _run("scheduler empty-data (M6):", check_scheduler_empty_data)
    _run("scheduler frame guards (M6):", check_scheduler_frames)
    print(f"\nextra M5/M6 tests PASSED: {_PASS} assertion groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
