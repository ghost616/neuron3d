"""scheduler.py -- M6 training-loop state machine (G1 protocol driver).

Strict state flow of the design-doc M6 state diagram:

    ADAPT -> CALIBRATE -> COLLECT -> READOUT -> EVAL   (optional loops <= 2)

- ADAPT: plasticity on (stdp_on/homeo_on/norm_on=True) for ``n_adapt_epochs``
  full passes over the training iterator; gate diagnostics after each pass
  (all-off measurements): mean firing rate in [3, 15] Hz and silent pool
  fraction < 5% are required.  While the gate is unmet, up to
  ``adapt_gate_extra_max`` extra single epochs run; beyond that the protocol
  force-advances with a recorded warning.
- CALIBRATE: all plasticity off; ``calibrate_samples`` inference samples give
  the per-pool mean rate; uniform negative feedback per FIX-B2:
  ``factor = clip(measured_rate / 8, 1/3, 3)`` and
  ``theta = clip(theta * factor, 0.5, 20)``; the band [3, 12] Hz must be met
  within ``calibrate_max_iter`` (2) iterations.  On failure the protocol
  returns to ADAPT with eta_stdp/eta_homeo halved (recorded) -- when no loop
  budget remains it force-advances with a warning.
- COLLECT: all off; pure inference over the full training iterator; features
  built with core.features.build_features (V0 default).
- READOUT: all off; core.readout.train_linear_readout (10% validation early
  stop).
- EVAL: all off; test inference; accuracy + rate / silence / capped-ratio /
  train-test cosine diagnostics (inline minimal stats; exp.diagnostics is
  preferred once available).
- Loops: at most ``extra_loops`` (2) further rounds re-entering ADAPT with
  eta_stdp x0.5 and eta_homeo x0.5 each round; a per-round checkpoint is kept
  and when a later round accuracy does not improve the previous best the
  protocol rolls the bundle back to that checkpoint.

Samples are fed one at a time (buckets via core.encoder.latency_encode) so
memory stays bounded.  Per-sample-end homeo/norm are handled inside
run_sample through its stdp_on/homeo_on/norm_on switches (existing kernel
convention) -- the scheduler only opens/closes those switches; it never
re-implements them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

from hstdn.core.layout import I8, F8, NetConfig
from hstdn.core.network import NetworkBundle
from hstdn.core import encoder as _encoder
from hstdn.core.features import FeaturesConfig, build_features
from hstdn.core.readout import ReadoutConfig, train_linear_readout, predict_linear_readout, accuracy
from hstdn.core import plasticity as _plast

__all__ = ["ProtocolConfig", "protocol_from_cfg", "G1Report",
           "run_g1_protocol"]

FrameLabel = Tuple[Any, int]
DataFn = Callable[[], Iterable[FrameLabel]]


@dataclass(frozen=True)
class ProtocolConfig:
    """Protocol (M6) hyper-parameters (design-doc protocol section).

    Timing / plasticity switches, gate bands and loop budgets.
    """

    t_ms: int = 200                # sample length in ms (matches kernel default)
    # --- ADAPT ---
    n_adapt_epochs: int = 2        # full training passes per ADAPT round
    adapt_diag_samples: int = 5    # all-off samples per gate measurement
    adapt_gate_extra_max: int = 3  # extra single epochs allowed before forcing
    silence_frac_max: float = 0.05  # gate: silent-pool fraction < 5%
    adapt_rate_lo: float = 3.0      # gate: mean rate band [3, 15] Hz
    adapt_rate_hi: float = 15.0
    # --- CALIBRATE (FIX-B2) ---
    calibrate_samples: int = 10
    calibrate_max_iter: int = 2
    calib_rate_lo: float = 3.0      # pass band [3, 12] Hz
    calib_rate_hi: float = 12.0
    rate_target: float = 8.0
    factor_min: float = 1.0 / 3.0
    factor_max: float = 3.0
    theta_lo: float = 0.5
    theta_hi: float = 20.0
    # --- loops & eta decay ---
    extra_loops: int = 2
    eta_scale: float = 0.5          # eta_stdp & eta_homeo multiplier per round
    # --- features (delegated to core.features) ---
    feat_mode: str = "v0"
    feat_n_bins: int = 5
    # --- randomness ---
    seed: int = 0


def protocol_from_cfg(cfg: Union[ProtocolConfig, Mapping[str, Any], None]
                      ) -> ProtocolConfig:
    """Normalize a protocol config (None | ProtocolConfig | mapping)."""
    if cfg is None or isinstance(cfg, ProtocolConfig):
        return cfg if cfg is not None else ProtocolConfig()
    known = set(ProtocolConfig.__dataclass_fields__)
    extra = set(cfg) - known
    if extra:
        raise ValueError(f"unknown protocol config keys: {sorted(extra)}")
    return replace(ProtocolConfig(), **{k: v for k, v in cfg.items() if k in known})


# ---------------------------------------------------------------------------
# frame encoding helpers
# ---------------------------------------------------------------------------


def _encode_frame(frame: Any, bundle: NetworkBundle) -> dict:
    """Latency-encode one input frame into buckets for the kernel.

    Supported frames: 1-D intensities of length N_IN, or any 2-D layout whose
    element count equals N_IN (flattened row-major; e.g. a (16,16) grid for
    N_IN=256).  Poisson coding stays off (cfg.poisson_on=False).

    Args:
        frame: intensity frame (values in [0, 1]).
        bundle: NetworkBundle (defines N_IN and the encoder config).

    Returns:
        BucketMap consumed by run_sample.

    Raises:
        AssertionError: unsupported frame shape.
    """
    arr = np.asarray(frame, dtype=F8)
    n_in = bundle.cfg.n_in
    if arr.ndim == 1:
        if arr.size != n_in:
            raise AssertionError(
                f"frame 1-D length {arr.size} != N_IN={n_in}"
            )
        inten = arr
    elif arr.ndim == 2:
        if arr.size != n_in:
            raise AssertionError(
                f"frame 2-D element count {arr.size} != N_IN={n_in}"
            )
        inten = arr.ravel()
    else:
        raise AssertionError(
            f"unsupported frame ndim {arr.ndim} (need 1-D or 2-D)"
        )
    return _encoder.latency_encode(inten, cfg=bundle.cfg)


# ---------------------------------------------------------------------------
# inference measurements (all plasticity off)
# ---------------------------------------------------------------------------


def _measure(data_fn: DataFn, bundle: NetworkBundle, n: int, t_ms: int
             ) -> Dict[str, Any]:
    """Run ``n`` all-off samples and aggregate per-pool spike statistics.

    Args:
        data_fn: callable returning an iterator of (frame, label).
        bundle: NetworkBundle.
        n: number of samples to consume.
        t_ms: sample length in ms.

    Returns:
        dict: mean_rate_hz (mean over pools of per-second rate), silence_frac
        (pools with zero spikes across the measurement), n_used.
    """
    cfg = bundle.cfg
    total = np.zeros(cfg.n_pool, dtype=F8)
    used = 0
    for frame, _lab in data_fn():
        if used >= n:
            break
        buckets = _encode_frame(frame, bundle)
        run_stats = _run_sample_quiet(bundle, buckets, t_ms)
        total += run_stats["counts_f8"]
        used += 1
    if used == 0:
        raise RuntimeError("measurement data_fn produced no samples")
    avg = total / used
    rate = avg / (t_ms * 1e-3)
    return {
        "mean_rate_hz": float(rate.mean()),
        "silence_frac": float(np.count_nonzero(total == 0.0) / cfg.n_pool),
        "n_used": used,
        "rate_per_pool_hz": rate,
    }


def _run_sample_quiet(bundle: NetworkBundle, buckets: dict, t_ms: int) -> Dict[str, Any]:
    """run_sample with all plasticity off; counts as float64."""
    # local import keeps scheduler.py independent of kernel's public surface
    from hstdn.core.kernel import run_sample
    st = run_sample(bundle, buckets, T=t_ms, stdp_on=False,
                    homeo_on=False, norm_on=False)
    return {"counts_f8": st["spike_counts"].astype(F8),
            "n_spikes": st["n_spikes_total"]}
# ---------------------------------------------------------------------------
# ADAPT / CALIBRATE stages
# ---------------------------------------------------------------------------


def _run_adapt_epoch(bundle: NetworkBundle, data_fn: DataFn, t_ms: int
                     ) -> Dict[str, Any]:
    """One full ADAPT pass: plasticity on for every sample."""
    cfg = bundle.cfg
    from hstdn.core.kernel import run_sample
    n = 0
    spikes = 0
    for frame, _lab in data_fn():
        buckets = _encode_frame(frame, bundle)
        st = run_sample(bundle, buckets, T=t_ms, stdp_on=True,
                        homeo_on=True, norm_on=True)
        n += 1
        spikes += st["n_spikes_total"]
    if n == 0:
        raise RuntimeError("ADAPT data_fn produced no samples")
    return {"n_samples": n, "n_spikes": spikes}


def _adapt_round(bundle: NetworkBundle, data_fn: DataFn, pcfg: ProtocolConfig
                 ) -> Dict[str, Any]:
    """ADAPT stage: adapt epochs + gate diagnostics (may push extra epochs).

    Returns:
        dict: passed (bool gate), forced (bool advanced beyond extra budget),
        mean_rate_hz, silence_frac, n_epochs_total, diag_samples.
    """
    t = pcfg.t_ms
    epochs = 0
    for _ in range(pcfg.n_adapt_epochs):
        _run_adapt_epoch(bundle, data_fn, t)
        epochs += 1
    extra = 0
    while True:
        diag = _measure(data_fn, bundle, pcfg.adapt_diag_samples, t)
        rate = diag["mean_rate_hz"]
        sil = diag["silence_frac"]
        passed = (pcfg.adapt_rate_lo <= rate <= pcfg.adapt_rate_hi
                  and sil < pcfg.silence_frac_max)
        if passed or extra >= pcfg.adapt_gate_extra_max:
            return {
                "passed": bool(passed),
                "forced": bool(not passed and extra >= pcfg.adapt_gate_extra_max),
                "mean_rate_hz": rate,
                "silence_frac": sil,
                "n_epochs_total": epochs,
                "diag_samples": diag["n_used"],
            }
        _run_adapt_epoch(bundle, data_fn, t)
        epochs += 1
        extra += 1


def _calibrate(bundle: NetworkBundle, data_fn: DataFn, pcfg: ProtocolConfig
               ) -> Dict[str, Any]:
    """CALIBRATE stage: uniform negative feedback (FIX-B2).

    Each iteration measures the per-pool mean rate over ``calibrate_samples``
    (all off), then applies per pool
    ``theta *= clip(rate_i / rate_target, factor_min, factor_max)`` clipped
    into [theta_lo, theta_hi].  Passes when the pool-mean rate falls inside
    [calib_rate_lo, calib_rate_hi].

    Returns:
        dict: ok (bool), iterations (int), mean_rate_hz history (list),
        final_mean_rate_hz, samples_per_iter.
    """
    rates_hist: List[float] = []
    ok = False
    iters = 0
    for it in range(pcfg.calibrate_max_iter):
        diag = _measure(data_fn, bundle, pcfg.calibrate_samples, pcfg.t_ms)
        rate = diag["mean_rate_hz"]
        rates_hist.append(rate)
        iters = it + 1
        if pcfg.calib_rate_lo <= rate <= pcfg.calib_rate_hi:
            ok = True
            break
        rate_pool = diag["rate_per_pool_hz"]
        factor = np.clip(
            rate_pool / pcfg.rate_target, pcfg.factor_min, pcfg.factor_max
        )
        bundle.theta[:] = np.clip(
            bundle.theta * factor, pcfg.theta_lo, pcfg.theta_hi
        )
    return {
        "ok": bool(ok),
        "iterations": iters,
        "mean_rate_hz_history": rates_hist,
        "final_mean_rate_hz": rates_hist[-1] if rates_hist else 0.0,
        "samples_per_iter": pcfg.calibrate_samples,
    }


# ---------------------------------------------------------------------------
# COLLECT / READOUT / EVAL stages
# ---------------------------------------------------------------------------


def _collect(data_fn: DataFn, bundle: NetworkBundle, pcfg: ProtocolConfig
             ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """COLLECT: all-off inference over the training iterator; build features.

    Returns:
        (X, y, info): X (n, feat_dim) f8, y (n,) i8 labels, info dict with
        n_samples and n_spikes_total.
    """
    cfg = bundle.cfg
    rows: List[np.ndarray] = []
    labels: List[int] = []
    spikes = 0
    for frame, lab in data_fn():
        buckets = _encode_frame(frame, bundle)
        st = _run_sample_quiet(bundle, buckets, pcfg.t_ms)
        rows.append(st["counts_f8"])
        labels.append(int(lab))
        spikes += st["n_spikes"]
    if not rows:
        raise RuntimeError("COLLECT data_fn produced no samples")
    counts = np.stack(rows, axis=0)
    y = np.asarray(labels, dtype=I8)
    fcfg = FeaturesConfig(mode=pcfg.feat_mode, n_bins=pcfg.feat_n_bins)
    X = build_features(counts, fcfg)
    return X, y, {"n_samples": int(counts.shape[0]),
                  "n_spikes_total": spikes, "counts": counts}


def _readout(X_train: np.ndarray, y_train: np.ndarray, pcfg: ProtocolConfig,
             seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """READOUT stage: L0 softmax linear readout with 10% validation early stop."""
    rcfg = ReadoutConfig(seed=seed)
    W, b, hist = train_linear_readout(X_train, y_train, None, None, rcfg)
    info = {
        "best_iter": hist["best_iter"],
        "best_val_acc": hist["best_val_acc"],
        "final_val_acc": hist["final_val_acc"],
        "d": hist["d"],
        "n_class": hist["n_class"],
        "n_train": hist["n_train"],
    }
    return W, b, info


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _eval_round(bundle: NetworkBundle, data_fn: DataFn, pcfg: ProtocolConfig,
                W: np.ndarray, b: np.ndarray, X_train: np.ndarray
                ) -> Dict[str, Any]:
    """EVAL: all-off test inference + accuracy and firing diagnostics."""
    rows: List[np.ndarray] = []
    labels: List[int] = []
    spikes = 0
    for frame, lab in data_fn():
        buckets = _encode_frame(frame, bundle)
        st = _run_sample_quiet(bundle, buckets, pcfg.t_ms)
        rows.append(st["counts_f8"])
        labels.append(int(lab))
        spikes += st["n_spikes"]
    if not rows:
        raise RuntimeError("EVAL data_fn produced no samples")
    counts = np.stack(rows, axis=0)
    y = np.asarray(labels, dtype=I8)
    fcfg = FeaturesConfig(mode=pcfg.feat_mode, n_bins=pcfg.feat_n_bins)
    X_test = build_features(counts, fcfg)
    pred = predict_linear_readout(X_test, W, b)
    acc = accuracy(y, pred)
    mean_counts = counts.mean(axis=0)
    rate = mean_counts / (pcfg.t_ms * 1e-3)
    capped = _plast.capped_ratio(bundle)
    cos_tt = _cosine(X_train.mean(axis=0), X_test.mean(axis=0))
    return {
        "acc": acc,
        "mean_rate_hz": float(rate.mean()),
        "silence_frac": float(np.count_nonzero(mean_counts == 0.0)
                              / bundle.n_pool),
        "capped_ratio": capped,
        "cos_train_test": cos_tt,
        "n_test": int(counts.shape[0]),
        "n_spikes_total": spikes,
    }


# ---------------------------------------------------------------------------
# checkpoints & report
# ---------------------------------------------------------------------------


@dataclass
class _Checkpoint:
    """Bundle/readout snapshot taken after each completed EVAL round."""

    theta: np.ndarray
    rate_ema: np.ndarray
    csr_w: np.ndarray
    W: np.ndarray
    b: np.ndarray
    acc: float


def _take_checkpoint(bundle: NetworkBundle, W: np.ndarray, b: np.ndarray,
                     acc: float) -> _Checkpoint:
    return _Checkpoint(theta=bundle.theta.copy(),
                       rate_ema=bundle.rate_ema.copy(),
                       csr_w=bundle.csr_w.copy(),
                       W=W.copy(), b=b.copy(), acc=float(acc))


def _restore_checkpoint(bundle: NetworkBundle, ck: _Checkpoint) -> None:
    bundle.theta[:] = ck.theta
    bundle.rate_ema[:] = ck.rate_ema
    bundle.csr_w[:] = ck.csr_w


@dataclass
class G1Report:
    """Full protocol report (scalars/strings only; safe to serialize)."""

    ok: bool
    n_rounds: int
    stage_sequence: List[str] = field(default_factory=list)
    adapt_rounds: List[Dict[str, Any]] = field(default_factory=list)
    calibrate_rounds: List[Dict[str, Any]] = field(default_factory=list)
    eval_rounds: List[Dict[str, Any]] = field(default_factory=list)
    collect_n: int = 0
    feat_dim: int = 0
    n_class: int = 0
    readout: Dict[str, Any] = field(default_factory=dict)
    eta_scale_final: float = 1.0
    rolled_back: bool = False
    best_acc: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view (JSON-safe)."""
        return {
            "ok": self.ok,
            "n_rounds": self.n_rounds,
            "stage_sequence": list(self.stage_sequence),
            "adapt_rounds": list(self.adapt_rounds),
            "calibrate_rounds": list(self.calibrate_rounds),
            "eval_rounds": list(self.eval_rounds),
            "collect_n": self.collect_n,
            "feat_dim": self.feat_dim,
            "n_class": self.n_class,
            "readout": dict(self.readout),
            "eta_scale_final": self.eta_scale_final,
            "rolled_back": self.rolled_back,
            "best_acc": self.best_acc,
            "notes": list(self.notes),
        }


def _halve_eta(bundle: NetworkBundle, pcfg: ProtocolConfig) -> float:
    """Halve eta_stdp (eta_ltp/eta_ltd) and eta_homeo (homeo_theta_lr)."""
    scale = pcfg.eta_scale
    old = bundle.cfg
    new_cfg = replace(
        old,
        eta_ltp=old.eta_ltp * scale,
        eta_ltd=old.eta_ltd * scale,
        homeo_theta_lr=old.homeo_theta_lr * scale,
    )
    bundle.cfg = new_cfg  # NetworkBundle.cfg is a mutable attribute slot
    return float(scale)


def run_g1_protocol(bundle: NetworkBundle, train_data_fn: DataFn,
                    test_data_fn: DataFn,
                    cfg: Union[ProtocolConfig, Mapping[str, Any], None] = None,
                    seed: Optional[int] = None) -> G1Report:
    """Run the full G1 protocol state machine on ``bundle`` (mutated in place).

    Args:
        bundle: NetworkBundle built with build_network.  Weights/theta/rate_ema
            are adapted in place; per-round checkpoints allow rollback.
        train_data_fn: zero-arg callable returning a fresh iterator of
            (frame, label) training pairs (called once per pass).
        test_data_fn: zero-arg callable returning a fresh iterator of
            (frame, label) test pairs.
        cfg: None | ProtocolConfig | mapping (protocol section).
        seed: optional seed override (data/readout randomness).

    Returns:
        G1Report with per-stage statistics, readout/eval results and the
        rollback flag.

    Raises:
        RuntimeError: an iterator produced no samples where one was required.
    """
    pcfg = protocol_from_cfg(cfg)
    if seed is None:
        seed = pcfg.seed
    rep = G1Report(ok=False, n_rounds=0)
    max_rounds = 1 + pcfg.extra_loops
    best_ck: Optional[_Checkpoint] = None
    rounds = 0
    halvings = 0
    while rounds < max_rounds:
        # Every re-entry into ADAPT (after the first round, after looping from
        # EVAL, or after a CALIBRATE failure) halves eta once -- documented
        # eta_stdp/eta_homeo decay per return to ADAPT.
        if rounds > 0:
            _halve_eta(bundle, pcfg)
            halvings += 1
        stage = ("ADAPT(eta x%.3f)" % (pcfg.eta_scale ** halvings)
                 if rounds > 0 else "ADAPT")
        rep.stage_sequence.append(stage)
        adapt = _adapt_round(bundle, train_data_fn, pcfg)
        rep.adapt_rounds.append(adapt)
        if adapt["forced"]:
            rep.notes.append(
                "ADAPT gate unmet after extra epochs; force-advancing "
                f"(rate={adapt['mean_rate_hz']:.2f} Hz, "
                f"silence={adapt['silence_frac']:.3f})"
            )
        rep.stage_sequence.append("CALIBRATE")
        calib = _calibrate(bundle, train_data_fn, pcfg)
        rep.calibrate_rounds.append(calib)
        if not calib["ok"]:
            # calibrate failed: go back to ADAPT (eta halved on re-entry at
            # the loop top) while a round budget remains, otherwise
            # force-advance with a warning
            if rounds + 1 < max_rounds:
                rep.notes.append(
                    "CALIBRATE band unmet; returning to ADAPT "
                    f"(mean rate {calib['final_mean_rate_hz']:.2f} Hz; "
                    "eta halved on re-entry)"
                )
                rounds += 1
                continue
            rep.notes.append(
                "CALIBRATE band unmet and no round budget left; "
                "force-advancing to COLLECT"
            )
        rep.stage_sequence.append("COLLECT")
        Xtr, ytr, cinfo = _collect(train_data_fn, bundle, pcfg)
        rep.collect_n = cinfo["n_samples"]
        if rep.feat_dim == 0:
            rep.feat_dim = int(Xtr.shape[1])
            rep.n_class = int(np.unique(ytr).size)
        rep.stage_sequence.append("READOUT")
        W, b, rinfo = _readout(Xtr, ytr, pcfg, seed=seed)
        rep.readout = rinfo
        rep.stage_sequence.append("EVAL")
        ev = _eval_round(bundle, test_data_fn, pcfg, W, b, Xtr)
        rep.eval_rounds.append(ev)
        acc = ev["acc"]
        ck = _take_checkpoint(bundle, W, b, acc)
        if best_ck is None or acc >= best_ck.acc:
            best_ck = ck
            rep.best_acc = acc
        else:
            # later round did not improve the previous best -> rollback
            _restore_checkpoint(bundle, best_ck)
            rep.rolled_back = True
            rep.notes.append(
                f"round {rounds} accuracy {acc:.3f} < best "
                f"{best_ck.acc:.3f}; rolled back to the best checkpoint"
            )
            break
        rounds += 1
    rep.n_rounds = rounds
    rep.eta_scale_final = float(pcfg.eta_scale ** halvings)
    rep.ok = True
    return rep