"""readout.py -- M5 L0 linear readout (G1) and L1/L2 interface stubs.

L0: NumPy multi-class softmax regression with cross-entropy and L2 weight
regularization.  Training hyper-parameters follow the design-document
``readout_l0`` config section (lr=0.5, iters=300, l2_lambda=1e-4, batch=64);
a 10% validation set selects the iteration count / restores the best-valid
weights to prevent over-fitting (800-dim features + 1k samples over-fit
easily; design-doc M5 review finding).

API:
    train_linear_readout(X_train, y_train, X_val=None, y_val=None, cfg=None)
        -> (W, b, history)
    predict_linear_readout(X, W, b) -> int64 predicted labels
    proba_linear_readout(X, W, b)   -> (n, C) softmax probabilities

L1 (PyTorch, 128 hidden units) and L2 (V2 surrogate gradient) entries are
declared with NotImplementedError until the G3 milestone -- no pass
placeholders.  The implementation is pure NumPy and fully deterministic given
the config seed (mini-batch order is seeded).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from hstdn.core.layout import I8, F8

__all__ = [
    "ReadoutConfig", "readout_from_cfg",
    "train_linear_readout", "predict_linear_readout", "proba_linear_readout",
    "accuracy",
    "train_pytorch_readout", "train_v2_readout",
]


@dataclass(frozen=True)
class ReadoutConfig:
    """readout_l0 hyper-parameters (design-doc config section).

    Fields:
        lr: learning rate (0.5).
        iters: number of training epochs over the train set (300).
        l2_lambda: L2 weight-decay coefficient (1e-4).
        batch: mini-batch size (64; clamped to the train size).
        val_frac: fraction held out as validation when X_val is None (0.1).
        seed: RNG seed for mini-batch shuffling / validation split (0).
        w_scale: initial weight std (1e-3).
    """

    lr: float = 0.5
    iters: int = 300
    l2_lambda: float = 1e-4
    batch: int = 64
    val_frac: float = 0.1
    seed: int = 0
    w_scale: float = 1e-3


def readout_from_cfg(cfg: Union[ReadoutConfig, Mapping[str, Any], None]
                     ) -> ReadoutConfig:
    """Normalize a readout_l0 config (None | ReadoutConfig | mapping)."""
    if cfg is None or isinstance(cfg, ReadoutConfig):
        return cfg if cfg is not None else ReadoutConfig()
    known = set(ReadoutConfig.__dataclass_fields__)
    extra = set(cfg) - known
    if extra:
        raise ValueError(f"unknown readout_l0 config keys: {sorted(extra)}")
    return replace(ReadoutConfig(), **{k: v for k, v in cfg.items() if k in known})


def _validate(x: np.ndarray, y: np.ndarray):
    xa = np.asarray(x, dtype=F8)
    ya = np.asarray(y)
    if xa.ndim != 2:
        raise AssertionError(f"X must be 2-D (n, d), got {xa.shape}")
    if not np.isfinite(xa).all():
        raise AssertionError("X contains NaN/Inf values")
    if ya.ndim != 1 or ya.size != xa.shape[0]:
        raise AssertionError(
            f"y must be 1-D with n={xa.shape[0]} entries, got shape {ya.shape}"
        )
    if ya.dtype.kind not in "iu":
        raise AssertionError(f"y must be integer labels, got dtype {ya.dtype}")
    return xa, ya.astype(np.int64)


def _onehot(y: np.ndarray, n_class: int) -> np.ndarray:
    oh = np.zeros((y.size, n_class), dtype=F8)
    oh[np.arange(y.size), y] = 1.0
    return oh


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _val_split(x: np.ndarray, y: np.ndarray, frac: float, rng: np.random.Generator
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic (seeded) permutation split of train/validation."""
    n = x.shape[0]
    n_val = max(1, int(round(n * frac)))
    perm = rng.permutation(n)
    val_i = perm[:n_val]
    tr_i = perm[n_val:]
    return x[tr_i], y[tr_i], x[val_i], y[val_i]


def train_linear_readout(X_train, y_train, X_val=None, y_val=None, cfg=None
                         ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Train an L0 softmax linear readout with validation-based early stop.

    Args:
        X_train: (n, d) float feature matrix.
        y_train: (n,) integer labels in [0, C).
        X_val/y_val: optional validation split; when None, ``val_frac`` of the
            train set is held out deterministically (seeded).
        cfg: None | ReadoutConfig | mapping (readout_l0 section).

    Returns:
        (W, b, history):
            W: (d, C) float64 weights of the best-valid iteration.
            b: (C,) float64 biases of the best-valid iteration.
            history: dict with per-epoch 'train_acc'/'val_acc'/'loss' lists,
            plus 'best_iter', 'best_val_acc', 'final_val_acc', 'n_train',
            'n_val', 'd', 'n_class'.

    Raises:
        AssertionError: invalid shapes/labels or fewer than 2 classes.
    """
    conf = readout_from_cfg(cfg)
    xa, ya = _validate(X_train, y_train)
    classes = np.unique(ya)
    if classes.size < 2:
        raise AssertionError(f"need at least 2 classes, got {classes.size}")
    if classes.min() != 0 or classes.max() != classes.size - 1:
        raise AssertionError("labels must be dense 0..C-1")
    n_class = int(classes.size)
    rng = np.random.default_rng(conf.seed)

    if X_val is None or y_val is None:
        xt, yt, xv, yv = _val_split(xa, ya, conf.val_frac, rng)
    else:
        xv, yv = _validate(X_val, y_val)
        if yv.max() >= n_class or yv.min() < 0:
            raise AssertionError("validation labels outside the class range")
        xt, yt = xa, ya
    n, d = xt.shape
    batch = min(conf.batch, n)
    if n < 1 or xv.shape[0] < 1:
        raise AssertionError("empty train or validation set")

    yt_oh = _onehot(yt, n_class)
    W = rng.normal(0.0, conf.w_scale, size=(d, n_class))
    b = np.zeros(n_class, dtype=F8)
    best_val = -1.0
    best_iter = 0
    best_W = W.copy()
    best_b = b.copy()
    history: Dict[str, Any] = {
        "train_acc": [], "val_acc": [], "loss": [], "lr": conf.lr,
        "l2_lambda": conf.l2_lambda, "batch": batch,
        "d": d, "n_class": n_class, "n_train": int(xt.shape[0]),
        "n_val": int(xv.shape[0]),
    }
    for it in range(conf.iters):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            xb, yb = xt[idx], yt_oh[idx]
            logits = xb @ W + b
            p = _softmax(logits)
            loss = -np.mean(np.sum(yb * np.log(np.clip(p, 1e-12, 1.0)), axis=1))
            loss += 0.5 * conf.l2_lambda * float(np.sum(W * W))
            grad_w = (xb.T @ (p - yb)) / xb.shape[0] + conf.l2_lambda * W
            grad_b = np.mean(p - yb, axis=0)
            W -= conf.lr * grad_w
            b -= conf.lr * grad_b
            epoch_loss += loss
            n_batches += 1
        pred_tr = predict_linear_readout(xt, W, b)
        pred_va = predict_linear_readout(xv, W, b)
        acc_tr = accuracy(yt, pred_tr)
        acc_va = accuracy(yv, pred_va)
        history["train_acc"].append(float(acc_tr))
        history["val_acc"].append(float(acc_va))
        history["loss"].append(float(epoch_loss / max(1, n_batches)))
        if acc_va > best_val:
            best_val = acc_va
            best_iter = it
            best_W = W.copy()
            best_b = b.copy()
    history["best_iter"] = best_iter
    history["best_val_acc"] = float(best_val)
    history["final_val_acc"] = float(history["val_acc"][-1])
    return best_W, best_b, history


def proba_linear_readout(X, W, b) -> np.ndarray:
    """Softmax class probabilities for the L0 readout.

    Args:
        X: (n, d) features.
        W: (d, C) weights.
        b: (C,) biases.

    Returns:
        (n, C) float64 probabilities (rows sum to 1).
    """
    xa = np.asarray(X, dtype=F8)
    if xa.ndim != 2:
        raise AssertionError(f"X must be 2-D (n, d), got {xa.shape}")
    Wa = np.asarray(W, dtype=F8)
    ba = np.asarray(b, dtype=F8)
    if Wa.ndim != 2 or Wa.shape[0] != xa.shape[1]:
        raise AssertionError(
            f"W must be (d={xa.shape[1]}, C), got {Wa.shape}"
        )
    return _softmax(xa @ Wa + ba)


def predict_linear_readout(X, W, b) -> np.ndarray:
    """Predict integer labels (argmax of the softmax) for the L0 readout.

    Args:
        X: (n, d) features.
        W: (d, C) weights.
        b: (C,) biases.

    Returns:
        (n,) int64 labels in [0, C).
    """
    return np.argmax(proba_linear_readout(X, W, b), axis=1).astype(I8)


def accuracy(y_true, y_pred) -> float:
    """Fraction of correct predictions.

    Args:
        y_true: (n,) integer labels.
        y_pred: (n,) integer labels.

    Returns:
        float accuracy in [0, 1].
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise AssertionError(f"label shape mismatch: {yt.shape} vs {yp.shape}")
    return float(np.mean(yt == yp)) if yt.size else 0.0


def train_pytorch_readout(*args, **kwargs):
    """L1 readout (PyTorch, 128 hidden units) -- G3 milestone entry.

    Raises:
        NotImplementedError: always at this milestone (no pass placeholder).
    """
    raise NotImplementedError(
        "train_pytorch_readout (L1, PyTorch 128-hidden) belongs to the G3 "
        "milestone and is not implemented in the D1-D5 scope"
    )


def train_v2_readout(*args, **kwargs):
    """L2 readout (V2 surrogate gradient) -- G3+ milestone entry.

    Raises:
        NotImplementedError: always at this milestone (no pass placeholder).
    """
    raise NotImplementedError(
        "train_v2_readout (L2, surrogate gradient) belongs to the G3+ "
        "milestone and is not implemented yet"
    )