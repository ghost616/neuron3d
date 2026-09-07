"""hstdn.core -- H-STDN core simulation stack (M1-M6).

Modules (imported explicitly by callers / lazily via PEP 562 __getattr__):
    layout        ID contract / shape/dtype specs / hyper-parameters (Sec.4)
    spatial_hash  SpatialHash3D (O(N) build, O(1)-amortized queries)
    network       build_network -> NetworkBundle (slab wiring, CSR/CSC, gains)
    kernel        run_sample: L0 time-wheel LIF kernel (strict order, B11/R1)
    encoder       latency / MNIST / DVS encoding (M2)
    plasticity    input-channel STDP, pre/post LTP/LTD, homeo, norm (M4)
    features      M5 feature extraction (v0 sqrt+L2 / v1 time bins / v2 note)
    readout       M5 L0 linear softmax readout (+L1/L2 interface stubs)
    scheduler     M6 G1 protocol state machine (ADAPT..EVAL, run_g1_protocol)

Import-light: importing this package itself does not pull numpy-heavy
modules; ``from hstdn.core import features`` resolves lazily on first
attribute access (PEP 562).
"""

from __future__ import annotations

import importlib as _importlib

__version__ = "0.2.0"

_MODULES = (
    "layout", "spatial_hash", "network", "kernel", "encoder", "plasticity",
    "features", "readout", "scheduler",
)

__all__ = ["__version__", *_MODULES]


def __getattr__(name: str):
    """Lazily import a submodule on first attribute access (PEP 562)."""
    if name in _MODULES:
        return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
