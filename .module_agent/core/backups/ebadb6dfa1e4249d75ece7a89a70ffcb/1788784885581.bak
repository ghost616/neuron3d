"""hstdn.core -- H-STDN core simulation stack (M1-M6), D1-D4 milestone.

Modules:
    layout        ID contract / shape/dtype specs / hyper-parameters (Sec.4)
    spatial_hash  SpatialHash3D (O(N) build, O(1)-amortized queries)
    network       build_network -> NetworkBundle (slab wiring, CSR/CSC, gains)
    kernel        run_sample: L0 time-wheel LIF kernel (strict order, B11/R1)
    encoder       latency / MNIST / DVS encoding (M2)
    plasticity    input-channel STDP, pre/post LTP/LTD, homeo, norm (M4)

Kept import-light: importing this package does not pull numpy-heavy modules;
individual modules are imported explicitly by the caller (gates/exp).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]