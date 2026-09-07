"""hstdn.configs -- Sec.4 hyper-parameter contract package (exp module).

``default.yaml`` under this package is the **single source of truth** for every
experiment hyper-parameter (freeze discipline before G1); magic numbers are
forbidden elsewhere.  This package provides:

- ``default_path()``          path of the contract file
- ``resolve_path(path)``      path resolution helper (explicit path wins, then
                              package-relative, then cwd-relative)
- ``load_config(path)``       pyyaml loader -> plain dict (path resolution
                              built in)
- ``validate_config_dict``    structural contract checks (known top-level
                              sections, required simulation sections, types)
- ``to_core_cfg(config)``     transitional bridge: maps the ``network`` /
                              ``lif`` / ``stdp`` / ``homeo`` sections onto
                              ``hstdn.core.layout.NetConfig`` so that gates and
                              experiments consume the contract values while the
                              core does not read yaml yet.  Once core adopts
                              this file as its own source, this bridge degrades
                              to a pure validator.

ID/notation note: yaml keys follow the design-doc Sec.4 notation
(``tau_mem``, ``radius_in``, ``velocity`` ...), which differs from the flat
NetConfig field names; the bridge below is the only allowed translation and it
raises on unknown keys inside the consumed sections (drift protection).

Section keys are returned by ``load_config`` exactly as written in the file
(no flattening); ``to_core_cfg`` is the only component that knows the mapping.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

__all__ = [
    "DEFAULT_FILE",
    "KNOWN_SECTIONS",
    "SIM_SECTIONS",
    "default_path",
    "resolve_path",
    "load_config",
    "validate_config_dict",
    "to_core_cfg",
]

#: contract file name inside this package
DEFAULT_FILE = "default.yaml"

#: all top-level sections allowed in the contract file (typo protection)
KNOWN_SECTIONS = (
    "network", "lif", "stdp", "homeo",
    "protocol", "readout_l0", "readout_l1", "features", "encoder",
)

#: sections required for any simulation config
SIM_SECTIONS = ("network", "lif", "stdp", "homeo")

_PKG_DIR = Path(__file__).resolve().parent


def default_path() -> Path:
    """Path of the Sec.4 contract file (``<pkg>/default.yaml``)."""
    return _PKG_DIR / DEFAULT_FILE


def resolve_path(path: Optional[str | Path] = None) -> Path:
    """Resolve a config path to an existing file.

    Order: explicit existing path -> package-relative -> cwd-relative ->
    package default.  Raises FileNotFoundError listing the candidates tried.
    """
    if path is None:
        cand = [default_path()]
    else:
        p = Path(path).expanduser()
        cand = [p, _PKG_DIR / p.name if not p.is_absolute() else p,
                Path(p.name) if p.parent == Path(".") else p]
        cand = list(dict.fromkeys(cand))
    for c in cand:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"config file not found; tried: " + ", ".join(str(c) for c in cand)
    )


def load_config(path: Optional[str | Path] = None) -> dict:
    """Load the yaml contract file into a plain dict (pyyaml -> dict).

    Args:
        path: optional explicit path (any resolvable location).

    Returns:
        dict with the top-level sections as keys (values are nested dicts or
        lists exactly as written).

    Raises:
        FileNotFoundError / yaml.YAMLError: unreadable or malformed file.
        AssertionError: malformed top level (not a mapping / missing required
            simulation sections / unknown section keys).
    """
    p = resolve_path(path)
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise AssertionError(
            f"config {p} must be a yaml mapping at the top level, got "
            f"{type(raw).__name__}"
        )
    validate_config_dict(raw)
    return dict(raw)


def validate_config_dict(cfg: Mapping[str, Any]) -> None:
    """Structural contract checks of a loaded config dict.

    Checks: top level is a mapping; every top-level key is a known section;
    the four simulation sections ``network/lif/stdp/homeo`` are all present.

    Raises:
        AssertionError with a readable message + offending keys.
    """
    unknown = sorted(set(cfg) - set(KNOWN_SECTIONS))
    if unknown:
        raise AssertionError(
            f"unknown config top-level section(s): {unknown}; known sections "
            f"are {list(KNOWN_SECTIONS)}"
        )
    missing = [s for s in SIM_SECTIONS if s not in cfg]
    if missing:
        raise AssertionError(
            f"required simulation section(s) missing from config: {missing}"
        )
    for s in SIM_SECTIONS:
        if not isinstance(cfg[s], Mapping):
            raise AssertionError(
                f"section '{s}' must be a mapping, got {type(cfg[s]).__name__}"
            )


# ---------------------------------------------------------------------------
# transitional bridge: contract yaml -> core.layout.NetConfig
# ---------------------------------------------------------------------------

#: (section, yaml key) -> NetConfig field name; the only legal translation.
_NET_MAP = {
    "n_in": "n_in", "n_pool": "n_pool", "e_ratio": "e_frac",
    "radius_in": "r_in", "radius_pool": "r_pool", "velocity": "vel",
    "k_in": "k_in", "k_pool": "k_pool", "max_delay": "delay_max",
    "k_pool_learn": "pool_learn",   # exp-g / Diehl-Cook 冻结开关（默认 True）
}
_LIF_MAP = {
    "tau_mem": "tau_m", "theta0": "theta0", "refractory": "refr",
}
_STDP_MAP = {
    "eta_ltp": "eta_ltp", "eta_ltd": "eta_ltd", "tau_trace": "tau_trace",
    "w_min": "w_lo", "w_max": "w_hi",
}
_HOMEO_MAP = {
    "target_rate_hz": "homeo_rate_target",
    "eta_homeo": "homeo_theta_lr",
    "ema_alpha": "homeo_ema_alpha",
}


def _mapped_kwargs(section: Mapping[str, Any], key_map: Mapping[str, str],
                   section_name: str,
                   extra_allowed: Sequence[str] = ()) -> dict:
    """Translate one section with strict unknown-key rejection.

    ``extra_allowed`` names keys that are legal contract keys of the section
    but are consumed separately (not mapped onto a NetConfig field).
    """
    allowed = set(key_map) | set(extra_allowed)
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(
            f"unknown key(s) in config section '{section_name}': {unknown}; "
            f"allowed: {sorted(allowed)}"
        )
    return {field: section[key] for key, field in key_map.items()
            if key in section}


def to_core_cfg(config: Optional[Mapping[str, Any]] = None):
    """Map the simulation sections of the contract dict to a core NetConfig.

    Args:
        config: loaded contract dict (default: ``load_config()``).

    Returns:
        ``hstdn.core.layout.NetConfig`` with derived fields filled (the
        network built from it is the G0/G1 default-scale network).

    Raises:
        ValueError/NotImplementedError: on contract values the L0 core cannot
            honour (dt != 1, non-square input grid, invalid extent/T ...).
    """
    from hstdn.core.layout import NetConfig  # lazy: keep this module light

    if config is None:
        config = load_config()
    validate_config_dict(config)
    net = config["network"]
    lif = config["lif"]
    stdp = config["stdp"]
    homeo = config["homeo"]

    kw = dict(_mapped_kwargs(net, _NET_MAP, "network",
                             extra_allowed=("extent",)))
    kw.update(_mapped_kwargs(lif, _LIF_MAP, "lif",
                             extra_allowed=("T", "dt")))
    kw.update(_mapped_kwargs(stdp, _STDP_MAP, "stdp"))
    kw.update(_mapped_kwargs(homeo, _HOMEO_MAP, "homeo"))

    # --- network geometry / scale side conditions ---
    n_in = int(kw["n_in"])
    side = int(round(math.sqrt(n_in)))
    if side * side != n_in:
        raise ValueError(
            "contract 'network.n_in' must be a perfect square (square input "
            f"grid layout, D13): n_in={n_in} (side={side}, side^2="
            f"{side * side})"
        )
    kw["n_input_cols"] = side

    extent = net.get("extent")
    if not isinstance(extent, (list, tuple)) or len(extent) != 3:
        raise ValueError(
            "contract 'network.extent' must be a [x, y, z] length-3 list, got "
            f"{extent!r}"
        )
    for v in extent:
        if not isinstance(v, (int, float)):
            raise ValueError(f"extent entries must be numbers: {extent!r}")
    if not float(extent[2]) > 0:
        raise ValueError(f"extent[2] (pool slab z) must be > 0: {extent!r}")
    kw["pool_z_hi"] = float(extent[2])

    # --- lif / time ---
    t_ms = int(lif["T"])
    if t_ms <= 0:
        raise ValueError(f"lif.T must be > 0, got {t_ms}")
    kw["window_s"] = t_ms / 1000.0          # homeo stats window matches T
    dt = lif.get("dt")
    if dt not in (None, 1):
        raise NotImplementedError(
            "L0 kernel steps by dt=1 ms only; contract lif.dt="
            f"{dt!r} is not supported yet"
        )

    return NetConfig(**kw).with_derived()
