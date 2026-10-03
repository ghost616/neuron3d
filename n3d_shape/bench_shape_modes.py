"""n3d_shape shape / geo-field / fc-wrap / scope mode-combination benchmark driver.

Sequential driver with at-the-run ledger accounting and resumable checkpoints, modelled on the
existing n3d_shape/run_ladder.py (subprocess driver + at-the-run accounting + resume).

Goal
----
Run one full-combination benchmark for the neuron-space shape variants:
5 shape variants (sphere / cube / cylinder lambda=0.5 / 1 / 2) x geo_field {none, additive}
x fc_dim {0, -1} x scope {any_isolated, all_isolated} x N {256, 512, 1024} = 120 products.
Every product is one full training run under a single frozen caliber
(--preset default --epochs 20 --seed 42), and every product records at-the-run numbers
for its artifact fingerprint, topology and structural assertions.

This script only DRIVES; it never implements training logic. Each product is trained by a
subprocess call into n3d_shape/train.py (process exit releases model and optimizer, which
avoids peak-memory accumulation across products at N=1024). Artifact names MUST be produced
at the run by train.resolve_checkpoint_path() -- never hand-assembled fingerprints --
the same discipline used by run_ladder.py.

Fixed caliber (frozen before the run, archived in ledger meta.caliber)
--------------------------------------------------------------
* preset = default, epochs = 20, seed = 42, threads = 0;
* D = H = 0.1 (Config hard constraint D <= H; the preset default is D = H = 0.10, so no
  extra --h / --d override is passed);
* y_in = y_out = 8, input_dim = 784, output_dim = 10, num_workers = 0, flow_axis = z,
  space_radius = 0 (formula lower bound R_min), placement = fcc;
* arch = neuron3d (preset default); input_scope = readout_scope = scope;
* --max-batches is NEVER used (limited-batch runs relocate the artifact into _verify/,
  which conflicts with the benchmark artifact caliber);
* epochs does not enter the artifact fingerprint, so adding --epochs 20 does not change
  artifact names; _nosyn is a constant format segment (independent of config).

Artifact naming (produced at the run by train)
--------------------------------------------
Names come exclusively from train.resolve_checkpoint_path(). All 120 target paths are unique
(asserted by --dry-run). NOTE, verified at the run: with this frozen caliber (--epochs 20)
EVERY product receives a full_*.pt fingerprint name, because train.is_default_config()
compares config.to_dict() including epochs while DEFAULT_CONFIG.epochs == 10. The
checkpoints/n3d_shape/model.pt branch is therefore unreachable for this batch; the driver
reports the measured count instead of assuming one.

Modes
-----
* --dry-run: enumerate the 120-product plan (index / N / shape / lambda / geo / fc /
  scope / target path / full command line), assert 120/120 unique target paths, exit 0.
  No training, no deletion.
* --check-topology: pure construction. For the 15 (shape, N) x 2 scope = 30 topology
  combinations build ThreeDNeuronSpace and assert E / K / S_in / S_out equal the fixed
  expectation table bit for bit; additionally assert that for the same (shape, N) the
  any_isolated and all_isolated builds have identical E and K.
* --check-step0-geo: pure construction. For the same 30 topology combinations (fc_dim=0)
  build the model twice, with geo_field=none and geo_field=additive, and assert that
  their _effective_edge_weight() is bit-identical at step 0 (torch.equal); also assert
  that the additive build really creates geo_rbf_theta (shape [k+1], all zeros) and
  geo_alpha (scalar, initial value 1.0).
* --wipe: A1 + A2. A1 recursively enumerates every file under checkpoints/n3d_shape/
  recording {path, bytes, sha256} plus the frozen values of the 6 anchor artifacts, and
  writes them to checkpoints/n3d_shape_deletion_manifest_<YYYYMMDD>.json (OUTSIDE the
  n3d_shape directory). A2 deletes everything under checkpoints/n3d_shape/ (including
  _verify/ and _verify/_void/), then recreates the empty directories
  checkpoints/n3d_shape/ and checkpoints/n3d_shape/_verify/ and asserts that the file
  count AT THE INSTANT AFTER DELETION is 0. Deletion is strictly record-first-then-delete
  and every entry is printed.
  [!] The A1 count is REPORTED (measured), not asserted equal to 176: this driver itself
  writes into checkpoints/n3d_shape/_verify/ (bench_dryrun_plan.txt,
  bench_ledger_<date>.json and one bench_log_*.txt per trained product), so on a pristine
  tree the count is 176 while after a full 120-product batch it is about 298. The pristine
  value 176 is kept only as a reference baseline in the report and the manifest.
* --batch {256,512,1024,all} (default None = all): run only the given N batch. It is ignored
  by --wipe, whose deletion scope is the entire checkpoints/n3d_shape/ tree.
* --limit N (default 0 = unlimited): run only the first N products (driver self-check).
* --fail-fast: stop at the FIRST failing product. [!] F17 -- DEFAULT FAILURE SEMANTICS: without
  this flag the driver keeps processing the rest of the batch, then exits non-zero at the end
  and lists every failing product together with its failing criteria (a permanently failing
  artifact therefore costs a full batch; use --fail-fast when a quick abort is wanted).
* --ledger PATH (default checkpoints/n3d_shape/_verify/bench_ledger_<YYYYMMDD>.json).
* --logs-dir PATH (default checkpoints/n3d_shape/_verify/).
* --check-real-artifact / --real-artifact PATH: drive the structural assertions with a REAL
  _nosyn artifact and require every assertion green. [!] The default fixture is
  `full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_nosyn_s42.pt` -- one of
  THIS batch own 120 products (plan index 3), because the fixture MUST BE SUPPLIED BY THIS
  BENCHMARK SET ITSELF (the previous default came from the round-24 rebuild and was removed by
  --wipe, so it could no longer exist when the check ran).
* --inject-syn-name NAME: negative proof for that check; NAME is appended to SYN_BUFFER_NAMES
  for THIS PROCESS ONLY (the file on disk is never modified) so the paired assertions must
  fail.
  [!] F12 -- EXIT-CODE SEMANTICS OF THE INJECTION MODE: exit 0 means ONLY that the negative
  proof held (the bogus name was indeed rejected). It does NOT mean the assertions passed --
  in this mode a FAILING assertion is the EXPECTED outcome. A human reviewing the run must
  read the [C] result line and the injected name, never the exit code alone.
  [!] F13 -- using --inject-syn-name WITHOUT --check-real-artifact still enters this
  self-test branch (--inject-syn-name implies --check-real-artifact) and now prints an
  explicit "[mode] inject mode WITHOUT --check-real-artifact: entering the real-artifact
  self-test, NOT training" line, so it cannot be mistaken for training mode.

Core behaviour (training mode)
------------------------------
1. Run --check-topology and --check-step0-geo first as a gate; if either fails, exit
   non-zero immediately and never enter training;
2. per product: resolve the target path at the run; if absent, train it; if present,
   compare its bytes + SHA256 against the ledger record for that product -- identical
   means NO training, re-run every assertion only, and record the result (status ok when
   the assertions pass, assert_failed when they do not); different means STOP with a
   non-zero exit code (conflict report, never silently overwrite);
3. training = subprocess [sys.executable, -X, utf8, n3d_shape/train.py, ...]; stdout and
   stderr are captured to bench_log_<NNN>_<artifact without .pt>.txt and the exit code
   is asserted to be 0;
4. after training, torch.load the artifact and record bytes / sha256 / params /
   test_acc / E / K / S_in / S_out / effective epochs / cmd / elapsed_s / status,
   asserting E/K/S_in/S_out bit-equal the expectation table, effective epochs == 20
   and test_acc inside [0, 1];
5. structural assertions (fc / geo / nosyn / persistence caliber), see
   assert_checkpoint_structure;
6. the ledger is refreshed atomically after EVERY product (write .tmp then replace);
7. at the end assert n_ok + n_skipped == batch size and print the summary, including a
   report of how many products in this batch have K > 9 (load-bearing report) and the
   RECORDED geometric-field data (alpha_final, theta non-zero count);
8. after Ctrl-C the driver may be re-run: products already on disk with matching bytes +
   SHA256 are NOT retrained -- their assertions are re-run and recorded.

Discipline
----------
Any failure (non-zero exit / NaN / topology mismatch / structural assertion failure)
makes the run exit non-zero -- never silently overwritten, never recorded as ok.

Run history of THIS benchmark set (numbers unchanged, only the narrative is updated -- B3)
----------------------------------------------------------------------------------------
1. FIRST RUN of `--batch all`: 60 ok + 60 assert_failed. The 60 failures were ALL
   `geo_field=additive` products and were caused by two over-strict product assertions that
   treated CONSTRUCTION invariants as trained-product criteria (SPEC-ERROR 1: theta must be
   all-zero and alpha == geo_alpha_init; SPEC-ERROR 2: the non-persistent buffer set must be
   exactly the declared 8). Every SUBSTANTIVE assertion (E/K/S_in/S_out vs the expectation
   table, epochs == 20, fc structure, nosyn 8-absent + 5-present) passed for all 120.
2. ASSERTION FIX: the step-0 properties are now asserted only by the `--check-step0-geo` gate;
   a trained product asserts presence / shape / finiteness and RECORDS alpha_final and the
   theta non-zero count. Set-type assertions now name the allowed extras explicitly
   (`allowed_extra_non_persistent`: empty for none, the four `edge_geo_feat*` for additive).
3. PURE RE-VALIDATION (no training at all): re-running `--batch all` produced 0 `[run]` lines
   and 120 `[resume]` lines, i.e. every artifact was re-checked from disk against the ledger,
   and the 60 additive products moved assert_failed -> ok.
   RESULT: 120/120 ok, 0 assert_failed. The 120 `.pt` files were byte-for-byte unchanged
   (digest-of-digests 606c1bfab4e1afc4245992f3bce2efac591cd6c12397fe104ddbfaa2fd4a38d4,
   total 265,762,352 B before and after).

[!] F17 -- DEFAULT FAILURE SEMANTICS: a failing product does NOT abort the batch. The driver
finishes the remaining products, then exits non-zero and lists every failing product with its
failing criteria (--fail-fast restores stop-at-first-failure). Both semantics were observed in
practice: the plan text originally said "stop on any failure", while the driver ran all 120
products of --batch all and then exited 1 -- this is now the documented default.
[!] Assertion scope discipline: CONSTRUCTION invariants (theta all-zero, alpha ==
geo_alpha_init at step 0) are asserted by the --check-step0-geo gate ONLY. A TRAINED artifact
asserts presence / shape / finiteness, and records alpha_final and theta_nonzero as data --
asserting initial values on a trained product would reject a correctly learning field.
Set-type assertions always name the allowed extras explicitly (a geo_field == "additive"
model legitimately registers four extra non-persistent edge_geo_feat* buffers).
Every number written to the ledger is sampled at the run; only the frozen caliber
constants and the TOPOLOGY_EXPECT table may appear as literals in code.

Usage
-----
    python n3d_shape/bench_shape_modes.py --dry-run
    python n3d_shape/bench_shape_modes.py --check-topology --check-step0-geo
    python n3d_shape/bench_shape_modes.py --wipe --batch all
    python n3d_shape/bench_shape_modes.py --batch 256
    python n3d_shape/bench_shape_modes.py --limit 2

Ledger: checkpoints/n3d_shape/_verify/bench_ledger_<YYYYMMDD>.json
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------
# Path constants
# ----------------------------------------------------------------------
MODULE_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.abspath(os.path.join(MODULE_DIR, os.pardir))
CHECKPOINTS_ROOT: str = os.path.join(PROJECT_ROOT, "checkpoints")
CHECKPOINT_DIR: str = os.path.join(CHECKPOINTS_ROOT, "n3d_shape")
VERIFY_DIR: str = os.path.join(CHECKPOINT_DIR, "_verify")
VOID_DIR: str = os.path.join(VERIFY_DIR, "_void")
TRAIN_PY: str = os.path.join(MODULE_DIR, "train.py")

# ----------------------------------------------------------------------
# Fixed caliber (frozen before the run; archived in ledger meta.caliber)
# ----------------------------------------------------------------------
PRESET: str = "default"
EPOCHS: int = 20
SEED: int = 42
THREADS: int = 0
Y_IN: int = 8
Y_OUT: int = 8
INPUT_DIM: int = 784
OUTPUT_DIM: int = 10
NUM_WORKERS: int = 0
H: float = 0.1
D: float = 0.1
FLOW_AXIS: str = "z"
SPACE_RADIUS: float = 0.0
PLACEMENT: str = "fcc"
ARCH: str = "neuron3d"

# Shape axis: 5 shape variants (sphere / cube / cylinder lambda=0.5 / 1 / 2)
SHAPE_VARIANTS: Tuple[Tuple[str, Optional[float]], ...] = (
    ("sphere", None),
    ("cube", None),
    ("cylinder", 0.5),
    ("cylinder", 1.0),
    ("cylinder", 2.0),
)
GEO_FIELDS: Tuple[str, ...] = ("none", "additive")
FC_DIMS: Tuple[int, ...] = (0, -1)
SCOPES: Tuple[str, ...] = ("any_isolated", "all_isolated")
N_BATCHES: Tuple[int, ...] = (256, 512, 1024)
EXPECTED_TOTAL: int = (
    len(SHAPE_VARIANTS) * len(GEO_FIELDS) * len(FC_DIMS) * len(SCOPES) * len(N_BATCHES)
)

# geo_field=additive RBF basis count (preset default 12): theta shape = [k+1] = [13]
GEO_RBF_K: int = 12
GEO_THETA_LEN: int = GEO_RBF_K + 1
GEO_ALPHA_INIT: float = 1.0
# SPEC-ERROR 2: geo_field == "additive" legitimately registers these four persistent=False feature
# buffers (model.py). They are the ONLY extras allowed on top of the 8 synapse-class buffers.
ADDITIVE_GEO_FIELD: str = "additive"
GEO_FEATURE_NON_PERSISTENT_BUFFERS: Tuple[str, ...] = (
    "edge_geo_feat",
    "edge_geo_feat_raw",
    "edge_geo_feat_min",
    "edge_geo_feat_max",
)

# ----------------------------------------------------------------------
# Topology expectation table (FROZEN criterion, not measurements).
# Key = (N, shape, cyl_aspect); value = (K, E, S_in_any, S_out_any, S_in_all, S_out_all).
# ----------------------------------------------------------------------
TOPOLOGY_EXPECT: Dict[Tuple[int, str, Optional[float]], Tuple[int, int, int, int, int, int]] = {
    (256, "sphere", None): (9, 736, 193, 187, 13, 14),
    (256, "cube", None): (9, 713, 195, 186, 29, 27),
    (256, "cylinder", 0.5): (5, 679, 189, 196, 52, 52),
    (256, "cylinder", 1.0): (9, 705, 196, 193, 27, 26),
    (256, "cylinder", 2.0): (15, 717, 200, 199, 15, 19),
    (512, "sphere", None): (13, 1559, 357, 363, 21, 21),
    (512, "cube", None): (11, 1523, 388, 378, 46, 46),
    (512, "cylinder", 0.5): (7, 1456, 392, 381, 77, 76),
    (512, "cylinder", 1.0): (11, 1495, 366, 381, 48, 47),
    (512, "cylinder", 2.0): (17, 1497, 387, 387, 29, 29),
    (1024, "sphere", None): (15, 3252, 748, 721, 36, 33),
    (1024, "cube", None): (13, 3192, 737, 727, 79, 79),
    (1024, "cylinder", 0.5): (9, 3121, 742, 743, 115, 113),
    (1024, "cylinder", 1.0): (15, 3156, 743, 732, 73, 68),
    (1024, "cylinder", 2.0): (23, 3202, 731, 713, 44, 43),
}

# A1: 6 anchor artifacts whose pre-wipe values must be frozen into the deletion manifest.
# The values themselves are READ at the run and written to the manifest -- never hard-coded.
ANCHOR_ARTIFACTS: Tuple[str, ...] = (
    "full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt",
    "full_shapecube_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt",
    "full_shapecylinder_a0.5_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt",
    "full_shapecylinder_a1_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt",
    "full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt",
    "full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.pt",
)
EXPECTED_PRE_WIPE_FILES: int = 176

# F8: real _nosyn fixture used to drive the structural assertions with an actual artifact
# (the object-under-test is a real product, not this script own constants). It is a genuine
# nosyn product (8 synapse-class tensors absent from state_dict) and must NEVER be deleted.
#
# [!] B1 (this round): the fixture MUST BE SUPPLIED BY THIS BENCHMARK SET ITSELF.
#     The previous default pointed at the round-24 rebuild product
#     `full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_nosyn_s42_fc_align.pt`;
#     `--wipe` deletes the WHOLE checkpoints/n3d_shape/ tree, so that file no longer exists and
#     the check would fail with "fixture missing" for a reason that has nothing to do with the
#     assertions under test. The default is now one of THIS batch own 120 products (index 3 of
#     the plan: N=256 / sphere / geo=none / fc=-1 / any_isolated), which the benchmark itself
#     trains and which is therefore guaranteed to be present after any full batch.
#     Its payload config DOES carry every dimension (N / shape / cyl_aspect / fc_dim / both
#     scopes) and -- from round 21 onwards -- also geo_field, so no dimension has to be assumed.
REAL_ARTIFACT_FIXTURE: str = (
    "full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_nosyn_s42.pt"
)
# Fallback dimensions, used ONLY when the payload carries no config (never for the default
# fixture). They describe the default fixture above, which is a plan product of this batch.
REAL_ARTIFACT_DIMS: Dict[str, Any] = {
    "N": 256,
    "shape": "sphere",
    "cyl_aspect": None,
    "geo_field": "none",
    "fc_dim": -1,
    "scope": "any_isolated",
}

# 8 synapse-class buffers (nosyn caliber: persistent=False, absent from state_dict).
# [!] F1 (blocking error, fixed after the review round): the last-but-one name used to read
#     representative_syn_output, which is NOT a real tensor: model.py registers
#     representative_syn_out. Enumerated live from the model (the only acceptable way to fill
#     this in -- never from memory):
#         input_isolated_mask / input_syn_pos / neuron_conn_mask / output_isolated_mask /
#         output_syn_pos / representative_syn_input / representative_syn_out / syn_dist
#     Consequence of the wrong name: absent_in_sd always contained the bogus key (so
#     nosyn_ok was always False) and named_buffers() always missed it (so every product was
#     judged failed). Both the positive and the negative direction are now asserted.
SYN_BUFFER_NAMES: Tuple[str, ...] = (
    "syn_dist",
    "input_syn_pos",
    "output_syn_pos",
    "representative_syn_input",
    "representative_syn_out",
    "input_isolated_mask",
    "output_isolated_mask",
    "neuron_conn_mask",
)
# 5 tensors that must stay persistent (topology / n3d_viz contract keys)
PERSISTENT_KEY_NAMES: Tuple[str, ...] = (
    "neuron_pos",
    "edge_src",
    "edge_dst",
    "edge_dist",
    "topo_index",
)
# geo_field != none: persistent geometric buffers and geometric parameters
GEO_PERSISTENT_BUFFERS: Tuple[str, ...] = ("geo_rbf_centers", "geo_rbf_width")
GEO_PARAM_NAMES: Tuple[str, ...] = ("geo_rbf_theta", "geo_alpha")
FC_KEY_NAMES: Tuple[str, ...] = (
    "fc_in_weight",
    "fc_in_bias",
    "fc_out_weight",
    "fc_out_bias",
)

# ======================================================================
# Basic helpers
# ======================================================================
def today_stamp() -> str:
    """Return the local date stamp YYYYMMDD (used by ledger / manifest file names)."""
    return datetime.datetime.now().strftime("%Y%m%d")


def utc_now() -> str:
    """Return the current UTC time (ISO 8601, second precision)."""
    return (
        datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    )


def sha256_of(path: str) -> str:
    """Return the file SHA256 (chunked read, so large artifacts never enter memory at once)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: str, payload: Any) -> None:
    """Atomically write JSON (write tmp first, then os.replace), UTF-8 without BOM."""
    tmp = path + ".tmp"
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: str, text: str) -> None:
    """Atomically write text (write tmp first, then os.replace), UTF-8 without BOM."""
    tmp = path + ".tmp"
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def git_head() -> Optional[str]:
    """Return the current git HEAD (short hash with a -dirty marker); None when unavailable."""
    if shutil.which("git") is None:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", PROJECT_ROOT, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if out.returncode != 0:
            return None
        head = (out.stdout or "").strip()
        st = subprocess.run(
            ["git", "-C", PROJECT_ROOT, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if st.returncode == 0 and (st.stdout or "").strip():
            head = head + "-dirty"
        return head or None
    except Exception:  # pragma: no cover - environment without git / timeout
        return None


def load_train_module() -> Any:
    """Import this module own train module (reuse its naming / config / validation paths).

    The train module is imported from MODULE_DIR (the directory holding this file), so the
    script keeps working when the n3d_shape package is not importable as a package.

    [!] Portability note (independent test round T7/T8, info level): when a COPY of this file
    is placed outside n3d_shape/, MODULE_DIR points at that copy directory and train.py is not
    next to it, so the import below fails and the driver cannot run. Injection-style
    reproductions on a temp copy must therefore export PYTHONPATH=<repo>/n3d_shape (or copy
    the whole n3d_shape directory), exactly as the reviewing scripts do.
    """
    sys.path.insert(0, MODULE_DIR)
    try:
        from . import train as train_mod  # type: ignore

        return train_mod
    except ImportError:  # pragma: no cover - when this file is run as a plain script
        try:
            import train as train_mod  # type: ignore

            return train_mod
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "cannot import train.py next to {0} (MODULE_DIR={1}). Run this driver in "
                "place, or export PYTHONPATH=<repo>/n3d_shape when working on a copy. "
                "Original error: {2}".format(__file__, MODULE_DIR, exc)
            ) from exc


def count_files_recursive(root: str) -> int:
    """Count files recursively under root (0 when the directory does not exist)."""
    if not os.path.isdir(root):
        return 0
    n = 0
    for _dirpath, _dirnames, filenames in os.walk(root):
        n += len(filenames)
    return n


def list_files_recursive(root: str) -> List[str]:
    """List every file under root recursively (sorted, so ordering is deterministic)."""
    out: List[str] = []
    if not os.path.isdir(root):
        return out
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            out.append(os.path.join(dirpath, name))
    return sorted(out)


def rel_to_root(path: str) -> str:
    """Return path relative to the project root, using forward slashes only."""
    rel = os.path.relpath(path, PROJECT_ROOT)
    return rel.replace(os.sep, "/")


# ======================================================================
# Plan: the 120 combinations (Config / target path / command line)
# ======================================================================
@dataclass(frozen=True)
class PlanSeed:
    """One combination identity (index plus every dimension), before artifact naming."""

    index: int
    n: int
    shape: str
    cyl_aspect: Optional[float]
    geo_field: str
    fc_dim: int
    scope: str


@dataclass(frozen=True)
class Variant:
    """One executable plan entry: dimensions plus the at-the-run artifact naming."""

    index: int
    n: int
    shape: str
    cyl_aspect: Optional[float]
    geo_field: str
    fc_dim: int
    scope: str
    key: str
    config: Any
    artifact: str
    path: str
    argv: List[str]
    command: List[str]
    is_default_path: bool


def build_argv(seed: PlanSeed) -> List[str]:
    """Build the CLI arguments of one training run (without interpreter and script path).

    Parameters
    ----------
    seed : PlanSeed
        The dimension combination.

    Returns
    -------
    List[str]
        Arguments such as ["--preset", "default", "--epochs", "20", ...].
    """
    argv: List[str] = [
        "--preset", PRESET,
        "--epochs", str(EPOCHS),
        "--shape", seed.shape,
        "--n", str(seed.n),
        "--seed", str(SEED),
        "--fc-dim", str(seed.fc_dim),
        "--geo-field", seed.geo_field,
        "--input-scope", seed.scope,
        "--readout-scope", seed.scope,
    ]
    if seed.cyl_aspect is not None:
        argv += ["--cyl-aspect", "{0:g}".format(seed.cyl_aspect)]
    argv += ["--threads", str(THREADS)]
    return argv


def seed_key(seed: PlanSeed) -> str:
    """Ledger primary key: uniquely determined by the config dimensions (not by naming)."""
    lam = "" if seed.cyl_aspect is None else "{0:g}".format(seed.cyl_aspect)
    return "N{0}|{1}|{2}|geo={3}|fc={4}|scope={5}".format(
        seed.n, seed.shape, lam, seed.geo_field, seed.fc_dim, seed.scope
    )


def enumerate_seeds() -> List[PlanSeed]:
    """Enumerate all 120 combinations in the frozen order: N -> shape -> geo -> fc -> scope."""
    out: List[PlanSeed] = []
    idx = 0
    for n in N_BATCHES:
        for shape, lam in SHAPE_VARIANTS:
            for geo in GEO_FIELDS:
                for fc in FC_DIMS:
                    for scope in SCOPES:
                        idx += 1
                        out.append(
                            PlanSeed(
                                index=idx,
                                n=int(n),
                                shape=shape,
                                cyl_aspect=lam,
                                geo_field=geo,
                                fc_dim=int(fc),
                                scope=scope,
                            )
                        )
    return out


def build_plan(seeds: Sequence[PlanSeed]) -> List[Variant]:
    """Turn combination identities into executable plan entries.

    The artifact name is ALWAYS produced at the run by the train module
    (train.parse_args + train.validate_override_args + train.apply_overrides +
    train.resolve_checkpoint_path), reusing its naming code path instead of re-implementing
    the fingerprint format here.

    Parameters
    ----------
    seeds : Sequence[PlanSeed]
        Combination identities.

    Returns
    -------
    List[Variant]
        Executable plan entries.
    """
    train_mod = load_train_module()
    out: List[Variant] = []
    for s in seeds:
        argv_plain = build_argv(s)
        # train.apply_overrides prints override details through log_info; swallow that so this
        # driver own stdout keeps only plan / progress / summary lines.
        with contextlib.redirect_stdout(io.StringIO()):
            args = train_mod.parse_args(argv_plain)
            train_mod.validate_override_args(args)
            cfg = train_mod.apply_overrides(train_mod.PRESETS[args.preset], args)
            path = train_mod.resolve_checkpoint_path(
                args.checkpoint, args.max_batches, cfg, args.tag
            )
        artifact = os.path.basename(path)
        out.append(
            Variant(
                index=s.index,
                n=s.n,
                shape=s.shape,
                cyl_aspect=s.cyl_aspect,
                geo_field=s.geo_field,
                fc_dim=s.fc_dim,
                scope=s.scope,
                key=seed_key(s),
                config=cfg,
                artifact=artifact,
                path=path,
                argv=argv_plain,
                command=[sys.executable, "-X", "utf8", TRAIN_PY] + argv_plain,
                is_default_path=(artifact == "model.pt"),
            )
        )
    return out


def filter_plan(plan: Sequence[Variant], batch: Optional[str], limit: int) -> List[Variant]:
    """Filter the plan by --batch / --limit (original indices are preserved).

    Parameters
    ----------
    plan : Sequence[Variant]
        The full 120-product plan.
    batch : Optional[str]
        "256" / "512" / "1024" / "all" / None; None or "all" keeps every N batch.
    limit : int
        Keep only the first limit entries (0 = unlimited).

    Returns
    -------
    List[Variant]
        The selected subset, in plan order.
    """
    if batch is None or batch == "all":
        sel = list(plan)
    else:
        sel = [v for v in plan if v.n == int(batch)]
    if limit > 0:
        sel = sel[: int(limit)]
    return sel


def caliber_dict() -> Dict[str, Any]:
    """Return the FROZEN caliber dictionary written to ledger meta.caliber (anti-drift)."""
    return {
        "preset": PRESET,
        "epochs": EPOCHS,
        "seed": SEED,
        "threads": THREADS,
        "arch": ARCH,
        "y_in": Y_IN,
        "y_out": Y_OUT,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "num_workers": NUM_WORKERS,
        "H": H,
        "D": D,
        "flow_axis": FLOW_AXIS,
        "space_radius": SPACE_RADIUS,
        "placement": PLACEMENT,
        "max_batches": 0,
        "shape_variants": [
            {"shape": s, "cyl_aspect": lam} for s, lam in SHAPE_VARIANTS
        ],
        "geo_fields": list(GEO_FIELDS),
        "fc_dims": list(FC_DIMS),
        "scopes": list(SCOPES),
        "n_batches": list(N_BATCHES),
        "expected_total": EXPECTED_TOTAL,
        "total_order": "N -> shape -> geo_field -> fc_dim -> scope",
        "reason_D_equals_H": (
            "Config enforces D <= H; H is the synapse-cloud radius (hence D=0.10 is the "
            "largest admissible threshold). D=H=0.10 makes the connection criterion run at "
            "its maximum admissible radius and matches the existing any/any products "
            "(DEFAULT_CONFIG already ships H=D=0.1), so this batch passes no extra "
            "--h / --d override."
        ),
        "nosyn_note": (
            "_nosyn is a constant format segment (independent of config): the 8 synapse-class "
            "buffers are persistent=False and never written to disk, so a product no longer "
            "self-certifies its synapse geometry; re-verification must recompute from "
            "config + seed."
        ),
        "model_pt_exception": (
            "train.is_default_config() compares config.to_dict() including epochs, so the "
            "checkpoints/n3d_shape/model.pt branch applies only when --epochs is omitted or "
            "equal to DEFAULT_CONFIG.epochs (10). With this frozen caliber (--epochs 20) no "
            "combination reaches it and all 120 products carry full_* fingerprint names; the "
            "driver reports the measured count at the run and never assumes one."
        ),
        "epochs_not_in_fingerprint": (
            "epochs does not enter the artifact fingerprint, so --epochs 20 does not change "
            "artifact names; the 120 artifact names of this batch are determined by "
            "shape / geo / fc / scope / N only."
        ),
    }


def invalidated_chains() -> Dict[str, Any]:
    """Register the two existing chains that this batch does NOT rebuild (deleted by --wipe)."""
    shared_note = (
        "the assertion logic of these scripts is UNCHANGED; after this batch --wipe their "
        "frozen constants and ledger paths point at files that no longer exist, so they are "
        "not reproducible within this batch (recorded as-is, never faked as passing)"
    )
    return {
        "n_ladder": {
            "description": "N ladder (run_ladder.py / verify_ladder.py): the original 22 products and ledger",
            "n_products": 22,
            "rebuild_this_batch": False,
            "ledger": "checkpoints/n3d_shape/_verify/ladder_runs.json",
            "ledger_deleted_with_wipe": True,
            "frozen_constants_invalidated": True,
            "scripts": ["n3d_shape/run_ladder.py", "n3d_shape/verify_ladder.py"],
            "assertion_logic_unchanged": shared_note,
        },
        "fc_alignment": {
            "description": "fc_align chain (run_fc_alignment.py / verify_fc_alignment.py): the original 18 rounds",
            "n_rounds": 18,
            "rebuild_this_batch": False,
            "ledger": "checkpoints/n3d_shape/_verify/fc_alignment_runs.json",
            "ledger_deleted_with_wipe": True,
            "frozen_constants_invalidated": True,
            "scripts": [
                "n3d_shape/run_fc_alignment.py",
                "n3d_shape/verify_fc_alignment.py",
            ],
            "assertion_logic_unchanged": shared_note,
        },
    }


# ======================================================================
# Assertion records (uniform printing; any failure stops the run)
# ======================================================================
@dataclass
class Assertion:
    """One machine-readable assertion record (stored in the ledger, auditable one by one)."""

    group: str
    name: str
    ok: bool
    detail: str


@dataclass
class RunState:
    """Run state of this process: collected assertions plus a failure flag."""

    assertions: List[Assertion] = field(default_factory=list)
    failed: bool = False


class BenchFailure(RuntimeError):
    """Benchmark driver failure (non-zero exit code); the run never silently continues."""


def record(
    state: RunState,
    group: str,
    name: str,
    ok: bool,
    detail: str,
    mark: str = "  ",
) -> None:
    """Record and print one assertion (ok=False sets state.failed)."""
    state.assertions.append(
        Assertion(group=group, name=name, ok=bool(ok), detail=detail)
    )
    flag = "PASS" if ok else "FAIL"
    print("  [{0}] <{1}> {2}: {3}".format(flag, mark, name, detail))
    if not ok:
        state.failed = True


def record_pair(
    state: RunState,
    group: str,
    name: str,
    got: Sequence[int],
    want: Sequence[int],
    mark: str = "  ",
) -> None:
    """Compare two integer sequences element by element (the single entry for table criteria)."""
    ok = list(got) == list(want)
    detail = "measured={0} vs expected={1}".format(tuple(got), tuple(want))
    if not ok:
        detail += " -> element-wise mismatch"
    record(state, group, name, ok, detail, mark=mark)


# ======================================================================
# Pure-construction helpers
# ======================================================================
def build_config_for(
    n: int,
    shape: str,
    cyl_aspect: Optional[float],
    scope: str,
    fc_dim: int = 0,
    geo_field: str = "none",
) -> Any:
    """Build a Config at the run, using the SAME override path as the training command line.

    Parameters
    ----------
    n : int
        Neuron count.
    shape : str
        sphere / cube / cylinder.
    cyl_aspect : Optional[float]
        Cylinder aspect ratio (None for non-cylinder shapes).
    scope : str
        any_isolated / all_isolated, used for both input_scope and readout_scope.
    fc_dim : int
        Two-end fully-connected wrap switch (0 / -1).
    geo_field : str
        none / additive.

    Returns
    -------
    Config
        The effective configuration object.
    """
    train_mod = load_train_module()
    seed = PlanSeed(
        index=0,
        n=int(n),
        shape=shape,
        cyl_aspect=cyl_aspect,
        geo_field=geo_field,
        fc_dim=int(fc_dim),
        scope=scope,
    )
    argv = build_argv(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        args = train_mod.parse_args(argv)
        train_mod.validate_override_args(args)
        cfg = train_mod.apply_overrides(train_mod.PRESETS[args.preset], args)
    return cfg


def construct_model(cfg: Any) -> Any:
    """Construct ThreeDNeuronSpace from a config (pure construction, no training)."""
    sys.path.insert(0, MODULE_DIR)
    from model import ThreeDNeuronSpace  # type: ignore

    return ThreeDNeuronSpace(cfg)


def expected_topo(
    n: int, shape: str, cyl_aspect: Optional[float], scope: str
) -> Tuple[int, int, int, int]:
    """Return the frozen expectation (E, K, S_in, S_out) for one combination and scope."""
    row = TOPOLOGY_EXPECT[(n, shape, cyl_aspect)]
    return (
        int(row[1]),
        int(row[0]),
        int(row[2]) if scope == "any_isolated" else int(row[4]),
        int(row[3]) if scope == "any_isolated" else int(row[5]),
    )


def topology_stats_of(cfg: Any) -> Tuple[int, int, int, int]:
    """Construct the model and return the measured (E, K, S_in, S_out)."""
    model = construct_model(cfg)
    stats = model.get_connection_stats()
    topo = model.get_topology_stats() or {}
    out = (
        int(stats["num_edges"]),
        int(topo["num_layers_true"]) if topo else 0,
        int(stats["num_in_scope"]),
        int(stats["num_out_scope"]),
    )
    del model
    gc.collect()
    return out


def combo_tag(
    n: int, shape: str, lam: Optional[float], scope: str = ""
) -> str:
    """Format one (shape, N, lambda, scope) combination for printing."""
    lam_s = "-" if lam is None else "{0:g}".format(lam)
    tag = "N={0:<5d} shape={1:<9s} lambda={2:<4s}".format(n, shape, lam_s)
    if scope:
        tag += " scope={0:<13s}".format(scope)
    return tag


def check_topology(state: RunState) -> Dict[str, Any]:
    """Group A: compare E / K / S_in / S_out of the 30 topology combinations with the table.

    Additionally asserts that for the same (shape, N) the any_isolated and all_isolated
    builds have identical E and K (scope must not affect topology).

    Returns
    -------
    Dict[str, Any]
        Machine-readable block (30 combination records plus a summary).
    """
    print("=" * 118)
    print("[A] --check-topology: 15 (shape, N) x 2 scope = 30 topology combinations (pure construction)")
    print("=" * 118)
    by_combo: Dict[Tuple[int, str, Optional[float]], Dict[str, Tuple[int, int, int, int]]] = {}
    for n in N_BATCHES:
        for shape, lam in SHAPE_VARIANTS:
            for scope in SCOPES:
                cfg = build_config_for(n, shape, lam, scope, fc_dim=0, geo_field="none")
                got = topology_stats_of(cfg)
                want = expected_topo(n, shape, lam, scope)
                by_combo.setdefault((n, shape, lam), {})[scope] = got
                record_pair(
                    state,
                    "topology",
                    "{0} (E, K, S_in, S_out)".format(combo_tag(n, shape, lam, scope)),
                    got,
                    want,
                    mark="A",
                )
    for (n, shape, lam), per_scope in by_combo.items():
        e_any, k_any = per_scope["any_isolated"][0], per_scope["any_isolated"][1]
        e_all, k_all = per_scope["all_isolated"][0], per_scope["all_isolated"][1]
        record_pair(
            state,
            "topology",
            "{0} any-vs-all (E, K)".format(combo_tag(n, shape, lam)),
            (e_any, k_any),
            (e_all, k_all),
            mark="A",
        )
    return {
        "n_combos": len(by_combo),
        "n_assertions": sum(1 for a in state.assertions if a.group == "topology"),
        "records": [
            {
                "N": n,
                "shape": shape,
                "cyl_aspect": lam,
                "scope": scope,
                "E": v[0],
                "K": v[1],
                "S_in": v[2],
                "S_out": v[3],
            }
            for (n, shape, lam), per_scope in by_combo.items()
            for scope, v in per_scope.items()
        ],
        "expected_source": "TOPOLOGY_EXPECT (frozen criterion table, not measurements)",
    }


def step0_geo_records(state: RunState) -> Dict[str, Any]:
    """Group B: step-0 equivalence of geo none vs additive over the 30 topology combinations.

    Criteria (3 per combination):
    1. _effective_edge_weight() is bit-identical at step 0 (torch.equal);
    2. every COMMON persistent tensor of the two builds (i.e. the key intersection of
       state_dict()) is bit-identical;
    3. the additive build really creates geo_rbf_theta (shape [k+1], all zeros) and
       geo_alpha (scalar, initial value geo_alpha_init = 1.0), and registers the
       geo_rbf_centers / geo_rbf_width persistent buffers, while the none build has
       neither geometric parameter.

    Returns
    -------
    Dict[str, Any]
        Machine-readable block (30 combination records plus a summary).
    """
    import torch

    print("=" * 118)
    print("[B] --check-step0-geo: 30 combinations (fc_dim=0), geo none vs additive at step 0")
    print("=" * 118)
    records: List[Dict[str, Any]] = []
    n_cmp = 0
    for n in N_BATCHES:
        for shape, lam in SHAPE_VARIANTS:
            for scope in SCOPES:
                tag = combo_tag(n, shape, lam, scope)
                cfg_none = build_config_for(n, shape, lam, scope, 0, "none")
                cfg_add = build_config_for(n, shape, lam, scope, 0, "additive")
                m_none = construct_model(cfg_none)
                m_add = construct_model(cfg_add)
                w_none = m_none._effective_edge_weight().detach()
                w_add = m_add._effective_edge_weight().detach()
                same_w = bool(torch.equal(w_none, w_add))
                max_abs = float((w_none - w_add).abs().max().item())
                record(
                    state,
                    "step0_geo",
                    "{0} _effective_edge_weight torch.equal".format(tag),
                    same_w,
                    "torch.equal={0} (max|dw|={1:.6e}, E={2})".format(
                        same_w, max_abs, int(w_none.numel())
                    ),
                    mark="B",
                )
                sd_none = m_none.state_dict()
                sd_add = m_add.state_dict()
                common = sorted(set(sd_none) & set(sd_add))
                unequal = [
                    k for k in common if not torch.equal(sd_none[k], sd_add[k])
                ]
                record(
                    state,
                    "step0_geo",
                    "{0} common persistent tensors bit-identical".format(tag),
                    not unequal,
                    "common keys={0}, unequal={1}{2}".format(
                        len(common),
                        len(unequal),
                        (": " + str(unequal)) if unequal else "",
                    ),
                    mark="B",
                )
                n_cmp += 1
                theta = getattr(m_add, "geo_rbf_theta", None)
                alpha = getattr(m_add, "geo_alpha", None)
                theta_zero = (
                    bool(torch.equal(theta.detach(), torch.zeros(GEO_THETA_LEN)))
                    if theta is not None
                    else False
                )
                alpha_val = (
                    float(alpha.detach().item()) if alpha is not None else None
                )
                missing_buf = [
                    k for k in GEO_PERSISTENT_BUFFERS if not hasattr(m_add, k)
                ]
                none_has_geo = hasattr(m_none, "geo_rbf_theta") or hasattr(
                    m_none, "geo_alpha"
                )
                ok_geo = (
                    theta is not None
                    and tuple(theta.shape) == (GEO_THETA_LEN,)
                    and theta_zero
                    and alpha is not None
                    and tuple(alpha.shape) == ()
                    and alpha_val is not None
                    and abs(alpha_val - GEO_ALPHA_INIT) <= 1e-6
                    and not missing_buf
                    and not none_has_geo
                )
                theta_shape = tuple(theta.shape) if theta is not None else None
                record(
                    state,
                    "step0_geo",
                    "{0} additive geo params and basis buffers".format(tag),
                    ok_geo,
                    "theta.shape={0} (want ({1},)) all-zero={2}; alpha={3} (want {4}); "
                    "missing basis buffers={5}; none build has no geo param={6}".format(
                        theta_shape,
                        GEO_THETA_LEN,
                        theta_zero,
                        alpha_val,
                        GEO_ALPHA_INIT,
                        missing_buf,
                        not none_has_geo,
                    ),
                    mark="B",
                )
                records.append(
                    {
                        "N": n,
                        "shape": shape,
                        "cyl_aspect": lam,
                        "scope": scope,
                        "E": int(w_none.numel()),
                        "effective_edge_weight_equal": same_w,
                        "max_abs_diff": max_abs,
                        "n_common_persistent_tensors": len(common),
                        "n_unequal_common": len(unequal),
                        "geo_theta_shape": list(theta.shape) if theta is not None else None,
                        "geo_theta_all_zero": theta_zero,
                        "geo_alpha": alpha_val,
                    }
                )
                del m_none, m_add, w_none, w_add, sd_none, sd_add
                gc.collect()
    return {
        "n_combos": n_cmp,
        "n_assertions": sum(1 for a in state.assertions if a.group == "step0_geo"),
        "records": records,
        "rule": (
            "theta is zero-initialised, hence w_e == w_free[e] when the switch is on, so at "
            "step 0 _effective_edge_weight() is bit-identical to the off build (torch.equal). "
            "This equivalence is the design expectation, not a relaxed criterion."
        ),
    }


# ======================================================================
# Ledger
# ======================================================================
def default_ledger_path() -> str:
    """Default ledger path (one file per date, so runs across days never overwrite each other)."""
    return os.path.join(VERIFY_DIR, "bench_ledger_{0}.json".format(today_stamp()))


def load_ledger(path: str) -> Dict[str, Any]:
    """Read the ledger; return an empty shell (meta + empty products) when absent."""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"meta": {}, "products": []}


def ensure_meta(
    ledger: Dict[str, Any], state: RunState, deletion_ref: Dict[str, Any]
) -> None:
    """Initialise / verify ledger meta (caliber drift is reported, never silently redefined)."""
    meta = ledger.setdefault("meta", {})
    cal = caliber_dict()
    if not meta.get("caliber"):
        meta["caliber"] = cal
        meta["created_at"] = utc_now()
        meta["git_head"] = git_head()
        print(
            "[ledger] initialised: {0} (git HEAD={1})".format(
                meta.get("created_at"), meta.get("git_head")
            )
        )
    else:
        same = meta.get("caliber") == cal
        detail = (
            "identical"
            if same
            else "MISMATCH: the stored ledger caliber differs from the frozen constants of "
            "this script. Changing the caliber after the fact is forbidden; point --ledger "
            "at a new file if the caliber really must change."
        )
        record(state, "ledger", "ledger caliber matches the frozen script caliber", same, detail, mark="L")
    meta["updated_at"] = utc_now()
    meta["script"] = "n3d_shape/bench_shape_modes.py"
    meta["run_rules"] = {
        "stop_on_any_failure": True,
        "atomic_ledger_refresh": "write <path>.tmp then os.replace",
        "skip_rule": "artifact exists AND ledger bytes+SHA256 match disk -> status skipped",
        "conflict_rule": "artifact exists but differs from the ledger -> stop, exit non-zero",
        "no_max_batches": "this benchmark never passes --max-batches (it would relocate artifacts)",
        "gates": "--check-topology and --check-step0-geo run before any training",
    }
    meta["invalidated_chains"] = invalidated_chains()
    meta["deletion_manifest_ref"] = deletion_ref


def verify_deletion_ref(ref: Dict[str, Any]) -> str:
    """Re-check the deletion-manifest reference (path exists plus SHA256 agrees)."""
    path = str(ref.get("path") or "")
    if not path or not os.path.isfile(path):
        return "no deletion manifest found (--wipe was not executed in this environment)"
    got = sha256_of(path)
    want = str(ref.get("sha256") or "")
    if want and got != want:
        return "SHA256 differs from the ledger record ({0} != {1})".format(
            got[:12], want[:12]
        )
    return "SHA256 agrees ({0})".format(got[:12])


# ======================================================================
# --wipe: A1 record, then A2 delete
# ======================================================================
def do_wipe(state: RunState) -> Tuple[Dict[str, Any], str]:
    """A1 + A2: record first, delete second, recreate empty directories.

    A1
    --
    Recursively enumerate every file under checkpoints/n3d_shape/ (including _verify/ and
    _verify/_void/), record {path, bytes, sha256} per entry plus the at-the-run values of the
    6 anchor artifacts, and write them to
    checkpoints/n3d_shape_deletion_manifest_<YYYYMMDD>.json (OUTSIDE the n3d_shape directory).

    A2
    --
    Delete everything under checkpoints/n3d_shape/, then recreate the empty directories
    checkpoints/n3d_shape/ and checkpoints/n3d_shape/_verify/, and assert the directory
    holds 0 files.

    Returns
    -------
    Tuple[Dict[str, Any], str]
        (deletion_manifest_ref, manifest_path).
    """
    print("=" * 118)
    print("[WIPE] A1: recursively record every file under checkpoints/n3d_shape/ (record first)")
    print("=" * 118)
    files = list_files_recursive(CHECKPOINT_DIR)
    # [!] DEF-1 (found by the independent test round): the A1 count must NOT be asserted equal
    #     to a hard-coded baseline. This driver itself writes into checkpoints/n3d_shape/_verify/
    #     (bench_dryrun_plan.txt, bench_ledger_<date>.json and one bench_log_*.txt per product),
    #     so at --wipe time the live count is 176 only on a pristine tree -- after a training batch
    #     it is roughly 176 + 120 + 2. Asserting == 176 made --wipe exit 1 although the deletion
    #     itself completed. The count is therefore REPORTED (measured) with the pristine baseline
    #     kept as a reference only; the real invariants stay hard: the manifest must be written and
    #     the directory must end up empty.
    n_recorded = len(files)
    entries: List[Dict[str, Any]] = []
    for i, p in enumerate(files, start=1):
        rel = rel_to_root(p)
        size = os.path.getsize(p)
        sha = sha256_of(p)
        entries.append(
            {"index": i, "path": rel, "abs_path": p, "bytes": size, "sha256": sha}
        )
        print("  A1[{0:03d}/{1:03d}] {2}  bytes={3}  sha256={4}".format(i, n_recorded, rel, size, sha))
    record(
        state,
        "wipe",
        "A1 recorded file count (measured; pristine-tree reference = {0})".format(
            EXPECTED_PRE_WIPE_FILES
        ),
        n_recorded > 0,
        "measured {0} files recorded (reference {1}; this driver own _verify/ artifacts are "
        "included by design, so the live count exceeds the pristine baseline after dry-run or "
        "training runs)".format(n_recorded, EXPECTED_PRE_WIPE_FILES),
        mark="W",
    )
    anchors: Dict[str, Any] = {}
    for name in ANCHOR_ARTIFACTS:
        ap = os.path.join(CHECKPOINT_DIR, name)
        if os.path.isfile(ap):
            anchors[name] = {
                "exists": True,
                "bytes": os.path.getsize(ap),
                "sha256": sha256_of(ap),
            }
            print(
                "  A1[anchor] {0}  bytes={1}  sha256={2}".format(
                    name, anchors[name]["bytes"], anchors[name]["sha256"]
                )
            )
        else:
            anchors[name] = {"exists": False, "bytes": None, "sha256": None}
            print("  A1[anchor] {0}  ** absent ** (recorded as-is, no invented old value)".format(name))
    missing_anchor = [k for k, v in anchors.items() if not v["exists"]]
    print("  A1[anchor] anchors missing: {0} of {1}{2}".format(
        len(missing_anchor), len(ANCHOR_ARTIFACTS),
        (": " + str(missing_anchor)) if missing_anchor else "",
    ))

    manifest_path = os.path.join(
        CHECKPOINTS_ROOT, "n3d_shape_deletion_manifest_{0}.json".format(today_stamp())
    )
    manifest = {
        "meta": {
            "created_at": utc_now(),
            "git_head": git_head(),
            "target_dir": rel_to_root(CHECKPOINT_DIR),
            "target_dir_abs": CHECKPOINT_DIR,
            "manifest_location_note": (
                "the manifest lives OUTSIDE the n3d_shape directory (checkpoints/ root), so "
                "it survives the A2 deletion it describes"
            ),
            "n_files_recorded": n_recorded,
            "pristine_baseline_n_files": EXPECTED_PRE_WIPE_FILES,
            "count_note": (
                "n_files_recorded is the live count at wipe time; it includes this driver own "
                "_verify/ artifacts (bench_dryrun_plan.txt, bench_ledger_<date>.json and one "
                "bench_log_*.txt per trained product), so it exceeds the pristine-tree "
                "reference pristine_baseline_n_files after any dry-run or training run."
            ),
            "caliber": caliber_dict(),
            "anchor_names": list(ANCHOR_ARTIFACTS),
            "anchors": anchors,
        },
        "files": entries,
    }
    atomic_write_json(manifest_path, manifest)
    sha_manifest = sha256_of(manifest_path)
    print("-" * 118)
    print("[WIPE] A1 manifest written: {0}".format(manifest_path))
    print("[WIPE] A1 manifest SHA256: {0}".format(sha_manifest))
    print("[WIPE] A1 recorded {0} files (pristine-tree reference {1})".format(
        n_recorded, EXPECTED_PRE_WIPE_FILES
    ))
    print("-" * 118)

    print("[WIPE] A2: delete everything under checkpoints/n3d_shape/ (with _verify/ and _verify/_void/)")
    for i, p in enumerate(files, start=1):
        if os.path.isfile(p):
            os.remove(p)
            print("  A2[{0:03d}/{1:03d}] removed {2}".format(i, n_recorded, rel_to_root(p)))
    for dirpath, dirnames, _filenames in os.walk(CHECKPOINT_DIR, topdown=False):
        for d in dirnames:
            target = os.path.join(dirpath, d)
            if os.path.isdir(target) and not os.listdir(target):
                os.rmdir(target)
                print("  A2[dir] removed empty directory {0}".format(rel_to_root(target)))
    # The count at the instant right after deletion is the asserted value (must be 0); the
    # ledger is only recreated afterwards, so it must never be folded into this number.
    n_at_deletion = count_files_recursive(CHECKPOINT_DIR)
    print("[WIPE] A2 deleted {0} files; files at the instant after deletion = {1}".format(
        n_recorded, n_at_deletion
    ))
    os.makedirs(VERIFY_DIR, exist_ok=True)
    dirs_after = sorted(
        rel_to_root(os.path.join(dp, d))
        for dp, dns, _fn in os.walk(CHECKPOINT_DIR)
        for d in dns
    )
    record(
        state,
        "wipe",
        "A2 file count at the instant after deletion == 0",
        n_at_deletion == 0,
        "measured {0} files right after deletion (expected 0); remaining subdirectories={1}".format(
            n_at_deletion, dirs_after
        ),
        mark="W",
    )
    # F3: the expected subdirectory list is derived from CHECKPOINT_DIR at the run, not from a
    # hard-coded literal, so relocating CHECKPOINT_DIR cannot produce a spurious failure.
    want_verify_rel = rel_to_root(os.path.join(CHECKPOINT_DIR, "_verify"))
    record(
        state,
        "wipe",
        "A2 _verify/ recreated and it is the only (empty) subdirectory",
        os.path.isdir(VERIFY_DIR) and dirs_after == [want_verify_rel],
        "{0} exists={1}; subdirectories={2} (want [{3}])".format(
            want_verify_rel, os.path.isdir(VERIFY_DIR), dirs_after, want_verify_rel
        ),
        mark="W",
    )
    return (
        {
            "path": manifest_path,
            "sha256": sha_manifest,
            "n_files": n_recorded,
            "n_files_at_deletion": n_at_deletion,
            "created_at": utc_now(),
        },
        manifest_path,
    )


# ======================================================================
# Artifact loading / metric collection / structural assertions
# ======================================================================
def load_checkpoint(path: str) -> Dict[str, Any]:
    """torch.load an artifact (weights_only=False: the payload carries a config dict)."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def config_from_dict(cfg_d: Dict[str, Any]) -> Any:
    """Rebuild a Config from the config dict stored inside an artifact."""
    sys.path.insert(0, MODULE_DIR)
    from config import Config  # type: ignore

    clean = {k: v for k, v in cfg_d.items() if not k.startswith("_")}
    return Config(**clean)


def model_facts(variant: Variant) -> Dict[str, Any]:
    """Rebuild the model ONCE and return both the parameter count and the buffer name sets.

    [!] F5 (fixed): the driver used to rebuild the model twice per product
    (params_from_rebuild + assert_model_buffers), and each rebuild re-computes syn_dist
    (N=1024: about 67.1M elements) plus every topology quantity. The two consumers are now
    served by a single construction, which leaves more of the 12-14 h budget to training.
    The parameter-count caliber is unchanged: it is still count_parameters() on a freshly
    constructed model (only the number of constructions changed, not the definition).

    Parameters
    ----------
    variant : Variant
        The plan entry (its config drives the construction).

    Returns
    -------
    Dict[str, Any]
        {params, n_named_buffers, named_buffer_names, non_persistent_buffers,
         syn_buffers_missing, syn_buffers_extra, geo_buffers, error}
        where params is None when the construction failed (recorded as-is, never invented).
    """
    try:
        model = construct_model(variant.config)
        n_params = int(model.count_parameters())
        named = set(n for n, _ in model.named_buffers())
        sd = model.state_dict()
        non_persistent = sorted(set(named) - set(sd))
        declared = set(SYN_BUFFER_NAMES)
        missing = sorted(declared - named)
        extra = sorted(set(non_persistent) - declared)
        geo = sorted(
            k for k in named if k.startswith("edge_geo_feat") or k.startswith("geo_rbf")
        )
        del model, sd
        gc.collect()
        return {
            "params": n_params,
            "n_named_buffers": len(named),
            "named_buffer_names": sorted(named),
            "non_persistent_buffers": non_persistent,
            "syn_buffers_missing": missing,
            "syn_buffers_extra": extra,
            "geo_buffers": geo,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - record the failure as-is
        print("      [WARN] model rebuild failed (recorded as-is): {0}".format(exc))
        return {
            "params": None,
            "n_named_buffers": None,
            "named_buffer_names": [],
            "non_persistent_buffers": [],
            "syn_buffers_missing": [],
            "syn_buffers_extra": [],
            "geo_buffers": [],
            "error": str(exc),
        }


def recorded_geo_value(record: Dict[str, Any], name: str) -> Any:
    """Read one RECORDED geometric-field quantity from a product record (B2 helper).

    B2 (this round, info): `validate_artifact` now mirrors `geo_alpha_final` and
    `geo_theta_nonzero` to the TOP LEVEL of the product record, next to the historical
    location inside the `structure` sub-block. Both generations of ledger records must read
    identically, so this helper prefers the top-level mirror and falls back to
    `structure.<name>` for records written by earlier rounds (no re-training required).

    Parameters
    ----------
    record : Dict[str, Any]
        One product record (from the ledger, or a freshly built record dict).
    name : str
        `"geo_alpha_final"` or `"geo_theta_nonzero"`.

    Returns
    -------
    Any
        The recorded value, or None when this product carries no geometric field.
    """
    value = record.get(name)
    if value is None:
        value = (record.get("structure") or {}).get(name)
    return value


def allowed_extra_non_persistent(geo_field: str) -> List[str]:
    """Return the non-persistent buffers that are LEGITIMATELY allowed beyond the 8 synapse ones.

    SPEC-ERROR 2 (spec error, fixed this round): the non-persistent set used to be asserted as EXACTLY
    the declared 8, but geo_field == "additive" legitimately registers four extra
    persistent=False feature buffers (edge_geo_feat / edge_geo_feat_raw / edge_geo_feat_min /
    edge_geo_feat_max, see model.py). The criterion is therefore: non_persistent_set -
    declared_8 == allowed_extra(geo_field), where the geo_field == "none" allowed set is EMPTY
    (so the strict "exactly equal" requirement is preserved for the none branch).

    Parameters
    ----------
    geo_field : str
        The effective geo_field of the product ("none" / "additive").

    Returns
    -------
    List[str]
        Sorted names allowed to appear in addition to the 8 synapse-class buffers.
    """
    if str(geo_field) == ADDITIVE_GEO_FIELD:
        return sorted(GEO_FEATURE_NON_PERSISTENT_BUFFERS)
    return []


def assert_model_buffers(
    variant: Variant, state: RunState, tag: str, facts: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Assert BOTH directions of the synapse-buffer name contract (F1), per geo mode (SPEC-ERROR 2).

    Positive: every declared synapse name must be present in model.named_buffers(), and the
    non-persistent set minus the declared 8 must equal the ALLOWED extras for this geo mode
    (empty for geo_field == "none", the four edge_geo_feat* feature buffers for "additive").
    Negative: none of the declared names may appear in model.state_dict().

    Parameters
    ----------
    variant : Variant
        The plan entry.
    state : RunState
        Run state (assertion records are appended here).
    tag : str
        Print prefix for this product.
    facts : Optional[Dict[str, Any]]
        Pre-computed model facts (F5: one shared rebuild). When None the model is rebuilt.

    Returns
    -------
    Dict[str, Any]
        Machine-readable evidence block.
    """
    if facts is None:
        facts = model_facts(variant)
    missing = list(facts.get("syn_buffers_missing") or [])
    extra = list(facts.get("syn_buffers_extra") or [])
    allowed = allowed_extra_non_persistent(str(variant.config.geo_field))
    unexpected = sorted(set(extra) - set(allowed))
    record(
        state,
        "structure",
        "{0} named_buffers(): the 8 declared synapse buffers exist and the only extra "
        "non-persistent buffers are the ones this geo mode allows".format(tag),
        (not missing) and (not unexpected) and (facts.get("error") is None),
        "named_buffers total={0}; declared={1}; missing={2}; extra non-persistent={3}; "
        "allowed extras for geo_field={4} -> {5}; unexpected extras={6}; rebuild error={7}".format(
            facts.get("n_named_buffers"),
            len(SYN_BUFFER_NAMES),
            missing,
            extra,
            variant.config.geo_field,
            allowed,
            unexpected,
            facts.get("error"),
        ),
        mark="P",
    )
    return {
        "n_named_buffers": facts.get("n_named_buffers"),
        "missing_syn_buffers": missing,
        "extra_non_persistent_buffers": extra,
        "allowed_extra_non_persistent_buffers": allowed,
        "unexpected_extra_non_persistent_buffers": unexpected,
        "declared_syn_buffers": list(SYN_BUFFER_NAMES),
        "non_persistent_buffers": facts.get("non_persistent_buffers"),
        "geo_buffers": facts.get("geo_buffers"),
        # F11: the same facts object is carried through so validate_artifact can read the
        # parameter count without a second rebuild (params is a real integer from the single
        # count_parameters() call, or None when the rebuild itself failed).
        "_facts": facts,
        "params": facts.get("params"),
        "rebuild_error": facts.get("error"),
    }


def assert_checkpoint_structure(
    variant: Variant,
    sd: Dict[str, Any],
    state: RunState,
    tag: str,
    facts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Structural assertions: fc / geo / nosyn / persistence caliber (hard, one by one).

    Criteria
    --------
    * fc_dim == 0: the file name carries no _fc segment and model_state_dict has no fc_ key;
    * fc_dim == -1: the name carries _fc-1 and the four keys fc_in_weight / fc_in_bias /
      fc_out_weight / fc_out_bias exist with shapes [H, 784] / [H] / [H, S_out] / [H];
    * geo_field == none: no geo_alpha / geo_rbf_theta parameter and no edge_geo_feat* or
      geo_rbf_* buffer at all;
    * geo_field == additive: geo_rbf_theta (shape [k+1], finite) and scalar geo_alpha (finite)
      exist and geo_rbf_centers / geo_rbf_width are persistent. [!] The step-0 properties
      (theta == 0, alpha == geo_alpha_init) are CONSTRUCTION invariants and are asserted by the
      --check-step0-geo gate only; on a trained product alpha_final and the theta non-zero count
      are recorded as data (they are expected to differ from the initial values);
    * every product: the name carries _nosyn (model.pt is the declared exception), the 8
      synapse-class tensors are all ABSENT from model_state_dict, neuron_pos / edge_src /
      edge_dst / edge_dist / topo_index are still present, and -- via assert_model_buffers --
      those same 8 names really exist in named_buffers() while the only extra non-persistent
      buffers are the ones the geo mode allows (empty for none, the four edge_geo_feat* for
      additive).

    Parameters
    ----------
    variant : Variant
        The plan entry.
    sd : Dict[str, Any]
        The payload model_state_dict.
    state : RunState
        Run state (assertion records are appended here).
    tag : str
        Print prefix for this product.
    facts : Optional[Dict[str, Any]]
        Pre-computed model facts (F5: one shared rebuild).

    Returns
    -------
    Dict[str, Any]
        Machine-readable structural evidence block.
    """
    import torch

    cfg = variant.config
    name = variant.artifact
    out: Dict[str, Any] = {}
    fc_keys = sorted(k for k in sd if k.startswith("fc_"))
    if int(cfg.fc_dim) == 0:
        ok = ("_fc" not in name) and (not fc_keys)
        record(
            state,
            "structure",
            "{0} fc_dim=0: no _fc segment in name, no fc_ key in state_dict".format(tag),
            ok,
            "name contains _fc={0}; fc_ keys={1}".format("_fc" in name, fc_keys),
            mark="P",
        )
        out["fc_dim_0_ok"] = ok
        out["fc_keys"] = fc_keys
    else:
        h = int(cfg.fc_width)
        s_out = expected_topo(variant.n, variant.shape, variant.cyl_aspect, variant.scope)[3]
        want_shapes = {
            "fc_in_weight": (h, int(cfg.input_dim)),
            "fc_in_bias": (h,),
            "fc_out_weight": (h, s_out),
            "fc_out_bias": (h,),
        }
        got_shapes = {k: tuple(sd[k].shape) for k in fc_keys}
        ok = (
            ("_fc-1" in name)
            and (set(fc_keys) == set(want_shapes))
            and (got_shapes == want_shapes)
        )
        record(
            state,
            "structure",
            "{0} fc_dim=-1: name carries _fc-1 and the four fc shapes are exact".format(tag),
            ok,
            "name contains _fc-1={0}; measured={1} vs expected={2}".format(
                "_fc-1" in name, got_shapes, want_shapes
            ),
            mark="P",
        )
        out["fc_dim_minus1_ok"] = ok
        out["fc_shapes"] = {k: list(v) for k, v in got_shapes.items()}

    geo_buffers = sorted(
        k for k in sd if k.startswith("edge_geo_feat") or k.startswith("geo_rbf")
    )
    if str(cfg.geo_field) == "none":
        ok = (
            ("geo_alpha" not in sd)
            and ("geo_rbf_theta" not in sd)
            and (not geo_buffers)
        )
        record(
            state,
            "structure",
            "{0} geo=none: no geometric parameter and no geometric buffer".format(tag),
            ok,
            "geo_alpha in state_dict={0}; geo_rbf_theta in state_dict={1}; geo buffers={2}".format(
                "geo_alpha" in sd, "geo_rbf_theta" in sd, geo_buffers
            ),
            mark="P",
        )
        out["geo_none_ok"] = ok
        out["geo_buffers_in_state_dict"] = geo_buffers
    else:
        # [!] SPEC-ERROR 1 (spec error, fixed this round): the initial-value properties of the
        #     geometric field (theta == 0, alpha == geo_alpha_init) hold ONLY at step 0. They are
        #     invariants of CONSTRUCTION, and they are asserted exactly where they belong -- by
        #     the --check-step0-geo gate (torch.equal on _effective_edge_weight, theta all-zero,
        #     alpha == 1.0). Asserting them on a TRAINED artifact is wrong: after 20 epochs alpha
        #     has moved (measured range [0.6332, 1.2143] over the 60 additive products, none equal
        #     to 1.0) and all 13 theta coefficients are non-zero, which is precisely the evidence
        #     that the field is learning. A trained product therefore asserts: parameters exist,
        #     shapes are exact, values are finite, and the two persistent basis buffers are
        #     present; alpha_final and the theta non-zero count are RECORDED as data.
        theta = sd.get("geo_rbf_theta")
        alpha = sd.get("geo_alpha")
        theta_finite = bool(torch.isfinite(theta).all().item()) if theta is not None else False
        theta_nonzero = (
            int(torch.count_nonzero(theta).item()) if theta is not None else None
        )
        alpha_val = float(alpha.cpu().item()) if alpha is not None else None
        alpha_finite = alpha_val is not None and math.isfinite(alpha_val)
        basis_ok = all(k in sd for k in GEO_PERSISTENT_BUFFERS)
        ok = (
            theta is not None
            and tuple(theta.shape) == (GEO_THETA_LEN,)
            and theta_finite
            and alpha is not None
            and tuple(alpha.shape) == ()
            and alpha_finite
            and basis_ok
        )
        theta_shape = tuple(theta.shape) if theta is not None else None
        record(
            state,
            "structure",
            "{0} geo=additive: theta({1}) + scalar alpha + two persistent basis buffers all "
            "present, finite; alpha/theta are RECORDED not asserted as initial values".format(
                tag, GEO_THETA_LEN
            ),
            ok,
            "theta.shape={0} (want ({1},)) finite={2} nonzero={3}/{1}; alpha={4} finite={5}; "
            "basis buffers complete={6}; NOTE step-0 properties (theta==0, alpha=={7}) are "
            "asserted by the --check-step0-geo gate, never on a trained product".format(
                theta_shape,
                GEO_THETA_LEN,
                theta_finite,
                theta_nonzero,
                alpha_val,
                alpha_finite,
                basis_ok,
                GEO_ALPHA_INIT,
            ),
            mark="P",
        )
        out["geo_additive_ok"] = ok
        out["geo_alpha_final"] = alpha_val
        out["geo_theta_nonzero"] = theta_nonzero
        out["geo_theta_shape"] = list(theta_shape) if theta_shape else None
        out["geo_initial_values_asserted_here"] = False
        out["geo_step0_properties_asserted_by"] = "--check-step0-geo gate only"

    if name == "model.pt":
        ok_name = "_nosyn" not in name
        note = (
            "model.pt exception: no _nosyn segment in the name (DEFAULT_CONFIG-identical "
            "combination), payload is still nosyn format"
        )
    else:
        ok_name = "_nosyn" in name
        note = "name carries the _nosyn format segment"
    absent_in_sd = [k for k in SYN_BUFFER_NAMES if k not in sd]
    present_keys = [k for k in PERSISTENT_KEY_NAMES if k in sd]
    ok_nosyn = (
        ok_name
        and len(absent_in_sd) == len(SYN_BUFFER_NAMES)
        and len(present_keys) == len(PERSISTENT_KEY_NAMES)
    )
    record(
        state,
        "structure",
        "{0} nosyn caliber: 8 synapse tensors absent from state_dict + 5 contract keys present".format(tag),
        ok_nosyn,
        "{0}; synapse tensors absent from state_dict={1}/{2}{3}; contract keys still in "
        "state_dict={4}/{5} {6}".format(
            note,
            len(absent_in_sd),
            len(SYN_BUFFER_NAMES),
            (" STILL PRESENT: " + str(sorted(set(SYN_BUFFER_NAMES) - set(absent_in_sd))))
            if len(absent_in_sd) != len(SYN_BUFFER_NAMES)
            else "",
            len(present_keys),
            len(PERSISTENT_KEY_NAMES),
            present_keys,
        ),
        mark="P",
    )
    out["nosyn_ok"] = bool(ok_nosyn)
    out["syn_buffers_absent_from_state_dict"] = absent_in_sd
    out["syn_buffers_still_in_state_dict"] = sorted(
        set(SYN_BUFFER_NAMES) - set(absent_in_sd)
    )
    out["persistent_keys_present"] = present_keys
    out["model_buffers"] = assert_model_buffers(variant, state, tag, facts=facts)
    # F11: surface the model-facts block (which carries the real parameter count) at the top
    # level of the structural evidence too, so BOTH consumers -- validate_artifact (record and
    # ledger path) and check_real_artifact (self-test printout) -- read the same real integer
    # instead of the evidence sub-block that never carried params.
    out["_facts"] = out["model_buffers"].get("_facts") or {}
    out["params"] = out["_facts"].get("params")
    return out


def validate_artifact(
    variant: Variant, state: RunState, tag: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load an artifact from disk, run every structural/topology assertion, report pass/fail.

    [!] F2 (blocking error, fixed): this function is the single validation entry point shared
    by the fresh-training path and the resume (skip) path. Before the fix the skip path
    trusted the ledger bytes+SHA256 alone and never re-ran the assertions, so a wrong
    assertion (or a tampered artifact whose bytes still matched the ledger) could be masked
    forever by resuming. Now nothing is ever marked skipped without a full green re-run.

    Parameters
    ----------
    variant : Variant
        The plan entry (supplies the on-disk path and the expected dimensions).
    state : RunState
        Run state; assertions are appended here and state.failed is set on any failure.
    tag : str
        Print prefix for this product.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        (record_fields, validation) where validation = {ok, n_assertions,
        n_failed_assertions, failed_details, facts}.

    B2 (this round, info)
    ---------------------
    The two RECORDED geometric-field quantities (`geo_alpha_final`, `geo_theta_nonzero`) are
    also mirrored to the TOP LEVEL of `record_fields`, in addition to their historical location
    inside `structure`. Readers therefore no longer need to know the structure sub-block layout
    (`recorded_geo_value` prefers the mirror and falls back to `structure` for older records).
    Both are RECORDED data, never criteria.
    """
    n_before = len(state.assertions)
    payload = load_checkpoint(variant.path)
    sd = dict(payload.get("model_state_dict") or {})
    cs = dict(payload.get("connection_stats") or {})
    ts = dict(payload.get("topology_stats") or {})
    got_topo = (
        int(cs.get("num_edges", -1)),
        int(ts.get("num_layers_true", -1)) if ts else -1,
        int(cs.get("num_in_scope", -1)),
        int(cs.get("num_out_scope", -1)),
    )
    want_topo = expected_topo(variant.n, variant.shape, variant.cyl_aspect, variant.scope)
    record_pair(
        state,
        "product",
        "{0} artifact E/K/S_in/S_out".format(tag),
        got_topo,
        want_topo,
        mark="P",
    )
    epochs_eff = int(payload.get("epochs", -1))
    record(
        state,
        "product",
        "{0} effective epochs == {1}".format(tag, EPOCHS),
        epochs_eff == EPOCHS,
        "measured {0}".format(epochs_eff),
        mark="P",
    )
    acc = float(payload.get("test_acc", float("nan")))
    record(
        state,
        "product",
        "{0} test_acc inside [0, 1]".format(tag),
        math.isfinite(acc) and 0.0 <= acc <= 1.0,
        "measured test_acc={0:.6f}".format(acc),
        mark="P",
    )
    struct = assert_checkpoint_structure(variant, sd, state, tag, facts=model_facts(variant))
    del payload, sd
    gc.collect()
    new = state.assertions[n_before:]
    failed = [a for a in new if not a.ok]
    if failed:
        state.failed = True
    # [!] F11 (warning, fixed): the model facts used to be read back out of
    #     struct["model_buffers"], whose evidence block never carried the parameter count, so the
    #     recorded params was ALWAYS None. The facts dict produced by model_facts() is now handed
    #     through assert_checkpoint_structure -> assert_model_buffers (one and the same object,
    #     zero extra rebuilds) and is surfaced here directly.
    facts = struct.get("_facts") or {}
    validation = {
        "ok": not failed,
        "n_assertions": len(new),
        "n_failed_assertions": len(failed),
        "failed_details": [
            {"group": a.group, "name": a.name, "detail": a.detail} for a in failed
        ],
        "facts": facts,
        "params": facts.get("params"),
        "params_source": (
            "single model rebuild count_parameters() (model_facts, F5/F11); None only when the "
            "rebuild itself failed, which also fails the named_buffers assertion and is recorded "
            "in facts.error"
        ),
        "checked_at": utc_now(),
        "check": "loaded from disk and fully re-validated (E/K/S_in/S_out, epochs, test_acc, "
        "fc, geo, nosyn presence/absence and named_buffers contract)",
    }
    if validation["params"] is None:
        # Honest, visible failure: never let a null parameter count look like a healthy record.
        record(
            state,
            "product",
            "{0} parameter count is a real integer".format(tag),
            False,
            "params=None (model rebuild failed; see validation.facts.error={0})".format(
                facts.get("error")
            ),
            mark="P",
        )
    record_fields = {
        "artifact": variant.artifact,
        "path": rel_to_root(variant.path),
        "abs_path": variant.path,
        "bytes": int(os.path.getsize(variant.path)),
        "sha256": sha256_of(variant.path),
        "test_acc": acc,
        "epochs_effective": epochs_eff,
        "E": got_topo[0],
        "K": got_topo[1],
        "S_in": got_topo[2],
        "S_out": got_topo[3],
        "params": validation["params"],
        "params_source": validation["params_source"],
        # ---- B2 (this round, info): mirror the two RECORDED geometric-field quantities to the
        #      TOP LEVEL of the product record. They already live at structure.geo_alpha_final /
        #      structure.geo_theta_nonzero, but every consumer (geo_stats, the batch summary and
        #      any external reader) would otherwise have to know the structure sub-block layout.
        #      Values are RECORDED data, never criteria (step-0 properties belong to
        #      --check-step0-geo); None for products without a geometric field.
        "geo_alpha_final": struct.get("geo_alpha_final"),
        "geo_theta_nonzero": struct.get("geo_theta_nonzero"),
        "structure": struct,
        "validation": validation,
        "exit_code": 0,
    }
    return record_fields, validation


def collect_product_metrics(
    variant: Variant,
    elapsed_s: float,
    log_path: str,
    exit_code: int,
    state: RunState,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate a freshly trained artifact and return ledger fields plus the validation block.

    F2: the record status is ok ONLY when every assertion passed; otherwise it is
    assert_failed with the failing assertion names, expected and measured values.
    """
    tag = "[{0:03d}] {1}".format(variant.index, variant.key)
    rec, validation = validate_artifact(variant, state, tag)
    rec["status"] = "ok" if validation["ok"] else "assert_failed"
    rec["assert_failed"] = int(not validation["ok"])
    if not validation["ok"]:
        rec["failure_reason"] = "assertion failure(s) recorded; see validation.failed_details"
    rec["cmd"] = variant.command
    rec["log"] = log_path
    rec["elapsed_s"] = round(float(elapsed_s), 1)
    rec["exit_code"] = int(exit_code)
    rec["finished_at"] = utc_now()
    return rec, validation


def real_artifact_variant(
    fixture_path: str, payload: Optional[Dict[str, Any]] = None
) -> Tuple[Variant, Dict[str, Any]]:
    """Build the plan Variant that describes a real artifact on disk.

    The dimensions are taken from the ARTIFACT ITSELF (its embedded config, plus an explicitly
    reported geo_field fallback for pre-round-20 products whose config has no geo_field key), so
    any real product -- additive or none, any N/shape/fc/scope -- can be fed through
    --check-real-artifact. The default fixture dimensions remain only as the no-payload fallback.

    Parameters
    ----------
    fixture_path : str
        Absolute path of the real nosyn artifact.
    payload : Optional[Dict[str, Any]]
        An already loaded payload (avoids a second torch.load). When None the artifact is loaded
        here; loading is skipped only when the caller guarantees the default fixture.

    Returns
    -------
    Tuple[Variant, Dict[str, Any]]
        (variant, derivation) where derivation records where each dimension came from.
    """
    cfg_d: Dict[str, Any] = {}
    if payload is not None:
        cfg_d = dict(payload.get("config") or {})
    elif os.path.basename(fixture_path) == REAL_ARTIFACT_FIXTURE:
        # Default fixture: its embedded config (from the round-24 rebuild) is available and is
        # what the caller passed through; falling back to the recorded dims keeps this function
        # usable standalone for that one artifact.
        cfg_d = dict(REAL_ARTIFACT_DIMS)
    d: Dict[str, Any] = dict(REAL_ARTIFACT_DIMS)
    derivation: Dict[str, Any] = {"source": "payload-config"}
    if cfg_d:
        if "N" in cfg_d:
            d["N"] = int(cfg_d["N"])
        if "shape" in cfg_d:
            d["shape"] = str(cfg_d["shape"])
        if "cyl_aspect" in cfg_d:
            d["cyl_aspect"] = (
                None if str(d["shape"]) != "cylinder" else float(cfg_d["cyl_aspect"])
            )
        if "fc_dim" in cfg_d:
            d["fc_dim"] = int(cfg_d["fc_dim"])
        if "input_scope" in cfg_d:
            d["scope"] = str(cfg_d["input_scope"])
        elif "readout_scope" in cfg_d:
            d["scope"] = str(cfg_d["readout_scope"])
    # geo_field: the payload config is authoritative when present; products that predate round 20
    # have no such key, so the artifact NAME marker is used and the source is reported explicitly
    # (never silently assumed).
    if "geo_field" in cfg_d:
        d["geo_field"] = str(cfg_d["geo_field"])
        derivation["geo_field_source"] = "payload-config"
    else:
        base = os.path.basename(fixture_path)
        if "_geo{n}_".format(n=ADDITIVE_GEO_FIELD) in base:
            d["geo_field"] = ADDITIVE_GEO_FIELD
            derivation["geo_field_source"] = (
                "filename-marker _geo{0}_ (payload config has no geo_field key)".format(
                    ADDITIVE_GEO_FIELD
                )
            )
        else:
            d["geo_field"] = "none"
            derivation["geo_field_source"] = (
                "default none (payload config has no geo_field key and the name carries no "
                "_geo..._ marker)"
            )
    derivation["geo_field_effective"] = str(d["geo_field"])
    derivation["dims"] = {
        "N": d["N"],
        "shape": d["shape"],
        "cyl_aspect": d["cyl_aspect"],
        "fc_dim": d["fc_dim"],
        "scope": d["scope"],
        "geo_field": d["geo_field"],
    }
    cfg = build_config_for(
        int(d["N"]),
        str(d["shape"]),
        d["cyl_aspect"],
        str(d["scope"]),
        int(d["fc_dim"]),
        str(d["geo_field"]),
    )
    seed = PlanSeed(
        index=0,
        n=int(d["N"]),
        shape=str(d["shape"]),
        cyl_aspect=d["cyl_aspect"],
        geo_field=str(d["geo_field"]),
        fc_dim=int(d["fc_dim"]),
        scope=str(d["scope"]),
    )
    variant = Variant(
        index=0,
        n=int(d["N"]),
        shape=str(d["shape"]),
        cyl_aspect=d["cyl_aspect"],
        geo_field=str(d["geo_field"]),
        fc_dim=int(d["fc_dim"]),
        scope=str(d["scope"]),
        key=seed_key(seed),
        config=cfg,
        artifact=os.path.basename(fixture_path),
        path=fixture_path,
        argv=build_argv(seed),
        command=[sys.executable, "-X", "utf8", TRAIN_PY] + build_argv(seed),
        is_default_path=False,
    )
    return variant, derivation


def check_real_artifact(
    state: RunState, fixture_path: str, injected_name: str = ""
) -> Dict[str, Any]:
    """F8: drive the structural assertions with a REAL _nosyn artifact and require all green.

    This is the check the earlier test round missed: it validated that the script own name
    constants were self-consistent, but never fed a real product through the assertions. Here
    the object-under-test is a genuine nosyn payload on disk, so a wrong buffer name (the
    blocking defect F1) fails immediately.

    Parameters
    ----------
    state : RunState
        Run state (assertion records are appended here).
    fixture_path : str
        Absolute path of the real nosyn artifact.
    injected_name : str
        When non-empty, this bogus name is appended to SYN_BUFFER_NAMES for this PROCESS ONLY
        (the file on disk is never modified) to demonstrate that the paired assertions fail.

    Returns
    -------
    Dict[str, Any]
        Machine-readable block: the artifact fingerprint, the payload-derived facts, the
        declared names, and whether every assertion passed.
    """
    import torch

    print("=" * 118)
    print("[C] --check-real-artifact: real _nosyn product drives the structural assertions")
    print("=" * 118)
    if not os.path.isfile(fixture_path):
        record(
            state, "real_artifact", "fixture artifact exists", False,
            "missing: {0}".format(fixture_path), mark="C",
        )
        return {"ok": False, "error": "fixture missing", "fixture": fixture_path}
    sha = sha256_of(fixture_path)
    size = os.path.getsize(fixture_path)
    print("  fixture: {0}".format(fixture_path))
    print("  fixture bytes={0} sha256={1}".format(size, sha))

    # ---- declared names: print them so the contract is visible in the evidence ----------
    print("  SYN_BUFFER_NAMES declared ({0}): {1}".format(
        len(SYN_BUFFER_NAMES), list(SYN_BUFFER_NAMES)
    ))
    if injected_name:
        print(
            "  [injection] SYN_BUFFER_NAMES + [{0}] for THIS PROCESS ONLY "
            "(the script on disk is not modified)".format(injected_name)
        )

    payload = torch.load(fixture_path, map_location="cpu", weights_only=False)
    cfg_d = dict(payload.get("config") or {})
    sd = dict(payload.get("model_state_dict") or {})
    # The variant is derived from THIS artifact (payload config + the explicit geo_field
    # fallback below), so any real product can be fed in with --real-artifact, not just the
    # default fixture. The derived dimensions are then echoed for cross-checking.
    variant, derivation = real_artifact_variant(fixture_path, payload=payload)
    checks = {
        "N": (int(cfg_d.get("N", -1)), int(variant.n)),
        "shape": (str(cfg_d.get("shape", "")), str(variant.shape)),
        "fc_dim": (int(cfg_d.get("fc_dim", 0)), int(variant.fc_dim)),
        "input_scope": (str(cfg_d.get("input_scope", "")), str(variant.scope)),
        "readout_scope": (str(cfg_d.get("readout_scope", "")), str(variant.scope)),
        "geo_field_effective": (
            str(derivation.get("geo_field_effective")), str(variant.config.geo_field)
        ),
    }
    mismatched = {k: v for k, v in checks.items() if v[0] != v[1]}
    record(
        state, "real_artifact",
        "derived variant dimensions agree with the payload config",
        not mismatched,
        "checked {0}; mismatched={1}".format(sorted(checks), mismatched), mark="C",
    )
    print(
        "  derived variant: {0}  [geo_field source: {1}]".format(
            variant.key, derivation.get("geo_field_source")
        )
    )
    if derivation.get("geo_field_source", "").startswith("filename-marker"):
        print(
            "  [note] payload config carries no geo_field key (product predates round 20); the "
            "effective geo_field was taken from the artifact NAME marker (_geoadditive_) and is "
            "reported explicitly rather than assumed."
        )
    n_before = len(state.assertions)
    flag_before = state.failed
    struct = assert_checkpoint_structure(
        variant, sd, state, "[REAL] {0}".format(variant.key)
    )
    del payload, sd
    gc.collect()
    new = state.assertions[n_before:]
    failed = [a for a in new if not a.ok]
    ok = not failed
    # F11: expose the recorded parameter count (and the exact path it would be written on)
    # so the return-path bug where params stayed null cannot recur unnoticed.
    real_params = struct.get("params")
    out = {
        "ok": ok,
        "fixture": fixture_path,
        "fixture_bytes": size,
        "fixture_sha256": sha,
        "declared_syn_buffers": list(SYN_BUFFER_NAMES),
        "injected_name": injected_name or None,
        "n_assertions": len(new),
        "n_failed": len(failed),
        "failed_details": [
            {"name": a.name, "detail": a.detail} for a in failed
        ],
        "params": real_params,
        "params_source": (
            "single model rebuild count_parameters() (model_facts); this is the SAME value the "
            "driver writes into the product record and the ledger"
        ),
        "structure": struct,
    }
    print("-" * 118)
    print(
        "  [C] assertions {0}/{1} passed; failed={2}".format(
            len(new) - len(failed), len(new), [a.name for a in failed]
        )
    )
    print(
        "  [C] recorded params (this is the value written to the product record / ledger) = {0}"
        " [source: {1}]".format(
            real_params,
            "single rebuild count_parameters()" if real_params is not None
            else "None: rebuild failed, see structure.rebuild_error={0}".format(
                struct.get("rebuild_error")
            ),
        )
    )
    if struct.get("geo_alpha_final") is not None or struct.get("geo_theta_nonzero") is not None:
        print(
            "  [C] recorded geo data (NOT asserted as initial values): geo_alpha_final={0}, "
            "geo_theta_nonzero={1}/{2}".format(
                struct.get("geo_alpha_final"),
                struct.get("geo_theta_nonzero"),
                GEO_THETA_LEN,
            )
        )
    else:
        print(
            "  [C] geo data: n/a (geo_field=none for this artifact -- no geometric field to "
            "report; step-0 geo invariants are asserted by the --check-step0-geo gate)"
        )
    print(
        "  [C] result: {0}".format(
            "ALL GREEN (real artifact drives every structural assertion)"
            if ok else "FAILED (as required for the injected-name negative proof)"
        )
    )
    print(
        "  [C] single-criterion probe: does the artifact contain a non-persistent buffer named "
        "in SYN_BUFFER_NAMES? {0}".format(
            [k for k in SYN_BUFFER_NAMES if k in struct.get("model_buffers", {}).get(
                "non_persistent_buffers", []
            )]
        )
    )
    if (not ok) and not injected_name:
        # A genuine failure must poison the run state; an injected failure is proof-only and
        # is therefore reported without marking the process as failed.
        state.failed = True
    elif injected_name:
        # Keep the process verdict clean for the injected negative proof, but remember that
        # state.failed was not tripped by the injection.
        state.failed = flag_before
    return out


# ======================================================================
# Pre-training gates
# ======================================================================
def run_gates(state: RunState, ledger: Dict[str, Any]) -> None:
    """Run --check-topology and --check-step0-geo as the pre-training gate.

    Raises
    ------
    BenchFailure
        When any gate criterion fails (the caller exits non-zero and never enters training).
    """
    topo = check_topology(state)
    geo = step0_geo_records(state)
    ledger.setdefault("topology_assertions", {}).update(topo)
    ledger.setdefault("step0_geo_equivalence", {}).update(geo)
    if state.failed:
        raise BenchFailure(
            "pre-training gate failed (--check-topology / --check-step0-geo); "
            "training is not started"
        )


# ======================================================================
# Training one product
# ======================================================================
def log_file_name(variant: Variant) -> str:
    """Per-product log name: bench_log_<NNN>_<artifact without .pt>.txt (NNN == plan index)."""
    base = variant.artifact[:-3] if variant.artifact.endswith(".pt") else variant.artifact
    return "bench_log_{0:03d}_{1}.txt".format(variant.index, base)


def train_one(
    variant: Variant,
    logs_dir: str,
    state: RunState,
    batch_size: int = 0,
    batch_label: str = "",
) -> Dict[str, Any]:
    """Train one product through a subprocess and return its ledger record.

    Parameters
    ----------
    variant : Variant
        The plan entry to train.
    logs_dir : str
        Directory holding the per-product training log.
    state : RunState
        Run state (assertion records are appended here).
    batch_size : int
        Number of products in THIS batch (F6: the progress denominator). 0 falls back to
        EXPECTED_TOTAL so the function stays usable standalone.
    batch_label : str
        Human-readable batch selector (for example batch=256, limit=4) shown in the header.

    Returns
    -------
    Dict[str, Any]
        Ledger record; status is ok only when every assertion passed (F2), otherwise
        assert_failed with the failing criteria recorded.

    Raises
    ------
    BenchFailure
        On a non-zero subprocess exit code or a missing artifact (never silently retried).
    """
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, log_file_name(variant))
    # F6: the denominator is the size of THIS batch, while the printed index stays the global
    # plan index (identical to the ledger index field and to the bench_log_<NNN> file name).
    denom = int(batch_size) if int(batch_size) > 0 else EXPECTED_TOTAL
    print(
        "[run] [{0:03d}] {1}  (this batch: {2}; selector {3})".format(
            variant.index, variant.key, denom, batch_label or "all"
        )
    )
    print("      target: {0}".format(rel_to_root(variant.path)))
    print("      command: {0}".format(" ".join(variant.command)))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("[BENCH] index={0} key={1}\n".format(variant.index, variant.key))
        fh.write("[BENCH] cwd={0}\n".format(PROJECT_ROOT))
        fh.write("[BENCH] cmd=" + " ".join(variant.command) + "\n")
        fh.write("[BENCH] start_utc={0}\n".format(utc_now()))
        fh.flush()
        proc = subprocess.run(
            variant.command,
            cwd=PROJECT_ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        rc = int(proc.returncode)
        fh.flush()
        fh.write("[BENCH] end_utc={0}\n".format(utc_now()))
        fh.write("[BENCH] exit_code={0}\n".format(rc))
    elapsed = time.perf_counter() - t0
    if rc != 0:
        raise BenchFailure(
            "training subprocess exit code {0} (no silent retry, no substitute config); "
            "log: {1}".format(rc, log_path)
        )
    if not os.path.isfile(variant.path):
        raise BenchFailure("exit code 0 but the artifact is missing: {0}".format(variant.path))
    tag = "[{0:03d}] {1}".format(variant.index, variant.key)
    rec, validation = collect_product_metrics(variant, elapsed, log_path, rc, state)
    print(
        "      -> exit={0} status={1} test_acc={2:.4f} E={3} K={4} params={5} elapsed={6}s "
        "K_gt_9={7} assertions={8} failed={9}".format(
            rc,
            rec["status"],
            rec["test_acc"],
            rec["E"],
            rec["K"],
            rec["params"],
            rec["elapsed_s"],
            int(rec["K"]) > 9,
            validation["n_assertions"],
            validation["n_failed_assertions"],
        )
    )
    if validation["n_failed_assertions"]:
        print(
            "      [FAIL] {0} assertion(s) failed for {1}; status=assert_failed (never ok)".format(
                validation["n_failed_assertions"], tag
            )
        )
        for item in validation["failed_details"]:
            print("        - {0}: {1}".format(item["name"], item["detail"]))
    return rec


# ======================================================================
# Ledger refresh / summary
# ======================================================================
def ledger_record_shell(variant: Variant) -> Dict[str, Any]:
    """Initial ledger shell for one product (status pending; every dimension sampled here)."""
    return {
        "index": variant.index,
        "key": variant.key,
        "N": variant.n,
        "shape": variant.shape,
        "cyl_aspect": variant.cyl_aspect,
        "geo_field": variant.geo_field,
        "fc_dim": variant.fc_dim,
        "scope": variant.scope,
        "artifact": variant.artifact,
        "path": rel_to_root(variant.path),
        "cmd": variant.command,
        "status": "pending",
    }


def refresh_products(ledger: Dict[str, Any], plan: Sequence[Variant]) -> None:
    """Register every plan entry in the ledger (entries already recorded are left untouched)."""
    products = ledger.setdefault("products", [])
    by_key = {p.get("key"): p for p in products}
    for v in plan:
        if v.key in by_key:
            continue
        products.append(ledger_record_shell(v))
    products.sort(key=lambda p: int(p.get("index", 0)))


def update_product(ledger: Dict[str, Any], key: str, rec: Dict[str, Any]) -> None:
    """Write one measured record back into the ledger (flushed atomically by the caller)."""
    for p in ledger.get("products", []):
        if p.get("key") == key:
            p.update(rec)
            break
    ledger.setdefault("meta", {})["updated_at"] = utc_now()


def geo_stats(ledger: Dict[str, Any], selected: Sequence[Variant]) -> Dict[str, Any]:
    """Aggregate the RECORDED post-training geometric-field quantities for the batch.

    F18: alpha_final and the theta non-zero count are data, not criteria (the step-0 values are
    asserted by the --check-step0-geo gate only). This block reports their observed range so the
    "the field really moved during training" evidence is visible in the ledger itself.

    Parameters
    ----------
    ledger : Dict[str, Any]
        The ledger (products carry structure.geo_alpha_final / structure.geo_theta_nonzero).
    selected : Sequence[Variant]
        The products of this batch.

    Returns
    -------
    Dict[str, Any]
        {n_products_with_geo, alpha_final: {min,max,mean,n}, theta_nonzero: {...},
         theta_len, per_group: {...}} or a note when no additive product is present.
    """
    by_key = {p.get("key"): p for p in ledger.get("products", [])}
    alphas: List[float] = []
    nonzeros: List[int] = []
    per_group: Dict[str, Any] = {}
    for v in selected:
        p = by_key.get(v.key, {})
        struct = p.get("structure") or {}
        # B2: read the TOP-LEVEL mirror first (written by validate_artifact this round), and
        # fall back to the structure sub-block for records written by earlier rounds -- so both
        # ledger generations aggregate identically (no re-training needed to read old records).
        a = p.get("geo_alpha_final")
        t = p.get("geo_theta_nonzero")
        if a is None:
            a = struct.get("geo_alpha_final")
        if t is None:
            t = struct.get("geo_theta_nonzero")
        if a is None and t is None:
            continue
        # B2: both read paths (top-level mirror preferred, structure sub-block fallback) must
        # return the same value for a record written by this round -- verified here so a future
        # edit that updates only one of the two locations fails loudly instead of silently
        # changing the aggregation.
        if p.get("geo_alpha_final") is not None and struct.get("geo_alpha_final") is not None:
            assert p.get("geo_alpha_final") == struct.get("geo_alpha_final"), (
                "B2 mirror mismatch for alpha_final on {0}: top-level={1} structure={2}".format(
                    v.key, p.get("geo_alpha_final"), struct.get("geo_alpha_final")
                )
            )
        if p.get("geo_theta_nonzero") is not None and struct.get("geo_theta_nonzero") is not None:
            assert p.get("geo_theta_nonzero") == struct.get("geo_theta_nonzero"), (
                "B2 mirror mismatch for theta_nonzero on {0}: top-level={1} structure={2}".format(
                    v.key, p.get("geo_theta_nonzero"), struct.get("geo_theta_nonzero")
                )
            )
        group = "N{0}|{1}|geo={2}|fc={3}|scope={4}".format(
            v.n, v.shape, v.geo_field, v.fc_dim, v.scope
        )
        slot = per_group.setdefault(
            group, {"n": 0, "alpha_final": [], "theta_nonzero": []}
        )
        slot["n"] += 1
        if a is not None:
            alphas.append(float(a))
            slot["alpha_final"].append(float(a))
        if t is not None:
            nonzeros.append(int(t))
            slot["theta_nonzero"].append(int(t))
    if not alphas and not nonzeros:
        return {
            "n_products_with_geo": 0,
            "note": "no product in this batch carries the geometric field (geo_field=none only)",
        }

    def _rng(vals: Sequence[float]) -> Dict[str, Any]:
        if not vals:
            return {"n": 0, "min": None, "max": None, "mean": None}
        return {
            "n": len(vals),
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / float(len(vals)),
        }

    group_summary = {
        g: {
            "n": d["n"],
            "alpha_final_min": min(d["alpha_final"]) if d["alpha_final"] else None,
            "alpha_final_max": max(d["alpha_final"]) if d["alpha_final"] else None,
            "theta_nonzero_min": min(d["theta_nonzero"]) if d["theta_nonzero"] else None,
            "theta_nonzero_max": max(d["theta_nonzero"]) if d["theta_nonzero"] else None,
        }
        for g, d in sorted(per_group.items())
    }
    return {
        "n_products_with_geo": max(len(alphas), len(nonzeros)),
        "geo_rbf_theta_len": GEO_THETA_LEN,
        "alpha_final": _rng(alphas),
        "theta_nonzero": _rng(nonzeros),
        "n_alpha_equal_initial": sum(1 for a in alphas if abs(a - GEO_ALPHA_INIT) <= 1e-6),
        "n_theta_all_zero": sum(1 for t in nonzeros if t == 0),
        "per_group": group_summary,
        "criterion_note": (
            "alpha_final / theta_nonzero are RECORDED data, never criteria: the step-0 properties "
            "(theta == 0, alpha == geo_alpha_init) belong to construction and are asserted by "
            "--check-step0-geo only"
        ),
    }


def batch_summary(ledger: Dict[str, Any], selected: Sequence[Variant]) -> Dict[str, Any]:
    """Build the machine-readable batch summary (status counts plus the K > 9 report).

    F2: assert_failed is a first-class status. It counts as neither ok nor pending, and it
    makes the batch fail; only ok + skipped complete a product.
    """
    by_key = {p.get("key"): p for p in ledger.get("products", [])}
    ok = skipped = pending = assert_failed = 0
    k_gt9: List[Dict[str, Any]] = []
    failed_products: List[Dict[str, Any]] = []
    for v in selected:
        p = by_key.get(v.key, {})
        st = str(p.get("status", "pending"))
        if st == "assert_failed":
            assert_failed += 1
            failed_products.append(
                {
                    "index": v.index,
                    "key": v.key,
                    "status": st,
                    "failed_assertions": (p.get("validation") or {}).get(
                        "failed_details", []
                    ),
                }
            )
            continue
        if st in ("ok", "skipped"):
            if st == "ok":
                ok += 1
            else:
                skipped += 1
            if int(p.get("K", 0)) > 9:
                k_gt9.append({"index": v.index, "key": v.key, "K": int(p["K"])})
        else:
            pending += 1
    return {
        "batch_size": len(selected),
        "n_ok": ok,
        "n_skipped": skipped,
        "n_assert_failed": assert_failed,
        "n_pending": pending,
        "n_done_or_skipped": ok + skipped,
        "assert_failed_products": failed_products,
        "geo_stats": geo_stats(ledger, selected),
        "status_note": (
            "a product counts as completed only when status is ok or skipped; assert_failed "
            "means a criterion (structure / topology / epochs / test_acc) did not hold and the "
            "batch must exit non-zero"
        ),
        "n_with_K_gt_9": len(k_gt9),
        "k_gt_9_products": k_gt9,
        "k_gt_9_note": (
            "the count of products with K > 9 is the load-bearing report: K drives both the "
            "DAG depth and the number of per-layer recurrence steps, so a larger K means a "
            "slower construction/training run. This report only counts; it changes no "
            "training caliber."
        ),
    }


def print_ledger_summary(ledger: Dict[str, Any], selected: Sequence[Variant]) -> bool:
    """Print the per-product status table and the batch summary; True only when the batch is clean.

    F2: the batch is clean when ok + skipped == batch size AND no product is assert_failed.
    """
    by_key = {p.get("key"): p for p in ledger.get("products", [])}
    print("=" * 118)
    print("per-product status of this batch (numbers sampled from the artifacts at the run)")
    print("=" * 118)
    print(
        "{0:>4s} {1:>8s} {2:>5s} {3:>9s} {4:>4s} {5:>9s} {6:>3s} {7:>13s} {8:>6s} {9:>3s} "
        "{10:>5s} {11:>6s} {12:>7s} {13:>10s}".format(
            "#", "status", "N", "shape", "lam", "geo", "fc", "scope", "E", "K", "S_in",
            "S_out", "acc", "bytes",
        )
    )
    for v in selected:
        p = by_key.get(v.key, {})
        st = str(p.get("status", "pending"))
        lam = "-" if v.cyl_aspect is None else "{0:g}".format(v.cyl_aspect)
        acc = p.get("test_acc")
        print(
            "{0:>4d} {1:>8s} {2:>5d} {3:>9s} {4:>4s} {5:>9s} {6:>3d} {7:>13s} {8:>6s} "
            "{9:>3s} {10:>5s} {11:>6s} {12:>7s} {13:>10s}".format(
                v.index,
                st,
                v.n,
                v.shape,
                lam,
                v.geo_field,
                v.fc_dim,
                v.scope,
                str(p.get("E", "-")),
                str(p.get("K", "-")),
                str(p.get("S_in", "-")),
                str(p.get("S_out", "-")),
                ("{0:.4f}".format(float(acc)) if acc is not None else "-"),
                str(p.get("bytes", "-")),
            )
        )
    summ = batch_summary(ledger, selected)
    print("-" * 118)
    print(
        "summary: batch={0}; ok={1} skipped={2} assert_failed={3} pending={4}; "
        "done+skipped={5}/{0}".format(
            summ["batch_size"],
            summ["n_ok"],
            summ["n_skipped"],
            summ["n_assert_failed"],
            summ["n_pending"],
            summ["n_done_or_skipped"],
        )
    )
    print("[load-bearing report] products with K > 9 in this batch = {0}".format(summ["n_with_K_gt_9"]))
    for item in summ["k_gt_9_products"]:
        print("  - [{0:03d}] {1}  K={2}".format(item["index"], item["key"], item["K"]))
    if summ["n_with_K_gt_9"] == 0:
        print("  (no product in this batch has K > 9: every combination has K <= 9)")
    if summ["n_assert_failed"]:
        print(
            "[FAIL] {0} product(s) recorded status=assert_failed (a criterion did not hold):".format(
                summ["n_assert_failed"]
            )
        )
        for item in summ["assert_failed_products"]:
            print("  - [{0:03d}] {1}".format(item["index"], item["key"]))
            for det in item["failed_assertions"]:
                print("      {0}: {1}".format(det.get("name"), det.get("detail")))
    gs = summ.get("geo_stats") or {}
    print(
        "[geo data] products with a geometric field = {0}; alpha_final "
        "min/max/mean = {1:.6f} / {2:.6f} / {3:.6f} (n={4}); theta nonzero "
        "min/max = {5} / {6} of {7} coefficients; alpha == initial count = {8}; theta all-zero "
        "count = {9} (RECORDED data, not criteria -- step-0 values are asserted by "
        "--check-step0-geo)".format(
            gs.get("n_products_with_geo", 0),
            (gs.get("alpha_final") or {}).get("min") or 0.0,
            (gs.get("alpha_final") or {}).get("max") or 0.0,
            (gs.get("alpha_final") or {}).get("mean") or 0.0,
            (gs.get("alpha_final") or {}).get("n", 0),
            (gs.get("theta_nonzero") or {}).get("min"),
            (gs.get("theta_nonzero") or {}).get("max"),
            gs.get("geo_rbf_theta_len"),
            gs.get("n_alpha_equal_initial"),
            gs.get("n_theta_all_zero"),
        )
    )
    # F2: a clean batch requires ok + skipped == batch size AND zero assert_failed entries.
    return (
        summ["n_done_or_skipped"] == summ["batch_size"]
        and summ["n_assert_failed"] == 0
    )


# ======================================================================
# Entry point
# ======================================================================
def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point.

    Returns
    -------
    int
        Exit code: 0 = every criterion of the requested mode holds; 1 = a failure occurred
        (non-zero training exit / topology mismatch / structural assertion failure / artifact
        conflict); 2 = invalid command line or precondition.
    """
    parser = argparse.ArgumentParser(
        description=(
            "shape x geo-field x fc-wrap x scope mode-combination benchmark driver "
            "(sequential, resumable)"
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="only enumerate the 120-product plan and assert unique artifacts; no training, no deletion",
    )
    parser.add_argument(
        "--check-topology", action="store_true",
        help="pure construction: compare E/K/S_in/S_out of the 30 topology combinations with the table",
    )
    parser.add_argument(
        "--check-step0-geo", action="store_true",
        help="pure construction: step-0 equivalence of geo none vs additive over the 30 combinations",
    )
    parser.add_argument(
        "--check-real-artifact", action="store_true",
        help=(
            "F8: drive assert_checkpoint_structure / assert_model_buffers with a REAL nosyn "
            "artifact and require every assertion to pass. Default fixture (B1): "
            "full_shapesphere_N256_..._fc-1_nosyn_s42.pt, i.e. one of THIS batch own 120 "
            "products -- the fixture must be supplied by the benchmark set itself, because "
            "--wipe removes everything under checkpoints/n3d_shape/"
        ),
    )
    parser.add_argument(
        "--real-artifact", default=REAL_ARTIFACT_FIXTURE,
        help=(
            "artifact used by --check-real-artifact/--inject-syn-name (default: this batch own "
            "product {0})".format(REAL_ARTIFACT_FIXTURE)
        ),
    )
    parser.add_argument(
        "--inject-syn-name", default="",
        help=(
            "F8 negative proof: append this bogus name to SYN_BUFFER_NAMES at RUN TIME (the file "
            "on disk is never modified) so the paired assertions must FAIL. EXIT CODE SEMANTICS "
            "(F12): in this mode exit 0 means ONLY that the negative proof held (the injected name "
            "was indeed rejected) -- it does NOT mean the assertions passed. Always read the "
            "[C] result line and the injected name; a green exit code here is the EXPECTED "
            "outcome of a FAILING assertion. Implies --check-real-artifact when used alone (F13)."
        ),
    )
    parser.add_argument(
        "--wipe", action="store_true",
        help=(
            "A1 record every file under checkpoints/n3d_shape/ into a manifest outside "
            "n3d_shape (the live MEASURED entry count is reported; the pristine-tree value 176 "
            "is only a reference baseline), then A2 delete and recreate the empty directories. "
            "--batch is ignored in this phase (the deletion scope is the whole tree)"
        ),
    )
    parser.add_argument(
        "--batch", choices=["256", "512", "1024", "all"], default=None,
        help=(
            "run only the given N batch (default None = all 120). Ignored by --wipe, whose "
            "deletion scope is the entire checkpoints/n3d_shape/ tree"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="run only the first N products (default 0 = unlimited; for driver self-check)",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help=(
            "stop at the FIRST failing product instead of finishing the batch. Default "
            "behaviour (F17): keep going through the whole batch, then exit non-zero and list "
            "every failing product and its failing criteria at the end"
        ),
    )
    parser.add_argument(
        "--ledger", default="",
        help="ledger path (default _verify/bench_ledger_<date>.json)",
    )
    parser.add_argument(
        "--logs-dir", default="",
        help="per-product training log directory (default _verify/)",
    )
    args = parser.parse_args(argv)

    # F8 negative proof support: --inject-syn-name may extend the declared name tuple for THIS
    # PROCESS ONLY. The script on disk is never modified, so its SHA256 is unchanged by the
    # proof and cannot drift.
    global SYN_BUFFER_NAMES

    state = RunState()
    ledger_path = args.ledger or default_ledger_path()
    logs_dir = args.logs_dir or VERIFY_DIR
    pre_files = count_files_recursive(CHECKPOINT_DIR)
    print("=" * 118)
    print("n3d_shape mode-combination benchmark driver (120 = 5 shapes x 2 geo x 2 fc x 2 scope x 3 N)")
    print("project root: {0}".format(PROJECT_ROOT))
    print(
        "caliber: preset={0} / epochs={1} / seed={2} / threads={3} / H=D={4:g} / arch={5} / "
        "never passes --max-batches".format(PRESET, EPOCHS, SEED, THREADS, H, ARCH)
    )
    print(
        "files under checkpoints/n3d_shape/ on entry = {0} (baseline {1})".format(
            pre_files, EXPECTED_PRE_WIPE_FILES
        )
    )
    print("=" * 118)

    ledger = load_ledger(ledger_path)
    deletion_ref = dict((ledger.get("meta") or {}).get("deletion_manifest_ref") or {})

    if args.wipe:
        print("[WIPE] this mode ONLY deletes: A1 record -> A2 delete -> recreate empty directories.")
        print("[WIPE] the real long run (--batch all) is launched separately as a background job.")
        # ---- F7: --wipe scope vs --batch ------------------------------------------------
        # The deletion scope is the WHOLE checkpoints/n3d_shape/ tree, so --batch cannot
        # narrow it. Before the fix the flag was silently ignored; now the interaction is
        # stated explicitly (and --batch default is None so an explicit value is visible).
        if args.batch is not None:
            print(
                "[WIPE] note: --batch {0} is IGNORED during the wipe phase -- the deletion "
                "scope is the entire {1} tree (it cannot be narrowed by N).".format(
                    args.batch, rel_to_root(CHECKPOINT_DIR)
                )
            )
        else:
            print(
                "[WIPE] note: the deletion scope is the entire {0} tree (--batch does not "
                "narrow it; it only selects which N batch a later TRAINING run processes).".format(
                    rel_to_root(CHECKPOINT_DIR)
                )
            )
        ref, manifest_path = do_wipe(state)
        # The A2 deletion removed the ledger as well (it lives under the wiped directory), so
        # re-read from disk: absent means an empty shell, which is the expected outcome.
        ledger = load_ledger(ledger_path)
        ledger.setdefault("meta", {})["deletion_manifest_ref"] = ref
        ensure_meta(ledger, state, ref)
        refresh_products(ledger, build_plan(enumerate_seeds()))
        ledger["last_run"] = {
            "mode": "wipe",
            "finished_at": utc_now(),
            "assertions": [a.__dict__ for a in state.assertions],
        }
        atomic_write_json(ledger_path, ledger)
        n_now = count_files_recursive(CHECKPOINT_DIR)
        print("-" * 118)
        print(
            "[WIPE] done: files before deletion = {0} (A1 recorded {1}), files at the instant "
            "after deletion = {2} (A2 criterion: must be 0), files now = {3} (the ledger and the "
            "manifest rewrite account for the difference after the deletion)".format(
                pre_files,
                ref["n_files"],
                ref.get("n_files_at_deletion"),
                n_now,
            )
        )
        print("[WIPE] manifest: {0}".format(manifest_path))
        print("[WIPE] ledger: {0}".format(ledger_path))
        return 1 if state.failed else 0

    os.makedirs(VERIFY_DIR, exist_ok=True)
    ensure_meta(ledger, state, deletion_ref)
    full_plan = build_plan(enumerate_seeds())

    names = [v.artifact for v in full_plan]
    paths = [v.path for v in full_plan]
    plan_ok = len(set(names)) == EXPECTED_TOTAL and len(set(paths)) == EXPECTED_TOTAL
    detail = "measured {0}/{1} unique artifacts".format(len(set(names)), EXPECTED_TOTAL)
    if not plan_ok:
        detail += "; collisions={0}".format(
            sorted({n for n in names if names.count(n) > 1})
        )
    record(
        state,
        "plan",
        "artifact names / target paths {0}/{0} unique".format(EXPECTED_TOTAL),
        plan_ok,
        detail,
        mark="0",
    )
    n_default = sum(1 for v in full_plan if v.is_default_path)
    # [!] The task brief asserted that "the combination identical to DEFAULT_CONFIG uses the
    #     checkpoints/n3d_shape/model.pt path". This is only true when the run omits --epochs:
    #     is_default_config() compares config.to_dict() INCLUDING epochs, and the frozen
    #     caliber passes --epochs 20 while DEFAULT_CONFIG.epochs == 10, so no combination in
    #     this plan can reach the model.pt branch (probe: is_default_config=False at
    #     --epochs 20; resolve_checkpoint_path returns the full_* fingerprint name).
    #     Reporting this as an assertion means any drift in that naming path is still caught;
    #     the count itself is whatever the naming function really produced at the run.
    record(
        state,
        "plan",
        "DEFAULT_CONFIG path (model.pt) products: {0} of {1}".format(
            n_default, EXPECTED_TOTAL
        ),
        True,
        "measured {0}; model.pt entries={1}; note=with the frozen caliber (--epochs 20) every "
        "product carries a full_* fingerprint name because is_default_config compares epochs "
        "too (DEFAULT_CONFIG.epochs={2}), hence model.pt is unreachable in this batch".format(
            n_default,
            [v.artifact for v in full_plan if v.is_default_path],
            int(load_train_module().DEFAULT_CONFIG.epochs),
        ),
        mark="0",
    )
    record(
        state,
        "plan",
        "deletion-manifest reference status",
        True,
        verify_deletion_ref(deletion_ref),
        mark="0",
    )
    refresh_products(ledger, full_plan)
    ledger["last_run"] = {
        "mode": "plan",
        "args": vars(args),
        "finished_at": utc_now(),
        "assertions": [a.__dict__ for a in state.assertions],
    }
    atomic_write_json(ledger_path, ledger)

    selected = filter_plan(full_plan, args.batch, args.limit)

    if args.dry_run:
        print("=" * 118)
        print(
            "[dry-run] plan holds {0} products; listing {1} (batch={2}, limit={3}); "
            "no training, no deletion".format(
                len(full_plan), len(selected), args.batch, args.limit
            )
        )
        print("=" * 118)
        lines: List[str] = []
        for v in selected:
            lam = "-" if v.cyl_aspect is None else "{0:g}".format(v.cyl_aspect)
            head = (
                "[{0:03d}] N={1:<5d} shape={2:<9s} lambda={3:<4s} geo={4:<9s} fc={5:<3d} "
                "scope={6:<13s}".format(
                    v.index, v.n, v.shape, lam, v.geo_field, v.fc_dim, v.scope
                )
            )
            print(head)
            print("      target path: {0}".format(rel_to_root(v.path)))
            print("      command    : {0}".format(" ".join(v.command)))
            lines.append(head)
            lines.append("      target path: {0}".format(rel_to_root(v.path)))
            lines.append("      command    : {0}".format(" ".join(v.command)))
        plan_txt = os.path.join(VERIFY_DIR, "bench_dryrun_plan.txt")
        atomic_write_text(plan_txt, "\n".join(lines) + "\n")
        ledger["last_run"] = {
            "mode": "dry-run",
            "args": vars(args),
            "finished_at": utc_now(),
            "n_listed": len(selected),
            "plan_file": plan_txt,
            "assertions": [a.__dict__ for a in state.assertions],
        }
        atomic_write_json(ledger_path, ledger)
        print("-" * 118)
        print("[dry-run] plan file written: {0}".format(plan_txt))
        print("[dry-run] artifact uniqueness: {0}/{1}".format(len(set(names)), EXPECTED_TOTAL))
        print(
            "[dry-run] no training was executed and nothing was deleted (files under "
            "checkpoints/n3d_shape/: {0} -> {1})".format(
                pre_files, count_files_recursive(CHECKPOINT_DIR)
            )
        )
        return 1 if state.failed else 0

    if args.check_real_artifact or args.inject_syn_name:
        # ---- F8: real-artifact-driven assertions (positive), plus the injected-name proof ----
        injected = str(args.inject_syn_name or "")
        # F13: a bare --inject-syn-name silently entered this self-test branch before. Say so.
        if injected and not args.check_real_artifact:
            print(
                "[mode] inject mode WITHOUT --check-real-artifact: entering the real-artifact "
                "self-test, NOT training (--inject-syn-name implies --check-real-artifact)."
            )
        if injected:
            # F12: in this branch exit 0 means the negative proof held, not that assertions passed.
            print(
                "[mode] negative-proof mode: the injected name [{0}] is appended to "
                "SYN_BUFFER_NAMES for THIS PROCESS ONLY (the script on disk is unchanged). "
                "Exit code 0 here means the negative proof HELD (the bogus name was rejected); "
                "it does NOT mean the assertions passed -- read the [C] result line below.".format(
                    injected
                )
            )
            SYN_BUFFER_NAMES = tuple(SYN_BUFFER_NAMES) + (injected,)
        fixture = args.real_artifact
        if not os.path.isabs(fixture):
            if os.sep in fixture or "/" in fixture:
                fixture = os.path.join(PROJECT_ROOT, fixture)
            else:
                fixture = os.path.join(CHECKPOINT_DIR, fixture)
        fx_sha_before = sha256_of(fixture) if os.path.isfile(fixture) else None
        real = check_real_artifact(state, fixture, injected_name=injected)
        fx_sha_after = sha256_of(fixture) if os.path.isfile(fixture) else None
        record(
            state,
            "real_artifact",
            "fixture artifact bytes+SHA256 unchanged by this self-test",
            (fx_sha_before is not None) and (fx_sha_before == fx_sha_after),
            "before={0} after={1}".format(
                (fx_sha_before or "missing")[:16], (fx_sha_after or "missing")[:16]
            ),
            mark="C",
        )
        if injected:
            record(
                state,
                "real_artifact",
                "injected bogus name [{0}] makes the paired assertions FAIL (negative proof)".format(
                    injected
                ),
                not real["ok"],
                "failed_assertions={0}; declared={1}".format(
                    [d["name"] for d in real.get("failed_details", [])],
                    list(SYN_BUFFER_NAMES),
                ),
                mark="C",
            )
        else:
            record(
                state,
                "real_artifact",
                "real artifact drives every structural assertion ALL GREEN",
                bool(real["ok"]),
                "assertions={0} failed={1}".format(
                    real.get("n_assertions"), real.get("n_failed")
                ),
                mark="C",
            )
        ledger["last_run"] = {
            "mode": "check-real-artifact",
            "args": vars(args),
            "finished_at": utc_now(),
            "injected_name": injected or None,
            "real_artifact": real,
            "assertions": [a.__dict__ for a in state.assertions],
        }
        atomic_write_json(ledger_path, ledger)
        n_pass = sum(1 for a in state.assertions if a.ok)
        n_all = len(state.assertions)
        print("-" * 118)
        print(
            "[real-artifact] assertions {0}/{1} passed (failed {2}); injected={3}".format(
                n_pass, n_all, n_all - n_pass, injected or "(none)"
            )
        )
        print(
            "[real-artifact] files under checkpoints/n3d_shape/: {0} -> {1} (this mode trains "
            "nothing and deletes nothing)".format(
                pre_files, count_files_recursive(CHECKPOINT_DIR)
            )
        )
        return 1 if state.failed else 0

    if args.check_topology or args.check_step0_geo:
        if args.check_topology:
            ledger.setdefault("topology_assertions", {}).update(check_topology(state))
        if args.check_step0_geo:
            ledger.setdefault("step0_geo_equivalence", {}).update(step0_geo_records(state))
        ledger["last_run"] = {
            "mode": "checks",
            "args": vars(args),
            "finished_at": utc_now(),
            "n_assertions": len(state.assertions),
            "assertions": [a.__dict__ for a in state.assertions],
        }
        atomic_write_json(ledger_path, ledger)
        n_pass = sum(1 for a in state.assertions if a.ok)
        n_all = len(state.assertions)
        print("-" * 118)
        print("[checks] assertions passed {0}/{1} (failed {2})".format(n_pass, n_all, n_all - n_pass))
        print(
            "[checks] files under checkpoints/n3d_shape/: {0} -> {1} (this mode trains nothing "
            "and deletes nothing)".format(pre_files, count_files_recursive(CHECKPOINT_DIR))
        )
        return 1 if state.failed else 0

    # ---- training mode ----
    try:
        run_gates(state, ledger)
        ledger["last_run"] = {
            "mode": "gates",
            "finished_at": utc_now(),
            "assertions": [a.__dict__ for a in state.assertions],
        }
        atomic_write_json(ledger_path, ledger)
        n_pass = sum(1 for a in state.assertions if a.ok)
        print(
            "[gate] --check-topology + --check-step0-geo passed (assertions {0}/{1}); "
            "entering training".format(n_pass, len(state.assertions))
        )
    except BenchFailure as exc:
        ledger["last_run"] = {
            "mode": "gates",
            "finished_at": utc_now(),
            "failure": str(exc),
            "assertions": [a.__dict__ for a in state.assertions],
        }
        atomic_write_json(ledger_path, ledger)
        print("-" * 118)
        print("[FAIL] {0}".format(exc))
        return 1

    by_key = {p.get("key"): p for p in ledger.get("products", [])}
    batch_label = "batch={0}, limit={1}".format(args.batch, args.limit)
    print("=" * 118)
    print(
        "[train] batch of {0} products ({1}); ledger {2}".format(
            len(selected), batch_label, ledger_path
        )
    )
    print(
        "[train] progress index = GLOBAL plan index (same as the ledger index field and the "
        "bench_log_<NNN> file name); the printed batch size is the size of THIS batch (F6)"
    )
    print("=" * 118)
    try:
        # F17: count failures of THIS batch locally. state.failed is a process-wide sticky flag
        # (a gate failure would set it), so it must not be the sole basis for the batch verdict.
        n_batch_failed = 0
        n_resumed = 0
        for v in selected:
            prev = by_key.get(v.key, {})
            if os.path.isfile(v.path):
                sha = sha256_of(v.path)
                size = os.path.getsize(v.path)
                rec_bytes = prev.get("bytes")
                rec_sha = prev.get("sha256")
                matches = bool(
                    rec_sha
                    and rec_bytes is not None
                    and int(rec_bytes) == int(size)
                    and rec_sha == sha
                )
                if matches:
                    # ---- F2: the skip path must NEVER trust the ledger alone -------------
                    # Re-run every structural/topology assertion against the artifact on disk.
                    # A previous run may have used a wrong criterion (that is exactly how the
                    # buffer-name defect stayed invisible), or the artifact may have been
                    # replaced by a byte-identical ledger entry from elsewhere; resuming must
                    # not be able to mask either case forever.
                    tag = "[{0:03d}] {1}".format(v.index, v.key)
                    prev_status = str(prev.get("status"))
                    print(
                        "[resume] [{0:03d}] {1} -> artifact exists and bytes/SHA256 match the "
                        "ledger ({2}); no training, re-running every assertion only".format(
                            v.index, v.key, sha[:12]
                        )
                    )
                    fields, validation = validate_artifact(v, state, tag)
                    # Resume semantics: a product whose assertions now pass is recorded as ok
                    # (identical to a freshly validated product); products that fail keep the
                    # assert_failed status. Nothing is ever marked skipped without a green
                    # re-run, and no bytes are rewritten.
                    fields["status"] = "ok" if validation["ok"] else "assert_failed"
                    fields["assert_failed"] = int(not validation["ok"])
                    fields["revalidated_at"] = utc_now()
                    fields["resumed"] = True
                    fields["resumed_from_status"] = prev_status
                    fields["trained_this_run"] = False
                    fields["cmd"] = prev.get("cmd", v.command)
                    fields["log"] = prev.get("log", "")
                    if not validation["ok"]:
                        fields["failure_reason"] = (
                            "resume re-validation failed (previous status was {0}); "
                            "see validation.failed_details".format(prev_status)
                        )
                        for item in validation["failed_details"]:
                            print("        - {0}: {1}".format(item["name"], item["detail"]))
                    update_product(ledger, v.key, fields)
                    atomic_write_json(ledger_path, ledger)
                    by_key[v.key] = dict(fields)
                    n_resumed += 1
                    if not validation["ok"]:
                        n_batch_failed += 1
                        if args.fail_fast:
                            raise BenchFailure(
                                "[--fail-fast] re-validation of [{0:03d}] {1} failed (status was "
                                "{2}); stopping at the first failure. Failing criteria: {3}".format(
                                    v.index,
                                    v.key,
                                    prev_status,
                                    [d["name"] for d in validation["failed_details"]],
                                )
                            )
                    print(
                        "          -> status={0} (was {1}) assertions={2} failed={3} params={4}"
                        " alpha_final={5} theta_nonzero={6}".format(
                            fields["status"],
                            prev_status,
                            validation["n_assertions"],
                            validation["n_failed_assertions"],
                            fields.get("params"),
                            recorded_geo_value(fields, "geo_alpha_final"),
                            recorded_geo_value(fields, "geo_theta_nonzero"),
                        )
                    )
                    continue
                raise BenchFailure(
                    "artifact already exists but differs from the ledger (silent overwrite is "
                    "forbidden):\n    path: {0}\n    on disk : bytes={1} sha256={2}\n    ledger  : "
                    "bytes={3} sha256={4} (status={5})\n    action: confirm the cause first "
                    "(for example a re-run after --wipe, or an externally modified artifact); "
                    "this driver deletes nothing, overwrites nothing and skips nothing.".format(
                        v.path, size, sha, rec_bytes, rec_sha, prev.get("status")
                    )
                )
            rec = train_one(
                v, logs_dir, state, batch_size=len(selected), batch_label=batch_label
            )
            update_product(ledger, v.key, rec)
            atomic_write_json(ledger_path, ledger)
            by_key[v.key] = dict(rec)
            # F17: default = keep processing the rest of the batch, then exit non-zero and list
            # every failing product; --fail-fast = stop at the first failure.
            if str(rec.get("status")) == "assert_failed":
                n_batch_failed += 1
                if args.fail_fast:
                    raise BenchFailure(
                        "[--fail-fast] product [{0:03d}] {1} recorded status=assert_failed; "
                        "stopping at the first failure. Failing criteria: {2}".format(
                            v.index,
                            v.key,
                            [
                                d.get("name")
                                for d in (rec.get("validation") or {}).get(
                                    "failed_details", []
                                )
                            ],
                        )
                    )
        if n_batch_failed:
            failed_keys = [
                p.get("key")
                for p in ledger.get("products", [])
                if str(p.get("status")) == "assert_failed"
                and p.get("key") in {v.key for v in selected}
            ]
            raise BenchFailure(
                "{0} product(s) recorded status=assert_failed (never ok). Default semantics "
                "(F17): the whole batch was processed first, and the failing products are listed "
                "in the summary below; use --fail-fast to stop at the first failure. Failing "
                "products: {1}".format(n_batch_failed, failed_keys)
            )
    except KeyboardInterrupt:
        ledger["last_run"] = {
            "mode": "train",
            "args": vars(args),
            "interrupted_at": utc_now(),
            "assertions": [a.__dict__ for a in state.assertions],
        }
        atomic_write_json(ledger_path, ledger)
        print("-" * 118)
        print(
            "[interrupt] Ctrl-C received; the ledger was flushed. Finished products are skipped "
            "on the next run only after both the ledger and the on-disk SHA256 agree."
        )
        return 130
    except BenchFailure as exc:
        ledger["last_run"] = {
            "mode": "train",
            "args": vars(args),
            "finished_at": utc_now(),
            "failure": str(exc),
            "assertions": [a.__dict__ for a in state.assertions],
        }
        atomic_write_json(ledger_path, ledger)
        print("-" * 118)
        print("[FAIL] {0}".format(exc))
        return 1

    all_done = print_ledger_summary(ledger, selected)
    ledger.setdefault("summary", {})["last_batch"] = batch_summary(ledger, selected)
    ledger["last_run"] = {
        "mode": "train",
        "args": vars(args),
        "finished_at": utc_now(),
        "n_assertions": len(state.assertions),
        "n_failed_assertions": sum(1 for a in state.assertions if not a.ok),
        "assertions": [a.__dict__ for a in state.assertions],
    }
    atomic_write_json(ledger_path, ledger)
    print("-" * 118)
    print("[ledger] {0}".format(ledger_path))
    print(
        "[check] files under checkpoints/n3d_shape/: {0} -> {1}".format(
            pre_files, count_files_recursive(CHECKPOINT_DIR)
        )
    )
    if not all_done:
        print("[FAIL] this batch still has pending products")
        return 1
    print("[PASS] every product of this batch finished (ok or skipped)")
    return 0


if __name__ == "__main__":
    # The Windows console may default to GBK: force stdout/stderr to UTF-8 with errors=replace,
    # the same caliber as train.py _reconfigure_stdio, so no non-GBK character crashes the entry point.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            pass
    raise SystemExit(main())
