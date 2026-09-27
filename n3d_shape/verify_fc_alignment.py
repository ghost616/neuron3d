"""fc_dim 同参对齐的**现跑复核**脚本（只读产物与台账，不写任何 checkpoint）。

职责
----
A. **fc_dim 语义与关闭路径零回归**（预先定义的硬断言）：
   * `fc_dim = -1` -> 两端有效宽度 `== N`；`> 0` -> `== 该值`；`0` -> 关闭（宽度 0）；
   * `fc_dim < -1` 在 `Config` 构造期被拒；
   * `fc_dim == 0` 时**不创建任何 FC 参数 / buffer**，且模型逐张量与**改动前快照**
     (`_verify/fc_pre_change_snapshot.json`，由改动前的源码现场生成) **逐位一致**；
   * `fc_dim == 0` 的产物名**不含** `_fc` 段（既有命名规则逐字不变），
     `fc_dim != 0` 的产物名**含** `_fc{n}` 段且与关闭路径不同名；
   * 解析式参数量公式 == `count_parameters()`；`count_dense_weight_tensors() == 0`。
B. **台账逐条复核**：产物存在、磁盘 SHA256 == 台账、产物 `config` 与固定口径一致、
   用**现场重建模型**独立复算 `params` / `E` / `K` / `|S_in|` / `|S_out|`、
   **同参偏差 <= 0.5% 断言**、产物名两两唯一且与既有产物零冲突。
C. **搜索块复核**：逐 seed 的 `chosen_N` 必须等于其 17 行实测表中 `argmin |params - target|`
   （含"记录的表"与"现场重建该 N 的实测值"两重核对）。
D. **零回归**：33 个既有产物（10 个三期 + 22 个 N 阶梯 + 1 个一期 MLP）SHA256 的
   **代码内冻结常量**承重断言。

用法
----
    python n3d_shape/verify_fc_alignment.py     # 退出码 0 = 全部通过；1 = 存在不一致

说明
----
本脚本**不 import** `run_fc_alignment.py`：固定口径与冻结常量都由本脚本**独立复述**，
避免"校验脚本与被测实现同源"的自洽性陷阱。
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, os.pardir))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "n3d_shape")
VERIFY_DIR = os.path.join(CHECKPOINT_DIR, "_verify")
LEDGER = os.path.join(VERIFY_DIR, "fc_alignment_runs.json")
SNAPSHOT = os.path.join(VERIFY_DIR, "fc_pre_change_snapshot.json")

# ----------------------------------------------------------------------
# 固定口径（**独立复述**；与 README §18 的表头一一对应）
# ----------------------------------------------------------------------
EXPECT_CONFIG: Dict[str, Any] = {
    "shape": "sphere",
    "flow_axis": "z",
    "placement": "fcc",
    "H": 0.10,
    "D": 0.10,
    "y_in": 8,
    "y_out": 8,
    "input_dim": 784,
    "output_dim": 10,
    "hidden_dim": 2048,
    "num_workers": 0,
    "input_scope": "any_isolated",
    "readout_scope": "any_isolated",
    "epochs": 20,
    "batch_size": 128,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "lr_schedule": "cosine",
    "grad_clip": 1.0,
    "readout_bias": True,
    "cyl_aspect": 1.0,
}
EXPECT_PRESET = "highacc"
FC_DIM = -1
N_SEARCH_LO = 815
N_SEARCH_HI = 831
TARGET_PARAMS = 1628170
DEV_TOL_PCT = 0.5
SEEDS: Tuple[int, ...] = (1, 2, 3, 7, 42, 43, 99, 123, 2024)
N3D_TAG = "fc_align"
MLP_TAG = "fc_align_mlp"
FLOAT_TOL = 1e-12

# 冻结常量：既有 33 个产物（10 个三期 + 22 个 N 阶梯 + 1 个一期 MLP）的 SHA256
# —— 承重断言：任一新轮次若覆盖了既有产物，这里立刻 FAIL。
FROZEN_BASELINE: List[Tuple[str, str]] = [
    (r"..\_control\mlp_highacc_ep12_seed42.pt", "f261716e7f3c3cba8729e46829714e9c981d5d1a896b74fd7c4d112d80a8b571"),
    ("full_shapecube_N1024_y8x8_H0.1_D0.0655_plfcc_axz_isall_rsall_s42.pt", "2a94103c8cf938ee133d434de3133d9afd4854c2c80745dfadddec3f1e3e7744"),
    ("full_shapecube_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt", "9d11e7dd07c10c9e6d7ffc35785a857b7b7aa08c4e1f51a76cf7f3f9991ad06d"),
    ("full_shapecylinder_a0.5_N1024_y8x8_H0.1_D0.069_plfcc_axz_isall_rsall_s42.pt", "67478e5ef9c5cd0b68615691acccd5e50742fd56cc0abc7ade4555857a3433a2"),
    ("full_shapecylinder_a0.5_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt", "9ca88bacb3a0e50b1a30523fb134e6f9b2725fb08df60c146736b03c52a06534"),
    ("full_shapecylinder_a1_N1024_y8x8_H0.1_D0.065531_plfcc_axz_isall_rsall_s42.pt", "6216f6403b20a31fd49138c5fc5a25ffce5cb7a144ec2e02c4202a145acb433e"),
    ("full_shapecylinder_a1_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt", "f18fff8b34c858ffed94ea56827690266b97adc42501989f8c0c18229d16f48c"),
    ("full_shapecylinder_a2_N1024_y8x8_H0.1_D0.064969_plfcc_axz_isall_rsall_s42.pt", "15a33badd043c917b98c4f934846eb5350e3771591fa814c34160bbd89ebbbda"),
    ("full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt", "05c70a9df2611735328710d926320f618f2a2cb3cd679c489ce6eef38a8f9f81"),
    ("full_shapesphere_N1024_y8x8_H0.1_D0.064996_plfcc_axz_isall_rsall_s42.pt", "beff8ce47fb0fb2503e6eeb607dbd1ba1f374bf526673423aa54eaf7b945a3e1"),
    ("full_shapesphere_N1024_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42_ladder_hacc.pt", "248cd9c5bf05e843ee6e928472a0864b10172fb079b99febd37854b9c5dcf611"),
    ("full_shapesphere_N2048_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s123_ladder_hacc.pt", "70f69600a80a3e9a21d34a6d423eb55783e468009ea4252a1e165d14805b2ac2"),
    ("full_shapesphere_N2048_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s1_ladder_hacc.pt", "c00226be36176b1e4eeabdfc6d6032a3f8035efe9d0c416c60104a92a3c7bd4f"),
    ("full_shapesphere_N2048_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s2024_ladder_hacc.pt", "29f83ba2daaaace3995d1a299537d84c2ae53f1d8197cc041180af408536691c"),
    ("full_shapesphere_N2048_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s2_ladder_hacc.pt", "8b8b1e79a1d590cec0370237ea09874e062625719ca4452f5ac23d65a1c2f512"),
    ("full_shapesphere_N2048_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s3_ladder_hacc.pt", "979e24c118fd62b68f1bffeebe085bbf768766eec7f4b8c19be2ce478dc0e25e"),
    ("full_shapesphere_N2048_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42_ladder_hacc.pt", "73fb728259c760d3ab4fe957938c18b7c6438c93d2323a6dad2bc0d297930d0e"),
    ("full_shapesphere_N2048_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s43_ladder_hacc.pt", "27a60fe252ce77fb24a3d33f5e13e3d5aa6befa009c1d22a0c4d7967ff2aafe2"),
    ("full_shapesphere_N2048_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s7_ladder_hacc.pt", "64ef3b1093bc11f01d9f4624afb9ea9678778f80d03eb27e70864e9d2ddfb4ea"),
    ("full_shapesphere_N2048_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s99_ladder_hacc.pt", "21539cf2342bb815db46b14f47eae5e7246ebf0f71b61bd739ced3e921480cea"),
    ("full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s123_ladder_hacc_mlp.pt", "8a7b213498fc82dbc191efa1597af1cc3a782d4a1ecd50475684063403c8a1c2"),
    ("full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s1_ladder_hacc_mlp.pt", "792cccfb1ddb183297ddbc40ec34f034c64bf1c2bcef70b9fffec4e5eaa4254c"),
    ("full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s2024_ladder_hacc_mlp.pt", "89eaa6e28e7fefa3063fc1eac6bea0c312b9e260395095d13fe66418c62d74d5"),
    ("full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s2_ladder_hacc_mlp.pt", "045038d0be85a86f6fb7c71a273e5628190287a3bf5d46ace393e915b500ba2f"),
    ("full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s3_ladder_hacc_mlp.pt", "11292948b8ed89fedf73d62571e537f68abcb8e0c20de312d50cea888489a560"),
    ("full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt", "c96a25d2e8bc86ece35ec61303687bd3ca94a30811d4512b0667870d86d11c1c"),
    ("full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42_ladder_hacc.pt", "a58d6d31ee93477fbf7b31a921abd4bde709b4e6a56ed7c90fd5667d3914a191"),
    ("full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42_ladder_hacc_mlp.pt", "5d0fa1ccabea73e1f42417a1f7e359520f3b21e9fa3ad870e82aef0dfb98beb3"),
    ("full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s43_ladder_hacc_mlp.pt", "3fb74c9584e3039a9e9d6242d2f3d3d97cfa6232516b2100a3049b0a39c6cd3b"),
    ("full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s7_ladder_hacc_mlp.pt", "8ddd373eafaf7d8f4ca04bc4291e726cc7d98c2f3459dcc0f518d9372a568af4"),
    ("full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s99_ladder_hacc_mlp.pt", "f06aef47fe79443fca78c56e74fac01d74ff459d70ed8d449e30695da63ffcfd"),
    ("full_shapesphere_N2976_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42_ladder_hacc.pt", "5e745878133ecee69c68d5eeb54b0afd243640eb2f7db5401f1d631ef95411d2"),
    ("full_shapesphere_N512_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42_ladder_hacc.pt", "3daf1d5f165dc3f9fb7c00b9b4a93635bb93c78748be67e12fb08bcefdd9aeba"),
]

REQUIRED_FIELDS = (
    "stage", "arch", "N", "seed", "fc_dim", "tag", "artifact", "artifact_path", "cmd",
    "log", "log_path", "exit_code", "elapsed_s", "finished_at", "status",
)
OK_FIELDS = ("sha256", "params", "dev_pct", "within_tol", "test_acc", "E", "K", "S_in", "S_out")

# ======================================================================
# 工具
# ======================================================================
def sha256_of(path: str) -> str:
    """返回文件的 SHA256 十六进制摘要（分块读取）。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tsha(t: Any) -> str:
    """张量原始 float32 字节的 SHA256（与改动前快照同口径）。"""
    import torch

    return hashlib.sha256(
        t.detach().to(torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()


def load_modules() -> Tuple[Any, Any, Any, Any]:
    """返回 (Config, ThreeDNeuronSpace, MLPBaseline, train_mod)。"""
    sys.path.insert(0, MODULE_DIR)
    try:
        from .config import Config  # type: ignore
        from .model import MLPBaseline, ThreeDNeuronSpace  # type: ignore
        from . import train as train_mod  # type: ignore
    except ImportError:  # pragma: no cover
        from config import Config  # type: ignore
        from model import MLPBaseline, ThreeDNeuronSpace  # type: ignore
        import train as train_mod  # type: ignore

    return Config, ThreeDNeuronSpace, MLPBaseline, train_mod


def expected_artifact_name(
    arch: str, n: Optional[int], seed: int, fc_dim: Optional[int]
) -> str:
    """按固定 CLI 口径复现 ``train.py`` 的产物名（与运行器同源命名）。"""
    import contextlib

    _, _, _, train_mod = load_modules()
    argv = ["--preset", EXPECT_PRESET, "--arch", arch, "--seed", str(seed)]
    if arch == "mlp":
        argv += ["--tag", MLP_TAG]
    else:
        argv += ["--shape", "sphere", "--n", str(int(n))]
        if fc_dim is not None:
            argv += ["--fc-dim", str(int(fc_dim))]
        argv += ["--tag", N3D_TAG]
    with contextlib.redirect_stdout(io.StringIO()):
        args = train_mod.parse_args(argv)
        cfg = train_mod.apply_overrides(train_mod.PRESETS[args.preset], args)
    return train_mod.full_checkpoint_name(cfg, args.tag)


def expected_caliber() -> Dict[str, Any]:
    """本脚本**独立复述**的固定口径（与台账 ``_meta.caliber`` 比对，防事后改口径）。"""
    return {
        "preset": EXPECT_PRESET, "shape": "sphere", "flow_axis": "z", "H": 0.10, "D": 0.10,
        "y_in": 8, "y_out": 8, "input_dim": 784, "output_dim": 10, "num_workers": 0,
        "input_scope": "any_isolated", "readout_scope": "any_isolated", "fc_dim": FC_DIM,
        "n_search": [N_SEARCH_LO, N_SEARCH_HI], "target_params": TARGET_PARAMS,
        "dev_tol_pct": DEV_TOL_PCT, "seeds": list(SEEDS), "n3d_tag": N3D_TAG,
        "mlp_tag": MLP_TAG, "mlp_hidden_dim": 2048,
        "preset_fields": {
            "epochs": 20, "batch_size": 128, "lr": 2e-3, "weight_decay": 1e-4,
            "lr_schedule": "cosine", "grad_clip": 1.0, "readout_bias": True,
        },
        "select_rule": "逐 seed 在 N in [815,831] 逐 N 实测 params，取 argmin |params - 1628170|",
    }


def check_frozen_baseline() -> List[str]:
    """核验 33 个既有产物的磁盘 SHA256 == 冻结常量（承重断言）。"""
    problems: List[str] = []
    for rel, frozen in FROZEN_BASELINE:
        path = os.path.normpath(os.path.join(CHECKPOINT_DIR, rel))
        if not os.path.isfile(path):
            problems.append(f"[零回归] 既有产物缺失：{rel}")
            continue
        got = sha256_of(path)
        if got != frozen:
            problems.append(
                f"[零回归] 既有产物被改写：{rel}（冻结 {frozen[:12]}，磁盘 {got[:12]}）")
    return problems


def base_kw(fc_dim: int, seed: int, n: int) -> Dict[str, Any]:
    """构造与固定口径一致的 Config 关键字。"""
    return dict(
        N=int(n), y_in=8, y_out=8, H=0.10, D=0.10, input_dim=784, output_dim=10,
        hidden_dim=2048, batch_size=128, lr=2e-3, epochs=20, seed=int(seed),
        device="cpu", weight_decay=1e-4, readout_bias=True, lr_schedule="cosine",
        grad_clip=1.0, shape="sphere", fc_dim=int(fc_dim),
        input_scope="any_isolated", readout_scope="any_isolated", num_workers=0,
    )


def live_params(seed: int, n: int, fc_dim: int) -> Dict[str, Any]:
    """现场重建模型并返回实测指标（params / fc_width / E / K / |S_in| / |S_out| 等）。"""
    Config, ThreeDNeuronSpace, _, _ = load_modules()
    cfg = Config(**base_kw(fc_dim, seed, n))
    model = ThreeDNeuronSpace(cfg)
    stats = model.get_connection_stats()
    topo = model.get_topology_stats()
    out = {
        "params": int(model.count_parameters()),
        "fc_width": int(model.fc_width),
        "fc_enabled": bool(model.fc_enabled),
        "E": int(stats["num_edges"]),
        "K": int(topo["num_layers_true"]),
        "S_in": int(stats["num_in_scope"]),
        "S_out": int(stats["num_out_scope"]),
        "dense_weight_tensors": int(model.count_dense_weight_tensors()),
    }
    if model.fc_enabled:
        sd = (model.out_degree[model.in_scope_mask] == 0)
        out["proj_struct_zero_grad_rows"] = int(sd.sum().item())
    del model
    gc.collect()
    return out

def rebuild_metrics(rec: Dict[str, Any]) -> Dict[str, Any]:
    """从产物现场重建模型，独立复算指标（与 verify_full_runs.py 同口径）。"""
    import torch

    Config, ThreeDNeuronSpace, MLPBaseline, _ = load_modules()
    path = rec["artifact_path"]
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover
        return {"error": f"torch.load 失败：{exc}"}
    arch = str(d.get("arch", "neuron3d"))
    cfg_d = dict(d.get("config", {}))
    cs = dict(d.get("connection_stats") or {})
    ts = dict(d.get("topology_stats") or {})
    small = {
        "test_acc": float(d.get("test_acc", float("nan"))),
        "ckpt_E": int(cs.get("num_edges", -1)),
        "ckpt_K": int(ts.get("num_layers_true", -1)) if ts else 0,
        "ckpt_S_in": int(cs.get("num_in_scope", -1)),
        "ckpt_S_out": int(cs.get("num_out_scope", -1)),
        "ckpt_preset": str(d.get("preset", "")),
        "ckpt_fc_dim": d.get("fc_dim", None),
        "ckpt_fc_width": d.get("fc_width", None),
        "ckpt_batches": d.get("batches_per_epoch"),
    }
    del d
    gc.collect()
    try:
        cfg = Config(**cfg_d)
        model = MLPBaseline(cfg) if arch == "mlp" else ThreeDNeuronSpace(cfg)
    except Exception as exc:
        return {"error": f"模型重建失败：{exc}", "config": cfg_d}
    stats = model.get_connection_stats()
    topo = model.get_topology_stats()
    out = {
        "arch": arch, "config": cfg_d, "test_acc": small["test_acc"],
        "fc_dim": int(cfg.fc_dim), "fc_width": int(cfg.fc_width),
        "params": int(model.count_parameters()),
        "E": int(stats["num_edges"]),
        "K": int(topo["num_layers_true"]) if topo else 0,
        "S_in": int(stats["num_in_scope"]),
        "S_out": int(stats["num_out_scope"]),
        "ckpt_preset": small["ckpt_preset"], "ckpt_fc_dim": small["ckpt_fc_dim"],
        "ckpt_fc_width": small["ckpt_fc_width"], "ckpt_batches": small["ckpt_batches"],
        "consistent_with_artifact": bool(
            int(stats["num_edges"]) == small["ckpt_E"]
            and int(stats["num_in_scope"]) == small["ckpt_S_in"]
            and int(stats["num_out_scope"]) == small["ckpt_S_out"]
            and (int(topo["num_layers_true"]) if topo else 0) == small["ckpt_K"]
        ),
    }
    del model
    gc.collect()
    return out


def check_fc_semantics(problems: List[str], report: List[str]) -> None:
    """A 段：fc_dim 语义 + 关闭路径逐位零回归 + 命名不变式。"""
    Config, ThreeDNeuronSpace, _, _ = load_modules()
    print("-" * 100)
    print("A. fc_dim 语义与关闭路径零回归（硬断言）")

    # A1: 取值域
    ok_domain = True
    for bad in (-2, -100):
        try:
            Config(**base_kw(bad, 42, 256))
            ok_domain = False
            problems.append(f"[fc 语义] fc_dim={bad} 应被拒但构造成功")
        except ValueError:
            pass
    for good in (-1, 0, 1, 256):
        try:
            Config(**base_kw(good, 42, 256))
        except Exception as exc:
            ok_domain = False
            problems.append(f"[fc 语义] fc_dim={good} 应被接受但报错：{exc}")
    report.append(f"  [{' ok ' if ok_domain else 'FAIL'}] fc_dim 取值域：< -1 被拒、-1/0/>0 被接受")
    print(report[-1])

    # A2: 宽度语义
    widths = {}
    for fd, expect in ((-1, 256), (0, 0), (1, 1), (256, 256), (37, 37)):
        cfg = Config(**base_kw(fd, 42, 256))
        widths[fd] = (int(cfg.fc_width), int(expect), bool(cfg.fc_enabled))
    bad_w = [k for k, (got, exp, _) in widths.items() if got != exp]
    if bad_w:
        problems.append(f"[fc 语义] fc_width 语义不符：{bad_w} -> {widths}")
    report.append(f"  [{' ok ' if not bad_w else 'FAIL'}] 宽度语义：fc_dim=-1 -> N=256、0 -> 关闭(0)、"
                  f">0 -> 该值（实测 {[(k, v[0]) for k, v in widths.items()]}）")
    print(report[-1])

    # A3: 关闭路径无 FC 参数/buffer，且与**改动前快照**逐位一致
    snap = json.load(io.open(SNAPSHOT, encoding="utf-8"))
    s_cfg = snap["config"]
    cfg0 = Config(**s_cfg)
    m0 = ThreeDNeuronSpace(cfg0)
    import torch

    torch.manual_seed(0)
    x = torch.randn(8, 784)
    a_in = m0.stage1_input_driven(x)
    a_up = m0.stage2_recurrence(a_in)
    h = m0.readout_activations(a_up)
    logits = m0(x)
    cur = {f"param:{n}": tsha(p) for n, p in m0.named_parameters()}
    cur.update({f"buffer:{n}": tsha(b) for n, b in m0.named_buffers()})
    fw = {"x": tsha(x), "a_in": tsha(a_in), "a_up": tsha(a_up), "h": tsha(h), "logits": tsha(logits)}
    mism = [k for k, v in snap["tensor_sha256"].items() if cur.get(k) != v]
    extra = [k for k in cur if k not in snap["tensor_sha256"]]
    missing = [k for k in snap["tensor_sha256"] if k not in cur]
    fw_mism = [k for k, v in snap["forward_sha256"].items() if fw.get(k) != v]
    if mism or extra or missing or fw_mism:
        problems.append(
            f"[fc 语义] fc_dim=0 与改动前快照不一致：张量 {mism}、多出 {extra}、"
            f"缺失 {missing}、前向 {fw_mism}")
    # [!] 判定前缀必须**精确**到 FC 专属名字：原先用 "buffer:out_scope" 会误命中**既有**
    #     buffer `out_scope_mask`（自检发现的校验脚本假阳性；产品侧无缺陷）。
    fc_names = [
        n for n in cur
        if n in ("param:fc_in_weight", "param:fc_in_bias", "param:proj_weight",
                 "param:fc_out_weight", "param:fc_out_bias", "param:head_weight",
                 "param:head_bias", "buffer:out_scope_index")
    ]
    if fc_names:
        problems.append(f"[fc 语义] fc_dim=0 仍创建了 FC 参数/buffer：{fc_names}")
    same_count = int(m0.count_parameters()) == int(snap["params_count"])
    if not same_count:
        problems.append(f"[fc 语义] fc_dim=0 参数量 {m0.count_parameters()} != 快照 {snap['params_count']}")
    report.append(
        f"  [{' ok ' if not (mism or extra or missing or fw_mism or fc_names or not same_count) else 'FAIL'}] "
        f"fc_dim=0 与改动前快照逐位一致：张量 {len(cur)} 个全等={not (mism or extra or missing)}，"
        f"前向 {len(fw)} 个全等={not fw_mism}，参数量 {m0.count_parameters()}"
        f"（快照 {snap['params_count']}），无 FC 参数/buffer={not fc_names}")
    print(report[-1])
    del m0
    gc.collect()

    # A4: 命名不变式（fc_dim=0 不加段；fc_dim!=0 加 _fc{n} 且彼此不同名）
    n_off = expected_artifact_name("neuron3d", 825, 42, None)
    n_on = expected_artifact_name("neuron3d", 825, 42, -1)
    n_pos = expected_artifact_name("neuron3d", 825, 42, 16)
    # [!] 不能用 `"_fc" not in name` 判定"不加段"：本轮的 tag 就是 `fc_align`，名字里必然含 "fc"。
    #     改为精确匹配 `_fc{n}_` 形态的**段**（自检发现的校验脚本假阳性；产品侧无缺陷）。
    import re as _re
    ok_n1 = _re.search(r"_fc-?\d+_", n_off) is None
    ok_n2 = (_re.search(r"_fc-1_", n_on) is not None and _re.search(r"_fc16_", n_pos) is not None)
    ok_n3 = len({n_off, n_on, n_pos}) == 3
    if not (ok_n1 and ok_n2 and ok_n3):
        problems.append(f"[fc 语义] 命名不变式失败：off={n_off} on={n_on} pos={n_pos}")
    report.append(f"  [{' ok ' if (ok_n1 and ok_n2 and ok_n3) else 'FAIL'}] 产物命名："
                  f"fc_dim=0 无 _fc 段={ok_n1}；fc_dim!=0 含 _fc{{n}} 段={ok_n2}；三者互不同名={ok_n3}")
    print(report[-1])
    print(f"        关闭路径名：{n_off}")
    print(f"        启用路径名：{n_on}")

    # A5: 解析式参数量公式 + dense 张量判据（用 fc_dim=-1、N=825、seed=42 现场核对）
    Config2, ThreeDNeuronSpace2, _, _ = load_modules()
    cfg = Config2(**base_kw(-1, 42, 825))
    model = ThreeDNeuronSpace2(cfg)
    expect = (
        model.fc_width * int(cfg.input_dim) + model.fc_width
        + model.num_in_scope * model.fc_width + model.num_edges + model.N
        + model.fc_width * model.num_out_scope + model.fc_width
        + int(cfg.output_dim) * model.fc_width + int(cfg.output_dim)
    )
    ok_formula = int(expect) == int(model.count_parameters())
    ok_dense = int(model.count_dense_weight_tensors()) == 0
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
    if not ok_formula:
        problems.append(f"[fc 语义] 解析式参数量 {expect} != count_parameters {model.count_parameters()}")
    if not ok_dense:
        problems.append("[fc 语义] count_dense_weight_tensors() != 0（新矩阵被误判为违规 dense 权重）")
    report.append(f"  [{' ok ' if (ok_formula and ok_dense) else 'FAIL'}] 解析式参数量公式成立={ok_formula}"
                  f"（{model.count_parameters()}）；dense 权重判据仍为 0={ok_dense}")
    print(report[-1])
    print(f"        fc_dim=-1, N=825, seed=42：|S_in|={model.num_in_scope} |S_out|={model.num_out_scope} "
          f"E={model.num_edges} H={model.fc_width}")
    print(f"        参数形状：{shapes}")
    sd = int((model.out_degree[model.in_scope_mask] == 0).sum().item())
    print(f"        输入侧结构性零梯度投影行（S_in 中出度为 0 的神经元）= {sd} / {model.num_in_scope}"
          f"（其 a_in 永不参与下游聚合，也不进 readout）")
    del model
    gc.collect()

def check_ledger(problems: List[str], ledger: Dict[str, Any]) -> Tuple[List[str], Dict[int, Dict[str, Any]]]:
    """B 段：台账逐条复核（SHA256 / 命名 / 字段 / config 口径 / 独立复算 / 同参容差）。"""
    runs: List[Dict[str, Any]] = ledger.get("runs", [])
    latest_index: Dict[Tuple[str, Any, int, Any], int] = {}
    for i, rec in enumerate(runs):
        latest_index[(str(rec.get("arch")), rec.get("N"), int(rec.get("seed")), rec.get("fc_dim"))] = i
    names_used: List[str] = []
    n3d_by_seed: Dict[int, Dict[str, Any]] = {}
    print("-" * 100)
    print(f"B. 台账逐条复核（{len(runs)} 条：字段 / 命名 / 磁盘 SHA / 产物 config 口径 / 独立复算 / 同参容差）")
    for i, rec in enumerate(runs):
        arch = str(rec.get("arch"))
        n = rec.get("N")
        seed = int(rec.get("seed"))
        fc = rec.get("fc_dim")
        tag = MLP_TAG if arch == "mlp" else N3D_TAG
        tag_txt = f"arch={arch:<9s} N={str(n):>6s} seed={seed:<6d} fc_dim={str(fc):>5s}"
        local: List[str] = []
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            local.append(f"缺少字段 {missing}")
        if not isinstance(rec.get("exit_code"), int):
            local.append(f"exit_code 非整数：{rec.get('exit_code')!r}")
        if rec.get("tag") != tag:
            local.append(f"tag={rec.get('tag')!r} != {tag!r}")
        try:
            exp_name = expected_artifact_name(arch, n, seed, fc)
        except Exception as exc:
            exp_name = ""
            local.append(f"命名复现失败：{exc}")
        if exp_name and rec.get("artifact") != exp_name:
            local.append(f"产物名 {rec.get('artifact')!r} != 规范名 {exp_name!r}")
        path = str(rec.get("artifact_path", ""))
        if not os.path.isfile(path):
            local.append(f"产物缺失：{path}")
        if rec.get("status") != "ok":
            if not rec.get("failure_reason"):
                local.append("失败轮缺少 failure_reason")
            if not os.path.isfile(str(rec.get("log_path", ""))):
                local.append("失败轮缺少原始日志")
        names_used.append(str(rec.get("artifact")))
        if latest_index[(arch, n, seed, fc)] == i and os.path.isfile(path):
            disk = sha256_of(path)
            if rec.get("sha256") != disk:
                local.append(f"磁盘 SHA {disk[:12]} != 台账 SHA {str(rec.get('sha256'))[:12]}")
            if rec.get("status") == "ok":
                for f in OK_FIELDS:
                    if f not in rec:
                        local.append(f"status=ok 但缺少 {f}")
                m = rebuild_metrics(rec)
                if "error" in m:
                    local.append(m["error"])
                else:
                    for f in ("params", "E", "K", "S_in", "S_out"):
                        if int(rec.get(f, -1)) != int(m[f]):
                            local.append(f"{f} 台账 {rec.get(f)} != 独立复算 {m[f]}")
                    if abs(float(rec.get("test_acc", -1.0)) - float(m["test_acc"])) > FLOAT_TOL:
                        local.append(f"test_acc 台账 {rec.get('test_acc')} != 产物 {m['test_acc']}")
                    if not m["consistent_with_artifact"]:
                        local.append("独立复算指标与产物自带 connection/topology 取证不一致")
                    if m.get("ckpt_preset") != EXPECT_PRESET:
                        local.append(f"产物 preset={m.get('ckpt_preset')!r} != {EXPECT_PRESET!r}")
                    if m.get("ckpt_batches") is not None:
                        local.append(f"产物为限批运行（batches_per_epoch={m.get('ckpt_batches')}）")
                    cfg = m["config"]
                    for fld, exp in EXPECT_CONFIG.items():
                        got = cfg.get(fld)
                        if isinstance(exp, float):
                            same = got is not None and abs(float(got) - exp) <= 1e-12
                        else:
                            same = got == exp
                        if not same:
                            local.append(f"产物 config['{fld}']={got!r} != 口径 {exp!r}")
                    if arch == "neuron3d":
                        if cfg.get("fc_dim") != FC_DIM:
                            local.append(f"产物 config['fc_dim']={cfg.get('fc_dim')!r} != {FC_DIM}")
                        if cfg.get("N") != n:
                            local.append(f"产物 config['N']={cfg.get('N')!r} != 台账 N={n!r}")
                        if m.get("ckpt_fc_dim") != FC_DIM or int(m.get("fc_width") or -1) != int(n):
                            local.append(
                                f"产物顶层 fc_dim/fc_width={m.get('ckpt_fc_dim')}/{m.get('ckpt_fc_width')} "
                                f"与 fc_dim=-1、宽度==N（{n}）不符")
                        dev_live = (int(m["params"]) - TARGET_PARAMS) / TARGET_PARAMS * 100.0
                        if abs(dev_live - float(rec.get("dev_pct", 1e9))) > 1e-4:
                            local.append(f"dev_pct 台账 {rec.get('dev_pct')} != 现算 {dev_live:.6f}")
                        if abs(dev_live) > DEV_TOL_PCT:
                            local.append(f"同参偏差 {dev_live:+.4f}% 超出 ±{DEV_TOL_PCT}%")
                        n3d_by_seed[seed] = rec
                    else:
                        if cfg.get("fc_dim") != 0:
                            local.append(f"MLP 产物 config['fc_dim']={cfg.get('fc_dim')!r} != 0")
                        if int(m["params"]) != TARGET_PARAMS:
                            local.append(
                                f"MLP 基线参数量 {m['params']} != 冻结目标 {TARGET_PARAMS}")
                    if cfg.get("seed") != seed:
                        local.append(f"产物 config['seed']={cfg.get('seed')!r} != 台账 seed={seed!r}")
                    if not math.isfinite(float(m["test_acc"])) or not (0.0 <= float(m["test_acc"]) <= 1.0):
                        local.append(f"test_acc 越界/非有限：{m['test_acc']}")
        status = "ok" if not local else "FAIL"
        print(f"  [{status:>4s}] run#{i:<3d} {tag_txt} artifact={str(rec.get('artifact'))[:52]:52s}"
              f" params={rec.get('params')} dev={rec.get('dev_pct')} exit={rec.get('exit_code')}")
        for msg in local:
            problems.append(f"run#{i} ({tag_txt}): {msg}")
            print(f"          - {msg}")
    # 产物名两两唯一 + 与既有产物名不冲突
    dup = sorted({nm for nm in names_used if names_used.count(nm) > 1})
    if dup:
        problems.append(f"产物名存在重复：{dup}")
    baseline_names = {os.path.basename(rel) for rel, _ in FROZEN_BASELINE}
    collide = sorted(set(names_used) & baseline_names)
    if collide:
        problems.append(f"与既有产物名冲突：{collide}")
    print(f"  本轮产物名 {len(names_used)} 条，去重后 {len(set(names_used))} 条："
          f"两两唯一={len(names_used) == len(set(names_used))}；与既有 33 个产物名冲突={len(collide)}")
    return names_used, n3d_by_seed


def check_search(problems: List[str], ledger: Dict[str, Any]) -> None:
    """C 段：搜索块复核（argmin 属性 + 现场重建该 N 的实测值）。"""
    search = ledger.get("search") or {}
    print("-" * 100)
    print("C. 同参搜索块复核（chosen_N 必须等于 17 行实测表的 argmin |params - target|）")
    for seed in SEEDS:
        res = search.get(str(seed))
        if not res:
            problems.append(f"[搜索] seed={seed} 缺搜索记录")
            print(f"  [FAIL] seed={seed:<6d} 缺搜索记录")
            continue
        table = res.get("table") or []
        ns = [int(r[0]) for r in table]
        ok_range = ns == list(range(N_SEARCH_LO, N_SEARCH_HI + 1))
        best = min(table, key=lambda r: abs(int(r[1]) - TARGET_PARAMS)) if table else None
        ok_argmin = best is not None and int(best[0]) == int(res.get("chosen_N")) and int(best[1]) == int(res.get("chosen_params"))
        live = live_params(seed, int(res.get("chosen_N", 0)), FC_DIM)
        ok_live = int(live["params"]) == int(res.get("chosen_params"))
        dev = (int(res.get("chosen_params", 0)) - TARGET_PARAMS) / TARGET_PARAMS * 100.0
        ok_tol = abs(dev) <= DEV_TOL_PCT
        if not (ok_range and ok_argmin and ok_live and ok_tol):
            problems.append(
                f"[搜索] seed={seed}: 覆盖 815..831={ok_range}、argmin={ok_argmin}、"
                f"现场重建一致={ok_live}（{live['params']} vs {res.get('chosen_params')}）、"
                f"容差={ok_tol}（{dev:+.4f}%）")
        print(f"  [{' ok ' if (ok_range and ok_argmin and ok_live and ok_tol) else 'FAIL'}] "
              f"seed={seed:<6d} chosen_N={res.get('chosen_N'):<5d} params={res.get('chosen_params')} "
              f"dev={dev:+.4f}% 覆盖={ok_range} argmin={ok_argmin} 现场重建一致={ok_live} "
              f"|S_in|={live['S_in']} |S_out|={live['S_out']} E={live['E']} K={live['K']}")
        if live.get("dense_weight_tensors"):
            problems.append(f"[搜索] seed={seed}: dense 权重判据命中 {live['dense_weight_tensors']} 个")


def main(argv: Optional[List[str]] = None) -> int:
    """现跑复核入口；返回 0（全部通过）/ 1（存在不一致）。"""
    problems: List[str] = []
    report: List[str] = []
    print("=" * 118)
    print("n3d_shape：fc_dim 同参对齐 —— 台账/产物现跑复核（对照 README §18）")
    print("=" * 118)
    if not os.path.isfile(LEDGER):
        print(f"[FAIL] 台账不存在：{LEDGER}")
        return 1
    with open(LEDGER, "r", encoding="utf-8") as fh:
        ledger = json.load(fh)
    meta = ledger.get("_meta", {})
    print(f"台账：{LEDGER}；记录数 {len(ledger.get('runs', []))}；搜索记录 {len(ledger.get('search', {}))}")

    if meta.get("caliber") != expected_caliber():
        problems.append("[口径] 台账 _meta.caliber 与脚本独立复述的固定口径不一致（疑似事后改口径）")
        print("  [FAIL] 口径不一致")
    else:
        print("  [ ok ] 台账 _meta.caliber == 脚本独立复述的固定口径")

    print("-" * 100)
    print("D. 既有 33 个产物 SHA256 冻结比对（10 个三期 + 22 个 N 阶梯 + 1 个一期 MLP）")
    frozen_problems = check_frozen_baseline()
    bad = 0
    for rel, frozen in FROZEN_BASELINE:
        path = os.path.normpath(os.path.join(CHECKPOINT_DIR, rel))
        ok = os.path.isfile(path) and sha256_of(path) == frozen
        bad += 0 if ok else 1
    print(f"  [{' ok ' if bad == 0 else 'FAIL'}] {len(FROZEN_BASELINE)} 个文件中 {len(FROZEN_BASELINE) - bad} 个"
          f"磁盘 SHA == 冻结常量（不一致 {bad} 个）")
    problems.extend(frozen_problems)

    check_fc_semantics(problems, report)
    names_used, n3d_by_seed = check_ledger(problems, ledger)
    check_search(problems, ledger)

    # ---- 汇总块交叉校验 ----
    print("-" * 100)
    print("E. 汇总块交叉校验（与台账 summary 逐字段比对）")
    mlp = {int(r["seed"]): r for r in ledger.get("runs", [])
           if r.get("arch") == "mlp" and r.get("status") == "ok"}
    summ = ledger.get("summary") or {}
    if not summ:
        problems.append("[汇总] 台账缺少 summary 块")
        print("  [FAIL] 台账缺少 summary 块")
    else:
        per_seed_ok = True
        for it in summ.get("per_seed", []):
            rec = n3d_by_seed.get(int(it["seed"]))
            if rec is None:
                problems.append(f"[汇总] per_seed 含 seed={it['seed']} 但台账无对应 N3D 成功记录")
                continue
            if int(it["params"]) != int(rec["params"]) or abs(float(it["dev_pct"]) - float(rec["dev_pct"])) > 1e-9:
                problems.append(f"[汇总] seed={it['seed']} params/dev 与记录不一致")
        print(f"  [{' ok ' if per_seed_ok else 'FAIL'}] per_seed "
              f"{len(summ.get('per_seed', []))} 条与记录一致={per_seed_ok}")
    if mlp:
        accs = [float(r["test_acc"]) for r in mlp.values()]
        print(f"  MLP 基线（同预算同 seed）：n={len(accs)} min={min(accs) * 100:.2f}% "
              f"max={max(accs) * 100:.2f}%")

    print("=" * 118)
    if problems:
        print(f"[FAIL] 发现 {len(problems)} 项不一致：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("[PASS] fc_dim 语义与关闭路径零回归成立、台账逐条与产物现场复核一致（含同参偏差 <= 0.5%）、"
          "搜索块 argmin 属性成立、产物名两两唯一、既有 33 个产物零回归")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())