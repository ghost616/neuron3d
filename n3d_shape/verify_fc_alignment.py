"""fc_dim 同参对齐的**现跑复核**脚本（只读产物与台账，不写任何 checkpoint）。

职责
----
A. **fc_dim 语义与关闭路径零回归**（预先定义的硬断言）：
   * `fc_dim = -1` -> 两端有效宽度 `== N`；`> 0` -> `== 该值`；`0` -> 关闭（宽度 0）；
   * `fc_dim < -1` 在 `Config` 构造期被拒；
   * `fc_dim == 0` 时**不创建任何 FC 参数 / buffer**，且模型逐张量与**改动前快照**
     (`_verify/fc_pre_change_snapshot.json`，由改动前的源码现场生成) **逐位一致**；
   * **承重前置断言：该快照"确属改动前"** —— 快照 `source_sha256` 中的 `config.py` /
     `model.py` / `train.py` 必须与**当前**三个源文件的 SHA256 **两两不等**
     （佐证判据：快照 `config` 不含本次改动新增的 `fc_dim` 键）。**理由**：上面的逐位
     比对是"关闭路径零回归"的**全部**证据，其效力完全依赖"快照早于改动"这一前提；
     若快照恰在含改动的源码上生成，该比对会退化为"实现与自身一致"的**自洽性检查**
     （不报错但证明力归零），故把该前提写成硬断言而非人工目视；
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
E. **汇总块交叉校验**：`summary.per_seed` 与逐条记录一致；**配对差符号计数**（正 / 负 / 零）
   由 `per_seed[*].delta_pp` **现场统计**，并与**台账登记值**及 **README §18 声明值**
   三方比对（承重断言，防止"4 正 5 负"被写成"5 正 4 负"这类无产物支撑的方向计数再犯）。

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
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, os.pardir))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "n3d_shape")
VERIFY_DIR = os.path.join(CHECKPOINT_DIR, "_verify")
LEDGER = os.path.join(VERIFY_DIR, "fc_alignment_runs.json")
SNAPSHOT = os.path.join(VERIFY_DIR, "fc_pre_change_snapshot.json")
README = os.path.join(MODULE_DIR, "README.md")

# 快照"确属改动前"的承重断言所覆盖的源文件（与快照 source_sha256 的键一一对应）。
SNAPSHOT_SOURCE_FILES: Tuple[str, ...] = ("config.py", "model.py", "train.py")

# 配对差符号计数的**冻结期望**（承重断言：README / 台账登记值 / 现场复算三者必须同时等于它）。
# 期望值来源：README §18.5 实测配对表（9 对）—— 正：seed 1/3/43/2024，负：seed 2/7/42/99/123。
# 之所以允许在此**硬编码期望条数**：它正是"防止 README 方向计数被写错"的锚点；
# 除此之外的一切数值（mean/stdev/min/max、逐 seed 的 delta_pp）都现场从台账取数、不得硬编码。
EXPECT_SIGN_POS = 4
EXPECT_SIGN_NEG = 5
EXPECT_SIGN_ZERO = 0

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


def check_snapshot_precedes_change(snap: Dict[str, Any]) -> List[str]:
    """**承重断言**：证明 ``fc_pre_change_snapshot.json`` 确属"改动前"生成的快照。

    职责
    ----
    A3 段用该快照做 ``fc_dim == 0`` **关闭路径逐位零回归**证明；该证明的**全部效力**都
    建立在"快照是在**尚未引入 `fc_dim` 的源码**上生成"这一前提上。若快照恰在**含改动的
    源码**上生成，A3 就退化为"实现与自身一致"的自洽性检查（与既有"同源比对"同类的
    失效模式）：不报错，但**零回归证明归零**。故此处把该前提写成硬断言。

    判据（主判据 + 佐证判据，两者都参与失败判定）
    --------------------------------------------
    * **主判据**：快照 ``source_sha256`` 的三个源文件（``config.py`` / ``model.py`` /
      ``train.py``）必须**都存在**，且当前磁盘上同名文件的 SHA256 与快照值**两两不等**
      —— 直接、可复现、与源码内容一一对应；源码一旦回退到改动前状态即 FAIL。
    * **佐证判据**：快照 ``config`` **不含** ``fc_dim`` 键 —— `fc_dim` 是本次改动新增的
      ``Config`` 字段，改动前的 ``config`` 不可能带该键（结构性证据）。

    参数
    ----
    snap : Dict[str, Any]
        已加载的快照 JSON。

    返回
    ----
    List[str]
        问题列表；**空列表 = 快照确属改动前**（调用方据此决定 A3 是否仍是零回归证明）。
    """
    problems: List[str] = []
    src = snap.get("source_sha256") or {}
    if not src:
        problems.append("[快照] 快照缺少 source_sha256 段 —— 无法证明其早于改动")
    else:
        for name in SNAPSHOT_SOURCE_FILES:
            frozen = str(src.get(name, ""))
            if not frozen:
                problems.append(f"[快照] 快照 source_sha256 缺少 {name} —— 无法证明其早于改动")
                continue
            path = os.path.join(MODULE_DIR, name)
            if not os.path.isfile(path):
                problems.append(f"[快照] 当前源文件缺失：{name}")
                continue
            live = sha256_of(path)
            if live == frozen:
                problems.append(
                    f"[快照] {name} 当前 SHA256 与快照相同（{live[:12]}）—— 该快照**不是**"
                    f"改动前生成的，A3 的零回归证明退化为自洽性检查")
                print(f"        [!] {name}: 快照={frozen[:12]} 当前={live[:12]}（相同 -> FAIL）")
            else:
                print(f"        [ok] {name}: 快照={frozen[:12]} 当前={live[:12]}（不同）")
    if "fc_dim" in (snap.get("config") or {}):
        problems.append("[快照] 快照 config 含 fc_dim 键（佐证判据失败）—— 快照非改动前生成")
    return problems


def readme_paired_sign_declaration() -> Optional[Tuple[int, int, int]]:
    """从 README §18.5 解析配对差**符号计数**声明。

    匹配形如 ``<n> 对中 <p> 对为正、<m> 对为负`` 的句子（允许 Markdown 粗体包裹）。

    返回
    ----
    Optional[Tuple[int, int, int]]
        ``(n_total, n_pos, n_neg)``；README 未声明该计数时返回 ``None``（调用方记为问题）。
    """
    if not os.path.isfile(README):
        return None
    with open(README, "r", encoding="utf-8") as fh:
        text = fh.read()
    hits = re.findall(r"(\d+)\s*对中\s*(\d+)\s*对为正[、,]\s*(\d+)\s*对为负", text)
    if not hits:
        return None
    uniq = {(int(a), int(b), int(c)) for a, b, c in hits}
    # 必须**唯一**：若文档既写了新口径、又原样引用了旧口径的**同一句式**，视为声明矛盾，
    # 交由断言 FAIL 提示人工统一（避免"谁先出现谁生效"的隐式口径）。
    if len(uniq) != 1:
        return None
    return uniq.pop()


def check_paired_sign_counts(summ: Dict[str, Any], report: List[str]) -> List[str]:
    """**承重断言**：配对差符号计数的「现场复算 == 台账登记值 == README 声明值」。

    背景：README §18.5 曾把实测的「4 对为正、5 对为负」写成「5 对为正、4 对为负」
    （无产物支撑的方向计数）。本函数用三重比对把该口径钉住：

    1. **现场复算**：由 ``summary.per_seed[*].delta_pp`` 现数正 / 负 / 零的条数；
    2. **台账登记值**：``summary.paired_delta_vs_mlp`` 的 ``n_pos`` / ``n_neg`` / ``n_zero``；
    3. **README 声明值**：§18.5 文本中的「N 对中 P 对为正、M 对为负」。

    三者必须同时等于冻结期望（``EXPECT_SIGN_POS`` / ``EXPECT_SIGN_NEG`` / ``EXPECT_SIGN_ZERO``），
    且 ``n_total`` 必须等于现场复算的配对条数。

    参数
    ----
    summ : Dict[str, Any]
        台账 ``summary`` 块。
    report : List[str]
        报告行收集器（本函数会 append 一行）。

    返回
    ----
    List[str]
        问题列表（空 = 三方一致）。
    """
    problems: List[str] = []
    deltas = [float(it["delta_pp"]) for it in (summ.get("per_seed") or []) if "delta_pp" in it]
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_zero = sum(1 for d in deltas if d == 0)
    reg = summ.get("paired_delta_vs_mlp") or {}
    reg_triplet = (reg.get("n_pos"), reg.get("n_neg"), reg.get("n_zero"))
    readme_triplet = readme_paired_sign_declaration()
    if not deltas:
        problems.append("[汇总] 台账 summary.per_seed 中没有 delta_pp，无法统计配对差符号计数")
        report.append("  [FAIL] 配对差符号计数：台账无可统计的 delta_pp")
        print(report[-1])
        return problems
    if reg_triplet != (n_pos, n_neg, n_zero):
        problems.append(
            f"[汇总] 配对差符号计数：台账登记值 {reg_triplet} != 现场复算 {(n_pos, n_neg, n_zero)}")
    if readme_triplet is None:
        problems.append(
            "[汇总] README 未声明「N 对中 P 对为正、M 对为负」符号计数"
            "（或声明**不唯一/自相矛盾**，例如纠错说明里原样引用了旧口径的同一句式）")
    else:
        if readme_triplet != (len(deltas), n_pos, n_neg):
            problems.append(
                f"[汇总] README §18.5 符号计数声明 {readme_triplet} != 现场复算 "
                f"{(len(deltas), n_pos, n_neg)}")
    expect = (EXPECT_SIGN_POS, EXPECT_SIGN_NEG, EXPECT_SIGN_ZERO)
    if (n_pos, n_neg, n_zero) != expect:
        problems.append(
            f"[汇总] 配对差符号计数 {n_pos} 正 / {n_neg} 负 / {n_zero} 零 != 冻结期望 "
            f"{EXPECT_SIGN_POS} 正 / {EXPECT_SIGN_NEG} 负 / {EXPECT_SIGN_ZERO} 零")
    ok = not problems
    report.append(
        f"  [{' ok ' if ok else 'FAIL'}] 配对差符号计数（承重断言）：现场复算 {n_pos} 正 / "
        f"{n_neg} 负 / {n_zero} 零（n={len(deltas)}）；台账登记 {reg_triplet}；"
        f"README §18.5 声明 {readme_triplet}")
    print(report[-1])
    return problems


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
    #
    # [!] 承重前置断言（本轮补上）：先证明"快照确属改动前"，再做逐位比对。
    #     否则若快照在含改动的源码上生成，下方比对会退化为"实现与自身一致"（证明力归零）。
    snap = json.load(io.open(SNAPSHOT, encoding="utf-8"))
    snap_pre_problems = check_snapshot_precedes_change(snap)
    snap_pre_ok = not snap_pre_problems
    problems.extend(snap_pre_problems)
    report.append(
        f"  [{' ok ' if snap_pre_ok else 'FAIL'}] 快照确属改动前（承重断言）：三个源文件当前 "
        f"SHA256 与快照 source_sha256 两两不等={snap_pre_ok}；"
        f"快照 config 无 fc_dim 键={'fc_dim' not in (snap.get('config') or {})}")
    print(report[-1])
    if not snap_pre_ok:
        print("        [!] 快照前置断言失败 —— 下方逐位比对已退化为**自洽性检查**，不再构成零回归证明")
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
    a3_ok = (not (mism or extra or missing or fw_mism or fc_names or not same_count)) and snap_pre_ok
    report.append(
        f"  [{' ok ' if a3_ok else 'FAIL'}] fc_dim=0 与**改动前**快照逐位一致：张量 {len(cur)} 个"
        f"全等={not (mism or extra or missing)}，前向 {len(fw)} 个全等={not fw_mism}，"
        f"参数量 {m0.count_parameters()}（快照 {snap['params_count']}），"
        f"无 FC 参数/buffer={not fc_names}，快照前置断言成立={snap_pre_ok}")
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
        # [!] 缺 `chosen_N` / `chosen_params` 时**不得崩溃**（本轮修复）：
        #     原先以 `int(res.get("chosen_N", 0))` 兜底调用 `live_params(seed, 0, FC_DIM)`，
        #     而 `Config` 对 `N <= 0` 会 raise ValueError 且此处不捕获 → 整个复核脚本
        #     **traceback 崩溃**，而不是以退码 1 报告"[搜索] seed=… 不一致"。
        #     现改为：先判 `chosen_N` 存在且 > 0（`chosen_params` 同步判），
        #     不满足则记 problem 并 `continue`（跳过该 seed 的现场重建，不中断其余核对）。
        chosen_n_raw = res.get("chosen_N")
        chosen_p_raw = res.get("chosen_params")
        if (not isinstance(chosen_n_raw, int) or isinstance(chosen_n_raw, bool)
                or chosen_n_raw <= 0
                or not isinstance(chosen_p_raw, int) or isinstance(chosen_p_raw, bool)):
            problems.append(
                f"[搜索] seed={seed}: chosen_N / chosen_params 缺失或非法"
                f"（chosen_N={chosen_n_raw!r}、chosen_params={chosen_p_raw!r}）"
                f"—— 无法现场重建该 N 做核对")
            print(f"  [FAIL] seed={seed:<6d} chosen_N={chosen_n_raw!r} chosen_params={chosen_p_raw!r} "
                  f"-> 缺失/非法，跳过现场重建（记 problem，不崩溃）")
            continue
        best = min(table, key=lambda r: abs(int(r[1]) - TARGET_PARAMS)) if table else None
        ok_argmin = (best is not None and int(best[0]) == chosen_n_raw
                     and int(best[1]) == chosen_p_raw)
        live = live_params(seed, chosen_n_raw, FC_DIM)
        ok_live = int(live["params"]) == chosen_p_raw
        dev = (chosen_p_raw - TARGET_PARAMS) / TARGET_PARAMS * 100.0
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
    print("E. 汇总块交叉校验（与台账 summary 逐字段比对 + 配对差符号计数承重断言）")
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
        # 配对差符号计数：现场复算（per_seed.delta_pp）== 台账登记值 == README §18.5 声明值
        problems.extend(check_paired_sign_counts(summ, report))
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
    print("[PASS] fc_dim 语义与关闭路径零回归成立（**含**「快照确属改动前」承重断言）、"
          "台账逐条与产物现场复核一致（含同参偏差 <= 0.5%）、搜索块 argmin 属性成立、"
          "配对差符号计数三方一致（现场复算 == 台账登记 == README §18.5）、"
          "产物名两两唯一、既有 33 个产物零回归")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())