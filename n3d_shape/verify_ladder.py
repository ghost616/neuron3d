"""N 阶梯 x 多 seed 同预算对照的**现跑复核**脚本（只读产物与台账，不写任何 checkpoint）。

职责
----
对 ``checkpoints/n3d_shape/_verify/ladder_runs.json`` 逐条现跑复核：

* 产物文件存在、**磁盘 SHA256 == 台账 SHA256**；
* 产物内 ``config`` 与台账口径一致（N / seed / shape / H / D / y / 两个 scope /
  flow_axis / epochs / batch_size / lr / weight_decay / lr_schedule / grad_clip /
  readout_bias / num_workers / hidden_dim / input_dim / output_dim）；
* 台账字段齐全（含 ``exit_code``）；``status=="ok"`` 的记录必须有完整实测指标；
* 用**现场重建模型**独立复算 ``E / K / |S_in| / |S_out| / params`` 并与台账逐一比对
  （不依赖台账、也不依赖产物自带的 ``connection_stats``）；
* 产物名**两两唯一**、且与既有产物名**不冲突**；
* 既有 11 个产物（10 个三期 + 1 个一期 MLP）的 SHA256 **零回归**（代码内**冻结常量**承重断言）；
* 按**事先固定的判据**现算达标判定并打印汇总（每 N 的 mean / min / max / 标准差），
  并与台账 ``summary`` 块交叉校验（防"汇总块与逐条记录脱节"）。

用法
----
    python n3d_shape/verify_ladder.py      # 退出码 0 = 全部通过；1 = 存在不一致

说明
----
本脚本**不 import** ``run_ladder.py``：固定口径与冻结常量都由本脚本**独立复述**，
避免"校验脚本与被测实现同源"的自洽性陷阱（本模块历史上 S3 与实现同源导致 D1 漏检）。
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, os.pardir))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "n3d_shape")
VERIFY_DIR = os.path.join(CHECKPOINT_DIR, "_verify")
CONTROL_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "_control")
LEDGER = os.path.join(VERIFY_DIR, "ladder_runs.json")

# ----------------------------------------------------------------------
# 固定口径（**独立复述**；与本模块 README §17 的表头一一对应）
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
EXPECT_ARCHS = ("neuron3d", "mlp")
N_LADDER: Tuple[int, ...] = (256, 512, 1024, 2048, 2976)
SEEDS: Tuple[int, ...] = (1, 2, 3, 7, 42, 43, 99, 123, 2024)
PILOT_SEED = 42
N3D_TAG = "ladder_hacc"
MLP_TAG = "ladder_hacc_mlp"
FLOAT_TOL = 1e-12

# ----------------------------------------------------------------------
# 冻结常量：既有 11 个产物的 SHA256（承重断言 —— 既有产物一旦被改写即 FAIL）
# ----------------------------------------------------------------------
# 说明：与 `verify_full_runs.py` 的 `FROZEN_ANYANY` 同源精神 —— 被冻结的是**产物**
# 哈希（产物不得变，故必须写死），而不是"源文件自身哈希"（源文件随修复演进）。
# 5 条 any/any 的取值与 `verify_full_runs.FROZEN_ANYANY` 逐位一致（交叉自洽）。
FROZEN_BASELINE: List[Tuple[str, str]] = [
    ("full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt",
     "c96a25d2e8bc86ece35ec61303687bd3ca94a30811d4512b0667870d86d11c1c"),
    ("full_shapecube_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt",
     "9d11e7dd07c10c9e6d7ffc35785a857b7b7aa08c4e1f51a76cf7f3f9991ad06d"),
    ("full_shapecylinder_a1_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt",
     "f18fff8b34c858ffed94ea56827690266b97adc42501989f8c0c18229d16f48c"),
    ("full_shapecylinder_a0.5_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt",
     "9ca88bacb3a0e50b1a30523fb134e6f9b2725fb08df60c146736b03c52a06534"),
    ("full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt",
     "05c70a9df2611735328710d926320f618f2a2cb3cd679c489ce6eef38a8f9f81"),
    ("full_shapesphere_N1024_y8x8_H0.1_D0.064996_plfcc_axz_isall_rsall_s42.pt",
     "beff8ce47fb0fb2503e6eeb607dbd1ba1f374bf526673423aa54eaf7b945a3e1"),
    ("full_shapecube_N1024_y8x8_H0.1_D0.0655_plfcc_axz_isall_rsall_s42.pt",
     "2a94103c8cf938ee133d434de3133d9afd4854c2c80745dfadddec3f1e3e7744"),
    ("full_shapecylinder_a1_N1024_y8x8_H0.1_D0.065531_plfcc_axz_isall_rsall_s42.pt",
     "6216f6403b20a31fd49138c5fc5a25ffce5cb7a144ec2e02c4202a145acb433e"),
    ("full_shapecylinder_a0.5_N1024_y8x8_H0.1_D0.069_plfcc_axz_isall_rsall_s42.pt",
     "67478e5ef9c5cd0b68615691acccd5e50742fd56cc0abc7ade4555857a3433a2"),
    ("full_shapecylinder_a2_N1024_y8x8_H0.1_D0.064969_plfcc_axz_isall_rsall_s42.pt",
     "15a33badd043c917b98c4f934846eb5350e3771591fa814c34160bbd89ebbbda"),
    ("../_control/mlp_highacc_ep12_seed42.pt",
     "f261716e7f3c3cba8729e46829714e9c981d5d1a896b74fd7c4d112d80a8b571"),
]

REQUIRED_FIELDS = (
    "stage", "arch", "N", "seed", "tag", "artifact", "artifact_path", "cmd",
    "log", "log_path", "exit_code", "elapsed_s", "finished_at", "status",
)
OK_FIELDS = ("sha256", "test_acc", "E", "K", "S_in", "S_out", "params")

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


def load_train_module() -> Any:
    """导入本模块自身的 ``train``（只为复用其**产物命名**代码路径，不复用其口径）。"""
    sys.path.insert(0, MODULE_DIR)
    try:
        from . import train as train_mod  # type: ignore

        return train_mod
    except ImportError:  # pragma: no cover
        import train as train_mod  # type: ignore

        return train_mod


def expected_artifact_name(arch: str, n: Optional[int], seed: int) -> str:
    """按固定 CLI 口径复现 ``train.py`` 的产物名（与运行器同源命名，避免手抄指纹）。"""
    import contextlib
    import io

    train_mod = load_train_module()
    argv = ["--preset", EXPECT_PRESET, "--arch", arch, "--seed", str(seed)]
    if arch == "mlp":
        argv += ["--tag", MLP_TAG]
    else:
        argv += ["--shape", "sphere", "--n", str(int(n)), "--tag", N3D_TAG]
    with contextlib.redirect_stdout(io.StringIO()):
        args = train_mod.parse_args(argv)
        cfg = train_mod.apply_overrides(train_mod.PRESETS[args.preset], args)
    return train_mod.full_checkpoint_name(cfg, args.tag)


def expected_caliber() -> Dict[str, Any]:
    """本脚本**独立复述**的固定口径（用于与台账 ``_meta.caliber`` 比对，防事后改口径）。"""
    return {
        "preset": EXPECT_PRESET,
        "shape": "sphere",
        "flow_axis": "z",
        "H": 0.10,
        "D": 0.10,
        "y_in": 8,
        "y_out": 8,
        "input_dim": 784,
        "output_dim": 10,
        "num_workers": 0,
        "input_scope": "any_isolated",
        "readout_scope": "any_isolated",
        "n_ladder": list(N_LADDER),
        "seeds": list(SEEDS),
        "pilot_seed": PILOT_SEED,
        "n3d_tag": N3D_TAG,
        "mlp_tag": MLP_TAG,
        "mlp_hidden_dim": 2048,
        "preset_fields": {
            "epochs": 20, "batch_size": 128, "lr": 2e-3, "weight_decay": 1e-4,
            "lr_schedule": "cosine", "grad_clip": 1.0, "readout_bias": True,
        },
        "pass_rule": "test_acc(N3D, N, seed) >= test_acc(MLP, seed) 即视为该点达标",
        "best_n_rule": "档 1 中 test_acc 最高者；并列取较大 N",
    }


def check_frozen_baseline() -> List[str]:
    """核验 11 个既有产物的磁盘 SHA256 == 冻结常量（承重断言）。"""
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


def rebuild_metrics(rec: Dict[str, Any]) -> Dict[str, Any]:
    """现场重建模型，独立复算 ``E / K / |S_in| / |S_out| / params``。

    口径与 ``verify_full_runs.py`` 一致：``params`` 由重建模型的
    ``count_parameters()`` 给出（不是从产物 state_dict 求和），
    ``E / K / S_in / S_out`` 由重建模型的 ``get_connection_stats()`` /
    ``get_topology_stats()`` 给出，并同时与产物内自带取证交叉比对。

    返回
    ----
    Dict[str, Any]
        含 ``params`` / ``E`` / ``K`` / ``S_in`` / ``S_out`` / ``test_acc`` /
        ``config`` / ``consistent_with_artifact`` 的字典；失败时含 ``error``。
    """
    import torch

    try:
        from .config import Config  # type: ignore
        from .model import MLPBaseline, ThreeDNeuronSpace  # type: ignore
    except ImportError:  # pragma: no cover
        from config import Config  # type: ignore
        from model import MLPBaseline, ThreeDNeuronSpace  # type: ignore

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
        "arch": arch,
        "config": cfg_d,
        "test_acc": small["test_acc"],
        "E": int(stats["num_edges"]),
        "K": int(topo["num_layers_true"]) if topo else 0,
        "S_in": int(stats["num_in_scope"]),
        "S_out": int(stats["num_out_scope"]),
        "params": int(model.count_parameters()),
        "ckpt_preset": small["ckpt_preset"],
        "ckpt_batches": small["ckpt_batches"],
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

# ======================================================================
# 入口
# ======================================================================
def main(argv: Optional[List[str]] = None) -> int:
    """现跑复核入口；返回 0（全部通过）/ 1（存在不一致）。"""
    problems: List[str] = []
    print("=" * 118)
    print("n3d_shape：N 阶梯 x 多 seed 同预算对照 —— 台账/产物现跑复核（对照 README §17）")
    print("=" * 118)
    if not os.path.isfile(LEDGER):
        print(f"[FAIL] 台账不存在：{LEDGER}")
        return 1
    with open(LEDGER, "r", encoding="utf-8") as fh:
        ledger = json.load(fh)
    meta = ledger.get("_meta", {})
    runs: List[Dict[str, Any]] = ledger.get("runs", [])
    print(f"台账：{LEDGER}")
    print(f"记录数：{len(runs)}；档位口径 _meta.caliber 存在：{bool(meta.get('caliber'))}")

    # ---- (0) 口径自洽：台账记录的口径必须等于本脚本独立复述的口径（防事后改口径）----
    if meta.get("caliber") != expected_caliber():
        problems.append("[口径] 台账 _meta.caliber 与脚本独立复述的固定口径不一致（疑似事后改口径）")
        print("  [FAIL] 口径不一致")
    else:
        print("  [ ok ] 台账 _meta.caliber == 脚本独立复述的固定口径")

    # ---- (1) 既有产物零回归（冻结常量承重断言）----
    print("-" * 100)
    print("既有 11 个产物 SHA256 冻结比对（10 个三期 + 1 个一期 MLP；产物一改即 FAIL）")
    frozen_problems = check_frozen_baseline()
    for rel, frozen in FROZEN_BASELINE:
        path = os.path.normpath(os.path.join(CHECKPOINT_DIR, rel))
        ok = os.path.isfile(path) and sha256_of(path) == frozen
        disk = sha256_of(path)[:12] if os.path.isfile(path) else "MISSING"
        print(f"  [{' ok ' if ok else 'FAIL'}] {rel[:74]:74s} disk={disk} frozen={frozen[:12]}")
    problems.extend(frozen_problems)

    # ---- (2) 逐条字段/命名/产物/SHA/口径/独立复算 ----
    print("-" * 100)
    print("逐条复核（字段齐全 / 命名规范 / 磁盘 SHA / 产物 config 口径 / 独立复算指标）")
    latest_index: Dict[Tuple[str, Any, int], int] = {}
    for i, rec in enumerate(runs):
        latest_index[(str(rec.get("arch")), rec.get("N"), int(rec.get("seed")))] = i
    names_used: List[str] = []
    per_seed_mlp: Dict[int, Dict[str, Any]] = {}
    per_n_seed_n3d: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for i, rec in enumerate(runs):
        arch = str(rec.get("arch"))
        n = rec.get("N")
        seed = int(rec.get("seed"))
        tag = MLP_TAG if arch == "mlp" else N3D_TAG
        tag_txt = f"arch={arch:<9s} N={str(n):>6s} seed={seed:<6d}"
        local: List[str] = []
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            local.append(f"缺少字段 {missing}")
        if not isinstance(rec.get("exit_code"), int):
            local.append(f"exit_code 非整数：{rec.get('exit_code')!r}")
        if rec.get("tag") != tag:
            local.append(f"tag={rec.get('tag')!r} != {tag!r}")
        if arch not in EXPECT_ARCHS:
            local.append(f"arch 非法：{arch!r}")
        try:
            exp_name = expected_artifact_name(arch, n, seed)
        except Exception as exc:
            exp_name = ""
            local.append(f"命名复现失败：{exc}")
        if exp_name and rec.get("artifact") != exp_name:
            local.append(f"产物名 {rec.get('artifact')!r} != 规范名 {exp_name!r}")
        if exp_name and os.path.basename(str(rec.get("artifact_path", ""))) != exp_name:
            local.append("artifact_path 的 basename 与规范名不一致")
        path = str(rec.get("artifact_path", ""))
        if not os.path.isfile(path):
            local.append(f"产物缺失：{path}")
        if rec.get("status") != "ok":
            if not rec.get("failure_reason"):
                local.append("失败轮缺少 failure_reason")
            if not os.path.isfile(str(rec.get("log_path", ""))):
                local.append("失败轮缺少原始日志")
        names_used.append(str(rec.get("artifact")))
        # 只有**最新一条**才做磁盘 SHA 与产物内容比对（重跑会更新产物与 SHA）
        if latest_index[(arch, n, seed)] == i and os.path.isfile(path):
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
                    for f in ("E", "K", "S_in", "S_out", "params"):
                        if int(rec.get(f, -1)) != int(m[f]):
                            local.append(f"{f} 台账 {rec.get(f)} != 独立复算 {m[f]}")
                    if abs(float(rec.get("test_acc", -1.0)) - float(m["test_acc"])) > FLOAT_TOL:
                        local.append(f"test_acc 台账 {rec.get('test_acc')} != 产物 {m['test_acc']}")
                    if not m["consistent_with_artifact"]:
                        local.append("独立复算指标与产物内自带 connection/topology 取证不一致")
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
                        if cfg.get("N") != n:
                            local.append(f"产物 config['N']={cfg.get('N')!r} != 台账 N={n!r}")
                    else:
                        if cfg.get("N") != 256:
                            local.append(f"MLP 产物 config['N']={cfg.get('N')!r} != 预设 256")
                    if cfg.get("seed") != seed:
                        local.append(f"产物 config['seed']={cfg.get('seed')!r} != 台账 seed={seed!r}")
                    if not math.isfinite(float(m["test_acc"])) or not (0.0 <= float(m["test_acc"]) <= 1.0):
                        local.append(f"test_acc 越界/非有限：{m['test_acc']}")
                    if arch == "mlp":
                        per_seed_mlp[seed] = rec
                    else:
                        per_n_seed_n3d[(int(n), seed)] = rec
        status = "ok" if not local else "FAIL"
        print(f"  [{status:>4s}] run#{i:<3d} {tag_txt} artifact={(str(rec.get('artifact'))[:56]):56s}"
              f" exit={rec.get('exit_code')} status={rec.get('status')}")
        for msg in local:
            problems.append(f"run#{i} ({tag_txt}): {msg}")
            print(f"          - {msg}")

    # ---- (3) 产物名两两唯一 + 与既有产物名不冲突 ----
    print("-" * 100)
    dup = sorted({nm for nm in names_used if names_used.count(nm) > 1})
    if dup:
        problems.append(f"产物名存在重复：{dup}")
    baseline_names = {os.path.basename(rel) for rel, _ in FROZEN_BASELINE}
    collide = sorted(set(names_used) & baseline_names)
    if collide:
        problems.append(f"与既有产物名冲突：{collide}")
    print(f"  本轮产物名 {len(names_used)} 条，去重后 {len(set(names_used))} 条："
          f"两两唯一={len(names_used) == len(set(names_used))}")
    print(f"  与既有 11 个产物名冲突数：{len(collide)}")

    # ---- (4) 按事先固定规则现算达标判定与每 N 区间统计 ----
    print("-" * 100)
    print("按固定规则现算（N3D test_acc >= 同 seed MLP test_acc 即达标）")
    print(f"{'N':>6s} {'seed':>6s} {'N3D acc':>9s} {'MLP acc':>9s} {'Δ(pp)':>8s} {'达标':>6s}")
    pass_points: List[Dict[str, Any]] = []
    for key in sorted(per_n_seed_n3d):
        n_v, seed = key
        rec = per_n_seed_n3d[key]
        m = per_seed_mlp.get(seed)
        if m is None:
            print(f"{n_v:>6d} {seed:>6d} {float(rec['test_acc']) * 100:>8.2f}% {'n/a':>9s} {'n/a':>8s} {'待补':>6s}")
            continue
        a, b = float(rec["test_acc"]), float(m["test_acc"])
        ok = a >= b
        if ok:
            pass_points.append({"N": n_v, "seed": seed, "n3d_acc": a, "mlp_acc": b,
                                "delta_pp": 100.0 * (a - b)})
        print(f"{n_v:>6d} {seed:>6d} {a * 100:>8.2f}% {b * 100:>8.2f}% {100 * (a - b):>+7.2f} "
              f"{('达标' if ok else '未达标'):>6s}")

    by_n: Dict[int, List[float]] = {}
    for (n_v, seed), rec in per_n_seed_n3d.items():
        by_n.setdefault(n_v, []).append(float(rec["test_acc"]))
    print("-" * 100)
    print("每 N 的 seed 区间统计（test_acc；样本标准差 ddof=1，n<2 记为 n/a）")
    print(f"{'N':>6s} {'n_seed':>7s} {'mean':>10s} {'min':>10s} {'max':>10s} {'stdev':>10s}")
    recomputed_stats: Dict[str, Any] = {}
    for n_v in sorted(by_n):
        vals = by_n[n_v]
        mean = sum(vals) / len(vals)
        sd = None
        if len(vals) >= 2:
            sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        recomputed_stats[str(n_v)] = {
            "n_seed": len(vals), "acc_mean": mean, "acc_min": min(vals),
            "acc_max": max(vals), "acc_stdev_ddof1": sd,
        }
        print(f"{n_v:>6d} {len(vals):>7d} {mean * 100:>9.4f}% {min(vals) * 100:>9.4f}% "
              f"{max(vals) * 100:>9.4f}% "
              f"{('n/a(n<2)' if sd is None else f'{sd * 100:.4f}%'):>10s}")
    print(f"达标点清单：{len(pass_points)} 个 / {len(per_n_seed_n3d)} 个有效点")

    # ---- (5) 与台账 summary 块交叉校验（防"汇总块与逐条记录脱节"）----
    print("-" * 100)
    summ = ledger.get("summary") or {}
    if not summ:
        problems.append("[汇总] 台账缺少 summary 块（应由 run_ladder.py 收尾写入）")
        print("  [FAIL] 台账缺少 summary 块")
    else:
        led_pass = {f"{d['N']}|{d['seed']}" for d in summ.get("pass_points", [])}
        my_pass = {f"{d['N']}|{d['seed']}" for d in pass_points}
        if led_pass != my_pass:
            problems.append(f"[汇总] 达标点集合不一致：台账 {sorted(led_pass)} vs 现算 {sorted(my_pass)}")
            print(f"  [FAIL] 达标点集合不一致：台账 {sorted(led_pass)} vs 现算 {sorted(my_pass)}")
        else:
            print(f"  [ ok ] 达标点集合与台账 summary 一致（{len(my_pass)} 个）")
        led_stats = summ.get("per_N_stats", {})
        stat_bad = []
        for k, v in recomputed_stats.items():
            lv = led_stats.get(k, {})
            if int(lv.get("n_seed", -1)) != v["n_seed"]:
                stat_bad.append(f"N={k} n_seed {lv.get('n_seed')} != {v['n_seed']}")
                continue
            for f in ("acc_mean", "acc_min", "acc_max"):
                if lv.get(f) is None or abs(float(lv[f]) - float(v[f])) > 1e-12:
                    stat_bad.append(f"N={k} {f} {lv.get(f)} != {v[f]}")
            ls, vs = lv.get("acc_stdev_ddof1"), v["acc_stdev_ddof1"]
            if (ls is None) != (vs is None) or (ls is not None and abs(float(ls) - float(vs)) > 1e-12):
                stat_bad.append(f"N={k} stdev {ls} != {vs}")
        if stat_bad:
            problems.extend(f"[汇总] {b}" for b in stat_bad)
            for b in stat_bad:
                print(f"  [FAIL] {b}")
        else:
            print(f"  [ ok ] 每 N 区间统计与台账 summary 逐字段一致（{len(recomputed_stats)} 档）")
        if int(summ.get("n_valid_points", -1)) != len(per_n_seed_n3d):
            problems.append(f"[汇总] n_valid_points {summ.get('n_valid_points')} != 现算 {len(per_n_seed_n3d)}")

    # ---- (6) 计划覆盖度 ----
    print("-" * 100)
    plan_seeds = [s for s in SEEDS if s != PILOT_SEED]
    # [!] 生成器表达式里必须用 `n_v` 而不是 `n`（离朱前自检发现：`n` 是 main 的局部变量，
    # 逐条循环结束后其值为最后一条记录的 N —— 对 MLP 记录恰为 None，会被闭包读到，
    # 导致 `sorted([None, ...])` 抛 TypeError；生成器表达式有独立作用域但**可读外层局部**）。
    pilot_ns = sorted(n_v for (n_v, s) in per_n_seed_n3d if s == PILOT_SEED)
    print(f"档 1（seed={PILOT_SEED}）覆盖的 N：{pilot_ns}（期望 {list(N_LADDER)}）")
    print(f"档 2 覆盖的 seed：{sorted(s for (n_v, s) in per_n_seed_n3d if s != PILOT_SEED)}"
          f"（期望 {plan_seeds}）")
    print(f"MLP 覆盖的 seed：{sorted(per_seed_mlp)}（期望 {list(SEEDS)}）")
    if pilot_ns != sorted(N_LADDER):
        print("  [信息] 档 1 尚未跑满（N 阶梯不完整）—— 若本次为档 1 中间态属预期")
    missing_mlp = [s for s in SEEDS if s not in per_seed_mlp]
    if missing_mlp:
        print(f"  [信息] 尚缺 MLP 基线的 seed：{missing_mlp}")

    print("=" * 118)
    if problems:
        print(f"[FAIL] 发现 {len(problems)} 项不一致：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("[PASS] 台账逐条与产物现场复核一致：SHA256 一致、口径一致、独立复算一致、"
          "产物名两两唯一、既有 11 个产物零回归")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())