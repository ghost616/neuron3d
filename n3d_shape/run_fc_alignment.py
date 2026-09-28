"""fc_dim 同参对齐运行器：**逐 seed 搜 N 使总参数最接近 MLP 基线**，再跑同预算训练对照。

目标
----
在 `n3d_shape` 主模型上新启用 `fc_dim = -1`（两端全连接包裹，有效宽度 `H = N`），
回答"**参数量对齐到同预算 MLP 基线后**，三期与 MLP 的性能差是多少"。
本对照为**同类比较**（同 preset / 同 seed / 同数据 / 同 bs/lr/epochs）；
**不得**表述为对架构优劣的因果结论。

固定口径（事先固定，不得事后更改）
----------------------------------
* preset = ``highacc``：epochs=20 / batch_size=128 / lr=2e-3 / AdamW(wd=1e-4) /
  lr_schedule=cosine / grad_clip=1.0 / readout_bias=True；
* ``shape=sphere``、**不传** ``--cyl-aspect``、``flow_axis=z``、``H=D=0.10``、
  **不传** ``--space-radius``（取公式下界 `R_min`）；
* ``y_in=y_out=8``、``input_dim=784``、``output_dim=10``、``num_workers=0``、
  ``input_scope=readout_scope=any_isolated``；
* **``fc_dim = -1``**（两端有效宽度 `H = N`）；两端激活 ReLU；最后一层线性无激活；
* 同参目标 = MLP 基线 `1,628,170`（`hidden_dim=2048`，`torch.load` 实测复核）；
* 逐 seed 在 ``N ∈ [815, 831]``（步长 1）**逐 N 实测**总参数、取 ``argmin |params - target|``
  （**不可解析求解**：`|S_in|` / `|S_out|` / `E` 逐 N 跳变，`params(N)` 局部非单调）；
* 断言 ``|偏差| <= 0.5%``；seed ∈ {1, 2, 3, 7, 42, 43, 99, 123, 2024}；
* 每个 seed 另跑一次**同预算同 seed 的 MLP 基线**（``--arch mlp``）。

档位
----
* ``--stage search``：9 个 seed 的 N 搜索（**只构造、不训练**，属**构造探测**）；
  台账中"该 seed 有记录但记录非法"（缺 ``chosen_N`` / 非正整数 / ``chosen_params`` 非整数）时
  **显式打印 [FAIL] 并重算该 seed**（不静默沿用非法记录、不崩溃），且该 seed 计入失败判定（退码 1）。
* ``--stage train``：按搜索解出的 N 跑 9 组 N3D+FC 与 9 组 MLP 基线（共 18 轮）；
* ``--stage all``（默认）：先 search 再 train。
* ``--allow-partial``：把**覆盖度硬断言**（``search`` 档须 9/9 个 seed 有合法搜索解、
  ``train`` 档须 18/18 轮齐备）**显式降级为信息**（如实打印缺口但不判失败）。
  **默认不降级** —— 覆盖度不足一律退码 ``1``；该开关只在"确实只想跑一部分"时使用。
  注意：它**只降级覆盖度断言**，**不放宽** ``train`` 档的**前置条件**
  （搜索解齐备 9/9；缺解仍退码 ``2``），也**不改变**"台账搜索记录非法并已重算"所导致的退码 ``1``。

产物防撞名
----------
``train.full_checkpoint_name`` 既不含 preset/epochs/batch_size/lr，也不含 arch：
故 N3D 带 ``--tag fc_align``、MLP 带 ``--tag fc_align_mlp``；
`fc_dim != 0` 时产物名另含 ``_fc{n}`` 段（`fc_dim == 0` 不加段，既有产物名逐字不变）。

用法
----
    python n3d_shape/run_fc_alignment.py --stage search
    python n3d_shape/run_fc_alignment.py --stage train
    python n3d_shape/run_fc_alignment.py --stage all
    python n3d_shape/run_fc_alignment.py --dry-run
    python n3d_shape/run_fc_alignment.py --stage train --allow-partial  # 显式允许覆盖度不足
    python n3d_shape/run_fc_alignment.py --extract <checkpoint>   # 内部用

台账：``checkpoints/n3d_shape/_verify/fc_alignment_runs.json``（全部现场取数，禁止手填）。
逐轮日志：``checkpoints/n3d_shape/_verify/log_fcalign_{arch}_{N|Npreset}_s{seed}.txt``
（显式 UTF-8，末尾写入**真实退出码**）。
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
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, os.pardir))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "n3d_shape")
VERIFY_DIR = os.path.join(CHECKPOINT_DIR, "_verify")
CONTROL_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "_control")
TRAIN_PY = os.path.join(MODULE_DIR, "train.py")
LEDGER = os.path.join(VERIFY_DIR, "fc_alignment_runs.json")

# ----------------------------------------------------------------------
# 固定口径（开工前固定；台账与 README §18 均以本处常量为准）
# ----------------------------------------------------------------------
PRESET = "highacc"
SHAPE = "sphere"
FLOW_AXIS = "z"
H = 0.10
D = 0.10
Y_IN = 8
Y_OUT = 8
INPUT_DIM = 784
OUTPUT_DIM = 10
NUM_WORKERS = 0
INPUT_SCOPE = "any_isolated"
READOUT_SCOPE = "any_isolated"
FC_DIM = -1
N_SEARCH_LO = 815
N_SEARCH_HI = 831
TARGET_PARAMS = 1628170          # MLP 基线（hidden_dim=2048）实测参数量
DEV_TOL_PCT = 0.5                # 同参偏差容差（事先固定）
SEEDS: Tuple[int, ...] = (1, 2, 3, 7, 42, 43, 99, 123, 2024)
MLP_HIDDEN_DIM = 2048
N3D_TAG = "fc_align"
MLP_TAG = "fc_align_mlp"
# 零回归快照排除标记：**只排除本轮自己的产物**，故既有 10 个三期产物 + 22 个 N 阶梯产物
# + 一期 MLP 控制产物全部纳入基线（33 个文件）。
BASELINE_TAG_MARKERS = ("_fc_align",)
CONTROL_ARTIFACT = os.path.join(CONTROL_DIR, "mlp_highacc_ep12_seed42.pt")

# ======================================================================
# 基础工具
# ======================================================================
def utc_now() -> str:
    """返回当前 UTC 时间（ISO 8601，秒精度）。"""
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def sha256_of(path: str) -> str:
    """返回文件的 SHA256 十六进制摘要（分块读取）。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def baseline_snapshot() -> Dict[str, str]:
    """对"既有产物"取 SHA256 快照（零回归断言基准）。

    集合 = ``checkpoints/n3d_shape/*.pt`` 中**不含** ``_fc_align`` 标记的文件
    （即本轮开工前的 10 个三期产物 + 22 个 N 阶梯产物）
    + ``checkpoints/_control/mlp_highacc_ep12_seed42.pt``（共 33 个）。
    排除本轮产物可保证**断点续跑**时快照集合与首轮一致。
    """
    snap: Dict[str, str] = {}
    if os.path.isdir(CHECKPOINT_DIR):
        for name in sorted(os.listdir(CHECKPOINT_DIR)):
            if not name.endswith(".pt"):
                continue
            if any(marker in name for marker in BASELINE_TAG_MARKERS):
                continue
            full = os.path.join(CHECKPOINT_DIR, name)
            snap[full] = sha256_of(full)
    if os.path.isfile(CONTROL_ARTIFACT):
        snap[CONTROL_ARTIFACT] = sha256_of(CONTROL_ARTIFACT)
    return snap


def baseline_problems(expected: Dict[str, str]) -> List[str]:
    """现场重算既有产物 SHA256 并与快照比对（空列表 = 零回归）。"""
    problems: List[str] = []
    for path, sha in expected.items():
        if not os.path.isfile(path):
            problems.append(f"[零回归] 既有产物消失：{path}")
            continue
        got = sha256_of(path)
        if got != sha:
            problems.append(f"[零回归] 既有产物被改写：{path}（{sha[:12]} -> {got[:12]}）")
    return problems


def load_train_module() -> Any:
    """导入本模块自身的 ``train``（复用其命名/覆盖代码路径，不重复实现指纹格式）。"""
    sys.path.insert(0, MODULE_DIR)
    try:
        from . import train as train_mod  # type: ignore

        return train_mod
    except ImportError:  # pragma: no cover
        import train as train_mod  # type: ignore

        return train_mod


def load_model_module() -> Any:
    """导入本模块自身的 ``model`` + ``config``（供 N 搜索的**构造探测**使用）。"""
    sys.path.insert(0, MODULE_DIR)
    try:
        from .config import Config  # type: ignore
        from .model import MLPBaseline, ThreeDNeuronSpace  # type: ignore
    except ImportError:  # pragma: no cover
        from config import Config  # type: ignore
        from model import MLPBaseline, ThreeDNeuronSpace  # type: ignore

    return Config, ThreeDNeuronSpace, MLPBaseline


def build_argv(arch: str, n: Optional[int], seed: int, fc_dim: Optional[int]) -> List[str]:
    """构造单轮训练的 CLI 参数（解释器与脚本路径之外的纯参数部分）。

    参数
    ----
    arch : str
        ``neuron3d`` / ``mlp``。
    n : Optional[int]
        N3D 的神经元数量；``None`` 表示不传 ``--n``（MLP 专用）。
    seed : int
        随机种子。
    fc_dim : Optional[int]
        ``None`` 表示不传 ``--fc-dim``（MLP 专用）。
    """
    argv = ["--preset", PRESET, "--arch", arch, "--seed", str(seed)]
    if arch == "mlp":
        argv += ["--tag", MLP_TAG]
        return argv
    argv += ["--shape", SHAPE, "--n", str(int(n))]
    if fc_dim is not None:
        argv += ["--fc-dim", str(int(fc_dim))]
    argv += ["--tag", N3D_TAG]
    return argv


def full_command(arch: str, n: Optional[int], seed: int, fc_dim: Optional[int]) -> List[str]:
    """返回可直接执行的完整命令行。"""
    return [sys.executable, "-X", "utf8", TRAIN_PY] + build_argv(arch, n, seed, fc_dim)


def expected_artifact(
    arch: str, n: Optional[int], seed: int, fc_dim: Optional[int]
) -> Tuple[str, str]:
    """返回该轮的规范产物名与绝对路径（复用 train.py 的命名代码路径）。"""
    train_mod = load_train_module()
    with contextlib.redirect_stdout(io.StringIO()):
        args = train_mod.parse_args(build_argv(arch, n, seed, fc_dim))
        cfg = train_mod.apply_overrides(train_mod.PRESETS[args.preset], args)
    if train_mod.is_default_config(cfg):
        name = "model.pt"
    else:
        name = train_mod.full_checkpoint_name(cfg, args.tag)
    return name, os.path.join(train_mod.CHECKPOINT_DIR, name)


def log_name(arch: str, n: Optional[int], seed: int) -> str:
    """逐轮日志文件名（含 arch / N / seed 三段）。"""
    n_part = "Npreset" if n is None else f"N{int(n)}"
    return f"log_fcalign_{arch}_{n_part}_s{seed}.txt"


# ======================================================================
# 台账读写
# ======================================================================
def load_ledger() -> Dict[str, Any]:
    """读取台账；不存在时返回空壳。"""
    if os.path.isfile(LEDGER):
        with open(LEDGER, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"_meta": {}, "search": {}, "runs": []}


def save_ledger(ledger: Dict[str, Any]) -> None:
    """把台账以 UTF-8（无 BOM）写回磁盘。"""
    os.makedirs(VERIFY_DIR, exist_ok=True)
    with open(LEDGER, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(ledger, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def find_run(
    ledger: Dict[str, Any], arch: str, n: Optional[int], seed: int, fc_dim: Optional[int]
) -> Optional[Dict[str, Any]]:
    """在台账中查找 ``(arch, N, seed, fc_dim)`` 对应的轮次记录（取最后一条）。"""
    hit = None
    for rec in ledger.get("runs", []):
        if (
            rec.get("arch") == arch
            and rec.get("seed") == seed
            and rec.get("N") == n
            and rec.get("fc_dim") == fc_dim
        ):
            hit = rec
    return hit

# ======================================================================
# N 搜索（**构造探测**，只构造不训练）
# ======================================================================
def base_config_kw(fc_dim: int, seed: int, n: int) -> Dict[str, Any]:
    """返回与固定口径一致的 Config 关键字（供构造探测与复核共用）。"""
    return dict(
        N=int(n), y_in=Y_IN, y_out=Y_OUT, H=H, D=D, input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM, hidden_dim=MLP_HIDDEN_DIM, batch_size=128, lr=2e-3,
        epochs=20, seed=int(seed), device="cpu", weight_decay=1e-4, readout_bias=True,
        lr_schedule="cosine", grad_clip=1.0, shape=SHAPE, fc_dim=int(fc_dim),
        input_scope=INPUT_SCOPE, readout_scope=READOUT_SCOPE, num_workers=NUM_WORKERS,
    )


def search_one_seed(seed: int) -> Dict[str, Any]:
    """对单个 seed 逐 N 构造并实测总参数，返回 argmin 解（**构造期量，非训练实测**）。

    参数
    ----
    seed : int
        随机种子（只影响突触采样 -> `E` / `|S_in|` / `|S_out|`）。

    返回
    ----
    Dict[str, Any]
        含 ``table``（逐 N 的 [(N, params, dev_pct)]）、``chosen_N``、``chosen_params``、
        ``chosen_dev_pct``、``elapsed_s`` 的字典。
    """
    Config, ThreeDNeuronSpace, _ = load_model_module()
    t0 = time.perf_counter()
    table: List[List[Any]] = []
    best: Optional[Tuple[int, int]] = None
    for n in range(N_SEARCH_LO, N_SEARCH_HI + 1):
        cfg = Config(**base_config_kw(FC_DIM, seed, n))
        model = ThreeDNeuronSpace(cfg)
        params = int(model.count_parameters())
        del model
        gc.collect()
        dev = (params - TARGET_PARAMS) / TARGET_PARAMS * 100.0
        table.append([int(n), params, round(dev, 6)])
        if best is None or abs(params - TARGET_PARAMS) < abs(best[1] - TARGET_PARAMS):
            best = (int(n), params)
    assert best is not None
    return {
        "kind": "construct_probe",
        "note": "构造期量（只构造不训练）；params 逐 N 跳变，故必须逐 N 实测取 argmin",
        "seed": int(seed),
        "n_range": [N_SEARCH_LO, N_SEARCH_HI],
        "target_params": TARGET_PARAMS,
        "fc_dim": FC_DIM,
        "table": table,
        "chosen_N": best[0],
        "chosen_params": best[1],
        "chosen_dev_pct": round((best[1] - TARGET_PARAMS) / TARGET_PARAMS * 100.0, 6),
        "within_tol": abs(best[1] - TARGET_PARAMS) / TARGET_PARAMS * 100.0 <= DEV_TOL_PCT,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


# ======================================================================
# 单产物实测指标提取（独立子进程）
# ======================================================================
def extract_metrics(path: str) -> Dict[str, Any]:
    """从产物现场提取实测指标（供 ``--extract`` 子进程调用）。

    口径与 ``verify_full_runs.py`` 一致：``params`` / ``E`` / ``K`` / ``|S_in|`` /
    ``|S_out|`` 由**现场重建模型**得到，并与产物自带的 ``connection_stats`` /
    ``topology_stats`` 交叉比对；``test_acc`` 取自产物内的训练结果。
    内存：先取出小字段并 ``del`` 掉巨大的 ``model_state_dict`` 再构造模型。
    """
    import torch

    Config, ThreeDNeuronSpace, MLPBaseline = load_model_module()
    d = torch.load(path, map_location="cpu", weights_only=False)
    arch = str(d.get("arch", "neuron3d"))
    cfg_d = dict(d.get("config", {}))
    cs = dict(d.get("connection_stats") or {})
    ts = dict(d.get("topology_stats") or {})
    small = {
        "test_acc": float(d.get("test_acc", float("nan"))),
        "ckpt_E": int(cs.get("num_edges", -1)),
        "ckpt_S_in": int(cs.get("num_in_scope", -1)),
        "ckpt_S_out": int(cs.get("num_out_scope", -1)),
        "ckpt_K": int(ts.get("num_layers_true", -1)) if ts else 0,
        "ckpt_preset": str(d.get("preset", "")),
        "ckpt_fc_dim": d.get("fc_dim", None),
        "ckpt_fc_width": d.get("fc_width", None),
        "ckpt_batches": d.get("batches_per_epoch"),
    }
    del d
    gc.collect()

    cfg = Config(**cfg_d)
    model = MLPBaseline(cfg) if arch == "mlp" else ThreeDNeuronSpace(cfg)
    stats = model.get_connection_stats()
    topo = model.get_topology_stats()
    out: Dict[str, Any] = dict(small)
    out.update(
        {
            "arch": arch,
            "config": cfg_d,
            "fc_dim": int(cfg.fc_dim),
            "fc_width": int(cfg.fc_width),
            "params": int(model.count_parameters()),
            "E": int(stats["num_edges"]),
            "K": int(topo["num_layers_true"]) if topo else 0,
            "S_in": int(stats["num_in_scope"]),
            "S_out": int(stats["num_out_scope"]),
            "dual_copy_count": int(topo["dual_copy_count"]) if topo else 0,
            "topology_matches_checkpoint": bool(
                int(stats["num_edges"]) == small["ckpt_E"]
                and int(stats["num_in_scope"]) == small["ckpt_S_in"]
                and int(stats["num_out_scope"]) == small["ckpt_S_out"]
                and (int(topo["num_layers_true"]) if topo else 0) == small["ckpt_K"]
            ),
        }
    )
    if arch != "mlp" and bool(cfg.fc_enabled):
        # 输入侧"出度为 0 的 S_in 神经元"：其 a_in 永不参与任何下游聚合（也不进 readout），
        # 故对应投影行 `proj_weight[j, :]`（j 为该神经元在 S_in 内的位置）**结构性零梯度**。
        sd = (model.out_degree[model.in_scope_mask] == 0)
        out["proj_struct_zero_grad_rows"] = int(sd.sum().item())
        out["proj_rows"] = int(model.num_in_scope)
    return out


def extract_metrics_subprocess(path: str) -> Dict[str, Any]:
    """在独立子进程内调用 ``--extract`` 并解析其 JSON 输出。"""
    cmd = [sys.executable, "-X", "utf8", os.path.abspath(__file__), "--extract", path]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", env=env
    )
    if proc.returncode != 0:
        return {"extract_error": f"exit={proc.returncode}", "extract_stderr": (proc.stderr or "")[-2000:]}
    line = [ln for ln in (proc.stdout or "").splitlines() if ln.startswith("{")]
    if not line:
        return {"extract_error": "no-json", "extract_stdout": (proc.stdout or "")[-2000:]}
    return json.loads(line[-1])


def scan_log_health(log_path: str, test_acc: float) -> List[str]:
    """扫描日志与 test_acc，返回 NaN/Inf 等健康问题列表（空 = 健康）。"""
    issues: List[str] = []
    if not math.isfinite(float(test_acc)):
        issues.append(f"test_acc 非有限值：{test_acc}")
    if os.path.isfile(log_path):
        # 统一用 `with` 管理句柄（异常/提前返回时也必定关闭，避免句柄泄漏）。
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            txt = fh.read().lower()
        for token in ("loss=nan", "loss=inf", "loss=-inf", "test_acc=nan", "test_acc=inf"):
            if token in txt:
                issues.append(f"日志命中 {token}")
    return issues

# ======================================================================
# 单轮执行
# ======================================================================
def run_round(
    stage: str,
    arch: str,
    n: Optional[int],
    seed: int,
    fc_dim: Optional[int],
    ledger: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    """执行（或跳过）单轮训练，返回并落库该轮台账记录。

    断点续跑：台账已有该 ``(arch, N, seed, fc_dim)`` 的成功记录、产物存在、
    且**磁盘 SHA256 == 台账 SHA256** 时跳过训练；``force=True`` 强制重跑。
    """
    name, path = expected_artifact(arch, n, seed, fc_dim)
    lg = log_name(arch, n, seed)
    lg_path = os.path.join(VERIFY_DIR, lg)
    prev = find_run(ledger, arch, n, seed, fc_dim)
    if not force and prev and prev.get("exit_code") == 0 and os.path.isfile(path):
        sha = sha256_of(path)
        if sha == prev.get("sha256"):
            print(
                f"[跳过] (arch={arch}, N={n}, seed={seed}, fc_dim={fc_dim}) 已完成且 "
                f"SHA256 与台账一致（{sha[:12]}），不重复训练"
            )
            return prev
    os.makedirs(VERIFY_DIR, exist_ok=True)
    cmd = full_command(arch, n, seed, fc_dim)
    print(f"[运行] (arch={arch}, N={n}, seed={seed}, fc_dim={fc_dim}) -> {name}")
    t0 = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with open(lg_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"[FCALIGN] stage={stage} arch={arch} N={n} seed={seed} fc_dim={fc_dim}\n")
        fh.write(f"[FCALIGN] cwd={PROJECT_ROOT}\n")
        fh.write("[FCALIGN] cmd=" + " ".join(cmd) + "\n")
        fh.write(f"[FCALIGN] start_utc={utc_now()}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
        rc = int(proc.returncode)
        fh.flush()
        fh.write(f"[FCALIGN] end_utc={utc_now()}\n")
        fh.write(f"[FCALIGN] exit_code={rc}\n")
    elapsed = time.perf_counter() - t0
    rec: Dict[str, Any] = {
        "stage": stage,
        "arch": arch,
        "N": n,
        "seed": seed,
        "fc_dim": fc_dim,
        "tag": MLP_TAG if arch == "mlp" else N3D_TAG,
        "artifact": name,
        "artifact_path": path,
        "cmd": cmd,
        "log": lg,
        "log_path": lg_path,
        "exit_code": rc,
        "elapsed_s": round(elapsed, 1),
        "finished_at": utc_now(),
        "status": "ok",
        "failure_reason": None,
    }
    if arch == "mlp":
        rec["N_note"] = "MLP 不使用 N / fc_dim；产物名与产物内 config 沿用预设 N=256、fc_dim=0"
    if rc != 0:
        rec["status"] = "failed"
        rec["failure_reason"] = f"训练进程退出码 {rc}（不静默重试、不用其它配置顶替）"
    elif not os.path.isfile(path):
        rec["status"] = "failed"
        rec["failure_reason"] = f"退出码 0 但产物缺失：{path}"
    else:
        m = extract_metrics_subprocess(path)
        if "extract_error" in m:
            rec["status"] = "failed"
            rec["failure_reason"] = f"产物指标提取失败：{m}"
        else:
            rec["sha256"] = sha256_of(path)
            rec["params"] = int(m["params"])
            rec["dev_pct"] = round(
                (rec["params"] - TARGET_PARAMS) / TARGET_PARAMS * 100.0, 6
            )
            rec["within_tol"] = bool(abs(rec["dev_pct"]) <= DEV_TOL_PCT)
            rec["test_acc"] = float(m["test_acc"])
            rec["E"] = int(m["E"])
            rec["K"] = int(m["K"])
            rec["S_in"] = int(m["S_in"])
            rec["S_out"] = int(m["S_out"])
            rec["fc_width"] = int(m["fc_width"])
            rec["dual_copy_count"] = int(m["dual_copy_count"])
            rec["proj_struct_zero_grad_rows"] = m.get("proj_struct_zero_grad_rows")
            rec["preset_in_checkpoint"] = m.get("ckpt_preset")
            rec["config"] = m.get("config")
            rec["topology_matches_checkpoint"] = bool(m.get("topology_matches_checkpoint"))
            bad = scan_log_health(lg_path, rec["test_acc"])
            if bad:
                rec["status"] = "failed"
                rec["failure_reason"] = "；".join(bad)
    ledger["runs"].append(rec)
    save_ledger(ledger)
    print(
        f"       -> exit={rec['exit_code']} status={rec['status']} "
        f"params={rec.get('params')} dev={rec.get('dev_pct')}% "
        f"test_acc={rec.get('test_acc')} elapsed={rec['elapsed_s']}s"
    )
    return rec


# ======================================================================
# 口径 / 汇总
# ======================================================================
def caliber_dict() -> Dict[str, Any]:
    """返回**开工前固定**的口径字典（写入台账 ``_meta.caliber``，防事后改口径）。"""
    return {
        "preset": PRESET,
        "shape": SHAPE,
        "flow_axis": FLOW_AXIS,
        "H": H,
        "D": D,
        "y_in": Y_IN,
        "y_out": Y_OUT,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "num_workers": NUM_WORKERS,
        "input_scope": INPUT_SCOPE,
        "readout_scope": READOUT_SCOPE,
        "fc_dim": FC_DIM,
        "n_search": [N_SEARCH_LO, N_SEARCH_HI],
        "target_params": TARGET_PARAMS,
        "dev_tol_pct": DEV_TOL_PCT,
        "seeds": list(SEEDS),
        "n3d_tag": N3D_TAG,
        "mlp_tag": MLP_TAG,
        "mlp_hidden_dim": MLP_HIDDEN_DIM,
        "preset_fields": {
            "epochs": 20, "batch_size": 128, "lr": 2e-3, "weight_decay": 1e-4,
            "lr_schedule": "cosine", "grad_clip": 1.0, "readout_bias": True,
        },
        "select_rule": "逐 seed 在 N in [815,831] 逐 N 实测 params，取 argmin |params - 1628170|",
    }


def latest_records(
    ledger: Dict[str, Any], arch: Optional[str] = None
) -> List[Dict[str, Any]]:
    """按 ``(arch, N, seed, fc_dim)`` 去重取最后一条记录。"""
    latest: Dict[Tuple[str, Any, int, Any], Dict[str, Any]] = {}
    for rec in ledger.get("runs", []):
        if arch is not None and rec.get("arch") != arch:
            continue
        latest[(str(rec.get("arch")), rec.get("N"), int(rec.get("seed")), rec.get("fc_dim"))] = rec
    return list(latest.values())


def summarize_ledger(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """生成机读汇总块（逐 seed 的 params/dev/test_acc、区间统计、与 MLP 的配对差）。"""
    n3d = sorted(
        [r for r in latest_records(ledger, arch="neuron3d") if r.get("status") == "ok"],
        key=lambda r: int(r["seed"]),
    )
    mlp = {int(r["seed"]): r for r in latest_records(ledger, arch="mlp") if r.get("status") == "ok"}
    per_seed: List[Dict[str, Any]] = []
    paired: List[float] = []
    for r in n3d:
        m = mlp.get(int(r["seed"]))
        item = {
            "seed": int(r["seed"]),
            "N": int(r["N"]),
            "params": int(r["params"]),
            "dev_pct": float(r["dev_pct"]),
            "within_tol": bool(r["within_tol"]),
            "test_acc": float(r["test_acc"]),
            "E": int(r["E"]),
            "K": int(r["K"]),
            "S_in": int(r["S_in"]),
            "S_out": int(r["S_out"]),
        }
        if m is not None:
            item["mlp_test_acc"] = float(m["test_acc"])
            item["delta_pp"] = 100.0 * (float(r["test_acc"]) - float(m["test_acc"]))
            paired.append(item["delta_pp"])
        per_seed.append(item)
    accs = [r["test_acc"] for r in n3d]
    stats = None
    if accs:
        mean = sum(accs) / len(accs)
        sd = None
        if len(accs) >= 2:
            sd = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5
        stats = {
            "n_seed": len(accs), "mean": mean, "min": min(accs), "max": max(accs),
            "stdev_ddof1": sd,
        }
    paired_stats = None
    if paired:
        pm = sum(paired) / len(paired)
        psd = None
        if len(paired) >= 2:
            psd = (sum((d - pm) ** 2 for d in paired) / (len(paired) - 1)) ** 0.5
        paired_stats = {
            "n": len(paired), "mean_pp": pm, "stdev_pp": psd,
            "min_pp": min(paired), "max_pp": max(paired),
            # 配对差**符号计数**：现场由 paired 逐条现算后**登记入台账**，
            # 供 `verify_fc_alignment.py` 的承重断言与 README §18.5 交叉比对
            # （历史缺陷：README 曾把"4 对为正、5 对为负"写成"5 对为正、4 对为负"）。
            "n_pos": sum(1 for d in paired if d > 0),
            "n_neg": sum(1 for d in paired if d < 0),
            "n_zero": sum(1 for d in paired if d == 0),
        }
    return {
        "per_seed": per_seed,
        "test_acc_stats": stats,
        "paired_delta_vs_mlp": paired_stats,
        "all_within_tol": all(bool(r["within_tol"]) for r in n3d) if n3d else False,
        "failed_rounds": [
            {"arch": r.get("arch"), "N": r.get("N"), "seed": r.get("seed"),
             "fc_dim": r.get("fc_dim"), "exit_code": r.get("exit_code"),
             "reason": r.get("failure_reason"), "log": r.get("log")}
            for r in ledger.get("runs", []) if r.get("status") != "ok"
        ],
    }

def print_summary(ledger: Dict[str, Any]) -> bool:
    """打印逐 seed 实测表、区间统计与配对差；返回是否存在失败轮。"""
    summ = ledger.get("summary") or {}
    per_seed = summ.get("per_seed") or []
    failed = summ.get("failed_rounds") or []
    print("=" * 118)
    print("逐 seed 实测表（同参搜索解出的 N + fc_dim=-1 两端全连接包裹；同预算同 seed MLP 基线）")
    print("=" * 118)
    print(f"{'seed':>6s} {'N':>6s} {'params':>10s} {'dev%':>9s} {'<=0.5%':>7s} "
          f"{'E':>6s} {'K':>4s} {'S_in':>6s} {'S_out':>6s} {'test_acc':>9s} "
          f"{'MLP acc':>9s} {'Δ(pp)':>8s}")
    for it in sorted(per_seed, key=lambda d: d["seed"]):
        mlp_txt = f"{it['mlp_test_acc'] * 100:.2f}%" if "mlp_test_acc" in it else "n/a"
        d_txt = f"{it['delta_pp']:+.2f}" if "delta_pp" in it else "n/a"
        print(f"{it['seed']:>6d} {it['N']:>6d} {it['params']:>10d} {it['dev_pct']:>+8.4f}% "
              f"{('是' if it['within_tol'] else '否'):>7s} {it['E']:>6d} {it['K']:>4d} "
              f"{it['S_in']:>6d} {it['S_out']:>6d} {it['test_acc'] * 100:>8.2f}% "
              f"{mlp_txt:>9s} {d_txt:>8s}")
    st = summ.get("test_acc_stats")
    if st:
        sd = "n/a(n<2)" if st["stdev_ddof1"] is None else f"{st['stdev_ddof1'] * 100:.4f}%"
        print("-" * 118)
        print(f"N3D+FC test_acc：n={st['n_seed']} mean={st['mean'] * 100:.4f}% "
              f"min={st['min'] * 100:.4f}% max={st['max'] * 100:.4f}% stdev={sd}")
    ps = summ.get("paired_delta_vs_mlp")
    if ps:
        pssd = "n/a" if ps["stdev_pp"] is None else f"{ps['stdev_pp']:.4f} pp"
        print(f"与 MLP 的配对差（同 seed 逐对）：n={ps['n']} mean={ps['mean_pp']:+.4f} pp "
              f"stdev={pssd} min={ps['min_pp']:+.2f} pp max={ps['max_pp']:+.2f} pp "
              f"（{ps.get('n_pos')} 对为正 / {ps.get('n_neg')} 对为负 / {ps.get('n_zero')} 对为零）")
    print(f"同参偏差全部 <= 0.5%：{summ.get('all_within_tol')}")
    if failed:
        print(f"[WARN] 存在 {len(failed)} 个失败轮次（如实呈现，不改口径）：")
        for r in failed:
            print(f"  - arch={r.get('arch')} N={r.get('N')} seed={r.get('seed')} "
                  f"fc_dim={r.get('fc_dim')} exit={r.get('exit_code')} "
                  f"原因={r.get('reason')} 日志={r.get('log')}")
    return not failed


def missing_search_solutions(ledger: Dict[str, Any]) -> List[int]:
    """返回 ``search`` 块中**缺合法解**的 seed 列表（``chosen_N`` 非正整数或 ``chosen_params`` 非整数）。

    用途（离朱 R22 发现的产品侧低危项）：
    ``train`` / ``all`` 档需要「每个 seed 都有正整数 ``chosen_N``」这一前置条件；
    原先只检查"``search`` 块非空"，于是当**某个** seed 缺 ``chosen_N`` 时，
    计划里该轮退化为哨兵值 ``-1``，随后训练循环执行
    ``int(ledger["search"][str(seed)]["chosen_N"])`` 会抛**未捕获的 ``KeyError``**
    （真实 CLI 表现为 traceback + 退码 1，**没有**人类可读的 ``[FAIL]`` 报文）。
    本函数把该前置条件集中成一处判据，供**运行前置检查**与**覆盖度硬断言**共用。
    """
    bad: List[int] = []
    for seed in SEEDS:
        res = (ledger.get("search") or {}).get(str(int(seed))) or {}
        cn, cp = res.get("chosen_N"), res.get("chosen_params")
        if (not isinstance(cn, int) or isinstance(cn, bool) or cn <= 0
                or not isinstance(cp, int) or isinstance(cp, bool)):
            bad.append(int(seed))
    return bad


def coverage_problems(
    ledger: Dict[str, Any], stage: str, allow_partial: bool = False
) -> List[str]:
    """**覆盖度硬断言**：本档要求的台账记录必须齐备，否则退码 1（缺口逐条打印）。

    为什么需要它
    ------------
    原实现的退出码只看"**已存在的**记录是否都成功"（`failed_rounds` 为空 + 容差达标）：
    * ``--stage search`` 在**全程零训练记录**（甚至零搜索记录）时 `search_vals` 为空、
      `n3d_recs` 为空，`tol_ok` 保持 `True` → **退码 0**，把"什么都没跑"报成成功；
    * ``--stage train`` 若某个 seed 的轮次**从未运行**（台账里根本没有该条记录），
      `failed_rounds` 同样为空 → **不被判失败**。
    本函数把"**应当有几条**"写成硬断言，堵住"缺记录 == 成功"的逃逸路径。

    判据（事先固定）
    ---------------
    * ``--stage search``：``search`` 块须 **9/9** 个 seed 都有合法解 —— ``chosen_N`` 为正整数、
      ``chosen_params`` 为整数、``within_tol`` 为真；
    * ``--stage train``：须 **18/18** 轮齐备 —— 9 个 seed 各一条 N3D（``fc_dim=-1``）记录
      + 各一条 MLP 记录，且 ``status == "ok"`` 且 ``exit_code == 0``（每 seed 取最后一条记录）；
    * ``--stage all``：以上两条**同时**成立。

    参数
    ----
    ledger : Dict[str, Any]
        已加载的台账。
    stage : str
        ``search`` / ``train`` / ``all``。
    allow_partial : bool
        显式降级开关（CLI ``--allow-partial``）：为真时把缺口**降级为信息**并返回空列表 ——
        "确实只想跑一部分"的场景必须由使用者**显式声明**，**默认不降级**。

    返回
    ----
    List[str]
        覆盖度缺口描述列表（空 = 覆盖度达标，或已由 ``allow_partial`` 显式降级）。
    """
    problems: List[str] = []
    tag = "WARN" if allow_partial else "FAIL"
    print("-" * 118)
    print(f"[覆盖度] 硬断言（stage={stage}；"
          f"--allow-partial {'已开启 -> 缺口降级为信息' if allow_partial else '未开启'}）")

    def _last_by_seed(arch: str) -> Dict[int, Dict[str, Any]]:
        """每个 seed 的**最后一条**该 arch 记录（按写入序取最后者，不筛 N / fc_dim）。"""
        out: Dict[int, Dict[str, Any]] = {}
        for rec in ledger.get("runs", []):
            if str(rec.get("arch")) != arch:
                continue
            out[int(rec.get("seed"))] = rec
        return out

    if stage in ("search", "all"):
        bad: List[int] = []
        for seed in SEEDS:
            res = (ledger.get("search") or {}).get(str(int(seed))) or {}
            cn, cp = res.get("chosen_N"), res.get("chosen_params")
            if (not isinstance(cn, int) or isinstance(cn, bool) or cn <= 0
                    or not isinstance(cp, int) or isinstance(cp, bool)
                    or not bool(res.get("within_tol"))):
                bad.append(int(seed))
        n_ok = len(SEEDS) - len(bad)
        print(f"  [{(' ok ' if not bad else tag):>4s}] search 档：{n_ok}/{len(SEEDS)} 个 seed 有合法搜索解"
              f"（chosen_N 为正整数、chosen_params 为整数、within_tol 为真）；缺/非法 seed={bad}")
        if bad:
            problems.append(
                f"[覆盖度] --stage {stage} 要求搜索解 {len(SEEDS)}/{len(SEEDS)}："
                f"实际 {n_ok}/{len(SEEDS)}；缺/非法 seed={bad}")

    if stage in ("train", "all"):
        # 前置条件（离朱 R22 建议）：train/all 档要求**每个 seed** 都有合法搜索解，
        # 否则训练循环无法确定 N（原实现会在循环内抛未捕获 KeyError）。
        bad_sol = missing_search_solutions(ledger)
        n_sol_ok = len(SEEDS) - len(bad_sol)
        print(f"  [{(' ok ' if not bad_sol else tag):>4s}] train 档 · 搜索解齐备："
              f"{n_sol_ok}/{len(SEEDS)}（chosen_N 为正整数、chosen_params 为整数；"
              f"缺/非法 seed={bad_sol}）")
        if bad_sol:
            problems.append(
                f"[覆盖度] --stage {stage} 要求 {len(SEEDS)}/{len(SEEDS)} 个 seed 的搜索解齐备"
                f"（chosen_N 为正整数、chosen_params 为整数）：实际 {n_sol_ok}/{len(SEEDS)}；"
                f"缺/非法 seed={bad_sol}")
        total_ok = 0
        total_need = 2 * len(SEEDS)
        for arch, label in (("neuron3d", "N3D+FC（fc_dim=-1）"), ("mlp", "MLP 基线")):
            last = _last_by_seed(arch)
            missing = [int(s) for s in SEEDS if int(s) not in last]
            notok = [int(s) for s in SEEDS
                     if int(s) in last and not (last[int(s)].get("status") == "ok"
                                                and int(last[int(s)].get("exit_code", -1)) == 0)]
            n_ok = len(SEEDS) - len(missing) - len(notok)
            total_ok += n_ok
            print(f"  [{(' ok ' if (not missing and not notok) else tag):>4s}] train 档 · {label}："
                  f"{n_ok}/{len(SEEDS)} 轮齐备（status=ok 且 exit_code=0）；"
                  f"缺失 seed={missing}；非成功 seed={notok}")
            if missing or notok:
                problems.append(
                    f"[覆盖度] --stage {stage} 要求 {label} 轮次 {len(SEEDS)}/{len(SEEDS)} 齐备"
                    f"（status=ok 且 exit_code=0）：实际 {n_ok}/{len(SEEDS)}；"
                    f"缺失 seed={missing}；非成功 seed={notok}")
        print(f"  [{(' ok ' if total_ok == total_need else tag):>4s}] train 档合计："
              f"{total_ok}/{total_need} 轮（9 组 N3D+FC + 9 组 MLP）")

    if allow_partial and problems:
        print(f"  [WARN] --allow-partial 已开启：上述 {len(problems)} 项覆盖度缺口**降级为信息**，不判失败")
        return []
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。

    返回
    ----
    int
        退出码：``0`` = 计划内全部轮次成功、**覆盖度达标**（search 9/9、train 18/18）、
        同参偏差全部 <= 0.5% 且既有产物零回归；
        ``1`` = 存在失败轮 / 偏差超容差 / **覆盖度不足**（缺记录，默认判失败）/
        零回归被破坏（失败轮**不静默重试**）；
        ``2`` = 前置条件不满足或台账口径漂移。
    """
    parser = argparse.ArgumentParser(
        description="fc_dim 同参对齐运行器（逐 seed 搜 N 使总参数最接近 MLP 基线，再跑同预算对照）"
    )
    parser.add_argument("--stage", choices=["search", "train", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="已完成的轮次也强制重跑")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="把覆盖度硬断言（search 9/9、train 18/18）显式降级为信息；默认不降级。"
             "注意：它只降级**覆盖度断言**，不放宽 train 档的**前置条件**"
             "（9/9 搜索解齐备；缺解仍退码 2），也不改变「台账搜索记录非法并已重算」"
             "所导致的退码 1",
    )
    parser.add_argument("--extract", default="", help="[内部] 打印单个产物的实测指标 JSON 后退出")
    args = parser.parse_args(argv)

    if args.extract:
        print(json.dumps(extract_metrics(args.extract), ensure_ascii=False))
        return 0

    os.makedirs(VERIFY_DIR, exist_ok=True)
    ledger = load_ledger()
    ledger.setdefault("search", {})
    meta = ledger.setdefault("_meta", {})
    cal = caliber_dict()
    if not meta.get("caliber"):
        meta["caliber"] = cal
        meta["created_at"] = utc_now()
        meta["baseline_sha256"] = baseline_snapshot()
        meta["baseline"] = {
            "rule": "checkpoints/n3d_shape/*.pt 中不含 _fc_align 标记者 + "
                    "checkpoints/_control/mlp_highacc_ep12_seed42.pt",
            "n_files": len(meta["baseline_sha256"]),
            "captured_at": utc_now(),
        }
        save_ledger(ledger)
        print(f"[台账] 初始化 {LEDGER}；既有产物基线快照 {len(meta['baseline_sha256'])} 个"
              f"（应为 10 个三期 + 22 个 N 阶梯 + 1 个一期 MLP = 33）")
    elif meta.get("caliber") != cal:
        print("[FAIL] 台账口径与当前 run_fc_alignment.py 的固定口径不一致 —— 禁止事后改口径")
        return 2

    baseline: Dict[str, str] = meta["baseline_sha256"]
    pre = baseline_problems(baseline)
    if pre:
        print("[FAIL] 运行前既有产物零回归断言已失败：")
        for p in pre:
            print(f"  - {p}")
        return 1

    plan: List[Tuple[str, str, Optional[int], int, Optional[int]]] = []
    if args.stage in ("search", "all"):
        for seed in SEEDS:
            plan.append(("search", "probe", None, int(seed), FC_DIM))
    search_done = {int(k) for k, v in ledger["search"].items() if v.get("chosen_N")}
    if args.stage in ("train", "all"):
        # [!] 前置条件强化（离朱 R22 发现的产品侧低危项）：原先只查"search 块非空"，
        #     于是**单个** seed 缺 chosen_N 时计划退化为哨兵值 -1，训练循环随后抛
        #     未捕获的 KeyError（真实 CLI = traceback + 无人类可读 [FAIL]）。
        #     现改为：train 档要求 **9/9** 个 seed 都有合法解，否则在**构造计划之前**
        #     以人类可读报文退码 2（前置条件不满足）；不猜测 N、不部分执行。
        if args.stage == "train":
            bad_search = missing_search_solutions(ledger)
            if bad_search:
                print("[FAIL] train 档的前置条件不满足：台账 search 块缺少合法解"
                      f"（chosen_N 须为正整数、chosen_params 须为整数）—— 缺/非法 seed={bad_search}")
                print("       处置：先执行 `--stage search` 补齐搜索解；本档不猜测 N、"
                      "不部分执行（避免 traceback 与静默丢弃轮次）")
                return 2
        for seed in SEEDS:
            if args.stage == "all" and int(seed) not in search_done:
                plan.append(("train", "neuron3d", -1, int(seed), FC_DIM))
            else:
                chosen = int(ledger["search"].get(str(seed), {}).get("chosen_N", -1))
                plan.append(("train", "neuron3d", chosen, int(seed), FC_DIM))
            plan.append(("train", "mlp", None, int(seed), None))

    print("=" * 118)
    print(f"档位={args.stage}；计划条目 = {len(plan)}")
    for st, arch, n, seed, fc in plan:
        if st == "search":
            print(f"  [search] seed={seed:<6d} N in [{N_SEARCH_LO}, {N_SEARCH_HI}] 逐 N 实测 params"
                  f"（构造探测，不训练）")
        else:
            if arch == "neuron3d" and n == -1:
                # dry-run 时 N 可能尚未搜索出来：如实标注"待搜索"，不要用预设 N 冒充
                n_show = ledger["search"].get(str(seed), {}).get("chosen_N")
                if n_show is None:
                    print(f"  [train ] arch={arch:<9s} N=(待搜索) seed={seed:<6d} fc_dim={fc} "
                          f"-> full_..._N<chosen>_..._fc-1_s{seed}_fc_align.pt")
                    continue
                n = int(n_show)
            nm, _ = expected_artifact(arch, n, seed, fc)
            print(f"  [train ] arch={arch:<9s} N={str(n):>6s} seed={seed:<6d} fc_dim={fc} -> {nm}")
    print("=" * 118)
    if args.dry_run:
        print("[dry-run] 未执行任何训练/构造")
        return 0

    # ---- 搜索档 ----
    repaired_seeds: List[int] = []
    for st, arch, n, seed, fc in plan:
        if st != "search":
            continue
        entry = ledger["search"].get(str(seed))
        need_search = bool(args.force) or entry is None
        # [!] 台账中"该 seed **有记录但记录非法**"（缺 chosen_N / 非正整数 / chosen_params 非整数）时
        #     **不得崩溃**（离朱 R22 第二轮发现 F1）：原实现的 `[跳过]` 打印对
        #     `ledger['search'][str(seed)]['chosen_N']` 直接下标，缺键即抛未捕获 KeyError
        #     -> traceback + 退码 1、无人类可读 [FAIL]（且该行在 R22 前即存在）。
        #     现改为：显式报告该非法记录、**重算**该 seed 的搜索解（不静默沿用），
        #     并把该 seed 记入 `repaired_seeds` 参与最终失败判定（退码 1）——
        #     即"台账输入异常如实判失败，但不让脚本崩溃、且把可算的解算出来落盘"。
        if entry is not None and not args.force:
            cn, cp = entry.get("chosen_N"), entry.get("chosen_params")
            if (not isinstance(cn, int) or isinstance(cn, bool) or cn <= 0
                    or not isinstance(cp, int) or isinstance(cp, bool)):
                print(f"[FAIL] seed={seed} 的台账搜索记录非法（chosen_N={cn!r}、"
                      f"chosen_params={cp!r}）—— 如实报告并**重算**该 seed（不静默沿用非法记录）")
                repaired_seeds.append(int(seed))
                need_search = True
        if need_search:
            print(f"[搜索] seed={seed}：N ∈ [{N_SEARCH_LO}, {N_SEARCH_HI}] 逐 N 构造并实测 params")
            res = search_one_seed(int(seed))
            ledger["search"][str(seed)] = res
            save_ledger(ledger)
            print(f"       -> chosen_N={res['chosen_N']} params={res['chosen_params']} "
                  f"dev={res['chosen_dev_pct']:+.4f}% within_tol={res['within_tol']} "
                  f"({res['elapsed_s']}s)")
            print("       逐 N 实测表（构造期量，非训练实测）：N:params:dev%")
            for row in res["table"]:
                print(f"         N={row[0]:>4d} params={row[1]:>9d} dev={row[2]:+9.4f}%")
        else:
            # 走到这里 entry 必为合法记录（need_search=False 且 entry is not None），
            # 仍用查表取值以防将来改动引入同类下标崩溃。
            print(f"[跳过] seed={seed} 的 N 搜索已完成"
                  f"（chosen_N={int((entry or {}).get('chosen_N', -1))}）")

    # ---- 训练档 ----
    unsolved_seeds: List[int] = []
    for st, arch, n, seed, fc in plan:
        if st != "train":
            continue
        if arch == "neuron3d" and n == -1:
            # 计划中的 -1 是"待搜索"哨兵（仅 `--stage all` 且该 seed 当时尚无解时出现）。
            # 若搜索档最终仍未给出合法解，这里**如实报错并跳过该轮**（不抛 KeyError 崩溃）。
            solved = int((ledger["search"].get(str(seed)) or {}).get("chosen_N", -1))
            if solved <= 0:
                print(f"[FAIL] seed={seed} 的 N 搜索解缺失或非法"
                      f"（台账 search['{seed}'].chosen_N）—— 跳过该轮，不猜测 N、不用预设值顶替")
                unsolved_seeds.append(int(seed))
                continue
            n = solved
        run_round(st, arch, n, seed, fc, ledger, force=args.force)

    meta["post_run_sha256"] = baseline_snapshot()
    problems = baseline_problems(baseline)
    meta["zero_regression"] = not problems
    meta["finished_at"] = utc_now()
    ledger["summary"] = summarize_ledger(ledger)
    save_ledger(ledger)

    ok = print_summary(ledger)
    print("-" * 118)
    if problems:
        print("[FAIL] 既有产物 SHA256 零回归断言失败：")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"[ ok ] 既有产物零回归：{len(baseline)} 个文件 SHA256 逐位不变")
    # [!] 退出码口径：搜索档只要求 9 个 seed 的搜索解都在容差内；训练档额外要求全部轮次成功。
    #     原先直接用 `summary.all_within_tol`（无训练记录时恒 False）会让 `--stage search`
    #     在**完全成功**的情况下退码 1 —— 自检发现并修复。
    search_vals = list((ledger.get("search") or {}).values())
    n3d_recs = [r for r in latest_records(ledger, arch="neuron3d") if r.get("status") == "ok"]
    tol_ok = True
    if search_vals:
        tol_ok = tol_ok and all(bool(v.get("within_tol")) for v in search_vals)
    if n3d_recs:
        tol_ok = tol_ok and all(bool(r.get("within_tol")) for r in n3d_recs)
    print(f"[同参容差] 搜索解 {len(search_vals)} 个 / 训练记录 {len(n3d_recs)} 个；"
          f"偏差全部 <= {DEV_TOL_PCT}%：{tol_ok}")
    # [!] 退出码口径补充（本轮）：除"已存在记录是否成功"外，还须断言**覆盖度** ——
    #     `search` 档 9/9 个 seed 有合法解、`train` 档 18/18 轮齐备（或被 `--allow-partial`
    #     显式降级）。否则"某一轮从未跑过 / 台账里根本没有该条记录"会被 `failed_rounds == []`
    #     报成成功（原实现：`--stage search` 全程零训练也退码 0）。
    cov_problems = coverage_problems(ledger, args.stage, allow_partial=bool(args.allow_partial))
    if unsolved_seeds:
        print(f"[FAIL] 有 {len(unsolved_seeds)} 个 seed 因搜索解缺失被跳过（不猜测 N）：{unsolved_seeds}")
    if repaired_seeds:
        print(f"[FAIL] 有 {len(repaired_seeds)} 个 seed 的台账搜索记录非法并已重算：{repaired_seeds}"
              f"（重算值已落盘；请复核台账后重跑以确认）")
    print(f"[台账] {LEDGER}")
    if not ok or problems or not tol_ok or cov_problems or unsolved_seeds or repaired_seeds:
        if cov_problems:
            print("[FAIL] 覆盖度断言失败：")
            for p in cov_problems:
                print(f"  - {p}")
        return 1
    print("[PASS] 计划内全部轮次成功（**含覆盖度断言**：search 9/9、train 18/18）、"
          "同参偏差全部 <= 0.5% 且既有产物零回归")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())