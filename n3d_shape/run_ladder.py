"""N 阶梯 x 多 seed 同预算对照运行器（顺序驱动，可断点续跑）。

目标
----
在**同一训练预算**（同 preset、同 seed、同数据、同 bs/lr/epochs）下，用 N 阶梯扫描 +
多 seed 重复，回答"三期（n3d_shape）仅通过提高 N 能否达到同预算 MLP 的性能"，
并给出可复核的实测区间与达标点清单。本对照为**同类比较**；**不得**把结果表述为
对架构优劣的因果结论。

固定口径（开工前固定，不得事后挑选或更改）
------------------------------------------
* preset = ``highacc``：epochs=20 / batch_size=128 / lr=2e-3 / AdamW(weight_decay=1e-4) /
  lr_schedule=cosine / grad_clip=1.0 / readout_bias=True。
  **三期 `Config` 无 dropout 字段，不得引入。**
* ``shape=sphere``、**不传** ``--cyl-aspect``、``flow_axis=z``、``H=0.10``、``D=0.10``（=H）、
  **不传** ``--space-radius``（取公式下界 ``R_min``）；
* ``y_in=y_out=8``、``input_dim=784``、``output_dim=10``、``num_workers=0``；
* ``input_scope=readout_scope=any_isolated``；
* N 阶梯 = (256, 512, 1024, 2048, 2976)；seed = (1, 2, 3, 7, 42, 43, 99, 123, 2024)；
* 对照基线：每个 seed 各一次 ``--arch mlp``（784->2048->10，``hidden_dim`` 用默认 2048）；
* 达标判定（**事先固定**）：某 (N, seed) 的 ``test_acc >= 该 seed 的 MLP test_acc``
  即视为该点达标；如实报告全部点，不因结果不利而改口径、不事后择优。

档位
----
* 档 1：N 阶梯 x ``seed=42``（5 次 N3D + 1 次 MLP）；
* 档 2：其余 8 个 seed x **档 1 选出的最优 N**（判据：``test_acc`` 最高，并列取较大 N），
  每个 seed 各 1 次 MLP。

产物防撞名（硬要求）
--------------------
``train.full_checkpoint_name`` **不含 preset / epochs / batch_size / lr**，故不加 tag 时
highacc 运行会**覆盖**既有 ``full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt``
（历史纠正 #1 的同类缺陷）。因此**全部** N3D 运行带 ``--tag ladder_hacc``。
MLP 运行必须用**另一个** tag（``ladder_hacc_mlp``）：``full_checkpoint_name`` 同样
**不含 arch**，若 N3D 与 MLP 共用同一 tag，同 N/seed 的两种架构会写同一文件名而互相覆盖
（违反"产物名两两唯一"硬要求）；两个 tag 均以 ``ladder_hacc`` 为前缀，防撞名意图一致。
MLP 运行**不传** ``--n``（``N`` 对 MLP 无意义，沿用预设 N=256 以保持产物名稳定）。

硬约束
------
* 既有 10 个 ``checkpoints/n3d_shape/*.pt`` 与 ``checkpoints/_control/mlp_highacc_ep12_seed42.pt``
  的 SHA256 **逐位不变**（运行前后各取一次快照并断言）；
* 只通过既有 CLI 参数（``--preset`` / ``--n`` / ``--seed`` / ``--tag`` / ``--arch``）驱动，
  **不改** ``train.py`` 的既有默认行为与既有产物命名规则；
* 不覆盖任何既有产物（含 ``_control/`` 下的一期 MLP 产物）。

用法
----
    python n3d_shape/run_ladder.py --stage 1      # 档 1（5 次 N3D + 1 次 MLP）
    python n3d_shape/run_ladder.py --stage 2      # 档 2（8 x (1 N3D + 1 MLP)）
    python n3d_shape/run_ladder.py --stage all
    python n3d_shape/run_ladder.py --dry-run      # 只打印计划，不训练
    python n3d_shape/run_ladder.py --extract <ckpt>   # 内部用：打印单产物实测 JSON

台账：``checkpoints/n3d_shape/_verify/ladder_runs.json``（全部现场取数，禁止手填）。
逐轮日志：``checkpoints/n3d_shape/_verify/log_ladder_{arch}_N{N}_s{seed}.txt``
（显式 UTF-8 写入，末尾写入**真实退出码**）。
"""

from __future__ import annotations

import argparse
import datetime
import contextlib
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
LEDGER = os.path.join(VERIFY_DIR, "ladder_runs.json")

# ----------------------------------------------------------------------
# 固定口径（开工前固定；台账与 README §17 均以本处常量为准）
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
N_LADDER: Tuple[int, ...] = (256, 512, 1024, 2048, 2976)
SEEDS: Tuple[int, ...] = (1, 2, 3, 7, 42, 43, 99, 123, 2024)
PILOT_SEED = 42
N3D_TAG = "ladder_hacc"
MLP_TAG = "ladder_hacc_mlp"
BASELINE_TAG_MARKERS = ("_ladder_hacc",)
CONTROL_ARTIFACT = os.path.join(CONTROL_DIR, "mlp_highacc_ep12_seed42.pt")

# ======================================================================
# 基础工具
# ======================================================================
def utc_now() -> str:
    """返回当前 UTC 时间（ISO 8601，秒精度）。"""
    return datetime.datetime.now(datetime.timezone.utc).replace(
        microsecond=0
    ).isoformat()


def sha256_of(path: str) -> str:
    """返回文件的 SHA256 十六进制摘要（分块读取，避免大产物一次性进内存）。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def baseline_snapshot() -> Dict[str, str]:
    """对"既有产物"取 SHA256 快照（零回归断言的基准）。

    集合 = ``checkpoints/n3d_shape/*.pt`` 中**不含** ``_ladder_hacc`` 标记的文件
    （即本轮开工前的 10 个三期产物）+ ``checkpoints/_control/mlp_highacc_ep12_seed42.pt``。
    排除本轮的 N3D/MLP 产物（两者 tag 均以 ``ladder_hacc`` 开头），
    使得**断点续跑**时快照集合与首轮一致。

    返回
    ----
    Dict[str, str]
        ``{绝对路径: SHA256}``（键按路径排序，保证顺序确定）。
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
    """现场重算既有产物 SHA256 并与快照比对，返回不一致描述列表（空列表 = 零回归）。"""
    problems: List[str] = []
    for path, sha in expected.items():
        if not os.path.isfile(path):
            problems.append(f"[零回归] 既有产物消失：{path}")
            continue
        got = sha256_of(path)
        if got != sha:
            problems.append(f"[零回归] 既有产物被改写：{path}（{sha[:12]} -> {got[:12]}）")
    return problems


# ======================================================================
# 命令行 / 产物命名（复用 train.py 自身的解析与命名，避免重复实现指纹格式）
# ======================================================================
def load_train_module() -> Any:
    """导入本模块自身的 ``train``（复用其 ``parse_args`` / ``apply_overrides`` / 命名函数）。"""
    sys.path.insert(0, MODULE_DIR)
    try:
        from . import train as train_mod  # type: ignore

        return train_mod
    except ImportError:  # pragma: no cover - 以脚本方式直接运行本文件时
        import train as train_mod  # type: ignore

        return train_mod


def build_argv(arch: str, n: Optional[int], seed: int) -> List[str]:
    """构造单轮训练的 CLI 参数（解释器与脚本路径之外的纯参数部分）。

    参数
    ----
    arch : str
        ``neuron3d`` / ``mlp``。
    n : Optional[int]
        N3D 的神经元数量；``None`` 表示**不传** ``--n``（MLP 专用：N 对 MLP 无意义）。
    seed : int
        随机种子。

    返回
    ----
    List[str]
        形如 ``["--preset", "highacc", ...]`` 的参数列表。
    """
    argv = [
        "--preset", PRESET,
        "--arch", arch,
        "--seed", str(seed),
    ]
    if arch == "mlp":
        # MLP 不传 --n / --shape：N 与形状对 MLP 无意义，沿用预设默认（N=256）以保证
        # 产物名在 9 个 seed 之间只差 seed 段，便于目视核对。
        argv += ["--tag", MLP_TAG]
        return argv
    argv += [
        "--shape", SHAPE,
        "--n", str(int(n)),
        "--tag", N3D_TAG,
    ]
    return argv


def full_command(arch: str, n: Optional[int], seed: int) -> List[str]:
    """返回可直接执行的完整命令行（``[解释器, -X, utf8, train.py, *参数]``）。"""
    return [sys.executable, "-X", "utf8", TRAIN_PY] + build_argv(arch, n, seed)


def expected_artifact(arch: str, n: Optional[int], seed: int) -> Tuple[str, str]:
    """返回该轮训练的**规范产物名**与绝对路径（复用 train.py 的命名代码路径）。

    说明：本函数通过 ``train.parse_args`` + ``train.apply_overrides`` 复现 ``main()``
    的配置构造，因此得到的名字与子进程实际写入的名字**同源**（不是手抄指纹格式）。

    返回
    ----
    Tuple[str, str]
        ``(文件名, 绝对路径)``。
    """
    train_mod = load_train_module()
    # train.apply_overrides 内部会经 log_info 打印覆盖明细；此处吞掉，
    # 保持运行器自身的 stdout 只剩"计划/进度/汇总"，便于落日志与阅读。
    with contextlib.redirect_stdout(io.StringIO()):
        args = train_mod.parse_args(build_argv(arch, n, seed))
        cfg = train_mod.apply_overrides(train_mod.PRESETS[args.preset], args)
    if train_mod.is_default_config(cfg):
        name = "model.pt"
    else:
        name = train_mod.full_checkpoint_name(cfg, args.tag)
    return name, os.path.join(train_mod.CHECKPOINT_DIR, name)


def log_name(arch: str, n: Optional[int], seed: int) -> str:
    """逐轮日志文件名（含 N / seed / arch 三段，便于核查）。"""
    n_part = "Npreset" if n is None else f"N{int(n)}"
    return f"log_ladder_{arch}_{n_part}_s{seed}.txt"


# ======================================================================
# 台账读写
# ======================================================================
def load_ledger() -> Dict[str, Any]:
    """读取台账；不存在时返回空壳（``_meta`` + 空 ``runs``）。"""
    if os.path.isfile(LEDGER):
        with open(LEDGER, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"_meta": {}, "runs": []}


def save_ledger(ledger: Dict[str, Any]) -> None:
    """把台账以 UTF-8（无 BOM）写回磁盘。"""
    os.makedirs(VERIFY_DIR, exist_ok=True)
    with open(LEDGER, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(ledger, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def find_run(ledger: Dict[str, Any], arch: str, n: Optional[int], seed: int) -> Optional[Dict[str, Any]]:
    """在台账中查找 (arch, N, seed) 对应的轮次记录（按写入顺序取最后一条）。"""
    hit = None
    for rec in ledger.get("runs", []):
        if rec.get("arch") == arch and rec.get("seed") == seed and rec.get("N") == n:
            hit = rec
    return hit

# ======================================================================
# 单产物实测指标提取（在**独立子进程**内完成，构造后即随进程退出释放内存）
# ======================================================================
def extract_metrics(path: str) -> Dict[str, Any]:
    """从产物现场提取实测指标（供 ``--extract`` 子进程调用）。

    口径
    ----
    * ``test_acc`` 取自产物内的训练结果（训练进程写入）；
    * ``E`` / ``K`` / ``|S_in|`` / ``|S_out|`` / ``params`` 由**现场重建模型**得到
      （与 ``verify_full_runs.py`` 同口径），同时与产物内自带的
      ``connection_stats`` / ``topology_stats`` 交叉比对；
    * 内存：先取出小字段并 ``del`` 掉巨大的 ``model_state_dict``（``syn_dist`` 达
      ``(N*y)^2`` 个 float32）再构造模型，避免"装载 + 构造"两份大张量同时驻留。

    参数
    ----
    path : str
        产物绝对路径。

    返回
    ----
    Dict[str, Any]
        实测指标字典（可直接 JSON 序列化）。
    """
    import gc

    import torch

    sys.path.insert(0, MODULE_DIR)
    try:
        from .config import Config  # type: ignore
        from .model import MLPBaseline, ThreeDNeuronSpace  # type: ignore
    except ImportError:  # pragma: no cover
        from config import Config  # type: ignore
        from model import MLPBaseline, ThreeDNeuronSpace  # type: ignore

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
        "ckpt_shape_tag": str(d.get("shape_tag", "")),
        "ckpt_selection_metric": float(d.get("selection_metric", float("nan"))),
        "ckpt_batches_per_epoch": d.get("batches_per_epoch"),
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
            "test_acc": small["test_acc"],
            "E": int(stats["num_edges"]),
            "K": int(topo["num_layers_true"]) if topo else 0,
            "S_in": int(stats["num_in_scope"]),
            "S_out": int(stats["num_out_scope"]),
            "params": int(model.count_parameters()),
            "topology_matches_checkpoint": bool(
                int(stats["num_edges"]) == small["ckpt_E"]
                and int(stats["num_in_scope"]) == small["ckpt_S_in"]
                and int(stats["num_out_scope"]) == small["ckpt_S_out"]
                and (int(topo["num_layers_true"]) if topo else 0) == small["ckpt_K"]
            ),
        }
    )
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
        return {
            "extract_error": f"exit={proc.returncode}",
            "extract_stderr": (proc.stderr or "")[-2000:],
        }
    line = [ln for ln in (proc.stdout or "").splitlines() if ln.startswith("{")]
    if not line:
        return {"extract_error": "no-json", "extract_stdout": (proc.stdout or "")[-2000:]}
    return json.loads(line[-1])


def scan_log_health(log_path: str, test_acc: float) -> List[str]:
    """扫描日志与 ``test_acc``，返回 loss NaN/Inf 等健康问题列表（空 = 健康）。

    参数
    ----
    log_path : str
        该轮训练日志路径。
    test_acc : float
        产物内记录的最终测试准确率。

    返回
    ----
    List[str]
        问题描述（例如 ``"log 命中 loss=nan"``）；空列表表示未发现异常。
    """
    issues: List[str] = []
    if not math.isfinite(float(test_acc)):
        issues.append(f"test_acc 非有限值：{test_acc}")
    if os.path.isfile(log_path):
        txt = open(log_path, "r", encoding="utf-8", errors="replace").read().lower()
        for token in ("loss=nan", "loss=inf", "loss=-inf", "test_acc=nan", "test_acc=inf"):
            if token in txt:
                issues.append(f"日志命中 {token}")
    return issues


# ======================================================================
# 单轮执行
# ======================================================================
def run_round(
    stage: int,
    arch: str,
    n: Optional[int],
    seed: int,
    ledger: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    """执行（或跳过）单轮训练，返回并落库该轮台账记录。

    断点续跑
    --------
    若台账已有该 ``(arch, N, seed)`` 的成功记录、产物存在、且**磁盘 SHA256 == 台账 SHA256**，
    则**跳过训练**（不重复训练）；``force=True`` 时强制重跑。

    参数
    ----
    stage : int
        档位（1 / 2）。
    arch : str
        架构（``neuron3d`` / ``mlp``）。
    n : Optional[int]
        N3D 的 N；MLP 为 ``None``。
    seed : int
        随机种子。
    ledger : Dict[str, Any]
        当前台账（本函数会把新记录追加进去并落盘）。
    force : bool
        是否强制重跑。

    返回
    ----
    Dict[str, Any]
        本轮台账记录。
    """
    name, path = expected_artifact(arch, n, seed)
    lg = log_name(arch, n, seed)
    lg_path = os.path.join(VERIFY_DIR, lg)
    prev = find_run(ledger, arch, n, seed)
    if not force and prev and prev.get("exit_code") == 0 and os.path.isfile(path):
        sha = sha256_of(path)
        if sha == prev.get("sha256"):
            print(
                f"[跳过] (arch={arch}, N={n}, seed={seed}) 已完成且 SHA256 与台账一致"
                f"（{sha[:12]}），不重复训练"
            )
            return prev
    os.makedirs(VERIFY_DIR, exist_ok=True)
    cmd = full_command(arch, n, seed)
    print(f"[运行] (arch={arch}, N={n}, seed={seed}) -> {name}")
    t0 = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with open(lg_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"[LADDER] stage={stage} arch={arch} N={n} seed={seed}\n")
        fh.write(f"[LADDER] cwd={PROJECT_ROOT}\n")
        fh.write("[LADDER] cmd=" + " ".join(cmd) + "\n")
        fh.write(f"[LADDER] start_utc={utc_now()}\n")
        fh.flush()
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env
        )
        rc = int(proc.returncode)
        fh.flush()
        fh.write(f"[LADDER] end_utc={utc_now()}\n")
        fh.write(f"[LADDER] exit_code={rc}\n")
    elapsed = time.perf_counter() - t0
    rec: Dict[str, Any] = {
        "stage": stage,
        "arch": arch,
        "N": n,
        "seed": seed,
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
        rec["N_note"] = "MLP 不使用 N（hidden_dim=2048）；产物名与产物内 config 沿用预设 N=256"
    if rc != 0:
        rec["status"] = "failed"
        rec["failure_reason"] = f"训练进程退出码 {rc}（不静默重试、不用其他配置顶替）"
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
            rec["test_acc"] = float(m["test_acc"])
            rec["E"] = int(m["E"])
            rec["K"] = int(m["K"])
            rec["S_in"] = int(m["S_in"])
            rec["S_out"] = int(m["S_out"])
            rec["params"] = int(m["params"])
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
        f"test_acc={rec.get('test_acc')} elapsed={rec['elapsed_s']}s"
    )
    return rec

# ======================================================================
# 档位计划 / 达标判定 / 汇总统计
# ======================================================================
def caliber_dict() -> Dict[str, Any]:
    """返回**开工前固定**的口径字典（写入台账 ``_meta.caliber``，用于防事后改口径）。"""
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


def latest_records(
    ledger: Dict[str, Any], arch: Optional[str] = None, n: Optional[int] = None
) -> List[Dict[str, Any]]:
    """按 ``(arch, N, seed)`` 去重取**最后一条**记录（重跑只认最新一次）。"""
    latest: Dict[Tuple[str, Any, int], Dict[str, Any]] = {}
    for rec in ledger.get("runs", []):
        if arch is not None and rec.get("arch") != arch:
            continue
        if n is not None and rec.get("N") != n:
            continue
        latest[(str(rec.get("arch")), rec.get("N"), int(rec.get("seed")))] = rec
    return list(latest.values())


def pick_best_n(ledger: Dict[str, Any]) -> Optional[int]:
    """档 1（``seed=42``）选出的最优 N：``test_acc`` 最高，并列取较大 N。"""
    pool = [
        r for r in latest_records(ledger, arch="neuron3d")
        if int(r.get("seed")) == PILOT_SEED and r.get("N") in N_LADDER
        and r.get("status") == "ok"
    ]
    if not pool:
        return None
    best = max(pool, key=lambda r: (float(r["test_acc"]), int(r["N"])))
    return int(best["N"])


def stage_plan(ledger: Dict[str, Any], stage: str) -> Tuple[List[Tuple[int, str, Optional[int], int]], Optional[int]]:
    """返回该档的轮次计划 ``[(stage, arch, N, seed), ...]`` 与档 1 最优 N。"""
    rounds: List[Tuple[int, str, Optional[int], int]] = []
    best: Optional[int] = None
    if stage in ("1", "all"):
        for n in N_LADDER:
            rounds.append((1, "neuron3d", int(n), PILOT_SEED))
        rounds.append((1, "mlp", None, PILOT_SEED))
    if stage in ("2", "all"):
        best = pick_best_n(ledger)
        if best is not None:
            for seed in SEEDS:
                if seed == PILOT_SEED:
                    continue
                rounds.append((2, "neuron3d", int(best), int(seed)))
                rounds.append((2, "mlp", None, int(seed)))
    return rounds, best


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    """返回 mean / min / max / 样本标准差（``ddof=1``；``n<2`` 时标准差记为 None）。"""
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None, "stdev": None}
    mean = sum(vals) / len(vals)
    stdev = None
    if len(vals) >= 2:
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        stdev = var ** 0.5
    return {
        "n": len(vals), "mean": mean, "min": min(vals), "max": max(vals),
        "stdev": stdev, "stdev_ddof": 1,
    }


def print_summary(ledger: Dict[str, Any]) -> bool:
    """打印逐 (N, seed) 实测表、每 N 的 seed 区间统计、达标点清单。

    返回
    ----
    bool
        是否存在失败轮次或口径内不完整项（``True`` = 全部齐备且无失败轮）。
    """
    n3d = latest_records(ledger, arch="neuron3d")
    mlp = {int(r["seed"]): r for r in latest_records(ledger, arch="mlp")}
    failed = [r for r in n3d + list(mlp.values()) if r.get("status") != "ok"]
    print("=" * 118)
    print("逐 (N, seed) 实测表（test_acc 来自产物现场取数）")
    print("=" * 118)
    print(f"{'N':>6s} {'seed':>6s} {'acc':>8s} {'E':>7s} {'K':>4s} {'S_in':>6s} "
          f"{'S_out':>6s} {'params':>9s} {'MLP acc':>8s} {'达标':>6s}")
    pass_list: List[Tuple[int, int, float, float]] = []
    by_n: Dict[int, List[Dict[str, Any]]] = {}
    for r in sorted(n3d, key=lambda r: (int(r["N"]), int(r["seed"]))):
        if r.get("status") != "ok":
            print(f"{r.get('N')!s:>6s} {int(r['seed']):>6d}  FAILED：{r.get('failure_reason')}")
            continue
        m = mlp.get(int(r["seed"]))
        macc = float(m["test_acc"]) if m and m.get("status") == "ok" else None
        ok = (macc is not None) and (float(r["test_acc"]) >= macc)
        if ok:
            pass_list.append((int(r["N"]), int(r["seed"]), float(r["test_acc"]), float(macc)))
        print(f"{int(r['N']):>6d} {int(r['seed']):>6d} {float(r['test_acc']) * 100:>7.2f}% "
              f"{int(r['E']):>7d} {int(r['K']):>4d} {int(r['S_in']):>6d} {int(r['S_out']):>6d} "
              f"{int(r['params']):>9d} "
              f"{(f'{macc * 100:.2f}%' if macc is not None else '   n/a'):>8s} "
              f"{('达标' if ok else '未达标') if macc is not None else '待补':>6s}")
        by_n.setdefault(int(r["N"]), []).append(r)

    print("-" * 118)
    print("MLP 基线（同 preset / 同 seed，N 对 MLP 无意义，产物名沿用预设 N=256）")
    for seed in sorted(mlp):
        m = mlp[seed]
        if m.get("status") == "ok":
            print(f"  seed={seed:>5d} acc={float(m['test_acc']) * 100:>7.2f}% "
                  f"params={int(m['params']):>9d} artifact={m['artifact']}")
        else:
            print(f"  seed={seed:>5d} FAILED：{m.get('failure_reason')}")

    print("-" * 118)
    print("每 N 的 seed 区间统计（test_acc；档 1 只含 seed=42，档 2 只补最优 N 的其余 8 个 seed）")
    print(f"{'N':>6s} {'n_seed':>7s} {'mean':>9s} {'min':>9s} {'max':>9s} {'stdev(ddof=1)':>14s} {'seeds':>40s}")
    for n in sorted(by_n):
        recs = by_n[n]
        st = _stats([float(r["test_acc"]) for r in recs])
        sd = "n/a(n<2)" if st["stdev"] is None else f"{st['stdev'] * 100:.4f}%"
        print(f"{n:>6d} {st['n']:>7d} {st['mean'] * 100:>8.4f}% {st['min'] * 100:>8.4f}% "
              f"{st['max'] * 100:>8.4f}% {sd:>14s} "
              f"{str(sorted(int(r['seed']) for r in recs)):>40s}")

    print("-" * 118)
    print(f"达标点清单（事先固定判据：N3D test_acc >= 同 seed 的 MLP test_acc）——"
          f"共 {len(pass_list)} 个达标点 / {len(n3d) - len(failed)} 个有效点")
    for n, seed, a, b in sorted(pass_list):
        print(f"  达标 N={n:<6d} seed={seed:<6d} N3D={a * 100:.2f}%  MLP={b * 100:.2f}%  "
              f"Δ={100 * (a - b):+.2f} pp")
    if failed:
        print(f"[WARN] 存在 {len(failed)} 个失败轮次（如实呈现，不因结果不利而改口径）：")
        for r in failed:
            print(f"  - arch={r.get('arch')} N={r.get('N')} seed={r.get('seed')} "
                  f"exit={r.get('exit_code')} 原因={r.get('failure_reason')} 日志={r.get('log_path')}")
    return not failed

# ======================================================================
# 入口
# ======================================================================
def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。

    返回
    ----
    int
        退出码：``0`` = 计划内全部轮次成功且既有产物零回归；
        ``1`` = 存在失败轮次或零回归被破坏（失败轮**不静默重试**）；
        ``2`` = 前置条件不满足（档 2 缺档 1 结果 / 台账口径漂移）。
    """
    parser = argparse.ArgumentParser(
        description="N 阶梯 x 多 seed 同预算对照运行器（顺序驱动，可断点续跑）"
    )
    parser.add_argument("--stage", choices=["1", "2", "all"], default="all",
                        help="运行档位：1 = N 阶梯 x seed=42；2 = 其余 seed x 最优 N；all = 两档顺序执行")
    parser.add_argument("--force", action="store_true", help="已完成的轮次也强制重跑")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行训练")
    parser.add_argument("--extract", default="", help="[内部] 打印单个产物的实测指标 JSON 后退出")
    args = parser.parse_args(argv)

    if args.extract:
        print(json.dumps(extract_metrics(args.extract), ensure_ascii=False))
        return 0

    os.makedirs(VERIFY_DIR, exist_ok=True)
    ledger = load_ledger()
    meta = ledger.setdefault("_meta", {})
    cal = caliber_dict()
    if not meta.get("caliber"):
        meta["caliber"] = cal
        meta["created_at"] = utc_now()
        meta["baseline_sha256"] = baseline_snapshot()
        meta["baseline"] = {
            "rule": "checkpoints/n3d_shape/*.pt 中不含 ladder_hacc 标记者 + checkpoints/_control/"
                    "mlp_highacc_ep12_seed42.pt",
            "n_files": len(meta["baseline_sha256"]),
            "captured_at": utc_now(),
        }
        save_ledger(ledger)
        print(f"[台账] 初始化 {LEDGER}；既有产物基线快照 {len(meta['baseline_sha256'])} 个"
              f"（应为 10 个三期产物 + 1 个一期 MLP 产物）")
    elif meta.get("caliber") != cal:
        print("[FAIL] 台账口径与当前 run_ladder.py 的固定口径不一致 —— 禁止事后改口径；"
              "如需改口径请新建台账文件并说明原因")
        return 2

    baseline: Dict[str, str] = meta["baseline_sha256"]
    pre = baseline_problems(baseline)
    if pre:
        print("[FAIL] 运行前既有产物零回归断言已失败：")
        for p in pre:
            print(f"  - {p}")
        return 1

    rounds, best = stage_plan(ledger, args.stage)

    print("=" * 118)
    print(f"档位={args.stage}；档 1 最优 N = {best}（判据：test_acc 最高，并列取较大 N）；"
          f"计划轮次 = {len(rounds)}")
    for st, arch, n, seed in rounds:
        nm, _ = expected_artifact(arch, n, seed)
        print(f"  [档{st}] arch={arch:<9s} N={str(n):>6s} seed={seed:<6d} -> {nm}")
    print("=" * 118)
    if args.dry_run:
        print("[dry-run] 未执行任何训练")
        return 0
    if args.stage in ("2", "all") and best is None:
        print("[FAIL] 档 2 的前置条件不满足：台账中没有档 1（seed=42, N 阶梯）的成功记录")
        return 2

    for st, arch, n, seed in rounds:
        run_round(st, arch, n, seed, ledger, force=args.force)

    meta["post_run_sha256"] = baseline_snapshot()
    problems = baseline_problems(baseline)
    meta["zero_regression"] = not problems
    meta["finished_at"] = utc_now()
    ledger["summary"] = summarize_ledger(ledger)
    save_ledger(ledger)

    all_ok = print_summary(ledger)
    print("-" * 118)
    if problems:
        print("[FAIL] 既有产物 SHA256 零回归断言失败：")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"[ ok ] 既有产物零回归：{len(baseline)} 个文件 SHA256 逐位不变")
    print(f"[台账] {LEDGER}")
    if not all_ok or problems:
        return 1
    print("[PASS] 计划内全部轮次成功，且既有产物零回归")
    return 0


def summarize_ledger(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """生成台账的机读汇总块（每 N 的 seed 区间统计 + 达标点清单）。"""
    n3d = latest_records(ledger, arch="neuron3d")
    mlp = {int(r["seed"]): r for r in latest_records(ledger, arch="mlp")}
    by_n: Dict[int, List[Dict[str, Any]]] = {}
    passed: List[Dict[str, Any]] = []
    for r in n3d:
        if r.get("status") != "ok":
            continue
        by_n.setdefault(int(r["N"]), []).append(r)
        m = mlp.get(int(r["seed"]))
        if m and m.get("status") == "ok":
            a, b = float(r["test_acc"]), float(m["test_acc"])
            if a >= b:
                passed.append({"N": int(r["N"]), "seed": int(r["seed"]),
                               "n3d_acc": a, "mlp_acc": b, "delta_pp": 100.0 * (a - b)})
    per_n = {}
    for n, recs in sorted(by_n.items()):
        st = _stats([float(r["test_acc"]) for r in recs])
        per_n[str(n)] = {
            "n_seed": st["n"], "acc_mean": st["mean"], "acc_min": st["min"],
            "acc_max": st["max"], "acc_stdev_ddof1": st["stdev"],
            "seeds": sorted(int(r["seed"]) for r in recs),
        }
    return {
        "per_N_stats": per_n,
        "pass_points": sorted(passed, key=lambda d: (d["N"], d["seed"])),
        "n_pass_points": len(passed),
        "n_valid_points": sum(1 for r in n3d if r.get("status") == "ok"),
        "failed_rounds": [
            {"arch": r.get("arch"), "N": r.get("N"), "seed": r.get("seed"),
             "exit_code": r.get("exit_code"), "reason": r.get("failure_reason"),
             "log": r.get("log")}
            for r in ledger.get("runs", []) if r.get("status") != "ok"
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())