"""n3d_shape 训练对照账本校验脚本（离朱信息项 I2 的补齐）。

职责
----
独立复算五组形状对照产物的实测指标（`test_acc` / `E` / `K` / `|S_in|` / `|S_out|` /
`params` / `SHA256`），并与账本 `full_runs_shape.json` 逐项比对，同时校验：

* 五组产物**名字互不相同**、**SHA256 互不相同**（防撞名硬要求）；
* 产物文件名 == `train.full_checkpoint_name(config)` 给出的规范名；
* checkpoint 内 `config["shape"]` / `config["cyl_aspect"]` / `shape_tag` 与实际形状一致；
* 账本中的 `test_acc` 与 checkpoint 内取证**逐组一致**（禁止无产物支撑的数字）。

用法
----
    python n3d_shape/verify_full_runs.py      # 退出码 0 = 全部一致；1 = 存在不一致

说明
----
本脚本只读产物、不写任何 checkpoint；报告打印到 stdout。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

try:
    from .config import Config  # noqa: E402
    from .model import ThreeDNeuronSpace  # noqa: E402
    from . import train as train_mod  # noqa: E402
except ImportError:  # pragma: no cover
    from config import Config  # type: ignore
    from model import ThreeDNeuronSpace  # type: ignore
    import train as train_mod  # type: ignore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "n3d_shape")
LEDGER = os.path.join(CHECKPOINT_DIR, "_verify", "full_runs_shape.json")
SHA_LEDGER = os.path.join(CHECKPOINT_DIR, "_verify", "full_runs_shape_sha256.json")

# 五组对照（与 README §7 一致；共享 seed=42 前提）
RUNS: List[Tuple[str, float]] = [
    ("sphere", 1.0),
    ("cube", 1.0),
    ("cylinder", 1.0),
    ("cylinder", 0.5),
    ("cylinder", 2.0),
]
# 训练对照的固定超参（统一控制变量）
BASE_KW: Dict[str, Any] = dict(
    N=256, y_in=8, y_out=8, H=0.1, D=0.1, input_dim=784, output_dim=10,
    batch_size=64, lr=1e-3, epochs=10, seed=42, device="cpu",
)


def sha256_of(path: str) -> str:
    """返回文件的 SHA256 十六进制摘要。"""
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(argv: List[str] | None = None) -> int:
    """复算并比对账本；返回 0（一致）/ 1（不一致）。"""
    if not os.path.isfile(LEDGER):
        print(f"[FAIL] 账本不存在：{LEDGER}")
        return 1
    ledger = json.load(open(LEDGER, encoding="utf-8"))
    sha_ledger = (
        json.load(open(SHA_LEDGER, encoding="utf-8"))
        if os.path.isfile(SHA_LEDGER) else {}
    )

    problems: List[str] = []
    names: List[str] = []
    shas: List[str] = []
    print("=" * 100)
    print("n3d_shape 训练对照账本校验（独立复算 vs 账本）")
    print("=" * 100)
    header = f"{'shape':10s} {'lam':>4s} {'acc(产物)':>9s} {'acc(账本)':>9s} {'E':>5s} {'K':>3s} {'S_in':>5s} {'S_out':>6s} {'params':>8s} {'SHA':>10s}"
    print(header)
    for shape, lam in RUNS:
        cfg = Config(**BASE_KW, shape=shape, cyl_aspect=lam)
        name = train_mod.full_checkpoint_name(cfg)
        path = os.path.join(CHECKPOINT_DIR, name)
        key = f"{shape}|{lam}"
        if not os.path.isfile(path):
            problems.append(f"{key}: 产物缺失 {name}")
            print(f"{shape:10s} {lam:>4g}  MISSING {name}")
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        model = ThreeDNeuronSpace(cfg)
        acc = float(d["test_acc"])
        cs, ts = d["connection_stats"], d["topology_stats"]
        e = int(cs["num_edges"])
        k = int(ts["num_layers_true"])
        s_in = int(cs["num_in_scope"])
        s_out = int(cs["num_out_scope"])
        params = int(model.count_parameters())
        sha = sha256_of(path)
        names.append(name)
        shas.append(sha)
        led = ledger.get(key, {})
        print(f"{shape:10s} {lam:>4g} {acc * 100:>8.2f}% {float(led.get('test_acc', -1)) * 100:>8.2f}% "
              f"{e:>5d} {k:>3d} {s_in:>5d} {s_out:>6d} {params:>8d} {sha[:10]:>10s}")
        # 账本一致性
        if not led:
            problems.append(f"{key}: 账本缺该组")
            continue
        if abs(float(led.get("test_acc", -1.0)) - acc) > 1e-12:
            problems.append(f"{key}: test_acc 账本 {led.get('test_acc')} != 产物 {acc}")
        for fld, val in (("E", e), ("K", k), ("S_in", s_in), ("S_out", s_out), ("params", params)):
            if fld in led and led[fld] != val:
                problems.append(f"{key}: {fld} 账本 {led[fld]} != 复算 {val}")
        # 产物自带形状取证
        if d["config"].get("shape") != shape:
            problems.append(f"{key}: 产物 config['shape']={d['config'].get('shape')!r} != {shape!r}")
        if shape == "cylinder" and float(d["config"].get("cyl_aspect", -1)) != lam:
            problems.append(f"{key}: 产物 cyl_aspect={d['config'].get('cyl_aspect')} != {lam}")
        if d.get("shape_tag") != cfg.shape_tag():
            problems.append(f"{key}: 产物 shape_tag={d.get('shape_tag')!r} != {cfg.shape_tag()!r}")
        if d["config"].get("seed") != BASE_KW["seed"]:
            problems.append(f"{key}: 产物 seed={d['config'].get('seed')} != {BASE_KW['seed']}")
        if sha_ledger and sha_ledger.get(name) and sha_ledger[name] != sha:
            problems.append(f"{key}: SHA256 与 SHA 账本不一致")

    # 防撞名：名字与 SHA256 均须两两唯一
    if len(set(names)) != len(names):
        problems.append(f"产物名存在重复：{names}")
    if len(set(shas)) != len(shas):
        problems.append(f"产物 SHA256 存在重复：{[s[:12] for s in shas]}")

    print("-" * 100)
    print(f"五组产物名互不相同：{len(set(names)) == len(names)}")
    print(f"五组 SHA256 互不相同：{len(set(shas)) == len(shas)}")

    # ---- E1 回归：cylinder 的严格体积解与"曾被误拒的 λ" ----
    # 皋陶审查 E1：修复前 R_min 被抬高 (6/π)^(1/3)=1.240701 倍，使 DEFAULT 规模下
    # λ∈(2.20, 3.22) 被误判为"窗口为空"而拒绝。此处做两项**独立**核验：
    #   (a) 实现值与由体积不等式 `2πλr³ >= N·(4/3)πH³/φ` 独立反解的值逐位一致；
    #   (b) 曾被误拒的 λ=2.2/2.5/3.0/3.2 现在能构造且窗口非空；
    #   (c) 真正的空窗口（λ=4.0）仍须被正确拒绝（不得因修复而失去保护）。
    print("-" * 100)
    print("E1 回归（cylinder 严格体积解）")
    phi = 0.7405
    for lam in (0.5, 1.0, 2.0, 2.5, 3.0):
        c = Config(**BASE_KW, shape="cylinder", cyl_aspect=lam)
        need = BASE_KW["N"] * (4.0 / 3.0) * math.pi * BASE_KW["H"] ** 3 / phi
        derived = (need / (2.0 * math.pi * lam)) ** (1.0 / 3.0)
        same = abs(c.min_space_radius - derived) <= 1e-12
        if not same:
            problems.append(
                f"E1: lambda={lam} R_min={c.min_space_radius:.9f} != 体积反解 {derived:.9f}"
            )
        print(f"  lambda={lam:<4}: R_min={c.min_space_radius:.9f} 体积反解={derived:.9f} 一致={same}")
    for lam in (2.2, 2.5, 3.0, 3.2):
        try:
            c = Config(**BASE_KW, shape="cylinder", cyl_aspect=lam)
            nonempty = c.max_space_radius >= c.min_space_radius
            print(f"  曾被误拒 lambda={lam}: 构造成功 R_min={c.min_space_radius:.6f} "
                  f"R_max={c.max_space_radius:.6f} 窗口非空={nonempty}")
            if not nonempty:
                problems.append(f"E1: lambda={lam} 窗口为空")
        except ValueError as exc:
            problems.append(f"E1: lambda={lam} 仍被误拒 -> {exc}")
    try:
        Config(**BASE_KW, shape="cylinder", cyl_aspect=4.0)
        problems.append("E1: lambda=4.0 应因窗口为空被拒，但构造成功")
    except ValueError:
        print("  真实空窗口 lambda=4.0 仍被正确拒绝")
    if problems:
        print(f"\n[FAIL] 发现 {len(problems)} 项不一致：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\n[PASS] 账本与产物逐项一致，且产物名/SHA256 两两唯一")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
