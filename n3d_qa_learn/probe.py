"""P0 探针（硬门禁）与单条端到端演练。

两级门禁（**必须先全绿再放全量**）
----------------------------------
1. **P0 探针**：现场构造三个后端模型（``n3d_shape`` / ``n3d_sphere`` / ``n3d_proto``），
   记录参数量 / ``E`` / ``K`` / ``|S_in|`` / ``|S_out|``；构造失败者给出可读原因并
   **显式登记为不可用**（禁止静默丢弃）。要求：**至少一个后端可用**，否则退码 1。
2. **单条端到端演练**：取 1 个 batch 做前向 + 反向 + 一步更新，断言**可学习参数梯度非零**；
   演练通过后才允许放全量训练。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .backends import BACKEND_NAMES, BackendRegistry, build_registry
from .encoders import ROLE_QUESTION, EncoderConfig, declared_dim
from .heads import N3DQA, N3DQAConfig


def probe_backends(
    dim: int,
    names: Sequence[str] = BACKEND_NAMES,
) -> Dict[str, Any]:
    """P0 探针：逐个现场构造后端并记录结构量（失败者显式登记不可用）。

    参数
    ----
    dim : int
        连接参数 ``D``。
    names : Sequence[str]
        要探测的后端名。

    返回
    ----
    Dict[str, Any]
        ``{"dim", "n_requested", "n_available", "entries", "unavailable", "all_failed"}``。
    """
    entries: List[Dict[str, Any]] = []
    unavailable: Dict[str, str] = {}
    for name in names:
        t0 = time.time()
        try:
            reg = BackendRegistry(int(dim))
            adapter = reg.register(name)
            stats = adapter.topology_stats()
            entries.append(
                {
                    "backend": name,
                    "available": True,
                    "parameters": int(adapter.count_parameters()),
                    "E": float(stats.get("E", float("nan"))),
                    "K": float(stats.get("K", float("nan"))),
                    "S_in": float(stats.get("S_in", float("nan"))),
                    "S_out": float(stats.get("S_out", float("nan"))),
                    "build_seconds": float(time.time() - t0),
                    "feature_source": adapter.spec.feature_source,
                }
            )
        except Exception as exc:  # noqa: BLE001 - 失败原因必须原样登记
            unavailable[name] = f"{type(exc).__name__}: {exc}"
            entries.append(
                {
                    "backend": name,
                    "available": False,
                    "reason": unavailable[name],
                    "build_seconds": float(time.time() - t0),
                }
            )
    return {
        "dim": int(dim),
        "n_requested": int(len(list(names))),
        "n_available": int(sum(1 for e in entries if e["available"])),
        "entries": entries,
        "unavailable": dict(unavailable),
        "all_failed": bool(sum(1 for e in entries if e["available"]) == 0),
    }


@dataclass
class DrillResult:
    """单条端到端演练结果。"""

    backend: str
    output_mode: str
    loss_before: float
    loss_after: float
    grad_norms: Dict[str, float]
    zero_grad_params: List[str]
    param_names: List[str]
    n_param_changed: int
    max_param_delta: float
    logits_shape: Tuple[int, ...]
    seconds: float

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化。"""
        return {
            "backend": self.backend,
            "output_mode": self.output_mode,
            "loss_before": float(self.loss_before),
            "loss_after": float(self.loss_after),
            "grad_norms": {k: float(v) for k, v in self.grad_norms.items()},
            "zero_grad_params": list(self.zero_grad_params),
            "param_names": list(self.param_names),
            "n_param_changed": int(self.n_param_changed),
            "max_param_delta": float(self.max_param_delta),
            "logits_shape": [int(x) for x in self.logits_shape],
            "seconds": float(self.seconds),
        }


def end_to_end_drill(
    dim: int,
    backend: str = "n3d_shape",
    output_mode: str = "index",
    n_answers: int = 4,
    n_classes_pointer: int = 3,
    batch_size: int = 8,
    seed: int = 42,
    *,
    encoder: str = "",
    role: str = ROLE_QUESTION,
) -> DrillResult:
    """单条端到端演练：1 batch 前向 + 反向 + 一步更新，断言梯度非零。

    参数
    ----
    dim : int
        连接参数 ``D``。
    backend : str
        后端名。
    output_mode : str
        ``index`` / ``pointer``。
    n_answers : int
        ``index`` 模式的答案类别数 ``C``。
    n_classes_pointer : int
        ``pointer`` 模式的候选取样宽度（``>= 2``，含末位「不相关」位）。
    batch_size : int
        演练批大小。
    seed : int
        种子（固定随机输入）。
    encoder : str
        **可插拔特征实现的注册表键**（空串 = 角色默认实现）；用于把 ``dim`` 与
        编码器注册表声明对账 —— 连接参数 ``D`` 的唯一来源是编码器注册表。
    role : str
        角色（决定声明维度与 ``max_length``）。

    返回
    ----
    DrillResult
        演练结果。

    异常
    ------
    ValueError
        ``dim`` 与编码器注册表声明维度不一致。
    RuntimeError
        任一可学习参数的梯度为 ``None`` 或全零（**硬门禁**）。
    """
    torch.manual_seed(int(seed))
    declared = int(declared_dim(EncoderConfig(name=str(encoder), role=str(role))))
    if int(dim) != declared:
        raise ValueError(
            f"演练的 dim={dim} 与编码器注册表声明维度 {declared} 不一致"
            "（连接参数唯一来源 = 可插拔编码器注册表）"
        )
    registry = build_registry(input_dim=int(dim))
    if backend not in registry.available:
        raise RuntimeError(
            f"后端 {backend!r} 不可用；已登记 = {registry.available}，"
            f"不可用 = {registry.unavailable}"
        )
    adapter = registry.get(backend)
    model = N3DQA(
        adapter,
        int(n_answers),
        N3DQAConfig(dim=int(dim), output_mode=str(output_mode)),
    )
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    x = torch.randn(int(batch_size), int(dim))
    if output_mode == "index":
        keys = None
        targets = torch.tensor(
            [(i * 3) % (int(n_answers) + 1) for i in range(int(batch_size))],
            dtype=torch.long,
        )
        # 候选键表必须是**有结构的**才能产生非零梯度：若候选键是随机初始化且
        # `normalize_query=True`，则全部候选键等概率、交叉熵对各候选的偏导相互抵消，
        # `q` 头首步梯度会**结构性为 0**（实测：RandomInit 下 grad_norm 全 0）。
        # 这里按 batch 自身的类别质心写入候选键（与 `answer_table_mode="centroid"` 同口径），
        # 使演练真正检验"参数是否拿到梯度"而不是检验初始化的对称性。
        if model.config.answer_table_mode == "centroid":
            model.set_answer_table_from_centroids(x, targets, int(n_answers))
    else:
        width = max(2, int(n_classes_pointer))
        keys = torch.randn(int(batch_size), width, int(dim))
        targets = torch.tensor(
            [i % width for i in range(int(batch_size))], dtype=torch.long
        )

    t0 = time.time()
    before = {
        n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
    }
    logits = model.logits(x, keys)
    loss_before = model.cross_entropy(logits, targets)
    opt.zero_grad(set_to_none=True)
    loss_before.backward()

    grad_norms: Dict[str, float] = {}
    zero_grad: List[str] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            grad_norms[n] = 0.0
            zero_grad.append(n)
            continue
        value = float(p.grad.norm().item())
        grad_norms[n] = value
        if value == 0.0:
            zero_grad.append(n)
    opt.step()
    with torch.no_grad():
        loss_after = model.cross_entropy(model.logits(x, keys), targets)
    changed = 0
    max_delta = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        delta = float((p.detach() - before[n]).abs().max().item())
        max_delta = max(max_delta, delta)
        if delta > 0.0:
            changed += 1
    seconds = time.time() - t0

    if zero_grad:
        raise RuntimeError(
            "端到端演练失败：以下可学习参数的梯度为 None 或全零 "
            f"{zero_grad}（参数全集 = {sorted(before.keys())}）"
        )
    if changed == 0:
        raise RuntimeError("端到端演练失败：一步更新后没有任何参数发生变化")

    return DrillResult(
        backend=backend,
        output_mode=output_mode,
        loss_before=float(loss_before.item()),
        loss_after=float(loss_after.item()),
        grad_norms=grad_norms,
        zero_grad_params=[],
        param_names=sorted(before.keys()),
        n_param_changed=int(changed),
        max_param_delta=float(max_delta),
        logits_shape=tuple(int(x) for x in logits.shape),
        seconds=float(seconds),
    )


__all__ = ["probe_backends", "DrillResult", "end_to_end_drill"]