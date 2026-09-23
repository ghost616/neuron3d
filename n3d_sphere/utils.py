"""N3D 通用工具层（**历史遗留**：四步闭环算子，当前纯球形分层 DAG 不调用）。

⚠️ 调用现状（皋陶审查 info 项，已如实标注）
------------------------------------------------
本文件由一期 `n3d_proto/utils.py` 拷贝而来，其中**四步闭环的稀疏分组/构图算子
在本模块当前的纯球形分层 DAG 架构中不被调用**：

* **仍在用**：`set_seed`（固定全局随机源，`train.build_model_and_data` 调用）、
  `get_device`、`log_info` / `log_warn` / `log_error`、`count_parameters`、
  `tensor_grad_norms`（冒烟判据用；缺失梯度记为 -1.0）。
* **不再调用**（保留原因：避免改动一期拷贝文件带来回归风险，且复现旧实验仍需）：
  `segment_softmax`、`build_edge_index`、`build_edge_index_from_dist`、
  `build_scatter_matrix`、`sparse_aggregate`、`sparse_broadcast`、
  `connection_density`（及其同义别名 `connection_sparsity`）、`zero_ratio`。

当前架构的连接聚合改为"按拓扑序 + 入边 CSR 区间"的稀疏实现，见 `model.py` 的
`stage2_recurrence`；几何量（距离矩阵等）由 `model.py` 自行构造。

职责
----
* 可复现性：`set_seed`
* 设备管理：`get_device`
* 日志输出：`log_info` / `log_warn` / `log_error`
* 稀疏分组计算：`segment_softmax`（优先 torch_scatter，缺失时纯 PyTorch fallback）
* 拓扑构建：`build_edge_index`（dense mask -> 稀疏 edge_index / edge_dist）、
  `build_scatter_matrix`（神经元 ↔ 突触的聚合/广播算子）
* 参数与统计：`count_parameters` / `tensor_grad_norms` / `connection_sparsity`（连接密度，
  规格口径）/ `zero_ratio`（零元素占比，便于与"稀疏"一词的日常语义对照）

设计约束（历史，属一期四步闭环）
--------------------------------
1. 所有分组（segment）运算必须用 scatter 实现，禁止 Python for 循环遍历突触/神经元；
2. 归一化方向固定为"对每个输出突触，在其连接的输入突触上做 softmax"；
3. 一切随机过程由 seed 控制，保证可复现。
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:  # 优先使用 torch_scatter（若环境已安装）
    import torch_scatter  # type: ignore

    _HAS_TORCH_SCATTER = True
except Exception:  # pragma: no cover - 环境缺失时走纯 PyTorch fallback
    torch_scatter = None  # type: ignore
    _HAS_TORCH_SCATTER = False

__all__ = [
    "HAS_TORCH_SCATTER",
    "set_seed",
    "get_device",
    "log_info",
    "log_warn",
    "log_error",
    "segment_softmax",
    "build_edge_index",
    "build_scatter_matrix",
    "count_parameters",
    "tensor_grad_norms",
    "connection_density",
    "connection_sparsity",
    "zero_ratio",
]

# torch_scatter 是否可用（日志与说明用）
HAS_TORCH_SCATTER: bool = _HAS_TORCH_SCATTER


# ======================================================================
# 可复现性与设备
# ======================================================================
def set_seed(seed: int) -> None:
    """固定所有随机源，保证坐标采样、权重初始化与数据打乱可复现。

    参数
    ----
    seed : int
        随机种子。会同时设置 `random`、`numpy` 与 `torch`(CPU/GPU) 的种子，
        并关闭 cudnn 的非确定性算法（若可用）。

    返回
    ----
    None
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed 必须为 int，当前类型 {type(seed).__name__}")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 关闭非确定性算法，保证多次运行结果一致
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # 限制 CPU 线程过度竞争（小模型下多线程反而更慢）
    os.environ.setdefault("OMP_NUM_THREADS", "4")


def get_device(prefer: str = "auto") -> torch.device:
    """自动选择计算设备。

    参数
    ----
    prefer : str
        "auto"：有 CUDA 则用 cuda，否则 cpu；
        "cpu" / "cuda" / "cuda:0" 等：显式指定（cuda 不可用时回退到 cpu 并告警）。

    返回
    ----
    torch.device
        选定的设备。
    """
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer.startswith("cuda") and not torch.cuda.is_available():
        log_warn(f"请求设备 {prefer} 但当前环境无可用 CUDA，已回退到 cpu")
        return torch.device("cpu")
    return torch.device(prefer)


# ======================================================================
# 日志
# ======================================================================
def log_info(msg: str) -> None:
    """打印普通信息日志（统一前缀，便于从长输出中检索）。"""
    print(f"[N3D][INFO ] {msg}", flush=True)


def log_warn(msg: str) -> None:
    """打印告警日志。"""
    print(f"[N3D][WARN ] {msg}", flush=True)


def log_error(msg: str) -> None:
    """打印错误日志。"""
    print(f"[N3D][ERROR] {msg}", flush=True)

# ======================================================================
# 分组（segment）softmax
# ======================================================================
def _segment_softmax_pure_torch(
    values: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """纯 PyTorch 实现的 segment softmax（torch_scatter 缺失时的 fallback）。

    算法（两趟 scatter，禁止任何 Python for 循环遍历元素）
    ------------------------------------------------------
    1. 第一趟 `scatter_reduce_(reduce="amax")` 求组内最大值 m_g（数值稳定：防止 exp 溢出）；
    2. `e = exp(values - m_g[g])`；
    3. 第二趟 `scatter_add_` 求组内归一化因子 Z_g = Σ e；
    4. 返回 `e / (Z_g[g] + eps)`（eps 防止空组或全 mask 组产生除零）。

    参数
    ----
    values : torch.Tensor
        形状 [E] 的边级 logits。
    index : torch.Tensor
        形状 [E] 的组索引（int64），取值范围 [0, num_segments)。
    num_segments : int
        组数（= 输出突触数 N*y_out）。

    返回
    ----
    torch.Tensor
        形状 [E] 的权重，同一组内求和为 1。
    """
    if num_segments <= 0:
        raise ValueError(f"num_segments 必须为正整数，当前 {num_segments}")
    if values.dim() != 1:
        raise ValueError(f"segment_softmax 仅支持 1D values，当前 shape={tuple(values.shape)}")
    if index.shape != values.shape:
        raise ValueError(
            f"index 与 values 形状必须一致，当前 index={tuple(index.shape)}, "
            f"values={tuple(values.shape)}"
        )
    if index.dtype != torch.long:
        index = index.long()

    # 第一趟：组内最大值（-inf 初值保证不干扰真实最大值）
    group_max = torch.full(
        (num_segments,),
        float("-inf"),
        dtype=values.dtype,
        device=values.device,
    )
    group_max = group_max.scatter_reduce(0, index, values, reduce="amax", include_self=True)
    # 空组仍为 -inf，替换为 0 避免后续 NaN（空组不会产生任何边贡献）
    group_max = torch.where(
        torch.isfinite(group_max), group_max, torch.zeros_like(group_max)
    )

    # 减去组内最大值：softmax 的数值稳定标准做法
    shifted = values - group_max[index]
    exp_values = torch.exp(shifted)

    # 第二趟：组内求和得到归一化因子
    denom = torch.zeros((num_segments,), dtype=values.dtype, device=values.device)
    denom = denom.scatter_add(0, index, exp_values)
    eps = torch.finfo(values.dtype).eps
    return exp_values / (denom[index] + eps)


def segment_softmax(
    values: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """按 `index` 分组的 softmax：w_i = exp(v_i - m_g) / Σ_{j∈g} exp(v_j - m_g)。

    归一化方向契约
    --------------
    调用方（`model.sparse_propagate`）传入的 `index = edge_index[0]`（输出突触索引），
    因此归一化正是"对每个输出突触 o，在其连接的输入突触 j 上做 softmax"。
    禁止把 index 换成 edge_index[1]（那会变成对输入突触维度归一化），禁止全局 softmax。

    参数
    ----
    values : torch.Tensor
        形状 [E] 的边级 logits（历史实现中为"距离衰减项 + 可学习边权"的组合；
        当前纯球形分层 DAG 架构不再调用本函数，连接聚合见 `model.py::stage2_recurrence`）。
    index : torch.Tensor
        形状 [E] 的组索引（int64），即 edge_index[0]。
    num_segments : int
        组数，即输出突触总数 N*y_out。

    返回
    ----
    torch.Tensor
        形状 [E] 的 softmax 权重（同一输出突触的所有入边之和为 1）。
    """
    if _HAS_TORCH_SCATTER:
        # 优先走 torch_scatter 的高效实现
        out = torch_scatter.scatter_softmax(values, index, dim=0, dim_size=num_segments)
        return out
    return _segment_softmax_pure_torch(values, index, num_segments)

# ======================================================================
# 拓扑构建
# ======================================================================
def build_edge_index(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """从 dense 连接掩码中提取稀疏边索引与边距离。

    参数
    ----
    mask : torch.Tensor
        形状 [N*y_out, N*y_in] 的 0/1 掩码；mask[o, j] = 1 表示输出突触 o 与
        输入突触 j 的距离 <= D，允许传递数据。调用方需同时提供 `dist`（见
        `build_edge_index_from_dist`），两者会一并用于构图。

    返回
    ----
    tuple[torch.Tensor, torch.Tensor]
        (edge_index, flat_idx)：
        * edge_index：形状 [2, E]，第 0 行是输出突触索引（行优先展开后的行号），
          第 1 行是输入突触索引；dtype=int64；
        * flat_idx：形状 [E] 的展平索引 `o * n_in + j`，便于直接从 dist 中 gather
          边距离，避免二次展开。

    关键不变量
    ----------
    * 行优先（row-major / C order）展开，保证 `flat_idx = o * n_in + j`；
    * E == mask.sum()，即边的数量等于掩码中 1 的个数。
    """
    if mask.dim() != 2:
        raise ValueError(f"mask 必须为 2D 张量，当前 shape={tuple(mask.shape)}")
    n_out, n_in = mask.shape
    # nonzero 返回 [E, 2]，第 0 列是行号（输出突触），第 1 列是列号（输入突触）
    flat_idx = torch.nonzero(mask.reshape(-1) > 0, as_tuple=False).reshape(-1)
    rows = torch.div(flat_idx, n_in, rounding_mode="floor")
    cols = flat_idx - rows * n_in
    edge_index = torch.stack([rows, cols], dim=0).to(torch.long)
    if edge_index.shape[1] != int(mask.sum().item()):
        raise AssertionError(
            f"edge 数量与 mask 不一致：E={edge_index.shape[1]}, mask.sum={int(mask.sum().item())}"
        )
    return edge_index, flat_idx.to(torch.long)


def build_edge_index_from_dist(
    dist: torch.Tensor,
    d_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """由距离矩阵一次性构建 mask / edge_index / edge_dist。

    参数
    ----
    dist : torch.Tensor
        形状 [N*y_out, N*y_in] 的输出突触-输入突触距离矩阵。
    d_threshold : float
        连接距离阈值 D。

    返回
    ----
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (mask, edge_index, edge_dist)：
        * mask：形状 [N*y_out, N*y_in] 的 float 掩码（1.0 表示可连接）；
        * edge_index：形状 [2, E]，见 `build_edge_index`；
        * edge_dist：形状 [E] 的边距离。

    异常
    ----
    ValueError
        当 D <= 0 或没有任何边被建立时抛出（拓扑为空则模型无法训练）。
    """
    if not (d_threshold > 0.0):
        raise ValueError(f"连接距离阈值 D 必须 > 0，当前 D={d_threshold}")
    mask = (dist <= d_threshold).to(dist.dtype)
    edge_index, flat_idx = build_edge_index(mask)
    e = edge_index.shape[1]
    if e == 0:
        raise ValueError(
            f"当前 D={d_threshold} 下没有任何边被建立（E=0），请增大 D 或减小 H。"
            f"dist 统计：min={float(dist.min()):.4f}, max={float(dist.max()):.4f}, "
            f"mean={float(dist.mean()):.4f}"
        )
    edge_dist = dist.reshape(-1)[flat_idx]
    # edge_index / edge_dist 统一由 dense 张量派生，保证 dtype/device 与 dist 一致
    return mask.to(dist.dtype), edge_index.to(dist.device), edge_dist.to(dist.dtype)

def build_scatter_matrix(
    neuron_ids: torch.Tensor,
    num_synapses: int,
    num_neurons: int,
    mode: str = "sum",
) -> torch.Tensor:
    """构建"突触 -> 神经元"聚合算子或"神经元 -> 突触"广播算子的**稠密**标记矩阵。

    参数
    ----
    neuron_ids : torch.Tensor
        形状 [num_synapses] 的 int64 张量，第 i 个元素是第 i 个突触所属神经元的池内局部 ID。
        * mode="sum"       ：索引 i 取值为输出突触局部 ID（0..num_synapses-1），值为神经元 ID；
        * mode="broadcast" ：索引 i 取值为神经元局部 ID，值为该神经元拥有的输入突触 ID 列表（展平后）。
    num_synapses : int
        突触总数（sum 模式为 N*y_out，broadcast 模式为 N*y_in）。
    num_neurons : int
        神经元数量 N。
    mode : str
        "sum"：返回 M[num_neurons, num_synapses]，M[n, i] = 1 当且仅当突触 i 属于神经元 n，
               则 neuron_input = s_out @ M.T（等价于 scatter_add 求和）；
        "broadcast"：返回 M[num_synapses, num_neurons]，M[i, n] = 1 当且仅当突触 i 属于神经元 n，
               则 s_in_new = a @ M.T（等价于按所属神经元广播）。

    返回
    ----
    torch.Tensor
        形状为 [num_neurons, num_synapses]（sum）或 [num_synapses, num_neurons]（broadcast）
        的稠密 0/1 标记矩阵，dtype=float32。

    注意
    ----
    该稠密矩阵仅用于小规模或需要显式矩阵语义的场合（其非零元个数为 num_synapses，
    本身并不稠密）。模型前向传播中为避免 O(N*y_out*N*y_in) 的显式稀疏矩阵，
    使用 `sparse_aggregate` / `sparse_broadcast` 两个 y 级（突触分组）高效算子。
    """
    if mode not in ("sum", "broadcast"):
        raise ValueError(f"mode 必须是 'sum' 或 'broadcast'，当前 mode={mode!r}")
    if num_synapses <= 0 or num_neurons <= 0:
        raise ValueError(
            f"num_synapses 与 num_neurons 必须为正整数，当前 {num_synapses}, {num_neurons}"
        )
    if neuron_ids.dim() != 1:
        raise ValueError(f"neuron_ids 必须为 1D 张量，当前 shape={tuple(neuron_ids.shape)}")

    if mode == "sum":
        if neuron_ids.numel() != num_synapses:
            raise ValueError(
                f"sum 模式要求 neuron_ids 长度为 num_synapses={num_synapses}，"
                f"当前长度 {neuron_ids.numel()}"
            )
        if int(neuron_ids.max()) >= num_neurons or int(neuron_ids.min()) < 0:
            raise ValueError(
                f"neuron_ids 取值必须落在 [0, {num_neurons})，"
                f"当前范围 [{int(neuron_ids.min())}, {int(neuron_ids.max())}]"
            )
        mat = torch.zeros((num_neurons, num_synapses), dtype=torch.float32)
        mat[neuron_ids.long(), torch.arange(num_synapses, dtype=torch.long)] = 1.0
        return mat

    # broadcast 模式：neuron_ids 是"每条输入突触所属神经元"
    if neuron_ids.numel() != num_synapses:
        raise ValueError(
            f"broadcast 模式要求 neuron_ids 长度为 num_synapses={num_synapses}，"
            f"当前长度 {neuron_ids.numel()}"
        )
    if int(neuron_ids.max()) >= num_neurons or int(neuron_ids.min()) < 0:
        raise ValueError(
            f"neuron_ids 取值必须落在 [0, {num_neurons})，"
            f"当前范围 [{int(neuron_ids.min())}, {int(neuron_ids.max())}]"
        )
    mat = torch.zeros((num_synapses, num_neurons), dtype=torch.float32)
    mat[torch.arange(num_synapses, dtype=torch.long), neuron_ids.long()] = 1.0
    return mat

def sparse_aggregate(
    s_out: torch.Tensor,
    neuron_of_output_syn: torch.Tensor,
    num_neurons: int,
) -> torch.Tensor:
    """步骤 2b：输出突触 -> 神经元的稀疏求和（scatter-add 语义）。

    参数
    ----
    s_out : torch.Tensor
        形状 [B, N*y_out] 的输出突触信号。
    neuron_of_output_syn : torch.Tensor
        形状 [N*y_out] 的 int64 张量，第 o 个输出突触所属神经元的局部 ID（= o // y_out）。
    num_neurons : int
        神经元数 N。

    返回
    ----
    torch.Tensor
        形状 [B, N] 的神经元输入：neuron_input[b, n] = Σ_{o ∈ n} s_out[b, o]。

    说明
    ----
    使用 `index_add_`（索引在最后一维上的 scatter_add 等价形式），
    **不构造任何 [N*y_out, N*y_in] 规模的稠密矩阵**，复杂度 O(B * N * y_out)。
    """
    if s_out.dim() != 2:
        raise ValueError(f"s_out 必须为 2D [B, N*y_out]，当前 shape={tuple(s_out.shape)}")
    if neuron_of_output_syn.numel() != s_out.shape[1]:
        raise ValueError(
            f"neuron_of_output_syn 长度必须等于 s_out.shape[1]={s_out.shape[1]}，"
            f"当前 {neuron_of_output_syn.numel()}"
        )
    neuron_input = torch.zeros(
        (s_out.shape[0], num_neurons), dtype=s_out.dtype, device=s_out.device
    )
    # index_add_ 沿 dim=1 按神经元 ID 累加：等价于对输出突触做 scatter_add
    neuron_input = neuron_input.index_add(1, neuron_of_output_syn.long(), s_out)
    return neuron_input


def sparse_broadcast(
    a: torch.Tensor,
    neuron_of_input_syn: torch.Tensor,
) -> torch.Tensor:
    """步骤 2d：神经元 -> 输入突触的稀疏广播（gather 语义）。

    参数
    ----
    a : torch.Tensor
        形状 [B, N] 的神经元激活值。
    neuron_of_input_syn : torch.Tensor
        形状 [N*y_in] 的 int64 张量，第 i 个输入突触所属神经元的局部 ID（= i // y_in）。

    返回
    ----
    torch.Tensor
        形状 [B, N*y_in] 的广播结果：s_in_new[b, i] = a[b, neuron_of(i)]。

    说明
    ----
    用 `index_select` 实现，复杂度 O(B * N * y_in)，天然支持 y_in ≠ y_out。
    """
    if a.dim() != 2:
        raise ValueError(f"a 必须为 2D [B, N]，当前 shape={tuple(a.shape)}")
    if neuron_of_input_syn.numel() == 0:
        raise ValueError("neuron_of_input_syn 不能为空张量")
    if int(neuron_of_input_syn.max()) >= a.shape[1] or int(neuron_of_input_syn.min()) < 0:
        raise ValueError(
            f"neuron_of_input_syn 取值必须落在 [0, {a.shape[1]})，"
            f"当前范围 [{int(neuron_of_input_syn.min())}, {int(neuron_of_input_syn.max())}]"
        )
    return a.index_select(1, neuron_of_input_syn.long())


# ======================================================================
# 参数与统计
# ======================================================================
def count_parameters(module: nn.Module) -> int:
    """统计模块中可学习参数（requires_grad=True）的元素总数。

    参数
    ----
    module : nn.Module
        任意 PyTorch 模块。

    返回
    ----
    int
        可学习参数总数。
    """
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def tensor_grad_norms(module: nn.Module) -> dict:
    """收集模块内所有可学习参数的梯度 L2 范数。

    参数
    ----
    module : nn.Module
        任意 PyTorch 模块。

    返回
    ----
    dict
        {参数名: grad_norm(float)}；梯度为 None 的参数返回 -1.0（表示"未参与反向"），
        便于验收时一眼看出哪个参数没拿到梯度。
    """
    out: dict = {}
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            out[name] = -1.0
        else:
            out[name] = float(p.grad.detach().norm().item())
    return out


def connection_density(num_edges: int, num_output_syn: int, num_input_syn: int) -> float:
    """计算连接密度：E / (N*y_out * N*y_in)。

    这是 current_spec.md 规定的"连接稀疏度"口径（规格公式即密度定义）：
    分子是实际存在的边数 E，分母是 dense 权重矩阵的元素总数 N*y_out*N*y_in。
    阶段 A 验收要求该值 < 0.1，即"连接只占可能连接对的不到 10%"。

    参数
    ----
    num_edges : int
        实际边数 E。
    num_output_syn : int
        输出突触总数 N*y_out。
    num_input_syn : int
        输入突触总数 N*y_in。

    返回
    ----
    float
        连接密度，取值 [0, 1]。阶段 A 验收要求 < 0.1。

    异常
    ------
    ValueError
        分母（可能连接对数）非正时抛出。
    """
    total = num_output_syn * num_input_syn
    if total <= 0:
        raise ValueError(f"输出/输入突触总数必须为正，当前 {num_output_syn}, {num_input_syn}")
    if num_edges < 0:
        raise ValueError(f"边数 E 不能为负，当前 {num_edges}")
    return float(num_edges) / float(total)


def zero_ratio(num_edges: int, num_output_syn: int, num_input_syn: int) -> float:
    """计算 dense 矩阵中的零元素占比（= 1 - 连接密度）。

    该值刻画"若把连接权重写成 dense 矩阵会有多稀疏"，与规格的
    `connection_density` 互为补数。仅用于日志对照，**不用于验收判据**。

    参数
    ----
    num_edges : int
        实际边数 E。
    num_output_syn : int
        输出突触总数 N*y_out。
    num_input_syn : int
        输入突触总数 N*y_in。

    返回
    ----
    float
        零元素占比，取值 [0, 1]。
    """
    return 1.0 - connection_density(num_edges, num_output_syn, num_input_syn)


# 兼容别名（**已弃用 / deprecated**）：规格文档使用"连接稀疏度"这一名称指代密度口径
# （E / (N*y_out*N*y_in)）。为便于按规格原文检索与兼容既有调用，保留
# connection_sparsity 作为 connection_density 的别名；新代码请统一使用
# connection_density，避免"稀疏度"一词再次引起口径歧义。
connection_sparsity = connection_density