"""N3D 二期架构变体核心模型：三维神经元空间（球形有向拓扑 + 立方体兼容路径）。

本模块由一期 `n3d_proto/model.py` 拷贝而来，在保持默认（`topology="cube"`）行为
**逐位不变**的前提下，新增 `topology="sphere"` 的球形有向几何与拓扑统计。

张量形状契约（严格遵循 current_spec.md 的"数据结构与张量形状总表"）
-----------------------------------------------------------------
| 名称                     | 形状                  | 类型      |
|--------------------------|-----------------------|-----------|
| neuron_pos               | [N, 3]                | buffer    |
| input_syn_pos            | [N*y_in, 3]           | buffer    |
| output_syn_pos           | [N*y_out, 3]          | buffer    |
| dist                     | [N*y_out, N*y_in]     | buffer    |
| mask                     | [N*y_out, N*y_in]     | buffer    |
| edge_index               | [2, E]                | buffer    |
| edge_dist                | [E]                   | buffer    |
| W_conn_sparse            | [E]                   | Parameter |
| tau_raw                  | 标量                  | Parameter |
| neuron_threshold         | [N]                   | Parameter |
| W_in                     | [input_dim, N*y_in]   | Parameter |
| W_out                    | [N*y_out, output_dim] | Parameter |
| W_out_bias（可选）        | [output_dim]          | Parameter（仅 readout_bias=True 时创建） |
| scatter_out_to_neuron    | [N, N*y_out]          | buffer    |
| broadcast_neuron_to_in   | [N*y_in, N]           | buffer    |

四步闭环（每一轮 T 的实现位置见 `forward`）
------------------------------------------
* 2a 稀疏空间连接（`sparse_propagate`）：对每个输出突触 o，在其连接的输入突触集合上
  做 masked softmax，得到 s_out[b, o] = Σ_j w_oj · s_in[b, j]；
* 2b 输出突触 -> 神经元（scatter sum）：`sparse_aggregate`，neuron_input = Σ_{o∈n} s_out；
* 2c 神经元激活：a = ReLU(neuron_input + neuron_threshold)；
* 2d 神经元 -> 输入突触（broadcast）+ 残差 + LayerNorm：
  s_in_next = LN(sparse_broadcast(a) + alpha · s_in)，残差系数 alpha 默认 0.1。

几何采样分支（二期新增，默认分支逐字保留）
------------------------------------------
* `cube`：神经元在 [0, L]^3 均匀；突触在所属神经元周围 **H 半径球体**内按体积均匀；
* `sphere`：神经元在**与立方体等体积**的球（半径 ≈ L·(3/4π)^(1/3) ≈ 0.620）
  内按体积均匀（保留 `min_neuron_dist` 拒绝采样）；输入突触取**负半球**（−axis）、
  输出突触取**正半球**（+axis），半径仍在 H 球内按体积均匀。

半径采样口径（两种拓扑一致，**勿误读为半径线性采样**）
----------------------------------------------------
方向在单位球面上均匀（正态归一化得到）；半径 `r = H · u^(1/3)`，其中
`u ~ U(0, 1)`、`u^(1/3)` 是 **u 的立方根**（代码写作 `H * u.pow(1.0 / 3.0)`）。
立方根保证采样点在**球体内按体积均匀**（半径的分布函数为 F(r) = (r/H)^3）；
sphere 分支的神经元球半径同理取 `R · u^(1/3)`。`cube` 分支与一期**逐字一致**，
不得改动（改共享采样逻辑会平移随机流、破坏逐位不变）。
该表述与 `_sample_synapse_positions` / `_sample_neuron_positions` 的 docstring、
`README.md` 第 2.1 节、`config.py` 完全一致。

关键几何性质（实测于 `get_topology_stats()`）
--------------------------------------------
**本文件只使用代码口径描述 gap，不做 Δz / δ 分解**——早期文档中的
`gap = Δz + δ_out − δ_in`、`δ_out − δ_in ∈ (0, 2H]` 属与实际实现不符的冗余推导
（代码并不单独计算 Δz 或 δ），已删除。统一口径为：

    gap = axis(output_syn_pos) − axis(input_syn_pos)

其中 `axis` 是由 `flow_axis` 选定的分量（`flow_axis_index`，取值 x/y/z）。
即 gap 就是"输出突触坐标 − 输入突触坐标"在流向轴上的分量，与 `__init__` 中
预计算的 `axis_gap` 及 `get_topology_stats()` 返回的 `axis_gap_*` 完全一致。

`gap > 0` 为沿 −axis → +axis 上行的**正向边**，`gap < 0` 为逆流向的**逆向边**。
因输入突触取 −axis 半球、输出突触取 +axis 半球，gap 的分布相对立方体**明显右偏**：
各向同性（cube）时正/逆向≈50%（实测 0.5009 / 0.4991），sphere 实测正向≈61.5%
（逆向占比 0.385006）—— 即所得为**明显上行偏向但非严格 DAG** 的有向图。
因此 `get_topology_stats()` 如实返回「逆向边占比」，不做任何理想化裁剪。
（实测条件：N=256 / y=8×8 / H=0.1 / D=0.15 / 等体积球 R≈0.620 / seed=42。）

稀疏实现约束
-----------
禁止 materialize dense [N*y_out, N*y_in] 权重矩阵（N=256, y=8 时达 4M 元素且 99% 为零）。
连接权重以边级参数 `W_conn_sparse` [E] 表示，2b/2d 分别用 `index_add_` / `index_select`
实现，复杂度均为 O(B · N · y)。
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # 兼容"以脚本方式直接运行 n3d_sphere/model.py"与"作为包导入"两种情形
    from .config import Config, DEFAULT_CONFIG, FLOW_AXIS_CHOICES
    from .utils import (
        build_edge_index_from_dist,
        connection_density,
        log_warn,
        segment_softmax,
        sparse_aggregate,
        sparse_broadcast,
    )
except ImportError:  # pragma: no cover
    from config import Config, DEFAULT_CONFIG, FLOW_AXIS_CHOICES  # type: ignore
    from utils import (  # type: ignore
        build_edge_index_from_dist,
        connection_density,
        log_warn,
        segment_softmax,
        sparse_aggregate,
        sparse_broadcast,
    )

__all__ = ["ThreeDNeuronSpace", "MLPBaseline", "NEURON_THRESHOLD_INIT"]

# 神经元阈值初始值（正偏置）：保证 2c 的 ReLU 在训练初期不会被完全抑制，
# 避免 neuron_threshold / ln_s_in 参数梯度为 0（初期激活通过率约 40%~60%）。
NEURON_THRESHOLD_INIT: float = 0.1


class ThreeDNeuronSpace(nn.Module):
    """三维神经元空间网络（二期：球形有向拓扑；默认仍为一期立方体几何）。

    参数
    ----
    config : Config
        超参配置对象（见 `n3d_sphere/config.py`）。几何由 `config.topology` 分支：
        `"cube"`（默认，逐位兼容一期）或 `"sphere"`（球形有向拓扑）。

    关键不变量
    ----------
    * `dist` / `mask` / `edge_index` / `edge_dist` / `scatter_out_to_neuron` /
      `broadcast_neuron_to_in` 全部在 `__init__` 中预计算并注册为 buffer，
      `forward` 中不得重算；
    * `tau = softplus(tau_raw) + 0.01`，构造后立即断言 `tau > 0`；
    * 边数 E > 0（拓扑为空时直接抛异常，避免训练出无意义结果）；
    * **逐位不变契约**：`topology="cube"` 时坐标采样与随机流与一期完全一致
      （`--smoke-test` 的 `loss` 必须同为 2.419689）。
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.N: int = config.N
        self.y_in: int = config.y_in
        self.y_out: int = config.y_out
        self.T: int = config.T
        self.alpha: float = config.alpha
        self.n_in_syn: int = config.n_input_syn   # N * y_in
        self.n_out_syn: int = config.n_output_syn  # N * y_out
        # ---- 二期拓扑字段：只在 sphere 分支被读取，cube 分支完全不读取（不影响随机流）----
        self.topology: str = str(config.topology)
        self.flow_axis: str = str(config.flow_axis)
        self.flow_axis_index: int = FLOW_AXIS_CHOICES.index(self.flow_axis)
        self.space_radius: float = float(config.effective_space_radius)
        if self.topology not in ("cube", "sphere"):
            raise ValueError(
                f"不支持的 topology={self.topology!r}，仅允许 'cube' 或 'sphere'"
            )
        # 用 config.seed 派生的局部生成器采样固定坐标，保证模块级可复现且不扰动全局随机状态
        gen = torch.Generator(device="cpu").manual_seed(int(config.seed))

        # ---------------- 固定三维坐标（buffer，不参与训练） ----------------
        neuron_pos = self._sample_neuron_positions(gen)
        # 输入/输出突触坐标：在所属神经元 H 半径的球体内均匀采样
        # （sphere 拓扑下额外限制在流向轴的正/负半球）
        input_syn_pos = self._sample_synapse_positions(
            neuron_pos, self.y_in, gen, hemisphere=-1.0
        )
        output_syn_pos = self._sample_synapse_positions(
            neuron_pos, self.y_out, gen, hemisphere=+1.0
        )
        self.register_buffer("neuron_pos", neuron_pos, persistent=True)
        self.register_buffer("input_syn_pos", input_syn_pos, persistent=True)
        self.register_buffer("output_syn_pos", output_syn_pos, persistent=True)

        # ---------------- 预计算拓扑（全部在 __init__ 完成） ----------------
        # dist[o, j] = ||output_syn_pos[o] - input_syn_pos[j]||，形状 [N*y_out, N*y_in]
        dist = torch.cdist(output_syn_pos, input_syn_pos, p=2)
        mask, edge_index, edge_dist = build_edge_index_from_dist(dist, config.D)
        self.register_buffer("dist", dist, persistent=True)
        self.register_buffer("mask", mask, persistent=True)
        self.register_buffer("edge_index", edge_index, persistent=True)
        self.register_buffer("edge_dist", edge_dist, persistent=True)
        self.num_edges: int = int(edge_index.shape[1])

        # 预计算的两个映射算子（稠密 0/1 标记矩阵，非零元个数 = 突触数，本身并不稠密）
        # scatter_out_to_neuron[n, o] = 1 当且仅当输出突触 o 属于神经元 n
        neuron_of_output_syn = torch.arange(
            self.n_out_syn, dtype=torch.long
        ) // self.y_out
        # broadcast_neuron_to_in[i, n] = 1 当且仅当输入突触 i 属于神经元 n
        neuron_of_input_syn = torch.arange(self.n_in_syn, dtype=torch.long) // self.y_in
        scatter_out_to_neuron = torch.zeros(
            (self.N, self.n_out_syn), dtype=dist.dtype
        )
        scatter_out_to_neuron[
            neuron_of_output_syn, torch.arange(self.n_out_syn, dtype=torch.long)
        ] = 1.0
        broadcast_neuron_to_in = torch.zeros(
            (self.n_in_syn, self.N), dtype=dist.dtype
        )
        broadcast_neuron_to_in[
            torch.arange(self.n_in_syn, dtype=torch.long), neuron_of_input_syn
        ] = 1.0
        self.register_buffer("scatter_out_to_neuron", scatter_out_to_neuron, persistent=True)
        self.register_buffer("broadcast_neuron_to_in", broadcast_neuron_to_in, persistent=True)
        # 下面两个整型映射是 2b/2d 的高效稀疏算子索引，随模型一起搬到目标设备
        self.register_buffer("neuron_of_output_syn", neuron_of_output_syn, persistent=True)
        self.register_buffer("neuron_of_input_syn", neuron_of_input_syn, persistent=True)

        # ---------------- 二期新增：静态拓扑统计（全部预计算为 buffer） ----------------
        # ① 逆向边：沿流向轴，"输出突触在输入突触**下方**"的边。
        #    流向定义为 -axis → +axis（输入突触取负半球、输出突触取正半球），
        #    因此 gap = 输出突触坐标 - 输入突触坐标 时：
        #      gap > 0 → 沿 +axis 向上传播（正向边）；gap < 0 → 逆流向向下传播（逆向边）。
        pos_rows = output_syn_pos.index_select(0, edge_index[0])
        pos_cols = input_syn_pos.index_select(0, edge_index[1])
        axis_gap = self._compute_edge_axis_gap(pos_rows, pos_cols)
        # 注意：一期/二期均**不裁剪**逆向边，这里只统计占比（保持与一期一致的构图规则）
        self.register_buffer(
            "edge_reverse_flag", (axis_gap < 0.0).to(torch.long), persistent=True
        )
        # ①' gap 张量缓存：供 get_topology_stats() 直接复用，避免每次调用都重算 O(E)。
        #     **非持久化 buffer**（persistent=False）：不进 state_dict，因此既不改变
        #     一/二期 state_dict 键集（逐位等价校验仍成立），又能随 .to(device) 搬设备。
        #     与 W1 容器同类（内存态），故 get_topology_stats() 对该属性取兜底并在缺失时
        #     用同一个 `_compute_edge_axis_gap` 重算，避免"两份实现分叉"。
        self.register_buffer("edge_axis_gap", axis_gap, persistent=False)
        # ② 弱连通分量（把每条边映射到"输出突触所属神经元 ↔ 输入突触所属神经元"）
        self.register_buffer(
            "neuron_component_id",
            self._compute_weak_components(edge_index, neuron_of_output_syn, neuron_of_input_syn),
            persistent=True,
        )
        # ③ 静态连通性：至少有 1 条入边的输出突触占比（不依赖输入数值）
        has_in_edge = torch.zeros(self.n_out_syn, dtype=torch.bool)
        has_in_edge[edge_index[0]] = True
        self.register_buffer(
            "connected_output_mask", has_in_edge.to(torch.long), persistent=True
        )
        # ④ 每轮 2a 输出非零覆盖比例（动态观测量，由 forward 逐轮累加，不参与任何判定）
        #    修复缺陷 W1：原实现用 `List[float]` 无界 append（12ep×469batch×T=4 ≈ 22512
        #    元素），且不随 checkpoint 持久化。现改为 **O(1) 定长累计容器**：
        #      _round_coverage_sum   : 各轮覆盖比例之和（用于算全程均值）
        #      _round_coverage_count : 已累计轮数
        #      _round_coverage_last  : 末轮覆盖比例
        #    三者均为 Python 标量（不占设备显存、不受 state_dict 影响）。
        #    持久化与跨阶段隔离由 `reset_round_output_coverage()`（训练开始时调用）与
        #    train.py 写入 checkpoint 的显式元数据字段共同保证。
        self.reset_round_output_coverage()

        # ---------------- 可学习参数 ----------------
        # 边级参数化：每条边一个权重（禁止 dense [N*y_out, N*y_in] 矩阵）
        self.W_conn_sparse = nn.Parameter(torch.zeros(self.num_edges))
        # tau = softplus(tau_raw) + 0.01 > 0 恒成立
        self.tau_raw = nn.Parameter(torch.tensor(float(config.tau_init)))
        # 神经元阈值（2c 中的正偏置）：初始化为正值常数，保证 ReLU 初期有激活、
        # 从而 neuron_threshold / ln_s_in 等参数都能拿到非零梯度（否则死 ReLU 会使
        # 这些参数的 .grad 为 None，违反"所有可学习参数梯度范数 > 0"的验收条款）
        self.neuron_threshold = nn.Parameter(
            torch.full((self.N,), NEURON_THRESHOLD_INIT)
        )
        self.W_in = nn.Parameter(torch.empty(config.input_dim, self.n_in_syn))
        self.W_out = nn.Parameter(torch.empty(self.n_out_syn, config.output_dim))
        # 可选输出层偏置：readout_bias=False 时**不创建**该参数（保持参数集与数值逐位不变）
        self.readout_bias_enabled: bool = bool(config.readout_bias)
        if self.readout_bias_enabled:
            self.W_out_bias = nn.Parameter(torch.zeros(config.output_dim))
        self._init_parameters()

        # 输入突触信号的层归一化：抑制纯线性迭代导致的信号坍缩/爆炸
        self.ln_s_in = nn.LayerNorm(self.n_in_syn)

        # 可选 dropout（作用在 s_out 送入 W_out 之前）。
        # 关键契约：dropout=0.0 时使用 nn.Identity —— 既不改变数值，也**不消耗任何随机数**，
        # 从而保证 SMALL_CONFIG / DEFAULT_CONFIG 的默认路径逐位不变、可复现性不受影响。
        self.dropout_p: float = float(config.dropout)
        self.readout_dropout: nn.Module = (
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0.0 else nn.Identity()
        )

        # 数值契约：构造后立即验证 tau > 0
        tau = self.current_tau()
        assert tau > 0.0, f"[契约失败] tau 必须 > 0，当前 tau={tau}"

    # ==================================================================
    # 边间距（gap）计算：__init__ 与 get_topology_stats() 兜底路径**共用同一实现**
    # ==================================================================
    def _compute_edge_axis_gap(
        self, pos_rows: torch.Tensor, pos_cols: torch.Tensor
    ) -> torch.Tensor:
        """计算沿流向轴的边间距 `gap = axis(输出突触坐标) - axis(输入突触坐标)`。

        口径（全文统一，不做 Δz / δ 分解）：`gap > 0` 为沿 -axis → +axis 上行的
        **正向边**，`gap < 0` 为逆流向的**逆向边**（见 `get_topology_stats`）。

        **本方法是唯一实现**：`__init__` 预计算 `edge_axis_gap` 缓存时调用它，
        `get_topology_stats()` 在缓存缺失（例如非标准加载路径绕过 `__init__`
        重建 buffer）时也调用它重算并**回写缓存**（第 5 轮：使缓存语义自愈，
        下次调用直接命中），从而**不存在两份可能分叉的实现**。

        参数
        ----
        pos_rows : torch.Tensor
            形状 [E, 3] 的边起点（输出突触）坐标，即 `output_syn_pos[edge_index[0]]`。
        pos_cols : torch.Tensor
            形状 [E, 3] 的边终点（输入突触）坐标，即 `input_syn_pos[edge_index[1]]`。

        返回
        ----
        torch.Tensor
            形状 [E] 的流向轴间距（与输入同 dtype）。
        """
        return pos_rows[:, self.flow_axis_index] - pos_cols[:, self.flow_axis_index]

    # ==================================================================
    # 坐标采样（仅在 __init__ 中调用）
    # ==================================================================
    def _sample_neuron_positions(self, gen: torch.Generator) -> torch.Tensor:
        """采样 N 个神经元坐标（带最小间距的稀疏分布），按 `topology` 分支。

        算法：逐神经元向量化拒绝采样——每轮为"尚未安置"的神经元一次性采样一批候选，
        检查其与固定神经元集合的最小距离是否满足 `min_neuron_dist`，满足者安置并固定，
        不满足者进入下一轮。整个过程是"轮 × 批量"的向量化操作，没有按神经元逐个循环。

        分布分支（**cube 分支逐字保留一期实现**）
        ----------------------------------------
        * `cube`（一期）：候选坐标在 [0, L]^3 均匀；
        * `sphere`（二期）：候选坐标在以原点为球心、半径 space_radius（默认取与立方体
          **等体积**的球半径 ≈ 0.620）的球体内按体积均匀 —— 方向球面均匀 +
          `r = R · u^(1/3)`（`u ~ U(0,1)`，`u^(1/3)` 为 **u 的立方根**，保证球内按体积
          均匀，而非半径线性采样），使神经元密度与一期保持一致
          （N / L^3 = N / ((4/3)πR^3)）。

        参数
        ----
        gen : torch.Generator
            由 config.seed 派生的随机生成器。

        返回
        ----
        torch.Tensor
            形状 [N, 3] 的神经元坐标。

        异常
        ------
        ValueError
            若达到 `max_sample_tries` 轮后仍有神经元无法安置（空间过密），
            给出可读错误并提示调小 `min_neuron_dist` 或减小 N。
        """
        L = float(self.config.L)
        min_dist = float(self.config.min_neuron_dist)
        if self.topology == "sphere":
            # ---- 二期新增：等体积球内按体积均匀采样（保持神经元密度不变） ----
            R = float(self.space_radius)
            dir_seed = torch.randn((self.N, 3), generator=gen)
            dir_seed = dir_seed / dir_seed.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            rad_seed = R * torch.rand((self.N, 1), generator=gen).pow(1.0 / 3.0)
            # 半径 = R · u^(1/3)（u ~ U(0,1)，立方根保证球内按体积均匀，非半径线性采样）
            positions = dir_seed * rad_seed
        else:
            # 初始坐标：立方体内均匀随机（**一期实现，逐字保留**）
            positions = torch.rand((self.N, 3), generator=gen) * L
        placed = torch.zeros(self.N, dtype=torch.bool)
        placed[0] = True  # 第一个神经元直接安置，作为拒绝采样的种子
        remaining = torch.nonzero(~placed, as_tuple=False).reshape(-1)

        if min_dist > 0.0:
            for _ in range(int(self.config.max_sample_tries)):
                if remaining.numel() == 0:
                    break
                # 为所有未安置神经元一次性生成候选坐标 [R, 3]
                if self.topology == "sphere":
                    cand_dir = torch.randn((remaining.numel(), 3), generator=gen)
                    cand_dir = cand_dir / cand_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    cand_rad = R * torch.rand(
                        (remaining.numel(), 1), generator=gen
                    ).pow(1.0 / 3.0)
                    cand = cand_dir * cand_rad
                else:
                    cand = torch.rand((remaining.numel(), 3), generator=gen) * L
                placed_pos = positions[placed]  # [P, 3]
                # 候选与已安置点的两两距离平方 [R, P]
                d2 = torch.cdist(cand, placed_pos, p=2)
                ok = (d2.min(dim=1).values >= min_dist)
                if ok.any():
                    idx = remaining[ok]
                    positions[idx] = cand[ok]
                    placed[idx] = True
                    remaining = torch.nonzero(~placed, as_tuple=False).reshape(-1)
            if remaining.numel() > 0:
                raise ValueError(
                    f"神经元坐标采样失败：{remaining.numel()}/{self.N} 个神经元在 "
                    f"{self.config.max_sample_tries} 轮内无法满足最小间距 "
                    f"min_neuron_dist={min_dist}（topology={self.topology}, L={L}, "
                    f"space_radius={self.space_radius}）。"
                    f"请调小 min_neuron_dist 或减小 N。"
                )
        return positions

    def _sample_synapse_positions(
        self,
        neuron_pos: torch.Tensor,
        y: int,
        gen: torch.Generator,
        hemisphere: float = 0.0,
    ) -> torch.Tensor:
        """在所属神经元 H 半径的球体内均匀采样突触坐标（按 `topology` 分支）。

        参数
        ----
        neuron_pos : torch.Tensor
            形状 [N, 3] 的神经元坐标。
        y : int
            每个神经元的突触数量（输入 y_in 或输出 y_out）。
        gen : torch.Generator
            随机生成器。
        hemisphere : float
            **仅对 `topology="sphere"` 生效**：`-1.0` 表示限制在流向轴的负半球
            （输入突触），`+1.0` 表示正半球（输出突触），`0.0` 表示不限制。
            `cube` 路径完全忽略该参数（逐字保留一期实现）。

        返回
        ----
        torch.Tensor
            形状 [N*y, 3] 的突触坐标；第 n*y + k 行属于神经元 n。

        采样实现（两种拓扑均按体积均匀）
        --------------------------------
        * `cube`（一期，**逐字保留**）：方向在单位球面上均匀（正态归一化），
          半径取 `r = H * u^(1/3)`，其中 `u ~ U(0, 1)`、`u^(1/3)` 是 **u 的立方根**
          （代码 `H * u.pow(1.0 / 3.0)`），保证采样点在 H 半径球体内**按体积均匀**；
          注意这**不是**半径线性采样（线性采样会让点向球心聚集）；
        * `sphere`（二期）：同一分布，但在生成方向后把流向轴分量翻转为目标半边
          （`d * sign`，sign = ±1）。该翻转是测度保持的：目标半球内任一方向恰好由
          "原始方向"与"翻转后的方向"两种来源各命中一次，故仍是**半球内**的均匀分布
          （等价于在半球上做拒绝采样，但不需要额外消耗随机数或引入重试上限）。
          翻转为确定性操作，且只发生在 sphere 分支，**完全不影响 cube 随机流**。
        """
        H = float(self.config.H)
        n = neuron_pos.shape[0]
        direction = torch.randn((n, y, 3), generator=gen)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.topology == "sphere" and hemisphere != 0.0:
            # ---- 二期新增：半球切分（输入取 −axis，输出取 +axis） ----
            sign = -1.0 if hemisphere < 0.0 else 1.0
            axis_comp = direction[..., self.flow_axis_index]
            # 负半球要求流向轴分量为负、正半球要求为正；不符合者翻转（测度保持）
            flipped = torch.where(
                (sign * axis_comp) < 0.0, -axis_comp, axis_comp
            )
            direction = torch.cat(
                [
                    direction[..., : self.flow_axis_index],
                    flipped.unsqueeze(-1),
                    direction[..., self.flow_axis_index + 1 :],
                ],
                dim=-1,
            )
        u = torch.rand((n, y, 1), generator=gen)
        # 半径 = H · u^(1/3)（u ~ U(0,1)，u^(1/3) 为 u 的立方根）：
        # 该变换使 r 的分布函数为 F(r) = (r/H)^3，即 H 球内**按体积均匀**；
        # 若误写为 H·u（半径线性采样），点会向球心聚集。此处与一期**逐字一致**。
        radius = H * u.pow(1.0 / 3.0)
        pos = neuron_pos.unsqueeze(1) + direction * radius
        return pos.reshape(n * y, 3).contiguous()

    # ==================================================================
    # 拓扑连通性（仅在 __init__ 中调用，纯张量运算，禁止 Python 循环遍历神经元）
    # ==================================================================
    @torch.no_grad()
    def _compute_weak_components(
        self,
        edge_index: torch.Tensor,
        neuron_of_output_syn: torch.Tensor,
        neuron_of_input_syn: torch.Tensor,
    ) -> torch.Tensor:
        """计算神经元层弱连通分量的分量 id（向量化 union-find / 标签传播 + 指针跳跃）。

        图的定义：把每条边 (输出突触 o, 输入突触 j) 映射为神经元层的无向边
        (neuron_of_output_syn[o], neuron_of_input_syn[j])，即"突触级有向边在神经元层的
        弱连通投影"。这样得到的连通性正是四步闭环信息流实际可达的神经元集合划分。

        算法（全部为张量操作，无 Python 循环遍历神经元）
        ------------------------------------------------
        1. 把每条突触级边投影为**两条**神经元级有向边 (a→b) 与 (b→a)，构成无向图；
        2. **最小标签传播**：`parent[dst] = min(parent[dst], parent[src])`
           （`scatter_reduce_(reduce="amin")`，只可能单调下降，因此有唯一固定点）；
        3. **指针跳跃**：`nxt = min(nxt, nxt[nxt])` 加速收敛；
        4. 反复执行 2~3 直到 `nxt == parent`（最大深度每轮至少减半，N=256 时 ≤ 64 轮）；
        5. 根节点是满足 `parent[i] == i` 的节点，分量 id 用根节点下标表示。

        历史缺陷（离朱第 2 轮实测，已修复）
        ----------------------------------
        初版实现先用 `scatter_reduce_(hi←lo, amin)` 与 `scatter_reduce_(lo←hi, amax)`
        制造"互为指针"的二点循环，再用对称化的指针跳跃，会把这种二点循环维持成
        **非真值的固定点**，导致分量被提前劈开（例如 N=32 仅一条边 (0,1) 时报出 32 个
        分量，真值为 31）。现改为"只做 amin 的最小标签传播"，单调性保证收敛到真值。
        该缺陷**只影响统计量**：`neuron_component_id` 不参与 forward/loss/梯度。

        参数
        ----
        edge_index : torch.Tensor
            形状 [2, E] 的边索引（第 0 行输出突触，第 1 行输入突触）。
        neuron_of_output_syn : torch.Tensor
            形状 [N*y_out] 的"输出突触 -> 神经元"映射。
        neuron_of_input_syn : torch.Tensor
            形状 [N*y_in] 的"输入突触 -> 神经元"映射。

        返回
        ----
        torch.Tensor
            形状 [N] 的分量 id（值为该分量根节点的神经元下标）。
        """
        n = int(self.N)
        if edge_index.shape[1] == 0:
            return torch.arange(n, dtype=torch.long)
        a = neuron_of_output_syn.index_select(0, edge_index[0])
        b = neuron_of_input_syn.index_select(0, edge_index[1])
        # 无向（弱连通）投影：每条边生成 a->b 与 b->a 两个方向
        src = torch.cat([a, b])
        dst = torch.cat([b, a])
        # 委托给生产算法的唯一实现（合成图回归自检复用**同一个**原语，见 `_label_propagate`）
        return self._label_propagate(torch.arange(n, dtype=torch.long), src, dst)

    @staticmethod
    def _label_propagate(
        parent: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        max_iters: int = 64,
    ) -> torch.Tensor:
        """连通分量的**生产算法原语**：最小标签传播 + 指针跳跃。

        这是 `_compute_weak_components`（真实拓扑）与 `_components_of_synthetic_graph`
        （合成图回归自检）**共用的同一份实现**——这样合成图回归检查覆盖的就是生产代码路径，
        而不是它的副本（离朱第 2 轮观察项②：此前索引 1 走的是副本实现，
        "只破坏生产路径"的变异在索引 1 上不会被发现；改为共用原语后两个索引都直接守护生产路径）。

        算法
        ----
        1. `parent[dst] = min(parent[dst], parent[src])`（`scatter_reduce(reduce="amin")`，
           只可能单调下降，因此固定点唯一，且必为"同一分量的最小下标"）；
        2. 指针跳跃 `nxt = min(nxt, nxt[nxt])` 加速收敛（指针深度每轮至少减半）；
        3. 直到 `nxt == parent` 收敛；上限 `max_iters` 防死循环（N=256 时实际 ≤ 8 轮）。

        历史缺陷（离朱第 2 轮实测，已修复）
        ----------------------------------
        初版生产实现用 `scatter_reduce_(hi←lo, amin)` + `scatter_reduce_(lo←hi, amax)`
        制造"互为指针"的二点循环，再用对称化指针跳跃，会把该循环维持成**非真值的固定点**，
        导致分量被提前劈开（N=32 仅一条边 (0,1) 时报出 32 个分量，真值 31）。
        本原语只做 amin 传播，单调性保证收敛到真值。

        参数
        ----
        parent : torch.Tensor
            形状 [n] 的 int64 初值（通常为 `arange(n)`）。
        src, dst : torch.Tensor
            形状 [2E] 的**无向**边端点（每条边两个方向都要给出）。
        max_iters : int
            最大迭代轮数上限。

        返回
        ----
        torch.Tensor
            形状 [n] 的分量 id（值为该分量根节点下标）。
        """
        for _ in range(int(max_iters)):
            # 最小标签传播：parent[dst] = min(parent[dst], parent[src])，单调下降
            nxt = parent.scatter_reduce(
                0, dst, parent.index_select(0, src), reduce="amin", include_self=True
            )
            # 指针跳跃：parent[i] 直接看向 parent[parent[i]]，加速收敛
            nxt = torch.minimum(nxt, nxt.index_select(0, nxt))
            if bool(torch.equal(nxt, parent)):
                return nxt
            parent = nxt
        return parent

    def _component_stats_crosscheck(self, comp: torch.Tensor) -> Dict[str, float]:
        """用**独立实现**的并查集交叉验证 `neuron_component_id`（回归自检，索引 0）。

        该实现刻意不使用 scatter/张量语法，而是用"路径压缩 + 按秩合并"的经典并查集
        逐条边处理（与 `_compute_weak_components` 的算法路径完全不同），用于捕获
        "向量化实现收敛到非真值固定点"这类难以从输出看出的错误。

        参数
        ----
        comp : torch.Tensor
            形状 [N] 的向量化实现输出（分量 id = 根节点下标）。

        返回
        ----
        Dict[str, float]
            {"vectorized_components", "unionfind_components", "match"}；
            `match` 为 1.0 表示两种实现的分量划分完全一致。
        """
        n = int(self.N)
        parent = list(range(n))

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:  # 路径压缩
                parent[x], x = root, parent[x]
            return root

        a = self.neuron_of_output_syn.index_select(0, self.edge_index[0]).tolist()
        b = self.neuron_of_input_syn.index_select(0, self.edge_index[1]).tolist()
        for u, v in zip(a, b):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[rv] = ru
        roots = [find(i) for i in range(n)]
        n_uf = len(set(roots))
        # 比较"划分"而非"标签本身"：同分量必须同标签、不同分量必须不同标签
        vec = comp.tolist()
        match = True
        for i in range(n):
            for j in range(i + 1, n):
                if (vec[i] == vec[j]) != (roots[i] == roots[j]):
                    match = False
                    break
            if not match:
                break
        n_vec = len(set(vec))
        return {
            "vectorized_components": float(n_vec),
            "unionfind_components": float(n_uf),
            "match": 1.0 if match else 0.0,
        }

    @staticmethod
    def _components_of_synthetic_graph(
        num_nodes: int, edges: List[tuple]
    ) -> torch.Tensor:
        """在**合成图**上运行生产算法原语 `_label_propagate`（回归自检，索引 1）。

        用于回归自检：合成图的真值可事先写出，若算法在合成图上出错（历史缺陷正是如此），
        本方法会与真值不符而暴露问题。**共用 `_label_propagate`** 意味着这里守护的是
        生产代码路径本身，而非其副本（离朱第 2 轮观察项②的加固）。

        参数
        ----
        num_nodes : int
            合成图节点数。
        edges : List[tuple]
            无向边列表，元素为 (u, v)。

        返回
        ----
        torch.Tensor
            形状 [num_nodes] 的分量 id（根节点下标）。
        """
        parent = torch.arange(int(num_nodes), dtype=torch.long)
        if not edges:
            return parent
        e = torch.tensor(edges, dtype=torch.long)
        src = torch.cat([e[:, 0], e[:, 1]])
        dst = torch.cat([e[:, 1], e[:, 0]])
        return ThreeDNeuronSpace._label_propagate(parent, src, dst)

    @staticmethod
    def _expected_components(num_nodes: int, edges: List[tuple]) -> int:
        """用朴素并查集计算合成图的真值分量数（对照用，与向量化实现路径不同）。"""
        parent = list(range(int(num_nodes)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u, v in edges:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[rv] = ru
        return len({find(i) for i in range(int(num_nodes))})

    def connectivity_selfcheck(self) -> Dict[str, float]:
        """连通性统计的**回归自检**（供验证脚本与冒烟诊断调用，不参与训练）。

        三重校验：
        * 索引 0：向量化 `neuron_component_id` 与独立并查集的分量划分是否一致；
        * 索引 1：向量化算法在 7 个**真值已知的合成图**上是否正确
          （单条边 / 链 / 两条链 / 星形 / 二部图+孤立点 / 三角形 / 完全图）；
        * 索引 2：由 `neuron_component_id` 导出的分量数与最大分量占比（用于报告）。

        返回
        ----
        Dict[str, float]
            * "synthetic_cases_passed" / "synthetic_cases_total"：合成图真值比对结果；
            * "vectorized_vs_unionfind_match"：真实拓扑上两种实现是否一致；
            * "weak_components" / "largest_component_ratio"：本次拓扑的连通性统计。
        """
        # ---- 索引 1：合成图真值比对 ----
        cases: List[tuple] = [
            (32, [(0, 1)]),                                              # 单条边
            (32, [(i, i + 1) for i in range(31)]),                       # 链
            (32, [(i, i + 1) for i in range(15)] + [(i, i + 1) for i in range(16, 31)]),  # 两条链
            (32, [(0, i) for i in range(1, 32)]),                        # 星形
            (32, [(i, 16 + j) for i in range(4) for j in range(4)]),     # K4,4 + 孤立点
            (32, [(0, 1), (1, 2), (0, 2)]),                              # 三角形
            (32, [(i, j) for i in range(32) for j in range(i + 1, 32)]), # 完全图 K32
        ]
        passed = 0
        for num_nodes, edges in cases:
            comp = self._components_of_synthetic_graph(num_nodes, edges)
            got = len(set(comp.tolist()))
            if got == self._expected_components(num_nodes, edges):
                passed += 1
        # ---- 索引 0：真实拓扑上两种实现的一致性 ----
        cross = self._component_stats_crosscheck(self.neuron_component_id)
        comp_sizes = torch.bincount(
            self.neuron_component_id, minlength=int(self.N)
        )
        return {
            "synthetic_cases_passed": float(passed),
            "synthetic_cases_total": float(len(cases)),
            "vectorized_vs_unionfind_match": float(cross["match"]),
            "vectorized_components": float(cross["vectorized_components"]),
            "unionfind_components": float(cross["unionfind_components"]),
            "weak_components": float(torch.unique(self.neuron_component_id).numel()),
            "largest_component_ratio": float(comp_sizes.max().item()) / float(self.N),
        }

    # ==================================================================
    # 参数初始化与数值量
    # ==================================================================
    def _init_parameters(self) -> None:
        """初始化全部可学习参数（由 config.seed 保证可复现）。

        * `W_in`：LeCun 风格正态初始化，std = 1/sqrt(fan_in)，保证叠加方差可控；
        * `W_out`：Xavier 均匀初始化；
        * `W_conn_sparse`：零初始化（softmax 权重初始即为"按距离的 Boltzmann 分布"）；
        * `tau_raw`：由 `config.tau_init` 决定；
        * `neuron_threshold`：初始化为正偏置 NEURON_THRESHOLD_INIT，保证初期存在激活。
        """
        c = self.config
        gen = torch.Generator(device="cpu").manual_seed(int(c.seed) + 1)
        with torch.no_grad():
            std = 1.0 / math.sqrt(float(c.input_dim))
            self.W_in.normal_(0.0, std, generator=gen)
            # Xavier 均匀：bound = sqrt(6 / (fan_in + fan_out))
            fan_in, fan_out = self.n_out_syn, int(c.output_dim)
            bound = math.sqrt(6.0 / float(fan_in + fan_out))
            self.W_out.uniform_(-bound, bound, generator=gen)
            self.W_conn_sparse.zero_()
            self.neuron_threshold.fill_(NEURON_THRESHOLD_INIT)
            self.tau_raw.fill_(float(c.tau_init))
            if self.readout_bias_enabled:
                # 输出层偏置零初始化：初始前向与无 bias 时完全一致
                self.W_out_bias.zero_()

    def current_tau(self) -> float:
        """返回当前 tau 的数值（= softplus(tau_raw) + 0.01 > 0）。"""
        return float((F.softplus(self.tau_raw) + 0.01).item())

    @property
    def has_readout_bias(self) -> bool:
        """输出层是否带 bias（由 `Config.readout_bias` 决定）。"""
        return bool(getattr(self, "readout_bias_enabled", False))

    def apply_readout(self, s_out: torch.Tensor) -> torch.Tensor:
        """输出汇总：可选 dropout -> 线性读出（可选 bias）。

        参数
        ----
        s_out : torch.Tensor
            形状 [B, N*y_out] 的最终轮输出突触信号。

        返回
        ----
        torch.Tensor
            形状 [B, output_dim] 的 logits。

        默认路径不变性
        --------------
        `dropout=0.0` 时 `readout_dropout` 为 `nn.Identity`（不消耗随机数、恒等映射）；
        `readout_bias=False` 时不加 bias。因此默认配置下本方法等价于原来的
        `s_out @ W_out`，数值逐位一致。
        """
        h = self.readout_dropout(s_out)
        logits = h @ self.W_out
        if self.has_readout_bias:
            logits = logits + self.W_out_bias
        return logits
    # ==================================================================
    # 步骤 2a：稀疏空间连接（masked softmax）
    # ==================================================================
    def sparse_propagate(self, s_in: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """步骤 2a：按边索引做 masked softmax 的空间连接传播（稀疏实现）。

        数学形式
        --------
        对每个输出突触 o：
            logits_oj = -dist[o, j] / tau + W_conn[o, j]
            w_oj      = exp(logits_oj - m_o) / Σ_{j' ∈ conn(o)} exp(logits_oj' - m_o)
            s_out[b, o] = Σ_{j ∈ conn(o)} w_oj · s_in[b, j]
        其中 conn(o) 是输出突触 o 连接的输入突触集合（mask=1 的边）。

        参数
        ----
        s_in : torch.Tensor
            形状 [B, N*y_in] 的输入突触信号。
        tau : torch.Tensor
            标量张量，tau > 0，距离衰减温度。

        返回
        ----
        torch.Tensor
            形状 [B, N*y_out] 的输出突触信号。

        关键不变量
        ----------
        * 归一化按 `edge_index[0]`（输出突触）分组——即"对每个输出突触，在其连接的
          输入突触上做 softmax"，禁止反向分组、禁止全局 softmax；
        * 仅遍历 E 条边，**不构造 dense [N*y_out, N*y_in] 权重矩阵**。
        """
        # ---- 形状与设备一致性校验（契约） ----
        if s_in.dim() != 2:
            raise ValueError(f"s_in 必须为 2D [B, N*y_in]，当前 shape={tuple(s_in.shape)}")
        if s_in.shape[1] != self.n_in_syn:
            raise ValueError(
                f"s_in 第二维必须为 N*y_in={self.n_in_syn}，当前 {s_in.shape[1]}"
            )
        if s_in.device != self.edge_dist.device:
            raise ValueError(
                f"设备不一致：s_in 在 {s_in.device}，模型拓扑 buffer 在 "
                f"{self.edge_dist.device}，请先 model.to(device) 或 x.to(device)"
            )
        assert tau > 0, f"[契约失败] sparse_propagate 要求 tau > 0，当前 tau={tau}"

        # ---- 边级 logits：-dist/tau + 可学习边权 ----
        edge_logits = -self.edge_dist / tau + self.W_conn_sparse
        # 分组 softmax（分组 = 输出突触）；mask 已体现在"哪些边存在"上
        edge_weight = segment_softmax(edge_logits, self.edge_index[0], self.n_out_syn)

        # ---- 消息传递：msgs[e] = w_e * s_in[:, edge_index[1][e]] ----
        msg = s_in.index_select(1, self.edge_index[1]) * edge_weight.unsqueeze(0)
        # ---- 稀疏聚合到输出突触（scatter_add 语义） ----
        s_out = torch.zeros(
            (s_in.shape[0], self.n_out_syn), dtype=s_in.dtype, device=s_in.device
        )
        s_out = s_out.index_add(1, self.edge_index[0], msg)
        return s_out

    # ==================================================================
    # 前向传播：输入编码 -> T 轮四步闭环 -> 输出汇总
    # ==================================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：严格按"输入编码 -> T 轮四步闭环 -> 输出汇总"实现。

        参数
        ----
        x : torch.Tensor
            形状 [B, input_dim] 的输入特征（MNIST 已展平为 784 维）。

        返回
        ----
        torch.Tensor
            形状 [B, output_dim] 的分类 logits。

        流程
        ----
        1. 输入编码：s_in = x @ W_in，[B, input_dim] -> [B, N*y_in]；
        2. 迭代 T 轮四步闭环：
           2a s_out  = sparse_propagate(s_in, tau)          # 稀疏空间连接
           2b n_inp  = sparse_aggregate(s_out, ...)          # scatter sum 到神经元
           2c a      = ReLU(n_inp + neuron_threshold)        # 神经元激活
           2d s_in   = LN(sparse_broadcast(a, ...) + alpha * s_in)  # 广播 + 残差 + LN
        3. 输出汇总：用最后一轮 2a 得到的 s_out 接输出层，logits = s_out @ W_out。

        说明
        ----
        当 T 轮迭代全部结束后，最后一步 2d 只更新了输入突触信号；为了让"输出突触"
        与最终状态一致，这里额外执行一次 2a（T 轮的最后一轮内则复用该结果），
        因此 `logits` 使用的是【最终轮 2a】的输出突触信号，符合规格第 3 条。

        二期新增（**只观测、不影响数值**）
        --------------------------------
        逐轮记录 2a 输出的**非零覆盖比例**（s_out 中非零列所占比例），供
        `get_topology_stats()` 报告；该记录只做布尔统计，且每个 batch 结束前把
        张量转为 Python 标量，不改变任何参与反向传播的张量。
        """
        # ---- 形状与设备一致性校验（契约） ----
        if not torch.is_tensor(x):
            raise TypeError(f"forward 输入必须为 torch.Tensor，当前类型 {type(x).__name__}")
        if x.dim() != 2:
            raise ValueError(f"输入必须为 2D [B, input_dim]，当前 shape={tuple(x.shape)}")
        if x.shape[1] != self.config.input_dim:
            raise ValueError(
                f"输入第二维必须为 input_dim={self.config.input_dim}，当前 {x.shape[1]}"
            )
        if x.device != self.W_in.device:
            raise ValueError(
                f"设备不一致：x 在 {x.device}，模型参数在 {self.W_in.device}，"
                f"请先 model.to(device) 或 x.to(device)"
            )
        if self.T < 1:
            raise ValueError(f"迭代轮数 T 必须 >= 1，当前 T={self.T}")

        tau = F.softplus(self.tau_raw) + 0.01
        assert tau.item() > 0.0, f"[契约失败] tau 必须 > 0，当前 tau={tau.item()}"

        # ---- 1. 输入编码：[B, input_dim] -> [B, N*y_in] ----
        s_in = x @ self.W_in

        s_out = torch.zeros(
            (x.shape[0], self.n_out_syn), dtype=s_in.dtype, device=s_in.device
        )
        for _ in range(self.T):
            # ---- 2a 稀疏空间连接（masked softmax）----
            s_out = self.sparse_propagate(s_in, tau)
            # 二期：仅作观测的 2a 非零覆盖比例（不参与计算图、不影响任何数值）
            # 修复 W1：由无界 list.append 改为 O(1) 定长累计（sum / count / last）
            with torch.no_grad():
                self._accumulate_round_coverage(
                    float((s_out.abs() > 0.0).any(dim=0).float().mean().item())
                )
            # ---- 2b 输出突触 -> 神经元（scatter sum）----
            neuron_input = sparse_aggregate(
                s_out, self.neuron_of_output_syn, self.N
            )
            # ---- 2c 神经元激活：ReLU(neuron_input + threshold) ----
            a = F.relu(neuron_input + self.neuron_threshold)
            # ---- 2d 神经元 -> 输入突触（broadcast）+ 残差 + LayerNorm ----
            s_in_new = sparse_broadcast(a, self.neuron_of_input_syn)
            s_in = self.ln_s_in(s_in_new + self.alpha * s_in)

        # ---- 3. 输出汇总：使用最终轮 2a 的输出突触信号 ----
        # apply_readout = 可选 dropout -> s_out @ W_out（可选 + bias）
        # 默认路径下 dropout 为 Identity、无 bias，故与原来的 `s_out @ W_out` 逐位一致
        return self.apply_readout(s_out)

    # ==================================================================
    # 统计接口
    # ==================================================================
    def reset_round_output_coverage(self) -> None:
        """重置"每轮 2a 非零覆盖"的动态累计容器（O(1) 定长，非无界列表）。

        设计动机（修复审查缺陷 W1）
        --------------------------
        原实现用 `List[float]` 逐轮 append：12ep×469batch×T=4 ≈ 22512 个元素且
        **只增不减**，并且既不在 `state_dict` 里、也不随 `model.to(device)` 重置；
        多阶段训练（先 `--max-batches` 快筛、再全量）会把两段统计混在一起，
        从 `.pt` 加载后 `get_topology_stats()` 的动态覆盖只能变成 NaN。

        现改为三个 Python 标量组成的定长容器：
        * `_round_coverage_sum`：各轮覆盖比例之和（用于算全程均值）；
        * `_round_coverage_count`：已累计轮数；
        * `_round_coverage_last`：末轮覆盖比例。

        **调用时机**：`__init__` 末尾自动调用一次；每次新训练开始时（见 train.py
        的 `run_smoke_test` / `run_full_training`）再次调用，保证不同训练阶段、
        不同 `model.to(device)` / `load_state_dict` 之间统计不互相污染。

        历史值的复核不依赖内存态：末轮/均值/轮数会被显式写入 checkpoint 的
        `topology_stats` 与顶层 `dynamic_coverage` 字段（见 train.py）。
        """
        self._round_coverage_sum: float = 0.0
        self._round_coverage_count: int = 0
        self._round_coverage_last: float = float("nan")

    def _accumulate_round_coverage(self, value: float) -> None:
        """把单轮 2a 非零覆盖比例累加进定长容器（纯 Python 标量，O(1) 内存）。

        参数
        ----
        value : float
            本轮 `s_out` 中非零列占比，取值 [0, 1]。

        不变量
        ------
        * 容器大小恒为 3 个标量，与训练轮数无关；
        * 只做观测量累加，**不参与计算图、不影响 loss / 梯度 / 随机流**。
        """
        self._round_coverage_sum += float(value)
        self._round_coverage_count += 1
        self._round_coverage_last = float(value)

    def round_output_coverage(self) -> Dict[str, float]:
        """返回动态 2a 覆盖的当前累计快照（末轮 / 全程均值 / 累计轮数）。

        返回
        ----
        Dict[str, float]
            * "out_nonzero_coverage_last"：末轮非零列占比；从未跑过前向时为 NaN；
            * "out_nonzero_coverage_mean"：全程均值；从未跑过前向时为 NaN；
            * "out_nonzero_coverage_rounds"：已累计的观测轮数。

        `out_nonzero_coverage_rounds` 的口径（**易误读点，此处明确限定**）
        ---------------------------------------------------------------
        该轮数覆盖**训练与评估两条路径的全部前向**，**不是"仅训练前向"**：
        `forward` 每执行一轮 2a 就累加 1 次，而 `forward` 同时被训练循环
        （`train.train_one_epoch`）与评估循环（`train.evaluate`）调用，因此

            rounds = (训练批数 + 评估批数) × epoch 数 × T

        **阶段隔离**由 `reset_round_output_coverage()` 保证：它在 `__init__` 末尾、
        以及每次训练开始时（`train.reset_dynamic_coverage` → `run_smoke_test` /
        `run_full_training` 各一次）重置，因此冒烟阶段与全量阶段各自从 0 开始累计、
        不会互相污染（`--max-batches` 限批快筛与随后全量之间亦如此）。

        **实测佐证**（SMALL_CONFIG：T=2，2 epoch ×（3 训练 batch + 3 评估 batch））：

            [epoch 1] 训练后 rounds=6；评估后 rounds=12
            [epoch 2] 训练后 rounds=18；评估后 rounds=24
            [最终]   rounds=24 = (3 训练 + 3 评估) × 2 epoch × T=2

        复现脚本：`checkpoints/n3d_sphere/_verify/verify_r4_round_count_scope.py`
        （同时验证重置后 rounds=0 且末轮为 NaN）。
        """
        if self._round_coverage_count <= 0:
            nan = float("nan")
            return {
                "out_nonzero_coverage_last": nan,
                "out_nonzero_coverage_mean": nan,
                "out_nonzero_coverage_rounds": 0.0,
            }
        return {
            "out_nonzero_coverage_last": float(self._round_coverage_last),
            "out_nonzero_coverage_mean": float(
                self._round_coverage_sum / float(self._round_coverage_count)
            ),
            "out_nonzero_coverage_rounds": float(self._round_coverage_count),
        }

    def count_parameters(self) -> int:
        """返回可学习参数总数（含 LayerNorm 的 weight/bias）。"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def count_dense_weight_tensors(self) -> int:
        """统计形状恰为 [N*y_out, N*y_in] 的**权重类**张量个数（应为 0）。

        这是"禁止 materialize dense 权重矩阵"这一设计约束的可执行判据：
        连接权重必须是边级参数 `W_conn_sparse` [E]，不允许出现 [N*y_out, N*y_in]
        形状的可学习权重。

        扫描范围（**比规格字面更严格**）
        --------------------------------
        规格只要求"禁止 materialize dense 权重矩阵"，本方法把检查面扩大到两类张量：
        1. `named_parameters()`：任何具有该形状的**可学习参数**都视为违规；
        2. `named_buffers()`：除显式豁免的 `dist` / `mask` 两个几何 buffer 外，
           任何具有该形状的 buffer 也视为违规——这可以捕获"本应是参数却被误注册为
           buffer"或"把稠密权重塞进 buffer"的隐蔽写法。

        豁免说明：拓扑量 `dist`（输出突触-输入突触距离）与 `mask`（距离阈值掩码）
        虽然是同一形状，但它们是**预计算的几何/掩码 buffer**（非权重、不参与训练、
        不存在可替代的稀疏表示），故不计入。

        返回
        ----
        int
            命中的权重类张量个数；本设计下恒为 0。
        """
        target = (self.n_out_syn, self.n_in_syn)
        hits = 0
        for name, p in self.named_parameters():
            if tuple(p.shape) == target:
                log_warn(f"发现 dense 权重张量：参数 {name} 形状 {tuple(p.shape)}")
                hits += 1
        # buffer 中只允许 dist / mask 两个拓扑量具有该形状（已知且必须存在）
        for name, b in self.named_buffers():
            if tuple(b.shape) == target and name not in ("dist", "mask"):
                log_warn(f"发现 dense 权重张量：buffer {name} 形状 {tuple(b.shape)}")
                hits += 1
        return hits

    def get_connection_stats(self) -> Dict[str, float]:
        """返回连接统计：连接密度（规格口径）、零元素占比、平均出度、当前 tau。

        返回
        ----
        Dict[str, float]
            * "num_edges"：边数 E；
            * "sparsity"：**连接密度** = E/(N*y_out*N*y_in)，即 current_spec.md 规定的
              "连接稀疏度"口径（规格公式即密度定义）；阶段 A 要求 < 0.1。
              注：`utils.connection_sparsity` 是该口径的同义别名（已弃用），
              新代码请统一使用 `utils.connection_density`；
            * "zero_ratio"：dense 矩阵的零元素占比 = 1 - E/(N*y_out*N*y_in)，
              仅用于日志对照，不参与验收判据；
            * "avg_out_degree"：平均每个输出突触的连接数 E/(N*y_out)；
            * "tau"：当前 tau 值（> 0）。

        二期扩展说明
        ------------
        二期新增的方向性/连通性指标放在 `get_topology_stats()` 中，本方法**保持
        与原版完全相同的 5 个键**，以免破坏一期既有的日志、判据与产物字段解析。
        """
        density = connection_density(self.num_edges, self.n_out_syn, self.n_in_syn)
        return {
            "num_edges": float(self.num_edges),
            "sparsity": float(density),
            "zero_ratio": float(1.0 - density),
            "avg_out_degree": float(self.num_edges) / float(self.n_out_syn),
            "tau": self.current_tau(),
        }

    def get_topology_stats(self) -> Dict[str, float]:
        """返回二期新增的拓扑统计量（几何方向性 + 2a 覆盖 + 连通性）。

        本方法**只读取** `__init__` 预计算的 buffer 与 forward 累计的标量，
        不重算任何拓扑量、不改变任何数值，可安全地在冒烟测试与正式训练中调用。
        （gap 直接复用 `__init__` 缓存的 `edge_axis_gap`，不再重复 index_select。）

        返回
        ----
        Dict[str, float]
            * "topology" / "flow_axis" / "space_radius"：本次生效的几何指纹；
            * "reverse_edge_ratio"：**逆向边占比** = 沿流向轴"输出突触低于输入突触"
              （gap < 0，逆着 −axis → +axis 的流向）的边 / E。这是"方向性是否真正
              建立"的判据：0 表示严格 DAG 式的上行传播，≈ 0.5 表示方向性几乎不存在；
            * "forward_edge_ratio"：= 1 − reverse_edge_ratio（gap > 0 的上行边占比）；
            * "axis_gap_mean"/"axis_gap_min"/"axis_gap_max"：沿流向轴的边间距分布，
              **口径为代码口径** `gap = axis(output_syn_pos) − axis(input_syn_pos)`
              （axis 由 `flow_axis` 选定；仅统计已有的边），不做 Δz / δ 分解；
            * "neuron_{axis}_mean/min/max/std"：**流向轴上的神经元高度分布**
              （球形拓扑下应落在 [−R, R]，立方体下落在 [0, L]）；
            * "connected_output_ratio"：**静态** 2a 覆盖 = 至少有 1 条入边的输出突触
              占比（sphere 有向拓扑下会显著低于 1，因为只有高度接近的神经元才存在边）；
            * "out_nonzero_coverage_last"/"out_nonzero_coverage_mean"：**动态** 2a 覆盖
              = forward 各轮 `s_out` 中非零列占比（末轮 / 全程均值），用于观察信息实际
              到达了多大范围的输出突触；尚未跑过前向时为 NaN；
            * "out_nonzero_coverage_rounds"：已累计的观测轮数，覆盖**训练与评估
              两条路径的全部前向**（口径 = (训练批数 + 评估批数) × epoch × T，
              详见 `round_output_coverage()` 的限定说明；阶段隔离由
              `reset_round_output_coverage()` 保证）。
              该容器为 O(1) 定长（见 `reset_round_output_coverage`），由 train.py 在
              每次训练开始时重置，并把末轮/均值/轮数显式写入 checkpoint 元数据，
              因此 `.pt` 加载后可脱离内存态直接复核历史动态覆盖；
            * "weak_components"：神经元层**弱连通分量数**；
            * "largest_component_ratio"：**最大分量占比** = 最大分量神经元数 / N。

            连通性说明（经修复与实测）：`D=0.15` 下两种拓扑的神经元层弱连通图都是
            **单连通**（分量数 1、最大分量占比 1.0）；即使把 D 放大到 0.25 仍为单连通。
            因此连通性不是本架构的瓶颈，`connected_output_ratio` 与逆向边占比才是
            刻画几何方向性的有效指标。

        异常
        ------
        ValueError
            当前配置下 E == 0（不可能发生：`build_edge_index_from_dist` 会先行抛错）。
        """
        if self.num_edges <= 0:
            raise ValueError(f"当前拓扑 E=0，无法统计拓扑指标（E={self.num_edges}）")
        axis = self.flow_axis_index
        # gap 口径（全文统一，不做 Δz / δ 分解）：
        #   gap = axis(output_syn_pos) - axis(input_syn_pos)
        # 正常路径直接复用 `__init__` 预计算的 edge_axis_gap（非持久化 buffer），
        # 不再每次调用都 index_select 重算 O(E)；数值与重算结果逐位相同。
        # **兜底路径（第 4 轮加固 + 第 5 轮回写自愈）**：`edge_axis_gap` 是
        # persistent=False 的内存态 buffer，与 W1 容器同类——任何绕过 `__init__`
        # （例如直接从 state_dict 重建 buffer）的加载路径都会缺该属性。此时按与
        # `__init__` **同一个** 计算逻辑（`_compute_edge_axis_gap`）重算，
        # 避免抛 AttributeError、也避免两份实现分叉。
        # 重算后**回写缓存**使缓存语义自愈（下次调用直接命中，不再重复 O(E) 重算）。
        # ⚠️ 回写必须走 `register_buffer(..., persistent=False)`：
        #   该名字在此路径下从未被 register 过，若直接写 `self._buffers[...]`
        #   会漏掉 `_non_persistent_buffers_set` 登记，使其被当作**持久化** buffer
        #   而泄漏进 `state_dict()`，破坏"state_dict 键集增量恰为 3 个"的不变量。
        #   `register_buffer` 会正确登记非持久化集合。helper 从 `self.output_syn_pos` /
        #   `self.input_syn_pos` 重算，结果本身即在目标设备上。
        gap = getattr(self, "edge_axis_gap", None)
        if gap is None:
            gap = self._compute_edge_axis_gap(
                self.output_syn_pos.index_select(0, self.edge_index[0]),
                self.input_syn_pos.index_select(0, self.edge_index[1]),
            )
            self.register_buffer("edge_axis_gap", gap, persistent=False)
        n_rev = int(self.edge_reverse_flag.sum().item())
        rev_ratio = float(n_rev) / float(self.num_edges)

        neuron_h = self.neuron_pos[:, axis]
        comp = self.neuron_component_id
        # 连通性统计：由预计算的 neuron_component_id 派生（修复后为最小标签传播结果）
        n_comp = int(torch.unique(comp).numel())
        comp_sizes = torch.bincount(comp, minlength=int(self.N))
        largest_ratio = float(comp_sizes.max().item()) / float(self.N)

        # 动态 2a 覆盖：来自 O(1) 定长累计容器（修复 W1），不再是内存态无界列表
        coverage = self.round_output_coverage()
        cover_last = coverage["out_nonzero_coverage_last"]
        cover_mean = coverage["out_nonzero_coverage_mean"]
        cover_rounds = coverage["out_nonzero_coverage_rounds"]

        return {
            "topology": 1.0 if self.topology == "sphere" else 0.0,
            "flow_axis": float(axis),
            "space_radius": float(self.space_radius),
            "num_edges": float(self.num_edges),
            "reverse_edges": float(n_rev),
            "reverse_edge_ratio": rev_ratio,
            "forward_edge_ratio": 1.0 - rev_ratio,
            "axis_gap_mean": float(gap.mean().item()),
            "axis_gap_min": float(gap.min().item()),
            "axis_gap_max": float(gap.max().item()),
            "neuron_axis_mean": float(neuron_h.mean().item()),
            "neuron_axis_min": float(neuron_h.min().item()),
            "neuron_axis_max": float(neuron_h.max().item()),
            "neuron_axis_std": float(neuron_h.std(unbiased=False).item()),
            "connected_output_ratio": float(
                self.connected_output_mask.to(torch.float32).mean().item()
            ),
            "out_nonzero_coverage_last": cover_last,
            "out_nonzero_coverage_mean": cover_mean,
            "out_nonzero_coverage_rounds": cover_rounds,
            "weak_components": float(n_comp),
            "largest_component_ratio": largest_ratio,
        }


class MLPBaseline(nn.Module):
    """对照实验用的普通 MLP 基线（**不是** N3D 模型，仅用于瓶颈归因）。

    结构：`784 -> hidden -> 10`，隐藏层后接 ReLU，可选 dropout。

    存在的意义
    ----------
    此前六轮已证明容量维度（N / y_in,y_out / D / T）与拓扑种子都无法把 MNIST test_acc
    推过 99%（最佳 97.84%@seed42 / 97.90%@seed2024）。为判定"瓶颈在架构还是在数据"，
    需要一个**参数量同量级、优化器/epoch/预处理完全相同**的普通 MLP 作对照：
    若 MLP 明显更高，说明四步闭环在损失信息（瓶颈在架构）；若两者接近，
    说明是数据/任务本身的上限（瓶颈在数据侧）。

    接口兼容
    --------
    为让**同一份训练循环代码**（`train.train_one_epoch` / `evaluate` /
    `_run_training_with_config`）无需分支地驱动它，本类提供与 `ThreeDNeuronSpace`
    同名的以下方法：`count_parameters()`、`count_dense_weight_tensors()`、
    `get_connection_stats()`。其中连接类指标（边数/稀疏度/平均出度/tau）对 MLP
    不适用，统一以 `0.0` 占位，由调用方（train.py）在 `arch == "mlp"` 时跳过相关判据。

    参数
    ----
    config : Config
        超参配置；使用 `input_dim`、`output_dim`、`hidden_dim`、`dropout` 与 `seed`。

    关键不变量
    ----------
    * 参数量与主模型（`HIGHACC_CONFIG` 下 1,691,730）同量级；
    * 初始化由 `config.seed` 派生的 `torch.Generator` 控制，与主模型同样可复现；
    * MLP 不使用 `topology` 几何（`get_topology_stats()` 返回空字典占位，
      调用方按 `arch` 跳过），因此 `--topology sphere --arch mlp` 不会改变 MLP 数值。
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(int(config.input_dim), int(config.hidden_dim))
        self.dropout: nn.Module = (
            nn.Dropout(float(config.dropout)) if config.dropout > 0.0 else nn.Identity()
        )
        self.fc2 = nn.Linear(int(config.hidden_dim), int(config.output_dim))
        # 用 config.seed 派生的局部生成器初始化，保证与主模型一致的可复现语义
        gen = torch.Generator(device="cpu").manual_seed(int(config.seed) + 1)
        with torch.no_grad():
            # fc1：Kaiming 均匀（fan_in=784, a=sqrt(5) 为 nn.Linear 默认），与主模型
            # W_in 的 LeCun 风格同属"按 fan_in 缩放"，量级可比；
            # fc2：Xavier 均匀，与主模型 W_out 完全一致。
            nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5), generator=gen)
            bound1 = 1.0 / math.sqrt(float(config.input_dim))
            self.fc1.bias.uniform_(-bound1, bound1, generator=gen)
            fan_in, fan_out = int(config.hidden_dim), int(config.output_dim)
            bound2 = math.sqrt(6.0 / float(fan_in + fan_out))
            self.fc2.weight.uniform_(-bound2, bound2, generator=gen)
            self.fc2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, input_dim] -> [B, output_dim]`：`fc1 -> ReLU -> dropout -> fc2`。

        参数
        ----
        x : torch.Tensor
            形状 [B, input_dim] 的输入特征。

        返回
        ----
        torch.Tensor
            形状 [B, output_dim] 的 logits。
        """
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)

    # ------------------------------------------------------------------
    # 与 ThreeDNeuronSpace 同名的方法（供训练循环无分支调用）
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """返回可学习参数总数。"""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def count_dense_weight_tensors(self) -> int:
        """MLP 无稀疏连接约束，恒返回 0（保持与主模型调用方兼容）。"""
        return 0

    def get_connection_stats(self) -> Dict[str, float]:
        """连接类指标对 MLP 不适用，统一返回 0.0 占位。"""
        return {
            "num_edges": 0.0,
            "sparsity": 0.0,
            "zero_ratio": 0.0,
            "avg_out_degree": 0.0,
            "tau": 0.0,
        }

    def get_topology_stats(self) -> Dict[str, float]:
        """拓扑指标对 MLP 不适用，返回空字典（调用方按 `arch` 跳过打印）。"""
        return {}
