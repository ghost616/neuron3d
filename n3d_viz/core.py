"""N3D 二期拓扑数据加载与抽取（纯逻辑层，与 GUI / CLI 解耦）。

本模块是 n3d_viz 的唯一数据入口，负责：

1. 加载 checkpoint（``torch.load(..., map_location="cpu")``）并做二期必需键校验，
   区分「路径不存在」「文件损坏」「非二期产物（缺拓扑键）」三类错误；
2. 把张量抽取为纯 Python 结构（``list[tuple[float, float, float]]`` 等），
   **不持有** syn_dist 等巨型张量；
3. 计算分层着色分组、度统计、阈值过滤统计；
4. 派生产物名并写出三件套。

几何口径（**对几何零假设**）：图形完全由 ``neuron_pos`` 决定，本模块不假设
球 / 立方体 / 圆柱，也不假设晶格（FCC）或分层规整性；任意 N3D 拓扑产物
（随机点云、任意曲面、非晶格、非均匀分层）都按同一套逻辑渲染。
「层」= ``level_node_reach`` 给出的**同时计算的神经元分组**（不是几何切片假设）；
层参考平面取该组神经元沿流向轴位置的**均值**，必然包含这一组神经元。

**两端全连接包裹（``fc_dim != 0``）**

`n3d_shape` 第三轮引入的 `fc_dim` 把 N3D 核心夹在两个全连接层之间
（``fc_dim == -1`` 表示宽度跟随 ``N``，``> 0`` 表示显式宽度 ``H``）。本模块按
**三态**判定是否展示该结构（见 :func:`is_fc_enabled` / :func:`extract_fc`）：

* ``config`` 无 ``fc_dim`` 键（二期 / 三期未启用产物）→ 无 FC，走既有展示、不报错；
* ``config.fc_dim == 0`` → 无 FC，走既有展示、不报错；
* ``config.fc_dim != 0`` 且 FC 键齐全 → **有 FC**，展示面板 / 边界块 / 抽样连线；
* ``config.fc_dim != 0`` 但缺任一 FC 键 → **产物损坏**，报错退出非 0（不静默降级）。

判定**只用** ``fc_dim != 0``，不依赖「键是否存在」来启用。

引用纪律：基准产物 ``checkpoints/n3d_sphere/model.pt`` 的 ``neuron_pos`` 与分层
结构由排布参数决定、与 seed 无关；而边集（E）、边权重、突触位置随 seed 变化，
故一切边级数字必须标注 seed。
"""

from __future__ import annotations

import colorsys
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 契约：二期产物 ``model_state_dict`` 必须包含的拓扑键。
# 实测来源：checkpoints/n3d_sphere/model.pt（seed=42, N=256, y=8x8）。
# 对照：一期产物 checkpoints/n3d_model_full.pt 缺 edge_src / edge_dst /
# edge_weight / in_scope_mask / out_scope_mask / level_node_reach 六键。
# ---------------------------------------------------------------------------
REQUIRED_KEYS: tuple[str, ...] = (
    "neuron_pos",
    "edge_src",
    "edge_dst",
    "edge_weight",
    "edge_dist",
    "in_scope_mask",
    "out_scope_mask",
    "topo_index",
    "level_node_reach",
    "level_edge_reach",
    "in_degree",
    "out_degree",
)

#: 供 CLI / GUI 展示的默认权重阈值（默认规模 seed=42 实测 |w|>=0.30 保留 379/736）。
DEFAULT_THRESHOLD: float = 0.30

#: 默认输出目录（相对仓库根）。
DEFAULT_OUT_DIR: str = "checkpoints/n3d_viz"

#: 内联 HTML 的默认保留位数（``pos_digits=None`` / ``weight_digits=None`` 时生效）。
#: 坐标 6 位小数对应 max|diff| 实测 4.510e-07 < 1e-6；权重 8 位小数对应 max|diff| 实测 4.991e-09。
DEFAULT_POS_DIGITS: int = 6
DEFAULT_WEIGHT_DIGITS: int = 8

# ---------------------------------------------------------------------------
# 两端全连接包裹（``fc_dim != 0``）的参数与几何常量
# ---------------------------------------------------------------------------
#: 抽样口径默认值：每个 S_in / S_out 神经元各取 ``|w|`` 最大的 top-k 条连线。
#: 实测（来源 `checkpoints/n3d_shape/full_shapesphere_N825_..._fc-1_s42_fc_align.pt`，
#: seed=42，|S_in|=582、|S_out|=588）：k=3 时 582×3 + 588×3 = **3,510** 条。
DEFAULT_FC_TOP_K: int = 3
#: ``--fc-top-k`` 的合法闭区间（越界一律报错，不做静默截断）。
MIN_FC_TOP_K: int = 1
MAX_FC_TOP_K: int = 8

#: 界面板在平面内两轴上的目标跨度 = 神经元云对应轴跨度 × 该比例。
#: 取 1.0 使面板与云**同尺度**（实测 0.90 时面板只占云横向尺寸的约 40%，
#: 在 825 神经元 + 2,588 边的视图里不容易看出「两端包裹」的结构）。
FC_PANEL_SPAN_RATIO: float = 1.0
#: 面板与神经元云的**流向轴间隙** = 云流向轴跨度 × 该比例（硬断言：「不相交」的余量来源）。
FC_PANEL_GAP_RATIO: float = 0.15
#: 面板在流向轴上的厚度 = 面板单元边长 × 该比例（再叠加到间隙之上）。
FC_PANEL_THICKNESS_RATIO: float = 0.20
#: 面板厚度下界（防止单元边长退化到 0 时厚度为 0、不重叠判据退化为「不接触」）。
FC_PANEL_MIN_THICKNESS: float = 1e-3
#: 面板厚度上界 = 云流向轴跨度 × 该比例（保证面板不会「厚到」与自身间隙比失衡）。
FC_PANEL_MAX_THICKNESS_RATIO: float = 0.10
#: 边界块在流向轴上的尺寸 = 云跨度 × 该比例；另外两轴尺寸 = 面板对应轴尺寸 × :data:`FC_BLOCK_LATERAL_RATIO`。
FC_BLOCK_FLOW_RATIO: float = 0.14
FC_BLOCK_LATERAL_RATIO: float = 0.60
#: 边界块中心到「面板外侧表面」的额外偏移 = 云跨度 × 该比例。
FC_BLOCK_OFFSET_RATIO: float = 0.06

#: 抽样口径的**显式声明文本**（写入 HTML meta / OBJ 注释 / README）：
#: 必须声明「非全部连接」，避免把抽样图误读为全连接结构。
FC_NOT_ALL_CONNECTIONS_TEXT: str = "抽样显示（每神经元 top-k），非全部连接"

#: 流向轴名 -> 坐标分量下标。
FLOW_AXIS_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}
#: 默认流向轴（二期 / 三期产物 config 均含 ``flow_axis``，缺省时按 ``z``）。
DEFAULT_FLOW_AXIS: str = "z"
#: 流向轴缺少 ``x`` / ``y`` 时的兜底平面法向（二维退化为单轴排布时使用）。
_FALLBACK_PLANE_AXIS: tuple[str, str] = ("x", "y")

#: **CLI 默认形式**（也即零回归锚点）所用的写出参数集 —— 全模块**唯一事实来源**。
#:
#: 为什么必须集中定义：零回归锚点 ``checkpoints/n3d_viz/viz_model.*`` 的语义是
#: 「**CLI 默认形式**的产物」。若 ``__main__`` 的默认值与锚点的重渲参数各自手写一份，
#: 日后任一默认值变化都会导致两种坏结局之一：锚点断言**无故 FAIL**（被误判为回归），
#: 或为了让断言变绿而两边一起改、**锚点悄悄漂移成「另一套默认形式」的产物**。
#: 因此 ``__main__`` 的默认路径与 ``verify_viz`` 的 ``[2d][重渲]`` 都经
#: :func:`render_default` 使用本集合；``verify_viz`` 另有 ``[2e]`` 组承重断言，
#: 比对 argparse 默认值与本集合（任一默认值被改动即 FAIL）。
DEFAULT_WRITE_OPTIONS: dict[str, Any] = {
    "threshold": DEFAULT_THRESHOLD,
    "ply_binary": True,
    "with_ply_edges": False,
    "include_planes": True,
    # 两端全连接包裹的抽样口径；无 FC 产物该参数不影响任何字节（零回归）。
    "fc_top_k": DEFAULT_FC_TOP_K,
}

#: 渲染器源码位置（构建时读入并内联，保证 HTML 单文件自包含）。
ASSETS_DIR: Path = Path(__file__).resolve().parent / "assets"

#: **两端全连接包裹的渲染器**（叠加层实现，见该文件头部的「为什么是叠加层」）。
#: 它**只在** ``data.fc`` 非 None 时才被内联：无 FC 产物的 HTML 因此逐字节不变。
#: 之所以单独成文件而不改进 ``viewer.js``：``viewer.js`` 与 ``viewer.html`` 都被
#: 逐字内联进每一份 HTML，改动它们的任何一个字节都会破坏「无 FC 产物逐字节零回归」。
FC_VIEWER_ASSET: str = "viewer_fc.js"

_TPL_MARKER: str = "/*__N3D_VIEWER_JS__*/"
_DATA_MARKER: str = "/*__N3D_DATA_JSON__*/"


# ---------------------------------------------------------------------------
# 层配色（HTML 与 PLY 的唯一同源实现）
#
# 本模块对几何零假设：图形完全由 ``neuron_pos`` 决定，不假设球 / 立方体 /
# 圆柱，也不假设晶格或分层规整性；层数 K 由 ``level_node_reach`` 决定，可以
# 任意大（例如非均匀分层的 cylinder λ=2 实测 K=15）。因此配色不能再用
# 「9 色 + ``k % 9`` 循环」——那会让第 9 层与第 0 层同色，分层着色失去可分辨性。
#
# 口径：
# * ``K <= 9``：取 :data:`LEVEL_PALETTE_BASE` 的**前 K 个**，逐字节不变，
#   既有产物（含二期 ``checkpoints/n3d_viz/viz_model.*``）零回归；
# * ``K > 9``：按**均匀色相**生成 K 个两两不同的颜色（标准库 ``colorsys``
#   做 HSV->RGB，零第三方依赖），颜色去重数恒等于 K（K ∈ [1, 64] 已断言）。
# ---------------------------------------------------------------------------

#: 层配色基础表（9 色 hex）；K <= 9 时按前 K 个取用，保证既有产物逐字节不变。
LEVEL_PALETTE_BASE: tuple[str, ...] = (
    "#4e8cff",
    "#00b7c2",
    "#2ecc71",
    "#a3d977",
    "#f7d154",
    "#f39c12",
    "#e8734a",
    "#d94f70",
    "#9b59b6",
)

#: 色相扩展调色板的参数（K > 9 时生效）。
_PALETTE_HUE_STEP: float = 0.0137     # 撞色时色相微移量（占整圈的比例）
_PALETTE_SAT: float = 0.62            # 饱和度固定
_PALETTE_VALUE: float = 0.95          # 明度基准
_PALETTE_VALUE_STEP: float = 0.02     # 撞色时明度微降量
_PALETTE_VALUE_LEVELS: int = 20       # 明度微降的循环档数
_PALETTE_SEARCH_LIMIT: int = 4096     # 单色最多尝试次数（保证必然终止）


class CheckpointError(RuntimeError):
    """加载 / 解析 checkpoint 时的错误基类。"""


class CheckpointNotFoundError(CheckpointError):
    """checkpoint 路径不存在或不是普通文件。"""


class CheckpointCorruptedError(CheckpointError):
    """checkpoint 文件损坏，或不是合法的 torch 序列化产物。"""


class CheckpointSchemaError(CheckpointError):
    """checkpoint 可读但不是二期产物（缺拓扑键），或张量形状不符合契约。"""


@dataclass
class PanelUnitGeometry:
    """一个全连接层单元在三维空间中的落位与配色依据。

    面板 = 垂直于流向轴的一片平面，内部按 ``ceil(sqrt(H))`` 列做网格排布。

    Attributes:
        unit: 该单元在全连接层内的下标（``0 .. H-1``）。
        pos: 单元中心的三维坐标（面板平面内）。
        norm: 配色依据——该单元「权重范数」。输入侧的 S_in 行取
            ``proj_weight[row, unit]``、S_out 行取 ``fc_out_weight[unit, col]``；
            输入侧 H 单元取 ``fc_in_weight[unit, :]`` 的 L2 范数。
        tags: 该单元被哪些角色使用（``"s_in"`` / ``"s_out"`` / ``"fc_in"``），
            用于渲染器标注与产物断言。
    """

    unit: int
    pos: tuple[float, float, float]
    norm: float
    tags: tuple[str, ...] = ()


@dataclass
class FcPanelGeometry:
    """一片全连接层面板的几何与不重叠证据（流轴向的区间硬断言在构造期完成）。

    Attributes:
        name: ``"input"`` / ``"output"``。
        axis: 流向轴名（``x`` / ``y`` / ``z``）。
        flow: 面板在流向轴上的中心坐标。
        thickness: 面板在流向轴上的厚度（半宽 = thickness / 2）。
        flow_interval: 面板占据的流向轴闭区间 ``(low, high)``。
        cols / rows: 网格列数 ``ceil(sqrt(H))`` 与行数。
        cell_size: 网格单元中心间距（面板平面内的排布步长）。
        units: ``H`` 个单元几何（含落位与配色范数）。
    """

    name: str
    axis: str
    flow: float
    thickness: float
    flow_interval: tuple[float, float]
    cols: int
    rows: int
    cell_size: float
    units: list[PanelUnitGeometry] = field(default_factory=list)


@dataclass
class BoundaryBlock:
    """输入 / 输出边界块（784 维输入、10 维输出）。

    Attributes:
        name: ``"input"`` / ``"output"``。
        label: 展示用标签（如 ``"输入 784"``）。
        dim: 边界块维度（784 / 10）。
        center: 块中心的三维坐标（置于面板**外侧**）。
        size: 块的三轴尺寸 ``(dx, dy, dz)``。
    """

    name: str
    label: str
    dim: int
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass
class FcData:
    """两端全连接包裹（``fc_dim != 0``）的纯 Python 抽取结果。

    仅在 ``fc_dim != 0`` 时构造；``fc_dim == 0`` / 无该键时为 ``None``
    （即 :attr:`TopologyData.fc` 为空），既有展示路径逐字节不变。

    Attributes:
        fc_dim: 产物记录的原始取值（``-1`` = 宽度跟随 N；``> 0`` = 显式宽度）。
        fc_width: 有效宽度 ``H``（由 FC 张量的实际形状推出，**不以 config 为准**）。
        input_dim / output_dim: 输入 / 输出维度（784 / 10）。
        proj_weight_shape / fc_out_weight_shape: ``[|S_in|, H]`` / ``[H, |S_out|]``。
        proj_count / fc_out_count: 两个矩阵的**全部**连线数（参数量，非抽样量）。
        proj_abs_mean / fc_out_abs_mean: 权重绝对值均值（摘要用，不逐元素留存）。
        fc_in_weight_shape: ``[H, input_dim]``（仅记录形状与参数量，不长期持有矩阵）。
        fc_in_count: ``fc_in_weight`` 参数量。
        in_unit_norms / out_unit_norms: 长度 H 的单元权重范数（配色依据）。
        s_in_order / s_out_order: 参与投影 / 读出的神经元 id（按 id 升序），
            与 ``proj_weight`` 的行序、``fc_out_weight`` 的列序一一对应。
        panels: ``{"input": FcPanelGeometry, "output": FcPanelGeometry}``。
        blocks: 输入 / 输出边界块。
        edges: 抽样连线三元组 ——
            (侧别, 单元下标, 神经元 id, 权重)。
        top_k: 抽样口径的 k。
        declared_text: 抽样口径的显式声明文本（含「非全部连接」）。
    """

    fc_dim: int
    fc_width: int
    input_dim: int
    output_dim: int
    proj_weight_shape: tuple[int, int]
    fc_out_weight_shape: tuple[int, int]
    proj_count: int
    fc_out_count: int
    proj_abs_mean: float
    fc_out_abs_mean: float
    fc_in_weight_shape: tuple[int, int]
    fc_in_count: int
    in_unit_norms: list[float] = field(default_factory=list)
    out_unit_norms: list[float] = field(default_factory=list)
    s_in_order: list[int] = field(default_factory=list)
    s_out_order: list[int] = field(default_factory=list)
    panels: dict[str, FcPanelGeometry] = field(default_factory=dict)
    blocks: list[BoundaryBlock] = field(default_factory=list)
    edges: list[tuple[str, int, int, float]] = field(default_factory=list)
    top_k: int = DEFAULT_FC_TOP_K
    declared_text: str = FC_NOT_ALL_CONNECTIONS_TEXT

    # -- 便捷视图 ---------------------------------------------------------
    @property
    def n_units(self) -> int:
        """有效宽度 H（每片面板的单元数）。"""
        return int(self.fc_width)

    @property
    def n_panel_points(self) -> int:
        """两片面板的单元点总数 == ``2 × H``。"""
        return 2 * int(self.fc_width)

    @property
    def n_edges(self) -> int:
        """抽样连线总数（＝两侧各 ``|S_*| × k`` 之和）。"""
        return len(self.edges)

    @property
    def expected_sample_edges(self) -> int:
        """抽样连线的**期望**条数 ``|S_in| * k + |S_out| * k``（用于自洽断言）。"""
        return (len(self.s_in_order) + len(self.s_out_order)) * int(self.top_k)

    def declared_statement(self) -> str:
        """返回完整的抽样口径声明（含两侧的全部连线数，供 meta / OBJ / README 引用）。

        Returns:
            形如「抽样显示（每神经元 top-k），非全部连接；实际 proj_weight
            480,150 条 + fc_out_weight 485,100 条」的文本。
        """
        return (
            f"{self.declared_text}；实际 proj_weight {self.proj_count:,} 条"
            f" + fc_out_weight {self.fc_out_count:,} 条"
        )



@dataclass
class TopologyData:
    """一期/二期拓扑的纯 Python 抽取结果。

    所有坐标与权重都以 Python ``float`` / ``int`` 保存，便于直接序列化进 HTML、
    写 PLY / OBJ，以及在无 torch 依赖的断言脚本中逐位比对。

    Attributes:
        checkpoint: checkpoint 绝对路径字符串。
        ckpt_stem: checkpoint 文件名（不含后缀），用于派生产物名。
        config: checkpoint 内的 config 字典（缺失则为空字典）。
        test_acc: checkpoint 记录的训练/测试准确率（缺失为 None）。
        n_neurons: 神经元数 N。
        n_edges: 神经元级连接数 E。
        neuron_pos: N 个神经元的三维坐标。
        edge_src: E 条边的源神经元下标。
        edge_dst: E 条边的目标神经元下标。
        edge_weight: E 条边的学得权重（有符号）。
        edge_dist: E 条边的神经元间距。
        in_scope_mask: N 个布尔值，True 表示该神经元接入输入层（S_in）。
        out_scope_mask: N 个布尔值，True 表示该神经元接出到读出头（S_out）。
        topo_index: N 个整数，拓扑（逐层）排序后的神经元 id。
        level_node_reach: K 个 ``(start, end)`` 半开区间，按 topo_index 分段。
        level_edge_reach: K 个 ``(start, end)`` 半开区间，表示第 k 层的入边。
        in_degree / out_degree: N 个整数的入度 / 出度。
        layer_z: K 个层参考平面的流向轴（z）坐标，取该层神经元位置的均值。
        layer_counts: K 个层的神经元数。
        layer_groups: K 个组的神经元 id 列表（分层着色分组）。
        layer_edge_counts: K 个层的入边数（第 1 层实测为 0）。
        degree_hist_in / degree_hist_out: 度 -> 神经元数 的直方图。
        conn_density: E / (N*N)，连接稀疏度。
        syn_dist_bytes: syn_dist 的字节数（仅记录，不读入内存、不嵌入 HTML）。
        fc: 两端全连接包裹的抽取结果（``config.fc_dim != 0`` 时非 None；
            无该键或 ``fc_dim == 0`` 时为 None，既有展示路径逐字节不变）。
    """

    checkpoint: str
    ckpt_stem: str
    config: dict[str, Any] = field(default_factory=dict)
    test_acc: float | None = None
    n_neurons: int = 0
    n_edges: int = 0
    neuron_pos: list[tuple[float, float, float]] = field(default_factory=list)
    edge_src: list[int] = field(default_factory=list)
    edge_dst: list[int] = field(default_factory=list)
    edge_weight: list[float] = field(default_factory=list)
    edge_dist: list[float] = field(default_factory=list)
    in_scope_mask: list[bool] = field(default_factory=list)
    out_scope_mask: list[bool] = field(default_factory=list)
    topo_index: list[int] = field(default_factory=list)
    level_node_reach: list[tuple[int, int]] = field(default_factory=list)
    level_edge_reach: list[tuple[int, int]] = field(default_factory=list)
    in_degree: list[int] = field(default_factory=list)
    out_degree: list[int] = field(default_factory=list)
    layer_z: list[float] = field(default_factory=list)
    layer_counts: list[int] = field(default_factory=list)
    layer_groups: list[list[int]] = field(default_factory=list)
    layer_edge_counts: list[int] = field(default_factory=list)
    degree_hist_in: dict[int, int] = field(default_factory=dict)
    degree_hist_out: dict[int, int] = field(default_factory=dict)
    conn_density: float = 0.0
    syn_dist_bytes: int = 0
    fc: FcData | None = None
    #: 抽取时的 ``model_state_dict`` 引用 —— **仅在** ``config.fc_dim != 0`` 时保留，
    #: 供 :func:`with_fc_top_k` 按新的 ``--fc-top-k`` 重抽抽样连线（无 FC 产物恒为
    #: None，因此不会长期持有任何张量，既有路径的内存行为完全不变）。
    fc_source: Mapping[str, Any] | None = None

    # -- 便捷视图 ---------------------------------------------------------
    @property
    def has_fc(self) -> bool:
        """是否展示两端全连接包裹（即 ``config.fc_dim != 0`` 且抽取成功）。"""
        return self.fc is not None

    @property
    def n_layers(self) -> int:
        """分层数 K（默认规模实测为 9）。"""
        return len(self.level_node_reach)

    @property
    def n_s_in(self) -> int:
        """S_in 神经元数（in_scope_mask 为真）。"""
        return sum(1 for v in self.in_scope_mask if v)

    @property
    def n_s_out(self) -> int:
        """S_out 神经元数（out_scope_mask 为真）。"""
        return sum(1 for v in self.out_scope_mask if v)

    def layer_of_neuron(self, neuron_id: int) -> int:
        """返回神经元所在的层号（0 基），越界返回 -1。"""
        for k, group in enumerate(self.layer_groups):
            if neuron_id in group:
                return k
        return -1

    def edge_threshold_stats(self, thresholds: Iterable[float]) -> dict[str, int]:
        """统计各阈值下 ``|edge_weight| >= threshold`` 的保留边数。

        Args:
            thresholds: 阈值序列（非负）。

        Returns:
            形如 ``{"0.30": 379}`` 的字典，键为阈值的 ``%.2f`` 文本。
        """
        abs_w = [abs(w) for w in self.edge_weight]
        return {
            f"{t:.2f}": sum(1 for w in abs_w if w >= t) for t in thresholds
        }

    def counts_at(self, threshold: float) -> int:
        """返回给定阈值下保留的边数。"""
        return sum(1 for w in self.edge_weight if abs(w) >= threshold)

    def weight_extremes(self) -> tuple[float, float, float, float]:
        """返回 ``|edge_weight|`` 的 (min, median, max, mean)。

        中位数口径（全模块唯一）：把 ``|edge_weight|`` 升序排列后取
        **上中位**（即下标 ``n // 2``）。偶数样本时不做两中位均值，
        也不使用 ``torch.median``（它返回下中位），以保证 README 与本函数输出一致。
        默认规模（E=736，seed=42）实测为 0.307065486907959。

        Returns:
            ``(min, median, max, mean)``；无边时返回四个 0.0。
        """
        abs_w = sorted(abs(w) for w in self.edge_weight)
        if not abs_w:
            return (0.0, 0.0, 0.0, 0.0)
        n = len(abs_w)
        median = abs_w[n // 2]  # 上中位
        return (abs_w[0], median, abs_w[-1], sum(abs_w) / n)


class _MissingKeys(list):
    """内部标记：checkpoint 缺失的二期键。"""


def load_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """加载 checkpoint 并返回原始字典。

    Args:
        path: checkpoint 文件路径。
        map_location: ``torch.load`` 的落盘设备，默认 CPU。

    Returns:
        checkpoint 的原始字典（含 ``model_state_dict`` / ``config`` 等顶层键）。

    Raises:
        CheckpointNotFoundError: 路径不存在或不是普通文件。
        CheckpointCorruptedError: 文件损坏、非 torch 序列化产物或读取失败。
    """
    ckpt_path = Path(path).expanduser()
    if not ckpt_path.exists():
        raise CheckpointNotFoundError(
            f"checkpoint 路径不存在：{ckpt_path}；"
            "请确认文件已训练产出，或使用 --checkpoint 指定正确路径。"
        )
    if not ckpt_path.is_file():
        raise CheckpointNotFoundError(
            f"checkpoint 路径不是文件：{ckpt_path}；请传入一个 .pt 训练产物。"
        )
    try:
        obj = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    except Exception as exc:  # noqa: BLE001 - 需要把底层异常翻译成可读错误
        raise CheckpointCorruptedError(
            f"checkpoint 文件损坏或无法反序列化：{ckpt_path}；"
            f"底层错误：{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(obj, Mapping):
        raise CheckpointCorruptedError(
            f"checkpoint 顶层不是字典：{ckpt_path}（得到 {type(obj).__name__}）。"
        )
    return dict(obj)


def get_state_dict(obj: Mapping[str, Any], path: str | Path = "") -> Mapping[str, Any]:
    """从 checkpoint 顶层字典中取出 ``model_state_dict``。

    Args:
        obj: ``load_checkpoint`` 的返回值。
        path: 仅用于错误信息的路径文本。

    Returns:
        状态字典（若无 ``model_state_dict`` 键则把顶层当作状态字典本身）。

    Raises:
        CheckpointSchemaError: 取出的对象不是映射。
    """
    sd = obj.get("model_state_dict", obj)
    if not isinstance(sd, Mapping):
        raise CheckpointSchemaError(
            f"checkpoint {path} 的 model_state_dict 不是映射（得到 {type(sd).__name__}）。"
        )
    return sd


def missing_required_keys(sd: Mapping[str, Any]) -> list[str]:
    """返回状态字典中缺失的二期必需键（按 REQUIRED_KEYS 顺序）。"""
    return [k for k in REQUIRED_KEYS if k not in sd]


def ensure_phase2_checkpoint(sd: Mapping[str, Any], path: str | Path) -> None:
    """校验状态字典含全部二期拓扑键。

    Args:
        sd: ``model_state_dict``。
        path: 路径文本，用于错误信息。

    Raises:
        CheckpointSchemaError: 存在缺失键（错误信息包含每个缺失键名）。
    """
    missing = missing_required_keys(sd)
    if missing:
        raise CheckpointSchemaError(
            "checkpoint 不是 N3D 二期产物，缺少二期拓扑必需键："
            + ", ".join(missing)
            + f"（文件：{path}；完整必需键列表："
            + ", ".join(REQUIRED_KEYS)
            + "）。"
            "该文件可能是一期（n3d_proto）产物，其仅有突触级 edge_index，"
            "不含神经元级连接与作用域掩码，无法可视化；请改用二期产物。"
        )


def _as_int_list(t: torch.Tensor) -> list[int]:
    """把整型张量转成 Python int 列表。"""
    return [int(v) for v in t.detach().to("cpu").reshape(-1).tolist()]


def _as_float_list(t: torch.Tensor) -> list[float]:
    """把浮点张量转成 Python float 列表（float32 -> float64 无损提升）。"""
    return [float(v) for v in t.detach().to("cpu").reshape(-1).tolist()]


def _as_bool_list(t: torch.Tensor) -> list[bool]:
    """把布尔/整型张量转成 Python bool 列表。"""
    return [bool(v) for v in t.detach().to("cpu").reshape(-1).tolist()]


def _as_pairs(t: torch.Tensor) -> list[tuple[int, int]]:
    """把 [K,2] 张量转成 ``(start, end)`` 区间列表。"""
    arr = t.detach().to("cpu")
    if arr.dim() != 2 or arr.shape[1] != 2:
        raise CheckpointSchemaError(
            f"期望 [K,2] 的分层区间张量，实际得到 {tuple(arr.shape)}。"
        )
    return [(int(a), int(b)) for a, b in arr.tolist()]


def _as_positions(t: torch.Tensor) -> list[tuple[float, float, float]]:
    """把 [N,3] 张量转成三元组列表。"""
    arr = t.detach().to("cpu")
    if arr.dim() != 2 or arr.shape[1] != 3:
        raise CheckpointSchemaError(
            f"neuron_pos 期望 [N,3] 形状，实际得到 {tuple(arr.shape)}。"
        )
    return [(float(x), float(y), float(z)) for x, y, z in arr.tolist()]


#: 报错时最多列出多少个越界项（阈值不变，仅供消息措辞与函数默认值共用）。
_OUT_OF_RANGE_LIMIT: int = 6


def _out_of_range(
    values: Sequence[int],
    low: int,
    high: int,
    limit: int = _OUT_OF_RANGE_LIMIT,
) -> list[tuple[int, int]]:
    """扫描序列中落在 ``[low, high)`` 之外的元素。

    Args:
        values: 待检查的整数序列。
        low: 合法下界（含）。
        high: 合法上界（不含）。
        limit: 最多返回多少个违规项（避免报错信息过长）。

    Returns:
        ``[(index, value), ...]``，最多 ``limit`` 项，按下标升序。
    """
    hits: list[tuple[int, int]] = []
    for i, v in enumerate(values):
        if not (low <= v < high):
            hits.append((i, int(v)))
            if len(hits) >= limit:
                break
    return hits


def _assert_in_range(
    name: str,
    values: Sequence[int],
    n: int,
) -> None:
    """校验索引序列全部落在 ``[0, n)``，否则抛可读的 :class:`CheckpointSchemaError`。

    为何必须在抽取阶段校验：

    * ``topo_index`` 含**负值**时，``pos[i]`` 会按 Python 负索引**静默**取到
      错误神经元，使 ``layer_groups`` / ``layer_z`` 静默错误——比崩溃更危险；
    * 含**越界值**时会抛 ``IndexError`` 而非 :class:`CheckpointSchemaError`，
      而 CLI 只捕获 :class:`CheckpointError`，于是损坏产物会以 traceback 崩溃、
      退出码 1，破坏「非二期产物给可读错误」的契约。

    Args:
        name: 张量名（用于报错）。
        values: 待校验的索引序列。
        n: 合法上界（神经元数）。

    Raises:
        CheckpointSchemaError: 存在越界元素（错误信息附越界值与其下标）。
    """
    hits = _out_of_range(values, 0, n)
    if hits:
        shown = "、".join(f"{name}[{i}]={v}" for i, v in hits)
        raise CheckpointSchemaError(
            f"checkpoint 索引越界：{name} 必须全部落在 [0, {n}) 内，"
            f"实测越界项（下标=值）：{shown}"
            # 注意：_out_of_range 在第 limit 项处就提前返回，此时 len(hits) 只是
            # 「已列出的项数」而非真实总数，故只能说「至少 N 项」。
            + (f" 等至少 {len(hits)} 项（仅列前 {len(hits)} 项）"
               if len(hits) >= _OUT_OF_RANGE_LIMIT else "")
            + "。"
        )


def _histogram(values: Sequence[int]) -> dict[int, int]:
    """统计非负整数序列的频次直方图。"""
    hist: dict[int, int] = {}
    for v in values:
        key = int(v)
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items()))


def extract_topology(
    sd: Mapping[str, Any],
    checkpoint: str | Path = "",
    config: Mapping[str, Any] | None = None,
    test_acc: float | None = None,
    fc_top_k: int = DEFAULT_FC_TOP_K,
) -> TopologyData:
    """把状态字典抽取成 :class:`TopologyData`（纯 Python 结构）。

    Args:
        sd: 二期 ``model_state_dict``（须先通过 :func:`ensure_phase2_checkpoint`）。
        checkpoint: 来源 checkpoint 路径（写进产物的溯源信息）。
        config: 顶层 config 字典。
        test_acc: 顶层 test_acc。
        fc_top_k: 两端全连接包裹的抽样口径 k；``config.fc_dim == 0`` 或无该键时
            完全不生效（既有产物逐字节不变）。

    Returns:
        TopologyData 实例（``config.fc_dim != 0`` 时含 :attr:`TopologyData.fc`）。

    Raises:
        CheckpointSchemaError: 键缺失、形状不符合契约，或
            ``topo_index`` / ``edge_src`` / ``edge_dst`` 存在越界索引；
            ``config.fc_dim != 0`` 但 FC 键缺失 / 形状不自洽 / 面板与云相交。
        ValueError: ``fc_top_k`` 越界。
    """
    ensure_phase2_checkpoint(sd, checkpoint)

    pos = _as_positions(sd["neuron_pos"])
    n = len(pos)
    src = _as_int_list(sd["edge_src"])
    dst = _as_int_list(sd["edge_dst"])
    n_edges = len(src)
    weight = _as_float_list(sd["edge_weight"])
    edist = _as_float_list(sd["edge_dist"])
    in_scope = _as_bool_list(sd["in_scope_mask"])
    out_scope = _as_bool_list(sd["out_scope_mask"])
    topo_index = _as_int_list(sd["topo_index"])
    level_nodes = _as_pairs(sd["level_node_reach"])
    level_edges = _as_pairs(sd["level_edge_reach"])
    in_deg = _as_int_list(sd["in_degree"])
    out_deg = _as_int_list(sd["out_degree"])

    # --- 形状契约校验（失败给出可读信息，而不是静默错图）---------------
    checks = (
        ("edge_dst", len(dst), n_edges),
        ("edge_weight", len(weight), n_edges),
        ("edge_dist", len(edist), n_edges),
        ("in_scope_mask", len(in_scope), n),
        ("out_scope_mask", len(out_scope), n),
        ("topo_index", len(topo_index), n),
        ("in_degree", len(in_deg), n),
        ("out_degree", len(out_deg), n),
    )
    for name, got, want in checks:
        if got != want:
            raise CheckpointSchemaError(
                f"checkpoint 形状不符合二期契约：{name} 长度为 {got}，期望 {want}"
                f"（N={n}, E={n_edges}）。"
            )

    # --- 取值域校验：索引必须全部 ∈ [0, N)。
    # 必须在用 topo_index 取 pos / 用 edge_src|edge_dst 取端点之前执行，
    # 否则负索引会静默取错神经元（不报错但图是错的）。
    _assert_in_range("topo_index", topo_index, n)
    _assert_in_range("edge_src", src, n)
    _assert_in_range("edge_dst", dst, n)

    # --- 分层：第 k 层 = level_node_reach[k] 给出的 topo_index 半开区间 ----
    layer_groups: list[list[int]] = []
    layer_counts: list[int] = []
    layer_z: list[float] = []
    for k, (start, end) in enumerate(level_nodes):
        if not (0 <= start <= end <= n):
            raise CheckpointSchemaError(
                f"level_node_reach[{k}] = ({start}, {end}) 越界，N={n}。"
            )
        ids = topo_index[start:end]
        layer_groups.append([int(i) for i in ids])
        layer_counts.append(len(ids))
        # 层参考平面取该层神经元沿流向轴（z）位置的**均值**：层本身就是这一组
        # 「同时计算的神经元」的定义，故该平面必然包含这一组神经元，与几何是否
        # 规整（晶格 / 任意曲面 / 非均匀分层）无关。不改成最小二乘拟合平面。
        zs = [pos[i][2] for i in ids]
        layer_z.append(float(sum(zs) / len(zs)) if zs else 0.0)

    layer_edge_counts = [int(b - a) for a, b in level_edges]

    data = TopologyData(
        checkpoint=str(checkpoint),
        ckpt_stem=Path(str(checkpoint)).stem if checkpoint else "checkpoint",
        config=dict(config or {}),
        test_acc=None if test_acc is None else float(test_acc),
        n_neurons=n,
        n_edges=n_edges,
        neuron_pos=pos,
        edge_src=src,
        edge_dst=dst,
        edge_weight=weight,
        edge_dist=edist,
        in_scope_mask=in_scope,
        out_scope_mask=out_scope,
        topo_index=topo_index,
        level_node_reach=level_nodes,
        level_edge_reach=level_edges,
        in_degree=in_deg,
        out_degree=out_deg,
        layer_z=layer_z,
        layer_counts=layer_counts,
        layer_groups=layer_groups,
        layer_edge_counts=layer_edge_counts,
        degree_hist_in=_histogram(in_deg),
        degree_hist_out=_histogram(out_deg),
        conn_density=(n_edges / float(n * n)) if n else 0.0,
        syn_dist_bytes=_syn_dist_bytes(sd),
    )
    # 两端全连接包裹：仅在 config.fc_dim != 0 时抽取（否则完全不触碰 FC 张量）。
    # 放在最后：此时 data 的 neuron_pos / 作用域掩码已抽取完毕，面板几何需要它们。
    data.fc = extract_fc(sd, data, config, top_k=fc_top_k)
    if data.fc is not None:
        # 保留状态字典引用，供 with_fc_top_k 重抽（无 FC 时不保留，内存行为不变）。
        data.fc_source = sd
    return data


def with_fc_top_k(data: TopologyData, top_k: int) -> TopologyData:
    """按新的 ``top_k`` 重新抽取 FC 抽样连线，返回**新的** :class:`TopologyData`。

    为什么需要它：抽样条数 ``top_k`` 是 CLI / GUI 参数，而 FC 抽取发生在
    :func:`load_topology` 阶段。渲染入口（:func:`write_outputs` /
    :func:`build_html_payload`）因此按「传入的 ``top_k``」在此重抽一次，
    使 ``--fc-top-k`` 真正影响产物（开关类参数不得是空操作）。

    若 ``top_k`` 与数据中已有的口径相同，则**原样返回入参对象**（不做任何拷贝，
    保证默认路径零开销、零行为变化）。

    Args:
        data: 已抽取的拓扑数据。
        top_k: 目标抽样条数 k。

    Returns:
        携带新抽样连线的 :class:`TopologyData`（无 FC 时原样返回入参）。

    Raises:
        ValueError: ``top_k`` 越界。
        CheckpointSchemaError: 重抽时发现状态字典缺失或自洽性被破坏。
    """
    k = validate_fc_top_k(top_k)
    if data.fc is None or int(data.fc.top_k) == k:
        return data
    sd = data.fc_source
    if not isinstance(sd, Mapping):
        raise CheckpointSchemaError(
            "需要按新的 fc_top_k 重新抽取全连接层连线，但该 TopologyData 未持有"
            "状态字典（fc_source）。请改用 core.load_topology(path, fc_top_k=k)。"
        )
    fresh = replace(data, fc=extract_fc(sd, data, data.config, top_k=k))
    return fresh


def _syn_dist_bytes(sd: Mapping[str, Any]) -> int:
    """记录 syn_dist 的字节数；**不**把它读入内存（约 16.8MB / 默认规模）。"""
    t = sd.get("syn_dist")
    if isinstance(t, torch.Tensor):
        return int(t.numel() * t.element_size())
    return 0


# ---------------------------------------------------------------------------
# 两端全连接包裹（``fc_dim != 0``）：触发判定 / 抽取 / 面板几何 / 抽样
# ---------------------------------------------------------------------------
#: ``fc_dim != 0`` 时**必须齐全**的状态字典键；缺任一即视为产物损坏（不得降级）。
FC_REQUIRED_KEYS: tuple[str, ...] = (
    "proj_weight",
    "fc_out_weight",
    "fc_in_weight",
    "fc_out_bias",
)


def _config_value(config: Mapping[str, Any] | None, key: str) -> Any:
    """从 config 字典中读取一个字段（缺失或 config 非映射时返回 None）。"""
    if not isinstance(config, Mapping):
        return None
    return config.get(key)


def is_fc_enabled(config: Mapping[str, Any] | None) -> bool:
    """判断产物是否启用「两端全连接包裹」（**只看** ``config.fc_dim != 0``）。

    三态判定（与 README / current_spec 的表格逐一对应）：

    * ``config`` 无 ``fc_dim`` 键（二期 / 三期未启用产物）→ 返回 False，走既有展示、不报错；
    * ``config.fc_dim == 0`` → 返回 False，走既有展示、不报错；
    * ``config.fc_dim != 0`` → 返回 True（此后若 FC 键缺失，由
      :func:`extract_fc` 报错退出，**不静默降级**为无 FC 展示）。

    注意：启用与否**不依赖「FC 键是否存在」**，否则「产物损坏」会被误判成「无 FC」。

    Args:
        config: 产物顶层 config 字典（可为 None）。

    Returns:
        是否启用 FC 展示。
    """
    raw = _config_value(config, "fc_dim")
    if raw is None:
        return False
    try:
        return int(raw) != 0
    except (TypeError, ValueError):
        return False


def _fc_hidden_width(
    sd: Mapping[str, Any],
    proj: torch.Tensor,
    fc_out: torch.Tensor,
    config: Mapping[str, Any] | None,
) -> int:
    """由 FC 张量的**实际形状**定出有效宽度 ``H``，并交叉校验各来源是否自洽。

    为何以形状为准：``fc_dim = -1``（宽度跟随 N）时 ``config`` 中不存在直接等于 ``H``
    的字段（默认预设的 ``config["hidden_dim"] = 2048`` 与实测 ``H = 825`` **不等**），
    因此 ``H`` 只能由 ``proj_weight`` 的列数 / ``fc_out_weight`` 的行数推出。

    Args:
        sd: 状态字典。
        proj: ``proj_weight``，期望 ``[|S_in|, H]``。
        fc_out: ``fc_out_weight``，期望 ``[H, |S_out|]``。
        config: 产物 config（用于交叉校验 ``fc_width`` / ``hidden_dim``）。

    Returns:
        有效宽度 ``H``。

    Raises:
        CheckpointSchemaError: 各来源给出的 ``H`` 不自洽，或 ``H <= 0``。
    """
    from_proj = int(proj.shape[1])
    from_out = int(fc_out.shape[0])
    candidates: list[tuple[str, int]] = [("proj_weight.shape[1]", from_proj),
                                         ("fc_out_weight.shape[0]", from_out)]
    fc_in = sd.get("fc_in_weight")
    if isinstance(fc_in, torch.Tensor) and fc_in.dim() == 2:
        candidates.append(("fc_in_weight.shape[0]", int(fc_in.shape[0])))
    declared = _config_value(config, "fc_width")
    if declared is not None:
        candidates.append(("config.fc_width", int(declared)))
    distinct = {v for _src, v in candidates}
    if len(distinct) != 1:
        raise CheckpointSchemaError(
            "checkpoint 的 fc_dim != 0，但全连接层宽度 H 的各来源不自洽："
            + "、".join(f"{src}={v}" for src, v in candidates)
            + "；请检查该产物是否被手工改写。"
        )
    h = from_proj
    if h <= 0:
        raise CheckpointSchemaError(f"checkpoint 的 fc_dim != 0 但有效宽度 H={h} <= 0。")
    return h


def _fc_forbidden_links(data: TopologyData, sd: Mapping[str, Any]) -> None:
    """校验**抽样视觉通道之外**的两端连接与产物字段自洽。

    覆盖两个真实风险（不通过即报错，避免静默画出与产物不符的图）：

    * ``head_weight`` 的列数必须 == ``H``（它是 (6) 线性输出层的输入宽度）；
    * ``out_scope_index``（若存在）必须与 ``out_scope_mask`` 下标的升序序列逐位相同，
      否则 (4) 索引收集的列序与 ``fc_out_weight`` 的列序会错位（图错而不报错）。

    Args:
        data: 已抽取的拓扑（提供 ``out_scope_mask``）。
        sd: 状态字典。

    Raises:
        CheckpointSchemaError: 上述任一不成立。
    """
    osi = sd.get("out_scope_index")
    if isinstance(osi, torch.Tensor):
        got = _as_int_list(osi)
        want = [i for i, flag in enumerate(data.out_scope_mask) if flag]
        if got != want:
            raise CheckpointSchemaError(
                "checkpoint 的 out_scope_index 与 out_scope_mask 不一致："
                f"out_scope_index 长度 {len(got)}、由掩码推出的 S_out 数 {len(want)}"
                + (f"、首个不同处 got={got[0]} want={want[0]}" if got and want and got[0] != want[0] else "")
                + "；列序错位会导致全连接读出连线画错。"
            )


def _grid_layout(n_units: int) -> tuple[int, int, int]:
    """把 ``n_units`` 个单元排成 ``ceil(sqrt(n))`` 列的网格。

    Args:
        n_units: 单元数（> 0）。

    Returns:
        ``(cols, rows, cols*rows)``；``rows`` 用整数上取整以保证 ``cols * rows >= n``。
    """
    cols = int(math.ceil(math.sqrt(n_units)))
    rows = -(-n_units // cols)  # 整数上取整
    return cols, rows, cols * rows


def _place_units_in_panel(
    n_units: int,
    flow: float,
    flow_idx: int,
    plane_axes: tuple[int, int],
    perp_lo: tuple[float, float],
    perp_hi: tuple[float, float],
    cell: float,
    norms: Sequence[float],
    tags: Sequence[str],
) -> tuple[list[PanelUnitGeometry], float, float, int, int]:
    """在**垂直于流向轴**的平面内，把 ``n_units`` 个单元按网格排布到三维坐标。

    原理：面板平面由流向轴之外的两个坐标轴张成（二维退化时用兜底轴）。网格取
    ``ceil(sqrt(H))`` 列，单元中心间距 ``cell`` 由调用方按「面板目标跨度 / 网格尺寸」
    给出，因此面板与神经元云**同尺度**（见 :data:`FC_PANEL_SPAN_RATIO`）。

    Args:
        n_units: 单元数 H。
        flow: 面板在流向轴上的中心坐标。
        flow_idx: 流向轴下标（0/1/2）。
        plane_axes: 面板平面内的两个坐标轴下标。
        perp_lo / perp_hi: 神经元云在两个平面轴上的最小 / 最大坐标（决定网格中心）。
        cell: 单元中心间距（> 0）。
        norms: 长度 H 的单元配色范数。
        tags: 长度 H 的单元角色标签。

    Returns:
        ``(units, cell_size, half_u, half_v, cols, rows)``：单元几何列表、单元中心间距、
        面板在平面两轴上的半跨度、网格列数与行数。
    """
    if cell <= 0.0:
        raise CheckpointSchemaError(
            f"无法为全连接层面板定出正的单元间距（H={n_units}）。"
        )
    cols, rows, _cells = _grid_layout(n_units)
    centers = [(perp_lo[a] + perp_hi[a]) / 2.0 for a in plane_axes]
    first = [(cols - 1) / 2.0, (rows - 1) / 2.0]
    units: list[PanelUnitGeometry] = []
    for unit in range(n_units):
        col = unit % cols
        row = unit // cols
        coords = [0.0, 0.0, 0.0]
        coords[flow_idx] = flow
        coords[plane_axes[0]] = centers[0] + (col - first[0]) * cell
        coords[plane_axes[1]] = centers[1] + (row - first[1]) * cell
        units.append(
            PanelUnitGeometry(
                unit=int(unit),
                pos=(coords[0], coords[1], coords[2]),
                norm=float(norms[unit]),
                tags=(str(tags[unit]),),
            )
        )
    # 面板在平面内的实际半跨度（用于边界块尺寸与渲染框）
    half_u = (cols - 1) / 2.0 * cell + cell / 2.0
    half_v = (rows - 1) / 2.0 * cell + cell / 2.0
    return units, cell, half_u, half_v, cols, rows


def _sample_top_k(
    weights: Sequence[float],
    k: int,
    side: str,
    unit_or_neuron: int,
) -> list[int]:
    """在一条权重向量内取 ``|w|`` 最大的 top-k 下标（**保持原下标升序**）。

    口径说明（与 README / meta 的声明一一对应）：对每个 S_in 神经元（或每个 S_out
    神经元）各取该行 / 该列内 ``|w|`` 最大的 k 条，因此**每个神经元都必然有连线**
    ——这是相对「全局阈值」的关键优势（实测 ``|w| >= 0.30`` 仅保留 1,308 条、
    ``>= 0.20`` 跳到 12,688 条，阈值极敏感且会静默丢掉整个神经元）。

    实现用 ``heapq.nlargest``（关键字为 ``|w|`` 与 ``-下标``）：返回结果按
    「``|w|`` 降序、下标升序」排列，最后再按下标升序排序。排序键完全确定，
    不含任何随机性或字典序依赖，因此产物可逐字节复现。

    Args:
        weights: 该行 / 该列的权重序列。
        k: 抽取条数（已在 :func:`validate_fc_top_k` 校验过范围）。
        side: ``"input"`` / ``"output"``（仅用于错误信息）。
        unit_or_neuron: 神经元 id（仅用于错误信息）。

    Returns:
        ``k`` 个下标，按升序排列；``k`` 大于可用元素数时返回全部下标（按口径截断）。

    Raises:
        CheckpointSchemaError: 权重序列为空。
    """
    import heapq

    n = len(weights)
    if n == 0:
        raise CheckpointSchemaError(
            f"全连接层抽样失败：{side} 侧神经元 #{unit_or_neuron} 对应的权重向量为空。"
        )
    take = min(int(k), n)
    picked = heapq.nlargest(take, range(n), key=lambda j: (abs(float(weights[j])), -j))
    return sorted(picked)


def validate_fc_top_k(k: int) -> int:
    """校验 ``--fc-top-k`` 的取值域 ``[MIN_FC_TOP_K, MAX_FC_TOP_K]``。

    **必须是整数，且不接受 bool 与 ``None``**：

    * 原实现写作 ``value = int(k)``，于是 ``2.5 -> 2``、``8.7 -> 8``、``True -> 1``
      都被**静默截断/强转**接受——与「非整数一律报错、不做静默截断」的契约相悖。
      虽经 CLI 调用时 argparse 的 ``type=int`` 已在入口拦下 ``--fc-top-k 1.9``，
      但 Python API 调用方传入 float / bool 时会被静默改写，故在此按**最严格**
      口径处理：**bool 一律拒绝**（``True``/``False`` 是 ``int`` 的子类，若不特判
      就会被当成 1/0），**非整数类型一律拒绝**（含 ``2.0`` 这类「看似整数的浮点」）。
    * ``None`` 抛 :class:`ValueError`（而不是裸 ``TypeError``），因为本函数的
      docstring 只声明 ``ValueError``，按契约只捕获 ``ValueError`` 的调用方
      （CLI / GUI）才接得住。

    Args:
        k: 待校验的 k（须为 ``int``，不含 ``bool``）。

    Returns:
        校验通过的整数 k。

    Raises:
        ValueError: 类型不是整数（含 ``None`` / ``bool`` / ``float``）或数值越界，
            错误信息可读（CLI / GUI 均复用同一条校验，不静默截断）。
    """
    if isinstance(k, bool):
        raise ValueError(
            f"fc_top_k 必须是整数，不接受布尔值（当前为 {k!r}）。"
        )
    if not isinstance(k, int):
        raise ValueError(
            f"fc_top_k 必须是整数，不接受 {type(k).__name__} 类型（当前为 {k!r}）；"
            "非整数一律报错，不做静默截断。"
        )
    if not (MIN_FC_TOP_K <= k <= MAX_FC_TOP_K):
        raise ValueError(
            f"fc_top_k 必须在 [{MIN_FC_TOP_K}, {MAX_FC_TOP_K}] 区间内，当前为 {k}；"
            "越界值一律报错，不做静默截断。"
        )
    return k


def _fc_panel_geometry(
    pos: Sequence[tuple[float, float, float]],
    norms_in: Sequence[float],
    norms_out: Sequence[float],
    h: int,
    axis: str,
) -> tuple[dict[str, FcPanelGeometry], list[BoundaryBlock], float, float]:
    """构造两片全连接层面板 + 输入 / 输出边界块，并**硬断言面板与云不相交**。

    几何口径（与 README「几何与不重叠判据」一节一一对应）::

        [输入边界块 784] ──▶ [H 单元面板·输入侧] ──(抽样连线)──▶ S_in 神经元
                                                               │ N3D 核心（既有展示不变）
        [输出边界块 10] ◀── [H 单元面板·输出侧] ◀──(抽样连线)── S_out 神经元

    * 两片面板**垂直于流向轴**，分别置于神经元云流向轴跨度 ``[lo, hi]`` 的**两端外侧**；
    * 间隙 = 云跨度 × :data:`FC_PANEL_GAP_RATIO`；面板自身在流向轴上的厚度 =
      单元中心间距 × :data:`FC_PANEL_THICKNESS_RATIO`，夹在
      ``[FC_PANEL_MIN_THICKNESS, 云跨度 × FC_PANEL_MAX_THICKNESS_RATIO]`` 之间；
    * **不重叠判据（硬断言）**：面板节点的流向轴坐标区间 ∩ 神经元云流向轴区间 = ∅，
      即 ``panel_in.flow_interval[1] < lo`` 且 ``hi < panel_out.flow_interval[0]``；
      断言在**构造期**执行，几何参数一旦被改坏会立即抛错而不是画出一张重叠的图。

    **入口校验（先于任何算术）**：``H`` 必须是**正整数**。原实现在此之前就会先算
    ``max(cols, rows)`` / ``math.sqrt(H)``，于是 ``H = 0`` 抛裸 ``ZeroDivisionError``、
    ``H = -1`` 抛裸 ``ValueError: math domain error`` —— 都是**未包装的逃逸异常**，
    与「几何参数退化时给可读的 :class:`CheckpointSchemaError`」的契约相悖。
    （主链路不受影响：:func:`extract_fc` 经 :func:`_fc_hidden_width` 已用 ``H <= 0``
    兜住；此处把**直接调用**该函数的路径也补齐。）

    Args:
        pos: N 个神经元坐标。
        norms_in: 长度 H 的输入侧单元配色范数。
        norms_out: 长度 H 的输出侧单元配色范数。
        h: 有效宽度 H（必须是正整数）。
        axis: 流向轴名。

    Returns:
        ``(panels, blocks, cloud_lo, cloud_hi)``。

    Raises:
        CheckpointSchemaError: ``H`` 不是正整数、流向轴非法、云流向轴跨度为 0
            （或两轴跨度均为 0 且流向轴跨度也为 0，无法定出正的单元间距），
            以及面板与神经元云在流向轴上相交。
    """
    # ---- 入口校验：必须早于任何算术（否则 H<=0 会抛出未包装的裸异常）----
    if isinstance(h, bool) or not isinstance(h, int):
        raise CheckpointSchemaError(
            f"全连接层面板要求宽度 H 为整数，实际得到 {type(h).__name__}（{h!r}）。"
        )
    if h <= 0:
        raise CheckpointSchemaError(
            f"全连接层面板要求宽度 H > 0，实际得到 H={h}。"
        )
    flow_idx = FLOW_AXIS_INDEX.get(axis)
    if flow_idx is None:
        raise CheckpointSchemaError(
            f"未知的流向轴 '{axis}'；合法取值为 {sorted(FLOW_AXIS_INDEX)}。"
        )
    axes = [a for a in (0, 1, 2) if a != flow_idx]
    perp = [c for c in pos]
    flow_vals = [p[flow_idx] for p in perp]
    cloud_lo, cloud_hi = float(min(flow_vals)), float(max(flow_vals))
    span = cloud_hi - cloud_lo
    if span <= 0.0:
        raise CheckpointSchemaError(
            f"神经元云在流向轴 '{axis}' 上的跨度为 0（实测 {cloud_lo}），"
            "无法在两端外侧放置互不重叠的全连接层面板。"
        )
    gap = span * FC_PANEL_GAP_RATIO
    perp_lo = tuple(min(p[a] for p in perp) for a in axes)
    perp_hi = tuple(max(p[a] for p in perp) for a in axes)
    # 单元中心间距：让面板在平面内的跨度 ≈ 云在对应轴上的跨度 × FC_PANEL_SPAN_RATIO。
    # 取两轴跨度的均值再除以网格边长，保证任一轴退化（平面内只有一个唯一坐标，跨度 0）
    # 时仍得到正的间距。
    perp_spans = [max(perp_hi[a] - perp_lo[a], 0.0) for a in axes]
    cols_pre, rows_pre, _ = _grid_layout(h)
    grid_side = float(max(cols_pre, rows_pre))
    cell = (perp_spans[0] + perp_spans[1]) / 2.0 * FC_PANEL_SPAN_RATIO / grid_side
    if cell <= 0.0:
        # 平面内两轴跨度都是 0（所有神经元共线于流向轴）时的兜底：用流向轴跨度定标。
        cell = span * FC_PANEL_SPAN_RATIO / grid_side
    thickness_cap = span * FC_PANEL_MAX_THICKNESS_RATIO

    panels: dict[str, FcPanelGeometry] = {}
    halves: dict[str, tuple[float, float]] = {}
    for name, center_flow, norm_list in (
        ("input", cloud_lo - gap, norms_in),
        ("output", cloud_hi + gap, norms_out),
    ):
        units, cell_used, half_u, half_v, cols, rows = _place_units_in_panel(
            h, center_flow, flow_idx, (axes[0], axes[1]), perp_lo, perp_hi, cell,
            norm_list, [name] * h,
        )
        # 厚度：单元间距 × 比例，夹在 [下界, 云跨度 × 上界比例] 之间
        thickness = min(max(cell_used * FC_PANEL_THICKNESS_RATIO, FC_PANEL_MIN_THICKNESS),
                        thickness_cap)
        interval = (center_flow - thickness / 2.0, center_flow + thickness / 2.0)
        for u in units:
            u.tags = (name,)
        panels[name] = FcPanelGeometry(
            name=name, axis=axis, flow=float(center_flow), thickness=float(thickness),
            flow_interval=(float(interval[0]), float(interval[1])),
            cols=int(cols), rows=int(rows), cell_size=float(cell), units=units,
        )
        halves[name] = (float(half_u), float(half_v))

    # ---- 不重叠硬断言（构造期，不留给渲染阶段）------------------------
    in_hi = panels["input"].flow_interval[1]
    out_lo = panels["output"].flow_interval[0]
    if not (in_hi < cloud_lo and cloud_hi < out_lo):
        raise CheckpointSchemaError(
            "全连接层面板与神经元云在流向轴上相交："
            f"输入面板区间 {panels['input'].flow_interval}、云区间 ({cloud_lo}, {cloud_hi})、"
            f"输出面板区间 {panels['output'].flow_interval}；"
            "要求「面板节点的流向轴坐标区间 ∩ 神经元云流向轴区间 = 空集」。"
        )

    # ---- 输入 / 输出边界块：置于**面板外侧**，以聚合箭头与面板相连 ------
    input_dim = 784
    output_dim = 10
    flow_size = span * FC_BLOCK_FLOW_RATIO
    offset = span * FC_BLOCK_OFFSET_RATIO
    blocks: list[BoundaryBlock] = []
    for name, half, dim, label, direction in (
        ("input", halves["input"], input_dim, f"输入 {input_dim}", -1.0),
        ("output", halves["output"], output_dim, f"输出 {output_dim}", +1.0),
    ):
        edge = (cloud_lo - gap) if direction < 0 else (cloud_hi + gap)
        surface = edge + direction * (panels[name].thickness / 2.0)
        center_flow = surface + direction * (offset + flow_size / 2.0)
        center = [0.0, 0.0, 0.0]
        center[flow_idx] = float(center_flow)
        center[axes[0]] = (perp_lo[0] + perp_hi[0]) / 2.0
        center[axes[1]] = (perp_lo[1] + perp_hi[1]) / 2.0
        size = [0.0, 0.0, 0.0]
        size[flow_idx] = float(flow_size)
        size[axes[0]] = float(half[0] * 2.0 * FC_BLOCK_LATERAL_RATIO)
        size[axes[1]] = float(half[1] * 2.0 * FC_BLOCK_LATERAL_RATIO)
        blocks.append(
            BoundaryBlock(
                name=name, label=label, dim=int(dim),
                center=(center[0], center[1], center[2]),
                size=(size[0], size[1], size[2]),
            )
        )
    return panels, blocks, cloud_lo, cloud_hi


def _fc_sample_edges(
    proj: torch.Tensor,
    fc_out: torch.Tensor,
    s_in_order: Sequence[int],
    s_out_order: Sequence[int],
    top_k: int,
) -> list[tuple[str, int, int, float]]:
    """按口径抽取两侧的 top-k 连线（``(侧别, 单元下标, 神经元 id, 权重)``）。

    * **输入侧**：对每个 S_in 神经元，取 ``proj_weight`` 该**行**内 ``|w|`` 最大的
      前 ``k`` 个单元 — ``(side="input", unit=j, neuron=S_in[i], w=proj[i, j])``；
    * **输出侧**：对每个 S_out 神经元，取 ``fc_out_weight`` 该**列**内 ``|w|`` 最大的
      前 ``k`` 个单元 — ``(side="output", unit=j, neuron=S_out[c], w=fc_out[j, c])``。

    逐行 / 逐列处理（不 materialize dense 中间矩阵），因此内存占用为 O(H)。

    Args:
        proj: ``proj_weight [|S_in|, H]``。
        fc_out: ``fc_out_weight [H, |S_out|]``。
        s_in_order: 参与投影的神经元 id（按 id 升序，对应 ``proj`` 的行序）。
        s_out_order: 参与读出的神经元 id（按 id 升序，对应 ``fc_out`` 的列序）。
        top_k: 抽样条数 k。

    Returns:
        抽样连线列表；顺序为「先全部输入侧、再全部输出侧」，
        每侧内部按（神经元在序列中的位置，单元下标）升序 —— 完全确定，可逐字节复现。
    """
    edges: list[tuple[str, int, int, float]] = []
    proj_rows = proj.detach().to("cpu")
    for i, neuron in enumerate(s_in_order):
        row = [float(v) for v in proj_rows[i].tolist()]
        for j in _sample_top_k(row, top_k, "input", int(neuron)):
            edges.append(("input", int(j), int(neuron), float(row[j])))
    out_cols = fc_out.detach().to("cpu")
    for c, neuron in enumerate(s_out_order):
        col = [float(v) for v in out_cols[:, c].tolist()]
        for j in _sample_top_k(col, top_k, "output", int(neuron)):
            edges.append(("output", int(j), int(neuron), float(col[j])))
    return edges


def extract_fc(
    sd: Mapping[str, Any],
    data: TopologyData,
    config: Mapping[str, Any] | None,
    top_k: int = DEFAULT_FC_TOP_K,
) -> FcData | None:
    """抽取「两端全连接包裹」（仅在 ``config.fc_dim != 0`` 时执行）。

    三态行为（**不得静默降级**）：

    * ``config`` 无 ``fc_dim`` 键 / ``fc_dim == 0`` → 直接返回 ``None``，
      **完全不触碰任何 FC 张量**（既有展示路径零回归）；
    * ``fc_dim != 0`` 且 FC 键齐全 → 抽取面板 / 边界块 / 抽样连线；
    * ``fc_dim != 0`` 但缺任一 FC 键 → 抛 :class:`CheckpointSchemaError`
      （报错退出非 0），而不是当作「无 FC」继续画一张不完整的图。

    内存口径：以 ``proj_weight``（默认规模 582×825 ≈ 1.9MB）与
    ``fc_out_weight``（825×588 ≈ 1.9MB）为工作集，两者**用完即弃**；
    ``fc_in_weight``（默认 825×784 ≈ 2.6MB）**只读形状与范数**、不长期持有。
    ``syn_dist``（约 16.8MB）始终不进入负载。

    Args:
        sd: 状态字典。
        data: 已抽取的拓扑（提供 ``neuron_pos`` / ``in_scope_mask`` / ``out_scope_mask``）。
        config: 产物顶层 config 字典。
        top_k: 抽样条数 k（须落在 ``[MIN_FC_TOP_K, MAX_FC_TOP_K]``）。

    Returns:
        :class:`FcData`，或在未启用 FC 时返回 ``None``。

    Raises:
        CheckpointSchemaError: 缺少 FC 键、形状不符合契约、``S_in`` / ``S_out`` 为空，
            或面板与神经元云在流向轴上相交。
        ValueError: ``top_k`` 类型/取值非法（须为 ``[1, 8]`` 内的整数）。
    """
    if not is_fc_enabled(config):
        return None
    # 注意：validate_fc_top_k 抛 ValueError（类型或取值非法），与 CLI/GUI 同一契约。
    k = validate_fc_top_k(top_k)
    missing = [name for name in FC_REQUIRED_KEYS if name not in sd]
    if missing:
        raise CheckpointSchemaError(
            "checkpoint 的 config.fc_dim != 0（启用两端全连接包裹），但缺少 FC 必需键："
            + ", ".join(missing)
            + f"；该产物不完整/已损坏，无法正确展示全连接层（不静默降级为无 FC 展示）。"
            f"完整 FC 键列表：{', '.join(FC_REQUIRED_KEYS)}"
        )
    for name in FC_REQUIRED_KEYS:
        if not isinstance(sd[name], torch.Tensor):
            raise CheckpointSchemaError(
                f"checkpoint 的 {name} 不是张量（得到 {type(sd[name]).__name__}）。"
            )
    proj = sd["proj_weight"]
    fc_out = sd["fc_out_weight"]
    if proj.dim() != 2 or fc_out.dim() != 2:
        raise CheckpointSchemaError(
            "checkpoint 的 FC 权重形状不符合契约：proj_weight 与 fc_out_weight 均须为二维，"
            f"实测 {tuple(proj.shape)} / {tuple(fc_out.shape)}。"
        )
    h = _fc_hidden_width(sd, proj, fc_out, config)
    s_in_order = [i for i, flag in enumerate(data.in_scope_mask) if flag]
    s_out_order = [i for i, flag in enumerate(data.out_scope_mask) if flag]
    if not s_in_order or not s_out_order:
        raise CheckpointSchemaError(
            "checkpoint 的 fc_dim != 0 要求 S_in 与 S_out 均非空，"
            f"实测 |S_in|={len(s_in_order)}、|S_out|={len(s_out_order)}；"
            "空的一侧无法展示「投影到 S_in / 从 S_out 收集」两条通道。"
        )
    if int(proj.shape[0]) != len(s_in_order) or int(fc_out.shape[1]) != len(s_out_order):
        raise CheckpointSchemaError(
            "checkpoint 的 FC 权重形状与作用域掩码不一致："
            f"proj_weight {tuple(proj.shape)} 期望行数 {len(s_in_order)}(=|S_in|)、"
            f"fc_out_weight {tuple(fc_out.shape)} 期望列数 {len(s_out_order)}(=|S_out|)。"
        )
    _fc_forbidden_links(data, sd)

    fc_in = sd["fc_in_weight"]
    input_dim = int(fc_in.shape[1]) if fc_in.dim() == 2 else 0
    output_dim = int(sd["head_weight"].shape[0]) if isinstance(sd.get("head_weight"), torch.Tensor) \
        and sd["head_weight"].dim() == 2 else 10
    if isinstance(sd.get("head_weight"), torch.Tensor) and sd["head_weight"].dim() == 2:
        if int(sd["head_weight"].shape[1]) != h:
            raise CheckpointSchemaError(
                f"checkpoint 的 head_weight {tuple(sd['head_weight'].shape)} 列数 != H={h}。"
            )

    # 单元配色范数（输入侧用 fc_in_weight 的行范数、输出侧用 fc_out_weight 的行范数）
    fc_in_cpu = fc_in.detach().to("cpu")
    in_unit_norms = [float(v) for v in fc_in_cpu.norm(dim=1).tolist()]
    out_unit_norms = [float(v) for v in fc_out.detach().to("cpu").norm(dim=1).tolist()]

    axis = str(_config_value(config, "flow_axis") or DEFAULT_FLOW_AXIS)
    panels, blocks, _lo, _hi = _fc_panel_geometry(
        data.neuron_pos, in_unit_norms, out_unit_norms, h, axis
    )
    edges = _fc_sample_edges(proj, fc_out, s_in_order, s_out_order, k)

    fc_dim_raw = _config_value(config, "fc_dim")
    proj_cpu = proj.detach().to("cpu")
    fco_cpu = fc_out.detach().to("cpu")
    result = FcData(
        fc_dim=int(fc_dim_raw),
        fc_width=int(h),
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        proj_weight_shape=(int(proj.shape[0]), int(proj.shape[1])),
        fc_out_weight_shape=(int(fc_out.shape[0]), int(fc_out.shape[1])),
        proj_count=int(proj.numel()),
        fc_out_count=int(fc_out.numel()),
        proj_abs_mean=float(proj_cpu.abs().mean().item()),
        fc_out_abs_mean=float(fco_cpu.abs().mean().item()),
        fc_in_weight_shape=(int(fc_in.shape[0]), int(fc_in.shape[1])) if fc_in.dim() == 2 else (0, 0),
        fc_in_count=int(fc_in.numel()),
        in_unit_norms=in_unit_norms,
        out_unit_norms=out_unit_norms,
        s_in_order=s_in_order,
        s_out_order=s_out_order,
        panels=panels,
        blocks=blocks,
        edges=edges,
        top_k=int(k),
    )
    # 自洽断言：抽样条数必须恰好等于「两侧神经元数 × k」（口径的算术后果）。
    if result.n_edges != result.expected_sample_edges:
        raise CheckpointSchemaError(
            f"全连接层抽样条数不自洽：实测 {result.n_edges} != 期望 "
            f"(|S_in|({len(s_in_order)}) + |S_out|({len(s_out_order)})) × k({k}) "
            f"= {result.expected_sample_edges}。"
        )
    if result.n_panel_points != 2 * h:
        raise CheckpointSchemaError(
            f"全连接层面板点数不自洽：实测 {result.n_panel_points} != 2 × H({h})。"
        )
    return result


def load_topology(path: str | Path, fc_top_k: int = DEFAULT_FC_TOP_K) -> TopologyData:
    """一步完成「加载 -> 键校验 -> 抽取」，返回 :class:`TopologyData`。

    Args:
        path: checkpoint 路径。
        fc_top_k: 两端全连接包裹的抽样口径 k（``config.fc_dim == 0`` 或无该键时无效）。

    Raises:
        CheckpointNotFoundError / CheckpointCorruptedError / CheckpointSchemaError
    """
    obj = load_checkpoint(path)
    sd = get_state_dict(obj, path)
    return extract_topology(
        sd,
        checkpoint=Path(str(path)),
        config=obj.get("config") or {},
        test_acc=obj.get("test_acc"),
        fc_top_k=fc_top_k,
    )


# ---------------------------------------------------------------------------
# 产物命名派生
# ---------------------------------------------------------------------------
def ckpt_stem(path: str | Path) -> str:
    """返回 checkpoint 文件名（不含扩展名），用于派生产物名。"""
    return Path(str(path)).stem


def derive_output_names(path: str | Path) -> dict[str, str]:
    """由 checkpoint 文件名派生三件套文件名。

    规则：``<ckpt名>.pt`` -> ``viz_<ckpt名>.html`` / ``.ply`` / ``.obj``，
    天然不撞名；禁止固定同名互相覆盖。

    Args:
        path: checkpoint 路径。

    Returns:
        ``{"stem":..., "html":..., "ply":..., "obj":...}``
    """
    stem = ckpt_stem(path)
    return {
        "stem": stem,
        "html": f"viz_{stem}.html",
        "ply": f"viz_{stem}.ply",
        "obj": f"viz_{stem}.obj",
    }


def resolve_output_paths(
    path: str | Path,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    out: str | Path | None = None,
) -> dict[str, Path]:
    """解析出三件套的最终输出路径。

    Args:
        path: checkpoint 路径。
        out_dir: 输出目录（不存在时由写出阶段创建）。
        out: 显式指定的 HTML 路径；给出时 PLY / OBJ 与该文件同目录同名。

    Returns:
        ``{"html":Path, "ply":Path, "obj":Path}``
    """
    if out is not None:
        html = Path(out).expanduser()
        parent = html.parent if str(html.parent) else Path(out_dir)
        stem = html.stem
        return {
            "html": html,
            "ply": parent / f"{stem}.ply",
            "obj": parent / f"{stem}.obj",
        }
    names = derive_output_names(path)
    base = Path(out_dir).expanduser()
    return {
        "html": base / names["html"],
        "ply": base / names["ply"],
        "obj": base / names["obj"],
    }


# ---------------------------------------------------------------------------
# 层配色函数（HTML 负载与 PLY 顶点色共用，保证两处层色同源）
# ---------------------------------------------------------------------------
#: 合法十六进制字符集（用于 `hex_to_rgb` 的严格校验）。
_HEX_DIGITS: frozenset[str] = frozenset("0123456789abcdefABCDEF")


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """把 ``#rrggbb`` 形式的颜色转成 ``(r, g, b)`` 字节三元组。

    Args:
        color: 形如 ``"#4e8cff"`` 的颜色文本（前导 ``#`` 可有可无、至多一个；两侧空白被忽略）。

    Returns:
        ``(r, g, b)``，各分量取值 0..255。

    Raises:
        ValueError: 文本不是恰好 6 位十六进制（含多余前导 ``#``、前导正负号、内部空白等）。
    """
    text = color.strip()
    # 只剥离**一个**前导 '#'：str.lstrip('#') 是字符集合语义，会把 "##4e8cff"
    # 这类多余井号一并吞掉，使非法文本被静默接受。
    if text.startswith("#"):
        text = text[1:]
    # 必须逐字符校验：`int(seg, 16)` 会接受前导正负号，"#-12345" 这类输入
    # 会静默产生负分量，与「各分量 0..255」的返回值契约相悖。
    if len(text) != 6 or any(ch not in _HEX_DIGITS for ch in text):
        raise ValueError(f"颜色文本必须是 6 位十六进制：{color!r}")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


def _hsv_to_rgb_bytes(hue: float, sat: float, val: float) -> tuple[int, int, int]:
    """HSV -> RGB 字节三元组（用标准库 :mod:`colorsys`，零第三方依赖）。"""
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, sat, val)
    return (
        max(0, min(255, int(round(r * 255.0)))),
        max(0, min(255, int(round(g * 255.0)))),
        max(0, min(255, int(round(b * 255.0)))),
    )


def _hue_palette_rgb(n_layers: int) -> list[tuple[int, int, int]]:
    """按均匀色相生成 ``n_layers`` 个两两不同的 RGB 颜色（K > 9 时使用）。

    原理：第 i 个颜色的色相取 ``i / n_layers``（均匀铺满整圈色相环），
    饱和度与明度固定在可读区间。由于 HSV 量化到 8 位后理论上存在撞色可能，
    这里再做一步**确定性搜索**：若候选色已被占用，就沿「色相微移 + 明度微降」
    的固定序列取下一个候选，直到得到未用颜色。搜索上限
    :data:`_PALETTE_SEARCH_LIMIT` 保证必然终止（超出则抛 ``ValueError``）。

    **撞色回退分支的覆盖口径（重要）**：在断言覆盖的 ``K <= 64`` 范围内，
    纯色相扩展**实测撞色数为 0**，即 `candidate not in used` 恒为真、
    ``step`` 永远停在 0 —— 该分支在这一区间内**不可达**。为使它不落入
    「改了也没人发现」的盲区，`verify_viz.py` 的
    ``[2a] _hue_palette_rgb 撞色回退分支（必撞色构造）`` 断言用
    ``K = 1536``（色相间隔 1/1536，量化到 8 位后必然撞色）**主动进入**该分支，
    并以调用计数证明回退确实发生、且最终去重数仍 == K。

    Args:
        n_layers: 需要的颜色数（> 0）。

    Returns:
        长度为 ``n_layers`` 的 RGB 列表，**两两不同**（去重数 == n_layers）。

    Raises:
        ValueError: 当 ``_PALETTE_SEARCH_LIMIT`` 次候选尝试都被占用时
            （``K <= 1536`` 实测不会发生）。
    """
    used: set[tuple[int, int, int]] = set()
    palette: list[tuple[int, int, int]] = []
    for i in range(n_layers):
        base_hue = (i / float(n_layers)) % 1.0
        picked: tuple[int, int, int] | None = None
        for step in range(_PALETTE_SEARCH_LIMIT):
            hue = (base_hue + step * _PALETTE_HUE_STEP) % 1.0
            val = _PALETTE_VALUE - _PALETTE_VALUE_STEP * (step % _PALETTE_VALUE_LEVELS)
            candidate = _hsv_to_rgb_bytes(hue, _PALETTE_SAT, val)
            if candidate not in used:
                picked = candidate
                break
        if picked is None:  # pragma: no cover - K <= 1536 实测不触发（见 docstring 覆盖口径）
            raise ValueError(f"无法为第 {i} 个层生成未占用的颜色（K={n_layers}）。")
        used.add(picked)
        palette.append(picked)
    return palette


def layer_palette_hex(n_layers: int) -> list[str]:
    """返回 ``n_layers`` 个层色（``#rrggbb``），供 HTML 负载使用。

    Args:
        n_layers: 层数 K（0 返回空列表）。

    Returns:
        K 个 hex 颜色；``K <= 9`` 时为 :data:`LEVEL_PALETTE_BASE` 的前 K 个
        （逐字节不变），``K > 9`` 时为均匀色相扩展色板，去重数 == K。
    """
    if n_layers <= 0:
        return []
    if n_layers <= len(LEVEL_PALETTE_BASE):
        return list(LEVEL_PALETTE_BASE[:n_layers])
    return ["#%02x%02x%02x" % rgb for rgb in _hue_palette_rgb(n_layers)]


def layer_palette_rgb(n_layers: int) -> list[tuple[int, int, int]]:
    """返回 ``n_layers`` 个层色的 RGB 字节三元组，供 PLY 顶点色使用。

    与 :func:`layer_palette_hex` **同源**（同一张基础色表 / 同一套色相扩展），
    因此 HTML 渲染器的层色与 PLY 顶点色逐位一致。
    """
    return [hex_to_rgb(c) for c in layer_palette_hex(n_layers)]


# ---------------------------------------------------------------------------
# HTML 内联数据负载
# ---------------------------------------------------------------------------
def _round_floats(values: Iterable[float], digits: int | None) -> list[float]:
    """按位数截断浮点序列。

    Args:
        values: 待处理的浮点序列。
        digits: 小数位；``None`` 表示不截断（只做 float 转换）。
            注意：本函数的 ``None`` 语义与
            :func:`build_html_payload` 的 ``None``（表示取默认位宽）**不同**。

    Returns:
        截断后的 Python ``float`` 列表。
    """
    if digits is None:
        return [float(v) for v in values]
    return [round(float(v), digits) for v in values]


def build_html_payload(
    data: TopologyData,
    threshold: float = DEFAULT_THRESHOLD,
    pos_digits: int | None = None,
    weight_digits: int | None = None,
    include_planes: bool = True,
    fc_top_k: int = DEFAULT_FC_TOP_K,
) -> dict[str, Any]:
    """构造内联进 HTML 的 JSON 负载。

    约定：``syn_dist [2048,2048]`` 约 16.8MB，**严禁**出现在本负载中；
    坐标与权重**默认按 6 / 8 位小数保留**（见 :data:`DEFAULT_POS_DIGITS` /
    :data:`DEFAULT_WEIGHT_DIGITS`），不是不截断；该默认精度下内联坐标与
    ``neuron_pos`` 的实测 ``max|diff| = 4.510e-07 < 1e-6``，因此仍能满足
    「产物坐标与 neuron_pos 逐位一致（容差 1e-6）」的可断言性；
    若需更高保留精度，可显式传入更大的 ``pos_digits`` / ``weight_digits``。

    FC 段（``"fc"``）**只在** ``config.fc_dim != 0`` 时出现：无 FC 产物的负载
    键集、键序与取值与改动前逐字节相同（零回归锚点）。

    Args:
        data: 拓扑数据。
        threshold: 初始边权重阈值。
        pos_digits: 坐标保留的小数位；``None`` 表示取默认值
            :data:`DEFAULT_POS_DIGITS`（= 6）。给定更大位数可提高保留精度。
        weight_digits: 权重保留的小数位；``None`` 表示取默认值
            :data:`DEFAULT_WEIGHT_DIGITS`（= 8）。
        include_planes: 层参考平面的初始开关状态。
        fc_top_k: 两端全连接包裹的抽样口径 k；与数据中已有口径不同时按
            :func:`with_fc_top_k` 重抽（无 FC 产物完全无影响）。

    Returns:
        可直接 ``json.dumps`` 的字典。
    """
    # None 统一解析为默认位宽，避免同一个 None 在不同函数里表示不同含义
    pos_digits_eff = DEFAULT_POS_DIGITS if pos_digits is None else int(pos_digits)
    weight_digits_eff = DEFAULT_WEIGHT_DIGITS if weight_digits is None else int(weight_digits)
    data = with_fc_top_k(data, fc_top_k)
    # 层色由共用色板函数给出：K <= 9 取既有 9 色前 K 个（逐字节不变），
    # K > 9 按均匀色相扩展，保证 K 个层色两两不同（否则分层着色失去可分辨性）。
    layer_colors = layer_palette_hex(data.n_layers)
    # 两端全连接包裹：仅 fc_dim != 0 时非 None（此时负载才含 "fc" 段）。
    fc_data = data.fc
    layers = []
    for k, (start, end) in enumerate(data.level_node_reach):
        layers.append(
            {
                "level": k,
                "start": int(start),
                "end": int(end),
                "count": int(data.layer_counts[k]),
                "z": float(data.layer_z[k]),
                "in_edges": int(data.layer_edge_counts[k]),
                "color": layer_colors[k],
            }
        )

    meta = {
        "checkpoint": data.checkpoint,
        "ckpt_stem": data.ckpt_stem,
        "seed": data.config.get("seed"),
        "N": data.n_neurons,
        "E": data.n_edges,
        "y_in": data.config.get("y_in"),
        "y_out": data.config.get("y_out"),
        "H": data.config.get("H"),
        "D": data.config.get("D"),
        "flow_axis": data.config.get("flow_axis"),
        "placement": data.config.get("placement"),
        "test_acc": data.test_acc,
        "n_layers": data.n_layers,
        "layer_counts": list(data.layer_counts),
        "layer_edge_counts": list(data.layer_edge_counts),
        "n_s_in": data.n_s_in,
        "n_s_out": data.n_s_out,
        "conn_density": data.conn_density,
        "excluded_matrix_bytes": data.syn_dist_bytes,
        "threshold": float(threshold),
        "threshold_counts": data.edge_threshold_stats((0.05, 0.10, 0.20, 0.30, 0.50)),
        # 层平面初始开关：必须写进负载，渲染器据此初始化复选框与绘制状态。
        "showPlanes": bool(include_planes),
    }
    # ---- 两端全连接包裹（fc_dim != 0）：抽样口径的**显式声明** -------------
    # 这些键**只在有 FC 时新增**（不是恒为 null）：无 FC 产物的 JSON 文本必须与
    # 改动前逐字节相同（零回归锚点），因此既有键的顺序与取值一字不动、新键只追加。
    if fc_data is not None:
        meta.update({
            "hasFc": True,
            "fcTopK": int(fc_data.top_k),
            "fcDim": int(fc_data.fc_dim),
            "fcWidth": int(fc_data.fc_width),
            "fcSampleEdges": int(len(fc_data.edges)),
            "fcNotAllConnections": True,
            "fcDeclaration": fc_data.declared_statement(),
        })

    neurons = [
        {
            "id": i,
            "x": round(p[0], pos_digits_eff),
            "y": round(p[1], pos_digits_eff),
            "z": round(p[2], pos_digits_eff),
            "layer": data.layer_of_neuron(i),
            "in_degree": int(data.in_degree[i]),
            "out_degree": int(data.out_degree[i]),
            "s_in": bool(data.in_scope_mask[i]),
            "s_out": bool(data.out_scope_mask[i]),
        }
        for i, p in enumerate(data.neuron_pos)
    ]

    edges = [
        {
            "src": int(s),
            "dst": int(d),
            "w": round(float(w), weight_digits_eff),
            "d": round(float(dd), 6),
        }
        for s, d, w, dd in zip(data.edge_src, data.edge_dst, data.edge_weight, data.edge_dist)
    ]

    # ---- 两端全连接包裹负载段（fc_dim != 0 时才存在；否则为 None）--------
    # 无 FC 产物该键恒为 None，其 JSON 文本与改动前**逐字节相同**（零回归）。
    fc_section: dict[str, Any] | None = None
    if fc_data is not None:
        fc_section = {
            "topK": int(fc_data.top_k),
            "declaration": fc_data.declared_statement(),
            "fcDim": int(fc_data.fc_dim),
            "fcWidth": int(fc_data.fc_width),
            "inputDim": int(fc_data.input_dim),
            "outputDim": int(fc_data.output_dim),
            "flowAxis": str(fc_data.panels["input"].axis),
            "projWeightShape": [int(v) for v in fc_data.proj_weight_shape],
            "fcOutWeightShape": [int(v) for v in fc_data.fc_out_weight_shape],
            # 参数量（**全部**连线数，与抽样条数分开标注）
            "projCount": int(fc_data.proj_count),
            "fcOutCount": int(fc_data.fc_out_count),
            "fcInWeightShape": [int(v) for v in fc_data.fc_in_weight_shape],
            "fcInCount": int(fc_data.fc_in_count),
            # 抽样条数（构造量：恰好 = (|S_in| + |S_out|) × k）
            "sampleEdges": int(len(fc_data.edges)),
            "sampleEdgesExpected": int(fc_data.expected_sample_edges),
            "projAbsMean": round(float(fc_data.proj_abs_mean), weight_digits_eff),
            "fcOutAbsMean": round(float(fc_data.fc_out_abs_mean), weight_digits_eff),
            "panels": [
                {
                    "name": p.name,
                    "axis": p.axis,
                    "flow": round(float(p.flow), pos_digits_eff),
                    "thickness": round(float(p.thickness), pos_digits_eff),
                    "flowInterval": [round(float(v), pos_digits_eff) for v in p.flow_interval],
                    "cols": int(p.cols),
                    "rows": int(p.rows),
                    "cellSize": round(float(p.cell_size), pos_digits_eff),
                    "units": [
                        {
                            "unit": int(u.unit),
                            "x": round(u.pos[0], pos_digits_eff),
                            "y": round(u.pos[1], pos_digits_eff),
                            "z": round(u.pos[2], pos_digits_eff),
                            "norm": round(float(u.norm), weight_digits_eff),
                        }
                        for u in p.units
                    ],
                }
                for p in (fc_data.panels["input"], fc_data.panels["output"])
            ],
            "blocks": [
                {
                    "name": b.name,
                    "label": b.label,
                    "dim": int(b.dim),
                    "x": round(b.center[0], pos_digits_eff),
                    "y": round(b.center[1], pos_digits_eff),
                    "z": round(b.center[2], pos_digits_eff),
                    "sx": round(b.size[0], pos_digits_eff),
                    "sy": round(b.size[1], pos_digits_eff),
                    "sz": round(b.size[2], pos_digits_eff),
                }
                for b in fc_data.blocks
            ],
            # 抽样连线：一侧一条，`w` 与 `unit` 与 `neuron` 一一对应
            "edges": [
                {
                    "side": side,
                    "unit": int(unit),
                    "neuron": int(neuron),
                    "w": round(float(w), weight_digits_eff),
                }
                for side, unit, neuron, w in fc_data.edges
            ],
        }

    return {
        "meta": meta,
        "layers": layers,
        "neurons": neurons,
        "edges": edges,
        # "fc" 段只在有 FC 时存在（无 FC 路径零回归，见上文 meta 注释）。
        **({"fc": fc_section} if fc_section is not None else {}),
        # labels 段保持既有 6 个键与取值**一字不动**：无 FC 产物的 HTML 必须与
        # 改动前逐字节相同（零回归锚点），因此「S_in/S_out 补充说明」不放这里，
        # 改由 FC 段自身的声明文本 + 图例呈现。
        "labels": {
            "neuron": "神经元",
            "edge": "连接",
            "plane": "层平面",
            "s_in": "S_in（接入输入层）",
            "s_out": "S_out（接出读出头）",
            "threshold": "边权重阈值 |w| >= ",
        },
    }


def build_html(
    data: TopologyData,
    threshold: float = DEFAULT_THRESHOLD,
    assets_dir: str | Path | None = None,
    include_planes: bool = True,
    fc_top_k: int = DEFAULT_FC_TOP_K,
) -> str:
    """读取 assets 模板与渲染器，生成单文件自包含 HTML 文本。

    原理：模板文件 ``assets/viewer.html`` 内含两个占位标记
    （``/*__N3D_DATA_JSON__*/`` 与 ``/*__N3D_VIEWER_JS__*/``），构建时分别替换为
    JSON 数据负载与 ``assets/viewer.js`` 源码，因而产物不含任何外部引用。

    Args:
        data: 拓扑数据。
        threshold: 初始边权重阈值。
        assets_dir: 资源目录，默认 ``n3d_viz/assets``。
        include_planes: 层参考平面的初始开关状态。
        fc_top_k: 两端全连接包裹的抽样口径 k（无 FC 产物无影响）。

    Returns:
        完整 HTML 文本。

    Raises:
        FileNotFoundError: 模板或渲染器缺失。
    """
    base = Path(assets_dir) if assets_dir is not None else ASSETS_DIR
    tpl = (base / "viewer.html").read_text(encoding="utf-8")
    js = (base / "viewer.js").read_text(encoding="utf-8")
    payload = build_html_payload(
        data, threshold=threshold, include_planes=include_planes, fc_top_k=fc_top_k
    )
    # ensure_ascii=False 让中文标签可读；separators 去掉多余空白以减小体积。
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # 防止数据中意外出现 </script> 提前闭合脚本标签（JSON 中只会来自字符串）。
    blob = blob.replace("</", "<\\/")
    if _TPL_MARKER not in tpl or _DATA_MARKER not in tpl:
        raise ValueError(
            f"HTML 模板缺少占位标记：{base / 'viewer.html'}；"
            f"需要 {_DATA_MARKER} 与 {_TPL_MARKER}。"
        )
    # ---- 两端全连接包裹：只在有 FC 时追加一段独立的 <script> ----------------
    # 为什么追加独立 <script> 而不是改 viewer.js / viewer.html：
    #   两者都被逐字内联进每一份 HTML，任何改动都会让**无 FC 产物**的字节数变化，
    #   破坏「无 FC 产物逐字节零回归」的硬约束。独立脚本块在无 FC 时完全不出现，
    #   因此既有产物一个字节都不变，而 FC 产物仍然保持单文件自包含。
    # 顺序要求：该块必须**先于** viewer.js 执行（viewer.js 末尾会同步完成首帧），
    # 因此它被插到「内联数据」块里（数据赋值之前），从而在 DOM 中排在 viewer.js 之前。
    fc_block = ""
    if data.fc is not None:
        fc_js = (base / FC_VIEWER_ASSET).read_text(encoding="utf-8")
        # 注意：这里**不**做 </ 转义 —— 渲染器源码里存在合法的 "</b>" 这类字符串
        # （悬停详情标签），转义会破坏它。该文件由本模块自行维护，不含 </script>。
        fc_block = ";\n</script>\n<script>\n" + fc_js + "\n"
    html = tpl.replace(_TPL_MARKER, js).replace(_DATA_MARKER, blob + fc_block)
    return html


# ---------------------------------------------------------------------------
# 三件套写出
# ---------------------------------------------------------------------------
def write_outputs(
    data: TopologyData,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    out: str | Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    ply_binary: bool = True,
    with_ply_edges: bool = False,
    include_planes: bool = True,
    fc_top_k: int = DEFAULT_FC_TOP_K,
    log: Callable[[str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """写出 HTML / PLY / OBJ 三件套。

    Args:
        data: 拓扑数据。
        out_dir: 输出目录（不存在则创建）。
        out: 显式 HTML 路径；给出时 PLY / OBJ 与它同目录同名。
        threshold: 初始边权重阈值（写进 HTML 的初始状态）。
        ply_binary: True 写 ``binary_little_endian``（float32 精确），False 写 ascii。
        with_ply_edges: PLY 是否附加 ``edge`` 元素（顶点索引对 + 权重）。
        include_planes: 是否在 HTML 中绘制层参考平面。
        fc_top_k: 两端全连接包裹的抽样口径 k（``fc_dim != 0`` 时才生效；
            与数据中已有口径不同时按 :func:`with_fc_top_k` 重抽）。
        log: 进度回调（CLI / GUI 共用；None 时静默）。

    Returns:
        ``{"html": {"path","bytes","existed"}, "ply": {...}, "obj": {...}}``

    Raises:
        ValueError: ``fc_top_k`` 越界。
    """
    emit = log if log is not None else (lambda _msg: None)
    # 抽样口径在此统一生效：HTML / PLY / OBJ 三条产物必须用同一个 k。
    data = with_fc_top_k(data, fc_top_k)
    paths = resolve_output_paths(data.checkpoint, out_dir=out_dir, out=out)
    for key in ("html", "ply", "obj"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)

    from .export_geometry import write_obj, write_ply  # 延迟导入避免循环

    reports: dict[str, dict[str, Any]] = {}

    if data.fc is not None:
        # 抽样口径必须在日志里显式声明（非全部连接），否则抽样图易被误读。
        emit(
            f"[FC] 两端全连接包裹：fc_dim={data.fc.fc_dim} H={data.fc.fc_width} "
            f"[{data.fc.proj_weight_shape[0]},{data.fc.proj_weight_shape[1]}]×"
            f"[{data.fc.fc_out_weight_shape[0]},{data.fc.fc_out_weight_shape[1]}] "
            f"参数量 {data.fc.proj_count:,}+{data.fc.fc_out_count:,}；"
            f"{data.fc.declared_statement()}；抽样 {data.fc.n_edges} 条"
            f"（top-k={data.fc.top_k}，面板点 {data.fc.n_panel_points} 个）"
        )

    # PLY 点云
    target = paths["ply"]
    existed = target.exists()
    if existed:
        emit(f"[提示] 产物已存在，将被覆盖：{target}")
    info = write_ply(data, target, binary=ply_binary, with_edges=with_ply_edges)
    info["existed"] = existed
    reports["ply"] = info
    emit(
        f"[PLY] {target}（{info['vertices']} 顶点"
        + (f" / {info['edges']} 边" if with_ply_edges else "")
        + f" / {info['bytes']} 字节 / {info['format']}）"
    )

    # OBJ 线框
    target = paths["obj"]
    existed = target.exists()
    if existed:
        emit(f"[提示] 产物已存在，将被覆盖：{target}")
    info = write_obj(data, target)
    info["existed"] = existed
    reports["obj"] = info
    emit(f"[OBJ] {target}（{info['vertices']} v 行 / {info['lines']} l 行 / {info['bytes']} 字节）")

    # 自包含 HTML
    target = paths["html"]
    existed = target.exists()
    if existed:
        emit(f"[提示] 产物已存在，将被覆盖：{target}")
    html = build_html(data, threshold=threshold, include_planes=include_planes,
                      fc_top_k=fc_top_k)
    target.write_text(html, encoding="utf-8")
    size = target.stat().st_size
    info = {"path": str(target), "bytes": size, "existed": existed}
    reports["html"] = info
    emit(f"[HTML] {target}（{size} 字节，单文件自包含）")

    return reports


def render_default(
    ckpt_path: str | Path = "",
    out_dir: str | Path = DEFAULT_OUT_DIR,
    out: str | Path | None = None,
    log: Callable[[str], None] | None = None,
    *,
    data: TopologyData | None = None,
) -> dict[str, dict[str, Any]]:
    """以 **CLI 默认形式**（:data:`DEFAULT_WRITE_OPTIONS`）渲染三件套。

    本函数是「CLI 默认路径」与「零回归锚点的代码回归层」共用的**唯一渲染入口**：
    两者必须走同一条路径，否则 ``__main__`` 的默认值一变，锚点就会静默漂移成
    「另一套默认形式」的产物，或让锚点断言无故失败（见 :data:`DEFAULT_WRITE_OPTIONS`）。

    它是 :func:`write_outputs` 的一层薄封装，自身**不再手写任何参数**，
    因此与 :data:`DEFAULT_WRITE_OPTIONS` 天然不会脱耦。

    Args:
        ckpt_path: checkpoint 路径；仅在 ``data`` 为 None 时用于加载。
        out_dir: 输出目录（不存在则创建）。
        out: 显式 HTML 路径；给出时 PLY / OBJ 与它同目录同名。
        log: 进度回调（CLI / GUI 共用；None 时静默）。
        data: 已加载的拓扑数据。CLI 为了打印 ``[数据]`` 摘要已解析过一次，
            传入可避免重复加载十几 MB 的产物。

    Returns:
        与 :func:`write_outputs` 相同的报告字典。

    Raises:
        CheckpointError: ``data`` 为 None 时由 :func:`load_topology` 抛出。
    """
    topology = load_topology(ckpt_path) if data is None else data
    return write_outputs(topology, out_dir=out_dir, out=out, log=log, **DEFAULT_WRITE_OPTIONS)
