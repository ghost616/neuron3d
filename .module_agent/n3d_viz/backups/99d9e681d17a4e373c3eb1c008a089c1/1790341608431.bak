"""N3D 二期拓扑数据加载与抽取（纯逻辑层，与 GUI / CLI 解耦）。

本模块是 n3d_viz 的唯一数据入口，负责：

1. 加载 checkpoint（``torch.load(..., map_location="cpu")``）并做二期必需键校验，
   区分「路径不存在」「文件损坏」「非二期产物（缺拓扑键）」三类错误；
2. 把张量抽取为纯 Python 结构（``list[tuple[float, float, float]]`` 等），
   **不持有** syn_dist 等巨型张量；
3. 计算分层着色分组、度统计、阈值过滤统计；
4. 派生产物名并写出三件套。

引用纪律：``neuron_pos`` 与分层结构由 FCC 排布决定，与 seed 无关；
边集（E）、边权重、突触位置随 seed 变化，故一切边级数字必须标注 seed。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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

#: 渲染器源码位置（构建时读入并内联，保证 HTML 单文件自包含）。
ASSETS_DIR: Path = Path(__file__).resolve().parent / "assets"

_TPL_MARKER: str = "/*__N3D_VIEWER_JS__*/"
_DATA_MARKER: str = "/*__N3D_DATA_JSON__*/"


class CheckpointError(RuntimeError):
    """加载 / 解析 checkpoint 时的错误基类。"""


class CheckpointNotFoundError(CheckpointError):
    """checkpoint 路径不存在或不是普通文件。"""


class CheckpointCorruptedError(CheckpointError):
    """checkpoint 文件损坏，或不是合法的 torch 序列化产物。"""


class CheckpointSchemaError(CheckpointError):
    """checkpoint 可读但不是二期产物（缺拓扑键），或张量形状不符合契约。"""


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
        layer_z: K 个层参考平面的 z 坐标（沿 flow_axis 的分层取值）。
        layer_counts: K 个层的神经元数。
        layer_groups: K 个组的神经元 id 列表（分层着色分组）。
        layer_edge_counts: K 个层的入边数（第 1 层实测为 0）。
        degree_hist_in / degree_hist_out: 度 -> 神经元数 的直方图。
        conn_density: E / (N*N)，连接稀疏度。
        syn_dist_bytes: syn_dist 的字节数（仅记录，不读入内存、不嵌入 HTML）。
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

    # -- 便捷视图 ---------------------------------------------------------
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
) -> TopologyData:
    """把状态字典抽取成 :class:`TopologyData`（纯 Python 结构）。

    Args:
        sd: 二期 ``model_state_dict``（须先通过 :func:`ensure_phase2_checkpoint`）。
        checkpoint: 来源 checkpoint 路径（写进产物的溯源信息）。
        config: 顶层 config 字典。
        test_acc: 顶层 test_acc。

    Returns:
        TopologyData 实例。

    Raises:
        CheckpointSchemaError: 键缺失、形状不符合契约，或
            ``topo_index`` / ``edge_src`` / ``edge_dst`` 存在越界索引。
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
        # 层参考平面取该层神经元 z 坐标的均值；二期分层沿 flow_axis=z 且同层 z 相同。
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
    return data


def _syn_dist_bytes(sd: Mapping[str, Any]) -> int:
    """记录 syn_dist 的字节数；**不**把它读入内存（约 16.8MB / 默认规模）。"""
    t = sd.get("syn_dist")
    if isinstance(t, torch.Tensor):
        return int(t.numel() * t.element_size())
    return 0


def load_topology(path: str | Path) -> TopologyData:
    """一步完成「加载 -> 键校验 -> 抽取」，返回 :class:`TopologyData`。

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
) -> dict[str, Any]:
    """构造内联进 HTML 的 JSON 负载。

    约定：``syn_dist [2048,2048]`` 约 16.8MB，**严禁**出现在本负载中；
    坐标与权重**默认按 6 / 8 位小数保留**（见 :data:`DEFAULT_POS_DIGITS` /
    :data:`DEFAULT_WEIGHT_DIGITS`），不是不截断；该默认精度下内联坐标与
    ``neuron_pos`` 的实测 ``max|diff| = 4.510e-07 < 1e-6``，因此仍能满足
    「产物坐标与 neuron_pos 逐位一致（容差 1e-6）」的可断言性；
    若需更高保留精度，可显式传入更大的 ``pos_digits`` / ``weight_digits``。

    Args:
        data: 拓扑数据。
        threshold: 初始边权重阈值。
        pos_digits: 坐标保留的小数位；``None`` 表示取默认值
            :data:`DEFAULT_POS_DIGITS`（= 6）。给定更大位数可提高保留精度。
        weight_digits: 权重保留的小数位；``None`` 表示取默认值
            :data:`DEFAULT_WEIGHT_DIGITS`（= 8）。
        include_planes: 层参考平面的初始开关状态。

    Returns:
        可直接 ``json.dumps`` 的字典。
    """
    # None 统一解析为默认位宽，避免同一个 None 在不同函数里表示不同含义
    pos_digits_eff = DEFAULT_POS_DIGITS if pos_digits is None else int(pos_digits)
    weight_digits_eff = DEFAULT_WEIGHT_DIGITS if weight_digits is None else int(weight_digits)
    layer_colors = ["#4e8cff", "#00b7c2", "#2ecc71", "#a3d977",
                    "#f7d154", "#f39c12", "#e8734a", "#d94f70", "#9b59b6"]
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
                "color": layer_colors[k % len(layer_colors)],
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

    return {
        "meta": meta,
        "layers": layers,
        "neurons": neurons,
        "edges": edges,
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

    Returns:
        完整 HTML 文本。

    Raises:
        FileNotFoundError: 模板或渲染器缺失。
    """
    base = Path(assets_dir) if assets_dir is not None else ASSETS_DIR
    tpl = (base / "viewer.html").read_text(encoding="utf-8")
    js = (base / "viewer.js").read_text(encoding="utf-8")
    payload = build_html_payload(data, threshold=threshold, include_planes=include_planes)
    # ensure_ascii=False 让中文标签可读；separators 去掉多余空白以减小体积。
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # 防止数据中意外出现 </script> 提前闭合脚本标签（JSON 中只会来自字符串）。
    blob = blob.replace("</", "<\\/")
    if _TPL_MARKER not in tpl or _DATA_MARKER not in tpl:
        raise ValueError(
            f"HTML 模板缺少占位标记：{base / 'viewer.html'}；"
            f"需要 {_DATA_MARKER} 与 {_TPL_MARKER}。"
        )
    html = tpl.replace(_TPL_MARKER, js).replace(_DATA_MARKER, blob)
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
        log: 进度回调（CLI / GUI 共用；None 时静默）。

    Returns:
        ``{"html": {"path","bytes","existed"}, "ply": {...}, "obj": {...}}``
    """
    emit = log if log is not None else (lambda _msg: None)
    paths = resolve_output_paths(data.checkpoint, out_dir=out_dir, out=out)
    for key in ("html", "ply", "obj"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)

    from .export_geometry import write_obj, write_ply  # 延迟导入避免循环

    reports: dict[str, dict[str, Any]] = {}

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
    html = build_html(data, threshold=threshold, include_planes=include_planes)
    target.write_text(html, encoding="utf-8")
    size = target.stat().st_size
    info = {"path": str(target), "bytes": size, "existed": existed}
    reports["html"] = info
    emit(f"[HTML] {target}（{size} 字节，单文件自包含）")

    return reports
