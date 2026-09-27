"""零依赖几何导出：PLY 点云写出器与 OBJ 线框写出器。

两个写出器都只使用 Python 标准库（``struct`` / ``pathlib``），坐标以
``float32`` 原值写出（PLY 二进制）或以 9 位有效数字写出（OBJ ascii），
足以支撑「产物坐标与 checkpoint 的 ``neuron_pos`` 逐位一致（容差 1e-6）」
这一硬断言。

颜色来源：神经元按所在的拓扑层着色（``TopologyData.layer_groups``），层色由
:func:`n3d_viz.core.layer_palette_rgb` 给出——与 HTML 渲染器的层色**同源**
（K <= 9 取既有 9 色前 K 个；K > 9 按均匀色相扩展，K 个层色两两不同）。

**两端全连接包裹（``config.fc_dim != 0``）**：在既有 PLY 顶点之后追加
**2×H 个面板单元点** + **2 个边界块中心点**，在既有 OBJ 顶点之后追加同样数量的
``v`` 行与抽样连线的 ``l`` 行，并用独立的 group / object 名把「核心边」与
「FC 边」分开。无 FC 产物（``fc_dim`` 缺失或为 0）的字节流与改动前**完全一致**。
"""

from __future__ import annotations

import pathlib
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

# 允许脚本直跑（python n3d_viz/export_geometry.py）：此时本文件不属于任何包，
# 相对导入会失败。先把上层目录加入 sys.path 并回退到绝对导入。
if __package__ in (None, ""):  # pragma: no cover - 仅脚本直跑时命中
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from n3d_viz.core import TopologyData
    from n3d_viz import core as _core
else:
    from .core import TopologyData
    from . import core as _core

#: 层配色基础表（9 色 RGB），与 :data:`n3d_viz.core.LEVEL_PALETTE_BASE` **同源**：
#: 由后者经 ``core.hex_to_rgb`` 转换而来，因此 PLY 层色与 HTML 层色逐位一致。
#: K <= 9 时按前 K 个取用；K > 9 时由 ``core.layer_palette_rgb`` 按均匀色相扩展。
LAYER_COLORS: tuple[tuple[int, int, int], ...] = tuple(
    _core.hex_to_rgb(c) for c in _core.LEVEL_PALETTE_BASE
)

#: OBJ 坐标小数位。float32 的有效十进制位约 9 位，故 9 位可无损回读。
_OBJ_COORD_DIGITS: int = 9

#: OBJ 中「核心边」与「两端全连接包裹抽样边」的 group 名（两者必须可区分）。
OBJ_GROUP_CORE: str = "n3d_viz_core_edges"
OBJ_GROUP_FC: str = "n3d_viz_fc_sampled_edges"

#: PLY 中 FC 抽样边元素的名称（与核心 ``edge`` 元素分开，互不干扰）。
PLY_FC_EDGE_ELEMENT: str = "fc_edge"
#: PLY 中标记面板 / 边界块顶点的边元素名（顶点分类标记的来源）。
PLY_FC_NODE_ELEMENT: str = "fc_node"


def _clamp_byte(value: float) -> int:
    """把 0..1 的浮点映射到 0..255 的字节。"""
    return max(0, min(255, int(round(value * 255.0))))


def layer_colors_rgb(data: TopologyData) -> list[tuple[int, int, int]]:
    """返回该产物的 K 个层色（RGB），K 可大于 9（按均匀色相扩展）。

    Args:
        data: 拓扑数据。

    Returns:
        长度为 ``data.n_layers`` 的颜色列表，两两不同。
    """
    return _core.layer_palette_rgb(data.n_layers)


def neuron_colors(data: TopologyData) -> list[tuple[int, int, int]]:
    """按层返回每个神经元的 RGB 颜色（层色与 HTML 渲染器同源）。"""
    colors: list[tuple[int, int, int]] = [(200, 200, 200)] * data.n_neurons
    palette = layer_colors_rgb(data)
    for k, group in enumerate(data.layer_groups):
        rgb = palette[k]
        for idx in group:
            if 0 <= idx < data.n_neurons:
                colors[idx] = rgb
    return colors


def weight_colors(data: TopologyData) -> list[tuple[int, int, int]]:
    """按 ``|edge_weight|`` 归一化映射边颜色（弱=青蓝，强=橙红）。"""
    if not data.edge_weight:
        return []
    abs_w = [abs(w) for w in data.edge_weight]
    lo, hi = min(abs_w), max(abs_w)
    span = (hi - lo) or 1.0
    out: list[tuple[int, int, int]] = []
    for w in abs_w:
        t = (w - lo) / span
        r = _clamp_byte(0.20 + 0.75 * t)
        g = _clamp_byte(0.60 - 0.45 * t)
        b = _clamp_byte(0.95 - 0.85 * t)
        out.append((r, g, b))
    return out


def weight_color_rgb(value: float, lo: float, hi: float) -> tuple[int, int, int]:
    """把单个权重绝对值映射成 RGB（**与 HTML 渲染器的 ``weightToRgb`` 同源**）。

    口径唯一化：本函数是 Python 侧的唯一实现，``assets/viewer.js`` 中的
    ``weightToRgb`` 是同一条公式的 JS 镜像（``r = 0.20+0.75t``、``g = 0.60-0.45t``、
    ``b = 0.95-0.85t``，``t = (|w| - lo) / (hi - lo)``，逐分量 clamp 到 0..255）。
    模块内**不存在第二份 FC 色表**——层色仍由 :func:`layer_colors_rgb` 提供。

    Args:
        value: 权重绝对值。
        lo / hi: 归一化区间的下界 / 上界（``hi == lo`` 时按 ``t = 0`` 处理）。

    Returns:
        ``(r, g, b)``，各分量 0..255。
    """
    span = (float(hi) - float(lo)) or 1.0
    t = (float(value) - float(lo)) / span
    return (
        _clamp_byte(0.20 + 0.75 * t),
        _clamp_byte(0.60 - 0.45 * t),
        _clamp_byte(0.95 - 0.85 * t),
    )


def fc_points(data: TopologyData) -> list[tuple[tuple[float, float, float], tuple[int, int, int], str]]:
    """返回两端全连接包裹的**附加点**（面板单元点 + 边界块中心点）。

    顺序固定为：输入面板 H 个单元 → 输出面板 H 个单元 → 输入边界块 → 输出边界块，
    因此「附加点总数 == 2×H + 2」，可被产物断言逐位核对。

    颜色口径：单元点按该单元的**权重范数**映射（输入侧取 ``fc_in_weight`` 行范数、
    输出侧取 ``fc_out_weight`` 行范数），边界块中心点用固定灰色（它不是权重单元）。

    Args:
        data: 拓扑数据（``fc`` 为 None 时返回空列表 → 无 FC 产物字节流不变）。

    Returns:
        ``[(坐标, RGB, 分类标记), ...]``；分类标记为 ``"panel_input"`` /
        ``"panel_output"`` / ``"block_input"`` / ``"block_output"``。
    """
    fc = data.fc
    if fc is None:
        return []
    norms = fc.in_unit_norms + fc.out_unit_norms
    lo = min(norms) if norms else 0.0
    hi = max(norms) if norms else 0.0
    out: list[tuple[tuple[float, float, float], tuple[int, int, int], str]] = []
    for panel_name, tag in (("input", "panel_input"), ("output", "panel_output")):
        for unit in fc.panels[panel_name].units:
            out.append((unit.pos, weight_color_rgb(unit.norm, lo, hi), tag))
    for block, tag in zip(fc.blocks, ("block_input", "block_output")):
        out.append((block.center, (200, 200, 200), tag))
    return out


def fc_edge_lines(data: TopologyData) -> list[tuple[int, int, float]]:
    """返回 FC 抽样连线在 **OBJ 顶点编号（1 基）** 下的三元组 ``(单元点号, 神经元号, 权重)``。

    映射原理（必须与 :func:`write_obj` 的顶点追加顺序一致）：

    * 神经元 ``i`` 的顶点号 = ``i + 1``；
    * 输入面板单元 ``u`` 的顶点号 = ``data.n_neurons + 1 + u``；
    * 输出面板单元 ``u`` 的顶点号 = ``data.n_neurons + 1 + H + u``；
    * 边界块中心点跟在面板点之后（本函数不使用它们，只列出以便核对）。

    Args:
        data: 拓扑数据（无 FC 时返回空列表）。

    Returns:
        ``[(unit_vertex, neuron_vertex, weight), ...]``，顺序与
        :attr:`n3d_viz.core.FcData.edges` 完全一致（确定性，可逐字节复现）。
    """
    fc = data.fc
    if fc is None:
        return []
    out: list[tuple[int, int, float]] = []
    for side, unit, neuron, w in fc.edges:
        base = data.n_neurons + 1 + (0 if side == "input" else fc.fc_width)
        out.append((base + unit, neuron + 1, float(w)))
    return out


def vertex_lines(data: TopologyData) -> list[str]:
    """返回 OBJ 的 ``v`` 行（按神经元 id 顺序，1 基索引由 :func:`write_obj` 处理）。"""
    return [
        f"v {x:.{_OBJ_COORD_DIGITS}g} {y:.{_OBJ_COORD_DIGITS}g} {z:.{_OBJ_COORD_DIGITS}g}"
        for x, y, z in data.neuron_pos
    ]


def fc_vertex_lines(data: TopologyData) -> list[str]:
    """返回 OBJ 中**追加**的 FC 顶点 ``v`` 行（面板单元点 + 边界块中心点）。

    无 FC 时返回空列表，:func:`write_obj` 因此不会多写任何一行（零回归）。
    """
    return [
        f"v {x:.{_OBJ_COORD_DIGITS}g} {y:.{_OBJ_COORD_DIGITS}g} {z:.{_OBJ_COORD_DIGITS}g}"
        for (x, y, z), _rgb, _tag in fc_points(data)
    ]


def line_lines(data: TopologyData) -> list[str]:
    """返回 OBJ 的 ``l`` 行（索引从 1 开始，与 ``v`` 行顺序一一对应）。"""
    return [f"l {s + 1} {d + 1}" for s, d in zip(data.edge_src, data.edge_dst)]



def write_ply(
    data: TopologyData,
    path: str | Path,
    binary: bool = True,
    with_edges: bool = False,
) -> dict[str, Any]:
    """写出 PLY 点云（可选把连接作为 ``edge`` 元素一并写出）。

    PLY 头声明顶点数 N 与元素属性；二进制模式以 little-endian float32 直接
    落盘，坐标与 ``neuron_pos`` 的 float32 位模式完全相同。

    两端全连接包裹（``config.fc_dim != 0``）时，在 ``vertex`` 元素**之后**追加：

    * ``element fc_node``：``2×H + 2`` 个附加点（面板单元点 + 边界块中心点），
      带 ``uchar kind`` 分类标记（0=输入面板 / 1=输出面板 / 2=输入边界块 /
      3=输出边界块），因此 PLY 里也能区分「神经元」与「全连接层节点」；
    * ``element fc_edge``：抽样连线（单元点索引 + 神经元索引 + 权重）。

    无 FC 产物不追加任何元素、头部注释一字不动，字节流与改动前**完全一致**。

    Args:
        data: 拓扑数据。
        path: 目标文件路径。
        binary: True 写 ``binary_little_endian``，False 写 ``ascii``。
        with_edges: 是否附加 ``edge`` 元素（顶点索引对）。

    Returns:
        ``{"path","bytes","vertices","edges","format","colors","fc_nodes","fc_edges"}``
    """
    target = Path(path)
    fmt = "binary_little_endian" if binary else "ascii"
    colors = neuron_colors(data)
    fc_pts = fc_points(data)
    fc_edges = fc_edge_lines(data)
    header = [
        "ply",
        f"format {fmt} 1.0",
        "comment N3D viz point cloud: neuron_pos (phase-2)",
        f"comment source checkpoint: {Path(data.checkpoint).name}",
        f"comment seed: {data.config.get('seed')}",
        f"element vertex {data.n_neurons}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if with_edges:
        header += [
            f"element edge {data.n_edges}",
            "property int vertex1",
            "property int vertex2",
            "property float weight",
        ]
    if fc_pts:
        header += [
            f"comment fc: two-end fully-connected wrap (sampled, NOT all connections)",
            f"comment fc: H={data.fc.fc_width} top-k={data.fc.top_k} "
            f"sampled={len(fc_edges)} of proj_weight {data.fc.proj_count} + "
            f"fc_out_weight {data.fc.fc_out_count}",
            f"element {PLY_FC_NODE_ELEMENT} {len(fc_pts)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            # 0=输入面板单元 1=输出面板单元 2=输入边界块 3=输出边界块
            "property uchar kind",
            f"element {PLY_FC_EDGE_ELEMENT} {len(fc_edges)}",
            "property int unit_vertex",
            "property int neuron_vertex",
            "property float weight",
        ]
    header.append("end_header")

    _FC_KIND = {"panel_input": 0, "panel_output": 1, "block_input": 2, "block_output": 3}

    buf = bytearray()
    if binary:
        buf.extend(("\n".join(header) + "\n").encode("ascii"))
        # 二进制体：每条记录按声明顺序紧密排布，无对齐填充。
        for (x, y, z), (r, g, b) in zip(data.neuron_pos, colors):
            buf.extend(struct.pack("<fffBBB", x, y, z, r, g, b))
        if with_edges:
            for s, d, w in zip(data.edge_src, data.edge_dst, data.edge_weight):
                buf.extend(struct.pack("<iif", s, d, w))
        for (x, y, z), (r, g, b), tag in fc_pts:
            buf.extend(struct.pack("<fffBBBB", x, y, z, r, g, b, _FC_KIND[tag]))
        for uv, nv, w in fc_edges:
            buf.extend(struct.pack("<iif", uv, nv, w))
    else:
        lines = list(header)
        for (x, y, z), (r, g, b) in zip(data.neuron_pos, colors):
            lines.append(
                f"{x:.{_OBJ_COORD_DIGITS}g} {y:.{_OBJ_COORD_DIGITS}g} "
                f"{z:.{_OBJ_COORD_DIGITS}g} {r} {g} {b}"
            )
        if with_edges:
            for s, d, w in zip(data.edge_src, data.edge_dst, data.edge_weight):
                lines.append(f"{s} {d} {w:.{_OBJ_COORD_DIGITS}g}")
        for (x, y, z), (r, g, b), tag in fc_pts:
            lines.append(
                f"{x:.{_OBJ_COORD_DIGITS}g} {y:.{_OBJ_COORD_DIGITS}g} "
                f"{z:.{_OBJ_COORD_DIGITS}g} {r} {g} {b} {_FC_KIND[tag]}"
            )
        for uv, nv, w in fc_edges:
            lines.append(f"{uv} {nv} {w:.{_OBJ_COORD_DIGITS}g}")
        buf.extend(("\n".join(lines) + "\n").encode("ascii"))

    target.write_bytes(bytes(buf))
    return {
        "path": str(target),
        "bytes": len(buf),
        "vertices": data.n_neurons,
        "edges": data.n_edges if with_edges else 0,
        "format": fmt,
        "colors": len(set(colors)),
        "fc_nodes": len(fc_pts),
        "fc_edges": len(fc_edges),
    }


def write_obj(data: TopologyData, path: str | Path, with_colors: bool = False) -> dict[str, Any]:
    """写出 OBJ 线框：N 个 ``v`` 行 + E 个 ``l`` 行。

    索引从 1 开始（OBJ 规范），且必须与 ``v`` 行出现顺序一致，否则线框错位。

    两端全连接包裹（``config.fc_dim != 0``）时追加（顺序固定）：

    1. ``v`` 行：``2×H + 2`` 个 FC 顶点（输入面板 H 个单元、输出面板 H 个单元、
       输入边界块中心、输出边界块中心）；
    2. ``g`` / ``o`` 分组：用 :data:`OBJ_GROUP_CORE` 与 :data:`OBJ_GROUP_FC`
       把「核心神经元边」与「FC 抽样边」分开，便于外部工具按组单独显示；
    3. ``l`` 行：``(H 单元 ↔ 神经元)`` 的抽样连线；**伴随注释显式声明
       「抽样显示（每神经元 top-k），非全部连接」**，避免被误读为全连接结构。

    无 FC 产物不追加任何行、头部注释一字不动，字节流与改动前**完全一致**。

    Args:
        data: 拓扑数据。
        path: 目标文件路径。
        with_colors: 是否追加 ``# c`` 注释形式的层色（非标准，仅作备注）。

    Returns:
        ``{"path","bytes","vertices","lines","format","fc_vertices","fc_lines"}``
    """
    target = Path(path)
    fc_v = fc_vertex_lines(data)
    fc_l = fc_edge_lines(data)
    out: list[str] = [
        "# N3D viz wireframe (phase-2 neuron-level edges)",
        f"# source checkpoint: {Path(data.checkpoint).name}",
        f"# seed: {data.config.get('seed')}  N={data.n_neurons}  E={data.n_edges}",
        f"# neurons={data.n_neurons} edges={data.n_edges} layers={data.n_layers}",
        "o n3d_viz_wireframe",
    ]
    out.extend(vertex_lines(data))
    if fc_v:
        # ---- FC 顶点紧跟神经元顶点（索引连续性即由「追加顺序」保证）----
        out.append(f"# fc vertices: {len(fc_v)} （输入面板 {data.fc.fc_width} 单元 + "
                   f"输出面板 {data.fc.fc_width} 单元 + 2 个边界块中心）")
        out.extend(fc_v)
        # 分组只在有 FC 时出现：无 FC 产物的 OBJ 字节流必须与改动前完全一致。
        out.append(f"o {OBJ_GROUP_CORE}")
        out.append(f"g {OBJ_GROUP_CORE}")
    out.extend(line_lines(data))
    if fc_l:
        # ---- FC 抽样边：独立 group / object + 抽样口径的显式声明 ----
        out.append(f"o {OBJ_GROUP_FC}")
        out.append(f"g {OBJ_GROUP_FC}")
        out.append(f"# {data.fc.declared_statement()}")
        out.append(f"# sampling: per-neuron top-k (k={data.fc.top_k}), NOT all connections; "
                   f"declared total would be {data.fc.proj_count + data.fc.fc_out_count} links")
        out.append(f"# H={data.fc.fc_width} unit-vertex base: input={data.n_neurons + 1}, "
                   f"output={data.n_neurons + 1 + data.fc.fc_width}")
        out.extend(f"l {uv} {nv}" for uv, nv, _w in fc_l)
    if with_colors:
        colors = neuron_colors(data)
        out.append("# layer colors (r g b) per vertex index")
        for i, (r, g, b) in enumerate(colors):
            out.append(f"# vc {i + 1} {r} {g} {b}")
    text = "\n".join(out) + "\n"
    # 编码口径：FC 产物的注释含中文（抽样口径声明必须可读），故用 UTF-8 写出；
    # 无 FC 产物虽标注 utf-8 但内容全为 ASCII，**字节流与改动前逐字节相同**。
    # newline="" 禁用换行转换，保证磁盘字节数与 len(text) 完全一致（Windows 上
    # 默认会把 \n 写成 \r\n，导致报告的字节数与实际文件长度不符）。
    target.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "vertices": data.n_neurons,
        "lines": data.n_edges,
        "format": "obj-ascii",
        "fc_vertices": len(fc_v),
        "fc_lines": len(fc_l),
    }


def parse_ply_vertices(path: str | Path) -> list[tuple[float, float, float]]:
    """回读 PLY 的顶点坐标（ascii 与 binary_little_endian 均支持）。

    本函数是 :mod:`n3d_viz.verify_viz` 的独立校验入口：不依赖写码路径，
    直接按 PLY 头声明的元素/属性顺序解析字节流，因此可以真实反映产物内容。

    只解析 ``element vertex``：FC 产物的 ``fc_node`` / ``fc_edge`` 元素紧随其后，
    必须**不**混入顶点属性，否则记录步长算错、坐标被解析成垃圾。

    Args:
        path: PLY 路径。

    Returns:
        顶点坐标列表。

    Raises:
        ValueError: 头部不合法或属性布局不受支持。
    """
    raw = Path(path).read_bytes()
    end = raw.find(b"end_header\n")
    if end < 0:
        raise ValueError(f"PLY 缺少 end_header：{path}")
    header_txt = raw[:end].decode("ascii")
    body = raw[end + len(b"end_header\n"):]
    fmt = ""
    vertex_count = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    n_edges = 0
    edge_props: list[tuple[str, str]] = []
    in_edge = False
    current_element = ""
    for line in header_txt.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            # 记录当前正在解析哪个元素，供 property 行归属判断使用
            current_element = parts[1]
            in_vertex = parts[1] == "vertex"
            in_edge = parts[1] == "edge"
            if in_vertex:
                vertex_count = int(parts[2])
            elif in_edge:
                n_edges = int(parts[2])
        elif parts[0] == "property":
            # 只有当前处于 vertex 元素内部时，属性才属于顶点记录；
            # 否则（例如紧跟在 "element edge" 之后）会把边属性误当顶点属性。
            if current_element == "vertex":
                props.append((parts[1], parts[2]))
            elif current_element == "edge":
                edge_props.append((parts[1], parts[2]))

    type_map = {
        "float": ("f", 4), "float32": ("f", 4),
        "double": ("d", 8), "float64": ("d", 8),
        "uchar": ("B", 1), "uint8": ("B", 1),
        "int": ("i", 4), "int32": ("i", 4),
        "uint": ("I", 4), "uint32": ("I", 4),
        "short": ("h", 2), "ushort": ("H", 2),
    }

    if fmt == "ascii":
        rows = body.decode("ascii").split()
        stride = len(props)
        coords: list[tuple[float, float, float]] = []
        idx = 0
        name_to_pos = {name: i for i, (_t, name) in enumerate(props)}
        for _ in range(vertex_count):
            row = rows[idx:idx + stride]
            idx += stride
            coords.append((
                float(row[name_to_pos["x"]]),
                float(row[name_to_pos["y"]]),
                float(row[name_to_pos["z"]]),
            ))
        return coords

    if fmt != "binary_little_endian":
        raise ValueError(f"不支持的 PLY format：{fmt}")

    rec_fmt = "<" + "".join(type_map[pt][0] for pt, _name in props)
    rec_size = sum(type_map[pt][1] for pt, _name in props)
    if len(body) < vertex_count * rec_size:
        raise ValueError(f"PLY 二进制体长度不足：{path}")
    name_to_pos = {name: i for i, (_t, name) in enumerate(props)}
    coords = []
    for i in range(vertex_count):
        vals = struct.unpack_from(rec_fmt, body, i * rec_size)
        coords.append((
            float(vals[name_to_pos["x"]]),
            float(vals[name_to_pos["y"]]),
            float(vals[name_to_pos["z"]]),
        ))
    return coords


def parse_obj(path: str | Path) -> dict[str, Any]:
    """回读 OBJ，返回顶点与线段（索引转回 0 基）以及 group 分布。

    Args:
        path: OBJ 路径。

    Returns:
        ``{"vertices": [(x,y,z)...], "edges": [(i,j)...], "v_lines": n, "l_lines": m,
        "groups": [group 名...], "group_l_counts": {group 名: l 行数}}``。

    分组口径：``g`` 行切换当前 group；未出现在任何 ``g`` 之后的 ``l`` 行归到
    ``""``（无 FC 产物因此得到 ``groups == []`` 且 ``group_l_counts == {"": E}``，
    与改动前的解析结果自洽）。
    """
    vertices: list[tuple[float, float, float]] = []
    edges: list[tuple[int, int]] = []
    v_lines = l_lines = 0
    groups: list[str] = []
    group_l_counts: dict[str, int] = {}
    current = ""
    # 用 UTF-8 读取：FC 产物的注释含中文（无 FC 产物为纯 ASCII，行为不变）。
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()
            vertices.append((float(x), float(y), float(z)))
            v_lines += 1
        elif line.startswith("g "):
            current = line[2:].strip()
            if current not in groups:
                groups.append(current)
            group_l_counts.setdefault(current, 0)
        elif line.startswith("l "):
            parts = line.split()
            idx = [int(p) for p in parts[1:]]
            for a, b in zip(idx, idx[1:]):
                edges.append((a - 1, b - 1))
            l_lines += 1
            group_l_counts[current] = group_l_counts.get(current, 0) + 1
    return {
        "vertices": vertices,
        "edges": edges,
        "v_lines": v_lines,
        "l_lines": l_lines,
        "groups": groups,
        "group_l_counts": group_l_counts,
    }


def parse_ply_fc_nodes(path: str | Path) -> tuple[int, int]:
    """读取 PLY 头中 ``fc_node`` / ``fc_edge`` 元素声明的条数。

    Args:
        path: PLY 路径。

    Returns:
        ``(fc_node 数, fc_edge 数)``；元素不存在时对应项为 0（即无 FC 产物）。

    Raises:
        ValueError: 缺少 ``end_header``。
    """
    raw = Path(path).read_bytes()
    end = raw.find(b"end_header\n")
    if end < 0:
        raise ValueError(f"PLY 缺少 end_header：{path}")
    counts = {PLY_FC_NODE_ELEMENT: 0, PLY_FC_EDGE_ELEMENT: 0}
    for line in raw[:end].decode("ascii").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] in counts:
            counts[parts[1]] = int(parts[2])
    return counts[PLY_FC_NODE_ELEMENT], counts[PLY_FC_EDGE_ELEMENT]


def parse_ply_edge_count(path: str | Path) -> int:
    """读取 PLY 头中 ``element edge`` 声明的边数（不存在返回 0）。"""
    raw = Path(path).read_bytes()
    end = raw.find(b"end_header\n")
    if end < 0:
        raise ValueError(f"PLY 缺少 end_header：{path}")
    for line in raw[:end].decode("ascii").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "edge":
            return int(parts[2])
    return 0


def _cli() -> int:
    """脚本直跑入口：只导出 PLY / OBJ（不生成 HTML）。

    用法::

        python n3d_viz/export_geometry.py --checkpoint <pt> --out-dir <dir>

    Returns:
        进程退出码（0 成功，3 checkpoint 错误）。
    """
    import argparse

    core = _core

    parser = argparse.ArgumentParser(description="导出 N3D 二期点云 PLY 与线框 OBJ")
    parser.add_argument("--checkpoint", "-c", required=True, help="二期 checkpoint 路径")
    parser.add_argument("--out-dir", "-d", default=core.DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--ply-ascii", action="store_true", help="PLY 用 ascii 格式")
    parser.add_argument("--with-ply-edges", action="store_true", help="PLY 附加 edge 元素")
    args = parser.parse_args()

    try:
        data = core.load_topology(args.checkpoint)
    except core.CheckpointError as exc:
        print(f"[错误] {exc}")
        return 3

    paths = core.resolve_output_paths(args.checkpoint, out_dir=args.out_dir)
    for key in ("ply", "obj"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    ply_info = write_ply(data, paths["ply"], binary=not args.ply_ascii, with_edges=args.with_ply_edges)
    obj_info = write_obj(data, paths["obj"])
    print(f"[PLY] {ply_info['path']}（{ply_info['vertices']} 顶点 / {ply_info['format']} / {ply_info['bytes']} 字节）")
    print(f"[OBJ] {obj_info['path']}（{obj_info['vertices']} v 行 / {obj_info['lines']} l 行 / {obj_info['bytes']} 字节）")
    return 0


if __name__ == "__main__":  # pragma: no cover - 脚本直跑分支
    raise SystemExit(_cli())
