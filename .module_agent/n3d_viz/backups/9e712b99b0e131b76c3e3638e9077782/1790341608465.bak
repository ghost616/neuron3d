"""零依赖几何导出：PLY 点云写出器与 OBJ 线框写出器。

两个写出器都只使用 Python 标准库（``struct`` / ``pathlib``），坐标以
``float32`` 原值写出（PLY 二进制）或以 9 位有效数字写出（OBJ ascii），
足以支撑「产物坐标与 checkpoint 的 ``neuron_pos`` 逐位一致（容差 1e-6）」
这一硬断言。

颜色来源：神经元按所在的拓扑层着色（``TopologyData.layer_groups``），
层色由 :data:`LAYER_COLORS` 循环取用，与 HTML 渲染器保持一致。
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

#: 层配色（与 assets/viewer.js 中的层色一致）。
LAYER_COLORS: tuple[tuple[int, int, int], ...] = (
    (78, 140, 255),
    (0, 183, 194),
    (46, 204, 113),
    (163, 217, 119),
    (247, 209, 84),
    (243, 156, 18),
    (232, 115, 74),
    (217, 79, 112),
    (155, 89, 182),
)

#: OBJ 坐标小数位。float32 的有效十进制位约 9 位，故 9 位可无损回读。
_OBJ_COORD_DIGITS: int = 9


def _clamp_byte(value: float) -> int:
    """把 0..1 的浮点映射到 0..255 的字节。"""
    return max(0, min(255, int(round(value * 255.0))))


def neuron_colors(data: TopologyData) -> list[tuple[int, int, int]]:
    """按层返回每个神经元的 RGB 颜色。"""
    colors: list[tuple[int, int, int]] = [(200, 200, 200)] * data.n_neurons
    for k, group in enumerate(data.layer_groups):
        rgb = LAYER_COLORS[k % len(LAYER_COLORS)]
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


def vertex_lines(data: TopologyData) -> list[str]:
    """返回 OBJ 的 ``v`` 行（按神经元 id 顺序，1 基索引由 :func:`write_obj` 处理）。"""
    return [
        f"v {x:.{_OBJ_COORD_DIGITS}g} {y:.{_OBJ_COORD_DIGITS}g} {z:.{_OBJ_COORD_DIGITS}g}"
        for x, y, z in data.neuron_pos
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

    Args:
        data: 拓扑数据。
        path: 目标文件路径。
        binary: True 写 ``binary_little_endian``，False 写 ``ascii``。
        with_edges: 是否附加 ``edge`` 元素（顶点索引对）。

    Returns:
        ``{"path","bytes","vertices","edges","format","colors"}``
    """
    target = Path(path)
    fmt = "binary_little_endian" if binary else "ascii"
    colors = neuron_colors(data)
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
    header.append("end_header")

    buf = bytearray()
    if binary:
        buf.extend(("\n".join(header) + "\n").encode("ascii"))
        # 二进制体：每条记录按声明顺序紧密排布，无对齐填充。
        for (x, y, z), (r, g, b) in zip(data.neuron_pos, colors):
            buf.extend(struct.pack("<fffBBB", x, y, z, r, g, b))
        if with_edges:
            for s, d, w in zip(data.edge_src, data.edge_dst, data.edge_weight):
                buf.extend(struct.pack("<iif", s, d, w))
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
        buf.extend(("\n".join(lines) + "\n").encode("ascii"))

    target.write_bytes(bytes(buf))
    return {
        "path": str(target),
        "bytes": len(buf),
        "vertices": data.n_neurons,
        "edges": data.n_edges if with_edges else 0,
        "format": fmt,
        "colors": len(set(colors)),
    }


def write_obj(data: TopologyData, path: str | Path, with_colors: bool = False) -> dict[str, Any]:
    """写出 OBJ 线框：N 个 ``v`` 行 + E 个 ``l`` 行。

    索引从 1 开始（OBJ 规范），且必须与 ``v`` 行出现顺序一致，否则线框错位。

    Args:
        data: 拓扑数据。
        path: 目标文件路径。
        with_colors: 是否追加 ``# c`` 注释形式的层色（非标准，仅作备注）。

    Returns:
        ``{"path","bytes","vertices","lines","format"}``
    """
    target = Path(path)
    out: list[str] = [
        "# N3D viz wireframe (phase-2 neuron-level edges)",
        f"# source checkpoint: {Path(data.checkpoint).name}",
        f"# seed: {data.config.get('seed')}  N={data.n_neurons}  E={data.n_edges}",
        f"# neurons={data.n_neurons} edges={data.n_edges} layers={data.n_layers}",
        "o n3d_viz_wireframe",
    ]
    out.extend(vertex_lines(data))
    out.extend(line_lines(data))
    if with_colors:
        colors = neuron_colors(data)
        out.append("# layer colors (r g b) per vertex index")
        for i, (r, g, b) in enumerate(colors):
            out.append(f"# vc {i + 1} {r} {g} {b}")
    text = "\n".join(out) + "\n"
    # newline="" 禁用换行转换，保证磁盘字节数与 len(text) 完全一致（Windows 上
    # 默认会把 \n 写成 \r\n，导致报告的字节数与实际文件长度不符）。
    target.write_text(text, encoding="ascii", newline="")
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "vertices": data.n_neurons,
        "lines": data.n_edges,
        "format": "obj-ascii",
    }


def parse_ply_vertices(path: str | Path) -> list[tuple[float, float, float]]:
    """回读 PLY 的顶点坐标（ascii 与 binary_little_endian 均支持）。

    本函数是 :mod:`n3d_viz.verify_viz` 的独立校验入口：不依赖写码路径，
    直接按 PLY 头声明的元素/属性顺序解析字节流，因此可以真实反映产物内容。

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
    """回读 OBJ，返回顶点与线段（索引转回 0 基）。

    Args:
        path: OBJ 路径。

    Returns:
        ``{"vertices": [(x,y,z)...], "edges": [(i,j)...], "v_lines": n, "l_lines": m}``
    """
    vertices: list[tuple[float, float, float]] = []
    edges: list[tuple[int, int]] = []
    v_lines = l_lines = 0
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()
            vertices.append((float(x), float(y), float(z)))
            v_lines += 1
        elif line.startswith("l "):
            parts = line.split()
            idx = [int(p) for p in parts[1:]]
            for a, b in zip(idx, idx[1:]):
                edges.append((a - 1, b - 1))
            l_lines += 1
    return {"vertices": vertices, "edges": edges, "v_lines": v_lines, "l_lines": l_lines}


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
