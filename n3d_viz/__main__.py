# -*- coding: utf-8 -*-
# N3D 二期拓扑可视化命令行入口。
#
# 用法示例：
#   python -m n3d_viz --checkpoint checkpoints/n3d_sphere/model.pt
#   python -m n3d_viz --checkpoint checkpoints/n3d_sphere/model.pt --threshold 0.30 --no-plan-planes
#   python -m n3d_viz --checkpoint checkpoints/n3d_sphere/model.pt --out-dir checkpoints/n3d_viz
#   # 非「二期默认产物」的用法：任意符合 N3D 拓扑 schema 的产物都按同一套逻辑渲染
#   # （下例是 K=15 的异构几何产物，走「K > 9 按均匀色相扩展」的层色板）
#   python -m n3d_viz -c checkpoints/n3d_shape/full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt
#   python -m n3d_viz                       # 不带参数 -> 启动 tkinter GUI
#
# 两端全连接包裹（`config.fc_dim != 0`）：
#   python -m n3d_viz -c checkpoints/n3d_shape/full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.pt
#   抽样口径由 `--fc-top-k`（默认 3，范围 1..8）决定：对每个 S_in / S_out 神经元
#   各取 |w| 最大的 top-k 条连线 —— **抽样显示，非全部连接**。
#   `config` 无 `fc_dim` 键或 `fc_dim == 0` 的产物不展示 FC 层（不报错）；
#   `fc_dim != 0` 但缺 FC 键的产物报错退出非 0（不静默降级）。
#
# 几何口径：本模块对几何零假设——图形完全由 neuron_pos 决定，不假设球 / 立方体 /
# 圆柱，也不假设晶格或分层规整性；但仍需符合 N3D 拓扑 schema（键名与形状契约），
# 一期（n3d_proto）产物仍以退出码 3 报错。
#
# 全部生成逻辑复用 core / export_geometry / render_html，本文件不含第二份绘图实现。

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from . import core


# 退出码约定：0 成功 / 2 参数错误 / 3 checkpoint 相关错误（供验收脚本断言非 0）
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CHECKPOINT = 3


def build_parser() -> argparse.ArgumentParser:
    # 构建 argparse 解析器；不带任何参数时启动 GUI。
    parser = argparse.ArgumentParser(
        prog="python -m n3d_viz",
        description="N3D 二期拓扑三维可视化：由 checkpoint 生成自包含 HTML + 点云 PLY + 线框 OBJ。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", "-c", default=None,
        help="二期训练产物 .pt 路径；省略时启动 tkinter GUI",
    )
    parser.add_argument(
        "--out-dir", "-d", default=core.DEFAULT_OUT_DIR,
        help="输出目录（不存在则创建）",
    )
    parser.add_argument(
        "--out", "-o", default=None,
        help="显式指定 HTML 输出路径；PLY / OBJ 与其同目录同名",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=core.DEFAULT_THRESHOLD,
        help="HTML 初始边权重阈值（|edge_weight| 下限）",
    )
    parser.add_argument(
        "--no-plan-planes", action="store_true",
        help="默认不显示层参考平面",
    )
    parser.add_argument(
        "--ply-ascii", action="store_true",
        help="PLY 用 ascii 格式写出（默认 binary_little_endian）",
    )
    parser.add_argument(
        "--with-ply-edges", action="store_true",
        help="PLY 中附加 edge 元素（顶点索引对）",
    )
    parser.add_argument(
        "--fc-top-k", "-k", type=int, default=core.DEFAULT_FC_TOP_K,
        help=(f"两端全连接包裹的抽样口径 k（范围 {core.MIN_FC_TOP_K}..{core.MAX_FC_TOP_K}，"
              "仅 fc_dim != 0 的产物生效；抽样显示，非全部连接）"),
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="只输出最终摘要",
    )
    return parser


def write_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """把 argparse 结果映射为 :func:`core.write_outputs` 的写出参数集。

    起点是 :data:`n3d_viz.core.DEFAULT_WRITE_OPTIONS`（**CLI 默认形式的唯一事实来源**），
    再按各布尔开关覆盖 —— 因此「默认值」在整个模块里只定义一次，不会与
    ``verify_viz`` 的 ``[2d]`` 锚点重渲参数脱耦。

    Args:
        args: :func:`build_parser` 解析结果。

    Returns:
        可直接 ``**`` 展开给 :func:`n3d_viz.core.write_outputs` 的参数字典。
    """
    options = dict(core.DEFAULT_WRITE_OPTIONS)
    options["threshold"] = float(args.threshold)
    options["fc_top_k"] = int(args.fc_top_k)
    if args.ply_ascii:
        options["ply_binary"] = False
    if args.with_ply_edges:
        options["with_ply_edges"] = True
    if args.no_plan_planes:
        options["include_planes"] = False
    return options


def run_cli(argv: Sequence[str] | None = None) -> int:
    # 命令行主流程：解析参数 -> 加载拓扑 -> 写出三件套 -> 打印摘要。
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.checkpoint:
        # 不带参数（或未给 --checkpoint）时启动 GUI；若只给了其它参数则报参数错误。
        rest = [a for a in (argv if argv is not None else sys.argv[1:]) if a not in ("-q", "--quiet")]
        if rest:
            parser.error("未指定 --checkpoint，但提供了其它参数；请给出 --checkpoint 或直接运行 python -m n3d_viz 启动 GUI。")
        try:
            from . import gui
        except Exception as exc:  # noqa: BLE001 - 无显示环境时给出可读提示
            print(f"[错误] 无法启动 GUI：{type(exc).__name__}: {exc}")
            print("提示：无图形环境时请使用 --checkpoint 走命令行入口。")
            return EXIT_USAGE
        return gui.main()

    emit = (lambda _m: None) if args.quiet else (lambda m: print(m))

    # 抽样口径的取值域校验（1..8）：越界一律报参数错误、退出码 2，不静默截断。
    try:
        core.validate_fc_top_k(args.fc_top_k)
    except ValueError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        emit(f"[加载] {args.checkpoint}")
        data = core.load_topology(args.checkpoint, fc_top_k=args.fc_top_k)
    except core.CheckpointNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return EXIT_CHECKPOINT
    except core.CheckpointCorruptedError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return EXIT_CHECKPOINT
    except core.CheckpointSchemaError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return EXIT_CHECKPOINT

    emit(
        f"[数据] N={data.n_neurons} E={data.n_edges} K={data.n_layers} "
        f"S_in={data.n_s_in} S_out={data.n_s_out} seed={data.config.get('seed')} "
        f"test_acc={data.test_acc}"
    )
    emit(f"[数据] 层规模 = {'/'.join(str(c) for c in data.layer_counts)}")
    emit(f"[数据] 层入边 = {'/'.join(str(c) for c in data.layer_edge_counts)}")
    emit(f"[数据] 阈值过滤统计 = {data.edge_threshold_stats((0.05, 0.10, 0.20, 0.30, 0.50))}")
    emit(f"[数据] syn_dist 体积 = {data.syn_dist_bytes} 字节（不嵌入 HTML）")
    if data.fc is None:
        emit("[数据] 两端全连接包裹：未启用（config 无 fc_dim 或 fc_dim == 0）")
    else:
        emit(
            f"[数据] 两端全连接包裹：fc_dim={data.fc.fc_dim} H={data.fc.fc_width} "
            f"proj_weight={tuple(data.fc.proj_weight_shape)} "
            f"fc_out_weight={tuple(data.fc.fc_out_weight_shape)} "
            f"参数量={data.fc.proj_count}/{data.fc.fc_out_count} "
            f"抽样={data.fc.n_edges} 条（top-k={data.fc.top_k}）"
        )

    options = write_options_from_args(args)
    if options == core.DEFAULT_WRITE_OPTIONS:
        # 默认形式：走与零回归锚点 [2d][重渲] 完全相同的共享入口，
        # 保证「锚点 = CLI 默认形式的产物」这一语义永不脱耦。
        reports = core.render_default(
            args.checkpoint, out_dir=args.out_dir, out=args.out, log=emit, data=data,
        )
    else:
        reports = core.write_outputs(
            data, out_dir=args.out_dir, out=args.out, log=emit, **options,
        )

    if reports["html"]["existed"] or reports["ply"]["existed"] or reports["obj"]["existed"]:
        print(
            "[提示] 本次覆盖了已存在的同名产物："
            + ", ".join(k for k, v in reports.items() if v.get("existed"))
        )
    print(
        "[完成] HTML={bytes}字节  PLY={pv}顶点  OBJ={ol}条l行  输出目录={d}".format(
            bytes=reports["html"]["bytes"],
            pv=reports["ply"]["vertices"],
            ol=reports["obj"]["lines"],
            d=Path(reports["html"]["path"]).parent,
        )
    )
    if data.fc is not None:
        print(
            "[完成] FC 抽样连线={fe}条（top-k={k}，面板点={fp}），{decl}".format(
                fe=reports["obj"].get("fc_lines", 0),
                k=data.fc.top_k,
                fp=reports["ply"].get("fc_nodes", 0),
                decl=data.fc.declared_statement(),
            )
        )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    # 统一入口：供 `python -m n3d_viz` 与测试调用。
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
