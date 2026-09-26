# -*- coding: utf-8 -*-
# N3D 二期拓扑可视化命令行入口。
#
# 用法示例：
#   python -m n3d_viz --checkpoint checkpoints/n3d_sphere/model.pt
#   python -m n3d_viz --checkpoint checkpoints/n3d_sphere/model.pt --threshold 0.30 --no-plan-planes
#   python -m n3d_viz --checkpoint checkpoints/n3d_sphere/model.pt --out-dir checkpoints/n3d_viz
#   python -m n3d_viz                       # 不带参数 -> 启动 tkinter GUI
#
# 全部生成逻辑复用 core / export_geometry / render_html，本文件不含第二份绘图实现。

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

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
        "--quiet", "-q", action="store_true", help="只输出最终摘要",
    )
    return parser


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

    try:
        emit(f"[加载] {args.checkpoint}")
        data = core.load_topology(args.checkpoint)
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

    reports = core.write_outputs(
        data,
        out_dir=args.out_dir,
        out=args.out,
        threshold=args.threshold,
        ply_binary=not args.ply_ascii,
        with_ply_edges=args.with_ply_edges,
        include_planes=not args.no_plan_planes,
        log=emit,
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
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    # 统一入口：供 `python -m n3d_viz` 与测试调用。
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
