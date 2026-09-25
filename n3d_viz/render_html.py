"""自包含 HTML 生成层。

职责：把 ``assets/viewer.html`` 模板与 ``assets/viewer.js`` 渲染器读入内存，
并将拓扑 JSON 负载内联进去，产出**单文件** HTML。

自包含含义（可被 verify_viz.py 静态断言）：

* 无 ``http://`` / ``https://`` / 协议相对 ``//`` 外部引用；
* 不含 ``syn_dist`` 巨型张量（默认规模约 16.8MB，严禁嵌入）；
* 体积 < 2MB；
* 断网可打开，运行时不再请求任何资源。

本模块不实现任何绘图算法，绘图逻辑全部在 ``assets/viewer.js``；
Python 侧只负责「取数据 -> 序列化 -> 填充占位符」。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path
from typing import Any, Callable

# 允许脚本直跑：此时 __package__ 为空，相对导入会失败。
if __package__ in (None, ""):  # pragma: no cover - 仅脚本直跑时命中
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from n3d_viz import core
else:
    from . import core


def render_html(
    data: core.TopologyData,
    threshold: float = core.DEFAULT_THRESHOLD,
    include_planes: bool = True,
    assets_dir: str | Path | None = None,
) -> str:
    """生成单文件自包含 HTML 文本。

    Args:
        data: 拓扑数据。
        threshold: 初始边权重阈值。
        include_planes: False 时把默认的层平面开关关闭。
        assets_dir: 资源目录，默认 ``n3d_viz/assets``。

    Returns:
        完整 HTML 文本（UTF-8 字符串）。
    """
    return core.build_html(
        data, threshold=threshold, assets_dir=assets_dir, include_planes=include_planes
    )


def write_html(
    data: core.TopologyData,
    path: str | Path,
    threshold: float = core.DEFAULT_THRESHOLD,
    include_planes: bool = True,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """把单文件 HTML 写到磁盘。

    Args:
        data: 拓扑数据。
        path: 目标路径。
        threshold: 初始边权重阈值。
        include_planes: 是否默认显示层平面。
        log: 日志回调。

    Returns:
        ``{"path","bytes","existed"}``
    """
    emit = log if log is not None else (lambda _m: None)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    if existed:
        emit(f"[提示] 产物已存在，将被覆盖：{target}")
    html = render_html(data, threshold=threshold, include_planes=include_planes)
    target.write_text(html, encoding="utf-8")
    return {"path": str(target), "bytes": target.stat().st_size, "existed": existed}


def assert_self_contained(html: str, max_bytes: int = 2 * 1024 * 1024) -> dict[str, Any]:
    """静态自包含断言：外部引用 / syn_dist / 体积。

    Args:
        html: HTML 文本。
        max_bytes: 体积上限（字节，按 UTF-8 编码计）。

    Returns:
        ``{"bytes","external_refs","has_syn_dist","ok"}``

    Raises:
        AssertionError: 任一硬约束被违反。
    """
    raw = html.encode("utf-8")
    nbytes = len(raw)
    hits = [pat for pat in ("http://", "https://") if pat in html]
    # 协议相对引用：src="// 或 href="// 或 url(//
    proto_relative = any(
        pat in html for pat in ('src="//', "src='//", 'href="//', "href='//", "url(//")
    )
    has_syn = "syn_dist" in html
    report = {
        "bytes": nbytes,
        "external_refs": hits + (["//protocol-relative"] if proto_relative else []),
        "has_syn_dist": has_syn,
        "ok": not hits and not proto_relative and not has_syn and nbytes < max_bytes,
    }
    assert not hits, f"HTML 含外部 URL 引用：{hits}"
    assert not proto_relative, "HTML 含协议相对外部引用（//...）"
    assert not has_syn, "HTML 不应包含 syn_dist（约 16.8MB）"
    assert nbytes < max_bytes, f"HTML 体积 {nbytes} 字节 >= 上限 {max_bytes} 字节"
    return report


def main(argv: list[str] | None = None) -> int:
    """脚本直跑入口：``python n3d_viz/render_html.py --checkpoint <pt>``。

    Returns:
        进程退出码（0 成功，2 参数错误，3 checkpoint 错误）。
    """
    parser = argparse.ArgumentParser(description="生成单文件自包含三维 HTML")
    parser.add_argument("--checkpoint", required=True, help="二期 checkpoint 路径")
    parser.add_argument("--out", default=None, help="输出 HTML 路径（默认由 ckpt 名派生）")
    parser.add_argument("--threshold", type=float, default=core.DEFAULT_THRESHOLD)
    parser.add_argument("--no-plan-planes", action="store_true", help="默认关闭层平面")
    args = parser.parse_args(argv)

    try:
        data = core.load_topology(args.checkpoint)
    except core.CheckpointError as exc:
        print(f"[错误] {exc}")
        return 3
    paths = core.resolve_output_paths(args.checkpoint, out=args.out)
    info = write_html(
        data,
        paths["html"],
        threshold=args.threshold,
        include_planes=not args.no_plan_planes,
        log=lambda m: print(m),
    )
    html = Path(info["path"]).read_text(encoding="utf-8")
    report = assert_self_contained(html)
    print(f"[校验] {report}")
    return 0


if __name__ == "__main__":  # pragma: no cover - 脚本直跑分支
    raise SystemExit(main())
