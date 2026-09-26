"""N3D 二期可视化模块的零依赖验证脚本。

逐条执行硬断言并在末尾汇总退出码（0 = 全部通过，1 = 存在失败）。

用法::

    python n3d_viz/verify_viz.py                       # 默认基准与产物路径
    python n3d_viz/verify_viz.py --report <out.md>     # 额外落一份 Markdown 报告

断言口径（全部来自真实执行，数字来源 checkpoint 与 seed 记录在报告里）：

1.  PLY 顶点数 == N；OBJ 的 ``l`` 行数 == E
2.  产物坐标与 checkpoint 的 ``neuron_pos`` 逐位一致（容差 1e-6）
3.  层着色组数 == K 且各层神经元数 == 期望序列
4.  S_in 高亮数、S_out 高亮数
5.  阈值 0.30 时保留边数 == 实测值
6.  HTML 体积 < 2MB；不含 ``syn_dist``；无 ``http://`` / ``https://`` / 协议相对引用
7.  内联数据规模自洽（neurons == N，edges == E，layers == K）
8.  传入一期产物 -> 子进程退出码非 0 且输出含缺失键名
9.  ``n3d_viz`` 源码不 import ``n3d_sphere`` / ``n3d_proto``（静态扫描）
10. GUI 冒烟：模块可导入且能在 ``withdraw()`` 状态下构造并销毁 Tk 窗口
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from n3d_viz import core, export_geometry  # noqa: E402

#: 默认基准：二期产物 + 实测值（checkpoints/n3d_sphere/model.pt, seed=42,
#: N=256, y=8x8, H=D=0.1, flow_axis=z, fcc, test_acc=0.9759）。
DEFAULT_CKPT = "checkpoints/n3d_sphere/model.pt"
#: 异常路径基准：一期产物（缺 6 个二期拓扑键）。
DEFAULT_PHASE1_CKPT = "checkpoints/n3d_model_full.pt"
EXPECTED_N = 256
EXPECTED_E = 736
EXPECTED_LAYER_COUNTS = [13, 24, 37, 35, 39, 34, 37, 24, 13]
EXPECTED_S_IN = 193
EXPECTED_S_OUT = 187
EXPECTED_KEEP_AT_030 = 379
THRESHOLD = 0.30
TOL = 1e-6
MAX_HTML_BYTES = 2 * 1024 * 1024
#: 负例 checkpoint 只保留这些键（即抽取所需的契约键），避免把 syn_dist 等巨型张量写进临时产物。
_REQUIRED_KEYS_FOR_NEGATIVE: tuple[str, ...] = core.REQUIRED_KEYS

#: 一期产物缺失的二期键（子进程错误信息必须包含其中每一个）。
PHASE1_MISSING_KEYS = (
    "edge_src", "edge_dst", "edge_weight",
    "in_scope_mask", "out_scope_mask", "level_node_reach",
)


def _brief(value, limit: int = 160) -> str:
    """把断言返回值压成一行短文本（避免把整份点云刷进日志）。"""
    text = str(value)
    return text if len(text) <= limit else text[:limit] + " ...(共 %d 字符)" % len(text)


class Checker:
    """断言收集器：记录每条检查的名称、通过与否、实测值与说明。"""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, name: str, fn: Callable[[], Any], detail: str = "") -> Any:
        """执行一条断言并记录结果；返回断言的返回值（失败返回 None）。"""
        try:
            value = fn()
            self.rows.append({"name": name, "ok": True, "value": value, "detail": detail})
            print(f"  [PASS] {name}: {_brief(value)}")
            return value
        except Exception as exc:  # noqa: BLE001 - 验证脚本需要把所有失败都收集起来
            self.rows.append(
                {"name": name, "ok": False, "value": f"{type(exc).__name__}: {exc}", "detail": detail}
            )
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            return None

    def skip(self, name: str, reason: str) -> None:
        """记录一条被跳过的检查（例如缺少无图形环境）。"""
        self.rows.append({"name": name, "ok": True, "value": "SKIP", "detail": reason})
        print(f"  [SKIP] {name}: {reason}")

    @property
    def passed(self) -> int:
        return sum(1 for r in self.rows if r["ok"])

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if not r["ok"])

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.rows if r["value"] == "SKIP")

    def to_markdown(self) -> str:
        """输出 Markdown 表格形式的检查清单。"""
        lines = ["| # | 检查项 | 结果 | 实测值 |", "|---|---|---|---|"]
        for i, r in enumerate(self.rows, 1):
            val = str(r["value"]).replace("|", "\\|")
            lines.append(
                f"| {i} | {r['name']} | {'通过' if r['ok'] else '失败'} | {val} |"
            )
        return "\n".join(lines)


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def _max_abs_diff(pts_a: list[tuple[float, float, float]],
                  pts_b: list[tuple[float, float, float]]) -> float:
    """返回两组点坐标的最大逐分量绝对差。"""
    assert len(pts_a) == len(pts_b), f"点数不一致：{len(pts_a)} vs {len(pts_b)}"
    worst = 0.0
    for pa, pb in zip(pts_a, pts_b):
        for x, y in zip(pa, pb):
            worst = max(worst, abs(float(x) - float(y)))
    return worst


def _source_scan(module_dir: Path) -> list[tuple[str, int, str]]:
    """静态扫描模块源码，返回含跨模块 import 的行 ``(文件, 行号, 内容)``。

    规则：除模块自身文档中对该名字的**说明性文字**外，不得出现
    ``import n3d_sphere`` / ``from n3d_sphere`` / ``import n3d_proto`` /
    ``from n3d_proto`` 形式的语句。
    """
    pattern = re.compile(
        r"^\s*(?:from|import)\s+n3d_(?:sphere|proto)\b", re.MULTILINE
    )
    hits: list[tuple[str, int, str]] = []
    for path in sorted(module_dir.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            hits.append((str(path.relative_to(module_dir.parent)), line_no, m.group(0).strip()))
    return hits


def _read_ply_colors(path: Path) -> list[tuple[int, int, int]]:
    """回读 PLY 顶点颜色（属性顺序由头部声明决定，这里取 red/green/blue 列）。"""
    import struct

    raw = path.read_bytes()
    end = raw.find(b"end_header\n")
    assert end > 0, "PLY 缺少 end_header"
    header = raw[:end].decode("ascii").splitlines()
    props: list[str] = []
    prop_types: list[str] = []
    count = 0
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element" and parts[1] == "vertex":
            count = int(parts[2])
        elif parts[0] == "property" and len(parts) == 3:
            prop_types.append(parts[1])
            props.append(parts[2])
    body = raw[end + len(b"end_header\n"):]
    code_map = {"float": "f", "float32": "f", "uchar": "B", "int": "i", "int32": "i", "double": "d", "float64": "d"}
    rec_fmt = "<" + "".join(code_map[pt] for pt in prop_types)
    code_size = {"f": 4, "B": 1, "i": 4, "d": 8}
    rec_size = sum(code_size[c] for c in rec_fmt[1:])
    idx = {name: i for i, name in enumerate(props)}
    out: list[tuple[int, int, int]] = []
    for i in range(count):
        vals = struct.unpack_from(rec_fmt, body, i * rec_size)
        out.append((vals[idx["red"]], vals[idx["green"]], vals[idx["blue"]]))
    return out


def verify(
    ckpt: str,
    phase1: str,
    out_dir: str,
    report: str | None = None,
    skip_gui: bool = False,
) -> int:
    """执行全部断言，打印结果并返回退出码。"""
    chk = Checker()
    ckpt_path = Path(ckpt)
    phase1_path = Path(phase1)
    out_path = Path(out_dir)

    print("=" * 78)
    print("N3D 二期可视化验证（n3d_viz.verify_viz）")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  一期对照   : {phase1_path}")
    print(f"  输出目录   : {out_path}")
    print("=" * 78)

    # ------------------------------------------------------------------ 数据源
    print("\n[1] 加载 checkpoint 并抽取拓扑")
    data = chk.check(
        "load_topology(二期产物)",
        lambda: core.load_topology(ckpt_path),
        detail="torch.load + 二期必需键校验 + 纯 Python 抽取",
    )
    if data is None:
        print("\n数据源加载失败，后续断言无法执行。")
        _emit_report(chk, report, ckpt, phase1, None)
        return 1

    chk.check("N == 256", lambda: _assert_eq(data.n_neurons, EXPECTED_N), f"来源 {ckpt} seed=42")
    chk.check("E == 736", lambda: _assert_eq(data.n_edges, EXPECTED_E), f"来源 {ckpt} seed=42")
    chk.check(
        "层规模 == 13/24/37/35/39/34/37/24/13",
        lambda: _assert_list_eq(data.layer_counts, EXPECTED_LAYER_COUNTS),
        "来自 level_node_reach 切片",
    )
    chk.check("层数 K == 9", lambda: _assert_eq(data.n_layers, 9))
    chk.check("S_in 高亮数 == 193", lambda: _assert_eq(data.n_s_in, EXPECTED_S_IN),
              "in_scope_mask 为真计数")
    chk.check("S_out 高亮数 == 187", lambda: _assert_eq(data.n_s_out, EXPECTED_S_OUT),
              "out_scope_mask 为真计数")
    chk.check(
        "阈值 0.30 保留边数 == 379 / 736",
        lambda: _assert_eq(data.counts_at(THRESHOLD), EXPECTED_KEEP_AT_030),
        "|edge_weight| >= 0.30",
    )

    # 地面真值：直接从 checkpoint 重新读一次 neuron_pos，避免只用抽取结果自证。
    import torch

    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)["model_state_dict"]
    truth = sd["neuron_pos"].tolist()
    chk.check(
        "抽取坐标与 checkpoint 逐位一致（容差 1e-6）",
        lambda: _assert_le(_max_abs_diff(data.neuron_pos, truth), TOL),
        "抽取路径不引入误差（float32 -> float64 无损提升）",
    )

    # ------------------------------------------------------------ 产物生成
    print("\n[2] 生成三件套产物")
    reports = chk.check(
        "write_outputs(三件套)",
        lambda: core.write_outputs(
            data, out_dir=out_path, threshold=THRESHOLD, ply_binary=True, include_planes=True
        ),
        "PLY=binary_little_endian, HTML 单文件自包含",
    )
    ply_path = Path(reports["ply"]["path"]) if reports else out_path / f"viz_{ckpt_path.stem}.ply"
    obj_path = Path(reports["obj"]["path"]) if reports else out_path / f"viz_{ckpt_path.stem}.obj"
    html_path = Path(reports["html"]["path"]) if reports else out_path / f"viz_{ckpt_path.stem}.html"
    chk.check("产物按 checkpoint 名派生（不撞名）",
              lambda: _assert_true(ply_path.name == f"viz_{ckpt_path.stem}.ply"
                                   and obj_path.name == f"viz_{ckpt_path.stem}.obj"
                                   and html_path.name == f"viz_{ckpt_path.stem}.html"),
              f"{ply_path.name} / {obj_path.name} / {html_path.name}")

    # ------------------------------------------------------------ PLY / OBJ
    if reports is None:
        print("\n[3-5] 产物生成失败，跳过 PLY / OBJ / HTML 断言")
        chk.skip("PLY / OBJ / HTML 断言", "write_outputs 未成功")
        return _finish(chk, report, ckpt, phase1, data, None)
    print("\n[3] PLY 点云断言")
    ply_pts = export_geometry.parse_ply_vertices(ply_path)
    ply_colors = _read_ply_colors(ply_path)
    chk.check("PLY 顶点数 == 256",
              lambda: _assert_eq(len(ply_pts), EXPECTED_N))
    chk.check("PLY 层着色组数 == 9",
              lambda: _assert_eq(len(set(ply_colors)), 9))
    if ply_colors:
        from collections import Counter

        hist = Counter(ply_colors)
        counts = sorted(hist.values(), reverse=True)
        chk.check("PLY 各层颜色计数 == 13/24/37/35/39/34/37/24/13",
                  lambda: _assert_list_eq(counts, sorted(EXPECTED_LAYER_COUNTS, reverse=True)),
                  "颜色分组与层规模一一对应")
    chk.check(
        "PLY 坐标与 neuron_pos 逐位一致（容差 1e-6）",
        lambda: _assert_le(_max_abs_diff(ply_pts, truth), TOL),
        "binary float32 原值写出",
    )

    print("\n[4] OBJ 线框断言")
    obj = export_geometry.parse_obj(obj_path)
    chk.check("OBJ 可解析", lambda: _assert_true(isinstance(obj, dict) and obj["v_lines"] > 0))
    chk.check("OBJ 的 l 行数 == 736", lambda: _assert_eq(obj["l_lines"], EXPECTED_E))
    chk.check("OBJ 的 v 行数 == 256", lambda: _assert_eq(obj["v_lines"], EXPECTED_N))
    chk.check(
        "OBJ 坐标与 neuron_pos 逐位一致（容差 1e-6）",
        lambda: _assert_le(_max_abs_diff(obj["vertices"], truth), TOL),
        "ascii 9 位有效数字",
    )
    chk.check(
        "OBJ 线段索引合法（1..N）",
        lambda: _assert_true(all(0 <= a < EXPECTED_N and 0 <= b < EXPECTED_N for a, b in obj["edges"])
                             and len(obj["edges"]) == EXPECTED_E),
    )

    # ------------------------------------------------------------ HTML
    print("\n[5] HTML 自包含断言")
    if reports:
        from n3d_viz import render_html

        html = html_path.read_text(encoding="utf-8")
        chk.check("HTML 体积 < 2MB", lambda: _assert_lt(html_path.stat().st_size, MAX_HTML_BYTES),
                  f"{html_path.stat().st_size} 字节")
        chk.check("HTML 不含 syn_dist", lambda: _assert_true("syn_dist" not in html),
                  "syn_dist 约 16.8MB，严禁嵌入")
        chk.check(
            "HTML 无 http:// / https:// / 协议相对引用",
            lambda: _assert_true(
                "http://" not in html and "https://" not in html
                and 'src="//' not in html and "href=\"//" not in html and "url(//" not in html
            ),
            "离线可打开",
        )
        report_dict = chk.check("assert_self_contained(HTML)",
                                lambda: render_html.assert_self_contained(html))
        payload = _parse_inline_payload(html)
        chk.check(
            "HTML 内联数据可解析",
            lambda: _assert_true(
                isinstance(payload, dict) and "neurons" in payload and "edges" in payload
            ),
            "neurons/edges/layers 与实际张量一致",
        )
        if payload:
            chk.check(
                "内联 neurons == 256 且 edges == 736 且 layers == 9",
                lambda: _assert_true(
                    len(payload["neurons"]) == EXPECTED_N
                    and len(payload["edges"]) == EXPECTED_E
                    and len(payload["layers"]) == 9
                ),
            )
            chk.check(
                "内联层规模与 meta 一致",
                lambda: _assert_list_eq(payload["meta"]["layer_counts"], EXPECTED_LAYER_COUNTS),
            )
            chk.check(
                "内联阈值统计 == {0.05:681, 0.10:620, 0.20:517, 0.30:379, 0.50:140}",
                lambda: _assert_eq(
                    {k: int(v) for k, v in payload["meta"]["threshold_counts"].items()},
                    {"0.05": 681, "0.10": 620, "0.20": 517, "0.30": 379, "0.50": 140},
                ),
                f"来源 {ckpt_path.name} seed=42",
            )
            chk.check(
                "内联坐标与 neuron_pos 逐位一致（容差 1e-6）",
                lambda: _assert_le(
                    _max_abs_diff([(n["x"], n["y"], n["z"]) for n in payload["neurons"]], truth), TOL
                ),
            )
            chk.check(
                "内联边权绝对值排序统计与 checkpoint 一致",
                lambda: _assert_leq(
                    max(
                        abs(abs(e["w"]) - abs(float(w)))
                        for e, w in zip(payload["edges"], data.edge_weight)
                    ),
                    1e-7,
                ),
                "JSON 保留 double 精度（round 8 位）",
            )
        chk.check("HTML 阈值初始值 == 0.30",
                  lambda: _assert_true('"threshold":0.3' in html))
        chk.check("HTML 无外部 <script src> / <link href>",
                  lambda: _assert_true("<script src=" not in html and "<link" not in html))
        chk.check("HTML 体积报告 == 实际文件长度",
                  lambda: _assert_eq(reports["html"]["bytes"], html_path.stat().st_size))
        chk.check("PLY 报告字节数 == 磁盘字节数",
                  lambda: _assert_eq(reports["ply"]["bytes"], ply_path.stat().st_size),
                  "避免报告值与实际文件不一致")
        chk.check("OBJ 报告字节数 == 磁盘字节数",
                  lambda: _assert_eq(reports["obj"]["bytes"], obj_path.stat().st_size),
                  "新行转换已禁用，避免 Windows \\n -> \\r\\n 差异")
        if report_dict:
            print(f"       自包含报告: {report_dict}")

    # --------------------------------------------- 开关类参数的产物可观测差异
    # 经验教训：参数被解析但未接入实现时，上面的默认路径断言依然全部通过。
    # 因此每个布尔开关都必须碰一条「开/关产物不同」的断言。
    print("\n[5c] 开关类参数的产物可观测差异")
    sw = out_path / "_switch"
    on_paths = core.resolve_output_paths(ckpt_path, out=sw / "on.html")
    off_paths = core.resolve_output_paths(ckpt_path, out=sw / "off.html")
    core.write_outputs(data, out=on_paths["html"], threshold=THRESHOLD, include_planes=True)
    core.write_outputs(data, out=off_paths["html"], threshold=THRESHOLD, include_planes=False)
    on_html = on_paths["html"].read_text(encoding="utf-8")
    off_html = off_paths["html"].read_text(encoding="utf-8")
    chk.check("include_planes=True/False 产物内容不同",
              lambda: _assert_true(on_html != off_html),
              "include_planes 必须真正影响 HTML")
    chk.check("开关状态写入负载 meta.showPlanes",
              lambda: _assert_true('"showPlanes":true' in on_html and '"showPlanes":false' in off_html),
              '渲染器据此初始化复选框与绘制状态')
    chk.check("关闭层平面时 平面相关颜色不再出现于默认绘制集",
              lambda: _assert_true(json.loads(_extract_payload_blob(on_html))["meta"]["showPlanes"] is True
                                   and json.loads(_extract_payload_blob(off_html))["meta"]["showPlanes"] is False))
    ply_on = sw / "edges_on.ply"
    ply_off = sw / "edges_off.ply"
    export_geometry.write_ply(data, ply_on, binary=True, with_edges=True)
    export_geometry.write_ply(data, ply_off, binary=True, with_edges=False)
    chk.check("PLY with_edges=True 含 element edge 声明",
              lambda: _assert_eq(export_geometry.parse_ply_edge_count(ply_on), EXPECTED_E),
              "否则 --with-ply-edges 是空操作")
    chk.check("PLY with_edges=False 无 element edge",
              lambda: _assert_eq(export_geometry.parse_ply_edge_count(ply_off), 0))
    chk.check("PLY 开/关 edges 产物不同",
              lambda: _assert_true(ply_on.read_bytes() != ply_off.read_bytes()))

    proc_planes = subprocess.run(
        [sys.executable, "-m", "n3d_viz", "--checkpoint", str(ckpt_path),
         "--out-dir", str(sw / "cli_planes"), "--no-plan-planes", "--quiet"],
        cwd=str(_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_subprocess_env(),
    )
    # 先断言进程成功，再读产物：否则失败时 read_text 会抛未捕获的
    # FileNotFoundError（绕过 Checker 收集机制，导致脚本 traceback 崩溃且不落报告）。
    chk.check("CLI --no-plan-planes 进程退出码 == 0",
              lambda: _assert_eq(proc_planes.returncode, 0),
              (proc_planes.stdout or "") + (proc_planes.stderr or ""))
    cli_planes_html = ""
    if proc_planes.returncode == 0:
        cli_planes_html = (sw / "cli_planes" / f"viz_{ckpt_path.stem}.html").read_text(encoding="utf-8")
    chk.check("CLI --no-plan-planes 产物中 showPlanes == false",
              lambda: _assert_true('"showPlanes":false' in cli_planes_html),
              f"returncode={proc_planes.returncode}")
    proc_edges = subprocess.run(
        [sys.executable, "-m", "n3d_viz", "--checkpoint", str(ckpt_path),
         "--out-dir", str(sw / "cli_edges"), "--with-ply-edges", "--quiet"],
        cwd=str(_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_subprocess_env(),
    )
    cli_edges_ply = sw / "cli_edges" / f"viz_{ckpt_path.stem}.ply"
    chk.check("CLI --with-ply-edges 产物含 736 条边",
              lambda: _assert_eq(export_geometry.parse_ply_edge_count(cli_edges_ply), EXPECTED_E),
              f"returncode={proc_edges.returncode}")

    render_smoke: str | None = None

    # ------------------------------------------------- 渲染器逻辑冒烟（Node + DOM 桩）
    print("\n[5b] 渲染器逻辑冒烟（真实执行内联 viewer.js）")
    if shutil.which("node") is None:
        chk.skip("渲染器逻辑冒烟", "未找到 node")
    else:
        render_smoke = chk.check("内联渲染器可执行且投影正确",
                                 lambda: _render_logic_smoke(html, out_path))

    # ------------------------------------------------------------ 异常路径
    print("\n[6] 异常路径：一期产物必须报错并含缺失键名")
    if phase1_path.exists():
        proc = subprocess.run(
            [sys.executable, "-m", "n3d_viz", "--checkpoint", str(phase1_path)],
            cwd=str(_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=_subprocess_env(),
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        chk.check("一期产物退出码非 0", lambda: _assert_true(proc.returncode != 0),
                  f"returncode={proc.returncode}")
        for key in PHASE1_MISSING_KEYS:
            chk.check(
                f"一期产物错误信息含缺失键 '{key}'",
                lambda k=key: _assert_true(k in combined),
                "CLI 错误路径实测",
            )
        proc_missing = subprocess.run(
            [sys.executable, "-m", "n3d_viz", "--checkpoint", "checkpoints/__not_exist__.pt"],
            cwd=str(_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=_subprocess_env(),
        )
        chk.check("路径不存在时退出码非 0", lambda: _assert_true(proc_missing.returncode != 0),
                  f"returncode={proc_missing.returncode}")
        missing_out = (proc_missing.stdout or "") + (proc_missing.stderr or "")
        print("       [debug] not-exist 输出: " + missing_out.strip().replace("\n", " | "))
        chk.check("路径不存在时报错可读",
                  lambda: _assert_true("不存在" in missing_out))
    else:
        chk.skip("一期产物异常路径", f"{phase1_path} 不存在")

    # --------------------------- 越界索引负例（取值域校验的回归防线）
    # 经验教训：topo_index 含负值时 pos[i] 会按 Python 负索引静默取错神经元
    # （图是错的但不报错）；含越界值时抛 IndexError 而非
    # CheckpointSchemaError，CLI 只捕获 CheckpointError，于是以 traceback 崩溃、退出码 1。
    print("\n[6b] 越界索引负例：必须报可读错误、退出码 3、无 traceback")
    bad_dir = out_path / "_bad_index"
    bad_cases = (
        ("topo_index 负值", "topo_index", 5, -1, "topo_index[5]=-1"),
        ("topo_index 越界", "topo_index", 7, 999, "topo_index[7]=999"),
        ("edge_src 负值", "edge_src", 3, -9, "edge_src[3]=-9"),
        ("edge_dst 越界", "edge_dst", 11, 256, "edge_dst[11]=256"),
    )
    for label, key, idx, bad_value, expect_text in bad_cases:
        ckpt_file = _make_bad_index_checkpoint(ckpt_path, bad_dir, key, idx, bad_value)
        proc_bad = subprocess.run(
            [sys.executable, "-m", "n3d_viz", "--checkpoint", str(ckpt_file),
             "--out-dir", str(bad_dir / "out"), "--quiet"],
            cwd=str(_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=_subprocess_env(),
        )
        bad_out = (proc_bad.stdout or "") + (proc_bad.stderr or "")
        chk.check(f"{label} -> 退出码 == 3",
                  lambda p=proc_bad: _assert_eq(p.returncode, 3),
                  bad_out.strip().replace("\n", " | ")[:200])
        chk.check(f"{label} -> 报错含 '{expect_text}'",
                  lambda o=bad_out, t=expect_text: _assert_true(t in o))
        chk.check(f"{label} -> 无 traceback",
                  lambda o=bad_out: _assert_true("Traceback (most recent call last)" not in o
                                                 and "CheckpointSchemaError" not in o))
        chk.check(f"{label} -> 不产生任何产物",
                  lambda b=bad_dir / "out": _assert_eq(
                      len(list(b.glob("viz_*"))) if b.exists() else 0, 0))
        # 负例产物立即删除：它们只是一次性输入，不应残留在 _verify 目录
        ckpt_file.unlink(missing_ok=True)

    # 负例全部结束后断言：不残留任何临时产物、残留体积为 0
    leftover = sorted(bad_dir.rglob("*")) if bad_dir.exists() else []
    leftover_files = [p for p in leftover if p.is_file()]
    leftover_bytes = sum(p.stat().st_size for p in leftover_files)
    chk.check("负例产物已清理（_bad_index/ 无文件残留）",
              lambda: _assert_eq(len(leftover_files), 0),
              f"残留 {len(leftover_files)} 个文件 / {leftover_bytes} 字节")
    chk.check("负例残留体积 == 0 字节",
              lambda: _assert_eq(leftover_bytes, 0),
              "避免 _verify 目录被 ~69MB 临时产物撑大")

    # ------------------------------------------------------------ 静态扫描
    print("\n[7] 零依赖与自包含静态扫描")
    hits = chk.check("n3d_viz 不 import n3d_sphere / n3d_proto",
                     lambda: _assert_list_eq(_source_scan(Path(__file__).resolve().parent), []),
                     "源码正则扫描 ^\\s*(from|import) n3d_(sphere|proto)")
    banned = ("matplotlib", "plotly", "pyvista", "tkinterdnd2", "scipy", "PIL", "pandas")
    imported = _scan_third_party(Path(__file__).resolve().parent)
    chk.check(
        "n3d_viz 未引入 torch/numpy/标准库以外的第三方包",
        lambda: _assert_list_eq(sorted(set(imported) & set(banned)), []),
        f"扫描到的顶层 import：{sorted(set(imported))}",
    )
    req = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
    chk.check("requirements.txt 无 n3d_viz 相关新增依赖",
              lambda: _assert_true(not re.search(r"^\s*(matplotlib|plotly|pyvista|tkinterdnd2)",
                                                 req, re.MULTILINE)),
              "requirements.txt 仅含 torch/torchvision/numpy")

    # ------------------------------------------------------------ GUI 冒烟
    print("\n[8] GUI 冒烟")
    if skip_gui:
        chk.skip("GUI 冒烟", "--skip-gui 指定")
    else:
        chk.check("gui 模块可导入", lambda: _import_gui())
        chk.check(
            "withdraw() 状态下可构造并销毁 Tk 窗口",
            lambda: _gui_smoke(),
            "不进入 mainloop",
        )

    return _finish(chk, report, ckpt, phase1, data, render_smoke)


def _finish(chk: Checker, report: str | None, ckpt: str, phase1: str,
            data, render_smoke: str | None = None) -> int:
    """打印汇总、落报告并返回退出码（0 = 全部通过）。

    Args:
        render_smoke: 渲染器逻辑冒烟的汇总行；非 None 时写进报告的对应小节。
    """
    print("\n" + "=" * 78)
    print(f"汇总：通过 {chk.passed} / 失败 {chk.failed} / 跳过 {chk.skipped}（共 {len(chk.rows)}）")
    print("=" * 78)
    _emit_report(chk, report, ckpt, phase1, data, render_smoke)
    return 0 if chk.failed == 0 else 1



def _render_logic_smoke(html: str, work_dir: Path) -> str:
    """把 HTML 里内联的 viewer.js 与数据提取到临时文件，用 Node + DOM 桩真实执行一次渲染。

    这是不依赖任何第三方浏览器自动化库的功能校验：只要 node 可用即可运行。

    Returns:
        冒烟输出的汇总行（作为断言返回值）。

    Raises:
        AssertionError: node 返回码非 0（存在失败项）。
    """
    blocks = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    assert len(blocks) >= 2, f"HTML 中未找到内联数据与渲染器（得到 {len(blocks)} 个 script 块）"
    js_body = blocks[1]
    marker = "window.N3D_DATA = "
    start = html.index(marker) + len(marker)
    end = html.index(";\n", start)
    data_blob = html[start:end].replace("<\\/", "</")
    work_dir.mkdir(parents=True, exist_ok=True)
    js_path = work_dir / "_viewer_extract.js"
    data_path = work_dir / "_data_extract.json"
    js_path.write_text(js_body, encoding="utf-8", newline="")
    data_path.write_text(data_blob, encoding="utf-8", newline="")
    smoke_js = Path(__file__).resolve().parent / "assets" / "viewer_smoke.js"
    assert smoke_js.exists(), f"缺少渲染器冒烟脚本：{smoke_js}"
    proc = subprocess.run(
        ["node", str(smoke_js), str(js_path), str(data_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=_subprocess_env(),
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    for line in out.strip().splitlines():
        print("       " + line)
    assert proc.returncode == 0, f"node 渲染器冒烟失败（returncode={proc.returncode}）"
    summary = [ln for ln in out.splitlines() if "汇总" in ln]
    # 清理临时提取物，避免污染 _verify 目录
    js_path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)
    return summary[-1].strip() if summary else "ok"


def _extract_payload_blob(html: str) -> str:
    """从 HTML 中取出内联的 window.N3D_DATA JSON 文本。"""
    marker = "window.N3D_DATA = "
    start = html.index(marker) + len(marker)
    end = html.index(";\n", start)
    return html[start:end].replace("<\\/", "</")


def _make_bad_index_checkpoint(
    src_ckpt: Path,
    out_dir: Path,
    key: str,
    index: int,
    value: int,
) -> Path:
    """从合法产物派生一份「指定索引越界」的 checkpoint，供负例断言使用。

    只改一个序列元素（不动形状），因此能精确地把负例定位到「取值域」而非「长度」校验。

    Args:
        src_ckpt: 合法的二期 checkpoint 路径。
        out_dir: 负例产物目录。
        key: 要改写的张量名（``topo_index`` / ``edge_src`` / ``edge_dst``）。
        index: 要改写的下标。
        value: 写入的越界值。

    Returns:
        负例 checkpoint 路径。
    """
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    obj = torch.load(str(src_ckpt), map_location="cpu", weights_only=False)
    sd = obj["model_state_dict"]
    # 只保留抽取所需的契约键，丢掉 syn_dist（16.8MB）等巨型张量：
    # 负例只需触发取值域校验，无需完整权重，产物从 ~18MB 降到 ~30KB。
    slim = {k: sd[k] for k in _REQUIRED_KEYS_FOR_NEGATIVE if k in sd}
    tensor = slim[key].clone()
    tensor[index] = value
    slim[key] = tensor
    path = out_dir / f"bad_{key}_{index}_{value}.pt".replace("-", "m")
    torch.save({"model_state_dict": slim, "config": "", "test_acc": None}, str(path))
    return path


def _subprocess_env() -> dict:
    """构造子进程环境：强制 UTF-8 标准流，避免控制台代码页干扰断言。"""
    import os

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _import_gui() -> str:
    """导入 gui 子模块并返回版本标记。"""
    from n3d_viz import gui

    return f"n3d_viz.gui 可导入，VizApp={gui.VizApp.__name__}"


def _gui_smoke() -> str:
    """在 withdraw 状态下构造 Tk 窗口并立即销毁（不阻塞、不进入 mainloop）。"""
    from n3d_viz import gui

    app = gui.build_smoke_window()
    try:
        assert app.root.winfo_exists()
        widgets = len(app.root.winfo_children())
    finally:
        app.destroy()
    return f"构造成功，顶层子控件 {widgets} 个，已销毁"


def _scan_third_party(module_dir: Path) -> list[str]:
    """收集模块源码中出现的顶层 import 名（用于第三方依赖白名单断言）。"""
    pattern = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)|import\s+([A-Za-z_][\w.]*))", re.MULTILINE)
    names: list[str] = []
    for path in sorted(module_dir.rglob("*.py")):
        for m in pattern.finditer(path.read_text(encoding="utf-8")):
            name = m.group(1) or m.group(2)
            names.append(name.split(".")[0])
    return names


def _parse_inline_payload(html: str) -> dict[str, Any]:
    """从 HTML 中取出内联的 ``window.N3D_DATA`` JSON 负载。"""
    marker = "window.N3D_DATA = "
    start = html.index(marker) + len(marker)
    end = html.index(";\n", start)
    blob = html[start:end].replace("<\\/", "</")
    return json.loads(blob)


# ---------------------------------------------------------------- 断言辅助
def _assert_eq(got: Any, want: Any) -> str:
    assert got == want, f"期望 {want}，实际 {got}"
    return f"{got} == {want}"


def _assert_list_eq(got: Any, want: Any) -> str:
    got_list = list(got)
    want_list = list(want)
    assert got_list == want_list, f"期望 {want_list}，实际 {got_list}"
    return f"{got_list} == {want_list}"


def _assert_true(cond: Any) -> str:
    assert cond, "条件为假"
    return "True"


def _assert_lt(got: float, limit: float) -> str:
    assert got < limit, f"{got} 不小于 {limit}"
    return f"{got} < {limit}"


def _assert_le(got: float, limit: float) -> str:
    assert got <= limit, f"{got} 大于 {limit}"
    return f"max|diff|={got:.3e} <= {limit:.1e}"


def _assert_leq(got: float, limit: float) -> str:
    return _assert_le(got, limit)


def _emit_report(chk: Checker, report: str | None, ckpt: str, phase1: str,
                 data: core.TopologyData | None,
                 render_smoke: str | None = None) -> None:
    """把检查清单写成 Markdown 报告（仅在 --report 指定时落盘）。"""
    if not report:
        return
    lines = [
        "# n3d_viz 验证报告（真实执行产物）",
        "",
        f"- 基准 checkpoint：`{ckpt}`",
        f"- 异常路径 checkpoint：`{phase1}`",
        f"- 汇总：通过 {chk.passed} / 失败 {chk.failed} / 跳过 {chk.skipped}",
        "",
    ]
    if data is not None:
        lines += [
            "## 数据源实测值",
            "",
            f"- N = {data.n_neurons}，E = {data.n_edges}，K = {data.n_layers}",
            f"- seed = {data.config.get('seed')}，test_acc = {data.test_acc}",
            f"- 层规模 = {'/'.join(str(c) for c in data.layer_counts)}",
            f"- 层入边 = {'/'.join(str(c) for c in data.layer_edge_counts)}",
            f"- S_in = {data.n_s_in}，S_out = {data.n_s_out}",
            f"- 阈值统计 = {data.edge_threshold_stats((0.05, 0.10, 0.20, 0.30, 0.50))}",
            f"- |w| min/median/max = {data.weight_extremes()[:3]}",
            f"- 连接密度 = {data.conn_density:.6f}",
            f"- syn_dist 体积 = {data.syn_dist_bytes} 字节（未嵌入 HTML）",
            "",
        ]
    if render_smoke:
        lines += [
            "## 渲染器逻辑冒烟（Node + DOM 桩 真实执行内联 viewer.js）",
            "",
            f"- {render_smoke}",
            "- 断言项：神经元 arc 计数 == N、绘制线段数 >= 阈值内边数、投影 bbox 有限且落在画布附近、",
            "  投影质心接近画布中心、阈值滑块联动、悬停命中并填充详情、图例色块数 == K+3",
            "",
        ]
    lines += ["## 断言清单", "", chk.to_markdown(), ""]
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines), encoding="utf-8")
    print(f"[报告] 已写入 {report}")


def main(argv: list[str] | None = None) -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="n3d_viz 零依赖验证脚本")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--phase1-checkpoint", default=DEFAULT_PHASE1_CKPT)
    parser.add_argument("--out-dir", default="checkpoints/n3d_viz/_verify")
    parser.add_argument("--report", default=None, help="Markdown 报告输出路径")
    parser.add_argument("--skip-gui", action="store_true", help="跳过 GUI 冒烟（无图形环境时）")
    args = parser.parse_args(argv)
    return verify(
        ckpt=args.checkpoint,
        phase1=args.phase1_checkpoint,
        out_dir=args.out_dir,
        report=args.report,
        skip_gui=args.skip_gui,
    )


if __name__ == "__main__":  # pragma: no cover - 脚本直跑分支
    raise SystemExit(main())