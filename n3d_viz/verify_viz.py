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
9.  ``n3d_viz`` 源码不 import ``n3d_sphere`` / ``n3d_proto`` / ``n3d_shape``（静态扫描）
10. GUI 冒烟：模块可导入且能在 ``withdraw()`` 状态下构造并销毁 Tk 窗口
11. ``[2a]`` 层配色可扩展：K ∈ [1,64] 去重数 == K，K <= 9 与既有 9 色逐字节相同，
    HTML 层色与 PLY 层色同源，且 `hex_to_rgb` 的输入契约（非法文本抛 ``ValueError``、
    分量恒落在 0..255）成立
12. ``[2b]`` **非规整几何合成组**：就地构造 5 类几何完全不规整的合法 state_dict
    （随机点云 / 螺旋线 / 每层 1 个神经元 / K=1 单层 / K=33），逐类断言顶点数、
    ``l`` 行数、层数、层色去重数、坐标逐位一致、payload 长度自洽、产物名派生
13. ``[2c]`` **真实异构几何组**：``checkpoints/n3d_shape/*.pt``（5 个真实非球几何
    产物）的泛化不变量；目录缺失时明确 SKIP 并计入报告，不静默跳过
14. ``[2d]`` **零回归锚点**：二期正式产物 ``checkpoints/n3d_viz/viz_model.{html,ply,obj}``
    的 SHA256 与改动前逐位相同
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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

#: 真实异构几何样本目录（`n3d_shape` 的 5 个产物：sphere / cube / cylinder λ=0.5/1/2）。
#: 缺失时 [2c] 组明确 SKIP 并计入报告，不静默跳过。
SHAPE_DIR = "checkpoints/n3d_shape"

#: 零回归锚点：改动前二期正式产物的 SHA256（[2d] 组逐位比对）。
#: 记录时间 = 本次几何无关化改动之前（checkpoints/n3d_viz/viz_model.* 当时字节
#: 88,521 / 4,120 / 15,695）。
#:
#: **复现口径（必须遵守，见 [I2]）**：HTML 内嵌的 ``meta.checkpoint`` 记录的是
#: **调用时给出的路径字符串**，因此字节级比对只在「同一调用形式」下成立。锚点是
#: 用**相对路径** ``checkpoints/n3d_sphere/model.pt`` 产出的；改用绝对路径重渲会让
#: HTML 仅因该字段多出若干字节而比对失败（PLY / OBJ 无色无路径字段，仍逐字节相同）。
#: 故 [2d] 的重渲断言一律使用下面的 :data:`ANCHOR_RERENDER_CKPT` 相对路径常量，
#: 不使用命令行传入的 ``--checkpoint``。
PHASE2_ANCHOR_DIR = "checkpoints/n3d_viz"
PHASE2_ANCHOR_SHA256: dict[str, str] = {
    # 旧值 15A80EBBF2FD586BFB3C4C41E25F79F0E3E8F0C19D3B60AE22BE3F296EDA518C（88,521 字节）：
    # 2026-09-27「相机单一事实来源」轮重基线（viewer.js 新增 1 行 window.__n3d_cam = cam;），
    # 详见 ANCHOR_REBASE_LOG；PLY / OBJ 的哈希**重基线前后逐字节相同**。
    "viz_model.html": "A5FEE937B8023B986497F8D7C9E6F34D8E3472C9DA03F98F69020251BA4C4F87",
    "viz_model.ply": "9A097D16306160F95C9E15826CB11ABED57C6809F2904660B700573889398801",
    "viz_model.obj": "1F594ECF466E28F751C174A85A8A4E5459A304973F6266B3E11C567C25A09FED",
}

#: 零回归锚点对应的字节数（与 :data:`PHASE2_ANCHOR_SHA256` 同源、一并比对）。
#:
#: **2026-09-27 相机单一事实来源轮的锚点重基线（已获用户批准）**：为修复
#: 「旋转神经元时 FC 叠加层不跟随」，``assets/viewer.js`` 新增 1 行
#: ``window.__n3d_cam = cam;``。该行使**每一份**产物的 HTML 内联渲染器源码变大，
#: 故所有 HTML 锚点必然变化；**PLY / OBJ 不受影响、必须逐字节不变**
#: （几何与写出路径与本次改动无关）。口径因此放宽为
#: 「**PLY/OBJ 逐字节不变 + HTML 锚点重新基线（登记旧→新与原因）**」。
#: 旧值登记在 :data:`ANCHOR_REBASE_LOG`，PLY/OBJ 的不变证据在 :data:`PLY_OBJ_PRE_REBASE`。
PHASE2_ANCHOR_BYTES: dict[str, int] = {
    "viz_model.html": 88741,   # 旧 88,521（+220 字节 = 新增那一行的注释+代码）
    "viz_model.ply": 4120,     # 重基线前后**不变**
    "viz_model.obj": 15695,    # 重基线前后**不变**
}

#: [2d] **代码回归层**重渲用的 checkpoint —— 必须是**相对仓库根**的路径字符串，
#: 以与锚点产物内嵌的 ``meta.checkpoint`` 形式一致（原因见上方口径说明）。
ANCHOR_RERENDER_CKPT = "checkpoints/n3d_sphere/model.pt"

#: [2d] **锚点组表**（2026-09-27 新增）：每组 = 一个 checkpoint + 其派生的三件套恒定值。
#:
#: 为什么要按「组」而不再只有单一二期锚点：单组锚点只覆盖 ``K <= 9`` 的层色路径；
#: 一旦该组被替换或删除，``K > 9``（走色相扩展色板）的代码回归就会**失去承重覆盖**
#: ——历史上正是「9 色 + ``k % 9`` 循环」在新几何下退化为重复色。故显式登记三组，
#: 并由 :func:`_check_anchor_groups_cover_k` 断言**必须同时覆盖 K<=9 与 K>9 两类**。
#: 每组的 ``ckpt`` 用**相对路径**（与字节级复现口径一致）。
ANCHOR_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "name": "二期 model.pt（K=9）",
        "ckpt": ANCHOR_RERENDER_CKPT,
        "kind": "lt9",
        "sha256": PHASE2_ANCHOR_SHA256,
        "bytes": PHASE2_ANCHOR_BYTES,
    },
    {
        "name": "三期 sphere N256（config 无 fc_dim，K=9）",
        "ckpt": (
            "checkpoints/n3d_shape/"
            "full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt"
        ),
        "kind": "lt9",
        "sha256": {
            "viz_full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.html":
                "63B9CD4D4CEDB24925173429A200A28F2919F4205BA96E06DDE6999B7940125C",
            "viz_full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.ply":
                "28034930AAC972C7F8D1CC5110FACF7322A8C709C2EC1A46909DF8CC32777F61",
            "viz_full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.obj":
                "5893E8825BE53C75DE39A344E68A5BA677EF57FA6A350F284BC2CA0C5C66BA5B",
        },
        "bytes": {
            "viz_full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.html": 88854,
            "viz_full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.ply": 4177,
            "viz_full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.obj": 15752,
        },
    },
    {
        "name": "三期 cylinder λ=2 N256（config 无 fc_dim，**K=15 -> K>9 路径**）",
        "ckpt": (
            "checkpoints/n3d_shape/"
            "full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt"
        ),
        "kind": "gt9",
        "sha256": {
            "viz_full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.html":
                "9606F91E52AC405349C504D43389190D294D84F2DF9DB66C1E90CFC528041706",
            "viz_full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.ply":
                "C4ED6E86461E7BCAAD99BBFF2E1070F6CB7239417083EE35063275E8D5FFAC9F",
            "viz_full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.obj":
                "F650ADE9662AAD9D0954A344A304AF8EF2FE1ECBF540CBD804E03331F6C53997",
        },
        "bytes": {
            "viz_full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.html": 88649,
            "viz_full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.ply": 4182,
            "viz_full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.obj": 15613,
        },
    },
)

#: 「**PLY / OBJ 逐字节不变**」的承重证据表（本次放宽后的零回归口径）。
#:
#: 结构：``[(清单名, ckpt 相对路径, {三件套文件名: (sha256, bytes)}), ...]``。
#: 覆盖 **4 组产物 × 2 个文件 = 8 项**（3 个无 FC 锚点组 + 1 个 FC 产物）。
#: :func:`_check_ply_obj_unchanged` 会用**当前代码**重新渲染每一组并与这里的恒定值
#: 逐项比对（因此这不是「只比常量」，而是真实的重渲回归断言）；**只比 PLY/OBJ**：
#: HTML 因 ``viewer.js`` 新增 1 行已按批准重基线（登记在 :data:`ANCHOR_REBASE_LOG`）。
#:
#: **承重前提（与 [2d] 锚点组同口径，必须一并声明）**：这 8 项的恒定值取自
#: ``HEAD = 927d32f``（「相机单一事实来源」轮之前的最后一次提交，即 ``viewer.js``
#: 新增 ``window.__n3d_cam = cam;`` **之前**）当时用真实 checkpoint 渲染出的产物。
#: 因此本组**仅在「上游 checkpoints 产物逐字节不变」的前提下才具承重意义**：
#: 一旦上游把 ``checkpoints/`` 破坏性重建 / 清空重训（几何、边集、层结构跟着变），
#: 这 8 项会**集体 FAIL**，而失败原因**不在渲染器**，不能据此判定 n3d_viz 回归。
#: 判读顺序固定为：**先**核对上游 checkpoint 是否仍是同一份产物，**再**怀疑渲染路径。
#: 换言之：本常量测量的是「渲染器 + 上游产物」这一对的稳定性，不是渲染器单独的稳定性。
PLY_OBJ_BASELINE: tuple[tuple[str, str, dict[str, tuple[str, int]]], ...] = (
    ("二期 model.pt", ANCHOR_RERENDER_CKPT, {
        "viz_model.ply": (
            "9A097D16306160F95C9E15826CB11ABED57C6809F2904660B700573889398801", 4120),
        "viz_model.obj": (
            "1F594ECF466E28F751C174A85A8A4E5459A304973F6266B3E11C567C25A09FED", 15695),
    }),
    ("三期 sphere N256（K=9）",
     "checkpoints/n3d_shape/"
     "full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt", {
        "viz_full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.ply": (
            "28034930AAC972C7F8D1CC5110FACF7322A8C709C2EC1A46909DF8CC32777F61", 4177),
        "viz_full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.obj": (
            "5893E8825BE53C75DE39A344E68A5BA677EF57FA6A350F284BC2CA0C5C66BA5B", 15752),
    }),
    ("三期 cylinder λ=2 N256（K=15，K>9 路径）",
     "checkpoints/n3d_shape/"
     "full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt", {
        "viz_full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.ply": (
            "C4ED6E86461E7BCAAD99BBFF2E1070F6CB7239417083EE35063275E8D5FFAC9F", 4182),
        "viz_full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.obj": (
            "F650ADE9662AAD9D0954A344A304AF8EF2FE1ECBF540CBD804E03331F6C53997", 15613),
    }),
    ("FC 产物（fc_dim=-1，H=825）",
     "checkpoints/n3d_shape/"
     "full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.pt", {
        "viz_full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.ply": (
            "25C1008C7C412F0AFDBEC359186548A2B0317523C7AE55557DBC77C2D44C52E5", 81681),
        "viz_full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.obj": (
            "4E0C2A2F125F7C4F637C7ECB3F5C0F3498065B39B8438EB045C03DA7EA75A503", 154372),
    }),
)

#: 兼容别名：只保留「文件名 -> (sha, bytes)」的扁平视图（供报告/文档引用）。
PLY_OBJ_PRE_REBASE: dict[str, tuple[str, int]] = {
    fname: want for _n, _c, files in PLY_OBJ_BASELINE for fname, want in files.items()
}


#: 锚点重基线时**已登记**的 HTML 变更（旧值 + 原因），供 [2d] 断言「登记齐备」。
#: 维护约定：任何使 HTML 锚点变化且**并非回归**的改动，都必须在此登记旧值/原因，
#: 并同步 README 与 current_spec；**PLY / OBJ 若变化一律视为回归**（须定位根因）。
ANCHOR_REBASE_LOG: dict[str, dict[str, Any]] = {
    "viz_model.html": {
        "old_sha256": "15A80EBBF2FD586BFB3C4C41E25F79F0E3E8F0C19D3B60AE22BE3F296EDA518C",
        "old_bytes": 88521,
        "reason": "修复「旋转时 FC 叠加层不跟随」：viewer.js 新增 1 行 window.__n3d_cam = cam;",
    },
    "viz_full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.html": {
        "old_sha256": "F9535DAD22FA1C01436CDC251B729CE61BB7B5DAF491F0BAD9BA2BF3FDBB8C6A",
        "old_bytes": 88634,
        "reason": "同上（viewer.js 新增 1 行，内联渲染器源码变大）",
    },
    "viz_full_shapecylinder_a2_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.html": {
        "old_sha256": "FA5D5426C2064D79B09B5ED69C23267832861273C794F60A652E0003C7508C9E",
        "old_bytes": 88429,
        "reason": "同上（viewer.js 新增 1 行，内联渲染器源码变大）",
    },
}

#: FC 产物 HTML 的**改动前**值（面板尺度调整后、相机修复前）。它不参与 [2d]
#: 的三组锚点（那三组都无 FC），仅用于文档溯源。
FC_HTML_PRE_CAM_FIX: tuple[str, int] = (
    "34CDDB922807787D3071043D46410545BF30DE923C84A559E2D51295D6A10A87", 591709)

#: [2d] 代码回归层的临时输出目录名（位于 ``--out-dir`` 之下，比对完即清理）。
ANCHOR_RERENDER_DIRNAME = "_anchor_rerender"

#: [I1] 撞色回退分支的**必撞色**探针规模：色相间隔 1/1536，量化到 8 位后必然撞色。
#: 该分支在 ``K <= 64`` 区间内不可达（实测撞色数 0），故此断言是该分支的唯一覆盖。
COLLISION_PROBE_K = 1536

#: [2b] 合成几何样本的随机种子（边集与坐标随 seed 变化，报告中必须标注）。
SYNTH_SEED = 20250925

#: [2f] **有 FC** 的真实产物（`config.fc_dim == -1` 表示宽度跟随 N）。
#: 实测（见 README §7）：N=825、seed=42、H=fc_width=825、|S_in|=582、|S_out|=588，
#: 默认 k=3 时抽样 (582+588)×3 = 3,510 条；proj_weight 582×825=480,150、
#: fc_out_weight 825×588=485,100 条（**参数口径**，与抽样条数分开标注）。
FC_PRODUCT_CKPT = (
    "checkpoints/n3d_shape/"
    "full_shapesphere_N825_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_fc-1_s42_fc_align.pt"
)

#: [2f] 三期**未启用 FC** 的真实产物（`config` 中**没有** `fc_dim` 键）。
THIRD_PARTY_NO_FC_CKPT = (
    "checkpoints/n3d_shape/"
    "full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isany_rsany_s42.pt"
)

#: [2f] 抽样口径声明的必需子串（HTML meta / OBJ 伴随说明都必须含它）。
#: 抽样图若被误读为「全连接结构」即为错误信息，故用断言把守。
FC_NOT_ALL_TEXT = "非全部连接"

#: [2f] OBJ / PLY 中标识「无 FC」的关键字（无 FC 产物里一个都不许出现）。
FC_ONLY_TOKENS = ("fc_node", "fc_edge", "n3d_viz_fc_sampled_edges")


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
    ``from n3d_proto`` / ``import n3d_shape`` / ``from n3d_shape`` 形式的语句。
    本模块对几何零假设，`n3d_shape` 的产物只作为**输入数据**被读取，
    其源码同样不得被 import（与 `n3d_sphere` / `n3d_proto` 同等约束）。
    """
    pattern = re.compile(
        r"^\s*(?:from|import)\s+n3d_(?:sphere|proto|shape)\b", re.MULTILINE
    )
    hits: list[tuple[str, int, str]] = []
    for path in sorted(module_dir.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            hits.append((str(path.relative_to(module_dir.parent)), line_no, m.group(0).strip()))
    return hits


def _sha256(path: Path) -> str:
    """返回文件的 SHA256（大写十六进制），供零回归锚点断言使用。"""
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _guard(err: str | None) -> None:
    """[2b]/[2c] 组：前置步骤失败时把错误原样抛出，作为该条断言的真实失败原因。"""
    if err is not None:
        raise AssertionError(err)


def _guarded(err: str | None, fn: Callable[[], Any]) -> Callable[[], Any]:
    """把「前置错误」与「真正断言」包成一个无参可调用对象（供 Checker 逐条执行）。"""
    def run() -> Any:
        _guard(err)
        return fn()
    return run


# --------------------------------------------------------------- [2a] 层配色
def _check_palette_dedup() -> str:
    """K ∈ [1,64] 的层色去重数必须恒等于 K（几何无关化后 K 可任意大）。"""
    bad = [k for k in range(1, 65) if len(set(core.layer_palette_hex(k))) != k]
    assert not bad, f"以下 K 的层色去重数 != K：{bad}"
    return "K=1..64 逐个检查，去重数均 == K"


def _check_palette_base_prefix() -> str:
    """K <= 9 必须逐字节等于既有 9 色表的前 K 个（既有产物零回归）。"""
    assert core.layer_palette_hex(0) == []
    for k in range(1, 10):
        got = core.layer_palette_hex(k)
        assert got == list(core.LEVEL_PALETTE_BASE[:k]), f"K={k}: {got}"
    return "K=0..9 层色 == LEVEL_PALETTE_BASE 前 K 个（逐字节不变）"


def _check_palette_same_source() -> str:
    """HTML 层色（hex）与 PLY 层色（RGB）必须同源：hex_to_rgb 后逐一相等。"""
    assert export_geometry.LAYER_COLORS == tuple(core.layer_palette_rgb(9)), "PLY 基础色表与 core 不同源"
    for k in (1, 9, 10, 15, 33, 64):
        hexes = core.layer_palette_hex(k)
        assert [core.hex_to_rgb(h) for h in hexes] == core.layer_palette_rgb(k), f"K={k} 不同源"
    return "K ∈ {1,9,10,15,33,64}：hex 转换后 == RGB 色板；出口表 == core 基础表"


def _check_palette_collision_fallback() -> str:
    """[I1] `core._hue_palette_rgb` 的**撞色回退分支**覆盖（该分支在 K <= 64 时不可达）。

    背景：在断言覆盖的 ``K ∈ [10, 64]`` 范围内，纯色相扩展的撞色数实测为 0，
    即 ``candidate not in used`` 恒为真、``step`` 永远停在 0，回退分支（色相微移 +
    明度微降 + `_PALETTE_SEARCH_LIMIT`）从未被真正执行 —— 属「不可达但未断言」的
    防御代码，一旦有缺陷则 158 条断言一条都拓不到。

    本断言用 ``K = COLLISION_PROBE_K``（1536，色相间隔 1/1536 < 8 位量化步长）
    **主动构造必然撞色**，三重把守：

    1. 前提成立：纯色相扩展在该 K 下撞色数 > 0（否则本断言无意义，直接报错）；
    2. 回退真的发生了：统计 ``_hsv_to_rgb_bytes`` 的调用次数，必须 > K
       （无回退时恰好 == K）；这排除了「撞色消失、断言空转」的假绿灯；
    3. 回退结果正确：最终去重数 == K，且公开入口 ``layer_palette_hex(K)`` 同样成立。

    ``_hsv_to_rgb_bytes`` 的替换在 ``finally`` 中恢复，不影响后续断言。

    Returns:
        实测摘要（撞色个数、候选调用次数、回退次数、最终去重数）。
    """
    k = COLLISION_PROBE_K
    # 1) 前提：纯色相扩展（无回退）在该 K 下必然撞色
    naive = [
        core._hsv_to_rgb_bytes((i / float(k)) % 1.0, core._PALETTE_SAT, core._PALETTE_VALUE)
        for i in range(k)
    ]
    collisions = k - len(set(naive))
    assert collisions > 0, (
        f"K={k} 的纯色相扩展未出现撞色（撞色数 {collisions}），无法覆盖回退分支；"
        "请调整 COLLISION_PROBE_K"
    )
    # 2) 回退确实被执行：候选调用次数必须 > K
    calls = {"n": 0}
    original = core._hsv_to_rgb_bytes

    def _counting(hue: float, sat: float, val: float) -> tuple[int, int, int]:
        calls["n"] += 1
        return original(hue, sat, val)

    core._hsv_to_rgb_bytes = _counting  # type: ignore[assignment]  # 白盒计数钩子
    try:
        palette = core._hue_palette_rgb(k)
    finally:
        core._hsv_to_rgb_bytes = original  # type: ignore[assignment]
    retries = calls["n"] - k
    assert retries > 0, (
        f"候选调用 {calls['n']} 次 == K={k}，说明撞色回退分支实际未被触发（断言将空转）"
    )
    # 3) 回退结果正确
    assert len(set(palette)) == k, f"回退后去重数 {len(set(palette))} != K={k}"
    assert len(set(core.layer_palette_hex(k))) == k, f"公开入口 layer_palette_hex({k}) 去重数 != K"
    return (
        f"K={k}：纯色相扩展撞色 {collisions} 个（前提成立）；候选调用 {calls['n']} 次、"
        f"回退 {retries} 次；最终去重数 == {k}"
    )


def _check_hex_contract() -> str:
    """`hex_to_rgb` 的输入契约：非法文本必须抛 ``ValueError``，分量必须落在 0..255。

    为何要单独把守：``int(seg, 16)`` 接受前导正负号（``"#-12345"`` 会静默得到负分量），
    ``str.lstrip("#")`` 是字符集合语义（``"##4e8cff"`` 会被静默接受）——两者都让
    「非 6 位十六进制即报错」+「分量 0..255」的契约出现缝隙。
    """
    good = {
        "#4e8cff": (78, 140, 255),
        "4e8cff": (78, 140, 255),
        "#4E8CFF": (78, 140, 255),
        "  #4e8cff  ": (78, 140, 255),
        "#000000": (0, 0, 0),
        "#ffffff": (255, 255, 255),
    }
    for text, want in good.items():
        got = core.hex_to_rgb(text)
        assert got == want, f"{text!r}：期望 {want}，实际 {got}"
    bad = (
        "#-12345", "#+12345", "#4e8c f", "# 4e8cf", "#xyz", "xyz",
        "#4e8cf", "#4e8cfff", "", "#", "0x4e8cff", "#4e8cfg", "#zzzzzz",
        "##4e8cff", "###4e8cff",
    )
    for text in bad:
        try:
            core.hex_to_rgb(text)
        except ValueError:
            continue
        raise AssertionError(f"非法文本 {text!r} 未被拒绝（未抛 ValueError）")
    for k in range(1, 65):
        for rgb in core.layer_palette_rgb(k):
            assert all(0 <= c <= 255 for c in rgb), f"K={k} 出现越界分量 {rgb}"
    return f"正向 {len(good)} 项、反向 {len(bad)} 项、K=1..64 分量均 ∈ [0,255]"


# ------------------------------------------- [2b] 非规整几何合成组（永久有效）
def _synth_geometries(seed: int = SYNTH_SEED) -> list[tuple[str, list[tuple[float, float, float]],
                                                           list[tuple[int, int]], list[int]]]:
    """构造 5 类**几何上完全不规整**的合成样本（不依赖任何既有产物）。

    每项返回 ``(名称, 坐标, 边(源,目标), 层规模)``。坐标与边只由本函数与 ``seed``
    决定，因此该组是永久有效的回归防线：即使所有既有产物被删除也依然可跑。

    * ``random_cloud``：三维均匀随机点云 + 随机边 —— 破除「晶格 / FCC」假设
    * ``helix``：螺旋线点云 —— 破除「凸包 / 中心对称」假设
    * ``one_per_layer``：每层仅 1 个神经元（K == N） —— 破除「层内并行度」假设
    * ``single_layer``：K=1 单层 —— 破除「多层」假设
    * ``k33``：K=33 —— 破除「K <= 9」假设（走色相扩展色板）
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    samples: list[tuple[str, list[tuple[float, float, float]], list[tuple[int, int]], list[int]]] = []

    def random_edges(n: int, e: int) -> list[tuple[int, int]]:
        """随机边：端点独立均匀采样，剔除自环（保证边表合法且形状不规整）。"""
        out: list[tuple[int, int]] = []
        while len(out) < e:
            s = int(rng.integers(0, n))
            d = int(rng.integers(0, n))
            if s != d:
                out.append((s, d))
        return out

    # 1) 随机点云（N=120, E=200, K=6）
    pts = rng.uniform(-1.0, 1.0, size=(120, 3))
    samples.append((
        "random_cloud",
        [(float(x), float(y), float(z)) for x, y, z in pts],
        random_edges(120, 200),
        [20, 20, 20, 20, 20, 20],
    ))

    # 2) 螺旋线（N=96, E=150, K=8）：非凸、无中心对称、无晶格
    n_helix = 96
    helix = [
        (float(math.cos(6.0 * math.pi * i / n_helix)),
         float(math.sin(6.0 * math.pi * i / n_helix)),
         float(2.0 * i / n_helix - 1.0))
        for i in range(n_helix)
    ]
    samples.append(("helix", helix, random_edges(n_helix, 150), [12] * 8))

    # 3) 每层仅 1 个神经元（N=24, K=24, E=40）
    pts = rng.uniform(-2.0, 2.0, size=(24, 3))
    samples.append((
        "one_per_layer",
        [(float(x), float(y), float(z)) for x, y, z in pts],
        random_edges(24, 40),
        [1] * 24,
    ))

    # 4) K=1 单层（N=40, E=60）
    pts = rng.uniform(-1.0, 1.0, size=(40, 3))
    samples.append((
        "single_layer",
        [(float(x), float(y), float(z)) for x, y, z in pts],
        random_edges(40, 60),
        [40],
    ))

    # 5) K=33（N=64, E=90）：31 层各 2 个 + 末尾 2 层各 1 个 = 64
    pts = rng.uniform(-1.5, 1.5, size=(64, 3))
    samples.append((
        "k33",
        [(float(x), float(y), float(z)) for x, y, z in pts],
        random_edges(64, 90),
        [2] * 31 + [1, 1],
    ))

    return samples


def _make_synthetic_checkpoint(
    out_dir: Path,
    name: str,
    positions: list[tuple[float, float, float]],
    edges: list[tuple[int, int]],
    layer_sizes: list[int],
    seed: int = SYNTH_SEED,
) -> Path:
    """就地构造一个几何上完全不规整、但**符合二期 schema** 的 checkpoint。

    做法与既有负例一致：只保留 :data:`core.REQUIRED_KEYS` 契约键（丢弃 ``syn_dist``
    等巨型张量），单个产物约数十 KB，跑完由调用方立即删除，不污染 ``_verify/``。

    「层」= 同时计算的神经元分组：这里按 ``layer_sizes`` 顺序切分 ``topo_index``，
    与几何形状无关（不假设晶格 / 凸包 / 层内并行度）。

    Args:
        out_dir: 临时目录。
        name: 样本名（决定 checkpoint 文件名，进而决定产物名）。
        positions: N 个三维坐标。
        edges: E 条 ``(src, dst)`` 边。
        layer_sizes: K 个层的神经元数（须合计为 N）。
        seed: 生成边权用的种子。

    Returns:
        合成 checkpoint 路径。
    """
    import torch

    n = len(positions)
    e = len(edges)
    assert sum(layer_sizes) == n, f"层规模之和 {sum(layer_sizes)} != N {n}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_t = torch.tensor(positions, dtype=torch.float32)
    src_t = torch.tensor([s for s, _ in edges], dtype=torch.long)
    dst_t = torch.tensor([d for _, d in edges], dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    # 边权与边长按真实口径生成：权重 ∈ [-1,1]（有符号），边长取端点欧氏距离。
    weight_t = torch.rand(e, generator=g) * 2.0 - 1.0
    dist_t = (pos_t[src_t] - pos_t[dst_t]).norm(dim=1)
    topo_index = torch.arange(n, dtype=torch.long)

    level_nodes: list[list[int]] = []
    cursor = 0
    for size in layer_sizes:
        level_nodes.append([cursor, cursor + size])
        cursor += size

    # 入边按层均分（只需自洽，不参与渲染正确性判定）。
    level_edges: list[list[int]] = []
    base, rem = divmod(e, len(layer_sizes))
    cursor = 0
    for k in range(len(layer_sizes)):
        cnt = base + (1 if k < rem else 0)
        level_edges.append([cursor, cursor + cnt])
        cursor += cnt

    in_deg = torch.zeros(n, dtype=torch.long)
    out_deg = torch.zeros(n, dtype=torch.long)
    for s, d in edges:
        out_deg[s] += 1
        in_deg[d] += 1

    sd = {
        "neuron_pos": pos_t,
        "edge_src": src_t,
        "edge_dst": dst_t,
        "edge_weight": weight_t,
        "edge_dist": dist_t,
        "in_scope_mask": torch.tensor([i % 3 != 0 for i in range(n)]),
        "out_scope_mask": torch.tensor([i % 5 != 0 for i in range(n)]),
        "topo_index": topo_index,
        "level_node_reach": torch.tensor(level_nodes, dtype=torch.long),
        "level_edge_reach": torch.tensor(level_edges, dtype=torch.long),
        "in_degree": in_deg,
        "out_degree": out_deg,
    }
    path = out_dir / f"{name}.pt"
    torch.save(
        {"model_state_dict": sd, "config": {"seed": seed, "synthetic": True}, "test_acc": None},
        str(path),
    )
    return path


#: [2e] 冻结的 CLI 选项表面（`build_parser()` 暴露的全部 option string）。
#: 新增/删除 CLI 选项都必须显式更新本清单，并确认该选项是否影响「渲染默认形式」：
#: 若影响，必须同步 :data:`n3d_viz.core.DEFAULT_WRITE_OPTIONS` 与 ``[2d]`` 的锚点语义。
CLI_OPTION_STRINGS: tuple[str, ...] = (
    "--checkpoint", "-c",
    "--out-dir", "-d",
    "--out", "-o",
    "--threshold", "-t",
    "--no-plan-planes",
    "--ply-ascii",
    "--with-ply-edges",
    # 两端全连接包裹的抽样口径（默认 3，范围 1..8；仅 fc_dim != 0 的产物生效）。
    # 它**参与**写出参数集 `core.DEFAULT_WRITE_OPTIONS`，因此改动其默认值会同时
    # 触发 [2e] 与 [2d] 的锚点断言 —— 这正是该清单要暴露的影响面。
    "--fc-top-k", "-k",
    "--quiet", "-q",
    "--help", "-h",
)


def _check_cli_default_options() -> str:
    """[2e] CLI 默认形式的写出参数集必须逐项等于 :data:`core.DEFAULT_WRITE_OPTIONS`。

    这是「锚点 = CLI 默认形式的产物」这一语义的**承重**把守：改动 ``__main__`` 的任一
    相关默认值（阈值、``--ply-ascii`` / ``--with- ply-edges`` / ``--no-plan-planes``
    的默认语义）都会让本条立即 FAIL，而不是让 ``[2d]`` 的锚点静默漂移。
    """
    from n3d_viz import __main__ as cli

    got = cli.write_options_from_args(cli.build_parser().parse_args([]))
    want = dict(core.DEFAULT_WRITE_OPTIONS)
    assert got == want, (
        f"CLI 默认形式的写出参数集与 core.DEFAULT_WRITE_OPTIONS 不一致：{got} != {want}；"
        "改动 CLI 默认值时必须同步确认 [2d] 的零回归锚点是否仍代表同一套「默认形式」"
    )
    return f"argparse([]) 映射 == {got}"


def _check_render_default_options() -> str:
    """[2e] ``core.render_default`` 实际使用的参数集必须等于 ``DEFAULT_WRITE_OPTIONS``。

    白盒做法：临时把 :func:`core.write_outputs` 换成记录实参的桩（因此**不落盘**），
    调用 ``render_default`` 后逐一比对记录到的关键字实参；替换在 ``finally`` 中恢复。
    这样即使日后有人在 ``render_default`` 内部重新手写参数（而非展开
    ``DEFAULT_WRITE_OPTIONS``），本条也会立刻 FAIL。
    """
    captured: dict[str, Any] = {}
    original = core.write_outputs

    def _capture(data: Any, out_dir: Any = None, out: Any = None,
                 log: Any = None, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        captured["_out_dir"] = out_dir
        captured["_out"] = out
        return {"captured": True}

    core.write_outputs = _capture  # type: ignore[assignment]  # 白盒捕获钩子
    try:
        # data 直接给哨兵对象：render_default 不会再加载 checkpoint
        result = core.render_default("__not_used__.pt", out_dir="__nowhere__", data=object())
    finally:
        core.write_outputs = original  # type: ignore[assignment]
    assert result == {"captured": True}, "render_default 未走 core.write_outputs"
    focused = {k: v for k, v in captured.items() if not k.startswith("_")}
    want = dict(core.DEFAULT_WRITE_OPTIONS)
    assert focused == want, (
        f"render_default 实际传给 write_outputs 的参数集 {focused} != {want}；"
        "该函数必须展开 core.DEFAULT_WRITE_OPTIONS，不得自己手写参数"
    )
    return f"render_default -> write_outputs(**{focused})"


def _check_cli_option_surface() -> str:
    """[2e] CLI 选项表面必须与 :data:`CLI_OPTION_STRINGS` 冻结清单一致。

    动机：风险场景里包含「新增/变更布尔开关」。若只比对既有开关的默认值，一个**新增**
    且默认影响渲染的开关会绕过 [2e]；冻结选项表面即可让「新增开关」这件事必须被显式
    处理（更新清单 + 判断是否影响默认形式），而不是悄悄改变锚点语义。
    """
    from n3d_viz import __main__ as cli

    parser = cli.build_parser()
    got = sorted({s for action in parser._actions for s in action.option_strings})
    want = sorted(set(CLI_OPTION_STRINGS))
    assert got == want, (
        f"CLI 选项表面发生变化：{got} != {want}；请更新 CLI_OPTION_STRINGS，"
        "并确认新增/删除的选项是否影响「CLI 默认形式」的渲染结果"
        "（若影响，必须同步 core.DEFAULT_WRITE_OPTIONS 与 [2d] 的锚点语义）"
    )
    return f"{len(want)} 个 option string 与冻结清单一致"


def _read_ply_colors(path: Path) -> list[tuple[int, int, int]]:
    """回读 PLY 顶点颜色（属性顺序由头部声明决定，这里取 red/green/blue 列）。

    **必须是「按元素归属」解析**：有 FC 的 PLY 在 ``vertex`` 之后还声明了
    ``fc_node`` / ``fc_edge`` 元素，若不判断当前处于哪个元素就收集全部
    ``property`` 行，记录步长会被算错，顶点色与坐标都会解析成垃圾。
    """
    import struct

    raw = path.read_bytes()
    end = raw.find(b"end_header\n")
    assert end > 0, "PLY 缺少 end_header"
    header = raw[:end].decode("ascii").splitlines()
    props: list[str] = []
    prop_types: list[str] = []
    count = 0
    current = ""
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            current = parts[1]
            if current == "vertex":
                count = int(parts[2])
        elif parts[0] == "property" and len(parts) == 3 and current == "vertex":
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

#: PLY 中 FC 附加顶点（面板单元 + 边界块中心）的分类标记值。
PLY_FC_KIND_NAMES: dict[int, str] = {
    0: "panel_input", 1: "panel_output", 2: "block_input", 3: "block_output",
}


def _read_ply_fc_nodes(path: Path) -> list[tuple[float, float, float, int]]:
    """回读 PLY 的 ``fc_node`` 元素（坐标 + 分类标记 kind）。

    与 :func:`_read_ply_colors` 同源口径：只有当前元素是 ``fc_node`` 的 ``property``
    行才归入该元素，从而能真实反映产物内容（而不是复用写入路径的中间结果）。

    Args:
        path: PLY 路径。

    Returns:
        ``[(x, y, z, kind), ...]``；无该元素时返回空列表。
    """
    import struct

    raw = path.read_bytes()
    end = raw.find(b"end_header\n")
    assert end > 0, "PLY 缺少 end_header"
    header = raw[:end].decode("ascii").splitlines()
    props: list[str] = []
    prop_types: list[str] = []
    count = 0
    vertex_count = 0
    vertex_props: list[str] = []
    vertex_types: list[str] = []
    with_edges = False
    edge_count = 0
    current = ""
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            current = parts[1]
            if current == "vertex":
                vertex_count = int(parts[2])
            elif current == "edge":
                with_edges = True
                edge_count = int(parts[2])
            elif current == export_geometry.PLY_FC_NODE_ELEMENT:
                count = int(parts[2])
        elif parts[0] == "property" and len(parts) == 3:
            if current == "vertex":
                vertex_props.append(parts[2])
                vertex_types.append(parts[1])
            elif current == export_geometry.PLY_FC_NODE_ELEMENT:
                props.append(parts[2])
                prop_types.append(parts[1])
    if count == 0:
        return []
    code_map = {"float": "f", "float32": "f", "uchar": "B", "int": "i", "int32": "i",
                "double": "d", "float64": "d"}
    code_size = {"f": 4, "B": 1, "i": 4, "d": 8}

    def _rec(types: list[str]) -> tuple[str, int]:
        fmt = "<" + "".join(code_map[t] for t in types)
        return fmt, sum(code_size[c] for c in fmt[1:])

    v_fmt, v_size = _rec(vertex_types)
    e_fmt, e_size = _rec(["int", "int", "float"])
    f_fmt, f_size = _rec(prop_types)
    offset = vertex_count * v_size + (edge_count * e_size if with_edges else 0)
    body = raw[end + len(b"end_header\n"):]
    idx = {name: i for i, name in enumerate(props)}
    out: list[tuple[float, float, float, int]] = []
    for i in range(count):
        vals = struct.unpack_from(f_fmt, body, offset + i * f_size)
        out.append((
            float(vals[idx["x"]]), float(vals[idx["y"]]), float(vals[idx["z"]]),
            int(vals[idx["kind"]]),
        ))
    return out


def _read_ply_fc_edge_count(path: Path) -> int:
    """返回 PLY 头中 ``fc_edge`` 元素声明的边数（不存在返回 0）。"""
    return export_geometry.parse_ply_fc_nodes(str(path))[1]


def _read_obj_group_lines(path: Path, group: str) -> list[list[float]]:
    """读取 OBJ 中某个 ``g`` 分组下的全部 ``l`` 行（返回 3 个 1 基索引 + 无权重时为 2 个）。

    Args:
        path: OBJ 路径。
        group: 目标分组名（例如 :data:`n3d_viz.export_geometry.OBJ_GROUP_FC`）。

    Returns:
        ``[[i, j], ...]``（1 基索引，未减 1，便于与产物文本直接对照）。
    """
    out: list[list[float]] = []
    current = ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("g "):
            current = line[2:].strip()
        elif line.startswith("l ") and current == group:
            out.append([float(tok) for tok in line.split()[1:]])
    return out


def _fc_weight_max_diff(src_ckpt: Path, data: core.TopologyData, seg: dict[str, Any]) -> float:
    """独立回读 FC 权重矩阵，核对产物里每条抽样连线的权重。

    这是 `[2f]` 的关键独立性检查：不复用 :func:`n3d_viz.core.extract_fc` 的结果，
    而是重新 ``torch.load`` 产物、按 `(side, unit, neuron)` 直接取矩阵元素比对，
    因此能抓出「连线画对了位置但权重取错元素」这类缺陷。

    Args:
        src_ckpt: FC 产物路径。
        data: 已抽取的拓扑（提供 ``s_in_order`` / ``s_out_order``）。
        seg: HTML 内联负载的 ``fc`` 段（提供 ``edges``）。

    Returns:
        权重最大绝对偏差。
    """
    import torch

    sd = torch.load(str(src_ckpt), map_location="cpu", weights_only=False)["model_state_dict"]
    proj = sd["proj_weight"]
    fc_out = sd["fc_out_weight"]
    assert data.fc is not None
    in_pos = {n: i for i, n in enumerate(data.fc.s_in_order)}
    out_pos = {n: i for i, n in enumerate(data.fc.s_out_order)}
    worst = 0.0
    for e in seg["edges"]:
        if e["side"] == "input":
            want = float(proj[in_pos[int(e["neuron"])], int(e["unit"])].item())
        else:
            want = float(fc_out[int(e["unit"]), out_pos[int(e["neuron"])]].item())
        worst = max(worst, abs(want - float(e["w"])))
    return worst


def _fc_cloud_span(data: core.TopologyData) -> float:
    """返回神经元云在流向轴上的跨度。"""
    axis = str((data.config or {}).get("flow_axis") or core.DEFAULT_FLOW_AXIS)
    idx = core.FLOW_AXIS_INDEX[axis]
    vals = [p[idx] for p in data.neuron_pos]
    return float(max(vals) - min(vals))


def _fc_gap_error(data: core.TopologyData) -> float:
    """复核「面板间隙 == 云跨度 × FC_PANEL_GAP_RATIO」的最大误差。"""
    assert data.fc is not None
    axis = data.fc.panels["input"].axis
    idx = core.FLOW_AXIS_INDEX[axis]
    vals = [p[idx] for p in data.neuron_pos]
    lo, hi = float(min(vals)), float(max(vals))
    want = (hi - lo) * core.FC_PANEL_GAP_RATIO
    got_in = lo - data.fc.panels["input"].flow
    got_out = data.fc.panels["output"].flow - hi
    return max(abs(got_in - want), abs(got_out - want))


def _fc_panels_disjoint(data: core.TopologyData) -> bool:
    """断言面板的流向轴区间与神经元云的流向轴区间**不相交**。

    判据（与 README「几何与不重叠判据」一致）：把面板在流向轴上的厚度也算进去，
    要求 ``面板区间`` 完全落在云区间之外：``in_hi < cloud_lo`` 且 ``cloud_hi < out_lo``。

    Returns:
        是否不相交（False 时调用方以断言失败上报）。
    """
    assert data.fc is not None
    axis = data.fc.panels["input"].axis
    idx = core.FLOW_AXIS_INDEX[axis]
    vals = [p[idx] for p in data.neuron_pos]
    cloud_lo, cloud_hi = float(min(vals)), float(max(vals))
    in_lo, in_hi = data.fc.panels["input"].flow_interval
    out_lo, out_hi = data.fc.panels["output"].flow_interval
    return bool(in_hi < cloud_lo and cloud_hi < out_lo and in_lo <= in_hi and out_lo <= out_hi)


def _fc_panels_disjoint_from_points(
    fc_pts: Sequence[tuple[float, float, float, int]],
    ply_colors: Sequence[tuple[int, int, int]],
    data: core.TopologyData,
    ply_path: Path,
) -> bool:
    """在**落盘坐标**上复核不重叠：PLY 的神经元顶点区间 vs ``fc_node`` 顶点区间。

    与 :func:`_fc_panels_disjoint`（内存结构）互补：这条断言只用产物字节，
    因此能抓出「内存里不相交、但写出/取整后落到云里的」缺陷。

    Args:
        fc_pts: ``fc_node`` 顶点 ``(x, y, z, kind)``。
        ply_colors: 顶点色（仅用于断言 PLY 可解析，长度须等于 N）。
        data: 拓扑数据（提供流向轴与神经元数）。
        ply_path: PLY 路径（重新回读顶点坐标）。

    Returns:
        是否不相交。
    """
    assert data.fc is not None
    axis = data.fc.panels["input"].axis
    idx = core.FLOW_AXIS_INDEX[axis]
    verts = export_geometry.parse_ply_vertices(ply_path)
    assert len(verts) == data.n_neurons, "PLY vertex 元素不等于 N"
    assert len(ply_colors) == data.n_neurons, "PLY 顶点色数不等于 N"
    cloud_vals = [v[idx] for v in verts]
    cloud_lo, cloud_hi = min(cloud_vals), max(cloud_vals)
    panel_in = [q[idx] for q in fc_pts if q[3] == 0]
    panel_out = [q[idx] for q in fc_pts if q[3] == 1]
    blocks = [q[idx] for q in fc_pts if q[3] in (2, 3)]
    assert panel_in and panel_out and blocks, "PLY fc_node 分类标记不完整"
    return bool(
        max(panel_in) < cloud_lo and cloud_hi < min(panel_out)
        and max(blocks) > max(panel_out) and min(blocks) < min(panel_in)
    )


def _ply_header_has_sampling(path: Path) -> bool:
    """PLY 头部注释是否声明了抽样口径（含「NOT all connections」与参数量）。"""
    raw = Path(path).read_bytes()
    end = raw.find(b"end_header\n")
    assert end > 0, "PLY 缺少 end_header"
    head = raw[:end].decode("ascii")
    return (
        "NOT all connections" in head
        and "proj_weight" in head
        and "fc_out_weight" in head
        and "top-k=" in head
    )


def _check_anchor_groups_cover_k() -> str:
    """[2d] 断言锚点组**同时覆盖** ``K <= 9`` 与 ``K > 9`` 两类产物。

    为什么必须承重：层色实现有两条分支 —— ``K <= 9`` 取既有 9 色前 K 个（逐字节不变），
    ``K > 9`` 走均匀色相扩展（去重数恒等于 K）。历史上「9 色 + ``k % 9`` 循环」正是在
    ``K > 9`` 的真实几何（cylinder λ=2，K=15）上退化为只有 9 种颜色。若锚点组只剩
    K<=9 的产物，那条分支的代码回归就会**失去承重覆盖**（断言全绿而产物已经错了）。

    Returns:
        形如 ``"锚点组 3 个：K<=9 2 组 / K>9 1 组"`` 的摘要。
    """
    kinds = [str(g.get("kind")) for g in ANCHOR_GROUPS]
    assert "lt9" in kinds, (
        f"锚点组缺少覆盖 K<=9 路径的产物（现有 kinds={kinds}）"
    )
    assert "gt9" in kinds, (
        f"锚点组缺少覆盖 K>9 路径的产物（现有 kinds={kinds}）——"
        "K>9 的色相扩展分支将失去承重覆盖"
    )
    assert all(g.get("sha256") and g.get("bytes") and g.get("ckpt") for g in ANCHOR_GROUPS), (
        "每个锚点组都必须同时给出 ckpt / sha256 / bytes"
    )
    return (f"锚点组 {len(ANCHOR_GROUPS)} 个：K<=9 {kinds.count('lt9')} 组 / "
            f"K>9 {kinds.count('gt9')} 组")


def _check_ply_obj_unchanged(work_dir: Path | None = None) -> str:
    """[2d] 断言所有锚点/登记组的 **PLY / OBJ 与改动前快照逐项相同**。

    这是「相机单一事实来源」轮放宽后的零回归口径的**核心硬断言**：本次只改渲染
    （HTML 内联脚本），几何与写出路径一行未动，因此 PLY / OBJ 必须逐字节不变。
    任何字节变化都视为**回归**（而不是「重基线」）。

    **承重方式**：用**当前代码**重新渲染 :data:`PLY_OBJ_BASELINE` 里每一组产物，
    再把磁盘字节与 :data:`PLY_OBJ_BASELINE` 的恒定值逐项比对，因此能抓住
    「几何/写出路径被改动」这类回归，而不只是「常量没被改」。

    **承重前提（与 [2d] 锚点组同口径）**：本组恒定值取自 ``HEAD = 927d32f`` 当时用
    真实 checkpoint 渲染出的产物，故**仅在「上游 checkpoints 产物逐字节不变」的前提下
    才具承重意义**。上游若破坏性重建 / 清空重训，这 8 项会集体 FAIL 而原因不在渲染器 ——
    判读顺序固定为「先核对上游产物是否同一份，再怀疑渲染路径」。

    Args:
        work_dir: 临时输出目录；None 时用 ``checkpoints/n3d_viz/_verify/_plyobj_chk``。

    Returns:
        形如 ``"4 组 × 2 文件 = 8 项 PLY/OBJ 重渲后逐项相同"`` 的摘要。
    """
    tmp = work_dir or (_ROOT / "checkpoints/n3d_viz/_verify/_plyobj_chk")
    tmp.mkdir(parents=True, exist_ok=True)
    checked = 0
    try:
        for label, ckpt, files in PLY_OBJ_BASELINE:
            ckpt_path = _ROOT / ckpt
            assert ckpt_path.exists(), f"[{label}] checkpoint 不存在：{ckpt_path}"
            reports = core.render_default(ckpt, out_dir=tmp)
            for fname, (want_sha, want_bytes) in files.items():
                key = "ply" if fname.endswith(".ply") else "obj"
                got = Path(reports[key]["path"])
                got_sha = _sha256(got)
                got_bytes = got.stat().st_size
                assert got_bytes == want_bytes, (
                    f"[{label}] {fname} 字节数变化 {want_bytes} -> {got_bytes}"
                    "（PLY/OBJ 必须逐字节不变；变化即为回归，须定位原因）"
                )
                assert got_sha == want_sha, (
                    f"[{label}] {fname} SHA256 变化 {want_sha[:16]} -> {got_sha[:16]}"
                    "（PLY/OBJ 必须逐字节不变；变化即为回归，须定位原因）"
                )
                checked += 1
    finally:
        for p in tmp.rglob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)
        leftover = [p for p in tmp.rglob("*") if p.is_file()]
        assert not leftover, f"临时产物未清理干净：{leftover[:3]}"
    return (f"{len(PLY_OBJ_BASELINE)} 组 × 2 文件 = {checked} 项 PLY/OBJ "
            f"用当前代码重渲后逐项相同（含 K<=9 / K>9 / FC 三类）")


def _check_anchor_rebase_logged() -> str:
    """[2d] 断言每个**发生变化**的 HTML 锚点都已登记「旧值 + 原因」。

    维护约定：使 HTML 锚点变化且并非回归的改动必须登记（否则「锚点悄悄漂移」无从追溯）；
    PLY / OBJ 不在登记之列 —— 它们**不允许**变化。

    Returns:
        形如 ``"3 条 HTML 重基线登记齐备（旧值 + 原因）"`` 的摘要。
    """
    html_names: list[str] = []
    for group in ANCHOR_GROUPS:
        html_names.extend(n for n in group["sha256"] if n.endswith(".html"))
    missing = [n for n in html_names if n not in ANCHOR_REBASE_LOG]
    assert not missing, (
        f"以下 HTML 锚点未在 ANCHOR_REBASE_LOG 中登记旧值/原因：{missing}"
    )
    for name, rec in ANCHOR_REBASE_LOG.items():
        assert rec.get("old_sha256") and rec.get("old_bytes") and rec.get("reason"), (
            f"{name} 的重基线登记不完整（需 old_sha256 / old_bytes / reason）"
        )
        new_sha = None
        for group in ANCHOR_GROUPS:
            if name in group["sha256"]:
                new_sha = group["sha256"][name]
        assert new_sha is not None, f"{name} 已登记但不在任何锚点组中"
        assert new_sha != rec["old_sha256"], (
            f"{name} 的新旧 SHA256 相同（{new_sha[:16]}）—— 要么未真正重基线，要么登记陈旧"
        )
    assert FC_HTML_PRE_CAM_FIX[0] and FC_HTML_PRE_CAM_FIX[1] > 0, "FC 产物旧值登记缺失"
    return (f"{len(ANCHOR_REBASE_LOG)} 条 HTML 重基线登记齐备（旧值 + 原因）；"
            f"覆盖 {len(html_names)} 个 HTML 锚点")


def _html_self_contained_ok(html: str) -> bool:
    """HTML 自包含静态检查（无外部引用 / 无 syn_dist / 体积 < 2MB）。"""
    from n3d_viz import render_html as _render_html

    report = _render_html.assert_self_contained(html)
    return bool(report["ok"] and not report["external_refs"] and not report["has_syn_dist"])


#: [2f] `validate_fc_top_k` 必须拒绝的输入（顺序即断言中的期望下标序列）。
#:
#: 口径说明：校验按**最严格**口径要求 ``int`` 且非 ``bool``，因此
#: ``3.0`` 这类「值合法但类型是 float」的输入也一律拒绝（不静默强转）。
#: 这里**不**放 ``0.0`` —— 它的值本身越界（0 < MIN_FC_TOP_K），
#: 与「类型非法」混在一起会让失败信息难以定位。
_FC_TOP_K_BAD: tuple[Any, ...] = (
    2.5, 1.5, 8.7, 3.0,      # 非整数浮点（含「看似整数的 3.0」）
    True, False,             # bool 是 int 子类，不特判会被当成 1/0
    None, "abc", [1],        # None / 字符串 / 容器
    0, 9, -1,                # 越界
)


def _fc_top_k_rejections() -> list[int]:
    """逐个校验 :data:`_FC_TOP_K_BAD` 都被 :func:`core.validate_fc_top_k` 拒绝。

    Returns:
        被拒绝项的**下标**列表（全部拒绝时为 ``range(len(_FC_TOP_K_BAD))``）。
    """
    rejected: list[int] = []
    for i, bad in enumerate(_FC_TOP_K_BAD):
        try:
            core.validate_fc_top_k(bad)  # type: ignore[arg-type]
        except ValueError:
            rejected.append(i)
    return rejected


def _fc_top_k_error_kinds() -> set[str]:
    """返回 :data:`_FC_TOP_K_BAD` 触发的**异常类型名集合**（应为 ``{"ValueError"}``）。

    为什么单独断言异常类型：原实现 ``int(None)`` 抛裸 ``TypeError``，而 docstring
    只声明 ``ValueError``，按契约只捕获 ``ValueError`` 的调用方（CLI / GUI）会漏接。
    """
    kinds: set[str] = set()
    for bad in _FC_TOP_K_BAD:
        try:
            core.validate_fc_top_k(bad)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - 这里就是要看异常类型
            kinds.add(type(exc).__name__)
    return kinds


def _fc_geometry_rejections() -> list[str]:
    """FC 几何入口对 ``H <= 0`` 与非法类型的拒绝标记（应为 ``["H=0","H=-1","H=True"]``）。"""
    pos = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    out: list[str] = []
    for h in (0, -1, True):
        try:
            core._fc_panel_geometry(pos, [1.0] * max(int(h), 1), [1.0] * max(int(h), 1),
                                    h, core.DEFAULT_FLOW_AXIS)  # type: ignore[arg-type]
        except core.CheckpointSchemaError:
            out.append(f"H={h}")
    return out


def _fc_geometry_rejections2() -> list[str]:
    """FC 几何入口对「流向轴非法 / 云跨度为 0」的拒绝标记。"""
    out: list[str] = []
    try:
        core._fc_panel_geometry([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], [1.0], [1.0], 1, "w")
    except core.CheckpointSchemaError:
        out.append("axis=w")
    try:
        core._fc_panel_geometry([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], [1.0], [1.0], 1, "z")
    except core.CheckpointSchemaError:
        out.append("zero-span")
    return out


def _fc_geometry_h1_ok() -> bool:
    """``H=1`` 与「两轴跨度均为 0 的共线点云」都必须能正常构造（走兜底间距）。"""
    try:
        core._fc_panel_geometry([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], [1.0], [1.0], 1, "z")
    except Exception:  # noqa: BLE001
        return False
    try:
        core._fc_panel_geometry([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)], [1.0, 1.0], [1.0, 1.0], 2, "z")
    except Exception:  # noqa: BLE001
        return False
    return True


#: [2f] 归一化时用来替换 checkpoint 路径 / 派生产物名的哨兵。
#: 两态临时样本的文件名天然不同，产物内容里的 ``meta.checkpoint`` / ``ckpt_stem`` /
#: PLY 与 OBJ 的 ``source checkpoint`` 注释都会随之不同 —— 那只反映「输入文件名」，
#: 与 FC 判定逻辑无关，故归一化掉；其余字节一律保留，仍能抓住真实内容回归。
_TRI_PATH_SENTINEL = "<same-ckpt>"

#: 两态样本的 checkpoint 名（衍生出的产物名/文件名都要归一化）。
_TRI_NAMES = ("fc_zero", "fc_absent")


def _norm_html(html: str) -> str:
    """归一化产物文本中与「来源 checkpoint 名」相关的字段，便于两态逐字节对拍。

    需要替换的三处形态（都已实测确认）：
      * ``meta.checkpoint`` = 完整路径（含 ``.pt``）；
      * ``meta.ckpt_stem`` = 去扩展名的文件名；
      * PLY / OBJ 的 ``source checkpoint`` 注释 = 带扩展名的文件名。
    因此先替换「带扩展名」的形态、再替换裸文件名，并用正则抹掉路径前缀。
    """
    text = html
    for name in _TRI_NAMES:
        text = text.replace(name + ".pt", _TRI_PATH_SENTINEL.strip("<>"))
        text = text.replace(name, _TRI_PATH_SENTINEL.strip("<>"))
    # 抹掉可能残留的目录前缀差异（例如两侧用了不同的临时子目录）
    return re.sub(r"[A-Za-z0-9_.\\/-]*?" + re.escape(_TRI_PATH_SENTINEL.strip("<>")),
                  _TRI_PATH_SENTINEL, text)



def _norm_bytes(raw: bytes) -> bytes:
    """:func:`_norm_html` 的字节版（用于 PLY / OBJ 的逐字节对拍）。"""
    return _norm_html(raw.decode("latin-1")).encode("latin-1")


def _norm_text(raw: bytes) -> str:
    """把产物字节先按 latin-1 解码（1 字符 = 1 字节），再做与 HTML 相同的归一化。

    为什么用 latin-1 而不是 utf-8：PLY 是二进制格式，任意字节序列都可能出现；
    latin-1 对任何字节序列都可逆且不抛异常，因此子串断言可以稳定执行。
    归一化本身复用 :func:`_norm_html`（**不是**原样返回）——PLY / OBJ 的来源
    checkpoint 注释同样要抹掉文件名差异，否则两个无 FC 样本永远「不相等」。
    """
    return _norm_html(raw.decode("latin-1"))




def _non_topology_reason(path: Path) -> str | None:
    """判断一份 ``.pt`` 是否**不是** N3D 拓扑产物；是则返回可读原因，否则返回 None。

    为什么需要预先分拣：``checkpoints/n3d_shape/`` 目录里同时存在 MLP 基线
    （``config.arch == "mlp"``，只有 4 个键、完全没有 N3D 拓扑）等产物。若直接让
    它们进入泛化断言，每条都会以 ``KeyError: 'level_node_reach'`` 失败 —— 那是
    「样本选错」而不是「代码坏了」，会让报告失去可读性。分拣后逐类**明确 SKIP
    并计入报告**（不静默跳过），拓扑产物仍走完整断言。

    Args:
        path: ``.pt`` 路径。

    Returns:
        跳过原因；可作为 N3D 拓扑产物时返回 ``None``。
    """
    import torch

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return f"无法反序列化：{type(exc).__name__}"
    sd = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(sd, Mapping):
        return "没有 model_state_dict"
    missing = [k for k in core.REQUIRED_KEYS if k not in sd]
    if missing:
        arch = (obj.get("config") or {}).get("arch") if isinstance(obj, dict) else None
        return f"缺 {len(missing)} 个二期拓扑键" + (f"（arch={arch}）" if arch else "")
    return None


def _make_fc_checkpoint(
    src_ckpt: Path,
    out_dir: Path,
    name: str,
    drop_keys: Sequence[str] = (),
) -> Path:
    """由一份**有 FC**的真实产物派生轻量 checkpoint，用于三态与异常路径断言。

    做法：只保留 :data:`core.REQUIRED_KEYS` + :data:`core.FC_REQUIRED_KEYS`
    （丢弃 ``syn_dist`` 等巨型张量），因此单个临时产物只有几十 KB；
    ``drop_keys`` 中的键会被删除，用来构造「``fc_dim != 0`` 但缺 FC 键」的负例。

    Args:
        src_ckpt: 有 FC 的真实产物路径。
        out_dir: 临时目录。
        name: 临时 checkpoint 文件名（不含后缀）。
        drop_keys: 要删除的键（构造残缺产物）。

    Returns:
        临时 checkpoint 路径。
    """
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    obj = torch.load(str(src_ckpt), map_location="cpu", weights_only=False)
    sd = obj["model_state_dict"]
    keep = tuple(core.REQUIRED_KEYS) + tuple(core.FC_REQUIRED_KEYS) + ("head_weight", "out_scope_index")
    slim = {k: sd[k].clone() for k in keep if k in sd and k not in tuple(drop_keys)}
    config = dict(obj.get("config") or {})
    test_acc = obj.get("test_acc")
    # **显式释放巨型产物**：FC 产物的 ``syn_dist`` 实测约 174 MB（`[6600,6600]` float32），
    # 若不在这里断开引用，它会一直活到函数返回才由 GC 回收；[2f] 组会调用本函数 7 次，
    # 叠加 [2d] 的 4 组重渲，堆峰值会显著抬高（曾观察到一次进程级崩溃 0xC0000409）。
    del sd
    del obj
    gc.collect()
    path = out_dir / f"{name}.pt"
    torch.save({"model_state_dict": slim, "config": config, "test_acc": test_acc}, str(path))
    return path


#: 「无 FC」合成样本的两种形态（用于 [2f] 三态判定断言）。
FC_TRIGGER_ABSENT = 0    # config 无 fc_dim 键（二期 / 三期未启用产物）
FC_TRIGGER_ZERO = 1      # config.fc_dim == 0
FC_TRIGGER_OTHER = 2     # config.fc_dim != 0 且 FC 键齐全 -> 有 FC


def _rotate_fc_config(src_ckpt: Path, out_dir: Path, name: str, fc_dim: Any) -> Path:
    """复制一份（轻量）checkpoint 并改写其 ``config.fc_dim``，构造三态样本。

    Args:
        src_ckpt: 有 FC 的真实产物（提供合法拓扑与 FC 张量）。
        out_dir: 临时目录。
        name: 临时文件名（不含后缀）。
        fc_dim: 目标 ``fc_dim``；传 :data:`FC_TRIGGER_ABSENT` 时**删除**该键。

    Returns:
        临时 checkpoint 路径。
    """
    import torch

    path = _make_fc_checkpoint(src_ckpt, out_dir, name)
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    config = dict(obj.get("config") or {})
    if fc_dim is FC_TRIGGER_ABSENT:
        config.pop("fc_dim", None)
    else:
        config["fc_dim"] = fc_dim
    slim = obj["model_state_dict"]
    test_acc = obj.get("test_acc")
    torch.save({"model_state_dict": slim, "config": config, "test_acc": test_acc}, str(path))
    # 与 _make_fc_checkpoint 同一套内存纪律：断开对大对象的引用后立即回收。
    del slim
    del obj
    gc.collect()
    return path


def _render_smoke_blocks(html: str) -> tuple[str, str | None, str]:
    """从 HTML 中抽出「基础渲染器源码」「FC 叠加渲染器源码（可缺）」「内联数据 JSON」。

    定位方式基于内容而不是块序号：``viewer_fc.js`` 是**只在有 FC 时**追加的独立
    脚本块，块序号会随产物变化，用序号会在无 FC / 有 FC 之间错位。

    Args:
        html: 完整 HTML 文本。

    Returns:
        ``(viewer_js, fc_js_or_None, data_blob)``。
    """
    blocks = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    viewer_js = None
    fc_js = None
    for body in blocks:
        if "N3D 两端全连接包裹渲染器" in body:
            fc_js = body
        elif "N3D 二期拓扑三维渲染器" in body:
            viewer_js = body
    assert viewer_js is not None, (
        f"HTML 中未找到基础渲染器（{len(blocks)} 个 script 块，长度 "
        f"{[len(b) for b in blocks]}）"
    )
    return viewer_js, fc_js, _extract_payload_blob(html)


def _run_render_smoke(html: str, work_dir: Path, label: str = "") -> str:
    """用 Node + DOM 桩真实执行 HTML 内联的渲染器，返回冒烟汇总行。

    Args:
        html: 完整 HTML。
        work_dir: 临时文件目录（跑完即清理）。
        label: 文件名前缀（同一次运行内多次冒烟时避免互相覆盖）。

    Returns:
        node 输出的汇总行。

    Raises:
        AssertionError: node 返回码非 0。
    """
    viewer_js, fc_js, data_blob = _render_smoke_blocks(html)
    work_dir.mkdir(parents=True, exist_ok=True)
    js_path = work_dir / f"_{label}_viewer.js"
    data_path = work_dir / f"_{label}_data.json"
    js_path.write_text(viewer_js, encoding="utf-8", newline="")
    data_path.write_text(data_blob, encoding="utf-8", newline="")
    smoke_js = Path(__file__).resolve().parent / "assets" / "viewer_smoke.js"
    assert smoke_js.exists(), f"缺少渲染器冒烟脚本：{smoke_js}"
    cmd = ["node", str(smoke_js), str(js_path), str(data_path)]
    fc_path: Path | None = None
    if fc_js is not None:
        fc_path = work_dir / f"_{label}_viewer_fc.js"
        fc_path.write_text(fc_js, encoding="utf-8", newline="")
        cmd.append(str(fc_path))
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_subprocess_env(),
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    for line in out.strip().splitlines():
        print("       " + line)
    assert proc.returncode == 0, f"node 渲染器冒烟失败（returncode={proc.returncode}）"
    summary = [ln for ln in out.splitlines() if "汇总" in ln]
    for p in (js_path, data_path, fc_path):
        if p is not None:
            p.unlink(missing_ok=True)
    return summary[-1].strip() if summary else "ok"


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

    # ------------------------------------------------------------ [2a] 层配色
    # 几何无关化后 K 可以任意大（例如非均匀分层的 cylinder λ=2 实测 K=15），
    # 原先「9 色 + k % 9 循环」会让第 9 层与第 0 层同色，分层着色失去可分辨性。
    print("\n[2a] 层配色可扩展性（K <= 9 逐字节不变；K > 9 按均匀色相扩展）")
    chk.check("[2a] K ∈ [1,64] 层色去重数 == K", _check_palette_dedup,
              "任意几何下 K 可任意大，必须两两可分辨")
    chk.check("[2a] K <= 9 层色 == 既有 9 色前 K 个（逐字节不变）", _check_palette_base_prefix,
              "现有产物（含二期 viz_model.*）零回归的锚点")
    chk.check("[2a] HTML 层色与 PLY 层色同源", _check_palette_same_source,
              "同一张基础色表 + 同一套色相扩展")
    chk.check("[2a] hex_to_rgb 输入契约（非法抛 ValueError、分量 0..255）", _check_hex_contract,
              "拒绝前导正负号 / 内部空白 / 多余前导 # / 非十六进制字符")
    chk.check("[2a] _hue_palette_rgb 撞色回退分支（必撞色构造）", _check_palette_collision_fallback,
              f"K <= 64 实测撞色数 0（该分支不可达），故用 K={COLLISION_PROBE_K} 主动进入")

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

    # ---------------------------------------------- [2b] 非规整几何合成组
    # 本组**不依赖任何既有产物**（就地构造合法 state_dict），因此永久有效：
    # 即使所有 checkpoint 被清理，它依然证明渲染链路对几何零假设。
    print("\n[2b] 非规整几何合成组（就地构造合法 state_dict，破除各类几何假设）")
    synth_dir = out_path / "_synthetic"
    synth_out = synth_dir / "out"
    synth_samples = _synth_geometries()
    for sname, s_pos, s_edges, s_layers in synth_samples:
        s_n, s_e, s_k = len(s_pos), len(s_edges), len(s_layers)
        s_ckpt = _make_synthetic_checkpoint(synth_dir, sname, s_pos, s_edges, s_layers)
        s_err: str | None = None
        s_pts: list[tuple[float, float, float]] = []
        s_cols: list[tuple[int, int, int]] = []
        s_obj: dict[str, Any] = {}
        s_payload: dict[str, Any] = {}
        s_truth: list[tuple[float, float, float]] = []
        s_names: dict[str, str] = {}
        s_reports: dict[str, dict[str, Any]] | None = None
        try:
            s_data = core.load_topology(s_ckpt)
            s_reports = core.write_outputs(s_data, out_dir=synth_out, threshold=THRESHOLD)
            s_ply = Path(s_reports["ply"]["path"])
            s_objp = Path(s_reports["obj"]["path"])
            s_htmlp = Path(s_reports["html"]["path"])
            s_pts = export_geometry.parse_ply_vertices(s_ply)
            s_cols = _read_ply_colors(s_ply)
            s_obj = export_geometry.parse_obj(s_objp)
            s_payload = _parse_inline_payload(s_htmlp.read_text(encoding="utf-8"))
            import torch as _torch
            s_truth = _torch.load(str(s_ckpt), map_location="cpu",
                                  weights_only=False)["model_state_dict"]["neuron_pos"].tolist()
            s_names = {
                "ply": s_ply.name, "obj": s_objp.name, "html": s_htmlp.name,
            }
        except Exception as exc:  # noqa: BLE001 - 把失败原因带到每条断言上
            s_err = f"{type(exc).__name__}: {exc}"

        detail = f"合成样本 {sname}：N={s_n} E={s_e} K={s_k}，seed={SYNTH_SEED}"
        chk.check(f"[2b] {sname}: PLY 顶点数 == N({s_n})",
                  _guarded(s_err, lambda p=s_pts, v=s_n: _assert_eq(len(p), v)), detail)
        chk.check(f"[2b] {sname}: OBJ l 行数 == E({s_e})",
                  _guarded(s_err, lambda o=s_obj, v=s_e: _assert_eq(o["l_lines"], v)), detail)
        chk.check(f"[2b] {sname}: 层数 == K({s_k})",
                  _guarded(s_err, lambda pl=s_payload, v=s_k: _assert_eq(len(pl["layers"]), v)), detail)
        chk.check(f"[2b] {sname}: 层色去重数 == K({s_k})",
                  _guarded(s_err, lambda pl=s_payload, v=s_k: _assert_eq(
                      len({ly["color"] for ly in pl["layers"]}), v)),
                  "分层着色必须两两可分辨（K>9 走色相扩展）")
        chk.check(f"[2b] {sname}: PLY 顶点色去重数 == K({s_k})",
                  _guarded(s_err, lambda c=s_cols, v=s_k: _assert_eq(len(set(c)), v)), detail)
        chk.check(f"[2b] {sname}: 坐标与 neuron_pos 逐位一致（容差 1e-6）",
                  _guarded(s_err, lambda p=s_pts, t=s_truth: _assert_le(_max_abs_diff(p, t), TOL)),
                  detail)
        chk.check(f"[2b] {sname}: payload neurons/edges/layers 长度自洽",
                  _guarded(s_err, lambda pl=s_payload, n=s_n, e=s_e: _assert_true(
                      len(pl["neurons"]) == n and len(pl["edges"]) == e
                      and len(pl["layers"]) == len(pl["meta"]["layer_counts"]))),
                  detail)
        chk.check(f"[2b] {sname}: 产物名由 ckpt 名派生",
                  _guarded(s_err, lambda nm=s_names, st=sname: _assert_true(
                      nm.get("ply") == f"viz_{st}.ply" and nm.get("obj") == f"viz_{st}.obj"
                      and nm.get("html") == f"viz_{st}.html")),
                  detail)

    # 合成产物（checkpoint 与三件套）全部用完即删，避免 _verify 目录膨胀。
    synth_files = [p for p in synth_dir.rglob("*") if p.is_file()] if synth_dir.exists() else []
    for p in synth_files:
        p.unlink(missing_ok=True)
    leftover_synth = [p for p in synth_dir.rglob("*") if p.is_file()] if synth_dir.exists() else []
    chk.check("[2b] 合成组临时产物已清理（无文件残留）",
              lambda: _assert_eq(len(leftover_synth), 0),
              f"清理前 {len(synth_files)} 个文件")
    chk.check("[2b] 合成组残留体积 == 0 字节",
              lambda: _assert_eq(sum(p.stat().st_size for p in leftover_synth), 0),
              "合成 checkpoint 只保留契约键，约数十 KB")

    # ---------------------------------------------- [2c] 真实异构几何组
    # 用 checkpoints/n3d_shape/ 的**真实**产物作泛化证据。
    # 只断言泛化不变量，**不断言任何形状标签**（模块里不存在「形状类型」概念）。
    #
    # 目录里同时存在两类非本模块契约的产物，必须**预先分拣**而不是让它们变成一堆
    # 令人生疑的 KeyError：
    #   * MLP 基线（`*_mlp.pt`）：config.arch == "mlp"，只有 4 个键，没有 N3D 拓扑；
    #   * 无拓扑键的其它基线：直接以核心契约键判定。
    # 分拣后逐类明确 SKIP 并计入报告（不静默跳过）；有 FC 的产物走**同一套**
    # 泛化不变量，因此 FC 路径也被真实产物覆盖。
    print("\n[2c] 真实异构几何组（checkpoints/n3d_shape/*.pt）")
    shape_dir = _ROOT / SHAPE_DIR
    shape_out = out_path / "_shape"
    shape_names: list[str] = []
    shape_k15_dedup: list[int] = []
    shape_all = sorted(shape_dir.glob("*.pt")) if shape_dir.exists() else []
    shape_ckpts: list[Path] = []
    shape_skipped: list[tuple[str, str]] = []
    for sp in shape_all:
        reason = _non_topology_reason(sp)
        if reason is None:
            shape_ckpts.append(sp)
        else:
            shape_skipped.append((sp.name, reason))
    shape_fc_dedup: list[tuple[int, int]] = []
    if not shape_all:
        chk.skip("[2c] 真实异构几何组", f"{shape_dir} 不存在或其中没有 .pt 产物")
    else:
        if shape_skipped:
            chk.skip(
                "[2c] 非 N3D 拓扑产物已明确跳过（逐个计入报告）",
                f"{len(shape_skipped)} 个：" + "; ".join(f"{n}（{r}）" for n, r in shape_skipped),
            )
        chk.check("[2c] 目录内产物可分类（拓扑产物 + 明确跳过的非拓扑产物）",
                  lambda: _assert_eq(len(shape_ckpts) + len(shape_skipped), len(shape_all)),
                  f"拓扑 {len(shape_ckpts)} 个 / 非拓扑 {len(shape_skipped)} 个 / 合计 {len(shape_all)} 个")
        for sp in shape_ckpts:
            h_err: str | None = None
            h_n = h_e = h_k = 0
            h_pts: list[tuple[float, float, float]] = []
            h_cols: list[tuple[int, int, int]] = []
            h_obj: dict[str, Any] = {}
            h_payload: dict[str, Any] = {}
            h_truth: list[tuple[float, float, float]] = []
            h_names: dict[str, str] = {}
            h_reach_rows = 0
            h_seed: Any = None
            h_has_fc = False
            h_h = 0
            h_fc_nodes = 0
            try:
                import torch as _torch
                h_obj_raw = _torch.load(str(sp), map_location="cpu", weights_only=False)
                h_sd = h_obj_raw["model_state_dict"]
                h_cfg = h_obj_raw.get("config") or {}
                h_seed = h_cfg.get("seed")
                h_reach_rows = int(h_sd["level_node_reach"].shape[0])
                h_data = core.load_topology(sp)
                h_n, h_e, h_k = h_data.n_neurons, h_data.n_edges, h_data.n_layers
                h_has_fc = h_data.fc is not None
                h_h = h_data.fc.fc_width if h_data.fc is not None else 0
                h_reports = core.write_outputs(h_data, out_dir=shape_out, threshold=THRESHOLD)
                h_ply = Path(h_reports["ply"]["path"])
                h_objp = Path(h_reports["obj"]["path"])
                h_htmlp = Path(h_reports["html"]["path"])
                h_pts = export_geometry.parse_ply_vertices(h_ply)
                h_cols = _read_ply_colors(h_ply)
                h_obj = export_geometry.parse_obj(h_objp)
                h_payload = _parse_inline_payload(h_htmlp.read_text(encoding="utf-8"))
                h_truth = h_sd["neuron_pos"].tolist()
                h_names = {"ply": h_ply.name, "obj": h_objp.name, "html": h_htmlp.name}
                shape_names.extend(h_names.values())
                shape_k15_dedup.append((h_k, len({ly["color"] for ly in h_payload["layers"]})))
                h_fc_nodes = export_geometry.parse_ply_fc_nodes(str(h_ply))[0]
                shape_fc_dedup.append((h_h, h_fc_nodes))
            except Exception as exc:  # noqa: BLE001
                h_err = f"{type(exc).__name__}: {exc}"

            tag = sp.stem
            hdet = f"来源 {sp.name}（seed={h_seed}）"
            chk.check(f"[2c] {tag}: PLY 顶点数 == N",
                      _guarded(h_err, lambda p=h_pts, d=h_n: _assert_eq(len(p), d)), hdet)
            chk.check(f"[2c] {tag}: OBJ 核心边 l 行数 == E",
                      _guarded(h_err, lambda o=h_obj, d=h_e, hf=h_has_fc: _assert_eq(
                          o["group_l_counts"][export_geometry.OBJ_GROUP_CORE] if hf
                          else o["l_lines"], d)),
                      "有 FC 时核心边被归入独立 group，须按 group 计数")
            chk.check(f"[2c] {tag}: OBJ 总 l 行数 == E + FC 抽样条数",
                      _guarded(h_err, lambda o=h_obj, d=h_e, hf=h_has_fc, pl=h_payload: _assert_eq(
                          o["l_lines"], d + (len(pl.get("fc", {}).get("edges", [])) if hf else 0))),
                      "无 FC 时退化为 == E（与改动前口径一致）")
            chk.check(f"[2c] {tag}: 层数 == level_node_reach.shape[0]",
                      _guarded(h_err, lambda a=h_k, b=h_reach_rows: _assert_eq(a, b)), hdet)
            chk.check(f"[2c] {tag}: 层色去重数 == K",
                      _guarded(h_err, lambda pl=h_payload, d=h_k: _assert_eq(
                          len({ly["color"] for ly in pl["layers"]}), d)),
                      "几何是否规整不影响层色可分辨性")
            chk.check(f"[2c] {tag}: 坐标与 neuron_pos 逐位一致（容差 1e-6）",
                      _guarded(h_err, lambda p=h_pts, t=h_truth: _assert_le(_max_abs_diff(p, t), TOL)),
                      hdet)
            chk.check(f"[2c] {tag}: 产物名由 ckpt 名派生",
                      _guarded(h_err, lambda nm=h_names, st=tag: _assert_true(
                          nm.get("ply") == f"viz_{st}.ply" and nm.get("obj") == f"viz_{st}.obj"
                          and nm.get("html") == f"viz_{st}.html")),
                      "天然不撞名")
            # FC 产物多一条不变量：面板点数与内联负载同源且等于 2×H + 2
            chk.check(f"[2c] {tag}: FC 面板点数与负载一致（有 FC 时为 2×H+2）",
                      _guarded(h_err, lambda hf=h_has_fc, hh=h_h, got=h_fc_nodes, pl=h_payload:
                               _assert_eq(got, 2 * hh + 2) if hf else _assert_true(True)),
                      f"fc_dim != 0 = {h_has_fc}，H={h_h}")
        if shape_ckpts:
            chk.check("[2c] 异构产物名两两不同",
                      lambda: _assert_eq(len(set(shape_names)), len(shape_ckpts) * 3),
                      f"{len(shape_ckpts)} 个拓扑 checkpoint × 3 件套")
            k15 = [d for k, d in shape_k15_dedup if k == 15]
            chk.check("[2c] K=15 产物层色去重数 == 15（修复前为 9）",
                      lambda: _assert_all_eq(k15, 15),
                      "非均匀分层 cylinder λ=2 实测 K=15；修复前仅 9 色循环 → 去重数为 9")
            fc_prods = [(h, n) for h, n in shape_fc_dedup if h > 0]
            chk.check("[2c] 有 FC 的真实产物面板点数 == 2×H + 2（逐产物）",
                      lambda: _assert_all_eq([n - 2 * h for h, n in fc_prods], 2),
                      f"有 FC 的产物 {len(fc_prods)} 个；H 与面板点数逐产物核对")

    # ---------------------------------------------- [2d] 零回归锚点（两层 × 多锚点组）
    # 教训：旧版 [2d] 只把**磁盘上已有**的产物与常量比对、不重渲，因此对「代码侧的
    # K <= 9 回归」完全无感（把 LEVEL_PALETTE_BASE 前两项对调后 [2a] 四项与磁盘比对
    # 全部 PASS，而重渲出来的 HTML / PLY 已与锚点不同）。现在拆成两层：
    #   层 1 产物完整性：磁盘锚点存在且 SHA256 / 字节数 == 常量（守护交付件未被改动）
    #   层 2 代码回归（承重）：用**当前代码**重渲到临时目录，与**同一组锚点常量**比对
    # 层 2 必须用相对路径 checkpoint（见 ANCHOR_RERENDER_CKPT 的口径说明）。
    #
    # 2026-09-27 扩展为**多锚点组**（ANCHOR_GROUPS）：单组只覆盖 K<=9 的层色路径，
    # 一旦被替换就会让 K>9（色相扩展色板）的回归失去承重覆盖。现按组分别做两层校验，
    # 并断言「锚点组必须同时覆盖 K<=9 与 K>9」，另加「PLY/OBJ 重基线前后逐字节不变」。
    print("\n[2d] 零回归锚点：磁盘产物完整性 + 用当前代码重渲的逐字节回归（多锚点组）")
    chk.check("[2d] 锚点组同时覆盖 K<=9 与 K>9 两类（防止丢掉色板扩展覆盖）",
              _check_anchor_groups_cover_k,
              f"{len(ANCHOR_GROUPS)} 组：{[(g['kind'], g['name']) for g in ANCHOR_GROUPS]}")
    chk.check("[2d] 锚点组的 PLY/OBJ 与改动前快照逐项相同（本次改动不得动几何）",
              lambda: _check_ply_obj_unchanged(out_path / "_plyobj_chk"),
              f"{len(PLY_OBJ_BASELINE)} 组 × 2 文件，用当前代码重渲后比对")
    chk.check("[2d] HTML 锚点变更全部已登记（旧值 + 原因）",
              _check_anchor_rebase_logged,
              f"登记 {len(ANCHOR_REBASE_LOG)} 条：全部注明「viewer.js 新增 1 行」")

    anchor_dir = out_path / ANCHOR_RERENDER_DIRNAME
    group_summaries: list[tuple[str, dict[str, tuple[Path, str, int]], str | None, dict]] = []
    for gi, group in enumerate(ANCHOR_GROUPS):
        gckpt = _ROOT / group["ckpt"]
        g_dir = anchor_dir / f"g{gi}"
        # ---- 层 1：磁盘产物完整性 ----
        for fname, want in group["sha256"].items():
            fpath = _ROOT / PHASE2_ANCHOR_DIR / fname
            tag = f"[2d][磁盘][组{gi}]"
            if not fpath.exists():
                chk.skip(f"{tag} {fname} SHA256 == 锚点", f"{fpath} 不存在")
                continue
            chk.check(f"{tag} {fname} SHA256 == 锚点",
                      _guarded(None, lambda p=fpath, w=want: _assert_eq(_sha256(p), w)),
                      f"{PHASE2_ANCHOR_DIR}/{fname}（{group['name']}）")
            chk.check(f"{tag} {fname} 字节数 == 锚点",
                      _guarded(None, lambda p=fpath, w=group['bytes'][fname]:
                               _assert_eq(p.stat().st_size, w)),
                      f"{PHASE2_ANCHOR_DIR}/{fname}")
        # ---- 层 2：用当前代码重渲 ----
        g_err: str | None = None
        g_files: dict[str, tuple[Path, str, int]] = {}
        if not gckpt.exists():
            chk.skip(f"[2d][重渲][组{gi}] 用当前代码重渲并比对锚点", f"{gckpt} 不存在")
        else:
            try:
                g_reports = core.render_default(group["ckpt"], out_dir=g_dir)
                for key in ("html", "ply", "obj"):
                    p = Path(g_reports[key]["path"])
                    g_files[p.name] = (p, _sha256(p), p.stat().st_size)
            except Exception as exc:  # noqa: BLE001 - 把失败原因带到每条断言上
                g_err = f"{type(exc).__name__}: {exc}"
            for fname, want in group["sha256"].items():
                entry = g_files.get(fname)
                chk.check(f"[2d][重渲][组{gi}] {fname} SHA256 == 锚点（{group['kind']} 承重）",
                          _guarded(g_err, lambda e=entry, w=want: _assert_eq(e[1], w) if e
                                   else _assert_true(False)),
                          f"core.render_default @ {group['ckpt']}")
                chk.check(f"[2d][重渲][组{gi}] {fname} 字节数 == 锚点",
                          _guarded(g_err, lambda e=entry, w=group['bytes'][fname]:
                                   _assert_eq(e[2], w) if e else _assert_true(False)),
                          f"core.render_default @ {group['ckpt']}")
            for p, _sha, _size in g_files.values():
                p.unlink(missing_ok=True)
        group_summaries.append((group["name"], g_files, g_err, group))

    leftover_anchor = [p for p in anchor_dir.rglob("*") if p.is_file()] if anchor_dir.exists() else []
    total_anchor_files = sum(len(f) for _n, f, _e, _g in group_summaries)
    chk.check("[2d] 重渲临时产物已清理（无文件残留）",
              lambda: _assert_eq(len(leftover_anchor), 0),
              f"清理前 {total_anchor_files} 个文件（{len(ANCHOR_GROUPS)} 组）")
    chk.check("[2d] 重渲临时残留体积 == 0 字节",
              lambda: _assert_eq(sum(p.stat().st_size for p in leftover_anchor), 0),
              "不污染 _verify 目录")

    # ---------------------------------------------- [2e] 参数集一致性（承重）
    # 动机：[2d] 的锚点语义是「CLI 默认形式的产物」。若 __main__ 的 argparse 默认值
    # 与 core.DEFAULT_WRITE_OPTIONS 各自演进，会出现两种坏结局之一 —— 锚点断言无故
    # FAIL（被误判为回归），或为了让断言变绿而两边一起改、锚点悄悄漂移成「另一套
    # 默认形式」的产物。故把该耦合**显式化并承重**。
    print("\n[2e] 参数集一致性：CLI 默认形式 == core.DEFAULT_WRITE_OPTIONS == render_default 实际参数")
    chk.check("[2e] argparse 默认值映射 == core.DEFAULT_WRITE_OPTIONS", _check_cli_default_options,
              "改任一 CLI 默认值都会让本条直接 FAIL")
    chk.check("[2e] render_default 实际使用的参数集 == core.DEFAULT_WRITE_OPTIONS",
              _check_render_default_options,
              "白盒捕获 render_default 传给 write_outputs 的实参（不落盘）")
    chk.check("[2e] CLI 选项表面与冻结清单一致", _check_cli_option_surface,
              "新增/删除 CLI 选项必须显式更新清单并确认是否影响「渲染默认形式」")

    # ------------------------------------------------- 渲染器逻辑冒烟（Node + DOM 桩）
    print("\n[5b] 渲染器逻辑冒烟（真实执行内联 viewer.js）")
    if shutil.which("node") is None:
        chk.skip("渲染器逻辑冒烟", "未找到 node")
    else:
        render_smoke = chk.check("内联渲染器可执行且投影正确",
                                 lambda: _run_render_smoke(html, out_path, "base"))

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
    hits = chk.check("n3d_viz 不 import n3d_sphere / n3d_proto / n3d_shape",
                     lambda: _assert_list_eq(_source_scan(Path(__file__).resolve().parent), []),
                     "源码正则扫描 ^\\s*(from|import) n3d_(sphere|proto|shape)")
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

    # ================================================================ [2f] FC 支持
    # 触发判定三态 / 几何不重叠 / 抽样口径 / CLI+G 边界 / 异常路径。
    # 无 FC 路径的零回归由上面的 [2d] / [2e] 承重断言把守；这里额外做一次
    # 「同一次运行内的无 FC 产物与磁盘锚点逐字节一致」的对拍（见 [2f-F]）。
    print("\n[2f] 两端全连接包裹：触发判定三态 / 几何 / 抽样口径 / 异常路径")
    fc_dir = out_path / "_fc"
    fc_out = fc_dir / "out"
    fc_src = _ROOT / FC_PRODUCT_CKPT
    fc_data = None
    fc_reports: dict[str, dict[str, Any]] | None = None
    fc_err: str | None = None
    fc_pts: list[tuple[float, float, float, int]] = []
    fc_lines: list[list[float]] = []
    fc_obj: dict[str, Any] = {}
    fc_payload: dict[str, Any] = {}
    fc_smoke = ""
    if not fc_src.exists():
        chk.skip("[2f] 两端全连接包裹", f"{fc_src} 不存在")
    else:
        try:
            fc_data = core.load_topology(fc_src, fc_top_k=core.DEFAULT_FC_TOP_K)
            fc_reports = core.write_outputs(fc_data, out_dir=fc_out, threshold=THRESHOLD)
            fc_ply = Path(fc_reports["ply"]["path"])
            fc_objp = Path(fc_reports["obj"]["path"])
            fc_htmlp = Path(fc_reports["html"]["path"])
            fc_pts = _read_ply_fc_nodes(fc_ply)
            fc_lines = _read_obj_group_lines(fc_objp, export_geometry.OBJ_GROUP_FC)
            fc_obj = export_geometry.parse_obj(fc_objp)
            fc_payload = _parse_inline_payload(fc_htmlp.read_text(encoding="utf-8"))
            # 渲染器逻辑冒烟（含 FC 叠加渲染器）：与 [5b] 同一套桩，多传一个脚本块
            fc_smoke = _run_render_smoke(fc_htmlp.read_text(encoding="utf-8"), out_path, "fc")
        except Exception as exc:  # noqa: BLE001 - 把失败原因带到每条断言上
            fc_err = f"{type(exc).__name__}: {exc}"

    fc_meta = (fc_payload or {}).get("meta", {}) or {}
    fc_seg = (fc_payload or {}).get("fc", {}) or {}
    fc_real = fc_data.fc if fc_data is not None and fc_data.fc is not None else None
    h_real = fc_real.fc_width if fc_real is not None else 0
    s_in_real = len(fc_real.s_in_order) if fc_real is not None else 0
    s_out_real = len(fc_real.s_out_order) if fc_real is not None else 0
    fc_prod_detail = (
        f"来源 {Path(FC_PRODUCT_CKPT).name}（seed={fc_data.config.get('seed') if fc_data else '?'}，"
        f"H={h_real}，|S_in|={s_in_real}，|S_out|={s_out_real}）"
    )

    # ---- ② 面板点数 == 2×H ---------------------------------------------
    chk.check("[2f] FC 面板点数 == 2×H（PLY fc_node 中的面板单元）",
              _guarded(fc_err, lambda: _assert_eq(
                  sum(1 for q in fc_pts if q[3] in (0, 1)), 2 * h_real)), fc_prod_detail)
    chk.check("[2f] PLY fc_node 总点数 == 2×H + 2（含两个边界块中心）",
              _guarded(fc_err, lambda: _assert_eq(len(fc_pts), 2 * h_real + 2)), fc_prod_detail)
    chk.check("[2f] FC 面板点数 == 2×H（内联负载 panels.units 之和）",
              _guarded(fc_err, lambda: _assert_eq(
                  sum(len(p["units"]) for p in fc_seg["panels"]), 2 * h_real)), fc_prod_detail)
    chk.check("[2f] 面板单元分类标记 == 输入 H 个 + 输出 H 个",
              _guarded(fc_err, lambda: _assert_list_eq(
                  [sum(1 for q in fc_pts if q[3] == k) for k in (0, 1)], [h_real, h_real])),
              "kind=0 输入面板 / kind=1 输出面板")
    chk.check("[2f] 边界块中心点数 == 2（kind=2/3）",
              _guarded(fc_err, lambda: _assert_list_eq(
                  [sum(1 for q in fc_pts if q[3] == k) for k in (2, 3)], [1, 1])),
              "输入 784 / 输出 10 各一块")
    chk.check("[2f] PLY fc_node 与负载面板坐标逐位一致（容差 1e-6）",
              _guarded(fc_err, lambda: _assert_le(_max_abs_diff(
                  [(q[0], q[1], q[2]) for q in fc_pts if q[3] in (0, 1)],
                  [(u["x"], u["y"], u["z"]) for p in fc_seg["panels"] for u in p["units"]]), TOL)),
              "PLY 写出与 HTML 负载同源（只比面板单元，边界块中心另行断言）")
    chk.check("[2f] PLY 边界块中心与负载 blocks 坐标一致（容差 1e-6）",
              _guarded(fc_err, lambda: _assert_le(_max_abs_diff(
                  [(q[0], q[1], q[2]) for q in fc_pts if q[3] in (2, 3)],
                  [(b["x"], b["y"], b["z"]) for b in fc_seg["blocks"]]), TOL)),
              "输入 784 / 输出 10 两个边界块中心")

    # ---- ③ 抽样条数 == |S_in|×k + |S_out|×k ----------------------------
    chk.check("[2f] 抽样连线条数 == |S_in|×k + |S_out|×k",
              _guarded(fc_err, lambda: _assert_eq(
                  len(fc_seg["edges"]), (s_in_real + s_out_real) * core.DEFAULT_FC_TOP_K)),
              f"({s_in_real}+{s_out_real})×{core.DEFAULT_FC_TOP_K} = "
              f"{(s_in_real + s_out_real) * core.DEFAULT_FC_TOP_K}")
    chk.check("[2f] meta.fcSampleEdges == 抽样连线条数",
              _guarded(fc_err, lambda: _assert_eq(
                  int(fc_meta["fcSampleEdges"]), len(fc_seg["edges"]))))
    chk.check("[2f] OBJ 中 FC group 的 l 行数 == 抽样条数",
              _guarded(fc_err, lambda: _assert_eq(len(fc_lines), len(fc_seg["edges"]))),
              f"group={export_geometry.OBJ_GROUP_FC}")
    chk.check("[2f] PLY fc_edge 元素条数 == 抽样条数",
              _guarded(fc_err, lambda: _assert_eq(
                  _read_ply_fc_edge_count(Path(fc_reports["ply"]["path"])), len(fc_seg["edges"]))))
    chk.check("[2f] 抽样连线每个 S_in / S_out 神经元都至少有一条连线",
              _guarded(fc_err, lambda: _assert_true(
                  len({e["neuron"] for e in fc_seg["edges"] if e["side"] == "input"}) == s_in_real
                  and len({e["neuron"] for e in fc_seg["edges"] if e["side"] == "output"}) == s_out_real)),
              "top-k 口径的保证（全局阈值会静默丢掉整个神经元）")
    chk.check("[2f] 抽样连线 unit 下标合法（0..H-1）",
              _guarded(fc_err, lambda: _assert_true(
                  all(0 <= e["unit"] < h_real for e in fc_seg["edges"]))))
    chk.check("[2f] 抽样连线 neuron 下标合法（0..N-1）",
              _guarded(fc_err, lambda: _assert_true(
                  all(0 <= e["neuron"] < fc_data.n_neurons for e in fc_seg["edges"]))))
    chk.check("[2f] 抽样连线权重与产物矩阵逐位一致（容差 1e-6）",
              _guarded(fc_err, lambda: _assert_le(_fc_weight_max_diff(fc_src, fc_data, fc_seg), TOL)),
              "回读 proj_weight / fc_out_weight 独立复核（非复用抽取结果）")
    chk.check("[2f] 产物命名派生（viz_<ckpt名>.*）",
              _guarded(fc_err, lambda: _assert_true(
                  Path(fc_reports["html"]["path"]).name == f"viz_{fc_src.stem}.html"
                  and Path(fc_reports["ply"]["path"]).name == f"viz_{fc_src.stem}.ply"
                  and Path(fc_reports["obj"]["path"]).name == f"viz_{fc_src.stem}.obj")),
              f"viz_{fc_src.stem}.*")
    # 参数量（**全部**连线数）必须与实际矩阵元素数一致，且与抽样条数分开标注
    chk.check("[2f] 参数量 == proj_weight / fc_out_weight 实际元素数",
              _guarded(fc_err, lambda: _assert_list_eq(
                  [int(fc_seg["projCount"]), int(fc_seg["fcOutCount"])],
                  [s_in_real * h_real, h_real * s_out_real])),
              "参数量与抽样条数是两个量，产物中分别标注")

    # ---- ④ 面板流向轴区间与神经元云区间不相交 ---------------------------
    chk.check("[2f] 面板流向轴区间 ∩ 神经元云流向轴区间 == 空集",
              _guarded(fc_err, lambda: _assert_true(_fc_panels_disjoint(fc_data))),
              fc_prod_detail)
    chk.check("[2f] 不重叠在产物坐标上同样成立（PLY 面板点 vs 神经元点）",
              _guarded(fc_err, lambda: _assert_true(
                  _fc_panels_disjoint_from_points(fc_pts, _read_ply_colors(Path(fc_reports["ply"]["path"])),
                                                  fc_data, Path(fc_reports["ply"]["path"])))),
              "不只看内存结构，也复核落盘坐标")
    chk.check("[2f] 面板厚度 > 0 且小于云跨度",
              _guarded(fc_err, lambda: _assert_true(
                  0.0 < fc_seg["panels"][0]["thickness"] < _fc_cloud_span(fc_data))),
              "厚度为 0 会让「不相交」退化成「不接触」")
    chk.check("[2f] 间隙 == 云跨度 × 0.15（误差 < 1e-6）",
              _guarded(fc_err, lambda: _assert_le(
                  _fc_gap_error(fc_data), TOL)),
              f"FC_PANEL_GAP_RATIO={core.FC_PANEL_GAP_RATIO}")
    chk.check("[2f] 面板网格列数 == ceil(sqrt(H))",
              _guarded(fc_err, lambda: _assert_list_eq(
                  [p["cols"] for p in fc_seg["panels"]],
                  [math.ceil(math.sqrt(h_real))] * 2)),
              f"ceil(sqrt({h_real})) = {math.ceil(math.sqrt(h_real))}")

    # ---- ⑤ meta 声明存在且含「非全部连接」 ------------------------------
    chk.check("[2f] meta 声明存在且含『非全部连接』",
              _guarded(fc_err, lambda: _assert_true(
                  isinstance(fc_meta.get("fcDeclaration"), str)
                  and FC_NOT_ALL_TEXT in fc_meta["fcDeclaration"])),
              "抽样图不得被误读为全连接结构")
    chk.check("[2f] meta 声明含两侧全部连线数（参数量分开标注）",
              _guarded(fc_err, lambda: _assert_true(
                  f"{s_in_real * h_real:,}" in fc_meta["fcDeclaration"]
                  and f"{h_real * s_out_real:,}" in fc_meta["fcDeclaration"])),
              f"proj {s_in_real * h_real:,} + fc_out {h_real * s_out_real:,}")
    chk.check("[2f] meta.fcNotAllConnections == true 且抽样条数与参数量都写出",
              _guarded(fc_err, lambda: _assert_true(
                  fc_meta.get("fcNotAllConnections") is True
                  and fc_meta.get("fcTopK") == core.DEFAULT_FC_TOP_K
                  and fc_meta.get("fcWidth") == h_real
                  and fc_meta.get("fcSampleEdges") == len(fc_seg["edges"]))))
    chk.check("[2f] OBJ 伴随说明含『非全部连接』",
              _guarded(fc_err, lambda: _assert_true(
                  FC_NOT_ALL_TEXT in Path(fc_reports["obj"]["path"]).read_text(encoding="utf-8"))),
              "OBJ 是离线可读产物，口径说明必须随它走")
    chk.check("[2f] PLY 头部注释含抽样口径与参数量",
              _guarded(fc_err, lambda: _assert_true(_ply_header_has_sampling(Path(fc_reports["ply"]["path"])))),
              "comment fc: ... sampled ... NOT all connections")
    chk.check("[2f] 抽样口径写入 OBJ 的 k 值与 meta 一致",
              _guarded(fc_err, lambda: _assert_true(
                  f"k={core.DEFAULT_FC_TOP_K}" in Path(fc_reports["obj"]["path"]).read_text(encoding="utf-8"))))

    # ---- ⑥ PLY / OBJ 含 FC 元素且 group 名正确 --------------------------
    chk.check("[2f] PLY 含 fc_node 与 fc_edge 元素声明",
              _guarded(fc_err, lambda: _assert_list_eq(
                  list(export_geometry.parse_ply_fc_nodes(str(fc_reports["ply"]["path"]))),
                  [2 * h_real + 2, len(fc_seg["edges"])])),
              f"fc_node={2 * h_real + 2}（2×H 面板单元 + 2 个边界块中心）/ "
              f"fc_edge={len(fc_seg['edges'])}")
    chk.check("[2f] OBJ group 名正确且与核心边分离",
              _guarded(fc_err, lambda: _assert_list_eq(
                  fc_obj["groups"],
                  [export_geometry.OBJ_GROUP_CORE, export_geometry.OBJ_GROUP_FC])),
              f"{export_geometry.OBJ_GROUP_CORE} / {export_geometry.OBJ_GROUP_FC}")
    chk.check("[2f] OBJ 核心边 l 行数 == E 且 FC l 行数 == 抽样条数",
              _guarded(fc_err, lambda: _assert_list_eq(
                  [fc_obj["group_l_counts"][export_geometry.OBJ_GROUP_CORE],
                   fc_obj["group_l_counts"][export_geometry.OBJ_GROUP_FC]],
                  [fc_data.n_edges, len(fc_seg["edges"])])),
              f"E={fc_data.n_edges if fc_data else '?'}")
    chk.check("[2f] OBJ v 行数 == N + 2×H + 2",
              _guarded(fc_err, lambda: _assert_eq(fc_obj["v_lines"], fc_data.n_neurons + 2 * h_real + 2)),
              "神经元 + 两片面板 + 两个边界块中心")
    chk.check("[2f] OBJ 神经元顶点坐标与 neuron_pos 逐位一致（容差 1e-6）",
              _guarded(fc_err, lambda: _assert_le(
                  _max_abs_diff(fc_obj["vertices"][:fc_data.n_neurons], fc_data.neuron_pos), TOL)))
    chk.check("[2f] OBJ FC 顶点坐标与负载面板坐标一致（容差 1e-6）",
              _guarded(fc_err, lambda: _assert_le(_max_abs_diff(
                  fc_obj["vertices"][fc_data.n_neurons:fc_data.n_neurons + 2 * h_real],
                  [(u["x"], u["y"], u["z"]) for p in fc_seg["panels"] for u in p["units"]]), TOL)))
    chk.check("[2f] PLY 顶点数仍为 N（FC 点写在独立元素，不混入 vertex）",
              _guarded(fc_err, lambda: _assert_eq(
                  len(export_geometry.parse_ply_vertices(fc_reports["ply"]["path"])), fc_data.n_neurons)))
    chk.check("[2f] FC 产物 HTML 自包含（无外部引用 / 无 syn_dist / < 2MB）",
              _guarded(fc_err, lambda: _assert_true(
                  _html_self_contained_ok(
                      Path(fc_reports["html"]["path"]).read_text(encoding="utf-8")))),
              "叠加渲染器内联后仍须单文件自包含")
    chk.check("[2f] FC 产物渲染器逻辑冒烟（含叠加渲染器）",
              _guarded(fc_err, lambda: _assert_true("失败 0" in fc_smoke)), fc_smoke or "未执行")

    # ---- k 边界：1 / 8 合法且产物真的随 k 变化 --------------------------
    k_results: dict[int, Any] = {}
    for kk in (core.MIN_FC_TOP_K, core.MAX_FC_TOP_K):
        kk_err: str | None = None
        kk_edges = -1
        kk_html = ""
        kk_obj = ""
        try:
            kk_paths = core.resolve_output_paths(fc_src, out=fc_dir / f"k{kk}.html")
            core.write_outputs(
                core.load_topology(fc_src, fc_top_k=kk), out=kk_paths["html"], fc_top_k=kk,
            )
            kk_html = kk_paths["html"].read_text(encoding="utf-8")
            kk_obj = kk_paths["obj"].read_text(encoding="utf-8")
            kk_edges = len(_read_obj_group_lines(kk_paths["obj"], export_geometry.OBJ_GROUP_FC))
        except Exception as exc:  # noqa: BLE001
            kk_err = f"{type(exc).__name__}: {exc}"
        k_results[kk] = (kk_err, kk_edges, kk_html, kk_obj)
    for kk in (core.MIN_FC_TOP_K, core.MAX_FC_TOP_K):
        kk_err, kk_edges, _h, _o = k_results[kk]
        chk.check(f"[2f] --fc-top-k={kk} 抽样条数 == (|S_in|+|S_out|)×{kk}",
                  _guarded(kk_err, lambda e=kk_edges, k=kk: _assert_eq(
                      e, (s_in_real + s_out_real) * k)),
                  f"k={kk}（合法边界）")
    chk.check("[2f] k=1 与 k=8 的产物不同（开关类参数不得是空操作）",
              _guarded(None, lambda: _assert_true(
                  k_results[core.MIN_FC_TOP_K][2] != k_results[core.MAX_FC_TOP_K][2]
                  and k_results[core.MIN_FC_TOP_K][3] != k_results[core.MAX_FC_TOP_K][3])),
              "HTML 与 OBJ 都必须随 k 变化")
    chk.check("[2f] k=8 时抽样条数 ≈ 9,360（实测渲染规模可接受）",
              _guarded(None, lambda: _assert_eq(
                  k_results[core.MAX_FC_TOP_K][1], (s_in_real + s_out_real) * 8)),
              "默认 k=3 为 3,510 条；k=8 约 9,360 条仍须可渲染")

    # ---- ① 无 FC 三态：config 无 fc_dim / fc_dim == 0 -> 走既有展示且不报错
    # （2026-09-27 实测确认：二期与三期**未启用**产物的 config 里都没有 fc_dim 键）
    tri_dir = fc_dir / "tri"
    tri_err: str | None = None
    tri_rows: list[tuple[str, str, str, str]] = []
    try:
        assert fc_src.exists(), f"{fc_src} 不存在"
        zero_ck = _rotate_fc_config(fc_src, tri_dir, "fc_zero", 0)
        absent_ck = _rotate_fc_config(fc_src, tri_dir, "fc_absent", FC_TRIGGER_ABSENT)
        for tag, path in (("fc_dim=0", zero_ck), ("无 fc_dim 键", absent_ck)):
            z = core.load_topology(path)
            z_paths = core.resolve_output_paths(path, out=tri_dir / f"{path.stem}.html")
            core.write_outputs(z, out=z_paths["html"], threshold=THRESHOLD)
            tri_rows.append((
                tag,
                _norm_html(z_paths["html"].read_text(encoding="utf-8")),
                _norm_text(z_paths["ply"].read_bytes()),
                _norm_text(z_paths["obj"].read_bytes()),
            ))
    except Exception as exc:  # noqa: BLE001
        tri_err = f"{type(exc).__name__}: {exc}"

    def _tri(idx: int, field: int) -> Any:
        _guard(tri_err)
        assert len(tri_rows) == 2, f"三态样本缺失（得到 {len(tri_rows)} 条）"
        return tri_rows[idx][field]

    chk.check("[2f] 三态①『无 fc_dim 键』走既有展示（无 fc 段、不报错）",
              _guarded(tri_err, lambda: _assert_true(
                  '"fc":' not in _tri(1, 1) and "hasFc" not in _tri(1, 1))),
              "二期产物实测形态（config 无 fc_dim 键）")
    chk.check("[2f] 三态②『fc_dim == 0』走既有展示（无 fc 段、不报错）",
              _guarded(tri_err, lambda: _assert_true(
                  '"fc":' not in _tri(0, 1) and "hasFc" not in _tri(0, 1))),
              "关闭路径不得因 FC 功能上线而改变")
    chk.check("[2f] 三态①②的 HTML / PLY / OBJ 两两逐字节一致",
              _guarded(tri_err, lambda: _assert_true(
                  _tri(0, 1) == _tri(1, 1) and _tri(0, 2) == _tri(1, 2)
                  and _tri(0, 3) == _tri(1, 3))),
              "无 FC 的两条路径必须完全同源")
    chk.check("[2f] 三态①②的产物里没有任何 FC 元素（PLY 无 fc_node / OBJ 无 FC group）",
              _guarded(tri_err, lambda: _assert_true(
                  all(tok not in _tri(i, 2) for i in (0, 1)
                      for tok in ("fc_node", "fc_edge"))
                  and all(export_geometry.OBJ_GROUP_FC not in _tri(i, 3) for i in (0, 1)))),
              "无 FC 产物不得出现 FC 元素")
    chk.check("[2f] 无 FC 产物 HTML 不含叠加渲染器源码",
              _guarded(tri_err, lambda: _assert_true(
                  "N3D 两端全连接包裹渲染器" not in _tri(0, 1)
                  and "N3D 两端全连接包裹渲染器" not in _tri(1, 1))),
              "叠加渲染器仅在 fc_dim != 0 时内联（零回归的关键）")
    chk.check("[2f] 三态③『fc_dim != 0 且 FC 键齐全』才出现 fc 段",
              _guarded(fc_err, lambda: _assert_true(
                  '"fc":' in Path(fc_reports["html"]["path"]).read_text(encoding="utf-8")
                  and fc_meta.get("hasFc") is True)),
              "判定只看 fc_dim != 0，不依赖键是否存在来启用")

    # ---- ⑦ fc_dim != 0 但缺 FC 键 -> 退出码非 0 且无产物 ---------------
    for bi, (label, drop) in enumerate((
        ("缺 proj_weight", ("proj_weight",)),
        ("缺 fc_out_weight", ("fc_out_weight",)),
        ("缺 fc_in_weight", ("fc_in_weight",)),
        ("缺 fc_out_bias", ("fc_out_bias",)),
        ("缺 fc_in_weight + fc_out_bias", ("fc_in_weight", "fc_out_bias")),
    )):
        if not fc_src.exists():
            chk.skip(f"[2f] {label}", f"{fc_src} 不存在")
            continue
        bad_fc = _make_fc_checkpoint(fc_src, fc_dir / "bad", f"bad_fc_{bi}", drop_keys=drop)
        proc_bad_fc = subprocess.run(
            [sys.executable, "-m", "n3d_viz", "--checkpoint", str(bad_fc),
             "--out-dir", str(fc_dir / "bad_out"), "--quiet"],
            cwd=str(_ROOT), capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=_subprocess_env(),
        )
        bad_fc_out = (proc_bad_fc.stdout or "") + (proc_bad_fc.stderr or "")
        chk.check(f"[2f] {label} -> 退出码非 0",
                  lambda p=proc_bad_fc: _assert_true(p.returncode != 0),
                  f"returncode={proc_bad_fc.returncode}")
        chk.check(f"[2f] {label} -> 报错含缺失键名 '{drop[0]}'",
                  lambda o=bad_fc_out, k=drop[0]: _assert_true(k in o),
                  bad_fc_out.strip().replace("\n", " | ")[:160])
        chk.check(f"[2f] {label} -> 无 traceback",
                  lambda o=bad_fc_out: _assert_true(
                      "Traceback (most recent call last)" not in o))
        chk.check(f"[2f] {label} -> 不产生任何产物（不静默降级为无 FC 展示）",
                  lambda b=fc_dir / "bad_out": _assert_eq(
                      len(list(b.glob("viz_*"))) if b.exists() else 0, 0))
        bad_fc.unlink(missing_ok=True)

    # ---- ⑧ --fc-top-k 边界（1/8 合法、0/9/非整数非法）------------------
    if fc_src.exists():
        proc_k1 = subprocess.run(
            [sys.executable, "-m", "n3d_viz", "--checkpoint", str(fc_src),
             "--out-dir", str(fc_dir / "cli_k1"), "--fc-top-k", "1", "--quiet"],
            cwd=str(_ROOT), capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=_subprocess_env(),
        )
        chk.check("[2f] CLI --fc-top-k 1 退出码 == 0（合法下界）",
                  lambda: _assert_eq(proc_k1.returncode, 0),
                  (proc_k1.stdout or "") + (proc_k1.stderr or ""))
        k1_obj = fc_dir / "cli_k1" / f"viz_{fc_src.stem}.obj"
        chk.check("[2f] CLI --fc-top-k 1 的 OBJ 抽样条数 == |S_in|+|S_out|",
                  lambda: _assert_eq(
                      len(_read_obj_group_lines(k1_obj, export_geometry.OBJ_GROUP_FC)),
                      s_in_real + s_out_real),
                  f"{s_in_real}+{s_out_real}")
        proc_k8 = subprocess.run(
            [sys.executable, "-m", "n3d_viz", "--checkpoint", str(fc_src),
             "--out-dir", str(fc_dir / "cli_k8"), "--fc-top-k", "8", "--quiet"],
            cwd=str(_ROOT), capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=_subprocess_env(),
        )
        chk.check("[2f] CLI --fc-top-k 8 退出码 == 0（合法上界）",
                  lambda: _assert_eq(proc_k8.returncode, 0),
                  (proc_k8.stdout or "") + (proc_k8.stderr or ""))
        for bad_k in ("0", "9", "-1", "abc", "1.5"):
            proc_bad_k = subprocess.run(
                [sys.executable, "-m", "n3d_viz", "--checkpoint", str(fc_src),
                 "--out-dir", str(fc_dir / "cli_badk"), "--fc-top-k", bad_k, "--quiet"],
                cwd=str(_ROOT), capture_output=True, text=True, encoding="utf-8",
                errors="replace", env=_subprocess_env(),
            )
            bad_k_out = (proc_bad_k.stdout or "") + (proc_bad_k.stderr or "")
            chk.check(f"[2f] --fc-top-k={bad_k!r} -> 退出码非 0",
                      lambda p=proc_bad_k: _assert_true(p.returncode != 0),
                      f"returncode={proc_bad_k.returncode}")
            chk.check(f"[2f] --fc-top-k={bad_k!r} -> 无 traceback",
                      lambda o=bad_k_out: _assert_true(
                          "Traceback (most recent call last)" not in o))
        core_err: str | None = None
        try:
            core.validate_fc_top_k(0)
            core_err = "未报错"
        except ValueError:
            core_err = None
        except Exception as exc:  # noqa: BLE001
            core_err = f"{type(exc).__name__}: {exc}"
        chk.check("[2f] core.validate_fc_top_k(0) 抛 ValueError",
                  _guarded(core_err, lambda: _assert_true(True)))
        ok_bound: str | None = None
        try:
            assert core.validate_fc_top_k(1) == 1 and core.validate_fc_top_k(8) == 8
        except Exception as exc:  # noqa: BLE001
            ok_bound = f"{type(exc).__name__}: {exc}"
        chk.check("[2f] core.validate_fc_top_k 接受 1 与 8",
                  _guarded(ok_bound, lambda: _assert_true(True)))
        # —— 契约收紧（离朱 R22 缺陷 P1/P2 的回归防线）——
        # 原实现 `int(k)` 会静默接受 2.5/8.7/True，None 抛裸 TypeError；
        # 现在必须是「整数、非 bool、可读 ValueError」。
        chk.check("[2f] core.validate_fc_top_k 拒绝非整数浮点与 bool（不静默截断）",
                  lambda: _assert_list_eq(_fc_top_k_rejections(), list(range(len(_FC_TOP_K_BAD)))),
                  "2.5 / 1.5 / 8.7 / 3.0 / True / False / None / 'abc' / [1] / 越界值")
        chk.check("[2f] core.validate_fc_top_k 越界与非法类型都抛 ValueError（非裸异常）",
                  lambda: _assert_true(_fc_top_k_error_kinds() == {"ValueError"}),
                  "None 抛裸 TypeError 会让只捕获 ValueError 的调用方漏接")
    else:
        chk.skip("[2f] --fc-top-k 边界组", f"{fc_src} 不存在")

    # ---- 负例与三态临时产物清理 ----------------------------------------
    # 只删负例 / 三态样本（_fc/tri、_fc/bad、_fc/bad_out）；_fc/out 与 _fc/k*.html 是
    # 本次运行的交付样本，保留以便人工打开查看。
    removed = 0
    for sub in ("tri", "bad", "bad_out"):
        d = fc_dir / sub
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file():
                    p.unlink(missing_ok=True)
                    removed += 1
    leftover_pt = [p for p in fc_dir.rglob("*.pt")] if fc_dir.exists() else []
    chk.check("[2f] 负例与三态临时 checkpoint 已清理（无 .pt 残留）",
              lambda: _assert_eq(len(leftover_pt), 0),
              f"本次删除 {removed} 个临时文件；残留 {len(leftover_pt)} 个 .pt")

    # ---- [2f] FC 几何入口的异常包装（离朱 R22 缺陷 P3 的回归防线）------
    # 原实现在 `_place_units_in_panel` 的 `cell <= 0` 哨兵之前先算 max(cols,rows) /
    # math.sqrt(H)，于是 H=0 抛裸 ZeroDivisionError、H=-1 抛裸 ValueError: math domain
    # error —— 都是未包装的逃逸异常。现统一为可读 CheckpointSchemaError。
    chk.check("[2f] FC 几何入口 H<=0 / 类型非法 -> 可读 CheckpointSchemaError",
              lambda: _assert_list_eq(_fc_geometry_rejections(), ["H=0", "H=-1", "H=True"]),
              "先于任何算术校验，不再抛裸 ZeroDivisionError / math domain error")
    chk.check("[2f] FC 几何入口 流向轴非法 / 云跨度为 0 -> 可读 CheckpointSchemaError",
              lambda: _assert_list_eq(_fc_geometry_rejections2(),
                                      ["axis=w", "zero-span"]),
              "三条退化入口口径统一")
    chk.check("[2f] FC 几何入口 H=1 与共线云（两轴跨度均为 0）仍可用",
              lambda: _assert_true(_fc_geometry_h1_ok()),
              "退化输入必须报错，但合法边界（H=1）与共线点云走兜底间距、不得误报")

    # ------------------------------------------ [2g] 内联脚本块序不变式（可执行约束）
    # 背景：FC 叠加渲染器必须排在 viewer.js 之后（否则读不到共享相机 window.__n3d_cam）。
    # 2026-09-27 之前这只是一条**中文注释**；本轮把它升级为 core.build_html 的构建期断言。
    # 这里把守两件事：① 正常产物的探针顺序；② 人为把 FC 块挪到 viewer.js 之前时判据判否。
    print("\n[2g] 内联脚本块序不变式（FC 块必须在 viewer.js 之后）")
    block_order_html = fc_htmlp.read_text(encoding="utf-8") if fc_reports else ""
    chk.check("[2g] 正常产物：FC 块探针排在 viewer.js 探针之后（构建期断言通过）",
              _guarded(fc_err, lambda: _block_order_ok(block_order_html)),
              f"来源 {Path(FC_PRODUCT_CKPT).name}；探针串取自 core._VIEWER_CAM_PROBE / _FC_MAIN_PROBE")
    chk.check("[2g] 注入拒绝：FC 块前置时块序判据判否（旧顺序会被当场拦下）",
              _block_order_injection_rejected,
              "注入不改动 core 源码：用真实资产按旧顺序拼一份文本，再交给同一条判据")
    chk.check("[2g] 块序断言是活断言：伪造「FC 在前」的探针位置后 build_html 抛 ValueError",
              lambda: _block_order_assertion_is_live(fc_data),
              "仅在调用期间替换 str.find，finally 无条件还原；core 源码一行未改")

    # ------------------------------- [9] 唯一两层一致性 E2E 的版本控制登记（收口缺口）
    # 现状：没有第二条断言能拦住「两层不同步」；该 E2E 此前位于 .lizhu_env/r22_e2e/
    # （.gitignore:48 忽略整个 .lizhu_env/，0 个文件被跟踪）—— 等于守门测试不在版本控制内。
    print("\n[9] 两层一致性 E2E 的版本控制登记与依赖口径")
    chk.check("[9] 两层一致性 E2E 在版本控制内（git ls-files 跟踪 + 26 项断言未被削减）",
              _e2e_registered,
              f"脚本 {_E2E_SCRIPT_REL}（自 .lizhu_env/r22_e2e/ 迁入）")
    chk.check("[9] 迁入未新增运行依赖（playwright 仍是既有那一份，版本/文件数逐项对拍）",
              _e2e_no_new_dependency,
              "副本 == .lizhu_env/r22_e2e/node_modules；node_modules 不入库；requirements.txt 0 变更")

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


# ---------------------------------------------------------------------------
# [2g] 内联脚本块序不变式 + [9] 唯一两层一致性 E2E 的版本控制登记
# ---------------------------------------------------------------------------
#: [2g] 块序不变式在产物里的**实际观测口径**：viewer.js 的共享相机探针必须早于
#: viewer_fc.js 的主语句探针（与 ``core._VIEWER_CAM_PROBE`` / ``core._FC_MAIN_PROBE`` 同源，
#: 这里直接引用 core 的常量，避免出现第二份「事实」）。
_E2E_SCRIPT_REL: str = "n3d_viz/tests/e2e_two_layer_cam.mjs"

#: [9] 该 E2E 的断言条数基线（**迁移时冻结**：迁移只允许搬位置/补文档，不得改断言逻辑）。
_E2E_EXPECTED_CHECKS: int = 26


def _cam_probe_positions(html: str) -> tuple[int, int]:
    """返回产物中两个块序探针的位置 ``(viewer.js 探针, FC 探针)``（缺失为 -1）。"""
    return html.find(core._VIEWER_CAM_PROBE), html.find(core._FC_MAIN_PROBE)


def _html_has_viewer_before_fc(html: str) -> bool:
    """**与 :func:`core.build_html` 逐字同一条判据**：viewer.js 探针是否早于 FC 探针。"""
    i_viewer, i_fc = _cam_probe_positions(html)
    return 0 <= i_viewer < i_fc


def _block_order_ok(html: str) -> str:
    """[2g] 断言产物里 FC 块确实排在 ``viewer.js`` 之后（块序不变式的可观测量）。

    为什么要有这条**产物侧**断言：:func:`core.build_html` 的构建期断言把守的是
    「本次构建的拼接顺序」，本断言则独立地在**落盘产物文本**上再核一遍探针位置 ——
    两条一起才能覆盖「构建期断言被误删 / 被绕过（例如有人改用别的函数拼 HTML）」。
    """
    i_viewer, i_fc = _cam_probe_positions(html)
    assert i_viewer >= 0, (
        f"产物里找不到 viewer.js 的共享相机探针 {core._VIEWER_CAM_PROBE!r}"
        "（若确属渲染器源码改动，请同步更新 core._VIEWER_CAM_PROBE）"
    )
    assert i_fc >= 0, (
        f"产物里找不到 FC 渲染器探针 {core._FC_MAIN_PROBE!r}"
        "（若确属渲染器源码改动，请同步更新 core._FC_MAIN_PROBE）"
    )
    assert i_viewer < i_fc, (
        f"块序不变式被破坏：viewer.js 探针 @{i_viewer} 未排在 FC 探针 @{i_fc} 之前"
        "（FC 叠加层会读不到共享相机）"
    )
    return f"viewer.js 探针 @{i_viewer} < FC 探针 @{i_fc}"


def _injected_fc_before_viewer_html() -> str:
    """构造一份「FC 块排在 viewer.js 之前」的产物文本（**不改动 core 源码**的注入）。

    手法：用真实资产走一遍 ``core.build_html`` 的同一套占位替换，但把 FC 块拼在
    ``viewer.js`` **之前**（即 2026-09-27 修正之前的顺序）。
    """
    assets = core.ASSETS_DIR
    tpl = (assets / "viewer.html").read_text(encoding="utf-8")
    js = (assets / "viewer.js").read_text(encoding="utf-8")
    fc_js = (assets / core.FC_VIEWER_ASSET).read_text(encoding="utf-8")
    fc_block = "\n</script>\n<script>\n" + fc_js
    return tpl.replace(core._DATA_MARKER, "{}").replace(core._TPL_MARKER, fc_block + js)


def _block_order_injection_rejected() -> str:
    """[2g] **拒绝证明**：把 FC 块人为挪到 ``viewer.js`` 之前，块序判据必须判否。

    判据用**与 :func:`core.build_html` 逐字同一条**（:func:`_html_has_viewer_before_fc`）：
    正常产物判 **True**、注入文本判 **False** —— 后者即「若按旧顺序拼接，构建期判据
    会当场拒绝」，与 :data:`core._BLOCK_ORDER_VIOLATION_PREFIX` 登记的拒绝消息相对应。

    Returns:
        形如 ``"注入：FC@… < viewer@…（判据判否；正常产物判是）"`` 的摘要。
    """
    assets = core.ASSETS_DIR
    real_html = (assets / "viewer.html").read_text(encoding="utf-8").replace(
        core._TPL_MARKER,
        (assets / "viewer.js").read_text(encoding="utf-8")
        + "\n</script>\n<script>\n" + (assets / core.FC_VIEWER_ASSET).read_text(encoding="utf-8"),
    )
    injected = _injected_fc_before_viewer_html()
    i_viewer, i_fc = _cam_probe_positions(injected)
    assert i_fc >= 0 and i_viewer >= 0, "注入文本里两个探针必须都在（否则实验无效）"
    assert i_fc < i_viewer, f"注入实验无效：期望 FC 探针 @{i_fc} 早于 viewer 探针 @{i_viewer}"
    assert _html_has_viewer_before_fc(real_html), "正常拼接顺序下判据必须为 True"
    assert not _html_has_viewer_before_fc(injected), (
        "注入（FC 前置）后判据仍为 True —— 判据失效，本轮新增断言形同虚设"
    )
    assert core._BLOCK_ORDER_VIOLATION_PREFIX.startswith("块序不变式被破坏"), (
        "core 的块序拒绝消息前缀不符合登记口径"
    )
    return (f"注入「FC 块在 viewer.js 之前」-> 实测 FC@{i_fc} < viewer@{i_viewer}，"
            f"判据判否（正常产物判是）；core 拒绝消息前缀已登记为 "
            f"{core._BLOCK_ORDER_VIOLATION_PREFIX[:12]}…")


def _block_order_assertion_is_live(fc_data: core.TopologyData | None) -> str:
    """[2g] 证明 :func:`core.build_html` 的断言**真的会抛**（活断言，不是死代码）。

    **手法（不改被测源码，也不触碰 CPython 的不可变内置类型）**：正常产物的 HTML 就是
    「模板把 ``_TPL_MARKER`` 替换成 ``viewer.js + FC 块``」的结果。本断言先把这一步的替换
    **对调成「先 FC 块、后 viewer.js」**（2026-09-27 修正之前的顺序），再把结果包进一个
    只覆写 :meth:`str.find` 的 ``str`` 子类，最后交给**真实入口**
    :func:`n3d_viz.core.build_html` 的私有钩子 ``_html_hook``。于是 build_html 内部那次
    探针比较看到的就是注入顺序 —— 必须抛 ``ValueError``，且消息前缀 ==
    :data:`core._BLOCK_ORDER_VIOLATION_PREFIX`。

    为什么不「在调用期间替换 ``str.find``」：``str`` 是 CPython 的**不可变内置类型**，
    赋值会抛 ``TypeError: cannot set 'find' attribute of immutable type 'str'`` ——
    本断言的首版就是这么写的，实测直接 FAIL（留档），故改为子类覆写 + 私有钩子。

    参数 ``fc_data`` 由调用方从 ``[2f]`` 传入**同一份已加载的拓扑数据**：FC 产物的
    ``syn_dist`` 实测约 174 MB，再 ``load_topology`` 一次会显著抬高堆峰值
    （本模块历史上出现过进程级崩溃，见 README 的稳定性缺陷留档）。
    """
    assert fc_data is not None, "缺少 FC 拓扑数据，无法执行块序活断言实验"
    tpl = (core.ASSETS_DIR / "viewer.html").read_text(encoding="utf-8")
    js = (core.ASSETS_DIR / "viewer.js").read_text(encoding="utf-8")
    fc_js = (core.ASSETS_DIR / core.FC_VIEWER_ASSET).read_text(encoding="utf-8")
    fc_block = "\n</script>\n<script>\n" + fc_js
    normal = tpl.replace(core._TPL_MARKER, js + fc_block)
    i_viewer, i_fc = _cam_probe_positions(normal)
    assert 0 <= i_viewer < i_fc, "正常顺序的探针位置不符合预期，实验前提不成立"
    injected = tpl.replace(core._TPL_MARKER, fc_block + js)
    i_viewer_inj, i_fc_inj = _cam_probe_positions(injected)
    assert 0 <= i_fc_inj < i_viewer_inj, "注入顺序的探针位置不符合预期，实验无效"

    class _OrderFlipped(str):
        """只把两个块序探针的 ``find`` 结果换成注入顺序的位置（其余查找原样）。"""

        def find(self, sub: str, *a: Any, **kw: Any) -> int:  # noqa: ANN401
            if sub == core._VIEWER_CAM_PROBE:
                return i_viewer_inj
            if sub == core._FC_MAIN_PROBE:
                return i_fc_inj
            return super().find(sub, *a, **kw)

    raised = ""
    try:
        core.build_html(fc_data, assets_dir=core.ASSETS_DIR, _html_hook=_OrderFlipped)
    except ValueError as exc:
        raised = str(exc)
    assert raised, (
        "把拼接顺序伪造成「FC 在前」后，core.build_html 竟然没有抛 ValueError —— "
        "块序断言是死代码（或判据被误删）"
    )
    assert raised.startswith(core._BLOCK_ORDER_VIOLATION_PREFIX), (
        f"拒绝消息前缀不符：实测 {raised[:60]!r}"
    )
    return (f"伪造「FC 在前」的探针位置（FC@{i_fc_inj} < viewer@{i_viewer_inj}；正常为 "
            f"viewer@{i_viewer} < FC@{i_fc}）后 build_html 抛 ValueError：{raised[:44]}…")


def _e2e_registered() -> str:
    """[9] 断言唯一的两层一致性 E2E **在版本控制内**（`.gitignore:48` 曾整体忽略 `.lizhu_env/`）。

    判据两条：① ``git ls-files`` 列得到该脚本（证明「被跟踪」，而不是仅存在于工作区）；
    ② 脚本的断言条数 == :data:`_E2E_EXPECTED_CHECKS`（证明迁移未顺手削减断言）。
    """
    script = _ROOT / _E2E_SCRIPT_REL
    assert script.exists(), f"{_E2E_SCRIPT_REL} 不存在（E2E 未在版本控制路径下）"
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", _E2E_SCRIPT_REL],
        cwd=str(_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0, (
        f"git ls-files 未跟踪 {_E2E_SCRIPT_REL}（退出码 {proc.returncode}）："
        f"{(proc.stderr or proc.stdout or '').strip()[:200]}"
    )
    src = script.read_text(encoding="utf-8")
    n_check = len(re.findall(r"(?<!function )\bcheck\(", src))
    assert n_check == _E2E_EXPECTED_CHECKS, (
        f"E2E 断言条数 {n_check} != 基线 {_E2E_EXPECTED_CHECKS}（迁移不得改动断言数量）"
    )
    return (f"git ls-files 跟踪 {_E2E_SCRIPT_REL}；脚本内 check(...) 调用 {n_check} 条 "
            f"（基线 {_E2E_EXPECTED_CHECKS}）")


def _e2e_no_new_dependency() -> str:
    """[9] 断言迁入的 E2E **没有带来新的运行依赖**。

    判据（都是「依赖仍是既有那一份」的可观测证据）：

    ① `n3d_viz/tests/node_modules` 若存在，必须**不是 git 跟踪内容**（`node_modules/`
       在 `.gitignore` 里），且其 `playwright/package.json` 的 `version` 与既有
       `.lizhu_env/r22_e2e/node_modules/playwright` **逐字相同** —— 证明这里是**复制品**，
       不是另装的第二份依赖；
    ② 该副本若存在，其文件数必须与源目录**逐个包相同**（`playwright` / `playwright-core`
       的文件数 == 既有目录），即「同一个安装、同一批文件」；
    ③ `requirements.txt` 内没有 playwright / puppeteer / selenium 之类条目；
    ④ 脚本内 `from "playwright"` 仍是**原样的裸导入**（迁移未改导入语句）。

    背景：ESM 裸导入不做向上逐级解析、本仓库所在卷不支持目录联接，故接入方式是把既有
    `node_modules` 整份复制到 `n3d_viz/tests/`（见 README）。既不做「无副本」断言，
    也不允许「另装一份」。
    """
    src_nm = _ROOT / ".lizhu_env/r22_e2e/node_modules"
    dst_nm = _ROOT / "n3d_viz/tests/node_modules"
    src_pkg = src_nm / "playwright/package.json"
    assert src_pkg.exists(), (
        f"既有 Playwright 设施不存在：{src_pkg}（E2E 将无法运行；本模块**不**新增该依赖）"
    )
    detail = "n3d_viz/tests 无副本（需按 README 接入后才能跑浏览器 E2E）"
    if dst_nm.exists():
        _, tracked = _git_ls_files("n3d_viz/tests")
        bad = [p for p in tracked if "node_modules" in Path(p).parts]
        assert not bad, f"node_modules 不得入库，却出现在 git 跟踪列表：{bad[:2]}"
        src_ver = json.loads(src_pkg.read_text(encoding="utf-8"))["version"]
        dst_pkg = dst_nm / "playwright/package.json"
        assert dst_pkg.exists(), f"副本缺 playwright/package.json：{dst_pkg}"
        dst_ver = json.loads(dst_pkg.read_text(encoding="utf-8"))["version"]
        assert dst_ver == src_ver, (
            f"副本版本 {dst_ver} != 既有设施版本 {src_ver}（说明是另装的第二份依赖）"
        )
        counts = []
        for pkg in ("playwright", "playwright-core"):
            n_src = len([p for p in (src_nm / pkg).rglob("*") if p.is_file()])
            n_dst = len([p for p in (dst_nm / pkg).rglob("*") if p.is_file()])
            assert n_src == n_dst, f"{pkg} 文件数不一致：既有 {n_src} vs 副本 {n_dst}"
            counts.append(f"{pkg} {n_dst}")
        detail = (f"副本 == 既有安装（playwright {src_ver}；{('、'.join(counts))} 个文件），"
                  f"且 node_modules 未入库")
    req = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(playwright|puppeteer|selenium)", req, re.MULTILINE), (
        "requirements.txt 不得新增浏览器自动化依赖"
    )
    src_js = (_ROOT / _E2E_SCRIPT_REL).read_text(encoding="utf-8")
    assert 'import { chromium } from "playwright";' in src_js, (
        "迁移不得改写脚本的导入语句（需保持原样裸导入）"
    )
    return f"playwright {json.loads(src_pkg.read_text(encoding='utf-8'))['version']}；{detail}；requirements.txt 0 新增"


def _git_ls_files(rel_dir: str) -> tuple[int, list[str]]:
    """返回 ``(退出码, git 跟踪的相对路径列表)``（限定在 ``rel_dir`` 下）。"""
    proc = subprocess.run(
        ["git", "ls-files", rel_dir],
        cwd=str(_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    files = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return proc.returncode, files


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


def _assert_all_eq(values: Any, want: Any) -> str:
    """断言序列非空且每个元素都等于 ``want``（用于「全部样本都满足」类断言）。"""
    got = list(values)
    assert got, f"没有可检查的样本（期望至少 1 个元素等于 {want}）"
    assert all(v == want for v in got), f"实测 {got}，期望全部 == {want}"
    return f"{got} 全部 == {want}"


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