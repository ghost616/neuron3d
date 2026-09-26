"""n3d_shape 独立验证脚本（形状变体硬断言）。

用法
----
    python n3d_shape/verify_shape.py            # 全量（含与二期 n3d_sphere 的张量级比对）
    python n3d_shape/verify_shape.py --quick    # 只跑 SMALL 规模，加速

退出码：0 = 全部通过；1 = 存在失败项。

覆盖判据（逐条硬断言；**本清单 = 代码实际断言集 = README §8.3 覆盖表**，三者须一致）
----------------------------------------------------------------------------------
 S1  三形状连通性下限：E >= N、K >= 2、|S_in| >= 1、|S_out| >= 1
 S2  最近邻距恰为 2H（FCC 规则堆积契约，三形状均须成立）
 S3  形状专项：shape / lambda / circum_coef 与**脚本内联的独立公式**一致
      （不用 `config.shape_spec` 作期望值 —— 那属"自洽性"而非"正确性"检查）
 S3b 尺寸窗口：`R_min` 由**体积不等式 V(R) >= N·(4/3)πH³/φ 就地反解**得到并与实现比对
      （**不复制实现的表达式形式**，避免"代码 == 抄写下来的同一公式"）；`R_max`/`ρ` 亦然
 S3c 越界拒绝：给定正确 `R_max` 时，超出它的 `space_radius` 必须被拒
      （**注意**：探针取 `R_max` 相对倍数，故它**不是** D1/E1 类缺陷的独立防线，那是 S3b 的职责）
 S3e 体积不等式余量：实现解出的 `R_min` 处形状体积 >= 需求，且 0.999·R_min 处不再够
 S4  形状专项：selection_metric == 用 _shape_metric 复算 neuron_pos 得到的最大度量
 S5  形状专项：选取集合恰好等于"按形状度量取最近 N 个"（与独立复算逐位一致）
 S6  形状专项：最近邻距与神经元的 xy/轴向 extents 随形状改变（形状真的生效了）
 S7  形状专项：窗口非空 R_min <= R_max，且默认 space_radius 落在窗口内
 S8  负例：非 cylinder 显式 cyl_aspect 报错；极端 lambda 空窗口报错；非法 shape；越界报错
 S9  DAG：无环（Kahn 覆盖全部）、严格上行（z_A < z_B）、拓扑序覆盖全部且 == 轴升序
 S10 scope 非零（|S_in| >= 1 且 |S_out| >= 1，且掩码与孤立突触一致）
 S11 指纹含形状维度，且**同配置不同形状产物名互不相同**（防撞名硬要求）
 S12 sphere 分支与二期 n3d_sphere 在相同配置下全部张量 torch.equal 逐位相等
      （依仓库 D1 口径：跨代码路径**不比文件 SHA256**，而是 torch.load 后逐张量比对）
 S13 层数 K 随长径比变化（架构深度随形状改变）——如实记录并断言 K 与唯一轴坐标数自洽
 S16 **首个失败 N 锚点**（仅全量模式）：用**加固前口径**容差复核四个文档锚点
      （`cube D=0.10`/`D=0.072` → 1720、`sphere D=0.10` → 3256、
      `cylinder λ=2 D=0.10` → 2865）的「前一点通过 / 该点失败」边界
 S15 **大规模 2H 容差常驻回归**（**仅全量模式**；`--quick` 跳过以免耗时 13 min）：
      `N=2048/3072/4096` × 5 形状（`all/all`）全部构造成功，
      并报告实测偏差/容差的最紧余量（防止第二道常数项被改小而无人发现）
 S14 `neuron_pos` 排序口径**如实取证**（索引顺序非严格轴升序 + 拓扑序/严格上行仍成立）
      —— 继承二期的缩放整数 key，单方面修会破坏"默认分支与二期逐位一致"，故如实披露

[!] S3d 曾作为"窗口比值自洽性"判据被加入，实测其推导不成立（`R_min` 亦含形状相关体积
    系数），在**正确代码**上也会失败，属**断言本身错误**，已删除（详见 S3d 处注释）。

产物
----
报告写入 `checkpoints/n3d_shape/_verify/verify_shape_report.md`（`--quick` 为
`verify_shape_report_quick.md`，两者互不覆盖），
机读结果写入同名 `.json`。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

try:
    from .config import Config, SHAPE_CHOICES, shape_spec  # noqa: E402
    from .model import ThreeDNeuronSpace  # noqa: E402
    from . import train as train_mod  # noqa: E402
except ImportError:  # pragma: no cover
    from config import Config, SHAPE_CHOICES, shape_spec  # type: ignore
    from model import ThreeDNeuronSpace  # type: ignore
    import train as train_mod  # type: ignore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
VERIFY_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "n3d_shape", "_verify")
# 验证脚本需要导入二期 `n3d_sphere` 做**张量级比对**（S12）—— 这是**验证侧**的依赖，
# 生产模块本身仍然零 import 一期 / 二期（自包含契约不受影响）。
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- 参试形状矩阵（与训练验收阶段五组对照一致）----
SHAPE_CASES: List[Tuple[str, float]] = [
    ("sphere", 1.0),
    ("cube", 1.0),
    ("cylinder", 1.0),
    ("cylinder", 0.5),
    ("cylinder", 2.0),
]

# ---- 规模矩阵 ----
SIZE_CASES: List[Tuple[str, Dict[str, Any]]] = [
    ("DEFAULT", dict(N=256, y_in=8, y_out=8, H=0.1, D=0.1, batch_size=64)),
    ("SMALL", dict(N=64, y_in=4, y_out=4, H=0.15, D=0.15, batch_size=32)),
]


class Report:
    """收集逐条断言的通过/失败结果，并渲染为 Markdown 与 JSON。"""

    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []
        self.records: Dict[str, Any] = {}

    def check(self, cid: str, title: str, ok: bool, detail: str) -> bool:
        """记录一条断言。ok=False 时打印 FAIL 行（便于终端直接定位）。"""
        self.items.append(
            {"id": cid, "title": title, "ok": bool(ok), "detail": str(detail)}
        )
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {cid} {title} —— {detail}", flush=True)
        return bool(ok)

    def record(self, key: str, value: Any) -> None:
        """记录一条机读实测数据（非断言）。"""
        self.records[key] = value

    @property
    def all_ok(self) -> bool:
        return all(it["ok"] for it in self.items)

    def to_markdown(self, title: str) -> str:
        """渲染 Markdown 报告（含逐条断言与实测数据表）。"""
        n_fail = sum(1 for it in self.items if not it["ok"])
        lines = [
            f"# {title}",
            "",
            f"- 断言总数：{len(self.items)}；通过：{len(self.items) - n_fail}；失败：{n_fail}",
            f"- 结论：{'全部通过' if self.all_ok else '存在失败项'}",
            f"- 设备：CPU；torch {torch.__version__}",
            "",
            "## 逐条断言",
            "",
            "| 编号 | 判据 | 结果 | 实测 |",
            "| --- | --- | --- | --- |",
        ]
        for it in self.items:
            detail = it["detail"].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {it['id']} | {it['title']} | {'PASS' if it['ok'] else '**FAIL**'} | {detail} |"
            )
        lines += ["", "## 实测数据（机读记录）", "", "```json",
                  json.dumps(self.records, ensure_ascii=False, indent=2, default=str), "```", ""]
        return "\n".join(lines)


def make_config(size_kw: Dict[str, Any], shape: str, lam: float) -> Config:
    """构造参试配置（固定 flow_axis=z、两个 any_isolated、seed=42、CPU）。"""
    return Config(
        **size_kw,
        shape=shape,
        cyl_aspect=lam,
        flow_axis="z",
        input_scope="any_isolated",
        readout_scope="any_isolated",
        input_dim=784,
        output_dim=10,
        lr=1e-3,
        epochs=1,
        seed=42,
        device="cpu",
    )


def independent_selection(config: Config) -> torch.Tensor:
    """完全独立地复算"按形状度量取最近 N 个格点"（不调用 model 的内部实现）。

    返回按 `(流向轴坐标, 度量名次)` 字典序排列的坐标（float32），用于与 `neuron_pos`
    逐位比对 —— 这是"形状真的改到了第②步度量上"的最强证据。
    """
    import itertools
    import math

    a = 2.0 * math.sqrt(2.0) * float(config.H)
    h = (float(config.N) / 0.7405) ** (1.0 / 3.0)
    rho = config.shape_circum_radius
    search_radius = max(rho, config.max_space_radius)
    motif = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
        dtype=torch.float64,
    )
    bound = int(search_radius / a) + 3
    rng = torch.arange(-bound, bound + 1, dtype=torch.float64)
    grid = torch.tensor(
        list(itertools.product(rng.tolist(), repeat=3)), dtype=torch.float64
    )
    cand = (grid.unsqueeze(1) + motif.unsqueeze(0)).reshape(-1, 3) * a
    axis = config.flow_axis_index
    if config.shape == "sphere":
        metric = cand.norm(dim=1)
    elif config.shape == "cube":
        metric = cand.abs().amax(dim=1)
    else:
        xy = [i for i in range(3) if i != axis]
        metric = torch.maximum(
            cand[:, xy].norm(dim=1), cand[:, axis].abs() / float(config.cyl_aspect)
        )
    keep = metric <= search_radius
    cand, metric = cand[keep], metric[keep]
    order = torch.argsort(metric, stable=True)[: config.N]
    pos = cand[order]
    rank = torch.arange(config.N, dtype=pos.dtype)
    key = pos[:, axis] * (float(config.N) + 1.0) + rank
    perm = torch.argsort(key, stable=True)
    return pos[perm].to(torch.float32).contiguous()


def run_shape_matrix(rep: Report, size_tag: str, size_kw: Dict[str, Any]) -> None:
    """在给定规模下遍历全部形状，做 S1/S2/S3/S4/S5/S6/S7/S9/S10/S13 断言与取证。"""
    print(f"\n===== 规模 {size_tag}: {size_kw} =====", flush=True)
    for shape, lam in SHAPE_CASES:
        cfg = make_config(size_kw, shape, lam)
        tag = f"{size_tag}/{shape}/lam={lam:g}"
        model = ThreeDNeuronSpace(cfg)
        st = model.get_connection_stats()
        ts = model.get_topology_stats()
        sc = model.connectivity_selfcheck()

        # ---- S1 连通性下限（三形状都必须成立）----
        e, n = int(model.num_edges), int(model.N)
        k = int(ts["num_layers_true"])
        s_in, s_out = int(st["num_in_scope"]), int(st["num_out_scope"])
        floor_ok = (e >= n) and (k >= 2) and (s_in >= 1) and (s_out >= 1)
        rep.check(
            "S1", f"连通性下限 [{tag}]", floor_ok,
            f"E={e}(>={n}) E/N={e / n:.4f}, K={k}(>=2), |S_in|={s_in}, |S_out|={s_out}",
        )

        # ---- S2 最近邻距 == 2H ----
        nn = float(ts["nearest_neighbour_dist"])
        rep.check(
            "S2", f"最近邻距==2H [{tag}]", abs(nn - 2.0 * cfg.H) <= 1e-6,
            f"实测={nn:.9f}, 2H={2.0 * cfg.H:.9f}",
        )

        # ---- S3 shape/lambda/circum_coef 与配置一致 ----
        # ⚠️ 历史缺陷（离朱实测 D1，本判据当初**没能抓住**该缺陷）：这里曾写成
        #    `model.circum_coef == spec.circum_coef`，而 `spec` 与 `model` 同源于
        #    `config.shape_spec` —— 属**自洽性**检查，不是**正确性**检查。
        #    于是当 `shape_spec` 把 cube 的 `circum_coef` 误写成 1.0（应为 √3）时，
        #    双方一起错，S3 依然 PASS，真正的缺陷从验证网里漏了出去。
        #    修复：改为与**独立复算的文档公式**（本文件内联、不调用 config）逐项比对。
        expect_coef = (
            1.0 if shape == "sphere"
            else (math.sqrt(3.0) if shape == "cube" else math.sqrt(1.0 + lam * lam))
        )
        spec = shape_spec(shape, lam)
        s3_ok = (
            model.shape == shape
            and float(model.cyl_aspect) == (float(lam) if shape == "cylinder" else 1.0)
            and abs(float(model.circum_coef) - expect_coef) <= 1e-12
            and abs(float(spec.circum_coef) - expect_coef) <= 1e-12
        )
        rep.check(
            "S3", f"shape/lambda/circum_coef 与独立公式一致 [{tag}]", s3_ok,
            f"shape={model.shape}, lam={model.cyl_aspect:g}, "
            f"circum_coef={model.circum_coef:.6f} (独立期望 {expect_coef:.6f}, "
            f"spec={spec.circum_coef:.6f})",
        )

        # ---- S3b 尺寸窗口与**独立复算的文档公式**逐项比对（D1 的防线）----
        phi = 0.7405
        h_scale = (float(cfg.N) / phi) ** (1.0 / 3.0)
        # ------------------------------------------------------------------
        # [!] W1 修复要点（皋陶审查）：**不复制实现的表达式形式**，而是由
        #     "该形状的体积 >= N·(4/3)πH³/φ" 这一**不等式就地反解**出特征尺度。
        #     实现写法是 `H*(2N/(3λφ))^(1/3)`；本判据写法是
        #     `((N*(4/3)πH³/φ) / (形状体积系数))^(1/3)`，两者形式不同、可互相证伪。
        #     `shape_volume_coef(R)` 返回"形状体积 / R³"（R = 特征尺度），
        #     定义完全来自几何（球 (4/3)π、立方体 8、圆柱 2πλ），不引用实现。
        # ------------------------------------------------------------------
        def shape_volume_coef(_shape: str, _lam: float) -> float:
            """返回形状体积 `V = coef · R³` 中的系数（R = 特征尺度）。"""
            if _shape == "sphere":
                return (4.0 / 3.0) * math.pi          # V = (4/3)πR³
            if _shape == "cube":
                return 8.0                            # V = (2s)³ = 8s³
            return 2.0 * math.pi * _lam               # V = πr²·(2λr) = 2πλr³

        # 由体积不等式反解：coef·R³ >= N·(4/3)πH³/φ  =>  R >= (N·(4/3)πH³/(φ·coef))^(1/3)
        vol_coef = shape_volume_coef(shape, lam)
        need_vol = float(cfg.N) * (4.0 / 3.0) * math.pi * float(cfg.H) ** 3 / phi
        exp_min = (need_vol / vol_coef) ** (1.0 / 3.0)
        # R_max 由"外接半径系数"折算（同样由几何定义：球 1 / 立方体 √3 / 圆柱 √(1+λ²)）
        exp_max = (float(cfg.H) + float(cfg.D)) * h_scale / expect_coef
        s3b_ok = (
            abs(cfg.min_space_radius - exp_min) <= 1e-12
            and abs(cfg.max_space_radius - exp_max) <= 1e-12
            and abs(cfg.shape_circum_radius - expect_coef * cfg.effective_space_radius) <= 1e-12
        )
        # 附加的**体积不等式余量**核算：给定实现解出的 R_min，形状体积必须**刚好**够
        # （>= 需求），且在略小处**不再够** —— 这直接检验"R_min 作为非重叠下界的紧致性"。
        got_vol = vol_coef * cfg.min_space_radius ** 3
        tol = max(1e-9, 1e-9 * need_vol)
        vol_tight = got_vol >= need_vol - tol
        vol_not_loose = vol_coef * (cfg.min_space_radius * 0.999) ** 3 < need_vol
        rep.check(
            "S3b", f"尺寸窗口由体积不等式独立反解一致 [{tag}]", s3b_ok,
            f"R_min={cfg.min_space_radius:.9f}(独立反解 {exp_min:.9f}), "
            f"R_max={cfg.max_space_radius:.9f}(独立折算 {exp_max:.9f}), "
            f"ρ={cfg.shape_circum_radius:.9f}(期望 {expect_coef * cfg.effective_space_radius:.9f})",
        )
        rep.check(
            "S3e", f"体积不等式余量（R_min 紧致）[{tag}]", vol_tight and vol_not_loose,
            f"形状体积系数={vol_coef:.6f}, 需求体积={need_vol:.9f}, "
            f"实现 R_min 处体积={got_vol:.9f}（>=需求={vol_tight}），"
            f"0.999·R_min 处体积={vol_coef * (cfg.min_space_radius * 0.999) ** 3:.9f}"
            f"（<需求={vol_not_loose}）",
        )

        # ---- S3c 窗口越界必须被拒（越界拒绝判据）----
        # [!] 判据强度说明（离朱第 2 轮 F3 提示，如实披露）：下面两个探针取 `R_max` 的
        #     **相对倍数**，当缺陷把 `R_max` 本身放大时（如 D1 少除 √3）它们**不会失败** ——
        #     真正钉住 `R_max` **绝对值**的是 S3b。故 S3c 的准确定位是"给定正确 `R_max`
        #     前提下的越界拒绝判据"，**不是** D1 类缺陷的独立防线。
        #     另注：曾尝试加入"固定 `space_radius=1.0`"的绝对探针，但实测量出该值对
        #     `sphere(R_max=1.4037)` 与 `cylinder(λ=0.5)(R_max=1.2555)` 而言**仍在合法窗口内**
        #     （构造期断言：绝对探针本身是错的），故改为依赖 S3b 的独立公式比对。
        rejected: List[str] = []
        probes: List[float] = [
            max(cfg.max_space_radius * 1.001, cfg.max_space_radius + 1e-6),
            cfg.max_space_radius * 1.2,
        ]
        for bad_sr in probes:
            try:
                Config(**{**size_kw, "shape": shape, "cyl_aspect": lam, "space_radius": float(bad_sr),
                          "input_dim": 784, "output_dim": 10, "seed": 42, "device": "cpu"})
                rejected.append(f"ACCEPTED {bad_sr:.6f}")
            except ValueError:
                pass
        rep.check(
            "S3c", f"超出 R_max 的 space_radius 被拒 [{tag}]", not rejected,
            f"探针 {[round(p, 6) for p in probes]}；{rejected if rejected else '全部被拒'}",
        )

        # ---- S3d 已删除（如实记录一次失败的尝试）----
        # 曾尝试加入"窗口比值自洽性"判据 `R_max/R_min == (B/H)/circum_coef`，期望它不依赖
        # `R_max` 的绝对量级而能**独立**抓出 D1 类缺陷。**实测该推导不成立**：`R_min` 本身
        # 也含形状相关的体积系数（`cube: (Nπ/(6φ))^(1/3)`、`cylinder: (N/(πλφ))^(1/3)·2^(2/3)`），
        # 故比值并非 `(B/H)/coef` 的简单形式。该判据在**正确代码**上也失败
        # （实测 cube 1.432638 vs 误推期望 1.154700），属**断言本身错误**，已删除。
        # 结论：钉住 D1 类缺陷的**正确**判据是 **S3（circum_coef 与内联独立公式）** 与
        #      **S3b（R_min/R_max/ρ 与内联独立公式）** —— 二者直接用绝对公式比对，
        #      不依赖任何"推导出来的关系"。

        # ---- S4 selection_metric 自洽 ----
        recheck = float(model._shape_metric(model.neuron_pos.to(torch.float64)).max().item())
        rep.check(
            "S4", f"selection_metric 自洽 [{tag}]",
            abs(recheck - float(model.selection_metric)) <= 1e-6,
            f"复算={recheck:.6f} vs 自报={model.selection_metric:.6f}",
        )

        # ---- S5 选取集合 == 独立复算（逐位）----
        ref = independent_selection(cfg)
        rep.check(
            "S5", f"选取集合==独立复算(最近N个) [{tag}]",
            torch.equal(ref, model.neuron_pos),
            f"torch.equal={torch.equal(ref, model.neuron_pos)}；形状={tuple(model.neuron_pos.shape)}",
        )

        # ---- S7 窗口非空 + 默认值落在窗口内 ----
        s7_ok = (
            cfg.max_space_radius >= cfg.min_space_radius
            and cfg.min_space_radius - 1e-9 <= cfg.effective_space_radius <= cfg.max_space_radius + 1e-9
        )
        rep.check(
            "S7", f"尺寸窗口非空且默认值在内 [{tag}]", s7_ok,
            f"R_min={cfg.min_space_radius:.6f} <= R_max={cfg.max_space_radius:.6f}, "
            f"eff={cfg.effective_space_radius:.6f}",
        )

        # ---- S9 DAG 三件套 ----
        rep.check(
            "S9", f"DAG 无环/严格上行/拓扑序覆盖 [{tag}]",
            int(sc["dag_acyclic"]) == 1
            and int(sc["all_edges_uphill"]) == 1
            and int(sc["topo_covers_all"]) == 1
            and int(sc["topo_matches_axis_order"]) == 1,
            f"acyclic={int(sc['dag_acyclic'])}, uphill={int(sc['all_edges_uphill'])}, "
            f"covers={int(sc['topo_covers_all'])}, topo==axis={int(sc['topo_matches_axis_order'])}",
        )

        # ---- S10 scope 非零 + 掩码与孤立突触一致 + K 自洽（S13）----
        axis_coord = model.neuron_pos[:, model.flow_axis_index]
        k_sync = k == int(torch.unique(axis_coord).numel())
        rep.check(
            "S10", f"scope 非零且掩码自洽 [{tag}]",
            s_in >= 1 and s_out >= 1
            and int(model.in_scope_mask.sum().item()) == s_in
            and int(model.out_scope_mask.sum().item()) == s_out,
            f"|S_in|={s_in}, |S_out|={s_out}, mask_sum={int(model.in_scope_mask.sum())}/"
            f"{int(model.out_scope_mask.sum())}",
        )
        rep.check(
            "S13", f"层数 K 与唯一轴坐标数自洽 [{tag}]", k_sync and k >= 2,
            f"K={k}, 唯一轴坐标={int(torch.unique(axis_coord).numel())}",
        )

        # ---- S14 `neuron_pos` 排序口径的**如实取证**（离朱实测 D2，继承自二期）----
        # 二期与本模块 `_build_fcc_positions` 都用**缩放整数 key**
        # `axis_coord * (N+1) + shell_rank` 做稳定排序，并把它当作"按 (轴坐标, 度量名次)
        # 字典序"。该缩放 key 等价于真字典序的**必要条件**是
        # `最小非零轴坐标差 × (N+1) > N-1`；FCC 晶格最小轴差为 `a/2 = √2·H`，实测不成立
        # （SMALL：0.212132×65 = 13.79 < 63），故**索引顺序并非轴坐标升序**，
        # 即"`neuron_pos` == 真字典序"这一表述**不成立**。
        # ⚠️ 但该行为**继承自二期**：若在本模块单方面改成真字典序，`shape="sphere"` 将与二期
        #    不再逐位一致，直接违反硬约束"默认分支必须与二期张量级逐位一致"。
        #    故本项**不改实现**，只把事实做成可复核判据：
        #      (a) 索引顺序**不**保证轴坐标升序（记录逆序位置数，不判失败）；
        #      (b) `topo_index` 覆盖全部神经元且每条边严格上行（真正的功能判据，必须成立）；
        #      (c) 选取**集合**与独立复算逐位一致（S5，已单独核验）。
        zc = model.neuron_pos[:, model.flow_axis_index]
        inversions = int((zc[1:] - zc[:-1] < -1e-9).sum().item())
        rep.check(
            "S14", f"排序口径如实取证（继承二期的缩放 key）[{tag}]",
            int(sc["topo_covers_all"]) == 1 and int(sc["all_edges_uphill"]) == 1,
            f"索引顺序与轴升序不一致的位置数={inversions}/{max(cfg.N - 1, 1)}"
            f"（**已知继承口径**，与二期同式；不影响选取集合/拓扑/前向）；"
            f"拓扑序覆盖={int(sc['topo_covers_all'])}, 全部边严格上行={int(sc['all_edges_uphill'])}",
        )

        # ---- 实测记录（形状报告的数据源）----
        rep.record(f"{size_tag}|{shape}|lam={lam:g}", {
            "shape": shape, "cyl_aspect": float(model.cyl_aspect),
            "N": n, "H": float(cfg.H), "D": float(cfg.D), "seed": int(cfg.seed),
            "E": e, "E_over_N": round(e / n, 6),
            "avg_out_degree": round(float(st["avg_out_degree"]), 6),
            "max_out_degree": int(st["max_out_degree"]),
            "K": k, "S_in": s_in, "S_out": s_out,
            "params": int(model.count_parameters()),
            "R_min": round(float(cfg.min_space_radius), 6),
            "R_max": round(float(cfg.max_space_radius), 6),
            "space_radius": round(float(model.space_radius), 6),
            "placement_radius": round(float(model.placement_radius), 6),
            "selection_metric": round(float(model.selection_metric), 6),
            "selection_metric_within_space": bool(model.selection_metric_within_space),
            "extent_xy": round(float(ts["neuron_extent_xy"]), 6),
            "extent_axis": round(float(ts["neuron_extent_axis"]), 6),
            "nearest_neighbour_dist": round(nn, 9),
            "lambda_range": _lam_range(model),
        })


def _lam_range(model: ThreeDNeuronSpace) -> str:
    """返回同规模其他形状的 K 取值区间（用于报告"K 随 λ 变化"的证据）。"""
    return "见全表"


def check_shape_changes(rep: Report) -> None:
    """S6：证明形状**真的生效**（不是只改了裁剪掩码）。"""
    print("\n===== S6 形状生效性（三形状互相不同的证据）=====", flush=True)
    size_kw = SIZE_CASES[0][1]
    built: Dict[str, ThreeDNeuronSpace] = {}
    for shape, lam in SHAPE_CASES:
        built[f"{shape}|{lam:g}"] = ThreeDNeuronSpace(make_config(size_kw, shape, lam))
    keys = list(built.keys())
    # (a) 覆盖集合两两不同
    distinct = True
    triplets = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            same = torch.equal(built[keys[i]].neuron_pos, built[keys[j]].neuron_pos)
            triplets.append(f"{keys[i]} vs {keys[j]}: equal={same}")
            distinct = distinct and not same
    # (b) 输出形状统计量两两不同（E / K / extents 至少一项不同）
    sigs = {
        k: (m.num_edges, int(m.get_topology_stats()["num_layers_true"]),
            round(float(m.get_topology_stats()["neuron_extent_xy"]), 6),
            round(float(m.get_topology_stats()["neuron_extent_axis"]), 6))
        for k, m in built.items()
    }
    sig_distinct = len(set(sigs.values())) == len(sigs)
    rep.check(
        "S6a", "同配置下三形状 neuron_pos 两两不同（形状真的生效）", distinct,
        "; ".join(triplets[:4]) + (" ..." if len(triplets) > 4 else ""),
    )
    rep.check(
        "S6b", "同配置下 (E, K, extents) 组合两两不同", sig_distinct,
        "; ".join(f"{k}->{v}" for k, v in sigs.items()),
    )
    # (c) K 随长径比单调变化的实证（λ=0.5/1/2 的 K）
    cy = {lam: int(built[f"cylinder|{lam:g}"].get_topology_stats()["num_layers_true"])
          for lam in (0.5, 1.0, 2.0)}
    rep.check(
        "S6c", "圆柱层数 K 随长径比 lambda 变化（架构深度随形状改变）",
        cy[0.5] < cy[2.0],
        f"K(lambda=0.5)={cy[0.5]}, K(lambda=1)={cy[1.0]}, K(lambda=2)={cy[2.0]}"
        f"（lambda 越大越细高 -> 层数越多）",
    )
    rep.record("S6_shape_effect", {
        "neuron_pos_pairwise_equal": triplets,
        "signatures": {k: list(v) for k, v in sigs.items()},
        "cylinder_K_by_lambda": cy,
    })


def check_negatives(rep: Report) -> None:
    """S8：负例 —— 拒绝静默无效参数与几何不可行的配置。"""
    print("\n===== S8 负例 =====", flush=True)
    # (a) 非 cylinder 显式 cyl_aspect
    try:
        Config(N=256, y_in=8, y_out=8, H=0.1, D=0.1, shape="cube", cyl_aspect=2.0)
        ok_a, msg_a = False, "未报错（缺陷：静默无效参数）"
    except ValueError as exc:
        ok_a, msg_a = True, str(exc)[:110]
    rep.check("S8a", "非 cylinder 显式 cyl_aspect 报错", ok_a, msg_a)
    # (b) 极端 lambda 空窗口
    try:
        Config(N=256, y_in=8, y_out=8, H=0.1, D=0.1, shape="cylinder", cyl_aspect=5.0)
        ok_b, msg_b = False, "未报错（缺陷：窗口为空仍构造）"
    except ValueError as exc:
        ok_b, msg_b = True, str(exc)[:110]
    rep.check("S8b", "极端 lambda 空窗口报错", ok_b, msg_b)
    # (c) 非法 shape
    try:
        Config(N=256, y_in=8, y_out=8, H=0.1, D=0.1, shape="torus")
        ok_c, msg_c = False, "未报错"
    except ValueError as exc:
        ok_c, msg_c = True, str(exc)[:110]
    rep.check("S8c", "非法 shape 报错", ok_c, msg_c)
    # (d) space_radius 越界
    try:
        c = Config(N=256, y_in=8, y_out=8, H=0.1, D=0.1, shape="cube", space_radius=0.01)
        ok_d, msg_d = False, f"未报错（R_min={c.min_space_radius:.6f}）"
    except ValueError as exc:
        ok_d, msg_d = True, str(exc)[:110]
    rep.check("S8d", "space_radius 越界报错", ok_d, msg_d)


def check_large_n_tolerance(rep: Report, full: bool = True) -> None:
    """S15：**大规模 2H 容差常驻回归**（离朱 R6/R8 建议补上的防线）。

    动机
    ----
    第二道（float32 坐标）容差含常数项 `3e-6`，用于覆盖 `torch.cdist` 的**距离域累加**伪影。
    若今后有人把该常数改小，`verify_shape.py` 原本**仍会全绿**（其规模矩阵最高只到
    `N=1024`，而该伪影在 `N=3072/4096` 才饱和到 `2.471e-6`）—— 即**大规模容差没有常驻防线**。
    本判据按离朱 R8 建议，把 `N=2048/3072/4096` 纳入验证脚本。

    判据
    ----
    * 对 `N ∈ {2048, 3072, 4096} × 5 形状`（`all/all, D=0.10`）逐组构造，**必须全部成功**
      （含离朱 R6 记录的全部失败点：`cube@3072`、`sphere@4096`、`cylinder λ=2@3072` 等）；
    * 同时核算实测偏差与容差的比值，报告最紧余量（须留有正向余量）。

    运行模式
    --------
    `N=4096` 单组构造需约 50s（`cdist [32768, 32768]`），全矩阵约 13 分钟，
    故与 `check_phase2_equality` 同口径：**仅在全量模式下运行**，
    `--quick` 跳过（否则会把快速验证拉到十几分钟，失去"快速"意义）。
    """
    if not full:
        print("\n===== S15 大规模 2H 容差常驻回归：--quick 模式跳过（耗时约 13 min，"
              "仅在 --quick 之外的全量模式运行）=====", flush=True)
        return
    n_list = (2048, 3072, 4096)
    print("\n===== S15 大规模 2H 容差常驻回归（N=2048/3072/4096 × 5 形状）=====", flush=True)
    worst_ratio = 0.0
    worst_tag = ""
    failures: List[str] = []
    for N in n_list:
        for shape, lam in SHAPE_CASES:
            cfg = Config(
                N=N, y_in=8, y_out=8, H=0.1, D=0.1, flow_axis="z",
                shape=shape, cyl_aspect=lam,
                input_scope="all_isolated", readout_scope="all_isolated",
                input_dim=784, output_dim=10, batch_size=64, lr=1e-3, epochs=10,
                seed=42, device="cpu",
            )
            try:
                m = ThreeDNeuronSpace(cfg)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"N={N} {shape} lam={lam}: {type(exc).__name__} {str(exc)[:90]}")
                print(f"  [FAIL] N={N} {shape} lam={lam}: {type(exc).__name__}")
                continue
            scale = max(1.0, float(m.neuron_pos.abs().max().item()))
            tol = 1e-6 * scale + 3e-6
            dev = abs(float(m.get_topology_stats()["nearest_neighbour_dist"]) - 2.0 * cfg.H)
            ratio = dev / tol
            if ratio > worst_ratio:
                worst_ratio, worst_tag = ratio, f"N={N}/{shape}/lam={lam}"
            print(f"  [ ok ] N={N:<5} {shape:9s} lam={lam:<4}: E={m.num_edges:<6d} "
                  f"dev={dev:.3e} tol={tol:.3e} ratio={ratio:.3f}")
    rep.check(
        "S15", "大规模容差回归：N=2048/3072/4096 × 5 形状全部构造成功",
        not failures,
        f"失败 {len(failures)} 项" + (f"：{failures[:3]}" if failures else "")
        + f"；最紧余量 ratio={worst_ratio:.3f}（{worst_tag}，<1 即通过）",
    )
    rep.record("S15_large_n", {"n_list": list(n_list), "failures": failures,
                               "worst_ratio": worst_ratio, "worst_tag": worst_tag})


def check_boundary_anchors(rep: Report, full: bool = True) -> None:
    """S16：**加固前容差的"首个失败 N"锚点**（离朱 R11 建议 2）。

    动机
    ----
    `README` §15.6(a) 表登记的四个"实测首个失败 `N`"（`cube D=0.10`/`cube D=0.072` → 1720、
    `sphere D=0.10` → 3256、`cylinder λ=2 D=0.10` → 2865）此前**仅以文档表格形式存在**。
    [!] **本判据首次运行即捕获一处文档错误**：`cylinder λ=2 D=0.10` 的锚点原登记为
    `2899`（来自 R6 稀疏采样上界），本判据逐点二分得到真值 **2865**（N=2864 ratio 0.6027 通过 /
    N=2865 ratio 1.1647 失败），已修正 README、spec 与本锚点表。这正是本判据存在的意义。
    若后续有人调小第二道常数项 `tol_const`、或改动采样/选取逻辑，**文档与实测会脱节**而无人发现
    （S15 只覆盖"加固后容差下全部通过"，不覆盖"加固前口径的边界位置"）。

    [!] **职责边界（离朱 R12 建议 4）**：本判据**不读**产品常量 `tol_const`（实测不引用），
    因此**不承担「常数项漂移」的守护** —— 那是 **S15** 的职责
    （实测：常数项取 0 时 S15 最紧点 `N=3072/cube/lam=1` 的 ratio 由 `0.578220` 跃至 `1.941097` → FAIL）。
    两者职责互补：**S16** 守「文档锚点与采样/选取逻辑的一致性」，
    **S15** 守「加固后容差在大规模下的覆盖性」。

    判据
    ----
    对每个锚点，用**加固前口径** `tol_old = 1e-6·max(1.0, max|coord|)`（无 `+3e-6`）
    直接读取 `model.neuron_pos` 计算 `cdist` 最近邻距与 `2H` 的偏差（**不触发**契约断言），
    断言边界相邻性成立：`N = 首个失败 N - 1` 处 `ratio <= 1`，且 `N = 首个失败 N` 处 `ratio > 1`。

    成本
    ----
    实测单点 0.05–0.1s（**远小于**离朱预估的 38s；因 `all/all` 下 `|S_in|/|S_out|` 小而
    `cdist` 规模可控），4 个锚点 × 2 点合计 < 1s。仍与 S15 同口径**仅全量模式运行**，
    以保持 `--quick` 极速。
    """
    if not full:
        print("\n===== S16 首个失败 N 锚点：--quick 模式跳过 =====", flush=True)
        return
    anchors = [
        ("cube", 1.0, 0.10, 1719, 1720),
        ("cube", 1.0, 0.072, 1719, 1720),
        ("sphere", 1.0, 0.10, 3255, 3256),
        ("cylinder", 2.0, 0.10, 2864, 2865),
    ]
    print("\n===== S16 加固前口径「首个失败 N」锚点（2H 契约边界）=====", flush=True)
    bad: List[str] = []
    for shape, lam, D, n_pass, n_fail in anchors:
        r_pass = _tol_old_ratio(shape, lam, D, n_pass)
        r_fail = _tol_old_ratio(shape, lam, D, n_fail)
        ok = (r_pass <= 1.0) and (r_fail > 1.0)
        if not ok:
            bad.append(f"{shape}/lam={lam}/D={D}: N={n_pass} ratio={r_pass:.4f}, "
                       f"N={n_fail} ratio={r_fail:.4f}")
        print(f"  [{' ok ' if ok else 'FAIL'}] {shape:9s} lam={lam:<4} D={D:<7}: "
              f"N={n_pass} ratio={r_pass:.4f}（须<=1）, N={n_fail} ratio={r_fail:.4f}（须>1）")
    rep.check(
        "S16", "首个失败 N 锚点：每个锚点「前一点通过、该点失败」成立",
        not bad, f"不成立 {len(bad)} 项" + (f"：{bad}" if bad else "（4 个锚点全部成立）"),
    )
    rep.record("S16_anchors", {
        "anchors": [{"shape": s, "lam": l, "D": d, "n_pass": a, "n_fail": b,
                     "ratio_pass": _tol_old_ratio(s, l, d, a),
                     "ratio_fail": _tol_old_ratio(s, l, d, b)} for s, l, d, a, b in anchors],
        "failures": bad,
    })


def _tol_old_ratio(shape: str, lam: float, D: float, N: int) -> float:
    """用**加固前口径**容差计算 `dev / tol_old`（不触发 `2H` 契约断言）。

    `tol_old = 1e-6 · max(1.0, max|coord|)`（**无** `+3e-6` 常数项），
    即 §15.6(a) 表登记"首个失败 N"时所处的判据口径。
    """
    cfg = Config(
        N=N, y_in=8, y_out=8, H=0.1, D=D, flow_axis="z", shape=shape, cyl_aspect=lam,
        input_scope="all_isolated", readout_scope="all_isolated",
        input_dim=784, output_dim=10, batch_size=64, lr=1e-3, epochs=10,
        seed=42, device="cpu",
    )
    # 直接调用内部放置方法不可行（其中含断言）；改由独立复算取坐标（与 S5 同源的独立路径）。
    pos32 = independent_selection(cfg)
    d = torch.cdist(pos32, pos32)
    d.fill_diagonal_(float("inf"))
    nn = float(d.min().item())
    tol_old = 1e-6 * max(1.0, float(pos32.abs().max().item()))
    return abs(nn - 2.0 * cfg.H) / tol_old


def check_fingerprint(rep: Report) -> None:
    """S11：指纹含形状维度，且同配置不同形状产物名互不相同（防撞名硬要求）。"""
    print("\n===== S11 产物指纹可区分性 =====", flush=True)
    verify_names: Dict[str, List[str]] = {}
    for shape, lam in SHAPE_CASES:
        cfg = make_config(SIZE_CASES[0][1], shape, lam)
        k = f"{shape}|lam={lam:g}"
        verify_names[k] = [
            train_mod.config_fingerprint(cfg, 0),
            train_mod.full_checkpoint_name(cfg),
            train_mod.smoke_checkpoint_path(
                cfg.flow_axis, cfg.input_scope, cfg.readout_scope, "neuron3d",
                train_mod.smoke_fingerprint(cfg),
            ),
        ]
    all_names = [n for v in verify_names.values() for n in v]
    distinct = len(set(all_names)) == len(all_names)
    rep.check(
        "S11a", "三形状 x 三种产物名 全部互不相同", distinct,
        f"共 {len(all_names)} 个名字，去重后 {len(set(all_names))} 个",
    )
    has_shape = all(
        ("shapesphere" in v[0]) or ("shapecube" in v[0]) or ("shapecylinder" in v[0])
        for v in verify_names.values()
    )
    rep.check("S11b", "指纹显式包含形状段（含 sphere）", has_shape,
              "; ".join(f"{k} -> {v[0]}" for k, v in verify_names.items()))
    lam_dims = {k: v[0] for k, v in verify_names.items() if k.startswith("cylinder")}
    rep.check(
        "S11c", "圆柱指纹含长径比（三档 lambda 互不相同）",
        len({v for v in lam_dims.values()}) == len(lam_dims),
        "; ".join(f"{k} -> {v}" for k, v in lam_dims.items()),
    )
    rep.record("S11_names", verify_names)


def check_phase2_equality(rep: Report) -> None:
    """S12：sphere 分支与二期 n3d_sphere 张量级逐位相等（跨代码路径不比文件 SHA）。"""
    print("\n===== S12 sphere 分支 vs 二期 n3d_sphere（张量级）=====", flush=True)
    try:
        from n3d_sphere.config import Config as Phase2Config  # noqa: WPS433
        from n3d_sphere.model import ThreeDNeuronSpace as Phase2Model  # noqa: WPS433
    except Exception as exc:  # pragma: no cover
        rep.check("S12a", "可导入二期 n3d_sphere（比对前提）", False,
                  f"导入失败：{type(exc).__name__}: {exc}")
        return
    rep.check("S12a", "可导入二期 n3d_sphere（比对前提）", True, "导入成功")

    tensors = [
        "neuron_pos", "input_syn_pos", "output_syn_pos", "syn_dist",
        "edge_src", "edge_dst", "edge_dist",
        "representative_syn_out", "representative_syn_input",
        "topo_index", "edge_perm", "edge_perm_in", "neuron_in_edge_reach",
        "edge_dst_in", "level_edge_reach", "level_node_reach",
        "in_scope_mask", "out_scope_mask", "in_degree", "out_degree",
        "input_isolated_mask", "output_isolated_mask", "neuron_conn_mask",
    ]
    params = ["edge_weight", "neuron_bias", "W_in", "W_out"]
    for size_tag, size_kw in SIZE_CASES:
        for shape, lam in SHAPE_CASES:
            if shape != "sphere":
                continue
            mine = ThreeDNeuronSpace(make_config(size_kw, shape, lam))
            # 二期 Config 只接受二期的字段：按 to_dict 逐字段过滤
            mine_dict = make_config(size_kw, shape, lam).to_dict()
            phase2_kw = {
                k: v for k, v in mine_dict.items()
                if k not in ("shape", "cyl_aspect")
            }
            other = Phase2Model(Phase2Config(**phase2_kw))
            bad = [n for n in tensors + params
                   if not torch.equal(getattr(mine, n), getattr(other, n))]
            rep.check(
                "S12b", f"全部张量 torch.equal [{size_tag}/sphere]", not bad,
                f"比对 {len(tensors) + len(params)} 个张量，不一致={bad if bad else '无'}；"
                f"E={mine.num_edges}/{other.num_edges}, "
                f"params={mine.count_parameters()}/{other.count_parameters()}",
            )
            rep.record(f"S12_{size_tag}", {
                "tensors_compared": len(tensors) + len(params),
                "mismatches": bad,
                "num_edges": [mine.num_edges, other.num_edges],
                "params": [mine.count_parameters(), other.count_parameters()],
                "note": "依仓库 D1 口径：跨代码路径不比文件 SHA256，而是 torch.load 后逐张量比对",
            })


def main(argv: List[str] | None = None) -> int:
    """入口：跑全部断言并落盘报告。返回 0（全通过）/ 1（有失败项）。"""
    parser = argparse.ArgumentParser(description="n3d_shape 形状变体独立验证")
    parser.add_argument("--quick", action="store_true", help="只跑 SMALL 规模（加速）")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("n3d_shape 形状变体独立验证开始")
    print(f"形状矩阵：{SHAPE_CASES}")
    print(f"规模矩阵：{[t for t, _ in SIZE_CASES]}{'（--quick: 仅 SMALL）' if args.quick else ''}")
    print("=" * 78, flush=True)

    rep = Report()
    sizes = [SIZE_CASES[1]] if args.quick else SIZE_CASES
    try:
        for size_tag, size_kw in sizes:
            run_shape_matrix(rep, size_tag, size_kw)
        check_shape_changes(rep)
        check_negatives(rep)
        check_fingerprint(rep)
        check_phase2_equality(rep)
        check_large_n_tolerance(rep, full=not args.quick)
        check_boundary_anchors(rep, full=not args.quick)
    except Exception:
        rep.check("X1", "验证脚本自身无异常", False, traceback.format_exc()[-800:])

    title = "n3d_shape 形状变体验证报告"
    if args.quick:
        # ⚠️ 离朱信息项 I1（已修复）：`--quick` 原先与全量写**同一份**报告文件，
        #    跑一次快速验证会把全量报告（103 条）覆盖成快速版（73 条），
        #    使"报告落盘的断言条数"与最近一次运行模式绑定，复核时容易误读。
        #    现按模式分名落盘，两份报告互不覆盖。
        title += "（--quick：仅 SMALL 规模）"
    md = rep.to_markdown(title)
    os.makedirs(VERIFY_DIR, exist_ok=True)
    stem = "verify_shape_report_quick" if args.quick else "verify_shape_report"
    md_path = os.path.join(VERIFY_DIR, f"{stem}.md")
    js_path = os.path.join(VERIFY_DIR, f"{stem}.json")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(js_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "all_ok": rep.all_ok,
                "checks": rep.items,
                "records": rep.records,
            },
            fh, ensure_ascii=False, indent=2, default=str,
        )
    n_fail = sum(1 for it in rep.items if not it["ok"])
    print("\n" + "=" * 78)
    print(f"断言总数 {len(rep.items)}；通过 {len(rep.items) - n_fail}；失败 {n_fail}")
    print(f"报告已写入：{md_path}")
    print(f"机读结果已写入：{js_path}")
    print(f"结论：{'全部通过' if rep.all_ok else '存在失败项'}")
    print("=" * 78, flush=True)
    return 0 if rep.all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
