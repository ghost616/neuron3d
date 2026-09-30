"""几何权重场 P0 诊断脚本（**仪表化，非门槛**）：读既有 free 权重 checkpoint，回归 `edge_weight ~ φ_e`。

职责
----
回答一个**纯诊断**问题：**未经几何场训练的** baseline 里，`edge_weight` 的取值与边级几何
特征是否已经存在相关性？（若存在，则"几何场能带来增益"这一假说在 P0 阶段就有先验信号；
若不存在，也不构成对假说的否证 —— 本脚本**只报数，不下结论**。）

输出
----
* **R^2**：`edge_weight` 对几何特征（含 RBF 基）的线性可解释比例；
* **等价类内方差**：按 `(流向轴位移 ζ, 横向位移 ρ)` 归并等价类后，类内 `edge_weight` 方差
  占比 —— 用于判断"同一几何位置的边是否已有系统性权重差异"；
* **以 `in_degree` 为协变量的偏效应**：`edge_weight` 的初始化界是 `1/√fan_in`
  （`fan_in = max_in_degree`，全模型同一个标量），而 `in_degree` 与几何天然相关，故必须
  把 `in_degree` 放进设计矩阵才能把"几何信号"与"度数信号"分开；
* **初始化权重的 R^2 作零假设基线**：对同一批特征，分别用
  (a) 从 checkpoint 读到的实际 `edge_weight`、(b) **按初始化分布重新抽样**的权重，
  各算 R^2；两者之差才是"几何信号"的证据量（**假说基线**）。

判定规则（事先固定）
--------------------
**纯诊断，不下结论**：本脚本的 R^2 只作参考，**不得**据此关闭或宣称"几何权重场假说成立"。
样本量 = 单个 seed 的边数 E（DEFAULT 规模实测 700~1000 条），自由度充裕但**单 seed 单点**。

用法
----
    python n3d_shape/probe_geo_field.py [--checkpoint <path>] [--seed 0] [--rbf-k 12]
                                        [--shape sphere|cube|cylinder] [--cyl-aspect L]
                                        [--json <out.json>]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

try:
    from .config import Config  # noqa: E402
    from .model import GEO_EDGE_FEATURE_NAMES, ThreeDNeuronSpace  # noqa: E402
except ImportError:  # pragma: no cover
    from config import Config  # type: ignore
    from model import GEO_EDGE_FEATURE_NAMES, ThreeDNeuronSpace  # type: ignore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "checkpoints", "n3d_shape",
    "full_shapesphere_N256_y8x8_H0.1_D0.1_plfcc_axz_isall_rsall_fc-1_s42.pt",
)


# ----------------------------------------------------------------------
# 线性代数（纯 torch，无新依赖）
# ----------------------------------------------------------------------
def _r2(y: torch.Tensor, design: torch.Tensor) -> Tuple[float, torch.Tensor]:
    """返回 `(R^2, 系数)`：对 `y` 做带截距的最小二乘回归 `y ~ design`。

    `design` 的形状为 `[n, p]`（**不含截距列**，本函数内部补一列全 1）。
    用 `torch.linalg.lstsq`（SVD 最小二乘），避免显式求逆带来的数值不稳。
    """
    n = int(y.numel())
    ones = torch.ones(n, 1, dtype=design.dtype)
    x = torch.cat([ones, design], dim=1)                       # [n, p+1]
    coef = torch.linalg.lstsq(x, y.reshape(-1, 1)).solution   # [p+1, 1]
    pred = (x @ coef).reshape(-1)
    ss_res = float(((y - pred) ** 2).sum().item())
    ss_tot = float(((y - y.mean()) ** 2).sum().item())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return r2, coef.reshape(-1)


def _partial_effect(
    y: torch.Tensor, geo: torch.Tensor, cov: torch.Tensor
) -> Tuple[float, torch.Tensor]:
    """以 `cov`（协变量，此处为 `in_degree`）为额外控制变量，返回几何的**偏 R^2**与系数。

    口径：`R^2(geo + cov) - R^2(cov)` —— 即"在已控制 `cov` 之后，几何特征额外解释的比例"。
    """
    r_cov, _ = _r2(y, cov)
    r_both, coef = _r2(y, torch.cat([geo, cov], dim=1))
    return r_both - r_cov, coef


# ----------------------------------------------------------------------
# 特征与等价类
# ----------------------------------------------------------------------
def _geo_features(model: ThreeDNeuronSpace, rbf_k: int) -> Tuple[torch.Tensor, List[str]]:
    """从**现场重建**的模型取边级几何特征（不依赖产物内的特征张量 —— 它们不落盘）。

    返回 `(feat [E, F'], 列名)`，其中：

    * 前若干列 = `edge_geo_feat`（归一化几何特征，含 `zeta/rho/dhat/slack/mult`）；
    * 其后 = RBF 基激活 `φ_k(φ_e)`（由 `model._geo_basis` 现算，k 由命令行给定）；
    * 末列 = `in_degree[edge_dst]`（**协变量**，初始化界 `1/√fan_in` 的代理量）。
    """
    if not model.geo_enabled:
        raise RuntimeError(
            "现场重建的模型 geo_field == 'none'，无法取几何特征；"
            "请用 --geo-field 打开（本脚本默认以 additive 档重建模型仅为取特征，"
            "产物内的 edge_weight 与几何场无关）"
        )
    feat = model.edge_geo_feat.detach()
    names = [f"feat:{n}" for n in model._geo_feature_names]
    basis = model._geo_basis(feat).detach()
    names += [f"rbf:{i}" for i in range(int(basis.shape[1]))]
    deg = model.in_degree.index_select(0, model.edge_dst.to(torch.long)).to(
        torch.float32
    ).reshape(-1, 1)
    return torch.cat([feat, basis, deg], dim=1), names + ["cov:in_degree"]


def _equivalence_classes(
    model: ThreeDNeuronSpace,
) -> Tuple[torch.Tensor, int]:
    """按 `(流向轴位移 ζ, 横向位移 ρ)` 归并等价类，返回 `(类标签 [E], 类数)`。

    这两列由 `neuron_pos` 的**中心位移**唯一决定（FCC 规则晶格上取值高度离散），
    是最自然的"几何等价类"定义：同一类内的边在**几何上不可区分**（除突触级采样/入度外）。
    """
    raw = model.edge_geo_feat_raw.detach()
    names = list(model._geo_feature_names)
    zeta = raw[:, names.index("zeta")].round(decimals=6)
    rho = raw[:, names.index("rho")].round(decimals=6)
    keys = zeta * 1.0e6 + rho
    uniq, inverse = torch.unique(keys, return_inverse=True)
    return inverse.to(torch.long), int(uniq.numel())


def _within_class_variance_ratio(y: torch.Tensor, labels: torch.Tensor) -> float:
    """返回**类内方差占比** `Σ_g Σ_{i∈g}(y_i - ybar_g)^2 / Σ_i (y_i - ybar)^2`（0 = 类间全解释）。"""
    total = float(((y - y.mean()) ** 2).sum().item())
    if total <= 0.0:
        return float("nan")
    within = 0.0
    for g in torch.unique(labels):
        mask = labels == g
        yg = y[mask]
        if int(yg.numel()) > 1:
            within += float(((yg - yg.mean()) ** 2).sum().item())
    return within / total


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def run_probe(
    checkpoint: str,
    seed: int,
    rbf_k: int,
    shape: str,
    cyl_aspect: float,
    null_repeats: int,
    signed_delta: bool = False,
) -> Dict[str, object]:
    """执行 P0 诊断并返回机读结果字典。"""
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = dict(payload["config"])
    # 现场重建模型：**打开 geo_field**（仅为复算边级几何特征；产物内的 `edge_weight`
    # 是"纯 free 权重"，与该开关无关）。
    cfg_dict["geo_field"] = "additive"
    cfg_dict["geo_rbf_k"] = int(rbf_k)
    cfg_dict["device"] = "cpu"
    cfg_dict["seed"] = int(seed)
    # [!] 皋陶审查 F3（warning，已修复）：`--shape` / `--cyl-aspect` 原先被 argparse 接受并
    #     传入本函数，却**全程未使用** —— `--shape cube` 会静默沿用产物内形状，属"静默无效
    #     参数"，与本模块既有纪律（`--cyl-aspect` 非 cylinder 拒绝、`--fc-dim` + `--arch mlp`
    #     拒绝）冲突。
    #
    #     处置选择：**实现为"产物配置断言"**（而不是皋陶建议的"直接写回 cfg_dict"，
    #     也不是"删除这两个 CLI 项"）。理由由**实测**给出：
    #       写回 shape 会让现场重建模型与产物**拓扑不一致**（sphere E=736 vs cube E=713），
    #       `load_state_dict` 立即报 `size mismatch for edge_dist/edge_weight/...`
    #       （实测报文见 SUMMARY）—— 因为**产物的权重形状由其自身形状唯一决定**，
    #       不可能"跨形状重算同名权重"。故"写回"会引入一个**新的必然失败路径**；
    #       而"删除"会削弱工具（诊断跨形状配置正是本脚本的自然用途之一）。
    #     断言口径：显式给出时必须与**产物内 config 逐值一致**，否则抛可读错误并说明原因；
    #     空串 / 哨兵 -1 = 不校验（沿用产物内取值）。这样"显式给出即生效（校验）"，
    #     不再有静默无效参数。
    if shape:
        art_shape = str(cfg_dict.get("shape", "sphere"))
        if str(shape) != art_shape:
            raise ValueError(
                f"--shape={shape!r} 与产物内 shape={art_shape!r} 不一致：本脚本读的是"
                f"**既有 free 权重产物**，其权重形状（E / |S_in| / |S_out| / 各拓扑量）"
                f"由产物自身的 shape 唯一决定，**不能跨形状重算**。"
                f"请改用 --shape {art_shape}（或不传该参数）。"
            )
    if float(cyl_aspect) > 0.0:
        art_lam = float(cfg_dict.get("cyl_aspect", 1.0))
        if abs(float(cyl_aspect) - art_lam) > 1e-12:
            raise ValueError(
                f"--cyl-aspect={float(cyl_aspect)} 与产物内 cyl_aspect={art_lam} 不一致："
                f"同上，产物权重形状由产物自身形状/长径比唯一决定，不能跨其重算。"
                f"请改用 --cyl-aspect {art_lam:g}（或不传该参数）。"
            )
    # 可选扩展开关（F9：`geo_signed_delta` 在 train.py CLI 中暂不暴露，此处仅供诊断）。
    # 该开关只改变**特征列数 F**（5 -> 7），不影响拓扑/权重形状，故可安全写回。
    cfg_dict["geo_signed_delta"] = bool(signed_delta)
    model = ThreeDNeuronSpace(Config(**cfg_dict))

    # ---- 载入产物权重（离朱 DEF-2 修复：此处原先漏了 load_state_dict）----
    # 不载入时 `model.edge_weight` 只是**按 seed 重新初始化的随机权重**，
    # 于是 R^2 描述的是"随机初始化权重 vs 几何"，而非"产物里的自由权重 vs 几何" ——
    # P0 诊断口径整体失效（且数值天然落在初始化零假设基线带内，会"看起来恰好一致"）。
    #
    # 载入口径（两处**预期差异**必须显式登记，而不是靠 `strict=False` 默默放过）：
    #   * `missing`：现场重建模型**打开了 `geo_field`**（只为复算边级几何特征），
    #     故 4 个几何张量（`geo_rbf_theta` / `geo_alpha` / `geo_rbf_centers` /
    #     `geo_rbf_width`）不在**既有 free 权重产物**里 —— 且它们与 `edge_weight` 无关；
    #   * `unexpected`：nosyn 口径下产物**不含**那 8 个突触类 buffer（`persistent=False`）。
    # 除这两组之外的任何 missing / unexpected 都是真实不一致，必须报错。
    sd = payload.get("model_state_dict")
    if not isinstance(sd, dict) or "edge_weight" not in sd:
        raise RuntimeError(
            f"产物缺少 model_state_dict['edge_weight']，无法做 P0 诊断：{checkpoint}"
        )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    expected_missing = {
        "geo_rbf_theta", "geo_alpha", "geo_rbf_centers", "geo_rbf_width",
    }
    expected_unexpected = {
        "input_syn_pos", "output_syn_pos", "syn_dist", "representative_syn_out",
        "representative_syn_input", "neuron_conn_mask", "input_isolated_mask",
        "output_isolated_mask",
    }
    odd_missing = sorted(set(missing) - expected_missing)
    odd_unexpected = sorted(set(unexpected) - expected_unexpected)
    if odd_missing or odd_unexpected:
        raise RuntimeError(
            f"载入产物权重出现**非预期**差异：missing={odd_missing}，"
            f"unexpected={odd_unexpected}（产物与现场重建模型不一致，P0 诊断口径无效）"
        )
    load_info: Dict[str, object] = {
        "strict": False,
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
        "expected_missing_reason": "现场重建模型打开了 geo_field，产物是 free 权重产物",
        "expected_unexpected_reason": "nosyn 口径：8 个突触类 buffer persistent=False",
    }
    # **自检判据**：载入后的 `edge_weight` 必须与产物逐位相同，否则本次诊断口径无效
    w_loaded = model.edge_weight.detach().clone()
    w_artifact = sd["edge_weight"].detach().clone()
    weight_bitwise_equal = bool(torch.equal(w_loaded, w_artifact))
    if not weight_bitwise_equal:
        raise RuntimeError(
            "载入后 edge_weight 与产物不逐位相同，P0 诊断口径无效"
            f"（max|diff|={float((w_loaded - w_artifact).abs().max()):.6g}）"
        )
    # 全部**公共**持久化张量也逐位比对（更宽的等价取证，不只 edge_weight 一项）
    sd_model = model.state_dict()
    common_keys = sorted(set(sd_model) & set(sd))
    common_mismatch = [
        k for k in common_keys if not torch.equal(sd_model[k], sd[k])
    ]
    if common_mismatch:
        raise RuntimeError(
            f"载入后公共张量与产物不逐位相同：{common_mismatch}"
        )

    weight = model.edge_weight.detach().reshape(-1)
    n_edges = int(weight.numel())
    design, names = _geo_features(model, int(rbf_k))
    geo_cols = [i for i, n in enumerate(names) if not n.startswith("cov:")]
    cov_cols = [i for i, n in enumerate(names) if n.startswith("cov:")]
    geo = design[:, geo_cols]
    cov = design[:, cov_cols]

    # (a) 实际 edge_weight 的解释力
    r2_geo, coef_geo = _r2(weight, geo)
    r2_cov, _ = _r2(weight, cov)
    r2_both, _ = _r2(weight, torch.cat([geo, cov], dim=1))
    partial, _ = _partial_effect(weight, geo, cov)

    # (b) 零假设基线：按初始化分布重新抽样（bound = 1/sqrt(max_in_degree)），重复若干次
    bound = 1.0 / math.sqrt(float(max(int(model.in_degree.max().item()), 1)))
    gen = torch.Generator().manual_seed(int(seed) + 10007)
    null_r2: List[float] = []
    for _ in range(max(1, int(null_repeats))):
        null_w = torch.empty(n_edges).uniform_(-bound, bound, generator=gen)
        r, _ = _r2(null_w, geo)
        null_r2.append(float(r))
    null_mean = sum(null_r2) / len(null_r2)
    null_std = (
        math.sqrt(sum((v - null_mean) ** 2 for v in null_r2) / len(null_r2))
        if len(null_r2) > 1 else 0.0
    )

    # (c) 等价类内方差
    labels, n_classes = _equivalence_classes(model)
    within_ratio = _within_class_variance_ratio(weight, labels)
    onehot = torch.nn.functional.one_hot(labels, num_classes=n_classes).to(torch.float32)
    # 去掉第一列（避免与内部截距列共线），其余类哑变量作为设计矩阵
    r2_classes, _ = _r2(weight, onehot[:, 1:] if n_classes > 1 else onehot)

    # (d) 标准化后的自由权重（用同一批样本重新标准化，消除"界"的影响）
    # [!] 皋陶审查 F8（info，已修复）：原先把偏效应写成 `partial_z = r2_geo_z - 0.0`
    #     （无意义死赋值），且同处算出的 `r2_cov_z` 从未进入结果字典 / 报告。
    #     现按 `r2_geo_z - r2_cov_z` **真算**，并把三个量全部写入返回字典与 `_render`。
    if float(weight.std()) > 0.0:
        free_z = (weight - weight.mean()) / weight.std()
        r2_geo_z, _ = _r2(free_z, geo)
        r2_cov_z, _ = _r2(free_z, cov)
        # 标准化权重上"控制 in_degree 后几何的额外解释力"（与 r2.geo_partial_given_in_degree 同口径）
        partial_z = r2_geo_z - r2_cov_z
    else:  # pragma: no cover
        r2_geo_z = r2_cov_z = partial_z = float("nan")

    return {
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "config": {
            k: cfg_dict[k] for k in
            ("N", "y_in", "y_out", "H", "D", "shape", "cyl_aspect",
             "input_scope", "readout_scope", "seed")
            if k in cfg_dict
        },
        "n_edges": n_edges,
        "weight_source": "artifact",
        "weight_loaded_from_artifact": True,
        "weight_bitwise_equal_to_artifact": weight_bitwise_equal,
        "common_tensors_compared": len(common_keys),
        "common_tensors_mismatch": common_mismatch,
        "load_state_dict": load_info,
        "feature_columns": names,
        "weight_init_bound": bound,
        "max_in_degree": int(model.in_degree.max().item()),
        "init_null_r2": {
            "repeats": len(null_r2),
            "mean": null_mean,
            "std": null_std,
            "min": min(null_r2),
            "max": max(null_r2),
        },
        "r2": {
            "geo_only": r2_geo,
            "in_degree_only": r2_cov,
            "geo_plus_in_degree": r2_both,
            "geo_partial_given_in_degree": partial,
            "standardized_free_weight_geo": r2_geo_z,
            "standardized_free_weight_in_degree": r2_cov_z,
            # F8 修复：标准化权重上"控制 in_degree 后几何的额外解释力"（真算，非死赋值）
            "standardized_free_weight_partial": partial_z,
        },
        "equivalence_classes": {
            "definition": "按 (zeta, rho) 取 6 位小数归并（中心位移的几何等价类）",
            "n_classes": n_classes,
            "within_class_variance_ratio": within_ratio,
            "between_class_r2": r2_classes,
        },
        "note": (
            "P0 诊断仅作参考：R^2 为**单 seed 单点**实测，且 RBF 中心/宽度只由几何决定，"
            "故 R^2 高不代表因果、R^2 低不否证假说。**不得据此关闭或宣称几何权重场假说成立。**"
        ),
    }


def _sha256(path: str) -> str:
    """返回文件的 SHA256（十六进制小写）。"""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _render(res: Dict[str, object]) -> str:
    """把结果渲染为人类可读报告（Markdown 片段）。"""
    r2 = res["r2"]  # type: ignore[index]
    null = res["init_null_r2"]  # type: ignore[index]
    eq = res["equivalence_classes"]  # type: ignore[index]
    lines = [
        "# n3d_shape 几何权重场 P0 诊断（仪表化，非门槛）",
        "",
        f"- checkpoint：`{res['checkpoint']}`",
        f"- SHA256：`{res['checkpoint_sha256']}`",
        f"- 配置：{res['config']}",
        f"- 边数 E：{res['n_edges']}（单 seed 单点）",
        f"- 权重来源：**产物 `model_state_dict['edge_weight']`**"
        f"（`load_state_dict(strict=False)`；与产物逐位相同 = "
        f"{res['weight_bitwise_equal_to_artifact']}；"
        f"公共持久化张量比对 {res['common_tensors_compared']} 个、"
        f"不一致 {len(res['common_tensors_mismatch'])} 个）"
        f"（离朱 DEF-2 修复：此前误用按 seed 重建的随机初始化权重）",
        f"- `edge_weight` 初始化界 `1/sqrt(fan_in)`：{res['weight_init_bound']:.6g}"
        f"（`max_in_degree`={res['max_in_degree']}）",
        f"- 特征列（{len(res['feature_columns'])}）：{res['feature_columns']}",  # type: ignore[arg-type]
        "",
        "## R^2（`edge_weight` 对几何特征 / 协变量）",
        "",
        "| 口径 | R^2 |",
        "| --- | --- |",
        f"| 几何特征（含 RBF 基） | {r2['geo_only']:.6f} |",  # type: ignore[index]
        f"| 仅 `in_degree`（协变量） | {r2['in_degree_only']:.6f} |",  # type: ignore[index]
        f"| 几何 + `in_degree` | {r2['geo_plus_in_degree']:.6f} |",  # type: ignore[index]
        f"| **偏效应**（控制 `in_degree` 后几何额外解释） | {r2['geo_partial_given_in_degree']:.6f} |",  # type: ignore[index]
        f"| 标准化自由权重的几何 R^2 | {r2['standardized_free_weight_geo']:.6f} |",  # type: ignore[index]
        f"| 标准化自由权重：仅 `in_degree` 的 R^2 | {r2['standardized_free_weight_in_degree']:.6f} |",  # type: ignore[index]
        f"| 标准化自由权重：**偏效应**（F8 修复，真算 `geo_z - in_degree_z`） | {r2['standardized_free_weight_partial']:.6f} |",  # type: ignore[index]
        "",
        "## 零假设基线（按初始化分布 `U(-b, b)` 重新抽样）",
        "",
        f"- 重复 {null['repeats']} 次：R^2 mean={null['mean']:.6f}，std={null['std']:.6f}，"
        f"min={null['min']:.6f}，max={null['max']:.6f}",  # type: ignore[index]
        "",
        "## 等价类（按 (zeta, rho) 归并）",
        "",
        f"- 类数：{eq['n_classes']}；**类内方差占比**={eq['within_class_variance_ratio']:.6f}"
        f"（0 = 权重完全由这组几何决定，1 = 与几何无关）",
        f"- 类标签对 `edge_weight` 的 R^2：{eq['between_class_r2']:.6f}",
        "",
        "## 判定规则（事先固定）",
        "",
        "- **纯诊断**：只报数，**不下结论**；",
        "- R^2 为**单 seed 单点**实测，高不代表因果、低不否证假说；",
        "- 与初始化零假设基线之差才是『几何信号』的证据量（但本脚本不据此做任何判定）。",
        "",
        f"> {res['note']}",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="几何权重场 P0 诊断（读既有 free 权重 checkpoint，回归 edge_weight ~ φ_e）"
    )
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                   help="既有 free 权重产物路径（默认取 DEFAULT 规模的 fc-1 产物）")
    p.add_argument("--seed", type=int, default=42, help="重建模型用的 seed（默认 42）")
    p.add_argument("--rbf-k", type=int, default=12, help="RBF 基个数（默认 12）")
    p.add_argument("--shape", type=str, default="",
                   help=(
                       "**校验**产物内 shape（空串 = 不校验）：必须与产物 config 中的 shape "
                       "逐值一致，否则报错。**F3 修复后不再静默无效** —— 本脚本读既有 free "
                       "权重产物，其权重形状由产物自身 shape 唯一决定，不能跨形状重算"
                       "（实测写回 shape 会使 load_state_dict 报 size mismatch）"
                   ))
    p.add_argument("--cyl-aspect", type=float, default=-1.0,
                   help=(
                       "**校验**产物内圆柱长径比 lambda（哨兵 -1 = 不校验）：必须与产物 "
                       "config 逐值一致，否则报错。**F3 修复后不再静默无效**"
                   ))
    p.add_argument("--null-repeats", type=int, default=50,
                   help="零假设基线重复次数（默认 50）")
    p.add_argument("--json", type=str, default="", help="机读结果输出路径（可选）")
    p.add_argument(
        "--signed-delta",
        action="store_true",
        default=False,
        help=(
            "把现场重建模型的 geo_signed_delta 置 True（追加 signed dx/dy 两列特征）。"
            "**注意（皋陶 F9）**：`geo_signed_delta` 在 `train.py` 的 CLI 中**暂不暴露**，"
            "此处是该开关在本模块内的唯一命令行入口（仅供诊断使用）。"
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """入口：跑诊断、打印报告、可选落盘 JSON。返回 0（成功）/ 1（前置条件不满足）。"""
    args = parse_args(argv)
    try:
        if not os.path.isfile(args.checkpoint):
            print(f"[FAIL] checkpoint 不存在：{args.checkpoint}", flush=True)
            return 1
        res = run_probe(
            args.checkpoint, int(args.seed), int(args.rbf_k),
            args.shape, float(args.cyl_aspect), int(args.null_repeats),
            bool(args.signed_delta),
        )
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] 诊断失败：{type(exc).__name__}: {exc}", flush=True)
        return 1
    print(_render(res), flush=True)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        print(f"[OK] 机读结果已写入：{os.path.abspath(args.json)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
