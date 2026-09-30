"""N3D 神经元空间**形状变体**（球体 / 立方体 / 圆柱体）超参数配置层。

本模块只承载超参数与常量，不含任何计算逻辑。与二期 `n3d_sphere` 的唯一差别是
新增了**形状维度**：`shape ∈ {sphere, cube, cylinder}` 与圆柱长径比 `cyl_aspect = λ`。

形状 = 生长度量，不是裁剪掩码
-----------------------------
二期放置是两步：① `keep = dist <= R_search` 裁剪；② `argsort(dist, stable=True)[:N]`
取离中心最近的 N 个。**第②步才是产生球分布的机制**（已实测：仅把第①步换成同尺度
立方体裁剪，选取集合与球体逐位相同）。故本模块**只替换第②步的排序度量**，保留
"从中心向外取最近 N 个"的语义与全部确定性 tie-break。

| shape      | 生长度量 `m(p)`                        | 等值面                       |
|------------|----------------------------------------|------------------------------|
| `sphere`   | `‖p‖2`                                 | 球（回归锚点，与二期逐位一致） |
| `cube`     | `‖p‖∞ = max(|x|,|y|,|z|)`              | 立方体                       |
| `cylinder` | `max(‖p_xy‖2, |p_axis| / λ)`           | 半径 r、半高 `c = λr` 的圆柱  |

形状尺度与尺寸窗口 `[R_min, R_max]`
-----------------------------------
`space_radius` 是**形状的特征空间尺度**（球 = 半径；立方体 = 外接立方体半边长；
圆柱 = 外接圆柱的横截面半径，半高 `= λ·space_radius`）。`space_radius = 0.0`（默认）
表示取该形状的公式下界 `R_min`；显式传入时必须落在 `[R_min, R_max]` 内，越界在构造期报错。

* **`R_min`（非重叠容纳下界）**：由"最优堆积系数 `φ = 0.7405` 下非重叠放入 N 个
  半径 H 的神经元所需的最小空间体积"推出，三种形状**各自代入自身体积**：

      sphere   : (4/3)πR^3        >= N·(4/3)πH^3/φ   -> R_min = H·(N/φ)^(1/3)
      cube     : (2s)^3           >= N·(4/3)πH^3/φ   -> s_min = H·(Nπ/(6φ))^(1/3)
      cylinder : 2πλr^3           >= N·(4/3)πH^3/φ   -> r_min = H·(2N/(3λφ))^(1/3)

  圆柱体积为 `πr^2·(2c) = 2πλr^3`（`r` = 横截面半径、`c = λr` = 沿轴半高）。

  [!] **历史缺陷（皋陶审查 E1，已修复）**：`cylinder` 分支原先写作
  `H·(N/(πλφ))^(1/3)·2^(2/3)`，对应把圆柱体积误当成 `(4/3)πλr^3`（球体积形式），
  与上式**恒差 `(6/π)^(1/3) = 1.240701`**（偏大 24.07%）；本行也曾把
  `(4/3)πλr^3·… = 2πλr^3` 写成**不成立的等式**（两者差 1.5 倍）。该错误抬高 `R_min`
  后把 DEFAULT 规模下 λ∈(2.20, 3.22) 的合法配置**误判为"窗口为空"而拒绝**
  （如 λ=2.5 误报 `R_max=0.521314 < R_min=0.560482`）。现改用严格体积解。

* **`R_max`（D 邻域完整落在空间内的上界）**：把二期球体的公式
  `R_max = (H+D)·(N/φ)^(1/3)` 按"特征半尺度 P_max = B·(N/φ)^(1/3)，`B = H+D`"推广，
  再由形状的外接半径系数 `ρ/R` 折算回特征尺度：

      sphere   : ρ/R = 1     -> R_max = B·(N/φ)^(1/3)          （= 二期原式，逐位保留）
      cube     : ρ/s = √3    -> s_max = B·(N/φ)^(1/3)/√3
      cylinder : ρ/r = √(1+λ^2)-> r_max = B·(N/φ)^(1/3)/√(1+λ^2)

[!] **实测披露（实测非臆造）**：`R_max` 并非"放置一定落在其中"的严格保证 ——
DEFAULT 口径下实测 `placement_radius / R_max` 为：`sphere` **0.5137**（落在窗口内）、
`cube` **1.2090**、`cylinder(λ=1)` 0.8308、`λ=0.5` 0.6944、`λ=2` **1.7157**
（详见 `model.py` 的 W2 披露与 README §3）。规格未被违反：窗口校验只作用于
**用户显式传入的 `space_radius`**，放置本身由形状度量排序决定，故本模块把
`R_min` / `R_max` 定位为**公式窗口与回归锚点**，几何守护由构造期**实测诊断量**
（`selection_metric` / `placement_radius` 及其 `*_within_*` 标志，见 `get_topology_stats()`）
如实记录。圆柱在极端 `λ` 下会出现 `R_max < R_min`（窗口为空，DEFAULT 口径首个空窗口约
λ≈3.235）→ 构造期直接报错并附实测值。

参考值（`N=256, H=0.10, D=0.10`，DEFAULT/HIGHACC 口径，`B = 0.20`）：
`sphere` 窗口 `[0.701840, 1.403681]`；`cube` 窗口 `[0.565680, 0.810415]`；
`cylinder(λ=1)` 窗口 `[0.613114, 0.992552]`；`cylinder(λ=0.5)` 窗口 `[0.772475, 1.255490]`；
`cylinder(λ=2)` 窗口 `[0.486629, 0.627745]`（极窄但非空）。
对照 SMALL 口径 `N=64, H=0.15, D=0.15`（`B = 0.30`）：`sphere [0.663198, 1.326395]`、
`cube [0.534535, 0.765795]`、`cylinder(λ=1) [0.579356, 0.937903]`、
`cylinder(λ=0.5) [0.729943, 1.186364]`、`cylinder(λ=2) [0.459835, 0.593182]`。

**连接半径硬约束 `D <= H`**：连接判据是"起点神经元的输出突触 `o` 与终点神经元的输入
突触 `j` 的距离 `<= D`"，而 `o` / `j` 各自落在所属神经元的 `H` 半径球内，故 `D` 不得
超过接收/发送范围半径 `H`。本模块三个预设一律取 `D = H`；`D > H` 在构造期直接报错。
此外图还有**连通性下限**（`E >= N`、层数 `K >= 2`、`|S_in| >= 1`、`|S_out| >= 1`），
由 `ThreeDNeuronSpace.__init__` 的 `check_connectivity_floor()` 把关。

神经元放置
----------
神经元坐标**不是随机采样**，而是 FCC（面心立方）规则堆积：晶格常数
`a = 2√2·H`，使最近邻距恰为 `2H`（相邻神经元的 H 半径突触云恰好相切、不重叠）。
形状只改变"从中心向外取最近 N 个"的**度量**，因此几何仍然完全确定、**与 seed 无关**。

关键不变量
----------
* `N > 0`、`y_in > 0`、`y_out > 0`；
* `H > 0`、`D > 0`；
* **`D <= H`**（连接半径不得超过接收/发送范围半径；本模块三个预设取 `D = H`）；
* `shape ∈ {"sphere", "cube", "cylinder"}`；
* `cyl_aspect > 0`，且**仅 `shape == "cylinder"` 时允许显式传入**（其余形状显式传入报错，
  避免静默无效参数）；
* `flow_axis ∈ {"x", "y", "z"}`；
* `geo_field ∈ {"none", "additive", "class_tied", "mlp"}`（本批**实现** `none` 与 `additive`；
  `class_tied` / `mlp` 仅接受枚举并在构造期**显式报错"未实现"**，避免静默降级）；
* `geo_rbf_k >= 1`、`geo_hidden >= 1`、`geo_alpha_init >= 0`；
* `input_scope ∈ {"any_isolated", "all_isolated"}`、`readout_scope` 同理；
* `space_radius == 0.0`（默认，表示取该形状的 `R_min`）或 `space_radius ∈ [R_min, R_max]`；
* 可学习参数的**连通性下限**（`E >= N`、层数 `K >= 2`、`|S_in| >= 1`、`|S_out| >= 1`）
  在 `ThreeDNeuronSpace.__init__` 中校验，不满足即抛异常。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

__all__ = [
    "Config",
    "ShapeSpec",
    "SMALL_CONFIG",
    "DEFAULT_CONFIG",
    "HIGHACC_CONFIG",
    "FLOW_AXIS_CHOICES",
    "SCOPE_CHOICES",
    "PLACEMENT_CHOICES",
    "SHAPE_CHOICES",
    "GEO_FIELD_CHOICES",
    "GEO_IMPLEMENTED_CHOICES",
    "CYLINDER_ASPECT_SENTINEL",
    "PACKING_PHI",
    "shape_spec",
    "min_space_radius",
    "max_space_radius",
    "SHAPE_VARIANT_NOTE",
    "GEO_FIELD_NOTE",
]

# 全局流向轴：输入突触取负半球（-axis）、输出突触取正半球（+axis）
FLOW_AXIS_CHOICES = ("x", "y", "z")

# 输入层驱动判据 / 读出判据的取值域
SCOPE_CHOICES = ("any_isolated", "all_isolated")

# 神经元放置方式：当前仅支持 FCC 规则堆积（不存在随机放置分支）
PLACEMENT_CHOICES = ("fcc",)

# ---- 形状维度（本模块相对二期 n3d_sphere 的唯一新增维度）----
# sphere   = 球体（回归锚点，默认分支必须与二期张量级逐位一致）
# cube     = 立方体
# cylinder = 圆柱体（横截面半径 r、沿流向轴半高 c = λr）
SHAPE_CHOICES = ("sphere", "cube", "cylinder")

# `cyl_aspect` 的"未提供"哨兵：只在显式赋值时参与"非 cylinder 形状禁止传入"的负例判定。
# 语义：`Config(shape="cube")` 合法（λ 取该字段默认值 1.0 但**不生效**）；
#      `Config(shape="cube", cyl_aspect=2.0)` 报错（避免静默无效参数）。
CYLINDER_ASPECT_SENTINEL: float = float("nan")

# 最优堆积系数（面心立方最密堆积）：φ = π/(3√2) ≈ 0.740480
PACKING_PHI: float = 0.7405

# 本模块为**形状变体**（球体 / 立方体 / 圆柱体）；保留此常量便于验证脚本做"形状维度已落地"断言
SHAPE_VARIANT_NOTE: str = "neuron_shape_variant:sphere|cube|cylinder"

# ---- 几何权重场（geo_field，本模块第 5 轮新增维度：形状 = 生长度量 + 几何权重场）----
# `none`       = **关闭**（默认）：**连几何特征都不构造**，buffer 不注册、参数不创建，
#                代码路径与改动前逐位一致（`shape=sphere` 仍与二期张量级 `torch.equal` 一致）；
# `additive`   = 本批**已实现**档：`w_e = w_free[e] + alpha · (Σ_k c_k · φ_k(φ_e) + c_0)`，
#                `c` 零初始化、`alpha` 初值 = `geo_alpha_init`（可学习）；
# `class_tied` = **枚举已接受但本批未实现**（构造期显式报错，不静默降级为 none）；
# `mlp`        = **枚举已接受但本批未实现**（构造期显式报错，不静默降级为 none）。
GEO_FIELD_CHOICES: Tuple[str, ...] = ("none", "additive", "class_tied", "mlp")
# 本批**真正实现**的子集（其余取值必须显式报错，而不是被静默忽略）
GEO_IMPLEMENTED_CHOICES: Tuple[str, ...] = ("none", "additive")

# 保留此常量便于验证脚本做"几何权重场维度已落地"断言
GEO_FIELD_NOTE: str = "neuron_geo_weight_field:none|additive"


@dataclass(frozen=True)
class ShapeSpec:
    """某一形状的几何规格（纯派生量，不含可学习/可调字段）。

    属性
    ----
    name : str
        形状名（`sphere` / `cube` / `cylinder`）。
    cyl_aspect : float
        圆柱长径比 `λ = c / r`（c = 沿流向轴半高，r = 横截面半径）；非圆柱恒为 1.0。
    circum_coef : float
        外接半径与特征尺度之比 `ρ / R`：
        `sphere -> 1`、`cube -> √3`、`cylinder -> √(1 + λ^2)`。
        用于把"特征尺度 R"折算成形状的外接半径（D 邻域上界推导的桥梁量）。
    """

    name: str
    cyl_aspect: float
    circum_coef: float

    def describe(self) -> str:
        """人类可读摘要：`shape=cube` 或 `shape=cylinder(lambda=0.5)`。"""
        if self.name == "cylinder":
            return f"shape=cylinder(lambda={self.cyl_aspect:g})"
        return f"shape={self.name}"


def shape_spec(shape: str, cyl_aspect: float = 1.0) -> ShapeSpec:
    """把形状名与长径比归一化为 `ShapeSpec`（**形状逻辑的唯一入口**）。

    参数
    ----
    shape : str
        `sphere` / `cube` / `cylinder`。
    cyl_aspect : float
        圆柱长径比 λ（> 0）；非圆柱形状忽略该值并归一化为 1.0。

    返回
    ----
    ShapeSpec
        归一化后的形状规格。

    异常
    ------
    ValueError
        `shape` 非法，或 `cylinder` 的 λ 非正时抛出。
    """
    if shape not in SHAPE_CHOICES:
        raise ValueError(f"shape 仅允许 {SHAPE_CHOICES}，当前 shape={shape!r}")
    if shape == "sphere":
        # 球体：外接半径 = 自身半径 → ρ/R = 1（该分支使 R_max 与二期逐位同式，回归锚点）
        return ShapeSpec(name="sphere", cyl_aspect=1.0, circum_coef=1.0)
    if shape == "cube":
        # 立方体：特征尺度 s 为半边长，外接半径 = 体对角线之半 = √3·s → ρ/R = √3。
        # [!] 历史缺陷（离朱实测 D1，已修复）：此处曾误写为"所有非 cylinder 形状都返回 1.0"，
        #    导致 `max_space_radius` 少除一个 √3 —— cube 的 `R_max` 与校验窗口被放大 √3 倍
        #    （实测 `N=256,H=D=0.1`：代码 `1.403681` vs 文档 `0.810415`，比值恰为 1.732051），
        #    于 `Config(shape="cube", space_radius=1.0/1.2/1.403681)` 这类**越界配置被静默接受**。
        #    `neuron_pos` / `E` / `K` / `|S_in|` / `|S_out|` / `params` / `test_acc` 不受影响
        #    （搜索半径取 `max(ρ, R_max)`，两种取值下候选集合都足够），但**配置校验严格性**
        #    与对外报告的派生量（`R_max` / `circum_coef` / `shape_circum_radius`）必须正确。
        return ShapeSpec(name="cube", cyl_aspect=1.0, circum_coef=math.sqrt(3.0))
    lam = float(cyl_aspect)
    if not (lam > 0.0):
        raise ValueError(f"cylinder 的 cyl_aspect 必须 > 0，当前 cyl_aspect={cyl_aspect}")
    return ShapeSpec(name="cylinder", cyl_aspect=lam, circum_coef=math.sqrt(1.0 + lam * lam))


def min_space_radius(shape: str, N: int, H: float, cyl_aspect: float = 1.0) -> float:
    """返回该形状的**特征尺度下界** `R_min`（非重叠容纳 N 个半径 H 的神经元）。

    推导（三种形状共用同一体积论证，仅代入自身体积）
    ------------------------------------------------
    最优堆积系数 `φ = 0.7405` 下，N 个半径 H 的球所需总体积为 `N·(4/3)πH^3/φ`；
    令**该形状自身的体积**不小于该值，即得特征尺度下界（`sqrt` 为开方）：

    * `sphere`（半径 R）：`V = (4/3)πR^3`
      `(4/3)πR^3 >= N(4/3)πH^3/φ` → `R_min = H·(N/φ)^(1/3)`（与二期同式）；
    * `cube`（半边长 s）：`V = (2s)^3 = 8s^3`
      `8s^3 >= N(4/3)πH^3/φ` → `s_min = H·(Nπ/(6φ))^(1/3)`；
    * `cylinder`（横截面半径 r、沿轴半高 `c = λr`）：`V = πr^2·(2c) = πr^2·(2λr) = 2πλr^3`
      `2πλr^3 >= N(4/3)πH^3/φ` → `r^3 >= 2N H^3/(3λφ)` → `r_min = H·(2N/(3λφ))^(1/3)`。

    [!] **历史缺陷（皋陶审查 E1，已修复）**：`cylinder` 分支原先写作
    `H·(N/(πλφ))^(1/3)·2^(2/3)`，它对应的是把圆柱体积误当成 `(4/3)πλr^3`（球体积形式）而非
    真实的 `2πλr^3`，与本节所列前提**恒差 `(6/π)^(1/3) = 1.240701`**（实现偏大 24.07%）。
    实测（`N=256, H=0.1, λ=1`）：实现 `0.760691` vs 严格解 `0.613114`。
    该错误使 `R_min` 被人为抬高，从而把本应合法的配置**误判为"窗口为空"而拒绝**：
    DEFAULT 规模下 λ∈(2.20, 3.22) 被误拒（如 λ=2.5：误报 `R_max=0.521314 < R_min=0.560482`，
    而严格 `R_min=0.451746` 时窗口实为非空），且错误信息把原因归为
    "该形状装不下 N 个神经元"这一**不实的几何事实**。现改用严格体积解 `2N/(3λφ)`。

    [!] 该下界**不保证**放置选取集合一定落在其中（数值见模块 docstring 的实测披露），
    构造期另有实测断言 `selection_metric <= space_radius` 兜底。

    参数
    ----
    shape : str
        形状名。
    N : int
        神经元数量（> 0）。
    H : float
        神经元（突触云）半径（> 0）。
    cyl_aspect : float
        圆柱长径比 λ（> 0），仅 `cylinder` 生效。

    返回
    ----
    float
        该形状的特征尺度下界 `R_min`。

    异常
    ------
    ValueError
        形状非法 / `N <= 0` / `H <= 0` / λ <= 0 时抛出。
    """
    spec = shape_spec(shape, cyl_aspect)
    if int(N) <= 0:
        raise ValueError(f"计算 R_min 要求 N > 0，当前 N={N}")
    if not (float(H) > 0.0):
        raise ValueError(f"计算 R_min 要求 H > 0，当前 H={H}")
    n, h = float(N), float(H)
    if spec.name == "sphere":
        return h * (n / PACKING_PHI) ** (1.0 / 3.0)
    if spec.name == "cube":
        return h * (n * math.pi / (6.0 * PACKING_PHI)) ** (1.0 / 3.0)
    lam = spec.cyl_aspect
    # 严格体积解：由 `2πλr^3 >= N·(4/3)πH^3/φ` 解出 `r >= H·(2N/(3λφ))^(1/3)`。
    # （历史缺陷皋陶 E1：原写作 `H·(N/(πλφ))^(1/3)·2^(2/3)`，对应把圆柱体积误当成
    #  `(4/3)πλr^3`，与上述前提恒差 (6/π)^(1/3)=1.240701，导致 λ∈(2.20,3.22) 被误拒。）
    return h * (2.0 * n / (3.0 * lam * PACKING_PHI)) ** (1.0 / 3.0)


def max_space_radius(
    shape: str, N: int, H: float, D: float, cyl_aspect: float = 1.0
) -> float:
    """返回该形状的**特征尺度上界** `R_max`（D 邻域完整落在空间内）。

    推导：把二期的 `R_max = B·(N/φ)^(1/3)`（`B = H + D`）解释为"特征半尺度上界"，再由
    形状的外接半径系数 `ρ/R`（`sphere: 1`、`cube: √3`、`cylinder: √(1+λ^2)`）折回特征尺度：
    `R_max = B·(N/φ)^(1/3) / (ρ/R)`。`sphere` 分支因此与二期**逐位同式**（回归锚点）。

    [!] 圆柱在极端 λ 下会出现 `R_max < R_min`（窗口为空），此时 `Config` 构造期直接报错并附实测值。
    DEFAULT 口径（`N=256, H=D=0.1`）按**修复后的严格 `R_min`** 实测：

    | λ | `R_min` | `R_max` | 窗口 |
    | --- | --- | --- | --- |
    | 2.0 | 0.486628883 | 0.627745046 | 非空 |
    | 2.5 | 0.451746238 | 0.521313886 | 非空 |
    | 3.0 | 0.425109486 | 0.443882779 | 非空 |
    | 3.2 | 0.416061843 | 0.418682820 | 非空（极窄） |
    | 3.22 | 0.415198642 | 0.416311707 | 非空（极窄） |
    | **3.234921405**（`λ*`，交叉点） | **0.414559275** | **0.414559275** | **临界（`R_min == R_max`）** |
    | > `λ*`（如 3.24） | — | — | **空（窗口为空，构造被拒）** |

    即空窗口只在 **λ > λ\\* = 3.234921405** 才真实出现（临界点处 `R_min == R_max = 0.414559275`；
    迭代二分定位得到 `3.234934872`，与解析交叉点仅差 `1.3e-5`，属迭代精度），
    属"细高圆柱在给定 N/H 下确实装不下"的几何事实。

    [!] **历史缺陷（皋陶审查 E1，已修复）**：修复前 `R_min` 被抬高 `1.240701` 倍，
    导致 λ∈(2.20, 3.22) 被**误拒**（如 λ=2.5 误报 `R_max=0.521314 < R_min=0.560482`），
    而按严格解该区间窗口本应非空。
    [!] **本表曾笔误（离朱第 3 轮 F1，已修复）**：上表 λ=3.2 / λ=3.22 两行原写作
    `0.416861` / `0.415443`，且误标 λ=3.22 为"首个空窗口"；实测 λ=3.22 **窗口非空**，
    与模块 docstring 的"首个空窗口约 λ≈3.235"自相矛盾。现全部按实现精确复算值更正，
    使**模块 docstring / 本 docstring / 实现**三者一致。

    参数
    ----
    shape : str
        形状名。
    N : int
        神经元数量（> 0）。
    H : float
        神经元（突触云）半径（> 0）。
    D : float
        连接距离阈值（> 0）。
    cyl_aspect : float
        圆柱长径比 λ（> 0），仅 `cylinder` 生效。

    返回
    ----
    float
        该形状的特征尺度上界 `R_max`。

    异常
    ------
    ValueError
        形状非法 / `N <= 0` / `H <= 0` / `D <= 0` 时抛出。
    """
    spec = shape_spec(shape, cyl_aspect)
    if int(N) <= 0:
        raise ValueError(f"计算 R_max 要求 N > 0，当前 N={N}")
    if not (float(H) > 0.0):
        raise ValueError(f"计算 R_max 要求 H > 0，当前 H={H}")
    if not (float(D) > 0.0):
        raise ValueError(f"计算 R_max 要求 D > 0，当前 D={D}")
    scale = (float(N) / PACKING_PHI) ** (1.0 / 3.0)
    return (float(H) + float(D)) * scale / spec.circum_coef


@dataclass
class Config:
    """N3D 神经元空间**形状变体**（球体 / 立方体 / 圆柱体）的完整超参数集合。

    参数
    ----
    N : int
        神经元数量。默认 256。神经元按 FCC 规则堆积放置，故**几何与 seed 无关**。
    y_in : int
        每个神经元的输入突触数量（限制在流向轴负半球内按体积均匀采样）。默认 8。
    y_out : int
        每个神经元的输出突触数量（限制在流向轴正半球内按体积均匀采样）。默认 8。
    H : float
        神经元（突触云）半径。突触方向在单位球面均匀、半径 `r = H · u^(1/3)`
        （`u ~ U(0,1)`，`u^(1/3)` 为 **u 的立方根**，保证半球内**按体积均匀**，
        而非半径线性采样）。默认 0.1。
    D : float
        连接距离阈值：输出突触与输入突触距离 `<= D` 才可能建立连接。
        **硬约束 `D <= H`**（连接半径不得超过接收/发送范围半径），
        违反时 `__post_init__` 抛 `ValueError`。默认 0.1（= `H`）。
    flow_axis : str
        全局流向轴 `x` / `y` / `z`（默认 `z`）。输入突触取负半球（-axis）、
        输出突触取正半球（+axis），合法连接要求起点神经元的流向轴坐标低于终点。
    space_radius : float
        **形状特征空间尺度**。`0.0`（默认）表示取该形状的公式下界 `R_min`；显式指定时必须
        落在 `[R_min, R_max]` 内，否则构造期报错。语义随形状而变 ——
        球 = 半径；立方体 = 外接立方体半边长；圆柱 = 外接圆柱横截面半径（半高 `= λ·R`）。
    shape : str
        神经元空间的**空间形状**（本模块相对二期的新增维度）：
        `"sphere"`（球体，默认，回归锚点）/ `"cube"`（立方体）/ `"cylinder"`（圆柱体）。
        形状只替换"从中心向外取最近 N 个"的**生长度量**，不改连接判据、不改突触半球切分、
        不引入随机放置：
        `sphere -> ‖p‖2`、`cube -> ‖p‖∞`、`cylinder -> max(‖p_xy‖2, |p_axis|/λ)`。
    cyl_aspect : float
        圆柱长径比 `λ = c / r`（`c` = 沿流向轴半高、`r` = 横截面半径），默认 1.0，
        **仅 `shape="cylinder"` 生效**。非 `cylinder` 形状**显式**传入非默认值会在
        `__post_init__` 报错（避免静默无效参数）。
    placement : str
        神经元放置方式。**当前仅支持 `"fcc"`**（面心立方规则堆积，晶格常数
        `a = 2√2·H`，最近邻距恰为 `2H`）。不存在随机放置分支；保留该字段是为了让
        配置/产物指纹显式记录放置方式（便于后续扩展时区分产物）。默认 `"fcc"`。
    input_scope : str
        输入层驱动判据：`"any_isolated"`（神经元有 ≥1 个输入突触孤立即进入 S_in）
        / `"all_isolated"`（神经元全部 y_in 个输入突触都孤立才进入 S_in）。
        默认 `"any_isolated"`。
    readout_scope : str
        读出判据：`"any_isolated"`（神经元有 ≥1 个输出突触孤立即进入 S_out）
        / `"all_isolated"`（神经元全部 y_out 个输出突触都孤立才进入 S_out）。
        默认 `"any_isolated"`。
    input_dim : int
        输入维度（MNIST 展平为 784）。默认 784。
    output_dim : int
        输出类别数（MNIST 为 10）。默认 10。
    hidden_dim : int
        仅 `--arch mlp` 对照基线使用（主模型不使用该字段）。默认 2048。
    fc_dim : int
        **两端全连接包裹开关**（本模块第三轮新增的单一配置）。取值语义：
        `0` = **关闭**（默认，走现状路径：`W_in` / `W_out` 直连，参数集与数值**逐位不变**）；
        `-1` = **跟随 N**（两端有效宽度 `H = N`）；
        `> 0` = 显式宽度 `H = fc_dim`。
        启用后的结构：`x → Linear(784→H)+b+ReLU → 投影 P(|S_in|,H) → N3D 核心（不变）`
        `→ h = a_up[S_out] → Linear(|S_out|→H)+b+ReLU → Linear(H→10)+b → logits`。
        **启用时不再创建 `W_in` / `W_out` / `W_out_bias`**（它们被上面两组全连接层取代），
        输入侧偏置复用 `neuron_bias[in_scope_mask]`。`< -1` 在构造期报错。默认 0。
    geo_field : str
        **几何权重场开关**（本模块第 5 轮新增的"形状 = 生长度量 + 几何权重场"维度）：
        `"none"`（默认）= **关闭** —— **连几何特征都不构造**（不注册 `edge_geo_feat` 等
        buffer、不创建几何参数），代码路径与改动前**逐位一致**；
        `"additive"` = RBF 加性档（本批实现的唯一档）：每条边的有效权重为
        `w_e = w_free[e] + alpha · (Σ_k c_k · φ_k(φ_e) + c_0)`，`c` 零初始化、`alpha`
        初值 = `geo_alpha_init`（可学习）；
        `"class_tied"` / `"mlp"` = **枚举已接受但本批未实现** —— 构造期显式报错
        （拒绝把未实现档静默降级成 `none`）。
        几何场只改**权重取值**，不改连接判据、不改突触半球切分、不引入随机放置、
        不改训练循环与数据管线；`neuron_pos` / `edge_dist` 只读、语义不变。
    geo_rbf_k : int
        RBF 基函数个数 `k`（`additive` 档；必须 >= 1）。基中心与宽度由 **model 侧**从
        边级特征按**确定性分位点 / 逐维间距**算出（无额外随机数消耗），详见 README 的
        「几何权重场（geo_field）」节。
    geo_hidden : int
        预留字段（后续 `class_tied` / `mlp` 档的隐藏宽度；必须 >= 1）。本批不消费该
        字段，但必须保留在 `to_dict()` 中，否则 `train.apply_overrides` 经
        `Config(**base.to_dict())` 往返会**静默丢字段**。
    geo_alpha_init : float
        `additive` 档场增益 `alpha` 的初值（> = 0，可学习）。默认 1.0。
        注意：`c`（RBF 系数）**零初始化**，故 `alpha` 的初值只决定"零初始化不影响前向"
        之后的起点尺度，不影响开关开启时的初始前向逐位等于基线这一性质。
    geo_signed_delta : bool
        **可选扩展开关**（默认 `False`）：为边级几何特征追加 signed `dx/H`、`dy/H` 两列
        （可区分横向位移的左右方向）。**默认关闭时特征列集合与文档表格逐字一致**（5 列）。
        非 `additive` 档时该字段无效果（不报错 —— 它是纯扩展位，不改变任何语义）。
    batch_size : int
        批大小。默认 64。
    lr : float
        学习率。默认 1e-3。
    epochs : int
        正式训练轮数。默认 10。
    seed : int
        全局随机种子。**神经元位置由 FCC 确定、与 seed 无关**；seed 只影响
        突触采样（进而影响边集与边数）、参数初始化与数据打乱。默认 42。
    device : str
        计算设备，cpu / cuda / auto（auto 表示自动选择）。默认 auto。
    data_root : str
        MNIST 数据根目录（工程内已存在的 IDX 文件所在目录）。默认 data/mnist。
    num_workers : int
        DataLoader 工作进程数（Windows 下建议 0）。默认 0。
        **可复现性前提**：> 0 时由 `data._worker_init_fn` 按 (seed, worker_id)
        为每个 worker 独立播种（base_seed = torch.initial_seed()），因此固定 `seed`
        即可复现多进程取数顺序；= 0 时完全由主进程 `generator`（seed 控制）决定顺序。
        注意：per-worker 播种的基种子取自父进程的 `torch.initial_seed()`，因此
        **必须先调用 `utils.set_seed(config.seed)`**（`train.build_model_and_data`
        已自动完成），该 seed 才会生效。
    log_interval : int
        训练日志打印间隔（按 batch 计）。默认 100。

    # ---- 可选训练增强 ----
    weight_decay : float
        AdamW 的权重衰减系数；> 0 时优化器切换为 AdamW。默认 0.0（= 用 Adam）。
    readout_bias : bool
        是否为输出层 W_out 增加 bias。默认 False（不创建 bias，前向不加）。
    lr_schedule : str
        学习率调度，仅允许 "none" / "cosine"。默认 "none"（不做调度）。
    grad_clip : float
        梯度范数裁剪阈值；> 0 时在 backward 与 step 之间执行 clip_grad_norm_。
        默认 0.0（不裁剪）。
    """

    # ---- 拓扑规模 ----
    N: int = 256
    y_in: int = 8
    y_out: int = 8

    # ---- 球体几何 ----
    H: float = 0.1
    D: float = 0.1  # 硬约束 D <= H（见 __post_init__）；默认取 D = H
    flow_axis: str = "z"
    space_radius: float = 0.0
    placement: str = "fcc"

    # ---- 空间形状（本模块相对二期的新增维度）----
    shape: str = "sphere"              # sphere / cube / cylinder
    # 圆柱长径比 λ = c / r（c = 沿流向轴半高，r = 横截面半径）；**仅 cylinder 生效**。
    # 默认 1.0；非 cylinder 形状**显式**传入非默认值会在 __post_init__ 报错（避免静默无效参数）。
    cyl_aspect: float = 1.0

    # ---- 判据开关 ----
    input_scope: str = "any_isolated"
    readout_scope: str = "any_isolated"

    # ---- 网络 ----
    input_dim: int = 784
    output_dim: int = 10
    hidden_dim: int = 2048  # 仅 `--arch mlp` 对照基线使用（主模型不使用该字段）
    # 两端全连接包裹开关：0 = 关闭（默认，逐位不变）/ -1 = 跟随 N / > 0 = 显式宽度
    fc_dim: int = 0

    # ---- 几何权重场（本模块第 5 轮新增维度："形状 = 生长度量 + 几何权重场"）----
    # `none` = 关闭（默认，逐位不变）；`additive` = RBF 加性档（本批实现）；
    # `class_tied` / `mlp` = 枚举已接受但本批未实现（构造期显式报错）。
    geo_field: str = "none"
    # RBF 基函数个数 k（中心与宽度由 model 侧按**确定性分位点**从边级特征算出，写入 README）。
    geo_rbf_k: int = 12
    # 预留：后续 `class_tied` / `mlp` 档的隐藏宽度（本批不消费该字段，但必须保留以便
    # `to_dict()` 往返不丢字段——`train.apply_overrides` 经 `Config(**base.to_dict())` 往返）。
    # [!] **无 train.py CLI 入口**（皋陶 F9，如实注明）：只能由脚本内构造 `Config` 触发。
    geo_hidden: int = 32
    # `alpha` 的初值（可学习；`additive` 档的场增益）。默认 1.0。
    geo_alpha_init: float = 1.0
    # 可选扩展开关（**默认关闭**）：为边级几何特征追加 signed `dx/H`、`dy/H` 两列。
    # 默认关闭时特征列集合与文档表格逐字一致（5 列）。
    # [!] **无 train.py CLI 入口**（皋陶 F9，如实注明）：该字段改变特征列数 `F`（5 -> 7），
    #     并影响 `_sd` 名段与 `is_default_smoke` 判定，但本批**只能由脚本内构造 `Config`**
    #     或诊断脚本 `probe_geo_field.py --signed-delta` 触发（`train.py` 只暴露
    #     `--geo-field` / `--geo-rbf-k` / `--geo-alpha-init`，即本批计划所要求的三项）。
    #     后续若在 `train.py` 暴露，**必须同步四处**：`explicit` 判定 / `apply_overrides` /
    #     `describe()`（与 `to_dict()` 一并）/ 指纹与 `is_default_smoke` 的维度对齐守卫
    #     —— 否则会重演"静默丢弃"或"静默覆盖默认冒烟产物"两类历史缺陷。
    geo_signed_delta: bool = False

    # ---- 训练 ----
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 10
    seed: int = 42
    device: str = "auto"

    # ---- 数据 ----
    data_root: str = "data/mnist"
    num_workers: int = 0
    log_interval: int = 100

    # ---- 可选训练增强 ----
    weight_decay: float = 0.0
    readout_bias: bool = False
    lr_schedule: str = "none"
    grad_clip: float = 0.0

    # ---- 派生量（不参与构造，供便捷访问） ----
    _derived: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """构造后校验超参合法性，并预计算球半径窗口等派生量。

        异常
        ----
        ValueError
            任一超参越界时抛出，并附带具体数值上下文。
        """
        # 显式校验：把错误暴露在构造期，而不是等到张量 shape mismatch
        if self.N <= 0:
            raise ValueError(f"Config.N 必须为正整数，当前 N={self.N}")
        if self.y_in <= 0:
            raise ValueError(f"Config.y_in 必须为正整数，当前 y_in={self.y_in}")
        if self.y_out <= 0:
            raise ValueError(f"Config.y_out 必须为正整数，当前 y_out={self.y_out}")
        if not (self.H > 0.0):
            raise ValueError(f"Config.H 必须 > 0，当前 H={self.H}")
        if not (self.D > 0.0):
            raise ValueError(f"Config.D 必须 > 0，当前 D={self.D}")
        # ---- 连接半径硬约束：D <= H（G1）----
        # 几何含义：连接判据是"起点神经元的输出突触 o 与终点神经元的输入突触 j 的距离
        # <= D"，而 o / j 各自落在所属神经元的 H 半径球内（且分别在流向轴正 / 负半球）。
        # `D > H` 表示**连接半径超过接收/发送范围半径** —— 此时连接不再受"突触云尺度"
        # 约束，等于把阈值放到比神经元本身还大，属越界配置，故在构造期直接拒绝，
        # 而不是留给下游生成拓扑才发现。
        if float(self.D) > float(self.H):
            raise ValueError(
                f"Config.D 不得超过 Config.H：连接半径 D 不得超过接收/发送范围半径 H，"
                f"当前 H={self.H}, D={self.D}（D/H={float(self.D) / float(self.H):.3f}）。"
                f"请令 D <= H（本模块三个预设 DEFAULT/HIGHACC/SMALL 均取 D = H）。"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"Config.batch_size 必须为正整数，当前 batch_size={self.batch_size}"
            )
        if self.lr <= 0.0:
            raise ValueError(f"Config.lr 必须 > 0，当前 lr={self.lr}")
        if self.hidden_dim <= 0:
            raise ValueError(
                f"Config.hidden_dim 必须为正整数，当前 hidden_dim={self.hidden_dim}"
            )
        # ---- fc_dim 校验（本模块第三轮新增）----
        # [!] 只允许 -1（跟随 N）或 >= 0（0 = 关闭）：`< -1` 是**无意义哨兵**，
        #    必须在构造期拒绝，而不是等到下游按"宽度 = -2"去建张量。
        if int(self.fc_dim) < -1:
            raise ValueError(
                f"Config.fc_dim 只允许 -1（跟随 N）或 >= 0（0 表示关闭），"
                f"当前 fc_dim={self.fc_dim}"
            )
        # ---- geo_field 校验（本模块第 5 轮新增：几何权重场）----
        # [!] 两条纪律：
        #   (a) 非法取值在**构造期**报错（不留给下游按陌生字符串去查表）；
        #   (b) `class_tied` / `mlp` 在 `GEO_FIELD_CHOICES` 里**但本批未实现** ——
        #       必须**显式报错"未实现"**，绝不允许静默降级成 `none` 后照常训练
        #       （那会产出"看起来开了几何场、实际什么都没改"的取证产物）。
        if self.geo_field not in GEO_FIELD_CHOICES:
            raise ValueError(
                f"Config.geo_field 仅允许 {GEO_FIELD_CHOICES}，当前 geo_field={self.geo_field!r}"
            )
        if self.geo_field not in GEO_IMPLEMENTED_CHOICES:
            raise ValueError(
                f"Config.geo_field={self.geo_field!r} 本批**未实现**"
                f"（已实现档：{GEO_IMPLEMENTED_CHOICES}）。"
                f"`class_tied` / `mlp` 档待批次 1 验证通过后按单独计划实现；"
                f"当前不接受该取值（拒绝把未实现档静默降级成 'none'）。"
            )
        if int(self.geo_rbf_k) < 1:
            raise ValueError(
                f"Config.geo_rbf_k 必须 >= 1（RBF 基函数个数），当前 geo_rbf_k={self.geo_rbf_k}"
            )
        if int(self.geo_hidden) < 1:
            raise ValueError(
                f"Config.geo_hidden 必须 >= 1（预留的隐藏宽度），当前 geo_hidden={self.geo_hidden}"
            )
        if not (float(self.geo_alpha_init) >= 0.0):
            raise ValueError(
                f"Config.geo_alpha_init 必须 >= 0（几何场增益初值），"
                f"当前 geo_alpha_init={self.geo_alpha_init}"
            )

        # ---- 几何字段校验 ----
        if self.flow_axis not in FLOW_AXIS_CHOICES:
            raise ValueError(
                f"Config.flow_axis 仅允许 {FLOW_AXIS_CHOICES}，当前 flow_axis={self.flow_axis!r}"
            )
        # ---- 形状字段校验（本模块新增维度）----
        if self.shape not in SHAPE_CHOICES:
            raise ValueError(
                f"Config.shape 仅允许 {SHAPE_CHOICES}，当前 shape={self.shape!r}"
            )
        # [!] 非 cylinder 形状**显式**传入非默认 cyl_aspect 必须报错：否则该参数被静默忽略，
        #    用户会以为"λ 生效了"而实际上形状仍是球/立方体 —— 典型"静默无效参数"缺陷。
        #    判定口径：dataclass 已把缺省值填成 1.0，故"非默认值"即"显式传入"。
        if self.shape != "cylinder" and float(self.cyl_aspect) != 1.0:
            raise ValueError(
                f"Config.cyl_aspect 仅在 shape='cylinder' 时生效；"
                f"当前 shape={self.shape!r} 但 cyl_aspect={self.cyl_aspect}"
                f"（非默认值 1.0）。请改用 shape='cylinder' 或移除 cyl_aspect，"
                f"避免静默无效参数。"
            )
        if not (float(self.cyl_aspect) > 0.0):
            raise ValueError(
                f"Config.cyl_aspect 必须 > 0（λ = c/r），当前 cyl_aspect={self.cyl_aspect}"
            )
        if self.input_scope not in SCOPE_CHOICES:
            raise ValueError(
                f"Config.input_scope 仅允许 {SCOPE_CHOICES}，当前 input_scope={self.input_scope!r}"
            )
        if self.placement not in PLACEMENT_CHOICES:
            raise ValueError(
                f"Config.placement 仅允许 {PLACEMENT_CHOICES}，当前 placement={self.placement!r}"
            )
        if self.readout_scope not in SCOPE_CHOICES:
            raise ValueError(
                f"Config.readout_scope 仅允许 {SCOPE_CHOICES}，当前 readout_scope={self.readout_scope!r}"
            )
        if self.space_radius < 0.0:
            raise ValueError(
                f"Config.space_radius 必须 >= 0（0 表示取公式下界 R_min），"
                f"当前 space_radius={self.space_radius}"
            )

        # ---- 形状专属尺寸窗口（由公式唯一确定）----
        spec = shape_spec(self.shape, self.cyl_aspect)
        r_min = min_space_radius(self.shape, self.N, self.H, self.cyl_aspect)
        r_max = max_space_radius(self.shape, self.N, self.H, self.D, self.cyl_aspect)
        # 窗口比较容差：`space_radius` 常由使用者按 `R_min` / `R_max` 的**显示值**（6 位小数）
        # 或其他路径的 float64 计算值回填，与公式值存在约 1e-7 量级的浮点差。若用严格比较，
        # "恰好取 R_min 的合法值"会被误判越界（实测：cube/N=256 的 R_min 全精度为
        # 0.5656804567776409，按 6 位小数回填 0.565680 会差 4.57e-7 而被判越界，
        # 而该值正是文档与 shape_tag 给出的默认下界）。故取 1e-6 相对容差
        # （远小于"真正越界"的量级：合法窗口宽度在此例为 0.838，越界测试值 0.01 仍被拒）。
        tol = 1e-6 * max(1.0, abs(r_min), abs(r_max))
        # 窗口必须非空：极端 λ（细高圆柱）下 R_min 会超过 R_max —— 这是"该形状装不下 N 个
        # 神经元"的几何事实，必须在构造期拦下并附实测值，而不是留给下游静默生成退化图。
        if not (r_max >= r_min - tol):
            raise ValueError(
                f"形状专属尺寸窗口为空：R_max={r_max:.6f} < R_min={r_min:.6f}"
                f"（shape={self.shape!r}, cyl_aspect={self.cyl_aspect}, N={self.N}, "
                f"H={self.H}, D={self.D}, φ={PACKING_PHI}）。"
                f"该形状在当前规模/长径比下无法非重叠容纳 N 个神经元且保证 D 邻域落在空间内；"
                f"请减小 N、减小 cyl_aspect（更扁而非更细高）或调整 H/D。"
            )
        if self.space_radius > 0.0:
            # 显式指定时必须落在 [R_min, R_max] 内，越界立即报错（附全部上下文）
            if not (r_min - tol <= self.space_radius <= r_max + tol):
                raise ValueError(
                    f"Config.space_radius={self.space_radius} 越界：必须落在 "
                    f"[R_min, R_max] = [{r_min:.6f}, {r_max:.6f}] 内"
                    f"（shape={self.shape!r}, cyl_aspect={self.cyl_aspect}, N={self.N}, "
                    f"H={self.H}, D={self.D}, φ={PACKING_PHI}）。"
                    f"space_radius=0 表示取该形状的默认下界 R_min={r_min:.6f}。"
                )
        # [!] 默认分支按形状取 R_min：二期球体口径下 R_min = H·(N/φ)^(1/3)，与 `n3d_sphere`
        #    逐位一致（回归锚点）；cube / cylinder 取各自体积论证得到的下界。
        effective = float(self.space_radius) if self.space_radius > 0.0 else r_min

        # ---- 形状外接尺度（供 model 的搜索范围与放置度量断言使用）----
        circum_radius = effective * spec.circum_coef

        # ---- 可选训练增强字段校验 ----
        if self.lr_schedule not in ("none", "cosine"):
            raise ValueError(
                f"Config.lr_schedule 仅允许 'none' 或 'cosine'，当前 lr_schedule={self.lr_schedule!r}"
            )
        if self.weight_decay < 0.0:
            raise ValueError(
                f"Config.weight_decay 必须 >= 0，当前 weight_decay={self.weight_decay}"
            )
        if self.grad_clip < 0.0:
            raise ValueError(f"Config.grad_clip 必须 >= 0，当前 grad_clip={self.grad_clip}")

        # ---- 派生规模量与几何量 ----
        self._derived = {
            "n_input_syn": self.N * self.y_in,
            "n_output_syn": self.N * self.y_out,
            "fcc_lattice_constant": 2.0 * math.sqrt(2.0) * float(self.H),
            "min_space_radius": r_min,
            "max_space_radius": r_max,
            "effective_space_radius": effective,
            "circum_coef": float(spec.circum_coef),
            "shape_circum_radius": float(circum_radius),
            # fc_dim 的**有效宽度**：0 -> 0（关闭）；-1 -> N；> 0 -> 该值。
            "fc_width": self._resolve_fc_width(),
        }

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------
    @property
    def fc_enabled(self) -> bool:
        """两端全连接包裹是否启用（`fc_dim != 0`）。"""
        return int(self.fc_dim) != 0

    @property
    def geo_enabled(self) -> bool:
        """几何权重场是否启用（`geo_field != "none"`）。

        关闭（`none`）时**连几何特征都不构造**：`edge_geo_feat` 等 buffer 不注册、
        几何参数不创建，代码路径与改动前逐位一致（硬约束第 1 条）。
        """
        return self.geo_field != "none"

    @property
    def fc_width(self) -> int:
        """两端全连接的**有效宽度** `H`：`0`=关闭 -> 0；`-1`=跟随 N -> N；`>0` -> 该值。"""
        return int(self._derived["fc_width"])

    def _resolve_fc_width(self) -> int:
        """把 `fc_dim` 的三种取值语义解析成有效宽度（构造期唯一解析点）。"""
        fd = int(self.fc_dim)
        if fd == 0:
            return 0
        if fd == -1:
            return int(self.N)
        return fd

    @property
    def n_input_syn(self) -> int:
        """输入突触总数 N * y_in。"""
        return self._derived["n_input_syn"]

    @property
    def n_output_syn(self) -> int:
        """输出突触总数 N * y_out。"""
        return self._derived["n_output_syn"]

    @property
    def fcc_lattice_constant(self) -> float:
        """FCC 晶格常数 a = 2√2·H（最近邻距恰为 2H）。"""
        return float(self._derived["fcc_lattice_constant"])

    @property
    def min_space_radius(self) -> float:
        """该形状的特征尺度下界 R_min（球：`H·(N/φ)^(1/3)`，与二期同式）。"""
        return float(self._derived["min_space_radius"])

    @property
    def max_space_radius(self) -> float:
        """该形状的特征尺度上界 R_max（球：`(H+D)·(N/φ)^(1/3)`，与二期同式）。"""
        return float(self._derived["max_space_radius"])

    @property
    def effective_space_radius(self) -> float:
        """实际使用的形状特征尺度（`space_radius=0` 时为该形状的 R_min）。"""
        return float(self._derived["effective_space_radius"])

    @property
    def circum_coef(self) -> float:
        """外接半径与特征尺度之比 `ρ/R`：sphere=1、cube=√3、cylinder=√(1+λ^2)。"""
        return float(self._derived["circum_coef"])

    @property
    def shape_circum_radius(self) -> float:
        """形状外接半径 `ρ = circum_coef · effective_space_radius`。

        用途：`model._build_fcc_positions` 的候选搜索范围按该值放大，保证"度量为 m 的
        区域"完整落在搜索盒内（球体口径下 `ρ = R = space_radius`，与二期逐位一致）。
        """
        return float(self._derived["shape_circum_radius"])

    @property
    def shape_spec(self) -> ShapeSpec:
        """归一化后的形状规格（`ShapeSpec`）。"""
        return shape_spec(self.shape, self.cyl_aspect)

    @property
    def flow_axis_index(self) -> int:
        """全局流向轴对应的坐标分量下标（x->0, y->1, z->2）。"""
        return FLOW_AXIS_CHOICES.index(self.flow_axis)

    def to_dict(self) -> dict:
        """返回超参字典（不含内部派生字段），便于日志打印与 checkpoint 存档。

        必须包含全部几何与判据字段（`flow_axis` / `space_radius` / `input_scope` /
        `readout_scope` **以及本模块新增的 `shape` / `cyl_aspect`**）：
        `train.apply_overrides` 经由 ``Config(**base.to_dict())`` 往返构造配置，
        若字段缺席会被静默丢弃（`--shape` / `--cyl-aspect` 就会失效）。
        """
        return {
            "N": self.N,
            "y_in": self.y_in,
            "y_out": self.y_out,
            "H": self.H,
            "D": self.D,
            "flow_axis": self.flow_axis,
            "space_radius": self.space_radius,
            "placement": self.placement,
            "shape": self.shape,
            "cyl_aspect": self.cyl_aspect,
            "input_scope": self.input_scope,
            "readout_scope": self.readout_scope,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "fc_dim": self.fc_dim,
            # geo_field 族（第 5 轮新增）：**必须全部进 to_dict()**，否则
            # `train.apply_overrides` 经 `Config(**base.to_dict())` 往返会静默丢字段，
            # `--geo-field additive` 就会失效（与 `--shape` / `--fc-dim` 的历史陷阱同源）。
            "geo_field": self.geo_field,
            "geo_rbf_k": self.geo_rbf_k,
            "geo_hidden": self.geo_hidden,
            "geo_alpha_init": self.geo_alpha_init,
            "geo_signed_delta": self.geo_signed_delta,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "epochs": self.epochs,
            "seed": self.seed,
            "device": self.device,
            "data_root": self.data_root,
            "num_workers": self.num_workers,
            "log_interval": self.log_interval,
            "weight_decay": self.weight_decay,
            "readout_bias": self.readout_bias,
            "lr_schedule": self.lr_schedule,
            "grad_clip": self.grad_clip,
        }

    def shape_tag(self) -> str:
        """返回产物指纹用的形状段：`shape{cube}` / `shape{cylinder}a{0.5}`。

        [!] 防撞名硬要求：指纹**必须**含形状维度（圆柱还须含长径比）。否则 N/H/D/seed/
        scope 全相同的三种形状会**互相覆盖**同一份取证产物（历史纠正记录中
        `verify_<bpe>.pt` 同名互覆导致取证失效的同一类缺陷）。
        """
        if self.shape == "cylinder":
            return f"shape{self.shape}_a{self.cyl_aspect:g}"
        return f"shape{self.shape}"

    def describe(self) -> str:
        """返回人类可读的超参摘要字符串（单行多段，便于终端输出）。"""
        parts = [
            f"N={self.N}",
            f"y_in={self.y_in}",
            f"y_out={self.y_out}",
            f"H={self.H}",
            f"D={self.D}",
            f"flow_axis={self.flow_axis}",
            self.shape_spec.describe(),
            f"space_radius={self.space_radius}->{self.effective_space_radius:.6f}",
            f"placement={self.placement}",
            f"input_scope={self.input_scope}",
            f"readout_scope={self.readout_scope}",
            f"input_dim={self.input_dim}",
            f"output_dim={self.output_dim}",
            f"batch_size={self.batch_size}",
            f"lr={self.lr}",
            f"epochs={self.epochs}",
            f"seed={self.seed}",
            f"device={self.device}",
            f"weight_decay={self.weight_decay}",
            f"fc_dim={self.fc_dim}",
            f"geo_field={self.geo_field}",
            f"geo_rbf_k={self.geo_rbf_k}",
            f"geo_alpha_init={self.geo_alpha_init}",
            f"readout_bias={self.readout_bias}",
            f"lr_schedule={self.lr_schedule}",
            f"grad_clip={self.grad_clip}",
        ]
        return "Config(" + ", ".join(parts) + ")"


# ----------------------------------------------------------------------
# 预设（三者均取 `shape="sphere"`：与二期口径一致，作为回归锚点）
# ----------------------------------------------------------------------
# SMALL_CONFIG：阶段 A 冒烟测试专用，规模小、CPU 友好
#   N=64, y_in=4, y_out=4, batch_size=32
#   D = H = 0.15（硬约束 D <= H，见 Config.__post_init__）
#   sphere 口径 R_min = 0.663198、R_max = 1.326395
SMALL_CONFIG = Config(
    N=64,
    y_in=4,
    y_out=4,
    H=0.15,
    D=0.15,
    input_dim=784,
    output_dim=10,
    batch_size=32,
    lr=1e-3,
    epochs=1,
    seed=42,
    device="auto",
    shape="sphere",
)

# DEFAULT_CONFIG：正式训练默认配置
#   N=256, y_in=8, y_out=8, batch_size=64
#   D = H = 0.10（硬约束 D <= H）
#   shape="sphere"（默认，回归锚点）；space_radius 取默认 0.0
#   -> sphere R_min = 0.701840（N=256, H=0.10, φ=0.7405）
DEFAULT_CONFIG = Config(
    N=256,
    y_in=8,
    y_out=8,
    H=0.1,
    D=0.1,
    input_dim=784,
    output_dim=10,
    batch_size=64,
    lr=1e-3,
    epochs=10,
    seed=42,
    device="auto",
    shape="sphere",
)

# HIGHACC_CONFIG：高精度冲刺配置
#   相比 DEFAULT_CONFIG 的变化：batch 64->128、lr 1e-3->2e-3、epochs 10->20，
#   并启用 AdamW(weight_decay=1e-4)、输出层 bias、cosine 学习率调度、梯度裁剪 1.0。
#   D = H = 0.10（硬约束 D <= H）；shape 同取 "sphere"。
HIGHACC_CONFIG = Config(
    N=256,
    y_in=8,
    y_out=8,
    H=0.1,
    D=0.1,
    input_dim=784,
    output_dim=10,
    batch_size=128,
    lr=2e-3,
    epochs=20,
    seed=42,
    device="auto",
    weight_decay=1e-4,
    readout_bias=True,
    lr_schedule="cosine",
    grad_clip=1.0,
    shape="sphere",
)
