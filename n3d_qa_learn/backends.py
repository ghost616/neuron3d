"""连接契约代理层（BackendAdapter 注册表）。

职责边界（纪律）
----------------
本层只承担**「数据集侧连接参数 -> 后端模型构造参数」**这一件事：

* 数据侧 **输入**：特征维 ``D``（由 :mod:`n3d_qa_learn.features` 决定）；
* 后端侧 **输出**：一个已构造好的 N3D 模型 + 一个 ``[B, D]`` 的特征提取函数。

本层**不承载训练编排**（没有 loss、没有优化器、没有 epoch），也**不干预后端模型的
内部超参**：除 ``input_dim`` / ``output_dim`` 这两个"连接参数"外，其余结构参数一律由
``recommended_config`` 给出（那是"该后端在 QA 任务上的推荐构型"，属**后端自身的**选择，
不是代理层替它调参）。

上游只读
--------
三个后端模块（``n3d_shape`` / ``n3d_sphere`` / ``n3d_proto``）一律**只读 import** 其
``model`` / ``config``；本模块不修改上游任何源码与产物。

三后端与 ``D`` 的连接口径（现场实测口径，非推测）
-----------------------------------------------
====================  ==========================  ==================================
backend               N3D 侧 ``output_dim``        ``[B, D]`` 特征来源
====================  ==========================  ==================================
``n3d_shape``         ``D``                       ``forward`` 直接输出 ``[B, D]``
``n3d_sphere``        ``D``                       ``forward`` 直接输出 ``[B, D]``
``n3d_proto``         ``D``                       ``forward`` 输出 ``[B, D]``（其一期
                                                  读出结构即 ``s_out @ W_out``）
====================  ==========================  ==================================

三者都天然产出 ``[B, D]``，因此代理层**不需要**再插入任何投影层 —— 代理层只做
"把 ``D`` 接到 ``input_dim`` / ``output_dim`` 上"。
若某个后端的原生特征维与 ``D`` 不匹配，``assert_feature_dim`` 会在**首次前向时**
显式报错，而不是静默返回错维度（拒绝静默错配）。

结构开关的透传（本轮新增，仍属"连接参数"职责）
--------------------------------------------
:class:`BackendStructure` 把**后端自身的结构开关**（``shape`` / ``cyl_aspect`` /
``fc_dim`` / ``N`` / ``y_in`` / ``y_out`` / ``geo_field``）从实验编排侧搬到
``recommended_config`` 的构造入参上。代理层对它们的纪律：

* **不设默认值以外的"调参"逻辑**：这 7 个字段原样透传给上游 ``Config``，代理层
  不换算、不裁剪、不猜默认；
* **不吞异常、不静默降级**：非法组合一律由上游 ``Config.__post_init__`` 在**构造期**
  抛出（现场实测：``y_in=y_out=2`` 触发连通性下限校验、``cyl_aspect=0.15/4.0`` 触发
  尺寸窗口校验、``geo_field="class_tied"`` 触发"未实现档"拒绝）；代理层**不做**
  二次校验、也不把它降级成默认档；
* **能力边界显式**：只有 :data:`STRUCTURE_AWARE_BACKENDS`（现场枚举 = ``n3d_shape``）
  的 ``Config`` 具备这些字段。对不具备该能力的后端，``structure`` 非默认时
  **显式报错**（``ValueError``），而不是静默丢弃结构覆盖参数；
* ``structure=None``（或全默认）时，三个后端的 ``Config`` 构造实参与改动前**逐字符
  相同**，``describe()`` 与参数量因此**逐位一致**。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .features import VectorizerConfig

# ---------------------------------------------------------------------------
# 后端标识
# ---------------------------------------------------------------------------

#: 三个已登记的后端名（注册表的**唯一**合法键集合）。
BACKEND_NAMES: Tuple[str, ...] = ("n3d_shape", "n3d_sphere", "n3d_proto")

#: 具备「结构开关」字段（``shape`` / ``cyl_aspect`` / ``fc_dim`` / ``geo_field``）的后端。
#: **现场枚举口径**：``n3d_shape.config.Config`` 的字段集合里同时存在这 4 个字段；
#: ``n3d_sphere.config.Config`` 与 ``n3d_proto.config.Config`` 的字段集合里**都没有**
#: 它们（现场读 ``dataclasses.fields`` 确认）。对不在本名单里的后端传非默认 ``structure``
#: 一律显式报错，禁止静默丢弃。
STRUCTURE_AWARE_BACKENDS: Tuple[str, ...] = ("n3d_shape",)

#: 形状取值（与上游 ``n3d_shape.config`` 的枚举一致；代理层不做二次校验）。
SHAPE_CHOICES: Tuple[str, ...] = ("sphere", "cube", "cylinder")

#: 几何权重场取值（``none`` 默认关闭 / ``additive`` 唯一已实现档；
#: ``class_tied`` / ``mlp`` 枚举已接受但上游构造期显式拒绝）。
GEO_FIELD_CHOICES: Tuple[str, ...] = ("none", "additive", "class_tied", "mlp")

#: 结构开关的**默认值**（= 改动前的硬编码取值；见 ``recommended_config``）。
STRUCTURE_DEFAULTS: Dict[str, Any] = {
    "shape": "sphere",
    "cyl_aspect": 1.0,
    "fc_dim": 0,
    "N": 64,
    "y_in": 4,
    "y_out": 4,
    "geo_field": "none",
}


@dataclass(frozen=True)
class BackendStructure:
    """后端结构开关（**连接参数**的一部分：把数据侧/实验侧的结构选择接到上游 Config）。

    属性
    ----
    shape : str
        空间形状（``sphere`` / ``cube`` / ``cylinder``），见 :data:`SHAPE_CHOICES`。
    cyl_aspect : float
        圆柱长径比 ``λ = c / r``；**仅** ``shape == "cylinder"`` 时允许非 1.0。
    fc_dim : int
        两端全连接包裹：``0`` 关闭 / ``-1`` 跟随 ``N`` / ``> 0`` 显式宽度。
    N : int
        神经元规模。
    y_in / y_out : int
        输入 / 输出突触数（本框架现场固定为 4：``y = 2`` 会被上游连通性下限校验拒绝）。
    geo_field : str
        几何权重场（``none`` / ``additive``），见 :data:`GEO_FIELD_CHOICES`。

    关键不变量
    ----------
    * 字段集合恒等于 :data:`STRUCTURE_DEFAULTS` 的键集合（现场断言，防止"加了字段忘了
      透传"这类静默失配）；
    * 本类是**不可变**的（``frozen=True``），因为它是产物 meta 的一部分。
    """

    shape: str = "sphere"
    cyl_aspect: float = 1.0
    fc_dim: int = 0
    N: int = 64
    y_in: int = 4
    y_out: int = 4
    geo_field: str = "none"

    def __post_init__(self) -> None:
        if self.shape not in SHAPE_CHOICES:
            raise ValueError(
                f"BackendStructure.shape 仅允许 {list(SHAPE_CHOICES)}，"
                f"当前 {self.shape!r}"
            )
        if self.geo_field not in GEO_FIELD_CHOICES:
            raise ValueError(
                f"BackendStructure.geo_field 仅允许 {list(GEO_FIELD_CHOICES)}，"
                f"当前 {self.geo_field!r}"
            )
        if int(self.N) < 1:
            raise ValueError(f"BackendStructure.N 必须 >= 1，当前 {self.N}")
        if int(self.y_in) < 1 or int(self.y_out) < 1:
            raise ValueError(
                f"BackendStructure.y_in / y_out 必须 >= 1，当前 "
                f"{self.y_in} / {self.y_out}"
            )
        if not (float(self.cyl_aspect) > 0.0):
            raise ValueError(
                f"BackendStructure.cyl_aspect 必须 > 0，当前 {self.cyl_aspect}"
            )

    def is_default(self) -> bool:
        """是否逐字段等于 :data:`STRUCTURE_DEFAULTS`（默认档 = 改动前的行为）。"""
        return all(
            getattr(self, key) == value for key, value in STRUCTURE_DEFAULTS.items()
        )

    def to_kwargs(self) -> Dict[str, Any]:
        """转成上游 ``Config`` 的关键字实参（字段与 :data:`STRUCTURE_DEFAULTS` 同集合）。"""
        return {key: getattr(self, key) for key in STRUCTURE_DEFAULTS}

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化（写进报告 / 产物 meta）。"""
        return {
            "shape": str(self.shape),
            "cyl_aspect": float(self.cyl_aspect),
            "fc_dim": int(self.fc_dim),
            "N": int(self.N),
            "y_in": int(self.y_in),
            "y_out": int(self.y_out),
            "geo_field": str(self.geo_field),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BackendStructure":
        """从（可能缺字段的）字典重建；缺字段取默认值（向后兼容旧产物）。"""
        payload = dict(data or {})
        return cls(
            shape=str(payload.get("shape", STRUCTURE_DEFAULTS["shape"])),
            cyl_aspect=float(payload.get("cyl_aspect", STRUCTURE_DEFAULTS["cyl_aspect"])),
            fc_dim=int(payload.get("fc_dim", STRUCTURE_DEFAULTS["fc_dim"])),
            N=int(payload.get("N", STRUCTURE_DEFAULTS["N"])),
            y_in=int(payload.get("y_in", STRUCTURE_DEFAULTS["y_in"])),
            y_out=int(payload.get("y_out", STRUCTURE_DEFAULTS["y_out"])),
            geo_field=str(payload.get("geo_field", STRUCTURE_DEFAULTS["geo_field"])),
        )


#: 结构字段集合的**现场自检**：``BackendStructure`` 的字段必须与默认值表逐键对齐。
assert set(BackendStructure.__dataclass_fields__) == set(STRUCTURE_DEFAULTS), (
    "BackendStructure 字段集合与 STRUCTURE_DEFAULTS 不一致（新增结构开关时必须同步）"
)


@dataclass(frozen=True)
class BackendSpec:
    """一个 N3D 后端的静态描述（不含任何模型实例）。

    属性
    ----
    name : str
        后端名（注册表键）。
    evolution : str
        该后端在 N3D 演进链上的位置（一期原型 / 二期球形 / 三期形状变体）。
    feature_source : str
        ``[B, D]`` 特征的来源说明（用于报告与文档自解释）。
    """

    name: str
    evolution: str
    feature_source: str


# ---------------------------------------------------------------------------
# 推荐构型（后端自身的选择；代理层只搬运，不改其内部超参）
# ---------------------------------------------------------------------------


def recommended_config(
    name: str, input_dim: int, structure: Optional[BackendStructure] = None
) -> Any:
    """返回某后端在 QA 任务上的**推荐构型**（只设置连接参数 + 该后端原生结构参数）。

    参数
    ----
    name : str
        后端名（``BACKEND_NAMES`` 之一）。
    input_dim : int
        连接参数：特征维 ``D``（同时作为该后端的 ``input_dim`` 与 ``output_dim``）。
    structure : Optional[BackendStructure]
        **结构开关覆盖**（本轮新增）。为 ``None`` 时取全默认
        （``shape="sphere"`` / ``cyl_aspect=1.0`` / ``fc_dim=0`` / ``N=64`` /
        ``y_in=y_out=4`` / ``geo_field="none"``），此时三个后端构造出的 ``Config``
        与改动前**逐位一致**。

    返回
    ----
    Any
        该后端自己的 ``Config`` 实例。

    说明
    ----
    * 规模取该后端既有 **SMALL** 口径（``N=64`` / ``y_in=y_out=4`` / ``H=D=0.15``），
      与 QA 数据集的样本量（1e3 量级）匹配；这是"后端自身推荐构型"，不是代理层调参；
    * ``n3d_shape`` 的 ``dataset`` 必须显式给 ``"npz"``：该模块的数据集层对 ``mnist``
      声明了固定维度 ``784/10``，只有**占位维**来源（``npz`` / ``csv`` / ``json``）
      才允许自定义 ``input_dim`` / ``output_dim``（现场实测：用 ``mnist`` 会抛
      ``ValueError: input_dim 与数据集规格不一致``）。此处只借用其"占位维"语义，
      不加载任何数据。
    * **非法组合交由上游构造期报错**（代理层不吞异常、不静默降级）。现场实测的
      上游拒绝：``y_in=y_out=2`` -> 连通性下限校验失败；``shape="cylinder"`` 且
      ``cyl_aspect ∈ {0.15, 4.0}`` -> 尺寸窗口 / FCC 点数不足；``geo_field="class_tied"``
      -> "本批未实现"；``shape != "cylinder"`` 且 ``cyl_aspect != 1.0`` -> 静默无效参数拒绝。
    * 对**不具备结构开关能力**的后端（:data:`STRUCTURE_AWARE_BACKENDS` 之外）传
      非默认 ``structure`` 时**显式报错**，禁止静默丢弃。
    """
    dim = int(input_dim)
    if dim < 1:
        raise ValueError(f"input_dim 必须 >= 1，当前 {dim}")
    spec = BackendStructure() if structure is None else structure
    if not isinstance(spec, BackendStructure):
        raise TypeError(
            "recommended_config 的 structure 必须是 BackendStructure 或 None，"
            f"当前 {type(spec).__name__}"
        )
    override = spec.to_kwargs()
    if name not in STRUCTURE_AWARE_BACKENDS and not spec.is_default():
        raise ValueError(
            f"后端 {name!r} 不支持结构开关覆盖（能力名单 = "
            f"{list(STRUCTURE_AWARE_BACKENDS)}）；收到的非默认结构 = "
            f"{spec.as_dict()}。拒绝静默丢弃结构参数；请改用支持结构开关的后端，"
            "或把结构恢复为默认档。"
        )
    if name == "n3d_shape":
        from n3d_shape.config import Config as _ShapeConfig

        return _ShapeConfig(
            N=int(override["N"]), y_in=int(override["y_in"]), y_out=int(override["y_out"]),
            H=0.15, D=0.15,
            input_dim=dim, output_dim=dim,
            input_scope="any_isolated", readout_scope="any_isolated",
            shape=str(override["shape"]), cyl_aspect=float(override["cyl_aspect"]),
            fc_dim=int(override["fc_dim"]), geo_field=str(override["geo_field"]),
            dataset="npz", seed=42,
        )
    if name == "n3d_sphere":
        from n3d_sphere.config import Config as _SphereConfig

        return _SphereConfig(
            N=64, y_in=4, y_out=4, H=0.15, D=0.15,
            input_dim=dim, output_dim=dim,
            input_scope="any_isolated", readout_scope="any_isolated", seed=42,
        )
    if name == "n3d_proto":
        from n3d_proto.config import Config as _ProtoConfig

        return _ProtoConfig(
            N=64, y_in=4, y_out=4, H=0.15, D=0.15, L=1.0, T=3,
            input_dim=dim, output_dim=dim, seed=42,
        )
    raise KeyError(f"未登记的后端名 {name!r}；合法集合 = {list(BACKEND_NAMES)}")


# ---------------------------------------------------------------------------
# 后端适配器
# ---------------------------------------------------------------------------


class BackendAdapter:
    """把一个 N3D 后端封成「``[B, D]`` 特征提取器 + 拓扑统计」的连接契约适配器。

    参数
    ----
    spec : BackendSpec
        后端静态描述。
    model : nn.Module
        已构造好的后端模型实例。
    input_dim : int
        连接参数 ``D``。

    关键不变量
    ----------
    * ``feature_dim(model, x)`` 恒等于 ``input_dim``（首次前向时强校验）；
    * 适配器**不持有**任何可学习参数（``self.model`` 的参数属于后端，不属代理层）。
    """

    def __init__(
        self,
        spec: BackendSpec,
        model: nn.Module,
        input_dim: int,
        structure: Optional[BackendStructure] = None,
    ) -> None:
        self.spec = spec
        self.model = model
        self.input_dim = int(input_dim)
        #: 构造该后端时使用的**结构开关**（``None`` 视为全默认档）。
        self.structure: BackendStructure = (
            BackendStructure() if structure is None else structure
        )

    # -- 连接参数 ---------------------------------------------------------
    @property
    def name(self) -> str:
        """后端名。"""
        return self.spec.name

    def build_input(self, features: torch.Tensor) -> torch.Tensor:
        """校验数据侧特征与连接参数一致，返回后端可直接消费的张量。"""
        if features.dim() != 2:
            raise ValueError(
                f"[{self.name}] 特征必须为 2D [B, D]，当前 shape={tuple(features.shape)}"
            )
        if int(features.shape[1]) != self.input_dim:
            raise ValueError(
                f"[{self.name}] 特征维度与连接参数不一致：D={self.input_dim}，"
                f"实际 {features.shape[1]}（说明数据侧向量化口径与代理层构造参数错配）"
            )
        return features

    def features(self, features: torch.Tensor) -> torch.Tensor:
        """前向并返回 ``[B, D]`` 特征（代理层唯一的前向入口）。"""
        x = self.build_input(features)
        out = self.model(x)
        if not torch.is_tensor(out) or out.dim() != 2:
            raise RuntimeError(
                f"[{self.name}] 后端前向返回非 2D 张量：{type(out).__name__}"
            )
        if int(out.shape[1]) != self.input_dim:
            raise RuntimeError(
                f"[{self.name}] 后端原生特征维 {out.shape[1]} 与连接参数 D="
                f"{self.input_dim} 不匹配；代理层不插入任何投影层，拒绝静默错配"
            )
        return out

    # -- 统计（供 P0 探针登记） -------------------------------------------
    def count_parameters(self) -> int:
        """可学习参数总数（取后端自身口径；无该方法时回退为 ``sum(p.numel())``）。"""
        counter: Optional[Callable[[], int]] = getattr(
            self.model, "count_parameters", None
        )
        if callable(counter):
            return int(counter())
        return int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def topology_stats(self) -> Dict[str, float]:
        """拓扑统计：``E`` / ``K`` / ``|S_in|`` / ``|S_out|``（后端口径各不相同）。

        返回
        ----
        Dict[str, float]
            键集合为 ``{"E", "K", "S_in", "S_out"}`` 的子集：
            * ``E``：边数（二期 / 三期用 ``num_edges``；一期用 ``num_edges``）；
            * ``K``：递推层数。二期 / 三期**没有** ``num_layers`` 属性，故由
              ``neuron_pos[:, flow_axis_index]`` 的去重计数现场算出（与
              ``get_connection_stats()["num_layers"]`` 同口径）；一期无分层概念，
              退化为其迭代轮数 ``T``；
            * ``S_in`` / ``S_out``：输入/输出作用域规模（一期无该概念，用 ``y_in`` /
              ``y_out`` 的突触侧规模替代，并在报告中显式标注口径）。
        """
        m = self.model
        stats: Dict[str, float] = {}
        if hasattr(m, "num_edges"):
            stats["E"] = float(getattr(m, "num_edges"))
        if hasattr(m, "T") and not hasattr(m, "flow_axis_index"):
            stats["K"] = float(getattr(m, "T"))
        elif hasattr(m, "neuron_pos") and hasattr(m, "flow_axis_index"):
            axis_coord = m.neuron_pos[:, int(m.flow_axis_index)]
            stats["K"] = float(torch.unique(axis_coord).numel())
        if hasattr(m, "num_in_scope"):
            stats["S_in"] = float(getattr(m, "num_in_scope"))
        elif hasattr(m, "n_in_syn"):
            stats["S_in"] = float(getattr(m, "n_in_syn"))
        if hasattr(m, "num_out_scope"):
            stats["S_out"] = float(getattr(m, "num_out_scope"))
        elif hasattr(m, "n_out_syn"):
            stats["S_out"] = float(getattr(m, "n_out_syn"))
        return stats

    def describe(self) -> Dict[str, Any]:
        """适配器自描述（写进产物 meta，供事后复核连接参数）。

        **键集合与改动前逐位一致**（新增的结构开关信息走
        :meth:`structure_manifest`，不混进这里，避免改动默认档的 describe 内容）。
        """
        return {
            "backend": self.name,
            "evolution": self.spec.evolution,
            "feature_source": self.spec.feature_source,
            "input_dim": int(self.input_dim),
            "output_dim": int(self.input_dim),
            "topology_stats": {k: float(v) for k, v in self.topology_stats().items()},
            "parameters": int(self.count_parameters()),
        }

    def structure_manifest(self) -> Dict[str, Any]:
        """构造该适配器所用的结构开关（现场枚举，供报告登记）。

        返回
        ----
        Dict[str, Any]
            ``{"structure": {...7 字段...}, "is_default": bool,
            "structure_aware": bool}``。
        """
        return {
            "structure": self.structure.as_dict(),
            "is_default": bool(self.structure.is_default()),
            "structure_aware": bool(self.name in STRUCTURE_AWARE_BACKENDS),
        }


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

#: 后端静态描述表（唯一注册点）。
BACKEND_SPECS: Dict[str, BackendSpec] = {
    "n3d_shape": BackendSpec(
        name="n3d_shape",
        evolution="三期：形状变体（球体 / 立方体 / 圆柱体）",
        feature_source="ThreeDNeuronSpace.forward -> [B, output_dim=D]",
    ),
    "n3d_sphere": BackendSpec(
        name="n3d_sphere",
        evolution="二期：球形有向拓扑（FCC 规则堆积）",
        feature_source="ThreeDNeuronSpace.forward -> [B, output_dim=D]",
    ),
    "n3d_proto": BackendSpec(
        name="n3d_proto",
        evolution="一期：随机三维放置原型（四步闭环）",
        feature_source="ThreeDNeuronSpace.forward -> [B, output_dim=D]",
    ),
}


class BackendRegistry:
    """后端适配器注册表：把「特征维 ``D``」映射为「已构造的适配器」。

    参数
    ----
    input_dim : int
        连接参数 ``D``（所有登记后端共享同一 ``D``）。
    structure : Optional[BackendStructure]
        **结构开关**（本轮新增；``None`` = 全默认档，行为与改动前逐位一致）。
        该结构对注册表内**全部**后端生效：对不具备结构能力的后端，非默认结构会在
        :meth:`register` 内显式报错（不静默丢弃）。

    关键不变量
    ----------
    * 同一 ``D`` 下，每个后端名最多登记一次；
    * 登记请求必须给出合法的后端名（未知名立即报错，并列出合法集合）；
    * ``assert_distinguishable()`` 断言不同后端的**结构身份**可区分
      （名 / 类 / 参数量 / ``E`` 四元组两两不同）。
    """

    def __init__(
        self, input_dim: int, structure: Optional[BackendStructure] = None
    ) -> None:
        if int(input_dim) < 1:
            raise ValueError(f"BackendRegistry.input_dim 必须 >= 1，当前 {input_dim}")
        self.input_dim = int(input_dim)
        self.structure: BackendStructure = (
            BackendStructure() if structure is None else structure
        )
        self._adapters: Dict[str, BackendAdapter] = {}
        self._unavailable: Dict[str, str] = {}

    # -- 登记 -------------------------------------------------------------
    def register(self, name: str) -> BackendAdapter:
        """构造并登记一个后端适配器（幂等：已登记则直接返回）。

        参数
        ----
        name : str
            后端名（``BACKEND_NAMES`` 之一）。

        返回
        ----
        BackendAdapter
            登记好的适配器。
        """
        if name not in BACKEND_SPECS:
            raise KeyError(
                f"未登记的后端名 {name!r}；合法集合 = {sorted(BACKEND_SPECS.keys())}"
            )
        if name in self._adapters:
            return self._adapters[name]
        model = self._build_model(name)
        adapter = BackendAdapter(BACKEND_SPECS[name], model, self.input_dim, self.structure)
        self._adapters[name] = adapter
        return adapter

    def register_all(self) -> Dict[str, BackendAdapter]:
        """登记全部三后端；**构造失败的登记为不可用**（不静默丢弃，返回时也不抛错）。

        返回
        ----
        Dict[str, BackendAdapter]
            成功登记的适配器（键 = 后端名）。
        """
        for name in BACKEND_NAMES:
            try:
                self.register(name)
            except Exception as exc:  # noqa: BLE001 - 失败原因必须原样登记
                self._unavailable[name] = f"{type(exc).__name__}: {exc}"
        return dict(self._adapters)

    def _build_model(self, name: str) -> nn.Module:
        """按后端名构造模型（只读 import 上游 ``model`` / ``config``）。"""
        cfg = recommended_config(name, self.input_dim, self.structure)
        if name == "n3d_shape":
            from n3d_shape.model import ThreeDNeuronSpace as _M

            return _M(cfg)
        if name == "n3d_sphere":
            from n3d_sphere.model import ThreeDNeuronSpace as _M

            return _M(cfg)
        if name == "n3d_proto":
            from n3d_proto.model import ThreeDNeuronSpace as _M

            return _M(cfg)
        raise KeyError(f"未登记的后端名 {name!r}")

    # -- 查询 -------------------------------------------------------------
    def get(self, name: str) -> BackendAdapter:
        """取已登记的适配器；未登记时立即报错（不隐式构造）。"""
        if name not in self._adapters:
            raise KeyError(
                f"后端 {name!r} 未登记；已登记 = {sorted(self._adapters.keys())}，"
                f"不可用 = {sorted(self._unavailable.keys())}"
            )
        return self._adapters[name]

    @property
    def available(self) -> List[str]:
        """已成功登记的后端名（升序）。"""
        return sorted(self._adapters.keys())

    @property
    def unavailable(self) -> Dict[str, str]:
        """构造失败的后端名 -> 可读原因（**显式登记，禁止静默丢弃**）。"""
        return dict(self._unavailable)

    def manifest(self) -> Dict[str, Any]:
        """注册表清单（供 P0 探针报告与产物 meta）。

        新增键 ``structure``：现场登记本注册表所用的结构开关（默认档也照实写出，
        便于事后回答"这份数字是哪一组结构跑出来的"）。
        """
        return {
            "input_dim": int(self.input_dim),
            "available": self.available,
            "unavailable": self.unavailable,
            "structure": self.structure.as_dict(),
            "structure_is_default": bool(self.structure.is_default()),
            "structure_aware_backends": list(STRUCTURE_AWARE_BACKENDS),
            "specs": {
                n: {
                    "evolution": s.evolution,
                    "feature_source": s.feature_source,
                }
                for n, s in BACKEND_SPECS.items()
            },
        }

    # -- 可区分性 ---------------------------------------------------------
    def structural_fingerprint(self, name: str) -> Dict[str, Any]:
        """返回某后端的**结构身份**四元组（名 / 类 / 参数量 / ``E``）。

        参数
        ----
        name : str
            后端名（必须已登记）。

        返回
        ----
        Dict[str, Any]
            ``{"name", "class", "params", "E"}``。
        """
        adapter = self.get(name)
        stats = adapter.topology_stats()
        return {
            "name": adapter.name,
            "class": type(adapter.model).__module__ + "." + type(adapter.model).__name__,
            "params": int(adapter.count_parameters()),
            "E": float(stats.get("E", -1.0)),
        }

    def assert_distinguishable(self) -> List[Dict[str, Any]]:
        """断言已登记后端两两**可区分**（构造期不变量）。

        返回
        ----
        List[Dict[str, Any]]
            各后端结构身份四元组（升序按名）。

        异常
        ------
        AssertionError
            任意两后端的结构身份四元组完全相同（报文点名冲突对与四元组）。
        """
        fps = [self.structural_fingerprint(n) for n in self.available]
        seen: Dict[str, Any] = {}
        for fp in fps:
            key = json.dumps(fp, sort_keys=True, ensure_ascii=False)
            if key in seen:
                raise AssertionError(
                    "后端可区分性断言失败：两个后端结构身份完全相同 "
                    f"{fp}（与 {seen[key]}）；注册表无法区分它们"
                )
            seen[key] = fp
        return fps


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def build_registry(
    vectorizer_config: Optional[VectorizerConfig] = None,
    input_dim: Optional[int] = None,
    structure: Optional[BackendStructure] = None,
) -> BackendRegistry:
    """按向量化口径构造注册表（``D`` 的唯一来源，避免两处各写一个维度）。

    参数
    ----
    vectorizer_config : Optional[VectorizerConfig]
        向量化口径；为 ``None`` 时用默认口径（若同时给了 ``input_dim`` 则以 ``input_dim``
        为唯一来源，见下）。
    input_dim : Optional[int]
        显式覆盖 ``D``；与向量化口径**同时给出**且冲突时立即报错。
    structure : Optional[BackendStructure]
        **结构开关**（本轮新增；``None`` = 全默认档，行为与改动前逐位一致）。

    返回
    ----
    BackendRegistry
        已登记全部可构造后端的注册表。

    说明
    ----
    「只给 ``input_dim``、不给 ``vectorizer_config``」是**可插拔编码器**（如
    :mod:`n3d_qa_learn.encoders` 的 HF 编码器，其 ``D = hidden_size`` 不由
    ``VectorizerConfig`` 决定）的入口：此时 ``input_dim`` 即连接参数 ``D``。
    """
    if input_dim is not None and vectorizer_config is None:
        dim = int(input_dim)
    else:
        cfg = vectorizer_config if vectorizer_config is not None else VectorizerConfig()
        dim = int(cfg.dim)
        if input_dim is not None and int(input_dim) != dim:
            raise ValueError(
                f"显式 input_dim={input_dim} 与向量化口径维度 {dim} 冲突；"
                "连接参数 D 只有唯一来源（向量化口径），拒绝两处各写一个维度"
            )
    reg = BackendRegistry(dim, structure=structure)
    reg.register_all()
    return reg


def dim_fingerprint(input_dim: int, backend: str) -> str:
    """生成 ``(D, 后端名)`` 的短指纹（产物命名 / 自检用）。"""
    blob = json.dumps({"D": int(input_dim), "backend": str(backend)}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


__all__ = [
    "BACKEND_NAMES",
    "BACKEND_SPECS",
    "STRUCTURE_AWARE_BACKENDS",
    "STRUCTURE_DEFAULTS",
    "SHAPE_CHOICES",
    "GEO_FIELD_CHOICES",
    "BackendSpec",
    "BackendStructure",
    "BackendAdapter",
    "BackendRegistry",
    "recommended_config",
    "build_registry",
    "dim_fingerprint",
]