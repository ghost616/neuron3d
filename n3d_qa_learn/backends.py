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


def recommended_config(name: str, input_dim: int) -> Any:
    """返回某后端在 QA 任务上的**推荐构型**（只设置连接参数 + 该后端原生结构参数）。

    参数
    ----
    name : str
        后端名（``BACKEND_NAMES`` 之一）。
    input_dim : int
        连接参数：特征维 ``D``（同时作为该后端的 ``input_dim`` 与 ``output_dim``）。

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
    """
    dim = int(input_dim)
    if dim < 1:
        raise ValueError(f"input_dim 必须 >= 1，当前 {dim}")
    if name == "n3d_shape":
        from n3d_shape.config import Config as _ShapeConfig

        return _ShapeConfig(
            N=64, y_in=4, y_out=4, H=0.15, D=0.15,
            input_dim=dim, output_dim=dim,
            input_scope="any_isolated", readout_scope="any_isolated",
            shape="sphere", dataset="npz", seed=42,
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

    def __init__(self, spec: BackendSpec, model: nn.Module, input_dim: int) -> None:
        self.spec = spec
        self.model = model
        self.input_dim = int(input_dim)

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
        """适配器自描述（写进产物 meta，供事后复核连接参数）。"""
        return {
            "backend": self.name,
            "evolution": self.spec.evolution,
            "feature_source": self.spec.feature_source,
            "input_dim": int(self.input_dim),
            "output_dim": int(self.input_dim),
            "topology_stats": {k: float(v) for k, v in self.topology_stats().items()},
            "parameters": int(self.count_parameters()),
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

    关键不变量
    ----------
    * 同一 ``D`` 下，每个后端名最多登记一次；
    * 登记请求必须给出合法的后端名（未知名立即报错，并列出合法集合）；
    * ``assert_distinguishable()`` 断言不同后端的**结构身份**可区分
      （名 / 类 / 参数量 / ``E`` 四元组两两不同）。
    """

    def __init__(self, input_dim: int) -> None:
        if int(input_dim) < 1:
            raise ValueError(f"BackendRegistry.input_dim 必须 >= 1，当前 {input_dim}")
        self.input_dim = int(input_dim)
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
        adapter = BackendAdapter(BACKEND_SPECS[name], model, self.input_dim)
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
        cfg = recommended_config(name, self.input_dim)
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
        """注册表清单（供 P0 探针报告与产物 meta）。"""
        return {
            "input_dim": int(self.input_dim),
            "available": self.available,
            "unavailable": self.unavailable,
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
    vectorizer_config: Optional[VectorizerConfig] = None, input_dim: Optional[int] = None
) -> BackendRegistry:
    """按向量化口径构造注册表（``D`` 的唯一来源，避免两处各写一个维度）。

    参数
    ----
    vectorizer_config : Optional[VectorizerConfig]
        向量化口径；为 ``None`` 时用默认口径。
    input_dim : Optional[int]
        显式覆盖 ``D``；与向量化口径冲突时立即报错。

    返回
    ----
    BackendRegistry
        已登记全部可构造后端的注册表。
    """
    cfg = vectorizer_config if vectorizer_config is not None else VectorizerConfig()
    dim = int(cfg.dim)
    if input_dim is not None and int(input_dim) != dim:
        raise ValueError(
            f"显式 input_dim={input_dim} 与向量化口径维度 {dim} 冲突；"
            "连接参数 D 只有唯一来源（向量化口径），拒绝两处各写一个维度"
        )
    reg = BackendRegistry(dim)
    reg.register_all()
    return reg


def dim_fingerprint(input_dim: int, backend: str) -> str:
    """生成 ``(D, 后端名)`` 的短指纹（产物命名 / 自检用）。"""
    blob = json.dumps({"D": int(input_dim), "backend": str(backend)}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


__all__ = [
    "BACKEND_NAMES",
    "BACKEND_SPECS",
    "BackendSpec",
    "BackendAdapter",
    "BackendRegistry",
    "recommended_config",
    "build_registry",
    "dim_fingerprint",
]