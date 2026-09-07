"""H-STDN（混合时空脉冲神经网络）包根 —— v3.2-final 工程化实现。

依据《H-STDN 详细设计文档 v3.2-final》（唯一权威规范：D1-D14 设计裁决、
B1-B11 勘误、§2 数据布局、§3 模块设计 M1-M6、§4 超参、§9 实施计划）构建，
按 G0-G4 分级验证推进；G0 十五项断言（hstdn/exp/gates.py）为项目门槛。

代码布局（各目录由对应业务模块维护）：
    hstdn/core/    核心模拟 M1-M6（layout/network/spatial_hash/encoder/kernel/
                   plasticity/features/readout/scheduler）
    hstdn/configs/ §4 超参唯一契约（default.yaml）
    hstdn/exp/     门禁 gates.py（G0 十五项断言）、诊断 diagnostics.py
    hstdn/data/    合成数据 / MNIST / DVS-Gesture / GPU 桥接
    hstdn/main/    训练与评估入口

本文件由 framework 模块维护：作为包公共入口，重导出 hstdn.checks 的轻量
G0 契约断言工具（供 core/layout 与 exp/gates 复用），并保持导入轻量——
业务契约断言的实现细节由 core 模块负责。
"""

from __future__ import annotations

from .checks import (
    assert_dtype,
    assert_finite,
    assert_global_ids,
    assert_local_ids,
    assert_non_decreasing,
    assert_shape,
)

__version__ = "0.1.0"

__all__ = [
    "assert_dtype",
    "assert_finite",
    "assert_global_ids",
    "assert_local_ids",
    "assert_non_decreasing",
    "assert_shape",
    "__version__",
]