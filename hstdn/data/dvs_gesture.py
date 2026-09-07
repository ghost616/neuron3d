"""dvs_gesture.py -- DVS-Gesture 加载与 8x8 patch 双极性聚合（G3，data 模块）。

**预留骨架**：本里程碑（G1 前）不实现任何功能；全部入口显式抛出
NotImplementedError（L1 纪律：禁止 pass 占位）。G3 排期时按 D14 裁决与
B 系列勘误实现。

D14/D3 设计要点（职责边界，G3 启动时定稿精确几何）：
1. **数据源**：DVS-Gesture 事件流经 tonic 加载（README：tonic 属 G3
   延迟依赖，届时单独安装；缺失时给出可读安装指引报错）—— 本模块与
   mnist.py 同构地提供「原始数据 -> 子集」加载入口；
2. **输入映射目标**：N_IN = 512 个 DVS 输入单元（D14）。设计草案：把事件
   平面按 **8x8 空间 patch** 聚合（每 patch 内按极性 on/off 双通道统计），
   几何猜想 8x8 空间位置 x 2 极性 x 4 时间窗 = 512 单元 —— 该分解仅为
   设计草图，最终 patch/时间窗布局以 G3 启动时 D14 裁决/勘误为准，
   本文件 docstring **不制造权威数字**；
3. **模块边界**：本模块产出「原始事件/单元级强度-计数数据」；
   单元 -> spike 桶化由 core.encoder 的 ``dvs_patch_aggregate`` 接口
   （N_IN=512 契约）负责，data 模块不重复实现编码器；
4. 顶层导入只依赖 numpy；tonic/torch 一律在函数体内延迟引入。

自测：见 ``__main__`` —— 仅验证占位入口确实抛 NotImplementedError。
"""

from __future__ import annotations

import sys

__all__ = ["load_dvs_gesture_subset", "aggregate_dvs_events"]


def load_dvs_gesture_subset(n: int = 0, *, root: str = "data/dvs_gesture",
                            train: bool = True, seed: int = 0):
    """(G3 预留) 加载 DVS-Gesture 的 n 样本确定性子集。

    Args:
        n: 子集规模（G3 定稿后生效；0 表示使用里程碑默认值）。
        root: tonic 数据集落盘目录。
        train: True=训练集，False=测试集。
        seed: 子集抽取随机种子。

    Returns:
        G3 定稿的原始事件子集结构（见里程碑设计文档）。

    Raises:
        NotImplementedError: 本里程碑（G1 前）未实现。
    """
    raise NotImplementedError(
        "load_dvs_gesture_subset belongs to the G3/DVS milestone (tonic "
        "delayed dependency + D14 patch aggregation plan); it is not "
        "implemented in the D1-D4/G1 data-module scope"
    )


def aggregate_dvs_events(events, *, patch: int = 8):
    """(G3 预留) 8x8 patch 双极性聚合：事件 -> 512 单元强度/计数。

    Args:
        events: DVS-Gesture 原始事件结构（G3 定稿 schema）。
        patch: 空间 patch 边长（草案 8；最终以 D14 裁决为准）。

    Returns:
        单元级聚合数据（供 core.encoder.dvs_patch_aggregate 消费）。

    Raises:
        NotImplementedError: 本里程碑（G1 前）未实现。
    """
    raise NotImplementedError(
        "aggregate_dvs_events belongs to the G3/DVS milestone; final "
        "8x8-patch x bipolar -> 512-unit layout is fixed against the D14 "
        "ruling at G3 planning time -- not implemented in the G1 data scope"
    )


def main() -> int:
    """骨架自测：占位入口必须抛 NotImplementedError（无静默 pass）。"""
    print("dvs_gesture skeleton (G3 placeholder):")
    for fn, args in [(load_dvs_gesture_subset, ()),
                     (aggregate_dvs_events, (None,))]:
        try:
            fn(*args)
        except NotImplementedError:
            print(f"  [PASS] {fn.__name__} raises NotImplementedError")
        else:
            raise AssertionError(f"{fn.__name__} must raise NotImplementedError")
    print("\ndvs_gesture skeleton check PASSED (placeholders compliant)")
    return 0


if __name__ == "__main__":
    sys.exit(main())