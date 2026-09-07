"""hstdn.bridge -- CPU-GPU 批量桥接（G2/G3 里程碑；data 模块预留）。

当前仅含 gpu_bridge.py 预留骨架：批量缓冲 copy_ 桥接设计说明 + 显式
NotImplementedError 占位（实现随 G2 引入 torch 后进行）。顶层导入轻量。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]