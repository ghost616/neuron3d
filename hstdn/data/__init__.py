"""hstdn.data -- H-STDN 数据模块（D3 数据部分与 D14 输入映射）。

子模块：
    synthetic.py    合成 10 类数据生成器（G1 首跑；MNIST 10x10 单元布局，
                    N_IN=100；固定活动模式 + 可加噪）—— 首期完整可用
    mnist.py        MNIST 1k 子集加载 + adaptive_avg_pool2d 28x28->10x10
                    映射（D14，G2 期使用；torch/torchvision 延迟引入）
    dvs_gesture.py  DVS-Gesture 加载与 8x8 patch x 双极性聚合（D14，G3）
                    —— 预留骨架（NotImplementedError 占位）
    hstdn/bridge/   CPU-GPU 批量缓冲 copy_ 桥接（G2/G3）—— 预留骨架

与 encoder 的输入契约对齐：本模块只产出「输入单元强度图/强度向量」
（MNIST 型 10x10 = 100 单元、DVS 型 512 单元），不做 ID 分配；
spike 桶（含 input 全局 ID 0..N_IN-1）由 core/encoder 的 latency_encode
等接口负责。保持导入轻量：除 layout 常量外不加载重型子模块。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]