"""gpu_bridge.py -- CPU-GPU 批量缓冲 copy_ 桥接（G2/G3；data 模块预留骨架）。

**目的与边界**
----------------
G2 起训练循环需在 CPU（NumPy 网络 / data 数据）与 GPU（torch 读出端）之间
搬运数据：输入单元强度向量（MNIST 型 (B, N_UNITS=100) / DVS 型 (B, 512)，
均按 D14 映射后的单元契约行主序）与池 spike 计数 / 读出标签。
**桥接只做纯搬运，不做 D14 映射**：强度向量的生成属 data 模块
（synthetic/mnist/dvs_gesture），强度 -> spike 桶属 core/encoder。

**设计要点（本骨架固定设计层面约定；实现随 G2 引入 torch）**
1. **batch 级 copy_**：整批数据一次性拷入/拷出连续缓冲 ——
   ``device_tensor.copy_(torch.from_numpy(host_batch))`` 或反向整块 copy_，
   **禁止逐样本 Python 循环拷贝**（热路径纪律，见模块约定）；
2. 缓冲布局 = 输入映射单元契约：MNIST (B, 100)、DVS (B, 512)，行主序；
   dtype 转换（float64 状态 / float32 torch 习惯）由调用方显式指定，
   桥接不做隐式转换；
3. 设备字符串（cuda:0 / cpu）与张量生命周期管理归 G2 训练循环，
   本模块只提供无状态搬运入口；
4. torch 延迟引入（README G2 规则）：import 本模块不触碰 torch，
   函数实现时按需 import 并给出可读报错。

**当前里程碑（G1 前）**：两个入口均显式 raise NotImplementedError
（无静默 pass）。自测见 ``__main__``。
"""

from __future__ import annotations

import sys

__all__ = ["copy_intensity_batch_to_device", "copy_tensor_batch_to_host"]


def copy_intensity_batch_to_device(host: "np.ndarray", device: str = "cuda:0",
                                   *, dtype: str = "float32"):
    """(G2 预留) CPU 强度批次 -> GPU 张量（整块 copy_）。

    Args:
        host: (B, n_units) CPU ndarray（n_units = 100 MNIST / 512 DVS，
            按 D14 单元契约）。
        device: 目标设备字符串（如 "cuda:0"）。
        dtype: 目标 dtype 名（"float32"/"float64"）。

    Returns:
        torch.Tensor（G2 定稿）。

    Raises:
        NotImplementedError: 本里程碑（G1 前）未实现。
    """
    raise NotImplementedError(
        "gpu_bridge.copy_intensity_batch_to_device: batch-level copy_ bridge "
        "belongs to the G2 milestone (torch delayed dependency); not "
        "implemented in the G1 data scope -- no per-sample copy loops allowed"
    )


def copy_tensor_batch_to_host(device_tensor, *, dtype: str = "float64"
                              ) -> "np.ndarray":
    """(G2 预留) GPU 张量批次 -> CPU ndarray（整块 copy_）。

    Args:
        device_tensor: GPU 上的 torch 张量（读出端输出）。
        dtype: 目标 numpy dtype 名（"float64"/"float32"）。

    Returns:
        CPU ndarray（G2 定稿）。

    Raises:
        NotImplementedError: 本里程碑（G1 前）未实现。
    """
    raise NotImplementedError(
        "gpu_bridge.copy_tensor_batch_to_host: batch-level copy_ bridge "
        "belongs to the G2 milestone (torch delayed dependency); not "
        "implemented in the G1 data scope"
    )


def main() -> int:
    """骨架自测：占位入口必须抛 NotImplementedError（无静默 pass）。"""
    print("gpu_bridge skeleton (G2 placeholder):")
    for fn, args in [(copy_intensity_batch_to_device, (None,)),
                     (copy_tensor_batch_to_host, (None,))]:
        try:
            fn(*args)
        except NotImplementedError:
            print(f"  [PASS] {fn.__name__} raises NotImplementedError")
        else:
            raise AssertionError(f"{fn.__name__} must raise NotImplementedError")
    print("\ngpu_bridge skeleton check PASSED (placeholders compliant)")
    return 0


if __name__ == "__main__":
    sys.exit(main())