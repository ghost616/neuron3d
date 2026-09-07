"""mnist.py -- MNIST 1k 子集加载 + adaptive_avg_pool2d 28x28->10x10（G2，data 模块）。

D14 映射：28x28 MNIST 灰度补丁经自适应平均池化到 10x10 单元网格
（= 100 个输入单元；unit id 行主序 = r * out + c），与 encoder 消费的布局
一致（cfg.n_in == 100 / n_input_cols == 10，G1/G2 网络共用）。池化语义与
torch ``adaptive_avg_pool2d`` 一致，并与 ``core.encoder.mnist_adaptive_pool``
逐帧等价 —— 本文件提供**批次向量化**实现（N 轴整块切分求均值），自测中与
encoder 的逐帧实现做交叉一致性校验（防止两处语义漂移）。

依赖纪律（README：torch/torchvision 属 G2 延迟引入，G1 前禁止安装）：
- ``adaptive_pool_ranges`` / ``pool_mnist_to_units`` 纯 NumPy：torch 未装
  也可用（G2 前即可被 G0/映射相关断言直接测试）；
- ``load_mnist_subset`` 需要 torchvision 下载 MNIST：在函数体内**延迟**
  import torch + torchvision，缺失时抛带安装指引的可读 ImportError；
- 本模块顶层导入不触碰 torch，MNIST 像素在加载函数内归一化到 [0,1]。

G2 交付范围：加载 + 池化映射（含 1k 确定性子集抽取）。数据集缓存 / 批量
DataLoader / 训练-测试划分约定留给 G2 读出端工作，不在此占位。

自测：``python -m hstdn.data.mnist`` —— torch-free 部分全部执行；下载路径
只做错误分支检查（torch 缺失时验证可读报错；torch 已装时跳过真实下载）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from hstdn.core.layout import F8, I8

__all__ = ["MNIST_SIDE", "DEFAULT_SUBSET_N", "DEFAULT_ROOT",
           "adaptive_pool_ranges", "pool_mnist_to_units", "MnistSubset",
           "load_mnist_subset"]

#: MNIST 补丁边长（像素）。
MNIST_SIDE = 28
#: 默认子集规模（G2 1k）。
DEFAULT_SUBSET_N = 1000
#: torchvision MNIST 数据落盘目录（相对项目根；调用时自动创建/下载）。
DEFAULT_ROOT = "data/mnist"

_PASS = 0


def _ok(name: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  [PASS] {name}")


def adaptive_pool_ranges(n_in: int, n_out: int) -> np.ndarray:
    """自适应平均池化区间分界（D14；torch adaptive_avg_pool2d 语义）。

    输出单元 j 平均输入区间 [start_j, end_j)：
        start_j = floor(j * n_in / n_out)
        end_j   = ceil((j + 1) * n_in / n_out)
    （与 core.encoder 内部同款公式；本模块自测对其做交叉一致性校验。）

    Args:
        n_in: 输入侧长度（如 28）。
        n_out: 输出侧长度（如 10）。

    Returns:
        (n_out, 2) int64：第 j 行为 (start_j, end_j)。
    """
    idx = np.arange(n_out, dtype=np.float64)
    starts = np.floor(idx * n_in / n_out).astype(np.int64)
    ends = np.ceil((idx + 1.0) * n_in / n_out).astype(np.int64)
    return np.stack([starts, ends], axis=1)


def pool_mnist_to_units(images, out: int = 10) -> np.ndarray:
    """批量自适应平均池化：28x28 补丁 -> (out, out) 单元强度图（D14）。

    Args:
        images: (N, MNIST_SIDE, MNIST_SIDE) 数值数组；像素应为 [0,1]
            归一化值（load_mnist_subset 已归一化；原生 0..255 输入请先 /255）。
        out: 单元网格边长（10 -> 100 单元，D14；unit id = r * out + c）。

    Returns:
        (N, out, out) float64 池化均值；与逐帧调用
        ``core.encoder.mnist_adaptive_pool`` 的结果一致。

    Raises:
        AssertionError: images 非 3-D / 非 28x28 方形补丁，或 out 非法。
    """
    arr = np.asarray(images)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise AssertionError(
            f"images must be (N, S, S) square patches, got {arr.shape}"
        )
    if arr.shape[1] != MNIST_SIDE:
        raise AssertionError(
            f"expected {MNIST_SIDE}x{MNIST_SIDE} MNIST patches, got "
            f"{arr.shape[1]}x{arr.shape[2]}"
        )
    if out <= 0:
        raise AssertionError(f"out must be positive, got {out}")
    rows = adaptive_pool_ranges(MNIST_SIDE, out)
    cols = adaptive_pool_ranges(MNIST_SIDE, out)
    n = int(arr.shape[0])
    pooled = np.zeros((n, out, out), dtype=F8)
    for r in range(out):
        r0, r1 = int(rows[r, 0]), int(rows[r, 1])
        for c in range(out):
            c0, c1 = int(cols[c, 0]), int(cols[c, 1])
            pooled[:, r, c] = arr[:, r0:r1, c0:c1].mean(axis=(1, 2))
    return pooled


@dataclass
class MnistSubset:
    """MNIST 子集容器：images (N,28,28) f8 像素 [0,1]，labels (N,) i8 数字。

    N 与采样方式（确定性随机子集）由 load_mnist_subset 决定；本容器额外提供
    D14 单元映射便捷方法（pool/flat，纯 NumPy，torch 无关）。
    """

    images: np.ndarray   # (N, 28, 28) f8
    labels: np.ndarray   # (N,) i8

    def __post_init__(self) -> None:
        im = np.asarray(self.images)
        lb = np.asarray(self.labels)
        if im.ndim != 3 or im.shape[1:] != (MNIST_SIDE, MNIST_SIDE):
            raise ValueError(
                f"images must be (N, {MNIST_SIDE}, {MNIST_SIDE}), "
                f"got {im.shape}"
            )
        if lb.shape != (im.shape[0],):
            raise ValueError(
                f"labels must be ({im.shape[0]},), got {lb.shape}"
            )
        if im.dtype != F8:
            im = im.astype(F8)
        if lb.dtype != I8:
            lb = lb.astype(I8)
        self.images = im
        self.labels = lb

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def pool(self, out: int = 10) -> np.ndarray:
        """(N, out, out) 单元强度图（adaptive avg pool，D14）。"""
        return pool_mnist_to_units(self.images, out=out)

    def flat(self, out: int = 10) -> np.ndarray:
        """(N, out * out) 行主序强度向量：unit id = r * out + c。"""
        pooled = self.pool(out)
        return pooled.reshape(int(pooled.shape[0]), out * out)


def load_mnist_subset(n: int = DEFAULT_SUBSET_N, *, train: bool = True,
                      root: str = DEFAULT_ROOT, seed: int = 0) -> MnistSubset:
    """加载 MNIST 的 n 样本确定性子集（torchvision 按需下载）。

    Args:
        n: 子集规模（> 0 且不超过对应数据集大小；默认 1000）。
        train: True=训练集（60000），False=测试集（10000）。
        root: torchvision 数据集落盘目录（不存在则自动创建并下载）。
        seed: 子集抽取随机种子（rng.choice 不放回，确定性可复现）。

    Returns:
        MnistSubset：images (n, 28, 28) f8 像素归一化到 [0,1]
        （ToTensor 变换），labels (n,) i8 数字 0..9。

    Raises:
        ImportError: torch / torchvision 未安装（G2 延迟依赖，附安装指引）。
        ValueError: n 非法或超过数据集大小。
    """
    if int(n) <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    try:
        import torch  # noqa: F401  —— 延迟引入（README：G2 才安装 torch）
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError(
            "load_mnist_subset requires torch + torchvision (delayed G2 "
            "dependencies; README forbids installing them before G2).  Once "
            "the G2 milestone starts, install e.g.:  pip install torch "
            "torchvision --index-url https://download.pytorch.org/whl/cpu"
        ) from exc
    ds = datasets.MNIST(root=root, train=train, download=True,
                        transform=transforms.ToTensor())
    if int(n) > len(ds):
        split = "train" if train else "test"
        raise ValueError(
            f"n={n} exceeds the MNIST {split} set size ({len(ds)})"
        )
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(ds), size=int(n), replace=False)
    items = [ds[int(i)] for i in idx]
    # ToTensor 已把 uint8 像素归一化到 [0,1]；仅做 numpy 化与形状整理
    images = np.stack([np.asarray(img) for img, _ in items])  # (n,1,28,28)
    images = np.ascontiguousarray(images[:, 0, :, :]).astype(F8)
    labels = np.asarray([int(lb) for _, lb in items], dtype=I8)
    return MnistSubset(images=images, labels=labels)


# ---------------------------------------------------------------------------
# 本地自测（torch-free 映射 + 加载器错误分支；真实下载不在自测中执行）
# ---------------------------------------------------------------------------


def check_pool_mapping() -> None:
    """池化映射与 core.encoder 交叉一致 + 输入守卫。"""
    print("mnist pool mapping (D14):")
    rng = np.random.default_rng(5)
    imgs = rng.uniform(0.0, 1.0, size=(6, MNIST_SIDE, MNIST_SIDE))
    pooled = pool_mnist_to_units(imgs, out=10)
    assert pooled.shape == (6, 10, 10)
    assert pooled.dtype == F8
    assert np.isfinite(pooled).all()
    from hstdn.core.encoder import mnist_adaptive_pool
    for i in range(6):
        ref = mnist_adaptive_pool(imgs[i], out=10)
        assert np.allclose(pooled[i], ref, atol=1e-12), \
            "batch pool must match per-frame encoder pool"
    ranges = adaptive_pool_ranges(MNIST_SIDE, 10)
    assert ranges.shape == (10, 2) and ranges.dtype == np.int64
    try:
        pool_mnist_to_units(np.zeros((4, 14, 14)))
    except AssertionError:
        pass
    else:
        raise AssertionError("non-28x28 patches must be rejected")
    try:
        pool_mnist_to_units(np.zeros((3, MNIST_SIDE, MNIST_SIDE)), out=0)
    except AssertionError:
        pass
    else:
        raise AssertionError("out <= 0 must be rejected")
    _ok("batch pool == per-frame encoder pool; shape guards hold")


def check_subset_container() -> None:
    """MnistSubset 形状/dtype/标签契约。"""
    print("mnist subset container:")
    imgs = np.zeros((3, MNIST_SIDE, MNIST_SIDE), dtype=F8)
    labels = np.asarray([1, 2, 3], dtype=I8)
    s = MnistSubset(images=imgs, labels=labels)
    assert len(s) == 3
    p = s.pool(out=10)
    assert p.shape == (3, 10, 10)
    f = s.flat(out=10)
    assert f.shape == (3, 100)
    assert np.isfinite(f).all()
    try:
        MnistSubset(images=np.zeros((3, MNIST_SIDE, MNIST_SIDE)),
                    labels=np.zeros(4, dtype=I8))
    except ValueError:
        pass
    else:
        raise AssertionError("labels length mismatch must raise")
    _ok("MnistSubset shape/dtype/labels validation")


def check_loader_gate() -> None:
    """加载器延迟依赖门（不触发真实下载）。"""
    print("mnist loader gate (torch lazy):")
    try:
        import torch  # noqa: F401
        from torchvision import datasets, transforms  # noqa: F401
    except ImportError:
        missing = True
    else:
        missing = False
    if missing:
        try:
            load_mnist_subset(2)
        except ImportError as exc:
            msg = str(exc)
            assert "torch" in msg and "G2" in msg, \
                "loader ImportError must carry install guidance"
        else:
            raise AssertionError(
                "load_mnist_subset must raise when torch is unavailable"
            )
        _ok("loader raises readable delayed-dependency error (torch absent)")
    else:
        print("  [SKIP] torch installed: real MNIST download is deferred to "
              "G2 and is not run inside the self-test")
        _ok("loader gate skipped (torch present; download deferred to G2)")


def main() -> int:
    """运行 mnist 模块自测（退出码 0 = 通过）。"""
    check_pool_mapping()
    check_subset_container()
    check_loader_gate()
    print(f"\nmnist selfcheck PASSED: {_PASS} assertion groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())