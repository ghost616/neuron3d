"""N3D 一期原型数据层：MNIST 加载（复用工程内已有 IDX 文件，避免重复联网下载）。

数据约定
--------
* 28x28 灰度图展平为 784 维向量，与 `Config.input_dim=784` 对应；
* 归一化到零均值、单位标准差（按 MNIST 全局统计量），有利于迭代闭环的数值稳定；
* 优先复用工程内 `data/mnist/*.gz` 原始 IDX 文件；仅当本地完全没有时才回退到
  torchvision 的联网下载（设置 `allow_download=True`）。

关键点
------
torchvision 的 `MNIST(root, ...)` 会按 `<root>/MNIST/raw/<name>` 查找文件。工程内文件
位于 `data/mnist/`，因此本模块提供"非破坏性探测"：若 `<root>/MNIST/raw` 不存在，
则尝试复用 `data/mnist/` 下的同名文件（复制到 torchvision 期望的位置）；若连复制都
不允许（只读环境），则直接解析 IDX 自行构造 Dataset。
"""

from __future__ import annotations

import gzip
import os
import random
import shutil
import struct
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:  # 兼容包导入与脚本直跑
    from .utils import log_info, log_warn
except ImportError:  # pragma: no cover
    from utils import log_info, log_warn  # type: ignore

__all__ = [
    "MNIST_IDX_FILES",
    "MNIST_MEAN",
    "MNIST_STD",
    "RawIdxMNIST",
    "ensure_mnist_files",
    "get_mnist_loaders",
]

# torchvision 期望的四个原始文件名
MNIST_IDX_FILES: Tuple[str, ...] = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

# MNIST 全局归一化统计量（训练集）
MNIST_MEAN: float = 0.1307
MNIST_STD: float = 0.3081

# 工程内可能存放原始 IDX 文件的候选目录（相对工程根目录）
_CANDIDATE_DIRS: Tuple[str, ...] = (
    "data/mnist",
    "data/MNIST/raw",
    "data/mnist/MNIST/raw",
)

def _read_idx_images(path: str) -> np.ndarray:
    """解析 MNIST 图像 IDX 文件（支持 .gz 与未压缩）。

    参数
    ----
    path : str
        IDX 文件路径。

    返回
    ----
    np.ndarray
        形状 [N, 784] 的 float32 数组，取值 [0, 1]（除以 255）。

    异常
    ------
    ValueError
        文件 magic number 或维度信息不合法时抛出。
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(
                f"{path} 不是合法的 MNIST 图像 IDX 文件（magic={magic}，期望 2051）"
            )
        buf = f.read(num * rows * cols)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows * cols)
    return (arr.astype(np.float32) / 255.0)


def _read_idx_labels(path: str) -> np.ndarray:
    """解析 MNIST 标签 IDX 文件。

    参数
    ----
    path : str
        IDX 文件路径。

    返回
    ----
    np.ndarray
        形状 [N] 的 int64 标签数组。
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(
                f"{path} 不是合法的 MNIST 标签 IDX 文件（magic={magic}，期望 2049）"
            )
        buf = f.read(num)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


class RawIdxMNIST(Dataset):
    """直接基于 IDX 文件的 MNIST Dataset（不依赖 torchvision，纯本地读取）。

    参数
    ----
    image_path : str
        图像 IDX 文件路径（.gz 或未压缩）。
    label_path : str
        标签 IDX 文件路径。
    normalize : bool
        是否按 MNIST 全局统计量做零均值单位方差归一化。默认 True。

    关键不变量
    ----------
    * 图像数与标签数一致；
    * `__getitem__` 返回 (float32 张量 [784], int64 标量标签)。
    """

    def __init__(self, image_path: str, label_path: str, normalize: bool = True) -> None:
        self.normalize = bool(normalize)
        self.image_path = image_path
        self.label_path = label_path
        # 惰性缓存：首次访问时才解析 IDX 文件，避免"仅跑 1 个 batch"的冒烟测试
        # 白白承担 60000 张图的解压解析开销
        self._images: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    @property
    def images(self) -> np.ndarray:
        """形状 [N, 784] 的 float32 图像数组（首次访问时从 IDX 文件解析并缓存）。"""
        if self._images is None:
            self._images = _read_idx_images(self.image_path)
        return self._images

    @property
    def labels(self) -> np.ndarray:
        """形状 [N] 的 int64 标签数组（首次访问时从 IDX 文件解析并缓存）。"""
        if self._labels is None:
            self._labels = _read_idx_labels(self.label_path)
            if self._labels.shape[0] != self.images.shape[0]:
                raise ValueError(
                    f"图像数与标签数不一致：{self.images.shape[0]} vs {self._labels.shape[0]}"
                    f"（图像 {self.image_path}，标签 {self.label_path}）"
                )
        return self._labels

    def __len__(self) -> int:
        """返回样本数。

        实现要点：只解析（并缓存）标签文件即可得到长度，**不会触发图像文件的全量
        解压解析**——这保证"仅跑 1 个 batch"的冒烟测试不会为 60000 张图付出无谓开销。
        """
        if self._labels is None:
            self._labels = _read_idx_labels(self.label_path)
        return int(self._labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回第 idx 个样本：(展平并归一化后的图像 [784], 标签标量)。"""
        img = self.images[idx]
        if self.normalize:
            # 按 MNIST 全局统计量归一化，稳定迭代闭环的数值范围
            img = (img - MNIST_MEAN) / MNIST_STD
        return torch.from_numpy(np.ascontiguousarray(img)), torch.tensor(
            int(self.labels[idx]), dtype=torch.long
        )


def _worker_init_fn(worker_id: int) -> None:
    """DataLoader 子进程的种子初始化（保证 `num_workers > 0` 时可复现）。

    PyTorch 的 DataLoader 会为每个 worker 复制父进程的随机状态；若不做 per-worker
    播种，多个 worker 会持有相同的随机状态。这里按 `base_seed + worker_id` 为
    `random` / `numpy` / `torch` 分别播种（遵循 PyTorch 官方推荐写法），
    使 (seed, worker_id) 唯一确定一个 worker 的随机序列。

    **前提**：基种子取自父进程的 `torch.initial_seed()`，因此调用方必须先执行
    `utils.set_seed(config.seed)`，父进程的初始种子才由该 seed 决定
    （`train.build_model_and_data` 已自动完成这一步）。

    参数
    ----
    worker_id : int
        由 DataLoader 传入的 worker 序号。

    返回
    ----
    None
    """
    worker_seed = (torch.initial_seed() + int(worker_id)) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _find_local_idx_dir(explicit_dir: Optional[str] = None) -> Optional[str]:
    """在若干候选目录中查找同时含有 4 个 MNIST IDX 文件的目录。

    参数
    ----
    explicit_dir : Optional[str]
        优先检查的目录（通常来自 `Config.data_root`）。

    返回
    ----
    Optional[str]
        命中的目录路径；若没有任何目录包含全部 4 个文件则返回 None。
    """
    candidates: List[str] = []
    if explicit_dir:
        candidates.append(explicit_dir)
    candidates.extend(_CANDIDATE_DIRS)
    # 同时尝试"工程根目录"与"当前工作目录"两种解析基准
    roots = [os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), os.getcwd()]
    seen = set()
    for cand in candidates:
        for root in roots:
            path = cand if os.path.isabs(cand) else os.path.join(root, cand)
            path = os.path.normpath(path)
            if path in seen:
                continue
            seen.add(path)
            if all(os.path.isfile(os.path.join(path, f)) for f in MNIST_IDX_FILES):
                return path
    return None


def ensure_mnist_files(root: str, allow_download: bool = False) -> str:
    """确保 `<root>/MNIST/raw/` 下存在 4 个 MNIST IDX 文件，并返回 raw 目录。

    策略（按优先级）
    ----------------
    1. `<root>/MNIST/raw/` 已齐备 -> 直接返回（不联网、不复制）；
    2. 在候选目录（`data/mnist` 等）中找到齐备的原始文件 -> 复制到 1 的位置；
    3. 若 `allow_download=True` -> 交给 torchvision 联网下载；
    4. 否则抛出 FileNotFoundError，并给出可读的排查信息。

    参数
    ----
    root : str
        torchvision 风格的数据根目录（`Config.data_root`，通常为 "data/mnist"）。
    allow_download : bool
        是否允许联网下载。默认 False（严格复用本地数据）。

    返回
    ----
    str
        含 4 个 IDX 文件的 raw 目录路径。

    异常
    ------
    FileNotFoundError
        本地找不到文件且不允许下载时抛出。
    """
    # data_root 为相对路径时，以"模块上级目录"（即工程根目录）为基准解析，
    # 这样无论从哪个工作目录启动 train.py 都能定位到工程内既有数据
    if not os.path.isabs(root):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        root = os.path.join(project_root, root)
    root = os.path.abspath(root)
    raw_dir = os.path.join(root, "MNIST", "raw")
    if all(os.path.isfile(os.path.join(raw_dir, f)) for f in MNIST_IDX_FILES):
        log_info(f"复用已有 MNIST IDX 文件：{raw_dir}")
        return raw_dir

    src_dir = _find_local_idx_dir(root)
    if src_dir is not None and os.path.normpath(src_dir) != os.path.normpath(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
        for name in MNIST_IDX_FILES:
            src = os.path.join(src_dir, name)
            dst = os.path.join(raw_dir, name)
            if not os.path.isfile(dst):
                shutil.copyfile(src, dst)
        log_info(f"已从 {src_dir} 复用 MNIST IDX 文件（复制到 {raw_dir}，未联网下载）")
        return raw_dir

    if allow_download:
        log_warn("本地未找到完整 MNIST IDX 文件，将交由 torchvision 联网下载")
        return raw_dir

    raise FileNotFoundError(
        f"未找到完整的 MNIST IDX 文件。已检查目录：{raw_dir} 与候选目录 {_CANDIDATE_DIRS}"
        f"（基准目录包含工程根目录与当前工作目录）。"
        f"请确认 {MNIST_IDX_FILES} 四个文件存在，或传入 allow_download=True 联网下载。"
    )


def get_mnist_loaders(
    batch_size: int = 64,
    data_root: str = "data/mnist",
    num_workers: int = 0,
    seed: int = 42,
    allow_download: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """构建 MNIST 的 train/test DataLoader（图像展平为 784 维并归一化）。

    参数
    ----
    batch_size : int
        批大小。
    data_root : str
        MNIST 数据根目录（工程内已存在的 IDX 文件位于其下）。
    num_workers : int
        DataLoader 工作进程数（Windows 下建议 0）。当 > 0 时，各 worker 通过
        `_worker_init_fn` 按 (seed, worker_id) 独立播种，保证多进程取数顺序可复现。
    seed : int
        打乱顺序所用种子（同时作为 per-worker 播种的基种子），保证可复现。
        注意：per-worker 播种依赖父进程的 `torch.initial_seed()`，故调用前需先执行
        `utils.set_seed(seed)`，该参数才会真正传导到 worker。
    allow_download : bool
        本地数据缺失时是否允许联网下载。默认 False。

    返回
    ----
    Tuple[DataLoader, DataLoader]
        (train_loader, test_loader)。训练集 shuffle=True，测试集 shuffle=False。

    可复现性前提
    ------------
    * 固定 `seed`；
    * `num_workers > 0` 时由 `_worker_init_fn` 为每个 worker 派生独立随机序列；
    * per-worker 的基种子取自父进程的 `torch.initial_seed()`，因此**必须先在父进程
      调用 `utils.set_seed(config.seed)`**，`seed` 才能通过初始种子传导到各 worker
      （`train.build_model_and_data` 已自动完成）；
    * 数据集本身不含随机数据增强（MNIST 仅做确定性归一化），故不存在跨 worker 的
      增强多样性差异问题。

    异常
    ------
    FileNotFoundError
        本地数据缺失且 allow_download=False 时抛出（附排查信息）。
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，当前 {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers 不能为负，当前 {num_workers}")
    raw_dir = ensure_mnist_files(data_root, allow_download=allow_download)
    train_ds = RawIdxMNIST(
        os.path.join(raw_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(raw_dir, "train-labels-idx1-ubyte.gz"),
    )
    test_ds = RawIdxMNIST(
        os.path.join(raw_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(raw_dir, "t10k-labels-idx1-ubyte.gz"),
    )
    # 注意：这里用常量 28*28=784 而非 train_ds[0][0].numel()，避免为了打印一行日志
    # 而触发 60000 张图的 IDX 全量解析（惰性加载的意义正在于此）
    log_info(f"MNIST 加载完成：train={len(train_ds)} 张，test={len(test_ds)} 张，"
             f"每张展平为 {28 * 28} 维")
    gen = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=gen,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, test_loader