"""N3D 一期原型数据层：MNIST 加载（复用工程内已有 IDX 文件，避免重复联网下载）
+ 数据集通用层（本模块第 6 轮新增：mnist / synthetic / npz / csv / json 可插拔）。

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

数据集通用层（本模块第 6 轮新增）
--------------------------------
把"数据集"从硬编码的 MNIST 变成**可插拔注册表**：

* `DATASET_SPECS` 是**唯一注册表**，每个条目是一个 `DatasetSpec`
  （`name` / `input_dim` / `num_classes` / `kind` / `norm_mean` / `norm_std` / `notes`）；
* `build_dataloaders(...)` 是**唯一通用入口**，返回 `(train_loader, test_loader, spec)`；
* 五种来源：`mnist`（**必须委派既有 `get_mnist_loaders`**，签名/行为/加载顺序/归一化
  逐字不变，MNIST 零回归）、`synthetic`（确定性生成、非线性可分、不联网）、
  `npz`（`X[M,D]` / `y[M]`）、`csv`（**末列标签**）、`json`（`{"X": [[...]], "y": [...]}`
  对象形态 + `.jsonl` 逐行样本对象）。

三条硬纪律
----------
1. **严格解析、提前报错**：json/jsonl 非法即报错，报文带**字段名**与**行号**；
   空集 / NaN 或 Inf / 列数不齐 / 维度与配置不一致 / 输入维度为 0 一律在加载阶段报错，
   绝不把坏数据留到训练循环里才炸。
2. **小集可跑完**：样本数 < `batch_size` 时用 `drop_last=False`（数据集构造路径
   一律 `drop_last=False`，与既有 MNIST 口径一致）。
3. **归一化口径**：显式给定的 `norm_mean` / `norm_std` 优先；缺省则用**训练集现场统计**
   （逐特征）并写入运行日志（`log_info`），使归一化口径可复核。
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import random
import shutil
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
    "DATASET_CHOICES",
    "DATASET_SPECS",
    "DEFAULT_SYNTHETIC_SAMPLES",
    "SYNTHETIC_SEED",
    "DatasetSpec",
    "RawIdxMNIST",
    "ArrayDataset",
    "ensure_mnist_files",
    "get_mnist_loaders",
    "build_dataloaders",
    "load_npz_arrays",
    "load_csv_arrays",
    "load_json_arrays",
    "make_synthetic_arrays",
    "dataset_spec",
    "resolve_dims",
    "normalization_stats",
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


# ======================================================================
# 数据集通用层（第 6 轮）：注册表 + 规格
# ======================================================================
@dataclass(frozen=True)
class DatasetSpec:
    """一个数据集来源的**完整规格**（数据集逻辑的唯一描述单元）。

    属性
    ----
    name : str
        注册表键，也是 `Config.dataset` 的合法取值。
    input_dim : int
        每个样本的特征维数 `D`（`> 0`）。`Config.input_dim` 的**缺省解析来源**。
    num_classes : int
        类别数 `C`（`> 0`）。`Config.output_dim` 的**缺省解析来源**。
    kind : str
        来源类型：`mnist` / `synthetic` / `npz` / `csv` / `json`。
        用于日志与"该来源是否需要外部文件"的判定。
    norm_mean : Optional[Tuple[float, ...]]
        **固定记录**的归一化均值（逐特征；长度 1 表示广播到全部特征）。
        `None` 表示该来源没有固定统计量，须由训练集现场统计。
    norm_std : Optional[Tuple[float, ...]]
        与 `norm_mean` 对应的标准差（逐特征，全部必须 `> 0`）。
    notes : str
        人类可读说明（口径 / 限制 / 陷阱）。

    关键不变量
    ----------
    * `input_dim > 0` 且 `num_classes > 0`；
    * `norm_mean` 与 `norm_std` **同时**为 `None` 或**同时**非 `None`，且长度一致；
    * `norm_std` 的每一项都 `> 0`（否则归一化会放大数值甚至产生 Inf）。
    """

    name: str
    input_dim: int
    num_classes: int
    kind: str
    norm_mean: Optional[Tuple[float, ...]]
    norm_std: Optional[Tuple[float, ...]]
    notes: str

    def describe(self) -> str:
        """人类可读摘要：`dataset=csv(D=8, C=3, kind=csv, norm=computed)`。"""
        fixed = "fixed" if self.norm_mean is not None else "computed"
        return (
            f"dataset={self.name}(D={self.input_dim}, C={self.num_classes}, "
            f"kind={self.kind}, norm={fixed})"
        )


# 合成数据集的默认规模（`--num-samples 0` 即取该值）与**确定性生成种子**。
# 生成过程完全不消耗全局 RNG（用独立的 np.random.default_rng(SYNTHETIC_SEED)），
# 故 synthetic 数据集与 `Config.seed` 无关：同 `--num-samples` 永远得到同一份数据。
DEFAULT_SYNTHETIC_SAMPLES: int = 4000
SYNTHETIC_SEED: int = 20261002
# synthetic 的类别数：两月牙的 4 个弧段 x 2 个支别 = **8 个部件**，每个部件一个类别。
# 口径：类别数必须**恰好**等于生成器产出的部件数 —— 多写空类会让"配置类别数"与
# "数据实际类别数"脱钩（`--output-dim` 与数据的对应关系会被稀释）。
SYNTHETIC_NUM_CLASSES: int = 8
# 两月牙的"月牙厚度"（无量纲，作用于半圆半径 1 的弧）。
SYNTHETIC_MOON_THICKNESS: float = 0.30


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        input_dim=784,
        num_classes=10,
        kind="mnist",
        # 固定记录 = MNIST 全局统计量（与 `RawIdxMNIST` 内联常量同源；两处必须一致）。
        norm_mean=(MNIST_MEAN,),
        norm_std=(MNIST_STD,),
        notes=(
            "MNIST IDX（工程内 data/mnist，不联网）；归一化固定为全局 mean/std，"
            "必须委派 get_mnist_loaders，行为与加载顺序逐字不变（零回归锚点）"
        ),
    ),
    "synthetic": DatasetSpec(
        name="synthetic",
        input_dim=64,
        num_classes=SYNTHETIC_NUM_CLASSES,
        kind="synthetic",
        # **确定性统计**：生成器用独立 Generator 播种，故"现场统计"本身也是确定性的
        # （同 --num-samples 同 seed 恒等），无需在此写死浮点常量。
        norm_mean=None,
        norm_std=None,
        notes=(
            "确定性合成数据（两月牙 + 弧段类别，非线性可分、不联网、不消耗全局 RNG）；"
            "归一化统计量由训练集现场统计（确定性）"
        ),
    ),
    "npz": DatasetSpec(
        name="npz",
        input_dim=0,
        num_classes=0,
        kind="npz",
        norm_mean=None,
        norm_std=None,
        notes=(
            "numpy .npz：数组 X[M, D] 与 y[M]（键名 X/y，允许小写）；"
            "D 与 C 由文件现场解析（注册表占位 0 表示\"由数据决定\"）"
        ),
    ),
    "csv": DatasetSpec(
        name="csv",
        input_dim=0,
        num_classes=0,
        kind="csv",
        norm_mean=None,
        norm_std=None,
        notes=(
            "CSV：首行表头可选（非数值首行自动视为表头，数值首行自动视为数据行），"
            "**末列 = 标签**，其余列 = 特征；D 与 C 由文件现场解析"
        ),
    ),
    "json": DatasetSpec(
        name="json",
        input_dim=0,
        num_classes=0,
        kind="json",
        norm_mean=None,
        norm_std=None,
        notes=(
            "JSON：{\"X\": [[...]], \"y\": [...]} 对象形态（整文件）；"
            ".jsonl 逐行样本对象 {\"X\": [...], \"y\": n}（一行一条，报文带行号）；"
            "D 与 C 由文件现场解析"
        ),
    ),
}

# 注册表键的稳定顺序（CLI choices 与日志用）
DATASET_CHOICES: Tuple[str, ...] = (
    "mnist",
    "synthetic",
    "npz",
    "csv",
    "json",
)


def dataset_spec(name: str) -> DatasetSpec:
    """按名字取 `DatasetSpec`（唯一查表点；未注册的名字立即报错）。

    参数
    ----
    name : str
        注册表键（`DATASET_CHOICES` 之一）。

    返回
    ----
    DatasetSpec
        对应规格。

    异常
    ------
    ValueError
        名字未注册时抛出（报文列出全部合法取值）。
    """
    key = str(name)
    if key not in DATASET_SPECS:
        raise ValueError(
            f"未注册的数据集 name={key!r}；已注册：{DATASET_CHOICES}"
        )
    return DATASET_SPECS[key]


def resolve_dims(
    name: str, input_dim: Optional[int], output_dim: Optional[int]
) -> Tuple[int, int, str]:
    """**数据集规格的单一解析点**：把 `input_dim` / `output_dim` 的生效值定下来。

    语义（仿 `Config._resolve_fc_width` 的"单一解析点"做法）
    ------------------------------------------------------
    * `input_dim is None` -> 取注册表 `spec.input_dim`（`0` 是**占位值**，表示
      "由数据文件现场决定"，此时必须显式给出 `--input-dim`，否则报错）；
    * `output_dim is None` -> 同上，取 `spec.num_classes`；
    * **显式值优先**，但立即做一致性校验：显式值与注册表声明值**不相等**时报错
      （拒绝"配置说 784、数据只有 64 列"这类静默不一致）。

    参数
    ----
    name : str
        数据集名。
    input_dim : Optional[int]
        显式输入维数（`None` = 未提供，取规格）。
    output_dim : Optional[int]
        显式类别数（`None` = 未提供，取规格）。

    返回
    ----
    Tuple[int, int, str]
        `(input_dim, output_dim, 来源说明)`；来源说明写入运行日志。

    异常
    ------
    ValueError
        名字未注册 / 显式值为非正 / 显式值与注册表声明值冲突 / 规格占位值为 0 时抛出。
    """
    spec = dataset_spec(name)
    explicit_in = input_dim is not None
    explicit_out = output_dim is not None
    if explicit_in and int(input_dim) <= 0:
        raise ValueError(
            f"input_dim 必须 > 0，当前 input_dim={input_dim}（dataset={spec.name!r}）"
        )
    if explicit_out and int(output_dim) <= 0:
        raise ValueError(
            f"output_dim 必须 > 0，当前 output_dim={output_dim}（dataset={spec.name!r}）"
        )
    if not explicit_in:
        if int(spec.input_dim) <= 0:
            raise ValueError(
                f"dataset={spec.name!r} 的 input_dim 由数据文件决定（注册表占位 "
                f"{spec.input_dim}），必须显式给出 input_dim（CLI: --input-dim D）"
            )
        got_in = int(spec.input_dim)
    else:
        # [!] 注册表维度为**占位 0** 时（`npz` / `csv` / `json` 的"由数据文件决定"），
        #     显式值是**唯一**的维度来源，故不做"与规格一致"的比对（0 是占位而非真值）；
        #     真值一致性由数据层 `_check_dims` 在**加载后**用真实列数/最大标签校验。
        if int(spec.input_dim) > 0 and int(input_dim) != int(spec.input_dim):
            raise ValueError(
                f"input_dim 与数据集规格不一致：dataset={spec.name!r} 的 "
                f"spec.input_dim={int(spec.input_dim)}，但显式给出 input_dim={int(input_dim)}"
            )
        got_in = int(input_dim)
    if not explicit_out:
        if int(spec.num_classes) <= 0:
            raise ValueError(
                f"dataset={spec.name!r} 的 num_classes 由数据文件决定（注册表占位 "
                f"{spec.num_classes}），必须显式给出 output_dim（CLI: --output-dim C）"
            )
        got_out = int(spec.num_classes)
    else:
        # [!] 注册表维度为**占位 0** 时（`npz` / `csv` / `json` 的"由数据文件决定"），
        #     显式值是**唯一**的维度来源，故不做"与规格一致"的比对（0 是占位而非真值）；
        #     真值一致性由数据层 `_check_dims` 在**加载后**用真实列数/最大标签校验。
        if int(spec.num_classes) > 0 and int(output_dim) != int(spec.num_classes):
            raise ValueError(
                f"output_dim 与数据集规格不一致：dataset={spec.name!r} 的 "
                f"spec.num_classes={int(spec.num_classes)}，但显式给出 "
                f"output_dim={int(output_dim)}"
            )
        got_out = int(output_dim)
    src = (
        f"dataset={spec.name!r}（kind={spec.kind}）："
        f"input_dim={got_in}{'(规格缺省)' if not explicit_in else '(显式)'}, "
        f"output_dim={got_out}{'(规格缺省)' if not explicit_out else '(显式)'}"
    )
    return got_in, got_out, src


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


# ======================================================================
# 数据集通用层（第 6 轮）：通用 Dataset / 各来源解析 / 通用入口
# ======================================================================
class ArrayDataset(Dataset):
    """由**内存中的 numpy 数组**驱动、逐特征归一化的数据集（通用层的基本载体）。

    参数
    ----
    X : np.ndarray
        形状 `[M, D]` 的 float32 特征矩阵。
    y : np.ndarray
        形状 `[M]` 的 int64 标签数组。
    norm_mean : np.ndarray
        形状 `[D]` 的逐特征均值（float32）。
    norm_std : np.ndarray
        形状 `[D]` 的逐特征标准差（float32，全部 `> 0`）。

    关键不变量
    ----------
    * `X.shape[0] == y.shape[0]`、`X.shape[1] == norm_mean.numel() == norm_std.numel()`；
    * `__getitem__` 返回 `(float32 [D] 张量, int64 标量标签)`；
    * 归一化在 `__getitem__` 中做（与 `RawIdxMNIST` 同口径），故 `num_workers > 0`
      时不引入任何随机性，只有主进程的 `generator` 决定取数顺序（可复现）。
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
    ) -> None:
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.int64)
        self.norm_mean = np.ascontiguousarray(norm_mean, dtype=np.float32)
        self.norm_std = np.ascontiguousarray(norm_std, dtype=np.float32)
        if self.X.ndim != 2:
            raise ValueError(f"ArrayDataset 的 X 必须是 2D [M, D]，当前 shape={self.X.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"ArrayDataset 的 y 必须是 1D [M]，当前 shape={self.y.shape}")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"ArrayDataset 的 X/y 样本数不一致：{self.X.shape[0]} vs {self.y.shape[0]}"
            )
        d = int(self.X.shape[1])
        if int(self.norm_mean.size) != d or int(self.norm_std.size) != d:
            raise ValueError(
                f"ArrayDataset 的归一化统计量维度与特征维度不一致："
                f"D={d}, mean={int(self.norm_mean.size)}, std={int(self.norm_std.size)}"
            )
        if not bool(np.all(self.norm_std > 0.0)):
            raise ValueError("ArrayDataset 的 norm_std 必须全部 > 0")

    def __len__(self) -> int:
        """返回样本数 `M`。"""
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回第 idx 个样本：`((x - mean) / std, y)`。"""
        x = (self.X[idx] - self.norm_mean) / self.norm_std
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.tensor(int(self.y[idx]), dtype=torch.long),
        )


def _as_float_matrix(X: Any, source: str) -> np.ndarray:
    """把任意嵌套序列转成 `[M, D]` 的 float32 矩阵，并做严格校验。

    参数
    ----
    X : Any
        待转换对象（通常是嵌套 list，或已是 np.ndarray）。
    source : str
        出错报文里用于定位的来源描述（文件路径 / 字段名）。

    返回
    ----
    np.ndarray
        形状 `[M, D]`、dtype `float32` 的矩阵。

    异常
    ------
    ValueError
        形状不是 2D / 空集 / 某一行长度与首行不一致 / 存在 NaN 或 Inf 时抛出。
        报文带字段名与行号（`第 i 行`，1-based，与人类阅读习惯一致）。
    """
    if isinstance(X, np.ndarray) and X.dtype == object:
        arr = X
    else:
        # 「空集」判据放在 `np.asarray` **之前**：`np.asarray([])` 会得到形状 (0,)，
        # 若不先拦下，报错会变成"必须是 2D 特征矩阵"而掩盖"空集"这一真实原因。
        if isinstance(X, (list, tuple)) and len(X) == 0:
            raise ValueError(f"{source} 为空集（样本数 M=0），无法用于训练/评估")
        arr = np.asarray(X)
    if arr.dtype == object:
        # 参差嵌套（列数不齐）会让 numpy 退化成 object 数组 -> 逐行给出具体行号
        rows = arr.tolist()
        if not rows:
            raise ValueError(f"{source} 为空集（样本数 M=0），无法用于训练/评估")
        if not isinstance(rows[0], (list, tuple)):
            raise ValueError(
                f"{source} 的每一行都必须是特征序列，当前首行是 {type(rows[0]).__name__}"
            )
        width = len(rows[0])
        for i, row in enumerate(rows):
            if not isinstance(row, (list, tuple)):
                raise ValueError(
                    f"{source} 第 {i + 1} 行不是特征序列（形态错误）：{type(row).__name__}"
                )
            if len(row) != width:
                raise ValueError(
                    f"{source} 第 {i + 1} 行的列数为 {len(row)}，与首行 {width} 不一致"
                    f"（列数不齐）"
                )
        raise ValueError(f"{source} 无法转成数值矩阵（存在非数值元素，请检查字段内容）")
    if arr.ndim != 2:
        raise ValueError(f"{source} 必须是 2D 特征矩阵 [M, D]，当前 shape={arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{source} 为空集（样本数 M=0），无法用于训练/评估")
    if arr.shape[1] == 0:
        raise ValueError(f"{source} 的特征维数为 0（D=0），无法用于训练/评估")
    f = arr.astype(np.float64)
    finite = np.isfinite(f)
    if not bool(finite.all()):
        bad = np.argwhere(~finite)
        i, j = int(bad[0][0]), int(bad[0][1])
        raise ValueError(
            f"{source} 第 {i + 1} 行第 {j + 1} 列的取值为 {float(f[i, j])!r}"
            f"（NaN/Inf 或溢出，一律拒绝）"
        )
    return f.astype(np.float32)


def _as_label_vector(y: Any, n_samples: int, source: str) -> np.ndarray:
    """把任意序列转成 `[M]` 的 int64 标签数组，并做严格校验。

    参数
    ----
    y : Any
        待转换对象（通常是 list / np.ndarray）。
    n_samples : int
        期望的样本数（与特征矩阵的行数一致）。
    source : str
        出错报文里用于定位的来源描述。

    返回
    ----
    np.ndarray
        形状 `[M]`、dtype `int64` 的标签数组。

    异常
    ------
    ValueError
        维度不是 1D / 长度与特征数不一致 / 含非整数 / 含负数 / 含 NaN/Inf 时抛出。
    """
    raw = np.asarray(y, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError(
            f"{source} 必须是 1D 标签数组 [M]，当前 shape={raw.shape}"
        )
    if raw.shape[0] != int(n_samples):
        raise ValueError(
            f"{source} 的样本数与特征数不一致：特征 {int(n_samples)} 行，标签 {raw.shape[0]} 个"
        )
    if not np.all(np.isfinite(raw)):
        bad = np.argwhere(~np.isfinite(raw))
        i = int(bad[0][0])
        raise ValueError(
            f"{source} 第 {i + 1} 个标签为 {float(raw[i])!r}"
            f"（NaN/Inf 一律拒绝，标签必须是有限整数）"
        )
    rounded = np.rint(raw)
    if not np.allclose(raw, rounded, atol=1e-9):
        bad = np.argwhere(np.abs(raw - rounded) > 1e-9)
        i = int(bad[0][0])
        raise ValueError(
            f"{source} 第 {i + 1} 个标签为 {raw[i]}，不是整数（类别标签必须是整数）"
        )
    labels = rounded.astype(np.int64)
    if np.any(labels < 0):
        i = int(np.argwhere(labels < 0)[0][0])
        raise ValueError(f"{source} 第 {i + 1} 个标签为 {labels[i]}，不能为负")
    return labels


def _check_dims(
    X: np.ndarray, y: np.ndarray, source: str, input_dim: int, output_dim: int
) -> None:
    """校验数据维度与配置维度一致（提前报错，绝不留给下游报 shape mismatch）。

    参数
    ----
    X : np.ndarray
        特征矩阵 `[M, D]`。
    y : np.ndarray
        标签数组 `[M]`。
    source : str
        出错报文里的来源描述。
    input_dim : int
        配置生效的输入维数（必须 `> 0`）。
    output_dim : int
        配置生效的类别数（必须 `> 0`）。

    异常
    ------
    ValueError
        输入维度为 0 / 与数据列数不一致 / 类别数小于数据实际最大标签 + 1 时抛出。
    """
    if int(input_dim) <= 0:
        raise ValueError(f"input_dim 必须 > 0，当前 {int(input_dim)}（来源 {source}）")
    if int(output_dim) <= 0:
        raise ValueError(f"output_dim 必须 > 0，当前 {int(output_dim)}（来源 {source}）")
    if int(X.shape[1]) != int(input_dim):
        raise ValueError(
            f"输入维度与数据不一致：配置 input_dim={int(input_dim)}，"
            f"但 {source} 只有 {int(X.shape[1])} 列特征"
        )
    need = int(y.max()) + 1 if y.size > 0 else 0
    if need > int(output_dim):
        raise ValueError(
            f"类别数与数据不一致：配置 output_dim={int(output_dim)}，"
            f"但 {source} 中出现了标签 {int(y.max())}（合法标签必须落在 "
            f"[0, output_dim-1] = [0, {int(output_dim) - 1}]；该数据至少需要 {need} 类）"
        )


def load_npz_arrays(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """从 `.npz` 读取 `X[M, D]` / `y[M]`（键名 `X`/`y`，允许小写）。

    参数
    ----
    path : str
        `.npz` 文件路径。

    返回
    ----
    Tuple[np.ndarray, np.ndarray]
        `(X float32 [M, D], y int64 [M])`。

    异常
    ------
    FileNotFoundError
        文件不存在时抛出。
    ValueError
        缺键 / 维度非法 / 含 NaN 或 Inf 时抛出（报文带键名与形状）。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"npz 文件不存在：{path}")
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.files)
        x_key = next((k for k in ("X", "x") if k in keys), None)
        y_key = next((k for k in ("y", "Y", "labels") if k in keys), None)
        if x_key is None or y_key is None:
            raise ValueError(
                f"npz 缺少必需键：需要 X 与 y，当前文件 {path} 含 {keys}"
            )
        X = _as_float_matrix(npz[x_key], f"{path}[{x_key!r}]")
        y = _as_label_vector(npz[y_key], int(X.shape[0]), f"{path}[{y_key!r}]")
    return X, y


def load_csv_arrays(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """从 `.csv` 读取特征与标签（**末列 = 标签**，其余列为特征）。

    表头口径
    --------
    首行**非数值**即视为表头（跳过）；首行是纯数值则视为数据行（不跳过）。
    每一行的列数必须与首行一致，否则报错并给出行号。

    参数
    ----
    path : str
        `.csv` 文件路径。

    返回
    ----
    Tuple[np.ndarray, np.ndarray]
        `(X float32 [M, D], y int64 [M])`，`D = 列数 - 1`。

    异常
    ------
    FileNotFoundError
        文件不存在时抛出。
    ValueError
        空文件 / 只有表头 / 列数不齐 / 末列标签非整数 / 含 NaN 或 Inf 时抛出（带行号）。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV 文件不存在：{path}")
    rows: List[List[float]] = []
    header_checked = False
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for lineno, raw_row in enumerate(reader, start=1):
            if not raw_row or all(str(c).strip() == "" for c in raw_row):
                raise ValueError(f"CSV {path} 第 {lineno} 行为空行（不允许空行）")
            if not header_checked:
                header_checked = True
                try:
                    first = [float(str(c).strip()) for c in raw_row]
                except ValueError:
                    # 首行非数值 -> 视为表头，跳过（不消耗数据行号）
                    log_info(f"CSV 表头已跳过（第 1 行）：{raw_row}")
                    continue
                rows.append(first)
                continue
            try:
                rows.append([float(str(c).strip()) for c in raw_row])
            except ValueError as exc:
                raise ValueError(
                    f"CSV {path} 第 {lineno} 行存在非数值单元格：{raw_row}（{exc}）"
                ) from exc
    if not rows:
        raise ValueError(f"CSV {path} 不含任何数据行（空集）")
    width = len(rows[0])
    if width < 2:
        raise ValueError(
            f"CSV {path} 首行只有 {width} 列，至少需要 1 列特征 + 1 列标签（末列 = 标签）"
        )
    for i, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"CSV {path} 第 {i + 1} 行的列数为 {len(row)}，与首行 {width} 不一致（列数不齐）"
            )
    mat = np.asarray(rows, dtype=np.float64)
    X = _as_float_matrix(mat[:, :-1].tolist(), f"{path}（特征列 = 前 {width - 1} 列）")
    y = _as_label_vector(mat[:, -1].tolist(), int(X.shape[0]), f"{path}（末列 = 标签）")
    return X, y


def _json_pick(obj: Dict[str, Any], names: Sequence[str], where: str) -> Any:
    """在字典里按候选键名取值（大小写两种写法都接受），缺失时报错并列出全部键。"""
    for n in names:
        if n in obj:
            return obj[n]
    raise ValueError(
        f"{where} 缺少字段：需要 {list(names)} 之一，当前对象的键为 {sorted(obj.keys())}"
    )


def load_json_arrays(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """从 `.json`（对象形态）或 `.jsonl`（逐行样本对象）读取特征与标签。

    两种形态
    --------
    * **对象形态**（`.json`）：整个文件是一个对象
      `{"X": [[...], ...], "y": [...]}`（键名大小写均接受）；
    * **逐行形态**（`.jsonl`，或按扩展名判定后逐行解析）：每行一个样本对象
      `{"X": [...], "y": n}`，报文带**行号**。

    参数
    ----
    path : str
        文件路径（`.jsonl` / `.ndjson` 走逐行形态，其余走对象形态）。

    返回
    ----
    Tuple[np.ndarray, np.ndarray]
        `(X float32 [M, D], y int64 [M])`。

    异常
    ------
    FileNotFoundError
        文件不存在时抛出。
    ValueError
        JSON 语法错误 / 缺字段 / 空集 / 列数不齐 / 标签非整数 / 含 NaN 或 Inf 时抛出。
        **报文一律带字段名（或行号）**，便于定位。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON 文件不存在：{path}")
    suffix = os.path.splitext(path)[1].lower()
    line_mode = suffix in (".jsonl", ".ndjson")
    with open(path, "r", encoding="utf-8-sig") as fh:
        text = fh.read()
    if line_mode:
        rows: List[List[float]] = []
        labels: List[float] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.strip() == "":
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL {path} 第 {lineno} 行不是合法 JSON：{exc.msg}"
                    f"（列 {exc.colno}）"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"JSONL {path} 第 {lineno} 行必须是对象（每行一个样本），"
                    f"当前是 {type(obj).__name__}"
                )
            xv = _json_pick(obj, ("X", "x", "features"), f"JSONL {path} 第 {lineno} 行")
            yv = _json_pick(obj, ("y", "Y", "label"), f"JSONL {path} 第 {lineno} 行")
            if not isinstance(xv, (list, tuple)):
                raise ValueError(
                    f"JSONL {path} 第 {lineno} 行的 'X' 字段必须是数组，"
                    f"当前是 {type(xv).__name__}"
                )
            try:
                rows.append([float(v) for v in xv])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"JSONL {path} 第 {lineno} 行的 'X' 字段含非数值元素：{xv}（{exc}）"
                ) from exc
            try:
                labels.append(float(yv))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"JSONL {path} 第 {lineno} 行的 'y' 字段不是数值：{yv!r}（{exc}）"
                ) from exc
        if not rows:
            raise ValueError(f"JSONL {path} 不含任何样本行（空集）")
        width = len(rows[0])
        for i, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(
                    f"JSONL {path} 第 {i + 1} 个样本的特征长度为 {len(row)}，"
                    f"与首个样本 {width} 不一致（列数不齐）"
                )
        X = _as_float_matrix(rows, f"JSONL {path} 的 'X' 字段")
        y = _as_label_vector(labels, int(X.shape[0]), f"JSONL {path} 的 'y' 字段")
        return X, y
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON {path} 不是合法 JSON：{exc.msg}（行 {exc.lineno}，列 {exc.colno}）"
        ) from exc
    if not isinstance(obj, dict):
        raise ValueError(
            f"JSON {path} 顶层必须是对象 {{\"X\": [[...]], \"y\": [...]}}，"
            f"当前是 {type(obj).__name__}"
        )
    xv = _json_pick(obj, ("X", "x"), f"JSON {path} 顶层对象")
    yv = _json_pick(obj, ("y", "Y"), f"JSON {path} 顶层对象")
    if not isinstance(xv, (list, tuple)):
        raise ValueError(
            f"JSON {path} 的 'X' 字段必须是二维数组（行 = 样本），"
            f"当前是 {type(xv).__name__}"
        )
    X = _as_float_matrix(xv, f"JSON {path} 的 'X' 字段")
    y = _as_label_vector(yv, int(X.shape[0]), f"JSON {path} 的 'y' 字段")
    return X, y


def make_synthetic_arrays(
    num_samples: int, dim: int, seed: int = SYNTHETIC_SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """确定性生成**非线性可分**的合成数据集（不联网、不消耗全局 RNG）。

    构造
    ----
    1. **两月牙骨架**：`u ~ U(0, pi)`，上支 `(cos u, sin u)`，下支 `(1 - cos u, -sin u)`
       —— 两支互相缠绕，**任何线性分类器都无法分开**（经典非线性可分构造）；
    2. **厚度**：沿随机方向加 `|N(0, 1)|` 幅度的抖动（服从 gamma(2) 的半径分布），
       使两类在边界处有少量重叠（真实数据的样子，不是完美可分）；
    3. **类别**：沿弧长切成 4 段 x 2 支 = 8 个部件，每个部件一个类别（`C = 8`）；
    4. **维度**：把 2D 坐标嵌入 `dim` 维 —— 旋转角由 `arange(dim)` 的余弦/正弦给出
       （无随机性）；`dim > 2` 时其余维由 `sin(k * angle)` 填充，故**信息全在低维子空间**，
       对神经元空间模型与 MLP 都是同一份数据。

    随机性口径
    ----------
    * 使用**独立** `np.random.default_rng(seed)`，**绝不触碰全局 RNG**，
      因此本函数可以在任何位置调用而不影响 `Config.seed` 决定的随机流；
    * 同 `(num_samples, dim, seed)` 恒等输出（逐位可复现）。

    参数
    ----
    num_samples : int
        总样本数 `M`（必须 `> 0`）。
    dim : int
        特征维数 `D`（必须 `> 0`）。
    seed : int
        生成器种子（默认 `SYNTHETIC_SEED`）。

    返回
    ----
    Tuple[np.ndarray, np.ndarray]
        `(X float32 [M, D], y int64 [M])`，标签取值 `[0, SYNTHETIC_NUM_CLASSES - 1]`。

    异常
    ------
    ValueError
        `num_samples <= 0` 或 `dim <= 0` 时抛出。
    """
    if int(num_samples) <= 0:
        raise ValueError(f"合成数据集的 num_samples 必须 > 0，当前 {num_samples}")
    if int(dim) <= 0:
        raise ValueError(f"合成数据集的特征维数必须 > 0，当前 {dim}")
    m = int(num_samples)
    rng = np.random.default_rng(int(seed))
    n_per = (m + 1) // 2
    half = np.pi * rng.random(n_per)
    branch = rng.integers(0, 2, size=n_per)
    base = np.empty((n_per, 2), dtype=np.float64)
    is_top = branch == 0
    base[is_top, 0] = np.cos(half[is_top])
    base[is_top, 1] = np.sin(half[is_top])
    base[~is_top, 0] = 1.0 - np.cos(half[~is_top])
    base[~is_top, 1] = -np.sin(half[~is_top])
    # 厚度：随机方向 + 半正态幅度（gamma(2) 等价形式），使两类在边界处轻微重叠
    angle = 2.0 * np.pi * rng.random(n_per)
    radius = SYNTHETIC_MOON_THICKNESS * rng.gamma(2.0, 0.5, size=n_per)
    base[:, 0] += radius * np.cos(angle)
    base[:, 1] += radius * np.sin(angle)
    # 沿弧长切 4 段（0..3），与支别组合成 8 个部件 -> 类别 0..7（余下 8/9 为空类）
    seg = np.minimum((half / np.pi * 4.0).astype(np.int64), 3)
    labels = (branch.astype(np.int64) * 4 + seg).astype(np.int64)
    # 交错切分用的自然顺序：按 (支别, 弧段位置) 排序，使"取每 5 个样本做测试集"
    # 时两个集合的类别分布一致（否则自然顺序下的前缀切分会严重偏类）
    order = np.lexsort((half, branch))
    base = base[order]
    labels = labels[order]
    # 2D -> D 维：旋转角由确定性序列给定（无随机性），保证信息落在低维子空间
    idx = np.arange(int(dim), dtype=np.float64)
    theta = idx * (2.0 * np.pi / float(max(int(dim), 1)))
    X = np.empty((base.shape[0], int(dim)), dtype=np.float64)
    X[:, 0] = base[:, 0]
    for j in range(1, int(dim)):
        X[:, j] = base[:, 1] * np.cos(theta[j]) + base[:, 0] * np.sin(theta[j])
    if base.shape[0] > m:
        X = X[:m]
        labels = labels[:m]
    if not np.all(np.isfinite(X)):
        raise ValueError("合成数据生成了 NaN/Inf（内部错误：请检查 SYNTHETIC_* 常量）")
    return X.astype(np.float32), labels.astype(np.int64)


def normalization_stats(
    X: np.ndarray,
    norm_mean: Optional[float],
    norm_std: Optional[float],
    dataset: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """决定归一化统计量（显式值优先；缺省用**训练集现场统计**）。

    口径
    ----
    * 两者都显式给出 -> 直接用（标量广播到全部特征），来源记为 `given`；
    * 都未给出 -> 逐特征现场统计训练集（`mean = X.mean(0)`、`std = X.std(0)`），
      来源记为 `computed`；某个特征方差为 0 时 `std` 取 `1.0` 并把该情况计入日志
      （否则会除以 0 产生 Inf）；
    * **只给一个** -> 报错（拒绝"半套统计量"这种静默不一致）。

    参数
    ----
    X : np.ndarray
        训练集特征矩阵 `[M, D]`（float32）。
    norm_mean : Optional[float]
        显式均值（`None` = 未给出）。
    norm_std : Optional[float]
        显式标准差（`None` = 未给出；给出时必须 `> 0`）。
    dataset : str
        数据集名（仅用于日志与报文）。

    返回
    ----
    Tuple[np.ndarray, np.ndarray, str]
        `(mean[D] float32, std[D] float32, 来源)`；来源取值 `given` / `computed`。

    异常
    ------
    ValueError
        只给一个 / 显式 `std <= 0` 时抛出。
    """
    if (norm_mean is None) != (norm_std is None):
        raise ValueError(
            f"归一化统计量必须同时给出：--norm-mean 与 --norm-std 只给了一个"
            f"（dataset={dataset!r}，norm_mean={norm_mean}, norm_std={norm_std}）"
        )
    d = int(X.shape[1])
    if norm_mean is not None:
        if not (float(norm_std) > 0.0):
            raise ValueError(
                f"--norm-std 必须 > 0，当前 {norm_std}（dataset={dataset!r}）"
            )
        return (
            np.full(d, float(norm_mean), dtype=np.float32),
            np.full(d, float(norm_std), dtype=np.float32),
            "given",
        )
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    n_zero = int(np.count_nonzero(std <= 0.0))
    if n_zero:
        std = np.where(std <= 0.0, np.float32(1.0), std).astype(np.float32)
    return mean, std, ("computed" if not n_zero else "computed(std=0 -> 1)")


def _build_loader(
    ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int
) -> DataLoader:
    """按统一口径构造 DataLoader（`drop_last=False`，小集可跑完）。

    参数
    ----
    ds : Dataset
        数据集。
    batch_size : int
        批大小（`> 0`）。
    shuffle : bool
        是否打乱（训练集 True、测试集 False；与既有 MNIST 口径一致）。
    num_workers : int
        worker 数（`> 0` 时按 (seed, worker_id) 播种）。
    seed : int
        打乱与 per-worker 播种的种子。

    返回
    ----
    DataLoader
        构造好的加载器。

    异常
    ------
    ValueError
        `batch_size <= 0` 或 `num_workers < 0` 时抛出。
    """
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size 必须为正整数，当前 {batch_size}")
    if int(num_workers) < 0:
        raise ValueError(f"num_workers 不能为负，当前 {num_workers}")
    if shuffle:
        return DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            generator=torch.Generator().manual_seed(int(seed)),
            worker_init_fn=_worker_init_fn if int(num_workers) > 0 else None,
            # drop_last=False：样本数 < batch_size 的小集也必须能跑完（不得静默丢样本）
            drop_last=False,
        )
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        drop_last=False,
    )


def _split_train_test(
    X: np.ndarray, y: np.ndarray, test_every: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把数组**交错**切成训练 / 测试两份（测试集取每 `test_every` 个样本）。

    交错而非前缀切分的原因：合成数据的自然顺序按部件聚集，前缀切分会让测试集
    **严重偏类**（实测某类只出现在测试集）。交错切分使两份的类别分布一致，
    且切分本身是 `M` 的确定性函数（无 RNG 消耗），可复现。

    参数
    ----
    X : np.ndarray
        特征矩阵 `[M, D]`。
    y : np.ndarray
        标签数组 `[M]`。
    test_every : int
        测试采样步长（`>= 2`）。

    返回
    ----
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        `(X_train, y_train, X_test, y_test)`；**两个集合都保证非空** —— 极小规模下
        （`M <= test_every`）测试集退化为最后一个样本（小集必须能跑完，不得因切分失败）。

    异常
    ------
    ValueError
        `M < 1` 或 `test_every < 2` 时抛出。
    """
    m = int(X.shape[0])
    if m < 1:
        raise ValueError(f"样本数至少为 1 才能切分训练/测试集，当前 M={m}")
    if int(test_every) < 2:
        raise ValueError(f"test_every 必须 >= 2，当前 {test_every}")
    mask = (np.arange(m) % int(test_every)) == (int(test_every) - 1)
    if not mask.any():
        # 极小集（M < test_every）：没有命中任何采样点 -> 取最后一个样本当测试集
        mask[-1] = True
    if mask.all():
        # M == 1 的退化情形：训练集与测试集都取该唯一样本（小集可跑完优先）
        mask[0] = False
    return X[~mask], y[~mask], X[mask], y[mask]


def build_dataloaders(
    dataset: str = "mnist",
    batch_size: int = 64,
    data_root: str = "data/mnist",
    dataset_path: str = "",
    num_workers: int = 0,
    seed: int = 42,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    num_samples: int = 0,
    norm_mean: Optional[float] = None,
    norm_std: Optional[float] = None,
    allow_download: bool = False,
) -> Tuple[DataLoader, DataLoader, DatasetSpec]:
    """**数据集通用入口**：按 `dataset` 分派到对应来源，返回两个 loader 与规格。

    `dataset == "mnist"` 时**必须委派既有 `get_mnist_loaders`**（签名 / 行为 /
    加载顺序 / 归一化逐字不变）—— 这是本模块的**零回归锚点**，不得在此重写 MNIST 路径。

    参数
    ----
    dataset : str
        数据集名（`DATASET_CHOICES` 之一）。
    batch_size : int
        批大小。
    data_root : str
        仅 MNIST 使用：IDX 文件根目录（`get_mnist_loaders` 的原样透传）。
    dataset_path : str
        `npz` / `csv` / `json` 的文件路径（相对路径按当前工作目录解析）。
    num_workers : int
        worker 数（`> 0` 时按 (seed, worker_id) 播种）。
    seed : int
        打乱顺序所用种子（MNIST 路径原样透传）。
    input_dim : Optional[int]
        显式输入维数（`None` = 取规格；两处都由 `resolve_dims` 解析）。
    output_dim : Optional[int]
        显式类别数（`None` = 取规格）。
    num_samples : int
        仅 `synthetic` 使用：总样本数（`0` = 取 `DEFAULT_SYNTHETIC_SAMPLES`）。
    norm_mean : Optional[float]
        显式归一化均值（`None` = 由训练集现场统计）。
    norm_std : Optional[float]
        显式归一化标准差（`None` = 由训练集现场统计；给出时必须 `> 0`）。
    allow_download : bool
        仅 MNIST 使用：本地缺失时是否允许联网下载（默认 False）。

    返回
    ----
    Tuple[DataLoader, DataLoader, DatasetSpec]
        `(train_loader, test_loader, spec)`。训练集 `shuffle=True`、测试集 `shuffle=False`，
        两者一律 `drop_last=False`（小集可跑完）。

    异常
    ------
    ValueError
        数据集未注册 / 显式维度与规格冲突 / 数据为空集 / 含 NaN 或 Inf / 列数不齐 /
        维度与配置不一致 / 输入维度为 0 / 归一化参数只给一个时抛出（报文带字段名与行号）。
    FileNotFoundError
        `npz` / `csv` / `json` 文件不存在，或 MNIST 本地文件缺失且不允许下载时抛出。
    """
    spec = dataset_spec(dataset)
    got_in, got_out, src = resolve_dims(dataset, input_dim, output_dim)
    log_info(f"数据集规格解析（单一解析点）：{src}")

    if spec.kind == "mnist":
        # ---- 零回归锚点：MNIST 一律委派既有实现，参数与返回顺序逐字不变 ----
        if norm_mean is not None or norm_std is not None:
            if not (
                float(norm_mean) == float(MNIST_MEAN)
                and float(norm_std) == float(MNIST_STD)
            ):
                raise ValueError(
                    f"MNIST 路径不支持自定义归一化：内置口径为 "
                    f"mean={MNIST_MEAN}, std={MNIST_STD}；当前 "
                    f"--norm-mean={norm_mean}, --norm-std={norm_std}。"
                    f"请移除这两个参数（或传入内置值）"
                )
        train_loader, test_loader = get_mnist_loaders(
            batch_size=int(batch_size),
            data_root=data_root,
            num_workers=int(num_workers),
            seed=int(seed),
            allow_download=bool(allow_download),
        )
        return train_loader, test_loader, spec

    if spec.kind == "synthetic":
        m = int(num_samples) if int(num_samples) > 0 else int(DEFAULT_SYNTHETIC_SAMPLES)
        X, y = make_synthetic_arrays(m, int(got_in))
        X_tr, y_tr, X_te, y_te = _split_train_test(X, y)
        path_desc = f"synthetic(M={m}, D={int(got_in)}, seed={SYNTHETIC_SEED})"
    else:
        if not dataset_path:
            raise ValueError(
                f"dataset={spec.name!r} 需要 --dataset-path 指定数据文件路径"
            )
        path = dataset_path
        if spec.kind == "npz":
            X, y = load_npz_arrays(path)
        elif spec.kind == "csv":
            X, y = load_csv_arrays(path)
        elif spec.kind == "json":
            X, y = load_json_arrays(path)
        else:  # pragma: no cover - 注册表只含上述来源
            raise ValueError(f"未实现的 dataset.kind={spec.kind!r}（dataset={spec.name!r}）")
        X_tr, y_tr, X_te, y_te = _split_train_test(X, y)
        path_desc = f"{spec.kind}={path}"

    # ---- 统一校验（提前报错：空集 / NaN / 列数不齐 / 维度不一致 / input_dim=0）----
    _check_dims(X_tr, y_tr, f"{path_desc} 的训练集", got_in, got_out)
    _check_dims(X_te, y_te, f"{path_desc} 的测试集", got_in, got_out)
    # [!] 类别数还必须覆盖**两个切分合起来**的标签：逐个切分各自校验会漏掉
    #     "某个标签只出现在训练集"（或只出现在测试集）的情形 —— 此时模型不可能
    #     预测出该标签，属"类别数与数据不一致"，必须在加载期拦下。
    if y_tr.size and y_te.size:
        _max_label = int(max(int(y_tr.max()), int(y_te.max())))
        if _max_label + 1 > int(got_out):
            raise ValueError(
                f"类别数与数据不一致：配置 output_dim={int(got_out)}，但 {path_desc} 中"
                f"出现了标签 {_max_label}（合法标签必须落在 [0, output_dim-1] = "
                f"[0, {int(got_out) - 1}]）"
            )

    mean, std, norm_src = normalization_stats(X_tr, norm_mean, norm_std, spec.name)
    train_ds = ArrayDataset(X_tr, y_tr, mean, std)
    test_ds = ArrayDataset(X_te, y_te, mean, std)
    n_classes_seen = int(max(y_tr.max(), y_te.max())) + 1
    # ---- 归一化口径写进运行日志（可复核；显式给定 vs 现场统计一目了然）----
    log_info(
        f"数据集加载完成（{spec.kind}）：train={len(train_ds)}，test={len(test_ds)}，"
        f"D={int(got_in)}，标签取值 [0, {int(max(y_tr.max(), y_te.max()))}]"
        f"（配置 output_dim={int(got_out)}，数据已见类别数={n_classes_seen}）"
    )
    log_info(
        f"归一化口径（{norm_src}）：mean[0..2]="
        f"{[round(float(v), 6) for v in mean[:3]]}，std[0..2]="
        f"{[round(float(v), 6) for v in std[:3]]}，"
        f"逐特征长度={int(mean.shape[0])}"
        f"（--norm-mean / --norm-std 显式给出时来源为 given，否则为训练集现场统计）"
    )
    train_loader = _build_loader(train_ds, batch_size, True, num_workers, seed)
    test_loader = _build_loader(test_ds, batch_size, False, num_workers, seed)
    return train_loader, test_loader, spec