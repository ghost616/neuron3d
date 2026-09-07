"""mnist.py -- MNIST 1k 子集加载 + adaptive_avg_pool2d 28x28->10x10（G2，data 模块）。

D14 映射：28x28 MNIST 灰度补丁经自适应平均池化到 10x10 单元网格
（= 100 个输入单元；unit id 行主序 = r * out + c），与 encoder 消费的布局
一致（cfg.n_in == 100 / n_input_cols == 10，G1/G2 网络共用）。池化语义与
torch ``adaptive_avg_pool2d`` 一致，并与 ``core.encoder.mnist_adaptive_pool``
逐帧等价 —— 本文件提供**批次向量化**实现（N 轴整块切分求均值），自测中与
encoder 的逐帧实现做交叉一致性校验（防止两处语义漂移）。

**纯 NumPy IDX 加载（无 torch 依赖）**
--------------------------------------
``load_mnist_subset`` / ``load_mnist_idx`` 使用 gzip + urllib.request 从 S3
镜像下载标准 MNIST IDX 文件并解析，**不要求 torch/torchvision**（G2 full
双臂 5 seeds 前置条件）。网络源：yann.lecun.com 主源不可达，固定使用
ossci-datasets S3 镜像（见 MNIST_SOURCE_URL；已验证可达）。

缓存语义：4 个 IDX gz 文件（train/test 各 images+labels）下载到 ``root``
（默认 DEFAULT_ROOT = data/mnist）；文件已存在且非空则**跳过下载**
（cache hit）。下载经 ``.part`` 临时文件 + 原子改名，不残留半截缓存；
缓存损坏（gzip 破损 / 魔数不符 / 维度或长度不符）抛带「删除缓存后重跑」
建议的可读 ValueError。像素 uint8 -> float64/255 归一化到 [0,1]，
标签 int64；确定性 n 样本子集 = rng(seed).choice(不放回)，与旧
torchvision 路径语义一致（相同数据顺序 + 相同 seed -> 相同子集）。

本模块顶层导入不触碰 torch。自测：``python -m hstdn.data.mnist``（离线，
含内存级 IDX 解析单测）；``python -m hstdn.data.mnist --download-smoke``
额外做一次真实小规模下载冒烟（失败仅记录原因，不阻塞）。
"""

from __future__ import annotations

import gzip
import shutil
import struct
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hstdn.core.layout import F8, I8

__all__ = ["MNIST_SIDE", "DEFAULT_SUBSET_N", "DEFAULT_ROOT",
           "MNIST_SOURCE_URL", "MNIST_IDX_FILES",
           "adaptive_pool_ranges", "pool_mnist_to_units", "MnistSubset",
           "fetch_mnist_idx", "parse_mnist_idx_file", "load_mnist_idx",
           "load_mnist_subset"]

#: MNIST 补丁边长（像素）。
MNIST_SIDE = 28
#: 默认子集规模（G2 1k）。
DEFAULT_SUBSET_N = 1000
#: MNIST IDX 数据落盘目录（相对项目根；调用时自动创建/下载）。
DEFAULT_ROOT = "data/mnist"
#: MNIST IDX 网络源（ossci-datasets S3 镜像；yann.lecun.com 主源不可达）。
MNIST_SOURCE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
#: 每个 split 的 (images 文件名, labels 文件名, 样本总数)。
MNIST_IDX_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
              60000),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
             10000),
}
_IDX_MAGIC_IMAGES = 2051
_IDX_MAGIC_LABELS = 2049
_DOWNLOAD_TIMEOUT_S = 120.0

_PASS = 0


def _ok(name: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  [PASS] {name}")


def _read_maybe_gzip(path) -> bytes:
    """按 gzip 魔数自动解压读取；否则按裸字节读取（纯 stdlib）。

    Raises:
        ValueError: gzip 流损坏（含删除缓存重下的建议）。
    """
    p = Path(path)
    with open(p, "rb") as fh:
        head = fh.read(2)
    if head != b"\x1f\x8b":
        with open(p, "rb") as fh:
            return fh.read()
    try:
        with gzip.open(p, "rb") as fh:
            return fh.read()
    except OSError as exc:  # BadGzipFile 是 OSError 子类
        raise ValueError(
            f"MNIST IDX gzip 缓存损坏 at {p}: {exc!r} —— 请删除该缓存文件"
            "后重试（将自动重新下载）"
        ) from exc


def parse_mnist_idx_file(path, *, expected_magic: int) -> np.ndarray:
    """解析单个 MNIST IDX 文件（自动识别 gz），校验魔数/维度/长度。

    IDX 格式：大端 4 字节魔数 + 4 字节样本数；images 追加 4 字节行数 +
    4 字节列数。images 魔数 2051、labels 魔数 2049（设计要求校验）。

    Args:
        path: IDX 文件路径（.gz 或裸字节均可）。
        expected_magic: 期望魔数（2051=images / 2049=labels）。

    Returns:
        uint8 ndarray：images -> (count, 28, 28)；labels -> (count,)。
        返回可写独立副本。

    Raises:
        ValueError: 文件过短 / gzip 损坏 / 魔数不符 / 维度不符 / 数据长度
            不完整 —— 消息含路径与统计数值，并建议删除缓存重下。
    """
    raw = _read_maybe_gzip(path)
    p = Path(path)

    def corrupt(detail: str) -> "NoReturn":
        raise ValueError(
            f"MNIST IDX 缓存损坏 at {p}: {detail}（期望魔数 "
            f"{expected_magic}）。请删除该缓存文件后重试，将自动从 "
            f"{MNIST_SOURCE_URL} 重新下载"
        )

    if len(raw) < 8:
        corrupt(f"文件过短：{len(raw)} 字节 < 8 字节头")
    magic = int.from_bytes(raw[0:4], "big")
    count = int.from_bytes(raw[4:8], "big")
    if magic != expected_magic:
        corrupt(f"魔数不符：实际 {magic}，期望 {expected_magic}")
    if expected_magic == _IDX_MAGIC_IMAGES:
        if len(raw) < 16:
            corrupt("images 头不足 16 字节")
        rows = int.from_bytes(raw[8:12], "big")
        cols = int.from_bytes(raw[12:16], "big")
        if (rows, cols) != (MNIST_SIDE, MNIST_SIDE):
            corrupt(f"图像维度不符：实际 {rows}x{cols}，期望 "
                    f"{MNIST_SIDE}x{MNIST_SIDE}")
        n_px = count * rows * cols
        if len(raw) < 16 + n_px:
            corrupt(f"像素数据不完整：期望 {n_px} 字节，实际仅 "
                    f"{len(raw) - 16} 字节（count={count}）")
        arr = np.frombuffer(raw, dtype=np.uint8, offset=16,
                            count=n_px).reshape(count, rows, cols)
    else:  # labels（无维度字段，标量每样本 1 字节）
        if len(raw) < 8 + count:
            corrupt(f"标签数据不完整：期望 {count} 字节，实际仅 "
                    f"{len(raw) - 8} 字节")
        arr = np.frombuffer(raw, dtype=np.uint8, offset=8, count=count)
    return np.array(arr, copy=True)


def fetch_mnist_idx(root: str = DEFAULT_ROOT, train: bool = True,
                    *, source_url: "str | None" = None) -> tuple:
    """下载 MNIST IDX gz 文件（纯 urllib+gzip；缓存命中则跳过下载）。

    Args:
        root: 落盘目录（自动创建；文件已存在且非空即视为缓存命中）。
        train: True=下载 train 两个文件，False=下载 test 两个文件。
        source_url: 覆盖 MNIST_SOURCE_URL 的下载源（默认 None 使用模块常量；
            供离线错误分支测试模拟网络失败，不改变对外默认行为）。

    Returns:
        (images_path, labels_path) 两个绝对路径字符串。

    Raises:
        RuntimeError: 网络不可达/下载失败（消息含 URL 与重试指引）。
    """
    split = "train" if train else "test"
    f_img, f_lab, _expected = MNIST_IDX_FILES[split]
    base_url = MNIST_SOURCE_URL if source_url is None else source_url
    rdir = Path(root)
    rdir.mkdir(parents=True, exist_ok=True)
    out: list = []
    for fname in (f_img, f_lab):
        dst = rdir / fname
        if dst.exists() and dst.stat().st_size > 0:
            out.append(str(dst))
            continue
        url = base_url + fname
        part = rdir / (fname + ".part")
        try:
            with urllib.request.urlopen(url,
                                        timeout=_DOWNLOAD_TIMEOUT_S) as resp:
                with open(part, "wb") as fh:
                    shutil.copyfileobj(resp, fh)
        except Exception as exc:  # noqa: BLE001 —— 统一转为带 URL 的下载错误
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"MNIST IDX 下载失败 {url}: {exc!r} —— 请检查网络后重试；"
                f"或将离线缓存的 {fname} 手动放入 {rdir}（存在即命中缓存，"
                "不再下载）"
            ) from exc
        if part.stat().st_size == 0:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"MNIST IDX 下载为空文件：{url}")
        part.replace(dst)  # 原子改名，避免半截缓存被当作命中
        out.append(str(dst))
    return (out[0], out[1])


def load_mnist_idx(n: int = DEFAULT_SUBSET_N, *, train: bool = True,
                   root: str = DEFAULT_ROOT, seed: int = 0) -> MnistSubset:
    """纯 NumPy 加载 MNIST 的 n 样本确定性子集（IDX 下载 + 解析 + 抽样）。

    Args:
        n: 子集规模（> 0 且不超过对应数据集大小；默认 1000）。
        train: True=训练集（60000），False=测试集（10000）。
        root: 落盘目录（默认 DEFAULT_ROOT=data/mnist；缓存缺失时自动从
            MNIST_SOURCE_URL 下载，已有文件则跳过下载）。
        seed: 子集抽取随机种子（rng.choice 不放回，确定性可复现）。

    Returns:
        MnistSubset：images (n, 28, 28) f8，像素 uint8/255 归一化到 [0,1]；
        labels (n,) i8 数字 0..9。

    Raises:
        ValueError: n 非法 / 超过数据集大小；缓存损坏（魔数/维度/长度/样本
            数不一致，含删除缓存重下建议）。
        RuntimeError: 网络下载失败（含 URL 与指引）。
    """
    if int(n) <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    img_path, lab_path = fetch_mnist_idx(root=root, train=train)
    raw_img = parse_mnist_idx_file(img_path,
                                   expected_magic=_IDX_MAGIC_IMAGES)
    raw_lab = parse_mnist_idx_file(lab_path,
                                   expected_magic=_IDX_MAGIC_LABELS)
    n_img = int(raw_img.shape[0])
    n_lab = int(raw_lab.size)
    if n_img != n_lab:
        raise ValueError(
            f"MNIST IDX images/labels 样本数不一致：images {n_img} vs "
            f"labels {n_lab}（{img_path} / {lab_path}）。缓存疑似损坏："
            "请删除缓存文件后重跑以重新下载"
        )
    if int(n) > n_img:
        split = "train" if train else "test"
        raise ValueError(f"n={n} exceeds the MNIST {split} set size "
                         f"({n_img})")
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n_img, size=int(n), replace=False)
    images = raw_img[idx].astype(F8) / 255.0   # uint8 -> [0,1] float64
    labels = raw_lab[idx].astype(I8)
    return MnistSubset(images=images, labels=labels)


def load_mnist_subset(n: int = DEFAULT_SUBSET_N, *, train: bool = True,
                      root: str = DEFAULT_ROOT, seed: int = 0) -> MnistSubset:
    """加载 MNIST 的 n 样本确定性子集（**纯 NumPy 路径，无 torch 依赖**）。

    等价于 ``load_mnist_idx``：IDX gz 文件从 MNIST_SOURCE_URL（ossci S3
    镜像）下载（root 下缓存已存在则跳过）-> 解析 -> 确定性子集抽取。
    保留本入口名以兼容既有调用方（如 g2_runner 的 load_mnist_subset
    调用）。不再依赖 torch/torchvision（移除强制 ImportError 分支）。

    Args/Returns/Raises: 同 load_mnist_idx。
    """
    return load_mnist_idx(n, train=train, root=root, seed=seed)


# ---------------------------------------------------------------------------
# D14 单元映射（28x28 -> 10x10）
# ---------------------------------------------------------------------------


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
            归一化值（load_mnist_subset/load_mnist_idx 已归一化；原生
            0..255 输入请先 /255）。
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

    N 与采样方式（确定性随机子集）由 load_mnist_subset/load_mnist_idx 决定；
    本容器额外提供 D14 单元映射便捷方法（pool/flat，纯 NumPy，torch 无关）。
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


# ---------------------------------------------------------------------------
# 本地自测（全部离线；真实下载仅由 --download-smoke 触发且失败不阻塞）
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


def _craft_idx_gz(magic: int, count: int, *, rows: int = MNIST_SIDE,
                  cols: int = MNIST_SIDE, payload: bytes = b"") -> bytes:
    """构造最小 IDX gz 字节流（内存级；用于离线单测，不触网）。

    默认 payload：images 为 bytes(i % 256 for i in range(count*rows*cols))
    （含 255/0 边界值），labels 为 0..count-1。
    """
    if not payload:
        n_item = rows * cols if magic == _IDX_MAGIC_IMAGES else 1
        payload = bytes(i % 256 for i in range(count * n_item))
    header = struct.pack(">II", magic, count)
    if magic == _IDX_MAGIC_IMAGES:
        header += struct.pack(">II", rows, cols)
    return gzip.compress(header + payload)


def _expect_value_error(fn, *args, key: str, **kwargs) -> str:
    """断言 fn 抛 ValueError 且消息含 key；返回消息文本。"""
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        msg = str(exc)
        assert key in msg, f"expected '{key}' in message, got: {msg}"
        return msg
    raise AssertionError(f"{fn.__name__} must raise ValueError (key={key})")


def check_idx_parser_offline() -> None:
    """纯 NumPy IDX 解析器：内存级字节流单测（不触网）。"""
    print("mnist IDX parser offline (pure gzip+struct):")
    with tempfile.TemporaryDirectory(prefix="hstdn_idx_") as td:
        root = Path(td)
        n = 3
        p_img = root / "train-images-idx3-ubyte.gz"
        p_lab = root / "train-labels-idx1-ubyte.gz"
        p_img.write_bytes(_craft_idx_gz(_IDX_MAGIC_IMAGES, n))
        p_lab.write_bytes(_craft_idx_gz(_IDX_MAGIC_LABELS, n))
        # --- parser 级：形状 / dtype / 大端内容 ---
        im = parse_mnist_idx_file(p_img, expected_magic=_IDX_MAGIC_IMAGES)
        assert im.shape == (n, 28, 28) and im.dtype == np.uint8
        assert im[0, 0, 0] == 0 and im[0, 0, 1] == 1
        # payload = 全局连续 i % 256 循环 -> 样本 k 末像素 = (k*784+783) % 256
        assert im[2, -1, -1] == (2 * MNIST_SIDE * MNIST_SIDE + 783) % 256
        lb = parse_mnist_idx_file(p_lab, expected_magic=_IDX_MAGIC_LABELS)
        assert lb.shape == (n,) and lb.dtype == np.uint8
        assert np.array_equal(lb, np.asarray([0, 1, 2], dtype=np.uint8))
        # --- 魔数拒绝路径（含删缓存建议）---
        bad_magic = root / "bad_magic.gz"
        bad_magic.write_bytes(_craft_idx_gz(9999, 2))
        _expect_value_error(parse_mnist_idx_file, bad_magic,
                            expected_magic=_IDX_MAGIC_IMAGES, key="魔数")
        # 交叉：把 labels 文件当 images 解析也必须拒绝
        _expect_value_error(parse_mnist_idx_file, p_lab,
                            expected_magic=_IDX_MAGIC_IMAGES, key="魔数")
        # --- 维度不符 ---
        bad_dim = root / "bad_dim.gz"
        bad_dim.write_bytes(_craft_idx_gz(_IDX_MAGIC_IMAGES, 1, rows=27))
        _expect_value_error(parse_mnist_idx_file, bad_dim,
                            expected_magic=_IDX_MAGIC_IMAGES, key="维度")
        # --- 数据长度不完整 ---
        short_px = root / "short_px.gz"
        header = struct.pack(">II", _IDX_MAGIC_IMAGES, 5) + \
            struct.pack(">II", 28, 28)
        short_px.write_bytes(gzip.compress(header + bytes(16)))  # 远短于 5*784
        _expect_value_error(parse_mnist_idx_file, short_px,
                            expected_magic=_IDX_MAGIC_IMAGES, key="不完整")
        # --- 文件过短 / gzip 破损 ---
        too_short = root / "too_short.gz"
        too_short.write_bytes(gzip.compress(b"\x00\x00\x00"))
        _expect_value_error(parse_mnist_idx_file, too_short,
                            expected_magic=_IDX_MAGIC_LABELS, key="过短")
        corrupt_gz = root / "corrupt.gz"
        corrupt_gz.write_bytes(b"\x1f\x8b" + bytes(64))
        _expect_value_error(parse_mnist_idx_file, corrupt_gz,
                            expected_magic=_IDX_MAGIC_LABELS, key="删除")
    _ok("parser magic/dims/length rejection; big-endian content verified")


def check_load_idx_cached() -> None:
    """缓存命中路径端到端：归一化 / 确定性子集 / 不一致与超规模错误。"""
    print("mnist IDX load via local cache (no network):")
    with tempfile.TemporaryDirectory(prefix="hstdn_idx_") as td:
        root = str(td)
        n = 5
        Path(root, "train-images-idx3-ubyte.gz").write_bytes(
            _craft_idx_gz(_IDX_MAGIC_IMAGES, n))
        Path(root, "train-labels-idx1-ubyte.gz").write_bytes(
            _craft_idx_gz(_IDX_MAGIC_LABELS, n))
        s1 = load_mnist_idx(n=2, train=True, root=root, seed=0)
        s2 = load_mnist_idx(n=2, train=True, root=root, seed=0)
        assert np.array_equal(s1.images, s2.images)
        assert np.array_equal(s1.labels, s2.labels)
        # 归一化与抽样一致性：raw/255 == f8 图像
        raw_img = parse_mnist_idx_file(Path(root, "train-images-idx3-ubyte.gz"),
                                       expected_magic=_IDX_MAGIC_IMAGES)
        raw_lab = parse_mnist_idx_file(Path(root, "train-labels-idx1-ubyte.gz"),
                                       expected_magic=_IDX_MAGIC_LABELS)
        expect_idx = np.random.default_rng(0).choice(n, 2, replace=False)
        assert np.array_equal(s1.labels, expect_idx.astype(I8))
        assert np.allclose(s1.images, raw_img[expect_idx].astype(F8) / 255.0,
                           atol=0.0)
        assert float(s1.images.min()) >= 0.0 and float(s1.images.max()) <= 1.0
        # images/labels 样本数不一致 -> 可读 ValueError
        Path(root, "train-labels-idx1-ubyte.gz").write_bytes(
            _craft_idx_gz(_IDX_MAGIC_LABELS, n - 1))
        _expect_value_error(load_mnist_idx, 2, train=True, root=root,
                            seed=0, key="不一致")
        # 恢复 labels 后：n 超规模 -> ValueError
        Path(root, "train-labels-idx1-ubyte.gz").write_bytes(
            _craft_idx_gz(_IDX_MAGIC_LABELS, n))
        _expect_value_error(load_mnist_idx, 99, train=True, root=root,
                            seed=0, key="exceeds")
        # fetch 缓存命中：已存在文件时不再触发下载（仅返回路径）
        img_p, lab_p = fetch_mnist_idx(root=root, train=True)
        assert Path(img_p).exists() and Path(lab_p).exists()
    _ok("cache-hit load, uint8/255 normalization, deterministic subset, "
        "count-mismatch & over-size errors")


def check_fetch_error_offline() -> None:
    """下载失败分支（离线：source_url 指向本机拒绝端口，不触真实网络）。"""
    print("mnist fetch error branch (patched URL, offline):")
    with tempfile.TemporaryDirectory(prefix="hstdn_idx_") as td:
        try:
            fetch_mnist_idx(root=td, train=True,
                            source_url="http://127.0.0.1:1/mnist/")
        except RuntimeError as exc:
            msg = str(exc)
            assert "下载失败" in msg and "127.0.0.1:1" in msg, msg
            assert "train-images-idx3-ubyte.gz" in msg, msg
        else:
            raise AssertionError("fetch must raise RuntimeError on download "
                                 "failure (carrying the URL)")
    _ok("download failure -> RuntimeError carrying the source URL")


def check_loader_torch_free() -> None:
    """load_mnist_subset 纯 NumPy 可用（torch 缺失不再是障碍）。"""
    print("mnist loader torch-free (compat entry):")
    with tempfile.TemporaryDirectory(prefix="hstdn_idx_") as td:
        Path(td, "train-images-idx3-ubyte.gz").write_bytes(
            _craft_idx_gz(_IDX_MAGIC_IMAGES, 4))
        Path(td, "train-labels-idx1-ubyte.gz").write_bytes(
            _craft_idx_gz(_IDX_MAGIC_LABELS, 4))
        s = load_mnist_subset(n=3, train=True, root=td, seed=1)  # 无 torch
        assert len(s) == 3
        assert s.images.dtype == F8 and s.labels.dtype == I8
        assert s.images.shape == (3, MNIST_SIDE, MNIST_SIDE)
        assert int(s.labels.min()) >= 0 and int(s.labels.max()) < 4
        # 兼容入口与 load_mnist_idx 结果一致
        s_ref = load_mnist_idx(n=3, train=True, root=td, seed=1)
        assert np.array_equal(s.images, s_ref.images)
        assert np.array_equal(s.labels, s_ref.labels)
    _ok("load_mnist_subset runs without torch on cached IDX files")


def _download_smoke(n: int = 10) -> bool:
    """真实小规模下载冒烟（--download-smoke；失败仅记录原因不阻塞）。"""
    td = tempfile.mkdtemp(prefix="hstdn_mnist_dl_")
    try:
        s = load_mnist_subset(n=n, train=True, root=td, seed=0)
        print(f"  [SMOKE] real download OK: train n={len(s)}, images in "
              f"[{float(s.images.min()):.3f}, {float(s.images.max()):.3f}], "
              f"labels in [{int(s.labels.min())}, {int(s.labels.max())}]")
        return True
    except Exception as exc:  # noqa: BLE001 —— 冒烟失败不硬性阻塞
        print(f"  [WARN] download smoke failed (non-blocking): {exc!r}")
        return False
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main(argv=None) -> int:
    """运行 mnist 模块自测（离线全跑）；--download-smoke 追加真实下载冒烟。"""
    if argv is None:
        argv = sys.argv[1:]
    check_pool_mapping()
    check_subset_container()
    check_idx_parser_offline()
    check_load_idx_cached()
    check_fetch_error_offline()
    check_loader_torch_free()
    if "--download-smoke" in argv:
        _download_smoke()
    print(f"\nmnist selfcheck PASSED: {_PASS} assertion groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())