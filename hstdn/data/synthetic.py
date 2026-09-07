"""synthetic.py -- 合成 10 类数据生成器（G1 首跑用，data 模块，D3/D14 输入契约）。

用途
----
G1 需要一份确定性的 10 类输入流，用来在真实数据（MNIST，G2）就绪前跑通
「encoder -> 储备池 -> 统计」全链路。本模块按 D14 的 MNIST 10x10 单元布局
生成类别条件 **强度帧**：N_GRID=10 行/列 -> N_UNITS=100 个输入单元，对应
``cfg.n_in == 100 / n_input_cols == 10`` 的网络（G1 布局，含 10x10 空间
布线的 input 平面）。

类别模型（固定活动模式）
------------------------
- 类别 0..8：互不重叠的 3x3「局部高亮块」，锚点位于 3x3 格点
  {0,3,6} x {0,3,6}（共 9 x 9 = 81 个单元）；
- 类别 9：L 形边框（底行 + 右列，10 + 9 = 19 个单元）。
十个类别掩码恰好**划分** 100 个单元：每个输入单元恰好属于一个类别，类别间
激活的输入单元集两两不相交（利于分离性验证与读出调试）。

默认 ``graded=True``：掩码内强度沿块内对角线 / 边框方向由 GLYPH_HI=1.0
渐变到 GLYPH_LO=0.55（仍远大于 encoder 阈值 enc_i_thr=0.12），latency
编码因此把每个类别映射成确定性的**时空首发波前**（亮单元早发放 t=enc_t0、
暗端晚发放 t=enc_t0+0.45*enc_t_span），供 STDP 学习时间结构；
``graded=False``：掩码单元强度全为 1.0（同一时刻 t=enc_t0 发放的纯空间码）。

噪声
----
``noise``（高斯强度噪声标准差；施加后截断到 [0,1]）为批次提供类内随机性。
``seed`` 或 ``rng`` 保证可复现：noise>0 而未提供任一随机源时报 ValueError
（数据契约：随机输出必须显式播种，避免 G1 结果不可复现）。

如何喂入 encoder（输入契约）
----------------------------
帧为 (N_GRID, N_GRID) f8 强度图，**单元 id = r * N_GRID + c（行主序）**，
与 ``core.encoder.mnist_adaptive_pool``/``mnist_encode`` 的 0..99 行主序
单元 id 约定一致。G1 典型喂入方式（cfg.n_in 必须 == 100，否则抛
ValueError）：

    from hstdn.core.layout import cfg_from_mapping
    from hstdn.core.encoder import latency_encode
    from hstdn.core.network import build_network
    from hstdn.core.kernel import run_sample
    from hstdn.data.synthetic import synthetic_batch

    cfg = cfg_from_mapping({"n_in": 100, "n_input_cols": 10})  # D14 布局
    data = synthetic_batch(n_per_class=20, noise=0.02, seed=1)  # (200,10,10)
    bundle = build_network(cfg)
    for k in range(len(data)):
        buckets = latency_encode(data.frames[k].ravel(), cfg=cfg)  # id 0..99
        stats = run_sample(bundle, buckets, T=200, stdp_on=True,
                           homeo_on=True, norm_on=True)

自测：``python -m hstdn.data.synthetic``（G0 数据契约断言；退出码 0 = 通过）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from hstdn.core.encoder import latency_encode
from hstdn.core.layout import F8, I8, NetConfig, cfg_from_mapping

__all__ = [
    "N_GRID", "N_UNITS", "N_CLASSES", "GLYPH_BLOCK", "GLYPH_HI", "GLYPH_LO",
    "CLASS_GLYPHS", "class_mask", "canonical_frame", "synthetic_frame",
    "synthetic_batch", "SyntheticBatch",
]

# --- 单元网格布局（D14 MNIST 10x10；G1 运行在 cfg.n_in == 100）---
N_GRID = 10            # 单元网格边长
N_UNITS = N_GRID ** 2  # = 100 个输入单元（G1 布局的 N_IN）
N_CLASSES = 10         # 合成问题类别数
GLYPH_BLOCK = 3        # 类别 0..8 高亮块边长（3x3）
GLYPH_HI = 1.0         # 掩码核心强度（t = enc_t0 即发）
GLYPH_LO = 0.55        # 掩码远端强度（仍 >> enc_i_thr = 0.12）

_PASS = 0


def _ok(name: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  [PASS] {name}")


# ---------------------------------------------------------------------------
# 类别原型（固定活动模式）构造
# ---------------------------------------------------------------------------


def _block_anchors() -> List[tuple]:
    """3x3 高亮块锚点格点 {0,3,6} x {0,3,6}（类别 0..8，9 个锚点）。"""
    step = GLYPH_BLOCK                                     # 格点步长 = 块边长
    hi = N_GRID - GLYPH_BLOCK + 1                          # 锚点上界 = 8
    return [(r, c) for r in range(0, hi, step)
            for c in range(0, hi, step)]


def _block_glyph(anchor: tuple) -> np.ndarray:
    """单个 3x3 高亮块：强度沿 d=(dr+dc) 对角线由 GLYPH_HI 渐变到 GLYPH_LO。"""
    r0, c0 = anchor
    g = np.zeros((N_GRID, N_GRID), dtype=F8)
    dmax = 2 * (GLYPH_BLOCK - 1)
    for dr in range(GLYPH_BLOCK):
        for dc in range(GLYPH_BLOCK):
            d = dr + dc
            g[r0 + dr, c0 + dc] = GLYPH_HI - (GLYPH_HI - GLYPH_LO) * d / dmax
    return g


def _frame_glyph() -> np.ndarray:
    """L 形边框（类别 9）：底行沿列、右列沿行渐变，角点收敛到 GLYPH_LO。"""
    g = np.zeros((N_GRID, N_GRID), dtype=F8)
    last = N_GRID - 1
    for i in range(N_GRID):
        g[last, i] = GLYPH_HI - (GLYPH_HI - GLYPH_LO) * i / last
        if i < last:
            g[i, last] = GLYPH_HI - (GLYPH_HI - GLYPH_LO) * i / last
    return g


_ANCHORS = _block_anchors()
if len(_ANCHORS) != N_CLASSES - 1:
    raise AssertionError(
        "block anchor lattice must yield N_CLASSES-1 = "
        f"{N_CLASSES - 1} anchors, got {len(_ANCHORS)}"
    )
#: 冻结的类别原型帧 (N_CLASSES, N_GRID, N_GRID) f8（只读；取用时返回拷贝）。
CLASS_GLYPHS = np.stack(
    [_block_glyph(a) for a in _ANCHORS] + [_frame_glyph()], axis=0
).astype(F8)
CLASS_GLYPHS.setflags(write=False)


# ---------------------------------------------------------------------------
# 单帧生成
# ---------------------------------------------------------------------------


def _validate_class_id(class_id: int) -> int:
    """校验类别 id 域 [0, N_CLASSES)。"""
    cid = int(class_id)
    if not 0 <= cid < N_CLASSES:
        raise ValueError(f"class_id must be in [0, {N_CLASSES}), got {class_id}")
    return cid


def class_mask(class_id: int) -> np.ndarray:
    """类别激活掩码：布尔 (N_GRID, N_GRID)，True = 该单元属于该类别。

    Args:
        class_id: 类别 0..N_CLASSES-1。

    Returns:
        (N_GRID, N_GRID) bool 数组（掩码集合两两不相交且划分全部单元）。
    """
    cid = _validate_class_id(class_id)
    return CLASS_GLYPHS[cid] > 0.0


def canonical_frame(class_id: int, *, graded: bool = True) -> np.ndarray:
    """返回类别无噪原型帧（独立拷贝，不共享 CLASS_GLYPHS 内存）。

    Args:
        class_id: 类别 0..N_CLASSES-1。
        graded: True=掩码内强度渐变（默认）；False=掩码单元全置 GLYPH_HI。

    Returns:
        (N_GRID, N_GRID) f8，值域 {0} ∪ [GLYPH_LO, GLYPH_HI]。
    """
    cid = _validate_class_id(class_id)
    if graded:
        return CLASS_GLYPHS[cid].copy()
    flat = np.zeros((N_GRID, N_GRID), dtype=F8)
    flat[CLASS_GLYPHS[cid] > 0.0] = GLYPH_HI
    return flat


def _resolve_rng(seed: Optional[int], rng: Optional[np.random.Generator]
                 ) -> np.random.Generator:
    """seed/rng 二选一解析；两者同时给出视为编程错误。"""
    if seed is not None and rng is not None:
        raise ValueError("seed and rng are mutually exclusive")
    return rng if rng is not None else np.random.default_rng(seed)


def synthetic_frame(class_id: int, *, noise: float = 0.0,
                    graded: bool = True, seed: Optional[int] = None,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """生成单个（可加噪）类别帧。

    Args:
        class_id: 类别 0..N_CLASSES-1。
        noise: 高斯强度噪声标准差（截断到 [0,1]）；0 = 无噪确定性帧。
        graded: 掩码内强度是否渐变。
        seed/rng: 随机源；noise>0 时二者至少提供一个（可复现契约）。

    Returns:
        (N_GRID, N_GRID) f8 帧，值域 [0,1]。

    Raises:
        ValueError: class_id/noise 非法，或 noise>0 未提供随机源。
    """
    cid = _validate_class_id(class_id)
    if noise < 0.0:
        raise ValueError(f"noise must be >= 0, got {noise}")
    frame = canonical_frame(cid, graded=graded)
    if noise > 0.0:
        if seed is None and rng is None:
            raise ValueError(
                "noise > 0 requires explicit seed or rng (reproducibility "
                f"contract); got noise={noise} without either"
            )
        r = _resolve_rng(seed, rng)
        frame = frame + r.normal(0.0, noise, size=frame.shape)
        frame = np.clip(frame, 0.0, 1.0)
    return frame


# ---------------------------------------------------------------------------
# 批次生成
# ---------------------------------------------------------------------------


@dataclass
class SyntheticBatch:
    """合成批次容器：样本按类别分块连续排列（labels 升序，每类 n_per_class 个）。

    Attributes:
        frames: (N, N_GRID, N_GRID) f8 强度帧，值域 [0,1]。
        labels: (N,) i8 类别标签（0..N_CLASSES-1）。
        n_per_class: 每类样本数；N == N_CLASSES * n_per_class。
        noise/graded: 生成参数记录（自描述）。
    """

    frames: np.ndarray
    labels: np.ndarray
    n_per_class: int
    noise: float = 0.0
    graded: bool = True

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames)
        labels = np.asarray(self.labels)
        if frames.ndim != 3 or frames.shape[1:] != (N_GRID, N_GRID):
            raise ValueError(
                f"frames must be (N, {N_GRID}, {N_GRID}), got {frames.shape}"
            )
        if labels.shape != (frames.shape[0],):
            raise ValueError(
                f"labels must be ({frames.shape[0]},), got {labels.shape}"
            )
        if frames.dtype != F8:
            frames = frames.astype(F8)
        if labels.dtype != I8:
            labels = labels.astype(I8)
        self.frames = frames
        self.labels = labels

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, idx: int) -> tuple:
        """返回 (frame, label)（frame 为拷贝）。"""
        return self.frames[idx].copy(), int(self.labels[idx])

    def intensities(self) -> np.ndarray:
        """行主序展平视图 (N, N_UNITS)：unit id = r * N_GRID + c。

        返回 reshape 视图（与 self.frames 共享内存）；需要独立副本请 .copy()。
        """
        return self.frames.reshape(int(self.frames.shape[0]), N_UNITS)

    def encode(self, cfg=None, ids_offset: int = 0) -> List[dict]:
        """逐样本 latency 编码（encoder 输入契约的批量入口）。

        Args:
            cfg: NetConfig 或 mapping；必须满足 cfg.n_in == N_UNITS
                （G1 MNIST 10x10 布局契约，D14）。
            ids_offset: 输入全局 id 偏移（默认 0）。

        Returns:
            [latency_encode(frame_k.ravel(), ...) for each k]；每个元素为
            Dict[int, (ids, strengths)]，ids 为输入全局 id（0..N_UNITS-1 加
            ids_offset）。

        Raises:
            ValueError: cfg.n_in != N_UNITS。
        """
        conf = cfg if isinstance(cfg, NetConfig) else cfg_from_mapping(cfg)
        if conf.n_in != N_UNITS:
            raise ValueError(
                f"G1 synthetic layout requires cfg.n_in == N_UNITS == "
                f"{N_UNITS}, got cfg.n_in={conf.n_in}"
            )
        return [
            latency_encode(self.frames[k].ravel(), cfg=conf,
                           ids_offset=ids_offset)
            for k in range(len(self))
        ]


def synthetic_batch(n_per_class: int, *, noise: float = 0.0,
                    graded: bool = True, seed: Optional[int] = None,
                    rng: Optional[np.random.Generator] = None) -> SyntheticBatch:
    """生成 10 类合成批次：每类 n_per_class 个样本，按类别分块连续排列。

    Args:
        n_per_class: 每类样本数（> 0）；总样本数 N = N_CLASSES * n_per_class。
        noise: 高斯强度噪声标准差（帧截断到 [0,1]）；0 = 确定性无噪。
        graded: 掩码内强度是否渐变（见模块 docstring）。
        seed/rng: 随机源；noise>0 时必须提供其一（可复现契约）。

    Returns:
        SyntheticBatch：frames (N, N_GRID, N_GRID) f8、labels (N,) i8。

    Raises:
        ValueError: n_per_class <= 0 / noise < 0 / noise>0 未提供随机源。
    """
    if int(n_per_class) <= 0:
        raise ValueError(f"n_per_class must be > 0, got {n_per_class}")
    if noise < 0.0:
        raise ValueError(f"noise must be >= 0, got {noise}")
    labels = np.repeat(np.arange(N_CLASSES, dtype=I8), int(n_per_class))
    frames = CLASS_GLYPHS[labels]          # fancy index -> (N,10,10) 拷贝
    if not graded:
        frames = (frames > 0.0).astype(F8)
    if noise > 0.0:
        if seed is None and rng is None:
            raise ValueError(
                "noise > 0 requires explicit seed or rng (reproducibility "
                f"contract); got noise={noise} without either"
            )
        r = _resolve_rng(seed, rng)
        frames = frames + r.normal(0.0, noise, size=frames.shape)
        frames = np.clip(frames, 0.0, 1.0)
    return SyntheticBatch(frames=frames.astype(F8), labels=labels,
                          n_per_class=int(n_per_class), noise=float(noise),
                          graded=bool(graded))


# ---------------------------------------------------------------------------
# 本地自测（G0 数据契约断言；gates.py 落地后可并入 exp 门禁）
# ---------------------------------------------------------------------------


def check_layout_contract() -> None:
    """D14 布局与类别掩码契约。"""
    print("synthetic layout contract (D3/D14):")
    assert N_GRID == 10, "N_GRID must stay 10 for the MNIST 10x10 layout"
    assert N_UNITS == 100 and N_CLASSES == 10
    assert CLASS_GLYPHS.shape == (N_CLASSES, N_GRID, N_GRID)
    assert CLASS_GLYPHS.dtype == F8
    assert np.isfinite(CLASS_GLYPHS).all()
    assert float(CLASS_GLYPHS.min()) >= 0.0
    assert float(CLASS_GLYPHS.max()) <= 1.0
    masks = CLASS_GLYPHS > 0.0
    # 掩码两两不相交且恰好划分 100 单元（每个单元属于恰好一个类别）
    assert np.all(masks.sum(axis=0) == 1), \
        "class masks must partition the N_UNITS unit grid"
    for k in range(N_CLASSES):
        active = masks[k]
        n_on = int(active.sum())
        assert n_on in (GLYPH_BLOCK ** 2, N_GRID + N_GRID - 1), \
            f"class {k} activates {n_on} units"
        vmin = float(CLASS_GLYPHS[k][active].min())
        assert abs(vmin - GLYPH_LO) < 1e-9, \
            f"class {k} active minimum {vmin} != GLYPH_LO {GLYPH_LO}"
    assert CLASS_GLYPHS.flags.writeable is False, \
        "CLASS_GLYPHS must be read-only (callers get copies)"
    _ok("10x10 layout, frozen graded glyphs, masks partition the 100 units")


def check_determinism_and_noise() -> None:
    """无噪确定性 / 带噪可复现 / 值域契约。"""
    print("synthetic determinism & noise contract:")
    a = canonical_frame(3)
    assert np.array_equal(a, canonical_frame(3))
    assert np.array_equal(a, synthetic_frame(3))          # noise=0 -> canonical
    assert np.array_equal(synthetic_frame(3, seed=7),
                          synthetic_frame(3, seed=7))
    flat = canonical_frame(3, graded=False)
    assert np.all(flat[class_mask(3)] == 1.0)
    assert not np.any(flat[~class_mask(3)])
    b1 = synthetic_batch(4, noise=0.05, seed=11)
    b2 = synthetic_batch(4, noise=0.05, seed=11)
    assert np.array_equal(b1.frames, b2.frames)
    assert np.array_equal(b1.labels, b2.labels)
    b3 = synthetic_batch(4, noise=0.05, seed=12)
    assert not np.array_equal(b1.frames, b3.frames), \
        "different seeds must give different noise draws"
    try:
        synthetic_batch(4, noise=0.05)
    except ValueError:
        pass
    else:
        raise AssertionError("noise>0 without seed/rng must raise ValueError")
    assert float(b1.frames.min()) >= 0.0 and float(b1.frames.max()) <= 1.0
    assert np.isfinite(b1.frames).all()
    _ok("noise-free determinism; seeded noise reproducible; [0,1] bounds kept")


def check_encoder_feeding() -> None:
    """合成帧 -> latency encoder 的输入契约（D14/M2）。"""
    print("synthetic -> encoder feeding contract (D14/M2):")
    cfg = cfg_from_mapping({"n_in": N_UNITS, "n_input_cols": N_GRID})
    for k in range(N_CLASSES):
        frame = canonical_frame(k)
        I = frame.ravel()
        buckets = latency_encode(frame.ravel(), cfg=cfg)
        fired = (np.concatenate([v[0] for v in buckets.values()])
                 if buckets else np.zeros(0, dtype=I8))
        # 掩码单元全部发放（强度 >= GLYPH_LO > enc_i_thr），非掩码单元不发放
        must_fire = np.flatnonzero(I > cfg.enc_i_thr)
        assert fired.size == np.unique(fired).size, "fired ids must be unique"
        assert np.array_equal(np.sort(fired), must_fire), \
            f"class {k}: fired unit set != threshold-firing set"
        exp_t = np.rint(cfg.enc_t0 + (1.0 - I) * cfg.enc_t_span).astype(I8)
        for t, (ids, _s) in buckets.items():
            assert int(t) >= int(cfg.enc_t0)
            assert bool(np.all(exp_t[ids] == t)), \
                f"class {k}: bucket time violates the latency rule"
    # 批次编码入口 + n_in 不匹配保护
    data = synthetic_batch(2, graded=False, seed=0)
    enc = data.encode(cfg)
    assert len(enc) == len(data)
    for buckets, lab in zip(enc, data.labels):
        fired = (np.concatenate([v[0] for v in buckets.values()])
                 if buckets else np.zeros(0, dtype=I8))
        assert np.array_equal(np.sort(fired),
                              np.flatnonzero(class_mask(int(lab))))
    assert len(enc[0]) == 1 and cfg.enc_t0 in enc[0], \
        "graded=False -> single t=enc_t0 bucket"
    bad = cfg_from_mapping({"n_in": 1024})
    try:
        data.encode(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("encode must reject cfg.n_in != N_UNITS")
    _ok("mask -> id set -> latency buckets consistent; cfg.n_in==100 enforced")


def check_batch_smoke() -> None:
    """批次结构 / 类别分块 / 展平契约。"""
    print("synthetic batch structure:")
    d = synthetic_batch(5, noise=0.03, seed=3)
    assert len(d) == N_CLASSES * 5
    assert d.frames.shape == (50, N_GRID, N_GRID)
    assert d.labels.shape == (50,)
    assert d.frames.dtype == F8 and d.labels.dtype == I8
    assert np.array_equal(np.unique(d.labels), np.arange(N_CLASSES))
    assert np.array_equal(d.labels,
                          np.repeat(np.arange(N_CLASSES, dtype=I8), 5))
    fr, lab = d[0]
    assert fr.shape == (N_GRID, N_GRID) and 0 <= lab < N_CLASSES
    iv = d.intensities()
    assert iv.shape == (50, N_UNITS)
    assert np.array_equal(iv, d.frames.reshape(50, N_UNITS))
    # 原型类别中心互异（掩码划分保证；用于后续分离性统计的基线）
    cents = np.stack([
        np.stack(np.nonzero(class_mask(k)), axis=1).mean(axis=0)
        for k in range(N_CLASSES)
    ])
    assert cents.shape == (N_CLASSES, 2)
    assert np.unique(cents, axis=0).shape[0] == N_CLASSES, \
        "per-class activity centroids must be distinct"
    _ok("batch shape/dtype, class-block ordering, flatten view, centroids")


def main() -> int:
    """运行全部数据契约自测（退出码 0 = 通过）。"""
    check_layout_contract()
    check_determinism_and_noise()
    check_encoder_feeding()
    check_batch_smoke()
    print(f"\nsynthetic selfcheck PASSED: {_PASS} assertion groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())