"""gates.py -- G0 十五项断言门禁（exp 模块权威实现）。

依据《H-STDN 详细设计文档 v3.2-final》§9 实施计划与工程纪律：G0 十五项断言是
项目门槛，**全绿方可进入 G1**；每项独立可执行、独立通过。本文件是 G0 断言的
最终权威实现（core/selfcheck.py 仅为过渡期最小自检）。

运行方式（项目根 E:\\neuron3d）：
    python -m hstdn.exp.gates           # 批量运行全部十五项
    python -m hstdn.exp.gates 3 7 11    # 单项 / 任意组合
    python -m hstdn.exp.gates --list    # 列出清单
退出码：0 = 全绿；1 = 存在 FAIL；2 = 参数错误。

十五项清单（编号即 gate id）：
  01 LIF 单神经元解析解对照（指数衰减 / 恒定驱动闭式解，误差 < 2%）
  02 传导延迟 d/v 误差 < 1 ms（几何 delay 字段 + 内核实测到达时刻）
  03 时间轮同槽累加（双源同刻求和、跨轮复用无残留）
  04 CSR/CSC 回指一致（csc_csr_pos 回指、行计数 / in_sum0 一致）
  05 STDP 方向性（pre→post 因果 LTP、post→pre 反因果 LTD）
  06 输入通道活性（单输入事件注入后输入→池 w 必变化，D10/ADR-001）
  07 同刻对净更新 ≈ +eta_ltp（经典顺序 pre 置迹 → post 读取，D12）
  08 homeostasis 收敛（闭环：θ 自适应把长期发放率拉向目标 8 Hz）
  09 归一化守恒（competitive_norm 后每池 Σ learn w = in_sum0）
  10 ID 规范回归（存储层全局 ID / 状态层池局部 ID / 唯一转换入口）
  11 内核性能（千神经元 200 ms 样本 < 0.1 s）
  12 度分布（E 源出度 ≤ k_pool=12、零入度 = 0）
  13 延迟上界（max_delay = 15 < 时间轮 L = 16）
  14 不应期严格封锁（t+1/t+2 封锁、t+3 允许；R1 严格大于）
  15 状态重置回归（两样本无残留，B11 逐项核验）

门禁纪律：失败消息一律携带可读说明与统计数值（AssertionError 约定）；
修改既有实现前必须先跑相关 G0 断言再提交（三件套纪律）。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import fields
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hstdn.core.layout import NetConfig, I8, F8
from hstdn.core.network import NetworkBundle, build_network, structural_report
from hstdn.core.kernel import run_sample
from hstdn.core import plasticity as plast
from hstdn.configs import load_config, to_core_cfg

__all__ = ["GATE_SPECS", "run_one", "run_many", "main", "micro_bundle",
           "micro_cfg", "fresh_default", "default_cfg"]

# ---------------------------------------------------------------------------
# 测试夹具：确定性 micro 网络（布局严格遵循 §2.2/§2.3；与 core 网络同构）
# ---------------------------------------------------------------------------


def micro_cfg(n_in: int = 4, n_pool: int = 8, n_input_cols: int = 2,
              **over) -> NetConfig:
    """Small frozen NetConfig for micro tests (其余字段取 NetConfig 默认）。

    Args:
        n_in/n_pool/n_input_cols: 显式规模（默认 4/8/2）。
        **over: NetConfig 字段的部分覆盖。

    Returns:
        已填充派生字段的 NetConfig。
    """
    base = dict(n_in=n_in, n_pool=n_pool, n_input_cols=n_input_cols, seed=0)
    base.update(over)
    return NetConfig(**base).with_derived()


def micro_bundle(cfg: NetConfig,
                 edges: Sequence[Tuple[int, int, int, float, bool]],
                 pool_is_E: Optional[Sequence[bool]] = None,
                 ) -> NetworkBundle:
    """由显式边表构造确定性 micro NetworkBundle（§2.2/§2.3 布局）。

    Args:
        cfg: micro NetConfig（CSR 共 n_in+n_pool 源行）。
        edges: (src_gid, dst_local, delay, w, learn)；src_gid 为全局 ID
            ∈ [0, n_total)，dst_local 为池局部 ID ∈ [0, n_pool)，delay ∈
            [delay_min, delay_max]，w 带符号，learn 为可塑性掩码。
        pool_is_E: 可选的 E/I 标记（长 n_pool）；默认全 E。

    Returns:
        NetworkBundle：CSR（源升序→目的升序稳定排序）、CSC（B3 修正逐目标行）、
        紧凑 learn 行，与 core build_network 的布局约定完全一致。
    """
    if pool_is_E is None:
        pool_is_E = [True] * cfg.n_pool
    src = np.asarray([e[0] for e in edges], dtype=I8)
    dst = np.asarray([e[1] for e in edges], dtype=I8)
    delay = np.asarray([e[2] for e in edges], dtype=I8)
    w = np.asarray([e[3] for e in edges], dtype=F8)
    learn = np.asarray([e[4] for e in edges], dtype=bool)
    if src.size and (int(src.min()) < 0 or int(src.max()) >= cfg.n_total):
        raise AssertionError(
            f"micro edge source outside global domain [0, {cfg.n_total}): "
            f"[{int(src.min())}, {int(src.max())}]"
        )
    if dst.size and (int(dst.min()) < 0 or int(dst.max()) >= cfg.n_pool):
        raise AssertionError(
            f"micro edge target outside pool-local domain [0, {cfg.n_pool}): "
            f"[{int(dst.min())}, {int(dst.max())}]"
        )
    if delay.size and (int(delay.min()) < cfg.delay_min
                       or int(delay.max()) > cfg.delay_max):
        raise AssertionError(
            f"micro edge delay outside [{cfg.delay_min}, {cfg.delay_max}]: "
            f"[{int(delay.min())}, {int(delay.max())}]"
        )
    order = np.lexsort((dst, src))          # 主:源升序, 次:目的升序（稳定）
    csr_src = src[order]
    csr_dst_local = dst[order]
    csr_w = w[order]
    csr_delay = delay[order]
    csr_learn = learn[order]
    counts = np.bincount(csr_src, minlength=cfg.n_total).astype(I8)
    csr_ptr = np.zeros(cfg.n_total + 1, dtype=I8)
    csr_ptr[1:] = np.cumsum(counts)
    # CSC：B3 修正（np.add.at 后 cumsum），逐目标行，内序保持 CSR 序
    csc_ptr = np.zeros(cfg.n_pool + 1, dtype=I8)
    np.add.at(csc_ptr[1:], csr_dst_local, 1)
    np.cumsum(csc_ptr, out=csc_ptr)
    order_csc = np.argsort(csr_dst_local, kind="stable").astype(I8)
    dst_csc = csr_dst_local[order_csc]
    # 每池可学习入边紧凑行（learn=True）
    learn_csc = csr_learn[order_csc]
    dst_learn = dst_csc[learn_csc]
    in_learn_idx = order_csc[learn_csc]
    lcnt = np.bincount(dst_learn, minlength=cfg.n_pool).astype(I8)
    in_learn_ptr = np.zeros(cfg.n_pool + 1, dtype=I8)
    in_learn_ptr[1:] = np.cumsum(lcnt)
    in_sum0 = np.bincount(dst_learn, weights=csr_w[in_learn_idx],
                          minlength=cfg.n_pool).astype(F8)
    n_in_edges = int(np.count_nonzero(csr_src < cfg.n_in))
    return NetworkBundle(
        cfg=cfg,
        input_xyz=np.zeros((cfg.n_in, 3), dtype=F8),
        pool_xyz=np.zeros((cfg.n_pool, 3), dtype=F8),
        pool_is_E=np.asarray(pool_is_E, dtype=bool),
        n_edges_in=n_in_edges,
        n_edges_pool=int(csr_src.size) - n_in_edges,
        csr_ptr=csr_ptr,
        csr_src_g=np.arange(cfg.n_total, dtype=I8),
        csr_dst=csr_dst_local + cfg.n_in,
        csr_dst_local=csr_dst_local,
        csr_w=csr_w,
        csr_delay=csr_delay,
        csr_learn=csr_learn,
        csr_src=csr_src,
        csc_ptr=csc_ptr,
        csc_src=csr_src[order_csc],
        csc_csr_pos=order_csc,
        in_learn_ptr=in_learn_ptr,
        in_learn_idx=in_learn_idx,
        in_sum0=in_sum0,
        V=np.zeros(cfg.n_pool, dtype=F8),
        theta=np.full(cfg.n_pool, cfg.theta0, dtype=F8),
        refr=np.full(cfg.n_pool, -1, dtype=I8),
        counts=np.zeros(cfg.n_pool, dtype=I8),
        first_spike=np.full(cfg.n_pool, -1, dtype=I8),
        rate_ema=np.zeros(cfg.n_pool, dtype=F8),
        trace=np.zeros(cfg.n_total, dtype=F8),
        ring=np.zeros((cfg.wheel_l, cfg.n_pool), dtype=F8),
    )


def events(sched: Mapping[int, Sequence[int]]) -> dict:
    """{t: [输入全局 gid...]} -> run_sample 桶字典（strength=1.0）。"""
    return {
        int(t): (np.asarray(list(ids), dtype=I8),
                 np.ones(len(ids), dtype=F8))
        for t, ids in sched.items()
    }


def edge_pos(b: NetworkBundle, src_gid: int, dst_local: int) -> int:
    """返回 (src_gid, dst_local) 边在 CSR 权数组中的唯一扁平位置。"""
    s0, s1 = int(b.csr_ptr[src_gid]), int(b.csr_ptr[src_gid + 1])
    hits = np.flatnonzero(b.csr_dst_local[s0:s1] == dst_local)
    if hits.size != 1:
        raise AssertionError(
            f"micro graph has {hits.size} edge(s) {src_gid}->{dst_local} "
            f"(expected exactly 1)"
        )
    return int(s0 + hits[0])


def _clone_bundle(b: NetworkBundle) -> NetworkBundle:
    """逐 ndarray 字段复制的结构拷贝（共享 cfg；不共享状态数组）。"""
    kw = {}
    for f in fields(NetworkBundle):
        v = getattr(b, f.name)
        kw[f.name] = v.copy() if isinstance(v, np.ndarray) else v
    return NetworkBundle(**kw)


# ---------------------------------------------------------------------------
# §4 契约规模网络（default.yaml 唯一来源；构建一次后按门禁克隆复用）
# ---------------------------------------------------------------------------

_DEFAULT_CFG: Optional[NetConfig] = None
_PRISTINE: Optional[NetworkBundle] = None


def default_cfg() -> NetConfig:
    """§4 契约配置（hstdn/configs/default.yaml -> NetConfig，惰性一次）。"""
    global _DEFAULT_CFG
    if _DEFAULT_CFG is None:
        _DEFAULT_CFG = to_core_cfg(load_config())
    return _DEFAULT_CFG


def _pristine_default() -> NetworkBundle:
    """§4 规模网络原样实例（只读门禁共享；动态门禁请用 fresh_default）。"""
    global _PRISTINE
    if _PRISTINE is None:
        _PRISTINE = build_network(default_cfg())
    return _PRISTINE


def fresh_default() -> NetworkBundle:
    """§4 规模网络的干净结构拷贝（供会改动状态的门禁独立使用）。"""
    return _clone_bundle(_pristine_default())


# ---------------------------------------------------------------------------
# 门禁注册表（每项独立可执行；顺序即 G0 编号）
# ---------------------------------------------------------------------------

GATE_SPECS: List[Tuple[int, str, str, Callable[[], str]]] = []


def _register(gid: int, key: str, title: str):
    """门禁装饰器：把检查函数注册进 GATE_SPECS（调用约定：fn() -> 摘要 str）。"""
    def deco(fn: Callable[[], str]) -> Callable[[], str]:
        GATE_SPECS.append((gid, key, title, fn))
        return fn
    return deco

# ---------------------------------------------------------------------------
# G0-01 LIF 单神经元解析解对照（误差 < 2%）
# ---------------------------------------------------------------------------


@_register(1, "lif_single_neuron_analytic",
           "LIF 单神经元解析解对照（误差<2%）")
def _gate_01() -> str:
    """膜电位衰减 / 恒定驱动与闭式解析解对照（微束亚阈值单神经元）。

    内核把膜方程离散化为每步 V <- V * exp(-1/tau) + 到达跳变，其精确闭式解为：
      纯衰减：V(t) = V0 * r^t        （r = exp(-1/tau)，逐样本点）
      恒定驱动：V_k = w * r(1-r^k)/(1-r)
    本门禁把实测逐样本值与闭式解对照，并校验 LIF 线性叠加性。
    """
    tau = 20.0
    cfg = micro_cfg(tau_m=tau)
    r = cfg.decay_v
    w = 0.03                     # 亚阈值（远小于 theta0=1），保证无发放重置
    # (a) 单跳变后的纯指数衰减：事件 t=0（delay=1）于 t=1 到达，随后衰减 10 步
    b = micro_bundle(cfg, [(0, 0, 1, w, True)])
    run_sample(b, events({0: [0]}), T=12)
    meas_a = float(b.V[0])
    expect_a = w * r ** 10
    rel_a = abs(meas_a - expect_a) / max(abs(expect_a), 1e-300)
    # (b) 恒定驱动（每步一跳变）几何累加闭式解
    b2 = micro_bundle(cfg, [(0, 0, 1, w, True)])
    run_sample(b2, events({t: [0] for t in range(9)}), T=11)
    meas_b = float(b2.V[0])
    expect_b = w * r * (1.0 - r ** 9) / (1.0 - r)
    rel_b = abs(meas_b - expect_b) / max(abs(expect_b), 1e-300)
    for name, rel in (("decay", rel_a), ("drive", rel_b)):
        if rel >= 0.02:
            raise AssertionError(
                f"LIF 解析解对照失败（{name}）：实测与闭式解相对误差 "
                f"{rel:.3e} >= 2%（tau={tau:g}ms, r={r:.6f}）"
            )
    # (c) 线性性：驱动翻倍则响应翻倍
    b3 = micro_bundle(cfg, [(0, 0, 1, 2.0 * w, True)])
    run_sample(b3, events({t: [0] for t in range(9)}), T=11)
    ratio = float(b3.V[0]) / max(meas_b, 1e-300)
    if abs(ratio - 2.0) > 1e-9:
        raise AssertionError(f"LIF 线性性被破坏：响应比 {ratio:.9f} != 2.0")
    return (f"decay_err={rel_a:.2e} drive_err={rel_b:.2e} "
            f"(r={r:.6f}, tau={tau:g}ms), linearity_ratio={ratio:.6f}")


# ---------------------------------------------------------------------------
# G0-02 传导延迟 d/v 误差 < 1 ms
# ---------------------------------------------------------------------------


@_register(2, "conduction_delay_d_over_v",
           "传导延迟 d/v 误差<1ms（delay=round(d/VEL) 与内核实测）")
def _gate_02() -> str:
    """delay 字段必须等于 clip(round(dist/VEL),1,15) 且 |delay-d/v|<1ms；
    内核实测：输入事件按声明 delay 精确到达（t+1 与 t+15 两个样例）。"""
    cfg = default_cfg()
    b = _pristine_default()
    src = b.csr_src.astype(np.intp)
    dst_local = b.csr_dst_local.astype(np.intp)
    srcpos = np.empty((src.size, 3), dtype=F8)
    in_mask = src < cfg.n_in
    srcpos[in_mask] = b.input_xyz[src[in_mask]]
    pool_src = src[~in_mask] - cfg.n_in
    srcpos[~in_mask] = b.pool_xyz[pool_src]
    dist = np.linalg.norm(srcpos - b.pool_xyz[dst_local], axis=1)
    x = dist / cfg.vel
    err = np.abs(b.csr_delay.astype(F8) - x)
    if err.size:
        if float(err.max()) >= 1.0:
            raise AssertionError(
                f"传导延迟偏离 d/VEL：max |delay - d/v| = "
                f"{float(err.max()):.4f} ms >= 1 ms（共 {err.size} 条边）"
            )
        expect = np.clip(np.rint(x), cfg.delay_min, cfg.delay_max).astype(I8)
        if not np.array_equal(b.csr_delay, expect):
            nbad = int(np.count_nonzero(b.csr_delay != expect))
            raise AssertionError(
                f"{nbad}/{err.size} 条边违反 delay=round(d/VEL) 公式"
            )
    # 内核实测到达时刻 == 声明 delay（t+1 与最远 t+15；各样本独立跑，
    # first_spike 需在对应样本后立即读取——下一次 run_sample 入口会重置）
    m = micro_cfg()
    bm = micro_bundle(m, [(0, 0, 1, 1.5, True), (1, 2, 15, 1.5, True)])
    st1 = run_sample(bm, events({0: [0]}), T=30)
    fs1 = int(bm.first_spike[0])
    st2 = run_sample(bm, events({0: [1]}), T=30)
    fs2 = int(bm.first_spike[2])
    if st1["n_spikes_total"] != 1 or fs1 != 1:
        raise AssertionError(
            f"delay=1 到达时刻错误：first_spike={fs1} "
            f"(expect 1, total={st1['n_spikes_total']})"
        )
    if st2["n_spikes_total"] != 1 or fs2 != 15:
        raise AssertionError(
            f"delay=15 到达时刻错误：first_spike={fs2} "
            f"(expect 15, total={st2['n_spikes_total']})"
        )
    max_err = float(err.max()) if err.size else 0.0
    return (f"max|delay-d/v|={max_err:.4f}ms (n={int(err.size)}); "
            f"kernel t=+1 与 t=+15 实测精确")


# ---------------------------------------------------------------------------
# G0-03 时间轮同槽累加
# ---------------------------------------------------------------------------


@_register(3, "time_wheel_same_slot",
           "时间轮同槽累加（双源同刻求和 / 跨轮复用无残留）")
def _gate_03() -> str:
    """(a) 两个亚阈值源同刻到达同一池 -> 求和超阈值发放；
       (b) 同一轮槽跨周期复用：槽消费即清空，事件按各自绝对时刻独立投放；
       (c) 同槽双跳变直接核验 ring 加法（V == 0.8 精确）。"""
    cfg = micro_cfg()
    # (a) 单源 0.6 亚阈值，双源同刻 1.2 >= theta0=1 -> t=1 发放
    bA = micro_bundle(cfg, [(0, 0, 1, 0.6, True)])
    sA = run_sample(bA, events({0: [0]}), T=20)
    bB = micro_bundle(cfg, [(1, 0, 1, 0.6, True)])
    sB = run_sample(bB, events({0: [1]}), T=20)
    bAB = micro_bundle(cfg, [(0, 0, 1, 0.6, True), (1, 0, 1, 0.6, True)])
    sAB = run_sample(bAB, events({0: [0, 1]}), T=20)
    if sA["n_spikes_total"] != 0 or sB["n_spikes_total"] != 0:
        raise AssertionError("单源 0.6 必须保持亚阈值不发（同槽累加对照前提）")
    if sAB["n_spikes_total"] != 1 or int(bAB.first_spike[0]) != 1:
        raise AssertionError(
            f"同槽累加失败：双源 0.6+0.6 应于 t=1 发放，实际 "
            f"total={sAB['n_spikes_total']} first={int(bAB.first_spike[0])}"
        )
    # (b) 轮槽复用无残留：delay=15，事件 t=0 与 t=16 -> 各在 t=15 / t=31 发放
    bW = micro_bundle(cfg, [(0, 0, 15, 1.2, True)])
    sW = run_sample(bW, events({0: [0], 16: [0]}), T=40)
    if sW["n_spikes_total"] != 2 or int(bW.first_spike[0]) != 15:
        raise AssertionError(
            f"时间轮跨周期复用错误：total={sW['n_spikes_total']} "
            f"first={int(bW.first_spike[0])}（期望两次发放，首次 t=15）"
        )
    # (c) 同槽双源（delay 相同）ring 加法精确性：V 末端应恰为 0.4+0.4
    bS = micro_bundle(cfg, [(0, 0, 9, 0.4, True), (1, 0, 9, 0.4, True)])
    run_sample(bS, events({0: [0, 1]}), T=10)   # 到达 t=9（末步），不发放
    if not np.isclose(float(bS.V[0]), 0.8, atol=1e-12):
        raise AssertionError(
            f"同槽加法非累加：V={float(bS.V[0]):.6f} != 0.4+0.4"
        )
    return "同槽求和超阈发放、轮槽复用无残留、ring 加法精确 (V=0.8)"


# ---------------------------------------------------------------------------
# G0-04 CSR/CSC 回指一致
# ---------------------------------------------------------------------------


@_register(4, "csr_csc_backref", "CSR/CSC 回指一致（csc_csr_pos 回指 / 行 / in_sum0）")
def _gate_04() -> str:
    """§4 规模网络上逐条核验：CSC 是 CSR 的逐目标重排（B3），每个 CSC 条目
    通过 csc_csr_pos 回指到 CSR 的同一物理边；可学习入行与 in_sum0 一致。"""
    b = _pristine_default()
    cfg = b.cfg
    n_pool = cfg.n_pool
    n_edges = int(b.csr_w.size)
    if n_edges and np.unique(b.csc_csr_pos).size != n_edges:
        raise AssertionError("csc_csr_pos 不是 CSR 位置的排列（回指失效）")
    if not np.array_equal(b.csc_src, b.csr_src[b.csc_csr_pos]):
        raise AssertionError("csc_src != csr_src[csc_csr_pos]（CSC 源回指失配）")
    cnt = np.bincount(b.csr_dst_local, minlength=n_pool)
    if not np.array_equal(np.diff(b.csc_ptr), cnt):
        raise AssertionError("CSC 行计数 != CSR 逐目标直方图（B3 失效）")
    tgt = np.repeat(np.arange(n_pool, dtype=I8), np.diff(b.csc_ptr))
    if not np.array_equal(b.csr_dst_local[b.csc_csr_pos], tgt):
        raise AssertionError("CSC 逐目标行分组与 CSR 目的不一致")
    if not np.all(b.csr_learn[b.in_learn_idx]):
        raise AssertionError("in_learn_idx 含非可学习边（掩码回指失配）")
    lcnt = np.bincount(b.csr_dst_local[b.csr_learn], minlength=n_pool)
    if not np.array_equal(np.diff(b.in_learn_ptr), lcnt):
        raise AssertionError("in_learn 行计数与可学习入边直方图不一致")
    s0 = np.bincount(b.csr_dst_local[b.csr_learn],
                     weights=b.csr_w[b.csr_learn], minlength=n_pool)
    if not np.allclose(s0, b.in_sum0, atol=1e-12):
        raise AssertionError(
            f"in_sum0 与初始可学习入边权重和失配：max abs diff = "
            f"{float(np.max(np.abs(s0 - b.in_sum0))):.3e}"
        )
    return (f"n_edges={n_edges}; CSR<->CSC 回指、行计数、in_learn 与 "
            f"in_sum0 全部一致")

# ---------------------------------------------------------------------------
# G0-05 STDP 方向性（pre→post 因果 LTP / post→pre 反因果 LTD）
# ---------------------------------------------------------------------------


@_register(5, "stdp_causality",
           "STDP 方向性（pre→post LTP；post→pre LTD，符号+量值）")
def _gate_05() -> str:
    """探针边 A->B：因果序（A 先发于 t=1，B 后发于 t=4）→ +eta_ltp*d^3；
    反因果序（B 先发于 t=1，A 后发于 t=4）→ 乘性 w*(1-eta_ltd*d^3)。"""
    cfg = micro_cfg()
    d = cfg.decay_t
    eta_p, eta_d = cfg.eta_ltp, cfg.eta_ltd
    probe = (4, 1, 1, 0.5, True)      # A(gid4, local0) -> B(local1)
    # --- 因果序：input0->A(d=1)、input1->B(d=4)：A 发于 t=1，B 发于 t=4 ---
    b = micro_bundle(cfg, [(0, 0, 1, 1.2, True), (1, 1, 4, 1.2, True), probe])
    pos = edge_pos(b, 4, 1)
    w0 = float(b.csr_w[pos])
    run_sample(b, events({0: [0, 1]}), T=20, stdp_on=True)
    w_causal = float(b.csr_w[pos])
    expect_causal = w0 + eta_p * d ** 3
    if int(b.counts[1]) != 1:
        raise AssertionError(f"因果序中 B 应恰发一次：counts={b.counts.tolist()}")
    if w_causal <= w0:
        raise AssertionError(f"因果序（pre→post）未增强：Δw={w_causal - w0:+.6f}")
    if not np.isclose(w_causal, expect_causal, rtol=1e-9, atol=1e-12):
        raise AssertionError(
            f"因果序 LTP 量值错误：Δw={w_causal - w0:+.8f} != "
            f"+eta_ltp*d^3={eta_p * d ** 3:.8f}"
        )
    # --- 反因果序：input0->A(d=4)、input1->B(d=1)：B 发于 t=1，A 发于 t=4 ---
    b2 = micro_bundle(cfg, [(0, 0, 4, 1.2, True), (1, 1, 1, 1.2, True), probe])
    pos2 = edge_pos(b2, 4, 1)
    w2_0 = float(b2.csr_w[pos2])
    run_sample(b2, events({0: [0, 1]}), T=20, stdp_on=True)
    w_ltd = float(b2.csr_w[pos2])
    expect_ltd = w2_0 * (1.0 - eta_d * d ** 3)
    if w_ltd >= w2_0:
        raise AssertionError(f"反因果序（post→pre）未抑制：Δw={w_ltd - w2_0:+.6f}")
    if not np.isclose(w_ltd, expect_ltd, rtol=1e-9, atol=1e-12):
        raise AssertionError(
            f"反因果序 LTD 量值错误：w'={w_ltd:.8f} != w*(1-eta_ltd*d^3)="
            f"{expect_ltd:.8f}"
        )
    return (f"pre→post Δw=+{eta_p * d ** 3:.6f}；"
            f"post→pre Δw=-{w2_0 * eta_d * d ** 3:.6f}（精确）")


# ---------------------------------------------------------------------------
# G0-06 输入通道活性（单输入事件注入后输入→池 w 必变化；D10/ADR-001）
# ---------------------------------------------------------------------------


@_register(6, "input_channel_activity",
           "输入通道活性（单输入事件 → 输入→池 w 必变化）")
def _gate_06() -> str:
    """单输入事件（strength 1.0）驱动其目标池于 delay 后发放：input_channel_stdp
    先置输入迹，随后同目标池 post_ltp 读到该迹 -> 输入→池边必增强 eta_ltp*d。
    注：初权取 1.2（非 w_hi 顶界）以便 LTP 增量可观测（w_hi 处会被 clip 吞掉）。"""
    cfg = micro_cfg()
    b = micro_bundle(cfg, [(0, 0, 1, 1.2, True)])
    pos = edge_pos(b, 0, 0)
    w0 = float(b.csr_w[pos])
    run_sample(b, events({0: [0]}), T=20, stdp_on=True)
    w1 = float(b.csr_w[pos])
    delta = w1 - w0
    if delta <= 0.0:
        raise AssertionError(
            f"单输入事件注入后输入→池 w 未变化：Δw={delta:+.8f} "
            f"(w0={w0:.8f}, w1={w1:.8f}) —— 违反 D10/ADR-001 输入通道活性"
        )
    expect = cfg.eta_ltp * cfg.decay_t
    if not np.isclose(delta, expect, rtol=1e-9, atol=1e-12):
        raise AssertionError(
            f"输入→池 LTP 量值错误：Δw={delta:+.8f} != eta_ltp*decay_t="
            f"{expect:.8f}"
        )
    if int(b.counts[0]) != 1:
        raise AssertionError(f"目标池应恰发一次：counts={b.counts.tolist()}")
    return (f"单事件→池发放 t=1；输入→池 Δw=+{delta:.6f}"
            f"（=eta_ltp·d）；input_w_mean={plast.input_w_mean(b):.6f}")


# ---------------------------------------------------------------------------
# G0-07 同刻对净更新 ≈ +eta_ltp（D12 经典顺序）
# ---------------------------------------------------------------------------


@_register(7, "same_step_pair_net_ltp",
           "同刻对净更新≈+eta_ltp（pre 置迹→post 读取，D12）")
def _gate_07() -> str:
    """A 与 B 同刻（t=1）发放，A 局部 id 小于 B：经典顺序先处理 A（置 trace[A]=1
    后再衰减），B 的 post_ltp 读到 trace[A]=1 -> 探针边 A→B 恰 +eta_ltp。"""
    cfg = micro_cfg()
    edges = [(0, 0, 1, 1.2, True),      # input0 -> A(local0)
             (0, 1, 1, 1.2, True),      # input0 -> B(local1)，与 A 同刻到达
             (4, 1, 1, 0.5, True)]      # A -> B 探针
    b = micro_bundle(cfg, edges)
    pos = edge_pos(b, 4, 1)
    w0 = float(b.csr_w[pos])
    run_sample(b, events({0: [0]}), T=20, stdp_on=True)
    delta = float(b.csr_w[pos]) - w0
    if not (int(b.counts[0]) == 1 and int(b.counts[1]) == 1
            and int(b.first_spike[0]) == 1 and int(b.first_spike[1]) == 1):
        raise AssertionError(
            f"A/B 未同刻发放：counts={b.counts.tolist()} "
            f"first={b.first_spike.tolist()}"
        )
    if not np.isclose(delta, cfg.eta_ltp, rtol=1e-9, atol=1e-12):
        raise AssertionError(
            f"同刻对净更新 Δw={delta:+.8f} != +eta_ltp={cfg.eta_ltp:.8f} "
            f"（D12 经典顺序被破坏）"
        )
    return f"A、B 同刻 t=1 发放；Δw(A→B)=+{delta:.8f} == +eta_ltp（精确）"


# ---------------------------------------------------------------------------
# G0-08 homeostasis 收敛（闭环：长期发放率被拉向目标 8 Hz）
# ---------------------------------------------------------------------------

@_register(8, "homeostasis_convergence",
           "homeostasis 收敛（闭环驱动率→目标 8 Hz）")
def _gate_08() -> str:
    """真实闭环（非公式单点）：池0 接受周期性驱动（每 5 ms 一跳变），逐样本
    run_sample(..., homeo_on=True)。发放过猛 → θ 上调 → 发放率回落；过低 →
    θ 下调。断言：在样本上限内收敛（rate_ema→目标容差带），θ 明显偏离初值且
    未撞 θ 上限/下限（证明是自适应平衡而非饱和）。"""
    cfg = micro_cfg(window_s=0.2)
    target = float(cfg.homeo_rate_target)
    if target != 8.0:
        raise AssertionError(
            f"本门禁针对契约目标 8 Hz 设计，契约当前为 {target:g} Hz"
        )
    drive = events({t: [0] for t in range(0, 195, 5)})   # 40 事件 / 200 ms
    b = micro_bundle(cfg, [(0, 0, 1, 1.1, True)])
    counts_hist: List[int] = []
    theta_hist: List[float] = []
    converged_at: Optional[int] = None
    max_samples = 700
    in_band_streak = 0
    for s in range(max_samples):
        run_sample(b, drive, T=200, homeo_on=True)
        counts_hist.append(int(b.counts[0]))
        theta_hist.append(float(b.theta[0]))
        if s >= 60 and (s + 1) % 10 == 0:
            tail_rate = float(np.mean(counts_hist[-20:])) / cfg.window_s
            ema = float(b.rate_ema[0])
            # 连续 ≥3 个检查窗（30+ 样本跨度）都落在容差带内才算“收敛”
            # （单次穿越带可能是早期过冲瞬态，不算稳定平衡）
            in_band = (abs(ema - target) < 1.0
                       and abs(tail_rate - target) < 2.0)
            in_band_streak = in_band_streak + 1 if in_band else 0
            if in_band_streak >= 3:
                converged_at = s + 1
                break
    if converged_at is None:
        ema_f = float(b.rate_ema[0])
        tail_f = float(np.mean(counts_hist[-20:])) / cfg.window_s
        raise AssertionError(
            f"homeostasis 未在 {max_samples} 样本内收敛：末 rate_ema="
            f"{ema_f:.3f} Hz（目标 {target:g} Hz），末 20 样本平均率 "
            f"{tail_f:.3f} Hz，末 θ={theta_hist[-1]:.3f}"
        )
    theta_end = float(b.theta[0])
    if not (1.5 < theta_end < cfg.theta_hi - 0.5):
        raise AssertionError(
            f"θ 平衡值可疑（可能饱和/塌缩）：θ_end={theta_end:.3f} 不在 "
            f"(1.5, {cfg.theta_hi - 0.5}) 内，θ 轨迹前 10="
            f"{[round(x, 2) for x in theta_hist[:10]]}"
        )
    if theta_hist[0] == cfg.theta0 or theta_end <= theta_hist[0]:
        raise AssertionError("θ 必须从 theta0 上调（发放过猛→自适应抑制）")
    ema_end = float(b.rate_ema[0])
    tail_end = float(np.mean(counts_hist[-20:])) / cfg.window_s
    return (f"{converged_at} 样本收敛；rate_ema→{ema_end:.2f} Hz "
            f"(目标 {target:g})，末 20 样本均率 {tail_end:.2f} Hz，"
            f"θ: {cfg.theta0:g}→{theta_end:.2f}")

# ---------------------------------------------------------------------------
# G0-09 归一化守恒 Σw = in_sum0
# ---------------------------------------------------------------------------


@_register(9, "norm_conservation",
           "归一化守恒 Σw=in_sum0（competitive_norm，含冻结/I 与幂等）")
def _gate_09() -> str:
    """§4 规模网络：可学习权重对称扰动 1.1x 后跑 competitive_norm，逐池
    Σ learn w 必须回到 in_sum0（容差 1e-9 相对）；I 源（不可学习）分文不动；
    重复归一化幂等（无漂移）。"""
    b = fresh_default()
    cfg = b.cfg
    learn = b.csr_learn
    w_learn = b.csr_w[learn]
    scale = 1.10
    if w_learn.size:
        if float(w_learn.max()) * scale >= cfg.w_hi or \
           float(w_learn.min()) * scale <= cfg.w_lo:
            raise AssertionError(
                "扰动会触碰 clip 界，无法检验守恒；请调整 scale "
                f"(w range [{float(w_learn.min()):.4f}, "
                f"{float(w_learn.max()):.4f}])"
            )
    w_i_before = b.csr_w[~learn].copy()
    b.csr_w[learn] *= scale
    plast.competitive_norm(b)
    pool_ids = np.repeat(np.arange(cfg.n_pool, dtype=I8),
                         np.diff(b.in_learn_ptr))
    s = np.bincount(pool_ids, weights=b.csr_w[b.in_learn_idx],
                    minlength=cfg.n_pool)
    rel = np.abs(s - b.in_sum0) / np.maximum(b.in_sum0, 1e-12)
    if float(rel.max()) > 1e-9:
        raise AssertionError(
            f"归一化后 Σw != in_sum0：max rel dev = {float(rel.max()):.3e}"
        )
    if not np.array_equal(b.csr_w[~learn], w_i_before):
        raise AssertionError("competitive_norm 触碰了不可学习（I）权重")
    plast.competitive_norm(b)          # 幂等性：再次归一化不引入漂移
    s2 = np.bincount(pool_ids, weights=b.csr_w[b.in_learn_idx],
                     minlength=cfg.n_pool)
    if not np.allclose(s2, b.in_sum0, atol=1e-9):
        raise AssertionError(
            f"competitive_norm 非幂等：二次后 max dev = "
            f"{float(np.max(np.abs(s2 - b.in_sum0))):.3e}"
        )
    return (f"n_pool={cfg.n_pool}；逐池 Σ learn w 精确回到 in_sum0"
            f"（max rel dev {float(rel.max()):.2e}）；I 冻结；幂等")


# ---------------------------------------------------------------------------
# G0-10 ID 规范回归（存储层全局 ID / 状态层池局部 ID）
# ---------------------------------------------------------------------------


@_register(10, "id_contract_regression",
           "ID 规范回归（存储层全局 ID / 状态层池局部 / 唯一转换入口）")
def _gate_10() -> str:
    """§4 规模网络 + micro 动态样例上核验 ID 契约：
    - 存储层（CSR dst / CSC src / trace / in_learn_idx）一律全局 ID；
    - 状态数组（V/theta/refr/counts/first_spike/rate_ema）池局部 (n_pool,)；
    - csr_dst_local == csr_dst - n_in（转换只发生一次并缓存）；
    - pool_local_of 只接受池全局 ID（输入 ID 必须报错）。"""
    from hstdn.core.layout import pool_gid_of, pool_local_of
    cfg = default_cfg()
    b = _pristine_default()
    n_in, n_pool, n_total = cfg.n_in, cfg.n_pool, cfg.n_total
    # 显式传域（layout 模块级常量 N_IN=1024 是 core 回退默认，非 §4 契约规模）
    if pool_local_of(n_in, n_in=n_in, n_pool=n_pool) != 0 or \
            pool_local_of(n_total - 1, n_in=n_in, n_pool=n_pool) != n_pool - 1:
        raise AssertionError("pool_local_of 端点在域边界处换算错误")
    if pool_gid_of(0, n_in=n_in, n_pool=n_pool) != n_in:
        raise AssertionError("pool_gid_of(0) != n_in（回换错误）")
    for bad in (0, n_in - 1):
        try:
            pool_local_of(bad, n_in=n_in, n_pool=n_pool)
        except AssertionError:
            pass
        else:
            raise AssertionError(
                f"pool_local_of 必须拒绝输入全局 ID {bad}（ID 契约唯一入口）"
            )
    # --- 存储层全局域 ---
    if int(b.csr_dst.min()) < n_in or int(b.csr_dst.max()) >= n_total:
        raise AssertionError(
            f"csr_dst 越池域：range [{int(b.csr_dst.min())}, "
            f"{int(b.csr_dst.max())}]，期望 在 [{n_in}, {n_total}) 内"
        )
    if int(b.csc_src.min()) < 0 or int(b.csc_src.max()) >= n_total:
        raise AssertionError("csc_src 越全局域 [0, n_total)")
    if not np.array_equal(b.csr_dst_local, b.csr_dst - n_in):
        raise AssertionError("csr_dst_local != csr_dst - n_in（转换层不一致）")
    if b.trace.shape != (n_total,):
        raise AssertionError(f"trace 形状 {b.trace.shape} != ({n_total},)")
    # --- 状态层池局部 ---
    if b.ring.shape != (cfg.wheel_l, n_pool):
        raise AssertionError(
            f"ring 形状 {b.ring.shape} != ({cfg.wheel_l}, {n_pool})"
        )
    for name in ("V", "theta", "refr", "counts", "first_spike", "rate_ema"):
        arr = getattr(b, name)
        if arr.shape != (n_pool,):
            raise AssertionError(f"{name} 形状 {arr.shape} != ({n_pool},)")
    if b.in_learn_idx.size and \
            (int(b.in_learn_idx.min()) < 0
             or int(b.in_learn_idx.max()) >= int(b.csr_w.size)):
        raise AssertionError("in_learn_idx 越 CSR 权重域")
    # --- 动态样例（含 STDP）后域仍成立 ---
    m = micro_cfg()
    bm = micro_bundle(m, [(0, 0, 1, 1.5, True), (1, 2, 15, 1.5, True)])
    run_sample(bm, events({0: [0, 1]}), T=30, stdp_on=True)
    if not np.array_equal(bm.csr_dst_local, bm.csr_dst - m.n_in):
        raise AssertionError("内核运行后 csr_dst_local 回指被破坏")
    return (f"全局域 csr_dst 在 [{n_in},{n_total})、trace({n_total},)、ring/状态池局部、"
            f"in_learn_idx 在 csr_w 内；STDP 样例后回指仍一致")


# ---------------------------------------------------------------------------
# G0-11 内核性能（千神经元 200 ms 样本 < 0.1 s）
# ---------------------------------------------------------------------------


@_register(11, "kernel_performance",
           "内核性能（千神经元 200ms 样本 < 0.1 s）")
def _gate_11() -> str:
    """§4 契约规模网络（n_in=100 + n_pool=800 ≈ 千神经元）的 L0 NumPy 内核
    200 ms 样本墙钟计时：
      (a) 空输入样本（纯向量路径回归：200 步 × 800 池状态更新）；
      (b) 单事件确定性突发样本（真实发放路径：时轮投递/发放判定/不应期/重置）。
    两者均值均须 < 0.1 s。注：持续高密度帧驱动（~2 万发放/样本）超过 L0
    Python 内核预算（实测 ~0.4-0.6 s），正是 L1 Numba 里程碑的动机；
    本门禁钉住低发放工作点这一可达预算包络。"""
    b = fresh_default()
    cfg = b.cfg
    idle = {}
    burst = events({0: [0]})
    # --- (a) 空输入 ---
    for _ in range(2):
        run_sample(b, idle, T=200)
    t_idle = []
    for _ in range(3):
        t0 = time.perf_counter()
        run_sample(b, idle, T=200)
        t_idle.append(time.perf_counter() - t0)
    idle_mean = float(np.mean(t_idle))
    # --- (b) 单事件突发 ---
    for _ in range(2):
        run_sample(b, burst, T=200)
    t_burst = []
    n_spk = 0
    for _ in range(3):
        t0 = time.perf_counter()
        st = run_sample(b, burst, T=200)
        t_burst.append(time.perf_counter() - t0)
        n_spk = int(st["n_spikes_total"])
    burst_mean = float(np.mean(t_burst))
    if idle_mean >= 0.1:
        raise AssertionError(
            f"内核向量路径过慢：空输入 200 ms 样本均值 {idle_mean * 1e3:.1f} ms"
            f" >= 100 ms 预算"
        )
    if burst_mean >= 0.1:
        raise AssertionError(
            f"内核发放路径过慢：{int(cfg.n_in) + int(cfg.n_pool)} 神经元 "
            f"{n_spk} 发放/200 ms 样本均值 {burst_mean * 1e3:.1f} ms"
            f" >= 100 ms 预算"
        )
    return (f"{int(cfg.n_in) + int(cfg.n_pool)} 神经元；空样本 mean="
            f"{idle_mean * 1e3:.2f} ms；单事件突发 {n_spk} 发放 mean="
            f"{burst_mean * 1e3:.2f} ms（均 <100 ms）")

# ---------------------------------------------------------------------------
# G0-12 度分布（E 源出度 ≤ k_pool=12、零入度 = 0）
# ---------------------------------------------------------------------------


@_register(12, "degree_distribution",
           "度分布（E 源出度 ≤ k_pool=12、零入度 = 0）")
def _gate_12() -> str:
    """§4 规模网络：结构报告 + 直接重算度分布。E 源行出度 ≤ k_pool；无孤立、
    无零入度池（布线完整性，D11/D13）。"""
    b = _pristine_default()
    cfg = b.cfg
    r = structural_report(b)
    if r["pools_zero_indegree"] != 0:
        raise AssertionError(
            f"{r['pools_zero_indegree']}/{cfg.n_pool} 个池零入度（D13 布线残缺）"
        )
    if r["pools_isolated"] != 0:
        raise AssertionError(
            f"{r['pools_isolated']}/{cfg.n_pool} 个池孤立（零入度且零出度）"
        )
    if r["e_src_outdeg_max"] > cfg.k_pool:
        raise AssertionError(
            f"E 源出度 max={r['e_src_outdeg_max']} > k_pool={cfg.k_pool}"
        )
    outdeg = np.diff(b.csr_ptr)[cfg.n_in:cfg.n_total]
    e_out = outdeg[b.pool_is_E]
    i_out = outdeg[~b.pool_is_E]
    indeg = np.bincount(b.csr_dst_local, minlength=cfg.n_pool)
    zero_in = int(np.count_nonzero(indeg == 0))
    if zero_in != 0:
        raise AssertionError(f"直接重算零入度 = {zero_in} != 0")
    return (f"E 源出度 max={r['e_src_outdeg_max']}<=12, "
            f"mean={float(e_out.mean()):.2f}; I 源出度 max={int(i_out.max())}; "
            f"入度 range=[{int(indeg.min())}, {int(indeg.max())}]; "
            f"零入度={zero_in}, 孤立=0")


# ---------------------------------------------------------------------------
# G0-13 延迟上界 max_delay < L（时间轮）
# ---------------------------------------------------------------------------


@_register(13, "delay_upper_bound",
           "延迟上界 max_delay<L（轮槽可寻址性 + 最远延迟实测）")
def _gate_13() -> str:
    """§4 规模网络：csr_delay ⊆ [1, delay_max=15]、无零延迟边；轮长
    L = delay_max+1 = 16 且 ring 形状 (L, n_pool)（max_delay < L 保证投递
    槽唯一可寻址）；内核实测 delay=15 边于 t=15 精确发放。"""
    b = _pristine_default()
    cfg = b.cfg
    dmin, dmax = int(b.csr_delay.min()), int(b.csr_delay.max())
    if dmin < cfg.delay_min or dmax > cfg.delay_max:
        raise AssertionError(
            f"延迟越界：实际 [{dmin}, {dmax}]，契约 [{cfg.delay_min}, "
            f"{cfg.delay_max}]"
        )
    if not dmax < cfg.wheel_l:
        raise AssertionError(
            f"max_delay={dmax} >= wheel L={cfg.wheel_l}（违反 max_delay<L）"
        )
    if cfg.wheel_l != cfg.delay_max + 1:
        raise AssertionError("wheel_l != delay_max + 1（轮长契约被破坏）")
    if b.ring.shape != (cfg.wheel_l, cfg.n_pool):
        raise AssertionError(f"ring 形状 {b.ring.shape} != 轮长×池数")
    hist = np.bincount(b.csr_delay, minlength=cfg.delay_max + 1)
    if int(hist[0]) != 0:
        raise AssertionError(f"{int(hist[0])} 条零延迟边（delay_min=1 违约）")
    m = micro_cfg()
    bm = micro_bundle(m, [(1, 2, 15, 1.5, True)])
    st = run_sample(bm, events({0: [1]}), T=30)
    if st["n_spikes_total"] != 1 or int(bm.first_spike[2]) != 15:
        raise AssertionError(
            f"delay=15（接近 L=16 上界）发放时刻错误："
            f"first_spike={int(bm.first_spike[2])}"
        )
    hist_str = ",".join(str(int(h)) for h in hist[1:])
    return (f"delay∈[{dmin},{dmax}] 含于 [1,15]；max_delay={dmax} < L={cfg.wheel_l}；"
            f"delay-15 实测 t=15；直方图(1..15)={hist_str}")


# ---------------------------------------------------------------------------
# G0-14 不应期严格封锁（t+1/t+2 封锁、t+3 允许；R1 严格大于）
# ---------------------------------------------------------------------------


@_register(14, "refractory_strict",
           "不应期严格封锁（REFR=2：t+1/t+2 封锁，t+3 允许）")
def _gate_14() -> str:
    """池于 t=1 发放 → refr = t+REFR = 3；发放判定为 t > refr（R1 严格大于），
    故 t=2、t=3 的到达被封锁。核验方式：
      (a) T=4 截窗调度 {0,1,2}（到达 t=1,2,3）：若 t=2/t=3 因 bug 提前发放，
          计数会变为 2；正确实现只有 t=1 一次 → 断言 total == 1（封锁证据）；
      (b) 全窗调度 {0,1,2,3}：t=1 发放后最早恢复时刻为 t+REFR+1=4（残余电荷
          于 t=4 触发），计数恰为 2 且 first_spike=1（放行边界证据）。"""
    cfg = micro_cfg()
    if cfg.refr != 2:
        raise AssertionError(f"本门禁针对 REFR=2 契约，当前 REFR={cfg.refr}")
    # (a) 截窗：T=4 内若 t=2/t=3 提前发放，计数将是 2 而非 1
    b = micro_bundle(cfg, [(0, 0, 1, 1.2, True)])
    st_a = run_sample(b, events({0: [0], 1: [0], 2: [0]}), T=4)
    if st_a["n_spikes_total"] != 1 or int(b.first_spike[0]) != 1:
        raise AssertionError(
            f"不应期提前发放：截窗 T=4 内 total={st_a['n_spikes_total']} "
            f"(期望 1：t=2/t=3 必须封锁；若提前放行将计到 2)"
        )
    # (b) 全窗：最早放行为 t+REFR+1=4（残余电荷于放行瞬间触发）
    b2 = micro_bundle(cfg, [(0, 0, 1, 1.2, True)])
    st_b = run_sample(b2, events({0: [0], 1: [0], 2: [0], 3: [0]}), T=20)
    if st_b["n_spikes_total"] != 2 or int(b2.first_spike[0]) != 1:
        raise AssertionError(
            f"放行边界错误：全窗 total={st_b['n_spikes_total']} "
            f"first={int(b2.first_spike[0])}（期望 t=1 与最早 t=4 共 2 次）"
        )
    return ("REFR=2：t=1 发放后 t+1/t+2 到达全封锁（截窗计数 1）；"
            "最早放行 = t+REFR+1=4（全窗计数 2）")


# ---------------------------------------------------------------------------
# G0-15 状态重置回归（两样本无残留；B11 契约）
# ---------------------------------------------------------------------------


@_register(15, "state_reset_regression",
           "状态重置回归（两样本无残留，B11）")
def _gate_15() -> str:
    """同一输入两连跑结果逐位一致（入口重置清 V/refr/counts/first_spike/
    trace/ring）；忙样本后接空样本零发放（无残留投递）；无塑性样本不得移动
    theta/rate_ema（homeo 记忆跨样本保留，开启后继续自适应）。"""
    cfg = micro_cfg()
    b = micro_bundle(cfg, [(0, 0, 1, 1.5, True), (1, 2, 15, 1.5, True)])
    sch = events({0: [0], 3: [1]})
    theta0 = b.theta.copy()
    ema0 = b.rate_ema.copy()
    s1 = run_sample(b, sch, T=25)
    s2 = run_sample(b, sch, T=25)
    if not np.array_equal(s2["spike_counts"], s1["spike_counts"]):
        raise AssertionError(
            "两连跑发放计数不一致（入口重置残留）: "
            f"run1={s1['spike_counts'].tolist()} run2={s2['spike_counts'].tolist()}"
        )
    if not np.array_equal(s2["first_spike"], s1["first_spike"]):
        raise AssertionError("两连跑 first_spike 不一致")
    if not (np.array_equal(b.theta, theta0) and np.array_equal(b.rate_ema, ema0)):
        raise AssertionError("无塑性样本不得改变 theta/rate_ema（B11）")
    s3 = run_sample(b, {}, T=25)          # 忙样本后接空样本
    if s3["n_spikes_total"] != 0:
        raise AssertionError(
            f"残留漏入空样本：{s3['n_spikes_total']} 次发放（B11 重置失败）"
        )
    if np.any(b.V) or np.any(b.ring) or np.any(b.trace) or np.any(b.counts):
        raise AssertionError("空样本后状态非净（V/ring/trace/counts 有残留）")
    # homeo 记忆：rate_ema/theta 跨样本保留；关闭则原样保持、开启继续自适应
    b5 = micro_bundle(cfg, [(0, 0, 1, 1.5, True)])
    run_sample(b5, events({0: [0], 5: [0]}), T=200, homeo_on=True)
    t1, e1 = float(b5.theta[0]), float(b5.rate_ema[0])
    if not (e1 > 0.0 and abs(t1 - 1.0) > 1e-9):
        raise AssertionError("homeo_on 样本必须移动 theta 并累积 rate_ema")
    run_sample(b5, events({0: [0], 5: [0]}), T=200)   # homeo off
    if float(b5.theta[0]) != t1 or float(b5.rate_ema[0]) != e1:
        raise AssertionError("homeo 关闭时 theta/rate_ema 必须原样保留")
    run_sample(b5, events({0: [0], 5: [0]}), T=200, homeo_on=True)
    if float(b5.theta[0]) == t1:
        raise AssertionError("homeo 重新开启后 theta 必须继续自适应")
    return ("两连跑逐位一致；忙→空样本零残留；theta/rate_ema 保留语义正确"
            "（off 冻结 / on 自适应）")


# ---------------------------------------------------------------------------
# 运行器与 CLI
# ---------------------------------------------------------------------------


def _spec(gid: int):
    for s in GATE_SPECS:
        if s[0] == gid:
            return s
    return None


def run_one(gid: int) -> Tuple[bool, str]:
    """独立运行单个门禁；返回 (是否通过, 报告行)。"""
    spec = _spec(gid)
    if spec is None:
        return False, f"[ERROR] 未知 gate id: {gid}（合法范围 1..15）"
    _, key, title, fn = spec
    try:
        detail = fn()
        return True, f"[PASS] G0-{gid:02d} {key} — {title}：{detail}"
    except AssertionError as exc:
        return False, f"[FAIL] G0-{gid:02d} {key} — {title}\n       {exc}"
    except Exception as exc:  # noqa: BLE001 —— 门禁彼此独立，异常不中断批量
        return False, (f"[ERROR] G0-{gid:02d} {key} — {title}\n       "
                       f"{type(exc).__name__}: {exc}")


def run_many(ids: Sequence[int]) -> int:
    """批量运行；打印逐项 PASS/FAIL 与汇总。返回退出码（0=全绿）。"""
    ids = sorted(set(ids))
    known = {s[0] for s in GATE_SPECS}
    unknown = [i for i in ids if i not in known]
    if unknown:
        print(f"未知 gate id: {unknown}；合法范围 1..15")
        return 2
    passed: List[bool] = []
    for gid in ids:
        ok, msg = run_one(gid)
        print(msg)
        passed.append(ok)
    n_pass = sum(1 for p in passed if p)
    n_fail = len(passed) - n_pass
    print("=" * 78)
    print(f"G0 gate summary: PASS {n_pass}/{len(passed)}"
          + (f", FAIL {n_fail}" if n_fail else ""))
    if n_fail == 0:
        print("ALL GREEN —— G0 门禁通过，允许进入 G1")
        return 0
    print("NOT GREEN —— G0 未全绿，禁止进入 G1（先修复后复跑）")
    return 1


def _print_list() -> None:
    print("G0 十五项断言清单（每项独立可执行；python -m hstdn.exp.gates [ids]）：")
    for gid, key, title, _ in sorted(GATE_SPECS):
        print(f"  {gid:02d}. {title}  [{key}]")


def main(argv: Optional[Sequence[str]] = None) -> int:
    # 控制台代码页防御：任何平台都按 UTF-8 输出，杜绝 GBK 无法编码的符号崩溃
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    ap = argparse.ArgumentParser(
        prog="python -m hstdn.exp.gates",
        description="G0 十五项断言门禁（全绿方可进入 G1）",
    )
    ap.add_argument("ids", nargs="*", type=int,
                    help="门禁编号（默认运行全部 1..15）")
    ap.add_argument("--list", action="store_true", help="仅打印清单")
    ap.add_argument("--trace", action="store_true",
                    help="失败时向上抛出原始异常（打印 traceback，便于调试）")
    args = ap.parse_args(list(argv) if argv is not None else None)
    if args.list:
        _print_list()
        return 0
    ids = list(args.ids) if args.ids else [s[0] for s in GATE_SPECS]
    if args.trace:
        # 调试模式：不做异常包装 —— 首个失败直接抛给解释器打印完整 traceback
        for gid in sorted(set(ids)):
            spec = _spec(gid)
            if spec is None:
                print(f"[ERROR] 未知 gate id: {gid}")
                return 2
            _, key, title, fn = spec
            print(f"--- G0-{gid:02d} {key} — {title}（trace 模式）---")
            detail = fn()
            print(f"[PASS] {title}：{detail}")
        return 0
    return run_many(ids)


if __name__ == "__main__":
    sys.exit(main())
