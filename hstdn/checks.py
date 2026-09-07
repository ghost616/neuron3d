"""轻量 G0 契约断言辅助（framework 模块维护，公共工具）。

供 hstdn/core（layout 等）与 hstdn/exp/gates 等模块复用。约定：
- 校验失败一律抛出 ``AssertionError``，消息含可读说明 + 相关统计数值
  （违反个数、实际范围、期望/实际形状或 dtype 等）；
- 本模块只做**通用**契约校验（形状/dtype/ID 域/有限性/单调性），
  不判定业务契约（如 CSR 结构、增益标定阈值、全局-局部 ID 对应关系等），
  具体业务断言实现细节由 core 模块负责；
- 调用方传错参数类型（非数组、非整数 ID、空域上界等）属于编程错误，
  同样以带上下文的断言失败暴露，便于 G0 逐项定位。
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "assert_dtype",
    "assert_finite",
    "assert_global_ids",
    "assert_local_ids",
    "assert_non_decreasing",
    "assert_shape",
]


def _asarray(arr: Any, name: str) -> np.ndarray:
    """将输入规整为 ndarray；失败时给出带对象名的可读断言消息。"""
    try:
        return np.asarray(arr)
    except Exception as exc:  # noqa: BLE001 —— 统一转成带上下文的断言失败
        raise AssertionError(f"{name} 无法转换为 ndarray：{exc!r}") from exc


def _require_1d_int(arr: np.ndarray, name: str) -> None:
    """前置校验：一维、整数 dtype 的数组；违背即抛断言（调用方编程错误）。"""
    if arr.ndim != 1:
        raise AssertionError(f"{name} 应为一维数组，实际 shape={arr.shape}（ndim={arr.ndim}）")
    if arr.dtype.kind not in "iu":
        raise AssertionError(f"{name} 应为整数 dtype（ID 数组），实际 dtype={arr.dtype}")


def assert_global_ids(ids: Any, n_total: int, *, name: str = "ids") -> None:
    """校验全局 ID 数组契约：一维整数，且全部落在 [0, n_total) 内。

    存储层（CSR/CSC/trace 等）一律使用全局 ID；n_total 为该存储对象的
    全局 ID 域上界（通常为 N_IN + N_POOL，或视图相关总数）。

    参数:
        ids: 一维整数 ID 数组（array-like，可为空）。
        n_total: 全局 ID 域上界，正整数。
        name: 出错消息中的对象名。

    返回:
        None；违背契约时抛 AssertionError（含越界个数与实际范围）。
    """
    _assert_ids_in_domain(ids, hi=n_total, name=name, domain_label="全局 ID")


def assert_local_ids(ids: Any, n_local: int, *, name: str = "ids") -> None:
    """校验池局部 ID 数组契约：一维整数，且全部落在 [0, n_local) 内。

    状态数组访问层（V/theta/refr/counts/rate_ema/ring/csc_ptr）一律使用
    池局部 ID；n_local 为该池规模（通常为 N_POOL）。全局→局部转换只发生在
    内核投递处（dst_local = dst - N_IN），本函数不负责该转换的语义判定。

    参数:
        ids: 一维整数 ID 数组（array-like，可为空）。
        n_local: 池局部 ID 域上界，正整数。
        name: 出错消息中的对象名。

    返回:
        None；违背契约时抛 AssertionError（含越界个数与实际范围）。
    """
    _assert_ids_in_domain(ids, hi=n_local, name=name, domain_label="池局部 ID")


def _assert_ids_in_domain(ids: Any, hi: int, *, name: str, domain_label: str) -> None:
    """通用 ID 域校验：一维整数数组取值 ⊆ [0, hi)。"""
    arr = _asarray(ids, name)
    _require_1d_int(arr, name)
    if arr.size == 0:
        return
    if hi <= 0:
        raise AssertionError(f"{name} 的 {domain_label} 域上界须为正整数，实际 hi={hi}")
    lo = int(arr.min())
    up = int(arr.max())
    if lo < 0 or up >= hi:
        n_bad = int(np.count_nonzero((arr < 0) | (arr >= hi)))
        raise AssertionError(
            f"{name} 违反{domain_label}域契约 [0, {hi})：{n_bad}/{arr.size} 个越界，"
            f"实际取值范围 [{lo}, {up}]"
        )


def assert_shape(arr: Any, shape: tuple[int, ...], *, name: str = "arr") -> None:
    """校验数组形状与期望完全一致（含维数）。

    参数:
        arr: array-like 输入。
        shape: 期望形状，如 (N_POOL, K_MAX)。
        name: 出错消息中的对象名。

    返回:
        None；形状不符时抛 AssertionError（含期望/实际形状与维数）。
    """
    a = _asarray(arr, name)
    expected = tuple(int(d) for d in shape)
    if a.ndim != len(expected) or a.shape != expected:
        raise AssertionError(
            f"{name} 形状不匹配：期望 {expected}（ndim={len(expected)}），"
            f"实际 {a.shape}（ndim={a.ndim}）"
        )


def assert_dtype(arr: Any, dtype: Any, *, name: str = "arr") -> None:
    """校验数组 dtype 与期望完全一致。

    参数:
        arr: array-like 输入。
        dtype: 期望 dtype（可为 numpy dtype、内置类型或类型字符串）。
        name: 出错消息中的对象名。

    返回:
        None；dtype 不符时抛 AssertionError（含期望/实际 dtype）。
    """
    a = _asarray(arr, name)
    want = np.dtype(dtype)
    if a.dtype != want:
        raise AssertionError(f"{name} dtype 不匹配：期望 {want}，实际 {a.dtype}")


def assert_finite(arr: Any, *, name: str = "arr") -> None:
    """校验数值数组不含 NaN / Inf。

    参数:
        arr: 数值 array-like（整数/浮点/复数/布尔）。
        name: 出错消息中的对象名。

    返回:
        None；含非有限值时抛 AssertionError（含非有限个数与占比）。
    """
    a = _asarray(arr, name)
    if a.dtype.kind not in "biufc":
        raise AssertionError(f"{name} 须为数值数组才能校验有限性，实际 dtype={a.dtype}")
    finite = np.isfinite(a)
    if not bool(finite.all()):
        n_bad = int(np.count_nonzero(~finite))
        raise AssertionError(
            f"{name} 含 {n_bad} 个非有限值（NaN/Inf）：元素总数 {a.size}，"
            f"占比 {n_bad / a.size:.2%}"
        )


def assert_non_decreasing(values: Any, *, name: str = "values") -> None:
    """校验一维数值数组单调不减（相邻元素允许相等）。

    典型用途：CSR indptr / CSC 偏移指针、排序后的 ID 列表等结构不变量。

    参数:
        values: 一维数值 array-like。
        name: 出错消息中的对象名。

    返回:
        None；存在严格递减相邻对时抛 AssertionError（含首个违反对的位置与取值）。
    """
    a = _asarray(values, name)
    if a.ndim != 1:
        raise AssertionError(f"{name} 应为一维数组，实际 shape={a.shape}（ndim={a.ndim}）")
    if a.dtype.kind not in "iuf":
        raise AssertionError(f"{name} 须为整数/浮点数组，实际 dtype={a.dtype}")
    if a.size < 2:
        return
    dec = np.flatnonzero(a[1:] < a[:-1])
    if dec.size:
        i = int(dec[0])
        raise AssertionError(
            f"{name} 违反单调不减契约：首个递减点 i={i}（v[{i}]={a[i]} > v[{i + 1}]={a[i + 1]}），"
            f"共 {int(dec.size)} 处递减"
        )