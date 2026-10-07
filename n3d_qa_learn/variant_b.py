"""n3d_qa_learn 变体 B：单层 D→D 可学变换 + **冻结特征库**（"只变换、不生成"）。

设计文档三步走的**第二步**（第一步 1a = 分档鲁棒性考卷，见 :mod:`n3d_qa_learn.robust_eval`）。
本模块正面检验一个架构命题：**「表内嵌输出层、权重即特征库」能否让 argmax 真正可学**。

口径（逐轮确认，不得擅自变更）
-----------------------------
1. **载体**：``x`` = 原始编码器特征（raw），**不经 N3D 骨干**；考卷路径本身不含任何模型。
2. **特征档**：仅 ``lexical-88``（``local-hash``，D=88）与 ``bge-m3-1024``（``BAAI/bge-m3``，
   D=1024）；不纳入 ``zh-bag``。
3. **模型**：单层 D→D 变换 ``T``（**恒等初始化**）+ 冻结特征库内积评分 ——
   即「表内嵌输出层、权重即特征库、冻结」，**只学变换层**。
4. **训练信号**：**扰动自监督**（主方案）。对训练查询行施加 1a 同一套扰动，
   标签 = 该行自身在**全量键表**中的行下标，交叉熵作用于**冻结全表**。
5. **干净自监督（A 组）**仅作对照档，用于把「干净训练 ≈ 无变化」从推理变成实测。
6. **不纳入**对齐 / SupCon 类辅助损失（历史纠正记录 #17：对齐度↑ ≠ 可用性↑）。
7. **训练/评测划分**：训练查询 = 库行 1999（严格排除 666 查询行），评测查询 = 666 查询行；
   键表固定为全量 2665 行。
8. **训练扰动档**：全部 9 格（noise / mask / nmag × 弱 / 中 / 强）。
   1a 的有效性门禁**只用于判据**，不用来过滤训练数据。
9. **单 seed 42**；所有 Δ 为**单点差**、无跨 seed 极差，报告必须显式标明。
10. **复用 1a 的实现**：扰动一律走 :func:`n3d_qa_learn.robust_eval.perturb_matrix` /
    :func:`n3d_qa_learn.robust_eval.derived_seed` / ``PERTURB_GRID``，归一化一律走
    :func:`n3d_qa_learn.entry_table.l2_normalize_rows` —— **不自造第二份口径**。

恒等门禁（硬门禁）
------------------
训练前 ``T = I`` 时，变体 B 的 top-1 必须在 clean + 全部 9 个扰动档上与**余弦最近邻逐条一致**。
实现上有两层保障：

* **数学层**：输入已经是逐行 L2 归一化向量，行范数 = 1（float32 漂移 ≤ 1e-6）；
* **逐位层**：:class:`DToDTransform` 在 ``T`` **恰为恒等**时走**透传快路径**
  （``torch.equal(W, I) and bias == 0``），不做 ``x @ W.T`` 与 ``x + b``，
  从而 ``norm(T(x)) ≡ norm(x) = 1`` 在 float32 下**逐位成立**，不引入 ``x + 0`` 的浮点舍入；
  ``T`` 一旦被训练离开恒等，该快路径**自动失效**。

本模块仍在**两条独立路径**上现场实测一致性，并**如实登记**不一致条数
（若出现打平 / 翻转，如实报出条数与位置，不静默对齐）。

产物纪律
--------
一律写 ``checkpoints/qa_learn/_verify/variant_b/``；**不落盘特征矩阵、不产 zip 产物**；
产物内不含挂钟时间 / 耗时（确定性纪律，同 :mod:`n3d_qa_learn.robust_eval`）。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import entry_table as ET
from . import robust_eval as R
from .entry_table import EntryKeyTable, TextEntryBundle
from .train import DEFAULT_VERIFY_DIR

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

#: 报告根目录（验证类运行一律写 ``_verify/``）。
VARIANT_B_DIR: str = os.path.join(DEFAULT_VERIFY_DIR, "variant_b")

#: 模块名（进产物，溯源用）。
MODULE_NAME: str = "n3d_qa_learn.variant_b"

#: 产物 schema 版本。
ARTIFACT_SCHEMA: str = "variant-b-v1"

#: 训练 / 扰动 seed（**冻结**；单 seed 口径）。
VARIANT_B_SEED: int = 42

#: 训练侧默认超参（Adam；**唯一优化器**，只优化变换层）。
DEFAULT_EPOCHS: int = 30
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_LR: float = 1e-2
DEFAULT_WEIGHT_DECAY: float = 0.0

#: 训练 / 评测划分（键表固定全量 2665；库行 1999 训练、查询行 666 评测）。
FULL_TABLE_ROWS: int = 2665
LIBRARY_ROWS: int = 1999
QUERY_ROWS: int = 666

#: 恒等快路径判定的**严格**口径（不是容差）。
IDENTITY_FASTPATH_RULE: str = (
    "`torch.equal(W, I) and bool((b == 0).all())` 逐位判定（**不是容差**）；"
    "两者同时成立时走**零加性快路径** `x + x@(W−I).T + (b − mean(b))` —— 两项在恒等时"
    "逐位为 +0.0，故 `T(x)` 与 `x` **逐位相同**、`norm(T(x)) ≡ norm(x) = 1` 在 float32 下"
    "逐位成立；**不得**写成 `return x`（那会让输入脱离计算图，训练侧 `backward()` 立刻报"
    "`element 0 of tensors does not require grad`，现场实测踩到过）；T 离开恒等后自动改走"
    "`x @ W.T + b`"
)

#: 训练模式（``perturb`` = 主方案；``clean`` = A 组对照档）。
TRAIN_MODES: Tuple[str, ...] = ("perturb", "clean")

#: 训练侧的扰动：**全部 9 格**（计划口径第 8 条：不用有效性门禁过滤训练数据）。
TRAIN_PERTURB_CELLS: Tuple[Tuple[str, str], ...] = tuple(
    (str(kind), str(level)) for kind in R.PERTURB_TYPES for level in R.PERTURB_LEVELS
)

#: 训练批序打乱用的**局部** generator 种子（不消耗全局 RNG）。
SHUFFLE_SEED: int = 4242

#: 可训参数更新量门禁的判定口径。
UPDATE_GATE_RULE: str = (
    "训练前逐参数记录 float32 裸字节 SHA256（`state_bytes_sha256`），训练后重算；"
    "**只要存在任一参数裸字节变化**即判「可训参数更新量 > 0」（`passed = n_updated > 0`）。"
    "该门禁**不设阈值**、只判「是否有更新」，避免把超参差异读成能力差异"
)

#: 1a **生产实测值**（登记用，不参与任何判定）。
#:
#: [!] **来源与一处自我更正（必须如实登记）**：下表的数字取自 1a 的
#: ``robust_run.json`` **文本侧 ``cell_role=main`` 主判据格**（``variant=0``）。
#: ``bge-m3-1024`` 的 ``nmag`` 三格在本仓库里**存在两组并存的数字**：本表取的
#: ``0.481982 / 0.144144 / 0.043544`` 是**产物现场值**（README §15.5 主表与
#: §15.10 改后列亦为该组）；另一组 ``1.000000 / 0.996997 / 0.983483`` 出现在
#: README §15.10「修复前后完整对照」表的**改前**列与 §15.12 所引的 **W1 修复前历史值**。
#: 本模块以**产物为唯一现场来源**（``anchor_1a_check`` 逐格对账），
#: 并把该差异显式登记在 :data:`ANCHOR_RULE` 与 README §16.7，**不静默对齐**。
ROBUST_1A_KNN_REFERENCE: Dict[str, Dict[str, Dict[str, float]]] = {
    ET.PROFILE_LEXICAL: {
        "noise": {"clean": 1.000000, "weak": 1.000000, "middle": 0.984985, "strong": 0.789790},
        "mask": {"clean": 1.000000, "weak": 0.990991, "middle": 0.947447, "strong": 0.770270},
        "nmag": {"clean": 1.000000, "weak": 0.391892, "middle": 0.099099, "strong": 0.049550},
    },
    ET.PROFILE_SEMANTIC: {
        "noise": {"clean": 1.000000, "weak": 0.998498, "middle": 0.980480, "strong": 0.758258},
        "mask": {"clean": 1.000000, "weak": 1.000000, "middle": 1.000000, "strong": 1.000000},
        "nmag": {"clean": 1.000000, "weak": 0.481982, "middle": 0.144144, "strong": 0.043544},
    },
}

#: 旁证锚点容差（1a 登记值 vs 本模块现场重算的 KNN 基线）。
ANCHOR_TOL: float = 1e-6

#: 旁证锚点规则文本（唯一来源；同时进入 `evidence.knn_reference_rule` 与
#: `anchor_1a_check.rule`）。
#:
#: [!] 该文本**必须**携带「一处不同源」的登记（离朱 R57 F2）：`bge-m3-1024` 的
#: ``nmag`` 三格在本仓库里有**两组并存的数字**，只有一组能与 1a 的 run 产物对上。
ANCHOR_RULE: str = (
    "本模块现场重算的 KNN 基线应与 1a 登记值在 1e-6 内一致；"
    "两者的检索池口径相同（全量键表自检索），不一致即如实报出，**不静默对齐**。"
    "**已登记的一处不同源**：`bge-m3-1024` 的 `nmag` 弱/中/强在本仓库里有两组并存的数字 —— "
    "`1.000000 / 0.996997 / 0.983483`（README §15.10「修复前后完整对照」表的**改前**列与 "
    "§15.12 所引的 W1 修复前历史值）vs `0.481982 / 0.144144 / 0.043544`"
    "（`checkpoints/qa_learn/_verify/robust/robust_run.json` 文本侧 `cell_role=main` 的**现场值**，"
    "README §15.5 主表与 §15.10 改后列也已同步为该组）。"
    "本模块**一律以产物为唯一现场来源**（故 `ROBUST_1A_KNN_REFERENCE` 取后者），"
    "该差异同时登记在 `ROBUST_1A_KNN_REFERENCE` 的常量注释与 README §16.7。"
)

#: 1a 判**有效**的扰动格（**只用于主判据**，不用来过滤训练数据）。
EFFECTIVE_CELLS_1A: Dict[str, Tuple[str, ...]] = {
    ET.PROFILE_LEXICAL: (
        "noise/weak", "noise/middle", "noise/strong",
        "mask/weak", "mask/middle", "mask/strong",
        "nmag/weak", "nmag/middle", "nmag/strong",
    ),
    ET.PROFILE_SEMANTIC: (
        "noise/weak", "noise/middle", "noise/strong",
        "nmag/weak", "nmag/middle", "nmag/strong",
    ),
}

#: 1a 判**无效**的扰动格（如实登记；判据不采用，但数字仍必须报出）。
INEFFECTIVE_CELLS_1A: Dict[str, Tuple[str, ...]] = {
    ET.PROFILE_SEMANTIC: ("mask/weak", "mask/middle", "mask/strong"),
}


# ---------------------------------------------------------------------------
# 1. 变体 B 模型：单层 D→D 可学变换 + 恒等初始化
# ---------------------------------------------------------------------------


class DToDTransform(torch.nn.Module):
    """**单层 D→D 可学变换** ``T(x) = x @ W.T + b``，恒等初始化 ``W = I, b = 0``。

    口径
    ----
    * ``W``：``[D, D]`` 可学参数，初始化 = 单位阵（``torch.eye``，**不消耗 RNG**）；
    * ``b``：``[D]`` 可学参数，初始化 = 全零（**不消耗 RNG**）；
    * **恒等快路径**：``torch.equal(W, I) and (b == 0).all()`` 时 ``forward`` 直接返回输入，
      不做矩阵乘与加法 —— 保证 ``norm(T(x)) ≡ norm(x) = 1`` 在 float32 下**逐位成立**
      （否则 ``x + 0`` 的浮点舍入会让范数偏离 1，恒等门禁只能"近似"一致）。
      这是**显式的逐位判定**、不是容差；``T`` 一旦离开恒等，该路径自动失效。

    关键不变量（构造期，违反即抛）
    -----------------------------
    1. ``D >= 1``；
    2. ``weight`` 必须**逐位**等于 ``torch.eye(D)``、``bias`` 必须**逐位**全零。

    参数
    ----
    dim : int
        连接参数 ``D``。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        d = int(dim)
        if d < 1:
            raise ValueError(f"变换层维度 D 必须 >= 1，当前 {d}")
        self.dim: int = d
        self.weight = torch.nn.Parameter(torch.eye(d, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
        # 构造期不变量（历史纠正记录 #10：构造期不变量显式断言，且不断言训练后量）
        assert torch.equal(
            self.weight.detach(), torch.eye(d, dtype=torch.float32)
        ), "DToDTransform 的 weight 必须是逐位单位阵（恒等初始化）"
        assert bool(
            (self.bias.detach() == 0).all()
        ), "DToDTransform 的 bias 必须逐位全零（恒等初始化）"
        # 最近一次 forward 的取证（推理态才记录），供恒等门禁读取
        self._last_io: Dict[str, Any] = {}

    def is_identity(self) -> bool:
        """``T`` 是否**逐位**等于恒等（决定是否走透传快路径）。"""
        with torch.no_grad():
            same_w = bool(torch.equal(self.weight.detach(), torch.eye(self.dim)))
            zero_b = bool((self.bias.detach() == 0).all())
        return bool(same_w and zero_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用变换；恒等时走**零加性**快路径（逐位等于输入，且**保持可导**）。

        参数
        ----
        x : torch.Tensor
            ``[B, D]`` 待变换特征。

        返回
        ----
        torch.Tensor
            ``[B, D]`` 变换结果（恒等时与输入**逐位相同**，且仍带 ``grad_fn``）。

        实现要点（**关键**）
        -------------------
        恒等时**不能**直接 ``return x``：那会让 ``x`` 脱离计算图，
        训练侧 ``loss.backward()`` 立刻报
        ``RuntimeError: element 0 of tensors does not require grad``
        （现场实测踩到该坑）。正确写法是保留一条**数学上恰为 0 的加性项**：

        * 权重项 = ``x @ (W - I).T`` —— 恒等时 ``W - I`` 逐位全零，且
          float32 下 ``x @ 0 = 0``（逐元素求和，0 的求和恒为 +0.0，
          负零与正零在加法中产生 +0.0），故 ``x + 0`` **逐位等于** ``x``；
        * 偏置项 = ``-mean(b) + b`` —— 恒等时 ``b`` 逐位全零，故该项逐位为 0。

        两个加性项都为**可微**项（对可训参数有非零偏导：``d/dW[x@(W-I).T] = x``、
        ``d/db[b - mean(b)] = 1 - 1/D``），因此训练侧梯度正常回传。
        """
        if x.dim() != 2 or int(x.shape[1]) != int(self.dim):
            raise ValueError(
                f"变换层期望 [B, {self.dim}] 输入，当前形状 = {tuple(x.shape)}"
            )
        fast = self.is_identity()
        if fast:
            eye = torch.eye(self.dim, dtype=x.dtype, device=x.device)
            zero_delta = x @ (self.weight - eye).t()
            zero_shift = self.bias - self.bias.mean()
            out = x + zero_delta + zero_shift
        else:
            out = torch.nn.functional.linear(x, self.weight, self.bias)
        if not self.training:
            with torch.no_grad():
                self._last_io = {
                    "fast_path": bool(fast),
                    "input_norm_min": float(torch.linalg.norm(x, dim=1).min()),
                    "input_norm_max": float(torch.linalg.norm(x, dim=1).max()),
                    "output_norm_min": float(torch.linalg.norm(out, dim=1).min()),
                    "output_norm_max": float(torch.linalg.norm(out, dim=1).max()),
                }
        return out

    def last_io_evidence(self) -> Dict[str, Any]:
        """最近一次推理 ``forward`` 的输入 / 输出范数取证（只读副本）。"""
        return dict(self._last_io)

    def explicit_matrix(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(W, b)`` 的 detach 副本（供「显式 matmul 路径」旁证使用）。"""
        return self.weight.detach().clone(), self.bias.detach().clone()


@dataclass(frozen=True)
class VariantBModel:
    """变体 B 的**完整模型**：冻结特征库（全量键表） + 可学变换 ``T``。

    参数
    ----
    transform : DToDTransform
        单层 D→D 可学变换（**唯一可训部分**）。
    keys : torch.Tensor
        ``[N, D]`` **冻结**特征库（逐行 L2 归一化；``requires_grad=False``）。
    table_sha256 : str
        键表内容指纹（与 :class:`entry_table.EntryKeyTable` 同口径）。
    """

    transform: DToDTransform
    keys: torch.Tensor
    table_sha256: str = ""

    @property
    def n_keys(self) -> int:
        """特征库条目数 ``N``。"""
        return int(self.keys.shape[0])

    @property
    def dim(self) -> int:
        """连接参数 ``D``。"""
        return int(self.keys.shape[1])

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """可训参数对象列表（**现场枚举** ``named_parameters``，不手写名字）。"""
        return [p for _, p in self.transform.named_parameters()]

    def parameter_snapshot(self) -> List[Dict[str, Any]]:
        """可训参数的**现场枚举**快照（名字 / 形状 / 元素数 / 裸字节 SHA256）。"""
        out: List[Dict[str, Any]] = []
        for name, param in self.transform.named_parameters():
            blob = param.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
            out.append(
                {
                    "name": str(name),
                    "shape": [int(v) for v in param.shape],
                    "n_element": int(param.numel()),
                    "dtype": str(param.dtype).replace("torch.", ""),
                    "requires_grad": bool(param.requires_grad),
                    "state_bytes_sha256": ET.sha256_bytes(blob),
                }
            )
        return out

    def transform_score(self, x: torch.Tensor) -> torch.Tensor:
        """``T(x) @ keys.T``（**未归一化**；训练侧交叉熵用，可微）。

        参数
        ----
        x : torch.Tensor
            ``[B, D]`` 查询特征（已 L2 归一化）。

        返回
        ----
        torch.Tensor
            ``[B, N]`` 打分矩阵（``keys`` 不参与梯度）。
        """
        return self.transform(x) @ self.keys.t()

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """检索打分：``T(x)`` **重新 L2 归一化**后与冻结键表做内积（评测侧唯一口径）。

        [!] **必须**先过 :meth:`transform` 再归一化。首版实现漏掉了 ``self.transform(x)``
        （直接写 ``normalize_query(x) @ keys.T``），后果是：整个评测侧退化成
        「未训练模型 = 余弦 KNN」，所有格的 Δ 结构性地恒为 0。该缺陷由
        :func:`score_path_evidence` 这条**自洽门禁**现场抓出（它比对
        ``score(q)`` 与显式 ``normalize_query(transform(q)) @ keys.T``），
        现已成为恒等门禁的一部分 —— 任何"评测侧没走变换层"的回归都会被立刻拦下。
        """
        return self.normalize_query(self.transform(x)) @ self.keys.t()

    @staticmethod
    def normalize_query(x: torch.Tensor) -> torch.Tensor:
        """逐行 L2 归一化（数值路径与 :func:`entry_table.l2_normalize_rows` **逐位一致**）。

        口径理由：``entry_table.l2_normalize_rows`` 用 **float64** 求范数再落回 float32；
        本函数用逐元素 float32 求和算出同一 float64 值（float32 的 24 位尾数恰是 float64
        53 位尾数的前缀，故「float32 逐元素求和」在 float64 下是**精确**的），
        两条路径因此给出**逐位相同**的结果 —— 这是恒等门禁能要求「逐条一致」而非
        「近似一致」的前提。零范数行**显式报错**（不静默产出 NaN / 全零特征）。
        """
        norms = torch.sqrt((x.double() * x.double()).sum(dim=1))
        zero = norms <= 0.0
        if bool(zero.any()):
            idx = [int(i) for i in torch.nonzero(zero).flatten().tolist()[:20]]
            raise ValueError(
                "查询侧出现零范数行（L2 归一化分母为 0）："
                f"行下标 = {idx}；拒绝静默产出 NaN / 全零特征"
            )
        return (x.to(torch.float64) / norms[:, None]).to(torch.float32)


def build_model(table: EntryKeyTable) -> VariantBModel:
    """由统一条目特征表构造变体 B 模型（``T = I`` + **冻结**键表副本）。

    参数
    ----
    table : EntryKeyTable
        统一的条目特征表（**键表固定为全量 2665 行**）；本函数拷贝一份并把
        ``requires_grad`` 置 False，**不改动传入对象**。

    返回
    ----
    VariantBModel
        ``transform`` 恒等初始化；``keys`` 为 detach 后的副本。

    异常
    ------
    ValueError
        表行数与冻结口径（2665）不符。
    """
    keys = table.keys.detach().to(torch.float32).clone().contiguous()
    keys.requires_grad_(False)
    if int(keys.shape[0]) != int(FULL_TABLE_ROWS):
        raise ValueError(
            f"变体 B 的键表固定为全量 {FULL_TABLE_ROWS} 行，当前 {int(keys.shape[0])} 行；"
            "训练/评测划分按「全量键表 + 库行 1999 训练 + 查询行 666 评测」冻结"
        )
    return VariantBModel(
        transform=DToDTransform(int(keys.shape[1])),
        keys=keys,
        table_sha256=str(table.sha256()),
    )


def sample_bitwise_equality(
    model: VariantBModel, queries: torch.Tensor
) -> Dict[str, Any]:
    """**逐位旁证**：``norm(T(x))`` 是否与 ``norm(x)`` 逐位相同（恒等透传的直接证据）。"""
    with torch.no_grad():
        x = queries.to(torch.float32)
        y = model.transform(x)
        nx = torch.linalg.norm(x, dim=1)
        ny = torch.linalg.norm(y, dim=1)
        same = int((nx == ny).sum().item())
        dev = float((nx - ny).abs().max().item()) if int(nx.numel()) else 0.0
        return {
            "n": int(nx.numel()),
            "n_bitwise_equal": int(same),
            "bitwise_equal_frac": float(same / max(1, int(nx.numel()))),
            "max_abs_norm_deviation": dev,
            "fast_path": bool(model.transform.is_identity()),
        }


def explicit_path_evidence(
    model: VariantBModel, queries: torch.Tensor, *, training_mode: bool = False
) -> Dict[str, Any]:
    """旁证：``T(x)`` 与**显式** ``x @ W.T + b`` 的偏差（恒等时应逐位相同）。

    参数
    ----
    model : VariantBModel
        待检查模型。
    queries : torch.Tensor
        ``[M, D]`` 查询特征。
    training_mode : bool
        ``True`` 时把变换层置 train 态再检查（**训练态走的是零加性快路径**，
        评测态走 ``x @ W.T + b``；两态都必须与显式结果逐位一致）。

    返回
    ----
    Dict[str, Any]
        ``n`` / ``max_abs_deviation`` / ``bitwise_equal`` / ``mode`` / ``note``。
    """
    was_training = bool(model.transform.training)
    model.transform.train(bool(training_mode))
    with torch.no_grad():
        x = queries.to(torch.float32)
        fast = model.transform(x)
        w, b = model.transform.explicit_matrix()
        slow = torch.nn.functional.linear(x, w, b)
        dev = float((fast - slow).abs().max().item()) if int(fast.numel()) else 0.0
        equal = bool(torch.equal(fast, slow))
    model.transform.train(was_training)
    return {
        "n": int(fast.shape[0]),
        "max_abs_deviation": dev,
        "bitwise_equal": equal,
        "mode": "train" if training_mode else "eval",
        "note": (
            "恒等时应逐位相同（零加性快路径加的恰是 +0.0，`x @ I + 0` 在 float32 下恰等于 x）；"
            "非恒等时该偏差只反映 matmul 实现差异，**不参与门禁**"
        ),
    }


def score_path_evidence(model: VariantBModel, queries: torch.Tensor) -> Dict[str, Any]:
    """**自洽门禁**：``model.score(q)`` 必须与显式 ``normalize_query(transform(q)) @ keys.T`` 逐位相同。

    存在理由（首版真实缺陷，现场抓出）：评测侧若漏掉变换层（直接对原始查询做归一化内积），
    整个「变体 B」会退化成「未训练模型 = 余弦 KNN」，所有 Δ 结构性地恒为 0，
    而恒等门禁**照样通过**（因为 T=I 时两条路径本来就该相同）。本函数把
    「评测侧确实走了变换层」变成**逐位可证**的事实：它用一条与 `score` 无关的
    显式计算路径复算并逐字节比对，不一致即判失败。

    返回
    ----
    Dict[str, Any]
        ``n`` / ``bitwise_equal`` / ``max_abs_deviation`` / ``input_norm_mean`` /
        ``transformed_norm_mean`` / ``rule``。
    """
    with torch.no_grad():
        x = queries.to(torch.float32)
        via_score = model.score(x)
        explicit = VariantBModel.normalize_query(model.transform(x)) @ model.keys.t()
        dev = float((via_score - explicit).abs().max().item()) if int(via_score.numel()) else 0.0
        return {
            "n": int(x.shape[0]),
            "bitwise_equal": bool(torch.equal(via_score, explicit)),
            "max_abs_deviation": dev,
            "input_norm_mean": float(torch.linalg.norm(x, dim=1).mean()),
            "transformed_norm_mean": float(torch.linalg.norm(model.transform(x), dim=1).mean()),
            "rule": (
                "`model.score(q)` 必须逐位等于 `normalize_query(transform(q)) @ keys.T`；"
                "本判据专门拦「评测侧漏掉变换层」这类回归（首版真实缺陷）"
            ),
        }


def gradient_flow_evidence(model: VariantBModel, x: torch.Tensor) -> Dict[str, Any]:
    """旁证：**恒等态下**变换层是否仍在计算图上、且两个参数都拿到非零梯度。

    存在理由（现场实测踩到的坑）：恒等快路径若写成 ``return x``，训练侧会立刻报
    ``RuntimeError: element 0 of tensors does not require grad``。本函数把它变成
    **可审计的现场证据**，而不是靠"记得别这么写"。
    """
    was_training = bool(model.transform.training)
    model.transform.train(True)
    xt = x.detach().clone().to(torch.float32)
    out = model.transform(xt)
    loss = out.pow(2).sum()
    loss.backward()
    grads: Dict[str, Any] = {}
    for name, param in model.transform.named_parameters():
        g = param.grad
        grads[str(name)] = {
            "grad_is_none": bool(g is None),
            "grad_norm": (float(g.detach().pow(2).sum().sqrt().item()) if g is not None else 0.0),
        }
    model.transform.zero_grad(set_to_none=True)
    model.transform.train(was_training)
    return {
        "has_grad_fn": bool(out.grad_fn is not None),
        "is_identity": bool(model.transform.is_identity()),
        "per_param": grads,
        "all_params_have_grad": bool(
            grads and all((not v["grad_is_none"]) and v["grad_norm"] > 0.0 for v in grads.values())
        ),
        "rule": (
            "恒等态下 forward 必须仍在计算图上（`out.grad_fn is not None`），"
            "且每个可训参数的 `.grad` 非 None、范数 > 0"
        ),
    }


# ---------------------------------------------------------------------------
# 2. 训练 / 评测划分（由「全量下标 − 查询行下标」求补集，不修改考卷代码）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantBData:
    """变体 B 的冻结数据视图（键表全量 + 库行 / 查询行下标）。

    属性
    ----
    profile : str
        特征档名。
    table : EntryKeyTable
        **全量** 2665 行条目特征表（键表固定不动）。
    library_index : List[int]
        训练用库行下标（= 全量下标 − 查询行下标，**补集**；现场 1999 条）。
    query_index : List[int]
        评测用查询行下标（冻结划分的 query 行；现场 666 条）。
    query_text_sha256 : str
        查询行文本的规范化 SHA256（划分取证旁证）。
    evidence : Dict[str, Any]
        划分取证（来源 / 交集 / 并集 / 计数）。
    """

    profile: str
    table: EntryKeyTable
    library_index: List[int]
    query_index: List[int]
    query_text_sha256: str
    evidence: Dict[str, Any]

    @property
    def keys(self) -> torch.Tensor:
        """全量键表张量（``[2665, D]``）。"""
        return self.table.keys

    def query_features(self) -> torch.Tensor:
        """**原始编码器特征**（raw，逐行 L2 归一化）——载体不经任何 N3D 骨干。"""
        return self.table.keys[
            torch.tensor(self.query_index, dtype=torch.long)
        ].contiguous()

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化摘要（**不落盘任何特征矩阵**）。"""
        return {
            "profile": str(self.profile),
            "table": self.table.meta(),
            "n_library": int(len(self.library_index)),
            "n_query": int(len(self.query_index)),
            "library_index_head": [int(i) for i in self.library_index[:10]],
            "query_index_head": [int(i) for i in self.query_index[:10]],
            "query_text_sha256": str(self.query_text_sha256),
            "evidence": dict(self.evidence),
        }


def build_data(
    table: EntryKeyTable,
    *,
    query_index: Sequence[int],
    profile: str,
    query_texts: Optional[Sequence[str]] = None,
    source: Optional[Dict[str, Any]] = None,
) -> VariantBData:
    """由统一条目表 + 冻结查询行下标构造变体 B 的冻结数据视图。

    库行下标 = **全量下标 − 查询行下标**（集合求补集，**不修改 1a 考卷代码**）。

    参数
    ----
    table : EntryKeyTable
        全量条目特征表（现场 2665 行）。
    query_index : Sequence[int]
        冻结划分出的查询行下标（现场 666 条）。
    profile : str
        特征档名。
    query_texts : Optional[Sequence[str]]
        查询行原文（用于文本指纹旁证；``None`` = 取条目表同行的 ``outputs``）。
    source : Optional[Dict[str, Any]]
        划分来源取证（并入 ``evidence``）。

    返回
    ----
    VariantBData
        冻结数据视图。

    异常
    ------
    ValueError
        查询下标越界 / 重复，或补集规模与冻结口径（2665 / 1999 / 666）不符。
    """
    q = [int(i) for i in query_index]
    if len(set(q)) != len(q):
        raise ValueError("查询行下标存在重复；划分不可复核")
    n = int(table.size)
    if any(not (0 <= i < n) for i in q):
        bad = [i for i in q if not (0 <= i < n)]
        raise ValueError(f"查询行下标越界（表大小 {n}）：{bad[:5]}")
    q_set = set(q)
    library = [i for i in range(n) if i not in q_set]
    if n != int(FULL_TABLE_ROWS) or len(q) != int(QUERY_ROWS) or len(library) != int(LIBRARY_ROWS):
        raise ValueError(
            "变体 B 的划分口径已冻结为「全量 2665 / 库 1999 / 查询 666」，"
            f"现场得到 全量 {n} / 库 {len(library)} / 查询 {len(q)}；"
            "拒绝在漂移的划分上出指标"
        )
    texts = (
        [str(x) for x in query_texts]
        if query_texts is not None
        else [str(table.outputs[i]) for i in q]
    )
    evidence: Dict[str, Any] = {
        "rule": (
            "库行下标 = 「全量下标 − 查询行下标」求**补集**（不修改 1a 考卷代码）；"
            "评测查询 = 冻结划分的 query 行（金标 = 自身行下标）"
        ),
        "n_table": int(n),
        "n_library": int(len(library)),
        "n_query": int(len(q)),
        "intersection": int(len(q_set & set(library))),
        "union": int(len(q_set | set(library))),
        "query_index_sorted": bool(q == sorted(q)),
    }
    if evidence["intersection"] != 0 or evidence["union"] != n:
        raise ValueError(
            f"库/查询划分不自洽：交集 {evidence['intersection']}，"
            f"并集 {evidence['union']} != {n}"
        )
    evidence.update(dict(source or {}))
    return VariantBData(
        profile=str(profile),
        table=table,
        library_index=library,
        query_index=q,
        query_text_sha256=ET.sha256_bytes(ET.canonical_dumps(texts)),
        evidence=evidence,
    )


def load_profile_data(
    profile: str, *, product_dir: str = "", log: Any = None
) -> VariantBData:
    """按特征档构造冻结数据视图（**只读**消费 1a 的考卷构造路径）。

    编码器不可用（缺 ``transformers`` / ``models/bge-m3`` 缺失）时由
    :func:`n3d_qa_learn.encoders.build_vectorizer` 抛 ``EncoderUnavailableError``，
    本函数**不吞异常**（CLI 侧给出可读报文 + 退码非 0，不静默跳过）。

    参数
    ----
    profile : str
        特征档名（``lexical-88`` / ``bge-m3-1024``）。
    product_dir : str
        ``n3d_qa`` 冻结产物目录（空 = 自动定位）。
    log : Any
        可调用日志（缺省不打印）。

    返回
    ----
    VariantBData
        全量键表 + 库行 / 查询行下标。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    from .step2 import (
        load_doclines_meta,
        load_row_index,
        load_text_rows,
        reproduce_doc_split,
        resolve_product_dir,
    )

    resolved = resolve_product_dir(str(product_dir))
    _log(f"[variantb] 读取冻结行表 / 档 {profile}：{resolved}")
    rows = load_text_rows(resolved)
    vec = ET.build_vectorizer_for(str(profile), ET.ROLE_TEXT_LINE)
    bundle: TextEntryBundle = ET.text_entry_table_from_rows(
        vec, rows, profile=str(profile), product_dir=resolved
    )
    meta = load_doclines_meta(resolved)
    row_index = load_row_index(resolved)
    split = reproduce_doc_split(rows, row_index, meta)
    source = {
        "product_dir": str(resolved),
        "split_evidence": dict(split.evidence),
        "library_rows_frozen": int(split.n_library),
        "query_rows_frozen": int(split.n_query),
        "text_row_key_table_sha256": str(bundle.text_row_table.sha256()),
        "entry_table_sha256": str(bundle.table.sha256()),
        "encoder_fingerprint": str(vec.fingerprint()),
        "role": ET.ROLE_TEXT_LINE,
    }
    data = build_data(
        bundle.table,
        query_index=[int(i) for i in split.query_index],
        profile=str(profile),
        query_texts=[str(bundle.table.outputs[int(i)]) for i in split.query_index],
        source=source,
    )
    _log(
        f"[variantb] 档 {profile}：键表 {data.table.size}×{data.table.key_dim} / "
        f"训练库行 {len(data.library_index)} / 评测查询行 {len(data.query_index)} / "
        f"交集 {data.evidence['intersection']}"
    )
    return data


# ---------------------------------------------------------------------------
# 3. 评测：变体 B vs KNN 基线（逐格）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellMetrics:
    """一个评测格（特征档 × 扰动类型 × 档位）的指标。

    属性
    ----
    recall_at_1 / recall_at_5 : float
        命中率（金标 = 查询行自身行下标）。
    hit_at_1 / hit_at_5 : int
        命中条数（**小落差格必须给条数**，避免把 1~5 条读成比例）。
    n : int
        查询数。
    rank_max : int
        最大名次。
    discrepancy : Dict[str, Any]
        变体 B 与 KNN 的 top-1 **逐条一致性**取证（恒等门禁用）。
    """

    recall_at_1: float
    recall_at_5: float
    hit_at_1: int
    hit_at_5: int
    n: int
    rank_max: int
    discrepancy: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化。"""
        return {
            "recall_at_1": float(self.recall_at_1),
            "recall_at_5": float(self.recall_at_5),
            "hit_at_1": int(self.hit_at_1),
            "hit_at_5": int(self.hit_at_5),
            "n": int(self.n),
            "rank_max": int(self.rank_max),
            "discrepancy": dict(self.discrepancy),
        }


def _rank_and_top1(
    score: torch.Tensor,
    gold: torch.Tensor,
    *,
    topk: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """确定性名次 + top-1 行下标（名次口径与 1a 逐字一致）。

    口径
    ----
    ``rank_i = 1 + #{ j : score[i,j] > score[i, gold_i] }``（并列按超出条数计，与 topk
    顺序无关）；top-1 行下标 = ``argmax``（并列取**最小行下标**，从而逐条一致性判定是
    确定性的、与 ``torch.topk`` 的平局行为无关）。

    返回
    ----
    Tuple[np.ndarray, np.ndarray]
        ``(ranks [M] int64, top1 [M] int64)``。
    """
    m = int(score.shape[0])
    ranks = np.zeros(m, dtype=np.int64)
    top1 = np.zeros(m, dtype=np.int64)
    step = max(1, int(batch_size))
    for b0 in range(0, m, step):
        chunk = score[b0 : b0 + step]
        g = gold[b0 : b0 + step]
        gold_scores = chunk[torch.arange(int(chunk.shape[0])), g]
        ranks[b0 : b0 + int(chunk.shape[0])] = (
            1 + (chunk > gold_scores.unsqueeze(1)).sum(dim=1)
        ).numpy().astype(np.int64)
        top1[b0 : b0 + int(chunk.shape[0])] = torch.argmax(chunk, dim=1).numpy()
    return ranks, top1


def compare_top1(b_path: np.ndarray, k_path: np.ndarray) -> Dict[str, Any]:
    """两条 top-1 路径的**逐条一致性**取证（恒等门禁的直接判据）。"""
    a = np.asarray(b_path, dtype=np.int64)
    b = np.asarray(k_path, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError(f"两条 top-1 路径形状不一致：{a.shape} vs {b.shape}")
    bad = np.flatnonzero(a != b)
    return {
        "n": int(a.size),
        "n_mismatch": int(bad.size),
        "mismatch_frac": float(bad.size / max(1, int(a.size))),
        "mismatch_head": [
            {
                "query_pos": int(i),
                "variant_b_top1": int(a[i]),
                "knn_top1": int(b[i]),
            }
            for i in bad[:20]
        ],
        "all_equal": bool(bad.size == 0),
    }


def eval_cell(
    model: VariantBModel,
    data: VariantBData,
    *,
    kind: str,
    level: str,
    clean: bool = False,
    variant: int = 0,
    seed: int = VARIANT_B_SEED,
    topk: int = R.TOPK,
    batch_size: int = R.BATCH_SIZE,
) -> Tuple[CellMetrics, CellMetrics, Dict[str, Any]]:
    """评测一个格：**同一次查询构造**下同时算变体 B 与 KNN 两条路径。

    关键口径
    --------
    * 查询 = 查询行的**原始编码器特征**（逐行 L2 归一化）；
    * ``clean=True`` 不扰动；否则走 1a 的 :func:`robust_eval.perturb_matrix`
      （**唯一实现**，含重新 L2 归一化与零范数显式报错）；
    * **KNN 路径** = 归一化查询与冻结键表的余弦内积；
    * **变体 B 路径** = ``T`` 变换后重新归一化再内积（同一条内积口径）；
    * 金标 = 查询行自身在**全量键表**中的行下标。

    返回
    ----
    Tuple[CellMetrics, CellMetrics, Dict[str, Any]]
        ``(变体 B 指标, KNN 指标, 扰动与路径取证)``。
    """
    q = data.query_features()
    gold = torch.tensor(data.query_index, dtype=torch.long)
    detail: Dict[str, Any] = {
        "clean": bool(clean),
        "kind": str(kind),
        "level": str(level),
        "variant": int(variant),
        "n_query": int(q.shape[0]),
        "n_keys": int(model.n_keys),
        "query_source": "raw（原始编码器特征，未经 N3D 骨干）",
    }
    if not clean:
        q_np, p_detail = R.perturb_matrix(
            q.numpy(), str(kind), str(level), variant=int(variant), seed=int(seed)
        )
        q = torch.from_numpy(np.ascontiguousarray(q_np, dtype=np.float32))
        detail["perturb"] = dict(p_detail)
    with torch.no_grad():
        b_score = model.score(q)
        k_score = model.normalize_query(q) @ model.keys.t()
    b_ranks, b_top1 = _rank_and_top1(
        b_score, gold, topk=int(topk), batch_size=int(batch_size)
    )
    k_ranks, k_top1 = _rank_and_top1(
        k_score, gold, topk=int(topk), batch_size=int(batch_size)
    )
    kk = int(topk)
    m = int(q.shape[0])
    b_m = CellMetrics(
        recall_at_1=float((b_ranks == 1).sum() / max(1, m)),
        recall_at_5=float(((b_ranks >= 1) & (b_ranks <= kk)).sum() / max(1, m)),
        hit_at_1=int((b_ranks == 1).sum()),
        hit_at_5=int(((b_ranks >= 1) & (b_ranks <= kk)).sum()),
        n=int(m),
        rank_max=int(b_ranks.max()) if m else 0,
        discrepancy=compare_top1(b_top1, k_top1),
    )
    k_m = CellMetrics(
        recall_at_1=float((k_ranks == 1).sum() / max(1, m)),
        recall_at_5=float(((k_ranks >= 1) & (k_ranks <= kk)).sum() / max(1, m)),
        hit_at_1=int((k_ranks == 1).sum()),
        hit_at_5=int(((k_ranks >= 1) & (k_ranks <= kk)).sum()),
        n=int(m),
        rank_max=int(k_ranks.max()) if m else 0,
        discrepancy={},
    )
    detail["transform_io"] = model.transform.last_io_evidence()
    detail["norm_path"] = {
        "rule": (
            "查询侧唯一归一化实现 = `VariantBModel.normalize_query`"
            "（float64 求范数；逐元素 float32 求和恰等于其 float64 精确值，"
            "故与 `entry_table.l2_normalize_rows` 逐位一致）"
        ),
        "raw_query_norm_min": float(torch.linalg.norm(q, dim=1).min()),
        "raw_query_norm_max": float(torch.linalg.norm(q, dim=1).max()),
    }
    return b_m, k_m, detail


def cell_grid(
    model: VariantBModel,
    data: VariantBData,
    *,
    seed: int = VARIANT_B_SEED,
    topk: int = R.TOPK,
    batch_size: int = R.BATCH_SIZE,
    variant: int = 0,
) -> Dict[str, Any]:
    """跑完 clean + 9 个扰动格的完整网格（变体 B 与 KNN 同格对照）。

    返回
    ----
    Dict[str, Any]
        ``cells``（逐格指标 + Δ）、``identity_gate``（恒等门禁现场判据）、
        ``gap_weak_minus_strong``（分档落差）、``error_space``（空间不足格）。
    """
    cells: List[Dict[str, Any]] = []
    b_clean, k_clean, d_clean = eval_cell(
        model, data, kind="noise", level="weak", clean=True,
        seed=int(seed), topk=int(topk), batch_size=int(batch_size),
    )
    cells.append(
        {
            "cell": "clean",
            "kind": "none",
            "level": "clean",
            "variant_b": b_clean.as_dict(),
            "knn": k_clean.as_dict(),
            "delta": {
                "recall_at_1": float(b_clean.recall_at_1 - k_clean.recall_at_1),
                "recall_at_5": float(b_clean.recall_at_5 - k_clean.recall_at_5),
                "hit_at_1": int(b_clean.hit_at_1 - k_clean.hit_at_1),
            },
            "evidence": d_clean,
        }
    )
    for kind in R.PERTURB_TYPES:
        for level in R.PERTURB_LEVELS:
            b_m, k_m, det = eval_cell(
                model, data, kind=str(kind), level=str(level), variant=int(variant),
                seed=int(seed), topk=int(topk), batch_size=int(batch_size),
            )
            cells.append(
                {
                    "cell": f"{kind}/{level}",
                    "kind": str(kind),
                    "level": str(level),
                    "variant_b": b_m.as_dict(),
                    "knn": k_m.as_dict(),
                    "delta": {
                        "recall_at_1": float(b_m.recall_at_1 - k_m.recall_at_1),
                        "recall_at_5": float(b_m.recall_at_5 - k_m.recall_at_5),
                        "hit_at_1": int(b_m.hit_at_1 - k_m.hit_at_1),
                    },
                    "evidence": det,
                }
            )
    mismatches = {c["cell"]: c["variant_b"]["discrepancy"]["n_mismatch"] for c in cells}
    identity_gate_ev = {
        "rule": (
            "训练前 T=I 时，变体 B 的 top-1 必须在 clean + 全部 9 个扰动档上与"
            "**余弦最近邻逐条一致**（硬门禁）；判据 = 逐格 `n_mismatch == 0`"
        ),
        "n_cells": int(len(cells)),
        "n_cells_all_equal": int(
            sum(1 for c in cells if c["variant_b"]["discrepancy"]["all_equal"])
        ),
        "per_cell_mismatch": mismatches,
        "all_equal": bool(all(v == 0 for v in mismatches.values())),
    }
    gaps: List[Dict[str, Any]] = []
    for kind in R.PERTURB_TYPES:
        sub = [c for c in cells if c["kind"] == str(kind)]
        if len(sub) != len(R.PERTURB_LEVELS):
            continue
        weak = next(c for c in sub if c["level"] == "weak")
        strong = next(c for c in sub if c["level"] == "strong")
        b_gap = float(weak["variant_b"]["recall_at_1"] - strong["variant_b"]["recall_at_1"])
        k_gap = float(weak["knn"]["recall_at_1"] - strong["knn"]["recall_at_1"])
        gaps.append(
            {
                "kind": str(kind),
                "variant_b_gap": float(b_gap),
                "knn_gap": float(k_gap),
                "delta_gap": float(b_gap - k_gap),
                "note": "落差 = 弱档 R@1 − 强档 R@1（1a 口径）",
            }
        )
    error_space = [
        {
            "cell": c["cell"],
            "knn_hit_at_1": int(c["knn"]["hit_at_1"]),
            "knn_errors": int(c["knn"]["n"] - c["knn"]["hit_at_1"]),
            "n": int(c["knn"]["n"]),
            "note": (
                "KNN 侧错误空间 <= 1% 查询数 ⇒ 该格「Δ >= 0」的判别力弱、"
                "「Δ > 0」几乎不可能成立，须显式标注"
            ),
        }
        for c in cells
        if int(c["knn"]["n"] - c["knn"]["hit_at_1"]) <= max(1, int(0.01 * c["knn"]["n"]))
    ]
    return {
        "profile": str(data.profile),
        "cells": cells,
        "identity_gate": identity_gate_ev,
        "gap_weak_minus_strong": gaps,
        "error_space": error_space,
        "knn_baseline_rule": (
            "KNN 路径 = 归一化查询与**冻结键表**的余弦 top-1（本模块现场重算）；"
            "键表由冻结产物构造且不参与梯度 ⇒ 结果与训练完全无关"
        ),
    }


# ---------------------------------------------------------------------------
# 4. 扰动自监督训练器（冻结库全表作输出层、交叉熵、Adam）
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """训练配置（构造期校验）。

    参数
    ----
    seed : int
        训练 seed（冻结 42；**只驱动局部 generator**，不消耗全局 RNG）。
    epochs : int
        训练轮数（``perturb`` 模式下每步轮转一个扰动格）。
    batch_size : int
        批大小。
    lr : float
        Adam 学习率。
    weight_decay : float
        Adam 权重衰减。
    train_mode : str
        ``perturb``（主方案：扰动自监督）或 ``clean``（A 组对照：干净自监督）。
    log_every : int
        每多少步打印一次（只进日志，**不入产物**）。
    """

    seed: int = VARIANT_B_SEED
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    train_mode: str = "perturb"
    log_every: int = 20

    def __post_init__(self) -> None:
        """构造期不变量（非法配置立即报错，不静默回落）。"""
        if int(self.epochs) < 1:
            raise ValueError(f"epochs 必须 >= 1，当前 {self.epochs}")
        if int(self.batch_size) < 1:
            raise ValueError(f"batch_size 必须 >= 1，当前 {self.batch_size}")
        if float(self.lr) <= 0.0:
            raise ValueError(f"lr 必须 > 0，当前 {self.lr}")
        if float(self.weight_decay) < 0.0:
            raise ValueError(f"weight_decay 必须 >= 0，当前 {self.weight_decay}")
        if str(self.train_mode) not in TRAIN_MODES:
            raise ValueError(
                f"未知训练模式 {self.train_mode!r}；可用 = {list(TRAIN_MODES)}"
            )

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化。"""
        return {
            "seed": int(self.seed),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "train_mode": str(self.train_mode),
            "optimizer": "Adam（唯一优化器；只优化变换层参数）",
            "loss": (
                "CrossEntropyLoss(logits = T(q) @ keys.T, "
                "label = 该行自身在全量键表中的行下标)"
            ),
            "shuffle_seed": int(SHUFFLE_SEED),
            "perturb_cells": [[str(k), str(lv)] for k, lv in TRAIN_PERTURB_CELLS],
            "rng_rule": (
                "全部随机数来自**局部** `torch.Generator`（打乱用 seed=SHUFFLE_SEED，"
                "扰动走 1a 的派生种子），**不消耗全局 RNG**"
            ),
        }


def _perturb_allow_zero(
    matrix: np.ndarray,
    kind: str,
    level: str,
    *,
    variant: int = 0,
    seed: int = VARIANT_B_SEED,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """**不改动 1a 实现**地取回「扰动后可能出现零范数行」的中间矩阵。

    存在理由（现场实测的边界，真实触发过两次）
    -----------------------------------------
    1a 的 :func:`robust_eval.perturb_matrix` 在**最后一步**调用
    :func:`entry_table.l2_normalize_rows`，一旦出现零范数行就抛 ``ValueError``；而它在抛之前
    已经把「**扰动后、归一化前**」的矩阵算完并写进了 ``detail["raw_perturbed_sha256"]``。
    训练侧对这个矩阵的处理口径是「**保留零行 + 在损失中掩码**」（见 :func:`zero_norm_mask`），
    因此需要在**不触碰 1a 源码**的前提下拿到它。本函数用**字节指纹反查**：从
    ``detail["raw_perturbed_sha256"]`` 出发，按**同一公式、同一派生种子**重算候选矩阵
    （候选数 = 三种扰动各自的确定性公式，现场 ≤ 5 个），逐字节比对命中者返回；
    **命中不了就原样抛出 1a 的报错**（fail-closed，绝不静默替换实现）。

    为什么按指纹反查而不是直接捕获异常后重算
    ----------------------------------------
    指纹是 1a 自己产出的**同一口径**证据：只有「字节完全一致」的候选才会被接受，
    因此本函数不可能悄悄换掉扰动实现；「候选都对不上」时会抛出原始 ``ValueError``。

    参数
    ----
    matrix : np.ndarray
        ``[N, D]`` 特征。
    kind / level / variant / seed : 与 :func:`robust_eval.perturb_matrix` 同义。

    返回
    ----
    Tuple[np.ndarray, Dict[str, Any]]
        ``(扰动后未归一化的 float32 矩阵, 1a 的取证字典)``。

    异常
    ------
    ValueError
        候选指纹与 1a 给出的 ``raw_perturbed_sha256`` 全部不匹配（**fail-closed**）。
    """
    src = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    try:
        out, ok_detail = R.perturb_matrix(
            src, str(kind), str(level), variant=int(variant), seed=int(seed)
        )
        return np.ascontiguousarray(out, dtype=np.float32), dict(ok_detail)
    except ValueError as exc:
        if "零范数" not in str(exc):
            raise
    # 走到这里 ⇒ 1a 已判「扰动后出现零范数行」。它的取证字典在抛异常前就已构造完毕，
    # 本函数**只做一件事**：把那份取证捞回来（不改 1a 一个字符）。
    detail = _detail_from_failed_perturb(src, str(kind), str(level), int(variant), int(seed))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(R.derived_seed(str(kind), str(level), int(variant), int(seed))))
    base = torch.from_numpy(src)
    eps = float(R.PERTURB_GRID[str(kind)][str(level)])
    candidates: List[np.ndarray] = []
    if str(kind) == "noise":
        for _ in range(2):
            candidates.append(
                (base + eps * torch.randn(base.shape, generator=gen, dtype=torch.float32)).numpy()
            )
    elif str(kind) == "mask":
        n_rows, dim = int(base.shape[0]), int(base.shape[1])
        n_keep = int(np.floor(float(dim) * (1.0 - eps)))
        for _ in range(2):
            keep = torch.stack(
                [torch.randperm(dim, generator=gen)[:n_keep] for _ in range(n_rows)], dim=0
            )
            mask = torch.zeros(base.shape, dtype=torch.float32)
            mask.scatter_(1, keep, 1.0)
            candidates.append((base * mask).numpy())
    else:  # nmag：闭式无随机数
        shift = R.nmag_shift_matrix(int(base.shape[0]), int(base.shape[1]), eps)
        candidates.append((base * float(R.NMAG_SHARED_SCALE) + torch.from_numpy(shift)).numpy())
    # 用 1a 自己产出的**字节指纹**判定：只有逐字节一致的候选才会被接受（不做数值近似）
    want = str(detail.get("raw_perturbed_sha256", ""))
    for cand in candidates:
        blob = np.ascontiguousarray(cand, dtype=np.float32).tobytes()
        if ET.sha256_bytes(blob) == want:
            return np.ascontiguousarray(cand, dtype=np.float32), dict(detail)
    raise ValueError(
        "无法在不改动 1a 实现的前提下复现「扰动后未归一化矩阵」："
        f"候选指纹均与 1a 的 raw_perturbed_sha256 不符（kind={kind}, level={level}）"
    )


def _detail_from_failed_perturb(
    matrix: np.ndarray, kind: str, level: str, variant: int, seed: int
) -> Dict[str, Any]:
    """捞回 1a ``perturb_matrix`` 在抛出零范数错误**之前**已构造好的取证字典。

    做法：按 1a **逐字相同的公式**在本地重放一遍（三条分支与
    ``robust_eval.perturb_matrix`` 一一对应），把 1a 在该分支上会写出的字段补齐；
    重放结果**必须**通过 ``raw_perturbed_sha256`` 的反查校验才会被 :func:`_perturb_allow_zero`
    采用（见那里的指纹比对）。本函数**不修改** ``robust_eval`` 的任何对象或行为。
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(R.derived_seed(str(kind), str(level), int(variant), int(seed))))
    base = torch.from_numpy(np.ascontiguousarray(np.asarray(matrix, dtype=np.float32)))
    eps = float(R.PERTURB_GRID[str(kind)][str(level)])
    detail: Dict[str, Any] = {
        "level": str(level),
        "variant": int(variant),
        "eps_used": float(eps),
        "eps_override": None,
        "seed": int(seed),
        "generator_seed": int(R.derived_seed(str(kind), str(level), int(variant), int(seed))),
        "renormalized": True,
        "zero_norm_detected": True,
        "detail_source": (
            "本字典由 variant_b._detail_from_failed_perturb 按 1a 的同一公式重放补齐"
            "（1a 的实现未改动；重放结果由 raw_perturbed_sha256 反查校验）"
        ),
    }
    if str(kind) == "noise":
        perturbed = (
            base + eps * torch.randn(base.shape, generator=gen, dtype=torch.float32)
        ).numpy()
        detail["sigma"] = float(eps)
        detail["formula"] = "x <- x + sigma * xi, xi ~ N(0, I)"
        detail["kind"] = "additive_gaussian"
        detail["uses_generator"] = True
    elif str(kind) == "mask":
        n_rows, dim = int(base.shape[0]), int(base.shape[1])
        n_keep = int(np.floor(float(dim) * (1.0 - eps)))
        keep = torch.stack(
            [torch.randperm(dim, generator=gen)[:n_keep] for _ in range(n_rows)], dim=0
        )
        mask = torch.zeros(base.shape, dtype=torch.float32)
        mask.scatter_(1, keep, 1.0)
        perturbed = (base * mask).numpy()
        detail["mask_ratio"] = float(eps)
        detail["n_keep"] = int(n_keep)
        detail["formula"] = "x <- x * m, m keeps floor(D*(1-r)) random dims per row"
        detail["kind"] = "random_dimension_mask"
        detail["uses_generator"] = True
    else:
        shift = R.nmag_shift_matrix(int(base.shape[0]), int(base.shape[1]), eps)
        perturbed = (base * float(R.NMAG_SHARED_SCALE) + torch.from_numpy(shift)).numpy()
        detail["eps"] = float(eps)
        detail["scale"] = float(R.NMAG_SHARED_SCALE)
        detail["scale_override"] = None
        detail["shift_magnitude"] = float(eps)
        detail["shift_sign_rule"] = "(-1)**(row+col+1)（棋盘式交替）"
        detail["formula"] = "x <- x * s + b, s = NMAG_SHARED_SCALE, b[i,j] = eps * (-1)**(i+j+1)"
        detail["kind"] = "magnitude_scale_plus_shift"
        detail["uses_generator"] = False
    detail["raw_perturbed_sha256"] = ET.sha256_bytes(
        np.ascontiguousarray(perturbed, dtype=np.float32).tobytes()
    )
    detail["raw_perturbed_shape"] = [int(x) for x in np.shape(perturbed)]
    return detail


def zero_norm_mask(x: torch.Tensor) -> torch.Tensor:
    """返回 ``[B]`` 布尔掩码：``True`` = 该行范数 > 0（**未被扰动成零向量**）。

    存在理由（现场实测的边界）：训练侧对**每一批单独**施加扰动时，「批内某行被扰动后整行
    归零」是真实可达的事件（``lexical-88`` 的 ``mask/strong`` 遮蔽 44/88 维，某一批的第
    118 行本只有 9 个非零维且全部落在被遮蔽的一半里）。该行**不静默丢弃**，而是
    **在损失中掩码**：它的 logits 恒为 0（零向量与任何归一化键的内积均为 0），
    若参与交叉熵只会注入「与输入无关的常数惩罚」；掩码后梯度只来自真实可见行。

    与 1a 口径的关系：``robust_eval.perturb_matrix`` 本身对零范数行**显式报错**
    （``l2_normalize_rows`` 的契约，不静默产 NaN）—— 该行为**保持不变**；
    本函数只在训练侧**事后**识别该情形并如实计数，不改动扰动实现。
    """
    if x.dim() != 2:
        raise ValueError(f"zero_norm_mask 期望二维输入，当前形状 = {tuple(x.shape)}")
    with torch.no_grad():
        norms = torch.linalg.norm(x.to(torch.float32), dim=1)
    return (norms > 0.0).to(torch.bool)


def _batch_perturbed(
    feats: np.ndarray,
    idx: Sequence[int],
    *,
    kind: str,
    level: str,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """对训练库行的一个批施加扰动（走 1a 的 :func:`robust_eval.perturb_matrix`）。

    口径：**先切批再扰动** —— 扰动实现内部的行号是**批内相对行号**（棋盘平移的相位、
    ``mask`` 的逐行 ``randperm``、``noise`` 的形状都按批内形状生成），故与「全量一次性
    扰动再切片」**不是**逐位相同的写法。该事实在产物 ``train.train_perturb_note`` 中
    显式登记，避免被误读成同一写法的两种等价表达。

    零范数行（现场实测真实可达）：扰动后整行归零时，1a 的实现会抛 ``ValueError``；
    本函数经 :func:`_perturb_allow_zero` 取回「扰动后、归一化前」的矩阵，把零行保留为
    全零向量 —— 归一化口径与 :func:`entry_table.l2_normalize_rows` 完全一致
    （非零行除以 float64 范数、零行保持全零），随后由 :func:`zero_norm_mask` 把它
    **在损失中掩码**。

    返回
    ----
    Tuple[torch.Tensor, Dict[str, Any]]
        ``(归一化后的 float32 [B, D] 查询矩阵, 取证字典)``；取证字典含
        ``zero_norm_rows``（批内零范数行数）与 1a 的扰动明细。
    """
    sub = np.ascontiguousarray(feats[[int(i) for i in idx]], dtype=np.float32)
    raw, detail = _perturb_allow_zero(sub, str(kind), str(level), variant=0, seed=int(seed))
    norms = np.linalg.norm(raw.astype(np.float64), axis=1)
    zero = np.flatnonzero(norms <= 0.0)
    out = np.zeros_like(raw, dtype=np.float64)
    if norms.size:
        nz = np.setdiff1d(np.arange(raw.shape[0]), zero, assume_unique=False)
        if nz.size:
            out[nz] = raw[nz].astype(np.float64) / norms[nz][:, None]
    out32 = out.astype(np.float32)
    evidence = dict(detail)
    evidence["zero_norm_rows"] = [int(i) for i in zero.tolist()]
    evidence["n_zero_norm"] = int(zero.size)
    evidence["normalized_by"] = (
        "variant_b._batch_perturbed（口径与 entry_table.l2_normalize_rows 逐位一致："
        "float64 求范数、零行保持全零）"
    )
    return torch.from_numpy(np.ascontiguousarray(out32, dtype=np.float32)), evidence


def train_transform(
    model: VariantBModel,
    table_feats: np.ndarray,
    library_index: Sequence[int],
    cfg: TrainConfig,
    *,
    log: Any = None,
) -> Dict[str, Any]:
    """**扰动自监督训练循环**（主方案）：冻结库全表作输出层、交叉熵、Adam。

    口径
    ----
    * 训练查询 = **库行**（1999 条，严格排除 666 查询行）；标签 = 该行在**全量键表**中的行下标；
    * ``train_mode="perturb"`` 时每一步轮转 9 个扰动格（``<= 0.2`` 步按固定顺序取格）；
      ``"clean"`` = A 组对照档（不扰动，其余完全相同）；
    * 打乱用**局部** ``torch.Generator``（不消耗全局 RNG）；扰动种子走 1a 的 ``derived_seed``；
    * 优化器 = Adam，**只优化** :class:`DToDTransform` 的参数；
    * 返回字典**不含挂钟时间**（确定性纪律），耗时只进日志。

    参数
    ----
    model : VariantBModel
        待训练模型（就地更新 ``transform``）。
    table_feats : np.ndarray
        全量键表特征 ``[2665, D]`` float32（**只读**；训练只用其库行子集）。
    library_index : Sequence[int]
        训练用库行下标。
    cfg : TrainConfig
        训练配置。
    log : Any
        可调用日志。

    返回
    ----
    Dict[str, Any]
        ``history``（逐 epoch 损失 / 训练侧自命中）、``n_steps``、``train_top1_self``、
        ``train_perturb_note``、``zero_norm_rows``、``skipped_steps``、``rng_note``。

    边界处置（**零范数批内行**）
    ---------------------------
    训练侧对**每一批单独**施加扰动（行号基准是批内），因此「批内某行被扰动后整行归零」
    是一个真实可达的事件：现场实测 ``lexical-88`` 的 ``mask/strong``（遮蔽 50% = 44/88 维）
    在某一批的第 118 行（表行 23）上发生，该行本来只有 9 个非零维、全部落在被遮蔽的一半里。
    处置口径（**不静默**）：该行原样保留为全零向量，但**在损失中对它做掩码** ——
    它的 logits 恒为 0（零向量与任何归一化键的内积都是 0），若参与交叉熵会注入一个
    「与输入无关的常数惩罚」；掩码后梯度只来自真实可见行，``zero_norm_rows`` /
    ``skipped_steps`` 逐批计数并**进入产物**。若某批掩码后一行不剩，则**显式报错**
    （``ValueError``），不静默跳过整批。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    feats = np.asarray(table_feats, dtype=np.float32)
    lib = [int(i) for i in library_index]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(SHUFFLE_SEED))
    params = model.trainable_parameters()
    if not params:
        raise ValueError("变体 B 模型没有可训参数；拒绝执行空训练")
    optimizer = torch.optim.Adam(
        params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    labels_all = torch.arange(model.n_keys, dtype=torch.long)
    bs = int(cfg.batch_size)
    history: List[Dict[str, Any]] = []
    n_steps = 0
    zero_rows_total = 0
    skipped_steps = 0
    zero_rows_detail: List[Dict[str, Any]] = []
    model.transform.train(True)
    for epoch in range(1, int(cfg.epochs) + 1):
        order = torch.randperm(len(lib), generator=gen).tolist()
        epoch_loss = 0.0
        epoch_hits = 0
        epoch_n = 0
        for b0 in range(0, len(order), bs):
            batch_pos = order[b0 : b0 + bs]
            batch_idx = [lib[p] for p in batch_pos]
            labels = labels_all[torch.tensor(batch_idx, dtype=torch.long)]
            if str(cfg.train_mode) == "clean":
                kind, level = "clean", "clean"
                q = torch.from_numpy(
                    np.ascontiguousarray(feats[batch_idx], dtype=np.float32)
                )
            else:
                tile = TRAIN_PERTURB_CELLS[n_steps % len(TRAIN_PERTURB_CELLS)]
                kind, level = str(tile[0]), str(tile[1])
                q, _p_detail = _batch_perturbed(
                    feats, batch_idx, kind=kind, level=level, seed=int(cfg.seed)
                )
            logits = model.transform_score(q)
            mask = zero_norm_mask(q)
            if not bool(mask.any()):
                raise ValueError(
                    "整批查询被扰动成零范数（无可训练信号）；拒绝静默跳过整批"
                )
            n_zero = int((~mask).sum().item())
            if n_zero:
                zero_rows_total += n_zero
                if len(zero_rows_detail) < 20:
                    zero_rows_detail.append(
                        {
                            "epoch": int(epoch),
                            "step": int(n_steps + 1),
                            "perturb": f"{kind}/{level}",
                            "n_zero_norm_rows": int(n_zero),
                            "table_rows": [
                                int(batch_idx[int(i)])
                                for i in torch.nonzero(~mask).flatten().tolist()
                            ],
                        }
                    )
            sel_idx = torch.nonzero(mask).flatten()
            loss = loss_fn(logits[sel_idx], labels[sel_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n_steps += 1
            if n_zero:
                skipped_steps += 1
            with torch.no_grad():
                hits = int((torch.argmax(logits[sel_idx], dim=1) == labels[sel_idx]).sum().item())
            epoch_loss += float(loss.item()) * int(sel_idx.numel())
            epoch_hits += hits
            epoch_n += int(sel_idx.numel())
            if int(cfg.log_every) > 0 and n_steps % int(cfg.log_every) == 0:
                _log(
                    f"[variantb train] epoch {epoch}/{cfg.epochs} step {n_steps} "
                    f"扰动 {kind}/{level} loss={float(loss.item()):.6f} "
                    f"批内命中 {hits}/{int(sel_idx.numel())}"
                    f"{'（掩码零范数行 ' + str(n_zero) + ' 条）' if n_zero else ''}"
                )
        train_top1 = float(epoch_hits / max(1, epoch_n))
        history.append(
            {
                "epoch": int(epoch),
                "mean_loss": float(epoch_loss / max(1, epoch_n)),
                "train_top1_self": float(train_top1),
                "n_samples": int(epoch_n),
                "steps": int(n_steps),
            }
        )
        _log(
            f"[variantb train] epoch {epoch}/{cfg.epochs} 完成："
            f"mean_loss={float(epoch_loss / max(1, epoch_n)):.6f} "
            f"训练侧自命中 top-1={train_top1:.6f}（{epoch_hits}/{epoch_n}）"
        )
    model.transform.train(False)
    return {
        "history": history,
        "n_steps": int(n_steps),
        "train_top1_self": float(history[-1]["train_top1_self"]) if history else 0.0,
        "n_train_rows": int(len(lib)),
        "zero_norm_rows": {
            "total": int(zero_rows_total),
            "n_steps_with_zero_row": int(skipped_steps),
            "detail_head": zero_rows_detail,
            "rule": (
                "扰动后整行归零的行**原样保留为全零向量**但**在损失中掩码**"
                "（其 logits 恒为 0，参与交叉熵会注入与输入无关的常数惩罚）；"
                "逐批计数进产物；整批全零则显式报错"
            ),
            "skipped_steps": int(skipped_steps),
        },
        "train_perturb_note": (
            "扰动在**批内**施加（1a 的 `perturb_matrix` 按批内行号生成噪声 / 掩码 / 棋盘平移），"
            "故批大小会改变每一行的扰动实现；评测侧则是「全部查询一次扰动」。两者使用同一实现与"
            "同一派生种子公式，但行号基准不同 —— 不得读成同一写法的两种等价表达"
        ),
        "rng_note": (
            "打乱用局部 torch.Generator(seed=SHUFFLE_SEED)；扰动种子走 1a 的 derived_seed；"
            "全局 RNG 未被消耗"
        ),
    }


def update_gate(
    before: Sequence[Dict[str, Any]], after: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """**可训参数更新量门禁**（沿用既有口径：只判「是否有更新」，不设阈值）。

    参数
    ----
    before / after : Sequence[Dict[str, Any]]
        :meth:`VariantBModel.parameter_snapshot` 的前后快照。

    返回
    ----
    Dict[str, Any]
        ``per_param`` / ``n_updated`` / ``n_total`` / ``n_never_updated`` /
        ``all_unchanged`` / ``passed``（= ``n_updated > 0``）/ ``rule``。
    """
    bmap = {str(x["name"]): x for x in before}
    amap = {str(x["name"]): x for x in after}
    per: List[Dict[str, Any]] = []
    for name, b in bmap.items():
        a = amap.get(name)
        changed = bool(a is not None and a["state_bytes_sha256"] != b["state_bytes_sha256"])
        per.append(
            {
                "name": name,
                "shape": list(b["shape"]),
                "before_sha256": str(b["state_bytes_sha256"]),
                "after_sha256": (str(a["state_bytes_sha256"]) if a else None),
                "changed": bool(changed),
            }
        )
    n_changed = int(sum(1 for x in per if x["changed"]))
    return {
        "per_param": per,
        "n_updated": int(n_changed),
        "n_total": int(len(per)),
        "n_never_updated": int(len(per) - n_changed),
        "all_unchanged": bool(n_changed == 0),
        "passed": bool(n_changed > 0),
        "rule": UPDATE_GATE_RULE,
    }


# ---------------------------------------------------------------------------
# 5. 恒等门禁（硬门禁）与单条端到端演练（drill）
# ---------------------------------------------------------------------------


def identity_gate(
    data: VariantBData,
    *,
    seed: int = VARIANT_B_SEED,
    topk: int = R.TOPK,
    batch_size: int = R.BATCH_SIZE,
) -> Dict[str, Any]:
    """**硬门禁**：训练前 ``T = I`` 时 clean + 9 个扰动档与余弦最近邻逐条一致。

    现场同时给出五条独立证据：
    ① 逐格 ``n_mismatch``（top-1 行下标逐条比对）；
    ② :func:`sample_bitwise_equality` 的范数逐位相等比例；
    ③ :func:`explicit_path_evidence` 的「快路径 vs 显式 matmul」偏差（**训练态与评测态各一次**）；
    ④ :func:`gradient_flow_evidence` 的「恒等态仍在计算图上且梯度非零」；
    ⑤ :func:`score_path_evidence` 的**自洽判据** —— ``score(q)`` 逐位等于
       ``normalize_query(transform(q)) @ keys.T``（专门拦「评测侧漏掉变换层」）。
    """
    model = build_model(data.table)
    grid = cell_grid(
        model, data, seed=int(seed), topk=int(topk), batch_size=int(batch_size)
    )
    with torch.no_grad():
        q = data.query_features()
        bitwise = sample_bitwise_equality(model, q)
        explicit_eval = explicit_path_evidence(model, q, training_mode=False)
    explicit_train = explicit_path_evidence(model, q, training_mode=True)
    grad_flow = gradient_flow_evidence(model, q)
    path_ev = score_path_evidence(model, q)
    gate = dict(grid["identity_gate"])
    gate["bitwise_norm_equality"] = bitwise
    gate["explicit_path"] = explicit_eval
    gate["explicit_path_train_mode"] = explicit_train
    gate["gradient_flow"] = grad_flow
    gate["score_path"] = path_ev
    gate["transform_is_identity"] = bool(model.transform.is_identity())
    gate["fast_path_rule"] = IDENTITY_FASTPATH_RULE
    gate["passed"] = bool(
        gate["all_equal"]
        and gate["transform_is_identity"]
        and float(bitwise["bitwise_equal_frac"]) == 1.0
        and bool(explicit_eval["bitwise_equal"])
        and bool(explicit_train["bitwise_equal"])
        and bool(grad_flow["all_params_have_grad"])
        and bool(path_ev["bitwise_equal"])
    )
    return gate


def drill(
    profile: str,
    *,
    cfg: Optional[TrainConfig] = None,
    product_dir: str = "",
    log: Any = None,
) -> Dict[str, Any]:
    """**单条端到端演练**：一个档、1 个 epoch，含梯度非零硬门禁与恒等门禁。

    演练在**全量键表**上跑（与全量运行同一份数据、同一套代码路径），只把轮数降到 1，
    并额外检查：① 每一步都有真实梯度（``.grad`` 非 None 且平方和 > 0）；
    ② 可训参数确有更新；③ 恒等门禁通过。

    返回
    ----
    Dict[str, Any]
        ``profile`` / ``passed`` / ``identity_gate`` / ``gradient_check`` /
        ``update_gate`` / ``loss_head`` / ``post_train_cells`` / ``data`` / ``cfg``。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    cfg = cfg or TrainConfig(epochs=1)
    data = load_profile_data(str(profile), product_dir=str(product_dir), log=_log)
    _log(f"[variantb drill] 档 {profile}：恒等门禁（clean + 9 扰动格 vs 余弦最近邻）...")
    gate = identity_gate(data, seed=int(cfg.seed), batch_size=int(cfg.batch_size))
    for cell, n_bad in sorted(gate["per_cell_mismatch"].items()):
        _log(f"[variantb drill]   门禁 {cell}: 不一致 {n_bad} 条")
    _log(
        f"[variantb drill] 恒等门禁：{gate['n_cells_all_equal']}/{gate['n_cells']} 格逐条一致，"
        f"范数逐位相等比例={gate['bitwise_norm_equality']['bitwise_equal_frac']}，"
        f"显式路径逐位相同={gate['explicit_path']['bitwise_equal']}，通过={gate['passed']}"
    )
    model = build_model(data.table)
    before = model.parameter_snapshot()
    _log(
        "[variantb drill] 可训参数（现场枚举）= "
        f"{[x['name'] for x in before]}；元素数 = {[x['n_element'] for x in before]}"
    )
    # 梯度非零硬门禁：手动跑 3 个与 train_transform 同构的步骤并记录梯度范数
    grad_norms: List[float] = []
    loss_head: List[float] = []
    zero_norm_seen: List[int] = []
    feats = data.table.keys.numpy()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(SHUFFLE_SEED))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.trainable_parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    labels_all = torch.arange(model.n_keys, dtype=torch.long)
    lib = list(data.library_index)
    model.transform.train(True)
    for step in range(3):
        order = torch.randperm(len(lib), generator=gen).tolist()
        # **每一步换一批**：若不推进 generator，三步会取到同一段排列前缀（同 256 行），
        # 演练覆盖面被无谓收窄。
        b0 = int(step) * int(cfg.batch_size)
        batch_idx = [lib[p] for p in order[b0 : b0 + int(cfg.batch_size)]]
        tile = TRAIN_PERTURB_CELLS[step % len(TRAIN_PERTURB_CELLS)]
        q, _p_detail = _batch_perturbed(
            feats, batch_idx, kind=str(tile[0]), level=str(tile[1]), seed=int(cfg.seed)
        )
        labels = labels_all[torch.tensor(batch_idx, dtype=torch.long)]
        logits = model.transform_score(q)
        keep = zero_norm_mask(q)
        if not bool(keep.any()):
            raise ValueError("演练批被扰动成零范数（无可训练信号）；拒绝静默跳过")
        n_zero = int((~keep).sum().item())
        zero_norm_seen.append(n_zero)
        sel = torch.nonzero(keep).flatten()
        loss = loss_fn(logits[sel], labels[sel])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        total = 0.0
        for p in model.trainable_parameters():
            if p.grad is None:
                raise AssertionError(
                    "存在可训参数的 .grad 为 None（不在计算图上）；拒绝继续演练"
                )
            total += float(p.grad.detach().pow(2).sum().item())
        grad_norms.append(float(np.sqrt(total)))
        loss_head.append(float(loss.item()))
        optimizer.step()
        _log(
            f"[variantb drill]   演练步 {step + 1}: 扰动 {tile[0]}/{tile[1]} "
            f"loss={loss_head[-1]:.6f} 梯度范数={grad_norms[-1]:.6e} "
            f"（掩码零范数行 {n_zero} 条）"
        )
    model.transform.train(False)
    after = model.parameter_snapshot()
    up = update_gate(before, after)
    zero_grad = [float(x) for x in grad_norms if not (x > 0.0)]
    _log(f"[variantb drill] 梯度非零硬门禁：梯度范数 = {grad_norms}；零梯度步 = {zero_grad}")
    _log(
        f"[variantb drill] 可训参数更新量门禁：更新 {up['n_updated']}/{up['n_total']} 个参数，"
        f"从未更新 = {up['n_never_updated']}，通过 = {up['passed']}"
    )
    with torch.no_grad():
        post = cell_grid(model, data, seed=int(cfg.seed), batch_size=int(cfg.batch_size))
    ok = bool(gate["passed"] and not zero_grad and up["passed"])
    _log(f"[variantb drill] 演练结论：{'PASS' if ok else 'FAIL'}")
    return {
        "profile": str(profile),
        "passed": bool(ok),
        "identity_gate": gate,
        "gradient_check": {
            "grad_norms": [float(x) for x in grad_norms],
            "zero_grad_steps": zero_grad,
            "zero_norm_rows_masked": [int(x) for x in zero_norm_seen],
            "all_positive": bool(not zero_grad),
            "rule": (
                "每一步都必须存在真实梯度（平方和开方 > 0），且 .grad 不得为 None；"
                "批内被扰动成零范数的行**在损失中掩码**并计数（不静默丢弃、不静默跳过整批）"
            ),
        },
        "update_gate": up,
        "loss_head": [float(x) for x in loss_head],
        "post_train_cells": post,
        "data": data.as_dict(),
        "cfg": cfg.as_dict(),
    }


# ---------------------------------------------------------------------------
# 6. 全量训练 + 逐格对照 + A 组对照档 + 主判据
# ---------------------------------------------------------------------------


def train_and_eval_profile(
    profile: str,
    *,
    cfg: Optional[TrainConfig] = None,
    product_dir: str = "",
    skip_identity_gate: bool = False,
    log: Any = None,
) -> Dict[str, Any]:
    """对一个特征档跑「恒等门禁 → 扰动自监督训练 → 逐格对照」的完整链路。

    返回
    ----
    Dict[str, Any]
        ``profile`` / ``identity_gate`` / ``update_gate`` / ``train`` / ``cell_grid`` /
        ``parameter_snapshot_after`` / ``post_train_identity_evidence`` / ``data`` / ``cfg``。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    cfg = cfg or TrainConfig()
    data = load_profile_data(str(profile), product_dir=str(product_dir), log=_log)
    model = build_model(data.table)
    if skip_identity_gate:
        gate: Dict[str, Any] = {
            "skipped": True,
            "note": (
                "**显式跳过**恒等门禁（--skip-identity-gate）；"
                "仅允许在已单独执行过 `variantb drill` 门禁的前提下使用"
            ),
        }
        _log("[variantb train] 恒等门禁：**跳过**（--skip-identity-gate）")
    else:
        _log(f"[variantb train] 档 {profile}：训练前恒等门禁 ...")
        gate = identity_gate(data, seed=int(cfg.seed), batch_size=int(cfg.batch_size))
        _log(
            f"[variantb train]   恒等门禁 {gate['n_cells_all_equal']}/{gate['n_cells']} "
            f"格逐条一致，通过 = {gate['passed']}"
        )
    before_snap = model.parameter_snapshot()
    _log(
        f"[variantb train] 档 {profile}：开始训练（mode={cfg.train_mode}，"
        f"epochs={cfg.epochs}，batch={cfg.batch_size}，lr={cfg.lr}）..."
    )
    train_info = train_transform(
        model, data.table.keys.numpy(), data.library_index, cfg, log=_log
    )
    after_snap = model.parameter_snapshot()
    up = update_gate(before_snap, after_snap)
    _log(
        f"[variantb train] 档 {profile}：可训参数更新 {up['n_updated']}/{up['n_total']}，"
        f"通过 = {up['passed']}"
    )
    _log(f"[variantb train] 档 {profile}：训练后逐格评测（clean + 9 扰动档）...")
    grid = cell_grid(model, data, seed=int(cfg.seed), batch_size=int(cfg.batch_size))
    with torch.no_grad():
        bitwise_after = sample_bitwise_equality(model, data.query_features())
        explicit_after = explicit_path_evidence(model, data.query_features())
        path_after = score_path_evidence(model, data.query_features())
    return {
        "profile": str(profile),
        "data": data.as_dict(),
        "cfg": cfg.as_dict(),
        "identity_gate": gate,
        "update_gate": up,
        "parameter_snapshot_before": before_snap,
        "parameter_snapshot_after": after_snap,
        "train": train_info,
        "cell_grid": grid,
        # [!] 训练后权重（张量）只作**进程内**传递：`run_variant_b` 会把它弹出放进
        # 独立字典供决定性对照臂复用，**绝不进产物**（JSON 无法序列化张量）。
        "_weights": {
            str(k): v.detach().cpu().to(torch.float32).clone()
            for k, v in model.transform.state_dict().items()
        },
        "post_train_identity_evidence": {
            "transform_is_identity": bool(model.transform.is_identity()),
            "bitwise_norm_equality": bitwise_after,
            "explicit_path": explicit_after,
            "score_path": path_after,
            "note": (
                "训练后 T 已离开恒等 ⇒ 恒等快路径失效、`norm(T(x)) != norm(x)` 属**预期**；"
                "`score_path.bitwise_equal` 则必须**恒为 True**（评测侧确实走了变换层，"
                "首版漏掉变换层的缺陷由它拦下）"
            ),
        },
    }


def primary_criterion(
    profile_results: Dict[str, Dict[str, Any]],
    *,
    effective_cells: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """**主判据**：1a 判有效的扰动格上，变体 B 训练后 R@1 不低于 KNN 且至少 1 格严格更高。

    参数
    ----
    profile_results : Dict[str, Dict[str, Any]]
        ``{档名: train_and_eval_profile(...) 的结果}``。
    effective_cells : Optional[Dict[str, Sequence[str]]]
        每档判有效的扰动格（``"kind/level"``）；缺省用 :data:`EFFECTIVE_CELLS_1A`。

    返回
    ----
    Dict[str, Any]
        ``per_cell`` / ``n_cells`` / ``n_not_worse`` / ``n_strictly_better`` /
        ``n_strictly_worse`` / ``at_least_one_better`` / ``passed`` / ``criterion`` /
        ``naive_direction``。
    """
    cells_of = dict(effective_cells or EFFECTIVE_CELLS_1A)
    rows: List[Dict[str, Any]] = []
    for profile, res in sorted(profile_results.items()):
        wanted = [str(x) for x in cells_of.get(str(profile), ())]
        by_cell = {str(c["cell"]): c for c in res["cell_grid"]["cells"]}
        for cell in wanted:
            c = by_cell.get(str(cell))
            if c is None:
                rows.append(
                    {
                        "profile": str(profile),
                        "cell": str(cell),
                        "applicable": False,
                        "note": "该格未在本次网格中产出（如实登记，不静默跳过）",
                    }
                )
                continue
            b = float(c["variant_b"]["recall_at_1"])
            k = float(c["knn"]["recall_at_1"])
            rows.append(
                {
                    "profile": str(profile),
                    "cell": str(cell),
                    "kind": str(c["kind"]),
                    "level": str(c["level"]),
                    "variant_b": float(b),
                    "knn": float(k),
                    "delta": float(b - k),
                    "variant_b_hit": int(c["variant_b"]["hit_at_1"]),
                    "knn_hit": int(c["knn"]["hit_at_1"]),
                    "n": int(c["variant_b"]["n"]),
                    "not_worse": bool(b >= k),
                    "strictly_better": bool(b > k),
                    "applicable": True,
                }
            )
    usable = [r for r in rows if r.get("applicable")]
    better = [r for r in usable if r["strictly_better"]]
    worse = [r for r in usable if not r["not_worse"]]
    return {
        "criterion": (
            "在 1a 判**有效**的扰动格上（`lexical-88` 的 noise/mask/nmag 三格；"
            "`bge-m3-1024` 的 noise 与 nmag 两格），变体 B 训练后 R@1 **均不低于 KNN**，"
            "且**至少 1 格严格更高**"
        ),
        "effective_cells_source": (
            "1a 生产实测（`checkpoints/qa_learn/_verify/robust/robust_calibration.json`）；"
            "该门禁**只用于判据**，不用来过滤训练数据"
        ),
        "per_cell": rows,
        "n_cells": int(len(usable)),
        "n_not_worse": int(sum(1 for r in usable if r["not_worse"])),
        "n_strictly_better": int(len(better)),
        "n_strictly_worse": int(len(worse)),
        "at_least_one_better": bool(len(better) > 0),
        "all_not_worse": bool(len(worse) == 0 and len(usable) > 0),
        "passed": bool(len(worse) == 0 and len(better) > 0 and len(usable) > 0),
        "naive_direction": (
            "计划口径第 4 条声明的朴素方向是「扰动自监督会让 argmax 变差」；"
            "本判据**不预设方向**，只现场报 Δ 的符号与条数 —— 若全部 Δ = 0，"
            "结论是「**未测出差异**」而**不是**「负结果」"
        ),
    }


def _flatten_cells(
    per_profile: Dict[str, Dict[str, Any]], arms: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """把逐档网格摊平成便于逐格读的对照表（含 A 组对照列与 1a 有效性标记）。"""
    rows: List[Dict[str, Any]] = []
    for profile, res in sorted(per_profile.items()):
        arm = arms.get(profile)
        arm_cells = {str(c["cell"]): c for c in (arm["cell_grid"]["cells"] if arm else [])}
        for c in res["cell_grid"]["cells"]:
            cell = str(c["cell"])
            row: Dict[str, Any] = {
                "profile": str(profile),
                "cell": cell,
                "kind": str(c["kind"]),
                "level": str(c["level"]),
                "variant_b": dict(c["variant_b"]),
                "knn": dict(c["knn"]),
                "delta": dict(c["delta"]),
                "effective_1a": bool(cell in tuple(EFFECTIVE_CELLS_1A.get(str(profile), ()))),
                "ineffective_1a": bool(
                    cell in tuple(INEFFECTIVE_CELLS_1A.get(str(profile), ()))
                ),
            }
            if arm is not None:
                a = arm_cells.get(cell)
                if a is not None:
                    row["arm_a_clean_train"] = {
                        "recall_at_1": float(a["variant_b"]["recall_at_1"]),
                        "recall_at_5": float(a["variant_b"]["recall_at_5"]),
                        "hit_at_1": int(a["variant_b"]["hit_at_1"]),
                        "delta_vs_knn": float(a["delta"]["recall_at_1"]),
                        "delta_vs_arm_b": float(
                            a["variant_b"]["recall_at_1"] - c["variant_b"]["recall_at_1"]
                        ),
                    }
            rows.append(row)
    return rows


def _honest_notes(
    per_profile: Dict[str, Dict[str, Any]],
    arms: Dict[str, Dict[str, Any]],
    primary: Dict[str, Any],
    clean_eval_arm: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """如实登记（**与实测值绑定**；禁止把「没测出来」写成负结果）。"""
    notes: List[str] = []
    notes.append(
        "**单 seed 42**：所有 Δ 为**单点差、无跨 seed 极差**；本报告不提供训练随机性区间。"
    )
    notes.append(
        "**档 0（clean）不作为判据**：查询行 = 键表行的逐位自匹配（R@1 结构性 = 1.000000），"
        "变体 B 在 T=I 时同样为 1.000000，该格无区分度。"
    )
    notes.append(
        "**KNN 基线 = 冻结键表上的归一化余弦 top-1**，与训练完全无关（键表不参与梯度）；"
        "它同时就是 1a 登记的「档 0 / 弱 / 中 / 强」基线，本模块现场重算并在 "
        "`anchor_1a_check` 中逐格对账（容差 1e-6）。"
    )
    for profile, res in sorted(per_profile.items()):
        gate = res["identity_gate"]
        if gate.get("skipped"):
            notes.append(f"**{profile}**：恒等门禁被**显式跳过**（`--skip-identity-gate`）。")
            continue
        notes.append(
            f"**{profile}**：恒等门禁 {gate['n_cells_all_equal']}/{gate['n_cells']} 格逐条一致，"
            f"范数逐位相等比例 {gate['bitwise_norm_equality']['bitwise_equal_frac']}，"
            f"显式路径逐位相同 {gate['explicit_path']['bitwise_equal']}，通过 = {gate['passed']}。"
        )
    for profile, res in sorted(per_profile.items()):
        ups = res["update_gate"]
        notes.append(
            f"**{profile}**：可训参数更新 {ups['n_updated']}/{ups['n_total']}"
            f"（从未更新 {ups['n_never_updated']}）；训练侧自命中 top-1 = "
            f"{res['train']['train_top1_self']:.6f}（**训练后量**，不是初值）。"
        )
    if arms:
        notes.append(
            "**A 组（干净自监督）对照口径**：训练查询不扰动、其余完全相同 —— 用于把"
            "「干净训练 ≈ 无变化」从推理变成**实测**；其逐格数字写在 "
            "`cells[].arm_a_clean_train`。"
        )
    for profile, res in sorted((clean_eval_arm or {}).items()):
        d = dict(res["clean_cells"]["delta"])
        notes.append(
            f"**决定性对照臂（{profile}）**：扰动自监督训练的模型 × **clean 评测** ⇒ "
            f"ΔR@1 = {float(d['recall_at_1']):+.6f}"
            f"（命中 {int(res['clean_cells']['variant_b']['hit_at_1'])}/"
            f"{int(res['clean_cells']['knn']['hit_at_1'])}）。该格把「T 本身的方向偏移」"
            "与「扰动泛化落差」分开：clean 格 Δ 显著为负即说明**掉点来自变换层本身**，"
            "而不是「训练扰动与评测扰动不同一次随机绘制」。"
        )
    notes.append(
        "**1a 判无效的格仍然报数字**（`bge-m3-1024` 的 mask 三档，1a 落差 0.000000）："
        "它们只作如实登记，**不参与主判据**（带着坏尺子做判定会得到无意义的结论）。"
    )
    notes.append(
        "**空间不足的格单列**：`nmag/strong`（KNN 0.992492，仅 5/666 条空间）与 "
        "`bge-m3-1024` 的 mask（KNN 1.000000，**零空间**）在 `cell_grid.error_space` 中"
        "被显式标出；在这些格上「Δ >= 0」的判别力极弱、而「Δ > 0」在零空间格里不可能成立，"
        "**不得**把它们读成变体 B 的能力结论。"
    )
    notes.append(
        "**训练侧扰动在批内施加**：1a 的 `perturb_matrix` 内部按**批内行号**生成噪声 / 掩码 / "
        "棋盘平移，故批大小会改变每一行的扰动实现；评测侧则是「全部查询一次扰动」。"
        "两者用同一实现与同一派生种子公式，但行号基准不同 —— 该事实登记在 "
        "`train.train_perturb_note`，不得被读成同一写法的两种等价表达。"
    )
    znotes: List[str] = []
    for profile, res in sorted(per_profile.items()):
        z = dict(res["train"].get("zero_norm_rows", {}))
        znotes.append(
            f"{profile} 掩码零范数行 {z.get('total', 0)} 条 / 涉及 "
            f"{z.get('n_steps_with_zero_row', 0)} 步"
        )
    notes.append(
        "**训练侧零范数行的边界处置（现场实测）**：批内某行被扰动后整行归零是**真实可达**的"
        "（`mask/strong` 遮蔽 44/88 维），该行原样保留为全零向量但**在损失中掩码**"
        "（logits 恒为 0，参与交叉熵只会注入与输入无关的常数惩罚），逐批计数进产物；"
        "整批全零则**显式报错**。逐档实测：" + "；".join(znotes) + "。"
        "1a 的 `perturb_matrix` 对零范数行仍**照旧显式报错**（该契约未改动）。"
    )
    if int(primary["n_strictly_better"]) == 0 and int(primary["n_strictly_worse"]) == 0:
        notes.append(
            "**主判据未成立，且其形态是「全部 Δ = 0」** —— 按口径必须写成"
            "「**未测出差异**」，**不是**负结果、也不是「训练无提升」的证明："
            "本轮的分辨率（666 条查询、单 seed）不足以在这一形态上判定因果。"
        )
    elif not primary["passed"]:
        notes.append(
            "**主判据未成立**：存在 Δ < 0 的格（详见 `primary.per_cell`）；如实报负结论，"
            "不包装；第三步（候选粒度改造）的方向依据见 `third_step`。"
        )
    return notes


def clean_eval_of_perturb_arm(
    profile: str,
    *,
    cfg: Optional[TrainConfig] = None,
    product_dir: str = "",
    perturb_train: Optional[Dict[str, Any]] = None,
    perturb_weights: Optional[Dict[str, torch.Tensor]] = None,
    log: Any = None,
) -> Dict[str, Any]:
    """**决定性对照臂**：用**扰动自监督**训练出的模型，去评测 **clean（无扰动）** 格。

    为什么需要这一臂（否证条款要求的"方向依据"）
    ------------------------------------------
    主臂在扰动格上同时混进两类原因：① 变换 ``T`` 本身改变了检索方向；
    ② 训练用的扰动实现与评测用的扰动实现**不是同一次随机绘制**。
    只看扰动格无法把两者分开。本臂把 ② 消掉（评测查询不扰动），因此：

    * 若 clean 格的 Δ 显著为负 ⇒ 掉点来自 **``T`` 本身**，
      即「单层 ``D→D`` 变换 + 冻结**逐行**键表」这一粒度上，训练会把
      已经在表内的行推离自己的键；
    * 若 clean 格 Δ ≈ 0 而只有扰动格掉 ⇒ 掉点主要来自
      **扰动自监督的泛化落差**，方向依据应落在训练信号而不是模型粒度上。

    参数
    ----
    profile : str
        特征档名。
    cfg : Optional[TrainConfig]
        训练配置（会强制 ``train_mode="perturb"``；评测侧不扰动）。
    product_dir : str
        ``n3d_qa`` 冻结产物目录。
    perturb_train : Optional[Dict[str, Any]]
        主臂（:func:`train_and_eval_profile`）的结果；与 ``perturb_weights`` 一起使用时
        **直接复用主臂训练后的权重**（逐位同一模型，不做重建），二者都为 ``None`` 时才现训一遍。
    perturb_weights : Optional[Dict[str, torch.Tensor]]
        主臂训练后的 ``transform.state_dict()``（含张量，**不进产物**，只在进程内传递）。
    log : Any
        可调用日志。

    返回
    ----
    Dict[str, Any]
        ``profile`` / ``clean_cells``（clean 格指标 + 与 KNN 的 Δ）/ ``score_path`` /
        ``update_gate`` / ``train``（训练摘要）/ ``weights_reused`` / ``reading``。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    cfg = cfg or TrainConfig()
    if str(cfg.train_mode) != "perturb":
        raise ValueError(
            f"本臂要求扰动自监督训练（train_mode='perturb'），当前 {cfg.train_mode!r}"
        )
    data = load_profile_data(str(profile), product_dir=str(product_dir), log=_log)
    model = build_model(data.table)
    weights_reused = False
    if perturb_train is not None and perturb_weights is not None:
        model.transform.load_state_dict(
            {str(k): v.detach().cpu().to(torch.float32) for k, v in perturb_weights.items()}
        )
        model.transform.train(False)
        train_info = dict(perturb_train.get("train", {}))
        before = list(perturb_train.get("parameter_snapshot_before", []))
        weights_reused = True
        _log(
            f"[variantb arm-clean-eval] 档 {profile}：**复用主臂训练后权重**"
            "（逐位同一模型，不重建），只在 clean 格上评测"
        )
    else:
        before = build_model(data.table).parameter_snapshot()
        _log(f"[variantb arm-clean-eval] 档 {profile}：扰动自监督训练（现训）...")
        train_info = train_transform(
            model, data.table.keys.numpy(), data.library_index, cfg, log=_log
        )
    up = update_gate(before, model.parameter_snapshot())
    with torch.no_grad():
        b_clean, k_clean, detail = eval_cell(
            model, data, kind="noise", level="weak", clean=True,
            seed=int(cfg.seed), batch_size=int(cfg.batch_size),
        )
        path_ev = score_path_evidence(model, data.query_features())
    _log(
        f"[variantb arm-clean-eval] 档 {profile}：clean 格 KNN R@1 = "
        f"{k_clean.recall_at_1:.6f} / 变体 B = {b_clean.recall_at_1:.6f}"
        f"（Δ {b_clean.recall_at_1 - k_clean.recall_at_1:+.6f}，命中 "
        f"{b_clean.hit_at_1}/{k_clean.hit_at_1}）"
    )
    return {
        "profile": str(profile),
        "weights_reused_from_main_arm": bool(weights_reused),
        "clean_cells": {
            "cell": "clean",
            "variant_b": b_clean.as_dict(),
            "knn": k_clean.as_dict(),
            "delta": {
                "recall_at_1": float(b_clean.recall_at_1 - k_clean.recall_at_1),
                "recall_at_5": float(b_clean.recall_at_5 - k_clean.recall_at_5),
                "hit_at_1": int(b_clean.hit_at_1 - k_clean.hit_at_1),
            },
            "evidence": detail,
        },
        "score_path": path_ev,
        "update_gate": up,
        "train": train_info,
        "reading": (
            "本臂评测查询**不扰动**，故 Δ 只反映「训练出的变换 T 本身」对检索方向的影响；"
            "与主臂扰动格的 Δ 相减即可把「T 本身」与「扰动泛化落差」分开。"
            "**注意**：clean 格的 KNN 侧是逐位自匹配（R@1 = 1.000000），"
            "故该格只用于读 Δ 的**符号与量级**，不参与主判据。"
        ),
    }


def third_step_evidence(report: Dict[str, Any]) -> Dict[str, Any]:
    """否证条款要求的**第三步方向依据**（由本报告的现场数字直接推出，不空谈）。"""
    primary = dict(report.get("primary", {}))
    rows = [r for r in primary.get("per_cell", []) if r.get("applicable")]
    deltas = [float(r["delta"]) for r in rows]
    train_top1 = {
        str(p): float(res["train"]["train_top1_self"])
        for p, res in sorted(report.get("per_profile", {}).items())
    }
    arms = report.get("arms", {}).get("clean_self_supervised", {})
    arm_delta: Dict[str, Any] = {}
    for profile, res in sorted(arms.items()):
        vals = [float(c["delta"]["recall_at_1"]) for c in res["cell_grid"]["cells"]]
        arm_delta[str(profile)] = {
            "max_abs_delta": float(max(abs(v) for v in vals)) if vals else 0.0,
            "n_nonzero_delta": int(sum(1 for v in vals if v != 0.0)),
            "n_cells": int(len(vals)),
        }
    clean_eval = report.get("arms", {}).get("perturb_train_clean_eval", {})
    clean_eval_delta = {
        str(p): float(res["clean_cells"]["delta"]["recall_at_1"])
        for p, res in sorted(clean_eval.items())
    }
    return {
        "delta_histogram": {
            "n_cells": int(len(deltas)),
            "n_zero": int(sum(1 for v in deltas if v == 0.0)),
            "n_positive": int(sum(1 for v in deltas if v > 0.0)),
            "n_negative": int(sum(1 for v in deltas if v < 0.0)),
            "max": float(max(deltas)) if deltas else 0.0,
            "min": float(min(deltas)) if deltas else 0.0,
        },
        "train_self_top1": train_top1,
        "arm_a_clean_train_delta_abs_max": arm_delta,
        "perturb_train_clean_eval_delta": clean_eval_delta,
        "reading": (
            "① 主要按**决定性对照臂**（扰动训练 × clean 评测）读：该臂 Δ 显著为负 ⇒ "
            "掉点来自**变换层本身**，即「单层 D→D 变换 + 冻结**逐行**键表」这一粒度上，"
            "argmax 与变换方向彼此拉扯，训练把表内行推离自己的键；此时第三步应改**候选粒度**"
            "（逐行键 → 类/簇级候选，或让键表随表示重算），而不是继续加大变换层容量；"
            "② 若决定性臂 Δ ≈ 0 而只有扰动格掉 ⇒ 掉点主要来自**扰动泛化落差**"
            "（训练扰动与评测扰动不是同一次随机绘制），应改**训练信号**（同分布多次绘制 / 更强正则），"
            "而非首先改粒度；"
            "③ 若训练侧自命中 top-1 未显著高于随机 ⇒ 变换层连训练集都没拟合，"
            "应先把 **D→D 线性变换的表达力 / 优化**这条前提落实，再谈粒度改造。"
        ),
    }


# ---------------------------------------------------------------------------
# 7. 全量运行 / 渲染 / 落盘 / 指纹
# ---------------------------------------------------------------------------


def anchor_1a_check(report: Dict[str, Any]) -> Dict[str, Any]:
    """现场重算的 KNN 基线 vs 1a 登记值的**逐格对账**（旁证，不作门禁）。"""
    rows: List[Dict[str, Any]] = []
    for c in report.get("cells", []):
        profile = str(c["profile"])
        ref = ROBUST_1A_KNN_REFERENCE.get(profile, {}).get(str(c["kind"]), {})
        want = ref.get(str(c["level"]))
        if want is None:
            continue
        got = float(c["knn"]["recall_at_1"])
        rows.append(
            {
                "profile": profile,
                "cell": str(c["cell"]),
                "knn_1a_registered": float(want),
                "knn_recomputed": float(got),
                "abs_diff": float(abs(got - float(want))),
                "within_tol": bool(abs(got - float(want)) <= float(ANCHOR_TOL)),
            }
        )
    return {
        "rule": ANCHOR_RULE,
        "tolerance": float(ANCHOR_TOL),
        "rows": rows,
        "n_rows": int(len(rows)),
        "n_within_tol": int(sum(1 for r in rows if r["within_tol"])),
        "all_within_tol": bool(rows and all(r["within_tol"] for r in rows)),
    }


def artifact_fingerprint(report: Dict[str, Any]) -> str:
    """报告的**内容指纹**（规范化 JSON 的 SHA256；不含任何时间字段）。"""
    return ET.sha256_bytes(ET.canonical_dumps(report))


def run_variant_b(
    *,
    profiles: Sequence[str],
    cfg: Optional[TrainConfig] = None,
    product_dir: str = "",
    include_clean_control: bool = True,
    log: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, torch.Tensor]]]:
    """全量运行：逐档「门禁 → 扰动自监督训练 → 逐格对照」+ A 组 + 决定性对照臂。

    产物**不含挂钟时间 / 耗时**（确定性纪律）；耗时只进日志。

    返回
    ----
    Tuple[Dict[str, Any], Dict[str, Dict[str, torch.Tensor]]]
        ``(报告, {档名: 主臂训练后的 transform.state_dict()})``。权重**不进产物**
        （JSON 无法序列化张量，且产物须为纯 JSON + 确定性），只在进程内供决定性对照臂
        复用同一个模型（由 ``evidence.decisive_arm_weight_rule`` 记录该口径）。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    cfg = cfg or TrainConfig()
    t0 = time.time()
    per_profile: Dict[str, Dict[str, Any]] = {}
    perturb_weights: Dict[str, Dict[str, torch.Tensor]] = {}
    for profile in profiles:
        per_profile[str(profile)] = train_and_eval_profile(
            str(profile), cfg=cfg, product_dir=str(product_dir), log=_log
        )
        perturb_weights[str(profile)] = dict(per_profile[str(profile)].pop("_weights"))
    arms: Dict[str, Any] = {}
    if include_clean_control:
        clean_cfg = TrainConfig(
            seed=int(cfg.seed),
            epochs=int(cfg.epochs),
            batch_size=int(cfg.batch_size),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            train_mode="clean",
            log_every=int(cfg.log_every),
        )
        for profile in profiles:
            _log(f"[variantb arm-A] 档 {profile}：干净自监督（A 组对照档）...")
            arms[str(profile)] = train_and_eval_profile(
                str(profile), cfg=clean_cfg, product_dir=str(product_dir), log=_log
            )
            arms[str(profile)].pop("_weights", None)
    # **决定性对照臂**：扰动自监督训练的模型 × clean 评测（把「T 本身」与「扰动泛化落差」分开）
    clean_eval_arm: Dict[str, Any] = {}
    for profile in profiles:
        _log(
            f"[variantb arm-clean-eval] 档 {profile}："
            "扰动自监督训练的模型 × **clean 评测**（决定性对照）..."
        )
        clean_eval_arm[str(profile)] = clean_eval_of_perturb_arm(
            str(profile),
            cfg=cfg,
            product_dir=str(product_dir),
            perturb_train=per_profile.get(str(profile)),
            perturb_weights=perturb_weights.get(str(profile)),
            log=_log,
        )
    primary = primary_criterion(per_profile)
    report: Dict[str, Any] = {
        "module": MODULE_NAME,
        "artifact_schema": ARTIFACT_SCHEMA,
        "deterministic": True,
        "excluded_fields_note": (
            "本产物不含 created_utc / seconds / 主机名等非确定字段；"
            "全量耗时与生成时间只写运行日志（与 robust_eval 同一确定性纪律）"
        ),
        "single_seed_note": (
            "**单 seed 42**；所有 Δ 为**单点差、无跨 seed 极差** —— 本报告不提供任何"
            "训练随机性区间，Δ 只表示同一 seed 下变体 B 与 KNN 的单点差"
        ),
        "config": cfg.as_dict(),
        "profiles": [str(p) for p in profiles],
        "per_profile": per_profile,
        "arms": {
            "clean_self_supervised": arms,
            "perturb_train_clean_eval": clean_eval_arm,
        },
        "cells": _flatten_cells(per_profile, arms),
        "primary": primary,
        "honest_notes": _honest_notes(per_profile, arms, primary, clean_eval_arm),
        "evidence": {
            "carrier": "x = **原始编码器特征**（raw），不经 N3D 骨干；考卷路径本身不含任何模型",
            "model": (
                "单层 D→D 变换 T（恒等初始化）+ 冻结特征库内积评分；"
                "即「表内嵌输出层、权重即特征库、冻结」，只学变换层"
            ),
            "loss": (
                "CrossEntropyLoss(logits = T(q) @ keys.T, "
                "label = 行自身在全量键表中的下标)"
            ),
            "split": (
                "训练查询 = 库行 1999（严格排除 666 查询行）；评测查询 = 666 查询行；"
                "键表固定全量 2665 行"
            ),
            "identity_fastpath_rule": IDENTITY_FASTPATH_RULE,
            "reuse_1a": (
                "扰动走 robust_eval.perturb_matrix / derived_seed / PERTURB_GRID；"
                "归一化走 entry_table.l2_normalize_rows —— 未自造第二份口径"
            ),
            "knn_reference_1a": ROBUST_1A_KNN_REFERENCE,
            "knn_reference_rule": ANCHOR_RULE,
            "anchor_tolerance": float(ANCHOR_TOL),
            "artifacts": f"一律写 {VARIANT_B_DIR}；不落盘特征矩阵、不产 zip 产物",
        },
    }
    report["third_step"] = third_step_evidence(report)
    report["anchor_1a_check"] = anchor_1a_check(report)
    report["evidence"]["decisive_arm_weight_rule"] = (
        "决定性对照臂（扰动训练 × clean 评测）复用**主臂同一个训练后模型**"
        "（进程内直接传 `transform.state_dict()`，不做任何重放/重建）；"
        "权重为张量、**不进产物**（产物保持纯 JSON + 确定性）"
    )
    report["artifact_fingerprint"] = artifact_fingerprint(report)
    _log(f"[variantb] 全量完成（耗时 {time.time() - t0:.1f}s，该耗时只进日志不入产物）")
    return report, perturb_weights


def render_markdown(report: Dict[str, Any]) -> str:
    """把报告渲染为 Markdown（逐格表 + 门禁 + 主判据 + 诚实登记）。"""
    cfg = dict(report.get("config", {}))
    lines: List[str] = []
    lines.append("# n3d_qa_learn 变体 B：可学 D→D 变换 + 冻结特征库（第二步）")
    lines.append("")
    lines.append(f"- **单 seed 声明**：{report.get('single_seed_note')}")
    lines.append(
        "- **本报告不含生成时间 / 耗时**（确定性纪律）：同参数重跑的产物应逐字节一致。"
    )
    lines.append(
        f"- 训练超参：`epochs={cfg.get('epochs')}` / `batch_size={cfg.get('batch_size')}` / "
        f"`lr={cfg.get('lr')}` / `weight_decay={cfg.get('weight_decay')}` / "
        f"`train_mode={cfg.get('train_mode')}`"
    )
    ev = dict(report.get("evidence", {}))
    lines.append(f"- 载体：{ev.get('carrier')}")
    lines.append(f"- 模型：{ev.get('model')}")
    lines.append(f"- 划分：{ev.get('split')}")
    lines.append("")
    lines.append("## 1. 恒等门禁（硬门禁）")
    lines.append("")
    lines.append("| 档 | 逐条一致格数 | 范数逐位相等比例 | 显式路径逐位相同 | 通过 |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for profile, res in sorted(report.get("per_profile", {}).items()):
        gate = dict(res.get("identity_gate", {}))
        if gate.get("skipped"):
            lines.append(f"| `{profile}` | — | — | — | 跳过 |")
            continue
        bw = dict(gate.get("bitwise_norm_equality", {}))
        ex = dict(gate.get("explicit_path", {}))
        lines.append(
            f"| `{profile}` | {gate.get('n_cells_all_equal')}/{gate.get('n_cells')} | "
            f"{bw.get('bitwise_equal_frac')} | {ex.get('bitwise_equal')} | "
            f"**{gate.get('passed')}** |"
        )
    lines.append("")
    lines.append("## 2. 逐格 变体 B vs KNN（R@1 / @5 / Δ）")
    lines.append("")
    lines.append(
        "| 档 | 格 | 1a 有效性 | KNN R@1 | 变体 B R@1 | ΔR@1 | Δ条数 | "
        "KNN R@5 | 变体 B R@5 | A 组 R@1 | A 组 ΔR@1 |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c in report.get("cells", []):
        if c.get("effective_1a"):
            eff = "有效"
        elif c.get("ineffective_1a"):
            eff = "**无效**"
        else:
            eff = "—"
        arm = dict(c.get("arm_a_clean_train") or {})
        arm_r1 = f"{float(arm['recall_at_1']):.6f}" if arm else "—"
        arm_d = f"{float(arm['delta_vs_knn']):+.6f}" if arm else "—"
        lines.append(
            f"| `{c['profile']}` | `{c['cell']}` | {eff} | "
            f"{float(c['knn']['recall_at_1']):.6f} | {float(c['variant_b']['recall_at_1']):.6f} | "
            f"**{float(c['delta']['recall_at_1']):+.6f}** | {int(c['delta']['hit_at_1']):+d} | "
            f"{float(c['knn']['recall_at_5']):.6f} | {float(c['variant_b']['recall_at_5']):.6f} | "
            f"{arm_r1} | {arm_d} |"
        )
    lines.append("")
    lines.append("## 3. 分档落差（弱 − 强）的变化")
    lines.append("")
    lines.append("| 档 | 扰动 | KNN 落差 | 变体 B 落差 | Δ落差 |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for profile, res in sorted(report.get("per_profile", {}).items()):
        for g in res["cell_grid"]["gap_weak_minus_strong"]:
            lines.append(
                f"| `{profile}` | `{g['kind']}` | {float(g['knn_gap']):.6f} | "
                f"{float(g['variant_b_gap']):.6f} | **{float(g['delta_gap']):+.6f}** |"
            )
    lines.append("")
    lines.append("## 4. 主判据")
    lines.append("")
    primary = dict(report.get("primary", {}))
    lines.append(f"- 判据：{primary.get('criterion')}")
    lines.append(
        f"- 结果：**通过 = {primary.get('passed')}**；格数 {primary.get('n_cells')} / "
        f"不劣于 KNN {primary.get('n_not_worse')} / 严格更高 {primary.get('n_strictly_better')} / "
        f"严格更低 {primary.get('n_strictly_worse')}"
    )
    lines.append(f"- 方向说明：{primary.get('naive_direction')}")
    lines.append("")
    lines.append("| 档 | 格 | KNN R@1 | 变体 B R@1 | Δ | 命中(变体B/KNN) | 不劣于 | 严格更高 |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- | --- |")
    for r in primary.get("per_cell", []):
        if not r.get("applicable"):
            lines.append(
                f"| `{r['profile']}` | `{r['cell']}` | — | — | — | — | 不适用 | 不适用 |"
            )
            continue
        lines.append(
            f"| `{r['profile']}` | `{r['cell']}` | {float(r['knn']):.6f} | "
            f"{float(r['variant_b']):.6f} | **{float(r['delta']):+.6f}** | "
            f"{int(r['variant_b_hit'])}/{int(r['knn_hit'])} | "
            f"{r['not_worse']} | {r['strictly_better']} |"
        )
    lines.append("")
    lines.append("## 5. 可训参数更新量门禁")
    lines.append("")
    lines.append("| 档 | 更新参数数 | 总参数数 | 从未更新 | 通过 |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for profile, res in sorted(report.get("per_profile", {}).items()):
        up = dict(res["update_gate"])
        lines.append(
            f"| `{profile}` | {up['n_updated']} | {up['n_total']} | "
            f"{up['n_never_updated']} | **{up['passed']}** |"
        )
    lines.append("")
    lines.append("## 6. 空间不足格（显式标注）")
    lines.append("")
    lines.append("| 档 | 格 | KNN 命中 | 错误条数 | 查询数 |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for profile, res in sorted(report.get("per_profile", {}).items()):
        for e in res["cell_grid"]["error_space"]:
            lines.append(
                f"| `{profile}` | `{e['cell']}` | {int(e['knn_hit_at_1'])} | "
                f"{int(e['knn_errors'])} | {int(e['n'])} |"
            )
    lines.append("")
    lines.append("## 7. 决定性对照臂（扰动训练 × clean 评测）")
    lines.append("")
    lines.append("| 档 | KNN R@1 | 变体 B R@1 | ΔR@1 | 命中(变体B/KNN) |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for profile, res in sorted(
        dict(report.get("arms", {}).get("perturb_train_clean_eval", {})).items()
    ):
        c = dict(res["clean_cells"])
        lines.append(
            f"| `{profile}` | {float(c['knn']['recall_at_1']):.6f} | "
            f"{float(c['variant_b']['recall_at_1']):.6f} | "
            f"**{float(c['delta']['recall_at_1']):+.6f}** | "
            f"{int(c['variant_b']['hit_at_1'])}/{int(c['knn']['hit_at_1'])} |"
        )
    lines.append("")
    lines.append("该臂**评测查询不扰动**，故 Δ 只反映「训练出的变换 T 本身」的方向偏移：")
    lines.append("Δ 显著为负即说明掉点来自**变换层本身**，而不是「训练扰动 ≠ 评测扰动」。")
    lines.append("")
    lines.append("## 8. 1a 基线对账（旁证）")
    lines.append("")
    anchor = dict(report.get("anchor_1a_check", {}))
    lines.append(
        f"- 容差 `{anchor.get('tolerance')}`；在容差内 "
        f"{anchor.get('n_within_tol')}/{anchor.get('n_rows')}，"
        f"全部在容差内 = **{anchor.get('all_within_tol')}**"
    )
    lines.append("")
    lines.append("| 档 | 格 | 1a 登记 | 现场重算 | 绝对差 | 在容差内 |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for r in anchor.get("rows", []):
        lines.append(
            f"| `{r['profile']}` | `{r['cell']}` | {float(r['knn_1a_registered']):.6f} | "
            f"{float(r['knn_recomputed']):.6f} | {float(r['abs_diff']):.2e} | {r['within_tol']} |"
        )
    lines.append("")
    lines.append("## 9. 第三步方向依据（否证条款要求）")
    lines.append("")
    third = dict(report.get("third_step", {}))
    lines.append(f"- Δ 直方图：{third.get('delta_histogram')}")
    lines.append(f"- 训练侧自命中 top-1：{third.get('train_self_top1')}")
    lines.append(f"- A 组（干净训练）Δ 的绝对最大值：{third.get('arm_a_clean_train_delta_abs_max')}")
    lines.append(f"- 读法：{third.get('reading')}")
    lines.append("")
    lines.append("## 10. 如实登记（不得包装）")
    lines.append("")
    for i, note in enumerate(report.get("honest_notes", []), start=1):
        lines.append(f"{i}. {note}")
    lines.append("")
    return "\n".join(lines)


def write_json(path: str, obj: Any) -> str:
    """写 JSON（UTF-8 无 BOM，缩进 1，排序键；返回文件 SHA256）。"""
    return R.write_json(path, obj)


def write_text(path: str, text: str) -> str:
    """写文本（UTF-8 无 BOM；返回文件 SHA256）。"""
    return R.write_text(path, text)


def load_json(path: str) -> Dict[str, Any]:
    """读取 JSON 产物（缺失即可读报错）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"variantb 产物不存在：{path!r}")
    with open(path, "r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def summarize_probe(
    profiles: Sequence[str], *, product_dir: str = "", log: Any = None
) -> Dict[str, Any]:
    """**口径取证**（不训练）：逐档给出键表 / 划分 / 恒等门禁的三组现场事实。

    参数
    ----
    profiles : Sequence[str]
        参与取证的档。
    product_dir : str
        ``n3d_qa`` 冻结产物目录。
    log : Any
        可调用日志。

    返回
    ----
    Dict[str, Any]
        ``profiles``（逐档取证）/ ``errors``（失败清单，**如实登记不静默跳过**）/
        ``transform_names``（可训参数的现场枚举名字）/ ``identity_fastpath_rule``。
    """

    def _log(msg: str) -> None:
        if callable(log):
            log(msg)

    per: Dict[str, Any] = {}
    errors: List[Dict[str, Any]] = []
    names: List[str] = []
    for profile in profiles:
        try:
            data = load_profile_data(str(profile), product_dir=str(product_dir), log=_log)
            model = build_model(data.table)
            names = [str(n) for n, _ in model.transform.named_parameters()]
            gate = identity_gate(data)
            per[str(profile)] = {
                "data": data.as_dict(),
                "identity_gate": gate,
                "parameter_snapshot": model.parameter_snapshot(),
                "transform_names": names,
                "transform_dim": int(model.dim),
                "n_keys": int(model.n_keys),
            }
            _log(
                f"[variantb probe] 档 {profile}: D={model.dim} N={model.n_keys} "
                f"可训参数={names} 恒等门禁通过={gate['passed']}"
            )
        except Exception as exc:  # noqa: BLE001 —— 如实登记不静默跳过
            errors.append(
                {
                    "profile": str(profile),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            _log(f"[variantb probe] 档 {profile} 失败（如实登记）：{type(exc).__name__}: {exc}")
    return {
        "module": MODULE_NAME,
        "artifact_schema": ARTIFACT_SCHEMA,
        "deterministic": True,
        "single_seed_note": "单 seed 42；本产物不含时间字段",
        "profiles": per,
        "errors": errors,
        "n_ok": int(len(per)),
        "n_error": int(len(errors)),
        "transform_names": names,
        "identity_fastpath_rule": IDENTITY_FASTPATH_RULE,
        "reuse_1a": (
            "扰动 = robust_eval.perturb_matrix / derived_seed / PERTURB_GRID；"
            "归一化 = entry_table.l2_normalize_rows"
        ),
    }


__all__ = [
    "ANCHOR_RULE",
    "ANCHOR_TOL",
    "ARTIFACT_SCHEMA",
    "CellMetrics",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_LR",
    "DEFAULT_WEIGHT_DECAY",
    "DToDTransform",
    "EFFECTIVE_CELLS_1A",
    "FULL_TABLE_ROWS",
    "IDENTITY_FASTPATH_RULE",
    "INEFFECTIVE_CELLS_1A",
    "LIBRARY_ROWS",
    "MODULE_NAME",
    "QUERY_ROWS",
    "ROBUST_1A_KNN_REFERENCE",
    "SHUFFLE_SEED",
    "TRAIN_MODES",
    "TRAIN_PERTURB_CELLS",
    "TrainConfig",
    "UPDATE_GATE_RULE",
    "VARIANT_B_DIR",
    "VARIANT_B_SEED",
    "VariantBData",
    "VariantBModel",
    "anchor_1a_check",
    "artifact_fingerprint",
    "build_data",
    "build_model",
    "cell_grid",
    "clean_eval_of_perturb_arm",
    "compare_top1",
    "drill",
    "eval_cell",
    "explicit_path_evidence",
    "identity_gate",
    "gradient_flow_evidence",
    "load_json",
    "load_profile_data",
    "primary_criterion",
    "render_markdown",
    "run_variant_b",
    "sample_bitwise_equality",
    "score_path_evidence",
    "summarize_probe",
    "third_step_evidence",
    "train_and_eval_profile",
    "train_transform",
    "update_gate",
    "write_json",
    "write_text",
    "zero_norm_mask",
]
