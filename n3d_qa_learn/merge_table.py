"""轮 C：**合并条目表**（QA 侧 + 文本侧 -> 单张 :class:`EntryKeyTable`）与两臂路由。

架构命题
--------
「QA 与文本是**同一个 map**」——即两侧条目本可共存于**一张**条目表里，由一次
排序（余弦 top-1）统一检索；而现状是**两级硬优先级路由**（QA 优先、未命中再查文本）。

本模块**只做零训练的离线对照**：不改生产路由、不训练、不动任何上游模块。产物一律写
``checkpoints/qa_learn/_verify/merge_table/``（不落盘裸特征矩阵、不产 zip、无挂钟字段）。

口径（冻结，不得在运行期变动）
------------------------------
* 条目总数 **N = 2814** = QA 侧 ``train_known`` **149** + 文本侧全量行 **2665**；
* 两侧键同 ``D``、同 ``dtype``、各自逐行 L2 归一化；
* ``local-hash`` 角色下 ``question`` 与 ``text_line`` 的编码**逐位相同**
  （hash 家族不使用 ``max_length``，故两角色对同一文本给出同一 float32 字节）；
* 合并只做**拼接**，不做任何重编码 / 重归一化（合并前后各半的键字节**零变化**）。

返回类型契约
------------
合并表的 :meth:`MergedEntryKeyTable.lookup_typed` **必须**能判定返回物类型：返回
:class:`LookupContract`，其中 ``returned_kind`` 与 ``expected_kind`` 都是**枚举值**，
``type_contract_ok`` 为假即「返回类型与期望类型不一致」（例如「返回文本却当成答案」）。
评测层必须统计并报告该计数，非零即登记为缺陷。

上游边界（零改动）
------------------
本模块**只读** import ``n3d_qa_learn`` 内既有模块（``entry_table`` / ``encoders`` /
``step2`` / ``data``）；不修改任何上游模块源码 / 产物 / ``current_spec.md``；
零新增依赖（torch / numpy + 标准库）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .encoders import (
    ENCODER_LOCAL_HASH,
    ROLE_QUESTION,
    ROLE_TEXT_LINE,
    EncoderConfig,
    build_vectorizer,
)
from .entry_table import EntryKeyTable
from .entry_table import (
    NORM_ATOL,
    QAEntryBundle,
    TextEntryBundle,
    encode_statements,
    sha256_bytes,
)

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

#: 合并表协议版本（结构变化必须递增；进指纹与产物）。
MERGE_TABLE_VERSION: str = "n3dqa-merge-table-v1"

#: 合并表内容指纹 schema（与单侧表的 ``entry_table.KEY_FINGERPRINT_SCHEMA`` 不同，
#: 防止「合并表」与「单侧表」在两侧条目恰好相同时撞同一指纹）。
MERGE_FINGERPRINT_SCHEMA: str = "merged-entry-key-table-v1"

#: 返回物类型枚举（**唯一**来源；禁止在别处手写字符串字面量）。
KIND_QA: str = "qa"
KIND_TEXT: str = "text"
KIND_NONE: str = "none"
ENTRY_KINDS: Tuple[str, str] = (KIND_QA, KIND_TEXT)
RETURN_KINDS: Tuple[str, str, str] = (KIND_QA, KIND_TEXT, KIND_NONE)

#: 「条目类型 -> 绑定输出的语义」。QA 侧 = 答案展示文本；文本侧 = 行原文。
KIND_SEMANTICS: Dict[str, str] = {
    KIND_QA: "答案展示文本（answer_display）",
    KIND_TEXT: "行原文（row text）",
}

#: 合并顺序（**固定**）：QA 半区在前、文本半区在后。该顺序决定条目下标区间，
#: 是「按 kind 切分视图」与「逐位零变化核对」的依据，不得在运行期变动。
MERGE_ORDER: Tuple[str, str] = (KIND_QA, KIND_TEXT)

#: 合并前的三条断言（任一失败即停止，不得继续出数）。
PREMERGE_ASSERTION_KEYS: Tuple[str, str, str] = (
    "same_dtype_and_norm",
    "role_bitwise_identity",
    "merge_key_bytes_unchanged",
)

#: 两臂的路由结构名。
ARM_QA_PRIORITY: str = "qa_priority"
ARM_MERGED_TOPK1: str = "merged_topk1"
ARM_ORDER: Tuple[str, str] = (ARM_QA_PRIORITY, ARM_MERGED_TOPK1)

#: 路由结构说明（两臂**共用**同一批查询与同一批键，唯一变量是路由结构）。
ARM_RULES: Dict[str, str] = {
    ARM_QA_PRIORITY: (
        "现状两级硬优先级：先在 QA 候选集（合并表前 149 行）取余弦 top-1；"
        "若其分数 >= TAU 则返回，否则再在文本候选集（后 2665 行）取 top-1，"
        "分数 < TAU 则返回「无匹配」。QA 命中即不再查文本。"
    ),
    ARM_MERGED_TOPK1: (
        "合并表单次排序取 top-1：在全部 2814 行上取余弦 top-1；"
        "分数 < TAU 则返回「无匹配」。不做任何 kind 优先级。"
    ),
}

#: **两臂共用的同一判定门限**（唯一变量是路由结构；门限不得按臂分别设定）。
TAU: float = 0.5

#: 门限口径文本。
TAU_RULE: str = (
    "TAU 是两臂**共用**的「有把握才算命中」门限：两臂对各自的 top-1 用**同一个** TAU "
    "作 >= 判定，低于门限一律返回「无匹配」。因此两臂的唯一结构差异是候选集合与"
    "优先级顺序，而不是门限。主判据（命中率 / R@1 / 类型正确率）只取**未过滤的 top-1**"
    "（不受 TAU 影响），TAU 过滤后的量单列为诊断，避免门限取值污染主对照。"
)

#: 原文支 R@1 = 1.0 是**结构性锚点**（查询文本即行原文、键表里就有该行自身）⇒ 不参与判定。
ANCHOR_VARIANT_NOTE: str = (
    "原文支 R@1 == 1.0 是结构性锚点（查询文本逐字节等于键表内该行自身），"
    "单列报告、**不参与**「不劣 / 劣化」判定。"
)

#: 判定用的非锚点变体（关键词 / 摘要 / 自适应）。
NON_ANCHOR_VARIANTS: Tuple[str, ...] = ("关键词", "摘要", "自适应")

#: 变体名（与 ``text_variants.VARIANT_ORDER`` 逐字一致；本模块**重新登记**为冻结常量，
#: 以免在导入期拉起 keybert 依赖）。
VARIANT_ORDER: Tuple[str, ...] = ("原文", "关键词", "摘要", "自适应")

#: 4 支变体的档位标签（报告分档用）。
VARIANT_LABELS: Dict[str, str] = {
    "原文": "原文（结构性锚点）",
    "关键词": "关键词",
    "摘要": "摘要",
    "自适应": "自适应",
}

#: 期望类型：QA 侧查询期望返回答案文本；文本侧查询期望返回行原文。
EXPECTED_RETURN_KIND: Dict[str, str] = {
    "qa": KIND_QA,
    "text": KIND_TEXT,
}

#: 返回类型正确率的口径文本。
RETURN_TYPE_RULE: str = (
    "返回类型正确率 = 「返回物类型 == 查询期望类型」的样本占比。期望类型由查询侧决定："
    "QA 侧查询期望 kind='qa'（答案展示文本），文本侧查询期望 kind='text'（行原文）；"
    "返回 kind='none'（无匹配）计入**不一致**。同时分类计数两类违规："
    "① returned_kind != expected_kind（含 none）；"
    "② 查询期望为答案展示文本、而返回物**不属于任何 qa 条目的答案展示文本**"
    "（即 returned_kind == 'qa' 但内容对不上）—— 第②类的**独立判定实现**是 "
    ":func:`content_view_violations`（**不依赖** kind 字段，只按输出内容判），"
    "其可达性口径单独登记在 `RETURN_TYPE_VIOLATION_CLASS_2_REACHABILITY`。"
)

#: 第②类违规的**可达性**口径（必须显式，不得含糊）。
RETURN_TYPE_VIOLATION_CLASS_2_REACHABILITY: str = (
    "第②类（期望 qa 但返回物不属于任何 qa 条目的展示文本）在当前合并表构造下"
    "**结构上不可达**：`MergedEntryKeyTable.__post_init__` 强制 `entry_kinds` 与"
    "「按输出内容归类的 `outputs_index`」**逐条一致**，因此只要 `returned_kind == 'qa'`，"
    "该下标一定在 qa 半区、其输出一定在 qa 展示文本集合内 ⇒ 第②类计数恒为 0。"
    "**这不是省略**：`:func:`content_view_violations`` 给了独立（不看 kind 字段的）判定实现，"
    "并在评测里逐条跑；同时在本模块的现场用例里用**伪造内容视图**证明该断言**会失败**"
    "（即断言非恒真）。若日后解除 kind↔内容 的构造期绑定，第②类就会变成可达 —— 届时"
    "该计数会立刻非 0，评测的缺陷登记自动生效。"
)

#: 文本侧 R@1 的金标集口径（**口径等价摘录** ``text_variants.GOLD_SET_RULE``）。
#:
#: **F3 修复（计划 1 遗留项）**：原文曾称「**逐字抄录**」，却漏抄了 ``rank = 金标集成员
#: 在 top-k 列表中的最前 1-based 位置（不在则 -1）`` 一句 —— 「逐字」不成立。现二选一
#: 取**改述**：本模块**只使用 ``hit@1``**（top-1 是否落在金标集内），**不使用 rank 字段**，
#: 故登记为「口径等价摘录（本轮不使用 rank 字段）」，而**不**再声称逐字一致。
#: 上源完整文本的唯一权威位置是 ``n3d_qa_learn.text_variants.GOLD_SET_RULE``。
GOLD_SET_RULE_SOURCE: str = "n3d_qa_learn.text_variants.GOLD_SET_RULE"
GOLD_SET_RULE_EXCERPT_NOTE: str = (
    "口径等价摘录（本轮不使用 rank 字段）：本模块只用 hit@1 判定，"
    "不计算也不依赖 top-k 名次（rank）。上源完整口径的唯一权威位置见 "
    "`GOLD_SET_RULE_SOURCE`；本摘录**不声称逐字一致**。"
)
GOLD_SET_RULE: str = (
    "金标集 = 自身行 ∪ 与查询行**原文逐字节相同**的其他行（仅接受逐字节相同，不接受"
    "模糊 / 归一化后相同）；hit@1 = top-1 落在金标集内。"
)

#: 合并表 vs 现状路由的判定口径。
VERDICT_RULE: str = (
    "非劣 = 合并表臂在**全部**可比指标上 >= 现状路由臂；劣化 = 存在至少一项严格低于；"
    "可比指标 = ① QA 侧命中率 ② 文本侧非锚点三支 R@1（原文支为结构性锚点、不参与）"
    "③ 返回类型正确率。判据必须先核实基线**不在可行域边界**（0 / 1 等饱和点），"
    "否则「严格优于 / 不劣于」在该指标上不可判。"
)

#: 本结论成立的口径范围（必须显式声明）。
SCOPE_NOTE: str = (
    "本结论**仅在 C=10 口径下成立**（QA 侧 train_known = 149 条目、答案类数 C = 10、"
    "文本侧全量 2665 行，合计 2814）。扩 C 后 QA 侧条目数与合并表规模都会变化，"
    "混合检索的异构干扰量随之变化，**必须另行复跑**；不得把本结论外推到其它 C。"
)

#: 产物目录（**唯一**允许的产物根目录之下）。
VERIFY_ROOT: str = os.path.join("checkpoints", "qa_learn", "_verify")
DEFAULT_OUT_DIR: str = os.path.join(VERIFY_ROOT, "merge_table")


# ---------------------------------------------------------------------------
# 类型标记契约
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LookupContract:
    """**类型标记契约**：一次合并表查询的返回物类型判定结果。

    参数
    ----
    index : int
        命中的条目下标（``-1`` = 未命中）。
    returned_kind : str
        返回物类型，取值 ∈ :data:`RETURN_KINDS`。
    expected_kind : str
        查询**期望**的返回物类型，取值 ∈ :data:`ENTRY_KINDS`。
    returned_output : str
        返回物本体（``returned_kind == 'none'`` 时为空串）。
    """

    index: int
    returned_kind: str
    expected_kind: str
    returned_output: str

    def __post_init__(self) -> None:
        if str(self.returned_kind) not in RETURN_KINDS:
            raise ValueError(
                f"returned_kind={self.returned_kind!r} 不在允许集合 {list(RETURN_KINDS)} 内"
            )
        if str(self.expected_kind) not in ENTRY_KINDS:
            raise ValueError(
                f"expected_kind={self.expected_kind!r} 不在允许集合 {list(ENTRY_KINDS)} 内"
            )

    @property
    def matched(self) -> bool:
        """是否命中了一个**具体条目**（``kind='none'`` 视为未命中）。"""
        return str(self.returned_kind) != KIND_NONE

    @property
    def type_contract_ok(self) -> bool:
        """返回物类型与查询期望类型是否一致（**本契约的主判据**）。"""
        return str(self.returned_kind) == str(self.expected_kind)

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化（``returned_output`` 只在命中时非空）。"""
        return {
            "index": int(self.index),
            "returned_kind": str(self.returned_kind),
            "expected_kind": str(self.expected_kind),
            "returned_output": str(self.returned_output),
            "matched": bool(self.matched),
            "type_contract_ok": bool(self.type_contract_ok),
        }


#: 类型契约违背的**分类枚举**（统计与报告必须按类分别计数）。
TYPE_VIOLATION_CLASSES: Tuple[str, ...] = (
    "returned_none_expected_entry",
    "returned_qa_expected_text",
    "returned_text_expected_qa",
)

TYPE_VIOLATION_CLASS_RULE: Dict[str, str] = {
    "returned_none_expected_entry": "未命中（kind='none'）而查询期望是具体条目",
    "returned_qa_expected_text": "文本侧查询却返回了答案展示文本（QA 条目抢走文本行）",
    "returned_text_expected_qa": "QA 侧查询却返回了行原文（文本条目抢走答案）",
}


def classify_type_violation(contract: LookupContract) -> str:
    """把一次类型契约失败归类（契约成立时返回空串）。

    现场枚举（唯一来源）：
    * ``kind='none'`` -> ``returned_none_expected_entry``；
    * ``returned='qa'`` 而 ``expected='text'`` -> ``returned_qa_expected_text``；
    * ``returned='text'`` 而 ``expected='qa'`` -> ``returned_text_expected_qa``。

    **与 ``RETURN_TYPE_RULE`` 的关系（F2 修复后必须读清）**：本函数覆盖的是合同里的
    **第①类**（``returned_kind != expected_kind``）。合同的**第②类**（期望 qa、但返回物
    不属于任何 qa 条目的展示文本）**不能**由本函数判定 —— 它需要看**输出内容**，由
    :func:`content_view_violations` 独立实现（不看 ``kind`` 字段）。二者的可达性口径见
    :data:`RETURN_TYPE_VIOLATION_CLASS_2_REACHABILITY`。
    """
    if bool(contract.type_contract_ok):
        return ""
    if str(contract.returned_kind) == KIND_NONE:
        return "returned_none_expected_entry"
    if str(contract.returned_kind) == KIND_QA and str(contract.expected_kind) == KIND_TEXT:
        return "returned_qa_expected_text"
    if str(contract.returned_kind) == KIND_TEXT and str(contract.expected_kind) == KIND_QA:
        return "returned_text_expected_qa"
    raise ValueError(
        "无法归类的类型契约失败：returned_kind="
        f"{contract.returned_kind!r} / expected_kind={contract.expected_kind!r}"
    )


def empty_violation_counts() -> Dict[str, int]:
    """类型契约违背的**零值计数表**（键集合 = :data:`TYPE_VIOLATION_CLASSES`）。"""
    return {str(name): 0 for name in TYPE_VIOLATION_CLASSES}


def content_view_violations(
    merged: "MergedEntryKeyTable",
    returned_indices: Sequence[int],
    expected_kinds: Sequence[str],
) -> Dict[str, Any]:
    """**第②类违规的独立判定**（F2 修复）：只看**输出内容**，不看 ``kind`` 字段。

    口径
    ----
    对每一次「返回下标 + 期望类型」判定：若 ``expected_kind == 'qa'`` 且返回物
    **不属于任何 qa 条目的答案展示文本**（按 ``outputs_index`` 的内容视图判），
    则记一次第②类违规。返回下标为 ``-1`` / ``returned_kind == 'none'`` 的情形归
    **第①类**（未命中），**不计入**第②类。

    **为什么不看 kind 字段**：这正是「独立校验」的意义 —— 若实现只复述
    ``entry_kinds[returned_index]``，那它与第①类的判定就是同一个视图、无法互相验证。
    本函数走的是「输出文本是否在 qa 展示文本集合内」这条**内容**路径。

    参数
    ----
    merged : MergedEntryKeyTable
        合并表（提供 ``outputs_index`` 内容视图与 ``table.outputs``）。
    returned_indices : Sequence[int]
        每次判定的返回下标（``-1`` 表示未命中）。
    expected_kinds : Sequence[str]
        与 ``returned_indices`` 同长的期望类型序列。

    返回
    ----
    Dict[str, Any]
        ``n_class2_violations`` / ``samples_head``（最多 20 条逐样本）/
        ``reachability``（可达性说明）/ ``rule``。

    异常
    ------
    ValueError
        两个序列长度不一致。
    """
    idx = [int(x) for x in returned_indices]
    kinds = [str(x) for x in expected_kinds]
    if len(idx) != len(kinds):
        raise ValueError(
            f"returned_indices 长度 {len(idx)} 与 expected_kinds 长度 {len(kinds)} 不一致"
        )
    n = 0
    samples: List[Dict[str, Any]] = []
    for pos, (i, exp) in enumerate(zip(idx, kinds)):
        if int(i) < 0:
            continue  # 未命中：属第①类
        if str(exp) != KIND_QA:
            continue
        out = str(merged.table.outputs[int(i)])
        # **内容视图**独立判定：该输出必须被登记为某个 qa 条目的展示文本
        if str(merged.outputs_index.get(out, "")) != KIND_QA:
            n += 1
            if len(samples) < 20:
                samples.append(
                    {
                        "position": int(pos),
                        "returned_index": int(i),
                        "expected_kind": str(exp),
                        "returned_kind_by_field": str(merged.kind_of(int(i))),
                        "returned_output_sha256": sha256_bytes(out.encode("utf-8")),
                        "content_view_kind": str(merged.outputs_index.get(out, "")),
                    }
                )
    return {
        "n_class2_violations": int(n),
        "n_checked": int(len(idx)),
        "samples_head": samples,
        "reachability": str(RETURN_TYPE_VIOLATION_CLASS_2_REACHABILITY),
        "rule": (
            "第②类 = `expected_kind == 'qa'` 且返回物**不在**「qa 条目展示文本集合」内；"
            "判定走**输出内容**（`outputs_index`）而非 `kind` 字段；返回下标 < 0 归第①类。"
            "**非恒真**：对伪造内容视图（把某条 qa 的输出改成未登记串）必须报出违规。"
        ),
    }


# ---------------------------------------------------------------------------
# 合并条目表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeProvenance:
    """两半的来源元数据（合并后仍**完整保留**，不得丢弃）。

    参数
    ----
    table : EntryKeyTable
        该半的条目表本体。
    meta : Dict[str, Any]
        该半的 :meth:`EntryKeyTable.meta`（含 dim / norm_spec / encoder_profile /
        encoder_fingerprint / key_table_sha256 / source）。
    kind : str
        该半的条目类型（``qa`` / ``text``）。
    """

    table: EntryKeyTable
    meta: Dict[str, Any]
    kind: str

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化（``meta`` 原样搬运，不做任何字段裁剪）。"""
        return {
            "kind": str(self.kind),
            "n_entries": int(self.table.size),
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class MergedEntryKeyTable:
    """**合并条目表**：一张 :class:`EntryKeyTable` + 逐条目的类型标记。

    设计要点
    --------
    * 合并表**不是**新发明的容器：它复用 1a 的 :class:`EntryKeyTable`
      （键 ``[N, D]`` float32 逐行 L2 归一化 + ``entry_ids`` + ``outputs`` + ``meta``）；
    * 新增的**唯一**结构是 :attr:`entry_kinds`（逐条目的返回类型标记）；按类型的
      下标区间由 :meth:`half_slice` / :meth:`indices_of` **现场推导并交叉校验**；
    * 类型标记同时由**内容不变量**背书：``entry_kinds[i]`` 必须等于该条目在
      :attr:`outputs_index` 中按内容归类的类型 —— 两个视图不一致即构造期报错。

    参数
    ----
    table : EntryKeyTable
        合并后的键表（``kind='merged'``）。
    entry_kinds : Sequence[str]
        逐条目类型（``qa`` / ``text``），长度 == ``table.size``。
    outputs_index : Dict[str, str]
        输出文本 -> 类型的**内容索引**（用于「按内容独立判定类型」的第二视图）。
    halves : Dict[str, MergeProvenance]
        两半的来源元数据（``qa`` / ``text``）。
    merge_order : Sequence[str]
        合并顺序（固定 :data:`MERGE_ORDER`）。
    """

    table: EntryKeyTable
    entry_kinds: List[str]
    outputs_index: Dict[str, str]
    halves: Dict[str, MergeProvenance]
    merge_order: List[str] = field(default_factory=lambda: list(MERGE_ORDER))

    def __post_init__(self) -> None:
        n = int(self.table.size)
        if len(self.entry_kinds) != n:
            raise ValueError(
                f"entry_kinds 数 {len(self.entry_kinds)} 与合并表行数 {n} 不一致"
            )
        bad = sorted({str(k) for k in self.entry_kinds} - set(ENTRY_KINDS))
        if bad:
            raise ValueError(
                f"entry_kinds 出现未登记的类型 {bad}；允许集合 = {list(ENTRY_KINDS)}"
            )
        if list(self.merge_order) != list(MERGE_ORDER):
            raise ValueError(
                f"merge_order={list(self.merge_order)} 与冻结顺序 {list(MERGE_ORDER)} 不一致"
            )
        if sorted(self.halves.keys()) != sorted(ENTRY_KINDS):
            raise ValueError(
                f"halves 的键集合 {sorted(self.halves.keys())} 必须恰为 {sorted(ENTRY_KINDS)}"
            )
        # 两半条目数必须与 entry_kinds 的分类计数一致（构造期不变量）。
        for kind in ENTRY_KINDS:
            want = int(sum(1 for k in self.entry_kinds if str(k) == kind))
            got = int(self.halves[kind].table.size)
            if want != got:
                raise ValueError(
                    f"类型 {kind!r} 的条目数不一致：entry_kinds 计数 {want} vs "
                    f"来源表 {got}"
                )
        # 逐条目：**内容视图**（outputs_index 反查）必须与 entry_kinds 一致。
        mismatch: List[int] = []
        for i, kind in enumerate(self.entry_kinds):
            expected = self.outputs_index.get(str(self.table.outputs[i]))
            if expected is None or str(expected) != str(kind):
                if len(mismatch) < 10:
                    mismatch.append(int(i))
        if mismatch:
            raise ValueError(
                "类型标记与「按输出内容独立归类的第二视图」不一致，行下标 = "
                f"{mismatch}；拒绝在无法自洽的类型标记上出数"
            )

    # -- 基本属性 ------------------------------------------------------

    @property
    def size(self) -> int:
        """条目总数 ``N``（本轮口径下 = 2814）。"""
        return int(self.table.size)

    @property
    def dim(self) -> int:
        """连接参数 ``D``。"""
        return int(self.table.dim)

    def half_size(self, kind: str) -> int:
        """某一类型的条目数。"""
        return int(self.halves[str(kind)].table.size)

    def half_slice(self, kind: str) -> slice:
        """某一类型在合并表中的**连续**下标区间（依据 :data:`MERGE_ORDER`）。"""
        offset = 0
        for name in MERGE_ORDER:
            if str(name) == str(kind):
                return slice(offset, offset + self.half_size(name))
            offset += self.half_size(name)
        raise KeyError(f"未登记的类型 {kind!r}；可用 = {list(MERGE_ORDER)}")

    def indices_of(self, kind: str) -> List[int]:
        """某一类型在合并表中的全部下标（**连续**区间，故与 :meth:`half_slice` 一致）。"""
        span = self.half_slice(str(kind))
        got = list(range(int(span.start), int(span.stop)))
        # 交叉校验：区间必须**恰好**是该 kind 的条目集合（防止合并顺序被静默改动）。
        expect = [i for i, k in enumerate(self.entry_kinds) if str(k) == str(kind)]
        if got != expect:
            raise ValueError(
                f"类型 {kind!r} 的下标区间 {got[:3]}... 与该类型的条目集合不完全一致；"
                "合并顺序已被改动，拒绝出数"
            )
        return got

    # -- 查表与类型契约 ------------------------------------------------

    def kind_of(self, index: int) -> str:
        """第 ``index`` 行的条目类型。"""
        i = int(index)
        if not (0 <= i < self.size):
            raise IndexError(f"行下标 {i} 越界（合并表大小 {self.size}）")
        return str(self.entry_kinds[i])

    def lookup_typed(self, index: int) -> Tuple[bytes, LookupContract]:
        """**带类型标记的逐位精确查表**。

        返回
        ----
        Tuple[bytes, LookupContract]
            ``(该行 float32 裸字节, 类型契约对象)``。``expected_kind`` 与
            ``returned_kind`` 都取自 :meth:`kind_of`（返回物类型由条目表自身判定，
            不由调用方口头声明）。
        """
        i = int(index)
        raw, out = self.table.lookup(i)
        kind = self.kind_of(i)
        return raw, LookupContract(
            index=i,
            returned_kind=kind,
            expected_kind=kind,
            returned_output=str(out),
        )

    def contract_for(self, index: int, expected_kind: str) -> LookupContract:
        """构造「某次查询返回第 ``index`` 行」的类型契约（含查询期望类型）。"""
        i = int(index)
        kind = self.kind_of(i)
        return LookupContract(
            index=i,
            returned_kind=kind,
            expected_kind=str(expected_kind),
            returned_output=str(self.table.outputs[i]),
        )

    def miss_contract(self, expected_kind: str) -> LookupContract:
        """构造「未命中」的类型契约（``index=-1`` / ``returned_kind='none'``）。"""
        return LookupContract(
            index=-1,
            returned_kind=KIND_NONE,
            expected_kind=str(expected_kind),
            returned_output="",
        )

    def sha256(self) -> str:
        """合并表**内容指纹**（schema + 形状 + 裸字节 + id + 输出 + 类型标记）。

        口径与单侧表 :func:`entry_table.key_table_fingerprint` 同源（裸字节优先），
        但 schema 不同（:data:`MERGE_FINGERPRINT_SCHEMA`）且**并入类型标记**。
        """
        payload = {
            "schema": MERGE_FINGERPRINT_SCHEMA,
            "version": MERGE_TABLE_VERSION,
            "shape": [int(x) for x in self.table.keys.shape],
            "dtype": str(self.table.keys.dtype).replace("torch.", ""),
            "dim": int(self.table.dim),
            "keys_bytes_sha256": sha256_bytes(
                self.table.keys.detach().cpu().contiguous().numpy().tobytes()
            ),
            "entry_ids": [str(x) for x in self.table.entry_ids],
            "outputs": [str(x) for x in self.table.outputs],
            "entry_kinds": [str(x) for x in self.entry_kinds],
            "merge_order": [str(x) for x in self.merge_order],
        }
        return sha256_bytes(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")
        )

    def half_sha256(self, kind: str) -> str:
        """某一半的**内容指纹**（与 :meth:`EntryKeyTable.sha256` 同一口径）。"""
        return str(self.halves[str(kind)].table.sha256())

    def encoder_consistency(self) -> Dict[str, Any]:
        """两半的**编码器口径一致性**取证（离朱 R74 D1：必须在顶层可见）。

        为什么单列一个方法：合并表把「题面编码器」的口径写在 ``EntryKeyTable`` 的
        ``encoder_profile`` / ``encoder_fingerprint`` 上（沿用 1a 的字段），若两半的
        口径其实**不同**，顶层字段就只代表其中一半 —— 这是必须**显式登记**的事实，
        不得让人以为「顶层那一串就是整张表的口径」。
        """
        per_half = {
            str(k): {
                "encoder_profile": str(self.halves[k].meta.get("encoder_profile", "")),
                "encoder_fingerprint": str(
                    self.halves[k].meta.get("encoder_fingerprint", "")
                ),
            }
            for k in MERGE_ORDER
        }
        profiles = {v["encoder_profile"] for v in per_half.values()}
        fingerprints = {v["encoder_fingerprint"] for v in per_half.values()}
        return {
            "per_half": per_half,
            "profiles_equal": bool(len(profiles) == 1),
            "fingerprints_equal": bool(len(fingerprints) == 1),
            "top_level_field_source": (
                "合并表的顶层 encoder_profile / encoder_fingerprint 沿用**前半区**"
                f"（{MERGE_ORDER[0]!r}）的取值（构造时透传）；两半口径不同时以本块的"
                "per_half 为准"
            ),
        }

    def meta(self) -> Dict[str, Any]:
        """合并表元数据（两侧元数据**完整保留**，见 ``halves``）。

        顶层除 1a 的通用字段外，**另含** ``encoder_profile`` / ``encoder_fingerprint``
        （沿用前半区取值）与 ``encoder_consistency``（两半口径一致性取证）。
        """
        return {
            "version": MERGE_TABLE_VERSION,
            "kind": "merged",
            "n_entries": int(self.size),
            "n_entries_by_kind": {str(k): int(self.half_size(k)) for k in MERGE_ORDER},
            "dim": int(self.dim),
            "norm_spec": str(self.table.norm_spec),
            "dtype": str(self.table.keys.dtype).replace("torch.", ""),
            "contiguous": bool(self.table.keys.is_contiguous()),
            "norm_range": self.table.norm_range(),
            "encoder_profile": str(self.table.encoder_profile),
            "encoder_fingerprint": str(self.table.encoder_fingerprint),
            "encoder_consistency": self.encoder_consistency(),
            "merge_key_table_sha256": self.sha256(),
            "merge_order": [str(x) for x in self.merge_order],
            "return_kinds": [str(x) for x in RETURN_KINDS],
            "kind_semantics": dict(KIND_SEMANTICS),
            "tau": float(TAU),
            "entry_ids_head": [str(x) for x in self.table.entry_ids[:3]],
            "outputs_head": [str(x) for x in self.table.outputs[:3]],
            "halves": {str(k): self.halves[k].as_dict() for k in MERGE_ORDER},
        }


# ---------------------------------------------------------------------------
# 合并前置断言（必须先跑，任一失败即停止）
# ---------------------------------------------------------------------------


def role_bitwise_identity(
    texts: Sequence[str],
    *,
    profile: str,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """断言 ②：``local-hash`` 两个角色（``question`` / ``text_line``）**逐位相同**。

    「同一文本在 question / text_line 两角色下字节一致」是「两侧键同 D 同一张表」
    的前提：若两角色口径不同，合并表内部就会有两套不可比的向量。

    返回
    ----
    Dict[str, Any]
        ``n_checked`` / ``n_bitwise_equal`` / ``exact_frac`` / ``mismatch_head`` /
        ``encoder_fingerprints``（两角色各自的口径指纹）。
    """
    if str(profile) != "lexical-88":
        raise ValueError(
            f"逐位同一性检查只对 hash 家族档口径成立（当前 profile={profile!r}）："
            "hf 家族的两个角色 max_length 不同（512 / 8192），长文本上**必然**不同口径"
        )
    vec_q = build_vectorizer(EncoderConfig(name=ENCODER_LOCAL_HASH, role=ROLE_QUESTION))
    vec_t = build_vectorizer(EncoderConfig(name=ENCODER_LOCAL_HASH, role=ROLE_TEXT_LINE))
    rows = [str(t) for t in texts]
    mat_q = encode_statements(vec_q, rows, batch_size=batch_size)
    mat_t = encode_statements(vec_t, rows, batch_size=batch_size)
    if mat_q.shape != mat_t.shape:
        raise ValueError(
            f"两角色编码形状不一致：question {mat_q.shape} vs text_line {mat_t.shape}"
        )
    mismatch: List[Dict[str, Any]] = []
    n_equal = 0
    for i in range(mat_q.shape[0]):
        a = np.ascontiguousarray(mat_q[i], dtype=np.float32).tobytes()
        b = np.ascontiguousarray(mat_t[i], dtype=np.float32).tobytes()
        if a == b:
            n_equal += 1
        elif len(mismatch) < 10:
            mismatch.append(
                {
                    "row": int(i),
                    "sha256_question": sha256_bytes(a),
                    "sha256_text_line": sha256_bytes(b),
                    "max_abs_diff": float(
                        np.max(
                            np.abs(
                                mat_q[i].astype(np.float64) - mat_t[i].astype(np.float64)
                            )
                        )
                    ),
                }
            )
    n = int(mat_q.shape[0])
    return {
        "n_checked": int(n),
        "n_bitwise_equal": int(n_equal),
        "exact_frac": float(n_equal / max(1, n)),
        "mismatch_head": mismatch,
        "encoder_fingerprints": {
            ROLE_QUESTION: str(vec_q.fingerprint()),
            ROLE_TEXT_LINE: str(vec_t.fingerprint()),
        },
        "rule": (
            "同一文本在 role=question 与 role=text_line 两角色下，编码结果的 float32 "
            "裸字节必须逐字节相等（hash 家族不使用 max_length）"
        ),
    }


def raw_keys_sha256(table: EntryKeyTable) -> str:
    """整张表的**裸字节** SHA256（单一比较口径；与 ``TextRowKeyTable.sha256()`` 同源）。

    **为什么必须单列一个口径**：条目表有两套**不同**的指纹方案，二者不可互比 ——
    ``EntryKeyTable.sha256()`` 是**内容指纹**（schema + 形状 + 裸字节 + 条目 id +
    绑定输出 + 维度 的规范化 JSON SHA256），而这里只对**键的裸字节**取 SHA256。
    「合并前后字节零变化」这条断言必须用**同一个**方案比较，否则会把两种指纹的
    天然不同误判成字节漂移（历史纠正记录的口径纪律）。
    """
    block = table.keys.detach().cpu().contiguous().numpy().tobytes()
    return sha256_bytes(block)


def half_key_bytes_sha256(merged: MergedEntryKeyTable, kind: str) -> str:
    """合并表某一半的**裸字节** SHA256（与 :func:`raw_keys_sha256` 同一口径）。"""
    span = merged.half_slice(str(kind))
    block = merged.table.keys[span.start : span.stop].detach().cpu().contiguous()
    return sha256_bytes(block.numpy().tobytes())


def merge_bytes_unchanged(
    merged: MergedEntryKeyTable,
    tables: Dict[str, EntryKeyTable],
) -> Dict[str, Any]:
    """断言 ③：合并前后各半的键字节**零变化**（禁止合并环节重编码 / 重归一化）。

    返回
    ----
    Dict[str, Any]
        逐半的 ``n_checked`` / ``n_bitwise_equal`` / ``exact_frac`` / ``mismatch_head``
        与该半合并前后的 SHA256 对照。
    """
    out: Dict[str, Any] = {}
    for kind in MERGE_ORDER:
        src = tables[str(kind)]
        span = merged.half_slice(str(kind))
        n_equal = 0
        mismatch: List[Dict[str, Any]] = []
        for j in range(src.size):
            a = src.row_bytes(int(j))
            b = merged.table.row_bytes(int(span.start) + int(j))
            if a == b:
                n_equal += 1
            elif len(mismatch) < 10:
                mismatch.append(
                    {
                        "source_index": int(j),
                        "merged_index": int(span.start) + int(j),
                        "source_sha256": sha256_bytes(a),
                        "merged_sha256": sha256_bytes(b),
                    }
                )
        out[str(kind)] = {
            "n_checked": int(src.size),
            "n_bitwise_equal": int(n_equal),
            "exact_frac": float(n_equal / max(1, src.size)),
            "mismatch_head": mismatch,
            "source_raw_keys_sha256": str(raw_keys_sha256(src)),
            "merged_half_raw_keys_sha256": str(half_key_bytes_sha256(merged, str(kind))),
            "source_content_fingerprint": str(src.sha256()),
            "raw_keys_sha256_rule": (
                "两侧同为**键的裸字节** SHA256（与 TextRowKeyTable.sha256() 同源）；"
                "不与 EntryKeyTable.sha256() 的内容指纹互比（两者是不同的指纹方案）"
            ),
        }
    return out


def premerge_assertions(
    qa_table: EntryKeyTable,
    text_table: EntryKeyTable,
    merged: MergedEntryKeyTable,
    *,
    profile: str,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """**合并前置断言**：三条全过才允许出数（任一失败即抛，不静默继续）。

    现场枚举（唯一来源，见 :data:`PREMERGE_ASSERTION_KEYS`）
    ---------------------------------------------------------
    1. ``same_dtype_and_norm``：两侧键同 ``D``、同 ``dtype``、各自逐行 L2 归一化；
    2. ``role_bitwise_identity``：``local-hash`` 两角色对同一文本逐位相同；
    3. ``merge_key_bytes_unchanged``：合并前后各半的键字节零变化。

    断言②用的文本集 = 文本侧**全量行原文**（2665 条），足以覆盖真实数据分布；
    它**只**验证「角色 -> 编码字节」这条映射，不涉及任何查询 / 金标。
    """
    # --- 断言 1 -------------------------------------------------------
    details_1: Dict[str, Any] = {
        "qa_dim": int(qa_table.dim),
        "text_dim": int(text_table.dim),
        "same_dim": bool(int(qa_table.dim) == int(text_table.dim)),
        "qa_dtype": str(qa_table.keys.dtype).replace("torch.", ""),
        "text_dtype": str(text_table.keys.dtype).replace("torch.", ""),
        "same_dtype": bool(qa_table.keys.dtype == text_table.keys.dtype),
        "qa_norm_spec": str(qa_table.norm_spec),
        "text_norm_spec": str(text_table.norm_spec),
        "same_norm_spec": bool(str(qa_table.norm_spec) == str(text_table.norm_spec)),
        "qa_norm_range": qa_table.norm_range(),
        "text_norm_range": text_table.norm_range(),
        "norm_atol": float(NORM_ATOL),
    }
    for name, rng in (
        ("qa", details_1["qa_norm_range"]),
        ("text", details_1["text_norm_range"]),
    ):
        if not (
            float(rng["min"]) >= 1.0 - NORM_ATOL and float(rng["max"]) <= 1.0 + NORM_ATOL
        ):
            raise ValueError(
                f"断言 ① 失败：{name} 侧行范数区间 {rng} 超出 L2 归一化允差 "
                f"[{1.0 - NORM_ATOL}, {1.0 + NORM_ATOL}]"
            )
    if not (
        details_1["same_dim"] and details_1["same_dtype"] and details_1["same_norm_spec"]
    ):
        raise ValueError(
            f"断言 ① 失败：两侧键不同 D / dtype / 归一化口径，详情 = {details_1}"
        )

    # --- 断言 2 -------------------------------------------------------
    details_2 = role_bitwise_identity(
        [str(t) for t in text_table.outputs], profile=str(profile), batch_size=batch_size
    )
    if float(details_2["exact_frac"]) != 1.0:
        raise ValueError(
            "断言 ② 失败：local-hash 两角色对同一文本并非逐位相同，"
            f"n_bitwise_equal={details_2['n_bitwise_equal']}/"
            f"{details_2['n_checked']}，样例 = {details_2['mismatch_head'][:3]}"
        )

    # --- 断言 3 -------------------------------------------------------
    details_3 = merge_bytes_unchanged(merged, {KIND_QA: qa_table, KIND_TEXT: text_table})
    for kind in MERGE_ORDER:
        block = details_3[str(kind)]
        if float(block["exact_frac"]) != 1.0:
            raise ValueError(
                f"断言 ③ 失败：{kind} 半区合并前后键字节有变化，"
                f"n_bitwise_equal={block['n_bitwise_equal']}/{block['n_checked']}"
            )
        if str(block["source_raw_keys_sha256"]) != str(
            block["merged_half_raw_keys_sha256"]
        ):
            raise ValueError(
                f"断言 ③ 失败：{kind} 半区合并前后**键裸字节** SHA256 不一致（"
                f"{str(block['source_raw_keys_sha256'])[:16]}... vs "
                f"{str(block['merged_half_raw_keys_sha256'])[:16]}...）"
            )
    return {
        "all_passed": True,
        "assertions": {
            "same_dtype_and_norm": {"passed": True, "detail": details_1},
            "role_bitwise_identity": {"passed": True, "detail": details_2},
            "merge_key_bytes_unchanged": {"passed": True, "detail": details_3},
        },
        "assertion_order": list(PREMERGE_ASSERTION_KEYS),
    }


# ---------------------------------------------------------------------------
# 合并表构造
# ---------------------------------------------------------------------------


def build_merged_table(
    qa_table: EntryKeyTable,
    text_table: EntryKeyTable,
) -> MergedEntryKeyTable:
    """把两侧条目表**拼接**为单张 :class:`EntryKeyTable`（+ 逐条目类型标记）。

    关键约束
    --------
    1. **只拼接**：``keys = cat(qa.keys, text.keys)``；不重编码、不重归一化、不排序；
    2. 两侧元数据**完整保留**（:class:`MergeProvenance`）；
    3. 类型标记由两个**独立视图**背书：写入的 ``entry_kinds`` 与「按输出内容归类」
       的 :attr:`MergedEntryKeyTable.outputs_index` —— 二者不一致即构造期报错。
    """
    if int(qa_table.dim) != int(text_table.dim):
        raise ValueError(
            f"两侧维度不一致：qa {qa_table.dim} vs text {text_table.dim}；"
            "合并表要求两侧键同 D"
        )
    if qa_table.keys.dtype != text_table.keys.dtype:
        raise ValueError(
            f"两侧 dtype 不一致：{qa_table.keys.dtype} vs {text_table.keys.dtype}"
        )
    qa_keys = qa_table.keys.detach().to(torch.float32).cpu().contiguous()
    text_keys = text_table.keys.detach().to(torch.float32).cpu().contiguous()
    keys = torch.cat([qa_keys, text_keys], dim=0).contiguous()
    entry_ids = [str(x) for x in qa_table.entry_ids] + [
        str(x) for x in text_table.entry_ids
    ]
    outputs = [str(x) for x in qa_table.outputs] + [str(x) for x in text_table.outputs]
    entry_kinds = [KIND_QA] * int(qa_table.size) + [KIND_TEXT] * int(text_table.size)
    merged = EntryKeyTable(
        keys=keys,
        entry_ids=entry_ids,
        outputs=outputs,
        kind="merged",
        dim=int(keys.shape[1]),
        norm_spec=str(qa_table.norm_spec),
        encoder_profile=str(qa_table.encoder_profile),
        encoder_fingerprint=str(qa_table.encoder_fingerprint),
        source={
            "merge_table_version": MERGE_TABLE_VERSION,
            "merge_order": list(MERGE_ORDER),
            "halves": {
                KIND_QA: {
                    "n_entries": int(qa_table.size),
                    "encoder_profile": str(qa_table.encoder_profile),
                    "encoder_fingerprint": str(qa_table.encoder_fingerprint),
                    "key_table_sha256": str(qa_table.sha256()),
                    "role": ROLE_QUESTION,
                },
                KIND_TEXT: {
                    "n_entries": int(text_table.size),
                    "encoder_profile": str(text_table.encoder_profile),
                    "encoder_fingerprint": str(text_table.encoder_fingerprint),
                    "key_table_sha256": str(text_table.sha256()),
                    "role": ROLE_TEXT_LINE,
                },
            },
        },
    )
    outputs_index: Dict[str, str] = {}
    for kind, tbl in ((KIND_QA, qa_table), (KIND_TEXT, text_table)):
        for out in tbl.outputs:
            key = str(out)
            prev = outputs_index.get(key)
            if prev is not None and prev != str(kind):
                raise ValueError(
                    "两侧存在输出文本相同但类型不同的条目，内容视图无法唯一判定类型："
                    f"output={key!r}，已登记 {prev!r}、又见 {kind!r}；"
                    "拒绝在无法唯一判定类型的合并表上出数"
                )
            outputs_index[key] = str(kind)
    return MergedEntryKeyTable(
        table=merged,
        entry_kinds=entry_kinds,
        outputs_index=outputs_index,
        halves={
            KIND_QA: MergeProvenance(table=qa_table, meta=qa_table.meta(), kind=KIND_QA),
            KIND_TEXT: MergeProvenance(
                table=text_table, meta=text_table.meta(), kind=KIND_TEXT
            ),
        },
        merge_order=list(MERGE_ORDER),
    )


# ---------------------------------------------------------------------------
# 打分（两臂共用同一批键，唯一变量是路由结构）
# ---------------------------------------------------------------------------


def topk_scores(
    query_feats: np.ndarray,
    keys: torch.Tensor,
    *,
    index: Optional[Sequence[int]] = None,
    topk: int = 1,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """在（可选的）候选下标子集上取余弦 top-k。

    参数
    ----
    query_feats : np.ndarray
        ``[Q, D]`` float32，**逐行 L2 归一化**（否则余弦无意义；本函数校验）。
    keys : torch.Tensor
        ``[N, D]`` float32 合并表键。
    index : Optional[Sequence[int]]
        候选行下标（``None`` = 全表）。两臂的差别就在这里：QA 优先臂先只在 QA 半区，
        合并表臂用全表。
    topk : int
        取前 k 个（本轮主判据只用 top-1，topk 留给诊断）。
    batch_size : int
        查询批大小。

    返回
    ----
    Tuple[np.ndarray, np.ndarray]
        ``(scores, indices)``，形状均为 ``[Q, k]``（``indices`` 是**合并表全局下标**，
        按分数降序、并列按下标升序，保证确定性）。
    """
    feats = np.asarray(query_feats, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError(f"query_feats 必须是二维 [Q, D]，当前形状 = {feats.shape}")
    if int(feats.shape[1]) != int(keys.shape[1]):
        raise ValueError(
            f"查询维度 {int(feats.shape[1])} 与键表维度 {int(keys.shape[1])} 不一致"
        )
    if feats.shape[0]:
        norms = np.linalg.norm(feats.astype(np.float64), axis=1)
        bad = int(np.sum(np.abs(norms - 1.0) > NORM_ATOL))
        if bad:
            raise ValueError(
                f"有 {bad} 条查询特征未做 L2 归一化（允差 {NORM_ATOL}）；"
                "拒绝在未归一化的特征上做余弦检索"
            )
    k = max(1, int(topk))
    candidates = (
        torch.arange(int(keys.shape[0]), dtype=torch.long)
        if index is None
        else torch.as_tensor([int(i) for i in index], dtype=torch.long)
    )
    sub = keys.index_select(0, candidates).detach()
    if int(sub.shape[0]) < k:
        raise ValueError(f"候选集合大小 {int(sub.shape[0])} 小于 topk={k}；无法取 top-k")
    q = int(feats.shape[0])
    scores_out = np.zeros((q, k), dtype=np.float32)
    indices_out = np.zeros((q, k), dtype=np.int64)
    step = max(1, int(batch_size))
    for b0 in range(0, q, step):
        block = torch.from_numpy(np.ascontiguousarray(feats[b0 : b0 + step]))
        sim = block @ sub.t()
        order = torch.argsort(sim, dim=1, descending=True, stable=True)[:, :k]
        vals = torch.gather(sim, 1, order)
        global_idx = candidates.index_select(0, order.reshape(-1)).reshape(order.shape)
        scores_out[b0 : b0 + block.shape[0]] = vals.detach().cpu().numpy()
        indices_out[b0 : b0 + block.shape[0]] = global_idx.detach().cpu().numpy()
    return scores_out, indices_out


def arm_top1(
    query_feats: np.ndarray,
    merged: MergedEntryKeyTable,
    *,
    arm: str,
    expected_kind: str,
    tau: float = TAU,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """对一批查询跑**一条臂**的完整路由，返回逐查询结果与类型契约。

    两臂的差别只体现在候选集合与是否分两级：

    * :data:`ARM_QA_PRIORITY`：QA 半区取 top-1；分数 >= ``tau`` 即返回；否则文本半区
      取 top-1；分数 >= ``tau`` 返回，否则「无匹配」；
    * :data:`ARM_MERGED_TOPK1`：全表取 top-1；分数 >= ``tau`` 返回，否则「无匹配」。

    返回
    ----
    Dict[str, Any]
        ``arm`` / ``n`` / ``expected_kind`` / ``tau`` / ``top1_index`` / ``top1_score``
        / ``top1_kind`` / ``returned_index`` / ``returned_kind`` / ``returned_output``
        / ``type_contract_ok`` / ``type_contract_violation`` / ``stage``
        （``stage ∈ {'qa','text','none'}``）。
    """
    key = str(arm)
    if key not in ARM_ORDER:
        raise ValueError(f"未登记的路由臂 {arm!r}；可用 = {list(ARM_ORDER)}")
    exp = str(expected_kind)
    if exp not in ENTRY_KINDS:
        raise ValueError(f"expected_kind={expected_kind!r} 不在 {list(ENTRY_KINDS)} 内")
    n = int(np.asarray(query_feats).shape[0])
    if key == ARM_QA_PRIORITY:
        qa_idx = merged.indices_of(KIND_QA)
        text_idx = merged.indices_of(KIND_TEXT)
        sc_q, ix_q = topk_scores(
            query_feats, merged.table.keys, index=qa_idx, topk=1, batch_size=batch_size
        )
        sc_t, ix_t = topk_scores(
            query_feats, merged.table.keys, index=text_idx, topk=1, batch_size=batch_size
        )
        take_qa = sc_q[:, 0] >= float(tau)
        chosen = np.where(take_qa, ix_q[:, 0], ix_t[:, 0])
        chosen_score = np.where(take_qa, sc_q[:, 0], sc_t[:, 0])
        hit = chosen_score >= float(tau)
        stage = [
            "none"
            if not bool(hit[i])
            else (KIND_QA if bool(take_qa[i]) else KIND_TEXT)
            for i in range(n)
        ]
    else:
        sc_all, ix_all = topk_scores(
            query_feats, merged.table.keys, index=None, topk=1, batch_size=batch_size
        )
        chosen = ix_all[:, 0]
        chosen_score = sc_all[:, 0]
        hit = chosen_score >= float(tau)
        stage = [
            "none"
            if not bool(hit[i])
            else (KIND_QA if merged.kind_of(int(chosen[i])) == KIND_QA else KIND_TEXT)
            for i in range(n)
        ]
    top1_index: List[int] = []
    top1_score: List[float] = []
    top1_kind: List[str] = []
    returned_index: List[int] = []
    returned_kind: List[str] = []
    returned_output: List[str] = []
    contract_ok: List[bool] = []
    violations: List[str] = []
    for i in range(n):
        idx = int(chosen[i])
        top1_index.append(idx)
        top1_score.append(float(chosen_score[i]))
        top1_kind.append(str(merged.kind_of(idx)))
        contract = (
            merged.contract_for(idx, exp)
            if bool(hit[i])
            else merged.miss_contract(exp)
        )
        returned_index.append(int(contract.index))
        returned_kind.append(str(contract.returned_kind))
        returned_output.append(str(contract.returned_output))
        contract_ok.append(bool(contract.type_contract_ok))
        violations.append(str(classify_type_violation(contract)))
    return {
        "arm": key,
        "n": int(n),
        "expected_kind": exp,
        "tau": float(tau),
        "top1_index": top1_index,
        "top1_score": top1_score,
        "top1_kind": top1_kind,
        "returned_index": returned_index,
        "returned_kind": returned_kind,
        "returned_output": returned_output,
        "type_contract_ok": contract_ok,
        "type_contract_violation": violations,
        "stage": [str(x) for x in stage],
    }


# ---------------------------------------------------------------------------
# 冻结变体产物读取（只读；禁止重新生成）
# ---------------------------------------------------------------------------


def load_frozen_variants(path: str) -> List[Dict[str, Any]]:
    """读取**已冻结**的变体产物（``variants.jsonl``；只读，不重新生成）。

    口径：变体必须**冻结落盘**（含 SHA256、生成轮次与来源题 id 映射）。本函数只读
    这些字节，**不**调用任何生成器（否则每次验收都会重新生成，违反冻结纪律）。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"缺少已冻结的变体产物：{path!r}；本模块禁止重新生成变体（冻结纪律）"
        )
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"变体产物为空：{path!r}")
    return records


def variant_fields_of(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """现场枚举变体记录的字段名与 4 支变体键（**禁止凭记忆手写**）。"""
    sample = dict(records[0])
    variants = dict(sample.get("variants", {}))
    return {
        "record_keys": sorted(str(k) for k in sample.keys()),
        "variant_keys": sorted(str(k) for k in variants.keys()),
        "n_records": int(len(records)),
    }
