"""轮 8：QA 侧答案表**同义归并**（离线、可审计、默认关闭）。

背景（为什么需要这一轮）
------------------------
QA 侧不可学的结构性原因之一是**标签噪声**：统一答案表（``checkpoints/qa_learn/dataset/
answer_table.jsonl``，现场 298 行）里有若干类是**同一实体的拼写变体**（如
``america(us)`` / ``americaus`` / ``america(unitedstates)`` / ``america'unitedstatesof'``，
``china(people'srepublic)`` / ``chinapeople'srepublic`` / ``chinapeoplesrepublic``）。
这些类**从问题文本上无从区分**（同一批问题可能同时命中多个变体），故它们在 macro 里的
贡献恒为 0 —— 不是模型学不会，而是**标签本身不可区分**。

本轮**只做数据层的离线对照**：不碰上游 ``n3d_qa``、不改模型、不做任何训练。

四条规则（**逐条可追溯；规则编号进产物**）
------------------------------------------
* **R1a 文本型保守归一**：小写 -> 去首尾空白 -> 折叠内部空白 -> 去**纯装饰性标点**
  （逗号 / 句点 / 撇号 / 引号，见 :data:`DECORATION_CHARS`）。
* **R1c 包裹式括号剥离**：仅当「剥离括号及其内容后**非空**」且「**主体字母部分不变**」
  才剥离（如 ``america(us) -> america``）；括号内容含数字或运算符时**不剥离**；
  括号不配平时**不剥离**（fail-closed）。
* **R2 显式同义表**（:data:`SYNONYM_RULES`）：**小型冻结表**，每条给出 ``basis``（依据），
  命中逐条落审计；**禁止随意扩张** —— 未采纳的候选见 :data:`SYNONYM_EXCLUSIONS`
  （含否决理由，同样落审计）。
* **R3 禁并项（硬约束）**：任何**含数字或运算符**（:data:`OPERATOR_CHARS`）的答案
  **只做大小写与空白归一** —— 不得删符号、不得剥括号、不得拼接相邻数字；
  因此 ``1`` 与 ``-1``、``1/2`` 与 ``12``、``(1,0)`` 与 ``1,0`` **必须保持独立**。
  带**否定差异**（not / no / non / never / none / 非 / 否 / 不 / 无 / 未）的类**不合并**。
* **R4 安全阀**：簇内成员数 > :data:`HIGH_RISK_MAX_CLUSTER_SIZE`（4）或簇内 display
  相似度下界 < :data:`HIGH_RISK_MIN_DISPLAY_SIMILARITY`（0.5）时**仍然合并**，但标记
  ``high_risk`` 并进**人工复核清单**。

可审计产物（由 :mod:`n3d_qa_learn.answer_merge_run` 落盘）
----------------------------------------------------------
* ``merge_map.jsonl``：逐类映射（原 ``index/key`` -> canonical）+ 逐簇成员与规则依据；
* ``audit.md``：人工复核表（含 high-risk 单独章节、R2 同义表、R3 阻断清单、反例断言）；
* ``before_after.json``：归并前（C=298）与归并后的零训练对照 + 结构决策门；
* ``determinism_check.json``：双跑逐字节一致证据。

**证据文件不自指、不内嵌时间戳**：``determinism_check.json`` 记录的是其它文件的
SHA256/字节数，它自身被排除在比对集合之外；全部产物无挂钟字段。

口径边界（**必须与代码一起读**）
--------------------------------
* 归并**只消标签噪声，不增加样本总量**：簇内类的 ``count`` 相加，问题/行数一个不增。
* canonical 下标是**本模块自造的确定性编号**（按 canonical key 升序），**不是**冻结表
  的全局下标；映射关系全部落在 ``merge_map.jsonl`` 与 :class:`MergePlan` 里。
* 默认关闭：:func:`n3d_qa_learn.qa_scale.build_scale_corpus` 的 ``merge_map=None`` 时
  行为与轮 1 **逐位不变**。
"""

from __future__ import annotations

import difflib
import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 冻结常量（口径进产物）
# ---------------------------------------------------------------------------

#: 模块版本（结构变化必须递增；进产物）。
ANSWER_MERGE_VERSION: str = "n3dqa-answer-merge-v1"

#: 冻结产物目录（**只读**）与统一答案表文件名 —— 与 :mod:`n3d_qa_learn.qa_scale` 同源。
PRODUCT_DIR: str = "checkpoints/qa_learn/dataset"
ANSWER_TABLE_FILE: str = "answer_table.jsonl"

#: **装饰性标点**（R1a 只在「不含数字与运算符」的答案上移除它们）。
DECORATION_CHARS: str = ",.'\"\u2019\u2018\u201c\u201d`\u00b4"

#: **运算符字符**（R3）：出现即进入「只做大小写与空白归一」模式。
#: 注意**不含** ``.`` / ``,`` / 引号 —— 它们是 R1a 的装饰性标点，不是运算符。
OPERATOR_CHARS: str = "+-*/=^<>%\u00b1\u221e\u2264\u2265\u2260\u22a5\u221a![]{}:"

#: **否定词元**（R3）：两份答案的否定签名不同则**不合并**。
NEGATION_TOKENS: Tuple[str, ...] = (
    "not",
    "no",
    "non",
    "never",
    "none",
    "非",
    "否",
    "不",
    "无",
    "未",
)

#: 括号剥离的最大迭代次数（嵌套括号；超过即停，fail-closed 保留剩余括号）。
MAX_PAREN_STRIP: int = 4

#: **R4 安全阀阈值**（两个阈值都进产物）。
HIGH_RISK_MAX_CLUSTER_SIZE: int = 4
HIGH_RISK_MIN_DISPLAY_SIMILARITY: float = 0.5

#: 规则编号（**唯一来源**；产物与审计表只准引用这里的字面量）。
RULE_CASE_WS: str = "R1a-case-ws"
RULE_DECORATION: str = "R1a-decoration"
RULE_PAREN: str = "R1c-paren"
RULE_SYNONYM: str = "R2-synonym"
RULE_RESTRICTED: str = "R3-restricted"
RULE_NEGATION: str = "R3-negation"
RULE_HIGH_RISK: str = "R4-high-risk"

#: 规则文本（进产物 / 审计表；与实现逐条对应）。
RULE_TEXTS: Dict[str, str] = {
    RULE_CASE_WS: "小写 + 去首尾空白 + 折叠内部空白（R1a）",
    RULE_DECORATION: (
        "移除纯装饰性标点（逗号 / 句点 / 撇号 / 引号，集合 = "
        + repr(DECORATION_CHARS)
        + "）；**只**在不含数字与运算符的答案上执行（R1a）"
    ),
    RULE_PAREN: (
        "剥离包裹式括号及其内容：仅当剥离后**非空**且**主体字母部分不变**才剥离；"
        "括号内容含数字/运算符、括号不配平、剥离后为空或剩纯标点时**不剥离**（R1c）"
    ),
    RULE_SYNONYM: "显式同义表命中（条目 id 与依据逐条落审计；禁止随意扩张）（R2）",
    RULE_RESTRICTED: (
        "禁并项：答案含数字或运算符（集合 = " + repr(OPERATOR_CHARS) + "）"
        "⇒ 只做大小写与空白归一，**不得**删符号 / 剥括号 / 拼接数字（R3）"
    ),
    RULE_NEGATION: (
        "禁并项：否定词元签名不同（词元 = " + repr(NEGATION_TOKENS) + "）的类不合并（R3）"
    ),
    RULE_HIGH_RISK: (
        "安全阀：簇内成员数 > %d 或簇内 display 相似度下界 < %.3f ⇒ **仍然合并**但标记 "
        "high_risk 并进人工复核清单（R4）"
        % (HIGH_RISK_MAX_CLUSTER_SIZE, HIGH_RISK_MIN_DISPLAY_SIMILARITY)
    ),
}

#: **R2 显式同义表**（小型冻结表；``aliases`` 是**归一化后**的形式 —— 即 R1 的输出，
#: 故 ``u.s.`` 已经变成 ``us``，无需单列）。每条必须给出 ``basis``。
SYNONYM_RULES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "R2-1",
        "aliases": ("us", "usa"),
        "canonical": "united states",
        "basis": (
            "ISO 3166-1 官方指派：alpha-2 = US、alpha-3 = USA，二者与 United States 是"
            "**同一国家的代码写法**（不是拼写变体）。属「明确登记才允许并」的一类。"
        ),
    },
    {
        "id": "R2-2",
        "aliases": ("united states of america",),
        "canonical": "united states",
        "basis": (
            "ISO 3166-1 / UN 会员国短名：United States of America 的短名即 United States；"
            "全称 -> 短名是**官方等价**，不是语义推断。"
        ),
    },
    {
        "id": "R2-3",
        "aliases": ("uk", "u k"),
        "canonical": "united kingdom",
        "basis": (
            "ISO 3166-1 惯用写法：GB 的通行代码 UK 与 UN 短名 United Kingdom 指同一主权国家。"
        ),
    },
)

#: **未采纳的同义候选**（禁止随意扩张的留痕；``reason`` 必填）。
SYNONYM_EXCLUSIONS: Tuple[Dict[str, str], ...] = (
    {
        "alias": "great britain",
        "would_merge_into": "united kingdom",
        "reason": (
            "不等价：Great Britain 不含北爱尔兰，United Kingdom 含 —— 是地理/政治**真子集**"
            "关系，不是同义。合并会把不同外延的答案混为一类。"
        ),
    },
    {
        "alias": "england",
        "would_merge_into": "united kingdom",
        "reason": "构成国 vs 主权国家：England 是 UK 的构成国之一，外延不同。",
    },
    {
        "alias": "america",
        "would_merge_into": "united states",
        "reason": (
            "有歧义：America 可指美洲大陆（North/South America），也可口语指美国；"
            "本轮不做「口语指代」推断，保留为独立类。"
        ),
    },
    {
        "alias": "peking",
        "would_merge_into": "beijing",
        "reason": (
            "属**历史名称**（旧称）而非拼写变体；且会引入「同一实体在不同时期/不同资料里"
            "不同名字」的口径问题，超出本轮「同义归并」范围。"
        ),
    },
)

#: **R3 误并反例**（验收硬要求：现场断言这些对**必须保持独立**）。
#: 每项 = ``(答案键 A, 答案键 B, 为什么必须独立)``；额外项显式登记（第 4 条为附加项）。
NO_MERGE_PROBE_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    (
        "1",
        "-1",
        "R3：含数字 ⇒ 只做大小写/空白归一；符号不得删除",
    ),
    (
        "1/2",
        "12",
        "R3：'/' 是运算符 ⇒ 不得删除；数字串不得拼接成 12",
    ),
    (
        "(1,0)",
        "1,0",
        "R3：含数字 ⇒ 不得剥括号、不得删逗号",
    ),
    (
        "1",
        "\u22121",
        "R3 附加项：U+2212 减号与 ASCII '-' 是不同码位，且含数字 ⇒ 不得做码位折叠",
    ),
)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _sha256_bytes(blob: bytes) -> str:
    """字节 SHA256（十六进制小写）。"""
    return hashlib.sha256(blob).hexdigest()


def _canonical_json(obj: Any) -> str:
    """规范化 JSON（``sort_keys`` + 紧凑分隔符；跨运行一致）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    """UTF-8 裸字节 SHA256。"""
    return _sha256_bytes(str(text).encode("utf-8"))


def collapse_ws(text: str) -> str:
    """折叠内部空白为单个空格并去首尾空白（``str.split`` 的确定性口径）。"""
    return " ".join(str(text).split())


def _dedupe(items: Sequence[str]) -> List[str]:
    """保序去重（确定性；不用 ``set`` 迭代）。"""
    out: List[str] = []
    for item in items:
        if str(item) not in out:
            out.append(str(item))
    return out


def has_digit_or_operator(text: str) -> Tuple[bool, Dict[str, str]]:
    """R3 判定：答案是否含**数字**或**运算符**（含则只做大小写与空白归一）。

    返回
    ----
    Tuple[bool, Dict[str, str]]
        ``(restricted, info)``；``info`` 给出命中的数字字符与运算符字符（去重保序），
        供审计表逐条追溯。
    """
    digits = _dedupe([c for c in str(text) if c.isdigit()])
    ops = _dedupe([c for c in str(text) if c in OPERATOR_CHARS])
    return bool(digits or ops), {
        "digits": "".join(digits),
        "operators": "".join(ops),
    }


def negation_signature(text: str) -> Tuple[int, ...]:
    """否定词元签名（R3）：按 :data:`NEGATION_TOKENS` 顺序给出各词元出现次数。

    签名不同即**不允许合并** —— 这把「带否定差异的答案不合并」变成可比较的确定性量，
    而不是人工判断。
    """
    low = str(text).lower()
    return tuple(low.count(tok) for tok in NEGATION_TOKENS)


def strip_one_paren_group(text: str) -> Dict[str, Any]:
    """剥离**一个**配平的包裹式括号组（R1c 的单步实现）。

    参数
    ----
    text : str
        已过 R1a（小写/空白/装饰标点）的文本。

    返回
    ----
    Dict[str, Any]
        ``ok``（是否剥离）、``result``（剥离后的文本，``ok=False`` 时为原文本）、
        ``content``（被剥离的括号内容）、``reason``（未剥离的原因码）。

    何时**不**剥离（fail-closed，全部是「保守」方向）
    -----------------------------------------------
    * ``no_paren``：没有括号；
    * ``unbalanced``：括号不配平；
    * ``stripped_empty``：剥离后为空（如 ``(us)``）；
    * ``content_has_digit_or_operator``：括号内容含数字或运算符；
    * ``body_letters_changed``：括号**外**的字母主体与剥离结果不一致（防御性校验）；
    * ``rest_not_alnum``：剥离后只剩标点/空白。
    """
    s = str(text)
    start = s.find("(")
    if start < 0:
        return {"ok": False, "result": s, "content": "", "reason": "no_paren"}
    depth = 0
    end = -1
    for pos in range(start, len(s)):
        if s[pos] == "(":
            depth += 1
        elif s[pos] == ")":
            depth -= 1
            if depth == 0:
                end = pos
                break
    if end < 0:
        return {"ok": False, "result": s, "content": "", "reason": "unbalanced"}
    content = s[start + 1 : end]
    outside = s[:start] + " " + s[end + 1 :]
    rest = collapse_ws(outside).strip()
    if not rest:
        return {
            "ok": False,
            "result": s,
            "content": content,
            "reason": "stripped_empty",
        }
    restricted_content, _info = has_digit_or_operator(content)
    if restricted_content:
        return {
            "ok": False,
            "result": s,
            "content": content,
            "reason": "content_has_digit_or_operator",
        }
    body_outside = "".join(ch for ch in outside if ch.isalpha())
    body_rest = "".join(ch for ch in rest if ch.isalpha())
    if body_outside != body_rest:
        return {
            "ok": False,
            "result": s,
            "content": content,
            "reason": "body_letters_changed",
        }
    if not any(ch.isalnum() for ch in rest):
        return {
            "ok": False,
            "result": s,
            "content": content,
            "reason": "rest_not_alnum",
        }
    return {"ok": True, "result": rest, "content": content, "reason": ""}


def normalize_answer(
    text: str,
    synonym_rules: Sequence[Mapping[str, Any]] = SYNONYM_RULES,
) -> Dict[str, Any]:
    """单条答案的确定性归一（R1a -> R1c -> R2；命中 R3 时提前返回）。

    参数
    ----
    text : str
        原始答案文本（本模块统一用冻结表的 ``key`` 字段）。
    synonym_rules : Sequence[Mapping[str, Any]]
        R2 同义表（默认 :data:`SYNONYM_RULES`；**不允许**运行期自由扩张）。

    返回
    ----
    Dict[str, Any]
        ``original`` / ``normalized`` / ``rule_ids``（保序去重）/ ``details``（逐条依据）/
        ``restricted``（是否命中 R3）/ ``restriction``（命中的数字与运算符）/
        ``stripped_groups``（被剥离的括号内容清单）。
    """
    original = str(text)
    rules: List[str] = []
    details: List[str] = []
    stripped_groups: List[str] = []
    s = collapse_ws(original.strip().lower())
    if s != original:
        rules.append(RULE_CASE_WS)
        details.append(RULE_TEXTS[RULE_CASE_WS] + "：" + repr(original) + " -> " + repr(s))
    restricted, info = has_digit_or_operator(s)
    if restricted:
        rules.append(RULE_RESTRICTED)
        details.append(
            RULE_TEXTS[RULE_RESTRICTED]
            + "：数字 = "
            + repr(info["digits"])
            + "，运算符 = "
            + repr(info["operators"])
        )
        return {
            "original": original,
            "normalized": s,
            "rule_ids": _dedupe(rules),
            "details": details,
            "restricted": True,
            "restriction": info,
            "stripped_groups": stripped_groups,
        }
    removed = _dedupe([c for c in s if c in DECORATION_CHARS])
    if removed:
        s = "".join(c for c in s if c not in DECORATION_CHARS)
        s = collapse_ws(s)
        rules.append(RULE_DECORATION)
        details.append(RULE_TEXTS[RULE_DECORATION] + "：移除 " + repr("".join(removed)))
    for _ in range(MAX_PAREN_STRIP):
        step = strip_one_paren_group(s)
        if not step["ok"]:
            if str(step["reason"]) not in ("no_paren",):
                details.append(
                    "R1c 未剥离（"
                    + str(step["reason"])
                    + "）：括号内容 = "
                    + repr(step["content"])
                    + "，文本 = "
                    + repr(s)
                )
            break
        stripped_groups.append(str(step["content"]))
        details.append(
            "R1c 剥离括号：内容 = "
            + repr(step["content"])
            + "，"
            + repr(s)
            + " -> "
            + repr(step["result"])
        )
        s = str(step["result"])
        if RULE_PAREN not in rules:
            rules.append(RULE_PAREN)
    for entry in synonym_rules:
        if s in tuple(str(a) for a in entry["aliases"]):
            rules.append(str(entry["id"]))
            details.append(
                "R2 命中 "
                + str(entry["id"])
                + "："
                + repr(s)
                + " -> "
                + repr(entry["canonical"])
                + "；依据 = "
                + str(entry["basis"])
            )
            s = str(entry["canonical"])
            break
    return {
        "original": original,
        "normalized": s,
        "rule_ids": _dedupe(rules),
        "details": details,
        "restricted": False,
        "restriction": info,
        "stripped_groups": stripped_groups,
    }


def _display_similarity(a: str, b: str) -> float:
    """两条 display 的相似度（``difflib.SequenceMatcher`` 比率；确定性、零新依赖）。

    口径：两侧先做「小写 + 折叠空白」，再算 ``SequenceMatcher(None, a, b).ratio()``。
    该比率是**字符串层面的**差异度，只用于 R4 安全阀标记，不参与归并判定本身。
    """
    left = collapse_ws(str(a).strip().lower())
    right = collapse_ws(str(b).strip().lower())
    if not left and not right:
        return 1.0
    return float(difflib.SequenceMatcher(None, left, right).ratio())


# ---------------------------------------------------------------------------
# 归并计划（簇 + 逐类映射）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeCluster:
    """一个 canonical 簇（1 个或多个冻结表类）。

    参数
    ----
    canonical_index : int
        canonical 下标（0-based、稠密、按 canonical key 升序编号）。
    canonical_key : str
        归一化后的类别键（簇内所有成员都归一到这里）。
    canonical_display : str
        代表成员的 display（代表 = 簇内 ``count`` 最大者，平票取最小原始下标）。
    total_count : int
        簇内各成员 ``count`` 之和（= 冻结表口径的**正样本条数**；归并不增加任何样本）。
    members : Tuple[Dict[str, Any], ...]
        成员明细（``index`` / ``key`` / ``display`` / ``count`` / ``normalized`` /
        ``rule_ids`` / ``details``），按原始下标升序。
    rule_ids : Tuple[str, ...]
        该簇用到过的规则编号（成员并集，保序去重）。
    high_risk : bool
        R4 安全阀标记（仍合并，但进人工复核清单）。
    high_risk_reasons : Tuple[str, ...]
        标记原因（``cluster_size_gt_4`` / ``display_divergence``）。
    display_similarity_min : float
        簇内两两 display 相似度的**下界**（单成员簇恒为 1.0）。
    """

    canonical_index: int
    canonical_key: str
    canonical_display: str
    total_count: int
    members: Tuple[Dict[str, Any], ...]
    rule_ids: Tuple[str, ...]
    high_risk: bool
    high_risk_reasons: Tuple[str, ...]
    display_similarity_min: float

    @property
    def size(self) -> int:
        """簇内成员数。"""
        return int(len(self.members))

    @property
    def member_indices(self) -> Tuple[int, ...]:
        """簇内成员的原始下标（升序）。"""
        return tuple(int(m["index"]) for m in self.members)


@dataclass(frozen=True)
class MergePlan:
    """整份归并计划（**唯一**归并决策来源）。

    参数
    ----
    version : str
        模块版本。
    n_original : int
        冻结表原始类数（现场 298）。
    n_canonical : int
        归并后 canonical 类数（**不再是 298**）。
    clusters : Tuple[MergeCluster, ...]
        按 ``canonical_index`` 升序的簇。
    index_map : Tuple[int, ...]
        原始下标 -> canonical 下标（长度 = ``n_original``）。
    blocked_merges : Tuple[Dict[str, Any], ...]
        被 R3 否定差异**阻断**的合并（归一化相同但否定签名不同）；空元组表示未发生。
    synonym_applications : Tuple[Dict[str, Any], ...]
        R2 的**实际命中**记录（本语料现场可能为 0 条，如实登记）。
    rule_counts : Dict[str, int]
        规则 -> 受影响的原始类数（现场统计）。
    """

    version: str
    n_original: int
    n_canonical: int
    clusters: Tuple[MergeCluster, ...]
    index_map: Tuple[int, ...]
    blocked_merges: Tuple[Dict[str, Any], ...]
    synonym_applications: Tuple[Dict[str, Any], ...]
    rule_counts: Dict[str, int]

    def as_mapping(self) -> Dict[int, int]:
        """原始下标 -> canonical 下标（可直接传给 ``build_scale_corpus(merge_map=...)``）。"""
        return {int(i): int(c) for i, c in enumerate(self.index_map)}

    def canonical_labels(self) -> Dict[int, Tuple[str, str]]:
        """canonical 下标 -> ``(canonical_key, canonical_display)``（供语料重建分类标签）。"""
        return {
            int(c.canonical_index): (str(c.canonical_key), str(c.canonical_display))
            for c in self.clusters
        }

    def fingerprint(self) -> str:
        """计划指纹（只含映射与规则版本；**不含**任何时间/路径）。"""
        payload = {
            "version": str(self.version),
            "n_original": int(self.n_original),
            "n_canonical": int(self.n_canonical),
            "index_map": [int(x) for x in self.index_map],
            "canonical_keys": [str(c.canonical_key) for c in self.clusters],
        }
        return sha256_text(_canonical_json(payload))

    def high_risk_clusters(self) -> List[MergeCluster]:
        """high-risk 簇清单（按 canonical 下标升序）。"""
        return [c for c in self.clusters if bool(c.high_risk)]

    def multi_member_clusters(self) -> List[MergeCluster]:
        """发生实际归并（成员数 >= 2）的簇（按 canonical 下标升序）。"""
        return [c for c in self.clusters if c.size >= 2]

    def summary(self) -> Dict[str, Any]:
        """计划摘要（规模 / 归并量 / 高危清单 / 规则计数）。"""
        multi = self.multi_member_clusters()
        return {
            "version": str(self.version),
            "n_original": int(self.n_original),
            "n_canonical": int(self.n_canonical),
            "n_reductions": int(self.n_original - self.n_canonical),
            "reduction_ratio": (
                float(self.n_original - self.n_canonical) / float(self.n_original)
                if self.n_original
                else 0.0
            ),
            "n_multi_member_clusters": int(len(multi)),
            "n_classes_in_multi_member_clusters": int(sum(c.size for c in multi)),
            "multi_member_clusters": [
                {
                    "canonical_index": int(c.canonical_index),
                    "canonical_key": str(c.canonical_key),
                    "cluster_size": int(c.size),
                    "member_indices": [int(x) for x in c.member_indices],
                    "total_count": int(c.total_count),
                    "high_risk": bool(c.high_risk),
                }
                for c in multi
            ],
            "n_high_risk_clusters": int(len(self.high_risk_clusters())),
            "high_risk_clusters": [
                {
                    "canonical_index": int(c.canonical_index),
                    "canonical_key": str(c.canonical_key),
                    "cluster_size": int(c.size),
                    "member_indices": [int(x) for x in c.member_indices],
                    "reasons": [str(x) for x in c.high_risk_reasons],
                    "display_similarity_min": float(c.display_similarity_min),
                }
                for c in self.high_risk_clusters()
            ],
            "n_blocked_merges": int(len(self.blocked_merges)),
            "n_synonym_applications": int(len(self.synonym_applications)),
            "rule_counts": {str(k): int(v) for k, v in sorted(self.rule_counts.items())},
            "fingerprint": self.fingerprint(),
        }


def build_merge_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    synonym_rules: Sequence[Mapping[str, Any]] = SYNONYM_RULES,
) -> MergePlan:
    """由统一答案表构造**确定性**归并计划。

    参数
    ----
    rows : Sequence[Mapping[str, Any]]
        统一答案表条目（字段现场枚举 = ``['count', 'display', 'index', 'key',
        'raw_variants', 'sources']``）；``index`` 必须恰为 ``0..N-1``。
    synonym_rules : Sequence[Mapping[str, Any]]
        R2 同义表。

    返回
    ----
    MergePlan

    关键不变量（构造期，违反即抛）
    -----------------------------
    1. ``index`` 必须是 ``0..N-1`` 的连续序列，且 ``key`` **互不相同**；
    2. 每个原始类**恰好**落在一个簇里（簇成员并集 = 全集、两两不交）；
    3. 簇 ``total_count`` 之和 = 原表 ``count`` 之和（归并不增样本）；
    4. canonical 下标稠密且 ``index_map`` 的取值集合 = ``0..C'-1``；
    5. canonical key **互不相同**（否则标签不可区分，必须 fail-closed）。

    异常
    ------
    ValueError
        上述任一不变量被破坏。
    """
    per_class: List[Dict[str, Any]] = []
    seen_keys: Dict[str, int] = {}
    for pos, row in enumerate(rows):
        index = int(row["index"])
        if index != int(pos):
            raise ValueError(
                "统一答案表的 index 必须是 0..N-1 的连续序列；现场第 "
                + str(pos)
                + " 行的 index = "
                + str(index)
            )
        key = str(row["key"])
        if key in seen_keys:
            raise ValueError(
                "统一答案表的 key 重复（"
                + repr(key)
                + "，index = "
                + str(seen_keys[key])
                + " / "
                + str(index)
                + "）；重复键会让归并映射不可逆，拒绝继续"
            )
        seen_keys[key] = int(index)
        norm = normalize_answer(key, synonym_rules)
        per_class.append(
            {
                "index": int(index),
                "key": key,
                "display": str(row["display"]),
                "count": int(row["count"]),
                "normalized": str(norm["normalized"]),
                "rule_ids": [str(x) for x in norm["rule_ids"]],
                "details": [str(x) for x in norm["details"]],
                "restricted": bool(norm["restricted"]),
                "negation_signature": [int(x) for x in negation_signature(key)],
            }
        )

    # ---- 按归一化结果分组，再按否定签名切分（R3：否定差异不合并） ------------------
    by_normalized: Dict[str, List[Dict[str, Any]]] = {}
    for rec in per_class:
        by_normalized.setdefault(str(rec["normalized"]), []).append(rec)
    groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    blocked_merges: List[Dict[str, Any]] = []
    for normalized in sorted(by_normalized.keys()):
        members = by_normalized[normalized]
        by_signature: Dict[Tuple[int, ...], List[Dict[str, Any]]] = {}
        for rec in members:
            sig = tuple(int(x) for x in rec["negation_signature"])
            by_signature.setdefault(sig, []).append(rec)
        signatures = sorted(by_signature.keys())
        if len(signatures) > 1:
            # 归一化把否定差异抹平了 —— 按签名拆簇，并逐对登记「被阻断的合并」。
            for i in range(len(signatures)):
                for j in range(i + 1, len(signatures)):
                    left = sorted(by_signature[signatures[i]], key=lambda r: int(r["index"]))
                    right = sorted(by_signature[signatures[j]], key=lambda r: int(r["index"]))
                    blocked_merges.append(
                        {
                            "normalized": str(normalized),
                            "negation_signature_a": [int(x) for x in signatures[i]],
                            "negation_signature_b": [int(x) for x in signatures[j]],
                            "indices_a": [int(r["index"]) for r in left],
                            "indices_b": [int(r["index"]) for r in right],
                            "keys_a": [str(r["key"]) for r in left],
                            "keys_b": [str(r["key"]) for r in right],
                            "rule": RULE_NEGATION,
                            "reason": RULE_TEXTS[RULE_NEGATION],
                        }
                    )
        # 每个签名一个候选簇；**最大**的签名组保留 canonical key，其余用最小成员的原始 key
        ordered = sorted(
            (by_signature[s] for s in signatures),
            key=lambda g: (-len(g), min(int(r["index"]) for r in g)),
        )
        for rank, group in enumerate(ordered):
            group = sorted(group, key=lambda r: int(r["index"]))
            if rank == 0:
                groups.append((str(normalized), group))
            else:
                groups.append((str(group[0]["key"]), group))

    # ---- canonical 编号（按 canonical key 升序；同 key 时按最小成员下标） -----------
    groups.sort(key=lambda item: (str(item[0]), min(int(r["index"]) for r in item[1])))
    canonical_keys_seen: Dict[str, int] = {}
    clusters: List[MergeCluster] = []
    for canonical_index, (canonical_key, members) in enumerate(groups):
        if canonical_key in canonical_keys_seen:
            raise ValueError(
                "canonical key 重复（"
                + repr(canonical_key)
                + "，簇 "
                + str(canonical_keys_seen[canonical_key])
                + " 与 "
                + str(canonical_index)
                + "）；归并后标签必须两两可区分，拒绝继续"
            )
        canonical_keys_seen[str(canonical_key)] = int(canonical_index)
        members = sorted(members, key=lambda r: int(r["index"]))
        # 代表成员 = count 最大者，平票取最小原始下标（确定性）
        representative = sorted(
            members, key=lambda r: (-int(r["count"]), int(r["index"]))
        )[0]
        rule_ids: List[str] = []
        for rec in members:
            for rule in rec["rule_ids"]:
                if str(rule) not in rule_ids:
                    rule_ids.append(str(rule))
        similarity = 1.0
        if len(members) > 1:
            values: List[float] = []
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    values.append(
                        _display_similarity(
                            str(members[i]["display"]), str(members[j]["display"])
                        )
                    )
            similarity = float(min(values)) if values else 1.0
        reasons: List[str] = []
        if len(members) > int(HIGH_RISK_MAX_CLUSTER_SIZE):
            reasons.append("cluster_size_gt_4")
        if len(members) > 1 and similarity < float(HIGH_RISK_MIN_DISPLAY_SIMILARITY):
            reasons.append("display_divergence")
        if reasons and RULE_HIGH_RISK not in rule_ids:
            rule_ids.append(RULE_HIGH_RISK)
        clusters.append(
            MergeCluster(
                canonical_index=int(canonical_index),
                canonical_key=str(canonical_key),
                canonical_display=str(representative["display"]),
                total_count=int(sum(int(r["count"]) for r in members)),
                members=tuple(
                    {
                        "index": int(r["index"]),
                        "key": str(r["key"]),
                        "display": str(r["display"]),
                        "count": int(r["count"]),
                        "normalized": str(r["normalized"]),
                        "rule_ids": [str(x) for x in r["rule_ids"]],
                        "details": [str(x) for x in r["details"]],
                    }
                    for r in members
                ),
                rule_ids=tuple(_dedupe(rule_ids)),
                high_risk=bool(reasons),
                high_risk_reasons=tuple(str(x) for x in reasons),
                display_similarity_min=float(similarity),
            )
        )
    n_original = int(len(per_class))
    n_canonical = int(len(clusters))
    index_map = [0] * n_original
    covered: List[int] = []
    for cluster in clusters:
        for rec in cluster.members:
            index_map[int(rec["index"])] = int(cluster.canonical_index)
            covered.append(int(rec["index"]))
    if sorted(covered) != list(range(n_original)):
        raise ValueError(
            "归并不变量被破坏：簇成员并集 != 全部原始下标（覆盖 "
            + str(len(covered))
            + " / "
            + str(n_original)
            + "）"
        )
    if sorted(set(index_map)) != list(range(n_canonical)):
        raise ValueError("归并不变量被破坏：canonical 下标不稠密")
    total_original = int(sum(int(r["count"]) for r in per_class))
    total_canonical = int(sum(int(c.total_count) for c in clusters))
    if total_original != total_canonical:
        raise ValueError(
            "归并不变量被破坏：簇 count 之和 "
            + str(total_canonical)
            + " != 原表 count 之和 "
            + str(total_original)
            + "（归并**不得**增删样本）"
        )
    rule_counts: Dict[str, int] = {}
    for rec in per_class:
        for rule in rec["rule_ids"]:
            rule_counts[str(rule)] = rule_counts.get(str(rule), 0) + 1
    synonym_applications = [
        {
            "index": int(rec["index"]),
            "key": str(rec["key"]),
            "rule_id": str(rule),
            "detail": next((d for d in rec["details"] if str(rule) in str(d)), ""),
        }
        for rec in per_class
        for rule in rec["rule_ids"]
        if str(rule).startswith("R2-")
    ]
    return MergePlan(
        version=ANSWER_MERGE_VERSION,
        n_original=n_original,
        n_canonical=n_canonical,
        clusters=tuple(clusters),
        index_map=tuple(int(x) for x in index_map),
        blocked_merges=tuple(blocked_merges),
        synonym_applications=tuple(synonym_applications),
        rule_counts={str(k): int(v) for k, v in sorted(rule_counts.items())},
    )


def validate_merge_map(
    merge_map: Mapping[int, int],
    *,
    n_original: int,
    merge_labels: Optional[Mapping[int, Sequence[str]]] = None,
) -> Dict[int, int]:
    """校验并规范化**归并映射**（``qa_scale.build_scale_corpus`` 的唯一入口）。

    参数
    ----
    merge_map : Mapping[int, int]
        原始下标 -> canonical 下标。
    n_original : int
        原始类数（必须与 ``merge_map`` 的定义域一致）。
    merge_labels : Optional[Mapping[int, Sequence[str]]]
        canonical 下标 -> ``(key, display)``；给定时校验其定义域与值域一致。

    返回
    ----
    Dict[int, int]
        规范化后的映射（键与值都强制为 ``int``）。

    异常
    ------
    ValueError
        定义域 != ``0..n_original-1``、值域不稠密、值非整数 / 为负、或
        ``merge_labels`` 与值域不一致。
    """
    out: Dict[int, int] = {}
    for key, value in dict(merge_map).items():
        try:
            k = int(key)
            v = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                "归并映射必须全是整数（现场键 = " + repr(key) + "，值 = " + repr(value) + "）"
            )
        if v < 0:
            raise ValueError("归并映射的值不得为负（现场 " + repr(k) + " -> " + repr(v) + "）")
        out[k] = v
    if sorted(out.keys()) != list(range(int(n_original))):
        raise ValueError(
            "归并映射的定义域必须是 0.."
            + str(int(n_original) - 1)
            + " 的全集（现场键数 = "
            + str(len(out))
            + "）"
        )
    values = sorted(set(out.values()))
    if values != list(range(len(values))):
        raise ValueError(
            "归并映射的值域必须是 0..C'-1 的稠密编号（现场值域 = "
            + str(values[:10])
            + "…共 "
            + str(len(values))
            + " 个）"
        )
    if merge_labels is not None:
        label_map = dict(merge_labels)
        label_keys = sorted(int(k) for k in label_map.keys())
        if label_keys != values:
            raise ValueError(
                "merge_labels 的定义域与归并映射的值域不一致（labels = "
                + str(label_keys[:10])
                + "…，values = "
                + str(values[:10])
                + "…）"
            )
        for key in label_keys:
            label = tuple(label_map[key])
            if len(label) != 2:
                raise ValueError(
                    "merge_labels["
                    + str(key)
                    + "] 必须是 (key, display) 二元组，现场 = "
                    + repr(label)
                )
    return {int(k): int(v) for k, v in sorted(out.items())}


def assert_no_merge_pairs(
    plan: MergePlan,
    pairs: Optional[Sequence[Sequence[str]]] = None,
) -> Dict[str, Any]:
    """**现场断言**：R3 误并反例必须保持独立（验收硬要求）。

    参数
    ----
    plan : MergePlan
        归并计划。
    pairs : Optional[Sequence[Sequence[str]]]
        ``(key_a, key_b, reason)`` 列表；默认 :data:`NO_MERGE_PROBE_PAIRS`。

    返回
    ----
    Dict[str, Any]
        ``probes``（逐对的原始下标 / canonical 下标 / 是否独立）、``all_distinct``、
        ``n_probes``、``allowed_extra``（允许的额外项 = 第 4 条之后的附加反例）。

    异常
    ------
    AssertionError
        ① 某个 key 在冻结表里查不到（口径前提被破坏）；② 任一对被归并到同一 canonical。
    """
    probes = [tuple(str(x) for x in p) for p in (pairs or NO_MERGE_PROBE_PAIRS)]
    key_to_index: Dict[str, int] = {}
    for cluster in plan.clusters:
        for rec in cluster.members:
            key_to_index[str(rec["key"])] = int(rec["index"])
    rows: List[Dict[str, Any]] = []
    for key_a, key_b, reason in probes:
        if key_a not in key_to_index:
            raise AssertionError(
                "误并反例的 key 在冻结表里查不到：" + repr(key_a) + "（口径前提被破坏）"
            )
        if key_b not in key_to_index:
            raise AssertionError(
                "误并反例的 key 在冻结表里查不到：" + repr(key_b) + "（口径前提被破坏）"
            )
        ia = int(key_to_index[key_a])
        ib = int(key_to_index[key_b])
        ca = int(plan.index_map[ia])
        cb = int(plan.index_map[ib])
        if ca == cb:
            raise AssertionError(
                "误并反例被违反："
                + repr(key_a)
                + "（index "
                + str(ia)
                + "）与 "
                + repr(key_b)
                + "（index "
                + str(ib)
                + "）被归并到同一个 canonical "
                + str(ca)
                + "；原因 = "
                + reason
            )
        rows.append(
            {
                "key_a": key_a,
                "index_a": ia,
                "canonical_a": ca,
                "key_b": key_b,
                "index_b": ib,
                "canonical_b": cb,
                "distinct": True,
                "reason": reason,
            }
        )
    return {
        "probes": rows,
        "n_probes": int(len(rows)),
        "all_distinct": True,
        "required_pairs": 3,
        "allowed_extra": (
            "允许的额外项 = 第 4 条起的附加反例（现场 1 条：U+2212 减号 vs ASCII '-'）；"
            "前 3 条是验收**硬要求**，缺一即失败。"
        ),
        "rule": (
            "R3 禁并项的现场证明：逐对查冻结表得到原始下标，再查 index_map 得到 canonical；"
            "任一 `ca == cb` 即抛 AssertionError（**非恒真**：把 R3 关掉后 1/-1 会被并在一起）。"
        ),
    }


# ---------------------------------------------------------------------------
# 产物记录（merge_map.jsonl 的两种记录类型）
# ---------------------------------------------------------------------------


def class_records(plan: MergePlan) -> List[Dict[str, Any]]:
    """``record_type='class'`` 记录：**逐类映射**（原 index/key -> canonical）。"""
    out: List[Dict[str, Any]] = []
    for cluster in plan.clusters:
        for rec in cluster.members:
            out.append(
                {
                    "record_type": "class",
                    "answer_index": int(rec["index"]),
                    "answer_key": str(rec["key"]),
                    "answer_display": str(rec["display"]),
                    "answer_count": int(rec["count"]),
                    "canonical_index": int(cluster.canonical_index),
                    "canonical_key": str(cluster.canonical_key),
                    "cluster_size": int(cluster.size),
                    "rule_ids": [str(x) for x in cluster.rule_ids],
                    "high_risk": bool(cluster.high_risk),
                }
            )
    out.sort(key=lambda r: int(r["answer_index"]))
    return out


def cluster_records(plan: MergePlan) -> List[Dict[str, Any]]:
    """``record_type='cluster'`` 记录：**逐簇成员与依据**（规则编号 + 逐条 detail）。"""
    out: List[Dict[str, Any]] = []
    for cluster in plan.clusters:
        out.append(
            {
                "record_type": "cluster",
                "cluster_id": int(cluster.canonical_index),
                "canonical_index": int(cluster.canonical_index),
                "canonical_key": str(cluster.canonical_key),
                "canonical_display": str(cluster.canonical_display),
                "total_count": int(cluster.total_count),
                "cluster_size": int(cluster.size),
                "rule_ids": [str(x) for x in cluster.rule_ids],
                "high_risk": bool(cluster.high_risk),
                "high_risk_reasons": [str(x) for x in cluster.high_risk_reasons],
                "display_similarity_min": float(cluster.display_similarity_min),
                "members": [
                    {
                        "index": int(m["index"]),
                        "key": str(m["key"]),
                        "display": str(m["display"]),
                        "count": int(m["count"]),
                        "normalized": str(m["normalized"]),
                        "rule_ids": [str(x) for x in m["rule_ids"]],
                        "details": [str(x) for x in m["details"]],
                    }
                    for m in cluster.members
                ],
            }
        )
    out.sort(key=lambda r: int(r["canonical_index"]))
    return out


def rules_documentation() -> Dict[str, Any]:
    """规则文档（**唯一来源**：进审计表与报告，避免文档与实现漂移）。"""
    return {
        "version": str(ANSWER_MERGE_VERSION),
        "rule_texts": {str(k): str(v) for k, v in sorted(RULE_TEXTS.items())},
        "decoration_chars": str(DECORATION_CHARS),
        "operator_chars": str(OPERATOR_CHARS),
        "negation_tokens": [str(x) for x in NEGATION_TOKENS],
        "max_paren_strip": int(MAX_PAREN_STRIP),
        "high_risk_max_cluster_size": int(HIGH_RISK_MAX_CLUSTER_SIZE),
        "high_risk_min_display_similarity": float(HIGH_RISK_MIN_DISPLAY_SIMILARITY),
        "synonym_rules": [
            {
                "id": str(e["id"]),
                "aliases": [str(a) for a in e["aliases"]],
                "canonical": str(e["canonical"]),
                "basis": str(e["basis"]),
            }
            for e in SYNONYM_RULES
        ],
        "synonym_exclusions": [
            {
                "alias": str(e["alias"]),
                "would_merge_into": str(e["would_merge_into"]),
                "reason": str(e["reason"]),
            }
            for e in SYNONYM_EXCLUSIONS
        ],
        "no_merge_probe_pairs": [
            {"key_a": str(a), "key_b": str(b), "reason": str(c)}
            for a, b, c in NO_MERGE_PROBE_PAIRS
        ],
    }


def _relaxed_normalization(text: str) -> str:
    """**诊断用**的放宽归一（**不参与**归并决策）：在 R1 之上再加「变音符折叠 + 去空格」。

    只用于审计表的「保守口径漏并清单」：把「若采用更宽松的口径会多并掉哪些类」算成
    可见的数字，而不是让读者拿 281 去猜为什么不是任务书举例的 277。**绝不**写回映射。
    """
    s = collapse_ws(str(text).strip().lower())
    s = "".join(c for c in s if c not in DECORATION_CHARS)
    for _ in range(MAX_PAREN_STRIP):
        step = strip_one_paren_group(s)
        if not step["ok"]:
            break
        s = str(step["result"])
    s = "".join(unicodedata.normalize("NFKD", s).split())
    return "".join(c for c in s if not unicodedata.combining(c))


def diagnostic_relaxed_merges(
    pairs: Sequence[Sequence[Any]],
    *,
    max_groups: int = 50,
) -> Dict[str, Any]:
    """**诊断量**（不参与归并）：若再叠加「变音符折叠 + 去空格」，会额外并掉哪些簇。

    参数
    ----
    pairs : Sequence[Sequence[Any]]
        ``(答案键, 当前 canonical 下标)`` 序列（可由 ``merge_map.jsonl`` 的
        ``record_type='class'`` 记录直接取得）。
    max_groups : int
        最多列出多少组（超出部分只计数，不静默丢弃）。

    返回
    ----
    Dict[str, Any]
        ``n_extra_merges``（额外并掉的类数）、``predicted_n_canonical``、
        ``groups``（逐组的放宽键 / 当前 canonical 集合 / 成员键 / 是否**违反 R3**）、
        ``n_groups_violating_r3``、``n_extra_merges_excluding_r3_violations``、``rule``。

    **为什么必须给出「是否违反 R3」**：放宽口径会把 ``1,0`` 与 ``10`` 并成一组
    （去逗号 + 去空格后都是 ``10``）—— 这正是 R3 禁止的误并。该标记让读者一眼看出
    「更接近 277 的那个数字是靠一条**不可采纳**的规则换来的」。
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for item in pairs:
        key = str(item[0])
        canonical = int(item[1])
        buckets.setdefault(_relaxed_normalization(key), []).append(
            {"key": key, "canonical_index": canonical}
        )
    groups: List[Dict[str, Any]] = []
    n_extra = 0
    n_extra_safe = 0
    n_violating = 0
    for relaxed in sorted(buckets.keys()):
        members = sorted(buckets[relaxed], key=lambda m: (int(m["canonical_index"]), str(m["key"])))
        canonicals = sorted({int(m["canonical_index"]) for m in members})
        if len(canonicals) < 2:
            continue
        extra = int(len(canonicals) - 1)
        violates = any(has_digit_or_operator(str(m["key"]))[0] for m in members)
        n_extra += extra
        if violates:
            n_violating += 1
        else:
            n_extra_safe += extra
        if len(groups) < int(max_groups):
            groups.append(
                {
                    "relaxed_key": str(relaxed),
                    "canonical_indices": [int(x) for x in canonicals],
                    "extra_merges": int(extra),
                    "member_keys": [str(m["key"]) for m in members],
                    "member_indices_sorted_by_canonical": [
                        [int(m["canonical_index"]), str(m["key"])] for m in members
                    ],
                    "violates_r3": bool(violates),
                }
            )
    return {
        "n_extra_merges": int(n_extra),
        "predicted_n_canonical": int(len(set(int(p[1]) for p in pairs)) - n_extra),
        "n_groups": int(
            sum(
                1
                for relaxed in buckets
                if len({int(m["canonical_index"]) for m in buckets[relaxed]}) >= 2
            )
        ),
        "n_groups_listed": int(len(groups)),
        "max_groups": int(max_groups),
        "groups": groups,
        "n_groups_violating_r3": int(n_violating),
        "n_extra_merges_excluding_r3_violations": int(n_extra_safe),
        "predicted_n_canonical_excluding_r3_violations": int(
            len(set(int(p[1]) for p in pairs)) - n_extra_safe
        ),
        "rule": (
            "**诊断量，不进归并映射**：放宽归一 = R1 + NFKD 去变音符 + 去全部空白；"
            "仅用于解释「保守口径为什么没有并到更少的类数」。`violates_r3 = True` 的组"
            "表示该放宽会误并含数字/运算符的类（如 '1,0' 与 '10'）⇒ 该组**不可采纳**。"
        ),
    }


def load_table_rows(product_dir: str = "") -> List[Dict[str, Any]]:
    """读取统一答案表（**只读**；复用 :mod:`n3d_qa_learn.qa_scale` 的加载器与不变量）。

    参数
    ----
    product_dir : str
        冻结产物目录；空串 = ``qa_scale.DEFAULT_PRODUCT_DIR``。

    返回
    ----
    List[Dict[str, Any]]
        按 ``index`` 升序的条目（字段与冻结表逐字相同）。
    """
    import n3d_qa_learn.qa_scale as QS

    directory = str(product_dir) if product_dir else str(QS.DEFAULT_PRODUCT_DIR)
    return list(QS.load_answer_table(directory))


__all__ = [
    "ANSWER_MERGE_VERSION",
    "PRODUCT_DIR",
    "ANSWER_TABLE_FILE",
    "DECORATION_CHARS",
    "OPERATOR_CHARS",
    "NEGATION_TOKENS",
    "MAX_PAREN_STRIP",
    "HIGH_RISK_MAX_CLUSTER_SIZE",
    "HIGH_RISK_MIN_DISPLAY_SIMILARITY",
    "RULE_CASE_WS",
    "RULE_DECORATION",
    "RULE_PAREN",
    "RULE_SYNONYM",
    "RULE_RESTRICTED",
    "RULE_NEGATION",
    "RULE_HIGH_RISK",
    "RULE_TEXTS",
    "SYNONYM_RULES",
    "SYNONYM_EXCLUSIONS",
    "NO_MERGE_PROBE_PAIRS",
    "MergeCluster",
    "MergePlan",
    "build_merge_plan",
    "validate_merge_map",
    "assert_no_merge_pairs",
    "class_records",
    "cluster_records",
    "rules_documentation",
    "diagnostic_relaxed_merges",
    "load_table_rows",
    "collapse_ws",
    "has_digit_or_operator",
    "negation_signature",
    "normalize_answer",
    "strip_one_paren_group",
    "sha256_text",
]
