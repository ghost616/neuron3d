"""业务路由（固定顺序，不可配置）。

业务逻辑（固定）
----------------
用户输入一个问题：

1. **步骤 1（QA 数据集匹配）**：用 N3D 模型对问题打分，取 ``top-1``。
   * 若 top-1 是**具体答案**（非「不相关」类）-> 返回该答案，来源标记 ``qa``；
   * 若 top-1 是「不相关」类 -> **命中判定为未命中**，进入步骤 2。
2. **步骤 2（文本数据集匹配）**：在文本行语料里找最相似的一行。
   * 相似度 >= 阈值 -> 返回该行文本，来源标记 ``text``；
   * 否则 -> 返回「无匹配」，来源标记 ``none``。

**命中判定 = 输出「不相关」类即视为未命中**（不是"分数低才算未命中"）。
路由返回**必须携带来源标记与分数**。

边界处置（显式定义）
------------------
======================  ==========================================================
输入                     行为
======================  ==========================================================
``""``（空问题）         返回「无匹配」，来源 ``none``，``reason="empty_question"``
仅空白                   同上（归一化后为空，``reason="empty_question"``）
超长（> 上限）           **立即报错**（``ValueError``，不静默截断）
候选集合为空             步骤 1 无候选 -> 直接跳过；步骤 2 文本语料为空 -> 「无匹配」
文本行超长               载入时按 ``MAX_LINE_CHARS`` 截断（``TextRecord.truncated`` 可见）
======================  ==========================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .data import MAX_QUESTION_CHARS, TextRecord, check_question
from .features import TextVectorizer, cosine, normalize_text

#: 来源标记（**唯一**合法取值集合）。
ROUTE_SOURCES: Tuple[str, ...] = ("qa", "text", "none")

#: 未命中/无匹配的固定展示文本。
NO_MATCH_TEXT: str = "无匹配"

#: 步骤 2 的默认余弦阈值（低于该值视为"文本数据集也未命中"）。
DEFAULT_TEXT_THRESHOLD: float = 0.28

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _word_set(text: str) -> set:
    """归一化文本的词集合（用于步骤 1 的精确/词级匹配参考分数）。"""
    return set(_WORD_RE.findall(normalize_text(text)))


@dataclass(frozen=True)
class RouteResult:
    """路由结果（**必须**携带来源标记与分数）。

    属性
    ----
    answer : str
        最终展示文本：具体答案 / 匹配到的文本行 / ``"无匹配"``。
    source : str
        ``"qa"``（步骤 1 命中）/ ``"text"``（步骤 2 命中）/ ``"none"``（两级均未命中）。
    score : float
        决定该结果的分数：``qa`` = 步骤 1 的 ``top-1`` logit；``text`` = 步骤 2 的余弦；
        ``none`` = 两级中较大者（便于观测"差多少没命中"）。
    step1_index : int
        步骤 1 的 ``top-1`` 候选下标（``-1`` = 无候选）。
    step1_is_irrelevant : bool
        步骤 1 的 ``top-1`` 是否为「不相关」类。
    step2_line_id : str
        步骤 2 命中的行 id（未命中为空串）。
    reason : str
        附加原因（``"empty_question"`` / ``"no_candidate"`` / ``"below_threshold"`` / ``""``）。
    """

    answer: str
    source: str
    score: float
    step1_index: int = -1
    step1_is_irrelevant: bool = False
    step2_line_id: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.source not in ROUTE_SOURCES:
            raise ValueError(
                f"RouteResult.source 仅允许 {list(ROUTE_SOURCES)}，当前 {self.source!r}"
            )

    def as_dict(self) -> Dict[str, object]:
        """JSON 化（CLI 输出与报告共用）。"""
        return {
            "answer": self.answer,
            "source": self.source,
            "score": float(self.score),
            "step1_index": int(self.step1_index),
            "step1_is_irrelevant": bool(self.step1_is_irrelevant),
            "step2_line_id": self.step2_line_id,
            "reason": self.reason,
        }


class QuestionRouter:
    """两级业务路由（步骤 1 走模型 logits，步骤 2 走词袋余弦）。

    参数
    ----
    answer_keys : Sequence[str]
        全局答案表（下标 = 模型候选下标；**不含**「不相关」）。
    answer_display : Dict[str, str]
        类别键 -> 展示文本。
    text_lines : Sequence[TextRecord]
        文本行语料（步骤 2 的匹配源；为空时步骤 2 恒为「无匹配」）。
    vectorizer : TextVectorizer
        与模型训练同源的向量化器（**必须**来自产物 meta，保证口径一致）。
    text_threshold : float
        步骤 2 的余弦阈值。

    关键不变量
    ----------
    * 步骤顺序固定：步骤 1 命中即返回，**不再查文本**；
    * 「不相关」是**未命中**判据，不是"低分"判据；
    * 文本行向量在构造期一次性预计算（避免每次查询重复编码）。
    """

    def __init__(
        self,
        answer_keys: Sequence[str],
        answer_display: Dict[str, str],
        text_lines: Sequence[TextRecord],
        vectorizer: TextVectorizer,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    ) -> None:
        self.answer_keys: List[str] = list(answer_keys)
        self.answer_display: Dict[str, str] = dict(answer_display)
        self.text_lines: List[TextRecord] = list(text_lines)
        self.vectorizer = vectorizer
        self.text_threshold = float(text_threshold)
        self.irrelevant_index: int = len(self.answer_keys)
        # 文本行向量预计算（构造期一次性；前向/查询中不重算）
        self._line_vectors: List[List[float]] = [
            vectorizer.encode(rec.text) for rec in self.text_lines
        ]

    # -- 步骤 1 -----------------------------------------------------------
    def step1_index_from_logits(self, logits: Sequence[float]) -> Tuple[int, float, bool]:
        """从 ``[C+1]`` logits 取 ``top-1``，并判定是否为「不相关」类。

        参数
        ----
        logits : Sequence[float]
            候选 logits（长度须为 ``C+1``；为空表示**候选集合为空**）。

        返回
        ----
        Tuple[int, float, bool]
            ``(top1_index, top1_score, is_irrelevant)``；候选为空时返回 ``(-1, 0.0, False)``。
        """
        n = len(logits)
        if n == 0:
            return -1, 0.0, False
        expected = len(self.answer_keys) + 1
        if n != expected:
            raise ValueError(
                f"logits 长度 {n} 与候选空间宽度 {expected}（C+1，C={len(self.answer_keys)}）不一致"
            )
        best = 0
        best_score = float(logits[0])
        for i in range(1, n):
            value = float(logits[i])
            if value > best_score:
                best = i
                best_score = value
        return best, best_score, best == self.irrelevant_index

    def answer_text(self, index: int) -> str:
        """把答案表下标翻译为展示文本（无候选/越界时抛错）。"""
        if index < 0 or index >= len(self.answer_keys):
            raise IndexError(
                f"答案表下标 {index} 越界（合法范围 0..{len(self.answer_keys) - 1}）"
            )
        key = self.answer_keys[index]
        return self.answer_display.get(key, key)

    # -- 步骤 2 -----------------------------------------------------------
    def step2_best_line(self, question: str) -> Tuple[Optional[TextRecord], float]:
        """在文本行语料里取余弦相似度最高的一行。

        参数
        ----
        question : str
            问题文本。

        返回
        ----
        Tuple[Optional[TextRecord], float]
            最佳行与其余弦分数；语料为空时返回 ``(None, 0.0)``。
        """
        if not self.text_lines:
            return None, 0.0
        qv = self.vectorizer.encode(question)
        best_i = -1
        best_score = -1.0
        for i, lv in enumerate(self._line_vectors):
            s = cosine(qv, lv)
            if s > best_score:
                best_score = s
                best_i = i
        return self.text_lines[best_i], float(best_score)

    # -- 总路由 -----------------------------------------------------------
    def route(self, question: str, logits: Optional[Sequence[float]] = None) -> RouteResult:
        """执行固定顺序的两级路由。

        参数
        ----
        question : str
            用户问题（**原样**传入；本方法内部做边界处置）。
        logits : Optional[Sequence[float]]
            步骤 1 的 ``[C+1]`` logits；``None`` 或空序列表示**候选集合为空**。

        返回
        ----
        RouteResult
            见 ``RouteResult``。

        异常
        ------
        TypeError
            ``question`` 非字符串。
        ValueError
            问题超长（``len > MAX_QUESTION_CHARS``）。
        """
        check_question(question)
        if len(question) > MAX_QUESTION_CHARS:  # 冗余防御（check_question 已覆盖）
            raise ValueError(f"问题长度 {len(question)} 超过上限 {MAX_QUESTION_CHARS}")
        # ---- 边界：空问题 / 仅空白 ----
        if normalize_text(question) == "":
            return RouteResult(
                answer=NO_MATCH_TEXT, source="none", score=0.0, reason="empty_question"
            )

        # ---- 步骤 1：QA 数据集匹配 ----
        if logits is None or len(logits) == 0:
            idx, score, irrelevant = -1, 0.0, False
        else:
            idx, score, irrelevant = self.step1_index_from_logits(logits)
        if idx >= 0 and not irrelevant:
            return RouteResult(
                answer=self.answer_text(idx),
                source="qa",
                score=float(score),
                step1_index=int(idx),
                step1_is_irrelevant=False,
            )

        # ---- 步骤 2：文本数据集匹配（仅在步骤 1 未命中时执行） ----
        line, line_score = self.step2_best_line(question)
        if line is not None and line_score >= self.text_threshold:
            return RouteResult(
                answer=line.text,
                source="text",
                score=float(line_score),
                step1_index=int(idx),
                step1_is_irrelevant=bool(irrelevant),
                step2_line_id=line.line_id,
            )
        # ---- 两级都未命中 ----
        reason = "no_candidate" if idx < 0 else "below_threshold"
        return RouteResult(
            answer=NO_MATCH_TEXT,
            source="none",
            score=float(max(score, line_score)),
            step1_index=int(idx),
            step1_is_irrelevant=bool(irrelevant),
            reason=reason,
        )


__all__ = [
    "ROUTE_SOURCES",
    "NO_MATCH_TEXT",
    "DEFAULT_TEXT_THRESHOLD",
    "RouteResult",
    "QuestionRouter",
]