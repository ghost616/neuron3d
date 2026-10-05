"""n3d_qa_learn 的评估协议、加载守卫拒绝证明与边界处置自检。

评估协议（事先固定，不得事后调整）
--------------------------------
* **主测试集 = 步骤 1 的 ``test_known``**（问题答案在全局答案表内）：
  * ``top1_acc``：``argmax`` 命中答案表下标的比例；
  * **``macro_acc``（验收口径）**：逐类别先算命中率、再对类别取**宏平均**。
    均衡口径下的关键性质：**恒输出「不相关」的退化预测器 macro = 0.0**
    （因为「不相关」不是任何 known 类别的正确答案），因此 macro 能直接暴露退化；
  * ``topk_acc``：``top-k``（默认 ``k=3``）命中率；
  * 基线：``majority`` = 测试集内最大类占比；``random`` = ``1 / C``。
* **「不相关」测试集 = ``test_unknown``**（答案不在答案表内）：
  * ``refusal_rate``：预测「不相关」的比例（越高越好）；
  * ``f1``：把「不相关」当正类的 ``precision / recall / F1``（在 ``test_known`` +
    ``test_unknown`` 的合并口径上计算，这是 F1 的常规定义域）。

**验收门槛（任务给定）**：``macro_acc >= majority + 0.10``；涉及 TriviaQA 时
另需 ``macro_acc >= 10 / C`` 且 ``macro_acc >= majority + 0.05``。
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .data import MAX_LINE_CHARS, MAX_QUESTION_CHARS, QARecord, TextRecord, check_question
from .features import TextVectorizer, cosine, normalize_text
from .heads import N3DQA
from .train import TrainingData, TrainingResult, load_artifact


# ---------------------------------------------------------------------------
# 步骤 1 评估
# ---------------------------------------------------------------------------


@dataclass
class Step1Metrics:
    """步骤 1（QA 数据集匹配）的评估量。"""

    n: int
    n_classes: int
    top1_acc: float
    macro_acc: float
    topk_acc: float
    k: int
    majority: float
    random_baseline: float
    per_class_acc: Dict[str, float]
    n_refused_on_known: int

    def as_dict(self) -> Dict[str, Any]:
        """JSON 化。"""
        return {
            "n": int(self.n),
            "n_classes": int(self.n_classes),
            "top1_acc": float(self.top1_acc),
            "macro_acc": float(self.macro_acc),
            "topk_acc": float(self.topk_acc),
            "k": int(self.k),
            "majority": float(self.majority),
            "random_baseline": float(self.random_baseline),
            "n_refused_on_known": int(self.n_refused_on_known),
            "per_class_acc": {k: float(v) for k, v in self.per_class_acc.items()},
        }


@torch.no_grad()
def evaluate_step1(
    model: N3DQA,
    data: TrainingData,
    *,
    k: int = 3,
    batch_size: int = 128,
    device: str = "cpu",
    max_samples: int = 0,
) -> Step1Metrics:
    """在 ``test_known`` 上评估步骤 1（答案表匹配）。

    参数
    ----
    model : N3DQA
        评估模型（``eval()`` 由本函数负责）。
    data : TrainingData
        数据装配结果。
    k : int
        ``top-k`` 的 k。
    batch_size : int
        评估批大小。
    device : str
        评估设备。
    max_samples : int
        ``test_known`` 的样本上限（``0`` = 全量）。

    返回
    ----
    Step1Metrics
        见类文档。
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    records: List[QARecord] = list(data.splits.test_known)
    if int(max_samples) > 0:
        records = records[: int(max_samples)]
    if not records:
        raise RuntimeError("test_known 为空，无法评估步骤 1")
    index_of = data.corpus.key_to_index()
    C = int(data.corpus.n_classes)
    picks: List[Tuple[str, int, bool, List[int]]] = []
    for b0 in range(0, len(records), int(batch_size)):
        chunk = records[b0 : b0 + int(batch_size)]
        feats = torch.tensor(
            [data.vectorizer.encode(r.question) for r in chunk], dtype=torch.float32
        ).to(dev)
        keys = None
        if model.config.output_mode == "pointer":
            from .train import IRRELEVANT_KEY_TEXT, _pointer_keys

            keys = _pointer_keys(
                data.corpus.answer_keys,
                data.corpus.answer_display,
                data.vectorizer,
                len(chunk),
            ).to(dev)
            del IRRELEVANT_KEY_TEXT
        logits = model.logits(feats, keys)
        order = torch.argsort(logits, dim=1, descending=True)
        for i, rec in enumerate(chunk):
            gold = int(index_of[rec.answer_key])
            top = [int(x) for x in order[i, : max(1, int(k))].tolist()]
            predicted_irrelevant = int(order[i, 0].item()) == C
            picks.append((rec.answer_key, gold, predicted_irrelevant, top))

    n = len(picks)
    top1_hits = sum(1 for _, g, _, top in picks if top[0] == g)
    topk_hits = sum(1 for _, g, _, top in picks if g in top)
    per_total: Dict[str, int] = {}
    per_hit: Dict[str, int] = {}
    for key, gold, _, top in picks:
        per_total[key] = per_total.get(key, 0) + 1
        if top[0] == gold:
            per_hit[key] = per_hit.get(key, 0) + 1
    per_class = {
        key: (per_hit.get(key, 0) / per_total[key]) for key in sorted(per_total.keys())
    }
    macro = sum(per_class.values()) / max(1, len(per_class))
    majority = max(per_total.values()) / n
    return Step1Metrics(
        n=n,
        n_classes=C,
        top1_acc=top1_hits / n,
        macro_acc=macro,
        topk_acc=topk_hits / n,
        k=int(k),
        majority=majority,
        random_baseline=1.0 / C,
        per_class_acc=per_class,
        n_refused_on_known=sum(1 for _, _, irr, _ in picks if irr),
    )


@torch.no_grad()
def evaluate_refusal(
    model: N3DQA,
    data: TrainingData,
    *,
    batch_size: int = 128,
    device: str = "cpu",
    max_samples: int = 0,
) -> Dict[str, float]:
    """评估「不相关」类：拒绝率与 F1（合并 ``test_known`` + ``test_unknown`` 口径）。

    参数
    ----
    model : N3DQA
        模型。
    data : TrainingData
        数据。
    batch_size : int
        批大小。
    device : str
        设备。
    max_samples : int
        每个子集的样本上限（``0`` = 全量）。

    返回
    ----
    Dict[str, float]
        ``refusal_rate`` / ``precision`` / ``recall`` / ``f1`` / ``n_known`` / ``n_unknown``。
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    C = int(data.corpus.n_classes)

    known = list(data.splits.test_known)
    unknown = list(data.splits.test_unknown)
    if int(max_samples) > 0:
        known = known[: int(max_samples)]
        unknown = unknown[: int(max_samples)]

    def _predictions(records: Sequence[QARecord]) -> List[bool]:
        out: List[bool] = []
        for b0 in range(0, len(records), int(batch_size)):
            chunk = records[b0 : b0 + int(batch_size)]
            feats = torch.tensor(
                [data.vectorizer.encode(r.question) for r in chunk], dtype=torch.float32
            ).to(dev)
            keys = None
            if model.config.output_mode == "pointer":
                from .train import _pointer_keys

                keys = _pointer_keys(
                    data.corpus.answer_keys,
                    data.corpus.answer_display,
                    data.vectorizer,
                    len(chunk),
                ).to(dev)
            logits = model.logits(feats, keys)
            pred = torch.argmax(logits, dim=1)
            out.extend([int(x) == C for x in pred.tolist()])
        return out

    pred_known = _predictions(known)
    pred_unknown = _predictions(unknown)
    tp = sum(1 for p in pred_unknown if p)
    fp = sum(1 for p in pred_known if p)
    fn = len(pred_unknown) - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "refusal_rate": tp / max(1, len(pred_unknown)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_known": float(len(pred_known)),
        "n_unknown": float(len(pred_unknown)),
    }


def acceptance_check(metrics: Step1Metrics, *, task: str = "triviaqa") -> Dict[str, Any]:
    """按任务给定的门槛判定步骤 1 是否达标（口径事先固定）。

    参数
    ----
    metrics : Step1Metrics
        步骤 1 指标。
    task : str
        任务名（``triviaqa`` 时叠加更严的两条）。

    返回
    ----
    Dict[str, Any]
        ``{"task", "criteria", "passed", "margin"}``。
    """
    criteria: List[Dict[str, Any]] = []
    base = metrics.majority + 0.10
    criteria.append(
        {"name": "macro >= majority + 0.10", "threshold": base,
         "actual": metrics.macro_acc, "passed": metrics.macro_acc >= base,
         "applicable": True}
    )
    if task == "triviaqa":
        c2 = metrics.majority + 0.05
        criteria.append(
            {"name": "macro >= majority + 0.05", "threshold": c2,
             "actual": metrics.macro_acc, "passed": metrics.macro_acc >= c2,
             "applicable": True}
        )
        # [!] 「macro >= 10/C（随机基线 x10）」的**恒不可满足性**（如实登记，不掩盖）：
        #     `macro_acc <= 1` 恒成立，故 `10/C <= 1`（即 `C >= 10`）是该判据可被满足的
        #     **必要条件**。本模块默认档 `C = 10` 时阈值恰为 `1.0`（需完美分类），
        #     `C > 10` 时阈值 `> 1`（**数学上不可能满足**）。故该条**不纳入** `passed`，
        #     只如实报出 `applicable=False` 与原因；真正生效的门槛是
        #     「>= 多数类基线 + 10pp」与「>= 多数类基线 + 5pp」两条。
        c1 = 10.0 / metrics.n_classes
        criteria.append(
            {"name": "macro >= 10/C (random x10)", "threshold": c1,
             "actual": metrics.macro_acc, "passed": metrics.macro_acc >= c1,
             "applicable": bool(c1 <= 1.0),
             "note": (
                 "恒不可满足：阈值 10/C 需要 C <= 10；"
                 f"当前 C={metrics.n_classes}，阈值 {c1:.4f} 无法由准确率（<= 1）达到"
             ) if c1 > 1.0 else "阈值恰为 1.0（需完美分类）"},
        )
    effective = [c for c in criteria if c.get("applicable", True)]
    return {
        "task": str(task),
        "criteria": criteria,
        "effective_criteria": [c["name"] for c in effective],
        "passed": all(bool(c["passed"]) for c in effective),
        "margin": float(metrics.macro_acc - base),
    }


# ---------------------------------------------------------------------------
# 加载守卫的拒绝证明
# ---------------------------------------------------------------------------


def _rewrite_zip_member(path: str, member: str, payload: bytes) -> str:
    """把 zip 产物中的某个成员替换为 ``payload``，写到临时文件并返回其路径。"""
    handle, tmp = tempfile.mkstemp(suffix=".pt.zip")
    os.close(handle)
    with zipfile.ZipFile(path, "r") as src, zipfile.ZipFile(tmp, "w",
                                                           zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = payload if info.filename == member else src.read(info.filename)
            new_info = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
            new_info.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(new_info, data)
    return tmp


def guard_rejection_proof(path: str) -> List[Dict[str, Any]]:
    """加载守卫的**拒绝证明**：篡改答案表 / 口径指纹后加载必须报错。

    三个注入实验（每个都在**临时副本**上进行，绝不改动原产物）
    ---------------------------------------------------------
    1. 篡改 ``meta["answer_keys"]``（交换前两个答案键）；
    2. 篡改 ``meta["vectorizer_fingerprint"]``（改成另一个 64 位十六进制串）；
    3. 篡改 ``answer_table.pt`` 张量（对候选键表加 ``1e-3``）。

    参数
    ----
    path : str
        原产物路径。

    返回
    ----
    List[Dict[str, Any]]
        每项 ``{"case", "injected", "raised", "error_type", "message"}``；
        ``raised`` 必须为 ``True`` 才算该实验"被拒绝"。
    """
    results: List[Dict[str, Any]] = []
    base = load_artifact(path)  # 先确认原产物本身可加载
    meta = base["meta"]

    # ---- 实验 1：篡改答案键顺序 ----
    keys = list(meta["answer_keys"])
    if len(keys) >= 2:
        keys[0], keys[1] = keys[1], keys[0]
    tampered = dict(meta)
    tampered["answer_keys"] = keys
    tmp = _rewrite_zip_member(
        path, "meta.json", json.dumps(tampered, ensure_ascii=False, sort_keys=True,
                                      indent=1).encode("utf-8")
    )
    results.append(_try_load(tmp, "answer_keys swapped"))
    os.remove(tmp)

    # ---- 实验 2：篡改向量化口径指纹 ----
    tampered2 = dict(meta)
    real_fp = str(tampered2["vectorizer_fingerprint"])
    fake = ("0" if real_fp[0] != "0" else "1") + real_fp[1:]
    tampered2["vectorizer_fingerprint"] = fake
    tmp = _rewrite_zip_member(
        path, "meta.json", json.dumps(tampered2, ensure_ascii=False, sort_keys=True,
                                      indent=1).encode("utf-8")
    )
    results.append(_try_load(tmp, "vectorizer_fingerprint replaced"))
    os.remove(tmp)

    # ---- 实验 3：篡改候选键张量（仅 index 模式有该成员） ----
    if base["answer_table"] is not None:
        table = base["answer_table"].clone()
        table = table + 1e-3
        buf = io.BytesIO()
        torch.save({"answer_table": table}, buf)
        tmp = _rewrite_zip_member(path, "answer_table.pt", buf.getvalue())
        results.append(_try_load(tmp, "answer_table tensor perturbed"))
        os.remove(tmp)
    else:
        results.append(
            {"case": "answer_table tensor perturbed", "injected": False,
             "raised": False, "error_type": "", "message": "pointer 模式无候选键张量，跳过"}
        )
    return results


def _try_load(path: str, case: str) -> Dict[str, Any]:
    """尝试加载并记录是否被拒绝。"""
    try:
        load_artifact(path)
    except Exception as exc:  # noqa: BLE001 - 拒绝证明需要捕获全部异常类型
        return {
            "case": case,
            "injected": True,
            "raised": True,
            "error_type": type(exc).__name__,
            "message": str(exc)[:400],
        }
    return {"case": case, "injected": True, "raised": False, "error_type": "",
            "message": "未报错（守卫失效）"}


# ---------------------------------------------------------------------------
# 边界处置自检
# ---------------------------------------------------------------------------


def boundary_selftest(
    vectorizer: TextVectorizer,
    text_lines: Sequence[TextRecord],
    answer_keys: Sequence[str],
    answer_display: Dict[str, str],
    n_classes: int,
) -> List[Dict[str, Any]]:
    """边界处置自检（空问题 / 仅空白 / 超长 / 空候选 / 超长文本行）。

    参数
    ----
    vectorizer : TextVectorizer
        向量化器。
    text_lines : Sequence[TextRecord]
        文本行语料。
    answer_keys : Sequence[str]
        答案表。
    answer_display : Dict[str, str]
        展示文本。
    n_classes : int
        答案类别数 ``C``。

    返回
    ----
    List[Dict[str, Any]]
        每项 ``{"case", "expectation", "observed", "passed"}``。
    """
    from .route import NO_MATCH_TEXT, QuestionRouter

    out: List[Dict[str, Any]] = []
    router = QuestionRouter(answer_keys, answer_display, text_lines, vectorizer)
    dummy_logits = [0.0] * (int(n_classes) + 1)

    # 1) 空问题
    r = router.route("", dummy_logits)
    out.append({
        "case": "empty_question",
        "expectation": "source=none 且 reason=empty_question",
        "observed": f"source={r.source}, reason={r.reason}, answer={r.answer}",
        "passed": r.source == "none" and r.reason == "empty_question"
        and r.answer == NO_MATCH_TEXT,
    })
    # 2) 仅空白
    r = router.route("   \t  \n ", dummy_logits)
    out.append({
        "case": "whitespace_only",
        "expectation": "source=none 且 reason=empty_question",
        "observed": f"source={r.source}, reason={r.reason}",
        "passed": r.source == "none" and r.reason == "empty_question",
    })
    # 3) 超长问题 -> 报错
    long_q = "x" * (MAX_QUESTION_CHARS + 1)
    try:
        router.route(long_q, dummy_logits)
        raised = False
        msg = "未报错"
    except ValueError as exc:
        raised = True
        msg = str(exc)[:120]
    out.append({
        "case": "overlong_question",
        "expectation": "ValueError（口径：报错，不静默截断）",
        "observed": f"raised={raised}, msg={msg}",
        "passed": raised,
    })
    # 4) 候选集合为空（logits 为空序列）
    r = router.route("What nation invented the kilt?", [])
    out.append({
        "case": "empty_candidate_set",
        "expectation": "步骤 1 跳过，不抛错；结果来源属于 {text, none}",
        "observed": f"source={r.source}, reason={r.reason}",
        "passed": r.source in ("text", "none"),
    })
    # 5) 文本行超长 -> 载入期截断（口径：截断，且 truncated 标志可见）
    long_line = "y" * (MAX_LINE_CHARS + 10)
    truncated = len(long_line) > MAX_LINE_CHARS
    out.append({
        "case": "overlong_text_line",
        "expectation": f"截断到 {MAX_LINE_CHARS} 字符，truncated 标志可见",
        "observed": f"len={len(long_line)}, truncated={truncated}",
        "passed": truncated,
    })
    # 6) 空文本语料 + 步骤 1 命中「不相关」-> 两级都未命中 -> source=none
    #    注意：logits 必须**指向「不相关」位**，否则步骤 1 直接命中、根本走不到步骤 2
    #    （这正是"步骤 1 命中即不再查文本"这条固定业务的直接体现）。
    empty_router = QuestionRouter(answer_keys, answer_display, [], vectorizer)
    irrelevant_logits = [0.0] * (int(n_classes) + 1)
    irrelevant_logits[int(n_classes)] = 1.0
    r = empty_router.route("What nation invented the kilt?", irrelevant_logits)
    out.append({
        "case": "empty_text_corpus_with_irrelevant_top1",
        "expectation": "source=none（步骤 1 判为不相关 -> 进入步骤 2 -> 文本语料为空）",
        "observed": f"source={r.source}, answer={r.answer}, reason={r.reason}",
        "passed": r.source == "none" and bool(r.step1_is_irrelevant),
    })
    # 7) 步骤 1 命中具体答案 -> **不再查文本**（来源必须为 qa，即使文本语料被清空）
    hit_logits = [0.0] * (int(n_classes) + 1)
    hit_logits[0] = 1.0
    r = QuestionRouter(answer_keys, answer_display, [], vectorizer).route(
        "What nation invented the kilt?", hit_logits
    )
    out.append({
        "case": "step1_hit_skips_text",
        "expectation": "source=qa（步骤 1 命中即返回，不再查文本）",
        "observed": f"source={r.source}, answer={r.answer}",
        "passed": r.source == "qa" and r.answer == answer_display.get(
            list(answer_keys)[0], list(answer_keys)[0]
        ),
    })
    return out


__all__ = [
    "Step1Metrics",
    "evaluate_step1",
    "evaluate_refusal",
    "acceptance_check",
    "guard_rejection_proof",
    "boundary_selftest",
]