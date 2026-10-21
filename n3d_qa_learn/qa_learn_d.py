"""轮 11 / 轮 D：A 口径（82 类）上的**可学变换** —— 第一次真正「上模型」。

问题
----
轮 9/10 已固化 A 口径（每类问题数 >= 10 → 82 类）与**零训练**原型基线：官方「全部有测试
类等权 macro」下原型 0.241276 对多数类 0.012987（+22.83pp 过门），但分层后 >=10 测试
样本类只有 +5.41pp，且 micro 反而更低（0.121055 vs 0.183312）、34/77 类零召回。本轮要回答
的是：**这到底是「特征瓶颈」还是「模型瓶颈」** —— 做法是在同一批冻结特征上训练一个
**可学 D->D 变换 M**（恒等初始化 M=I），接**冻结**的类质心原型，打分 = L2 归一化余弦 ×
可学温度，softmax 交叉熵。

结构（**只有一处可学**）
------------------------
    x (D=88, 冻结词面特征)  --M (D x D, 恒等初始化, 唯一的主可学矩阵)-->  z = normalize(M x)
    score = tau * <z, c_k>   (c_k = 该类**训练侧**问题特征的质心，冻结；tau 可学标量)
    loss  = multi-positive CE（一个问题可有多金标，逐金标取 -log p 的均值）

* **恒等初始化**是硬约束：M=I 且 tau=1 时，step 0 的测试侧预测必须与轮 9 的原型基线
  **逐样本 argmax 相同**、macro/micro **完全相等**（见 :func:`step0_equivalence`）。
  否则任何「训练后增益」都不能归因于学习。
* **不重新调参**：切分（20% 按比例留出 / question_id 整组 / 单 seed 42）、类集（82）、
  特征档（lexical-88 / D=88）、原型口径全部沿用轮 9/10；本轮只练 M 与 tau。
* **判据（双判据，必须同时满足）**：``macro > 0.241276`` **且** ``micro >= 0.183312``
  （基线：原型 macro 0.2412763685865515、多数类 micro 0.18331171638564633）。
* **禁止代理量**：gap/sigma、对齐度、有效秩一类的代理量在历史上两次被证否，**不得**替代
  本判据；本模块也不产出它们的「结论」用法（只在需要时作为诊断，且不参与判定）。

产物与纪律
----------
* 产物只写 ``checkpoints/qa_learn/_verify/qa_learn_d/``；
* **不产 zip、不落盘裸特征矩阵**（矩阵只在内存里）；无挂钟字段 ⇒ 同参双跑逐字节一致；
* 上游 ``n3d_qa`` 冻结产物**只读**；零新增依赖（torch / numpy 已是既有依赖）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# 冻结常量（口径进产物）
# ---------------------------------------------------------------------------

#: 模块版本（结构变化必须递增；进产物）。
QA_LEARN_D_VERSION: str = "n3dqa-qa-learn-d-v1"

#: 冻结产物目录（**只读**）。
DEFAULT_PRODUCT_DIR: str = "checkpoints/qa_learn/dataset"

#: 产物目录（**唯一**允许写盘的根）。
VERIFY_ROOT: str = os.path.join("checkpoints", "qa_learn", "_verify")
DEFAULT_OUT_DIR: str = os.path.join(VERIFY_ROOT, "qa_learn_d")

#: 单 seed / 单特征档（与轮 9 逐位相同）。
DEFAULT_SEED: int = 42
DEFAULT_PROFILE: str = "lexical-88"

#: 预注册训练配置（在看到测试集结果之前固定；**不得**按判据回调参数）。
#: 步数是**上限**：停止准则只用训练侧损失（相对变化 < 0.5% / 50 步即停）。
DEFAULT_STEPS: int = 3000
DEFAULT_LR: float = 0.01
PROBE_STEPS: int = 200
DIAG_EVERY: int = 50
CONVERGE_WINDOW: int = 50
CONVERGE_REL_TOL: float = 0.005

#: 冻结基线（轮 9/10，A 口径、单 seed 42）。本轮**复算核对**，不一致即停并报差异。
FROZEN_PROTOTYPE_MACRO: float = 0.2412763685865515
FROZEN_PROTOTYPE_MICRO: float = 0.12105490704712495
FROZEN_MAJORITY_MACRO: float = 0.012987012987012988
FROZEN_MAJORITY_MICRO: float = 0.18331171638564633
FROZEN_N_CLASSES: int = 82
FROZEN_N_QUESTIONS: int = 12005
FROZEN_N_PAIRS: int = 47495
FROZEN_N_TRAIN_QUESTIONS: int = 9692
FROZEN_N_TEST_QUESTIONS: int = 2313
FROZEN_N_TRAIN_ROWS: int = 38361
FROZEN_N_TEST_ROWS: int = 9134

#: 双判据阈值（任务书 3b 的字面值）。
MACRO_CRITERION: float = 0.241276
MICRO_CRITERION: float = 0.183312

#: 分层档（沿用轮 10 的四档）。
STRATIFICATION_THRESHOLDS: Tuple[int, ...] = (1, 2, 5, 10)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    """UTF-8 裸字节 SHA256（用于给预测序列留指纹）。"""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _canonical_json(obj: Any) -> str:
    """规范化 JSON（跨运行一致）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _assert_frozen(label: str, observed: float, expected: float) -> None:
    """复算值必须与轮 9/10 的冻结值**完全相等**（不一致即停，不默默改数）。"""
    if float(observed) != float(expected):
        raise ValueError(
            "冻结基线复算不一致：%s 现场 = %r，轮 9/10 冻结值 = %r（差 %.17g）；"
            "按任务书要求**停止并报差异**，不得默默改数"
            % (label, float(observed), float(expected), float(observed) - float(expected))
        )


def _quantiles(values: Sequence[float]) -> Dict[str, Any]:
    """一组数的 min/p25/median/p75/max/mean（空序列返回 0）。"""
    import numpy as np

    arr = np.asarray([float(x) for x in values], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }
# ---------------------------------------------------------------------------
# 数据集（**冻结**：A 口径 82 类 + 轮 9 的特征 / 切分 / 原型）
# ---------------------------------------------------------------------------


def load_dataset(
    product_dir: str = DEFAULT_PRODUCT_DIR,
    profile: str = DEFAULT_PROFILE,
) -> Dict[str, Any]:
    """按轮 9 的口径复算 A 口径数据集，并与冻结基线**逐值核对**。

    返回
    ----
    Dict[str, Any]
        ``corpus`` / ``split`` / ``qids`` / ``matrix``（**只在内存**）/ ``gold`` /
        ``centroids`` / ``baseline``（预测与指标）/ ``majority`` / ``prototype_support`` /
        ``document``（语料摘要）。

    异常
    ------
    ValueError
        语料规模或基线指标与轮 9/10 冻结值不一致（**停并报差异**）。
    """
    import n3d_qa_learn.qa_scoped as S
    import n3d_qa_learn.qa_scale as QS

    built = S.build_scoped_corpus(str(product_dir))
    corpus = built["corpus"]
    split = built["split"]
    if int(corpus.n_classes) != FROZEN_N_CLASSES:
        raise ValueError(
            "A 口径类数 = %d，与冻结值 %d 不一致；拒绝继续"
            % (int(corpus.n_classes), FROZEN_N_CLASSES)
        )
    for label, observed, expected in (
        ("n_questions", int(corpus.n_questions), FROZEN_N_QUESTIONS),
        ("n_pairs", int(corpus.n_pairs), FROZEN_N_PAIRS),
        ("n_train_questions", int(len(split.train_questions)), FROZEN_N_TRAIN_QUESTIONS),
        ("n_test_questions", int(len(split.test_questions)), FROZEN_N_TEST_QUESTIONS),
        ("n_train_rows", int(len(split.train_rows)), FROZEN_N_TRAIN_ROWS),
        ("n_test_rows", int(len(split.test_rows)), FROZEN_N_TEST_ROWS),
    ):
        if int(observed) != int(expected):
            raise ValueError(
                "切分/语料规模与轮 9 冻结值不一致：%s 现场 = %d，冻结 = %d；拒绝继续"
                % (label, int(observed), int(expected))
            )
    texts = S.load_question_texts(str(product_dir))
    gold = S.gold_local_map(corpus)
    qids = sorted(str(q) for q in corpus.question_gold.keys())
    features = S.build_question_matrix(
        texts["texts"], qids, profile=str(profile), role=S.ROLE_QUESTION
    )
    prototypes = S.class_prototypes(
        features["matrix"],
        qids,
        gold,
        split.train_questions,
        n_classes=int(corpus.n_classes),
    )
    predicted = S.predict_by_cosine(
        features["matrix"], qids, split.test_questions, prototypes["centroids"]
    )
    proto_metrics = S.question_level_metrics(
        predictions=predicted["predictions"],
        questions=split.test_questions,
        gold_local=gold,
        n_classes=int(corpus.n_classes),
    )
    train_counts = QS.side_class_counts(corpus, split, side="train", unit="question")
    bonus = S.majority_predictions(
        train_counts=train_counts, n_test_questions=len(split.test_questions)
    )
    majority_metrics = S.question_level_metrics(
        predictions=bonus["predictions"],
        questions=split.test_questions,
        gold_local=gold,
        n_classes=int(corpus.n_classes),
    )
    # **复算核对**：与轮 9/10 冻结值完全相等，否则停并报差异（不默默改数）
    _assert_frozen(
        "prototype.macro", float(proto_metrics["macro"]), FROZEN_PROTOTYPE_MACRO
    )
    _assert_frozen(
        "prototype.micro", float(proto_metrics["micro"]), FROZEN_PROTOTYPE_MICRO
    )
    _assert_frozen(
        "majority.macro", float(majority_metrics["macro"]), FROZEN_MAJORITY_MACRO
    )
    _assert_frozen(
        "majority.micro", float(majority_metrics["micro"]), FROZEN_MAJORITY_MICRO
    )
    return {
        "corpus": corpus,
        "split": split,
        "qids": qids,
        "matrix": features["matrix"],
        "feature": {
            "profile": str(features["profile"]),
            "role": str(features["role"]),
            "dim": int(features["dim"]),
            "fingerprint": str(features["fingerprint"]),
            "n_texts": int(features["n_texts"]),
            "n_zero_norm": int(features["n_zero_norm"]),
        },
        "gold": gold,
        "centroids": prototypes["centroids"],
        "prototype_support": [int(x) for x in prototypes["counts"]],
        "baseline": {
            "predictions": [int(x) for x in predicted["predictions"]],
            "metrics": dict(proto_metrics),
            "n_ties": int(predicted["n_ties"]),
            "margin_distribution": _quantiles(predicted["margin"]),
        },
        "majority": {
            "predictions": [int(x) for x in bonus["predictions"]],
            "metrics": dict(majority_metrics),
            "class_index": int(bonus["class_index"]),
            "class_key": str(corpus.answer_keys[int(bonus["class_index"])]),
            "class_train_question_count": int(bonus["class_train_question_count"]),
            "train_counts": [int(x) for x in train_counts],
        },
        "document": built["document"],
    }


def per_class_stats(
    predictions: Sequence[int],
    questions: Sequence[str],
    gold: Mapping[str, Sequence[int]],
    n_classes: int,
) -> Dict[str, Any]:
    """逐类测试问题数与命中数（与 :func:`qa_scoped.question_level_metrics` 同一口径）。"""
    total = [0] * int(n_classes)
    hit = [0] * int(n_classes)
    for qid, pred in zip([str(q) for q in questions], [int(x) for x in predictions]):
        for cls in sorted({int(x) for x in gold[str(qid)]}):
            total[int(cls)] += 1
            if int(pred) == int(cls):
                hit[int(cls)] += 1
    return {"total": total, "hit": hit}


def stratified_rows(
    *,
    model_predictions: Sequence[int],
    prototype_predictions: Sequence[int],
    majority_predictions: Sequence[int],
    questions: Sequence[str],
    gold: Mapping[str, Sequence[int]],
    n_classes: int,
    class_keys: Sequence[str],
) -> Dict[str, Any]:
    """**按测试样本量分层**（沿用轮 10 四档）的 model / prototype / majority macro。

    每档给出 ``model_macro``、``prototype_macro``、``majority_macro``、
    ``delta_vs_prototype``、``delta_vs_majority``；空档记 ``applicable = False``（不抛错）。
    """
    model_stats = per_class_stats(model_predictions, questions, gold, n_classes)
    base_stats = per_class_stats(prototype_predictions, questions, gold, n_classes)
    major_stats = per_class_stats(majority_predictions, questions, gold, n_classes)
    rows: List[Dict[str, Any]] = []
    for threshold in STRATIFICATION_THRESHOLDS:
        keep = [
            int(c)
            for c in range(int(n_classes))
            if int(model_stats["total"][int(c)]) >= int(threshold)
        ]
        if not keep:
            rows.append(
                {
                    "min_test_questions": int(threshold),
                    "applicable": False,
                    "n_classes": 0,
                    "rule": "该档没有任何类（测试侧问题数 >= %d 的类集为空）" % int(threshold),
                }
            )
            continue

        def _macro(stats: Mapping[str, Any]) -> float:
            recalls = [
                float(stats["hit"][c]) / float(stats["total"][c]) for c in keep
            ]
            return float(sum(recalls) / len(recalls))

        model_macro = _macro(model_stats)
        base_macro = _macro(base_stats)
        major_macro = _macro(major_stats)
        rows.append(
            {
                "min_test_questions": int(threshold),
                "n_classes": int(len(keep)),
                "n_test_questions_in_scope": int(
                    sum(int(model_stats["total"][c]) for c in keep)
                ),
                "model_macro": model_macro,
                "prototype_macro": base_macro,
                "majority_macro": major_macro,
                "delta_vs_prototype": float(model_macro - base_macro),
                "delta_vs_majority": float(model_macro - major_macro),
                "reaches_macro_criterion": bool(model_macro > MACRO_CRITERION),
                "class_keys": [str(class_keys[c]) for c in keep[:5]],
            }
        )
    return {
        "unit": "question",
        "thresholds": [int(x) for x in STRATIFICATION_THRESHOLDS],
        "rows": rows,
        "rule": (
            "按「该类测试侧问题数 >= k」分层后重算同一问题级口径的 macro（分母 = 该层类数）；"
            "`delta_vs_prototype` = 本模型 macro − 轮 9 冻结原型 macro（同一批测试问题、"
            "同一层类集）。该表是**评测**的一部分，不改变主判据（主判据仍用全部有测试类）。"
        ),
    }
# ---------------------------------------------------------------------------
# 模型：可学 D->D 变换（恒等初始化）+ 冻结质心 + 可学温度
# ---------------------------------------------------------------------------


class LearnableTransform(torch.nn.Module):
    """``z = normalize(M x)``、``logits = tau * (z @ C^T)``；``M`` 恒等初始化、``tau`` 初值 1。

    只有 **M（D×D）** 与 **log_temperature（标量）** 可学；类质心 ``C`` 是冻结的 buffer
    （由训练侧特征算出，不进优化器）。

    关键不变量（构造期）
    --------------------
    * ``M`` 的初值**逐位等于**单位阵（``torch.eye``），``log_temperature`` 初值为 0 ⇒
      ``tau = 1`` ⇒ step 0 的 logits 等于「未做任何变换的余弦相似度」；
    * ``M`` 用 float64：步 0 与 numpy 基线路径的差异只来自「是否再归一化」，量级 ~1e-7，
      远小于基线的**最小 margin 1.39e-6**（现场），故 argmax 不会被浮点噪声翻转。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim: int = int(dim)
        self.M = torch.nn.Parameter(torch.eye(self.dim, dtype=torch.float64))
        self.log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def temperature(self) -> torch.Tensor:
        """``tau = exp(clamp(log_temperature, -8, 8))``（clamp 只为数值安全）。"""
        return torch.exp(torch.clamp(self.log_temperature, -8.0, 8.0))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """``z = normalize(M x)``（逐行 L2 归一化；零向量由 eps 兜底）。"""
        return torch.nn.functional.normalize(x @ self.M.t(), p=2, dim=1, eps=1e-12)

    def logits(self, x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """``tau * (normalize(M x) @ C^T)``。"""
        return self.temperature() * (self.transform(x) @ centroids.t())


def multi_positive_ce(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """**多正标**交叉熵：一个问题可有多金标，逐金标取 ``-log p`` 的均值再按问题平均。"""
    logp = torch.log_softmax(logits, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1.0)
    return -((logp * mask).sum(dim=1) / denom).mean()


def _as_tensor(matrix: Any, rows: Sequence[int]) -> torch.Tensor:
    """从内存特征矩阵取指定行，转 float64 张量（**不落盘**）。"""
    import numpy as np

    sub = np.asarray([matrix[int(r)] for r in rows], dtype=np.float64)
    return torch.tensor(sub, dtype=torch.float64)


def _gold_mask(
    qids: Sequence[str], gold: Mapping[str, Sequence[int]], n_classes: int
) -> torch.Tensor:
    """问题 -> 金标类 0/1 掩码 ``[N, C]``（float64）。"""
    import numpy as np

    mask = np.zeros((len(qids), int(n_classes)), dtype=np.float64)
    for i, qid in enumerate(qids):
        classes = sorted({int(x) for x in gold[str(qid)]})
        if not classes:
            raise ValueError("训练问题 " + repr(qid) + " 没有金标，无法构造多正标掩码")
        for cls in classes:
            mask[int(i), int(cls)] = 1.0
    return torch.tensor(mask, dtype=torch.float64)


def _predict(logits: torch.Tensor) -> List[int]:
    """argmax 预测（平票取最小类下标，与原型基线同一 numpy 语义）。"""
    return [int(x) for x in logits.argmax(dim=1).tolist()]


def train_transform(
    model: LearnableTransform,
    x_train: torch.Tensor,
    mask: torch.Tensor,
    centroids: torch.Tensor,
    *,
    steps: int,
    lr: float,
    diag_every: int,
    train_diagnostics: Any,
    converge_window: int = 50,
    converge_rel_tol: float = 0.005,
) -> Dict[str, Any]:
    """全批 Adam 训练 ``M`` 与 ``tau``（**无随机性**：无 shuffle、无 dropout、无初始化随机）。

    **停止准则只用训练侧信息**（轮 11 现场补充，见 ``stopped_reason``）：先跑 300 步的
    试点显示训练损失每 50 步仍下降 5.9%（**远未收敛**），故正式配置改为「训练损失相对
    变化 < ``converge_rel_tol``（0.5%）/ ``converge_window``（50）步即停，上限 ``steps``」。
    该准则**不看测试集**，不构成按判据调参。

    参数
    ----
    model : LearnableTransform
        step 0 = 恒等初始化。
    x_train / mask / centroids : torch.Tensor
        训练侧特征 ``[N_tr, D]``、多正标掩码 ``[N_tr, C]``、冻结质心 ``[C, D]``。
    steps / lr / diag_every : int / float / int
        步数上限、学习率（Adam）、诊断间隔。
    train_diagnostics : Callable[[int], Dict[str, Any]]
        在第 ``k`` 步（含第 0 步）被调用，返回**训练侧**指标（**仅诊断**，不参与判据）。
    converge_window / converge_rel_tol : int / float
        收敛窗口（步）与相对变化门限（训练损失）。

    返回
    ----
    Dict[str, Any]
        ``losses``（每步一个）、``diagnostics``（每 diag_every 步）、``steps_run``（实际步数）、
        ``stopped_reason`` / ``last_window_rel_change`` / ``lr`` / ``optimizer`` / ``n_parameters``。
    """
    optimizer = torch.optim.Adam([model.M, model.log_temperature], lr=float(lr))
    losses: List[float] = []
    diagnostics: List[Dict[str, Any]] = []
    steps_run = 0
    stopped_reason = "max_steps"
    rel_change = 0.0
    window = max(1, int(converge_window))
    for step in range(int(steps) + 1):
        if step % int(diag_every) == 0 or step == int(steps):
            diagnostics.append(train_diagnostics(int(step)))
        if step == int(steps):
            break
        optimizer.zero_grad(set_to_none=True)
        logits = model.logits(x_train, centroids)
        loss = multi_positive_ce(logits, mask)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        steps_run = step + 1
        if steps_run > window and steps_run % window == 0:
            prev = losses[-window - 1]
            rel_change = abs(losses[-1] - prev) / max(abs(prev), 1e-12)
            if rel_change < float(converge_rel_tol):
                stopped_reason = "converged"
                break
    return {
        "losses": losses,
        "diagnostics": diagnostics,
        "steps": int(steps),
        "steps_run": int(steps_run),
        "stopped_reason": str(stopped_reason),
        "last_window_rel_change": float(rel_change),
        "converge_window": int(window),
        "converge_rel_tol": float(converge_rel_tol),
        "lr": float(lr),
        "optimizer": "Adam(full-batch, no weight decay)",
        "diag_every": int(diag_every),
        "n_parameters": int(model.M.numel() + model.log_temperature.numel()),
        "shuffle": False,
        "dropout": False,
    }


def step0_equivalence(
    model: LearnableTransform,
    x_test: torch.Tensor,
    centroids: torch.Tensor,
    dataset: Mapping[str, Any],
) -> Dict[str, Any]:
    """**硬证据**：step 0（未训练）的测试侧预测必须与原型基线逐位相同。

    判据（三条全成立才算过，任一不成立即抛 ``AssertionError``）：
    1. 逐样本 argmax 相同（``n_argmax_diff == 0``）；
    2. ``macro`` / ``micro`` 与基线**完全相等**（float 相等，不是近似）；
    3. ``torch`` 路径与 numpy 基线余弦的最大绝对差记录在案（诊断量，门限 1e-5）。
    """
    import numpy as np

    import n3d_qa_learn.qa_scoped as S

    if not torch.equal(model.M, torch.eye(model.dim, dtype=torch.float64)):
        raise AssertionError("step 0 检查要求 M 逐位等于单位阵（恒等初始化）")
    if float(model.log_temperature.detach()) != 0.0:
        raise AssertionError("step 0 检查要求 log_temperature == 0（tau = 1）")
    with torch.no_grad():
        logits0 = model.logits(x_test, centroids)
        preds0 = _predict(logits0)
    base_preds = [int(x) for x in dataset["baseline"]["predictions"]]
    diff_idx = [i for i, (a, b) in enumerate(zip(preds0, base_preds)) if int(a) != int(b)]
    metrics0 = S.question_level_metrics(
        predictions=preds0,
        questions=dataset["split"].test_questions,
        gold_local=dataset["gold"],
        n_classes=int(dataset["corpus"].n_classes),
    )
    base_metrics = dataset["baseline"]["metrics"]
    # numpy 基线余弦（用于量化两条路径的数值差；不是判据本身）
    sims_np = np.asarray(x_test.numpy(), dtype=np.float64) @ np.asarray(
        centroids.numpy(), dtype=np.float64
    ).T
    max_abs_diff = float(np.max(np.abs(logits0.numpy() - sims_np)))
    equal_macro = float(metrics0["macro"]) == float(base_metrics["macro"])
    equal_micro = float(metrics0["micro"]) == float(base_metrics["micro"])
    doc: Dict[str, Any] = {
        "applicable": True,
        "init": "M = I（逐位）, log_temperature = 0 ⇒ tau = 1",
        "n_test_questions": int(len(preds0)),
        "n_argmax_diff": int(len(diff_idx)),
        "argmax_identical": bool(not diff_idx),
        "diff_indices_head": [int(x) for x in diff_idx[:10]],
        "macro_step0": float(metrics0["macro"]),
        "macro_baseline": float(base_metrics["macro"]),
        "micro_step0": float(metrics0["micro"]),
        "micro_baseline": float(base_metrics["micro"]),
        "macro_exactly_equal": bool(equal_macro),
        "micro_exactly_equal": bool(equal_micro),
        "max_abs_logit_diff_vs_numpy": max_abs_diff,
        "max_abs_logit_diff_tolerance": 1e-5,
        "n_ties_baseline": int(dataset["baseline"]["n_ties"]),
        "rule": (
            "M=I、tau=1 时，torch 路径的 argmax 必须与 numpy 原型基线逐样本相同，且 macro/micro "
            "float 完全相等；否则「训练带来的增益」无法归因于学习，本轮结论不得成立。"
        ),
    }
    if diff_idx:
        raise AssertionError(
            "step 0 等价性失败：%d / %d 个测试问题的 argmax 与原型基线不同（前几个 = %s）；"
            "拒绝在该初始化上归因增益" % (len(diff_idx), len(preds0), doc["diff_indices_head"])
        )
    if not (equal_macro and equal_micro):
        raise AssertionError(
            "step 0 等价性失败：macro/micro 与基线不相等（step0 = %.17g / %.17g，基线 = "
            "%.17g / %.17g）"
            % (
                float(metrics0["macro"]),
                float(metrics0["micro"]),
                float(base_metrics["macro"]),
                float(base_metrics["micro"]),
            )
        )
    if max_abs_diff > 1e-5:
        raise AssertionError(
            "step 0 等价性失败：torch 与 numpy 余弦最大绝对差 = %.3e > 1e-5" % max_abs_diff
        )
    return doc
# ---------------------------------------------------------------------------
# 评测、逐类 delta、双判据
# ---------------------------------------------------------------------------


def evaluate_model(
    model: LearnableTransform,
    x_test: torch.Tensor,
    centroids: torch.Tensor,
    dataset: Mapping[str, Any],
) -> Dict[str, Any]:
    """测试侧评测：macro / micro / 分层四档 / 每类 recall / 零 recall 类（**主判据口径**）。"""
    import n3d_qa_learn.qa_scoped as S

    corpus = dataset["corpus"]
    split = dataset["split"]
    gold = dataset["gold"]
    n_classes = int(corpus.n_classes)
    with torch.no_grad():
        preds = _predict(model.logits(x_test, centroids))
    metrics = S.question_level_metrics(
        predictions=preds,
        questions=split.test_questions,
        gold_local=gold,
        n_classes=n_classes,
    )
    strata = stratified_rows(
        model_predictions=preds,
        prototype_predictions=dataset["baseline"]["predictions"],
        majority_predictions=dataset["majority"]["predictions"],
        questions=split.test_questions,
        gold=gold,
        n_classes=n_classes,
        class_keys=corpus.answer_keys,
    )
    stats = per_class_stats(preds, split.test_questions, gold, n_classes)
    base_stats = per_class_stats(
        dataset["baseline"]["predictions"], split.test_questions, gold, n_classes
    )
    per_class = []
    for cls in range(n_classes):
        total = int(stats["total"][cls])
        if total <= 0:
            continue
        per_class.append(
            {
                "class_index": int(cls),
                "class_key": str(corpus.answer_keys[cls]),
                "test_questions": total,
                "model_recall": float(stats["hit"][cls]) / float(total),
                "prototype_recall": float(base_stats["hit"][cls]) / float(total),
            }
        )
    with torch.no_grad():
        tau = float(model.temperature().item())
        m_disp = float(torch.linalg.matrix_norm(model.M - torch.eye(model.dim, dtype=torch.float64)).item())
    return {
        "version": str(QA_LEARN_D_VERSION),
        "caliber": {
            "class_set": "A（每类问题数 >= 10，82 类）",
            "feature": str(dataset["feature"]["profile"]) + "（D=%d）" % int(dataset["feature"]["dim"]),
            "split": "20% 按比例留出 / question_id 整组 / 单 seed 42",
            "seed": int(DEFAULT_SEED),
            "n_classes": n_classes,
            "n_test_questions": int(len(split.test_questions)),
            "n_train_questions": int(len(split.train_questions)),
        },
        "metrics": dict(metrics),
        "strata": strata,
        "per_class": per_class,
        "model_after_training": {
            "temperature": tau,
            "frobenius_norm_M_minus_I": m_disp,
            "n_parameters": int(model.M.numel() + model.log_temperature.numel()),
        },
        "predictions_sha256": _sha256_text(_canonical_json(preds)),
        "rule": (
            "主判据口径 = 问题级：micro = 预测类 ∈ 该问题金标集合的比例；macro = **有测试问题的类**"
            "的 recall 算术平均（分母 = 77）。与轮 9 原型基线**同一批测试问题、同一分母**，"
            "因此可直接比较；**不得**用行级口径或任何代理量替代。"
        ),
    }


def class_delta_document(
    model_predictions: Sequence[int],
    dataset: Mapping[str, Any],
) -> Dict[str, Any]:
    """**逐类 delta**（model − prototype）与增益分解（哪些类提升、哪些下降）。"""
    corpus = dataset["corpus"]
    split = dataset["split"]
    gold = dataset["gold"]
    n_classes = int(corpus.n_classes)
    model_stats = per_class_stats(model_predictions, split.test_questions, gold, n_classes)
    base_stats = per_class_stats(
        dataset["baseline"]["predictions"], split.test_questions, gold, n_classes
    )
    rows: List[Dict[str, Any]] = []
    for cls in range(n_classes):
        total = int(model_stats["total"][cls])
        if total <= 0:
            continue
        model_recall = float(model_stats["hit"][cls]) / float(total)
        base_recall = float(base_stats["hit"][cls]) / float(total)
        rows.append(
            {
                "class_index": int(cls),
                "class_key": str(corpus.answer_keys[cls]),
                "test_questions": total,
                "prototype_recall": base_recall,
                "model_recall": model_recall,
                "delta": float(model_recall - base_recall),
            }
        )
    improved = [r for r in rows if float(r["delta"]) > 0.0]
    degraded = [r for r in rows if float(r["delta"]) < 0.0]
    unchanged = [r for r in rows if float(r["delta"]) == 0.0]
    by_delta = sorted(rows, key=lambda r: (-float(r["delta"]), int(r["class_index"])))
    return {
        "version": str(QA_LEARN_D_VERSION),
        "n_classes_with_test_questions": int(len(rows)),
        "summary": {
            "n_improved": int(len(improved)),
            "n_degraded": int(len(degraded)),
            "n_unchanged": int(len(unchanged)),
            "sum_positive_delta": float(sum(float(r["delta"]) for r in improved)),
            "sum_negative_delta": float(sum(float(r["delta"]) for r in degraded)),
            "n_newly_learned": int(
                len([r for r in rows if float(r["prototype_recall"]) == 0.0 and float(r["model_recall"]) > 0.0])
            ),
            "n_lost": int(
                len([r for r in rows if float(r["prototype_recall"]) > 0.0 and float(r["model_recall"]) == 0.0])
            ),
            "mean_delta": float(sum(float(r["delta"]) for r in rows) / len(rows)),
        },
        "top_gains": by_delta[:20],
        "top_losses": by_delta[-20:][::-1],
        "rows": rows,
        "rule": (
            "delta = 本模型 recall − 轮 9 冻结原型 recall（同一类、同一批测试问题）。"
            "按类等权汇总（mean_delta）与 macro 的增量同向但不等价（macro 分母 77）；"
            "`n_newly_learned` = 原型零召回而模型学到的类数，`n_lost` = 反之。"
        ),
    }


def gate_document(
    eval_doc: Mapping[str, Any],
    class_delta: Mapping[str, Any],
) -> Dict[str, Any]:
    """**双判据**：``macro > 0.241276`` **且** ``micro >= 0.183312``（两条同时满足才过门）。"""
    macro = float(eval_doc["metrics"]["macro"])
    micro = float(eval_doc["metrics"]["micro"])
    passed_macro = bool(macro > MACRO_CRITERION)
    passed_micro = bool(micro >= MICRO_CRITERION)
    passed = bool(passed_macro and passed_micro)
    summary = class_delta["summary"]
    common = {
        "version": str(QA_LEARN_D_VERSION),
        "criterion": {
            "macro": {"operator": ">", "threshold": MACRO_CRITERION},
            "micro": {"operator": ">=", "threshold": MICRO_CRITERION},
            "both_required": True,
            "source": "轮 11 任务书 3b；阈值字面 = 基线值四舍五入到 6 位（macro 原型 / micro 多数类）",
        },
        "observed": {
            "macro": macro,
            "micro": micro,
            "baseline_prototype_macro": FROZEN_PROTOTYPE_MACRO,
            "baseline_prototype_micro": FROZEN_PROTOTYPE_MICRO,
            "baseline_majority_micro": FROZEN_MAJORITY_MICRO,
            "delta_macro_vs_prototype": float(macro - FROZEN_PROTOTYPE_MACRO),
            "delta_micro_vs_prototype": float(micro - FROZEN_PROTOTYPE_MICRO),
            "delta_micro_vs_majority": float(micro - FROZEN_MAJORITY_MICRO),
        },
        "passed_macro": passed_macro,
        "passed_micro": passed_micro,
        "passed": passed,
        "gain_decomposition": dict(summary),
        "proxy_rule": (
            "**禁止代理量**：gap/sigma、对齐度、有效秩等代理量在历史上两次被证否，"
            "不得替代本双判据；本判定只用测试侧 macro 与 micro。"
        ),
        "not_claimed": [
            "已学会 / 模型能力（单 seed、单特征档、单留出比例）",
            "跨 seed / 跨留出比例稳定性",
            "N3D 后端的可学上限（本轮的可学变换是 D->D 线性映射，不是 N3D 骨干）",
        ],
    }
    if passed:
        verdict = (
            "过门：双判据同时满足 —— macro = %.6f > %.6f 且 micro = %.6f >= %.6f"
            "（Δ vs 原型 macro %+.6f、Δ vs 原型 micro %+.6f）⇒ 在 A 口径（82 类 / "
            "lexical-88 / 单 seed 42）上，可学 D->D 变换**带来真实增益**（%d 类提升 / %d 类下降 / "
            "%d 类不变；新学到 %d 个原型零召回类，丢失 %d 个）。**仍不等于「已学会」**："
            "判据只是「超过零训练下限且 micro 不低于多数类」，绝对水平仍低，且单 seed / 单特征档。"
            % (
                macro,
                MACRO_CRITERION,
                micro,
                MICRO_CRITERION,
                float(macro - FROZEN_PROTOTYPE_MACRO),
                float(micro - FROZEN_PROTOTYPE_MICRO),
                int(summary["n_improved"]),
                int(summary["n_degraded"]),
                int(summary["n_unchanged"]),
                int(summary["n_newly_learned"]),
                int(summary["n_lost"]),
            )
        )
    else:
        verdict = (
            "未过门：**按双判据模型未带来增益**（macro = %.6f vs 阈值 %.6f（%s）；"
            "micro = %.6f vs 阈值 %.6f（%s）；Δ vs 原型 macro = %+.6f、Δ vs 原型 micro = %+.6f）。"
            "判断：同一批 lexical-88 词面特征上，零训练原型与可学 D->D 变换都达不到双判据"
            " ⇒ 现状更像**特征瓶颈**（词面哈希特征的类间可分性不足），而不是「把模型换大就能"
            "解决」。建议：换 **bge-m3 语义档（D=1024）**按同一口径（同 A 口径 / 同切分 / "
            "同 seed / 同判据）重测，再判断是否仍为模型瓶颈。**不得**用 gap/sigma、对齐度等"
            "代理量替代本判据，也不得把未过门包装成过门。"
            % (
                macro,
                MACRO_CRITERION,
                "达到" if passed_macro else "未达到",
                micro,
                MICRO_CRITERION,
                "达到" if passed_micro else "未达到",
                float(macro - FROZEN_PROTOTYPE_MACRO),
                float(micro - FROZEN_PROTOTYPE_MICRO),
            )
        )
    common["verdict"] = verdict
    return common


# ---------------------------------------------------------------------------
# 实验编排（基线冻结 -> step0 等价 -> 训练 -> 评测 -> 双判据）
# ---------------------------------------------------------------------------


def baseline_document(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """**冻结基线**文档：constant_majority / prototype_centroid / 分层四档 delta。"""
    import n3d_qa_learn.qa_scoped as S

    corpus = dataset["corpus"]
    split = dataset["split"]
    proto = dataset["baseline"]["metrics"]
    majority = dataset["majority"]
    strata = S.stratified_sensitivity(
        prototype_predictions=dataset["baseline"]["predictions"],
        majority_predictions=majority["predictions"],
        questions=split.test_questions,
        gold_local=dataset["gold"],
        n_classes=int(corpus.n_classes),
    )
    return {
        "version": str(QA_LEARN_D_VERSION),
        "caliber": {
            "name": "A（每类问题数 >= 10）",
            "n_classes": int(corpus.n_classes),
            "n_questions": int(corpus.n_questions),
            "n_pairs": int(corpus.n_pairs),
            "n_train_questions": int(len(split.train_questions)),
            "n_test_questions": int(len(split.test_questions)),
            "n_train_rows": int(len(split.train_rows)),
            "n_test_rows": int(len(split.test_rows)),
            "feature": str(dataset["feature"]["profile"]) + "（D=%d）" % int(dataset["feature"]["dim"]),
            "seed": int(DEFAULT_SEED),
            "holdout_pct": int(split.holdout_pct),
            "merge_enabled": False,
        },
        "constant_majority": {
            "macro": float(majority["metrics"]["macro"]),
            "micro": float(majority["metrics"]["micro"]),
            "class_index": int(majority["class_index"]),
            "class_key": str(majority["class_key"]),
            "class_train_question_count": int(majority["class_train_question_count"]),
        },
        "prototype_centroid": {
            "macro": float(proto["macro"]),
            "micro": float(proto["micro"]),
            "macro_all_classes": float(proto["macro_all_classes"]),
            "n_classes_with_test_questions": int(proto["n_classes_with_test_questions"]),
            "n_zero_recall_classes": int(proto["n_zero_recall_classes"]),
            "recall_distribution": dict(proto["recall_distribution"]),
        },
        "prototype_support": _quantiles(dataset["prototype_support"]),
        "strata_delta_vs_majority": {
            "thresholds": list(strata["thresholds"]),
            "rows": [dict(r) for r in strata["rows"]],
            "official": dict(strata["official"]),
            "statement": str(strata["statement"]),
            "not_gate_redefinition": True,
        },
        "frozen_check": {
            "prototype_macro_expected": FROZEN_PROTOTYPE_MACRO,
            "prototype_macro_observed": float(proto["macro"]),
            "prototype_micro_expected": FROZEN_PROTOTYPE_MICRO,
            "prototype_micro_observed": float(proto["micro"]),
            "majority_macro_expected": FROZEN_MAJORITY_MACRO,
            "majority_macro_observed": float(majority["metrics"]["macro"]),
            "majority_micro_expected": FROZEN_MAJORITY_MICRO,
            "majority_micro_observed": float(majority["metrics"]["micro"]),
            "all_exactly_equal": True,
            "rule": (
                "本轮**不重新调参**：切分 / 类集 / 特征档 / 原型口径全部沿用轮 9/10；"
                "上面的 observed 与 expected 逐位相等（不等则本轮直接拒绝出数）。"
            ),
        },
        "predictions_sha256": _sha256_text(
            _canonical_json([int(x) for x in dataset["baseline"]["predictions"]])
        ),
        "matrix_persisted": False,
    }


def config_document(
    dataset: Mapping[str, Any],
    steps: int,
    lr: float,
    seed: int,
    trained: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """训练配置（**预注册**：在看到测试集结果之前固定；不含挂钟字段）。

    ``trained`` 给出后追加 ``steps_run`` / ``stopped_reason`` 等**实际执行**信息（停止准则
    只看训练侧损失，不看测试集）。
    """
    import numpy as np

    corpus = dataset["corpus"]
    training: Dict[str, Any] = {
        "max_steps": int(steps),
        "lr": float(lr),
        "optimizer": "Adam",
        "batch": "full",
        "convergence": {
            "window": int(CONVERGE_WINDOW),
            "rel_tol": float(CONVERGE_REL_TOL),
            "rule": (
                "训练损失每 %d 步的相对变化 < %.3f 即停（上限 %d 步）；"
                "**只用训练侧损失**，不看测试集" % (int(CONVERGE_WINDOW), float(CONVERGE_REL_TOL), int(steps))
            ),
        },
    }
    if trained is not None:
        training.update(
            {
                "steps_run": int(trained["steps_run"]),
                "stopped_reason": str(trained["stopped_reason"]),
                "last_window_rel_change": float(trained["last_window_rel_change"]),
            }
        )
    return {
        "version": str(QA_LEARN_D_VERSION),
        "pre_registration": (
            "配置在看到本轮测试集结果之前固定（单配置、无按判据回调）；"
            "基线阈值来自轮 9/10 冻结值，本轮不重新调参。停止准则只用训练侧损失。"
        ),
        "seed": int(seed),
        "deterministic": {
            "algorithm": "full-batch Adam（无 shuffle / 无 dropout）",
            "torch_deterministic_algorithms": True,
            "num_threads": 1,
            "dtype": "float64",
        },
        "model": {
            "transform": "z = normalize(M x)",
            "M_shape": [int(dataset["feature"]["dim"]), int(dataset["feature"]["dim"])],
            "M_init": "identity（逐位）",
            "M_learnable": True,
            "temperature": "tau = exp(clamp(log_temperature, -8, 8))，初值 1.0，可学",
            "centroids": "冻结：训练侧问题特征均值再 L2 归一化",
            "score": "tau * <z, c_k>",
            "loss": "multi-positive CE（逐金标 -log p 的均值，再按问题平均）",
            "n_parameters": int(dataset["feature"]["dim"]) ** 2 + 1,
        },
        "training": training,
        "dataset": {
            "product_dir": str(DEFAULT_PRODUCT_DIR),
            "feature_profile": str(dataset["feature"]["profile"]),
            "dim": int(dataset["feature"]["dim"]),
            "feature_fingerprint": str(dataset["feature"]["fingerprint"]),
            "n_classes": int(corpus.n_classes),
            "n_train_questions": int(len(dataset["split"].train_questions)),
            "n_test_questions": int(len(dataset["split"].test_questions)),
            "n_train_rows": int(len(dataset["split"].train_rows)),
            "n_test_rows": int(len(dataset["split"].test_rows)),
        },
        "criteria": {
            "macro": {"operator": ">", "threshold": MACRO_CRITERION},
            "micro": {"operator": ">=", "threshold": MICRO_CRITERION},
            "both_required": True,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
        },
    }


def run_experiment(
    product_dir: str = DEFAULT_PRODUCT_DIR,
    profile: str = DEFAULT_PROFILE,
    *,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    seed: int = DEFAULT_SEED,
    log: Any = None,
) -> Dict[str, Any]:
    """完整实验：复算并冻结基线 → step0 等价证明 → 训练 → 评测 → 逐类 delta → 双判据。

    返回
    ----
    Dict[str, Any]
        ``baseline`` / ``config`` / ``step0`` / ``curve`` / ``eval`` / ``class_delta`` /
        ``gate`` 七份可直接序列化的文档（由 CLI 落盘）。
    """
    import n3d_qa_learn.qa_scoped as S

    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if log is not None:
        log("[run] 读取 A 口径语料并复算冻结基线（不重新调参）…")
    dataset = load_dataset(str(product_dir), str(profile))
    corpus = dataset["corpus"]
    split = dataset["split"]
    qids = dataset["qids"]
    row_of = {str(q): i for i, q in enumerate(qids)}
    train_qids = [str(q) for q in split.train_questions]
    test_qids = [str(q) for q in split.test_questions]
    x_train = _as_tensor(dataset["matrix"], [row_of[q] for q in train_qids])
    x_test = _as_tensor(dataset["matrix"], [row_of[q] for q in test_qids])
    import numpy as np

    centroids = torch.tensor(
        np.asarray(dataset["centroids"], dtype=np.float64), dtype=torch.float64
    )
    mask = _gold_mask(train_qids, dataset["gold"], int(corpus.n_classes))
    model = LearnableTransform(int(dataset["feature"]["dim"]))

    if log is not None:
        log("[run] step 0 等价性检查（M=I、tau=1 vs 轮 9 原型基线）…")
    step0 = step0_equivalence(model, x_test, centroids, dataset)

    def _train_diagnostics(step: int) -> Dict[str, Any]:
        with torch.no_grad():
            logits = model.logits(x_train, centroids)
            preds = _predict(logits)
            loss_value = float(multi_positive_ce(logits, mask).item())
        metrics = S.question_level_metrics(
            predictions=preds,
            questions=split.train_questions,
            gold_local=dataset["gold"],
            n_classes=int(corpus.n_classes),
        )
        return {
            "step": int(step),
            "train_loss": loss_value,
            "train_macro": float(metrics["macro"]),
            "train_micro": float(metrics["micro"]),
            "note": "训练侧指标**仅诊断**，不参与判据",
        }

    if log is not None:
        log(
            "[run] 训练（full-batch Adam, lr=%g, float64, 单线程；上限 %d 步，收敛准则 %d 步相对变化 < %.3f）…"
            % (float(lr), int(steps), int(CONVERGE_WINDOW), float(CONVERGE_REL_TOL))
        )
    trained = train_transform(
        model,
        x_train,
        mask,
        centroids,
        steps=int(steps),
        lr=float(lr),
        diag_every=int(DIAG_EVERY),
        train_diagnostics=_train_diagnostics,
        converge_window=int(CONVERGE_WINDOW),
        converge_rel_tol=float(CONVERGE_REL_TOL),
    )
    if not all(isinstance(x, float) and x == x and abs(x) != float("inf") for x in trained["losses"]):
        raise AssertionError("训练 loss 出现 NaN/Inf，拒绝出数")
    if log is not None:
        log(
            "[run] 训练结束：steps_run=%d（%s），最后 50 步相对变化 %.4f%%"
            % (
                int(trained["steps_run"]),
                str(trained["stopped_reason"]),
                float(trained["last_window_rel_change"]) * 100.0,
            )
        )
        log("[run] 开始评测（测试侧 macro / micro / 分层 / 每类 recall）…")
    eval_doc = evaluate_model(model, x_test, centroids, dataset)
    class_delta = class_delta_document(
        [int(x) for x in _eval_predictions(model, x_test, centroids)], dataset
    )
    gate = gate_document(eval_doc, class_delta)
    baseline = baseline_document(dataset)
    config = config_document(dataset, int(steps), float(lr), int(seed), trained)
    curve = {
        "version": str(QA_LEARN_D_VERSION),
        "steps_max": int(steps),
        "steps_run": int(trained["steps_run"]),
        "stopped_reason": str(trained["stopped_reason"]),
        "last_window_rel_change": float(trained["last_window_rel_change"]),
        "converge_window": int(trained["converge_window"]),
        "converge_rel_tol": float(trained["converge_rel_tol"]),
        "lr": float(lr),
        "losses": [float(x) for x in trained["losses"]],
        "loss_first": float(trained["losses"][0]),
        "loss_last": float(trained["losses"][-1]),
        "loss_min": float(min(trained["losses"])),
        "loss_finite": True,
        "diagnostics": [dict(d) for d in trained["diagnostics"]],
        "rule": (
            "train loss 曲线与训练侧 macro 均为**诊断**：训练侧 macro 与测试侧 macro 分母不同"
            "（训练侧分母 = 有训练问题的类数），**不得**用于判据或替代测试侧指标。"
            "停止准则只看训练侧损失（相对变化 < rel_tol），不看测试集。"
        ),
    }
    if log is not None:
        log(
            "[run] 测试侧：macro=%.6f / micro=%.6f；双判据 passed=%s（macro %s / micro %s）"
            % (
                float(eval_doc["metrics"]["macro"]),
                float(eval_doc["metrics"]["micro"]),
                str(gate["passed"]),
                str(gate["passed_macro"]),
                str(gate["passed_micro"]),
            )
        )
        log("[run] step0 等价：n_argmax_diff=%d，macro/micro 完全相等=%s/%s"
            % (
                int(step0["n_argmax_diff"]),
                str(step0["macro_exactly_equal"]),
                str(step0["micro_exactly_equal"]),
            ))
    return {
        "baseline": baseline,
        "config": config,
        "step0": step0,
        "curve": curve,
        "eval": eval_doc,
        "class_delta": class_delta,
        "gate": gate,
    }


def _eval_predictions(
    model: LearnableTransform, x_test: torch.Tensor, centroids: torch.Tensor
) -> List[int]:
    """训练后测试侧 argmax 预测（供逐类 delta 复用；与 :func:`evaluate_model` 同一次前向口径）。"""
    with torch.no_grad():
        return _predict(model.logits(x_test, centroids))


__all__ = [
    "QA_LEARN_D_VERSION",
    "DEFAULT_PRODUCT_DIR",
    "VERIFY_ROOT",
    "DEFAULT_OUT_DIR",
    "DEFAULT_SEED",
    "DEFAULT_PROFILE",
    "DEFAULT_STEPS",
    "DEFAULT_LR",
    "PROBE_STEPS",
    "DIAG_EVERY",
    "FROZEN_PROTOTYPE_MACRO",
    "FROZEN_PROTOTYPE_MICRO",
    "FROZEN_MAJORITY_MACRO",
    "FROZEN_MAJORITY_MICRO",
    "MACRO_CRITERION",
    "MICRO_CRITERION",
    "STRATIFICATION_THRESHOLDS",
    "LearnableTransform",
    "baseline_document",
    "class_delta_document",
    "config_document",
    "evaluate_model",
    "gate_document",
    "load_dataset",
    "multi_positive_ce",
    "per_class_stats",
    "run_experiment",
    "step0_equivalence",
    "stratified_rows",
    "train_transform",
]