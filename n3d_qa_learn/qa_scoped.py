"""轮 9：**类集收缩**（A 口径：``>= min_class_questions`` 题/类）+ **零训练可学性下限**。

为什么做这一轮
--------------
轮 8（答案表同义归并）已把 298 类收到 281 类，但**每类训练问题中位数仍 5.0、测试零样本类
仍 51** ⇒ 结构门未过：归并只消标签噪声、**不增加样本总量**。真实分布里有一批类本来就够厚：
``>= 10 题/类`` 的 82 个类覆盖 **12374 个「问题-类」关联**、每类中位 **21** 题 —— 把类集
收缩到这个子集，才是「QA 侧样本量够不够学」的现实前提。

本轮口径（**必须整段引用，不得只引数字**）
------------------------------------------
* **A 口径 = 仅筛选类集，归并关闭**。轮 8 的答案归并有两个 high-risk 簇语义有误
  （``china(Peking)`` 并进 ``china``；``america(US)`` / ``america(USA)`` 并进有歧义的
  ``america``）——**在 high-risk 未解决前不得启用归并**，本模块也不接受 ``merge_map``。
* 类集门槛：**每类的「问题数」**（该类的不同 ``question_id`` 个数，一问只计一次）``>=
  min_class_questions``；**不是**正样本行数、也不是冻结表 ``count``。
* 门槛用**筛选前**（全量类集、全部问题）的计数判定；随后按轮 1 的规则，只有
  **金标全部落在保留类内**的问题才进语料（含越界金标的问题**整体**排除 —— 不把
  「答案不在表内」偷偷变成「答案在表内」）。因此语料内每类问题数（post-scope）会
  略小于门槛计数，两个数必须**分开报**。
* 切分沿用轮 1：20% 按比例留出、``question_id`` 整组、单 seed 42、边界归属 = 严格小于。

零训练「可学性下限」（**不训练任何模型**）
------------------------------------------
* **原型质心分类器**：每类的原型 = 该类**训练侧**问题特征的质心（L2 归一化）；测试侧问题按
  余弦相似度取 argmax 预测（平票取最小类下标，确定性）。
* 口径：**问题级**（不是行级）。``micro`` = 问题级命中率（预测类 ∈ 该问题金标集合）；
  某类的 ``recall`` = 「金标含该类」的测试问题中预测恰为该类的比例；``macro`` = 有测试
  问题的类的 recall 算术平均（分母 = **有测试问题的类数**），另给 ``macro_all_classes``
  （分母 = 全部类数，零测试类按 0 计入）。
* 对照基线用**同一口径**：``constant_majority`` = 一律预测训练侧问题数最多的类。
* **决策门**：``原型 macro - 多数类 macro >= +10pp`` 才算过门。不过门就如实写
  「特征空间不可分」并给后续建议（换 bge-m3 语义档 / 继续上游放量），**不得粉饰**。

产物与纪律
----------
* 产物只写 ``checkpoints/qa_learn/_verify/qa_scoped/``（不在 ``_verify`` 别处落盘）；
* **不产 zip、不落盘裸特征矩阵**（特征矩阵只在内存里，产物中只有 ``persisted=False`` 的声明）、
  **无挂钟字段** ⇒ 同参双跑逐字节一致；
* 上游 ``n3d_qa`` 冻结产物**只读**；本模块**不做任何训练**（只算质心与余弦）。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 冻结常量（口径进产物）
# ---------------------------------------------------------------------------

#: 模块版本（结构变化必须递增；进产物）。
QA_SCOPED_VERSION: str = "n3dqa-qa-scoped-v1"

#: 冻结产物目录（**只读**）。
DEFAULT_PRODUCT_DIR: str = "checkpoints/qa_learn/dataset"

#: 统一答案表 / 问答对文件名（与 :mod:`n3d_qa_learn.qa_scale` 同源）。
ANSWER_TABLE_FILE: str = "answer_table.jsonl"
ALL_PAIRS_FILE: str = "n3d_qa_all_pairs.jsonl"

#: 产物目录（**唯一**允许写盘的根；轮 9 · W3 的收窄口径同样适用）。
VERIFY_ROOT: str = os.path.join("checkpoints", "qa_learn", "_verify")
DEFAULT_OUT_DIR: str = os.path.join(VERIFY_ROOT, "qa_scoped")

#: A 口径的默认类集门槛（题/类）与切分口径（与轮 1 逐项一致）。
DEFAULT_MIN_CLASS_QUESTIONS: int = 10
DEFAULT_HOLDOUT_PCT: int = 20
DEFAULT_SEED: int = 42

#: 原型基线的特征档（沿用现行词面档，与 ``entry_table.PROFILE_LEXICAL`` 逐字一致）。
DEFAULT_PROFILE: str = "lexical-88"

#: 特征角色（QA 题面）。
ROLE_QUESTION: str = "question"

#: 决策门阈值：原型 macro 需比多数类基线高出至少这么多（+10pp）。
GATE_MARGIN: float = 0.10

#: 决策门口径文本（**唯一来源**；产物与文档同源）。
GATE_RULE: str = (
    "决策门（**只判零训练下的可学性下限**）：原型质心分类器的**问题级 macro** 必须比"
    "同一口径的多数类基线（constant_majority）高出 >= %.2f（+%.0fpp）才算过门。"
    "不过门 ⇒ 在当前特征档下「82 类闭集分类」的可学前提**未建立**，须如实登记并给后续建议；"
    "过门也**只**说明「零训练下限成立」，**不得**宣称「已学会」。"
    % (GATE_MARGIN, GATE_MARGIN * 100.0)
)

#: A 口径与归并的关系声明（**进产物**）。
CALIBER_STATEMENT: str = (
    "A 口径 = 仅按「每类问题数 >= %d」筛选类集，**归并关闭**。轮 8 的答案归并有 2 个 "
    "high-risk 簇语义有误（china(Peking) -> china；america(US)/(USA) -> 有歧义的 america），"
    "**在 high-risk 未解决前不得启用归并**；本模块不接受 merge_map，也不与轮 8 的 281 类口径"
    "混用结论。" % DEFAULT_MIN_CLASS_QUESTIONS
)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _sha256_file(path: str) -> str:
    """文件 SHA256（分块读）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(obj: Any) -> str:
    """规范化 JSON（跨运行一致）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _quantiles(values: Sequence[float]) -> Dict[str, Any]:
    """一组数的 min/p25/median/p75/max/mean（空序列返回 0）。"""
    import numpy as np

    arr = np.asarray([float(x) for x in values], dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
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
# 题面文本与特征（**只读**冻结产物；特征矩阵只在内存里）
# ---------------------------------------------------------------------------


def load_question_texts(product_dir: str = DEFAULT_PRODUCT_DIR) -> Dict[str, Any]:
    """读取 ``question_id -> question_text``（只读；逐行核对同一问题的文本一致）。

    参数
    ----
    product_dir : str
        冻结产物目录。

    返回
    ----
    Dict[str, Any]
        ``texts``（dict）、``n_rows``、``n_questions``、``inconsistent_questions``（同一
        ``question_id`` 出现不同文本的 id 列表，空表示一致）、``sha256``。

    异常
    ------
    FileNotFoundError
        缺 ``n3d_qa_all_pairs.jsonl``。
    ValueError
        存在同一问题的多种文本（**拒绝**在文本不一致的数据上做原型质心）。
    """
    path = os.path.join(str(product_dir), ALL_PAIRS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError("缺少问答对文件：" + repr(path))
    texts: Dict[str, str] = {}
    inconsistent: List[str] = []
    n_rows = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            obj = json.loads(line)
            qid = str(obj["question_id"])
            text = str(obj["question_text"])
            prev = texts.get(qid)
            if prev is None:
                texts[qid] = text
            elif str(prev) != text and qid not in inconsistent:
                inconsistent.append(qid)
    if inconsistent:
        raise ValueError(
            "同一 question_id 出现多种问题文本（前几个 = "
            + str(inconsistent[:5])
            + "）；拒绝在文本不一致的数据上做原型质心"
        )
    return {
        "texts": texts,
        "n_rows": int(n_rows),
        "n_questions": int(len(texts)),
        "inconsistent_questions": [],
        "sha256": _sha256_file(path),
        "rule": (
            "同一 question_id 的全部行必须共享同一 question_text（现场核对，不一致即抛）；"
            "只读冻结产物，不写任何文件。"
        ),
    }


def build_question_matrix(
    texts_by_qid: Mapping[str, str],
    qids: Sequence[str],
    *,
    profile: str = DEFAULT_PROFILE,
    role: str = ROLE_QUESTION,
) -> Dict[str, Any]:
    """把一批问题编码成 ``float32[N, D]``（**L2 归一化由向量化器保证**，零向量保持零）。

    参数
    ----
    texts_by_qid : Mapping[str, str]
        问题 id -> 文本。
    qids : Sequence[str]
        要编码的问题 id（**顺序即矩阵行序**，调用方负责保持）。
    profile : str
        特征档（默认 ``lexical-88``；档注册表在 :mod:`n3d_qa_learn.entry_table`）。
    role : str
        编码角色（``question``）。

    返回
    ----
    Dict[str, Any]
        ``matrix``（``numpy.ndarray``，**只在内存**）、``dim``、``fingerprint``、
        ``profile``、``role``、``n_texts``、``n_zero_norm``（零范数行数）、
        ``persisted``（恒 False：不落盘裸特征矩阵）。

    异常
    ------
    KeyError
        某个问题 id 没有文本。
    """
    import numpy as np

    import n3d_qa_learn.entry_table as ET

    vectorizer = ET.build_vectorizer_for(str(profile), str(role))
    missing = [str(q) for q in qids if str(q) not in texts_by_qid]
    if missing:
        raise KeyError(
            "以下问题 id 在冻结问答对里没有文本（前几个 = " + str(missing[:5]) + "）"
        )
    texts = [str(texts_by_qid[str(q)]) for q in qids]
    matrix = np.asarray(vectorizer.encode_matrix(texts), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    return {
        "matrix": matrix,
        "dim": int(matrix.shape[1]),
        "profile": str(profile),
        "role": str(role),
        "fingerprint": str(vectorizer.fingerprint()),
        "n_texts": int(matrix.shape[0]),
        "n_zero_norm": int((norms <= 0.0).sum()),
        "persisted": False,
        "rule": (
            "**不落盘裸特征矩阵**：矩阵只在内存里参与质心与余弦计算，产物中只登记 "
            "dim / 档名 / 口径指纹 / 零范数行数。"
        ),
    }


# ---------------------------------------------------------------------------
# 原型质心 + 余弦 argmax（零训练）
# ---------------------------------------------------------------------------


def class_prototypes(
    matrix: Any,
    qids: Sequence[str],
    gold_local: Mapping[str, Sequence[int]],
    train_questions: Sequence[str],
    *,
    n_classes: int,
) -> Dict[str, Any]:
    """每类的**训练侧**问题特征质心（L2 归一化后作为原型）。

    参数
    ----
    matrix : numpy.ndarray
        ``[N, D]``（行序 = ``qids``）。
    qids : Sequence[str]
        行序对应的问题 id。
    gold_local : Mapping[str, Sequence[int]]
        问题 id -> 本语料类下标集合（金标）。
    train_questions : Sequence[str]
        训练侧问题 id（只有它们参与质心）。
    n_classes : int
        类数 ``C``。

    返回
    ----
    Dict[str, Any]
        ``centroids``（``[C, D]`` float32）、``counts``（每类参与质心的训练问题数）、
        ``n_classes_without_train_questions``、``n_train_questions_used``、
        ``n_zero_norm_centroids``。

    异常
    ------
    ValueError
        训练侧为空、或某类质心为零向量（无训练问题 / 训练问题全为零向量）。
    """
    import numpy as np

    index = {str(q): i for i, q in enumerate(qids)}
    sums = np.zeros((int(n_classes), int(matrix.shape[1])), dtype=np.float64)
    counts = [0] * int(n_classes)
    used = 0
    for qid in train_questions:
        row = index.get(str(qid))
        if row is None:
            raise KeyError("训练问题 " + repr(qid) + " 不在特征矩阵的行序里")
        vec = matrix[row].astype(np.float64)
        for cls in gold_local[str(qid)]:
            sums[int(cls)] += vec
            counts[int(cls)] += 1
        used += 1
    if used == 0:
        raise ValueError("训练侧问题为空，无法构造原型质心")
    centroids = np.zeros_like(sums, dtype=np.float64)
    empty: List[int] = []
    for cls in range(int(n_classes)):
        if counts[cls] <= 0:
            empty.append(cls)
            continue
        mean = sums[cls] / float(counts[cls])
        norm = float(np.linalg.norm(mean))
        if norm > 0.0:
            centroids[cls] = mean / norm
    zero_norm = [int(c) for c in range(int(n_classes)) if float(np.linalg.norm(centroids[c])) <= 0.0]
    if empty:
        raise ValueError(
            "有 %d 个类在训练侧没有问题，无法构造原型（前几个 = %s）；"
            "A 口径要求每类都有训练样本，拒绝继续" % (len(empty), str(empty[:5]))
        )
    if zero_norm:
        raise ValueError(
            "有 %d 个类的质心为零向量（前几个 = %s）；零原型无法参与余弦 argmax，拒绝继续"
            % (len(zero_norm), str(zero_norm[:5]))
        )
    return {
        "centroids": centroids.astype(np.float32),
        "counts": [int(x) for x in counts],
        "n_classes_without_train_questions": 0,
        "n_train_questions_used": int(used),
        "n_zero_norm_centroids": 0,
        "rule": (
            "原型 = 该类**训练侧**问题特征的算术平均再 L2 归一化（零范数即抛）；"
            "参数只为原型，不参与任何训练过程。"
        ),
    }


def predict_by_cosine(
    matrix: Any,
    qids: Sequence[str],
    questions: Sequence[str],
    centroids: Any,
) -> Dict[str, Any]:
    """按余弦相似度取 argmax 预测（**零训练**；平票取最小类下标，确定性）。

    参数
    ----
    matrix : numpy.ndarray
        ``[N, D]``。
    qids : Sequence[str]
        行序。
    questions : Sequence[str]
        要预测的问题 id（测试侧）。
    centroids : numpy.ndarray
        ``[C, D]``（已 L2 归一化）。

    返回
    ----
    Dict[str, Any]
        ``predictions``（长度 = ``len(questions)`` 的类下标列表）、``top1_cosine``、
        ``margin``（top1 - top2）、``n_ties``、``tie_rule``。
    """
    import numpy as np

    index = {str(q): i for i, q in enumerate(qids)}
    rows = []
    for qid in questions:
        row = index.get(str(qid))
        if row is None:
            raise KeyError("测试问题 " + repr(qid) + " 不在特征矩阵的行序里")
        rows.append(int(row))
    if not rows:
        raise ValueError("测试侧问题为空，无法预测")
    sub = np.asarray([matrix[int(r)] for r in rows], dtype=np.float32)
    sims = sub.astype(np.float64) @ np.asarray(centroids, dtype=np.float64).T
    pred = np.argmax(sims, axis=1)
    top1 = sims[np.arange(sims.shape[0]), pred]
    if sims.shape[1] >= 2:
        masked = sims.copy()
        masked[np.arange(sims.shape[0]), pred] = -np.inf
        top2 = masked.max(axis=1)
    else:  # pragma: no cover - C >= 2 由调用方保证
        top2 = top1
    # 平票检测：若 top1 与 top2 完全相等则 argmax 的选择依赖下标顺序（登记下来）
    ties = int(np.sum(np.isclose(top1, top2, rtol=0.0, atol=1e-12)))
    return {
        "predictions": [int(x) for x in pred.tolist()],
        "top1_cosine": [float(x) for x in top1.tolist()],
        "margin": [float(x) for x in (top1 - top2).tolist()],
        "n_ties": int(ties),
        "tie_rule": "余弦完全相等时取**最小类下标**（numpy argmax 语义，确定性）",
    }


# ---------------------------------------------------------------------------
# 问题级指标（micro / macro / 每类 recall）
# ---------------------------------------------------------------------------


def question_level_metrics(
    *,
    predictions: Sequence[int],
    questions: Sequence[str],
    gold_local: Mapping[str, Sequence[int]],
    n_classes: int,
) -> Dict[str, Any]:
    """问题级 micro / macro / 每类 recall 分布（**与预测器无关的纯函数**）。

    参数
    ----
    predictions : Sequence[int]
        与 ``questions`` 同序的预测类下标。
    questions : Sequence[str]
        被测问题（测试侧）。
    gold_local : Mapping[str, Sequence[int]]
        问题 id -> 本语料类下标集合。
    n_classes : int
        类数。

    返回
    ----
    Dict[str, Any]
        ``n_questions`` / ``n_hit`` / ``micro`` / ``macro``（分母 = **有测试问题的类数**）/
        ``macro_all_classes``（分母 = 全部类数）/ ``recall_distribution`` /
        ``n_classes_with_test_questions`` / ``n_zero_recall_classes`` /
        ``zero_recall_classes_head`` / ``n_multi_gold_questions`` / ``rule``。

    异常
    ------
    ValueError
        长度不一致或 ``questions`` 为空。
    """
    qs = [str(q) for q in questions]
    preds = [int(x) for x in predictions]
    if len(qs) != len(preds):
        raise ValueError(
            "预测条数 %d 与问题条数 %d 不一致" % (len(preds), len(qs))
        )
    if not qs:
        raise ValueError("测试问题为空，指标无定义")
    per_class_tot = [0] * int(n_classes)
    per_class_hit = [0] * int(n_classes)
    hit = 0
    multi = 0
    for qid, pred in zip(qs, preds):
        gold = sorted({int(x) for x in gold_local[qid]})
        if len(gold) > 1:
            multi += 1
        if pred in gold:
            hit += 1
        for cls in gold:
            per_class_tot[int(cls)] += 1
            if int(pred) == int(cls):
                per_class_hit[int(cls)] += 1
    with_test = [c for c in range(int(n_classes)) if per_class_tot[c] > 0]
    recalls = [float(per_class_hit[c]) / float(per_class_tot[c]) for c in with_test]
    macro = float(sum(recalls) / len(recalls)) if recalls else 0.0
    macro_all = float(sum(recalls) / float(n_classes)) if n_classes else 0.0
    zero_recall = [int(c) for c in with_test if per_class_hit[c] == 0]
    return {
        "n_questions": int(len(qs)),
        "n_hit": int(hit),
        "micro": float(hit) / float(len(qs)),
        "macro": float(macro),
        "macro_all_classes": float(macro_all),
        "n_classes": int(n_classes),
        "n_classes_with_test_questions": int(len(with_test)),
        "n_zero_recall_classes": int(len(zero_recall)),
        "zero_recall_classes_head": zero_recall[:20],
        "n_multi_gold_questions": int(multi),
        "recall_distribution": _quantiles(recalls),
        "rule": (
            "问题级口径：micro = 预测类 ∈ 该问题金标集合的比例；某类 recall = 「金标含该类」的"
            "测试问题中预测恰为该类的比例；`macro` 的分母 = **有测试问题的类数**"
            "（零测试类不进分母），`macro_all_classes` 的分母 = 全部类数（零测试类按 0 计）。"
            "多金标问题会同时计入每个金标类的分母（与类集门槛的「问题数」口径一致）。"
        ),
    }


def majority_predictions(
    *,
    train_counts: Sequence[int],
    n_test_questions: int,
) -> Dict[str, Any]:
    """多数类基线的预测（一律预测训练侧问题数最多的类；平票取最小下标）。"""
    counts = [int(x) for x in train_counts]
    if not counts:
        raise ValueError("train_counts 为空")
    best = 0
    for i, value in enumerate(counts):
        if int(value) > int(counts[best]):
            best = i
    return {
        "predictions": [int(best)] * int(n_test_questions),
        "class_index": int(best),
        "class_train_question_count": int(counts[best]),
        "predictor": "scope_side_majority_class",
        "rule": "训练侧问题数最多的类（平票取最小类下标）；与原型基线**同一问题级口径**。",
    }


# ---------------------------------------------------------------------------
# 切分不变量（现场复核）
# ---------------------------------------------------------------------------


def verify_split_invariants(corpus: Any, split: Any) -> Dict[str, Any]:
    """复核轮 1 的**三条切分不变量**（问题组级切分；违反即抛）。

    1. ``train_rows`` 与 ``test_rows`` 不相交、并集 = 全部行下标；
    2. ``train_questions`` 与 ``test_questions`` 不相交、并集 = 全部问题；
    3. **每个问题的全部行必须落在同一侧** —— 逐个问题用 ``corpus.question_rows``
       把该问题的行集合与两侧行集合求交后判定。

    **轮 9 离朱 R2 修复（第 3 条旧实现恒真）**：旧版只检查「问题 id 是否只出现在一侧
    列表里」—— 而两个列表本就是按问题 id 切成的不相交列表，这条**天然恒真**；
    于是「把某测试问题的行挪进训练侧、问题 id 列表不动」这种破坏**不会被发现**
    （离朱现场构造、旧实现返回三条全 True）。修复后改为**用行归属反查**：某问题的行若
    同时出现在两侧、或两侧都没有，都算违反。

    异常
    ------
    AssertionError
        任一条不成立（报文给出违反的问题 id / 行号前若干个）。
    """
    train_rows = [int(x) for x in split.train_rows]
    test_rows = [int(x) for x in split.test_rows]
    train_q = [str(q) for q in split.train_questions]
    test_q = [str(q) for q in split.test_questions]
    train_row_set = set(train_rows)
    test_row_set = set(test_rows)
    train_q_set = {str(q) for q in train_q}
    test_q_set = {str(q) for q in test_q}
    rows_disjoint = not (train_row_set & test_row_set)
    rows_complete = sorted(train_rows + test_rows) == list(range(int(corpus.n_pairs)))
    q_disjoint = not (set(train_q) & set(test_q))
    q_complete = sorted(train_q + test_q) == sorted(str(q) for q in corpus.question_gold.keys())
    # ③ 逐个问题反查**行归属**（不再只看 id 列表）
    violated_mixed: List[Dict[str, Any]] = []
    violated_missing: List[str] = []
    violated_side_mismatch: List[Dict[str, Any]] = []
    for qid, rows in corpus.question_rows.items():
        row_set = {int(x) for x in rows}
        in_train = row_set & train_row_set
        in_test = row_set & test_row_set
        if in_train and in_test:
            violated_mixed.append(
                {
                    "question_id": str(qid),
                    "n_rows": int(len(row_set)),
                    "n_rows_in_train": int(len(in_train)),
                    "n_rows_in_test": int(len(in_test)),
                    "rows_in_train_head": sorted(int(x) for x in in_train)[:5],
                    "rows_in_test_head": sorted(int(x) for x in in_test)[:5],
                }
            )
        elif not in_train and not in_test:
            violated_missing.append(str(qid))
        # ④ **id 侧 ↔ 行侧一致性**：这条才是「整组挪行」的照妖镜
        #    （纯挪行不破坏「组内不跨侧」，但会让 id 侧与行侧对不上）
        declared = "train" if str(qid) in train_q_set else ("test" if str(qid) in test_q_set else "none")
        actual = (
            "train"
            if (in_train and not in_test)
            else ("test" if (in_test and not in_train) else ("both" if in_train else "none"))
        )
        if declared != "none" and declared != actual:
            violated_side_mismatch.append(
                {
                    "question_id": str(qid),
                    "declared_side": str(declared),
                    "row_side": str(actual),
                    "n_rows": int(len(row_set)),
                    "n_rows_in_train": int(len(in_train)),
                    "n_rows_in_test": int(len(in_test)),
                }
            )
    whole_on_one_side = not violated_mixed and not violated_missing
    side_matches_rows = not violated_side_mismatch
    broken: List[str] = []
    if not rows_disjoint:
        broken.append("rows_disjoint")
    if not rows_complete:
        broken.append("rows_complete")
    if not q_disjoint:
        broken.append("questions_disjoint")
    if not q_complete:
        broken.append("questions_complete")
    if not whole_on_one_side:
        broken.append("every_question_wholly_on_one_side")
    if not side_matches_rows:
        broken.append("question_side_matches_row_side")
    if broken:
        raise AssertionError(
            "切分不变量被破坏："
            + str(broken)
            + "；跨侧问题前几个 = "
            + str(violated_mixed[:3])
            + "；两侧都没有行的问题前几个 = "
            + str(violated_missing[:3])
            + "；id 侧与行侧不一致的问题前几个 = "
            + str(violated_side_mismatch[:3])
            + "；拒绝在错位切分上出数"
        )
    return {
        "rows_disjoint_and_union_complete": bool(rows_disjoint and rows_complete),
        "questions_disjoint_and_union_complete": bool(q_disjoint and q_complete),
        "every_question_wholly_on_one_side": bool(whole_on_one_side),
        "question_side_matches_row_side": bool(side_matches_rows),
        "n_questions_checked_by_row_membership": int(len(corpus.question_rows)),
        "n_questions_with_rows_on_both_sides": int(len(violated_mixed)),
        "n_questions_with_no_rows_on_either_side": int(len(violated_missing)),
        "n_questions_whose_id_side_disagrees_with_rows": int(
            len(violated_side_mismatch)
        ),
        "n_train_rows": int(len(train_rows)),
        "n_test_rows": int(len(test_rows)),
        "n_train_questions": int(len(train_q)),
        "n_test_questions": int(len(test_q)),
        "seed": int(split.seed),
        "holdout_pct": int(split.holdout_pct),
        "rule": (
            "按 question_id **整组**切分（同一问题的全部行同侧）；四条不变量逐条现场复核："
            "①行不交且并集完整 ②问题不交且并集完整 ③**组内行不跨两侧** ④**问题 id 所在侧"
            "与其行所在侧一致**。③④ 都用 `corpus.question_rows` 反查行归属（③ 挡「组内被拆开」，"
            "④ 挡「整组挪行但 id 列表不动」——离朱 R2 现场构造的那种破坏）。任一不成立即抛 "
            "AssertionError（不是口头声明）。"
        ),
    }


def gold_local_map(corpus: Any) -> Dict[str, List[int]]:
    """把 ``question_gold``（可能是全量/归并后的下标空间）映射到**本语料的类下标**。"""
    out: Dict[str, List[int]] = {}
    for qid, gold in corpus.question_gold.items():
        local: List[int] = []
        for cls in gold:
            pos = corpus.class_pos.get(int(cls))
            if pos is not None:
                local.append(int(pos))
        if local:
            out[str(qid)] = sorted(set(local))
    return out


# ---------------------------------------------------------------------------
# 语料摘要（2b）与原型基线（3a/3c）+ 决策门（3b）
# ---------------------------------------------------------------------------


def build_scoped_corpus(
    product_dir: str = DEFAULT_PRODUCT_DIR,
    *,
    min_class_questions: int = DEFAULT_MIN_CLASS_QUESTIONS,
    holdout_pct: int = DEFAULT_HOLDOUT_PCT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """构建 A 口径语料 + 切分，并给出**语料摘要文档**（2b）。

    参数
    ----
    product_dir : str
        冻结产物目录（只读）。
    min_class_questions : int
        类集门槛（题/类）。``0`` 表示不过滤（**但本轮的 A 口径必须是 >= 1**；``0`` 仅供对照）。
    holdout_pct / seed : int
        按比例留出比例与 seed（与轮 1 逐项一致：20 / 42）。

    返回
    ----
    Dict[str, Any]
        ``corpus`` / ``split`` / ``document``（可直接序列化的摘要）。

    异常
    ------
    ValueError
        门槛 <= 0（本轮 A 口径要求显式门槛）或筛选后语料不可评测。
    """
    import n3d_qa_learn.qa_scale as QS

    if int(min_class_questions) <= 0:
        raise ValueError(
            "A 口径要求显式的类集门槛（min_class_questions >= 1）；当前 = "
            + str(int(min_class_questions))
        )
    corpus = QS.build_scale_corpus(
        str(product_dir), max_classes=0, min_class_questions=int(min_class_questions)
    )
    split = QS.make_scale_split(corpus, holdout_pct=int(holdout_pct), seed=int(seed))
    invariants = verify_split_invariants(corpus, split)
    class_scope = dict(corpus.source.get("class_scope", {}))
    if not class_scope.get("enabled"):
        raise ValueError(
            "语料 source 里没有 class_scope 块 ⇒ 类集筛选没有生效；拒绝把全量口径当 A 口径出数"
        )
    q_train = QS.side_class_counts(corpus, split, side="train", unit="question")
    q_test = QS.side_class_counts(corpus, split, side="test", unit="question")
    r_train = QS.side_class_counts(corpus, split, side="train", unit="row")
    r_test = QS.side_class_counts(corpus, split, side="test", unit="row")
    document: Dict[str, Any] = {
        "version": str(QA_SCOPED_VERSION),
        "caliber": {
            "name": "A",
            "statement": str(CALIBER_STATEMENT),
            "min_class_questions": int(min_class_questions),
            "count_unit": "questions",
            "merge_enabled": False,
            "merge_note": (
                "本轮**不使用**轮 8 的答案归并（high-risk 两簇语义有误：china(Peking) -> china、"
                "america(US)/(USA) -> 有歧义的 america）；high-risk 未解决前不得启用归并。"
            ),
        },
        "product_dir": str(product_dir),
        "product_files_sha256": {
            ANSWER_TABLE_FILE: _sha256_file(
                os.path.join(str(product_dir), ANSWER_TABLE_FILE)
            ),
            ALL_PAIRS_FILE: _sha256_file(os.path.join(str(product_dir), ALL_PAIRS_FILE)),
        },
        "effective_scope": {
            "min_class_questions": int(min_class_questions),
            "holdout_pct": int(holdout_pct),
            "seed": int(seed),
            "single_seed": True,
            "max_classes": 0,
            "statement": (
                "A 口径 = 每类问题数 >= %d + 全量类集上的 (-count, key) 不截断 + %d%% 按比例"
                "留出 + 单 seed(%d) + **归并关闭** + **零训练**"
                % (int(min_class_questions), int(holdout_pct), int(seed))
            ),
        },
        "corpus": {
            "n_classes": int(corpus.n_classes),
            "n_questions": int(corpus.n_questions),
            "n_pairs": int(corpus.n_pairs),
            "n_answer_classes_in_table": int(len(QS.load_answer_table(str(product_dir)))),
            "class_scope": class_scope,
            "class_question_counts_post_scope": QS.class_size_distribution(
                corpus.class_question_counts
            ),
            "class_row_counts_post_scope": QS.class_size_distribution(
                corpus.class_row_counts
            ),
            "answer_keys_head": [str(x) for x in corpus.answer_keys[:5]],
            "answer_keys_sha256": QS.sha256_text(_canonical_json(list(corpus.answer_keys))),
        },
        "split": {
            **split.summary(),
            "actual_test_question_ratio": (
                float(len(split.test_questions)) / float(corpus.n_questions)
            ),
            "actual_test_row_ratio": float(len(split.test_rows)) / float(corpus.n_pairs),
            "target_ratio": float(holdout_pct) / 100.0,
            "invariants": invariants,
        },
        "side_class_counts": {
            "train_by_question": QS.class_size_distribution(q_train),
            "test_by_question": QS.class_size_distribution(q_test),
            "train_by_row": QS.class_size_distribution(r_train),
            "test_by_row": QS.class_size_distribution(r_test),
        },
        "coverage": {
            "test_by_question": QS.class_coverage(q_test),
            "test_by_row": QS.class_coverage(r_test),
            "train_by_question": QS.class_coverage(q_train),
        },
        "honest_notes": [
            (
                "门槛用**筛选前**每类问题数判定（pre_scope）；语料内实际每类问题数"
                "（post_scope）会略小：含越界金标的问题被**整体**排除。"
            ),
            (
                "现场 post_scope 的每类问题数下界可能低于门槛：某些类的问题里有相当一部分"
                "同时带有被剔除类的金标，整组排除后就掉到门槛以下 —— 这是一个**口径事实**，"
                "不是缺陷；若要求「语料内每类也 >= 门槛」，需要再来一轮迭代筛（本轮不做）。"
            ),
            "类集收缩**不增加任何样本**：只是把类与问题一起缩小到子集，问题/行数都变少。",
            "本轮零训练：原型基线只说明「零训练可学性下限」，不代表任何模型能力。",
        ],
    }
    return {"corpus": corpus, "split": split, "document": document}


def evaluate_prototype(
    *,
    corpus: Any,
    split: Any,
    product_dir: str = DEFAULT_PRODUCT_DIR,
    profile: str = DEFAULT_PROFILE,
) -> Dict[str, Any]:
    """零训练**原型质心基线** + 同口径多数类基线 + 决策门（3a/3b/3c）。

    参数
    ----
    corpus / split : Any
        :func:`build_scoped_corpus` 给出的语料与切分。
    product_dir : str
        冻结产物目录（读问题文本）。
    profile : str
        特征档（默认 ``lexical-88``）。

    返回
    ----
    Dict[str, Any]
        ``prototype_document`` / ``gate_document``（两份可直接序列化的文档）。
    """
    import n3d_qa_learn.qa_scale as QS

    texts = load_question_texts(str(product_dir))
    gold = gold_local_map(corpus)
    all_qids = sorted(str(q) for q in corpus.question_gold.keys())
    features = build_question_matrix(
        texts["texts"], all_qids, profile=str(profile), role=ROLE_QUESTION
    )
    prototypes = class_prototypes(
        features["matrix"],
        all_qids,
        gold,
        split.train_questions,
        n_classes=int(corpus.n_classes),
    )
    predicted = predict_by_cosine(
        features["matrix"], all_qids, split.test_questions, prototypes["centroids"]
    )
    proto_metrics = question_level_metrics(
        predictions=predicted["predictions"],
        questions=split.test_questions,
        gold_local=gold,
        n_classes=int(corpus.n_classes),
    )
    train_counts = QS.side_class_counts(corpus, split, side="train", unit="question")
    bonus = majority_predictions(
        train_counts=train_counts, n_test_questions=len(split.test_questions)
    )
    majority_metrics = question_level_metrics(
        predictions=bonus["predictions"],
        questions=split.test_questions,
        gold_local=gold,
        n_classes=int(corpus.n_classes),
    )
    row_reference = QS.deterministic_baseline_metrics(corpus, split, side="test")
    prototype_document: Dict[str, Any] = {
        "version": str(QA_SCOPED_VERSION),
        "classifier": {
            "name": "prototype_centroid_cosine",
            "training": "none",
            "rule": (
                "每类原型 = 该类**训练侧**问题特征的算术平均再 L2 归一化；测试侧按余弦 "
                "argmax 预测（平票取最小类下标）。**不做任何梯度更新、不加载任何模型。**"
            ),
        },
        "feature": {
            "profile": str(features["profile"]),
            "role": str(features["role"]),
            "dim": int(features["dim"]),
            "fingerprint": str(features["fingerprint"]),
            "n_texts": int(features["n_texts"]),
            "n_zero_norm": int(features["n_zero_norm"]),
            "persisted": bool(features["persisted"]),
            "note": "**不落盘裸特征矩阵**：矩阵只在内存中参与质心与余弦计算。",
        },
        "prototypes": {
            "n_classes": int(corpus.n_classes),
            "n_train_questions_used": int(prototypes["n_train_questions_used"]),
            "prototype_support_distribution": _quantiles(prototypes["counts"]),
            "n_zero_norm_centroids": int(prototypes["n_zero_norm_centroids"]),
            "rule": str(prototypes["rule"]),
        },
        "predictions": {
            "n_test_questions": int(len(predicted["predictions"])),
            "n_ties": int(predicted["n_ties"]),
            "tie_rule": str(predicted["tie_rule"]),
            "top1_cosine_distribution": _quantiles(predicted["top1_cosine"]),
            "margin_distribution": _quantiles(predicted["margin"]),
            "n_margin_le_1e-6": int(
                sum(1 for m in predicted["margin"] if abs(float(m)) <= 1e-6)
            ),
        },
        "metrics": {
            "prototype": proto_metrics,
            "majority_same_caliber": majority_metrics,
            "majority_predictor": {
                "class_index": int(bonus["class_index"]),
                "class_key": str(corpus.answer_keys[int(bonus["class_index"])]),
                "class_train_question_count": int(bonus["class_train_question_count"]),
                "predictor": str(bonus["predictor"]),
                "rule": str(bonus["rule"]),
            },
            "caliber_note": (
                "两个指标用的是**同一问题级口径**（同一批测试问题、同一 macro 分母），"
                "因此可以直接相减；行级（round-1）基线只作参照，**不可**与它们直接比较。"
            ),
        },
        "row_level_reference": {
            "source": "n3d_qa_learn.qa_scale.deterministic_baseline_metrics(side='test')",
            "constant_majority": dict(row_reference["constant_majority"]),
            "row_argmax": dict(row_reference["row_argmax"]),
            "caliber_warning": (
                "**行级口径**（每个候选行判一次），与上面的问题级口径不同量纲："
                "只作与轮 1 的连续性参照，**不得**与原型 macro 直接比较、也不得代入判据。"
            ),
        },
    }
    gate_document = decision_gate(proto_metrics, majority_metrics, corpus=corpus)
    return {
        "prototype_document": prototype_document,
        "gate_document": gate_document,
    }


def decision_gate(
    prototype_metrics: Mapping[str, Any],
    majority_metrics: Mapping[str, Any],
    *,
    corpus: Any = None,
    margin: float = GATE_MARGIN,
) -> Dict[str, Any]:
    """**决策门**：原型 macro 是否比同口径多数类基线高出 >= ``margin``（+10pp）。

    **只判零训练下的可学性下限**：不过门就如实给「特征空间不可分」的判断与后续建议，
    不粉饰；过门也只说明下限成立。
    """
    proto_macro = float(prototype_metrics["macro"])
    base_macro = float(majority_metrics["macro"])
    delta = float(proto_macro - base_macro)
    passed = bool(delta >= float(margin))
    proto_micro = float(prototype_metrics["micro"])
    base_micro = float(majority_metrics["micro"])
    n_classes = int(prototype_metrics.get("n_classes", 0))
    n_with_test = int(prototype_metrics.get("n_classes_with_test_questions", 0))
    n_zero_recall = int(prototype_metrics.get("n_zero_recall_classes", 0))
    recall_dist = dict(prototype_metrics.get("recall_distribution", {}))
    if passed:
        verdict = (
            "过门：原型质心基线的问题级 macro = %.6f，比同口径多数类基线 %.6f 高 %.6f"
            "（>= %.2f）⇒ **零训练可学性下限成立**（在当前特征档与 A 口径下）。"
            "**限定条件（必须与结论一起引用，见 quality_caveats）**："
            "① 原型 micro = %.6f **低于**多数类基线 micro = %.6f（基线把预测压在单一超大类上，"
            "故判据用 macro 而非 micro）；② 有 %d / %d 个有测试问题的类 recall = 0"
            "（recall 中位 = %.3f）⇒ 只能说「下限成立、覆盖面有限」，**不是全类可分**；"
            "③ `macro_all_classes` = %.6f（分母 = 全部 %d 类）与 `macro` 不同量纲。"
            "这**不等于已学会**，也不代表 N3D 后端能学到什么。"
            % (
                proto_macro,
                base_macro,
                delta,
                float(margin),
                proto_micro,
                base_micro,
                int(n_zero_recall),
                int(n_with_test),
                float(recall_dist.get("median", 0.0)),
                float(prototype_metrics.get("macro_all_classes", 0.0)),
                int(n_classes),
            )
        )
    else:
        verdict = (
            "未过门：原型质心基线的问题级 macro = %.6f，同口径多数类基线 = %.6f，"
            "差值 = %.6f < %.2f（不足 +%.0fpp）⇒ **在当前特征档下 82 类闭集分类的可学前提"
            "未建立**；这不是「模型不行」，而是「连零训练的原型分类都分不开」⇒ 特征空间不可分"
            "（micro/macro 同时偏低即为证据）。后续建议：①换语义特征档（bge-m3，D=1024）重测"
            "同一口径；②继续上游放量（每类问题数再翻倍）；③复核 A 口径门槛是否过松/过紧。"
            "**不得**把本结果包装成「已可学」。"
            % (proto_macro, base_macro, delta, float(margin), float(margin) * 100.0)
        )
    caveats: List[str] = []
    if float(proto_micro - base_micro) < 0.0:
        caveats.append(
            "原型 micro = %.6f **低于**多数类基线的 micro = %.6f：基线把全部预测压在"
            "一个超大类上（micro 靠大类刷出来），而原型把预测摊到各类 ⇒ 两者 micro 不可"
            "直接当作优劣；本条判据用 **macro**（逐类 recall 的算术平均）正是为了不被超大"
            "类主导。" % (float(proto_micro), float(base_micro))
        )
    if n_zero_recall > 0:
        caveats.append(
            "有 **%d / %d** 个「有测试问题的类」recall = 0（recall 分位：p25 = %.3f、"
            "median = %.3f、p75 = %.3f）⇒ macro 由**可分类**支撑，覆盖面有限；"
            "这不是「全类可学」。"
            % (
                int(n_zero_recall),
                int(n_with_test),
                float(recall_dist.get("p25", 0.0)),
                float(recall_dist.get("median", 0.0)),
                float(recall_dist.get("p75", 0.0)),
            )
        )
    if n_with_test < n_classes:
        caveats.append(
            "类数 = %d，但**有测试问题**的类只有 %d ⇒ `macro` 的分母是 %d"
            "（零测试类不进分母），`macro_all_classes` = %.6f 才是「按全部类数」的口径；"
            "两个数必须分开引用。"
            % (
                int(n_classes),
                int(n_with_test),
                int(n_with_test),
                float(prototype_metrics.get("macro_all_classes", 0.0)),
            )
        )
    return {
        "version": str(QA_SCOPED_VERSION),
        "gate_rule": str(GATE_RULE),
        "margin": float(margin),
        "baseline": {
            "name": "constant_majority",
            "macro": float(base_macro),
            "micro": float(base_micro),
            "n_classes_in_macro": int(majority_metrics.get("n_classes_with_test_questions", 0)),
        },
        "prototype": {
            "macro": float(proto_macro),
            "micro": float(proto_micro),
            "macro_all_classes": float(prototype_metrics.get("macro_all_classes", 0.0)),
            "n_classes_in_macro": int(prototype_metrics.get("n_classes_with_test_questions", 0)),
            "n_zero_recall_classes": int(prototype_metrics.get("n_zero_recall_classes", 0)),
            "recall_distribution": dict(prototype_metrics.get("recall_distribution", {})),
        },
        "delta_macro": float(delta),
        "delta_micro": float(proto_micro - base_micro),
        "passed": bool(passed),
        "quality_caveats": caveats,
        "verdict": str(verdict),
        "not_claimed": [
            "已学会 / 模型能力（零训练、未评测任何模型）",
            "N3D 后端可学到该上限（本轮不涉及任何后端训练）",
            "跨 seed / 跨留出比例稳定性（单 seed 42、单 20%）",
        ],
        "n_classes": int(n_classes),
        "feature_profile_note": (
            "本轮特征档 = lexical-88（词面哈希 + 长度特征，D=88）；换档须**重跑同一口径**"
            "（同 seed / 同留出 / 同问题级 macro 分母）后才可比较，不得跨档混用结论。"
        ),
    }


__all__ = [
    "QA_SCOPED_VERSION",
    "DEFAULT_PRODUCT_DIR",
    "ANSWER_TABLE_FILE",
    "ALL_PAIRS_FILE",
    "VERIFY_ROOT",
    "DEFAULT_OUT_DIR",
    "DEFAULT_MIN_CLASS_QUESTIONS",
    "DEFAULT_HOLDOUT_PCT",
    "DEFAULT_SEED",
    "DEFAULT_PROFILE",
    "ROLE_QUESTION",
    "GATE_MARGIN",
    "GATE_RULE",
    "CALIBER_STATEMENT",
    "build_question_matrix",
    "build_scoped_corpus",
    "class_prototypes",
    "decision_gate",
    "evaluate_prototype",
    "gold_local_map",
    "load_question_texts",
    "majority_predictions",
    "predict_by_cosine",
    "question_level_metrics",
    "verify_split_invariants",
]
