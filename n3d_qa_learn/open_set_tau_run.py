"""轮 3 · B4 的 CLI：``python -m n3d_qa_learn.open_set_tau_run {calibrate|report}``。

现场场景（**真实语料、真实键表、同总体同口径**）
------------------------------------------------
用冻结的 QA 语料（``max_classes=298`` 全量）构造一个**自洽的开集检索场景**：

* **键表 K** = 训练侧问题的题面向量（``lexical-88`` / ``role=question``），
  并按 ``question_id`` 哈希再抽掉 10% 作为「已知但不在表内」的候选；
* **正总体**（应被**识别**）= 被抽掉的那 10% 训练侧问题 —— 它们**内容上属于**训练侧
  分布，是"已知类别"的留出样本；
* **负总体**（应被**拒识**）= **测试侧**问题 —— 按轮 1 的整组切分它们**从不进键表**，
  正是开集意义上的「未识别」样本。

⇒ 正负两总体**同一侧之外**的唯一差别就是「是否在键表内」，其余口径（向量化器、键表、
batch）**完全一致** —— 满足「同总体、同样本量、同口径」的纪律；样本量差异如实报出。

产物：``checkpoints/qa_learn/_verify/robust/open_set_tau.json`` / ``.md``
（**不落盘裸特征矩阵、不产 zip、无挂钟字段**）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 日志 / IO
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """打印一行进度（不可表示字符做 ``backslashreplace`` 兜底）。"""
    text = str(msg)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = str(getattr(sys.stdout, "encoding", "") or "ascii")
        print(text.encode(enc, errors="backslashreplace").decode(enc, errors="replace"),
              flush=True)


def _write_text(path: str, text: str) -> Tuple[str, int]:
    """写 UTF-8 无 BOM 文本（LF）；返回 ``(sha256, bytes)``。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    blob = str(text).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(blob)
    return hashlib.sha256(blob).hexdigest(), int(len(blob))


def _write_json(path: str, obj: Any) -> Tuple[str, int]:
    """写 JSON（UTF-8 无 BOM、``sort_keys``、缩进 1、末尾换行）。"""
    return _write_text(path, json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1) + "\n")


def _read_json(path: str) -> Any:
    """读 JSON；缺文件给出可读报错。"""
    if not os.path.isfile(path):
        raise FileNotFoundError("缺少输入产物 " + repr(path) + "；请先跑 calibrate")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _guard_out_dir(out_dir: str) -> str:
    """产物目录守卫：只允许写在 ``checkpoints/qa_learn/_verify/`` 之下。"""
    import n3d_qa_learn.open_set_tau as OT

    text = str(out_dir) if out_dir else OT.DEFAULT_OUT_DIR
    root_abs = os.path.abspath(OT.VERIFY_ROOT)
    norm_abs = os.path.abspath(text)
    if not (norm_abs == root_abs or norm_abs.startswith(root_abs + os.sep)):
        raise ValueError(
            "产物目录只允许写在 "
            + repr(OT.VERIFY_ROOT.replace(chr(92), "/")) + " 之下，当前 = "
            + repr(norm_abs.replace(chr(92), "/"))
        )
    return text


# ---------------------------------------------------------------------------
# 场景构造
# ---------------------------------------------------------------------------

#: 键表外留出的训练侧问题比例（百分比；用于给正总体，**必须 **不**进键表**）。
POS_HOLDOUT_PCT: int = 10


def _pos_holdout(question_id: str, seed: int) -> bool:
    """确定性判定「该训练侧问题是否被抽掉、作为正总体」（与轮 1 同风格哈希抽稀）。"""
    raw = ("posholdout\0" + str(int(seed)) + "\0" + str(question_id)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % 100 < int(POS_HOLDOUT_PCT)


def _max_cosine(chunk: np.ndarray, keys: np.ndarray, batch: int = 64) -> np.ndarray:
    """逐条求「对键表的最大余弦」（分批，避免一次性大矩阵）。"""
    out = np.zeros(int(chunk.shape[0]), dtype=np.float64)
    kk = np.ascontiguousarray(keys, dtype=np.float32)
    for b0 in range(0, int(chunk.shape[0]), max(1, int(batch))):
        part = np.ascontiguousarray(chunk[b0:b0 + max(1, int(batch))], dtype=np.float32)
        sims = part @ kk.T
        out[b0:b0 + int(part.shape[0])] = sims.max(axis=1)
    return out


def build_scenario(
    *,
    product_dir: str = "",
    seed: int,
    holdout_pct: int,
    limit_questions: int = 0,
) -> Dict[str, Any]:
    """构造开集场景并现场算出正 / 负两总体的最高余弦。

    返回
    ----
    Dict[str, Any]
        ``positive`` / ``negative``（最高余弦列表）、``population``（同侧同 profile
        同一键表的取证）、``sizes``、``vectorizer``、``score_stats``。
    """
    import n3d_qa_learn.entry_table as ET
    import n3d_qa_learn.qa_scale as QS
    import n3d_qa_learn.qa_variants as QV

    root = str(product_dir) if product_dir else str(QS.DEFAULT_PRODUCT_DIR)
    corpus = QS.build_scale_corpus(root, max_classes=0)
    texts = QV.load_question_texts(root)
    units = QV.build_question_units(
        corpus, question_texts=texts, limit_questions=int(limit_questions)
    )
    vec = ET.build_vectorizer_for("lexical-88", "question")
    qids = [str(u.question_id) for u in units]
    sides = [
        QS.question_side(q, str(u.task), int(seed), int(holdout_pct))
        for q, u in zip(qids, units)
    ]
    train_idx = [i for i, s in enumerate(sides) if str(s) == "train"]
    test_idx = [i for i, s in enumerate(sides) if str(s) == "test"]
    pos_hold = [i for i in train_idx if _pos_holdout(qids[i], int(seed))]
    key_idx = [i for i in train_idx if not _pos_holdout(qids[i], int(seed))]
    if not key_idx or not pos_hold or not test_idx:
        raise ValueError(
            "开集场景构造失败：键表 %d / 正总体 %d / 负总体 %d（任一组为空）"
            % (len(key_idx), len(pos_hold), len(test_idx))
        )
    key_texts = [str(units[i].question_text) for i in key_idx]
    pos_texts = [str(units[i].question_text) for i in pos_hold]
    neg_texts = [str(units[i].question_text) for i in test_idx]
    kmat = ET.encode_statements(vec, key_texts, batch_size=64)
    pmat = ET.encode_statements(vec, pos_texts, batch_size=64)
    nmat = ET.encode_statements(vec, neg_texts, batch_size=64)
    pos_scores = _max_cosine(pmat, kmat)
    neg_scores = _max_cosine(nmat, kmat)
    table_sha = hashlib.sha256(
        np.ascontiguousarray(kmat, dtype=np.float32).tobytes()
    ).hexdigest()
    population = {
        "side": "qa_questions",
        "pos_side": "qa_questions",
        "neg_side": "qa_questions",
        "profile": "lexical-88",
        "role": "question",
        "table_sha256": str(table_sha),
        "table_kind": "train_side_question_keys（按 question_id 哈希再抽掉 %d%% 作正总体）"
                      % int(POS_HOLDOUT_PCT),
        "positive_kind": "train 侧、被抽掉、**不在键表内**的问题",
        "negative_kind": "test 侧问题（按轮 1 整组切分，从不进键表）",
        "same_population_rule": (
            "正负两总体共用**同一**向量化器、**同一**键表、**同一**批大小；"
            "唯一差别是「是否在键表内」。样本量差异如实报出（见 sizes）。"
        ),
    }

    def _stats(a: np.ndarray) -> Dict[str, Any]:
        return {
            "n": int(a.size),
            "min": float(a.min()) if a.size else 0.0,
            "mean": float(a.mean()) if a.size else 0.0,
            "max": float(a.max()) if a.size else 0.0,
            "p05": float(np.quantile(a, 0.05)) if a.size else 0.0,
            "p50": float(np.quantile(a, 0.50)) if a.size else 0.0,
            "p95": float(np.quantile(a, 0.95)) if a.size else 0.0,
        }

    return {
        "positive": [float(x) for x in pos_scores.tolist()],
        "negative": [float(x) for x in neg_scores.tolist()],
        "population": population,
        "sizes": {
            "n_train_questions": int(len(train_idx)),
            "n_test_questions": int(len(test_idx)),
            "n_table_keys": int(len(key_idx)),
            "n_positive": int(len(pos_hold)),
            "n_negative": int(len(test_idx)),
            "pos_holdout_pct": int(POS_HOLDOUT_PCT),
        },
        "vectorizer": {
            "profile": "lexical-88",
            "role": "question",
            "dim": int(vec.dim),
            "fingerprint": str(vec.fingerprint()),
        },
        "score_stats": {"positive": _stats(pos_scores), "negative": _stats(neg_scores)},
    }


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    """现场标定开集 τ 并落盘（含边界用例与严格性对照）。"""
    import n3d_qa_learn.open_set_tau as OT
    import n3d_qa_learn.qa_scale as QS

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    product_dir = str(getattr(args, "product_dir", "") or "") or str(QS.DEFAULT_PRODUCT_DIR)
    seed = int(getattr(args, "seed", 42) or 42)
    holdout_pct = int(getattr(args, "holdout_pct", QS.DEFAULT_HOLDOUT_PCT))
    limit_q = int(getattr(args, "limit_questions", 0) or 0)
    _log("[calibrate] 构造开集场景（product_dir=%s, seed=%d, holdout=%d%%）"
         % (str(product_dir), int(seed), int(holdout_pct)))
    sc = build_scenario(
        product_dir=str(product_dir),
        seed=int(seed),
        holdout_pct=int(holdout_pct),
        limit_questions=int(limit_q),
    )
    _log("[calibrate] 键表 %d / 正总体 %d / 负总体 %d；开始标定"
         % (sc["sizes"]["n_table_keys"], sc["sizes"]["n_positive"],
            sc["sizes"]["n_negative"]))
    out = OT.calibrate_tau(
        sc["positive"], sc["negative"], population=sc["population"], factor=1.0
    )
    sweep = OT.factor_sweep(
        sc["positive"], sc["negative"],
        population=sc["population"], factors=(1.0, 2.0, 3.0),
    )
    bnd = OT.boundary_cases()
    ctl = OT.strictness_control_case()
    doc: Dict[str, Any] = {
        "version": str(OT.OPEN_SET_TAU_VERSION),
        "scope": {
            "max_classes": int(QS.FULL_CLASSES),
            "seed": int(seed),
            "holdout_pct": int(holdout_pct),
            "limit_questions": int(limit_q),
        },
        "scenario": {
            "sizes": sc["sizes"],
            "population": sc["population"],
            "vectorizer": sc["vectorizer"],
            "score_stats": sc["score_stats"],
            "score_distributions_sha256": {
                "positive": hashlib.sha256(
                    np.asarray(sc["positive"], dtype=np.float64).tobytes()
                ).hexdigest(),
                "negative": hashlib.sha256(
                    np.asarray(sc["negative"], dtype=np.float64).tobytes()
                ).hexdigest(),
            },
        },
        "calibration": out,
        "factor_sweep": sweep,
        "boundary_evidence": bnd,
        "strictness_control": ctl,
        "rules": {
            "guard_boundary_rule": str(OT.GUARD_BOUNDARY_RULE),
            "noise_floor_rule": str(OT.NOISE_FLOOR_RULE),
            "granularity_rule": str(OT.GRANULARITY_RULE),
            "unrecognized_rule": "未识别 ⇔ 最高余弦 < τ；误召回 ⇔ 最高余弦 >= τ",
        },
    }
    sha, size = _write_json(os.path.join(out_dir, "open_set_tau.json"), doc)
    _log("[calibrate] tau=%s margin=%s noise_floor=%s granularity=%s threshold=%s effective=%s"
         % (out.get("tau"), out.get("margin"), out.get("noise_floor"),
            out.get("granularity_tau"), out.get("threshold"), out.get("effective")))
    _log("[calibrate] 边界等号用例 %d 条全部 ineffective=%s；严格性对照 proven=%s"
         % (int(bnd["n_found"]), str(bnd["all_effective_false"]),
            str(ctl.get("strictness_proven"))))
    _log("[calibrate] open_set_tau.json = %d B %s" % (int(size), sha[:16]))
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def render_markdown(doc: Dict[str, Any]) -> str:
    """渲染 Markdown 报告（确定性，无时间戳）。"""
    cal = doc["calibration"]
    sc = doc["scenario"]
    lines: List[str] = []
    lines.append("# 轮 3 · B4：开集「未识别」档的 τ 标定（补齐 1a 遗留）")
    lines.append("")
    lines.append("## 1. 现场场景")
    lines.append("")
    sz = sc["sizes"]
    lines.append("| 项 | 现场值 |")
    lines.append("| --- | --- |")
    lines.append("| 训练侧问题 | " + str(sz["n_train_questions"]) + " |")
    lines.append("| 测试侧问题 | " + str(sz["n_test_questions"]) + " |")
    lines.append("| 键表 K（训练侧、抽掉正总体后） | " + str(sz["n_table_keys"]) + " |")
    lines.append("| 正总体（应识别，n） | " + str(sz["n_positive"]) + " |")
    lines.append("| 负总体（应拒识，n） | " + str(sz["n_negative"]) + " |")
    lines.append("| 正总体抽稀比例 | " + str(sz["pos_holdout_pct"]) + "% |")
    lines.append("")
    lines.append("- 向量化器：`" + str(sc["vectorizer"]["profile"]) + "` / role=`"
                 + str(sc["vectorizer"]["role"]) + "` / dim=" + str(sc["vectorizer"]["dim"])
                 + " / fingerprint=`" + str(sc["vectorizer"]["fingerprint"]) + "`")
    lines.append("- 键表内容 SHA256：`" + str(sc["population"]["table_sha256"]) + "`")
    lines.append("- **同总体口径**：" + str(sc["population"]["same_population_rule"]))
    lines.append("")
    st = sc["score_stats"]
    lines.append("| 总体 | n | min | mean | max | p05 | p50 | p95 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in ("positive", "negative"):
        d = st[str(name)]
        lines.append(
            "| `" + str(name) + "` | " + str(d["n"]) + " | " + ("%.6f" % d["min"])
            + " | " + ("%.6f" % d["mean"]) + " | " + ("%.6f" % d["max"])
            + " | " + ("%.6f" % d["p05"]) + " | " + ("%.6f" % d["p50"])
            + " | " + ("%.6f" % d["p95"]) + " |"
        )
    lines.append("")
    lines.append("## 2. 标定结果")
    lines.append("")
    if bool(cal.get("applicable")):
        wp = cal["workpoint"]
        lines.append("| 量 | 现场值 |")
        lines.append("| --- | --- |")
        lines.append("| **τ** | **" + ("%.6f" % float(cal["tau"])) + "** |")
        lines.append("| 平衡准确率 | " + ("%.6f" % float(wp["balanced_accuracy"])) + " |")
        lines.append("| recall（正总体被识别率） | " + ("%.6f" % float(wp["recall"])) + " |")
        lines.append("| specificity（负总体被拒识率） | "
                     + ("%.6f" % float(wp["specificity"])) + " |")
        lines.append("| 被识别正样本数 | " + str(wp["n_pos_kept"]) + " / "
                     + str(cal["n_pos"]) + " |")
        lines.append("| 被拒识负样本数 | " + str(wp["n_neg_rejected"]) + " / "
                     + str(cal["n_neg"]) + " |")
        lines.append("| **margin**（实测间隔） | **"
                     + ("%.6f" % float(cal["margin"])) + "** |")
        lines.append("| **noise_floor** | " + ("%.6f" % float(cal["noise_floor"]))
                     + " |")
        lines.append("| **granularity_tau** | " + ("%.6f" % float(cal["granularity_tau"]))
                     + " |")
        lines.append("| **threshold** = max(factor×floor, granularity) | **"
                     + ("%.6f" % float(cal["threshold"])) + "** |")
        lines.append("| **effective** | **" + str(bool(cal["effective"])) + "** |")
        lines.append("")
        lines.append("- 判据：" + str(cal["threshold_rule"]))
        lines.append("- **边界归属**：" + str(cal["boundary_rule"]))
        lines.append("")
        lines.append("### 2.1 噪声底（同总体、同口径、样本量减半）")
        lines.append("")
        lines.append("| 半集 | n_pos | n_neg | τ_half | 与全量 τ 的偏差 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for h in cal["noise_floor_block"]["halves"]:
            lines.append(
                "| " + str(h["half"]) + " | " + str(h["n_pos"]) + " | " + str(h["n_neg"])
                + " | " + ("" if h["tau_half"] is None else "%.6f" % h["tau_half"])
                + " | " + ("" if h["abs_deviation"] is None
                           else "%.6f" % h["abs_deviation"]) + " |"
            )
        lines.append("")
        lines.append("- " + str(cal["noise_floor_block"]["rule"]))
        lines.append("")
        lines.append("### 2.2 粒度（1/n 必须与落差一并核算）")
        lines.append("")
        gr = cal["granularity_block"]
        lines.append("- `granularity_tau` = " + ("%.6f" % float(gr["granularity_tau"]))
                     + "；工作点两侧最近间隙 = `" + str(gr["near_gaps"]) + "`")
        lines.append("- `granularity_rate_pos` = 1/" + str(cal["n_pos"]) + " = "
                     + ("%.6f" % float(gr["granularity_rate_pos"]))
                     + "；`granularity_rate_neg` = 1/" + str(cal["n_neg"]) + " = "
                     + ("%.6f" % float(gr["granularity_rate_neg"])))
        lines.append("- " + str(gr["rule"]))
        # **E-B4a 修复（离朱 R82）**：倍数扫描 —— 把「factor 取 1/2/3 时的 threshold
        # 与判定」逐一给出，使「判定对 factor 的敏感性」成为可核对的数，而不只是一行公式。
        lines.append("")
        lines.append("### 2.3 倍数扫描（`factor ∈ {1, 2, 3}`：threshold 与判定）")
        lines.append("")
        sw = doc.get("factor_sweep") or {}
        if sw.get("rows"):
            lines.append("| factor | `factor × noise_floor` | `granularity_tau` | **threshold** | 主导项 | effective |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for row in sw["rows"]:
                if row.get("threshold") is None:
                    lines.append(
                        "| " + str(row["factor"]) + " | — | — | — | 不适用 | — |"
                    )
                    continue
                lines.append(
                    "| " + ("%.1f" % float(row["factor"])) + " | "
                    + ("%.10f" % float(row["noise_floor_term"])) + " | "
                    + ("%.10f" % float(row["granularity_term"])) + " | **"
                    + ("%.10f" % float(row["threshold"])) + "** | `"
                    + str(row["binding_term"]) + "` | **"
                    + str(row["effective"]) + "** |"
                )
            lines.append("")
            lines.append("- 判定不随 factor 变化：**"
                         + str(sw.get("verdict_stable")) + "**（现场主因 = `granularity` 项主导，"
                         "factor 从 1 提到 3 时 `factor × noise_floor` 仍小于 `granularity_tau`）。")
            lines.append("- " + str(sw.get("rule", "")))
        else:
            lines.append("- 本次产物未携带倍数扫描块。")
    else:
        lines.append("- **不适用**：" + str(cal.get("reason", "")))
    lines.append("")
    lines.append("## 3. 边界用例（**恰好等于阈值**，等号归无效）")
    lines.append("")
    bnd = doc["boundary_evidence"]
    lines.append("- 搜索网格：" + str(bnd["search_grid"]["note"]) + "；扫描 "
                 + str(bnd["search_grid"]["n_scanned"]) + " 组，可分 "
                 + str(bnd["search_grid"]["n_after_separable_filter"]) + " 组，"
                 + "**命中等号 " + str(bnd["n_found"]) + " 组**")
    lines.append("")
    lines.append("| 用例 | pos | neg | τ | margin | noise_floor | granularity | threshold | margin==threshold | effective |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in bnd["cases"]:
        lines.append(
            "| `" + str(c["case"]) + "` | `" + str(c["pos"]) + "` | `" + str(c["neg"])
            + "` | " + ("%.4f" % c["tau"]) + " | " + ("%.4f" % c["margin"])
            + " | " + ("%.4f" % c["noise_floor"]) + " | "
            + ("%.4f" % c["granularity_tau"]) + " | " + ("%.4f" % c["threshold"])
            + " | " + str(c["margin_equals_threshold"]) + " | **"
            + str(c["effective"]) + "** |"
        )
    lines.append("")
    lines.append("- 全部等号用例判无效：**" + str(bnd["all_effective_false"]) + "**")
    lines.append("- " + str(bnd["boundary_rule"]))
    lines.append("")
    lines.append("## 4. 严格性对照（证明判据**不是恒假**）")
    lines.append("")
    ctl = doc["strictness_control"]
    lines.append("| 情形 | τ | margin | threshold | effective |")
    lines.append("| --- | --- | --- | --- | --- |")
    for label in ("at_equality", "lifted"):
        d = ctl[str(label)]
        lines.append(
            "| `" + str(label) + "` | " + ("%.4f" % d["tau"]) + " | "
            + ("%.4f" % d["margin"]) + " | " + ("%.4f" % d["threshold"]) + " | **"
            + str(d["effective"]) + "** |"
        )
    lines.append("")
    lines.append("- 严格性成立：**" + str(ctl.get("strictness_proven")) + "**")
    lines.append("- " + str(ctl.get("rule", "")))
    lines.append("")
    lines.append("## 5. 如实登记的局限")
    lines.append("")
    lines.append("1. 本场景是**题面自检索**（键表 = 训练侧题面向量），不是「问题 -> 答案」的"
                 "闭环检索；它只用来给「未识别」档标一个**可审计**的 τ，"
                 "不代表任务准确率。")
    lines.append("2. 正 / 负两总体**样本量不同**（见第 1 节），噪声底按各自对半拆分估计，"
                 "故两侧的 1/n 粒度也不同 —— 已在 2.2 显式给出。")
    lines.append("3. 单 seed（" + str(doc["scope"]["seed"]) + "）；跨 seed / 跨抽稀比例未测。")
    lines.append("4. τ 的值**只对本场景与本口径**成立；换向量化器或换键表必须重新标定"
                 "（`population.table_sha256` 会变 ⇒ 禁止跨总体复用）。")
    lines.append("")
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    """渲染 ``open_set_tau_report.{json,md}``。"""
    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    doc = _read_json(os.path.join(out_dir, "open_set_tau.json"))
    md = render_markdown(doc)
    md_sha, md_size = _write_text(os.path.join(out_dir, "open_set_tau_report.md"), md)
    fps: Dict[str, Any] = {}
    for name in ("open_set_tau.json",):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            fps[str(name)] = {"bytes": int(os.path.getsize(p)), "sha256": _sha256_file(p)}
    fps["open_set_tau_report.md"] = {"bytes": int(md_size), "sha256": str(md_sha)}
    doc["artifact_fingerprints"] = fps
    sha, size = _write_json(os.path.join(out_dir, "open_set_tau_report.json"), doc)
    _log("[report] open_set_tau_report.json = %d B %s；open_set_tau_report.md = %d B %s"
         % (int(size), sha[:16], int(md_size), md_sha[:16]))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _Tee:
    """把 stdout 同时写到控制台与 UTF-8 无 BOM 日志文件。"""

    def __init__(self, primary: Any, handle: Any) -> None:
        self._primary = primary
        self._handle = handle

    def write(self, text: str) -> int:
        """两侧同时写。"""
        self._primary.write(text)
        self._handle.write(text)
        return len(text)

    def flush(self) -> None:
        """两侧同时 flush。"""
        self._primary.flush()
        self._handle.flush()

    def __getattr__(self, item: Any) -> Any:
        """其余属性透传。"""
        return getattr(self._primary, item)


def cmd_open_set_tau(args: argparse.Namespace) -> int:
    """子命令分派。"""
    sub = str(getattr(args, "tau_cmd", "") or "calibrate")
    handlers = {"calibrate": cmd_calibrate, "report": cmd_report}
    if sub not in handlers:
        _log("[open_set_tau] 未知子命令 " + repr(sub) + "；可用 = " + str(sorted(handlers)))
        return 2
    try:
        return int(handlers[sub](args))
    except FileNotFoundError as exc:
        _log("[open_set_tau] 输入产物缺失：" + str(exc))
        return 1
    except (ValueError, KeyError) as exc:
        _log("[open_set_tau] 契约校验失败：" + str(exc))
        return 2
    except RuntimeError as exc:
        _log("[open_set_tau] 运行失败：" + str(exc))
        return 1


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    import n3d_qa_learn.open_set_tau as OT

    parser = argparse.ArgumentParser(
        prog="n3d_qa_learn.open_set_tau_run",
        description=(
            "轮 3 · B4：开集「未识别」档的 τ 标定（含噪声底 / 粒度 / 边界归属 / 反例）。"
            "产物一律写 " + OT.DEFAULT_OUT_DIR.replace("\\", "/") + " 之下。"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, helptext in (
        ("calibrate", "现场构造开集场景并标定 τ（含边界用例与严格性对照）"),
        ("report", "渲染 open_set_tau_report.{json,md}"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--product-dir", default="", help="n3d_qa 冻结产物目录（只读）")
        p.add_argument("--out-dir", default="", help="产物目录（默认 " + OT.DEFAULT_OUT_DIR + "）")
        p.add_argument("--log-file", default="", help="同时写 UTF-8(无 BOM) 日志文件")
        p.add_argument("--seed", type=int, default=42, help="抽稀 seed（单 seed）")
        p.add_argument("--holdout-pct", type=int, default=20, help="测试集留出比例（百分比）")
        p.add_argument("--limit-questions", type=int, default=0,
                       help="小批演练：只取前 N 个问题（0 = 全量）")
        p.set_defaults(func=cmd_open_set_tau, tau_cmd=str(name))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    log_file = str(getattr(args, "log_file", "") or "")
    handle = None
    original = sys.stdout
    if log_file:
        parent = os.path.dirname(os.path.abspath(log_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        handle = open(log_file, "w", encoding="utf-8", newline="\n")
        sys.stdout = _Tee(original, handle)
    try:
        return int(args.func(args))
    finally:
        if handle is not None:
            sys.stdout = original
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
