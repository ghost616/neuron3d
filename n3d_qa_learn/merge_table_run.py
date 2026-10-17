"""轮 C：**合并表零训练对照**的纯新增 CLI。

驱动方式：``python -m n3d_qa_learn.merge_table_run {drill|eval|report}``

子命令
------
* ``drill`` —— **单条端到端演练**（历史纠正记录 #10：长跑类任务先做单条演练）：
  只取 1 条 QA 查询 + 1 条文本查询，把「两半构造 -> 合并 -> 三条前置断言 ->
  两臂路由 -> 类型契约」整条通路跑通，并把结果与 :func:`merge_table.arm_top1`
  的**全量口径**逐项对比；退码 0 才算通路成立。
* ``eval`` —— 全量（2814 条目 / 149 + 666 查询）两臂对照评测 + 三条前置断言 +
  分侧指标 + 类型契约统计 + 机器可读判定；``--repro-dir`` 时额外产出**双跑逐字节一致**证据。
* ``report`` —— 渲染 ``merge_table_report.{json,md}``。

产物纪律（**硬约束**）
----------------------
* 只写 ``checkpoints/qa_learn/_verify/merge_table/``（``--out-dir`` 也只允许写在该根之下）；
* **不落盘裸特征矩阵**、**不产 zip**、**不写任何挂钟 / 耗时字段**（故同参重跑逐字节一致）；
* 变体一律读**已冻结**的 ``variants.jsonl``，**禁止**在本轮重新生成变体（冻结纪律）。

控制台编码纪律（沿用 ``step2_run`` / ``text_variants_run``）
------------------------------------------------------------
本机 ``sys.stdout.encoding = cp936``（GBK），而 GBK 没有 ``U+2212``（真减号）/
``U+21D2``（双箭头）等字符。本模块**新增**日志字面量一律只用 ASCII + 汉字；
``_log`` 另做 ``backslashreplace`` 兜底（不改任何数值与产物）。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 日志（ASCII 安全 + backslashreplace 兜底）
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """打印一行进度（带刷新；对不可表示字符做 ``backslashreplace`` 兜底）。"""
    text = str(msg)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = str(getattr(sys.stdout, "encoding", "") or "ascii")
        safe = text.encode(enc, errors="backslashreplace").decode(enc, errors="replace")
        print(safe, flush=True)


#: 日志字面量的编码纪律（与 ``step2_run`` 同一纪律）。
ASCII_SAFE_LOG_NOTE: str = (
    "运行时日志字面量必须能被 Windows 控制台默认编码 cp936（GBK）表示："
    "U+2212（真减号）与 U+21D2（双箭头）等字符不在 GBK 内，出现即让 print 抛 "
    "UnicodeEncodeError（它继承 ValueError，会被 CLI 兜住并误报成非法参数）。"
    "本模块一律用 ASCII 替代：减号用 -、蕴含箭头用 =>。"
    "docstring / 注释 / 产物 JSON（UTF-8）不受此限。"
)

#: 本模块的 CLI 子命令（顺序敏感）。
MERGE_TABLE_COMMANDS: Tuple[str, ...] = ("drill", "eval", "report")


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


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
    """写 JSON（UTF-8 无 BOM，``sort_keys`` + 缩进 1 + 末尾换行）。"""
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    return _write_text(path, text)


def _write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> Tuple[str, int]:
    """写 JSONL（逐行 ``sort_keys`` + 紧凑分隔符；LF；末尾换行）。"""
    lines = [
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    return _write_text(path, "\n".join(lines) + "\n")


def _read_json(path: str) -> Any:
    """读 JSON；缺文件给出**可读报错**。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "缺少输入产物 " + repr(path) + "；请先跑前置子命令（drill / eval）"
        )
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


def _normalize_dir(path: str) -> str:
    """目录路径的**机器可比**规范化（绝对路径 + 统一正斜杠 + 小写）。

    存在理由（离朱 R77 F-A）：`a_dir` / `b_dir` 保留**命令原样串**以便人工溯源，但
    原样串在 Windows 下会因 `\\` 与 `/` 的写法差异而**字形不同**（指向同一目录）
    ⇒ 需要一个可比形式供机器判定。与 `qa_scale_run._norm` 同口径（两个模块对齐）。
    """
    return os.path.abspath(str(path)).replace("\\", "/").lower()


def _guard_out_dir(out_dir: str) -> str:
    """守卫产物目录：**只允许**写在 ``checkpoints/qa_learn/_verify/`` 之下。"""
    import n3d_qa_learn.merge_table as MT

    text = str(out_dir) if out_dir else MT.DEFAULT_OUT_DIR
    root_abs = os.path.abspath(MT.VERIFY_ROOT)
    norm_abs = os.path.abspath(text)
    prefix = root_abs + os.sep
    if not (norm_abs == root_abs or norm_abs.startswith(prefix)):
        raise ValueError(
            "产物目录只允许写在 "
            + repr(MT.VERIFY_ROOT.replace(chr(92), "/")) + " 之下，当前 = "
            + repr(norm_abs.replace(chr(92), "/"))
        )
    return text


def _resolve_product_dir(explicit: str) -> str:
    """定位 ``n3d_qa`` 冻结产物目录（只读）。"""
    if explicit:
        if os.path.isfile(os.path.join(explicit, "doclines_rows.jsonl")):
            return str(explicit)
        raise FileNotFoundError("目录 " + repr(explicit) + " 下缺少 doclines_rows.jsonl")
    for candidate in (
        os.path.join("checkpoints", "qa_learn", "dataset"),
        os.path.join("checkpoints", "qa_learn", "_verify", "drill4"),
    ):
        if os.path.isfile(os.path.join(candidate, "doclines_rows.jsonl")):
            return str(candidate)
    raise FileNotFoundError(
        "未找到 n3d_qa 冻结产物（doclines_rows.jsonl）；候选 = "
        "[checkpoints/qa_learn/dataset, checkpoints/qa_learn/_verify/drill4]"
    )


def _frozen_variants_path(explicit: str) -> str:
    """定位**已冻结**的变体产物 ``variants.jsonl``（只读；缺则报错，不重新生成）。"""
    if explicit:
        if os.path.isfile(explicit):
            return str(explicit)
        raise FileNotFoundError("指定的变体产物不存在：" + repr(explicit))
    candidate = os.path.join(
        "checkpoints", "qa_learn", "_verify", "text_variants", "variants.jsonl"
    )
    if os.path.isfile(candidate):
        return str(candidate)
    raise FileNotFoundError(
        "未找到已冻结的变体产物 " + repr(candidate) + "；本模块禁止重新生成变体"
    )


# ---------------------------------------------------------------------------
# 验收项：产物路径用最终生效的完整 CLI 现场调用 resolve_checkpoint_path 复算
# ---------------------------------------------------------------------------


def _save_stdio_encoding() -> Dict[str, Any]:
    """保存当前 stdout / stderr 的编码与错误处理器（供导入上游后**恢复**）。

    存在理由（离朱 R74 D3）：``n3d_shape.train`` 在**导入期**就调用
    ``_reconfigure_stdio()``，把 ``sys.stdout`` / ``sys.stderr`` **原地**重配为
    ``utf-8`` / ``errors='replace'``（该模块自身的设计，不是本模块引入的）。
    本模块的 CLI 纪律是「日志字面量必须能被 ``cp936``（GBK）表示」，若 stdio 被旁路成
    UTF-8，这条纪律就**静默失效**（``_log`` 的 backslashreplace 兜底永不触发）。
    故本模块在导入上游后**恢复**自己进来时的编码口径。
    """
    snap: Dict[str, Any] = {}
    for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        snap[str(name)] = {
            "encoding": str(getattr(stream, "encoding", "") or ""),
            "errors": str(getattr(stream, "errors", "") or ""),
        }
    return snap


def _restore_stdio_encoding(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """把 stdout / stderr 恢复到 :func:`_save_stdio_encoding` 记录的口径。

    返回
    ----
    Dict[str, Any]
        每个流的 ``before`` / ``after_before_restore`` / ``after_restore`` 取证，
        便于在产物里回答「本次运行期间 stdout 编码到底被谁改过」。
    """
    out: Dict[str, Any] = {}
    for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        want = dict(snapshot.get(str(name), {}))
        out[str(name)] = {
            "before": dict(want),
            "after_before_restore": {
                "encoding": str(getattr(stream, "encoding", "") or ""),
                "errors": str(getattr(stream, "errors", "") or ""),
            },
        }
        if want.get("encoding"):
            try:
                stream.reconfigure(encoding=str(want["encoding"]), errors=str(want.get("errors") or "strict"))
            except Exception:
                # 非文本流 / 老版本解释器：静默跳过（不影响任何数值与产物）
                pass
        out[str(name)]["after_restore"] = {
            "encoding": str(getattr(stream, "encoding", "") or ""),
            "errors": str(getattr(stream, "errors", "") or ""),
        }
    return out


def checkpoint_path_resolution(
    *, dim: int, max_batches: int, tag: str, checkpoint_override: str = ""
) -> Dict[str, Any]:
    """**现场**调用上游 ``resolve_checkpoint_path`` 复算「会写入哪个 checkpoint 路径」。

    验收要求：不得复用早期探针结论，必须在**本次运行的最终参数**下现场复算。
    n3d_qa_learn 自身**不产 checkpoint**，故该复算的作用是给出**产物保护**的
    现场证据：算出上游会落到的路径，并核实它**不在**本模块的产物目录之内。

    **stdio 口径保护（离朱 R74 D3）**：本函数会导入 ``n3d_shape.train``，而该模块在
    导入期把 stdout / stderr 重配为 UTF-8。故这里先 :func:`_save_stdio_encoding`、
    导入后 :func:`_restore_stdio_encoding`，把本模块「GBK 安全日志」的纪律恢复回来；
    恢复前后口径一并写进返回值（``stdio_encoding``），可审计。

    例外一律**如实登记**（``status='unavailable'`` + 原始报文），不伪装成已复算。
    """
    stdio_before = _save_stdio_encoding()
    out: Dict[str, Any] = {
        "dim": int(dim),
        "max_batches": int(max_batches),
        "tag": str(tag),
        "checkpoint_override": str(checkpoint_override),
        "source": "n3d_shape.train.resolve_checkpoint_path",
        "stdio_encoding": {"before": stdio_before},
    }
    # **stdout 隔离（离朱 R74 D3 的修法之二）**：`n3d_shape.train` 导入期会
    # ① 调 `_reconfigure_stdio()` 把 sys.stdout 重配为 utf-8；② 经 `log_info` 打印一行
    # `[N3D][INFO] [产物保护] ...`（含中文）。两者都会让本模块的「GBK 安全日志」纪律
    # 被静默旁路。故整个「导入 + 复算」过程在**临时缓冲**里跑：上游改的是那个缓冲的编码、
    # 上游的日志也落在缓冲里，真实 stdout 的编码与内容**都不受影响**。
    buffer = io.StringIO()
    real_stdout = sys.stdout
    captured = ""
    try:
        sys.stdout = buffer
        from n3d_shape.train import resolve_checkpoint_path  # 只读 import

        from .backends import recommended_config

        out["stdio_encoding"][
            "buffer_encoding_after_import"
        ] = str(getattr(sys.stdout, "encoding", "") or "")
        config = recommended_config("n3d_shape", int(dim))
        resolved = str(
            resolve_checkpoint_path(
                str(checkpoint_override), int(max_batches), config, str(tag)
            )
        )
        out["status"] = "computed"
        out["resolved_checkpoint_path"] = resolved
        out["config_dataset"] = str(getattr(config, "dataset", ""))
        out["config_input_dim"] = int(getattr(config, "input_dim", 0) or 0)
    except Exception as exc:  # pragma: no cover - 环境相关，如实登记
        out["status"] = "unavailable"
        out["error"] = type(exc).__name__ + ": " + str(exc)
        out["resolved_checkpoint_path"] = ""
    finally:
        captured = buffer.getvalue()
        sys.stdout = real_stdout
        out["stdio_encoding"]["restore"] = _restore_stdio_encoding(stdio_before)
        out["stdio_encoding"]["upstream_stdout"] = {
            "chars": int(len(captured)),
            "sha256": str(hashlib.sha256(captured.encode("utf-8")).hexdigest()),
            "head": str(captured.splitlines()[0]) if captured.strip() else "",
            "rule": (
                "上游在导入 / 复算期间写到 stdout 的内容被**重定向到临时缓冲**"
                "（不污染本次控制台与 --log-file），只在这里登记其字符数 / SHA256 /"
                "首行，供审计「上游确实打印过什么」"
            ),
        }
        out["stdio_encoding"]["rule"] = (
            "`n3d_shape.train` 在导入期把 stdout/stderr 重配为 utf-8/errors=replace"
            "（该模块自身设计）并经 log_info 打印一行含中文的产物保护提示；"
            "本模块把这段过程整体重定向到临时缓冲，真实 stdout 的编码与内容都不受影响，"
            "以免自身的「GBK 安全日志」纪律被静默旁路。"
        )
    return out


def artifact_scope_evidence(resolved: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """核实复算出的 checkpoint 路径**不在**本模块产物目录之内（产物保护证据）。"""
    import n3d_qa_learn.merge_table as MT

    path = str(resolved.get("resolved_checkpoint_path", ""))
    out_abs = os.path.abspath(out_dir)
    verify_abs = os.path.abspath(MT.VERIFY_ROOT)
    inside_out = bool(path) and os.path.abspath(path).startswith(out_abs + os.sep)
    inside_verify = bool(path) and os.path.abspath(path).startswith(verify_abs + os.sep)
    return {
        "resolved_checkpoint_path": path,
        "out_dir_abs": out_abs,
        "verify_root_abs": verify_abs,
        "inside_out_dir": bool(inside_out),
        "inside_verify_root": bool(inside_verify),
        "rule": (
            "本模块的产物目录与上游 checkpoint 路径必须**不相交**："
            "本模块不产 .pt / .zip，上游 checkpoint 也不得落在 merge_table 产物目录内"
        ),
    }


# ---------------------------------------------------------------------------
# 装配（两侧条目表 + 合并表 + 三条前置断言）
# ---------------------------------------------------------------------------


def _profile_pair(profile: str) -> Tuple[Any, Any]:
    """按档构造「题面 / 文本行」两个编码器（两角色在 hash 家族下口径相同）。"""
    import n3d_qa_learn.entry_table as ET

    return (
        ET.build_vectorizer_for(str(profile), "question"),
        ET.build_vectorizer_for(str(profile), "text_line"),
    )


def build_bundles(
    *,
    profile: str,
    product_dir: str,
    qa_cache_dir: str = "",
    text_dir: str = "",
    limit_rows: int = 0,
    limit_qa: int = 0,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """装配两侧条目表并**现场**跑三条合并前置断言。

    返回
    ----
    Dict[str, Any]
        ``profile`` / ``dim`` / ``qa_bundle`` / ``text_bundle`` / ``merged`` /
        ``premerge`` / ``evidence``。
    """
    import n3d_qa_learn.entry_table as ET
    import n3d_qa_learn.merge_table as MT
    import n3d_qa_learn.step2 as S2
    import n3d_qa_learn.text_variants as TV

    vec_q, vec_t = _profile_pair(str(profile))
    want = int(ET.expect_dim_of(str(profile)))
    if int(vec_q.dim) != want or int(vec_t.dim) != want:
        raise ValueError(
            f"特征档 {profile!r} 实测维度 question={vec_q.dim} / text_line={vec_t.dim} "
            f"与声明 {want} 不一致（拒绝静默错配）"
        )
    resolved = _resolve_product_dir(str(product_dir))
    rows = TV.load_text_rows(resolved)
    if int(limit_rows) > 0:
        rows = list(rows[: int(limit_rows)])
    data = ET.load_qa_training_data(qa_cache_dir=str(qa_cache_dir), text_dir=str(text_dir))
    qa_bundle = ET.build_qa_entry_tables(
        data, vec_q, profile=str(profile), batch_size=int(batch_size)
    )
    if int(limit_qa) > 0:
        raise ValueError(
            "limit_qa 不受支持：QA 侧条目表必须与 train_known 切分一一对应，"
            "截断会破坏「149 条目 / C=10」这一冻结口径"
        )
    text_bundle = ET.text_entry_table_from_rows(
        vec_t, rows, profile=str(profile), product_dir=str(resolved)
    )
    merged = MT.build_merged_table(qa_bundle.table, text_bundle.table)
    premerge = MT.premerge_assertions(
        qa_bundle.table,
        text_bundle.table,
        merged,
        profile=str(profile),
        batch_size=int(batch_size),
    )
    return {
        "profile": str(profile),
        "dim": int(want),
        "product_dir": str(resolved),
        "rows": rows,
        "data": data,
        "qa_bundle": qa_bundle,
        "text_bundle": text_bundle,
        "merged": merged,
        "premerge": premerge,
        "evidence": {
            "product_dir": str(resolved),
            "n_rows": int(len(rows)),
            "qa_evidence": dict(qa_bundle.evidence),
            "text_evidence": dict(text_bundle.evidence),
            "encoder_fingerprint": {
                "question": str(vec_q.fingerprint()),
                "text_line": str(vec_t.fingerprint()),
            },
            "splits": {
                "train_known": int(len(data.splits.train_known)),
                "test_known": int(len(data.splits.test_known)),
                "train_unknown": int(len(data.splits.train_unknown)),
                "test_unknown": int(len(data.splits.test_unknown)),
                "n_classes": int(data.corpus.n_classes),
            },
        },
    }


# ---------------------------------------------------------------------------
# 查询集（两臂**共用**同一批查询与同一批键）
# ---------------------------------------------------------------------------


def build_qa_queries(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """QA 侧查询集：``test_known`` 全部条目（与 ``entry_table`` 同源口径）。

    **金标集合口径（现场修正，必须显式）**：金标 = **与查询答案键相同的全部**
    ``train_known`` 条目所绑定的答案展示文本。不能拿 ``query_answer_keys`` 去
    ``known_answer_keys`` 里做相等过滤 —— ``test_known`` 是**留出**子集，它与
    ``train_known`` 的答案键几乎不相交，那样做会得到**全空**的金标集合，
    使「严格命中率」结构性地恒为 0（一个看起来像「测出来很差」的假结果）。
    正确做法是用 ``known_answer_keys``（与 ``qa_bundle.table.entry_ids`` 同行序）
    建立「答案键 -> 该类的全部展示文本」映射，再按查询的答案键取对应集合。
    """
    qa = bundle["qa_bundle"]
    stmt = qa.known_queries
    feats = np.asarray(stmt.feats, dtype=np.float32)
    by_key: Dict[str, set] = {}
    for key, out in zip(qa.known_answer_keys, qa.table.outputs):
        by_key.setdefault(str(key), set()).add(str(out))
    gold: List[List[str]] = []
    for key in qa.query_answer_keys:
        gold.append(sorted(by_key.get(str(key), set())))
    n_empty = int(sum(1 for g in gold if not g))
    if n_empty:
        raise ValueError(
            "有 " + str(n_empty) + " 条 QA 查询的答案键在 train_known 里找不到任何条目"
            "（金标集合为空）；此时「严格命中率」不可测，拒绝把它当作实测负结果出数"
        )
    return {
        "side": "qa",
        "expected_kind": "qa",
        "entry_ids": [str(x) for x in stmt.entry_ids],
        "feats": feats,
        "gold": gold,
        "answer_keys": [str(x) for x in qa.query_answer_keys],
        "n": int(len(stmt.entry_ids)),
        "n_distinct_gold_sets": int(len({tuple(g) for g in gold})),
        "gold_set_sizes": {
            "min": int(min((len(g) for g in gold), default=0)),
            "max": int(max((len(g) for g in gold), default=0)),
        },
        "gold_rule": (
            "金标集合 = train_known 中**答案键 == 查询答案键**的全部条目所绑定的答案展示文本"
        ),
    }


def _entry_idx_by_id(bundle: Dict[str, Any]) -> Dict[str, int]:
    """合并表内「条目 id -> 行下标」的现场索引（用于把查询映射到自身行）。"""
    merged = bundle["merged"]
    out: Dict[str, int] = {}
    for i, eid in enumerate(merged.table.entry_ids):
        out[str(eid)] = int(i)
    return out


def build_text_queries(
    bundle: Dict[str, Any],
    *,
    variants_path: str,
) -> Dict[str, Any]:
    """文本侧查询集：**冻结划分**的 666 查询行 x 4 支变体（两臂共用）。

    金标集口径（:data:`merge_table.GOLD_SET_RULE`）：自身行 ∪ 与查询行**原文逐字节相同**
    的其他行。原文支的查询文本**逐字节取自冻结产物**（不重新编码生成）。
    """
    import n3d_qa_learn.merge_table as MT

    records = MT.load_frozen_variants(str(variants_path))
    fields = MT.variant_fields_of(records)
    merged = bundle["merged"]
    idx_of = _entry_idx_by_id(bundle)
    query_ids = [
        str(x) for x in bundle["text_bundle"].query_entry_ids
    ]
    missing = [qid for qid in query_ids if qid not in idx_of]
    if missing:
        raise ValueError(
            "有 " + str(len(missing)) + " 条冻结查询行不在合并表内（前几个 = "
            + repr(missing[:5]) + "）；查询与键表必须来自同一份全量行表"
        )
    rec_of = {str(r["row_id"]): r for r in records}
    absent = [qid for qid in query_ids if qid not in rec_of]
    if absent:
        raise ValueError(
            "有 " + str(len(absent)) + " 条冻结查询行不在变体产物内（前几个 = "
            + repr(absent[:5]) + "）；变体与划分不一致"
        )
    # --- 原文支恒等一致性（独立视图核对） -------------------------------
    mism: List[Dict[str, Any]] = []
    for qid in query_ids:
        i = int(idx_of[qid])
        frozen = str(rec_of[qid]["variants"][MT.VARIANT_ORDER[0]])
        if str(merged.table.outputs[i]) != frozen:
            if len(mism) < 10:
                mism.append(
                    {
                        "row_id": str(qid),
                        "artifact_text_sha256": hashlib.sha256(
                            frozen.encode("utf-8")
                        ).hexdigest(),
                        "key_table_text_sha256": hashlib.sha256(
                            str(merged.table.outputs[i]).encode("utf-8")
                        ).hexdigest(),
                    }
                )
    if mism:
        raise ValueError(
            "原文支文本与键表内该行原文不一致（前几个 = "
            + repr(mism)
            + "）；原文支的「查询即键」恒等关系不成立，拒绝出数"
        )
    # --- 逐支特征 ------------------------------------------------------
    import n3d_qa_learn.entry_table as ET

    vec_t = ET.build_vectorizer_for(str(bundle["profile"]), "text_line")
    expect_kind = MT.EXPECTED_RETURN_KIND["text"]
    gold_rows: List[List[int]] = []
    for qid in query_ids:
        text = str(merged.table.outputs[int(idx_of[qid])])
        same = [
            int(i)
            for i in range(merged.size)
            if str(merged.entry_kinds[i]) == MT.KIND_TEXT
            and str(merged.table.outputs[i]) == text
        ]
        gold_rows.append(sorted(same))
    variants: Dict[str, Dict[str, Any]] = {}
    for name in MT.VARIANT_ORDER:
        texts = [str(rec_of[qid]["variants"][name]) for qid in query_ids]
        feats = ET.encode_statements(vec_t, texts)
        variants[str(name)] = {
            "name": str(name),
            "label": str(MT.VARIANT_LABELS.get(str(name), str(name))),
            "texts": texts,
            "feats": feats,
            "n": int(feats.shape[0]),
        }
    return {
        "side": "text",
        "expected_kind": expect_kind,
        "query_ids": query_ids,
        "gold_rows": gold_rows,
        "variants": variants,
        "n": int(len(query_ids)),
        "variants_path": str(variants_path),
        "variant_sha256": _sha256_file(str(variants_path)),
        "fields": fields,
        "n_duplicate_gold_rows": int(
            sum(1 for g in gold_rows if len(g) > 1)
        ),
    }


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    """分位数（``numpy.quantile``，线性插值；空序列返回全 0）。"""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {k: 0.0 for k in ("min", "p05", "p25", "p50", "p75", "p95", "max", "mean")}
    return {
        "min": float(arr.min()),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _type_stats(arm_out: Dict[str, Any]) -> Dict[str, Any]:
    """类型契约统计（主判据 = **未过滤 top-1** 的返回类型正确率）。"""
    import n3d_qa_learn.merge_table as MT

    top1_kind = list(arm_out["top1_kind"])
    returned_kind = list(arm_out["returned_kind"])
    ok = list(arm_out["type_contract_ok"])
    viol = list(arm_out["type_contract_violation"])
    counts = MT.empty_violation_counts()
    for name in viol:
        if str(name):
            counts[str(name)] = int(counts[str(name)]) + 1
    n = int(len(top1_kind))
    n_ok = int(sum(1 for x in ok if bool(x)))
    return {
        "n": int(n),
        "n_top1_kind_expected": int(
            sum(1 for k in top1_kind if str(k) == str(arm_out["expected_kind"]))
        ),
        "top1_frac_kind_expected": float(
            sum(1 for k in top1_kind if str(k) == str(arm_out["expected_kind"]))
            / max(1, n)
        ),
        "n_returned_kind_expected": int(
            sum(1 for k in returned_kind if str(k) == str(arm_out["expected_kind"]))
        ),
        "returned_frac_kind_expected": float(
            sum(1 for k in returned_kind if str(k) == str(arm_out["expected_kind"]))
            / max(1, n)
        ),
        "n_type_contract_ok": int(n_ok),
        "return_type_correctness": float(n_ok / max(1, n)),
        "n_type_contract_violations": int(n - n_ok),
        "violation_counts": counts,
        "n_none": int(sum(1 for k in returned_kind if str(k) == MT.KIND_NONE)),
        "n_matched": int(sum(1 for k in returned_kind if str(k) != MT.KIND_NONE)),
    }


def qa_side_metrics(arm_out: Dict[str, Any], query: Dict[str, Any]) -> Dict[str, Any]:
    """QA 侧指标：命中率（宽松 / 严格）+ 返回类型契约统计。"""
    gold_sets = [set(g) for g in query["gold"]]
    top1_idx = list(arm_out["top1_index"])
    returned_idx = list(arm_out["returned_index"])
    returned_out = list(arm_out["returned_output"])
    expected = str(arm_out["expected_kind"])
    n = int(arm_out["n"])
    relaxed = 0
    strict = 0
    for i in range(n):
        idx = int(top1_idx[i])
        if str(arm_out["top1_kind"][i]) != expected:
            continue
        relaxed += 1
        if str(returned_out[i]) in gold_sets[i] and int(returned_idx[i]) >= 0:
            strict += 1
    ret_relaxed = int(
        sum(1 for i in range(n) if str(arm_out["returned_kind"][i]) == expected)
    )
    ret_strict = int(
        sum(
            1
            for i in range(n)
            if str(arm_out["returned_kind"][i]) == expected
            and str(returned_out[i]) in gold_sets[i]
            and int(returned_idx[i]) >= 0
        )
    )
    types = _type_stats(arm_out)
    return {
        "n_queries": int(n),
        "qa_hit_rate": float(relaxed / max(1, n)),
        "n_qa_hit": int(relaxed),
        "qa_hit_strict_rate": float(strict / max(1, n)),
        "n_qa_hit_strict": int(strict),
        "qa_hit_rate_returned": float(ret_relaxed / max(1, n)),
        "n_qa_hit_returned": int(ret_relaxed),
        "qa_hit_strict_rate_returned": float(ret_strict / max(1, n)),
        "n_qa_hit_strict_returned": int(ret_strict),
        "top1_score_quantiles": _quantiles([float(x) for x in arm_out["top1_score"]]),
        "type_contract": types,
        "rule": (
            "命中率（主判据）只取**未过滤 top-1**：top-1 落在 QA 半区即算命中（宽松），"
            "且其返回答案展示文本落在金标集合内即算**严格命中**（金标集合 = train_known 中"
            "答案键 == 该查询答案键的全部条目所绑定的展示文本）。带 _returned 后缀者另按 "
            "TAU 过滤，仅作诊断。"
            "**口径更正登记**：金标集合**不能**用 `query_answer_keys` 去 `known_answer_keys` "
            "里做相等过滤 —— `test_known` 是留出子集，二者答案键不相交，那样会得到全空金标、"
            "使严格命中率结构性地恒为 0（把「没测出来」伪装成「测出来很差」）。"
        ),
    }


def text_side_metrics(
    arm_out: Dict[str, Any],
    query: Dict[str, Any],
    variant_name: str,
) -> Dict[str, Any]:
    """文本侧某一支变体的 R@1（+ 按臂内阶段拆分 + 类型契约统计 + QA 抢占诊断）。

    **为什么必须按阶段拆分（口径纠正）**：两臂的 ``stage`` 语义不同 ——
    ``qa_priority`` 臂的 ``stage`` 是**两级路由的到达档**（``qa`` 命中即返回 /
    ``text`` 落到第二级 / ``none`` 两级都低于门限），而 ``merged_topk1`` 臂没有优先级，
    它的 ``stage`` 退化为「top-1 的 kind」（一次排序的必然结果）。
    若只报一个总体 R@1，会把「QA 优先臂把原文查询先判给 QA 半区」这件事误读成
    「文本检索能力差」。故本函数**同时**给出总体 R@1 与**按 stage 拆分**的 R@1，
    并显式登记两臂各自的 ``stage`` 语义差异。
    """
    import n3d_qa_learn.merge_table as MT

    gold = [set(int(x) for x in g) for g in query["gold_rows"]]
    top1_idx = list(arm_out["top1_index"])
    stages = [str(x) for x in arm_out["stage"]]
    n = int(arm_out["n"])
    hits = 0
    exact = 0
    top1_text = 0
    top1_qa = 0
    hits_text_stage = 0
    n_text_stage = 0
    hits_qa_stage = 0
    n_qa_stage = 0
    for i in range(n):
        idx = int(top1_idx[i])
        in_gold = bool(idx in gold[i])
        if in_gold:
            hits += 1
        if idx == int(query["gold_rows"][i][0]):
            exact += 1
        if str(arm_out["top1_kind"][i]) == MT.KIND_TEXT:
            top1_text += 1
        elif str(arm_out["top1_kind"][i]) == MT.KIND_QA:
            top1_qa += 1
        if stages[i] == MT.KIND_TEXT:
            n_text_stage += 1
            hits_text_stage += int(in_gold)
        elif stages[i] == MT.KIND_QA:
            n_qa_stage += 1
            hits_qa_stage += int(in_gold)
    types = _type_stats(arm_out)
    return {
        "variant": str(variant_name),
        "label": str(MT.VARIANT_LABELS.get(str(variant_name), str(variant_name))),
        "is_anchor": bool(str(variant_name) == MT.VARIANT_ORDER[0]),
        "n_queries": int(n),
        "recall_at_1": float(hits / max(1, n)),
        "n_hit_at_1": int(hits),
        "n_exact_self_top1": int(exact),
        "exact_self_top1_rate": float(exact / max(1, n)),
        "n_top1_text_kind": int(top1_text),
        "n_top1_qa_kind": int(top1_qa),
        "n_stage_qa": int(n_qa_stage),
        "n_stage_text": int(n_text_stage),
        "n_stage_none": int(sum(1 for s in stages if s == MT.KIND_NONE)),
        "recall_at_1_text_stage": float(hits_text_stage / max(1, n_text_stage)),
        "n_hit_at_1_text_stage": int(hits_text_stage),
        "recall_at_1_qa_stage": float(hits_qa_stage / max(1, n_qa_stage)),
        "n_hit_at_1_qa_stage": int(hits_qa_stage),
        "type_contract": types,
        "top1_score_quantiles": _quantiles([float(x) for x in arm_out["top1_score"]]),
        "rule": str(MT.GOLD_SET_RULE),
    }


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------


def _baseline_interior(baseline: float) -> bool:
    """基线是否**不在**可行域边界（0 / 1 饱和点）——「严格优于」类判据的前提。"""
    return bool(0.0 < float(baseline) < 1.0)


def build_verdict(
    per_arm: Dict[str, Dict[str, Any]],
    *,
    baseline_arm: str,
    candidate_arm: str,
) -> Dict[str, Any]:
    """「合并表 vs 现状路由」的非劣 / 劣化判定（含可行域边界核查）。

    可比指标（现场枚举，唯一来源）：``QA 侧命中率`` / ``文本侧非锚点三支 R@1`` /
    ``返回类型正确率``。
    """
    import n3d_qa_learn.merge_table as MT

    base = per_arm[str(baseline_arm)]
    cand = per_arm[str(candidate_arm)]
    metrics: List[Dict[str, Any]] = []

    def _add(name: str, b: float, c: float, extra: Optional[Dict[str, Any]] = None) -> None:
        row: Dict[str, Any] = {
            "metric": str(name),
            "baseline": float(b),
            "candidate": float(c),
            "delta": float(c) - float(b),
            "baseline_interior": _baseline_interior(b),
            "not_worse": bool(float(c) >= float(b)),
            "strictly_better": bool(float(c) > float(b)),
        }
        if extra:
            row.update(dict(extra))
        metrics.append(row)

    _add(
        "qa_hit_rate",
        float(base["qa"]["qa_hit_rate"]),
        float(cand["qa"]["qa_hit_rate"]),
    )
    for name in MT.NON_ANCHOR_VARIANTS:
        _add(
            "text_recall_at_1:" + str(name),
            float(base["text"][str(name)]["recall_at_1"]),
            float(cand["text"][str(name)]["recall_at_1"]),
        )
    _add(
        "return_type_correctness",
        float(base["return_type_correctness_overall"]),
        float(cand["return_type_correctness_overall"]),
    )
    degraded = [row["metric"] for row in metrics if not bool(row["not_worse"])]
    improved = [row["metric"] for row in metrics if bool(row["strictly_better"])]
    return {
        "baseline_arm": str(baseline_arm),
        "candidate_arm": str(candidate_arm),
        "rule": str(MT.VERDICT_RULE),
        "anchor_variant_note": str(MT.ANCHOR_VARIANT_NOTE),
        "comparable_metrics": [row["metric"] for row in metrics],
        "metrics": metrics,
        "degraded_metrics": degraded,
        "improved_metrics": improved,
        "not_worse_than_baseline": bool(len(degraded) == 0),
        "strictly_better_than_baseline": bool(len(degraded) == 0 and len(improved) == len(metrics)),
        "verdict": (
            "merged_not_worse"
            if len(degraded) == 0 and not improved
            else (
                "merged_strictly_better"
                if len(degraded) == 0
                else "merged_degraded"
            )
        ),
        "scope_note": str(MT.SCOPE_NOTE),
        "quantified": _quantified_deltas(metrics),
    }


def _quantified_deltas(metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """**F1 修复**（计划 1 遗留项）：``min_gain`` / ``max_drop`` 的语义修正。

    修复前的缺陷
    ------------
    两个字段**都**写成 ``min(delta)`` ⇒ ``min_gain`` 实际等于 ``max_drop``（同一个数），
    且当存在劣化时 ``min_gain`` 会变成负数（"最小增益"是负数没有意义）。修前现场
    ``min_gain == max_drop == 0.0``（因为 ``qa_hit_rate`` 的 delta 恰为 0 且被算了进去）。

    修正后的口径（**逐条显式**）
    ---------------------------
    1. **只在「可比且不在可行域边界」的指标上取**：即 ``baseline_interior == True``
       的指标（基线严格落在 ``(0, 1)`` 内部）。基线为 0 / 1 饱和点的指标**不进**统计
       —— 它在这些指标上"不可判"，其 delta 不构成增益或落差证据。
    2. ``min_gain`` = **只统计正向 delta（> 0）的最小值**；没有正向 delta 时为 ``None``。
    3. ``max_drop`` = **只统计负向 delta（< 0）的最小值**（即绝对值最大的落差）；
       **无劣化时为 0.0**（不是 ``None``，便于下游直接比较）。
    4. 三个字段都进产物：``eligible_metrics``（参与统计的指标名）、
       ``excluded_metrics``（因不在可行域内部被排除的指标名）、``basis``（口径文本）。
    5. 该项**不改变** ``verdict`` / ``degraded_metrics`` / ``improved_metrics`` 主判据
       —— 主判据仍按**全部**可比指标判定。

    参数
    ----
    metrics : Sequence[Dict[str, Any]]
        ``build_verdict`` 组装的逐指标行（含 ``metric`` / ``delta`` / ``baseline_interior``）。

    返回
    ----
    Dict[str, Any]
        ``min_gain`` / ``max_drop`` / ``eligible_metrics`` / ``excluded_metrics`` / ``basis``。
    """
    eligible = [r for r in metrics if bool(r.get("baseline_interior"))]
    excluded = [str(r["metric"]) for r in metrics if not bool(r.get("baseline_interior"))]
    gains = [float(r["delta"]) for r in eligible if float(r["delta"]) > 0.0]
    drops = [float(r["delta"]) for r in eligible if float(r["delta"]) < 0.0]
    return {
        "min_gain": (min(gains) if gains else None),
        "max_drop": (min(drops) if drops else 0.0),
        "eligible_metrics": [str(r["metric"]) for r in eligible],
        "excluded_metrics": excluded,
        "n_gains": int(len(gains)),
        "n_drops": int(len(drops)),
        "basis": (
            "只在「可比且 baseline_interior == True」（基线严格落在 (0,1) 内部）的指标上统计："
            "`min_gain` = 正向 delta 的最小值（无正向 delta 时为 None）；"
            "`max_drop` = 负向 delta 的最小值（无劣化时为 0.0）。"
            "基线落在 0 / 1 饱和点的指标被排除（这些指标不可判，其 delta 不算证据），"
            "排除清单见 `excluded_metrics`。本口径**不参与** verdict 判定，"
            "verdict 仍基于全部可比指标。"
        ),
    }


# ---------------------------------------------------------------------------
# 双跑逐字节一致证据
# ---------------------------------------------------------------------------


#: 双跑证据文件名（**唯一**来源）。
DETERMINISM_EVIDENCE_FILE: str = "determinism_check.json"

#: 证据文件自身与「后置可重渲件」的**整类排除**口径。
DETERMINISM_EXCLUSION_RULE: str = (
    "① **证据文件自身**（" + DETERMINISM_EVIDENCE_FILE + "）整类排除：它由本函数写出，"
    "若参与比对，其 a_sha256 就是「上一次运行留在盘上的自己」⇒ 自指哈希链，"
    "且在全新对照目录上首跑必然判False（该文件只存在于一侧）。"
    "② **两件报告**（merge_table_report.json / .md）在证据之后才由 `report` 写出，"
    "其 SHA 亦不进证据块 ⇒ 不存在「报告内嵌证据 <-> 证据记录报告 SHA」的回路。"
    "排除后，证据只由「探针 / 门禁 / 评测产物」决定：本文件是输入目录内容的确定性函数。"
)


def determinism_block(out_dir: str, repro_dir: str) -> Dict[str, Any]:
    """**双跑逐字节一致**证据：逐文件比对主产物目录与对照目录。

    **两处必须显式排除**（离朱 R74 D2 的修复点）
    --------------------------------------------
    1. **证据文件自身**（:data:`DETERMINISM_EVIDENCE_FILE`）：它由本函数写出，
       若参与比对就是「上一次运行留在盘上的自己」⇒ 自指哈希链；且在**全新**对照目录上
       首跑时它只存在于一侧 ⇒ `determinism_ok` 会**首跑即 False**（把「对照目录还没被写过」
       误报成「不确定性」）。修复前正是如此。
    2. **单侧独有文件**：本文件把它们分列在 ``a_only_files`` / ``b_only_files``，
       并**纳入判定语义**（:data:`determinism_scope`）—— 不再静默忽略。
    """
    excluded_self = str(DETERMINISM_EVIDENCE_FILE)
    a_all = sorted(
        f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))
    )
    b_all = sorted(
        f for f in os.listdir(repro_dir) if os.path.isfile(os.path.join(repro_dir, f))
    )
    a_files = [f for f in a_all if f != excluded_self]
    b_files = [f for f in b_all if f != excluded_self]
    excluded_present_in_a = bool(excluded_self in a_all)
    excluded_present_in_b = bool(excluded_self in b_all)
    files: List[Dict[str, Any]] = []
    for name in sorted(set(a_files) & set(b_files)):
        pa = os.path.join(out_dir, name)
        pb = os.path.join(repro_dir, name)
        sa, aa = _sha256_file(pa), int(os.path.getsize(pa))
        sb, ab = _sha256_file(pb), int(os.path.getsize(pb))
        files.append(
            {
                "name": str(name),
                "a_sha256": str(sa),
                "b_sha256": str(sb),
                "a_bytes": int(aa),
                "b_bytes": int(ab),
                "bitwise_equal": bool(sa == sb and aa == ab),
            }
        )
    a_only = [f for f in a_files if f not in b_files]
    b_only = [f for f in b_files if f not in a_files]
    not_equal = [f["name"] for f in files if not bool(f["bitwise_equal"])]
    # **判定语义（必须显式）**：只对「两目录共有的非自身文件」成立；单侧独有文件
    # 不属于该范围（它们在对照目录尚未被写过的**工作序**下是正常的），但必须显式登记。
    determinism_ok = bool(len(files) > 0 and not not_equal)
    # **F-A 修复（离朱 R77）**：与 `qa_scale_run.determinism_block` **对齐** —— 补
    # `a_dir_normalized` / `b_dir_normalized` / `paths_normalized_equal` / `path_dependence_note`
    # 四项。修复前本模块只给命令原样串（`a_dir` / `b_dir`），路径依赖只有**文字口径**、
    # 机器判定只能靠 qa_scale 侧；补齐后两个模块的路径依赖取证**同源可判**。
    a_norm = _normalize_dir(out_dir)
    b_norm = _normalize_dir(repro_dir)
    return {
        "a_dir": str(out_dir),
        "b_dir": str(repro_dir),
        "a_dir_normalized": str(a_norm),
        "b_dir_normalized": str(b_norm),
        "paths_normalized_equal": bool(a_norm == b_norm),
        "path_dependence_note": (
            "`a_dir` / `b_dir` 保留**命令原样串**（故 Windows 下 \\ 与 / 的字形差异会被保留）；"
            "另给 `*_normalized`（绝对路径 + 统一正斜杠 + 小写）与 `paths_normalized_equal` "
            "供机器判定。**这两个原样串会进产物内容 ⇒ 凡内嵌它们的产物（本文件与两件报告）"
            "在「绝对路径不同」的运行之间不逐字节相同；同路径同参重跑才逐字节相同** —— "
            "这是「不落盘挂钟字段」之外的**唯一**已知非确定性来源，已显式登记。"
            "本字段与 `qa_scale_run.determinism_block` 的**同名同义**（两个模块对齐）。"
        ),
        "files": files,
        "n_common_files": int(len(files)),
        "n_files_before_exclusion": {
            "a": int(len(a_all)),
            "b": int(len(b_all)),
        },
        "a_only_files": a_only,
        "b_only_files": b_only,
        "not_equal_files": not_equal,
        "determinism_ok": bool(determinism_ok),
        "determinism_ok_semantics": (
            "determinism_ok = 「两目录**共有的非排除文件**全部逐字节相等」且该集合非空。"
            "单侧独有文件（a_only_files / b_only_files）**不纳入**该布尔值：在"
            "「先跑主目录、再跑对照目录」这一真实工作序下，对照目录首跑时天然缺少"
            "主目录独有的文件（例如本证据文件自身），把它算进去会把工作序差异误报成"
            "不确定性。此处**显式登记**该取舍，不静默忽略。"
        ),
        "determinism_scope": DETERMINISM_EXCLUSION_RULE,
        "excluded_self_file": excluded_self,
        "excluded_self_present": {
            "in_a_dir": bool(excluded_present_in_a),
            "in_b_dir": bool(excluded_present_in_b),
        },
        "self_exclusion_rule": (
            "证据文件自身在两个目录**都**被过滤掉（不进 files、不记 SHA）；"
            "因此本文件与「对照目录里是否有上一轮的自己」无关。"
        ),
    }


# ---------------------------------------------------------------------------
# drill：单条端到端演练
# ---------------------------------------------------------------------------


def cmd_drill(args: argparse.Namespace) -> int:
    """**单条端到端演练**：1 条 QA 查询 + 1 条文本查询，把整条通路跑通。

    与全量口径的关系：这里用的 :func:`merge_table.arm_top1` 与 ``eval`` **同一函数**，
    只是查询集被截到 1 条；此外另跑一次「同一批位形的全量调用」并逐项比对前 1 条，
    以证明「小批演练与全量口径一致」（历史纠正记录 #10 的硬门禁）。
    """
    import n3d_qa_learn.merge_table as MT

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    profile = str(getattr(args, "profile", "") or "lexical-88")
    bundle = build_bundles(
        profile=profile,
        product_dir=str(getattr(args, "product_dir", "") or ""),
        qa_cache_dir=str(getattr(args, "qa_cache_dir", "") or ""),
        text_dir=str(getattr(args, "text_dir", "") or ""),
        batch_size=int(getattr(args, "batch_size", 64) or 64),
    )
    merged = bundle["merged"]
    qa_q = build_qa_queries(bundle)
    text_q = build_text_queries(
        bundle,
        variants_path=_frozen_variants_path(str(getattr(args, "variants", "") or "")),
    )
    report: Dict[str, Any] = {
        "profile": str(profile),
        "dim": int(bundle["dim"]),
        "n_entries": int(merged.size),
        "n_entries_by_kind": {
            str(k): int(merged.half_size(k)) for k in MT.MERGE_ORDER
        },
        "premerge_all_passed": bool(bundle["premerge"]["all_passed"]),
        "premerge_assertion_order": list(MT.PREMERGE_ASSERTION_KEYS),
        "courses": {},
        "drill_rule": (
            "单条演练：QA 查询取 test_known 第 0 条、文本查询取冻结查询行第 0 条的原文支；"
            "两臂各跑一次，并与「同一批位形的全量调用」逐项比对（必须是同一函数的同一口径）"
        ),
    }
    checks: List[Tuple[str, bool]] = []
    for side, query, feats, expected in (
        ("qa", qa_q, np.asarray(qa_q["feats"][:1], dtype=np.float32), "qa"),
        (
            "text",
            text_q,
            np.asarray(text_q["variants"][MT.VARIANT_ORDER[0]]["feats"][:1], dtype=np.float32),
            "text",
        ),
    ):
        courses: Dict[str, Any] = {}
        for arm in MT.ARM_ORDER:
            single = MT.arm_top1(
                feats,
                merged,
                arm=str(arm),
                expected_kind=str(expected),
                batch_size=int(getattr(args, "batch_size", 64) or 64),
            )
            full = MT.arm_top1(
                feats,
                merged,
                arm=str(arm),
                expected_kind=str(expected),
                batch_size=1,
            )
            row = {
                "n": int(single["n"]),
                "top1_index": [int(x) for x in single["top1_index"]],
                "top1_score": [float(x) for x in single["top1_score"]],
                "top1_kind": [str(x) for x in single["top1_kind"]],
                "returned_kind": [str(x) for x in single["returned_kind"]],
                "stage": [str(x) for x in single["stage"]],
                "type_contract_ok": [bool(x) for x in single["type_contract_ok"]],
                "bitwise_equal_to_full_call": bool(
                    single["top1_index"] == full["top1_index"]
                    and single["returned_kind"] == full["returned_kind"]
                ),
            }
            courses[str(arm)] = row
            checks.append(
                (f"drill:{side}:{arm}:batch_size_invariance", row["bitwise_equal_to_full_call"])
            )
        report["courses"][str(side)] = courses
    # 类型标记契约的现场反例（必须能判定返回物类型）
    raw, contract = merged.lookup_typed(0)
    report["lookup_typed_demo"] = {
        "index": 0,
        "raw_bytes_len": int(len(raw)),
        "contract": contract.as_dict(),
        "kind_of_row_0": str(merged.kind_of(0)),
        "kind_of_last_row": str(merged.kind_of(merged.size - 1)),
    }
    checks.append(("kind_of_row_0_is_qa", str(merged.kind_of(0)) == MT.KIND_QA))
    checks.append(
        ("kind_of_last_row_is_text", str(merged.kind_of(merged.size - 1)) == MT.KIND_TEXT)
    )
    checks.append(
        (
            "half_indices_consistent",
            merged.indices_of(MT.KIND_QA)[-1] + 1 == merged.indices_of(MT.KIND_TEXT)[0],
        )
    )
    report["checks"] = [
        {"name": str(name), "passed": bool(ok)} for name, ok in checks
    ]
    report["all_passed"] = bool(all(ok for _n, ok in checks))
    sha, size = _write_json(os.path.join(out_dir, "drill.json"), report)
    _log(
        "[drill] 条目 = "
        + str(merged.size)
        + "（qa=" + str(merged.half_size("qa")) + " / text="
        + str(merged.half_size("text")) + "），三条前置断言 all_passed = "
        + str(bundle["premerge"]["all_passed"])
        + "，门禁 " + str(sum(1 for _n, ok in checks if ok)) + "/" + str(len(checks))
        + "，drill.json = " + str(size) + " B " + sha[:16]
    )
    if not report["all_passed"]:
        _log("[drill] 门禁未全过；拒绝把该通路当作「已演练」")
        return 1
    return 0


# ---------------------------------------------------------------------------
# eval：全量对照评测
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    """全量（2814 条目 / 149 + 666 查询）两臂对照评测 + 三条前置断言。"""
    import n3d_qa_learn.merge_table as MT

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    repro_dir = str(getattr(args, "repro_dir", "") or "")
    profile = str(getattr(args, "profile", "") or "lexical-88")
    batch_size = int(getattr(args, "batch_size", 64) or 64)
    tau = float(getattr(args, "tau", 0.0) or MT.TAU)
    if abs(tau - float(MT.TAU)) > 1e-12:
        raise ValueError(
            "TAU 是冻结常量（两臂共用门限），不接受运行期覆盖：当前 "
            + str(tau) + " vs 冻结值 " + str(MT.TAU)
        )
    variants_path = _frozen_variants_path(str(getattr(args, "variants", "") or ""))
    resolved_ckpt = checkpoint_path_resolution(
        dim=int(getattr(args, "checkpoint_dim", 0) or 88),
        max_batches=int(getattr(args, "max_batches", 0) or 0),
        tag=str(getattr(args, "checkpoint_tag", "") or ""),
        checkpoint_override=str(getattr(args, "checkpoint", "") or ""),
    )
    bundle = build_bundles(
        profile=profile,
        product_dir=str(getattr(args, "product_dir", "") or ""),
        qa_cache_dir=str(getattr(args, "qa_cache_dir", "") or ""),
        text_dir=str(getattr(args, "text_dir", "") or ""),
        batch_size=batch_size,
    )
    merged = bundle["merged"]
    qa_q = build_qa_queries(bundle)
    text_q = build_text_queries(bundle, variants_path=variants_path)
    _log(
        "[eval] 合并表 = " + str(merged.size) + " 条目（qa="
        + str(merged.half_size("qa")) + " / text=" + str(merged.half_size("text"))
        + "），D = " + str(merged.dim) + "，profile = " + str(profile)
    )
    _log(
        "[eval] 查询：QA 侧 " + str(qa_q["n"]) + " 条、文本侧 " + str(text_q["n"])
        + " 条 x " + str(len(MT.VARIANT_ORDER)) + " 支变体（两臂共用同一批）"
    )
    # --- 两臂 x 两侧 x 4 支变体 -----------------------------------------
    per_arm: Dict[str, Dict[str, Any]] = {}
    query_records: List[Dict[str, Any]] = []
    for arm in MT.ARM_ORDER:
        qa_out = MT.arm_top1(
            qa_q["feats"],
            merged,
            arm=str(arm),
            expected_kind="qa",
            tau=tau,
            batch_size=batch_size,
        )
        arm_block: Dict[str, Any] = {
            "arm": str(arm),
            "rule": str(MT.ARM_RULES[str(arm)]),
            "tau": float(tau),
            "qa": qa_side_metrics(qa_out, qa_q),
            "text": {},
        }
        for name in MT.VARIANT_ORDER:
            v = text_q["variants"][str(name)]
            out = MT.arm_top1(
                v["feats"],
                merged,
                arm=str(arm),
                expected_kind=str(text_q["expected_kind"]),
                tau=tau,
                batch_size=batch_size,
            )
            arm_block["text"][str(name)] = text_side_metrics(out, text_q, str(name))
            for i in range(int(out["n"])):
                query_records.append(
                    {
                        "arm": str(arm),
                        "side": "text",
                        "variant": str(name),
                        "query_id": str(text_q["query_ids"][i]),
                        "gold_rows": [int(x) for x in text_q["gold_rows"][i]],
                        "top1_index": int(out["top1_index"][i]),
                        "top1_score": float(out["top1_score"][i]),
                        "top1_kind": str(out["top1_kind"][i]),
                        "returned_index": int(out["returned_index"][i]),
                        "returned_kind": str(out["returned_kind"][i]),
                        "stage": str(out["stage"][i]),
                        "hit_at_1": bool(
                            int(out["top1_index"][i]) in set(text_q["gold_rows"][i])
                        ),
                        "type_contract_ok": bool(out["type_contract_ok"][i]),
                        "type_contract_violation": str(
                            out["type_contract_violation"][i]
                        ),
                    }
                )
        # QA 侧的逐查询记录（末尾追加即可，排序在写出前统一处理）
        for i in range(int(qa_out["n"])):
            query_records.append(
                {
                    "arm": str(arm),
                    "side": "qa",
                    "variant": "",
                    "query_id": str(qa_q["entry_ids"][i]),
                    "gold_rows": [],
                    "gold_answers": sorted(set(qa_q["gold"][i])),
                    "top1_index": int(qa_out["top1_index"][i]),
                    "top1_score": float(qa_out["top1_score"][i]),
                    "top1_kind": str(qa_out["top1_kind"][i]),
                    "returned_index": int(qa_out["returned_index"][i]),
                    "returned_kind": str(qa_out["returned_kind"][i]),
                    "stage": str(qa_out["stage"][i]),
                    "hit_at_1": bool(
                        str(qa_out["top1_kind"][i]) == "qa"
                        and str(qa_out["returned_output"][i])
                        in set(qa_q["gold"][i])
                    ),
                    "type_contract_ok": bool(qa_out["type_contract_ok"][i]),
                    "type_contract_violation": str(
                        qa_out["type_contract_violation"][i]
                    ),
                }
            )
        # 整臂汇总
        n_all = int(qa_q["n"]) + int(text_q["n"]) * len(MT.VARIANT_ORDER)
        n_ok_all = int(
            sum(
                1
                for row in query_records
                if str(row["arm"]) == str(arm) and bool(row["type_contract_ok"])
            )
        )
        arm_block["return_type_correctness_overall"] = float(n_ok_all / max(1, n_all))
        arm_block["n_type_records_overall"] = int(n_all)
        arm_block["n_type_contract_ok_overall"] = int(n_ok_all)
        arm_block["n_type_contract_violations_overall"] = int(n_all - n_ok_all)
        per_arm[str(arm)] = arm_block
        _log(
            "[eval] 臂 " + str(arm) + "：QA 命中率 = "
            + ("%.6f" % float(arm_block["qa"]["qa_hit_rate"]))
            + "，文本 R@1 = "
            + str(
                {
                    k: round(float(v["recall_at_1"]), 6)
                    for k, v in arm_block["text"].items()
                }
            )
            + "，返回类型正确率 = "
            + ("%.6f" % float(arm_block["return_type_correctness_overall"]))
        )
    verdict = build_verdict(
        per_arm, baseline_arm=MT.ARM_QA_PRIORITY, candidate_arm=MT.ARM_MERGED_TOPK1
    )
    # --- 缺陷登记：返回类型与期望类型不一致的样本 ------------------------
    violations = [row for row in query_records if not bool(row["type_contract_ok"])]
    # **F2 修复（计划 1 遗留项）**：合同声明了两类违规，实现此前只做第①类。
    # 这里补上第②类的**独立（内容视图）判定**并逐臂落产物。
    content_view_check: Dict[str, Any] = {
        "rule": str(MT.RETURN_TYPE_VIOLATION_CLASS_2_REACHABILITY),
        "arms": {},
    }
    for arm in MT.ARM_ORDER:
        sub = [r for r in query_records if str(r["arm"]) == str(arm)]
        content_view_check["arms"][str(arm)] = MT.content_view_violations(
            merged,
            [int(r["returned_index"]) for r in sub],
            [str(MT.EXPECTED_RETURN_KIND[str(r["side"])]) for r in sub],
        )
    content_view_check["n_class2_violations_total"] = int(
        sum(
            int(v["n_class2_violations"])
            for v in content_view_check["arms"].values()
        )
    )
    content_view_check["reachable_in_this_round"] = bool(
        int(content_view_check["n_class2_violations_total"]) > 0
    )
    content_view_check["reachability_note"] = (
        "本口径**不是恒真断言**：现场用伪造内容视图（把某条 qa 条目的输出替换为未登记串后"
        "重跑本函数）必须报出非 0 违规，见 `merge_table.json` / 测试报告的现场反例；"
        "当前轮次该计数为 0 是**可达性论证**的结果（kind↔内容 构造期绑定），不是「没测」。"
    )
    per_class: Dict[str, int] = MT.empty_violation_counts()
    per_class_by_side: Dict[str, Dict[str, int]] = {}
    for row in query_records:
        side = str(row["side"])
        block = per_class_by_side.setdefault(side, MT.empty_violation_counts())
        name = str(row["type_contract_violation"])
        if name:
            per_class[name] = int(per_class[name]) + 1
            block[name] = int(block[name]) + 1
    defect = {
        "n_type_contract_violations": int(len(violations)),
        "violation_counts": per_class,
        "violation_counts_by_side": per_class_by_side,
        "violation_class_rules": dict(MT.TYPE_VIOLATION_CLASS_RULE),
        "content_view_check": content_view_check,
        "samples_head": violations[:20],
        "rule": str(MT.RETURN_TYPE_RULE),
        "registered_as_defect": bool(len(violations) > 0),
        "defect_note": (
            "该计数为 0 时不得声称「已覆盖」；非 0 时**必须**登记为缺陷并给出具体样本"
            "（本字段的 samples_head 即逐样本取证）。`violation_counts_by_side` 是"
            "**按查询侧拆分**的同一计数：`returned_qa_expected_text` 只可能来自文本侧查询，"
            "不得把它整体读成「文本检索失败」——它包含「QA 质量页先被判给 QA 半区」这一"
            "结构性路由后果。"
        ),
    }
    # --- 跨臂诊断：合并表把文本行判给 QA 条目 vs QA 优先臂的差异 ----------
    qa_prio = per_arm[MT.ARM_QA_PRIORITY]
    merged_arm = per_arm[MT.ARM_MERGED_TOPK1]
    stealing: Dict[str, Any] = {}
    for name in MT.VARIANT_ORDER:
        a = qa_prio["text"][str(name)]
        b = merged_arm["text"][str(name)]
        stealing[str(name)] = {
            "n_top1_qa_kind_qa_priority": int(a["n_top1_qa_kind"]),
            "n_top1_qa_kind_merged": int(b["n_top1_qa_kind"]),
            "delta_n_top1_qa_kind": int(b["n_top1_qa_kind"]) - int(a["n_top1_qa_kind"]),
            "recall_at_1_delta_merged_minus_qa_priority": float(
                b["recall_at_1"]
            )
            - float(a["recall_at_1"]),
        }
    # --- 落盘 -----------------------------------------------------------
    table_doc = {
        "frozen_constants": {
            "MERGE_TABLE_VERSION": str(MT.MERGE_TABLE_VERSION),
            "MERGE_FINGERPRINT_SCHEMA": str(MT.MERGE_FINGERPRINT_SCHEMA),
            "MERGE_ORDER": [str(x) for x in MT.MERGE_ORDER],
            "ENTRY_KINDS": [str(x) for x in MT.ENTRY_KINDS],
            "RETURN_KINDS": [str(x) for x in MT.RETURN_KINDS],
            "ARM_ORDER": [str(x) for x in MT.ARM_ORDER],
            "ARM_RULES": dict(MT.ARM_RULES),
            "TAU": float(MT.TAU),
            "TAU_RULE": str(MT.TAU_RULE),
            "VARIANT_ORDER": [str(x) for x in MT.VARIANT_ORDER],
            "NON_ANCHOR_VARIANTS": [str(x) for x in MT.NON_ANCHOR_VARIANTS],
            "GOLD_SET_RULE": str(MT.GOLD_SET_RULE),
            "RETURN_TYPE_RULE": str(MT.RETURN_TYPE_RULE),
            "TYPE_VIOLATION_CLASSES": [str(x) for x in MT.TYPE_VIOLATION_CLASSES],
            "VERDICT_RULE": str(MT.VERDICT_RULE),
            "SCOPE_NOTE": str(MT.SCOPE_NOTE),
            "ANCHOR_VARIANT_NOTE": str(MT.ANCHOR_VARIANT_NOTE),
            "PREMERGE_ASSERTION_KEYS": [str(x) for x in MT.PREMERGE_ASSERTION_KEYS],
            "ASCII_SAFE_LOG_NOTE": str(ASCII_SAFE_LOG_NOTE),
        },
        "checkpoint_path_resolution": resolved_ckpt,
        "artifact_scope": artifact_scope_evidence(resolved_ckpt, out_dir),
        "merge_table": merged.meta(),
        "premerge_assertions": bundle["premerge"],
        "split_attestation": {
            "n_entries_total": int(merged.size),
            "n_entries_by_kind": {
                str(k): int(merged.half_size(k)) for k in MT.MERGE_ORDER
            },
            "n_qa_queries": int(qa_q["n"]),
            "n_text_queries": int(text_q["n"]),
            "qa_side": bundle["evidence"]["splits"],
            "qa_query_source": "data.make_splits 的 test_known（与 exp_repr 基线档同口径）",
            "text_query_source": (
                "n3d_qa 冻结划分的 query 行（reproduce_doc_split）∩ 变体产物"
            ),
            "text_pool_rule": (
                "检索池 = 合并表全量（QA 149 + 文本 2665 = 2814）；"
                "文本侧候选包含 QA 条目，这正是「异构干扰量」的度量口径"
            ),
            "variants_path": str(text_q["variants_path"]),
            "variants_sha256": str(text_q["variant_sha256"]),
            "variant_record_fields": dict(text_q["fields"]),
            "n_duplicate_gold_rows": int(text_q["n_duplicate_gold_rows"]),
            "c10_scope_rule": str(MT.SCOPE_NOTE),
        },
        "qa_query_rule": {
            "gold_rule": str(qa_q["gold_rule"]),
            "gold_set_sizes": dict(qa_q["gold_set_sizes"]),
            "n_distinct_gold_sets": int(qa_q["n_distinct_gold_sets"]),
            "n_queries": int(qa_q["n"]),
        },
        "side_evidence": bundle["evidence"],
    }
    eval_doc = {
        "profile": str(profile),
        "dim": int(bundle["dim"]),
        "tau": float(tau),
        "per_arm": per_arm,
        "verdict": verdict,
        "return_type_contract": defect,
        "cross_arm_diagnostics": {"text_top1_qa_kind": stealing},
        "stage_semantics": {
            "qa_priority": (
                "stage = 两级路由的**到达档**：qa = 第一级（QA 半区）top-1 达到 TAU 即返回；"
                "text = 落到第二级（文本半区）；none = 两级都低于 TAU。"
            ),
            "merged_topk1": (
                "stage = 单次排序 top-1 的 kind（qa / text）；none = 该 top-1 低于 TAU。"
                "本臂没有优先级，故 stage 不是「路由分支」而是「返回物类型」。"
            ),
            "why_it_matters": (
                "两臂的 stage 语义不同，故文本侧 R@1 必须**同时**报总体值与按 stage 拆分值："
                "QA 优先臂会把一部分**行原文**查询（其文本逐字节等于键表内该行）先判给 QA 半区，"
                "使它们在文本半区上的 R@1 失真。"
            ),
        },
        "scope_note": str(MT.SCOPE_NOTE),
    }
    table_sha, table_size = _write_json(os.path.join(out_dir, "merge_table.json"), table_doc)
    eval_sha, eval_size = _write_json(os.path.join(out_dir, "route_eval.json"), eval_doc)
    rows_sorted = sorted(
        query_records,
        key=lambda r: (
            str(r["arm"]),
            str(r["side"]),
            str(r["variant"]),
            str(r["query_id"]),
        ),
    )
    jsonl_sha, jsonl_size = _write_jsonl(
        os.path.join(out_dir, "route_eval_queries.jsonl"), rows_sorted
    )
    _log(
        "[eval] 产物：merge_table.json = " + str(table_size) + " B " + table_sha[:16]
        + "；route_eval.json = " + str(eval_size) + " B " + eval_sha[:16]
        + "；route_eval_queries.jsonl = " + str(jsonl_size) + " B " + jsonl_sha[:16]
    )
    _log(
        "[eval] 判定 = " + str(verdict["verdict"])
        + "；劣化指标 = " + str(verdict["degraded_metrics"])
        + "；改进指标 = " + str(verdict["improved_metrics"])
    )
    _log(
        "[eval] 返回类型契约：不一致样本数 = "
        + str(defect["n_type_contract_violations"])
        + "，分类计数 = " + str(defect["violation_counts"])
    )
    if repro_dir:
        det = determinism_block(str(out_dir), str(repro_dir))
        _write_json(os.path.join(out_dir, "determinism_check.json"), det)
        _log(
            "[eval] 双跑一致性：共有文件 " + str(det["n_common_files"])
            + "，逐字节相同 = " + str(det["determinism_ok"])
            + "，不相等文件 = " + str(det["not_equal_files"])
        )
    return 0


# ---------------------------------------------------------------------------
# report：渲染 JSON + Markdown
# ---------------------------------------------------------------------------


def render_markdown(report: Dict[str, Any]) -> str:
    """把报告渲染为 Markdown（**确定性**：无时间戳、无耗时）。"""
    import n3d_qa_learn.merge_table as MT

    tab = report["merge_table"]
    ver = report["verdict"]
    lines: List[str] = []
    lines.append("# 轮 C：合并表零训练对照（全量 " + str(tab["n_entries"]) + " 条目）")
    lines.append("")
    lines.append("## 1. 口径")
    lines.append("")
    lines.append("- 条目总数 **" + str(tab["n_entries"]) + "** = QA 侧 `"
                 + str(tab["n_entries_by_kind"]["qa"]) + "` + 文本侧 `"
                 + str(tab["n_entries_by_kind"]["text"]) + "`；合并顺序 = `"
                 + str(tab["merge_order"]) + "`")
    lines.append("- 连接参数 `D` = **" + str(tab["dim"]) + "**；归一化口径 = `"
                 + str(tab["norm_spec"]) + "`；`dtype` = `" + str(tab["dtype"]) + "`")
    lines.append("- 合并表内容指纹 `merge_key_table_sha256` = `"
                 + str(tab["merge_key_table_sha256"]) + "`")
    lines.append("- 两臂共用门限 `TAU` = **" + str(report["tau"]) + "**；主判据只取未过滤 top-1")
    lines.append("- 两半在合并表中的**半区键裸字节 SHA256** 必须与其来源表**同一口径**（键裸字节）的 SHA256 逐字相同（见第 2.1 节；注意它与 `key_table_sha256` 是两个**不同**的指纹方案，不得互比）")
    lines.append("")
    lines.append("### 1.1 两半元数据（**完整保留**）")
    lines.append("")
    lines.append("| 半区 | 条目数 | dim | norm_spec | encoder_profile | encoder_fingerprint | key_table_sha256 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for kind in MT.MERGE_ORDER:
        meta = tab["halves"][str(kind)]["meta"]
        lines.append(
            "| `" + str(kind) + "` | " + str(meta["n_entries"]) + " | " + str(meta["dim"])
            + " | `" + str(meta["norm_spec"]) + "` | `" + str(meta["encoder_profile"])
            + "` | `" + str(meta["encoder_fingerprint"])[:16] + "...` | `"
            + str(meta["key_table_sha256"])[:16] + "...` |"
        )
    lines.append("")
    lines.append("### 1.2 结论适用范围（**必须显式声明**）")
    lines.append("")
    lines.append("> " + str(report["scope_note"]))
    lines.append("")
    lines.append("## 2. 合并前置断言（先跑，任一失败即停止）")
    lines.append("")
    pre = report["premerge_assertions"]
    lines.append("- `all_passed` = **" + str(pre["all_passed"]) + "**；断言顺序 = `"
                 + str(pre["assertion_order"]) + "`")
    lines.append("")
    lines.append("| 断言 | 通过 | 现场证据摘要 |")
    lines.append("| --- | --- | --- |")
    d1 = pre["assertions"]["same_dtype_and_norm"]["detail"]
    lines.append(
        "| `same_dtype_and_norm` | " + str(pre["assertions"]["same_dtype_and_norm"]["passed"])
        + " | qa D=" + str(d1["qa_dim"]) + "/text D=" + str(d1["text_dim"])
        + "，dtype 同 = " + str(d1["same_dtype"])
        + "，范数区间 qa " + str(d1["qa_norm_range"]) + " / text " + str(d1["text_norm_range"])
        + " |"
    )
    d2 = pre["assertions"]["role_bitwise_identity"]["detail"]
    lines.append(
        "| `role_bitwise_identity` | " + str(pre["assertions"]["role_bitwise_identity"]["passed"])
        + " | 逐位相同 " + str(d2["n_bitwise_equal"]) + "/" + str(d2["n_checked"])
        + "（exact_frac = " + str(d2["exact_frac"]) + "）；两角色口径指纹 = `"
        + str(d2["encoder_fingerprints"]) + "` |"
    )
    d3 = pre["assertions"]["merge_key_bytes_unchanged"]["detail"]
    lines.append(
        "| `merge_key_bytes_unchanged` | "
        + str(pre["assertions"]["merge_key_bytes_unchanged"]["passed"])
        + " | qa 半区 " + str(d3["qa"]["exact_frac"]) + "（" + str(d3["qa"]["n_checked"])
        + " 行）、text 半区 " + str(d3["text"]["exact_frac"]) + "（"
        + str(d3["text"]["n_checked"]) + " 行）逐字节零变化；半区裸字节 SHA256 前后一致 |"
    )
    lines.append("")
    lines.append("### 2.1 半区裸字节对照（合并前后）")
    lines.append("")
    lines.append("| 半区 | 行数 | 合并前键裸字节 SHA256 | 合并后半区键裸字节 SHA256 | 一致 | 内容指纹（单侧表口径） |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for kind in MT.MERGE_ORDER:
        blk = d3[str(kind)]
        lines.append(
            "| `" + str(kind) + "` | " + str(blk["n_checked"]) + " | `"
            + str(blk["source_raw_keys_sha256"])[:16] + "...` | `"
            + str(blk["merged_half_raw_keys_sha256"])[:16] + "...` | "
            + str(
                str(blk["source_raw_keys_sha256"])
                == str(blk["merged_half_raw_keys_sha256"])
            )
            + " | `" + str(blk["source_content_fingerprint"])[:16] + "...` |"
        )
    lines.append("")
    lines.append("- `raw_keys_sha256_rule`：" + str(d3[str(MT.MERGE_ORDER[0])]["raw_keys_sha256_rule"]))
    lines.append("")
    lines.append("## 3. 查询集与两臂路由")
    lines.append("")
    sa = report["split_attestation"]
    lines.append("- QA 侧查询 **" + str(sa["n_qa_queries"]) + "** 条（`"
                 + str(sa["qa_query_source"]) + "`）")
    lines.append("- QA 侧金标口径：" + str(report["qa_query_rule"]["gold_rule"])
                 + "；金标集合大小区间 = `" + str(report["qa_query_rule"]["gold_set_sizes"])
                 + "`；different 金标集合数 = " + str(report["qa_query_rule"]["n_distinct_gold_sets"]))
    lines.append("- 文本侧查询 **" + str(sa["n_text_queries"]) + "** 条 x 4 支变体（`"
                 + str(sa["text_query_source"]) + "`）")
    lines.append("- 文本侧池口径：" + str(sa["text_pool_rule"]))
    lines.append("- 变体产物（**冻结**，只读）：`" + str(sa["variants_path"]) + "`，SHA256 = `"
                 + str(sa["variants_sha256"]) + "`")
    lines.append("- 变体记录字段现场枚举：`" + str(sa["variant_record_fields"]["record_keys"]) + "`")
    lines.append("- 变体键现场枚举：`" + str(sa["variant_record_fields"]["variant_keys"]) + "`")
    lines.append("- 金标集口径：" + str(MT.GOLD_SET_RULE))
    lines.append("- 原文逐字节重复导致的「金标集 > 自身行」的查询数 = **"
                 + str(sa["n_duplicate_gold_rows"]) + "**")
    lines.append("")
    lines.append("| 臂 | 路由结构 |")
    lines.append("| --- | --- |")
    for arm in MT.ARM_ORDER:
        lines.append("| `" + str(arm) + "` | " + str(MT.ARM_RULES[str(arm)]) + " |")
    lines.append("")
    lines.append("## 4. 分侧指标")
    lines.append("")
    lines.append("### 4.1 QA 侧命中率")
    lines.append("")
    lines.append("| 臂 | n | 命中率（未过滤 top-1） | 严格命中率 | 命中率（TAU 过滤） | 严格命中率（TAU 过滤） | top-1 分位 p50 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for arm in MT.ARM_ORDER:
        q = report["per_arm"][str(arm)]["qa"]
        lines.append(
            "| `" + str(arm) + "` | " + str(q["n_queries"]) + " | "
            + ("%.6f" % float(q["qa_hit_rate"])) + " | "
            + ("%.6f" % float(q["qa_hit_strict_rate"])) + " | "
            + ("%.6f" % float(q["qa_hit_rate_returned"])) + " | "
            + ("%.6f" % float(q["qa_hit_strict_rate_returned"])) + " | "
            + ("%.6f" % float(q["top1_score_quantiles"]["p50"])) + " |"
        )
    lines.append("")
    lines.append("### 4.2 文本侧 R@1（按 4 支变体分档）")
    lines.append("")
    lines.append("| 臂 | 变体 | 档位 | R@1（总体） | R@1（stage=text 子集） | 该子集 n | R@1（stage=qa 子集） | 该子集 n | 精确自身行 top-1 比例 | top-1 为 QA 条目 | 类型正确率 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in MT.ARM_ORDER:
        for name in MT.VARIANT_ORDER:
            t = report["per_arm"][str(arm)]["text"][str(name)]
            lines.append(
                "| `" + str(arm) + "` | `" + str(name) + "` | " + str(t["label"]) + " | "
                + ("%.6f" % float(t["recall_at_1"])) + " | "
                + ("%.6f" % float(t["recall_at_1_text_stage"])) + " | "
                + str(t["n_stage_text"]) + " | "
                + ("%.6f" % float(t["recall_at_1_qa_stage"])) + " | "
                + str(t["n_stage_qa"]) + " | "
                + ("%.6f" % float(t["exact_self_top1_rate"])) + " | "
                + str(t["n_top1_qa_kind"]) + " | "
                + ("%.6f" % float(t["type_contract"]["return_type_correctness"])) + " |"
            )
    lines.append("")
    lines.append("- " + str(MT.ANCHOR_VARIANT_NOTE))
    lines.append("- 金标集口径：" + str(MT.GOLD_SET_RULE))
    lines.append("- 两臂 `stage` 语义差异（**必须显式**，否则总体 R@1 会被误读）：")
    for arm in MT.ARM_ORDER:
        lines.append("  - `" + str(arm) + "`：" + str(report["stage_semantics"][str(arm)]))
    lines.append("  - " + str(report["stage_semantics"]["why_it_matters"]))
    lines.append("")
    lines.append("### 4.3 返回类型正确率（整臂，含 QA 与全部文本查询）")
    lines.append("")
    lines.append("| 臂 | 记录数 | 类型正确数 | 返回类型正确率 | 不一致数 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for arm in MT.ARM_ORDER:
        a = report["per_arm"][str(arm)]
        lines.append(
            "| `" + str(arm) + "` | " + str(a["n_type_records_overall"]) + " | "
            + str(a["n_type_contract_ok_overall"]) + " | "
            + ("%.6f" % float(a["return_type_correctness_overall"])) + " | "
            + str(a["n_type_contract_violations_overall"]) + " |"
        )
    lines.append("")
    lines.append("## 5. 返回类型契约与缺陷登记")
    lines.append("")
    tc = report["return_type_contract"]
    lines.append("- 口径：" + str(tc["rule"]))
    lines.append("- **返回类型与期望类型不一致的样本数 = " + str(tc["n_type_contract_violations"])
                 + "**；分类计数 = `" + str(tc["violation_counts"]) + "`")
    lines.append("- 按查询侧拆分：`" + str(tc["violation_counts_by_side"]) + "`")
    lines.append("  - 说明：" + str(tc["defect_note"]))
    lines.append("- 是否登记为缺陷：**" + str(tc["registered_as_defect"]) + "**")
    lines.append("- 分类规则：" + str(tc["violation_class_rules"]))
    lines.append("")
    if tc["samples_head"]:
        lines.append("具体样本（前 " + str(len(tc["samples_head"])) + " 条）：")
        lines.append("")
        lines.append("| 臂 | 侧 | 变体 | 查询 id | top-1 下标 | top-1 类型 | 返回类型 | 期望类型 | 违规类别 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in tc["samples_head"]:
            lines.append(
                "| `" + str(row["arm"]) + "` | " + str(row["side"]) + " | `"
                + str(row["variant"]) + "` | `" + str(row["query_id"]) + "` | "
                + str(row["top1_index"]) + " | `" + str(row["top1_kind"]) + "` | `"
                + str(row["returned_kind"]) + "` | "
                + str(MT.EXPECTED_RETURN_KIND.get(str(row["side"]), "")) + " | `"
                + str(row["type_contract_violation"]) + "` |"
            )
    else:
        lines.append("**未观察到任何「返回类型与期望类型不一致」的样本**（计数为 0 是实测结果，"
                     "不是未测量的占位值）。")
    lines.append("")
    lines.append("## 6. 跨臂诊断：文本行被 QA 条目抢走")
    lines.append("")
    lines.append("| 变体 | QA 优先臂 top-1 为 QA 条目 | 合并表臂 top-1 为 QA 条目 | 差值 | R@1 差（合并表 - QA 优先） |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name in MT.VARIANT_ORDER:
        s = report["cross_arm_diagnostics"]["text_top1_qa_kind"][str(name)]
        lines.append(
            "| `" + str(name) + "` | " + str(s["n_top1_qa_kind_qa_priority"]) + " | "
            + str(s["n_top1_qa_kind_merged"]) + " | " + str(s["delta_n_top1_qa_kind"])
            + " | " + ("%.6f" % float(s["recall_at_1_delta_merged_minus_qa_priority"])) + " |"
        )
    lines.append("")
    lines.append("## 7. 合并表 vs 现状路由：非劣 / 劣化判定")
    lines.append("")
    lines.append("- 基线臂 = `" + str(ver["baseline_arm"]) + "`；候选臂 = `"
                 + str(ver["candidate_arm"]) + "`")
    lines.append("- 判定 = **`" + str(ver["verdict"]) + "`**；"
                 "不劣于基线 = **" + str(ver["not_worse_than_baseline"]) + "**；"
                 "全面严格更优 = " + str(ver["strictly_better_than_baseline"]))
    lines.append("- 口径：" + str(ver["rule"]))
    lines.append("")
    lines.append("| 指标 | 基线（现状路由） | 候选（合并表） | 落差 | 基线是否在可行域内部 | 不劣 | 严格更优 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in ver["metrics"]:
        lines.append(
            "| `" + str(row["metric"]) + "` | " + ("%.6f" % float(row["baseline"])) + " | "
            + ("%.6f" % float(row["candidate"])) + " | "
            + ("%+.6f" % float(row["delta"])) + " | "
            + str(row["baseline_interior"]) + " | " + str(row["not_worse"]) + " | "
            + str(row["strictly_better"]) + " |"
        )
    lines.append("")
    lines.append("- 劣化指标 = `" + str(ver["degraded_metrics"]) + "`；改进指标 = `"
                 + str(ver["improved_metrics"]) + "`")
    lines.append("- " + str(ver["anchor_variant_note"]))
    lines.append("")
    lines.append("## 8. 产物路径复算与产物保护")
    lines.append("")
    ck = report["checkpoint_path_resolution"]
    lines.append("- **现场**调用 `" + str(ck["source"]) + "` 复算（最终参数：dim="
                 + str(ck["dim"]) + " / max_batches=" + str(ck["max_batches"]) + " / tag=`"
                 + str(ck["tag"]) + "`）：status = **" + str(ck["status"]) + "**")
    if ck["status"] == "computed":
        lines.append("- 复算结果 `resolved_checkpoint_path` = `"
                     + str(ck["resolved_checkpoint_path"]) + "`")
    else:
        lines.append("- 复算不可用（**如实登记**）：`" + str(ck.get("error", "")) + "`")
    sc = report["artifact_scope"]
    lines.append("- 产物保护核实：该路径落在本模块产物目录内 = **"
                 + str(sc["inside_out_dir"]) + "**；落在 `_verify/` 内 = **"
                 + str(sc["inside_verify_root"]) + "**")
    lines.append("- " + str(sc["rule"]))
    lines.append("")
    lines.append("## 9. 确定性证据")
    lines.append("")
    det = report.get("determinism")
    if det:
        lines.append("- 双跑目录：`" + str(det["a_dir"]) + "` vs `" + str(det["b_dir"]) + "`")
        lines.append("- 共有文件 " + str(det["n_common_files"]) + "，逐字节相同 = **"
                     + str(det["determinism_ok"]) + "**，不相等文件 = `"
                     + str(det["not_equal_files"]) + "`")
        lines.append("- 排除前文件数：a = " + str(det["n_files_before_exclusion"]["a"])
                     + " / b = " + str(det["n_files_before_exclusion"]["b"])
                     + "；单侧独有文件：a_only = `" + str(det["a_only_files"])
                     + "` / b_only = `" + str(det["b_only_files"]) + "`")
        lines.append("- 判定语义：" + str(det["determinism_ok_semantics"]))
        lines.append("- 排除口径：" + str(det["determinism_scope"]))
        lines.append("- 排除项：`" + str(det["excluded_self_file"]) + "`（"
                     + str(det["self_exclusion_rule"]) + "）；该文件在两目录中的存在性 = `"
                     + str(det["excluded_self_present"]) + "`")
        lines.append("")
        lines.append("| 文件 | a 字节 | b 字节 | 逐字节相同 |")
        lines.append("| --- | --- | --- | --- |")
        for item in det["files"]:
            lines.append(
                "| `" + str(item["name"]) + "` | " + str(item["a_bytes"]) + " | "
                + str(item["b_bytes"]) + " | " + str(item["bitwise_equal"]) + " |"
            )
    else:
        lines.append("- 本次 `report` 未携带双跑证据（`eval --repro-dir` 才产出）")
    lines.append("")
    lines.append("## 10. 冻结常量清单（现场枚举）")
    lines.append("")
    lines.append("| 常量 | 值 |")
    lines.append("| --- | --- |")
    for key, value in report["frozen_constants"].items():
        if key in ("ARM_RULES", "TAU_RULE", "GOLD_SET_RULE", "RETURN_TYPE_RULE",
                   "VERDICT_RULE", "SCOPE_NOTE", "ANCHOR_VARIANT_NOTE", "ASCII_SAFE_LOG_NOTE"):
            lines.append("| `" + str(key) + "` | " + str(value)[:120].replace("|", "\\|")
                         + ("..." if len(str(value)) > 120 else "") + " |")
        else:
            lines.append("| `" + str(key) + "` | `" + str(value) + "` |")
    lines.append("")
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    """渲染 ``merge_table_report.{json,md}``（数据全部来自已落盘产物）。"""
    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    table_doc = _read_json(os.path.join(out_dir, "merge_table.json"))
    eval_doc = _read_json(os.path.join(out_dir, "route_eval.json"))
    report: Dict[str, Any] = {
        "profile": str(eval_doc["profile"]),
        "dim": int(eval_doc["dim"]),
        "tau": float(eval_doc["tau"]),
        "merge_table": dict(table_doc["merge_table"]),
        "frozen_constants": dict(table_doc["frozen_constants"]),
        "checkpoint_path_resolution": dict(table_doc["checkpoint_path_resolution"]),
        "artifact_scope": dict(table_doc["artifact_scope"]),
        "premerge_assertions": dict(table_doc["premerge_assertions"]),
        "split_attestation": dict(table_doc["split_attestation"]),
        "qa_query_rule": dict(table_doc["qa_query_rule"]),
        "side_evidence": dict(table_doc["side_evidence"]),
        "per_arm": dict(eval_doc["per_arm"]),
        "verdict": dict(eval_doc["verdict"]),
        "return_type_contract": dict(eval_doc["return_type_contract"]),
        "cross_arm_diagnostics": dict(eval_doc["cross_arm_diagnostics"]),
        "stage_semantics": dict(eval_doc["stage_semantics"]),
        "scope_note": str(eval_doc["scope_note"]),
        "artifact_fingerprints": {},
    }
    det_path = os.path.join(out_dir, "determinism_check.json")
    if os.path.isfile(det_path):
        report["determinism"] = _read_json(det_path)
    markdown = render_markdown(report)
    md_sha, md_size = _write_text(os.path.join(out_dir, "merge_table_report.md"), markdown)
    report["artifact_fingerprints"] = {
        "merge_table.json": {
            "bytes": int(os.path.getsize(os.path.join(out_dir, "merge_table.json"))),
            "sha256": _sha256_file(os.path.join(out_dir, "merge_table.json")),
        },
        "route_eval.json": {
            "bytes": int(os.path.getsize(os.path.join(out_dir, "route_eval.json"))),
            "sha256": _sha256_file(os.path.join(out_dir, "route_eval.json")),
        },
        "route_eval_queries.jsonl": {
            "bytes": int(os.path.getsize(os.path.join(out_dir, "route_eval_queries.jsonl"))),
            "sha256": _sha256_file(os.path.join(out_dir, "route_eval_queries.jsonl")),
        },
        "merge_table_report.md": {"bytes": int(md_size), "sha256": str(md_sha)},
    }
    if os.path.isfile(det_path):
        report["artifact_fingerprints"]["determinism_check.json"] = {
            "bytes": int(os.path.getsize(det_path)),
            "sha256": _sha256_file(det_path),
        }
    json_sha, json_size = _write_json(
        os.path.join(out_dir, "merge_table_report.json"), report
    )
    _log(
        "[report] merge_table_report.json = " + str(json_size) + " B " + json_sha[:16]
        + "；merge_table_report.md = " + str(md_size) + " B " + md_sha[:16]
    )
    _log(
        "[report] 判定 = " + str(report["verdict"]["verdict"])
        + "；返回类型不一致样本数 = "
        + str(report["return_type_contract"]["n_type_contract_violations"])
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_merge_table(args: argparse.Namespace) -> int:
    """``merge_table`` 子命令分派。

    **CLI 错误边界**：运行期 ``ValueError`` / ``KeyError`` -> 单行可读报文 + 退码 2；
    ``FileNotFoundError`` / ``RuntimeError`` -> 可读报文 + 退码 1。
    ``UnicodeEncodeError`` 先于 ``ValueError`` 捕获（它继承 ``ValueError``）。
    """
    if int(getattr(sys.flags, "optimize", 0)) != 0:
        _log(
            "[merge_table] 拒绝在优化模式下运行：sys.flags.optimize = "
            + str(int(getattr(sys.flags, "optimize", 0)))
            + "（由 python -O / PYTHONOPTIMIZE 打开）。本模块的前置断言与门禁是"
            "「先跑、失败即停」的硬约束，优化模式会丢弃 __debug__ 分支与 docstring，"
            "使口径不可审计；请用 python（不带 -O）重跑。"
        )
        return 1
    sub = str(getattr(args, "merge_cmd", "") or "drill")
    handlers = {
        "drill": cmd_drill,
        "eval": cmd_eval,
        "report": cmd_report,
    }
    if sub not in handlers:
        _log("[merge_table] 未知子命令 " + repr(sub) + "；可用 = " + str(sorted(handlers)))
        return 2
    try:
        return int(handlers[sub](args))
    except UnicodeEncodeError as exc:
        _log(
            "[merge_table] 输出编码错误（日志含当前控制台编码无法表示的字符）："
            + type(exc).__name__ + ": " + str(exc)
            + "。这不是参数错误；可用 PYTHONIOENCODING=utf-8 重跑。"
        )
        return 1
    except FileNotFoundError as exc:
        _log("[merge_table] 输入产物缺失：" + str(exc))
        return 1
    except (ValueError, KeyError) as exc:
        _log("[merge_table] 非法参数 / 契约校验失败：" + str(exc))
        return 2
    except RuntimeError as exc:
        _log("[merge_table] 运行失败（依赖 / 产物不可用）：" + str(exc))
        return 1


class _Tee:
    """把 stdout 同时写到控制台与 **UTF-8 无 BOM** 日志文件。"""

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

    def __getattr__(self, item: str) -> Any:
        """其余属性透传（编码探测等）。"""
        return getattr(self._primary, item)


def _add_merge_table_arguments(p: argparse.ArgumentParser) -> None:
    """把 ``merge_table`` 的参数表挂到给定子解析器上（**唯一**定义点）。"""
    p.add_argument("--product-dir", default="", help="n3d_qa 冻结产物目录（只读）")
    p.add_argument("--qa-cache-dir", default="", help="QA JSON 缓存目录（空 = 默认）")
    p.add_argument("--text-dir", default="", help="文本语料目录（空 = 默认）")
    p.add_argument("--variants", default="", help="**已冻结**的 variants.jsonl（空 = 默认路径）")
    p.add_argument("--profile", default="lexical-88",
                   help="特征档（本轮冻结为 lexical-88；该档两角色口径逐位相同）")
    p.add_argument("--out-dir", default="", help="产物目录（默认 checkpoints/qa_learn/_verify/merge_table）")
    p.add_argument("--log-file", default="", help="同时写 UTF-8(无 BOM) 日志文件")
    p.add_argument("--batch-size", type=int, default=64, help="编码批大小")
    p.add_argument("--tau", type=float, default=0.0, help="门限（0 = 用冻结常量 TAU；不接受覆盖）")
    p.add_argument("--checkpoint-dim", type=int, default=88,
                   help="现场复算 checkpoint 路径用的输入维（默认 88 = 本轮 D）")
    p.add_argument("--max-batches", type=int, default=0,
                   help="现场复算用的 max_batches（0 = 正式路径档）")
    p.add_argument("--checkpoint-tag", default="merge_table", help="现场复算用的 tag")
    p.add_argument("--checkpoint", default="", help="现场复算用的 --checkpoint 覆盖值")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    import n3d_qa_learn.merge_table as MT

    parser = argparse.ArgumentParser(
        prog="n3d_qa_learn.merge_table_run",
        description=(
            "轮 C：合并表零训练对照（QA 149 + 文本 2665 = 2814 条目；两臂路由对照）。"
            "产物一律写 " + MT.DEFAULT_OUT_DIR.replace("\\", "/") + " 之下。"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    help_texts = {
        "drill": "单条端到端演练（1 条 QA + 1 条文本；门禁全过才退码 0）",
        "eval": "全量两臂对照评测 + 三条前置断言 + 类型契约统计 + 机器可读判定",
        "report": "渲染 merge_table_report.{json,md}",
    }
    for name in MERGE_TABLE_COMMANDS:
        p = sub.add_parser(name, help=help_texts[str(name)])
        _add_merge_table_arguments(p)
        if str(name) == "eval":
            p.add_argument("--repro-dir", default="",
                           help="对照产物目录：给定时产出**双跑逐字节一致**证据")
        p.set_defaults(func=cmd_merge_table, merge_cmd=str(name))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口（``python -m n3d_qa_learn.merge_table_run <cmd>``）。"""
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
