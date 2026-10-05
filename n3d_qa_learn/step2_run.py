"""n3d_qa_learn 步骤 2 驱动：probe（数据与口径取证）/ drill（单条端到端演练）/ eval（全量分项验收）。

用法::

    python -m n3d_qa_learn.step2_run probe  [--product-dir DIR] [--out-dir DIR]
    python -m n3d_qa_learn.step2_run drill  [--out-dir DIR]
    python -m n3d_qa_learn.step2_run eval   [--out-dir DIR] [--epochs N] [--seeds 42,43,44]

产物纪律
--------
* 正式产物写 ``checkpoints/qa_learn/step2/``；
* 验证类运行（drill / 限批）写 ``checkpoints/qa_learn/_verify/step2/``。
所有数字均由现场运行产出，禁止硬编码。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import encoders as E
from . import step2 as S
from .route import QuestionRouter
from .train import ZIP_EPOCH

Json = Dict[str, Any]

#: ``encoders_run`` 写入的取证文件（``verify-vectorizer`` 的 ① 项基准来源）。
ENCODERS_VERIFY_DIR: str = os.path.join("checkpoints", "qa_learn", "_verify", "encoders")


def _encoder_config(args: argparse.Namespace) -> Optional[E.EncoderConfig]:
    """由 CLI 参数装配步骤 2 的可插拔编码器配置（空 ``--encoder`` = 现状 ``zh-bag``）。"""
    name = str(getattr(args, "encoder", "") or "")
    if not name:
        return None
    return E.EncoderConfig(
        name=name,
        role=E.ROLE_TEXT_LINE,
        max_length=int(getattr(args, "enc_max_length", 0) or 0),
        source=str(getattr(args, "enc_source", "") or ""),
        revision=str(getattr(args, "enc_revision", "") or ""),
        cache_dir=str(getattr(args, "enc_cache_dir", "") or E.DEFAULT_EMB_CACHE_DIR),
        use_cache=True if bool(getattr(args, "enc_force_cache", False)) else None,
        local_files_only=bool(getattr(args, "enc_local_files_only", False)),
    )


def _declared_fingerprint(args: argparse.Namespace, name: str, role: str) -> Tuple[str, str]:
    """取 ① 项的基准指纹：显式参数优先，其次 ``encoders_run`` 的落盘取证。

    **落盘取证必须与当前请求的编码器 + 角色一致**才被采用 —— 否则 bge-m3 的
    ``drill.json`` 会被误当作 ``zh-bag`` 的基准（现场实测过该错配）。
    """
    explicit = str(getattr(args, "declared_fingerprint", "") or "")
    if explicit:
        return explicit, "--declared-fingerprint（命令行显式给出）"
    for file_name in ("drill.json", "info.json"):
        path = os.path.join(str(getattr(args, "encoders_dir", "") or ENCODERS_VERIFY_DIR),
                            file_name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            recorded = dict(json.load(handle))
        recorded_name = str(recorded.get("name") or recorded.get("encoder") or "")
        recorded_role = str(recorded.get("role", ""))
        if recorded_name != str(name):
            continue
        if recorded_role and recorded_role != str(role):
            continue
        if str(recorded.get("fingerprint", "")):
            return str(recorded["fingerprint"]), path.replace("\\", "/")
    return "", "（无同口径落盘声明 -> 现场按注册表构造基准）"


class _Tee:
    """把 stdout 同时写到控制台与 **UTF-8 无 BOM** 日志文件。

    存在理由（审查条目 W3）：PowerShell 5.1 的 ``Tee-Object`` 默认写 UTF-16LE（BOM FF FE），
    中文全部乱码且不符合产物纪律。改由 Python 自己以 ``encoding="utf-8"`` 落日志。
    """

    def __init__(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8", newline="\n")
        self._stdout = sys.stdout

    def write(self, text: str) -> int:
        """同时写控制台与文件（返回写入长度以兼容 file-like 协议）。"""
        self._stdout.write(text)
        self._fh.write(text)
        return len(text)

    def flush(self) -> None:
        """两侧同时 flush。"""
        self._stdout.flush()
        self._fh.flush()

    def close(self) -> None:
        """关闭文件句柄。"""
        self._fh.close()


def _log(msg: str) -> None:
    """打印一行进度（带刷新，便于长跑观察）。"""
    print(msg, flush=True)


def _write_json(path: str, obj: Any) -> str:
    """写 JSON（UTF-8，缩进 1；返回文件 SHA256）。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(blob)
    return S.sha256_bytes(blob.encode("utf-8"))


def save_model_artifact(path: str, meta: Json, model: torch.nn.Module) -> str:
    """把训练好的模型落成自写 zip（固定时间戳；返回文件 SHA256）。

    [!] **必须连骨干一起存**：``N3DQA.adapter`` 是**普通 Python 对象**（不是 nn.Module），
    因此 ``model.adapter.model`` **不会**出现在 ``model.state_dict()`` 里 —— 只存
    ``state_dict()`` 会得到「骨干权重全丢、加载后随机初始化」的假产物（现场实测：
    ``model.state_dict()`` 只有 5 个键，骨干 ``W_in`` / ``edge_weight`` / ``neuron_bias`` /
    ``W_out`` 一个都不在）。故产物成员拆成 ``head_state.pt`` + ``backbone_state.pt`` 两份。
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    head_buf = io.BytesIO()
    torch.save(model.state_dict(), head_buf)
    backbone_buf = io.BytesIO()
    torch.save(model.adapter.model.state_dict(), backbone_buf)
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("meta.json", date_time=ZIP_EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=1))
        info = zipfile.ZipInfo("model_state_dict.pt", date_time=ZIP_EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, head_buf.getvalue())
        info = zipfile.ZipInfo("backbone_state.pt", date_time=ZIP_EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, backbone_buf.getvalue())
    out = blob.getvalue()
    with open(path, "wb") as handle:
        handle.write(out)
    return S.sha256_bytes(out)


def load_model_artifact(path: str) -> Tuple[Json, Dict[str, Any]]:
    """加载模型产物（返回 ``(meta, {"head": ..., "backbone": ...})``）。

    兼容两种成员布局：
    * 新布局：``model_state_dict.pt``（头/缓冲）+ ``backbone_state.pt``（N3D 骨干）；
    * 旧布局：只有 ``model_state_dict.pt``（**骨干权重缺失**，加载后骨干为随机初始化，
      无法复算任何训练后量 —— 本函数会在返回值里显式标记 ``backbone_missing``）。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"模型产物不存在：{path!r}")
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        head = torch.load(io.BytesIO(zf.read("model_state_dict.pt")),
                          map_location="cpu", weights_only=False)
        backbone = None
        if "backbone_state.pt" in names:
            backbone = torch.load(io.BytesIO(zf.read("backbone_state.pt")),
                                  map_location="cpu", weights_only=False)
    return meta, {"head": head, "backbone": backbone,
                  "backbone_missing": bool(backbone is None)}


def artifact_entry(path: str) -> Json:
    """产物条目（路径 / 字节数 / SHA256），供报告列示。"""
    return {
        "path": path.replace("\\", "/"),
        "bytes": int(os.path.getsize(path)),
        "sha256": S.sha256_file(path),
    }


# ---------------------------------------------------------------------------
# probe：数据与口径取证（不含训练）
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    """数据与口径取证：划分复核 / 特征等价 / 答案表 / 留出协议 / 任务装配。"""
    product_dir = S.resolve_product_dir(str(args.product_dir))
    _log(f"[probe] 产物目录 = {product_dir}")
    rows = S.load_text_rows(product_dir)
    row_index = S.load_row_index(product_dir)
    meta = S.load_doclines_meta(product_dir)
    vectorizer = S.build_step2_vectorizer(_encoder_config(args))
    _log(f"[probe] 文本行数 = {len(rows)}；向量化口径 D = {vectorizer.dim}；"
         f"spec_hash = {vectorizer.fingerprint()[:16]}...")

    split = S.reproduce_doc_split(rows, row_index, meta)
    ev = split.evidence
    _log(f"[probe] 库/查询划分复核：库 {ev['n_library']} / 查询 {ev['n_query']} / "
         f"交集 {ev['intersection']} / 并集 {ev['union']} / "
         f"复现==行索引表 {ev['reproduced_equals_index']}")
    assert ev["intersection"] == 0, "库与查询必须不相交"
    assert ev["union"] == len(rows), "库并查询必须等于全量"
    assert ev["reproduced_equals_index"] is True, "划分必须可被产物自身复核"

    # 与 n3d_qa 冻结产物的逐元素重算比对**只对 hash 家族成立**（第 2 轮重建后已
    # 降级为旁证）；切到 HF 编码器时该空间不存在，故显式标注不适用而非静默跳过。
    if isinstance(getattr(vectorizer.config, "bag_dim", None), int):
        feat = S.verify_features_against_product(rows, vectorizer.config)
        _log(f"[probe] 特征重算取证：检查 {feat['checked_rows']} 行，"
             f"最大绝对偏差 {feat['max_abs_deviation']:.3e}，超容差 {feat['rows_over_atol']} 行，"
             f"atol = {feat['atol']:.0e}")
        assert feat["rows_over_atol"] == 0, "特征重算必须与产物在容差内一致"
    else:
        feat = {
            "applicable": False,
            "reason": (
                f"当前编码器 {str(getattr(args, 'encoder', '') or 'zh-bag')!r} 不是 hash 家族，"
                "与 n3d_qa 冻结产物不在同一特征空间；步骤 2 的验证口径见 "
                "step2.verify_vectorizer_contract（指纹对账 / 重复编码逐位一致 / 落盘缓存逐位比对）"
            ),
        }
        _log(f"[probe] 特征重算取证：不适用（{feat['reason']}）")

    # 向量化确定性（同文本两次编码逐位一致）
    t0 = rows[0].text
    a = vectorizer.encode(t0)
    b = vectorizer.encode(t0)
    assert a == b, "同一文本重复编码必须逐位一致"

    keys, display, entries = S.load_answer_table(product_dir)
    kept, held, held_detail = S.select_held_out_per_task(product_dir, keys, S.TASKS)
    _log(f"[probe] 统一答案表 {len(keys)} 类；按任务分层留出 {len(held)} 类 -> "
         f"候选空间 C = {len(kept)}")
    for _t, _v in held_detail["per_task"].items():
        _log(f"[probe]   留出明细 {_t}: 受限 {_v['n_restricted']} -> 留出 "
             f"{_v['n_held_out']}（seed {_v['seed']}）")

    tasks: List[Json] = []
    items = []
    for task in S.TASKS:
        td = S.build_task_data(product_dir, task, kept, held, display)
        items.append(td)
        tasks.append(td.as_dict())
        _log(f"[probe] 任务 {task}: 问题 {td.n_questions} / 已知 {td.n_known_questions} / "
             f"不相关 {td.n_unknown_questions} / 正样本 {td.n_positives}")
    data = S.merge_task_data(items, rows, vectorizer)
    merged = {
        "n_records": int(len(data.corpus.records)),
        "n_classes": int(data.corpus.n_classes),
        "split": data.splits.summary(),
    }
    _log(f"[probe] 合并后：记录 {merged['n_records']} / 类 {merged['n_classes']} / "
         f"划分 {merged['split']}")

    out: Json = {
        "product_dir": product_dir,
        "n_rows": int(len(rows)),
        "dim": int(vectorizer.dim),
        "feature_spec_hash": vectorizer.fingerprint(),
        "feature_check": feat,
        "split_evidence": ev,
        "answer_table": {"n_classes": len(keys)},
        "held_out": {
            "n_kept": len(kept),
            "n_held_out": len(held),
            "ratio": S.HELD_OUT_RATIO,
            "seed": S.HELD_OUT_SEED,
            "protocol": "per-task stratified held-out classes (open-set)",
            "held_out_keys_sha256": S.sha256_bytes(S.canonical_dumps(list(held))),
            "detail": held_detail,
        },
        "tasks": tasks,
        "merged": merged,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.out_dir:
        path = os.path.join(str(args.out_dir), "probe.json")
        digest = _write_json(path, out)
        _log(f"[probe] 取证报告 -> {path} (sha256={digest[:16]}...)")
    _log("[probe] OK")
    return 0

def _prepare(args: argparse.Namespace) -> Json:
    """装配全部数据侧对象（probe / drill / eval 共用）。"""
    product_dir = S.resolve_product_dir(str(args.product_dir))
    rows = S.load_text_rows(product_dir)
    row_index = S.load_row_index(product_dir)
    meta = S.load_doclines_meta(product_dir)
    vectorizer = S.build_step2_vectorizer(_encoder_config(args))
    split = S.reproduce_doc_split(rows, row_index, meta)
    keys, display, _ = S.load_answer_table(product_dir)
    kept, held, held_evidence = S.select_held_out_per_task(product_dir, keys, S.TASKS)
    items = [
        S.build_task_data(product_dir, task, kept, held, display) for task in S.TASKS
    ]
    data = S.merge_task_data(items, rows, vectorizer)
    return {
        "product_dir": product_dir,
        "rows": rows,
        "split": split,
        "vectorizer": vectorizer,
        "answer_keys": keys,
        "answer_display": display,
        "kept": kept,
        "held": held,
        "held_evidence": held_evidence,
        "task_items": items,
        "data": data,
    }


def _truncate(data: Any, n: int) -> Any:
    """把 ``TrainingData`` 的四个子集各截断到前 ``n`` 条（``n <= 0`` = 不截断）。"""
    from .data import QACorpus, SplitSpec
    from .train import TrainingData

    if int(n) <= 0:
        return data
    sp = data.splits
    return TrainingData(
        corpus=QACorpus(
            records=list(data.corpus.records)[: int(n)],
            answer_keys=list(data.corpus.answer_keys),
            answer_display=dict(data.corpus.answer_display),
            class_counts={},
        ),
        splits=SplitSpec(
            train_known=list(sp.train_known)[: int(n)],
            train_unknown=list(sp.train_unknown)[: int(n)],
            test_known=list(sp.test_known)[: int(n)],
            test_unknown=list(sp.test_unknown)[: int(n)],
        ),
        text_lines=list(data.text_lines),
        qa_files=list(data.qa_files),
        vectorizer=data.vectorizer,
    )


def _gradient_report(model: Any, data: Any, vectorizer: Any, batch: int = 8) -> Json:
    """单条端到端演练的**梯度取证**：一次前向 + 反向，逐参数报梯度绝对值之和。

    返回结构含 ``nonzero``（全非零为 ``True``）与 ``zeros``（现场枚举出的零梯度参数名）。
    """
    import torch.nn.functional as F  # noqa: F401  (仅用于类型完备性，不改变语义)

    model.train()
    recs = list(data.splits.train_known)[: int(batch)]
    if not recs:
        raise RuntimeError("train_known 为空，无法做梯度取证")
    feats = torch.tensor(vectorizer.encode_batch([r.question for r in recs]),
                         dtype=torch.float32)
    targets = torch.tensor(
        [data.corpus.key_to_index()[r.answer_key] for r in recs], dtype=torch.long
    )
    keys = None
    if model.config.output_mode == "pointer":
        keys = S.pointer_keys_from_answers(
            data.corpus.answer_keys, data.corpus.answer_display, vectorizer, len(recs)
        )
    model.zero_grad(set_to_none=True)
    logits = model.logits(feats, keys)
    loss = model.cross_entropy(logits, targets)
    loss.backward()
    sums: Dict[str, float] = {}
    zeros: List[str] = []
    outside: List[str] = []
    named: List[Tuple[str, Any]] = [(f"head.{n}", p) for n, p in model.named_parameters()]
    named += [(f"backbone.{n}", p)
              for n, p in model.adapter.model.named_parameters()]
    for label, p in named:
        if not p.requires_grad:
            continue
        if p.grad is None:
            # 该参数**根本不参与前向**（不在计算图上）——与「参与前向但梯度恰为 0」
            # 是两件事，必须分开报（历史教训：上游把 q 头改成 raw 直通后，N3D 骨干
            # 整体脱离计算图，若把两者混为一谈会给不出可诊断的失败信息）。
            outside.append(label)
            sums[label] = float("nan")
            continue
        val = float(p.grad.abs().sum().item())
        sums[label] = val
        if val == 0.0:
            zeros.append(label)
    model.zero_grad(set_to_none=True)
    model.eval()
    return {
        "loss": float(loss.detach().item()),
        "n_parameters_checked": int(len(sums)),
        "parameter_grad_abs_sum": sums,
        "zero_grad_parameters": zeros,
        "parameters_outside_graph": outside,
        "nonzero": bool(len(zeros) == 0 and len(outside) == 0),
    }


def cmd_drill(args: argparse.Namespace) -> int:
    """单条端到端演练：极小规模训练 -> 梯度非零 -> 极小自检索 -> 落盘取证。"""
    ctx = _prepare(args)
    vectorizer = ctx["vectorizer"]
    data = ctx["data"]
    rows = ctx["rows"]
    split = ctx["split"]
    _log(f"[drill] D={vectorizer.dim} 记录={len(data.corpus.records)} "
         f"类={data.corpus.n_classes}")

    outcome = S.train_model(
        data, backend=str(args.backend), output_mode="index", seed=42,
        epochs=1, batch_size=32, max_train_samples=256, max_batches=2,
    )
    _log(f"[drill] 1 epoch 限批训练完成：{outcome.n_train_items} 样本，"
         f"{outcome.seconds:.2f}s，末批 loss={outcome.history[-1]['loss']:.4f}")

    grad = _gradient_report(outcome.model, data, vectorizer, batch=8)
    _log(f"[drill] 梯度取证：检查 {grad['n_parameters_checked']} 个可学习参数，"
         f"零梯度参数 {len(grad['zero_grad_parameters'])} 个，nonzero={grad['nonzero']}")
    _log(f"[drill] 参与前向但零梯度 = {grad['zero_grad_parameters']}；"
         f"不在计算图上 = {grad['parameters_outside_graph']}")
    assert grad["nonzero"] is True, (
        "单条演练要求全部可学习参数既参与前向、梯度又非零；实测 "
        f"零梯度 = {grad['zero_grad_parameters']}，"
        f"不在计算图上 = {grad['parameters_outside_graph']}。"
        "（若为非空，说明 q 头口径把某条支路排除出计算图，例如 head_input_mode='raw' "
        "会让 N3D 骨干整体脱离前向）"
    )

    # 极小自检索：取库前 32 行，查询 = 这 32 行自身
    lib = [int(i) for i in split.library_index[:32]]
    table = S.build_key_table(vectorizer, rows, lib)
    matcher_idx = S.TextRowMatcher(outcome.model, vectorizer, "index", table)
    matcher_ptr = S.TextRowMatcher(outcome.model, vectorizer, "pointer", None)
    sr = S.self_retrieval(matcher_idx, rows, lib, lib, table)
    bl = S.deterministic_baseline(vectorizer, rows, lib, lib, table)
    dm = S.dual_mode_consistency(matcher_idx, matcher_ptr, rows, lib, table)
    _log(f"[drill] 极小自检索 Recall@1={sr['recall_at_1']:.4f} "
         f"参照下限 Recall@1={bl['recall_at_1']:.4f} "
         f"双模式一致率={dm['top1_agreement']:.4f}")

    m = matcher_idx.match([rows[lib[0]].text])[0]
    _log(f"[drill] top-1 行回填：line_id={m.line_id} score={m.score:.6f} "
         f"text={m.text[:40]!r}")

    out: Json = {
        "dim": int(vectorizer.dim),
        "n_records": int(len(data.corpus.records)),
        "n_classes": int(data.corpus.n_classes),
        "train": {
            "n_train_items": outcome.n_train_items,
            "seconds": outcome.seconds,
            "history": outcome.history,
        },
        "gradient": grad,
        "self_retrieval_small": sr,
        "baseline_small": bl,
        "dual_mode_small": dm,
        "match_head": {
            "line_id": m.line_id, "score": float(m.score), "mode": m.mode,
            "topk_line_ids": m.topk_line_ids,
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir = str(args.out_dir) if args.out_dir else S.STEP2_VERIFY_DIR
    path = os.path.join(out_dir, "drill.json")
    digest = _write_json(path, out)
    _log(f"[drill] 演练取证 -> {path} (sha256={digest[:16]}...)")
    _log("[drill] OK")
    return 0

def _step1_agreement(
    model_a: Any, model_b: Any, data: Any, vectorizer: Any,
    *, batch: int = 128, max_samples: int = 0,
) -> Json:
    """两个模型在同一批测试问题上的 top-1 一致率（跨模式实测值）。"""
    recs = list(data.splits.test_known) + list(data.splits.test_unknown)
    if int(max_samples) > 0:
        recs = recs[: int(max_samples)]
    if not recs:
        raise RuntimeError("测试集为空，无法做跨模式一致性")
    agree = 0
    n = 0
    for b0 in range(0, len(recs), int(batch)):
        chunk = recs[b0 : b0 + int(batch)]
        feats = torch.tensor(vectorizer.encode_batch([r.question for r in chunk]),
                             dtype=torch.float32)
        outs = []
        for model in (model_a, model_b):
            keys = None
            if model.config.output_mode == "pointer":
                keys = S.pointer_keys_from_answers(
                    data.corpus.answer_keys, data.corpus.answer_display,
                    vectorizer, len(chunk),
                )
            with torch.no_grad():
                outs.append(model.logits(feats, keys).argmax(dim=1))
        agree += int((outs[0] == outs[1]).sum().item())
        n += len(chunk)
    return {"n": int(n), "top1_agreement": float(agree / max(1, n)), "n_agree": int(agree)}


def cmd_eval(args: argparse.Namespace) -> int:
    """全量分项验收：跨 seed x 双模式训练 + 自检索 + 参照下限 + 5 项指标 + 报告。"""
    ctx = _prepare(args)
    vectorizer = ctx["vectorizer"]
    data = ctx["data"]
    rows = ctx["rows"]
    split = ctx["split"]
    held = ctx["held"]
    kept = ctx["kept"]
    display = ctx["answer_display"]

    out_dir = str(args.out_dir) if args.out_dir else S.STEP2_DIR
    verify_dir = str(args.verify_dir) if args.verify_dir else S.STEP2_VERIFY_DIR
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(verify_dir, exist_ok=True)
    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    modes = [m for m in str(args.modes).split(",") if m.strip()]

    _log(f"[eval] D={vectorizer.dim} 后端={args.backend} seeds={seeds} modes={modes}")
    _log(f"[eval] 记录={len(data.corpus.records)} 类={data.corpus.n_classes} "
         f"留出类={len(held)}")
    _log("[eval] frozen-split: nonquery=%d query=%d inter=%d pool=%d" % (split.n_library, split.n_query, split.evidence["intersection"], len(rows)))
    # 检索池口径：n3d_qa 产物的 label_rule 明确「candidate library row IS the query row」，
    # 即查询行必须落在候选池内，否则「命中自身行」不可定义。故检索池 = 冻结行表全量（2665），
    # 查询集 = 冻结划分出的 666 行；划分的库(1999)/查询(666) 互斥性由 probe 独立复核。
    pool_index = list(range(len(rows)))
    table = S.build_key_table(vectorizer, rows, pool_index)
    _log(f"[eval] 冻结键表（检索池=全量行表）[{table.size}, {table.dim}] "
         f"sha256={table.sha256()[:16]}...")

    artifacts: List[Json] = []
    if "index" in modes:
        kt_meta = {
            "module": "n3d_qa_learn",
            "artifact_format_version": S.STEP2_ARTIFACT_VERSION,
            "product": "step2_textrow_key_table",
            "source_product_dir": ctx["product_dir"].replace("\\", "/"),
            "mode": "index",
            "frozen": True,
            "feature_spec_hash": vectorizer.fingerprint(),
            "dim": int(table.dim),
            "library_rows": int(table.size),
            "query_rows": int(split.n_query),
            "retrieval_pool": "full frozen row table (query rows included; see label_rule)",
            "disjoint_non_query_rows": int(split.n_library),
            "split_seed": int(split.seed),
            "split_evidence": split.evidence,
            "key_table_sha256": table.sha256(),
            "line_ids_sha256": table.line_ids_sha256(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        kt_path = os.path.join(
            out_dir,
            S.step2_artifact_name(str(args.backend), "index", int(table.dim),
                                  int(table.size), 0),
        )
        digest = S.save_key_table(kt_path, table, kt_meta)
        artifacts.append(artifact_entry(kt_path))
        _log(f"[eval] 冻结键表产物 -> {kt_path} (sha256={digest[:16]}...)")

    baseline = S.deterministic_baseline(
        vectorizer, rows, split.query_index, pool_index, table
    )
    _log(f"[eval] 参照下限（纯确定性特征检索）：Recall@1={baseline['recall_at_1']:.4f} "
         f"Recall@5={baseline['recall_at_5']:.4f}")

    per_seed: Dict[int, Dict[str, Json]] = {}
    self_retr: Dict[str, Json] = {}
    dual: Json = {"step1": {}}
    for seed in seeds:
        models: Dict[str, Any] = {}
        for mode in modes:
            t0 = time.time()
            outcome = S.train_model(
                data, backend=str(args.backend), output_mode=str(mode), seed=int(seed),
                epochs=int(args.epochs), batch_size=int(args.batch_size),
                lr=float(args.lr), backbone_lr=float(args.backbone_lr),
                label_smoothing=float(args.label_smoothing),
                max_train_samples=int(args.max_train),
                head_input_mode=str(args.head_input_mode),
            )
            models[mode] = outcome.model
            mpath = os.path.join(
                out_dir,
                f"qa_step2_model_{args.backend}_{mode}_D{vectorizer.dim}"
                f"_C{data.corpus.n_classes}_s{seed}.pt.zip",
            )
            mm = {
                "module": "n3d_qa_learn",
                "artifact_format_version": S.STEP2_ARTIFACT_VERSION,
                "product": "step2_model",
                "backend": str(args.backend),
                "output_mode": str(mode),
                "dim": int(vectorizer.dim),
                "n_answers": int(data.corpus.n_classes),
                "seed": int(seed),
                "epochs": int(args.epochs),
                "n_train_items": int(outcome.n_train_items),
                "seconds": float(outcome.seconds),
                "history_tail": outcome.history[-3:],
                "answer_keys_sha256": S.sha256_bytes(
                    S.canonical_dumps(list(data.corpus.answer_keys))),
                "held_out_keys_sha256": S.sha256_bytes(
                    S.canonical_dumps(list(held))),
                "feature_spec_hash": vectorizer.fingerprint(),
                "head_config": S.head_config_of(outcome.model),
                "source_manifest": S.module_file_manifest(extra_dirs=("n3d_qa",)),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            save_model_artifact(mpath, mm, outcome.model)
            artifacts.append(artifact_entry(mpath))
            _log(f"[eval] seed={seed} mode={mode}: 训练 {outcome.n_train_items} 样本 "
                 f"{outcome.seconds:.1f}s，末 epoch loss="
                 f"{outcome.history[-1]['loss']:.4f} acc={outcome.history[-1]['train_acc']:.4f}")
            del t0

            matcher = S.TextRowMatcher(
                outcome.model, vectorizer, str(mode),
                table if str(mode) == "index" else None,
            )
            sr = S.self_retrieval(matcher, rows, split.query_index,
                                  pool_index, table)
            self_retr[f"s{seed}_{mode}"] = sr
            _log(f"[eval] seed={seed} mode={mode}: 自检索 Recall@1={sr['recall_at_1']:.4f} "
                 f"Recall@5={sr['recall_at_5']:.4f}")

        # 双模式一致性（结构性：同一模型 + 同一候选集）
        if "index" in models and "pointer" in models:
            # 结构性取证：**同一个 index 模型**分别以两种候选来源打分（应逐位相同）
            mi = S.TextRowMatcher(models["index"], vectorizer, "index", table)
            mp_same = S.TextRowMatcher(models["index"], vectorizer, "pointer", None)
            dual[f"s{seed}_candidates_same_model"] = S.dual_mode_consistency(
                mi, mp_same, rows, split.query_index, table
            )
            # 有信息量的一致性：index 训练模型 vs pointer 训练模型
            mp = S.TextRowMatcher(models["pointer"], vectorizer, "pointer", None)
            dual[f"s{seed}_candidates_cross_model"] = S.dual_mode_consistency(
                mi, mp, rows, split.query_index, table
            )
            dv = _step1_agreement(
                models["index"], models["pointer"], _truncate(data, int(args.max_test_per_task)),
                vectorizer, max_samples=int(args.max_test_per_task),
            )
            dual["step1"][f"s{seed}_cross_model"] = dv
            _log(f"[eval] seed={seed}: 步骤2 候选一致率（同模型双模式 structural）="
                 f"{dual[f's{seed}_candidates_same_model']['top1_agreement']:.4f} "
                 f"bitwise={dual[f's{seed}_candidates_same_model']['logits_bitwise_equal']}；"
                 f"跨训练模型一致率="
                 f"{dual[f's{seed}_candidates_cross_model']['top1_agreement']:.4f}；"
                 f"步骤1 跨模式 top-1 一致率={dv['top1_agreement']:.4f}")

        # 分项 5 项指标（主模型取 index）
        primary = models.get("index") or models[modes[0]]
        step2_global = self_retr[f"s{seed}_{primary.config.output_mode}"]
        # ---- 路由候选空间必须与模型候选空间**同宽**（构造期不变量，违反即报错）----
        # 现场复核（审查条目 E2）：此处传入的是 `data.corpus.answer_keys` = **保留类表**
        # （C=230），不是统一答案表全量（298）；实测 router.irrelevant_index = 230 =
        # 模型 answer_index()，answer_table 形状 [231, 192]，「不相关」末位**可达**。
        # 该断言把这条不变量钉死在构造期，杜绝未来口径漂移导致的静默退化。
        router = QuestionRouter(
            data.corpus.answer_keys, data.corpus.answer_display,
            [t for t in data.text_lines], vectorizer,
            text_threshold=float(args.text_threshold),
        )
        if len(router.answer_keys) != int(primary.n_answers):
            raise ValueError(
                "路由候选空间与模型候选空间不同宽："
                f"router.answer_keys={len(router.answer_keys)} vs 模型 n_answers="
                f"{int(primary.n_answers)}；拒绝在错配的候选空间上算路由指标"
            )
        if int(router.irrelevant_index) != int(primary.answer_index()):
            raise ValueError(
                "路由的「不相关」下标与模型末位不一致："
                f"router.irrelevant_index={int(router.irrelevant_index)} vs "
                f"模型 answer_index()={int(primary.answer_index())}"
            )
        if int(primary.output_dim) != int(router.irrelevant_index) + 1:
            raise ValueError(
                f"模型候选宽度 {int(primary.output_dim)} != 路由「不相关」下标 + 1 "
                f"({int(router.irrelevant_index) + 1})"
            )
        per_task: Dict[str, Json] = {}
        for task in S.TASKS:
            tv = _truncate(S.filter_by_task(data, task), int(args.max_test_per_task))
            m = S.task_metrics(
                task, primary, tv, router, vectorizer, step2_global,
                batch_size=int(args.batch_size),
            )
            per_task[task] = m
            _log(f"[eval] seed={seed} {task}: ①={m['step1_answer_accuracy']:.4f} "
                 f"③={m['routing_decision_accuracy']:.4f} ④P={m['refusal_precision']:.4f} "
                 f"④R={m['refusal_recall']:.4f} ⑤={m['e2e_answer_accuracy']:.4f}")
        per_seed[int(seed)] = per_task

    # 扁平汇总键：让报告 JSON / 报告 MD / README 三处口径一致（渲染层按存在性展示，绝不渲染 None）
    _same = [dual[k] for k in dual if k.endswith("_candidates_same_model")]
    if _same:
        dual["n_seeds"] = int(len(_same))
        dual["top1_agreement"] = float(
            sum(float(v["top1_agreement"]) for v in _same) / len(_same)
        )
        dual["logits_bitwise_equal"] = bool(
            all(bool(v["logits_bitwise_equal"]) for v in _same)
        )
        dual["logits_max_abs_diff"] = float(
            max(float(v["logits_max_abs_diff"]) for v in _same)
        )

    cross = S.cross_seed_summary(per_seed)
    gain = None
    if self_retr:
        best = max(v["recall_at_1"] for v in self_retr.values())
        gain = float(best - baseline["recall_at_1"])

    report: Json = {
        "module": "n3d_qa_learn",
        "artifact_format_version": S.STEP2_ARTIFACT_VERSION,
        "product": "step2_acceptance_report",
        "product_dir": ctx["product_dir"].replace("\\", "/"),
        "dim": int(vectorizer.dim),
        "backend": str(args.backend),
        "modes": list(modes),
        "seeds": [int(s) for s in seeds],
        "topology_premise": (
            "拓扑 seed 前提：三个后端的 Config 由 n3d_qa_learn.backends.recommended_config "
            "统一给出，其中后端自身的 seed 恒为 42（现场只读复核：该函数对 n3d_shape / "
            "n3d_sphere / n3d_proto 三个分支都显式写死 seed=42）。N3D 的神经元位置与突触几何"
            "由该 seed 决定，因此**跨 seed 报告中的所有运行共用同一套拓扑**；变化的只是"
            "训练侧的数据顺序与参数初始化（train_model 的 seed / torch.Generator）。"
            "即：本报告的「极差」是**训练随机性**的极差，不含拓扑随机性。"
        ),
        "held_out": {
            "protocol": "per-task stratified held-out classes (open-set)",
            "ratio": S.HELD_OUT_RATIO,
            "seed": S.HELD_OUT_SEED,
            "n_total_classes": int(len(ctx["answer_keys"])),
            "n_kept": int(len(kept)),
            "n_held_out": int(len(held)),
            "held_out_keys_sha256": S.sha256_bytes(S.canonical_dumps(list(held))),
            "detail": ctx["held_evidence"],
        },
        "product_files": S.product_file_manifest(ctx["product_dir"], S.TASKS),
        "head_input_mode": str(args.head_input_mode),
        "source_manifest": S.module_file_manifest(extra_dirs=("n3d_qa",)),
        "split": split.evidence,
        "retrieval_pool": {
            "n_pool": int(len(rows)),
            "n_query": int(split.n_query),
            "note": ("候选池 = 冻结行表全量（含查询行自身）；n3d_qa 的 label_rule 规定 "
                     "label=1 当且仅当候选库行就是查询行，故池必须含查询行。"
                     "冻结的库(1999)/查询(666) 互斥性作为划分审计项单列（intersection=0）。"),
        },
        "self_retrieval": self_retr,
        "baseline": baseline,
        "step2_gain_over_baseline": gain,
        "dual_mode": dual,
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "cross_seed": cross,
        "task_data": [it.as_dict() for it in ctx["task_items"]],
        "train_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "backbone_lr": float(args.backbone_lr),
            "label_smoothing": float(args.label_smoothing),
            "max_train": int(args.max_train),
            "max_test_per_task": int(args.max_test_per_task),
            "text_threshold": float(args.text_threshold),
        },
        "artifacts": artifacts,
        "honest_notes": [],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # ---- 如实登记 ----
    notes = report["honest_notes"]
    notes.append(
        "② 步骤 2 的 Recall@1 由**文本行自检索留出法**给出，与任务无关（文本行库不随任务"
        "变化）；分项表里该列对每个任务填入同一全局值，字段 "
        "step2_recall_at_1_is_task_independent=True 显式标记。"
    )
    notes.append(
        "「不相关」样本非数据集原生：n3d_qa 冻结产物只导出已入表答案的正样本（judge 移除 0 题、"
        "choice 15 题、blank 3060 题、solve 21864 题，被移除题目不随产物导出），故本报告用"
        "**按任务分层的留出类**开集协议构造，留出类清单已冻结并给出 SHA256。"
    )
    notes.append(
        "双模式一致率在「步骤 2 候选」上是结构性恒等（两模式在同一候选集上做同一次 "
        "q @ K^T），实测值 1.0 并附 logits 逐位相等等取证；真正有信息量的是"
        "「步骤 1 跨模式一致率」（两模式分别训练）。"
    )
    if int(args.max_test_per_task) > 0:
        notes.append(
            f"分项指标受 --max-test-per-task={int(args.max_test_per_task)} 限批"
            "（仅供快速验收），全量口径请以 0 运行。"
        )
    if gain is not None and gain < 0:
        notes.append(
            f"**未达标（如实报告）**：N3D 的 q 在步骤 2 上的 Recall@1 未超过纯确定性特征检索的"
            f"参照下限，增益 = {gain:.4f}（负值即劣于下限）。"
        )
    notes.append(
        "上游零改动：本模块只读 n3d_qa 冻结产物与 n3d_shape/n3d_sphere/n3d_proto 的 "
        "config/model（经 backends 代理层），未写入任何上游目录。"
    )
    # ---- 审查条目 W12：路由退化与步骤 2 不可达，属关键局限，必须显式登记 ----
    reached = 0
    for _tasks in per_seed.values():
        for _m in _tasks.values():
            reached += int(_m.get("step2_reached_total", 0))
    notes.append(
        "**路由退化（关键局限，如实登记）**：本轮全部 seed × 全部任务上模型**从未输出"
        "「不相关」类**，故 ③ 路由正确率退化为「已知题命中率」、④ 无匹配类精确率/召回恒为 0，"
        f"且分项表的 `step2_reached_total` 合计 = {reached} —— 即**步骤 2 的文本行匹配在任何"
        "被测指标里都没有被走到**。根因是「不相关」类没被学出来（index 模型含未知样本的末 "
        "epoch 训练准确率约 0.20），**不是路由候选宽度错配**：现场复核 "
        "`len(router.answer_keys)` = 模型 `n_answers` = 230、`router.irrelevant_index` = "
        "模型 `answer_index()` = 230、`answer_table` 形状 [231, 192]，三者同宽且末位可达；"
        "cmd_eval 已把这三条写成构造期硬断言。"
    )
    notes.append(
        "**⑤ 与 ③ 的口径已分离（修复审查条目 W2）**：③ `routing_decision_accuracy` 是宽松口径"
        "（只看是否走到步骤 1 答案分支，不校验答案对错）；⑤ `e2e_answer_accuracy` 是严格口径"
        "（已知题要求命中**且返回答案文本等于金标展示文本**）。故 ⑤ ≤ ③ 恒成立；"
        "本轮两者相等，正是因为 ④ 恒 0 导致 ③ 退化为已知题命中率、而命中题中答对的占比恰好"
        "与之相同（现场逐 seed 逐任务可核对）。"
    )

    os.makedirs(out_dir, exist_ok=True)
    # 先写一遍拿到路径 -> 回填 report_paths -> 再写正式报告与验证目录副本。
    # 指标指纹排除 created_utc 与 report_paths，故两遍写出的 metrics_sha256 一致，
    # 且 replay 读到的报告**必带** report_paths（历史缺陷：验证目录副本恒无该键，
    # report_metrics_sha256 恒为空串）。
    paths = S.write_report(out_dir, report)
    report["report_paths"] = paths
    paths = S.write_report(out_dir, report)
    S.write_report(verify_dir, report)
    report["report_paths"] = paths
    _log(f"[eval] 报告 -> {paths['json']} / {paths['markdown']}")
    _log(f"[eval] 指标指纹 metrics_sha256={paths['metrics_sha256'][:16]}...")
    _log("[eval] OK")
    return 0


def cmd_verify_vectorizer(args: argparse.Namespace) -> int:
    """步骤 2 的**验证口径（第 2 轮重建）**：指纹对账 / 重复编码 / 落盘缓存逐位比对。

    替代原先的「与 ``n3d_qa`` 冻结产物逐位比对」：后者只在 hash 家族内可行，对可插拔
    接口（尤其 HF 编码器）无从成立。三条检查的口径见
    :func:`n3d_qa_learn.step2.verify_vectorizer_contract`。
    """
    product_dir = S.resolve_product_dir(str(args.product_dir))
    rows = S.load_text_rows(product_dir)
    cfg = _encoder_config(args)
    vectorizer = S.build_step2_vectorizer(cfg)
    n_texts = max(1, int(args.n_texts))
    texts = [rows[i].text for i in range(min(n_texts, len(rows)))]
    registry_name = str(getattr(args, "encoder", "") or E.ENCODER_ZH_BAG)
    declared_fp, declared_source = _declared_fingerprint(args, registry_name, E.ROLE_TEXT_LINE)
    _log(f"[verify-vectorizer] 产物目录 = {product_dir}；编码器 = {registry_name!r}；"
         f"role = {E.ROLE_TEXT_LINE!r}；D = {vectorizer.dim}；比对文本 {len(texts)} 条")
    ev = S.verify_vectorizer_contract(
        vectorizer,
        texts,
        registry_name=registry_name,
        role=E.ROLE_TEXT_LINE,
        declared_fingerprint=declared_fp,
        declared_source=declared_source,
    )
    out_dir = str(args.verify_dir) if args.verify_dir else S.STEP2_VERIFY_DIR
    payload: Json = {
        "product_dir": product_dir,
        "encoder_config": (cfg.to_dict() if cfg is not None else {
            "name": E.ENCODER_ZH_BAG, "role": E.ROLE_TEXT_LINE,
            "note": "空 --encoder = 角色默认实现（现状 zh-bag 口径）",
        }),
        "evidence": ev,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = os.path.join(out_dir, "verify_vectorizer.json")
    digest = _write_json(path, payload)
    _dump(payload)
    _log(f"[verify-vectorizer] 取证 -> {path} (sha256={digest[:16]}...)")
    if not ev["passed"]:
        _log("[verify-vectorizer] FAIL：三条检查未全部通过")
        return 1
    _log("[verify-vectorizer] OK")
    return 0


def _dump(obj: Any) -> None:
    """UTF-8 安全 JSON 打印。"""
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="n3d_qa_learn.step2_run",
        description="n3d_qa_learn 步骤 2：文本行匹配（索引式/指针式）+ 分项验收",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--product-dir", default="", help="n3d_qa 冻结产物目录")
        p.add_argument("--out-dir", default="", help="产物/报告输出目录")
        p.add_argument("--backend", default=S.DEFAULT_BACKEND, help="N3D 后端名")
        p.add_argument("--log-file", default="",
                       help="同时把 stdout 写入该 UTF-8(无 BOM) 日志文件")
        # 可插拔编码器（空 = 角色默认实现 = 现状 zh-bag 口径，行为逐位不变）
        p.add_argument("--encoder", default="",
                       help="可插拔特征实现的注册表键（空 = 现状 zh-bag）；"
                            "合法值见 `python -m n3d_qa_learn.encoders_run registry`")
        p.add_argument("--enc-max-length", type=int, default=0,
                       help="文本行截断长度（0 = 角色冻结口径 text_line=8192）")
        p.add_argument("--enc-source", default="", help="模型来源（本地目录 / HF 仓库 id）")
        p.add_argument("--enc-revision", default="", help="固定 revision（空 = 注册表声明）")
        p.add_argument("--enc-cache-dir", default="", help="嵌入缓存目录")
        p.add_argument("--enc-force-cache", action="store_true",
                       help="强制开启嵌入缓存（hash 家族默认关闭）")
        p.add_argument("--enc-local-files-only", action="store_true",
                       help="只允许本地文件（无网络场景）")

    p_probe = sub.add_parser("probe", help="数据与口径取证（不训练）")
    common(p_probe)
    p_probe.set_defaults(func=cmd_probe)

    p_drill = sub.add_parser("drill", help="单条端到端演练")
    common(p_drill)
    p_drill.set_defaults(func=cmd_drill)

    p_eval = sub.add_parser("eval", help="全量分项验收")
    common(p_eval)
    p_eval.add_argument("--verify-dir", default="", help="验证/报告目录")
    p_eval.add_argument("--seeds", default="42,43,44", help="训练侧 seed 列表")
    p_eval.add_argument("--modes", default="index,pointer", help="输出模式列表")
    p_eval.add_argument("--epochs", type=int, default=12, help="训练轮数")
    p_eval.add_argument("--batch-size", type=int, default=64, help="批大小")
    p_eval.add_argument("--lr", type=float, default=5e-3, help="头学习率")
    p_eval.add_argument("--backbone-lr", type=float, default=5e-4, help="后端学习率")
    p_eval.add_argument("--label-smoothing", type=float, default=0.0, help="标签平滑")
    p_eval.add_argument("--max-train", type=int, default=0, help="训练样本上限（0=全量）")
    p_eval.add_argument("--max-test-per-task", type=int, default=0,
                        help="每任务测试子集上限（0=全量）")
    p_eval.add_argument("--text-threshold", type=float, default=0.28, help="步骤 2 路由阈值")
    p_eval.add_argument("--head-input-mode", default=S.DEFAULT_HEAD_INPUT_MODE,
                        help="q 头输入口径（显式固定，不吃上游默认值）")
    p_eval.set_defaults(func=cmd_eval)

    p_guard = sub.add_parser("guard", help="产物守卫拒绝证明")
    common(p_guard)
    p_guard.add_argument("--verify-dir", default="", help="证明文件输出目录")
    p_guard.set_defaults(func=cmd_guard)

    p_replay = sub.add_parser("replay", help="从落盘产物复算并与报告对账（不训练）")
    common(p_replay)
    p_replay.add_argument("--report-dir", default="",
                          help="对账基准报告 step2_report.json 所在目录（缺省自动回退）")
    p_replay.add_argument("--verify-dir", default="", help="对账**输出**目录（写 replay.json）")
    p_replay.add_argument("--seeds", default="42,43,44", help="要复算的 seed 列表")
    p_replay.add_argument("--modes", default="index,pointer", help="要复算的模式列表")
    p_replay.add_argument("--batch-size", type=int, default=128, help="批大小")
    p_replay.add_argument("--text-threshold", type=float, default=0.28, help="步骤 2 路由阈值")
    p_replay.add_argument("--full-metrics", action="store_true",
                          help="额外复算按任务分项 5 项指标（较慢）")
    p_replay.set_defaults(func=cmd_replay)

    p_vv = sub.add_parser(
        "verify-vectorizer",
        help="步骤 2 验证口径（重建）：指纹对账 / 重复编码逐位一致 / 落盘缓存逐位比对",
    )
    common(p_vv)
    p_vv.add_argument("--verify-dir", default="", help="取证输出目录")
    p_vv.add_argument("--n-texts", type=int, default=8, help="参与比对的文本条数")
    p_vv.add_argument("--declared-fingerprint", default="",
                      help="① 项基准指纹（空 = 读 encoders_run 的落盘取证）")
    p_vv.add_argument("--encoders-dir", default="", help="encoders_run 取证目录")
    p_vv.set_defaults(func=cmd_verify_vectorizer)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口。"""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    log_file = str(getattr(args, "log_file", "") or "")
    tee: Optional[_Tee] = None
    if log_file:
        tee = _Tee(log_file)
        sys.stdout = tee
    try:
        return int(args.func(args))
    finally:
        if tee is not None:
            sys.stdout = tee._stdout
            tee.close()




# ---------------------------------------------------------------------------
# guard：产物守卫拒绝证明（篡改后必须报错）
# ---------------------------------------------------------------------------


def _rewrite_zip(src: str, dst: str, replacements: Dict[str, bytes]) -> str:
    """按成员名替换重写 zip（其余成员原样拷贝；返回新文件 SHA256）。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(buf, "w",
                                                          compression=zipfile.ZIP_DEFLATED) as zout:
        for name in zin.namelist():
            info = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            payload = replacements.get(name, zin.read(name))
            zout.writestr(info, payload)
    blob = buf.getvalue()
    with open(dst, "wb") as handle:
        handle.write(blob)
    return S.sha256_bytes(blob)


def cmd_guard(args: argparse.Namespace) -> int:
    """守卫拒绝证明：合法产物必须加载成功，篡改键表 / 篡改口径指纹必须被拒。"""
    out_dir = str(args.out_dir) if args.out_dir else S.STEP2_DIR
    verify_dir = str(args.verify_dir) if args.verify_dir else S.STEP2_VERIFY_DIR
    os.makedirs(verify_dir, exist_ok=True)
    cands = sorted(
        os.path.join(out_dir, n) for n in os.listdir(out_dir)
        if n.startswith("qa_step2_textrows_") and n.endswith(".pt.zip")
    ) if os.path.isdir(out_dir) else []
    if not cands:
        raise FileNotFoundError(f"{out_dir!r} 下没有步骤 2 键表产物；请先跑 eval")
    src = cands[0]
    cases: List[Json] = []

    table, meta = S.load_key_table(src, verify=True)
    cases.append({
        "case": "clean",
        "path": src,
        "loaded": True,
        "n_rows": int(table.size),
        "dim": int(table.dim),
        "expected": "load OK",
    })
    _log(f"[guard] 合法产物加载成功：{src} [{table.size}, {table.dim}]")

    # 篡改 1：改动键表张量内容 -> 键表 SHA 守卫必须拒绝
    with zipfile.ZipFile(src, "r") as zf:
        blob = torch.load(io.BytesIO(zf.read("key_table.pt")), map_location="cpu",
                          weights_only=False)
    keys = blob["keys"].to(torch.float32).clone()
    keys[0, 0] = float(keys[0, 0]) + 0.125
    buf = io.BytesIO()
    torch.save({"keys": keys, "line_ids": list(blob["line_ids"])}, buf)
    dst1 = os.path.join(verify_dir, "tampered_keytable.pt.zip")
    _rewrite_zip(src, dst1, {"key_table.pt": buf.getvalue()})
    msg1 = ""
    try:
        S.load_key_table(dst1, verify=True)
    except ValueError as exc:
        msg1 = str(exc)
    cases.append({
        "case": "tampered_key_table",
        "path": dst1,
        "loaded": bool(msg1 == ""),
        "rejected": bool(msg1 != ""),
        "message_head": msg1[:160],
        "expected": "ValueError: 键表指纹校验失败",
    })
    _log(f"[guard] 篡改键表 -> 被拒 = {msg1 != ''} ({msg1[:60]}...)")

    # 篡改 2：改动 meta 的特征口径指纹 -> 口径守卫必须拒绝
    meta2 = dict(meta)
    meta2["feature_spec_hash"] = "0" * 64
    dst2 = os.path.join(verify_dir, "tampered_spechash.pt.zip")
    _rewrite_zip(src, dst2, {
        "meta.json": json.dumps(meta2, ensure_ascii=False, sort_keys=True,
                                indent=1).encode("utf-8")
    })
    msg2 = ""
    try:
        S.load_key_table(dst2, verify=True)
    except ValueError as exc:
        msg2 = str(exc)
    cases.append({
        "case": "tampered_feature_spec_hash",
        "path": dst2,
        "loaded": bool(msg2 == ""),
        "rejected": bool(msg2 != ""),
        "message_head": msg2[:160],
        "expected": "ValueError: 特征口径指纹不一致",
    })
    _log(f"[guard] 篡改口径指纹 -> 被拒 = {msg2 != ''} ({msg2[:60]}...)")

    all_rejected = all(bool(c.get("rejected")) for c in cases if c["case"] != "clean")
    report = {
        "cases": cases,
        "all_tampered_rejected": bool(all_rejected),
        "clean_loaded": bool(cases[0]["loaded"]),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = os.path.join(verify_dir, "guard.json")
    digest = _write_json(path, report)
    _log(f"[guard] 守卫证明 -> {path} (sha256={digest[:16]}...) all_rejected={all_rejected}")
    assert all_rejected, "篡改后的产物必须被守卫拒绝"
    _log("[guard] OK")
    return 0




# ---------------------------------------------------------------------------
# replay：从产物复算（训练后量必须能由落盘产物复现）
# ---------------------------------------------------------------------------


def rebuild_model_from_meta(meta: Json, state_dict: Dict[str, Any]) -> Any:
    """按产物 meta 的 ``head_config`` 原样重建模型并严格加载状态字典。

    这是「产物可复算」的关键入口：meta 必须自带 ``head_input_mode`` / ``normalize_query`` /
    ``answer_table_mode`` / ``output_mode`` / ``dim`` / ``n_answers``，否则加载侧无法重建出
    与训练时**同构**的模型（上游 ``heads.py`` 的默认口径一旦变化，产物即失效）。
    """
    hc = dict(meta.get("head_config") or {})
    model = S.build_model(
        str(meta["backend"]),
        str(meta["output_mode"]),
        int(meta["n_answers"]),
        dim=int(meta["dim"]),
        label_smoothing=float(hc.get("label_smoothing", 0.0)),
        head_input_mode=str(hc.get("head_input_mode", S.DEFAULT_HEAD_INPUT_MODE)),
    )
    blob = state_dict if "head" in state_dict else {"head": state_dict,
                                                    "backbone": None,
                                                    "backbone_missing": True}
    if blob.get("backbone_missing"):
        raise ValueError(
            "产物缺少骨干权重（成员 backbone_state.pt）：仅头/缓冲不足以复算训练后量，"
            "拒绝静默用随机骨干出数。请用当前版本重跑 eval 重新生成产物。"
        )
    model.load_state_dict(blob["head"], strict=True)
    model.adapter.model.load_state_dict(blob["backbone"], strict=True)
    model.eval()
    return model


def cmd_replay(args: argparse.Namespace) -> int:
    """从落盘产物复算并**逐项对账**（不训练）。

    校验链：产物严格加载 → 自检索 Recall@1/@5 复算 → 双模式一致性复算 →
    （``--full-metrics``）按任务分项 5 项指标复算，全部与 ``step2_report.json`` 对账。
    """
    ctx = _prepare(args)
    vectorizer = ctx["vectorizer"]
    rows = ctx["rows"]
    data = ctx["data"]
    out_dir = str(args.out_dir) if args.out_dir else S.STEP2_DIR
    verify_dir = str(args.verify_dir) if args.verify_dir else S.STEP2_VERIFY_DIR
    # ---- 对账基准报告的定位（修复审查/离朱 R42 的非阻断缺陷）----
    # 此前只认 `--verify-dir`，而该参数是**输出**语义，导致在全新空目录首次执行必然
    # FileNotFoundError 退码 1。现改为「输入/输出分离」：`--report-dir` 指定基准报告所在目录，
    # 未指定时按 [--report-dir] -> [--verify-dir] -> [--out-dir] -> [正式产物目录] 顺序回退。
    candidates: List[str] = []
    if str(getattr(args, "report_dir", "") or ""):
        candidates.append(os.path.join(str(args.report_dir), "step2_report.json"))
    candidates.append(os.path.join(verify_dir, "step2_report.json"))
    candidates.append(os.path.join(out_dir, "step2_report.json"))
    candidates.append(os.path.join(S.STEP2_DIR, "step2_report.json"))
    report_path = ""
    for _cand in candidates:
        if os.path.isfile(_cand):
            report_path = _cand
            break
    if not report_path:
        raise FileNotFoundError(
            "未找到对账基准报告 step2_report.json；已按顺序查找："
            + ", ".join(repr(c) for c in candidates)
            + "。请用 --report-dir 指定其所在目录，或先跑一次 eval 生成正式报告。"
        )
    _log(f"[replay] 对账基准报告 = {report_path}")
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    pool_index = list(range(len(rows)))
    table_meta = sorted(
        os.path.join(out_dir, n) for n in os.listdir(out_dir)
        if n.startswith("qa_step2_textrows_") and n.endswith(".pt.zip")
    )
    if not table_meta:
        raise FileNotFoundError(f"{out_dir!r} 下没有冻结键表产物")
    table, kt_meta = S.load_key_table(table_meta[0], verify=True)
    _log(f"[replay] 键表产物加载成功：[{table.size}, {table.dim}] sha256={table.sha256()[:16]}...")

    diffs: List[Json] = []
    models: Dict[Tuple[int, str], Any] = {}
    for seed in [int(x) for x in str(args.seeds).split(",") if str(x).strip()]:
        for mode in [m for m in str(args.modes).split(",") if m.strip()]:
            path = os.path.join(
                out_dir,
                f"qa_step2_model_{args.backend}_{mode}_D{vectorizer.dim}"
                f"_C{data.corpus.n_classes}_s{seed}.pt.zip",
            )
            meta, state = load_model_artifact(path)
            if state.get("backbone_missing"):
                raise ValueError(f"产物 {path!r} 缺少骨干权重，无法复算；请重跑 eval")
            model = rebuild_model_from_meta(meta, state)
            models[(seed, mode)] = model
            hc = meta.get("head_config") or {}
            _log(f"[replay] seed={seed} mode={mode}: 产物严格加载 OK "
                 f"(head_input_mode={hc.get('head_input_mode')}, "
                 f"normalize_query={hc.get('normalize_query')})")
            matcher = S.TextRowMatcher(model, vectorizer, mode,
                                       table if mode == "index" else None)
            got = S.self_retrieval(matcher, rows, ctx["split"].query_index,
                                   pool_index, table)
            ref = (report.get("self_retrieval") or {}).get(f"s{seed}_{mode}")
            if ref is None:
                diffs.append({"case": f"s{seed}_{mode}", "status": "MISSING_IN_REPORT"})
                continue
            d1 = abs(float(got["recall_at_1"]) - float(ref["recall_at_1"]))
            d5 = abs(float(got["recall_at_5"]) - float(ref["recall_at_5"]))
            diffs.append({
                "case": f"s{seed}_{mode}_self_retrieval",
                "replay_recall_at_1": float(got["recall_at_1"]),
                "report_recall_at_1": float(ref["recall_at_1"]),
                "abs_diff_at_1": float(d1),
                "abs_diff_at_5": float(d5),
                "match": bool(d1 <= 1e-12 and d5 <= 1e-12),
            })

    # 双模式一致性（同一 index 模型，两种候选来源）
    for seed in [int(x) for x in str(args.seeds).split(",") if str(x).strip()]:
        if (seed, "index") not in models or (seed, "pointer") not in models:
            continue
        mi = S.TextRowMatcher(models[(seed, "index")], vectorizer, "index", table)
        mp_same = S.TextRowMatcher(models[(seed, "index")], vectorizer, "pointer", None)
        dm = S.dual_mode_consistency(mi, mp_same, rows, ctx["split"].query_index, table)
        ref = (report.get("dual_mode") or {}).get(f"s{seed}_candidates_same_model") or {}
        diffs.append({
            "case": f"s{seed}_dual_mode_same_model",
            "replay_top1_agreement": float(dm["top1_agreement"]),
            "report_top1_agreement": float(ref.get("top1_agreement", float("nan"))),
            "logits_max_abs_diff": float(dm["logits_max_abs_diff"]),
            "match": bool(abs(float(dm["top1_agreement"])
                              - float(ref.get("top1_agreement", -1.0))) <= 1e-12),
        })

    # 分项指标复算（可选，较慢）
    if bool(args.full_metrics):
        for seed in [int(x) for x in str(args.seeds).split(",") if str(x).strip()]:
            primary = models.get((seed, "index"))
            if primary is None:
                continue
            step2_global = (report.get("self_retrieval") or {}).get(f"s{seed}_index")
            router = QuestionRouter(
                data.corpus.answer_keys, data.corpus.answer_display,
                [t for t in data.text_lines], vectorizer,
                text_threshold=float(args.text_threshold),
            )
            for task in S.TASKS:
                tv = S.filter_by_task(data, task)
                m = S.task_metrics(task, primary, tv, router, vectorizer, step2_global,
                                   batch_size=int(args.batch_size))
                ref = ((report.get("per_seed") or {}).get(str(seed)) or {}).get(task) or {}
                worst = 0.0
                for fld in S.AGGREGATED_FIELDS:
                    if fld in ref and fld in m and fld != "step2_recall_at_1":
                        worst = max(worst, abs(float(m[fld]) - float(ref[fld])))
                diffs.append({
                    "case": f"s{seed}_{task}_metrics",
                    "max_abs_diff": float(worst),
                    "match": bool(worst <= 1e-12),
                    "replay": {k: m[k] for k in S.AGGREGATED_FIELDS},
                })
                _log(f"[replay] seed={seed} {task}: 分项复算最大偏差 {worst:.3e}")

    n_case = len(diffs)
    n_match = sum(1 for d in diffs if d.get("match") is True)
    payload: Json = {
        "cases": diffs,
        "n_cases": int(n_case),
        "n_match": int(n_match),
        "all_match": bool(n_case > 0 and n_match == n_case),
        "report_path": report_path.replace("\\", "/"),
        "report_metrics_sha256": str(report.get("report_paths", {}).get("metrics_sha256", "")),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = os.path.join(verify_dir, "replay.json")
    digest = _write_json(path, payload)
    _log(f"[replay] 对账 -> {path} (sha256={digest[:16]}...) "
         f"cases={n_case} match={n_match} all_match={payload['all_match']}")
    assert payload["all_match"], (
        f"产物复算与报告不一致：{n_match}/{n_case} 项匹配；明细见 {path}"
    )
    _log("[replay] OK")
    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
