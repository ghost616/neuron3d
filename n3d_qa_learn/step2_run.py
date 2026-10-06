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
from . import robust_eval as R
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


def _robust_dir(args: argparse.Namespace) -> str:
    """robust 子命令的报告目录（**验证类运行一律写 ``_verify/robust/``**）。"""
    for attr in ("verify_dir", "out_dir"):
        value = str(getattr(args, attr, "") or "")
        if value:
            return value
    return str(R.ROBUST_DIR)


def _csv_names(raw: str, allowed: Sequence[str], label: str) -> Tuple[str, ...]:
    """解析逗号分隔的名字列表并**在 CLI 边界做可读校验**。

    存在理由（离朱 R49 条目 33）：`--profiles bogus` / `--perturb bogus` 原先由
    ``RobustConfig.__post_init__`` 抛出的 ``KeyError`` / ``ValueError`` **穿透到解释器**，
    用户可见的是 30 行裸 Traceback。底层校验本身是对的（报文已带可用值清单），缺的是
    CLI 边界的捕获与可读呈现 —— 本函数把它前移到参数解析层，与 ``--side`` 的
    ``choices=`` 行为一致（退码 2、单行可读报文、无 Traceback）。

    参数
    ----
    raw : str
        逗号分隔的原始串（空串 = 用 ``allowed`` 全量）。
    allowed : Sequence[str]
        允许的名字集合（顺序敏感，空串回退时原样使用）。
    label : str
        参数名（仅用于报文）。

    返回
    ----
    Tuple[str, ...]
        解析后的名字元组。

    异常
    ------
    ValueError
        存在未登记的名字（报文含可用值清单）。
    """
    items = tuple(x.strip() for x in str(raw or "").split(",") if x.strip())
    if not items:
        return tuple(str(x) for x in allowed)
    unknown = [x for x in items if x not in set(str(a) for a in allowed)]
    if unknown:
        raise ValueError(
            f"{label} 含未登记的值 {unknown}；可用 = {sorted(str(a) for a in allowed)}"
        )
    return items


def _robust_config(args: argparse.Namespace, *, dry_run: bool = False) -> R.RobustConfig:
    """由 CLI 参数装配 robust 的冻结配置（不传新参数时等于默认档）。"""
    side = str(getattr(args, "side", "both") or "both")
    sides: Tuple[str, ...] = (
        tuple(R.SIDES) if side == "both" else (side,)
    )
    profiles = _csv_names(
        str(getattr(args, "profiles", "") or ""), R.ET.profile_names(), "--profiles"
    )
    kinds = _csv_names(
        str(getattr(args, "perturb", "") or ""), R.PERTURB_TYPES, "--perturb"
    )
    if dry_run:
        # 单组合演练：只跑「词面档 × 文本侧 × noise 三档」，把全量放到演练通过之后
        profiles = (R.ET.PROFILE_LEXICAL,)
        sides = ("text",)
        kinds = ("noise",)
    return R.RobustConfig(
        seed=int(getattr(args, "seed", R.ROBUST_SEED)),
        sides=sides,
        profiles=profiles,
        perturb_types=kinds,
        variants=int(getattr(args, "variants", R.ROBUST_VARIANTS)),
        stability_k=int(getattr(args, "stability_k", R.ROBUST_STABILITY_K)),
        topk=int(getattr(args, "topk", R.TOPK)),
        batch_size=int(getattr(args, "batch_size", R.BATCH_SIZE)),
        out_dir=_robust_dir(args),
    )


def _robust_run_json(args: argparse.Namespace) -> str:
    """``calibrate`` / ``report`` 要读的 run 产物路径。"""
    explicit = str(getattr(args, "run_json", "") or "")
    return explicit if explicit else os.path.join(_robust_dir(args), "robust_run.json")


def _robust_calibration_json(args: argparse.Namespace) -> str:
    """``report`` 要读的标定产物路径。"""
    explicit = str(getattr(args, "calibration_json", "") or "")
    return explicit if explicit else os.path.join(_robust_dir(args), "robust_calibration.json")


def cmd_robust_probe(args: argparse.Namespace) -> int:
    """``robust probe``：表构造与口径取证（**不评测**）。"""
    cfg = _robust_config(args)
    out_dir = cfg.resolved_out_dir()
    _log(f"[robust probe] 输出目录 = {out_dir}；种子 = {cfg.seed}；"
         f"编码器档 = {list(cfg.profiles)}；侧 = {list(cfg.sides)}")
    payload = R.summarize_probe(cfg, log=_log)
    for key, meta in sorted(payload["entry_tables"].items()):
        norm = meta["norm_range"]
        bl = meta["bitwise_lookup"]
        _log(f"[robust probe] {key}: keys={meta['n_entries']}×{meta['dim']} / "
             f"范数 [{norm['min']:.8f}, {norm['max']:.8f}] / "
             f"键表 SHA256 = {str(meta['key_table_sha256'])[:16]}... / "
             f"编码器指纹 = {str(meta['encoder_fingerprint'])[:16]}... / "
             f"逐位精确查表 {bl['n_exact']}/{bl['n_checked']}（exact_frac={bl['exact_frac']}）")
    for err in payload["probe_errors"]:
        _log(f"[robust probe] 失败（如实登记）：{err}")
    for row in payload["perturb_self_check"]:
        _log(f"[robust probe] 扰动自检 {row['kind']}/{row['level']}: "
             f"ε={row.get('eps')} formula={row.get('formula')!r} "
             f"shape={row['shape']} finite={row['finite']} "
             f"零范数={row['n_zero_norm']} 位移={row['mean_row_displacement']:.6f} "
             f"seed={row['generator_seed']}（随机数={row.get('uses_generator')}）")
    g7 = payload.get("bitwise_formula_assertions", {})
    if g7:
        _log(f"[robust probe] G7 扰动公式逐位断言：{g7.get('n_bitwise_equal')}/"
             f"{g7.get('n_cases')} 例逐字节相等，全通过 = {g7.get('all_bitwise_equal')}")
        for case in g7.get("cases", []):
            _log(f"[robust probe]   G7 {case['kind']}/{case['level']}: "
                 f"未归一化={case['raw_bitwise_equal']} 归一化后={case['bitwise_equal']} "
                 f"（{case['recompute']}）")
    path = os.path.join(out_dir, "robust_probe.json")
    digest = R.write_json(path, payload)
    _log(f"[robust probe] 取证报告 -> {path} (sha256={digest[:16]}...)")
    if payload["probe_errors"]:
        _log("[robust probe] 存在失败项（退码 1，不以成功状态落账）")
        return 1
    if g7 and not bool(g7.get("all_bitwise_equal")):
        _log("[robust probe] G7 扰动公式逐位断言未通过（退码 1，不以成功状态落账）")
        return 1
    _log("[robust probe] OK")
    return 0


def cmd_robust_run(args: argparse.Namespace) -> int:
    """``robust run``：分档全量评测（三档 × 三扰动 × 两侧 × 两编码器档）。"""
    dry_run = bool(getattr(args, "dry_run", False))
    cfg = _robust_config(args, dry_run=dry_run)
    out_dir = cfg.resolved_out_dir()
    _log(f"[robust run] 输出目录 = {out_dir}；种子 = {cfg.seed}；"
         f"编码器档 = {list(cfg.profiles)}；侧 = {list(cfg.sides)}；"
         f"扰动 = {list(cfg.perturb_types)}；变体 = {cfg.variants}；K = {cfg.stability_k}")
    if dry_run:
        _log("[robust run] **演练口径**：单组合（词面档 × 文本侧 × noise 三档）")
    result = R.run_evaluation(cfg, log=_log)
    name = "robust_run_dry.json" if dry_run else "robust_run.json"
    path = os.path.join(out_dir, name)
    digest = R.write_json(path, result)
    inv = result["invariants"]
    _log(f"[robust run] 不变量：{inv['n_checks']} 条 / 失败 {inv['n_failed']} 条 / "
         f"全通过 = {inv['all_passed']}")
    g7 = result.get("bitwise_formula_assertions", {})
    _log(f"[robust run] G7 扰动公式逐位断言：{g7.get('n_bitwise_equal')}/"
         f"{g7.get('n_cases')} 例逐字节相等，全通过 = {g7.get('all_bitwise_equal')}")
    # [!] 耗时只进日志：产物必须逐字节可复现（见 robust_eval.run_evaluation 的纪律注释）
    _log(f"[robust run] 产物 -> {path} (sha256={digest[:16]}...)")
    if int(inv["n_failed"]) > 0:
        _log("[robust run] 存在失败的不变量检查（退码 1，不以成功状态落账）")
        return 1
    _log("[robust run] OK")
    return 0


def _validate_robust_selector_args(args: argparse.Namespace) -> None:
    """校验 ``robust`` 的选择类参数（``--side`` / ``--profiles`` / ``--perturb``）。

    存在理由（审查 W4）：``probe`` / ``run`` 经 ``_robust_config`` 会校验 ``--profiles`` /
    ``--perturb``；而 ``calibrate`` / ``report`` **不构造** ``RobustConfig``，此前会把这些
    参数**静默忽略**（`robust calibrate --profiles bogus` 退码 0），与 README 里
    「非法值给可读报文 + 退码 2」的表述不符。本函数把这层校验显式前移到四个子命令的入口，
    使「不适用」变成**显式校验**而不是静默忽略。

    异常
    ------
    ValueError
        存在未登记取值（由 :func:`_csv_names` 抛出，报文含可用值清单）。
    """
    _csv_names(
        str(getattr(args, "profiles", "") or ""), R.ET.profile_names(), "--profiles"
    )
    _csv_names(
        str(getattr(args, "perturb", "") or ""), R.PERTURB_TYPES, "--perturb"
    )
    side = str(getattr(args, "side", "both") or "both")
    if side not in ("qa", "text", "both"):
        raise ValueError(f"--side 含未登记的值 {side!r}；可用 = ['qa', 'text', 'both']")


def cmd_robust_calibrate(args: argparse.Namespace) -> int:
    """``robust calibrate``：阈值现场标定（读 run 结果，产出标定阈值与依据）。

    **参数校验**（审查 W4）：本子命令虽然不消费 ``--profiles`` / ``--perturb``，但
    为了与 probe/run 行为一致（不给用户「输错了却静默通过」的错觉），仍在此显式校验。
    """
    t0 = time.time()
    _validate_robust_selector_args(args)
    run_path = _robust_run_json(args)
    out_dir = _robust_dir(args)
    if not os.path.isfile(run_path):
        _log(
            f"[robust calibrate] 缺少 run 产物：{run_path!r}；"
            "请先执行 `robust run`（或显式给 --run-json 指向实际产物）"
        )
        return 1
    _log(f"[robust calibrate] 读 run 产物 = {run_path}")
    run_result = R.load_run(run_path)
    calibration = R.calibrate_thresholds(
        run_result,
        factor=float(getattr(args, "factor", R.THRESHOLD_FACTOR)),
        primary_side=str(getattr(args, "primary_side", "text") or "text"),
    )
    _log(f"[robust calibrate] 噪声底上界 = {calibration['noise_floor_max']:.6f}；"
         f"标定阈值 τ = {calibration['tau_calibrated']:.6f}（factor = {calibration['factor']}）")
    for e in calibration["entries"]:
        _log(f"[robust calibrate]   {e['side']}/{e['profile']}/{e['kind']}: "
             f"档0={e['clean_hit_rate']:.6f} 弱={e['level_hit_rate'].get('weak', 0.0):.6f} "
             f"强={e['level_hit_rate'].get('strong', 0.0):.6f} 落差={e['gap_weak_to_strong']:.6f} "
             f"噪声底={e['noise_floor']:.6f} τ={e['tau']:.6f} -> "
             f"{'有效' if e['effective'] else '无效'}")
    # 把标定阈值登记进未识别档的**参照**字段（**不重跑评测、不改特征口径**）
    tau_text = calibration["tau_calibrated"]
    acknowledged: List[Json] = [
        {
            "side": u.get("side"),
            "profile": u.get("profile"),
            "tau_reference": float(tau_text),
            "note": u["report"].get("rate_note", ""),
        }
        for u in run_result.get("unrecognized", [])
    ]
    calibration["unrecognized_tau_reference"] = float(tau_text)
    calibration["unrecognized_note"] = (
        "未识别档的 τ 口径：本批**不标定**该档阈值（属第二步变体 B）；报告里已给"
        "不依赖 τ 的分位数。此处记录 calibrate 现场得到的 τ 仅作参照，不作为判定依据。"
    )
    calibration["unrecognized_acknowledged"] = acknowledged
    path = os.path.join(out_dir, "robust_calibration.json")
    digest = R.write_json(path, calibration)
    _log(f"[robust calibrate] 标定产物 -> {path} (sha256={digest[:16]}...)")
    _log(f"[robust calibrate] 耗时 {time.time() - t0:.1f}s（只进日志，不入产物）")
    _log("[robust calibrate] OK")
    return 0


def cmd_robust_report(args: argparse.Namespace) -> int:
    """``robust report``：报告渲染（JSON + Markdown）。

    **参数校验**（审查 W4）：同 ``calibrate``，对本子命令不适用的 ``--profiles`` /
    ``--perturb`` 仍显式校验并给出可读报文（退码 2），不静默忽略。
    """
    _validate_robust_selector_args(args)
    run_path = _robust_run_json(args)
    out_dir = _robust_dir(args)
    if not os.path.isfile(run_path):
        _log(
            f"[robust report] 缺少 run 产物：{run_path!r}；"
            "请先执行 `robust run`（或显式给 --run-json 指向实际产物）"
        )
        return 1
    calibration: Optional[Json] = None
    cal_path = _robust_calibration_json(args)
    _log(f"[robust report] 读 run 产物 = {run_path}")
    if os.path.isfile(cal_path):
        _log(f"[robust report] 读标定产物 = {cal_path}")
        calibration = R.load_run(cal_path)
    else:
        _log(f"[robust report] 标定产物不存在（{cal_path}）-> 报告「阈值现场标定」一节标「尚未标定」")
    run_result = R.load_run(run_path)
    md = R.render_markdown(run_result, calibration)
    md_path = os.path.join(out_dir, "robust_report.md")
    digest = R.write_text(md_path, md)
    json_path = os.path.join(out_dir, "robust_report.json")
    # [!] 产物**确定性纪律**（G3 / 审查 W3）：不再写 created_utc / seconds_run —— 它们会让
    # 同命令重复运行的 report 产物字节不同，并使内嵌的 *_artifact_sha256 重跑即失效。
    # 生成时间与耗时只进运行日志。此处内嵌的两个 SHA256 全部来自**确定性产物**
    # （run / calibration 本身已不含非确定字段），故重跑后仍然闭环。
    report_json = {
        "artifact_schema": "robust-report-v1",
        "deterministic": True,
        "excluded_fields_note": (
            "本产物不含 created_utc / seconds / 绝对路径等非确定字段；生成时间与耗时、"
            "输入产物的**绝对路径**只写运行日志（此处只留 basename）。内嵌的 "
            "run_artifact_sha256 / calibration_artifact_sha256 均来自确定性产物，"
            "故同命令重复运行（含换输出目录）可逐字节复现"
        ),
        ### 只留 basename：绝对路径依赖 cwd/输出目录，会把「同命令重跑」的产物字节带偏
        "run_artifact_basename": os.path.basename(run_path),
        "run_artifact_sha256": S.sha256_file(run_path),
        "calibration_artifact_basename": (
            os.path.basename(cal_path) if calibration is not None else ""
        ),
        "calibration_artifact_sha256": S.sha256_file(cal_path) if calibration is not None else "",
        "grid": run_result.get("grid", {}),
        "tiers": run_result.get("tiers", {}),
        "entry_tables": run_result.get("entry_tables", {}),
        "euclidean_axis": run_result.get("euclidean_axis", {}),
        "cells": run_result.get("cells", []),
        "stability": run_result.get("stability", []),
        "unrecognized": run_result.get("unrecognized", []),
        "invariants": run_result.get("invariants", {}),
        "bitwise_formula_assertions": run_result.get("bitwise_formula_assertions", {}),
        "calibration": calibration or {},
        "honest_notes": run_result.get("honest_notes", []),
        "markdown_basename": os.path.basename(md_path),
        "markdown_sha256": digest,
    }
    json_digest = R.write_json(json_path, report_json)
    _log(f"[robust report] 输入：run = {run_path}；calibration = {cal_path if calibration else '（缺）'}")
    _log(f"[robust report] Markdown -> {md_path} (sha256={digest[:16]}...)")
    _log(f"[robust report] JSON     -> {json_path} (sha256={json_digest[:16]}...)")
    _log("[robust report] 注：绝对路径只进日志，产物内只留 basename（保证换目录重跑逐字节一致）")
    _log("[robust report] OK")
    return 0


def cmd_robust(args: argparse.Namespace) -> int:
    """``robust`` 子命令分派：probe / run / calibrate / report。

    **CLI 错误边界**（离朱 R49 条目 33）：非法的 ``--profiles`` / ``--perturb`` /
    ``--side`` 等配置项在此被转换为**单行可读报文 + 退码 2**，不再抛裸 Traceback；
    捕获范围**只包住 robust 子命令族**，既有子命令的分发逻辑逐字符未改。
    """
    sub = str(getattr(args, "robust_cmd", "") or "run")
    handlers = {
        "probe": cmd_robust_probe,
        "run": cmd_robust_run,
        "calibrate": cmd_robust_calibrate,
        "report": cmd_robust_report,
    }
    if sub not in handlers:
        _log(f"[robust] 未知子命令 {sub!r}；可用 = {sorted(handlers)}")
        return 2
    try:
        return int(handlers[sub](args))
    except (ValueError, KeyError) as exc:
        _log(f"[robust] 非法参数：{exc}")
        return 2


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

    p_rb = sub.add_parser(
        "robust",
        help="分档鲁棒性考卷（第一步 1a）：条目特征表 / 逐位精确查表 / 分档评测 / KNN 基线",
        description=(
            "分档鲁棒性考卷（不改网络结构、不做训练）：probe（表构造与口径取证）/ "
            "run（三档 × 三扰动 × 两侧 × 两编码器档全量评测）/ calibrate（阈值现场标定）/ "
            "report（JSON + Markdown 渲染）。报告写 checkpoints/qa_learn/_verify/robust/，"
            "不落盘扰动后的特征矩阵。"
        ),
    )
    common(p_rb)
    p_rb.add_argument("--side", default="both", choices=["qa", "text", "both"],
                      help="评测侧（both = 两侧都跑）")
    p_rb.add_argument("--profiles", default=",".join(R.ROBUST_PROFILES),
                      help="编码器档列表（逗号分隔；见 entry_table.profile_names()）")
    p_rb.add_argument("--perturb", default=",".join(R.PERTURB_TYPES),
                      help="扰动类型列表（逗号分隔；noise / mask / nmag）")
    p_rb.add_argument("--seed", type=int, default=R.ROBUST_SEED,
                      help="扰动种子（局部 torch.Generator，不消耗全局 RNG）")
    p_rb.add_argument("--variants", type=int, default=R.ROBUST_VARIANTS,
                      help="每档每条目的变体数（主判据 = 1）")
    p_rb.add_argument("--stability-k", type=int, default=R.ROBUST_STABILITY_K,
                      help="稳定性佐证变体数 K（均值 ± 极差）")
    p_rb.add_argument("--topk", type=int, default=R.TOPK, help="归一化余弦 top-k 的 k")
    p_rb.add_argument("--batch-size", type=int, default=R.BATCH_SIZE, help="检索批大小")
    p_rb.add_argument("--verify-dir", default="", help="报告目录（与 --out-dir 同义）")
    p_rb.add_argument("--run-json", default="", help="run 产物路径（calibrate/report 读它）")
    p_rb.add_argument("--calibration-json", default="", help="标定产物路径（report 读它）")
    p_rb.add_argument("--factor", type=float, default=R.THRESHOLD_FACTOR,
                      help="阈值标定倍数（观测噪声的约 2 倍）")
    p_rb.add_argument("--primary-side", default="text", choices=["qa", "text"],
                      help="主判据侧（默认 text；QA 侧只作辅助）")
    p_rb.add_argument("--dry-run", action="store_true",
                      help="run 的演练口径：只跑「单个组合」（local-hash × 文本侧 × noise 三档）")
    p_rb.add_argument("robust_cmd", nargs="?", default="run",
                      choices=["probe", "run", "calibrate", "report"],
                      help="robust 子命令（缺省 run）")
    p_rb.set_defaults(func=cmd_robust)
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
