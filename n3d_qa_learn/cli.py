"""n3d_qa_learn 的纯 CLI 入口（``python -m n3d_qa_learn.cli <子命令>``）。

子命令
------
===========  ==========================================================================
``probe``    P0 探针（硬门禁）：三后端可用性登记
``drill``    单条端到端演练（1 batch 前向 + 反向 + 一步更新，断言梯度非零）
``train``    全量训练 + 落盘产物 + 步骤 1 评估报告
``ask``      单次问答：``--question "..."`` -> 答案 / 匹配行 / 无匹配（含来源与分数）
``eval``     对既有产物做步骤 1 评估与「不相关」评估
``guard``    加载守卫的拒绝证明（篡改答案表 / 口径指纹后必须报错）
``selftest`` 边界处置自检（空问题 / 仅空白 / 超长 / 空候选 / 超长文本行）
===========  ==========================================================================

退出码
------
``0`` 成功；``1`` 业务失败（门禁未通过 / 守卫未拒绝 / 自检失败）；``2`` 参数错误（argparse）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

# 控制台编码加固：源码与运行日志含中文，GBK 控制台下必须显式重配（否则可能 UnicodeEncodeError）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - 老环境无 reconfigure 时忽略
        pass

from .backends import BACKEND_NAMES, build_registry
from .data import load_text_lines
from .evaluate import (
    acceptance_check,
    boundary_selftest,
    evaluate_refusal,
    evaluate_step1,
    guard_rejection_proof,
)
from .features import VectorizerConfig
from .probe import end_to_end_drill, probe_backends
from .route import NO_MATCH_TEXT, QuestionRouter
from .train import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_VERIFY_DIR,
    TrainConfig,
    artifact_name,
    build_training_data,
    load_artifact,
    rebuild_model,
    run_training,
)


def _dump(obj: Any) -> None:
    """以 UTF-8 安全的 JSON 打印（``ensure_ascii=False``，中文可读）。"""
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    """P0 探针：登记三后端可用性。"""
    cfg = VectorizerConfig(hash_dim=int(args.hash_dim))
    result = probe_backends(int(args.dim) if args.dim > 0 else cfg.dim)
    _dump(result)
    if result["all_failed"]:
        print("[FAIL] 三个后端全部构造失败，P0 门禁未通过", file=sys.stderr)
        return 1
    print(
        f"[OK] P0 探针通过：{result['n_available']}/{result['n_requested']} 后端可用，"
        f"不可用 = {sorted(result['unavailable'].keys())}"
    )
    return 0


def cmd_drill(args: argparse.Namespace) -> int:
    """单条端到端演练。"""
    cfg = VectorizerConfig(hash_dim=int(args.hash_dim))
    try:
        res = end_to_end_drill(
            dim=int(args.dim) if args.dim > 0 else cfg.dim,
            backend=str(args.backend),
            output_mode=str(args.output_mode),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
        )
    except Exception as exc:  # noqa: BLE001 - 演练失败必须退码 1 并给出原因
        print(f"[FAIL] 端到端演练失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _dump(res.as_dict())
    print(
        f"[OK] 端到端演练通过：loss {res.loss_before:.4f} -> {res.loss_after:.4f}，"
        f"{res.n_param_changed}/{len(res.param_names)} 个参数被更新，"
        f"最大参数变化 {res.max_param_delta:.3e}"
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """全量训练 + 评估 + 落盘。"""
    cfg = TrainConfig(
        backend=str(args.backend),
        output_mode=str(args.output_mode),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        split_seed=int(args.split_seed),
        max_classes=int(args.max_classes),
        min_questions=int(args.min_questions),
        test_every=int(args.test_every),
        test_per_class=int(args.test_per_class),
        unknown_class_weight=float(args.unknown_class_weight),
        unknown_train_cap=int(args.unknown_train_cap),
        label_smoothing=float(args.label_smoothing),
        max_train_samples=int(args.max_train_samples),
        text_threshold=float(args.text_threshold),
        qa_cache_dir=str(args.qa_cache_dir),
        text_dir=str(args.text_dir),
        train_backbone=bool(args.train_backbone),
        backbone_lr=float(args.backbone_lr),
        head_input_mode=str(args.head_input_mode),
        logit_scale_init=float(args.logit_scale_init),
        answer_table_mode=str(args.answer_table_mode),
        train_head=bool(args.train_head),
        learn_logit_scale=not bool(args.no_learn_logit_scale),
    )
    out_dir = DEFAULT_VERIFY_DIR if args.verify else DEFAULT_ARTIFACT_DIR
    artifact = str(args.artifact) if args.artifact else os.path.join(
        out_dir, artifact_name(cfg, VectorizerConfig(hash_dim=int(args.hash_dim)).dim)
    )
    t0 = time.time()
    result = run_training(cfg, max_batches=int(args.max_batches), artifact_path=artifact)
    metrics = evaluate_step1(
        result.model, result.data, k=int(args.topk), device=result.device
    )
    refusal = evaluate_refusal(result.model, result.data, device=result.device)
    verdict = acceptance_check(metrics, task=str(args.task))
    report: Dict[str, Any] = {
        "artifact": result.artifact_path,
        "artifact_sha256": result.artifact_bytes_sha256,
        "dim": int(result.data.vectorizer.dim),
        "backend": cfg.backend,
        "output_mode": cfg.output_mode,
        "split": result.data.splits.summary(),
        "epochs": cfg.epochs,
        "final_loss": result.final_loss(),
        "step1": metrics.as_dict(),
        "refusal": refusal,
        "acceptance": verdict,
        "seconds": time.time() - t0,
    }
    _dump(report)
    if not verdict["passed"]:
        print("[FAIL] 步骤 1 未达验收门槛", file=sys.stderr)
        return 1
    print(
        f"[OK] 训练完成：macro_acc={metrics.macro_acc:.4f} "
        f"(多数类基线 {metrics.majority:.4f}，随机基线 {metrics.random_baseline:.4f})，"
        f"产物 {result.artifact_path}"
    )
    report["report_path"] = _write_report(out_dir, report)
    print(f"[INFO] 报告已写入 {report['report_path']}")
    return 0


def _write_report(out_dir: str, report: Dict[str, Any]) -> str:
    """把评估报告写为 UTF-8 JSON（固定文件名按后端与模式区分）。"""
    os.makedirs(out_dir, exist_ok=True)
    name = f"report_{report['backend']}_{report['output_mode']}_s{report.get('seed', '')}.json"
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1, default=str)
    return path


def _load_and_route(artifact: str, question: str, text_dir: str, threshold: float) -> int:
    """载入产物、执行两级路由、打印结果（``ask`` 的核心）。"""
    bundle = load_artifact(artifact)
    meta = bundle["meta"]
    model = rebuild_model(meta, bundle["state_dict"])
    from .features import vectorizer_from_meta

    vectorizer = vectorizer_from_meta(meta)
    answer_keys = list(meta["answer_keys"])
    answer_display = dict(meta["answer_display"])
    lines = load_text_lines(text_dir)
    router = QuestionRouter(
        answer_keys, answer_display, lines, vectorizer, text_threshold=float(threshold)
    )
    import torch as _torch

    with _torch.no_grad():
        feats = _torch.tensor([vectorizer.encode(question)], dtype=_torch.float32)
        keys = None
        if model.config.output_mode == "pointer":
            from .train import _pointer_keys

            keys = _pointer_keys(
                answer_keys, answer_display, vectorizer, 1
            )
        logits = model.logits(feats, keys)[0].tolist()
    result = router.route(question, logits)
    _dump(
        {
            "question": question,
            "answer": result.answer,
            "source": result.source,
            "score": result.score,
            "step1_index": result.step1_index,
            "step1_is_irrelevant": result.step1_is_irrelevant,
            "step2_line_id": result.step2_line_id,
            "reason": result.reason,
            "output_mode": str(meta["output_mode"]),
            "dim": int(meta["dim"]),
        }
    )
    label = {
        "qa": "答案（步骤 1：QA 数据集命中）",
        "text": "匹配行（步骤 2：文本数据集命中）",
        "none": "无匹配（两级均未命中）",
    }[result.source]
    print(f"[{label}] {result.answer}  (score={result.score:.4f})")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """单次问答。"""
    artifact = str(args.model)
    if not os.path.isfile(artifact):
        print(f"[FAIL] 产物不存在：{artifact}", file=sys.stderr)
        return 1
    return _load_and_route(
        artifact, str(args.question), str(args.text_dir), float(args.text_threshold)
    )


def cmd_eval(args: argparse.Namespace) -> int:
    """对既有产物做步骤 1 评估与「不相关」评估。"""
    bundle = load_artifact(str(args.model))
    meta = bundle["meta"]
    model = rebuild_model(meta, bundle["state_dict"])
    cfg = TrainConfig(
        backend=str(meta["backend"]["backend"]),
        output_mode=str(meta["output_mode"]),
        max_classes=int(meta["n_answers"]),
        min_questions=int(meta["train_config"]["min_questions"]),
        test_every=int(meta["train_config"]["test_every"]),
        test_per_class=int(meta["train_config"].get("test_per_class", 0)),
        seed=int(meta["train_config"]["seed"]),
        # 切分种子的**向后兼容读取**：旧产物 meta 无该键 -> -1 -> 沿用 seed（历史行为）
        split_seed=int(meta["train_config"].get("split_seed", -1)),
        qa_cache_dir=str(meta["train_config"]["qa_cache_dir"]),
        text_dir=str(meta["train_config"]["text_dir"]),
    )
    data = build_training_data(cfg)
    dim_ok = int(meta["dim"]) == int(data.vectorizer.dim)
    metrics = evaluate_step1(model, data, k=int(args.topk))
    refusal = evaluate_refusal(model, data)
    verdict = acceptance_check(metrics, task=str(args.task))
    report = {
        "artifact": str(args.model),
        "dim": int(meta["dim"]),
        "dim_matches_current_vectorizer": bool(dim_ok),
        "step1": metrics.as_dict(),
        "refusal": refusal,
        "acceptance": verdict,
    }
    _dump(report)
    return 0 if verdict["passed"] else 1


def cmd_guard(args: argparse.Namespace) -> int:
    """加载守卫的拒绝证明。"""
    results = guard_rejection_proof(str(args.model))
    _dump({"artifact": str(args.model), "cases": results})
    injected = [r for r in results if r.get("injected")]
    rejected = [r for r in injected if r["raised"]]
    if not injected:
        print("[FAIL] 没有任何注入实验被执行", file=sys.stderr)
        return 1
    if len(rejected) != len(injected):
        print(
            f"[FAIL] 守卫拒绝证明不完整：{len(rejected)}/{len(injected)} 个注入被拒绝",
            file=sys.stderr,
        )
        return 1
    print(f"[OK] 守卫拒绝证明通过：{len(rejected)}/{len(injected)} 个注入全部被拒绝")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """边界处置自检。"""
    bundle = load_artifact(str(args.model))
    meta = bundle["meta"]
    from .features import vectorizer_from_meta

    vectorizer = vectorizer_from_meta(meta)
    lines = load_text_lines(str(args.text_dir))
    cases = boundary_selftest(
        vectorizer,
        lines,
        list(meta["answer_keys"]),
        dict(meta["answer_display"]),
        int(meta["n_answers"]),
    )
    _dump({"cases": cases})
    failed = [c for c in cases if not c["passed"]]
    if failed:
        print(f"[FAIL] 边界处置自检失败 {len(failed)} 项：{[c['case'] for c in failed]}",
              file=sys.stderr)
        return 1
    print(f"[OK] 边界处置自检通过：{len(cases)}/{len(cases)}")
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 解析器（子命令 + 全局开关）。"""
    parser = argparse.ArgumentParser(
        prog="n3d_qa_learn",
        description="N3D 问答学习框架：两级业务路由（QA 数据集 -> 文本数据集 -> 无匹配）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--hash-dim", type=int, default=80,
                       help="哈希词袋桶数（默认 80；D = hash_dim + 8）")
        p.add_argument("--dim", type=int, default=0,
                       help="特征维 D（0 = 由向量化口径决定；显式给出必须与口径一致）")

    p = sub.add_parser("probe", help="P0 探针：三后端可用性登记")
    add_common(p)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("drill", help="单条端到端演练（梯度非零门禁）")
    add_common(p)
    p.add_argument("--backend", type=str, default="n3d_shape", choices=list(BACKEND_NAMES))
    p.add_argument("--output-mode", type=str, default="index",
                   choices=["index", "pointer"], help="全局输出模式开关")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_drill)

    p = sub.add_parser("train", help="训练 + 评估 + 落盘")
    add_common(p)
    p.add_argument("--backend", type=str, default="n3d_shape", choices=list(BACKEND_NAMES))
    p.add_argument("--output-mode", type=str, default="index",
                   choices=["index", "pointer"], help="全局输出模式开关（全局一个，不做级联）")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=-1,
                   help="切分种子（-1 = 沿用 --seed，即历史行为）；"
                        "置为 >= 0 时切分只由它驱动，训练仍由 --seed 驱动")
    p.add_argument("--max-classes", type=int, default=8)
    p.add_argument("--min-questions", type=int, default=8)
    p.add_argument("--test-every", type=int, default=3)
    p.add_argument("--test-per-class", type=int, default=2,
                   help="主测试集逐类样本上限（0 = 不限制；用于让基线口径可比）")
    p.add_argument("--unknown-class-weight", type=float, default=1.0)
    p.add_argument("--unknown-train-cap", type=int, default=500)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=0, help="每 epoch 的 batch 上限（验证类运行）")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--text-threshold", type=float, default=0.28)
    p.add_argument("--task", type=str, default="triviaqa", choices=["triviaqa", "generic"])
    p.add_argument("--qa-cache-dir", type=str, default="checkpoints/triviaqa/_cache")
    p.add_argument("--text-dir", type=str, default="data/doc")
    p.add_argument("--train-backbone", action="store_true",
                   help="冻结后端 N3D 权重，只训练 q 头与候选键表（对照实验用）")
    p.add_argument("--backbone-lr", type=float, default=5e-4)
    p.add_argument("--head-input-mode", type=str, default="raw",
                   choices=["raw", "concat", "n3d"], help="q 头的连接口径")
    p.add_argument("--logit-scale-init", type=float, default=20.0)
    p.add_argument("--train-head", action="store_true",
                   help="训练 q 头（默认冻结；打开会退化为一律拒绝，见 TrainConfig.train_head）")
    p.add_argument("--no-learn-logit-scale", action="store_true",
                   help="固定 logit 尺度（默认让它学习；它不影响 argmax）")
    p.add_argument("--answer-table-mode", type=str, default="centroid",
                   choices=["centroid", "free"])
    p.add_argument("--artifact", type=str, default="", help="产物路径（空 = 自动命名）")
    p.add_argument("--verify", action="store_true", help="写入 _verify 子目录")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("ask", help="单次问答")
    p.add_argument("--question", type=str, required=True)
    p.add_argument("--model", type=str, required=True, help="模型产物路径")
    p.add_argument("--text-dir", type=str, default="data/doc")
    p.add_argument("--text-threshold", type=float, default=0.28)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="对既有产物做评估")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--task", type=str, default="triviaqa", choices=["triviaqa", "generic"])
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("guard", help="加载守卫拒绝证明")
    p.add_argument("--model", type=str, required=True)
    p.set_defaults(func=cmd_guard)

    p = sub.add_parser("selftest", help="边界处置自检")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--text-dir", type=str, default="data/doc")
    p.set_defaults(func=cmd_selftest)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口（返回进程退出码）。"""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())