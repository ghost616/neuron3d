"""n3d_qa_learn 表示训练对照实验的 CLI 入口。

用法
----
* ``python -m n3d_qa_learn.exp_repr_run drill``
  单条端到端演练：基线组 **1 个 epoch**，断言「可训参数更新量 > 0」，
  并与 ``train.run_training``（``save=False``）**逐项对账**，证明本实验的训练循环
  与模块自带循环同构（等价性检查，不是"看起来跑了"）。
* ``python -m n3d_qa_learn.exp_repr_run run``
  放全量对照矩阵（8 组），执行切分一致性断言，写报告到
  ``checkpoints/qa_learn/_verify/exp_repr/``。
* ``python -m n3d_qa_learn.exp_repr_run compare``
  **同切分**下逐组对照两个**特征档**（词面 ``D=88`` vs ``bge-m3`` ``D=1024``）：
  跨档切分一致性断言（G1）+ 逐组 macro / 步骤 2 自检索 ``Recall@1`` / ``gap``/σ /
  ``refusal_rate`` / CPU 耗时 + 「换特征 vs 打开表示训练」的**分开归因** +
  ``D`` 的代价实测，写 ``exp_repr_compare.json`` / ``exp_repr_compare.md``。
* ``python -m n3d_qa_learn.exp_repr_run summary``
  读取已落盘报告并打印逐组摘要表。

退出码
------
``0`` 成功；``1`` 业务失败（门禁未通过 / 切分不一致 / 报告缺失）；``2`` 参数错误。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

# 控制台编码加固：源码与运行日志含中文，GBK 控制台下必须显式重配
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - 老环境无 reconfigure 时忽略
        pass

from . import exp_repr
from .evaluate import evaluate_refusal, evaluate_step1
from .train import TrainConfig, run_training


class Logger:
    """同时写 stdout 与 UTF-8（无 BOM）日志文件的极简 tee（禁用 PS 的 Tee-Object）。"""

    def __init__(self, path: str = "") -> None:
        self.path = str(path)
        if self.path:
            parent = os.path.dirname(os.path.abspath(self.path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self.handle = (
            open(self.path, "w", encoding="utf-8", newline="\n") if self.path else None
        )

    def log(self, message: str) -> None:
        """打印一行并（若配置了日志文件）追加落盘。"""
        print(message)
        if self.handle is not None:
            self.handle.write(message + "\n")
            self.handle.flush()

    def close(self) -> None:
        """关闭日志文件（幂等）。"""
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _dump(obj: Any) -> None:
    """以 UTF-8 安全的 JSON 打印。"""
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def _kwargs(args: argparse.Namespace, *, epochs: int) -> Dict[str, Any]:
    """把 CLI 参数映射为 run_group 的关键字参数。"""
    return {
        "split_seed": int(args.split_seed),
        "train_seed": int(args.train_seed),
        "epochs": int(epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "backbone_lr": float(args.backbone_lr),
        "logit_scale_init": float(args.logit_scale_init),
        "unknown_train_cap": int(args.unknown_train_cap),
        "max_classes": int(args.max_classes),
        "min_questions": int(args.min_questions),
        "test_every": int(args.test_every),
        "test_per_class": int(args.test_per_class),
        "stage1_epochs": int(args.stage1_epochs),
        "stage2_epochs": int(args.stage2_epochs),
        "qa_cache_dir": str(args.qa_cache_dir),
        "text_dir": str(args.text_dir),
        "profile": str(getattr(args, "profile", "") or exp_repr.DEFAULT_PROFILE),
    }


def _axis_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """收集 CLI 上显式给出的**轴覆盖**（``None`` = 未给出，不覆盖）。

    这些参数是本轮新增的「逐轴单换」入口：只要给出任意一个，本次运行就**只跑**由
    :func:`exp_repr.custom_group` 从公共基线构造出来的那一个自定义组；一个都不给时，
    行为与历史**逐位一致**（跑 ``--group`` 指定的矩阵组，或整张矩阵）。
    """
    mapping = {
        "carrier": getattr(args, "carrier", None),
        "shape": getattr(args, "shape", None),
        "cyl_aspect": getattr(args, "cyl_aspect", None),
        "fc_dim": getattr(args, "fc_dim", None),
        "N": getattr(args, "structure_N", None),
        "y_in": getattr(args, "y_in", None),
        "y_out": getattr(args, "y_out", None),
        "geo_field": getattr(args, "geo_field", None),
        "align_mode": getattr(args, "align_mode", None),
        "align_lambda": getattr(args, "align_lambda", None),
        "proj_dim": getattr(args, "proj_dim", None),
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _resolve_groups(args: argparse.Namespace) -> tuple:
    """解析本次要跑的组（返回 ``(groups_or_None, names_or_None, note)``）。

    * 给出了任意轴覆盖 -> 只跑一个 :func:`exp_repr.custom_group` 造出的自定义组；
    * 否则 -> 沿用 ``--groups`` 子集（空 = 整张矩阵），与历史行为一致。
    """
    overrides = _axis_overrides(args)
    if overrides:
        # 逐轴单换的起点默认是**公共基线**（S0_base）；显式给了 --group 时以它为准。
        base = str(getattr(args, "group", None) or exp_repr.STRUCT_BASELINE_GROUP)
        group = exp_repr.custom_group(base=base, **overrides)
        return [group], [group.name], (
            f"检测到轴覆盖 {overrides}；本次**只跑 1 个自定义组**（自 {base} 逐轴单换）："
            f"{group.name}"
        )
    groups = getattr(args, "groups", "")
    names = [s for s in str(groups).split(",") if s.strip()] if groups else None
    return None, names, "未给出轴覆盖：按矩阵组运行（与历史行为逐位一致）"


def cmd_drill(args: argparse.Namespace) -> int:
    """单条端到端演练（1 组 1 epoch）+ 与模块自带训练循环的等价性对账。

    **epochs 口径（审查收口，皋陶 info —— 口径必须显式标注）**：本子命令的
    ``epochs`` **硬编码为 1**（``_kwargs(args, epochs=1)``），且 ``--epochs``
    **不在** ``drill`` 的参数表内（它只属于 ``run``）。保留硬编码而非暴露
    ``--epochs`` 的理由：``drill`` 的职责是"通路是否成立 + 与 ``run_training``
    是否逐位等价"这条**门槛**，多 epoch 会同时放大运行时间与"看起来跑了"的
    误导空间；矩阵档的 epoch 预算由 ``run --epochs`` 提供（现场实测 40，
    ``C2``/``C3`` 为 ``stage1 30 + stage2 10``）。为避免把演练档误读为矩阵档，
    该口径同时打印在运行日志首行与 ``--help`` 文本中。
    """
    logger = Logger(str(args.log_file))
    try:
        groups, names, note = _resolve_groups(args)
        if groups is not None:
            group = groups[0]
        else:
            group = exp_repr.group_by_name(
                str(names[0]) if names else str(args.group or exp_repr.BASELINE_GROUP)
            )
        kwargs = _kwargs(args, epochs=1)
        t0 = time.time()
        logger.log(
            f"[drill] 组 = {group.name}，"
            f"epochs = 1（演练口径，非矩阵档；矩阵档见 `run --epochs`），"
            f"split_seed = {kwargs['split_seed']}"
        )
        logger.log(f"[drill] {note}")
        logger.log(
            f"[drill] 结构：{group.structure().as_dict()}；"
            f"对齐：mode={group.align_mode}, λ={group.align_lambda}, "
            f"proj_dim={group.proj_dim}"
        )
        # ---- 前置门禁：可构造性预检（构造失败 -> 可读原因 + 退码 1，不硬崩）----
        probe = exp_repr.construction_probe(group, int(exp_repr.profile_by_name(
            kwargs["profile"]).expect_dim))
        if not probe["ok"]:
            logger.log(
                f"[FAIL] 可构造性预检失败：{probe['exc_type']}: {probe['reason']}"
            )
            print(
                f"[FAIL] drill 失败：该组不可构造（{probe['exc_type']}）：{probe['reason']}",
                file=sys.stderr,
            )
            return 1
        logger.log(
            f"[drill] 可构造性预检通过：骨干参数 {probe['parameters']} 个，"
            f"拓扑 {probe['topology']}，参数名 {probe['backbone_parameter_names']}"
        )
        res = exp_repr.run_group(group, **kwargs)
        logger.log(f"[drill] 本实验循环完成（{time.time() - t0:.1f}s）")
        logger.log("[drill] 门禁：可训参数 = " + repr(res["gate"]["trainable_params"]))
        logger.log(
            "[drill] 门禁：更新量 L2 = {:.6e}，更新参数 {} 个 -> {}".format(
                res["gate"]["total_update_l2"], res["gate"]["n_updated_params"],
                "PASS" if res["gate"]["passed"] else "FAIL",
            )
        )
        if not res["gate"]["passed"]:
            print("[FAIL] 门禁未通过：可训参数的更新量不大于 0", file=sys.stderr)
            return 1

        # ---- 等价性对账：同一配置跑模块自带训练循环（save=False，不落盘） ----
        # 口径说明（本轮新增）：`run_training` **只优化交叉熵**，不施加对齐附加项；
        # 因此当 align_mode != "off" 时，"与 run_training 逐位等价"这条对账**不适用**
        # （不是失败），如实登记 applicable=False 并改由**对齐专项门禁**替代：
        # align 项必须参与过 >= 1 个 batch、且全部有限。
        align_on = str(group.align_mode) != "off"
        checks: List[Dict[str, Any]] = []
        all_match = True
        align_gate: Dict[str, Any] = {"applicable": bool(align_on)}
        if align_on:
            ast = res.get("align_stats") or {}
            align_gate.update({
                "mode": ast.get("mode"),
                "lambda": ast.get("lambda"),
                "batches_total": int(ast.get("batches_total", 0)),
                "batches_with_align": int(ast.get("batches_with_align", 0)),
                "batches_align_skipped": int(ast.get("batches_align_skipped", 0)),
                "align_mean": ast.get("align_mean"),
                "align_max": ast.get("align_max"),
                "align_min": ast.get("align_min"),
                "align_nonfinite": int(ast.get("align_nonfinite", 0)),
                "passed": bool(
                    int(ast.get("batches_with_align", 0)) > 0
                    and int(ast.get("align_nonfinite", 0)) == 0
                    and ast.get("align_mean") is not None
                    and abs(float(ast.get("align_mean"))) != float("inf")
                ),
                "reason": (
                    "align_mode != 'off'：run_training 只优化交叉熵、不施加对齐附加项，"
                    "该对账不适用；改由「对齐项有限性 + 至少参与 1 个 batch」替代判定"
                ),
            })
            logger.log(
                "[drill] 对齐专项门禁（G7）：适用 batch {batches_with_align} / "
                "跳过 {batches_align_skipped} / 总 {batches_total}，align 均值 "
                "{align_mean}，非有限计数 {align_nonfinite} -> {verdict}".format(
                    verdict="PASS" if align_gate["passed"] else "FAIL", **align_gate
                )
            )
            if not align_gate["passed"]:
                print("[FAIL] 对齐专项门禁未通过（对齐损失有限性）", file=sys.stderr)
                return 1
        else:
            cfg_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ("stage1_epochs", "stage2_epochs", "profile")}
            cfg = exp_repr.build_group_config(group, **cfg_kwargs)
            prof = exp_repr.profile_by_name(str(kwargs["profile"]))
            ref = run_training(cfg, max_batches=0, artifact_path="", save=False,
                               encoder_config=prof.question)
            ref_metrics = evaluate_step1(ref.model, ref.data, k=3, device=ref.device)
            ref_refusal = evaluate_refusal(ref.model, ref.data, device=ref.device)
            mine = res["metric_step1"]
            checks = [
                {"item": "final_loss", "mine": float(res["history"][-1]["loss"]),
                 "reference": float(ref.final_loss())},
                {"item": "top1_acc", "mine": float(mine["top1_acc"]),
                 "reference": float(ref_metrics.top1_acc)},
                {"item": "macro_acc", "mine": float(mine["macro_acc"]),
                 "reference": float(ref_metrics.macro_acc)},
                {"item": "refusal_rate", "mine": float(res["refusal"]["refusal_rate"]),
                 "reference": float(ref_refusal["refusal_rate"])},
                {"item": "logit_scale_after", "mine": float(res["shifts"]["logit_scale_after"]),
                 "reference": float(ref.model.logit_scale.detach().item())},
            ]
            for row in checks:
                row["match"] = bool(abs(float(row["mine"]) - float(row["reference"])) <= 1e-9)
            all_match = bool(all(row["match"] for row in checks))
        _dump({
            "group": group.as_dict(),
            "split": res["split"],
            "split_seed_effective": res["split_seed_effective"],
            "train_seed_effective": res["train_seed_effective"],
            "gate": res["gate"],
            "ungrouped_trainable_names": res["ungrouped_trainable_names"],
            "geo": res["geo"],
            "align_stats": res.get("align_stats"),
            "alignment_degree": res.get("alignment_degree"),
            "spectrum": res.get("spectrum"),
            "structure_stats": res.get("structure_stats"),
            "metric_step1": res["metric_step1"],
            "refusal": res["refusal"],
            "shifts": res["shifts"],
            "equivalence_with_run_training": {
                "applicable": not align_on, "checks": checks, "all_match": all_match,
                "align_gate": align_gate,
            },
        })
        out_dir = str(args.out_dir) if args.out_dir else exp_repr.EXP_REPR_DIR
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "drill.json"), "w", encoding="utf-8",
                  newline="\n") as handle:
            handle.write(json.dumps(
                {"group": group.as_dict(), "gate": res["gate"],
                 "ungrouped_trainable_names": res["ungrouped_trainable_names"],
                 "geo": res["geo"],
                 "align_stats": res.get("align_stats"),
                 "alignment_degree": res.get("alignment_degree"),
                 "spectrum": res.get("spectrum"),
                 "structure_stats": res.get("structure_stats"),
                 "metric_step1": res["metric_step1"], "refusal": res["refusal"],
                 "shifts": res["shifts"],
                 "equivalence_with_run_training": {
                     "applicable": not align_on, "checks": checks,
                     "all_match": all_match, "align_gate": align_gate,
                 }},
                ensure_ascii=False, indent=1, default=str))
        if not all_match:
            print("[FAIL] 与 run_training 的等价性对账不一致", file=sys.stderr)
            return 1
        logger.log(
            "[OK] drill 通过：1 组 1 epoch 通路成立，门禁 PASS，"
            + ("且对齐专项门禁（有限性）PASS" if align_on
               else "且与 run_training(save=False) 的 5 项训练后量逐位一致")
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - 演练失败必须退码 1 并给出原因
        print(f"[FAIL] drill 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        logger.close()


def cmd_run(args: argparse.Namespace) -> int:
    """放全量对照矩阵，执行可构造性预检与切分一致性断言并写报告。"""
    logger = Logger(str(args.log_file))
    try:
        groups, names, note = _resolve_groups(args)
        kwargs = _kwargs(args, epochs=int(args.epochs))
        t0 = time.time()
        logger.log(
            "[run] 组 = " + repr(names if names else [g.name for g in exp_repr.MATRIX])
        )
        logger.log(f"[run] {note}")
        logger.log(
            "[run] 固定切分：split_seed={split_seed}；训练 seed={train_seed}；"
            "epochs={epochs}；batch_size={batch_size}".format(**kwargs)
        )
        report = exp_repr.run_experiment(
            group_names=names, groups=groups, progress=logger.log, **kwargs
        )
        logger.log(f"[run] 全部组完成（{time.time() - t0:.1f}s）")
        logger.log("[run] 切分一致性断言：通过（qid 有序序列逐位相同）")
        out_dir = str(args.out_dir) if args.out_dir else exp_repr.EXP_REPR_DIR
        paths = exp_repr.write_report(report, out_dir)
        logger.log(f"[run] 报告：{paths['json']}")
        logger.log(f"[run] 报告：{paths['markdown']}")
        _dump({"summary": exp_repr.summarize(report),
               "anchor_comparison": report["anchor_comparison"],
               "construction_precheck": report.get("construction_precheck"),
               "split_identical": report["split_identical"]})
        failed = [r["group"]["name"] for r in report["groups"] if not r["gate"]["passed"]]
        if failed:
            logger.log(f"[FAIL] 以下组门禁未通过（结果判无效）：{failed}")
            return 1
        cf = report.get("construction_failures") or []
        if cf:
            # 「构造失败 -> 判该组无效并跳过」，但**绝不以成功状态落账**：显式列出并退码 1
            logger.log(
                "[FAIL] 以下组构造失败（已判无效并跳过，结果不落成功账）："
                + repr([{"group": x["group"], "exc_type": x["exc_type"],
                         "reason": str(x["reason"])[:160]} for x in cf])
            )
            return 1
        logger.log("[OK] run 完成：全部组门禁 PASS，切分一致性断言通过，无构造失败组")
        return 0
    except KeyError as exc:
        # 未知组名：给出可读原因与可用组名列表，避免只抛裸 KeyError
        print(f"[FAIL] run 失败：组名不可用 —— {exc}", file=sys.stderr)
        print(
            f"[INFO] 可用组名 = {[g.name for g in exp_repr.MATRIX]}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] run 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        logger.close()


def cmd_summary(args: argparse.Namespace) -> int:
    """读取已落盘报告并打印逐组摘要。"""
    logger = Logger(str(args.log_file))
    try:
        # 报告缺失时给出可读原因（不向 stderr 抛全量 traceback），退码 1
        report = exp_repr.load_report(str(args.report))
        rows = exp_repr.summarize(report)
        logger.log(
            "| 组 | freeze_scope | head_input_mode | loss_mode | gap | σ(within) | gap/σ | "
            "1-NN(raw) | 1-NN(repr) | macro | top1 | refusal | q 位移 | 答案表位移 | "
            "argmax 变化率 | 门禁 |"
        )
        logger.log(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
            "--- | --- | --- | --- |"
        )
        for row in rows:
            logger.log(
                "| {group} | {freeze_scope} | {head_input_mode} | {loss_mode} | "
                "{gap:.4f} | {sigma:.4f} | {gap_over_sigma:.4f} | {nn1_top1_raw:.4f} | "
                "{nn1_top1_repr:.4f} | {macro_acc:.4f} | {top1_acc:.4f} | "
                "{refusal_rate:.4f} | {q_shift_mean_l2:.4e} | "
                "{answer_table_shift_l2:.4e} | {argmax_changed_frac:.4f} | {gate} |".format(
                    gate="PASS" if row["gate_passed"] else "**FAIL**", **row
                )
            )
        comp = report.get("anchor_comparison")
        if comp:
            logger.log("")
            logger.log("锚点对账（基线组）：全部落在容差内 = "
                       + ("是" if comp["all_within_tolerance"] else "否"))
            for r in comp["rows"]:
                logger.log(
                    "  - {metric}: 锚点 {anchor:.4f} vs 实测 {measured:.4f} "
                    "(偏差 {delta:+.4f}, 容差 {tolerance:.4f})".format(**r)
                )
        return 0
    except FileNotFoundError as exc:
        print(f"[FAIL] summary 失败：报告缺失 —— {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] summary 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        logger.close()


def cmd_compare(args: argparse.Namespace) -> int:
    """**同切分**下逐组对照两个特征档（词面 D=88 vs bge-m3 D=1024）并写报告。"""
    logger = Logger(str(args.log_file))
    try:
        groups, names, note = _resolve_groups(args)
        profiles = [s for s in str(args.profiles).split(",") if s.strip()]
        kwargs = _kwargs(args, epochs=int(args.epochs))
        # profile 由 --profiles 决定，逐档注入；这里先剔除单档写法
        kwargs.pop("profile", None)
        step2_topk = int(args.step2_topk)
        t0 = time.time()
        logger.log("[compare] 组 = " + repr(names if names else [g.name for g in exp_repr.MATRIX]))
        logger.log(f"[compare] {note}")
        logger.log(f"[compare] 特征档 = {profiles}")
        logger.log(
            "[compare] 固定切分：split_seed={split_seed}；训练 seed={train_seed}；"
            "epochs={epochs}；batch_size={batch_size}".format(
                split_seed=kwargs["split_seed"], train_seed=kwargs["train_seed"],
                epochs=kwargs["epochs"], batch_size=kwargs["batch_size"],
            )
        )
        logger.log(
            "[compare] 步骤 2 口径：文本行用**与该档步骤 1 同族同维**的编码器"
            "（词面档 local-hash / 语义档 bge-m3 role=text_line）；"
            f"检索池上限={int(args.step2_pool_cap)}（0=全量），topk={step2_topk}"
        )

        def _progress(msg: str) -> None:
            logger.log(msg)

        report = exp_repr.run_comparison(
            profiles=profiles,
            group_names=names,
            groups=groups,
            step2_product_dir=str(args.step2_product_dir),
            step2_pool_cap=int(args.step2_pool_cap),
            step2_topk=step2_topk,
            progress=_progress,
            **kwargs,
        )
        logger.log(f"[compare] 全部组完成（{time.time() - t0:.1f}s）")
        out_dir = str(args.out_dir) if args.out_dir else exp_repr.EXP_REPR_DIR
        paths = exp_repr.write_comparison_report(report, out_dir)
        logger.log(f"[compare] 报告：{paths['json']}")
        logger.log(f"[compare] 报告：{paths['markdown']}")

        # ---- G5：逐组门禁 ----
        bad: List[str] = []
        for pname, run in report["runs"].items():
            for res in run["groups"]:
                if not res["gate"]["passed"]:
                    bad.append(f"{pname}/{res['group']['name']}")
        v = report["verdict"]
        logger.log(
            "[compare] 主判据（macro 与 步骤2 自检索 Recall@1 同时更优）："
            f"同时更优 = {v['groups_both_better']}；"
            f"未同时更优 = {v['groups_not_both_better']}；"
            f"any={v['any_group_both_better']} all={v['all_groups_both_better']}"
        )
        logger.log(
            "[compare] G1 跨档切分一致性断言："
            + ("通过" if report["split"]["identical_across_profiles"] else "未通过")
            + f"（{len(report['split']['cross_profile_checks'])} 项）"
        )
        cf = report.get("construction_failures") or {}
        n_cf = sum(len(v) for v in cf.values())
        logger.log(
            f"[compare] G6 构造失败清单：{n_cf} 项"
            + ("" if n_cf == 0 else f" -> {cf}")
        )
        ax = report.get("axis_attribution") or {}
        logger.log(
            "[compare] 逐轴主判据（各档都相对公共基线同时更优，"
            f"基线={ax.get('baseline')}）：通过 = {ax.get('groups_axis_both_better_all_profiles')}；"
            f"未通过 = {ax.get('groups_axis_not_both_better')}"
        )
        logger.log(f"[compare] CPU 耗时（秒）：{report['seconds']['per_profile_total']}；"
                   f"总计 {report['seconds']['grand_total']:.1f}")
        _dump({
            "verdict": v,
            "axis_attribution_summary": {
                "baseline": ax.get("baseline"),
                "criterion": ax.get("criterion"),
                "both_better_all_profiles": ax.get("groups_axis_both_better_all_profiles"),
                "not_both_better": ax.get("groups_axis_not_both_better"),
            },
            "attribution_feature_only": report["attribution"]["feature_only"],
            "attribution_training_only": report["attribution"]["training_only"],
            "construction_failures": {k: len(v) for k, v in cf.items()},
            "split_identical_across_profiles": report["split"]["identical_across_profiles"],
            "seconds": report["seconds"]["per_profile_total"],
        })
        if bad:
            logger.log(f"[FAIL] 以下组门禁未通过（结果判无效）：{bad}")
            return 1
        if n_cf:
            logger.log("[FAIL] 存在构造失败组（已判无效并跳过，结果不落成功账）")
            return 1
        logger.log("[OK] compare 完成：全部组门禁 PASS，跨档切分一致性断言通过，无构造失败组")
        return 0
    except KeyError as exc:
        print(f"[FAIL] compare 失败：组名/档名不可用 —— {exc}", file=sys.stderr)
        print(f"[INFO] 可用组名 = {[g.name for g in exp_repr.MATRIX]}", file=sys.stderr)
        print(f"[INFO] 可用档 = {sorted(exp_repr.FEATURE_PROFILES)}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] compare 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        logger.close()


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 解析器。"""
    parser = argparse.ArgumentParser(
        prog="n3d_qa_learn.exp_repr_run",
        description="n3d_qa_learn 表示训练对照实验（只写 checkpoints/qa_learn/_verify/exp_repr/）",
        epilog=(
            "结构 / 对齐轴开关（挂在 drill / run / compare 三个子命令上；"
            "给出任一即只跑 1 个从公共基线 S0_base 逐轴单换出来的自定义组）："
            " --carrier / --shape / --cyl-aspect / --fc-dim / --struct-N（别名 --N）"
            " / --y-in / --y-out / --geo-field / --align-mode / --align-lambda / --proj-dim。"
            "详见 `exp_repr_run <子命令> --help`。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--split-seed", type=int, default=exp_repr.BASE_SPLIT_SEED,
                       help="切分种子（与训练种子分离；所有组共用同一份切分）")
        p.add_argument("--train-seed", type=int, default=exp_repr.BASE_TRAIN_SEED)
        p.add_argument("--batch-size", type=int, default=exp_repr.BASE_BATCH_SIZE)
        p.add_argument("--lr", type=float, default=exp_repr.BASE_LR)
        p.add_argument("--backbone-lr", type=float, default=exp_repr.BASE_BACKBONE_LR)
        p.add_argument("--logit-scale-init", type=float, default=exp_repr.BASE_LOGIT_SCALE_INIT)
        p.add_argument("--unknown-train-cap", type=int, default=exp_repr.BASE_UNKNOWN_TRAIN_CAP)
        p.add_argument("--max-classes", type=int, default=exp_repr.BASE_MAX_CLASSES)
        p.add_argument("--min-questions", type=int, default=exp_repr.BASE_MIN_QUESTIONS)
        p.add_argument("--test-every", type=int, default=exp_repr.BASE_TEST_EVERY)
        p.add_argument("--test-per-class", type=int, default=exp_repr.BASE_TEST_PER_CLASS)
        p.add_argument("--stage1-epochs", type=int, default=exp_repr.STAGE1_EPOCHS)
        p.add_argument("--stage2-epochs", type=int, default=exp_repr.STAGE2_EPOCHS)
        p.add_argument("--qa-cache-dir", type=str, default="checkpoints/triviaqa/_cache")
        p.add_argument("--text-dir", type=str, default="data/doc")
        p.add_argument("--profile", type=str, default=exp_repr.DEFAULT_PROFILE,
                       choices=sorted(exp_repr.FEATURE_PROFILES.keys()),
                       help="特征档：lexical-88（词面，现状）或 bge-m3-1024（语义）")
        p.add_argument("--out-dir", type=str, default="", help="输出目录（默认验证目录）")
        p.add_argument("--log-file", type=str, default="", help="UTF-8 无 BOM 日志文件")
        # ---- 本轮新增：结构 / 对齐轴（逐轴单换；给出任意一个即只跑 1 个自定义组）----
        p.add_argument("--carrier", type=str, default=None,
                       choices=sorted(exp_repr.CARRIER_GROUPS),
                       help="载体组：base（head_backbone+concat）/ B1_concat / B2_n3d")
        p.add_argument("--shape", type=str, default=None,
                       choices=list(exp_repr.SHAPES),
                       help="空间形状：sphere / cube / cylinder")
        p.add_argument("--cyl-aspect", type=float, default=None,
                       help="圆柱长径比 λ（仅 shape=cylinder 生效；本轮矩阵用 1.0）")
        p.add_argument("--fc-dim", type=int, default=None,
                       help="两端全连接包裹：0 关闭 / -1 跟随 N / >0 显式宽度")
        p.add_argument("--struct-N", "--N", dest="structure_N", type=int, default=None,
                       help="神经元规模 N（本轮边界 64 / 256）")
        p.add_argument("--y-in", dest="y_in", type=int, default=None,
                       help="输入突触数（默认 4；y=2 会被上游连通性下限校验拒绝）")
        p.add_argument("--y-out", dest="y_out", type=int, default=None,
                       help="输出突触数（默认 4）")
        p.add_argument("--geo-field", type=str, default=None,
                       choices=list(exp_repr.GEO_FIELDS),
                       help="几何权重场：none / additive")
        p.add_argument("--align-mode", type=str, default=None,
                       choices=list(exp_repr.ALIGN_MODE_CHOICES),
                       help="对齐机制：off / proj_supcon（机制 B）/ distill（机制 C）")
        p.add_argument("--align-lambda", dest="align_lambda", type=float, default=None,
                       help="对齐损失权重 λ（总损失 = ce + λ·align）")
        p.add_argument("--proj-dim", dest="proj_dim", type=int, default=None,
                       help="投影头宽度：0 关闭 / -1 跟随 D（D→D，机制 B）")

    p = sub.add_parser(
        "drill",
        help="单条端到端演练（1 组 1 epoch + 等价性对账）",
        description=(
            "单条端到端演练：固定 epochs = 1（演练口径，非矩阵档），"
            "断言门禁并 train.run_training(save=False) 逐项对账。"
            "本子命令不提供 --epochs；矩阵档的 epoch 预算请用 `run --epochs`。"
        ),
    )
    add_common(p)
    p.add_argument("--group", type=str, default=None,
                   help=f"组名（默认 {exp_repr.BASELINE_GROUP}；给了轴覆盖时起点默认 "
                        f"{exp_repr.STRUCT_BASELINE_GROUP}）")
    p.set_defaults(func=cmd_drill)

    p = sub.add_parser("run", help="全量对照矩阵")
    add_common(p)
    p.add_argument("--epochs", type=int, default=exp_repr.BASE_EPOCHS)
    p.add_argument("--groups", type=str, default="",
                   help="逗号分隔的组名子集（空 = 全部 8 组）")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("summary", help="打印已落盘报告的摘要表")
    p.add_argument("--report", type=str, default="", help="报告 JSON 路径")
    p.add_argument("--log-file", type=str, default="")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser(
        "compare",
        help="同切分下逐组对照两个特征档（词面 D=88 vs bge-m3 D=1024）并写报告",
        description=(
            "在**同一 split_seed / train_seed** 下逐组跑两个特征档，执行跨档切分一致性"
            "断言（G1），逐组报 macro / 步骤 2 自检索 Recall@1 / gap/σ / refusal / 耗时，"
            "并把「换特征」与「打开表示训练」的贡献分开归因。"
        ),
    )
    add_common(p)
    p.add_argument("--epochs", type=int, default=exp_repr.BASE_EPOCHS)
    p.add_argument("--groups", type=str, default="",
                   help="逗号分隔的组名子集（空 = 全部 8 组）")
    p.add_argument("--profiles", type=str,
                   default=f"{exp_repr.PROFILE_LEXICAL},{exp_repr.PROFILE_SEMANTIC}",
                   help="逗号分隔的特征档（第一个为参照档）")
    p.add_argument("--step2-product-dir", type=str, default="",
                   help="n3d_qa 冻结产物目录（空 = 自动定位）")
    p.add_argument("--step2-pool-cap", type=int, default=0,
                   help="步骤 2 检索池行数上限（0 = 全量；限批会在报告中显式登记）")
    p.add_argument("--step2-topk", type=int, default=5)
    p.set_defaults(func=cmd_compare)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口（返回进程退出码）。"""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
