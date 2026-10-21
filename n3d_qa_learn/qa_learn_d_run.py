"""轮 11 / 轮 D 的 CLI：``probe``（200 步计时探针）与 ``run``（正式配置）。

驱动方式：``python -m n3d_qa_learn.qa_learn_d_run {probe|run}``

* ``probe`` —— 用**同一套代码**跑 200 步并计时，确认单次 < 600 s（产物：``qa_learn_d_probe.json``）；
* ``run`` —— 正式配置（300 步 / lr=1e-2 / full-batch Adam），落 7 件内容产物 +
  可选 ``determinism_check.json``（``--repro-dir``）。

纪律
----
* 产物只写 ``checkpoints/qa_learn/_verify/qa_learn_d/``；``--out-dir`` / ``--log-file``
  建目录**之前**校验，越界退码 2 且不创建目录；
* **不产 zip、不落盘裸特征矩阵**；无挂钟字段 ⇒ 同参双跑逐字节一致；
* 日志 UTF-8 无 BOM；控制台 GBK 安全（不可表示字符 backslashreplace 兜底）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """打印一行进度（对不可表示字符做 ``backslashreplace`` 兜底）。"""
    text = str(msg)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = str(getattr(sys.stdout, "encoding", "") or "ascii")
        safe = text.encode(enc, errors="backslashreplace").decode(enc, errors="replace")
        print(safe, flush=True)


#: CLI 子命令（顺序敏感）。
QA_LEARN_D_COMMANDS: Tuple[str, ...] = ("probe", "run")

#: 正式产物文件名（**唯一来源**）。
BASELINE_FILE: str = "qa_learn_d_baseline.json"
CONFIG_FILE: str = "qa_learn_d_config.json"
STEP0_FILE: str = "qa_learn_d_step0.json"
CURVE_FILE: str = "qa_learn_d_curve.json"
EVAL_FILE: str = "qa_learn_d_eval.json"
CLASS_DELTA_FILE: str = "qa_learn_d_class_delta.json"
GATE_FILE: str = "qa_learn_d_gate.json"
DETERMINISM_EVIDENCE_FILE: str = "determinism_check.json"
PROBE_FILE: str = "qa_learn_d_probe.json"

#: 双跑证据实际比对的**七件内容产物**（不含证据文件自身、不含 probe）。
COMPARED_FILES: Tuple[str, ...] = (
    BASELINE_FILE,
    CONFIG_FILE,
    STEP0_FILE,
    CURVE_FILE,
    EVAL_FILE,
    CLASS_DELTA_FILE,
    GATE_FILE,
)


# ---------------------------------------------------------------------------
# 小工具与守卫
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


def out_root() -> str:
    """本模块**唯一**允许写盘的根目录。"""
    import n3d_qa_learn.qa_learn_d as QD

    return str(QD.DEFAULT_OUT_DIR)


def _guard_out_dir(out_dir: str) -> str:
    """守卫产物目录：**只允许**写在 ``checkpoints/qa_learn/_verify/qa_learn_d/`` 之下。"""
    text = str(out_dir) if out_dir else out_root()
    root_abs = os.path.abspath(out_root())
    norm_abs = os.path.abspath(text)
    if not (norm_abs == root_abs or norm_abs.startswith(root_abs + os.sep)):
        raise ValueError(
            "产物目录只允许写在 "
            + repr(out_root().replace(chr(92), "/"))
            + " 之下（避免覆盖别轮交付产物），当前 = "
            + repr(norm_abs.replace(chr(92), "/"))
        )
    return text


def _guard_log_file(log_file: str) -> str:
    """守卫日志文件（与产物目录**同等**规则；越界抛 ``ValueError``，不创建目录）。"""
    text = str(log_file)
    if not text:
        return text
    root_abs = os.path.abspath(out_root())
    norm_abs = os.path.abspath(text)
    if not norm_abs.startswith(root_abs + os.sep):
        raise ValueError(
            "日志文件只允许写在 "
            + repr(out_root().replace(chr(92), "/"))
            + " 之下（日志与产物同等守卫），当前 = "
            + repr(norm_abs.replace(chr(92), "/"))
        )
    return text


def _resolve_product_dir(explicit: str) -> str:
    """定位 ``n3d_qa`` 冻结产物目录（只读）。"""
    import n3d_qa_learn.qa_learn_d as QD

    if explicit:
        if os.path.isfile(os.path.join(explicit, "answer_table.jsonl")):
            return str(explicit)
        raise FileNotFoundError("目录 " + repr(explicit) + " 下缺少 answer_table.jsonl")
    if os.path.isfile(os.path.join(QD.DEFAULT_PRODUCT_DIR, "answer_table.jsonl")):
        return str(QD.DEFAULT_PRODUCT_DIR)
    raise FileNotFoundError(
        "未找到 n3d_qa 冻结产物（answer_table.jsonl）；候选 = " + repr(QD.DEFAULT_PRODUCT_DIR)
    )


# ---------------------------------------------------------------------------
# probe / run
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    """200 步计时探针：确认单次 < 600 s，且 step0 等价成立。"""
    import n3d_qa_learn.qa_learn_d as QD

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    product_dir = _resolve_product_dir(str(getattr(args, "product_dir", "") or ""))
    seed = int(getattr(args, "seed", QD.DEFAULT_SEED))
    started = time.perf_counter()
    result = QD.run_experiment(
        product_dir,
        str(QD.DEFAULT_PROFILE),
        steps=int(QD.PROBE_STEPS),
        lr=float(QD.DEFAULT_LR),
        seed=seed,
        log=_log,
    )
    elapsed = float(time.perf_counter() - started)
    doc = {
        "version": str(QD.QA_LEARN_D_VERSION),
        "kind": "probe",
        "steps": int(QD.PROBE_STEPS),
        "lr": float(QD.DEFAULT_LR),
        "elapsed_seconds": elapsed,
        "under_600s": bool(elapsed < 600.0),
        "step0": dict(result["step0"]),
        "test_macro": float(result["eval"]["metrics"]["macro"]),
        "test_micro": float(result["eval"]["metrics"]["micro"]),
        "passed": bool(result["gate"]["passed"]),
        "rule": (
            "探针只用于确认「单次前台 < 600 s」与 step0 等价成立；它不是正式结论，"
            "也不落 7 件内容产物。"
        ),
    }
    sha, size = _write_json(os.path.join(out_dir, PROBE_FILE), doc)
    _log(
        "[probe] 200 步耗时 %.1f s（< 600 s = %s）；step0 n_argmax_diff=%d；测试侧 macro=%.6f micro=%.6f"
        % (
            elapsed,
            str(doc["under_600s"]),
            int(result["step0"]["n_argmax_diff"]),
            float(doc["test_macro"]),
            float(doc["test_micro"]),
        )
    )
    _log("[probe] " + PROBE_FILE + " = " + str(size) + " B " + sha[:16])
    if not doc["under_600s"]:
        _log("[probe] 失败：单次探针超过 600 s，禁止据此跑正式配置")
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """正式配置：落 7 件内容产物；给 ``--repro-dir`` 时落双跑证据。"""
    import n3d_qa_learn.qa_learn_d as QD

    out_dir = _guard_out_dir(str(getattr(args, "out_dir", "") or ""))
    repro_dir = str(getattr(args, "repro_dir", "") or "")
    product_dir = _resolve_product_dir(str(getattr(args, "product_dir", "") or ""))
    seed = int(getattr(args, "seed", QD.DEFAULT_SEED))
    steps = int(getattr(args, "steps", QD.DEFAULT_STEPS))
    lr = float(getattr(args, "lr", QD.DEFAULT_LR))
    result = QD.run_experiment(
        product_dir, str(QD.DEFAULT_PROFILE), steps=steps, lr=lr, seed=seed, log=_log
    )
    names = {
        BASELINE_FILE: result["baseline"],
        CONFIG_FILE: result["config"],
        STEP0_FILE: result["step0"],
        CURVE_FILE: result["curve"],
        EVAL_FILE: result["eval"],
        CLASS_DELTA_FILE: result["class_delta"],
        GATE_FILE: result["gate"],
    }
    for name, obj in names.items():
        sha, size = _write_json(os.path.join(out_dir, name), obj)
        _log("[run] " + name + " = " + str(size) + " B " + sha[:16])
    metrics = result["eval"]["metrics"]
    gate = result["gate"]
    _log(
        "[run] 测试侧：macro=%.6f / micro=%.6f；零召回类 %d / %d；双判据 passed=%s"
        % (
            float(metrics["macro"]),
            float(metrics["micro"]),
            int(metrics["n_zero_recall_classes"]),
            int(metrics["n_classes_with_test_questions"]),
            str(gate["passed"]),
        )
    )
    for row in result["eval"]["strata"]["rows"]:
        if not int(row.get("n_classes", 0)):
            _log("[run] 分层 min_test>=%d：不适用（该层为空）" % int(row["min_test_questions"]))
            continue
        _log(
            "[run] 分层 min_test>=%d：类 %d / model macro %.6f / 原型 macro %.6f / Δ %+.6f"
            % (
                int(row["min_test_questions"]),
                int(row["n_classes"]),
                float(row["model_macro"]),
                float(row["prototype_macro"]),
                float(row["delta_vs_prototype"]),
            )
        )
    _log("[run] 双判据结论：" + str(gate["verdict"])[:160] + "…")
    if repro_dir:
        det = determinism_block(str(out_dir), str(repro_dir))
        det["missing_file_counterexample"] = determinism_missing_file_probe(
            str(out_dir), str(repro_dir)
        )
        probe = det["missing_file_counterexample"]
        if not bool(probe["determinism_ok_when_complete"]):
            raise AssertionError(
                "双跑自检失败：**齐备态** determinism_ok 应为 True，实际为 False ⇒ 判据不可用"
            )
        if bool(probe["determinism_ok_when_missing"]):
            raise AssertionError(
                "双跑自检失败：删掉一件产物后 determinism_ok 仍为 True ⇒ 判据恒真"
            )
        _write_json(os.path.join(out_dir, DETERMINISM_EVIDENCE_FILE), det)
        _log(
            "[run] 双跑一致性：应比对 %d 件、实比对 %d 件，缺失 = %s，逐字节相同 = %s"
            % (
                int(det["n_expected_files"]),
                int(det["n_common_files"]),
                str(det["missing_files"]),
                str(det["determinism_ok"]),
            )
        )
    return 0


# ---------------------------------------------------------------------------
# 双跑证据（与 qa_scoped_run / answer_merge_run 同口径：七件齐备且逐字节相等）
# ---------------------------------------------------------------------------


def determinism_block(out_dir: str, repro_dir: str) -> Dict[str, Any]:
    """双跑证据：**七件内容产物齐备且逐字节相等**才算 ``determinism_ok``。"""
    expected = [str(x) for x in COMPARED_FILES]
    files: List[Dict[str, Any]] = []
    missing: List[str] = []
    for name in expected:
        pa = os.path.join(out_dir, str(name))
        pb = os.path.join(repro_dir, str(name))
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            missing.append(str(name))
            continue
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
    not_equal = [f["name"] for f in files if not bool(f["bitwise_equal"])]
    return {
        "a_dir": str(out_dir),
        "b_dir": str(repro_dir),
        "path_dependence_note": (
            "`a_dir` / `b_dir` 保留命令原样串（会进本文件内容）；**内容产物本身不含绝对 / "
            "机器相关路径串**（相对 `product_dir` 仍登记，如 `checkpoints/qa_learn/dataset`）。"
            "本文件自身被排除在比对集合外。"
        ),
        "files": files,
        "n_common_files": int(len(files)),
        "n_expected_files": int(len(expected)),
        "missing_files": missing,
        "compared_files": expected,
        "not_equal_files": not_equal,
        "determinism_ok": bool(len(files) == len(expected) and not not_equal),
        "determinism_ok_semantics": (
            "determinism_ok = 「七件内容产物齐备且逐个逐字节相等」；缺任何一件即 False 并登记。"
        ),
    }


def determinism_missing_file_probe(out_dir: str, repro_dir: str) -> Dict[str, Any]:
    """**非恒真证明**：齐备时 True；删一件后必须 False。"""
    import shutil
    import tempfile

    expected = [str(x) for x in COMPARED_FILES]
    with tempfile.TemporaryDirectory(prefix="qld_det_probe_") as tmp:
        left = os.path.join(tmp, "a")
        right = os.path.join(tmp, "b")
        os.makedirs(left)
        os.makedirs(right)
        for name in expected:
            for base, source_dir in ((left, out_dir), (right, repro_dir)):
                src = os.path.join(str(source_dir), str(name))
                if os.path.isfile(src):
                    shutil.copyfile(src, os.path.join(base, str(name)))
        complete_block = determinism_block(left, right)
        removed = str(expected[-1])
        target = os.path.join(right, removed)
        removed_ok = os.path.isfile(target)
        if removed_ok:
            os.remove(target)
        broken_block = determinism_block(left, right)
    return {
        "removed_file": removed,
        "removed_in_temp_b_side": bool(removed_ok),
        "determinism_ok_when_complete": bool(complete_block["determinism_ok"]),
        "determinism_ok_when_missing": bool(broken_block["determinism_ok"]),
        "missing_files_when_missing": [str(x) for x in broken_block["missing_files"]],
        "rule": (
            "同一份代码、同一批文件：齐备时 True；删掉一件后必须 False 且 `missing_files` "
            "恰为被删文件。临时目录在 TEMP 下、用完即删，**路径不进产物**。"
        ),
        "temp_paths_not_persisted": True,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        """其余属性透传。"""
        return getattr(self._primary, item)


def _add_arguments(p: argparse.ArgumentParser) -> None:
    """把参数表挂到给定子解析器上（**唯一**定义点）。"""
    import n3d_qa_learn.qa_learn_d as QD

    p.add_argument("--product-dir", default="", help="n3d_qa 冻结产物目录（只读）")
    p.add_argument("--out-dir", default="", help="产物目录（默认 _verify/qa_learn_d）")
    p.add_argument("--log-file", default="", help="同时写 UTF-8(无 BOM) 日志文件")
    p.add_argument("--seed", type=int, default=int(QD.DEFAULT_SEED), help="单 seed（默认 42）")
    p.add_argument("--steps", type=int, default=int(QD.DEFAULT_STEPS), help="训练步数**上限**（默认 3000；训练侧收敛即提前停）")
    p.add_argument("--lr", type=float, default=float(QD.DEFAULT_LR), help="Adam 学习率（默认 1e-2）")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    import n3d_qa_learn.qa_learn_d as QD

    parser = argparse.ArgumentParser(
        prog="n3d_qa_learn.qa_learn_d_run",
        description=(
            "轮 11 / 轮 D：A 口径（82 类）上的可学 D->D 变换（恒等初始化）+ 冻结类质心 + 可学温度；"
            "双判据 macro > 0.241276 且 micro >= 0.183312；产物一律写 "
            + str(QD.DEFAULT_OUT_DIR).replace(os.sep, "/")
            + " 之下。"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    helps = {
        "probe": "200 步计时探针（确认 < 600 s 且 step0 等价成立；落 qa_learn_d_probe.json）",
        "run": "正式配置（上限 3000 步，训练侧收敛即停）：落 7 件内容产物，可附双跑证据",
    }
    for name in QA_LEARN_D_COMMANDS:
        p = sub.add_parser(name, help=helps[str(name)])
        _add_arguments(p)
        if str(name) == "run":
            p.add_argument(
                "--repro-dir", default="", help="对照产物目录：给定时产出双跑逐字节一致证据"
            )
        p.set_defaults(func=cmd_probe if str(name) == "probe" else cmd_run)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口（``python -m n3d_qa_learn.qa_learn_d_run <cmd>``）。"""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    log_file = str(getattr(args, "log_file", "") or "")
    out_dir = str(getattr(args, "out_dir", "") or "")
    try:
        _guard_out_dir(out_dir)
        _guard_log_file(log_file)
    except ValueError as exc:
        _log("[qa_learn_d] 非法参数 / 契约校验失败：" + str(exc))
        return 2
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
    except UnicodeEncodeError as exc:
        _log("[qa_learn_d] 输出编码错误：" + type(exc).__name__ + ": " + str(exc))
        return 1
    except FileNotFoundError as exc:
        _log("[qa_learn_d] 输入产物缺失：" + str(exc))
        return 1
    except (ValueError, KeyError) as exc:
        _log("[qa_learn_d] 非法参数 / 契约校验失败：" + str(exc))
        return 2
    except AssertionError as exc:
        _log("[qa_learn_d] 现场断言失败：" + str(exc))
        return 1
    except RuntimeError as exc:
        _log("[qa_learn_d] 运行失败（依赖 / 产物不可用）：" + str(exc))
        return 1
    finally:
        if handle is not None:
            sys.stdout = original
            handle.close()

if __name__ == "__main__":
    raise SystemExit(main())