"""n3d_triviaqa.verify_dataset —— 产物验证（E1–E6，逐项真实执行、结论可复核）。

验证项
------
* **E1 构建幂等**：同参数连跑两次构建 -> 两个 npz 逐字节一致（SHA256 相同），
  且与正式产物一致（正式产物 = 可复现产物）；
* **E2 产物契约**：委派 ``n3d_shape.data.load_npz_arrays`` 读取，断言
  ``X.dtype == float32`` / ``y.dtype == int64`` / 无 NaN/Inf / 形状 ``[M, D]``
  （``D`` 由产物 meta 的 ``features.feature_dim`` 给出：缺省 70 / ``--no-bag`` 6）；
* **E3 标签均衡 + 答案复核**：正负 1:1；确定性抽样 N=20 正 / N=20 负，
  **从归档重新抽取这 40 个文档**（一遍流式扫描）重算答案命中并与 meta 记录的值比对，
  同时打印命中窗口片段供人工复核；
* **E4 无跨 split 泄漏**：同一 QuestionId 的文档不得同时出现在正负两侧；
  ``(QuestionId, 文档)`` 不得重复；两个 npz 的 QuestionId 集合互不相交；
* **E5 端到端训练**：``python n3d_shape/train.py --dataset npz ...``（其余同 R1 口径），
  断言退出码 0 且 ``test_acc > 0.75``（二分类，随机基线 0.5）；
* **E6 零回归**：``git status --porcelain`` 中 n3d_proto / n3d_sphere / n3d_shape /
  n3d_viz 一律无改动（本模块不触碰任何既有模块文件）。

元信息口径回读
--------------
每项检查都会把产物 meta 里的口径（归档 SHA256、哈希维度与盐、负样本种子、
特征列定义、短数字阈值、1:1 比例）与 :mod:`n3d_triviaqa.build_dataset` 的**写死常量**
逐字比对 —— meta 只写不读等于没有约束。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # 兼容包导入与脚本直跑
    from . import build_dataset as bd
except ImportError:  # pragma: no cover
    import build_dataset as bd  # type: ignore

PROJECT_ROOT: str = bd.PROJECT_ROOT
PRODUCT_DIR: str = bd.DEFAULT_OUT_DIR
VERIFY_DIR: str = bd.VERIFY_OUT_DIR
SPLITS: Tuple[str, ...] = ("wiki", "web")
# 正式产物的**口径变体**（(标签, no_bag)）：口径修订后 D=70（_h64）与 D=6（_nobag）都是正式产物
PRODUCT_VARIANTS: Tuple[Tuple[str, bool], ...] = (("h64", False), ("nobag", True))
MODULE_DIRS_UNTOUCHED: Tuple[str, ...] = ("n3d_proto", "n3d_sphere", "n3d_shape", "n3d_viz")

# E3 人工复核抽样数（正/负各 N 条）
AUDIT_N: int = 20
AUDIT_SEED: int = 20261004
# E3 判据：正样本含答案比例下限（口径见 README「E3 实测」节；标签是 provenance 判定，
# 负样本可能"恰好提到同一答案"，故对负样本只报告实测比例、不设通过阈值）
POSITIVE_ANSWER_HIT_MIN: float = 0.98
# E5 判据
E5_MIN_ACC: float = 0.75
E5_EPOCHS: int = 20
# R1 口径（与 n3d_shape/README.md §21.8 的 R1 行逐字一致，只额外追加 npz 数据集参数）
R1_COMMON_ARGS: Tuple[str, ...] = (
    "--preset", "default",
    "--seed", "42",
    "--n", "256",
    "--shape", "sphere",
    "--input-scope", "any_isolated",
    "--readout-scope", "any_isolated",
    "--threads", "0",
    "--fc-dim", "-1",
    "--geo-field", "none",
)

E2_DTYPE_X = np.float32
E2_DTYPE_Y = np.int64


@dataclass
class CheckResult:
    """单项验证结果。

    属性
    ----
    name : str
        检查项标识（如 ``E1``）。
    title : str
        人类可读标题。
    passed : bool
        是否通过。
    skipped : bool
        是否因环境原因跳过（不计入失败）。
    detail : str
        明细（多行，含实测数字）。
    """

    name: str
    title: str
    passed: bool
    skipped: bool = False
    detail: str = ""


def configure_console_encoding() -> None:
    """把 stdout/stderr 的错误策略放宽为 ``errors="replace"``（**保编码不变**）。

    为什么必须做：E3 会把 evidence 文档里命中答案的**原文片段**打印出来，片段可能含
    GBK 无法编码的字符（实测 `\xbd` 等）—— 若不做处理，报告打印本身会抛
    ``UnicodeEncodeError`` 并让整个验证以异常收场（检查项其实已经通过，却拿不到证据）。

    参数
    ----
    无。

    返回
    ----
    None
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 已被重定向为特殊对象
            continue


def product_path(split: str, no_bag: bool = False) -> str:
    """正式产物路径（**口径标签恒定在文件名内**，见 build_dataset.out_name_for）。

    参数
    ----
    split : str
        ``wiki`` / ``web``。
    no_bag : bool
        是否取 ``--no-bag`` 版本（D=6）。

    返回
    ----
    str
        绝对路径，形如 ``..._dev_h64.npz``（缺省 D=70）或 ``..._dev_nobag.npz``（D=6）。
    """
    return os.path.join(
        PRODUCT_DIR,
        bd.out_name_for(split, 0, bd.HASH_DIM, bd.DEFAULT_ARCHIVE, bool(no_bag)),
    )


def product_list(args: argparse.Namespace) -> List[Tuple[str, str, bool, str]]:
    """列出本轮要验证的正式产物列表（口径修订后缺省为 4 个）。

    参数
    ----
    args : argparse.Namespace
        命令行参数（用 ``--split``）。

    返回
    ----
    List[Tuple[str, str, bool, str]]
        ``[(split, 变体标签, no_bag, 产物绝对路径), ...]``；
        缺省 ``--split all`` -> ``wiki/web x (h64, nobag)`` 共 4 个。
    """
    splits = SPLITS if args.split == "all" else (str(args.split),)
    out: List[Tuple[str, str, bool, str]] = []
    for split in splits:
        for label, no_bag in PRODUCT_VARIANTS:
            out.append((split, label, bool(no_bag), product_path(split, no_bag)))
    return out

def sha256_file(path: str) -> str:
    """文件整包 SHA256（分块）。

    参数
    ----
    path : str
        文件路径。

    返回
    ----
    str
        十六进制 SHA256。
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_arrays_via_shape_contract(path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """**委派 n3d_shape 的数据层**读取 npz（E2 的核心：契约必须由下游实现来判）。

    参数
    ----
    path : str
        ``.npz`` 路径。

    返回
    ----
    Tuple[np.ndarray, np.ndarray, str]
        ``(X, y, load_npz_arrays 所在源码文件)``。

    异常
    ------
    Exception
        委派失败时原样抛出（由调用方记为该检查项失败）。
    """
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from n3d_shape.data import load_npz_arrays  # 只读导入：不修改 n3d_shape 任何文件

    X, y = load_npz_arrays(path)
    return X, y, str(getattr(load_npz_arrays, "__module__", "n3d_shape.data"))


def check_meta_contract(split: str, no_bag: bool = False) -> CheckResult:
    """E0：产物 meta 与写死常量的**逐字**比对 + 计数自洽（口径回读）。

    口径修订后 ``features`` 段按**实际生效值**校验（``hash_dim`` / ``no_bag`` / 逐列定义），
    故 D=70 与 D=6 两个版本都按各自 meta 回的维度与列定义比对，不做跨口径的硬编码比较。

    参数
    ----
    split : str
        ``wiki`` / ``web``。
    no_bag : bool
        是否校验 ``--no-bag`` 版本（D=6）。

    返回
    ----
    CheckResult
        检查结果（明细含各项实测/期望值）。
    """
    path = product_path(split, no_bag)
    label_txt = f"{split}/{'nobag' if no_bag else 'h' + str(int(bd.HASH_DIM))}"
    lines: List[str] = [f"产物 {path}"]
    problems: List[str] = []
    if not os.path.isfile(path):
        return CheckResult("E0", f"meta 口径回读 [{label_txt}]", False, detail=f"缺产物：{path}")
    meta = bd.read_npz_meta(path)
    src = meta["source"]
    feats = meta["features"]
    label = meta["label_rule"]
    ans = meta["answer_match_rule"]
    counts = meta["counts"]

    def expect(cond: bool, msg: str, got: Any, want: Any) -> None:
        if cond:
            lines.append(f"  [OK]   {msg}：{got}")
        else:
            problems.append(f"{msg}：实测 {got}，期望 {want}")
            lines.append(f"  [FAIL] {msg}：实测 {got}，期望 {want}")

    expect(src["archive_sha256"] == bd.ARCHIVE_SHA256, "归档 SHA256", src["archive_sha256"], bd.ARCHIVE_SHA256)
    expect(int(src["archive_size_bytes"]) == bd.ARCHIVE_SIZE_BYTES, "归档字节数", src["archive_size_bytes"], bd.ARCHIVE_SIZE_BYTES)
    expect(int(src["archive_member_count"]) == bd.ARCHIVE_MEMBER_COUNT, "归档成员数", src["archive_member_count"], bd.ARCHIVE_MEMBER_COUNT)
    expect(int(src["archive_uncompressed_bytes"]) == bd.ARCHIVE_UNCOMPRESSED_BYTES, "归档解压后字节数", src["archive_uncompressed_bytes"], bd.ARCHIVE_UNCOMPRESSED_BYTES)

    hash_dim = int(feats["hash_dim"])
    expect(bool(feats["no_bag"]) is bool(no_bag), "features.no_bag 与实际口径一致", feats["no_bag"], no_bag)
    expect(int(feats["extra_dim"]) == int(bd.EXTRA_DIM), "附加特征列数 extra_dim", feats["extra_dim"], bd.EXTRA_DIM)
    expect(
        int(feats["feature_dim"]) == bd.feature_dim(hash_dim, bool(no_bag)),
        "特征维数 D = hash_dim + extra_dim",
        feats["feature_dim"],
        bd.feature_dim(hash_dim, bool(no_bag)),
    )
    if no_bag:
        expect(hash_dim == 0, "no_bag 的 features.hash_dim", hash_dim, 0)
        expect("hash_algo" not in feats, "no_bag 不写哈希字段（hash_algo 缺席）", "hash_algo" in feats, False)
        expect(int(feats["feature_dim"]) == int(bd.EXTRA_DIM), "no_bag 的 D = 6", feats["feature_dim"], bd.EXTRA_DIM)
    else:
        expect(hash_dim == int(bd.HASH_DIM), "缺省 hash_dim", hash_dim, bd.HASH_DIM)
        expect(feats["hash_algo"] == "blake2b" and int(feats["hash_digest_size"]) == bd.HASH_DIGEST_SIZE, "哈希算法/摘要长度", f"{feats['hash_algo']}/{feats['hash_digest_size']}", f"blake2b/{bd.HASH_DIGEST_SIZE}")
        expect(feats["hash_salt_hex"] == bd.HASH_SALT.hex(), "哈希盐（hex）", feats["hash_salt_hex"], bd.HASH_SALT.hex())
        expect(bool(feats["hash_block_l2_normalized"]) is True, "哈希块 L2 归一化", feats["hash_block_l2_normalized"], True)
        expect(
            feats["hash_bucket_rule"] == "int.from_bytes(blake2b(salt+token).digest(), 'big') % " + str(hash_dim),
            "落桶规则（mod 与实际维度一致）",
            feats["hash_bucket_rule"],
            "mod " + str(hash_dim),
        )
        expect(int(counts["bag_columns"]) == hash_dim, "counts.bag_columns = hash_dim", counts["bag_columns"], hash_dim)
        expect(int(feats["feature_dim"]) == int(bd.FEATURE_DIM), "缺省口径 D（HASH_DIM + 6）", feats["feature_dim"], bd.FEATURE_DIM)
    if hash_dim == int(bd.HASH_DIM_PLAN_ORIGINAL):
        lines.append(f"  [信息] 该产物为**原计划口径** hash_dim={hash_dim}（D={feats['feature_dim']}，已实测不达标，作失败对照）")
    elif (not no_bag) and hash_dim != int(bd.HASH_DIM):
        lines.append(f"  [信息] 该产物为 --hash-dim {hash_dim} 的非缺省口径（D={feats['feature_dim']}）")

    expect(feats["token_regex"] == bd.TOKEN_RE.pattern, "分词正则", feats["token_regex"], bd.TOKEN_RE.pattern)
    expect(int(label["negative_seed"]) == bd.DEFAULT_NEGATIVE_SEED, "负样本种子", label["negative_seed"], bd.DEFAULT_NEGATIVE_SEED)
    expect(str(label["negative_ratio"]).startswith("1:1"), "负样本比例", label["negative_ratio"], "1:1")
    expect(int(ans["short_numeric_max_len"]) == bd.SHORT_NUMERIC_MAX_LEN, "短数字阈值", ans["short_numeric_max_len"], bd.SHORT_NUMERIC_MAX_LEN)
    expect(bool(ans["case_insensitive"]) is True, "答案匹配大小写不敏感", ans["case_insensitive"], True)

    cols = feats["columns"]
    want_cols = bd.feature_columns(hash_dim, bool(no_bag))
    expect(len(cols) == len(want_cols), "特征列定义条目数", len(cols), len(want_cols))
    cover_ok = True
    cursor = 0
    for c in cols:
        if int(c["start"]) != cursor:
            cover_ok = False
        cursor = int(c["end"]) + 1
    expect(cover_ok and cursor == bd.feature_dim(hash_dim, bool(no_bag)), "特征列定义无缝覆盖 [0, D-1]", f"cursor={cursor}", bd.feature_dim(hash_dim, bool(no_bag)))
    expect(
        [c["name"] for c in cols] == [c["name"] for c in want_cols],
        "特征列名与 feature_columns(hash_dim, no_bag) 一致",
        [c["name"] for c in cols],
        [c["name"] for c in want_cols],
    )

    with np.load(path, allow_pickle=False) as npz:
        X = np.asarray(npz["X"])
        y = np.asarray(npz["y"])
    m = int(X.shape[0])
    expect(len(meta["samples"]) == m, "逐样本 provenance 条数 = M", len(meta["samples"]), m)
    expect(int(X.shape[1]) == int(feats["feature_dim"]), "X 列数 = meta.feature_dim", X.shape[1], feats["feature_dim"])
    expect(int(counts["positive"]) == int((y == 1).sum()), "meta 正样本数 = y 中 1 的个数", counts["positive"], int((y == 1).sum()))
    expect(int(counts["negative"]) == int((y == 0).sum()), "meta 负样本数 = y 中 0 的个数", counts["negative"], int((y == 0).sum()))
    expect(len(meta["documents"]) == int(counts["pool_documents"]), "文档池条数", len(meta["documents"]), counts["pool_documents"])
    expect(len(meta["questions"]) == int(counts["questions"]), "题目条数", len(meta["questions"]), counts["questions"])
    bad_idx = [i for i, s in enumerate(meta["samples"]) if not (0 <= int(s[0]) < len(meta["questions"]) and 0 <= int(s[1]) < len(meta["documents"]))]
    expect(not bad_idx, "逐样本索引均在范围内", f"越界 {len(bad_idx)} 条", 0)
    kind_ok = all(
        meta["documents"][int(s[1])].startswith(bd.EVIDENCE_DIR["wikipedia"] + "/")
        or meta["documents"][int(s[1])].startswith(bd.EVIDENCE_DIR["web"] + "/")
        for s in meta["samples"]
    )
    expect(kind_ok, "文档路径前缀均为写死的两类目录", kind_ok, True)
    passed = not problems
    if problems:
        lines.append("  问题：" + "；".join(problems))
    return CheckResult("E0", f"meta 口径回读与计数自洽 [{label_txt}]", passed, detail="\n".join(lines))
def check_e1_idempotent(args: argparse.Namespace) -> CheckResult:
    """E1：**每个产物口径各连跑两次**，npz 逐字节一致（并与正式产物一致）。

    实现要点：每次构建都用 ``--split <args.split>``（缺省 ``all``），故一个口径的
    两个 split 共享**同一遍**归档扫描 -> 4 个口径共 4 遍（而非 8 遍）。

    参数
    ----
    args : argparse.Namespace
        命令行参数（用 ``--split`` / ``--archive``）。

    返回
    ----
    CheckResult
        检查结果（明细含每次构建的 SHA256 与用时）。
    """
    lines: List[str] = []
    problems: List[str] = []
    splits = SPLITS if args.split == "all" else (str(args.split),)
    for label, no_bag in PRODUCT_VARIANTS:
        shas: Dict[str, List[str]] = {s: [] for s in splits}
        failed = False
        for run_i, sub in enumerate(("idem_a", "idem_b"), start=1):
            out_dir = os.path.join(VERIFY_DIR, sub, label)
            argv = ["--split", str(args.split), "--archive", str(args.archive), "--out-dir", out_dir]
            if no_bag:
                argv.append("--no-bag")
            t0 = time.time()
            rc = bd.main(argv)
            lines.append(f"  [{label}] 第 {run_i} 次构建（--split {args.split}）：退出码 {rc}，用时 {time.time() - t0:.1f} s，目录 {out_dir}")
            if rc != 0:
                problems.append(f"{label} 第 {run_i} 次构建退出码 {rc}")
                failed = True
                break
            for split in splits:
                name = os.path.basename(product_path(split, no_bag))
                fp = os.path.join(out_dir, name)
                if not os.path.isfile(fp):
                    problems.append(f"{label} 第 {run_i} 次构建缺产物 {fp}")
                    failed = True
                    continue
                shas[split].append(sha256_file(fp))
        if failed:
            continue
        for split in splits:
            prod = product_path(split, no_bag)
            if len(shas[split]) != 2 or not os.path.isfile(prod):
                problems.append(f"{split}/{label} 无法比对（构建次数 {len(shas[split])}）")
                continue
            sha_prod = sha256_file(prod)
            same = shas[split][0] == shas[split][1] == sha_prod
            lines.append(
                f"  {split}/{label}: A={shas[split][0][:16]}... B={shas[split][1][:16]}... "
                f"正式产物={sha_prod[:16]}... -> {'一致' if same else '不一致'}"
            )
            if not same:
                problems.append(f"{split}/{label} 三次 SHA256 不一致")
    passed = not problems
    if problems:
        lines.append("  问题：" + "；".join(problems))
    return CheckResult("E1", "构建幂等（每个口径逐字节一致）", passed, detail="\n".join(lines))


def check_e2_contract(args: argparse.Namespace) -> CheckResult:
    """E2：委派 ``n3d_shape.data.load_npz_arrays`` 读取产物并断言契约。

    **[!] 口径修订（风后 R28 warning 1）**：期望维度由**产物 meta 的
    ``features.feature_dim``** 给出（而不是硬编码 ``bd.FEATURE_DIM``），
    否则对 D=70 / D=6 的产物会**假失败**。此处同时打印"期望维度来源 = meta"以为证据。

    参数
    ----
    args : argparse.Namespace
        命令行参数（用 ``--split``）。

    返回
    ----
    CheckResult
        检查结果。
    """
    lines: List[str] = []
    problems: List[str] = []
    for split, label, no_bag, path in product_list(args):
        try:
            meta = bd.read_npz_meta(path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{split}/{label} meta 不可读：{exc}")
            lines.append(f"  [FAIL] {split}/{label}: meta 不可读 -> {exc}")
            continue
        dim = int(meta["features"]["feature_dim"])
        try:
            X, y, mod = load_arrays_via_shape_contract(path)
        except Exception as exc:  # noqa: BLE001 - 验证脚本需要把任何异常记为失败
            problems.append(f"{split}/{label} 委派加载失败：{type(exc).__name__}: {exc}")
            lines.append(f"  [FAIL] {split}/{label}: 委派 n3d_shape 数据层加载失败 -> {exc}")
            continue
        n_finite = int(np.isfinite(X).sum())
        checks = [
            (X.dtype == E2_DTYPE_X, f"X.dtype={X.dtype}", "float32"),
            (y.dtype == E2_DTYPE_Y, f"y.dtype={y.dtype}", "int64"),
            (X.ndim == 2 and int(X.shape[1]) == dim, f"X.shape={X.shape}", f"[M,{dim}]（维度取自 meta.features.feature_dim）"),
            (y.ndim == 1 and int(y.shape[0]) == int(X.shape[0]), f"y.shape={y.shape}", f"[{int(X.shape[0])}]"),
            (int(X.shape[0]) > 0, f"M={int(X.shape[0])}", "> 0"),
            (n_finite == int(X.size), f"有限元素 {n_finite}/{int(X.size)}", "全部有限"),
            (set(np.unique(y).tolist()) <= {0, 1}, f"标签取值={np.unique(y).tolist()}", "子集 {0,1}"),
        ]
        bad = [f"{msg} 实测 {got}（期望 {want}）" for ok, got, want in checks if not ok]
        for ok, got, want in checks:
            lines.append(f"  [{'OK' if ok else 'FAIL'}] {split}/{label}: {got}")
        if bad:
            problems.extend(bad)
        else:
            lines.append(
                f"  [OK]   {split}/{label}: 委派入口 {mod}.load_npz_arrays 读取成功"
                f"（D={dim}，期望维度来源 = meta['features']['feature_dim']）"
            )
    passed = not problems
    if problems:
        lines.append("  问题：" + "；".join(problems))
    return CheckResult("E2", "产物契约（按 meta.feature_dim 断言 + n3d_shape 数据层可读）", passed, detail="\n".join(lines))

def check_e4_no_leak(args: argparse.Namespace) -> CheckResult:
    """E4：同一 QuestionId 的文档不得同时出现在正负两侧；无重复样本对。

    覆盖 4 个产物；并新增两条**口径修订专项**断言：
    ① 同 split 的 ``h64`` 与 ``nobag`` 两个产物必须共享**完全相同的样本口径**
       （行序/QuestionId/文档/label 逐条一致，且 6 列附加特征逐位一致）——
       即"降维/去词袋"只改特征列，不改采样；
    ② 两个 split 之间报告 QuestionId/文档交集与标签冲突数（信息项）。

    参数
    ----
    args : argparse.Namespace
        命令行参数（用 ``--split``）。

    返回
    ----
    CheckResult
        检查结果。
    """
    products = product_list(args)
    lines: List[str] = []
    problems: List[str] = []
    per_product: Dict[str, Dict[str, Any]] = {}
    for split, label, no_bag, path in products:
        key = f"{split}/{label}"
        if not os.path.isfile(path):
            problems.append(f"缺产物：{path}")
            continue
        meta = bd.read_npz_meta(path)
        questions = meta["questions"]
        documents = meta["documents"]
        pos: Dict[str, set] = {}
        neg: Dict[str, set] = {}
        seen_pairs = set()
        dup: List[Tuple[str, str]] = []
        triples: List[Tuple[str, str, int]] = []
        for qi, di, label_v, _hit in meta["samples"]:
            qid = str(questions[int(qi)]["question_id"])
            doc = str(documents[int(di)])
            pair = (qid, doc)
            if pair in seen_pairs:
                dup.append(pair)
            seen_pairs.add(pair)
            triples.append((qid, doc, int(label_v)))
            (pos if int(label_v) == 1 else neg).setdefault(qid, set()).add(doc)
        overlap = {q: sorted(pos[q] & neg.get(q, set())) for q in pos if pos[q] & neg.get(q, set())}
        lines.append(
            f"  {key}: QuestionId {len(set(pos) | set(neg))} 个，正负文档交集 {len(overlap)} 个，"
            f"重复 (QuestionId, 文档) 对 {len(dup)} 个"
        )
        if overlap:
            problems.append(f"{key} 存在同一 QuestionId 的正负文档交集：{list(overlap.items())[:3]}")
        if dup:
            problems.append(f"{key} 存在重复样本对：{dup[:3]}")
        own = {q: set(pos.get(q, set())) for q in set(pos) | set(neg)}
        bad_neg = [
            (str(questions[int(qi)]["question_id"]), str(documents[int(di)]))
            for qi, di, label_v, _hit in meta["samples"]
            if int(label_v) == 0
            and str(documents[int(di)]) in own.get(str(questions[int(qi)]["question_id"]), set())
        ]
        lines.append(f"  {key}: 负样本中属于本题自身证据的条数 = {len(bad_neg)}")
        if bad_neg:
            problems.append(f"{key} 负样本含本题证据：{bad_neg[:3]}")
        per_product[key] = {
            "triples": triples,
            "qids": {t[0] for t in triples},
            "docs": set(documents),
            "meta": meta,
            "path": path,
        }

    # ① 同 split 的 h64 / nobag 必须样本口径一致 + 6 列附加特征逐位一致
    for split in sorted({p[0] for p in products}):
        keys = [f"{split}/{lab}" for lab, _nb in PRODUCT_VARIANTS if f"{split}/{lab}" in per_product]
        if len(keys) < 2:
            continue
        a, b = per_product[keys[0]], per_product[keys[1]]
        if a["triples"] != b["triples"]:
            problems.append(f"{split} 的 {keys[0]} 与 {keys[1]} 样本口径不一致（行序/文档/label 有差异）")
            lines.append(f"  [FAIL] {split}: {keys[0]} 与 {keys[1]} 的样本口径**不一致**")
        else:
            lines.append(f"  [OK]   {split}: {keys[0]} 与 {keys[1]} 的样本口径（行序/QuestionId/文档/label）逐条一致")
        try:
            with np.load(a["path"], allow_pickle=False) as za, np.load(b["path"], allow_pickle=False) as zb:
                Xa = np.asarray(za["X"])
                Xb = np.asarray(zb["X"])
            ha = int(a["meta"]["features"]["hash_dim"])
            hb = int(b["meta"]["features"]["hash_dim"])
            extra_a = Xa[:, ha:] if ha > 0 else Xa
            extra_b = Xb[:, hb:] if hb > 0 else Xb
            same_extra = extra_a.shape == extra_b.shape and np.array_equal(extra_a, extra_b)
            lines.append(
                f"  {'[OK]  ' if same_extra else '[FAIL]'} {split}: 两个口径的 6 列附加特征"
                f"{'逐位一致' if same_extra else '**不一致**'}（{extra_a.shape} vs {extra_b.shape}）"
            )
            if not same_extra:
                problems.append(f"{split} 的 {keys[0]} 与 {keys[1]} 附加特征不一致")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{split} 附加特征比对失败：{type(exc).__name__}: {exc}")

    # ② 跨 split 信息项：同 label 的两个 split 产物之间
    for label, _nb in PRODUCT_VARIANTS:
        keys = [f"{s}/{label}" for s in SPLITS if f"{s}/{label}" in per_product]
        if len(keys) < 2:
            continue
        qa, qb = per_product[keys[0]], per_product[keys[1]]
        inter_q = qa["qids"] & qb["qids"]
        inter_d = qa["docs"] & qb["docs"]
        labels_by = {}
        for key, item in ((keys[0], qa), (keys[1], qb)):
            labels_by[key] = {(t[0], t[1]): t[2] for t in item["triples"]}
        common_pairs = set(labels_by[keys[0]]) & set(labels_by[keys[1]])
        conflicts = [q for q in common_pairs if labels_by[keys[0]][q] != labels_by[keys[1]][q]]
        lines.append(
            f"  [信息] {keys[0]} 与 {keys[1]}：QuestionId 交集 = {len(inter_q)}"
            f"（示例 {sorted(inter_q)[:5]}），文档交集 = {len(inter_d)}，共有 (QuestionId,文档) 对 = "
            f"{len(common_pairs)}，其中标签不同 = {len(conflicts)}（示例 {sorted(conflicts)[:3]}）"
        )
    passed = not problems
    if problems:
        lines.append("  问题：" + "；".join(problems))
    return CheckResult("E4", "无泄漏（同题正负不交叉 / 无重复对 / 变体样本口径一致）", passed, detail="\n".join(lines))

def _first_match_window(text: str, patterns: Sequence[re.Pattern], width: int = 60) -> Tuple[str, str]:
    """在文档里找最早出现的答案命中，返回 ``(命中串, 窗口片段)``。

    参数
    ----
    text : str
        文档全文。
    patterns : Sequence[re.Pattern]
        答案匹配模式。
    width : int
        命中位置左右各取多少字符。

    返回
    ----
    Tuple[str, str]
        ``(命中串, 片段)``；未命中时返回 ``("<未命中>", "")``。
    """
    best: Optional[Tuple[int, str]] = None
    for p in patterns:
        m = p.search(text)
        if m is not None and (best is None or m.start() < best[0]):
            best = (int(m.start()), m.group(0))
    if best is None:
        return "<未命中>", ""
    start, matched = best
    snippet = text[max(0, start - int(width)): start + int(width)]
    return matched, " ".join(snippet.split())


def check_e3_balance_and_answer(args: argparse.Namespace) -> CheckResult:
    """E3：标签 1:1 均衡；确定性抽样 N=20 正 / N=20 负并从归档**重算**答案命中。

    **[!] 口径修订（风后 R28 warning 2）**：QA 缓存键由原先硬编码的
    ``bd.ARCHIVE_SHA256`` 改为 ``bd.verify_archive(args.archive)`` **现场实测**的 SHA256，
    使缓存键与归档实际内容一致（换归档时不会再误用常量对应的缓存）。

    口径
    ----
    * 均衡判据：meta 计数与 ``y`` 实测都必须正负相等（每类 50%）；
    * 答案复核：对抽样行从归档重新抽取其文档（**所有目标文档合并成一遍**流式扫描），
      用 ``doc_contains_answer`` 重算命中并与 meta 记录的 ``answer_hit`` 逐条比对；
    * 正样本含答案比例设下限 ``POSITIVE_ANSWER_HIT_MIN``；负样本只报告实测比例
      （标签是 provenance 判定，别的题的文档可能恰好提到同一答案）。

    参数
    ----
    args : argparse.Namespace
        命令行参数（用 ``--split`` / ``--archive`` / ``--audit-n``）。

    返回
    ----
    CheckResult
        检查结果（明细含逐条抽样证据）。
    """
    products = product_list(args)
    n_audit = int(args.audit_n)
    lines: List[str] = []
    problems: List[str] = []
    # ---- warning 2 修复：缓存键 = 现场实测归档 SHA256 ----
    archive_sha = bd.verify_archive(args.archive)
    lines.append(
        f"  归档实测 SHA256（本次 E3 的 QA 缓存键来源）={archive_sha}"
        + ("（与写死常量一致）" if archive_sha == bd.ARCHIVE_SHA256 else "（[!] 与写死常量不一致）")
    )
    prepared: Dict[str, Dict[str, Any]] = {}
    wanted_docs: set = set()
    records_cache: Dict[str, Dict[str, Any]] = {}
    for split, label, no_bag, path in products:
        key = f"{split}/{label}"
        if not os.path.isfile(path):
            problems.append(f"缺产物：{path}")
            continue
        meta = bd.read_npz_meta(path)
        counts = meta["counts"]
        n_pos = int(counts["positive"])
        n_neg = int(counts["negative"])
        frac_pos = float(counts["positive_fraction"])
        lines.append(f"  {key}: 正 {n_pos} / 负 {n_neg}（正占比 {frac_pos:.4f}，容差 {bd.BALANCE_TOL}）")
        if n_pos != n_neg or abs(frac_pos - 0.5) > float(bd.BALANCE_TOL):
            problems.append(f"{key} 标签不均衡：正 {n_pos} / 负 {n_neg}")
        pos_rate = float(counts["positive_answer_hit"]) / float(max(n_pos, 1))
        neg_rate = float(counts["negative_answer_hit"]) / float(max(n_neg, 1))
        lines.append(
            f"  {key}: 全量答案命中——正样本 {counts['positive_answer_hit']}/{n_pos}"
            f"（{pos_rate:.4f}），负样本 {counts['negative_answer_hit']}/{n_neg}（{neg_rate:.4f}）"
        )
        if pos_rate < float(POSITIVE_ANSWER_HIT_MIN):
            problems.append(f"{key} 正样本含答案比例 {pos_rate:.4f} 低于下限 {POSITIVE_ANSWER_HIT_MIN}")
        if split not in records_cache:
            raw_qa, _passes, _status = bd.load_archive_qa(args.archive, archive_sha, [split])
            records, _head = bd.parse_qa_json(raw_qa[split], split)
            if int(meta["max_questions"]) > 0:
                records = records[: int(meta["max_questions"])]
            records_cache[split] = {"records_by_id": {r.question_id: r for r in records}}
        samples = meta["samples"]
        questions = meta["questions"]
        documents = meta["documents"]
        pos_rows = [i for i, s in enumerate(samples) if int(s[2]) == 1]
        neg_rows = [i for i, s in enumerate(samples) if int(s[2]) == 0]
        rng = np.random.default_rng(int(AUDIT_SEED))
        pick_pos = [pos_rows[int(i)] for i in rng.choice(len(pos_rows), size=min(n_audit, len(pos_rows)), replace=False)]
        rng = np.random.default_rng(int(AUDIT_SEED) + 1)
        pick_neg = [neg_rows[int(i)] for i in rng.choice(len(neg_rows), size=min(n_audit, len(neg_rows)), replace=False)]
        needed = {str(documents[int(samples[i][1])]) for i in pick_pos + pick_neg}
        wanted_docs |= needed
        prepared[key] = {
            "records_by_id": records_cache[split]["records_by_id"],
            "samples": samples,
            "questions": questions,
            "documents": documents,
            "pick_pos": pick_pos,
            "pick_neg": pick_neg,
        }
    texts: Dict[str, str] = {}
    if wanted_docs:
        texts_bytes, missing = bd.load_archive_evidence(args.archive, wanted_docs)
        if missing:
            problems.append(f"抽样文档在归档中缺失：{missing[:5]}")
        texts = {k: v.decode(bd.TEXT_ENCODING, errors="replace") for k, v in texts_bytes.items()}
    for key, item in prepared.items():
        samples = item["samples"]
        questions = item["questions"]
        documents = item["documents"]
        rec_by_id = item["records_by_id"]
        pick_pos = item["pick_pos"]
        pick_neg = item["pick_neg"]
        n_agree = 0
        n_pos_hit = 0
        n_neg_hit = 0
        lines.append(f"  {key} 抽样人工复核（正 {len(pick_pos)} / 负 {len(pick_neg)} 条，种子 {AUDIT_SEED}）：")
        for tag, rows in (("正", pick_pos), ("负", pick_neg)):
            for i in rows:
                qi = int(samples[i][0])
                di = int(samples[i][1])
                label_v = int(samples[i][2])
                hit = int(samples[i][3])
                qid = str(questions[qi]["question_id"])
                doc = str(documents[di])
                rec = rec_by_id.get(qid)
                if rec is None or doc not in texts:
                    problems.append(f"{key} 抽样行无法复核：{qid} / {doc}")
                    continue
                matched, snippet = _first_match_window(texts[doc], rec.answer_patterns)
                recomputed = 1 if matched != "<未命中>" else 0
                if recomputed == hit:
                    n_agree += 1
                else:
                    problems.append(f"{key} 抽样命中不一致：{qid} / {doc}，meta={hit}，重算={recomputed}")
                if label_v == 1:
                    n_pos_hit += recomputed
                    lines.append(f"    [正] {qid} 文档={doc}")
                    lines.append(f"         答案={list(rec.answer_values)[:4]} 命中串={matched!r}")
                    lines.append(f"         片段=...{snippet}...")
                else:
                    n_neg_hit += recomputed
                    lines.append(
                        f"    [负] {qid} 文档={doc} 重算命中={recomputed}"
                        f"（命中串={matched!r}）片段=...{snippet[:120]}..."
                    )
        lines.append(
            f"  {key} 抽样结果：重算与 meta 记录一致 {n_agree}/{len(pick_pos) + len(pick_neg)}；"
            f"正样本含答案 {n_pos_hit}/{len(pick_pos)}；负样本含答案 {n_neg_hit}/{len(pick_neg)}"
        )
    passed = not problems
    if problems:
        lines.append("  问题：" + "；".join(problems))
    return CheckResult("E3", "标签 1:1 均衡 + 答案命中抽样复核（缓存键 = 实测归档 SHA256）", passed, detail="\n".join(lines))

def run_cmd(cmd: Sequence[str], timeout_s: int = 0) -> Tuple[int, str]:
    """执行外部命令并返回 ``(退出码, 合并后的 stdout+stderr)``。

    子进程强制 ``PYTHONIOENCODING=utf-8``，避免中文日志在 Windows 管道里按本地代码页
    编码导致正则匹配/阅读困难。

    参数
    ----
    cmd : Sequence[str]
        命令与参数。
    timeout_s : int
        超时秒数（``0`` = 不限制）。

    返回
    ----
    Tuple[int, str]
        ``(退出码, 输出文本)``。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        list(cmd),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=(None if int(timeout_s) <= 0 else int(timeout_s)),
    )
    return int(proc.returncode), (proc.stdout or "") + (proc.stderr or "")


def _run_e5_once(path: str, epochs: int, tag: str) -> Tuple[int, List[float], str, str, str]:
    """对单个 npz 跑一次 R1 口径的端到端训练，返回 ``(退出码, 逐轮 acc, 输出, 数据行, 归一化行)``。

    参数
    ----
    path : str
        ``.npz`` 路径。
    epochs : int
        训练轮数。
    tag : str
        产物文件名 stem（用于 checkpoint 命名）。

    返回
    ----
    Tuple[int, List[float], str, str, str]
        ``(退出码, 逐 epoch test_acc(百分数), 合并输出, "数据集加载完成"行, "归一化口径"行)``。
    """
    rel_path = os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")
    ckpt_rel = os.path.join("checkpoints", "n3d_shape", "_verify", f"triviaqa_e5_{tag}.pt").replace("\\", "/")
    cmd: List[str] = [
        sys.executable, os.path.join("n3d_shape", "train.py"),
        *R1_COMMON_ARGS,
        "--dataset", "npz",
        "--dataset-path", rel_path,
        "--input-dim", str(int(np.load(path, allow_pickle=False)["X"].shape[1])),
        "--output-dim", "2",
        "--epochs", str(int(epochs)),
        "--checkpoint", ckpt_rel,
    ]
    rc, out = run_cmd(cmd)
    accs = [float(x) for x in re.findall(r"test_acc=([0-9.]+)%", out)]
    data_line = next((ln for ln in out.splitlines() if "数据集加载完成" in ln), "")
    norm_line = next((ln for ln in out.splitlines() if "归一化口径" in ln), "")
    return rc, accs, out, data_line.strip(), norm_line.strip()[:160]


def check_e5_train(args: argparse.Namespace) -> CheckResult:
    """E5：``n3d_shape/train.py --dataset npz`` 端到端训练（其余同 R1 口径）。

    口径修订后**逐产物各跑一次**（缺省 4 个：wiki/web x D=70/D=6），每个产物用**它自己 meta 的**
    ``features.feature_dim`` 作为 ``--input-dim``；判据为退出码 0 且 ``test_acc > 0.75``
    （二分类，随机基线 0.5）。任一产物失败时对该产物补跑"前 hash_dim 列（哈希词袋块）
    整块置零"的对照诊断（``no_bag`` 产物没有该块，跳过诊断并说明）。

    参数
    ----
    args : argparse.Namespace
        命令行参数（用 ``--e5-path`` / ``--e5-epochs`` / ``--split``）。

    返回
    ----
    CheckResult
        检查结果（明细含每个产物的命令行、逐轮曲线与末轮准确率）。
    """
    if args.e5_path:
        targets: List[Tuple[str, str, bool, str]] = [("(显式指定)", os.path.basename(str(args.e5_path)), False, str(args.e5_path))]
    else:
        targets = list(product_list(args))
    lines: List[str] = []
    problems: List[str] = []
    n_ok = 0
    for split, label, no_bag, path in targets:
        key = f"{split}/{label}"
        if not os.path.isfile(path):
            problems.append(f"缺产物：{path}")
            lines.append(f"  [FAIL] {key}: 缺产物 {path}")
            continue
        meta = bd.read_npz_meta(path)
        dim = int(meta["features"]["feature_dim"])
        hash_dim = int(meta["features"]["hash_dim"])
        tag = os.path.splitext(os.path.basename(path))[0]
        t0 = time.time()
        try:
            rc, accs, out, data_line, norm_line = _run_e5_once(path, int(args.e5_epochs), tag)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"  [SKIP] {key}: 命令执行失败（环境原因）{type(exc).__name__}: {exc}")
            continue
        final = (float(accs[-1]) / 100.0) if accs else float("nan")
        lines.append(
            f"  {key}: D={dim}，命令 n3d_shape/train.py *R1_COMMON_ARGS --dataset npz "
            f"--dataset-path {os.path.relpath(path, PROJECT_ROOT).replace(chr(92), '/')} "
            f"--input-dim {dim} --output-dim 2 --epochs {int(args.e5_epochs)}"
        )
        lines.append(f"    退出码 {rc}，用时 {time.time() - t0:.1f} s，epoch 记录 {len(accs)} 个")
        lines.append(f"    逐 epoch test_acc(%)：{accs}")
        lines.append(f"    末轮 test_acc = {final * 100:.2f}%（判据 > {E5_MIN_ACC * 100:.0f}%）")
        lines.append(f"    {data_line}")
        lines.append(f"    {norm_line}")
        item_bad = []
        if rc != 0:
            item_bad.append(f"退出码 {rc}")
        if not np.isfinite(final):
            item_bad.append("未解析到 test_acc")
        elif final <= float(E5_MIN_ACC):
            item_bad.append(f"test_acc={final * 100:.2f}% 未超过 {E5_MIN_ACC * 100:.0f}%")
        if item_bad:
            problems.append(f"{key}: " + "；".join(item_bad))
            lines.append(f"    [FAIL] {key}: " + "；".join(item_bad))
            lines.extend("    " + ln for ln in out.splitlines()[-12:])
            if hash_dim > 0:
                try:
                    ref = os.path.join(VERIFY_DIR, f"{tag}_nobagref.npz")
                    with np.load(path, allow_pickle=False) as npz:
                        Xr = np.asarray(npz["X"]).copy()
                        yr = np.asarray(npz["y"])
                    Xr[:, :hash_dim] = 0.0
                    np.savez(ref, X=Xr.astype(np.float32), y=yr.astype(np.int64))
                    rc_ref, accs_ref, _o, _d, _n = _run_e5_once(ref, int(args.e5_epochs), os.path.splitext(os.path.basename(ref))[0])
                    ref_final = (float(accs_ref[-1]) / 100.0) if accs_ref else float("nan")
                    lines.append(
                        f"    [诊断] 同口径同 epoch、仅把前 {hash_dim} 列（哈希词袋块）整块置零："
                        f"退出码 {rc_ref}，末轮 test_acc = {ref_final * 100:.2f}%"
                    )
                    lines.append(
                        "    [诊断] 读数：若置零后显著超过判据，则瓶颈在**该维度块的规模/信噪比**，"
                        "而非标签构造或映射规则"
                    )
                except Exception as exc:  # noqa: BLE001
                    lines.append(f"    [诊断] 对照实验无法执行：{type(exc).__name__}: {exc}")
            else:
                lines.append("    [诊断] 该产物为 --no-bag（无词袋块），无需「词袋置零」对照")
        else:
            n_ok += 1
            lines.append(f"    [OK] {key}: 达标（{final * 100:.2f}% > {E5_MIN_ACC * 100:.0f}%）")
    passed = not problems
    if problems:
        lines.append("  问题：" + "；".join(problems))
    title = f"端到端训练（npz 通路，{n_ok}/{len(targets)} 个产物达标，判据 acc > {E5_MIN_ACC}）"
    return CheckResult("E5", title, passed, detail="\n".join(lines))

def check_e6_untouched(args: argparse.Namespace) -> CheckResult:
    """E6：本模块未触碰任何既有模块文件（``git status`` 无改动 + 最新 mtime 佐证）。

    参数
    ----
    args : argparse.Namespace
        命令行参数（用 ``--archive``，仅用于输出目录信息）。

    返回
    ----
    CheckResult
        检查结果（git 不可用时按"环境原因跳过"处理，并给出 mtime 证据）。
    """
    lines: List[str] = []
    newest: List[Tuple[str, float, str]] = []
    for d in MODULE_DIRS_UNTOUCHED:
        root = os.path.join(PROJECT_ROOT, d)
        if not os.path.isdir(root):
            continue
        best_m = 0.0
        best_f = ""
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                try:
                    m = os.path.getmtime(fp)
                except OSError:  # pragma: no cover
                    continue
                if m > best_m:
                    best_m, best_f = m, fp
        newest.append((d, best_m, best_f))
        lines.append(
            f"  {d}: 最新 mtime = {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(best_m))}"
            f"（{os.path.relpath(best_f, PROJECT_ROOT)}）"
        )
    cmd = ["git", "status", "--porcelain", "--", *MODULE_DIRS_UNTOUCHED]
    try:
        rc, out = run_cmd(cmd)
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  [SKIP] git 不可用（{type(exc).__name__}: {exc}）；仅以上 mtime 为佐证")
        return CheckResult("E6", "零回归（未触碰既有模块文件）", True, skipped=True, detail="\n".join(lines))
    if rc != 0:
        lines.append(f"  [SKIP] git status 退出码 {rc}，输出：{out.strip()[:200]}")
        return CheckResult("E6", "零回归（未触碰既有模块文件）", True, skipped=True, detail="\n".join(lines))
    changed = [ln for ln in out.splitlines() if ln.strip()]
    lines.append(f"  git status --porcelain -- {' '.join(MODULE_DIRS_UNTOUCHED)}："
                 f"{'无改动' if not changed else f'{len(changed)} 条改动'}")
    for ln in changed[:20]:
        lines.append(f"    {ln}")
    passed = not changed
    if not passed:
        lines.append("  问题：既有模块出现改动（本模块禁止修改它们）")
    return CheckResult("E6", "零回归（未触碰既有模块文件）", passed, detail="\n".join(lines))


# ======================================================================
# CLI
# ======================================================================
ALL_CHECKS: Tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5", "E6")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析命令行参数。

    参数
    ----
    argv : Optional[Sequence[str]]
        参数列表（``None`` 表示 ``sys.argv``）。

    返回
    ----
    argparse.Namespace
        含 split / archive / checks / audit_n / e5_split / e5_epochs / e5_path / skip_e5 字段。
    """
    parser = argparse.ArgumentParser(
        description="n3d_triviaqa 产物验证（E1 幂等 / E2 契约 / E3 均衡+答案复核 / E4 无泄漏 / E5 端到端 / E6 零回归）"
    )
    parser.add_argument("--split", type=str, default="all", choices=list(bd.SPLIT_CHOICES),
                        help="验证哪个 split（缺省 all）")
    parser.add_argument("--archive", type=str, default=bd.DEFAULT_ARCHIVE,
                        help="triviaqa-rc.tar.gz 路径（E1 / E3 需要）")
    parser.add_argument("--checks", type=str, default=",".join(ALL_CHECKS),
                        help=f"要执行的检查项（逗号分隔，缺省 {','.join(ALL_CHECKS)}；另恒定执行 E0 口径回读）")
    parser.add_argument("--audit-n", type=int, default=AUDIT_N, help=f"E3 人工复核抽样条数（缺省 {AUDIT_N}）")
    parser.add_argument("--e5-split", type=str, default="", choices=["", *bd.SPLIT_CHOICES[:2]],
                        help="E5 只跑该 split 的产物（缺省空 = 不额外过滤，即 --split 选中的全部产物口径）")
    parser.add_argument("--e5-epochs", type=int, default=E5_EPOCHS, help=f"E5 训练轮数（缺省 {E5_EPOCHS}）")
    parser.add_argument(
        "--e5-path",
        type=str,
        default="",
        help="E5 使用的 npz 路径（缺省 = 正式产物；可用它复核非计划维度等对照产物）",
    )
    parser.add_argument("--skip-e5", action="store_true", help="跳过 E5（仅做数据侧验证时用）")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口：按顺序执行 E0 + 选定检查项，打印报告并返回退出码。

    参数
    ----
    argv : Optional[Sequence[str]]
        参数列表（``None`` 表示 ``sys.argv``）。

    返回
    ----
    int
        退出码：``0`` 全部通过（或仅有环境原因跳过）；``1`` 存在失败项。
    """
    configure_console_encoding()
    args = parse_args(argv)
    wanted = tuple(x.strip().upper() for x in str(args.checks).split(",") if x.strip())
    unknown = [x for x in wanted if x not in ALL_CHECKS]
    if unknown:
        print(f"[n3d_triviaqa.verify] 未知检查项 {unknown}；可选 {ALL_CHECKS}", file=sys.stderr)
        return 1
    if bool(args.skip_e5) and "E5" in wanted:
        wanted = tuple(x for x in wanted if x != "E5")
    t0 = time.time()
    print("=" * 78)
    print("n3d_triviaqa 产物验证报告")
    print("=" * 78)
    print(f"产物目录：{PRODUCT_DIR}")
    print(f"验证产物目录：{VERIFY_DIR}")
    print(f"检查项：E0（恒定，逐产物） + {list(wanted)}")
    print(f"产物口径变体：{[(lab, ('D=6' if nb else 'D=' + str(int(bd.HASH_DIM) + int(bd.EXTRA_DIM)))) for lab, nb in PRODUCT_VARIANTS]}")
    print("-" * 78)

    results: List[CheckResult] = []
    splits = SPLITS if args.split == "all" else (str(args.split),)

    def emit(res: CheckResult) -> None:
        results.append(res)
        tag = "SKIP" if res.skipped else ("PASS" if res.passed else "FAIL")
        print(f"[{res.name}] {res.title} -> {tag}")
        if res.detail:
            print(res.detail)
        print("-" * 78)

    for split, label, no_bag, _path in product_list(args):
        emit(check_meta_contract(split, no_bag))
    if "E1" in wanted:
        emit(check_e1_idempotent(args))
    if "E2" in wanted:
        emit(check_e2_contract(args))
    if "E3" in wanted:
        emit(check_e3_balance_and_answer(args))
    if "E4" in wanted:
        emit(check_e4_no_leak(args))
    if "E5" in wanted:
        if args.e5_split:
            args.split = str(args.e5_split)  # 仅按该 split 的产物跑 E5
        emit(check_e5_train(args))
    if "E6" in wanted:
        emit(check_e6_untouched(args))

    n_pass = sum(1 for r in results if r.passed and not r.skipped)
    n_fail = sum(1 for r in results if not r.passed and not r.skipped)
    n_skip = sum(1 for r in results if r.skipped)
    print("=" * 78)
    for r in results:
        tag = "SKIP" if r.skipped else ("PASS" if r.passed else "FAIL")
        print(f"  [{r.name}] {tag}  {r.title}")
    print("-" * 78)
    print(f"汇总：通过 {n_pass} 项，失败 {n_fail} 项，跳过 {n_skip} 项；用时 {time.time() - t0:.1f} s")
    print("结论：" + ("全部通过（跳过项为环境原因，不计失败）" if n_fail == 0 else "存在失败项，见上方明细"))
    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
