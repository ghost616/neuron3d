"""n3d_qa.verify_dataset —— 通用 QA 数据集处理模块的产物验证（E0–E8，逐项真实执行、结论可复核）。

定位
----
本模块是**通用 QA 数据集处理模块**（任意问答数据集 -> N3D 数组格式）；验证器当前校验的是
内置的 TriviaQA 参考实现产物（下述 split / 列定义 / 口径均为 TriviaQA 口径），
后续挂入的 QA 数据集 adapter 复用同一套 E0–E8 契约检查。

验证项
------
* **E0 meta 口径回读**（恒定，逐产物）：按**产物自己 meta 的实际生效值**比对
  ``no_bag`` / ``hash_dim`` / ``features``（base/rich）/ ``extra_dim`` / 逐列定义 /
  ``counts`` 自洽；``rich`` 额外校验 IDF 口径（公式、语料文档数 M、词表规模、
  ``counts.extra.rich.column_start`` 与 ``counts.extra.idf.*``）。
* **E1 构建幂等**：同参数连跑两次构建 -> 两个 npz 逐字节一致（SHA256 相同），
  且与正式产物一致（正式产物 = 可复现产物）。
* **E2 产物契约**：委派 ``n3d_shape.data.load_npz_arrays`` 读取，断言
  ``X.dtype == float32`` / ``y.dtype == int64`` / 无 NaN/Inf / 形状 ``[M, D]``
  （``D`` 由产物 meta 的 ``features.feature_dim`` 给出：base 70 / base+no-bag 6 /
  rich 74 / rich+no-bag 10）。
* **E3 标签均衡 + 答案复核**：正负 1:1；确定性抽样 N=20 正 / N=20 负，
  **从归档重新抽取这 40 个文档**（一遍流式扫描）重算答案命中并与 meta 记录的值比对。
* **E4 无泄漏 + 变体一致**：同一 QuestionId 的文档不得同时出现在正负两侧；
  ``(QuestionId, 文档)`` 不得重复；同 split 的 ``h64``/``nobag`` 与 ``base``/``rich``
  必须样本口径逐条一致，且附加特征块逐位一致（base 的 6 列 = rich 附加块前 6 列）。
* **E5 端到端训练**：``python n3d_shape/train.py --dataset npz ...``（其余同 R1 口径），
  断言退出码 0 且 ``test_acc > 0.75``（二分类，随机基线 0.5）。
* **E6 零回归**：``git status --porcelain`` 中 n3d_proto / n3d_sphere / n3d_shape /
  n3d_viz 一律无改动（本模块不触碰任何既有模块文件）。
* **E7 rich 特征判别力分解**（新增）：对每个 verified split 做 5 折逻辑回归 CV，
  逐列给 |Pearson r|、逐子集给 acc/AUC，并按实测给出"rich 是否优于 base"的**结论**
  （无增益则如实报无增益，不粉饰）。
* **E8 无泄漏（新增）**：同 QuestionId 正负交集 = 0、无重复对，且 **verified 与 dev 的
  QuestionId 交集 = 0**（覆盖 base / rich / dev 全部产物；缺产物记 SKIP 不计失败）。

产物组与缺省行为
----------------
``--product-set {base,rich,dev,all}``（缺省 ``base`` = 既有 4 个正式产物）；
``--checks`` 缺省 ``E1,E2,E3,E4,E5,E6``（保持历史行为），E7/E8 需显式加入
（``--checks E7,E8`` 或 ``--checks E1,E2,E3,E4,E5,E6,E7,E8``）；E0 恒定执行。

元信息口径回读
--------------
每项检查都会把产物 meta 里的口径（归档 SHA256、哈希维度与盐、负样本种子、
特征列定义与 IDF 口径、短数字阈值、1:1 比例）与 :mod:`n3d_qa.build_dataset` 的
**写死常量**逐字比对 —— meta 只写不读等于没有约束。
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
# 新增（2026-10-04 扩容）产物的**特征集合变体**：(文件名标签, 特征集合)
PRODUCT_FEATURE_SETS: Tuple[Tuple[str, str], ...] = (("base", "base"), ("rich", "rich"))
# 非 verified 的 dev split（compact 路径产出，只有 rich 口径；base 口径不存在故不校验）
DEV_SPLITS: Tuple[str, ...] = ("wiki-dev", "web-dev")
DEV_FEATURE_SETS: Tuple[Tuple[str, str], ...] = (("rich", "rich"),)
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


def product_path(split: str, no_bag: bool = False, features: str = "base") -> str:
    """产物路径（**口径标签恒定在文件名内**，见 build_dataset.out_name_for）。

    参数
    ----
    split : str
        ``wiki`` / ``web`` / ``wiki-dev`` / ``web-dev``。
    no_bag : bool
        是否取 ``--no-bag`` 版本（base 为 D=6 / rich 为 D=10）。
    features : str
        ``base`` / ``rich``：``base`` 取正式产物目录，``rich`` 取 ``_verify/``（构建侧
        ``resolve_out_dir`` 对非缺省 ``--features`` 的路由口径——见 build_dataset.resolve_out_dir）。

    返回
    ----
    str
        绝对路径，形如 ``..._dev_h64.npz`` / ``..._dev_h64_rich.npz``。
    """
    base_dir = PRODUCT_DIR if str(features) == "base" else VERIFY_DIR
    return os.path.join(
        base_dir,
        bd.out_name_for(split, 0, bd.HASH_DIM, bd.DEFAULT_ARCHIVE, bool(no_bag), str(features)),
    )


def product_list(
    args: argparse.Namespace, product_set: str = "base"
) -> List[Tuple[str, str, bool, str]]:
    """列出本轮要验证的产物列表。

    参数
    ----
    args : argparse.Namespace
        命令行参数（用 ``--split``）。
    product_set : str
        ``base``（缺省，既有 4 个正式产物：wiki/web x (h64, nobag)）/ ``rich``
        （新增 4 个 rich 产物：wiki/web x (h64_rich, nobag_rich)，落 ``_verify/``）/
        ``dev``（2 个非 verified dev 产物：wiki-dev/web-dev x h64_rich）/ ``all``。

    返回
    ----
    List[Tuple[str, str, bool, str]]
        ``[(split, 变体标签, no_bag, 产物绝对路径), ...]``。
    """
    out: List[Tuple[str, str, bool, str]] = []

    def _push(split: str, label: str, no_bag: bool, features: str) -> None:
        out.append((split, label, bool(no_bag), product_path(split, bool(no_bag), features)))

    if product_set in ("base", "all"):
        splits = SPLITS if args.split == "all" else (str(args.split),)
        for split in splits:
            for label, no_bag in PRODUCT_VARIANTS:
                _push(split, label, bool(no_bag), "base")
    if product_set in ("rich", "all"):
        splits = SPLITS if args.split == "all" else (str(args.split),)
        for split in splits:
            for label, no_bag in PRODUCT_VARIANTS:
                _push(split, label + "_rich", bool(no_bag), "rich")
    if product_set in ("dev", "all"):
        for split in DEV_SPLITS:
            for label, feat in DEV_FEATURE_SETS:
                _push(split, f"{label}_{split.replace('-', '_')}", False, feat)
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


def check_meta_contract(
    split: str, no_bag: bool = False, features: str = "base"
) -> CheckResult:
    """E0：产物 meta 与写死常量的**逐字**比对 + 计数自洽（口径回读）。

    ``features`` 段按**实际生效值**校验（``hash_dim`` / ``no_bag`` / ``features`` 特征集合 /
    逐列定义），故 D=70 / D=6 / D=74 / D=10 各版本都按各自 meta 回的维度与列定义比对，
    不做跨口径的硬编码比较。``rich`` 版本额外回读 IDF 语料口径（公式、语料文档数、词表规模）。

    参数
    ----
    split : str
        ``wiki`` / ``web`` / ``wiki-dev`` / ``web-dev``。
    no_bag : bool
        是否校验 ``--no-bag`` 版本。
    features : str
        ``base`` / ``rich``。

    返回
    ----
    CheckResult
        检查结果（明细含各项实测/期望值）。
    """
    path = product_path(split, no_bag, features)
    label_txt = f"{split}/{'nobag' if no_bag else 'h' + str(int(bd.HASH_DIM))}" + (
        "_rich" if features == "rich" else ""
    )
    lines: List[str] = [f"产物 {path}"]
    problems: List[str] = []
    if not os.path.isfile(path):
        return CheckResult(
            "E0",
            f"meta 口径回读 [{label_txt}]",
            True,
            skipped=True,
            detail=f"[SKIP] 缺产物（该口径尚未构建）：{path}",
        )
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
    expect(
        int(feats["extra_dim"]) == bd.extra_dim_for(features),
        "附加特征列数 extra_dim",
        feats["extra_dim"],
        bd.extra_dim_for(features),
    )
    expect(
        int(feats["feature_dim"]) == bd.feature_dim(hash_dim, bool(no_bag), features),
        "特征维数 D = hash_dim + extra_dim",
        feats["feature_dim"],
        bd.feature_dim(hash_dim, bool(no_bag), features),
    )
    want_cols_rich = bd.feature_columns(hash_dim, bool(no_bag), "rich")
    _rich_base = (0 if bool(no_bag) else int(hash_dim)) + int(bd.EXTRA_DIM)
    if features == "rich":
        expect(feats.get("feature_block") == "rich", "features.feature_block", feats.get("feature_block"), "rich")
        expect(
            int(feats.get("base_extra_dim", -1)) == int(bd.EXTRA_DIM),
            "rich 的基础附加列数 base_extra_dim",
            feats.get("base_extra_dim"),
            bd.EXTRA_DIM,
        )
        expect(
            int(feats.get("rich_extra_dim", -1)) == int(bd.RICH_EXTRA_DIM),
            "rich 的追加列数 rich_extra_dim",
            feats.get("rich_extra_dim"),
            bd.RICH_EXTRA_DIM,
        )
        idf_meta = feats.get("idf") or {}
        expect(
            str(idf_meta.get("formula", ""))
            == "idf(token) = log((1 + M) / (1 + df(token))) + " + str(bd.IDF_SMOOTH_OFFSET),
            "IDF 公式",
            idf_meta.get("formula"),
            "log((1 + M) / (1 + df)) + " + str(bd.IDF_SMOOTH_OFFSET),
        )
        expect(
            int(idf_meta.get("M", -1)) == int(counts["pool_documents"]),
            "IDF 语料文档数 M = 文档池大小",
            idf_meta.get("M"),
            counts["pool_documents"],
        )
        extra_meta = counts.get("extra") or {}
        rich_stat = extra_meta.get("rich") or {}
        idf_stat = extra_meta.get("idf") or {}
        col_base = (0 if bool(no_bag) else int(hash_dim)) + int(bd.EXTRA_DIM)
        expect(
            int(rich_stat.get("column_start", -1)) == col_base,
            "rich 列起点 counts.extra.rich.column_start",
            rich_stat.get("column_start"),
            col_base,
        )
        expect(
            [str(x) for x in rich_stat.get("columns", [])]
            == [c["name"] for c in want_cols_rich[-int(bd.RICH_EXTRA_DIM):]],
            "rich 追加列名与 feature_columns(hash_dim, no_bag, rich) 一致",
            rich_stat.get("columns"),
            [c["name"] for c in want_cols_rich[-int(bd.RICH_EXTRA_DIM):]],
        )
        expect(
            int(idf_stat.get("vocab_size") or 0) == int(idf_meta.get("vocab_size") or 0)
            and int(idf_stat.get("vocab_size") or 0) > 0,
            "IDF 词表规模 > 0 且 counts.extra.idf.vocab_size = meta.features.idf.vocab_size",
            idf_stat.get("vocab_size"),
            idf_meta.get("vocab_size"),
        )
        expect(
            int(idf_stat.get("corpus_documents", -1)) == int(counts["pool_documents"]),
            "counts.extra.idf.corpus_documents = 文档池大小",
            idf_stat.get("corpus_documents"),
            counts["pool_documents"],
        )
    elif "feature_block" in feats:
        expect(False, "base 产物不应写 feature_block 字段", feats.get("feature_block"), "缺席")
    if no_bag:
        expect(hash_dim == 0, "no_bag 的 features.hash_dim", hash_dim, 0)
        expect("hash_algo" not in feats, "no_bag 不写哈希字段（hash_algo 缺席）", "hash_algo" in feats, False)
        expect(
            int(feats["feature_dim"]) == int(bd.extra_dim_for(features)),
            f"no_bag 的 D = extra_dim_for({features})",
            feats["feature_dim"],
            int(bd.extra_dim_for(features)),
        )
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
        expect(
            int(feats["feature_dim"]) == bd.feature_dim(int(bd.HASH_DIM), False, features),
            "缺省口径 D（HASH_DIM + extra_dim_for(features)）",
            feats["feature_dim"],
            bd.feature_dim(int(bd.HASH_DIM), False, features),
        )
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
    want_cols = bd.feature_columns(hash_dim, bool(no_bag), features)
    expect(len(cols) == len(want_cols), "特征列定义条目数", len(cols), len(want_cols))
    cover_ok = True
    cursor = 0
    for c in cols:
        if int(c["start"]) != cursor:
            cover_ok = False
        cursor = int(c["end"]) + 1
    expect(cover_ok and cursor == bd.feature_dim(hash_dim, bool(no_bag), features), "特征列定义无缝覆盖 [0, D-1]", f"cursor={cursor}", bd.feature_dim(hash_dim, bool(no_bag), features))
    expect(
        [c["name"] for c in cols] == [c["name"] for c in want_cols],
        "特征列名与 feature_columns(hash_dim, no_bag, features) 一致",
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

    覆盖范围：**产物口径变体 × 特征集合**（base/rich）的笛卡尔积，外加 dev split 的 rich 产物
    —— 即 base 4 个 + rich 4 个 + dev 2 个（共 10 个口径，每个口径连跑两次并与该口径的产物 SHA256 比对）。
    ``--split all`` 时不触发 dev 全量 compact 构建（单 split 数小时），dev 相关口径记 SKIP。

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
    # [!] R34-1 修复：A6 重写时误删了 ``splits`` 的定义，导致 E1（以及缺省
    #     ``--checks E1,E2,...``）直接 NameError。这里补回，并让 dev 分支复用同一口径。
    ambiguous = str(args.split) == "all"
    splits: Tuple[str, ...] = SPLITS if ambiguous else (str(args.split),)
    # [!] 覆盖范围 = **产物口径变体 × 特征集合**（base/rich），外加 dev split 的 rich 产物：
    #     即 base 4 个（h64/nobag × wiki/web）+ rich 4 个 + dev 2 个。
    #     ``--split all`` 时对 dev 走 **compact 全量构建非常耗时**（单 split 数小时），故与
    #     dev 相关的两个口径记 SKIP 并说明——"缺产物记 SKIP"是本模块的统一口径。
    targets: List[Tuple[str, str, bool, str]] = []
    for label, no_bag in PRODUCT_VARIANTS:
        for _flab, feat in PRODUCT_FEATURE_SETS:
            for split in splits:
                targets.append((split, label, bool(no_bag), feat))
    if ambiguous:
        for split in DEV_SPLITS:
            for flab, feat in DEV_FEATURE_SETS:
                lines.append(
                    f"  [SKIP] {split}/{flab}: --split all 不触发 dev 全量 compact 构建"
                    f"（单 split 数小时）；如需覆盖请显式 --split {split}"
                )
    elif str(args.split) in DEV_SPLITS:
        # [!] 只有显式指定 **dev split** 时才追加 dev 口径（base/rich 的 dev 产物都在 dev split 下）；
        #     早期实现无条件追加，导致 `--split wiki` 会多跑一遍 wiki 的 rich 口径（既慢又是重复覆盖）。
        for flab, feat in DEV_FEATURE_SETS:
            targets.append((str(args.split), flab, False, feat))
    built: Dict[Tuple[str, bool, str], List[str]] = {}
    for split, label, no_bag, feat in targets:
        key = (split, bool(no_bag), feat)
        shas: List[str] = []
        failed = False
        for run_i, sub in enumerate(("idem_a", "idem_b"), start=1):
            out_dir = os.path.join(VERIFY_DIR, sub, f"{label}_{feat}")
            argv = [
                "--split", split,
                "--archive", str(args.archive),
                "--out-dir", out_dir,
                "--features", feat,
            ]
            if no_bag:
                argv.append("--no-bag")
            t0 = time.time()
            rc = bd.main(argv)
            lines.append(
                f"  [{split}/{label}/{feat}] 第 {run_i} 次构建：退出码 {rc}，"
                f"用时 {time.time() - t0:.1f} s，目录 {out_dir}"
            )
            if rc != 0:
                problems.append(f"{split}/{label}/{feat} 第 {run_i} 次构建退出码 {rc}")
                failed = True
                break
            name = os.path.basename(product_path(split, no_bag, feat))
            fp = os.path.join(out_dir, name)
            if not os.path.isfile(fp):
                problems.append(f"{split}/{label}/{feat} 第 {run_i} 次构建缺产物 {fp}")
                failed = True
                continue
            shas.append(sha256_file(fp))
        if failed:
            continue
        built[key] = shas
    for (split, no_bag, feat), shas in built.items():
        prod = product_path(split, no_bag, feat)
        tag = f"{split}/{'nobag' if no_bag else 'h64'}" + ("_rich" if feat == "rich" else "")
        if len(shas) != 2 or not os.path.isfile(prod):
            problems.append(f"{tag} 无法比对（构建次数 {len(shas)}）")
            continue
        sha_prod = sha256_file(prod)
        same = shas[0] == shas[1] == sha_prod
        lines.append(
            f"  {tag}: A={shas[0][:16]}... B={shas[1][:16]}... "
            f"正式产物={sha_prod[:16]}... -> {'一致' if same else '不一致'}"
        )
        if not same:
            problems.append(f"{tag} 三次 SHA256 不一致")
    passed = not problems
    if problems:
        lines.append("  问题：" + "；".join(problems))
    return CheckResult("E1", "构建幂等（每个口径逐字节一致）", passed, detail="\n".join(lines))


def check_e2_contract(args: argparse.Namespace, product_set: str = "base") -> CheckResult:
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
    for split, label, no_bag, path in product_list(args, product_set):
        if not os.path.isfile(path):
            lines.append(f"  [SKIP] {split}/{label}: 缺产物（该口径尚未构建）：{path}")
            continue
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

def check_e4_no_leak(args: argparse.Namespace, product_set: str = "base") -> CheckResult:
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
    products = product_list(args, product_set)
    lines: List[str] = []
    problems: List[str] = []
    per_product: Dict[str, Dict[str, Any]] = {}
    for split, label, no_bag, path in products:
        key = f"{split}/{label}"
        if not os.path.isfile(path):
            lines.append(f"  [SKIP] {key}: 缺产物（该口径尚未构建）：{path}")
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
    def _compare_variants(split: str, key_a: str, key_b: str, tag: str) -> None:
        """对比同 split 两个产物的样本口径与附加特征块（逐条 + 逐位）。"""
        a, b = per_product[key_a], per_product[key_b]
        if a["triples"] != b["triples"]:
            problems.append(f"{split} 的 {key_a} 与 {key_b} 样本口径不一致（行序/文档/label 有差异）")
            lines.append(f"  [FAIL] {split}: {key_a} 与 {key_b} 的样本口径**不一致**")
        else:
            lines.append(f"  [OK]   {split}: {key_a} 与 {key_b} 的样本口径（行序/QuestionId/文档/label）逐条一致")
        try:
            with np.load(a["path"], allow_pickle=False) as za, np.load(b["path"], allow_pickle=False) as zb:
                Xa = np.asarray(za["X"])
                Xb = np.asarray(zb["X"])
            ha = int(a["meta"]["features"]["hash_dim"])
            hb = int(b["meta"]["features"]["hash_dim"])
            extra_a = Xa[:, ha:] if ha > 0 else Xa
            extra_b = Xb[:, hb:] if hb > 0 else Xb
            if tag == "base_rich":
                # base 的附加块 = rich 附加块的前 6 列（同一 token 集合口径；rich 只追加 4 列）
                if extra_b.shape[1] < extra_a.shape[1]:
                    raise AssertionError(f"rich 附加块列数 {extra_b.shape[1]} < base {extra_a.shape[1]}")
                head_b = extra_b[:, : extra_a.shape[1]]
                same_extra = extra_a.shape == head_b.shape and np.array_equal(extra_a, head_b)
                detail = f"base 附加块 {extra_a.shape} vs rich 附加块前 {extra_a.shape[1]} 列 {head_b.shape}"
            else:
                same_extra = extra_a.shape == extra_b.shape and np.array_equal(extra_a, extra_b)
                detail = f"{extra_a.shape} vs {extra_b.shape}"
            lines.append(
                f"  {'[OK]  ' if same_extra else '[FAIL]'} {split}: {key_a} 与 {key_b} 的附加特征"
                f"{'逐位一致' if same_extra else '**不一致**'}（{detail}）"
            )
            if not same_extra:
                problems.append(f"{split} 的 {key_a} 与 {key_b} 附加特征不一致")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{split} 附加特征比对失败（{key_a} vs {key_b}）：{type(exc).__name__}: {exc}")

    for split in sorted({p[0] for p in products}):
        keys = [f"{split}/{lab}" for lab, _nb in PRODUCT_VARIANTS if f"{split}/{lab}" in per_product]
        if len(keys) < 2:
            continue
        _compare_variants(split, keys[0], keys[1], "hash_nobag")

    # ①-b base 与 rich 必须共享同一采样（追加的特征列不得改变样本口径），且 base 的附加块
    #     必须与 rich 附加块的前 6 列逐位一致（rich 只在末尾追加 4 列）
    for split in sorted({p[0] for p in products}):
        for lab, _nb in PRODUCT_VARIANTS:
            ka = f"{split}/{lab}"
            kb = f"{split}/{lab}_rich"
            if ka in per_product and kb in per_product:
                _compare_variants(split, ka, kb, "base_rich")

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


def check_e3_balance_and_answer(args: argparse.Namespace, product_set: str = "base") -> CheckResult:
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
    products = product_list(args, product_set)
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
            # [!] 缺产物记 SKIP（该口径尚未构建），不得记 FAIL —— 否则 --product-set all/dev
            #     在 dev 全量产物未就绪时会被误判为失败（R32-1 即此）。
            lines.append(f"  [SKIP] {key}: 缺产物（该口径尚未构建）：{path}")
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
            raw_qa, _passes, _status, _members = bd.load_archive_qa(args.archive, archive_sha, [split])
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


def check_e5_train(args: argparse.Namespace, product_set: str = "base") -> CheckResult:
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
        targets = list(product_list(args, product_set))
    lines: List[str] = []
    problems: List[str] = []
    n_ok = 0
    for split, label, no_bag, path in targets:
        key = f"{split}/{label}"
        if not os.path.isfile(path):
            # [!] 与 E2/E3/E4 同族口径：缺产物 = 该口径尚未构建 -> 记 SKIP，不得记 FAIL
            #     （否则 --product-set all 在不带 --skip-e5 时会因 dev 全量产物缺失而整体失败）
            lines.append(f"  [SKIP] {key}: 缺产物（该口径尚未构建）：{path}")
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
CV_FOLDS: int = 5
CV_SEED: int = 0
CV_ITERS: int = 400
CV_LR: float = 0.5
CV_LAM: float = 1e-3
E7_MAX_SAMPLES: int = 12000
E7_TOL: float = 1e-12


def _fit_logistic(
    Xtr: np.ndarray, ytr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L2 正则逻辑回归（全批量梯度下降，确定性；不依赖 sklearn）。

    参数
    ----
    Xtr : np.ndarray
        ``[n, d]`` 训练特征（float64）。
    ytr : np.ndarray
        ``[n]`` 0/1 标签（float64）。

    返回
    ----
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(coef[b, w...], mu, sd)``：均值/标准差标准化参数与系数。
    """
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd <= 0, 1.0, sd)
    Z = (Xtr - mu) / sd
    w = np.zeros(Z.shape[1], dtype=np.float64)
    b = 0.0
    n = Z.shape[0]
    for _ in range(int(CV_ITERS)):
        logits = Z @ w + b
        p = 1.0 / (1.0 + np.exp(-logits))
        g = p - ytr
        w -= CV_LR * (Z.T @ g / n + CV_LAM * w)
        b -= CV_LR * float(g.mean())
    return np.concatenate([[b], w]), mu, sd


def _cv_scores(X: np.ndarray, y: np.ndarray, folds: int = CV_FOLDS) -> Tuple[float, float]:
    """5 折交叉验证，返回 ``(准确率, AUC)``（折划分由固定 seed 决定，确定性）。

    参数
    ----
    X : np.ndarray
        ``[M, d]`` 特征（float64）。
    y : np.ndarray
        ``[M]`` 0/1 标签。
    folds : int
        折数。

    返回
    ----
    Tuple[float, float]
        ``(平均准确率, 池化 AUC)``。
    """
    m = X.shape[0]
    order = np.random.default_rng(CV_SEED).permutation(m)
    scores = np.zeros(m, dtype=np.float64)
    for k in range(int(folds)):
        te = order[k::folds]
        tr = np.setdiff1d(order, te, assume_unique=True)
        coef, mu, sd = _fit_logistic(X[tr], y[tr].astype(np.float64))
        Z = (X[te] - mu) / np.where(sd <= 0, 1.0, sd)
        scores[te] = Z @ coef[1:] + coef[0]
    acc = float(((scores >= 0.0).astype(np.int64) == y).mean())
    order2 = np.argsort(scores, kind="mergesort")
    ranks = np.empty(m, dtype=np.float64)
    ranks[order2] = np.arange(1, m + 1, dtype=np.float64)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos <= 0 or n_neg <= 0:
        return acc, float("nan")
    auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return acc, auc


def check_e7_feature_power(args: argparse.Namespace) -> CheckResult:
    """E7：rich 特征判别力分解（5 折 CV）——**rich 是否优于 base**，结论按实测给出。

    口径：对每个 verified split，最多取 ``--e7-max-samples`` 条样本（固定 seed 抽样，
    控制运行时长），对 6 组列子集各做一次 5 折逻辑回归 CV，并逐列算 |Pearson r|。
    判据（**只作读数，不作粉饰**）：``qcov_idf`` 相对 ``qcov`` 有提升、且 ``tfidf_cos``
    相对 ``qcov`` 有提升；两者都无提升即如实报「rich 相对 base 无增益」。

    参数
    ----
    args : argparse.Namespace
        命令行参数（用 ``--split`` / ``--e7-max-samples``）。

    返回
    ----
    CheckResult
        检查结果（明细含逐列 |r| 与 6 组 CV 数字）。
    """
    n_max = int(getattr(args, "e7_max_samples", E7_MAX_SAMPLES))
    splits = SPLITS if args.split == "all" else (str(args.split),)
    lines: List[str] = []
    problems: List[str] = []
    def _col_start(meta: Dict[str, Any]) -> Dict[str, int]:
        """从产物 meta 的 ``features.columns`` 回读 ``列名 -> 起始列号``（避免硬编码错位）。"""
        return {str(c["name"]): int(c["start"]) for c in meta["features"]["columns"]}
    for split in splits:
        pb = product_path(split, False, "base")
        pr = product_path(split, False, "rich")
        if not (os.path.isfile(pb) and os.path.isfile(pr)):
            lines.append(f"  [SKIP] {split}: 缺 base/rich 产物（{pb} / {pr}）")
            continue
        with np.load(pb, allow_pickle=False) as z:
            Xb = np.asarray(z["X"], dtype=np.float64)
            y = np.asarray(z["y"]).astype(np.int64)
        with np.load(pr, allow_pickle=False) as z:
            Xr = np.asarray(z["X"], dtype=np.float64)
        m = int(y.size)
        if n_max > 0 and m > n_max:
            idx = np.random.default_rng(CV_SEED).choice(m, size=n_max, replace=False)
            idx = np.sort(idx)
            Xb = Xb[idx]
            Xr = Xr[idx]
            y = y[idx]
            lines.append(f"  {split}: 抽样 {n_max}/{m} 条（种子 {CV_SEED}）用于 CV")
        idx_r = _col_start(bd.read_npz_meta(pr))
        idx_b = _col_start(bd.read_npz_meta(pb))
        i_qcov = idx_b["q_to_d_coverage"]
        i_qidf = idx_r["qcov_idf"]
        i_cos = idx_r["tfidf_cos"]
        # [!] 附加块起点**按产物 meta 的 features.hash_dim 推导桶数边界**，不再按列名
        #     （旧写法 `c["name"] != "bow_hash64"` 一旦列名变更就会静默错位——与 R30-D7 同族隐患）。
        #     口径：词袋块占前 hash_dim 列（no_bag 时 meta 记 hash_dim=0，故起点自然是 0），
        #     故"非词袋列的起点" == hash_dim。该值由产物自己声明，与列名解耦。
        feat_b = bd.read_npz_meta(pb)["features"]
        feat_r = bd.read_npz_meta(pr)["features"]
        start_b = int(feat_b["hash_dim"])
        start_b_rich = int(feat_r["hash_dim"])
        # 交叉校验（防 meta 自相矛盾，且**不依赖任何列名**）。口径：meta 的 columns[] 只登记
        # **命名的非词袋列**（词袋块作为整体只登记一行、起点 0），故可断言：
        #   (a) 起点严格递增且首列起点 = 0；
        #   (b) 命名列条数 = feature_dim - hash_dim（词袋块宽度 = hash_dim）；
        #   (c) 末列起点 = feature_dim - 1（铺满到最后一列）；
        #   (d) no_bag=True 时 hash_dim 必须为 0。
        for _tag, _feat in (("base", feat_b), ("rich", feat_r)):
            _starts = [int(c["start"]) for c in _feat["columns"]]
            _fdim = int(_feat["feature_dim"])
            _hdim = int(_feat["hash_dim"])
            if _starts != sorted(_starts) or len(set(_starts)) != len(_starts) or _starts[0] != 0:
                problems.append(f"{split}/{_tag}: columns[].start 非严格递增或首列起点非 0：{_starts[:5]}")
            # columns[] 会**把词袋块整体登记为 1 行**（起点 0），故条数 = 词袋行数 + 命名列数，
            # 其中词袋行数 = 0（no_bag）/ 1（有词袋块）—— 该口径由现场枚举 meta 得到。
            _expect = (0 if bool(_feat["no_bag"]) else 1) + _fdim - _hdim
            if len(_starts) != _expect:
                problems.append(
                    f"{split}/{_tag}: columns 条数 {len(_starts)} != 词袋行数+命名列数 = {_expect}"
                )
            if _starts[-1] != _fdim - 1:
                problems.append(f"{split}/{_tag}: 末列起点 {_starts[-1]} != feature_dim - 1 = {_fdim - 1}")
            if bool(_feat["no_bag"]) and _hdim != 0:
                problems.append(f"{split}/{_tag}: no_bag=True 但 hash_dim={_hdim}（应为 0）")
        start_r = min(idx_r[n] for n in ("qcov_idf", "dcov_idf", "tfidf_cos", "ans_isnum_frac"))
        extra_base = Xb[:, start_b:]
        combos: List[Tuple[str, np.ndarray]] = [
            ("base 附加块", extra_base),
            ("base + qcov_idf", np.hstack([extra_base, Xr[:, i_qidf : i_qidf + 1]])),
            ("base + tfidf_cos", np.hstack([extra_base, Xr[:, i_cos : i_cos + 1]])),
            ("base + 两列 IDF", np.hstack([extra_base, Xr[:, i_qidf : i_qidf + 1], Xr[:, i_cos : i_cos + 1]])),
            ("rich 附加块（十列）", Xr[:, start_b_rich:]),
            ("单列 qcov(base)", Xb[:, i_qcov : i_qcov + 1]),
        ]
        acc: Dict[str, float] = {}
        auc: Dict[str, float] = {}
        for name, Xs in combos:
            a, u = _cv_scores(Xs, y)
            acc[name] = a
            auc[name] = u
            lines.append(f"    CV {name}: acc={a:.4f} auc={u:.4f}（d={Xs.shape[1]}）")
        yc = y.astype(np.float64) - float(y.mean())
        r_line: List[str] = []
        for name in ("q_to_d_coverage", "qcov_idf", "dcov_idf", "tfidf_cos", "ans_isnum_frac"):
            col = Xr[:, idx_r[name]]
            xc = col - float(col.mean())
            den = float(np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
            r = float((xc * yc).sum() / den) if den > 0 else 0.0
            r_line.append(f"|r({name})|={abs(r):.4f}")
        lines.append("    逐列相关性：" + "，".join(r_line))
        base_acc = acc["base 附加块"]
        qcov_acc = acc["单列 qcov(base)"]
        gain_idf = acc["base + qcov_idf"] - base_acc
        gain_cos = acc["base + tfidf_cos"] - base_acc
        gain_both = acc["base + 两列 IDF"] - base_acc
        gain_rich = acc["rich 附加块（十列）"] - base_acc
        qcov_gain_idf = acc["base + qcov_idf"] - qcov_acc
        qcov_gain_cos = acc["base + tfidf_cos"] - qcov_acc
        lines.append(
            f"    相对 base 增量：+qcov_idf {gain_idf:+.4f}，+tfidf_cos {gain_cos:+.4f}，"
            f"+两列 {gain_both:+.4f}，rich 十列 {gain_rich:+.4f}"
        )
        lines.append(
            f"    相对单列 qcov：+qcov_idf {qcov_gain_idf:+.4f}，+tfidf_cos {qcov_gain_cos:+.4f}"
        )
        lines.append(
            "    [结论] "
            + (
                f"rich 优于 base（rich 十列 {acc['rich 附加块（十列）']:.4f} > base {base_acc:.4f}）"
                if acc["rich 附加块（十列）"] > base_acc + E7_TOL
                else f"**rich 相对 base 无增益**（rich 十列 {acc['rich 附加块（十列）']:.4f} <= base {base_acc:.4f}）"
            )
        )
    passed = not problems
    if problems:
        lines.append("  问题：" + "；".join(problems))
    return CheckResult(
        "E7", "rich 特征判别力分解（5 折 CV：rich 是否优于 base）", passed, detail="\n".join(lines)
    )


def _question_ids_from_artifact(split: str, no_bag: bool, features: str) -> Tuple[Optional[set], str]:
    """从产物 meta 里取该 split 的 QuestionId 集合（缺产物返回 ``(None, 原因)``）。"""
    path = product_path(split, no_bag, features)
    if not os.path.isfile(path):
        return None, f"缺产物 {path}"
    meta = bd.read_npz_meta(path)
    return {str(q["question_id"]) for q in meta["questions"]}, ""

def check_e8_no_leak(args: argparse.Namespace) -> CheckResult:
    """E8：无泄漏（同 QuestionId 正负交集 = 0；verified 与 dev 的 QuestionId 交集 = 0）。

    覆盖既有 4 个正式产物 + 新增 rich / dev 产物；缺产物时记 SKIP 并说明，不计失败。

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
    targets: List[Tuple[str, bool, str]] = []
    for split in (SPLITS if args.split == "all" else (str(args.split),)):
        for _lab, no_bag in PRODUCT_VARIANTS:
            targets.append((split, bool(no_bag), "base"))
            targets.append((split, bool(no_bag), "rich"))
    for split in DEV_SPLITS:
        for _lab, feat in DEV_FEATURE_SETS:
            targets.append((split, False, feat))
    qids: Dict[str, set] = {}
    for split, no_bag, feat in targets:
        key = f"{split}/{'nobag' if no_bag else 'h64'}" + ("_rich" if feat == "rich" else "")
        path = product_path(split, no_bag, feat)
        if not os.path.isfile(path):
            lines.append(f"  [SKIP] {key}: 缺产物 {path}")
            continue
        meta = bd.read_npz_meta(path)
        questions = meta["questions"]
        documents = meta["documents"]
        pos: Dict[str, set] = {}
        neg: Dict[str, set] = {}
        dup = 0
        seen_pairs = set()
        for qi, di, lab, _hit in meta["samples"]:
            qid = str(questions[int(qi)]["question_id"])
            doc = str(documents[int(di)])
            if (qid, doc) in seen_pairs:
                dup += 1
            seen_pairs.add((qid, doc))
            (pos if int(lab) == 1 else neg).setdefault(qid, set()).add(doc)
        overlap = [q for q in pos if pos[q] & neg.get(q, set())]
        lines.append(
            f"  {key}: QuestionId {len(set(pos) | set(neg))} 个，同题正负文档交集 {len(overlap)}，"
            f"重复 (QuestionId,文档) 对 {dup}"
        )
        if overlap:
            problems.append(f"{key} 存在同一 QuestionId 的正负文档交集（示例 {overlap[:3]}）")
        if dup:
            problems.append(f"{key} 存在重复 (QuestionId,文档) 对 {dup} 个")
        if not overlap and not dup:
            qids[key] = {str(q["question_id"]) for q in questions}
    # verified 与 非 verified 的 QuestionId 交集必须为空
    for verified_key, dev_key in (
        ("wiki/h64", "wiki-dev/rich_wiki_dev"),
        ("web/h64", "web-dev/rich_web_dev"),
    ):
        a, b = qids.get(verified_key), qids.get(dev_key)
        if a is None or b is None:
            lines.append(f"  [SKIP] {verified_key} vs {dev_key}: 缺一侧产物，交集未校验")
            continue
        inter = a & b
        lines.append(
            f"  {verified_key} vs {dev_key}: QuestionId 交集 = {len(inter)}"
            f"（示例 {sorted(inter)[:5]}），两侧规模 {len(a)} / {len(b)}"
        )
        if inter:
            problems.append(f"verified 与 dev 的 QuestionId 交集非空：{sorted(inter)[:5]}")
    passed = not problems
    if problems:
        lines.append("  问题：" + "；".join(problems))
    return CheckResult("E8", "无泄漏（同题正负不交叉 + verified/dev QuestionId 交集为 0）", passed, detail="\n".join(lines))


# CLI
# ======================================================================
ALL_CHECKS: Tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8")
# 缺省执行顺序（E7/E8 为新增检查：E7 需 rich 产物、E8 需 dev 产物，缺产物时记 SKIP）
DEFAULT_CHECKS: Tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5", "E6")


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
        description="n3d_qa 产物验证（E1 幂等 / E2 契约 / E3 均衡+答案复核 / E4 无泄漏 / E5 端到端 / E6 零回归）"
    )
    parser.add_argument("--split", type=str, default="all", choices=list(bd.SPLIT_CHOICES),
                        help="验证哪个 split（缺省 all）")
    parser.add_argument("--archive", type=str, default=bd.DEFAULT_ARCHIVE,
                        help="triviaqa-rc.tar.gz 路径（E1 / E3 需要）")
    parser.add_argument("--checks", type=str, default=",".join(DEFAULT_CHECKS),
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
    parser.add_argument(
        "--product-set",
        type=str,
        default="base",
        choices=["base", "rich", "dev", "all"],
        help=(
            "验证哪一组产物：base（缺省，既有 4 个正式产物）/ rich（新增 4 个 rich 产物）/"
            "dev（2 个非 verified dev 产物）/ all（上述全部）"
        ),
    )
    parser.add_argument(
        "--e7-max-samples",
        type=int,
        default=E7_MAX_SAMPLES,
        help=f"E7 每个 split 最多用多少条样本做 5 折 CV（缺省 {E7_MAX_SAMPLES}，0 = 全量）",
    )
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
        print(f"[n3d_qa.verify] 未知检查项 {unknown}；可选 {ALL_CHECKS}", file=sys.stderr)
        return 1
    if bool(args.skip_e5) and "E5" in wanted:
        wanted = tuple(x for x in wanted if x != "E5")
    t0 = time.time()
    product_sets: Tuple[str, ...] = (
        ("base", "rich", "dev") if str(args.product_set) == "all" else (str(args.product_set),)
    )
    print("=" * 78)
    print("n3d_qa 产物验证报告")
    print("=" * 78)
    print(f"产物目录：{PRODUCT_DIR}")
    print(f"验证产物目录：{VERIFY_DIR}")
    print(f"检查项：E0（恒定，逐产物） + {list(wanted)}")
    print(f"产物口径变体：{[(lab, ("D=6" if nb else "D=" + str(int(bd.HASH_DIM) + int(bd.EXTRA_DIM)))) for lab, nb in PRODUCT_VARIANTS]}")
    print(f"验证产物组：{list(product_sets)}（base = 既有 4 个正式产物；rich / dev 落 _verify/）")
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

    for pset in product_sets:
        for split, label, no_bag, _path in product_list(args, pset):
            emit(check_meta_contract(split, no_bag, "rich" if label.endswith("_rich") else "base"))
    for pset in product_sets:
        if "E2" in wanted:
            emit(check_e2_contract(args, pset))
        if "E3" in wanted:
            emit(check_e3_balance_and_answer(args, pset))
        if "E4" in wanted:
            emit(check_e4_no_leak(args, pset))
    if "E1" in wanted:
        emit(check_e1_idempotent(args))
    if "E7" in wanted:
        emit(check_e7_feature_power(args))
    if "E8" in wanted:
        emit(check_e8_no_leak(args))
    if "E5" in wanted:
        if args.e5_split:
            args.split = str(args.e5_split)  # 仅按该 split 的产物跑 E5
        emit(check_e5_train(args, str(args.product_set)))
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
