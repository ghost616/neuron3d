"""``n3d_qa_learn.encoders`` 的验收入口（CLI）。

驱动脚本::

    python -m n3d_qa_learn.encoders_run {registry|fetch|info|drill|verify|boundary}

子命令
------
=================  ==================================================================
``registry``       列编码器注册表（名字 / 类型 / 声明维度 / 池化 / revision）
``fetch``          从镜像下载 ``BAAI/bge-m3`` 固定 revision 到 ``models/bge-m3/``
``info``           **G1**：加载模型并现场核实 ``hidden_size`` / ``max_position_embeddings``
                   / CLS pooling / revision / 权重 SHA256（如实报告下载来源）
``drill``          **G2**：单条端到端演练（1 条文本 -> 编码 -> 缓存写入 -> 二次读取）
``verify``         **G3**：① 指纹 == 注册表/落盘声明 ② 同文本两次编码逐位一致
                   ③ 与落盘缓存逐位比对
``boundary``       **G4**：边界处置自检（空 / 超长 / 模型缺失 / 维度不符）
=================  ==================================================================

退出码
------
``0`` 全部通过；``1`` 业务失败（某项未通过 / 编码器不可用）；``2`` 参数错误（argparse）。

产物纪律
--------
报告与日志一律写 ``checkpoints/qa_learn/_verify/encoders/``；嵌入缓存写
``checkpoints/qa_learn/_cache/emb/``（**跨进程复用**，与验证目录分离）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

# 控制台编码加固（与 cli.py 同口径：源码含中文，GBK 控制台会 UnicodeEncodeError）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

from . import encoders as E

#: 验证类运行目录（G2/G3/G4 的报告与日志）。
VERIFY_DIR: str = os.path.join("checkpoints", "qa_learn", "_verify", "encoders")

#: ``drill`` 落盘的声明文件名（供 ``verify`` 的 ① 项对账）。
DRILL_JSON: str = "drill.json"


class _Tee:
    """把 stdout 同时写到控制台与 **UTF-8 无 BOM** 日志文件。

    存在理由（与 ``step2_run._Tee`` 同口径）：PowerShell 的 ``>`` / ``Tee-Object``
    在 Windows PowerShell 5.1 下默认写 **UTF-16LE**（BOM ``FF FE``），中文虽可解但
    不符合本项目的日志产物纪律。日志一律由 Python 自己以 UTF-8 落盘。
    """

    def __init__(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8", newline="\n")
        self._stdout = sys.stdout

    def write(self, text: str) -> int:
        """同时写控制台与文件。"""
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


def _dump(obj: Any) -> None:
    """UTF-8 安全 JSON 打印。"""
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def _write_json(path: str, obj: Any) -> str:
    """写 UTF-8（无 BOM）JSON；返回内容 SHA256。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    blob = json.dumps(obj, ensure_ascii=False, indent=1, default=str).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(blob)
    return E.sha256_bytes(blob)


def _encoder_config(args: argparse.Namespace, *, cache_default: Optional[bool] = None) -> E.EncoderConfig:
    """由 CLI 参数装配编码器配置（唯一的选择实现入口）。

    参数
    ----
    args : argparse.Namespace
        已解析参数。
    cache_default : Optional[bool]
        ``None`` = 自动（``hf`` 家族开、``hash`` 家族关）；
        ``True`` = 默认开启缓存（``drill`` 的职责就是演练「缓存写入 + 二次读取」链路，
        ``hash`` 家族也必须在默认档下走通，见 D1 修复）。
    """
    if cache_default is True:
        use_cache: Optional[bool] = True if args.use_cache else False
    elif cache_default is False:
        use_cache = False
    else:
        use_cache = True if getattr(args, "force_cache", False) else (
            None if args.use_cache else False
        )
    return E.EncoderConfig(
        name=str(args.encoder),
        role=str(args.role),
        max_length=int(args.max_length),
        source=str(args.source),
        revision=str(args.revision),
        cache_dir=str(args.cache_dir),
        use_cache=use_cache,
        batch_size=int(args.batch_size),
        device=str(args.device),
        local_files_only=bool(args.local_files_only),
        verify_weights=not bool(args.no_verify_weights),
    )


def _cache_key_of(encoder: Any, text: str) -> str:
    """取编码器对某文本的缓存键（实现自身提供 ``cache_key`` 时才有值）。"""
    fn = getattr(encoder, "cache_key", None)
    return str(fn(text)) if callable(fn) else ""


def _cache_stats_of(encoder: Any) -> Dict[str, Any]:
    """取编码器本进程的缓存命中统计（无缓存时如实登记）。"""
    fn = getattr(encoder, "cache_stats", None)
    return dict(fn()) if callable(fn) else {"note": "缓存未启用"}


def _declaration_of(encoder: Any) -> Dict[str, Any]:
    """取编码器的口径声明（HF 家族有 ``declaration``；hash 家族退化为通用表）。"""
    fn = getattr(encoder, "declaration", None)
    if callable(fn):
        return dict(fn())
    return {
        "kind": "hash",
        "schema_version": str(E.ENCODER_SCHEMA_VERSION),
        "pooling": "hash-bag",
        "normalize": str(E.NORMALIZE_MODE),
        "dim": int(encoder.dim),
        "fingerprint": str(encoder.fingerprint()),
    }


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def cmd_registry(args: argparse.Namespace) -> int:
    """列编码器注册表（含声明维度与固定 revision）。"""
    entries: List[Dict[str, Any]] = []
    for name in E.list_encoders():
        spec = E.get_spec(name)
        entries.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "expect_dim": int(spec.expect_dim),
                "pooling": spec.pooling,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "weight_file": spec.weight_file,
                "mirror_endpoint": spec.mirror_endpoint,
                "interface": spec.interface,
                "note": spec.note,
            }
        )
    out = {
        "schema_version": str(E.ENCODER_SCHEMA_VERSION),
        "n_encoders": int(len(entries)),
        "role_defaults": dict(E.ROLE_DEFAULT_ENCODER),
        "role_max_length": dict(E.ROLE_MAX_LENGTH),
        "cache_dir": str(E.DEFAULT_EMB_CACHE_DIR),
        "entries": entries,
    }
    _dump(out)
    if args.out_dir:
        path = os.path.join(str(args.out_dir), "registry.json")
        digest = _write_json(path, out)
        print(f"[INFO] 注册表 -> {path} (sha256={digest[:16]}...)")
    print(f"[OK] 编码器注册表：{len(entries)} 项 -> {[e['name'] for e in entries]}")
    return 0


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def cmd_fetch(args: argparse.Namespace) -> int:
    """从镜像下载固定 revision 的模型到本地目录（**可复算的下载来源**）。"""
    os.environ.setdefault("HF_ENDPOINT", E.HF_MIRROR_ENDPOINT)
    # 现场实测：hf-xet 的 CAS 服务在镜像下返回 401；强制走普通 HTTP 下载。
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print(f"[FAIL] 缺少 huggingface_hub：{exc}", file=sys.stderr)
        return 1
    spec = E.get_spec(str(args.encoder))
    revision = str(args.revision) or str(spec.revision)
    if not revision:
        print(f"[FAIL] 编码器 {spec.name!r} 未声明固定 revision", file=sys.stderr)
        return 1
    local_dir = str(args.local_dir) or E.DEFAULT_MODEL_DIR
    patterns = [
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "sentencepiece.bpe.model",
        str(spec.weight_file) or "pytorch_model.bin",
        "1_Pooling/*", "sentence_bert_config.json", "modules.json", "README.md",
    ]
    t0 = time.time()
    try:
        path = snapshot_download(
            repo_id=str(spec.model_id), revision=revision, local_dir=local_dir,
            allow_patterns=patterns,
        )
    except Exception as exc:  # noqa: BLE001 - 下载失败必须可读报错
        print(f"[FAIL] 下载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    files: List[Dict[str, Any]] = []
    for root, _dirs, names in os.walk(path):
        for name in sorted(names):
            full = os.path.join(root, name)
            files.append(
                {
                    "path": os.path.relpath(full, path).replace("\\", "/"),
                    "bytes": int(os.path.getsize(full)),
                }
            )
    digest = ""
    weight_path = os.path.join(path, str(spec.weight_file) or "pytorch_model.bin")
    if os.path.isfile(weight_path):
        digest = E.file_sha256(weight_path)
    out = {
        "encoder": str(spec.name),
        "repo_id": str(spec.model_id),
        "revision": revision,
        "endpoint": str(os.environ.get("HF_ENDPOINT", "")),
        "hf_hub_disable_xet": str(os.environ.get("HF_HUB_DISABLE_XET", "")),
        "local_dir": str(path).replace("\\", "/"),
        "seconds": float(time.time() - t0),
        "weight_file": os.path.basename(weight_path),
        "weight_sha256": digest,
        "files": files,
    }
    _dump(out)
    if args.out_dir:
        target = os.path.join(str(args.out_dir), "fetch.json")
        print(f"[INFO] 下载取证 -> {target} (sha256={_write_json(target, out)[:16]}...)")
    print(f"[OK] 已下载 {len(files)} 个文件到 {path}")
    return 0


# ---------------------------------------------------------------------------
# info（G1）
# ---------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    """G1：加载模型并现场核实结构量与来源（如实报告，不做任何包装）。"""
    cfg = _encoder_config(args)
    t0 = time.time()
    try:
        encoder = E.build_vectorizer(cfg)
    except Exception as exc:  # noqa: BLE001 - 不可用必须可读报错
        print(f"[FAIL] 编码器不可用：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    load_seconds = time.time() - t0
    describe_fn = getattr(encoder, "describe", None)
    info: Dict[str, Any] = dict(describe_fn()) if callable(describe_fn) else _declaration_of(encoder)
    info["load_seconds"] = float(load_seconds)
    info["fingerprint"] = str(encoder.fingerprint())
    info["registry_expect_dim"] = int(E.get_spec(cfg.resolved_name()).expect_dim)
    info["declared_dim"] = int(E.declared_dim(cfg))
    info["actual_dim"] = int(encoder.dim)
    info["role_max_length"] = int(cfg.resolved_max_length())
    info["cache_enabled"] = bool(cfg.use_cache_effective())
    _dump(info)
    if args.out_dir:
        target = os.path.join(str(args.out_dir), "info.json")
        print(f"[INFO] G1 取证 -> {target} (sha256={_write_json(target, info)[:16]}...)")
    if str(info.get("kind")) == "hf":
        print(
            f"[OK] G1：hidden_size={info.get('hidden_size')} / "
            f"max_position_embeddings={info.get('max_position_embeddings')} / "
            f"pooling={info.get('pooling')} / revision={info.get('revision')} / "
            f"weight_sha256={str(info.get('weight_sha256'))[:16]}..."
        )
    else:
        print(f"[OK] G1：hash 家族编码器 {info.get('name')!r}，D={info.get('actual_dim')}")
    return 0


# ---------------------------------------------------------------------------
# drill（G2）
# ---------------------------------------------------------------------------


def cmd_drill(args: argparse.Namespace) -> int:
    """G2：单条端到端演练（编码 -> 缓存写入 -> 二次读取 -> 位级一致）。

    缓存默认**开启**（本子命令的职责就是演练缓存链路）；``--no-cache`` 时缓存相关
    判据标为**不适用**并如实登记，而不是伪装成通过或失败。
    """
    cfg = _encoder_config(args, cache_default=True)
    text = str(args.text)
    out_dir = str(args.out_dir) if args.out_dir else VERIFY_DIR
    drill_path = os.path.join(out_dir, DRILL_JSON)
    evidence: Dict[str, Any] = {
        "phase": "reuse" if args.reuse_only else "cold_and_warm",
        "encoder": str(cfg.resolved_name()),
        "role": str(cfg.role),
        "max_length": int(cfg.resolved_max_length()),
        "cache_dir": str(cfg.cache_dir),
        "text_sha256": E.sha256_text(text),
    }

    if args.reuse_only:
        # 二次读取：**全新进程、不加载模型**，只从落盘缓存取回并逐位比对
        if not os.path.isfile(drill_path):
            print(f"[FAIL] 缺少 {drill_path}；请先执行 drill（cold 相位）", file=sys.stderr)
            return 1
        with open(drill_path, "r", encoding="utf-8") as handle:
            first = dict(json.load(handle))
        if str(first.get("text_sha256")) != str(evidence["text_sha256"]):
            print("[FAIL] 二次读取的文本与首轮不同；拒绝在错配文本上比对", file=sys.stderr)
            return 1
        cache = E.EmbeddingCache(str(cfg.cache_dir))
        record = dict(first.get("cache_record") or {})
        key = cache.key_for(
            model_id=str(record.get("model_id", "")),
            revision=str(record.get("revision", "")),
            pooling=str(record.get("pooling", "")),
            max_length=int(record.get("max_length", 0)),
            normalize=str(record.get("normalize", E.NORMALIZE_MODE)),
            weight_sha256=str(record.get("weight_sha256", "")),
            text=text,
        )
        blob = cache.read_bytes(key)
        ok = blob is not None and E.sha256_bytes(blob) == str(first.get("vector_sha256", ""))
        evidence.update(
            {
                "cache_key": str(key),
                "cache_hit": bool(blob is not None),
                "bytes": int(len(blob)) if blob is not None else 0,
                "vector_sha256": E.sha256_bytes(blob) if blob is not None else "",
                "matches_first_run": bool(ok),
                "note": "本相位不加载模型、不依赖 transformers，只读落盘缓存（跨进程复用证明）",
            }
        )
        _dump(evidence)
        if not ok:
            print("[FAIL] 二次读取与首轮不一致", file=sys.stderr)
            return 1
        print(f"[OK] G2(reuse)：缓存跨进程复用成功，key={key[:16]}... bytes={evidence['bytes']}")
        return 0

    t0 = time.time()
    try:
        encoder = E.build_vectorizer(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 编码器不可用：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    load_seconds = time.time() - t0

    t1 = time.time()
    first = E.encode_with_stats_of(encoder, text)
    encode_seconds = time.time() - t1
    first_bytes = E.vector_bytes(first.vector)
    key = _cache_key_of(encoder, text)
    cache_on = bool(key)
    disk = E.EmbeddingCache(str(cfg.cache_dir)).read_bytes(key) if cache_on else None
    second = E.encode_with_stats_of(encoder, text)
    second_bytes = E.vector_bytes(second.vector)
    meta_path = os.path.join(str(cfg.cache_dir), f"{key}.json")
    meta: Dict[str, Any] = {}
    if key and os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = dict(json.load(handle))

    evidence.update(
        {
            "dim": int(encoder.dim),
            "fingerprint": str(encoder.fingerprint()),
            "n_tokens": int(first.n_tokens),
            "n_truncated": int(first.n_truncated),
            "vector_l2_norm": float(E.vector_l2_norm(first.vector)),
            "vector_sha256": E.sha256_bytes(first_bytes),
            "first_cached": bool(first.cached),
            "second_cached": bool(second.cached),
            "second_bitwise_equal": bool(first_bytes == second_bytes),
            "cache_key": str(key),
            "cache_file_written": bool(disk is not None),
            "cache_file_bitwise_equal": bool(disk == first_bytes),
            "cache_entry_meta": {
                k: meta.get(k)
                for k in ("key", "dim", "bytes", "sha256", "model_id", "revision",
                          "pooling", "max_length", "normalize", "weight_sha256",
                          "n_tokens", "n_truncated", "created_utc")
            },
            "cache_record": {
                "model_id": str(meta.get("model_id", "")),
                "revision": str(meta.get("revision", "")),
                "pooling": str(meta.get("pooling", "")),
                "max_length": int(meta.get("max_length", 0)),
                "normalize": str(meta.get("normalize", E.NORMALIZE_MODE)),
                "weight_sha256": str(meta.get("weight_sha256", "")),
            },
            "declaration": _declaration_of(encoder),
            "cache_stats": _cache_stats_of(encoder),
            "load_seconds": float(load_seconds),
            "encode_seconds": float(encode_seconds),
            "seconds": float(time.time() - t0),
        }
    )
    _dump(evidence)
    if args.out_dir:
        target = os.path.join(out_dir, DRILL_JSON)
        print(f"[INFO] G2 取证 -> {target} (sha256={_write_json(target, evidence)[:16]}...)")
    checks = {
        "len(vector) == dim": (int(len(first.vector)) == int(encoder.dim), True),
        "second_bitwise_equal": (bool(first_bytes == second_bytes), True),
        # 缓存相关判据在 --no-cache 下**不适用**（如实登记，不伪装成通过/失败）
        "cache_written": (bool(disk is not None), bool(cache_on)),
        "cache_bitwise_equal": (bool(disk == first_bytes), bool(cache_on)),
        "second_served_from_cache": (bool(second.cached), bool(cache_on)),
    }
    failed = [k for k, (ok, applicable) in checks.items() if applicable and not ok]
    evidence_checks = {
        k: {"passed": bool(ok), "applicable": bool(applicable)}
        for k, (ok, applicable) in checks.items()
    }
    evidence["checks"] = evidence_checks
    evidence["cache_enabled"] = bool(cache_on)
    if failed:
        print(f"[FAIL] G2 未通过：{failed}", file=sys.stderr)
        return 1
    if cache_on:
        print(
            f"[OK] G2：D={encoder.dim}，1 条文本编码 + 缓存写入 + 二次读取逐位一致，"
            f"key={key[:16]}... 耗时 {evidence['seconds']:.1f}s"
        )
    else:
        print(
            f"[OK] G2：D={encoder.dim}，1 条文本编码 + 二次读取逐位一致"
            f"（缓存未启用，缓存相关判据不适用）耗时 {evidence['seconds']:.1f}s"
        )
    return 0


# ---------------------------------------------------------------------------
# verify（G3）
# ---------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    """G3：指纹对账 + 重复编码逐位一致 + 与落盘缓存逐位比对。"""
    cfg = _encoder_config(args, cache_default=True)
    text = str(args.text)
    out_dir = str(args.out_dir) if args.out_dir else VERIFY_DIR
    drill_path = os.path.join(out_dir, DRILL_JSON)
    try:
        encoder = E.build_vectorizer(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 编码器不可用：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # --- ① fingerprint == 注册表 / 落盘声明 ----------------------------
    registry_fp = str(encoder.fingerprint())
    declaration = _declaration_of(encoder)
    declaration_source = "registry(注册表现场构造)"
    declared_fp = registry_fp
    if os.path.isfile(drill_path):
        with open(drill_path, "r", encoding="utf-8") as handle:
            recorded = dict(json.load(handle))
        # 落盘声明必须与当前请求的**编码器 + 角色**同口径才被采用，否则会被另一口径的
        # drill.json 误当作基准（现场实测过该错配）。
        same_encoder = str(recorded.get("encoder", "")) == str(cfg.resolved_name())
        same_role = str(recorded.get("role", "")) == str(cfg.role)
        if same_encoder and same_role and str(recorded.get("fingerprint", "")):
            declared_fp = str(recorded["fingerprint"])
            declaration_source = str(drill_path).replace("\\", "/")
        else:
            declaration_source = (
                f"{str(drill_path).replace(chr(92), '/')}"
                f"（口径不符：encoder={recorded.get('encoder')!r} role={recorded.get('role')!r}"
                f" -> 改用注册表现场构造的基准）"
            )
    check1 = bool(declared_fp) and str(declared_fp) == str(registry_fp)

    # --- ② 同文本两次编码逐位一致 -------------------------------------
    # 判据**只用 float32 裸字节**（修复 N1）：缓存命中会把条目里的 float32 还原成
    # Python float，与首次实算的 float64 值在末位可能不同，再叠 list 相等会产生
    # 「字节相同却判失败」的假阴性。逐位一致的语义就是字节一致。
    a = encoder.encode(text)
    b = encoder.encode(text)
    a_bytes, b_bytes = E.vector_bytes(a), E.vector_bytes(b)
    check2 = bool(a_bytes == b_bytes)

    # --- ③ 与落盘缓存逐位比对 -----------------------------------------
    key = _cache_key_of(encoder, text)
    applicable3 = bool(key)
    disk = E.EmbeddingCache(str(cfg.cache_dir)).read_bytes(key) if applicable3 else None
    check3 = bool(disk is not None) and bool(disk == a_bytes)

    out = {
        "encoder": str(cfg.resolved_name()),
        "role": str(cfg.role),
        "max_length": int(cfg.resolved_max_length()),
        "dim": int(encoder.dim),
        "declaration_source": declaration_source,
        "declaration": declaration,
        "check1_fingerprint_matches_declaration": {
            "passed": bool(check1),
            "applicable": True,
            "declared_fingerprint": str(declared_fp),
            "registry_fingerprint": str(registry_fp),
        },
        "check2_repeat_encode_bitwise_equal": {
            "passed": bool(check2),
            "applicable": True,
            "first_sha256": E.sha256_bytes(a_bytes),
            "second_sha256": E.sha256_bytes(b_bytes),
            "n_bytes": int(len(a_bytes)),
        },
        "check3_cache_bitwise_equal": {
            "passed": bool(check3),
            "applicable": bool(applicable3),
            "cache_key": str(key),
            "cache_bytes": int(len(disk)) if disk is not None else 0,
            "encode_sha256": E.sha256_bytes(a_bytes),
            "cache_sha256": E.sha256_bytes(disk) if disk is not None else "",
            "note": ("" if applicable3 else "缓存未启用 -> 本项不适用（不计入 passed）"),
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out["passed"] = bool(check1 and check2 and (check3 if applicable3 else True))
    _dump(out)
    if args.out_dir:
        target = os.path.join(out_dir, "verify.json")
        print(f"[INFO] G3 取证 -> {target} (sha256={_write_json(target, out)[:16]}...)")
    if not out["passed"]:
        print(
            "[FAIL] G3 未通过："
            f"① 指纹对账 {check1} / ② 重复编码逐位一致 {check2} / ③ 落盘缓存逐位比对 {check3}",
            file=sys.stderr,
        )
        return 1
    tail = (
        f"③ 与落盘缓存逐位比对通过（key={str(key)[:16]}...）"
        if applicable3 else "③ 缓存未启用 -> 不适用"
    )
    print(
        f"[OK] G3：① 指纹 == 声明 {str(registry_fp)[:16]}...；"
        f"② 重复编码逐位一致（{len(a_bytes)} 字节）；{tail}"
    )
    return 0


# ---------------------------------------------------------------------------
# boundary（G4）
# ---------------------------------------------------------------------------


def cmd_boundary(args: argparse.Namespace) -> int:
    """G4：边界处置自检（空 / 超长 / 模型缺失 / 维度不符）。

    口径（修复 D2）：``applicable=False`` 的用例**不计入** ``passed``，也**不**算失败
    —— 例如 ``zh-bag`` 不声明词元截断统计，故「超长 -> n_truncated > 0」对它不适用。
    """
    cfg = _encoder_config(args)
    cases = E.boundary_selftest(cfg, text=str(args.text))
    summary = E.summarize_selftest(cases)
    out = {
        "encoder": str(cfg.resolved_name()),
        "role": str(cfg.role),
        "n_cases": int(summary["n_cases"]),
        "n_applicable": int(summary["n_applicable"]),
        "n_inapplicable": int(summary["n_inapplicable"]),
        "n_passed": int(summary["n_passed"]),
        "inapplicable_cases": [str(c["case"]) for c in cases
                               if not bool(c.get("applicable", True))],
        "cases": cases,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _dump(out)
    if args.out_dir:
        target = os.path.join(str(args.out_dir), "boundary.json")
        print(f"[INFO] G4 取证 -> {target} (sha256={_write_json(target, out)[:16]}...)")
    failed = E.failed_selftest_cases(cases)
    if failed:
        print(f"[FAIL] G4 未通过：{failed}", file=sys.stderr)
        return 1
    suffix = (
        f"（另有 {summary['n_inapplicable']} 项对本实现不适用：{out['inapplicable_cases']}）"
        if summary["n_inapplicable"] else ""
    )
    print(
        f"[OK] G4：边界处置自检 {out['n_passed']}/{summary['n_applicable']} PASS"
        f"（共 {summary['n_cases']} 项）{suffix}"
    )
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _add_encoder_args(p: argparse.ArgumentParser, default: str) -> None:
    """编码器选择开关（唯一的选择实现入口）。"""
    p.add_argument("--encoder", type=str, default=default,
                   help=f"注册表键（默认 {default!r}）；registry 子命令列全部合法值")
    p.add_argument("--role", type=str, default=E.ROLE_QUESTION, choices=list(E.ROLES),
                   help="角色：question（max_length 512）/ text_line（8192）")
    p.add_argument("--max-length", type=int, default=0,
                   help="显式覆盖截断长度（0 = 用角色冻结口径）")
    p.add_argument("--source", type=str, default="",
                   help="模型来源（本地目录或 HF 仓库 id；空 = 注册表 model_id）")
    p.add_argument("--revision", type=str, default="",
                   help="固定 revision（空 = 注册表声明）")
    p.add_argument("--cache-dir", type=str, default=E.DEFAULT_EMB_CACHE_DIR)
    p.add_argument("--no-cache", dest="use_cache", action="store_false", default=True,
                   help="关闭嵌入缓存（hash 家族本就默认关闭）")
    p.add_argument("--force-cache", action="store_true",
                   help="强制开启嵌入缓存（hash 家族默认关闭；"
                        "开启后 ③ 与落盘缓存逐位比对对两家族都适用）")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--local-files-only", action="store_true",
                   help="只允许本地文件（无网络场景）")
    p.add_argument("--no-verify-weights", action="store_true",
                   help="跳过权重 SHA256（指纹显式记 unverified，不冒充已核实）")
    p.add_argument("--out-dir", type=str, default="",
                   help=f"取证报告目录（空 = {VERIFY_DIR}）")
    p.add_argument("--log-file", type=str, default="",
                   help="同时把 stdout 写入该 UTF-8（无 BOM）日志文件")


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 解析器。"""
    parser = argparse.ArgumentParser(
        prog="n3d_qa_learn.encoders_run",
        description="可插拔特征生成接口的验收入口（G1 模型核实 / G2 单条演练 / "
                    "G3 验证口径 / G4 边界处置）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("registry", help="列编码器注册表")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--log-file", type=str, default="")
    p.set_defaults(func=cmd_registry)

    p = sub.add_parser("fetch", help="从镜像下载模型（固定 revision）")
    p.add_argument("--encoder", type=str, default=E.ENCODER_BGE_M3)
    p.add_argument("--revision", type=str, default="")
    p.add_argument("--local-dir", type=str, default="")
    p.add_argument("--out-dir", type=str, default=VERIFY_DIR)
    p.add_argument("--log-file", type=str, default="")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("info", help="G1：模型加载 + 结构量与来源核实")
    _add_encoder_args(p, E.ENCODER_BGE_M3)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("drill", help="G2：单条端到端演练（含缓存写入与二次读取）")
    _add_encoder_args(p, E.ENCODER_BGE_M3)
    p.add_argument("--text", type=str, default="N3D 问答学习框架的单条端到端演练文本")
    p.add_argument("--reuse-only", action="store_true",
                   help="二次读取相位：不加载模型，只读落盘缓存（跨进程复用证明）")
    p.set_defaults(func=cmd_drill)

    p = sub.add_parser("verify", help="G3：指纹对账 / 重复编码 / 落盘缓存逐位比对")
    _add_encoder_args(p, E.ENCODER_BGE_M3)
    p.add_argument("--text", type=str, default="N3D 问答学习框架的单条端到端演练文本")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("boundary", help="G4：边界处置自检")
    _add_encoder_args(p, E.ENCODER_BGE_M3)
    p.add_argument("--text", type=str, default="N3D 问答学习框架的边界处置自检文本")
    p.set_defaults(func=cmd_boundary)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口（返回进程退出码）。"""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
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


if __name__ == "__main__":
    sys.exit(main())
