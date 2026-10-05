"""n3d_qa/tools/snapshot_triviaqa_sha256.py -- pre/post-change SHA256 snapshot of TriviaQA products.

Purpose
-------
The zero-regression gate of the incremental extension: record ``(relative path, bytes, SHA256)`` of
every ``.npz`` under ``checkpoints/triviaqa/`` **before** the change and compare it afterwards.

Read-only by design: it never writes inside ``checkpoints/triviaqa/``; the snapshot lives in
``checkpoints/qa_learn/_snapshot/``.

Usage
-----
    python n3d_qa/tools/snapshot_triviaqa_sha256.py            # write the snapshot
    python n3d_qa/tools/snapshot_triviaqa_sha256.py --check    # compare against it (exit 1 on drift)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT: str = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)
TRIVIAQA_DIR: str = os.path.join(PROJECT_ROOT, "checkpoints", "triviaqa")
SNAPSHOT_DIR: str = os.path.join(PROJECT_ROOT, "checkpoints", "qa_learn", "_snapshot")
SNAPSHOT_PATH: str = os.path.join(SNAPSHOT_DIR, "triviaqa_npz_sha256.json")


def sha256_file(path: str, chunk_bytes: int = 1 << 22) -> str:
    """Whole-file SHA256 (chunked; memory use independent of file size)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(int(chunk_bytes)), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(root: str) -> List[Dict[str, Any]]:
    """Enumerate every ``.npz`` under ``root`` (sorted by relative path, deterministic).

    Parameters
    ----------
    root : str
        Product root directory.

    Returns
    -------
    List[Dict[str, Any]]
        ``[{"path": relative path, "size": bytes, "sha256": ...}, ...]``.
    """
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if not fn.lower().endswith(".npz"):
                continue
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, PROJECT_ROOT).replace("\\", "/")
            out.append({"path": rel, "size": int(os.path.getsize(fp)), "sha256": sha256_file(fp)})
    out.sort(key=lambda d: str(d["path"]))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: write or check the snapshot.

    Returns
    -------
    int
        0 = snapshot written, or the check found no drift; 1 = drift (regression) or missing file.
    """
    ap = argparse.ArgumentParser(description="TriviaQA product SHA256 snapshot / zero-regression check")
    ap.add_argument("--check", action="store_true", help="compare against the snapshot instead of writing it")
    ap.add_argument("--root", default=TRIVIAQA_DIR, help="product root (default checkpoints/triviaqa)")
    ap.add_argument("--snapshot", default=SNAPSHOT_PATH, help="snapshot JSON path")
    args = ap.parse_args(argv)

    items = collect(str(args.root))
    print("[snapshot] scanned %s: %d .npz file(s)" % (args.root, len(items)))
    for it in items:
        print("  %s  %12d  %s" % (it["sha256"], it["size"], it["path"]))

    if not args.check:
        os.makedirs(os.path.dirname(os.path.abspath(str(args.snapshot))), exist_ok=True)
        with open(str(args.snapshot), "w", encoding="utf-8") as fh:
            json.dump({"root": str(args.root), "items": items}, fh, ensure_ascii=False, indent=1)
        print("[snapshot] written: %s" % args.snapshot)
        return 0

    if not os.path.isfile(str(args.snapshot)):
        print("[snapshot] missing snapshot file %s" % args.snapshot, file=sys.stderr)
        return 1
    with open(str(args.snapshot), "r", encoding="utf-8") as fh:
        old = json.load(fh)
    old_items: Dict[str, Tuple[int, str]] = {
        str(d["path"]): (int(d["size"]), str(d["sha256"])) for d in old.get("items", [])
    }
    new_items: Dict[str, Tuple[int, str]] = {
        str(d["path"]): (int(d["size"]), str(d["sha256"])) for d in items
    }
    problems: List[str] = []
    for path in sorted(set(old_items) | set(new_items)):
        a, b = old_items.get(path), new_items.get(path)
        if a is None:
            problems.append("new file: %s" % path)
        elif b is None:
            problems.append("missing file: %s" % path)
        elif a != b:
            problems.append("SHA256/size changed: %s old=%s new=%s" % (path, a, b))
    if problems:
        print("[snapshot] [!] zero-regression gate BROKEN:")
        for p in problems:
            print("  " + p)
        return 1
    print("[snapshot] [OK] %d file(s) bit-identical to the snapshot (zero regression holds)"
          % len(new_items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())