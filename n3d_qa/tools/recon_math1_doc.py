"""n3d_qa/tools/recon_math1_doc.py -- on-site reconnaissance for Math1 and data/doc (read-only).

Purpose
-------
Before the incremental extension, produce the following facts FROM COMMANDS (never from memory):

* Math1 eight files: path / byte size / record count;
* per-file actual key set of the first record and field types (is ``answer`` a list? what is ``choices``?);
* per qtype record counts, answer value distribution before/after normalization, duplication;
* ``data/doc/*.md`` non-empty line counts and character composition (CJK / ASCII share).

Read-only: writes nothing; all output goes to stdout.

Usage
-----
    python n3d_qa/tools/recon_math1_doc.py --max-lines 0
    python n3d_qa/tools/recon_math1_doc.py --max-lines 3000
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import unicodedata
from typing import Any, Dict, Iterator, Tuple

PROJECT_ROOT: str = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)
MATH1_DIR: str = os.path.join(
    PROJECT_ROOT, "data", "kupasai", "math1",
    "Kupasai___HighQualityEducationCoTDataset-Math1", "data",
)
DOC_DIR: str = os.path.join(PROJECT_ROOT, "data", "doc")
SUBJECTS: Tuple[str, ...] = ("\u79bb\u6563\u6570\u5b66", "\u9ad8\u7b49\u6570\u5b66")
QTYPES: Tuple[str, ...] = ("\u5224\u65ad\u9898", "\u586b\u7a7a\u9898", "\u89e3\u7b54\u9898", "\u9009\u62e9\u9898")


def iter_records(path: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Stream (1-based line number, parsed dict) from a jsonl file; blank lines are skipped."""
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def norm_answer(s: str) -> str:
    """Normalization rule shown on site: NFKC -> drop all Unicode whitespace -> casefold."""
    return "".join(unicodedata.normalize("NFKC", s).split()).casefold()


def recon_math1(max_lines: int) -> None:
    """Recon the eight Math1 files (key sets / counts / answer distributions)."""
    for subject in SUBJECTS:
        for qtype in QTYPES:
            path = os.path.join(MATH1_DIR, subject, subject + "_" + qtype + ".jsonl")
            print("=" * 100)
            print("[FILE] %s  bytes=%d" % (os.path.relpath(path, PROJECT_ROOT), os.path.getsize(path)))
            n_lines = 0
            key_counter = collections.Counter()
            ans_raw = collections.Counter()
            ans_norm = collections.Counter()
            n_choice_lens = collections.Counter()
            first_printed = False
            for lineno, rec in iter_records(path):
                n_lines += 1
                keys = tuple(sorted(rec.keys()))
                key_counter[keys] += 1
                if not first_printed:
                    first_printed = True
                    print("[FIRST] line=%d" % lineno)
                    for k in keys:
                        v = rec[k]
                        vs = repr(v)
                        if len(vs) > 200:
                            vs = vs[:200] + "..."
                        print("        %r: %s = %s" % (k, type(v).__name__, vs))
                ans = rec.get("answer")
                if isinstance(ans, list):
                    for a in ans:
                        ans_raw[str(a)] += 1
                        ans_norm[norm_answer(str(a))] += 1
                elif isinstance(ans, str):
                    ans_raw[ans] += 1
                    ans_norm[norm_answer(ans)] += 1
                ch = rec.get("choices")
                if isinstance(ch, list):
                    n_choice_lens[len(ch)] += 1
                if max_lines and n_lines >= int(max_lines):
                    break
            print("[COUNT] records=%d (cap=%s)" % (n_lines, max_lines or "none"))
            print("[KEYS] key-set -> count:")
            for keys, cnt in key_counter.most_common():
                print("        %s -> %d" % (list(keys), cnt))
            print("[CHOICES] length distribution: %s" % dict(n_choice_lens))
            print("[ANSWERS] distinct raw=%d distinct normalized=%d (delta=%d)"
                  % (len(ans_raw), len(ans_norm), len(ans_raw) - len(ans_norm)))
            print("[ANSWERS] top10 by normalized frequency:")
            for val, cnt in ans_norm.most_common(10):
                shown = val if len(val) <= 40 else val[:40] + "..."
                print("        %6d  %r" % (cnt, shown))
            for thr in (2, 3, 5, 10):
                n_cls = sum(1 for _v, c in ans_norm.items() if c >= thr)
                n_smp = sum(c for _v, c in ans_norm.items() if c >= thr)
                print("[ANSWERS] freq >= %d: classes=%d samples=%d coverage=%.4f"
                      % (thr, n_cls, n_smp, n_smp / max(n_lines, 1)))


def recon_doc() -> None:
    """Recon data/doc/*.md non-empty line splitting and character composition."""
    print("=" * 100)
    total = 0
    for name in sorted(os.listdir(DOC_DIR)):
        if not name.lower().endswith(".md"):
            continue
        path = os.path.join(DOC_DIR, name)
        with open(path, "r", encoding="utf-8") as fh:
            raw_lines = fh.read().split("\n")
        nonempty = [ln for ln in raw_lines if ln.strip()]
        n_cjk = sum(1 for ln in nonempty for ch in ln if "\u4e00" <= ch <= "\u9fff")
        n_ascii = sum(1 for ln in nonempty for ch in ln if ch.isascii() and not ch.isspace())
        n_chars = sum(len(ln) for ln in nonempty)
        total += len(nonempty)
        print("[DOC] %s: physical=%d nonempty=%d chars=%d cjk=%d (%.4f) ascii_nonblank=%d (%.4f)"
              % (name, len(raw_lines), len(nonempty), n_chars, n_cjk, n_cjk / max(n_chars, 1),
                 n_ascii, n_ascii / max(n_chars, 1)))
        for i, ln in enumerate(nonempty[:2], start=1):
            print("        sample %d: %r" % (i, ln[:90]))
    print("[DOC] non-empty line total = %d" % total)


def main(argv: "List[str] | None" = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Math1 / data/doc on-site reconnaissance")
    ap.add_argument("--max-lines", type=int, default=0, help="cap records per file (0 = no cap)")
    args = ap.parse_args(argv)
    recon_math1(int(args.max_lines))
    recon_doc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())