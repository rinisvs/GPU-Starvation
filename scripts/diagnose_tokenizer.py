#!/usr/bin/env python3
"""Investigate the tokenizer-load asymmetry between two model directories.

    python scripts/diagnose_tokenizer.py --config configs/desktop_rtx5060.yaml

Tokenizer construction is reproducibly slower for one container than another in
our sweep (0.63 s vs 0.25 s) despite both directories holding a `tokenizer.json`.
Since the tokenizer is identical content, the cause must be either the files
present, the configuration selected, or the cost of construction itself.

This script separates those three:

1. **Files.** Names, sizes and byte-identity of every tokenizer-relevant file.
   If the directories differ, the anomaly is an artifact of how they were
   created, not a property of the format.
2. **Configuration.** `tokenizer_class` and `use_fast`. A configuration that
   selects the Python (slow) tokenizer pays a large one-off construction cost on
   a 150k-token vocabulary, which would explain the gap entirely.
3. **Construction, timed directly.** Repeated `AutoTokenizer.from_pretrained`
   calls with the file cache warm, isolating construction from I/O.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpustarve.config import ExperimentConfig  # noqa: E402

TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json", "config.json",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]


def describe(path: Path) -> dict[str, tuple[int, str]]:
    found = {}
    for name in TOKENIZER_FILES:
        f = path / name
        if f.exists():
            found[name] = (f.stat().st_size, digest(f))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/desktop_rtx5060.yaml")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    dirs = [(f.name, Path(f.path)) for f in cfg.formats if Path(f.path).is_dir()]
    if len(dirs) < 2:
        print("Need at least two directory-based formats to compare.")
        return 1

    # ---- 1. files -------------------------------------------------------
    print("=" * 70)
    print("FILES")
    print("=" * 70)
    tables = {}
    for name, path in dirs:
        tables[name] = describe(path)
        print(f"\n{name}  ({path})")
        for fn, (size, dg) in sorted(tables[name].items()):
            print(f"  {fn:<28} {size:>12,}  sha={dg}")

    print("\n--- differences ---")
    (a_name, _), (b_name, _) = dirs[0], dirs[1]
    a, b = tables[a_name], tables[b_name]
    same = True
    for fn in sorted(set(a) | set(b)):
        if fn not in a:
            print(f"  {fn}: MISSING from {a_name}"); same = False
        elif fn not in b:
            print(f"  {fn}: MISSING from {b_name}"); same = False
        elif a[fn][1] != b[fn][1]:
            print(f"  {fn}: DIFFERENT content "
                  f"({a[fn][0]:,} vs {b[fn][0]:,} bytes)"); same = False
    if same:
        print("  none — every tokenizer file is byte-identical")

    # ---- 2. configuration ----------------------------------------------
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    for name, path in dirs:
        tc = path / "tokenizer_config.json"
        if not tc.exists():
            print(f"  {name:<14} no tokenizer_config.json")
            continue
        c = json.loads(tc.read_text(encoding="utf-8"))
        print(f"  {name:<14} tokenizer_class={c.get('tokenizer_class')!r:<28} "
              f"use_fast={c.get('use_fast')!r}")
        extra = {k: v for k, v in c.items()
                 if k in ("model_max_length", "clean_up_tokenization_spaces",
                          "add_prefix_space", "split_special_tokens")}
        if extra:
            print(f"                 {extra}")

    # ---- 3. construction cost, cache warm -------------------------------
    print("\n" + "=" * 70)
    print(f"CONSTRUCTION ({args.repeats} repeats, file cache warm)")
    print("=" * 70)
    from transformers import AutoTokenizer

    for name, path in dirs:
        AutoTokenizer.from_pretrained(str(path))  # warm, discarded
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            tok = AutoTokenizer.from_pretrained(str(path))
            times.append(time.perf_counter() - t0)
        mean = sum(times) / len(times)
        print(f"  {name:<14} {mean*1000:>7.1f} ms  "
              f"(min {min(times)*1000:.1f}, max {max(times)*1000:.1f})  "
              f"fast={tok.is_fast}  vocab={len(tok)}")

    print("\nIf construction times match here but differ in the sweep, the cost is\n"
          "I/O rather than construction, and the file layout on disk is the place\n"
          "to look next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
