#!/usr/bin/env python3
"""Quantify GPU occupancy and DMA throughput from Nsight traces.

    python scripts/analyze_traces.py
    python scripts/analyze_traces.py --traces results/traces --out results/processed

Reads the ``.sqlite`` databases that ``nsys`` exports alongside each
``.nsys-rep`` and answers two questions the phase timings cannot:

1. **Was the accelerator idle, and during which phase?** Kernel and memcpy
   activity is intersected with the NVTX phase ranges, so occupancy is reported
   per phase rather than as a single average.

2. **Why is one format's transfer slower?** Copy count, total bytes and the
   size distribution are compared across formats. When those are identical and
   the achieved bandwidth is not, the difference cannot be attributed to
   transfer volume or fragmentation — it has to be the source memory.

**Wall-clock times in a trace are inflated by instrumentation** and must not be
quoted as latency; use the sweep for that. Comparisons *between* traces captured
under the same profiler remain valid, and the bandwidth ratio is one.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PHASES = ("tokenizer", "deserialize", "transfer", "generate")
BUCKETS = [(0, 1 << 20, "< 1 MB"), (1 << 20, 8 << 20, "1-8 MB"),
           (8 << 20, 64 << 20, "8-64 MB"), (64 << 20, 1 << 40, "> 64 MB")]


def phase_ranges(db: sqlite3.Connection) -> list[tuple[str, int, int]]:
    rows = db.execute("""
        SELECT COALESCE(e.text, s.value), e.start, e.end
        FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id
        WHERE e.end IS NOT NULL ORDER BY e.start
    """).fetchall()
    return [(n, a, b) for n, a, b in rows if n in PHASES]


def has_table(db: sqlite3.Connection, name: str) -> bool:
    """Whether a CUPTI table exists in this export.

    nsys only creates a CUPTI activity table when at least one such activity was
    recorded. A trace of a process that launched no kernels and issued no copies
    therefore has *no* kernel or memcpy table at all, and querying one raises
    OperationalError.

    That absence is a finding, not a failure: it means the GPU did no work in
    the traced interval. It must be reported rather than crashed on -- and
    crucially, it must not prevent the other formats from being analysed.
    """
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def gpu_busy_ns(db: sqlite3.Connection, t0: int, t1: int) -> int:
    """GPU-active nanoseconds overlapping a window.

    Kernels and copies are summed separately; they rarely overlap on a single
    stream, so this slightly over-counts in the worst case and is therefore a
    conservative estimate of *idle* time.
    """
    total = 0
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        if not has_table(db, table):
            continue
        for s, e in db.execute(
            f"SELECT start, end FROM {table} WHERE end > ? AND start < ?", (t0, t1)
        ):
            total += min(e, t1) - max(s, t0)
    return total


def report(traces: Path) -> str:
    dbs = sorted(traces.glob("trace_*.sqlite"))
    if not dbs:
        raise SystemExit(f"No trace_*.sqlite in {traces}. Run scripts/run_nsight.py first.")

    out: list[str] = ["# Nsight trace analysis\n"]
    out.append(
        "> Wall-clock times here are inflated by instrumentation and are **not** "
        "latency measurements — see `results/processed/` for those. What the trace "
        "establishes is *where the accelerator was idle* and *how fast the copies "
        "actually ran*.\n"
    )

    out.append("\n## GPU occupancy by phase\n")
    out.append("| Format | Phase | Wall (s) | GPU busy (s) | Occupancy |\n|---|---|---|---|---|\n")
    for path in dbs:
        fmt = path.stem.replace("trace_", "")
        db = sqlite3.connect(path)
        for name, a, b in phase_ranges(db):
            wall = (b - a) / 1e9
            busy = gpu_busy_ns(db, a, b) / 1e9
            pct = 100 * busy / wall if wall else 0.0
            flag = " **idle**" if pct < 1 else ""
            out.append(f"| {fmt} | {name} | {wall:.3f} | {busy:.3f} | {pct:.1f}%{flag} |\n")

    missing: list[str] = []
    out.append("\n## Transfer characteristics\n")
    out.append("| Format | Copies | Bytes | Mean copy | Copy time (s) | Achieved |\n"
               "|---|---|---|---|---|---|\n")
    for path in dbs:
        fmt = path.stem.replace("trace_", "")
        db = sqlite3.connect(path)
        if not has_table(db, "CUPTI_ACTIVITY_KIND_MEMCPY"):
            out.append(f"| {fmt} | **0** | — | — | — | **no device copies recorded** |\n")
            missing.append(fmt)
            continue
        n, tot, dur = db.execute(
            "SELECT COUNT(*), SUM(bytes), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ).fetchone()
        if not n:
            out.append(f"| {fmt} | **0** | — | — | — | **no device copies recorded** |\n")
            missing.append(fmt)
            continue
        out.append(f"| {fmt} | {n} | {tot/1e9:.3f} GB | {tot/n/1e6:.2f} MB | "
                   f"{dur/1e9:.3f} | **{tot/dur:.2f} GB/s** |\n")

    out.append("\n## Copy size distribution\n")
    out.append("| Format | " + " | ".join(l for _, _, l in BUCKETS) + " |\n")
    out.append("|---" * (len(BUCKETS) + 1) + "|\n")
    for path in dbs:
        fmt = path.stem.replace("trace_", "")
        db = sqlite3.connect(path)
        if not has_table(db, "CUPTI_ACTIVITY_KIND_MEMCPY"):
            continue
        cells = []
        for lo, hi, _ in BUCKETS:
            c, b = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE bytes >= ? AND bytes < ?", (lo, hi)).fetchone()
            cells.append(f"{c} ops / {b/1e9:.3f} GB" if c else "—")
        out.append(f"| {fmt} | " + " | ".join(cells) + " |\n")

    out.append(
        "\n## Reading this\n\n"
        "Where two formats issue the **same number of copies, of the same sizes, "
        "totalling the same bytes**, but achieve different bandwidth, the transfer "
        "work is identical and the difference lies in the source memory: a copy "
        "whose source pages are not yet resident stalls the DMA engine on page "
        "faults served from storage, so the measured rate reflects disk, not the "
        "interconnect.\n\n"
        "Phases at ~0% occupancy are intervals where the accelerator was powered "
        "but issuing no work.\n"
    )
    if missing:
        out.append(
            "\n> **Warning — no GPU activity recorded for: "
            + ", ".join(f"`{m}`" for m in missing)
            + "**.\n>\n"
            "> nsys creates a CUPTI activity table only when such an activity\n"
            "> occurred, so an absent table means the traced process launched no\n"
            "> kernels and issued no device copies at all. Weights cannot reach\n"
            "> VRAM without copies, so either the runtime fell back to CPU for this\n"
            "> run, or it uploaded through a path CUPTI does not instrument.\n>\n"
            "> Resolve this before quoting *any* number for that format. Check\n"
            "> `llama_supports_gpu_offload()` and that `n_gpu_layers` took effect;\n"
            "> `scripts/preflight.py` tests exactly this.\n"
        )
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default="results/traces")
    ap.add_argument("--out", default="results/processed")
    args = ap.parse_args()

    text = report(Path(args.traces))
    print(text)
    dest = Path(args.out) / "trace_analysis.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    print(f"\nWrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
