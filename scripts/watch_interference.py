#!/usr/bin/env python3
"""Identify the background process that is disturbing a sweep.

Run this in a second terminal, at the same time as the benchmark:

    python scripts/watch_interference.py                 # terminal 2, start first
    python scripts/run_benchmark.py --config ... --runs 20   # terminal 1

Then correlate the log with the sweep's flagged runs:

    python scripts/watch_interference.py --analyze \\
        results/raw/interference-<stamp>.csv results/raw/sweep-<...>.json

Motivation. A sweep whose runs are bimodal -- a fast mode and a slow mode
recurring on a timer -- is being disturbed by something outside the experiment.
Guessing at the culprit wastes more time than measuring it, and antivirus,
search indexers, sync clients and update services all present the same symptom.

The sampler records per-process CPU time and disk read deltas once a second.
The analyser then compares each process's activity *inside* the sweep's
disturbance windows against its own baseline outside them, so a process that is
merely busy throughout is not mistaken for one that is busy at exactly the wrong
moments. This is a paired comparison against the process itself, which is why it
can single out a culprit among dozens of noisy candidates.

Excludes the benchmark's own process tree, which is by definition busy when the
sweep is busy and would otherwise top every list.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SELF_NAMES = {"python.exe", "python", "python3", "nsys.exe", "nsys"}


def sample(interval_s: float, duration_s: float, out_path: Path) -> int:
    try:
        import psutil
    except ImportError:
        print("psutil is required:  pip install psutil", file=sys.stderr)
        return 2

    prev: dict[int, tuple[float, int]] = {}
    rows = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["utc", "pid", "name", "cpu_s_delta", "read_mb_delta"])
        deadline = time.monotonic() + duration_s
        print(f"Sampling every {interval_s:g}s for {duration_s / 60:.0f} min -> {out_path}")
        print("Start the benchmark now. Ctrl-C to stop early.\n")

        try:
            while time.monotonic() < deadline:
                t_wall = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                cur: dict[int, tuple[float, int]] = {}
                for p in psutil.process_iter(["pid", "name"]):
                    try:
                        with p.oneshot():
                            cpu = sum(p.cpu_times()[:2])
                            try:
                                read = p.io_counters().read_bytes
                            except (psutil.AccessDenied, AttributeError):
                                read = 0
                        cur[p.pid] = (cpu, read)
                        if p.pid in prev:
                            d_cpu = cpu - prev[p.pid][0]
                            d_read = (read - prev[p.pid][1]) / 1e6
                            # Only record activity; idle processes are noise.
                            if d_cpu > 0.01 or d_read > 0.5:
                                w.writerow([t_wall, p.pid, p.info["name"],
                                            f"{d_cpu:.3f}", f"{d_read:.2f}"])
                                rows += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                prev = cur
                fh.flush()
                time.sleep(interval_s)
        except KeyboardInterrupt:
            print("\nstopped early")

    print(f"\nWrote {rows} activity records to {out_path}")
    return 0


def windows_from_sweep(sweep: Path, k: float = 3.0, rel: float = 0.10,
                       pad_s: float = 8.0) -> list[tuple[dt.datetime, dt.datetime]]:
    """Time windows around runs that deviate from their format's median."""
    payload = json.loads(sweep.read_text())
    recs = [r for r in payload["results"] if r.get("sampled") and r.get("started_utc")]
    by_fmt: dict[str, list] = defaultdict(list)
    for r in recs:
        by_fmt[r["format"]].append(r)

    spans = []
    for _, rs in by_fmt.items():
        vals = sorted(r["ttft_s"] for r in rs)
        if len(vals) < 4:
            continue
        mid = len(vals) // 2
        median = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
        devs = sorted(abs(v - median) for v in vals)
        mad = devs[mid] if len(devs) % 2 else (devs[mid - 1] + devs[mid]) / 2
        sigma = 1.4826 * mad
        if sigma <= 0 or median <= 0:
            continue
        for r in rs:
            dev = abs(r["ttft_s"] - median)
            if dev / sigma > k and dev / median > rel:
                start = dt.datetime.strptime(r["started_utc"], "%Y-%m-%dT%H:%M:%SZ")
                # started_utc is stamped at the end of the load, so look backwards.
                spans.append((start - dt.timedelta(seconds=r["ttft_s"] + pad_s),
                              start + dt.timedelta(seconds=pad_s)))
    return sorted(spans)


def analyze(csv_path: Path, sweep: Path) -> int:
    spans = windows_from_sweep(sweep)
    if not spans:
        print("No disturbed runs in that sweep — nothing to correlate.")
        return 0

    def disturbed(ts: dt.datetime) -> bool:
        return any(a <= ts <= b for a, b in spans)

    inside: dict[str, list[float]] = defaultdict(list)
    outside: dict[str, list[float]] = defaultdict(list)
    secs_in: set = set()
    secs_out: set = set()

    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                ts = dt.datetime.strptime(row["utc"], "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue
            name = row["name"] or "?"
            if name in SELF_NAMES:
                continue
            load = float(row["cpu_s_delta"]) + float(row["read_mb_delta"]) / 200.0
            if disturbed(ts):
                inside[name].append(load)
                secs_in.add(ts)
            else:
                outside[name].append(load)
                secs_out.add(ts)

    n_in, n_out = max(len(secs_in), 1), max(len(secs_out), 1)
    print(f"{len(spans)} disturbance window(s); "
          f"{len(secs_in)} disturbed seconds vs {len(secs_out)} quiet seconds\n")

    scored = []
    for name in set(inside) | set(outside):
        rate_in = sum(inside.get(name, [])) / n_in
        rate_out = sum(outside.get(name, [])) / n_out
        if rate_in < 0.02:
            continue
        ratio = rate_in / rate_out if rate_out > 1e-9 else float("inf")
        scored.append((ratio, rate_in, rate_out, name))
    scored.sort(reverse=True)

    print(f"{'process':<34}{'disturbed':>11}{'quiet':>10}{'ratio':>9}")
    print("-" * 64)
    for ratio, r_in, r_out, name in scored[:15]:
        rs = "only" if ratio == float("inf") else f"{ratio:.1f}x"
        print(f"{name[:33]:<34}{r_in:>11.3f}{r_out:>10.3f}{rs:>9}")

    print("\nActivity is CPU-seconds plus disk reads (200 MB counted as 1 s), per\n"
          "wall-clock second. A process near the top ran markedly harder while the\n"
          "sweep was disturbed than while it was clean — that is your culprit.\n"
          "Ratios close to 1.0 are background load, not interference.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", nargs=2, metavar=("CSV", "SWEEP_JSON"),
                    help="correlate a log with a sweep's disturbed runs")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--minutes", type=float, default=15.0)
    args = ap.parse_args()

    if args.analyze:
        return analyze(Path(args.analyze[0]), Path(args.analyze[1]))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    return sample(args.interval, args.minutes * 60,
                  Path("results/raw") / f"interference-{stamp}.csv")


if __name__ == "__main__":
    raise SystemExit(main())
