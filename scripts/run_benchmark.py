#!/usr/bin/env python3
"""Run a cold-start sweep.

    python scripts/run_benchmark.py --config configs/desktop_rtx5060.yaml
    python scripts/run_benchmark.py --config configs/desktop_rtx5060.yaml --runs 3 --no-overhead-arm

Writes one JSON file to results/raw/ containing the config, full provenance,
and every individual run record. Aggregation happens separately in
scripts/make_tables.py, so raw data is never overwritten by analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpustarve.config import ExperimentConfig  # noqa: E402
from gpustarve.harness import run_sweep  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--out", default="results/raw", help="output directory")
    ap.add_argument("--runs", type=int, help="override repetitions per configuration")
    ap.add_argument("--no-overhead-arm", action="store_true",
                    help="skip the unsampled arm (faster, but hides observer effect)")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.runs:
        cfg.runs = args.runs
    if args.no_overhead_arm:
        cfg.overhead_arm = False

    print(f"Platform : {cfg.label}")
    print(f"Formats  : {', '.join(f.name for f in cfg.formats)}")
    print(f"Runs     : {cfg.runs} x {2 if cfg.overhead_arm else 1} arm(s)")
    print(f"Config   : {cfg.fingerprint()}\n")

    # Fail fast on missing weights rather than midway through a long sweep.
    for spec in cfg.formats:
        try:
            gb = spec.resolve_size_bytes() / 1e9
            print(f"  {spec.name:<14} {gb:6.3f} GB on disk")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  {spec.name:<14} ERROR: {exc}")
            return 1
    print()

    run_sweep(cfg, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
