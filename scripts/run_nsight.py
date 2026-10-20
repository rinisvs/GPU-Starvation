#!/usr/bin/env python3
"""Capture Nsight Systems traces of a cold start, one format at a time.

    python scripts/run_nsight.py --config configs/desktop_rtx5060.yaml
    python scripts/run_nsight.py --config configs/desktop_rtx5060.yaml --format pickle

Produces one ``.nsys-rep`` per format in ``results/traces/``, plus a CSV of NVTX
range durations extracted with ``nsys stats``.

**Why the traces matter.** The harness times phases with ``perf_counter`` on the
Python side. That tells you how long each phase took, but not what the GPU was
doing during it. The trace is the evidence for the central claim: during the
Pickle deserialize phase the CUDA hardware row shows *zero* kernel activity
while one CPU thread sits pinned in ``_pickle.loads()``. A number cannot show
that; a timeline can.

**Cross-validation.** ``--verify`` compares NVTX-derived phase durations against
the harness's own timings. They are independent measurements of the same
quantity and should agree within a few percent. A large gap means one of them is
wrong, and you want to know that before it reaches a reviewer.

This script re-invokes itself: the parent orchestrates ``nsys``, and the child
(``--inner``) performs exactly one load so the trace contains one clean cold
start and nothing else. Power sampling is disabled inside the trace, since the
sampler threads would appear as noise on the timeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpustarve.config import ExperimentConfig  # noqa: E402

# NVTX ranges emitted by loaders.py, in pipeline order.
PHASES = ["tokenizer", "deserialize", "transfer", "generate"]


# --------------------------------------------------------------------------
# Locating nsys
# --------------------------------------------------------------------------


def find_nsys(explicit: str | None = None) -> str:
    """Locate the nsys binary, including the usual Windows install roots."""
    if explicit:
        if not Path(explicit).exists():
            sys.exit(f"--nsys path does not exist: {explicit}")
        return explicit

    found = shutil.which("nsys") or shutil.which("nsys.exe")
    if found:
        return found

    roots = [
        Path(r"C:/Program Files/NVIDIA Corporation/Nsight Systems"),
        Path(r"C:/Program Files/NVIDIA/Nsight Systems"),
        Path("/opt/nvidia/nsight-systems"),
        Path("/usr/local/cuda/bin"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("**/nsys.exe", "**/bin/nsys"):
            for hit in sorted(root.glob(pattern), reverse=True):  # newest version first
                return str(hit)

    sys.exit(
        "Could not find `nsys`. Install NVIDIA Nsight Systems, or pass "
        "--nsys /path/to/nsys."
    )


# --------------------------------------------------------------------------
# Child: one clean cold start under the profiler
# --------------------------------------------------------------------------


def run_inner(config_path: str, fmt: str) -> int:
    from gpustarve.harness import measure_once

    cfg = ExperimentConfig.from_yaml(config_path)
    spec = next((f for f in cfg.formats if f.name == fmt), None)
    if spec is None:
        sys.exit(f"format '{fmt}' not present in {config_path}")

    # sampled=False: sampler threads would pollute the timeline we are here to read.
    result = measure_once(spec, cfg, run_index=1, sampled=False)

    print(json.dumps({
        "format": result.format,
        "ttft_s": result.ttft_s,
        "tokenizer_s": result.tokenizer_s,
        "deserialize_s": result.deserialize_s,
        "transfer_s": result.transfer_s,
        "generate_s": result.generate_s,
    }))
    return 0


# --------------------------------------------------------------------------
# Parent: orchestrate nsys, then extract NVTX ranges
# --------------------------------------------------------------------------


def default_trace() -> str:
    """Trace sources differ by platform.

    ``osrt`` (OS runtime — the blocking syscalls we care about) exists only on
    Linux. The Windows equivalent surface is ``wddm``, the display driver model
    scheduler. Passing a source the local nsys does not recognise aborts the
    whole capture, so the set is chosen per platform rather than assumed.
    """
    return "cuda,nvtx,wddm" if sys.platform == "win32" else "cuda,nvtx,osrt"


def profile_format(nsys: str, config_path: str, fmt: str, out_dir: Path,
                   trace: str, sample: str) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"trace_{fmt}"

    def build(trace_arg: str, sample_arg: str | None) -> list[str]:
        cmd = [nsys, "profile", f"--trace={trace_arg}"]
        if sample_arg and sample_arg != "none":
            cmd.append(f"--sample={sample_arg}")
        cmd += [
            "--cuda-memory-usage=true",
            "--force-overwrite=true",
            "-o", str(stem),
            sys.executable, str(Path(__file__).resolve()),
            "--inner", "--config", config_path, "--format", fmt,
        ]
        return cmd

    print(f"\n=== nsys profile: {fmt} ===")
    rep = Path(f"{stem}.nsys-rep")

    # Attempt the rich flag set, then fall back to the minimum that still
    # answers the question. nsys rejects the entire command on one bad flag,
    # and flag names drift between releases, so a capture is worth more than
    # the extra trace sources.
    attempts = [(trace, sample), ("cuda,nvtx", None)]
    for i, (t, s) in enumerate(attempts):
        cmd = build(t, s)
        print("  " + " ".join(cmd))
        res = subprocess.run(cmd, cwd=ROOT)
        if rep.exists():
            if i:
                print(f"  wrote {rep} (fell back to --trace={t})")
            else:
                print(f"  wrote {rep}")
            return rep
        print(f"  [warn] nsys exited {res.returncode} and produced no report"
              + ("; retrying with a minimal trace set" if i == 0 else ""))

    print(f"  [error] no report produced at {rep}")
    return None


def extract_nvtx(nsys: str, rep: Path) -> dict[str, float]:
    """Pull per-range NVTX durations out of a trace, in seconds.

    Report naming has changed across Nsight releases, so several are attempted.
    """
    for report in ("nvtx_sum", "nvtx_pushpop_sum", "nvtxsum"):
        proc = subprocess.run(
            [nsys, "stats", "--report", report, "--format", "csv", "--force-export=true", str(rep)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            continue

        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        start = next((i for i, ln in enumerate(lines) if "," in ln and "Range" in ln), None)
        if start is None:
            continue

        out: dict[str, float] = {}
        for row in csv.DictReader(lines[start:]):
            name = (row.get("Range") or "").strip()
            if name not in PHASES:
                continue
            for key in ("Total Time (ns)", "Total Time", "Duration (ns)"):
                if row.get(key):
                    try:
                        out[name] = float(str(row[key]).replace(",", "")) / 1e9
                    except ValueError:
                        pass
                    break
        if out:
            return out

    print(f"  [warn] could not extract NVTX ranges from {rep.name}; open it in the GUI")
    return {}


def _check_interpreter(raw_dir: str = "results/raw") -> bool:
    """Verify this interpreter matches the one that produced the latest sweep.

    Traces are only comparable to a sweep if both ran against the same installed
    packages. "Am I in a virtualenv?" is a proxy for that, and a poor one: a
    sweep legitimately run under the system interpreter must be traced under the
    system interpreter, and refusing to do so pushes the user toward a venv whose
    packages were never the ones measured.

    So compare against the recorded interpreter where one exists, and fall back
    to the virtualenv heuristic only for sweeps predating that field.
    """
    sweeps = sorted(Path(raw_dir).glob("sweep-*.json"), key=lambda p: p.stat().st_mtime)
    recorded = None
    if sweeps:
        try:
            prov = json.loads(sweeps[-1].read_text()).get("provenance", {})
            recorded = prov.get("python_executable")
        except (json.JSONDecodeError, OSError):
            recorded = None

    if recorded:
        if os.path.normcase(recorded) == os.path.normcase(sys.executable):
            print(f"interpreter matches sweep {sweeps[-1].name}")
            return True
        print(f"ERROR: interpreter does not match the most recent sweep.\n"
              f"       this process : {sys.executable}\n"
              f"       {sweeps[-1].name} : {recorded}\n"
              "       Traces would profile different packages than the sweep measured.\n"
              "       Re-run this command with that interpreter, or pass\n"
              "       --allow-system-python if the mismatch is deliberate.",
              file=sys.stderr)
        return False

    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        print("WARNING: no sweep records an interpreter, and this is not a virtualenv.\n"
              f"         Interpreter: {sys.executable}\n"
              "         Traces may profile different packages than the sweep used.\n"
              "         Override with --allow-system-python.", file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--format", help="profile only this format (default: all in config)")
    ap.add_argument("--out", default="results/traces")
    ap.add_argument("--nsys", help="explicit path to the nsys binary")
    ap.add_argument("--trace", default=None,
                    help=f"nsys trace sources (default: {default_trace()})")
    ap.add_argument("--sample", default="process-tree",
                    help="nsys CPU sampling mode, or 'none'")
    ap.add_argument("--verify", action="store_true",
                    help="compare NVTX durations against harness timings")
    ap.add_argument("--allow-system-python", action="store_true",
                    help="profile even if the interpreter cannot be matched to a sweep")
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.inner:
        return run_inner(args.config, args.format)

    if not _check_interpreter() and not args.allow_system_python:
        return 1

    cfg = ExperimentConfig.from_yaml(args.config)
    formats = [args.format] if args.format else [f.name for f in cfg.formats]
    nsys = find_nsys(args.nsys)
    print(f"nsys: {nsys}")
    print(f"python: {sys.executable}")

    out_dir = Path(args.out)
    summary: dict[str, dict[str, float]] = {}

    for fmt in formats:
        rep = profile_format(nsys, args.config, fmt, out_dir,
                             args.trace or default_trace(), args.sample)
        if rep is None:
            continue
        ranges = extract_nvtx(nsys, rep)
        if ranges:
            summary[fmt] = ranges
            total = sum(ranges.values())
            print(f"  NVTX phases: " + "  ".join(f"{k}={v:.3f}s" for k, v in ranges.items()))
            print(f"  NVTX total : {total:.3f}s")

    if summary:
        dest = out_dir / "nvtx_phases.json"
        dest.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {dest}")

    if args.verify and summary:
        print("\n=== NVTX vs harness timings ===")
        print("Independent measurements of the same phases; expect agreement "
              "within a few percent.\n")
        print(f"{'format':<14} {'phase':<13} {'NVTX (s)':>10} {'harness (s)':>12} {'delta':>9}")
        for fmt, ranges in summary.items():
            spec = next(f for f in cfg.formats if f.name == fmt)
            from gpustarve.harness import measure_once
            ref = measure_once(spec, cfg, run_index=1, sampled=False)
            harness = {
                "tokenizer": ref.tokenizer_s, "deserialize": ref.deserialize_s,
                "transfer": ref.transfer_s, "generate": ref.generate_s,
            }
            for phase in PHASES:
                nv, hv = ranges.get(phase), harness.get(phase, 0.0)
                if nv is None or not hv:
                    continue
                print(f"{fmt:<14} {phase:<13} {nv:>10.3f} {hv:>12.3f} "
                      f"{100 * (nv - hv) / hv:>8.1f}%")

    print("\nOpen a trace in the Nsight Systems GUI and look for:")
    print("  * the CUDA HW row showing zero kernels during 'deserialize' (pickle)")
    print("  * one CPU thread at 100% in _pickle.loads() over the same interval")
    print("  * that gap absent or much shorter for safetensors and gguf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
