#!/usr/bin/env python3
"""Verify power telemetry before trusting any energy number.

    python scripts/check_power.py

Reports which sampler each subsystem resolves to, the sensor it names, and a
short live capture. Run this before a sweep: if the CPU source says MODELLED,
your energy figures are a model, and on the desktop platform CPU energy is the
majority of the total.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpustarve.power import (  # noqa: E402
    EstimatedCpuSampler, HwinfoSampler, NvmlGpuSampler,
    RaplLinuxSampler, autodetect_cpu_sampler,
)

DURATION_S = 3.0


def probe(name: str, factory) -> None:
    print(f"[{name}]")
    try:
        sampler = factory()
    except Exception as exc:
        print(f"  unavailable: {type(exc).__name__}: {exc}\n")
        return
    print(f"  source: {sampler.source}")
    with sampler:
        time.sleep(DURATION_S)
    t = sampler.trace
    if t.n_samples < 2:
        print(f"  PROBLEM: only {t.n_samples} sample(s) in {DURATION_S}s -- "
              "sensor is not updating at the requested rate\n")
        return
    rate = t.n_samples / DURATION_S
    requested = 1.0 / 0.1
    print(f"  samples: {t.n_samples} ({rate:.1f}/s, requested {requested:.0f}/s)  "
          f"mean={t.mean_w:.2f}W  peak={t.peak_w:.2f}W  idle energy={t.energy_j():.2f}J")
    print(f"  read latency: {t.mean_read_latency_s * 1000:.1f} ms/call"
          f"   missed deadlines: {t.missed_deadlines}")

    if len(set(t.watts)) == 1:
        print("  PROBLEM: every reading identical -- sensor is likely stuck")

    if rate < 0.5 * requested:
        print(f"  PROBLEM: achieved rate is {100 * rate / requested:.0f}% of requested.")
        print(f"           At {rate:.1f}/s a 0.7 s load window yields ~{rate * 0.7:.0f} samples,")
        print("           which is too few to integrate. See the note printed below.")
    print()


def main() -> int:
    print(f"Sampling each source for {DURATION_S:.0f}s at idle.\n")
    probe("GPU / NVML", lambda: NvmlGpuSampler(0.1))
    probe("CPU / HWiNFO (Windows)", lambda: HwinfoSampler(0.1))
    probe("CPU / RAPL (Linux)", lambda: RaplLinuxSampler(0.1))
    probe("CPU / modelled fallback", lambda: EstimatedCpuSampler(0.1))

    chosen = autodetect_cpu_sampler(0.1)
    print("=" * 68)
    print("Sample-rate guidance")
    print("  Your fastest configuration (GGUF) loads in well under a second.")
    print("  To integrate energy over that window you need >=10 samples, i.e.")
    print("  a sustained rate above ~15/s. If NVML cannot reach that, either")
    print("  lower power_interval_s and re-check, or treat energy for the fast")
    print("  formats as coarse and say so.")
    print()
    print(f"autodetect will use: {type(chosen).__name__}")
    print(f"  {chosen.source}")
    if isinstance(chosen, EstimatedCpuSampler):
        print("\n  WARNING: CPU energy will be MODELLED, not measured.")
        print("  On the desktop platform CPU energy is the majority of the total,")
        print("  so this materially affects every energy claim. Either start")
        print("  HWiNFO64 with Shared Memory Support enabled, or disclose the")
        print("  model explicitly in the paper.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
