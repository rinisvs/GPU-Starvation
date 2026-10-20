"""Unified single-pass measurement harness.

The design constraint that motivates this module: **latency and energy must be
measured in the same run.** Measuring them separately produces two datasets
that disagree, because power sampling perturbs the very window being timed. The
previous codebase hit exactly that, and the resulting numbers differed by up to
44% on shared quantities.

Two mechanisms address it here:

1. Latency and energy come from one pass. There is no second sweep to reconcile.
2. An optional *overhead arm* runs each configuration twice, once with samplers
   active and once without, so the perturbation is reported as a measured
   quantity rather than assumed away.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ._power_state import keep_system_awake
from .config import ExperimentConfig, FormatSpec, Provenance
from .loaders import LOADERS, PhaseTimings, drop_caches
from .power import PowerSampler, NvmlGpuSampler, autodetect_cpu_sampler


@dataclass
class RunResult:
    """One cold-start measurement."""

    format: str
    run_index: int
    sampled: bool

    # Latency
    ttft_s: float
    load_s: float
    tokenizer_s: float
    deserialize_s: float
    transfer_s: float
    generate_s: float

    # Size and throughput. Both normalizations are recorded, always, because
    # reporting one in a figure and the other in prose is how datasets diverge.
    on_disk_bytes: int
    reference_bytes: int
    throughput_ondisk_gbs: float
    throughput_reference_gbs: float

    # Energy. NaN when unsampled -- never 0.0, which would read as "no energy used".
    gpu_energy_j: float = float("nan")
    cpu_energy_j: float = float("nan")
    total_energy_j: float = float("nan")
    gpu_mean_w: float = float("nan")
    gpu_peak_w: float = float("nan")
    cpu_mean_w: float = float("nan")
    cpu_peak_w: float = float("nan")

    # Wall-clock start, so a gap caused by sleep or an interruption stays visible
    # in the raw records even when the durations themselves look plausible.
    started_utc: str = ""

    # Self-instrumentation
    gpu_samples: int = 0
    cpu_samples: int = 0
    sampler_busy_s: float = 0.0
    gpu_source: str = ""
    cpu_source: str = ""

    notes: list[str] = field(default_factory=list)


def _throughput_gbs(num_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return float("nan")
    return (num_bytes / 1e9) / seconds


def measure_once(
    spec: FormatSpec,
    cfg: ExperimentConfig,
    run_index: int,
    sampled: bool,
) -> RunResult:
    """Execute one cold start and return its full measurement record."""
    loader = LOADERS.get(spec.name)
    if loader is None:
        raise KeyError(f"no loader registered for format '{spec.name}'")

    on_disk = spec.resolve_size_bytes()
    drop_caches()
    time.sleep(cfg.cooldown_s)

    gpu_sampler: PowerSampler | None = None
    cpu_sampler: PowerSampler | None = None
    if sampled:
        try:
            gpu_sampler = NvmlGpuSampler(cfg.power_interval_s)
        except Exception:
            gpu_sampler = None
        cpu_sampler = autodetect_cpu_sampler(cfg.power_interval_s)

    # ExitStack rather than nested `with`: whichever samplers exist are started,
    # and none can be constructed-but-never-started. A sampler that was created
    # but not entered would leave an empty trace that reads as zero energy.
    model: Any = None
    with ExitStack() as stack:
        for sampler in (gpu_sampler, cpu_sampler):
            if sampler is not None:
                stack.enter_context(sampler)
        model, phases = loader(spec, cfg)

    result = _assemble(
        spec, cfg, run_index, sampled, phases, on_disk, gpu_sampler, cpu_sampler
    )

    del model
    drop_caches()
    return result


def _assemble(
    spec: FormatSpec,
    cfg: ExperimentConfig,
    run_index: int,
    sampled: bool,
    phases: PhaseTimings,
    on_disk: int,
    gpu_sampler: PowerSampler | None,
    cpu_sampler: PowerSampler | None,
) -> RunResult:
    load_s = phases.load_total
    r = RunResult(
        format=spec.name,
        run_index=run_index,
        sampled=sampled,
        ttft_s=round(phases.ttft, 4),
        load_s=round(load_s, 4),
        tokenizer_s=round(phases.tokenizer, 4),
        deserialize_s=round(phases.deserialize, 4),
        transfer_s=round(phases.transfer, 4),
        generate_s=round(phases.generate, 4),
        on_disk_bytes=on_disk,
        reference_bytes=cfg.reference_bytes,
        throughput_ondisk_gbs=round(_throughput_gbs(on_disk, load_s), 4),
        throughput_reference_gbs=round(_throughput_gbs(cfg.reference_bytes, load_s), 4),
        notes=list(phases.notes),
    )
    r.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if gpu_sampler is not None:
        t = gpu_sampler.trace
        r.gpu_energy_j = round(t.energy_j(), 4)
        r.gpu_mean_w = round(t.mean_w, 4)
        r.gpu_peak_w = round(t.peak_w, 4)
        r.gpu_samples = t.n_samples
        r.gpu_source = t.source
        r.sampler_busy_s += t.sampler_busy_s
        # The fastest formats have the shortest windows, so poor sample
        # resolution biases exactly the comparison being made.
        warn = t.resolution_warning(phases.ttft)
        if warn:
            r.notes.append(f"GPU energy resolution: {warn}")

    if cpu_sampler is not None:
        t = cpu_sampler.trace
        r.cpu_energy_j = round(t.energy_j(), 4)
        r.cpu_mean_w = round(t.mean_w, 4)
        r.cpu_peak_w = round(t.peak_w, 4)
        r.cpu_samples = t.n_samples
        r.cpu_source = t.source
        r.sampler_busy_s += t.sampler_busy_s
        if "MODELLED" in t.source:
            r.notes.append("CPU energy is MODELLED, not measured -- disclose in any writeup")
        warn = t.resolution_warning(phases.ttft)
        if warn:
            r.notes.append(f"CPU energy resolution: {warn}")

    energies = [e for e in (r.gpu_energy_j, r.cpu_energy_j) if e == e]  # drop NaN
    r.total_energy_j = round(sum(energies), 4) if energies else float("nan")
    r.sampler_busy_s = round(r.sampler_busy_s, 4)
    return r


def run_sweep(cfg: ExperimentConfig, out_dir: str | Path) -> list[RunResult]:
    """Run the full experiment and persist raw records plus provenance."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prov = Provenance.collect()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{cfg.fingerprint()}-{stamp}"

    arms = [True, False] if cfg.overhead_arm else [cfg.sample_power]
    results: list[RunResult] = []

    with keep_system_awake() as awake:
        if awake:
            print("System sleep suppressed for the duration of this sweep.\n", flush=True)
        elif sys.platform == "win32":
            print("WARNING: could not suppress system sleep; disable it manually.\n", flush=True)
        results = _sweep_body(cfg, arms)

    payload = {
        "config": {**asdict(cfg), "fingerprint": cfg.fingerprint()},
        "provenance": prov.to_dict(),
        # File inventory per format: the fingerprint hashes the configuration,
        # not the weights, so two sweeps can share a fingerprint while loading
        # different files. This makes that visible.
        "model_inventory": {f.name: f.inventory() for f in cfg.formats},
        "results": [asdict(r) for r in results],
    }
    path = out / f"sweep-{tag}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {len(results)} records to {path}")
    return results


def _sweep_body(cfg: ExperimentConfig, arms: list[bool]) -> list[RunResult]:
    """Run every format x arm x repetition, interleaved on both axes.

    Nothing here is blocked. Running all of one condition before the next
    confounds that condition with elapsed time: any transient -- a background
    scan, a thermal excursion, a clock change -- lands entirely on whichever
    condition happened to be running, and is then indistinguishable from a
    property of it.

    That failure has occurred twice in this project. Blocked *arms* produced a
    spurious +98% sampling overhead. Blocked *formats* let a ~50 s disturbance
    inflate one format's standard deviation sevenfold while the formats that ran
    afterwards stayed clean. Both axes are therefore interleaved.
    """
    results: list[RunResult] = []

    # Warm up every format before any measurement, rather than immediately
    # before that format's block. The first load of a format pays one-off costs
    # -- CUDA context creation, kernel JIT, lazy imports, allocator growth --
    # that belong to no condition and would dominate the mean at small n.
    for spec in cfg.formats:
        for w in range(cfg.warmup):
            print(f"[{spec.name:<14}] warmup    {w + 1}/{cfg.warmup} (discarded)", flush=True)
            try:
                measure_once(spec, cfg, run_index=0, sampled=False)
            except Exception as exc:
                print(f"    FAILED: {exc}", flush=True)

    n_fmt = len(cfg.formats)
    for i in range(1, cfg.runs + 1):
        # Rotate which format leads each repetition, so no format is
        # systematically first, and alternate arm order, so neither arm is
        # systematically first within a pair.
        shift = (i - 1) % n_fmt if n_fmt else 0
        specs = cfg.formats[shift:] + cfg.formats[:shift]
        order = arms if i % 2 else list(reversed(arms))

        for spec in specs:
            for sampled in order:
                label = "sampled" if sampled else "unsampled"
                print(f"[{spec.name:<14}] {label:<9} run {i}/{cfg.runs}", flush=True)
                try:
                    results.append(measure_once(spec, cfg, i, sampled))
                except Exception as exc:  # keep the sweep alive; record the gap
                    print(f"    FAILED: {exc}", flush=True)
    return results
