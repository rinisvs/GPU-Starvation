# GPU Starvation

Cold-start latency and energy measurement for edge LLM inference.

Before a language model can answer anything, several gigabytes of weights have
to travel from storage into accelerator-visible memory. On a server this happens
once and is amortised over thousands of requests. At the edge, memory is scarce
enough that models are evicted when idle and reloaded on demand — so the cold
path is not an edge case, it is the normal operating condition.

This repository measures that path: how long it takes, how much energy it costs,
and which stage is responsible.

**The headline finding.** Loading the same 1.5B model from PyTorch Pickle rather
than SafeTensors costs 1.31× the time and 1.23× the energy. Nsight traces show
both formats issue *byte-for-byte identical* transfer work — 355 copies,
3.088 GB, the same size distribution, the same 1857 kernels through the same
4943 runtime calls — yet Pickle achieves **2.52 GB/s against SafeTensors'
14.45 GB/s**. The transfer work is not the variable. The state of the source
memory when the DMA engine runs is.

---

## Design principles

Measurement code is easy to write and hard to trust. Four rules are enforced
structurally here, so that violating them requires deliberately editing the
harness rather than merely forgetting.

| Principle | How it is enforced |
|---|---|
| Latency and energy come from **one pass** | `harness.measure_once()` produces both from a single load. There is no second sweep to reconcile. |
| Sampling overhead is **measured, not assumed** | `overhead_arm: true` runs every configuration with and without power sampling; the delta is reported before any result. |
| Nothing measurable is **hardcoded** | `FormatSpec.resolve_size_bytes()` reads sizes from disk. Configs have no size field to get wrong. |
| Instrumentation **names its own sensor** | Every `PowerSampler.source` string propagates into the raw records and every generated report. A modelled value cannot be cited as a measured one. |

Two supporting habits follow from these. A failed sampler yields `NaN`, never
`0.0`, so a broken sensor cannot masquerade as a zero-energy result. And every
sweep records its own provenance — git SHA, driver version, torch build,
hostname, timestamp — because results without it become unusable within weeks.

---

## Quick start

Assumes only Python 3.10+ and an NVIDIA driver.

```powershell
powershell -ExecutionPolicy Bypass -File bootstrap.ps1     # Windows
```
```bash
bash bootstrap.sh                                          # Linux / macOS
```

Creates `.venv`, installs dependencies from a CUDA index, runs the tests, then
runs the preflight. It does **not** download model weights — point a config at
wherever yours live.

Then, at any time:

```bash
python scripts/preflight.py --config configs/desktop_rtx5060.yaml
```

One command answers *can this machine produce a publishable measurement?*
Exit 0 means ready. Every failure prints a specific remedy.

### Telemetry setup

**CPU power** is platform-specific, and the harness will not pretend otherwise:

- **Linux** — `RaplLinuxSampler` reads `/sys/class/powercap/intel-rapl:0/energy_uj`. Genuine RAPL; cite it as such.
- **Windows** — `HwinfoSampler` reads HWiNFO64's shared memory via ctypes. Run HWiNFO64 in Sensors-only mode, enable Settings → General/User Interface → **Shared Memory Support**, and set the sensor polling period to **100 ms** (the 2000 ms default cannot resolve a sub-second load).
- **Fallback** — `EstimatedCpuSampler` *models* power from utilisation. Its `source` string says so, and that string appears in every report.

CPU energy is the majority of the total on a desktop or laptop, so a modelled
value would propagate into every energy claim. The preflight fails on it rather
than warning.

**GPU power** via NVML needs nothing beyond the driver, but the preflight
measures the *achievable sample rate* both idle and under load. A sub-second
load window needs roughly ten samples to integrate, and an idle laptop GPU can
be an order of magnitude slower to answer than a busy one.

### Machine-specific paths

The committed configs point at one particular machine. Override without touching
tracked files:

```bash
cp configs/local.yaml.example configs/local.yaml   # gitignored, shallow-merged
```

or set `GPUSTARVE_MODELS` and write `${GPUSTARVE_MODELS}/...` in a config —
paths are environment-expanded at load.

---

## Running a sweep

```bash
python scripts/run_benchmark.py  --config configs/desktop_rtx5060.yaml
python scripts/make_tables.py    results/raw/sweep-<fingerprint>-<stamp>.json
python scripts/make_figures.py   results/raw/sweep-<fingerprint>-<stamp>.json --rated-gbs 7.0
python scripts/run_nsight.py     --config configs/desktop_rtx5060.yaml --verify
python scripts/analyze_traces.py
```

Check the plumbing first — two minutes rather than an hour:

```bash
python scripts/run_benchmark.py --config configs/desktop_rtx5060.yaml --runs 3
```

At `runs: 10` with the overhead arm, a sweep is 60 measurements plus warm-ups —
about six minutes at roughly 5 s per run. It is cheap enough that `--runs 20` is
usually worth it, and cheap enough that a disturbed sweep should be repeated
rather than argued with.

Formats, arms and repetitions are all interleaved, so no condition is
systematically first and a transient cannot be charged to one format. System
sleep is suppressed for the duration automatically.

**[docs/RUNBOOK.md](docs/RUNBOOK.md) is the gated step-by-step order** — start
there rather than here if you are actually running this.

---

## Results

RTX 5060 Laptop · Core Ultra 9 · PCIe Gen4 NVMe · Qwen2.5-1.5B ·
torch 2.11.0+cu128 · transformers 5.14.1 · llama-cpp-python 0.3.34 ·
sweep `9dd0dcefecf1-20260809-183356`, **n=20**, warm-up discarded,
formats and arms interleaved

| Format | TTFT (s) | Speedup | Energy (J) | Reduction | Welch t (df) |
|---|---|---|---|---|---|
| pickle | 2.71 ± 0.19 | 1.00× | 96.4 ± 7.1 | — | — |
| safetensors | 2.08 ± 0.14 | 1.31× | 78.3 ± 10.8 | 18.8% | 6.26 (32.7) |
| gguf (Q4\_K\_M) | 1.00 ± 0.05 | 2.70× | 26.5 ± 3.7 | 72.6% | 39.28 (28.6) |

Sampling overhead −2.3%, +3.7%, +1.8% — all below the 5% threshold, so latency
and energy are quoted from the sampled arm without further caveat. No run was
flagged as externally disturbed, and no energy window was too short to
integrate.

### Where the time goes

| Format | Tokenizer | Deserialize | Transfer | Generate | Sum | TTFT |
|---|---|---|---|---|---|---|
| pickle | 0.26 | **0.12** | **2.21** | 0.12 | 2.71 | 2.71 |
| safetensors | 0.26 | 0.78 | 0.98 | 0.06 | 2.08 | 2.08 |
| gguf | 0.00 | 0.96 | 0.00 | 0.04 | 1.00 | 1.00 |

Pickle's penalty is large and reproducible, but it does **not** sit in
deserialization. Its deserialize phase is 0.12 s; **82% of its cold path is
device transfer**. Under `transformers` 5.14 a modern `from_pretrained` returns
without materialising tensors, deferring the read into `.to(device)`.

Note the tokenizer column: 0.26 s in both HuggingFace formats. Earlier sweeps
showed 0.63 vs 0.25 s here, which turned out to be a differently serialised
`tokenizer.json` rather than anything about the container — see
[docs/METHODOLOGY.md](docs/METHODOLOGY.md). It is now a control rather than an
anomaly.

### Why the transfer is slower — from the traces

`scripts/analyze_traces.py` reads the Nsight SQLite exports and compares the
copies each format actually issued:

| Format | Copies | Bytes | < 1 MB | 1–8 MB | 8–64 MB | > 64 MB | Achieved |
|---|---|---|---|---|---|---|---|
| pickle | 355 | 3.088 GB | 214 | 56 | 84 | 1 | **2.52 GB/s** |
| safetensors | 355 | 3.088 GB | 214 | 56 | 84 | 1 | **14.45 GB/s** |
| gguf | 345 | 0.981 GB | 204 | 126 | 14 | 1 | 2.23 GB/s |

Pickle and SafeTensors issue the **same number of copies, of the same sizes,
totalling the same bytes**, launch the same **1857 kernels** through the same
**4943 CUDA runtime calls**. The two loads are indistinguishable at every level
of the trace except achieved bandwidth. Transfer volume and fragmentation are
therefore eliminated by construction as explanations for a 5.7× gap.

What differs is the state of the source memory when the DMA engine runs.
SafeTensors populates its page mappings during `from_pretrained` — 1.28 s at 0%
GPU occupancy — after which copies proceed at near-interconnect speed. Pickle's
`from_pretrained` returns having done almost nothing, so each copy stalls on
faults served from NVMe: **2.52 GB/s is storage service time leaking into the
copy engine.**

Both formats pay the same read. SafeTensors pays it up front, in a window where
the accelerator is idle regardless. Pickle pays it *inside* the transfer, and
loses because sequential read-then-copy beats read interleaved with faults.

Residency is the leading explanation but not the only candidate: buffer
alignment and host-memory pinning predict the same sign. A pinned-buffer arm
and a pre-warmed page cache separate them, and both are cheap.

### GPU occupancy by phase

| Format | Tokenizer | Deserialize | Transfer | Generate |
|---|---|---|---|---|
| pickle | **0.0%** | **0.0%** | 60.4% | 11.6% |
| safetensors | **0.0%** | **0.0%** | 53.2% | 1.8% |
| gguf | — | 42.8% | — | 5.1% |

The accelerator is powered but issuing no work whatsoever through tokenizer and
deserialization, in both formats. That idle window is real and measurable; what
the traces revise is *which stage* is responsible for Pickle's extra time.

> Trace wall-clock times are inflated by instrumentation and are not latency
> measurements — the sweep above provides those. The bandwidth figures come from
> single traced loads and vary between captures; the robust quantity is the
> transfer-phase ratio at n=20 (2.21 vs 0.98 s), which the traces *explain*
> rather than establish. Full output:
> [`results/processed/trace_analysis.md`](results/processed/trace_analysis.md).

## What gets measured

**Phases**, in a vocabulary shared across formats so they stay comparable:
`tokenizer` → `deserialize` → `transfer` → `generate`. Where a runtime genuinely
collapses stages — llama.cpp performs read and upload inside one opaque call —
the combined duration is reported and the others recorded as `0.0` with an
explicit note attached to the run, never silently omitted.

**Throughput**, under both normalisations, always:

- `throughput_ondisk_gbs` — real bytes moved. *How fast is this file?*
- `throughput_reference_gbs` — FP16-equivalent bytes. *How fast is this model?*

A quantized model moves roughly a third of the bytes for the same semantic
payload, so these differ by ~3× for Q4\_K\_M. Both are recorded on every run and
printed side by side, because quoting one in a figure and the other in prose is
how a dataset silently contradicts itself.

**Energy**, integrated trapezoidally over the sampled window, with mean and peak
watts recorded under separate names. Any run whose window was too short to
integrate carries a resolution warning into the report.

---

## Layout

```
src/gpustarve/
  config.py       experiment definition, disk-measured sizes, run provenance
  power.py        NVML / RAPL / HWiNFO / modelled samplers, each naming its source
  loaders.py      per-format loaders with a shared phase vocabulary
  harness.py      single-pass measurement, warm-up, interleaved arms
  stats.py        aggregation with n and dispersion, Welch t, observer effect
  diagnostics.py  every preflight check, each carrying a concrete remedy
  _cuda_dlls.py   registers CUDA DLL directories before importing llama.cpp
  _power_state.py suppresses system sleep for the duration of a sweep
scripts/
  preflight.py       can this machine measure anything publishable?
  run_benchmark.py   execute a sweep
  make_tables.py     raw sweep -> publication tables
  make_figures.py    raw sweep -> paper figures (PDF + PNG)
  run_nsight.py      Nsight traces + NVTX cross-validation
  analyze_traces.py  GPU occupancy and DMA bandwidth from the traces
  check_models.py    focused model validation (subset of preflight)
  check_power.py     focused telemetry probe (subset of preflight)
configs/             platform definitions
results/raw/         one JSON per sweep: config + provenance + every run (gitignored)
results/processed/   generated Markdown tables (committed)
results/traces/      Nsight captures (gitignored)
figures/             generated PDF and PNG
paper/               SEC 2026 poster abstract
```

---

## Documentation

- **[docs/RUNBOOK.md](docs/RUNBOOK.md)** — gated execution order, bare machine to poster
- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** — paper-ready wording for the measurement protocol
- **[docs/AUDIT.md](docs/AUDIT.md)** — audit of an earlier dataset, and which design decision here answers each finding

---

## Tests

```bash
pytest tests/ -q
```

The suite guards the failure modes that make measurement code untrustworthy:
`NaN` reading as zero, means reported without n, experimental arms being mixed,
and sampling overhead going undetected.

---

## License

MIT — see [LICENSE](LICENSE).
