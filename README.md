# GPU Starvation

Cold-start latency and energy measurement for edge LLM inference.

**PyTorch Conference North America 2026 — poster.**
*Your GPU Is Starving: The Hidden Cost of Loading LLMs at the Edge.*

## What this does

Before a model answers anything, gigabytes of weights must travel from storage
into GPU memory. On a server that happens once. At the edge, memory is scarce
enough that models are evicted when idle and reloaded on demand, so the cold
start is the normal case rather than an exception.

This repository measures that path — how long it takes, how much energy it
costs, and which stage is responsible — for three weight formats: PyTorch
Pickle, SafeTensors and GGUF.

**The finding.** Loading the same 1.5B model from Pickle rather than SafeTensors
costs 1.31× the time and 1.23× the energy. Nsight traces show both formats issue
identical transfer work — 355 copies, 3.088 GB — yet Pickle achieves 2.52 GB/s
against SafeTensors' 14.45 GB/s. The transfer work is not the variable; the
state of the source memory when the DMA engine runs is.

## Results

| Format | TTFT (s) | Speedup | Energy (J) | Reduction |
|---|---|---|---|---|
| pickle | 2.71 ± 0.19 | 1.00× | 96.4 ± 7.1 | — |
| safetensors | 2.08 ± 0.14 | 1.31× | 78.3 ± 10.8 | 18.8% |
| gguf (Q4\_K\_M) | 1.00 ± 0.05 | 2.70× | 26.5 ± 3.7 | 72.6% |

Pickle's penalty is not in deserialization — that phase is 0.12 s. 82% of its
cold path is host-to-device transfer, because the read is deferred into the
copy and each copy then stalls on page faults served from NVMe.

Full tables: [`results/processed/`](results/processed/).

## Setup

Requires Python 3.10+ and an NVIDIA driver. Results above were measured on an
RTX 5060 Laptop with Qwen2.5-1.5B, torch 2.11.0+cu128 and transformers 5.14.1,
at n=20.

```bash
bash bootstrap.sh                                          # Linux / macOS
powershell -ExecutionPolicy Bypass -File bootstrap.ps1     # Windows
```

Creates `.venv`, installs dependencies, runs the tests and the preflight.
Model weights are not downloaded — point a config at your own.

Check the machine can produce a valid measurement:

```bash
python scripts/preflight.py --config configs/desktop_rtx5060.yaml
```

Exit 0 means ready. Each failure prints its remedy.

CPU power needs setup: on Linux it reads RAPL directly; on Windows, run
HWiNFO64 with Shared Memory Support enabled and polling set to 100 ms.

## Running it

```bash
# smoke test first (~2 min)
python scripts/run_benchmark.py --config configs/desktop_rtx5060.yaml --runs 3

# full sweep
python scripts/run_benchmark.py  --config configs/desktop_rtx5060.yaml
python scripts/make_tables.py    results/raw/sweep-<fingerprint>-<stamp>.json
python scripts/make_figures.py   results/raw/sweep-<fingerprint>-<stamp>.json --rated-gbs 7.0

# traces, to explain the numbers
python scripts/run_nsight.py     --config configs/desktop_rtx5060.yaml --verify
python scripts/analyze_traces.py
```

Output lands in `results/processed/` and `figures/`.

## Measurement design

Latency and energy are captured in a single pass, so both always describe the
same load. Power sampling perturbs what it measures, so each configuration runs
with and without it and the overhead is reported. Formats and repetitions are
interleaved, keeping a passing slowdown from landing on one format. Model sizes
are read from disk, and every sampler records which sensor it used.

## Tests

```bash
pytest tests/ -q
```
