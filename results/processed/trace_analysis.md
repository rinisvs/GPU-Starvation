# Nsight trace analysis
> Wall-clock times here are inflated by instrumentation and are **not** latency measurements — see `results/processed/` for those. What the trace establishes is *where the accelerator was idle* and *how fast the copies actually ran*.

## GPU occupancy by phase
| Format | Phase | Wall (s) | GPU busy (s) | Occupancy |
|---|---|---|---|---|
| gguf | deserialize | 1.028 | 0.439 | 42.8% |
| gguf | generate | 0.151 | 0.008 | 5.1% |
| pickle | tokenizer | 0.288 | 0.000 | 0.0% **idle** |
| pickle | deserialize | 0.723 | 0.000 | 0.0% **idle** |
| pickle | transfer | 2.030 | 1.227 | 60.4% |
| pickle | generate | 0.992 | 0.115 | 11.6% |
| safetensors | tokenizer | 0.200 | 0.000 | 0.0% **idle** |
| safetensors | deserialize | 1.281 | 0.000 | 0.0% **idle** |
| safetensors | transfer | 0.401 | 0.214 | 53.2% |
| safetensors | generate | 0.817 | 0.015 | 1.8% |

## Transfer characteristics
| Format | Copies | Bytes | Mean copy | Copy time (s) | Achieved |
|---|---|---|---|---|---|
| gguf | 345 | 0.981 GB | 2.84 MB | 0.439 | **2.23 GB/s** |
| pickle | 355 | 3.088 GB | 8.70 MB | 1.227 | **2.52 GB/s** |
| safetensors | 355 | 3.088 GB | 8.70 MB | 0.214 | **14.45 GB/s** |

## Copy size distribution
| Format | < 1 MB | 1-8 MB | 8-64 MB | > 64 MB |
|---|---|---|---|---|
| gguf | 204 ops / 0.015 GB | 126 ops / 0.616 GB | 14 ops / 0.158 GB | 1 ops / 0.191 GB |
| pickle | 214 ops / 0.045 GB | 56 ops / 0.264 GB | 84 ops / 2.312 GB | 1 ops / 0.467 GB |
| safetensors | 214 ops / 0.045 GB | 56 ops / 0.264 GB | 84 ops / 2.312 GB | 1 ops / 0.467 GB |

## Reading this

Where two formats issue the **same number of copies, of the same sizes, totalling the same bytes**, but achieve different bandwidth, the transfer work is identical and the difference lies in the source memory: a copy whose source pages are not yet resident stalls the DMA engine on page faults served from storage, so the measured rate reflects disk, not the interconnect.

Phases at ~0% occupancy are intervals where the accelerator was powered but issuing no work.
