"""GPU Starvation: cold-start latency and energy measurement for edge LLM inference."""

__version__ = "0.1.0"

from .config import ExperimentConfig, FormatSpec, Provenance
from .harness import RunResult, measure_once, run_sweep
from .stats import aggregate, phase_table, results_table, sampling_overhead

__all__ = [
    "ExperimentConfig",
    "FormatSpec",
    "Provenance",
    "RunResult",
    "measure_once",
    "run_sweep",
    "aggregate",
    "results_table",
    "phase_table",
    "sampling_overhead",
]
