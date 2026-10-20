"""Tests for aggregation and reporting.

The regressions guarded here are the specific failures found in the previous
codebase: NaN masquerading as zero, means reported without n, and mixing arms.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpustarve.stats import aggregate, by_format, sampling_overhead, welch_t  # noqa: E402


def test_aggregate_basic():
    a = aggregate([1.0, 2.0, 3.0])
    assert a.n == 3
    assert math.isclose(a.mean, 2.0)
    assert math.isclose(a.std, 1.0)
    assert a.minimum == 1.0 and a.maximum == 3.0


def test_aggregate_ignores_nan():
    a = aggregate([1.0, float("nan"), 3.0])
    assert a.n == 2
    assert math.isclose(a.mean, 2.0)


def test_aggregate_all_nan_is_nan_not_zero():
    """A failed sampler must not read as a zero-energy result."""
    a = aggregate([float("nan"), float("nan")])
    assert a.n == 0
    assert math.isnan(a.mean)


def test_aggregate_single_run_has_zero_spread():
    a = aggregate([5.0])
    assert a.n == 1 and a.std == 0.0


def test_welch_separates_distinct_samples():
    t, df = welch_t([100.0, 98.0, 102.0, 99.0], [65.0, 63.0, 67.0, 64.0])
    assert t > 10 and df > 1


def test_welch_undefined_for_tiny_samples():
    t, df = welch_t([1.0], [2.0])
    assert math.isnan(t) and math.isnan(df)


def _payload():
    def rec(fmt, sampled, load):
        return {"format": fmt, "sampled": sampled, "load_s": load,
                "ttft_s": load + 0.1, "total_energy_j": 50.0}
    return {"results": [
        rec("gguf", True, 1.20), rec("gguf", True, 1.20),
        rec("gguf", False, 1.00), rec("gguf", False, 1.00),
    ]}


def test_by_format_filters_arm():
    assert len(by_format(_payload(), sampled=True)["gguf"]) == 2
    assert len(by_format(_payload(), sampled=False)["gguf"]) == 2


def test_sampling_overhead_detected():
    """20% perturbation must be reported, not silently absorbed."""
    rep = sampling_overhead(_payload())
    assert math.isclose(rep["gguf"]["overhead_pct"], 20.0, abs_tol=0.01)
