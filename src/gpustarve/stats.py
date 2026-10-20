"""Aggregation and reporting.

Every aggregate carries n and dispersion. A mean without an n and a standard
deviation is not a result, and a table generated from this module will not let
you produce one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass
class Aggregate:
    """Summary of one metric over repeated runs."""

    n: int
    mean: float
    std: float
    ci95: float
    minimum: float
    maximum: float

    def render(self, places: int = 2) -> str:
        return f"{self.mean:.{places}f} ± {self.std:.{places}f}"


def aggregate(values: Sequence[float]) -> Aggregate:
    """Summarize a sample, ignoring NaN."""
    clean = [v for v in values if v == v]
    n = len(clean)
    if n == 0:
        nan = float("nan")
        return Aggregate(0, nan, nan, nan, nan, nan)
    mean = sum(clean) / n
    if n == 1:
        return Aggregate(1, mean, 0.0, 0.0, mean, mean)
    var = sum((v - mean) ** 2 for v in clean) / (n - 1)
    std = math.sqrt(var)
    return Aggregate(n, mean, std, 1.96 * std / math.sqrt(n), min(clean), max(clean))


def welch_t(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Welch's t statistic and approximate degrees of freedom.

    Reported so that claimed reductions can be defended when sample variance is
    large, which it typically is for energy measurements.
    """
    ag, bg = aggregate(a), aggregate(b)
    if ag.n < 2 or bg.n < 2:
        return float("nan"), float("nan")
    va, vb = ag.std**2 / ag.n, bg.std**2 / bg.n
    denom = math.sqrt(va + vb)
    if denom == 0:
        return float("nan"), float("nan")
    t = (ag.mean - bg.mean) / denom
    df = (va + vb) ** 2 / (va**2 / (ag.n - 1) + vb**2 / (bg.n - 1))
    return t, df


def load_sweep(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def by_format(payload: dict[str, Any], sampled: bool | None = True) -> dict[str, list[dict]]:
    """Group raw records by format, optionally filtering to one arm."""
    out: dict[str, list[dict]] = {}
    for r in payload["results"]:
        if sampled is not None and r["sampled"] != sampled:
            continue
        out.setdefault(r["format"], []).append(r)
    return out


def sampling_overhead(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Quantify the observer effect: how much does sampling slow the load?

    This is the check the previous codebase lacked. If the number here is
    large, latency and energy from separate sweeps cannot be reported side by
    side, and this harness exists to make that visible before publication.
    """
    on, off = by_format(payload, True), by_format(payload, False)
    report: dict[str, dict[str, float]] = {}
    for fmt in on:
        if fmt not in off:
            continue
        a, b = aggregate([r["load_s"] for r in on[fmt]]), aggregate(
            [r["load_s"] for r in off[fmt]]
        )
        if b.mean and b.mean == b.mean:
            report[fmt] = {
                "load_sampled_s": round(a.mean, 4),
                "load_unsampled_s": round(b.mean, 4),
                "overhead_pct": round(100 * (a.mean - b.mean) / b.mean, 2),
            }
    return report


def results_table(payload: dict[str, Any], baseline: str = "pickle") -> str:
    """Render the main results table in Markdown.

    Both throughput normalizations appear side by side. Speedups are computed
    within a single arm, so they can never be silently mixed across sweeps.
    """
    groups = by_format(payload, sampled=True)
    if not groups:
        return "_no sampled records_"

    base = aggregate([r["ttft_s"] for r in groups[baseline]]).mean if baseline in groups else None

    head = (
        "| Format | n | TTFT (s) | Speedup | Thrpt on-disk (GB/s) "
        "| Thrpt FP16-eq (GB/s) | Energy (J) |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for fmt, recs in groups.items():
        ttft = aggregate([r["ttft_s"] for r in recs])
        tod = aggregate([r["throughput_ondisk_gbs"] for r in recs])
        tref = aggregate([r["throughput_reference_gbs"] for r in recs])
        en = aggregate([r["total_energy_j"] for r in recs])
        speed = f"{base / ttft.mean:.2f}×" if base and ttft.mean else "—"
        rows.append(
            f"| {fmt} | {ttft.n} | {ttft.render(2)} | {speed} | "
            f"{tod.render(2)} | {tref.render(2)} | {en.render(2)} |"
        )
    return head + "\n".join(rows)


def phase_table(payload: dict[str, Any]) -> str:
    """Phase breakdown, with the sum shown so it can be checked against TTFT."""
    groups = by_format(payload, sampled=True)
    head = (
        "| Format | Tokenizer | Deserialize | Transfer | Generate | Sum | TTFT |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for fmt, recs in groups.items():
        cols = [aggregate([r[k] for r in recs]).mean for k in
                ("tokenizer_s", "deserialize_s", "transfer_s", "generate_s")]
        ttft = aggregate([r["ttft_s"] for r in recs]).mean
        rows.append(
            "| " + fmt + " | " + " | ".join(f"{c:.2f}" for c in cols)
            + f" | {sum(cols):.2f} | {ttft:.2f} |"
        )
    return head + "\n".join(rows)
