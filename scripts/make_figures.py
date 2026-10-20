#!/usr/bin/env python3
"""Generate paper figures from a raw sweep.

    python scripts/make_figures.py results/raw/sweep-<fingerprint>-<stamp>.json

Writes PDF (for LaTeX) and PNG (for slides) into figures/.

Every figure is drawn from a single sweep file and labels its own n, so a
figure and the prose beside it can never come from different experiments --
the failure that produced two contradicting throughput numbers in the previous
dataset.

Throughput is plotted under one normalization at a time, and the axis label
says which. Mixing them in one chart is exactly the mistake being guarded
against.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gpustarve.stats import aggregate, by_format, load_sweep  # noqa: E402

ORDER = ["pickle", "safetensors", "gguf", "gguf_nommap"]
PHASE_LABELS = {
    "tokenizer_s": "Tokenizer",
    "deserialize_s": "Deserialize",
    "transfer_s": "Host→device",
    "generate_s": "First token",
}
PHASE_COLORS = ["#9db4c0", "#c1272d", "#5085a5", "#7d8c7c"]

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def ordered(groups: dict) -> list[str]:
    known = [f for f in ORDER if f in groups]
    return known + [f for f in groups if f not in ORDER]


def save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_dir / stem}.pdf / .png")


def fig_phases(groups: dict, out: Path) -> None:
    """Stacked phase breakdown -- the figure that shows where the time goes."""
    names = ordered(groups)
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    bottoms = [0.0] * len(names)

    for (key, label), colour in zip(PHASE_LABELS.items(), PHASE_COLORS):
        vals = [aggregate([r[key] for r in groups[f]]).mean for f in names]
        ax.barh(names, vals, left=bottoms, label=label, color=colour,
                edgecolor="white", linewidth=0.6)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    for i, total in enumerate(bottoms):
        ax.text(total + max(bottoms) * 0.015, i, f"{total:.2f}s",
                va="center", fontsize=8)

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_xlim(0, max(bottoms) * 1.14)
    # Legend above the axes: an in-axes legend collides with the total labels,
    # which sit at the end of the longest bar.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4,
              frameon=False, fontsize=8, columnspacing=1.2, handlelength=1.2)
    ax.set_title("Cold-start phase breakdown", loc="left", fontsize=10, pad=24)
    save(fig, out, "fig1_phases")


def fig_ttft(groups: dict, out: Path, baseline: str) -> None:
    names = ordered(groups)
    aggs = {f: aggregate([r["ttft_s"] for r in groups[f]]) for f in names}
    base = aggs.get(baseline)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.bar(names, [aggs[f].mean for f in names],
           yerr=[aggs[f].std for f in names], capsize=4,
           color="#5085a5", edgecolor="black", linewidth=0.6)

    for i, f in enumerate(names):
        a = aggs[f]
        txt = f"{a.mean:.2f}s"
        if base and base.mean and f != baseline:
            txt += f"\n({base.mean / a.mean:.1f}×)"
        ax.text(i, a.mean + a.std + max(a.mean for a in aggs.values()) * 0.04,
                txt, ha="center", fontsize=8)

    n = aggs[names[0]].n
    ax.set_ylabel("Time to first token (s)")
    ax.set_ylim(0, max(a.mean + a.std for a in aggs.values()) * 1.28)
    ax.set_title(f"Cold-start TTFT (mean ± s.d., n={n})", loc="left", fontsize=10)
    save(fig, out, "fig2_ttft")


def fig_throughput(groups: dict, out: Path, rated_gbs: float | None) -> None:
    """Both normalizations, side by side and explicitly labelled."""
    names = ordered(groups)
    disk = {f: aggregate([r["throughput_ondisk_gbs"] for r in groups[f]]) for f in names}
    ref = {f: aggregate([r["throughput_reference_gbs"] for r in groups[f]]) for f in names}

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    x = range(len(names))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [disk[f].mean for f in names], w,
           yerr=[disk[f].std for f in names], capsize=3, label="On-disk bytes",
           color="#5085a5", edgecolor="black", linewidth=0.6)
    ax.bar([i + w / 2 for i in x], [ref[f].mean for f in names], w,
           yerr=[ref[f].std for f in names], capsize=3, label="FP16-equivalent bytes",
           color="#c9a227", edgecolor="black", linewidth=0.6)

    if rated_gbs:
        ax.axhline(rated_gbs, ls="--", lw=1, color="#c1272d")
        ax.text(len(names) - 0.5, rated_gbs * 1.02,
                f"rated {rated_gbs:.1f} GB/s", ha="right", fontsize=8, color="#c1272d")

    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Effective throughput (GB/s)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Storage-to-memory throughput", loc="left", fontsize=10)
    save(fig, out, "fig3_throughput")


def fig_energy(groups: dict, out: Path, baseline: str) -> None:
    names = ordered(groups)
    aggs = {f: aggregate([r["total_energy_j"] for r in groups[f]]) for f in names}
    if all(a.n == 0 for a in aggs.values()):
        print("  (no energy data in this sweep -- skipping fig4)")
        return
    base = aggs.get(baseline)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.bar(names, [aggs[f].mean for f in names],
           yerr=[aggs[f].std for f in names], capsize=4,
           color="#7d8c7c", edgecolor="black", linewidth=0.6)

    peak = max(a.mean + a.std for a in aggs.values() if a.n)
    for i, f in enumerate(names):
        a = aggs[f]
        if not a.n:
            continue
        txt = f"{a.mean:.1f} J"
        if base and base.mean and f != baseline:
            txt += f"\n(−{100 * (base.mean - a.mean) / base.mean:.0f}%)"
        ax.text(i, a.mean + a.std + peak * 0.04, txt, ha="center", fontsize=8)

    n = max(a.n for a in aggs.values())
    ax.set_ylabel("Energy per cold start (J)")
    ax.set_ylim(0, peak * 1.28)
    ax.set_title(f"Load-phase energy (mean ± s.d., n={n})", loc="left", fontsize=10)
    save(fig, out, "fig4_energy")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", help="path to a results/raw/*.json sweep file")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--baseline", default="pickle")
    ap.add_argument("--rated-gbs", type=float, default=None,
                    help="rated sequential read of the storage device, drawn as a "
                         "reference line (e.g. 7.0 for PCIe Gen4 NVMe)")
    args = ap.parse_args()

    payload = load_sweep(args.sweep)
    groups = by_format(payload, sampled=True)
    if not groups:
        print("No sampled records in this sweep.", file=sys.stderr)
        return 1

    out = Path(args.out)
    print(f"Figures from {args.sweep}:")
    fig_phases(groups, out)
    fig_ttft(groups, out, args.baseline)
    fig_throughput(groups, out, args.rated_gbs)
    fig_energy(groups, out, args.baseline)

    modelled = any("MODELLED" in (r.get("cpu_source") or "") for r in payload["results"])
    if modelled:
        print("\nNOTE: CPU energy in this sweep is modelled, not measured. "
              "The energy figure inherits that caveat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
