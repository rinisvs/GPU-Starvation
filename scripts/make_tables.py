#!/usr/bin/env python3
"""Turn a raw sweep into publication tables.

    python scripts/make_tables.py results/raw/sweep-<fingerprint>-<stamp>.json

Emits Markdown to stdout and writes results/processed/<stem>.md. The observer
effect report is printed first and deliberately cannot be suppressed: if
sampling perturbs the load significantly, every downstream number needs that
caveat attached.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpustarve.stats import (  # noqa: E402
    aggregate, by_format, load_sweep, phase_table, results_table,
    sampling_overhead, welch_t,
)

WARN_PCT = 5.0


def energy_significance(payload: dict, baseline: str = "pickle") -> str:
    groups = by_format(payload, sampled=True)
    if baseline not in groups:
        return "_baseline absent; no comparison_"
    base = [r["total_energy_j"] for r in groups[baseline]]
    lines = ["| Comparison | Reduction | Welch t | df |", "|---|---|---|---|"]
    for fmt, recs in groups.items():
        if fmt == baseline:
            continue
        vals = [r["total_energy_j"] for r in recs]
        a, b = aggregate(base), aggregate(vals)
        if not (a.mean == a.mean and b.mean == b.mean and a.mean):
            continue
        t, df = welch_t(base, vals)
        lines.append(
            f"| {baseline} → {fmt} | {100 * (a.mean - b.mean) / a.mean:.1f}% "
            f"| {t:.2f} | {df:.1f} |"
        )
    return "\n".join(lines)


def run_stability(payload: dict, k: float = 3.0, rel_floor: float = 0.10) -> str:
    """Flag runs that a transient disturbance is likely to have inflated.

    Uses the median and the median absolute deviation rather than the mean and
    standard deviation, because a handful of disturbed runs corrupt the very
    statistics used to detect them.

    Flagged runs are **reported, never removed**. Post-hoc deletion of
    inconvenient points is indistinguishable from fitting the data to the
    hypothesis. What the flag licenses is re-running the sweep under better
    conditions, or disclosing the disturbance -- not quietly dropping it.

    A genuine external disturbance usually moves *every* phase at once,
    including phases with no dependence on the condition under test. The
    tokenizer phase is a pure CPU parse of an identical file across formats, so
    it functions as an internal control: when it moves with the rest, the run
    was disturbed rather than slow.
    """
    groups = by_format(payload, sampled=True)
    rows: list[str] = []
    for fmt, recs in groups.items():
        vals = sorted(r["ttft_s"] for r in recs)
        if len(vals) < 4:
            continue
        mid = len(vals) // 2
        median = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
        devs = sorted(abs(v - median) for v in vals)
        mad = devs[mid] if len(devs) % 2 else (devs[mid - 1] + devs[mid]) / 2
        sigma = 1.4826 * mad
        if sigma <= 0 or median <= 0:
            continue
        for r in recs:
            dev = abs(r["ttft_s"] - median)
            z = dev / sigma
            # Statistical *and* practical significance. A very tight format has
            # a near-zero MAD, which turns a millisecond of jitter into a
            # z-score of twenty. Requiring the deviation to also be a
            # meaningful fraction of the median keeps the flag pointed at
            # disturbances rather than at precision.
            if z > k and dev / median > rel_floor:
                rows.append(
                    f"| {fmt} | {r['run_index']} | {r['ttft_s']:.2f} | {median:.2f} "
                    f"| {z:.1f} | {r['tokenizer_s']:.2f} | {r.get('started_utc', '')} |"
                )
    if not rows:
        return ("No run deviates more than "
                f"{k:g} robust s.d. from its format's median. No disturbance detected.\n")

    rows.sort(key=lambda s: s.rsplit("|", 2)[1])
    body = (
        "> **Warning:** the runs below are far from their format's median.\n"
        ">\n"
        "> Two signatures distinguish a disturbed machine from a slow format. The\n"
        "> `Tokenizer` column is an identical CPU-bound parse in every HuggingFace\n"
        "> format, so if it rose with the rest, the machine was disturbed. And\n"
        "> because formats are interleaved, a disturbance lands on whichever\n"
        "> formats happen to be running — so flagged runs **clustered in time\n"
        "> across different formats** are external by construction. A genuinely\n"
        "> slow format cannot make a different format slow two seconds later.\n"
        ">\n"
        "> Repeat the sweep. Do not delete the runs.\n\n"
        "| Format | Run | TTFT (s) | Median | Robust z | Tokenizer (s) | Started |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )
    return body + _disturbance_windows(rows)


def _disturbance_windows(rows: list[str], gap_s: float = 45.0) -> str:
    """Group flagged runs into time clusters spanning more than one format.

    The tokenizer control cannot speak for GGUF: llama.cpp performs tokenizer
    load, weight read and upload inside one opaque call, so that column is
    always 0.00 there. Temporal co-occurrence covers the gap, and only works
    because the formats are interleaved -- under a blocked design every flagged
    run of a disturbed window would belong to the same format and would be
    indistinguishable from that format simply being slow.
    """
    import datetime as _dt

    parsed = []
    for r in rows:
        c = [x.strip() for x in r.strip("|").split("|")]
        try:
            parsed.append((_dt.datetime.strptime(c[6], "%Y-%m-%dT%H:%M:%SZ"), c[0]))
        except (ValueError, IndexError):
            return ""
    parsed.sort()

    clusters: list[list[tuple]] = [[parsed[0]]]
    for item in parsed[1:]:
        if (item[0] - clusters[-1][-1][0]).total_seconds() <= gap_s:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    multi = [c for c in clusters if len({f for _, f in c}) > 1]
    if not multi:
        return ("\nNo flagged run coincides in time with a flagged run of another "
                "format. The deviations are not obviously external.\n")

    out = [f"\n**{len(multi)} disturbance window(s)** — flagged runs spanning more than "
           "one format within {:.0f} s, so the cause is external to any format:\n\n".format(gap_s)]
    for c in multi:
        span = (c[-1][0] - c[0][0]).total_seconds()
        out.append(f"- `{c[0][0]:%H:%M:%S}`–`{c[-1][0]:%H:%M:%S}` ({span:.0f} s): "
                   + ", ".join(f for _, f in c) + "\n")
    if len(multi) > 2:
        gaps = [(multi[i + 1][0][0] - multi[i][-1][0]).total_seconds() / 60
                for i in range(len(multi) - 1)]
        out.append(f"\nWindows recur every {min(gaps):.1f}–{max(gaps):.1f} min, which "
                   "indicates a scheduled process rather than thermal drift. Identify "
                   "and stop it, then repeat the sweep.\n")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", help="path to a results/raw/*.json sweep file")
    ap.add_argument("--baseline", default="pickle")
    args = ap.parse_args()

    payload = load_sweep(args.sweep)
    prov = payload["provenance"]
    cfg = payload["config"]

    out: list[str] = []
    out.append(f"# Results — {cfg.get('label', 'unknown platform')}\n")
    out.append(
        f"- Config fingerprint: `{cfg.get('fingerprint')}`\n"
        f"- Git: `{prov.get('git_sha')}`"
        f"{' **(dirty working tree)**' if prov.get('git_dirty') else ''}\n"
        f"- GPU: {prov.get('gpu_name')} · driver {prov.get('driver_version')}\n"
        f"- Torch {prov.get('torch_version')} · CUDA {prov.get('cuda_version')}\n"
        f"- Run at {prov.get('timestamp_utc')} on {prov.get('hostname')}\n"
    )

    # Sensor provenance -- so a modelled number can never be cited as measured.
    sources = {(r.get("cpu_source"), r.get("gpu_source")) for r in payload["results"]}
    out.append("## Instrumentation\n")
    for cpu_src, gpu_src in sorted(s for s in sources if any(s)):
        out.append(f"- CPU: {cpu_src or 'n/a'}\n- GPU: {gpu_src or 'n/a'}\n")
    if any("MODELLED" in (c or "") for c, _ in sources):
        out.append(
            "\n> **Warning:** CPU energy is modelled from utilization, not measured. "
            "Any writeup must disclose this.\n"
        )

    # Tokenizer/config files shared between directory-based formats must match,
    # or part of any measured difference is an artifact of how the directories
    # were built rather than of the format under test.
    inv = payload.get("model_inventory", {})
    shared = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "special_tokens_map.json", "added_tokens.json")
    dirs = {k: v for k, v in inv.items() if len(v) > 1}
    if len(dirs) > 1:
        rows = []
        for fn in shared:
            sizes = {k: v.get(fn) for k, v in dirs.items()}
            present = {k: s for k, s in sizes.items() if s is not None}
            if present and len(set(present.values()) | ({None} if len(present) < len(dirs) else set())) > 1:
                rows.append((fn, sizes))
        if rows:
            out.append("\n## Auxiliary-file mismatch\n")
            out.append("> **Warning:** these files differ between formats. Any timing\n"
                       "> difference they cause is an artifact of directory construction,\n"
                       "> not of the serialization format under test.\n\n")
            cols = list(dirs)
            out.append("| File | " + " | ".join(cols) + " |\n")
            out.append("|---" * (len(cols) + 1) + "|\n")
            for fn, sizes in rows:
                cells = [f"{sizes[c]:,}" if sizes.get(c) else "absent" for c in cols]
                out.append(f"| `{fn}` | " + " | ".join(cells) + " |\n")

    out.append("\n## Observer effect\n")
    overhead = sampling_overhead(payload)
    if not overhead:
        out.append(
            "_No unsampled arm in this sweep. Sampling overhead is unquantified — "
            "re-run with `overhead_arm: true` before publishing._\n"
        )
    else:
        out.append("| Format | Load sampled (s) | Load unsampled (s) | Overhead |\n|---|---|---|---|\n")
        worst = 0.0
        for fmt, d in overhead.items():
            worst = max(worst, abs(d["overhead_pct"]))
            out.append(
                f"| {fmt} | {d['load_sampled_s']:.3f} | "
                f"{d['load_unsampled_s']:.3f} | {d['overhead_pct']:+.1f}% |\n"
            )
        if worst > WARN_PCT:
            out.append(
                f"\n> **Warning:** sampling perturbs the load by up to {worst:.1f}%. "
                "Latency reported here is the sampled arm; if you quote latency and "
                "energy together, they must both come from this arm.\n"
            )

    # Surface per-run warnings: resolution problems and modelled telemetry.
    notes: dict[str, int] = {}
    for r in payload["results"]:
        for n in r.get("notes", []):
            key = n.split(" -- ")[0].split(":")[0].strip()
            notes[key] = notes.get(key, 0) + 1
    if notes:
        out.append("\n## Measurement warnings\n")
        for k, c in sorted(notes.items(), key=lambda kv: -kv[1]):
            out.append(f"- **{k}** — raised on {c} run(s)\n")

    out.append("\n## Run stability\n")
    out.append(run_stability(payload))

    out.append("\n## Main results\n")
    out.append(results_table(payload, args.baseline) + "\n")
    out.append("\nAll figures are mean ± s.d. Throughput is reported under both "
               "normalizations: on-disk bytes, and FP16-equivalent bytes for "
               "cross-format comparison.\n")

    out.append("\n## Phase breakdown\n")
    out.append(phase_table(payload) + "\n")
    out.append("\n`Sum` should reconcile with `TTFT`. A gap means unaccounted time.\n")

    out.append("\n## Energy significance\n")
    out.append(energy_significance(payload, args.baseline) + "\n")

    text = "".join(out)
    print(text)

    dest = Path("results/processed") / (Path(args.sweep).stem + ".md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    print(f"\nWrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
