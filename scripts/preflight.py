#!/usr/bin/env python3
"""Single preflight. Run this first on any machine.

    python scripts/preflight.py
    python scripts/preflight.py --config configs/desktop_rtx5060.yaml
    python scripts/preflight.py --quick        # skip the live sampling probes

Answers one question: can this machine produce a publishable measurement?
Exit code 0 means yes, 1 means something blocking is wrong. Every failure comes
with a concrete remedy.

Nothing here installs or changes anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpustarve.diagnostics import Report, run_all  # noqa: E402

GLYPH = {"ok": "  OK  ", "warn": " WARN ", "fail": " FAIL "}
WIDTH = 74


def render(report: Report) -> None:
    print("=" * WIDTH)
    print("GPU Starvation — preflight")
    print("=" * WIDTH)

    for c in report.checks:
        print(f"[{GLYPH[c.status]}] {c.name}")
        print(f"          {c.detail}")
        if c.remedy and c.status != "ok":
            for i, line in enumerate(_wrap(c.remedy, WIDTH - 12)):
                print(f"          {'-> ' if i == 0 else '   '}{line}")
        print()

    print("=" * WIDTH)
    blockers, fails, warns = report.blockers, report.failures, report.warnings

    if blockers:
        print(f"BLOCKED — {len(blockers)} check(s) prevent any measurement:")
        for c in blockers:
            print(f"  * {c.name}: {c.detail}")
    elif fails:
        print(f"NOT READY — {len(fails)} check(s) failed:")
        for c in fails:
            print(f"  * {c.name}: {c.detail}")
        print("\nA sweep will run, but the affected numbers are not publishable.")
    elif warns:
        print(f"READY, with {len(warns)} warning(s):")
        for c in warns:
            print(f"  * {c.name}: {c.detail}")
    else:
        print("READY — all checks passed.")

    print()
    if not fails:
        print("Next:  python scripts/run_benchmark.py --config <config> --runs 2")
        print("       python scripts/make_tables.py results/raw/<sweep>.json")
    print("=" * WIDTH)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="also validate the models named in this config")
    ap.add_argument("--quick", action="store_true",
                    help="skip live power sampling probes (~3s faster, less useful)")
    args = ap.parse_args()

    default = Path("configs/desktop_rtx5060.yaml")
    config = args.config or (str(default) if default.exists() else None)

    report = run_all(config, quick=args.quick)
    render(report)
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
