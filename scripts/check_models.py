#!/usr/bin/env python3
"""Validate model weights before running a sweep.

    python scripts/check_models.py --config configs/desktop_rtx5060.yaml

Checks that every configured format resolves, reports measured sizes, and warns
about the two conditions that silently invalidate a format comparison:

1. **Variant mismatch.** The claim under test is "identical model architecture
   and size, three serialization formats." If one path holds the base model and
   another holds the instruct-tuned variant, the comparison is confounded and no
   amount of careful timing rescues it.

2. **Implausible size ratios.** An FP16 checkpoint and its Q4_K_M quantization
   should differ by roughly 3x. A ratio far from that usually means one path
   points at the wrong model or an incomplete download.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpustarve.config import ExperimentConfig  # noqa: E402


def read_hf_config(path: Path) -> dict | None:
    cfg = path / "config.json" if path.is_dir() else None
    if cfg and cfg.exists():
        try:
            return json.loads(cfg.read_text())
        except Exception:
            return None
    return None


def describe_variant(path: Path) -> str:
    """Best-effort identification of which checkpoint a directory holds."""
    meta = read_hf_config(path)
    if meta:
        name = meta.get("_name_or_path") or ""
        arch = ",".join(meta.get("architectures", [])) or "?"
        hidden = meta.get("hidden_size")
        layers = meta.get("num_hidden_layers")
        return f"{arch} hidden={hidden} layers={layers} name={name or '(unset)'}"
    return "(no config.json -- single-file format)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    print(f"Platform: {cfg.label}\n")

    sizes: dict[str, float] = {}
    problems: list[str] = []

    for spec in cfg.formats:
        p = Path(spec.path)
        print(f"[{spec.name}]")
        print(f"  path : {spec.path}")
        if not p.exists():
            print("  STATUS: MISSING\n")
            problems.append(f"{spec.name}: path does not exist")
            continue
        try:
            b = spec.resolve_size_bytes()
        except Exception as exc:
            print(f"  STATUS: ERROR -- {exc}\n")
            problems.append(f"{spec.name}: {exc}")
            continue

        gb = b / 1e9
        sizes[spec.name] = gb
        print(f"  size : {gb:.3f} GB ({b:,} bytes)")
        if p.is_dir():
            shards = sorted(f.name for f in p.glob(f"*{spec.extension}"))
            print(f"  files: {len(shards)} x {spec.extension} -> {', '.join(shards[:4])}"
                  + (" ..." if len(shards) > 4 else ""))
            print(f"  model: {describe_variant(p)}")
        print()

    # Cross-format sanity: FP16-class formats should agree closely with each other.
    fp16_like = {k: v for k, v in sizes.items() if k in ("pickle", "safetensors")}
    if len(fp16_like) == 2:
        a, b = fp16_like["pickle"], fp16_like["safetensors"]
        drift = abs(a - b) / max(a, b) * 100
        print(f"pickle vs safetensors size drift: {drift:.1f}%")
        if drift > 2:
            problems.append(
                f"pickle and safetensors differ by {drift:.1f}% -- these should be the "
                "same weights in two containers; check they are the same variant"
            )

    if "gguf" in sizes and fp16_like:
        ratio = max(fp16_like.values()) / sizes["gguf"]
        print(f"FP16 : Q4_K_M size ratio: {ratio:.2f}x")
        if not 2.4 <= ratio <= 3.8:
            problems.append(
                f"FP16:Q4_K_M ratio is {ratio:.2f}x, expected roughly 3x -- one path "
                "may point at a different model or parameter count"
            )

    # Does the configured parameter count match the measured FP16 footprint?
    if fp16_like:
        implied = max(fp16_like.values()) * 1e9 / 2
        drift = abs(implied - cfg.param_count) / cfg.param_count * 100
        print(f"implied params from FP16 size: {implied/1e9:.3f} B "
              f"(config says {cfg.param_count/1e9:.3f} B, {drift:.1f}% off)")
        if drift > 10:
            problems.append(
                f"param_count in the config is {drift:.0f}% away from the measured FP16 "
                "footprint; FP16-equivalent throughput will be wrong"
            )

    print()
    if problems:
        print("PROBLEMS:")
        for p_ in problems:
            print(f"  - {p_}")
        return 1
    print("All formats resolve and sizes are self-consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
