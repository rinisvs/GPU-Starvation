"""Experiment configuration and run provenance.

Design rule: nothing measurable is ever hardcoded. Model sizes are read from
disk, not typed into a constant. Every result file carries enough metadata to
reconstruct the conditions it was produced under.
"""

from __future__ import annotations

import hashlib
import os
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class FormatSpec:
    """One serialization format under test.

    Attributes:
        name: Identifier used in results ("pickle", "safetensors", "gguf").
        path: File or directory holding the weights.
        extension: Suffix used to sum directory size. None for single files.
    """

    name: str
    path: str
    extension: str | None = None

    def resolve_size_bytes(self) -> int:
        """Measure actual on-disk size. Never trust a hardcoded constant."""
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"format '{self.name}': path does not exist: {p}")
        if p.is_file():
            return p.stat().st_size
        if self.extension is None:
            raise ValueError(f"format '{self.name}': directory path requires an extension filter")
        files = list(p.glob(f"*{self.extension}"))
        if not files:
            raise FileNotFoundError(f"format '{self.name}': no *{self.extension} files in {p}")
        return sum(f.stat().st_size for f in files)

    def inventory(self) -> dict[str, int]:
        """Every file in the model directory, by name and size.

        The config fingerprint hashes the *configuration*, not the weights, so
        two sweeps can share a fingerprint while loading materially different
        files. Recording the inventory makes such a difference visible after the
        fact.

        This is not hypothetical: a tokenizer serialized two different ways
        (11.4 MB inline vs 7.0 MB plus separate vocab and merges) accounted for
        43% of a measured format gap, and nothing in the configuration
        distinguished the two runs.
        """
        p = Path(self.path)
        if p.is_file():
            return {p.name: p.stat().st_size}
        if not p.is_dir():
            return {}
        return {f.name: f.stat().st_size for f in sorted(p.iterdir()) if f.is_file()}


@dataclass
class ExperimentConfig:
    """Full experiment definition.

    Attributes:
        formats: Serialization formats to sweep.
        runs: Repetitions per configuration. n>=10 for publishable error bars.
        warmup: Loads performed and discarded before measurement begins. The
            first load of a format pays one-off costs (CUDA context, kernel JIT,
            lazy imports) that belong to no arm and skew small-n means badly.
        prompt: Fixed prompt, identical across all runs.
        max_new_tokens: Generation length. 1 is enough for TTFT.
        sample_power: Whether to run power samplers during the measured window.
        power_interval_s: Sampling period. Appears in the observer-effect arm.
        overhead_arm: If true, each config is run twice (sampling on and off)
            so that sampling overhead is measured rather than assumed away.
        cooldown_s: Idle gap between runs, to bound thermal drift.
        reference_precision_bytes_per_param: Used to compute an FP16-equivalent
            byte volume for cross-format throughput comparison.
        param_count: Model parameter count, for FP16-equivalent normalization.
        device: Torch device string.
        label: Human-readable platform label recorded in results.
    """

    formats: list[FormatSpec]
    runs: int = 10
    warmup: int = 1
    prompt: str = "The future of edge computing is"
    max_new_tokens: int = 1
    sample_power: bool = True
    power_interval_s: float = 0.1
    overhead_arm: bool = True
    cooldown_s: float = 3.0
    reference_precision_bytes_per_param: int = 2
    param_count: int = 1_543_714_304  # Qwen2.5-1.5B
    device: str = "cuda"
    label: str = "unspecified-platform"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a config, applying any local override and expanding env vars.

        Two mechanisms keep machine-specific paths out of version control while
        letting a checkout run unmodified on the machine it was written for:

        * ``${VAR}`` and ``$VAR`` in any path are expanded from the environment,
          so ``GPUSTARVE_MODELS`` can relocate the whole model tree.
        * ``configs/local.yaml``, if present, is shallow-merged over the loaded
          config. It is gitignored, so personal paths never reach a commit.
        """
        raw = yaml.safe_load(Path(path).read_text())

        local = Path(path).parent / "local.yaml"
        if local.exists():
            overrides = yaml.safe_load(local.read_text()) or {}
            raw.update(overrides)

        specs = []
        for f in raw.pop("formats"):
            f = dict(f)
            f["path"] = os.path.expandvars(str(f["path"]))
            specs.append(FormatSpec(**f))
        raw["formats"] = specs
        return cls(**raw)

    @property
    def reference_bytes(self) -> int:
        """FP16-equivalent byte volume, for cross-format throughput comparison.

        A quantized model moves fewer bytes for the same semantic payload.
        Reporting throughput on raw bytes and on this reference volume answers
        two different questions; the harness records both and labels them.
        """
        return self.param_count * self.reference_precision_bytes_per_param

    def fingerprint(self) -> str:
        """Stable hash of the config, so results can be matched to their inputs."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def _shell(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class Provenance:
    """Everything needed to explain why a result looks the way it does.

    Recorded alongside every sweep. The absence of this metadata is the single
    most common reason old benchmark numbers become unusable.
    """

    timestamp_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    git_sha: str | None = field(default_factory=lambda: _shell(["git", "rev-parse", "HEAD"]))
    git_dirty: bool = field(default_factory=lambda: bool(_shell(["git", "status", "--porcelain"])))
    hostname: str = field(default_factory=platform.node)
    os: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")
    python: str = field(default_factory=platform.python_version)
    # The interpreter path, not merely the version. Traces are only comparable
    # to a sweep if both ran against the same installed packages, and "is this a
    # virtualenv" is the wrong test for that -- a sweep run under the system
    # interpreter should be traced under the system interpreter.
    python_executable: str = field(default_factory=lambda: sys.executable)
    cpu: str = field(default_factory=platform.processor)
    torch_version: str | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    driver_version: str | None = None

    @classmethod
    def collect(cls) -> Provenance:
        p = cls()
        try:
            import torch

            p.torch_version = torch.__version__
            p.cuda_version = torch.version.cuda
            if torch.cuda.is_available():
                p.gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        p.driver_version = _shell(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
        return p

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
