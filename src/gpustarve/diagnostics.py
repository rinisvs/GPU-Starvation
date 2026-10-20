"""Preflight checks.

One place that answers "can this machine produce a publishable measurement?"
Every check returns a :class:`Check` with a status and, when something is wrong,
a concrete remedy. Nothing here mutates the system.

The design goal is that a person cloning this repo onto a bare machine can run
``scripts/preflight.py`` and be told exactly what is missing, rather than
discovering it forty minutes into a sweep.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

Status = Literal["ok", "warn", "fail"]

# Sampling below this rate cannot resolve a sub-second load window.
MIN_SAMPLE_RATE_HZ = 8.0
# Below this, energy for the fastest formats is not meaningfully integrable.
MIN_SAMPLES_PER_WINDOW = 10


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    remedy: str | None = None
    blocking: bool = False


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, c: Check) -> None:
        self.checks.append(c)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def blockers(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail" and c.blocking]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == "warn"]


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def check_python() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    return Check(
        "Python version",
        "ok" if ok else "fail",
        f"{v.major}.{v.minor}.{v.micro} at {sys.executable}",
        None if ok else "Python 3.10+ required.",
        blocking=not ok,
    )


def check_virtualenv() -> Check:
    """Verify the interpreter, and *fail* if the project's own venv is bypassed.

    Running outside any virtualenv is merely inadvisable. Running outside a
    virtualenv that exists in this very repository is almost always an accident,
    and a costly one: the two interpreters carry different package builds, so
    the sweep silently measures something other than what the environment was
    prepared to measure.

    This is not defensive programming for its own sake. It happened here. Three
    sweeps and a full set of traces were collected against the system
    interpreter, which held a *CPU-only* llama-cpp-python 0.3.20 in place of the
    venv's CUDA 0.3.34, plus a different torch and a downgraded transformers.
    Every GGUF number in those sweeps measured host-RAM loading, and the check
    that would have caught it was a warning nobody had reason to read twice.
    """
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        return Check("Virtual environment", "ok", f"active — {sys.prefix}")

    # Look for either layout rather than only the current platform's. A repo
    # shared between Windows and Linux, or inspected from the other side of a
    # mount, still has a .venv worth warning about.
    local = Path.cwd() / ".venv"
    if any((local / d / "activate").exists() for d in ("Scripts", "bin")):
        return Check(
            "Virtual environment",
            "fail",
            f"NOT active, but {local} exists — this interpreter is probably not "
            "the one this project was prepared with",
            "Activate it first, then re-run:\n"
            + ("             .\\.venv\\Scripts\\Activate.ps1   (PowerShell)"
               if os.name == "nt" else "             source .venv/bin/activate"),
            blocking=True,
        )

    return Check(
        "Virtual environment",
        "warn",
        "not active — using the system interpreter",
        "Recommended: python -m venv .venv && .venv\\Scripts\\activate (Windows)",
    )


REQUIRED = ["torch", "transformers", "safetensors", "yaml", "psutil", "numpy"]
# import name -> (pip name, what breaks without it)
OPTIONAL = {
    "llama_cpp": ("llama-cpp-python", "GGUF format will be unavailable"),
    "pynvml": ("nvidia-ml-py", "GPU power sampling will be unavailable"),
}


def check_packages() -> list[Check]:
    out: list[Check] = []
    missing: list[str] = []
    versions: list[str] = []
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            versions.append(f"{mod}={getattr(m, '__version__', '?')}")
        except ImportError:
            missing.append(mod)
    out.append(Check(
        "Required packages",
        "ok" if not missing else "fail",
        ", ".join(versions) if not missing else f"missing: {', '.join(missing)}",
        None if not missing else "pip install -r requirements.txt",
        blocking=bool(missing),
    ))
    for mod, (pip_name, consequence) in OPTIONAL.items():
        try:
            importlib.import_module(mod)
            out.append(Check(f"Optional: {mod}", "ok", "present"))
        except ImportError:
            out.append(Check(f"Optional: {mod}", "warn", f"absent — {consequence}",
                             f"pip install {pip_name}"))
    return out


def check_pynvml_flavour() -> Check:
    """The deprecated `pynvml` distribution shadows `nvidia-ml-py`.

    Both provide the importable name ``pynvml``. Having the deprecated one
    installed emits a FutureWarning on every torch import and is a common
    source of confusion, though it does not break measurement.
    """
    try:
        from importlib.metadata import distributions
    except ImportError:
        return Check("NVML package flavour", "warn", "cannot determine")
    names = {d.metadata["Name"].lower() for d in distributions() if d.metadata["Name"]}
    has_old, has_new = "pynvml" in names, "nvidia-ml-py" in names
    if has_old:
        return Check(
            "NVML package flavour", "warn",
            "deprecated `pynvml` distribution installed"
            + (" alongside nvidia-ml-py" if has_new else ""),
            "pip uninstall -y pynvml && pip install nvidia-ml-py "
            "(both provide `import pynvml`; this only silences the warning)",
        )
    if has_new:
        return Check("NVML package flavour", "ok", "nvidia-ml-py")
    return Check("NVML package flavour", "warn", "neither installed",
                 "pip install nvidia-ml-py")


# --------------------------------------------------------------------------
# Accelerator
# --------------------------------------------------------------------------


def check_cuda() -> Check:
    """Verify torch can actually reach the GPU.

    The common failure is silent: ``pip install torch`` from the default PyPI
    index yields a CPU-only wheel on Windows, which imports fine and reports no
    CUDA device. The build tag is checked explicitly so the remedy can name the
    real problem instead of "install a CUDA build".
    """
    try:
        import torch
    except ImportError:
        return Check("CUDA / torch", "fail", "torch not importable",
                     "pip install -r requirements.txt", blocking=True)

    cpu_only = "+cpu" in torch.__version__ or torch.version.cuda is None

    if not torch.cuda.is_available():
        if cpu_only:
            return Check(
                "CUDA / torch", "fail",
                f"torch {torch.__version__} is a CPU-ONLY build -- it can never see the GPU",
                "The default PyPI wheel is CPU-only on Windows. Reinstall from a "
                "CUDA index, newest first: "
                "pip uninstall -y torch && "
                "pip install torch --index-url https://download.pytorch.org/whl/cu130 "
                "(if that 404s, try cu128, then cu126). "
                "Check `nvidia-smi` for the CUDA version your driver supports.",
                blocking=True,
            )
        return Check(
            "CUDA / torch", "fail",
            f"torch {torch.__version__} (CUDA {torch.version.cuda}) reports no CUDA device",
            "The build is CUDA-capable but no device is visible. Check `nvidia-smi` "
            "and CUDA_VISIBLE_DEVICES.",
            blocking=True,
        )

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = torch.cuda.get_device_capability(0)
    detail = (f"{name}, {vram:.1f} GB VRAM, sm_{cap[0]}{cap[1]}, "
              f"torch {torch.__version__}, CUDA {torch.version.cuda}")

    # A Blackwell consumer part on a wheel built without sm_120 kernels imports
    # and reports a device, then fails at the first kernel launch.
    try:
        arches = torch.cuda.get_arch_list()
    except Exception:
        arches = []
    target = f"sm_{cap[0]}{cap[1]}"
    if arches and target not in arches:
        return Check(
            "CUDA / torch", "warn",
            f"{detail} -- wheel was not built for {target} (has: {', '.join(arches)})",
            f"Kernels may fail at launch. A wheel with {target} support is needed; "
            "try a newer CUDA index (cu130/cu128) or a nightly build.",
        )
    return Check("CUDA / torch", "ok", detail)


def _sample_nvml(duration_s: float, interval_s: float):  # noqa: ANN202
    from .power import NvmlGpuSampler

    s = NvmlGpuSampler(interval_s)
    with s:
        time.sleep(duration_s)
    return s.trace


def check_nvml_rate(duration_s: float = 2.0, interval_s: float = 0.02) -> Check:
    """Measure the achievable GPU power sample rate, idle and under load.

    This decides whether energy for the fastest format is a measurement or a
    guess: a sub-second load window needs roughly ten samples to integrate.

    Both states are probed because they can differ by an order of magnitude. An
    idle NVIDIA GPU -- especially a laptop part under Dynamic Boost -- sits in a
    deep power state, and each NVML query may have to wake it. During an actual
    cold start the GPU is active, so the *loaded* rate is the one that governs
    real measurements. Reporting only the idle figure would condemn a setup that
    works fine in practice.
    """
    try:
        idle = _sample_nvml(duration_s, interval_s)
    except Exception as exc:
        return Check("GPU power sampling", "fail", f"NVML unavailable: {exc}",
                     "pip install nvidia-ml-py; confirm the NVIDIA driver is loaded.")

    idle_rate = idle.n_samples / duration_s
    idle_lat = idle.mean_read_latency_s * 1000 if idle.n_samples else float("nan")

    # Repeat with the GPU busy, so the number reflects measurement conditions.
    loaded_rate = loaded_lat = None
    try:
        import threading

        import torch

        if torch.cuda.is_available():
            stop = threading.Event()

            def spin() -> None:
                a = torch.randn(2048, 2048, device="cuda")
                while not stop.is_set():
                    a = a @ a.T
                    a = a / (a.norm() + 1e-6)
                torch.cuda.synchronize()

            worker = threading.Thread(target=spin, daemon=True)
            worker.start()
            time.sleep(0.5)  # let the GPU leave its idle power state
            try:
                busy = _sample_nvml(duration_s, interval_s)
                loaded_rate = busy.n_samples / duration_s
                loaded_lat = busy.mean_read_latency_s * 1000 if busy.n_samples else None
            finally:
                stop.set()
                worker.join(timeout=5)
    except Exception:
        pass

    detail = f"idle {idle_rate:.1f}/s ({idle_lat:.2f} ms/read)"
    if loaded_rate is not None:
        detail += f", under load {loaded_rate:.1f}/s ({loaded_lat:.2f} ms/read)"
    detail += f"; requested {1 / interval_s:.0f}/s"

    # The loaded rate is what governs real measurements; fall back to idle.
    effective = loaded_rate if loaded_rate is not None else idle_rate

    if effective < MIN_SAMPLE_RATE_HZ:
        return Check(
            "GPU power sampling", "fail", detail,
            f"At {effective:.1f}/s a 0.7 s GGUF load yields ~{effective * 0.7:.0f} "
            f"samples, below the {MIN_SAMPLES_PER_WINDOW} needed to integrate. The "
            "read latency itself is the limit, so a faster polling interval will not "
            "help. Options: report GPU energy for the fastest formats as coarse and "
            "say so, or restrict energy claims to the slower formats where the window "
            "is long enough.",
        )
    if loaded_rate is not None and idle_rate < MIN_SAMPLE_RATE_HZ <= loaded_rate:
        return Check(
            "GPU power sampling", "ok",
            detail + " — idle rate is low but the GPU is active during measurement",
        )
    return Check("GPU power sampling", "ok", detail)


def _update_period_s(timestamps: list[float], watts: list[float]) -> float | None:
    """Estimate how often the sensor actually produces a NEW value.

    Polling faster than the sensor refreshes returns the same cached number
    repeatedly. The refresh period -- not the polling rate -- is what bounds
    energy resolution, so it has to be measured rather than assumed.
    """
    changes = [timestamps[i] for i in range(1, len(watts)) if watts[i] != watts[i - 1]]
    if len(changes) < 2:
        return None
    gaps = [b - a for a, b in zip(changes, changes[1:])]
    return sum(gaps) / len(gaps)


def check_cpu_telemetry(duration_s: float = 4.0) -> Check:
    """Is CPU energy measured, and does the sensor refresh fast enough?

    Two distinct failures, often confused:

    1. The sampler falls back to a utilization *model*. CPU energy is the
       majority of the total on desktop, so this propagates everywhere.
    2. The sensor is real but refreshes slowly. HWiNFO64 polls its sensors
       every 2000 ms by default; sampling it at 100 ms then returns twenty
       copies of one value. A 0.7 s load window would contain a single distinct
       reading, which is not an energy measurement.
    """
    try:
        from .power import EstimatedCpuSampler, autodetect_cpu_sampler
    except Exception as exc:
        return Check("CPU power telemetry", "fail", str(exc))

    s = autodetect_cpu_sampler(0.1)
    if isinstance(s, EstimatedCpuSampler):
        return Check(
            "CPU power telemetry", "fail",
            "falling back to a utilization model — CPU energy will NOT be measured",
            "Windows: start HWiNFO64 in Sensors-only mode, then Settings > "
            "General/User Interface > enable 'Shared Memory Support'. "
            "Linux: ensure /sys/class/powercap/intel-rapl:0/energy_uj is readable. "
            "Otherwise the model must be disclosed explicitly in any writeup.",
        )

    with s:
        time.sleep(duration_s)
    t = s.trace
    if t.n_samples < 2:
        return Check("CPU power telemetry", "fail",
                     f"{s.source} — only {t.n_samples} sample(s) in {duration_s:.0f}s")

    distinct = len(set(t.watts))
    period = _update_period_s(t.timestamps, t.watts)
    base = f"{s.source} — {t.n_samples} polls, {distinct} distinct values, mean {t.mean_w:.1f} W"

    if distinct == 1:
        return Check(
            "CPU power telemetry", "fail",
            f"{base} — sensor did not change in {duration_s:.0f}s",
            "Either the sensor is stuck, or HWiNFO's polling period is longer than "
            "this probe. In HWiNFO: Settings > Sensors > set 'Sensor polling period' "
            "to 100-200 ms, then re-run. The default 2000 ms is far too coarse for "
            "sub-second load windows.",
        )

    if period is None:
        return Check("CPU power telemetry", "warn", base,
                     "Could not estimate the sensor refresh period; check it manually.")

    detail = f"{base}, refreshes every ~{period * 1000:.0f} ms"
    if period > 0.5:
        return Check(
            "CPU power telemetry", "fail", detail,
            f"At a {period * 1000:.0f} ms refresh, a 0.7 s GGUF load contains about "
            f"{max(1, int(0.7 / period))} distinct reading(s) — too few to integrate. "
            "Lower HWiNFO's 'Sensor polling period' (Settings > Sensors) to 100-200 ms.",
        )
    if period > 0.2:
        return Check("CPU power telemetry", "warn", detail,
                     "Refresh is usable but coarse for the fastest format; consider "
                     "lowering HWiNFO's polling period to 100 ms.")
    return Check("CPU power telemetry", "ok", detail)


def check_llama_gpu_offload() -> Check:
    """Does the installed llama-cpp-python build actually support GPU offload?

    The version string does not reveal this. A CPU-only wheel imports fine and
    silently ignores ``n_gpu_layers=-1``, so the GGUF arm would run entirely on
    the CPU while pickle and safetensors load to the GPU. The comparison would
    then be between different things, with no error anywhere to reveal it.
    """
    from ._cuda_dlls import ensure_cuda_dll_path

    registered = ensure_cuda_dll_path()
    try:
        import llama_cpp
    except ImportError:
        return Check("llama.cpp GPU offload", "warn",
                     "llama-cpp-python not installed — GGUF runs will be skipped",
                     "pip install llama-cpp-python "
                     "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
    except Exception as exc:
        # The wheel is installed but llama.dll cannot resolve its dependencies --
        # almost always the CUDA runtime DLLs.
        return Check(
            "llama.cpp GPU offload", "fail",
            f"llama_cpp installed but failed to load: {type(exc).__name__}",
            "llama.dll cannot find its CUDA dependencies (cudart/cublas). "
            f"Searched {len(registered)} CUDA directory(ies): "
            f"{registered if registered else 'none found'}. "
            "If empty, install the runtime into this venv: "
            "pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 . "
            "Otherwise the wheel's CUDA major version may not match what is present.",
        )

    ver = getattr(llama_cpp, "__version__", "?")
    try:
        supported = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as exc:
        return Check("llama.cpp GPU offload", "warn",
                     f"v{ver} — cannot determine offload support ({exc})")

    if not supported:
        return Check(
            "llama.cpp GPU offload", "fail",
            f"v{ver} is a CPU-ONLY build — n_gpu_layers=-1 will be ignored",
            "GGUF would load on the CPU while the other formats load to the GPU, "
            "making the comparison invalid. Install a CUDA build: pip install "
            "--force-reinstall llama-cpp-python --extra-index-url "
            "https://abetlen.github.io/llama-cpp-python/whl/cu124 . If no CUDA wheel "
            "is available, either drop GGUF from the comparison or disclose it as "
            "CPU-side loading.",
        )
    return Check("llama.cpp GPU offload", "ok", f"v{ver}, GPU offload supported")


def check_nsys() -> Check:
    found = shutil.which("nsys") or shutil.which("nsys.exe")
    if found:
        try:
            ver = subprocess.run([found, "--version"], capture_output=True,
                                 text=True, timeout=15).stdout.strip().splitlines()
            return Check("Nsight Systems", "ok", f"{found} ({ver[0] if ver else 'version unknown'})")
        except Exception:
            return Check("Nsight Systems", "ok", found)
    for root in (Path(r"C:/Program Files/NVIDIA Corporation/Nsight Systems"),
                 Path("/opt/nvidia/nsight-systems")):
        if root.exists():
            for pat in ("**/nsys.exe", "**/bin/nsys"):
                for hit in sorted(root.glob(pat), reverse=True):
                    return Check("Nsight Systems", "warn", f"not on PATH; found {hit}",
                                 f'Pass --nsys "{hit}" to run_nsight.py')
    return Check("Nsight Systems", "warn", "not found",
                 "Timelines are optional for the sweep but are the evidence for "
                 "the Deserialization Wall. Install NVIDIA Nsight Systems.")


# --------------------------------------------------------------------------
# Models and workspace
# --------------------------------------------------------------------------


def check_models(config_path: str | Path) -> list[Check]:
    from .config import ExperimentConfig

    try:
        cfg = ExperimentConfig.from_yaml(config_path)
    except Exception as exc:
        return [Check("Config", "fail", f"cannot load {config_path}: {exc}", blocking=True)]

    out = [Check("Config", "ok", f"{config_path} — {cfg.label}")]
    sizes: dict[str, float] = {}

    for spec in cfg.formats:
        try:
            gb = spec.resolve_size_bytes() / 1e9
            sizes[spec.name] = gb
            out.append(Check(f"Model: {spec.name}", "ok", f"{gb:.3f} GB — {spec.path}"))
        except Exception as exc:
            out.append(Check(f"Model: {spec.name}", "fail", str(exc),
                             f"Fix the path for '{spec.name}' in {config_path}",
                             blocking=True))

    fp16 = {k: v for k, v in sizes.items() if k in ("pickle", "safetensors")}
    if len(fp16) == 2:
        a, b = fp16["pickle"], fp16["safetensors"]
        drift = abs(a - b) / max(a, b) * 100
        out.append(Check(
            "Weights match across formats",
            "ok" if drift <= 2 else "fail",
            f"pickle vs safetensors differ by {drift:.1f}%",
            None if drift <= 2 else
            "These should be identical weights in two containers. A real "
            "difference means different checkpoints and the comparison is confounded.",
        ))

    if "gguf" in sizes and fp16:
        ratio = max(fp16.values()) / sizes["gguf"]
        ok = 2.4 <= ratio <= 3.8
        out.append(Check("Quantization ratio", "ok" if ok else "warn",
                         f"FP16:Q4_K_M = {ratio:.2f}x",
                         None if ok else "Expected roughly 3x; check the GGUF is the same model."))

    if fp16:
        implied = max(fp16.values()) * 1e9 / 2
        drift = abs(implied - cfg.param_count) / cfg.param_count * 100
        out.append(Check("param_count", "ok" if drift <= 10 else "fail",
                         f"config {cfg.param_count/1e9:.3f} B vs {implied/1e9:.3f} B implied "
                         f"({drift:.1f}% off)",
                         None if drift <= 10 else
                         "FP16-equivalent throughput will be wrong; correct param_count."))
    return out


def check_disk(path: str | Path = ".") -> Check:
    total, used, free = shutil.disk_usage(path)
    gb = free / 1e9
    return Check("Free disk space", "ok" if gb > 2 else "warn", f"{gb:.1f} GB free",
                 None if gb > 2 else "Traces and raw sweeps need headroom.")


def check_git() -> Check:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        return Check("Git provenance", "warn", "not a git repo or git unavailable",
                     "Results will record a null commit SHA.")
    if not sha:
        return Check("Git provenance", "warn", "no commits yet",
                     "Commit before the run so results are traceable.")
    lock = Path(".git/index.lock")
    if lock.exists():
        return Check("Git provenance", "warn", f"{sha} but .git/index.lock present",
                     "Delete .git/index.lock (and any other *.lock under .git/) to allow commits.")
    return Check("Git provenance", "ok" if not dirty else "warn",
                 f"{sha}{' (uncommitted changes)' if dirty else ''}",
                 "Commit before the final sweep so results cite a clean tree." if dirty else None)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_all(config_path: str | Path | None = None, quick: bool = False) -> Report:
    r = Report()
    steps: list[Callable[[], Check | list[Check]]] = [
        check_python, check_virtualenv, check_packages, check_pynvml_flavour,
        check_cuda,
    ]
    if not quick:
        steps += [check_nvml_rate, check_cpu_telemetry]
    steps += [check_llama_gpu_offload, check_nsys, check_disk, check_git]

    for step in steps:
        try:
            res = step()
        except Exception as exc:  # a broken check must not abort the report
            r.add(Check(step.__name__, "fail", f"check raised: {exc}"))
            continue
        r.checks.extend(res) if isinstance(res, list) else r.add(res)

    if config_path:
        r.checks.extend(check_models(config_path))
    return r
