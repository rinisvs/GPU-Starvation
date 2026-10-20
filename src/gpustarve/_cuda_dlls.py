"""Make CUDA runtime DLLs discoverable before loading llama.cpp (Windows).

``llama-cpp-python``'s CUDA wheels link against ``cudart``, ``cublas`` and
``cublasLt``. Those DLLs are not installed system-wide by pip -- they arrive as
``nvidia-*`` packages nested under ``site-packages/nvidia/*/bin``, and torch adds
them to its own search path at import time but not to the process-wide one.

The result is a confusing failure: ``llama.dll`` is present at exactly the path
in the error message, yet ``ctypes.CDLL`` reports it missing, because Windows
cannot resolve *its dependencies*. Since Python 3.8, ``PATH`` is not consulted
for this -- directories must be registered via ``os.add_dll_directory``.

Minor-version compatibility within CUDA 12.x means a wheel built against 12.4
generally runs against a 12.8 runtime, so pairing a cu124 llama.cpp with a cu128
torch normally works once the directories are registered.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_applied = False


def _candidate_dirs() -> list[Path]:
    """Directories that may hold CUDA runtime DLLs, most likely first."""
    dirs: list[Path] = []

    # torch bundles its CUDA runtime under site-packages/nvidia/*/bin
    try:
        import torch

        site = Path(torch.__file__).resolve().parent.parent
        dirs.extend(sorted((site / "nvidia").glob("*/bin")))
        dirs.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        pass

    # A system CUDA toolkit, if one is installed
    for var in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(var)
        if root:
            dirs.append(Path(root) / "bin")

    return [d for d in dirs if d.is_dir()]


def ensure_cuda_dll_path() -> list[str]:
    """Register CUDA DLL directories with the Windows loader.

    Idempotent and safe to call on any platform; a no-op off Windows. Returns
    the directories that were registered, for diagnostics.
    """
    global _applied
    if _applied or sys.platform != "win32":
        return []

    registered: list[str] = []
    for d in _candidate_dirs():
        try:
            os.add_dll_directory(str(d))
            registered.append(str(d))
        except (OSError, AttributeError):
            continue

    _applied = True
    return registered


def import_llama_cpp():  # noqa: ANN201
    """Import ``llama_cpp`` with the CUDA search path prepared.

    Raises the original ImportError/RuntimeError if it still fails, so the
    caller can report the real reason rather than a generic "not installed".
    """
    ensure_cuda_dll_path()
    import llama_cpp

    return llama_cpp
