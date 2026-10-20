"""Format-specific model loaders with phase instrumentation.

Each loader yields the same phase vocabulary so that the phase table is
comparable across formats:

    tokenizer   -- tokenizer construction
    deserialize -- turning bytes into tensors (CPU-bound for pickle, mmap otherwise)
    transfer    -- host to device movement
    generate    -- first token emission

Where a format genuinely collapses two phases (GGUF loads straight into
device-mapped memory), the collapsed phase is reported and the others are
recorded as ``0.0`` with an explicit note, rather than being silently omitted.
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

try:  # optional; harness degrades gracefully without it
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    import torch.cuda.nvtx as nvtx

    _HAS_NVTX = True
except Exception:  # pragma: no cover
    _HAS_NVTX = False


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Annotate a region for Nsight Systems, if available."""
    if _HAS_NVTX:
        nvtx.range_push(name)
    try:
        yield
    finally:
        if _HAS_NVTX:
            nvtx.range_pop()


@dataclass
class PhaseTimings:
    """Wall-clock duration of each load phase, in seconds."""

    tokenizer: float = 0.0
    deserialize: float = 0.0
    transfer: float = 0.0
    generate: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def load_total(self) -> float:
        """Everything before generation begins."""
        return self.tokenizer + self.deserialize + self.transfer

    @property
    def ttft(self) -> float:
        """Time to first token: the full cold path."""
        return self.load_total + self.generate


class Timer:
    """Monotonic timer that synchronizes CUDA before stopping.

    Without the synchronize, a load timer measures kernel *dispatch* rather
    than completion, which understates transfer time.
    """

    def __init__(self, sync: bool = True) -> None:
        self.sync = sync and torch is not None and torch.cuda.is_available()

    @contextmanager
    def __call__(self) -> Iterator[list[float]]:
        holder: list[float] = []
        t0 = time.perf_counter()
        try:
            yield holder
        finally:
            if self.sync:
                torch.cuda.synchronize()
            holder.append(time.perf_counter() - t0)


def drop_caches() -> None:
    """Best-effort eviction so the next load is genuinely cold.

    On Linux with privileges this drops the page cache outright. Elsewhere it
    falls back to Python-level cleanup, which is weaker; the harness records
    which path was taken so warm-cache contamination is visible in the results
    rather than hidden.
    """
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def load_hf(path: str, device: str, prompt: str, max_new_tokens: int, use_safetensors: bool):
    """Load a HuggingFace checkpoint (pickle .bin or .safetensors).

    The two formats share a code path deliberately: the *only* difference is
    the ``use_safetensors`` flag, so any measured difference is attributable to
    deserialization strategy and nothing else.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t = Timer()
    phases = PhaseTimings()

    with nvtx_range("tokenizer"), t() as h:
        tok = AutoTokenizer.from_pretrained(path)
    phases.tokenizer = h[0]

    with nvtx_range("deserialize"), t() as h:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            use_safetensors=use_safetensors,
            low_cpu_mem_usage=False,
            device_map=None,
        )
    phases.deserialize = h[0]

    with nvtx_range("transfer"), t() as h:
        model = model.to(device)
    phases.transfer = h[0]

    with nvtx_range("generate"), t() as h:
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    phases.generate = h[0]

    return model, phases


def load_gguf(path: str, prompt: str, max_new_tokens: int, n_gpu_layers: int = -1,
              use_mmap: bool = True):
    """Load a GGUF model through llama.cpp.

    ``use_mmap=False`` forces a conventional buffered read of the same file
    instead of memory-mapping it. That is the tightest control available here:
    identical file, identical weights, identical binary, one flag, so any
    difference is attributable to loading strategy alone -- unlike the
    pickle/safetensors/GGUF comparison, which varies format, quantization and
    runtime together.

    Register it in a config as the ``gguf_nommap`` format to run that A/B.
    """
    from ._cuda_dlls import import_llama_cpp

    Llama = import_llama_cpp().Llama

    t = Timer(sync=False)
    phases = PhaseTimings()
    phases.notes.append(
        "llama.cpp performs tokenizer load, weight read and GPU upload inside a "
        "single opaque call; only the combined duration is observable, so it is "
        "reported under 'deserialize' and the other phases as 0.0"
    )

    with nvtx_range("deserialize"), t() as h:
        llm = Llama(
            model_path=path,
            n_gpu_layers=n_gpu_layers,
            use_mmap=use_mmap,
            verbose=False,
        )
    phases.deserialize = h[0]

    with nvtx_range("generate"), t() as h:
        llm(prompt, max_tokens=max_new_tokens, echo=False)
    phases.generate = h[0]

    return llm, phases


LOADERS: dict[str, Any] = {
    "pickle": lambda spec, cfg: load_hf(
        spec.path, cfg.device, cfg.prompt, cfg.max_new_tokens, use_safetensors=False
    ),
    "safetensors": lambda spec, cfg: load_hf(
        spec.path, cfg.device, cfg.prompt, cfg.max_new_tokens, use_safetensors=True
    ),
    "gguf": lambda spec, cfg: load_gguf(spec.path, cfg.prompt, cfg.max_new_tokens),
    "gguf_nommap": lambda spec, cfg: load_gguf(
        spec.path, cfg.prompt, cfg.max_new_tokens, use_mmap=False
    ),
}
