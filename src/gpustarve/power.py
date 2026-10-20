"""Power sampling.

Two rules this module exists to enforce:

1. **Say what you actually measured.** Each sampler reports a ``source`` string
   that names the sensor it read. That string is written into the results and
   should be quoted verbatim in any paper methodology section. Do not write
   "Intel RAPL" if you read HWiNFO; write what the sampler tells you.

2. **Sampling is not free.** Every sampler records how many samples it took and
   how long its own work occupied, so the observer effect can be quantified
   rather than assumed negligible.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PowerTrace:
    """A sampled power series over one measurement window."""

    source: str
    timestamps: list[float] = field(default_factory=list)
    watts: list[float] = field(default_factory=list)
    sampler_busy_s: float = 0.0
    missed_deadlines: int = 0

    @property
    def n_samples(self) -> int:
        return len(self.watts)

    @property
    def mean_read_latency_s(self) -> float:
        """Average cost of one sensor read. Bounds the achievable sample rate."""
        return self.sampler_busy_s / self.n_samples if self.n_samples else float("nan")

    def resolution_warning(self, window_s: float) -> str | None:
        """Flag windows too short for the achieved sample rate to integrate.

        Trapezoidal integration over two or three points on a fast-changing
        signal is not a measurement of energy. The fastest formats have the
        shortest load windows, so this check matters most exactly where the
        headline energy reduction is claimed.
        """
        if self.n_samples < 2:
            return f"only {self.n_samples} sample(s) over {window_s:.2f}s -- energy is undefined"
        if self.n_samples < 10:
            return (
                f"only {self.n_samples} samples over {window_s:.2f}s -- energy integral is "
                "coarse; reduce power_interval_s or use a faster sensor path"
            )
        return None

    @property
    def mean_w(self) -> float:
        return sum(self.watts) / len(self.watts) if self.watts else float("nan")

    @property
    def peak_w(self) -> float:
        return max(self.watts) if self.watts else float("nan")

    def energy_j(self) -> float:
        """Trapezoidal integration of power over the sampled window.

        Returns NaN rather than 0.0 when there is insufficient data, so that a
        failed sampler cannot silently masquerade as a zero-energy result.
        """
        if len(self.watts) < 2:
            return float("nan")
        total = 0.0
        for i in range(1, len(self.watts)):
            dt = self.timestamps[i] - self.timestamps[i - 1]
            total += 0.5 * (self.watts[i] + self.watts[i - 1]) * dt
        return total


class PowerSampler(ABC):
    """Background power sampler.

    Subclasses implement :meth:`read_watts`. The base class owns the thread,
    the timing, and the self-instrumentation.
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.trace = PowerTrace(source=self.source)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    @abstractmethod
    def source(self) -> str:
        """Human-readable name of the sensor actually being read."""

    @abstractmethod
    def read_watts(self) -> float:
        """Return one instantaneous power reading in watts."""

    def available(self) -> bool:
        """Whether this sampler can run on this machine."""
        try:
            self.read_watts()
            return True
        except Exception:
            return False

    def _loop(self) -> None:
        """Sample on a fixed schedule, not a fixed sleep.

        Sleeping ``interval_s`` *after* each read makes the true period
        ``read_latency + interval_s``. When a sensor read is slow -- NVML on
        consumer Windows drivers can take >100 ms -- that silently halves or
        quarters the achieved rate, and the deficit lands hardest on the
        shortest load phases, which are exactly the ones being compared.

        Deadline scheduling keeps the period at ``interval_s`` whenever the read
        is fast enough to allow it, and records the shortfall when it is not.
        """
        t0 = time.perf_counter()
        deadline = t0
        while not self._stop.is_set():
            tick = time.perf_counter()
            try:
                w = self.read_watts()
                self.trace.timestamps.append(tick - t0)
                self.trace.watts.append(w)
            except Exception:
                pass
            self.trace.sampler_busy_s += time.perf_counter() - tick

            deadline += self.interval_s
            slack = deadline - time.perf_counter()
            if slack <= 0:
                # Read is slower than the requested period; do not accumulate
                # unbounded lag, and note that the target rate is unachievable.
                self.trace.missed_deadlines += 1
                deadline = time.perf_counter()
                continue
            self._stop.wait(slack)

    def prime(self) -> None:
        """Perform and discard one read, outside any measured window.

        The first query to an idle sensor can be orders of magnitude slower than
        subsequent ones -- on a laptop NVIDIA GPU the initial NVML call has been
        observed near 1.7 s while later calls are sub-millisecond, because the
        driver has to bring the device out of a deep power state.

        Left unprimed, that cost lands inside the very window being timed and
        corrupts the first run after every cooldown. Priming happens in
        ``__enter__`` before the worker thread starts, so it is charged to setup
        rather than to the measurement.
        """
        try:
            self.read_watts()
        except Exception:
            pass

    def __enter__(self) -> PowerSampler:
        self._stop.clear()
        self.prime()
        self.trace = PowerTrace(source=self.source)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class NvmlGpuSampler(PowerSampler):
    """GPU board power via NVML (``nvmlDeviceGetPowerUsage``)."""

    def __init__(self, interval_s: float = 0.1, device_index: int = 0) -> None:
        import pynvml

        self._nvml = pynvml
        self._nvml.nvmlInit()
        self._handle = self._nvml.nvmlDeviceGetHandleByIndex(device_index)
        super().__init__(interval_s)

    @property
    def source(self) -> str:
        return "NVML nvmlDeviceGetPowerUsage (GPU board power)"

    def read_watts(self) -> float:
        return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0


class RaplLinuxSampler(PowerSampler):
    """CPU package power from the Linux powercap RAPL interface.

    Reads the cumulative energy counter and differentiates it. This is genuine
    RAPL; on Linux it is the correct thing to cite.
    """

    RAPL_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"

    def __init__(self, interval_s: float = 0.1) -> None:
        self._last_uj: int | None = None
        self._last_t: float | None = None
        super().__init__(interval_s)

    @property
    def source(self) -> str:
        return "Linux powercap intel-rapl:0 (CPU package energy counter)"

    def read_watts(self) -> float:
        with open(self.RAPL_PATH) as fh:
            uj = int(fh.read().strip())
        now = time.perf_counter()
        if self._last_uj is None or self._last_t is None:
            self._last_uj, self._last_t = uj, now
            raise RuntimeError("priming read")
        dt = now - self._last_t
        duj = uj - self._last_uj
        self._last_uj, self._last_t = uj, now
        if dt <= 0 or duj < 0:  # counter wrap
            raise RuntimeError("invalid interval")
        return (duj / 1e6) / dt


class _HwinfoSharedMemory:
    """Low-level reader for the HWiNFO64 sensor shared-memory segment (Windows).

    Uses ctypes only -- no third-party bindings. HWiNFO64 must be running with
    *Shared Memory Support* enabled (Settings > General/User Interface).

    Struct layout is HWiNFO shared-memory format v2. The header gives the offset,
    stride and count of the reading array; each reading carries a type, an ASCII
    label, and a current value as a little-endian double.
    """

    MAP_NAME = "Global\\HWiNFO_SENS_SM2"
    SENSOR_TYPE_POWER = 5

    LABEL_OFFSET, LABEL_SIZE = 12, 128
    VALUE_OFFSET = 284
    EXPECTED_READING_SIZE = 460

    def __init__(self) -> None:
        self._handle = None
        self._view = None
        self._offset = self._stride = self._count = 0
        self.index: int | None = None
        self.label: str | None = None

    def connect(self) -> None:
        import ctypes
        import ctypes.wintypes
        import struct

        self._ctypes, self._struct = ctypes, struct
        k32 = ctypes.windll.kernel32
        k32.OpenFileMappingW.restype = ctypes.wintypes.HANDLE
        k32.MapViewOfFile.restype = ctypes.c_void_p

        self._handle = k32.OpenFileMappingW(0x0004, False, self.MAP_NAME)  # FILE_MAP_READ
        if not self._handle:
            raise RuntimeError(
                "Cannot open HWiNFO shared memory. Start HWiNFO64 in Sensors-only "
                "mode and enable Settings > General/User Interface > Shared Memory Support."
            )
        self._view = k32.MapViewOfFile(self._handle, 0x0004, 0, 0, 0)
        if not self._view:
            k32.CloseHandle(self._handle)
            raise RuntimeError("Cannot map HWiNFO shared memory view.")

        hdr = self._read(0, 44)
        self._offset = struct.unpack_from("<I", hdr, 32)[0]
        self._stride = struct.unpack_from("<I", hdr, 36)[0]
        self._count = struct.unpack_from("<I", hdr, 40)[0]

        if self._stride != self.EXPECTED_READING_SIZE:
            # Not fatal -- HWiNFO has changed this before -- but the offsets below
            # are calibrated for the known layout, so say so rather than emit
            # plausible-looking garbage.
            raise RuntimeError(
                f"Unexpected HWiNFO reading stride {self._stride} "
                f"(expected {self.EXPECTED_READING_SIZE}). Struct offsets need "
                "recalibration for this HWiNFO version; use EstimatedCpuSampler "
                "and disclose it, or update the offsets."
            )

        self.index, self.label = self._find_cpu_power()
        if self.index is None:
            raise RuntimeError(
                "HWiNFO is running but exposes no CPU package power sensor. "
                "Check that CPU sensors are enabled in HWiNFO."
            )

    def _read(self, offset: int, size: int) -> bytes:
        buf = (self._ctypes.c_char * size)()
        self._ctypes.memmove(buf, self._ctypes.c_void_p(self._view + offset), size)
        return bytes(buf)

    def _reading_type(self, i: int) -> int:
        return self._struct.unpack_from("<I", self._read(self._offset + i * self._stride, 4), 0)[0]

    def _reading_label(self, i: int) -> str:
        raw = self._read(self._offset + i * self._stride + self.LABEL_OFFSET, self.LABEL_SIZE)
        return raw.split(b"\x00")[0].decode("ascii", errors="replace").strip()

    def _find_cpu_power(self) -> tuple[int | None, str | None]:
        candidates = [
            i for i in range(self._count) if self._reading_type(i) == self.SENSOR_TYPE_POWER
        ]
        for i in candidates:  # exact match first
            if self._reading_label(i).lower() == "cpu package power":
                return i, self._reading_label(i)
        for i in candidates:  # then anything CPU-ish that is not a limit
            lab = self._reading_label(i).lower()
            if "cpu" in lab and "power" in lab and "limit" not in lab:
                return i, self._reading_label(i)
        return None, None

    def read_watts(self) -> float:
        if self.index is None:
            raise RuntimeError("HWiNFO sensor index not resolved")
        raw = self._read(self._offset + self.index * self._stride + self.VALUE_OFFSET, 8)
        return self._struct.unpack_from("<d", raw, 0)[0]

    def close(self) -> None:
        if self._view is None and self._handle is None:
            return
        k32 = self._ctypes.windll.kernel32
        k32.UnmapViewOfFile.argtypes = [self._ctypes.c_void_p]
        k32.CloseHandle.argtypes = [self._ctypes.wintypes.HANDLE]
        if self._view:
            k32.UnmapViewOfFile(self._ctypes.c_void_p(self._view))
            self._view = None
        if self._handle:
            k32.CloseHandle(self._handle)
            self._handle = None


class HwinfoSampler(PowerSampler):
    """CPU package power via HWiNFO64 shared memory (Windows).

    HWiNFO's "CPU Package Power" is derived from the processor's RAPL MSRs, so
    the underlying quantity matches :class:`RaplLinuxSampler`. The ``source``
    string is explicit about the indirection *and* reports the sensor label
    actually found, because the read path is what a reviewer will ask about.

    Two deliberate departures from the original implementation this was ported
    from: a missing sensor raises instead of returning ``0.0`` (a silent zero
    would integrate to a zero-energy result), and an unrecognized struct layout
    raises rather than reading uncalibrated offsets.

    Requires HWiNFO64 running with "Shared Memory Support" enabled.
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        self._shm = _HwinfoSharedMemory()
        self._shm.connect()  # raises if unavailable; autodetect catches it
        super().__init__(interval_s)

    @property
    def source(self) -> str:
        label = getattr(getattr(self, "_shm", None), "label", None) or "unresolved"
        return f"HWiNFO64 shared memory, sensor '{label}' (RAPL-derived package power)"

    def read_watts(self) -> float:
        return float(self._shm.read_watts())

    def __exit__(self, *exc: object) -> None:
        super().__exit__(*exc)
        self._shm.close()


class EstimatedCpuSampler(PowerSampler):
    """Utilization-based CPU power *model*. Not a measurement.

    Provided as a last resort. The ``source`` string says so explicitly, and
    that string propagates into the results file and any generated table, so a
    modelled number cannot be mistaken for a measured one downstream.
    """

    def __init__(self, interval_s: float = 0.1, tdp_w: float = 45.0, idle_fraction: float = 0.15):
        import psutil

        self._psutil = psutil
        self.tdp_w = tdp_w
        self.idle_fraction = idle_fraction
        super().__init__(interval_s)

    @property
    def source(self) -> str:
        return (
            f"MODELLED (not measured): P_idle + u*(TDP-P_idle), "
            f"TDP={self.tdp_w}W, P_idle={self.idle_fraction:.0%} of TDP"
        )

    def read_watts(self) -> float:
        u = self._psutil.cpu_percent(interval=None) / 100.0
        p_idle = self.idle_fraction * self.tdp_w
        return p_idle + u * (self.tdp_w - p_idle)


def autodetect_cpu_sampler(interval_s: float = 0.1, tdp_w: float = 45.0) -> PowerSampler:
    """Pick the best available CPU sampler, preferring real measurement."""
    for factory in (
        lambda: RaplLinuxSampler(interval_s),
        lambda: HwinfoSampler(interval_s),
    ):
        try:
            s = factory()
            if s.available():
                return s
        except Exception:
            continue
    return EstimatedCpuSampler(interval_s, tdp_w=tdp_w)
