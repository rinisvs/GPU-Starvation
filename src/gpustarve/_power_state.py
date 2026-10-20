"""Hold the machine awake for the duration of a sweep (Windows).

A sweep runs unattended for an hour with deliberate idle gaps between runs --
exactly the pattern that triggers system sleep. Sleeping mid-sweep is worse than
a slow run: the measurement spanning it records an absurd duration, the CUDA
context may be lost, and NVML handles can go stale, so the failure is not always
obvious in the output.

``SetThreadExecutionState`` asks Windows not to sleep while the sweep holds the
request. The display is deliberately *not* pinned: only ``ES_SYSTEM_REQUIRED``
is asserted, so the screen may still blank while the machine keeps working.

On other platforms this is a no-op and the caller is told so, rather than being
given a false assurance.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_system_awake() -> Iterator[bool]:
    """Prevent system sleep for the duration of the block.

    Yields True if the request was accepted, False otherwise. The previous
    execution state is restored on exit, including when the block raises.
    """
    if sys.platform != "win32":
        yield False
        return

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadExecutionState.restype = ctypes.c_uint
        previous = kernel32.SetThreadExecutionState(
            ctypes.c_uint(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        )
        acquired = previous != 0
    except Exception:
        yield False
        return

    try:
        yield acquired
    finally:
        try:
            kernel32.SetThreadExecutionState(ctypes.c_uint(ES_CONTINUOUS))
        except Exception:
            pass
