import signal
import time
from contextlib import contextmanager

import pytest


@contextmanager
def fail_after(seconds: int):
    """Fail a test instead of letting it hang, for code that used to loop forever.

    Uses `SIGALRM`, so it only works in the main thread and only interrupts Python
    code, not a native call that never returns. A deadline that was already set, by
    an outer `fail_after` or a plugin, is put back with the time it has left. Without
    `SIGALRM` (Windows) the code simply runs unguarded."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def on_alarm(signum, frame):
        pytest.fail(f"Did not finish within {seconds} s.")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    start = time.monotonic()
    outer_seconds = signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        if outer_seconds:
            elapsed = int(time.monotonic() - start)
            signal.alarm(max(1, outer_seconds - elapsed))
