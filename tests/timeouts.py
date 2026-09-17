import signal
import time
from contextlib import contextmanager

import pytest


@contextmanager
def fail_after(seconds: float):
    """Fail a test instead of letting it hang, for code that used to loop forever.

    Uses `SIGALRM`, so it only works in the main thread and only interrupts Python
    code, not a native call that never returns. A deadline that was already set, by
    an outer `fail_after` or a plugin, is kept: it still goes off when it was due,
    to its own handler, and is put back afterwards with the time it has left.
    Without `SIGALRM` (Windows) the code simply runs unguarded."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    outer_fired = False

    def on_alarm(signum, frame):
        nonlocal outer_fired
        if outer_is_first:
            outer_fired = True
            signal.signal(signal.SIGALRM, outer_handler)
            signal.raise_signal(signal.SIGALRM)
        else:
            pytest.fail(f"Did not finish within {seconds} s.")

    start = time.monotonic()
    outer_seconds, _ = signal.setitimer(signal.ITIMER_REAL, 0)
    outer_is_first = 0 < outer_seconds < seconds
    outer_handler = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, outer_seconds if outer_is_first else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, outer_handler)
        if outer_seconds and not outer_fired:
            remaining = outer_seconds - (time.monotonic() - start)
            signal.setitimer(signal.ITIMER_REAL, max(remaining, 1e-3))
