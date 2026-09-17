import signal
from contextlib import contextmanager

import pytest


@contextmanager
def fail_after(seconds: int):
    """Fail a test instead of letting it hang, for code that used to loop forever.
    Without `SIGALRM` (Windows) the code simply runs unguarded."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def on_alarm(signum, frame):
        pytest.fail(f"Did not finish within {seconds} s.")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
