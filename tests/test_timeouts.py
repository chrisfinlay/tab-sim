import signal

import pytest

from timeouts import fail_after

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"), reason="fail_after needs SIGALRM"
)


def test_fail_after_fails_a_python_loop_that_never_ends():
    with pytest.raises(pytest.fail.Exception, match="Did not finish within 1 s"):
        with fail_after(1):
            while True:
                pass


def test_fail_after_leaves_no_alarm_or_handler_behind():
    handler = signal.getsignal(signal.SIGALRM)

    with fail_after(5):
        pass

    assert signal.getsignal(signal.SIGALRM) is handler
    assert signal.alarm(0) == 0


def test_fail_after_keeps_an_outer_deadline():
    with fail_after(100):
        with fail_after(5):
            pass
        remaining = signal.alarm(0)
        signal.alarm(remaining)

    assert 95 <= remaining <= 100
