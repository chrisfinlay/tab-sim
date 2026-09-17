import signal
import time

import pytest

from timeouts import fail_after

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"), reason="fail_after needs SIGALRM"
)

# These tests do not assume that the alarm is theirs alone: a timeout plugin may
# have a deadline of its own running, which is to be left as it was found.


def remaining_seconds():
    return signal.getitimer(signal.ITIMER_REAL)[0]


def test_fail_after_fails_a_python_loop_that_never_ends():
    with pytest.raises(pytest.fail.Exception, match="Did not finish within 1 s"):
        with fail_after(1):
            while True:
                pass


def test_fail_after_leaves_no_alarm_or_handler_behind():
    handler = signal.getsignal(signal.SIGALRM)
    inherited = remaining_seconds()

    with fail_after(5):
        pass

    assert signal.getsignal(signal.SIGALRM) is handler
    assert (remaining_seconds() > 0) == (inherited > 0)
    assert remaining_seconds() <= inherited


def test_fail_after_puts_back_an_outer_deadline_with_the_time_it_has_left():
    with fail_after(100):
        outer = remaining_seconds()
        start = time.monotonic()
        with fail_after(50):
            time.sleep(1.2)
        remaining = remaining_seconds()
        elapsed = time.monotonic() - start

    # Not the full time again, which would be more than a second out
    assert elapsed >= 1.2
    assert abs(remaining - (outer - elapsed)) < 0.5


def test_fail_after_keeps_an_earlier_outer_deadline():
    handler = signal.getsignal(signal.SIGALRM)
    inherited = remaining_seconds()
    start = time.monotonic()

    with pytest.raises(pytest.fail.Exception, match="Did not finish within 1 s"):
        with fail_after(1):
            with fail_after(30):
                while True:
                    pass

    assert time.monotonic() - start < 5
    assert signal.getsignal(signal.SIGALRM) is handler
    assert (remaining_seconds() > 0) == (inherited > 0)


@pytest.mark.parametrize("seconds", [0, -1])
def test_fail_after_refuses_no_time_at_all_and_changes_nothing(seconds):
    with fail_after(100):
        handler = signal.getsignal(signal.SIGALRM)
        outer = remaining_seconds()

        with pytest.raises(ValueError, match="needs a time to allow"):
            with fail_after(seconds):
                pass

        assert signal.getsignal(signal.SIGALRM) is handler
        assert outer - 1 < remaining_seconds() <= outer
