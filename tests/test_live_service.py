"""What the live check skips for, and what it must still fail for.

A guard that is too generous turns a regression into a green run with a skip
nobody reads, so each case below is one the guard has to tell apart.
"""

import pytest

from satchecker_client import (
    SatCheckerError,
    SatCheckerRateLimitError,
    SatCheckerResponseError,
    SatCheckerTransportError,
)

from live_service import service_outage, skip_if_satchecker_is_down


def chained(error, cause):
    """*error* raised from *cause*, as ``raise ... from ...`` leaves it."""
    error.__cause__ = cause
    return error


@pytest.mark.parametrize(
    "error",
    [
        SatCheckerTransportError("timed out"),
        SatCheckerRateLimitError("429", retry_after=30.0),
        # How tabsim reports an outage on both of its routes: the catalogue search
        # that could not be answered, and the coverage error naming the satellites
        # left without a record. Neither is a transport error itself, and the chain
        # is followed however deep it runs.
        chained(SatCheckerError("no search"), SatCheckerTransportError("timed out")),
        chained(
            SatCheckerError("no coverage"),
            chained(SatCheckerError("no search"), SatCheckerTransportError("timed out")),
        ),
    ],
    ids=["transport", "rate-limit", "chained", "chained-twice"],
)
def test_an_unreachable_service_skips_the_check(error):
    with pytest.raises(pytest.skip.Exception, match="could not be reached"):
        with skip_if_satchecker_is_down():
            raise error


@pytest.mark.parametrize(
    "error",
    [
        # The service answered, and what it said was unusable: a renamed endpoint,
        # a malformed body, a response shape that changed. The live check exists
        # to catch exactly this.
        SatCheckerResponseError("HTTP 404", status=404),
        SatCheckerError("no record for this satellite"),
    ],
    ids=["response", "unchained"],
)
def test_a_service_that_answered_still_fails_the_check(error):
    with pytest.raises(SatCheckerError):
        with skip_if_satchecker_is_down():
            raise error


def test_a_failure_merely_handled_beside_an_outage_is_not_one():
    """Implicit chaining does not excuse a test: ``__context__`` is not a cause.

    Raised inside an ``except SatCheckerTransportError`` block, so the outage is
    on ``__context__`` and nothing says the two are related.
    """
    try:
        raise SatCheckerTransportError("timed out")
    except SatCheckerTransportError:
        with pytest.raises(SatCheckerError, match="something else"):
            with skip_if_satchecker_is_down():
                raise SatCheckerError("something else")


def test_a_failed_assertion_is_not_swallowed():
    """The guard covers the service's failures, not the test's own."""
    with pytest.raises(AssertionError, match="no satellites"):
        with skip_if_satchecker_is_down():
            assert False, "no satellites"


def test_service_outage_survives_a_cause_cycle():
    """A chain that points back at itself must not be walked forever."""
    first = SatCheckerError("first")
    chained(first, chained(SatCheckerError("second"), first))

    assert service_outage(first) is None
