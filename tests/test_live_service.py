"""What the live check skips for, and what it must still fail for.

A guard that is too generous turns a regression into a green run with a skip
nobody reads, so each case below is one the guard has to tell apart. The second
half drives the real resolver, because the failures that are hardest to tell
apart are not raised by the transport at all: the client *synthesises* a
transport failure once SatChecker answers a long enough run of requests with the
same status, which is how a renamed endpoint comes to look like an outage.
"""

import pytest

from satchecker_client import (
    SatCheckerError,
    SatCheckerRateLimitError,
    SatCheckerResponseError,
    SatCheckerTransportError,
)
from satchecker_client.service import RESPONSE_WALL_THRESHOLD

from tabsim import orbit

from live_service import service_outage, skip_if_satchecker_is_down
from orbit_helpers import ISS_EPOCH_JD, ISS_NORAD_ID, stub_endpoints, tle_record_at


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


# The resolver's own failures: what reaches the guard when nothing was raised at
# the point the guard wraps, and the coverage error is built from a whole batch.


def coverage_failure(norad_ids, epoch_jd=ISS_EPOCH_JD):
    """Resolve *norad_ids* and return the coverage error it could not avoid."""
    with pytest.raises(orbit.OrbitError) as excinfo:
        orbit.require_complete_coverage(orbit.resolve_orbits(norad_ids, epoch_jd))
    return excinfo.value


def test_a_wall_of_rejections_is_not_an_outage(monkeypatch):
    """Every record request answered HTTP 404: an endpoint change, not a silence.

    Past ``RESPONSE_WALL_THRESHOLD`` identical statuses the client stops sending
    and files a transport failure of its own against the IDs it never sent, so
    most of this resolution's errors *are* transport errors. The few real 404s it
    collected first are what says the service was answering, and the live check
    exists to catch exactly this.
    """
    ids = [ISS_NORAD_ID + i for i in range(RESPONSE_WALL_THRESHOLD + 5)]
    stub_endpoints(
        monkeypatch,
        tle_default=SatCheckerResponseError("HTTP 404", status=404),
        omm_default=SatCheckerResponseError("HTTP 404", status=404),
    )

    assert service_outage(coverage_failure(ids)) is None


def test_one_satellite_answered_for_keeps_a_timeout_from_excusing_the_run(
    monkeypatch,
):
    """A 404 for one satellite and a timeout for another is not an outage."""
    answered, silent = ISS_NORAD_ID, ISS_NORAD_ID + 1
    stub_endpoints(
        monkeypatch,
        tle={
            answered: SatCheckerResponseError("HTTP 404", status=404),
            silent: SatCheckerTransportError("timed out"),
        },
        omm={
            answered: SatCheckerResponseError("HTTP 404", status=404),
            silent: SatCheckerTransportError("timed out"),
        },
    )

    assert service_outage(coverage_failure([answered, silent])) is None


def test_an_archive_that_answered_is_not_forgotten_when_its_fallback_times_out(
    monkeypatch,
):
    """One satellite, first archive 404, fallback timed out — still not an outage.

    ``service_errors`` keeps only the last failure, so the 404 survives on its
    attempt alone. Reading the final gaps is not enough to honour the contract.
    """
    stub_endpoints(
        monkeypatch,
        tle={ISS_NORAD_ID: SatCheckerResponseError("HTTP 404", status=404)},
        omm={ISS_NORAD_ID: SatCheckerTransportError("timed out")},
    )

    assert service_outage(coverage_failure([ISS_NORAD_ID])) is None


def test_a_silent_service_is_still_an_outage(monkeypatch):
    """The case the guard is for: nothing answered, so nothing says otherwise.

    Alongside a satellite that resolved, so a resolution is not required to be a
    total loss before an outage can be recognised in it.
    """
    silent, served = ISS_NORAD_ID, ISS_NORAD_ID + 1
    timeout = SatCheckerTransportError("the read operation timed out")
    stub_endpoints(
        monkeypatch,
        tle={silent: timeout, served: tle_record_at(served, ISS_EPOCH_JD)},
        omm={silent: timeout},
    )

    assert service_outage(coverage_failure([silent, served])) is timeout
