"""Skipping the one live check when SatChecker is down, without hiding a failure.

``tests/test_sim-vis.py::test_simulation_runs_with_config`` is the suite's only
test that reaches the live service, and it earns its place: a mocked transport
cannot notice the API changing shape underneath us. It also makes every pull
request depend on a service this project does not run. On 2026-09-19 that service
stopped answering and CI went red on branches that had not been near an orbit.

The client already draws the distinction to skip on, and it draws it by
classification rather than by message. ``SatCheckerTransportError`` means the
service could not be reached *at all* — connection, TLS, timeout, 5xx, or a 429
asking this client to back off — and says nothing about the satellite that
happened to be asked for. Anything else is the service answering: a 4xx, a
malformed body, a record that will not parse, an assertion about what came back.
Those are results, and this suite must report them.

Only an explicitly chained ``__cause__`` counts. ``__context__`` would also catch
an unrelated error raised while a transport failure was being handled, and a live
check that excuses itself for the wrong reason is worse than one that fails.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import pytest

from satchecker_client import SatCheckerError, SatCheckerTransportError


def service_outage(error: BaseException) -> Optional[BaseException]:
    """The transport failure *error* was raised from, or ``None`` if it was not.

    tabsim reports an outage through its own error — the catalogue search that
    could not be answered, or the coverage error naming every satellite left
    without a record — so the chain is walked rather than only its first link.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, SatCheckerTransportError):
            return error
        seen.add(id(error))
        error = error.__cause__
    return None


@contextmanager
def skip_if_satchecker_is_down():
    """Skip the guarded live check if, and only if, the service was unreachable."""
    try:
        yield
    except SatCheckerError as error:
        outage = service_outage(error)
        if outage is None:
            raise
        pytest.skip(f"SatChecker could not be reached: {outage}")
