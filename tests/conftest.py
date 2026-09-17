"""Suite-wide protection: an isolated cache, no network, no IERS downloads.

Every module in this suite gets these, so a new test file cannot silently start
reading the developer's real ``~/.cache/orbit-cache`` or querying the live
SatChecker service. The one test that is *meant* to reach the network —
``test_sim-vis.py::test_simulation_runs_with_config``, the suite's single live
integration check — opts out with ``@pytest.mark.allow_network``.

The network block covers two layers deliberately. Patching
``satchecker_client.client._http_get`` is what a test replaces when it wants to
serve a recorded response; patching ``urllib.request.urlopen`` underneath it is
what catches a code path that reaches the service some *other* way. That second
layer is not hypothetical: tabsim held its own copy of the client's private
transport helper, and a search through it bypassed every stub a test installed,
which is how a malformed reply became "no satellite matches this name".
"""

from __future__ import annotations

import urllib.request

import pytest

from satchecker_client import client


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "allow_network: this test may reach the live SatChecker service",
    )


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the managed cache at a temporary directory for every test.

    Without this the tests would read and write the developer's real
    ``~/.cache/orbit-cache``, which would make them order-dependent on whatever
    a previous simulation happened to fetch.
    """
    monkeypatch.setenv("ORBIT_CACHE_DIR", str(tmp_path / "orbit-cache"))
    return tmp_path / "orbit-cache"


@pytest.fixture(autouse=True)
def no_iers_download():
    """Never fetch Earth-orientation data while propagating in a test.

    Astropy downloads IERS tables on first use and then warns or blocks for
    dates outside them. A propagation test that quietly depends on that is both
    slow and non-reproducible, so the tables stay at their bundled values and
    out-of-range dates are accepted at reduced accuracy rather than raising.
    """
    from contextlib import ExitStack

    from astropy.utils import iers

    with ExitStack() as stack:
        stack.enter_context(iers.conf.set_temp("auto_download", False))
        if hasattr(iers.conf, "iers_degraded_accuracy"):  # astropy >= 5.1
            stack.enter_context(iers.conf.set_temp("iers_degraded_accuracy", "ignore"))
        yield


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    """Fail loudly if a test reaches the network without saying it means to."""
    if request.node.get_closest_marker("allow_network"):
        return

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "test made an unexpected SatChecker request; serve it through "
            "satchecker_client.client._http_get (tabsim must reach the service "
            "only through the public client)"
        )

    monkeypatch.setattr(client, "_http_get", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
