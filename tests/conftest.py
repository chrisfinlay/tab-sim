"""Suite-wide protection: an isolated cache, no network, no IERS downloads.

Every module gets these autouse fixtures, so a new test file cannot silently read
the developer's real ``~/.cache/orbit-cache`` or query the live SatChecker
service. The suite's one live check, ``test_sim-vis.py::
test_simulation_runs_with_config``, opts out with ``@pytest.mark.allow_network``.
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
    """Point the managed cache at a temporary directory for every test."""
    monkeypatch.setenv("ORBIT_CACHE_DIR", str(tmp_path / "orbit-cache"))
    return tmp_path / "orbit-cache"


@pytest.fixture(autouse=True)
def no_iers_download():
    """Never fetch Earth-orientation data while propagating in a test.

    The bundled tables stay in force and out-of-range dates are accepted at
    reduced accuracy rather than raising.
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
    """Fail loudly if a test reaches the network without saying it means to.

    Both transports are blocked: ``_http_get`` is the seam a test replaces to
    serve a recorded reply, and ``urlopen`` beneath it catches code reaching the
    service some other way — which a private copy of the client's transport did.
    """
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
