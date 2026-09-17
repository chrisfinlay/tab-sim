"""tabsim.orbit as an adapter over satchecker-client's resolver and replay API.

``tests/test_orbit.py`` says what tabsim's orbit policy *is*, and goes on saying
it after this change: source precedence, the age ceilings, the checksum rule,
the coverage errors and the frozen-replay contract are tabsim's, whoever
executes them. This module says who executes them. Every test here is about the
seam — that the selection, acquisition and serialisation tabsim used to
implement itself now come from the client's public API, and that what comes back
through that seam still has tabsim's shape, tabsim's labels and tabsim's
wording.

**The seam.** ``tabsim.orbit`` already does ``import satchecker_client as
satchecker`` and calls ``satchecker.nearest_endpoints_for(...)`` at call time
rather than binding it at import. The adoption keeps exactly that one
convention for everything else it takes from the client, so every function these
tests patch is an attribute of the package module:

    satchecker.resolve_orbits          satchecker.save_orbits_for_reuse
    satchecker.read_extra_orbit_dir    satchecker.save_replay_orbits
                                       satchecker.load_replay_orbits

A ``from satchecker_client import resolve_orbits`` at module scope would bind
the function at import and make it unpatchable here — and, for the endpoints,
would freeze the archive choice at import time. :data:`CLIENT_SEAM_NAMES` is the
list, and :func:`spy_on` is the only way these tests install anything on it.

**Public API only.** Nothing here patches the client's resolver internals, its
candidate selection or its serialisation helpers: a test that reached in there
would pass against an implementation nobody else could use. Delegation spies
wrap the *real* function wherever the behaviour is also being asserted, so a
test proves both that the call happened and that the answer is right. Where a
test needs a result tabsim could not have produced by itself — an outage that
blocked a fallback, a refresh that failed — it builds the client's own result
dataclasses and hands them back through the same public seam.

Everything runs offline: ``tests/conftest.py``'s autouse fixtures cover this
module as they cover every other, and a request that escapes the endpoint stubs
fails the test rather than reaching SatChecker.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import satchecker_client
from satchecker_client import (
    EndpointAttempt,
    OrbitInputError,
    ResolutionEvent,
    SatCheckerResponseError,
    SatCheckerTransportError,
    client,
    validated_record,
)
from satchecker_client import OrbitResolution as ClientOrbitResolution
from satchecker_client import RejectedOrbit as ClientRejectedOrbit
from satchecker_client import ResolvedOrbit as ClientResolvedOrbit
from satchecker_client.cache import TextOrbitCache, read_legacy_tle_records
from satchecker_client.records import record_elements

from tabsim import config as config_module
from tabsim import orbit
from tabsim.tle import get_satellite_positions

from compat import pr44_loader
from orbit_helpers import (
    CHECKSUM_STATUS_FIELD,
    GPS_EPOCH_JD,
    GPS_NORAD_ID,
    ISS_EPOCH_JD,
    ISS_NORAD_ID,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    comparable_record,
    compat_fixture,
    forbidden,
    omm_record_at,
    stub_endpoints,
    tle_record_at,
    without_checksum,
)


# ---------------------------------------------------------------------------
# The vocabulary each side is written in
# ---------------------------------------------------------------------------

#: The client's stable source codes. Mirrored here rather than imported from the
#: client's own test package, which is not installed with it.
GROUP_EXTRA, GROUP_REMOTE = "extra", "remote"
SOURCE_EXTRA, SOURCE_CACHE, SOURCE_SERVICE = "extra", "cache", "service"

#: ``EndpointAttempt.status`` values.
ATTEMPT_EMPTY = "empty"
ATTEMPT_USABLE = "usable"
ATTEMPT_OVER_AGE = "over_age"
ATTEMPT_ERROR = "error"
ATTEMPT_NOT_SENT = "not_sent"

#: ``RejectedOrbit.reason_code`` values.
REASON_OVER_AGE, REASON_INVALID = "over_age", "invalid"

#: ``OrbitResolution.unavailable`` classifications.
UNAVAILABLE_ABSENT = "absent"
UNAVAILABLE_OFFLINE = "offline"
UNAVAILABLE_NOT_ATTEMPTED = "not_attempted"
UNAVAILABLE_INVALID_LOCAL = "invalid_local"

#: The ``ResolutionEvent.code`` values these tests supply or expect.
EVENT_OUTAGE = "outage"
EVENT_REFRESH_FAILED = "refresh_failed"
EVENT_CANDIDATE_REJECTED = "candidate_rejected"

#: The client's endpoint labels, which its own ``nearest_endpoints_for``
#: produces and which tabsim's display labels embed.
TLE_ENDPOINT, OMM_ENDPOINT = "nearest-TLE", "nearest-OMM"

#: tabsim's display labels. These are the strings logs, coverage errors and
#: saved provenance have always used, and they are not the client's codes: an
#: application says "extra_orbit_dir" because that is the configuration key the
#: user set, and names the archive that answered because the two behave
#: differently near the handover.
LABEL_EXTRA = "extra_orbit_dir"
LABEL_CACHE = "managed per-satellite cache"
LABEL_TLE = "SatChecker (nearest-TLE)"
LABEL_OMM = "SatChecker (nearest-OMM)"

#: The observation every constructed result is measured against.
OBS_EPOCH_JD = ISS_EPOCH_JD

#: The client dependency this adoption is written against; see §3.6 of the plan
#: and the comment in ``pyproject.toml``.
PINNED_CLIENT_SHA = "bfc2cddc5ddeace7694b8161befc4a4e35b27584"
PINNED_REQUIREMENT = (
    "satchecker-client @ git+https://github.com/epfl-radio-astro/"
    f"satchecker-client.git@{PINNED_CLIENT_SHA}"
)
#: Revisions this replaces: PR #4's head, and the PR #5 head that predates
#: ``RejectedOrbit.error`` and ``OrbitInputError.code``. Kept so the test can say
#: "and not one of the old ones" — both install cleanly and then misreport.
SUPERSEDED_CLIENT_SHAS = (
    "06dcbf5cff5ce581bf694d689bfe08de358c5e72",
    "bb7027042b6ed6f5f76049201335d3cdc1dd1c06",
    "9096df99cab6c268041b7f954a352855500c4101",
    "e2c83d617e09de3561a8674c29b0775c9079700b",
)


# ---------------------------------------------------------------------------
# The seam, and the only way these tests touch it
# ---------------------------------------------------------------------------

#: Every client function tabsim must look up on the package module at call time.
CLIENT_SEAM_NAMES = (
    "resolve_orbits",
    "read_extra_orbit_dir",
    "save_orbits_for_reuse",
    "save_replay_orbits",
    "load_replay_orbits",
)


class Spy:
    """One client function, recorded and then called.

    Wraps the real implementation by default, so a test asserts delegation *and*
    the behaviour that follows from it rather than only the former: a spy that
    swallowed the call would pass against an adapter that does nothing useful.
    """

    def __init__(self, name, target):
        self.name = name
        self.target = target
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.target(*args, **kwargs)

    @property
    def call(self) -> tuple:
        assert len(self.calls) == 1, (
            f"expected exactly one call to satchecker_client.{self.name}, got "
            f"{len(self.calls)}"
        )
        return self.calls[0]

    def argument(self, position: int, name: str):
        """The *name* argument of the single call, given or passed positionally."""
        args, kwargs = self.call
        if len(args) > position:
            return args[position]
        assert name in kwargs, (
            f"satchecker_client.{self.name} was called without {name}"
        )
        return kwargs[name]


def spy_on(monkeypatch, name: str, answer=None) -> Spy:
    """Record calls to ``satchecker_client.<name>``; *answer* replaces the real one."""
    assert name in CLIENT_SEAM_NAMES, f"{name} is not one of the adopted seams"
    spy = Spy(name, getattr(satchecker_client, name) if answer is None else answer)
    monkeypatch.setattr(satchecker_client, name, spy)
    return spy


def deliver(monkeypatch, result) -> Spy:
    """Make the public client resolver answer with *result*, whatever is asked.

    For the cases a stubbed endpoint cannot produce: an outage that stopped a
    fallback before it was sent, a refresh that failed for a satellite the run
    resolved anyway. The result is the client's own, built from its own
    dataclasses, so what is being tested is the adaptation and nothing else.
    """
    return spy_on(monkeypatch, "resolve_orbits", lambda *args, **kwargs: result)


def forbid_replay_fallbacks(monkeypatch) -> None:
    """Every source a frozen replay must not touch, made to raise.

    Raising rather than answering nothing is the point: a replay that quietly
    resolved a satellite from the cache or the service would still produce a
    plausible simulation, just not the one it claims to reproduce.
    """
    monkeypatch.setattr(
        satchecker_client, "resolve_orbits", forbidden("the client resolver")
    )
    monkeypatch.setattr(
        satchecker_client, "read_extra_orbit_dir", forbidden("the directory scan")
    )
    monkeypatch.setattr(
        satchecker_client, "search_satellites", forbidden("the catalogue search")
    )
    monkeypatch.setattr(orbit, "TextOrbitCache", forbidden("the managed orbit cache"))
    monkeypatch.setattr(client, "fetch_nearest_tle", forbidden("the TLE endpoint"))
    monkeypatch.setattr(client, "fetch_nearest_omm", forbidden("the OMM endpoint"))


# ---------------------------------------------------------------------------
# Building client results by hand
# ---------------------------------------------------------------------------

def canonical(record) -> dict:
    """*record* as the client hands it over: kind stated, checksum provenance on it."""
    return validated_record(record, allow_missing_checksum=True)


def client_resolved(
    norad_id, source, *, endpoint=None, offset_days=0.0, provider="spacetrack",
    record=None,
) -> ClientResolvedOrbit:
    """One accepted entry, in the client's own result type."""
    epoch_jd = OBS_EPOCH_JD + offset_days
    if record is None:
        record = tle_record_at(norad_id, epoch_jd, DATA_SOURCE=provider)
    return ClientResolvedOrbit(
        norad_id=int(norad_id),
        record=canonical(record),
        source=source,
        endpoint=endpoint,
        provider=provider,
        epoch_jd=epoch_jd,
        offset_days=offset_days,
    )


def client_rejected(
    norad_id, source, *, endpoint=None, offset_days=None, provider="spacetrack",
    reason_code=REASON_OVER_AGE, ceiling_days=3.0, limit_name="remote_max_age_days",
    error=None,
) -> ClientRejectedOrbit:
    """One near-miss, in the client's own result type.

    *error* is the exception that refused *this* candidate, which the client
    carries on the rejection it kept; ``None`` for an age rejection, where
    nothing refused the record — it was read, measured and found too far away.
    """
    return ClientRejectedOrbit(
        norad_id=int(norad_id),
        source=source,
        endpoint=endpoint,
        provider=provider,
        epoch_jd=None if offset_days is None else OBS_EPOCH_JD + offset_days,
        offset_days=offset_days,
        reason_code=reason_code,
        ceiling_days=ceiling_days,
        limit_name=limit_name,
        error=error,
    )


def client_result(requested, **fields) -> ClientOrbitResolution:
    """A client resolution over *requested*, with this suite's policy stated.

    *requested* is what the **client** was asked, which is the sorted normalised
    list tabsim passes it; the original request order is tabsim's to restore.
    """
    fields.setdefault("remote_max_age_days", 3.0)
    fields.setdefault("cache_reuse_max_age_days", 1.0)
    fields.setdefault("extra_orbit_max_age_days", None)
    return ClientOrbitResolution(
        requested=[int(nid) for nid in requested],
        obs_epoch_jd=OBS_EPOCH_JD,
        **fields,
    )


def attempts_of(norad_id, *statuses) -> dict:
    """``{norad_id: [EndpointAttempt, ...]}`` over tabsim's two endpoints, in order."""
    return {
        int(norad_id): [
            EndpointAttempt(endpoint=label, status=status)
            for label, status in zip((TLE_ENDPOINT, OMM_ENDPOINT), statuses)
        ]
    }


def write_orbit_json(path, records) -> Path:
    """*records* as one column-oriented orbit table, the shape the readers read."""
    rows = [dict(record) for record in records]
    columns: list[str] = []
    for row in rows:
        columns += [column for column in row if column not in columns]
    payload = {
        column: {str(index): row.get(column) for index, row in enumerate(rows)}
        for column in columns
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def awkward_omm(norad_id, epoch_jd, **extra) -> dict:
    """An OMM carrying the two values the replay format exists to protect."""
    record = omm_record_at(norad_id, epoch_jd, **extra)
    record["ECCENTRICITY"] = 0.0066635
    record["BSTAR"] = 3.2e-05
    return record


# ---------------------------------------------------------------------------
# Delegation: one configured call to the client resolver
# ---------------------------------------------------------------------------

#: ``case -> (what tabsim is asked, what the client must be told)``.
POLICY_CASES = {
    "defaults": (
        {},
        {
            "remote_max_age_days": 3.0,
            "cache_reuse_max_age_days": 1.0,
            "extra_orbit_max_age_days": None,
            "offline": False,
            "allow_missing_checksum": False,
            "max_workers": satchecker_client.MAX_WORKERS,
        },
    ),
    "nondefault": (
        {
            "remote_max_age_days": 400.0,
            "cache_reuse_max_age_days": 0.0,
            "extra_orbit_max_age_days": 500.0,
            "allow_missing_checksum": True,
            "max_workers": 2,
        },
        {
            "remote_max_age_days": 400.0,
            "cache_reuse_max_age_days": 0.0,
            "extra_orbit_max_age_days": 500.0,
            "offline": False,
            "allow_missing_checksum": True,
            "max_workers": 2,
        },
    ),
    "offline": (
        {"offline": True},
        {
            "remote_max_age_days": 3.0,
            "cache_reuse_max_age_days": 1.0,
            "extra_orbit_max_age_days": None,
            "offline": True,
            "allow_missing_checksum": False,
            "max_workers": satchecker_client.MAX_WORKERS,
        },
    ),
}


@pytest.mark.parametrize("case", sorted(POLICY_CASES))
def test_resolution_delegates_with_tabsim_policy(case, monkeypatch, tmp_path):
    """Every selection and acquisition rule reaches the client, stated explicitly.

    The client resolver defaults nothing on purpose, so "which policy is in
    force" is entirely a question about this call. Asserting the keywords is
    asserting tabsim's policy survived the move; asserting the records is
    asserting the call did the work.
    """
    settings, expected = POLICY_CASES[case]

    extra_dir = tmp_path / "extra"
    held = tle_record_at(GPS_NORAD_ID, GPS_EPOCH_JD, DATA_SOURCE="local file")
    write_orbit_json(extra_dir / "gps.json", [held])
    served = tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD, DATA_SOURCE="spacetrack")
    nearest_tle, nearest_omm = stub_endpoints(
        monkeypatch, tle={ISS_NORAD_ID: served}
    )

    spy = spy_on(monkeypatch, "resolve_orbits")
    resolution = orbit.resolve_orbits(
        [GPS_NORAD_ID, ISS_NORAD_ID],
        OBS_EPOCH_JD,
        extra_orbit_dir=str(extra_dir),
        **settings,
    )

    # The satellites, in the acquisition order #44 established: sorted, so an
    # outage stops the run after the same requests it always did.
    assert spy.argument(0, "norad_ids") == sorted([GPS_NORAD_ID, ISS_NORAD_ID])
    # The observation's own epoch, never today's and never the grid's first sample.
    assert spy.argument(1, "obs_epoch_jd") == OBS_EPOCH_JD

    _, kwargs = spy.call
    for keyword, value in expected.items():
        assert kwargs[keyword] == value, keyword
    assert tuple(kwargs["source_order"]) == (GROUP_EXTRA, GROUP_REMOTE)
    assert kwargs["replacement"] == "strictly_fresher"
    assert kwargs["strict_response"] is True
    assert kwargs["fallback"] is True
    assert kwargs["log"] is print
    assert callable(kwargs["on_event"])

    # An explicit cache, with no path discovered inside the client.
    cache = kwargs["cache"]
    assert cache is not None
    assert all(callable(getattr(cache, name, None)) for name in ("get", "store", "path"))

    # The endpoints for this epoch, resolved at call time: the stubs installed a
    # moment ago are what arrives, which a module-level import would not give.
    assert [label for label, _ in kwargs["endpoints"]] == [TLE_ENDPOINT, OMM_ENDPOINT]
    assert [fetch for _, fetch in kwargs["endpoints"]] == [nearest_tle, nearest_omm]

    # The explicit directory, read once, by the application, through the reader.
    extra_records = kwargs["extra_records"]
    assert extra_records is not None and len(extra_records) == 1
    assert int(extra_records["NORAD_CAT_ID"].iloc[0]) == GPS_NORAD_ID

    # ... and the answer is the one the fixtures describe.
    assert resolution.resolved[GPS_NORAD_ID].source == LABEL_EXTRA
    assert resolution.resolved[GPS_NORAD_ID].record["TLE_LINE1"] == held["TLE_LINE1"]
    if expected["offline"]:
        assert nearest_tle.calls == [] and nearest_omm.calls == []
        assert ISS_NORAD_ID not in resolution.resolved
    else:
        assert nearest_tle.requested == [ISS_NORAD_ID]
        assert resolution.norad_ids() == [GPS_NORAD_ID, ISS_NORAD_ID]
        assert resolution.resolved[ISS_NORAD_ID].source == LABEL_TLE
        assert (
            resolution.resolved[ISS_NORAD_ID].record["TLE_LINE1"] == served["TLE_LINE1"]
        )


def test_empty_request_reads_nothing_and_builds_no_cache(monkeypatch, tmp_path):
    """No satellites is a legitimate configuration, and it costs nothing.

    Not a delegation test: whether an empty request reaches the client at all is
    an implementation choice. What is not a choice is that it must not open a
    directory, construct a managed cache, or send a request to find out that
    there is nothing to resolve.
    """
    monkeypatch.setattr(orbit, "TextOrbitCache", forbidden("the managed orbit cache"))
    monkeypatch.setattr(
        satchecker_client, "read_extra_orbit_dir", forbidden("the directory scan")
    )
    monkeypatch.setattr(client, "fetch_nearest_tle", forbidden("the TLE endpoint"))
    monkeypatch.setattr(client, "fetch_nearest_omm", forbidden("the OMM endpoint"))
    resolver = spy_on(monkeypatch, "resolve_orbits")

    resolution = orbit.resolve_orbits([], OBS_EPOCH_JD, extra_orbit_dir=str(tmp_path))

    assert resolution.requested == []
    assert resolution.complete
    assert resolution.records() == []
    for _, kwargs in resolver.calls:
        assert kwargs["cache"] is None
        extra_records = kwargs.get("extra_records")
        assert extra_records is None or not len(extra_records)


# ---------------------------------------------------------------------------
# Which record is used when several are held
# ---------------------------------------------------------------------------

def test_an_unusable_nearest_cached_record_does_not_hide_a_usable_one(
    monkeypatch, isolated_cache
):
    """Of the records held for one satellite, the nearest *usable* one is chosen.

    A deliberate difference from #44, which read the nearest cached record
    first, refused it for its provenance, and went on to ask SatChecker. The
    client judges each candidate before comparing epochs, so a row this run's
    checksum policy cannot use is simply not a candidate, and the nearest one
    that remains is selected.

    That is the better rule for the case it changes. The record it lands on is
    inside every limit the user set — within ``remote_max_age_days`` and within
    ``cache_reuse_max_age_days`` — so the run already holds what it asked for
    and the request would buy nothing that the configuration says it needs. And
    the same input offline has to resolve: failing a run that holds an
    acceptable record, because a *nearer* row happens to be unusable, is the
    unusable row deciding the outcome twice.

    The cost is that a strict run near an unverifiable row sends one request
    fewer and may model a slightly older record than #44 would have. The gain is
    that nothing a run cannot use changes what it does.
    """
    epoch_jd = ISS_EPOCH_JD
    unusable = tle_record_at(ISS_NORAD_ID, epoch_jd, DATA_SOURCE="permissive run")
    # Valid checksum digits, and a status saying nothing ever verified the ones
    # its source omitted: the record the strict policy refuses is the near one.
    unusable[CHECKSUM_STATUS_FIELD] = STATUS_UNVERIFIED
    usable = tle_record_at(ISS_NORAD_ID, epoch_jd + 0.5, DATA_SOURCE="spacetrack")
    TextOrbitCache(isolated_cache).store(
        ISS_NORAD_ID, pd.DataFrame([unusable, usable])
    )
    nearest_tle, nearest_omm = stub_endpoints(
        monkeypatch,
        tle={ISS_NORAD_ID: tle_record_at(ISS_NORAD_ID, epoch_jd + 0.1)},
    )

    resolution = orbit.resolve_orbits([ISS_NORAD_ID], epoch_jd)

    entry = resolution.resolved[ISS_NORAD_ID]
    assert entry.source == LABEL_CACHE
    assert entry.offset_days == pytest.approx(0.5, abs=1e-3)
    assert entry.record[CHECKSUM_STATUS_FIELD] == STATUS_VERIFIED
    # Nothing was asked: the record in hand is within the reuse threshold, so
    # the nearer record the service holds is never learned about.
    assert nearest_tle.calls == [] and nearest_omm.calls == []

    # The point of the rule: the same state resolves with nothing reachable.
    offline = orbit.resolve_orbits([ISS_NORAD_ID], epoch_jd, offline=True)
    assert offline.complete
    assert offline.resolved[ISS_NORAD_ID].offset_days == pytest.approx(0.5, abs=1e-3)

    # And the refused row is refused by policy, not by anything about the row:
    # the run that may use it gets the record at the observation's own epoch.
    permissive = orbit.resolve_orbits(
        [ISS_NORAD_ID], epoch_jd, allow_missing_checksum=True
    )
    accepted = permissive.resolved[ISS_NORAD_ID]
    assert accepted.source == LABEL_CACHE
    assert accepted.offset_days == pytest.approx(0.0, abs=1e-3)
    assert accepted.record[CHECKSUM_STATUS_FIELD] == STATUS_UNVERIFIED
    assert nearest_tle.calls == [] and nearest_omm.calls == []


# ---------------------------------------------------------------------------
# The duplicated machinery is gone
# ---------------------------------------------------------------------------

#: The selection, acquisition and serialisation helpers the client now owns.
#: Keeping a second copy is how the two implementations start disagreeing about
#: an age comparison or a projected column, with only one of them tested.
DELETED_NAMES = (
    "_add_parsed_elements",
    "_finalise_records",
    "_checked_extra_ids",
    "_select_from_extra_dir",
    "_select_from_records",
    "_cached_candidates",
    "_accept_remote",
    "_fetch_from_service",
    "_replay_record",
    "_json_scalar",
    "_own_norad_id",
    "_read_replay_ids",
    "_AGE_TOL_DAYS",
    "_REPLAY_COLUMNS",
    "_REPLAY_REQUIRED",
)

#: What callers import from :mod:`tabsim.orbit` and must keep importing, whoever
#: implements it underneath.
KEPT_PUBLIC_NAMES = (
    "OrbitError",
    "TLEError",
    "OrbitConfig",
    "TLEConfigurationError",
    "OrbitResolution",
    "ResolvedOrbit",
    "RejectedOrbit",
    "TextOrbitCache",
    "read_orbit_file",
    "orbit_cache_dir",
    "observation_epoch_jd",
    "resolve_orbits",
    "resolve_names",
    "get_orbits_by_id",
    "require_complete_coverage",
    "report_named_coverage",
    "read_extra_orbit_dir",
    "save_orbits_for_reuse",
    "save_replay_orbits",
    "load_replay_orbits",
    "REPLAY_IDS_FILE",
    "REPLAY_RECORDS_FILE",
)


def test_local_resolver_and_replay_machinery_is_removed():
    """tabsim keeps the policy and the wording; the client keeps the machinery."""
    duplicated = [name for name in DELETED_NAMES if hasattr(orbit, name)]
    assert duplicated == [], (
        "tabsim.orbit still carries its own copy of machinery the client now "
        f"owns: {duplicated}"
    )

    missing = [name for name in KEPT_PUBLIC_NAMES if not hasattr(orbit, name)]
    assert missing == [], f"tabsim.orbit no longer exports {missing}"

    # The two replay filenames are one definition, in the module that writes them.
    assert orbit.REPLAY_IDS_FILE == satchecker_client.REPLAY_IDS_FILE
    assert orbit.REPLAY_RECORDS_FILE == satchecker_client.REPLAY_RECORDS_FILE


# ---------------------------------------------------------------------------
# The element frame
# ---------------------------------------------------------------------------

def test_resolution_frame_uses_client_frame(monkeypatch):
    """The simulator's element frame is derived once, in the client, per read.

    Deriving the elements again in tabsim is how a record's lines and the
    element columns beside them start disagreeing — the exact failure the client
    documents ``frame()`` as preventing.
    """
    calls = []
    original = satchecker_client.OrbitResolution.frame

    def frame_spy(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(satchecker_client.OrbitResolution, "frame", frame_spy)

    served_tle = tle_record_at(GPS_NORAD_ID, OBS_EPOCH_JD)
    served_omm = awkward_omm(ISS_NORAD_ID, OBS_EPOCH_JD)
    stub_endpoints(
        monkeypatch, tle={GPS_NORAD_ID: served_tle}, omm={ISS_NORAD_ID: served_omm}
    )

    resolution = orbit.resolve_orbits([GPS_NORAD_ID, ISS_NORAD_ID], OBS_EPOCH_JD)
    frame = resolution.frame()

    assert len(calls) == 1, "tabsim derived the element frame itself"
    # Requested order, not sorted and not the order the archives answered in.
    assert [int(value) for value in frame["NORAD_CAT_ID"]] == [
        GPS_NORAD_ID,
        ISS_NORAD_ID,
    ]
    # Assigned column by column, so a legacy file's own element columns are
    # overwritten rather than duplicated beside the derived ones.
    assert len(set(frame.columns)) == len(frame.columns)
    for position, norad_id in enumerate([GPS_NORAD_ID, ISS_NORAD_ID]):
        for column, value in record_elements(
            resolution.resolved[norad_id].record
        ).items():
            assert frame.loc[position, column] == value


# ---------------------------------------------------------------------------
# The public shape of an adapted result
# ---------------------------------------------------------------------------

EXTRA_ID = 25544
CACHE_ID = 32260
TLE_ID = 43013
OMM_ID = 44713
OVER_AGE_ID = 48274
INVALID_ID = 49260
ERROR_ID = 51044

#: Deliberately not sorted, and not grouped by outcome: the order a run asked in
#: is the order its records, IDs and frame rows come back in.
SHAPE_REQUEST = [OMM_ID, EXTRA_ID, OVER_AGE_ID, TLE_ID, INVALID_ID, CACHE_ID, ERROR_ID]


def test_client_results_keep_tabsim_public_shape(monkeypatch):
    """A client result becomes a tabsim result without either being flattened.

    ``tabsim.orbit``'s result types are what ``tabsim.tle``, ``tabsim.config``
    and every coverage message read, and their fields are a published surface.
    The client's are a different vocabulary for the same facts. One conversion
    boundary, one direction, and the client's own object comes out of it
    unchanged — it is also what the coverage classifier reads its evidence from.
    """
    stub_endpoints(monkeypatch)  # today's own acquisition finds nothing

    failure = SatCheckerResponseError("nearest-TLE answered 503")
    refresh = SatCheckerResponseError("the refresh for the cached record failed")
    refusal = ValueError("TLE line 1 checksum is 3, expected 7")
    events = [
        ResolutionEvent(
            code=EVENT_CANDIDATE_REJECTED,
            norad_ids=(INVALID_ID,),
            source=SOURCE_EXTRA,
            error=refusal,
            details={"reason_code": REASON_INVALID},
        )
    ]
    result = client_result(
        sorted(SHAPE_REQUEST),
        resolved={
            EXTRA_ID: client_resolved(
                EXTRA_ID, SOURCE_EXTRA, offset_days=-2.0, provider="local file"
            ),
            CACHE_ID: client_resolved(CACHE_ID, SOURCE_CACHE, offset_days=0.5),
            TLE_ID: client_resolved(
                TLE_ID, SOURCE_SERVICE, endpoint=TLE_ENDPOINT, offset_days=-0.25
            ),
            OMM_ID: client_resolved(
                OMM_ID, SOURCE_SERVICE, endpoint=OMM_ENDPOINT, offset_days=1.5
            ),
        },
        rejected={
            OVER_AGE_ID: client_rejected(
                OVER_AGE_ID, SOURCE_SERVICE, endpoint=TLE_ENDPOINT, offset_days=-4.2
            ),
            INVALID_ID: client_rejected(
                INVALID_ID,
                SOURCE_EXTRA,
                provider=None,
                reason_code=REASON_INVALID,
                ceiling_days=None,
                limit_name=None,
                error=refusal,
            ),
        },
        service_errors={ERROR_ID: failure},
        refresh_errors={CACHE_ID: refresh},
        unavailable={INVALID_ID: UNAVAILABLE_INVALID_LOCAL},
        attempts=attempts_of(ERROR_ID, ATTEMPT_ERROR, ATTEMPT_ERROR),
        events=events,
    )
    deliver(monkeypatch, result)

    resolution = orbit.resolve_orbits(SHAPE_REQUEST, OBS_EPOCH_JD)

    # Order: the request's, in every accessor, whatever order the client was
    # given the IDs in or answered them in.
    assert resolution.norad_ids() == [OMM_ID, EXTRA_ID, TLE_ID, CACHE_ID]
    assert resolution.requested == SHAPE_REQUEST
    assert resolution.missing == [OVER_AGE_ID, INVALID_ID, ERROR_ID]
    assert resolution.complete is False
    assert resolution.obs_epoch_jd == OBS_EPOCH_JD
    assert resolution.remote_max_age_days == 3.0

    # Accepted entries.
    extra = resolution.resolved[EXTRA_ID]
    assert extra.record == result.resolved[EXTRA_ID].record
    assert extra.epoch_jd == OBS_EPOCH_JD - 2.0
    assert extra.offset_days == -2.0  # signed, not an age
    assert extra.age_days == 2.0
    # An explicit file is the user's own data, so tabsim has never presented a
    # provider for it, and inheriting the client's source comparison unchanged
    # would classify it as remote.
    assert extra.remote is False
    assert extra.provider is None
    for norad_id in (CACHE_ID, TLE_ID, OMM_ID):
        assert resolution.resolved[norad_id].remote is True
        assert resolution.resolved[norad_id].provider == "spacetrack"
    assert resolution.resolved[OMM_ID].offset_days == 1.5
    assert resolution.resolved[OMM_ID].age_days == 1.5

    # Rejections keep the wording a user acts on, and the structure a caller can.
    stale = resolution.rejected[OVER_AGE_ID]
    assert stale.norad_id == OVER_AGE_ID
    assert stale.source == LABEL_TLE
    assert stale.provider == "spacetrack"
    assert stale.epoch_jd == OBS_EPOCH_JD - 4.2
    assert stale.offset_days == -4.2
    assert stale.age_days == pytest.approx(4.2)
    assert stale.reason == "remote_max_age_days=3"
    assert stale.reason_code == REASON_OVER_AGE
    assert stale.ceiling_days == 3.0
    assert stale.limit_name == "remote_max_age_days"
    assert stale.endpoint == TLE_ENDPOINT

    unusable = resolution.rejected[INVALID_ID]
    assert unusable.source == LABEL_EXTRA
    assert unusable.epoch_jd is None and unusable.offset_days is None
    assert unusable.age_days is None
    assert unusable.reason_code == REASON_INVALID
    # The diagnostic is the exception the client attached to *this* rejection,
    # never parsed from the client's log prose and never taken from another
    # candidate's event.
    assert "checksum is 3, expected 7" in unusable.reason

    # Failures stay the client's own objects, in the two maps that mean
    # different things: one can make coverage fatal, the other never can.
    assert resolution.service_errors[ERROR_ID] is failure
    assert resolution.refresh_errors[CACHE_ID] is refresh

    # The client's evidence is exposed, not flattened into tabsim's vocabulary.
    assert resolution.unavailable == {INVALID_ID: UNAVAILABLE_INVALID_LOCAL}
    assert [attempt.status for attempt in resolution.attempts[ERROR_ID]] == [
        ATTEMPT_ERROR,
        ATTEMPT_ERROR,
    ]
    assert list(resolution.events) == events

    # records() hands out copies: a caller editing one must not edit the run's.
    records = resolution.records()
    assert [int(record["NORAD_CAT_ID"]) for record in records] == [
        OMM_ID,
        EXTRA_ID,
        TLE_ID,
        CACHE_ID,
    ]
    records[0]["OBJECT_NAME"] = "MUTATED"
    assert resolution.records()[0]["OBJECT_NAME"] != "MUTATED"

    # And the client's result is untouched by any of it.
    assert result.requested == sorted(SHAPE_REQUEST)
    assert result.resolved[EXTRA_ID].source == SOURCE_EXTRA
    assert result.resolved[EXTRA_ID].provider == "local file"
    assert result.rejected[OVER_AGE_ID].source == SOURCE_SERVICE
    assert result.rejected[OVER_AGE_ID].reason_code == REASON_OVER_AGE
    assert result.events == events


def corrupt_checksum(line: str) -> str:
    """*line* with its checksum digit wrong, which no policy accepts."""
    return line[:68] + str((int(line[68]) + 1) % 10)


#: ``case -> (what holds the second unusable candidate)``. The first is always
#: the alphabetically first file in the explicit directory, which is what the
#: client reads first and therefore what it keeps the rejection of.
SECOND_CANDIDATE_CASES = ["another_source", "the_same_source"]


@pytest.mark.parametrize("case", SECOND_CANDIDATE_CASES)
def test_a_rejection_reason_describes_the_candidate_its_source_names(
    case, monkeypatch, tmp_path, isolated_cache
):
    """One satellite, two unusable records: the reported reason is the kept one's.

    Only one rejection per satellite survives, and it is the first — a later
    epoch-less rejection never displaces it. So the diagnostic has to come from
    that same rejection: pairing it with the last ``candidate_rejected`` event
    reports the *other* candidate's defect beside the source that supplied
    nothing of the sort, which sends a user to the wrong file.
    """
    first = tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD, DATA_SOURCE="my archive")
    first["TLE_LINE1"] = corrupt_checksum(first["TLE_LINE1"])
    write_orbit_json(tmp_path / "a-corrupt.json", [first])

    second = tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD + 0.25)
    second[CHECKSUM_STATUS_FIELD] = "believed_fine"
    if case == "another_source":
        TextOrbitCache(isolated_cache).store(ISS_NORAD_ID, pd.DataFrame([second]))
    else:
        write_orbit_json(tmp_path / "b-unknown-status.json", [second])
    nearest_tle, nearest_omm = stub_endpoints(monkeypatch)  # both archives empty

    resolution = orbit.resolve_orbits(
        [ISS_NORAD_ID], ISS_EPOCH_JD, extra_orbit_dir=str(tmp_path)
    )

    assert nearest_tle.requested == nearest_omm.requested == [ISS_NORAD_ID]
    refused = resolution.rejected[ISS_NORAD_ID]
    assert refused.source == LABEL_EXTRA
    assert refused.reason_code == REASON_INVALID
    # The first candidate's defect, from the rejection that was kept...
    assert "checksum mismatch" in refused.reason
    # ...and not the second's, which belongs to a record this rejection is not
    # about (a different file, or the cache the source label does not name).
    assert "believed_fine" not in refused.reason

    with pytest.raises(orbit.OrbitError) as raised:
        orbit.require_complete_coverage(resolution)
    message = str(raised.value)
    assert f"{ISS_NORAD_ID}: best candidate unusable — {refused.reason}" in message


def test_tabsim_result_constructors_keep_their_positional_order():
    """Existing callers build these positionally; new metadata is keyword-only.

    A compatibility guard rather than an adoption assertion: #44 already has
    these signatures. It is here because the cheapest way to adopt the client's
    result types is to alias them, and the client's have a different field order
    — ``endpoint`` between ``source`` and ``provider`` — which silently moves
    every positional argument along by one.
    """
    record = canonical(tle_record_at(ISS_NORAD_ID, OBS_EPOCH_JD))
    accepted = orbit.ResolvedOrbit(
        ISS_NORAD_ID, record, LABEL_CACHE, "spacetrack", OBS_EPOCH_JD, -0.5
    )
    assert accepted.norad_id == ISS_NORAD_ID
    assert accepted.record == record
    assert accepted.source == LABEL_CACHE
    assert accepted.provider == "spacetrack"
    assert accepted.epoch_jd == OBS_EPOCH_JD
    assert accepted.offset_days == -0.5
    assert accepted.age_days == 0.5
    assert accepted.remote is True

    refused = orbit.RejectedOrbit(
        ISS_NORAD_ID,
        LABEL_EXTRA,
        None,
        OBS_EPOCH_JD - 9.0,
        -9.0,
        "extra_orbit_max_age_days=5.0",
    )
    assert refused.norad_id == ISS_NORAD_ID
    assert refused.source == LABEL_EXTRA
    assert refused.provider is None
    assert refused.epoch_jd == OBS_EPOCH_JD - 9.0
    assert refused.offset_days == -9.0
    assert refused.reason == "extra_orbit_max_age_days=5.0"
    assert refused.age_days == 9.0

    # Keyword-only, so the positions are the six each type has always had and a
    # seventh cannot be read as one of them by a caller written against #44.
    with pytest.raises(TypeError):
        orbit.ResolvedOrbit(
            ISS_NORAD_ID, record, LABEL_CACHE, "spacetrack", OBS_EPOCH_JD, -0.5,
            TLE_ENDPOINT,
        )
    with pytest.raises(TypeError):
        orbit.RejectedOrbit(
            ISS_NORAD_ID, LABEL_EXTRA, None, None, None, "invalid record",
            REASON_INVALID,
        )

    failure = SatCheckerResponseError("boom")
    refresh = SatCheckerResponseError("stale")
    resolution = orbit.OrbitResolution(
        [ISS_NORAD_ID, GPS_NORAD_ID],
        OBS_EPOCH_JD,
        3.0,
        {ISS_NORAD_ID: accepted},
        {GPS_NORAD_ID: refused},
        {GPS_NORAD_ID: failure},
        {ISS_NORAD_ID: refresh},
        True,
    )
    assert resolution.requested == [ISS_NORAD_ID, GPS_NORAD_ID]
    assert resolution.obs_epoch_jd == OBS_EPOCH_JD
    assert resolution.remote_max_age_days == 3.0
    assert resolution.resolved == {ISS_NORAD_ID: accepted}
    assert resolution.rejected == {GPS_NORAD_ID: refused}
    assert resolution.service_errors == {GPS_NORAD_ID: failure}
    assert resolution.refresh_errors == {ISS_NORAD_ID: refresh}
    assert resolution.offline is True
    assert resolution.missing == [GPS_NORAD_ID]


SOURCE_LABEL_CASES = [
    (SOURCE_EXTRA, None, LABEL_EXTRA),
    (SOURCE_CACHE, None, LABEL_CACHE),
    (SOURCE_SERVICE, TLE_ENDPOINT, LABEL_TLE),
    (SOURCE_SERVICE, OMM_ENDPOINT, LABEL_OMM),
]


@pytest.mark.parametrize(
    "source,endpoint,label", SOURCE_LABEL_CASES, ids=[c[2] for c in SOURCE_LABEL_CASES]
)
def test_client_source_codes_map_to_tabsim_labels(source, endpoint, label, monkeypatch):
    """Stable codes on one side, the user's own words on the other.

    The service label names the archive that answered because the two are not
    interchangeable near the handover: "SatChecker" alone would leave a log
    unable to say which of them a record came from.
    """
    stub_endpoints(monkeypatch)
    accepted_id, refused_id = ISS_NORAD_ID, GPS_NORAD_ID
    result = client_result(
        [accepted_id, refused_id],
        resolved={
            accepted_id: client_resolved(
                accepted_id, source, endpoint=endpoint, offset_days=-0.5
            )
        },
        rejected={
            refused_id: client_rejected(
                refused_id, source, endpoint=endpoint, offset_days=-9.0
            )
        },
    )
    deliver(monkeypatch, result)

    resolution = orbit.resolve_orbits([accepted_id, refused_id], OBS_EPOCH_JD)

    assert resolution.resolved[accepted_id].source == label
    assert resolution.rejected[refused_id].source == label
    assert resolution.resolved[accepted_id].remote is (source != SOURCE_EXTRA)
    assert resolution.resolved[accepted_id].provider == (
        None if source == SOURCE_EXTRA else "spacetrack"
    )


# ---------------------------------------------------------------------------
# Coverage: what counts as knowing a satellite has nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "with_rejection", [False, True], ids=["no-rejection", "over-age-rejection"]
)
def test_outage_blocked_not_sent_is_fatal_for_named_coverage(
    with_rejection, monkeypatch
):
    """An answer that was never asked for is not an answer.

    With two archives and fallback on, an unresolved satellite still needs the
    second one's reply. An outage that stopped acquisition leaves that reply
    ``not_sent``, and the client deliberately files such an ID under neither
    ``service_errors`` nor ``unavailable`` — so "no error recorded" cannot mean
    "the archives have nothing", which is how an outage used to become a
    complete-looking observation with no satellite RFI in it.
    """
    stub_endpoints(monkeypatch)
    blocked, answered = ISS_NORAD_ID, GPS_NORAD_ID
    outage = SatCheckerTransportError("SatChecker is unreachable.")
    result = client_result(
        [blocked, answered],
        resolved={
            answered: client_resolved(
                answered, SOURCE_SERVICE, endpoint=TLE_ENDPOINT, offset_days=-0.1
            )
        },
        rejected=(
            {blocked: client_rejected(blocked, SOURCE_CACHE, offset_days=-4.2)}
            if with_rejection
            else {}
        ),
        attempts=attempts_of(blocked, ATTEMPT_EMPTY, ATTEMPT_NOT_SENT),
        events=[
            ResolutionEvent(
                code=EVENT_OUTAGE,
                norad_ids=(blocked,),
                endpoint=TLE_ENDPOINT,
                error=outage,
            )
        ],
    )
    deliver(monkeypatch, result)
    resolution = orbit.resolve_orbits([blocked, answered], OBS_EPOCH_JD)

    with pytest.raises(orbit.OrbitError) as raised:
        orbit.report_named_coverage(resolution)
    message = str(raised.value)
    assert str(blocked) in message
    assert "SatChecker could not answer" in message
    assert TLE_ENDPOINT in message
    assert "unreachable" in message
    if with_rejection:
        # The measurable near-miss survives alongside the inability to ask for a
        # closer one: it is what says which ceiling to change.
        assert "4.200" in message

    # Numbered coverage rejects every gap, as it always has.
    with pytest.raises(orbit.OrbitError):
        orbit.require_complete_coverage(resolution)

    # The client's own result was read, not rewritten.
    assert blocked not in result.service_errors


def test_outage_blocked_fallback_through_real_client(monkeypatch):
    """The same case, produced by the real resolver rather than described to it.

    One worker, so the order is the order the IDs were passed in: the first
    satellite's archive answers empty, the second's raises, and the batch stops
    before the fallback goes out. That leaves the first satellite with a reply
    from one archive and silence from the other — and no failure of its own.
    """
    blocked, failing = ISS_NORAD_ID, GPS_NORAD_ID  # blocked sorts first
    outage = SatCheckerTransportError("connection refused")
    nearest_tle, nearest_omm = stub_endpoints(
        monkeypatch, tle={blocked: None, failing: outage}
    )

    resolution = orbit.resolve_orbits(
        [blocked, failing], OBS_EPOCH_JD, max_workers=1
    )

    # Asking a service that cannot serve us a different question is still asking
    # a service that cannot serve us.
    assert nearest_tle.requested == [blocked, failing]
    assert nearest_omm.calls == []

    assert [
        (attempt.endpoint, attempt.status) for attempt in resolution.attempts[blocked]
    ] == [(TLE_ENDPOINT, ATTEMPT_EMPTY), (OMM_ENDPOINT, ATTEMPT_NOT_SENT)]
    assert blocked not in resolution.service_errors
    assert blocked not in resolution.unavailable
    assert failing in resolution.service_errors

    with pytest.raises(orbit.OrbitError) as raised:
        orbit.report_named_coverage(resolution)
    for_blocked = [
        line
        for line in str(raised.value).splitlines()
        if line.strip().startswith(f"{blocked}:")
    ]
    assert for_blocked, f"no coverage line for {blocked} in:\n{raised.value}"
    assert "SatChecker could not answer" in for_blocked[0]


def test_not_sent_cache_hits_are_not_coverage_failures(monkeypatch):
    """Failing closed is about what is *missing*, not about what was not asked.

    A cached record fresh enough to suppress the request has ``not_sent`` at
    every endpoint, and a satellite resolved from an explicit file never reached
    the remote group at all. Both are ordinary, complete runs; treating an
    unsent request as uncertainty wherever it appears would make them fatal.
    """
    stub_endpoints(monkeypatch)
    cached_id, extra_id = ISS_NORAD_ID, GPS_NORAD_ID
    result = client_result(
        [cached_id, extra_id],
        resolved={
            cached_id: client_resolved(cached_id, SOURCE_CACHE, offset_days=0.25),
            extra_id: client_resolved(
                extra_id, SOURCE_EXTRA, offset_days=-1.0, provider=None
            ),
        },
        attempts=attempts_of(cached_id, ATTEMPT_NOT_SENT, ATTEMPT_NOT_SENT),
    )
    spy = deliver(monkeypatch, result)

    resolution = orbit.resolve_orbits([cached_id, extra_id], OBS_EPOCH_JD)

    assert len(spy.calls) == 1
    assert resolution.missing == []
    assert orbit.require_complete_coverage(resolution) is resolution
    assert orbit.report_named_coverage(resolution) is resolution


def evidence_cases():
    """``id -> (client fields for the unresolved ID, tabsim settings, fatal?)``.

    One satellite, asked for by name, with nothing accepted for it. What differs
    is the evidence about *why*, and only two kinds of evidence are an answer
    from the catalogue: both archives replied and had nothing, or what they had
    was measurably too old and the acquisition that measured it finished.
    """
    unresolved = ISS_NORAD_ID
    over_age = client_rejected(
        unresolved, SOURCE_SERVICE, endpoint=OMM_ENDPOINT, offset_days=-4.2
    )
    invalid = client_rejected(
        unresolved,
        SOURCE_EXTRA,
        provider=None,
        reason_code=REASON_INVALID,
        ceiling_days=None,
        limit_name=None,
    )
    return {
        # Both archives answered and neither holds one: a real catalogue answer.
        "absent": (
            dict(
                attempts=attempts_of(unresolved, ATTEMPT_EMPTY, ATTEMPT_EMPTY),
                unavailable={unresolved: UNAVAILABLE_ABSENT},
            ),
            {},
            False,
        ),
        # A record was seen and measured, and the acquisition that measured it
        # finished: the age ceiling is the answer, and it says which one.
        "completed_over_age": (
            dict(
                rejected={unresolved: over_age},
                attempts=attempts_of(unresolved, ATTEMPT_OVER_AGE, ATTEMPT_OVER_AGE),
            ),
            {},
            False,
        ),
        # An unusable local candidate is not a claim about the archives, and two
        # successful empty replies afterwards still are.
        "invalid_local_then_two_empty_replies": (
            dict(
                rejected={unresolved: invalid},
                attempts=attempts_of(unresolved, ATTEMPT_EMPTY, ATTEMPT_EMPTY),
                unavailable={unresolved: UNAVAILABLE_ABSENT},
            ),
            {},
            False,
        ),
        # Nothing was asked, so nothing was answered — about this machine's
        # state, not about the catalogue.
        "offline": (
            dict(
                offline=True,
                attempts=attempts_of(unresolved, ATTEMPT_NOT_SENT, ATTEMPT_NOT_SENT),
                unavailable={unresolved: UNAVAILABLE_OFFLINE},
            ),
            {"offline": True},
            True,
        ),
        "offline_over_age_local": (
            dict(
                offline=True,
                rejected={
                    unresolved: client_rejected(
                        unresolved,
                        SOURCE_CACHE,
                        offset_days=-10.0,
                        limit_name="remote_max_age_days",
                    )
                },
                attempts=attempts_of(unresolved, ATTEMPT_NOT_SENT, ATTEMPT_NOT_SENT),
                unavailable={unresolved: UNAVAILABLE_OFFLINE},
            ),
            {"offline": True},
            True,
        ),
        # The only local evidence was unusable and nothing reached an archive.
        "invalid_local": (
            dict(
                rejected={unresolved: invalid},
                unavailable={unresolved: UNAVAILABLE_INVALID_LOCAL},
            ),
            {},
            True,
        ),
        "not_attempted": (
            dict(unavailable={unresolved: UNAVAILABLE_NOT_ATTEMPTED}),
            {},
            True,
        ),
        # One archive answered, the other never did, and nothing says why.
        "unexplained_incomplete": (
            dict(attempts=attempts_of(unresolved, ATTEMPT_EMPTY, ATTEMPT_NOT_SENT)),
            {},
            True,
        ),
    }


EVIDENCE_CASES = evidence_cases()


@pytest.mark.parametrize("case", list(EVIDENCE_CASES))
def test_named_exclusion_requires_completed_acquisition_evidence(case, monkeypatch):
    """A name is a query, so "nothing there" is an answer — and only that is."""
    fields, settings, fatal = EVIDENCE_CASES[case]
    stub_endpoints(monkeypatch)
    unresolved = ISS_NORAD_ID
    deliver(monkeypatch, client_result([unresolved], **fields))

    resolution = orbit.resolve_orbits([unresolved], OBS_EPOCH_JD, **settings)

    if fatal:
        with pytest.raises(orbit.OrbitError) as raised:
            orbit.report_named_coverage(resolution)
        assert str(unresolved) in str(raised.value)
        return

    logged: list[str] = []
    assert orbit.report_named_coverage(resolution, log=logged.append) is resolution
    report = "\n".join(logged)
    assert "No acceptable record" in report
    assert str(unresolved) in report
    if case == "completed_over_age":
        # The exclusion says how far off, from where, and against which ceiling.
        assert "4.200" in report
        assert LABEL_OMM in report
        assert "remote_max_age_days=3" in report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("offline", [False, True], ids=["online", "offline"])
def test_the_refresh_summary_reports_only_requests_that_were_made(
    offline, monkeypatch, isolated_cache, capsys
):
    """"SatChecker did not improve N" is a result, so it needs a request.

    Offline, the same cached record is retained for a different reason: nothing
    was asked. The line that says so is the skipped-refresh count, and printing
    an acquisition *result* beside it reports on requests that were never sent.
    """
    epoch_jd = ISS_EPOCH_JD + 2.0  # inside the 3 d ceiling, outside the 1 d reuse
    TextOrbitCache(isolated_cache).store(
        ISS_NORAD_ID, pd.DataFrame([tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD)])
    )
    nearest_tle, nearest_omm = stub_endpoints(monkeypatch)  # both archives empty

    resolution = orbit.resolve_orbits([ISS_NORAD_ID], epoch_jd, offline=offline)
    out = capsys.readouterr().out

    assert resolution.resolved[ISS_NORAD_ID].source == LABEL_CACHE
    summary = "SatChecker did not improve 1 ID(s)"
    if offline:
        assert nearest_tle.calls == [] and nearest_omm.calls == []
        assert "offline: 1 ID(s) would have been refreshed" in out
        assert summary not in out
    else:
        assert nearest_tle.requested == [ISS_NORAD_ID]
        assert "offline:" not in out
        assert summary in out


def test_one_healthy_run_says_each_thing_once_and_warns_about_nothing(
    monkeypatch, isolated_cache, capsys
):
    """Cache reuse, a refresh and a fallback in one run — and one log to read.

    Events arrive per set of satellites rather than per run, so the heading a
    batch prints, the cache-hit count and the fallback explanation can each be
    emitted more than once for what a user experiences as one thing. The counts
    are the assertion; so is the absence of every line that is a report of
    something having gone wrong, none of which did.
    """
    reused, refreshed, fetched = ISS_NORAD_ID, ISS_NORAD_ID + 1, ISS_NORAD_ID + 2
    epoch_jd = ISS_EPOCH_JD
    cache = TextOrbitCache(isolated_cache)
    cache.store(
        reused, pd.DataFrame([tle_record_at(reused, epoch_jd + 0.5)])
    )  # within the reuse threshold: no request
    cache.store(
        refreshed, pd.DataFrame([tle_record_at(refreshed, epoch_jd + 2.0)])
    )  # inside the ceiling, outside the reuse threshold: asked about
    nearest_tle, nearest_omm = stub_endpoints(
        monkeypatch,
        omm={
            norad_id: omm_record_at(norad_id, epoch_jd + 0.1)
            for norad_id in (refreshed, fetched)
        },
    )  # the first archive has nothing for either, the second has both

    resolution = orbit.resolve_orbits([reused, refreshed, fetched], epoch_jd)
    out = capsys.readouterr().out

    assert resolution.complete
    assert resolution.resolved[reused].source == LABEL_CACHE
    assert resolution.resolved[refreshed].source == LABEL_OMM
    assert resolution.resolved[fetched].source == LABEL_OMM
    assert nearest_tle.requested == [refreshed, fetched]
    assert nearest_omm.requested == [refreshed, fetched]

    # One heading for the batch that was announced, and none for the fallback,
    # which the handover sentence already announced.
    assert out.count("Fetching ") == 1
    assert "nearest-TLE" in out.split("Fetching ")[1].splitlines()[0]
    assert out.count("trying nearest-OMM") == 1
    assert out.count("Cache hits") == 1
    assert out.count("Remote orbit records") == 1

    # Nothing failed, nothing was retained for want of something better, and
    # nothing about any record is unverifiable — so none of those lines exist.
    assert "warning" not in out
    assert "SatChecker did not improve" not in out
    assert "Unverified TLE" not in out
    assert "offline" not in out


def test_a_refresh_failure_reaches_the_summary_through_the_real_callbacks(
    monkeypatch, isolated_cache, capsys
):
    """The same warning, driven by the client's own event sequence.

    The parametrised case below hands tabsim a result it could not have
    produced, which tests the reporting but not the wiring: that the resolver's
    ``on_event`` callbacks, its error bookkeeping and its final result agree
    about which satellites failed a refresh and what each is continuing from.
    """
    from_cache, rescued = ISS_NORAD_ID, GPS_NORAD_ID
    epoch_jd = ISS_EPOCH_JD
    TextOrbitCache(isolated_cache).store(
        from_cache, pd.DataFrame([tle_record_at(from_cache, epoch_jd + 2.0)])
    )
    outage = SatCheckerResponseError("nearest-TLE answered 503")
    stub_endpoints(
        monkeypatch,
        tle_default=outage,
        omm={rescued: omm_record_at(rescued, epoch_jd - 0.1)},
        omm_default=outage,
    )

    resolution = orbit.resolve_orbits([from_cache, rescued], epoch_jd)
    out = capsys.readouterr().out

    assert resolution.complete
    assert orbit.require_complete_coverage(resolution) is resolution
    assert sorted(resolution.refresh_errors) == sorted([from_cache, rescued])
    assert resolution.service_errors == {}
    assert resolution.resolved[from_cache].source == LABEL_CACHE
    assert resolution.resolved[rescued].source == LABEL_OMM

    assert "warning: a SatChecker request failed for 2 ID(s)" in out
    for norad_id, label in ((from_cache, LABEL_CACHE), (rescued, LABEL_OMM)):
        assert (
            f"{norad_id} — {resolution.refresh_errors[norad_id]} (from {label})" in out
        )


@pytest.mark.parametrize("detail", [False, True], ids=["truncated", "detailed"])
def test_client_refresh_failures_produce_tabsim_summary(detail, monkeypatch, capsys):
    """A failed refresh is not fatal, and the run is not quite the one asked for.

    The only place that can say so is this warning, and what it has to name is
    the source each satellite is *continuing from* — which is not always the
    cache: an ID whose first archive failed and whose second answered is
    bookkept here too, and saying "from the cache" would describe a record the
    run never held.
    """
    if detail:
        monkeypatch.setenv("TABSIM_TLE_LOG_DETAIL", "1")
    else:
        monkeypatch.delenv("TABSIM_TLE_LOG_DETAIL", raising=False)
    stub_endpoints(monkeypatch)

    # Thirteen, one past the grouping threshold, so the truncation is exercised.
    norad_ids = [ISS_NORAD_ID + offset for offset in range(13)]
    from_cache, rescued = norad_ids[0], norad_ids[1]
    errors = {
        norad_id: SatCheckerResponseError(f"nearest-TLE answered 503 for {norad_id}")
        for norad_id in norad_ids
    }
    resolved = {
        norad_id: client_resolved(norad_id, SOURCE_CACHE, offset_days=0.75)
        for norad_id in norad_ids
    }
    resolved[rescued] = client_resolved(
        rescued, SOURCE_SERVICE, endpoint=OMM_ENDPOINT, offset_days=-0.1
    )
    events = [
        ResolutionEvent(
            code=EVENT_REFRESH_FAILED,
            norad_ids=(norad_id,),
            source=resolved[norad_id].source,
            endpoint=resolved[norad_id].endpoint,
            error=errors[norad_id],
        )
        for norad_id in norad_ids
    ]
    deliver(
        monkeypatch,
        client_result(
            norad_ids, resolved=resolved, refresh_errors=errors, events=events
        ),
    )

    resolution = orbit.resolve_orbits(norad_ids, OBS_EPOCH_JD)
    out = capsys.readouterr().out

    assert resolution.refresh_errors == errors
    assert resolution.missing == []
    assert orbit.require_complete_coverage(resolution) is resolution

    assert "warning: a SatChecker request failed for" in out
    assert f"{from_cache} — {errors[from_cache]} (from {LABEL_CACHE})" in out
    assert f"{rescued} — {errors[rescued]} (from {LABEL_OMM})" in out
    if detail:
        for norad_id in norad_ids:
            assert f"{norad_id} — {errors[norad_id]}" in out
        assert "more (set TABSIM_TLE_LOG_DETAIL=1" not in out
    else:
        assert "and 1 more (set TABSIM_TLE_LOG_DETAIL=1 for the full list)" in out


# ---------------------------------------------------------------------------
# Explicit files
# ---------------------------------------------------------------------------

def test_extra_reader_delegates_and_preserves_contextual_errors(monkeypatch, tmp_path):
    """An explicitly named directory is read strictly, and says which file failed.

    "Cannot be read" must never be indistinguishable from "has no record for
    this satellite": the second falls through to the cache and the service, so
    the run is built from exactly the records the user said not to use.
    """
    spy = spy_on(monkeypatch, "read_extra_orbit_dir")

    good = tmp_path / "good"
    write_orbit_json(good / "iss.json", [tle_record_at(ISS_NORAD_ID, OBS_EPOCH_JD)])
    frame = orbit.read_extra_orbit_dir(good)

    assert len(spy.calls) == 1
    assert Path(str(spy.argument(0, "directory"))) == good
    assert [int(value) for value in frame["NORAD_CAT_ID"]] == [ISS_NORAD_ID]

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "broken.json").write_text("{not json")

    not_a_table = tmp_path / "not_a_table"
    write_orbit_json(not_a_table / "notes.json", [{"NOTE": "nothing orbital here"}])

    bad_identity = tmp_path / "bad_identity"
    write_orbit_json(
        bad_identity / "two.json",
        [
            tle_record_at(ISS_NORAD_ID, OBS_EPOCH_JD),
            dict(
                tle_record_at(GPS_NORAD_ID, GPS_EPOCH_JD),
                NORAD_CAT_ID="not-a-satellite",
            ),
        ],
    )

    for directory, offending in (
        (malformed, "broken.json"),
        (not_a_table, "notes.json"),
        (bad_identity, "two.json"),
    ):
        with pytest.raises(orbit.OrbitError) as raised:
            orbit.read_extra_orbit_dir(directory)
        assert offending in str(raised.value)

    # A malformed identity on a row nobody asked for still stops the run: a bad
    # identity that survives to a wanted-ID filter simply vanishes from it, and
    # the service then answers for the satellite the file was meant to supply.
    monkeypatch.setattr(client, "fetch_nearest_tle", forbidden("the TLE endpoint"))
    monkeypatch.setattr(client, "fetch_nearest_omm", forbidden("the OMM endpoint"))
    with pytest.raises(orbit.OrbitError) as raised:
        orbit.resolve_orbits(
            [ISS_NORAD_ID], OBS_EPOCH_JD, extra_orbit_dir=str(bad_identity)
        )
    assert "two.json" in str(raised.value)

    # A failure the client raised keeps everything it knew about it.
    offending_path = tmp_path / "raised" / "used_orbits.json"
    supplied = OrbitInputError(
        "row 3 is not filed against a satellite: NORAD_CAT_ID is 25544.5",
        path=offending_path,
        row=3,
        norad_id=ISS_NORAD_ID,
    )

    def raise_supplied(*args, **kwargs):
        raise supplied

    monkeypatch.setattr(satchecker_client, "read_extra_orbit_dir", raise_supplied)
    with pytest.raises(orbit.OrbitError) as raised:
        orbit.read_extra_orbit_dir(tmp_path / "anywhere")
    assert raised.value is not supplied
    assert raised.value.__cause__ is supplied
    message = str(raised.value)
    assert str(offending_path) in message
    assert "row 3" in message
    assert str(ISS_NORAD_ID) in message
    assert "is not filed against a satellite" in message


# ---------------------------------------------------------------------------
# Writing what a run used
# ---------------------------------------------------------------------------

def writer_cases():
    tle = tle_record_at(ISS_NORAD_ID, OBS_EPOCH_JD, DATA_SOURCE="spacetrack")
    omm = awkward_omm(GPS_NORAD_ID, GPS_EPOCH_JD, DATA_SOURCE="spacetrack")
    return {
        "empty": ([], [], None),
        "tle": ([ISS_NORAD_ID], [tle], None),
        "omm": ([GPS_NORAD_ID], [omm], None),
        "mixed": ([GPS_NORAD_ID, ISS_NORAD_ID], [omm, tle], None),
        "misaligned": ([GPS_NORAD_ID, ISS_NORAD_ID], [tle], ValueError),
        "mismatched_identity": ([GPS_NORAD_ID], [tle], ValueError),
    }


WRITER_CASES = writer_cases()


@pytest.mark.parametrize("case", list(WRITER_CASES))
def test_single_file_writer_delegates(case, monkeypatch, tmp_path):
    """One writer, in the module that also reads the format back.

    The alignment and identity checks are not incidental validation: ``zip``
    would truncate to the shorter sequence and write a file that reads back
    cleanly while describing different satellites than the run propagated. They
    stay ``ValueError``, because a misaligned call is a bug in the caller and
    not a failure to obtain an orbit.
    """
    norad_ids, records, error = WRITER_CASES[case]
    spy = spy_on(monkeypatch, "save_orbits_for_reuse")
    path = tmp_path / "used_orbits.json"

    if error is None:
        assert orbit.save_orbits_for_reuse(path, norad_ids, records) == str(path)
        assert path.exists()
    else:
        with pytest.raises(error):
            orbit.save_orbits_for_reuse(path, norad_ids, records)

    assert len(spy.calls) == 1
    assert str(spy.argument(0, "path")) == str(path)
    assert list(spy.argument(1, "norad_ids")) == norad_ids
    assert list(spy.argument(2, "records")) == records

    if error is not None:
        return
    written = json.loads(path.read_text())
    assert "EPOCH_JD" not in written and "SEMIMAJOR_AXIS" not in written
    if not records:
        assert written == {}
        return
    back = read_legacy_tle_records(tmp_path)
    assert len(back) == len(records)
    if case in ("omm", "mixed"):
        row = back[back["NORAD_CAT_ID"].astype("int64") == GPS_NORAD_ID].iloc[0]
        # The same double, not a near one: at its maximum precision
        # DataFrame.to_json writes 0.0066635 as 0.006663499999999999.
        assert row["ECCENTRICITY"] == 0.0066635
        assert row["BSTAR"] == 3.2e-05
        # Provenance is read off the file itself: read_legacy_tle_records keeps
        # only identity and elements, by its own contract.
        position = str(norad_ids.index(GPS_NORAD_ID))
        assert written["DATA_SOURCE"][position] == "spacetrack"
        assert written["ECCENTRICITY"][position] == 0.0066635
        assert written["BSTAR"][position] == 3.2e-05


# ---------------------------------------------------------------------------
# Frozen replay
# ---------------------------------------------------------------------------

def test_replay_loader_delegates_and_keeps_tabsim_messages(monkeypatch, capsys):
    """A replay reads two files and has no second source, by design."""
    forbid_replay_fallbacks(monkeypatch)
    spy = spy_on(monkeypatch, "load_replay_orbits")

    directory, expected = compat_fixture("mixed_tle_first")
    norad_ids, records = orbit.load_replay_orbits(directory)

    assert len(spy.calls) == 1
    assert Path(str(spy.argument(0, "directory"))) == Path(directory)
    # Stated, never defaulted: the client requires it and there is no safe
    # default for whether unverifiable lines may be replayed.
    assert spy.call[1]["allow_missing_checksum"] is False
    assert norad_ids == expected["norad_ids"]  # saved order, not sorted
    assert [comparable_record(record) for record in records] == expected["records"]

    unverified_dir, unverified = compat_fixture("unverified")
    _, permissive = orbit.load_replay_orbits(
        unverified_dir, allow_missing_checksum=True
    )
    out = capsys.readouterr().out
    assert "Unverified TLE: missing checksum" in out
    assert permissive[0][CHECKSUM_STATUS_FIELD] == STATUS_UNVERIFIED

    # The same records under this run's policy are a stop, with the opt-in named.
    with pytest.raises(orbit.OrbitError) as raised:
        orbit.load_replay_orbits(unverified_dir)
    message = str(raised.value)
    assert str(unverified_dir / "used_orbits.json") in message
    assert "rfi_sources.tle_satellite.allow_missing_checksum: true" in message
    assert "--allow-missing-checksum" in message

    missing = Path(str(directory)) / "not-a-replay"
    with pytest.raises(orbit.OrbitError) as raised:
        orbit.load_replay_orbits(missing)
    assert str(missing) in str(raised.value)


def write_replay_pair(directory, norad_ids, records) -> Path:
    """The two replay files, written exactly as given.

    Deliberately not through the pair writer: these cases are about what the
    *loader* says when a saved selection does not hold together, and the writer
    exists to refuse producing some of them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    write_orbit_json(directory / "used_orbits.json", records)
    (directory / "norad_ids.yaml").write_text(
        "".join(f"{int(norad_id)}\n" for norad_id in norad_ids)
    )
    return directory


def replay_refusal_cases():
    """``case -> (saved IDs, saved records, is the checksum opt-in the remedy?)``.

    Every case names a satellite, which is exactly why the satellite cannot be
    the evidence: a duplicated ID line and a listed satellite with no record are
    failures of the saved *selection*, and no checksum policy repairs either.
    """
    unverified = canonical(tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD))
    unverified[CHECKSUM_STATUS_FIELD] = STATUS_UNVERIFIED
    corrupt = canonical(tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD))
    corrupt["TLE_LINE1"] = corrupt_checksum(corrupt["TLE_LINE1"])
    intact = canonical(tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD))
    return {
        "duplicate_id_line": ([ISS_NORAD_ID, ISS_NORAD_ID], [intact], False),
        "listed_id_with_no_record": ([ISS_NORAD_ID, GPS_NORAD_ID], [intact], False),
        "unverified_record": ([ISS_NORAD_ID], [unverified], True),
        "corrupt_checksum": ([ISS_NORAD_ID], [corrupt], False),
    }


REPLAY_REFUSAL_CASES = replay_refusal_cases()


@pytest.mark.parametrize("case", list(REPLAY_REFUSAL_CASES))
def test_the_checksum_remedy_is_offered_only_where_it_is_one(
    case, monkeypatch, tmp_path
):
    """The opt-in is suggested for the refusal it lifts, and for no other.

    tabsim's own sentence is the only thing the client's message cannot supply,
    and it is worth nothing unless it is true: telling a user to enable
    ``--allow-missing-checksum`` for a duplicated ID line sends them to a
    setting that cannot change the outcome, and reads as though the run were
    refusing something it is willing to accept.
    """
    norad_ids, records, remediable = REPLAY_REFUSAL_CASES[case]
    forbid_replay_fallbacks(monkeypatch)
    directory = write_replay_pair(tmp_path / "input_data", norad_ids, records)

    with pytest.raises(orbit.OrbitError) as raised:
        orbit.load_replay_orbits(directory)
    message = str(raised.value)

    assert str(directory) in message  # the file it stopped on, whichever it is
    remedy = "rfi_sources.tle_satellite.allow_missing_checksum: true"
    assert (remedy in message) is remediable
    assert ("--allow-missing-checksum" in message) is remediable
    # The whole displayed message, the client's part included: no spelling of
    # the opt-in may reach a user it cannot help.
    assert ("allow_missing_checksum" in message) is remediable
    assert ("allow-missing-checksum" in message) is remediable

    if remediable:
        # ...and it really is the remedy: the same files load with it set.
        loaded_ids, loaded = orbit.load_replay_orbits(
            directory, allow_missing_checksum=True
        )
        assert loaded_ids == norad_ids
        assert loaded[0][CHECKSUM_STATUS_FIELD] == STATUS_UNVERIFIED
    else:
        # A permissive run fails the same way, so the remedy would have been a
        # suggestion to re-run and get the identical error.
        with pytest.raises(orbit.OrbitError):
            orbit.load_replay_orbits(directory, allow_missing_checksum=True)


class SavedObservation:
    """The little of ``Observation`` that ``tabsim.config.save_inputs`` reads."""

    def __init__(self, norad_ids, orbit_records):
        import dask.array as da

        self.norad_ids = (
            [da.from_array(np.asarray(norad_ids, dtype=np.int64))]
            if len(norad_ids)
            else []
        )
        self.orbit_records = list(orbit_records)


def save_config():
    """A ``sim_config`` with no input files to copy, so only the pair is written."""
    return {
        "telescope": {"enu_path": None, "itrf_path": None},
        "ast_sources": {},
        "rfi_sources": {
            "tle_satellite": {
                "norad_ids_path": None,
                "norad_spec_model": None,
                "replay_orbit_dir": None,
            },
            "satellite": {"circ_path": None, "spec_model": None},
            "stationary": {"geo_path": None, "spec_model": None},
        },
    }


def save_cases():
    tle = canonical(tle_record_at(ISS_NORAD_ID, OBS_EPOCH_JD, DATA_SOURCE="spacetrack"))
    omm = canonical(awkward_omm(GPS_NORAD_ID, GPS_EPOCH_JD, DATA_SOURCE="spacetrack"))
    return {
        "records": ([GPS_NORAD_ID, ISS_NORAD_ID], [omm, tle], None),
        "empty": ([], [], None),
        "misaligned": ([GPS_NORAD_ID, ISS_NORAD_ID], [omm], ValueError),
        "duplicate": ([ISS_NORAD_ID, ISS_NORAD_ID], [tle, tle], ValueError),
    }


SAVE_CASES = save_cases()


@pytest.mark.parametrize("case", list(SAVE_CASES))
def test_config_saves_one_validated_replay_pair(case, monkeypatch, tmp_path, capsys):
    """The IDs and the records a run saves are one decision, written together.

    Writing the ID file separately is what lets the two disagree — and, when the
    records turn out not to be writable, leaves a directory holding an ID list
    for records that are not there. The pair writer validates and serialises
    both before either destination is opened.
    """
    norad_ids, records, error = SAVE_CASES[case]
    pair = spy_on(monkeypatch, "save_replay_orbits")
    single = spy_on(monkeypatch, "save_orbits_for_reuse")

    save_path = tmp_path / "input_data"
    save_path.mkdir()
    ids_path = save_path / "norad_ids.yaml"
    records_path = save_path / "used_orbits.json"
    sentinel = "# written by an earlier run\n"
    ids_path.write_text(sentinel)
    records_path.write_text(sentinel)

    observation = SavedObservation(norad_ids, records)
    if error is None:
        config_module.save_inputs(observation, save_config(), str(save_path))
    else:
        with pytest.raises(error):
            config_module.save_inputs(observation, save_config(), str(save_path))

    assert len(pair.calls) == 1
    assert single.calls == []
    assert list(pair.argument(1, "norad_ids")) == norad_ids
    assert list(pair.argument(2, "records")) == records

    if error is not None:
        # Nothing that failed validation reached either destination.
        assert ids_path.read_text() == sentinel
        assert records_path.read_text() == sentinel
        return

    assert ids_path.read_text() == "".join(f"{nid}\n" for nid in norad_ids)
    assert f"Orbit records used written to : {records_path}" in capsys.readouterr().out
    replayed_ids, replayed = orbit.load_replay_orbits(save_path)
    assert replayed_ids == norad_ids
    for original, loaded in zip(records, replayed):
        expected = comparable_record(original)
        read_back = comparable_record(loaded)
        for key, value in expected.items():
            assert read_back[key] == value, key
        # One column-oriented table holding both kinds gives each row the other
        # kind's columns as nulls, which is the format, not an invention: an OMM
        # row reads back with a null TLE_LINE1. Nothing else may appear.
        assert {key for key, value in read_back.items() if value is not None} <= set(
            expected
        )


@pytest.mark.parametrize(
    "case", ["empty", "tle_verified", "omm", "mixed_tle_first", "mixed_omm_first",
             "unverified"]
)
def test_pr44_replay_loads_through_client_adapter(case, monkeypatch):
    """Directories written by #44 replay unchanged, through the client's loader.

    The fixtures are #44's own output, frozen before any of this landed; see
    ``tests/compat/fixtures/PROVENANCE.txt``. What has to survive is the saved
    selection, the retained doubles, the checksum provenance and the
    trajectories — not the bytes.
    """
    forbid_replay_fallbacks(monkeypatch)
    spy = spy_on(monkeypatch, "load_replay_orbits")
    directory, expected = compat_fixture(case)
    policy = expected["allow_missing_checksum"]

    norad_ids, records = orbit.load_replay_orbits(
        directory, allow_missing_checksum=policy
    )

    assert len(spy.calls) == 1
    assert spy.call[1]["allow_missing_checksum"] is policy
    assert norad_ids == expected["norad_ids"]
    assert [comparable_record(record) for record in records] == expected["records"]

    if policy:
        # A permissive run cannot be laundered into a strict one by saving it.
        with pytest.raises(orbit.OrbitError):
            orbit.load_replay_orbits(directory)

    if not records:
        return
    # The elements are the trajectory: rebuilt from the frozen JSON rather than
    # from the loader, so the comparison is against the fixture and not itself.
    rebuilt = [
        {key: value for key, value in record.items() if value is not None}
        for record in expected["records"]
    ]
    times_jd = OBS_EPOCH_JD + np.linspace(0.0, 0.2, 5)
    np.testing.assert_array_equal(
        get_satellite_positions(records, times_jd),
        get_satellite_positions(rebuilt, times_jd),
    )


def adopted_cases():
    tle = canonical(tle_record_at(ISS_NORAD_ID, OBS_EPOCH_JD, DATA_SOURCE="spacetrack"))
    omm = canonical(awkward_omm(GPS_NORAD_ID, GPS_EPOCH_JD, DATA_SOURCE="spacetrack"))
    unverified = tle_record_at(ISS_NORAD_ID, OBS_EPOCH_JD, DATA_SOURCE="spacetrack")
    unverified["TLE_LINE1"] = without_checksum(unverified["TLE_LINE1"])
    unverified["TLE_LINE2"] = without_checksum(unverified["TLE_LINE2"])
    return {
        "empty": ([], [], False),
        "tle": ([ISS_NORAD_ID], [tle], False),
        "omm": ([GPS_NORAD_ID], [omm], False),
        "mixed": ([GPS_NORAD_ID, ISS_NORAD_ID], [omm, tle], False),
        "unverified": ([ISS_NORAD_ID], [canonical(unverified)], True),
    }


ADOPTED_CASES = adopted_cases()


@pytest.mark.parametrize("case", list(ADOPTED_CASES))
def test_adopted_replay_is_readable_by_pr44_loader(case, monkeypatch, tmp_path):
    """And the other direction: #44 can still read what this writes.

    The oracle is #44's real loader, frozen under ``tests/compat/``, because the
    adopted loader cannot answer this question about itself.
    """
    norad_ids, records, unverified = ADOPTED_CASES[case]
    spy = spy_on(monkeypatch, "save_replay_orbits")

    ids_path, records_path = orbit.save_replay_orbits(tmp_path, norad_ids, records)

    assert len(spy.calls) == 1
    assert Path(ids_path).name == pr44_loader.REPLAY_IDS_FILE
    assert Path(records_path).name == pr44_loader.REPLAY_RECORDS_FILE

    if unverified:
        # #44 refuses unverifiable lines unless the replay opts in, exactly as
        # the run that accepted them had to.
        with pytest.raises(pr44_loader.OrbitError):
            pr44_loader.load_replay_orbits(tmp_path)

    back_ids, back_records = pr44_loader.load_replay_orbits(
        tmp_path, allow_missing_checksum=unverified
    )
    assert back_ids == norad_ids  # saved order, not sorted

    for original, loaded in zip(records, back_records):
        expected = comparable_record(original)
        read_back = comparable_record(loaded)
        for key, value in expected.items():
            assert read_back[key] == value, key
        # Nothing invented on the way through: a mixed table's null cells are
        # the only extra keys a row may come back with.
        assert {
            key for key, value in read_back.items() if value is not None
        } <= set(expected)
    if records:
        # Every TLE that went in comes back carrying its checksum provenance, and
        # only a TLE does: the format makes no checksum claim about an OMM, so an
        # OMM-only selection legitimately has no such row.
        status = STATUS_UNVERIFIED if unverified else STATUS_VERIFIED
        tle_rows = [
            record for record in back_records if record.get("RECORD_KIND") == "tle"
        ]
        assert len(tle_rows) == sum(
            1 for record in records if record.get("RECORD_KIND") == "tle"
        )
        assert all(record[CHECKSUM_STATUS_FIELD] == status for record in tle_rows)


# ---------------------------------------------------------------------------
# What the resolver leaves in the shared cache
# ---------------------------------------------------------------------------

def test_resolver_cache_writes_canonical_verified_records(monkeypatch, isolated_cache):
    """The cache keeps the copy that was judged, not the wire row that arrived.

    The shared cache is read by every application on this package, at whatever
    version each is on, and a row with no stated kind or checksum provenance is
    one every reader has to re-infer. Writing the validated copy makes one
    canonicalisation, applied where the record is judged.
    """
    served = tle_record_at(ISS_NORAD_ID, OBS_EPOCH_JD, DATA_SOURCE="spacetrack")
    served.pop("RECORD_KIND")  # the endpoint does not send one
    stub_endpoints(monkeypatch, tle={ISS_NORAD_ID: served})

    resolution = orbit.resolve_orbits([ISS_NORAD_ID], OBS_EPOCH_JD)
    assert resolution.complete

    stored = TextOrbitCache(orbit.orbit_cache_dir()).get(ISS_NORAD_ID)
    assert len(stored) == 1
    assert "RECORD_KIND" in stored.columns
    assert CHECKSUM_STATUS_FIELD in stored.columns
    row = stored.iloc[0]
    assert row["RECORD_KIND"] == "tle"
    assert row[CHECKSUM_STATUS_FIELD] == STATUS_VERIFIED
    assert row["TLE_LINE1"] == served["TLE_LINE1"]


# ---------------------------------------------------------------------------
# What is installed
# ---------------------------------------------------------------------------

def test_satchecker_dependency_pins_resolver_head():
    """The checkout's own metadata, not the installed package's neighbour.

    Under a non-editable install — which is how CI runs — there is no
    ``pyproject.toml`` beside the module, and the pin is the only thing that
    says which client the suite is describing.
    """
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()

    assert PINNED_REQUIREMENT in pyproject
    for superseded in SUPERSEDED_CLIENT_SHAS:
        assert superseded not in pyproject

    workflow = (root / ".github" / "workflows" / "test.yml").read_text()
    installs = [line for line in workflow.splitlines() if "pip install" in line]
    assert any(".[test]" in line for line in installs), installs
    # Nothing may install the client another way: an editable sibling checkout
    # or an older release would hide a public API the pin is there to require.
    assert not any(
        "satchecker" in line or "-e " in line or "--editable" in line
        for line in installs
    ), installs
    # And CI checks what it *installed*, not what the checkout asked for: every
    # revision of this branch of the client exposes the same names and reports
    # the same version, so only the recorded commit distinguishes them.
    assert "direct_url.json" in workflow
    assert "vcs_info" in workflow
    assert "editable" in workflow
    # One source of truth for the revision. Repeating the SHA in the workflow is
    # a second place for it to be right, which is a place for it to be wrong.
    assert PINNED_CLIENT_SHA not in workflow
