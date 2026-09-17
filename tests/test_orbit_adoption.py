"""tabsim.orbit as an adapter over satchecker-client's resolver and replay API.

``tests/test_orbit.py`` says what tabsim's orbit policy *is*; this module says
who executes it — that the selection, acquisition and serialisation tabsim used
to implement itself now come from the client's public API, and that what comes
back through that seam still has tabsim's shape, labels and wording.

**The seam.** Every adopted client function is looked up on the package module at
call time, never bound by a module-scope ``from ... import``, which would make it
unpatchable here and would freeze the archive choice at import.
``orbit_helpers.CLIENT_SEAM_NAMES`` is the list and its ``spy_on`` the only way
these tests install anything on it.

**Public API only.** Nothing patches the client's resolver internals. Delegation
spies wrap the *real* function wherever the behaviour is also asserted, so a test
proves both that the call happened and that the answer is right; where a test
needs a result tabsim could not produce by itself — an outage that blocked a
fallback, a refresh that failed — it builds the client's own result dataclasses
and hands them back through the same seam.
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
from satchecker_client.cache import TextOrbitCache
from satchecker_client.records import record_elements

from tabsim import config as config_module
from tabsim import orbit
from tabsim.tle import get_satellite_positions

from orbit_helpers import (
    CHECKSUM_STATUS_FIELD,
    GPS_EPOCH_JD,
    GPS_NORAD_ID,
    ISS_EPOCH_JD,
    ISS_NORAD_ID,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    Spy,
    comparable_record,
    compat_fixture,
    forbid_orbit_acquisition,
    forbidden,
    omm_record_at,
    reject_json_constant,
    spy_on,
    stub_endpoints,
    tle_record_at,
    write_orbit_json,
    write_replay_pair,
)


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

#: tabsim's display labels — the configuration key the user set, and the archive
#: that answered, because the two behave differently near the handover.
LABEL_EXTRA = "extra_orbit_dir"
LABEL_CACHE = "managed per-satellite cache"
LABEL_TLE = "SatChecker (nearest-TLE)"
LABEL_OMM = "SatChecker (nearest-OMM)"

#: The observation every constructed result is measured against.
OBS_EPOCH_JD = ISS_EPOCH_JD

#: The client dependency this adoption is written against.
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


def deliver(monkeypatch, result) -> Spy:
    """Make the public client resolver answer with *result*, whatever is asked.

    For the cases a stubbed endpoint cannot produce; the result is built from the
    client's own dataclasses, so only the adaptation is under test.
    """
    return spy_on(monkeypatch, "resolve_orbits", lambda *args, **kwargs: result)


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

    *error* is the exception that refused *this* candidate; ``None`` for an age
    rejection, where the record was read, measured and found too far away.
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

    *requested* is what the **client** was asked — the sorted normalised list;
    restoring the run's own order is tabsim's job.
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


def awkward_omm(norad_id, epoch_jd, **extra) -> dict:
    """An OMM carrying the two values the replay format exists to protect."""
    record = omm_record_at(norad_id, epoch_jd, **extra)
    record["ECCENTRICITY"] = 0.0066635
    record["BSTAR"] = 3.2e-05
    return record


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

    The client resolver defaults nothing, so "which policy is in force" is
    entirely a question about this call; the records assert it did the work.
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

    Whether an empty request reaches the client is an implementation choice; that
    it opens no directory, builds no cache and sends no request is not.
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


def test_an_unusable_nearest_cached_record_does_not_hide_a_usable_one(
    monkeypatch, isolated_cache
):
    """Of the records held for one satellite, the nearest *usable* one is chosen.

    A deliberate difference from #44, which refused the nearest cached record for
    its provenance and then asked SatChecker: a row this run's checksum policy
    cannot use is not a candidate, so it neither displaces a usable record inside
    every ceiling the user set nor decides the outcome of an offline run.
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


def test_resolution_frame_uses_client_frame(monkeypatch):
    """The simulator's element frame is derived once, in the client, per read.

    Deriving them again in tabsim is how a record's lines and the element columns
    beside them start disagreeing.
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

    One conversion boundary, one direction, and the client's own object comes out
    unchanged — it is where the coverage classifier reads its evidence from.
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
    # An explicit file is the user's own data: no provider, and not remote, which
    # the client's own source comparison would call it.
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
    # The exception the client attached to *this* rejection, never parsed from
    # its log prose nor taken from another candidate's event.
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

    Only the first rejection per satellite survives, so pairing it with the last
    ``candidate_rejected`` event reports the other candidate's defect beside a
    source that supplied nothing of the sort.
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

    The cheapest way to adopt the client's result types is to alias them, and its
    field order puts ``endpoint`` between ``source`` and ``provider``, moving
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

    The service label names the archive that answered: "SatChecker" alone leaves
    a log unable to say which of them a record came from.
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


@pytest.mark.parametrize(
    "with_rejection", [False, True], ids=["no-rejection", "over-age-rejection"]
)
def test_outage_blocked_not_sent_is_fatal_for_named_coverage(
    with_rejection, monkeypatch
):
    """An answer that was never asked for is not an answer.

    An outage that stopped acquisition leaves the second archive's reply
    ``not_sent``, and the client files such an ID under neither ``service_errors``
    nor ``unavailable``, so "no error recorded" cannot mean "nothing there".
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

    One worker, so the first satellite's archive answers empty, the second's
    raises, and the batch stops before the fallback goes out — leaving the first
    with one reply, silence from the other, and no failure of its own.
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

    A cached record fresh enough to suppress the request, and a satellite resolved
    from an explicit file, both have ``not_sent`` endpoints in ordinary runs.
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

    One satellite, asked for by name, with nothing accepted. Only two kinds of
    evidence are an answer from the catalogue: both archives replied and had
    nothing, or what they had was too old and the acquisition finished.
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


@pytest.mark.parametrize("offline", [False, True], ids=["online", "offline"])
def test_the_refresh_summary_reports_only_requests_that_were_made(
    offline, monkeypatch, isolated_cache, capsys
):
    """"SatChecker did not improve N" is a result, so it needs a request.

    Offline the same record is retained because nothing was asked; the
    skipped-refresh count says that, and an acquisition result beside it would
    report on requests that were never sent.
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

    Events arrive per set of satellites, so a heading, a cache-hit count or a
    fallback explanation can each be emitted more than once for one thing. The
    counts are the assertion, as is the absence of every went-wrong line.
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


@pytest.mark.parametrize(
    "retained", [False, True], ids=["omm-rescue", "cache-and-omm-rescue"]
)
def test_a_refresh_failure_reaches_the_summary_through_the_real_callbacks(
    retained, monkeypatch, isolated_cache, capsys
):
    """The failed-refresh warning, driven by the client's own event sequence.

    What has to agree is the resolver's ``on_event`` callbacks, its error
    bookkeeping and its final result: which satellites failed a refresh, and what
    each is continuing from — never the cache for an ID the second archive
    rescued, which is a record the run never held.
    """
    rescued, epoch_jd = GPS_NORAD_ID, ISS_EPOCH_JD
    expected = {rescued: LABEL_OMM}
    if retained:
        expected[ISS_NORAD_ID] = LABEL_CACHE
        TextOrbitCache(isolated_cache).store(
            ISS_NORAD_ID, pd.DataFrame([tle_record_at(ISS_NORAD_ID, epoch_jd + 2.0)])
        )
    outage = SatCheckerResponseError("nearest-TLE answered 503")
    stub_endpoints(
        monkeypatch,
        tle_default=outage,
        omm={rescued: omm_record_at(rescued, epoch_jd - 0.1)},
        omm_default=outage,
    )

    resolution = orbit.resolve_orbits(sorted(expected), epoch_jd)
    out = capsys.readouterr().out

    assert resolution.complete
    assert orbit.require_complete_coverage(resolution) is resolution
    assert sorted(resolution.refresh_errors) == sorted(expected)
    assert resolution.service_errors == {}

    assert f"warning: a SatChecker request failed for {len(expected)} ID(s)" in out
    for norad_id, label in expected.items():
        assert resolution.resolved[norad_id].source == label
        assert (
            f"{norad_id} — {resolution.refresh_errors[norad_id]} (from {label})" in out
        )
    if not retained:
        # Nothing was held, so nothing may claim a cached record was kept.
        assert "cached record" not in out


def test_extra_reader_delegates_and_preserves_contextual_errors(monkeypatch, tmp_path):
    """An explicitly named directory is read through the client, and a failure the
    client raises keeps the file, row and satellite it named.

    The malformed files themselves are
    ``test_orbit.py::TestSourcePrecedence::test_an_unusable_extra_orbit_file_stops_the_run_naming_it``;
    what is here is the delegation and the error translation.
    """
    spy = spy_on(monkeypatch, "read_extra_orbit_dir")

    good = tmp_path / "good"
    write_orbit_json(good / "iss.json", [tle_record_at(ISS_NORAD_ID, OBS_EPOCH_JD)])
    frame = orbit.read_extra_orbit_dir(good)

    assert len(spy.calls) == 1
    assert Path(str(spy.argument(0, "directory"))) == good
    assert [int(value) for value in frame["NORAD_CAT_ID"]] == [ISS_NORAD_ID]

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


def replay_refusal_cases():
    """``case -> (saved IDs, saved records, is the checksum opt-in the remedy?)``.

    Every case names a satellite, which is why the satellite cannot be the
    evidence: no checksum policy repairs a broken saved *selection*.
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

    Offering ``--allow-missing-checksum`` for a duplicated ID line sends a user to
    a setting that cannot change the outcome.
    """
    norad_ids, records, remediable = REPLAY_REFUSAL_CASES[case]
    forbid_orbit_acquisition(monkeypatch)
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

    Writing the ID file separately leaves a directory holding an ID list for
    records that are not there; the pair writer validates and serialises both
    before either destination is opened.
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
        # One table holding both kinds gives each row the other kind's columns as
        # nulls — the format, not an invention. Nothing else may appear.
        assert {key for key, value in read_back.items() if value is not None} <= set(
            expected
        )


#: The three frozen #44 replay directories kept as historical samples: an
#: explicitly empty selection, a mixed table with unsorted IDs, null cells and
#: awkward doubles, and a record whose checksum digits were never verified.
COMPAT_CASES = ["empty", "mixed_tle_first", "unverified"]


@pytest.mark.parametrize("case", COMPAT_CASES)
def test_a_pr44_replay_loads_through_the_client_adapter(case, monkeypatch, capsys):
    """Directories written by #44 replay unchanged, through the client's loader.

    The fixtures are #44's own output, frozen before any of this landed. What has
    to survive is the saved selection, the retained doubles, the checksum
    provenance and the trajectories — not the bytes.
    """
    forbid_orbit_acquisition(monkeypatch)
    spy = spy_on(monkeypatch, "load_replay_orbits")
    directory, expected = compat_fixture(case)
    policy = expected["allow_missing_checksum"]

    norad_ids, records = orbit.load_replay_orbits(
        directory, allow_missing_checksum=policy
    )
    out = capsys.readouterr().out

    assert len(spy.calls) == 1
    assert Path(str(spy.argument(0, "directory"))) == Path(directory)
    # Stated, never defaulted: there is no safe default for whether unverifiable
    # lines may be replayed.
    assert spy.call[1]["allow_missing_checksum"] is policy
    assert norad_ids == expected["norad_ids"]  # saved order, not sorted
    assert [comparable_record(record) for record in records] == expected["records"]

    if policy:
        assert "Unverified TLE: missing checksum" in out
        assert records[0][CHECKSUM_STATUS_FIELD] == STATUS_UNVERIFIED
        # A permissive run cannot be laundered into a strict one by saving it,
        # and the refusal names the file and the setting that would lift it.
        with pytest.raises(orbit.OrbitError) as raised:
            orbit.load_replay_orbits(directory)
        message = str(raised.value)
        assert str(Path(directory) / "used_orbits.json") in message
        assert "rfi_sources.tle_satellite.allow_missing_checksum: true" in message
        assert "--allow-missing-checksum" in message

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


@pytest.mark.parametrize("case", COMPAT_CASES)
def test_what_a_run_saves_still_matches_the_frozen_file_format(
    case, monkeypatch, tmp_path
):
    """And the other direction: the pair writer still produces #44's two files.

    The expectation is the frozen directory itself, read with the standard
    library — never re-derived through the client's serialiser or its loader,
    which would compare the implementation with itself. Executing #44's own
    loader over this output is historical test-host evidence recorded in the PR,
    not something this suite still does.
    """
    directory, expected = compat_fixture(case)
    norad_ids = expected["norad_ids"]
    # #44's own projection of each record, with the null cells one table holding
    # both kinds gives a row dropped again.
    records = [
        {key: value for key, value in record.items() if value is not None}
        for record in expected["records"]
    ]
    spy = spy_on(monkeypatch, "save_replay_orbits")

    ids_path, records_path = orbit.save_replay_orbits(tmp_path, norad_ids, records)

    assert len(spy.calls) == 1
    assert Path(str(spy.argument(0, "directory"))) == tmp_path
    assert list(spy.argument(1, "norad_ids")) == norad_ids
    assert list(spy.argument(2, "records")) == records
    assert Path(ids_path).name == "norad_ids.yaml"
    assert Path(records_path).name == "used_orbits.json"

    # The saved IDs, in saved order, one per line and nothing else.
    assert Path(ids_path).read_text() == (directory / "norad_ids.yaml").read_text()
    # Positional string indices, projected fields, null cells, kinds, checksum
    # provenance and the exact retained doubles, all in one comparison — and
    # parse_constant refuses a file only Python's own JSON parser would accept.
    written = json.loads(
        Path(records_path).read_text(), parse_constant=reject_json_constant
    )
    assert written == json.loads(
        (directory / "used_orbits.json").read_text(),
        parse_constant=reject_json_constant,
    )
    if not records:
        # An explicitly empty table and an empty ID file, not two absent files.
        assert written == {}
        assert Path(ids_path).read_text() == ""


def test_satchecker_dependency_pins_resolver_head():
    """The checkout's own metadata, not the installed package's neighbour.

    Under a non-editable install, which is how CI runs, the pin is the only thing
    that says which client the suite is describing.
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
    # And CI checks what it *installed*: every revision of this client branch
    # exposes the same names and version, so only the commit distinguishes them.
    assert "direct_url.json" in workflow
    assert "vcs_info" in workflow
    assert "editable" in workflow
    # One source of truth for the revision. Repeating the SHA in the workflow is
    # a second place for it to be right, which is a place for it to be wrong.
    assert PINNED_CLIENT_SHA not in workflow
