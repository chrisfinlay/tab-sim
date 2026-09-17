"""Resolution policy, propagation and configuration validation for orbit records.

Everything here runs offline and is about the tabsim side of the seam: source
precedence, checksum and age policy in :mod:`tabsim.orbit`, epoch-aware name
discovery and search caching in :mod:`tabsim.satchecker_names`, the frozen-replay
contract, and the propagation of both record kinds in :mod:`tabsim.tle`. The
client's own machinery is covered by its repository's suite.

The network-touching seams are ``client.fetch_nearest_tle`` /
``fetch_nearest_omm`` and ``client._http_get``; ``tests/conftest.py`` fails any
test that reaches past them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from astropy.time import Time

from tabsim import orbit
from tabsim import tle as tle_module
from tabsim.orbit_config import (
    TLEConfigurationError,
    normalise_norad_ids,
    normalise_orbit_config,
    observation_epoch_jd,
    read_norad_ids_file,
    validate_age_days,
)
from satchecker_client import client
from satchecker_client.cache import TextOrbitCache, read_legacy_tle_records
from satchecker_client.records import KIND_FIELD, KIND_OMM, KIND_TLE
from tabsim.tle import (
    as_record,
    get_satellite_positions,
    record_tle_lines,
)

from orbit_helpers import (
    CHECKSUM_STATUS_FIELD,
    GPS_LINE1,
    GPS_LINE2,
    GPS_NORAD_ID,
    ISS_EPOCH_JD,
    ISS_LINE1,
    ISS_LINE2,
    ISS_NORAD_ID,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    forbid_orbit_acquisition,
    forbid_search,
    forbidden,
    jd,
    omm_record_from_tle,
    record_at,
    reject_json_constant,
    search_frame,
    search_payload,
    search_row,
    serve_raw_search,
    serve_search,
    spy_on,
    stub_failing_service,
    stub_service,
    tle_lines,
    tle_record,
    tle_record_at,
    with_stray_backslash,
    without_checksum,
    write_orbit_json,
    write_replay_dir,
)


UTC = timezone.utc


def utc(*parts) -> datetime:
    """A timezone-aware UTC instant, for the wall-clock freshness fixtures."""
    return datetime(*parts, tzinfo=UTC)


#: The three ways SatChecker can fail to answer, as the client reports them. They
#: are not interchangeable: a rate limit carries a retry delay and a 5xx is
#: per-request, so each reaches the user with a different remedy.
SERVICE_FAILURES = {
    "transport": client.SatCheckerTransportError("connection refused"),
    "rate-limit": client.SatCheckerRateLimitError("slow down", retry_after=30.0),
    "response": client.SatCheckerResponseError("unreadable reply", status=500),
}

#: A MeerKAT-ish observer and target, for the selection helper below. None of the
#: geometry matters to these tests: visibility is always stubbed.
_OBSERVER = (-30.7, 21.44, 1050.0, 27.0, -30.0, 180.0, -90.0)


def select_visible(
    monkeypatch,
    *,
    names=(),
    norad_ids=(),
    epoch_jd=ISS_EPOCH_JD,
    visible=None,
    **policy,
):
    """Run the simulation's satellite selection over a two-sample time grid.

    *visible* is the NORAD IDs the visibility search should report; ``None``
    forbids the search outright, which is how a test asserts that selection
    failed *before* any propagation happened.
    """
    mjd = epoch_jd - 2400000.5
    times = Time(np.array([mjd - 5e-5, mjd + 5e-5]), format="mjd")
    if visible is None:
        monkeypatch.setattr(
            tle_module,
            "check_satellite_visibilibities",
            forbidden("the visibility search"),
        )
    else:
        monkeypatch.setattr(
            tle_module,
            "check_satellite_visibilibities",
            lambda *args, **kwargs: pd.DataFrame({"norad_id": [int(i) for i in visible]}),
        )
    return tle_module.get_visible_satellite_tles(
        times,
        *_OBSERVER,
        names=list(names),
        norad_ids=list(norad_ids),
        **policy,
    )


class TestPropagation:
    def test_tle_and_omm_paths_agree(self):
        """The same elements propagate identically whichever format carries them.

        The OMM record is built from the *same* TLE it is compared against, so a
        degrees-for-radians or rev/day-for-rad/min slip shows up as kilometres.
        """
        # 1 m, not zero: an ISO 8601 EPOCH round-trips only to microseconds,
        # ~15 us here, which is ~0.1 m of along-track motion at 7.7 km/s.
        records = [tle_record(), omm_record_from_tle()]
        times_jd = ISS_EPOCH_JD + np.linspace(0.0, 0.5, 25)

        positions = get_satellite_positions(records, times_jd)

        separation = np.linalg.norm(positions[0] - positions[1], axis=-1)
        assert separation.max() < 1.0
        # ...and the orbit is the one we asked for, not merely self-consistent.
        radii = np.linalg.norm(positions[0], axis=-1)
        assert np.all((6.7e6 < radii) & (radii < 6.9e6))

    def test_omm_elements_transfer_exactly(self):
        """Every element reaches the propagator unchanged; only the epoch rounds."""
        from skyfield.api import load

        from tabsim.tle import earth_satellite

        ts = load.timescale()
        from_tle = earth_satellite(tle_record(), ts).model
        from_omm = earth_satellite(omm_record_from_tle(), ts).model

        for field in ("bstar", "ecco", "argpo", "inclo", "mo", "no_kozai", "nodeo"):
            assert getattr(from_omm, field) == getattr(from_tle, field), field
        assert from_omm.jdsatepoch == from_tle.jdsatepoch
        assert from_omm.jdsatepochF == pytest.approx(from_tle.jdsatepochF, abs=1e-9)

    def test_bare_line_pair_still_accepted(self):
        """A ``(line1, line2)`` pair keeps working wherever a record is taken."""
        assert as_record((ISS_LINE1, ISS_LINE2)) == {
            "TLE_LINE1": ISS_LINE1,
            "TLE_LINE2": ISS_LINE2,
        }
        positions = get_satellite_positions(
            [(ISS_LINE1, ISS_LINE2)], [ISS_EPOCH_JD]
        )
        assert positions.shape == (1, 1, 3)

    def test_record_tle_lines_empty_for_omm(self):
        """The output schema's line columns are left empty rather than invented."""
        assert record_tle_lines(tle_record()) == (ISS_LINE1, ISS_LINE2)
        assert record_tle_lines(omm_record_from_tle()) == ("", "")

    def test_derived_fixture_lines_are_checksum_valid(self):
        """The fixture builder must produce lines the production parser accepts.

        Every historical-epoch test rests on this: a stale checksum would fail
        those tests for a reason unrelated to the behaviour under test.
        """
        from satchecker_client.tle_parse import tle_epoch_jd, validate_tle_pair

        epoch = jd(2001, 3, 9, 4, 30)
        line1, line2 = tle_lines(GPS_NORAD_ID, epoch)
        assert validate_tle_pair(line1, line2) == GPS_NORAD_ID
        assert tle_epoch_jd(line1) == pytest.approx(epoch, abs=1e-7)


class TestSourcePrecedence:
    def test_extra_dir_wins_and_suppresses_the_request(self, tmp_path, monkeypatch):
        calls = stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})
        pd.DataFrame([tle_record()]).to_json(tmp_path / "mine.json")

        resolution = orbit.resolve_orbits(
            [ISS_NORAD_ID], ISS_EPOCH_JD, extra_orbit_dir=str(tmp_path)
        )

        assert resolution.complete
        assert resolution.resolved[ISS_NORAD_ID].source == "extra_orbit_dir"
        assert calls == []

    def test_extra_dir_over_age_falls_through_to_the_service(
        self, tmp_path, monkeypatch
    ):
        """An over-age local record is skipped, not silently used."""
        stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})
        pd.DataFrame([tle_record()]).to_json(tmp_path / "old.json")

        resolution = orbit.resolve_orbits(
            [ISS_NORAD_ID],
            ISS_EPOCH_JD + 2.0,
            extra_orbit_dir=str(tmp_path),
            extra_orbit_max_age_days=0.5,
            remote_max_age_days=None,
        )

        assert resolution.resolved[ISS_NORAD_ID].source.startswith("SatChecker")

    def test_extra_dir_default_age_is_unlimited(self, tmp_path, monkeypatch):
        """Replay must work however far the record is from the observation."""
        stub_service(monkeypatch, {})
        pd.DataFrame([tle_record()]).to_json(tmp_path / "replay.json")

        resolution = orbit.resolve_orbits(
            [ISS_NORAD_ID], ISS_EPOCH_JD + 400.0, extra_orbit_dir=str(tmp_path)
        )

        assert resolution.complete
        assert resolution.resolved[ISS_NORAD_ID].source == "extra_orbit_dir"

    def test_record_for_the_wrong_satellite_is_rejected(self, tmp_path, monkeypatch):
        """A file filed under one ID but carrying another's lines is not used."""
        stub_service(monkeypatch, {})
        mislabelled = tle_record(norad_id=GPS_NORAD_ID)  # ISS lines, GPS ID
        pd.DataFrame([mislabelled]).to_json(tmp_path / "wrong.json")

        resolution = orbit.resolve_orbits(
            [GPS_NORAD_ID], ISS_EPOCH_JD, extra_orbit_dir=str(tmp_path)
        )

        assert not resolution.complete

    @pytest.mark.parametrize(
        "damage,expected",
        [
            ("broken-json", ["used_orbits.json"]),
            ("not-an-orbit-table", ["notes.json"]),
            ("fractional-identity", ["mine.json", "25544.5"]),
            ("identity-on-an-unrequested-row", ["two.json"]),
        ],
    )
    def test_an_unusable_extra_orbit_file_stops_the_run_naming_it(
        self, tmp_path, monkeypatch, damage, expected
    ):
        """An explicit source we cannot read must name its path, not fall through.

        Coercing an identity before validating it dropped the row silently and
        substituted the service record the file existed to replace; the last case
        damages a row nobody asked for, which a wanted-ID filter would swallow.
        """
        if damage == "broken-json":
            (tmp_path / "used_orbits.json").write_text(
                '{"TLE_LINE1": {"0": "1 25544U 98067A   23055'
            )
        elif damage == "not-an-orbit-table":
            write_orbit_json(tmp_path / "notes.json", [{"NOTE": "nothing orbital"}])
        elif damage == "fractional-identity":
            record = tle_record()  # valid ISS lines...
            record["NORAD_CAT_ID"] = 25544.5  # ...under something that is not an ID
            pd.DataFrame([record]).to_json(tmp_path / "mine.json")
        else:
            write_orbit_json(
                tmp_path / "two.json",
                [
                    tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD),
                    dict(
                        tle_record_at(GPS_NORAD_ID, ISS_EPOCH_JD),
                        NORAD_CAT_ID="not-a-satellite",
                    ),
                ],
            )
        # The service must not be asked for what the file was meant to supply.
        monkeypatch.setattr(client, "fetch_nearest_tle", forbidden("the TLE endpoint"))
        monkeypatch.setattr(client, "fetch_nearest_omm", forbidden("the OMM endpoint"))

        with pytest.raises(orbit.OrbitError) as excinfo:
            orbit.resolve_orbits(
                [ISS_NORAD_ID], ISS_EPOCH_JD, extra_orbit_dir=str(tmp_path)
            )

        message = str(excinfo.value)
        for fragment in expected:
            assert fragment in message, fragment

    def test_fresh_cache_avoids_the_request(self, monkeypatch, isolated_cache):
        TextOrbitCache(isolated_cache).store(
            ISS_NORAD_ID, pd.DataFrame([tle_record()])
        )
        calls = stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})

        resolution = orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD + 0.5)

        assert resolution.resolved[ISS_NORAD_ID].source.endswith("cache")
        assert calls == []

    def test_stale_cache_still_asks_but_survives_a_service_failure(
        self, monkeypatch, isolated_cache
    ):
        """An acceptable-but-stale cached record is the offline fallback."""
        TextOrbitCache(isolated_cache).store(
            ISS_NORAD_ID, pd.DataFrame([tle_record()])
        )
        calls = stub_service(monkeypatch, {})  # service has nothing

        resolution = orbit.resolve_orbits(
            [ISS_NORAD_ID],
            ISS_EPOCH_JD + 2.0,
            remote_max_age_days=3.0,
            cache_reuse_max_age_days=1.0,
        )

        assert calls, "a stale cached record should still prompt a request"
        assert resolution.complete
        assert resolution.resolved[ISS_NORAD_ID].source.endswith("cache")

    def test_closer_cached_record_survives_a_worse_refresh(
        self, monkeypatch, isolated_cache
    ):
        """A successful response further from the observation must not displace it.

        The rule is strictly fresher, not most recently seen; otherwise a refresh
        quietly makes the simulation worse and the log reports a successful fetch.
        """
        epoch_jd = ISS_EPOCH_JD + 2.0
        TextOrbitCache(isolated_cache).store(
            ISS_NORAD_ID, pd.DataFrame([tle_record_at(ISS_NORAD_ID, epoch_jd - 0.5)])
        )
        calls = stub_service(
            monkeypatch, {ISS_NORAD_ID: tle_record_at(ISS_NORAD_ID, epoch_jd - 2.0)}
        )

        resolution = orbit.resolve_orbits(
            [ISS_NORAD_ID],
            epoch_jd,
            remote_max_age_days=3.0,
            cache_reuse_max_age_days=0.1,
        )

        assert calls, "a cached record outside the reuse threshold still gets asked"
        entry = resolution.resolved[ISS_NORAD_ID]
        assert entry.source.endswith("cache")
        assert entry.age_days == pytest.approx(0.5, abs=1e-3)

    def test_service_response_is_cached_for_later_runs(
        self, monkeypatch, isolated_cache
    ):
        """What is cached is the copy that was judged, not the wire row that came.

        A row with no stated kind or checksum provenance is one every other reader
        of the shared cache has to re-infer, at whatever version each is on.
        """
        served = tle_record()
        served.pop(KIND_FIELD)  # the endpoint does not send one
        calls = stub_service(monkeypatch, {ISS_NORAD_ID: served})
        assert orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD).complete

        cached = TextOrbitCache(isolated_cache).get(ISS_NORAD_ID)
        assert len(cached) == 1
        row = cached.iloc[0]
        assert row["TLE_LINE1"] == ISS_LINE1
        assert row[KIND_FIELD] == KIND_TLE
        assert row[CHECKSUM_STATUS_FIELD] == STATUS_VERIFIED

        # ...and the next run at the same epoch uses it without asking again.
        calls.clear()
        again = orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD)
        assert again.resolved[ISS_NORAD_ID].source.endswith("cache")
        assert calls == []

    def test_strict_response_is_requested_from_both_endpoints(self, monkeypatch):
        """Every nearest-record request must opt in to strict response parsing.

        Without it an HTTP-200 error envelope — how SatChecker reports its own
        failures — normalises to an empty frame and an outage becomes "this
        satellite has no record".
        """
        stale = tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD - 40.0)
        tle_calls, omm_calls = [], []

        def nearest_tle(norad_id, _epoch_jd, *, strict_response=False):
            tle_calls.append(strict_response)
            return pd.DataFrame([stale])  # over-age: forces the fallback

        def nearest_omm(norad_id, _epoch_jd, *, strict_response=False):
            omm_calls.append(strict_response)
            return pd.DataFrame([omm_record_from_tle()])

        monkeypatch.setattr(client, "fetch_nearest_tle", nearest_tle)
        monkeypatch.setattr(client, "fetch_nearest_omm", nearest_omm)

        resolution = orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD)

        assert resolution.complete
        assert tle_calls and omm_calls, "both endpoints should have been asked"
        assert all(strict is True for strict in tle_calls + omm_calls)

    def test_endpoint_failover_when_the_first_archive_is_too_old(self, monkeypatch):
        """An over-age answer is the signal the record lives in the other archive.

        Neither endpoint reports "I have nothing that near", so this is the only
        way the epoch's archive can be got wrong and recovered from.
        """
        epoch_jd = ISS_EPOCH_JD  # pre-handover: nearest-TLE is tried first
        omm = omm_record_from_tle()
        stale_tle = tle_record(line1=GPS_LINE1, line2=GPS_LINE2, norad_id=GPS_NORAD_ID)

        def nearest_tle(norad_id, _epoch_jd, *, strict_response=False):
            return pd.DataFrame([stale_tle])  # wrong satellite -> rejected

        def nearest_omm(norad_id, _epoch_jd, *, strict_response=False):
            return pd.DataFrame([omm])

        monkeypatch.setattr(client, "fetch_nearest_tle", nearest_tle)
        monkeypatch.setattr(client, "fetch_nearest_omm", nearest_omm)

        resolution = orbit.resolve_orbits([ISS_NORAD_ID], epoch_jd)

        assert resolution.complete
        assert "nearest-OMM" in resolution.resolved[ISS_NORAD_ID].source

    def test_acceptable_primary_answer_suppresses_the_other_archive(self, monkeypatch):
        """The fallback is for an unusable answer, not a second opinion.

        Selection is *not* globally nearest across both archives: the epoch picks
        one, and the other is consulted only when the first has nothing usable.
        """
        omm_calls = []

        def nearest_omm(norad_id, _epoch_jd, *, strict_response=False):
            omm_calls.append(int(norad_id))
            return pd.DataFrame([omm_record_from_tle()])

        stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})
        monkeypatch.setattr(client, "fetch_nearest_omm", nearest_omm)

        resolution = orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD)

        assert resolution.complete
        assert "nearest-TLE" in resolution.resolved[ISS_NORAD_ID].source
        assert omm_calls == []

    def test_outage_stops_the_batch_rather_than_trying_the_other_archive(
        self, monkeypatch
    ):
        """A service that cannot serve us is not asked a different question."""
        omm_calls = []

        def unreachable(norad_id, _epoch_jd, *, strict_response=False):
            raise client.SatCheckerTransportError("connection refused")

        def nearest_omm(norad_id, _epoch_jd, *, strict_response=False):
            omm_calls.append(norad_id)
            return pd.DataFrame()

        monkeypatch.setattr(client, "fetch_nearest_tle", unreachable)
        monkeypatch.setattr(client, "fetch_nearest_omm", nearest_omm)

        resolution = orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD)

        assert omm_calls == []
        assert ISS_NORAD_ID in resolution.service_errors


class TestCoverage:
    @pytest.mark.parametrize(
        "failure,expected",
        [
            ("absent", ["extra_orbit_dir", "remote_max_age_days"]),
            ("over-age", ["10.000 d from the"]),
            ("outage", ["Re-run when the service"]),
        ],
    )
    def test_the_coverage_error_says_which_of_the_three_failures_it_is(
        self, monkeypatch, failure, expected
    ):
        """Nothing there, too far away and could not ask need different remedies.

        "10 d away" is actionable where "not found" is not, and only the outage is
        a reason to re-run an unchanged configuration.
        """
        epoch_jd, policy = ISS_EPOCH_JD, {}
        if failure == "absent":
            stub_service(monkeypatch, {})
        elif failure == "over-age":
            stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})
            epoch_jd, policy = ISS_EPOCH_JD + 10.0, {"remote_max_age_days": 3.0}
        else:
            stub_failing_service(
                monkeypatch, client.SatCheckerTransportError("connection refused")
            )

        resolution = orbit.resolve_orbits([ISS_NORAD_ID], epoch_jd, **policy)

        with pytest.raises(orbit.OrbitError) as excinfo:
            orbit.require_complete_coverage(resolution)

        message = str(excinfo.value)
        assert str(ISS_NORAD_ID) in message
        for fragment in expected:
            assert fragment in message, fragment

    def test_no_ceiling_accepts_anything(self, monkeypatch):
        stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})

        resolution = orbit.resolve_orbits(
            [ISS_NORAD_ID],
            ISS_EPOCH_JD + 1000.0,
            remote_max_age_days=None,
            cache_reuse_max_age_days=None,
        )

        assert resolution.complete

    # -- unresolved failures are fatal for *both* selection routes -----------

    @pytest.mark.parametrize("route", ["numbered", "named"])
    @pytest.mark.parametrize("failure", list(SERVICE_FAILURES) + ["invalid-record"])
    def test_unresolved_service_failure_is_fatal_for_names_and_numbers(
        self, monkeypatch, route, failure
    ):
        """"We could not find out" is not "there is nothing there".

        A named satellite whose record could not be obtained used to be dropped,
        so an outage produced a complete-looking observation with no RFI in it.
        """
        norad_id = 7001
        if failure == "invalid-record":  # a reply that carries no usable record
            corrupt = tle_record_at(norad_id, ISS_EPOCH_JD)
            corrupt["TLE_LINE2"] = corrupt["TLE_LINE2"][:68] + "9"
            stub_service(monkeypatch, {norad_id: corrupt})
        else:
            stub_failing_service(monkeypatch, SERVICE_FAILURES[failure])

        if route == "named":
            serve_search(
                monkeypatch, {"THING": [search_row(norad_id, "THING ONE")]}
            )
            kwargs = {"names": ["thing"]}
        else:
            kwargs = {"norad_ids": [norad_id]}

        with pytest.raises(orbit.OrbitError) as excinfo:
            select_visible(monkeypatch, visible=None, **kwargs)

        message = str(excinfo.value)
        assert str(norad_id) in message
        assert "SatChecker could not answer" in message

    def test_http_200_error_envelope_is_an_outage_not_an_absent_satellite(
        self, monkeypatch
    ):
        """Through the real endpoint wrappers: an error envelope is not an empty reply.

        Every other test stubs ``fetch_nearest_tle`` and records
        ``strict_response``; this one goes through the real wrapper from the
        transport up, so the opt-in is shown to reach the parser.
        """
        norad_id = 7002

        def fake_get(url, timeout=None):
            if "search-satellites" in url:
                return search_payload([search_row(norad_id, "THING ONE")])
            return json.dumps({"error": "service unavailable"}).encode()

        monkeypatch.setattr(client, "_http_get", fake_get)

        with pytest.raises(orbit.OrbitError) as excinfo:
            select_visible(monkeypatch, names=["thing"], visible=None)

        message = str(excinfo.value)
        assert str(norad_id) in message
        assert "SatChecker could not answer" in message

    def test_legitimate_absence_and_over_age_have_distinct_diagnostics(
        self, monkeypatch, capsys
    ):
        """Nothing there, too far away, and could not ask are three answers.

        Only the third is a reason to re-run unchanged, so collapsing them costs
        the user the only remedy that works.
        """
        absent, over_age = 7001, 7002
        stub_service(
            monkeypatch, {over_age: tle_record_at(over_age, ISS_EPOCH_JD - 10.0)}
        )
        serve_search(
            monkeypatch,
            {
                "THING": [
                    search_row(absent, "THING ONE"),
                    search_row(over_age, "THING TWO"),
                ]
            },
        )

        norad_ids, records = select_visible(
            monkeypatch,
            names=["thing"],
            visible=[],
            remote_max_age_days=3.0,
            cache_reuse_max_age_days=1.0,
        )

        assert list(norad_ids) == [] and records == []
        out = capsys.readouterr().out
        assert "No acceptable record" in out
        assert str(absent) in out and str(over_age) in out
        # The over-age exclusion says how far off and against which ceiling.
        assert "10.0" in out
        assert "remote_max_age_days" in out
        # Neither satellite failed for want of an answer, so nothing may claim so.
        assert "SatChecker could not answer" not in out

        # The same two IDs asked for by number are a coverage failure instead.
        with pytest.raises(orbit.OrbitError) as excinfo:
            select_visible(
                monkeypatch,
                norad_ids=[absent, over_age],
                visible=None,
                remote_max_age_days=3.0,
            )
        assert str(absent) in str(excinfo.value)
        assert str(over_age) in str(excinfo.value)

    def test_outage_can_use_acceptable_cached_incumbent(
        self, monkeypatch, isolated_cache, capsys
    ):
        """A failed refresh must be recorded even when the run can continue.

        The cached record is within the hard ceiling, so the run is legitimate but
        not the one asked for, and the log is the only place that can say so.
        """
        TextOrbitCache(isolated_cache).store(
            ISS_NORAD_ID, pd.DataFrame([tle_record()])
        )
        stub_failing_service(
            monkeypatch, client.SatCheckerTransportError("connection refused")
        )

        resolution = orbit.resolve_orbits(
            [ISS_NORAD_ID],
            ISS_EPOCH_JD + 2.0,
            remote_max_age_days=3.0,
            cache_reuse_max_age_days=1.0,
        )

        assert resolution.complete
        assert resolution.resolved[ISS_NORAD_ID].source.endswith("cache")
        # Recorded as a refresh failure, not as an unanswered request: the ID is
        # resolved, so it must not appear in the fatal-coverage bookkeeping.
        assert ISS_NORAD_ID in resolution.refresh_errors
        assert ISS_NORAD_ID not in resolution.service_errors
        orbit.require_complete_coverage(resolution)  # no hard-ceiling bypass needed
        out = capsys.readouterr().out
        assert str(ISS_NORAD_ID) in out
        assert "cache" in out.lower()

    @pytest.mark.parametrize("detail", [False, True], ids=["truncated", "detailed"])
    def test_the_refresh_failure_summary_honours_the_log_switch(
        self, monkeypatch, isolated_cache, capsys, detail
    ):
        """A failed refresh is not fatal, and ``TABSIM_TLE_LOG_DETAIL`` reaches here.

        The summary sliced to the first twelve whatever the switch said, so the
        one thing that recovers a full per-satellite listing could not recover
        this one; each entry names the source its satellite is continuing from.
        """
        norad_ids = list(range(7500, 7513))  # thirteen: one over the grouping limit
        cache = TextOrbitCache(isolated_cache)
        for nid in norad_ids:
            cache.store(nid, pd.DataFrame([tle_record_at(nid, ISS_EPOCH_JD)]))

        def failing(norad_id, _epoch_jd, *, strict_response=False):
            # Distinct statuses: a wall of one status is an outage, and the batch
            # then stops early with fewer errors than there are satellites.
            raise client.SatCheckerResponseError(
                f"no answer for {int(norad_id)}", status=500 + int(norad_id) % 5
            )

        monkeypatch.setattr(client, "fetch_nearest_tle", failing)
        monkeypatch.setattr(client, "fetch_nearest_omm", failing)
        if detail:
            monkeypatch.setenv("TABSIM_TLE_LOG_DETAIL", "1")
        else:
            monkeypatch.delenv("TABSIM_TLE_LOG_DETAIL", raising=False)

        resolution = orbit.resolve_orbits(
            norad_ids,
            ISS_EPOCH_JD + 2.0,
            remote_max_age_days=3.0,
            cache_reuse_max_age_days=1.0,
        )

        # Not fatal: every ID resolved, from the record it already held.
        assert resolution.complete
        assert sorted(resolution.refresh_errors) == norad_ids
        assert resolution.missing == []
        assert orbit.require_complete_coverage(resolution) is resolution

        out = capsys.readouterr().out
        assert f"warning: a SatChecker request failed for {len(norad_ids)} ID(s)" in out
        # tabsim's own summary entry, not the client's per-request log line.
        entry = "{0} — no answer for {0} (from managed per-satellite cache)".format
        for nid in norad_ids[:12]:
            assert entry(nid) in out
        if detail:
            assert entry(norad_ids[-1]) in out
            assert "more (set TABSIM_TLE_LOG_DETAIL=1" not in out
        else:
            assert entry(norad_ids[-1]) not in out
            assert "and 1 more (set TABSIM_TLE_LOG_DETAIL=1 for the full list)" in out

    # -- offline -------------------------------------------------------------

    def test_offline_orbit_resolution_never_requests_refresh(
        self, monkeypatch, isolated_cache
    ):
        """Offline means no request, but not a relaxed age ceiling."""
        TextOrbitCache(isolated_cache).store(
            ISS_NORAD_ID, pd.DataFrame([tle_record()])
        )
        calls = stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})

        resolution = orbit.resolve_orbits(
            [ISS_NORAD_ID],
            ISS_EPOCH_JD + 2.0,
            remote_max_age_days=3.0,
            cache_reuse_max_age_days=1.0,
            offline=True,
        )

        assert calls == []
        assert resolution.complete
        assert resolution.resolved[ISS_NORAD_ID].source.endswith("cache")

        # ...and the hard ceiling still applies to what the cache holds.
        over_ceiling = orbit.resolve_orbits(
            [ISS_NORAD_ID],
            ISS_EPOCH_JD + 5.0,
            remote_max_age_days=3.0,
            cache_reuse_max_age_days=1.0,
            offline=True,
        )
        assert not over_ceiling.complete
        assert calls == []

    def test_offline_missing_orbit_is_not_claimed_as_remote_absence(
        self, monkeypatch, isolated_cache
    ):
        """A cached search says nothing about why an uncached orbit is missing.

        A run with offline discovery but not offline acquisition must say it ran
        out of local state, not that SatChecker has no record.
        """
        norad_id = 7001
        TextOrbitCache(isolated_cache).store_search(
            "THING",
            search_frame([search_row(norad_id, "THING ONE")]),
            fetched_at=datetime.now(UTC),
        )

        with pytest.raises(orbit.OrbitError) as excinfo:
            select_visible(
                monkeypatch, names=["thing"], visible=None, offline=True
            )

        message = str(excinfo.value)
        assert str(norad_id) in message
        assert "offline" in message.lower()

    @pytest.mark.parametrize("source", ["cache", "extra_orbit_dir"])
    def test_offline_named_age_rejection_is_insufficient_local_state(
        self, monkeypatch, isolated_cache, tmp_path, source
    ):
        """An over-age local record is not evidence that nothing closer exists.

        Nothing was asked: a ten-day-old cached record says only that this machine
        holds one, so excluding the satellite fakes a legitimate quiet sky.
        """
        norad_id = 7003
        cache = TextOrbitCache(isolated_cache)
        cache.store_search(
            "THING",
            search_frame([search_row(norad_id, "THING ONE")]),
            fetched_at=datetime.now(UTC),
        )
        stale = tle_record_at(norad_id, ISS_EPOCH_JD - 10.0)
        policy = {}
        if source == "cache":
            cache.store(norad_id, pd.DataFrame([stale]))
        else:
            pd.DataFrame([stale]).to_json(tmp_path / "old.json")
            policy = {
                "extra_orbit_dir": str(tmp_path),
                "extra_orbit_max_age_days": 3.0,
            }

        with pytest.raises(orbit.OrbitError) as excinfo:
            tle_module.get_tles_by_name(
                ["thing"],
                ISS_EPOCH_JD,
                remote_max_age_days=3.0,
                cache_reuse_max_age_days=1.0,
                offline=True,
                **policy,
            )

        message = str(excinfo.value)
        assert str(norad_id) in message
        assert "offline" in message.lower()
        # ...and the age detail survives into the insufficient-local-state error.
        assert "10.000 d" in message

    # -- historical epochs ---------------------------------------------------

    @pytest.mark.parametrize(
        "epoch_jd",
        [
            jd(2000, 6, 15, 12),
            jd(2010, 3, 1),
            jd(2017, 5, 14, 16, 15),
            jd(2019, 11, 20, 3),
        ],
    )
    def test_historical_epoch_is_sent_unchanged(self, monkeypatch, epoch_jd):
        """Past observations are resolved at their own epoch, not at today's.

        Both halves matter: the request carries the observation epoch, and the
        catalogue filter uses it too, or a decayed satellite is dropped from a
        historical run while one launched since is added.
        """
        current, launched_later = 7101, 7102
        calls = stub_service(
            monkeypatch, {current: tle_record_at(current, epoch_jd)}
        )
        serve_search(
            monkeypatch,
            {
                "HIST": [
                    search_row(
                        current,
                        "HIST ONE",
                        launch_date="1999-01-01",
                        decay_date="2021-01-01",
                    ),
                    search_row(
                        launched_later, "HIST TWO", launch_date="2022-01-01"
                    ),
                ]
            },
        )

        ids = orbit.resolve_names(["hist"], epoch_jd, log=lambda *_: None)
        assert ids == [current]

        resolution = orbit.resolve_orbits(ids, epoch_jd, remote_max_age_days=3.0)
        assert resolution.complete
        assert [nid for nid, _, _ in calls] == [current]
        assert calls[0][1] == pytest.approx(epoch_jd)
        assert resolution.resolved[current].age_days < 1e-3


def _messages(log_lines, capsys):
    """Everything a lookup reported, whichever channel it used."""
    return "\n".join(log_lines) + "\n" + capsys.readouterr().out


class TestNameDiscovery:
    def test_names_use_public_substring_search(self, monkeypatch):
        """Discovery goes through the package's public search, nothing private.

        A private copy of the transport and parser is how a malformed reply
        became "no satellite matches this name".
        """
        calls = serve_search(
            monkeypatch,
            {
                "NAVSTAR": [search_row(24876, "NAVSTAR 43 (USA 132)")],
                "STARLINK": [search_row(44713, "STARLINK-1007")],
            },
        )

        ids = orbit.resolve_names(
            ["navstar", "starlink"], ISS_EPOCH_JD, log=lambda *_: None
        )

        assert ids == [24876, 44713]
        # Upper-cased: the endpoint is a case-sensitive substring match against an
        # upper-case catalogue, so a lower-case name would silently find nothing.
        assert [name for name, _ in calls] == ["NAVSTAR", "STARLINK"]
        for _, url in calls:
            assert "search-satellites" in url
            # Not the exact-name index, which would reduce "navstar" to nothing.
            assert "norad-ids-from-name" not in url

    def test_named_candidates_follow_observation_epoch(self, monkeypatch):
        """Which satellites existed is a question about the observation's date.

        Filtering on "has a decay date at all" answers a different question, and
        wrongly in both directions for any epoch that is not today.
        """
        rows = [
            search_row(1001, "THING A", launch_date="2000-01-01",
                       decay_date="2024-06-01"),
            search_row(1002, "THING B", launch_date="2000-01-01",
                       decay_date="2019-01-01"),
            search_row(1003, "THING C", launch_date="2025-01-01"),
            search_row(1004, "THING D"),
        ]
        calls = serve_search(monkeypatch, {"THING": rows})

        # 2023: A still in orbit, B already re-entered, C not yet launched.
        assert orbit.resolve_names(
            ["thing"], ISS_EPOCH_JD, log=lambda *_: None
        ) == [1001, 1004]

        # The second epoch reuses the cached search, so a different answer can
        # only have come from the epoch.
        forbid_search(monkeypatch)
        assert orbit.resolve_names(
            ["thing"], jd(2025, 6, 1), log=lambda *_: None
        ) == [1003, 1004]
        assert len(calls) == 1

    def test_alias_dates_are_combined_before_id_deduplication(self, monkeypatch):
        """One satellite's alias rows need not agree; together they are the evidence.

        A null or later launch date on one row means that row does not say, so
        de-duplicating first rules out satellites the catalogue never ruled out.
        """
        first_query = [
            search_row(2001, "SAT X", object_id="1998-067A",
                       decay_date="2019-01-01"),
            search_row(2002, "SAT Y", launch_date="2025-01-01"),
        ]
        second_query = [
            search_row(2001, "SAT X [DTC]", object_id="1998-067A",
                       decay_date="2030-01-01"),
            search_row(2002, "SAT Y ALT", launch_date="2001-01-01"),
        ]
        serve_search(
            monkeypatch, {"ALIASA": first_query, "ALIASB": second_query}
        )

        ids = orbit.resolve_names(
            ["aliasa", "aliasb"], ISS_EPOCH_JD, log=lambda *_: None
        )

        # Latest known decay and earliest known launch keep both in play, once.
        assert ids == [2001, 2002]

    @pytest.mark.parametrize(
        "both_resolve", [True, False], ids=["both-resolve", "one-over-age"]
    )
    def test_two_ids_for_one_object_are_kept_and_reported_as_candidates(
        self, monkeypatch, capsys, both_resolve
    ):
        """Two catalogue numbers for one object are two candidates, not one.

        Nothing says which is current, so an identity merge would silently drop a
        satellite; and the warning runs at discovery, so it can only speak of
        candidates — either may still fail age coverage, as one does here.
        """
        rows = [
            search_row(61608, "TWIN SAT", object_id="2024-100A"),
            search_row(72115, "TWIN SAT", object_id="2024-100A"),
        ]
        serve_search(monkeypatch, {"TWIN": rows})
        log_lines = []

        ids = orbit.resolve_names(["twin"], ISS_EPOCH_JD, log=log_lines.append)
        assert ids == [61608, 72115]

        stub_service(
            monkeypatch,
            {
                61608: tle_record_at(61608, ISS_EPOCH_JD),
                72115: tle_record_at(
                    72115, ISS_EPOCH_JD - (0.0 if both_resolve else 10.0)
                ),
            },
        )
        resolution = orbit.resolve_orbits(ids, ISS_EPOCH_JD, remote_max_age_days=3.0)
        orbit.report_named_coverage(resolution, log=log_lines.append)

        assert sorted(resolution.resolved) == (
            [61608, 72115] if both_resolve else [61608]
        )
        messages = _messages(log_lines, capsys)
        assert "OBJECT_ID" in messages
        assert "2024-100A" in messages
        assert "61608" in messages and "72115" in messages
        assert "candidate NORAD catalogue ID" in messages
        assert "may be modelled separately" in messages
        assert "if both survive" in messages
        # ...and never a promise about a selection that has not happened yet.
        assert "both are kept as distinct satellites" not in messages
        assert "simulated once per number that resolves" not in messages
        if not both_resolve:
            assert "No acceptable record for named satellite 72115" in messages

    def test_search_snapshot_is_full_and_beside_orbit_files(
        self, monkeypatch, isolated_cache
    ):
        """The whole catalogue result is cached, not the IDs it boiled down to.

        A snapshot reduced to one epoch's candidates could not answer a second
        observation, and the alias rows are the launch/decay evidence.
        """
        rows = [
            search_row(24876, "NAVSTAR 43 (USA 132)", object_id="1997-035A",
                       launch_date="1997-07-23"),
            search_row(24876, "NAVSTAR 43", object_id="1997-035A"),
            search_row(28129, "NAVSTAR 53 (USA 175)", launch_date="2003-12-21"),
        ]
        serve_search(monkeypatch, {"NAVSTAR": rows})
        ids = orbit.resolve_names(["navstar"], ISS_EPOCH_JD, log=lambda *_: None)
        stub_service(
            monkeypatch, {nid: tle_record_at(nid, ISS_EPOCH_JD) for nid in ids}
        )
        orbit.resolve_orbits(ids, ISS_EPOCH_JD)

        cache = TextOrbitCache(isolated_cache)
        snapshot = cache.get_search("NAVSTAR")
        assert snapshot is not None
        assert len(snapshot.found) == len(rows), "alias rows must all survive"
        assert sorted(snapshot.found["NORAD_CAT_ID"]) == [24876, 24876, 28129]
        launches = [
            value for value in snapshot.found["LAUNCH_DATE"] if not pd.isna(value)
        ]
        assert sorted(launches) == ["1997-07-23", "2003-12-21"]
        # ...beside, not instead of, the per-satellite orbit files.
        assert sorted(p.name for p in isolated_cache.glob("orbit-*.json")) == [
            "orbit-24876.json",
            "orbit-28129.json",
        ]
        assert len(list(isolated_cache.glob("search-*.json"))) == 1

    @pytest.mark.parametrize(
        "fetched_at,now,freshness,requests,expected",
        [
            (utc(2026, 1, 1), utc(2026, 1, 1, 12), 1.0, 0, [3001]),
            # Zero is "always ask", not "reuse anything younger than zero days",
            # which a snapshot fetched at this very instant satisfies.
            (utc(2026, 1, 1), utc(2026, 1, 1), 0, 1, [3002]),
            # null is the opt-out from refreshing at all, at any age.
            (utc(2020, 1, 1), utc(2026, 1, 1), None, 0, [3001]),
        ],
        ids=["fresh", "zero-at-the-same-instant", "null-six-years-old"],
    )
    def test_search_freshness_decides_whether_the_catalogue_is_asked(
        self, monkeypatch, isolated_cache, fetched_at, now, freshness, requests, expected
    ):
        """Wall-clock freshness, and the two settings that are not a duration."""
        from tabsim import satchecker_names

        cache = TextOrbitCache(isolated_cache)
        cache.store_search(
            "THING", search_frame([search_row(3001, "THING A")]), fetched_at=fetched_at
        )
        monkeypatch.setattr(satchecker_names, "_utc_now", lambda: now)
        calls = []
        if requests:
            calls = serve_search(monkeypatch, {"THING": [search_row(3002, "THING B")]})
        else:
            forbid_search(monkeypatch)

        ids = orbit.resolve_names(
            ["thing"],
            ISS_EPOCH_JD,
            search_cache_max_age_days=freshness,
            log=lambda *_: None,
        )

        assert ids == expected
        assert len(calls) == requests
        # A refresh replaces the snapshot; a reuse leaves it exactly as it was.
        assert sorted(cache.get_search("THING").found["NORAD_CAT_ID"]) == expected

    def test_stale_search_refresh_replaces_snapshot(
        self, monkeypatch, isolated_cache
    ):
        """A refreshed search replaces the old one; it is not merged into it.

        Keeping a satellite because an older search saw it would resurrect exactly
        the rows the refresh was for.
        """
        from tabsim import satchecker_names

        cache = TextOrbitCache(isolated_cache)
        cache.store_search(
            "THING",
            search_frame(
                [search_row(3001, "THING A"), search_row(3002, "THING B")]
            ),
            fetched_at=utc(2026, 1, 1),
        )
        monkeypatch.setattr(
            satchecker_names, "_utc_now", lambda: utc(2026, 1, 11)
        )
        serve_search(
            monkeypatch,
            {"THING": [search_row(3002, "THING B"), search_row(3003, "THING C")]},
        )

        ids = orbit.resolve_names(
            ["thing"],
            ISS_EPOCH_JD,
            search_cache_max_age_days=1.0,
            log=lambda *_: None,
        )

        assert ids == [3002, 3003]
        stored = cache.get_search("THING")
        assert sorted(stored.found["NORAD_CAT_ID"]) == [3002, 3003]

    @pytest.mark.parametrize(
        "error", list(SERVICE_FAILURES.values()), ids=list(SERVICE_FAILURES)
    )
    @pytest.mark.parametrize("cached_rows", [2, 0], ids=["rows", "empty"])
    def test_search_refresh_failure_uses_snapshot_with_warning(
        self, monkeypatch, isolated_cache, capsys, error, cached_rows
    ):
        """A stale snapshot is better than no satellites, if the log says so.

        The warning carries which query, when it was fetched, how old that makes
        it, how many rows it holds and why the refresh failed. A cached empty
        result is a valid answer too.
        """
        from tabsim import satchecker_names

        rows = (
            [search_row(3001, "THING A"), search_row(3002, "THING B")]
            if cached_rows
            else []
        )
        TextOrbitCache(isolated_cache).store_search(
            "THING", search_frame(rows), fetched_at=utc(2026, 9, 1)
        )
        monkeypatch.setattr(
            satchecker_names, "_utc_now", lambda: utc(2026, 9, 11)
        )
        serve_search(monkeypatch, {"THING": error})
        log_lines = []

        ids = orbit.resolve_names(
            ["thing"],
            ISS_EPOCH_JD,
            search_cache_max_age_days=1.0,
            log=log_lines.append,
        )

        assert ids == ([3001, 3002] if cached_rows else [])
        messages = _messages(log_lines, capsys)
        assert "Using cached catalogue search" in messages
        assert "THING" in messages
        assert "2026-09-01" in messages        # when it was fetched
        assert "10" in messages                # ...and how old that makes it
        assert str(len(rows)) in messages      # how many rows it holds
        assert str(error) in messages          # and why the refresh failed

    @pytest.mark.parametrize(
        "error", list(SERVICE_FAILURES.values()), ids=list(SERVICE_FAILURES)
    )
    def test_search_failure_without_snapshot_is_fatal(self, monkeypatch, error):
        """With nothing cached, a failed search is not an unmatched name.

        Reporting it as one drops every satellite the query selects and blames the
        configuration.
        """
        serve_search(monkeypatch, {"THING": error})

        with pytest.raises(orbit.OrbitError) as excinfo:
            orbit.resolve_names(["thing"], ISS_EPOCH_JD, log=lambda *_: None)

        message = str(excinfo.value)
        assert "matched nothing" not in message
        assert "thing" in message.lower()

    def test_cached_empty_search_is_not_a_miss(
        self, monkeypatch, isolated_cache, capsys
    ):
        """A search that matched nothing is an answer, and it caches as one."""
        from tabsim import satchecker_names

        cache = TextOrbitCache(isolated_cache)
        cache.store_search(
            "NAVSTAR", search_frame([]), fetched_at=utc(2026, 1, 1)
        )
        monkeypatch.setattr(
            satchecker_names,
            "_utc_now",
            lambda: utc(2026, 1, 1, 1),
        )
        log_lines = []

        # Fresh enough to reuse: the network block proves no request went out.
        assert orbit.resolve_names(
            ["navstar"],
            ISS_EPOCH_JD,
            search_cache_max_age_days=1.0,
            log=log_lines.append,
        ) == []

        # Offline reuses it regardless of age.
        monkeypatch.setattr(
            satchecker_names, "_utc_now", lambda: utc(2026, 6, 1)
        )
        assert orbit.resolve_names(
            ["navstar"],
            ISS_EPOCH_JD,
            search_cache_max_age_days=1.0,
            offline=True,
            log=log_lines.append,
        ) == []
        messages = _messages(log_lines, capsys)
        assert "navstar" in messages.lower()
        assert "no satellite" in messages.lower()

        # ...but offline with nothing cached is a missing-state failure, not a
        # statement about the catalogue.
        with pytest.raises(orbit.OrbitError) as excinfo:
            orbit.resolve_names(
                ["nosuchsat"], ISS_EPOCH_JD, offline=True, log=lambda *_: None
            )
        assert "offline" in str(excinfo.value).lower()

    def test_search_cache_write_failure_is_visible(
        self, monkeypatch, isolated_cache, capsys
    ):
        """A cache we cannot write to costs offline reuse, not the run."""
        from tabsim import satchecker_names

        cache = TextOrbitCache(isolated_cache)
        cache.store_search(
            "THING",
            search_frame([search_row(3001, "THING A")]),
            fetched_at=utc(2026, 1, 1),
        )
        monkeypatch.setattr(
            satchecker_names, "_utc_now", lambda: utc(2026, 1, 11)
        )
        serve_search(monkeypatch, {"THING": [search_row(3002, "THING B")]})
        log_lines = []

        def unwritable(*args, **kwargs):
            raise OSError("read-only file system")

        # Scoped, so the snapshot check below reads through the real method.
        with monkeypatch.context() as patched:
            patched.setattr(TextOrbitCache, "store_search", unwritable)
            ids = orbit.resolve_names(
                ["thing"],
                ISS_EPOCH_JD,
                search_cache_max_age_days=1.0,
                log=log_lines.append,
            )

        # The live result is still what the run uses.
        assert ids == [3002]
        messages = _messages(log_lines, capsys)
        assert cache.search_path("THING").name in messages
        assert "cache" in messages.lower()
        # ...and the snapshot that was already there is untouched.
        remaining = TextOrbitCache(isolated_cache).get_search("THING")
        assert sorted(remaining.found["NORAD_CAT_ID"]) == [3001]

    @pytest.mark.parametrize(
        "payload",
        [
            {"count": 0},                       # no data field at all
            {"count": 0, "data": 0},            # falsy, and not a list
            {"error": "service unavailable"},   # an error envelope, HTTP 200
            {"count": 5, "data": []},           # truncated: count disagrees
        ],
        ids=["no-data", "falsy-data", "error", "count-mismatch"],
    )
    def test_malformed_search_is_not_empty_success(self, monkeypatch, payload):
        """A reply we cannot read is not a catalogue with nothing in it.

        Every lenient reading of a broken envelope ends as a search that matched
        nothing, which drops satellites while reporting a configuration problem.
        """
        serve_raw_search(monkeypatch, payload)

        with pytest.raises(orbit.OrbitError):
            orbit.resolve_names(["thing"], ISS_EPOCH_JD, log=lambda *_: None)

    def test_large_constellation_is_deduplicated_and_not_truncated(
        self, monkeypatch, capsys
    ):
        """A constellation-sized query is costed honestly and kept whole.

        A satellite appears once per alias, so a warning phrased in rows misstates
        the request count twofold here; and a silent cap would model a subset of
        the RFI asked for.
        """
        unique_ids = list(range(80000, 80501))
        rows = []
        for nid in unique_ids:
            rows.append(search_row(nid, f"BIGCON-{nid}"))
            rows.append(search_row(nid, f"BIGCON-{nid} [DTC]"))
        calls = serve_search(monkeypatch, {"BIGCON": rows})
        log_lines = []

        ids = orbit.resolve_names(
            ["bigcon", "bigcon"], ISS_EPOCH_JD, log=log_lines.append
        )

        assert ids == unique_ids
        assert len(set(ids)) == len(ids)
        assert len(calls) == 1, "the repeated name is one search"
        messages = _messages(log_lines, capsys)
        assert str(len(rows)) in messages          # 1002 catalogue rows
        assert str(len(unique_ids)) in messages    # 501 unique candidates

        # ...and a duplicated ID is one request, not two.
        record_calls = stub_service(
            monkeypatch,
            {
                ISS_NORAD_ID: tle_record(),
                GPS_NORAD_ID: tle_record(
                    norad_id=GPS_NORAD_ID, line1=GPS_LINE1, line2=GPS_LINE2
                ),
            },
        )
        orbit.resolve_orbits(
            [ISS_NORAD_ID, ISS_NORAD_ID, GPS_NORAD_ID],
            ISS_EPOCH_JD,
            remote_max_age_days=None,
        )
        assert sorted(nid for nid, _, _ in record_calls) == sorted(
            [ISS_NORAD_ID, GPS_NORAD_ID]
        )

    def test_valid_empty_search_is_not_fatal(self, monkeypatch):
        """A name the catalogue really does not know contributes nothing.

        The one search outcome that is not an error: there is no satellite for a
        record to be missing for.
        """
        serve_search(monkeypatch, {"NOSUCHSAT": []})

        assert orbit.resolve_names(
            ["nosuchsat"], ISS_EPOCH_JD, log=lambda *_: None
        ) == []


def damaged_record(norad_id, epoch_jd, damage):
    """A TLE record with the checksum digit missing from the named line(s)."""
    line1, line2 = tle_lines(norad_id, epoch_jd)
    if damage in ("line1", "both"):
        line1 = without_checksum(line1)
    if damage in ("line2", "both"):
        line2 = without_checksum(line2)
    return tle_record(norad_id=norad_id, line1=line1, line2=line2)


class TestChecksumPolicy:
    @pytest.mark.parametrize("damage", ["line1", "line2", "both"])
    @pytest.mark.parametrize("allow", [False, True])
    @pytest.mark.parametrize("route", ["service", "extra_orbit_dir"])
    def test_checksum_policy_matches_fetch_and_explicit_read(
        self, monkeypatch, tmp_path, damage, allow, route
    ):
        """One policy for every route a record can arrive by.

        Rejecting checksum-less lines remotely and accepting them from a file
        advertises a strictness whose workaround is to save the record once.
        """
        norad_id = 7201
        record = damaged_record(norad_id, ISS_EPOCH_JD, damage)
        policy = {"allow_missing_checksum": allow}

        if route == "service":
            stub_service(monkeypatch, {norad_id: record})
        else:
            stub_service(monkeypatch, {})
            pd.DataFrame([record]).to_json(tmp_path / "mine.json")
            policy["extra_orbit_dir"] = str(tmp_path)

        resolution = orbit.resolve_orbits([norad_id], ISS_EPOCH_JD, **policy)

        if not allow:
            assert not resolution.complete
            return
        assert resolution.complete
        accepted = resolution.resolved[norad_id].record
        assert accepted[CHECKSUM_STATUS_FIELD] == STATUS_UNVERIFIED

    @pytest.mark.parametrize("allow", [False, True])
    def test_unverified_provenance_survives_valid_lines_from_a_file(
        self, monkeypatch, tmp_path, allow
    ):
        """A record marked unverified stays unverified however well its lines parse.

        The laundering case: these lines carry correct checksum digits, so every
        check passes on what the record holds now. The status says nothing ever
        verified the digits its source omitted.
        """
        norad_id = 7301
        record = tle_record_at(norad_id, ISS_EPOCH_JD)
        record[CHECKSUM_STATUS_FIELD] = STATUS_UNVERIFIED
        stub_service(monkeypatch, {})
        pd.DataFrame([record]).to_json(tmp_path / "mine.json")

        resolution = orbit.resolve_orbits(
            [norad_id],
            ISS_EPOCH_JD,
            extra_orbit_dir=str(tmp_path),
            allow_missing_checksum=allow,
        )

        if not allow:
            assert not resolution.complete
            return
        assert (
            resolution.resolved[norad_id].record[CHECKSUM_STATUS_FIELD]
            == STATUS_UNVERIFIED
        )

    @pytest.mark.parametrize("allow", [False, True])
    def test_unverified_provenance_survives_valid_lines_through_replay(
        self, monkeypatch, tmp_path, allow
    ):
        """...and through the replay reader, which sees only what was written."""
        norad_id = 7302
        record = tle_record_at(norad_id, ISS_EPOCH_JD)
        record[CHECKSUM_STATUS_FIELD] = STATUS_UNVERIFIED
        replay_dir = write_replay_dir(tmp_path / "input_data", [norad_id], [record])
        forbid_orbit_acquisition(monkeypatch, tmp_path)

        if not allow:
            with pytest.raises(orbit.OrbitError, match="allow_missing_checksum"):
                orbit.load_replay_orbits(str(replay_dir))
            return
        _, records = orbit.load_replay_orbits(
            str(replay_dir), allow_missing_checksum=True
        )
        assert records[0][CHECKSUM_STATUS_FIELD] == STATUS_UNVERIFIED

    def test_verified_records_say_so(self, monkeypatch):
        """The status is recorded for good records too, not only for bad ones.

        An absent status could mean "verified" or "written before the field
        existed", and a later reader must not resolve that in favour of trust.
        """
        stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})

        resolution = orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD)

        assert (
            resolution.resolved[ISS_NORAD_ID].record[CHECKSUM_STATUS_FIELD]
            == STATUS_VERIFIED
        )

    def test_unverified_run_saves_and_replays_without_network(
        self, monkeypatch, tmp_path
    ):
        """A permissive run must be replayable, and only with the same permission.

        Checksum-less records stay out of the shared cache, so the saved run
        records are the only way such a run reproduces offline.
        """
        norad_id = 7202
        record = damaged_record(norad_id, ISS_EPOCH_JD, "both")
        stub_service(monkeypatch, {norad_id: record})

        first = orbit.resolve_orbits(
            [norad_id], ISS_EPOCH_JD, allow_missing_checksum=True
        )
        assert first.complete
        saved_records = first.records()

        replay_dir = write_replay_dir(
            tmp_path / "input_data", first.norad_ids(), saved_records
        )
        forbid_orbit_acquisition(monkeypatch, tmp_path)

        ids, records = orbit.load_replay_orbits(
            str(replay_dir), allow_missing_checksum=True
        )

        assert ids == [norad_id]
        assert records[0]["TLE_LINE1"] == saved_records[0]["TLE_LINE1"]
        assert records[0]["TLE_LINE2"] == saved_records[0]["TLE_LINE2"]
        assert records[0][CHECKSUM_STATUS_FIELD] == STATUS_UNVERIFIED
        times_jd = ISS_EPOCH_JD + np.linspace(0.0, 0.2, 5)
        np.testing.assert_array_equal(
            get_satellite_positions(records, times_jd),
            get_satellite_positions(saved_records, times_jd),
        )

        with pytest.raises(orbit.OrbitError, match="allow_missing_checksum"):
            orbit.load_replay_orbits(str(replay_dir))

    def test_unverified_records_do_not_enter_shared_cache(
        self, monkeypatch, isolated_cache, capsys
    ):
        """The shared cache stays readable, and honest, for every other consumer.

        Older clients reject a whole cache file on meeting a line they cannot
        validate, so the record is used and saved with the run instead.
        """
        norad_id = ISS_NORAD_ID
        cache = TextOrbitCache(isolated_cache)
        cache.store(norad_id, pd.DataFrame([tle_record()]))
        before = cache.get(norad_id).to_dict(orient="records")

        epoch_jd = ISS_EPOCH_JD + 30.0
        stub_service(
            monkeypatch, {norad_id: damaged_record(norad_id, epoch_jd, "both")}
        )

        resolution = orbit.resolve_orbits(
            [norad_id],
            epoch_jd,
            remote_max_age_days=3.0,
            allow_missing_checksum=True,
        )

        assert resolution.complete
        assert (
            resolution.resolved[norad_id].record[CHECKSUM_STATUS_FIELD]
            == STATUS_UNVERIFIED
        )
        assert cache.get(norad_id).to_dict(orient="records") == before
        out = capsys.readouterr().out
        assert "Unverified TLE: missing checksum" in out
        assert str(norad_id) in out
        assert (
            "Not stored in the shared orbit cache; saved run records are "
            "required for offline replay" in out
        )

    def test_local_backslash_repair_reaches_propagator(self, monkeypatch, tmp_path):
        """A repaired line must be repaired everywhere, not merely tolerated.

        If only the validator strips the archive's stray backslash, the defective
        line is what gets propagated, written out and saved for replay.
        """
        stub_service(monkeypatch, {})
        record = tle_record(
            line1=with_stray_backslash(ISS_LINE1),
            line2=with_stray_backslash(ISS_LINE2),
        )
        pd.DataFrame([record]).to_json(tmp_path / "mine.json")

        resolution = orbit.resolve_orbits(
            [ISS_NORAD_ID], ISS_EPOCH_JD, extra_orbit_dir=str(tmp_path)
        )

        assert resolution.complete
        accepted = resolution.resolved[ISS_NORAD_ID].record
        assert accepted["TLE_LINE1"] == ISS_LINE1
        assert accepted["TLE_LINE2"] == ISS_LINE2
        assert record_tle_lines(accepted) == (ISS_LINE1, ISS_LINE2)
        assert accepted[CHECKSUM_STATUS_FIELD] == STATUS_VERIFIED


def writer_cases():
    """``case -> (saved IDs, saved records, the error a bad pairing must raise)``.

    The accepted records carry ``EPOCH_JD`` and ``SEMIMAJOR_AXIS`` because that is
    what ``frame()`` puts on everything a run propagates: without them on the
    input, "the writer does not store them" would prove nothing.
    """
    derived = {"EPOCH_JD": ISS_EPOCH_JD, "SEMIMAJOR_AXIS": 6796.0}
    tle = tle_record(
        norad_id=GPS_NORAD_ID, line1=GPS_LINE1, line2=GPS_LINE2, **derived
    )
    omm = omm_record_from_tle(**derived)
    # Neither double survives DataFrame.to_json at its maximum precision, which
    # writes the first as 0.006663499999999999, nor pandas' own float parser.
    omm["ECCENTRICITY"] = 0.0066635
    omm["BSTAR"] = 3.2e-05
    iss = tle_record()
    return {
        "empty": ([], [], None),
        "tle": ([GPS_NORAD_ID], [tle], None),
        "omm": ([ISS_NORAD_ID], [omm], None),
        "mixed": ([ISS_NORAD_ID, GPS_NORAD_ID], [omm, tle], None),
        "more-ids-than-records": ([ISS_NORAD_ID, GPS_NORAD_ID], [iss], ValueError),
        "more-records-than-ids": ([ISS_NORAD_ID], [iss, tle], ValueError),
        "mismatched-identity": ([GPS_NORAD_ID], [iss], ValueError),
    }


WRITER_CASES = writer_cases()


class TestReplay:
    @pytest.mark.parametrize("case", list(WRITER_CASES))
    def test_the_writer_delegates_and_writes_a_replayable_table(
        self, case, monkeypatch, tmp_path
    ):
        """One writer, in the module that also reads the format back.

        ``zip`` would truncate to the shorter sequence and write a file that reads
        back cleanly while describing different satellites than the run
        propagated, so a misaligned pair is a ``ValueError`` in the caller.
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

        # parse_constant is what refuses a file only Python's own JSON parser
        # would take: json.loads reads a bare NaN happily.
        written = json.loads(path.read_text(), parse_constant=reject_json_constant)
        # A stored EPOCH_JD could only drift out of step with its elements.
        assert "EPOCH_JD" not in written and "SEMIMAJOR_AXIS" not in written
        if not records:
            # Writing nothing would make "no satellite passed the target" and
            # "this is not a replay" the same state on disk.
            assert written == {}
            return

        back = read_legacy_tle_records(tmp_path)
        assert len(back) == len(records)
        kinds = {record[KIND_FIELD] for record in records}
        for position, record in enumerate(records):
            cell = str(position)
            assert written["NORAD_CAT_ID"][cell] == norad_ids[position]
            assert written["DATA_SOURCE"][cell] == record["DATA_SOURCE"]
            if len(kinds) > 1:
                # One table holding both kinds gives each row the other kind's
                # columns as nulls — the format, not an invention.
                absent = (
                    "TLE_LINE1" if record[KIND_FIELD] == KIND_OMM else "MEAN_MOTION"
                )
                assert written[absent][cell] is None
            if record[KIND_FIELD] != KIND_OMM:
                assert written["TLE_LINE1"][cell] == record["TLE_LINE1"]
                continue
            # The same double, not a near one: in the file, and read back out.
            assert written["ECCENTRICITY"][cell] == record["ECCENTRICITY"]
            assert written["BSTAR"][cell] == record["BSTAR"]
            row = back[
                back["NORAD_CAT_ID"].astype("int64") == int(record["NORAD_CAT_ID"])
            ].iloc[0]
            assert row["ECCENTRICITY"] == record["ECCENTRICITY"]
            assert row["BSTAR"] == record["BSTAR"]

    @pytest.mark.parametrize("build", [tle_record, omm_record_from_tle])
    def test_saved_records_read_back_as_themselves(
        self, build, tmp_path, monkeypatch
    ):
        """A run's ``used_orbits.json`` reproduces its trajectories exactly."""
        record = build()
        stub_service(
            monkeypatch,
            {ISS_NORAD_ID: record},
            endpoint="tle" if build is tle_record else "omm",
        )
        first = orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD)
        assert first.complete

        replay_dir = tmp_path / "input_data"
        replay_dir.mkdir()
        orbit.save_orbits_for_reuse(
            replay_dir / "used_orbits.json", first.norad_ids(), first.records()
        )

        # The seams raise rather than answer nothing: a replay that quietly asked
        # the service would still produce a plausible simulation and pass.
        monkeypatch.setattr(client, "fetch_nearest_tle", forbidden("the TLE endpoint"))
        monkeypatch.setattr(client, "fetch_nearest_omm", forbidden("the OMM endpoint"))
        monkeypatch.setenv("ORBIT_CACHE_DIR", str(tmp_path / "empty-cache"))

        second = orbit.resolve_orbits(
            [ISS_NORAD_ID], ISS_EPOCH_JD, extra_orbit_dir=str(replay_dir)
        )
        assert second.complete

        # Exactly, not approximately: for OMM this needs the elements written at
        # full float64 precision.
        times_jd = ISS_EPOCH_JD + np.linspace(0.0, 0.2, 5)
        before = get_satellite_positions(first.records(), times_jd)
        after = get_satellite_positions(second.records(), times_jd)
        np.testing.assert_array_equal(before, after)

    @pytest.mark.parametrize("damaged", ["record", "aligned_id"])
    def test_save_rejects_a_lossy_identity_match(self, tmp_path, damaged):
        """25544.5 is not 25544, and truncating it saves a different satellite.

        Two identities per row — the record's own and the aligned one — and
        ``int()`` on either repaired the mismatch into agreement.
        """
        record = tle_record()  # valid ISS lines...
        norad_ids = [ISS_NORAD_ID]
        if damaged == "record":
            record["NORAD_CAT_ID"] = 25544.5  # ...under an identity that is not an ID
        else:
            norad_ids = [25544.5]

        with pytest.raises(ValueError, match="25544.5"):
            orbit.save_orbits_for_reuse(
                tmp_path / "used_orbits.json", norad_ids, [record]
            )

    @pytest.mark.parametrize("value", [float("nan"), None], ids=["nan", "absent"])
    def test_save_rejects_an_omm_missing_a_required_element(self, tmp_path, value):
        """A dropped element writes a file that cannot be replayed at all.

        Skipping every null cell is right for the OMM columns a TLE row acquires
        in a mixed frame and wrong for an OMM's own elements.
        """
        record = omm_record_from_tle()
        record["MEAN_MOTION"] = value

        with pytest.raises(ValueError, match="MEAN_MOTION"):
            orbit.save_orbits_for_reuse(
                tmp_path / "used_orbits.json", [ISS_NORAD_ID], [record]
            )

    # -- frozen replay -------------------------------------------------------

    def test_replay_freezes_final_ids(self, monkeypatch, tmp_path):
        """The saved IDs are the selection, not an input to it.

        ``extra_orbit_dir`` only changed where a record came from; a replay has to
        reproduce a run whose name search no longer returns the same catalogue.
        """
        replay_dir = write_replay_dir(
            tmp_path / "input_data", [ISS_NORAD_ID], [tle_record()]
        )
        forbid_orbit_acquisition(monkeypatch, tmp_path)

        ids, records = orbit.load_replay_orbits(str(replay_dir))

        assert ids == [ISS_NORAD_ID]
        assert records[0]["TLE_LINE1"] == ISS_LINE1

    @pytest.mark.parametrize(
        "damage,expected",
        [
            ("missing-directory", "input_data"),
            ("missing-records-file", "used_orbits.json"),
            ("missing-id-file", "norad_ids.yaml"),
            ("corrupt-json", "used_orbits.json"),
            ("missing-id", "99999"),
            ("extra-id", str(GPS_NORAD_ID)),
            ("duplicate-id-line", "more than once"),
            ("duplicate-record-rows", "holds 2 records"),
            ("wrong-embedded-id", "not acceptable under this run's policy"),
            ("rounding-id", "non-integer"),
        ],
    )
    def test_replay_requires_exact_saved_records(
        self, monkeypatch, tmp_path, damage, expected
    ):
        """Replay has no second source, so anything short of exact must stop.

        Each case is pinned to the rejection it is about: two rows for one
        satellite and two ID lines for one are different checks, and sharing a
        "duplicate" fixture let whichever ran first answer for both.
        """
        gps = tle_record(norad_id=GPS_NORAD_ID, line1=GPS_LINE1, line2=GPS_LINE2)
        replay_dir = write_replay_dir(
            tmp_path / "input_data", [ISS_NORAD_ID, GPS_NORAD_ID], [tle_record(), gps]
        )
        records_path = replay_dir / "used_orbits.json"
        ids_path = replay_dir / "norad_ids.yaml"

        if damage == "missing-directory":
            records_path.unlink()
            ids_path.unlink()
            replay_dir.rmdir()
        elif damage == "missing-records-file":
            records_path.unlink()
        elif damage == "missing-id-file":
            ids_path.unlink()
        elif damage == "corrupt-json":
            records_path.write_text(records_path.read_text()[:-12])
        elif damage == "missing-id":
            ids_path.write_text(f"{ISS_NORAD_ID}\n{GPS_NORAD_ID}\n99999\n")
        elif damage == "extra-id":
            ids_path.write_text(f"{ISS_NORAD_ID}\n")
        elif damage == "duplicate-id-line":
            ids_path.write_text(f"{ISS_NORAD_ID}\n{ISS_NORAD_ID}\n")
        elif damage == "rounding-id":
            # A float conversion rounds this to exactly 25544.0 and the replay
            # would quietly proceed with the ISS.
            ids_path.write_text(f"{ISS_NORAD_ID}.000000000001\n{GPS_NORAD_ID}\n")
        elif damage == "duplicate-record-rows":
            # Two saved records for one satellite: choosing between them is the
            # reselection a frozen replay exists to prevent.
            orbit.save_orbits_for_reuse(
                records_path,
                [ISS_NORAD_ID, ISS_NORAD_ID],
                [tle_record(), tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD - 1.0)],
            )
            ids_path.write_text(f"{ISS_NORAD_ID}\n")
        else:  # a row filed under one ID carrying another satellite's lines
            payload = json.loads(records_path.read_text())
            payload["NORAD_CAT_ID"]["1"] = 99999  # unique, so the row IDs still are
            records_path.write_text(json.dumps(payload))
            ids_path.write_text(f"{ISS_NORAD_ID}\n99999\n")

        forbid_orbit_acquisition(monkeypatch, tmp_path)

        with pytest.raises(orbit.OrbitError) as excinfo:
            orbit.load_replay_orbits(str(replay_dir))

        assert expected in str(excinfo.value)

    def test_empty_replay_is_explicit_and_network_free(self, monkeypatch, tmp_path):
        """Replaying a satellite-free run returns zero satellites, deliberately."""
        replay_dir = write_replay_dir(tmp_path / "input_data", [], [])
        assert (replay_dir / "used_orbits.json").exists()
        assert (replay_dir / "norad_ids.yaml").exists()
        forbid_orbit_acquisition(monkeypatch, tmp_path)

        ids, records = orbit.load_replay_orbits(str(replay_dir))

        assert ids == []
        assert records == []

    @pytest.mark.parametrize(
        "kinds",
        [(KIND_TLE, KIND_TLE), (KIND_OMM, KIND_OMM), (KIND_TLE, KIND_OMM)],
        ids=["all-tle", "all-omm", "mixed"],
    )
    def test_mixed_and_omm_replay_preserve_values_and_positions(
        self, monkeypatch, tmp_path, kinds
    ):
        """Frozen replay reproduces both kinds, and the awkward floats in them."""
        ids = [ISS_NORAD_ID, GPS_NORAD_ID]
        records = [
            record_at(kind, nid, ISS_EPOCH_JD) for kind, nid in zip(kinds, ids)
        ]
        for record in records:
            if record[KIND_FIELD] == KIND_OMM:
                record["ECCENTRICITY"] = 0.0066635
                record["BSTAR"] = 3.2e-05

        replay_dir = write_replay_dir(tmp_path / "input_data", ids, records)
        forbid_orbit_acquisition(monkeypatch, tmp_path)

        replayed_ids, replayed = orbit.load_replay_orbits(str(replay_dir))

        assert replayed_ids == ids
        for original, back in zip(records, replayed):
            if original[KIND_FIELD] == KIND_OMM:
                assert back["ECCENTRICITY"] == original["ECCENTRICITY"]
                assert back["BSTAR"] == original["BSTAR"]
        times_jd = ISS_EPOCH_JD + np.linspace(0.0, 0.2, 5)
        np.testing.assert_array_equal(
            get_satellite_positions(replayed, times_jd),
            get_satellite_positions(records, times_jd),
        )


class FakeObservation:
    """The little of ``Observation`` that satellite source selection touches."""

    def __init__(self, epoch_jd=ISS_EPOCH_JD, times_jd=None):
        import dask.array as da

        mjd = epoch_jd - 2400000.5
        self.times_mjd = (
            np.array([mjd - 5e-5, mjd + 5e-5])
            if times_jd is None
            else np.asarray(times_jd, dtype=float) - 2400000.5
        )
        self.latitude, self.longitude, self.elevation = -30.7, 21.44, 1050.0
        self.ra, self.dec = 27.0, -30.0
        self.freqs = da.asarray([1.227e9])
        self.n_freq = 1
        self.added = []

    def addTLESatelliteRFI(self, Pv, norad_ids, orbits):
        self.added.append((list(int(n) for n in norad_ids), list(orbits)))


def sim_rfi_config(**overrides):
    """The ``rfi_sources`` section the simulation's satellite selection reads."""
    from importlib.resources import files

    rfi_dir = files("tabsim.data").joinpath("rfi").__str__()
    tle_satellite = {
        "sat_names": [],
        "norad_ids": [],
        "norad_ids_path": None,
        "norad_spec_model": f"{rfi_dir}/norad_satellite.rfimodel",
        "max_n_sat": None,
        "max_ang_sep": 90,
        "min_alt": 0,
        "extra_orbit_dir": None,
        "extra_orbit_max_age_days": None,
        "remote_max_age_days": 3,
        "cache_reuse_max_age_days": 1,
        "replay_orbit_dir": None,
        "offline": False,
        "allow_missing_checksum": False,
        "vis_step": 1.0,
        "power_scale": 1.0,
    }
    tle_satellite.update(overrides)
    return {"rfi_sources": {"tle_satellite": tle_satellite}}


def replay_sim_config(replay_dir, **overrides):
    """The same section, with a frozen replay selected."""
    return sim_rfi_config(replay_orbit_dir=str(replay_dir), **overrides)


class TestSelectionEpoch:
    def test_selection_epoch_is_the_observation_not_the_visibility_grid(
        self, monkeypatch
    ):
        """Satellites are judged at the observation's epoch, not the check grid's.

        The visibility grid steps past the last sample, so this observation's grid
        has a mean on the following date — which excluded a satellite the
        catalogue says decayed on the observation's own date.
        """
        from tabsim import config as config_module

        epoch_jd = jd(2023, 2, 21, 23, 59, 45)
        times_jd = [jd(2023, 2, 21, 23, 59, 40), jd(2023, 2, 21, 23, 59, 50)]
        serve_search(
            monkeypatch,
            {
                "THING": [
                    search_row(
                        ISS_NORAD_ID,
                        "THING ONE",
                        launch_date="2000-01-01",
                        decay_date="2023-02-21",
                    )
                ]
            },
        )
        calls = stub_service(
            monkeypatch, {ISS_NORAD_ID: tle_record_at(ISS_NORAD_ID, epoch_jd)}
        )
        monkeypatch.setattr(
            tle_module,
            "check_satellite_visibilibities",
            lambda *args, **kwargs: pd.DataFrame({"norad_id": [ISS_NORAD_ID]}),
        )
        obs = FakeObservation(times_jd=times_jd)

        config_module.add_tle_satellite_sources(
            obs, sim_rfi_config(sat_names=["thing"], vis_step=1.0)
        )

        assert [nid for ids, _ in obs.added for nid in ids] == [ISS_NORAD_ID]
        assert calls[0][1] == pytest.approx(epoch_jd, abs=1e-6)


class TestReplaySelection:
    @pytest.mark.parametrize("max_n_sat", [None, 1, 0])
    def test_replay_is_not_reselected_by_limits_or_spectra(
        self, monkeypatch, tmp_path, max_n_sat
    ):
        """Saved IDs are the answer; the settings that produced them are not re-run.

        A ``max_n_sat`` or visibility cut changed since the original run would
        otherwise drop satellites from a replay that exists to keep them.
        """
        from tabsim import config as config_module

        gps = tle_record(norad_id=GPS_NORAD_ID, line1=GPS_LINE1, line2=GPS_LINE2)
        replay_dir = write_replay_dir(
            tmp_path / "input_data", [ISS_NORAD_ID, GPS_NORAD_ID], [tle_record(), gps]
        )
        forbid_orbit_acquisition(monkeypatch, tmp_path)
        obs = FakeObservation()

        config_module.add_tle_satellite_sources(
            obs, replay_sim_config(replay_dir, max_n_sat=max_n_sat, min_alt=89)
        )

        added = sorted(nid for ids, _ in obs.added for nid in ids)
        assert added == sorted([ISS_NORAD_ID, GPS_NORAD_ID])

    def test_missing_spectral_model_for_a_saved_id_is_fatal(
        self, monkeypatch, tmp_path
    ):
        """A saved satellite with no spectrum is a broken replay, not a smaller one."""
        from tabsim import config as config_module

        unknown = 99999  # not in the shipped norad_satellite.rfimodel
        replay_dir = write_replay_dir(
            tmp_path / "input_data",
            [unknown],
            [tle_record_at(unknown, ISS_EPOCH_JD)],
        )
        forbid_orbit_acquisition(monkeypatch, tmp_path)

        with pytest.raises(orbit.OrbitError) as excinfo:
            config_module.add_tle_satellite_sources(
                FakeObservation(), replay_sim_config(replay_dir)
            )

        assert str(unknown) in str(excinfo.value)

    def test_replay_logs_what_it_overrides(self, monkeypatch, tmp_path, capsys):
        """The log has to say the run's own selection settings were ignored."""
        from tabsim import config as config_module

        replay_dir = write_replay_dir(
            tmp_path / "input_data", [ISS_NORAD_ID], [tle_record()]
        )
        forbid_orbit_acquisition(monkeypatch, tmp_path)
        obs = FakeObservation()

        config_module.add_tle_satellite_sources(
            obs,
            replay_sim_config(replay_dir, sat_names=["navstar"], norad_ids=[7001]),
        )

        # The run's own names and ID list were overridden, not merged in.
        assert sorted(nid for ids, _ in obs.added for nid in ids) == [ISS_NORAD_ID]
        out = capsys.readouterr().out
        assert "Frozen orbit replay" in out


class TestConfiguration:
    @pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "soon", True])
    def test_bad_age_is_a_configuration_error(self, value):
        with pytest.raises(TLEConfigurationError):
            validate_age_days(value, "remote_max_age_days")

    def test_null_age_means_no_limit(self):
        assert validate_age_days(None, "remote_max_age_days") is None

    def test_reuse_age_may_not_exceed_the_ceiling(self):
        """Otherwise a cached record suppresses the request that would replace it."""
        with pytest.raises(TLEConfigurationError, match="must not exceed"):
            normalise_orbit_config(
                {"remote_max_age_days": 1, "cache_reuse_max_age_days": 5}
            )

    @pytest.mark.parametrize(
        "value",
        [
            [1.5], [0], [-3], ["abc"], [None], "25544", ["25544.000000000001"],
            [True], [np.bool_(True)], np.array([True, False]),
        ],
    )
    def test_bad_norad_ids_are_rejected_before_the_resolver(self, value):
        # "25544.000000000001" is the case a float conversion gets wrong: it
        # rounds to exactly 25544.0 and would select the ISS.
        with pytest.raises(TLEConfigurationError):
            normalise_norad_ids(value)

    @pytest.mark.parametrize(
        "value,expected",
        [
            (["25544.0", " 32260 ", "2.5544e4"], [25544, 32260]),
            ([3, 1, 3, 2, "1"], [3, 1, 2]),
            # Config lists routinely arrive as floats; only fractional ones are wrong.
            (np.array([25544.0, 32260.0]), [25544, 32260]),
        ],
        ids=["integral-decimals", "deduplicated-in-order", "numpy-floats"],
    )
    def test_exactly_integral_norad_ids_are_accepted(self, value, expected):
        assert normalise_norad_ids(value) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("# satellites\n25544 ISS\n\n32260  GPS\n25544\n", [25544, 32260]),
            ("25544\nnot-an-id\n", r"ids\.txt:2"),
            # A float conversion rounds this to exactly 25544.0 and selects the ISS.
            ("25544.000000000001\n", r"ids\.txt:1"),
        ],
        ids=["first-column-only", "error-names-the-line", "only-rounds-to-an-integer"],
    )
    def test_the_norad_id_file_is_read_line_by_line(self, tmp_path, text, expected):
        path = tmp_path / "ids.txt"
        path.write_text(text)
        if isinstance(expected, list):
            assert read_norad_ids_file(path) == expected
            return
        with pytest.raises(TLEConfigurationError, match=expected):
            read_norad_ids_file(path)

    def test_config_merges_ids_and_the_id_file(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("32260\n25544\n")
        config = normalise_orbit_config(
            {"norad_ids": [25544], "norad_ids_path": str(path), "sat_names": ["ISS"]}
        )
        assert config.norad_ids == [25544, 32260]
        assert config.sat_names == ["ISS"]

    def test_config_defaults(self):
        config = normalise_orbit_config({})
        assert config.remote_max_age_days == 3.0
        assert config.cache_reuse_max_age_days == 1.0
        assert config.extra_orbit_max_age_days is None
        assert config.extra_orbit_dir is None

    def test_observation_epoch_is_the_mean(self):
        assert observation_epoch_jd([2460000.0, 2460002.0]) == 2460001.0


class TestFrame:
    @pytest.mark.parametrize(
        "build,kind", [(tle_record, KIND_TLE), (omm_record_from_tle, KIND_OMM)]
    )
    def test_frame_carries_locally_derived_elements(self, build, kind, monkeypatch):
        stub_service(
            monkeypatch,
            {ISS_NORAD_ID: build()},
            endpoint="tle" if kind == KIND_TLE else "omm",
        )
        frame = orbit.get_orbits_by_id([ISS_NORAD_ID], ISS_EPOCH_JD)

        assert list(frame["NORAD_CAT_ID"]) == [ISS_NORAD_ID]
        assert frame[KIND_FIELD].iloc[0] == kind
        assert frame["EPOCH_JD"].iloc[0] == pytest.approx(ISS_EPOCH_JD, abs=1e-8)
        assert frame["SEMIMAJOR_AXIS"].iloc[0] == pytest.approx(6796.0, abs=5.0)
        assert frame["INCLINATION"].iloc[0] == pytest.approx(51.6389)

    def test_rows_follow_the_requested_order(self, monkeypatch):
        stub_service(
            monkeypatch,
            {
                ISS_NORAD_ID: tle_record(),
                GPS_NORAD_ID: tle_record(
                    norad_id=GPS_NORAD_ID, line1=GPS_LINE1, line2=GPS_LINE2
                ),
            },
        )
        frame = orbit.get_orbits_by_id(
            [GPS_NORAD_ID, ISS_NORAD_ID], ISS_EPOCH_JD, remote_max_age_days=None
        )
        assert list(frame["NORAD_CAT_ID"]) == [GPS_NORAD_ID, ISS_NORAD_ID]

    def test_no_satellites_is_an_empty_frame_not_an_error(self):
        assert orbit.get_orbits_by_id([], ISS_EPOCH_JD).empty
