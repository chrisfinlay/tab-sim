"""Resolution policy, propagation and configuration validation for orbit records.

Everything here runs offline. The :mod:`satchecker_client` dependency is covered
by its own repository's suite; what these tests exercise is the tabsim side of
the seam — the source precedence, checksum and age policy in :mod:`tabsim.orbit`,
the epoch-aware name discovery and search caching in
:mod:`tabsim.satchecker_names`, the frozen-replay contract, and the propagation of
both record kinds in :mod:`tabsim.tle`.

The network-touching seams are ``client.fetch_nearest_tle`` /
``fetch_nearest_omm`` for records and ``client._http_get`` for the catalogue
search; each test that needs one stubs it, and ``tests/conftest.py`` fails any
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
    forbid_search,
    forbidden,
    jd,
    omm_record_from_tle,
    record_at,
    search_frame,
    search_payload,
    search_row,
    serve_raw_search,
    serve_search,
    stub_failing_service,
    stub_service,
    tle_lines,
    tle_record,
    tle_record_at,
    with_stray_backslash,
    without_checksum,
)


UTC = timezone.utc

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


def reject_json_constant(name):
    """Refuse the non-standard JSON literals ``json`` accepts by default.

    ``json.loads`` reads a bare ``NaN`` or ``Infinity`` happily, so parsing a
    replay file with the defaults cannot show that it contains none. Anything that
    is not another Python JSON parser would reject the file outright.
    """
    raise AssertionError(f"replay file carries the non-standard JSON literal {name}")


def write_replay_dir(directory, norad_ids, records):
    """Write the two files a frozen replay reads, as a completed run would."""
    directory.mkdir(parents=True, exist_ok=True)
    orbit.save_orbits_for_reuse(
        directory / "used_orbits.json", list(norad_ids), list(records)
    )
    (directory / "norad_ids.yaml").write_text(
        "".join(f"{int(nid)}\n" for nid in norad_ids)
    )
    return directory


def forbid_every_orbit_source(monkeypatch, tmp_path):
    """Make every way of obtaining a record other than the replay file fail.

    A replay that quietly fell back to the cache or the service would still
    produce a plausible simulation, just not the one it claims to reproduce — so
    the guard has to *raise* rather than return nothing.
    """
    monkeypatch.setenv("ORBIT_CACHE_DIR", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(client, "fetch_nearest_tle", forbidden("the TLE endpoint"))
    monkeypatch.setattr(client, "fetch_nearest_omm", forbidden("the OMM endpoint"))
    monkeypatch.setattr(orbit, "TextOrbitCache", forbidden("the managed orbit cache"))
    monkeypatch.setattr(
        tle_module, "check_satellite_visibilibities", forbidden("the visibility search")
    )


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

class TestPropagation:
    def test_tle_and_omm_paths_agree(self):
        """The same elements propagate identically whichever format carries them.

        Nothing in the OMM branch of ``earth_satellite`` is checksum- or
        parser-protected, so a wrong unit there would yield a plausible-looking
        orbit that is simply the wrong one. The guard is this direct comparison:
        the OMM record is built from the *same* TLE it is compared against, so a
        degrees-for-radians or rev/day-for-rad/min slip would show up as
        kilometres.

        The residual is bounded at 1 m rather than at zero because an OMM
        ``EPOCH`` is an ISO 8601 string, so the epoch survives the round trip
        only to microsecond resolution — ~15 us here, which is ~0.1 m of
        along-track motion at 7.7 km/s. Every orbital element transfers exactly.
        """
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

        Every historical-epoch test below rests on this: a fixture whose checksum
        was left stale would be rejected by :mod:`satchecker_client`, and the test
        using it would then "fail" for a reason that has nothing to do with the
        behaviour under test.
        """
        from satchecker_client.tle_parse import tle_epoch_jd, validate_tle_pair

        epoch = jd(2001, 3, 9, 4, 30)
        line1, line2 = tle_lines(GPS_NORAD_ID, epoch)
        assert validate_tle_pair(line1, line2) == GPS_NORAD_ID
        assert tle_epoch_jd(line1) == pytest.approx(epoch, abs=1e-7)


# ---------------------------------------------------------------------------
# Source precedence and age policy
# ---------------------------------------------------------------------------

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

    def test_unreadable_extra_orbit_file_is_not_silently_skipped(
        self, tmp_path, monkeypatch
    ):
        """An explicit source we cannot read must name its path, not fall through.

        Silently skipping it is the worst of the options: the run continues, the
        service is asked instead, and the simulation is built from records the
        user explicitly said not to use — with nothing in the log to say the file
        they pointed at was never read.
        """
        stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})
        broken = tmp_path / "used_orbits.json"
        broken.write_text('{"TLE_LINE1": {"0": "1 25544U 98067A   23055')

        with pytest.raises(orbit.OrbitError) as excinfo:
            orbit.resolve_orbits(
                [ISS_NORAD_ID], ISS_EPOCH_JD, extra_orbit_dir=str(tmp_path)
            )

        assert "used_orbits.json" in str(excinfo.value)

    def test_malformed_extra_orbit_identity_is_not_silently_dropped(
        self, tmp_path, monkeypatch
    ):
        """A row whose NORAD_CAT_ID is not an ID stops the run naming the file.

        Coercing identities before validating them made the row disappear: 25544.5
        became NaN, the filter dropped it with no diagnostic at all, and the
        service record the user's file existed to replace was substituted for it.
        An int() cast is worse still — it would truncate to a different
        satellite's catalogue number.
        """
        stub_service(
            monkeypatch, {ISS_NORAD_ID: tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD)}
        )
        record = tle_record()  # valid ISS lines...
        record["NORAD_CAT_ID"] = 25544.5  # ...under an identity that is not an ID
        pd.DataFrame([record]).to_json(tmp_path / "mine.json")

        with pytest.raises(orbit.OrbitError) as excinfo:
            orbit.resolve_orbits(
                [ISS_NORAD_ID], ISS_EPOCH_JD, extra_orbit_dir=str(tmp_path)
            )

        message = str(excinfo.value)
        assert "mine.json" in message
        assert "25544.5" in message

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

        The rule is strictly fresher, not most recently seen. A service that
        answers a refresh with a record further away than the cached one would
        otherwise quietly make the simulation worse than it already was, and the
        log would report a successful fetch.
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
        stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})
        orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD)

        cached = TextOrbitCache(isolated_cache).get(ISS_NORAD_ID)
        assert len(cached) == 1
        assert cached["TLE_LINE1"].iloc[0] == ISS_LINE1

    def test_strict_response_is_requested_from_both_endpoints(self, monkeypatch):
        """Every nearest-record request must opt in to strict response parsing.

        Without it an HTTP-200 error envelope — which is how SatChecker has been
        observed to report its own failures — normalises to an empty frame, and
        an outage becomes "this satellite has no record": the satellite is
        dropped from the simulation and the log says nothing was available.
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
        one, and the other is consulted only when the first has nothing
        acceptable. One request per satellite in the common case is what keeps a
        constellation-sized run affordable for a courtesy service.
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
    def test_missing_record_raises_with_the_remedies(self, monkeypatch):
        stub_service(monkeypatch, {})
        resolution = orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD)

        with pytest.raises(orbit.OrbitError) as excinfo:
            orbit.require_complete_coverage(resolution)

        message = str(excinfo.value)
        assert str(ISS_NORAD_ID) in message
        assert "extra_orbit_dir" in message
        assert "remote_max_age_days" in message

    def test_coverage_error_reports_how_close_the_best_record_was(self, monkeypatch):
        """"4.2 d away" is actionable; "not found" is not."""
        stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})

        resolution = orbit.resolve_orbits(
            [ISS_NORAD_ID], ISS_EPOCH_JD + 10.0, remote_max_age_days=3.0
        )

        with pytest.raises(orbit.OrbitError, match="10.000 d from the"):
            orbit.require_complete_coverage(resolution)

    def test_service_outage_is_distinguished_from_a_missing_satellite(
        self, monkeypatch
    ):
        def unreachable(norad_id, _epoch_jd, *, strict_response=False):
            raise client.SatCheckerTransportError("connection refused")

        monkeypatch.setattr(client, "fetch_nearest_tle", unreachable)
        monkeypatch.setattr(client, "fetch_nearest_omm", unreachable)

        resolution = orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD)

        with pytest.raises(orbit.OrbitError, match="Re-run when the service"):
            orbit.require_complete_coverage(resolution)

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
    @pytest.mark.parametrize(
        "failure",
        [
            "transport",
            "rate-limit",
            "response",
            "invalid-record",
        ],
    )
    def test_unresolved_service_failure_is_fatal_for_names_and_numbers(
        self, monkeypatch, route, failure
    ):
        """"We could not find out" is not "there is nothing there".

        A named satellite whose record could not be *obtained* used to be warned
        about and dropped, so a SatChecker outage produced a complete-looking
        observation with no satellite RFI in it — indistinguishable from a
        correct simulation of a quiet sky. Both routes now stop.
        """
        norad_id = 7001
        if failure == "transport":
            stub_failing_service(
                monkeypatch, client.SatCheckerTransportError("connection refused")
            )
        elif failure == "rate-limit":
            stub_failing_service(
                monkeypatch,
                client.SatCheckerRateLimitError("slow down", retry_after=30.0),
            )
        elif failure == "response":
            stub_failing_service(
                monkeypatch,
                client.SatCheckerResponseError("unreadable reply", status=500),
            )
        else:  # a reply that arrived but carries no usable record
            corrupt = tle_record_at(norad_id, ISS_EPOCH_JD)
            corrupt["TLE_LINE2"] = corrupt["TLE_LINE2"][:68] + "9"
            stub_service(monkeypatch, {norad_id: corrupt})

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

        SatChecker has been observed reporting its own failures with HTTP 200 and
        an ``error`` field, which the lenient reading normalises to an empty frame
        — so the named route would exclude the satellite and the run would finish
        without it. Every other test here stubs ``fetch_nearest_tle`` itself and
        records the ``strict_response`` argument; this one goes through the real
        wrapper from the transport up, so what is under test is that the opt-in
        reaches the parser and changes the answer.
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

        Only the third is a reason to re-run unchanged, so collapsing them into
        one message costs the user the only remedy that works.
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

        The cached record is within the hard ceiling, so the simulation is
        legitimate — but it is not the simulation the user asked for, and the log
        is the only place that can say so.
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

    def test_refresh_failure_names_the_source_that_answered(self, monkeypatch, capsys):
        """The warning has to name the record the run actually continued with.

        An ID rescued by the other archive is bookkept as a failed refresh, and
        every one of those was reported as "continuing with the acceptable cached
        record(s) already held" — with an empty cache and a record that had just
        come back from SatChecker.
        """

        def failing_tle(norad_id, _epoch_jd, *, strict_response=False):
            raise client.SatCheckerResponseError("unreadable reply", status=500)

        def nearest_omm(norad_id, _epoch_jd, *, strict_response=False):
            return pd.DataFrame([omm_record_from_tle()])

        monkeypatch.setattr(client, "fetch_nearest_tle", failing_tle)
        monkeypatch.setattr(client, "fetch_nearest_omm", nearest_omm)

        resolution = orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD)

        assert resolution.complete
        assert ISS_NORAD_ID in resolution.refresh_errors
        out = capsys.readouterr().out
        assert "nearest-OMM" in out
        assert "cached record" not in out

    def test_refresh_failure_detail_honours_the_log_switch(
        self, monkeypatch, isolated_cache, capsys
    ):
        """``TABSIM_TLE_LOG_DETAIL=1`` has to reach the refresh summary too.

        It sliced to the first twelve whatever the switch said, so the one thing
        that exists to recover a full per-satellite listing could not recover
        this one.
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
        monkeypatch.setenv("TABSIM_TLE_LOG_DETAIL", "1")

        resolution = orbit.resolve_orbits(
            norad_ids,
            ISS_EPOCH_JD + 2.0,
            remote_max_age_days=3.0,
            cache_reuse_max_age_days=1.0,
        )

        assert resolution.complete
        out = capsys.readouterr().out
        for nid in norad_ids:
            # tabsim's own summary entry, not the client's per-request log line.
            assert f"{nid} — no answer for {nid} (from " in out

    def test_fallback_success_clears_prior_failure(self, monkeypatch):
        """A per-ID failure from one archive is not a failure of the run."""
        record = omm_record_from_tle()

        def failing_tle(norad_id, _epoch_jd, *, strict_response=False):
            raise client.SatCheckerResponseError("unreadable reply", status=500)

        def nearest_omm(norad_id, _epoch_jd, *, strict_response=False):
            return pd.DataFrame([record])

        monkeypatch.setattr(client, "fetch_nearest_tle", failing_tle)
        monkeypatch.setattr(client, "fetch_nearest_omm", nearest_omm)

        resolution = orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD)

        assert resolution.complete
        assert resolution.service_errors == {}
        orbit.require_complete_coverage(resolution)

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

        Offline discovery and offline record acquisition are separate
        guarantees, so a run that has the first and not the second must say it
        ran out of local state — not that SatChecker has no record.
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

        Offline, an age rejection excluded the named satellite as though the
        catalogue had been asked and had nothing near the observation. Nothing was
        asked: a ten-day-old cached record says only that this machine holds a
        ten-day-old record. The exclusion made a satellite-free simulation look
        like a legitimate answer, which is what offline running must never do.
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

        Both halves matter. The request has to carry the observation epoch, or
        SatChecker answers with a record from the wrong decade; and the catalogue
        filtering has to use it too, or a satellite that decayed between the
        observation and now is silently dropped from a historical run while one
        launched since is silently added.
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


# ---------------------------------------------------------------------------
# Name discovery and the shared search cache
# ---------------------------------------------------------------------------

def _messages(log_lines, capsys):
    """Everything a lookup reported, whichever channel it used."""
    return "\n".join(log_lines) + "\n" + capsys.readouterr().out


class TestNameDiscovery:
    def test_names_use_public_substring_search(self, monkeypatch):
        """Discovery goes through the package's public search, nothing private.

        tabsim held its own copy of the client's private transport helper and its
        own response parser, which is how a malformed reply became "no satellite
        matches this name". Using the public wrapper is what makes the client's
        envelope checks apply here, and is the only seam a test can stub.
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
        answers it wrongly in both directions for any epoch that is not today.
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

        A null or later launch date on one alias row means that row does not say,
        not that the satellite had not launched. De-duplicating by ID first and
        reading the surviving row's dates therefore rules out satellites the
        catalogue never ruled out.
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

    def test_same_object_under_two_ids_is_not_collapsed(self, monkeypatch):
        """Two catalogue numbers for one object are two candidates, not one.

        Nothing in the response says which number is current, so guessing an
        identity merge would silently drop a satellite. The ambiguity is reported
        instead.
        """
        rows = [
            search_row(61608, "TWIN SAT", object_id="2024-100A"),
            search_row(72115, "TWIN SAT", object_id="2024-100A"),
        ]
        serve_search(monkeypatch, {"TWIN": rows})

        ids = orbit.resolve_names(["twin"], ISS_EPOCH_JD, log=lambda *_: None)
        assert ids == [61608, 72115]

        # Only the nearer record is acceptable, so only one resolves.
        stub_service(
            monkeypatch,
            {
                61608: tle_record_at(61608, ISS_EPOCH_JD),
                72115: tle_record_at(72115, ISS_EPOCH_JD - 10.0),
            },
        )
        resolution = orbit.resolve_orbits(ids, ISS_EPOCH_JD, remote_max_age_days=3.0)
        assert sorted(resolution.resolved) == [61608]

    def test_two_ids_for_one_object_that_both_resolve_are_both_kept(
        self, monkeypatch, capsys
    ):
        """...and the shared international designator is reported as ambiguous."""
        rows = [
            search_row(61608, "TWIN SAT", object_id="2024-100A"),
            search_row(72115, "TWIN SAT", object_id="2024-100A"),
        ]
        serve_search(monkeypatch, {"TWIN": rows})
        log_lines = []

        ids = orbit.resolve_names(["twin"], ISS_EPOCH_JD, log=log_lines.append)
        stub_service(
            monkeypatch,
            {
                61608: tle_record_at(61608, ISS_EPOCH_JD),
                72115: tle_record_at(72115, ISS_EPOCH_JD),
            },
        )
        resolution = orbit.resolve_orbits(ids, ISS_EPOCH_JD, remote_max_age_days=3.0)

        assert sorted(resolution.resolved) == [61608, 72115]
        messages = _messages(log_lines, capsys)
        assert "OBJECT_ID" in messages
        assert "2024-100A" in messages
        assert "61608" in messages and "72115" in messages

    def test_shared_designator_warning_is_about_candidates(self, monkeypatch, capsys):
        """The warning is issued at discovery, so it can only speak of candidates.

        Either number may still fail age coverage and be excluded, as one does
        here. Saying then and there that both are kept as distinct satellites and
        "if they are one object it is modelled twice" describes a resolution that
        has not happened yet, and in this run does not happen.
        """
        rows = [
            search_row(61608, "TWIN SAT", object_id="2024-100A"),
            search_row(72115, "TWIN SAT", object_id="2024-100A"),
        ]
        serve_search(monkeypatch, {"TWIN": rows})
        log_lines = []

        ids = orbit.resolve_names(["twin"], ISS_EPOCH_JD, log=log_lines.append)
        stub_service(
            monkeypatch,
            {
                61608: tle_record_at(61608, ISS_EPOCH_JD),
                72115: tle_record_at(72115, ISS_EPOCH_JD - 10.0),
            },
        )
        orbit.report_named_coverage(
            orbit.resolve_orbits(ids, ISS_EPOCH_JD, remote_max_age_days=3.0),
            log=log_lines.append,
        )

        messages = _messages(log_lines, capsys)
        assert "candidate NORAD catalogue ID" in messages
        assert "both are kept as distinct satellites" not in messages
        # ...and one of the two was in fact excluded.
        assert "No acceptable record for named satellite 72115" in messages

    def test_search_snapshot_is_full_and_beside_orbit_files(
        self, monkeypatch, isolated_cache
    ):
        """The whole catalogue result is cached, not the IDs it boiled down to.

        The epoch filter is applied per observation, so a snapshot reduced to one
        epoch's candidates could not answer a second observation — and the alias
        rows are the evidence the launch/decay combination rests on.
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

    def test_fresh_search_snapshot_suppresses_transport(
        self, monkeypatch, isolated_cache
    ):
        """A search fresh enough by the wall clock is reused without a request."""
        from tabsim import satchecker_names

        TextOrbitCache(isolated_cache).store_search(
            "NAVSTAR",
            search_frame([search_row(24876, "NAVSTAR 43 (USA 132)")]),
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        monkeypatch.setattr(
            satchecker_names,
            "_utc_now",
            lambda: datetime(2026, 1, 1, 12, tzinfo=UTC),
        )

        # The suite's network block is still in force: any request fails here.
        ids = orbit.resolve_names(
            ["navstar"],
            ISS_EPOCH_JD,
            search_cache_max_age_days=1.0,
            log=lambda *_: None,
        )

        assert ids == [24876]

    def test_stale_search_refresh_replaces_snapshot(
        self, monkeypatch, isolated_cache
    ):
        """A refreshed search replaces the old one; it is not merged into it.

        A satellite absent from the new result is absent from the catalogue as it
        stands, and keeping it because an older search saw it would resurrect
        exactly the rows the refresh was for.
        """
        from tabsim import satchecker_names

        cache = TextOrbitCache(isolated_cache)
        cache.store_search(
            "THING",
            search_frame(
                [search_row(3001, "THING A"), search_row(3002, "THING B")]
            ),
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        monkeypatch.setattr(
            satchecker_names, "_utc_now", lambda: datetime(2026, 1, 11, tzinfo=UTC)
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
        "error",
        [
            client.SatCheckerTransportError("connection refused"),
            client.SatCheckerRateLimitError("slow down", retry_after=30.0),
            client.SatCheckerResponseError("unreadable reply", status=500),
        ],
        ids=["transport", "rate-limit", "response"],
    )
    @pytest.mark.parametrize("cached_rows", [2, 0], ids=["rows", "empty"])
    def test_search_refresh_failure_uses_snapshot_with_warning(
        self, monkeypatch, isolated_cache, capsys, error, cached_rows
    ):
        """A stale snapshot is better than no satellites, if the log says so.

        The warning has to carry enough to judge the result: which query, when it
        was fetched, how old that makes it, how many rows it holds, and why the
        refresh failed. A cached empty result is a valid answer too.
        """
        from tabsim import satchecker_names

        rows = (
            [search_row(3001, "THING A"), search_row(3002, "THING B")]
            if cached_rows
            else []
        )
        TextOrbitCache(isolated_cache).store_search(
            "THING", search_frame(rows), fetched_at=datetime(2026, 9, 1, tzinfo=UTC)
        )
        monkeypatch.setattr(
            satchecker_names, "_utc_now", lambda: datetime(2026, 9, 11, tzinfo=UTC)
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

    def test_zero_search_freshness_refreshes_every_online_lookup(
        self, monkeypatch, isolated_cache
    ):
        """``search_cache_max_age_days: 0`` refreshes, however new the snapshot is.

        Not "reuse anything younger than zero days", which a snapshot fetched this
        instant satisfies: the setting exists to say *always ask*, and the run has
        to use what comes back.
        """
        from tabsim import satchecker_names

        cache = TextOrbitCache(isolated_cache)
        cache.store_search(
            "THING",
            search_frame([search_row(3001, "THING A")]),
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        monkeypatch.setattr(
            satchecker_names, "_utc_now", lambda: datetime(2026, 1, 1, tzinfo=UTC)
        )
        calls = serve_search(monkeypatch, {"THING": [search_row(3002, "THING B")]})

        ids = orbit.resolve_names(
            ["thing"], ISS_EPOCH_JD, search_cache_max_age_days=0, log=lambda *_: None
        )

        assert ids == [3002]
        assert len(calls) == 1
        assert sorted(cache.get_search("THING").found["NORAD_CAT_ID"]) == [3002]

    def test_null_search_freshness_reuses_a_snapshot_indefinitely(
        self, monkeypatch, isolated_cache
    ):
        """``null`` is the opt-out from refreshing at all, at any age."""
        from tabsim import satchecker_names

        TextOrbitCache(isolated_cache).store_search(
            "THING",
            search_frame([search_row(3001, "THING A")]),
            fetched_at=datetime(2020, 1, 1, tzinfo=UTC),
        )
        monkeypatch.setattr(
            satchecker_names, "_utc_now", lambda: datetime(2026, 1, 1, tzinfo=UTC)
        )
        forbid_search(monkeypatch)  # six years old, and still no request

        assert orbit.resolve_names(
            ["thing"],
            ISS_EPOCH_JD,
            search_cache_max_age_days=None,
            log=lambda *_: None,
        ) == [3001]

    @pytest.mark.parametrize(
        "error",
        [
            client.SatCheckerTransportError("connection refused"),
            client.SatCheckerRateLimitError("slow down", retry_after=30.0),
            client.SatCheckerResponseError("unreadable reply", status=500),
        ],
        ids=["transport", "rate-limit", "response"],
    )
    def test_search_failure_without_snapshot_is_fatal(self, monkeypatch, error):
        """With nothing cached, a failed search is not an unmatched name.

        Reporting it as one drops every satellite the query would have selected
        and says the configuration was wrong.
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
            "NAVSTAR", search_frame([]), fetched_at=datetime(2026, 1, 1, tzinfo=UTC)
        )
        monkeypatch.setattr(
            satchecker_names,
            "_utc_now",
            lambda: datetime(2026, 1, 1, 1, tzinfo=UTC),
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
            satchecker_names, "_utc_now", lambda: datetime(2026, 6, 1, tzinfo=UTC)
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
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        monkeypatch.setattr(
            satchecker_names, "_utc_now", lambda: datetime(2026, 1, 11, tzinfo=UTC)
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

        Every lenient way of reading a broken envelope ends identically — as a
        search that matched nothing — which drops every satellite the query
        selected while reporting a configuration problem.
        """
        serve_raw_search(monkeypatch, payload)

        with pytest.raises(orbit.OrbitError):
            orbit.resolve_names(["thing"], ISS_EPOCH_JD, log=lambda *_: None)

    def test_large_constellation_is_deduplicated_and_not_truncated(
        self, monkeypatch, capsys
    ):
        """A constellation-sized query is costed honestly and kept whole.

        Catalogue rows are not satellites — a satellite appears once per alias —
        so a warning phrased in rows misstates the request count by a factor of
        two here. And nothing may be capped: a silent truncation would model a
        subset of the RFI the user asked for.
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

        This is the one search outcome that is not an error: there is no
        satellite for a record to be missing for. A *failed* search is a
        different thing and now stops the run.
        """
        serve_search(monkeypatch, {"NOSUCHSAT": []})

        assert orbit.resolve_names(
            ["nosuchsat"], ISS_EPOCH_JD, log=lambda *_: None
        ) == []


# ---------------------------------------------------------------------------
# Checksum policy
# ---------------------------------------------------------------------------

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

        A default that rejects checksum-less lines remotely and accepts them from
        a file is the worst combination: the strictness is advertised, and the
        way around it is to save the record once.
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

        This is the laundering case, and the only one where provenance is doing
        work no current evidence could: these lines carry correct checksum digits,
        so every check passes on what the record holds *now*. What it says is that
        nothing ever verified the digits its source omitted, which saving and
        re-reading cannot change.
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
        forbid_every_orbit_source(monkeypatch, tmp_path)

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

        An absent status could mean either "verified" or "written before the
        field existed", which is exactly the ambiguity a later reader must not
        resolve in favour of trust.
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

        Checksum-less records are deliberately kept out of the shared cache, so
        the saved run records are the *only* way such a run can be reproduced
        offline. Replaying them strictly has to fail, and has to say which
        setting would allow it.
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
        forbid_every_orbit_source(monkeypatch, tmp_path)

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
        validate, and tabascal's strict default must not be undermined by
        whatever tabsim was allowed to accept. So the record is used and saved
        with the run, and the cache is left as it was.
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

        The archive's stray backslash is accepted because the checksum still
        verifies once it is stripped — but if only the validator strips it, the
        defective line is what gets propagated, written to the output schema and
        saved for replay.
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


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReplay:
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

        # A later run with no network and no cache must reproduce the same
        # trajectory from the file alone. The seams *raise* rather than return
        # nothing: a replay that quietly asked the service would still produce a
        # plausible simulation, and the test would still pass.
        monkeypatch.setattr(client, "fetch_nearest_tle", forbidden("the TLE endpoint"))
        monkeypatch.setattr(client, "fetch_nearest_omm", forbidden("the OMM endpoint"))
        monkeypatch.setenv("ORBIT_CACHE_DIR", str(tmp_path / "empty-cache"))

        second = orbit.resolve_orbits(
            [ISS_NORAD_ID], ISS_EPOCH_JD, extra_orbit_dir=str(replay_dir)
        )
        assert second.complete

        # Exactly, not approximately: a replay that only nearly reproduces the
        # run is a claim that quietly stops being true. For OMM this depends on
        # the elements being written at full float64 precision.
        times_jd = ISS_EPOCH_JD + np.linspace(0.0, 0.2, 5)
        before = get_satellite_positions(first.records(), times_jd)
        after = get_satellite_positions(second.records(), times_jd)
        np.testing.assert_array_equal(before, after)

    def test_derived_columns_are_not_written(self, tmp_path):
        """A stored EPOCH_JD could only drift out of step with its elements."""
        path = tmp_path / "used_orbits.json"
        orbit.save_orbits_for_reuse(
            path, [ISS_NORAD_ID], [orbit._finalise_records([tle_record()]).iloc[0].to_dict()]
        )
        written = json.loads(path.read_text())
        assert "EPOCH_JD" not in written
        assert "SEMIMAJOR_AXIS" not in written
        assert read_legacy_tle_records(tmp_path).shape[0] == 1

    def test_omm_elements_survive_the_file_bit_for_bit(self, tmp_path):
        """The written float must read back as the *same* double, not a near one.

        Either half can break it. On the way out, ``DataFrame.to_json`` formats
        to a fixed number of decimal places: its default 10 rounds this
        eccentricity away, and its maximum 15 writes it as 0.006663499999999999,
        a different double. On the way in, satchecker-client before 0.1.2 parsed
        with pandas' imprecise float parser, which reads the eccentricity back as
        that same wrong double and a BSTAR of 3.2e-05 as 3.2000000000000005e-05.
        Either makes a replayed trajectory quietly disagree with the run it
        reproduces.
        """
        awkward = omm_record_from_tle()
        awkward["ECCENTRICITY"] = 0.0066635
        awkward["BSTAR"] = 3.2e-05

        orbit.save_orbits_for_reuse(
            tmp_path / "used_orbits.json", [ISS_NORAD_ID], [awkward]
        )
        back = read_legacy_tle_records(tmp_path).iloc[0]

        assert back["ECCENTRICITY"] == awkward["ECCENTRICITY"]
        assert back["BSTAR"] == awkward["BSTAR"]

    def test_mixed_kinds_in_one_file_stay_valid_json(self, tmp_path):
        """A TLE row has no MEAN_MOTION; that must be null, not a bare NaN.

        ``json.loads`` accepts the non-standard ``NaN`` literal by default, so the
        parse alone proves nothing: ``parse_constant`` is what makes the read
        refuse a file only Python's own parser would accept.
        """
        path = tmp_path / "used_orbits.json"
        orbit.save_orbits_for_reuse(
            path,
            [ISS_NORAD_ID, GPS_NORAD_ID],
            [
                omm_record_from_tle(),
                tle_record(
                    norad_id=GPS_NORAD_ID, line1=GPS_LINE1, line2=GPS_LINE2
                ),
            ],
        )
        payload = json.loads(path.read_text(), parse_constant=reject_json_constant)
        assert payload["MEAN_MOTION"]["1"] is None
        assert payload["TLE_LINE1"]["0"] is None
        assert len(read_legacy_tle_records(tmp_path)) == 2

    def test_empty_selection_is_saved_explicitly(self, tmp_path):
        """A completed run with no satellites still needs its replay artifacts.

        Writing nothing makes "no satellite passed the target" and "this
        directory is not a replay" the same state on disk, so a frozen replay of
        a legitimately satellite-free run cannot be told from a missing file.
        """
        path = tmp_path / "used_orbits.json"

        assert orbit.save_orbits_for_reuse(path, [], []) == str(path)
        assert path.exists()
        json.loads(path.read_text())  # an explicitly empty table, still valid JSON

    def test_save_rejects_misaligned_ids_and_records(self, tmp_path):
        """Misaligned inputs are a bug, and ``zip`` hides it.

        Truncating to the shorter sequence writes a file that reads back cleanly
        and describes different satellites than the run propagated.
        """
        path = tmp_path / "used_orbits.json"
        gps = tle_record(norad_id=GPS_NORAD_ID, line1=GPS_LINE1, line2=GPS_LINE2)

        with pytest.raises(ValueError):
            orbit.save_orbits_for_reuse(
                path, [ISS_NORAD_ID, GPS_NORAD_ID], [tle_record()]
            )
        with pytest.raises(ValueError):
            orbit.save_orbits_for_reuse(
                path, [ISS_NORAD_ID], [tle_record(), gps]
            )
        # An ID that does not match the record filed against it, too.
        with pytest.raises(ValueError):
            orbit.save_orbits_for_reuse(path, [GPS_NORAD_ID], [tle_record()])

    def test_save_rejects_a_lossy_identity_match(self, tmp_path):
        """25544.5 is not 25544, and truncating it saves a different satellite.

        Both identity checks cast with ``int()``, so a record whose own
        ``NORAD_CAT_ID`` disagreed with the ID it was filed against passed the
        alignment check and produced a valid-looking replay file.
        """
        record = tle_record()  # valid ISS lines...
        record["NORAD_CAT_ID"] = 25544.5  # ...under an identity that is not an ID

        with pytest.raises(ValueError, match="25544.5"):
            orbit.save_orbits_for_reuse(
                tmp_path / "used_orbits.json", [ISS_NORAD_ID], [record]
            )

    @pytest.mark.parametrize("value", [float("nan"), None], ids=["nan", "absent"])
    def test_save_rejects_an_omm_missing_a_required_element(self, tmp_path, value):
        """A dropped element writes a file that cannot be replayed at all.

        The projection skipped every null cell, which is right for the OMM columns
        a TLE row acquires in a mixed frame and wrong for an OMM's own elements:
        the record went out without its mean motion and the run it exists to
        reproduce no longer could be.
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

        ``extra_orbit_dir`` never froze anything: it only changed where a record
        came from, so the original names, ID list, visibility cuts and
        ``max_n_sat`` still chose the satellites. A replay has to be able to
        reproduce a run whose name search no longer returns the same catalogue.
        """
        replay_dir = write_replay_dir(
            tmp_path / "input_data", [ISS_NORAD_ID], [tle_record()]
        )
        forbid_every_orbit_source(monkeypatch, tmp_path)

        ids, records = orbit.load_replay_orbits(str(replay_dir))

        assert ids == [ISS_NORAD_ID]
        assert records[0]["TLE_LINE1"] == ISS_LINE1

    @pytest.mark.parametrize(
        "damage,expected",
        [
            ("missing-records-file", "used_orbits.json"),
            ("missing-id-file", "norad_ids.yaml"),
            ("corrupt-json", "used_orbits.json"),
            ("missing-id", "99999"),
            ("extra-id", str(GPS_NORAD_ID)),
            ("duplicate-id-line", "more than once"),
            ("duplicate-record-rows", "holds 2 records"),
            ("wrong-embedded-id", "not acceptable under this run's policy"),
        ],
    )
    def test_replay_requires_exact_saved_records(
        self, monkeypatch, tmp_path, damage, expected
    ):
        """Replay has no second source, so anything short of exact must stop.

        Every alternative — skipping a record, taking the first of two, asking
        the cache — silently changes the orbital inputs of a run whose whole
        purpose is to keep them fixed. Each case is pinned to the rejection it is
        about: with two rows for one satellite, and with two ID lines for one, both
        sharing a single "duplicate" fixture, whichever check ran first answered
        for both.
        """
        gps = tle_record(norad_id=GPS_NORAD_ID, line1=GPS_LINE1, line2=GPS_LINE2)
        replay_dir = write_replay_dir(
            tmp_path / "input_data", [ISS_NORAD_ID, GPS_NORAD_ID], [tle_record(), gps]
        )
        records_path = replay_dir / "used_orbits.json"
        ids_path = replay_dir / "norad_ids.yaml"

        if damage == "missing-records-file":
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

        forbid_every_orbit_source(monkeypatch, tmp_path)

        with pytest.raises(orbit.OrbitError) as excinfo:
            orbit.load_replay_orbits(str(replay_dir))

        assert expected in str(excinfo.value)

    def test_empty_replay_is_explicit_and_network_free(self, monkeypatch, tmp_path):
        """Replaying a satellite-free run returns zero satellites, deliberately."""
        replay_dir = write_replay_dir(tmp_path / "input_data", [], [])
        assert (replay_dir / "used_orbits.json").exists()
        assert (replay_dir / "norad_ids.yaml").exists()
        forbid_every_orbit_source(monkeypatch, tmp_path)

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
        forbid_every_orbit_source(monkeypatch, tmp_path)

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


# ---------------------------------------------------------------------------
# Replay at the simulation-selection level
# ---------------------------------------------------------------------------

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

        The visibility grid steps ``vis_step`` from the first sample to *past* the
        last, so an observation ending at 23:59:50 with a one-minute step has a
        grid of 23:59:40 and next-day 00:00:40 — a mean on the following date. A
        satellite the catalogue says decayed on the observation's own date was
        excluded from a simulation it belongs in, and the record request and every
        age comparison used the shifted epoch too.
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

        A ``max_n_sat`` or a visibility cut changed since the original run would
        otherwise silently drop satellites from a replay that exists precisely to
        keep them.
        """
        from tabsim import config as config_module

        gps = tle_record(norad_id=GPS_NORAD_ID, line1=GPS_LINE1, line2=GPS_LINE2)
        replay_dir = write_replay_dir(
            tmp_path / "input_data", [ISS_NORAD_ID, GPS_NORAD_ID], [tle_record(), gps]
        )
        forbid_every_orbit_source(monkeypatch, tmp_path)
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
        forbid_every_orbit_source(monkeypatch, tmp_path)

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
        forbid_every_orbit_source(monkeypatch, tmp_path)
        obs = FakeObservation()

        config_module.add_tle_satellite_sources(
            obs,
            replay_sim_config(replay_dir, sat_names=["navstar"], norad_ids=[7001]),
        )

        # The run's own names and ID list were overridden, not merged in.
        assert sorted(nid for ids, _ in obs.added for nid in ids) == [ISS_NORAD_ID]
        out = capsys.readouterr().out
        assert "Frozen orbit replay" in out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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

    @pytest.mark.parametrize("value", [[1.5], [0], [-3], ["abc"], [None], "25544"])
    def test_bad_norad_ids_are_rejected_before_the_resolver(self, value):
        with pytest.raises(TLEConfigurationError):
            normalise_norad_ids(value)

    def test_norad_ids_are_deduplicated_in_order(self):
        assert normalise_norad_ids([3, 1, 3, 2, "1"]) == [3, 1, 2]

    def test_float_ids_from_numpy_are_accepted(self):
        """Config lists routinely arrive as floats; only fractional ones are wrong."""
        assert normalise_norad_ids(np.array([25544.0, 32260.0])) == [25544, 32260]

    def test_norad_ids_file_keeps_the_first_column(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("# satellites\n25544 ISS\n\n32260  GPS\n25544\n")
        assert read_norad_ids_file(path) == [25544, 32260]

    def test_norad_ids_file_error_names_the_line(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("25544\nnot-an-id\n")
        with pytest.raises(TLEConfigurationError, match=r"ids\.txt:2"):
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


# ---------------------------------------------------------------------------
# Frame contract
# ---------------------------------------------------------------------------

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
