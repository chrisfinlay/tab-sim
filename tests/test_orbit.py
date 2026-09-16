"""Resolution policy, propagation and configuration validation for orbit records.

Everything here runs offline. The :mod:`satchecker_client` dependency is covered
by its own repository's suite; what these tests exercise is the tabsim side of
the seam — the source precedence and age policy in :mod:`tabsim.orbit`, the
configuration normalisation in :mod:`tabsim.orbit_config`, the name lookup in
:mod:`tabsim.satchecker_names`, and the propagation of both record kinds in
:mod:`tabsim.tle`.

The only network-touching seams are ``client.fetch_nearest_tle`` /
``fetch_nearest_omm`` and ``satchecker_names._http_get``; each test that needs
one patches it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tabsim import orbit
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
    GPS_LINE1,
    GPS_LINE2,
    GPS_NORAD_ID,
    ISS_EPOCH_JD,
    ISS_LINE1,
    ISS_LINE2,
    ISS_NORAD_ID,
    omm_record_from_tle,
    tle_record,
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
def no_network(monkeypatch):
    """Fail loudly if a test reaches the network without saying it means to."""

    def forbidden(*args, **kwargs):
        raise AssertionError("test made an unexpected SatChecker request")

    monkeypatch.setattr(client, "_http_get", forbidden)


def stub_service(monkeypatch, records_by_id, endpoint="tle"):
    """Serve *records_by_id* from the nearest-record endpoint of the given kind."""
    calls = []

    def fetch(norad_id, epoch_jd):
        calls.append((int(norad_id), float(epoch_jd)))
        record = records_by_id.get(int(norad_id))
        if record is None:
            return pd.DataFrame()
        return pd.DataFrame([record])

    def empty(norad_id, epoch_jd):
        return pd.DataFrame()

    monkeypatch.setattr(
        client, "fetch_nearest_tle", fetch if endpoint == "tle" else empty
    )
    monkeypatch.setattr(
        client, "fetch_nearest_omm", fetch if endpoint == "omm" else empty
    )
    return calls


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

    def test_service_response_is_cached_for_later_runs(
        self, monkeypatch, isolated_cache
    ):
        stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})
        orbit.resolve_orbits([ISS_NORAD_ID], ISS_EPOCH_JD)

        cached = TextOrbitCache(isolated_cache).get(ISS_NORAD_ID)
        assert len(cached) == 1
        assert cached["TLE_LINE1"].iloc[0] == ISS_LINE1

    def test_endpoint_failover_when_the_first_archive_is_too_old(self, monkeypatch):
        """An over-age answer is the signal the record lives in the other archive.

        Neither endpoint reports "I have nothing that near", so this is the only
        way the epoch's archive can be got wrong and recovered from.
        """
        epoch_jd = ISS_EPOCH_JD  # pre-handover: nearest-TLE is tried first
        omm = omm_record_from_tle()
        stale_tle = tle_record(line1=GPS_LINE1, line2=GPS_LINE2, norad_id=GPS_NORAD_ID)

        def nearest_tle(norad_id, _epoch_jd):
            return pd.DataFrame([stale_tle])  # wrong satellite -> rejected

        def nearest_omm(norad_id, _epoch_jd):
            return pd.DataFrame([omm])

        monkeypatch.setattr(client, "fetch_nearest_tle", nearest_tle)
        monkeypatch.setattr(client, "fetch_nearest_omm", nearest_omm)

        resolution = orbit.resolve_orbits([ISS_NORAD_ID], epoch_jd)

        assert resolution.complete
        assert "nearest-OMM" in resolution.resolved[ISS_NORAD_ID].source

    def test_outage_stops_the_batch_rather_than_trying_the_other_archive(
        self, monkeypatch
    ):
        """A service that cannot serve us is not asked a different question."""
        omm_calls = []

        def unreachable(norad_id, _epoch_jd):
            raise client.SatCheckerTransportError("connection refused")

        def nearest_omm(norad_id, _epoch_jd):
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
        def unreachable(norad_id, _epoch_jd):
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
        # trajectory from the file alone.
        monkeypatch.setattr(client, "fetch_nearest_tle", lambda *a: pd.DataFrame())
        monkeypatch.setattr(client, "fetch_nearest_omm", lambda *a: pd.DataFrame())
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
        """A TLE row has no MEAN_MOTION; that must be null, not a bare NaN."""
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
        payload = json.loads(path.read_text())  # bare NaN would raise here
        assert payload["MEAN_MOTION"]["1"] is None
        assert payload["TLE_LINE1"]["0"] is None
        assert len(read_legacy_tle_records(tmp_path)) == 2

    def test_nothing_to_save_writes_nothing(self, tmp_path):
        assert orbit.save_orbits_for_reuse(tmp_path / "none.json", [], []) is None
        assert not (tmp_path / "none.json").exists()


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
# Name lookup
# ---------------------------------------------------------------------------

class TestNameLookup:
    def _serve(self, monkeypatch, payload):
        from tabsim import satchecker_names

        seen = {}

        def fake_get(url, timeout=None):
            seen["url"] = url
            return json.dumps(payload).encode()

        monkeypatch.setattr(satchecker_names, "_http_get", fake_get)
        return seen

    def test_substring_matches_resolve_to_ids(self, monkeypatch):
        from tabsim.satchecker_names import search_satellites_by_name

        seen = self._serve(
            monkeypatch,
            {
                "count": 2,
                "data": [
                    {"satellite_id": 24876, "satellite_name": "NAVSTAR 43 (USA 132)",
                     "decay_date": None},
                    {"satellite_id": 28129, "satellite_name": "NAVSTAR 53 (USA 175)",
                     "decay_date": None},
                ],
            },
        )

        assert search_satellites_by_name("navstar") == [
            (24876, "NAVSTAR 43 (USA 132)"),
            (28129, "NAVSTAR 53 (USA 175)"),
        ]
        assert "search-satellites" in seen["url"]
        # Upper-cased: the endpoint is a case-sensitive substring match against an
        # upper-case catalogue, so a lower-case name would silently find nothing.
        assert "name=NAVSTAR" in seen["url"]

    def test_decayed_satellites_are_dropped(self, monkeypatch):
        """A re-entered namesake has no current record and would fail coverage."""
        from tabsim.satchecker_names import search_satellites_by_name

        self._serve(
            monkeypatch,
            {
                "count": 2,
                "data": [
                    {"satellite_id": 1, "satellite_name": "OLD", "decay_date": "1999-01-01"},
                    {"satellite_id": 2, "satellite_name": "NEW", "decay_date": None},
                ],
            },
        )

        assert search_satellites_by_name("thing") == [(2, "NEW")]

    def test_unmatched_name_is_reported_not_raised(self, monkeypatch):
        from tabsim.satchecker_names import norad_ids_from_names

        self._serve(monkeypatch, {"count": 0, "data": []})

        ids, unmatched = norad_ids_from_names(["nosuchsat"], log=lambda *_: None)
        assert ids == []
        assert unmatched == ["nosuchsat"]

    def test_malformed_row_becomes_a_satchecker_error(self, monkeypatch):
        from tabsim.satchecker_names import search_satellites_by_name

        self._serve(monkeypatch, {"count": 1, "data": [{"satellite_name": "X"}]})

        with pytest.raises(client.SatCheckerResponseError):
            search_satellites_by_name("x")

    def test_names_never_break_a_run(self, monkeypatch):
        """Unmatched names contribute nothing; they do not stop the simulation."""
        from tabsim import satchecker_names

        monkeypatch.setattr(
            satchecker_names,
            "_http_get",
            lambda url, timeout=None: json.dumps({"count": 0, "data": []}).encode(),
        )
        assert orbit.resolve_names(["nosuchsat"], log=lambda *_: None) == []


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
