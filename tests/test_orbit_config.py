"""Orbit configuration: the new policy keys, and the ones that are now obsolete.

A key that used to change which satellites a run modelled and now does nothing is
worse than a removed one: the run succeeds and quietly stops honouring a setting the
user still believes in. So ``tle_dir`` and ``spacetrack_path`` are rejected by
*presence*, before the observation is built and before any request goes out.
"""

from __future__ import annotations

import sys

import pytest

from tabsim import config as config_module
from tabsim import orbit_config as orbit_config_module
from tabsim.orbit_config import TLEConfigurationError, normalise_orbit_config

from orbit_helpers import (
    ISS_LINE1,
    ISS_LINE2,
    ISS_NORAD_ID,
    forbidden,
    restored_stdout,
    write_replay_dir,
    write_sim_config,
)


#: The exact migration text each obsolete key must produce. Compared with
#: backticks stripped from both sides, so the wording is pinned without pinning
#: whether the message quotes its key names.
MIGRATION_TEXT = {
    "tle_dir": (
        "rfi_sources.tle_satellite.tle_dir is obsolete. Use extra_orbit_dir for "
        "existing orbit JSON files; use ORBIT_CACHE_DIR to relocate the managed "
        "cache."
    ),
    "spacetrack_path": (
        "spacetrack_path is obsolete. SatChecker requires no credentials; remove "
        "this key."
    ),
}


def plain(text: str) -> str:
    return str(text).replace("`", "")


class Reached(Exception):
    """Raised by a stub to prove control got as far as it."""


class TestObsoleteKeys:
    @pytest.mark.parametrize("key", sorted(MIGRATION_TEXT))
    @pytest.mark.parametrize("value", [None, "some/path"], ids=["null", "path"])
    def test_obsolete_orbit_keys_have_migration_errors(self, key, value):
        """Present is enough: a null value is still a key someone means something by."""
        with pytest.raises(TLEConfigurationError) as excinfo:
            normalise_orbit_config({key: value, "norad_ids": [ISS_NORAD_ID]})

        assert plain(MIGRATION_TEXT[key]) in plain(str(excinfo.value))

    @pytest.mark.parametrize("key", sorted(MIGRATION_TEXT))
    def test_obsolete_keys_stop_startup_before_the_observation(
        self, key, tmp_path, monkeypatch
    ):
        """...and they stop the run before it builds anything or asks anything.

        ``max_n_sat: 0`` disables satellite simulation entirely, the one
        configuration where nothing downstream would ever read the section.
        """
        config_path = write_sim_config(
            tmp_path / "sim.yaml",
            tle_satellite={key: "some/path", "max_n_sat": 0},
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            config_module, "load_obs", forbidden("observation construction")
        )

        with restored_stdout(), pytest.raises(TLEConfigurationError) as excinfo:
            config_module.run_sim_config(config_path=config_path)

        assert plain(MIGRATION_TEXT[key]) in plain(str(excinfo.value))

    def test_a_failed_run_restores_stdout(self, tmp_path, monkeypatch):
        """A fatal error must not leave the process writing into that run's log.

        ``run_sim_config`` tees ``sys.stdout`` into ``log_sim_*.txt`` and the checks
        can raise first. Deliberately *without* ``restored_stdout``: this is the
        guard that it is not needed.
        """
        config_path = write_sim_config(
            tmp_path / "sim.yaml", tle_satellite={"tle_dir": "gone"}
        )
        monkeypatch.chdir(tmp_path)
        before = sys.stdout

        with pytest.raises(TLEConfigurationError):
            config_module.run_sim_config(config_path=config_path)

        assert sys.stdout is before


class TestNewSettings:
    def test_new_config_defaults_and_validation(self):
        """Strict by default, and every value validated where it is written."""
        config = normalise_orbit_config({})

        assert config.allow_missing_checksum is False
        assert config.search_cache_max_age_days == 1.0
        assert config.offline is False
        assert config.replay_orbit_dir is None

    @pytest.mark.parametrize("key", ["allow_missing_checksum", "offline"])
    @pytest.mark.parametrize("value", ["yes", "false", 1, 0, 1.0, []])
    def test_non_boolean_policy_values_are_rejected(self, key, value):
        """``allow_missing_checksum: 0`` must not mean "on" by truthiness.

        These decide whether unverifiable orbital data is accepted and whether the
        service is contacted, so a near-miss value must be an error.
        """
        with pytest.raises(TLEConfigurationError, match=key):
            normalise_orbit_config({key: value})

    @pytest.mark.parametrize("value", [-1, "soon", float("nan"), float("inf"), True])
    def test_bad_search_freshness_is_a_configuration_error(self, value):
        with pytest.raises(TLEConfigurationError, match="search_cache_max_age_days"):
            normalise_orbit_config({"search_cache_max_age_days": value})

    def test_search_freshness_extremes_are_legitimate(self):
        """``null`` reuses a snapshot indefinitely; ``0`` refreshes every lookup."""
        assert normalise_orbit_config(
            {"search_cache_max_age_days": None}
        ).search_cache_max_age_days is None
        assert normalise_orbit_config(
            {"search_cache_max_age_days": 0}
        ).search_cache_max_age_days == 0.0

    def test_existing_age_cross_constraint_still_applies(self):
        """The new keys must not have displaced the constraint between the old ones."""
        with pytest.raises(TLEConfigurationError, match="must not exceed"):
            normalise_orbit_config(
                {
                    "remote_max_age_days": 1,
                    "cache_reuse_max_age_days": 5,
                    "offline": True,
                    "allow_missing_checksum": True,
                    "search_cache_max_age_days": 2,
                }
            )


class TestReplaySelection:
    @pytest.mark.parametrize(
        "ignored",
        [
            {
                "norad_ids_path": "gone.txt",
                "norad_ids": [ISS_NORAD_ID],
                "sat_names": ["navstar"],
            },
            {"norad_ids": ["bad"]},
            {"norad_ids": [1.5]},
            {"sat_names": "NAVSTAR"},
        ],
        ids=["missing-id-file", "non-numeric-id", "fractional-id", "sat-names-str"],
    )
    def test_replay_neither_reads_nor_validates_what_it_overrides(
        self, tmp_path, monkeypatch, ignored
    ):
        """A value a replay never reads cannot be a reason to refuse the run.

        The saved IDs are authoritative, so an ID file that has since moved and a
        malformed leftover in the config must both stop mattering.
        """
        replay_dir = tmp_path / "input_data"
        replay_dir.mkdir()
        monkeypatch.setattr(
            orbit_config_module,
            "read_norad_ids_file",
            forbidden("the original NORAD ID file"),
        )

        config = normalise_orbit_config(
            {"replay_orbit_dir": str(replay_dir), **ignored}
        )

        assert config.replay_orbit_dir == str(replay_dir)
        # ...and they are not the selection either, so they come back empty.
        assert config.norad_ids == []
        assert config.sat_names == []

    def test_replay_and_extra_orbit_dir_are_rejected_together(self, tmp_path):
        """Their source-selection contracts differ, so the pair has no meaning.

        ``extra_orbit_dir`` is per-ID precedence within the run's own selection;
        ``replay_orbit_dir`` replaces that selection.
        """
        with pytest.raises(TLEConfigurationError, match="replay_orbit_dir"):
            normalise_orbit_config(
                {
                    "replay_orbit_dir": str(tmp_path / "replay"),
                    "extra_orbit_dir": str(tmp_path / "extra"),
                }
            )

    def test_max_n_sat_zero_does_not_suppress_frozen_replay(
        self, tmp_path, monkeypatch
    ):
        """A replay of a run that limited itself is still the saved selection.

        The startup guard that skips satellite handling when ``max_n_sat`` is zero
        would otherwise discard a non-empty replay.
        """
        replay_dir = write_replay_dir(
            tmp_path / "input_data",
            [ISS_NORAD_ID],
            [
                {
                    "NORAD_CAT_ID": ISS_NORAD_ID,
                    "RECORD_KIND": "tle",
                    "TLE_LINE1": ISS_LINE1,
                    "TLE_LINE2": ISS_LINE2,
                }
            ],
        )

        config_path = write_sim_config(
            tmp_path / "sim.yaml",
            tle_satellite={"replay_orbit_dir": str(replay_dir), "max_n_sat": 0},
        )
        monkeypatch.chdir(tmp_path)
        seen = []

        def record_replay(obs, sim_config):
            seen.append(
                sim_config["rfi_sources"]["tle_satellite"]["replay_orbit_dir"]
            )
            raise Reached()

        monkeypatch.setattr(
            config_module, "add_tle_satellite_sources", record_replay
        )
        # If the max_n_sat guard skipped the replay, the run would sail past it.
        monkeypatch.setattr(
            config_module,
            "add_stationary_sources",
            forbidden("the rest of the simulation"),
        )

        with restored_stdout(), pytest.raises(Reached):
            config_module.run_sim_config(config_path=config_path)

        assert seen == [str(replay_dir)]
