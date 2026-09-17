"""The command-line surface of the orbit policy, and the credential-free promise.

Two rules run through all of it. A boolean flag defaults to ``None`` rather than
``False``, so omitting it leaves the YAML choice alone instead of overwriting a
deliberate ``offline: true``. And a path typed on the command line is relative to
where it was typed, while one written in a config is relative to the config.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tabsim import orbit
from tabsim.scripts import sim_vis

from orbit_helpers import (
    ISS_EPOCH_JD,
    ISS_NORAD_ID,
    search_row,
    serve_search,
    stub_service,
    tle_record,
    write_sim_config,
)


class Reached(Exception):
    """Raised by a stub to prove control got as far as it."""


def run_sim_vis_config(monkeypatch, config_path, *args, cwd=None):
    """Run ``sim-vis`` up to the point it would start simulating; return its config."""
    captured = {}

    def capture(sim_config=None, config_path=None):
        captured["sim_config"] = sim_config
        return None, "output-path"

    monkeypatch.setattr(sim_vis, "run_sim_config", capture)
    if cwd is not None:
        monkeypatch.chdir(cwd)
    argv = ["sim-vis", "--config", str(config_path), *args]
    with patch.object(sys, "argv", argv):
        sim_vis.main()
    return captured["sim_config"]


def test_the_console_entry_point_exits_zero_on_success(monkeypatch, tmp_path):
    """``sys.exit(main())`` reads a returned tuple as failure.

    Any return that is not ``None`` or an int becomes exit status 1, so a
    successful run reported failure to every shell that checked. The entry point
    is ``cli``, which discards the result ``main`` returns for Python callers.
    """
    config_path = write_sim_config(tmp_path / "sim.yaml", {"sat_names": ["navstar"]})
    run_sim_vis_config(monkeypatch, config_path)  # installs the capture stub
    argv = ["sim-vis", "--config", str(config_path)]
    with patch.object(sys, "argv", argv):
        assert sim_vis.main() == (None, "output-path")
        assert sim_vis.cli() is None
    # The checkout's pyproject, not the installed package's neighbour: under a
    # non-editable install (CI) there is no pyproject.toml beside the module.
    text = Path(__file__).parents[1].joinpath("pyproject.toml").read_text()
    assert 'sim-vis = "tabsim.scripts.sim_vis:cli"' in text


def run_sim_vis(monkeypatch, config_path, *args, cwd=None):
    """...and just the satellite section of it, which most of these tests want."""
    config = run_sim_vis_config(monkeypatch, config_path, *args, cwd=cwd)
    return config["rfi_sources"]["tle_satellite"]


def import_tle_sat_region(monkeypatch):
    """Import ``tle_sat_region`` without the optional ``regions`` dependency.

    It only serialises the output file, long after the orbit policy this test is
    about has been forwarded; skipping would leave that unchecked on most machines.
    """
    import importlib
    import types

    try:  # pragma: no cover - depends on the environment's extras
        import regions  # noqa: F401
    except ImportError:
        stub = types.ModuleType("regions")
        for name in ("CircleSkyRegion", "RectangleSkyRegion", "TextSkyRegion"):
            setattr(stub, name, object)
        monkeypatch.setitem(sys.modules, "regions", stub)
    return importlib.import_module("tabsim.scripts.tle_sat_region")


@pytest.fixture
def layout(tmp_path):
    """A config directory and a separate working directory, as a user would have."""
    conf = tmp_path / "conf"
    work = tmp_path / "work"
    conf.mkdir()
    work.mkdir()
    return conf, work


class TestPolicyFlags:
    @pytest.mark.parametrize(
        "written,args,expected",
        [
            (
                {"offline": True, "allow_missing_checksum": True},
                (),
                {"offline": True, "allow_missing_checksum": True},
            ),
            (
                {"offline": False, "allow_missing_checksum": True},
                ("--offline", "--no-allow-missing-checksum"),
                {"offline": True, "allow_missing_checksum": False},
            ),
            (
                {"allow_missing_checksum": False},
                ("--allow-missing-checksum",),
                {"allow_missing_checksum": True},
            ),
        ],
        ids=["omitted", "overridden", "switched-on"],
    )
    def test_a_boolean_flag_only_speaks_when_it_is_typed(
        self, layout, monkeypatch, written, args, expected
    ):
        """A flag that was not typed is not an instruction to turn anything off."""
        conf, work = layout
        config_path = write_sim_config(
            conf / "sim.yaml",
            tle_satellite={**written, "extra_orbit_dir": "yaml_extra"},
        )

        satellites = run_sim_vis(monkeypatch, config_path, *args, cwd=work)

        for key, value in expected.items():
            assert satellites[key] is value, key
        # A config path is relative to the config, as every other path here is.
        assert satellites["extra_orbit_dir"] == str(conf / "yaml_extra")

    @pytest.mark.parametrize(
        "flag", ["--extra-orbit-dir", "--extra_orbit_dir", "-eod"]
    )
    def test_extra_orbit_dir_keeps_its_old_spellings(
        self, layout, monkeypatch, flag
    ):
        """The canonical dashed name is new; the two older ones still have users."""
        conf, work = layout
        config_path = write_sim_config(conf / "sim.yaml")

        satellites = run_sim_vis(monkeypatch, config_path, flag, "mine", cwd=work)

        # Typed on the command line, so resolved against the working directory.
        assert satellites["extra_orbit_dir"] == str(work / "mine")

    def test_replay_orbit_dir_resolves_from_both_places(self, layout, monkeypatch):
        conf, work = layout
        from_yaml = write_sim_config(
            conf / "yaml.yaml", tle_satellite={"replay_orbit_dir": "yaml_replay"}
        )
        assert (
            run_sim_vis(monkeypatch, from_yaml, cwd=work)["replay_orbit_dir"]
            == str(conf / "yaml_replay")
        )

        plain = write_sim_config(conf / "plain.yaml")
        satellites = run_sim_vis(
            monkeypatch, plain, "--replay-orbit-dir", "previous/input_data", cwd=work
        )
        assert satellites["replay_orbit_dir"] == str(work / "previous" / "input_data")

    def test_replay_does_not_process_the_id_file_it_overrides(
        self, layout, monkeypatch
    ):
        """A replay must not even look at the ID-file setting it overrides.

        ``norad_ids_path`` went through ``get_abs_path``, which raises on a value
        that is not a path — so a leftover the run never reads stopped it.
        """
        conf, work = layout
        config_path = write_sim_config(
            conf / "sim.yaml",
            tle_satellite={
                "replay_orbit_dir": "previous/input_data",
                "norad_ids_path": 123,
            },
        )

        satellites = run_sim_vis(monkeypatch, config_path, cwd=work)

        assert satellites["norad_ids_path"] == 123, "left exactly as written"
        assert satellites["replay_orbit_dir"] == str(conf / "previous" / "input_data")

    def test_an_id_file_is_still_resolved_without_a_replay(self, layout, monkeypatch):
        """...and an ordinary run still resolves it against the config directory."""
        conf, work = layout
        config_path = write_sim_config(
            conf / "sim.yaml", tle_satellite={"norad_ids_path": "ids.txt"}
        )

        satellites = run_sim_vis(monkeypatch, config_path, cwd=work)

        assert satellites["norad_ids_path"] == str(conf / "ids.txt")

    @pytest.mark.parametrize(
        "args,expected",
        [((), True), (("-o",), True), (("--no-overwrite",), False)],
        ids=["omitted", "on", "off"],
    )
    def test_overwrite_obeys_the_same_omission_rule(
        self, layout, monkeypatch, args, expected
    ):
        """``-o`` is a boolean flag like the new ones, and omitting it says nothing.

        Defaulting to ``False`` and assigning unconditionally turned
        ``output.overwrite: true`` back off on every command line without ``-o``.
        """
        conf, work = layout
        config_path = write_sim_config(conf / "sim.yaml", output={"overwrite": True})

        sim_config = run_sim_vis_config(monkeypatch, config_path, *args, cwd=work)

        assert sim_config["output"]["overwrite"] is expected


class TestHelpAndMigration:
    def test_cli_help_lists_the_new_flags(self, capsys):
        with patch.object(sys, "argv", ["sim-vis", "--help"]):
            with pytest.raises(SystemExit) as excinfo:
                sim_vis.main()
        assert excinfo.value.code == 0

        # Normalised, because argparse wraps the help text but never inside a word.
        help_text = " ".join(capsys.readouterr().out.split())
        for flag in (
            "--offline",
            "--replay-orbit-dir",
            "--allow-missing-checksum",
            "--no-allow-missing-checksum",
            "--extra-orbit-dir",
            "-eod",
        ):
            assert flag in help_text, flag

    @pytest.mark.parametrize("flag", ["-st", "-td"])
    def test_removed_space_track_flags_fail_loudly(self, tmp_path, flag):
        """Silently ignoring them would leave a run looking configured when it is not."""
        config_path = write_sim_config(tmp_path / "sim.yaml")
        argv = ["sim-vis", "--config", config_path, flag, "whatever"]

        with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as excinfo:
            sim_vis.main()

        assert excinfo.value.code != 0

    def test_tle_region_forwards_the_orbit_policy(self, tmp_path, monkeypatch):
        """``tle-region`` resolves records too, so the policy is forwarded rather
        than re-defaulted: a region file drawn from records the simulation would
        have refused is a region file for a different run.
        """
        tle_sat_region = import_tle_sat_region(monkeypatch)

        class Column:
            def __init__(self, values):
                self.data = self
                self._values = values

            def compute(self):
                return self._values

        class FakeXds:
            def __init__(self, times_mjd):
                self.TIME = Column(times_mjd)

        times_mjd = np.array(
            [ISS_EPOCH_JD - 2400000.5, ISS_EPOCH_JD - 2400000.5 + 1e-4]
        )
        monkeypatch.setattr(
            tle_sat_region, "xds_from_ms", lambda path: [FakeXds(times_mjd)]
        )
        captured = {}

        def capture(norad_ids, epoch_jd, **policy):
            captured["norad_ids"] = list(norad_ids)
            captured["epoch_jd"] = epoch_jd
            captured.update(policy)
            raise Reached()

        monkeypatch.setattr(tle_sat_region, "get_tles_by_id", capture)
        argv = [
            "tle-region",
            "-m",
            str(tmp_path / "fake.ms"),
            "-ni",
            str(ISS_NORAD_ID),
            "--extra-orbit-dir",
            str(tmp_path / "extra"),
            "--offline",
            "--allow-missing-checksum",
        ]

        with patch.object(sys, "argv", argv), pytest.raises(Reached):
            tle_sat_region.main()

        assert captured["norad_ids"] == [ISS_NORAD_ID]
        assert captured["offline"] is True
        assert captured["allow_missing_checksum"] is True
        assert captured["extra_orbit_dir"] == str(tmp_path / "extra")
        assert captured["epoch_jd"] == pytest.approx(ISS_EPOCH_JD, abs=1e-3)


def forbid_spacetrack_imports(monkeypatch):
    """Fail the test if anything under it imports ``spacetrack``."""
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if str(name).split(".")[0] == "spacetrack":
            raise AssertionError("tabsim must not import spacetrack")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for variable in (
        "SPACETRACK_USER",
        "SPACETRACK_USERNAME",
        "SPACETRACK_PASSWORD",
        "SPACETRACK_LOGIN",
    ):
        monkeypatch.delenv(variable, raising=False)


class TestCredentialFreeRuntime:
    def test_no_space_track_requirement_or_setup_command(self):
        """The dependency and its credential-setup entry point are both gone.

        A guard: losing it means a dependency CI cannot install and an unattended
        run that asks for a password.
        """
        pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()

        assert "spacetrack" not in pyproject.lower()
        assert "setup-spacetrack-login" not in pyproject
        assert "setup_spacetrack_login" not in pyproject

    def test_numbered_acquisition_needs_no_credentials(self, monkeypatch):
        """Resolving records by number touches no credential store. (A guard.)"""
        forbid_spacetrack_imports(monkeypatch)
        stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})

        frame = orbit.get_orbits_by_id([ISS_NORAD_ID], ISS_EPOCH_JD)

        assert list(frame["NORAD_CAT_ID"]) == [ISS_NORAD_ID]

    def test_named_discovery_needs_no_credentials(self, monkeypatch):
        """...and so does the name search, which used to be the credentialed half."""
        forbid_spacetrack_imports(monkeypatch)
        serve_search(monkeypatch, {"ISS": [search_row(ISS_NORAD_ID, "ISS (ZARYA)")]})
        stub_service(monkeypatch, {ISS_NORAD_ID: tle_record()})

        norad_ids = orbit.resolve_names(["iss"], ISS_EPOCH_JD, log=lambda *_: None)
        frame = orbit.get_orbits_by_id(norad_ids, ISS_EPOCH_JD)

        assert list(frame["NORAD_CAT_ID"]) == [ISS_NORAD_ID]
