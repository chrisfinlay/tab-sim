"""End-to-end ``sim-vis`` behaviour: the live check, frozen replay, and outages.

``test_simulation_runs_with_config`` is the suite's only test that reaches the
live SatChecker service; everything else drives the same code over a two-antenna
observation with a mocked transport, fast enough to assert on the visibilities.

What a completed simulation has to prove is not that it completed: a run whose
satellites all silently failed to resolve also completes and produces valid
output, so the assertions below are about the satellites being *there*.
"""

import sys
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from satchecker_client import client
from satchecker_client.cache import TextOrbitCache, read_legacy_tle_records

from tabsim import orbit
from tabsim.config import load_config, run_sim_config
from tabsim.orbit_config import read_norad_ids_file
from tabsim.scripts import sim_vis

from orbit_helpers import (
    GPS_NORAD_ID,
    ISS_EPOCH_JD,
    ISS_NORAD_ID,
    forbidden,
    omm_record_at,
    restored_stdout,
    search_frame,
    search_row,
    serve_search,
    tle_record_at,
    write_sim_config,
)


#: Both satellites carry a spectral model in the shipped
#: ``norad_satellite.rfimodel``, which these configurations leave
#: ``norad_spec_model`` unset for, so the packaged table is what fills it in.
SIM_IDS = [ISS_NORAD_ID, GPS_NORAD_ID]

#: The packaged MeerKAT definition the telescope tests below are about. None of
#: these three values is in ``sim_config_base.yaml``, so each one is evidence
#: that the named definition was applied.
TELESCOPE_DIR = Path(str(files("tabsim.data").joinpath("telescopes")))
PACKAGED_DISH_D = 13.5
PACKAGED_ELEVATION = 1050.0


def stub_records(monkeypatch, tle=None, omm=None):
    """Serve records per endpoint, so one run can mix the two archives."""
    tle, omm = tle or {}, omm or {}

    def server(records):
        def fetch(norad_id, epoch_jd, *, strict_response=False):
            record = records.get(int(norad_id))
            if record is None:
                return pd.DataFrame()
            return pd.DataFrame([record])

        return fetch

    monkeypatch.setattr(client, "fetch_nearest_tle", server(tle))
    monkeypatch.setattr(client, "fetch_nearest_omm", server(omm))


def tiny_sim_config(path, output_path, telescope=None, **tle_satellite):
    """A two-antenna, three-sample, single-channel observation of two satellites.

    ``max_ang_sep``/``min_alt`` are wide open so the visibility filter cannot
    change the selection between a run and its replay. *telescope* overrides the
    ``MeerKAT``/two-antenna default section.
    """
    satellites = {
        "norad_ids": [],
        "sat_names": [],
        "max_ang_sep": 180,
        "min_alt": -90,
        "vis_step": 1,
        "power_scale": 1e-2,
    }
    satellites.update(tle_satellite)
    return write_sim_config(
        path,
        tle_satellite=satellites,
        telescope=dict(telescope or {}),
        observation={
            "start_time_lha": None,
            "start_time_jd": ISS_EPOCH_JD,
            "n_time": 3,
            "int_time": 2.0,
            "n_int": 2,
        },
        output={
            "zarr": True,
            "ms": False,
            "path": str(output_path),
            "prefix": "tiny",
            "overwrite": True,
        },
        # Explicit fluxes rather than "3sigma": an observation this small has a
        # theoretical image noise above 1 Jy, and the power-law draw loops forever
        # when its minimum exceeds its maximum.
        ast_sources={
            "point": {
                "random": {
                    "n_src": 2,
                    "min_I": 0.1,
                    "max_I": 1.0,
                    "random_seed": 123456,
                }
            }
        },
    )


#: Distinguishable from the 0.001 every satellite carries in the shipped
#: ``norad_satellite.rfimodel``, so which table a run used is visible in the
#: simulated RFI amplitude rather than only in the log.
CUSTOM_POWER = 10.0
PACKAGED_POWER = 0.001


def run_sim_vis(config_path, *args):
    argv = ["sim-vis", "--config", str(config_path), "-o", *args]
    with restored_stdout(), patch.object(sys, "argv", argv):
        return sim_vis.main()


def packaged_positions(n_ant=2):
    """The first *n_ant* antenna positions of the packaged MeerKAT ITRF file."""
    return np.loadtxt(
        TELESCOPE_DIR / "MeerKAT.itrf.txt", usecols=(0, 1, 2), max_rows=n_ant
    )


def custom_itrf_file(path, n_ant=2, offset=1000.0):
    """An antenna file of the run's own, distinguishable from the packaged one.

    The packaged MeerKAT positions moved a kilometre along ITRF X: still a
    plausible array at that site, and an exact comparison says which file was read.
    """
    positions = packaged_positions(n_ant) + np.array([offset, 0.0, 0.0])
    np.savetxt(path, positions)
    return str(path), positions


def spec_model(path, norad_ids, power=CUSTOM_POWER):
    """A spectral-model CSV for *norad_ids*, in the shipped ``.rfimodel`` shape.

    Same centre frequency and bandwidth as the packaged table, so the emission
    power is the only thing that differs between the two.
    """
    rows = ["norad_id,sat_name,object_id,sig_type,power,freq,band_width"]
    rows += [
        f"{int(nid)},TEST {nid},2024-000A,gauss,{power},1000000000.0,1000000000.0"
        for nid in norad_ids
    ]
    Path(path).write_text("\n".join(rows) + "\n")
    return str(path)


def write_replay_dir(directory, norad_ids, records):
    """The two files a frozen replay reads, as a completed run would write them."""
    directory.mkdir(parents=True, exist_ok=True)
    orbit.save_orbits_for_reuse(
        directory / "used_orbits.json", list(norad_ids), list(records)
    )
    (directory / "norad_ids.yaml").write_text(
        "".join(f"{int(nid)}\n" for nid in norad_ids)
    )
    return directory


def final_ids(obs):
    """The NORAD IDs the observation actually propagated, flattened once."""
    if not len(obs.norad_ids):
        return []
    return [int(nid) for nid in np.concatenate(obs.norad_ids).compute()]


def satellite_xyz(obs):
    import dask.array as da

    return da.concatenate(obs.rfi_tle_satellite_xyz, axis=0).compute()


def tle_line_slots(obs):
    """The two TLE-line columns the output schema carries, one row per satellite."""
    import dask.array as da

    return da.concatenate(obs.rfi_tle_satellite_orbit, axis=0).compute()


def test_sim_vis_help_exists(capsys):
    test_args = ["sim-vis", "--help"]

    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as e:
            sim_vis.main()

    assert e.value.code == 0  # argparse exits with code 0 for --help
    out = capsys.readouterr().out
    assert "usage:" in out.lower()
    assert "--help" in out


def test_missing_config_file():
    args = ["sim-vis", "--config", "nonexistent.yaml"]

    with patch.object(sys, "argv", args), pytest.raises(SystemExit):
        sim_vis.main()


@pytest.mark.allow_network
def test_simulation_runs_with_config(capsys, tmp_path):
    """The one live check: a real NAVSTAR selection, resolved against SatChecker.

    "It finished" is not the assertion: an outage that dropped every named
    satellite would finish too, so the run must be shown to have modelled
    satellites and saved the records it modelled them from.
    """
    config_path = (
        Path(__file__).parent.parent / "examples" / "test" / "sim_test_16A.yaml"
    )

    assert config_path.is_file(), f"Missing config file: {config_path}"
    args = [
        "sim-vis",
        "--config",
        str(config_path),
        "-o",
        "-sp",
        str(tmp_path / "out"),
    ]

    with patch.object(sys, "argv", args):
        obs, output_path = sim_vis.main()

    output = capsys.readouterr().out
    assert "Total simulation time" in output
    assert obs.n_rfi_tle_satellite > 0

    saved = Path(output_path) / "input_data"
    saved_ids = read_norad_ids_file(saved / "norad_ids.yaml")
    assert saved_ids == final_ids(obs)
    assert saved_ids, "a completed run must record the satellites it modelled"

    records = read_legacy_tle_records(saved)
    assert sorted(int(nid) for nid in records["NORAD_CAT_ID"]) == sorted(saved_ids)


@pytest.mark.parametrize("archives", ["tle", "omm", "mixed"])
def test_sim_vis_offline_frozen_replay(tmp_path, monkeypatch, archives):
    """A replayed run reproduces the visibilities, from its saved records alone.

    The cache is emptied and every transport seam raises, so anything less than an
    exact match on ``vis_obs`` means the replay modelled a different sky.
    """
    tle_records = {nid: tle_record_at(nid, ISS_EPOCH_JD) for nid in SIM_IDS}
    omm_records = {nid: omm_record_at(nid, ISS_EPOCH_JD) for nid in SIM_IDS}
    if archives == "tle":
        stub_records(monkeypatch, tle=tle_records)
    elif archives == "omm":
        stub_records(monkeypatch, omm=omm_records)
    else:
        stub_records(
            monkeypatch,
            tle={ISS_NORAD_ID: tle_records[ISS_NORAD_ID]},
            omm={GPS_NORAD_ID: omm_records[GPS_NORAD_ID]},
        )

    first_config = tiny_sim_config(
        tmp_path / "run.yaml", tmp_path / "run", norad_ids=SIM_IDS
    )
    obs, save_path = run_sim_vis(first_config)

    assert final_ids(obs) == SIM_IDS
    replay_dir = Path(save_path) / "input_data"
    assert (replay_dir / "used_orbits.json").exists()

    # An OMM record has no lines to give, so the output schema's line columns are
    # left empty for those rows rather than filled with something TLE-shaped.
    if archives == "omm":
        assert set(tle_line_slots(obs).ravel()) == {""}

    # Nothing but the saved files: no cache, no service, no name search.
    monkeypatch.setenv("ORBIT_CACHE_DIR", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(client, "fetch_nearest_tle", forbidden("the TLE endpoint"))
    monkeypatch.setattr(client, "fetch_nearest_omm", forbidden("the OMM endpoint"))

    replay_config = tiny_sim_config(
        tmp_path / "replay.yaml", tmp_path / "replay", norad_ids=[]
    )
    replayed, _ = run_sim_vis(
        replay_config, "--replay-orbit-dir", str(replay_dir)
    )

    assert final_ids(replayed) == SIM_IDS
    np.testing.assert_array_equal(satellite_xyz(replayed), satellite_xyz(obs))
    np.testing.assert_array_equal(
        replayed.vis_obs.compute(), obs.vis_obs.compute()
    )


def test_sim_vis_named_outage_does_not_write_successful_observation(
    tmp_path, monkeypatch
):
    """A named satellite whose record could not be fetched stops the run.

    This used to finish: the search succeeded, the acquisition failed, and the
    simulation wrote a complete observation with no satellite RFI in it.
    """
    serve_search(monkeypatch, {"THING": [search_row(ISS_NORAD_ID, "THING ONE")]})

    def unreachable(norad_id, epoch_jd, *, strict_response=False):
        raise client.SatCheckerTransportError("connection refused")

    monkeypatch.setattr(client, "fetch_nearest_tle", unreachable)
    monkeypatch.setattr(client, "fetch_nearest_omm", unreachable)

    output_path = tmp_path / "out"
    config_path = tiny_sim_config(
        tmp_path / "sim.yaml", output_path, sat_names=["thing"]
    )

    with pytest.raises(orbit.OrbitError) as excinfo:
        run_sim_vis(config_path)

    assert str(ISS_NORAD_ID) in str(excinfo.value)
    assert list(output_path.glob("**/*.zarr")) == []


def test_configured_spectral_model_is_not_replaced_by_the_packaged_one(
    tmp_path, monkeypatch
):
    """Startup's packaged defaults must not overwrite a configured spectral model.

    ``deep_update`` applied the shipped table over the configured one, so a
    satellite only the user's model covers was left out of the simulation.
    """
    unknown = 99999  # deliberately absent from the shipped norad_satellite.rfimodel
    stub_records(monkeypatch, tle={unknown: tle_record_at(unknown, ISS_EPOCH_JD)})
    config_path = tiny_sim_config(
        tmp_path / "sim.yaml",
        tmp_path / "out",
        norad_ids=[unknown],
        norad_spec_model=spec_model(tmp_path / "mine.rfimodel", [unknown]),
    )

    obs, _ = run_sim_vis(config_path)

    assert final_ids(obs) == [unknown]
    assert obs.n_rfi_tle_satellite == 1


def test_configured_spectral_model_survives_into_a_frozen_replay(
    tmp_path, monkeypatch
):
    """...and a replay reports a missing spectrum only when one is really missing."""
    unknown = 99999
    replay_dir = write_replay_dir(
        tmp_path / "input_data", [unknown], [tle_record_at(unknown, ISS_EPOCH_JD)]
    )
    monkeypatch.setattr(client, "fetch_nearest_tle", forbidden("the TLE endpoint"))
    monkeypatch.setattr(client, "fetch_nearest_omm", forbidden("the OMM endpoint"))
    config_path = tiny_sim_config(
        tmp_path / "replay.yaml",
        tmp_path / "out",
        norad_spec_model=spec_model(tmp_path / "mine.rfimodel", [unknown]),
        replay_orbit_dir=str(replay_dir),
    )

    obs, _ = run_sim_vis(config_path)

    assert final_ids(obs) == [unknown]


def test_configured_spectral_model_sets_the_simulated_power(tmp_path, monkeypatch):
    """For a satellite both tables cover, the configured power is the one simulated.

    The quiet half of the same defect: the run completes and models the satellite,
    with its emission power from a file the configuration replaced.
    """
    stub_records(
        monkeypatch, tle={ISS_NORAD_ID: tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD)}
    )
    packaged, _ = run_sim_vis(
        tiny_sim_config(
            tmp_path / "packaged.yaml", tmp_path / "packaged", norad_ids=[ISS_NORAD_ID]
        )
    )
    custom, _ = run_sim_vis(
        tiny_sim_config(
            tmp_path / "custom.yaml",
            tmp_path / "custom",
            norad_ids=[ISS_NORAD_ID],
            norad_spec_model=spec_model(
                tmp_path / "mine.rfimodel", [ISS_NORAD_ID], power=CUSTOM_POWER
            ),
        )
    )

    # Apparent amplitude goes as the square root of the emission power.
    ratio = np.max(custom.rfi_tle_satellite_A_app) / np.max(
        packaged.rfi_tle_satellite_A_app
    )
    assert float(ratio) == pytest.approx(
        np.sqrt(CUSTOM_POWER / PACKAGED_POWER), rel=1e-6
    )


def test_named_telescope_does_not_replace_a_configured_antenna_file(tmp_path):
    """Completing a telescope section must not overwrite the part that was set.

    ``dish_d: null`` is how one asks a named telescope for its diameter, and
    ``deep_update`` then replaced the configured antenna file with the packaged
    ``MeerKAT.itrf.txt``: a different array, simulated silently.
    """
    itrf_path, positions = custom_itrf_file(tmp_path / "mine.itrf.txt")
    obs, _ = run_sim_vis(
        tiny_sim_config(
            tmp_path / "sim.yaml",
            tmp_path / "out",
            telescope={"itrf_path": itrf_path, "dish_d": None},
        )
    )

    np.testing.assert_allclose(obs.ITRF.compute(), positions)
    assert float(obs.dish_d.compute()) == PACKAGED_DISH_D


def test_named_telescope_does_not_replace_a_configured_dish_diameter(tmp_path):
    """...and the mirror case: a configured diameter, geometry from the definition.

    ``dish_d: 25`` with no antenna file became the packaged 13.5, which changes
    the primary beam and every apparent source amplitude through it.
    """
    obs, _ = run_sim_vis(
        tiny_sim_config(
            tmp_path / "sim.yaml", tmp_path / "out", telescope={"dish_d": 25.0}
        )
    )

    assert float(obs.dish_d.compute()) == 25.0
    np.testing.assert_allclose(obs.ITRF.compute(), packaged_positions())
    # Nothing said about elevation, so the named telescope's own applies.
    assert float(obs.elevation.compute()) == PACKAGED_ELEVATION


def test_named_telescope_does_not_replace_an_explicit_zero_elevation(tmp_path):
    """``elevation: 0`` at a named site is a setting, not an omission.

    The template default was ``0`` too, so the definition's 1050 m won either way
    — and an ENU array is placed on the ellipsoid through that elevation, so the
    configured array moved. The template default is now null.
    """
    enu = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    np.savetxt(tmp_path / "mine.enu.txt", enu)
    telescope = {
        "enu_path": str(tmp_path / "mine.enu.txt"),
        "latitude": -30.71333,
        "longitude": 21.44306,
        "elevation": 0,
        "dish_d": None,  # asks the named telescope for its diameter
    }
    obs, _ = run_sim_vis(
        tiny_sim_config(tmp_path / "sim.yaml", tmp_path / "out", telescope=telescope)
    )
    assert float(obs.elevation.compute()) == 0.0
    assert float(obs.dish_d.compute()) == PACKAGED_DISH_D

    # The array really is placed at sea level: the same ENU file at the packaged
    # elevation gives ITRF positions further from the geocentre.
    raised, _ = run_sim_vis(
        tiny_sim_config(
            tmp_path / "sim2.yaml", tmp_path / "out2",
            telescope={**telescope, "elevation": PACKAGED_ELEVATION},
        )
    )
    radius = lambda o: np.linalg.norm(o.ITRF.compute(), axis=1)  # noqa: E731
    assert np.all(radius(raised) > radius(obs))
    np.testing.assert_allclose(radius(raised) - radius(obs), PACKAGED_ELEVATION, atol=1.0)


def test_unset_elevation_without_a_named_telescope_is_sea_level(tmp_path):
    # A complete custom definition never reaches apply_telescope_definition, so
    # the old template default has to be supplied somewhere; load_obs does it.
    enu = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    np.savetxt(tmp_path / "mine.enu.txt", enu)
    obs, _ = run_sim_vis(
        tiny_sim_config(
            tmp_path / "sim.yaml", tmp_path / "out",
            telescope={
                "name": "nowhere",
                "enu_path": str(tmp_path / "mine.enu.txt"),
                "latitude": -30.71333,
                "longitude": 21.44306,
                "dish_d": 13.5,
            },
        )
    )
    assert float(obs.elevation.compute()) == 0.0


def test_named_telescope_does_not_override_the_configured_antenna_frame(tmp_path):
    """An ENU array is a choice of source, so the packaged ITRF file must not win.

    ``Telescope`` lets ITRF positions replace ENU ones, so filling in the
    definition's ``itrf_path`` discards the configured array rather than
    completing it.
    """
    enu = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    np.savetxt(tmp_path / "mine.enu.txt", enu)
    obs, _ = run_sim_vis(
        tiny_sim_config(
            tmp_path / "sim.yaml",
            tmp_path / "out",
            telescope={"enu_path": str(tmp_path / "mine.enu.txt"), "dish_d": None},
        )
    )

    np.testing.assert_allclose(obs.ENU.compute(), enu)
    assert not np.allclose(obs.ITRF.compute(), packaged_positions())


def test_shared_designator_candidates_are_not_promised_to_be_modelled(
    tmp_path, monkeypatch, capsys
):
    """Two candidates under one OBJECT_ID, both resolving, and ``max_n_sat: 1``.

    Discovery promised "simulated once per number that resolves", but it runs
    before ``max_n_sat`` has its say, which here models one of the two.
    """
    serve_search(
        monkeypatch,
        {"TWIN": [search_row(nid, "TWIN SAT", object_id="2024-100A") for nid in SIM_IDS]},
    )
    stub_records(
        monkeypatch, tle={nid: tle_record_at(nid, ISS_EPOCH_JD) for nid in SIM_IDS}
    )

    obs, _ = run_sim_vis(
        tiny_sim_config(
            tmp_path / "sim.yaml", tmp_path / "out", sat_names=["twin"], max_n_sat=1
        )
    )

    output = capsys.readouterr().out
    assert "candidate NORAD catalogue ID" in output
    assert "may be modelled separately" in output
    assert "simulated once per number that resolves" not in output
    # Both numbers resolved to an acceptable record; one of them was modelled.
    assert final_ids(obs) == [ISS_NORAD_ID]
    assert obs.n_rfi_tle_satellite == 1


def test_replay_logs_an_ignored_numpy_id_array(tmp_path, monkeypatch, capsys):
    """An ignored ``norad_ids`` array must be reportable, not truth-tested.

    The log filtered the overridden settings with ``if value``, on which a NumPy
    array of IDs raises. Driven through ``run_sim_config`` directly, because a
    YAML file cannot carry one.
    """
    replay_dir = write_replay_dir(
        tmp_path / "input_data",
        SIM_IDS,
        [tle_record_at(nid, ISS_EPOCH_JD) for nid in SIM_IDS],
    )
    monkeypatch.setattr(client, "fetch_nearest_tle", forbidden("the TLE endpoint"))
    monkeypatch.setattr(client, "fetch_nearest_omm", forbidden("the OMM endpoint"))
    sim_config = load_config(
        tiny_sim_config(
            tmp_path / "replay.yaml",
            tmp_path / "out",
            replay_orbit_dir=str(replay_dir),
        ),
        config_type="sim",
    )
    sim_config["rfi_sources"]["tle_satellite"]["norad_ids"] = np.array(SIM_IDS)

    with restored_stdout():
        obs, _ = run_sim_config(sim_config=sim_config)

    output = capsys.readouterr().out
    assert final_ids(obs) == SIM_IDS
    assert "satellite selection is overridden" in output
    assert "norad_ids=" in output


def test_replay_runs_with_a_deleted_original_id_file(tmp_path, monkeypatch):
    """A replay of a run whose ID file has since moved must still complete.

    Normalisation validated ``norad_ids_path`` and ``save_inputs`` copied it, so
    either one made a replay require the previous run's inputs to still exist.
    """
    replay_dir = write_replay_dir(
        tmp_path / "input_data",
        SIM_IDS,
        [tle_record_at(nid, ISS_EPOCH_JD) for nid in SIM_IDS],
    )
    monkeypatch.setattr(client, "fetch_nearest_tle", forbidden("the TLE endpoint"))
    monkeypatch.setattr(client, "fetch_nearest_omm", forbidden("the OMM endpoint"))
    config_path = tiny_sim_config(
        tmp_path / "replay.yaml",
        tmp_path / "out",
        replay_orbit_dir=str(replay_dir),
        norad_ids_path=str(tmp_path / "gone.txt"),
        sat_names="NAVSTAR",  # not even a list: the replay never reads it
    )

    obs, save_path = run_sim_vis(config_path)

    assert final_ids(obs) == SIM_IDS
    assert not (Path(save_path) / "input_data" / "gone.txt").exists()


def test_sim_vis_offline_over_age_record_does_not_silently_drop_a_satellite(
    tmp_path, isolated_cache
):
    """Offline, an over-age cached record must stop the run, not exclude the satellite.

    The cached search says the satellite exists and the cached record is ten days
    out, with nothing asked for a closer one; calling that "the catalogue has
    nothing acceptable" wrote a complete observation with no satellite RFI.
    """
    cache = TextOrbitCache(isolated_cache)
    cache.store_search(
        "THING",
        search_frame([search_row(ISS_NORAD_ID, "THING ONE")]),
        fetched_at=datetime.now(timezone.utc),
    )
    cache.store(
        ISS_NORAD_ID, pd.DataFrame([tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD - 10.0)])
    )

    output_path = tmp_path / "out"
    config_path = tiny_sim_config(
        tmp_path / "sim.yaml", output_path, sat_names=["thing"], offline=True
    )

    with pytest.raises(orbit.OrbitError) as excinfo:
        run_sim_vis(config_path)

    message = str(excinfo.value)
    assert str(ISS_NORAD_ID) in message
    assert "offline" in message.lower()
    assert list(output_path.glob("**/*.zarr")) == []
