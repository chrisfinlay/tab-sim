import sys
from unittest.mock import patch

import pytest
import yaml

from tabsim.scripts import sim_vis
from tabsim.sky import SourcePlacementError
from timeouts import fail_after


def small_sim_config(point_random: dict) -> dict:
    """Two antennas and three time steps: little data, so a lot of image noise."""
    return {
        "telescope": {"name": "MeerKAT", "n_ant": 2},
        "observation": {
            "target_name": "target",
            "ra": 27.0,
            "dec": -30.0,
            "start_time_lha": 0.0,
            "int_time": 2.0,
            "n_time": 3,
            "n_int": 2,
            "start_freq": 1.227e9,
            "chan_width": 209e3,
            "n_freq": 1,
            "SEFD": 420,
            "random_seed": 12345,
        },
        "ast_sources": {"point": {"random": point_random}},
        "output": {"path": "./data", "prefix": "small", "zarr": True, "ms": False},
        "diagnostics": {"uv_cov": False, "src_alt": False, "rfi_seps": False},
    }


def run_sim_vis(sim_config: dict, tmp_path, monkeypatch):
    config_path = tmp_path / "sim.yaml"
    config_path.write_text(yaml.safe_dump(sim_config))
    # The simulation log is written to the working directory
    monkeypatch.chdir(tmp_path)

    stdout = sys.stdout
    try:
        with patch.object(sys, "argv", ["sim-vis", "-c", str(config_path), "-o"]):
            sim_vis.main()
    finally:
        # A simulation that fails leaves stdout copied to its log
        sys.stdout = stdout


def test_min_flux_in_sigma_above_max_flux_raises(tmp_path, monkeypatch):
    """3 sigma of this observation is 1.125 Jy, above the default `max_I` of 1 Jy.
    This used to hang `sim-vis`."""
    sim_config = small_sim_config({"n_src": 2, "min_I": "3sigma", "max_I": 1.0})

    with fail_after(300):
        with pytest.raises(ValueError) as err:
            run_sim_vis(sim_config, tmp_path, monkeypatch)

    msg = str(err.value)
    assert "ast_sources.point.random" in msg
    assert "min_I = '3sigma'" in msg and "max_I = 1.0" in msg
    assert "1.125 Jy > 1 Jy" in msg
    for setting in [
        "sigma = 0.3751 Jy",
        "sqrt(3 * 1)",
        "SEFD = 420 Jy",
        "chan_width = 2.09e+05 Hz",
        "int_time = 2 s",
        "n_ant = 2",
    ]:
        assert setting in msg


def test_min_flux_above_max_flux_raises(tmp_path, monkeypatch):
    sim_config = small_sim_config({"n_src": 2, "min_I": 2.0, "max_I": 1.0})

    with fail_after(300):
        with pytest.raises(ValueError) as err:
            run_sim_vis(sim_config, tmp_path, monkeypatch)

    msg = str(err.value)
    assert "ast_sources.point.random" in msg
    assert "min_I = 2.0" in msg and "max_I = 1.0" in msg
    assert "sigma" not in msg


def test_sources_too_crowded_to_separate_raises(tmp_path, monkeypatch):
    """The two-antenna synthesized beam is wide, so `max_sep` sets the separation:
    300 sources 300 arcsec apart do not fit in the 1.27 degree primary beam."""
    sim_config = small_sim_config(
        {"n_src": 300, "min_I": 1e-3, "max_I": 1.0, "n_beam": 5, "max_sep": 300.0}
    )

    with fail_after(300):
        with pytest.raises(SourcePlacementError) as err:
            run_sim_vis(sim_config, tmp_path, monkeypatch)

    msg = str(err.value)
    assert "ast_sources.point.random" in msg
    assert "n_src = 300" in msg
    assert "max_sep / n_beam (60.0 arcsec)" in msg
    assert "the settings to lower are n_src, n_beam and max_sep" in msg
    assert "max_sep, which is in arcseconds" in msg


def test_small_observation_with_a_valid_flux_range_runs(tmp_path, monkeypatch, capsys):
    sim_config = small_sim_config({"n_src": 2, "min_I": "1sigma", "max_I": 1.0})

    with fail_after(300):
        run_sim_vis(sim_config, tmp_path, monkeypatch)

    assert "Total simulation time" in capsys.readouterr().out
