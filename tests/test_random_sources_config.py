import sys
from types import SimpleNamespace
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import yaml

from tabsim import config
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
    """4 sigma of this observation is 1.061 Jy, above the default `max_I` of 1 Jy.
    This used to hang `sim-vis`."""
    sim_config = small_sim_config({"n_src": 2, "min_I": "4sigma", "max_I": 1.0})

    with fail_after(300):
        with pytest.raises(ValueError) as err:
            run_sim_vis(sim_config, tmp_path, monkeypatch)

    msg = str(err.value)
    assert "ast_sources.point.random" in msg
    assert "min_I = '4sigma'" in msg and "max_I = 1.0" in msg
    assert "1.061 Jy > 1 Jy" in msg
    for setting in [
        "sigma = 0.2652 Jy",
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


@pytest.mark.parametrize(
    "min_I,problem",
    [
        (None, "None is neither a flux in Jy nor a number of sigma"),
        ([1.0], "[1.0] is neither a flux in Jy nor a number of sigma"),
        ("3sig", "'3sig' is not a number of sigma such as '3sigma'"),
        (float("nan"), "min_I = nan is not a number (NaN)"),
        ("nansigma", "min_I = 'nansigma' is not a number (NaN)"),
    ],
)
def test_flux_limit_that_is_not_a_flux_raises(min_I, problem, tmp_path, monkeypatch):
    """These used to be a TypeError or an empty ValueError from deep inside, or
    sources with fluxes of NaN."""
    sim_config = small_sim_config({"n_src": 2, "min_I": min_I, "max_I": 1.0})

    with fail_after(300):
        with pytest.raises(ValueError) as err:
            run_sim_vis(sim_config, tmp_path, monkeypatch)

    assert "ast_sources.point.random" in str(err.value)
    assert "min_I" in str(err.value) and problem in str(err.value)


def test_failing_to_explain_sigma_does_not_hide_the_empty_range(tmp_path, monkeypatch):
    def broken(obs):
        raise RuntimeError("no explanation")

    monkeypatch.setattr(config, "describe_image_noise", broken)
    sim_config = small_sim_config({"n_src": 2, "min_I": "4sigma", "max_I": 1.0})

    with fail_after(300):
        with pytest.raises(ValueError) as err:
            run_sim_vis(sim_config, tmp_path, monkeypatch)

    assert "min_I = '4sigma' is greater than max_I = 1.0" in str(err.value)
    assert "Lower min_I, raise max_I, or lower sigma" in str(err.value)
    assert "no explanation" not in str(err.value)


def test_sigma_value_of_numbers_and_of_sigma():
    obs = SimpleNamespace(noise_std=0.6 * da.ones(3), n_time=4, n_bl=1)

    assert config.sigma_value(0.5, obs) == 0.5
    assert config.sigma_value(2, obs) == 2.0
    assert config.sigma_value(np.float32(0.5), obs) == 0.5
    assert config.sigma_value(np.int64(2), obs) == 2.0
    assert config.sigma_value("2sigma", obs) == pytest.approx(2 * 0.6 / 2)


def test_sigma_value_does_not_hide_an_error_from_the_noise():
    """It used to turn any error at all into an empty ValueError."""
    obs = SimpleNamespace(noise_std=da.ones(3), n_time=4, n_bl="one")

    with pytest.raises(TypeError):
        config.sigma_value("3sigma", obs)

