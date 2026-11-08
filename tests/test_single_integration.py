"""An observation of one integration samples it at its own int_time."""
import jax
import numpy as np
import pytest
import skyfield.api  # Initialize its urllib feature probe before the network guard.

from benchmarks.harness import add_sources, build_observation
from tabsim.jax.interferometry import int_sample_times


def test_one_time_uses_the_given_int_time():
    np.testing.assert_allclose(int_sample_times(np.array([5.0]), 3, 2.0), [5 - 2 / 3, 5, 5 + 2 / 3])
    with pytest.raises(ValueError, match="int_time"):
        int_sample_times(np.array([5.0]), 3)
    with pytest.raises(ValueError, match="spacing"):
        int_sample_times(np.array([0.0, 10.0]), 1, 2.0)


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def single(samples):
    case = dict(telescope="SKA-Low-AA0.5", antennas=4, times=1, channels=2,
                samples=samples, point_sources=1, rfi_sources=1)
    obs = build_observation(case, 0.001)
    add_sources(obs, case)
    obs.calculate_vis(random_seed=0)
    return obs


def test_single_integration_observation_computes(x64):
    obs = single(3)
    np.testing.assert_allclose(obs.times_fine.compute(), [-2 / 3, 0, 2 / 3], atol=1e-12)
    assert np.isfinite(obs.dataset.vis_obs.values).all()


def test_one_sample_skips_fringe_estimate_and_reads_back(x64, tmp_path, capsys):
    from tabsim.config import print_fringe_freq_sat, print_fringe_freq_tle_sat
    from tabsim.read_ms import read_time_data_from_ms
    obs = single(1)
    print_fringe_freq_sat(obs)
    print_fringe_freq_tle_sat(obs)
    assert capsys.readouterr().out.count("Fringe frequency estimate skipped") == 2
    obs.write_to_ms(str(tmp_path / "one.ms"))
    assert read_time_data_from_ms(str(tmp_path / "one.ms"))[1:] == (2.0, 1)
