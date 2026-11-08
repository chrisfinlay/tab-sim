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


def test_single_integration_observation_computes():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        case = dict(telescope="SKA-Low-AA0.5", antennas=4, times=1, channels=2,
                    samples=3, point_sources=1, rfi_sources=1)
        obs = build_observation(case, 0.001)
        add_sources(obs, case)
        obs.calculate_vis(random_seed=0)
        np.testing.assert_allclose(obs.times_fine.compute(), [-2 / 3, 0, 2 / 3], atol=1e-12)
        assert np.isfinite(obs.dataset.vis_obs.values).all()
    finally:
        jax.config.update("jax_enable_x64", previous)
