"""Numerical gates for the experimental staged writer; no performance assertions."""
import dask
import jax
import numpy as np
import pytest
import xarray as xr

# Skyfield probes urllib's signature with an empty URL during import. Load it
# before the per-test network guard, as the other observation tests do.
import tabsim.config

from benchmarks.harness import add_sources, build_observation
from benchmarks.staged_zarr import write_staged_observation


@pytest.fixture(autouse=True)
def double_precision():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


@pytest.mark.parametrize('flags,zero_noise,seed,sources', [
    (True, False, 0, True),
    (True, True, 17, True),
    (False, False, 23, True),
    (True, False, None, False),
])
def test_staged_matches_all_outputs(tmp_path, flags, zero_noise, seed, sources):
    case = dict(telescope='SKA-Low-AA0.5', antennas=4, times=4, channels=3,
                samples=3, point_sources=2 if sources else 0,
                rfi_sources=1 if sources else 0)
    with dask.config.set(scheduler='threads', num_workers=1):
        obs = build_observation(case, .001)
        add_sources(obs, case)
        if zero_noise:
            obs.noise_std = obs.noise_std * 0
        obs.calculate_vis(flags=flags, random_seed=seed)
        expected = obs.dataset.compute()
        write_staged_observation(obs, tmp_path, flags=flags)
        with xr.open_zarr(tmp_path / 'result.zarr', chunks={}) as actual:
            assert actual.attrs == expected.attrs
            assert set(actual.variables) == set(expected.variables)
            for name in expected.variables:
                assert actual[name].dims == expected[name].dims
                assert actual[name].dtype == expected[name].dtype
                if np.issubdtype(expected[name].dtype, np.number):
                    np.testing.assert_allclose(actual[name].values, expected[name].values,
                                               rtol=1e-12, atol=1e-12)
                else:
                    np.testing.assert_array_equal(actual[name].values, expected[name].values)
