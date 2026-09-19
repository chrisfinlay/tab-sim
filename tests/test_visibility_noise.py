"""Noise is lazy, chunk-bounded, reproducible and correctly normalized."""
import dask
import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest
import skyfield.api  # Initialize its urllib feature probe before the network guard.

from tabsim.dask.interferometry import add_noise, SEFD_to_noise_std


def test_noise_construction_is_lazy_and_preserves_chunks():
    # A 1 TiB logical cube must only construct a small graph, not allocate a cube.
    vis = da.zeros((65536, 1024, 1024), chunks=(256, 1024, 1024), dtype=complex)
    scale = da.ones(1024, chunks=128)
    def forbidden(*args):
        raise AssertionError("noise construction executed a Dask task")
    with Callback(pretask=forbidden):
        observed, noise = add_noise(vis, scale, 7)
    assert isinstance(noise, da.Array)
    assert noise.chunks == observed.chunks == vis.chunks
    assert noise.dtype == np.complex128
    assert len(noise.__dask_graph__()) < 10000


@pytest.mark.parametrize("scale", [2., np.array([0., 1., 2.]), da.from_array([0., 1., 2.], chunks=1)])
def test_repeatability_scheduling_and_added_noise(scale):
    vis = da.full((24, 8, 3), 4 + 2j, chunks=(6, 4, 2))
    observed, noise = add_noise(vis, scale, 0)
    a = noise.compute(scheduler="synchronous")
    b = noise.compute(scheduler="threads", num_workers=3)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, add_noise(vis, scale, 0)[1].compute())
    assert not np.array_equal(a, add_noise(vis, scale, 1)[1].compute())
    obs, n = dask.compute(observed, noise)
    np.testing.assert_array_equal(obs, 4 + 2j + n)
    assert noise.chunks == vis.chunks


def test_components_channels_and_blocks_have_expected_statistics():
    scales = np.array([0., .5, 2., 5.])
    vis = da.zeros((2048, 128, 4), chunks=(256, 64, 2), dtype=complex)
    noise = add_noise(vis, scales, 123)[1].compute()
    assert np.count_nonzero(noise[..., 0]) == 0
    for part in (noise.real, noise.imag):
        np.testing.assert_allclose(part.mean(axis=(0, 1)), 0, atol=.04)
        np.testing.assert_allclose(part.std(axis=(0, 1)), scales, rtol=.015, atol=1e-12)
    assert abs(np.corrcoef(noise.real[..., 1].ravel(), noise.imag[..., 1].ravel())[0, 1]) < .015
    first = noise[:256, :64, 1].real.ravel()
    next_block = noise[256:512, :64, 1].real.ravel()
    assert not np.array_equal(first, next_block)
    assert abs(np.corrcoef(first, next_block)[0, 1]) < .04


def test_radiometer_equation_matches_jax_per_component_scale():
    from tabsim.jax.interferometry import SEFD_to_noise_std as jax_std
    sefd = np.array([1000., 2000., 4000.])
    bandwidth = np.array([1e4, 2e4, 4e4])
    expected = sefd / np.sqrt(2 * bandwidth * 2.)
    np.testing.assert_allclose(SEFD_to_noise_std(sefd, bandwidth, 2.), expected)
    np.testing.assert_allclose(jax_std(sefd, bandwidth, 2.), expected, rtol=1e-6)


def test_observation_honors_explicit_zero_seed(monkeypatch):
    from types import SimpleNamespace
    pytest.importorskip("casacore")
    from tabsim.dask import observation as module
    vis = da.zeros((4, 2, 3), chunks=(2, 2, 3), dtype=complex)
    obs = SimpleNamespace(vis_ast=vis, vis_rfi=vis, gains_ants=None, a1=None, a2=None,
        time_chunk=2, bl_chunk=2, freq_chunk=3, noise_std=da.ones(3), random_seed=123)
    monkeypatch.setattr(module, "apply_gains", lambda data, *args: data)
    # Gains are inverted before the second call.
    obs.gains_ants = 1.
    monkeypatch.setattr(module, "construct_observation_ds", lambda obs: None)
    module.Observation.calculate_vis(obs, flags=False, random_seed=0)
    np.testing.assert_array_equal(obs.noise_data.compute(), add_noise(vis, 1., 0)[1].compute())


def test_jax_noise_uses_per_component_standard_deviation():
    import jax
    from tabsim.jax.interferometry import add_noise as jax_noise
    scales = np.array([.5, 2.])
    # Enable x64 explicitly as production does; avoid global test-order effects.
    with jax.experimental.enable_x64():
        _, noise = jax_noise(np.zeros((1024, 128, 2)), scales, jax.random.PRNGKey(123))
        noise = np.asarray(noise)
    for part in (noise.real, noise.imag):
        np.testing.assert_allclose(part.std(axis=(0, 1)), scales, rtol=.015)
