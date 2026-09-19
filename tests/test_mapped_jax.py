"""Mapped kernels execute inside their owning Dask task without another scheduler."""
import dask
import dask.array as da
from dask.base import DaskMethodsMixin
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import skyfield.api  # Run its urllib feature probe before the offline fixture.

from tabsim.dask import coordinates as dc
from tabsim.dask import interferometry as di
from tabsim.jax import coordinates as jc
from tabsim.jax import interferometry as ji


def _array(value, chunks):
    return da.from_array(np.asarray(value), chunks=chunks)


def _cases():
    # Uneven time/frequency/source tails exercise more than the first block.
    rng = np.random.default_rng(51)
    frequencies = _array(np.linspace(1e8, 1.1e8, 3), 2)
    uvw = _array(rng.normal(size=(3, 3, 3)), (2, 2, 3))
    sources = _array(rng.uniform(.2, 1, (2, 3, 3)), (2, 2, 2))
    lmn = _array([[.01, .02, np.sqrt(1 - .0005)], [0, 0, 1]], (2, 3))
    shape = _array([.1, .2], 2)
    a1, a2 = _array([0, 0, 1], 2), _array([1, 2, 2], 2)
    gains = _array(rng.normal(size=(3, 3, 3)) + 1j, (2, 3, 2))
    vis = _array(rng.normal(size=(3, 3, 3)) + 2j, (2, 2, 2))
    yield di, ji, 'astro_vis', 'astro_vis', [sources, uvw, lmn, frequencies]
    yield di, ji, 'astro_vis_gauss', 'astro_vis_gauss', [sources, shape, shape, shape, uvw, lmn, frequencies]
    yield di, ji, 'astro_vis_exp', 'astro_vis_exp', [sources, shape, uvw, lmn, frequencies]
    amplitude = _array(rng.uniform(.1, 1, (2, 3, 2, 3, 3)), (2, 2, 2, 3, 2))
    distance = _array(rng.uniform(1, 10, (2, 3, 2, 3)), (2, 2, 2, 3))
    yield di, ji, 'rfi_vis', 'rfi_vis', [amplitude, distance, frequencies, a1, a2]
    yield di, ji, 'ants_to_bl', 'ants_to_bl', [gains, a1, a2]
    theta = _array(rng.uniform(.1, 1, (2, 3, 3)), (1, 2, 2))
    yield di, ji, 'airy_beam', 'airy_beam', [theta, frequencies, 14.]
    power = _array(np.ones((2, 3, 3)), (1, 2, 2))
    distances = _array(np.full((2, 3, 3), 1000.), (1, 2, 2))
    yield di, ji, 'Pv_to_Sv', 'Pv_to_Sv', [power, distances]
    yield di, ji, 'apply_gains', 'apply_gains', [vis, vis, gains, a1, a2]
    ra, dec = _array([10., 11., 12.], 2), _array([-30., -31., -32.], 2)
    yield dc, jc, 'radec_to_lmn', 'radec_to_lmn', [ra, dec, _array([10., -30.], 2)]
    yield dc, jc, 'radec_to_XYZ', 'radec_to_XYZ', [ra, dec]
    times = _array([0., 10., 20.], 2)
    geo = _array([[10., 20., 100.]] * 3, (2, 3))
    yield dc, jc, 'GEO_to_XYZ', 'GEO_to_XYZ', [geo, times]
    yield dc, jc, 'GEO_to_XYZ_vmap0', 'GEO_to_XYZ_vmap0', [_array(np.tile([[10., 20., 100.]] * 3, (2, 1, 1)), (1, 2, 3)), times]
    yield dc, jc, 'GEO_to_XYZ_vmap1', 'GEO_to_XYZ_vmap1', [_array(np.tile(np.array([[10., 20., 100.]] * 3)[:, None, :], (1, 2, 1)), (2, 1, 3)), times]
    itrf = _array(rng.uniform(1e6, 2e6, (3, 3)), (2, 3))
    yield dc, jc, 'ITRF_to_XYZ', 'itrf_to_xyz', [itrf, times]
    yield dc, jc, 'ENU_to_ITRF', 'enu_to_itrf', [_array(rng.normal(size=(3, 3)), (2, 3)), -30., 20., 100.]
    # UVW uses antenna zero as origin, so the antenna axis must remain whole.
    yield dc, jc, 'ITRF_to_UVW', 'itrf_to_uvw', [itrf.rechunk((3, 3)), times, _array([-30.], 1)]
    rfi_xyz = _array(rng.uniform(1e7, 2e7, (2, 3, 3)), (1, 2, 3))
    ants_xyz = _array(rng.uniform(1e6, 2e6, (3, 3, 3)), (2, 2, 3))
    yield dc, jc, 'angular_separation', 'angular_separation', [rfi_xyz, ants_xyz, 10., -30.]
    yield dc, jc, 'orbit_vmap', 'orbit_vmap', [times, _array([5e5, 6e5], 1), _array([30., 45.], 1), _array([10., 20.], 1), _array([0., 5.], 1)]


CASES = list(_cases())


@pytest.fixture(autouse=True)
def _double_precision():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)



@pytest.mark.parametrize('case', CASES, ids=[case[2] for case in CASES])
@pytest.mark.parametrize('scheduler', ['synchronous', 'threads'])
def test_mapped_kernel_matches_primitive_without_nested_compute(case, scheduler, monkeypatch):
    mapped, primitive, name, kernel, args = case
    values = [arg.compute(scheduler='synchronous') if isinstance(arg, da.Array) else arg for arg in args]
    expected = np.asarray(getattr(primitive, kernel)(*[jnp.asarray(value) for value in values]))
    def forbidden(*args, **kwargs):
        raise AssertionError('mapped callback invoked a nested Dask compute')
    monkeypatch.setattr(DaskMethodsMixin, 'compute', forbidden)
    result = getattr(mapped, name)(*args)
    actual, = dask.compute(result, scheduler=scheduler, num_workers=2)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)


def test_coordinate_graph_with_process_scheduler(monkeypatch):
    # Exercise process serialization on CPU: two JAX GPU processes could each
    # reserve most VRAM under the default allocator policy. Other tests use the
    # selected backend, including GPU. Set child precision explicitly.
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "true")
    # Dask's default spawn context avoids forking JAX's multithreaded runtime.
    ra, dec = _array([10., 11., 12.], 2), _array([-30., -31., -32.], 2)
    graph = dc.radec_to_XYZ(ra, dec)
    expected = np.asarray(jc.radec_to_XYZ(np.array([10., 11., 12.]), np.array([-30., -31., -32.])))
    actual = graph.compute(scheduler='processes', num_workers=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_same_shape_blocks_reuse_jitted_callable(monkeypatch):
    traces = []
    def traced_kernel(ra, dec):
        # Python executes during tracing, never during compiled execution.
        traces.append((ra.shape, ra.dtype))
        return jc.radec_to_XYZ(ra, dec)
    monkeypatch.setattr(dc, '_radec_to_XYZ_jit', jax.jit(traced_kernel))
    for offset in [0., 1.]:
        graph = dc.radec_to_XYZ(_array(np.arange(6.) + offset, 2), _array(np.full(6, -30.), 2))
        graph.compute(scheduler='synchronous')
    assert len(traces) == 1
    # A legitimate new input shape still specializes the callable.
    dc.radec_to_XYZ(_array(np.arange(3.), 3), _array(np.full(3, -30.), 3)).compute(scheduler='synchronous')
    assert len(traces) == 2
