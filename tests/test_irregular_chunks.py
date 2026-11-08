"""Dask wrappers keep an irregular chunk layout and its values (issue #81)."""
import dask.array as da
import jax
import numpy as np
import pytest

from tabsim.dask import coordinates as dc
from tabsim.dask import interferometry as di

rng = np.random.default_rng(0)
T, A, S, F = 5, 3, 2, 2
TIME = (2, 1, 2)
a1, a2 = np.triu_indices(A, 1)
GEO = np.stack([rng.uniform(-30, 30, (T, A)), rng.uniform(0, 360, (T, A)), np.full((T, A), 1e3)], -1)
XYZ_ANT = rng.uniform(1e6, 2e6, (T, A, 3))
XYZ_SRC = rng.uniform(1e7, 2e7, (S, T, 3))
VIS = rng.normal(size=(T, len(a1), F)) + 1j * rng.normal(size=(T, len(a1), F))
GAINS = rng.normal(size=(T, A, F)) + 1j * rng.normal(size=(T, A, F))

# (wrapper, arrays, chunks with the time axis set to TIME, time axis of the output)
CASES = {
    "ITRF_to_XYZ": (dc.ITRF_to_XYZ, [rng.uniform(1e6, 2e6, (A, 3)), np.linspace(0, 1, T)], [(A, 3), TIME], 0),
    "GEO_to_XYZ": (dc.GEO_to_XYZ, [GEO[:, 0], np.linspace(0, 40, T)], [(TIME, 3), TIME], 0),
    "GEO_to_XYZ_vmap0": (dc.GEO_to_XYZ_vmap0, [GEO.transpose(1, 0, 2), np.linspace(0, 40, T)], [((1, 2), TIME, 3), TIME], 1),
    "GEO_to_XYZ_vmap1": (dc.GEO_to_XYZ_vmap1, [GEO, np.linspace(0, 40, T)], [(TIME, (2, 1), 3), TIME], 0),
    "angular_separation": (lambda r, a: dc.angular_separation(r, a, 27.0, -30.0), [XYZ_SRC, XYZ_ANT], [(1, TIME, 3), (TIME, A, 3)], 1),
    "orbit_vmap": (dc.orbit_vmap, [np.linspace(0, 40, T)] + [rng.uniform(1, 2, S) * s for s in (5e5, 50, 30, 10)], [TIME] + [(1, 1)] * 4, 1),
    "ants_to_bl": (lambda g: di.ants_to_bl(g, da.from_array(a1), da.from_array(a2)), [GAINS], [(TIME, A, F)], 0),
    "apply_gains": (lambda v, r, g: di.apply_gains(v, r, g, da.from_array(a1), da.from_array(a2)), [VIS, VIS, GAINS], [(TIME, len(a1), F)] * 2 + [(TIME, A, F)], 0),
}


@pytest.fixture(autouse=True)
def x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


@pytest.mark.parametrize("name", CASES)
def test_wrapper_keeps_irregular_chunks(name):
    fn, arrays, chunks, axis = CASES[name]
    out = fn(*[da.from_array(x, chunks=c) for x, c in zip(arrays, chunks)])
    assert out.chunks[axis] == TIME
    np.testing.assert_allclose(out.compute(), fn(*[da.from_array(x) for x in arrays]).compute(), rtol=1e-12)


@pytest.mark.parametrize("fn", [
    lambda itrf: dc.ITRF_to_UVW(itrf, da.from_array(np.array([0.0, 15.0])), -30.0),
    lambda enu: dc.ENU_to_GEO(da.from_array(np.array([-30.0, 21.0, 1e3])), enu),
])
def test_split_antenna_axis_matches_one_block(fn):
    positions = np.array([[10.0, 0, 0], [20, 5, 0], [30, 0, 2], [40, 1, 0]])
    split = fn(da.from_array(positions, chunks=((1, 2, 1), 3))).compute()
    np.testing.assert_allclose(split, fn(da.from_array(positions)).compute(), rtol=1e-12)


SPLIT = {
    # Zero phase everywhere: the visibility is the sum of fluxes (15), the mean
    # squared sample amplitude (11) and the product of gains 1 and 2 (2).
    "astro_vis_sources": (lambda s, l: di.astro_vis(s, da.zeros((T, 1, 3)), l, da.from_array(np.full(1, 1e8))),
                          [np.arange(1.0, 6)[:, None, None] * np.ones((5, T, 1)), np.tile([0.0, 0, 1], (5, 1))],
                          [((2, 1, 2), T, 1), ((2, 1, 2), 3)], 15),
    "rfi_vis_samples": (lambda a, d: di.rfi_vis(a, d, da.from_array(np.full(1, 1e8)), da.from_array([0]), da.from_array([1])),
                        [np.arange(1.0, 6)[None, None, :, None, None] * np.ones((1, T, 5, 2, 1)), np.zeros((1, T, 5, 2))],
                        [(1, T, (2, 1, 2), 2, 1), (1, T, (2, 1, 2), 2)], 11),
    "ants_to_bl_antennas": (lambda g: di.ants_to_bl(g, da.from_array([0]), da.from_array([1])),
                            [np.arange(1.0, 6)[None, :, None] * np.ones((T, 5, 1), complex)], [(T, (2, 1, 2), 1)], 2),
}


@pytest.mark.parametrize("name", SPLIT)
def test_split_reduced_axis_is_not_truncated(name):
    fn, arrays, chunks, expected = SPLIT[name]
    np.testing.assert_allclose(fn(*[da.from_array(x, chunks=c) for x, c in zip(arrays, chunks)]).compute(), expected)
