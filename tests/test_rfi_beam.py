"""Device RFI beam: legacy host-beam composition, tiling and lazy diagnostics.

Contract of the new functions in tabsim.jax.interferometry (angles in degrees):
  rfi_vis_beam(Pv (S,T,I,F), distances (S,T,I,A), w (T,I,A), ang_sep (S,T,I,A),
               freqs (F,), dish_d, a1, a2, *, visibility_precision) -> (T, n_bl, F)
  rfi_amplitude(Pv (...,F), distances (...,A), ang_sep (...,A), freqs (F,), dish_d)
      -> (...,A,F) float64, i.e. sqrt(|Pv_to_Sv|) * airy_beam for any leading dims.
"""
import re
from functools import partial

import dask.array as da
import jax
import numpy as np
import pytest
from astropy.utils import iers

import tabsim.config as config
from benchmarks.harness import add_sources, build_observation
from orbit_helpers import GPS_LINE1, GPS_LINE2, GPS_NORAD_ID, ISS_NORAD_ID, tle_record
from tabsim.beam import airy_beam
from tabsim.jax import interferometry as itf
from test_rfi_vis_kernel import compare, rfi_inputs

POWER = {'satellite': 2e-14, 'tle_satellite': 1e-14, 'stationary': 1e-12}
DATASET_NAME = {'satellite': 'rfi_sat_A', 'tle_satellite': 'rfi_tle_sat_A', 'stationary': 'rfi_stat_A'}


@pytest.fixture(autouse=True)
def x64_offline():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    with iers.conf.set_temp('auto_download', False), iers.conf.set_temp('iers_degraded_accuracy', 'ignore'):
        try:
            yield
        finally:
            jax.config.update('jax_enable_x64', previous)


def observation(rfi=True):
    case = dict(telescope='SKA-Low-AA0.5', antennas=4, times=4, channels=4, samples=5,
                point_sources=1, rfi_sources=int(rfi))
    obs = build_observation(case, .001)
    if rfi:
        # Equal Kepler and TLE counts: write.py gives both families the sat_src dimension.
        obs.addSatelliteRFI(np.full((2, 1, 4), POWER['satellite']), [20200e3, 19140e3],
                            [55., 64.8], [41., 17.], [-5., -31.])
        obs.addTLESatelliteRFI(np.full((2, 1, 4), POWER['tle_satellite']), [ISS_NORAD_ID, GPS_NORAD_ID],
                               [tle_record(), tle_record(GPS_NORAD_ID, GPS_LINE1, GPS_LINE2)])
    add_sources(obs, case)
    obs.calculate_vis(random_seed=0)
    return obs


def legacy_amplitude(obs, family):
    """The historical fine-time amplitude sqrt|Sv| * host Airy beam, and the distances."""
    xyz, ang, ants = da.compute(da.concatenate(getattr(obs, f'rfi_{family}_xyz')),
                                da.concatenate(getattr(obs, f'rfi_{family}_ang_sep')), obs.ants_xyz)
    distances = np.linalg.norm(ants[None] - xyz[:, :, None], axis=-1)
    Sv = itf.Pv_to_Sv(np.full(xyz.shape[:2] + (obs.n_freq,), POWER[family]), distances)
    return np.sqrt(np.abs(np.asarray(Sv))) * airy_beam(ang, np.asarray(obs.freqs), obs.dish_d), distances


def legacy_vis(obs):
    """Sum of historical rfi_vis calls over families, with compare-ready inputs."""
    w, freqs, a1, a2 = (np.asarray(x) for x in (obs.ants_uvw[:, :, -1], obs.freqs, obs.a1, obs.a2))
    shape = (-1, obs.n_time, obs.n_int_samples, obs.n_ant)
    amps, c_distances = zip(*[(amp.reshape(shape + (obs.n_freq,)), (d + w[None]).reshape(shape))
                              for amp, d in (legacy_amplitude(obs, f) for f in POWER)])
    vis = sum(np.asarray(itf.rfi_vis(amp, c_d, freqs, a1, a2, visibility_precision=obs.visibility_precision))
              for amp, c_d in zip(amps, c_distances))
    return vis, (np.concatenate(amps), np.concatenate(c_distances), freqs, a1, a2)


@pytest.mark.parametrize('precision', ['single', 'double'])
def test_rfi_vis_beam_matches_legacy_composition(precision):
    """rfi_vis_beam and rfi_amplitude reproduce Pv_to_Sv, the host beam and rfi_vis."""
    _, distances, freqs, a1, a2 = rfi_inputs(n_freq=3)
    S, T, I, A = distances.shape
    rng = np.random.default_rng(1)
    Pv = rng.uniform(1e-15, 1e-13, (S, T, I, 3))
    w = rng.normal(0., 50., (T, I, A))
    ang = rng.uniform(0., 90., distances.shape)
    ang[0, 0, 0] = 0.
    Sv = np.asarray(itf.Pv_to_Sv(Pv.reshape(S, -1, 3), distances.reshape(S, -1, A)))
    amp = (np.sqrt(np.abs(Sv)) * airy_beam(ang.reshape(S, -1, A), freqs, 13.5)).reshape(S, T, I, A, 3)
    args = (amp, distances + w, freqs, a1, a2)
    expected = np.asarray(itf.rfi_vis(*args, visibility_precision=precision))
    actual = jax.jit(partial(itf.rfi_vis_beam, visibility_precision=precision))(
        Pv, distances, w, ang, freqs, 13.5, a1, a2)
    compare(np.asarray(actual), expected, args, precision, double_tolerance=1e-11)
    np.testing.assert_allclose(itf.rfi_amplitude(Pv, distances, ang, freqs, 13.5), amp,
                               rtol=1e-12, atol=1e-14 * np.abs(amp).max())


def test_observation_rfi_matches_legacy_composition():
    """All three RFI families match the legacy composition; other components stay bit-identical."""
    obs, plain = observation(), observation(rfi=False)
    assert min(len(obs.vis_rfi.chunks[0]), len(obs.vis_rfi.chunks[2])) >= 2
    assert obs.vis_rfi.chunks == plain.vis_ast.chunks
    ds, reference = obs.dataset.compute(), plain.dataset.compute()
    for name in ('vis_ast', 'gains_ants', 'noise_data'):
        np.testing.assert_array_equal(ds[name], reference[name])
    for family, name in DATASET_NAME.items():
        amp = legacy_amplitude(obs, family)[0]
        np.testing.assert_allclose(ds[name], amp, rtol=1e-12, atol=1e-14 * np.abs(amp).max())
    expected, args = legacy_vis(obs)
    compare(ds.vis_rfi.values, expected, args, obs.visibility_precision, double_tolerance=1e-11)


def test_rfi_path_is_bounded_and_on_device(tmp_path, monkeypatch, capsys):
    """vis_rfi stages without the host beam in tile-sized tasks; diagnostics never materialise lists."""
    import scipy.special
    import tabsim.beam
    from dask.callbacks import Callback
    from tabsim.execution import execution_stats
    obs = observation()
    expected_max = f"{legacy_amplitude(obs, 'tle_satellite')[0].max():.5f}"
    def forbidden(*args, **kwargs):
        raise AssertionError('forbidden host evaluation')
    for module in (tabsim.beam, scipy.special):
        monkeypatch.setattr(module, 'jv', forbidden)
    tile = 16 * obs.time_fine_chunk * obs.ant_chunk * obs.freq_chunk * 2  # two sources per family
    sizes = []
    execution_stats(reset=True)
    with Callback(posttask=lambda key, result, *_: sizes.append(getattr(result, 'nbytes', 0))):
        obs.write_to_zarr(tmp_path / 'rfi.zarr', save_arrays=['vis_rfi'])
        monkeypatch.setattr(da.Array, '__array__', forbidden)
        config.print_fringe_freq_sat(obs)
        config.print_fringe_freq_tle_sat(obs)
    assert 0 < max(sizes) <= tile
    assert re.search(r'Maximum RFI Amplitude\s*: (\S+)', capsys.readouterr().out).group(1) == expected_max
    if jax.default_backend() == 'gpu':
        assert execution_stats()['calls'] > 0
