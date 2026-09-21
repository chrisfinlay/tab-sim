"""Bound geometry setup by time tiles while preserving offline frame/orbit replay."""
import dask.array as da
import jax
import numpy as np
import pytest

from tabsim.dask import coordinates as dc
from tabsim import tle
from benchmarks.harness import build_observation, offline
from orbit_helpers import ISS_EPOCH_JD, ISS_NORAD_ID, tle_record, omm_record_from_tle


@pytest.fixture(autouse=True)
def precision():
    old = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    offline()
    yield
    jax.config.update('jax_enable_x64', old)


@pytest.mark.parametrize('chunk', [1, 3, 7])
def test_antenna_blocks_match_eager_including_tail(chunk):
    positions = np.array([[5109000., 20000., -3200000.], [5109020., 20010., -3199990.]])
    times = ISS_EPOCH_JD + np.arange(7) / 86400
    expected = dc.itrs_to_gcrs_sf(positions, times)
    graph = dc.itrs_to_gcrs_blocks(positions, da.from_array(times, chunks=chunk))
    assert graph.chunks[0] == da.from_array(times, chunks=chunk).chunks[0]
    np.testing.assert_allclose(graph.compute(), expected, rtol=1e-13, atol=1e-8)


@pytest.mark.parametrize('record', [tle_record(), omm_record_from_tle()])
@pytest.mark.parametrize('chunk', [1, 3, 7])
def test_orbit_blocks_match_eager_and_are_offline(record, chunk):
    times = ISS_EPOCH_JD + np.arange(7) / 86400
    expected = tle.get_satellite_positions([record], times)
    graph = dc.satellite_position_blocks([record], da.from_array(times, chunks=chunk))
    np.testing.assert_allclose(graph.compute(), expected, rtol=1e-13, atol=1e-7)


def test_graph_construction_and_slice_do_not_propagate_full_history(monkeypatch):
    times = da.from_array(ISS_EPOCH_JD + np.arange(31) / 86400, chunks=3)
    calls = []
    def propagate(records, dates):
        calls.append(len(dates))
        return np.zeros((len(records), len(dates), 3))
    monkeypatch.setattr(tle, 'get_satellite_positions', propagate)
    graph = dc.satellite_position_blocks([tle_record()], times)
    assert calls == []
    assert graph[:, 6:8].compute(scheduler='synchronous').shape == (1, 2, 3)
    assert calls == [3]
    calls.clear()
    def transform(positions, dates):
        calls.append(len(dates))
        return np.zeros((len(dates), len(positions), 3))
    monkeypatch.setattr(dc, 'itrs_to_gcrs_sf', transform)
    graph = dc.itrs_to_gcrs_blocks(np.ones((4, 3)), times)
    assert calls == []
    graph[6:8].compute(scheduler='synchronous')
    assert calls == [3]


def test_observation_setup_only_evaluates_first_uvw_tile(monkeypatch):
    calls = []
    original = dc._itrf_to_uvw_jit
    def uvw(positions, hour_angle, dec):
        calls.append(len(hour_angle))
        return original(positions, hour_angle, dec)
    monkeypatch.setattr(dc, '_itrf_to_uvw_jit', uvw)
    def forbidden(*args, **kwargs):
        raise AssertionError('Skyfield geometry eagerly computed during setup')
    monkeypatch.setattr(dc, 'itrs_to_gcrs_sf', forbidden)
    obs = build_observation(dict(telescope='SKA-Low-AA0.5', antennas=4, times=32,
        channels=4, samples=3, point_sources=0, rfi_sources=0), .001)
    assert calls and sum(calls) <= obs.time_fine_chunk
    assert obs.time_fine_chunk < obs.n_time_fine
    obs.addTLESatelliteRFI(np.ones((1, 1, 4))*1e-14, [ISS_NORAD_ID], [tle_record()])
    assert sum(calls) <= obs.time_fine_chunk


def test_orbit_visibility_matches_eager_geometry(monkeypatch):
    import tabsim.dask.observation as om
    case = dict(telescope='SKA-Low-AA0.5', antennas=4, times=4,
                channels=3, samples=3, point_sources=0, rfi_sources=1)
    lazy = build_observation(case, .001)
    records = [omm_record_from_tle()]
    lazy.addTLESatelliteRFI(np.ones((1, 1, 3))*1e-14, [ISS_NORAD_ID], records)
    actual = lazy.vis_rfi.compute(scheduler='threads', num_workers=2)
    eager = build_observation(case, .001)
    eager.ants_uvw = da.from_array(np.asarray(eager.ants_uvw.compute()), chunks=eager.ants_uvw.chunks)
    eager.bl_uvw = eager.ants_uvw[:, eager.a1, :] - eager.ants_uvw[:, eager.a2, :]
    eager.ants_xyz = da.from_array(dc.itrs_to_gcrs_sf(np.asarray(eager.ITRF),
        np.asarray(eager.times_mjd_fine.compute())+2400000.5), chunks=eager.ants_xyz.chunks)
    def eager_orbits(records, times):
        return da.from_array(tle.get_satellite_positions(records, times.compute()),
                             chunks=(len(records), times.chunks[0], 3))
    monkeypatch.setattr(om, 'satellite_position_blocks', eager_orbits)
    eager.addTLESatelliteRFI(np.ones((1, 1, 3))*1e-14, [ISS_NORAD_ID], records)
    np.testing.assert_allclose(actual, eager.vis_rfi.compute(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('no_w', [False, True])
def test_uvw_and_baselines_preserve_origin_and_no_w(no_w):
    from tabsim.dask.observation import Observation
    from tabsim.jax.coordinates import itrf_to_uvw
    positions = np.array([[5109000., 20000., -3200000.],
                          [5109020., 20010., -3199990.],
                          [5109030., 20030., -3199950.]])
    obs = Observation(latitude=-30., longitude=20., elevation=100.,
        ra=30., dec=-30., ITRF_array=positions.tolist(), times_mjd=60000.+np.arange(8)*2/86400,
        freqs=np.array([150e6,151e6]), SEFD=np.ones(2)*5000, int_time=2,
        n_int_samples=3, max_chunk_MB=.001, no_w=no_w)
    expected = np.array(itrf_to_uvw(positions, np.asarray(obs.gha), -30.))
    if no_w:
        expected[:, :, 2] = 0
    np.testing.assert_allclose(obs.ants_uvw.compute(), expected, rtol=1e-12, atol=1e-9)
    baseline = expected[:, np.asarray(obs.a1)] - expected[:, np.asarray(obs.a2)]
    np.testing.assert_allclose(obs.bl_uvw.compute(), baseline, rtol=1e-12, atol=1e-9)


def test_orbit_records_are_frozen_for_replay():
    record = tle_record()
    times = da.from_array(ISS_EPOCH_JD + np.arange(3)/86400, chunks=2)
    expected = tle.get_satellite_positions([record], np.asarray(times))
    graph = dc.satellite_position_blocks([record], times)
    record['TLE_LINE2'] = 'mutated after graph construction'
    np.testing.assert_allclose(graph.compute(), expected, rtol=1e-13, atol=1e-7)



def test_multiple_orbits_preserve_order_in_bounded_slice():
    from orbit_helpers import GPS_NORAD_ID, GPS_LINE1, GPS_LINE2
    records = [tle_record(GPS_NORAD_ID, GPS_LINE1, GPS_LINE2), tle_record()]
    dates = ISS_EPOCH_JD + np.arange(11)/86400
    graph = dc.satellite_position_blocks(records, da.from_array(dates, chunks=3))
    expected = tle.get_satellite_positions(records, dates[3:5])
    np.testing.assert_allclose(graph[:, 3:5].compute(), expected, rtol=1e-13, atol=1e-7)
    assert not np.allclose(expected[0], expected[1])
