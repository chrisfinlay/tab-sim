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


def test_single_store_encodings_and_one_write(tmp_path, monkeypatch):
    import json
    import dask.array as da
    case = dict(telescope='SKA-Low-AA0.5', antennas=4, times=5, channels=3,
                samples=3, point_sources=2, rfi_sources=1)
    obs = build_observation(case, .001)
    add_sources(obs, case)
    obs.calculate_vis(random_seed=0)
    obs.dataset['encoded_extra'] = xr.DataArray(da.from_array(
        np.array([1., 2., np.nan, 4., 5.]), chunks=(2,)), dims='time',
        attrs={'description': 'packing test'})
    obs.dataset.encoded_extra.encoding.update(dtype='int16', scale_factor=.1, _FillValue=-999)
    obs.dataset['date_extra'] = xr.DataArray(da.from_array(
        np.arange('2026-01-01', '2026-01-06', dtype='datetime64[D]'), chunks=2), dims='time')
    obs.dataset.to_zarr(tmp_path / 'reference.zarr')
    writes = []
    original_write = xr.Dataset.to_zarr
    def counted(dataset, *args, **kwargs):
        if kwargs.get('mode') == 'a':
            writes.extend(dataset.data_vars)
            if not set(dataset.data_vars) <= {'vis_ast', 'vis_rfi', 'gains_ants', 'noise_data'}:
                assert kwargs.get('compute') is True
        return original_write(dataset, *args, **kwargs)
    monkeypatch.setattr(xr.Dataset, 'to_zarr', counted)
    target = tmp_path / 'staged'
    write_staged_observation(obs, target, flags=True, component_workers=2)
    assert set(target.iterdir()) == {target / 'result.zarr', target / 'staged-status.json'}
    assert len(writes) == len(set(writes))
    assert json.loads((target / 'staged-status.json').read_text())['complete']
    with xr.open_zarr(tmp_path / 'reference.zarr') as expected, xr.open_zarr(target / 'result.zarr') as actual:
        xr.testing.assert_allclose(actual, expected)
        assert actual.attrs == expected.attrs


def test_component_failure_leaves_incomplete_store(tmp_path, monkeypatch):
    import json
    import dask
    case = dict(telescope='SKA-Low-AA0.5', antennas=4, times=4, channels=3,
                samples=3, point_sources=0, rfi_sources=0)
    obs = build_observation(case, .001)
    add_sources(obs, case)
    obs.calculate_vis()
    original_write = xr.Dataset.to_zarr
    def fail():
        raise RuntimeError('injected component failure')
    def injected(dataset, *args, **kwargs):
        graph = original_write(dataset, *args, **kwargs)
        if kwargs.get('mode') == 'a' and set(dataset.data_vars) == {'vis_rfi'}:
            return dask.delayed(fail)()
        return graph
    monkeypatch.setattr(xr.Dataset, 'to_zarr', injected)
    with pytest.raises(RuntimeError, match='injected component failure'):
        write_staged_observation(obs, tmp_path, flags=True, component_workers=2)
    status = json.loads((tmp_path / 'staged-status.json').read_text())
    assert not status['complete']
    assert 'vis_obs' not in status['completed']
    assert not (tmp_path / 'result.zarr' / '.zmetadata').exists()


def test_components_overlap_before_composition(tmp_path, monkeypatch):
    from threading import Barrier
    case = dict(telescope='SKA-Low-AA0.5', antennas=4, times=4, channels=3,
                samples=3, point_sources=0, rfi_sources=0)
    obs = build_observation(case, .001)
    add_sources(obs, case)
    obs.calculate_vis()
    barrier = Barrier(2, timeout=10)
    original_write = xr.Dataset.to_zarr
    def wait_for_other_worker():
        barrier.wait()
    def wrapped(dataset, *args, **kwargs):
        graph = original_write(dataset, *args, **kwargs)
        if kwargs.get('mode') == 'a' and set(dataset.data_vars) in ({'vis_ast'}, {'vis_rfi'}):
            def compute_after_barrier():
                wait_for_other_worker()
                graph.compute(scheduler='synchronous')
            return dask.delayed(compute_after_barrier)()
        return graph
    monkeypatch.setattr(xr.Dataset, 'to_zarr', wrapped)
    write_staged_observation(obs, tmp_path, flags=True, component_workers=2)


def test_composition_writes_before_reading_whole_component(tmp_path, monkeypatch):
    import zarr
    case = dict(telescope='SKA-Low-AA0.5', antennas=4, times=8, channels=8,
                samples=3, point_sources=0, rfi_sources=0)
    obs = build_observation(case, .000288)  # exact minimum: preserve the 1x1 read-ahead probe
    add_sources(obs, case)
    obs.calculate_vis(random_seed=0)
    active = False
    reads = 0
    first_write_reads = []
    getitem, setitem = zarr.Array.__getitem__, zarr.Array.__setitem__
    def read(array, key):
        nonlocal reads
        if active and array.path in {'vis_ast', 'vis_rfi', 'gains_ants', 'noise_data'}:
            reads += 1
        return getitem(array, key)
    def write(array, key, value):
        if active and array.path == 'vis_obs' and not first_write_reads:
            first_write_reads.append(reads)
        return setitem(array, key, value)
    def progress(event, details):
        nonlocal active
        if details.get('variable') == 'vis_obs':
            active = event == 'stage_start'
    monkeypatch.setattr(zarr.Array, '__getitem__', read)
    monkeypatch.setattr(zarr.Array, '__setitem__', write)
    write_staged_observation(obs, tmp_path, flags=True, progress=progress)
    # Four inputs per tile; allow small read-ahead, but not a full 64-tile cube.
    assert first_write_reads and 0 < first_write_reads[0] <= 12


@pytest.mark.parametrize('save_arrays', [None, ['vis_obs']])
def test_composition_failure_leaves_store_incomplete(tmp_path, monkeypatch, save_arrays):
    import json
    case = dict(telescope='SKA-Low-AA0.5', antennas=4, times=4, channels=3,
                samples=3, point_sources=0, rfi_sources=0)
    obs = build_observation(case, .001)
    add_sources(obs, case)
    obs.calculate_vis()
    original_write = xr.Dataset.to_zarr
    def injected(dataset, *args, **kwargs):
        if set(dataset.data_vars) == {'vis_obs'}:
            raise RuntimeError('injected composition failure')
        return original_write(dataset, *args, **kwargs)
    monkeypatch.setattr(xr.Dataset, 'to_zarr', injected)
    with pytest.raises(RuntimeError, match='injected composition failure'):
        write_staged_observation(obs, tmp_path, flags=True, save_arrays=save_arrays)
    status = json.loads((tmp_path / 'staged-status.json').read_text())
    assert not status['complete']
    assert set(status['completed']) == {'vis_ast', 'vis_rfi', 'gains_ants', 'noise_data'}
    assert not (tmp_path / 'result.zarr' / '.zmetadata').exists()


@pytest.mark.parametrize('save_arrays', [
    ['vis_obs'], ['vis_calibrated'], ['flags'], ['vis_ast'], ['noise_std'], [],
])
@pytest.mark.parametrize('unity', [True, False])
def test_selected_arrays_match_reference(tmp_path, monkeypatch, save_arrays, unity):
    import json
    import tabsim.staged as writer
    case = dict(telescope='SKA-Low-AA0.5', antennas=4, times=4, channels=3,
                samples=3, point_sources=2, rfi_sources=1)
    obs = build_observation(case, .001)
    add_sources(obs, case)
    if unity:
        obs.gains_ants = dask.array.ones_like(obs.gains_ants)
    else:
        obs.gains_ants = obs.gains_ants * (1.2 + .3j)
    obs.calculate_vis(random_seed=7)
    expected = obs.dataset.compute()
    calls = []
    apply = writer.apply_gains
    def counted(*args, **kwargs):
        calls.append(1)
        return apply(*args, **kwargs)
    monkeypatch.setattr(writer, 'apply_gains', counted)
    write_staged_observation(obs, tmp_path, flags=True, save_arrays=save_arrays)
    with xr.open_zarr(tmp_path / 'result.zarr') as actual:
        assert set(actual.data_vars) == set(save_arrays)
        assert set(actual.coords) == set(expected.coords)
        assert actual.attrs == expected.attrs
        for name in actual.variables:
            xr.testing.assert_allclose(actual[name], expected[name])
    status = json.loads((tmp_path / 'staged-status.json').read_text())
    assert status['complete']
    if set(save_arrays) & {'vis_calibrated', 'flags'}:
        assert status['calibration_skipped'] == unity
        assert len(calls) == (1 if unity else 2)
        if unity and 'vis_calibrated' not in save_arrays:
            assert 'vis_calibrated' not in status['completed']
    elif 'vis_obs' in save_arrays:
        assert len(calls) == 1
    else:
        assert not calls
        assert not (set(status['completed']) & {'vis_obs', 'vis_calibrated'})


@pytest.mark.parametrize('gain', [1.0, 1.0 + 1e-12])
def test_unity_calibrated_output_skips_inverse_kernel(tmp_path, monkeypatch, gain):
    import tabsim.staged as writer
    case = dict(telescope='SKA-Low-AA0.5', antennas=4, times=4, channels=3,
                samples=3, point_sources=1, rfi_sources=1)
    obs = build_observation(case, .001)
    add_sources(obs, case)
    obs.gains_ants = dask.array.ones_like(obs.gains_ants) * gain
    obs.calculate_vis(random_seed=7)
    apply = writer.apply_gains
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return apply(*args, **kwargs)
    monkeypatch.setattr(writer, 'apply_gains', counted)
    write_staged_observation(obs, tmp_path, flags=True,
                             save_arrays=['vis_obs', 'vis_calibrated'])
    assert len(calls) == (1 if gain == 1 else 2)
    with xr.open_zarr(tmp_path / 'result.zarr') as actual:
        if gain == 1:
            np.testing.assert_array_equal(actual.vis_obs.values, actual.vis_calibrated.values)
        else:
            assert np.max(np.abs(actual.vis_obs.values - actual.vis_calibrated.values)) > 0


def test_disabled_flags_need_no_components(tmp_path, monkeypatch):
    import tabsim.staged as writer
    case = dict(telescope='SKA-Low-AA0.5', antennas=4, times=4, channels=3,
                samples=3, point_sources=0, rfi_sources=0)
    obs = build_observation(case, .001)
    obs.calculate_vis(flags=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('No visibility computation is required for disabled flags')
    monkeypatch.setattr(writer, 'apply_gains', forbidden)
    stages = write_staged_observation(obs, tmp_path, flags=False, save_arrays=['flags'])
    assert not ({s['variable'] for s in stages} & {'vis_ast', 'vis_rfi', 'noise_data', 'gains_ants'})
    with xr.open_zarr(tmp_path / 'result.zarr') as actual:
        assert set(actual.data_vars) == {'flags'}
        assert not actual.flags.values.any()


@pytest.mark.parametrize('selection', [['typo'], 'vis_obs'])
def test_invalid_selection_fails_before_writing(tmp_path, selection):
    case = dict(telescope='SKA-Low-AA0.5', antennas=4, times=4, channels=3,
                samples=3, point_sources=0, rfi_sources=0)
    obs = build_observation(case, .001)
    obs.calculate_vis()
    target = tmp_path / 'new'
    with pytest.raises(ValueError):
        write_staged_observation(obs, target, flags=True, save_arrays=selection)
    assert not target.exists()
