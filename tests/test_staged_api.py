"""Production entry points, resource guards and downstream output contracts."""
import json
from pathlib import Path
from unittest.mock import patch

import dask.array as da
import jax
import numpy as np
import pytest
import xarray as xr

import tabsim.config as config
from benchmarks.harness import build_observation, add_sources
from tabsim.staged import ResourceGuard, select_arrays


@pytest.fixture
def obs():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    case = dict(telescope='SKA-Low-AA0.5', antennas=4, times=4, channels=3,
                samples=3, point_sources=2, rfi_sources=1)
    observation = build_observation(case, .001)
    add_sources(observation, case)
    observation.calculate_vis(random_seed=0)
    yield observation
    jax.config.update('jax_enable_x64', previous)


def test_default_python_stages_and_omits_amplitudes(obs, tmp_path):
    expected = obs.dataset.compute()
    actual = obs.write_to_zarr(tmp_path / 'result.zarr')
    assert 'rfi_stat_A' not in actual
    assert 'rfi_stat_xyz' in actual
    for name in actual.data_vars:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-12)
    status = json.loads((tmp_path / 'result.zarr/staged-status.json').read_text())
    assert status['complete']
    assert set(status['completed']) >= {'vis_ast', 'vis_rfi', 'gains_ants', 'noise_data', 'vis_obs'}


def test_all_rfi_families_are_optional(obs):
    for name in ('rfi_stat_A', 'rfi_sat_A', 'rfi_tle_sat_A'):
        obs.dataset[name] = xr.DataArray(da.ones(2, chunks=1), dims='dummy')
    assert not any(name.endswith('_A') for name in select_arrays(obs.dataset))
    assert {'rfi_stat_A', 'rfi_sat_A', 'rfi_tle_sat_A'} <= select_arrays(obs.dataset, save_rfi_amplitudes=True)
    assert select_arrays(obs.dataset, ['rfi_stat_A']) == {'rfi_stat_A'}


def test_selection_skips_calibration_and_flags(obs, tmp_path):
    events = []
    actual = obs.write_to_zarr(tmp_path / 'selected.zarr', save_arrays=['vis_obs'],
                              progress=lambda event, data: events.append((event, data)))
    assert set(actual.data_vars) == {'vis_obs'}
    writes = [data['variable'] for event, data in events if event == 'stage_complete']
    assert 'vis_calibrated' not in writes and 'flags' not in writes


def test_modified_composition_is_not_silently_discarded(obs, tmp_path):
    obs.dataset['vis_obs'] = obs.dataset.vis_obs + 1
    with pytest.raises(ValueError, match='modified'):
        obs.write_to_zarr(tmp_path / 'bad.zarr')
    assert not (tmp_path / 'bad.zarr').exists()


def test_repeated_write_and_overwrite_protection(obs, tmp_path):
    first = tmp_path / 'first.zarr'
    obs.write_to_zarr(first, save_arrays=['vis_obs'])
    expected = obs.dataset.compute()
    with pytest.raises(ValueError, match='backing'):
        obs.write_to_zarr(first, overwrite=True)
    second = tmp_path / 'second.zarr'
    obs.write_to_zarr(second)
    xr.testing.assert_equal(obs.dataset.compute(), expected)
    with pytest.raises(FileExistsError):
        obs.write_to_zarr(first)


@pytest.mark.parametrize('options', [dict(component_workers=0), dict(component_workers=1.5),
    dict(max_memory_gb=-1), dict(memory_fraction=2), dict(timeout_s=float('nan')),
    dict(disk_reserve_gb=-1)])
def test_invalid_options_preserve_existing_output(obs, tmp_path, options):
    target = tmp_path / 'existing.zarr'
    target.mkdir()
    sentinel = target / 'keep'
    sentinel.write_text('untouched')
    with pytest.raises(ValueError):
        obs.write_to_zarr(target, overwrite=True, **options)
    assert sentinel.read_text() == 'untouched'


def test_host_guard_rejects_before_output(obs, tmp_path):
    target = tmp_path / 'too-small.zarr'
    with pytest.raises(MemoryError, match='RSS'):
        obs.write_to_zarr(target, max_memory_gb=1e-9)
    assert not target.exists()


def test_guard_applies_to_every_component_scheduler(obs, tmp_path, monkeypatch):
    import threading
    seen = set()
    def check(self, *args):
        seen.add(threading.current_thread().name)
    monkeypatch.setattr(ResourceGuard, '_check_task', check)
    obs.write_to_zarr(tmp_path / 'result.zarr', component_workers=2)
    assert len([name for name in seen if name.startswith('ThreadPoolExecutor')]) == 2


def test_runtime_guard_leaves_incomplete_store(obs, tmp_path, monkeypatch):
    def fail(self, *args):
        raise MemoryError('injected runtime guard')
    monkeypatch.setattr(ResourceGuard, '_check_task', fail)
    with pytest.raises(MemoryError, match='runtime guard'):
        obs.write_to_zarr(tmp_path / 'result.zarr')
    assert not json.loads((tmp_path / 'result.zarr/staged-status.json').read_text())['complete']


@pytest.mark.parametrize('zarr_output,ms_output', [(True, False), (True, True), (False, True)])
def test_config_consumers_read_staged_arrays_before_pruning(obs, tmp_path, monkeypatch, zarr_output, ms_output):
    from tabsim.write import MS_REQUIRED_ARRAYS
    called = []
    def ms(ds, *args):
        assert MS_REQUIRED_ARRAYS <= set(ds.variables)
        assert any('open_dataset' in str(key) for key in ds.vis_obs.data.dask)
        called.append('ms')
    def accumulate(ds, *args):
        assert 'vis_rfi' in ds
        called.append('accumulate')
    monkeypatch.setattr(config, 'write_to_ms', ms)
    monkeypatch.setattr('tabsim.write.add_to_ms', accumulate)
    monkeypatch.setattr(config, 'print_signal_specs', lambda *args: called.append('stats'))
    settings = dict(output=dict(zarr=zarr_output, ms=ms_output, overwrite=False,
        flag_data=False, save_arrays=['vis_obs'], accumulate_ms='existing.ms'),
        diagnostics=dict(signal_stats=True))
    config.save_data(obs, settings, tmp_path / 'result.zarr', tmp_path / 'result.ms')
    assert called == (['ms'] if ms_output else []) + ['accumulate', 'stats']
    if zarr_output:
        assert set(obs.dataset.data_vars) == {'vis_obs'}
    else:
        assert not (tmp_path / 'result.zarr').exists()
        assert not list(tmp_path.glob('tabsim-stage-*'))


def test_python_ms_uses_temporary_disk_stage(obs, tmp_path, monkeypatch):
    original = obs.dataset
    def ms(ds, *args, **kwargs):
        assert any('open_dataset' in str(key) for key in ds.vis_obs.data.dask)
        assert 'rfi_stat_A' not in ds
    monkeypatch.setattr('tabsim.dask.observation.write_ms', ms)
    obs.write_to_ms(tmp_path / 'result.ms')
    assert obs.dataset is original
    assert not list(tmp_path.glob('tabsim-stage-*'))


def test_cli_execution_overrides(tmp_path, monkeypatch):
    import sys
    from tabsim.scripts import sim_vis
    path = tmp_path / 'config.yaml'
    path.write_text('dask:\n  component_workers: 1\noutput:\n  save_rfi_amplitudes: true\n')
    captured = {}
    def run(sim_config, **kwargs):
        captured.update(sim_config)
        return None, 'out'
    monkeypatch.setattr(sim_vis, 'run_sim_config', run)
    monkeypatch.setattr(sys, 'argv', ['sim-vis', '-c', str(path), '--max-chunk-mb', '64',
        '--component-workers', '2', '--max-memory-gb', '8', '--memory-fraction', '.6',
        '--timeout-s', '60', '--disk-reserve-gb', '2', '--save-arrays', 'vis_obs',
        '--no-save-rfi-amplitudes', '--no-flag-data', '--signal-stats'])
    sim_vis.main()
    assert captured['dask']['max_chunk_MB'] == 64
    assert captured['dask']['component_workers'] == 2
    assert captured['dask']['max_memory_gb'] == 8
    assert captured['dask']['timeout_s'] == 60
    assert captured['output']['save_arrays'] == ['vis_obs']
    assert captured['output']['save_rfi_amplitudes'] is False
    assert captured['output']['flag_data'] is False
    assert captured['diagnostics']['signal_stats'] is True


@pytest.mark.parametrize('retain_zarr', [False, True])
def test_real_measurement_set_conversion(obs, tmp_path, retain_zarr):
    from casacore.tables import table
    # Use the config's calculation seed, then compare persisted MS values.
    obs.calculate_vis(flags=False)
    expected = obs.dataset.vis_obs.compute().values.reshape(-1, obs.n_freq, 1)
    settings = dict(output=dict(zarr=retain_zarr, ms=True, overwrite=False,
        flag_data=False, save_arrays=['vis_obs']), diagnostics=dict(signal_stats=False))
    config.save_data(obs, settings, tmp_path / 'result.zarr', tmp_path / 'result.ms')
    with table(str(tmp_path / 'result.ms'), ack=False) as ms:
        np.testing.assert_allclose(ms.getcol('DATA'), expected, rtol=1e-6, atol=1e-6)
    if retain_zarr:
        assert set(obs.dataset.data_vars) == {'vis_obs'}


@pytest.mark.parametrize('writer', ['write_to_zarr', 'write_to_ms'])
@pytest.mark.parametrize('destination', ['same', 'parent', 'child'])
def test_overlapping_destinations_preserve_backing_store(obs, tmp_path, writer, destination):
    source = tmp_path / 'source.zarr'
    obs.write_to_zarr(source)
    expected = obs.dataset.vis_obs.compute()
    target = {'same': source, 'parent': tmp_path, 'child': source / 'child'}[destination]
    with pytest.raises(ValueError, match='backing'):
        getattr(obs, writer)(target, overwrite=True)
    xr.testing.assert_equal(obs.dataset.vis_obs.compute(), expected)
    assert json.loads((source / 'staged-status.json').read_text())['complete']


def test_failure_is_latched_after_memory_recovers(tmp_path):
    guard = ResourceGuard(None, .7, None, None, tmp_path, 0)
    guard.abort(MemoryError('first failure'))
    for action in (guard.check, guard._check_task):
        with pytest.raises(MemoryError, match='first failure'):
            action()
    guard.abort(RuntimeError('later failure'))
    with pytest.raises(MemoryError, match='first failure'):
        guard.scheduler({'x': 1}, 'x')


def test_registered_dask_callbacks_are_preserved(obs, tmp_path):
    from dask.callbacks import Callback
    tasks = []
    with Callback(posttask=lambda key, *args: tasks.append(key)):
        obs.write_to_zarr(tmp_path / 'result.zarr', save_arrays=['vis_obs'])
    assert len(tasks) > 10


def test_minimal_profile_keeps_metadata_and_skips_ancillary_graphs(obs, tmp_path):
    import dask
    from tabsim.staged import minimal_arrays, MINIMAL_METADATA
    expected = obs.dataset.vis_obs.compute()
    def forbidden():
        raise AssertionError('Omitted ancillary graph was executed')
    for name in ('rfi_stat_A', 'unused_fine_geometry'):
        obs.dataset[name] = xr.DataArray(da.from_delayed(dask.delayed(forbidden)(),
            shape=(2,), dtype=float), dims='unused')
    assert select_arrays(obs.dataset, output_profile='full') == set(obs.dataset.data_vars)
    actual = obs.write_to_zarr(tmp_path/'minimal.zarr', output_profile='minimal')
    assert set(actual.data_vars) == MINIMAL_METADATA | {'vis_obs'}
    assert set(actual.data_vars) == minimal_arrays(actual)
    np.testing.assert_allclose(actual.vis_obs, expected, rtol=1e-12, atol=1e-12)


def test_profile_and_exact_selection_are_unambiguous(obs, tmp_path):
    from tabsim.staged import minimal_arrays
    assert {'vis_calibrated', 'antenna1'} <= minimal_arrays(obs.dataset, ['vis_calibrated'])
    with pytest.raises(ValueError, match='not both'):
        obs.write_to_zarr(tmp_path/'bad.zarr', output_profile='minimal', save_arrays=['vis_obs'])
    assert not (tmp_path/'bad.zarr').exists()
    with pytest.raises(ValueError, match='output_profile'):
        select_arrays(obs.dataset, output_profile='unknown')
    assert select_arrays(obs.dataset, ['vis_obs']) == {'vis_obs'}


def test_minimal_profile_ms_stages_required_arrays_then_prunes(obs, tmp_path, monkeypatch):
    from tabsim.staged import MINIMAL_METADATA
    from tabsim.write import MS_REQUIRED_ARRAYS
    calls = []
    def consume(dataset, *args):
        assert set(MS_REQUIRED_ARRAYS) <= set(dataset.data_vars)
        assert np.isfinite(dataset.vis_obs.isel(time=0).compute()).all()
        calls.append(1)
    monkeypatch.setattr(config, 'write_to_ms', consume)
    cfg = dict(output=dict(zarr=True, ms=True, overwrite=False, flag_data=False,
                           output_profile='minimal'), diagnostics=dict(signal_stats=False))
    config.save_data(obs, cfg, str(tmp_path/'minimal.zarr'), str(tmp_path/'result.ms'))
    assert calls == [1]
    assert set(obs.dataset.data_vars) == MINIMAL_METADATA | {'vis_obs'}


def test_scalar_diagnostics_share_one_graph_execution(capsys):
    import dask
    calls = []
    @dask.delayed
    def shared():
        calls.append(1)
        return np.full((4,), 3.+0j)
    data = da.from_delayed(shared(), shape=(4,), dtype=complex)
    with dask.config.set(scheduler='synchronous'):
        config.print_signal_specs(data, data, data, data.real > 0)
    assert calls == [1]
    output = capsys.readouterr().out
    assert '3.00 Jy' in output and '0.00 Jy' in output and '100.0 %' in output
