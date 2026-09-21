"""Single-store, disk-staged simulation output using public Dask/xarray APIs."""
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import math
import psutil
from dask.callbacks import Callback
from pathlib import Path
from threading import Lock
import time

import dask
import dask.array as da
import xarray as xr
import zarr

from tabsim.dask.interferometry import apply_gains


def write_staged_observation(obs, path, *, flags, progress=None, component_workers=2,
                             save_arrays=None, overwrite=False, recompose=True,
                             max_memory_gb=None, memory_fraction=0.7,
                             max_device_memory_gb=None, timeout_s=None,
                             disk_reserve_gb=1.0):
    """Write each variable once, then compose into separate arrays in that store.

    Requires the original calculate_vis flag setting and an unmodified composed
    dataset. Metadata preparation is serial; workers execute only deferred data
    writes for distinct arrays. Failed stores remain marked incomplete.

    save_arrays selects data variables retained in the final store (None keeps
    all); coordinates and dataset attributes are always preserved. Dependencies
    are staged in the same store and removed only after all consumers finish.
    Exactly unity gains bypass inverse-gain calibration. If the calibrated array
    is requested it is still saved, with the observed values and its own encoding.
    """
    validate_limits(component_workers, max_memory_gb, memory_fraction,
                    max_device_memory_gb, timeout_s, disk_reserve_gb)
    original = obs.dataset
    if isinstance(save_arrays, str):
        raise ValueError('save_arrays must be a collection of data variable names, not a string')
    selected = set(original.data_vars if save_arrays is None else save_arrays)
    unknown = selected - set(original.data_vars)
    if unknown:
        raise ValueError(f'Unknown data arrays: {sorted(unknown)}')
    components = ('vis_ast', 'vis_rfi', 'gains_ants', 'noise_data')
    composed = ('vis_obs', 'vis_calibrated', 'flags')
    need_calibrated = recompose and ('vis_calibrated' in selected or ('flags' in selected and flags))
    need_observed = recompose and ('vis_obs' in selected or need_calibrated)
    required = selected | set(original.coords)
    if need_observed:
        required.update(components)
        required.add('vis_obs')
    if need_calibrated:
        required.add('vis_calibrated')
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Conservative admission: dependencies coexist with outputs until consumers finish.
    expected_bytes = sum(original[name].nbytes for name in required)
    free = shutil.disk_usage(path.parent).free
    if free < expected_bytes * 1.1 + disk_reserve_gb * 2**30:
        raise OSError('Insufficient free disk space for staged arrays and disk reserve')
    guard = ResourceGuard(max_memory_gb, memory_fraction, max_device_memory_gb,
                          timeout_s, path.parent, disk_reserve_gb)
    guard.check()
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ValueError('Output must be a directory, not a file or symbolic link')
        shutil.rmtree(path)
    marker = path / 'staged-status.json'
    stages, opened = [], []
    lock = Lock()
    status = {'complete': False, 'completed': [], 'save_arrays': sorted(selected),
              'temporary_removed': []}

    def record_status():
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(marker)

    # Initialize only eager variables. Lazy arrays are created immediately before
    # their write: decoding unwritten CF datetime arrays can otherwise overflow.
    lazy_names = [name for name in original.variables
                  if isinstance(original[name].data, da.Array)]
    original.drop_vars(set(lazy_names) | (set(original.variables) - required)).to_zarr(
        path, mode='w-', consolidated=False)
    record_status()

    def payload_for(name, array):
        # Preserve xarray encoders and safe-chunk validation. All preparation
        # happens serially; omit coordinates so parallel writes never share them.
        template = original[name]
        array = da.asarray(array)
        if isinstance(template.data, da.Array):
            array = array.rechunk(template.data.chunks)
        variable = xr.Variable(template.dims, array, attrs=template.attrs,
                               encoding=template.encoding.copy())
        return xr.Dataset({name: variable}, attrs=original.attrs)

    def prepare(name, array):
        return payload_for(name, array).to_zarr(
            path, mode='a', compute=False, consolidated=False)

    def execute(name, graph, *, immediate=False):
        try:
            return execute_checked(name, graph, immediate=immediate)
        except BaseException as error:
            guard.abort(error)
            raise

    def execute_checked(name, graph, *, immediate=False):
        guard.check()
        start = time.perf_counter()
        with lock:
            if progress:
                progress('stage_start', {'variable': name})
        if immediate:
            # Keep the Array store graph out of delayed collection conversion.
            # Dask 2026.8 wraps its data roots as tasks on that path, causing
            # whole-component read-ahead with the synchronous scheduler.
            with dask.config.set(scheduler=guard.scheduler):
                graph.to_zarr(path, mode='a', compute=True, consolidated=False)
        else:
            graph.compute(scheduler=guard.scheduler)
        guard.check()
        with lock:
            item = {'variable': name, 'elapsed_s': time.perf_counter() - start}
            stages.append(item)
            status['completed'].append(name)
            record_status()
            if progress:
                progress('stage_complete', item)

    def reopen(name):
        dataset = xr.open_zarr(path, chunks={}, consolidated=False,
                               drop_variables=[key for key in original.variables if key != name])
        opened.append(dataset)
        return dataset[name].data

    def save(name, array):
        execute(name, payload_for(name, array), immediate=True)
        return reopen(name)

    try:
        # Separate scheduler invocations prevent cross-component dependency-cache
        # coupling. Each worker has one synchronous task stream, not its own pool.
        graphs = [(name, prepare(name, original[name].data)) for name in components
                  if name in required and recompose]
        with ThreadPoolExecutor(max_workers=component_workers) as pool:
            futures = [pool.submit(execute, name, graph) for name, graph in graphs]
            for future in futures:
                future.result()
        if need_observed:
            ast, rfi, gains, noise = (reopen(name) for name in components)
            a1, a2 = original.antenna1.data, original.antenna2.data
            observed = apply_gains(ast, rfi, gains, a1, a2).rechunk(ast.chunks) + noise
            observed = save('vis_obs', observed)
        if need_calibrated:
            # Read persisted gain chunks, reducing to one scalar without loading
            # a visibility cube. Exact equality avoids changing near-unity gains.
            unity = bool(da.all(gains == 1).compute(scheduler=guard.scheduler))
            status['calibration_skipped'] = unity
            if unity:
                calibrated = observed
                if progress:
                    progress('calibration_skipped', {'reason': 'unity_gains'})
                if 'vis_calibrated' in selected:
                    calibrated = save('vis_calibrated', calibrated)
            else:
                calibrated = save('vis_calibrated', apply_gains(
                    observed, da.zeros_like(observed), 1.0 / gains, a1, a2))
        if recompose and 'flags' in selected:
            if flags:
                sigma = original.noise_std.data
                threshold = (3.0 * sigma[None, None, :] if bool((sigma.mean() > 0).compute(scheduler=guard.scheduler))
                             else 3.0 * da.std(ast, axis=0)[None, ...])
                flag_array = da.abs(calibrated - ast) > threshold
            else:
                flag_array = da.zeros(original.flags.shape, chunks=original.flags.data.chunks,
                                      dtype=original.flags.dtype)
            save('flags', flag_array)

        # Non-Dask variables were written during initialization; all remaining
        # ancillary arrays get exactly one data write, with their xarray encoding.
        for name in original.variables:
            if name in required and (not recompose or name not in components + composed) and isinstance(original[name].data, da.Array):
                execute(name, payload_for(name, original[name].data), immediate=True)
        # Release readers before deleting scratch arrays. On failure before this
        # point, preserve completed dependencies for diagnosis/recovery.
        for dataset in opened:
            dataset.close()
        opened.clear()
        group = zarr.open_group(str(path), mode='a')
        for name in sorted(required - selected - set(original.coords)):
            if name in group:
                del group[name]
                status['temporary_removed'].append(name)
        guard.check()
        zarr.consolidate_metadata(str(path))
        status['complete'] = True
        record_status()
        if progress:
            progress('single_store_complete', {})
        return stages
    finally:
        for dataset in opened:
            dataset.close()


def validate_limits(component_workers=2, max_memory_gb=None, memory_fraction=0.7,
                    max_device_memory_gb=None, timeout_s=None, disk_reserve_gb=1.0):
    if isinstance(component_workers, bool) or not isinstance(component_workers, int) or component_workers < 1:
        raise ValueError('component_workers must be a positive integer')
    for name, value in [('max_memory_gb', max_memory_gb),
                        ('max_device_memory_gb', max_device_memory_gb), ('timeout_s', timeout_s)]:
        if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be finite and positive, or None')
    if not 0 < memory_fraction <= 1 or not math.isfinite(memory_fraction):
        raise ValueError('memory_fraction must be in (0, 1]')
    if not math.isfinite(disk_reserve_gb) or disk_reserve_gb < 0:
        raise ValueError('disk_reserve_gb must be finite and nonnegative')


class ResourceGuard:
    """Task-boundary guards, not allocator caps: a running task may overshoot.

    Host usage is process RSS (including existing allocations). The automatic
    ceiling adds memory_fraction of currently available RAM to baseline RSS.
    Device usage is live JAX allocations, excluding its reserved allocator pool.
    GPU allocator limits must be configured before initializing JAX.
    """
    def __init__(self, max_memory_gb, memory_fraction, max_device_memory_gb,
                 timeout_s, directory, disk_reserve_gb):
        self.callbacks = tuple(Callback.active)
        self.process = psutil.Process()
        self.host_limit = (max_memory_gb * 2**30 if max_memory_gb is not None else
                           self.process.memory_info().rss + memory_fraction * psutil.virtual_memory().available)
        self.device_limit = None if max_device_memory_gb is None else max_device_memory_gb * 2**30
        self.deadline = None if timeout_s is None else time.monotonic() + timeout_s
        self.directory, self.reserve = directory, disk_reserve_gb * 2**30
        if self.device_limit is not None:
            import jax
            self.devices = jax.local_devices()
            if any(d.memory_stats() is None or 'bytes_in_use' not in d.memory_stats() for d in self.devices):
                raise ValueError('Device memory guard requires JAX device memory statistics')
        self.lock = Lock()
        self.abort_lock = Lock()
        self.failure = None
        self.last_check = 0.0

    def abort(self, error):
        with self.abort_lock:
            if self.failure is None:
                self.failure = error

    def scheduler(self, graph, keys, **kwargs):
        # Pass callbacks explicitly: Dask's global callback context is shared
        # across threads, so overlapping scheduler invocations can lose it.
        kwargs['callbacks'] = list(self.callbacks) + [(None, None, self._check_task, self._check_task, None)]
        self.check()
        return dask.get(graph, keys, **kwargs)

    def _check_task(self, *args):
        if self.failure is not None:
            raise self.failure
        with self.lock:
            now = time.monotonic()
            if now - self.last_check >= 0.2:
                self.check()
                self.last_check = now

    def check(self):
        if self.failure is not None:
            raise self.failure
        if self.process.memory_info().rss > self.host_limit:
            raise MemoryError('Staged writer exceeded host RSS budget; reduce chunk size or component_workers')
        if self.device_limit is not None:
            if any(d.memory_stats()['bytes_in_use'] > self.device_limit for d in self.devices):
                raise MemoryError('Staged writer exceeded live device memory budget')
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise TimeoutError('Staged writer exceeded timeout_s')
        if shutil.disk_usage(self.directory).free < self.reserve:
            raise OSError('Staged writer reached disk space reserve')


MINIMAL_METADATA = frozenset(('antenna1', 'antenna2', 'ants_itrf', 'bl_uvw',
                              'time_idx', 'noise_std', 'SEFD'))
VISIBILITY_PRODUCTS = frozenset(('vis_obs', 'vis_calibrated', 'vis_ast', 'vis_rfi',
                                'noise_data', 'flags'))


def minimal_arrays(dataset, products=('vis_obs',)):
    """Requested visibility products plus baseline/geometry/noise metadata.

    Coordinates and attributes are retained by the writer automatically. This
    is a standalone Zarr selection, not the full Measurement Set input schema.
    """
    if isinstance(products, str):
        raise ValueError('products must be a collection of visibility names')
    products = set(products)
    if not products or products - VISIBILITY_PRODUCTS:
        raise ValueError('Choose one or more visibility products for minimal output')
    selected = products | MINIMAL_METADATA
    missing = selected - set(dataset.data_vars)
    if missing:
        raise ValueError(f'Minimal output requires missing arrays: {sorted(missing)}')
    return selected


def select_arrays(dataset, save_arrays=None, save_rfi_amplitudes=False, output_profile=None):
    """Select exact data variable names. Explicit selection overrides the default."""
    if output_profile not in (None, 'full', 'minimal'):
        raise ValueError('output_profile must be full, minimal, or None')
    if output_profile is not None:
        if save_arrays is not None:
            raise ValueError('Choose output_profile or exact save_arrays, not both')
        if output_profile == 'minimal':
            return minimal_arrays(dataset)
        return set(dataset.data_vars)
    if isinstance(save_arrays, str):
        raise ValueError('save_arrays must be a list of names, not a string')
    selected = set(dataset.data_vars if save_arrays is None else save_arrays)
    if save_arrays is None and not save_rfi_amplitudes:
        selected = {name for name in selected if not (name.startswith('rfi_') and name.endswith('_A'))}
    unknown = selected - set(dataset.data_vars)
    if unknown:
        raise ValueError(f'Unknown data arrays: {sorted(unknown)}')
    return selected


def prune_store(path, selected):
    """Remove temporary data variables only after all external consumers finish."""
    with xr.open_zarr(path, chunks={}) as dataset:
        remove = set(dataset.data_vars) - set(selected)
    group = zarr.open_group(str(path), mode='a')
    for name in remove:
        del group[name]
    zarr.consolidate_metadata(str(path))
    marker = Path(path) / 'staged-status.json'
    if marker.exists():
        status = json.loads(marker.read_text())
        status['save_arrays'] = sorted(selected)
        status['temporary_removed'] = sorted(set(status['temporary_removed']) | remove)
        marker.write_text(json.dumps(status, indent=2))
