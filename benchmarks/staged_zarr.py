"""Experimental single-store staged writer, with bounded component parallelism."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Lock
import time

import dask
import dask.array as da
import xarray as xr
import zarr

from tabsim.dask.interferometry import apply_gains


def write_staged_observation(obs, directory, *, flags, progress=None, component_workers=2):
    """Write each variable once, then compose into separate arrays in that store.

    Requires the original calculate_vis flag setting and an unmodified composed
    dataset. Metadata preparation is serial; workers execute only deferred data
    writes for distinct arrays. Failed stores remain marked incomplete.
    """
    if component_workers < 1:
        raise ValueError('component_workers must be positive')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'result.zarr'
    marker = directory / 'staged-status.json'
    original = obs.dataset
    stages, opened = [], []
    lock = Lock()
    status = {'complete': False, 'completed': []}

    def record_status():
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(marker)

    # Initialize only eager variables. Lazy arrays are created immediately before
    # their write: decoding unwritten CF datetime arrays can otherwise overflow.
    lazy_names = [name for name in original.variables
                  if isinstance(original[name].data, da.Array)]
    original.drop_vars(lazy_names).to_zarr(
        path, mode='w-', consolidated=False)
    record_status()

    def payload_for(name, array):
        # Preserve xarray encoders and safe-chunk validation. All preparation
        # happens serially; omit coordinates so parallel writes never share them.
        template = original[name]
        array = da.asarray(array).rechunk(template.data.chunks)
        variable = xr.Variable(template.dims, array, attrs=template.attrs,
                               encoding=template.encoding.copy())
        return xr.Dataset({name: variable}, attrs=original.attrs)

    def prepare(name, array):
        return payload_for(name, array).to_zarr(
            path, mode='a', compute=False, consolidated=False)

    def execute(name, graph, *, immediate=False):
        start = time.perf_counter()
        with lock:
            if progress:
                progress('stage_start', {'variable': name})
        if immediate:
            # Keep the Array store graph out of delayed collection conversion.
            # Dask 2026.8 wraps its data roots as tasks on that path, causing
            # whole-component read-ahead with the synchronous scheduler.
            with dask.config.set(scheduler='synchronous'):
                graph.to_zarr(path, mode='a', compute=True, consolidated=False)
        else:
            graph.compute(scheduler='synchronous')
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

    components = ('vis_ast', 'vis_rfi', 'gains_ants', 'noise_data')
    composed = ('vis_obs', 'vis_calibrated', 'flags')
    try:
        # Separate scheduler invocations prevent cross-component dependency-cache
        # coupling. Each worker has one synchronous task stream, not its own pool.
        graphs = [(name, prepare(name, original[name].data)) for name in components]
        with ThreadPoolExecutor(max_workers=component_workers) as pool:
            futures = [pool.submit(execute, name, graph) for name, graph in graphs]
            for future in futures:
                future.result()
        ast, rfi, gains, noise = (reopen(name) for name in components)
        a1, a2 = original.antenna1.data, original.antenna2.data
        observed = apply_gains(ast, rfi, gains, a1, a2).rechunk(ast.chunks) + noise
        observed = save('vis_obs', observed)
        calibrated = save('vis_calibrated', apply_gains(
            observed, da.zeros_like(observed), 1.0 / gains, a1, a2))
        sigma = original.noise_std.data
        if flags:
            threshold = (3.0 * sigma[None, None, :] if bool((sigma.mean() > 0).compute())
                         else 3.0 * da.std(ast, axis=0)[None, ...])
            flag_array = da.abs(calibrated - ast) > threshold
        else:
            flag_array = da.zeros_like(calibrated, dtype=original.flags.dtype)
        save('flags', flag_array)

        # Non-Dask variables were written during initialization; all remaining
        # ancillary arrays get exactly one data write, with their xarray encoding.
        for name in original.variables:
            if name not in components + composed and isinstance(original[name].data, da.Array):
                execute(name, payload_for(name, original[name].data), immediate=True)
        zarr.consolidate_metadata(str(path))
        status['complete'] = True
        record_status()
        if progress:
            progress('single_store_complete', {})
        return stages
    finally:
        for dataset in opened:
            dataset.close()
