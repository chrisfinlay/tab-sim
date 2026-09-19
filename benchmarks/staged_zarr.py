"""Experimental disk boundaries for capacity diagnosis, not a production writer.

Caller supplies the flag option used for calculate_vis. Noise is persisted from
that calculation, including any seed override. Separate stores deliberately
avoid overwriting arrays still used by the composition graph.
"""
import gc
from pathlib import Path
import time

import dask.array as da
import xarray as xr

from tabsim.dask.interferometry import apply_gains


def write_staged_observation(obs, directory, *, flags, progress=None):
    """Write components, compose from disk, then write the complete dataset.

    This diagnostic assumes an unmodified dataset from calculate_vis(flags=...).
    It does not yet support arbitrary user changes to composed dataset variables.
    Temporary component stores remain for measuring their storage cost; caller
    owns cleanup. Every store operation finishes before the next starts.
    """
    directory = Path(directory)
    staging = directory / 'components'
    staging.mkdir(parents=True, exist_ok=False)
    original = obs.dataset
    replacement = {}
    opened = []
    stages = []

    def save(name, array):
        start = time.perf_counter()
        if progress:
            progress('stage_start', {'variable': name})
        path = staging / (name + '.zarr')
        original[name].copy(data=array).to_dataset(name=name).to_zarr(path, mode='w-')
        # Empty chunks mapping uses the stored chunk layout, preserving the
        # original tile boundaries rather than choosing new automatic chunks.
        reopened = xr.open_zarr(path, chunks={})
        opened.append(reopened)
        replacement[name] = reopened[name].data
        elapsed = time.perf_counter() - start
        stages.append({'variable': name, 'elapsed_s': elapsed})
        if progress:
            progress('stage_complete', stages[-1])
        gc.collect()
        return replacement[name]

    try:
        for name in ('vis_ast', 'vis_rfi', 'gains_ants', 'noise_data'):
            save(name, original[name].data)

        ast, rfi, gains, noise = (replacement[name] for name in
                                 ('vis_ast', 'vis_rfi', 'gains_ants', 'noise_data'))
        a1, a2 = original.antenna1.data, original.antenna2.data
        observed = apply_gains(ast, rfi, gains, a1, a2).rechunk(ast.chunks) + noise
        observed = save('vis_obs', observed)
        calibrated = apply_gains(observed, da.zeros_like(observed), 1.0 / gains, a1, a2)
        calibrated = save('vis_calibrated', calibrated)
        sigma = original.noise_std.data
        if flags:
            if bool((sigma.mean() > 0).compute()):
                flag_array = da.abs(calibrated - ast) > 3.0 * sigma[None, None, :]
            else:
                flag_array = da.abs(calibrated - ast) > 3.0 * da.std(ast, axis=0)[None, ...]
        else:
            flag_array = da.zeros_like(calibrated, dtype=original.flags.dtype)
        save('flags', flag_array)

        # All visibility products now refer only to stored component arrays.
        # Ancillary variables and complete metadata are retained for equivalence.
        final = original.copy()
        for name, array in replacement.items():
            final[name] = original[name].copy(data=array)
        if progress:
            progress('final_write_start', {})
        start = time.perf_counter()
        final.to_zarr(directory / 'result.zarr', mode='w-')
        stages.append({'variable': 'complete_dataset', 'elapsed_s': time.perf_counter() - start})
        if progress:
            progress('final_write_complete', stages[-1])
        return stages
    finally:
        for dataset in opened:
            dataset.close()
