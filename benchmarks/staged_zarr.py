"""Benchmark adapter for the production staged writer."""
from pathlib import Path
import shutil
from tabsim.staged import write_staged_observation as _write


def write_staged_observation(obs, directory, *, flags, progress=None, component_workers=2, save_arrays=None):
    directory = Path(directory)
    try:
        return _write(obs, directory / 'result.zarr', flags=flags, progress=progress,
                      component_workers=component_workers, save_arrays=save_arrays,
                      disk_reserve_gb=0)
    finally:
        marker = directory / 'result.zarr' / 'staged-status.json'
        if marker.exists():
            shutil.copyfile(marker, directory / 'staged-status.json')
