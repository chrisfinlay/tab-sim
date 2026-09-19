"""Opt-in pytest plugin to run the staged experiment under existing supervision.

Set PYTEST_PLUGINS=benchmarks.staged_plugin for an experimental capacity run.
This changes the benchmark workload and must never be compared as identical
output-I/O timing without accounting for the single-store stage boundaries.
"""
from pathlib import Path
import shutil
import time


def pytest_configure(config):
    import pytest
    if not config.getoption('--capacity', default=False):
        raise pytest.UsageError('The staged experiment is capacity-only; do not compare it as an ordinary benchmark')
    if config.getoption('--memory-model') != 'staged':
        raise pytest.UsageError('Use --memory-model staged for single-store staged admission')
    from benchmarks import harness

    def staged_simulation(case, mode, chunk_mb, directory, progress=None):
        import pytest
        from benchmarks.cases import estimates
        from benchmarks.staged_zarr import write_staged_observation
        if mode != 'zarr':
            raise ValueError('The staged experiment supports Zarr only')
        disk = shutil.disk_usage(directory)
        required = estimates(case, mode, chunk_mb)['disk_plan_bytes']
        if required > disk.free - max(2 * 2**30, .05 * disk.total):
            pytest.skip('Staged experiment needs space for the complete single store')
        start = time.perf_counter()
        if progress:
            progress('setup_start', {'experiment': 'staged-zarr-v2'})
        obs = harness.build_observation(case, chunk_mb)
        setup = time.perf_counter()
        harness.add_sources(obs, case)
        obs.calculate_vis()
        graph = time.perf_counter()
        stages = write_staged_observation(obs, directory, flags=True, progress=progress,
                                          component_workers=config.getoption("--workers"))
        end = time.perf_counter()
        sizes = {name: sum(p.stat().st_size for p in (Path(directory) / name).rglob('*') if p.is_file())
                 for name in ('result.zarr',)}
        return {'setup_s': setup-start, 'graph_build_s': graph-setup,
                'simulation_and_first_write_s': end-graph, 'zarr_to_ms_s': None,
                'total_s': end-start, 'experimental_stages': stages,
                'experimental_output_bytes': sizes, 'experiment': 'staged-zarr-v2',
                'actual_chunks': {'time': obs.time_chunk, 'frequency': obs.freq_chunk,
                                  'baseline': obs.bl_chunk, 'integration': obs.n_int_samples}}

    harness.simulation = staged_simulation
