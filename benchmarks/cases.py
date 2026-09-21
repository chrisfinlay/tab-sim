"""Recorded workloads and allocation-aware planning estimates (not memory bounds)."""
import hashlib
import json
import math
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures/cases.json"
CASES = json.loads(FIXTURE.read_text())
MODES = ("astro-kernel", "rfi-kernel", "zarr", "ms", "zarr-ms")


def fixture_hash():
    return hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def planned_chunks(case, chunk_mb):
    """Mirror the strict nominal planner without importing a numerical backend."""
    def factors(n):
        return sorted({v for k in range(1, math.isqrt(n) + 1) if n % k == 0
                       for v in (k, n // k)})
    unit = 16 * case['samples'] * (case['antennas'] * (case['antennas'] - 1) // 2)
    candidates = [(t, f) for t in factors(case['times']) for f in factors(case['channels'])
                  if unit * t * f <= math.floor(chunk_mb * 1e6)]
    if not candidates:
        raise ValueError(f'No feasible nominal tile: minimum {unit} bytes exceeds chunk limit')
    return max(candidates, key=lambda tf: (tf[0] * tf[1], -tf[0]))


def estimates(case, mode, chunk_mb=16, workers=1, memory_model="conservative", device="cpu"):
    """Plan current chunked Zarr; keep conservative whole-array MS/kernel plans.

    Includes full baseline geometry, eager antenna geometry, Fourier-gain mode
    temporaries and a graph allowance. Coefficients are headroom, not an XLA bound.
    The supervisor is required for capacity experiments and enforces actual RSS.
    """
    a, t, f, i = (case[k] for k in ('antennas', 'times', 'channels', 'samples'))
    b = a * (a - 1) // 2
    if mode.endswith('kernel'):
        t, f = min(t, 8), min(f, 16)
    cube = 16 * t * b * f
    source = 8 * max(case['rfi_sources'], 1) * t * i * a * f
    legacy = 512 * 2**20 + 10 * cube + 4 * source
    chunk_error = None
    try:
        ct, cf = planned_chunks(case, chunk_mb)
    except ValueError as error:
        ct, cf = 1, 1
        chunk_error = str(error)
    geometry = 24 * t * i * b
    antenna_geometry = 24 * t * i * a
    tile = 16 * ct * cf * b * i
    gain_tile = 8 * 1000 * ct * a
    blocks = math.ceil(t / ct) * math.ceil(f / cf)
    graph = blocks * (256 + 32 * (case['point_sources'] + case['rfi_sources'])) * 1024
    host = (1024 * 2**20 + 6 * geometry + 4 * antenna_geometry + 6 * 1000 * a * 8
            + graph + workers * (16 * tile + 6 * gain_tile + 8 * (8 * case["rfi_sources"] * ct * i * a * cf))) if mode == 'zarr' and memory_model in ('chunked', 'staged') else legacy
    # Empirical CPU retention allowance: initial capacity probes exceeded the
    # tile-only estimate. This is headroom, not proof of a full-cube allocation.
    retention = cube if mode == 'zarr' and memory_model == 'chunked' and device == 'cpu' else 0
    host += retention
    # Five complex cubes (including noise), flags, geometry, source/gain products,
    # plus conservative metadata/compression headroom. Never assume compression.
    disk = 0 if mode.endswith('kernel') else 6 * cube + 2 * geometry + 4 * source + 64 * 2**20
    if mode == 'zarr-ms':
        disk *= 2
    return {'chunk_plan_error': chunk_error, 'single_visibility_bytes': cube, 'host_plan_bytes': host,
            'legacy_host_plan_bytes': legacy, 'disk_plan_bytes': disk,
            'kernel_plan_bytes': 4 * cube * i + 4 * source,
            'planned_time_chunk': ct, 'planned_frequency_chunk': cf,
            'full_baseline_geometry_bytes': geometry, 'gain_mode_tile_bytes': gain_tile,
            'graph_allowance_bytes': graph, 'cpu_retention_allowance_bytes': retention, 'memory_model': 'staged-zarr-v2' if mode == 'zarr' and memory_model == 'staged' else 'chunked-zarr-v2' if mode == 'zarr' and memory_model == 'chunked' else 'conservative-eager-v1'}


def guard_reason(case, mode, host_budget, disk_free, gpu_budget=None, chunk_mb=16, workers=1, memory_model="conservative", device="cpu"):
    e = estimates(case, mode, chunk_mb, workers, memory_model, device)
    if e['chunk_plan_error']:
        return e['chunk_plan_error']
    if e['host_plan_bytes'] > host_budget:
        return f"host plan {e['host_plan_bytes']} exceeds budget {host_budget} bytes"
    if e['disk_plan_bytes'] > disk_free:
        return f"disk plan {e['disk_plan_bytes']} exceeds available budget {disk_free} bytes"
    if mode.endswith('kernel') and gpu_budget and e['kernel_plan_bytes'] > gpu_budget:
        return f"kernel plan {e['kernel_plan_bytes']} exceeds GPU budget {gpu_budget} bytes"
    return None
