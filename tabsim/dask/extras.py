"""Host-only chunk planning. Estimates are not allocator or process limits."""
import math
import numbers


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def get_factors(n: int):
    """Return sorted divisors without initializing JAX or transferring scalars."""
    n = _integer('dimension', n)
    return sorted({v for k in range(1, math.isqrt(n) + 1) if n % k == 0 for v in (k, n // k)})


def estimate_working_set(time, freq, n_int, n_bl, *, n_ant, n_rfi=0, n_ast=0,
                         workers=2, backend='cpu', scratch_factor=None, full_n_freq=None):
    """Estimate x64 task buffers for the scan kernels and staged composition.

    Source counts are conservative totals, not a claim of source/baseline
    vectorization. Scratch is a configurable allowance, not an XLA allocation
    measurement. Excludes graph/allocator caches, full-history eager geometry,
    Python/runtime baseline and optional ancillary rechunking.
    """
    for name, value in locals().copy().items():
        if name in ('time', 'freq', 'n_int', 'n_bl', 'n_ant', 'workers'):
            _integer(name, value)
        elif name in ('n_rfi', 'n_ast'):
            _integer(name, value, 0)
    if backend not in ('cpu', 'gpu'):
        raise ValueError('working-set backend must be cpu or gpu')
    factor = (4.0 if backend == 'cpu' else 6.0) if scratch_factor is None else _positive('scratch_factor', scratch_factor)
    fine = time * n_int
    output = 16 * time * n_bl * freq
    nominal = output * n_int
    amplitude = 8 * n_rfi * fine * n_ant * freq
    distances = 8 * n_rfi * fine * n_ant
    geometry = 24 * fine * n_bl
    astro = 8 * n_ast * time * freq + 24 * n_ast
    gains = 16 * time * n_ant * freq
    full_n_freq = freq if full_n_freq is None else _integer('full_n_freq', full_n_freq)
    if full_n_freq < freq:
        raise ValueError('full_n_freq must cover the selected frequency chunk')
    full_band_gains = 16 * (time + 1) * n_ant * full_n_freq
    # The gain generator uses 1000 Fourier modes; allow real/complex temporaries.
    gain_modes = 6 * 8 * 1000 * time * n_ant
    inputs = amplitude + 4 * distances + geometry + astro + gains
    scratch = math.ceil(factor * max(nominal, amplitude, distances))
    kernel = inputs + 2 * output + scratch
    composition = 6 * output + 2 * gains
    host = max(workers * max(kernel if backend == 'cpu' else inputs + 2 * output,
                             gain_modes + 3 * full_band_gains), composition)
    device = max(workers * max(kernel, gain_modes + 3 * full_band_gains), composition) if backend == 'gpu' else 0
    return dict(host_bytes=host, device_bytes=device, estimated_bytes=max(host, device),
                amplitude_bytes=amplitude, distance_bytes=distances,
                geometry_tile_bytes=geometry, astronomical_input_bytes=astro,
                output_bytes=output, gain_bytes=gains, full_band_gain_bytes=full_band_gains, gain_mode_allowance_bytes=gain_modes,
                scratch_bytes=scratch, scratch_factor=factor, workers=workers,
                backend=backend, rfi_sources=n_rfi, ast_sources=n_ast,
                composition_bytes=composition, model='scan-staged-x64-v1')


def get_chunksizes(n_t, n_f, n_int, n_bl, MB_max, *, n_ant=None, n_rfi=0, n_ast=0,
                   workers=2, backend='cpu', working_set_MB=None, scratch_factor=None, full_n_freq=None):
    """Largest uniform divisor tile satisfying strict nominal and optional estimated limits.

    MB_max is decimal MB for the fine-time complex128 visibility tile. The
    separate working_set_MB budget applies to each estimated memory space,
    including workers. Neither is a bound on total RSS/VRAM. Ties prefer smaller
    time chunks (less geometry/gain scratch); all selected shapes divide inputs.
    """
    n_t, n_f, n_int, n_bl = (_integer(name, value) for name, value in
                             zip(('n_t', 'n_f', 'n_int', 'n_bl'), (n_t, n_f, n_int, n_bl)))
    limit = math.floor(_positive('max_chunk_MB', MB_max) * 1e6)
    budget = None if working_set_MB is None else _positive('working_set_MB', working_set_MB) * 1e6
    if n_ant is None:
        # Conservative for cross correlations when only a baseline count is known.
        n_ant = math.ceil((1 + math.sqrt(1 + 8 * n_bl)) / 2)
    def memory(t, f):
        return estimate_working_set(t, f, n_int, n_bl, n_ant=n_ant, n_rfi=n_rfi,
            n_ast=n_ast, workers=workers, backend=backend, scratch_factor=scratch_factor,
            full_n_freq=n_f if full_n_freq is None else full_n_freq)
    minimum = memory(1, 1)
    tile_unit = 16 * n_int * n_bl
    if tile_unit > limit or (budget is not None and minimum['estimated_bytes'] > budget):
        raise ValueError(f'No feasible (time=1, freq=1) tile: minimum nominal {tile_unit} bytes, '
                         f'estimated working set {minimum["estimated_bytes"]} bytes. '
                         'Increase the corresponding budget, reduce workers/sources/antennas, '
                         'or use source/baseline batching (not implemented here).')
    candidates = []
    for t in get_factors(n_t):
        for f in get_factors(n_f):
            if tile_unit * t * f <= limit:
                model = memory(t, f)
                if budget is None or model['estimated_bytes'] <= budget:
                    candidates.append((t * f, -t, t, f, model))
    _, _, t, f, model = max(candidates, key=lambda item: item[:2])
    size = tile_unit * t * f
    return dict(time=t, freq=f, chunk_bytes=f'{size / 1e6:.0f} MB',
                nominal_bytes=size, max_chunk_bytes=limit, working_set=model,
                working_set_budget_bytes=budget)
