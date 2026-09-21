"""Process-local GPU admission and completed host results for Dask callbacks.

This module intentionally does not import JAX until execution. Select visible
GPUs before importing simulation modules; run one process per visible GPU.
"""
from contextlib import contextmanager
from threading import Condition
from time import perf_counter
import numbers

_condition = Condition()
_limit = 1
_active = 0
_waiting = 0
_stats = dict(calls=0, peak_active=0, queue_s=0.0, execution_s=0.0,
              host_result_bytes=0, devices=[])


def configure_execution(*, gpu_concurrency=1):
    """Set this worker's GPU block limit; call while no blocks are executing.

    Host results are mandatory. This bounds active kernels through readback, not
    the whole Dask graph: use the default staged writer to bound result lifetime.
    CPU work is not serialized by this limit. Configure each distributed worker
    separately; this process-local setting is not embedded in a serialized graph.
    """
    if (isinstance(gpu_concurrency, bool) or not isinstance(gpu_concurrency, numbers.Integral)
            or gpu_concurrency < 1):
        raise ValueError('gpu_concurrency must be a positive integer')
    global _limit
    with _condition:
        if _active or _waiting:
            raise RuntimeError('Cannot change execution policy while GPU blocks are active or waiting')
        _limit = int(gpu_concurrency)


def execution_stats(*, reset=False):
    """Admission diagnostics; byte counts describe host results, not physical I/O."""
    with _condition:
        result = dict(_stats, devices=list(_stats['devices']), gpu_concurrency=_limit,
                      active=_active, waiting=_waiting)
        if reset:
            if _active or _waiting:
                raise RuntimeError('Cannot reset statistics while GPU blocks are active or waiting')
            _stats.update(calls=0, peak_active=0, queue_s=0.0, execution_s=0.0,
                          host_result_bytes=0, devices=[])
        return result


@contextmanager
def _gpu_slot():
    global _active, _waiting
    start = perf_counter()
    with _condition:
        _waiting += 1
        try:
            _condition.wait_for(lambda: _active < _limit)
            _active += 1
            _stats['peak_active'] = max(_stats['peak_active'], _active)
            _stats['queue_s'] += perf_counter() - start
        finally:
            _waiting -= 1
    try:
        yield
    finally:
        with _condition:
            _active -= 1
            _condition.notify_all()


def execute_kernel(function, *args, **kwargs):
    """Explicit placement, kernel completion and host readback inside one slot."""
    import jax
    import numpy as np
    from contextlib import nullcontext

    devices = jax.local_devices()
    device = jax.config.jax_default_device or devices[0]
    gpu = device.platform == 'gpu'
    if gpu and len(devices) != 1:
        raise RuntimeError('Expose exactly one GPU per worker with CUDA_VISIBLE_DEVICES '
                           'before JAX initialization; launch one worker process per GPU')
    with _gpu_slot() if gpu else nullcontext():
        start = perf_counter()
        with jax.default_device(device):
            placed_args, placed_kwargs = jax.device_put((args, kwargs), device)
            result = function(*placed_args, **placed_kwargs)
            # device_get completes the transfer before the admission slot is
            # released. No JAX object enters the Dask scheduler's result cache.
            host = jax.tree_util.tree_map(np.asarray, jax.device_get(result))
            del result, placed_args, placed_kwargs
        if gpu:
            with _condition:
                _stats['calls'] += 1
                _stats['execution_s'] += perf_counter() - start
                _stats['host_result_bytes'] += sum(x.nbytes for x in jax.tree_util.tree_leaves(host))
                label = f'{device.platform}:{device.id}'
                if label not in _stats['devices']:
                    _stats['devices'].append(label)
        return host
