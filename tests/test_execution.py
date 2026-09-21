"""GPU slots include completion/readback; mapped task caches contain host arrays."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
import subprocess
import sys
import time

import jax
import numpy as np
import pytest

from tabsim import execution as ex


@pytest.fixture(autouse=True)
def policy():
    ex.configure_execution(gpu_concurrency=1)
    ex.execution_stats(reset=True)
    yield
    ex.configure_execution(gpu_concurrency=1)
    ex.execution_stats(reset=True)


@pytest.mark.parametrize('value', [0, -1, True, 1.5, None])
def test_invalid_limit(value):
    with pytest.raises(ValueError):
        ex.configure_execution(gpu_concurrency=value)


def test_readback_holds_gpu_slot_and_exception_releases_it(monkeypatch):
    device = SimpleNamespace(platform='gpu', id=0)
    monkeypatch.setattr(jax, 'local_devices', lambda: [device])
    monkeypatch.setattr(jax, 'default_device', lambda d: nullcontext())
    monkeypatch.setattr(jax, 'device_put', lambda values, d: values)
    readback, release, second = Event(), Event(), Event()
    def get(value):
        if value[0] == 1:
            readback.set()
            assert release.wait(5)
        return value
    monkeypatch.setattr(jax, 'device_get', get)
    def kernel(value):
        if value[0] == 2:
            second.set()
        return value
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(ex.execute_kernel, kernel, np.array([1]))
        assert readback.wait(5)
        following = pool.submit(ex.execute_kernel, kernel, np.array([2]))
        try:
            assert not second.wait(.1)
            with pytest.raises(RuntimeError):
                ex.configure_execution(gpu_concurrency=2)
        finally:
            release.set()
        np.testing.assert_array_equal(first.result(), [1])
        np.testing.assert_array_equal(following.result(), [2])
    assert ex.execution_stats()['peak_active'] == 1
    def fail(x):
        raise ValueError('kernel failed')
    with pytest.raises(ValueError, match='kernel failed'):
        ex.execute_kernel(fail, np.array([1]))
    assert ex.execution_stats()['active'] == 0
    ex.execute_kernel(kernel, np.array([2]))


def test_two_slots_allow_overlap():
    ex.configure_execution(gpu_concurrency=2)
    entered = Event()
    release = Event()
    def task():
        with ex._gpu_slot():
            entered.set()
            assert release.wait(5)
    with ThreadPoolExecutor(2) as pool:
        future = pool.submit(task)
        assert entered.wait(5)
        try:
            with ex._gpu_slot():
                assert ex.execution_stats()['active'] == 2
        finally:
            release.set()
        future.result()
    assert ex.execution_stats()['peak_active'] == 2


def test_mapped_result_cache_is_host_backed():
    import dask.array as da
    from dask.callbacks import Callback
    from tabsim.dask.coordinates import radec_to_XYZ
    cached = []
    def inspect(key, result, dsk, state, worker):
        cached.extend(jax.tree_util.tree_leaves(result))
    graph = radec_to_XYZ(da.from_array(np.arange(6.), chunks=2),
                        da.from_array(np.full(6, -30.), chunks=2))
    with Callback(posttask=inspect):
        result = graph.compute(scheduler='threads', num_workers=2)
    assert isinstance(result, np.ndarray)
    assert not any(isinstance(value, jax.Array) for value in cached)


def test_explicit_device_and_host_result():
    seen = []
    def kernel(value):
        seen.extend(value.devices())
        return value + 1
    result = ex.execute_kernel(kernel, np.array([1., 2.]))
    assert seen == [jax.local_devices()[0]]
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [2., 3.])


def test_cli_import_does_not_initialize_jax():
    subprocess.run([sys.executable, '-c',
        'import sys; import tabsim.scripts.sim_vis; assert "jax" not in sys.modules'], check=True)
