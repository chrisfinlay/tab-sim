"""The ri_kernels route of ``rfi_vis`` against the pure-JAX route it replaces."""

import jax
import numpy as np
import pytest

from tabsim.jax import interferometry as itf


@pytest.fixture(autouse=True)
def cold_probe():
    itf._kernel_usable.cache_clear()
    yield
    itf._kernel_usable.cache_clear()


@pytest.fixture(params=[True, False], ids=["x64", "x32"])
def x64(request):
    enabled = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param)
    yield request.param
    jax.config.update("jax_enable_x64", enabled)


def rfi_inputs(x64, complex_amp=False, n_src=3, n_time=4, n_int=5, n_ant=6, n_freq=2):
    rng = np.random.default_rng(0)
    real = np.float64 if x64 else np.float32
    # A satellite's range needs float64; single precision only resolves a near field.
    offset, spread = (1e6, 1e3) if x64 else (0.0, 10.0)
    amp = rng.uniform(0.5, 2.0, (n_src, n_time, n_int, n_ant, n_freq))
    if complex_amp:
        amp = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, amp.shape))
    dist = offset + spread * rng.standard_normal((n_src, n_time, n_int, n_ant))
    freqs = np.linspace(1.2e9, 1.4e9, n_freq)
    # A baseline chunk: shuffled, incomplete, and with autocorrelations.
    a1, a2 = rng.permutation(np.argwhere(np.tri(n_ant, dtype=bool)))[:-4].T
    amp = amp.astype((np.complex128 if x64 else np.complex64) if complex_amp else real)
    return amp, dist.astype(real), freqs.astype(real), a1, a2


@pytest.mark.parametrize("complex_amp", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_kernel_matches_pure_jax(x64, complex_amp, jit):
    pytest.importorskip("ri_kernels")
    args = rfi_inputs(x64, complex_amp)
    vis = np.asarray((jax.jit(itf.rfi_vis) if jit else itf.rfi_vis)(*args))
    expected = np.asarray(itf.rfi_vis_jax(*args))

    assert itf.kernel_usable()
    assert vis.shape == expected.shape == (4, len(args[3]), 2)
    assert vis.dtype == expected.dtype == (np.complex128 if x64 else np.complex64)
    np.testing.assert_allclose(vis, expected, rtol=0, atol=1e-8 if x64 else 1e-3)


class NoLibrary:
    def __init__(self, *args):
        pass

    def eval(self, *args):
        raise RuntimeError("GPU library not found")


@pytest.mark.parametrize("op", [None, NoLibrary], ids=["not-installed", "no-library"])
def test_falls_back_to_pure_jax_from_a_cold_jit(op, monkeypatch, recwarn):
    monkeypatch.setattr(itf, "RFIVisOp", op)
    args = rfi_inputs(jax.config.jax_enable_x64)
    # jit caches by function, so a fresh one is what makes this trace cold.
    vis = jax.jit(lambda *args: itf.rfi_vis(*args))(*args)

    np.testing.assert_array_equal(vis, jax.jit(itf.rfi_vis_jax)(*args))
    assert [str(w.message) for w in recwarn if "pure JAX" in str(w.message)] == (
        [] if op is None else
        ["RFI visibilities fall back to pure JAX, which is slower: GPU library not found"]
    )
