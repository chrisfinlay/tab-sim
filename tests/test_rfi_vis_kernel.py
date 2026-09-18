"""The ri_kernels route of ``rfi_vis`` against the pure-JAX route it replaces."""

import jax
import numpy as np
import pytest

from tabsim.jax import interferometry as itf

pytest.importorskip("ri_kernels")


@pytest.fixture(params=[True, False], ids=["x64", "x32"])
def x64(request):
    enabled = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param)
    itf.kernel_usable.cache_clear()
    yield request.param
    jax.config.update("jax_enable_x64", enabled)
    itf.kernel_usable.cache_clear()


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
    # Shuffled, with autocorrelations: a baseline chunk need not be sorted or complete.
    a1, a2 = rng.permutation(np.argwhere(np.tri(n_ant, dtype=bool))).T
    amp = amp.astype((np.complex128 if x64 else np.complex64) if complex_amp else real)
    return amp, dist.astype(real), freqs.astype(real), a1, a2


@pytest.mark.parametrize("complex_amp", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_kernel_matches_pure_jax(x64, complex_amp, jit):
    args = rfi_inputs(x64, complex_amp)
    kernel = jax.jit(itf.rfi_vis_kernel) if jit else itf.rfi_vis_kernel
    vis = np.asarray(kernel(*args))
    expected = np.asarray(itf.rfi_vis_jax(*args))

    assert vis.shape == expected.shape == (4, len(args[3]), 2)
    assert vis.dtype == expected.dtype == (np.complex128 if x64 else np.complex64)
    np.testing.assert_allclose(vis, expected, rtol=0, atol=1e-8 if x64 else 1e-3)


def test_rfi_vis_uses_the_kernel_and_falls_back_without_it(x64, monkeypatch):
    args = rfi_inputs(x64)
    assert itf.kernel_usable()
    np.testing.assert_array_equal(itf.rfi_vis(*args), itf.rfi_vis_kernel(*args))

    monkeypatch.setattr(itf, "RFIVisOp", None)
    itf.kernel_usable.cache_clear()
    assert not itf.kernel_usable()
    np.testing.assert_array_equal(itf.rfi_vis(*args), itf.rfi_vis_jax(*args))


def test_a_missing_backend_library_warns_and_falls_back(x64, monkeypatch):
    class NoLibrary:
        def __init__(self, *args):
            pass

        def eval(self, *args):
            raise RuntimeError("GPU library not found")

    monkeypatch.setattr(itf, "RFIVisOp", NoLibrary)
    itf.kernel_usable.cache_clear()
    args = rfi_inputs(x64)
    with pytest.warns(UserWarning, match="GPU library not found"):
        vis = itf.rfi_vis(*args)
    np.testing.assert_array_equal(vis, itf.rfi_vis_jax(*args))
