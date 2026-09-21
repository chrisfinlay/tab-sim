"""Native RFI routing preserves double geometry and chosen visibility precision."""

from functools import partial

import jax
import numpy as np
import pytest

from tabsim.jax import interferometry as itf


@pytest.fixture(autouse=True)
def double_geometry_and_cold_probe():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    itf._kernel_usable.cache_clear()
    try:
        yield
    finally:
        itf._kernel_usable.cache_clear()
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture(params=["single", "double"])
def precision(request):
    return request.param


def rfi_inputs(
    complex_amp=False,
    n_src=3,
    n_time=4,
    n_int=5,
    n_ant=6,
    n_freq=2,
    offset=1e6,
    spread=1e3,
):
    rng = np.random.default_rng(0)
    amp = rng.uniform(0.5, 2.0, (n_src, n_time, n_int, n_ant, n_freq))
    if complex_amp:
        amp = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, amp.shape))
    dist = offset + spread * rng.standard_normal((n_src, n_time, n_int, n_ant))
    freq = np.linspace(1.2e9, 1.4e9, n_freq)
    # Include every autocorrelation plus noncanonical baseline order/direction.
    pairs = np.array(
        [(0, 0), (3, 1), (5, 2), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 4), (4, 0)]
    )
    pairs = pairs[rng.permutation(len(pairs))]
    return amp, dist, freq, pairs[:, 0], pairs[:, 1]


def require_native():
    pytest.importorskip("ri_kernels")
    if not itf.kernel_usable():
        pytest.skip("Native library unavailable for the selected backend")


def compare(actual, expected, args, precision):
    assert (
        actual.dtype
        == expected.dtype
        == np.dtype("complex64" if precision == "single" else "complex128")
    )
    assert actual.shape == (args[0].shape[1], len(args[3]), args[0].shape[-1])
    # Normalize against incoherent input-flux bound; cancellations may have zero vis.
    flux = np.sum(np.max(np.abs(args[0]), axis=(1, 2, 3, 4)) ** 2)
    tolerance = 5e-6 if precision == "single" else 2e-9
    assert np.max(np.abs(actual - expected)) / flux < tolerance
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("complex_amp", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_kernel_matches_pure_jax(precision, complex_amp, compiled):
    require_native()
    args = rfi_inputs(complex_amp)
    fn = partial(itf.rfi_vis, visibility_precision=precision)
    if compiled:
        fn = jax.jit(fn)
    actual = np.asarray(fn(*args))
    expected = np.asarray(itf.rfi_vis_jax(*args, visibility_precision=precision))
    compare(actual, expected, args, precision)
    # Autocorrelations are real sums of source powers, averaged over samples.
    for index, (a, b) in enumerate(zip(args[3], args[4])):
        if a == b:
            power = np.sum(np.mean(np.abs(args[0][:, :, :, a, :]) ** 2, axis=2), axis=0)
            np.testing.assert_allclose(
                actual[:, index, :],
                power,
                rtol=3e-6 if precision == "single" else 1e-12,
                atol=2e-6 if precision == "single" else 1e-12,
            )


@pytest.mark.parametrize("compiled", [False, True])
def test_large_common_range_preserves_submeter_differences(precision, compiled):
    require_native()
    args = rfi_inputs(True, offset=1e9, spread=0.0625)
    fn = partial(itf.rfi_vis, visibility_precision=precision)
    actual = np.asarray((jax.jit(fn) if compiled else fn)(*args))
    expected = np.asarray(itf.rfi_vis_jax(*args, visibility_precision=precision))
    compare(actual, expected, args, precision)
    # Ensure the case would catch subtracting paths after narrowing to float32.
    broken = list(args)
    broken[1] = args[1].astype(np.float32).astype(np.float64)
    rounded = np.asarray(itf.rfi_vis_jax(*broken, visibility_precision="double"))
    assert np.max(np.abs(rounded - expected)) > 0.01


class NoLibrary:
    def __init__(self, *args):
        pass

    def eval(self, *args):
        raise RuntimeError("GPU library not found")


@pytest.mark.parametrize("op", [None, NoLibrary], ids=["not-installed", "no-library"])
def test_falls_back_to_pure_jax_from_a_cold_jit(precision, op, monkeypatch, recwarn):
    monkeypatch.setattr(itf, "RFIVisOp", op)
    args = rfi_inputs(True)
    actual = jax.jit(lambda *xs: itf.rfi_vis(*xs, visibility_precision=precision))(
        *args
    )
    expected = jax.jit(partial(itf.rfi_vis_jax, visibility_precision=precision))(*args)
    np.testing.assert_array_equal(actual, expected)
    assert [str(w.message) for w in recwarn if "pure JAX" in str(w.message)] == (
        []
        if op is None
        else [
            "RFI visibilities fall back to pure JAX, which is slower: GPU library not found"
        ]
    )


def test_disabled_x64_rejected_before_native_probe(monkeypatch):
    jax.config.update("jax_enable_x64", False)

    def forbidden():
        raise AssertionError("Must reject geometry precision before probing")

    monkeypatch.setattr(itf, "kernel_usable", forbidden)
    with pytest.raises(ValueError, match="x64"):
        itf.rfi_vis(*rfi_inputs(), visibility_precision="single")


def test_single_native_boundary_receives_prephased_amplitudes(monkeypatch):
    require_native()
    original = itf.RFIVisOp
    seen = []

    class Inspect:
        def __init__(self, *args):
            self.operation = original(*args)

        def eval(self, amplitude, phase):
            assert amplitude.dtype == np.complex64
            assert phase.dtype == np.float32
            np.testing.assert_array_equal(np.asarray(phase), 0.0)
            assert np.max(np.abs(np.asarray(amplitude).imag)) > 0.01
            seen.append(amplitude.shape)
            return self.operation.eval(amplitude, phase)

    monkeypatch.setattr(itf, "RFIVisOp", Inspect)
    args = rfi_inputs(False, offset=1e9, spread=0.0625)
    actual = np.asarray(itf.rfi_vis_kernel(*args, visibility_precision="single"))
    compare(
        actual,
        np.asarray(itf.rfi_vis_jax(*args, visibility_precision="single")),
        args,
        "single",
    )
    assert seen == [(6, 2, 4, 3, 1, 5)]


def test_asynchronous_probe_failure_is_cached_and_falls_back(
    precision, monkeypatch, recwarn
):
    calls = []

    class PendingFailure:
        def block_until_ready(self):
            calls.append("wait")
            raise RuntimeError("asynchronous native probe failure")

    class AsyncFailure:
        def __init__(self, *args):
            pass

        def eval(self, *args):
            calls.append("eval")
            return PendingFailure()

    monkeypatch.setattr(itf, "RFIVisOp", AsyncFailure)
    args = rfi_inputs(True)
    # Separate cold traces exercise the cached unavailable decision, not JIT reuse.
    for _ in range(2):
        actual = jax.jit(lambda *xs: itf.rfi_vis(*xs, visibility_precision=precision))(
            *args
        )
        expected = jax.jit(partial(itf.rfi_vis_jax, visibility_precision=precision))(
            *args
        )
        np.testing.assert_array_equal(actual, expected)
    assert itf.kernel_usable() is False
    assert calls == ["eval", "wait"]
    assert [str(w.message) for w in recwarn if "pure JAX" in str(w.message)] == [
        "RFI visibilities fall back to pure JAX, which is slower: asynchronous native probe failure"
    ]
