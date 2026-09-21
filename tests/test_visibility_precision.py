"""Scientific accuracy and dtype contracts for mixed visibility precision."""

import skyfield.api  # Complete urllib probing before any offline fixture.
import dask.array as da
import jax
import numpy as np
import pytest

from tabsim.jax import interferometry as ji
from tabsim.dask import interferometry as di

C = 299792458.0


@pytest.fixture(autouse=True)
def double_geometry():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _assert_flux_error(actual, reference, flux, tolerance=3e-6):
    """Normalize by input flux, never by a potentially cancelling visibility."""
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual - reference)) / flux < tolerance


@pytest.mark.parametrize(
    "precision,dtype", [("single", np.complex64), ("double", np.complex128)]
)
def test_large_phase_uses_double_geometry_and_trigonometry(precision, dtype):
    freqs = np.array([9.123456789e8, 1.4123456789e9])
    uvw = np.array([[[12345678.123456, -2345678.987654, 876543.123456]]])
    lmn = np.array(
        [
            [
                0.37123456789,
                0.21987654321,
                np.sqrt(1 - 0.37123456789**2 - 0.21987654321**2),
            ]
        ]
    )
    intensity = np.ones((1, 1, 2))
    phase = (
        2
        * np.pi
        / C
        * np.einsum("tbc,sc->stb", uvw, lmn - [0, 0, 1])[..., None]
        * freqs
    )
    expected = np.exp(1j * phase).sum(axis=0)
    actual = np.asarray(
        ji.astro_vis(intensity, uvw, lmn, freqs, visibility_precision=precision)
    )
    assert actual.dtype == dtype
    _assert_flux_error(actual, expected, 1.0, tolerance=3e-6)
    # This fixture must discriminate early float32 geometry/phase rounding.
    rounded = np.exp(1j * phase.astype(np.float32)).sum(axis=0)
    assert np.max(np.abs(rounded - expected)) > 0.01


@pytest.mark.parametrize(
    "precision,dtype", [("single", np.complex64), ("double", np.complex128)]
)
def test_rfi_submeter_path_differences_on_large_common_offset(precision, dtype):
    freq = np.array([150e6, 1.4e9])
    a1 = np.array([0, 0, 1])
    a2 = np.array([1, 2, 2])
    distance = 1e9 + np.array([0.0, 0.03125, 0.15625])[None, None, None, :]
    amplitude = np.ones((1, 1, 1, 3, 2))
    phase = -2 * np.pi / C * (distance[..., a1] - distance[..., a2])[..., None] * freq
    expected = np.exp(1j * phase).sum(axis=0).mean(axis=1)
    actual = np.asarray(
        ji.rfi_vis(amplitude, distance, freq, a1, a2, visibility_precision=precision)
    )
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    assert np.max(np.abs(expected - 1)) > 0.1


@pytest.mark.parametrize(
    "family", ["astro_vis", "astro_vis_gauss", "astro_vis_exp", "rfi_vis"]
)
@pytest.mark.parametrize(
    "precision,dtype", [("single", np.complex64), ("double", np.complex128)]
)
def test_all_families_dask_uneven_tails(family, precision, dtype):
    rng = np.random.default_rng(2026)

    def arr(x, chunks):
        return da.from_array(np.asarray(x), chunks=chunks)

    freq = arr(np.linspace(150e6, 170e6, 5), 2)
    uvw = arr(rng.normal(size=(3, 3, 3)) * 3000, (2, 2, 3))
    source = arr(rng.uniform(0.1, 1, (3, 3, 5)), (3, 2, 2))
    lm = np.array([[0.01, 0.02], [0.03, -0.01], [-0.02, 0.01]])
    lmn = arr(np.c_[lm, np.sqrt(1 - np.sum(lm**2, axis=1))], (3, 3))
    shape = arr([0.1, 0.2, 0.3], 3)
    if family == "astro_vis":
        args = [source, uvw, lmn, freq]
    elif family == "astro_vis_gauss":
        args = [source, shape, shape, shape, uvw, lmn, freq]
    elif family == "astro_vis_exp":
        args = [source, shape, uvw, lmn, freq]
    else:
        amplitude = arr(rng.uniform(0.1, 1, (3, 3, 3, 3, 5)), (3, 2, 3, 3, 2))
        distance = arr(1e8 + rng.uniform(0, 10, (3, 3, 3, 3)), (3, 2, 3, 3))
        args = [amplitude, distance, freq, arr([0, 0, 1], 2), arr([1, 2, 2], 2)]
    eager = [x.compute(scheduler="synchronous") for x in args]
    reference = np.asarray(getattr(ji, family)(*eager, visibility_precision="double"))
    graph = getattr(di, family)(*args, visibility_precision=precision)
    assert graph.dtype == dtype
    actual = graph.compute(scheduler="synchronous")
    assert actual.dtype == dtype and actual.shape == (3, 3, 5)
    _assert_flux_error(actual, reference, 3.0, 3e-6 if precision == "single" else 1e-9)


def test_cancellation_error_is_bounded_by_total_source_flux():
    n = 256
    frequency = np.array([150e6])
    lmn = np.tile([[0.0, 0.0, 1.0], [0.1, 0.0, np.sqrt(0.99)]], (n // 2, 1))
    uvw = np.array([[[C / (0.2 * frequency[0]), 0.0, 0.0]]])
    source = np.ones((n, 1, 1))
    reference = np.asarray(
        ji.astro_vis(source, uvw, lmn, frequency, visibility_precision="double")
    )
    actual = np.asarray(
        ji.astro_vis(source, uvw, lmn, frequency, visibility_precision="single")
    )
    assert np.max(np.abs(reference)) < 1e-10 * n
    _assert_flux_error(actual, reference, float(n), 1e-6)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_gain_and_noise_paths_preserve_visibility_dtype(dtype):
    vis = da.from_array(np.full((3, 3, 5), 1 + 2j, dtype=dtype), chunks=(2, 2, 2))
    gains = da.from_array(
        np.full((3, 3, 5), 1.02 + 0.03j, dtype=np.complex128), chunks=(2, 3, 2)
    )
    a1 = da.from_array(np.array([0, 0, 1]), chunks=2)
    a2 = da.from_array(np.array([1, 2, 2]), chunks=2)
    amplified = di.apply_gains(vis, da.zeros_like(vis), gains, a1, a2)
    noisy, noise = di.add_noise(
        amplified, da.from_array(np.full(5, 0.1), chunks=2), 123
    )
    for array in [amplified, noisy, noise]:
        assert array.dtype == dtype
        actual = array.compute(scheduler="synchronous")
        assert actual.dtype == dtype and np.isfinite(actual).all()
    np.testing.assert_allclose(
        noisy.compute() - amplified.compute(), noise.compute(), rtol=2e-5, atol=5e-7
    )


def _observation(**kwargs):
    from tabsim.dask.observation import Observation

    positions = np.array(
        [
            [5109000.0, 20000.0, -3200000.0],
            [5109020.0, 20010.0, -3199990.0],
            [5109030.0, 20030.0, -3199950.0],
        ]
    )
    return Observation(
        latitude=-30.0,
        longitude=20.0,
        elevation=100.0,
        ra=30.0,
        dec=-30.0,
        ITRF_array=positions,
        times_mjd=60000.0 + np.arange(3) * 2 / 86400,
        freqs=np.array([150e6, 151e6, 152e6]),
        SEFD=np.ones(3) * 5000,
        int_time=2.0,
        n_int_samples=3,
        max_chunk_MB=0.001,
        **kwargs,
    )


def test_observation_default_single_enables_double_geometry():
    jax.config.update("jax_enable_x64", False)
    obs = _observation()
    assert jax.config.jax_enable_x64
    assert obs.visibility_precision == "single"
    assert obs.vis_ast.dtype == obs.vis_rfi.dtype == np.dtype("complex64")
    for array in (obs.ITRF, obs.ants_uvw, obs.ants_xyz, obs.times_mjd_fine, obs.freqs):
        assert array.dtype == np.dtype("float64")
    assert obs.ants_uvw.compute().dtype == np.float64


@pytest.mark.parametrize(
    "precision,dtype", [("single", np.complex64), ("double", np.complex128)]
)
def test_staged_all_arrays_preserve_precision_and_geometry(precision, dtype, tmp_path):
    import xarray as xr

    obs = _observation(visibility_precision=precision)
    obs.addAstro(np.ones(3), np.array([30.01]), np.array([-30.01]))
    obs.addStationaryRFI(
        np.full(3, 1e-14), np.array([-26.0]), np.array([116.0]), np.array([1000.0])
    )
    expected = obs.calculate_vis(random_seed=0).compute(scheduler="synchronous")
    path = tmp_path / "all.zarr"
    actual = obs.write_to_zarr(path, save_arrays=list(expected.data_vars))
    try:
        assert set(actual.data_vars) == set(expected.data_vars)
        for name in ("vis_ast", "vis_rfi", "vis_obs", "vis_calibrated", "noise_data"):
            assert actual[name].dtype == expected[name].dtype == dtype
            np.testing.assert_allclose(
                actual[name].compute(),
                expected[name],
                rtol=3e-6 if precision == "single" else 1e-11,
                atol=3e-6 if precision == "single" else 1e-11,
            )
        for name in ("ants_itrf", "ants_uvw", "ants_xyz", "bl_uvw"):
            assert actual[name].dtype == np.float64
        xr.testing.assert_equal(actual.flags.compute(), expected.flags)
    finally:
        actual.close()


@pytest.mark.parametrize("invalid", ["half", "float32", "", None])
def test_invalid_precision_rejected(invalid):
    from tabsim.precision import visibility_dtype

    with pytest.raises(ValueError, match="visibility_precision"):
        visibility_dtype(invalid)
    with pytest.raises(ValueError, match="visibility_precision"):
        _observation(visibility_precision=invalid)


def test_standalone_kernel_refuses_disabled_double_geometry():
    jax.config.update("jax_enable_x64", False)
    with pytest.raises(ValueError, match="x64"):
        ji.astro_vis(
            np.ones((1, 1, 1)),
            np.ones((1, 1, 3)),
            np.array([[0.01, 0.02, np.sqrt(1 - 0.0005)]]),
            np.array([150e6]),
            visibility_precision="single",
        )


def test_precision_modes_use_same_seeded_noise_draws():
    zero = da.zeros((3, 3, 5), chunks=(2, 2, 2), dtype=np.complex128)
    scale = da.from_array(np.linspace(0.1, 0.5, 5), chunks=2)
    _, double = di.add_noise(zero, scale, 0)
    _, single = di.add_noise(zero.astype(np.complex64), scale, 0)
    np.testing.assert_array_equal(
        single.compute(), double.compute().astype(np.complex64)
    )
    other = di.add_noise(zero, scale, 1)[1].compute()
    assert not np.array_equal(other, double.compute())


@pytest.mark.parametrize(
    "yaml_precision,override,expected",
    [
        (None, None, "single"),
        ("double", None, "double"),
        ("double", "single", "single"),
        ("single", "double", "double"),
    ],
)
def test_config_and_cli_precision_precedence(
    yaml_precision, override, expected, tmp_path, monkeypatch
):
    import sys
    from tabsim.scripts import sim_vis

    path = tmp_path / "config.yaml"
    path.write_text(
        "{}"
        if yaml_precision is None
        else "observation:\n  visibility_precision: " + yaml_precision + "\n"
    )
    captured = {}

    def run(sim_config, **kwargs):
        captured.update(sim_config)
        return None, "out"

    monkeypatch.setattr(sim_vis, "run_sim_config", run)
    argv = ["sim-vis", "-c", str(path)]
    if override:
        argv += ["--visibility-precision", override]
    monkeypatch.setattr(sys, "argv", argv)
    sim_vis.main()
    assert captured["observation"]["visibility_precision"] == expected


def test_mixed_gain_inputs_have_honest_dask_metadata():
    ast = da.ones((2, 1, 2), chunks=(1, 1, 2), dtype=np.complex64)
    rfi = da.ones_like(ast, dtype=np.complex128)
    gains = da.ones((2, 2, 2), chunks=(1, 2, 2), dtype=np.complex64)
    a1 = da.from_array([0])
    a2 = da.from_array([1])
    vis = di.apply_gains(ast, rfi, gains, a1, a2)
    baseline = di.ants_to_bl(gains, a1, a2)
    assert vis.dtype == vis.compute().dtype == np.complex128
    assert baseline.dtype == baseline.compute().dtype == np.complex64


def test_many_sources_with_large_dynamic_range():
    rng = np.random.default_rng(51)
    n = 1024
    source = np.geomspace(1e-6, 1e3, n)[rng.permutation(n), None, None]
    lm = rng.uniform(-0.3, 0.3, (n, 2))
    lmn = np.c_[lm, np.sqrt(1 - np.sum(lm**2, axis=1))]
    uvw = np.array([[[12_345.67, -7654.321, 3456.789]]])
    frequency = np.array([150e6])
    double = np.asarray(
        ji.astro_vis(source, uvw, lmn, frequency, visibility_precision="double")
    )
    single = np.asarray(ji.astro_vis(source, uvw, lmn, frequency))
    _assert_flux_error(single, double, source.sum(), 3e-6)
