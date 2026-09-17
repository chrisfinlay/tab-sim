from contextlib import contextmanager
from decimal import Decimal, localcontext

import dask.array as da
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabsim import sky
from tabsim.sky import (
    SourcePlacementError,
    generate_random_sky,
    random_power_law,
    truncated_power_law_inv_cdf,
    uniform_points_disk,
)
from timeouts import fail_after


@contextmanager
def single_precision():
    enabled = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", enabled)


def reference_power_law(n_src, I_min, I_max, alpha, rng):
    """The rejection sampler as it was before its loop was bounded."""

    def inv_cdf(x):
        return I_min * (1.0 - x) ** (1.0 / (1.0 - alpha))

    I = np.array(inv_cdf(rng.uniform(size=(n_src,))))
    while np.any(I > I_max):
        idx = np.where(I > I_max)[0]
        I[idx] = inv_cdf(rng.uniform(size=(idx.shape[0],)))

    return I


def reference_random_sky(
    n_src, freqs, min_I, max_I, fov, beam_width, rng, n_beam, alpha=1.6
):
    """`generate_random_sky` as it was before its loops were bounded."""
    I = da.atleast_1d(reference_power_law(n_src, min_I, max_I, alpha, rng))
    positions = uniform_points_disk(fov / 2.0, 1, rng)
    while positions.shape[1] < n_src:
        n_sample = 2 * (n_src - positions.shape[1])
        new_positions = uniform_points_disk(fov / 2.0, n_sample, rng)
        positions = np.concatenate([positions, new_positions], axis=1)
        s1, s2 = np.triu_indices(positions.shape[1], 1)
        d = np.linalg.norm(positions[:, s1] - positions[:, s2], axis=0)
        idx = np.where(d < n_beam * beam_width)[0]
        remove_source_idx = np.unique(jnp.concatenate([s1[idx], s2[idx]]))
        positions = np.delete(positions, remove_source_idx, axis=1)

    d_ra, d_dec = positions[:, :n_src]

    spectral_indices = rng.normal(loc=0.7, scale=0.2, size=(n_src,))
    I = I[:, None] * ((freqs[None, :] / freqs[0]) ** -spectral_indices[:, None])

    return I, d_ra, d_dec


def pairwise_separations(x, y):
    s1, s2 = np.triu_indices(len(x), 1)
    return np.hypot(x[s1] - x[s2], y[s1] - y[s2])


#####################
# random_power_law
#####################


def test_power_law_min_above_max_raises():
    with fail_after(10):
        with pytest.raises(ValueError) as err:
            random_power_law(2, I_min=1.5, I_max=1.0, random_seed=0)

    assert "I_min" in str(err.value) and "1.5" in str(err.value)
    assert "I_max" in str(err.value) and "1.0" in str(err.value)


def test_power_law_without_a_maximum_is_not_truncated():
    rng, rng_ref = np.random.default_rng(5), np.random.default_rng(5)

    I = random_power_law(100, I_min=0.5, I_max=np.inf, random_seed=rng)

    I_ref = reference_power_law(100, 0.5, np.inf, 1.6, rng_ref)
    np.testing.assert_array_equal(I, I_ref)
    assert I.max() > 1.0


@pytest.mark.parametrize(
    "n_src,I_min,I_max,alpha",
    [(100, 1e-4, 1.0, 1.6), (50, 0.5, 1.0, 1.6), (200, 0.9, 1.0, 2.5)],
)
def test_power_law_draws_are_unchanged(n_src, I_min, I_max, alpha):
    """Seeded skies must not change: same fluxes and same generator state after."""
    rng, rng_ref = np.random.default_rng(123456), np.random.default_rng(123456)

    I = random_power_law(n_src, I_min, I_max, alpha, rng)
    I_ref = reference_power_law(n_src, I_min, I_max, alpha, rng_ref)

    np.testing.assert_array_equal(I, I_ref)
    assert rng.uniform() == rng_ref.uniform()


def test_power_law_equal_limits_gives_that_flux():
    with fail_after(30):
        I = random_power_law(10, I_min=0.3, I_max=0.3, random_seed=1)

    np.testing.assert_allclose(I, 0.3, rtol=1e-12)


def test_power_law_narrow_range_terminates_inside_the_range():
    I_min, I_max = 1.0, 1.0 + 1e-9
    with fail_after(30):
        I = random_power_law(1000, I_min=I_min, I_max=I_max, random_seed=2)

    assert I.shape == (1000,)
    assert np.all((I >= I_min) & (I <= I_max))


def test_power_law_narrow_range_follows_the_truncated_distribution():
    """A range this narrow is drawn directly, not by rejection."""
    from scipy.stats import kstest

    I_min, I_max, alpha = 1.0, 1.0002, 1.6

    def cdf(I):
        a = 1.0 - alpha
        return (I_min**a - I**a) / (I_min**a - I_max**a)

    with fail_after(60):
        I = random_power_law(5000, I_min, I_max, alpha, random_seed=3)

    assert np.all((I >= I_min) & (I <= I_max))
    assert kstest(I, cdf).pvalue > 1e-3


def exact_truncated_inv_cdf(x, I_min, I_max, alpha):
    """The inverse CDF as it is written on paper, limits raised to the power
    1 - alpha, in 80 digit decimal arithmetic that neither overflows nor rounds."""
    with localcontext() as ctx:
        ctx.prec = 80
        a = Decimal(1) - Decimal.from_float(float(alpha))
        lower = Decimal.from_float(float(I_min)) ** a
        upper = Decimal(0) if np.isinf(I_max) else Decimal.from_float(float(I_max)) ** a
        I = [
            ((Decimal(1) - u) * lower + u * upper) ** (Decimal(1) / a)
            for u in map(Decimal.from_float, x.tolist())
        ]
        return np.array([float(i) for i in I])


@pytest.mark.parametrize(
    "I_min,I_max,alpha",
    [
        (1e-4, 1.0, 1.6),
        (1e-4, 1.000001e-4, 100.0),  # I_min ** (1 - alpha) overflows
        (np.float64(1e-4), np.float64(1.000001e-4), 100.0),
        (1e200, 1.000001e200, 3.0),  # I_min ** (1 - alpha) underflows to zero
        (1.0, 1.0 + 1e-8, 1.6),  # limits nearly equal
        (1e-4, 1.0, 1.0 + 1e-9),  # index nearly one
        (1e-200, 1e200, 1.0 + 1e-9),  # I_max / I_min overflows
        (1e-300, np.inf, 1.001),  # I / I_min overflows where I does not
    ],
)
def test_truncated_inv_cdf_matches_exact_arithmetic(I_min, I_max, alpha):
    u = np.linspace(0.0, 1.0, 101)[: None if np.isfinite(I_max) else 56]

    I = truncated_power_law_inv_cdf(u, I_min, I_max, alpha)
    I_exact = exact_truncated_inv_cdf(u, I_min, I_max, alpha)

    assert np.all(np.isfinite(I))
    assert np.all(np.diff(I) >= 0) and I[-1] > I[0]
    np.testing.assert_allclose(I, I_exact, rtol=1e-10)
    if np.isfinite(I_max):
        # Where in the range, which the flux alone says little of in a narrow one
        np.testing.assert_allclose(I[[0, -1]], [I_min, I_max], rtol=1e-12)
        where = (I - I_min) / (float(I_max) - float(I_min))
        where_exact = (I_exact - I_min) / (float(I_max) - float(I_min))
        np.testing.assert_allclose(where, where_exact, atol=1e-7)


def test_truncated_inv_cdf_of_nearly_equal_limits_is_nearly_uniform():
    u = np.array([0.25, 0.5, 0.75])

    I = truncated_power_law_inv_cdf(u, 1e-4, 1.000001e-4, 100.0)

    np.testing.assert_allclose((I / 1e-4 - 1.0) / 1e-6, u, atol=1e-4)


def test_truncated_inv_cdf_of_index_nearly_one_is_log_uniform():
    u = np.array([0.25, 0.5, 0.75])

    I = truncated_power_law_inv_cdf(u, 1e-4, 1.0, 1.0 + 1e-9)

    np.testing.assert_allclose(I, 1e-4 * 1e4**u, rtol=1e-6)


def test_truncated_inv_cdf_of_equal_limits_is_that_flux():
    I = truncated_power_law_inv_cdf(np.array([0.0, 0.5, 1.0]), 0.3, 0.3, 1.6)

    np.testing.assert_array_equal(I, 0.3)


def test_truncated_inv_cdf_without_a_maximum_is_the_power_law():
    u = np.array([0.0, 0.25, 0.5, 0.75])

    I = truncated_power_law_inv_cdf(u, 1e-4, np.inf, 1.6)

    np.testing.assert_allclose(I, 1e-4 * (1.0 - u) ** (1.0 / (1.0 - 1.6)), rtol=1e-12)


def test_power_law_returns_whatever_drawing_directly_gives(monkeypatch):
    """Fluxes are drawn directly once. Looking at them again could loop forever
    on any that are still above the maximum, as rounding can leave them."""
    direct_draws = []

    def above_the_maximum(x, *limits):
        direct_draws.append(x.shape)
        return np.full(x.shape, 2.0)

    monkeypatch.setattr(sky, "MAX_FLUX_ROUNDS", 3)
    monkeypatch.setattr(sky, "truncated_power_law_inv_cdf", above_the_maximum)

    with fail_after(30):
        I = random_power_law(10, I_min=1.0, I_max=1.0, random_seed=0)

    assert direct_draws == [(10,)]
    np.testing.assert_array_equal(I, 2.0)


def test_power_law_in_single_precision_terminates(monkeypatch):
    """0.1 in single precision is above 0.1 in double precision, so single precision
    fluxes of exactly `I_max` = 0.1 never pass a check against it."""
    monkeypatch.setattr(sky, "MAX_FLUX_ROUNDS", 3)

    with single_precision(), fail_after(30):
        alpha = jnp.array(1.6, dtype=jnp.float32)
        I = random_power_law(4, np.float64(0.1), np.float64(0.1), alpha, 0)

    assert I.dtype == np.float32
    np.testing.assert_array_equal(I, np.float32(0.1))


def test_power_law_zero_sources():
    assert random_power_law(0, random_seed=0).shape == (0,)


#######################
# generate_random_sky
#######################

FREQS = np.array([1.227e9, 1.228e9])


@pytest.mark.parametrize(
    "n_src,min_I,max_I,beam_width",
    [
        (50, 1e-3, 1.0, 200.0 / 3600 / 5),  # as the example configs, placed at once
        (50, 0.5, 1.0, 0.0139),  # many flux redraws and several placement rounds
        (10, 0.9, 1.0, 0.04),  # few crowded sources
        (1, 1e-3, 1.0, 10.0),  # a single source is never too close to another
        (3, 1e-3, 1.0, 0.0),  # no minimum separation
    ],
)
def test_random_sky_is_unchanged(n_src, min_I, max_I, beam_width):
    """Same seed, same sky as before the loops were bounded: every flux at every
    frequency, every position, and the generator left in the same state."""
    fov, n_beam = 1.27, 5
    rng, rng_ref = np.random.default_rng(123456), np.random.default_rng(123456)

    with fail_after(120):
        I_ref, ra_ref, dec_ref = reference_random_sky(
            n_src, FREQS, min_I, max_I, fov, beam_width, rng_ref, n_beam
        )
        I, d_ra, d_dec = generate_random_sky(
            n_src=n_src,
            freqs=FREQS,
            min_I=min_I,
            max_I=max_I,
            fov=fov,
            beam_width=beam_width,
            random_seed=rng,
            n_beam=n_beam,
        )

    np.testing.assert_array_equal(np.asarray(I), np.asarray(I_ref))
    np.testing.assert_array_equal(np.asarray(d_ra), np.asarray(ra_ref))
    np.testing.assert_array_equal(np.asarray(d_dec), np.asarray(dec_ref))
    assert rng.uniform() == rng_ref.uniform()


def test_random_sky_respects_separation_and_fov():
    n_src, fov, beam_width, n_beam = 30, 1.0, 0.01, 5

    I, d_ra, d_dec = generate_random_sky(
        n_src=n_src,
        freqs=FREQS,
        fov=fov,
        beam_width=beam_width,
        random_seed=7,
        n_beam=n_beam,
    )
    d_ra, d_dec = np.asarray(d_ra), np.asarray(d_dec)

    assert np.asarray(I).shape == (n_src, len(FREQS))
    assert d_ra.shape == d_dec.shape == (n_src,)
    assert np.all(np.hypot(d_ra, d_dec) <= fov / 2)
    assert np.all(pairwise_separations(d_ra, d_dec) >= n_beam * beam_width)


def test_random_sky_separation_wider_than_fov_raises():
    """No two points in a disk are further apart than its diameter."""
    with fail_after(30):
        with pytest.raises(SourcePlacementError, match="no such arrangement") as err:
            generate_random_sky(
                n_src=2, freqs=FREQS, fov=1.0, beam_width=0.3, n_beam=5, random_seed=0
            )

    msg = str(err.value)
    for name in ["n_src", "n_beam", "beam_width", "fov"]:
        assert name in msg


def test_random_sky_more_sources_than_there_is_area_for_raises():
    """Separation below the FoV, but the sources need more area than there is."""
    with fail_after(30):
        with pytest.raises(SourcePlacementError, match="no such arrangement"):
            generate_random_sky(
                n_src=20, freqs=FREQS, fov=1.0, beam_width=0.06, n_beam=5, random_seed=0
            )


def test_random_sky_gives_up_placing_crowded_sources():
    """Not provably impossible, but the placement loop never gets there."""
    with fail_after(120):
        with pytest.raises(SourcePlacementError) as err:
            generate_random_sky(
                n_src=10, freqs=FREQS, fov=1.0, beam_width=0.06, n_beam=5, random_seed=0
            )

    msg = str(err.value)
    assert f"gave up after {sky.MAX_SEP_ROUNDS} rounds" in msg
    for name in ["n_src = 10", "n_beam", "beam_width", "fov", "density"]:
        assert name in msg


def test_random_sky_gives_up_on_the_pair_budget(monkeypatch):
    """Many sources are stopped by the work done, long before the round limit."""
    monkeypatch.setattr(sky, "MAX_SEP_PAIRS", 100_000)

    with fail_after(60):
        with pytest.raises(SourcePlacementError, match="gave up after") as err:
            generate_random_sky(
                n_src=100,
                freqs=FREQS,
                fov=1.0,
                beam_width=0.012,
                n_beam=5,
                random_seed=0,
            )

    n_rounds = int(str(err.value).split("gave up after ")[1].split(" rounds")[0])
    assert 0 < n_rounds < 100


@pytest.mark.parametrize(
    "fov,beam_width",
    [
        (1.0, -1.0),  # no distance is less than a negative separation
        (np.int64(3037000500), 0),  # (fov + min_sep) ** 2 overflows 64 bit integers
        (-1.0, 0.01),  # points are drawn within |fov|
    ],
)
def test_random_sky_precheck_only_rejects_what_it_can_prove(fov, beam_width):
    rng, rng_ref = np.random.default_rng(0), np.random.default_rng(0)

    with fail_after(30):
        I_ref, ra_ref, dec_ref = reference_random_sky(
            2, FREQS, 1e-4, 1.0, fov, beam_width, rng_ref, 1
        )
        I, d_ra, d_dec = generate_random_sky(
            n_src=2,
            freqs=FREQS,
            fov=fov,
            beam_width=beam_width,
            n_beam=1,
            random_seed=rng,
        )

    np.testing.assert_array_equal(np.asarray(d_ra), np.asarray(ra_ref))
    np.testing.assert_array_equal(np.asarray(d_dec), np.asarray(dec_ref))
    assert rng.uniform() == rng_ref.uniform()


def test_random_sky_no_room_in_a_point_raises():
    with fail_after(30):
        with pytest.raises(SourcePlacementError, match="no such arrangement"):
            generate_random_sky(
                n_src=2, freqs=FREQS, fov=0.0, beam_width=0.01, n_beam=5, random_seed=0
            )


def test_random_sky_single_source_ignores_separation():
    I, d_ra, d_dec = generate_random_sky(
        n_src=1, freqs=FREQS, fov=1.0, beam_width=10.0, n_beam=5, random_seed=0
    )

    assert np.asarray(d_ra).shape == (1,)


def test_random_sky_min_flux_above_max_raises():
    with fail_after(10):
        with pytest.raises(ValueError, match="I_min = 1.5") as err:
            generate_random_sky(
                n_src=2, freqs=FREQS, min_I=1.5, max_I=1.0, fov=1.0, random_seed=0
            )

    assert not isinstance(err.value, SourcePlacementError)
