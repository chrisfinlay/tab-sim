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


def reference_power_law(n_src, I_min, I_max, alpha, rng):
    """The rejection sampler as it was before its loop was bounded."""

    def inv_cdf(x):
        return I_min * (1.0 - x) ** (1.0 / (1.0 - alpha))

    I = np.array(inv_cdf(rng.uniform(size=(n_src,))))
    while np.any(I > I_max):
        idx = np.where(I > I_max)[0]
        I[idx] = inv_cdf(rng.uniform(size=(idx.shape[0],)))

    return I


def reference_positions(n_src, fov, min_sep, rng):
    """The minimum separation loop as it was before it was bounded."""
    positions = uniform_points_disk(fov / 2.0, 1, rng)
    while positions.shape[1] < n_src:
        n_sample = 2 * (n_src - positions.shape[1])
        new_positions = uniform_points_disk(fov / 2.0, n_sample, rng)
        positions = np.concatenate([positions, new_positions], axis=1)
        s1, s2 = np.triu_indices(positions.shape[1], 1)
        d = np.linalg.norm(positions[:, s1] - positions[:, s2], axis=0)
        idx = np.where(d < min_sep)[0]
        remove_source_idx = np.unique(np.concatenate([s1[idx], s2[idx]]))
        positions = np.delete(positions, remove_source_idx, axis=1)

    return positions[:, :n_src]


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


def test_truncated_inv_cdf_inverts_the_truncated_cdf():
    I_min, I_max, alpha = 1e-4, 1.0, 1.6
    a = 1.0 - alpha
    u = np.linspace(0.0, 1.0, 1001)

    I = truncated_power_law_inv_cdf(u, I_min, I_max, alpha)

    np.testing.assert_allclose(I[[0, -1]], [I_min, I_max], rtol=1e-12)
    assert np.all(np.diff(I) > 0)
    np.testing.assert_allclose((I_min**a - I**a) / (I_min**a - I_max**a), u, atol=1e-12)


def test_power_law_zero_sources():
    assert random_power_law(0, random_seed=0).shape == (0,)


#######################
# generate_random_sky
#######################

FREQS = np.array([1.227e9, 1.228e9])


def test_random_sky_is_unchanged():
    """Same seed, same sky as before the loops were bounded."""
    n_src, fov, beam_width, n_beam = 50, 1.27, 200.0 / 3600 / 5, 5
    rng_ref = np.random.default_rng(123456)
    I_ref = reference_power_law(n_src, 1e-3, 1.0, 1.6, rng_ref)
    ra_ref, dec_ref = reference_positions(n_src, fov, n_beam * beam_width, rng_ref)

    I, d_ra, d_dec = generate_random_sky(
        n_src=n_src,
        freqs=FREQS,
        min_I=1e-3,
        max_I=1.0,
        fov=fov,
        beam_width=beam_width,
        random_seed=123456,
        n_beam=n_beam,
    )

    np.testing.assert_array_equal(np.asarray(I)[:, 0], I_ref)
    np.testing.assert_array_equal(np.asarray(d_ra), np.asarray(ra_ref))
    np.testing.assert_array_equal(np.asarray(d_dec), np.asarray(dec_ref))


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
