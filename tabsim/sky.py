import numpy as np
import jax.numpy as jnp
import dask.array as da

from typing import Optional

# Limits on the rejection loops below so that they cannot spin forever.
# Rounds of redrawing fluxes above `I_max` before the rest are drawn directly.
MAX_FLUX_ROUNDS = 1_000
# Rounds of placing sources, and source pair separations evaluated, before giving up.
# A few sources can need thousands of rounds, which are cheap. The cost of a round
# grows as n_src**2, so many sources are stopped by the pair limit, within minutes.
MAX_SEP_ROUNDS = 10_000
MAX_SEP_PAIRS = 2_000_000_000


class SourcePlacementError(ValueError):
    """Random sources could not be placed the minimum separation apart."""


def uniform_points_disk(radius: float, n_src: int, random_seed: Optional[int] = None):
    """
    Generate uniformly distributed random points on a disk.

    Parameters:
    -----------
    radius: float
        Radius of the disk.
    n_src: int
        The number of sources/points to generate.
    seed: int
        Random number generator seed/key.

    Returns:
    --------
    points: array_like (2, n_src)
        The coordinate positions of the random points centred on (0,0).
    """
    rng = np.random.default_rng(random_seed)
    r = jnp.sqrt(rng.uniform(low=0.0, high=radius**2, size=(n_src,)))
    theta = rng.uniform(low=0.0, high=2.0 * jnp.pi, size=(n_src,))

    return r * jnp.array([jnp.cos(theta), jnp.sin(theta)])


def truncated_power_law_inv_cdf(
    x: np.ndarray, I_min: float, I_max: float, alpha: float
) -> np.ndarray:
    """Inverse cumulative distribution of a power law truncated to [`I_min`, `I_max`].

    Parameters:
    -----------
    x: array_like
        Cumulative probabilities in [0, 1].
    I_min: float
        Minimum source flux.
    I_max: float
        Maximum source flux.
    alpha: float
        Power law index. Must be greater than 1.

    Returns:
    --------
    I: array_like
        Source fluxes with the same shape as `x`."""
    a = 1.0 - alpha
    I = (I_min**a - x * (I_min**a - I_max**a)) ** (1.0 / a)

    return np.clip(I, I_min, I_max)


def random_power_law(
    n_src: int,
    I_min: float = 1e-4,
    I_max: float = 1.0,
    alpha: float = 1.6,
    random_seed: int = None,
):
    """Generate a random power law distribution of source fluxes between `I_min`
    and `I_max`.

    Parameters:
    -----------
    n_src: int
        Number of source fluxes to draw.
    I_min: float
        Minimum source flux.
    I_max: float
        Maximum source flux. Must not be less than `I_min`.
    alpha: float
        Power law index. Must be greater than 1.
    random_seed: int
        Random number generator seed/key.

    Returns:
    --------
    I: array_like (n_src,)
        Array of source fluxes."""

    if I_min > I_max:
        raise ValueError(
            f"The minimum flux I_min = {I_min} is greater than the maximum flux "
            f"I_max = {I_max}, so there is no range to draw source fluxes from."
        )

    def inv_cdf(x):
        return I_min * (1.0 - x) ** (1.0 / (1.0 - alpha))

    rng = np.random.default_rng(random_seed)
    rand_unif = rng.uniform(size=(n_src,))
    I = np.array(inv_cdf(rand_unif))
    n_rounds = 0
    while np.any(I > I_max):
        idx = np.where(I > I_max)[0]
        rand_unif = rng.uniform(size=(idx.shape[0],))
        if n_rounds < MAX_FLUX_ROUNDS:
            I[idx] = inv_cdf(rand_unif)
        else:
            # `I_max` is too close to `I_min` for redrawing to get there. Redrawing
            # samples the power law truncated to [I_min, I_max], so draw from that.
            I[idx] = truncated_power_law_inv_cdf(rand_unif, I_min, I_max, alpha)
        n_rounds += 1

    return I


def generate_random_sky(
    n_src: int,
    freqs: jnp.ndarray,
    min_I: float = 1e-4,
    max_I: float = 1.0,
    I_power_law: float = 1.6,
    spec_idx_mean: float = 0.7,
    spec_idx_std: float = 0.2,
    fov: float = 1.0,
    beam_width: float = 0.0,
    random_seed: int = None,
    n_beam: int = 3,
) -> tuple:
    """
    Generate uniformly distributed point sources inside the field of view with
    a power law intensity distribution. Setting the beam width will make
    sure souces are separated by 5 beam widths apart.

    Parameters:
    -----------
    n_src: int
        Number of sources to generate.
    freqs: array_like (n_freq,)
        Frequencies to generate the sources at.
    min_I: float
        Minimum intensity of the sources.
    max_I: float
        Maximum intensity of the sources. Must not be less than `min_I`.
    I_power_law: float
        Power law index of the intensity distribution. Must be greater than 1.
    spec_idx_mean: float
        Mean spectral index of the sources.
    spec_idx_std: float
        Standard deviation of the spectral index of the sources.
    fov: float
        Field of view to generate positions within. Same units as beam_width.
    beam_width: float
        Width of the resolving beam to ensure sources do not overlap. Sources will be
        separated by >5*`beam_width`. Same units as fov.
    random_seed: int
        Random number generator seed/key.
    n_beam: int
        Number of beam_widths apart sources should be separated.

    Returns:
    --------
    I: array_like (n_src, n_freq)
        The sources intensities.
    delta_ra: array_like
        The sources right ascensions relative to (0,0).
    delta_dec: array_like
        The sources declinations relative to (0,0).

    Raises:
    -------
    ValueError
        If `min_I` is greater than `max_I`.
    SourcePlacementError
        If the sources cannot be placed `n_beam` beam widths apart within the field
        of view. Placement gives up after `MAX_SEP_ROUNDS` rounds or `MAX_SEP_PAIRS`
        source pair separations.
    """
    rng = np.random.default_rng(random_seed)

    min_sep = n_beam * beam_width

    def too_crowded(reason: str) -> SourcePlacementError:
        density = n_src * (min_sep / fov) ** 2 if fov > 0 else np.inf
        return SourcePlacementError(
            f"Could not place n_src = {n_src} sources at least n_beam * beam_width = "
            f"{float(n_beam):g} * {float(beam_width):.4g} = {float(min_sep):.4g} apart "
            f"within fov = {float(fov):.4g}: {reason}. The source density "
            f"n_src * (n_beam * beam_width / fov)**2 is {float(density):.2g} and "
            "placement is only reliable below about 0.1. Reduce n_src, n_beam or "
            "beam_width, or increase fov."
        )

    # Sources cannot be further apart than the FoV, nor can the area they keep clear
    # of each other exceed the area available.
    if n_src > 1 and (min_sep > fov or n_src * min_sep**2 > (fov + min_sep) ** 2):
        raise too_crowded("no such arrangement exists")

    I = da.atleast_1d(random_power_law(n_src, min_I, max_I, I_power_law, rng))
    positions = uniform_points_disk(fov / 2.0, 1, rng)
    n_rounds, n_pairs, n_placed = 0, 0, 1
    while positions.shape[1] < n_src:
        if n_rounds >= MAX_SEP_ROUNDS or n_pairs >= MAX_SEP_PAIRS:
            raise too_crowded(
                f"gave up after {n_rounds} rounds and {n_pairs:.2g} source pairs, "
                f"having placed at most {n_placed} sources at once"
            )
        n_sample = 2 * (n_src - positions.shape[1])
        new_positions = uniform_points_disk(fov / 2.0, n_sample, rng)
        positions = np.concatenate([positions, new_positions], axis=1)
        s1, s2 = np.triu_indices(positions.shape[1], 1)
        d = np.linalg.norm(positions[:, s1] - positions[:, s2], axis=0)
        idx = np.where(d < min_sep)[0]
        remove_source_idx = np.unique(jnp.concatenate([s1[idx], s2[idx]]))
        positions = np.delete(positions, remove_source_idx, axis=1)
        n_rounds += 1
        n_pairs += len(s1)
        n_placed = max(n_placed, positions.shape[1])

    d_ra, d_dec = positions[:, :n_src]

    spectral_indices = rng.normal(loc=spec_idx_mean, scale=spec_idx_std, size=(n_src,))
    I = I[:, None] * ((freqs[None, :] / freqs[0]) ** -spectral_indices[:, None])

    return I, d_ra, d_dec
