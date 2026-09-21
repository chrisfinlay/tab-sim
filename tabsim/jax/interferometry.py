import jax.numpy as jnp
from jax import jit, random, config
from tabsim.precision import visibility_dtype
from jax.lax import scan

from functools import partial

c = 2.99792458e8


def _visibility_operand(value, precision):
    dtype = visibility_dtype(precision)
    value = jnp.asarray(value)
    return value.astype(
        dtype
        if jnp.iscomplexobj(value)
        else (jnp.float32 if dtype.itemsize == 8 else jnp.float64)
    )


def _require_x64():
    if not config.x64_enabled:
        raise ValueError(
            "Geometry and phase require JAX x64; set JAX_ENABLE_X64=true "
            "or jax.config.update('jax_enable_x64', True)"
        )


def _phasor(phase, precision):
    # Never cast the unwrapped phase: float32 loses fringes on long baselines.
    # Require x64 also for standalone kernel callers and remote Dask workers.
    _require_x64()
    dtype = visibility_dtype(precision)
    return jnp.exp(1.0j * phase).astype(dtype)


def rfi_vis(
    app_amplitude, c_distances, freqs, a1, a2, *, visibility_precision="single"
):
    """
    Calculate visibilities from distances to rfi sources.

    Parameters
    ----------
    app_amplitude: array_like (n_src, n_time, n_ant, n_freq)
        Apparent amplitude at the antennas.
    c_distances: array_like (n_src, n_time, n_ant)
        The phase corrected distances between the rfi sources and the antennas in metres.
    freqs: array_like (n_freq,)
        Frequencies in Hz.
    a1: array_like (n_bl,)
        Antenna 1 indexes, between 0 and n_ant-1.
    a2: array_like (n_bl,)
        Antenna 2 indexes, between 0 and n_ant-1.

    Returns
    -------
    vis: array_like (n_time, n_bl, n_freq)
        The visibilities.
    """
    _require_x64()
    app_amplitude = jnp.asarray(app_amplitude)
    c_distances = jnp.asarray(c_distances, dtype=jnp.float64)
    n_src = app_amplitude.shape[0]
    vis = _rfi_vis(
        app_amplitude[0, None],
        c_distances[0, None],
        freqs,
        a1,
        a2,
        visibility_precision=visibility_precision,
    )

    # This is a scan over the sources, but we can't use scan it unless we jit decorate this function
    def _add_vis(vis, i):
        return (
            vis
            + _rfi_vis(
                app_amplitude[i, None],
                c_distances[i, None],
                freqs,
                a1,
                a2,
                visibility_precision=visibility_precision,
            ),
            i,
        )

    return scan(_add_vis, vis, jnp.arange(1, n_src))[0]
    # return _rfi_vis(app_amplitude, c_distances, freqs, a1, a2)


def astro_vis(sources, uvw, lmn, freqs, *, visibility_precision="single"):
    """
    Calculate visibilities from a set of point sources using DFT.

    Parameters
    ----------
    sources: array_like (n_src, n_time, n_freq)
        Array of point source intensities in Jy.
    uvw: array_like (ntime, n_bl, 3)
        (u,v,w) coordinates of each baseline.
    lmn: array_like (n_src, 3)
        (l,m,n) coordinate of each source.
    freqs: array_like (n_freq,)
        Frequencies in Hz.

    Returns
    -------
    vis: array_like (n_time, n_bl, n_freq)
        Visibilities of the given set of sources and baselines.
    """
    _require_x64()
    sources = jnp.asarray(sources)
    lmn = jnp.asarray(lmn, dtype=jnp.float64)
    n_src = sources.shape[0]
    vis = _astro_vis(
        sources[0, None],
        uvw,
        lmn[0, None],
        freqs,
        visibility_precision=visibility_precision,
    )

    # This is a scan over the sources, but we can't use scan it unless we jit decorate this function
    @jit
    def _add_vis(vis, i):
        return (
            vis
            + _astro_vis(
                sources[i, None],
                uvw,
                lmn[i, None],
                freqs,
                visibility_precision=visibility_precision,
            ),
            i,
        )

    return scan(_add_vis, vis, jnp.arange(1, n_src))[0]


def astro_vis_gauss(
    sources, major, minor, pos_angle, uvw, lmn, freqs, *, visibility_precision="single"
):
    """
    Calculate visibilities from a set of point sources using DFT.

    Parameters
    ----------
    sources: array_like (n_src, n_time, n_freq)
        Array of point source intensities in Jy.
    shapes: array_like (n_src,)
        Array of standard deviations of the gaussian shape sources. These are
        assumed to be circular gaussians for now.
    uvw: array_like (ntime, n_bl, 3)
        (u,v,w) coordinates of each baseline.
    lmn: array_like (n_src, 3)
        (l,m,n) coordinate of each source.
    freqs: array_like (n_freq,)
        Frequencies in Hz.

    Returns
    -------
    vis: array_like (n_time, n_bl, n_freq)
        Visibilities of the given set of sources and baselines.
    """
    _require_x64()
    sources = jnp.asarray(sources)
    lmn = jnp.asarray(lmn, dtype=jnp.float64)
    n_src = sources.shape[0]
    major, minor, pos_angle = (
        jnp.asarray(v, dtype=jnp.float64) for v in (major, minor, pos_angle)
    )
    vis = _astro_vis_gauss(
        sources[0, None],
        major[0, None],
        minor[0, None],
        pos_angle[0, None],
        uvw,
        lmn[0, None],
        freqs,
        visibility_precision=visibility_precision,
    )

    # This is a scan over the sources, but we can't use scan it unless we jit decorate this function
    def _add_vis(vis, i):
        return (
            vis
            + _astro_vis_gauss(
                sources[i, None],
                major[i, None],
                minor[i, None],
                pos_angle[i, None],
                uvw,
                lmn[i, None],
                freqs,
                visibility_precision=visibility_precision,
            ),
            i,
        )

    return scan(_add_vis, vis, jnp.arange(1, n_src))[0]


def astro_vis_exp(sources, shapes, uvw, lmn, freqs, *, visibility_precision="single"):
    """
    Calculate visibilities from a set of point sources using DFT.

    Parameters
    ----------
    sources: array_like (n_src, n_time, n_freq)
        Array of point source intensities in Jy.
    shapes: array_like (n_src,)
        Array of standard deviations of the gaussian shape sources. These are
        assumed to be circular gaussians for now.
    uvw: array_like (ntime, n_bl, 3)
        (u,v,w) coordinates of each baseline.
    lmn: array_like (n_src, 3)
        (l,m,n) coordinate of each source.
    freqs: array_like (n_freq,)
        Frequencies in Hz.

    Returns
    -------
    vis: array_like (n_time, n_bl, n_freq)
        Visibilities of the given set of sources and baselines.
    """
    _require_x64()
    sources = jnp.asarray(sources)
    lmn = jnp.asarray(lmn, dtype=jnp.float64)
    n_src = sources.shape[0]
    shapes = jnp.asarray(shapes, dtype=jnp.float64)
    vis = _astro_vis_exp(
        sources[0, None],
        shapes[0, None],
        uvw,
        lmn[0, None],
        freqs,
        visibility_precision=visibility_precision,
    )

    # This is a scan over the sources, but we can't use scan it unless we jit decorate this function
    def _add_vis(vis, i):
        return (
            vis
            + _astro_vis_exp(
                sources[i, None],
                shapes[i, None],
                uvw,
                lmn[i, None],
                freqs,
                visibility_precision=visibility_precision,
            ),
            i,
        )

    return scan(_add_vis, vis, jnp.arange(1, n_src))[0]


def ants_to_bl(G, a1, a2):
    """
    Calculate the complex gains for each baseline given the per antenna gains.

    Parameters
    ----------
    G: array_like (n_time, n_ant, n_freq)
        Complex gains at each antenna over time.
    a1: array_like (n_bl,)
        Antenna 1 indexes, between 0 and n_ant-1.
    a2: array_like (n_bl,)
        Antenna 2 indexes, between 0 and n_ant-1.

    Returns
    -------
    G_bl: array_like (n_time, n_bl, n_freq)
        Complex gains on each baseline over time.
    """

    return _ants_to_bl(G, a1, a2)


def minus_two_pi_over_lamda(freqs):
    """Calculate -2pi/lambda for each frequency.

    Args:
        freqs (jnp.ndarray): Frequencies in Hz. (n_freq,)

    Returns:
        jnp.ndarray: -2pi/lambda for each frequency. (n_freq,)
    """
    return -2.0 * jnp.pi * jnp.asarray(freqs, dtype=jnp.float64) / c


def amp_to_intensity(amps, a1, a2):
    """Calculate intensity on a baseline ffrom the amplitudes at each antenna.

    Args:
        amps (jnp.ndarray): Amplitudes at the antennas. (n_src, n_time, n_int, n_ant, n_freq)
        a1 (jnp.ndarray): Antenna 1 indexes, between 0 and n_ant-1. (n_bl,)
        a2 (jnp.ndarray): Antenna 2 indexes, between 0 and n_ant-1. (n_bl,)

    Returns:
        jnp.ndarray: Intensity on baselines.
    """
    return amps[:, :, :, a1] * jnp.conjugate(amps[:, :, :, a2])


def phase_from_distances(distances, a1, a2, freqs):
    """Calculate phase differences between antennas from distances.

    Args:
        distances (jnp.ndarray): Distances to antennas. (n_src, n_time, n_int, n_ant)
        a1 (jnp.ndarray): Antenna 1 indexes, between 0 and n_ant-1. (n_bl,)
        a2 (jnp.ndarray): Antenna 2 indexes, between 0 and n_ant-1. (n_bl,)
        freqs (jnp.ndarray): Frequencies in Hz. (n_freq,)

    Returns:
        jnp.ndarray: Phases on baselines.
    """
    # Create array of shape (n_src, n_time, n_bl, n_freq)
    freqs = jnp.asarray(freqs, dtype=jnp.float64)[None, None, None, None, :]
    distances = jnp.asarray(distances, dtype=jnp.float64)[:, :, :, :, None]

    phases = minus_two_pi_over_lamda(freqs) * (
        distances[:, :, :, a1, :] - distances[:, :, :, a2, :]
    )

    return phases


def _rfi_vis(
    app_amplitude, c_distances, freqs, a1, a2, *, visibility_precision="single"
):
    # Create array of shape (n_src, n_time, n_bl, n_freq), then sum over n_src

    app_amplitude = _visibility_operand(app_amplitude, visibility_precision)
    c_distances = jnp.asarray(c_distances, dtype=jnp.float64)
    freqs = jnp.asarray(freqs, dtype=jnp.float64)
    a1 = jnp.asarray(a1)
    a2 = jnp.asarray(a2)

    phase = phase_from_distances(c_distances, a1, a2, freqs)
    intensity = amp_to_intensity(app_amplitude, a1, a2)

    vis = jnp.sum(intensity * _phasor(phase, visibility_precision), axis=0)
    vis_avg = jnp.mean(vis, axis=1)

    return vis_avg


def _astro_vis(sources, uvw, lmn, freqs, *, visibility_precision="single"):
    #     Create array of shape (n_src, n_time, n_bl, n_freq), then sum over n_src

    sources = _visibility_operand(
        sources[:, :, None, :], visibility_precision
    )  #     (n_src, 1, 1, n_freq)
    freqs = jnp.asarray(
        freqs[None, None, None, :], dtype=jnp.float64
    )  #      (1, 1, 1, n_freq)
    uvw = jnp.asarray(
        uvw[None, :, :, None, :], dtype=jnp.float64
    )  #          (1, n_time, n_bl, 1, 3)
    lmn = jnp.asarray(
        lmn[:, None, None, None, :], dtype=jnp.float64
    )  #       (n_src, 1, 1, 1, 3)
    s0 = jnp.array([0, 0, 1])[None, None, None, None, :]  # (1, 1, 1, 1, 3)

    phase = minus_two_pi_over_lamda(freqs) * jnp.sum(uvw * (lmn - s0), axis=-1)

    vis = jnp.sum(sources * _phasor(-phase, visibility_precision), axis=0)

    return vis


def gauss(uvw, shapes, freqs):
    uv_mag = jnp.linalg.norm(uvw[..., :-1], axis=-1) / (c / freqs)

    sigmas = shapes / (2.0 * jnp.sqrt(2.0 * jnp.log(2)))

    sigmas_uv = 1.0 / (2.0 * jnp.pi * sigmas)

    return jnp.exp(-((uv_mag / sigmas_uv) ** 2))


def source_to_abc(major, minor, pa):
    """Calculate the coefficients of the quadratic for a Gaussian source."""
    sigma_factor = 2 * jnp.sqrt(2 * jnp.log(2))
    sigma_x = jnp.deg2rad(minor / 3600) / sigma_factor
    sigma_y = jnp.deg2rad(major / 3600) / sigma_factor
    theta = jnp.deg2rad(pa)

    a = jnp.cos(theta) ** 2 / (2 * sigma_x**2) + jnp.sin(theta) ** 2 / (2 * sigma_y**2)
    b = jnp.sin(2 * theta) / (4 * sigma_x**2) - jnp.sin(2 * theta) / (4 * sigma_y**2)
    c = jnp.sin(theta) ** 2 / (2 * sigma_x**2) + jnp.cos(theta) ** 2 / (2 * sigma_y**2)

    return a, b, c


def gauss_uv(uvw, major, minor, pos_a, freqs):

    cc = 2.99792458e8
    lamda = cc / freqs
    u = uvw[..., 0] / lamda
    v = uvw[..., 1] / lamda
    a, b, c = source_to_abc(major, minor, pos_a)
    det = (a * c - b**2) / (4 * jnp.pi**2)

    return jnp.exp(-(c * u**2 - 2 * b * u * v + a * v**2) / (4 * det))


def gauss_lm(l, m, a, b, c):
    return jnp.exp(-(a * l**2 + 2 * b * l * m + c * m**2))


def _astro_vis_gauss(
    sources, major, minor, pos_angle, uvw, lmn, freqs, *, visibility_precision="single"
):
    #     Create array of shape (n_src, n_time, n_bl, n_freq), then sum over n_src

    sources = _visibility_operand(
        sources[:, :, None, :], visibility_precision
    )  #     (n_src, n_time, 1, n_freq)
    major = jnp.asarray(
        major[:, None, None, None], dtype=jnp.float64
    )  #     (n_src, 1, 1, 1)
    minor = jnp.asarray(
        minor[:, None, None, None], dtype=jnp.float64
    )  #     (n_src, 1, 1, 1)
    pos_angle = jnp.asarray(
        pos_angle[:, None, None, None], dtype=jnp.float64
    )  #     (n_src, 1, 1, 1)
    freqs = jnp.asarray(
        freqs[None, None, None, :], dtype=jnp.float64
    )  #      (1, 1, 1, n_freq)
    uvw = jnp.asarray(
        uvw[None, :, :, None, :], dtype=jnp.float64
    )  #          (1, n_time, n_bl, 1, 3)
    lmn = jnp.asarray(
        lmn[:, None, None, None, :], dtype=jnp.float64
    )  #       (n_src, 1, 1, 1, 3)
    s0 = jnp.array([0, 0, 1])[None, None, None, None, :]  # (1, 1, 1, 1, 3)

    phase = minus_two_pi_over_lamda(freqs) * jnp.sum(uvw * (lmn - s0), axis=-1)

    uv_filter = _visibility_operand(
        gauss_uv(uvw, major, minor, pos_angle, freqs), visibility_precision
    )

    vis = jnp.sum(uv_filter * sources * _phasor(-phase, visibility_precision), axis=0)

    return vis


def exp_uv(uvw, shapes, freqs):
    U = jnp.linalg.norm(uvw[..., :-1], axis=-1) / (c / freqs)

    return 1.0 / (1.0 + (2 * jnp.pi * shapes * U) ** 2) ** 1.5


def _astro_vis_exp(sources, shapes, uvw, lmn, freqs, *, visibility_precision="single"):
    #     Create array of shape (n_src, n_time, n_bl, n_freq), then sum over n_src

    sources = _visibility_operand(
        sources[:, :, None, :], visibility_precision
    )  #     (n_src, n_time, 1, n_freq)
    shapes = jnp.asarray(
        shapes[:, None, None, None], dtype=jnp.float64
    )  #     (n_src, 1, 1, 1)
    freqs = jnp.asarray(
        freqs[None, None, None, :], dtype=jnp.float64
    )  #      (1, 1, 1, n_freq)
    uvw = jnp.asarray(
        uvw[None, :, :, None, :], dtype=jnp.float64
    )  #          (1, n_time, n_bl, 1, 3)
    lmn = jnp.asarray(
        lmn[:, None, None, None, :], dtype=jnp.float64
    )  #       (n_src, 1, 1, 1, 3)
    s0 = jnp.array([0, 0, 1])[None, None, None, None, :]  # (1, 1, 1, 1, 3)

    phase = minus_two_pi_over_lamda(freqs) * jnp.sum(uvw * (lmn - s0), axis=-1)

    vis = jnp.sum(
        _visibility_operand(exp_uv(uvw, shapes, freqs), visibility_precision)
        * sources
        * _phasor(-phase, visibility_precision),
        axis=0,
    )

    return vis


def _ants_to_bl(G, a1, a2):
    G_bl = G[:, a1, :] * jnp.conjugate(G[:, a2, :])

    return G_bl


# Compatibility import: beam blocks execute on NumPy/SciPy. The historical
# public name remains available, but now explicitly returns a host ndarray.
from tabsim.beam import airy_beam


def Pv_to_Sv(Pv: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """
    Convert emission power to received intensity in Jy. Assumes constant
    power across the bandwidth.

    Parameters
    ----------
    Pv: ndarray (n_src, n_time, n_freq)
        Specific emission power in W/Hz.
    d: ndarray (n_src, n_time, n_ant)
        Distances from source to receiving antennas in m.

    Returns
    -------
    Sv: ndarray (n_src, n_time, n_ant, n_freq)
        Spectral flux density at the receiving antennas in Jy.
    """
    Pv = jnp.asarray(Pv)
    d = jnp.asarray(d)
    return Pv[:, :, None, :] / (4 * jnp.pi * d[:, :, :, None] ** 2) * 1e26


def add_noise(vis: jnp.ndarray, noise_std: jnp.ndarray, key: jnp.ndarray):
    """
    Add complex gaussian noise to the integrated visibilities. The real and
    imaginary components will each get this level of noise.

    Parameters
    ----------
    vis: ndarray (n_time, n_bl, n_freq)
        The visibilities to add noise to.
    noise_std: (n_freq, )
        Standard deviation of each real/imaginary component.
    key: jax.random.PRNGKey
        Random number generator key.
    """
    vis = jnp.asarray(vis)
    noise_std = jnp.asarray(noise_std)
    key = jnp.asarray(key)
    noise = (
        random.normal(key, shape=vis.shape, dtype=jnp.complex128)
        * jnp.sqrt(2.0)  # JAX complex normal has variance 1/2 per component.
        * noise_std[None, None, :]
    )
    noise = noise.astype(jnp.result_type(vis.dtype, jnp.complex64))
    return vis + noise, noise


def SEFD_to_noise_std(
    SEFD: jnp.ndarray, chan_width: jnp.ndarray, int_time: jnp.ndarray
):
    """Calculate the per-real/imaginary-component noise standard deviation
    given the system equivalent flux density, the channel width and integration time.

    Parameters
    ----------
    SEFD: ndarray (n_freq, )
        System equivalent flux density in Jy.
    chan_width: ndarray (n_time, n_ant, n_freq)
        Channel width in Hz.
    int_time: float
        Integration time in seconds.

    Returns
    -------
    noise_std: ndarray (n_time, n_ant, n_freq)
        Standard deviation of each real/imaginary component in Jy.
    """
    SEFD = jnp.asarray(SEFD)
    chan_width = jnp.asarray(chan_width)
    int_time = jnp.asarray(int_time)
    return SEFD / jnp.sqrt(2 * chan_width * int_time)


def int_sample_times(times: jnp.ndarray, n_int_samples: int = 1):
    """Calculate the times at which to sample the visibilities given the time centroids.
    This shoudl produce `n_int_samples` times per integration time that are evenly
    spaced around the time centroid.

    Parameters
    ----------
    times: ndarray (n_time, )
        The time centroids at which to sample the visibilities.
    n_int_samples: int
        The number of samples to take per integration time.

    Returns
    -------
    times_fine: ndarray (n_time * n_int_samples, )
        The times at which to sample the visibilities.
    """
    times = jnp.asarray(times)
    n_int_samples = jnp.asarray(n_int_samples)  # type: ignore
    int_time = times[1] - times[0]
    times_fine = (
        int_time / (2 * n_int_samples)
        + jnp.arange(
            times[0] - int_time / 2,
            times[-1] + int_time / 2,
            int_time / n_int_samples,
        )[: n_int_samples * len(times)]
    )
    return times_fine


def generate_gains(
    G0_mean: complex,
    G0_std: float,
    Gt_std_amp: float,
    Gt_std_phase: float,
    times: jnp.ndarray,
    n_ant: int,
    n_freq: int,
    key: jnp.ndarray,
):
    """
    Generate complex antenna gains. Gain amplitudes and phases
    are modelled as linear time-variates. Gains for all antennas at t = 0
    are randomly sampled from a Gaussian described by the G0 parameters.
    The rate of change of both ampltudes and phases are sampled from a zero
    mean Gaussian with standard deviation as provided.

    Parameters
    ----------
    G0_mean: complex
        Mean of Gaussian at t = 0.
    G0_std: float
        Standard deviation of Gaussian at t = 0.
    Gt_std_amp: float
        Standard deviation of Gaussian describing the rate of change in the
        gain amplitudes in 1/seconds.
    Gt_std_phase: float
        Standard deviation of Gaussian describing the rate of change in the
        gain phases in rad/seconds.
    key: jax.random.PRNGKey
        Random number generator key.
    """
    G0_mean = jnp.asarray(G0_mean)  # type: ignore
    G0_std = jnp.asarray(G0_std)  # type: ignore
    Gt_std_amp = jnp.asarray(Gt_std_amp)  # type: ignore
    Gt_std_phase = jnp.asarray(Gt_std_phase)  # type: ignore
    times = jnp.asarray(times) - jnp.asarray(times[0])
    n_ant = jnp.asarray(n_ant)  # type: ignore
    n_freq = jnp.asarray(n_freq)  # type: ignore
    key = jnp.asarray(key)
    G0 = G0_mean * jnp.exp(
        1.0j * jnp.pi * (random.uniform(key, (1, n_ant, n_freq)) - 0.5)
    )
    key, subkey = random.split(key)
    gains_noise = G0_std * random.normal(key, (n_ant, n_freq), dtype=jnp.complex128)
    key, subkey = random.split(key)

    gains_amp = Gt_std_amp * random.normal(key, (1, n_ant, 1)) * (times)[:, None, None]
    key, subkey = random.split(key)
    gains_phase = (
        Gt_std_phase * random.normal(key, (1, n_ant, 1)) * (times)[:, None, None]
    )
    key, subkey = random.split(key)
    gains_ants = G0 + gains_noise + gains_amp * jnp.exp(1.0j * gains_phase)
    gains_ants = gains_ants.at[:, -1, :].set(jnp.abs(gains_ants[:, -1, :]))
    return gains_ants


def apply_gains(
    vis_ast: jnp.ndarray,
    vis_rfi: jnp.ndarray,
    gains: jnp.ndarray,
    a1: jnp.ndarray,
    a2: jnp.ndarray,
):
    """Apply antenna gains to visibilities.

    Parameters
    ----------
    vis_ast: ndarray (n_time, n_bl, n_freq)
        The astronomical visibilities.
    vis_rfi: ndarray (n_time, n_bl, n_freq)
        The RFI visibilities.
    gains: ndarray (n_time, n_ant, n_freq)
        The antenna gains.
    a1: ndarray (n_bl,)
        The first antenna index for each baseline.
    a2: ndarray (n_bl,)
        The second antenna index for each baseline.

    Returns
    -------
    vis_obs: ndarray (n_time, n_bl, n_freq)
        The visibilities with gains applied.
    """
    vis_ast = jnp.asarray(vis_ast)
    vis_rfi = jnp.asarray(vis_rfi)
    gains = jnp.asarray(
        gains, dtype=jnp.result_type(vis_ast.dtype, vis_rfi.dtype, jnp.complex64)
    )
    a1 = jnp.asarray(a1)
    a2 = jnp.asarray(a2)
    vis_obs = gains[:, a1] * (vis_ast + vis_rfi) * jnp.conj(gains[:, a2])
    return vis_obs


@partial(jit, static_argnums=(1,))
def time_avg(vis: jnp.ndarray, n_int_samples: int = 1):
    """Average visibilities in time.

    Parameters
    ----------
    vis: ndarray (n_time_fine, n_bl, n_freq)
        The visibilities to average in time.
    n_int_samples: int
        The number of samples to take per integration time.

    Returns
    -------
    vis_avg: ndarray (n_time, n_bl, n_freq)
        The averaged visibilities.
    """
    vis = jnp.asarray(vis)
    # n_int_samples = jnp.asarray(n_int_samples)
    vis_avg = jnp.mean(
        jnp.reshape(vis, (-1, n_int_samples, vis.shape[1], vis.shape[2])),
        axis=1,
    )
    return vis_avg


def db_to_lin(dB: float):
    """
    Convert deciBels to linear units.

    Parameters
    ----------
    dB: float, ndarray
        deciBel value to convert.

    Returns
    -------
    lin: float, ndarray
    """
    dB = jnp.asarray(dB)  # type: ignore
    return 10.0 ** (dB / 10.0)
