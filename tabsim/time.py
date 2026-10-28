"""Sidereal angles and the direction a correlator tracks for a J2000 phase centre.

The phase-tracking geometry pairs a sidereal angle with a direction, and the two
must be referred to the same equator and equinox. GAST is measured from the true
equinox of date, so it pairs with the phase centre's apparent place of date, not
with its J2000 coordinates. Pairing GAST with J2000 displaces every modelled RFI
source by the precession since J2000 -- 0.25 deg by 2019, growing by about 50
arcsec a year. See chrisfinlay/tab-sim#73 and epfl-radio-astro/tabascal#252.

:func:`apparent_radec_of_date` is the same definition as ``tabascal.time``'s
function of that name, deliberately: tabascal models what tab-sim simulates, so
the two packages have to agree on where the phase centre is. Both apply annual
aberration from pyerfa's analytic series and then skyfield's
bias-precession-nutation matrix.

Times here are read as UT1, as every other skyfield call in tab-sim does (see
:func:`skyfield_time`).
"""

from functools import lru_cache

import erfa
import numpy as np
from numpy.typing import NDArray
from skyfield.api import load


@lru_cache(maxsize=1)
def timescale():
    """Skyfield's timescale, loaded once per process."""

    return load.timescale()


def skyfield_time(times_jd):
    """Julian Dates as a :class:`skyfield.timelib.Time`, read as UTC.

    A Julian Date is a number until a scale says what it counts, and the scale
    is not cosmetic. tab-sim reads its epochs as UTC: that is what a config's
    ``start_time_isot`` and ``start_time_jd`` state, what the Measurement Set's
    ``TIME`` column is written as, and what tabascal reads that column back on.
    Reading the same number as UT1 instead omits DUT1, under 0.2 s at present,
    which turns the Earth by about 2 arcsec and drags every satellite along its
    track by the distance it covers in that time.

    Every skyfield call in tab-sim comes through here, so the antenna positions,
    the satellite trajectories, the sidereal angle and the tracked direction are
    all read on one scale.

    Only UTC is read. Honouring a scale that an input declares for itself, as
    tabascal does for an MS ``TIME`` column's ``MEASINFO`` record, is deferred.

    The Julian Date is split into whole and fractional parts before skyfield
    sees it: a JD's ~2.5e6 day magnitude leaves f64 only about 5e-10 days of
    resolution on the value as a whole, which is some 40 microseconds.

    This uses skyfield's private ``_utc_jd``, which is why ``pyproject.toml``
    pins skyfield. Keeping it to this one call site means the pin protects a
    single line.

    Parameters
    ----------
    times_jd : array_like
        Observation times as Julian Dates, on UTC.

    Returns
    -------
    skyfield.timelib.Time
        The same times, read as UTC.
    """

    times_jd = np.atleast_1d(np.asarray(times_jd, dtype=float))
    whole = np.floor(times_jd)

    return timescale()._utc_jd(whole, times_jd - whole)


def gast_deg(times_jd) -> NDArray:
    """Greenwich Apparent Sidereal Time in degrees, one per time.

    Apparent, not mean: it includes the equation of the equinoxes, so this is
    GAST and not GMST. It is the angle to pair with
    :func:`apparent_radec_of_date`, and it is the rotation skyfield's
    ITRS-to-GCRS transform uses, so the tracking term built from it shares a
    frame with the antenna positions the RFI distances are measured from.

    Parameters
    ----------
    times_jd : array_like
        Observation times as Julian Dates.

    Returns
    -------
    np.ndarray
        GAST in degrees.
    """

    return np.asarray(skyfield_time(times_jd).gast) * 15.0  # GAST hours -> degrees


def apparent_radec_of_date(ra_deg: float, dec_deg: float, times_jd):
    """A J2000 (ICRS) direction as the apparent geocentric RA/Dec of date.

    This is the direction a correlator tracks for a J2000 phase centre, on the
    true equator and equinox of date, and so the one to pair with
    :func:`gast_deg`: ``GAST - ra`` is then its Greenwich hour angle. It applies
    annual aberration (up to 20.5 arcsec) and then frame bias, precession and
    nutation (about 0.25 deg by 2019).

    Diurnal aberration (at most 0.32 arcsec, and common to the whole array) and
    gravitational light deflection (a few milliarcseconds away from the Sun) are
    not applied. The phase centre is a direction at infinity, so there is no
    parallax either.

    The Earth's barycentric velocity is the analytic ``erfa.epv00`` series, good
    to a few mm/s, or micro-arcseconds of aberration, with no ephemeris file to
    download, so this works on offline compute nodes. The
    bias-precession-nutation matrix is skyfield's, the same one :func:`gast_deg`
    and skyfield's ITRS-to-GCRS rotation use, so the tracked direction and the
    antenna positions share one frame.

    Parameters
    ----------
    ra_deg, dec_deg : float
        The J2000 phase centre in degrees, as a simulation config gives it.
    times_jd : array_like
        Observation times as Julian Dates.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Apparent RA in ``[0, 360)`` and Dec, in degrees, one per input time.
    """

    t = skyfield_time(times_jd)

    s_app = erfa.ab(_unit_vector(ra_deg, dec_deg), *_annual_aberration(t))
    s_date = np.einsum("ijt,tj->ti", np.reshape(t.M, (3, 3, -1)), s_app)

    return _radec_deg(s_date)


def tracking_basis_roll(ra_deg: float, dec_deg: float, times_jd) -> NDArray:
    """Roll of the uvw basis about w, in degrees, one per time.

    ``w`` points at the phase centre's apparent place of date
    (:func:`apparent_radec_of_date`), and :func:`tabsim.jax.coordinates.itrf_to_uvw`
    would otherwise complete the basis with the east and north of *that* frame's
    pole. Those are not the J2000 east and north, and
    :func:`tabsim.jax.coordinates.radec_to_lmn` takes its tangent basis from the
    pole of the frame it is handed, so J2000 direction cosines would not pair
    with them.

    This angle rolls the basis so that ``u`` and ``v`` are instead the J2000
    east and north carried rigidly onto the tracked frame. J2000 direction
    cosines then pair with them exactly, the sky model needs no transformation,
    and a position angle measured from J2000 north stays a position angle.

    It is small where the celestial pole is far away -- 0.08 deg at the equator,
    0.11 at a declination of -45 -- and large near it: 72 deg at a declination
    of -89.9, where the J2000 and of-date poles are 0.145 deg apart and the
    field sits between them. Completing the basis from the of-date pole there
    would rotate an extended source by that much.

    Parameters
    ----------
    ra_deg, dec_deg : float
        The J2000 phase centre in degrees.
    times_jd : array_like
        Observation times as Julian Dates.

    Returns
    -------
    np.ndarray
        The roll in degrees, one per input time.
    """

    t = skyfield_time(times_jd)

    east0, north0, w0 = _tangent_vectors(ra_deg, dec_deg)
    s_app = erfa.ab(w0, *_annual_aberration(t))

    # Rodrigues for the rigid rotation carrying w0 onto its apparent place,
    # applied to the J2000 east. The axis degenerates only where there is no
    # aberration to apply, and there the rotation is the identity.
    axis = np.cross(w0, s_app)
    sin_theta = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = np.divide(axis, sin_theta, out=np.zeros_like(axis), where=sin_theta > 0)
    cos_theta = (s_app @ w0)[:, None]
    east = (
        east0
        + sin_theta * np.cross(axis, east0)
        + (1.0 - cos_theta) * np.cross(axis, np.cross(axis, east0))
    )

    bpn = np.reshape(t.M, (3, 3, -1))  # GCRS -> true equator of date
    east = np.einsum("ijt,tj->ti", bpn, east)

    tracked = _tangent_vectors(*_radec_deg(np.einsum("ijt,tj->ti", bpn, s_app)))

    return np.rad2deg(
        np.arctan2(np.sum(east * tracked[1], axis=-1), np.sum(east * tracked[0], axis=-1))
    )


def _tangent_vectors(ra_deg, dec_deg):
    """East, north and line-of-sight unit vectors at each direction, ``(..., 3)``.

    East and north are taken with respect to the pole of whatever frame the
    coordinates are referred to, which is the whole reason
    :func:`tracking_basis_roll` exists.
    """

    ra = np.deg2rad(np.atleast_1d(np.asarray(ra_deg, dtype=float)))
    dec = np.deg2rad(np.atleast_1d(np.asarray(dec_deg, dtype=float)))
    zero = np.zeros_like(ra)

    east = np.stack([-np.sin(ra), np.cos(ra), zero], axis=-1)
    north = np.stack(
        [-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)], axis=-1
    )
    line_of_sight = np.stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1
    )

    if np.ndim(ra_deg) == 0 and np.ndim(dec_deg) == 0:
        return east[0], north[0], line_of_sight[0]

    return east, north, line_of_sight


def _unit_vector(ra_deg: float, dec_deg: float) -> NDArray:
    ra, dec = np.deg2rad(float(ra_deg)), np.deg2rad(float(dec_deg))

    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _radec_deg(s: NDArray):
    """Unit vectors ``(n, 3)`` to RA in ``[0, 360)`` and Dec, in degrees."""

    ra = np.rad2deg(np.arctan2(s[:, 1], s[:, 0])) % 360.0
    dec = np.rad2deg(np.arcsin(np.clip(s[:, 2], -1.0, 1.0)))

    return ra, dec


def _annual_aberration(t):
    """``erfa.ab``'s velocity, Sun distance and Lorentz factor for skyfield times."""

    pvh, pvb = erfa.epv00(t.whole, t.tdb_fraction)
    v = pvb["v"] / erfa.DC  # barycentric velocity in units of c

    return v, np.linalg.norm(pvh["p"], axis=-1), np.sqrt(1.0 - np.sum(v * v, axis=-1))
