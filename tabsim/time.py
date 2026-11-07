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

Times here are read as UTC, as every skyfield call in tab-sim is (see
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

    A UTC Julian Date is not a count of 86400-second days: on a day carrying a
    leap second the extra second is folded into the same unit interval, so the
    fraction cannot simply be multiplied out. ``erfa.utctai`` decodes the date
    on that convention -- the one erfa and astropy share -- and hands skyfield
    TAI, which is a uniform count with no such ambiguity. casacore does not
    share it: an MS ``TIME`` counts 86400-second days, so it is converted where
    it is written and read (:func:`ms_time_from_utc_mjd`).

    Skyfield's ``_utc_jd`` does multiply the fraction out, and is up to a full
    second wrong across a leap-second day: an hour of 2016-12-31 read that way
    lands 0.5 to 1.0 s early, which is 7 km of satellite track. On every other
    day the two agree exactly.

    erfa's leap-second table is compiled in, so this needs no cache and no
    network. astropy would do the same conversion, but its first UTC-to-TAI
    call tries to refresh that table from IERS and IANA, which on an offline
    node stalls every worker on name resolution or connection before it falls
    back to the table it ships with.

    ``tai_jd`` takes the two-part Julian Date, so the day number and the
    fraction stay separate all the way through and no precision is lost.

    Parameters
    ----------
    times_jd : array_like
        Observation times as Julian Dates, on UTC.

    Returns
    -------
    skyfield.timelib.Time
        The same times, read as UTC, in the shape they were given.
    """

    times_jd = np.asarray(times_jd, dtype=float)
    whole = np.floor(times_jd)

    return timescale().tai_jd(*erfa.utctai(whole, times_jd - whole))


def utc_mjd_after(start_mjd, elapsed_seconds) -> NDArray:
    """UTC Modified Julian Dates a given number of SI seconds after a UTC start.

    A UTC day is not always 86400 seconds: one carrying a leap second has
    86401, and that extra second is folded into the same unit interval of
    Julian Date. ``start_mjd + seconds / 86400`` therefore drifts across such a
    day -- half a second by midday and a full second by the end, which is some
    7 km of satellite track. The elapsed seconds are added on TAI, which is a
    uniform count, and the result converted back to UTC. The day number and
    its fraction stay separate throughout, so no precision is lost to a full
    Julian Date.

    Where the start and a sample both fall on ordinary days, 86400 SI seconds
    with no leap at their end, and no leap lies between them, that sample is
    ``start_mjd + elapsed / 86400`` itself, bit for bit. Samples on a
    leap-second day, or separated from the start by a leap, in either
    direction, take the TAI route, as does anything before 1972.

    Parameters
    ----------
    start_mjd : float
        The first epoch, as a UTC Modified Julian Date.
    elapsed_seconds : array_like
        SI seconds elapsed since ``start_mjd``.

    Returns
    -------
    np.ndarray
        One UTC Modified Julian Date per elapsed time.
    """

    start_mjd = float(start_mjd)
    day = np.floor(start_mjd)
    elapsed = np.asarray(elapsed_seconds, dtype=float)
    naive = start_mjd + elapsed / 86400.0

    tai_day, tai_fraction = erfa.utctai(erfa.DJM0 + day, start_mjd - day)
    utc_day, utc_fraction = erfa.taiutc(
        np.broadcast_to(tai_day, elapsed.shape), tai_fraction + elapsed / 86400.0
    )
    stepped = (utc_day - erfa.DJM0) + utc_fraction

    # The naive sum is exact only when both endpoint days are 86400 SI seconds
    # long, with no leap at their end and no pre-1972 drift, and TAI - UTC is
    # the same on both, so that no leap lies between them in either direction.
    sample_day = np.floor(naive)
    uniform = (
        _ordinary_day(day)
        & _ordinary_day(sample_day)
        & (_tai_minus_utc(sample_day) == _tai_minus_utc(day))
    )

    return np.where(uniform, naive, stepped)


def ms_time_from_utc_mjd(times_mjd) -> NDArray:
    """A Measurement Set ``TIME``, in seconds, for UTC Modified Julian Dates.

    casacore does not read ``TIME`` on erfa's convention. It counts
    86400-second days and applies TAI - UTC as a step at 0h, so on a day
    carrying a leap second a second of day ``s`` is ``s / 86401`` of erfa's
    day, not ``s / 86400``. Multiplying the date out would put a casacore
    reader up to a full second away from the simulated instant by the end of
    such a day.

    The leap second itself has no name on casacore's count. A sample inside it
    is written as casacore's own TAI-to-UTC conversion writes it, as a repeat
    of the second before, and so reads back one second early.

    On a day with no leap second at its end this is ``times_mjd * 86400``, bit
    for bit. On a leap-second day every sample moves, by up to a second.

    Parameters
    ----------
    times_mjd : array_like
        Times as UTC Modified Julian Dates, on erfa's convention.

    Returns
    -------
    np.ndarray
        Seconds since MJD 0 on UTC, as casacore reads them.
    """

    mjd = np.asarray(times_mjd, dtype=float)
    day = np.floor(mjd)
    leap = _leap_at_end_of(day)

    seconds = (mjd - day) * (86400.0 + leap)
    seconds = np.where(seconds >= 86400.0, seconds - leap, seconds)

    return np.where(leap == 0.0, mjd * 86400.0, day * 86400.0 + seconds)


def utc_mjd_from_ms_time(seconds) -> NDArray:
    """UTC Modified Julian Dates, on erfa's convention, for an MS ``TIME``.

    The inverse of :func:`ms_time_from_utc_mjd` everywhere but the leap second
    itself, which casacore's count cannot name.
    """

    seconds = np.asarray(seconds, dtype=float)
    day = np.floor(seconds / 86400.0)
    leap = _leap_at_end_of(day)

    return np.where(
        leap == 0.0, seconds / 86400.0, day + (seconds - day * 86400.0) / (86400.0 + leap)
    )


def ms_time_tai_minus_utc(seconds) -> NDArray:
    """TAI - UTC, in seconds, at each MS ``TIME``, as casacore reckons it.

    Adding its differences to those of ``TIME`` gives elapsed SI seconds, which
    ``TIME`` alone is not across a leap second.
    """

    seconds = np.asarray(seconds, dtype=float)
    day = np.floor(seconds / 86400.0)
    year, month, date, _ = erfa.jd2cal(erfa.DJM0, day)

    return erfa.dat(year, month, date, (seconds - day * 86400.0) / 86400.0)


def _tai_minus_utc(mjd_day) -> NDArray:
    """TAI - UTC in seconds at 0h UTC of each whole MJD."""

    year, month, day, _ = erfa.jd2cal(erfa.DJM0, mjd_day)

    return erfa.dat(year, month, day, 0.0)


def _ordinary_day(mjd_day) -> NDArray:
    """Whether each whole MJD is 86400 SI seconds of uniform UTC: no leap, no drift."""

    year, month, day, _ = erfa.jd2cal(erfa.DJM0, mjd_day)
    start, noon = erfa.dat(year, month, day, 0.0), erfa.dat(year, month, day, 0.5)

    return (noon == start) & (_tai_minus_utc(mjd_day + 1.0) == start)


def _leap_at_end_of(mjd_day) -> NDArray:
    """Seconds a leap at the end of each whole MJD adds to it, as erfa reckons it.

    Pre-1972 drift is not a leap: it is carried by both conventions alike.
    """

    year, month, day, _ = erfa.jd2cal(erfa.DJM0, mjd_day)
    start, noon = erfa.dat(year, month, day, 0.0), erfa.dat(year, month, day, 0.5)

    return _tai_minus_utc(mjd_day + 1.0) - (start + 2.0 * (noon - start))


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

    return _gast_deg(skyfield_time(np.atleast_1d(times_jd)))


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

    tracked, _ = _tracked_frame(ra_deg, dec_deg, skyfield_time(np.atleast_1d(times_jd)))

    return _radec_deg(tracked)


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

    tracked, east = _tracked_frame(
        ra_deg, dec_deg, skyfield_time(np.atleast_1d(times_jd))
    )

    return _basis_roll(east, *_radec_deg(tracked))


def phase_frame(ra_deg: float, dec_deg: float, times_jd) -> NDArray:
    """GAST, the tracked RA and Dec, and the basis roll, in degrees, ``(n_time, 4)``.

    :func:`gast_deg`, :func:`apparent_radec_of_date` and
    :func:`tracking_basis_roll` in one pass, sharing the skyfield time, its
    nutation and the aberration, which dominate the cost. The simulation
    evaluates it one time chunk at a time.
    """

    t = skyfield_time(np.atleast_1d(times_jd))
    tracked, east = _tracked_frame(ra_deg, dec_deg, t)
    ra, dec = _radec_deg(tracked)

    return np.stack([_gast_deg(t), ra, dec, _basis_roll(east, ra, dec)], axis=-1)


def _gast_deg(t) -> NDArray:
    return np.asarray(t.gast) * 15.0  # GAST hours -> degrees


def _tracked_frame(ra_deg, dec_deg, t):
    """The tracked direction and the carried J2000 east, ``(n, 3)``, of date."""

    east0, _, w0 = _tangent_vectors(ra_deg, dec_deg)
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

    return np.einsum("ijt,tj->ti", bpn, s_app), np.einsum("ijt,tj->ti", bpn, east)


def _basis_roll(east, ra_deg, dec_deg) -> NDArray:
    """Angle from the tracked frame's east to the carried J2000 east, degrees."""

    tracked = _tangent_vectors(ra_deg, dec_deg)

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
