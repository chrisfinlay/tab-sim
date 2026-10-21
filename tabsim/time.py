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
    """Julian Dates as a :class:`skyfield.timelib.Time`, read as UT1.

    tab-sim reads its observation times as UT1 throughout -- antenna positions
    (:func:`tabsim.dask.coordinates.itrs_to_gcrs_sf`) and satellite propagation
    (:mod:`tabsim.tle`) both call ``ut1_jd`` -- so the sidereal angle and the
    tracked direction are read the same way and the whole simulation stays on one
    time scale.

    Note this differs from tabascal, which reads a Measurement Set's ``TIME``
    column on the scale the column declares, almost always UTC. Reading a UTC
    epoch as UT1 omits DUT1, currently under 0.2 s, which rotates the Earth by
    about 2 arcsec. It is a separate question from the frame pairing this module
    exists for, and changing it would move every simulated satellite along its
    track, so it is left as it is here and tracked separately.

    Parameters
    ----------
    times_jd : array_like
        Observation times as Julian Dates.

    Returns
    -------
    skyfield.timelib.Time
        The same times, read as UT1.
    """

    return timescale().ut1_jd(np.atleast_1d(np.asarray(times_jd, dtype=float)))


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
