"""The direction the visibilities are fringe-stopped to.

The phase-tracking geometry pairs a sidereal angle with a direction, and GAST is
measured from the true equinox of date, so it pairs with the phase centre's
apparent place of date and not with its J2000 RA/Dec. Pairing it with J2000
displaced every modelled RFI source by the precession since J2000: 15 m of path
on MeerKAT for a 2019 observation, which is 63 wavelengths at L band.

tab-sim#73, and tabascal#252 for the real-data reproducer.
"""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord, TETE, PrecessedGeocentric
from astropy.time import Time
from astropy.utils import iers
import astropy.units as u

from tabsim.dask.observation import Observation
from tabsim.jax.coordinates import mjd_to_gast, mjd_to_gmst, mjd_to_jd
from tabsim.time import apparent_radec_of_date, gast_deg, skyfield_time

# MeerKAT, 2019-09-15 ~15:15 UTC, field CGCG021-063: the observation tabascal#252
# reproduces the displacement on.
MJD_2019 = 58741.6354
RA_J2000, DEC_J2000 = 229.1675, 0.2506

# Three MeerKAT dishes spanning ~4 km of the array, enough baseline for the
# tracking error to be metres of path.
ITRF = np.array(
    [
        [5109360.133, 2006852.586, -3238948.127],
        [5109350.000, 2006860.000, -3238940.000],
        [5107000.000, 2005000.000, -3241000.000],
    ]
)


@pytest.fixture(autouse=True)
def _offline_iers():
    """astropy is only an oracle here; it must not reach for the network to be one.

    The epochs below are in the past and inside the IERS-B table astropy ships,
    so the UT1 these tests need is available without a download.
    """

    download, max_age = iers.conf.auto_download, iers.conf.auto_max_age
    iers.conf.auto_download, iers.conf.auto_max_age = False, None
    try:
        yield
    finally:
        iers.conf.auto_download, iers.conf.auto_max_age = download, max_age


def _observation(**kwargs):
    options = dict(
        latitude=-30.7130,
        longitude=21.4430,
        elevation=1086.6,
        ra=RA_J2000,
        dec=DEC_J2000,
        ITRF_array=ITRF.tolist(),
        times_mjd=MJD_2019 + np.arange(4) * 8 / 86400,
        freqs=np.array([1.227e9]),
        SEFD=np.ones(1) * 420.0,
        int_time=8.0,
        n_int_samples=3,
        max_chunk_MB=1.0,
    )
    options.update(kwargs)

    return Observation(**options)


def _unit_vectors(ra_deg, dec_deg):
    """RA/Dec in degrees to unit vectors ``(n, 3)``."""

    ra, dec = np.deg2rad(np.atleast_1d(ra_deg)), np.deg2rad(np.atleast_1d(dec_deg))

    return np.stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1
    )


def _tracked_direction_gcrs(obs):
    """The tracked direction as a unit vector in the frame of ``obs.ants_xyz``.

    ``ants_xyz`` is GCRS, so the of-date direction the tracking term is built
    from is rotated back through skyfield's bias-precession-nutation matrix.
    """

    times_jd = mjd_to_jd(np.asarray(obs.times_mjd_fine.compute()))
    s_date = _unit_vectors(np.asarray(obs.tracked_ra), np.asarray(obs.tracked_dec))
    bpn = np.reshape(skyfield_time(times_jd).M, (3, 3, -1))  # GCRS -> of date

    return np.einsum("jit,tj->ti", bpn, s_date)


def test_apparent_place_matches_astropy_tete():
    """The tracked direction is the apparent geocentric place of date.

    astropy's TETE is an independent implementation of the same true-equator,
    true-equinox apparent place, aberration included.
    """

    for mjd, ra, dec in [
        (MJD_2019, RA_J2000, DEC_J2000),
        (55197.3, 83.633, 22.0145),
        (60000.8, 300.0, -60.0),
    ]:
        jd = mjd_to_jd(mjd)
        obstime = Time(jd, format="jd", scale="ut1")
        ra_app, dec_app = apparent_radec_of_date(ra, dec, [jd])

        expected = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(
            TETE(obstime=obstime)
        )
        actual = SkyCoord(
            ra=ra_app[0] * u.deg, dec=dec_app[0] * u.deg, frame=TETE(obstime=obstime)
        )

        assert actual.separation(expected).arcsec < 0.1


def test_apparent_place_includes_annual_aberration():
    """Aberration is applied, not only precession and nutation.

    tabascal#254 measured aberration improving its lock on a real satellite, so
    the correlator tracks the apparent place. Dropping it would leave the
    direction more than 10 arcsec out, and the precession-only place is that far
    from what this returns.
    """

    jd = mjd_to_jd(MJD_2019)
    obstime = Time(jd, format="jd", scale="ut1")
    ra_app, dec_app = apparent_radec_of_date(RA_J2000, DEC_J2000, [jd])

    tracked = SkyCoord(
        ra=ra_app[0] * u.deg, dec=dec_app[0] * u.deg, frame=TETE(obstime=obstime)
    )
    no_aberration = SkyCoord(
        ra=RA_J2000 * u.deg, dec=DEC_J2000 * u.deg, frame="icrs"
    ).transform_to(PrecessedGeocentric(obstime=obstime, equinox=obstime))

    separation = tracked.separation(
        SkyCoord(no_aberration.ra, no_aberration.dec, frame=TETE(obstime=obstime))
    ).arcsec

    assert 10.0 < separation < 21.0  # annual aberration is at most 20.5 arcsec


def test_tracked_direction_moves_off_j2000_by_the_precession():
    """The 2019 displacement tabascal#252 documents: +0.248 deg RA, -0.070 Dec."""

    ra_app, dec_app = apparent_radec_of_date(
        RA_J2000, DEC_J2000, [mjd_to_jd(MJD_2019)]
    )

    assert ra_app[0] - RA_J2000 == pytest.approx(0.248, abs=0.005)
    assert dec_app[0] - DEC_J2000 == pytest.approx(-0.070, abs=0.005)


def test_tracking_term_matches_geometric_delay_in_gcrs():
    """w is the delay toward the tracked centre, measured from the GCRS positions.

    The RFI distances are measured from ``ants_xyz`` in GCRS, so the tracking
    term that is added to them has to be the projection of those same positions
    onto the tracked direction. Pairing GAST with J2000 breaks exactly this, by
    about 15 m here.
    """

    obs = _observation()
    ants_xyz = np.asarray(obs.ants_xyz.compute())
    ants_uvw = np.asarray(obs.ants_uvw.compute())

    expected = np.einsum(
        "tai,ti->ta", ants_xyz - ants_xyz[:, :1], _tracked_direction_gcrs(obs)
    )

    np.testing.assert_allclose(ants_uvw[..., 2], expected, atol=1e-6)


def test_source_at_infinity_in_tracked_direction_fringe_stops_flat():
    """A source at infinity in the phase-centre direction gives one path for all antennas.

    This is the invariant the fringe-stopping term exists to produce, written in
    the quantity ``rfi_vis`` consumes: ``distance + w``. The source is put far
    enough away that its wavefront curvature across the array is well under the
    tolerance, which f64 resolution at that distance sets.
    """

    obs = _observation()
    ants_xyz = np.asarray(obs.ants_xyz.compute())
    ants_uvw = np.asarray(obs.ants_uvw.compute())

    distance = 1e12  # metres; curvature over 4 km is ~1e-5 m
    source_xyz = distance * _tracked_direction_gcrs(obs)

    distances = np.linalg.norm(source_xyz[:, None, :] - ants_xyz, axis=-1)
    path = distances + ants_uvw[..., 2]

    spread = np.abs(path - path[:, :1]).max()
    assert spread < 1e-2, f"fringe-stopped path varies by {spread} m across the array"


def test_j2000_phase_centre_is_what_the_output_records():
    """The field direction stays J2000; only the tracking geometry is of date."""

    obs = _observation()

    assert float(obs.ra.compute()) == RA_J2000
    assert float(obs.dec.compute()) == DEC_J2000

    # The tracked direction is a different, time-dependent quantity.
    assert np.asarray(obs.tracked_ra).shape == (obs.n_time_fine,)
    assert np.asarray(obs.tracked_dec).shape == (obs.n_time_fine,)
    assert abs(np.asarray(obs.tracked_ra).mean() - RA_J2000) > 0.2


def test_phase_tracking_convention_is_recorded():
    """So a consumer can rebuild the geometry rather than guess it (tabascal#253)."""

    from tabsim.write import get_observation_attributes

    obs = _observation()
    attrs = get_observation_attributes(obs)

    assert obs.phase_tracking == "apparent"
    assert attrs["phase_tracking"] == "apparent"
    assert attrs["target_ra"] == RA_J2000
    assert attrs["target_dec"] == DEC_J2000


def test_mjd_to_gast_is_apparent_not_mean_sidereal_time():
    """The equation of the equinoxes is applied in degrees, not radians.

    It used to be added in radians to a GMST in degrees, which applied it at
    1/57th of its size and left `mjd_to_gast` returning GMST, up to 17 arcsec out.
    """

    mjds = np.array([51544.5, 55000.2, MJD_2019, 60000.3, 61000.0, 62000.7])
    gast = np.asarray(mjd_to_gast(mjds))
    gmst = np.asarray(mjd_to_gmst(mjds))

    eqeq = (gast - gmst + 180) % 360 - 180
    reference = (gast_deg(mjd_to_jd(mjds)) - gmst + 180) % 360 - 180

    # The equation of the equinoxes is tens of arcsec and must be there in full.
    assert np.abs(eqeq).max() * 3600 > 10.0
    np.testing.assert_allclose(eqeq * 3600, reference * 3600, atol=0.5)


def test_mjd_to_gast_matches_skyfield():
    """tab-sim's analytic GAST and the skyfield GAST the geometry uses agree."""

    mjds = MJD_2019 + np.arange(5) * 8 / 86400
    analytic = np.asarray(mjd_to_gast(mjds))
    reference = gast_deg(mjd_to_jd(mjds))

    offset = (analytic - reference + 180) % 360 - 180
    assert np.abs(offset).max() * 3600 < 1.0
