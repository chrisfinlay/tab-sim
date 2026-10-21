"""The direction the visibilities are fringe-stopped to.

The phase-tracking geometry pairs a sidereal angle with a direction, and GAST is
measured from the true equinox of date, so it pairs with the phase centre's
apparent place of date and not with its J2000 RA/Dec. Pairing it with J2000
displaced every modelled RFI source by the precession since J2000: 15 m of path
on MeerKAT for a 2019 observation, which is 63 wavelengths at L band.

tab-sim#73, and tabascal#252 for the real-data reproducer.
"""

import erfa
import numpy as np
import pytest
from astropy.coordinates import SkyCoord, TETE, PrecessedGeocentric
from astropy.time import Time
from astropy.utils import iers
import astropy.units as u

from tabsim.dask.observation import Observation
from tabsim.jax.coordinates import mjd_to_gast, mjd_to_gmst, mjd_to_jd
from tabsim.time import (
    apparent_radec_of_date,
    apparent_radec_of_sources,
    gast_deg,
    skyfield_time,
)
from tabsim.time import _unit_vectors

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


def _radec_degrees(s):
    """One unit vector to RA in ``[0, 360)`` and Dec, in degrees."""

    return (
        np.rad2deg(np.arctan2(s[1], s[0])) % 360.0,
        np.rad2deg(np.arcsin(np.clip(s[2], -1.0, 1.0))),
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
    """Aberration is applied, not only frame bias, precession and nutation.

    tabascal#254 measured aberration improving its lock on a real satellite, so
    the correlator tracks the apparent place.

    The reference is the same direction carried to the equator of date by the
    bias-precession-nutation rotation alone, from erfa's `pnm06a`, so the
    separation measured here is the aberration and nothing else. astropy's
    `PrecessedGeocentric` is not that reference: it is GCRS based, so it carries
    aberration of its own, and it is on the mean rather than the true equinox,
    which would make this a measurement of nutation instead.
    """

    jd = mjd_to_jd(MJD_2019)
    obstime = Time(jd, format="jd", scale="ut1")
    ra_app, dec_app = apparent_radec_of_date(RA_J2000, DEC_J2000, [jd])

    bpn = erfa.pnm06a(obstime.tt.jd1, obstime.tt.jd2)
    ra_bpn, dec_bpn = _radec_degrees(bpn @ _unit_vectors(RA_J2000, DEC_J2000)[0])

    tracked = SkyCoord(
        ra=ra_app[0] * u.deg, dec=dec_app[0] * u.deg, frame=TETE(obstime=obstime)
    )
    unaberrated = SkyCoord(
        ra=ra_bpn * u.deg, dec=dec_bpn * u.deg, frame=TETE(obstime=obstime)
    )

    separation = tracked.separation(unaberrated).arcsec

    # Annual aberration is at most 20.5 arcsec, and 13.1 for this direction and
    # date. Removing it from apparent_radec_of_date drops this to zero.
    assert 5.0 < separation < 20.5
    assert separation == pytest.approx(13.1, abs=0.5)

    # And the mean-equinox frame really is the wrong reference: it sits far
    # closer to the unaberrated direction than a full aberration away.
    mean_equinox = SkyCoord(
        ra=RA_J2000 * u.deg, dec=DEC_J2000 * u.deg, frame="icrs"
    ).transform_to(PrecessedGeocentric(obstime=obstime, equinox=obstime))
    assert (
        SkyCoord(mean_equinox.ra, mean_equinox.dec, frame=TETE(obstime=obstime))
        .separation(unaberrated)
        .arcsec
        < separation
    )


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


def _gcrs_unit_vectors(ra_deg, dec_deg, time_jd):
    """J2000 directions as GCRS unit vectors, annual aberration included.

    GCRS shares the ICRS axes, so this is the frame ``ants_xyz`` is in, and the
    directions in it are the aberrated ones that a correlator tracks.
    """

    t = skyfield_time([time_jd])
    pvh, pvb = erfa.epv00(t.whole, t.tdb_fraction)
    v = pvb["v"][0] / erfa.DC

    return erfa.ab(
        _unit_vectors(ra_deg, dec_deg),
        v,
        np.linalg.norm(pvh["p"][0]),
        np.sqrt(1.0 - v @ v),
    )


def test_off_axis_astronomical_phase_matches_the_geometric_path():
    """u.l + v.m + w.(n-1) must be the real path difference, off axis too.

    The direction cosines and the uvw have to be on one equator and equinox.
    `radec_to_lmn` takes its tangent basis from the pole of the frame it is
    given, so J2000 source positions paired with the of-date uvw leave the two
    bases rotated about w by about 0.1 degrees. That is invisible on axis and
    grows with the offset: 0.2 wavelengths at 1 degree on these baselines, which
    is 76 degrees of phase.

    The reference here is independent of `radec_to_lmn` altogether: the baseline
    and both directions in GCRS, dotted directly.
    """

    offsets = np.array([0.0, 0.5, 1.0, 2.0])
    ra_src = RA_J2000 + offsets / np.cos(np.deg2rad(DEC_J2000))
    dec_src = np.full(offsets.shape, DEC_J2000)

    obs = _observation()
    obs.addAstro(
        I=np.ones((len(offsets), 1)), ra=ra_src, dec=dec_src
    )

    lmn = np.asarray(obs.ast_p_lmn[0].compute())
    bl_uvw = np.asarray(obs.bl_uvw[obs.t_idx].compute())
    ants_xyz = np.asarray(obs.ants_xyz.compute())[np.asarray(obs.t_idx)]
    a1, a2 = np.asarray(obs.a1), np.asarray(obs.a2)
    times_jd = mjd_to_jd(np.asarray(obs.times_mjd.compute()))

    modelled = np.einsum("tbi,si->tbs", bl_uvw, lmn) - bl_uvw[..., 2][:, :, None]

    expected = np.empty_like(modelled)
    for t, time_jd in enumerate(times_jd):
        baselines = ants_xyz[t, a1] - ants_xyz[t, a2]
        centre = _gcrs_unit_vectors(RA_J2000, DEC_J2000, time_jd)[0]
        sources = _gcrs_unit_vectors(ra_src, dec_src, time_jd)
        expected[t] = baselines @ (sources - centre).T

    error = np.abs(modelled - expected)
    wavelength = 299792458.0 / 1.227e9

    # On axis this is exact; the tolerance covers the whole range of offsets.
    assert error.max() / wavelength < 0.02, (
        f"off-axis phase off by {error.max():.4f} m "
        f"= {error.max() / wavelength:.3f} wavelengths"
    )
