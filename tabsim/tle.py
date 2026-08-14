"""Satellite orbit records, propagation, and visibility windows.

Records come from the IAU CPS SatChecker service through :mod:`tabsim.orbit` — no
account or credentials are required, and the ``spacetrack`` dependency is gone.
Two record kinds are handled: the TLEs SatChecker's frozen archive serves up to
2026-07-11, and the OMM element sets it serves from 2026-07-12 onward. Everything
below works off an opaque *record*; :func:`earth_satellite` is the only place
that asks which kind it is.

The names are historical — "TLE" appears throughout tabsim's configuration and
its output schema — but a record here may be either kind.
"""

from astropy.time import Time
from astropy.coordinates import EarthLocation
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.positionlib import position_of_radec
from sgp4.api import WGS72, Satrec
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from tqdm import tqdm

import string
import random

from typing import Optional

from tabsim.orbit import (  # noqa: F401  OrbitError re-exported for callers
    DEFAULT_CACHE_REUSE_MAX_AGE_DAYS,
    DEFAULT_REMOTE_MAX_AGE_DAYS,
    OrbitError,
    get_orbits_by_id,
    require_complete_coverage,
    resolve_names,
    resolve_orbits,
)
from tabsim.satchecker.records import KIND_TLE, record_elements, record_kind


#: Julian Date of 1949 December 31 00:00 UT, the epoch SGP4 counts days from.
_SGP4_EPOCH_JD = 2433281.5


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def as_record(entry) -> dict:
    """Coerce *entry* to an orbit record.

    A record is normally a mapping (a resolved row from :mod:`tabsim.orbit`), but
    a bare ``(line1, line2)`` pair is accepted so the many places that have always
    passed two TLE lines — including simulations replayed from an older script —
    keep working unchanged.
    """
    if isinstance(entry, pd.Series):
        return entry.to_dict()
    if isinstance(entry, dict):
        return entry
    lines = list(entry)
    if len(lines) != 2:
        raise ValueError(
            f"an orbit record must be a mapping or a (line1, line2) pair, got "
            f"{len(lines)} items"
        )
    return {"TLE_LINE1": str(lines[0]), "TLE_LINE2": str(lines[1])}


def earth_satellite(record, ts) -> EarthSatellite:
    """A Skyfield ``EarthSatellite`` for one orbit record, whichever kind it is.

    A TLE goes through Skyfield's line parser exactly as it always has, so nothing
    about the TLE path changes. An OMM has no lines to parse — that is the whole
    point of the format — so its element set is loaded straight into an
    ``sgp4.Satrec`` via ``sgp4init``, which is the entry point the sgp4 library
    provides for precisely this. Both end up as the same propagator over the same
    model; only the way the elements are read in differs.

    Units: ``sgp4init`` wants radians and rad/min, while OMM (and tabsim's element
    columns) use degrees and rev/day.

    ``ndot`` and ``nddot`` are passed as zero. SGP4 models drag through ``bstar``
    alone and never reads them during propagation — they exist in the TLE format
    for other consumers — so dropping them costs nothing here.
    """
    record = as_record(record)
    if record_kind(record) == KIND_TLE:
        return EarthSatellite(record["TLE_LINE1"], record["TLE_LINE2"], ts=ts)

    elements = record_elements(record)
    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        "i",  # improved mode, matching what twoline2rv uses for the TLE path
        int(record["NORAD_CAT_ID"]),
        elements["EPOCH_JD"] - _SGP4_EPOCH_JD,
        float(elements["BSTAR"]),
        0.0,  # ndot: stored by the TLE format, unused by the propagator
        0.0,  # nddot: likewise
        float(elements["ECCENTRICITY"]),
        np.deg2rad(elements["ARG_OF_PERICENTER"]),
        np.deg2rad(elements["INCLINATION"]),
        np.deg2rad(elements["MEAN_ANOMALY"]),
        elements["MEAN_MOTION"] * 2.0 * np.pi / 1440.0,  # rev/day -> rad/min
        np.deg2rad(elements["RA_OF_ASC_NODE"]),
    )
    return EarthSatellite.from_satrec(satrec, ts)


def record_tle_lines(record) -> tuple[str, str]:
    """The two TLE lines of *record*, or two empty strings for an OMM record.

    tabsim's output schema has a fixed-width string column for the lines. An OMM
    record has none to give — its elements are the record — so the column is left
    empty for those rows rather than filled with something that looks like a TLE
    but is not one. The full record is written to ``used_orbits.json`` instead,
    which is what a later run reads back.
    """
    record = as_record(record)
    if record_kind(record) != KIND_TLE:
        return "", ""
    return str(record["TLE_LINE1"]), str(record["TLE_LINE2"])


def get_satellite_positions(records: list, times_jd: list) -> ArrayLike:
    """Calculate the ICRS positions of satellites by propagating their orbit records.

    Parameters
    ----------
    records : sequence of records, length n_sat
        Orbit records — TLE or OMM — as resolved by :mod:`tabsim.orbit`. A bare
        ``(line1, line2)`` pair is also accepted; see :func:`as_record`.
    times_jd : Array (n_time,)
        Times to calculate positions at in Julian date.

    Returns
    -------
    Array (n_sat, n_time, 3)
        Satellite positions over time, in metres.
    """

    ts = load.timescale()
    sf_times = ts.ut1_jd(times_jd)

    sat_pos = np.array(
        [
            earth_satellite(record, ts).at(sf_times).position.km.T * 1e3
            for record in records
        ]
    )

    return sat_pos


def ant_pos(ant_itrf: ArrayLike, times_jd: ArrayLike) -> ArrayLike:

    ts = load.timescale()
    t = ts.ut1_jd(times_jd)

    location = EarthLocation(x=ant_itrf[0], y=ant_itrf[1], z=ant_itrf[2], unit="m")
    observer = wgs84.latlon(
        location.lat.degree, location.lon.degree, location.height.value
    )

    return (observer.at(t).position.km * 1e3).T


def ants_pos(ants_itrf: ArrayLike, times_jd: ArrayLike) -> ArrayLike:

    return np.transpose(
        np.array([ant_pos(ant_itrf, times_jd) for ant_itrf in ants_itrf]),
        axes=(1, 0, 2),
    )


def sat_distance(record, times_jd: ArrayLike, obs_itrf: ArrayLike) -> ArrayLike:

    ts = load.timescale()

    t = ts.ut1_jd(times_jd)

    satellite = earth_satellite(record, ts)

    location = EarthLocation(x=obs_itrf[0], y=obs_itrf[1], z=obs_itrf[2], unit="m")

    observer = wgs84.latlon(location.lat.degree, location.lon.degree, location.height)

    topo = (satellite - observer).at(t)

    return topo.distance().m


def get_sat_pos_tle(record, sat_name: str, times_jd: float) -> ArrayLike:
    """Calculate the satellite position in GCRS (ECI) frame at the given Julian dates.

    Parameters
    ----------
    record : dict or (str, str)
        Orbit record, or a ``(line1, line2)`` TLE pair.
    sat_name : str
        Satellite name. Retained for call compatibility; Skyfield does not use it
        for propagation.
    times_jd : float
        Julian dates at which to evaluate the satellite position.

    Returns
    -------
    ArrayLike
        Satellite positions in metres in the GCRS (ECI) frame.
    """

    ts = load.timescale()
    sat = earth_satellite(record, ts)
    t_s = ts.ut1_jd(times_jd)
    sat_pos = sat.at(t_s).position.m

    return sat_pos


# ---------------------------------------------------------------------------
# Record retrieval
# ---------------------------------------------------------------------------

def get_tles_by_id(
    norad_ids: list[int],
    epoch_jd: float,
    extra_orbit_dir: Optional[str] = None,
    extra_orbit_max_age_days: Optional[float] = None,
    remote_max_age_days: Optional[float] = DEFAULT_REMOTE_MAX_AGE_DAYS,
    cache_reuse_max_age_days: Optional[float] = DEFAULT_CACHE_REUSE_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Orbit records for *norad_ids* nearest *epoch_jd*, one row per requested ID.

    Raises :class:`~tabsim.orbit.OrbitError` unless every requested ID resolved —
    tabsim does not silently drop a satellite that was asked for by number.
    """
    return get_orbits_by_id(
        norad_ids,
        epoch_jd,
        extra_orbit_dir=extra_orbit_dir,
        extra_orbit_max_age_days=extra_orbit_max_age_days,
        remote_max_age_days=remote_max_age_days,
        cache_reuse_max_age_days=cache_reuse_max_age_days,
    )


def get_tles_by_name(
    names: list[str],
    epoch_jd: float,
    extra_orbit_dir: Optional[str] = None,
    extra_orbit_max_age_days: Optional[float] = None,
    remote_max_age_days: Optional[float] = DEFAULT_REMOTE_MAX_AGE_DAYS,
    cache_reuse_max_age_days: Optional[float] = DEFAULT_CACHE_REUSE_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Orbit records for satellites *named* in the catalogue, nearest *epoch_jd*.

    Names are matched whole and case-insensitively against SatChecker's name
    index; a name matching nothing contributes no satellites and is reported. An
    empty frame is returned when no name matches at all, rather than an error —
    the caller is asking the catalogue a question, and "nothing" is an answer.
    """
    norad_ids = resolve_names(names)
    if not norad_ids:
        return pd.DataFrame()
    return get_orbits_by_id(
        norad_ids,
        epoch_jd,
        extra_orbit_dir=extra_orbit_dir,
        extra_orbit_max_age_days=extra_orbit_max_age_days,
        remote_max_age_days=remote_max_age_days,
        cache_reuse_max_age_days=cache_reuse_max_age_days,
    )


def get_visible_satellite_tles(
    times: ArrayLike,
    observer_lat: float,
    observer_lon: float,
    observer_elevation: float,
    target_ra: float,
    target_dec: float,
    max_angular_separation: float,
    min_elevation: float,
    names: ArrayLike = [],
    norad_ids: ArrayLike = [],
    extra_orbit_dir: Optional[str] = None,
    extra_orbit_max_age_days: Optional[float] = None,
    remote_max_age_days: Optional[float] = DEFAULT_REMOTE_MAX_AGE_DAYS,
    cache_reuse_max_age_days: Optional[float] = DEFAULT_CACHE_REUSE_MAX_AGE_DAYS,
) -> tuple:
    """Get the orbit records of satellites that satisfy the conditions given.

    Parameters
    ----------
    times : ArrayLike
        Times to consider in Astropy.time.Time format.
    observer_lat : float
        Observer latitude in degrees.
    observer_lon : float
        Observer longitude in degrees.
    observer_elevation : float
        Observer elevation in metres above sea level.
    target_ra : float
        Right ascension of the target direction.
    target_dec : float
        Declination of the target direction.
    max_angular_separation : float
        Maximum angular separation, in degrees, to accept a satellite pass.
    min_elevation : float
        Minimum elevation, in degrees, above the horizon to accept the satellite
        pass.
    names : ArrayLike
        Satellite names to consider. Matched whole and case-insensitively.
    norad_ids : ArrayLike
        NORAD IDs to consider. Every one of these must resolve to an acceptable
        record or :class:`~tabsim.orbit.OrbitError` is raised.
    extra_orbit_dir : str, optional
        Directory of user-supplied orbit files, searched before the managed cache
        and SatChecker.
    extra_orbit_max_age_days : float, optional
        Age ceiling for ``extra_orbit_dir`` records. ``None`` (the default) means
        unlimited, which is what makes exact replay of a previous run possible.
    remote_max_age_days : float, optional
        Age ceiling for records accepted from SatChecker or its managed cache.
    cache_reuse_max_age_days : float, optional
        A cached record this close to the observation avoids a network request.

    Returns
    -------
    tuple
        - NORAD IDs that pass the criteria, as an integer array.
        - Orbit records for those satellites, as a list of dicts. ``(array([]),
          [])`` when nothing passes.
    """

    epoch_jd = float(np.mean(times.jd))

    resolution = resolve_orbits(
        norad_ids,
        epoch_jd,
        extra_orbit_dir=extra_orbit_dir,
        extra_orbit_max_age_days=extra_orbit_max_age_days,
        remote_max_age_days=remote_max_age_days,
        cache_reuse_max_age_days=cache_reuse_max_age_days,
    )
    # Numbered satellites were asked for individually, so every one has to be
    # accounted for before any of them is filtered on visibility: a satellite
    # dropped here for want of a record would look exactly like one that simply
    # never passed the target.
    require_complete_coverage(resolution)

    ids = list(resolution.norad_ids())
    records = list(resolution.records())

    named_ids = [nid for nid in resolve_names(names) if nid not in set(ids)]
    if named_ids:
        # Named satellites are a catalogue query: one whose record cannot be
        # obtained is reported and skipped, not fatal.
        by_name = resolve_orbits(
            named_ids,
            epoch_jd,
            extra_orbit_dir=extra_orbit_dir,
            extra_orbit_max_age_days=extra_orbit_max_age_days,
            remote_max_age_days=remote_max_age_days,
            cache_reuse_max_age_days=cache_reuse_max_age_days,
        )
        if by_name.missing:
            print(
                f"  warning: {len(by_name.missing)} named satellite(s) have no "
                f"acceptable orbit record and are excluded: {by_name.missing}"
            )
        ids += by_name.norad_ids()
        records += by_name.records()

    if not records:
        return np.array([], dtype=int), []

    windows = check_satellite_visibilibities(
        ids,
        records,
        times,
        observer_lat,
        observer_lon,
        observer_elevation,
        target_ra,
        target_dec,
        max_angular_separation,
        min_elevation,
    )

    if len(windows) == 0:
        return np.array([], dtype=int), []

    visible = set(np.atleast_1d(windows["norad_id"].values).astype(int).tolist())
    keep = [i for i, nid in enumerate(ids) if int(nid) in visible]

    return np.array([ids[i] for i in keep], dtype=int), [records[i] for i in keep]


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

def make_window(
    times: ArrayLike, alt: ArrayLike, angular_sep: ArrayLike, idx: ArrayLike
) -> dict:
    """Make a dictionary containing the start and end times of a satellite pass including some stats.

    Parameters
    ----------
    times : ArrayLike[Time]
        Times of the satellite pass.
    alt : ArrayLike
        Altitude of the satellite during pass.
    angular_sep : ArrayLike
        Angular separation of the satellite during pass from target.
    idx: ArrayLike
        Index locations of the window.
    Returns
    -------
    dict
        Dictionary of stats.
    """

    window = {
        "start_time": times[idx][0].datetime.strftime(
            f"%Y-%m-%d-%H:%M:%S.%f {times.scale.upper()}"
        ),
        "end_time": times[idx][-1].datetime.strftime(
            f"%Y-%m-%d-%H:%M:%S.%f {times.scale.upper()}"
        ),
        "visible_period": (times[idx][-1] - times[idx][0]).sec,
        "min_ang_sep": np.min(angular_sep[idx]),
        "max_elevation": np.max(alt[idx]),
    }

    return window


def check_visibility(
    record,
    times: list[Time],
    observer_lat: float,
    observer_lon: float,
    observer_elevation: float,
    target_ra: float,
    target_dec: float,
    max_ang_sep: float,
    min_elev: float,
) -> list:
    """Calculate visibility windows for a satellite when observing a celestial target.

    This function determines time windows when a satellite will pass a celestial
    target based on the satellite's orbit record, observer location, target
    coordinates, and visibility constraints.

    Parameters
    ----------
    record : dict or (str, str)
        Orbit record — TLE or OMM — or a bare ``(line1, line2)`` TLE pair.
    times : list[Time]
        Array of observation times as Astropy Time objects.
    observer_lat : float
        Observer's latitude in degrees.
    observer_lon : float
        Observer's longitude in degrees.
    observer_elevation : float
        Observer's elevation above sea level in meters.
    target_ra : float
        Right Ascension of the target in degrees.
    target_dec : float
        Declination of the target in degrees.
    max_ang_sep : float
        Maximum allowed angular separation between satellite and target in degrees.
    min_elev : float
        Minimum required elevation of the satellite above horizon in degrees.

    Returns
    -------
    list
        List of visibility windows, where each window is a dictionary containing:
        - 'start_time': Start time of the visibility window
        - 'end_time': End time of the visibility window
        - 'max_elevation': Maximum elevation during the window
        - 'min_angular_separation': Minimum angular separation during the window

    Notes
    -----
    The function uses the WGS84 Earth model and converts the satellite's position
    to topocentric coordinates for elevation calculations. Visibility windows are
    determined based on both elevation constraints and angular separation from the
    target.
    """

    ts = load.timescale()
    sf_times = ts.ut1_jd(times.jd)

    # Set up observer location
    observer_location = wgs84.latlon(observer_lat, observer_lon, observer_elevation)

    # Create satellite object
    satellite = earth_satellite(record, ts)

    # Create celestial target position
    target = position_of_radec(
        ra_hours=target_ra / 15, dec_degrees=target_dec
    )  # Convert RA to hours

    satellite_position = satellite.at(sf_times)

    topocentric = satellite_position - observer_location.at(sf_times)
    alt, az, distance = topocentric.altaz()

    angular_sep = topocentric.separation_from(target).degrees

    vis_idx = np.where((alt.degrees > min_elev) & (angular_sep < max_ang_sep))[0]
    break_idx = np.where(np.diff(vis_idx) > 1)[0]
    if len(break_idx) > 0 or len(vis_idx) > 0:
        break_idx = np.concatenate([[0], break_idx, [len(times)]])
        windows = [
            make_window(
                times,
                alt.degrees,
                angular_sep,
                vis_idx[break_idx[i] : break_idx[i + 1]],
            )
            for i in range(len(break_idx) - 1)
        ]
    else:
        windows = []

    return windows


def check_satellite_visibilibities(
    norad_ids: list[int],
    records: list,
    times: list[Time],
    observer_lat: float,
    observer_lon: float,
    observer_elevation: float,
    target_ra: float,
    target_dec: float,
    max_ang_sep: float,
    min_elev: float,
) -> pd.DataFrame:
    """Calculate visibility windows for satellites when observing a celestial target.

    Parameters
    ----------
    norad_ids: list[int]
        NORAD IDs to calculate for.
    records : list
        Orbit records for those IDs, in the same order.
    times : list[Time]
        Array of observation times as Astropy Time objects.
    observer_lat : float
        Observer's latitude in degrees.
    observer_lon : float
        Observer's longitude in degrees.
    observer_elevation : float
        Observer's elevation above sea level in meters.
    target_ra : float
        Right Ascension of the target in degrees.
    target_dec : float
        Declination of the target in degrees.
    max_ang_sep : float
        Maximum allowed angular separation between satellite and target in degrees.
    min_elev : float
        Minimum required elevation of the satellite above horizon in degrees.

    Returns
    -------
    pandas.DataFrame
        One row per visibility window, with a ``norad_id`` column and the window
        statistics from :func:`make_window`.
    """

    print()
    print(
        f"Searching which satellites satisfy max_ang_sep: {max_ang_sep:.0f} and min_elev: {min_elev:.0f}"
    )
    all_windows = []
    for i in tqdm(range(len(norad_ids))):
        windows = check_visibility(
            records[i],
            times,
            observer_lat,
            observer_lon,
            observer_elevation,
            target_ra,
            target_dec,
            max_ang_sep,
            min_elev,
        )
        if len(windows) > 0:
            all_windows += [{"norad_id": norad_ids[i], **window} for window in windows]

    print(f"Found {len(all_windows)} matching satellites")
    return pd.DataFrame(all_windows)
