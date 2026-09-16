from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion, RectangleSkyRegion, TextSkyRegion

import xarray as xr
import numpy as np

from daskms import xds_from_ms, xds_from_table

import argparse

import os

import matplotlib.colors as mcolors

from tabsim.config import yaml_load
from tabsim.tle import earth_satellite, get_tles_by_id, get_satellite_positions
from tabsim.jax.coordinates import itrf_to_xyz, mjd_to_jd

from astropy.coordinates import EarthLocation
from astropy.time import Time

from skyfield.api import load, wgs84


def sat_radec(record, times_jd: np.ndarray, obs_xyz: np.ndarray) -> np.ndarray:

    ts = load.timescale()
    t = ts.ut1_jd(times_jd)

    satellite = earth_satellite(record, ts)
    location = EarthLocation(x=obs_xyz[0], y=obs_xyz[1], z=obs_xyz[2], unit="m")
    observer = wgs84.latlon(
        location.lat.degree, location.lon.degree, location.height.value
    )
    radec = (satellite - observer).at(t).radec()

    return np.array([radec[0]._degrees, radec[1].degrees]).T


def main():

    colors = list(mcolors.TABLEAU_COLORS.values())
    n_c = len(colors)

    parser = argparse.ArgumentParser(
        description="Extract satellite paths to ds9 region file."
    )
    parser.add_argument(
        "-m",
        "--ms_path",
        required=True,
        help="File path to the tabascal zarr simulation file.",
    )
    parser.add_argument(
        "-d",
        "--max_d",
        default=90.0,
        type=float,
        help="Maximum angular distance from phase centre (in degrees) of the satellite path to include.",
    )
    parser.add_argument(
        "-ni", "--norad_ids", help="NORAD IDs of satellites to include."
    )
    parser.add_argument(
        "-np",
        "--norad_path",
        help="Path to YAML config file containing list of norad_ids.",
    )
    parser.add_argument(
        "-eod",
        "--extra-orbit-dir",
        "--extra_orbit_dir",
        dest="extra_orbit_dir",
        help="Directory of local orbit files (TLE or OMM) to use, per NORAD ID, "
        "before the managed cache and SatChecker.",
    )
    parser.add_argument(
        "--offline",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Forbid every SatChecker request; use only local orbit files and the "
        "managed cache, within the same age ceiling.",
    )
    parser.add_argument(
        "--allow-missing-checksum",
        "--allow_missing_checksum",
        dest="allow_missing_checksum",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Accept TLE lines that reached us without their checksum digit, and "
        "carry them as unverified.",
    )
    parser.add_argument(
        "-s",
        "--sim",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use simulation coordinates. Only small difference from default skyfield coordinates.",
    )

    args = parser.parse_args()
    ms_path = args.ms_path
    max_d = args.max_d
    norad_ids = []
    if args.norad_ids:
        norad_ids += [int(x) for x in np.atleast_1d(args.norad_ids.split(","))]
    if args.norad_path:
        norad_ids += [
            int(x) for x in np.atleast_1d(str(yaml_load(args.norad_path)).split())
        ]

    if ms_path[-1] == "/":
        ms_path = ms_path[:-1]

    region_path = os.path.join(os.path.split(ms_path)[0], "satelitte_paths.reg")

    def xyz_to_radec(xyz):
        if xyz.ndim == 2:
            xyz = xyz[None, :, :]

        xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
        radec = np.zeros((*xyz.shape[:2], 2))
        radec[:, :, 0] = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])
        radec[:, :, 1] = np.arcsin(xyz[:, :, 2])

        return np.rad2deg(radec)

    xds = xds_from_ms(ms_path)[0]
    times_mjd = xds.TIME.data.compute()
    times_jd = mjd_to_jd(np.linspace(np.min(times_mjd), np.max(times_mjd), 10))
    epoch_jd = mjd_to_jd(np.mean(times_mjd))

    # Forwarded rather than re-defaulted: a region file drawn from records the
    # simulation would have refused is a region file for a different run.
    orbits_df = get_tles_by_id(
        norad_ids,
        epoch_jd,
        extra_orbit_dir=args.extra_orbit_dir,
        offline=bool(args.offline),
        allow_missing_checksum=bool(args.allow_missing_checksum),
    )
    if len(orbits_df) == 0:
        raise ValueError("No orbit records found.")

    orbits = [row for _, row in orbits_df.iterrows()]
    ids = np.atleast_1d(orbits_df["NORAD_CAT_ID"].values)
    n_tles = len(orbits)

    print(f"Found {n_tles} matching orbit records.")

    if n_tles > 0:

        xds_src = xds_from_table(ms_path + "::SOURCE")[0]
        ra, dec = np.rad2deg(xds_src.DIRECTION.data[0].compute())

        xds_ants = xds_from_table(ms_path + "::ANTENNA")[0]
        ants_itrf = np.mean(xds_ants.POSITION.data.compute(), axis=0, keepdims=True)

        if args.sim:
            # ants_xyz = itrf_to_xyz(ants_itrf, gmsa_from_jd(times_jd))[:, 0]
            ants_xyz = itrf_to_xyz(
                ants_itrf,
                Time(times_jd, format="jd").sidereal_time("mean", "greenwich").hour
                * 15,
            )[:, 0]
            rfi_xyz = get_satellite_positions(orbits, times_jd)
            xyz = rfi_xyz - ants_xyz[None, :, :]
            radec = xyz_to_radec(xyz)
        else:
            radec = np.array(
                [sat_radec(orbit, times_jd, ants_itrf[0]) for orbit in orbits]
            )

        c0 = SkyCoord(ra, dec, unit="deg", frame="fk5")
        c = SkyCoord(radec[:, :, 0], radec[:, :, 1], unit="deg", frame="fk5")

        min_sep = c0.separation(c).min(axis=1).deg
        print(
            f"Minimum angular separation from target : {[round(x, 1) for x in min_sep]} deg."
        )
        print(
            f"Only including satellites within {max_d:.1f} degrees of pointing direction."
        )
        idx = np.where(min_sep < max_d)[0]

        min_idx = np.argmin(c0.separation(c)[:, :-1], axis=1)

        ang_v = np.diff(radec, axis=1)
        ang_theta = np.arctan2(ang_v[:, :, 1], ang_v[:, :, 0])

        with open(region_path, "w") as fp:
            for c_i, i in enumerate(idx):
                # fp.write(RectangleSkyRegion(c0, 0.05*u.deg, 0.3*u.deg, ang_theta[i,min_idx[i]]*u.rad, visual={"color": colors[c_i%n_c]}).serialize(format="ctrf"))
                # fp.write(TextSkyRegion(c[i,-1], str(ids[i]), visual={"fontsize": 16, "color": colors[c_i%n_c]}).serialize(format="ds9"))
                for j in range(len(times_jd)):
                    fp.write(
                        CircleSkyRegion(
                            c[i, j],
                            radius=0.1 * u.deg,
                            visual={"color": colors[c_i % n_c]},
                        ).serialize(format="ds9")
                    )


if __name__ == "__main__":
    main()
