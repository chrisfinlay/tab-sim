from daskms import xds_from_table, xds_from_ms
import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation

from typing import Union
import os
import yaml

from tabsim.time import ms_time_tai_minus_utc, utc_mjd_from_ms_time


def write_antenna_itrf_from_ms(
    ms_path: str, tel_name: str = "telescope", save_dir: str = "./"
) -> tuple[str, int, float]:

    xds_ant = xds_from_table(ms_path + "::ANTENNA")[0]

    itrf = xds_ant.POSITION.data.compute()
    n_ant = len(itrf)

    dish_d = float(xds_ant.DISH_DIAMETER.data[0].compute())

    itrf_path = os.path.abspath(os.path.join(save_dir, f"{tel_name}.itrf.txt"))

    np.savetxt(itrf_path, itrf, fmt="%.6f")

    return itrf_path, n_ant, dish_d


def read_tel_location_from_ms(ms_path: str) -> tuple[float, float, float]:

    xds_ant = xds_from_table(ms_path + "::ANTENNA")[0]

    tel_itrf = xds_ant.POSITION.data.mean(axis=0).compute()

    tel_loc = EarthLocation(x=tel_itrf[0], y=tel_itrf[1], z=tel_itrf[2], unit="m")

    lat, lon, elevation = (
        float(tel_loc.lat.deg),
        float(tel_loc.lon.deg),
        float(tel_loc.height.value),
    )

    return lat, lon, elevation


def read_time_data_from_ms(ms_path: str) -> tuple[str, float, int]:

    xds_ms = xds_from_ms(ms_path)[0]

    times = np.unique(xds_ms.TIME.data.compute())

    n_time = len(times)
    # On SI seconds: TIME steps back by any leap second between the samples.
    first = np.sort(times)[:2]
    int_time = float(np.diff(first)[0] + np.diff(ms_time_tai_minus_utc(first))[0])

    start_time_isot = Time(utc_mjd_from_ms_time(np.min(times)), format="mjd").isot

    return start_time_isot, int_time, n_time


def read_freq_data_from_ms(
    ms_path: str,
) -> tuple[Union[float, None], Union[float, None], Union[int, None]]:

    try:
        xds_spec = xds_from_table(ms_path + "::SPECTRAL_WINDOW")[0]

        freqs = xds_spec.CHAN_FREQ.data[0].compute()
        start_freq = float(freqs[0])
        n_freq = len(freqs)

        chan_width = float(xds_spec.CHAN_WIDTH.data[0, 0].compute())
    except:
        print("Frequency information could not be red from MS file.")
        start_freq, chan_width, n_freq = None, None, None

    return start_freq, chan_width, n_freq


def read_phase_centre_from_ms(
    ms_path: str,
) -> tuple[Union[float, None], Union[float, None]]:
    """Read the first FIELD phase centre in the config's ICRS frame."""
    from casacore.measures import measures
    from casacore.quanta import quantity
    from casacore.tables import table

    try:
        with table(str(ms_path) + "::FIELD", ack=False) as field:
            info = field.getcolkeyword("PHASE_DIR", "MEASINFO")
            if "VarRefCol" in info or field.getcell("NUM_POLY", 0) != 0:
                raise ValueError("Only fixed-frame, constant FIELD directions are supported")
            if (
                "EPHEMERIS_ID" in field.colnames()
                and field.getcell("EPHEMERIS_ID", 0) >= 0
            ):
                raise ValueError("FIELD direction is an offset from a moving ephemeris")
            direction = field.getcell("PHASE_DIR", 0)[0]
        ref = info["Ref"].upper()
        # Only celestial frames convertible without an epoch or observer. Check
        # before direction(): casacore silently falls back to J2000 for typos.
        if ref not in {
            "ICRS", "J2000", "B1950", "B1950_VLA", "GALACTIC", "SUPERGAL", "ECLIPTIC"
        }:
            raise ValueError(f"Unsupported FIELD direction reference: {ref}")
        if ref != "ICRS":
            m = measures()
            converted = m.measure(
                m.direction(ref, *(quantity(float(x), "rad") for x in direction)),
                "ICRS",
            )
            direction = [quantity(converted[k]).get_value("rad") for k in ("m0", "m1")]
        ra, dec = map(float, np.rad2deg(direction))
        ra %= 360.0
    except Exception as error:
        print(f"Phase centre could not be read from MS file: {error}")
        ra, dec = None, None

    return ra, dec


def construct_config_from_ms(
    ms_path: str, tel_name: str = "telescope", save_dir: str = "./"
) -> tuple[str, dict]:

    itrf_path, n_ant, dish_d = write_antenna_itrf_from_ms(ms_path, tel_name, save_dir)
    lat, lon, elevation = read_tel_location_from_ms(ms_path)
    ra, dec = read_phase_centre_from_ms(ms_path)
    start_time_isot, int_time, n_time = read_time_data_from_ms(ms_path)
    start_freq, chan_width, n_freq = read_freq_data_from_ms(ms_path)

    config = {
        "telescope": {
            "name": tel_name,
            "latitude": lat,
            "longitude": lon,
            "elevation": elevation,
            "itrf_path": itrf_path,
            "n_ant": n_ant,
            # "dish_d": dish_d,
        },
        "observation": {
            "start_time_isot": start_time_isot,
            "int_time": int_time,
            "n_time": n_time,
            "start_freq": start_freq,
            "chan_width": chan_width,
            "n_freq": n_freq,
        },
    }

    if ra is not None:
        config["observation"].update(
            {
                "target_name": f"pointing_{ra:.3f}_{dec:.3f}",
                "ra": ra,
                "dec": dec,
            }
        )

    config_path = os.path.abspath(os.path.join(save_dir, f"{tel_name}_sim_config.yaml"))

    yaml.dump(config, open(config_path, "w"))

    return config_path, config
