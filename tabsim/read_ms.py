from daskms import xds_from_table, xds_from_ms
import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation

from typing import Union
import os
import yaml


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
    int_time = float(np.diff(np.sort(times)[:2])[0])

    start_time_isot = Time(np.min(times) / (24 * 3600), format="mjd").isot

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

    try:
        xds_src = xds_from_table(ms_path + "::SOURCE")[0]

        ra, dec = [float(x) for x in np.rad2deg(xds_src.DIRECTION.data[0]).compute()]
    except:
        print("Phase centre could not be read from MS file.")
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
