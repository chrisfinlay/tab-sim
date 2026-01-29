import argparse
import os
import sys
from pathlib import Path

from tabsim.config import load_config, run_sim_config

from typing import Union

from jax import config

config.update("jax_enable_x64", True)


def get_abs_path(rel_path: Union[str, None], work_dir: str) -> Union[str, None]:
    if rel_path:
        return os.path.abspath(os.path.join(work_dir, rel_path))
    else:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Simulate an observation defined by a YAML config file."
    )
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        help="File path to the observation config file.",
    )
    parser.add_argument(
        "-sp",
        "--save_path",
        help="Output directory path of simulation files.",
    )
    parser.add_argument(
        "-r",
        "--rfi_amp",
        default=1.0,
        type=float,
        help="Scale the RFI power. Default is 1.0",
    )
    parser.add_argument(
        "-a", "--n_ant", default=None, type=int, help="Number of antennas to include."
    )
    parser.add_argument(
        "-i", "--n_int", default=None, type=int, help="Number of integration samples."
    )
    parser.add_argument(
        "-n",
        "--SEFD",
        default=None,
        type=float,
        help="System Equivalent flux density in Jy. Same across frequency and antennas.",
    )
    parser.add_argument(
        "-dt", "--int_time", default=None, type=float, help="Time step in seconds."
    )
    parser.add_argument(
        "-nt", "--n_time", default=None, type=int, help="Number of time steps."
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing observation.",
    )
    parser.add_argument(
        "-st", "--spacetrack", help="Path to Space-Track login details."
    )
    parser.add_argument(
        "-ra", "--ra", type=float, help="Right Ascension of the observation."
    )
    args = parser.parse_args()
    rfi_amp = args.rfi_amp
    spacetrack_path = args.spacetrack
    config_path = Path(args.config_path)

    if not config_path.is_file():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)  # Exit with error code

    config_path = os.path.abspath(args.config_path)
    work_dir = os.path.split(config_path)[0]

    sim_config = load_config(config_path, config_type="sim")

    if args.save_path:
        save_path = Path(args.save_path)
        sim_config["output"]["path"] = save_path

    if args.ra is not None:
        sim_config["observation"]["ra"] = args.ra

    config_st_path = sim_config["rfi_sources"]["tle_satellite"]["spacetrack_path"]
    if spacetrack_path:
        sim_config["rfi_sources"]["tle_satellite"]["spacetrack_path"] = os.path.abspath(
            os.path.join(work_dir, spacetrack_path)
        )
    elif config_st_path:
        config_st_path = get_abs_path(config_st_path, work_dir)
        sim_config["rfi_sources"]["tle_satellite"]["spacetrack_path"] = config_st_path
        spacetrack_path = config_st_path

    sim_config["rfi_sources"]["tle_satellite"]["power_scale"] *= rfi_amp
    sim_config["rfi_sources"]["satellite"]["power_scale"] *= rfi_amp
    sim_config["rfi_sources"]["stationary"]["power_scale"] *= rfi_amp

    sim_config["output"]["overwrite"] = args.overwrite

    if args.n_ant is not None:
        sim_config["telescope"]["n_ant"] = args.n_ant

    if args.n_int is not None:
        sim_config["observation"]["n_int"] = args.n_int

    suffix = sim_config["output"]["suffix"]
    if suffix:
        suffix = f"{rfi_amp:.1e}RFI_" + suffix
    else:
        suffix = f"_{rfi_amp:.1e}RFI"
        sim_config["output"]["suffix"] = f"{rfi_amp:.1e}RFI"

    if args.SEFD is not None:
        sim_config["observation"]["SEFD"] = args.SEFD

    if args.int_time is not None:
        sim_config["observation"]["int_time"] = args.int_time

    if args.n_time is not None:
        sim_config["observation"]["n_time"] = args.n_time

    for ast in ["exp", "gauss", "point", "pow_spec"]:
        sim_config["ast_sources"][ast]["path"] = get_abs_path(
            sim_config["ast_sources"][ast]["path"], work_dir
        )

    sim_config["output"]["path"] = get_abs_path(sim_config["output"]["path"], work_dir)
    sim_config["rfi_sources"]["tle_satellite"]["norad_ids_path"] = get_abs_path(
        sim_config["rfi_sources"]["tle_satellite"]["norad_ids_path"], work_dir
    )
    sim_config["rfi_sources"]["tle_satellite"]["norad_spec_model"] = get_abs_path(
        sim_config["rfi_sources"]["tle_satellite"]["norad_spec_model"], work_dir
    )
    sim_config["telescope"]["enu_path"] = get_abs_path(
        sim_config["telescope"]["enu_path"], work_dir
    )
    sim_config["telescope"]["itrf_path"] = get_abs_path(
        sim_config["telescope"]["itrf_path"], work_dir
    )

    return run_sim_config(sim_config=sim_config, spacetrack_path=spacetrack_path)


if __name__ == "__main__":
    obs, obs_path = main()
