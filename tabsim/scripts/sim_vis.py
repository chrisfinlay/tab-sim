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
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing observation. Omitted, the config's own "
        "'overwrite' is left alone.",
    )
    parser.add_argument(
        "-eod",
        "--extra-orbit-dir",
        "--extra_orbit_dir",
        dest="extra_orbit_dir",
        help="Directory of local orbit files (TLE or OMM) to use, per NORAD ID, "
        "before the managed cache and SatChecker. This is ordinary source "
        "precedence: the run still chooses its own satellites, from its own names, "
        "NORAD IDs, visibility cuts and max_n_sat. To reproduce a previous run's "
        "selection as well as its trajectories, use --replay-orbit-dir.",
    )
    parser.add_argument(
        "--replay-orbit-dir",
        "--replay_orbit_dir",
        dest="replay_orbit_dir",
        help="A previous simulation's 'input_data' directory. Its saved NORAD IDs "
        "and orbit records are the selection: no catalogue search, no cache, no "
        "SatChecker request, no visibility reselection and no max_n_sat. Cannot be "
        "combined with --extra-orbit-dir.",
    )
    parser.add_argument(
        "--offline",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Forbid every SatChecker request. Cached catalogue searches are "
        "reused whatever their age; cached orbit records still have to satisfy "
        "remote_max_age_days. Omitted, the config's own 'offline' is left alone.",
    )
    parser.add_argument(
        "--allow-missing-checksum",
        "--allow_missing_checksum",
        dest="allow_missing_checksum",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Accept TLE lines that reached us without their checksum digit — "
        "roughly 2001-2018 in SatChecker's archive — and carry them as "
        "unverified. Applies to remote records, local files and replay alike. "
        "Omitted, the config's own 'allow_missing_checksum' is left alone.",
    )
    parser.add_argument(
        "-ra", "--ra", type=float, help="Right Ascension of the observation."
    )
    for flag, kind, help_text in (
        ('max-chunk-mb', float, 'Target compute chunk size in decimal MB; chosen before source/noise construction.'),
        ('component-workers', int, 'Number of simultaneous component streams (default 2).'),
        ('max-memory-gb', float, 'Host process RSS guard in GiB; checked between tasks.'),
        ('memory-fraction', float, 'Automatic host guard fraction of available RAM (default 0.7).'),
        ('max-device-memory-gb', float, 'Live JAX device allocation guard in GiB, excluding allocator reservation.'),
        ('timeout-s', float, 'Writer time limit checked between tasks.'),
        ('disk-reserve-gb', float, 'Free disk reserve in GiB (default 1).'),
    ):
        parser.add_argument('--' + flag, type=kind, default=None, help=help_text)
    parser.add_argument('--save-arrays', nargs='*', default=None,
                        help='Exact data variable names to retain; no names keeps metadata/coordinates only.')
    parser.add_argument('--save-rfi-amplitudes', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--flag-data', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--signal-stats', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    rfi_amp = args.rfi_amp
    config_path = Path(args.config_path)

    if not config_path.is_file():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)  # Exit with error code

    config_path = os.path.abspath(args.config_path)
    work_dir = os.path.split(config_path)[0]

    sim_config = load_config(config_path, config_type="sim")

    for key in ('component_workers', 'max_memory_gb', 'memory_fraction',
                'max_device_memory_gb', 'timeout_s', 'disk_reserve_gb'):
        value = getattr(args, key)
        if value is not None:
            sim_config['dask'][key] = value
    if args.max_chunk_mb is not None:
        sim_config['dask']['max_chunk_MB'] = args.max_chunk_mb
    for key in ('save_arrays', 'save_rfi_amplitudes', 'flag_data'):
        value = getattr(args, key)
        if value is not None:
            sim_config['output'][key] = value
    if args.signal_stats is not None:
        sim_config['diagnostics']['signal_stats'] = args.signal_stats

    if args.save_path:
        save_path = Path(args.save_path)
        sim_config["output"]["path"] = save_path

    if args.ra is not None:
        sim_config["observation"]["ra"] = args.ra

    # A path typed on the command line is relative to where it was typed; one
    # written in the config is relative to the config, like every other path here.
    satellites = sim_config["rfi_sources"]["tle_satellite"]
    for key, typed in (
        ("extra_orbit_dir", args.extra_orbit_dir),
        ("replay_orbit_dir", args.replay_orbit_dir),
    ):
        satellites[key] = (
            os.path.abspath(typed) if typed else get_abs_path(satellites.get(key), work_dir)
        )

    # A boolean flag defaults to None, so omitting it is not an instruction to
    # turn anything off: a deliberate `offline: true` in the config survives a
    # command line that says nothing about it.
    if args.offline is not None:
        satellites["offline"] = args.offline
    if args.allow_missing_checksum is not None:
        satellites["allow_missing_checksum"] = args.allow_missing_checksum
    if args.overwrite is not None:
        sim_config["output"]["overwrite"] = args.overwrite

    sim_config["rfi_sources"]["tle_satellite"]["power_scale"] *= rfi_amp
    sim_config["rfi_sources"]["satellite"]["power_scale"] *= rfi_amp
    sim_config["rfi_sources"]["stationary"]["power_scale"] *= rfi_amp

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
    # A frozen replay never opens the original ID file — its saved IDs are the
    # selection — so the setting is not an input of this run and is left exactly as
    # written. Processing it anyway made a leftover nothing will read able to stop
    # the run: os.path.join raises on a value that is not a path at all.
    if not satellites["replay_orbit_dir"]:
        satellites["norad_ids_path"] = get_abs_path(
            satellites["norad_ids_path"], work_dir
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

    return run_sim_config(sim_config=sim_config)


def cli() -> None:
    """The ``sim-vis`` console entry point.

    :func:`main` returns the observation and its output path for callers that
    drive a simulation from Python, tests included. The console script wraps its
    entry point in ``sys.exit(...)``, which reads any value that is not ``None``
    or an integer as failure: it prints the tuple and exits 1 after a successful
    run. So the script points here, and the result stays with :func:`main`.
    """
    main()


if __name__ == "__main__":
    obs, obs_path = main()
