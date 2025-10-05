from tabsim.read_ms import construct_config_from_ms


def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Extract simulation configuration parameters from an MS file and write to disk."
    )
    parser.add_argument(
        "-ms",
        "--ms_path",
        required=True,
        help="File path to the MS file.",
    )
    parser.add_argument(
        "-t",
        "--tel_name",
        default="telescope",
        help="Telescope name.",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        default="./",
        help="Path to directory where config and antenna coordinates should be saved.",
    )
    args = parser.parse_args()

    config_path, config = construct_config_from_ms(
        args.ms_path, args.tel_name, args.save_dir
    )

    print(f"Simulation configuration file save to : {config_path}")


if __name__ == "__main__":

    main()
