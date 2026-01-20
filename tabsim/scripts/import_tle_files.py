#!/usr/bin/env python3
"""
Import TLE text files and convert them to JSON format for tabsim.

This script reads TLE files in standard 3-line format and converts them to JSON
files compatible with tabsim's TLE caching system.

Standard TLE format (3-line):
    SATELLITE NAME
    1 NNNNNC NNNNNAAA NNNNN.NNNNNNNN ...
    2 NNNNN NNN.NNNN NNN.NNNN ...

Usage:
    python scripts/import_tle_files.py <tle_file1> [tle_file2 ...]
    python scripts/import_tle_files.py --directory <dir>

Or if installed as a package:
    tabsim-import-tles <tle_file1> [tle_file2 ...]
    tabsim-import-tles --directory <dir>

The script will process the TLE files and save them to the tabsim data directory
with filenames based on the epoch date and satellite names.
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from astropy.time import Time


def get_data_directory():
    """Get the tabsim TLE data directory path."""
    try:
        from importlib.resources import files
        data_dir = files("tabsim.data").joinpath("rfi/tles")
        return str(data_dir)
    except (ImportError, AttributeError):
        # Fallback for older Python or if package not installed
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir.parent / "tabsim" / "data" / "rfi" / "tles",
            Path.cwd() / "tabsim" / "data" / "rfi" / "tles",
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)

        # If not found, create in current directory
        data_dir = Path.cwd() / "tabsim" / "data" / "rfi" / "tles"
        data_dir.mkdir(parents=True, exist_ok=True)
        return str(data_dir)


def parse_tle_epoch(tle_line1):
    """
    Extract epoch from TLE line 1.

    TLE line 1 positions 18-32 contain the epoch:
    - Positions 18-19: Year (last 2 digits)
    - Positions 20-32: Day of year with fractional portion

    Returns ISO format timestamp string.
    """
    try:
        year_str = tle_line1[18:20]
        day_of_year = float(tle_line1[20:32])

        # Convert 2-digit year to 4-digit year
        year = int(year_str)
        if year < 57:  # Standard TLE convention
            year += 2000
        else:
            year += 1900

        # Create datetime from year and day of year
        epoch = datetime(year, 1, 1, tzinfo=timezone.utc)
        epoch = epoch.replace(tzinfo=None)  # Remove timezone for calculation

        # Add fractional days
        from datetime import timedelta
        epoch = epoch + timedelta(days=day_of_year - 1)

        # Convert to ISO format
        return epoch.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]  # Milliseconds

    except Exception as e:
        print(f"Warning: Could not parse epoch from TLE line: {e}")
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def parse_tle_file(file_path):
    """
    Parse a TLE file and return a list of TLE records.

    Supports both 2-line (no name) and 3-line (with name) formats.

    Returns list of dicts with keys: OBJECT_NAME, TLE_LINE1, TLE_LINE2, NORAD_CAT_ID, EPOCH
    """
    tles = []

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    i = 0
    while i < len(lines):
        # Check if current line is a TLE line 1
        if i < len(lines) and lines[i].startswith('1 '):
            tle_line1 = lines[i]

            # Next line should be TLE line 2
            if i + 1 < len(lines) and lines[i + 1].startswith('2 '):
                tle_line2 = lines[i + 1]

                # Previous line (if exists and doesn't start with 1 or 2) is satellite name
                object_name = "UNKNOWN"
                if i > 0 and not lines[i - 1].startswith(('1 ', '2 ')):
                    object_name = lines[i - 1]

                # Extract NORAD ID from line 1 (positions 2-7)
                try:
                    norad_id = int(tle_line1[2:7].strip())
                except ValueError:
                    print(f"Warning: Could not extract NORAD ID from: {tle_line1}")
                    i += 2
                    continue

                # Extract epoch
                epoch = parse_tle_epoch(tle_line1)

                tles.append({
                    'OBJECT_NAME': object_name,
                    'NORAD_CAT_ID': norad_id,
                    'TLE_LINE1': tle_line1,
                    'TLE_LINE2': tle_line2,
                    'EPOCH': epoch,
                    'Fetch_Timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                })

                i += 2
            else:
                i += 1
        else:
            i += 1

    return tles


def group_tles_by_satellite(tles):
    """
    Group TLEs by satellite name.

    Returns dict mapping satellite names to lists of TLE records.
    """
    grouped = {}

    for tle in tles:
        name = tle['OBJECT_NAME']
        if name not in grouped:
            grouped[name] = []
        grouped[name].append(tle)

    return grouped


def save_tles_to_json(tles, output_dir, satellite_name=None):
    """
    Save TLE records to JSON file in tabsim cache format.

    If satellite_name is provided, uses that name for the file.
    Otherwise, uses the name from the TLE records.
    """
    if not tles:
        return None

    # Create DataFrame
    df = pd.DataFrame(tles)

    # Get satellite name for filename
    if satellite_name is None:
        satellite_name = tles[0]['OBJECT_NAME']

    # Get epoch date for filename (use first TLE's epoch)
    try:
        epoch_time = Time(tles[0]['EPOCH'], format='isot', scale='utc')
        epoch_str = epoch_time.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Warning: Could not parse epoch, using current date: {e}")
        epoch_str = datetime.now().strftime("%Y-%m-%d")

    # Clean satellite name for filename (remove special characters)
    safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in satellite_name)

    # Create output filename
    output_file = os.path.join(output_dir, f"{epoch_str}-{safe_name}.json")

    # Save to JSON
    df.to_json(output_file, orient='records', indent=2)

    return output_file


def process_tle_file(file_path, output_dir):
    """Process a single TLE file."""
    print(f"Processing: {file_path}")

    try:
        tles = parse_tle_file(file_path)

        if not tles:
            print(f"  Warning: No TLEs found in {file_path}")
            return

        print(f"  Found {len(tles)} TLE(s)")

        # Group by satellite
        grouped = group_tles_by_satellite(tles)

        # Save each satellite's TLEs
        for sat_name, sat_tles in grouped.items():
            output_file = save_tles_to_json(sat_tles, output_dir, sat_name)
            if output_file:
                print(f"  Saved {len(sat_tles)} TLE(s) for '{sat_name}' to: {output_file}")

    except Exception as e:
        print(f"  Error processing {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Import TLE text files and convert to tabsim JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'files',
        nargs='*',
        help='TLE file(s) to import'
    )
    parser.add_argument(
        '--directory', '-d',
        help='Directory containing TLE files to import'
    )

    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory (default: tabsim data directory)'
    )

    args = parser.parse_args()

    # Validate that at least one input method is provided
    if not args.files and not args.directory:
        parser.error("Either specify TLE files or use --directory to specify a directory")

    # Validate that both aren't specified
    if args.files and args.directory:
        parser.error("Cannot specify both files and --directory at the same time")

    # Get output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_data_directory()

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("TLE Import Tool")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()

    # Get list of files to process
    files_to_process = []

    if args.directory:
        # Process all files in directory
        dir_path = Path(args.directory)
        if not dir_path.exists():
            print(f"Error: Directory not found: {args.directory}")
            sys.exit(1)

        # Find all text files
        for ext in ['.txt', '.tle', '.TLE']:
            files_to_process.extend(dir_path.glob(f'*{ext}'))

        # Also check files without extension
        for file_path in dir_path.iterdir():
            if file_path.is_file() and not file_path.suffix:
                files_to_process.append(file_path)

    else:
        # Process specified files
        files_to_process = [Path(f) for f in args.files]

    if not files_to_process:
        print("No TLE files found to process")
        sys.exit(1)

    # Process each file
    for file_path in files_to_process:
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue

        process_tle_file(file_path, output_dir)

    print()
    print("=" * 70)
    print("Import complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
