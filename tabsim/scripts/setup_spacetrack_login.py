#!/usr/bin/env python3
"""
Setup script for SpaceTrack credentials.

This script prompts the user for their Space-Track.org credentials and saves them
to a YAML file in the TLE data directory for use by tabsim.

Run this script once to configure your SpaceTrack credentials:
    python scripts/setup_spacetrack_login.py

Or if installed as a package:
    tabsim-setup-spacetrack
"""

import os
import sys
import getpass
import yaml
from pathlib import Path


def get_data_directory():
    """Get the tabsim data directory path."""
    try:
        from importlib.resources import files
        data_dir = files("tabsim.data").joinpath("rfi/tles")
        return str(data_dir)
    except (ImportError, AttributeError):
        # Fallback for older Python or if package not installed
        # Try to find it relative to this script
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


def main():
    print("=" * 70)
    print("SpaceTrack Login Setup")
    print("=" * 70)
    print()
    print("This script will save your Space-Track.org credentials for use with tabsim.")
    print("If you don't have an account, register at: https://www.space-track.org/")
    print()

    # Get credentials from user
    username = input("Enter your Space-Track username: ").strip()
    if not username:
        print("Error: Username cannot be empty")
        sys.exit(1)

    password = getpass.getpass("Enter your Space-Track password: ")
    if not password:
        print("Error: Password cannot be empty")
        sys.exit(1)

    # Confirm password
    password_confirm = getpass.getpass("Confirm your Space-Track password: ")
    if password != password_confirm:
        print("Error: Passwords do not match")
        sys.exit(1)

    # Get data directory
    data_dir = get_data_directory()
    os.makedirs(data_dir, exist_ok=True)

    # Save credentials
    credentials_file = os.path.join(data_dir, "spacetrack_login.yaml")
    credentials = {
        "username": username,
        "password": password
    }

    try:
        with open(credentials_file, 'w') as f:
            yaml.dump(credentials, f, default_flow_style=False)

        # Set restrictive permissions (owner read/write only)
        os.chmod(credentials_file, 0o600)

        print()
        print("✓ Credentials saved successfully!")
        print(f"  Location: {credentials_file}")
        print(f"  Permissions: owner read/write only (600)")
        print()
        print("You can now use tabsim to fetch TLE data from Space-Track.org")

    except Exception as e:
        print(f"Error saving credentials: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
