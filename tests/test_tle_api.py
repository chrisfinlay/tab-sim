"""
Comprehensive tests for TLE API functions after migration to gp_history endpoint.

Tests cover:
- fetch_tle_data() - Direct API calls with gp_history
- get_tles_by_id() - Fetch TLEs by NORAD ID
- get_tles_by_name() - Fetch TLEs by satellite name
- get_visible_satellite_tles() - Integration test for visibility calculations
- Data structure validation
- Caching functionality
- Error handling
"""

import pytest
import os
import yaml
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from astropy.time import Time, TimeDelta
from astropy import units as u

from tabsim.tle import (
    fetch_tle_data,
    get_tles_by_id,
    get_tles_by_name,
    get_visible_satellite_tles,
    get_space_track_client,
    get_satellite_positions,
    spacetrack_time_to_isot,
    get_closest_times,
    type_cast_tles,
    load_spacetrack_credentials,
)


@pytest.fixture(scope="module")
def spacetrack_credentials():
    """Load Space-Track credentials from YAML file or environment variable.

    This fixture supports both:
    1. GitHub Actions: SPACETRACK_LOGIN environment variable with YAML content
    2. Local development: spacetrack_login.yaml file (use tabsim-setup-spacetrack to create)

    Credentials are searched in order:
    1. SPACETRACK_LOGIN environment variable
    2. tabsim/data/rfi/tles/spacetrack_login.yaml (preferred location)
    3. ~/.credentials/spacetrack_login.yaml
    4. examples/test/spacetrack_login.yaml (for legacy compatibility)

    If no credentials are found, tests will be skipped.
    """
    # First, check for environment variable (GitHub Actions)
    spacetrack_login_env = os.environ.get('SPACETRACK_LOGIN')

    if spacetrack_login_env:
        # Parse YAML from environment variable
        creds = yaml.safe_load(spacetrack_login_env)
        return creds['username'], creds['password']

    # Try loading from file using the standard function
    username, password = load_spacetrack_credentials()

    if username and password:
        return username, password

    # Legacy fallback: check examples/test/ directory (for GitHub Actions compatibility)
    current_file = os.path.abspath(__file__)
    test_dir = os.path.dirname(current_file)
    tabsim_root = os.path.dirname(test_dir)
    legacy_path = os.path.join(tabsim_root, "examples", "test", "spacetrack_login.yaml")

    if os.path.exists(legacy_path):
        with open(legacy_path, 'r') as f:
            creds = yaml.safe_load(f)
        return creds['username'], creds['password']

    # No credentials found - skip tests
    pytest.skip(
        "Space-Track credentials not found. To set up credentials, run:\n"
        "  tabsim-setup-spacetrack\n\n"
        "Or set the SPACETRACK_LOGIN environment variable."
    )


@pytest.fixture
def temp_tle_dir():
    """Create a temporary directory for TLE caching tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_epoch():
    """Test epoch for fetching TLEs."""
    # Use a recent date for testing
    return Time("2024-01-15T12:00:00", format="isot", scale="ut1").jd


@pytest.fixture
def gps_norad_ids():
    """Sample GPS satellite NORAD IDs for testing."""
    return [
        32260,  # GPS BIIR-10 (PRN 05)
        40730,  # GPS BIIF-2 (PRN 01)
        41019,  # GPS BIIF-4 (PRN 31)
    ]


@pytest.fixture
def starlink_names():
    """Sample Starlink satellite names for testing."""
    return ["STARLINK-1007", "STARLINK-1008"]


class TestFetchTLEData:
    """Tests for fetch_tle_data() function using gp_history endpoint."""

    def test_fetch_tle_data_and_window(self, spacetrack_credentials, test_epoch, gps_norad_ids):
        """Test basic TLE fetching and window_days parameter.

        Combined test to reduce API calls for rate limit compliance.
        """
        username, password = spacetrack_credentials
        client = get_space_track_client(username, password)

        # Test 1: Basic fetching with multiple satellites
        df = fetch_tle_data(
            client,
            norad_ids=gps_norad_ids[:2],  # Test with 2 satellites
            epoch_jd=test_epoch,
            window_days=1.0,
            limit=100
        )

        # Verify we got data
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Verify required columns exist
        required_columns = ['NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2', 'EPOCH']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

        # Verify NORAD IDs match what we requested
        returned_ids = set(int(x) for x in df['NORAD_CAT_ID'].unique())
        requested_ids = set(gps_norad_ids[:2])
        assert returned_ids.issubset(requested_ids), "Returned IDs don't match requested"

        # Test 2: Window parameter (narrow vs wide) - reuses same client/credentials
        df_narrow = fetch_tle_data(
            client,
            norad_ids=[gps_norad_ids[0]],
            epoch_jd=test_epoch,
            window_days=0.5,
            limit=100
        )

        df_wide = fetch_tle_data(
            client,
            norad_ids=[gps_norad_ids[0]],
            epoch_jd=test_epoch,
            window_days=2.0,
            limit=100
        )

        # Wider window should return same or more results
        assert len(df_wide) >= len(df_narrow)

class TestGetTLEsByID:
    """Tests for get_tles_by_id() function."""

    def test_get_tles_by_id_basic_and_caching(
        self, spacetrack_credentials, test_epoch, gps_norad_ids, temp_tle_dir
    ):
        """Test basic TLE fetching and caching functionality.

        Combined test to reduce API calls for rate limit compliance.
        """
        username, password = spacetrack_credentials

        # First call - fetch from API and cache
        df1 = get_tles_by_id(
            username,
            password,
            norad_ids=gps_norad_ids[:1],
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=temp_tle_dir
        )

        # Verify basic data structure
        assert isinstance(df1, pd.DataFrame)
        assert len(df1) > 0

        # Verify required columns exist
        required_columns = [
            'NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2',
            'EPOCH', 'EPOCH_JD', 'time_diff', 'time_diff_abs'
        ]
        for col in required_columns:
            assert col in df1.columns, f"Missing column: {col}"

        # Verify TLE format
        for _, row in df1.iterrows():
            assert row['TLE_LINE1'].startswith('1 ')
            assert row['TLE_LINE2'].startswith('2 ')
            assert len(row['TLE_LINE1']) == 69
            assert len(row['TLE_LINE2']) == 69

        # Verify cache file was created
        cache_files = os.listdir(temp_tle_dir)
        assert len(cache_files) > 0, "No cache files created"

        # Second call - should load from cache (no additional API call)
        df2 = get_tles_by_id(
            username,
            password,
            norad_ids=gps_norad_ids[:1],
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=temp_tle_dir
        )

        # Results should be identical when loaded from cache
        pd.testing.assert_frame_equal(
            df1[['NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2']],
            df2[['NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2']]
        )

class TestGetTLEsByName:
    """Tests for get_tles_by_name() function."""

    def test_get_tles_by_name_basic_caching_multiple(
        self, spacetrack_credentials, test_epoch, temp_tle_dir
    ):
        """Test name-based TLE fetching, caching, and multiple satellites.

        Combined test to reduce API calls for rate limit compliance.
        Tests: basic fetching, caching functionality, and multiple satellite names.
        """
        username, password = spacetrack_credentials

        # Test 1: Single satellite with caching
        names_single = ["ISS"]
        df1 = get_tles_by_name(
            username,
            password,
            names=names_single,
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=temp_tle_dir
        )

        # Verify data structure
        assert isinstance(df1, pd.DataFrame)

        # Verify required columns if we got data
        if len(df1) > 0:
            assert 'NORAD_CAT_ID' in df1.columns
            assert 'TLE_LINE1' in df1.columns
            assert 'TLE_LINE2' in df1.columns
            assert 'OBJECT_NAME' in df1.columns

            # Check cache file exists for valid data
            epoch_str = Time(test_epoch, format="jd", scale="ut1").strftime("%Y-%m-%d")
            cache_file = os.path.join(temp_tle_dir, f"{epoch_str}-{names_single[0]}.json")
            assert os.path.exists(cache_file), "Cache file not created for valid data"

            # Second call - should load from cache (no additional API call)
            df2 = get_tles_by_name(
                username,
                password,
                names=names_single,
                epoch_jd=test_epoch,
                window_days=1.0,
                tle_dir=temp_tle_dir
            )

            # Results should be identical when loaded from cache
            pd.testing.assert_frame_equal(
                df1[['NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2']],
                df2[['NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2']]
            )
        else:
            # No data returned - verify second call is also empty
            df2 = get_tles_by_name(
                username,
                password,
                names=names_single,
                epoch_jd=test_epoch,
                window_days=1.0,
                tle_dir=temp_tle_dir
            )
            assert len(df2) == 0, "Both calls should return empty when no data available"

        # Test 2: Multiple satellites using SAME cache dir to reduce API calls
        names_multiple = ["ISS", "HUBBLE"]
        df_multi = get_tles_by_name(
            username,
            password,
            names=names_multiple,
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=temp_tle_dir  # Use same cache - ISS cached, only HUBBLE needs API
        )

        # Verify it's a DataFrame (may or may not have results depending on API)
        assert isinstance(df_multi, pd.DataFrame)

    def test_get_tles_by_name_nonexistent(self, spacetrack_credentials, test_epoch):
        """Test handling of non-existent satellite names."""
        username, password = spacetrack_credentials
        names = ["NONEXISTENT-SATELLITE-XYZ-999"]

        df = get_tles_by_name(
            username,
            password,
            names=names,
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=None
        )

        # Should return empty DataFrame
        assert isinstance(df, pd.DataFrame)
        # May be empty or have no results
        if len(df) > 0:
            # If not empty, shouldn't match our fake name
            assert not any(names[0] in str(name) for name in df.get('OBJECT_NAME', []))


class TestGetVisibleSatelliteTLEs:
    """Tests for get_visible_satellite_tles() integration function."""

    def test_get_visible_satellite_tles(
        self, spacetrack_credentials, gps_norad_ids
    ):
        """Test visibility calculation with both NORAD IDs and names.

        Combined test to reduce API calls for rate limit compliance.
        """
        username, password = spacetrack_credentials

        # MeerKAT telescope location
        observer_lat = -30.721
        observer_lon = 21.411
        observer_elevation = 1035.0
        target_ra = 83.63  # Orion region
        target_dec = -5.39

        # Observation times (1 hour)
        start_time = Time("2024-01-15T20:00:00", format="isot", scale="ut1")
        times = start_time + np.linspace(0, 1, 60) * u.hour

        # Test 1: By NORAD IDs
        norad_ids_result, tles_result = get_visible_satellite_tles(
            username,
            password,
            times,
            observer_lat,
            observer_lon,
            observer_elevation,
            target_ra,
            target_dec,
            max_angular_separation=30.0,
            min_elevation=10.0,
            norad_ids=gps_norad_ids,
            tle_dir=None
        )

        # Results may be empty if no satellites are visible
        assert isinstance(norad_ids_result, (list, np.ndarray))

        if len(norad_ids_result) > 0:
            assert tles_result is not None
            assert tles_result.shape[0] == len(norad_ids_result)
            assert tles_result.shape[1] == 2  # TLE_LINE1, TLE_LINE2

            # Verify TLE format
            for tle_pair in tles_result:
                assert tle_pair[0].startswith('1 ')
                assert tle_pair[1].startswith('2 ')

        # Test 2: By satellite name (reuses same credentials/client)
        norad_ids_by_name, tles_by_name = get_visible_satellite_tles(
            username,
            password,
            times,
            observer_lat,
            observer_lon,
            observer_elevation,
            target_ra,
            target_dec,
            max_angular_separation=40.0,
            min_elevation=5.0,
            names=["GPS BIIR-10"],
            tle_dir=None
        )

        # May or may not be visible during this time
        assert isinstance(norad_ids_by_name, (list, np.ndarray))


class TestHelperFunctions:
    """Tests for helper and utility functions."""

    def test_spacetrack_time_to_isot(self):
        """Test Space-Track time format conversion."""
        spacetrack_time = "2024-01-15 12:30:45"
        isot = spacetrack_time_to_isot(spacetrack_time)

        assert isot == "2024-01-15T12:30:45.000"

        # Verify it's valid for astropy
        t = Time(isot, format="isot")
        assert t.isot == "2024-01-15T12:30:45.000"

    def test_get_closest_times(self, test_epoch):
        """Test finding closest TLE to target epoch."""
        # Create sample TLE data
        data = {
            'NORAD_CAT_ID': [12345, 12345, 12345, 67890, 67890],
            'EPOCH_JD': [
                test_epoch - 1.0,
                test_epoch - 0.5,
                test_epoch + 0.3,
                test_epoch - 2.0,
                test_epoch + 1.0
            ],
            'TLE_LINE1': ['1 ' * 35] * 5,
            'TLE_LINE2': ['2 ' * 35] * 5,
        }
        df = pd.DataFrame(data)

        result = get_closest_times(df, test_epoch)

        # Should have one entry per NORAD ID
        assert len(result) == 2

        # For first satellite, closest should be EPOCH_JD = test_epoch + 0.3
        sat1 = result[result['NORAD_CAT_ID'] == 12345].iloc[0]
        assert abs(sat1['EPOCH_JD'] - (test_epoch + 0.3)) < 1e-6

        # For second satellite, closest should be EPOCH_JD = test_epoch + 1.0
        sat2 = result[result['NORAD_CAT_ID'] == 67890].iloc[0]
        assert abs(sat2['EPOCH_JD'] - (test_epoch + 1.0)) < 1e-6

    def test_type_cast_tles(self):
        """Test TLE type casting function."""
        # Create sample TLE data with string types
        data = {
            'NORAD_CAT_ID': ['12345', '67890'],
            'EPOCH_MICROSECONDS': ['123456', '789012'],
            'MEAN_MOTION': ['15.5', '14.2'],
            'ECCENTRICITY': ['0.001', '0.002'],
            'INCLINATION': ['51.6', '55.0'],
            'DECAYED': ['0', '1'],
            'TLE_LINE1': ['line1', 'line1'],
            'TLE_LINE2': ['line2', 'line2'],
        }
        df = pd.DataFrame(data)

        result = type_cast_tles(df)

        # Verify numeric columns are numeric types
        assert pd.api.types.is_numeric_dtype(result['NORAD_CAT_ID'])
        assert pd.api.types.is_numeric_dtype(result['MEAN_MOTION'])
        assert pd.api.types.is_numeric_dtype(result['ECCENTRICITY'])
        assert pd.api.types.is_bool_dtype(result['DECAYED'])

        # Verify values are correct
        assert result['NORAD_CAT_ID'].iloc[0] == 12345
        assert abs(result['MEAN_MOTION'].iloc[0] - 15.5) < 1e-6
        assert result['DECAYED'].iloc[0] == False
        assert result['DECAYED'].iloc[1] == True

    def test_get_satellite_positions(self, test_epoch):
        """Test satellite position calculation from TLEs."""
        # Use a real GPS TLE for testing (GPS BIIR-10, NORAD 32260)
        tles = [[
            '1 32260U 07047A   24015.50000000 -.00000024  00000+0  00000+0 0  9998',
            '2 32260  55.9642 157.2345 0109375  38.7890 321.8234  2.00565440120456'
        ]]

        # Calculate positions for a few time steps
        times_jd = [test_epoch, test_epoch + 0.1, test_epoch + 0.2]

        sat_pos = get_satellite_positions(tles, times_jd)

        # Verify shape: (n_sat, n_time, 3)
        assert sat_pos.shape == (1, 3, 3)

        # Verify positions are in reasonable range for GPS satellites
        # GPS orbits are ~26,600 km altitude, so ~20,000-28,000 km from Earth center
        for t_idx in range(3):
            pos = sat_pos[0, t_idx, :]
            distance = np.sqrt(np.sum(pos**2))
            assert 20e6 < distance < 30e6, f"Satellite position outside expected range: {distance/1e6:.1f} km"


class TestDataIntegrity:
    """Tests for data structure and integrity."""



# Run tests with: pytest tab-sim/tests/test_tle_api.py -v
# Run specific test: pytest tab-sim/tests/test_tle_api.py::TestFetchTLEData::test_fetch_tle_data_basic -v
# Run with output: pytest tab-sim/tests/test_tle_api.py -v -s
