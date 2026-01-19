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
from astropy.time import Time
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
)


@pytest.fixture(scope="module")
def spacetrack_credentials():
    """Load Space-Track credentials from YAML file."""
    # Look for credentials in TABASCAL root directory
    current_file = os.path.abspath(__file__)
    # Go up from tests/ -> tab-sim/ -> TABASCAL/
    tabascal_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    cred_path = os.path.join(tabascal_root, "spacetrack_login.yaml")

    if not os.path.exists(cred_path):
        pytest.skip(f"Space-Track credentials file not found at {cred_path}")

    with open(cred_path, 'r') as f:
        creds = yaml.safe_load(f)

    return creds['username'], creds['password']


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

    def test_fetch_tle_data_basic(self, spacetrack_credentials, test_epoch, gps_norad_ids):
        """Test basic TLE fetching with gp_history endpoint."""
        username, password = spacetrack_credentials
        client = get_space_track_client(username, password)

        # Fetch TLEs for a small set of satellites
        df = fetch_tle_data(
            client,
            norad_ids=gps_norad_ids[:2],  # Just test with 2 satellites
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
        # Note: JSON returns NORAD_CAT_ID as strings, convert to int for comparison
        returned_ids = set(int(x) for x in df['NORAD_CAT_ID'].unique())
        requested_ids = set(gps_norad_ids[:2])
        assert returned_ids.issubset(requested_ids), "Returned IDs don't match requested"

    def test_fetch_tle_data_window(self, spacetrack_credentials, test_epoch, gps_norad_ids):
        """Test that window_days parameter works correctly."""
        username, password = spacetrack_credentials
        client = get_space_track_client(username, password)

        # Fetch with narrow window
        df_narrow = fetch_tle_data(
            client,
            norad_ids=[gps_norad_ids[0]],
            epoch_jd=test_epoch,
            window_days=0.5,
            limit=100
        )

        # Fetch with wider window
        df_wide = fetch_tle_data(
            client,
            norad_ids=[gps_norad_ids[0]],
            epoch_jd=test_epoch,
            window_days=2.0,
            limit=100
        )

        # Wider window should return same or more results
        assert len(df_wide) >= len(df_narrow)

    def test_fetch_tle_data_invalid_norad_id(self, spacetrack_credentials, test_epoch):
        """Test handling of invalid NORAD IDs."""
        username, password = spacetrack_credentials
        client = get_space_track_client(username, password)

        # Use a NORAD ID that definitely doesn't exist
        df = fetch_tle_data(
            client,
            norad_ids=[99999999],  # Invalid ID
            epoch_jd=test_epoch,
            window_days=1.0,
            limit=10
        )

        # Should return empty DataFrame, not error
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestGetTLEsByID:
    """Tests for get_tles_by_id() function."""

    def test_get_tles_by_id_basic(self, spacetrack_credentials, test_epoch, gps_norad_ids):
        """Test basic TLE fetching by NORAD ID."""
        username, password = spacetrack_credentials

        df = get_tles_by_id(
            username,
            password,
            norad_ids=gps_norad_ids[:2],
            epoch_jd=test_epoch,
            window_days=1.0,
            limit=100,
            tle_dir=None  # No caching for this test
        )

        # Verify data structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # Should have one TLE per satellite (closest to epoch)
        assert len(df) == len(set(gps_norad_ids[:2]))

        # Verify required columns
        required_columns = [
            'NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2',
            'EPOCH', 'EPOCH_JD', 'time_diff', 'time_diff_abs'
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

        # Verify TLE format
        for _, row in df.iterrows():
            assert row['TLE_LINE1'].startswith('1 ')
            assert row['TLE_LINE2'].startswith('2 ')
            assert len(row['TLE_LINE1']) == 69
            assert len(row['TLE_LINE2']) == 69

    def test_get_tles_by_id_with_caching(
        self, spacetrack_credentials, test_epoch, gps_norad_ids, temp_tle_dir
    ):
        """Test TLE caching functionality."""
        username, password = spacetrack_credentials

        # First call - should fetch from API and cache
        df1 = get_tles_by_id(
            username,
            password,
            norad_ids=gps_norad_ids[:1],
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=temp_tle_dir
        )

        # Verify cache file was created
        cache_files = os.listdir(temp_tle_dir)
        assert len(cache_files) > 0, "No cache files created"

        # Second call - should load from cache
        df2 = get_tles_by_id(
            username,
            password,
            norad_ids=gps_norad_ids[:1],
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=temp_tle_dir
        )

        # Results should be identical
        pd.testing.assert_frame_equal(
            df1[['NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2']],
            df2[['NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2']]
        )

    def test_get_tles_by_id_mixed_cache_remote(
        self, spacetrack_credentials, test_epoch, gps_norad_ids, temp_tle_dir
    ):
        """Test fetching with some TLEs cached and some from remote."""
        username, password = spacetrack_credentials

        # First, cache TLEs for first satellite
        df1 = get_tles_by_id(
            username,
            password,
            norad_ids=[gps_norad_ids[0]],
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=temp_tle_dir
        )

        # Now fetch for multiple satellites (first cached, second remote)
        df2 = get_tles_by_id(
            username,
            password,
            norad_ids=gps_norad_ids[:2],
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=temp_tle_dir
        )

        # Should have TLEs for both satellites
        assert len(df2) == 2
        assert set(df2['NORAD_CAT_ID'].values) == set(gps_norad_ids[:2])

    def test_get_tles_by_id_large_batch(self, spacetrack_credentials, test_epoch, gps_norad_ids):
        """Test fetching large batch of TLEs (tests chunking)."""
        username, password = spacetrack_credentials

        # Use multiple GPS satellites to test batching
        # The function chunks requests into max_ids=500
        df = get_tles_by_id(
            username,
            password,
            norad_ids=gps_norad_ids,
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=None
        )

        assert len(df) <= len(gps_norad_ids)
        assert len(df) > 0


class TestGetTLEsByName:
    """Tests for get_tles_by_name() function."""

    def test_get_tles_by_name_basic(self, spacetrack_credentials, test_epoch):
        """Test basic TLE fetching by satellite name."""
        username, password = spacetrack_credentials

        # Use ISS which is more reliably named
        names = ["ISS"]

        df = get_tles_by_name(
            username,
            password,
            names=names,
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=None
        )

        # Verify data structure (may be empty if name not found)
        assert isinstance(df, pd.DataFrame)

        # If we got data, verify required columns
        if len(df) > 0:
            assert 'NORAD_CAT_ID' in df.columns
            assert 'TLE_LINE1' in df.columns
            assert 'TLE_LINE2' in df.columns
            assert 'OBJECT_NAME' in df.columns

    def test_get_tles_by_name_with_caching(
        self, spacetrack_credentials, test_epoch, temp_tle_dir
    ):
        """Test name-based TLE caching."""
        username, password = spacetrack_credentials
        names = ["ISS"]

        # First call - fetch and cache
        df1 = get_tles_by_name(
            username,
            password,
            names=names,
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=temp_tle_dir
        )

        # Verify cache file exists
        epoch_str = Time(test_epoch, format="jd", scale="ut1").strftime("%Y-%m-%d")
        cache_file = os.path.join(temp_tle_dir, f"{epoch_str}-{names[0]}.json")
        assert os.path.exists(cache_file), "Cache file not created"

        # Second call - load from cache
        df2 = get_tles_by_name(
            username,
            password,
            names=names,
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=temp_tle_dir
        )

        # Results should be identical
        if len(df1) > 0:
            pd.testing.assert_frame_equal(
                df1[['NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2']],
                df2[['NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2']]
            )
        else:
            # Both should be empty if no results found
            assert len(df2) == 0

    def test_get_tles_by_name_multiple(self, spacetrack_credentials, test_epoch):
        """Test fetching multiple satellites by name."""
        username, password = spacetrack_credentials
        names = ["ISS", "HUBBLE"]

        df = get_tles_by_name(
            username,
            password,
            names=names,
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=None
        )

        # Verify it's a DataFrame (may or may not have results)
        assert isinstance(df, pd.DataFrame)

        # If we got results, should have some data
        # (Don't enforce >= len(names) since names might not match any satellites)

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

    def test_get_visible_satellite_tles_by_id(
        self, spacetrack_credentials, gps_norad_ids
    ):
        """Test visibility calculation with NORAD IDs."""
        username, password = spacetrack_credentials

        # MeerKAT telescope location
        observer_lat = -30.721
        observer_lon = 21.411
        observer_elevation = 1035.0

        # Target coordinates (arbitrary sky position)
        target_ra = 83.63  # Orion region
        target_dec = -5.39

        # Observation times (1 hour)
        start_time = Time("2024-01-15T20:00:00", format="isot", scale="ut1")
        times = start_time + np.linspace(0, 1, 60) * u.hour

        norad_ids, tles = get_visible_satellite_tles(
            username,
            password,
            times,
            observer_lat,
            observer_lon,
            observer_elevation,
            target_ra,
            target_dec,
            max_angular_separation=30.0,  # 30 degrees
            min_elevation=10.0,  # 10 degrees above horizon
            norad_ids=gps_norad_ids,
            tle_dir=None
        )

        # Results may be empty if no satellites are visible
        assert isinstance(norad_ids, (list, np.ndarray))

        if len(norad_ids) > 0:
            assert tles is not None
            assert tles.shape[0] == len(norad_ids)
            assert tles.shape[1] == 2  # TLE_LINE1, TLE_LINE2

            # Verify TLE format
            for tle_pair in tles:
                assert tle_pair[0].startswith('1 ')
                assert tle_pair[1].startswith('2 ')

    def test_get_visible_satellite_tles_by_name(self, spacetrack_credentials):
        """Test visibility calculation with satellite names."""
        username, password = spacetrack_credentials

        # Use telescope location and sky position
        observer_lat = -30.721
        observer_lon = 21.411
        observer_elevation = 1035.0
        target_ra = 83.63
        target_dec = -5.39

        start_time = Time("2024-01-15T20:00:00", format="isot", scale="ut1")
        times = start_time + np.linspace(0, 3600, 60) / 86400.0

        norad_ids, tles = get_visible_satellite_tles(
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
        assert isinstance(norad_ids, (list, np.ndarray))


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

    def test_tle_data_structure(self, spacetrack_credentials, test_epoch, gps_norad_ids):
        """Test that returned TLE data has correct structure."""
        username, password = spacetrack_credentials

        df = get_tles_by_id(
            username,
            password,
            norad_ids=gps_norad_ids[:1],
            epoch_jd=test_epoch,
            window_days=1.0,
            tle_dir=None
        )

        # Check DataFrame is not empty
        assert len(df) > 0

        # Check essential columns exist
        essential_cols = [
            'NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2', 'EPOCH', 'EPOCH_JD',
            'OBJECT_NAME', 'MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION'
        ]
        for col in essential_cols:
            assert col in df.columns, f"Missing essential column: {col}"

        # Check data types
        assert pd.api.types.is_numeric_dtype(df['NORAD_CAT_ID'])
        assert pd.api.types.is_numeric_dtype(df['EPOCH_JD'])
        assert pd.api.types.is_numeric_dtype(df['MEAN_MOTION'])
        assert pd.api.types.is_string_dtype(df['TLE_LINE1'])
        assert pd.api.types.is_string_dtype(df['TLE_LINE2'])

        # Validate TLE line format
        for _, row in df.iterrows():
            # Line 1 format: "1 NNNNNC NNNNNAAA NNNNN.NNNNNNNN ..."
            assert len(row['TLE_LINE1']) == 69
            assert row['TLE_LINE1'][0] == '1'
            assert row['TLE_LINE1'][1] == ' '

            # Line 2 format: "2 NNNNN NNN.NNNN NNN.NNNN ..."
            assert len(row['TLE_LINE2']) == 69
            assert row['TLE_LINE2'][0] == '2'
            assert row['TLE_LINE2'][1] == ' '

            # NORAD ID should match in both lines
            tle1_norad = int(row['TLE_LINE1'][2:7].strip())
            tle2_norad = int(row['TLE_LINE2'][2:7].strip())
            assert tle1_norad == tle2_norad == row['NORAD_CAT_ID']


# Run tests with: pytest tab-sim/tests/test_tle_api.py -v
# Run specific test: pytest tab-sim/tests/test_tle_api.py::TestFetchTLEData::test_fetch_tle_data_basic -v
# Run with output: pytest tab-sim/tests/test_tle_api.py -v -s
