"""MS column descriptors must preserve visibility precision on disk."""

import numpy as np
import pytest

pytest.importorskip("casacore.tables")
import dask
import dask.array as da
import xarray as xr
from casacore.tables import table
from daskms import xds_from_ms
from tabsim.write import write_ms, construct_ms_data_table, add_to_ms


def observation_dataset(dtype):
    dtype = np.dtype(dtype)
    values = np.full((2, 1, 3), 1.0 + 2**-35 + 1j * (2.0 + 2**-34), dtype=dtype)
    variables = {
        name: (("time", "bl", "freq"), da.from_array(values, chunks=(1, 1, 3)))
        for name in ("vis_obs", "vis_ast", "vis_rfi", "vis_calibrated", "noise_data")
    }
    variables.update(
        flags=(
            ("time", "bl", "freq"),
            da.zeros(values.shape, chunks=(1, 1, 3), dtype=bool),
        ),
        noise_std=(("freq",), da.ones(3)),
        antenna1=(("bl",), da.from_array([0])),
        antenna2=(("bl",), da.from_array([1])),
        time_idx=(("time",), da.arange(2)),
        bl_uvw=(
            ("time_fine", "bl", "uvw"),
            da.full((2, 1, 3), 123456.123456789, dtype=np.float64),
        ),
        ants_itrf=(
            ("ant", "xyz"),
            da.full((2, 3), 1e6 + 0.123456789, dtype=np.float64),
        ),
    )
    return xr.Dataset(
        variables,
        coords=dict(
            time=[0.0, 2.0],
            time_mjd=("time", np.array([60000.0, 60000.0 + 2 / 86400.0])),
            freq=150e6 + np.arange(3) * 1e5,
        ),
        attrs=dict(
            n_time=2,
            n_freq=3,
            n_bl=1,
            int_time=2.0,
            n_ant=2,
            dish_diameter=35.0,
            target_ra=30.0,
            target_dec=-30.0,
            tel_name="test",
            target_name="test",
            chan_width=1e5,
            visibility_precision=(
                "double" if dtype == np.dtype(np.complex128) else "single"
            ),
        ),
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_ms_all_visibility_columns_roundtrip(dtype, tmp_path):
    ds = observation_dataset(dtype)
    path = tmp_path / "result.ms"
    write_ms(ds, path)
    expected = ds.vis_obs.compute().values.reshape(2, 3, 1)
    with table(str(path), ack=False) as ms:
        assert (
            ms.getkeyword("TABSIM_VISIBILITY_PRECISION")
            == ds.attrs["visibility_precision"]
        )
        for name in (
            "DATA",
            "MODEL_DATA",
            "CORRECTED_DATA",
            "CAL_DATA",
            "RFI_MODEL_DATA",
            "AST_MODEL_DATA",
            "NOISE_DATA",
            "RFI_DATA",
            "AST_DATA",
        ):
            value = ms.getcol(name)
            assert value.dtype == np.dtype(dtype), name
            want = (
                np.zeros_like(expected)
                if name in ("MODEL_DATA", "CORRECTED_DATA")
                else expected * (2 if name in ("RFI_DATA", "AST_DATA") else 1)
            )
            np.testing.assert_array_equal(value, want)
        assert ms.getcol("UVW").dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(
            ms.getcol("UVW"), np.full((2, 3), 123456.123456789)
        )
    with table(str(path) + "::ANTENNA", ack=False) as antennas:
        assert antennas.getcol("POSITION").dtype == np.dtype(np.float64)
    read = xds_from_ms(str(path), columns=["DATA"], group_cols=[])[0]
    np.testing.assert_array_equal(read.DATA.compute(), expected)
    read.close()


def test_existing_single_columns_reject_double_before_writing(tmp_path):
    path = tmp_path / "existing.ms"
    single = observation_dataset(np.complex64)
    write_ms(single, path)
    with pytest.raises(ValueError, match="cannot safely store"):
        construct_ms_data_table(observation_dataset(np.complex128), str(path))
    with pytest.raises(ValueError, match="Cannot add"):
        add_to_ms(observation_dataset(np.complex128), str(path))
    with table(str(path), ack=False) as ms:
        np.testing.assert_array_equal(
            ms.getcol("DATA"), single.vis_obs.compute().values.reshape(2, 3, 1)
        )


def test_single_rfi_can_accumulate_into_double_ms(tmp_path):
    path = tmp_path / "existing.ms"
    double = observation_dataset(np.complex128)
    write_ms(double, path)
    single = observation_dataset(np.complex64)
    add_to_ms(single, str(path))
    expected = (
        double.vis_obs.compute().values.reshape(2, 3, 1)
        + single.vis_rfi.compute().values.reshape(2, 3, 1).conj()
    )
    with table(str(path), ack=False) as ms:
        assert ms.getcol("DATA").dtype == np.dtype(np.complex128)
        np.testing.assert_array_equal(ms.getcol("DATA"), expected)
