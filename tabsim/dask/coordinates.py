from tabsim.jax import coordinates as coord

from jax import jit

import dask.array as da
from dask.array.core import Array
import xarray as xr


# Keep one callable per kernel in each worker. This does not imply that the
# previous repeated jit() wrapping recompiled every block. No explicit device
# placement or synchronization policy is added. Removing delayed tokenization
# also removes its incidental serialization/synchronization of JAX inputs.
_radec_to_lmn_jit = jit(coord.radec_to_lmn)
_radec_to_XYZ_jit = jit(coord.radec_to_XYZ)
_ENU_to_GEO_jit = jit(coord.ENU_to_GEO)
_GEO_to_XYZ_jit = jit(coord.GEO_to_XYZ)
_GEO_to_XYZ_vmap0_jit = jit(coord.GEO_to_XYZ_vmap0)
_GEO_to_XYZ_vmap1_jit = jit(coord.GEO_to_XYZ_vmap1)
_itrf_to_xyz_jit = jit(coord.itrf_to_xyz)
_enu_to_itrf_jit = jit(coord.enu_to_itrf)
_itrf_to_uvw_jit = jit(coord.itrf_to_uvw)
_angular_separation_jit = jit(coord.angular_separation)
_orbit_vmap_jit = jit(coord.orbit_vmap)


def radec_to_lmn(ra: Array, dec: Array, phase_centre: Array) -> Array:
    n_src = ra.shape[0]

    src_chunk = ra.chunksize[0]

    input = xr.Dataset(
        {
            "ra": (["src"], ra),
            "dec": (["src"], dec),
            "phase_centre": (["cel_space"], phase_centre),
        }
    )
    output = xr.Dataset(
        {
            "lmn": (
                ["src", "lmn_space"],
                da.zeros(  # type: ignore
                    shape=(n_src, 3),
                    chunks=(src_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _radec_to_lmn(ds):
        lmn = _radec_to_lmn_jit(
            ds.ra.data, ds.dec.data, ds.phase_centre.data
        )
        ds_out = xr.Dataset({"lmn": (["src", "lmn_space"], lmn)})
        return ds_out

    ds = xr.map_blocks(_radec_to_lmn, input, template=output)

    return ds.lmn.data


radec_to_lmn.__doc__ = coord.radec_to_lmn.__doc__


def radec_to_XYZ(ra: Array, dec: Array) -> Array:
    n_src = ra.shape[0]

    src_chunk = ra.chunksize[0]

    input = xr.Dataset(
        {
            "ra": (["src"], ra),
            "dec": (["src"], dec),
        }
    )
    output = xr.Dataset(
        {
            "XYZ": (
                ["src", "space"],
                da.zeros(  # type: ignore
                    shape=(n_src, 3),
                    chunks=(src_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _radec_to_XYZ(ds):
        XYZ = _radec_to_XYZ_jit(
            ds.ra.data, ds.dec.data
        )
        ds_out = xr.Dataset({"XYZ": (["src", "space"], XYZ)})
        return ds_out

    ds = xr.map_blocks(_radec_to_XYZ, input, template=output)

    return ds.XYZ.data


radec_to_XYZ.__doc__ = coord.radec_to_XYZ.__doc__


def ENU_to_GEO(geo_ref: Array, ENU: Array) -> Array:
    n_ant = ENU.shape[0]

    ant_chunk = ENU.chunksize[0]

    input = xr.Dataset(
        {
            "geo_ref": (["geo_space"], geo_ref),
            "ENU": (["ant", "space"], ENU),
        }
    )
    output = xr.Dataset(
        {
            "GEO": (
                ["bl", "space"],
                da.zeros(  # type: ignore
                    shape=(n_ant, 3),
                    chunks=(ant_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _ENU_to_GEO(ds):
        GEO = _ENU_to_GEO_jit(
            ds.geo_ref.data, ds.ENU.data
        )
        ds_out = xr.Dataset({"GEO": (["ant", "space"], GEO)})
        return ds_out

    ds = xr.map_blocks(_ENU_to_GEO, input, template=output)

    return ds.GEO.data


ENU_to_GEO.__doc__ = coord.ENU_to_GEO.__doc__


def GEO_to_XYZ(geo: Array, times: Array) -> Array:
    n_time = geo.shape[0]

    time_chunk = geo.chunksize[0]

    input = xr.Dataset(
        {
            "geo": (["time", "space"], geo),
            "times": (["time"], times),
        }
    )
    output = xr.Dataset(
        {
            "XYZ": (
                ["time", "space"],
                da.zeros(  # type: ignore
                    shape=(n_time, 3),
                    chunks=(time_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _GEO_to_XYZ(ds):
        XYZ = _GEO_to_XYZ_jit(
            ds.geo.data, ds.times.data
        )
        ds_out = xr.Dataset({"XYZ": (["time", "space"], XYZ)})
        return ds_out

    ds = xr.map_blocks(_GEO_to_XYZ, input, template=output)

    return ds.XYZ.data


GEO_to_XYZ.__doc__ = coord.GEO_to_XYZ.__doc__


def GEO_to_XYZ_vmap0(geo: Array, times: Array) -> Array:
    n_src, n_time = geo.shape[:2]

    src_chunk = geo.chunks[0][0]
    time_chunk = geo.chunks[1][0]

    input = xr.Dataset(
        {
            "geo": (["src", "time", "space"], geo),
            "times": (["time"], times),
        }
    )
    output = xr.Dataset(
        {
            "XYZ": (
                ["src", "time", "space"],
                da.zeros(  # type: ignore
                    shape=(n_src, n_time, 3),
                    chunks=(src_chunk, time_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _GEO_to_XYZ(ds):
        XYZ = _GEO_to_XYZ_vmap0_jit(
            ds.geo.data, ds.times.data
        )
        ds_out = xr.Dataset({"XYZ": (["src", "time", "space"], XYZ)})
        return ds_out

    ds = xr.map_blocks(_GEO_to_XYZ, input, template=output)

    return ds.XYZ.data


GEO_to_XYZ_vmap0.__doc__ = coord.GEO_to_XYZ_vmap0.__doc__


def GEO_to_XYZ_vmap1(geo: Array, times: Array) -> Array:
    n_time, n_ant = geo.shape[:2]

    time_chunk, ant_chunk = geo.chunksize[:2]

    input = xr.Dataset(
        {
            "geo": (["time", "ant", "space"], geo),
            "times": (["time"], times),
        }
    )
    output = xr.Dataset(
        {
            "XYZ": (
                ["time", "ant", "space"],
                da.zeros(  # type: ignore
                    shape=(n_time, n_ant, 3),
                    chunks=(time_chunk, ant_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _GEO_to_XYZ(ds):
        XYZ = _GEO_to_XYZ_vmap1_jit(
            ds.geo.data, ds.times.data
        )
        ds_out = xr.Dataset({"XYZ": (["time", "ant", "space"], XYZ)})
        return ds_out

    ds = xr.map_blocks(_GEO_to_XYZ, input, template=output)

    return ds.XYZ.data


GEO_to_XYZ_vmap1.__doc__ = coord.GEO_to_XYZ_vmap1.__doc__

from skyfield.api import Distance, load
from skyfield.toposlib import ITRSPosition
import numpy as np
from numpy.typing import NDArray

def itrs_to_gcrs_sf(pos_itrs: NDArray, times_jd: NDArray) -> NDArray:

    ts = load.timescale()
    t_sf = ts.ut1_jd(np.array(times_jd))

    pos_gcrs = np.stack(
        [ITRSPosition(Distance(m=pos)).at(t_sf).position.m.T for pos in pos_itrs], axis=1
    )

    return pos_gcrs


def ITRF_to_XYZ(itrf: Array, gsa: Array) -> Array:
    n_time = gsa.shape[0]
    n_ant = itrf.shape[0]

    time_chunk = gsa.chunksize[0]
    ant_chunk = itrf.chunksize[0]

    input = xr.Dataset(
        {
            "itrf": (["ant", "space"], itrf),
            "gsa": (["time"], gsa),
        }
    )
    output = xr.Dataset(
        {
            "xyz": (
                ["time", "ant", "space"],
                da.zeros(  # type: ignore
                    shape=(n_time, n_ant, 3),
                    chunks=(time_chunk, ant_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _ITRF_to_XYZ(ds):
        XYZ = _itrf_to_xyz_jit(
            ds.itrf.data, ds.gsa.data
        )
        ds_out = xr.Dataset({"xyz": (["time", "ant", "space"], XYZ)})
        return ds_out

    ds = xr.map_blocks(_ITRF_to_XYZ, input, template=output)

    return ds.xyz.data


ITRF_to_XYZ.__doc__ = coord.itrf_to_xyz.__doc__


def ENU_to_ITRF(ENU: Array, lat: Array, lon: Array, el: Array) -> Array:
    n_ant = ENU.shape[0]

    ant_chunk = ENU.chunksize[0]

    input = xr.Dataset(
        {
            "ENU": (["ant", "space"], ENU),
            "lat": lat,
            "lon": lon,
            "el": el,
        }
    )
    output = xr.Dataset(
        {
            "ITRF": (
                ["ant", "space"],
                da.zeros(  # type: ignore
                    shape=(n_ant, 3),
                    chunks=(ant_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _ENU_to_ITRF(ds):
        ITRF = _enu_to_itrf_jit(
            ds.ENU.data,
            ds.lat.data,
            ds.lon.data,
            ds.el.data,
        )
        ds_out = xr.Dataset({"ITRF": (["ant", "space"], ITRF)})
        return ds_out

    ds = xr.map_blocks(_ENU_to_ITRF, input, template=output)

    return ds.ITRF.data


ENU_to_ITRF.__doc__ = coord.enu_to_itrf.__doc__


def ITRF_to_UVW(
    itrf: Array,
    h0: Array,
    dec: Array,
) -> Array:
    n_ant = itrf.shape[0]
    n_time = h0.shape[0]

    ant_chunk = itrf.chunksize[0]
    time_chunk = h0.chunksize[0]

    input = xr.Dataset(
        {
            "itrf": (["ant", "space"], itrf),
            "h0": (["time"], da.atleast_1d(h0)),  # type: ignore
            "dec": (["cel_space_1"], da.atleast_1d(dec)),  # type: ignore
        }
    )

    output = xr.Dataset(
        {
            "uvw": (
                ["time", "ant", "space"],
                da.zeros(  # type: ignore
                    shape=(n_time, n_ant, 3),
                    chunks=(time_chunk, ant_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _ITRF_to_UVW(ds):
        uvw = _itrf_to_uvw_jit(
            ds.itrf.data,
            ds.h0.data,
            ds.dec.data,
        )
        ds_out = xr.Dataset({"uvw": (["time", "ant", "space"], uvw)})
        return ds_out

    ds = xr.map_blocks(_ITRF_to_UVW, input, template=output)

    return ds.uvw.data


ITRF_to_UVW.__doc__ = coord.itrf_to_uvw.__doc__


def angular_separation(rfi_xyz: Array, ants_xyz: Array, ra: Array, dec: Array) -> Array:
    n_src, n_time = rfi_xyz.shape[:2]
    n_ant = ants_xyz.shape[1]

    src_chunk, time_chunk = rfi_xyz.chunksize[:2]
    ant_chunk = ants_xyz.chunksize[1]

    input = xr.Dataset(
        {
            "rfi_xyz": (["src", "time", "space"], rfi_xyz),
            "ants_xyz": (["time", "ant", "space"], ants_xyz),
            "ra": (["cel_space_0"], da.from_array([ra])),  # type: ignore
            "dec": (["cel_space_1"], da.from_array([dec])),  # type: ignore
        }
    )

    output = xr.Dataset(
        {
            "sep": (
                ["src", "time", "ant"],
                da.zeros(  # type: ignore
                    shape=(n_src, n_time, n_ant),
                    chunks=(src_chunk, time_chunk, ant_chunk),
                    dtype=float,
                ),
            )
        }
    )

    def _angular_separation(ds):
        sep = _angular_separation_jit(
            ds.rfi_xyz.data, ds.ants_xyz.data, ds.ra.data, ds.dec.data
        )
        ds_out = xr.Dataset({"sep": (["src", "time", "ant"], sep)})
        return ds_out

    ds = xr.map_blocks(_angular_separation, input, template=output)

    return ds.sep.data


angular_separation.__doc__ = coord.angular_separation.__doc__


def orbit_vmap(
    times: Array,
    elevation: Array,
    inclination: Array,
    lon_asc_node: Array,
    periapsis,
) -> Array:
    n_time = times.shape[0]
    n_src = elevation.shape[0]

    time_chunk = times.chunksize[0]
    src_chunk = elevation.chunksize[0]

    input = xr.Dataset(
        {
            "times": (["time"], times),
            "elevation": (["src"], elevation),
            "inclination": (["src"], inclination),
            "lon_asc_node": (["src"], lon_asc_node),
            "periapsis": (["src"], periapsis),
        }
    )

    output = xr.Dataset(
        {
            "orbit": (
                ["src", "time", "space"],
                da.zeros(  # type: ignore
                    shape=(n_src, n_time, 3),
                    chunks=(src_chunk, time_chunk, 3),
                    dtype=float,
                ),
            )
        }
    )

    def _orbit_vmap(ds):
        orbit = _orbit_vmap_jit(
            ds.times.data,
            ds.elevation.data,
            ds.inclination.data,
            ds.lon_asc_node.data,
            ds.periapsis.data,
        )
        ds_out = xr.Dataset({"orbit": (["src", "time", "space"], orbit)})
        return ds_out

    ds = xr.map_blocks(_orbit_vmap, input, template=output)

    return ds.orbit.data


orbit_vmap.__doc__ = coord.orbit_vmap.__doc__
