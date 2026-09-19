"""Validate the shipped planned SKA-Low layouts, offline and from package data."""
from importlib.resources import files

import numpy as np
import pytest
import yaml


STAGES = {
    "SKA-Low-AA0.5": 4,
    "SKA-Low-AA1": 16,
    "SKA-Low-AA2": 68,
    "SKA-Low-AA*-Phase-1": 108,
    "SKA-Low-AA*": 307,
    "SKA-Low-AA4": 512,
}
DATA = files("tabsim.data").joinpath("telescopes")


def definitions():
    return yaml.safe_load(DATA.joinpath("_telescopes.yaml").read_text())


@pytest.mark.parametrize("name,count", STAGES.items())
def test_packaged_ska_layout(name, count):
    definition = definitions()[name.lower()]
    assert definition["name"] == name
    assert definition["latitude"] == pytest.approx(-26.82472208)
    assert definition["longitude"] == pytest.approx(116.7644482)
    assert definition["elevation"] == 365.0
    assert definition["dish_d"] == 39.0
    # Public names retain AA*, but the package filenames are portable.
    assert not any(c in definition["itrf_path"] for c in '*?<>:"|')
    with DATA.joinpath(definition["itrf_path"]).open() as handle:
        xyz = np.loadtxt(handle)
    assert xyz.shape == (count, 3)
    assert np.isfinite(xyz).all()
    assert len(np.unique(xyz, axis=0)) == count
    # Earth-centred metres at the Australian site, not ENU or angular coordinates.
    assert np.all((np.linalg.norm(xyz, axis=1) > 6.3e6)
                  & (np.linalg.norm(xyz, axis=1) < 6.4e6))
    assert np.all(xyz[:, 0] < 0) and np.all(xyz[:, 1] > 0) and np.all(xyz[:, 2] < 0)
    with DATA.joinpath(definitions()["ska-low-aa4"]["itrf_path"]).open() as handle:
        full = {tuple(row) for row in np.loadtxt(handle)}
    assert all(tuple(row) in full for row in xyz)


@pytest.mark.parametrize("name,count", STAGES.items())
def test_named_ska_configuration_loads(name, count):
    from tabsim.config import apply_telescope_definition, get_telescope_definitions
    from tabsim.dask.observation import Telescope

    definition = get_telescope_definitions(name.swapcase())
    telescope = Telescope(
        latitude=definition["latitude"], longitude=definition["longitude"],
        elevation=definition["elevation"], ITRF_path=definition["itrf_path"],
        tel_name=definition["name"],
    )
    assert telescope.n_ant == count
    assert telescope.ITRF.shape == (count, 3)
    assert float(telescope.elevation.compute()) == 365.0
    # Explicit scientific choices still take precedence over named defaults.
    selected = apply_telescope_definition({
        "name": name, "dish_d": 25.0, "elevation": 0.0,
        "itrf_path": "custom.itrf.txt", "enu_path": None,
    })
    assert selected["dish_d"] == 25.0
    assert selected["elevation"] == 0.0
    assert selected["itrf_path"] == "custom.itrf.txt"
