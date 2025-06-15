Simulation Configuration
========================

tab-sim uses YAML configuration files to define simulations.

Default Config Structure
------------------------

The base config is located at:
``tab-sim/tabsim/data/config_files/sim_config_base.yaml``

Key Sections
------------

**ast_sources**:
- `exp`, `gauss`, `point`, `pow_spec`: Define different source shapes and their random generation parameters.

**rfi_sources**:
- `satellite`: Circular trajectory satellites
- `tle_satellite`: TLE-based satellites (e.g., Starlink)
- `stationary`: Ground-based RFI sources

**observation**:
- Defines start time, frequency setup, number of steps, and target coordinates.

**output**:
- Specifies whether to output `.zarr` or Measurement Set (`.ms`) files, plus naming and path controls.

**telescope**:
- Location and antenna setup, required to simulate real interferometric measurements.

**dask**, **diagnostics**, **gains**:
- Tuning for memory and plotting diagnostics, and simulation of instrument gain fluctuations.
