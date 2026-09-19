Output Files
============

tab-sim outputs a simulation directory containing all input and output data.

Directory Structure
-------------------

.. code-block:: text

   sim_name/
   ├── sim_name.zarr/
   ├── sim_name.ms/
   ├── AngularSeps.png
   ├── SourceAltitude.png
   ├── UV.png
   ├── log_sim_*.txt
   └── input_data/
       ├── MeerKAT.itrf.txt
       ├── norad_ids.yaml
       ├── used_orbits.json
       ├── norad_satellite.rfimodel
       └── sim_config.yaml

``norad_ids.yaml`` and ``used_orbits.json`` are written together, as one validated
pair: the satellites the run actually propagated, and the orbit record it used for
each, with the provenance each record arrived with. ``sim-vis --replay-orbit-dir
<sim_name>/input_data`` reads those two files and nothing else, so the run's
orbital input can be reproduced without the shared cache, without the network and
whatever the service serves by then. They are written even when the run modelled no
satellites, since a missing file cannot state that. Directories written by earlier
versions replay unchanged, and these replay with them.

Zarr Output (.zarr)
-------------------

Use `xarray` to open `.zarr` files:

.. code-block:: python

   import xarray as xr
   xds = xr.open_zarr("path/to/sim_name.zarr/")
   print(xds)

Includes:
- Coordinates: `ant`, `freq`, `time`, `uvw`, `radec`, etc.
- Data: `vis_obs`, `vis_rfi`, `vis_ast`, `noise_data`, `rfi_tle_sat_xyz`, etc.
- Attributes: observation metadata and simulation parameters

Measurement Set (.ms)
---------------------

Standard columns:
- `DATA`, `CORRECTED_DATA`, `MODEL_DATA`

Extended columns:
- `CAL_DATA`, `AST_MODEL_DATA`, `RFI_MODEL_DATA`, `NOISE_DATA`

Each contains different subsets or processed versions of the simulated data, useful for analysis and calibration testing.


Thermal noise and reproducibility
--------------------------------

``noise_std`` is the standard deviation of **each** real and imaginary component
for one baseline and channel, in Jy. For identical antennas and unit correlator
efficiency it is ``SEFD / sqrt(2 * chan_width * int_time)``. Thus the expected
squared magnitude of the complex noise is ``2 * noise_std**2``. Both the Dask and
JAX helpers use this convention. The Dask equation previously omitted the factor
of two in the denominator; correcting it reduces its component standard deviation
by ``sqrt(2)``. The standalone JAX noise generator now compensates for JAX's complex
normal convention (variance one half per component).

Visibility noise is generated lazily in chunks matching the visibility array,
using independent Dask random streams for blocks and real/imaginary components.
For a fixed integer seed, shape, chunk layout and NumPy/Dask version, values repeat
regardless of task execution order or worker count. Changing the input chunk layout
can change the samples, and samples differ from earlier eager NumPy releases.
Explicit ``random_seed=0`` is supported by ``Observation.calculate_vis``.
These changes intentionally affect noisy visibilities, calibrated noise and flags;
astronomical and RFI signals with fixed physical inputs are unchanged. Source
flux limits expressed in image-noise ``sigma`` units also decrease with the
corrected equation, so such configurations can generate different source fluxes.

The baseline noise equation follows the `NRAO baseline sensitivity convention
<https://science.nrao.edu/facilities/vlba/docs/manuals/oss2013a/baseline-sensitivity>`_.
