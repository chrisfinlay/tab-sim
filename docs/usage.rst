Usage Guide
===========

Installation
------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/chrisfinlay/tab-sim.git

Install via pip (CPU-only):

.. code-block:: bash

   pip install -e ./tab-sim/

Or with GPU support:

.. code-block:: bash

   pip install -e ./tab-sim/[gpu]

Alternatively, use Docker:

.. code-block:: bash

   docker build -t tab-sim:latest ./tab-sim/
   docker run -it -v $(pwd):/data tab-sim:latest bash

Running Simulations
-------------------

Simulations are defined by YAML config files and can be launched using:

.. code-block:: bash

   sim-vis -c path/to/config.yaml

For help:

.. code-block:: bash

   sim-vis -h

Satellite orbital records
-------------------------

Orbital records come from the IAU CPS SatChecker service and need no credentials.
Three ways to run a simulation that includes satellites:

.. code-block:: bash

   sim-vis -c observation.yaml
   sim-vis -c observation.yaml --offline
   sim-vis -c observation.yaml --replay-orbit-dir previous/input_data

The first fetches what it needs, caching records and catalogue searches under
``ORBIT_CACHE_DIR`` (by default the platform user-cache directory).

``--offline`` makes no requests at all. It reuses cached catalogue searches whatever
their age, with a warning saying how old they are, and cached orbit records within
``remote_max_age_days`` — offline is about what can be reached, not about what an
acceptable record is. A satellite with no acceptable local record stops the run
rather than being dropped from it.

``--replay-orbit-dir`` reads the ``norad_ids.yaml`` and ``used_orbits.json`` a
previous run saved in its ``input_data`` directory and treats them as the
selection: no catalogue search, no cache, no request, no visibility reselection and
no ``max_n_sat``. Reproducing that run's visibilities exactly also assumes the same
observation, spectral models, random seeds and numerical environment; what replay
freezes is the orbital input. ``--extra-orbit-dir`` (also ``-eod``) is not the same
thing — it is ordinary per-satellite source precedence, and the run still chooses
its own satellites.

TLE lines whose checksum digit is missing are refused by default, which can leave no
record for dates in roughly 2001–2018. ``--allow-missing-checksum`` accepts them and
marks the records unverified for good; the same flag is then needed to replay them.

See :doc:`config` for the full set of ``rfi_sources.tle_satellite`` settings.
