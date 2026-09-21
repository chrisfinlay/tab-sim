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
Choosing and fetching them is done by ``satchecker-client``, shared with TABASCAL;
the source order, the age defaults, what happens when a satellite cannot be
resolved, and every message about it are tab-sim's own. ``satchecker-client`` 0.2.0
or later is required; it is the first release with the API used here.

Three ways to run a simulation that includes satellites:

.. code-block:: bash

   sim-vis -c observation.yaml
   sim-vis -c observation.yaml --offline
   sim-vis -c observation.yaml --replay-orbit-dir previous/input_data

The first fetches what it needs, caching records and catalogue searches under
``ORBIT_CACHE_DIR`` (by default the platform user-cache directory).

Where the cache holds several records for one satellite, the one selected is the
nearest to the observation epoch *that this run can use*: a record the run's checksum policy
refuses is not a candidate, so a nearer unusable row neither displaces a usable one
nor provokes a request. What is chosen is still within ``remote_max_age_days``, and
if it is also within ``cache_reuse_max_age_days`` no request is sent — which is what
lets an offline run resolve from an acceptable record it holds rather than fail
because a nearer row is unreadable. Earlier versions looked at the nearest record
first and asked SatChecker when it proved unusable.

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
freezes is the orbital input. A directory written by an earlier tab-sim replays
here, and one written here replays there: the two files hold the same identities,
retained values and checksum provenance, which is the contract — not that the bytes
are identical. ``--extra-orbit-dir`` (also ``-eod``) is not the same thing — it is
ordinary per-satellite source precedence, searched ahead of the cache and the
service, and the run still chooses its own satellites from its own names, IDs,
visibility cuts and ``max_n_sat``.

TLE lines whose checksum digit is missing are refused by default, which can leave no
record for dates in roughly 2001–2018. ``--allow-missing-checksum`` accepts them and
marks the records unverified for good; the same flag is then needed to replay them.

See :doc:`config` for the full set of ``rfi_sources.tle_satellite`` settings.

Visibility precision
--------------------

Visibilities default to complex64. Set ``observation.visibility_precision: double``
in YAML, ``--visibility-precision double`` on ``sim-vis``, or
``Observation(..., visibility_precision="double")`` for complex128. Geometry
and phase remain float64. See `Visibility precision <visibility-precision.md>`_
for numerical, storage and benchmark details.
