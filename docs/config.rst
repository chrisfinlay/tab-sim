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
- `tle_satellite`: satellites propagated from real orbital records (e.g. Starlink)
- `stationary`: Ground-based RFI sources

Orbital records for ``tle_satellite``
-------------------------------------

Records come from the `IAU CPS SatChecker <https://satchecker.cps.iau.org/>`_ and
need no account or credentials. Which satellites a run models, and which records
are acceptable for them, is controlled entirely by this section.

Selecting satellites
~~~~~~~~~~~~~~~~~~~~

``sat_names``
   Substring match against the catalogue name, as Space-Track's ``like`` search
   was: ``navstar`` selects every ``NAVSTAR nn (USA nnn)`` entry. The catalogue is
   upper case and the service matches case-sensitively, so the query is upper-cased
   for you; a few mixed-case names cannot be reached by any single spelling, and
   ``%`` and ``_`` are SQL wildcards. Satellites the catalogue says had decayed, or
   had not launched, **at the observation epoch** are excluded. A name the catalogue
   does not know contributes nothing and is reported; a search that *failed* stops
   the run.
``norad_ids`` / ``norad_ids_path``
   Catalogue numbers. Every one of them must end up with an acceptable record or
   the run stops — tabsim does not silently omit a satellite asked for by number.
``max_n_sat``
   A final limit on this run's selection. Not an acquisition budget, and never
   applied to a frozen replay.

Record age
~~~~~~~~~~

Four age settings, deliberately distinct:

``remote_max_age_days`` (default 3)
   Hard ceiling on ``|record epoch − observation epoch|`` for a record from
   SatChecker or its managed cache. An emergency backstop, not a claim of
   three-day positional accuracy; ``1`` restores the old ±1-day acceptance bound,
   and ``null`` is an expert opt-out that removes the ceiling.
``cache_reuse_max_age_days`` (default 1)
   A cached record this close to the observation avoids a new request. Must not
   exceed ``remote_max_age_days``, or a cached record could suppress the request
   that would have replaced it and then be refused by the ceiling anyway.
``extra_orbit_max_age_days`` (default ``null`` = unlimited)
   Acceptance of records from ``extra_orbit_dir``. That is your data, so the
   service's age policy never constrains it.
``search_cache_max_age_days`` (default 1)
   Wall-clock age at which a cached *catalogue search* is refreshed — how stale the
   picture of which satellites exist may be, not how old a record may be. ``null``
   reuses a snapshot indefinitely; ``0`` refreshes on every online lookup.

Sources, offline running and replay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``extra_orbit_dir``
   Directory of your own orbit tables (TLE or OMM JSON), consulted per NORAD ID
   ahead of the managed cache and SatChecker. Ordinary source precedence: the run
   still chooses its own satellites. Every ``*.json`` in it must be a readable
   orbit table with a usable catalogue number on every row, or the run stops naming
   the file.
``replay_orbit_dir``
   A previous run's ``input_data`` directory. Its ``norad_ids.yaml`` and
   ``used_orbits.json`` *are* the selection: no catalogue search, no cache, no
   request, no visibility reselection and no ``max_n_sat``. Cannot be combined with
   ``extra_orbit_dir``, whose contract differs.
``offline`` (default ``false``)
   Forbid every SatChecker request. Cached searches are reused whatever their age,
   with a warning; cached orbit records still have to satisfy
   ``remote_max_age_days``. A satellite with no acceptable local record — none held
   at all, or one held only outside that ceiling — stops the run saying the local
   state is insufficient. Neither case says anything about what SatChecker has:
   nothing asked it.
``allow_missing_checksum`` (default ``false``)
   Accept TLE lines that reached us without their checksum digit (roughly
   2001–2018 in SatChecker's archive) and carry them as
   ``unverified_missing_checksum`` for the life of the record. A line whose
   checksum is *present and wrong* is refused under either setting. The policy
   applies identically to remote records, ``extra_orbit_dir`` and replay, and such
   records are never written to the shared orbit cache — the saved run records are
   the only way to replay them.

Obsolete keys
~~~~~~~~~~~~~

``tle_dir`` and ``spacetrack_path`` are rejected by presence, null included, with
the migration each needs. They used to decide where orbital records came from, so a
run that kept one would otherwise succeed while quietly ignoring it.

Caches
~~~~~~

Orbit records and catalogue searches are cached under the platform user-cache
directory (``~/.cache/orbit-cache`` on Linux, ``~/Library/Caches/orbit-cache`` on
macOS). ``ORBIT_CACHE_DIR`` relocates both. Set ``TABSIM_TLE_LOG_DETAIL=1`` for the
full per-satellite listing in a run's log.

**observation**:
- Defines start time, frequency setup, number of steps, and target coordinates.

**output**:
- Specifies whether to output `.zarr` or Measurement Set (`.ms`) files, plus naming and path controls.

**telescope**:
- Location and antenna setup, required to simulate real interferometric measurements.

**dask**, **diagnostics**, **gains**:
- Tuning for memory and plotting diagnostics, and simulation of instrument gain fluctuations.

Planned SKA-Low configurations
-----------------------------

Set ``telescope.name`` to one of the following (case-insensitive), leaving
``itrf_path``, ``enu_path`` and ``dish_d`` unset to use the packaged definition:

.. list-table:: Packaged planned station layouts
   :header-rows: 1

   * - Name
     - Stations
   * - ``SKA-Low-AA0.5``
     - 4
   * - ``SKA-Low-AA1``
     - 16
   * - ``SKA-Low-AA2``
     - 68
   * - ``SKA-Low-AA*-Phase-1``
     - 108
   * - ``SKA-Low-AA*``
     - 307
   * - ``SKA-Low-AA4``
     - 512

For example::

   telescope:
     name: "SKA-Low-AA1"

These are versioned **planned** layouts, not a live record of commissioned
stations. Each row is a station centre in geocentric metres, rounded to 1 mm.
The reference location is WGS84 longitude 116.7644482 degrees, latitude
-26.82472208 degrees, and ellipsoidal height 365 m. The default 39 m diameter
uses tab-sim's existing Airy beam approximation, not a detailed SKA station beam.
Configured geometry, diameter and elevation overrides retain their usual
precedence. ``n_ant`` selects the first rows of the chosen table, not another
assembly stage.

The public ``AA*`` names are literal names; their packaged filenames use
``AAstar`` for portability. Source revisions and regeneration instructions are
in ``tabsim/data/telescopes/SKA-Low-PROVENANCE.md``.
