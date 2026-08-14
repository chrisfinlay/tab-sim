"""IAU CPS SatChecker access for tabsim, split by responsibility.

**Vendored from TABASCAL.** This package is a direct copy of ``tabascal/satchecker/``
as it landed in epfl-radio-astro/tabascal#92, at commit
``bd51a1109d662a88b3271b32d07a41ae09c1acd0``, so the two copies stay diffable and a
fix on either side transplants cleanly. Keep it that way: tabsim-specific additions
belong outside this package (:mod:`tabsim.satchecker_names` is the one that exists
today), not inside it.

Deviations from that copy, in full:

* the package name in cross-references, and the ``USER_AGENT`` contact URL;
* ``cache.read_legacy_tle_records`` passes ``precise_float=True`` to
  ``pandas.read_json``. Without it pandas' float parser is not correctly rounded
  and reads an OMM eccentricity of 0.0066635 back as 0.006663499999999999, so a
  replayed trajectory disagrees with the run its file was written to reproduce.
  A bug in the upstream copy too, and worth sending back to it.

- :mod:`tabsim.satchecker.client` — HTTP transport and response normalisation
  for both nearest-record endpoints; no tabsim/JAX/casacore imports, so it can
  be extracted later.
- :mod:`tabsim.satchecker.tle_parse` — TLE line parsing, and the element range
  checks both record kinds share.
- :mod:`tabsim.satchecker.records` — what a record is, when it is valid, and
  what it means; the only place either format is named.
- :mod:`tabsim.satchecker.cache` — validated per-NORAD record storage.
- :mod:`tabsim.satchecker.service` — endpoint selection, bounded concurrent
  acquisition, response validation, and resilient cache writes.

tabsim-specific orchestration (source precedence, ``extra_orbit_dir`` age policy,
remote-age acceptance, and complete-coverage enforcement) lives in
:mod:`tabsim.orbit`.

The names most callers need are re-exported here.
"""

from .client import (
    BASE_URL,
    HANDOVER_JD,
    OMM_COLUMNS,
    TLE_COLUMNS,
    SatCheckerError,
    SatCheckerRateLimitError,
    SatCheckerResponseError,
    SatCheckerTransportError,
    fetch_nearest_omm,
    fetch_nearest_tle,
)
from .cache import (
    CacheValidationError,
    TextOrbitCache,
    read_legacy_tle_records,
)
from .records import (
    KIND_OMM,
    KIND_TLE,
    KIND_FIELD,
    RecordKindError,
    record_elements,
    record_epoch_jd,
    record_kind,
    validate_record,
)
from .service import (
    MAX_WORKERS,
    NearestBatchResult,
    fetch_nearest_batch,
    nearest_endpoints_for,
    store_or_warn,
)

__all__ = [
    "BASE_URL",
    "HANDOVER_JD",
    "OMM_COLUMNS",
    "TLE_COLUMNS",
    "fetch_nearest_omm",
    "nearest_endpoints_for",
    "SatCheckerError",
    "SatCheckerRateLimitError",
    "SatCheckerResponseError",
    "SatCheckerTransportError",
    "fetch_nearest_tle",
    "CacheValidationError",
    "TextOrbitCache",
    "read_legacy_tle_records",
    "KIND_OMM",
    "KIND_TLE",
    "KIND_FIELD",
    "RecordKindError",
    "record_elements",
    "record_epoch_jd",
    "record_kind",
    "validate_record",
    "MAX_WORKERS",
    "NearestBatchResult",
    "fetch_nearest_batch",
    "store_or_warn",
]
