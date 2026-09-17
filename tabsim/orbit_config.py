"""One normalisation path for the orbit-record configuration.

Every value that decides which satellites are modelled, and which orbital records
are acceptable for them, is validated here, so a malformed entry surfaces as
:class:`TLEConfigurationError` naming the key rather than as a ``ValueError``
from inside pandas several steps later.

Four age settings exist and are deliberately kept distinct:

``extra_orbit_max_age_days``
    Acceptance of explicit user/replay files (``extra_orbit_dir``). ``null`` by
    default so exact ``used_orbits.json`` replay is always possible; remote
    service policy must never constrain it.
``remote_max_age_days``
    Acceptance ceiling for SatChecker records and its managed cache.
``cache_reuse_max_age_days``
    Age below which a cached SatChecker record avoids a new nearest-record
    request.
``search_cache_max_age_days``
    Wall-clock age below which a cached *catalogue search* is reused instead of
    repeated. Nothing to do with the age of an orbital record: it measures how
    stale our picture of which satellites exist is allowed to be.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Optional

import numpy as np
from platformdirs import user_cache_path

from satchecker_client import SatCheckerError as TLEError
from satchecker_client.records import norad_id_of


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Hard ceiling on the age of an orbital record accepted from SatChecker or its
#: managed cache. An emergency backstop against obviously unsuitable remote
#: records, *not* a promise of three-day positional accuracy.
DEFAULT_REMOTE_MAX_AGE_DAYS = 3.0

#: A cached record this close to the observation avoids a new nearest-record
#: request. A request/latency trade-off, not the safety ceiling above.
DEFAULT_CACHE_REUSE_MAX_AGE_DAYS = 1.0

#: Wall-clock age at which a cached catalogue search is refreshed: a day keeps a
#: repeated run off the search endpoint while still noticing a launch, a decay or
#: a new alias. ``None`` reuses a snapshot indefinitely; ``0`` always refreshes.
DEFAULT_SEARCH_CACHE_MAX_AGE_DAYS = 1.0

#: Keys that used to decide where orbital records came from and now do nothing.
#: Silently ignoring one is worse than removing it, so each is rejected by
#: *presence*, null included, with the migration it needs.
OBSOLETE_KEYS = {
    "tle_dir": (
        "`rfi_sources.tle_satellite.tle_dir` is obsolete. Use `extra_orbit_dir` "
        "for existing orbit JSON files; use `ORBIT_CACHE_DIR` to relocate the "
        "managed cache."
    ),
    "spacetrack_path": (
        "`spacetrack_path` is obsolete. SatChecker requires no credentials; "
        "remove this key."
    ),
}


class TLEConfigurationError(TLEError, ValueError):
    """An orbit-related configuration value is missing, malformed or out of range.

    Subclasses :class:`~satchecker_client.client.SatCheckerError` so a caller
    catching "the orbit records could not be obtained" catches a bad
    configuration too, and :class:`ValueError` as these helpers always have.
    """


# ---------------------------------------------------------------------------
# Scalar validation
# ---------------------------------------------------------------------------

def _as_finite_float(value, name: str) -> float:
    """Coerce *value* to a finite float or raise :class:`TLEConfigurationError`."""
    if isinstance(value, bool) or not isinstance(value, (Real, str)):
        raise TLEConfigurationError(f"{name} must be a number, got {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError) as e:
        raise TLEConfigurationError(f"{name} must be a number, got {value!r}") from e
    if not math.isfinite(out):
        raise TLEConfigurationError(f"{name} must be finite, got {value!r}")
    return out


def validate_age_days(value, name: str) -> Optional[float]:
    """Validate an age limit in days: ``None`` (no limit) or a non-negative number.

    ``None`` is an explicit expert opt-out. Negative, non-numeric, NaN and
    infinite values are configuration errors, never silently coerced.
    """
    if value is None:
        return None
    out = _as_finite_float(value, name)
    if out < 0:
        raise TLEConfigurationError(
            f"{name} must be null or a non-negative number of days, got {value!r}"
        )
    return out


def validate_bool(value, name: str) -> bool:
    """Validate a policy switch as a real boolean, never by truthiness.

    These decide whether unverifiable orbital data is accepted and whether the
    service is contacted, so ``"false"``, ``0`` or ``1.0`` has to be an error
    rather than being coerced in whichever direction the accident points.
    """
    if not isinstance(value, bool):
        raise TLEConfigurationError(
            f"{name} must be true or false, got {value!r}"
        )
    return value


def reject_obsolete_keys(satellites: dict) -> None:
    """Raise the migration error for any :data:`OBSOLETE_KEYS` that is present.

    By presence, not by value: a ``null`` is still a key someone meant something
    by. Called before the observation is built and before any request goes out,
    including when satellite simulation is disabled entirely — that is exactly
    the run where nothing else would ever read the section.
    """
    for key, migration in OBSOLETE_KEYS.items():
        if key in (satellites or {}):
            raise TLEConfigurationError(migration)


# ---------------------------------------------------------------------------
# NORAD ID validation
# ---------------------------------------------------------------------------

def _as_norad_id(value, where: str) -> int:
    """Coerce one entry to a positive integral NORAD catalogue ID, exactly.

    The client's :func:`~satchecker_client.records.norad_id_of` does the checking
    because it is exact: a string goes through ``Decimal``, so ``"25544.0"`` — how
    a list written by numpy or read out of a spectral-model CSV routinely arrives
    — is 25544, while ``"25544.000000000001"`` is refused rather than rounded to
    25544 by a float conversion and quietly selecting the ISS. A fractional,
    non-finite, non-positive or non-numeric value is an error naming *where*.
    """
    # bool is Integral and np.bool_ is finite and equal to 0 or 1, so both would
    # pass an exactness check as satellite 1; neither is an identity.
    if isinstance(value, (bool, np.bool_)):
        raise TLEConfigurationError(f"{where}: {value!r} is not a NORAD catalogue ID")
    try:
        return norad_id_of({"NORAD_CAT_ID": value}, where)
    except ValueError as e:
        raise TLEConfigurationError(str(e)) from e


def normalise_norad_ids(values, source: str = "norad_ids") -> list[int]:
    """Return validated, order-preserving, de-duplicated NORAD IDs.

    ``None`` normalises to an empty list — a satellite-free simulation is a
    legitimate configuration. Every other malformed value raises
    :class:`TLEConfigurationError` before any NumPy/Pandas conversion, so a
    fractional or non-numeric entry can never reach the resolver.
    """
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raise TLEConfigurationError(
            f"{source} must be a list of NORAD catalogue IDs, got {values!r}"
        )
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, Sequence):
        try:
            values = list(values)
        except TypeError as e:
            raise TLEConfigurationError(
                f"{source} must be a list of NORAD catalogue IDs, got {values!r}"
            ) from e

    out: list[int] = []
    seen: set[int] = set()
    for i, value in enumerate(values):
        nid = _as_norad_id(value, f"{source}[{i}]")
        if nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


def read_norad_ids_file(path) -> list[int]:
    """Read NORAD catalogue IDs from the first column of *path*.

    Blank lines and ``#`` comments are ignored; the first whitespace-separated
    field of every other line must be a positive integer and anything after it is
    ignored, so a file pairing IDs with a name or a note still reads. Errors name
    the file *and line number*, and IDs keep their first occurrence's order.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as e:
        raise TLEConfigurationError(
            f"NORAD ID file could not be read ({path}): {e}"
        ) from e

    out: list[int] = []
    seen: set[int] = set()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        nid = _as_norad_id(line.split()[0], f"{path}:{lineno}")
        if nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


# ---------------------------------------------------------------------------
# Normalised configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrbitConfig:
    """Fully validated orbit-record configuration."""

    #: Empty in replay mode: the saved selection is the selection, so the run's
    #: own IDs and names are not read.
    norad_ids: list[int] = field(default_factory=list)
    sat_names: list[str] = field(default_factory=list)
    extra_orbit_dir: Optional[str] = None
    extra_orbit_max_age_days: Optional[float] = None
    remote_max_age_days: Optional[float] = DEFAULT_REMOTE_MAX_AGE_DAYS
    cache_reuse_max_age_days: Optional[float] = DEFAULT_CACHE_REUSE_MAX_AGE_DAYS
    #: Accept TLE lines whose checksum digit is missing, carried as
    #: ``unverified_missing_checksum`` for the life of the record. Strict by
    #: default, and identical on remote acquisition, explicit files and replay.
    allow_missing_checksum: bool = False
    #: Wall-clock freshness of a cached catalogue search; see
    #: :data:`DEFAULT_SEARCH_CACHE_MAX_AGE_DAYS`.
    search_cache_max_age_days: Optional[float] = DEFAULT_SEARCH_CACHE_MAX_AGE_DAYS
    #: Forbid every SatChecker request. Cached searches are reused at any age and
    #: cached orbit records within ``remote_max_age_days``: the hard ceiling still
    #: applies, offline being about reachability and not about acceptability.
    offline: bool = False
    #: Directory of a previous run's ``input_data``. When set, its saved NORAD IDs
    #: and records *are* the selection: no discovery, no cache, no network, no
    #: visibility reselection and no ``max_n_sat`` truncation.
    replay_orbit_dir: Optional[str] = None


def validate_remote_ages(
    remote_max_age_days, cache_reuse_max_age_days
) -> tuple[Optional[float], Optional[float]]:
    """Validate the two remote age settings and their one cross-constraint.

    A reuse threshold above the hard ceiling would let a cached record suppress
    the request that could have replaced it, and then be rejected by the ceiling
    anyway — no record, and no attempt to get one.
    """
    remote = validate_age_days(remote_max_age_days, "remote_max_age_days")
    reuse = validate_age_days(cache_reuse_max_age_days, "cache_reuse_max_age_days")
    if reuse is not None and remote is not None and reuse > remote:
        raise TLEConfigurationError(
            f"cache_reuse_max_age_days ({reuse:g}) must not exceed "
            f"remote_max_age_days ({remote:g})"
        )
    return remote, reuse


def normalise_orbit_config(satellites: dict) -> OrbitConfig:
    """Validate the ``rfi_sources.tle_satellite`` section into an :class:`OrbitConfig`.

    NORAD IDs come from ``norad_ids`` and, when set, the first column of
    ``norad_ids_path``; ``sat_names`` are resolved later by
    :func:`tabsim.orbit.resolve_names`, which needs the network where this does not.

    ``replay_orbit_dir`` changes what this reads. A frozen replay's saved IDs are
    the selection, so the run's own ``norad_ids``, ``sat_names`` and
    ``norad_ids_path`` are neither read nor validated and come back empty: a
    replay of a run whose ID file has since moved must still be possible, and a
    leftover nothing will look at cannot refuse the run. What was overridden is
    logged from the raw configuration by
    :func:`tabsim.config.add_tle_satellite_sources`.
    """
    satellites = satellites or {}
    reject_obsolete_keys(satellites)

    replay_orbit_dir = satellites.get("replay_orbit_dir") or None
    extra_orbit_dir = satellites.get("extra_orbit_dir") or None
    if replay_orbit_dir and extra_orbit_dir:
        raise TLEConfigurationError(
            "replay_orbit_dir and extra_orbit_dir must not both be set: their "
            "source-selection contracts differ. extra_orbit_dir is ordinary "
            "per-ID precedence within this run's own satellite selection, while "
            "replay_orbit_dir replaces that selection with a previous run's saved "
            "one. Letting either win silently would make the other look effective."
        )

    if replay_orbit_dir:
        norad_ids: list[int] = []
        names: list[str] = []
    else:
        norad_ids = normalise_norad_ids(
            satellites.get("norad_ids"), "tle_satellite.norad_ids"
        )
        ids_path = satellites.get("norad_ids_path")
        if ids_path:
            seen = set(norad_ids)
            norad_ids += [
                nid for nid in read_norad_ids_file(ids_path) if nid not in seen
            ]

        names = satellites.get("sat_names") or []
        if isinstance(names, (str, bytes)):
            raise TLEConfigurationError(
                f"tle_satellite.sat_names must be a list of names, got {names!r}"
            )
        names = [str(name).strip() for name in names if str(name).strip()]

    remote_max_age, cache_reuse_age = validate_remote_ages(
        satellites.get("remote_max_age_days", DEFAULT_REMOTE_MAX_AGE_DAYS),
        satellites.get("cache_reuse_max_age_days", DEFAULT_CACHE_REUSE_MAX_AGE_DAYS),
    )

    return OrbitConfig(
        norad_ids=norad_ids,
        sat_names=names,
        extra_orbit_dir=str(extra_orbit_dir) if extra_orbit_dir else None,
        extra_orbit_max_age_days=validate_age_days(
            satellites.get("extra_orbit_max_age_days"), "extra_orbit_max_age_days"
        ),
        remote_max_age_days=remote_max_age,
        cache_reuse_max_age_days=cache_reuse_age,
        allow_missing_checksum=validate_bool(
            satellites.get("allow_missing_checksum", False),
            "tle_satellite.allow_missing_checksum",
        ),
        search_cache_max_age_days=validate_age_days(
            satellites.get(
                "search_cache_max_age_days", DEFAULT_SEARCH_CACHE_MAX_AGE_DAYS
            ),
            "search_cache_max_age_days",
        ),
        offline=validate_bool(
            satellites.get("offline", False), "tle_satellite.offline"
        ),
        replay_orbit_dir=str(replay_orbit_dir) if replay_orbit_dir else None,
    )


def orbit_cache_dir() -> Path:
    """Return the managed orbit cache directory, creating it if possible.

    ``ORBIT_CACHE_DIR`` if set, otherwise the platform user-cache directory
    (``~/.cache/orbit-cache`` on Linux, ``~/Library/Caches/orbit-cache`` on
    macOS). A directory that cannot be created is *not* an error here: the path
    is returned regardless, reads then miss and writes are reported and skipped,
    so a run with a valid fetch is never lost to an unusable cache location.

    Lives here because both the record cache in :mod:`tabsim.orbit` and the
    catalogue-search cache in :mod:`tabsim.satchecker_names` share it.
    """
    p = Path(os.environ.get("ORBIT_CACHE_DIR") or user_cache_path("orbit-cache"))
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return p


def observation_epoch_jd(times_jd) -> float:
    """Mean observation epoch (UTC JD) of an already-read time array.

    Every age comparison is measured from this one value, so it is derived once
    rather than recomputed at each call site.
    """
    return float(np.atleast_1d(np.asarray(times_jd, dtype=float)).mean())
