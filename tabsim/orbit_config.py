"""One normalisation path for the orbit-record configuration.

Every value that decides which satellites are modelled, and which orbital records
are acceptable for them, is validated here, so a malformed entry surfaces as
:class:`TLEConfigurationError` with the key that caused it rather than as a
``ValueError`` from somewhere inside pandas several steps later.

Three age settings exist and are deliberately kept distinct:

``extra_orbit_max_age_days``
    Acceptance of explicit user/replay files (``extra_orbit_dir``). ``null`` by
    default so exact ``used_orbits.json`` replay is always possible; remote
    service policy must never constrain it.
``remote_max_age_days``
    Acceptance ceiling for SatChecker records and its managed cache.
``cache_reuse_max_age_days``
    Age below which a cached SatChecker record avoids a new nearest-record
    request.

Ported from ``tabascal/orbit_config.py`` (epfl-radio-astro/tabascal#92), less the
Measurement Set epoch derivation and the model-component introspection, neither
of which tabsim has an equivalent of: tabsim builds its own time grid from the
simulation configuration, and its satellite sources are named directly rather
than implied by a trajectory component.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Optional

import numpy as np

from satchecker_client import SatCheckerError as TLEError


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Hard ceiling on the age of an orbital record accepted from SatChecker or its
#: managed cache. An emergency backstop against obviously unsuitable remote
#: records, *not* a promise of three-day positional accuracy.
DEFAULT_REMOTE_MAX_AGE_DAYS = 3.0

#: A cached record this close to the observation is good enough to avoid a new
#: nearest-record request. A request/latency trade-off, not the safety ceiling
#: above.
DEFAULT_CACHE_REUSE_MAX_AGE_DAYS = 1.0


class TLEConfigurationError(TLEError, ValueError):
    """An orbit-related configuration value is missing, malformed or out of range.

    Subclasses :class:`~satchecker_client.client.SatCheckerError` so a caller
    catching "the orbit records could not be obtained" catches a bad
    configuration too, and :class:`ValueError` because that is what a bad
    argument to these helpers has always raised.
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


# ---------------------------------------------------------------------------
# NORAD ID validation
# ---------------------------------------------------------------------------

def _as_norad_id(value, where: str) -> int:
    """Coerce one entry to a positive integral NORAD catalogue ID."""
    if isinstance(value, bool):
        raise TLEConfigurationError(f"{where}: {value!r} is not a NORAD catalogue ID")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise TLEConfigurationError(f"{where}: empty NORAD catalogue ID")
        try:
            value = int(text)
        except ValueError:
            # A NORAD ID list written by numpy, or read out of a spectral-model
            # CSV, routinely arrives as "25544.0"; the fractional check below is
            # what keeps that from admitting a genuinely fractional entry.
            try:
                value = float(text)
            except ValueError as e:
                raise TLEConfigurationError(
                    f"{where}: {value!r} is not a NORAD catalogue ID"
                ) from e
    if isinstance(value, Integral):
        out = int(value)
    elif isinstance(value, Real):
        as_float = float(value)
        # Reject NaN/inf and fractional values before int(): a fractional ID would
        # truncate to a *different* satellite and an infinity raises inside numpy.
        if not math.isfinite(as_float) or as_float != round(as_float):
            raise TLEConfigurationError(
                f"{where}: {value!r} is not a finite integer NORAD catalogue ID"
            )
        out = int(round(as_float))
    else:
        raise TLEConfigurationError(f"{where}: {value!r} is not a NORAD catalogue ID")
    if out <= 0:
        raise TLEConfigurationError(
            f"{where}: NORAD catalogue IDs must be positive, got {out}"
        )
    return out


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
    field of every other line must be a positive integer, and anything after it
    is ignored — matching the ``usecols=0`` this replaces, so a file pairing IDs
    with a name or a note still reads. Errors name the file *and line number* so
    a typo in a long list is trivially located. IDs are de-duplicated with their
    first occurrence's order preserved.
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

    norad_ids: list[int] = field(default_factory=list)
    sat_names: list[str] = field(default_factory=list)
    extra_orbit_dir: Optional[str] = None
    extra_orbit_max_age_days: Optional[float] = None
    remote_max_age_days: Optional[float] = DEFAULT_REMOTE_MAX_AGE_DAYS
    cache_reuse_max_age_days: Optional[float] = DEFAULT_CACHE_REUSE_MAX_AGE_DAYS


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

    NORAD IDs come from ``norad_ids`` and, when set, are extended by the first
    column of ``norad_ids_path``; ``sat_names`` are resolved against SatChecker's
    name index later, by :func:`tabsim.orbit.resolve_names`, because that needs
    the network and this does not.
    """
    satellites = satellites or {}

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
    extra_orbit_dir = satellites.get("extra_orbit_dir") or None

    return OrbitConfig(
        norad_ids=norad_ids,
        sat_names=names,
        extra_orbit_dir=str(extra_orbit_dir) if extra_orbit_dir else None,
        extra_orbit_max_age_days=validate_age_days(
            satellites.get("extra_orbit_max_age_days"), "extra_orbit_max_age_days"
        ),
        remote_max_age_days=remote_max_age,
        cache_reuse_max_age_days=cache_reuse_age,
    )


def observation_epoch_jd(times_jd) -> float:
    """Mean observation epoch (UTC JD) of an already-read time array.

    Every age comparison is measured from this one value, so it is derived in one
    place rather than recomputed at each call site.
    """
    return float(np.atleast_1d(np.asarray(times_jd, dtype=float)).mean())
