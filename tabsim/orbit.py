"""tabsim orbit-record orchestration and local orbital-element derivation.

Records are sourced from the IAU CPS SatChecker service via
:mod:`satchecker_client` — no account or credentials are required. This module is
the tabsim adapter: it resolves each requested NORAD ID against an ordered set of
sources, applies the configurable age policies, drives the per-satellite cache,
and derives the orbital elements locally.

SatChecker serves two record formats from two non-overlapping archives — TLEs up
to 2026-07-11, OMM from 2026-07-12 — and a run near that boundary may need
either. Nothing in this module branches on which: every format question is
answered by :mod:`satchecker_client.records`, so the policy below works off an
epoch and an opaque record.

``replay_orbit_dir`` is not part of that ordering at all: a frozen replay
(:func:`load_replay_orbits`) *replaces* the selection with a previous run's saved
IDs and records, reading nothing else — no discovery, no cache, no network. It is
selected before anything below, by :func:`tabsim.config.add_tle_satellite_sources`.

For an ordinary run, source precedence is resolved **independently per NORAD ID**:

  1. ``extra_orbit_dir`` — user-supplied local files, of either kind, read
     strictly: a file that is not a readable orbit table stops the run naming
     itself, rather than falling through to the service and quietly modelling
     records the user said not to use. The record whose epoch is closest to the
     observation epoch is chosen; it is accepted only if within
     ``extra_orbit_max_age_days`` (``None`` = unlimited). An accepted record wins
     outright — later sources are not consulted for that ID. This is *your* data:
     the remote service's age policy never applies to it. Note that this is
     ordinary per-ID precedence and freezes nothing: the run's own names, ID list,
     visibility cuts and ``max_n_sat`` still choose the satellites.
  2. Per-satellite cache — the cached record whose epoch is closest to the
     observation. If it is within ``cache_reuse_max_age_days``, it avoids a
     network request. An older record within the hard ceiling remains an offline
     fallback while tabsim asks SatChecker for something closer.
  3. SatChecker — exact-epoch lookups run with bounded concurrency for the
     remaining IDs, against the archive the observation epoch falls in, with the
     other archive as a fallback (see :func:`_fetch_from_service`). Valid
     responses are merged into the per-NORAD cache and may serve nearby
     observations later. ``offline`` skips this step entirely without relaxing the
     age ceiling: offline is about what can be reached, not about what an
     acceptable record is.

**Checksums.** A TLE line whose checksum is present and wrong is refused under
every setting. ``allow_missing_checksum`` decides only what happens to a line that
arrived without its checksum digit at all, as some of SatChecker's 2001–2018
archive did, and it decides it the same way on every route — remote, local file
and replay — because a default that rejected them remotely and accepted them from
a file would advertise strictness whose workaround is to save the record once.
Accepted ones carry :data:`CHECKSUM_STATUS_FIELD` for the rest of their lives and
stay out of the shared cache, which other applications also read.

**Coverage.** Every explicitly requested NORAD ID must end up with an accepted
record, and :func:`require_complete_coverage` raises :class:`OrbitError` naming
each failure and its remedies if one does not. Satellites named rather than
numbered (``sat_names``) are a catalogue *query*, so an unrecognised name, a
genuinely empty reply or a record rejected on age excludes that satellite with its
reason — :func:`report_named_coverage`. What neither route tolerates is not
*knowing*: an unresolved request, response or validation failure, and running out
of local state offline, are fatal on both, through the same error. Reporting an
outage as an absent satellite is how a SatChecker failure used to become a
complete-looking observation with no satellite RFI in it.

Ported from ``tabascal/orbit.py`` (epfl-radio-astro/tabascal#92), less the
multi-process broadcast and the Measurement Set preflight, neither of which
tabsim has: simulation is single-process and builds its own time grid.
"""

from __future__ import annotations

import functools
import importlib.metadata as _metadata
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import satchecker_client as satchecker
from satchecker_client import (
    CacheValidationError,
    TextOrbitCache,
    read_orbit_file,
)
from satchecker_client.cache import REQUIRED_COLUMNS_BY_KIND
from satchecker_client import SatCheckerError as OrbitError

#: Historical name, from when every record was a TLE.
TLEError = OrbitError

# The TLE parser lives in satchecker_client.tle_parse so cache validation and
# element extraction exercise the *same* code; re-exported here under this
# module's historical names.
from satchecker_client.tle_parse import (  # noqa: E402
    parse_tle_elements,  # noqa: F401  re-export
    tle_epoch_jd,  # noqa: F401  re-export
    validate_tle_pair,  # noqa: F401  re-export
)
# Format dispatch. Nothing below this line asks whether a record is a TLE or an
# OMM: it asks for its epoch, its elements, or whether it is valid, and these
# three answer for either kind.
from satchecker_client.records import (  # noqa: E402
    CHECKSUM_STATUS_FIELD,
    CHECKSUM_UNVERIFIED_MISSING,
    KIND_FIELD,
    KIND_OMM,
    KIND_TLE,
    OMM_ELEMENT_COLUMNS,
    norad_id_of,
    record_elements,
    record_epoch_jd,
    record_kind,
    validate_record,  # noqa: F401  re-export
    validated_record,
)
from satchecker_client._time import jd_to_datetime  # noqa: E402
from tabsim.orbit_config import (  # noqa: E402,F401  re-exported for callers
    DEFAULT_CACHE_REUSE_MAX_AGE_DAYS,
    DEFAULT_REMOTE_MAX_AGE_DAYS,
    DEFAULT_SEARCH_CACHE_MAX_AGE_DAYS,
    OrbitConfig,
    TLEConfigurationError,
    orbit_cache_dir,
    validate_remote_ages,
    normalise_norad_ids,
    normalise_orbit_config,
    observation_epoch_jd,
    validate_age_days,
)
from tabsim.satchecker_names import norad_ids_from_names  # noqa: E402

# Name tabsim in the shared client's outgoing User-Agent. SatChecker is run as a
# courtesy to the community, so traffic from here should be attributable to
# tabsim rather than to the client library every application shares.
try:
    _TABSIM_VERSION = _metadata.version("tabsim")
except _metadata.PackageNotFoundError:  # a checkout on sys.path, not an install
    _TABSIM_VERSION = "unknown"
satchecker.set_client_identifier(
    f"tabsim/{_TABSIM_VERSION} (+https://github.com/chrisfinlay/tab-sim)"
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A TLE line-1 epoch is quantised to ~1e-8 day (8 decimal places of a day, ~0.9 ms),
# and the datetime<->JD round-trip adds only sub-microsecond-day noise (measured
# ~3.7e-9 day). This tolerance covers one epoch quantum plus that slack (~2.6 ms), so
# ``extra_orbit_max_age_days: 0`` accepts a record matching the observation to TLE
# precision while rejecting one several ms away — matching the documented semantics.
_AGE_TOL_DAYS = 3e-8

# Above this many remote records the per-satellite log lines are replaced by a
# grouped summary; set ``TABSIM_TLE_LOG_DETAIL=1`` to force the full listing.
_GROUPED_LOG_THRESHOLD = 12
_LOG_DETAIL_ENV = "TABSIM_TLE_LOG_DETAIL"

# Source labels used in logs, errors and provenance.
_SRC_EXTRA = "extra_orbit_dir"
_SRC_CACHE = "managed per-satellite cache"
# Qualified with the endpoint that answered, so a log or a coverage error says
# which of the two archives a record came from.
_SRC_SATCHECKER = "SatChecker"


# ---------------------------------------------------------------------------
# Resolution results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedOrbit:
    """One accepted record, with everything needed to explain *why* it was accepted."""

    norad_id: int
    record: dict
    source: str
    provider: Optional[str]
    epoch_jd: float
    offset_days: float          # signed: record epoch minus observation epoch

    @property
    def age_days(self) -> float:
        return abs(self.offset_days)

    @property
    def remote(self) -> bool:
        """True for records that came from the service or its managed cache."""
        return self.source != _SRC_EXTRA


@dataclass(frozen=True)
class RejectedOrbit:
    """The best (nearest-epoch) candidate that was found but not acceptable."""

    norad_id: int
    source: str
    provider: Optional[str]
    epoch_jd: Optional[float]
    offset_days: Optional[float]
    reason: str

    @property
    def age_days(self) -> Optional[float]:
        return None if self.offset_days is None else abs(self.offset_days)


@dataclass
class OrbitResolution:
    """The authoritative outcome of resolving one run's satellites."""

    requested: list[int]
    obs_epoch_jd: float
    remote_max_age_days: Optional[float]
    resolved: dict[int, ResolvedOrbit] = field(default_factory=dict)
    rejected: dict[int, RejectedOrbit] = field(default_factory=dict)
    #: Why the service could not answer for an ID, when it was asked and failed.
    #: Kept separate from ``rejected``: a rejection is a record we saw and judged,
    #: whereas this is the absence of an answer. Without it a coverage failure
    #: during an outage reads as "this satellite does not exist", which is a
    #: different problem with different remedies.
    service_errors: dict[int, Exception] = field(default_factory=dict)
    #: Why a *refresh* failed for an ID that stayed resolved anyway — from an
    #: acceptable cached record, or from the other archive. Kept apart from
    #: ``service_errors`` so a fatal-coverage decision never sees it: the run can
    #: continue, but it is not quite the run that was asked for, and the log is
    #: the only place that can say so.
    refresh_errors: dict[int, Exception] = field(default_factory=dict)
    #: True when acquisition was forbidden, so an unresolved ID means "not held
    #: locally", never "SatChecker has no record".
    offline: bool = False

    @property
    def missing(self) -> list[int]:
        """Requested IDs with no accepted record, in the order they were requested."""
        return [nid for nid in self.requested if nid not in self.resolved]

    @property
    def complete(self) -> bool:
        return not self.missing

    def norad_ids(self) -> list[int]:
        """Accepted IDs, in requested order — aligned with :meth:`records`."""
        return [nid for nid in self.requested if nid in self.resolved]

    def records(self) -> list[dict]:
        """Accepted raw records (identity + elements/lines + provenance), in requested order."""
        return [dict(self.resolved[nid].record) for nid in self.norad_ids()]

    def frame(self) -> pd.DataFrame:
        """Accepted records plus locally derived orbital elements, in requested order."""
        return _finalise_records(self.records())


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

def _add_parsed_elements(records: pd.DataFrame) -> pd.DataFrame:
    """Populate OMM-style element columns by deriving them from each row.

    A TLE row is parsed from its two lines; an OMM row's element columns are
    read directly, with the semi-major axis recomputed from the mean motion so
    both kinds agree. Columns are assigned (overwriting any element columns
    already present in a legacy Space-Track cache file) so the locally derived
    values always win and no duplicate columns are produced.
    """
    records = records.copy()
    parsed = pd.DataFrame(
        [record_elements(r) for _, r in records.iterrows()],
        index=records.index,
    )
    for col in parsed.columns:
        records[col] = parsed[col]
    return records


def _finalise_records(records: list[dict]) -> pd.DataFrame:
    """Turn accepted raw records into the element frame the simulator consumes."""
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["NORAD_CAT_ID"] = pd.to_numeric(frame["NORAD_CAT_ID"]).astype(int)
    return _add_parsed_elements(frame).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-ID source resolution
# ---------------------------------------------------------------------------

def read_extra_orbit_dir(extra_orbit_dir) -> pd.DataFrame:
    """Every orbit table in *extra_orbit_dir*, read strictly, concatenated.

    An explicit directory is *named* by the user, so "cannot be read" must never
    be indistinguishable from "has no record for this satellite": the latter falls
    through to the managed cache and the service, which would build the simulation
    from exactly the records the user said not to use, with nothing in the log to
    say the file they pointed at was never read. So every ``*.json`` here is read
    through :func:`satchecker_client.read_orbit_file` and anything it refuses —
    unreadable, malformed, not a table — raises :class:`OrbitError` naming the
    file. So does a non-empty table carrying neither record kind's columns: an
    explicitly supplied file that is not an orbit table at all is a mistake worth
    stopping for. An explicitly *empty* table is fine — that is a completed run
    stating it selected no satellites.

    Each row's identity is validated here too, while the file it came from is
    still known, and the returned frame's ``NORAD_CAT_ID`` is a checked integer.
    """
    directory = Path(extra_orbit_dir)
    frames = []
    for path in sorted(directory.glob("*.json")):
        try:
            frame = read_orbit_file(path)
        except (CacheValidationError, OSError, ValueError) as e:
            raise OrbitError(
                f"extra_orbit_dir file {path} could not be read as an orbit "
                f"table: {e}\n"
                "An explicitly supplied orbit file is not skipped: the run stops "
                "rather than silently falling back to the managed cache or "
                "SatChecker for the satellites this file was meant to supply. "
                "Fix or remove the file, or point extra_orbit_dir elsewhere."
            ) from e
        if not len(frame):
            continue
        if not any(
            all(column in frame.columns for column in required)
            for required in REQUIRED_COLUMNS_BY_KIND.values()
        ):
            raise OrbitError(
                f"extra_orbit_dir file {path} is not an orbit table: it carries "
                f"neither a TLE's {list(REQUIRED_COLUMNS_BY_KIND[KIND_TLE])} nor "
                f"an OMM's {list(REQUIRED_COLUMNS_BY_KIND[KIND_OMM])}. Move "
                "non-orbit JSON out of the directory extra_orbit_dir points at."
            )
        frames.append(_checked_extra_ids(frame, path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _checked_extra_ids(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """*frame* with every ``NORAD_CAT_ID`` validated, or an error naming *path*.

    Validation has to come before coercion, not instead of it. ``to_numeric``
    turns a malformed identity into a null and the row then simply vanishes from
    the wanted-ID filter — so an explicitly supplied record disappears with no
    diagnostic and the service answers for the satellite the file was meant to
    supply, which is exactly what an explicit directory is supposed to prevent.
    ``int()`` is worse: 25544.5 truncates to a *different* satellite's number.
    """
    ids = []
    for position, row in enumerate(frame.to_dict(orient="records")):
        try:
            ids.append(norad_id_of(row, f"row {position} of {path}"))
        except ValueError as e:
            raise OrbitError(
                f"extra_orbit_dir file {path} carries a row that is not filed "
                f"against a satellite: {e}. An explicitly supplied record is not "
                "skipped — the run stops rather than resolving that satellite "
                "from the managed cache or SatChecker instead. Fix the file, or "
                "point extra_orbit_dir elsewhere."
            ) from e
    frame = frame.copy()
    frame["NORAD_CAT_ID"] = ids
    return frame


def _select_from_extra_dir(
    extra_orbit_dir: str,
    wanted: set[int],
    obs_epoch_jd: float,
    max_age_days: Optional[float],
    allow_missing_checksum: bool = False,
) -> tuple[dict[int, ResolvedOrbit], dict[int, RejectedOrbit]]:
    """Resolve IDs from ``extra_orbit_dir`` with per-ID nearest + age policy.

    Returns the IDs whose nearest local record is within ``max_age_days`` of
    *obs_epoch_jd* (``None`` = unlimited), plus the rejected near-misses. The age
    is measured from the record's own epoch — a TLE's line-1 field, an OMM's
    ``EPOCH`` — never from the filename or the file modification time.

    Accepted records go through
    :func:`~satchecker_client.records.validated_record`, the same normalisation
    the remote route uses, so the checksum policy is one policy and a repaired
    line is repaired *on the record* — not merely tolerated by the validator while
    the defective line goes on to the propagator and the saved replay file.
    """
    resolved: dict[int, ResolvedOrbit] = {}
    rejected: dict[int, RejectedOrbit] = {}
    # Identities were validated against their own files by read_extra_orbit_dir,
    # which is the only place that still knows which file a row came from.
    records = read_extra_orbit_dir(extra_orbit_dir)
    if not len(records):
        return resolved, rejected
    records = records[records["NORAD_CAT_ID"].isin(wanted)]
    if not len(records):
        return resolved, rejected

    valid_rows = []
    for _, row in records.iterrows():
        nid = int(row["NORAD_CAT_ID"])
        try:
            record = validated_record(
                row, allow_missing_checksum=allow_missing_checksum
            )
            epoch_jd = record_epoch_jd(record)
        except (ValueError, TypeError) as e:
            print(f"  {nid}: invalid extra_orbit_dir record rejected — {e}")
            continue
        record["EPOCH_JD"] = epoch_jd
        valid_rows.append(record)
    if not valid_rows:
        return resolved, rejected
    records = pd.DataFrame(valid_rows)

    for nid, group in records.groupby("NORAD_CAT_ID"):
        best = group.loc[(group["EPOCH_JD"] - obs_epoch_jd).abs().idxmin()]
        epoch_jd = float(best["EPOCH_JD"])
        offset = epoch_jd - obs_epoch_jd
        record = {
            k: v
            for k, v in best.to_dict().items()
            if k != "EPOCH_JD" and not (v is None or (isinstance(v, float) and v != v))
        }
        if max_age_days is None or abs(offset) <= max_age_days + _AGE_TOL_DAYS:
            resolved[int(nid)] = ResolvedOrbit(
                norad_id=int(nid),
                record=record,
                source=_SRC_EXTRA,
                provider=None,
                epoch_jd=epoch_jd,
                offset_days=offset,
            )
        else:
            rejected[int(nid)] = RejectedOrbit(
                norad_id=int(nid),
                source=_SRC_EXTRA,
                provider=None,
                epoch_jd=epoch_jd,
                offset_days=offset,
                reason=f"extra_orbit_max_age_days={max_age_days}",
            )
            print(
                f"  {nid}: extra_orbit_dir record rejected — {abs(offset):.3f} d old "
                f"> extra_orbit_max_age_days={max_age_days}; trying managed cache"
            )
    if resolved:
        print(f"  {len(resolved)} record(s) taken from extra_orbit_dir")
    return resolved, rejected


def _select_from_records(
    records: pd.DataFrame,
    wanted: set[int],
    reference_epoch_jd: float,
) -> dict[int, dict]:
    """One record per wanted ID from a normalised record frame.

    The service may legitimately carry several distinct records for one NORAD
    ID. When it does, the one whose epoch is nearest *reference_epoch_jd* is
    chosen, so the selection is deterministic and independent of row order. The
    epoch comes from :func:`~satchecker_client.records.record_epoch_jd`, which is
    a row-wise call rather than a column map because a frame may mix kinds for
    one satellite around the archive handover.
    """
    resolved: dict[int, dict] = {}
    if not len(records):
        return resolved
    records = records.copy()
    records["NORAD_CAT_ID"] = pd.to_numeric(records["NORAD_CAT_ID"]).astype(int)
    match = records[records["NORAD_CAT_ID"].isin(wanted)]
    for nid, group in match.groupby("NORAD_CAT_ID"):
        if len(group) > 1:
            epochs = pd.Series(
                [record_epoch_jd(row) for _, row in group.iterrows()],
                index=group.index,
                dtype=float,
            )
            offsets = (epochs - reference_epoch_jd).abs()
            best = group.loc[offsets.idxmin()]
        else:
            best = group.iloc[0]
        resolved[int(nid)] = best.to_dict()
    return resolved


def _cached_candidates(
    cache: TextOrbitCache, wanted: set[int], obs_epoch_jd: float
) -> dict[int, dict]:
    """Select the nearest validated cached record independently for each ID."""
    selected: dict[int, dict] = {}
    for norad_id in sorted(wanted):
        records = cache.get(norad_id)
        if records.empty:
            continue
        candidates = _select_from_records(records, {norad_id}, obs_epoch_jd)
        if norad_id in candidates:
            selected[norad_id] = candidates[norad_id]
    return selected


def _accept_remote(
    candidates: dict[int, dict],
    source: str,
    obs_epoch_jd: float,
    max_age_days: Optional[float],
    resolved: dict[int, ResolvedOrbit],
    rejected: dict[int, RejectedOrbit],
    allow_missing_checksum: bool = False,
) -> set[int]:
    """Apply the remote age ceiling to *candidates*, updating accept/reject maps.

    The epoch comes from :func:`~satchecker_client.records.record_epoch_jd` and
    is compared against the actual mean observation epoch. For a TLE that means
    re-deriving it from line 1 — a provider's own ``epoch`` field is never
    trusted. An OMM record has no lines to re-derive from, so its ``EPOCH`` is
    parsed and range-checked instead; that is a real reduction in what can be
    caught here, and is why the plausibility window exists.

    A rejected candidate is remembered (nearest one wins) so the coverage error
    can report exactly how close the best available record was; it is never
    silently re-admitted once the remaining sources are exhausted.

    One rule covers both filling a gap and improving on what is already held: a
    candidate replaces the incumbent only when it is *strictly fresher*. That
    makes refresh safe by construction — a failed or staler response leaves the
    existing record untouched — and it also stops a later source from quietly
    downgrading an earlier one.

    Returns the IDs *candidates* answered with a record inside the age ceiling,
    whether it was accepted or discarded as no improvement on the incumbent. That
    is the question "did this source have something usable for the ID", which a
    lookup in *resolved* cannot answer once an earlier source has put an
    incumbent there.
    """
    within_ceiling: set[int] = set()
    for nid, record in candidates.items():
        provider = record.get("DATA_SOURCE") or None
        incumbent = resolved.get(nid)
        try:
            # The same normalisation the explicit-file route uses: canonical
            # lines, a checked identity, and an explicit checksum status carried
            # on the record for the rest of its life.
            record = validated_record(
                record, allow_missing_checksum=allow_missing_checksum
            )
            epoch_jd = record_epoch_jd(record)
        except (KeyError, ValueError, TypeError) as e:
            # Never displace a rejection that carries a real epoch and offset:
            # "the best candidate was 4.2 d away" tells the user what to do about
            # it, "unparseable" does not. The over-age branch below is symmetric
            # — it replaces an epoch-less rejection when it has a measurable one.
            if incumbent is None and nid not in rejected:
                rejected[nid] = RejectedOrbit(
                    nid, source, provider, None, None, f"unparseable epoch: {e}"
                )
            continue
        offset = epoch_jd - obs_epoch_jd
        if max_age_days is not None and abs(offset) > max_age_days + _AGE_TOL_DAYS:
            # Only worth reporting when nothing acceptable is held for this ID;
            # an over-age upgrade candidate is simply discarded.
            if incumbent is None:
                previous = rejected.get(nid)
                if (
                    previous is None
                    or previous.age_days is None
                    or abs(offset) < previous.age_days
                ):
                    rejected[nid] = RejectedOrbit(
                        norad_id=nid,
                        source=source,
                        provider=provider,
                        epoch_jd=epoch_jd,
                        offset_days=offset,
                        reason=f"remote_max_age_days={max_age_days:g}",
                    )
            continue
        within_ceiling.add(nid)
        if incumbent is not None and incumbent.age_days <= abs(offset):
            continue  # no improvement — keep what we have
        resolved[nid] = ResolvedOrbit(
            norad_id=nid,
            record=record,
            source=source,
            provider=provider,
            epoch_jd=epoch_jd,
            offset_days=offset,
        )
        rejected.pop(nid, None)
    return within_ceiling


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _detail_requested() -> bool:
    return os.environ.get(_LOG_DETAIL_ENV, "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


def _id_list(norad_ids) -> str:
    """IDs for a log line, truncated unless ``TABSIM_TLE_LOG_DETAIL`` is set.

    Truncated, never summarised away: a report about specific satellites has to
    name enough of them to act on, and the full list stays one environment
    variable away.
    """
    ids = sorted(int(nid) for nid in norad_ids)
    if _detail_requested() or len(ids) <= _GROUPED_LOG_THRESHOLD:
        return str(ids)
    return (
        f"{ids[:_GROUPED_LOG_THRESHOLD]} and {len(ids) - _GROUPED_LOG_THRESHOLD} "
        f"more (set {_LOG_DETAIL_ENV}=1 for the full list)"
    )


def _describe(entry: ResolvedOrbit) -> str:
    provider = f" [{entry.provider}]" if entry.provider else ""
    return (
        f"  {entry.norad_id}: {entry.source}{provider} "
        f"epoch {jd_to_datetime(entry.epoch_jd).isoformat()} UTC, "
        f"offset {entry.offset_days:+.4f} d, age {entry.age_days:.4f} d"
    )


def _report_remote_selection(resolution: OrbitResolution) -> None:
    """Log provider, epoch, signed offset and age for every accepted remote record.

    Small ID sets get one line each. Larger ones get a grouped summary — an
    all-Starlink run would otherwise bury the rest of the log — with the oldest
    records still named individually, and the full listing available on demand
    via ``TABSIM_TLE_LOG_DETAIL=1``.
    """
    remote = [e for e in resolution.resolved.values() if e.remote]
    if not remote:
        return
    limit = resolution.remote_max_age_days
    limit_text = "no limit" if limit is None else f"limit {limit:g} d"

    if len(remote) <= _GROUPED_LOG_THRESHOLD or _detail_requested():
        print(f"Remote orbit records   : {len(remote)} accepted ({limit_text})")
        for entry in sorted(remote, key=lambda e: e.norad_id):
            print(_describe(entry))
        return

    ages = np.array([e.age_days for e in remote])
    by_source: dict[str, int] = {}
    for entry in remote:
        key = entry.source + (f" [{entry.provider}]" if entry.provider else "")
        by_source[key] = by_source.get(key, 0) + 1
    print(f"Remote orbit records   : {len(remote)} accepted ({limit_text})")
    for key, count in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {count} from {key}")
    print(
        f"  age vs observation   : min {ages.min():.4f} d, "
        f"median {np.median(ages):.4f} d, max {ages.max():.4f} d"
    )
    oldest = sorted(remote, key=lambda e: -e.age_days)[:5]
    print(
        "  oldest               : "
        + ", ".join(f"{e.norad_id} ({e.offset_days:+.4f} d)" for e in oldest)
    )
    print(f"  (set {_LOG_DETAIL_ENV}=1 for a per-satellite listing)")


def _coverage_error(resolution: OrbitResolution, named: bool = False) -> OrbitError:
    """Build the actionable error raised when some configured ID has no record."""
    missing = resolution.missing
    lines = [
        f"Orbital records could not be resolved for {len(missing)} of "
        f"{len(resolution.requested)} configured satellites at observation epoch "
        f"{jd_to_datetime(resolution.obs_epoch_jd).isoformat()} UTC:"
    ]
    for nid in missing:
        bad = resolution.rejected.get(nid)
        failure = resolution.service_errors.get(nid)
        if bad is None:
            if failure is not None:
                lines.append(f"  {nid}: SatChecker could not answer — {failure}")
            elif resolution.offline:
                # Not "SatChecker has no record": nothing asked it. A cached
                # catalogue search can say a satellite exists without saying
                # anything about why no orbit record for it is held here.
                lines.append(
                    f"  {nid}: offline: true, and no acceptable record for it is "
                    f"held locally — neither in extra_orbit_dir nor in the managed "
                    f"per-satellite cache. This says nothing about whether "
                    f"SatChecker has one"
                )
            else:
                lines.append(
                    f"  {nid}: no record found in extra_orbit_dir, the managed "
                    f"per-satellite cache, or SatChecker"
                )
            continue
        if bad.age_days is None:
            lines.append(f"  {nid}: best candidate unusable — {bad.reason}")
        else:
            provider = f", provider {bad.provider}" if bad.provider else ""
            lines.append(
                f"  {nid}: best candidate is {bad.age_days:.3f} d from the "
                f"observation (epoch {jd_to_datetime(bad.epoch_jd).isoformat()} "
                f"UTC, from {bad.source}{provider}) — rejected by {bad.reason}"
            )
        if failure is not None:
            # Both matter: how close the best record was, *and* that a fresher
            # one could not be requested.
            lines.append(
                f"      SatChecker could not be asked for a closer one — {failure}"
            )
        elif resolution.offline:
            # The rejection above is the whole of the local evidence, and it is
            # about this machine, not about the catalogue: nothing was asked for
            # something nearer.
            lines.append(
                "      offline: true, so nothing was asked for a closer one. This "
                "is the local state being insufficient, not SatChecker lacking a "
                "nearer record"
            )

    limit = resolution.remote_max_age_days
    lines += [
        "",
        f"The remote age ceiling in force is remote_max_age_days="
        f"{'null (disabled)' if limit is None else f'{limit:g}'}. Remedies:",
    ]
    # A service failure is not the user's configuration being wrong, so lead with
    # the remedy that actually applies before the ones that change the model.
    if resolution.service_errors:
        retry_after = max(
            (
                seconds
                for seconds in (
                    getattr(error, "retry_after", None)
                    for error in resolution.service_errors.values()
                )
                if seconds is not None
            ),
            default=None,
        )
        when = (
            f" It asked for {retry_after:g} s before the next request."
            if retry_after is not None
            else ""
        )
        lines.append(
            f"  - SatChecker did not answer for "
            f"{len(resolution.service_errors)} of these.{when} Re-run when the "
            "service is reachable; nothing about the configuration need change"
        )
    if resolution.offline:
        lines.append(
            "  - run once without rfi_sources.tle_satellite.offline (or without "
            "--offline) so the records can be fetched and cached, or replay a "
            "previous run with --replay-orbit-dir <run>/input_data"
        )
    lines += [
        "  - put an acceptable record for these satellites in a directory and set "
        "rfi_sources.tle_satellite.extra_orbit_dir (or pass --extra-orbit-dir)",
        "  - deliberately change rfi_sources.tle_satellite.remote_max_age_days "
        "(null removes the ceiling entirely; this is an expert opt-out, not a "
        "default)",
        (
            "  - drop the names that select these satellites from sat_names"
            if named
            else "  - remove these NORAD IDs from norad_ids / norad_ids_path"
        ),
        "",
        "tabsim will not silently omit a configured satellite from the simulated "
        "RFI: the run stops here rather than writing an observation that is "
        "missing a source it was asked for.",
    ]
    return OrbitError("\n".join(lines))


# ---------------------------------------------------------------------------
# Service acquisition
# ---------------------------------------------------------------------------

def _fetch_from_service(
    to_fetch: list[int],
    obs_epoch_jd: float,
    remote_max_age: Optional[float],
    cache,
    resolution: OrbitResolution,
    max_workers: int,
    allow_missing_checksum: bool = False,
) -> None:
    """Ask SatChecker for *to_fetch*, falling back to its other archive.

    SatChecker keeps two archives that do not overlap — TLEs up to 2026-07-11,
    OMM from 2026-07-12 — so the observation epoch decides which endpoint to ask
    first. In the common case that is the whole story: one request per satellite,
    answered from the right archive.

    The fallback exists because neither endpoint reports "I have nothing that
    near". Ask ``get-nearest-omm`` for a 2021 epoch and it returns its earliest
    2026 record with nothing to flag the 4.6-year gap; ask ``get-nearest-tle``
    for a 2027 epoch and it returns the last TLE ever published. Both are
    rejected here by the age ceiling, which is exactly the signal that the record
    wanted lives in the *other* archive. Within a few days either side of the
    handover that is the normal case, not an exceptional one.

    An ID the first pass produced no in-ceiling record for therefore earns one
    more request. Note that this is *not* the same as an unresolved ID: an ID
    whose stale-but-acceptable cached record is already the incumbent stays
    resolved throughout, and would never reach the second archive if the loop
    filtered on ``resolution.resolved``. Those IDs are exactly the ones in
    ``to_fetch`` for whom the nearer record is the point of the request.

    What does **not** earn one is an outage: a transport failure, an HTTP 429, or
    a uniform wall of rejections means the service cannot serve us, and asking a
    down service a different question is still asking a down service. So the
    batch's ``outage`` stops the loop, while a per-ID response failure or an
    over-age record does not.

    Each pass merges its valid records into the cache before they are judged, so
    a record rejected on age is still available offline to a later run whose
    epoch it does suit.

    Every request opts in to strict response parsing. Without it, an HTTP-200
    error envelope — which is how SatChecker has been observed to report its own
    failures — normalises to an empty frame, so an outage becomes "this satellite
    has no record": the satellite is dropped and the log says nothing was
    available. ``strict_response`` is a keyword, and the batch layer calls
    ``fetch(norad_id, epoch_jd)``, so a partial carries it in.
    """
    remaining = list(to_fetch)
    endpoints = [
        (endpoint, functools.partial(fetch_nearest, strict_response=True))
        for endpoint, fetch_nearest in satchecker.nearest_endpoints_for(obs_epoch_jd)
    ]

    for attempt, (endpoint, fetch_nearest) in enumerate(endpoints):
        if not remaining:
            return
        if attempt:
            print(
                f"  {len(remaining)} ID(s) unresolved from {endpoints[0][0]}; "
                f"trying {endpoint} — the archives meet at "
                f"{jd_to_datetime(satchecker.HANDOVER_JD).date()} and an "
                "observation near that boundary can fall either side of it"
            )
        else:
            print(
                f"Fetching {len(remaining)} nearest record(s) from SatChecker "
                f"{endpoint} with up to {min(max_workers, len(remaining))} "
                f"concurrent requests"
            )

        batch = satchecker.fetch_nearest_batch(
            remaining,
            obs_epoch_jd,
            fetch_nearest=fetch_nearest,
            endpoint=endpoint,
            max_workers=max_workers,
            allow_missing_checksum=allow_missing_checksum,
        )
        served: set[int] = set()
        if not batch.records.empty:
            # The cache leaves out any TLE whose checksum digit is missing, by
            # its own rule: every application sharing these files reads them at
            # whatever version it is on, and an older client rejects a whole file
            # on meeting a line it cannot validate. So a permissively accepted
            # record is used by this run and saved with it, and the shared cache
            # is left exactly as it was.
            for norad_id, records in batch.records.groupby("NORAD_CAT_ID"):
                satchecker.store_or_warn(
                    lambda nid=int(norad_id), rows=records: cache.store(nid, rows),
                    cache.path(int(norad_id)),
                    f"orbit cache for NORAD {int(norad_id)}",
                )
            served = _accept_remote(
                _select_from_records(batch.records, set(remaining), obs_epoch_jd),
                f"{_SRC_SATCHECKER} ({endpoint})",
                obs_epoch_jd,
                remote_max_age,
                resolution.resolved,
                resolution.rejected,
                allow_missing_checksum=allow_missing_checksum,
            )

        # Keep why the service could not answer, for the IDs still without a
        # record. Discarding it makes an outage indistinguishable from a
        # satellite that genuinely has no record — the same error text, but
        # remedies that do not include the only one that works: try again. An ID
        # that *is* resolved — from an acceptable cached record — records the same
        # failure as a refresh failure instead, which cannot make coverage fatal
        # but still has to reach the log.
        for norad_id, error in batch.errors.items():
            if norad_id in resolution.resolved:
                resolution.refresh_errors[norad_id] = error
            else:
                resolution.service_errors[norad_id] = error

        if batch.outage is not None:
            return
        # Filter on what this archive actually served, not on what is resolved:
        # an ID riding a stale cached incumbent is resolved from the start, and
        # dropping it here would deny it the fallback archive it was fetched for.
        remaining = [nid for nid in remaining if nid not in served]
        # An ID the fallback resolved is no longer a service failure, whatever
        # the first pass recorded against it — but it is still a failed request,
        # so it is kept as one.
        for norad_id in list(resolution.service_errors):
            if norad_id in resolution.resolved:
                resolution.refresh_errors[norad_id] = resolution.service_errors.pop(
                    norad_id
                )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_orbits(
    norad_ids,
    obs_epoch_jd: float,
    extra_orbit_dir: Optional[str] = None,
    extra_orbit_max_age_days: Optional[float] = None,
    remote_max_age_days: Optional[float] = DEFAULT_REMOTE_MAX_AGE_DAYS,
    cache_reuse_max_age_days: Optional[float] = DEFAULT_CACHE_REUSE_MAX_AGE_DAYS,
    max_workers: int = satchecker.MAX_WORKERS,
    offline: bool = False,
    allow_missing_checksum: bool = False,
) -> OrbitResolution:
    """Resolve every requested NORAD ID at *obs_epoch_jd*, without raising on gaps.

    Returns the full :class:`OrbitResolution` — accepted records, rejected
    near-misses and the epochs everything was judged against. Callers decide what
    an incomplete result means; :func:`require_complete_coverage` is the policy
    tabsim simulations use.

    *offline* forbids every SatChecker request. It does **not** relax the age
    ceiling: offline is about what can be reached, not about what an acceptable
    record is, so a cached record outside ``remote_max_age_days`` is refused
    exactly as it would be online.

    *allow_missing_checksum* accepts TLE lines whose checksum digit the archive
    omitted — the same policy on every route a record can arrive by, because a
    default that rejects them remotely and accepts them from a file is the worst
    combination: the strictness is advertised and the way round it is to save the
    record once. Such records carry
    :data:`~satchecker_client.records.CHECKSUM_STATUS_FIELD` for the rest of their
    lives and never enter the shared cache.
    """
    requested = normalise_norad_ids(norad_ids)
    extra_max_age = validate_age_days(
        extra_orbit_max_age_days, "extra_orbit_max_age_days"
    )
    remote_max_age, reuse_max_age = validate_remote_ages(
        remote_max_age_days, cache_reuse_max_age_days
    )
    obs_epoch_jd = float(obs_epoch_jd)

    resolution = OrbitResolution(
        requested=requested,
        obs_epoch_jd=obs_epoch_jd,
        remote_max_age_days=remote_max_age,
        offline=bool(offline),
    )
    if not requested:
        return resolution

    print(f"Orbit requested epoch  : {jd_to_datetime(obs_epoch_jd).isoformat()} UTC")
    print(f"Satellites requested   : {len(requested)}")
    if offline:
        print(
            "Offline                : no SatChecker requests; only "
            "extra_orbit_dir and the managed cache, within the same age ceiling"
        )
    if allow_missing_checksum:
        print(
            "Checksum policy        : allow_missing_checksum: true — TLE lines "
            "with no checksum digit are accepted and carried as unverified"
        )

    wanted = set(requested)

    # 1. extra_orbit_dir (per-ID precedence + its own age policy)
    if extra_orbit_dir:
        print(
            f"Extra orbit dir        : {Path(extra_orbit_dir).resolve()} "
            f"(max age {'unlimited' if extra_max_age is None else f'{extra_max_age:g} d'})"
        )
        # A directory that is not there was almost certainly meant to be. Staying
        # silent turns a typo in a replay path into a run that quietly models
        # different satellites than the ones asked for, while the line above
        # implies the directory was searched.
        if not Path(extra_orbit_dir).is_dir():
            print(
                "  warning: this extra_orbit_dir does not exist (or is not a "
                "directory); no local records will be found there. Check the path "
                "if you meant to supply your own."
            )
        from_extra, extra_rejected = _select_from_extra_dir(
            extra_orbit_dir,
            wanted,
            obs_epoch_jd,
            extra_max_age,
            allow_missing_checksum=allow_missing_checksum,
        )
        resolution.resolved.update(from_extra)
        resolution.rejected.update(extra_rejected)

    remaining = wanted - set(resolution.resolved)

    # 2. Managed per-NORAD cache. A sufficiently close record is a cache hit. An
    # older but still acceptable record is retained as an offline fallback while
    # the service is asked whether it now has something closer.
    if remaining:
        cache = TextOrbitCache(orbit_cache_dir())
        cached = _cached_candidates(cache, remaining, obs_epoch_jd)

        # Every acceptable cached record becomes its ID's incumbent *before* any
        # request goes out. Two things depend on that: it is what gives
        # _accept_remote's strictly-fresher rule something to compare a response
        # against (otherwise a staler response would be accepted unopposed), and
        # it is the offline fallback if the request never comes back.
        _accept_remote(
            cached,
            _SRC_CACHE,
            obs_epoch_jd,
            remote_max_age,
            resolution.resolved,
            resolution.rejected,
            allow_missing_checksum=allow_missing_checksum,
        )

        # Whether to still ask the service is a *separate* question from whether
        # we already hold something usable. Only a record that is both within the
        # reuse threshold and actually accepted suppresses the request: without
        # the intersection, `cache_reuse_max_age_days: null` would make every
        # cached record a hit — including ones the hard ceiling then rejects —
        # and the ID would never be fetched at all.
        near_enough_to_reuse = {
            norad_id
            for norad_id, record in cached.items()
            if reuse_max_age is None
            or abs(record_epoch_jd(record) - obs_epoch_jd)
            <= reuse_max_age + _AGE_TOL_DAYS
        }
        to_fetch = sorted(remaining - (near_enough_to_reuse & set(resolution.resolved)))

        # 3. Exact-epoch nearest lookups for cache misses/stale cache candidates,
        # against whichever archive the observation epoch falls in — with the
        # other one as a fallback. See _fetch_from_service.
        if to_fetch and offline:
            print(
                f"  offline: {len(to_fetch)} ID(s) would have been refreshed from "
                f"SatChecker and were not — {_id_list(to_fetch)}"
            )
        elif to_fetch:
            _fetch_from_service(
                to_fetch,
                obs_epoch_jd,
                remote_max_age,
                cache,
                resolution,
                max_workers,
                allow_missing_checksum=allow_missing_checksum,
            )

            # A service failure — or a response no fresher than what we hold —
            # does not invalidate a cached record within the hard ceiling. Those
            # records are already the incumbents, so nothing has to be recovered
            # here; report the ones the request did not improve on.
            retained = [
                nid
                for nid in to_fetch
                if nid in resolution.resolved
                and resolution.resolved[nid].source == _SRC_CACHE
            ]
            if retained:
                print(
                    f"  SatChecker did not improve {len(retained)} ID(s); "
                    "continuing with acceptable cached records"
                )
            # A refresh that failed for an ID that stays resolved is not fatal,
            # but the run is then not quite the one that was asked for. Each ID is
            # named with the source that did answer for it: these are not all
            # cached incumbents — an ID whose first archive failed and whose second
            # succeeded is bookkept here too, and saying it continued from the
            # cache would describe a record the run never held.
            failed_refresh = [
                nid for nid in to_fetch if nid in resolution.refresh_errors
            ]
            if failed_refresh:
                shown = (
                    failed_refresh
                    if _detail_requested()
                    else failed_refresh[:_GROUPED_LOG_THRESHOLD]
                )
                print(
                    f"  warning: a SatChecker request failed for "
                    f"{len(failed_refresh)} ID(s) the run could still resolve; each "
                    f"is listed with the source it is continuing from: "
                    + "; ".join(
                        f"{nid} — {resolution.refresh_errors[nid]} "
                        f"(from {resolution.resolved[nid].source})"
                        for nid in shown
                    )
                    + (
                        ""
                        if len(shown) == len(failed_refresh)
                        else f"; and {len(failed_refresh) - len(shown)} more (set "
                        f"{_LOG_DETAIL_ENV}=1 for the full list)"
                    )
                )
        else:
            print(f"Cache hits             : {len(near_enough_to_reuse)} (no requests sent)")

    _report_unverified(resolution)
    _report_remote_selection(resolution)
    return resolution


def _report_unverified(resolution: OrbitResolution) -> None:
    """Name the accepted records nothing has verified, and what follows from that."""
    unverified = [
        nid
        for nid, entry in resolution.resolved.items()
        if entry.record.get(CHECKSUM_STATUS_FIELD) == CHECKSUM_UNVERIFIED_MISSING
    ]
    if not unverified:
        return
    print(
        f"  warning: Unverified TLE: missing checksum for {len(unverified)} "
        f"satellite(s) — {_id_list(unverified)}. allow_missing_checksum: true is "
        "in force, so lines the archive served without their checksum digit were "
        "accepted; nothing verifies their contents, and the status travels with "
        "each record for the rest of its life."
    )
    print(
        "  Not stored in the shared orbit cache; saved run records are required "
        "for offline replay of these satellites — replay this run with "
        "--replay-orbit-dir <run>/input_data --allow-missing-checksum."
    )


def require_complete_coverage(resolution: OrbitResolution) -> OrbitResolution:
    """Return *resolution* unchanged, or raise the actionable coverage error.

    The policy for satellites asked for **by number**: every one of them must end
    up with an accepted record. They were named individually, so one dropped for
    want of a record would be indistinguishable from one that simply never passed
    the target.
    """
    if resolution.requested and not resolution.complete:
        raise _coverage_error(resolution)
    return resolution


def report_named_coverage(
    resolution: OrbitResolution, log=print
) -> OrbitResolution:
    """Coverage policy for satellites selected by *name*, sharing the numbered one.

    A name is a catalogue *query*, so "nothing acceptable exists for this
    satellite near this observation" is an answer: the satellite is excluded and
    the reason — a genuinely empty reply, or a record rejected on age — is
    reported.

    "We could not find out" is not an answer. An unresolved request or response
    failure used to be warned about and dropped here, which turned a SatChecker
    outage into a complete-looking observation with no satellite RFI in it,
    indistinguishable from a correct simulation of a quiet sky. Running out of
    local state offline is the same kind of not-knowing. Both are fatal on this
    route exactly as they are for a numbered satellite, and through the same
    error, so the two routes cannot drift apart.

    Offline, *every* unresolved candidate is that second kind, an age-rejected
    local record included: a ten-day-old cached record is a fact about this
    machine, not an answer from the catalogue, and nothing asked for a nearer
    one. Excluding the satellite on that basis reported an exclusion the
    catalogue never made, and let a satellite-free simulation stand as the
    result. The age detail stays in the error — it is what says which limit to
    change, or which records to fetch.
    """
    if not resolution.requested:
        return resolution

    unanswered = [
        nid for nid in resolution.missing if nid in resolution.service_errors
    ]
    unknown_locally = [
        nid
        for nid in resolution.missing
        if resolution.offline and nid not in resolution.service_errors
    ]
    if unanswered or unknown_locally:
        raise _coverage_error(resolution, named=True)

    for nid in resolution.missing:
        bad = resolution.rejected.get(nid)
        if bad is None:
            log(
                f"  No acceptable record for named satellite {nid}: both "
                "SatChecker archives answered and neither holds one — excluded "
                "from the simulation"
            )
        elif bad.age_days is None:
            log(
                f"  No acceptable record for named satellite {nid}: the best "
                f"candidate was unusable ({bad.reason}) — excluded"
            )
        else:
            log(
                f"  No acceptable record for named satellite {nid}: the nearest is "
                f"{bad.age_days:.3f} d from the observation (epoch "
                f"{jd_to_datetime(bad.epoch_jd).isoformat()} UTC, from "
                f"{bad.source}), rejected by {bad.reason} — excluded"
            )
    if resolution.missing:
        log(
            f"  {len(resolution.missing)} named satellite(s) excluded for want of "
            f"an acceptable orbit record: {_id_list(resolution.missing)}"
        )
    return resolution


# ---------------------------------------------------------------------------
# Public orchestration
# ---------------------------------------------------------------------------

def resolve_names(
    names,
    obs_epoch_jd: float,
    *,
    search_cache_max_age_days: Optional[float] = DEFAULT_SEARCH_CACHE_MAX_AGE_DAYS,
    offline: bool = False,
    log=print,
) -> list[int]:
    """NORAD IDs for satellites named in the configuration, at *obs_epoch_jd*.

    Names are matched as substrings against an upper-case catalogue, reproducing
    what Space-Track's ``op.like`` did — see :mod:`tabsim.satchecker_names` for the
    exact semantics and their sharp edges, and for the search-cache and offline
    policy this forwards.

    *obs_epoch_jd* is not optional and not "now": which satellites existed is a
    question about the observation's date. A satellite that decayed between a 2019
    observation and today belongs in that simulation; one launched since does not.

    A name the catalogue genuinely does not know contributes no satellites and is
    reported — there is no satellite for a record to be missing for. A search that
    could not be *run* is a different thing and stops the run; see
    :func:`tabsim.satchecker_names.search_satellites`.
    """
    names = [str(name).strip() for name in (names or []) if str(name).strip()]
    if not names:
        return []
    log(f"Resolving {len(names)} satellite name(s) against the SatChecker catalogue")
    return norad_ids_from_names(
        names,
        obs_epoch_jd,
        search_cache_max_age_days=search_cache_max_age_days,
        offline=offline,
        log=log,
    )


def get_orbits_by_id(
    norad_ids,
    epoch_jd: float,
    extra_orbit_dir: Optional[str] = None,
    extra_orbit_max_age_days: Optional[float] = None,
    remote_max_age_days: Optional[float] = DEFAULT_REMOTE_MAX_AGE_DAYS,
    cache_reuse_max_age_days: Optional[float] = DEFAULT_CACHE_REUSE_MAX_AGE_DAYS,
    max_workers: int = satchecker.MAX_WORKERS,
    offline: bool = False,
    allow_missing_checksum: bool = False,
) -> pd.DataFrame:
    """Resolve orbital records for *norad_ids* at *epoch_jd*.

    Returns one row per requested ID, in the requested order, with OMM-style
    element columns derived locally. Raises :class:`OrbitError` unless every
    requested ID resolved.
    """
    return require_complete_coverage(
        resolve_orbits(
            norad_ids,
            epoch_jd,
            extra_orbit_dir=extra_orbit_dir,
            extra_orbit_max_age_days=extra_orbit_max_age_days,
            remote_max_age_days=remote_max_age_days,
            cache_reuse_max_age_days=cache_reuse_max_age_days,
            max_workers=max_workers,
            offline=offline,
            allow_missing_checksum=allow_missing_checksum,
        )
    ).frame()


# ---------------------------------------------------------------------------
# Reproducibility: persist the records a run actually used
# ---------------------------------------------------------------------------

#: What each kind needs written out to be readable back as itself. A TLE needs
#: only its lines — every element is encoded in them — plus the checksum status,
#: which is provenance nothing can re-derive: lines accepted without their
#: checksum digits stay unverified however well they now parse. An OMM needs its
#: epoch and its seven elements, because nothing else carries them, and gets no
#: checksum claim because the format has no checksum. Both keep whatever provider
#: and fetch provenance the record arrived with.
_REPLAY_COLUMNS = {
    KIND_TLE: (
        KIND_FIELD,
        "OBJECT_NAME",
        "TLE_LINE1",
        "TLE_LINE2",
        CHECKSUM_STATUS_FIELD,
        "DATA_SOURCE",
        "FETCHED_AT",
    ),
    KIND_OMM: (
        KIND_FIELD,
        "OBJECT_NAME",
        "OBJECT_ID",
        "EPOCH",
        *OMM_ELEMENT_COLUMNS,
        "DATA_SOURCE",
        "FETCHED_AT",
    ),
}

#: Of those, the ones without which the file cannot be read back as the record it
#: claims to be. A cell missing here is an invalid record; a cell missing anywhere
#: else in :data:`_REPLAY_COLUMNS` is one kind's column on the other kind's row,
#: which a mixed file has by construction.
_REPLAY_REQUIRED = {
    KIND_TLE: ("TLE_LINE1", "TLE_LINE2"),
    KIND_OMM: ("EPOCH", *OMM_ELEMENT_COLUMNS),
}

#: The two files a completed run writes into its ``input_data`` directory, and the
#: only two a frozen replay reads.
REPLAY_IDS_FILE = "norad_ids.yaml"
REPLAY_RECORDS_FILE = "used_orbits.json"


def _replay_record(norad_id: int, record: dict) -> dict:
    """One record validated, canonicalised and projected onto the replay columns.

    Validation comes first, because the projection cannot tell a missing cell from
    an invalid one: it skipped every null, which is right for the OMM columns a
    TLE row acquires in a mixed frame and wrong for an OMM's own mean motion. A
    dropped element writes a file that reads back as a record nothing can
    propagate — a replay that cannot replay — so a record that is not valid stops
    the save instead.

    *allow_missing_checksum* is deliberately not a parameter here. The checksum
    decision was made when the record was accepted, and
    :func:`~satchecker_client.records.validated_record` never *upgrades* a status,
    so validating permissively writes the provenance the record already carries
    and launders nothing: the replay applies the reader's own policy to it.

    Derived columns are dropped: ``EPOCH_JD`` and ``SEMIMAJOR_AXIS`` are computed
    from the others on every read, so writing them would create a second copy
    that a later edit could silently contradict.
    """
    try:
        record = validated_record(record, allow_missing_checksum=True)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"the record filed against NORAD {norad_id} cannot be written to a "
            f"replay file: {e}"
        ) from e
    kind = record_kind(record)
    out = {"NORAD_CAT_ID": int(norad_id), KIND_FIELD: kind}
    for column in _REPLAY_COLUMNS[kind]:
        value = record.get(column)
        if value is None or pd.isna(value):
            if column in _REPLAY_REQUIRED[kind]:
                raise ValueError(
                    f"the {kind.upper()} record for NORAD {norad_id} has no "
                    f"{column}, which a replay of it needs"
                )
            continue
        out[column] = value
    return out


def _json_scalar(value):
    """A JSON-encodable copy of one cell, preserving float64 exactly.

    NumPy scalars are unwrapped with ``.item()``, which yields the Python float
    that :func:`json.dump` then writes through ``repr`` — the shortest
    representation that reads back as the same double. Missing values become
    ``null`` so a mixed TLE/OMM file stays valid JSON, since ``json`` would
    otherwise emit a bare ``NaN``. An infinity is refused outright: ``json`` would
    write the non-standard ``Infinity`` literal, and no orbital element may be one
    anyway.
    """
    if value is None or (isinstance(value, float) and value != value):
        return None
    item = getattr(value, "item", None)
    if item is not None and getattr(value, "shape", ()) == ():
        value = item()
    if isinstance(value, float):
        if value != value:
            return None
        if value in (float("inf"), float("-inf")):
            raise ValueError(
                f"an orbit record carries {value!r}, which is not a number a "
                "record may hold and not valid JSON"
            )
    return value


def _own_norad_id(record) -> Optional[int]:
    """The record's own ``NORAD_CAT_ID``, checked, or ``None`` if it carries none.

    Validated rather than cast. ``int(25544.5)`` is 25544, so a lossy repair here
    lets a record whose identity disagrees with the ID it is filed against pass
    the alignment check and be written as the satellite it is not. Raises
    ``ValueError`` for an identity that is present and unusable; only a record
    with no identity at all returns ``None``.
    """
    value = record.get("NORAD_CAT_ID")
    if value is None or (isinstance(value, float) and value != value):
        return None
    return norad_id_of(record, "orbit record")


def save_orbits_for_reuse(path, norad_ids, records) -> str:
    """Write the orbit records a run used to *path*, as an explicit orbit table.

    The file is a column-oriented JSON carrying, per record, exactly what is
    needed to read it back as the same record — a TLE's two lines, or an OMM's
    epoch and elements — plus its checksum provenance and whatever the provider
    said. ``sim-vis --replay-orbit-dir <this directory>`` then reproduces this
    run's *selection* and its trajectories, independent of the shared cache, of
    what SatChecker serves by then, and of the remote age ceiling.

    ``RECORD_KIND`` is written explicitly. Inference exists for exports we did not
    write; for a file tabsim produced itself there is no reason to make a later
    reader guess.

    ``norad_ids`` and ``records`` are aligned sequences, as produced by
    :meth:`OrbitResolution.norad_ids` and :meth:`OrbitResolution.records`, and the
    alignment is checked rather than assumed: ``zip`` would truncate to the shorter
    of the two and write a file that reads back cleanly while describing different
    satellites than the run propagated.

    Always writes, and returns the path — an empty selection included. Writing
    nothing would make "this run modelled no satellites" and "this directory is
    not a replay" the same state on disk, so a frozen replay of a legitimately
    satellite-free run could not be told from a missing file.
    """
    # Not `norad_ids or []`: a dask/NumPy array raises on truth-testing rather
    # than answering "is it empty", which turns a satellite-free run into a crash.
    ids = [] if norad_ids is None else [int(nid) for nid in norad_ids]
    rows = [] if records is None else list(records)
    if len(ids) != len(rows):
        raise ValueError(
            f"norad_ids and records must be aligned sequences, got {len(ids)} ID(s) "
            f"and {len(rows)} record(s). Truncating to the shorter one would write "
            "a replay file describing different satellites than the run used."
        )
    projected = []
    for nid, record in zip(ids, rows):
        record = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        try:
            own = _own_norad_id(record)
        except ValueError as e:
            raise ValueError(
                f"the record filed against NORAD {nid} has an unusable identity: "
                f"{e}. Repairing it would save a record of whichever satellite the "
                "repair happened to name."
            ) from e
        if own is not None and own != nid:
            raise ValueError(
                f"record filed against NORAD {nid} carries NORAD_CAT_ID {own}; the "
                "IDs and the records a run saves must be aligned or the replay "
                "reproduces the wrong satellites"
            )
        projected.append(_replay_record(nid, record))

    # Written by hand rather than with DataFrame.to_json, which formats floats to
    # a fixed number of decimal places: the default 10 rounds an OMM element
    # outright, and even the maximum 15 writes 0.0066635 as 0.006663499999999999,
    # which is a *different* double. Either way a replayed trajectory silently
    # stops matching the run it claims to reproduce. json.dump writes a float
    # through repr, the shortest representation that reads back identically.
    # Column-oriented, one of the two shapes read_orbit_file reads, and the one
    # read_legacy_tle_records also accepts — so an older consumer can still read it.
    columns: list[str] = []
    for row in projected:
        columns += [column for column in row if column not in columns]
    payload = {
        column: {
            str(index): _json_scalar(row.get(column))
            for index, row in enumerate(projected)
        }
        for column in columns
    }

    path = str(path)
    with open(path, "w") as handle:
        json.dump(payload, handle)
    return path


# ---------------------------------------------------------------------------
# Frozen replay
# ---------------------------------------------------------------------------

def _read_replay_ids(path: Path) -> list[int]:
    """The saved final NORAD IDs, in saved order, with duplicates refused.

    Deliberately not :func:`~tabsim.orbit_config.read_norad_ids_file`, which
    de-duplicates: this file is a replay's record of what it propagated, so two
    lines naming one satellite means the file disagrees with itself and cannot be
    matched one-to-one against the saved records.
    """
    try:
        text = path.read_text()
    except OSError as e:
        raise OrbitError(
            f"frozen orbit replay could not read {path}: {e}. A replay reads only "
            f"{REPLAY_IDS_FILE} and {REPLAY_RECORDS_FILE} from the directory given, "
            "and has no other source to fall back to."
        ) from e

    out: list[int] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        nid = normalise_norad_ids([line.split()[0]], f"{path}:{lineno}")[0]
        if nid in out:
            raise OrbitError(
                f"{path} lists NORAD {nid} more than once; a frozen replay needs "
                "exactly one saved record per saved satellite, so the ID list has "
                "to be unique"
            )
        out.append(nid)
    return out


def load_replay_orbits(
    replay_orbit_dir, *, allow_missing_checksum: bool = False
) -> tuple[list[int], list[dict]]:
    """The saved NORAD IDs and records of a previous run, frozen exactly as saved.

    Reads only ``norad_ids.yaml`` and ``used_orbits.json`` from
    *replay_orbit_dir* and returns ``(norad_ids, records)`` — the saved IDs in
    saved order, and one aligned record each. An empty saved selection is
    ``([], [])``: a completed run that modelled no satellites.

    **There is no second source.** No name discovery, no managed cache, no
    request, no visibility reselection, no ``max_n_sat``, and no age-based
    replacement: a replay deliberately uses saved records however far their epochs
    are from the observation, because that is what makes them the same records.
    Anything short of exact therefore stops the run naming the file or the
    satellite — skipping a record, taking the first of two, or asking the cache
    would each silently change the orbital inputs of a run whose whole purpose is
    to keep them fixed.

    The configured checksum policy still applies, including to provenance: a
    record saved as ``unverified_missing_checksum`` needs
    *allow_missing_checksum* on every pass, whatever its lines carry now, so a
    permissive run cannot be laundered into a strict one by saving it.

    Exact reproduction of the previous run's trajectories and visibilities assumes
    the same observation, spectral inputs, random seeds and numerical environment;
    what this freezes is the orbital input.
    """
    directory = Path(replay_orbit_dir)
    ids_path = directory / REPLAY_IDS_FILE
    records_path = directory / REPLAY_RECORDS_FILE

    norad_ids = _read_replay_ids(ids_path)
    try:
        frame = read_orbit_file(records_path)
    except (CacheValidationError, OSError, ValueError) as e:
        raise OrbitError(
            f"frozen orbit replay could not read {records_path}: {e}. A replay has "
            "no alternative source by design, so the run stops here rather than "
            "resolving these satellites from somewhere else."
        ) from e

    rows_by_id: dict[int, list[dict]] = {}
    for position, row in enumerate(frame.to_dict(orient="records")):
        try:
            nid = _own_norad_id(row)
        except ValueError as e:
            raise OrbitError(
                f"row {position} of {records_path} is not filed against a "
                f"satellite: {e}"
            ) from e
        if nid is None:
            raise OrbitError(
                f"row {position} of {records_path} has no usable NORAD_CAT_ID, so "
                "it cannot be matched to a saved satellite"
            )
        rows_by_id.setdefault(nid, []).append(row)

    unlisted = sorted(set(rows_by_id) - set(norad_ids))
    if unlisted:
        raise OrbitError(
            f"{records_path} carries record(s) for NORAD {unlisted}, which "
            f"{ids_path} does not list. The two files describe one selection and "
            "have to agree about it."
        )

    records: list[dict] = []
    for nid in norad_ids:
        rows = rows_by_id.get(nid, [])
        if not rows:
            raise OrbitError(
                f"{ids_path} lists NORAD {nid} but {records_path} holds no record "
                "for it. A replay cannot fetch the missing one — that would make it "
                "a different run — so it stops here."
            )
        if len(rows) > 1:
            raise OrbitError(
                f"{records_path} holds {len(rows)} records for NORAD {nid}; a "
                "frozen replay needs exactly one, since choosing between them "
                "would be reselecting the orbital input it exists to freeze."
            )
        try:
            records.append(
                validated_record(
                    rows[0], allow_missing_checksum=allow_missing_checksum
                )
            )
        except (ValueError, TypeError) as e:
            raise OrbitError(
                f"the saved record for NORAD {nid} in {records_path} is not "
                f"acceptable under this run's policy: {e}. If the original run "
                "accepted TLE lines without their checksum digits, the replay has "
                "to say so too — set "
                "rfi_sources.tle_satellite.allow_missing_checksum: true, or pass "
                "--allow-missing-checksum."
            ) from e

    unverified = [
        int(record["NORAD_CAT_ID"])
        for record in records
        if record.get(CHECKSUM_STATUS_FIELD) == CHECKSUM_UNVERIFIED_MISSING
    ]
    if unverified:
        print(
            f"  Unverified TLE: missing checksum for {len(unverified)} replayed "
            f"satellite(s) — {_id_list(unverified)}. The original run accepted "
            "lines the archive served without their checksum digit, and the status "
            "travels with the record: nothing has verified these, then or now."
        )
    return norad_ids, records
