"""tabsim orbit-record orchestration and local orbital-element derivation.

Records are sourced from the IAU CPS SatChecker service via
:mod:`tabsim.satchecker` — no account or credentials are required. This module is
the tabsim adapter: it resolves each requested NORAD ID against an ordered set of
sources, applies the configurable age policies, drives the per-satellite cache,
and derives the orbital elements locally.

SatChecker serves two record formats from two non-overlapping archives — TLEs up
to 2026-07-11, OMM from 2026-07-12 — and a run near that boundary may need
either. Nothing in this module branches on which: every format question is
answered by :mod:`tabsim.satchecker.records`, so the policy below works off an
epoch and an opaque record.

Source precedence is resolved **independently per NORAD ID**:

  1. ``extra_orbit_dir`` — user-supplied local files, of either kind. The record
     whose epoch is closest to the observation epoch is chosen; it is accepted
     only if within ``extra_orbit_max_age_days`` (``None`` = unlimited). An
     accepted record wins outright — later sources are not consulted for that ID.
     This is *your* data: the remote service's age policy never applies to it, so
     exact replay of a previous run's ``used_orbits.json`` is always possible.
  2. Per-satellite cache — the cached record whose epoch is closest to the
     observation. If it is within ``cache_reuse_max_age_days``, it avoids a
     network request. An older record within the hard ceiling remains an offline
     fallback while tabsim asks SatChecker for something closer.
  3. SatChecker — exact-epoch lookups run with bounded concurrency for the
     remaining IDs, against the archive the observation epoch falls in, with the
     other archive as a fallback (see :func:`_fetch_from_service`). Valid
     responses are merged into the per-NORAD cache and may serve nearby
     observations later.

**Complete coverage.** Every explicitly requested NORAD ID must end up with an
accepted record, and :func:`require_complete_coverage` raises :class:`OrbitError`
naming each failure and its remedies if one does not. Satellites named rather
than numbered (``sat_names``) are a catalogue *query* — an unrecognised name
resolves to no IDs at all and is reported by :func:`resolve_names`, since there
is no satellite to be missing a record for.

Ported from ``tabascal/orbit.py`` (epfl-radio-astro/tabascal#92), less the
multi-process broadcast and the Measurement Set preflight, neither of which
tabsim has: simulation is single-process and builds its own time grid.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from platformdirs import user_cache_path

import numpy as np
import pandas as pd

from tabsim import satchecker
from tabsim.satchecker import (
    TextOrbitCache,
    read_legacy_tle_records,
)
from tabsim.satchecker import SatCheckerError as OrbitError

#: Historical name, from when every record was a TLE.
TLEError = OrbitError

# The TLE parser lives in tabsim.satchecker.tle_parse so cache validation and
# element extraction exercise the *same* code; re-exported here under this
# module's historical names.
from tabsim.satchecker.tle_parse import (  # noqa: E402
    parse_tle_elements,  # noqa: F401  re-export
    tle_epoch_jd,  # noqa: F401  re-export
    validate_tle_pair,  # noqa: F401  re-export
)
# Format dispatch. Nothing below this line asks whether a record is a TLE or an
# OMM: it asks for its epoch, its elements, or whether it is valid, and these
# three answer for either kind.
from tabsim.satchecker.records import (  # noqa: E402
    KIND_FIELD,
    KIND_OMM,
    KIND_TLE,
    OMM_ELEMENT_COLUMNS,
    record_elements,
    record_epoch_jd,
    record_kind,
    validate_record,
)
from tabsim.satchecker._time import jd_to_datetime  # noqa: E402
from tabsim.orbit_config import (  # noqa: E402,F401  re-exported for callers
    DEFAULT_CACHE_REUSE_MAX_AGE_DAYS,
    DEFAULT_REMOTE_MAX_AGE_DAYS,
    OrbitConfig,
    TLEConfigurationError,
    validate_remote_ages,
    normalise_norad_ids,
    normalise_orbit_config,
    observation_epoch_jd,
    validate_age_days,
)
from tabsim.satchecker_names import norad_ids_from_names  # noqa: E402


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
# Cache directory
# ---------------------------------------------------------------------------

def orbit_cache_dir() -> Path:
    """Return the managed orbit cache directory, creating it if possible.

    The directory is resolved in priority order:

    1. ``ORBIT_CACHE_DIR`` environment variable (if set).
    2. The platform user-cache directory (e.g. ``~/.cache/orbit-cache`` on Linux,
       ``~/Library/Caches/orbit-cache`` on macOS).

    A directory that cannot be created (read-only filesystem, no permission,
    quota) is *not* an error here: the path is returned regardless, reads then
    miss and writes are reported and skipped, so a run with a valid fetch is
    never lost to an unusable cache location.
    """
    p = Path(os.environ.get("ORBIT_CACHE_DIR") or user_cache_path("orbit-cache"))
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return p


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

def _select_from_extra_dir(
    extra_orbit_dir: str,
    wanted: set[int],
    obs_epoch_jd: float,
    max_age_days: Optional[float],
) -> tuple[dict[int, ResolvedOrbit], dict[int, RejectedOrbit]]:
    """Resolve IDs from ``extra_orbit_dir`` with per-ID nearest + age policy.

    Returns the IDs whose nearest local record is within ``max_age_days`` of
    *obs_epoch_jd* (``None`` = unlimited), plus the rejected near-misses. The age
    is measured from the record's own epoch — a TLE's line-1 field, an OMM's
    ``EPOCH`` — never from the filename or the file modification time.
    """
    resolved: dict[int, ResolvedOrbit] = {}
    rejected: dict[int, RejectedOrbit] = {}
    records = read_legacy_tle_records(extra_orbit_dir)
    if not len(records):
        return resolved, rejected
    records = records.copy()
    numeric_ids = pd.to_numeric(records["NORAD_CAT_ID"], errors="coerce")
    valid_ids = numeric_ids.notnull() & np.isfinite(numeric_ids)
    valid_ids &= numeric_ids == numeric_ids.round()
    records = records.loc[valid_ids].copy()
    records["NORAD_CAT_ID"] = numeric_ids.loc[valid_ids].astype(int)
    records = records[records["NORAD_CAT_ID"].isin(wanted)]
    if not len(records):
        return resolved, rejected

    valid_rows = []
    for _, row in records.iterrows():
        nid = int(row["NORAD_CAT_ID"])
        try:
            embedded_id = validate_record(row)
            if embedded_id != nid:
                raise ValueError(
                    f"record belongs to satellite {embedded_id}, not {nid}"
                )
            epoch_jd = record_epoch_jd(row)
        except (ValueError, TypeError) as e:
            print(f"  {nid}: invalid extra_orbit_dir record rejected — {e}")
            continue
        valid_row = row.copy()
        valid_row["EPOCH_JD"] = epoch_jd
        valid_rows.append(valid_row)
    if not valid_rows:
        return resolved, rejected
    records = pd.DataFrame(valid_rows)

    for nid, group in records.groupby("NORAD_CAT_ID"):
        best = group.loc[(group["EPOCH_JD"] - obs_epoch_jd).abs().idxmin()]
        epoch_jd = float(best["EPOCH_JD"])
        offset = epoch_jd - obs_epoch_jd
        record = {k: v for k, v in best.to_dict().items() if k != "EPOCH_JD"}
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
    epoch comes from :func:`~tabsim.satchecker.records.record_epoch_jd`, which is
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
) -> set[int]:
    """Apply the remote age ceiling to *candidates*, updating accept/reject maps.

    The epoch comes from :func:`~tabsim.satchecker.records.record_epoch_jd` and
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


def _coverage_error(resolution: OrbitResolution) -> OrbitError:
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
    lines += [
        "  - put an acceptable record for these satellites in a directory and set "
        "rfi_sources.tle_satellite.extra_orbit_dir (or pass --extra-orbit-dir)",
        "  - deliberately change rfi_sources.tle_satellite.remote_max_age_days "
        "(null removes the ceiling entirely; this is an expert opt-out, not a "
        "default)",
        "  - remove these NORAD IDs from norad_ids / norad_ids_path",
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
    """
    remaining = list(to_fetch)
    endpoints = satchecker.nearest_endpoints_for(obs_epoch_jd)

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
        )
        served: set[int] = set()
        if not batch.records.empty:
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
            )

        # Keep why the service could not answer, for the IDs still without a
        # record. Discarding it makes an outage indistinguishable from a
        # satellite that genuinely has no record — the same error text, but
        # remedies that do not include the only one that works: try again.
        for norad_id, error in batch.errors.items():
            if norad_id not in resolution.resolved:
                resolution.service_errors[norad_id] = error

        if batch.outage is not None:
            return
        # Filter on what this archive actually served, not on what is resolved:
        # an ID riding a stale cached incumbent is resolved from the start, and
        # dropping it here would deny it the fallback archive it was fetched for.
        remaining = [nid for nid in remaining if nid not in served]
        # An ID the fallback resolved is no longer a service failure, whatever
        # the first pass recorded against it.
        for norad_id in list(resolution.service_errors):
            if norad_id in resolution.resolved:
                del resolution.service_errors[norad_id]


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
) -> OrbitResolution:
    """Resolve every requested NORAD ID at *obs_epoch_jd*, without raising on gaps.

    Returns the full :class:`OrbitResolution` — accepted records, rejected
    near-misses and the epochs everything was judged against. Callers decide what
    an incomplete result means; :func:`require_complete_coverage` is the policy
    tabsim simulations use.
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
    )
    if not requested:
        return resolution

    print(f"Orbit requested epoch  : {jd_to_datetime(obs_epoch_jd).isoformat()} UTC")
    print(f"Satellites requested   : {len(requested)}")

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
            extra_orbit_dir, wanted, obs_epoch_jd, extra_max_age
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
        if to_fetch:
            _fetch_from_service(
                to_fetch,
                obs_epoch_jd,
                remote_max_age,
                cache,
                resolution,
                max_workers,
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
        else:
            print(f"Cache hits             : {len(near_enough_to_reuse)} (no requests sent)")

    _report_remote_selection(resolution)
    return resolution


def require_complete_coverage(resolution: OrbitResolution) -> OrbitResolution:
    """Return *resolution* unchanged, or raise the actionable coverage error."""
    if resolution.requested and not resolution.complete:
        raise _coverage_error(resolution)
    return resolution


# ---------------------------------------------------------------------------
# Public orchestration
# ---------------------------------------------------------------------------

#: A name search returning more than this many satellites is worth a warning
#: before the requests go out. SatChecker's nearest-record endpoints are
#: per-satellite, so a broad name — "starlink" matches over twenty thousand
#: objects — turns into that many requests and that many propagations.
_WIDE_NAME_MATCH_THRESHOLD = 500


def resolve_names(names, log=print) -> list[int]:
    """NORAD IDs for satellites named in the configuration.

    Names are matched as substrings, case-insensitively, reproducing what
    Space-Track's ``op.like`` did — see :mod:`tabsim.satchecker_names`.

    Unmatched names are reported and skipped rather than raised on: a name is a
    catalogue *query*, so "nothing called that" means there is no satellite for a
    record to be missing for, which is a different thing from a configured
    satellite whose record could not be obtained. Numbered satellites keep the
    strict coverage rule; see :func:`require_complete_coverage`.
    """
    names = [str(name).strip() for name in (names or []) if str(name).strip()]
    if not names:
        return []
    log(f"Resolving {len(names)} satellite name(s) against the SatChecker catalogue")
    norad_ids, unmatched = norad_ids_from_names(names, log=log)
    if unmatched:
        log(
            f"  warning: {len(unmatched)} name(s) matched nothing in orbit and "
            f"contribute no satellites: {unmatched}"
        )
    if len(norad_ids) > _WIDE_NAME_MATCH_THRESHOLD:
        log(
            f"  warning: these names match {len(norad_ids)} satellites. SatChecker "
            "serves one record per request, so this is that many requests on a "
            "cold cache, followed by that many visibility propagations. Narrow "
            "the names, or list the NORAD IDs you actually want."
        )
    return norad_ids


def get_orbits_by_id(
    norad_ids,
    epoch_jd: float,
    extra_orbit_dir: Optional[str] = None,
    extra_orbit_max_age_days: Optional[float] = None,
    remote_max_age_days: Optional[float] = DEFAULT_REMOTE_MAX_AGE_DAYS,
    cache_reuse_max_age_days: Optional[float] = DEFAULT_CACHE_REUSE_MAX_AGE_DAYS,
    max_workers: int = satchecker.MAX_WORKERS,
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
        )
    ).frame()


# ---------------------------------------------------------------------------
# Reproducibility: persist the records a run actually used
# ---------------------------------------------------------------------------

#: What each kind needs written out to be readable back as itself. A TLE needs
#: only its lines — every element is encoded in them. An OMM needs its epoch and
#: its seven elements, because nothing else carries them.
_REPLAY_COLUMNS = {
    KIND_TLE: (KIND_FIELD, "OBJECT_NAME", "TLE_LINE1", "TLE_LINE2"),
    KIND_OMM: (KIND_FIELD, "OBJECT_NAME", "OBJECT_ID", "EPOCH", *OMM_ELEMENT_COLUMNS),
}


def _replay_record(norad_id: int, record: dict) -> dict:
    """One record projected onto the columns a replay file needs.

    Derived columns are dropped: ``EPOCH_JD`` and ``SEMIMAJOR_AXIS`` are computed
    from the others on every read, so writing them would create a second copy
    that a later edit could silently contradict.
    """
    kind = record_kind(record)
    out = {"NORAD_CAT_ID": int(norad_id), KIND_FIELD: kind}
    for column in _REPLAY_COLUMNS[kind]:
        value = record.get(column)
        if value is not None and not pd.isna(value):
            out[column] = value
    return out


def _json_scalar(value):
    """A JSON-encodable copy of one cell, preserving float64 exactly.

    NumPy scalars are unwrapped with ``.item()``, which yields the Python float
    that :func:`json.dump` then writes through ``repr`` — the shortest
    representation that reads back as the same double. Missing values become
    ``null`` so a mixed TLE/OMM file stays valid JSON, since ``json`` would
    otherwise emit a bare ``NaN``.
    """
    if value is None or (isinstance(value, float) and value != value):
        return None
    item = getattr(value, "item", None)
    if item is not None and getattr(value, "shape", ()) == ():
        value = item()
    if isinstance(value, float) and value != value:
        return None
    return value


def save_orbits_for_reuse(path, norad_ids, records) -> Optional[str]:
    """Write the orbit records a run used to *path* in ``extra_orbit_dir`` format.

    The file is a pandas-oriented JSON carrying, per record, exactly what
    :func:`~tabsim.satchecker.cache.read_legacy_tle_records` needs to read it back
    as the same record — a TLE's two lines, or an OMM's epoch and elements. A
    later run reproduces this run's trajectories by pointing ``extra_orbit_dir``
    at the file's directory (with the default unlimited
    ``extra_orbit_max_age_days``), independent of the shared cache, of what
    SatChecker serves by then, and of the remote age ceiling.

    ``RECORD_KIND`` is written explicitly. Inference exists for exports we did not
    write; for a file tabsim produced itself there is no reason to make a later
    reader guess.

    ``norad_ids`` and ``records`` are aligned sequences, as produced by
    :meth:`OrbitResolution.norad_ids` and :meth:`OrbitResolution.records`.
    Returns the written path, or ``None`` when there is nothing to save.
    """
    # Not `norad_ids or []`: a dask/NumPy array raises on truth-testing rather
    # than answering "is it empty", which turns a satellite-free run into a crash.
    ids = [] if norad_ids is None else list(norad_ids)
    rows = [] if records is None else list(records)
    if not ids or not rows:
        return None
    projected = [_replay_record(nid, record) for nid, record in zip(ids, rows)]

    # Written by hand rather than with DataFrame.to_json, which formats floats to
    # a fixed number of decimal places: the default 10 rounds an OMM element
    # outright, and even the maximum 15 writes 0.0066635 as 0.006663499999999999,
    # which is a *different* double. Either way a replayed trajectory silently
    # stops matching the run it claims to reproduce. json.dump writes a float
    # through repr, the shortest representation that reads back identically.
    # Column-oriented, so pandas.read_json (and therefore read_legacy_tle_records)
    # reads it back unchanged.
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
