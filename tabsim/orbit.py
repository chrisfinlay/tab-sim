"""tabsim's orbit-record policy, executed by the shared SatChecker client.

Records come from the IAU CPS SatChecker service via :mod:`satchecker_client`,
which needs no credentials and executes the selection. What stays here is the
policy it has no view on — which sources, in what order, how old a record may be,
whether an incomplete resolution is fatal, and every sentence a tabsim user reads —
stated explicitly on one call to :func:`satchecker_client.resolve.resolve_orbits`
and converted back into the public shape (:class:`OrbitResolution`, :func:`_adapt`).
The service's two non-overlapping archives are
:mod:`satchecker_client.records`' business; nothing here branches on which.

A frozen replay is outside the ordering entirely: :func:`load_replay_orbits`
*replaces* the selection with a previous run's saved IDs and records, reading
nothing else, and :func:`tabsim.config.add_tle_satellite_sources` selects it
first. Otherwise sources are consulted **independently per NORAD ID**:

  1. ``extra_orbit_dir`` — the user's own files, read strictly (one that is not a
     readable orbit table stops the run naming itself rather than falling through
     to the service), accepted within ``extra_orbit_max_age_days`` (``None`` =
     unlimited) and then final for that ID. The remote age policy never applies
     to it, and this freezes nothing: it is ordinary per-ID precedence.
  2. The per-satellite cache — within ``cache_reuse_max_age_days`` a usable
     cached record avoids a request; older but within the hard ceiling
     ``remote_max_age_days`` it is the offline fallback while SatChecker is asked
     for something closer, and only a strictly fresher answer replaces it.
  3. SatChecker — one exact-epoch lookup per remaining satellite against the
     archive the epoch falls in, with the other as a fallback for an unusable
     answer rather than a second opinion: selection is *not* globally nearest
     across both. Valid responses are cached. ``offline`` skips this step without
     relaxing the age ceiling.

Each source offers the candidate nearest the observation epoch *that this run can
use*, so a record the checksum policy refuses is not a candidate: it neither
displaces a usable one nor provokes a request. ``allow_missing_checksum`` governs
only a line that arrived without its checksum digit — a present but wrong one is
always refused — identically on every route. Every accepted TLE record carries
:data:`CHECKSUM_STATUS_FIELD` for life (an OMM has no checksum to make a claim about), and the ones accepted *unverified* that
way stay out of the shared cache other applications read; verified records are
cached as usual.

Coverage fails closed. Every numbered satellite must end up with a record
(:func:`require_complete_coverage`); a *named* one may be excluded when the
catalogue answered — nothing there, or a record too old — but never when tabsim
could not find out (:func:`report_named_coverage`), and both routes raise the
same error for that.
"""

from __future__ import annotations

import importlib.metadata as _metadata
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import satchecker_client as satchecker
from satchecker_client import (
    REPLAY_IDS_FILE,  # noqa: F401  re-export: defined where the files are written
    REPLAY_RECORDS_FILE,  # noqa: F401  re-export
    OrbitInputError,
    TextOrbitCache,
    read_orbit_file,  # noqa: F401  re-export
)
from satchecker_client import SatCheckerError as OrbitError

#: Historical name, from when every record was a TLE.
TLEError = OrbitError

# The parser lives in the client so cache validation and element extraction
# exercise the *same* code; re-exported under this module's historical names.
from satchecker_client.tle_parse import (  # noqa: E402
    parse_tle_elements,  # noqa: F401  re-export
    tle_epoch_jd,  # noqa: F401  re-export
    validate_tle_pair,  # noqa: F401  re-export
)
# Nothing here asks whether a record is a TLE or an OMM: the checksum status is
# the one field this module reads off a record.
from satchecker_client.records import (  # noqa: E402
    CHECKSUM_STATUS_FIELD,
    CHECKSUM_UNVERIFIED_MISSING,
    validate_record,  # noqa: F401  re-export
)
# The client's stable vocabulary, in which the evidence below is read; the words
# a tabsim user sees are this module's.
from satchecker_client.resolve import (  # noqa: E402
    ATTEMPT_NOT_SENT,
    EVENT_BATCH_STARTED,
    EVENT_CANDIDATE_REJECTED,
    EVENT_ENDPOINT_FALLBACK,
    EVENT_OUTAGE,
    EVENT_REFRESH_REQUIRED,
    EVENT_REFRESH_SKIPPED,
    EVENT_SOURCE_SELECTED,
    GROUP_EXTRA,
    GROUP_REMOTE,
    INPUT_CHECKSUM_POLICY,
    REASON_OVER_AGE,
    REPLACEMENT_STRICTLY_FRESHER,
    SOURCE_CACHE,
    SOURCE_EXTRA,
    UNAVAILABLE_ABSENT,
)
from satchecker_client import jd_to_datetime  # noqa: E402
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

# Name tabsim in the shared client's User-Agent: SatChecker is run as a courtesy,
# so traffic should be attributable to the application, not to the library.
try:
    _TABSIM_VERSION = _metadata.version("tabsim")
except _metadata.PackageNotFoundError:  # a checkout on sys.path, not an install
    _TABSIM_VERSION = "unknown"
satchecker.set_client_identifier(
    f"tabsim/{_TABSIM_VERSION} (+https://github.com/chrisfinlay/tab-sim)"
)


# Constants

# Above this many remote records the per-satellite log lines are replaced by a
# grouped summary; set ``TABSIM_TLE_LOG_DETAIL=1`` to force the full listing.
_GROUPED_LOG_THRESHOLD = 12
_LOG_DETAIL_ENV = "TABSIM_TLE_LOG_DETAIL"

# Source labels for logs, errors and provenance: the client files a record under
# a stable code, a user is told the name of the thing they configured. The
# service label is qualified with the archive that answered, because the two
# behave differently near the handover.
_SRC_EXTRA = "extra_orbit_dir"
_SRC_CACHE = "managed per-satellite cache"
_SRC_SATCHECKER = "SatChecker"


def _source_label(source: str, endpoint: Optional[str]) -> str:
    """The client's source code, in the words tabsim's messages are written in."""
    if source == SOURCE_EXTRA:
        return _SRC_EXTRA
    if source == SOURCE_CACHE:
        return _SRC_CACHE
    return f"{_SRC_SATCHECKER} ({endpoint})" if endpoint else _SRC_SATCHECKER


# Resolution results

@dataclass(frozen=True)
class ResolvedOrbit:
    """One accepted record, with everything needed to explain *why* it was accepted.

    A display adapter over the client's accepted entry, with tabsim's source
    label and tabsim's field order. Callers build these positionally, so added
    metadata is keyword-only and the positions stay the six they have always been.
    """

    norad_id: int
    record: dict
    source: str
    provider: Optional[str]
    epoch_jd: float
    offset_days: float          # signed: record epoch minus observation epoch
    endpoint: Optional[str] = field(default=None, kw_only=True)

    @property
    def age_days(self) -> float:
        return abs(self.offset_days)

    @property
    def remote(self) -> bool:
        """True for records that came from the service or its managed cache."""
        return self.source != _SRC_EXTRA


@dataclass(frozen=True)
class RejectedOrbit:
    """The best (nearest-epoch) candidate that was found but not acceptable.

    ``reason`` is the sentence a user acts on; ``reason_code``, ``ceiling_days``
    and ``limit_name`` are the same rejection as structure. Keyword-only, so the
    positions stay the six this type has always had.
    """

    norad_id: int
    source: str
    provider: Optional[str]
    epoch_jd: Optional[float]
    offset_days: Optional[float]
    reason: str
    reason_code: Optional[str] = field(default=None, kw_only=True)
    ceiling_days: Optional[float] = field(default=None, kw_only=True)
    limit_name: Optional[str] = field(default=None, kw_only=True)
    endpoint: Optional[str] = field(default=None, kw_only=True)

    @property
    def age_days(self) -> Optional[float]:
        return None if self.offset_days is None else abs(self.offset_days)


class OrbitResolution(satchecker.OrbitResolution):
    """The authoritative outcome of resolving one run's satellites.

    The client's result with tabsim's entries in it: ``missing``, ``complete``,
    ``norad_ids()``, ``records()`` and ``frame()`` are inherited, so element
    derivation and the requested-order contract have one implementation, while
    the constructor is stated explicitly because callers build these positionally.

    ``requested`` is the order the *run* asked in. The client is asked in sorted
    order — the acquisition order, and so which requests are reached before an
    outage stops acquisition — and everything comes back in the run's.
    """

    def __init__(
        self,
        requested,
        obs_epoch_jd,
        remote_max_age_days=None,
        resolved=None,
        rejected=None,
        service_errors=None,
        refresh_errors=None,
        offline=False,
        *,
        cache_reuse_max_age_days=None,
        extra_orbit_max_age_days=None,
        unavailable=None,
        attempts=None,
        events=None,
    ):
        super().__init__(
            requested=list(requested),
            obs_epoch_jd=obs_epoch_jd,
            remote_max_age_days=remote_max_age_days,
            cache_reuse_max_age_days=cache_reuse_max_age_days,
            extra_orbit_max_age_days=extra_orbit_max_age_days,
            offline=offline,
            resolved=dict(resolved or {}),
            rejected=dict(rejected or {}),
            service_errors=dict(service_errors or {}),
            refresh_errors=dict(refresh_errors or {}),
            unavailable=dict(unavailable or {}),
            attempts=dict(attempts or {}),
            events=list(events or []),
        )


def _rejection_reason(entry) -> str:
    """The sentence tabsim has always shown for a refused candidate.

    An age rejection names the governing limit, the one thing that says what to
    change. An unusable candidate has no measurement, so the diagnostic is the
    exception the client attached to *this* rejection — never a
    ``candidate_rejected`` event: only the first rejection per satellite
    survives, so the last event may be another candidate's.
    """
    if entry.reason_code == REASON_OVER_AGE and entry.limit_name:
        ceiling = entry.ceiling_days
        if ceiling is None:
            return f"{entry.limit_name}={ceiling}"
        if entry.limit_name == "extra_orbit_max_age_days":
            return f"{entry.limit_name}={ceiling}"
        return f"{entry.limit_name}={ceiling:g}"
    error = entry.error
    return f"invalid record: {error}" if error is not None else "invalid record"


def _adapt(result, requested: list[int]) -> OrbitResolution:
    """The client's result as tabsim's, in the run's own request order.

    One conversion boundary, one direction: the client's own result is left as it
    was, since it is where the coverage classifier reads its evidence.
    """
    resolved = {
        int(norad_id): ResolvedOrbit(
            norad_id=int(norad_id),
            record=entry.record,
            source=_source_label(entry.source, entry.endpoint),
            # An explicit file is the user's own data: no provider is shown for
            # it, whatever DATA_SOURCE the file happens to carry.
            provider=None if entry.source == SOURCE_EXTRA else entry.provider,
            epoch_jd=entry.epoch_jd,
            offset_days=entry.offset_days,
            endpoint=entry.endpoint,
        )
        for norad_id, entry in result.resolved.items()
    }
    rejected = {
        int(norad_id): RejectedOrbit(
            norad_id=int(norad_id),
            source=_source_label(entry.source, entry.endpoint),
            provider=None if entry.source == SOURCE_EXTRA else entry.provider,
            epoch_jd=entry.epoch_jd,
            offset_days=entry.offset_days,
            reason=_rejection_reason(entry),
            reason_code=entry.reason_code,
            ceiling_days=entry.ceiling_days,
            limit_name=entry.limit_name,
            endpoint=entry.endpoint,
        )
        for norad_id, entry in result.rejected.items()
    }
    return OrbitResolution(
        requested,
        result.obs_epoch_jd,
        result.remote_max_age_days,
        resolved,
        rejected,
        result.service_errors,
        result.refresh_errors,
        result.offline,
        cache_reuse_max_age_days=result.cache_reuse_max_age_days,
        extra_orbit_max_age_days=result.extra_orbit_max_age_days,
        unavailable=result.unavailable,
        attempts=result.attempts,
        events=result.events,
    )


# Explicit files

def read_extra_orbit_dir(extra_orbit_dir) -> pd.DataFrame:
    """Every orbit table in *extra_orbit_dir*, read strictly, concatenated.

    An explicit directory is *named* by the user, so "cannot be read" must never
    be indistinguishable from "has no record for this satellite": the latter
    falls through to the cache and the service, building the run from exactly the
    records the user said not to use.
    """
    try:
        return satchecker.read_extra_orbit_dir(extra_orbit_dir)
    except OrbitInputError as e:
        raise _extra_dir_error(e) from e


def _extra_dir_error(error: OrbitInputError) -> OrbitError:
    """A reader failure as a tabsim run failure, keeping what the client knew."""
    where = "extra_orbit_dir"
    if error.path is not None:
        where += f" file {error.path}"
    if error.row is not None:
        where += f", row {error.row}"
    if error.norad_id is not None:
        where += f" (NORAD {error.norad_id})"
    return OrbitError(
        f"{where}: {error}\n"
        "An explicitly supplied orbit file is not skipped: the run stops rather "
        "than silently falling back to the managed cache or SatChecker for the "
        "satellites this file was meant to supply. Fix or remove the file, or "
        "point extra_orbit_dir elsewhere."
    )


class _LazyOrbitCache:
    """The managed cache, opened only if the resolution actually reaches it.

    Where a cache lives is the application's decision, so the client takes an
    explicit one; but constructing it creates the directory, which a run resolved
    entirely from ``extra_orbit_dir`` has no business doing. Every call forwards
    to the one cache built on first use; no policy lives here.
    """

    def __init__(self):
        self._cache = None

    def _open(self):
        if self._cache is None:
            self._cache = TextOrbitCache(orbit_cache_dir())
        return self._cache

    def get(self, *args, **kwargs):
        return self._open().get(*args, **kwargs)

    def store(self, *args, **kwargs):
        return self._open().store(*args, **kwargs)

    def path(self, *args, **kwargs):
        return self._open().path(*args, **kwargs)


# Logging

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
    name enough of them to act on.
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


class _Reporter:
    """tabsim's account of a resolution, written from the client's events.

    The client states what happened in stable event codes; the sentences a tabsim
    user reads are here. Nothing infers state from the client's prose, and the
    summaries that need the outcome rather than the commentary wait for
    :meth:`finish`.
    """

    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        #: Accepted from the explicit directory, counted as the events arrive and
        #: reported once the resolution moves on to the cache and the service.
        self.from_extra = 0
        self.extra_reported = False
        #: The IDs a request was going to be made for, which is not the same set
        #: as the IDs that ended up with a fetched record.
        self.refresh_required: list[int] = []
        #: Endpoints already announced as a fallback, so the first-batch heading
        #: is not printed a second time for them.
        self.announced: set = set()

    def __call__(self, event) -> None:
        if event.source == SOURCE_EXTRA and event.code in (
            EVENT_SOURCE_SELECTED,
            EVENT_CANDIDATE_REJECTED,
        ):
            self._from_extra(event)
            return
        self._report_extra()
        if event.code == EVENT_REFRESH_SKIPPED:
            print(
                f"Cache hits             : {len(event.norad_ids)} "
                "(no requests sent)"
            )
        elif event.code == EVENT_REFRESH_REQUIRED:
            self.refresh_required += [int(nid) for nid in event.norad_ids]
            if event.details.get("offline"):
                print(
                    f"  offline: {len(event.norad_ids)} ID(s) would have been "
                    f"refreshed from SatChecker and were not — "
                    f"{_id_list(event.norad_ids)}"
                )
        elif event.code == EVENT_ENDPOINT_FALLBACK:
            self.announced.add(event.endpoint)
            print(
                f"  {len(event.norad_ids)} ID(s) unresolved from "
                f"{event.details.get('after')}; trying {event.endpoint} — the "
                f"archives meet at "
                f"{jd_to_datetime(satchecker.HANDOVER_JD).date()} and an "
                "observation near that boundary can fall either side of it"
            )
        elif event.code == EVENT_BATCH_STARTED:
            if event.endpoint in self.announced:
                return  # the fallback line above already said what is happening
            asked = len(event.norad_ids)
            print(
                f"Fetching {asked} nearest record(s) from SatChecker "
                f"{event.endpoint} with up to {min(self.max_workers, asked)} "
                "concurrent requests"
            )

    def _from_extra(self, event) -> None:
        if event.code == EVENT_SOURCE_SELECTED:
            self.from_extra += 1
        elif event.details.get("reason_code") == REASON_OVER_AGE:
            offset = event.details.get("offset_days")
            print(
                f"  {event.norad_ids[0]}: extra_orbit_dir record rejected — "
                f"{abs(offset):.3f} d old > extra_orbit_max_age_days="
                f"{event.details.get('ceiling_days')}; trying managed cache"
            )
        # An unusable explicit record is already reported by the client's own
        # per-candidate log line, which names the satellite and the defect.

    def _report_extra(self) -> None:
        if self.extra_reported:
            return
        self.extra_reported = True
        if self.from_extra:
            print(f"  {self.from_extra} record(s) taken from extra_orbit_dir")

    def finish(self, resolution: OrbitResolution) -> None:
        """What the run ended up with, which no single event can say."""
        self._report_extra()
        # A result, so it needs a request: offline, the same records are retained
        # because nothing was asked, which the skipped-refresh line already says.
        retained = [
            nid
            for nid in self.refresh_required
            if not resolution.offline
            and nid in resolution.resolved
            and resolution.resolved[nid].source == _SRC_CACHE
        ]
        if retained:
            # A failure, or an answer no fresher than what we hold, does not
            # invalidate a cached record within the hard ceiling.
            print(
                f"  SatChecker did not improve {len(retained)} ID(s); "
                "continuing with acceptable cached records"
            )
        # A refresh that failed for an ID that stays resolved is not fatal, but
        # the run is then not quite the one asked for. Each ID is named with the
        # source that did answer for it: an ID whose first archive failed and
        # whose second succeeded is bookkept here too, and "from the cache" would
        # describe a record the run never held.
        failed = [
            nid
            for nid in resolution.requested
            if nid in resolution.refresh_errors and nid in resolution.resolved
        ]
        if not failed:
            return
        shown = failed if _detail_requested() else failed[:_GROUPED_LOG_THRESHOLD]
        print(
            f"  warning: a SatChecker request failed for {len(failed)} ID(s) the "
            "run could still resolve; each is listed with the source it is "
            "continuing from: "
            + "; ".join(
                f"{nid} — {resolution.refresh_errors[nid]} "
                f"(from {resolution.resolved[nid].source})"
                for nid in shown
            )
            + (
                ""
                if len(shown) == len(failed)
                else f"; and {len(failed) - len(shown)} more (set "
                f"{_LOG_DETAIL_ENV}=1 for the full list)"
            )
        )


def _report_remote_selection(resolution: OrbitResolution) -> None:
    """Log provider, epoch, signed offset and age for every accepted remote record.

    Small ID sets get one line each; larger ones a grouped summary that still
    names the oldest records, with the full listing on ``TABSIM_TLE_LOG_DETAIL=1``.
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


# Coverage

#: Why one requested satellite has no record. The first four are tabsim not
#: *knowing*, fatal on both selection routes; the last two are answers the
#: catalogue gave, on which a named satellite may be excluded.
_GAP_FAILED = "failed"        # the service was asked and did not answer usably
_GAP_BLOCKED = "blocked"      # an archive the fallback needed was never asked
_GAP_OFFLINE = "offline"      # nothing was asked, by configuration
_GAP_UNKNOWN = "unknown"      # incomplete evidence with nothing to explain it
_GAP_ABSENT = "absent"        # every archive answered, and none holds one
_GAP_OVER_AGE = "over_age"    # a record was found, measured, and refused on age

_ANSWERED = (_GAP_ABSENT, _GAP_OVER_AGE)


@dataclass(frozen=True)
class _Gap:
    """The application's reading of the client's evidence for one unresolved ID."""

    norad_id: int
    kind: str
    error: Optional[BaseException] = None
    endpoint: Optional[str] = None
    pending: tuple = ()

    @property
    def fatal(self) -> bool:
        return self.kind not in _ANSWERED


def _outage_for(events, norad_id: int):
    """The outage event covering *norad_id*, if the client recorded one."""
    for event in reversed(events):
        if event.code == EVENT_OUTAGE and norad_id in event.norad_ids:
            return event
    return None


def _classify_gap(resolution: OrbitResolution, norad_id: int) -> _Gap:
    """Why *norad_id* has no record — and whether that is an answer or a gap.

    Fails closed. Only two states are the catalogue telling us something: every
    archive the fallback needed answered and none had a record, or one was found,
    measured against a ceiling and refused by it with the acquisition complete.
    An endpoint that was never asked is not one that had nothing — the client
    files an ID whose fallback an outage prevented under neither
    ``service_errors`` nor ``unavailable``.
    """
    error = resolution.service_errors.get(norad_id)
    if error is not None:
        return _Gap(norad_id, _GAP_FAILED, error=error)
    if resolution.offline:
        # Including an over-age local record: that is a fact about this machine,
        # not an answer from the catalogue.
        return _Gap(norad_id, _GAP_OFFLINE)
    attempts = resolution.attempts.get(norad_id, ())
    pending = tuple(
        attempt.endpoint for attempt in attempts if attempt.status == ATTEMPT_NOT_SENT
    )
    if pending:
        event = _outage_for(resolution.events, norad_id)
        return _Gap(
            norad_id,
            _GAP_BLOCKED,
            error=None if event is None else event.error,
            endpoint=None if event is None else event.endpoint,
            pending=pending,
        )
    if resolution.unavailable.get(norad_id) == UNAVAILABLE_ABSENT:
        return _Gap(norad_id, _GAP_ABSENT)
    rejection = resolution.rejected.get(norad_id)
    if attempts and rejection is not None and rejection.reason_code == REASON_OVER_AGE:
        return _Gap(norad_id, _GAP_OVER_AGE)
    return _Gap(norad_id, _GAP_UNKNOWN)


def _gap_sentence(gap: _Gap) -> str:
    """What to say about a satellite with no near-miss to report."""
    if gap.kind == _GAP_FAILED:
        return f"SatChecker could not answer — {gap.error}"
    if gap.kind == _GAP_BLOCKED:
        return f"SatChecker could not answer — {_blocked_detail(gap)}"
    if gap.kind == _GAP_OFFLINE:
        # Not "SatChecker has no record": nothing asked it.
        return (
            "offline: true, and no acceptable record for it is held locally — "
            "neither in extra_orbit_dir nor in the managed per-satellite cache. "
            "This says nothing about whether SatChecker has one"
        )
    if gap.kind == _GAP_ABSENT:
        return (
            "no record found in extra_orbit_dir, the managed per-satellite "
            "cache, or SatChecker"
        )
    return (
        "no acceptable record, and nothing establishes that SatChecker has none "
        "for it"
    )


def _blocked_detail(gap: _Gap) -> str:
    """The outage that stopped acquisition, named with the archive it stopped at."""
    pending = ", ".join(gap.pending)
    if gap.error is None:
        return (
            f"{pending} was never asked and nothing recorded why, so whether "
            "SatChecker has a record for this satellite is unknown"
        )
    return (
        f"acquisition stopped at {gap.endpoint} before {pending} could be asked "
        f"— {gap.error}"
    )


def _closer_sentence(gap: _Gap) -> Optional[str]:
    """What stopped a *nearer* record being obtained, beside a near-miss."""
    if gap.kind == _GAP_FAILED:
        return f"SatChecker could not be asked for a closer one — {gap.error}"
    if gap.kind == _GAP_BLOCKED:
        return (
            "SatChecker could not answer for a closer record — "
            f"{_blocked_detail(gap)}"
        )
    if gap.kind == _GAP_OFFLINE:
        return (
            "offline: true, so nothing was asked for a closer one. This is the "
            "local state being insufficient, not SatChecker lacking a nearer "
            "record"
        )
    return None


def _coverage_error(resolution: OrbitResolution, named: bool = False) -> OrbitError:
    """Build the actionable error raised when some configured ID has no record."""
    gaps = [_classify_gap(resolution, nid) for nid in resolution.missing]
    lines = [
        f"Orbital records could not be resolved for {len(gaps)} of "
        f"{len(resolution.requested)} configured satellites at observation epoch "
        f"{jd_to_datetime(resolution.obs_epoch_jd).isoformat()} UTC:"
    ]
    for gap in gaps:
        bad = resolution.rejected.get(gap.norad_id)
        if bad is None:
            lines.append(f"  {gap.norad_id}: {_gap_sentence(gap)}")
            continue
        if bad.age_days is None:
            lines.append(f"  {gap.norad_id}: best candidate unusable — {bad.reason}")
        else:
            provider = f", provider {bad.provider}" if bad.provider else ""
            lines.append(
                f"  {gap.norad_id}: best candidate is {bad.age_days:.3f} d from "
                f"the observation (epoch "
                f"{jd_to_datetime(bad.epoch_jd).isoformat()} UTC, from "
                f"{bad.source}{provider}) — rejected by {bad.reason}"
            )
        # Both matter: how close the best record was, and that a fresher one
        # could not be requested.
        closer = _closer_sentence(gap)
        if closer is not None:
            lines.append(f"      {closer}")

    limit = resolution.remote_max_age_days
    lines += [
        "",
        f"The remote age ceiling in force is remote_max_age_days="
        f"{'null (disabled)' if limit is None else f'{limit:g}'}. Remedies:",
    ]
    # A service failure is not the configuration being wrong, so lead with the
    # remedy that applies before the ones that change the model.
    unreachable = [gap for gap in gaps if gap.kind in (_GAP_FAILED, _GAP_BLOCKED)]
    if unreachable:
        retry_after = max(
            (
                seconds
                for seconds in (
                    getattr(gap.error, "retry_after", None) for gap in unreachable
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
            f"  - SatChecker did not answer for {len(unreachable)} of these."
            f"{when} Re-run when the service is reachable; nothing about the "
            "configuration need change"
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
    error = OrbitError("\n".join(lines))
    # Chain the service failure these gaps came from, so that "SatChecker was
    # down" can be told from "this configuration asks for records the catalogue
    # does not have" without parsing the message. A failed catalogue search
    # already raises this way, so both routes out of an outage carry it. One
    # error stands for all of them: a transport failure is whole-service, so the
    # batch stopped at the first and the blocked IDs were never sent — and a
    # blocked gap carries that same error.
    outage = next(
        (
            gap.error
            for gap in unreachable
            if isinstance(gap.error, satchecker.SatCheckerTransportError)
        ),
        None,
    )
    if outage is not None:
        error.__cause__ = outage
    return error


def require_complete_coverage(resolution: OrbitResolution) -> OrbitResolution:
    """Return *resolution* unchanged, or raise the actionable coverage error.

    Numbered satellites were each named individually, so one dropped for want of a
    record is indistinguishable from one that simply never passed the target.
    """
    if resolution.requested and not resolution.complete:
        raise _coverage_error(resolution)
    return resolution


def report_named_coverage(
    resolution: OrbitResolution, log=print
) -> OrbitResolution:
    """Coverage policy for satellites selected by *name*, sharing the numbered one.

    A name is a catalogue *query*, so an answer excludes that satellite with its
    reason, while not knowing is fatal — exactly as for a numbered satellite and
    through the same error, so the two routes cannot drift apart.
    :func:`_classify_gap` is where the two are told apart, and the age detail stays
    in the error: it says which limit to change, or which records to fetch.
    """
    if not resolution.requested:
        return resolution

    gaps = [_classify_gap(resolution, nid) for nid in resolution.missing]
    if any(gap.fatal for gap in gaps):
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


# Resolution

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
    near-misses and the epochs everything was judged against; callers decide what
    an incomplete result means, and :func:`require_complete_coverage` is the
    policy tabsim simulations use. Every rule is stated explicitly on the one
    client call, which defaults none of them, while the configuration validation
    stays here so a bad setting is a :class:`TLEConfigurationError` before
    anything is asked.

    *offline* forbids every request without relaxing the age ceiling: a cached
    record outside ``remote_max_age_days`` is refused exactly as it would be
    online. *allow_missing_checksum* accepts TLE lines whose checksum digit the
    archive omitted, on every route a record can arrive by; such records carry
    :data:`~satchecker_client.records.CHECKSUM_STATUS_FIELD` for the rest of
    their lives and never enter the shared cache.
    """
    requested = normalise_norad_ids(norad_ids)
    extra_max_age = validate_age_days(
        extra_orbit_max_age_days, "extra_orbit_max_age_days"
    )
    remote_max_age, reuse_max_age = validate_remote_ages(
        remote_max_age_days, cache_reuse_max_age_days
    )
    obs_epoch_jd = float(obs_epoch_jd)

    if not requested:
        # A satellite-free run is legitimate, and costs nothing: no directory is
        # read, no cache is built and nothing is asked.
        return OrbitResolution(
            [],
            obs_epoch_jd,
            remote_max_age,
            offline=bool(offline),
            cache_reuse_max_age_days=reuse_max_age,
            extra_orbit_max_age_days=extra_max_age,
        )

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

    extra_records = None
    if extra_orbit_dir:
        print(
            f"Extra orbit dir        : {Path(extra_orbit_dir).resolve()} "
            f"(max age {'unlimited' if extra_max_age is None else f'{extra_max_age:g} d'})"
        )
        # Staying silent turns a typo into a run that models different satellites
        # than the ones asked for, while the line above implies it was searched.
        if not Path(extra_orbit_dir).is_dir():
            print(
                "  warning: this extra_orbit_dir does not exist (or is not a "
                "directory); no local records will be found there. Check the path "
                "if you meant to supply your own."
            )
        extra_records = read_extra_orbit_dir(extra_orbit_dir)

    reporter = _Reporter(max_workers)
    result = satchecker.resolve_orbits(
        # Sorted: the acquisition order, so an outage stops the run after the
        # same requests it always did. The run's own order is restored below.
        sorted(requested),
        obs_epoch_jd,
        source_order=(GROUP_EXTRA, GROUP_REMOTE),
        remote_max_age_days=remote_max_age,
        cache_reuse_max_age_days=reuse_max_age,
        extra_orbit_max_age_days=extra_max_age,
        replacement=REPLACEMENT_STRICTLY_FRESHER,
        offline=bool(offline),
        allow_missing_checksum=bool(allow_missing_checksum),
        # Without it an HTTP-200 error envelope — how SatChecker reports its own
        # failures — normalises to an empty frame, and an outage becomes "this
        # satellite has no record".
        strict_response=True,
        # Resolved at call time, from this observation's epoch: the archives do
        # not overlap, and neither endpoint reports "I have nothing that near".
        endpoints=satchecker.nearest_endpoints_for(obs_epoch_jd),
        fallback=True,
        max_workers=max_workers,
        cache=_LazyOrbitCache(),
        extra_records=extra_records,
        on_event=reporter,
        log=print,
    )

    resolution = _adapt(result, requested)
    reporter.finish(resolution)
    _report_unverified(resolution)
    _report_remote_selection(resolution)
    return resolution


# Public orchestration

def resolve_names(
    names,
    obs_epoch_jd: float,
    *,
    search_cache_max_age_days: Optional[float] = DEFAULT_SEARCH_CACHE_MAX_AGE_DAYS,
    offline: bool = False,
    log=print,
) -> list[int]:
    """NORAD IDs for satellites named in the configuration, at *obs_epoch_jd*.

    Names are matched as substrings against an upper-case catalogue; see
    :mod:`tabsim.satchecker_names` for the exact semantics, their sharp edges,
    and the search-cache and offline policy this forwards.

    *obs_epoch_jd* is not optional and not "now": which satellites existed is a
    question about the observation's date. A name the catalogue genuinely does not
    know contributes no satellites and is reported — there is no satellite for a
    record to be missing for — while a search that could not be *run* stops it.
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


# Reproducibility: persist the records a run actually used

def save_orbits_for_reuse(path, norad_ids, records) -> str:
    """Write the orbit records a run used to *path*, as an explicit orbit table.

    ``sim-vis --replay-orbit-dir <this directory>`` then reproduces this run's
    *selection* and its trajectories, independent of the shared cache, of what
    SatChecker serves by then, and of the remote age ceiling. ``norad_ids`` and
    ``records`` are aligned sequences, as :meth:`OrbitResolution.norad_ids` and
    :meth:`OrbitResolution.records` produce them; a misaligned pair is a caller
    bug and raises ``ValueError`` rather than writing a file that reads back
    cleanly and describes different satellites.

    Always writes, and returns the path — an empty selection included: writing
    nothing would make "modelled no satellites" and "not a replay" the same state.
    """
    return satchecker.save_orbits_for_reuse(path, norad_ids, records)


def save_replay_orbits(directory, norad_ids, records) -> tuple[str, str]:
    """Write the two files a frozen replay reads, and return their paths.

    ``(ids_path, records_path)``. The IDs and the records a run saves are one
    decision: both are validated and serialised before either destination is
    opened — there is no atomic replacement across a filesystem failure, but a
    validation failure leaves both files untouched — and a satellite listed twice
    is refused, since matching one record to each ID would be the reselection a
    replay exists to prevent.
    """
    return satchecker.save_replay_orbits(directory, norad_ids, records)


def load_replay_orbits(
    replay_orbit_dir, *, allow_missing_checksum: bool = False
) -> tuple[list[int], list[dict]]:
    """The saved NORAD IDs and records of a previous run, frozen exactly as saved.

    Reads only ``norad_ids.yaml`` and ``used_orbits.json`` from
    *replay_orbit_dir* and returns ``(norad_ids, records)`` — the saved IDs in
    saved order, one aligned record each. An empty saved selection is ``([], [])``:
    a completed run that modelled no satellites.

    **There is no second source.** No name discovery, no managed cache, no
    request, no visibility reselection, no ``max_n_sat``, and no age-based
    replacement: saved records are used however far their epochs are from the
    observation, because that is what makes them the same records, and anything
    short of exact stops the run naming the file or the satellite.

    The configured checksum policy still applies and is passed explicitly, there
    being no safe default. It covers provenance as well as lines: a record saved
    as ``unverified_missing_checksum`` needs *allow_missing_checksum* on every
    pass, whatever its lines carry now, so a permissive run cannot be laundered
    into a strict one by saving it.

    Exact reproduction of the trajectories assumes the same observation, spectral
    inputs, seeds and numerical environment; what this freezes is orbital input.
    """
    try:
        norad_ids, records = satchecker.load_replay_orbits(
            replay_orbit_dir, allow_missing_checksum=allow_missing_checksum
        )
    except OrbitInputError as e:
        raise _replay_error(e, allow_missing_checksum) from e

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


def _replay_error(error: OrbitInputError, allow_missing_checksum: bool) -> OrbitError:
    """A replay that cannot be read exactly, as a tabsim run failure.

    The client's message already names the file, the row and the satellite; what
    it cannot name is the tabsim setting that would have allowed the record.
    Which refusals get that sentence is the client's ``code``
    (:data:`INPUT_CHECKSUM_POLICY`), not the fact that a satellite is named: a
    duplicated ID line and a listed satellite with no saved record both name one,
    and no checksum policy repairs either.
    """
    lines = [str(error)]
    if error.code == INPUT_CHECKSUM_POLICY and not allow_missing_checksum:
        lines.append(
            "If the original run accepted TLE lines without their checksum "
            "digits, the replay has to say so too — set "
            "rfi_sources.tle_satellite.allow_missing_checksum: true, or pass "
            "--allow-missing-checksum."
        )
    return OrbitError(" ".join(lines))
