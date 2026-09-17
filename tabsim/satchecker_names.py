"""Satellite-name discovery: which satellites a configured name selects.

tabsim lets an observation name its RFI satellites (``sat_names``) instead of
listing catalogue numbers, so this module holds the *application policy* around
the catalogue search: how a configured name becomes a query (stripped,
upper-cased, de-duplicated); how long a cached result may be reused, in
wall-clock time; what happens when the search cannot be run, offline or through
a failed refresh; which matched satellites were in orbit at the **observation**
epoch; and what the whole thing cost, in satellites rather than catalogue rows.

The transport, the response envelope and the epoch arithmetic are the client's.
Nothing here parses a reply or builds a URL — a private copy of the client's
transport is exactly how a malformed response used to become "no satellite
matches this name".

**Matching semantics.** ``search-satellites`` matches the query *anywhere* in a
catalogue name, so ``"navstar"`` finds all the ``NAVSTAR nn (USA nnn)`` entries
and ``"starlink"`` every Starlink — what Space-Track's ``op.like(name)`` did, so
configurations written against the old backend keep selecting the same
satellites. The match is **case-sensitive** against a catalogue written almost
entirely in upper case, so the query is upper-cased before it goes out. Two
edges are the service's, not ours: a handful of catalogue names are mixed case
(``DMSat-1``), which no single spelling of a query reaches, and ``%`` and ``_``
are SQL ``LIKE`` wildcards, unescaped.

**Which satellites existed is a question about the observation's date**, not
about today. A satellite that decayed between a 2019 observation and now belongs
in that simulation, and one launched since does not, so the epoch filter runs per
observation against the full cached search result — never against one already
reduced to some other epoch's candidates.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

import satchecker_client as satchecker
from satchecker_client import SatCheckerError, TextOrbitCache, in_orbit_candidates
from satchecker_client import SatCheckerError as OrbitError

from tabsim.orbit_config import (
    DEFAULT_SEARCH_CACHE_MAX_AGE_DAYS,
    orbit_cache_dir,
)


#: Above this many satellites a name query is worth a warning before the requests
#: go out: the nearest-record endpoints are per-satellite, so a broad name —
#: "starlink" matches over twenty thousand objects — is that many requests.
WIDE_NAME_MATCH_THRESHOLD = 500

#: How many IDs a report lists before it summarises instead. The full listing is
#: always available through ``TABSIM_TLE_LOG_DETAIL=1``.
_LIST_LIMIT = 12


def _utc_now() -> datetime:
    """Now, as a timezone-aware UTC datetime.

    A deliberate seam: search freshness is measured against wall-clock time, so
    this is the one thing a freshness test has to be able to freeze.
    """
    return datetime.now(timezone.utc)


def _age_days(snapshot) -> float:
    """Wall-clock age of *snapshot* in days, from its own ``fetched_at``."""
    fetched = snapshot.fetched_at
    if fetched.tzinfo is None:  # a naive stamp is UTC, as store_search writes it
        fetched = fetched.replace(tzinfo=timezone.utc)
    return (_utc_now() - fetched).total_seconds() / 86400.0


def _reuse_message(query: str, snapshot, age_days: float, reason: str) -> str:
    """One line carrying everything needed to judge a reused search result."""
    return (
        f"  Using cached catalogue search for {query!r}: fetched "
        f"{snapshot.fetched_at.isoformat()}, {age_days:.1f} d old, "
        f"{len(snapshot.found)} catalogue row(s) — {reason}"
    )


def _id_list(norad_ids) -> str:
    """IDs for a log line, truncated unless the detail flag is set."""
    ids = sorted(int(nid) for nid in norad_ids)
    detailed = os.environ.get("TABSIM_TLE_LOG_DETAIL", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )
    if detailed or len(ids) <= _LIST_LIMIT:
        return str(ids)
    return (
        f"{ids[:_LIST_LIMIT]} and {len(ids) - _LIST_LIMIT} more (set "
        "TABSIM_TLE_LOG_DETAIL=1 for the full list)"
    )


def search_satellites(
    query: str,
    *,
    cache: Optional[TextOrbitCache] = None,
    max_age_days: Optional[float] = DEFAULT_SEARCH_CACHE_MAX_AGE_DAYS,
    offline: bool = False,
    log=print,
) -> pd.DataFrame:
    """The complete catalogue search result for *query*, cached or freshly fetched.

    The **full** result is what is cached and what comes back: every alias row,
    with its launch and decay dates. Those rows are the evidence the epoch filter
    works from, so a snapshot reduced to one observation's IDs would be useless
    to the next run and misleading to this one.

    Reuse policy, in order:

    - ``offline``: a snapshot is reused whatever its age and no request is made.
      Without one, that is a missing-local-state failure and says so — it is not
      a statement about the catalogue.
    - ``max_age_days`` ``None``: reuse indefinitely. Within it: reuse.
    - otherwise refresh, **replacing** the snapshot. A satellite absent from the
      new result is absent from the catalogue as it stands; keeping it because an
      older search saw it would resurrect exactly the rows the refresh was for.

    A failed refresh falls back to an existing snapshot — a cached empty result
    included, which is a valid answer — with a warning carrying the query, when
    it was fetched, how old that makes it, how many rows it holds and why the
    refresh failed. With nothing cached the failure is raised: reporting it as an
    unmatched name would drop every satellite the query selects.
    """
    query = str(query).strip().upper()
    cache = TextOrbitCache(orbit_cache_dir()) if cache is None else cache
    snapshot = cache.get_search(query, log=log)

    if snapshot is not None:
        age = _age_days(snapshot)
        if offline:
            log(
                _reuse_message(
                    query,
                    snapshot,
                    age,
                    "offline: true, so it is reused whatever its age and no "
                    "refresh was attempted",
                )
            )
            return snapshot.found
        if max_age_days is None:
            log(
                _reuse_message(
                    query,
                    snapshot,
                    age,
                    "search_cache_max_age_days: null, so it is reused indefinitely",
                )
            )
            return snapshot.found
        # Strictly positive, so `search_cache_max_age_days: 0` means what it says —
        # refresh on every online lookup — rather than reusing a snapshot whose age
        # happens to round to zero.
        if max_age_days > 0 and age <= max_age_days:
            log(
                _reuse_message(
                    query,
                    snapshot,
                    age,
                    f"fresh within search_cache_max_age_days={max_age_days:g}",
                )
            )
            return snapshot.found
    elif offline:
        raise OrbitError(
            f"offline: true, but no cached catalogue search for {query!r} exists "
            f"in {cache.cache_dir}. Offline name discovery needs the search to "
            "have been run online at least once; it cannot be inferred from the "
            "orbit records that happen to be cached. Run once without offline, or "
            "list the NORAD IDs in norad_ids / norad_ids_path instead of naming "
            "the satellites."
        )

    try:
        found = satchecker.search_satellites(query)
    except SatCheckerError as error:
        if snapshot is None:
            raise OrbitError(
                f"SatChecker could not answer the catalogue search for {query!r} — "
                f"{error}. Nothing was learnt about this name, so every satellite "
                "it would have selected is unaccounted for; this is not a name the "
                "catalogue does not know. Re-run when the service is reachable, "
                "list the NORAD IDs you want in norad_ids / norad_ids_path, or "
                "re-run a configuration whose searches are already cached."
            ) from error
        log(
            _reuse_message(
                query,
                snapshot,
                _age_days(snapshot),
                f"the refresh failed — {error}",
            )
        )
        return snapshot.found

    satchecker.store_or_warn(
        lambda: cache.store_search(query, found),
        cache.search_path(query),
        f"cached catalogue search for {query!r}",
        log,
    )
    return found


def norad_ids_from_names(
    names,
    obs_epoch_jd: float,
    *,
    search_cache_max_age_days: Optional[float] = DEFAULT_SEARCH_CACHE_MAX_AGE_DAYS,
    offline: bool = False,
    log=print,
) -> list[int]:
    """NORAD IDs the configured *names* select at *obs_epoch_jd*.

    Every distinct query's full result is collected and the rows combined
    *before* the epoch filter and the per-ID de-duplication. That order matters:
    a null or later launch date on one alias row means that row does not say, not
    that the satellite had not launched, so reading the dates off whichever row
    survived de-duplication rules out satellites the catalogue never ruled out.

    Distinct catalogue numbers are kept even when two share an ``OBJECT_ID``:
    nothing says which is current, so the ambiguity is reported rather than
    guessing an identity merge.
    """
    queries = list(
        dict.fromkeys(
            str(name).strip().upper() for name in (names or []) if str(name).strip()
        )
    )
    if not queries:
        return []

    cache = TextOrbitCache(orbit_cache_dir())
    frames: list[pd.DataFrame] = []
    ids_by_query: dict[str, set[int]] = {}
    unmatched: list[str] = []

    for query in queries:
        found = search_satellites(
            query,
            cache=cache,
            max_age_days=search_cache_max_age_days,
            offline=offline,
            log=log,
        )
        ids_by_query[query] = {int(nid) for nid in found["NORAD_CAT_ID"]}
        if not len(found):
            unmatched.append(query)
            log(f"  {query!r}: no satellite in the catalogue matches this name")
            continue
        examples = ", ".join(
            str(name) for name in list(found["OBJECT_NAME"])[:3] if not pd.isna(name)
        )
        log(
            f"  {query!r}: {len(found)} catalogue row(s), "
            f"{len(ids_by_query[query])} satellite(s)"
            + (f" — e.g. {examples}" if examples else "")
        )
        frames.append(found)

    total_rows = sum(len(frame) for frame in frames)
    if not frames:
        log(
            f"Catalogue search       : {len(queries)} query/queries, no catalogue "
            "rows, so no satellite is selected by name"
        )
        return []

    combined = pd.concat(frames, ignore_index=True)
    candidates = in_orbit_candidates(combined, obs_epoch_jd)
    norad_ids = [int(nid) for nid in candidates["NORAD_CAT_ID"]]
    selected = set(norad_ids)
    unique_ids = {nid for ids in ids_by_query.values() for nid in ids}

    log(
        f"Catalogue search       : {len(queries)} query/queries, {total_rows} "
        f"catalogue row(s), {len(unique_ids)} satellite(s), {len(norad_ids)} of "
        "them in orbit at the observation epoch"
    )
    if unmatched:
        log(
            f"  warning: {len(unmatched)} name(s) match nothing in the catalogue "
            f"and contribute no satellites: {unmatched}"
        )
    excluded = unique_ids - selected
    if excluded:
        log(
            f"  {len(excluded)} satellite(s) excluded: the catalogue says they had "
            f"decayed or had not launched at the observation epoch — "
            f"{_id_list(excluded)}"
        )
    for query in queries:
        matched = ids_by_query[query]
        if matched and not (matched & selected):
            log(
                f"  {query!r}: no satellite matched by this name was in orbit at "
                "the observation epoch"
            )

    _warn_shared_designators(combined, selected, log)

    if len(norad_ids) > WIDE_NAME_MATCH_THRESHOLD:
        log(
            f"  warning: these names select {len(norad_ids)} satellites (from "
            f"{total_rows} catalogue rows — a satellite appears once per alias). "
            "SatChecker serves one orbit record per request, so a cold cache costs "
            f"that many requests, at most {satchecker.MAX_WORKERS} in flight, plus "
            "a possible second request each against the other archive, followed by "
            "that many visibility propagations. Narrow the names, or list the "
            "NORAD IDs you actually want."
        )
    return norad_ids


def _warn_shared_designators(combined: pd.DataFrame, selected: set, log) -> None:
    """Report *candidate* satellites that share an international designator.

    One object can be listed under two NORAD IDs — a reassigned catalogue number
    — with nothing in the response saying which is current, so both are kept and
    the ambiguity is named rather than guessing a merge.

    This runs at discovery, before any record is acquired, so it can only speak
    of candidates: either number may still have no acceptable record, fail the
    visibility cuts, have no spectral model, or fall outside ``max_n_sat``.
    Promising here that both will be modelled describes a selection that has not
    happened yet.
    """
    designators: dict[str, set[int]] = {}
    for norad_id, object_id in zip(
        combined["NORAD_CAT_ID"], combined["OBJECT_ID"]
    ):
        if pd.isna(object_id) or not str(object_id).strip():
            continue
        if int(norad_id) not in selected:
            continue
        designators.setdefault(str(object_id).strip(), set()).add(int(norad_id))
    shared = {
        object_id: ids for object_id, ids in designators.items() if len(ids) > 1
    }
    if not shared:
        return
    for object_id, ids in sorted(shared.items()):
        log(
            f"  warning: OBJECT_ID {object_id} is carried by more than one "
            f"candidate NORAD catalogue ID: {sorted(ids)}. Nothing in the "
            "catalogue says which number is current, so none of them is dropped. "
            "They remain distinct candidates, and one object under two numbers "
            "may be modelled separately if both survive this run's final "
            "selection: an acceptable record, the visibility cuts, a spectral "
            "model and max_n_sat."
        )
