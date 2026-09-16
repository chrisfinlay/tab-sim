"""Satellite-name lookup against SatChecker, which satchecker-client does not provide.

tabsim lets an observation name its RFI satellites (``sat_names``) instead of
listing catalogue numbers, which TABASCAL does not — so the endpoint that turns a
name into NORAD IDs has no counterpart in :mod:`satchecker_client`, and lives here.

It is built on that client's transport, so it inherits its timeout, its
``User-Agent``, and its error contract: :class:`SatCheckerResponseError` for a
reply that cannot be used, :class:`SatCheckerTransportError` (or
:class:`SatCheckerRateLimitError`) for a service that cannot be reached. Two of
the helpers it uses, ``_http_get`` and ``_load_json``, are private to
satchecker-client, so a release of that package can rename them without notice.

**Matching semantics.** ``search-satellites`` matches the query *anywhere* in a
catalogue name, so ``"navstar"`` finds all 80 ``NAVSTAR nn (USA nnn)`` entries and
``"starlink"`` finds every Starlink. That is what Space-Track's ``op.like(name)``
did — its ``~~`` operator wraps the pattern in wildcards — so configurations
written against the old backend keep selecting the same satellites. SatChecker
also has an exact-name index (``norad-ids-from-name``), which is *not* what tabsim
wants here: it would have silently reduced ``sat_names: [navstar]`` to nothing.

The search is **case-sensitive** against a catalogue written in upper case, so
the query is upper-cased before it goes out — as tabsim always did for
Space-Track (``op.like(name.upper())``). Without that, the lower-case names in
the shipped example configurations would match nothing at all, silently, which is
exactly the failure this note exists to prevent recurring.

**Decayed objects are dropped.** The catalogue keeps re-entered satellites, and a
name search returns them; they cannot be observed and have no orbital record near
any present-day epoch, so including them would turn every historical namesake
into a coverage failure.
"""

from __future__ import annotations

import urllib.parse

from satchecker_client.client import (
    BASE_URL,
    SatCheckerResponseError,
    _http_get,
    _load_json,
)


#: SatChecker's substring name search. Distinct from the ``get-`` tools
#: satchecker-client uses, and unversioned in the same way.
SEARCH_ENDPOINT = "search-satellites"


def _rows(payload, url: str) -> list:
    if not isinstance(payload, dict):
        raise SatCheckerResponseError(
            f"SatChecker returned an unexpected response shape ({url}): "
            f"{type(payload).__name__}"
        )
    rows = payload.get("data") or []
    if not isinstance(rows, list):
        raise SatCheckerResponseError(
            f"SatChecker name search returned unexpected rows ({url}): "
            f"{type(rows).__name__}"
        )
    return rows


def search_satellites_by_name(name: str) -> list[tuple[int, str]]:
    """``(norad_id, satellite_name)`` for every catalogue entry containing *name*.

    Matched case-insensitively by upper-casing the query, since the catalogue is
    upper case and the endpoint is not. Still in orbit only — an entry carrying a
    ``decay_date`` is skipped. Results keep the order the service listed them in
    and are de-duplicated by ID, since one object can appear under several names.

    An unmatched name is an empty list, not an error: naming a satellite that has
    not launched (or is spelt differently in the catalogue) is a configuration
    matter for the caller to report against the whole set, not a transport
    failure.
    """
    query = urllib.parse.urlencode({"name": str(name).strip().upper()})
    url = f"{BASE_URL}/{SEARCH_ENDPOINT}/?{query}"

    found: list[tuple[int, str]] = []
    seen: set[int] = set()
    for row in _rows(_load_json(_http_get(url), url), url):
        if not isinstance(row, dict):
            raise SatCheckerResponseError(
                f"SatChecker name search returned a non-object row ({url}): "
                f"{type(row).__name__}"
            )
        if row.get("decay_date"):
            continue
        try:
            norad_id = int(row["satellite_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise SatCheckerResponseError(
                f"SatChecker name search row has no usable satellite_id ({url}): "
                f"{error}"
            ) from error
        if norad_id in seen:
            continue
        seen.add(norad_id)
        found.append((norad_id, str(row.get("satellite_name") or "")))
    return found


def norad_ids_from_names(names, log=print) -> tuple[list[int], list[str]]:
    """Resolve *names* to NORAD IDs, reporting the ones the catalogue does not know.

    Returns ``(norad_ids, unmatched_names)``. IDs keep the order the names were
    given in and are de-duplicated across names, since two names can legitimately
    match the same object.

    Unmatched names are returned rather than raised on so the caller can decide.
    tabsim's satellite selection is a *filter* — a named satellite that never
    passes near the target contributes nothing anyway — so one unrecognised name
    should not cost an otherwise complete simulation.
    """
    norad_ids: list[int] = []
    unmatched: list[str] = []
    seen: set[int] = set()
    for name in names:
        text = str(name).strip()
        if not text:
            continue
        found = search_satellites_by_name(text)
        if not found:
            unmatched.append(text)
            log(f"  {text!r}: no satellite in orbit matches this name")
            continue
        new = [norad_id for norad_id, _ in found if norad_id not in seen]
        seen.update(new)
        norad_ids += new
        examples = ", ".join(sat_name for _, sat_name in found[:3] if sat_name)
        log(
            f"  {text!r}: {len(found)} satellite(s)"
            + (f" — e.g. {examples}" if examples else "")
        )
    return norad_ids, unmatched
