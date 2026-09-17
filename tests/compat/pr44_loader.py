"""HISTORICAL TEST CODE — PR #44's frozen-replay reader, kept as an oracle.

This is ``tabsim/orbit.py``'s replay loader and the helpers it needs, copied
verbatim from tab-sim ``e3d957d7846b98617cc65874ef6d4c71bd8d2377`` (the head of
``satchecker-orbits``, PR #44), together with the two ``tabsim.orbit_config``
helpers it calls. It exists for one question: *can a replay directory written by
the adopted code still be read by the code that wrote the format?* Answering
that with the current implementation would be answering it with itself.

**Not production code, and not to be imported by it.** Nothing under
``tabsim/`` may import this module, and it is never to be "kept in sync",
refactored or improved: the moment it stops being a byte-faithful copy of #44 it
stops being evidence of anything. It is frozen at the SHA above. If the
backward-compatibility contract is ever deliberately broken, delete this file
and the test that uses it rather than editing either.

The only deviations from the original source are mechanical, and none of them
touch behaviour:

- ``OrbitError``/``TLEError`` are bound here rather than imported from
  ``tabsim.orbit``, so the oracle does not read the module under test;
- ``normalise_norad_ids``, ``_as_norad_id`` and ``TLEConfigurationError`` are
  copied from ``tabsim/orbit_config.py`` at the same SHA for the same reason;
- the docstrings are the originals.

Everything it does import — ``read_orbit_file``, ``validated_record``,
``norad_id_of`` and the checksum-status constants — comes from
``satchecker_client``, which #44 also imported them from. Those are the
interfaces #44 was written against; the pin in ``pyproject.toml`` is what says
which revision of them is in play, and the fixture provenance records it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np

from satchecker_client import (
    CacheValidationError,
    SatCheckerError as OrbitError,
    read_orbit_file,
)
from satchecker_client.records import (
    CHECKSUM_STATUS_FIELD,
    CHECKSUM_UNVERIFIED_MISSING,
    norad_id_of,
    validated_record,
)


TLEError = OrbitError

#: Above this many remote records the per-satellite log lines are replaced by a
#: grouped summary; set ``TABSIM_TLE_LOG_DETAIL=1`` to force the full listing.
_GROUPED_LOG_THRESHOLD = 12
_LOG_DETAIL_ENV = "TABSIM_TLE_LOG_DETAIL"

#: The two files a completed run writes into its ``input_data`` directory, and the
#: only two a frozen replay reads.
REPLAY_IDS_FILE = "norad_ids.yaml"
REPLAY_RECORDS_FILE = "used_orbits.json"


# ---------------------------------------------------------------------------
# From tabsim/orbit_config.py at e3d957d
# ---------------------------------------------------------------------------

class TLEConfigurationError(TLEError, ValueError):
    """An orbit-related configuration value is missing, malformed or out of range."""


def _as_norad_id(value, where: str) -> int:
    """Coerce one entry to a positive integral NORAD catalogue ID, exactly."""
    # bool is Integral and np.bool_ is finite and equal to 0 or 1, so both would
    # pass an exactness check as satellite 1; neither is an identity.
    if isinstance(value, (bool, np.bool_)):
        raise TLEConfigurationError(f"{where}: {value!r} is not a NORAD catalogue ID")
    try:
        return norad_id_of({"NORAD_CAT_ID": value}, where)
    except ValueError as e:
        raise TLEConfigurationError(str(e)) from e


def normalise_norad_ids(values, source: str = "norad_ids") -> list[int]:
    """Return validated, order-preserving, de-duplicated NORAD IDs."""
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


# ---------------------------------------------------------------------------
# From tabsim/orbit.py at e3d957d
# ---------------------------------------------------------------------------

def _detail_requested() -> bool:
    return os.environ.get(_LOG_DETAIL_ENV, "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


def _id_list(norad_ids) -> str:
    """IDs for a log line, truncated unless ``TABSIM_TLE_LOG_DETAIL`` is set."""
    ids = sorted(int(nid) for nid in norad_ids)
    if _detail_requested() or len(ids) <= _GROUPED_LOG_THRESHOLD:
        return str(ids)
    return (
        f"{ids[:_GROUPED_LOG_THRESHOLD]} and {len(ids) - _GROUPED_LOG_THRESHOLD} "
        f"more (set {_LOG_DETAIL_ENV}=1 for the full list)"
    )


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
