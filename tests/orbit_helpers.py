"""Shared fixtures for the orbit-record tests: real records, no network.

Checksum-valid TLE lines at an arbitrary NORAD ID and epoch, the two record kinds
built from them, the historical archive defects derived from a valid line, and
stubs for the three service seams (both nearest-record endpoints and the
catalogue name search).

TLE fixtures are *derived*, never typed out: the checksum is what makes a
single-character corruption detectable, so a fixture whose ID or epoch was edited
by hand would be rejected by the same parser the production code uses.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.parse
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from satchecker_client import client
from satchecker_client.client import SEARCH_COLUMNS
from satchecker_client.records import KIND_FIELD, KIND_OMM, KIND_TLE
from satchecker_client.tle_parse import parse_tle_elements, tle_checksum
from satchecker_client import datetime_to_jd, jd_to_datetime


def jd(year, month, day, hour=0, minute=0, second=0) -> float:
    """UTC calendar time -> Julian Date, for readable historical fixtures."""
    return datetime_to_jd(datetime(year, month, day, hour, minute, second))


#: A real, checksum-valid ISS TLE. NORAD 25544, epoch 2023-02-24 13:44:58 UTC.
ISS_LINE1 = "1 25544U 98067A   23055.57290509  .00017000  00000-0  31277-3 0  9993"
ISS_LINE2 = "2 25544  51.6389 166.3375 0005390  23.7047 166.5963 15.49291762384327"
ISS_NORAD_ID = 25544
ISS_EPOCH_JD = 2459999.5 + 0.57290509

#: A real, checksum-valid GPS TLE. NORAD 32260, epoch 2024-01-16 05:49:43 UTC.
GPS_LINE1 = "1 32260U 07047A   24016.24286508  .00000002  00000+0  00000+0 0  9999"
GPS_LINE2 = "2 32260  53.5283 116.5894 0152287  73.4949 288.1818  2.00572233119108"
GPS_NORAD_ID = 32260
GPS_EPOCH_JD = 2460310.5 + 0.24286508

#: Where a record says whether anything ever verified its TLE lines. Written by
#: the client and carried through the managed cache, the saved run records and
#: frozen replay, so a run that opted in to checksum-less lines cannot later be
#: mistaken for one that did not. The two values are the client's.
CHECKSUM_STATUS_FIELD = "TLE_CHECKSUM_STATUS"
STATUS_VERIFIED = "verified"
STATUS_UNVERIFIED = "unverified_missing_checksum"


def with_checksum(line: str) -> str:
    """*line* with its column-69 modulo-10 checksum recomputed.

    Substituting an ID or an epoch invalidates the original checksum, and every
    parser in this stack rejects a bad one — as it should.
    """
    body = line[:68]
    return body + str(tle_checksum(body))


def tle_lines(norad_id: int = ISS_NORAD_ID, epoch_jd: float = ISS_EPOCH_JD):
    """A checksum-valid ``(line1, line2)`` pair for *norad_id* at *epoch_jd*.

    The ID and the line-1 epoch field are substituted into the ISS template at
    their fixed columns; the parsed epoch round-trips within the format's ~1 ms.
    """
    nid = f"{int(norad_id):05d}"
    stamp = jd_to_datetime(float(epoch_jd))
    year_field = stamp.year % 100
    day_of_year = (
        (stamp - datetime(stamp.year, 1, 1)).total_seconds() / 86400.0 + 1.0
    )
    epoch_field = f"{year_field:02d}{day_of_year:012.8f}"  # columns 19-32
    line1 = "1 " + nid + ISS_LINE1[7:18] + epoch_field + ISS_LINE1[32:]
    line2 = "2 " + nid + ISS_LINE2[7:]
    return with_checksum(line1), with_checksum(line2)


def without_checksum(line: str) -> str:
    """*line* with its checksum digit removed, as SatChecker's archive serves some.

    Derived from a valid line, so the only thing wrong with it is the one defect.
    """
    return line[:68]


def with_stray_backslash(line: str) -> str:
    """*line* with the archive's trailing backslash appended.

    The checksum still verifies once the backslash is stripped, so a record
    carrying this is as trustworthy as a clean one — provided the canonical line
    is what gets written back out.
    """
    return line + "\\"


def tle_record(norad_id=ISS_NORAD_ID, line1=ISS_LINE1, line2=ISS_LINE2, **extra):
    """A TLE record in the shape :mod:`tabsim.orbit` resolves and caches."""
    return {
        "NORAD_CAT_ID": norad_id,
        KIND_FIELD: KIND_TLE,
        "OBJECT_NAME": "TEST SAT",
        "TLE_LINE1": line1,
        "TLE_LINE2": line2,
        "DATA_SOURCE": "test",
        **extra,
    }


def tle_record_at(norad_id: int, epoch_jd: float, **extra):
    """A checksum-valid TLE record for *norad_id* with its epoch at *epoch_jd*."""
    line1, line2 = tle_lines(norad_id, epoch_jd)
    return tle_record(norad_id=norad_id, line1=line1, line2=line2, **extra)


def omm_record_from_tle(
    norad_id=ISS_NORAD_ID, line1=ISS_LINE1, line2=ISS_LINE2, **extra
):
    """An OMM record carrying the *same* elements as the given TLE pair.

    Deriving it from a TLE is what makes the two propagation paths directly
    comparable: a degrees-for-radians slip shows up as a position difference
    against a satellite whose elements are known to be identical.
    """
    elements = parse_tle_elements(line1, line2)
    return {
        "NORAD_CAT_ID": norad_id,
        KIND_FIELD: KIND_OMM,
        "OBJECT_NAME": "TEST SAT",
        "OBJECT_ID": "1998-067A",
        # An ISO 8601 EPOCH carries microseconds, and no more.
        "EPOCH": jd_to_datetime(elements["EPOCH_JD"]).isoformat(),
        "INCLINATION": elements["INCLINATION"],
        "RA_OF_ASC_NODE": elements["RA_OF_ASC_NODE"],
        "ECCENTRICITY": elements["ECCENTRICITY"],
        "ARG_OF_PERICENTER": elements["ARG_OF_PERICENTER"],
        "MEAN_ANOMALY": elements["MEAN_ANOMALY"],
        "MEAN_MOTION": elements["MEAN_MOTION"],
        "BSTAR": elements["BSTAR"],
        "DATA_SOURCE": "test",
        **extra,
    }


def omm_record_at(norad_id: int, epoch_jd: float, **extra):
    """An OMM record for *norad_id* with its epoch at *epoch_jd*."""
    line1, line2 = tle_lines(norad_id, epoch_jd)
    return omm_record_from_tle(norad_id=norad_id, line1=line1, line2=line2, **extra)


def record_at(kind: str, norad_id: int, epoch_jd: float, **extra):
    """Either kind, for the parametrisations that must hold for both."""
    builder = tle_record_at if kind == KIND_TLE else omm_record_at
    return builder(norad_id, epoch_jd, **extra)


def forbidden(what: str):
    """A callable that fails the test if anything calls it."""

    def refuse(*args, **kwargs):
        raise AssertionError(f"{what} must not be reached by this test")

    return refuse


def stub_service(monkeypatch, records_by_id, endpoint="tle"):
    """Serve *records_by_id* from the nearest-record endpoint of the given kind.

    Each call is recorded as ``(norad_id, epoch_jd, strict_response)``; the third
    element is the opt-in without which an HTTP-200 error envelope reads as "this
    satellite has no record".
    """
    calls = []

    def fetch(norad_id, epoch_jd, *, strict_response=False):
        calls.append((int(norad_id), float(epoch_jd), strict_response))
        record = records_by_id.get(int(norad_id))
        if record is None:
            return pd.DataFrame()
        return pd.DataFrame([record])

    def empty(norad_id, epoch_jd, *, strict_response=False):
        calls.append((int(norad_id), float(epoch_jd), strict_response))
        return pd.DataFrame()

    monkeypatch.setattr(
        client, "fetch_nearest_tle", fetch if endpoint == "tle" else empty
    )
    monkeypatch.setattr(
        client, "fetch_nearest_omm", fetch if endpoint == "omm" else empty
    )
    return calls


class EndpointStub:
    """One nearest-record endpoint with scripted per-ID answers, recording calls.

    An *answers* entry is a record dict, a DataFrame, or an exception to raise; an
    ID with no entry gets *default*, and ``None`` is an empty frame — the service
    saying it has no such record.
    """

    def __init__(self, label, answers=None, default=None):
        self.label = label
        self.answers = dict(answers or {})
        self.default = default
        #: ``(norad_id, epoch_jd, strict_response)`` per call, as it arrived.
        self.calls: list[tuple] = []

    def __call__(self, norad_id, epoch_jd, *, strict_response=False):
        self.calls.append((int(norad_id), float(epoch_jd), strict_response))
        answer = self.answers.get(int(norad_id), self.default)
        if isinstance(answer, BaseException):
            raise answer
        if answer is None:
            return pd.DataFrame()
        if isinstance(answer, pd.DataFrame):
            return answer.copy()
        return pd.DataFrame([answer])

    @property
    def requested(self) -> list[int]:
        return [norad_id for norad_id, _, _ in self.calls]


def stub_endpoints(monkeypatch, *, tle=None, omm=None, tle_default=None, omm_default=None):
    """Install an :class:`EndpointStub` on each nearest-record endpoint.

    Returns ``(nearest_tle, nearest_omm)`` in the order
    :func:`satchecker_client.nearest_endpoints_for` returns them for a
    pre-handover epoch, so a test can assert what each archive was asked.
    """
    stubs = (
        EndpointStub("nearest-TLE", tle, tle_default),
        EndpointStub("nearest-OMM", omm, omm_default),
    )
    monkeypatch.setattr(client, "fetch_nearest_tle", stubs[0])
    monkeypatch.setattr(client, "fetch_nearest_omm", stubs[1])
    return stubs


def stub_failing_service(monkeypatch, error, endpoint=None):
    """Make one or both nearest-record endpoints raise *error*.

    *endpoint* is ``"tle"``, ``"omm"`` or ``None`` for both. Raised fresh per call
    so one exception instance is not shared between the threads of a batch.
    """
    calls = []

    def fail(norad_id, epoch_jd, *, strict_response=False):
        calls.append((int(norad_id), float(epoch_jd), strict_response))
        raise error

    def empty(norad_id, epoch_jd, *, strict_response=False):
        calls.append((int(norad_id), float(epoch_jd), strict_response))
        return pd.DataFrame()

    monkeypatch.setattr(
        client, "fetch_nearest_tle", fail if endpoint in (None, "tle") else empty
    )
    monkeypatch.setattr(
        client, "fetch_nearest_omm", fail if endpoint in (None, "omm") else empty
    )
    return calls


def search_row(
    norad_id: int,
    name: str,
    object_id=None,
    launch_date=None,
    decay_date=None,
    object_type="PAYLOAD",
    rcs_size="LARGE",
) -> dict:
    """One ``search-satellites`` row, in the service's own field spelling."""
    return {
        "satellite_id": int(norad_id),
        "satellite_name": name,
        "international_designator": object_id,
        "object_type": object_type,
        "rcs_size": rcs_size,
        "launch_date": launch_date,
        "decay_date": decay_date,
    }


def search_payload(rows) -> bytes:
    """A well-formed ``search-satellites`` reply carrying *rows*."""
    rows = list(rows)
    return json.dumps({"count": len(rows), "data": rows}).encode()


def search_frame(rows) -> pd.DataFrame:
    """*rows* as the normalised frame ``search_satellites`` returns.

    Seeds the cache directly, so a snapshot test need not go through the
    transport to create the state it is testing.
    """
    normalised = [
        {
            "NORAD_CAT_ID": int(row["satellite_id"]),
            "OBJECT_NAME": row["satellite_name"],
            "OBJECT_ID": row["international_designator"],
            "OBJECT_TYPE": row["object_type"],
            "RCS_SIZE": row["rcs_size"],
            "LAUNCH_DATE": row["launch_date"],
            "DECAY_DATE": row["decay_date"],
        }
        for row in rows
    ]
    if not normalised:
        return pd.DataFrame(columns=SEARCH_COLUMNS)
    return pd.DataFrame(normalised, columns=SEARCH_COLUMNS)


@contextmanager
def restored_stdout():
    """``run_sim_config`` redirects stdout and does not restore it on failure.

    Any test that expects a run to stop early needs this, or the rest of the
    session writes into that run's closed log file.
    """
    backup = sys.stdout
    try:
        yield
    finally:
        sys.stdout = backup


def write_sim_config(path, tle_satellite=None, **sections) -> str:
    """Write a minimal simulation config that runs in about a second.

    Two antennas, two time steps and one channel: enough for ``load_obs`` to
    build a real observation, little enough that a test asserting *when*
    something is validated does not pay for a simulation. Everything not given
    here comes from ``sim_config_base.yaml``.
    """
    import yaml

    config = {
        "telescope": {"name": "MeerKAT", "n_ant": 2},
        "observation": {
            "start_time_lha": 0.0,
            "ra": 27.0,
            "dec": -30.0,
            "int_time": 2.0,
            "n_time": 2,
            "n_int": 2,
            "start_freq": 1.227e9,
            "chan_width": 209e3,
            "n_freq": 1,
            "SEFD": 420,
            "random_seed": 12345,
        },
        "output": {"zarr": False, "ms": False, "path": "./out", "overwrite": True},
        "diagnostics": {"rfi_seps": False, "src_alt": False, "uv_cov": False},
        "rfi_sources": {"tle_satellite": dict(tle_satellite or {})},
    }
    for section, values in sections.items():
        config.setdefault(section, {}).update(values)
    path = str(path)
    with open(path, "w") as handle:
        yaml.safe_dump(config, handle)
    return path


def _serve_transport(monkeypatch, fake_get):
    """Install a search transport at the one seam there is.

    ``client._http_get`` is what the public :func:`search_satellites` uses, so it
    is the only place a catalogue search can leave from. Nothing else is patched:
    a reintroduced private bypass must hit the suite's network block instead of
    keeping the "public search" test passing.
    """
    monkeypatch.setattr(client, "_http_get", fake_get)


def forbid_search(monkeypatch):
    """Fail the test if a catalogue search request is attempted at all."""
    _serve_transport(monkeypatch, forbidden("the catalogue search"))


def serve_raw_search(monkeypatch, payload):
    """Answer every ``search-satellites`` request with one literal JSON payload.

    For the malformed-envelope cases, which have no rows to key on.
    """
    body = json.dumps(payload).encode()
    _serve_transport(monkeypatch, lambda url, timeout=None: body)


def serve_search(monkeypatch, by_query):
    """Answer ``search-satellites`` requests from *by_query*.

    Keys are the query as it goes out on the wire — the catalogue is upper case
    and matched case-sensitively, so ``"NAVSTAR"``, not ``"navstar"``. A value is
    a list of :func:`search_row` rows or an exception. Calls record ``(name, url)``.
    """
    calls = []

    def fake_get(url, timeout=None):
        if "search-satellites" not in url:
            raise AssertionError(f"unexpected non-search request: {url}")
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        name = query.get("name", [""])[0]
        calls.append((name, url))
        try:
            answer = by_query[name]
        except KeyError:
            raise AssertionError(
                f"no stubbed search response for {name!r} (stubbed: "
                f"{sorted(by_query)})"
            ) from None
        if isinstance(answer, Exception):
            raise answer
        return search_payload(answer)

    _serve_transport(monkeypatch, fake_get)
    return calls


#: Replay directories written by tab-sim e3d957d (PR #44), with what #44's own
#: loader read back out of each. See ``tests/compat/fixtures/PROVENANCE.txt``.
COMPAT_FIXTURE_DIR = Path(__file__).resolve().parent / "compat" / "fixtures"


def compat_fixture(name: str):
    """One frozen #44 replay directory and the pair #44 read back from it.

    Returns ``(directory, expected)``, *expected* carrying the checksum policy the
    case needs, the saved IDs in saved order, and one record each, spelled the way
    :func:`comparable_record` spells them.
    """
    directory = COMPAT_FIXTURE_DIR / name
    expected = json.loads((directory / "expected.json").read_text())
    return directory, expected


def json_ready(value):
    """One record cell as the frozen fixtures spell it.

    A NumPy scalar is unwrapped and every flavour of null becomes ``None``:
    ``NaN != NaN`` makes a record carrying one impossible to compare by equality.
    """
    item = getattr(value, "item", None)
    if item is not None and getattr(value, "shape", ()) == ():
        value = item()
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def comparable_record(record) -> dict:
    """*record* in the spelling the frozen fixtures are stored in."""
    return {key: json_ready(value) for key, value in dict(record).items()}
