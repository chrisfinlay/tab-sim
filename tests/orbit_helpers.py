"""Shared fixtures for the orbit-record tests: real records, no network.

Checksum-valid TLE lines at an arbitrary NORAD ID and epoch, the two record kinds
built from them, the historical archive defects derived from a valid line, and stubs
for the three service seams. TLE fixtures are *derived*, never typed out: the
checksum is what makes a single-character corruption detectable, so one edited by
hand would be rejected by the same parser the production code uses.
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

import satchecker_client
from satchecker_client import client
from satchecker_client.client import SEARCH_COLUMNS
from satchecker_client.records import KIND_FIELD, KIND_OMM, KIND_TLE
from satchecker_client.tle_parse import parse_tle_elements, tle_checksum
from satchecker_client import datetime_to_jd, jd_to_datetime

from tabsim import orbit
from tabsim import tle as tle_module


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

    The checksum still verifies once the backslash is stripped, so a record carrying
    this is as trustworthy as a clean one — if the canonical line is written back out.
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

    Deriving it from a TLE is what makes the two propagation paths comparable: a
    degrees-for-radians slip shows up as a position difference between them.
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


class EndpointStub:
    """One nearest-record endpoint with scripted per-ID answers, recording calls.

    An *answers* entry is a record dict, a DataFrame, or an exception to raise; an ID
    with no entry gets *default*, and ``None`` is an empty frame — the service saying
    it has no such record.
    """

    def __init__(self, label, answers=None, default=None, calls=None):
        self.label = label
        self.answers = dict(answers or {})
        self.default = default
        #: ``(norad_id, epoch_jd, strict_response)`` per call, as it arrived. A
        #: list passed in is shared with the other endpoint, which is how a test
        #: asserts the order two archives were asked in rather than two
        #: unordered per-endpoint lists.
        self.calls: list[tuple] = [] if calls is None else calls

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


def stub_endpoints(
    monkeypatch, *, tle=None, omm=None, tle_default=None, omm_default=None, calls=None
):
    """Install an :class:`EndpointStub` on each nearest-record endpoint.

    Returns ``(nearest_tle, nearest_omm)`` in the order
    :func:`satchecker_client.nearest_endpoints_for` gives them for a pre-handover
    epoch, so a test can assert what each archive was asked.
    """
    stubs = (
        EndpointStub("nearest-TLE", tle, tle_default, calls),
        EndpointStub("nearest-OMM", omm, omm_default, calls),
    )
    monkeypatch.setattr(client, "fetch_nearest_tle", stubs[0])
    monkeypatch.setattr(client, "fetch_nearest_omm", stubs[1])
    return stubs


def stub_service(monkeypatch, records_by_id, endpoint="tle"):
    """Serve *records_by_id* from the nearest-record endpoint of the given kind.

    The other endpoint answers empty. Returns one ordered stream of ``(norad_id,
    epoch_jd, strict_response)`` across both, the third being the opt-in without which
    an HTTP-200 error envelope reads as "no record".
    """
    calls: list[tuple] = []
    served = dict(records_by_id)
    stub_endpoints(
        monkeypatch,
        tle=served if endpoint == "tle" else None,
        omm=served if endpoint == "omm" else None,
        calls=calls,
    )
    return calls


def stub_failing_service(monkeypatch, error, endpoint=None):
    """Make one or both nearest-record endpoints raise *error*.

    *endpoint* is ``"tle"``, ``"omm"`` or ``None`` for both; the other answers
    empty. Returns the same combined call stream as :func:`stub_service`.
    """
    calls: list[tuple] = []
    stub_endpoints(
        monkeypatch,
        tle_default=error if endpoint in (None, "tle") else None,
        omm_default=error if endpoint in (None, "omm") else None,
        calls=calls,
    )
    return calls


def write_orbit_json(path, records) -> Path:
    """*records* as one column-oriented orbit table, the shape the readers read."""
    rows = [dict(record) for record in records]
    columns: list[str] = []
    for row in rows:
        columns += [column for column in row if column not in columns]
    payload = {
        column: {str(index): row.get(column) for index, row in enumerate(rows)}
        for column in columns
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def write_replay_dir(directory, norad_ids, records) -> Path:
    """The two files a frozen replay reads, as a completed run would write them."""
    directory.mkdir(parents=True, exist_ok=True)
    orbit.save_orbits_for_reuse(
        directory / "used_orbits.json", list(norad_ids), list(records)
    )
    (directory / "norad_ids.yaml").write_text(
        "".join(f"{int(nid)}\n" for nid in norad_ids)
    )
    return directory


def write_replay_pair(directory, norad_ids, records) -> Path:
    """The two replay files, written exactly as given and validated by nothing.

    Deliberately not :func:`write_replay_dir`, whose writer exists to refuse
    producing some of the corruption the loader cases need.
    """
    directory.mkdir(parents=True, exist_ok=True)
    write_orbit_json(directory / "used_orbits.json", records)
    (directory / "norad_ids.yaml").write_text(
        "".join(f"{int(norad_id)}\n" for norad_id in norad_ids)
    )
    return directory


#: ``(module, attribute, diagnostic)`` for every route to a record other than the
#: two replay files. The diagnostic is what a test failure says was reached.
FORBIDDEN_SOURCES = (
    (client, "fetch_nearest_tle", "the TLE endpoint"),
    (client, "fetch_nearest_omm", "the OMM endpoint"),
    (orbit, "TextOrbitCache", "the managed orbit cache"),
    (satchecker_client, "resolve_orbits", "the client resolver"),
    (satchecker_client, "read_extra_orbit_dir", "the directory scan"),
    (satchecker_client, "search_satellites", "the catalogue search"),
    (tle_module, "check_satellite_visibilibities", "the visibility search"),
)


def forbid_orbit_acquisition(monkeypatch, tmp_path=None) -> None:
    """Every source in :data:`FORBIDDEN_SOURCES`, made to raise.

    Raising rather than answering nothing: a replay that quietly resolved a satellite
    from the cache or the service would still produce a plausible simulation.
    """
    if tmp_path is not None:
        monkeypatch.setenv("ORBIT_CACHE_DIR", str(tmp_path / "empty-cache"))
    for module, attribute, diagnostic in FORBIDDEN_SOURCES:
        monkeypatch.setattr(module, attribute, forbidden(diagnostic))


#: Every client function tabsim must look up on the package module at call time.
#: A module-scope ``from satchecker_client import ...`` would bind it at import,
#: make it unpatchable, and for the endpoints freeze the archive choice.
CLIENT_SEAM_NAMES = (
    "resolve_orbits",
    "read_extra_orbit_dir",
    "save_orbits_for_reuse",
    "save_replay_orbits",
    "load_replay_orbits",
)


class Spy:
    """One client function, recorded and then called.

    Wraps the real implementation by default: a spy that swallowed the call would
    pass against an adapter that does nothing useful.
    """

    def __init__(self, name, target):
        self.name = name
        self.target = target
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.target(*args, **kwargs)

    @property
    def call(self) -> tuple:
        assert len(self.calls) == 1, (
            f"expected exactly one call to satchecker_client.{self.name}, got "
            f"{len(self.calls)}"
        )
        return self.calls[0]

    def argument(self, position: int, name: str):
        """The *name* argument of the single call, given or passed positionally."""
        args, kwargs = self.call
        if len(args) > position:
            return args[position]
        assert name in kwargs, (
            f"satchecker_client.{self.name} was called without {name}"
        )
        return kwargs[name]


def spy_on(monkeypatch, name: str, answer=None) -> Spy:
    """Record calls to ``satchecker_client.<name>``; *answer* replaces the real one."""
    assert name in CLIENT_SEAM_NAMES, f"{name} is not one of the adopted seams"
    spy = Spy(name, getattr(satchecker_client, name) if answer is None else answer)
    monkeypatch.setattr(satchecker_client, name, spy)
    return spy


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

    Two antennas, two time steps and one channel: enough for ``load_obs`` to build a
    real observation, little enough that a test asserting *when* something is
    validated does not pay for a simulation. The rest is ``sim_config_base.yaml``.
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

    ``client._http_get`` is what the public :func:`search_satellites` uses; nothing
    else is patched, so a reintroduced private bypass hits the suite's network block
    instead of keeping the "public search" test passing.
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

    Keys are the query as it goes out on the wire — the catalogue is upper case and
    matched case-sensitively, so ``"NAVSTAR"``, not ``"navstar"``. A value is a list
    of :func:`search_row` rows or an exception; calls record ``(name, url)``.
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


def reject_json_constant(name):
    """Refuse the non-standard JSON literals ``json`` accepts by default.

    ``json.loads`` reads a bare ``NaN`` happily, so parsing with the defaults
    cannot show a replay file contains none.
    """
    raise AssertionError(f"replay file carries the non-standard JSON literal {name}")


#: Replay directories written by tab-sim e3d957d (PR #44), with what #44's own
#: loader read back out of each. See ``tests/compat/fixtures/PROVENANCE.txt``.
COMPAT_FIXTURE_DIR = Path(__file__).resolve().parent / "compat" / "fixtures"


def compat_fixture(name: str):
    """One frozen #44 replay directory and the pair #44 read back from it.

    Returns ``(directory, expected)``, *expected* carrying the checksum policy the case
    needs, the saved IDs in saved order, and the records spelled the way
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
