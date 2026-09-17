"""Write the frozen PR #44 replay fixtures under ``tests/compat/fixtures/``.

Run **once**, on the Linux test host, against tab-sim
``e3d957d7846b98617cc65874ef6d4c71bd8d2377`` — that is, before any of the
adoption lands — because the whole value of these files is that #44's own writer
produced them:

    python tests/compat/generate_pr44_fixtures.py

Nothing in the default suite runs this: the committed output *is* the fixture,
and regenerating it with the adopted code would turn the cross-version test into
a comparison of the new code with itself. It is kept beside the fixtures so the
provenance can be checked rather than believed.

Each case is a directory holding the two files a completed #44 run writes —
``norad_ids.yaml`` through the same ``numpy.savetxt`` call
``tabsim.config.save_inputs`` uses, and ``used_orbits.json`` through
``tabsim.orbit.save_orbits_for_reuse`` — plus ``expected.json``, which is what
#44's own ``load_replay_orbits`` reads back out of them under the checksum
policy the case needs. The cases cover what the format has to carry: an empty
selection, each kind alone, both mixed row orders with the IDs deliberately not
in ascending order, provider and fetch provenance on every record, the two OMM
values that a rounding writer or an imprecise parser gets wrong, and a record
accepted without its checksum digits.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tests"))

import satchecker_client  # noqa: E402
from tabsim import orbit  # noqa: E402

from orbit_helpers import (  # noqa: E402
    GPS_EPOCH_JD,
    GPS_NORAD_ID,
    ISS_EPOCH_JD,
    ISS_NORAD_ID,
    omm_record_at,
    tle_record_at,
    without_checksum,
)

#: The revisions these fixtures were produced by. Recorded rather than derived:
#: the checkout is rsynced to the test host without its history.
TABSIM_SHA = "e3d957d7846b98617cc65874ef6d4c71bd8d2377"
CLIENT_SHA = "bb7027042b6ed6f5f76049201335d3cdc1dd1c06"

FIXTURES = HERE / "fixtures"

#: Provenance every non-empty record carries, so the fixtures prove the provider
#: and fetch fields survive a round trip rather than merely the elements.
PROVENANCE = {"DATA_SOURCE": "spacetrack", "FETCHED_AT": "2026-08-13T09:15:00+00:00"}

#: The two OMM values the replay format exists to protect. ``DataFrame.to_json``
#: writes 0.0066635 as 0.006663499999999999 even at its maximum precision, and
#: satchecker-client before 0.1.2 read 3.2e-05 back as 3.2000000000000005e-05.
#: Either is a different double, and so a different trajectory.
AWKWARD_ECCENTRICITY = 0.0066635
AWKWARD_BSTAR = 3.2e-05


def unverified_tle_record(norad_id, epoch_jd, **extra):
    """A TLE whose lines arrived without their checksum digits, as the archive serves some."""
    record = tle_record_at(norad_id, epoch_jd, **extra)
    record["TLE_LINE1"] = without_checksum(record["TLE_LINE1"])
    record["TLE_LINE2"] = without_checksum(record["TLE_LINE2"])
    return record


def omm_with_awkward_elements(norad_id, epoch_jd, **extra):
    record = omm_record_at(norad_id, epoch_jd, **extra)
    record["ECCENTRICITY"] = AWKWARD_ECCENTRICITY
    record["BSTAR"] = AWKWARD_BSTAR
    return record


#: ``name -> (norad_ids, records, allow_missing_checksum)``. The IDs in the two
#: mixed cases are deliberately descending: a writer or reader that sorts them
#: would still pass a test whose IDs were already in order.
CASES = {
    "empty": ([], [], False),
    "tle_verified": (
        [ISS_NORAD_ID],
        [tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD, **PROVENANCE)],
        False,
    ),
    "omm": (
        [GPS_NORAD_ID],
        [omm_with_awkward_elements(GPS_NORAD_ID, GPS_EPOCH_JD, **PROVENANCE)],
        False,
    ),
    "mixed_tle_first": (
        [GPS_NORAD_ID, ISS_NORAD_ID],
        [
            tle_record_at(GPS_NORAD_ID, GPS_EPOCH_JD, **PROVENANCE),
            omm_with_awkward_elements(ISS_NORAD_ID, ISS_EPOCH_JD, **PROVENANCE),
        ],
        False,
    ),
    "mixed_omm_first": (
        [GPS_NORAD_ID, ISS_NORAD_ID],
        [
            omm_with_awkward_elements(GPS_NORAD_ID, GPS_EPOCH_JD, **PROVENANCE),
            tle_record_at(ISS_NORAD_ID, ISS_EPOCH_JD, **PROVENANCE),
        ],
        False,
    ),
    "unverified": (
        [ISS_NORAD_ID],
        [unverified_tle_record(ISS_NORAD_ID, ISS_EPOCH_JD, **PROVENANCE)],
        True,
    ),
}


def json_ready(value):
    """One cell as it will be compared: NumPy unwrapped, every null spelled ``None``."""
    item = getattr(value, "item", None)
    if item is not None and getattr(value, "shape", ()) == ():
        value = item()
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_case(name, norad_ids, records, allow_missing_checksum):
    directory = FIXTURES / name
    directory.mkdir(parents=True, exist_ok=True)

    # Exactly what tabsim.config.save_inputs writes at e3d957d, both files.
    np.savetxt(
        directory / "norad_ids.yaml", np.asarray(norad_ids, dtype=int), fmt="%i"
    )
    orbit.save_orbits_for_reuse(
        directory / "used_orbits.json", list(norad_ids), list(records)
    )

    read_ids, read_records = orbit.load_replay_orbits(
        directory, allow_missing_checksum=allow_missing_checksum
    )
    expected = {
        "allow_missing_checksum": allow_missing_checksum,
        "norad_ids": [int(nid) for nid in read_ids],
        "records": [
            {key: json_ready(value) for key, value in record.items()}
            for record in read_records
        ],
    }
    (directory / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return expected


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (norad_ids, records, allow_missing_checksum) in CASES.items():
        summary[name] = write_case(name, norad_ids, records, allow_missing_checksum)
        print(f"wrote {name}: {summary[name]['norad_ids']}")

    lines = [
        "Frozen PR #44 replay fixtures",
        "=============================",
        "",
        f"tab-sim            : {TABSIM_SHA} (branch satchecker-orbits, PR #44)",
        f"satchecker-client  : {CLIENT_SHA} (branch orbit-resolver, PR #5)",
        f"satchecker-client __version__: {satchecker_client.__version__}",
        f"python             : {sys.version.split()[0]}",
        f"numpy              : {np.__version__}",
        "",
        "Generated by, on the Linux test host, with tabsim/ still at the SHA above:",
        "",
        "    python tests/compat/generate_pr44_fixtures.py",
        "",
        "Each case directory holds the two files a completed #44 run writes —",
        "norad_ids.yaml (numpy.savetxt, fmt='%i', as tabsim.config.save_inputs",
        "writes it) and used_orbits.json (tabsim.orbit.save_orbits_for_reuse) —",
        "and expected.json, the (norad_ids, records) pair #44's own",
        "tabsim.orbit.load_replay_orbits reads back from them, with every null",
        "spelled None and NumPy scalars unwrapped.",
        "",
        "Cases and their expected projected records:",
        "",
    ]
    for name, expected in summary.items():
        lines.append(f"{name}:")
        lines.append(f"  allow_missing_checksum = {expected['allow_missing_checksum']}")
        lines.append(f"  norad_ids = {expected['norad_ids']}")
        for record in expected["records"]:
            kept = {
                key: value for key, value in sorted(record.items()) if value is not None
            }
            lines.append(f"  record {kept['NORAD_CAT_ID']} ({kept['RECORD_KIND']}):")
            for key, value in kept.items():
                lines.append(f"    {key} = {value!r}")
        lines.append("")
    lines += [
        "These files are inputs, not outputs: do not regenerate them with the",
        "adopted code. Doing so would turn the cross-version test into a",
        "comparison of the new implementation with itself.",
        "",
    ]
    (FIXTURES / "PROVENANCE.txt").write_text("\n".join(lines))
    print(f"wrote {FIXTURES / 'PROVENANCE.txt'}")


if __name__ == "__main__":
    main()
