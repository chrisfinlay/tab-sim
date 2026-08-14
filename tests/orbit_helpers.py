"""Shared fixtures for the orbit-record tests: real records, no network."""

from __future__ import annotations

from tabsim.satchecker.records import KIND_FIELD, KIND_OMM, KIND_TLE
from tabsim.satchecker.tle_parse import parse_tle_elements
from tabsim.satchecker._time import jd_to_datetime


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


def omm_record_from_tle(
    norad_id=ISS_NORAD_ID, line1=ISS_LINE1, line2=ISS_LINE2, **extra
):
    """An OMM record carrying the *same* elements as the given TLE pair.

    Deriving it from a TLE rather than writing numbers by hand is what makes the
    two propagation paths directly comparable: any unit error in the OMM branch
    of :func:`tabsim.tle.earth_satellite` shows up as a position difference
    against a satellite whose elements are known to be identical.

    The epoch goes out at microsecond resolution, which is all an ISO 8601
    ``EPOCH`` field carries.
    """
    elements = parse_tle_elements(line1, line2)
    return {
        "NORAD_CAT_ID": norad_id,
        KIND_FIELD: KIND_OMM,
        "OBJECT_NAME": "TEST SAT",
        "OBJECT_ID": "1998-067A",
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
