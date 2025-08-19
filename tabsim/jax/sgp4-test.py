import numpy as np
import jax.numpy as jnp
from typing import Tuple, NamedTuple

from tabsim.tle import get_tles_by_id

from astropy.time import Time

JD = 2460860.987072
# JD = 2460538.395853102


class SGP4Constants:
    """SGP4 physical and mathematical constants from Vallado 2023"""

    # WGS-84 constants
    WGS84_A = 6378137.0  # Semi-major axis in meters
    WGS84_F = 1.0 / 298.257223563  # Flattening
    WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2  # Eccentricity squared

    # Gravitational parameter (m³/s²)
    MU = 3.986004418e14

    # Earth's angular velocity (rad/s)
    OMEGA_E = 7.2921158553e-5

    # Conversion factors
    DEG2RAD = jnp.pi / 180.0
    RAD2DEG = 180.0 / jnp.pi

    # SGP4 specific constants (Vallado 2023)
    # KE is the conversion factor between mean motion and canonical units
    KE = 0.0743669161331734132  # Exact value from Vallado
    QOMS2T = 1.88027916e-9  # (Q0 - S)^4 ERI^4
    S = 1.01222928  # S* ERI
    XJ3 = -0.00000253215306  # J3 harmonic
    XJ4 = -0.00000161098761  # J4 harmonic
    XJ2 = 0.00108262998905  # J2 harmonic

    # Atmospheric drag constants
    AE = 1.0  # Earth radius in canonical units
    XKMPER = 6378.135  # Earth radius in km

    # Conversion from minutes to canonical time units
    MINUTES_PER_DAY = 1440.0


class TLE(NamedTuple):
    """Two-Line Element set representation"""

    name: str
    line1: str
    line2: str

    # Parsed elements
    epoch_year: int
    epoch_day: float
    ndot: float  # First derivative of mean motion
    nddot: float  # Second derivative of mean motion
    bstar: float  # Ballistic coefficient
    inclo: float  # Inclination (radians)
    nodeo: float  # Right ascension of ascending node (radians)
    ecco: float  # Eccentricity
    argpo: float  # Argument of perigee (radians)
    mo: float  # Mean anomaly (radians)
    no_kozai: float  # Mean motion (radians/minute)
    epoch_jd: float  # Epoch as Julian date


class SGP4State(NamedTuple):
    """SGP4 internal state variables"""

    # Orbital elements
    a: float  # Semi-major axis
    e: float  # Eccentricity
    i: float  # Inclination
    omega: float  # Argument of perigee
    Omega: float  # Right ascension of ascending node
    M: float  # Mean anomaly
    n: float  # Mean motion

    # Drag terms
    bstar: float
    ndot: float
    nddot: float

    # Epoch
    epoch_jd: float


def parse_tle(line1: str, line2: str, name: str = "") -> TLE:
    """Parse a two-line element set into SGP4 parameters"""

    def parse_exponential(s: str) -> float:
        """Parse exponential notation like '-12345-3' -> -0.12345e-3"""
        s = s.strip()
        if len(s) == 0:
            return 0.0

        # Handle the exponential format
        if s[-2] in "+-":
            mantissa = s[:-2]
            exponent = s[-2:]
            if mantissa[0] not in "+-":
                mantissa = "+" + mantissa
            return float(mantissa) * 10.0 ** int(exponent)
        else:
            return float(s)

    # Parse line 1
    epoch_year = int(line1[18:20])
    if epoch_year < 57:
        epoch_year += 2000
    else:
        epoch_year += 1900

    epoch_day = float(line1[20:32])
    ndot = float(line1[33:43])
    nddot = parse_exponential(line1[44:52])
    bstar = parse_exponential(line1[53:61])

    # Parse line 2
    inclo = float(line2[8:16]) * SGP4Constants.DEG2RAD
    nodeo = float(line2[17:25]) * SGP4Constants.DEG2RAD
    ecco = float("0." + line2[26:33])
    argpo = float(line2[34:42]) * SGP4Constants.DEG2RAD
    mo = float(line2[43:51]) * SGP4Constants.DEG2RAD
    no_kozai = float(line2[52:63])

    # Convert mean motion to radians/minute
    no_kozai = no_kozai * 2.0 * jnp.pi / SGP4Constants.MINUTES_PER_DAY

    # Calculate epoch Julian date
    epoch_jd = jd_from_year_day(epoch_year, epoch_day)

    return TLE(
        name=name,
        line1=line1,
        line2=line2,
        epoch_year=epoch_year,
        epoch_day=epoch_day,
        ndot=ndot,
        nddot=nddot,
        bstar=bstar,
        inclo=inclo,
        nodeo=nodeo,
        ecco=ecco,
        argpo=argpo,
        mo=mo,
        no_kozai=no_kozai,
        epoch_jd=epoch_jd,
    )


def sgp4_init(tle_params: Tuple) -> SGP4State:
    """Initialize SGP4 state from TLE parameters"""
    (inclo, nodeo, ecco, argpo, mo, no_kozai, ndot, nddot, bstar, epoch_jd) = tle_params

    # Convert mean motion from radians/minute to revolutions/day for semi-major axis calculation
    no_revs_per_day = no_kozai * SGP4Constants.MINUTES_PER_DAY / (2.0 * jnp.pi)

    # Calculate semi-major axis using Kepler's third law
    # n = sqrt(mu/a^3) => a = (mu/n^2)^(1/3)
    # Convert to canonical units where Earth radius = 1
    a = (SGP4Constants.KE / no_kozai) ** (2.0 / 3.0)

    return SGP4State(
        a=a,
        e=ecco,
        i=inclo,
        omega=argpo,
        Omega=nodeo,
        M=mo,
        n=no_kozai,
        bstar=bstar,
        ndot=ndot,
        nddot=nddot,
        epoch_jd=epoch_jd,
    )


def jd_from_year_day(year: int, day: float) -> float:
    """Convert year and day of year to Julian date"""
    # Calculate Julian date for January 1 of given year
    a = (14 - 1) // 12
    y = year + 4800 - a
    m = 1 + 12 * a - 3
    jd_jan1 = 1 + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045

    # Add the day of year (accounting for the fact that day 1 is January 1)
    return jd_jan1 + day - 1.0 - 0.5


from tabsim.jax.coordinates import solve_kepler


tles_df = get_tles_by_id(
    "chrisfinlay93@gmail.com",
    "Breath4-Precision3-Goatskin9-Modulator1",
    [
        1291,
    ],
    JD,
    window_days=1,
    tle_dir="./tles",
)

orbit_elements = jnp.atleast_2d(
    tles_df[
        [
            "SEMIMAJOR_AXIS",
            "ECCENTRICITY",
            "INCLINATION",
            "RA_OF_ASC_NODE",
            "ARG_OF_PERICENTER",
            "MEAN_ANOMALY",
        ]
    ].values
)

epoch_jd = tles_df["EPOCH_JD"].values

tles = tles_df[["TLE_LINE1", "TLE_LINE2"]].values


tle = parse_tle(tles[0, 0], tles[0, 1])


# print(f"Self parse :   {tle.epoch_jd}")
# print(f"tabsim :       {epoch_jd[0]}")

times_jd = JD + np.linspace(0, 1.0 / 24, 2)

from tabsim.tle import get_satellite_positions, get_satellite_positions_kepler
from tabsim.jax.coordinates import kepler_orbit_many

sat_pos = get_satellite_positions(tles, times_jd)[0] / 1e3

kep_pos1 = get_satellite_positions_kepler(tles, times_jd)[0] / 1e3

kep_pos2 = (
    kepler_orbit_many(times_jd, np.array([tle.epoch_jd]), orbit_elements)[0] / 1e3
)

# print(sat_pos)
# print(kep_pos1)
# print(kep_pos2)
print()
print(np.linalg.norm(sat_pos - kep_pos1, axis=-1))
print(np.linalg.norm(sat_pos - kep_pos2, axis=-1))

# print(orbit_elements)
