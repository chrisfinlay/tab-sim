import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from typing import Tuple, NamedTuple
import re
from datetime import datetime, timezone

# Enable double precision for accurate orbital calculations
jax.config.update("jax_enable_x64", True)


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


def jd_from_year_day(year: int, day: float) -> float:
    """Convert year and day of year to Julian date"""
    # Calculate Julian date for January 1 of given year
    a = (14 - 1) // 12
    y = year + 4800 - a
    m = 1 + 12 * a - 3
    jd_jan1 = 1 + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045

    # Add the day of year (accounting for the fact that day 1 is January 1)
    return jd_jan1 + day - 1.0 - 0.5


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


@jit
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


@jit
def kepler_solve(M: float, e: float, max_iter: int = 10) -> float:
    """Solve Kepler's equation using Newton-Raphson method"""
    E = M  # Initial guess

    def body_fun(carry):
        E, _ = carry
        f = E - e * jnp.sin(E) - M
        fp = 1.0 - e * jnp.cos(E)
        dE = -f / fp
        E_new = E + dE
        return E_new, dE

    def cond_fun(carry):
        _, dE = carry
        return jnp.abs(dE) > 1e-12

    E_final, _ = jax.lax.while_loop(cond_fun, body_fun, (E, 1.0))
    return E_final


@jit
def orbital_to_cartesian(
    a: float, e: float, E: float, i: float, omega: float, Omega: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert orbital elements to Cartesian position and velocity"""
    # Position and velocity in orbital plane
    cos_E = jnp.cos(E)
    sin_E = jnp.sin(E)

    # Position in orbital plane (in Earth radii)
    x_orb = a * (cos_E - e)
    y_orb = a * jnp.sqrt(1.0 - e * e) * sin_E

    # Distance from center
    r = a * (1.0 - e * cos_E)

    # Velocity in orbital plane (canonical units)
    # Standard orbital mechanics in SGP4 canonical units
    beta = jnp.sqrt(1.0 - e * e)
    r = a * (1.0 - e * cos_E)  # Current radius in Earth radii

    # Velocity components in orbital frame (canonical units)
    # Using KE which is sqrt(mu) in canonical units
    v_scale = SGP4Constants.KE * jnp.sqrt(1.0 / r)

    vx_orb = -v_scale * sin_E
    vy_orb = v_scale * beta * cos_E

    # Rotation matrices for ECI transformation
    cos_omega = jnp.cos(omega)
    sin_omega = jnp.sin(omega)
    cos_Omega = jnp.cos(Omega)
    sin_Omega = jnp.sin(Omega)
    cos_i = jnp.cos(i)
    sin_i = jnp.sin(i)

    # Transform to ECI frame
    # R = R_z(-Omega) * R_x(-i) * R_z(-omega)
    r11 = cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i
    r12 = -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i
    r13 = sin_Omega * sin_i

    r21 = sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i
    r22 = -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i
    r23 = -cos_Omega * sin_i

    r31 = sin_omega * sin_i
    r32 = cos_omega * sin_i
    r33 = cos_i

    # Position in ECI (Earth radii)
    x = r11 * x_orb + r12 * y_orb
    y = r21 * x_orb + r22 * y_orb
    z = r31 * x_orb + r32 * y_orb

    # Velocity in ECI (canonical units)
    vx = r11 * vx_orb + r12 * vy_orb
    vy = r21 * vx_orb + r22 * vy_orb
    vz = r31 * vx_orb + r32 * vy_orb

    # Convert to meters and m/s
    # Position: Earth radii to meters
    position = jnp.array([x, y, z]) * SGP4Constants.XKMPER * 1000.0

    # Velocity: Convert from canonical units to m/s
    # This is where we need to figure out the correct conversion factor
    velocity = jnp.array([vx, vy, vz]) * SGP4Constants.XKMPER * 1000.0 / 60.0

    return position, velocity


@jit
def sgp4_propagate_single(
    state: SGP4State, t_since_epoch_min: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Propagate a single satellite state using SGP4"""

    # Time since epoch in minutes
    t = t_since_epoch_min

    # Secular effects of atmospheric drag and gravitation
    # Simplified model - for full precision, implement all SGP4 perturbations

    # Mean motion with drag (always apply, even if terms are zero)
    n = state.n + state.ndot * t + 0.5 * state.nddot * t * t

    # Mean anomaly
    M = state.M + n * t
    M = M % (2.0 * jnp.pi)  # Wrap to [0, 2π]

    # Solve Kepler's equation
    E = kepler_solve(M, state.e)

    # For this minimal implementation, we'll use the basic orbital elements
    # without the full SGP4 perturbations for simplicity while maintaining precision

    # Convert to Cartesian coordinates
    position, velocity = orbital_to_cartesian(
        state.a,  # Already in canonical units (Earth radii)
        state.e,
        E,
        state.i,
        state.omega,
        state.Omega,
    )

    return position, velocity


@jit
def sgp4_propagate_batch(
    state: SGP4State, times_since_epoch_min: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Propagate satellite state for multiple times using SGP4"""

    def propagate_single_time(t):
        return sgp4_propagate_single(state, t)

    # Vectorize over time
    positions, velocities = jax.vmap(propagate_single_time)(times_since_epoch_min)

    return positions, velocities


def sgp4_propagate(
    tle: TLE, julian_dates: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Main SGP4 propagation function

    Args:
        tle: Two-Line Element set
        julian_dates: Array of Julian dates for propagation

    Returns:
        positions: Array of positions in ECI frame (meters) [N, 3]
        velocities: Array of velocities in ECI frame (m/s) [N, 3]
    """

    # Convert TLE to parameters for JAX
    tle_params = (
        tle.inclo,
        tle.nodeo,
        tle.ecco,
        tle.argpo,
        tle.mo,
        tle.no_kozai,
        tle.ndot,
        tle.nddot,
        tle.bstar,
        tle.epoch_jd,
    )

    # Initialize SGP4 state
    state = sgp4_init(tle_params)

    # Convert Julian dates to minutes since epoch
    times_since_epoch_min = (
        julian_dates - tle.epoch_jd
    ) * SGP4Constants.MINUTES_PER_DAY

    # Propagate
    positions, velocities = sgp4_propagate_batch(state, times_since_epoch_min)

    return positions, velocities


def debug_sgp4_calculation(tle: TLE, julian_date: float):
    """Debug function to show intermediate SGP4 calculation values"""

    # Convert TLE to parameters for JAX
    tle_params = (
        tle.inclo,
        tle.nodeo,
        tle.ecco,
        tle.argpo,
        tle.mo,
        tle.no_kozai,
        tle.ndot,
        tle.nddot,
        tle.bstar,
        tle.epoch_jd,
    )

    # Initialize SGP4 state
    state = sgp4_init(tle_params)

    # Time since epoch
    t_since_epoch_min = (julian_date - tle.epoch_jd) * SGP4Constants.MINUTES_PER_DAY

    print(f"=== SGP4 Debug for t = {t_since_epoch_min:.1f} minutes ===")
    print(f"Semi-major axis: a = {state.a:.6f} Earth radii")
    print(f"Eccentricity: e = {state.e:.6f}")
    print(f"Mean motion: n = {state.n:.6f} rad/min")

    # Mean motion with drag
    n = (
        state.n
        + state.ndot * t_since_epoch_min
        + 0.5 * state.nddot * t_since_epoch_min * t_since_epoch_min
    )
    print(f"Corrected mean motion: n = {n:.6f} rad/min")

    # Mean anomaly
    M = state.M + n * t_since_epoch_min
    M = M % (2.0 * jnp.pi)
    print(f"Mean anomaly: M = {M:.6f} rad = {M * 180/jnp.pi:.2f}°")

    # Solve Kepler's equation
    E = kepler_solve(M, state.e)
    cos_E = jnp.cos(E)
    sin_E = jnp.sin(E)
    print(f"Eccentric anomaly: E = {E:.6f} rad = {E * 180/jnp.pi:.2f}°")
    print(f"cos(E) = {cos_E:.6f}, sin(E) = {sin_E:.6f}")

    # Orbital radius and velocity
    r = state.a * (1.0 - state.e * cos_E)
    beta = jnp.sqrt(1.0 - state.e * state.e)
    v_scale = SGP4Constants.KE * jnp.sqrt(1.0 / r)

    print(
        f"Current radius: r = {r:.6f} Earth radii = {r * SGP4Constants.XKMPER:.1f} km"
    )
    print(f"Beta: β = {beta:.6f}")
    print(f"Velocity scale: v_scale = {v_scale:.6f} canonical units")

    # Velocity components
    vx_orb = -v_scale * sin_E
    vy_orb = v_scale * beta * cos_E
    orbital_speed = jnp.sqrt(vx_orb**2 + vy_orb**2)

    print(f"Orbital velocity components:")
    print(f"  vx_orb = {vx_orb:.6f} canonical units")
    print(f"  vy_orb = {vy_orb:.6f} canonical units")
    print(f"  Orbital speed = {orbital_speed:.6f} canonical units")

    # Convert to physical units
    speed_km_s = orbital_speed * SGP4Constants.XKMPER / 60.0
    print(f"  Orbital speed = {speed_km_s:.3f} km/s")

    return E, cos_E, sin_E, r, v_scale, orbital_speed


# Example usage and testing
if __name__ == "__main__":
    # Example TLE for ISS
    tle_line1 = "1 25544U 98067A   24001.50000000  .00001234  00000-0  23439-4 0  9990"
    tle_line2 = "2 25544  51.6400 339.7760 0003350  83.2789 276.9150 15.49297658123456"

    tle = parse_tle(tle_line1, tle_line2, "ISS")

    tle_params = (
        tle.inclo,
        tle.nodeo,
        tle.ecco,
        tle.argpo,
        tle.mo,
        tle.no_kozai,
        tle.ndot,
        tle.nddot,
        tle.bstar,
        tle.epoch_jd,
    )

    state = sgp4_init(tle_params)

    # # Debug the first calculation (epoch)
    # E, cos_E, sin_E, r, v_scale, orbital_speed = debug_sgp4_calculation(
    #     tle, tle.epoch_jd
    # )

    # print("\n" + "=" * 50)

    # Test propagation for various time intervals
    epoch_jd = tle.epoch_jd
    test_times = jnp.array(
        [
            epoch_jd + 0.0,  # Epoch (0 hours)
            epoch_jd + 0.041666666667,  # 1 hour later (1/24)
            # epoch_jd + 0.25,  # 6 hours later (6/24)
            # epoch_jd + 0.5,  # 12 hours later (12/24)
            # epoch_jd + 0.75,  # 18 hours later (18/24)
            # epoch_jd + 1.0,  # 1 day later
            # epoch_jd + 7.0,  # 1 week later
        ]
    )

    # print(f"Debug - Epoch JD: {epoch_jd}")
    # print(f"Debug - Test times: {test_times}")
    # print(f"Debug - Time differences: {test_times - epoch_jd}")

    positions, velocities = sgp4_propagate(tle, test_times)

    from skyfield.api import load, EarthSatellite

    ts = load.timescale()
    sf_times = ts.ut1_jd(np.array(test_times))
    sat_pos = jnp.array(
        EarthSatellite(tle_line1, tle_line2, ts=ts).at(sf_times).position.m.T
    )
    sat_vel = jnp.array(
        EarthSatellite(tle_line1, tle_line2, ts=ts).at(sf_times).velocity.m_per_s.T
    )

    from tabsim.jax.coordinates import R_z, R_x, kepler_orbit

    # kep_pos = kepler_orbit(
    #     test_times,
    #     tle.epoch_jd,
    #     (state.a, state.e, *np.rad2deg([state.i, state.omega, state.Omega, state.M])),
    # )

    from tabsim.tle import get_satellite_positions_kepler

    kep_pos = get_satellite_positions_kepler([[tle_line1, tle_line2]], test_times)[0]

    # print(positions / 1e3)
    # print(sat_pos / 1e3)
    # print(kep_pos / 1e3)
    print()
    print(jnp.linalg.norm(sat_pos - kep_pos, axis=1) / 1e3)
    print(jnp.linalg.norm(sat_pos - positions, axis=1) / 1e3)
    # print(jnp.linalg.norm(kep_pos, axis=1) / 1e3)

    # print()
    # print(velocities / 1e3)
    # print(sat_vel / 1e3)
    # print(jnp.linalg.norm(velocities, axis=1) / 1e3)
    # print(jnp.linalg.norm(sat_vel, axis=1) / 1e3)

    # print("SGP4 Propagation Results:")
    # print("=" * 70)
    # time_labels = [
    #     "Epoch",
    #     "+1 hour",
    #     "+6 hours",
    #     "+12 hours",
    #     "+18 hours",
    #     "+1 day",
    #     "+7 days",
    # ]
    # time_offsets = [0.0, 1.0, 6.0, 12.0, 18.0, 24.0, 168.0]

    # for i, (pos, vel, label, hours) in enumerate(
    #     zip(positions, velocities, time_labels, time_offsets)
    # ):
    #     print(f"{label} ({hours:.1f}h): JD {test_times[i]:.6f}")
    #     print(f"  Position (m): [{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}]")
    #     print(f"  Velocity (m/s): [{vel[0]:.1f}, {vel[1]:.1f}, {vel[2]:.1f}]")
    #     print(f"  Altitude (km): {(jnp.linalg.norm(pos) - 6378137.0) / 1000.0:.1f}")
    #     print(f"  Speed (km/s): {jnp.linalg.norm(vel) / 1000.0:.3f}")
    #     print()
