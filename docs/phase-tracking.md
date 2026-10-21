# Phase tracking

The visibilities are fringe-stopped to the phase centre's **apparent place of
date**, not to its J2000 coordinates. The J2000 centre is still what you give in
the config and what the output records as the target direction.

## Why the two differ

The phase-tracking term rotates the antenna positions into a frame whose `w`
axis points at the phase centre, using the centre's Greenwich hour angle,
`GAST - RA`. GAST is measured from the true equator and equinox *of date*, so
the RA and Dec it is paired with have to be on that equator and equinox too.

A J2000 RA/Dec is not. Pairing them leaves out frame bias, precession, nutation
and annual aberration: 0.248 degrees in RA and -0.070 degrees in Dec for a 2019
observation, growing by about 50 arcsec a year from J2000. tab-sim did pair them
up to and including v0.0.13, which displaced every modelled RFI source by that
angle — about 15 m of extra path on MeerKAT's longest baseline, or 63
wavelengths at L band.

Astronomical sources are unaffected. Their `lmn` are offsets from the phase
centre, and the sky model and the centre are both J2000, so a rotation applied
to both leaves the offsets unchanged.

## What is applied

`tabsim.time.apparent_radec_of_date` takes the J2000 (ICRS) centre to its
apparent geocentric place at each fine time step:

1. annual aberration, up to 20.5 arcsec, from pyerfa's analytic `epv00` series —
   no ephemeris file, so it works on offline compute nodes;
2. then skyfield's bias-precession-nutation matrix, which is the same rotation
   skyfield's ITRS-to-GCRS transform uses for the antenna positions, so the
   tracked direction and those positions share one frame exactly.

Diurnal aberration (at most 0.32 arcsec, and common to the whole array) and
gravitational light deflection (milliarcseconds away from the Sun) are not
applied. The centre is a direction at infinity, so there is no parallax.

The sidereal angle is skyfield's GAST, for the same reason: it is the rotation
inside the ITRS-to-GCRS transform, so the tracking term and the RFI distances
measured from the antenna positions cannot drift apart.

## What the output records

The zarr attribute `phase_tracking` names the convention a run used:

| value | meaning |
|-------|---------|
| `apparent` | the apparent place of date, as above |

`target_ra` and `target_dec`, and the Measurement Set's `FIELD::PHASE_DIR`,
remain the J2000 centre. The tracked direction is not recorded separately
because it follows from those two and the convention, through
`apparent_radec_of_date`.

Simulations from v0.0.13 and earlier carry no `phase_tracking` attribute, and
used the J2000 pairing. tabascal reads such data with its
`tabsim: {phase_tracking: j2000}` setting.

## References

- tab-sim [#73](https://github.com/chrisfinlay/tab-sim/issues/73)
- tabascal [#252](https://github.com/epfl-radio-astro/tabascal/issues/252), with
  a real-data reproducer: correcting the tracked direction took the
  matched-filter detection of a Galileo satellite from -0.3 sigma to 92.8 sigma.
