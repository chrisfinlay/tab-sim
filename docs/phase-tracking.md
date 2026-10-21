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

## The sky model stays in J2000, the basis rolls to meet it

It is tempting to argue that astronomical sources are unaffected, since `lmn`
are offsets from the phase centre and rotating centre and source alike leaves
them unchanged. That is wrong: `radec_to_lmn` builds its tangent basis from the
pole of the frame it is handed, and the J2000 pole is not the true pole of date.

So `w` is pointed at the apparent place, but `u` and `v` are then **rolled**
about it, by `tabsim.time.tracking_basis_roll`, so that they are the J2000 east
and north carried rigidly onto the tracked frame. J2000 direction cosines pair
with them exactly. The roll leaves `w` untouched, so the RFI fringe-stopping
term does not depend on it.

Completing the basis from the of-date pole instead would leave the two bases
rolled apart by that angle:

| phase centre declination | roll |
|---|---|
| near the equator | 0.08 deg |
| -45 | 0.11 deg |
| -89 | 5.1 deg |
| -89.9 | 72 deg |

It is small away from the pole and enormous beside it, where the J2000 and
of-date poles are 0.145 deg apart and the field sits between them. Ignoring it
costs 0.5 wavelengths of path for a source 2 deg off axis near the equator, 1.0
at -45 and 500 for a polar field, and it would rotate an extended source
outright.

Rolling the basis rather than transforming the sky keeps three things simple:
source positions are still given *and* recorded in J2000; a position angle
measured from J2000 north stays a position angle, with no correction; and the
direction cosines stay genuinely time independent, so nothing has to be frozen
at a reference time.

What remains is differential aberration: the whole field is carried by the phase
centre's aberration, so a source at the edge keeps a residual of (v/c) times its
offset, which is 0.035 wavelengths at 2 deg on a 4 km baseline at L band. That
is deliberately not modelled -- tabascal's astronomical model makes the same
uniform assumption, and standard imaging ignores the effect.

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

Two zarr attributes name the conventions the geometry was built with:

| attribute | value | meaning |
|-----------|-------|---------|
| `phase_tracking` | `apparent` | the apparent place of date, as above |
| `geometry_time_scale` | `ut1` | the scale the epochs behind the geometry are read on |

The direction convention alone does not pin the geometry down, which is why the
second is there. tab-sim reads its epochs as UT1, in `tabsim.time.skyfield_time`
and everywhere else it calls skyfield, and the Measurement Set's `TIME` column
carries those same numbers. Reading them as UTC instead moves the Earth by DUT1
— currently under 0.2 s, about 0.07 wavelengths of `w` at L band on a 4 km
baseline — and every satellite along its track by the distance it covers in that
time. So a consumer that selects `apparent` but reads `TIME` as UTC still
reconstructs a different geometry. That mismatch predates the tracking fix and
is tracked in
[#79](https://github.com/chrisfinlay/tab-sim/issues/79); recording the scale at
least makes it a stated convention rather than an implied one.

`target_ra` and `target_dec`, and the Measurement Set's `FIELD::PHASE_DIR`,
remain the J2000 centre. The tracked direction is not recorded separately
because it follows from those, the convention and the times, through
`apparent_radec_of_date`.

Simulations from v0.0.13 and earlier carry no `phase_tracking` attribute, and
used the J2000 pairing. tabascal reads such data with its
`tabsim: {phase_tracking: j2000}` setting.

## References

- tab-sim [#73](https://github.com/chrisfinlay/tab-sim/issues/73)
- tabascal [#252](https://github.com/epfl-radio-astro/tabascal/issues/252), with
  a real-data reproducer: correcting the tracked direction took the
  matched-filter detection of a Galileo satellite from -0.3 sigma to 92.8 sigma.
