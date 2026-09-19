# Planned SKA-Low station layouts

These six tables reproduce the coordinates and row order submitted in PR #41.
They are a pinned planned-layout snapshot, **not** a claim about today's
commissioned stations. No station coordinate has been changed during verification.

## Verified sources

- Station memberships: SKAO `ska-ost-array-config` **3.2.0**, commit
  `4e6013086b94772e00843d4ec490dbbf869e84d5`,
  [`src/ska_ost_array_config/array_assembly.py`](https://gitlab.com/ska-telescope/ost/ska-ost-array-config/-/blob/4e6013086b94772e00843d4ec490dbbf869e84d5/src/ska_ost_array_config/array_assembly.py).
  SHA256: `611da260bb65dfe1052b1eb299f831e49cf74d11a8912748e83128e1a9f88b47`.
- Geocentric station centres: SKAO `ska-low-tmdata`, commit
  `ef8f0c1dc7096066ea75d9628f6cb7f0cd6ee7c0`,
  [`tmdata/instrument/ska1_low/layout/low-layout.json`](https://gitlab.com/ska-telescope/ska-low-tmdata/-/blob/ef8f0c1dc7096066ea75d9628f6cb7f0cd6ee7c0/tmdata/instrument/ska1_low/layout/low-layout.json).
  SHA256: `2aee3f1204be139335e627cead6c4f9025994b41f2378ff49a193714d8cb70ae`.
- The reference longitude/latitude/ellipsoidal height (116.7644482 degrees,
  -26.82472208 degrees, 365 m, WGS84) is `LOW_ARRAY_REF` in
  [`array_config.py` at the same 3.2.0 commit](https://gitlab.com/ska-telescope/ost/ska-ost-array-config/-/blob/4e6013086b94772e00843d4ec490dbbf869e84d5/src/ska_ost_array_config/array_config.py).
  It is a shared telescope reference, not each station's height.

These are independently verified matching sources; the original notebook did
not record which revisions it used. Newer upstream releases change memberships
and names, so updating these tables must be an explicit scientific data change.

## Transformation and row order

The generator reads membership constants `LOW_AA05`, `LOW_AA1`, `LOW_AA2`,
`LOW_AAstar_Phase_1`, and `LOW_AAstar`. Bare numeric station IDs acquire a `C`
prefix. Labels are sorted lexicographically and deduplicated, matching
`LowSubArray`'s `numpy.unique` ordering. AA4 uses all 512 station labels, excluding
the JSON's `ARRAY-CENTRE` reference receptor; this matches the 3.2.0
`static/low_array_coords.dat` label set. Join labels to JSON `station_label`
(case-insensitive), then write geocentric x/y/z in metres with three decimal
places. No geodetic conversion or epoch adjustment is applied. The `.itrf.txt`
extension is tab-sim's coordinate-input convention; the source's frame/epoch
limitations are retained.

| Public name | Stations | File |
|---|---:|---|
| SKA-Low-AA0.5 | 4 | SKA-Low-AA0.5.itrf.txt |
| SKA-Low-AA1 | 16 | SKA-Low-AA1.itrf.txt |
| SKA-Low-AA2 | 68 | SKA-Low-AA2.itrf.txt |
| SKA-Low-AA*-Phase-1 | 108 | SKA-Low-AAstar-Phase-1.itrf.txt |
| SKA-Low-AA* | 307 | SKA-Low-AAstar.itrf.txt |
| SKA-Low-AA4 | 512 | SKA-Low-AA4.itrf.txt |

The 39 m station diameter is used with tab-sim's existing Airy beam approximation;
it is not a model of the detailed phased-array station beam. The two `AAstar`
filenames avoid literal `*` characters; public telescope names retain `AA*`.

## Reproduce from a repository checkout

Python 3's standard library is sufficient. Download the two immutable inputs
into a scratch directory (or extract these exact paths with `git show` from
local clones), then run from the tab-sim repository root:

```sh
mkdir -p /tmp/tabsim-ska-low
curl --fail --location 'https://gitlab.com/ska-telescope/ost/ska-ost-array-config/-/raw/4e6013086b94772e00843d4ec490dbbf869e84d5/src/ska_ost_array_config/array_assembly.py' -o /tmp/tabsim-ska-low/array_assembly.py
curl --fail --location 'https://gitlab.com/ska-telescope/ska-low-tmdata/-/raw/ef8f0c1dc7096066ea75d9628f6cb7f0cd6ee7c0/tmdata/instrument/ska1_low/layout/low-layout.json' -o /tmp/tabsim-ska-low/low-layout.json
python tools/generate_ska_low.py --layout /tmp/tabsim-ska-low/low-layout.json --array-assembly /tmp/tabsim-ska-low/array_assembly.py --check
```

The script verifies SHA256 before parsing; it does not execute upstream Python.
`--check` makes no changes. Omit it to regenerate, or use `--output-dir` to write
elsewhere. It replaces the exploratory notebook, whose local input and Python
state were not reproducible. SKAO software is not a simulation-time dependency.
