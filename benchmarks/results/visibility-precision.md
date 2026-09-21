# Single-precision visibilities: measurements and capacity

Parent: `abebdee` (PRs #69–#71 squash-merged). Candidate: PR #72. Raw results,
versions, source hashes, output samples and per-stage durations:
[visibility-precision.json](visibility-precision.json).

## Workloads and method

| Case | Antennas / baselines | Time × channels | Fine samples | Point / RFI sources | Fixed time × frequency tile | One visibility cube, double → single |
|---|---:|---:|---:|---:|---:|---:|
| AA2, source-heavy | 68 / 2,278 | 16 × 32 | 3 | 8 / 512 | 8 × 32 | 17.80 → 8.90 MiB |
| AA4, storage-heavy | 512 / 130,816 | 8 × 64 | 3 | 8 / 2 | 1 × 8 | 0.998 → 0.499 GiB |
| AA4, capacity | 512 / 130,816 | 32 × 544 | 3 | 8 / 2 | 1 × 8 | single: 16.967 GiB |

Each normal comparison has five isolated parent-double/candidate-single pairs,
with AB/BA order alternating and one full warm-up write before each timed write.
Both sides retain the same default full output schema (RFI amplitude ancillary
arrays omitted). There are two component workers and one admitted GPU block.
Nominal chunk budgets are 45 MB for AA2 and 64 MB for AA4, unchanged between
precisions. mini uses its external SSD scratch; GTX 1060 uses local disk scratch;
Daint uses `$SCRATCH` in the debug partition with one exclusive GPU allocation.
Normal timing includes construction, components and the complete staged write;
validation reads and deletion are outside timing. These are API completion times
without a forced filesystem sync, not durable physical disk-throughput measurements. No competing task benchmarks
ran on the same machine. The supervisor enforces RSS/timeout and rejects missing,
invalid or numerically mismatched results.

The working-tree source hashes identify each measured snapshot. Final dtype
metadata/API cleanup and test corrections followed the early warm runs; the
completed implementation is covered by double controls and the capacity run.
No speed claim is made from the one-pair double controls.

## Warm runtime and stored bytes

Seconds are median ± median absolute deviation (MAD), not confidence intervals.
A change below 5% is treated as neutral here, and variation must also be considered.
GB is decimal stored bytes, including all retained arrays and metadata.

| Machine / case | Double seconds | Single seconds | Time reduction | Store GB, double → single |
|---|---:|---:|---:|---:|
| mini · CPU · 512 RFI | 11.728 ± 0.071 | 11.821 ± 0.301 | -0.8% | 0.099 → 0.054 |
| GTX 1060 · GPU · 512 RFI | 19.373 ± 0.012 | 19.081 ± 0.007 | 1.5% | 0.099 → 0.054 |
| Daint GH200 · GPU · 512 RFI | 13.703 ± 0.028 | 13.800 ± 0.414 | -0.7% | 0.099 → 0.054 |
| mini · CPU · AA4 | 12.752 ± 0.268 | 7.857 ± 0.051 | 38.4% | 5.110 → 2.530 |
| Daint GH200 · GPU · AA4 | 24.127 ± 4.317 | 16.560 ± 0.923 | 31.4% | 5.110 → 2.530 |

Daint AA4 double timings ranged from 19.81 to 38.18 s, versus 15.36 to 17.82 s
in single mode; every matched pair improved, but the magnitude varies.

AA4 gains are substantial; the 512-RFI workload is neutral. This is consistent
with less visibility data to compose, encode and write while RFI beam, geometry,
amplitude generation and phase evaluation remain double precision. It does not
isolate physical disk bandwidth from compression, copies or arithmetic.

## Memory

Peak host RSS below is the maximum supervisor observation across the five
processes, including warm-up and validation. GPU live allocation is JAX's
`peak_bytes_in_use`; it is **not** the preallocated pool or NVML process memory.
The GPU runs retain the existing 50% JAX preallocation setting.

| Machine / case | Peak host GiB, double → single | Peak live JAX MiB, double → single |
|---|---:|---:|
| mini · CPU · 512 RFI | 1.606 → 1.566 | N/A |
| GTX 1060 · GPU · 512 RFI | 2.549 → 2.458 | 417.38 → 417.38 |
| Daint GH200 · GPU · 512 RFI | 3.966 → 3.917 | 417.38 → 417.38 |
| mini · CPU · AA4 | 1.030 → 1.468 | N/A |
| Daint GH200 · GPU · AA4 | 2.910 → 2.821 | 85.03 → 64.11 |

There is no general promise of halved peak memory. mini AA4 host RSS actually
increased in these measurements. Double workspaces, allocator/cache retention
and concurrent stages remain; the cause of that RSS increase was not isolated.
The working-set estimator deliberately keeps its conservative double allowance.
The source-heavy GPU live peak stays near 417 MiB in both modes.

## Beyond-host-RAM capacity

**PASS** on mini (16 GiB host RAM): a 16.967 GiB
complex64 `vis_obs` array, 304.28 s cold wall time, peak RSS
1.902 GiB, final store 17.252 GB.
All component stages and composition completed in one Zarr store. Sparse samples
validated the component/gain/noise equation before scratch pruning, and reopened
final samples matched exactly. This checks a complete larger-than-RAM write,
not every output value against an independently computed full cube.

Capacity timing includes sparse validation before component pruning and is not
comparable to warm speed figures. Only `vis_obs` was retained; this is not an
all-output capacity claim. The temporary store was removed after validation.

The previous 127.75 GiB Daint case was reviewed but not rerun in this PR.
At complex64 the same dimensions produce 63.875 GiB; preserving 127.75 GiB would
require twice as many channels/tiles. That is a separate long capacity test,
not evidence supplied by these smaller Daint timing runs. The successful mini
case is now eligible for benchmark figures; it remains cold selected-output
capacity, separately labelled from repeated full-output performance.

## Numerical validation and review

Both precision modes retain double geometry and phase/trigonometry. Scientific
tests cover all four source families, uneven mapped chunks, very large phases,
submeter RFI path differences on a large common offset, 256 cancelling sources,
and 1,024 sources spanning nine orders of magnitude in flux. Gain/noise dtype,
seeded draws, staged Zarr, MS descriptors/readback, and CLI precedence are checked.
Near-null relative accuracy is not promised; cancellation errors are assessed
against total input flux. High dynamic range may require the double option.

Double-mode controls on mini, gpu, daint matched all six sampled products exactly (including gains and calibrated visibilities).

Maximum sampled visibility/noise difference between modes across recorded regular runs: 5.22e-06 Jy. This is a fixture result, not a universal error bound.

Validation: 673 offline CPU tests passed on mini (one network test deselected),
94 targeted GPU tests passed, and 68 mapped/scientific tests passed after the
legacy double-tolerance regression was made explicitly double. The original CI
failure was ~one float32 ULP under a float64 tolerance; the strict double test
was preserved and single precision remains separately tested. Core/API and
benchmark review loops found no remaining blocking findings after fixes.

## Reproduction

See [precision configuration and driver usage](../../docs/visibility-precision.md).
For capacity, use `--case capacity --channels 544 --candidate-only --cold
--rounds 1 --timeout 1200`; for compatibility use `--candidate-precision double`.
These commands use public simulation APIs and retain reports after removing
validated scratch stores. Source hashes and dependency versions in the JSON
are part of the comparison; do not mix output precision or chunk layouts in
historical speed plots. Older benchmark fixtures remain explicitly double.
