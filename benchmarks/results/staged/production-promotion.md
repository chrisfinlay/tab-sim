# Production promotion validation

The shared writer now powers CLI and `Observation` output by default. These
checks validate the production API changes; the older timing tables retain their
original source revisions and must not be presented as new-revision timings.

- mini, CPU / Dask 2024.10.0: **540 passed**, 1 live-network test deselected
  (full offline test suite, 41.45 s).
- mini, CPU / Dask 2026.8.0: **101 passed** (staged writer/API, benchmark harness
  and visibility-noise tests, 16.12 s).
- gpu, CUDA / Dask 2026.8.0: **56 passed** (staged writer and production API,
  43.00 s). GPU allocator reservation was configured to 50% before JAX startup.
  These are small numerical/integration tests, not a capacity or speed result.
- Two subagent review rounds; overwrite overlap and cross-worker failure
  propagation findings resolved and covered by regressions; final review had no
  outstanding blockers.

Coverage includes selected output equivalence, disabled flags, exact/near-unity
gains, seed zero, encodings, component concurrency/barriers, bounded composition
read-ahead, incomplete failures, task callbacks, host-budget rejection, modified
core-array rejection, backing-store overlap protection, CLI overrides, real
MS-only/combined export, and mocked accumulation/statistics before pruning.
The live SatChecker integration test remains in the suite; it was not removed
or weakened and is unrelated to this writer change.

## Capacity reassessment

At this round, mini has 16 GiB physical RAM and 93.01 GiB free on the external SSD.
The existing `aa4-host-out-of-core` fixture has 512 stations, 32 times, 512 channels
and a **31.94 GiB** complex128 visibility. Its retained-all benchmark planning
estimate at 64 MB / 2 workers is 5.70 GiB host working memory and 193.75 GiB scratch.
That conservative benchmark estimate is for retained-all output, not the new
production selection default.

Selecting only `vis_obs` still requires the astronomical, RFI and noise cubes to
coexist with observed visibility: **127.75 GiB** before gains, coordinates,
metadata, encoding headroom and the free-space reserve. Consequently even this
selection cannot fit the currently available scratch. No new attempt was made,
and this case is not eligible for successful timing figures yet. RFI source
count does not remove that four-cube scratch lower bound; the same limitation
applies to the fixed 512-RFI workload. The cancelled large GPU/HDD run was not
restarted.

The previously completed 7.984 GiB CPU/GPU/Daint runs remain evidence at their
recorded revisions. A full output larger than host RAM has still not completed
validation on mini; this PR does not claim otherwise. Production memory guards
check between tasks and cannot make an oversized individual kernel safe.

## Why omit RFI amplitude output by default?

In the recorded Daint 512-RFI run, total time was 1420.51 s, RFI visibility took
655.79 s and the subsequent `rfi_stat_A` write took 657.43 s. That latter stage
re-evaluated/rechunked amplitudes and wrote 24 GiB uncompressed. Omitting it avoids
that entire optional stage, while RFI visibility still computes the amplitudes
it needs. Subtracting the stage suggests about 46% less wall time in that run;
this is an estimate, **not a measured new-default speedup**. Further repeated
performance runs should retain the 512-RFI workload and record selected outputs.
