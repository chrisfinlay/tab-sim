# Direct mapped JAX callbacks: paired measurements

Immediate parent: `a2d94c1` (merged #61). Candidate: `723e8c0`.
Cumulative runs use original baseline `a134653` and candidate `b2c65dc`, whose
production code is identical to `723e8c0`.

Each comparison uses the same harness and dependency environment on its host,
one Dask worker, five alternating fresh-process pairs, five warm rounds per
process, and full Zarr output. mini is the CPU host; gpu is the GTX 1060 6 GB host.
CPU and GPU results are independent host comparisons, not accelerator speed ratios.
Immediate-parent comparisons require matching samples of all five products,
including noisy/calibrated data and flags. Every run checks cold/warm repeatability.
Those bounded signatures complement full-array primitive-equivalence unit tests;
they are not a whole-simulation proof of bitwise equality.

## End-to-end timings

Values are medians of five process medians, with percentage change relative to the
parent. The paired range retains all five individual changes, including regressions.
Actual chunks show integrated time × frequency; all baselines remain in each block.
Chunk MB is a target for the existing heuristic, not a full working-set bound.

| Host / case / target MB | Actual t × f | Parent s | Candidate s | Change | Paired change range |
|---|---:|---:|---:|---:|---:|
| mini / aa1-mixed / 0.125 | 16 × 1 | 1.2483 | 0.9268 | -25.8% | -26.0% to -25.4% |
| mini / aa1-mixed / 16 | 64 × 32 | 0.1764 | 0.1680 | -4.7% | -6.0% to -3.3% |
| mini / aa2-rfi / 16 | 32 × 4 | 0.5706 | 0.5458 | -4.3% | -6.2% to -1.0% |
| gpu / aa1-mixed / 0.125 | 16 × 1 | 6.8374 | 4.8398 | -29.2% | -29.5% to -28.9% |
| gpu / aa1-mixed / 16 | 64 × 32 | 0.6036 | 0.5259 | -12.9% | -16.0% to -12.6% |
| gpu / aa2-rfi / 16 | 32 × 4 | 1.4068 | 1.2608 | -10.4% | -11.3% to -9.2% |

## Assessment

Small tiles improve end-to-end time by 25.8% on CPU and 29.2% on GPU,
consistently across the five pairs. Large-tile GPU controls improve 10–13%.
Large-tile CPU changes are about 4–5%, below the predefined 5% material-benefit
threshold, and are treated as neutral. The benefit is consistent with removing
per-callback scheduling work; the instrumented timings below are supporting
diagnostics, not an additive breakdown of the uninstrumented speedup.

## Separate callback diagnostics

These observations come from an additional instrumented round, excluded from the
timed repetitions. CPU time is the sum of profiled **calling-thread** CPU time
inside mapped callbacks. It includes profiler overhead; nested scheduler threads
are not profiled. Tokenization is inclusive cProfile wall time; inner-compute time
in the raw data also includes kernel dispatch/execution/waiting. It is not pure
scheduler overhead. Inclusive timings overlap and must not be added. Callback
wall/CPU times are not completed GPU kernel timings.

Callback counts are unchanged in every pair. Median outer task counts are unchanged;
a few GPU AA1 runs execute seven extra outer tasks in either revision (261–268
for large tiles and 8707–8714 for small tiles). These counts do not establish an
outer-graph reduction. Removal of inner compute and tokenization calls is measured
directly. Warm compile-log counts are zero in both revisions for these fixtures; the
results do not support a claim that repeated JIT wrapping recompiled every block.

| Host / case / target MB | Callback calls parent → candidate | Inner compute calls | Tokenize calls | Tokenize inclusive s | Profiled callback thread CPU s | Outer executed tasks | Warm compile events |
|---|---:|---:|---:|---:|---:|---:|---:|
| mini / aa1-mixed / 0.125 | 813 → 813 | 813 → 0 | 1671 → 0 | 0.4601 → 0.0000 | 0.8622 → 0.4330 | 8778 → 8778 | 0 → 0 |
| mini / aa1-mixed / 16 | 11 → 11 | 11 → 0 | 27 → 0 | 0.0116 → 0.0000 | 0.0184 → 0.0363 | 182 → 182 | 0 → 0 |
| mini / aa2-rfi / 16 | 43 → 43 | 43 → 0 | 81 → 0 | 0.0321 → 0.0000 | 0.0582 → 0.1495 | 628 → 628 | 0 → 0 |
| gpu / aa1-mixed / 0.125 | 813 → 813 | 813 → 0 | 7362 → 0 | 2.0780 → 0.0000 | 3.6161 → 2.1444 | 8707 → 8707 | 0 → 0 |
| gpu / aa1-mixed / 16 | 11 → 11 | 11 → 0 | 104 → 0 | 0.1701 → 0.0000 | 0.1927 → 0.0816 | 261 → 261 | 0 → 0 |
| gpu / aa2-rfi / 16 | 43 → 43 | 43 → 0 | 382 → 0 | 0.1985 → 0.0000 | 0.2817 → 0.3783 | 672 → 672 | 0 → 0 |

## Pipeline memory and graph construction

Peak RSS is a median sampled process peak across cold/warm/diagnostic phases;
it is not live GPU memory. NVML reservations and JAX allocator statistics remain
in the raw data. Graph construction includes source setup and `calculate_vis`.

| Host / case / target MB | Peak RSS MiB parent → candidate | RSS change | Warm graph construction s parent → candidate |
|---|---:|---:|---:|
| mini / aa1-mixed / 0.125 | 727.4 → 716.2 | -1.5% | 0.0667 → 0.0664 |
| mini / aa1-mixed / 16 | 818.6 → 809.8 | -1.1% | 0.0313 → 0.0308 |
| mini / aa2-rfi / 16 | 965.7 → 959.8 | -0.6% | 0.0307 → 0.0306 |
| gpu / aa1-mixed / 0.125 | 1342.5 → 1314.8 | -2.1% | 0.2691 → 0.2697 |
| gpu / aa1-mixed / 16 | 1323.6 → 1314.6 | -0.7% | 0.1378 → 0.1436 |
| gpu / aa2-rfi / 16 | 1314.7 → 1295.0 | -1.5% | 0.1135 → 0.1125 |

## Cumulative check against the original baseline

This is a separate controlled AA1 mixed / 16 MB experiment, not arithmetic on
historical timings. It includes #50's intentional RNG and noise-scale changes,
so the noise-aware comparator checks fixed signal samples and output shapes
rather than claiming unchanged noisy products. It is not used to attribute this
PR's individual benefit; the strict immediate-parent comparisons above do that.

| Host | Original s | Candidate s | Change | Paired change range |
|---|---:|---:|---:|---:|
| mini | 0.1799 | 0.1647 | -8.5% | -9.3% to -7.0% |
| gpu | 0.6179 | 0.5225 | -15.4% | -16.2% to -14.6% |

## Validation and provenance

- 30 successful immediate-parent processes and 15 strict comparisons per host.
- 10 cumulative processes and five noise-aware comparisons per host.
- 55 focused tests passed on mini and gpu: 18 valid mapped wrappers on synchronous
  and threaded schedulers, uneven tails, numerical/dtype agreement, forbidden
  nested compute, compilation reuse, CPU process serialization, and harness checks.
- Direct MS and Zarr-to-MS smoke checks passed on both hosts.
- Independent review fixed a collision in callback profile labels; subsequent
  review found no outstanding actionable issues.
- CI: 459 passed and eight optional benchmark-dependency skips on Python 3.10,
  3.11 and 3.13. The optional checks run on the benchmark hosts. Docs build passed.

The pre-existing ENU_to_GEO template dimension mismatch remains outside this PR;
ITRF_to_UVW tests retain its whole-antenna reference-origin semantics. No backend,
transfer, synchronization, precision, or chunk-selection policy was introduced.

The adjacent compressed JSON files contain raw round statistics, phase times,
profile counters, sample signatures, hardware/environment/dependency fingerprints,
source and harness revisions/hashes, memory and task/compilation observations.
Read them with `gzip -dc filename.json.gz` or Python's `gzip.open`.

Reproduction commands are in the [benchmark guide](../../README.md#mapped-callback-scheduling-measurements-51).
