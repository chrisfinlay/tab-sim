# Baseline measurements — 19 September 2026

These are measurements of the unchanged simulator following PR #41. The pure-JAX visibility kernels are unchanged; installed `ri-kernels` versions
in metadata do not mean the optional kernel PR was used. The primary
size sweep ran at harness commit `cab18df`; later commits repair cross-checkout
pytest isolation and add trace postprocessing. Every record contains its exact
revision and harness/source hashes. There is **no optimization speedup claim**.

Each host ran all eight normal cases, five warm single-iteration rounds per mode
after a cold call, x64, one Dask worker, and a nominal 16 MB chunk target.
Each host completed 21 applicable size-sweep measurements; three kernel/case
combinations with no corresponding sources were explicitly skipped.

CPU: `mini`, Apple Silicon, 16 GiB RAM. GPU: `gpu`, GTX 1060 6 GB.
The machines and Python/dependency versions differ; compare a future PR against
its own host baseline, not these columns as an accelerator speedup ratio.

## Warm observation-to-Zarr medians

Times include observation/graph construction, gains/noise and the full default
Zarr output. They exclude Python/pytest startup, validation and deletion.

| Case | mini CPU median (s) | IQR (s) | gpu GPU median (s) | IQR (s) |
|---|---:|---:|---:|---:|
| aa05-point | 0.0820 | 0.0008 | 0.3261 | 0.0041 |
| aa1-mixed | 0.1800 | 0.0020 | 0.6140 | 0.0182 |
| aa2-rfi | 0.6158 | 0.0038 | 1.5221 | 0.0060 |
| phase1-mixed | 0.5234 | 0.0054 | 1.1879 | 0.0086 |
| aastar-rfi | 0.7546 | 0.0138 | 1.7791 | 0.0131 |
| aa4-mixed | 1.1720 | 0.0086 | 2.8551 | 0.0233 |
| aa1-many-sources | 0.2334 | 0.0055 | 0.6372 | 0.0067 |
| aa1-long | 1.9020 | 0.0442 | 4.8993 | 0.0368 |

## Memory and output paths

RSS peaks across the size sweep were approximately 0.60–1.96 GiB on mini and
1.19–2.05 GiB on gpu. On gpu, JAX reported about 36–103 MiB peak live allocator
use for Zarr cases while NVML showed roughly 4.5 GiB process allocation. The
difference is the pool/reservation, not evidence that every task needs 4.5 GiB.
Memory sampling includes untimed validation/diagnostics and may miss short spikes.

Both direct MS and Zarr-to-MS completed on the AA1 mixed fixture on both hosts.
This is a bounded reproduction result, not proof that every failure in #42 is fixed.

| Host | Direct MS total median (s) | Zarr then MS total median (s) |
|---|---:|---:|
| mini | 0.2513 | 0.3246 |
| gpu | 1.9734 | 2.1908 |

Both stress fixtures were skipped on both hosts by preflight. AA4 out-of-core
has an 8.57 GB single visibility cube and an 86.7 GB conservative host plan,
exceeding the 4 GiB budget. These records demonstrate safe refusal, **not**
successful out-of-core execution. No OOM was deliberately triggered.

## Comparison validation

Five alternating AB/BA pairs on mini compared merged main `a3a2980` with the
benchmark branch, using the same harness and environment. All ten kernel runs
passed, and all five comparisons accepted the source metadata and NumPy/output
signatures. The implementation is unchanged, so these are runner validation
results and a noise illustration, not a performance improvement.

## GPU transfer trace

A separate AA1 mixed Zarr run at `aaa43ad` used reduced host tracing and no
Python tracing. It contains 11,129 events, below the million-event warning:

| Observed direction | Events | Bytes with captured size |
|---|---:|---:|
| Host → device | 310 | 48,304,218 |
| Device → host | 62 | 52,546,950 |
| Device → device | 224 | 45,478,568 |

These are measured events from a diagnostic run, not inferred task counts.
The compressed trace and SHA256-linked summary are included. Trace collectors
can still drop events; durations can overlap. An initial default-options trace
hit a million events and was replaced, rather than treated as a complete census.

## Reproduce and inspect

See [the benchmark guide](../README.md). JSON files retain configuration,
dependency versions, cold/phase timings, warm statistics, diagnostic counts,
memory statistics and bounded numerical signatures. Raw timing logs and temporary simulation outputs are not checked in; one small
compressed GPU trace is retained for auditing. Source files retain the pinned SKAO
provenance from PR #41. GPU kernel and end-to-end results are separate modes.
