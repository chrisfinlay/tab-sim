# Lazy visibility noise: parent/candidate measurements

Parent: `a134653` (merged #60). End-to-end candidate: `9165283`.
Isolated-noise candidate and corrected measurement adapter: `58103d0`.
The noise implementation is unchanged in the later report/test fixes.
Five alternating fresh-process pairs per size, using the same interpreter,
dependencies, fixture and harness on each host. End-to-end runs have five warm
rounds per process, one Dask worker, 16 MB target chunks and full Zarr output.
These are separate CPU/GPU host comparisons, not a CPU-versus-GPU speedup claim.

Noise RNG samples and the radiometer scale intentionally change. The normal
benchmark comparator remains strict; `noise_comparison` explicitly permits only
noisy/calibrated visibility and flag values to differ, checks their shapes and
checks fixed astronomical/RFI signal samples across checkouts. Every run still
checks its own cold/warm output repeatability. Statistical tests independently
check the noise distribution, component/block independence and corrected scale.

## Assessment

Noise-only end-to-end medians improved 6.9–9.2% on mini and 5.6–8.9% on gpu,
with no slower paired run in these cases. The mixed-source control improved
4.3%/4.8%, below the rollout's 5% material-speed threshold; treat those as modest
or neutral under that gate. The 960 MiB isolated noise case reduces sampled peak
RSS by 83.2%, establishing the allocation/scaling benefit independently of the
end-to-end timing claim. Full-pipeline peak RSS also falls in the measured cases.
Other eager allocations remain, so this does not prove whole-pipeline out-of-core
execution or justify raising the existing stress-case memory caps.

End-to-end timings include both the lazy RNG and the requested equation correction.
The isolated RNG experiment keeps its channel scales identical in both checkouts.

## End-to-end observation to Zarr

Times below are medians of the five process medians; delta is the ratio of those
medians. The paired range shows all five individual percentage changes, including
regressions. RSS is the median whole-process sampled peak, including cold/warm
and diagnostics, not an isolated noise-allocation measurement. Graph build includes
source additions and `calculate_vis`, not just RNG graph creation.

| Host/case | Parent s | Candidate s | Time delta | Paired delta range | Peak RSS MiB parent → candidate | Graph build s parent → candidate |
|---|---:|---:|---:|---:|---:|---:|
| mini/aa1-noise-512 | 0.2978 | 0.2771 | -6.9% | -8.5% to -6.9% | 968.6 → 937.3 | 0.0424 → 0.0163 |
| mini/aa1-noise-2048 | 1.0441 | 0.9554 | -8.5% | -10.5% to -7.7% | 1966.1 → 1896.8 | 0.1212 → 0.0211 |
| mini/aa1-noise-8192 | 3.9135 | 3.5538 | -9.2% | -9.8% to -8.8% | 3032.5 → 2803.1 | 0.4202 → 0.0308 |
| mini/aa1-long | 1.8984 | 1.8173 | -4.3% | -6.0% to -0.2% | 2064.4 → 2047.3 | 0.1413 → 0.0406 |
| gpu/aa1-noise-512 | 0.8860 | 0.8360 | -5.6% | -7.4% to -4.0% | 1390.2 → 1356.5 | 0.1494 → 0.0594 |
| gpu/aa1-noise-2048 | 2.8194 | 2.5680 | -8.9% | -9.2% to -8.2% | 1892.5 → 1770.4 | 0.3817 → 0.0727 |
| gpu/aa1-noise-8192 | 10.7467 | 9.8712 | -8.1% | -11.2% to -7.3% | 2310.8 → 1819.1 | 1.2630 → 0.1058 |
| gpu/aa1-long | 4.8971 | 4.6639 | -4.8% | -5.2% to -3.9% | 2102.6 → 1903.8 | 0.4665 → 0.1578 |

## Isolated CPU noise scaling

Only `add_noise` plus channel-mean/second-moment reductions are measured here,
with AA1's 120 baselines, 16 channels, `(256,120,16)` chunks and one worker.
This separates the allocation improvement from other full-pipeline allocations.
The eager parent's returned ndarray is adapted through verified zero-copy chunk
views; adapter time is recorded and included in total. The API's own allocations
are retained. All statistical checks passed. Five fresh-process alternating pairs
per size; peak RSS is sampled every 5 ms and can miss brief peaks.

| Logical noise MiB | Construction s parent → candidate | RSS added at construction MiB parent → candidate | Peak RSS MiB parent → candidate | Peak delta | Construction + adapter + consumption s parent → candidate |
|---|---:|---:|---:|---:|---:|
| 60 | 0.1602 → 0.0025 | 135.2 → 0.3 | 380.3 → 279.8 | -26.4% | 0.1803 → 0.1246 |
| 240 | 0.4573 → 0.0056 | 495.4 → 0.3 | 740.3 → 320.9 | -56.6% | 0.5198 → 0.2974 |
| 960 | 1.6610 → 0.0190 | 975.2 → 0.5 | 2091.6 → 351.0 | -83.2% | 1.9015 → 0.9913 |

The compressed JSON reports retain individual observations, source revisions,
round statistics and available provenance. The end-to-end reports also contain
hardware, dependency versions, shapes/chunks, cold/phase times, output bytes,
RSS/NVML/JAX allocator measurements, and task/compile diagnostics. Read them with
`gzip -dc filename.json.gz` or Python's `gzip.open`.

Reproduce using the commands in [the benchmark guide](../../README.md#noise-migration-measurements-50).
Earlier trial measurements were superseded during review and are not included.

## Validation and evidence

- 40 fresh end-to-end processes and 20 successful comparisons per host.
- 30 fresh isolated-noise processes, all distribution checks passed.
- 37 targeted noise, harness and source-configuration tests passed on each host.
- Direct MS and Zarr-to-MS AA1 mixed smoke benchmarks passed on both hosts,
  including cold/warm repeatability and both Zarr/MS output validation.
- CI: 419 passed, 8 skipped on each of Python 3.10, 3.11 and 3.13. The skips are
  optional benchmark checks whose dependencies are absent in the CI environment;
  those checks run on the benchmark hosts.
- Independent review found and removed an adapter-only parent array copy before
  the final isolated measurements. The second review was clear.

Raw reports: [CPU pipeline](mini-e2e.json.gz), [GPU pipeline](gpu-e2e.json.gz),
[CPU noise scaling](mini-scaling.json.gz).
