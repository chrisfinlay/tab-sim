# Native RFI accumulation after single precision

PR #47 is stacked on PR #72 (`8503a0b`). This comparison isolates native RFI
accumulation: both parent and candidate use single precision, the same staged
single-store writer, and identical output selection. It does not compare against
the old monolithic writer or double-precision default.

## Workload and method

512 stationary RFI sources, 8 astronomical point sources, SKA-Low AA2 with 68
antennas / 2,278 baselines, 8 coarse times and 32 channels. Only integration
samples vary: 3, 9, 33, 99, 199 and 299. The nominal chunk ceiling stays at 45 decimal MB;
selected coarse-time × channel tiles are respectively 8×32, 4×32, 1×32, 1×8, 1×4 and 1×4.
Both branches have identical chunks within each comparison. One final visibility
cube is 4.45 MiB: these source-heavy timing cases are not capacity tests.

At 3, 9 and 33 samples, five isolated AB/BA pairs alternate parent/candidate order.
At 99, 199 and 299 samples, completed comparisons have one exploratory pair.
For 99–299 samples, Daint runs each side in a separate debug allocation on the same physical node;
strict provenance and numeric checks are applied after both finish. Each process
performs a complete warm-up before its timed construction and staged write. Two
component streams and one admitted GPU block are used. Default saved arrays are
retained; RFI amplitude ancillary arrays remain omitted. All six sampled numeric
products must agree within documented single-precision tolerances, including an
independent sampled component-composition identity. A candidate must prove the
native backend is usable; fallback cannot count as a native benchmark.

mini uses external SSD scratch, GTX 1060 local disk scratch, and Daint GH200
`$SCRATCH` on the debug partition. Benchmarks on each machine run sequentially.
Timings are API completion, without forced filesystem sync, and include source
preparation and writes. They are not isolated kernel speed or durable disk
throughput. Reports preserve software versions, source hashes and native binary
hashes (including the GTX 1060 SM6.1 build). Compare within a machine, not across
unmatched software/hardware environments.

The 3/9/33-sample performance snapshots precede the planner's explicit native operand floor.
That floor leaves the default scratch multipliers (4 on CPU, 6 on GPU) and these selected
chunks unchanged. Final production code is covered by CI and the capacity
recheck. RSS is externally sampled process-tree RSS including startup/warm-up;
live GPU memory is JAX's process-lifetime peak allocation, separately from NVML
allocator reservations. Neither metric is an exact per-operator workspace bound.

## End-to-end results

Completed runs were about 20–29% shorter on mini and 8–16% shorter on GTX. Daint’s end-to-end change remained around 0–1%; the five-pair differences overlap run-to-run variation, and the larger single pairs do not establish a small speedup. The isolated kernels improved much more, so remaining pipeline work limits the overall benefit.

All times are seconds. Lower is better. Five-pair rows report medians; one-pair rows are exploratory, without a statistical confidence claim.

| Host | Integration samples | Pairs | JAX | Native | Time reduction | Peak host RSS GiB, JAX → native |
|---|---:|---:|---:|---:|---:|---:|
| mini (M4 CPU) | 3 | 5 | 5.970 | 4.542 | 23.9% | 1.50 → 1.47 |
| mini (M4 CPU) | 9 | 5 | 19.438 | 13.889 | 28.5% | 2.02 → 2.17 |
| mini (M4 CPU) | 33 | 5 | 63.997 | 51.425 | 19.6% | 2.99 → 2.73 |
| mini (M4 CPU) | 99 | 1 | 195.580 | 142.409 | 27.2% | 2.42 → 3.32 |
| mini (M4 CPU) | 199 | 1 | 421.848 | 299.300 | 29.1% | 5.80 → 4.08 |
| mini (M4 CPU) | 299 | 1 | 597.682 | 453.587 | 24.1% | 6.08 → 7.43 |
| GTX 1060 | 3 | 5 | 9.728 | 8.933 | 8.2% | 2.16 → 2.20 |
| GTX 1060 | 9 | 5 | 28.246 | 25.945 | 8.1% | 3.23 → 3.24 |
| GTX 1060 | 33 | 5 | 115.203 | 96.416 | 16.3% | 4.94 → 4.91 |
| GTX 1060 | 99 | 1 | 336.669 | 287.578 | 14.6% | 9.12 → 9.43 |
| Daint GH200 | 3 | 5 | 7.817 | 7.782 | 0.4% | 3.33 → 3.58 |
| Daint GH200 | 9 | 5 | 18.794 | 18.745 | 0.3% | 4.58 → 4.67 |
| Daint GH200 | 33 | 5 | 64.732 | 64.268 | 0.7% | 6.23 → 6.33 |
| Daint GH200 | 99 | 1 | 187.889 | 187.285 | 0.3% | 10.16 → 10.53 |
| Daint GH200 | 199 | 1 | 384.699 | 380.178 | 1.2% | 17.30 → 17.28 |
| Daint GH200 | 299 | 1 | 565.956 | 562.365 | 0.6% | 24.75 → 24.84 |

GTX 199 did not complete under the 16 GiB host guard on either implementation, so it has no end-to-end timing or speedup. GTX 299 was cancelled by the user. The corresponding failures and all per-stage times, five-pair spreads, source/native-binary hashes and numeric samples are retained in [raw results](native-rfi-precision.json).

## Isolated resident-input kernels

Single precision; five synchronized alternating repetitions per implementation. These use synthetic inputs at the chosen compute-tile shape, not the complete simulation.

| Host | Samples | Tile time × channels | JAX ms | Native ms | Speedup |
|---|---:|---:|---:|---:|---:|
| mini (M4 CPU) | 3 | 8 × 32 | 1943.751 | 315.587 | 6.16× |
| mini (M4 CPU) | 9 | 4 × 32 | 3918.762 | 590.835 | 6.63× |
| mini (M4 CPU) | 33 | 1 × 32 | 2369.048 | 486.057 | 4.87× |
| mini (M4 CPU) | 99 | 1 × 8 | 1998.222 | 383.381 | 5.21× |
| mini (M4 CPU) | 199 | 1 × 4 | 2216.551 | 384.524 | 5.76× |
| mini (M4 CPU) | 299 | 1 × 4 | 3355.872 | 589.029 | 5.70× |
| GTX 1060 | 3 | 8 × 32 | 916.220 | 115.437 | 7.94× |
| GTX 1060 | 9 | 4 × 32 | 1387.370 | 251.615 | 5.51× |
| GTX 1060 | 33 | 1 × 32 | 2455.862 | 134.527 | 18.26× |
| GTX 1060 | 99 | 1 × 8 | 1229.082 | 109.489 | 11.23× |
| GTX 1060 | 199 | 1 × 4 | 1093.459 | 122.301 | 8.94× |
| Daint GH200 | 3 | 8 × 32 | 13.131 | 3.784 | 3.47× |
| Daint GH200 | 9 | 4 × 32 | 18.850 | 5.680 | 3.32× |
| Daint GH200 | 33 | 1 × 32 | 30.267 | 4.312 | 7.02× |
| Daint GH200 | 99 | 1 × 8 | 18.618 | 3.325 | 5.60× |
| Daint GH200 | 199 | 1 × 4 | 17.771 | 3.474 | 5.11× |
| Daint GH200 | 299 | 1 × 4 | 26.444 | 5.081 | 5.20× |

At 33 samples, the native single-vs-double control used the same tile: mini 0.486 vs 0.828 s (1.70×), GTX 0.135 vs 0.546 s (4.06×). Geometry and phasor evaluation remain double precision. These controls isolate resident-kernel behavior and are not full-pipeline single-vs-double measurements.

## GPU utilisation and allocation

NVML sampled once per second on the GPU that held the benchmark allocation. This reports device busy time, not SM occupancy. Short bursts can be missed; a 95th percentile of zero is compatible with occasional high peaks. The logs below cover different sample counts/hardware and are not a matched utilisation comparison.

| Host / scope | Samples | Mean busy | p95 | Peak | Zero samples |
|---|---:|---:|---:|---:|---:|
| GTX 1060 | 1108 | 1.095% | 0% | 91% | 97.2% |
| Daint GH200 | 1158 | 0.145% | 0% | 27% | 97.6% |

- **GTX 1060:** 199-sample native attempt, startup/warm-up/timed work until the host guard stopped it; incomplete simulation. [Raw NVML log](native-rfi-gtx199-utilization.csv.gz).

- **Daint GH200:** 299-sample native job, including startup, warm-up, timed write, validation and final short kernel diagnostic. [Raw NVML log](native-rfi-daint299-utilization.csv.gz).

Earlier completed PR #47 runs did not record utilisation time series. The later 65-second Daint baseline window had 0% reported GPU activity throughout; GTX had 0% in 27 active samples before its host guard stopped it. Those snapshots cannot establish whole-run averages. Full-job logs also contain startup/host work; do not interpret reserved VRAM as computation.

JAX preallocates half the GPU memory in these runs. Daint therefore reports roughly 48 GiB reserved even though live JAX peaks are far smaller. Across the matched 3/9/33 GPU cases, native live peaks rise from 417→525, 626→779 and 574→711 MiB respectively. Per-case live peaks are in the raw report; host RSS can rise while these tile-sized allocations stay bounded.

## Capacity, guards and interpretation

An external RSS guard aborts a benchmark; it does not change simulation chunks or
force a particular working-set size. Initial Daint limits were too conservative
for high integration counts and caused avoidable stops. The raw report preserves those stops and
the successful retry guard settings. The current Daint policy uses a 128 GiB cap
with an available-memory check. The GTX host has only 23 GiB physical RAM and keeps
its separate 16 GiB cap. A guard stop is not evidence of CUDA OOM or physical RAM
exhaustion.

The isolated kernel probes use synthetic resident amplitudes/distances and 5
synchronized alternating repetitions, with full-array flux-normalized numerical
checks. They exclude upstream geometry, beam formation, Dask, transfers and
storage; their speedups must not be substituted for full-pipeline improvement.
Probe shapes follow the actual tile shapes and therefore change across sample
counts. Double controls at 33 samples isolate the native precision effect.

The capacity recheck is a different, two-RFI-source AA4 workload: 512 antennas,
32 times, 544 channels, 3 integration samples. Its 16.967 GiB final vis_obs cube
exceeds mini's 16 GiB physical RAM; it passed in 335.319 s with 1.831 GiB sampled peak
host RSS. Components were validated before cleanup, and only vis_obs retained.
This establishes that case's out-of-core write, not a host-memory bound for the
512-source timing cases.

The GPU tile model remained bounded, but host RSS increased substantially with
integration samples. Those peaks cover the complete warm worker lifecycle and
are not native-kernel memory measurements. Frequency-independent geometry grows
with integration samples even when frequency tiles shrink. Dask host caches,
ancillary rechunking/recomputation and runtime allocation retention are additional
candidates; no dominant cause is established. Stage/cache/RSS attribution is
required before recommending a targeted fix.

The successful Daint 299 baseline retained its 32 GiB external guard; the native job used the corrected 128 GiB policy. Neither threshold changes chunks or numerical work. The original 24 GiB attempts both stopped and are preserved. On GTX, 99 samples completed after the cap was raised from 8 to 16 GiB; 199 samples hit 16 GiB on both routes. Those are benchmark abort thresholds, not demonstrated physical-machine capacity limits.

## Reproduction and validation

Use `benchmarks.precision_sweep` with `--base-precision single --candidate-precision single --samples N --times 8 --case rfi --require-native`; retain the 45 MB fixture chunk limit. Use five rounds for 3/9/33 and one for 99/199/299. GPU cases use `--device gpu`; Daint high-sample cases use sequential debug jobs on the same node with one checkout per job, then the driver’s strict `compare` function. `benchmarks.rfi_kernel_probe` reproduces the resident-input diagnostics with the tile dimensions above.

Native integration and benchmark drivers passed independent review loops. CI passed on Python 3.10/3.11/3.13; the final mini offline suite passed 707 tests (one network test deselected), and selected GTX tests passed 107. All completed timing pairs pass strict workload/provenance/numeric checks. All isolated probes pass full-array flux-normalized error checks. No numerical pass is claimed for a guard-stopped simulation.
