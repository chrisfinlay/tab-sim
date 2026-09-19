# Offline SKA-Low performance baselines

This suite addresses #49. It measures the existing simulator without changing its
algorithms. Performance work should compare each PR to its immediate parent on
the **same machine and dependency environment**. CPU and GPU reports from different
hosts are independent baselines, not a controlled accelerator speedup comparison.

## Install and run

From a checkout with the usual casacore dependencies installed:

```sh
pip install '.[test,benchmark]'       # CPU
# pip install '.[test,benchmark,gpu]' # supported NVIDIA GPU environment
python -m benchmarks.run --device cpu --output benchmark-runs/baseline
```

Requires Python 3.10+ and a POSIX host for process-group timeout/memory enforcement.
The runner imports the requested checkout and verifies `tabsim.__file__`; it does
not silently benchmark an unrelated installed package. GPU requests set
`JAX_PLATFORMS=cuda` before import and fail if CUDA cannot initialize. x64 is on;
persistent compilation caching is disabled. Existing allocator environment
settings are preserved and recorded. No production JAX flags or algorithms change.

`pytest-benchmark` pedantic mode uses exactly one operation per round, five rounds
by default, following an explicit cold call. There is no automatic repetition
calibration. The runner starts a fresh process for every case/mode/checkout and
writes raw pytest-benchmark JSON, logs, JUnit outcomes, and `summary.json`. Output
directories must be new. Full simulation products are deleted after each untimed
validation, keeping disk usage bounded. Failed/MS-only runs are reported as failures
and make the runner exit nonzero; #42 is not silently xfailed or worked around.

Ordinary `pytest` runs only `tests/`. Performance runs are opt-in:

```sh
python -m pytest benchmarks --case aa1-mixed --mode zarr --device cpu \
  --benchmark-json=benchmark.json
```

Use the **runner** for memory/timeout protection: direct pytest has preflight
checks but no supervising process. Run the small smoke command below to check the harness; shared CI hardware must
not establish performance targets. The existing test suite includes harness guard
and comparison checks (optional dependency checks skip without the benchmark extra).

```sh
python -m benchmarks.run --cases aa05-point aa1-mixed --modes zarr rfi-kernel \
  --device cpu --output benchmark-runs/smoke
```

## Recorded cases

All cases use complete named layouts from PR #41, not the first N antennas of a
larger array. Frequencies start at 150 MHz with 100 kHz spacing, integration time
is 2 s with 3 fine samples, epoch is MJD 60000, and phase centre is (30, -30) deg.
Point fluxes/positions and stationary emitters are fixed analytic fixtures in
`harness.py`; gains/noise have fixed seeds. There is no source catalogue, random
placement rejection, live orbit acquisition, IERS download or diagnostics plotting.
The benchmark worker rejects socket connections. Cases and the renderer are hashed
in reports. SKA coordinates retain the pinned provenance supplied by PR #41.

| Case | Stations | Times | Channels | Point / RFI sources |
|---|---:|---:|---:|---:|
| aa05-point | 4 | 32 | 16 | 8 / 0 |
| aa1-mixed | 16 | 64 | 32 | 8 / 2 |
| aa2-rfi | 68 | 32 | 32 | 0 / 4 |
| phase1-mixed | 108 | 16 | 16 | 16 / 4 |
| aastar-rfi | 307 | 8 | 8 | 0 / 4 |
| aa4-mixed | 512 | 4 | 8 | 8 / 2 |
| aa1-many-sources | 16 | 32 | 16 | 256 / 16 |
| aa1-long | 16 | 2048 | 16 | 8 / 2 |
| aa4-out-of-core (stress) | 512 | 32 | 128 | 8 / 2 |
| aa1-host-stress (stress) | 16 | 32768 | 32 | 8 / 2 |

```sh
python -m benchmarks.run --device gpu \
  --cases aa05-point aa1-mixed aa2-rfi phase1-mixed aastar-rfi aa4-mixed aa1-many-sources aa1-long \
  --modes zarr astro-kernel rfi-kernel --output benchmark-runs/full
python -m benchmarks.run --device cpu --cases aa1-mixed \
  --modes ms zarr-ms --output benchmark-runs/ms
python -m benchmarks.run --cases aa4-out-of-core aa1-host-stress \
  --modes zarr --plan --output benchmark-runs/plan
```

The largest stress case has an 8.57 GB (7.98 GiB) **single** complex128 visibility
cube, larger than the test GPU's 6 GiB. The current eager noise allocation prevents
claiming bounded host RAM for such runs. Stress cases are listed, estimated and
safely skipped if the host/disk plan exceeds its budget; a skip is **not** evidence
that out-of-core execution works. The host estimate includes ten visibility cubes,
source arrays and 512 MiB overhead; it is conservative, not a proven XLA bound.
Preflight limits host use to the smaller of the configured cap and half currently
available RAM, and output estimates to half free disk. The supervising runner also
terminates a process exceeding its RSS cap or timeout. A 100 ms polling interval
cannot prevent every instantaneous allocation spike.

Default caps: 4 GiB host RSS, 4 GiB isolated-kernel working-set estimate, 1200 s per
process. `--host-budget-gib`, `--gpu-budget-gib`, `--timeout`, `--chunk-mb` and
`--workers` are explicit controls. GPU budget is a preflight estimate for isolated
kernels, not a cap on JAX's allocation pool or a GPU spill mechanism. Leave room
for other users; do not raise limits just to provoke an OOM.

## What each measurement means

- **Kernel modes:** `astro-kernel` and `rfi-kernel` use at most 8 times and 16
  channels from the case size and all its stations/sources. Deterministic synthetic
  input geometry uses the selected station layout. Report host-to-device copy,
  tracing/lowering, compilation, first completed execution, warm completed calls,
  and device-to-host readback separately. Byte counts are logical array sizes;
  CPU placement/readback may alias memory rather than physically copy it. Every timed call waits with
  `block_until_ready()`. A NumPy scalar reference checks first/middle/last outputs.
  These tiles are intentionally not the full end-to-end workload.
- **Zarr:** `Observation` construction → source/gain graph → `calculate_vis()` →
  complete default dataset write. Phase timings separate geometry setup, graph
  construction (including current eager noise) and simulation plus writing.
  This excludes CLI argument parsing, config loading, logging and plotting.
- **MS:** same construction, then direct `write_ms` with no Zarr intermediary.
- **Zarr-MS:** complete Zarr write followed by MS conversion from the saved data;
  conversion time is reported separately. Total benchmark time includes both.
- **Cold/startup:** one cold call per process; kernel compile is isolated above.
  `runtime_import_s` is imports/device initialization from the test body;
  `process_wall_s` includes Python/pytest startup, all rounds, validation and
  diagnostics. It must not be mislabeled as startup or a single simulation time.
- **Memory:** sampled RSS and NVML per-process GPU allocation peak over the worker
  lifetime, plus JAX allocator statistics where available. CPU GPU fields are null.
  Reserved/pool bytes are not live buffers. Sampling every 50 ms can miss brief
  peaks; memory figures also include untimed verification/diagnostics.
- **Graph/JIT diagnostics:** a separate instrumented run records Dask completed-task
  count (including nested graphs) and JAX `Finished XLA compilation` log events.
  These are diagnostic-run events, not lifetime unique executables. A separate
  graph build counts unoptimized dataset tasks. None of these timings enter the
  warm statistics.
- **Transfers:** explicit kernel copy bytes/times are recorded. Pipeline transfer
  counts are **not inferred** from Dask task counts or array types. Use `--trace`
  for a separate end-to-end JAX trace and inspect host/device copy events; trace
  availability depends on the host profiler/CUPTI installation. Tracing never wraps
  the measured warm rounds. On JAX versions exposing `ProfileOptions`, Python
  tracing is disabled and host tracing reduced to avoid overflowing the event
  buffer. Summarize observed copy events with:

  ```sh
  python -m benchmarks.trace_summary path/to/host.trace.json.gz --output transfers.json
  ```

  The summary marks million-event traces as potentially truncated; copy counts
  and byte totals describe captured events, not a guaranteed complete census.
- **Correctness:** warm results are sampled against the cold result; comparison
  mode also compares baseline/candidate products and shapes (rtol 1e-7, atol 1e-8).
  Samples take first/middle/last along each output dimension. This is a bounded
  regression signature, not a full scientific validation or whole-cube equality.

## Compare one implementation PR at a time

Keep both checkouts and environments available. The *same harness* runs against
both source roots. No editable-install switching is required:

```sh
python -m benchmarks.run --source-root /path/to/parent \
  --candidate-root /path/to/candidate --pairs 5 --device gpu \
  --cases aa1-mixed aa2-rfi --modes zarr rfi-kernel \
  --output benchmark-runs/paired
```

Optional `--candidate-python` selects another interpreter, but mismatched dependency
versions are rejected when comparing. Order alternates AB/BA between pairs. Each
pair uses fresh processes and five warm rounds per side, so inspect between-process
variation as well as the within-process median/IQR/stddev. Do not compare the
cold first run of one implementation to the warm run of the other. Reports include
revision, tracked diff hash, source hash, fixture/harness hashes, versions, hardware,
device, precision, chunks, worker count and relevant allocator environment.

A proposed speed acceptance gate is a >=5% median end-to-end gain **larger than
observed variation** with matching outputs; it is not an automatic statistical
significance test. Capacity changes may qualify through lower memory or a newly
feasible case while explicitly reporting runtime cost. Changing requested output
products is a tradeoff, not a speedup for identical computation. Kernel wins alone
do not establish end-to-end benefit. Always retain failed/skipped outcomes.
