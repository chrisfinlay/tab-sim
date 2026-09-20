# Disk-staged simulation output

`sim-vis`, `run_sim_config`, and `Observation.write_to_zarr()` now use the same
single-store staged writer by default. `Observation.write_to_ms()` stages to a
temporary Zarr store beside the requested MS when no dataset is supplied.
`calculate_vis()` still constructs lazy graphs: calling `.compute()` yourself
materializes the result and does not invoke the staged writer.

The workflow is:

1. Prepare metadata serially, then write `vis_ast`, `vis_rfi`, `gains_ants`, and
   `noise_data` to distinct arrays in one Zarr store. At most `component_workers`
   independent synchronous Dask task streams run at once.
2. Wait for all required components. Read their stored chunks to write `vis_obs`
   back to that store, followed by calibration and flags only when needed.
3. Write selected ancillary arrays. Run requested MS conversion, MS accumulation
   and signal statistics using persisted data, then delete dependencies that the
   user did not select for retention. Consolidate Zarr metadata.

There is no second Zarr store or final dataset copy. MS-only output uses a
managed temporary store on the output filesystem and removes it after conversion
(or failure). A permanent Zarr failure leaves an incomplete store for diagnosis;
`staged-status.json` inside it records completed stages. There is no automatic
resume. Use a new output path or explicit `overwrite=True` to restart. Never
consume a store whose marker says `complete: false`. The marker describes Zarr
writing, not successful completion of downstream MS conversion.

## Output selection and optional work

By default all data variables except `rfi_*_A` are retained. These large RFI
amplitude diagnostics are still computed where required by visibility kernels;
omitting them avoids their *second* computation/rechunking and disk write.
`save_rfi_amplitudes: true` restores their retention. This applies to stationary,
circular satellite and TLE satellite sources.

`save_arrays` specifies an exact list of retained data variables and overrides
that default. Coordinates and attributes are always kept. An empty list keeps
only coordinates/attributes. Unknown names fail before output is created.
Dependencies may be staged temporarily even when not selected for retention.
For example `vis_obs` requires all four components. Explicitly selecting
`rfi_stat_A` retains it even when `save_rfi_amplitudes` is false.

If neither calibrated output nor active flags is needed, calibration is not run.
Exactly unity gains skip inverse-gain calibration; if requested, the calibrated
array is still saved with observed values and its own encoding. Near-unity gains
are not treated as unity. `flag_data: false` writes false flags if flags are
selected; excluding `flags` skips flag output entirely. MS export currently
requires calibrated visibilities, flags and model/noise columns regardless of
which variables are retained in Zarr.

`diagnostics.signal_stats` defaults to false because it adds full-array
reductions. Existing `rfi_seps`, `src_alt` and `uv_cov` switches control plots.

## Configuration and CLI

```yaml
dask:
  max_chunk_MB: 64
  component_workers: 2
  max_memory_gb: 8
  memory_fraction: 0.7
  max_device_memory_gb: null
  timeout_s: 7200
  disk_reserve_gb: 2
output:
  zarr: true
  ms: false
  save_arrays: [vis_obs, vis_ast, vis_rfi, gains_ants, noise_data]
  save_rfi_amplitudes: false
  flag_data: false
diagnostics:
  signal_stats: false
  rfi_seps: false
  src_alt: false
  uv_cov: false
```

```sh
sim-vis -c observation.yaml --max-chunk-mb 64 --component-workers 2 \
  --max-memory-gb 8 --timeout-s 7200 --disk-reserve-gb 2 \
  --save-arrays vis_obs --no-flag-data --no-signal-stats
```

All CLI options are overrides; omission preserves YAML values. Use
`--save-rfi-amplitudes` / `--no-save-rfi-amplitudes` to change the default
selection, or `--save-arrays` with no following names for metadata only.

## Python

Pass execution controls when constructing `Observation`, along with the usual
geometry/time/frequency parameters:

```python
obs = Observation(..., max_chunk_MB=64, component_workers=2,
                  max_memory_gb=8, timeout_s=7200, disk_reserve_gb=2)
# Add sources and gains as usual.
obs.calculate_vis(flags=False, random_seed=0)
ds = obs.write_to_zarr("observation.zarr", save_arrays=["vis_obs"])
```

Writer resource options can also be overridden per call:
`obs.write_to_zarr(path, component_workers=1, max_memory_gb=6)`.
Set compute chunk size in the constructor, before creating source/noise graphs;
rechunking output later does not change kernel working sets. Different chunk
shapes can change Dask's seeded noise samples. `write_to_zarr` returns a dataset
backed by the persisted store and updates `obs.dataset`.

Dataset attributes, variable attributes, encodings and added ancillary variables
are preserved. Replacing/removing calculated core arrays in `obs.dataset` is
rejected before writing, rather than silently ignoring replacements. Change the
observation inputs and call `calculate_vis()` again. A persisted dataset can be
copied to a new store; overwriting its own backing store is rejected. An explicit
external `ds=` passed to `write_to_ms` remains a lower-level caller-managed path.

## Meaning of the controls

| Control | Meaning |
| --- | --- |
| `max_chunk_MB` | Existing target (decimal MB) used to choose time/frequency factors for the fine-time visibility tile. Default 100. It is not a total memory cap; factor rounding can exceed the target. Baseline and source axes are not split. |
| `component_workers` | Maximum concurrent component task streams; default 2. Composition and ancillary writes are sequential. More streams can increase memory and disk contention. |
| `max_memory_gb` | Process host RSS guard in **GiB**, including allocations present before writing. `null` chooses baseline RSS + `memory_fraction` × currently available host RAM. |
| `memory_fraction` | Automatic host budget fraction, default 0.7; used only when `max_memory_gb` is null. |
| `max_device_memory_gb` | Optional per-device live JAX allocation guard in **GiB**. Requires device memory statistics; unavailable CPU backends reject it. It excludes JAX's reserved allocator pool and other processes. |
| `timeout_s` | Writer deadline, checked at task/stage boundaries. |
| `disk_reserve_gb` | Free disk reserve in **GiB**, default 1. Admission also requires 1.1× uncompressed required-array bytes. Compression may use less; this conservative estimate is not an exact codec prediction. |

Memory, disk and timeout checks occur at task boundaries (resource polling is
throttled to at most once per 0.2 seconds, plus stage boundaries). They are
**guards, not hard allocation limits**: an in-flight JAX kernel or Dask task can
overshoot before the next check, and cannot be interrupted by this mechanism.
These controls cover the staged writer, not source setup, diagnostic plotting,
or subsequent MS conversion/statistics. Use smaller chunks/fewer workers to
reduce the working set. Large source/geometry tensors may still require a
further source-axis batching change. A truly hard process limit requires an
external supervisor such as the capacity benchmark runner.

Configure JAX allocator environment settings before importing/initializing JAX
when a reserved GPU pool limit is needed. The writer never changes allocator
settings after initialization or treats a low live-allocation reading as low
reserved VRAM usage. The chunk/worker defaults remain conservative existing
values; the Daint sweep is evidence for tuning, not a universal auto-heuristic.
