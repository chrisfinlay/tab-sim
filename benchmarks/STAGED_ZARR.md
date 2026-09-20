# Staged Zarr investigation

Started from merged main `e61dcdf` after #62 and #63 for #54. The investigated
writer is now promoted to `tabsim.staged` and is the CLI/Python default; see
[production usage and controls](../docs/staged-output.md). Historical measurements
below and in `results/staged` used the revisions explicitly recorded there. The
benchmark adapter retains all arrays for comparability, unlike the production
default which omits RFI amplitude diagnostics.

`staged_zarr.write_staged_observation` writes `vis_ast`, `vis_rfi`, `gains_ants`
and the existing `noise_data` concurrently into distinct arrays of one
`result.zarr`. Metadata preparation is serial; `component_workers` bounds the
number of independent synchronous Dask task streams (two in the command below).
After all components finish, new graphs read their stored chunks and append
`vis_obs`, then `vis_calibrated`, then `flags` to that same store. Ancillary
variables are written once with their xarray encodings. Sequential composition
and ancillary writes use `Dataset.to_zarr(compute=True)` under the synchronous
scheduler. This keeps the Dask Array store graph out of deferred collection
conversion, which caused input-read accumulation with Dask 2026.8.0. It does
not call `Dataset.compute()` or materialize the full dataset. The parallel
component phase still prepares metadata serially before executing its writes.
No second output store
or final copy is created. Metadata is consolidated only after success; the
`staged-status.json` inside the store (also copied adjacent by the benchmark adapter) marks incomplete stores and completed stages.

The caller must provide the original `flags` option. Noise comes from the
existing calculation, preserving its seed. The production Observation API rejects changes to core calculated arrays and preserves ancillary/metadata customizations. Retained components support manual
investigation; automatic restart/resume is not implemented.

## Run under the capacity supervisor

```sh
PYTEST_PLUGINS=benchmarks.staged_plugin python -m benchmarks.run \
  --cases aa4-out-of-core --modes zarr --capacity --memory-model staged \
  --device cpu --workers 2 --host-budget-gib 8 --available-memory-fraction 0.7 \
  --timeout 7200 --keep-output --output /large-disk/new-staged-result-directory
```

The plugin refuses ordinary timed benchmarks. Records identify `staged-zarr-v2`.
Disk admission allows one complete store plus the existing free-space reserve.
The experimental staged memory model removes the full-cube CPU retention
allowance but retains geometry, graph and per-worker working-set allowances.
This is a hypothesis tested under the supervisor's actual RSS limit, not a
proven capacity guarantee. `--keep-output` is capacity-only and retains scratch
on success and failure; select a fresh output directory for every attempt.

Tests compare every output with the ordinary writer, including seed overrides,
disabled flags, noiseless flags, empty sources, uneven chunks and encoded
ancillary variables. Separate tests exercise concurrent component execution,
failure barriers, output retention and admission accounting. A read-ahead
regression verifies that the first composition chunk is written before a whole
input component is read. Composition failures must leave the store incomplete
and unconsolidated.

## Remaining capacity investigations

- Automatic restart/resume is not implemented; failed permanent stores are retained.
- Measure peak host/device memory across all stages and parallel workers.
- Verify retained full output and CPU/GPU numerical equivalence.
- Test increasing datasets at fixed tiles/concurrency, including a visibility
  larger than physical host RAM as well as one larger than VRAM.
- Promote successful cases to repeated timing figures and account for extra
  component reads; a cold experimental success is not a speedup.
