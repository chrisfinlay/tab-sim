# Staged Zarr investigation

Started from merged main `e61dcdf` after #62 and #63. This is an experiment for
#54, not a replacement for the production writer or a merge-ready capacity fix.

`staged_zarr.write_staged_observation` receives an observation whose visibility
calculation has already been configured. It completes separate writes of
`vis_ast`, `vis_rfi`, `gains_ants` and the existing `noise_data`, reopening each
store with its stored chunk layout. It composes observed visibility from those
stored arrays, writes/reopens that product, then does calibration and flags in
separate stages. Finally it writes all original variables and metadata with the
visibility products replaced by arrays read from disk.

The caller must explicitly provide the original `flags` option. Noise comes from
the existing calculation, so a seed override is not regenerated or lost. This
prototype assumes no custom changes to the composed dataset variables.

## Run under the capacity supervisor

```sh
PYTEST_PLUGINS=benchmarks.staged_plugin python -m benchmarks.run \
  --cases aa4-out-of-core --modes zarr --capacity --memory-model chunked \
  --device gpu --host-budget-gib 8 --available-memory-fraction 0.7 \
  --timeout 2400 --output /large-disk/new-staged-result-directory
```

The plugin refuses ordinary timed benchmarks. Phase records identify the
`staged-zarr-v1` workload and separate component/final output bytes. Total output
bytes include both temporary components and final output; they are not directly
comparable with a writer that stores each product only once. The experiment
requires twice the existing disk-planning allowance and uses the existing
supervisor for RSS, free-space, timeout and cleanup. CPU admission still uses
the existing conservative retention allowance; do not silently bypass it based
on GPU stage-boundary observations.

Four complete-output equivalence tests pass on both CPU test machines. They
compare every variable, dimension, dtype, attribute and value at double
precision, including explicit seed zero, another seed override, disabled flags,
zero-noise flags and empty source families. The prototype passed the subagent
review loop; GPU capacity results are recorded separately when complete.

## Requirements before the production PR

- Preserve caller calculation options and supported dataset customizations.
- Define cleanup, interrupted stages, restart and overwrite semantics.
- Measure all stages and their peak host/device memory, not just phase boundaries.
- Retain full output and verify CPU/GPU equivalence across source families.
- Test growing datasets at fixed tiles/concurrency, including a single visibility
  larger than physical host RAM as well as one larger than VRAM.
- Account for component storage in admission checks and disk-space supervision.
- Investigate appending composed variables to the component store to avoid the
  duplicate final write, with input arrays protected from overwrite.
- Promote successful production cases to repeated timing figures and record
  additional disk reads/writes; a cold experimental success is not a speedup.
