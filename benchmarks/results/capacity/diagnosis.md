# Why small visibility chunks still exhaust device memory

These are **instrumented diagnostic experiments, not timing benchmarks**. They
use the `aa4-out-of-core` fixture, complex128, a 16 MB planner target, one local
Dask thread, the same GPU and default JAX pool as the capacity report, and
production revision `de43129`. Actual visibility tiles are `(2, 130816, 1)`:
**4,186,112 bytes (3.992 MiB)**. The single visibility cube is 7.984 GiB.

## Established mechanism

The local Dask scheduler retains completed JAX task results on the GPU while
other tasks still need them. A task cache sample before the baseline failure
contains **1,211 distinct JAX Python objects, totaling 4,401,816,192 logical
bytes (4.100 GiB)**. JAX reports 4,724,194,816 bytes in use against its
4,771,020,800-byte allocation limit at that sample. The largest individual
cached visibility arrays are ordinary 3.992 MiB tiles. This is accumulation of
many tiles, not evidence that the visibility kernel silently computed a whole
cube or that the failed 63.88 MiB allocation is the tile shape.

The dependency trace identifies a concrete retention path:

1. `Observation.__init__` creates identical astronomical and RFI zero-array
   accumulators. Their identical Dask keys share the zero tasks.
2. `Observation.addAstro` returns JAX-backed tiles from
   `tabsim.dask.interferometry.astro_vis`, then adds them to its accumulator.
3. Hundreds of completed `_astro_vis` tiles remain in the scheduler cache.
   Their `add` consumers are waiting for the corresponding `zeros_like` tasks.
   At one sample, 570 such consumers hold approximately 2.39 GB of visibility
   results. This is an execution-order backlog, not a dependency cycle.
4. Removing that particular dependency does not eliminate retention: with
   scalar-zero accumulators, results accumulate behind gain application and
   output consumers instead. Sampled gain tasks await RFI and antenna-index
   inputs; output tasks await another visibility result.

The full graph assembled by `Observation.calculate_vis`,
`write.construct_observation_ds` and `Observation.write_to_zarr` has several
consumers for each visibility family. Ordinary local scheduling does not spill
these GPU buffers. Limiting Python workers or the size of one kernel tile does
not bound the total live device working set.

## Controlled interventions

The [sanitized numeric records](diagnosis-probes.json.gz) preserve sampled cache and allocator
statistics. Cache bytes deduplicate Python object identities, not underlying
storage; views can alias. Per-task-family counts can include references to the
same object and must not be summed as physical memory. Allocator statistics are
reported separately. Sampling misses peaks. Instrumentation wraps map callbacks
and can affect task identifiers/order, so these runs are diagnostic corroboration
of the previously reproduced uninstrumented OOM, not timing comparisons.

| Intervention, separately from baseline | Result | What it establishes |
|---|---|---|
| Original callback behavior | GPU OOM | Thousands of small device results can remain live across the graph. |
| Wait for every mapped callback result with `jax.block_until_ready` | GPU OOM | Completion fencing alone is insufficient; completed tiles remain retained. |
| Initialize the two accumulators with scalar zeros before adding sources | GPU OOM | Removing the initial shared-zero barrier alone is insufficient. This is a diagnostic substitution, not a safe public-API change for zero-source observations. |
| Disable Dask low-level task fusion | GPU OOM | This particular fusion switch alone does not bound the graph's retained device results. |
| Write only `vis_obs` with the original callback behavior | GPU OOM | Removing optional/other output products alone is insufficient; the simulation graph still retains device tiles. |
| Return NumPy arrays from every mapped callback | Stopped at 8 GiB host RSS guard | GPU cache accumulation disappears, but retained host results grow instead. Moving the backlog to RAM is not proof of out-of-core execution. |

In the host-return probe, sampled JAX cache bytes remain zero and the observed
JAX allocator peak is 73,372,160 bytes. The monitor explicitly records a
`host_rss` stop at **8,625,029,120 bytes (8.03 GiB)** against its 8 GiB cap,
with 422,879,698,944 bytes free on the scratch filesystem and 57.69 seconds
elapsed against the 2,400-second limit. Host-array cache samples are included
separately, confirming that completed NumPy results accumulate. Those logical
byte totals may include views of shared storage. No complete output/readback
was obtained. The guarded stop is not a machine-wide host OOM.

## Reproduction method

Build the fixture using `benchmarks.harness.build_observation(case, 16)`,
`add_sources`, `calculate_vis`, and the normal full-dataset Zarr writer. Enable
JAX x64 before runtime imports, select CUDA, and use one Dask worker. Run each
intervention in a fresh process with the same memory pool and resource guards.
Do not combine interventions when attributing their effects.

Wrap `xarray.map_blocks` callbacks only for the chosen intervention. A
`dask.callbacks.Callback(posttask=...)` samples `state['cache']` recursively
through tuples, dictionaries, xarray datasets and data arrays without converting
JAX values. Deduplicate JAX objects by identity and record their shape/nbytes.
For large cached entries, inspect `state['waiting_data'][key]` and each consumer's
`state['waiting']` prerequisites. Record `jax.devices()[0].memory_stats()`
separately. The baseline/synchronization probes sampled every 100 completed
tasks; dependency and other intervention probes sampled every 500. This
instrumentation must remain outside timed benchmark rounds.

## Implementation implications

Keep strict chunk planning in #52, but do not treat it as sufficient to repair
this failure. Prioritize a bounded lifetime/placement policy in #54: only a
bounded set of tiles should be in flight across all branches, through completion
and writing. Test task-graph/output scheduling changes against the same complete
output; do not call a reduced-output experiment a capacity fix for the default
writer. If host staging is used, bound or spill that working set too.

First-contribution assignment can remove redundant zero-array dependencies, but
the measured scalar-zero failure rules it out as a complete remedy on its own.
Similarly, neither synchronization nor host conversion alone is a demonstrated
solution. A bounded compute-and-write tile task may be necessary if scheduling
and placement changes cannot establish a stable memory ceiling; evaluate its
transfer/recomputation tradeoffs before choosing that larger refactor.

Acceptance requires full writes and numerical checks with a single visibility
larger than VRAM, then larger than host RAM, plus a size sweep showing a bounded
working set at fixed tile/concurrency settings. Newly passing cases enter the
capacity table immediately and attempt repeated timing promotion in the same PR.
