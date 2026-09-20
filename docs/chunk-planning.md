# Strict chunk planning and working-set estimates

`max_chunk_MB` is now a strict decimal-MB upper bound on the nominal fine-time
complex128 visibility tile: `16 * time_chunk * n_int_samples * baselines * freq_chunk`.
The planner uses Python integers and divisors, without initializing JAX. It picks
the largest feasible divisor product and prefers shorter time chunks on ties,
reducing time-dependent geometry/Fourier-gain buffers. Prime lengths may force
small tiles. If even `(time=1, frequency=1)` exceeds the limit, construction fails
with the minimum byte requirement instead of silently exceeding the budget.
Baseline and source axes are still not split by this planner.

An optional, separate `working_set_MB` bounds the **estimate** for each memory
space (host, device), including `component_workers`. It is not a guarantee of
actual RSS or GPU usage. Configure it before adding sources:

```yaml
dask:
  max_chunk_MB: 64
  working_set_MB: 2048
  planned_rfi_sources: 512
  planned_ast_sources: 8
  component_workers: 2
  task_scratch_factor: null
```

The same names are constructor arguments to `Observation`; CLI flags are
`--max-chunk-mb`, `--working-set-mb`, `--planned-rfi-sources`,
`--planned-ast-sources`, `--component-workers`, and `--task-scratch-factor`.
Counts are planning hints, not source generators. Use conservative totals for
mixed source families or unknown catalogue selections. When adding sources the
actual cumulative counts are checked before broadcasting, trajectories or graph
mutation. A broadcast spectrum with multiple source coordinates counts all of
them. An infeasible addition raises an actionable error; existing graphs and
seeded noise are never silently rechunked. Rebuild with appropriate source hints,
a lower nominal limit or a larger modeled budget. Increasing writer concurrency
also rechecks the estimate.

The x64 `scan-staged-x64-v1` model exposes its terms in `obs.chunk_plan` and the
saved dataset's `chunk_plan` attribute:

- All-source RFI amplitude input: `8*S*T*n_int*A*F` bytes.
- Source distances, geometry and astronomical intensity/direction inputs.
- One-source baseline scratch: the kernels scan sources; the model does **not**
  assume all `S*B` visibility intermediates materialize simultaneously.
- Coarse visibility outputs and six visibility buffers for sequential composition.
- Antenna gains, 1000-mode Fourier gain allowance and full-band gain generation
  before its final frequency rechunk. Shrinking the frequency tile does not hide
  those upstream full-band buffers.
- Concurrent component streams; CPU and GPU host/device budgets are distinct.

Scratch uses an explicit allowance of four times the largest nominal tile,
amplitude or distance buffer on CPU, six on GPU. `task_scratch_factor` can override
it. These coefficients are planning headroom, **not measured XLA temporaries**.
Compiler `memory_analysis()` and sampled runtime peaks must be reported separately.
The model takes the maximum of concurrent components and sequential composition,
not their sum, because the writer has a phase barrier.

The model excludes Python/runtime baseline, graph/cache retention, allocator
reservations, full-history eager antenna/TLE geometry, and optional ancillary
rechunking. Source totals can overestimate the buffers of individual add-call
batches. Keep the existing task-boundary host/device guards and an external RSS
supervisor for capacity tests; neither a nominal tile bound nor this estimate
establishes out-of-core execution. A changed chunk shape also changes Dask's
seeded noise partitioning; compare deterministic sky/RFI outputs across chunk
plans and validate noisy products against their own gain/noise components.
