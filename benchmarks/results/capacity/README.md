# Capacity probes and admission calibration

These experiments run the full `Observation` → Zarr path once, including bounded
finite-value readback of all five visibility/flag products. They do not claim
repeatability, full-array numerical equivalence or performance improvements.
Some server CPU/GPU runs overlap; their elapsed times are operational observations,
not comparable benchmark timings. No simulation algorithm was changed.

## Completed execution outcomes

| Host / backend | Fixture | Single complex128 visibility GiB | Target chunk MB | Outcome | Supervised peak RSS GiB | RSS cap GiB |
|---|---|---:|---:|---|---:|---:|
| mini / CPU | aa4-out-of-core | 7.984 | 16 | Stopped by RSS supervisor | 5.021 | 5 |
| mini / CPU | aa1-host-stress | 1.875 | 16 | Stopped by RSS supervisor | 5.029 | 5 |
| mini / CPU | aa4-out-of-core | 7.984 | 32 | Stopped by RSS supervisor | 5.022 | 5 |
| gpu / CUDA | aa4-out-of-core | 7.984 | 16 | JAX GPU out-of-memory error during write | 2.830 | 8 |
| gpu / CUDA | aa1-host-stress | 1.875 | 16 | Completed full write and sampled readback | 3.188 | 8 |
| gpu / CUDA | aa4-host-out-of-core | 31.938 | 64 | JAX GPU out-of-memory error during write | 2.851 | 7.798 |
| gpu / CUDA, 90% pool | aa4-out-of-core | 7.984 | 16 | JAX GPU out-of-memory error during write | 2.979 | 7.839 |
| gpu / CUDA, parent #61 | aa4-out-of-core | 7.984 | 16 | JAX GPU out-of-memory error during write | 2.890 | 7.876 |
| gpu host / CPU | aa4-out-of-core | 7.984 | 16 | Completed full write and sampled readback | 9.861 | 10 |
| gpu host / CPU | aa1-host-stress | 1.875 | 16 | Completed full write and sampled readback | 4.320 | 10 |

The GPU AA1 run wrote **9.725 GiB** of files in **435.70 s** for the cold simulation
and write (455.13 s whole process). Its JAX allocator peak was **2.168 GiB**;
NVML sampled process allocation peaked at **4.527 GiB**, including the preallocated
pool. Those are different memory metrics. A successful dataset larger than VRAM
does not establish that a *single visibility array* larger than VRAM succeeds:
this case's individual arrays are 1.875 GiB.

The AA4 GPU traceback reports `RESOURCE_EXHAUSTED` allocating **63.88 MiB** in
`jit_add`. This is an actual GPU execution failure, not a preflight skip or host
RSS stop. It does not identify all live allocations or prove which operation
retains them. The ordinary local Dask scheduler does not provide GPU spilling.

The 31.938 GiB wide case also failed in `jit_add`, allocating **215.58 MiB**,
after 466.76 s whole-process time. Its single array exceeds the server's host RAM
as well as VRAM, but the observed failure was on GPU; it does not establish a
host OOM. Both default-pool probes used JAX's default allocator allowance, about
4.44 GiB on this card, rather than the entire physical 6 GiB. Repeating the original 7.984 GiB case with
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` also failed (122.03 s whole process). Raising
the pool allowance alone did not resolve the failure in this probe.

The same 7.984 GiB / 16 MB GPU case was also run against **parent `a2d94c1`**,
before PR #62. It failed with the same `jit_add` / 63.88 MiB allocation error.
Thus this failure is reproduced in the parent; these attempts do not demonstrate
that #62 introduced it, nor do they establish equal capacity for every workload.

Source inspection on the GPU host (Dask 2026.8.0, JAX 0.10.2) shows that pure
Dask delayed-task tokenization falls back to pickling JAX inputs. JAX serialization
reads a host value, and reconstruction uses `device_put`. Removing inner delayed
tasks therefore removes incidental serialization/synchronization as well as
scheduling work. One Dask worker does not guarantee one completed GPU operation
at a time. This is a potential memory-lifetime contributor, not a proven explanation
of the OOM; both implementations fail the tested case.

The mini stops establish that these runs exceed the configured **5 GiB** budget;
they do not establish machine-wide OOM on its 16 GiB host. Polling can overshoot
a limit briefly, explaining peaks slightly above 5 GiB.

The server CPU AA4 run wrote **38.352 GiB**, exceeding its 23.446 GiB host RAM,
with **9.861 GiB** peak process-tree RSS. The cold simulation/write took **1809.71 s**
(1855.44 s whole process). This demonstrates a complete dataset larger than host
RAM, but its individual 7.984 GiB visibility arrays still fit in host RAM.

The server CPU AA1 run also completed: **9.725 GiB** written, **4.320 GiB**
peak RSS, **199.84 s** cold simulation/write (220.48 s whole process). Its peak
is lower than mini's observed 5 GiB stop, illustrating that a working-set estimate
must allow for host/runtime differences; these are not cross-host speed comparisons.

## Recalibrated admission

The old default estimate budgets ten complete visibility cubes plus source data.
It skipped both original stress fixtures before executing them. The new explicit
`--memory-model chunked` estimates geometry, gain-mode temporaries, active tiles,
worker concurrency and graph storage. Actual chunk factors are used, including
the heuristic's minimum tile when the requested budget is too small.

Initial capacity probes showed that a tile-only host estimate underestimated CPU
RSS. Version 2 therefore adds one visibility cube as **empirical CPU retention
headroom**. This is not proof of a particular eager allocation, and the terminated
mini runs are lower bounds, not measurements of the eventual peak. GPU host
planning retains the tile/geometry allowance; GPU device residency remains a
separate limitation, as the AA4 failure demonstrates.

| Fixture / target MB | Old conservative host GiB | New CPU host GiB | New GPU host GiB | Disk plan GiB |
|---|---:|---:|---:|---:|
| aa4-out-of-core / 16 | 80.719 | 12.054 | 4.070 | 48.905 |
| aa1-host-stress / 16 | 22.250 | 6.532 | 4.657 | 14.840 |
| aa4-host-out-of-core / 64 | 321.375 | 36.707 | 4.770 | 193.749 |

The successful server CPU probe used the initial v1 estimate and a 10 GiB cap.
The final v2 estimate is deliberately higher (12.054 GiB), so it would skip that
exact 10 GiB admission setting. Reproducing with the calibrated model requires
a cap above the plan, for example 13 GiB with available-memory fraction 0.7 on
an otherwise available server. The allowance is conservative headroom, not an
assertion that the simulation needs 12.054 GiB.

The final CPU admission checks correctly skip both original cases at mini's
5 GiB cap. The observed earlier failures remain failures in the evidence; they
have not been relabelled as skips. Conservative planning remains the default for
historical comparisons and is retained for MS/kernel modes. Passing admission is
not a guarantee that a run fits, especially on GPU.

## Supervision and provenance

- mini has 16 GiB RAM; gpu has 23.446 GiB host RAM and a GTX 1060 with 6 GiB VRAM.
- Stress runs on the server use its dedicated data volume, not its nearly full
  system disk. Planning reserves uncompressed products and ancillary arrays.
- Each worker runs with one Dask thread. RSS, timeout and free disk are monitored;
  the effective host cap also respects initially available RAM and a 2 GiB startup
  reserve. Other applications can consume that reserve later.
- Capacity uses one cold simulation and readback, without warm/diagnostic reruns.
  The normal five-round path still passes its smoke test. Capacity records are
  explicitly rejected as paired speed comparisons.
- Owned output scratch is cleaned after success, failure, interruption or malformed
  reports. Summary, worker log and metadata remain, including failed-run peak RSS.
- Final harness tests: 32 passed on mini and on the server (CPU backend for harness unit tests).
  Final single-pass smoke tests passed on both CPU and GPU. CI: 469 passed, 13 optional-dependency
  skips on each of Python 3.10, 3.11 and 3.13. The optional checks ran on mini.
- Independent review identified missing exception-path scratch cleanup, which was
  fixed and tested. Follow-up review found no outstanding actionable issues.

Adjacent gzip JSON files contain public measurement records: fixture dimensions,
source/harness revisions, execution parameters, timings, RSS/allocator metrics,
status and concise allocation-error details. Successful readback records retain
product shapes and a sample digest. Raw worker logs, XML reports, commands and
machine-specific paths remain local and are not included in these public files.

Candidate probes use the same numerical implementation from PR #62; the explicit
parent probe uses `a2d94c1`. Later production-file edits only clarify comments.
Harness revisions are recorded per batch; do not treat unlike capacity attempts
as paired timing comparisons.

An initial `gpu-host-large` attempt used **128 times × 128 channels** (31.938 GiB)
and was skipped while the server CPU job occupied RAM: its 9.836 GiB host plan
exceeded the effective 9.569 GiB cap. The final `gpu-host-wide` fixture uses
**32 times × 512 channels**, preserving the original AA4 time geometry while
producing the same single-array size. These are distinct attempts, not equivalent
memory experiments. The final fixture definition is the latter.

See the [benchmark guide](../../README.md#capacity-admission-and-supervision) for
commands and the meaning of each limit. To reproduce the wide GPU attempt, add
`--cases aa4-host-out-of-core --chunk-mb 64 --host-budget-gib 8
--available-memory-fraction 0.65 --timeout 3600` to the capacity command.
