# Staged Zarr evidence

This is an opt-in capacity experiment, not completion of #54 or a production default.
All figures below are isolated cold diagnostic outcomes, not repeated speed comparisons.

| Test | Result | Time | Peak host RSS |
|---|---|---:|---:|
| Single-store AA4 CPU, 2 component workers, revision e4670a0 | Full writer and sampled readback passed | 127.96 s | 2.08 GiB |
| Same writer on GPU, e4670a0 | 8 GiB RSS guard stopped observed composition; four components retained | 598.85 s | 8.00 GiB |
| Fresh GPU process, original deferred observed write | 8 GiB RSS guard; no output chunks | 37.22 s | 8.02 GiB |
| Direct scheduler invocation, unchanged deferred roots | 8 GiB RSS guard | 48.73 s | 8.00 GiB |
| Public immediate xarray observed write, retained inputs | Complete output and sampled checks passed | 415.89 s | 1.07 GiB |
| Public immediate calibration and flags, retained inputs | Both complete; start/middle/end checks passed for all three products | 381.22 s | 2.78 GiB |

AA4 input shape: (32 times, 130816 baselines, 128 channels), complex128.
One visibility is 7.984 GiB; stored chunks are (2, 130816, 1).
There are 2048 chunks in each composed product. CPU host: 16 GiB RAM;
GPU host: 23.45 GiB RAM and 6 GiB VRAM. Both use JAX 0.10.2 and xarray
2026.7.0; Dask differs (CPU 2024.10.0, GPU 2026.8.0).

The final two runs test the public-API change in 082a713, in separate processes
using retained components. They are not a new complete end-to-end writer run;
missing ancillary arrays and the original incomplete marker were preserved.
Diagnostic arrays use distinct names to protect the existing evidence. The
normal writer creates each product once in a single store.

## Cause and public-API remedy

In the affected deferred-write path, Dask's collection-to-delayed conversion
wraps literal roots in runnable identity tasks. With the local synchronous
scheduler, the first root releases its chunk reads before the other roots run.
Read chunks accumulate while consumers wait for those other inputs. A
scheduler-only reproduction and a private root-normalization diagnostic isolated
this mechanism; the private manipulation is not included in the implementation.

The supported remedy is `Dataset.to_zarr(compute=True)` under an explicitly
synchronous scheduler for sequential composition and ancillary stages. This
executes the chunked Array store graph without the extra delayed conversion.
It does not materialize a full dataset. Xarray encoding and chunk validation are
preserved. Component metadata is still prepared serially before parallel writes.

44 targeted tests pass with both Dask versions. The read-ahead regression fails
against the original implementation on the affected version (193 input reads
before the first output write); the fixed path satisfies a limit of 12. Tests
also cover component/composition failures, encoding, seed overrides, empty
sources, disabled flags, noiseless flags, concurrency and output retention.

## Outstanding measurements

- Complete end-to-end GPU run at the final revision, followed by a single
  visibility larger than physical host RAM, under unchanged supervision.
- GPU activity, memory-bandwidth activity, host RSS, process GPU allocation and
  allocator live-buffer peaks across chunk targets and component worker counts.
- Repeated finalist timings before choosing defaults. Report actual chunk shapes:
  the planner rounds targets to valid time/frequency factors.
- Stage-specific measurements: component workers affect component creation;
  composition remains sequential. GPU process allocation includes reserved pool
  memory and must not be treated as live buffer demand.
