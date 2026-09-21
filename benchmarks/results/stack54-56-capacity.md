# Issues 54–56: final-stack capacity reassessment

These are **cold capacity checks of the combined stack**, source `60e9f8b`, not isolated speed comparisons for any one PR. They use the production staged writer through the all-array benchmark adapter, with 64 MB nominal chunks and two component streams. Every present data variable is retained, including RFI amplitudes; this differs from the default schema and from the `vis_obs`-only capacity tests in issue54.

| Platform | Fixture | Single visibility GiB | Cold simulation + write seconds | External peak RSS GiB | Result |
| --- | --- | ---: | ---: | ---: | --- |
| CPU, 16 GiB Apple M4 | AA4 out of core | 7.984 | 155.46 | 1.364 | PASS |
| CPU, 16 GiB Apple M4 | AA1 long host stress | 1.875 | 69.59 | 2.218 | PASS |
| CPU, 16 GiB Apple M4 | AA4 host out of core | 31.938 | — | — | Disk admission skip |
| GPU, Daint GH200 | AA4 out of core | 7.984 | 187.83 | 3.721 | PASS |
| GPU, Daint GH200 | AA1 long host stress | 1.875 | 105.32 | 5.467 | PASS |
| GPU, Daint GH200 | AA4 host out of core | 31.938 | 741.42 | 4.636 | PASS |

Each PASS includes a complete write and bounded readback. Time excludes subsequent validation/deletion and process startup. Whole-process times are in the JSON. CPU and GPU samples for the shared fixtures agree at rtol=atol=1e-9 (maximum absolute differences 3.10e-12 and 1.08e-12 respectively). These are bounded sample checks, not exhaustive cube comparisons.

AA4 uses 512 antennas, 32 times, eight point sources and two RFI sources; frequency sizes are 128 and 512, with fixed (1,8) time/frequency chunks. AA1 uses 16 antennas, 32,768 times, 32 channels and (256,32) chunks. All use three integration samples. The AA1 result also exercises long-duration lazy geometry.

The two Daint AA4 sizes reach the same **85.04 MiB live-device high-water mark** while visibility size grows fourfold. Host RSS grows from 3.72 to 4.64 GiB, so this is not a claim of perfectly constant process memory: graph/cache and metadata costs remain. The default 75% allocator pool is **71.25 GiB**, with approximately 71.9 GiB process GPU allocation reported by NVML; reservation is not live tile residency. The separate issue54 127.75 GiB selected-output run used a 50% pool and must not be compared as the same output contract.

The 31.94 GiB all-output run completed astronomical/RFI/gain/noise components by 194.39 s, observed visibility by 439.98 s and calibrated visibility by 577.81 s. The prior 400-second cap stopped during composition; extending the guarded run to 900 seconds allowed the complete output/readback to pass. The final stack completes within 900 seconds; the earlier timeout was not evidence of an allocation failure. Both the implementation revision and timeout changed, so this is not an isolated timeout experiment. The run measured no physical storage bandwidth, so these timings do not by themselves establish an I/O bottleneck.

Mini's largest all-output fixture is still skipped because its conservative disk plan is **208.04 GB (193.75 GiB)**, above the approximately **94.86 GB (88.34 GiB)** usable disk budget after reserve. The initial source-copy attempt failed before computation due to missing Git provenance; the recorded CPU results here are the successful rerun from a Git checkout. No memory failure is inferred from either condition.

[Portable capacity measurements, readback samples and allocator counters](stack54-56-capacity.json).

Reproduce from a Git checkout with the benchmark dependencies installed:

```sh
PYTEST_PLUGINS=benchmarks.staged_plugin python -m benchmarks.run \
  --cases aa4-out-of-core aa1-host-stress aa4-host-out-of-core --modes zarr \
  --capacity --memory-model staged --device gpu --workers 2 --chunk-mb 64 \
  --host-budget-gib 32 --available-memory-fraction .7 --timeout 900 \
  --output /scratch/new-capacity-results
```

On the CPU host use `--device cpu --host-budget-gib 8`. Retain the disk guard; larger all-output products need larger scratch capacity. These results inform fixture admission each PR round, but cold passes remain separate from repeated timing figures.
