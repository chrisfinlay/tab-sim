# Issue 54: GPU task boundaries and admission

These measurements came from private prototype drivers, not the subsequently published reproduction harness. Each end-to-end group contains five parent-then-candidate pairs with fixed AB order, an untimed warm-up and a timed staged write. Fixed order does not control filesystem cache, thermal drift or other temporal effects. Do not interpret small differences as speedups. The matched workload is SKA-Low AA2, 68 antennas, 16 times, 32 channels, three integration samples, eight point sources and 512 stationary RFI sources, with (8,32) chunks.

The prototype result revision fields are null because sources were copied without `.git`. A subsequent audit compared every retained tracked `tabsim` file byte-for-byte with git: baseline `354f214` (49 files), issue54 `2be16b0`, issue55 `37db1b5`, and issue56 `60e9f8b` (50 files each) matched on their tested hosts. This recovers source identity from retained snapshots; it is not contemporaneous revision capture. The public reproduction drivers now record revisions, source/harness hashes and environment identity directly. RSS includes sampled validation/cleanup; live JAX high-water marks can include warm-up and are distinct from reserved allocator memory.

### cpu54

| Label | Total seconds: median / MAD / range | Maximum sampled RSS (GiB) | Maximum live JAX bytes |
| --- | --- | ---: | ---: |
| parent | 11.5250 / 0.0222 / 11.2185–11.5835 | 1.669 | Unavailable |
| candidate | 11.6824 / 0.1989 / 11.2595–12.4002 | 1.531 | Unavailable |

Candidate median change: +1.37%. All 675 paired complex samples pass rtol=atol=1e-9; maximum absolute difference 1.54e-14. This is bounded sample agreement, not exhaustive cube equivalence.

### gpu54-slot1

| Label | Total seconds: median / MAD / range | Maximum sampled RSS (GiB) | Maximum live JAX bytes |
| --- | --- | ---: | ---: |
| parent | 19.3027 / 0.0087 / 19.2791–19.3224 | 2.468 | 437654272 |
| candidate | 19.2885 / 0.0247 / 19.2435–19.3518 | 2.561 | 437654272 |

Candidate median change: -0.07%. All 675 paired complex samples pass rtol=atol=1e-9; maximum absolute difference 0. This is bounded sample agreement, not exhaustive cube equivalence.

### gpu54-slots12

| Label | Total seconds: median / MAD / range | Maximum sampled RSS (GiB) | Maximum live JAX bytes |
| --- | --- | ---: | ---: |
| parent | 19.3179 / 0.0097 / 19.3057–19.3660 | 2.509 | 437654528 |
| candidate | 19.3207 / 0.0099 / 19.3045–19.3380 | 2.535 | 437654272 |

Candidate median change: +0.01%. All 675 paired complex samples pass rtol=atol=1e-9; maximum absolute difference 0. This is bounded sample agreement, not exhaustive cube equivalence.

The slots12 experiment compares one versus two admitted GPU calls, not parent versus a second implementation. It demonstrates admission overlap, not GPU utilization or a throughput win. Small geometry/composition kernels and host beam work mean queue time is not a direct measure of GPU starvation or GPU execution duration. CPU execution counters remain zero because that instrumentation covers the GPU boundary.

## External supervision

Peak process-tree RSS across each group (GiB), which has a broader lifetime than the internal sampler:

- cpu54: 1.551 → 1.411.
- gpu54-slot1: 2.478 → 3.594.
- gpu54-slots12: 2.503 → 2.485.

## Selected-output cold CPU capacity

| Channels | One visibility (GiB) | Elapsed seconds | Sampled RSS (GiB) |
| ---: | ---: | ---: | ---: |
| 128 | 7.984 | 108.70 | 1.693 |
| 272 | 16.967 | 275.98 | 1.745 |

These are single cold issue54 runs retaining only vis_obs, with 512 antennas, 32 times, three integration samples, eight point sources, two stationary RFI sources and (1,8) chunks. Records mark bounded validation successful but contain no exported numerical samples to independently recheck here. They are not a paired speed comparison, nor the historical all-output capacity fixture. The 272-channel visibility alone is 16.967 GiB; on the stated 16 GiB host this is beyond physical RAM. This supports only the tested selected-output configuration, not arbitrary outputs, sources or concurrency. No warm-performance promotion follows from these two passes.


CPU: Apple M4, 16 GiB RAM, SSD; GPU: GTX 1060, 6 GiB VRAM, HDD-backed output. Each side uses the same environment on each host. JAX/JAXlib 0.10.2, xarray 2026.7.0, Zarr 2.18.7; CPU NumPy 2.5.3, SciPy 1.18.0, Dask 2024.10.0; GPU NumPy 2.4.6, SciPy 1.17.1, Dask 2026.8.0. Python versions and hardware details not present in every prototype report are not inferred from its JSON.

The paired baseline is merged main `354f214`; candidate is issue54 `2be16b0`. The benefit is explicit ownership and bounded GPU admission, not a measured throughput gain. Keep one GPU slot by default: two slots produce no measurable benefit here. The roughly 3 GiB allocator pool is separate from the 0.408 GiB live-buffer high-water mark. GPU call counters include warm-up and timed execution; `host_result_bytes` counts logical returned arrays, not PCIe traffic.

The capacity planner predicts 0.455 GiB of CPU task buffers at (1,8), unchanged between 128 and 272 channels. Actual process peaks are 1.69–1.75 GiB because runtime, graph and allocator caches are explicitly outside this model. The model is not an RSS ceiling; these data do not justify reducing its conservative scratch factors.

Reproduce from the issue54 checkout (paths supplied by the operator):

```sh
python -m benchmarks.execution_sweep --base /path/to/main --candidate /path/to/issue54 --python /path/to/python --output /scratch/execution --device gpu --rounds 5
python -m benchmarks.execution_sweep --base /path/to/issue54 --candidate /path/to/issue54 --base-slots 1 --candidate-slots 2 --python /path/to/python --output /scratch/slots --device gpu --rounds 5
python -m benchmarks.selected_capacity_run --root /path/to/issue54 --output /scratch/capacity --channels 272 --device cpu --timeout 600 --rss-gib 8
```

Individual measurements and bounded readback samples are in [the measurement JSON](issue54-measurements.json).

## Larger-than-VRAM GPU capacity

A Daint GH200 debug run at issue54 `2be16b0` completed a **127.75 GiB single visibility** (512 antennas, 32 times, 2,048 channels, eight point sources, two RFI sources), retaining `vis_obs` only. Fixed (1,8) chunks, two component streams, one admitted GPU call, a 32 GiB RSS stop and 50% JAX pool. This exceeds both the nominal 120 GB GPU capacity and the device allocator limit.

Complete write/prune/readback took **1,451.90 s**; process including cleanup took **1,478.08 s**. Internal sampled peak RSS was **6.025 GiB**, external process-tree peak **5.884 GiB**. JAX peak live buffers were **85.037 MiB**; its **47.500 GiB pool** and roughly **48.145 GiB NVML allocation** are separate reservations, not live tile memory. Component completion: astronomy 351.39 s, RFI 401.76 s, gains 414.46 s, noise 767.33 s; composition began at 767.70 s and the single store finished at 1,451.41 s.

At completion, 32,865 GPU boundary calls returned 384.75 GiB of logical host results. Their aggregate placement/kernel/readback time was 96.26 s, not GPU kernel-only time or utilization. Composition, storage, host noise, compression and scheduling dominate the remaining wall time; this run did not measure physical disk throughput and cannot attribute all overhead to storage. Bounded component arithmetic and final stored samples passed. Temporary components were pruned; the disposable completed store was removed after validation.

The planner predicted 0.642 GiB task buffers for this GPU tile, above the observed 0.083 GiB live-device high-water mark but below whole-process RSS because it excludes runtime/graph/cache overhead. Keep the separate RSS guard and scratch allowance; do not present the buffer estimate as a process-memory bound. [Capacity measurements and stage counters](issue54-capacity.json) include both CPU sizes and this GPU run. These are cold capacity results, not speed ratios.

For the 512-RFI (8,32) timing tile, estimated CPU buffers were 2.080 GiB and GPU buffers 2.877 GiB, versus 0.408 GiB peak live JAX buffers on the GPU. This calibration supports retaining conservative allowances, not equating them with measured process RSS.
