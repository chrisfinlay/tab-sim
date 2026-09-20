# CPU/SSD versus GPU/HDD staged-writer pilot

Both systems ran revision `7635bf8`, with the same AA4 fixture, actual chunk shapes, component-worker settings and public-API staged writer. Each configuration is one cold run, not a repeated performance estimate. These compare complete systems: processor, GPU availability, storage, operating system and Dask version differ. No isolated SSD/HDD or CPU/GPU speedup is claimed.

## Matching 7.984 GiB visibility cases

| Chunk target | Workers | CPU/SSD seconds | GPU/HDD seconds | Observed elapsed ratio (GPU/CPU) | CPU peak host GiB | GPU peak host GiB | GPU peak live buffers MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 MB | 1 | 153.09 | 1579.23 | 10.32× | 2.25 | 3.15 | 165.9 |
| 64 MB | 1 | 117.36 | 1628.42 | 13.88× | 2.56 | 3.14 | 329.8 |
| 16 MB | 2 | 133.80 | 1602.46 | 11.98× | 3.31 | 4.54 | 176.9 |
| 64 MB | 2 | 101.12 | 1229.96 | 12.16× | 2.92 | 3.15 | 400.7 |

All completed matching cases passed sampled cross-system output comparisons (`rtol=1e-7`, `atol=1e-8`). The samples cover start/middle/end indices in all three dimensions for astronomical, RFI, observed and calibrated visibility and flags. This is bounded sample validation, not a full-array cross-system comparison.

GPU process allocation was approximately 4.53 GiB throughout, including the JAX pool; the live-buffer figures above are allocator peaks, not that reservation. Host peaks are the supervisor measurements. Both requested an 8 GiB host cap with a 70%-available-memory guard; the effective cap is lower on the 16 GiB CPU system. The GPU system has approximately 23.45 GiB host RAM and 6 GiB VRAM.

The 16 MB planner target gives visibility chunks `(2, 130816, 1)` (~3.99 MiB). The 64 MB target gives `(8, 130816, 1)` (~15.97 MiB); the planner also accounts for integration samples.

## Stage example: 16 MB, one worker

| Stage | CPU/SSD seconds | GPU/HDD seconds |
|---|---:|---:|
| vis_ast | 27.36 | 79.70 |
| vis_rfi | 20.33 | 226.18 |
| noise_data | 12.60 | 207.87 |
| vis_obs | 38.42 | 544.64 |
| vis_calibrated | 27.60 | 333.02 |
| flags | 14.29 | 141.95 |

During CPU/SSD observed composition, whole-host disk counters averaged approximately 665 MB/s read and 251 MB/s write. These include other host activity and caching effects; they are not process-exclusive or durable-media bandwidth measurements. The corresponding GPU/HDD run had about 0.74% average GPU activity in composition. This supports investigating storage/CPU-side stalls before increasing GPU memory allowance, but does not isolate disk as the sole bottleneck.

JAX 0.10.2, xarray 2026.7.0, Zarr 2.18.7 and numcodecs 0.15.1 match. Dask is 2024.10.0 on CPU/SSD and 2026.8.0 on GPU/HDD. CPU-side background activity was not excluded; no competing GPU compute process was observed.

## Larger-than-host-memory case

- CPU/SSD: **skipped**. The 7.68 GiB host plan exceeded the 7.46 GiB admission budget. Independently, the 193.75 GiB disk plan exceeds approximately 75.04 GiB usable SSD space after the reserve. No large CPU output was attempted.
- GPU/HDD: no terminal result collected yet.

No default or automatic tuning rule should be treated as established from this pilot. Repeat the leading settings, collect disk latency/queueing and CPU activity on the GPU host, and keep admission guards when testing larger outputs.
