# Issue 52: strict chunks and working-set planning

The configured 45 MB limit previously selected a **55.98 MB** fine-time tile.
The strict planner selects **27.99 MB**, with time/frequency chunks changing
from `(16,32)` to `(8,32)`. Both shapes divide the observation. Source hints and
an optional modeled budget can further constrain this choice; runtime RSS/VRAM
guards remain separate.

## Matched end-to-end measurements

Five alternating parent/candidate processes per machine, each with one untimed
warm-up followed by one timed complete staged write. SKA-Low AA2: 68 stations,
16 times, 32 channels, 3 integration samples, **512 stationary RFI sources**,
8 point sources, two component streams, x64. Identical retained outputs (default
all except `rfi_*_A`); filesystem caches were not cleared. CPU is mini/Apple M4
with 16 GiB RAM; GPU is the GTX 1060 with 6 GiB VRAM. Compare within each machine,
not CPU versus GPU as an isolated hardware experiment.

| Metric | Parent | Issue 52 | Change |
| --- | ---: | ---: | ---: |
| CPU median total write, including setup/graphs | 12.326 s | 12.107 s | −1.8%, neutral |
| CPU measured range | 12.103–12.994 s | 11.874–12.962 s | Overlapping variation |
| CPU maximum sampled process RSS | 2.665 GiB | 1.827 GiB | −31.5% |
| GPU median total write, including setup/graphs | 34.417 s | 19.204 s | −44.2% |
| GPU measured range | 26.798–41.399 s | 19.182–19.251 s | Every candidate run faster |
| GPU maximum sampled host RSS | 3.945 GiB | 2.529 GiB | −35.9% |

All deterministic astronomical/RFI sample values match across the chunk change.
Noisy products are validated against their own persisted gains/noise: changing
Dask chunk shape changes seeded noise partitioning, so noisy arrays are not
claimed bitwise equal across plans. RSS sampling includes untimed readback and
cleanup; JAX allocator peaks include the warm-up. Pool reservation was 50% on the
GTX, and is distinct from live allocation. Exact samples, median absolute
deviations and device allocation peaks are in [paired.json](paired.json).

The CPU change is below the proposed 5% speed threshold and is **not** a speedup
claim. The GPU result exceeds 5% and its entire observed ranges are separated;
the parent still has substantial variability. This is a matched pilot, not a
universal speedup prediction. An earlier CPU sweep that overlapped tests was
discarded from the timing comparison.

## A constrained case now completes

At a **3 GiB process-tree RSS threshold**, the same parent workload was stopped
at 3.0001 GiB. The candidate completed at **2.298 GiB**, about **23% measured
headroom**. The external supervisor samples every 0.2 s and terminates exceeding
processes; it is not an OS allocator cap. This demonstrates a capacity benefit
at fixed inputs and concurrency. It is not a larger-than-physical-RAM test.

## Scratch is measured separately from the estimate

Public JAX `lower(...).compile().memory_analysis()` on the GTX reports:

| Time tile | Kernel arguments | Output | Compiler temporary buffers |
| --- | ---: | ---: | ---: |
| 16 | 441,225,056 B | 18,661,376 B | 905,536 B |
| 8 | 220,630,880 B | 9,330,688 B | 455,232 B |

The scan kernel does not materialize an all-source baseline phase cube. The
planner's larger scratch allowance covers pipeline/beam/readback headroom, not
literal compiler temporaries. The original GPU host estimate omitted that
host-side allowance; measurement prompted its inclusion. This reporting fix
leaves `estimated_bytes`, selected chunks and execution unchanged. The measured
candidate source is `2c5907c`, with that estimate-only correction in `4b38348`.
The CPU estimate at the chosen shape is 2.233 GB versus observed 1.827 GiB whole
process RSS; GPU estimates intentionally remain conservative and exclude fixed
runtime/cache/eager-history allocations. See [the model](../../../docs/chunk-planning.md).

## Reproduction

Measured environments used JAX/jaxlib 0.10.2, xarray 2026.7.0 and Zarr 2.18.7.
Mini used NumPy 2.5.3 / SciPy 1.18.0 / Dask 2024.10.0; the GTX host used
NumPy 2.4.6 / SciPy 1.17.1 / Dask 2026.8.0. Comparisons are paired within each
environment. This dependency difference is another reason not to interpret
cross-machine timings as a controlled hardware comparison.

Use identical dependencies for each pair and committed source checkouts:

```sh
python -m benchmarks.stack_sweep --base /path/to/6d90261 \
  --candidate /path/to/issue52 --output /scratch/fresh-results \
  --device gpu --python /path/to/venv/bin/python
```

The driver uses an 8 GiB external RSS threshold and 600-second per-process
maximum. For the constrained pair add `--rounds 1 --rss-gib 3
--continue-on-failure`. `benchmarks/stack_profile.py` records beam transfer events
and compiled RFI memory. Recorded measurements compare the immediate parent
(merged PR 64), not the pre-noise-fix original baseline; no cumulative speedup
against that older, different output implementation is claimed.

Large-fixture capacity results are in [capacity.md](capacity.md) and are not
inferred from these small timing cases. A cold capacity pass alone is not
included in the speed table.
