# Capacity reassessment: issue 52

Cold full-store execution, bounded sample validation, two component streams and
a 64 MB chunk target. These retain the historical fixtures' source counts (two
RFI for AA4), not the separate 512-RFI performance pilot. Process wall time
includes startup, readback and deletion. It is not a warm timing statistic.

| Machine / fixture | Single visibility | Parent | Issue 52 |
| --- | ---: | --- | --- |
| mini / `aa4-out-of-core` | 7.98 GiB | PASS, 171.3 s, 2.23 GiB RSS | PASS, 168.5 s, 2.54 GiB RSS |
| mini / `aa1-host-stress` | 1.88 GiB | Host preflight SKIP | PASS, 64.8 s, 2.69 GiB RSS |
| mini / `aa4-host-out-of-core` | 31.94 GiB | Disk preflight SKIP | Disk preflight SKIP |
| Daint / `aa4-out-of-core` | 7.98 GiB | PASS, 190.6 s, 4.42 GiB RSS | PASS, 185.2 s, 4.72 GiB RSS |
| Daint / `aa1-host-stress` | 1.88 GiB | PASS, 94.1 s, 5.54 GiB RSS | PASS, 97.9 s, 5.17 GiB RSS |
| Daint / `aa4-host-out-of-core` | 31.94 GiB | TIMEOUT at 400 s, 4.81 GiB RSS | Allocation-limited TIMEOUT at 357 s, 4.58 GiB RSS |

RSS here is the external supervisor's process-tree peak. See [capacity.json](capacity.json)
for exact values, source revisions, cold simulation times and JAX allocation
statistics on passing runs. JAX pool reservation on Daint is separate from live
allocation and must not be described as active working-set usage.

## Admission and blockers

On mini, the AA1 parent selects `(8192,1)` time/frequency chunks and the staged
host admission estimate is 17.19 GB, exceeding the approximately 8 GB available
budget. The candidate selects `(256,32)`, reducing that estimate to 5.00 GB;
its actual RSS peaks at 2.69 GiB. The parent was skipped by a guard, not run to
an OOM. On Daint both execute, showing that the parent skip is budget-specific.

The 31.94 GiB mini fixture needs a projected **208.04 GB** store, exceeding
**94.86 GB** usable SSD space after reserve. The disk guard is retained; reducing
retained arrays would change this workload. No memory-failure inference follows
from that skip.

Daint uses a GH200 (device-reported model `NVIDIA GH200 120GB`), approximately
856 GiB host RAM and the debug partition, with a 32 GiB process RSS guard and
400-second maximum per-case limit (issue 52 received 357 seconds because of
the remaining allocation time). The parent's largest case completed all component
writes by 188 s, then timed out during `vis_obs` composition. Host RSS remained
well below its guard. This isolates the unfinished phase; it does not distinguish
CPU scheduling from filesystem throughput without further instrumentation.
Issue 52 completed components by 197 s and was also in `vis_obs` composition
at its shortened deadline. These incomplete runs establish neither success nor
a timing regression at this size.

These are cold capacity checks, not five-repeat performance figures. The staged
capacity plugin deliberately disallows warm benchmark mode; promoting these
passes requires a matching repeated full-store protocol and sufficient allocation
time. The largest fixture additionally remains blocked by scratch on mini and
timeout on Daint. No fixture guards were bypassed to obtain a pass, and no claim
of a completed larger-than-physical-memory run is made: the completed mini
visibility is below 16 GiB and all these fixtures fit Daint's physical memory.

Reproduce each entry from its committed checkout with:

```sh
PYTEST_PLUGINS=benchmarks.staged_plugin python -m benchmarks.run \
  --cases aa4-out-of-core --modes zarr --capacity --memory-model staged \
  --device gpu --workers 2 --chunk-mb 64 --host-budget-gib 32 \
  --available-memory-fraction .7 --timeout 400 --output /scratch/fresh-case
```

For mini use `--device cpu --host-budget-gib 8` and its external SSD scratch.
The initial Daint source archive attempt failed provenance collection before any
simulation; the recorded matrix uses committed Git checkouts instead. Reports
are retained separately from disposable stores.
