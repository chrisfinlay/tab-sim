# Issue 55: lazy geometry setup

These measurements came from private prototype drivers, not the subsequently published reproduction harness. Each end-to-end group contains five parent-then-candidate pairs with fixed AB order, an untimed warm-up and a timed staged write. Fixed order does not control filesystem cache, thermal drift or other temporal effects. Do not interpret small differences as speedups. The matched workload is SKA-Low AA2, 68 antennas, 16 times, 32 channels, three integration samples, eight point sources and 512 stationary RFI sources, with (8,32) chunks.

The prototype result revision fields are null because sources were copied without `.git`. A subsequent audit compared every retained tracked `tabsim` file byte-for-byte with git: baseline `354f214` (49 files), issue54 `2be16b0`, issue55 `37db1b5`, and issue56 `60e9f8b` (50 files each) matched on their tested hosts. This recovers source identity from retained snapshots; it is not contemporaneous revision capture. The public reproduction drivers now record revisions, source/harness hashes and environment identity directly. RSS includes sampled validation/cleanup; live JAX high-water marks can include warm-up and are distinct from reserved allocator memory.

### gpu55

| Label | Total seconds: median / MAD / range | Maximum sampled RSS (GiB) | Maximum live JAX bytes |
| --- | --- | ---: | ---: |
| parent | 19.3051 / 0.0200 / 19.2384–19.3349 | 2.508 | 437654272 |
| candidate | 19.3253 / 0.0250 / 19.2852–19.3935 | 2.500 | 437654528 |

Candidate median change: +0.10%. All 675 paired complex samples pass rtol=atol=1e-9; maximum absolute difference 0. This is bounded sample agreement, not exhaustive cube equivalence.

### cpu55

| Label | Total seconds: median / MAD / range | Maximum sampled RSS (GiB) | Maximum live JAX bytes |
| --- | --- | ---: | ---: |
| parent | 12.4456 / 0.3057 / 12.0974–12.8104 | 1.553 | Unavailable |
| candidate | 12.5370 / 0.2650 / 12.2720–12.9172 | 1.624 | Unavailable |

Candidate median change: +0.73%. All 675 paired complex samples pass rtol=atol=1e-9; maximum absolute difference 0. This is bounded sample agreement, not exhaustive cube equivalence.

Full-write CPU overhead must be assessed separately from the setup-memory benefit: lazy geometry shifts computation into writes and can repeat work for distinct consumers. A setup improvement alone is not an end-to-end performance improvement.

## Setup-only scaling

| Times | Label | Count | Setup seconds: median / MAD / range | Maximum sampled RSS (GiB) |
| ---: | --- | ---: | --- | ---: |
| 256 | parent | 1 | 0.4196 / 0.0000 / 0.4196–0.4196 | 0.677 |
| 256 | candidate | 1 | 0.3352 / 0.0000 / 0.3352–0.3352 | 0.492 |
| 2048 | parent | 5 | 1.1794 / 0.0041 / 1.1753–1.1886 | 1.769 |
| 2048 | candidate | 5 | 0.4218 / 0.0023 / 0.4195–0.4288 | 0.496 |
| 16384 | parent | 1 | 11.6083 / 0.0000 / 11.6083–11.6083 | 7.727 |
| 16384 | candidate | 1 | 1.2328 / 0.0000 / 1.2328–1.2328 | 0.536 |

Geometry samples match at rtol=1e-13, atol=1e-8. Only the 2048-time case has five repeats; the endpoints are single observations, not stable timing estimates. Setup timing is not a complete simulation timing: lazy execution defers geometry work and independent consumers may recompute it. These setup files contain no revision identifiers.

## External supervision

Peak process-tree RSS across each group (GiB), which has a broader lifetime than the internal sampler:

- cpu55: 1.409 → 1.795.
- gpu55: 2.415 → 2.530.

CPU: Apple M4, 16 GiB RAM, SSD; GPU: GTX 1060, 6 GiB VRAM, HDD-backed output. Each side uses the same environment on each host. JAX/JAXlib 0.10.2, xarray 2026.7.0, Zarr 2.18.7; CPU NumPy 2.5.3, SciPy 1.18.0, Dask 2024.10.0; GPU NumPy 2.4.6, SciPy 1.17.1, Dask 2026.8.0. Python versions and hardware details not present in every prototype report are not inferred from its JSON.

Immediate parent is issue54 `2be16b0`; candidate is issue55 `37db1b5`. At 2,048 times, setup median falls 64.2% and sampled peak RSS 72.0%. Setup warms the runtime on 16 times first; it does not warm every target shape. The 16,384-time single scaling observation reduces peak RSS about 93.1%. Neither result establishes complete-write acceleration.

Reproduce from the issue55 checkout:

```sh
/path/to/python -m benchmarks.geometry_setup_sweep --base /path/to/issue54 --candidate /path/to/issue55 --output /scratch/setup
python -m benchmarks.execution_sweep --base /path/to/issue54 --candidate /path/to/issue55 --python /path/to/python --output /scratch/write --device cpu --rounds 5
```

Individual measurements and bounded readback samples are in [the measurement JSON](issue55-measurements.json).
Setup samples and measurements are in [the setup JSON](issue55-setup-measurements.json).
