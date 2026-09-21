# Issue 56: output profiles and optional diagnostics

These CPU measurements used private prototype drivers, not the new public reproduction harness. Five parent-then-candidate pairs used fixed AB order, with an untimed warm-up and a timed full staged write. Cache and temporal effects are not randomized. The workload was SKA-Low AA4, 512 antennas, eight times, 64 frequencies, three integration samples, eight astronomical sources and two stationary RFI sources.

Original records lack dependency versions, revision IDs and actual chunk metadata. A subsequent source-snapshot audit matched all 50 tracked tabsim files to issue55 `37db1b5` and issue56 `60e9f8b`; this is post-run corroboration, not provenance captured during execution. The same CPU virtual environment was reported as for issues54/55.

## Same retained output: default before/after

| Label | Total seconds: median / MAD / range | Writer seconds: median / MAD / range | Timed RSS max GiB | Supervisor RSS max GiB |
| --- | --- | --- | ---: | ---: |
| parent (default) | 13.0913 / 0.1167 / 12.9414–13.2112 | 12.9643 / 0.1191 / 12.8089–13.0833 | 0.937 | 0.951 |
| candidate (default) | 13.1093 / 0.0643 / 12.9413–13.1908 | 12.9772 / 0.0544 / 12.8117–13.0505 | 0.919 | 0.933 |

| Label | Retained data variables | Output bytes range | Selected graph tasks range | Executed writer tasks range |
| --- | ---: | --- | --- | --- |
| parent | 22 | 5109683528–5109683528 | 4977–4977 | 5119–5119 |
| candidate | 22 | 5109683528–5109683528 | 4977–4977 | 5119–5119 |

Median total change: +0.14%. All 135 complex vis_obs sample values agree at rtol=atol=1e-9, max absolute difference 0. This checks the common sampled product, not every saved variable.

This is the same-output comparison. Its timing distribution is the appropriate evidence for implementation overhead; the separate full/minimal comparison must not be substituted as a code speedup claim.

## Product choice: full versus minimal

| Label | Total seconds: median / MAD / range | Writer seconds: median / MAD / range | Timed RSS max GiB | Supervisor RSS max GiB |
| --- | --- | --- | ---: | ---: |
| parent (full) | 13.4545 / 0.0373 / 12.9923–13.4923 | 13.3183 / 0.0284 / 12.8617–13.3498 | 0.934 | 0.948 |
| candidate (minimal) | 10.5890 / 0.0356 / 10.5303–10.6395 | 10.4661 / 0.0380 / 10.3929–10.5040 | 0.952 | 0.966 |

| Label | Retained data variables | Output bytes range | Selected graph tasks range | Executed writer tasks range |
| --- | ---: | --- | --- | --- |
| parent | 23 | 5118749061–5118749061 | 4978–4978 | 6026–6026 |
| candidate | 8 | 1092237387–1092237387 | 4187–4187 | 3803–3803 |

Median total change: -21.30%. All 135 complex vis_obs sample values agree at rtol=atol=1e-9, max absolute difference 0. This checks the common sampled product, not every saved variable.

Minimal retains `SEFD`, `antenna1`, `antenna2`, `ants_itrf`, `bl_uvw`, `noise_std`, `time_idx`, `vis_obs`. Removed data variables: `ants_uvw`, `ants_xyz`, `ast_p_I`, `ast_p_lmn`, `ast_p_radec`, `flags`, `gains_ants`, `noise_data`, `rfi_stat_A`, `rfi_stat_ang_sep`, `rfi_stat_geo`, `rfi_stat_xyz`, `vis_ast`, `vis_calibrated`, `vis_rfi`. Coordinates/metadata remain. Reduced time, task count and bytes reflect fewer requested products; this is not a same-output code speedup. Dependencies may still be computed and temporarily written.

RSS scopes differ: the internal sampler covers the measured workload/readback/cleanup; the external sampler covers process startup and warm-up too. They have different sample intervals, so either can observe a transient missed by the other. Graph counts describe the selected unoptimized dataset graph, not peak resident tasks or bytes. Executed task counts include separate staged scheduler invocations. CPU live-device memory is unavailable.

CPU environment: Apple M4, 16 GiB RAM, SSD scratch; JAX/JAXlib 0.10.2, NumPy 2.5.3, SciPy 1.18.0, Dask 2024.10.0, xarray 2026.7.0, Zarr 2.18.7. The prototype requested a 64 MB chunk budget and two component streams; its omitted realized chunk fields are a reproducibility limitation, addressed in the public driver.

The same-output default remains unchanged (22 data variables). Explicit full adds `rfi_stat_A` in this fixture. `minimal` omits calibrated visibility and flags as well as components and ancillaries; their absence must suit the downstream analysis. Optional diagnostics are covered by a regression test proving four reductions share upstream work once; no separate diagnostic speedup is claimed.

Reproduce from the issue56 checkout:

```sh
python -m benchmarks.output_sweep --base /path/to/issue55 --candidate /path/to/issue56 --python /path/to/python --output /scratch/identical --device cpu --rounds 5
python -m benchmarks.output_sweep --base /path/to/issue56 --candidate /path/to/issue56 --base-profile full --candidate-profile minimal --python /path/to/python --output /scratch/profiles --device cpu --rounds 5
```

These public drivers alternate AB/BA and reject incomparable environments, missing reports and mismatched samples. Their stronger controls were added after the measurements above.

Individual measurements and bounded readback samples are in [the measurement JSON](issue56-measurements.json).
