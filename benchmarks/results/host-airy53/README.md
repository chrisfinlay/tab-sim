# Issue 53: consistent host Airy evaluation

The beam block now stays in NumPy/SciPy and returns a host array. The historical
`tabsim.jax.interferometry.airy_beam` import remains an alias. Signed sidelobes,
the zero-angle epsilon and input precision are preserved; JAX's global x64
switch no longer silently narrows host float64 inputs. See
[the numerical contract](../../../docs/airy-beam.md).

## Matched measurements against issue 52

Five alternating parent/candidate processes, each with an untimed warm-up then
one timed staged write. Same 512-RFI, eight-point-source SKA-Low AA2 workload,
45 MB chunk bound, two streams and retained outputs as the
[issue 52 measurements](../planning52/README.md). Both use `(8,32)` chunks.
Filesystem caches were not flushed. Exact timings and dispersion are in
[paired.json](paired.json).

| Metric | Issue 52 parent | Issue 53 | Interpretation |
| --- | ---: | ---: | --- |
| CPU median total | 12.319 s | 12.629 s | +2.5%; neutral under 5% threshold |
| CPU range | 12.220–12.933 s | 12.534–12.870 s | Overlapping variation |
| GPU median total | 19.252 s | 19.344 s | +0.48%; no end-to-end speedup |
| GPU range | 19.194–19.346 s | 19.332–19.388 s | Small slowdown |
| CPU maximum sampled RSS | 1.903 GiB | 1.656 GiB | Includes readback/cleanup |
| GPU maximum sampled host RSS | 2.506 GiB | 2.495 GiB | Essentially unchanged |
| GPU peak live JAX allocation | 1,340,803,840 B | 437,654,272 B | −67.4% |
| GPU reserved allocator pool | 3,181,379,584 B | 3,181,379,584 B | Fixed 50% pool |

The benefit is consistent backend behavior and fewer transient device buffers,
not a demonstrated total speedup. GPU live allocator peaks include the warm-up;
RSS is sampled during the timed workload and validation. Both are distinct from
physical GPU memory reservation. All five paired readback samples of all five
visibility/noise products agree within `rtol=atol=1e-9` on each machine.

## Transfer evidence

A separate public JAX profiler trace encloses three warmed host beam calls with
angle shape `(512,3,68)` and eight frequencies on the GTX 1060:

| Copy direction | Parent events / bytes | Candidate events / bytes |
| --- | ---: | ---: |
| Host → device | 48 / 45,122,160 | 0 / 0 |
| Device → host | 12 / 80,216,064 | 0 / 0 |
| Device → device | 36 / 5,014,416 | 0 / 0 |

Both traces contain events. Counts/bytes sum device-copy events across the three
calls, not full simulation transfers. Host SciPy previously required device
readback; no separate synchronization-duration claim is made. Public
`jax.transfer_guard('disallow')` also passes for the new direct and Dask beam
paths. The subsequent visibility kernel still needs its normal device inputs.

The 25 isolated RFI-beam calls have GPU-host medians 0.3290 → 0.3270 s and CPU
medians 0.1711 → 0.1715 s. This representative large beam is neutral even though
copies disappear; SciPy Bessel work remains on the CPU. Compiled visibility
kernel memory is unchanged (see issue 52's compiler analysis).

## Reproduction and validation

Use `python -m benchmarks.stack_sweep` with the issue 52 checkout as `--base`
and issue 53 as `--candidate`; other arguments match issue 52's reproduction.
Run `benchmarks/stack_profile.py --root CHECKOUT --output FRESH_DIRECTORY` in the
same GPU environment for each revision. Original timings used the final portable
beam fix `1aa5c39` against `2c5907c`; later rebasing and the model reporting fix
change neither beam execution nor chunk selection. No cumulative speedup against
the historical pre-noise-correction baseline is claimed.

Final stack CPU suite: 581 passed, one live-network test deselected. Focused GPU
suite: 82 passed. Direct numerical tests pass on NumPy 1.26 and NumPy 2;
boresight, near zero, six nulls and signed sidelobes at three frequencies,
homogeneous/mixed dtypes, typed diameters and x64 behavior are covered. Subagent
review found two dtype-promotion issues; both were fixed and re-reviewed.
Large-fixture capacity passes are separate from the five-repeat speed table.
