# Visibility precision

Simulations now default to **single-precision complex visibilities** (`complex64`,
8 bytes per value). Select double (`complex128`, 16 bytes) when the required
accuracy or dynamic range warrants it:

```python
obs = Observation(..., visibility_precision="double")
```

```yaml
observation:
  visibility_precision: single  # single or double
```

```sh
sim-vis --config_path simulation.yaml --visibility-precision double
```

The CLI option overrides YAML; omitting it preserves the YAML choice. The
Python constructor defaults to single independently of YAML. Dask and JAX
`astro_vis`, `astro_vis_gauss`, `astro_vis_exp`, and `rfi_vis` accept the same
keyword (single by default). When directly applying `jax.jit`, bind the option
with `functools.partial`, or declare `visibility_precision` a static argument.

## What stays double

Antenna/source coordinates, UVW, time, frequency, path differences, phase
formation, and trigonometric phase evaluation remain float64. Gaussian and
exponential source attenuation is evaluated in float64 too. Only after phase
and attenuation evaluation are phasors and intensity/amplitude operands cast
for visibility products and accumulation. RFI time averaging uses the chosen
visibility precision.

`Observation`/`Telescope` enable JAX x64 before geometry construction. Standalone
visibility-kernel callers must enable it themselves; workers that execute a
serialized Dask graph need `JAX_ENABLE_X64=true`. A visibility kernel rejects
disabled x64 rather than silently calculating phases in float32. Supplying
already-rounded float32 coordinates cannot recover their lost input precision.

Gains are generated with the previous double-precision phases/random draws and
then cast to the chosen complex dtype. Noise uses the same double-precision
random draws for a fixed seed, shape, chunk layout and library versions, then
casts each bounded block. Composition and calibration preserve the visibility
dtype. Single precision changes rounding and can change flags very near a
threshold. Long sums, high dynamic range and near-cancelling sources can need
double precision; there is no universal relative-error guarantee near a null.

## Storage and memory

Zarr records `visibility_precision` and preserves the selected dtype for
`vis_ast`, `vis_rfi`, `vis_obs`, `vis_calibrated`, `noise_data`, and `gains_ants`.
Source/geometry ancillary arrays retain their existing precision. The staged
single-store computation and output-selection options are unchanged.

Measurement Sets preserve complex64/complex128 visibility columns through the
public dask-ms descriptor API, including the extra visibility columns. Double
columns round-trip with casacore/dask-ms; support by downstream software that
expects standard single-precision imaging columns must be checked. Updating or
accumulating double data into an existing single-precision column is rejected
rather than silently rounded. Use an explicitly single-precision simulation or
create a new double-precision MS.

Single precision halves **logical visibility storage**, not necessarily the
whole store, compressed bytes, or peak memory. Geometry, phases, source
amplitudes before visibility formation, gain generation and random draws still
need double buffers. Chunk selection and the working-set estimator deliberately
keep their conservative double-precision allowance in both modes. In
particular, `max_chunk_MB` still budgets a fine-time complex128-equivalent tile;
changing precision alone does not enlarge tiles or change noise partitioning.
Runtime host/device guards and disk checks continue to use actual arrays.

## Reproducing measurements

`benchmarks.precision_sweep` alternates isolated parent-double/candidate-single
processes, with a warm-up write followed by a timed write, RSS/timeout guards,
fixed chunks, identical selected outputs, and numerical checks. Use
`--candidate-precision double` for the double-compatibility control. `--case rfi`
uses AA2 with 512 RFI sources; `--case io` uses AA4 and a larger visibility cube.

```sh
python -m benchmarks.precision_sweep \
  --base /path/to/parent --candidate /path/to/candidate \
  --python /path/to/venv/bin/python --output /scratch/precision \
  --case rfi --device cpu --rounds 5
```

Normal-case validation reads occur **after** the timed write. Cold capacity
runs (`--case capacity --channels 544 --candidate-only --cold --rounds 1`)
validate component samples before their scratch arrays are pruned; their wall
times include this validation and are not warm speed comparisons. Outputs are
removed after validation; reports and logs remain. The legacy benchmark
fixtures stay explicitly double precision to preserve historical byte sizes
and comparisons. New precision cases state their precision and actual byte
counts.
