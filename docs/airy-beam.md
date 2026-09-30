# Host-block Airy beam evaluation

The Dask Airy wrapper now calls `tabsim.beam.airy_beam`, using NumPy and SciPy for
the entire block. The previous implementation created JAX arrays for angles,
frequencies and diameter, copied the Bessel argument to SciPy on the host, then
mixed its NumPy result with a JAX denominator. On GPU this introduced avoidable
transfers and synchronization in every block.

The expression remains `2 * scipy.special.jv(1, x) / x`, with
`x = pi * frequency * diameter * sin(theta_radians) / c`. Boresight uses the same
machine-epsilon substitution. Signed sidelobes, nulls and behavior beyond 90
degrees are preserved. No clipping, alternate Bessel approximation, GPU Bessel
implementation or change to visibility kernels is introduced.

Inputs have shapes `(source, time, antenna)` and `(frequency,)`. The result is a
NumPy array `(source, time, antenna, frequency)`. The historical
`tabsim.jax.interferometry.airy_beam` name remains an alias, but now explicitly
returns a host ndarray. The old function was not JIT-compatible either (SciPy
requires concrete host values). Direct callers supplying device arrays incur
input conversion to host; the Dask path supplies host blocks.

x64 regression tolerance against the previous expression is `rtol=2e-12,
atol=2e-14`, including boresight, tiny angles, negative angles, sidelobes, six
Bessel nulls and their neighbourhoods at 50 MHz, 150 MHz and 1.4 GHz. Float32 is
checked separately at `rtol=3e-6, atol=3e-8`. A public JAX transfer-guard test forbids
implicit transfers during both direct host evaluation and mapped Dask execution.
Runtime and profiler results are recorded separately from numerical equivalence;
removing transfers alone is not evidence of an end-to-end speedup.

Host precision follows NumPy input promotion (including explicitly typed float64
diameters) independently of JAX's global x64 flag. In particular, float64 host
inputs are no longer silently truncated when JAX x64 is disabled. Python scalar
diameters remain weakly typed under NumPy 2. Mixed floating dtypes are tested.

## Device beam for RFI

RFI visibilities evaluate the beam on the device instead. `tabsim.jax.beam.j1` is a
float64 port of the Cephes `j1` bundled with SciPy (`xsf/cephes/j1.h`), and
`tabsim.jax.beam.airy_beam` repeats the host expression, operation order and
boresight substitution. `jax.scipy.special.bessel_jn` is not used: it loses accuracy
above x of about 40, and RFI arguments reach about 200.
`tabsim.jax.interferometry.rfi_vis_beam` fuses `Pv_to_Sv`, the beam and `rfi_vis` in
one jitted block, so computing `vis_rfi` never transfers the fine-time amplitude to
the host. Saved `rfi_*_A` amplitudes and the fringe diagnostics are computed lazily
by `rfi_amplitude`, which transfers bounded tiles through `execute_kernel`. Both
entry points require JAX x64. `J1` matches `scipy.special.j1` within `5e-16` on
[-1000, 1000] on every backend, keeps signed zeros, and the beam keeps the host
tolerances above. Astronomical sources keep the host beam.
