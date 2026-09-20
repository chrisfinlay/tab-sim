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
