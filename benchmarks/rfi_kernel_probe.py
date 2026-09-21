"""Bounded, resident-input RFI kernel comparison; excludes upstream simulation work."""

import argparse
import json
import math
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [
        ("samples", 3),
        ("times", 8),
        ("sources", 512),
        ("antennas", 68),
        ("channels", 32),
        ("repeats", 5),
    ]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--precision", choices=("single", "double"), default="single")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if any(
        getattr(args, name) < 1
        for name in ("samples", "times", "sources", "antennas", "channels")
    ):
        parser.error("Dimensions must be positive")
    if args.antennas < 2 or args.repeats < 5:
        parser.error("Require at least two antennas and five repeats")
    # Account for the full NumPy input payload before allocating it.
    shape = (args.sources, args.times, args.samples, args.antennas, args.channels)
    distance_shape = shape[:-1]
    baselines = args.antennas * (args.antennas - 1) // 2
    input_bytes = 8 * (
        math.prod(shape) + math.prod(distance_shape) + args.channels + 2 * baselines
    )
    if input_bytes >= 2**30:
        parser.error("Input payload must be smaller than 1 GiB; reduce dimensions")
    output_bytes = (
        (8 if args.precision == "single" else 16)
        * args.times
        * baselines
        * args.channels
    )
    native_operand_bytes = (12 if args.precision == "single" else 24) * math.prod(shape)
    # Both executables and their outputs coexist. Reject obviously unsafe shapes
    # before allocating inputs; this is not a bound on compiler/runtime memory.
    payload_allowance = input_bytes + native_operand_bytes + 4 * output_bytes
    if payload_allowance >= 2**31:
        parser.error(
            "Inputs, native operands and output allowance must total less than 2 GiB"
        )
    if args.output.exists():
        parser.error("Output already exists; choose a fresh JSON path")

    import importlib.metadata
    from functools import partial
    import numpy as np
    import jax
    import jax.numpy as jnp
    from tabsim.jax import interferometry as itf

    jax.config.update("jax_enable_x64", True)
    if not itf.kernel_usable():
        raise RuntimeError(
            "Native RFI library is unavailable on the selected JAX backend"
        )
    rng = np.random.default_rng(4701)
    amplitude = rng.uniform(0.5, 1.5, size=shape)
    # A source-dependent common satellite range plus antenna-scale paths.
    common = rng.uniform(0.9e6, 1.1e6, size=distance_shape[:-1] + (1,))
    distance = common + rng.uniform(-1000.0, 1000.0, size=distance_shape)
    frequency = 100e6 + np.arange(args.channels, dtype=np.float64) * 1e5
    a1, a2 = np.triu_indices(args.antennas, 1)
    host_inputs = (amplitude, distance, frequency, a1, a2)
    transfer_start = time.perf_counter()
    inputs = tuple(jax.device_put(value) for value in host_inputs)
    for value in inputs:
        value.block_until_ready()
    transfer_s = time.perf_counter() - transfer_start
    flux_bound = float(np.sum(np.max(amplitude, axis=(1, 2, 3, 4)) ** 2))
    del host_inputs, amplitude, distance, common, frequency, a1, a2

    compiled, results = {}, {}
    for label, function in [("jax", itf.rfi_vis_jax), ("native", itf.rfi_vis_kernel)]:
        start = time.perf_counter()
        executable = (
            jax.jit(partial(function, visibility_precision=args.precision))
            .lower(*inputs)
            .compile()
        )
        compile_s = time.perf_counter() - start
        start = time.perf_counter()
        value = executable(*inputs)
        value.block_until_ready()
        first_execution_s = time.perf_counter() - start
        memory = executable.memory_analysis()
        fields = (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "alias_size_in_bytes",
        )
        results[label] = dict(
            lower_and_compile_s=compile_s,
            first_execution_s=first_execution_s,
            compiler_memory=(
                None
                if memory is None
                else {key: getattr(memory, key, None) for key in fields}
            ),
            warm_seconds=[],
        )
        compiled[label] = executable
    # Alternate AB/BA; resident inputs and synchronized results isolate these kernels.
    outputs = {}
    for index in range(args.repeats):
        for label in (("jax", "native") if index % 2 == 0 else ("native", "jax")):
            start = time.perf_counter()
            value = compiled[label](*inputs)
            value.block_until_ready()
            results[label]["warm_seconds"].append(time.perf_counter() - start)
            outputs[label] = value
    reference = np.asarray(outputs["jax"])
    actual = np.asarray(outputs["native"])
    expected_dtype = np.dtype(
        "complex64" if args.precision == "single" else "complex128"
    )
    if (
        actual.shape != reference.shape
        or actual.dtype != expected_dtype
        or reference.dtype != expected_dtype
    ):
        raise AssertionError("RFI kernel output shape/dtype mismatch")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise AssertionError("Nonfinite RFI kernel output")
    max_error = float(np.max(np.abs(actual - reference)))
    normalized_error = max_error / flux_bound
    tolerance = 5e-6 if args.precision == "single" else 2e-9
    for result in results.values():
        values = result["warm_seconds"]
        median = statistics.median(values)
        result.update(
            median_s=median,
            mad_s=statistics.median(abs(v - median) for v in values),
            min_s=min(values),
            max_s=max(values),
        )
    report = dict(
        status="PASS" if normalized_error <= tolerance else "FAIL",
        precision=args.precision,
        seed=4701,
        repeats=args.repeats,
        inputs=[dict(shape=list(v.shape), dtype=str(v.dtype)) for v in inputs],
        input_payload_bytes=input_bytes,
        output_payload_bytes=output_bytes,
        native_operand_bytes=native_operand_bytes,
        payload_allowance_bytes=payload_allowance,
        initial_device_transfer_s=transfer_s,
        output_shape=list(actual.shape),
        output_dtype=str(actual.dtype),
        device=str(inputs[0].device),
        device_kind=inputs[0].device.device_kind,
        x64=bool(jax.config.jax_enable_x64),
        versions={
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "ri-kernels")
        },
        implementations=results,
        correctness=dict(
            max_absolute_error=max_error,
            flux_bound=flux_bound,
            flux_normalized_error=normalized_error,
            tolerance=tolerance,
        ),
        scope="Resident-input synchronized kernels only. Excludes source geometry, beam formation, Dask, output storage and host readback. Compile timings can use persistent compiler caches. Compiler memory is not measured peak RSS/VRAM.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    if report["status"] != "PASS":
        raise SystemExit("Native kernel exceeded the flux-normalized error tolerance")


if __name__ == "__main__":
    main()
