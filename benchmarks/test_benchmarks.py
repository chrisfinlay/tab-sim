"""Run explicitly: pytest benchmarks --case aa1-mixed --mode zarr ..."""
import gc
import json
from pathlib import Path
import shutil
import time

import pytest

from benchmarks.cases import CASES, MODES, estimates, guard_reason


def test_workload(benchmark, request, tmp_path):
    started = time.perf_counter()
    import dask
    import jax
    import numpy as np
    import psutil
    from benchmarks.harness import (Diagnostics, MemorySampler, check_samples,
        kernel_inputs, kernel_reference_sample, offline, output_sample, provenance, simulation)

    get = request.config.getoption
    name, mode = get("--case"), get("--mode")
    if name not in CASES or mode not in MODES:
        raise pytest.UsageError(f"Unknown case/mode: {name}/{mode}")
    case = CASES[name]
    applicable = case["point_sources"] if mode == "astro-kernel" else case["rfi_sources"]
    if mode.endswith("kernel") and not applicable:
        pytest.skip("This fixture has no sources for that kernel")
    # Honor the configured cap/fraction and leave at least 2 GiB available.
    budget = min(get("--host-budget-gib") * 2**30, psutil.virtual_memory().available * get("--available-memory-fraction"),
                 max(0, psutil.virtual_memory().available - 2 * 2**30))
    reason = guard_reason(case, mode, budget, max(0, shutil.disk_usage(tmp_path).free - max(2 * 2**30, shutil.disk_usage(tmp_path).total * 0.05)),
                          get("--gpu-budget-gib") * 2**30 if get("--device") == "gpu" else None,
                          get("--chunk-mb"), get("--workers"), get("--memory-model"), get("--device"))
    if reason:
        pytest.skip("Memory preflight: " + reason)
    offline()
    opts = {k: get("--" + k.replace("_", "-")) for k in (
        "device", "rounds", "chunk_mb", "workers", "host_budget_gib", "gpu_budget_gib", "memory_model", "capacity", "available_memory_fraction")}
    info = benchmark.extra_info
    info.update(provenance(get("--source-root"), case, opts))
    info.update(case_id=name, mode=mode, estimates=estimates(case, mode, get("--chunk-mb"), get("--workers"), get("--memory-model"), get("--device")),
                runtime_import_s=time.perf_counter() - started)
    info["measurement_notes"] = {
        "startup": "runtime_import_s excludes Python/pytest startup; runner records process_wall_s",
        "warm": "five or more single-iteration rounds after an explicit first call",
        "memory": "50ms whole-process sampling includes untimed validation and diagnostics",
        "transfer_counts": "isolated explicit copy timings/bytes only; pipeline counts require optional trace",
        "diagnostics": "separate untimed instrumented round; not included in benchmark statistics",
    }
    with dask.config.set(scheduler="threads", num_workers=get("--workers")), MemorySampler() as memory:
        if mode.endswith("kernel"):
            from tabsim.jax import interferometry as itf
            host = kernel_inputs(case, mode)
            info["input_bytes"] = sum(x.nbytes for x in host)
            start = time.perf_counter()
            device = jax.device_put(host)
            jax.block_until_ready(device)
            info["host_to_device_s"] = time.perf_counter() - start
            info["input_shapes"] = [list(x.shape) for x in host]
            kernel = jax.jit(itf.astro_vis if mode == "astro-kernel" else itf.rfi_vis)
            start = time.perf_counter()
            lowered = kernel.lower(*device)
            info["trace_and_lower_s"] = time.perf_counter() - start
            start = time.perf_counter()
            compiled = lowered.compile()
            info["compile_s"] = time.perf_counter() - start
            analysis = compiled.memory_analysis()
            info["compiled_memory_analysis"] = {key: getattr(analysis, key, None) for key in (
                "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes",
                "temp_size_in_bytes", "host_temp_size_in_bytes")}
            start = time.perf_counter()
            first = compiled(*device).block_until_ready()
            info["first_execute_s"] = time.perf_counter() - start
            start = time.perf_counter()
            expected = np.asarray(jax.device_get(first))
            info["device_to_host_s"] = time.perf_counter() - start
            info["output_bytes"] = expected.nbytes
            def target():
                return compiled(*device).block_until_ready()
            result = benchmark.pedantic(target, iterations=1, rounds=get("--rounds"))
            np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-7, atol=1e-8)
            assert np.isfinite(expected).all()
            idx = sorted({0, expected.size // 2, expected.size - 1})
            sample = expected.ravel()[idx]
            np.testing.assert_allclose(sample, kernel_reference_sample(host, mode, expected.shape, idx),
                                       rtol=1e-7, atol=1e-8)
            info["numpy_reference_sample_passed"] = True
            info["output_sample"] = {"kernel": {"shape": list(expected.shape),
                                      "real": sample.real.tolist(), "imag": sample.imag.tolist()}}
            with Diagnostics() as diagnostics:
                target()
        else:
            path = tmp_path / "output"
            phases = []
            samples = []
            def setup():
                path.mkdir()
            def progress(stage, details):
                print("CAPACITY " + json.dumps({"stage": stage, "elapsed_s": time.perf_counter() - started,
                      "rss_bytes": psutil.Process().memory_info().rss, "details": details}), flush=True)
            def target():
                record = simulation(case, mode, get("--chunk-mb"), path,
                                    progress=progress if get("--capacity") else None)
                phases.append(record)
            def teardown():
                samples.append(output_sample(path, mode))
                if len(samples) > 1:
                    check_samples(samples[-1], samples[0])
                info["output_bytes"] = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                shutil.rmtree(path)
                gc.collect()
            if get("--capacity"):
                # One complete cold simulation and bounded readback; no speed claim.
                benchmark.pedantic(target, setup=setup, teardown=teardown, rounds=1, iterations=1)
                info["capacity_phases"] = phases
                info["output_sample"] = samples[0]
                info["measurement_notes"]["warm"] = "capacity: one cold execution; no warm timing or repeatability claim"
                info["measurement_notes"]["diagnostics"] = "capacity: no extra diagnostic execution"
                info["memory"] = memory.report()
                return
            setup()
            try:
                target()
                teardown()
            except Exception as exc:
                # #42 is not xfailed: record/report MS-only failures as real failures.
                info["cold_error"] = f"{type(exc).__name__}: {exc}"
                raise
            info["cold_phases"] = phases.pop()
            benchmark.pedantic(target, setup=setup, teardown=teardown,
                               rounds=get("--rounds"))
            info["warm_phases"] = list(phases)
            info["output_sample"] = samples[0]
            # Additional diagnostics are deliberately excluded from measured rounds.
            setup()
            with Diagnostics() as diagnostics:
                if get("--trace-dir"):
                    trace_kwargs = {}
                    if hasattr(jax.profiler, "ProfileOptions"):
                        options = jax.profiler.ProfileOptions()
                        options.host_tracer_level = 1
                        options.python_tracer_level = 0
                        trace_kwargs["profiler_options"] = options
                        info["trace_options"] = {"host_tracer_level": 1, "python_tracer_level": 0}
                    else:
                        info["trace_options"] = "JAX defaults; check for event truncation"
                    with jax.profiler.trace(get("--trace-dir"), create_perfetto_link=False, **trace_kwargs):
                        target()
                else:
                    target()
            teardown()
            # Count the unoptimized combined dataset graph separately from execution.
            from benchmarks.harness import build_observation, add_sources
            obs = build_observation(case, get("--chunk-mb"))
            add_sources(obs, case)
            obs.calculate_vis()
            info["unoptimized_graph_tasks"] = len(obs.dataset.__dask_graph__())
        info["mapped_callback_diagnostics"] = diagnostics.mapped.report()
        info["diagnostic_tasks_executed"] = diagnostics.tasks
        info["diagnostic_xla_compile_log_events"] = diagnostics.compiles
    info["memory"] = memory.report()
