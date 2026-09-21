"""One isolated cold/warm staged-write measurement at fixed geometry and chunks."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import platform
import socket

p = argparse.ArgumentParser()
for name in ("root", "output"):
    p.add_argument("--" + name, required=True)
p.add_argument("--precision", choices=("single", "double"), required=True)
p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
p.add_argument("--case", choices=("rfi", "io", "capacity"), default="rfi")
p.add_argument("--channels", type=int, default=544)
p.add_argument("--cold", action="store_true")
a = p.parse_args()
os.environ.update(
    JAX_PLATFORMS="cuda" if a.device == "gpu" else "cpu",
    JAX_ENABLE_X64="true",
    XLA_PYTHON_CLIENT_MEM_FRACTION=".5",
)
sys.path.insert(0, a.root)
import dask
import jax
import numpy as np
import zarr
from benchmarks.harness import build_observation, add_sources, offline, MemorySampler
import tabsim

assert Path(tabsim.__file__).resolve().is_relative_to(Path(a.root).resolve())
from tabsim.execution import configure_execution

configure_execution(gpu_concurrency=1)
offline()
case = dict(
    telescope="SKA-Low-AA2",
    antennas=68,
    times=16,
    channels=32,
    samples=3,
    point_sources=8,
    rfi_sources=512,
)
if a.case != "rfi":
    case.update(
        telescope="SKA-Low-AA4",
        antennas=512,
        times=8 if a.case == "io" else 32,
        channels=64 if a.case == "io" else a.channels,
        rfi_sources=2,
    )
case["visibility_precision"] = a.precision
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)


def run(name):
    start = time.perf_counter()
    store = out / (name + ".zarr")
    stages = []
    samples = {}
    indices = (
        [0, case["times"] // 2, case["times"] - 1],
        [0, 100, case["antennas"] * (case["antennas"] - 1) // 2 - 1],
        [0, case["channels"] // 2, case["channels"] - 1],
    )

    def progress(event, detail):
        if event != "stage_complete":
            return
        stages.append(detail)
        if detail["variable"] == "vis_obs" and a.case == "capacity":
            validate_components()

    def validate_components():
        group = zarr.open_group(str(store), mode="r")
        for key in ("vis_obs", "vis_ast", "vis_rfi", "noise_data"):
            samples[key] = group[key].get_orthogonal_selection(indices)
        pairs = np.triu_indices(case["antennas"], 1)
        gains = [
            group["gains_ants"].get_orthogonal_selection(
                (indices[0], pair[indices[1]], indices[2])
            )
            for pair in pairs
        ]
        expected = (
            gains[0] * (samples["vis_ast"] + samples["vis_rfi"]) * gains[1].conj()
            + samples["noise_data"]
        )
        np.testing.assert_allclose(samples["vis_obs"], expected, rtol=2e-6, atol=2e-6)

    with dask.config.set(scheduler="synchronous", num_workers=2):
        obs = build_observation(case, 64 if a.case != "rfi" else 45)
        add_sources(obs, case)
        obs.calculate_vis()
        dtype = np.dtype("complex64" if a.precision == "single" else "complex128")
        assert (
            obs.vis_ast.dtype == dtype
        ), "Selected checkout did not apply requested precision"
        writer_start = time.perf_counter()
        ds = obs.write_to_zarr(
            store,
            save_arrays=["vis_obs"] if a.case == "capacity" else None,
            progress=progress,
            disk_reserve_gb=2,
        )
        end = time.perf_counter()
        elapsed = end - start
        writer_s = end - writer_start
        if a.case != "capacity":
            validate_components()
            for key in ("vis_calibrated", "gains_ants"):
                samples[key] = np.asarray(
                    ds[key]
                    .isel(
                        time=indices[0],
                        freq=indices[2],
                        **(
                            {"bl": indices[1]}
                            if key == "vis_calibrated"
                            else {"ant": [0, 1, case["antennas"] - 1]}
                        ),
                    )
                    .compute()
                )
        for key in (
            "vis_obs",
            "vis_ast",
            "vis_rfi",
            "noise_data",
            "vis_calibrated",
            "gains_ants",
        ):
            if key in ds:
                assert ds[key].dtype == dtype
        assert ds.vis_obs.dtype == dtype
        np.testing.assert_array_equal(
            ds.vis_obs.isel(time=indices[0], bl=indices[1], freq=indices[2]).compute(),
            samples["vis_obs"],
        )
        assert all(np.isfinite(v).all() for v in samples.values())
        result = dict(
            total_s=elapsed,
            writer_s=writer_s,
            precision=a.precision,
            case=case,
            chunks=[obs.time_chunk, obs.freq_chunk],
            stages=stages,
            retained=sorted(ds.data_vars),
            single_visibility_bytes=ds.vis_obs.nbytes,
            logical_bytes=ds.nbytes,
            output_bytes=sum(f.stat().st_size for f in store.rglob("*") if f.is_file()),
            samples={
                k: dict(real=v.ravel().real.tolist(), imag=v.ravel().imag.tolist())
                for k, v in samples.items()
            },
            validated=True,
        )
        ds.close()
    shutil.rmtree(store)
    return result


if not a.cold:
    warmup = run("warmup")
with MemorySampler() as memory:
    result = run("timed")
if not a.cold:
    assert result["samples"] == warmup["samples"], "Warm repetitions changed values"
result.update(
    peak_rss=memory.peak_rss,
    peak_gpu_reserved=memory.peak_gpu,
    jax_memory=jax.devices()[0].memory_stats(),
    cold=a.cold,
)
result["provenance"] = dict(
    python=platform.python_version(),
    host=socket.gethostname(),
    environment={
        key: os.environ.get(key)
        for key in (
            "JAX_PLATFORMS",
            "JAX_ENABLE_X64",
            "CUDA_VISIBLE_DEVICES",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "OMP_NUM_THREADS",
        )
    },
    harness_sha256=hashlib.sha256(
        Path(sys.modules["benchmarks.harness"].__file__).read_bytes()
    ).hexdigest(),
    device_kind=jax.devices()[0].device_kind,
    versions={
        n: importlib.metadata.version(n)
        for n in ("jax", "jaxlib", "numpy", "scipy", "dask", "xarray", "zarr")
    },
    source_sha256={
        str(f.relative_to(a.root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted((Path(a.root) / "tabsim").rglob("*.py"))
    },
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
(out / "result.json").write_text(json.dumps(result, indent=2))
