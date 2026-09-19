"""Measurement adapters around the public simulation API (no algorithm changes)."""
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import socket
import subprocess
import threading
import time

import numpy as np
import psutil


def offline():
    """Forbid network connections, including accidental orbital/IERS downloads."""
    def forbidden(*args, **kwargs):
        raise RuntimeError("Benchmark fixtures are offline: network access attempted")
    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    from astropy.utils import iers
    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "ignore"


def provenance(root, case, options):
    import jax
    import tabsim
    from .cases import fixture_hash
    root = Path(root).resolve()
    if not Path(tabsim.__file__).resolve().is_relative_to(root):
        raise RuntimeError(f"Wrong tabsim import: {tabsim.__file__}; expected {root}")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    versions = {}
    for name in ("tabsim", "jax", "jaxlib", "numpy", "scipy", "dask", "xarray", "zarr",
                 "dask-ms", "python-casacore", "pytest-benchmark", "satchecker-client", "ri-kernels"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
            "fixture_sha256": fixture_hash(),
            "harness_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest(),
            "host": socket.gethostname(),
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "host_total_bytes": psutil.virtual_memory().total,
            "python": platform.python_version(), "versions": versions,
            "jax_devices": [str(x) for x in jax.devices()],
            "device_kind": jax.devices()[0].device_kind,
            "x64": bool(jax.config.jax_enable_x64),
            "kernel_entrypoints": "tabsim.jax.interferometry (selected checkout)",
            "kernel_source_sha256": hashlib.sha256((root / "tabsim/jax/interferometry.py").read_bytes()).hexdigest(),
            "environment": {k: os.getenv(k) for k in (
                "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION", "OMP_NUM_THREADS")},
            "case": case, "options": options}


class MemorySampler:
    """Sample RSS and NVML process allocation; these are NOT live XLA buffer sizes."""
    def __init__(self):
        self.stop = threading.Event()
        self.peak_rss = 0
        self.peak_gpu = None
        self.nvml = None
        self.handles = []
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                            for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            pass

    def sample(self):
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        if self.nvml:
            try:
                total = sum(p.usedGpuMemory for h in self.handles
                            for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(h)
                            if p.pid == os.getpid() and p.usedGpuMemory < 2**60)
                self.peak_gpu = max(self.peak_gpu or 0, total)
            except Exception:
                pass

    def __enter__(self):
        self.sample()
        def loop():
            while not self.stop.wait(0.05):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.sample()
        if self.nvml:
            self.nvml.nvmlShutdown()

    def report(self):
        import jax
        return {"rss_peak_sampled_bytes": self.peak_rss,
                "gpu_process_peak_sampled_bytes": self.peak_gpu,
                "sample_interval_s": 0.05,
                "jax_allocator_stats": jax.devices()[0].memory_stats(),
                "scope": "whole benchmark process; allocator peaks include cold/warm/diagnostic work"}


class Diagnostics:
    """Separate instrumented run; never mix its timing with warm measurements."""
    def __enter__(self):
        import jax
        from dask.callbacks import Callback
        self.tasks = 0
        self.compiles = 0
        self.stack = contextlib.ExitStack()
        def task(*args):
            self.tasks += 1
        self.stack.enter_context(Callback(posttask=task))
        owner = self
        class Count(logging.Handler):
            def emit(self, record):
                if "Finished XLA compilation" in record.getMessage():
                    owner.compiles += 1
        self.handler = Count()
        self.logger = logging.getLogger("jax")
        self.logger.addHandler(self.handler)
        self.stack.enter_context(jax.log_compiles(True))
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        self.stack.close()


def build_observation(case, chunk_mb):
    from tabsim.config import get_telescope_definitions
    from tabsim.dask.observation import Observation
    definition = get_telescope_definitions(case["telescope"])
    t, f = case["times"], case["channels"]
    obs = Observation(latitude=definition["latitude"], longitude=definition["longitude"],
        elevation=definition["elevation"], ra=30.0, dec=-30.0,
        times_mjd=60000.0 + np.arange(t) * 2.0 / 86400.0,
        freqs=150e6 + np.arange(f) * 1e5, SEFD=np.full(f, 5000.0),
        ITRF_path=definition["itrf_path"], dish_d=definition["dish_d"],
        int_time=2.0, chan_width=1e5, n_int_samples=case["samples"],
        random_seed=20260919, tel_name=case["telescope"], max_chunk_MB=chunk_mb)
    assert obs.n_ant == case["antennas"]
    assert obs.n_int_samples == case["samples"]
    return obs


def add_sources(obs, case):
    # Analytic fixed fixtures avoid stochastic catalogue/placement rejection and network.
    n = case["point_sources"]
    if n:
        idx = np.arange(n)
        obs.addAstro((0.1 + idx[:, None, None] / max(n, 1))
                     * (np.asarray(obs.freqs)[None, None, :] / 150e6) ** -0.7,
                     30.0 + 0.4 * np.sin(idx), -30.0 + 0.4 * np.cos(idx))
    n = case["rfi_sources"]
    if n:
        idx = np.arange(n)
        obs.addStationaryRFI(np.full((n, 1, case["channels"]), 1e-12),
                             -26.7 + idx * 0.01, 116.9 + idx * 0.01,
                             np.full(n, 1000.0))
    obs.addGains(G0_mean=1.0, G0_std=0.01, Gt_std_amp=0.01, Gt_std_phase=0.01,
                 Gt_corr_amp=3, Gt_corr_phase=3, random_seed=1234)


def simulation(case, mode, chunk_mb, directory):
    """Actual Observation → graph → output path. Deletion/validation is untimed."""
    from tabsim.write import write_ms
    import xarray as xr
    start = time.perf_counter()
    obs = build_observation(case, chunk_mb)
    setup = time.perf_counter()
    add_sources(obs, case)
    obs.calculate_vis()
    graph = time.perf_counter()
    if mode == "ms":
        write_ms(obs.dataset, str(directory / "result.ms"))
    else:
        obs.write_to_zarr(str(directory / "result.zarr"))
    first_write = time.perf_counter()
    if mode == "zarr-ms":
        with xr.open_zarr(directory / "result.zarr") as ds:
            write_ms(ds, str(directory / "result.ms"))
    end = time.perf_counter()
    return {"setup_s": setup - start, "graph_build_s": graph - setup,
            "simulation_and_first_write_s": first_write - graph,
            "zarr_to_ms_s": end - first_write if mode == "zarr-ms" else None,
            "total_s": end - start,
            "actual_chunks": {"time": obs.time_chunk, "frequency": obs.freq_chunk,
                              "baseline": obs.bl_chunk, "integration": obs.n_int_samples}}


def output_sample(directory, mode):
    """Read bounded samples after timing. A regression signature, not a full-cube proof."""
    import xarray as xr
    if mode == "ms":
        from daskms import xds_from_ms
        ds = xds_from_ms(str(directory / "result.ms"))[0]
        names = ("DATA", "MODEL_DATA", "RFI_MODEL_DATA", "AST_MODEL_DATA", "FLAG")
    else:
        ds = xr.open_zarr(directory / "result.zarr")
        names = ("vis_ast", "vis_rfi", "vis_obs", "vis_calibrated", "flags")
    result = {}
    try:
        for name in names:
            if name not in ds:
                continue
            arr = ds[name]
            selection = {dim: sorted({0, size // 2, size - 1})
                         for dim, size in arr.sizes.items()}
            data = np.asarray(arr.isel(selection).compute()).ravel()
            if not np.isfinite(data).all():
                raise ValueError(f"Nonfinite {name} output")
            result[name] = {"shape": list(arr.shape), "real": data.real.tolist(),
                            "imag": data.imag.tolist() if np.iscomplexobj(data) else None}
    finally:
        ds.close()
    if not result:
        raise ValueError("No visibility products were sampled")
    return result


def kernel_inputs(case, mode):
    """Fixed small tile using the selected layout; independent from end-to-end data."""
    from tabsim.config import get_telescope_definitions
    definition = get_telescope_definitions(case["telescope"])
    xyz = np.loadtxt(definition["itrf_path"])
    a1, a2 = np.triu_indices(len(xyz), 1)
    t, f = min(case["times"], 8), min(case["channels"], 16)
    freqs = 150e6 + np.arange(f) * 1e5
    if mode == "astro-kernel":
        n = case["point_sources"]
        rng = np.random.default_rng(20260919)
        lm = rng.uniform(-0.005, 0.005, (n, 2))
        lmn = np.column_stack([lm, np.sqrt(1 - np.sum(lm**2, axis=1))])
        return (rng.uniform(.1, 1., (n, t, f)),
                np.broadcast_to(xyz[a1] - xyz[a2], (t, len(a1), 3)).copy(), lmn, freqs)
    n, i = case["rfi_sources"], case["samples"]
    rng = np.random.default_rng(20260919)
    # A deterministic synthetic emitter trajectory in ECEF; no orbital acquisition.
    positions = xyz.mean(0) + rng.normal(size=(n, t, i, 3)) * 1000 + [2e5, 3e5, 4e5]
    distances = np.linalg.norm(positions[..., None, :] - xyz, axis=-1)
    return (rng.uniform(.1, 1., (n, t, i, len(xyz), f)), distances, freqs, a1, a2)


def check_samples(actual, expected, rtol=1e-7, atol=1e-8):
    if actual.keys() != expected.keys():
        raise AssertionError("Output product names changed")
    for key in actual:
        if actual[key]["shape"] != expected[key]["shape"]:
            raise AssertionError(f"Output shape changed: {key}")
        for part in ("real", "imag"):
            if expected[key][part] is None:
                assert actual[key][part] is None
            else:
                np.testing.assert_allclose(actual[key][part], expected[key][part],
                                           rtol=rtol, atol=atol, err_msg=f"{key}.{part}")


def kernel_reference_sample(host, mode, shape, indices):
    """Independent NumPy scalar reference: sum sources, average RFI integrations."""
    result = []
    for index in indices:
        t, b, f = np.unravel_index(index, shape)
        if mode == "astro-kernel":
            intensity, uvw, lmn, freqs = host
            phase = 2 * np.pi * freqs[f] / 299792458.0 * ((lmn - [0, 0, 1]) @ uvw[t, b])
            value = np.sum(intensity[:, t, f] * np.exp(1j * phase))
        else:
            amp, distance, freqs, a1, a2 = host
            phase = -2 * np.pi * freqs[f] / 299792458.0 * (distance[:, t, :, a1[b]] - distance[:, t, :, a2[b]])
            intensity = amp[:, t, :, a1[b], f] * np.conj(amp[:, t, :, a2[b], f])
            value = np.sum(np.mean(intensity * np.exp(1j * phase), axis=1))
        result.append(value)
    return np.asarray(result)
