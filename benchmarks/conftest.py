import os
from pathlib import Path
import sys

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("tabsim benchmarks")
    group.addoption("--case", default="aa05-point")
    group.addoption("--mode", default="zarr")
    group.addoption("--rounds", type=int, default=5)
    group.addoption("--device", choices=("cpu", "gpu"), default="cpu")
    group.addoption("--chunk-mb", type=float, default=16.0)
    group.addoption("--workers", type=int, default=1)
    group.addoption("--host-budget-gib", type=float, default=4.0)
    group.addoption("--gpu-budget-gib", type=float, default=4.0)
    group.addoption("--source-root", default=str(Path(__file__).resolve().parents[1]))
    group.addoption("--trace-dir", default=None)


def pytest_configure(config):
    # Pin this harness before adding another implementation checkout to sys.path.
    harness_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(harness_root))
    import benchmarks
    if Path(benchmarks.__file__).resolve().parent != harness_root / "benchmarks":
        raise pytest.UsageError("An unrelated benchmarks package was imported")
    # Must precede imports of JAX or tabsim. Never silently fall back from GPU.
    sys.path.insert(0, str(Path(config.getoption("--source-root")).resolve()))
    os.environ["JAX_PLATFORMS"] = "cuda" if config.getoption("--device") == "gpu" else "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    if config.getoption("--rounds") < 5:
        raise pytest.UsageError("Use at least five measured warm rounds")
    for name in ("--workers", "--chunk-mb", "--host-budget-gib", "--gpu-budget-gib"):
        if config.getoption(name) <= 0:
            raise pytest.UsageError(f"{name} must be positive")
