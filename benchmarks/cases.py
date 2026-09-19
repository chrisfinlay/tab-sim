"""Recorded workload definitions and conservative preflight estimates."""
import hashlib
import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures/cases.json"
CASES = json.loads(FIXTURE.read_text())
MODES = ("astro-kernel", "rfi-kernel", "zarr", "ms", "zarr-ms")


def fixture_hash():
    return hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def estimates(case, mode):
    """Planning estimates, not an XLA memory bound. Never allocate a stress cube."""
    a, t, f, i = (case[k] for k in ("antennas", "times", "channels", "samples"))
    b = a * (a - 1) // 2
    if mode.endswith("kernel"):
        t, f = min(t, 8), min(f, 16)
    cube = 16 * t * b * f
    source = 8 * max(case["rfi_sources"], 1) * t * i * a * f
    # Existing eager complex noise plus temporaries and multiple visibility products.
    host = 512 * 2**20 + 10 * cube + 4 * source
    return {"single_visibility_bytes": cube, "host_plan_bytes": host,
            "disk_plan_bytes": 0 if mode.endswith("kernel") else 10 * cube + 4 * source,
            "kernel_plan_bytes": 4 * cube * i + 4 * source}


def guard_reason(case, mode, host_budget, disk_free, gpu_budget=None):
    e = estimates(case, mode)
    if e["host_plan_bytes"] > host_budget:
        return f"host plan {e['host_plan_bytes']} exceeds budget {host_budget} bytes"
    if e["disk_plan_bytes"] > disk_free:
        return f"disk plan {e['disk_plan_bytes']} exceeds available budget {disk_free} bytes"
    if mode.endswith("kernel") and gpu_budget and e["kernel_plan_bytes"] > gpu_budget:
        return f"kernel plan {e['kernel_plan_bytes']} exceeds GPU budget {gpu_budget} bytes"
    return None
