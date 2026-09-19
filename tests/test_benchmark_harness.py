"""Small safety/measurement-contract checks, never performance assertions in CI."""
import copy
import json
from pathlib import Path

import pytest

from benchmarks.cases import CASES, estimates, guard_reason


def test_stress_case_is_larger_than_six_gib_and_guarded():
    case = CASES["aa4-out-of-core"]
    assert estimates(case, "zarr")["single_visibility_bytes"] > 6 * 2**30
    assert "host plan" in guard_reason(case, "zarr", 4 * 2**30, 100 * 2**30)


def test_disk_guard_and_kernel_tile_do_not_allocate_stress_cube():
    case = CASES["aa1-host-stress"]
    assert "disk plan" in guard_reason(case, "zarr", 10**15, 1)
    assert estimates(case, "rfi-kernel")["host_plan_bytes"] < 2**30
    assert guard_reason(case, "rfi-kernel", 4 * 2**30, 1, 4 * 2**30) is None


def test_case_registry_uses_actual_ska_assemblies():
    from importlib.resources import files
    import numpy as np
    import yaml
    data = files("tabsim.data").joinpath("telescopes")
    registry = yaml.safe_load(data.joinpath("_telescopes.yaml").read_text())
    for case in CASES.values():
        with data.joinpath(registry[case["telescope"].lower()]["itrf_path"]).open() as stream:
            assert len(np.loadtxt(stream)) == case["antennas"]
        assert case["samples"] % 2 == 1


def sample_record():
    return {"status": "passed", "stats": {"median": 1., "iqr": .1}, "extra_info": {
        "fixture_sha256": "fixture", "harness_sha256": "harness", "environment": {}, "case": {}, "options": {}, "host": "test",
        "device_kind": "cpu", "x64": True, "versions": {"jax": "test"},
        "output_sample": {"vis": {"shape": [1], "real": [1.], "imag": [0.]}}
    }}


def test_comparison_rejects_different_dependencies_and_output():
    pytest.importorskip("psutil")
    from benchmarks.run import compare_records
    base = sample_record()
    candidate = copy.deepcopy(base)
    candidate["extra_info"]["versions"]["jax"] = "different"
    with pytest.raises(ValueError, match="Dependency"):
        compare_records(base, candidate)
    candidate = copy.deepcopy(base)
    candidate["extra_info"]["output_sample"]["vis"]["real"] = [2.]
    with pytest.raises(AssertionError):
        compare_records(base, candidate)
    candidate["status"] = "failed"
    assert compare_records(base, candidate)["status"] == "unavailable"


def test_gpu_budget_blocks_oversized_kernel():
    assert "GPU budget" in guard_reason(CASES["aa4-mixed"], "rfi-kernel", 10**12, 10**12, 1)
