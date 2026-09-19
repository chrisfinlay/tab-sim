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
        "device_kind": "cpu", "x64": True, "python": "3.11.0", "versions": {"jax": "test"},
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


def test_pytest_collection_keeps_requested_implementation(tmp_path):
    """An installed editable checkout must not win after pytest prepends its root."""
    import subprocess
    import sys
    package = tmp_path / "tabsim"
    package.mkdir()
    (package / "__init__.py").write_text('BENCHMARK_IMPORT_SENTINEL = True\n')
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/conftest.py").write_text(
        'raise RuntimeError("Foreign checkout test fixtures must not load")\n')
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    code = '''
import pathlib, pytest, sys
root = pathlib.Path(sys.argv[2])
rc = pytest.main([str(root / "benchmarks/test_benchmarks.py"), "--collect-only", "-q",
    "-c", str(root / "pytest.ini"), "--rootdir", str(root),
    "--confcutdir", str(root / "benchmarks"), "--source-root=" + sys.argv[1]])
assert rc == 0
import tabsim
assert tabsim.BENCHMARK_IMPORT_SENTINEL
assert pathlib.Path(tabsim.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]))
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path), str(root)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_trace_summary_counts_only_copy_events_and_keeps_unknown_sizes(tmp_path):
    from benchmarks.trace_summary import summarize
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [
        {"name": "MemcpyH2D", "ph": "X", "dur": 2,
         "args": {"memcpy_details": "size:128 dest:0"}},
        {"name": "MemcpyH2D", "ph": "X", "dur": 3},
        {"name": "MemcpyH2D", "ph": "M"},
        {"name": "Other", "ph": "X", "dur": 900},
    ]}))
    result = summarize(path)
    assert result["copies"]["MemcpyH2D"] == {
        "events": 2, "duration_us": 5, "bytes_with_known_size": 128,
        "events_with_known_size": 1}
    assert not result["million_event_warning"]


@pytest.mark.parametrize("limit,expected", [("timeout", "timeout"), ("memory", "host_memory_limit"), ("monitor", "monitor_error")])
def test_supervisor_terminates_only_its_worker(tmp_path, monkeypatch, limit, expected):
    import os
    import sys
    from types import SimpleNamespace
    pytest.importorskip("psutil")
    if os.name != "posix":
        pytest.skip("Process-group supervision is POSIX-only")
    from benchmarks.run import run_one
    worker = tmp_path / "worker"
    worker.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    worker.chmod(0o700)
    args = SimpleNamespace(device="cpu", rounds=5, workers=1, chunk_mb=16,
        host_budget_gib=0.000001 if limit == "memory" else 4,
        gpu_budget_gib=4, timeout=0.1 if limit == "timeout" else 10, trace=False)
    if limit == "monitor":
        def denied(*args, **kwargs):
            raise PermissionError("process inspection unavailable")
        monkeypatch.setattr("benchmarks.run.psutil.Process", denied)
    result = run_one(args, "aa05-point", "zarr", tmp_path, str(worker), tmp_path / "run")
    assert result["status"] == expected
    assert result["returncode"] != 0
    assert result["process_wall_s"] < 10


def test_comparison_rejects_different_python():
    pytest.importorskip("psutil")
    from benchmarks.run import compare_records
    base = sample_record()
    candidate = copy.deepcopy(base)
    candidate["extra_info"]["python"] = "3.13.0"
    with pytest.raises(ValueError, match="python"):
        compare_records(base, candidate)


def test_successful_worker_without_measurements_fails(tmp_path):
    import sys
    from types import SimpleNamespace
    pytest.importorskip("psutil")
    from benchmarks.run import run_one
    worker = tmp_path / "worker"
    worker.write_text(f"#!{sys.executable}\n")
    worker.chmod(0o700)
    args = SimpleNamespace(device="cpu", rounds=5, workers=1, chunk_mb=16,
        host_budget_gib=4, gpu_budget_gib=4, timeout=10, trace=False)
    result = run_one(args, "aa05-point", "zarr", tmp_path, str(worker), tmp_path / "run")
    assert result["status"] == "failed"
    assert result["reason"] == "No benchmark statistics produced"


def test_conversion_samples_both_products_and_rejects_missing_ms(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import numpy as np
    pytest.importorskip("psutil")
    from benchmarks.harness import output_sample, check_samples

    class Array:
        sizes = {"row": 1}
        shape = (1,)
        def __init__(self, value):
            self.value = value
        def isel(self, selection):
            return self
        def compute(self):
            return np.array([self.value])

    class Dataset(dict):
        def close(self):
            pass

    ms = Dataset(DATA=Array(1))
    monkeypatch.setitem(sys.modules, "xarray", SimpleNamespace(
        open_zarr=lambda path: Dataset(vis_obs=Array(1))))
    monkeypatch.setitem(sys.modules, "daskms", SimpleNamespace(
        xds_from_ms=lambda path: [ms]))
    expected = output_sample(tmp_path, "zarr-ms")
    assert set(expected) == {"zarr/vis_obs", "ms/DATA"}
    ms["DATA"] = Array(2)
    with pytest.raises(AssertionError):
        check_samples(output_sample(tmp_path, "zarr-ms"), expected)
    ms.clear()
    ms["FLAG"] = Array(False)
    with pytest.raises(ValueError, match="DATA"):
        output_sample(tmp_path, "zarr-ms")


def test_noise_comparison_only_allows_documented_output_changes():
    pytest.importorskip("psutil")
    from benchmarks.noise_comparison import compare_noise_records
    base = sample_record()
    value = base["extra_info"]["output_sample"]["vis"]
    base["extra_info"]["output_sample"] = {name: copy.deepcopy(value) for name in
        ("vis_ast", "vis_rfi", "vis_obs", "vis_calibrated", "flags")}
    candidate = copy.deepcopy(base)
    candidate["extra_info"]["output_sample"]["vis_obs"]["real"] = [2.]
    assert compare_noise_records(base, candidate)["sampled_signals_match"]
    candidate["extra_info"]["output_sample"]["vis_ast"]["real"] = [2.]
    with pytest.raises(AssertionError):
        compare_noise_records(base, candidate)


def test_noise_scaling_adapter_shares_eager_memory():
    import numpy as np
    from benchmarks.noise_scaling import chunk_views
    original = np.arange(48).reshape(4, 3, 4)
    wrapped = chunk_views(original, (2, 3, 2))
    assert all(np.shares_memory(original, block) for block in wrapped.dask.values())
    original[0, 0, 0] = 99
    np.testing.assert_array_equal(wrapped.compute(), original)


def test_mapped_diagnostics_detects_inner_graph_and_restores_hook():
    import dask.array as da
    from dask import delayed
    import xarray as xr
    from benchmarks.mapped_diagnostics import MappedDiagnostics
    original = xr.map_blocks
    ds = xr.Dataset({"x": (["t"], da.zeros(6, chunks=3))})
    for nested in (True, False):
        def callback(block):
            data = block.x.data
            if nested:
                data = delayed(lambda x: x + 1, pure=True)(data).compute(scheduler="synchronous")
            else:
                data = data + 1
            return xr.Dataset({"x": (["t"], data)})
        with MappedDiagnostics() as diagnostic:
            xr.map_blocks(callback, ds, template=ds).compute(scheduler="threads", num_workers=1)
        assert xr.map_blocks is original
        result = diagnostic.report()["callbacks"]["callback"]
        assert result["calls"] == 2
        assert result["nested_compute_calls"] == (2 if nested else 0)
        assert result["tokenize_calls"] > 0 if nested else result["tokenize_calls"] == 0
