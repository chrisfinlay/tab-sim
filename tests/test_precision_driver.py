"""Precision comparisons must fail closed, including candidate-only capacity."""

import copy
import json
import sys
import pytest
from benchmarks import precision_sweep as driver


def report():
    return dict(
        validated=True,
        precision="double",
        total_s=1.0,
        writer_s=0.8,
        chunks=[1, 2],
        retained=["vis_obs"],
        samples={
            k: dict(real=[1.0] * 27, imag=[0.0] * 27)
            for k in ("vis_ast", "vis_rfi", "vis_obs", "noise_data")
        },
        case=dict(
            visibility_precision="double",
            antennas=4,
            times=4,
            channels=4,
            samples=3,
            point_sources=1,
            rfi_sources=1,
        ),
        provenance={
            k: "same"
            for k in (
                "device_kind",
                "versions",
                "driver_sha256",
                "python",
                "host",
                "environment",
                "harness_sha256",
                "source_sha256",
            )
        },
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(samples={}),
        lambda r: r.update(total_s=float("nan")),
        lambda r: r.update(chunks=[True, 2]),
        lambda r: r["samples"]["vis_obs"].update(real=[float("nan")] * 27),
        lambda r: r.update(retained=["vis_obs", "vis_ast"]),
        lambda r: r["case"].update(visibility_precision="single"),
    ],
)
def test_invalid_reports_rejected(mutation):
    r = report()
    mutation(r)
    with pytest.raises((ValueError, KeyError)):
        driver.validate_report(r, "double")


def test_comparison_rejects_values_and_provenance():
    a = report()
    b = copy.deepcopy(a)
    driver.compare(a, b)
    b["samples"]["vis_obs"]["real"][0] = 1.1
    with pytest.raises(AssertionError):
        driver.compare(a, b)
    b = copy.deepcopy(a)
    b["provenance"]["host"] = "different"
    with pytest.raises(ValueError):
        driver.compare(a, b)


def test_missing_report_is_never_recorded_as_pass(tmp_path, monkeypatch):
    class Process:
        pid = 999999
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precision",
            "--base",
            "base",
            "--candidate",
            "candidate",
            "--python",
            sys.executable,
            "--output",
            str(tmp_path / "run"),
            "--candidate-only",
            "--rounds",
            "1",
        ],
    )
    with pytest.raises(FileNotFoundError):
        driver.main()
    rows = json.loads((tmp_path / "run/summary.json").read_text())
    assert rows[0]["status"] == "INVALID REPORT"
    assert rows[0]["report"] is None
