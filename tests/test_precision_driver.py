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


@pytest.mark.parametrize(
    "option,value",
    [
        ("--samples", "0"),
        ("--samples", "2"),
        ("--samples", "-1"),
        ("--times", "0"),
        ("--times", "1"),
    ],
)
def test_invalid_integration_or_time_rejected_before_launch(
    tmp_path, monkeypatch, option, value
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid workload must not launch a worker")

    monkeypatch.setattr(driver.subprocess, "Popen", forbidden)
    output = tmp_path / "run"
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
            str(output),
            option,
            value,
        ],
    )
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("implementation", [None, "jax", "unknown"])
def test_required_native_report_rejects_fallback_or_missing_evidence(implementation):
    row = report()
    if implementation is not None:
        row["rfi_implementation"] = implementation
    with pytest.raises(ValueError):
        driver.validate_report(row, "double", require_native=True)
    # Ordinary parent reports need not originate from a native-enabled checkout.
    if implementation in (None, "jax"):
        driver.validate_report(row, "double")


def _fake_precision_worker(monkeypatch, commands, *, candidate_native=True):
    from pathlib import Path

    class Process:
        pid = 999999
        returncode = 0

        def poll(self):
            return self.returncode

    def launch(command, **kwargs):
        commands.append(command)

        def option(name):
            return command[command.index(name) + 1]

        row = report()
        precision = option("--precision")
        row.update(
            precision=precision,
            rfi_implementation=(
                "native"
                if candidate_native and option("--root") == "candidate"
                else "jax"
            ),
        )
        row["case"].update(
            visibility_precision=precision,
            samples=int(option("--samples")),
            times=int(option("--times")),
        )
        output = Path(option("--output"))
        output.mkdir(parents=True)
        (output / "result.json").write_text(json.dumps(row))
        return Process()

    monkeypatch.setattr(driver.subprocess, "Popen", launch)


def test_native_requirement_only_reaches_candidate_and_preserves_workload(
    tmp_path, monkeypatch
):
    commands = []
    _fake_precision_worker(monkeypatch, commands)
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
            "--rounds",
            "2",
            "--samples",
            "9",
            "--times",
            "8",
            "--base-precision",
            "single",
            "--candidate-precision",
            "single",
            "--require-native",
        ],
    )
    driver.main()
    roots = [cmd[cmd.index("--root") + 1] for cmd in commands]
    assert roots == ["base", "candidate", "candidate", "base"]
    for command, root in zip(commands, roots):
        assert ("--require-native" in command) == (root == "candidate")
        assert command[command.index("--samples") + 1] == "9"
        assert command[command.index("--times") + 1] == "8"
        assert command[command.index("--precision") + 1] == "single"
    rows = json.loads((tmp_path / "run/summary.json").read_text())
    assert all(row["status"] == "PASS" for row in rows)


def test_candidate_only_native_fallback_is_invalid_not_pass(tmp_path, monkeypatch):
    commands = []
    _fake_precision_worker(monkeypatch, commands, candidate_native=False)
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
            "--rounds",
            "1",
            "--candidate-only",
            "--require-native",
        ],
    )
    with pytest.raises(ValueError):
        driver.main()
    rows = json.loads((tmp_path / "run/summary.json").read_text())
    assert len(rows) == 1 and rows[0]["status"] == "INVALID REPORT"
    assert rows[0]["report"] is None
