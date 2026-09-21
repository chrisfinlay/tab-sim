"""AB/BA precision comparisons with subprocess RSS/timeout supervision."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import psutil
from benchmarks.output_sweep import stop_group


def validate_report(report, precision, require_native=False):
    import numpy as np

    if report.get("validated") is not True or report.get("precision") != precision:
        raise ValueError("Invalid precision or validation")
    if require_native and report.get("rfi_implementation") != "native":
        raise ValueError("Native RFI implementation required")
    for name in ("total_s", "writer_s"):
        value = report[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Invalid time")
    chunks = report["chunks"]
    if len(chunks) != 2 or any(
        isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in chunks
    ):
        raise ValueError("Invalid chunks")
    expected = {"vis_obs", "vis_ast", "vis_rfi", "noise_data"}
    if report["retained"] != ["vis_obs"]:
        expected |= {"vis_calibrated", "gains_ants"}
        if not expected <= set(report["retained"]):
            raise ValueError("Invalid retained schema")
    if set(report["samples"]) != expected:
        raise ValueError("Missing sample products")
    for value in report["samples"].values():
        for part in ("real", "imag"):
            values = np.asarray(value[part])
            if values.shape != (27,) or not np.isfinite(values).all():
                raise ValueError("Invalid sample values")
    case = report["case"]
    if case["visibility_precision"] != precision:
        raise ValueError("Inconsistent precision")
    for key in (
        "antennas",
        "times",
        "channels",
        "samples",
        "point_sources",
        "rfi_sources",
    ):
        if not isinstance(case[key], int) or case[key] <= 0:
            raise ValueError("Invalid case")
    for key in (
        "device_kind",
        "versions",
        "driver_sha256",
        "python",
        "host",
        "environment",
        "harness_sha256",
        "source_sha256",
    ):
        if key not in report["provenance"]:
            raise ValueError("Missing provenance")
    return report


def compare(left, right):
    validate_report(left, left["precision"])
    validate_report(right, right["precision"])
    import numpy as np

    for key in (
        "device_kind",
        "versions",
        "driver_sha256",
        "python",
        "host",
        "environment",
    ):
        if left["provenance"][key] != right["provenance"][key]:
            raise ValueError("Incomparable " + key)
    lc, rc = dict(left["case"]), dict(right["case"])
    lc.pop("visibility_precision")
    rc.pop("visibility_precision")
    if (
        lc != rc
        or left["chunks"] != right["chunks"]
        or left["retained"] != right["retained"]
    ):
        raise ValueError("Different workload, chunks or retained products")
    errors = {}
    for key in left["samples"]:

        def complex_values(row):
            value = row["samples"][key]
            return np.asarray(value["real"]) + 1j * np.asarray(value["imag"])

        x, y = complex_values(left), complex_values(right)
        if (
            x.shape != y.shape
            or not x.size
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("Invalid samples")
        # These deterministic fixtures have no near-nulls in sampled values.
        # Cancellation-specific accuracy is checked by scientific unit tests.
        np.testing.assert_allclose(
            y,
            x,
            rtol=3e-5 if "single" in (left["precision"], right["precision"]) else 1e-9,
            atol=2e-6 if "single" in (left["precision"], right["precision"]) else 1e-9,
        )
        errors[key] = float(np.max(np.abs(x - y)))
    return errors


def main():
    p = argparse.ArgumentParser()
    for name in ("base", "candidate", "output", "python"):
        p.add_argument("--" + name, required=True)
    p.add_argument(
        "--candidate-precision", choices=("single", "double"), default="single"
    )
    p.add_argument("--base-precision", choices=("single", "double"), default="double")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--times", type=int, default=16)
    p.add_argument(
        "--require-native", action="store_true", help="Require native RFI on candidate"
    )
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--case", choices=("rfi", "io", "capacity"), default="rfi")
    p.add_argument("--channels", type=int, default=544)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--rss-gib", type=float, default=8)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--candidate-only", action="store_true")
    a = p.parse_args()
    if (
        a.rounds < 1
        or a.channels < 1
        or a.samples < 1
        or a.samples % 2 != 1
        or a.times < 2
        or any(not math.isfinite(v) or v <= 0 for v in (a.rss_gib, a.timeout))
    ):
        p.error("Require positive rounds, channels, finite RSS and timeout")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for index in range(a.rounds):
        pair = {}
        targets = [
            ("parent", a.base, a.base_precision),
            ("candidate", a.candidate, a.candidate_precision),
        ]
        if a.candidate_only:
            targets = targets[1:]
        if index % 2:
            targets.reverse()
        for label, root, precision in targets:
            target = out / f"{index}-{label}"
            command = [
                a.python,
                str(Path(__file__).with_name("precision_worker.py")),
                "--root",
                root,
                "--output",
                str(target),
                "--device",
                a.device,
                "--precision",
                precision,
                "--case",
                a.case,
                "--channels",
                str(a.channels),
                "--samples",
                str(a.samples),
                "--times",
                str(a.times),
            ]
            if a.require_native and label == "candidate":
                command.append("--require-native")
            if a.cold:
                command.append("--cold")
            started = time.monotonic()
            peak = 0
            status = "ERROR"
            proc = None
            try:
                with (out / f"{index}-{label}.log").open("w") as log:
                    proc = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    while proc.poll() is None:
                        try:
                            process = psutil.Process(proc.pid)
                            peak = max(
                                peak,
                                sum(
                                    v.memory_info().rss
                                    for v in [process]
                                    + process.children(recursive=True)
                                ),
                            )
                        except psutil.NoSuchProcess:
                            pass
                        except (psutil.Error, OSError):
                            status = "MONITOR ERROR"
                            break
                        if peak > a.rss_gib * 2**30:
                            status = "RSS STOP"
                            break
                        if time.monotonic() - started > a.timeout:
                            status = "TIMEOUT"
                            break
                        time.sleep(0.2)
                    else:
                        status = "PASS" if proc.returncode == 0 else "ERROR"
                if status == "PASS":
                    report = json.loads((target / "result.json").read_text())
                    pair[label] = validate_report(
                        report,
                        precision,
                        require_native=a.require_native and label == "candidate",
                    )
            except Exception:
                status = "INVALID REPORT" if status == "PASS" else status
                raise
            except KeyboardInterrupt:
                status = "INTERRUPTED"
                raise
            finally:
                if proc is not None and proc.poll() is None:
                    stop_group(proc)
                rows.append(
                    dict(
                        round=index,
                        label=label,
                        status=status,
                        peak_rss=peak,
                        process_s=time.monotonic() - started,
                        report=pair.get(label),
                    )
                )
                (out / "summary.json").write_text(json.dumps(rows, indent=2))
            if status != "PASS":
                raise SystemExit(f"{label}: {status}")
        if len(pair) == 2:
            try:
                rows[-1]["comparison_max_abs_error"] = compare(
                    pair["parent"], pair["candidate"]
                )
            except Exception:
                rows[-1]["status"] = "COMPARISON FAILED"
                (out / "summary.json").write_text(json.dumps(rows, indent=2))
                raise
        (out / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
