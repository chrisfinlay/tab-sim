"""Isolated benchmark runner with guarded processes and paired checkout comparisons."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import psutil

from .cases import CASES, MODES, estimates


def terminate_worker(proc):
    """Reap our process group even after monitoring errors or Ctrl-C."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def run_one(args, case, mode, root, python, prefix):
    scratch = prefix.parent / (prefix.name + "-scratch")
    scratch.mkdir()
    try:
        return _run_one(args, case, mode, root, python, prefix, scratch)
    finally:
        shutil.rmtree(scratch)


def _run_one(args, case, mode, root, python, prefix, scratch):
    report, log, junit = (prefix.with_suffix(s) for s in (".json", ".txt", ".xml"))
    cmd = [python, "-m", "pytest", str(Path(__file__).parent / "test_benchmarks.py"),
           "-c", str(Path(__file__).resolve().parents[1] / "pytest.ini"),
           "--rootdir", str(Path(__file__).resolve().parents[1]),
           "--confcutdir", str(Path(__file__).parent.resolve()),
           "--case", case, "--mode", mode, "--device", args.device,
           f"--source-root={root}", "--rounds", str(args.rounds),
           "--workers", str(args.workers), "--chunk-mb", str(args.chunk_mb),
           "--host-budget-gib", str(args.host_budget_gib),
           "--gpu-budget-gib", str(args.gpu_budget_gib),
           "--benchmark-json", str(report),
           "--junitxml", str(junit), "-q"]
    cmd += ["--available-memory-fraction", str(getattr(args, "available_memory_fraction", 0.5))]
    cmd += ["--memory-model", getattr(args, "memory_model", "conservative")]
    if getattr(args, "capacity", False):
        cmd += ["--capacity"]
    if args.trace:
        cmd += ["--trace-dir", str(prefix.parent / (prefix.name + "-trace"))]
    started = time.perf_counter()
    status = None
    peak_rss = 0
    host_limit = min(args.host_budget_gib * 2**30, psutil.virtual_memory().available * getattr(args, "available_memory_fraction", 0.5),
                     max(0, psutil.virtual_memory().available - 2 * 2**30))
    cmd += ["--basetemp", str(scratch)]
    disk_reserve = max(2 * 2**30, shutil.disk_usage(scratch).total * 0.05)
    with log.open("w") as stream:
        proc = subprocess.Popen(cmd, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            while proc.poll() is None:
                try:
                    process = psutil.Process(proc.pid)
                    rss = sum(p.memory_info().rss for p in [process] + process.children(recursive=True))
                    peak_rss = max(peak_rss, rss)
                    if rss > host_limit:
                        status = "host_memory_limit"
                    elif shutil.disk_usage(scratch).free < disk_reserve:
                        status = "disk_space_limit"
                    elif time.perf_counter() - started > args.timeout:
                        status = "timeout"
                except psutil.NoSuchProcess:
                    pass
                except (psutil.Error, OSError):
                    # Fail closed rather than continue an unmonitored allocation.
                    status = "monitor_error"
                if status:
                    break
                time.sleep(0.1)
        finally:
            terminate_worker(proc)
        code = proc.wait()
    result = {"case": case, "mode": mode, "root": str(root), "python": python,
              "status": status or ("passed" if code == 0 else "failed"),
              "supervisor_peak_rss_bytes": peak_rss, "enforced_host_limit_bytes": host_limit,
              "disk_reserve_bytes": disk_reserve,
              "returncode": code, "process_wall_s": time.perf_counter() - started,
              "report": report.name, "log": log.name, "command": cmd}
    if junit.exists():
        skipped = ET.parse(junit).findall(".//skipped")
        if skipped and code == 0 and status is None:
            result.update(status="skipped", reason=skipped[0].get("message"))
    if report.exists():
        content = json.loads(report.read_text())
        benches = content.get("benchmarks", [])
        if benches:
            bench = benches[0]
            result.update(stats=bench["stats"], extra_info=bench["extra_info"])
        elif result["status"] == "passed":
            result.update(status="failed", reason="No benchmark statistics produced")
    if result["status"] == "passed" and "stats" not in result:
        result.update(status="failed", reason="No benchmark statistics produced")
    return result


def compare_records(base, candidate):
    """Never call unlike devices/configs comparable or treat a failed run as a win."""
    if base["status"] != "passed" or candidate["status"] != "passed":
        return {"status": "unavailable"}
    a, b = base["extra_info"], candidate["extra_info"]
    if a["options"].get("capacity") or b["options"].get("capacity"):
        raise ValueError("Capacity runs are not repeated performance comparisons")
    for key in ("fixture_sha256", "harness_sha256", "case", "options", "host", "device_kind", "x64", "environment", "python"):
        if a[key] != b[key]:
            raise ValueError(f"Incomparable benchmark metadata: {key}")
    # tabsim's version can change with the implementation; dependency versions cannot.
    va = {k: v for k, v in a["versions"].items() if k != "tabsim"}
    vb = {k: v for k, v in b["versions"].items() if k != "tabsim"}
    if va != vb:
        raise ValueError("Dependency versions differ; rerun in matched environments")
    from .harness import check_samples
    check_samples(b["output_sample"], a["output_sample"])
    old, new = base["stats"], candidate["stats"]
    return {"status": "compared", "sampled_outputs_match": True,
            "median_change_percent": 100 * (new["median"] / old["median"] - 1),
            "baseline_iqr_s": old["iqr"], "candidate_iqr_s": new["iqr"],
            "note": "Per-pair result, not a significance test; inspect all alternating pairs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=["aa05-point", "aa1-mixed", "aa2-rfi"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["zarr", "rfi-kernel"])
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--candidate-python")
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-mb", type=float, default=16)
    parser.add_argument("--host-budget-gib", type=float, default=4)
    parser.add_argument("--available-memory-fraction", type=float, default=0.5)
    parser.add_argument("--gpu-budget-gib", type=float, default=4)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--capacity", action="store_true", help="One cold full-output execution; no performance comparison")
    parser.add_argument("--memory-model", choices=("conservative", "chunked"), default="conservative",
                        help="Chunked is calibrated for current lazy-noise Zarr only; use conservative for older checkouts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if not 0 < args.available_memory_fraction <= 0.8:
        parser.error("Available-memory fraction must be in (0, 0.8]")
    if args.pairs < 1 or args.rounds < 5:
        parser.error("Require pairs >= 1 and rounds >= 5")
    for name in ("workers", "chunk_mb", "host_budget_gib", "gpu_budget_gib", "timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.capacity and (args.candidate_root or any(m != "zarr" for m in args.modes)):
        parser.error("Capacity runs support Zarr only and cannot compare checkouts")
    if args.candidate_root and args.pairs < 5:
        parser.error("Checkout comparisons require at least five alternating pairs")
    args.source_root = args.source_root.resolve()
    if args.candidate_root:
        args.candidate_root = args.candidate_root.resolve()
    if args.plan:
        print(json.dumps({case: {mode: estimates(CASES[case], mode, args.chunk_mb, args.workers, args.memory_model) for mode in args.modes}
                          for case in args.cases}, indent=2))
        return
    # Exclusive directory prevents accidentally overwriting prior evidence.
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, comparisons = [], []
    for case in args.cases:
        for mode in args.modes:
            for pair in range(args.pairs):
                targets = [("baseline", args.source_root, args.python)]
                if args.candidate_root:
                    targets.append(("candidate", args.candidate_root, args.candidate_python or args.python))
                    if pair % 2:
                        targets.reverse()
                paired = {}
                for label, root, python in targets:
                    prefix = args.output / f"{case}-{mode}-{pair}-{label}"
                    row = run_one(args, case, mode, root, python, prefix)
                    row.update(label=label, pair=pair)
                    rows.append(row)
                    paired[label] = row
                    print(f"{case} {mode} {label}: {row['status']}", flush=True)
                    (args.output / "summary.json").write_text(json.dumps({"schema_version": 1,
                        "runs": rows, "comparisons": comparisons}, indent=2, allow_nan=False))
                if args.candidate_root:
                    try:
                        comparison = compare_records(paired["baseline"], paired["candidate"])
                    except (ValueError, AssertionError) as exc:
                        comparison = {"status": "failed", "reason": str(exc)}
                    comparisons.append(dict(case=case, mode=mode, pair=pair, **comparison))
    (args.output / "summary.json").write_text(json.dumps({"schema_version": 1,
        "runs": rows, "comparisons": comparisons}, indent=2, allow_nan=False))
    if any(r["status"] not in ("passed", "skipped") for r in rows) or any(c["status"] == "failed" for c in comparisons):
        sys.exit(1)


if __name__ == "__main__":
    main()
