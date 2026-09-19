"""Isolated benchmark runner with guarded processes and paired checkout comparisons."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import psutil

from .cases import CASES, MODES, estimates


def run_one(args, case, mode, root, python, prefix):
    report, log, junit = (prefix.with_suffix(s) for s in (".json", ".txt", ".xml"))
    cmd = [python, "-m", "pytest", str(Path(__file__).parent / "test_benchmarks.py"),
           "-c", str(Path(__file__).resolve().parents[1] / "pytest.ini"),
           "--rootdir", str(Path(__file__).resolve().parents[1]),
           "--confcutdir", str(Path(__file__).parent.resolve()),
           "--case", case, "--mode", mode, "--device", args.device,
           "--source-root", str(root), "--rounds", str(args.rounds),
           "--workers", str(args.workers), "--chunk-mb", str(args.chunk_mb),
           "--host-budget-gib", str(args.host_budget_gib),
           "--gpu-budget-gib", str(args.gpu_budget_gib),
           "--benchmark-json", str(report),
           "--junitxml", str(junit), "-q"]
    if args.trace:
        cmd += ["--trace-dir", str(prefix.parent / (prefix.name + "-trace"))]
    started = time.perf_counter()
    status = None
    with log.open("w") as stream:
        proc = subprocess.Popen(cmd, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        while proc.poll() is None:
            try:
                process = psutil.Process(proc.pid)
                rss = sum(p.memory_info().rss for p in [process] + process.children(recursive=True))
                if rss > args.host_budget_gib * 2**30:
                    status = "host_memory_limit"
                elif time.perf_counter() - started > args.timeout:
                    status = "timeout"
                if status:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.1)
        code = proc.wait()
    result = {"case": case, "mode": mode, "root": str(root), "python": python,
              "status": status or ("passed" if code == 0 else "failed"),
              "returncode": code, "process_wall_s": time.perf_counter() - started,
              "report": report.name, "log": log.name, "command": cmd}
    if junit.exists():
        skipped = ET.parse(junit).findall(".//skipped")
        if skipped and code == 0:
            result.update(status="skipped", reason=skipped[0].get("message"))
    if report.exists():
        content = json.loads(report.read_text())
        benches = content.get("benchmarks", [])
        if benches:
            bench = benches[0]
            result.update(stats=bench["stats"], extra_info=bench["extra_info"])
        elif result["status"] == "passed":
            result.update(status="failed", reason="No benchmark statistics produced")
    return result


def compare_records(base, candidate):
    """Never call unlike devices/configs comparable or treat a failed run as a win."""
    if base["status"] != "passed" or candidate["status"] != "passed":
        return {"status": "unavailable"}
    a, b = base["extra_info"], candidate["extra_info"]
    for key in ("fixture_sha256", "harness_sha256", "case", "options", "host", "device_kind", "x64", "environment"):
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
    parser.add_argument("--gpu-budget-gib", type=float, default=4)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.pairs < 1 or args.rounds < 5:
        parser.error("Require pairs >= 1 and rounds >= 5")
    if args.candidate_root and args.pairs < 5:
        parser.error("Checkout comparisons require at least five alternating pairs")
    args.source_root = args.source_root.resolve()
    if args.candidate_root:
        args.candidate_root = args.candidate_root.resolve()
    if args.plan:
        print(json.dumps({case: {mode: estimates(CASES[case], mode) for mode in args.modes}
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
