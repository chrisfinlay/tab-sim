"""Issue #50: paired Zarr runs allowing only the documented noise-output changes.

Uses the standard harness unchanged on both checkouts. Only deterministic signal
products must match across implementations; noise distribution is validated by
statistical tests, while each run still validates its own cold/warm repeatability.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

from .run import run_one, compare_records


def compare_noise_records(base, candidate):
    if base["status"] != "passed" or candidate["status"] != "passed":
        return {"status": "unavailable"}
    records = copy.deepcopy([base, candidate])
    unchanged = {"vis_ast", "vis_rfi"}
    changed = {"vis_obs", "vis_calibrated", "flags"}
    for row in records:
        sample = row["extra_info"]["output_sample"]
        if sample.keys() != unchanged | changed:
            raise ValueError("Unexpected output products in noise comparison")
        row["extra_info"]["output_sample"] = {k: sample[k] for k in unchanged}
    for name in changed:
        if base["extra_info"]["output_sample"][name]["shape"] != candidate["extra_info"]["output_sample"][name]["shape"]:
            raise ValueError(f"Output shape changed: {name}")
    result = compare_records(*records)
    result.pop("sampled_outputs_match")
    result.update(sampled_signals_match=True, intentionally_changed_products=sorted(changed),
        correctness_note="Noise equation/RNG change: statistical tests replace parent noise equality; no full-output equivalence claim")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.rounds, args.workers, args.chunk_mb = 5, 1, 16
    args.host_budget_gib, args.gpu_budget_gib, args.timeout, args.trace = 4, 4, 1200, False
    args.source_root = args.source_root.resolve()
    args.candidate_root = args.candidate_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, comparisons = [], []
    for case in ("aa1-noise-512", "aa1-noise-2048", "aa1-noise-8192", "aa1-long"):
        for pair in range(5):
            targets = [("baseline", args.source_root), ("candidate", args.candidate_root)]
            if pair % 2:
                targets.reverse()
            paired = {}
            for label, root in targets:
                row = run_one(args, case, "zarr", root, sys.executable,
                              args.output / f"{case}-{pair}-{label}")
                row.update(label=label, pair=pair)
                rows.append(row)
                paired[label] = row
                print(f"{case} pair {pair} {label}: {row['status']}", flush=True)
            try:
                result = compare_noise_records(paired["baseline"], paired["candidate"])
            except (ValueError, AssertionError) as exc:
                result = {"status": "failed", "reason": str(exc)}
            comparisons.append(dict(case=case, pair=pair, **result))
            (args.output / "summary.json").write_text(json.dumps(
                {"runs": rows, "comparisons": comparisons}, indent=2, allow_nan=False))
    if any(row["status"] != "passed" for row in rows) or any(c["status"] != "compared" for c in comparisons):
        sys.exit(1)


if __name__ == "__main__":
    main()
