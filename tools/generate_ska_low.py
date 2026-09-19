"""Reproduce the planned SKA-Low tables from pinned inputs (standard library only).

See tabsim/data/telescopes/SKA-Low-PROVENANCE.md for input retrieval.
No downloaded Python code is executed.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


SOURCE_HASHES = {
    "layout": "2aee3f1204be139335e627cead6c4f9025994b41f2378ff49a193714d8cb70ae",
    "assembly": "611da260bb65dfe1052b1eb299f831e49cf74d11a8912748e83128e1a9f88b47",
}
STAGES = {
    "AA0.5": ("LOW_AA05", 4),
    "AA1": ("LOW_AA1", 16),
    "AA2": ("LOW_AA2", 68),
    "AAstar-Phase-1": ("LOW_AAstar_Phase_1", 108),
    "AAstar": ("LOW_AAstar", 307),
    "AA4": (None, 512),
}


def read_verified(path, kind):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_HASHES[kind]:
        raise ValueError(f"{path}: not the pinned {kind} source")
    return data.decode("utf-8")


def literal_string(node):
    """Only accept literal strings concatenated with +, never execute Python."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal_string(node.left) + literal_string(node.right)
    raise ValueError("Expected a literal station-list string")


def render_tables(layout, assembly):
    receptors = json.loads(layout)["receptors"]
    positions = {
        station["station_label"].upper(): station["location"]["geocentric"]
        for station in receptors
    }
    if len(positions) != len(receptors):
        raise ValueError("Duplicate station labels")
    wanted = {symbol for symbol, _ in STAGES.values() if symbol}
    selections = {}
    for node in ast.parse(assembly).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                labels = literal_string(node.value).split(",")
                selections[target.id] = [
                    "C" + label.strip() if label.strip()[0].isdigit() else label.strip()
                    for label in labels
                ]
    tables = {}
    for stage, (symbol, count) in STAGES.items():
        # LowSubArray uses numpy.unique: preserve its lexicographic row order.
        labels = sorted(set(selections[symbol] if symbol else positions.keys() - {"ARRAY-CENTRE"}))
        if len(labels) != count:
            raise ValueError(f"{stage}: expected {count} stations, got {len(labels)}")
        tables[f"SKA-Low-{stage}.itrf.txt"] = "".join(
            " ".join(f"{float(positions[label][axis]):.3f}" for axis in "xyz") + "\n"
            for label in labels
        )
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--array-assembly", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1]
                        / "tabsim/data/telescopes")
    parser.add_argument("--check", action="store_true", help="Compare without writing")
    args = parser.parse_args()
    tables = render_tables(read_verified(args.layout, "layout"),
                           read_verified(args.array_assembly, "assembly"))
    if args.check:
        mismatches = [name for name, body in tables.items()
                      if not (args.output_dir / name).is_file()
                      or (args.output_dir / name).read_bytes() != body.encode("utf-8")]
        if mismatches:
            parser.error("Tables differ: " + ", ".join(mismatches))
        print("All six tables match the pinned sources byte for byte.")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, body in tables.items():
            (args.output_dir / name).write_bytes(body.encode("utf-8"))


if __name__ == "__main__":
    main()
