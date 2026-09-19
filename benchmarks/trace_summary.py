"""Summarize observed CUPTI memcpy events in an optional JAX Chrome trace."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re


COPY_NAMES = ("MemcpyH2D", "MemcpyD2H", "MemcpyD2D")


def summarize(path):
    raw = path.read_bytes()
    trace = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    events = trace.get("traceEvents", [])
    copies = {name: {"events": 0, "duration_us": 0., "bytes_with_known_size": 0,
                     "events_with_known_size": 0} for name in COPY_NAMES}
    for event in events:
        if event.get("ph") != "X" or event.get("name") not in copies:
            continue
        record = copies[event["name"]]
        record["events"] += 1
        record["duration_us"] += event.get("dur", 0.)
        details = event.get("args", {}).get("memcpy_details", "")
        size = re.search(r"\bsize:(\d+)\b", details)
        if size:
            record["events_with_known_size"] += 1
            record["bytes_with_known_size"] += int(size.group(1))
    return {"trace_sha256": hashlib.sha256(raw).hexdigest(), "trace_events": len(events),
            "copies": copies, "million_event_warning": len(events) >= 1_000_000,
            "coverage": "Observed events only; profiler may drop/truncate events. Zero means none captured, not none executed. Durations can overlap."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    body = json.dumps(summarize(args.trace), indent=2) + "\n"
    if args.output:
        args.output.write_text(body)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
