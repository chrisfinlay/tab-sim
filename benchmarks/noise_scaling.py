"""Issue #50 bounded-memory RNG microbenchmark (CPU, independent fresh processes).

The measured operation constructs then consumes noise through scalar reductions,
never gathering a full candidate cube. Compare 60, 240 and 960 MiB logical cubes.
The eager parent's cube allocation is deliberately included in construction time.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def chunk_views(array, chunks):
    """Wrap an eager result without Dask's ndarray copying/hashing adapter.

    Each graph value is a NumPy slice sharing the original allocation. This is a
    local threaded benchmark; distributing these views is intentionally unsupported.
    """
    from itertools import product
    from uuid import uuid4
    from dask.array.core import Array, normalize_chunks, slices_from_chunks
    normalized = normalize_chunks(chunks, shape=array.shape, dtype=array.dtype)
    name = "noise-views-" + uuid4().hex
    keys = product(*[range(len(c)) for c in normalized])
    graph = {(name, *index): array[selection] for index, selection in
             zip(keys, slices_from_chunks(normalized))}
    return Array(graph, name, normalized, dtype=array.dtype)


def worker(root, length):
    # Bind the selected implementation before any pytest/editable path can win.
    sys.path.insert(0, str(root))
    import tabsim
    assert Path(tabsim.__file__).resolve().is_relative_to(root)
    import dask
    import dask.array as da
    import numpy as np
    import psutil
    import threading
    from tabsim.dask.interferometry import add_noise
    shape, chunks = (length, 120, 16), (256, 120, 16)
    cube_bytes = np.prod(shape).item() * 16
    # Eager parent has several simultaneous cube-sized intermediates.
    if cube_bytes > 2**30 or psutil.virtual_memory().available < 4 * cube_bytes + 2**30:
        return {"status": "skipped", "reason": "Insufficient host memory for conservative eager-parent plan"}
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(.005):
            peak[0] = max(peak[0], process.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        vis = da.zeros(shape, chunks=chunks, dtype=complex)
        scale = np.array([1. + i / 16 for i in range(16)])
        before = process.memory_info().rss
        start = time.perf_counter()
        _, noise = add_noise(vis, scale, 20260919)
        construction_s = time.perf_counter() - start
        graph_rss = process.memory_info().rss
        # Legacy returns ndarray. Use chunk views: da.asarray/from_array copy it.
        start = time.perf_counter()
        noise = noise if isinstance(noise, da.Array) else chunk_views(noise, vis.chunks)
        adapter_s = time.perf_counter() - start
        start = time.perf_counter()
        statistics = dask.compute(noise.real.mean(axis=(0, 1)), noise.imag.mean(axis=(0, 1)),
            (noise.real**2).mean(axis=(0, 1)), (noise.imag**2).mean(axis=(0, 1)),
            scheduler="threads", num_workers=1)
        consume_s = time.perf_counter() - start
        for mean in statistics[:2]:
            np.testing.assert_allclose(mean, 0, atol=.02)
        for second_moment in statistics[2:]:
            np.testing.assert_allclose(second_moment, scale**2, rtol=.02)
    finally:
        peak[0] = max(peak[0], process.memory_info().rss)
        stop.set()
        thread.join()
    return {"status": "passed", "shape": shape, "chunks": chunks, "cube_bytes": cube_bytes,
        "source_root": str(root), "revision": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "versions": {"python": sys.version, "numpy": np.__version__, "dask": dask.__version__},
        "workers": 1, "rss_before_bytes": before, "rss_after_construction_bytes": graph_rss,
        "rss_peak_sampled_bytes": peak[0], "sample_interval_s": .005,
        "construction_s": construction_s, "adapter_s": adapter_s, "consume_s": consume_s,
        "total_s": construction_s + adapter_s + consume_s, "channel_means_and_second_moments": [x.tolist() for x in statistics]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-length", type=int, choices=(2048, 8192, 32768))
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    if args.worker_length:
        result = worker(args.source_root, args.worker_length)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        return
    if args.candidate_root is None:
        parser.error("--candidate-root is required")
    args.candidate_root = args.candidate_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for length in (2048, 8192, 32768):
        for pair in range(5):
            targets = [("baseline", args.source_root), ("candidate", args.candidate_root)]
            if pair % 2:
                targets.reverse()
            for label, root in targets:
                path = args.output / f"{length}-{pair}-{label}.json"
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--source-root", str(root), "--worker-length", str(length), "--output", str(path)],
                    timeout=180, capture_output=True, text=True)
                row = json.loads(path.read_text()) if path.exists() else {
                    "status": "failed", "returncode": result.returncode, "stderr": result.stderr[-4000:]}
                if result.returncode:
                    row["status"] = "failed"
                row.update(label=label, pair=pair, length=length)
                records.append(row)
                print(f"{length} pair {pair} {label}: {row['status']}", flush=True)
                (args.output / "summary.json").write_text(json.dumps(records, indent=2, allow_nan=False))
    if any(row["status"] != "passed" for row in records):
        sys.exit(1)


if __name__ == "__main__":
    main()
