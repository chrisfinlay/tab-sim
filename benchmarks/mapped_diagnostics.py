"""Untimed per-callback CPU profiling for issue #51's removed inner graphs."""
import cProfile
from functools import wraps
import threading
import time


# Python 3.13 cProfile uses a process-wide monitoring tool id. Never wait for
# ownership: nested callbacks must run, and diagnostics must not serialize work.
_profile_owner = threading.Lock()


class MappedDiagnostics:
    """Instrument local threaded/synchronous map callbacks, never timed rounds.

    Inclusive compute time includes kernel dispatch/execution/waiting; it is NOT
    pure scheduler overhead. Profiler CPU/wall observations are diagnostic, not
    completed GPU kernel measurements. Nested/contending callbacks keep wall/CPU
    timing but skip cProfile; report counters expose incomplete profile coverage.
    """
    def __enter__(self):
        import dask.base
        import dask.tokenize
        import xarray as xr
        self.compute_code = dask.base.compute.__code__
        self.tokenize_code = dask.tokenize.tokenize.__code__
        self.lock = threading.Lock()
        self.records = []
        self.original = xr.map_blocks

        def instrument(func, *args, **kwargs):
            @wraps(func)
            def callback(*block_args, **block_kwargs):
                profiler = cProfile.Profile()
                owned = _profile_owner.acquire(blocking=False)
                enabled = False
                wall, cpu = time.perf_counter(), time.thread_time()
                try:
                    if owned:
                        try:
                            profiler.enable()
                            enabled = True
                        except ValueError as error:
                            # Another tool may own CPython's monitoring slot.
                            if 'tool' not in str(error) or 'already in use' not in str(error):
                                raise
                    return func(*block_args, **block_kwargs)
                finally:
                    try:
                        if enabled:
                            profiler.disable()
                    finally:
                        if owned:
                            _profile_owner.release()
                    record = {"callback": f"{func.__module__}.{func.__qualname__}", "wall_s": time.perf_counter() - wall,
                              "thread_cpu_s": time.thread_time() - cpu,
                              "profiled_calls": int(enabled), "unprofiled_calls": int(not enabled),
                              "nested_compute_calls": 0, "nested_compute_inclusive_s": 0.,
                              "tokenize_calls": 0, "tokenize_inclusive_s": 0.}
                    for entry in profiler.getstats() if enabled else ():
                        if entry.code is self.compute_code:
                            record["nested_compute_calls"] += entry.callcount
                            record["nested_compute_inclusive_s"] += entry.totaltime
                        elif entry.code is self.tokenize_code:
                            record["tokenize_calls"] += entry.callcount
                            record["tokenize_inclusive_s"] += entry.totaltime
                    with self.lock:
                        self.records.append(record)
            return self.original(callback, *args, **kwargs)

        xr.map_blocks = instrument
        return self

    def __exit__(self, *exc):
        import xarray as xr
        xr.map_blocks = self.original

    def report(self):
        totals = {}
        for record in self.records:
            name = record["callback"]
            values = totals.setdefault(name, {"calls": 0})
            values["calls"] += 1
            for key, value in record.items():
                if key != "callback":
                    values[key] = values.get(key, 0) + value
        return {"callbacks": totals,
                "scope": "separate instrumented local map callback round; inclusive compute includes kernel/waiting, not pure scheduler time; GPU completion is not fenced per callback; cProfile covers profiled_calls only, nested/contending/unavailable callbacks are counted as unprofiled_calls; wall/thread CPU cover all callbacks"}
