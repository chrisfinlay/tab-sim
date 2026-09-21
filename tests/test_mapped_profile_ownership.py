"""Partial cProfile coverage must never block or break mapped execution."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import pytest
import xarray as xr
from benchmarks.mapped_diagnostics import MappedDiagnostics


def test_nested_and_concurrent_callbacks_keep_running(monkeypatch):
    monkeypatch.setattr(xr,'map_blocks',lambda func,*a,**k:func)
    entered,release=Event(),Event()
    with MappedDiagnostics() as diagnostic:
        inner=xr.map_blocks(lambda: 7)
        def outer():
            assert inner()==7
            entered.set()
            assert release.wait(5)
            return 8
        wrapped=xr.map_blocks(outer)
        with ThreadPoolExecutor(2) as pool:
            first=pool.submit(wrapped)
            assert entered.wait(5)
            try:
                assert pool.submit(inner).result(timeout=2)==7
            finally:
                release.set()
            assert first.result()==8
    totals=diagnostic.report()['callbacks'].values()
    assert sum(x['calls'] for x in totals)==3
    assert sum(x['profiled_calls'] for x in totals)==1
    assert sum(x['unprofiled_calls'] for x in totals)==2
    assert all(x['wall_s']>=0 and x['thread_cpu_s']>=0 for x in totals)


def test_external_profiler_conflict_preserves_callback_and_reports_coverage(monkeypatch):
    monkeypatch.setattr(xr,'map_blocks',lambda func,*a,**k:func)
    class Busy:
        def enable(self):raise ValueError('tool 2 is already in use')
        def disable(self):raise AssertionError('Must not disable another profiler')
        def getstats(self):raise AssertionError('Unavailable profile must not count as complete')
    monkeypatch.setattr('benchmarks.mapped_diagnostics.cProfile.Profile',Busy)
    with MappedDiagnostics() as diagnostic:
        assert xr.map_blocks(lambda: 9)()==9
    record=next(iter(diagnostic.report()['callbacks'].values()))
    assert record['profiled_calls']==0 and record['unprofiled_calls']==1
