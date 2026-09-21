"""Fail-closed reproduction-driver contracts; no performance runs."""
import copy
import json
import sys
import pytest
pytest.importorskip('psutil')
from benchmarks import execution_sweep as driver


def report():
    return dict(total_s=1.,chunks=[2,3],case={},sample=dict(real=[1.],imag=[0.]),provenance=dict(
        python='3',host='host',device_kind='CPU',versions={},harness_sha256='same',environment={}))


def test_pair_rejects_changed_outputs_and_provenance(tmp_path):
    a=report();b=copy.deepcopy(a)
    driver.compare_reports(a,b)
    b['sample']['real']=[2.]
    with pytest.raises(AssertionError):driver.compare_reports(a,b)
    b=copy.deepcopy(a);b['provenance']['python']='different'
    with pytest.raises(ValueError):driver.compare_reports(a,b)
    a['total_s']=float('nan');path=tmp_path/'result.json';path.write_text(json.dumps(a))
    with pytest.raises(ValueError):driver.validate_report(path)


@pytest.mark.parametrize('monitor_error',[False,True])
def test_missing_report_and_monitor_failure_are_nonzero_even_when_continuing(tmp_path,monkeypatch,monitor_error):
    processes=[]
    class Process:
        pid=999999
        returncode=None if monitor_error else 0
        def poll(self):return self.returncode
    def spawn(*args,**kwargs):
        p=Process();processes.append(p);return p
    monkeypatch.setattr(driver.subprocess,'Popen',spawn)
    monkeypatch.setattr(driver,'stop_group',lambda p:setattr(p,'returncode',-15))
    def denied(pid):raise driver.psutil.AccessDenied(pid)
    monkeypatch.setattr(driver.psutil,'Process',denied)
    monkeypatch.setattr(sys,'argv',['sweep','--base','base','--candidate','candidate','--python',sys.executable,
        '--output',str(tmp_path/'run'),'--rounds','2','--continue-on-failure'])
    with pytest.raises(SystemExit) as error:driver.main()
    assert error.value.code==1
    rows=json.loads((tmp_path/'run/summary.json').read_text())
    assert len(rows)==4
    assert [r['label'] for r in rows]==['parent','candidate','candidate','parent']
    assert all(row['status']==('MONITOR ERROR' if monitor_error else 'INVALID REPORT') for row in rows)
    assert all(p.poll() is not None for p in processes)


def test_chunk_layout_required_and_matched(tmp_path):
    a=report();b=copy.deepcopy(a);b['chunks']=[1,3]
    with pytest.raises(ValueError,match='chunk'):driver.compare_reports(a,b)
    path=tmp_path/'result.json'
    for chunks in (None,[],[0,3],[True,3]):
        row=report()
        if chunks is None:del row['chunks']
        else:row['chunks']=chunks
        path.write_text(json.dumps(row))
        with pytest.raises((ValueError,KeyError)):driver.validate_report(path)
