"""Alternating setup-only comparisons; retained samples must match."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def compare_reports(a, b):
    import numpy as np
    for key in ('python', 'host', 'device_kind', 'versions', 'harness_sha256', 'environment'):
        if a['provenance'][key] != b['provenance'][key]:
            raise ValueError('Incomparable provenance: ' + key)
    if a['times'] != b['times']:
        raise ValueError('Different observation sizes')
    for row in (a, b):
        if not math.isfinite(row['setup_s']) or row['setup_s'] <= 0:
            raise ValueError('Invalid duration')
        if not row['geometry_sample'] or not np.isfinite(row['geometry_sample']).all():
            raise ValueError('Invalid geometry sample')
    np.testing.assert_allclose(a['geometry_sample'], b['geometry_sample'], rtol=1e-9, atol=1e-9)


def main():
    p=argparse.ArgumentParser()
    for key in ('base','candidate','output'):
        p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for times,rounds in ((256,1),(2048,5),(16384,1)):
        for i in range(rounds):
            targets=[('parent',a.base),('candidate',a.candidate)]
            if i % 2:
                targets.reverse()
            paired={}
            for label,root in targets:
                path=out/f'{times}-{i}-{label}.json'
                subprocess.run([sys.executable,str(Path(__file__).with_name('geometry_setup_worker.py')),
                    '--root',root,'--times',str(times),'--output',str(path)],check=True,timeout=300)
                paired[label]=json.loads(path.read_text())
            compare_reports(paired['parent'],paired['candidate'])


if __name__ == '__main__':
    main()
