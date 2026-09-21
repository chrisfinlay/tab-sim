"""Alternate isolated warm parent/candidate measurements under RSS supervision."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil


def stop_group(process):
    """Reap our worker even if CUDA does not respond to SIGTERM."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=20)



def validate_report(path):
    report = json.loads(path.read_text())
    value = report['total_s']
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid duration')
    chunks = report['chunks']
    if (not isinstance(chunks, list) or len(chunks) != 2 or
            any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in chunks)):
        raise ValueError('Invalid time/frequency chunks')
    for field in ('python', 'host', 'device_kind', 'versions', 'harness_sha256', 'environment'):
        if field not in report['provenance']:
            raise ValueError('Missing provenance: ' + field)
    return report


def compare_reports(parent, candidate):
    import numpy as np
    for field in ('python', 'host', 'device_kind', 'versions', 'harness_sha256', 'environment'):
        if parent['provenance'][field] != candidate['provenance'][field]:
            raise ValueError('Incomparable provenance: ' + field)
    if parent['case'] != candidate['case']:
        raise ValueError('Different cases')
    if parent['chunks'] != candidate['chunks']:
        raise ValueError('Different chunk layouts')
    if parent.get('profile') == candidate.get('profile') and parent.get('retained') != candidate.get('retained'):
        raise ValueError('Different retained schemas for identical profiles')
    samples_a = parent.get('samples', {'vis_obs': parent.get('sample')})
    samples_b = candidate.get('samples', {'vis_obs': candidate.get('sample')})
    if samples_a.keys() != samples_b.keys():
        raise ValueError('Different sample products')
    for product in samples_a:
        for part in ('real', 'imag'):
            a, b = np.asarray(samples_a[product][part]), np.asarray(samples_b[product][part])
            if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Invalid output samples')
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def main():
    parser = argparse.ArgumentParser()
    for name in ('base', 'candidate', 'output', 'python'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--base-slots',type=int,default=1)
    parser.add_argument('--candidate-slots',type=int,default=1)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--rss-gib', type=float, default=8)
    parser.add_argument('--continue-on-failure', action='store_true')
    args = parser.parse_args()
    if args.rounds < 1 or not math.isfinite(args.rss_gib) or args.rss_gib <= 0:
        parser.error('Require positive rounds and RSS limit')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for index in range(args.rounds):
        targets = [('parent', args.base), ('candidate', args.candidate)]
        if index % 2:
            targets.reverse()
        paired = {}
        for label, root in targets:
            target = out / f'{index}-{label}'
            command = [args.python, str(Path(__file__).with_name('execution_worker.py')),
                       '--root', root, '--output', str(target), '--device', args.device,
                       '--gpu-slots',str(args.base_slots if label=='parent' else args.candidate_slots)]
            process = None
            start = time.monotonic()
            status, peak = 'ERROR', 0
            try:
                with (out / f'{index}-{label}.log').open('w') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    while process.poll() is None:
                        try:
                            proc = psutil.Process(process.pid)
                            rss = proc.memory_info().rss + sum(
                                child.memory_info().rss for child in proc.children(recursive=True))
                            peak = max(peak, rss)
                        except psutil.NoSuchProcess:
                            pass
                        except (psutil.Error, OSError):
                            status = 'MONITOR ERROR'
                            stop_group(process)
                            break
                        if peak > args.rss_gib * 2**30 or time.monotonic() - start > 600:
                            status = 'RSS STOP' if peak > args.rss_gib * 2**30 else 'TIMEOUT'
                            stop_group(process)
                            break
                        time.sleep(.2)
                    else:
                        status = 'PASS' if process.returncode == 0 else 'ERROR'
                        if status == 'PASS':
                            try:
                                paired[label] = validate_report(target / 'result.json')
                            except (OSError, ValueError, KeyError, TypeError):
                                status = 'INVALID REPORT'
            except KeyboardInterrupt:
                status = 'INTERRUPTED'
                raise
            finally:
                try:
                    if process is not None and process.poll() is None:
                        stop_group(process)
                finally:
                    rows.append(dict(round=index, label=label, status=status,
                                     supervisor_peak_rss=peak, process_s=time.monotonic()-start))
                    (out / 'summary.json').write_text(json.dumps(rows, indent=2))
            if status != 'PASS' and not args.continue_on_failure:
                raise SystemExit(f'{label}: {status}')


        if len(paired) == 2:
            try:
                compare_reports(paired['parent'], paired['candidate'])
            except (ValueError, KeyError, TypeError, AssertionError) as error:
                rows.append(dict(round=index, label='comparison', status='MISMATCH', reason=str(error)))
                (out / 'summary.json').write_text(json.dumps(rows, indent=2))
                if not args.continue_on_failure:
                    raise SystemExit('Comparison failed: ' + str(error))
    if any(row['status'] != 'PASS' for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
