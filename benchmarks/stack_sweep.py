"""Alternate isolated warm parent/candidate measurements under RSS supervision."""
import argparse
import json
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


def main():
    parser = argparse.ArgumentParser()
    for name in ('base', 'candidate', 'output', 'python'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--rss-gib', type=float, default=8)
    parser.add_argument('--continue-on-failure', action='store_true')
    args = parser.parse_args()
    if args.rounds < 1 or args.rss_gib <= 0:
        parser.error('Require positive rounds and RSS limit')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for index in range(args.rounds):
        for label, root in [('parent', args.base), ('candidate', args.candidate)]:
            target = out / f'{index}-{label}'
            command = [args.python, str(Path(__file__).with_name('stack_worker.py')),
                       '--root', root, '--output', str(target), '--device', args.device]
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
                        except psutil.Error:
                            pass
                        if peak > args.rss_gib * 2**30 or time.monotonic() - start > 600:
                            status = 'RSS STOP' if peak > args.rss_gib * 2**30 else 'TIMEOUT'
                            stop_group(process)
                            break
                        time.sleep(.2)
                    else:
                        status = 'PASS' if process.returncode == 0 else 'ERROR'
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


if __name__ == '__main__':
    main()
