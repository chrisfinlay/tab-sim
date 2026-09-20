import argparse,subprocess,time,json,os,signal
from pathlib import Path
import psutil
p=argparse.ArgumentParser();p.add_argument('--base');p.add_argument('--candidate');p.add_argument('--output');p.add_argument('--device',default='cpu');p.add_argument('--python');p.add_argument('--rounds',type=int,default=5);p.add_argument('--rss-gib',type=float,default=8);p.add_argument('--continue-on-failure',action='store_true');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[]
for i in range(a.rounds):
    for label,root in [('parent',a.base),('candidate',a.candidate)]:
        target=out/f'{i}-{label}'
        command=[a.python,str(Path(__file__).with_name('stack_worker.py')),'--root',root,'--output',str(target),'--device',a.device]
        with (out/f'{i}-{label}.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start=time.monotonic();status=None;peak=0
            while process.poll() is None:
                try:
                    proc=psutil.Process(process.pid);rss=proc.memory_info().rss+sum(c.memory_info().rss for c in proc.children(recursive=True))
                    peak=max(peak,rss)
                except psutil.Error: pass
                if peak>a.rss_gib*2**30 or time.monotonic()-start>600:
                    status='RSS STOP' if peak>a.rss_gib*2**30 else 'TIMEOUT';os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=20);break
                time.sleep(.2)
        status=status or ('PASS' if process.returncode==0 else 'ERROR')
        rows.append(dict(round=i,label=label,status=status,supervisor_peak_rss=peak,process_s=time.monotonic()-start))
        (out/'summary.json').write_text(json.dumps(rows,indent=2))
        if status!='PASS' and not a.continue_on_failure:raise SystemExit(f'{label}: {status}')
