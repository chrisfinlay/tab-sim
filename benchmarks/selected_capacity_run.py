import argparse,json,os,signal,subprocess,time,math
from pathlib import Path
import psutil
from benchmarks.stack_sweep import stop_group
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--channels',type=int,required=True);p.add_argument('--device',default='cpu');p.add_argument('--timeout',type=int,default=800);p.add_argument('--rss-gib',type=float,default=8);a=p.parse_args()
if not math.isfinite(a.rss_gib) or a.rss_gib<=0 or a.timeout<=0 or a.channels<=0:p.error('Require positive finite RSS budget, timeout and channels')
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
cmd=[__import__('sys').executable,str(Path(__file__).with_name('selected_capacity_worker.py')),'--root',a.root,'--output',str(out/'worker'),'--channels',str(a.channels),'--device',a.device]
start=time.perf_counter();peak=0;status=None
with (out/'worker.log').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  while proc.poll() is None:
   try:
    process=psutil.Process(proc.pid);peak=max(peak,sum(p.memory_info().rss for p in [process]+process.children(recursive=True)))
   except psutil.NoSuchProcess:pass
   except (psutil.Error,OSError):status='MONITOR ERROR';break
   if peak>a.rss_gib*2**30:status='RSS STOP';break
   if time.perf_counter()-start>a.timeout:status='TIMEOUT';break
   time.sleep(.2)
 finally:
  if proc.poll() is None:stop_group(proc)
status=status or ('PASS' if proc.returncode==0 else 'ERROR')
if status=='PASS':
 try:
  report=json.loads((out/'worker/result.json').read_text())
  if report.get('status')!='PASS' or report.get('validated') is not True:status='INVALID REPORT'
 except (OSError,ValueError,TypeError):status='INVALID REPORT'
(out/'summary.json').write_text(json.dumps(dict(status=status,peak_rss=peak,process_s=time.perf_counter()-start,returncode=proc.returncode),indent=2))
raise SystemExit(0 if status=='PASS' else 1)
