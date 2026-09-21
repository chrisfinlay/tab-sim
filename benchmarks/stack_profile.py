import argparse,os,sys,json,gzip,collections,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--output');a=p.parse_args()
os.environ.update(JAX_PLATFORMS='cuda',JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_MEM_FRACTION='.5')
sys.path.insert(0,a.root)
import numpy as np,jax
from tabsim.jax.interferometry import airy_beam,rfi_vis
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
theta=np.linspace(0,85,512*3*68).reshape(512,3,68);freq=150e6+np.arange(8)*1e5
np.asarray(airy_beam(theta,freq,35.))
with jax.profiler.trace(str(out/'trace')):
    for _ in range(3): np.asarray(airy_beam(theta,freq,35.))
traces=list((out/'trace').rglob('*.trace.json.gz'));events=[]
for f in traces:
    with gzip.open(f,'rt') as stream: events+=json.load(stream).get('traceEvents',[])
if not traces or not events:
    raise RuntimeError('No profiler trace events captured; zero transfers cannot be established')
counts=collections.Counter();details=[];copy_bytes=collections.Counter()
for event in events:
    name=event.get('name','')
    if 'memcpy' in name.lower() or 'synchronize' in name.lower():
        counts[name]+=1
        size=re.search(r"size:(\d+)",str(event.get("args",{})))
        if size:copy_bytes[name]+=int(size[1])
        if len(details)<40:details.append({k:event[k] for k in ('name','cat','args','dur') if k in event})
analysis={}
for ct in (8,16):
    shapes=[jax.ShapeDtypeStruct((512,ct,3,68,32),np.float64),jax.ShapeDtypeStruct((512,ct,3,68),np.float64),
            jax.ShapeDtypeStruct((32,),np.float64),jax.ShapeDtypeStruct((2278,),np.int64),jax.ShapeDtypeStruct((2278,),np.int64)]
    compiled=jax.jit(rfi_vis).lower(*shapes).compile();mem=compiled.memory_analysis()
    analysis[str(ct)]={key:getattr(mem,key,None) for key in ['argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes']}
import hashlib,importlib.metadata,subprocess
revision=subprocess.check_output(['git','-C',a.root,'rev-parse','HEAD'],text=True).strip()
provenance=dict(revision=revision,versions={name:importlib.metadata.version(name) for name in ('jax','jaxlib','numpy','scipy')},
    source_sha256={name:hashlib.sha256((Path(a.root)/name).read_bytes()).hexdigest()
        for name in ('tabsim/jax/interferometry.py','tabsim/beam.py') if (Path(a.root)/name).exists()})
(out/'profile.json').write_text(json.dumps(dict(provenance=provenance,trace_event_count=len(events),counts=dict(counts),copy_bytes=dict(copy_bytes),event_details=details,kernel_memory=analysis,trace_files=len(traces)),indent=2))
