"""Matched retained-output benchmark, separating product choice from code speed."""
import argparse,json,os,sys,time,shutil,threading
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--profile',default='default');p.add_argument('--gpu-slots',type=int,default=1);a=p.parse_args()
os.environ.update(JAX_PLATFORMS='cuda' if a.device=='gpu' else 'cpu',JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_MEM_FRACTION='.5');sys.path.insert(0,a.root)
import dask,jax,numpy as np
from dask.callbacks import Callback
import tabsim
assert Path(tabsim.__file__).resolve().is_relative_to(Path(a.root).resolve())
from benchmarks.harness import build_observation,add_sources,offline,MemorySampler
from tabsim.execution import configure_execution
configure_execution(gpu_concurrency=a.gpu_slots);offline()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
case=dict(telescope='SKA-Low-AA4',antennas=512,times=8,channels=64,samples=3,point_sources=8,rfi_sources=2)
def run(name):
 start=time.perf_counter();tasks=0;lock=threading.Lock();stages=[]
 def posttask(*args):
  nonlocal tasks
  with lock:tasks+=1
 with dask.config.set(scheduler='synchronous',num_workers=2):
  obs=build_observation(case,64);add_sources(obs,case);obs.calculate_vis()
  from tabsim.staged import select_arrays
  selected=select_arrays(obs.dataset) if a.profile=='default' else select_arrays(obs.dataset,output_profile=a.profile)
  graph_size=len(obs.dataset[list(selected)].__dask_graph__())
  writer_start=time.perf_counter()
  with Callback(posttask=posttask):
   ds=obs.write_to_zarr(out/name,save_arrays=selected,progress=lambda e,d:stages.append(d) if e=='stage_complete' else None)
  end=time.perf_counter()
  sample=np.asarray(ds.vis_obs.isel(time=[0,3,7],bl=[0,100,130815],freq=[0,31,63]).compute()).reshape(-1)
  assert np.isfinite(sample).all()
  result=dict(chunks=[obs.time_chunk,obs.freq_chunk],total_s=end-start,writer_s=end-writer_start,profile=a.profile,case=case,retained=sorted(ds.data_vars),logical_bytes=ds.nbytes,output_bytes=sum(f.stat().st_size for f in (out/name).rglob('*') if f.is_file()),selected_unoptimized_graph_tasks=graph_size,executed_writer_tasks=tasks,stages=stages,sample=dict(real=sample.real.tolist(),imag=sample.imag.tolist()))
  ds.close();shutil.rmtree(out/name)
 return result
warm=run('warmup')
with MemorySampler() as memory:result=run('timed')
for part in ('real','imag'):np.testing.assert_allclose(result['sample'][part],warm['sample'][part],rtol=1e-9,atol=1e-9)
result.update(peak_rss=memory.peak_rss,peak_gpu_reserved=memory.peak_gpu,jax_memory=jax.devices()[0].memory_stats())
import hashlib,subprocess,importlib.metadata
try:
 revision=subprocess.check_output(['git','-C',a.root,'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
except subprocess.CalledProcessError:
 revision=None
import platform, socket
result['provenance']=dict(environment={k:os.getenv(k) for k in ('JAX_PLATFORMS','JAX_ENABLE_X64','CUDA_VISIBLE_DEVICES','XLA_PYTHON_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_PREALLOCATE','OMP_NUM_THREADS')}, python=platform.python_version(), host=socket.gethostname(), device_kind=jax.devices()[0].device_kind, harness_sha256=hashlib.sha256(Path(__file__).read_bytes() + Path(sys.modules['benchmarks.harness'].__file__).read_bytes()).hexdigest(), revision=revision,versions={n:importlib.metadata.version(n) for n in ('jax','jaxlib','numpy','scipy','dask','xarray','zarr','astropy','pandas','numcodecs')},source_sha256={n:hashlib.sha256((Path(a.root)/n).read_bytes()).hexdigest() for n in ('tabsim/execution.py','tabsim/dask/observation.py','tabsim/dask/coordinates.py','tabsim/staged.py','tabsim/config.py') if (Path(a.root)/n).exists()})
(out/'result.json').write_text(json.dumps(result,indent=2))
