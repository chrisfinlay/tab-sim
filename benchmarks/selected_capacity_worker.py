"""Bounded-memory full-size selected-output check; intentionally no speed claim."""
import argparse,os,sys,json,time,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--channels',type=int,required=True);p.add_argument('--device',default='cpu');p.add_argument('--slots',type=int,default=1);a=p.parse_args()
os.environ.update(JAX_PLATFORMS='cuda' if a.device=='gpu' else 'cpu',JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_MEM_FRACTION='.5')
sys.path.insert(0,a.root)
import dask,jax,numpy as np,zarr,psutil
from benchmarks.harness import build_observation,add_sources,offline,MemorySampler
from tabsim.execution import configure_execution,execution_stats
configure_execution(gpu_concurrency=a.slots);offline()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);store=out/'result.zarr';events=[];sample=None
case=dict(telescope='SKA-Low-AA4',antennas=512,times=32,channels=a.channels,samples=3,point_sources=8,rfi_sources=2)
start=time.perf_counter()
def progress(event,data):
 global sample
 row=dict(event=event,elapsed_s=time.perf_counter()-start,rss=psutil.Process().memory_info().rss,details=data,execution=execution_stats(),jax_memory=jax.devices()[0].memory_stats())
 events.append(row);(out/'progress.json').write_text(json.dumps(events,indent=2))
 if event=='stage_complete' and data['variable']=='vis_obs':
  group=zarr.open_group(str(store),mode='r');ts=[0,15,31];bs=[0,100,130815];fs=[0,a.channels//2,a.channels-1]
  idx=(ts,bs,fs);values={k:group[k].get_orthogonal_selection(idx) for k in ('vis_obs','vis_ast','vis_rfi','noise_data')}
  pairs=np.triu_indices(case['antennas'],1);a1=pairs[0][bs];a2=pairs[1][bs]
  g1=group['gains_ants'].get_orthogonal_selection((ts,a1,fs));g2=group['gains_ants'].get_orthogonal_selection((ts,a2,fs))
  np.testing.assert_allclose(values['vis_obs'],(values['vis_ast']+values['vis_rfi'])*g1*g2.conj()+values['noise_data'],rtol=1e-9,atol=1e-9)
  sample=values['vis_obs']
with MemorySampler() as memory, dask.config.set(scheduler='synchronous',num_workers=2):
 obs=build_observation(case,64);add_sources(obs,case);obs.calculate_vis()
 ds=obs.write_to_zarr(store,save_arrays=['vis_obs'],progress=progress,max_memory_gb=8 if a.device=='cpu' else 32,disk_reserve_gb=2)
 np.testing.assert_allclose(ds.vis_obs.isel(time=[0,15,31],bl=[0,100,130815],freq=[0,a.channels//2,a.channels-1]).compute(),sample,rtol=0,atol=0)
 result=dict(status='PASS',case=case,selected=['vis_obs'],single_visibility_bytes=obs.n_time*obs.n_bl*obs.n_freq*16,elapsed_s=time.perf_counter()-start,peak_rss=memory.peak_rss,peak_gpu_reserved=memory.peak_gpu,jax_memory=jax.devices()[0].memory_stats(),execution=execution_stats(),chunks=[obs.time_chunk,obs.freq_chunk],events=events,validated=True)
 ds.close()
import hashlib,subprocess,importlib.metadata
try:
 revision=subprocess.check_output(['git','-C',a.root,'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
except subprocess.CalledProcessError:
 revision=None
result['provenance']=dict(revision=revision,versions={n:importlib.metadata.version(n) for n in ('jax','jaxlib','numpy','scipy','dask','xarray','zarr')},source_sha256={n:hashlib.sha256((Path(a.root)/n).read_bytes()).hexdigest() for n in ('tabsim/execution.py','tabsim/dask/observation.py','tabsim/dask/coordinates.py','tabsim/staged.py','tabsim/config.py') if (Path(a.root)/n).exists()})
(out/'result.json').write_text(json.dumps(result,indent=2))
shutil.rmtree(store)
