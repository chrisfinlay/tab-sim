"""Matched single warm measurement; orchestrator alternates parent/candidate five times."""
import argparse, json, os, shutil, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--output');p.add_argument('--chunk',type=float,default=45);p.add_argument('--working',type=float);p.add_argument('--device',default='cpu');p.add_argument('--gpu-slots',type=int,default=1);a=p.parse_args()
os.environ['JAX_PLATFORMS']='cuda' if a.device=='gpu' else 'cpu'
os.environ['JAX_ENABLE_X64']='true';os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.5'
import sys
sys.path.insert(0,a.root)
import dask, jax, numpy as np
import tabsim
assert Path(tabsim.__file__).resolve().is_relative_to(Path(a.root).resolve())
from benchmarks.harness import build_observation,add_sources,MemorySampler,offline
from tabsim.dask.extras import get_chunksizes
case=dict(telescope='SKA-Low-AA2',antennas=68,times=16,channels=32,samples=3,point_sources=8,rfi_sources=512)
try:
    from tabsim.execution import configure_execution,execution_stats
    configure_execution(gpu_concurrency=a.gpu_slots)
except ImportError:
    execution_stats=None
offline();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
def run(name):
    started=time.perf_counter()
    with dask.config.set(scheduler='synchronous',num_workers=2):
        if a.working is None:
            obs=build_observation(case,a.chunk)
        else:
            # Constructor planning must happen before sources; rebuild with same public inputs.
            from tabsim.dask.observation import Observation
            from tabsim.config import get_telescope_definitions
            definition=get_telescope_definitions(case['telescope'])
            obs=Observation(latitude=definition['latitude'],longitude=definition['longitude'],elevation=definition['elevation'],
              ra=30.,dec=-30.,times_mjd=60000.+np.arange(16)*2/86400.,freqs=150e6+np.arange(32)*1e5,
              SEFD=np.full(32,5000.),ITRF_path=definition['itrf_path'],dish_d=definition['dish_d'],
              int_time=2.,chan_width=1e5,n_int_samples=3,random_seed=20260919,max_chunk_MB=a.chunk,
              working_set_MB=a.working,planned_rfi_sources=512,planned_ast_sources=8,component_workers=2)
        add_sources(obs,case);obs.calculate_vis()
        stages=[]
        ds=obs.write_to_zarr(out/name,progress=lambda event,data:stages.append(data) if event=='stage_complete' else None)
        elapsed=time.perf_counter()-started
        sample={key:np.asarray(ds[key].isel(time=[0,7,15],bl=[0,100,2277],freq=[0,15,31]).compute()).reshape(-1) for key in ['vis_ast','vis_rfi','vis_obs','noise_data','vis_calibrated']}
        ant1=np.asarray(ds.antenna1)[[0,100,2277]];ant2=np.asarray(ds.antenna2)[[0,100,2277]]
        gains=np.asarray(ds.gains_ants.compute())[[0,7,15]][:,:,[0,15,31]]
        product=(gains[:,ant1,:]*gains[:,ant2,:].conj()).reshape(-1)
        np.testing.assert_allclose(sample['vis_obs'],(sample['vis_ast']+sample['vis_rfi'])*product+sample['noise_data'],rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(sample['vis_calibrated'],sample['vis_obs']/product,rtol=1e-9,atol=1e-9)
        result=dict(total_s=elapsed,chunks=[obs.time_chunk,obs.freq_chunk],case=case,stages=stages,
          chunk_plan=getattr(obs,'chunk_plan',None),samples={k:dict(real=v.real.tolist(),imag=v.imag.tolist()) for k,v in sample.items()},
          output_bytes=sum(f.stat().st_size for f in (out/name).rglob('*') if f.is_file()))
        ds.close();shutil.rmtree(out/name)
        return result
run('warmup')
with MemorySampler() as memory:
    result=run('timed')
result['execution']=execution_stats() if execution_stats else None
result['peak_rss']=memory.peak_rss
result['peak_gpu_reserved']=memory.peak_gpu
result['jax_memory']=jax.devices()[0].memory_stats()
import hashlib, importlib.metadata, subprocess
try:
    revision=subprocess.check_output(['git','-C',a.root,'rev-parse','HEAD'],text=True).strip()
except subprocess.CalledProcessError:
    revision=None
import platform, socket
result['provenance']=dict(environment={k:os.getenv(k) for k in ('JAX_PLATFORMS','JAX_ENABLE_X64','CUDA_VISIBLE_DEVICES','XLA_PYTHON_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_PREALLOCATE','OMP_NUM_THREADS')}, python=platform.python_version(), host=socket.gethostname(), device_kind=jax.devices()[0].device_kind, harness_sha256=hashlib.sha256(Path(__file__).read_bytes() + Path(sys.modules['benchmarks.harness'].__file__).read_bytes()).hexdigest(), revision=revision, device=a.device,
    versions={name:importlib.metadata.version(name) for name in ('jax','jaxlib','numpy','scipy','dask','xarray','zarr','astropy','pandas','numcodecs')},
    source_sha256={name:hashlib.sha256((Path(a.root)/name).read_bytes()).hexdigest()
       for name in ('tabsim/dask/extras.py','tabsim/jax/interferometry.py','tabsim/dask/interferometry.py','tabsim/beam.py','tabsim/execution.py','tabsim/dask/coordinates.py','tabsim/dask/observation.py','tabsim/staged.py','tabsim/config.py') if (Path(a.root)/name).exists()})
from tabsim.jax.interferometry import airy_beam
micro={}
for name,shape,nfreq in [('point',(512,1,1),32),('rfi',(512,3,68),8)]:
    angles=np.linspace(0,85,np.prod(shape)).reshape(shape)
    frequencies=150e6+np.arange(nfreq)*1e5
    np.asarray(airy_beam(angles,frequencies,35.))
    durations=[]
    for _ in range(5):
        started=time.perf_counter();value=np.asarray(airy_beam(angles,frequencies,35.));durations.append(time.perf_counter()-started)
    try:
        with jax.transfer_guard('disallow'):
            np.asarray(airy_beam(angles,frequencies,35.))
        transfers_allowed=False
    except Exception:
        transfers_allowed=True
    micro[name]=dict(shape=shape,nfreq=nfreq,seconds=durations,requires_implicit_transfer=transfers_allowed)
result['beam_micro']=micro
(out/'result.json').write_text(json.dumps(result,indent=2))
