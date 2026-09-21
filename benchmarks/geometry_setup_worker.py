"""Geometry setup scaling with a small runtime warm-up and bounded slice check."""
import argparse,sys,os,json,time,gc
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--times',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
os.environ.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true');sys.path.insert(0,a.root)
import numpy as np,psutil,jax
import tabsim
assert Path(tabsim.__file__).resolve().is_relative_to(Path(a.root).resolve())
from benchmarks.harness import build_observation,offline,MemorySampler
offline();base=dict(telescope='SKA-Low-AA2',antennas=68,times=16,channels=32,samples=3,point_sources=0,rfi_sources=0)
obs=build_observation(base,16);del obs;gc.collect();baseline=psutil.Process().memory_info().rss
case=dict(base,times=a.times)
with MemorySampler() as memory:
 started=time.perf_counter();obs=build_observation(case,16);elapsed=time.perf_counter()-started
result=dict(times=a.times,setup_s=elapsed,baseline_rss=baseline,peak_rss=memory.peak_rss,time_chunk=obs.time_chunk,frequency_chunk=obs.freq_chunk,logical_geometry_bytes=obs.ants_uvw.nbytes+obs.ants_xyz.nbytes)
# Bounded slice after the measured setup validates that deferred results execute.
result['geometry_sample']=np.asarray(obs.ants_xyz[:2].compute()).reshape(-1)[:12].tolist()
import hashlib,subprocess,importlib.metadata
try:
 revision=subprocess.check_output(['git','-C',a.root,'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
except subprocess.CalledProcessError:
 revision=None
import platform,socket
result['provenance']=dict(environment={k:os.getenv(k) for k in ('JAX_PLATFORMS','JAX_ENABLE_X64','CUDA_VISIBLE_DEVICES','XLA_PYTHON_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_PREALLOCATE','OMP_NUM_THREADS')}, python=platform.python_version(),host=socket.gethostname(),device_kind=jax.devices()[0].device_kind,harness_sha256=hashlib.sha256(Path(__file__).read_bytes() + Path(sys.modules['benchmarks.harness'].__file__).read_bytes()).hexdigest(), revision=revision,versions={n:importlib.metadata.version(n) for n in ('jax','jaxlib','numpy','scipy','dask','xarray','zarr','astropy','pandas','numcodecs')},source_sha256={n:hashlib.sha256((Path(a.root)/n).read_bytes()).hexdigest() for n in ('tabsim/execution.py','tabsim/dask/observation.py','tabsim/dask/coordinates.py','tabsim/staged.py','tabsim/config.py') if (Path(a.root)/n).exists()})
Path(a.output).write_text(json.dumps(result,indent=2))
