# GPU execution boundaries

The Dask visibility and coordinate callbacks explicitly place their inputs on
the worker's device, compute, and return completed NumPy results. A process-wide
GPU slot stays occupied through device-to-host readback. The default is one
active GPU block; independent component streams may still overlap host work and
storage I/O. Airy evaluation stays entirely on the host.

This complements the staged writer from PR 64. A slot bounds active GPU work;
it cannot bound an arbitrary Dask graph's completed host results. Use the default
single-store writer for large outputs. Neither Dask spilling of JAX arrays nor
`persist()` of an entire observation is part of this strategy. Direct functions
in `tabsim.jax` remain ordinary JAX primitives; this policy covers the mapped
Dask callbacks, not all JAX use in a process.

## Local execution

```sh
sim-vis --config_path observation.yaml --gpu-id 0 --gpu-concurrency 1
```

`--gpu-id` sets `CUDA_VISIBLE_DEVICES` before importing simulation/JAX modules.
Use a fresh CLI process; changing the mask after another library initializes
JAX is ineffective. Allocator settings such as `XLA_PYTHON_CLIENT_MEM_FRACTION`
must also be set before startup. CLI/YAML `dask.gpu_concurrency` controls active
GPU callbacks; `component_workers` separately controls concurrent component
streams. Try two GPU slots only with measured headroom and throughput benefit.

Programmatically, set visibility before importing simulation modules:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tabsim.execution import configure_execution, execution_stats
configure_execution(gpu_concurrency=1)
from tabsim.dask.observation import Observation
# Construct and write the observation with its normal public API.
print(execution_stats())
```

The policy is process-local and may only change while no GPU callbacks are active
or waiting. CPU callbacks return host results too, but are not serialized by GPU
admission. The diagnostics report admitted calls, peak active blocks, queue time,
execution/readback time, logical host-result bytes, and the selected logical
device. Device indices are relative to the visibility mask. Live JAX allocations
and allocator-pool reservation must still be measured separately.

## Multiple GPUs: one worker process per GPU

Expose exactly one GPU to each worker before JAX initialization. Mapped callbacks
reject a worker with multiple visible GPUs instead of silently sending every
thread to GPU zero. Do not launch multiple JAX worker processes on the same GPU.
For example, start two workers in separate processes:

```sh
CUDA_VISIBLE_DEVICES=0 dask worker tcp://SCHEDULER:8786 --nworkers 1 --nthreads 2 --resources 'GPU=1'
CUDA_VISIBLE_DEVICES=1 dask worker tcp://SCHEDULER:8786 --nworkers 1 --nthreads 2 --resources 'GPU=1'
```

Submit one complete staged simulation per GPU resource, returning only the path:

```python
def simulate_to_path(config_path):
    from tabsim.execution import configure_execution
    configure_execution(gpu_concurrency=1)
    from tabsim.config import run_sim_config
    _, output_path = run_sim_config(config_path=config_path)
    return output_path

futures = [client.submit(simulate_to_path, path, resources={'GPU': 1}, pure=False)
           for path in config_paths]
```

Give every simulation a distinct output path and make input/configuration files
available on its worker. This recipe distributes independent simulations. The
staged writer deliberately uses local bounded task streams inside each worker;
it does not distribute one observation across GPUs or serialize device buffers
between workers. Configure concurrency in each simulation's YAML when using
`run_sim_config`, which applies that configuration on the worker.
