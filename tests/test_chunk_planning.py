import math
import subprocess
import sys

import numpy as np
import pytest

from tabsim.dask.extras import get_chunksizes, get_factors, estimate_working_set


@pytest.mark.parametrize('t,f,i,b,limit', [(17,19,3,6,.001), (32,128,3,130816,64),
    (1,1,1,1,.000016), (12,12,5,15,.01), (101,103,1,1,.001)])
def test_strict_feasible_uniform_chunks(t,f,i,b,limit):
    result = get_chunksizes(t,f,i,b,limit)
    ct,cf=result['time'],result['freq']
    assert t % ct == f % cf == 0
    assert result['nominal_bytes'] == 16*i*b*ct*cf <= limit*1e6
    assert ct*cf == max(x*y for x in get_factors(t) for y in get_factors(f) if 16*i*b*x*y<=limit*1e6)


def test_planner_does_not_import_jax():
    code = 'from tabsim.dask.extras import get_chunksizes; import sys; get_chunksizes(17,19,3,6,.001); assert "jax" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('limit', [0,-1,float('nan'),float('inf'),True])
def test_invalid_limits(limit):
    with pytest.raises(ValueError): get_chunksizes(4,4,3,6,limit)


def test_below_minimum_and_exact_boundary():
    with pytest.raises(ValueError, match='No feasible'):
        get_chunksizes(4,4,3,6,.000287)
    assert get_chunksizes(4,4,3,6,.000288)['nominal_bytes'] == 288


def test_working_set_budget_and_source_scaling():
    kwargs=dict(n_ant=512,n_rfi=512,n_ast=8,workers=2,backend='gpu')
    minimum=estimate_working_set(1,1,3,130816,**kwargs)['estimated_bytes']
    with pytest.raises(ValueError,match='No feasible'):
        get_chunksizes(32,128,3,130816,1024,working_set_MB=minimum/1e6-.001,**kwargs)
    plan=get_chunksizes(32,128,3,130816,1024,working_set_MB=minimum*4/1e6,**kwargs)
    assert plan['working_set']['estimated_bytes']<=minimum*4
    model=plan['working_set']
    assert model['amplitude_bytes']==8*512*plan['time']*3*512*plan['freq']
    assert model['device_bytes']>0
    assert estimate_working_set(1,1,3,6,n_ant=4)['device_bytes']==0


def test_source_budget_checks_before_broadcast_or_mutation():
    # Use real Observation methods with a tiny lightweight planning state.
    from tabsim.dask.observation import Observation
    import dask.array as da
    obs=Observation.__new__(Observation)
    obs._planning=dict(n_ant=4,n_rfi=0,n_ast=0,workers=1,backend='cpu',scratch_factor=4)
    obs.n_rfi=0; obs.n_ast=0
    obs.n_int_samples=3; obs.n_bl=6; obs.time_chunk=1; obs.freq_chunk=1
    obs.working_set_MB=.2
    with pytest.raises(ValueError,match='Sources/concurrency'):
        obs._check_source_budget('rfi',da.ones((1,1,1)),np.arange(10000))
    assert obs.n_rfi==0
    obs.working_set_MB=None
    data=obs._check_source_budget('rfi',da.ones((1,1,1)),np.arange(3))
    assert data.shape==(3,1,1)
    with pytest.raises(ValueError,match='must match'):
        obs._check_source_budget('rfi',da.ones((2,1,1)),np.arange(3))


def test_real_source_methods_broadcast_single_spectrum():
    import jax
    import tabsim.config
    from benchmarks.harness import build_observation
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        obs=build_observation(dict(telescope='SKA-Low-AA0.5',antennas=4,times=4,channels=3,samples=3),.001)
        obs.addAstro(np.ones(3),np.array([30.,31.]),np.array([-30.]))
        obs.addStationaryRFI(np.full(3,1e-12),np.array([-26.,-25.]),np.array([116.]),np.array([1000.]))
        assert obs.n_ast==obs.n_rfi==2
        ds=obs.calculate_vis()
        assert ds.rfi_stat_A.shape[0]==2
        assert np.isfinite(ds.vis_obs.compute()).all()
    finally:
        jax.config.update('jax_enable_x64', previous)


def test_full_band_gain_allowance_survives_frequency_tiling():
    small=estimate_working_set(1,1,3,6,n_ant=4,full_n_freq=4096)
    large=estimate_working_set(1,32,3,6,n_ant=4,full_n_freq=4096)
    assert small['full_band_gain_bytes']==large['full_band_gain_bytes']==16*2*4*4096
    assert small['host_bytes']>=2*(small['gain_mode_allowance_bytes']+3*small['full_band_gain_bytes'])


def test_gpu_host_model_includes_beam_and_readback_buffers():
    model=estimate_working_set(8,32,3,2278,n_ant=68,n_rfi=512,backend='gpu')
    assert model['host_bytes'] >= 2*(model['amplitude_bytes']+model['scratch_bytes'])
