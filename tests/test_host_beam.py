"""Numerical contract and absence of implicit accelerator work for host blocks."""
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import jv, jn_zeros

from tabsim.beam import airy_beam


def legacy(theta, freqs, diameter):
    theta=jnp.deg2rad(jnp.asarray(theta[...,None]))
    freqs=jnp.asarray(freqs)
    diameter=jnp.asarray(diameter).flatten()[0]
    x=jnp.where(theta==0.,sys.float_info.epsilon,
                jnp.pi*freqs[None,None,None,:]*diameter*jnp.sin(theta)/2.99792458e8)
    return np.asarray(2*jv(1,x)/x)


@pytest.fixture(autouse=True)
def x64():
    previous=jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64',True)
    yield
    jax.config.update('jax_enable_x64',previous)


def test_boresight_near_zero_sidelobes_and_nulls():
    freqs=np.array([50e6,150e6,1.4e9]);diameter=35.
    # Include exact null arguments for the middle frequency and their neighbourhood.
    nulls=np.rad2deg(np.arcsin(jn_zeros(1,6)*2.99792458e8/(np.pi*150e6*diameter)))
    angles=np.r_[0.,1e-14,1e-9,-1e-9,.001,1.,15.,45.,89.,90.,120.,180.,
                 nulls,nulls-1e-10,nulls+1e-10].reshape(-1,1,1)
    actual=airy_beam(angles,freqs,diameter)
    np.testing.assert_allclose(actual,legacy(angles,freqs,diameter),rtol=2e-12,atol=2e-14)
    np.testing.assert_allclose(actual[0],1.,rtol=0,atol=3e-15)
    assert np.any(actual<0)
    assert actual.shape==angles.shape+(3,)
    assert isinstance(actual,np.ndarray) and actual.dtype==np.float64


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_input_contract(dtype):
    theta=np.linspace(0,85,24,dtype=dtype).reshape(2,3,4)
    freqs=np.array([50e6,150e6],dtype=dtype)
    actual=airy_beam(theta,freqs,dtype(13.5))
    tol=3e-6 if dtype==np.float32 else 2e-12
    np.testing.assert_allclose(actual,legacy(theta,freqs,dtype(13.5)),rtol=tol,atol=tol*1e-2)
    assert actual.dtype==dtype


def test_host_beam_disallows_all_implicit_device_transfers():
    theta=np.ones((2,3,4));freq=np.array([100e6,200e6])
    # Public JAX transfer guard: no JAX array construction is permitted here.
    with jax.transfer_guard('disallow'):
        actual=airy_beam(theta,freq,35.)
    assert np.isfinite(actual).all()


def test_dask_host_blocks_and_compatibility_name():
    import dask.array as da
    from tabsim.dask.interferometry import airy_beam as mapped
    from tabsim.jax.interferometry import airy_beam as historical
    theta=np.linspace(0,90,30).reshape(2,5,3);freq=np.arange(4)*1e6+100e6
    blocks=mapped(da.from_array(theta,chunks=(1,2,3)),da.from_array(freq,chunks=3),35.)
    with jax.transfer_guard('disallow'):
        actual=blocks.compute(scheduler='synchronous')
    np.testing.assert_allclose(actual,airy_beam(theta,freq,35.),rtol=2e-12,atol=2e-14)
    assert historical is airy_beam


@pytest.mark.parametrize('angle_dtype,freq_dtype,diameter', [
    (np.float32,np.float32,np.float64(35.)),
    (np.float64,np.float32,np.float32(35.)),
    (np.float32,np.float64,np.float32(35.)),
    (np.float32,np.float32,35.),
])
def test_mixed_float_precision(angle_dtype,freq_dtype,diameter):
    theta=np.array([0.,.001,15.,45.],dtype=angle_dtype).reshape(-1,1,1)
    freq=np.array([50e6,150e6],dtype=freq_dtype)
    expected=legacy(theta,freq,diameter)
    actual=airy_beam(theta,freq,diameter)
    assert actual.dtype==expected.dtype
    np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=3e-8)


def test_host_precision_is_independent_of_jax_global_x64():
    jax.config.update('jax_enable_x64',False)
    result=airy_beam(np.zeros((1,1,1),dtype=np.float64),np.array([150e6]),35.)
    assert result.dtype==np.float64
