"""Device primary beams, numerically matching the host reference in tabsim.beam."""
import sys

import jax.numpy as jnp

# Cephes j1 (Moshier, release 2.8) as bundled with SciPy in xsf/cephes/j1.h.
# jax.scipy.special.bessel_jn loses accuracy for x above about 40.
_RP = (-8.99971225705559398224e8, 4.52228297998194034323e11,
       -7.27494245221818276015e13, 3.68295732863852883286e15)
_RQ = (1.0, 6.20836478118054335476e2, 2.56987256757748830383e5,
       8.35146791431949253037e7, 2.21511595479792499675e10,
       4.74914122079991414898e12, 7.84369607876235854894e14,
       8.95222336184627338078e16, 5.32278620332680085395e18)
_PP = (7.62125616208173112003e-4, 7.31397056940917570436e-2,
       1.12719608129684925192e0, 5.11207951146807644818e0,
       8.42404590141772420927e0, 5.21451598682361504063e0,
       1.00000000000000000254e0)
_PQ = (5.71323128072548699714e-4, 6.88455908754495404082e-2,
       1.10514232634061696926e0, 5.07386386128601488557e0,
       8.39985554327604159757e0, 5.20982848682361821619e0,
       9.99999999999999997461e-1)
_QP = (5.10862594750176621635e-2, 4.98213872951233449420e0,
       7.58238284132545283818e1, 3.66779609360150777800e2,
       7.10856304998926107277e2, 5.97489612400613639965e2,
       2.11688757100572135698e2, 2.52070205858023719784e1)
_QQ = (1.0, 7.42373277035675149943e1, 1.05644886038262816351e3,
       4.98641058337653607651e3, 9.56231892404756170795e3,
       7.99704160447350683650e3, 2.82619278517639096600e3,
       3.36093607810698293419e2)
_Z1 = 1.46819706421238932572e1
_Z2 = 4.92184563216946036703e1
_THPIO4 = 2.356194490192344928846982537459627163148
_SQRT2OPI = 7.978845608028653558798921198687637369517e-1
_SQRT1_2 = 0.707106781186547524400844362104849039
_SQRT_EPS = 1.490116119384765625e-8


def _polevl(x, coef):
    # Horner's scheme, highest degree first, as Cephes polevl/p1evl.
    y = coef[0]
    for c in coef[1:]:
        y = y * x + c
    return y


def j1(x):
    """Bessel function of the first kind of order one, in float64."""
    x = jnp.asarray(x, dtype=jnp.float64)
    ax = jnp.abs(x)
    # Rational approximation on [0, 5]; keep the other branch finite there.
    xs = jnp.minimum(ax, 5.0)
    z = xs * xs
    small = _polevl(z, _RP) / _polevl(z, _RQ) * xs * (z - _Z1) * (z - _Z2)
    small = jnp.where(ax < _SQRT_EPS, 0.5 * ax, small)
    # Asymptotic P/Q expansion above 5.
    xl = jnp.maximum(ax, 5.0)
    w = 5.0 / xl
    z = w * w
    p = _polevl(z, _PP) / _polevl(z, _PQ)
    q = _polevl(z, _QP) / _polevl(z, _QQ)
    xn = xl - _THPIO4
    mid = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    a = _SQRT1_2 * (w * q - p)
    b = _SQRT1_2 * (p + w * q)
    far = a * jnp.cos(xl) + b * jnp.sin(xl)
    large = jnp.where(xl < 10.0, mid, far) * _SQRT2OPI / jnp.sqrt(xl)
    y = jnp.where(ax <= 5.0, small, large)
    return jnp.where(x < 0, -y, y)


def airy_beam(theta, freqs, dish_d):
    """Airy voltage for angular separations in degrees and frequencies in Hz.

    Device counterpart of :func:`tabsim.beam.airy_beam` for float64 inputs:
    theta (..., antenna) gives (..., antenna, frequency), with the same
    operation order, signed sidelobes and epsilon substitution at boresight.
    """
    theta = jnp.deg2rad(jnp.asarray(theta)[..., None])
    scale = jnp.pi * jnp.asarray(freqs) * jnp.asarray(dish_d).reshape(-1)[0]
    x = jnp.where(theta == 0.0, sys.float_info.epsilon,
                  scale * jnp.sin(theta) / 2.99792458e8)
    return 2 * j1(x) / x
