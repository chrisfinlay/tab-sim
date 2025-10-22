from jax import jit
from collections.abc import Callable


def jit_with_doc(func: Callable) -> Callable:
    jit_func = jit(func)
    jit_func.__doc__ = func.__doc__
    return jit_func


jit_with_doc.__doc__ = jit.__doc__
