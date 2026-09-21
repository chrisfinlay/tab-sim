"""Host-block primary beams. No accelerator dispatch or device synchronization."""
import sys

import numpy as np
from scipy.special import jv


def airy_beam(theta: np.ndarray, freqs: np.ndarray, dish_d: float) -> np.ndarray:
    """Airy voltage for angular separations in degrees and frequencies in Hz.

    theta has shape (source, time, antenna); the returned NumPy array has shape
    (source, time, antenna, frequency). Preserve the original signed sidelobes,
    sin(theta) expression and epsilon substitution at boresight. NumPy/SciPy
    evaluate the entire host block; callers with device arrays explicitly incur
    conversion of those inputs to host memory.
    """
    theta = np.deg2rad(np.asarray(theta)[..., None])
    freqs = np.asarray(freqs)
    # Python scalars are weak; explicitly typed diameters retain their dtype.
    # Preserve operation order too: do not promote float32 angles before sin.
    scale = np.pi * freqs
    if type(dish_d) in (int, float):
        scale = scale * dish_d
    else:
        diameter = np.asarray(dish_d).reshape(-1)[0]
        # NumPy 1.x uses value-based scalar promotion; request dtype promotion
        # here so explicitly typed diameters behave consistently with NumPy 2.
        dtype = np.result_type(scale.dtype, diameter.dtype)
        scale = scale.astype(dtype, copy=False) * np.asarray(diameter, dtype=dtype)
    x = np.where(theta == 0.0, sys.float_info.epsilon,
                 scale[None, None, None, :] * np.sin(theta) / 2.99792458e8)
    return 2 * jv(1, x) / x
