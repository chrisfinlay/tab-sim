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
    diameter = np.asarray(dish_d).reshape(-1)[0].item()
    x = np.where(theta == 0.0, sys.float_info.epsilon,
                 np.pi * freqs[None, None, None, :] * diameter * np.sin(theta) / 2.99792458e8)
    return 2 * jv(1, x) / x
