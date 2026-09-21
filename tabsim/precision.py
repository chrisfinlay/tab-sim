"""Visibility storage/accumulation precision; geometry remains float64."""

import numpy as np


def visibility_dtype(visibility_precision="single"):
    """Validate the public precision choice and return its complex dtype."""
    if visibility_precision not in ("single", "double"):
        raise ValueError("visibility_precision must be 'single' or 'double'")
    return np.dtype("complex64" if visibility_precision == "single" else "complex128")
