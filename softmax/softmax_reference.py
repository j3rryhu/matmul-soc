"""Stable floating-point softmax oracle; normalization is over the last axis."""
import numpy as np


def softmax_reference(scores):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] == 0 or not np.isfinite(values).all():
        raise ValueError("Expected nonempty rows of finite scores")
    exponentials = np.exp(values - values.max(axis=-1, keepdims=True))
    return exponentials / exponentials.sum(axis=-1, keepdims=True)
