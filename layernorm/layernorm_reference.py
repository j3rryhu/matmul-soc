"""Independent float64 LayerNorm over the last (channel) axis."""
import numpy as np


def validate_values(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 0 or x.shape[-1] == 0 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite channel vectors")
    return x


def affine_parameter(value, channels, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape not in ((), (channels,)) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite and scalar or shape ({channels},)")
    return value


def layernorm_reference(values, gamma=1., beta=0., eps=1e-5):
    x = validate_values(values)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    gamma = affine_parameter(gamma, x.shape[-1], "gamma")
    beta = affine_parameter(beta, x.shape[-1], "beta")
    centered = x - x.mean(axis=-1, keepdims=True)
    # Population variance, not the unbiased/sample estimator.
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + eps) * gamma + beta
