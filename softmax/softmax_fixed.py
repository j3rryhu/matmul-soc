"""Integer softmax baseline: signed scores, exponential ROM, integer division."""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    score_bits: int = 16
    fractional_bits: int = 8
    cutoff: int = 16
    exponential_bits: int = 16
    probability_bits: int = 16

    def __post_init__(self):
        if not (2 <= self.score_bits <= 24 and 0 <= self.fractional_bits <= 12
                and 1 <= self.cutoff <= 32 and 2 <= self.exponential_bits <= 24
                and 2 <= self.probability_bits <= 24):
            raise ValueError("Unsupported fixed-point configuration")

    def metadata(self, max_row_length=196):
        exp_max = (1 << self.exponential_bits) - 1
        return {**asdict(self), "score_scale": 2. ** -self.fractional_bits,
                "difference_bits": self.score_bits + 1,
                "lut_entries": self.cutoff * (1 << self.fractional_bits) + 1,
                "sum_bits": (max_row_length * exp_max).bit_length(),
                "numerator_bits": (exp_max * ((1 << self.probability_bits) - 1)).bit_length(),
                "rounded_numerator_bits": (exp_max * ((1 << self.probability_bits) - 1) +
                                           max_row_length * exp_max // 2).bit_length(),
                "max_row_length": max_row_length,
                "score_rounding": "nearest, ties to even; signed saturation",
                "lut_rounding": "nearest, ties to even",
                "normalization": "(exp_code * probability_max + sum//2) // sum; half up",
                "tail_policy": "differences greater than cutoff are zero, not clamped to last ROM entry",
                "probability_scale": 1 / ((1 << self.probability_bits) - 1)}


def make_lut(config):
    addresses = np.arange(config.cutoff * (1 << config.fractional_bits) + 1)
    return np.rint(np.exp(-addresses / (1 << config.fractional_bits)) *
                   ((1 << config.exponential_bits) - 1)).astype(np.int64)


def quantize_scores(scores, config):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 0 or scores.shape[-1] == 0 or not np.isfinite(scores).all():
        raise ValueError("Expected nonempty rows of finite scores")
    rounded = np.rint(scores * (1 << config.fractional_bits))
    low, high = -(1 << (config.score_bits - 1)), (1 << (config.score_bits - 1)) - 1
    return np.clip(rounded, low, high).astype(np.int64), int(np.count_nonzero((rounded < low) | (rounded > high)))


def softmax_codes(score_codes, config, lut=None):
    q = np.asarray(score_codes)
    if q.dtype.kind not in "iu" or q.ndim == 0 or not 1 <= q.shape[-1] <= 196:
        raise ValueError("Expected integer rows of length 1..196")
    if np.any(q < -(1 << (config.score_bits - 1))) or np.any(q > (1 << (config.score_bits - 1)) - 1):
        raise ValueError("Score codes exceed configured signed width")
    q = q.astype(np.int64)
    lut = make_lut(config) if lut is None else np.asarray(lut, dtype=np.int64)
    difference = q.max(axis=-1, keepdims=True) - q
    exp_codes = np.where(difference < len(lut), lut[np.minimum(difference, len(lut)-1)], 0)
    total = exp_codes.sum(axis=-1, keepdims=True)
    maximum = (1 << config.probability_bits) - 1
    codes = (exp_codes * maximum + total // 2) // total
    return codes, {"tail_zero_count": int(np.count_nonzero(difference >= len(lut))),
                   "zero_exponential_count": int(np.count_nonzero(exp_codes == 0)),
                   "sum_min": int(total.min()), "sum_max": int(total.max())}


def softmax_fixed(scores, config):
    q, clipped = quantize_scores(scores, config)
    codes, stats = softmax_codes(q, config)
    stats["input_clipped_count"] = clipped
    return codes / ((1 << config.probability_bits) - 1), codes, stats
