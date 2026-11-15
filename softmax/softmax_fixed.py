"""Signed INT8 softmax boundary, zero points zero, wide integer internals."""
from dataclasses import asdict, dataclass
import numpy as np

PROBABILITY_MAX = 127
EXPONENTIAL_MAX = 65535
MAX_ROW_LENGTH = 196


@dataclass(frozen=True)
class Config:
    score_scale: float = 1/8

    def __post_init__(self):
        if not np.isfinite(self.score_scale) or self.score_scale <= 0:
            raise ValueError('Score scale must be finite and positive')

    def metadata(self):
        return {**asdict(self), 'score_bits': 8, 'score_signed': True,
                'probability_bits': 8, 'probability_signed': True,
                'input_zero_point': 0, 'output_zero_point': 0,
                'probability_scale': 1/PROBABILITY_MAX,
                'probability_max': PROBABILITY_MAX, 'exponential_bits': 16,
                'difference_bits': 9, 'lut_entries': 256,
                'sum_bits': (MAX_ROW_LENGTH*EXPONENTIAL_MAX).bit_length(),
                'numerator_bits': (EXPONENTIAL_MAX*PROBABILITY_MAX).bit_length(),
                'max_row_length': MAX_ROW_LENGTH,
                'score_rounding': 'nearest ties to even, signed INT8 saturation',
                'lut_rounding': 'nearest ties to even',
                'normalization': 'round_ties_even(exp_code*127 / sum); independently rounded',
                'tail_policy': 'all differences 0..255 have a LUT entry; tiny exponentials round to zero'}


def make_lut(config):
    return np.rint(np.exp(-np.arange(256)*config.score_scale)*EXPONENTIAL_MAX).astype(np.int64)


def quantize_scores(scores, config):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 0 or scores.size == 0 or scores.shape[-1] == 0 or not np.isfinite(scores).all():
        raise ValueError('Expected nonempty finite score rows')
    with np.errstate(over='ignore'):
        rounded = np.rint(scores/config.score_scale)
    clipped = int(np.count_nonzero((rounded < -128) | (rounded > 127)))
    return np.clip(rounded, -128, 127).astype(np.int8), clipped


def softmax_codes(score_codes, config, lut=None):
    q = np.asarray(score_codes)
    if q.dtype.kind not in 'iu' or q.ndim == 0 or q.size == 0 or not 1 <= q.shape[-1] <= MAX_ROW_LENGTH:
        raise ValueError('Expected integer rows of length 1..196')
    if np.any(q < -128) or np.any(q > 127):
        raise ValueError('Score codes must be signed INT8 in [-128,127]')
    q = q.astype(np.int64)
    table = make_lut(config) if lut is None else np.asarray(lut)
    if table.shape != (256,) or table.dtype.kind not in 'iu' or np.any(table < 0) or np.any(table > EXPONENTIAL_MAX) or table[0] != EXPONENTIAL_MAX:
        raise ValueError('Expected 256 unsigned 16-bit ROM entries with ROM[0]=65535')
    table = table.astype(np.int64)
    difference = q.max(axis=-1, keepdims=True)-q
    exp_codes = table[difference]
    total = exp_codes.sum(axis=-1, keepdims=True)
    numerator = exp_codes*PROBABILITY_MAX
    quotient, remainder = np.divmod(numerator, total)
    quotient += ((2*remainder > total) | ((2*remainder == total) & ((quotient & 1) != 0)))
    return quotient.astype(np.int8), {'zero_exponential_count': int(np.count_nonzero(exp_codes == 0)),
                                     'sum_min': int(total.min()), 'sum_max': int(total.max())}


def softmax_fixed(scores, config=Config()):
    q, clipped = quantize_scores(scores, config)
    codes, stats = softmax_codes(q, config)
    stats['input_clipped_count'] = clipped
    return codes.astype(np.float64)/PROBABILITY_MAX, codes, stats
