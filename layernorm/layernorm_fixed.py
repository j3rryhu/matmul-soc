"""Standalone INT8 LayerNorm with the matmul/GELU byte and scale convention.

Input, gamma, beta and output storage is signed INT8, zero point zero.
Wide integer intermediates preserve reductions; no alternative storage widths.
"""
from dataclasses import asdict, dataclass
from math import isqrt
import numpy as np

from layernorm_reference import affine_parameter, validate_values

NORM_FRACTION = 14
ROOT_FRACTION = 8
MULTIPLIER_FRACTION = 16
MAX_CHANNELS = 576


def validate_scale(scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Scale must be finite and positive')


def round_divide(numerator, denominator):
    """Signed nearest division, ties to even, matching the GELU quantizer."""
    if denominator <= 0:
        raise ValueError('Denominator must be positive')
    quotient, remainder = divmod(abs(numerator), denominator)
    quotient += int(2 * remainder > denominator or
                    (2 * remainder == denominator and quotient % 2 == 1))
    return -quotient if numerator < 0 else quotient


@dataclass(frozen=True)
class Config:
    # The same demonstration scales as activation/example/gelu_metadata.json.
    input_scale: float = 1/32
    output_scale: float = 1/32
    gamma_scale: float = 1/32
    beta_scale: float = 1/32

    def __post_init__(self):
        for scale in asdict(self).values():
            validate_scale(scale)
        self.multipliers()

    def multipliers(self):
        values = [self.gamma_scale / self.output_scale * 2**MULTIPLIER_FRACTION,
                  self.beta_scale / self.output_scale * 2**MULTIPLIER_FRACTION]
        if any(not np.isfinite(value) or value < .5 for value in values):
            raise ValueError('Scale ratios exceed supported multiplier precision')
        return tuple(int(round(value)) for value in values)

    def metadata(self, eps=1e-5):
        n = MAX_CHANNELS
        e = epsilon_code(eps, n, self)
        d_bound = n*n*128*128
        gm, bm = self.multipliers()
        norm_abs = 2 * (isqrt((n-1) << (2*NORM_FRACTION)) + 1)
        affine_bound = norm_abs*128*gm + (128*bm << NORM_FRACTION)
        return {**asdict(self), 'input_bits': 8, 'output_bits': 8,
                'gamma_bits': 8, 'beta_bits': 8, 'zero_points': 0,
                'eps': eps, 'max_channels': n,
                'norm_fraction': NORM_FRACTION, 'root_fraction': ROOT_FRACTION,
                'multiplier_fraction': MULTIPLIER_FRACTION,
                'gamma_multiplier': gm, 'beta_multiplier': bm,
                'sum_signed_bits': (n*128).bit_length()+1,
                'sum_squares_unsigned_bits': (n*128*128).bit_length(),
                'variance_numerator_unsigned_bits': d_bound.bit_length(),
                'root_argument_unsigned_bits': ((d_bound+e) << (2*ROOT_FRACTION)).bit_length(),
                'normalize_numerator_signed_bits': ((2*n*128) << (ROOT_FRACTION+NORM_FRACTION)).bit_length()+1,
                'affine_accumulator_signed_bits': affine_bound.bit_length()+1,
                'rounding': 'nearest ties to even; signed INT8 saturation',
                'sqrt': 'floor integer square root',
                'epsilon_quantization': 'max(1, round(eps*N*N/input_scale**2))'}


def epsilon_code(eps, n, config):
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    scaled = eps * n*n / config.input_scale / config.input_scale
    if not np.isfinite(scaled):
        raise ValueError('Epsilon exceeds supported configuration range')
    return max(1, int(round(scaled)))


def quantize_int8(values, scale):
    """Same rounding, zero point and saturation as activation/gelu_lut.py."""
    validate_scale(scale)
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Expected finite values')
    with np.errstate(over='ignore'):
        rounded = np.rint(values / scale)
    clipped = int(np.count_nonzero((rounded < -128) | (rounded > 127)))
    return np.clip(rounded, -128, 127).astype(np.int8), clipped


def checked_codes(values, name):
    values = np.asarray(values)
    if values.dtype.kind not in 'iu' or np.any(values < -128) or np.any(values > 127):
        raise ValueError(f'{name} must be signed INT8 integer codes in [-128,127]')
    return values.astype(np.int8)


def layernorm_codes(values, gamma, beta, config=Config(), eps=1e-5):
    q = checked_codes(values, 'input')
    if q.ndim == 0 or q.size == 0 or not 1 <= q.shape[-1] <= MAX_CHANNELS:
        raise ValueError('Expected nonempty rows with 1..576 channels')
    n = q.shape[-1]
    g, b = checked_codes(gamma, 'gamma'), checked_codes(beta, 'beta')
    if g.shape not in ((), (n,)) or b.shape not in ((), (n,)):
        raise ValueError('Affine codes must be scalar or a channel vector')
    g, b = np.broadcast_to(g, (n,)), np.broadcast_to(b, (n,))
    e = epsilon_code(eps, n, config)
    gm, bm = config.multipliers()
    output = np.empty_like(q, dtype=np.int8)
    saturated, min_d, max_d = 0, None, 0
    for row, dst in zip(q.reshape(-1, n), output.reshape(-1, n)):
        row = [int(v) for v in row]
        total = sum(row)
        d = n * sum(v*v for v in row) - total*total
        min_d = d if min_d is None else min(min_d, d)
        max_d = max(max_d, d)
        denominator = isqrt((d+e) << (2*ROOT_FRACTION))
        for i, v in enumerate(row):
            z = round_divide((n*v-total) << (ROOT_FRACTION+NORM_FRACTION), denominator)
            affine = z*int(g[i])*gm + (int(b[i])*bm << NORM_FRACTION)
            code = round_divide(affine, 1 << (NORM_FRACTION+MULTIPLIER_FRACTION))
            saturated += int(code < -128 or code > 127)
            dst[i] = min(127, max(-128, code))
    return output, {'output_saturated_count': saturated,
                    'variance_numerator_min': min_d, 'variance_numerator_max': max_d}


def layernorm_fixed(values, gamma=1., beta=0., config=Config(), eps=1e-5):
    x = validate_values(values)
    g = affine_parameter(gamma, x.shape[-1], 'gamma')
    b = affine_parameter(beta, x.shape[-1], 'beta')
    q, xc = quantize_int8(x, config.input_scale)
    gq, gc = quantize_int8(g, config.gamma_scale)
    bq, bc = quantize_int8(b, config.beta_scale)
    codes, stats = layernorm_codes(q, gq, bq, config, eps)
    stats.update(input_clipped_count=xc, gamma_clipped_count=gc, beta_clipped_count=bc)
    return codes.astype(np.float64)*config.output_scale, codes, stats
