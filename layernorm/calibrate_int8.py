"""Select INT8 scales on calibration images, then evaluate fixed held-out images.

All storage remains INT8. The vectorized search is checked against the scalar
integer kernel for every final calibration/validation output.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from math import isqrt
from pathlib import Path

import numpy as np

from layernorm_fixed import (Config, NORM_FRACTION, ROOT_FRACTION,
    MULTIPLIER_FRACTION, checked_codes, epsilon_code, layernorm_fixed, quantize_int8)


def divide_even(numerator, denominator):
    absolute = np.abs(numerator)
    quotient, remainder = np.divmod(absolute, denominator)
    quotient += ((2*remainder > denominator) |
                 ((2*remainder == denominator) & ((quotient & 1) != 0)))
    return np.where(numerator < 0, -quotient, quotient)


def normalized_codes(q, config, eps):
    """Exact vectorized version of the existing integer normalization stage."""
    metadata = config.metadata(eps)
    for field in ('normalize_numerator_signed_bits', 'root_argument_unsigned_bits',
                  'variance_numerator_unsigned_bits', 'sum_squares_unsigned_bits'):
        if metadata[field] >= 63:
            raise ValueError('Search intermediate would exceed signed int64')
    q = checked_codes(q, 'input').astype(np.int64)
    n = q.shape[-1]
    total = q.sum(axis=-1, keepdims=True)
    d = n*(q*q).sum(axis=-1, keepdims=True)-total*total
    argument = (d+epsilon_code(eps, n, config)) << (2*ROOT_FRACTION)
    root = np.fromiter((isqrt(int(v)) for v in argument.flat), dtype=np.int64).reshape(argument.shape)
    return divide_even((n*q-total) << (ROOT_FRACTION+NORM_FRACTION), root)


def affine_codes(z, g, b, config, eps):
    if config.metadata(eps)['affine_accumulator_signed_bits'] >= 63:
        raise ValueError('Search affine intermediate would exceed signed int64')
    g = checked_codes(g, 'gamma').astype(np.int64)
    b = checked_codes(b, 'beta').astype(np.int64)
    gm, bm = config.multipliers()
    numerator = z*g*gm + ((b*bm) << NORM_FRACTION)
    unbounded = divide_even(numerator, 1 << (NORM_FRACTION+MULTIPLIER_FRACTION))
    return np.clip(unbounded, -128, 127).astype(np.int8)


def scales_for(values, powers, percentiles):
    candidates = [2.**exponent for exponent in powers]
    candidates += [float(np.percentile(np.abs(values), p))/127 for p in percentiles]
    return sorted(set(value for value in candidates if value > 0))


def parameter_scale(values):
    candidates = scales_for(values, range(-16, 0), (95, 99, 99.5, 100))
    def key(scale):
        q, _ = quantize_int8(values, scale)
        return float(np.mean((q.astype(float)*scale-values)**2)), scale
    return min(candidates, key=key)


def evaluate(records, config, eps, out=None):
    squared, reference_squared, count, maximum = 0., 0., 0, 0.
    counts = {key: 0 for key in ('input_clipped_count', 'gamma_clipped_count',
                                'beta_clipped_count', 'output_saturated_count')}
    per_image = []
    for record in records:
        x, g, b, ref = (record[key] for key in ('inputs', 'gamma', 'beta', 'reference_output'))
        actual, codes, stats = layernorm_fixed(x, g, b, config, eps)
        q, _ = quantize_int8(x, config.input_scale)
        gq, _ = quantize_int8(g, config.gamma_scale)
        bq, _ = quantize_int8(b, config.beta_scale)
        fast = affine_codes(normalized_codes(q, config, eps), gq, bq, config, eps)
        np.testing.assert_array_equal(codes, fast)
        delta = actual-ref
        squared += float(np.sum(delta**2))
        reference_squared += float(np.sum(ref**2))
        count += ref.size
        maximum = max(maximum, float(np.abs(delta).max()))
        for key in counts:
            counts[key] += stats[key]
        per_image.append({'image': record['image'], 'image_sha256': record['image_sha256'],
                          'rmse': float(np.sqrt(np.mean(delta**2))), 'stats': stats})
        if out is not None:
            np.savez_compressed(out/f'{record["index"]:03d}_int8.npz', input_codes=q,
                gamma_codes=gq, beta_codes=bq, output_codes=codes, fixed_output=actual)
    return {'rmse': float(np.sqrt(squared/count)), 'max_absolute_error': maximum,
            'relative_rmse_percent': float(100*np.sqrt(squared/reference_squared)),
            'scalar_count': count, 'input_clipped_percent': 100*counts['input_clipped_count']/count,
            'counts': counts, 'images': per_image, 'fast_scalar_bit_exact': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=Path(__file__).resolve().parent/'runs/int8_21m')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'runs/calibration_21m')
    args = parser.parse_args()
    report_path = args.capture/'report.json'
    source = json.loads(report_path.read_text(encoding='utf-8'))
    eps = source['config']['eps']
    records = []
    for index, image in enumerate(sorted(source['images'], key=lambda item: item['image'])):
        with np.load(args.capture/image['artifact'], allow_pickle=False) as data:
            record = {key: data[key].copy() for key in ('inputs', 'gamma', 'beta', 'reference_output')}
        record.update(index=index, image=image['image'], image_sha256=image['sha256'])
        records.append(record)
    if len(records) < 3 or len({r['image_sha256'] for r in records}) != len(records):
        raise ValueError('Need at least 3 unique images for a disjoint split')
    split = min(len(records)-1, max(1, int(len(records)*.7)))
    calibration, validation = records[:split], records[split:]
    x = np.concatenate([r['inputs'].reshape(-1, r['inputs'].shape[-1]) for r in calibration])
    ref = np.concatenate([r['reference_output'].reshape(-1, r['reference_output'].shape[-1]) for r in calibration])
    g, b = calibration[0]['gamma'], calibration[0]['beta']
    if any(not np.array_equal(r['gamma'], g) or not np.array_equal(r['beta'], b) for r in records):
        raise ValueError('Expected one shared LayerNorm parameter set')
    gs, bs = parameter_scale(g), parameter_scale(b)
    gq, _ = quantize_int8(g, gs)
    bq, _ = quantize_int8(b, bs)
    inputs = scales_for(x, range(-5, 0), (95, 99, 99.5, 99.9, 99.95, 100))
    outputs = scales_for(ref, range(-8, -3), (95, 99, 99.5, 99.9, 99.95, 100))
    candidates = []
    for sx in inputs:
        q, clipped = quantize_int8(x, sx)
        z = normalized_codes(q, Config(input_scale=sx), eps)
        for sy in outputs:
            config = Config(input_scale=sx, output_scale=sy, gamma_scale=gs, beta_scale=bs)
            codes = affine_codes(z, gq, bq, config, eps)
            rmse = float(np.sqrt(np.mean((codes.astype(float)*sy-ref)**2)))
            candidates.append({'scales': asdict(config), 'calibration_rmse': rmse,
                               'calibration_input_clipped_count': clipped})
        print(f'Calibration input scale {sx:.8g}: evaluated {len(outputs)} output scales', flush=True)
    selected = min(candidates, key=lambda item: (item['calibration_rmse'], item['scales']['input_scale'], item['scales']['output_scale']))
    config = Config(**selected['scales'])  # Freeze selection BEFORE validation.
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'selected_config.json').write_text(json.dumps(asdict(config), indent=2)+'\n', encoding='utf-8')
    final = {'target': source['fixed_target'], 'source_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'checkpoint_sha256': source['checkpoint_sha256'], 'eps': eps,
        'selection': 'gamma/beta minimize parameter MSE; input/output minimize integer LayerNorm calibration RMSE within stated grid',
        'split': {'calibration': [r['image'] for r in calibration], 'validation': [r['image'] for r in validation]},
        'selected_config': config.metadata(eps), 'candidates': candidates,
        'selected': {'calibration': evaluate(calibration, config, eps, args.out),
                     'validation': evaluate(validation, config, eps, args.out)},
        'baseline': {'calibration': evaluate(calibration, Config(), eps),
                     'validation': evaluate(validation, Config(), eps)},
        'limitations': ['One LayerNorm site; all operands/output storage remains INT8',
            'Validation excluded from this scale search but these same local images were inspected in earlier experiments',
            'Small unlabeled set; no classification accuracy claim or guarantee for other sites',
            'Cached floating model boundary inputs, not actual upstream FPGA captures',
            'Fixed-point square root/division baseline; no RTL or hardware approximation validation']}
    (args.out/'report.json').write_text(json.dumps(final, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'selected_scales': asdict(config),
        'calibration_relative_rmse_percent': final['selected']['calibration']['relative_rmse_percent'],
        'validation_relative_rmse_percent': final['selected']['validation']['relative_rmse_percent'],
        'validation_input_clipped_percent': final['selected']['validation']['input_clipped_percent']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
