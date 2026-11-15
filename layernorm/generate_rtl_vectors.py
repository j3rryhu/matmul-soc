"""Export calibrated integer contract and standalone RTL verification vectors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from layernorm_fixed import Config, epsilon_code, layernorm_codes


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=root/'runs/calibrated_21m')
    parser.add_argument('--out', type=Path, default=root/'example')
    args = parser.parse_args()
    config_path = root/'example/int8_config.json'
    config = Config(**json.loads(config_path.read_text()))
    if config.multipliers() != (45764, 55988):
        raise ValueError('RTL affine constants must be updated for this config')
    eps = 1e-5
    table = [0]+[epsilon_code(eps, n, config) for n in range(1, 577)]+[0]*447
    if max(table) > 255:
        raise ValueError('RTL epsilon ROM entry exceeds 8 bits')
    args.out.mkdir(parents=True, exist_ok=True)
    epsilon_text = ''.join(f'{e:02x}\n' for e in table)
    (args.out/'epsilon.hex').write_text(epsilon_text, encoding='ascii')
    rng = np.random.default_rng(20261114)
    rows = []
    def add(q, g, b, source):
        q = np.asarray(q, dtype=np.int8)
        g = np.broadcast_to(np.asarray(g, dtype=np.int8), q.shape).copy()
        b = np.broadcast_to(np.asarray(b, dtype=np.int8), q.shape).copy()
        expected, stats = layernorm_codes(q, g, b, config, eps)
        rows.append((q, g, b, expected, source, stats['output_saturated_count']))
    identity = int(round(1/config.gamma_scale))
    for n in (1, 2, 3, 7, 31, 191, 192, 383, 384, 575, 576):
        for value in (-128, 0, 127):
            add(np.full(n, value), rng.integers(-128,128,n), rng.integers(-128,128,n), 'constant')
        add(np.resize([-128,127], n), identity, 0, 'full_scale')
        add([127]+[-128]*(n-1), 127, 127, 'positive_outlier_saturation')
        add([-128]+[127]*(n-1), -128, -128, 'negative_affine_saturation')
        add([1]+[0]*(n-1), identity, 0, 'near_zero_variance')
        add(rng.integers(-128,128,n), rng.integers(-128,128,n), rng.integers(-128,128,n), 'random_affine')
    # Every legal length exercises the generated epsilon ROM address and N logic.
    for n in range(1, 577):
        add(rng.integers(-128,128,n), identity, 0, 'all_lengths')
    source_path = args.capture/'report.json'
    capture = json.loads(source_path.read_text())
    for key in config.__dataclass_fields__:
        if capture['config'][key] != getattr(config, key):
            raise ValueError('Captured input/affine/output scales do not match RTL')
    real_rows = 0
    for image in capture['images']:
        with np.load(args.capture/image['artifact'], allow_pickle=False) as data:
            n = data['input_codes'].shape[-1]
            q = data['input_codes'].reshape(-1,n)
            expected = data['output_codes'].reshape(-1,n)
            for i in rng.choice(len(q), min(8,len(q)), replace=False):
                add(q[i], data['gamma_codes'], data['beta_codes'], image['image'])
                np.testing.assert_array_equal(rows[-1][3], expected[i])
                real_rows += 1
    with (args.out/'rows.txt').open('w', encoding='ascii') as stream:
        stream.write(f'{len(rows)} {sum(len(row[0]) for row in rows)}\n')
        for q, g, b, expected, _, _ in rows:
            stream.write(f'{len(q)}\n')
            for values in zip(q, g, b, expected):
                stream.write(' '.join(str(int(v)) for v in values)+'\n')
    manifest = {'config': config.metadata(eps), 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'source_report_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'epsilon_sha256': hashlib.sha256((args.out/'epsilon.hex').read_bytes()).hexdigest(),
        'vectors_sha256': hashlib.sha256((args.out/'rows.txt').read_bytes()).hexdigest(),
        'seed': 20261114, 'rows': len(rows), 'elements': sum(len(row[0]) for row in rows),
        'real_rows': real_rows, 'synthetic_rows': len(rows)-real_rows,
        'saturated_outputs': sum(row[5] for row in rows), 'epsilon_entries': len(table),
        'scope': 'All 576 legal lengths, extrema/affine/constant/near-zero cases and 8 actual rows per image; no tuning'}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
