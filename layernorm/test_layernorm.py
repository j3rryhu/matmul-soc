import json
from pathlib import Path
import unittest
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext

import numpy as np

from layernorm_reference import layernorm_reference
from layernorm_fixed import (Config, MAX_CHANNELS, NORM_FRACTION, ROOT_FRACTION,
                             MULTIPLIER_FRACTION, layernorm_codes, layernorm_fixed,
                             quantize_int8, round_divide)

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


class Int8Tests(unittest.TestCase):
    def test_gelu_scale_byte_encoding_and_quantization(self):
        metadata = json.loads((ROOT/'activation/example/gelu_metadata.json').read_text())
        c = Config()
        self.assertEqual(c.input_scale, metadata['input_scale'])
        self.assertEqual(c.output_scale, metadata['output_scale'])
        self.assertEqual(metadata['input_zero_point'], 0)
        codes = np.arange(-128, 128)
        q, clipped = quantize_int8(codes*c.input_scale, c.input_scale)
        np.testing.assert_array_equal(q, codes)
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(clipped, 0)
        np.testing.assert_array_equal(q.view(np.uint8), codes & 255)
        values = np.array([-1e300, 1e300, .5, 1.5, 2.5, -.5, -1.5, -2.5])*c.input_scale
        q, clipped = quantize_int8(values, c.input_scale)
        np.testing.assert_array_equal(q, [-128, 127, 0, 2, 2, 0, -2, -2])
        self.assertEqual(clipped, 2)
        for numerator in range(-17, 18):
            for denominator in range(1, 9):
                self.assertEqual(round_divide(numerator, denominator), round(numerator/denominator))

    def test_reference_population_variance_and_axis(self):
        x = np.array([[[1., 3.], [10., 14.]]])
        centered = np.array([[[-1., 1.], [-2., 2.]]])
        expected = centered/np.sqrt(np.mean(centered**2, axis=-1, keepdims=True)+1.)
        np.testing.assert_allclose(layernorm_reference(x, [2., 3.], [4., 5.], 1.), expected*[2., 3.]+[4., 5.])

    @unittest.skipIf(torch is None, 'Optional PyTorch oracle unavailable')
    def test_pytorch_oracle_on_int8_dequantized_inputs(self):
        rng = np.random.default_rng(3)
        c = Config()
        for n in (1, 2, 192, 384, 576):
            x = rng.integers(-128, 128, (2, 3, n))*c.input_scale
            g = rng.integers(-32, 33, n)*c.gamma_scale
            b = rng.integers(-32, 33, n)*c.beta_scale
            expected = torch.nn.functional.layer_norm(torch.from_numpy(x), (n,), torch.from_numpy(g), torch.from_numpy(b), 1e-5)
            np.testing.assert_allclose(layernorm_reference(x, g, b), expected.numpy(), rtol=1e-11, atol=1e-11)

    def test_integer_kernel_against_decimal_oracle(self):
        rng = np.random.default_rng(73)
        # Test same-scale contract and an explicit change in real-value scales,
        # always storing all operands and results as signed INT8.
        for c in (Config(), Config(input_scale=1/16, output_scale=1/64,
                                   gamma_scale=1/64, beta_scale=1/32)):
            gm, bm = c.multipliers()
            with localcontext() as context:
                context.prec = 100
                for n in (1, 2, 192, 384, 576):
                    q = rng.integers(-128, 128, (4, n))
                    q[1] = 127
                    q[2] = 0
                    q[2, 0] = 1
                    q[3, ::2], q[3, 1::2] = -128, 127
                    g, b = rng.integers(-128, 128, (2, n))
                    actual, _ = layernorm_codes(q, g, b, c)
                    expected = []
                    for row in q:
                        total = sum(map(int, row))
                        d = n*sum(int(v)**2 for v in row)-total**2
                        e = max(1, round(1e-5*n*n/c.input_scale/c.input_scale))
                        root = Decimal((d+e)*2**(2*ROOT_FRACTION)).sqrt().to_integral_value(rounding=ROUND_FLOOR)
                        z = [(Decimal((n*int(v)-total)*2**(ROOT_FRACTION+NORM_FRACTION))/root).to_integral_value(rounding=ROUND_HALF_EVEN) for v in row]
                        codes = [int(((v*int(gi)*gm + int(bi)*bm*2**NORM_FRACTION)/Decimal(2**(NORM_FRACTION+MULTIPLIER_FRACTION))).to_integral_value(rounding=ROUND_HALF_EVEN)) for v, gi, bi in zip(z, g, b)]
                        expected.append(np.clip(codes, -128, 127))
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.dtype, np.int8)

    def test_int32_matmul_requantized_boundary_and_gelu_bytes(self):
        c = Config()
        # output_loader.v: arithmetic right shift, then signed saturation.
        psums = np.array([-2**31, -1000, -257, -256, -3, -1, 0, 1, 3, 254, 255, 256, 1000, 2**31-1], dtype=np.int64)
        matmul = np.clip((psums*32768) >> 16, -128, 127).astype(np.int8)
        np.testing.assert_array_equal(matmul, [-128, -128, -128, -128, -2, -1, 0, 0, 1, 127, 127, 127, 127, 127])
        lut = np.array([int(line, 16) for line in (ROOT/'activation/example/gelu_int8.hex').read_text().splitlines()], dtype=np.uint8).view(np.int8)
        gelu = lut[matmul.astype(np.int64) & 255]
        for upstream in (matmul, gelu):
            for n in (192, 384, 576):
                row = np.resize(upstream, n)
                actual, _ = layernorm_codes(row, 32, 0, c)
                reference = layernorm_reference(row.astype(float)*c.input_scale)
                expected, _ = quantize_int8(reference, c.output_scale)
                # Integer sqrt/epsilon/normalization rounding may cross a code tie.
                self.assertLessEqual(np.abs(actual.astype(int)-expected.astype(int)).max(), 1)

    def test_constants_near_zero_variance_saturation_and_permutation(self):
        rng = np.random.default_rng(4)
        for n in (1, 192, 384, 576):
            beta = rng.integers(-128, 128, n)
            codes, _ = layernorm_codes(np.full((2, n), 127), 32, beta)
            np.testing.assert_array_equal(codes, np.tile(beta, (2, 1)))
            q = rng.integers(-128, 128, (2, n))
            q[0] = 0
            q[0, 0] = 1
            codes, _ = layernorm_codes(q, 32, beta)
            perm = rng.permutation(n)
            permuted, _ = layernorm_codes(q[:, perm], 32, beta[perm])
            np.testing.assert_array_equal(permuted, codes[:, perm])
        codes, stats = layernorm_codes([-128, 127], 127, 127)
        self.assertEqual(codes[1], 127)
        self.assertEqual(stats['output_saturated_count'], 1)

    def test_default_intermediate_bounds(self):
        c = Config()
        row = np.tile([-128, 127], MAX_CHANNELS//2)
        codes, stats = layernorm_codes(row, 32, 0)
        np.testing.assert_array_equal(codes, np.tile([-32, 32], MAX_CHANNELS//2))
        m = c.metadata()
        self.assertLess(stats['variance_numerator_max'], 2**m['variance_numerator_unsigned_bits'])
        self.assertEqual(m['sum_signed_bits'], 18)
        self.assertEqual(m['sum_squares_unsigned_bits'], 24)
        self.assertEqual(m['variance_numerator_unsigned_bits'], 33)
        self.assertEqual(m['root_argument_unsigned_bits'], 49)
        self.assertLessEqual(m['affine_accumulator_signed_bits'], 64)

    def test_calibration_search_matches_scalar_integer_kernel(self):
        from calibrate_int8 import affine_codes, normalized_codes
        rng = np.random.default_rng(901)
        for c in (Config(), Config(input_scale=.17, output_scale=.013,
                                   gamma_scale=.0103, beta_scale=.0128)):
            for n in (1, 2, 192, 384, 576):
                q = rng.integers(-128, 128, (4, n), dtype=np.int64)
                q[1] = 127
                q[2] = 0
                q[2, 0] = 1
                q[3, ::2], q[3, 1::2] = -128, 127
                g, b = rng.integers(-128, 128, (2, n), dtype=np.int64)
                actual = affine_codes(normalized_codes(q, c, 1e-5), g, b, c, 1e-5)
                expected, _ = layernorm_codes(q, g, b, c, 1e-5)
                np.testing.assert_array_equal(actual, expected)

    def test_wrapper_and_invalid_contract(self):
        c = Config()
        x = np.array([[-1., 0., 1.]])
        y, codes, _ = layernorm_fixed(x)
        self.assertEqual(codes.dtype, np.int8)
        np.testing.assert_array_equal(y, codes.astype(float)*c.output_scale)
        for q in ([1., 2.], [128], [-129], [], np.zeros(577, dtype=int), [2**64-1]):
            with self.assertRaises(ValueError):
                layernorm_codes(q, 32, 0)
        for param in (128, [1, 2, 3]):
            with self.assertRaises(ValueError):
                layernorm_codes([1, 2], param, 0)
        for scale in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                Config(input_scale=scale)
        for eps in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                layernorm_codes([1, 2], 32, 0, eps=eps)
        for x in (1., [], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                layernorm_fixed(x)


if __name__ == '__main__':
    unittest.main()
