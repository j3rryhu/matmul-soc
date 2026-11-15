import unittest
from decimal import Decimal, ROUND_HALF_EVEN
import numpy as np
from softmax_fixed import Config, make_lut, quantize_scores, softmax_codes, softmax_fixed


class FixedTests(unittest.TestCase):
    def test_integer_normalization_against_decimal_oracle(self):
        config = Config(score_scale=.137)
        lut = make_lut(config)
        rng = np.random.default_rng(15)
        for n in (1, 2, 49, 196):
            q = rng.integers(-128, 128, size=(5, n), dtype=np.int64)
            actual, _ = softmax_codes(q, config)
            self.assertEqual(actual.dtype, np.int8)
            for row, output in zip(q, actual):
                weights = [int(lut[int(max(row)-x)]) for x in row]
                total = sum(weights)
                expected = [int((Decimal(w*127)/Decimal(total)).to_integral_value(rounding=ROUND_HALF_EVEN)) for w in weights]
                np.testing.assert_array_equal(output, expected)

    def test_int8_quantization_and_matmul_gelu_boundary(self):
        config = Config()
        q, clipped = quantize_scores([[-1e300, 1e300, .5/8, 1.5/8, -.5/8, -1.5/8]], config)
        np.testing.assert_array_equal(q, [[-128, 127, 0, 2, 0, -2]])
        self.assertEqual(clipped, 2)
        all_codes = np.arange(-128,128)
        q, clipped = quantize_scores(all_codes*config.score_scale, config)
        np.testing.assert_array_equal(q, all_codes)
        self.assertEqual(clipped, 0)
        np.testing.assert_array_equal(q.view(np.uint8), all_codes & 255)
        # Same output_loader arithmetic-shift + signed saturation byte contract.
        psums = np.array([-2**31,-257,-3,-1,0,1,3,254,256,2**31-1], dtype=np.int64)
        upstream = np.clip((psums*32768)>>16,-128,127).astype(np.int8)
        p, codes, _ = softmax_fixed(upstream.astype(float)*config.score_scale, config)
        expected, _ = softmax_codes(upstream, config)
        np.testing.assert_array_equal(codes, expected)
        np.testing.assert_array_equal(p, codes.astype(float)/127)

    def test_equal_rows_ties_and_probability_mass_limit(self):
        config = Config()
        for n, code in ((1,127),(2,64),(49,3),(196,1)):
            p, codes, _ = softmax_fixed(np.zeros((2,n)), config)
            np.testing.assert_array_equal(codes, np.full((2,n),code))
            self.assertLessEqual(float(np.abs(p.sum(-1)-1).max()),n/(2*127))
        # Explicitly expose the unavoidable coarse resolution of signed INT8.
        p, _, _ = softmax_fixed(np.zeros(196), config)
        self.assertAlmostEqual(float(p.sum()),196/127)

    def test_full_extrema_and_permutation(self):
        c = Config()
        _, codes, _ = softmax_fixed([[-1e6,1e6]],c)
        np.testing.assert_array_equal(codes,[[0,127]])
        rng = np.random.default_rng(99)
        q = rng.integers(-128,128,(3,196))
        perm = rng.permutation(196)
        actual, _ = softmax_codes(q,c)
        shuffled, _ = softmax_codes(q[:,perm],c)
        np.testing.assert_array_equal(shuffled,actual[:,perm])
        self.assertGreaterEqual(actual.min(),0)
        self.assertLessEqual(actual.max(),127)

    def test_rom_and_intermediate_widths(self):
        lut = make_lut(Config())
        self.assertEqual(len(lut),256)
        self.assertEqual(lut[0],65535)
        self.assertTrue(np.all(np.diff(lut)<=0))
        m=Config().metadata()
        self.assertEqual(m['sum_bits'],24)
        self.assertEqual(m['numerator_bits'],23)
        self.assertEqual(m['difference_bits'],9)

    def test_invalid_contract(self):
        for value in ([1.,2.], [128], [-129], [], np.zeros(197,dtype=int), [2**64-1]):
            with self.assertRaises(ValueError):
                softmax_codes(value,Config())
        for scale in (0,-1,np.nan,np.inf):
            with self.assertRaises(ValueError):
                Config(score_scale=scale)
        for scores in ([np.nan],[np.inf],1,np.empty((0,49))):
            with self.assertRaises(ValueError):
                quantize_scores(scores,Config())
        with self.assertRaises(ValueError):
            softmax_codes([0,1],Config(),np.zeros(256,dtype=int))


if __name__ == '__main__':
    unittest.main()
