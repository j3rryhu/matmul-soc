import unittest
import numpy as np
from softmax_fixed import Config, make_lut, quantize_scores, softmax_codes, softmax_fixed


class FixedTests(unittest.TestCase):
    def test_integer_division_against_scalar_oracle(self):
        config = Config(fractional_bits=4, cutoff=8)
        lut = make_lut(config)
        rng = np.random.default_rng(15)
        for n in (1, 49, 196):
            q = rng.integers(-200, 201, size=(5, n), dtype=np.int64)
            actual, _ = softmax_codes(q, config)
            for row, output in zip(q, actual):
                weights = [int(lut[int(max(row)-x)]) if max(row)-x < len(lut) else 0 for x in row]
                total = sum(weights)
                expected = [(w * 65535 + total//2)//total for w in weights]
                np.testing.assert_array_equal(output, expected)

    def test_boundaries_and_saturation(self):
        config = Config()
        q, clipped = quantize_scores([[-1e6, 1e6, .5/256, 1.5/256]], config)
        np.testing.assert_array_equal(q, [[-32768, 32767, 0, 2]])
        self.assertEqual(clipped, 2)
        _, codes, stats = softmax_fixed([[-1e6, 1e6]], config)
        np.testing.assert_array_equal(codes, [[0, 65535]])
        self.assertEqual(stats["input_clipped_count"], 2)

    def test_equal_rows_and_rom(self):
        config = Config()
        self.assertEqual(config.metadata()["rounded_numerator_bits"], 33)
        lut = make_lut(config)
        self.assertEqual(lut[0], 65535)
        self.assertTrue(np.all(np.diff(lut) <= 0))
        for n in (1, 49, 196):
            p, _, _ = softmax_fixed(np.zeros((2, n)), config)
            self.assertLessEqual(np.abs(p.sum(-1)-1).max(), n/(2*65535))
        with self.assertRaises(ValueError):
            softmax_codes(np.zeros((1, 197), dtype=int), config)


if __name__ == "__main__":
    unittest.main()
