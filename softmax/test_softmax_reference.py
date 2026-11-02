import unittest
import numpy as np
from softmax_reference import softmax_reference


class SoftmaxTests(unittest.TestCase):
    def test_equal_and_extreme_rows(self):
        for length in (1, 49, 196):
            np.testing.assert_allclose(softmax_reference(np.full((2, length), 10000.)), 1 / length)
        np.testing.assert_array_equal(softmax_reference([[10000., -10000.]]), [[1., 0.]])

    def test_shift_invariance_and_last_axis(self):
        scores = np.arange(24).reshape(2, 3, 4)
        p = softmax_reference(scores)
        np.testing.assert_allclose(p, softmax_reference(scores + 10000))
        np.testing.assert_allclose(p.sum(axis=-1), 1)
        self.assertTrue(np.all(p >= 0))

    def test_invalid_input(self):
        for scores in ([], 1, [[float("nan")]], [[float("inf")]]):
            with self.assertRaises(ValueError):
                softmax_reference(scores)


if __name__ == "__main__":
    unittest.main()
