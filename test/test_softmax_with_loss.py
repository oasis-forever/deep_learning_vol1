import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from softmax_with_loss import SoftmaxWithLoss

class TestSoftmaxWithLoss(unittest.TestCase):
    def setUp(self):
        self.swl = SoftmaxWithLoss()

    def test_forward(self):
        x = np.array([0.3, 0.6, 0.9])
        t = np.array([0, 0, 1])
        loss = self.swl.forward(x, t)
        self.assertEqual(0.8283899409431649, loss)

    def test_backward(self):
        x = np.array([0.3, 0.6, 0.9])
        t = np.array([0, 0, 1])
        self.swl.forward(x, t)
        dx = self.swl.backward()
        assert_almost_equal(np.array([0.07989816,  0.10785123, -0.18774939]), dx)

if __name__ == "__main__":
    unittest.main()
