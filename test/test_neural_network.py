import unittest
import numpy as np
from numpy.testing import assert_array_equal
import os.path
from os import path
import sys
sys.path.append("../lib")
from neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nw = NeuralNetwork()

    def test_step_func(self):
        x = np.arange(-5.0, 5.0, 0.1)
        y = self.nw.step_func(x)
        assert_array_equal(np.array(
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ]
        ), y)
        self.assertEqual(True, path.exists("../img/step_func.png"))

if __name__ == "__main__":
    unittest.main()
