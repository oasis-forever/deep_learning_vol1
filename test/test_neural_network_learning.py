import unittest
import numpy as np
import sys
sys.path.append("../lib")
from neural_network_learning import NeuralNetworkLearning

class TestNeuralNetworkLearning(unittest.TestCase):
    def setUp(self):
        self.nwl = NeuralNetworkLearning()

    def test_mean_squared_error_1(self):
        t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
        result = self.nwl.mean_squared_error(np.array(y), np.array(t))
        self.assertEqual(0.09750000000000003, result)

    def test_mean_squared_error_2(self):
        t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
        result = self.nwl.mean_squared_error(np.array(y), np.array(t))
        self.assertEqual(0.5975, result)

if __name__ == "__main__":
    unittest.main()
