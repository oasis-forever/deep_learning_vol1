import sys
import numpy as np
import matplotlib.pylab as plt

class NeuralNetwork:
    def __init__(self):
        pass

    def _save_image(self, x, y, func_name):
        plt.plot(x, y)
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig("../img/{}.png".format(func_name))

    def step_func(self, x):
        y =  np.array(x > 0, dtype=np.int)
        # self._save_image(x, y, sys._getframe().f_code.co_name)
        return y

    def sigmoid(self, x):
        y = 1 / (1 + np.exp(-x))
        # self._save_image(x, y, sys._getframe().f_code.co_name)
        return y

    def relu(self, x):
        y = np.maximum(0, x)
        # self._save_image(x, y, sys._getframe().f_code.co_name)
        return y

    def matrix_product(self, a, b):
        return np.dot(a, b)

    def init_network(self):
        network = {}
        network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        network["b1"]  = np.array([0.1, 0.2, 0.3])
        network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        network["b2"]  = np.array([0.1, 0.2])
        network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
        network["b3"]  = np.array([0.1, 0.2])
        return network

    def identify_function(self, x):
        return x

    def forward(self, network, x):
        W1, W2, W3 = network["W1"], network["W2"], network["W3"]
        b1, b2, b3 = network["b1"], network["b2"], network["b3"]
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y  = self.identify_function(a3)
        return y

    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y
