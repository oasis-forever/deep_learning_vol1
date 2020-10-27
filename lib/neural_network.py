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
        self._save_image(x, y, sys._getframe().f_code.co_name)
        return y

    def sigmoid(self, x):
        y = 1 / (1 + np.exp(-x))
        self._save_image(x, y, sys._getframe().f_code.co_name)
        return y
