import numpy as np
import os, sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

class NeuralNetworkLearning:
    def __init__(self):
        pass

    def _get_train_data(self):
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, one_hot_label=True)
        return x_train, t_train

    def mean_squared_error(self, y, t):
        return 0.5 * np.sum((y - t) ** 2)

    def cross_entropy_error(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))
