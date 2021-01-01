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

    def _random_choice(self):
        x_train, t_train = self._get_train_data()
        train_size = x_train.shape[0]
        batch_size = 10
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        return x_batch, t_batch

    def mean_squared_error(self, y, t):
        return 0.5 * np.sum((y - t) ** 2)

    def cross_entropy_error(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))
