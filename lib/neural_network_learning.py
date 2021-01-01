import numpy as np
class NeuralNetworkLearning:
    def __init__(self):
        pass

    def mean_squared_error(self, y, t):
        return 0.5 * np.sum((y - t) ** 2)

