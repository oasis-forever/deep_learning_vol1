import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append("../dataset")
sys.path.append("../lib")
sys.path.append("../lib/concerns")
from mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = net.numerical_gradient(x_batch, t_batch)
grad_backprop  = net.gradient(x_batch, t_batch)

# Calculate absolute diff of each weight
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print("{}: {}".format(key, str(diff)))
