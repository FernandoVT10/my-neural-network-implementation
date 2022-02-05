import numpy as np

from .Activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def d_sigmoid(x):
            return x * (1 - x)

        super().__init__(sigmoid, d_sigmoid)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.d_activation(self.output)
