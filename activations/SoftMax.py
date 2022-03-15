import numpy as np

from .Activation import Activation

class SoftMax(Activation):
    def __init__(self):
        def activation(x):
            tmp = np.exp(x)
            return tmp / np.sum(tmp)

        def d_activation(x):
            return x * (1 - x)

        super().__init__(activation, d_activation)

    def backward(self, output_gradient, learning_rate):
        #  return output_gradient

        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient.T).T
