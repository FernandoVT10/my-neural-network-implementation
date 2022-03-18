import numpy as np

from layers import Layer

class SoftMax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        #  return output_gradient

        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient.T).T
