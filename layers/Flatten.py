import numpy as np

from .Layer import Layer

class Flatten(Layer):
    def prepare(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = np.prod(self.input_shape)

        return self.output_shape

    def forward(self, input):
        return np.reshape(input, (1, self.output_shape))

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

