from .Layer import Layer

import numpy as np

class Dense(Layer):
    def __init__(self, output_size):
        self.output_size = output_size

    def prepare(self, input_shape):
        stddev = np.sqrt(1 / input_shape)
        self.weights = np.random.normal(0.0, stddev, (input_shape, self.output_size))
        self.bias = np.ones(self.output_size)
        return self.output_size

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        self.weights -= np.dot(output_gradient.T, self.input).T * learning_rate
        self.bias -= output_gradient[0] * learning_rate

        return np.dot(self.weights, output_gradient.T).T
