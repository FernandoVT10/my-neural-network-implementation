from .Layer import Layer

import numpy as np

class Dense(Layer):
    def __init__(self, input_shape, output_shape):
        self.weights = np.random.random((input_shape, output_shape))
        self.bias = np.ones(output_shape)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        self.weights -= np.matmul(output_gradient.T, self.input).T * learning_rate
        self.bias -= output_gradient[0] * learning_rate

        return np.matmul(output_gradient, self.weights.T)