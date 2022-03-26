import numpy as np

from scipy import signal

from .Layer  import Layer

class Convolution(Layer):
    def __init__(self, depth, kernel_shape):
        self.kernel_shape = kernel_shape
        self.depth = depth

    def prepare(self, input_shape):
        input_depth, input_height, input_width = input_shape
        kernel_height, kernel_width = self.kernel_shape

        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (self.depth, input_height - kernel_height + 1, input_width - kernel_width + 1)

        self.kernels = np.random.random((self.depth, input_depth, *self.kernel_shape))
        self.biases = np.zeros(shape=self.output_shape)

        return self.output_shape

    def forward(self, input):
        self.input = input

        output = np.copy(self.biases)

        for i in range(self.depth):
            for k in range(self.input_depth):
                output[i] += signal.correlate2d(input[k], self.kernels[i, k], mode="valid")

        return output
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros((self.depth, *self.input_shape[1:]))

        for i in range(self.depth):
            for k in range(self.input_depth):
                input_gradient[i] += signal.convolve2d(output_gradient[i], self.kernels[i, k], mode="full")
                self.kernels[i, k] -= signal.correlate2d(self.input[k], output_gradient[i], mode="valid") * learning_rate

        self.biases -= output_gradient * learning_rate

        return input_gradient;
