import numpy as np

from scipy import signal

from .Layer  import Layer

class Convolution(Layer):
    def __init__(self, input_size: int, num_of_kernels: int, kernel_size = 3, depth = 3):
        self.input_size = input_size
        self.output_size = self.input_size - kernel_size + 1 
        self.kernel_size = kernel_size
        self.num_of_kernels = num_of_kernels

        # This is the number of color channels, for example a rgb image is goona have 3 color channels:
        # red, green, and blue
        self.depth = depth

        self.kernels = np.random.random((self.num_of_kernels, self.depth, self.kernel_size, self.kernel_size))
        self.biases = np.zeros((self.num_of_kernels, self.output_size, self.output_size))

    def forward(self, input):
        self.input = input

        output = np.zeros((self.num_of_kernels, self.output_size, self.output_size))

        for i in range(self.num_of_kernels):
            for k in range(self.depth):
                output[i] += signal.correlate2d(input[k], self.kernels[i, k], mode="valid")
        
            output[i] += self.biases[i]

        return output
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros((self.num_of_kernels, self.input_size, self.input_size))

        for i in range(self.num_of_kernels):
            for k in range(self.depth):
                self.kernels[i, k] -= signal.correlate2d(self.input[k], output_gradient[i], mode="valid") * learning_rate
                input_gradient[i] += signal.convolve2d(output_gradient[i], self.kernels[i, k], mode="full")

        self.biases -= output_gradient * learning_rate

        return input_gradient;
