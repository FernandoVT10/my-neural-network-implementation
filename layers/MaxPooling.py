import numpy as np
from .Layer import Layer

from math import floor

class MaxPooling(Layer):
    def __init__(self, pooling_shape):
        self.shape = pooling_shape

    def prepare(self, input_shape):
        self.input_shape = input_shape

        input_depth, input_height, input_width = self.input_shape
        pool_height, pool_width = self.shape

        self.output_shape = (
            input_depth,
            floor(input_height / pool_height),
            floor(input_width / pool_width)
        )

        return self.output_shape

    def forward(self, input):
        self.input = input

        pool_height, pool_width = self.shape
        output_height, output_width = self.output_shape[1:]

        self.output = np.zeros(self.output_shape)

        for y in range(output_height):
            top = y * pool_height
            bottom = top + pool_height

            for x in range(output_width):
                left = x * pool_width
                right = left + pool_width
                
                matrix = self.input[:, top:bottom, left:right]
                self.output[:, y, x] = np.max(matrix, axis=(1, 2))

        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros(self.input_shape)

        pool_width, pool_height = self.shape
        output_depth, output_height, output_width = self.output_shape

        temp_ones = np.ones((output_depth, pool_height, pool_width))

        for y in range(output_height):
            top = y * pool_height
            bottom = top + pool_height

            for x in range(output_width):
                left = x * pool_width
                right = left + pool_width

                mask = temp_ones * self.output[:, y, x][:,np.newaxis,np.newaxis]
                local_gradient = self.input[:, top:bottom, left:right]==mask
                input_gradient[:, top:bottom, left:right] = local_gradient * output_gradient[:, y, x][:, np.newaxis, np.newaxis]

        return input_gradient
