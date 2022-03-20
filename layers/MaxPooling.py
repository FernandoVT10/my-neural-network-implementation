import numpy as np
from .Layer import Layer

from math import floor


class MaxPooling(Layer):
    def __init__(self, pooling_shape):
        self.shape = pooling_shape

    def forward(self, input):
        """
            Here the output shape is given by:
            1. The input depth. In this case this layer is gonna be after of a convolutional
            layer, thus we need to return the same depth or number of channels (i.e: r g b)

            2. The columns of the input divided by the width of the pooling "filter".

            3. The rows of the input divided by the height of the pooling "filter"
        """
        self.input = input
        output_shape = (
            self.input.shape[0],
            floor(self.input.shape[2] / self.shape[0]),
            floor(self.input.shape[1] / self.shape[1])
        )

        self.output = np.zeros(output_shape)

        for y in range(output_shape[2]):
            column_top = y * self.shape[1]
            column_bottom = column_top + self.shape[1]

            for x in range(output_shape[1]):
                row_left = x * self.shape[0]
                row_right = row_left + self.shape[0]
                
                a = self.input[:, column_top:column_bottom, row_left:row_right]
                self.output[:, x, y] = np.max(a, axis=(1, 2))

        return self.output

    def backward(self, output_gradient, learning_rate):
        pass
