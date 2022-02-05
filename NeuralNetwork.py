import numpy as np

import matplotlib.pyplot as plt

from layers import Layer

class NeuralNetwork:
    def __init__(self, layers: "list[Layer]", loss, d_loss):
        self.layers = layers
        self.loss = loss
        self.d_loss = d_loss

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y, output, learning_rate):
        output_gradient = (output - y)
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)

    def train(self, x_train, y_train, epochs = 10000, learning_rate = 0.1):
        for e in range(epochs):
            error = 0

            for x, y in zip(x_train, y_train):
                output = self.forward(x)
                output_gradient = self.d_loss(output, y)

                error += self.loss(output, y)

                self.backward(y, output_gradient, learning_rate)

            error /= np.size(x_train)

            if e % 1000 == 0:
                print(f"Epoch: {e + 1000} / {epochs}  Error: {error}")


    def predict(self, x):
        return self.forward(x)
