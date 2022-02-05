from layers import Layer

class Activation(Layer):
    def __init__(self, activation, d_activation):
        self.activation = activation
        self.d_activation = d_activation

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.d_activation(self.input)
