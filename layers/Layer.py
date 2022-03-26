class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def prepare(self, input_shape):
        return input_shape

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass
