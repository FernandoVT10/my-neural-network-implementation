import numpy as np

from losses import losses_list

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.error_array = []

    def prepare(self, input_shape, loss):
        self.loss, self.loss_prime = losses_list[loss]

        for layer in self.layers:
            input_shape = layer.prepare(input_shape)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)

    def train(self, x_train, y_train, epochs = 10000, learning_rate = 0.1):
        for e in range(epochs):
            error = 0

            for x, y in zip(x_train, y_train):
                output = self.forward(x)
                output_gradient = self.loss_prime(output, y)

                error += self.loss(output, y)

                self.backward(output_gradient, learning_rate)

            error /= x_train.shape[0]
            self.error_array.append(error)

            print(f"Epoch: {e + 1} / {epochs}  Error: {error}")

    def test(self, x_test, y_test, verbose=False):
        accuracy: "list[int]" = []

        for x, y in zip(x_test, y_test):
            prediction = self.forward(x)

            local_accuracy = 0
            if(y.shape[0] == 1):
                # if there's just a prediction it means that we're using binary cross entropy and therefore
                # we may take 0 as a correct value
                local_accuracy = 1 - abs(y - prediction)
            else:
                local_accuracy = 1 - np.max((y - prediction))

            accuracy.append(np.max(local_accuracy))

            if verbose:
                print(f"Prediction: {prediction} True: {y} Accuracy: {local_accuracy}")

        average_accuracy = np.sum(accuracy) / np.shape(y_test)[0]
        print(f"Total Accuracy: {average_accuracy}")

    def predict(self, x):
        return self.forward(x)
