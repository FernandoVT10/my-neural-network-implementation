import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.weights = []
        self.biases = []

        for i in range(len(layers)):
            if i == len(layers) - 1:
                continue

            num_of_neurons = layers[i]
            num_of_neurons_of_next_layer = layers[i + 1]

            # here we create our connections for each neuron
            self.weights.append(
                np.random.random((num_of_neurons_of_next_layer, num_of_neurons))
            )

            # creating the biases of the hidden layers and the output
            self.biases.append(np.ones(num_of_neurons_of_next_layer))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return x * (1 - x)

    def forward(self, x):
        a = [x]
        for l in range(len(self.weights)):
            z = np.dot(self.weights[l], a[l]) + self.biases[l]
            a.append(self.sigmoid(z));
        return a

    def backward(self, a, y):
        alpha = 0.2

        deltas = []

        for x in range(len(self.weights)):
            l = len(self.weights) - x - 1

            delta = []

            if x == 0:
                delta = (a[l + 1] - y) * self.d_sigmoid(a[l + 1])
            else:
                delta = np.matmul(deltas[x - 1], self.weights[l + 1]) * self.d_sigmoid(a[l + 1])

            self.weights[l] -= alpha * np.matmul(
                delta.reshape(1, delta.shape[0]).T,
                a[l].reshape(1, a[l].shape[0])
            )
            self.biases[l] -= alpha * delta[0]

            deltas.append(delta)

    def predict(self, x):
        outputs = self.forward(x)
        output = outputs[-1]
        return output

    def train(self, x_train, y_train, epochs = 10000):
        for epoch in range(epochs):
            error = 0

            for x, y in zip(x_train, y_train):
                a = self.forward(x)

                error += np.sum(0.5 * np.power((a[-1] - y), 2))

                self.backward(a, y)

            error /= len(x_train)

            if(epoch % 1000 == 0):
                print(f"Epochs: {epoch}/{epochs}. Error={error}")
