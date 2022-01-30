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
            local_a = np.zeros(len(self.weights[l]))

            for n in range(len(local_a)):
                z = np.dot(self.weights[l][n], a[l]) + self.biases[l][n]
                local_a[n] = self.sigmoid(z)

            a.append(local_a)

        return a

    def predict(self, x):
        outputs = self.forward(x)
        output = outputs[-1]
        return output

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

    def train(self, X, Y):
        epoch = 10000
        #  epoch = 1

        while epoch > 0:
            epoch -= 1

            for i in range(len(X)):
                a = self.forward(X[i])
                self.backward(a, Y[i])


x = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0],

    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0],

    [1, 0],
    [0, 0],
    [1, 1],
    [0, 1],
    
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
])
y = np.array([
    1, 0, 0, 1,
    0, 0, 1, 1,
    0, 1, 1, 0,
    1, 1, 0, 0
])

#  x = np.array([
#      [1, 1],
#      [0, 1],
#      [1, 0],
#      [0, 0]
#  ])

#  y = np.array([
#      [1, 1],
#      [0, 1],
#      [1, 0],
#      [0, 0]
#  ])

nn = NeuralNetwork([2, 4, 1])

nn.train(x, y)

while True:
    x1 = input("Enter a x1: ")
    x2 = input("Enter a x2: ")

    print(nn.predict([
        int(x1), int(x2)
    ]))
