import numpy as np

from .Activation import Activation

class ReLU(Activation):
    def __init__(self):
        def activation(x):
            return np.maximum(0, x)

        def activation_prime(x):
            return np.sign(np.maximum(0, x))

        super().__init__(activation, activation_prime)
