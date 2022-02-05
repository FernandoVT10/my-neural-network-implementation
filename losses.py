import numpy as np

def mse(y_predicted, y_true):
    return np.mean(np.power(y_true - y_predicted, 2))

def d_mse(y_predicted, y_true):
    return 2 * (y_predicted - y_true) / np.size(y_true)
