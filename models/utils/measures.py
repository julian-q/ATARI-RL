import numpy as np

class Measure:
    def evaluate(self, y, y_hat):
        raise NotImplementedError

    def derivative(self, y, y_hat):
        raise NotImplementedError

class LL(Measure):
    def evaluate(self, y, y_hat):
        return y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)

    def derivative(self, y, y_hat):
        eps = np.finfo(float).eps
        return (y / (y_hat + eps)) - ((1 - y) / (1 - (y_hat + eps)))