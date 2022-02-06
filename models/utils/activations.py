import numpy as np

class Activation:
    def evaluate(self, z):
        raise NotImplementedError

    def derivative(self, z):
        raise NotImplementedError

class ReLU(Activation):
    def evaluate(self, z):
        return np.maximum(0, z)

    def derivative(self, z):
        d = np.copy(z)
        d[d < 0] = 0
        d[d > 0] = 1

        return d

class Sigmoid(Activation):
    def evaluate(self, z):
        return 1 / (1 + np.exp(-z))

    def derivative(self, z):
        return self.evaluate(z) * (1 - self.evaluate(z))

    