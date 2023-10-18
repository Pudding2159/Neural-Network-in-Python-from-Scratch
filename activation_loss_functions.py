import numpy as np

def one_hot_encoding(y):
    y = np.array(y, dtype=int)
    num_categories = np.amax(y) + 1
    return np.eye(num_categories)[y.reshape(-1)]
class CrossEntropy:
    def _clip_probabilities(self, probabilities):
        self.epsilon = 1e-9
        return np.clip(probabilities, self.epsilon, 1 - self.epsilon)

    def loss(self, y_true, y_pred):
        y_pred_clipped = self._clip_probabilities(y_pred)
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

    def backward(self, y_true, y_pred):
        y_pred_clipped = self._clip_probabilities(y_pred)
        return -(y_true / y_pred_clipped) + (1 - y_true) / (1 - y_pred_clipped)

class Sigmoid:
    def activation(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def derivative(self):
        return self.output * (1.0 - self.output)

class ReLU:

    def activation(self, inputs):
        self.input = inputs
        return np.maximum(0, inputs)

    def derivative(self):
        return (self.input > 0).astype(int)

class ELU:

    def activation(self, input, alpha=1.0):
        self.input = input
        self.output = np.where(input > 0, input, alpha * (np.exp(input) - 1))
        return np.where(input > 0, input, alpha * (np.exp(input) - 1))

    def derivative(self, alpha=1.0):
        fx = self.output
        return np.where(self.input > 0, 1, fx + alpha)

class Softmax:

    def activation(self, inputs):
        self.input = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        return probabilities

    def derivative(self):
        p = self.activation(self.input)
        return p * (1 - p)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred, axis=0) / len(y_true)

