import numpy as np
from  activation_loss_functions import ReLU,Softmax,ELU



class Dense:
    def __init__(self, input_nodes, output_nodes):
        self.input = None
        self.output = None
        weight_lim = np.sqrt(6 / (input_nodes + output_nodes))
        self.weights = np.random.uniform(low=-weight_lim, high=weight_lim, size=(input_nodes, output_nodes))
        self.biases = np.zeros((1, output_nodes))


    def forward(self, x):
        self.input = x
        self.output = np.dot(self.input, self.weights) + self.biases

        return self.output

    def backward(self, out_err, learning_rate=0.01):
        error = np.dot(out_err, self.weights.T)
        weight_update = np.dot(self.input.T, out_err)

        self.weights -= learning_rate * weight_update
        self.biases -= learning_rate * np.mean(out_err)

        return error


class MLP:
    def __init__(self, input_dim, output_dim):

        dense1 = ('Dense', Dense(input_dim, 64))
        activation1 = ('activation', ELU())
        dense2 = ('Dense', Dense(64, 64))
        activation2 = ('activation', ELU())
        dense3 = ('Dense', Dense(64, 64))
        activation3 = ('activation', ELU())
        dense4 = ('Dense', Dense(64, 64))
        activation4 = ('activation', ReLU())
        dense5 = ('Dense', Dense(64, 32))
        activation5 = ('activation', ReLU())
        dense6 = ('Dense', Dense(32, output_dim))
        activation6 = ('activation', Softmax())

        self.layers = [dense1,activation1,dense2,activation2,dense3
            ,activation3,dense4,activation4,dense5,activation5,dense6,activation6]

    def forward(self, input_data):
        for layer_type, layer in self.layers:
            if layer_type == 'Dense':
                input_data = layer.forward(input_data)
            else:
                input_data = layer.activation(input_data)
        return input_data

    def backward(self, L_grad, learning_rate):
        for layer_type, layer in reversed(self.layers):
            if layer_type == 'Dense':
                L_grad = layer.backward(L_grad, learning_rate)
            else:
                L_grad = layer.derivative() * L_grad
