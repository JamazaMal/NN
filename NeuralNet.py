import numpy as np


class NeuralNetwork(object):
    @staticmethod
    def activation(s):
        return s  # Linear
        return s / (1 + np.abs(s))  # Sigmoid

    @staticmethod
    def activation_d(s):  # Sigmoid
        return 1  # Linear
        return 1 / (1 + np.abs(s)) ** 2

    def __init__(self, layers, lrate=0.001):
        # class variables
        self.learningrate = lrate
        self.netsize = len(layers)-1

        self.layeroutputs = [None] * self.netsize
        self.layeroutputdeltas = [None] * self.netsize

        self.weights = [None] * self.netsize
        for i in range(self.netsize):
            self.weights[i] = np.random.rand(layers[i], layers[i+1])

    def forward(self, inputs):
        self.layeroutputs[0] = self.activation(np.dot(inputs, self.weights[0]))
        for i in range(1, self.netsize):
            self.layeroutputs[i] = self.activation(np.dot(self.layeroutputs[i-1], self.weights[i]))
        return self.layeroutputs[self.netsize-1]

    def backward(self, inputs, expectedoutputs, outputs):
        self.layeroutputdeltas[self.netsize-1] = (expectedoutputs - outputs) * self.activation_d(outputs)
        for i in range(self.netsize-2, -1, -1):
            self.layeroutputdeltas[i] = self.layeroutputdeltas[i+1].dot(self.weights[i+1].T) * self.activation_d(self.layeroutputs[i])

        self.weights[0] += inputs.T.dot(self.layeroutputdeltas[0]) * self.learningrate
        for i in range(1, self.netsize):
            self.weights[i] += self.layeroutputs[i].T.dot(self.layeroutputdeltas[i]) * self.learningrate

    def train(self, inputs, expectedoutputs):
        outputs = self.forward(inputs)
        self.backward(inputs, expectedoutputs, outputs)
