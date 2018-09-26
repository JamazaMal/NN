import numpy as np


class NeuralNetwork(object):
    @staticmethod
    def activation(s):
        return s/(1+np.abs(s))

    @staticmethod
    def activation_d(s):
        return 1/(1+(np.abs(s))**2)

    def __init__(self, layers):
        # class variables
        self.z2 = None
        self.o_error = None
        self.o_delta = None
        self.z2_error = None
        self.z2_delta = None

        self.learningrate = 0.01
        self.netsize = len(layers)-1

        self.layeroutputs = [None] * self.netsize
        self.layeroutputdeltas = [None] * self.netsize

        self.weights = [None] * self.netsize
        for i in range(self.netsize):
            self.weights[i] = np.random.randn(layers[i], layers[i+1])

    def forward(self, inputs):
        self.layeroutputs[0] = self.activation(np.dot(inputs, self.weights[0]))
        for i in range(1,self.netsize):
            self.layeroutputs[i] = self.activation(np.dot(self.layeroutputs[i-1], self.weights[i]))

        #self.z2 = self.activation(np.dot(_inputs, self.weights[0]))
        #_outputs = self.activation(np.dot(self.z2, self.weights[1]))

        return self.layeroutputs[self.netsize-1]

    def backward(self, inputs, y, o):

        self.layeroutputdeltas[self.netsize] = (y - o)*self.activation_d(o)
        for i in range(self.netsize-1, -1):
            self.layeroutputdeltas[i] = self.layeroutputdeltas[i+1].dot(self.weights[i].T) * self.activation_d(self.layeroutputs[i])

        self.weights[0] += inputs.T.dot(self.layeroutputdeltas[0]) * self.learningrate
        for i in range(1, self.netsize):
            self.weights[i] += self.layeroutputdeltas[i-1].T.dot(self.layeroutputdeltas[i]) * self.learningrate

        #self.o_delta = (y - o)*self.activation_d(o)

        #self.z2_error = self.o_delta.dot(self.weights[1].T)
        #self.z2_delta = self.z2_error*self.activation_d(self.z2)

        #self.weights[0] += x.T.dot(self.z2_delta)*self.learningrate
        #self.weights[1] += self.z2.T.dot(self.o_delta)*self.learningrate

    def train(self, inputs, expectedoutputs):
        outputs = self.forward(inputs)
        self.backward(inputs, expectedoutputs, outputs)
