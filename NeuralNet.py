import numpy as np


class NeuralNetwork(object):
    @staticmethod
    def sigmoid(s):
        # activation function
        return s/(1+np.abs(s))
        return 1/(1+np.exp(-s))

    @staticmethod
    def sigmoidprime(s):
        # derivative of sigmoid
        return 1/((np.abs(s)+1)**2)
        return s * (1 - s)

    def __init__(self, _i, _h, _o):
        # class variables
        self.z2 = None
        self.o_error = None
        self.o_delta = None
        self.z2_error = None
        self.z2_delta = None

        self.learningrate = 0.01

        # weights
        self.W1 = np.random.randn(_i, _h)
        self.W2 = np.random.randn(_h, _o)

    def forward(self, _inputs):
        # forward propagation through our network
        self.z2 = self.sigmoid(np.dot(_inputs, self.W1))
        _outputs = self.sigmoid(np.dot(self.z2, self.W2))
        return _outputs

    def backward(self, x, y, o):
        # backward propagate through the network
        self.o_error = y - o  # error in output
        self.o_delta = (y - o)*self.sigmoidprime(o)  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T)  # z2 error: how much our layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidprime(self.z2)  # applying derivative of sigmoid to z2 error

        self.W1 += x.T.dot(self.z2_delta)*self.learningrate  # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta)*self.learningrate  # adjusting second set (hidden --> output) weights

    def train(self, x, y):
        o = self.forward(x)
        self.backward(x, y, o)
