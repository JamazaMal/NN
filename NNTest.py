import numpy as np
import NeuralNet

# Inputs
inputs = np.array(([0, 0], [1, 0], [0, 1], [1, 1]), dtype=float)
outputs = np.array(([0], [1], [1], [0]), dtype=float)

NN = NeuralNet.NeuralNetwork([2, 3, 1])

for i in range(1):  # trains the NN 1,000 times
    print("Loss: {}".format(np.mean(np.square(outputs - NN.forward(inputs)))))
    NN.train(inputs, outputs)


