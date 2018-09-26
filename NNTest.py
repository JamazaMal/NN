import numpy as np
import NeuralNet
import TestData

# Inputs
t_inputs, t_outputs = TestData.getData(10)

NN = NeuralNet.NeuralNetwork([1, 200, 1], .0001)

for _ in range(100000):
    inputs, outputs = TestData.getData(300)
    NN.train(inputs, outputs)
    print("Loss: {}".format(np.mean(np.square(t_outputs - NN.forward(t_inputs)))))

print(t_inputs)
print(NN.forward(t_inputs))
