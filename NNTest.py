import numpy as np
import NeuralNet
import TestData

# Inputs
inputs, outputs = TestData.getData(3)
t_inputs, t_outputs = TestData.getData(3)


NN = NeuralNet.NeuralNetwork([2, 3, 4, 1], 0.001)

for _ in range(100000):
    NN.train(inputs, outputs)
    print("Loss: {}".format(np.mean(np.square(t_outputs - NN.forward(t_inputs)))))

print(t_inputs)
print(NN.forward(t_inputs))
