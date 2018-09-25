import numpy as np
import NeuralNet

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X = X/np.amax(X, axis=0)  # maximum of X array
y = y/100  # max test score is 100

NN = NeuralNet.NeuralNetwork()
print("Input: " + str(X))
print("Output: " + str(y))

for i in range(1000):  # trains the NN 1,000 times
    # print ("Input: " + str(X))
    # print ("Actual Output: " + str(y))
    # print ("Predicted Output: " + str(NN.forward(X)))
    print("Loss: " + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
    NN.train(X, y)
