import numpy as np
from random import random, shuffle

from numpy.core.numeric import count_nonzero

class Perceptron:
    def __init__(self, n, lr) -> None:
        # Init perceptron with weights, bias and learning rate
        self.weights = np.random.random(n) - 0.5
        self.bias = np.random.random()-0.5
        self.learning_rate = lr
        print(self.weights)
    
    def error(self, t, p):
        # target - prediction
        return t-p

    def activation(self, input):
        # Step activation
        if input >= 0:
            return 1
        else:
            return 0

    def predict(self, x):
        y = np.dot(self.weights, x) + self.bias
        return self.activation(y)
    
    def update_weights(self, input, error):
        for i in range(len(self.weights)):
            # w_i(p+1)      = w_i(p)          + η                 * xi(p)  * e(p)
            self.weights[i] = self.weights[i] + self.learning_rate*input[i]*error
        # θ(p+1)  = θ(p)     + η                * θ(p)    * e(p)
        self.bias = self.bias+self.learning_rate*self.bias*error

# Init with 2 inputs and lr = 0.01
percept = Perceptron(2, 0.1)
dataset = [[0,0],[0,1],[1,0],[1,1]]
labels = [0,0,0,1]
order = [i for i in range(4)]

e_total = None
count = 0
print("Bias: ",percept.bias)
print("Weights: ",percept.weights)
fail = 0

while e_total is None or e_total != 0:
    e_list = []
    e_sqr = []
    e_total = 0
    for i in order:
        # Predict and calcultae error
        e = percept.error(labels[i], percept.predict(dataset[i]))
        # Update W and B
        percept.update_weights(dataset[i], e)
        e_sqr.append(e**2)
        e_list.append(e)
    e_total = sum(e_sqr)
    count += 1
    print(f"Epoch {count}: Mean Squared Error: {e_total/4}")
    if count >= 100:
        break

print("Bias: ",percept.bias)
print("Weights: ",percept.weights)
