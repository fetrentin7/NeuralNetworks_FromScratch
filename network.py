import numpy as np
import random
import perceptron

class Network(object):
    def __init__(self, size):
        self.num_layers = len(size)
        self.size = size

        self.b = [np.random.randn(y, 1)for y in size[1:]]
        self.w = [np.random.randn(y, x)for x, y in zip(size[:-1], size[1:])]

    def feedforward(self, a):

        for b, w in zip(self.b, self.w):
            a = perceptron.sigmoid(np.dot(w, a) + b)
        return a

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def stochastic_gradient(self, training_data, epochs,mini_batch_size, eta, test_data=None):

        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(
                    "Epoch {0}: {1} / {2}".format(j,self.evaluate(test_data),n_test)
                )
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.b]
        nabla_w = [np.zeros(w.shape) for w in self.w]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [
                nb + dnb
                for nb, dnb in zip(nabla_b, delta_nabla_b)
            ]
            nabla_w = [
                nw + dnw
                for nw, dnw in zip(nabla_w, delta_nabla_w)
            ]
        self.w = [w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.w, nabla_w)
        ]
        self.b = [
            b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.b, nabla_b)
        ]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.b]
        nabla_w = [np.zeros(w.shape) for w in self.w]

        # Feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.b, self.w):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = perceptron.sigmoid(z)
            activations.append(activation)
        # Backward pass
        delta = (self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]))
        nabla_b[-1] = delta

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = (np.dot(self.w[-l + 1].transpose(), delta)* sp)

            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)), y)
            for x, y in test_data]

        return sum(int(x == y)
            for x, y in test_results)


def sigmoid_prime(z):
    return perceptron.sigmoid(z) * (1 - perceptron.sigmoid(z))