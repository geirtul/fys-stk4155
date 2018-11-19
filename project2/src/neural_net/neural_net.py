import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class NeuralNet:
    """
    Implementation of a neural net of the type 'Multi Layer Perceptron' (MLP).
    This implementation is currently spicific to binary classification, but
    can be expanded to generalize by passing a n_classes parameter and
    using for example the Softmax function.
    TODO: Implement early stopping algorithm
    """

    def __init__(
            self,
            x,
            y,
            x_test,
            y_test,
            n_layers=1,
            n_nodes=(50,),
            n_classes=1,
            epochs=10,
            eta=0.01,
            batch_size=100,
            lmda=0.0,
    ):
        """
        Initialize the neural network
        :param x: Training data for the network
        :param y: Targets for the training data
        :param x_test: Validation data
        :param y_test: Targets for validation data
        :param n_layers: How many hidden layers in the network.
        :param n_nodes: tuple, length = n_layers. Number of nodes in the
                            hidden layers
        :param n_classes: int, number of output classes/categories
        :param epochs: Number of times the training algorithm feeds all training
                        data through the network. Default 10
        :param eta: Learning rate, default 0.01
        :param batch_size: Size of mini-batches for Gradient Descent.
        :param lmda: Lambda, regularization parameter.
        """

        # Data
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test

        # Variables
        self.n_inputs = self.x.shape[0]
        self.n_features = self.x.shape[1]
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.n_nodes = n_nodes + (n_classes,)  # hidden layers + output layer
        self.epochs = int(epochs)  # cast to int in case 1e1 etc.
        self.eta = eta
        self.batch_size = batch_size
        self.n_iter = self.n_inputs // self.batch_size
        self.lmda = lmda

        # Weights and biases.
        self.weights = np.zeros(self.n_layers + 1, dtype=object)
        self.bias = np.zeros(self.n_layers + 1, dtype=object)

        self.initialize_weights_biases()

    def initialize_weights_biases(self):
        """
        Initializes the weights to random values and biases to a small value.
        """

        # + 1 is for the output layer

        for i in range(self.n_layers + 1):
            if i == 0:
                self.weights[i] = np.random.randn(self.n_features, self.n_nodes[i])
            else:
                self.weights[i] = np.random.randn(self.n_nodes[i-1], self.n_nodes[i])
            self.bias[i] = np.zeros(self.n_nodes[i]) + 0.01

    def feed_forward(self, x):
        """
        Feeds the input forward through the network to generate an output.
        If x is not provided self.x is used.
        :param x: Input to be fed forward. Default: None.
        :return: activations in hidden layers, and output.
        """
        # Calculate activations in hidden layers
        activations = np.zeros(self.n_layers + 1, dtype=object)
        for i in range(self.n_layers + 1):
            if i == 0:
                z = np.matmul(x, self.weights[i]) + self.bias[i]
                activations[i] = self.sigmoid(z)
            else:
                z = np.matmul(
                    activations[i - 1], self.weights[i]) + self.bias[i]
                activations[i] = self.sigmoid(z)

        return activations

    def backwards_propagation(self, a):
        """
        Calculate error in output and backwards propagate to update weights and
        biases.

        :param a: Activations in all layers, last one being output.
        """

        # Storage arrays
        errors = np.zeros(self.n_layers + 1, dtype=object)
        delta_w = np.zeros(self.n_layers + 1, dtype=object)
        delta_b = np.zeros(self.n_layers + 1, dtype=object)

        # Calculate errors and changes in weights and biases
        for i in reversed(range(self.n_layers + 1)):
            if i == self.n_layers:
                errors[i] = a[i] - self.y_batch
            else:
                errors[i] = np.matmul(errors[i+1], self.weights[i+1].T) * a[i] * (1 - a[i])

            if i > 0:
                delta_w[i] = np.matmul(a[i - 1].T, errors[i])
                delta_b[i] = np.sum(errors[i], axis=0)
            else:
                delta_w[i] = np.matmul(self.x_batch.T, errors[i])
                delta_b[i] = np.sum(errors[i], axis=0)

        # Regularization
        if self.lmda > 0.0:
            delta_w += self.lmda * self.weights

        # Update weights and biases
        self.weights -= self.eta * delta_w
        self.bias -= self.eta * delta_b

    def train(self):
        """
        Runs the training algorithm with Batch Gradient Descent.
        """
        indices = np.arange(self.n_inputs)

        for i in tqdm(range(self.epochs)):
            for j in range(self.n_iter):
                chosen_indices = np.random.choice(
                    indices, size=self.batch_size, replace=False)

                # Batch training data and targets
                self.x_batch = self.x[chosen_indices]
                self.y_batch = self.y[chosen_indices]

                activations = self.feed_forward(self.x_batch)
                self.backwards_propagation(activations)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        y_pred = self.feed_forward(x)[-1]
        return y_pred

    def accuracy(self, x, y):
        """
        Calculate the accuracy score for the network.
        :param x: test / validation data to generate output with
        :param y: test / validation targets to check with
        :return: accuracy score
        """

        output = np.round(self.predict(x))
        score = np.sum(y == output) / len(y)

        return score
