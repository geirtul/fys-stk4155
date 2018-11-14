import numpy as np
from tqdm import tqdm


class NeuralNet:
    """
    Implementation of a neural net of the type 'Single Layer Perceptron' (SLP).
    This implementation is currently spicific to binary classification, but
    can be expanded to generalize by passing a n_classes parameter and
    using for example the Softmax function.

    Net will output P(class 0), and assumes that P(class 1) = 1 - P(class 0)
    """
    def __init__(
            self,
            x,
            y,
            x_test,
            y_test,
            n_hidden=10,
            epochs=1e2,
            eta = 0.01,
            batch_size=100,
    ):
        """
        Initialize the neural network
        :param x: Training data for the network
        :param y: Targets for the training data
        :param x_test: Validation data
        :param y_test: Targets for validation data
        :param n_hidden: Number of hidden nodes in the hidden layer, default 10
        :param epochs: Number of times the training algorithm feeds all training
                        data through the network. Default 100
        :param eta: Learning rate, default 0.01
        """

        # Data. Reshaping targets to be (N, 1)
        self.x = x
        self.y = y.reshape((len(y), 1))
        self.x_test = x_test
        self.y_test = y_test.reshape((len(y_test), 1))

        # Variables
        self.n_inputs = self.x.shape[0]
        self.n_features = self.x.shape[1]
        self.n_hidden = n_hidden
        self.epochs = int(epochs)
        self.eta = eta
        self.batch_size = batch_size
        self.n_iter = self.n_inputs // self.batch_size

        # Weights (w) and biases (b)
        self.w_hidden = None
        self.w_output = None
        self.b_hidden = None
        self.b_output = None
        self.setup_weights_biases()

        # Keep track of accuracy while training
        self.accuracies = []

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

                a_h, a_o = self.feed_forward()
                self.backwards_propagation(a_h, a_o)

            # Save the accuracy for each epoch
            current_accuracy = self.accuracy()
            self.accuracies.append(current_accuracy)

    def feed_forward(self, x=None):
        """
        Feeds the input forward through the network to generate an output.
        If x is not provided self.x is used.
        :param x: Input to be fed forward. Default: None.
        :return: activations in hidden layer, and output.
        """

        # Calculate activations in hidden layer
        if x is not None:
            z_hidden = np.matmul(x, self.w_hidden) + self.b_hidden
        else:
            z_hidden = np.matmul(self.x_batch, self.w_hidden) + self.b_hidden
        a_hidden = self.sigmoid(z_hidden)

        # Calculate activations in output layer
        z_output = np.matmul(a_hidden, self.w_output) + self.b_output
        a_output = self.sigmoid(z_output)

        return a_hidden, a_output

    def backwards_propagation(self, a_h, a_o):
        """
        Calculate error in output and backwards propagate to update weights and
        biases.

        :param a_h: Activations in the hidden layer
        :param a_o: Activations in the output layer (probability of class 0)
        """

        # Error in output layer
        error_output = self.y_batch - a_o

        # Gradients for the output weights and bias
        delta_w_output = np.matmul(a_h.T, error_output)
        delta_b_output = np.sum(error_output)

        # Error in the hidden layer
        error_hidden = np.matmul(error_output, self.w_output.T) * a_h * (1 - a_h)

        # Gradients for the hidden weights and biases
        delta_w_hidden = np.matmul(self.x_batch.T, error_hidden)
        delta_b_hidden = np.sum(error_hidden)

        # Update weights and biases
        self.w_output -= self.eta * delta_w_output
        self.b_output -= self.eta * delta_b_output

        self.w_hidden -= self.eta * delta_w_hidden
        self.b_hidden -= self.eta * delta_b_hidden

    def setup_weights_biases(self):
        """
        Initializes the weights to random values and biases to a small value.
        """
        # Weights and biases in hidden layer
        self.w_hidden = np.random.randn(self.n_features, self.n_hidden)
        self.b_hidden = np.zeros(self.n_hidden) + 0.01

        # Weights and biases in output layer
        self.w_output = np.random.randn(self.n_hidden, 1)
        self.b_output = 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def accuracy(self, x=None, y=None):
        """
        Calculate the accuracy score for the network.
        :param x: test / validation data to generate output with
        :param y: test / validation targets to check with
        :return: accuracy score
        """
        # TODO: Check if ordered or disordered is class 0 in the data set.

        if x is not None and y is not None:
            y = y.reshape((len(y), 1))
            output = self.feed_forward(x)[1]
            score = np.sum(y == output) / len(y)
        else:
            output = self.feed_forward(self.x_test)[1]
            score = np.sum(self.y_test == output)/len(self.y_test)

        return score
