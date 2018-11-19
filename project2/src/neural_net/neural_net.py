import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class NeuralNet:
    """
    Implementation of a neural net of the type 'Multi Layer Perceptron' (MLP).
    This implementation is currently spicific to binary classification, but
    can be expanded to generalize by passing a n_classes parameter and
    using for example the Softmax function.
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
            eta = 0.01,
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
        self.hidden_nodes = n_nodes
        self.n_classes = n_classes
        self.epochs = int(epochs) # cast to int in case 1e1 etc.
        self.eta = eta
        self.batch_size = batch_size
        self.n_iter = self.n_inputs // self.batch_size
        self.lmda = lmda

        self.setup_weights_biases()

        # Keep track of accuracy while training
        self.accuracies = []

    def setup_weights_biases(self):
        """
        Initializes the weights to random values and biases to a small value.
        """
        # Weights and biases in hidden layers
        self.w_hidden = np.zeros(self.n_layers)
        self.b_hidden = np.zeros(self.n_layers)
        for i in range(self.n_layers):
            if i == 0:
                self.w_hidden[i] = np.random.randn(self.n_features, self.hidden_nodes[0])
            else:
                self.w_hidden[i] = np.random.randn(self.hidden_nodes[i-1], self.hidden_nodes[i])
            self.b_hidden[i] = np.zeros(self.hidden_nodes[i]) + 0.01

        self.w_output = np.random.randn(self.hidden_nodes[-1], self.n_classes)
        self.b_output = np.zeros(self.n_classes) + 0.01

        # self.w_hidden1 = np.random.randn(self.n_features, self.n_hidden1)
        # self.b_hidden1 = np.zeros(self.n_hidden1) + 0.01
        #
        # self.w_hidden2 = np.random.randn(self.n_hidden1, self.n_hidden2)
        # self.b_hidden2 = np.zeros(self.n_hidden2) + 0.01
        #
        # # Weights and biases in output layer
        # self.w_output = np.random.randn(self.n_hidden2, 1)
        # self.b_output = 0.01

    def train(self):
        """
        Runs the training algorithm with Batch Gradient Descent.
        """
        indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.n_iter):
                chosen_indices = np.random.choice(
                    indices, size=self.batch_size, replace=False)

                # Batch training data and targets
                self.x_batch = self.x[chosen_indices]
                self.y_batch = self.y[chosen_indices]

                a_h1, a_h2, a_o = self.feed_forward()
                self.backwards_propagation(a_h1, a_h2, a_o)

            # Save the accuracy for each epoch
            current_accuracy = self.accuracy()
            self.accuracies.append(current_accuracy)

    def feed_forward(self, x=None):
        """
        Feeds the input forward through the network to generate an output.
        If x is not provided self.x is used.
        :param x: Input to be fed forward. Default: None.
        :return: activations in hidden layers, and output.
        """

        # Calculate activations in hidden layers
        a_hidden = np.zeros(self.n_layers)
        for i in range(self.n_layers):
            if i == 0:
                if x is not None:
                    z = np.matmul(x, self.w_hidden[i]) + self.b_hidden[i]
                else:
                    z = np.matmul(self.x_batch, self.w_hidden[i]) + self.b_hidden[i]
                a_hidden[i] = self.sigmoid(z)
            else:
                z = np.matmul(a_hidden[i-1], self.w_hidden[i]) + self.b_hidden[i]
                a_hidden[i] = self.sigmoid(z)
        z_output = np.matmul(a_hidden[-1], self.w_output) + self.b_output
        a_output = self.sigmoid(z_output)

        # if x is not None:
        #     z_hidden1 = np.matmul(x, self.w_hidden1) + self.b_hidden1
        # else:
        #     z_hidden1 = np.matmul(self.x_batch, self.w_hidden1) + self.b_hidden1
        #
        # a_hidden1 = self.sigmoid(z_hidden1)
        # z_hidden2 = np.matmul(a_hidden1, self.w_hidden2) + self.b_hidden2
        # a_hidden2 = self.sigmoid(z_hidden2)
        #
        # # Calculate activations in output layer
        # z_output = np.matmul(a_hidden2, self.w_output) + self.b_output
        # a_output = self.sigmoid(z_output)
        # return a_hidden1, a_hidden2, a_output
        return a_hidden, a_output

    def backwards_propagation(self, a_h, a_o):
        """
        Calculate error in output and backwards propagate to update weights and
        biases.

        :param a_h: Activations in hidden layers
        :param a_o: Activations in the output layer (probability of class 0)
        """

        # Storage arrays
        error_hidden = np.zeros(self.n_layers)
        delta_w_hidden = np.zeros(self.n_layers)
        delta_b_hidden = np.zeros(self.n_layers)

        # Calculate errors and changes in weights, handling output first.
        error_output = a_o - self.y_batch
        delta_w_output = np.matmul(a_h[-1].T, error_output)
        delta_b_output = np.sum(error_output, axis=0)

        for i in reversed(range(self.n_layers)):
            if i == self.n_layers - 1:
                error_hidden[i] = np.matmul(error_output, self.w_output.T)*a_h[i]*(1 - a_h[i])
            else:
                error_hidden[i] = np.matmul(error_hidden[i+1], self.w_hidden[i+1].T)*a_h[i]*(1 - a_h[i])

            if i > 0:
                delta_w_hidden[i] = np.matmul(a_h[i-1].T, error_hidden[i])
                delta_b_hidden[i] = np.sum(error_hidden[i], axis=0)
            else:
                delta_w_hidden[i] = np.matmul(self.x_batch.T, error_hidden[i])
                delta_b_hidden[i] = np.sum(error_hidden[i], axis=0)

        # error_hidden2 = np.matmul(error_output, self.w_output.T) * a_h2 * (1 - a_h2)
        # error_hidden1 = np.matmul(error_hidden2, self.w_hidden2.T) * a_h1 * (1 - a_h1)
        #
        # # Gradients for the weights and biases
        # delta_w_output = np.matmul(a_h2.T, error_output)
        # delta_b_output = np.sum(error_output, axis=0)
        #
        # delta_w_hidden2 = np.matmul(a_h1.T, error_hidden2)
        # delta_b_hidden2 = np.sum(error_hidden2, axis=0)
        #
        # delta_w_hidden1 = np.matmul(self.x_batch.T, error_hidden1)
        # delta_b_hidden1 = np.sum(error_hidden1, axis=0)

        # Regularization
        if self.lmda > 0.0:
            for i in range(self.n_layers):
            delta_w_output += self.lmda * self.w_output
            delta_w_hidden2 += self.lmda * self.w_hidden2
            delta_w_hidden1 += self.lmda * self.w_hidden1

        # Update weights and biases
        self.w_output -= self.eta * delta_w_output
        self.b_output -= self.eta * delta_b_output

        self.w_hidden2 -= self.eta * delta_w_hidden2
        self.b_hidden2 -= self.eta * delta_b_hidden2

        self.w_hidden1 -= self.eta * delta_w_hidden1
        self.b_hidden1 -= self.eta * delta_b_hidden1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        y_pred = self.feed_forward(x)[2]
        return y_pred

    def accuracy(self, x=None, y=None):
        """
        Calculate the accuracy score for the network.
        :param x: test / validation data to generate output with
        :param y: test / validation targets to check with
        :return: accuracy score
        """

        if x is not None and y is not None:
           y = y.reshape((len(y), 1))
           output = self.predict(x)
           score = np.sum(y == np.round(output)) / len(y)
        else:
           output = self.predict(self.x_test)
           score = np.sum(self.y_test == np.round(output))/len(self.y_test)
        
        # #Run with sklearn's accuracy score.
        # if x is not None and y is not None:
        #     y = y.reshape((len(y), 1))
        #     output = np.round(self.feed_forward(x)[2])
        #     score = accuracy_score(y, output)
        # else:
        #     output = np.round(self.feed_forward(self.x_test)[2])
        #     score = accuracy_score(self.y_test, output)

        return score
