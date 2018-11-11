import numpy as np
from tqdm import tqdm


class MLP:
    def __init__(self,
                 inputs,
                 targets,
                 test_inputs,
                 test_targets,
                 n_hidden=50,
                 n_classes=2,
                 epochs=10,
                 batch_size=100,
                 eta=0.01,
                 lmd=0.0,
                 ):
        """
        Initialize the network. This MLP network has one hidden layer, and is
        thus a Single Layer Perceptron.Heavily inspired by lecture notes on
        Neural Networks:
        https://compphysics.github.io/MachineLearning/doc/pub/NeuralNet/pdf/NeuralNet-minted.pdf

        :param inputs: training inputs
        :param targets: training targets
        :param n_hidden: number of hidden nodes in the network
        :param n_classes: number of categories/classes for data.
        :param epochs: number of times the training data will be fed through network.
        :param eta: Learning rate
        :param lmd: Regularization parameter
        """
        self.eta = eta  # Learning rate
        self.lmb = lmd  # Regularization parameter
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_inputs = inputs.shape[0]
        self.n_features = inputs.shape[1]
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.iterations = self.n_inputs // self.batch_size

        self.inputs_full = inputs
        self.targets_full = self.handle_targets_shape(targets)
        self.test_inputs = test_inputs
        self.test_targets = test_targets
        self.setup_weights_biases_targets()
        self.accuracies_test = []

    def train(self):
        """
        Trains the network. Runs the backwards phase of the training algorithm
        to adjust the weights. The network outputs probabilities for
        n_classes - 1, thus assuming that the probability of the last class is
        1 - sum(the rest).
        """
        indices = np.arange(self.n_inputs)

        # Train with mini-batches.
        for i in tqdm(range(self.epochs)):
            for j in range(self.iterations):
                chosen_indices = np.random.choice(
                    indices, size=self.batch_size, replace=False)

                # Batch the training data
                self.inputs = self.inputs_full[chosen_indices]
                self.targets = self.targets_full[chosen_indices]

                self.forward()
                self.backwards()
            # For each epoch, store the accuracy of the network.
            curr_acc = self.accuracy_score(self.test_inputs, self.test_targets)
            self.accuracies_test.append(curr_acc)

    def forward(self):
        """
        Feed inputs forward through the network.
        :return: activations in hidden layer and probabilities in output layer.
        """

        # Calculate activations in the hidden layer
        self.z_hidden = np.matmul(self.inputs, self.weights_hidden) + self.bias_hidden
        self.a_hidden = self.sigmoid(self.z_hidden)

        # Calculate output probabilities
        self.z_output = np.matmul(self.a_hidden, self.weights_output) + self.bias_output
        exp_term = np.exp(self.z_output)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def forward_output(self, inputs):
        """
        The same as self.forward(), but takes input in order to return
        probabilities for test datasets.
        :param inputs: test data to predict classification for.
        :return: probabilities
        """

        z_hidden = np.matmul(inputs, self.weights_hidden) + self.bias_hidden
        a_hidden = self.sigmoid(z_hidden)

        # Calculate output probabilities
        z_output = np.matmul(a_hidden, self.weights_output) + self.bias_output
        exp_term = np.exp(z_output)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

        return probabilities

    def backwards(self):
        """
        Backwards propagation of error, updating weights accordingly.
        """

        error_output = self.probabilities - self.targets
        error_hidden = np.matmul(error_output, self.weights_output.T) * self.a_hidden * (1 - self.a_hidden)

        # Gradients for the output layer weights and bias
        dWo = np.matmul(self.a_hidden.T, error_output)  # output_weights_gradient
        dBo = np.sum(error_output, axis=0)  # output_bias_gradient

        # Gradients for the hidden layer weights and bias
        dWh = np.matmul(self.inputs.T, error_hidden)  # hidden_weights_gradient
        dBh = np.sum(error_hidden, axis=0)  # hidden_bias_gradient

        # Regularization term gradients
        if self.lmb > 0.0:
            dWo += self.lmb * self.weights_output
            dWh += self.lmb * self.weights_hidden

        # Update weights and biases
        self.weights_hidden -= self.eta * dWh
        self.bias_hidden -= self.eta * dBh
        self.weights_output -= self.eta * dWo
        self.bias_output -= self.eta * dBo

    def setup_weights_biases_targets(self):
        """
        Initializes randomized weights, sets up bias.
        """
        weights_scale = np.sqrt(2/self.n_inputs)
        # Initialize randomized weights
        self.weights_hidden = np.random.randn(self.n_features, self.n_hidden)*weights_scale
        self.weights_output = np.random.randn(self.n_hidden, self.n_classes)*weights_scale

        # Bias
        self.bias_hidden = np.zeros(self.n_hidden) + 0.01
        self.bias_output = np.zeros(self.n_classes) + 0.01

    def sigmoid(self, x):
        """
        Calculate sigmoid of x.
        """
        return 1 / (1 + np.exp(-x))

    def handle_targets_shape(self, targets):
        """
        Handles input targets with shape (N, ), turning them into (N,1)
        to avoid issues with matrix / vector multiplication.
        :param targets: The targets to reshape if necessary
        :return: reshaped targets
        """

        if len(targets.shape) == 1:
            targets = targets.reshape(len(targets), 1)
            return targets
        else:
            return targets

    def predict(self, test_input):
        probabilites = self.forward_output(test_input)
        return np.argmax(probabilites, axis=1)

    def predict_probabilities(self, inputs):
        """
        Predict the probabilities of classes for a given input.
        :param inputs: The input to be fed through the network.
        :return: Predicted probabilities for classification.
        """
        probabilities = self.forward_output(inputs)
        return probabilities

    def accuracy_score(self, test_inputs, test_targets):
        """
        Calculate the current accuracy score of the network.
        """
        prediction = self.predict(test_inputs)

        return np.sum(test_targets == prediction) / len(test_targets)
