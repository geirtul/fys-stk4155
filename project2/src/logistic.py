import numpy as np
from tqdm import tqdm


class LogisticRegression():
    """
    Perform logistic regression  on a data set y, to fit a set of
    weights, beta.
    """

    def __init__(self, eta = 1e-2, gamma = 0.01, intercept = True, epochs = 1e2, batch_size=100):
        #:param eta: How fast should we learn?
        #:param intercept: boolean - Specify if a constant should be added
        #:param n_iter: number of iterations

        # Initialize x based on intercept boolean

        self.eta = eta
        self.gamma = gamma
        self.intercept = intercept
        self.epochs = int(epochs)
        self.batch_size = batch_size

        self.weights = None
        self.accuracies = []
        self.cost_function = []

    def sigmoid(self, y):
        """
        Use the sigmoid function to generate a set of predicted
        probabilites.

        :return: predicted probabilites based on the sigmoid function
        """

        return 1/(1 + np.exp(-y))

    def gradient_descent(self):
        """
        Use stochastic gradient descent to optimize the weights.
        Calculate the gradient of the cost function (cross entropy).
        """

        # Run mini-batches for the stochastic gradient descent

        indices = np.arange(self.n_inputs)
        gradient_scale = (1 / self.n_iter)

        for i in tqdm(range(self.epochs)):
            tmp_cost = []
            prev_gradient = 0
            for j in range(self.n_iter):
                chosen_indices = np.random.choice(
                    indices, size=self.batch_size, replace=True)

                # Batch the training data and targets
                self.inputs = self.inputs_full[chosen_indices]
                self.targets = self.targets_full[chosen_indices]

                scores = np.dot(self.inputs, self.weights)
                y_predict = self.sigmoid(scores)

                gradient = gradient_scale * np.dot(self.inputs.T,
                                                   self.targets - y_predict)

                # update weights including momentum (gamma parameter)
                self.weights += self.gamma*prev_gradient + self.eta * gradient
                prev_gradient = gradient

            # Store accuracy on test set for plotting.
            self.accuracies.append(self.accuracy(self.x_test, self.y_test))

    def fit(self, x, y, x_test, y_test):
        """
        Fit the weights to inputs.
        """

        # Initialize input arrays and constants
        if self.intercept:
            self.inputs_full = np.c_[np.ones(x.shape[0]), x]
        else:
            self.inputs_full = x

        self.targets_full = y
        self.n_inputs = self.inputs_full.shape[0]
        self.n_features = self.inputs_full.shape[1]
        self.n_iter = self.n_inputs // self.batch_size

        # Initializing weights randomly
        #self.weights = np.random.randn(self.n_features)
        self.weights = np.zeros(self.n_features)

        self.x_test = x_test
        self.y_test = y_test

        # Gradient descent to optimize weights
        self.gradient_descent()

        return self.weights

    def predict(self, x):
        """
        Generate a set of probabilites for some new input x.
        :param x: Input values to predict new data for
        :return: predicted values
        """
        while x.shape[1] != self.inputs_full.shape[1]:
            x = np.c_[np.ones(x.shape[0]), x]
        y_predict = self.sigmoid(np.matmul(x, self.weights))

        return y_predict

    def cross_entropy(self, input, targets):
        """
        Calculate the cross-entropy (negative log likelihood).
        :return: cross-entropy
        """
        scores = np.dot(input, self.weights)
        log_likelihood = np.sum(targets*scores - np.log(1 + np.exp(scores)))
        ce = -log_likelihood

        return ce

    def accuracy(self, x, y):
        """
        Evaluate the accuracy of the model based on the number of correctly
        labeled classes divided by the number of classes in total.
        """
        y_predict = np.round(self.predict(x))
        score = np.sum(y_predict == y) / len(y)

        return score