import numpy as np
from tqdm import tqdm
from ols import OrdinaryLeastSquares


class LogisticRegression():
    """
    Perform logistic regression  on a data set y, to fit a set of
    weights, beta.
    """

    def __init__(self, eta = 1e-2, intercept = True, epochs = 1e2, batch_size=100):
        #:param eta: How fast should we learn?
        #:param intercept: boolean - Specify if a constant should be added
        #:param n_iter: number of iterations

        # Initialize x based on intercept boolean

        self.eta = eta
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

        exp_predict = np.exp(y)
        return exp_predict/(1 + exp_predict)

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
            for j in range(self.n_iter):
                chosen_indices = np.random.choice(
                    indices, size=self.batch_size, replace=False)

                # Batch the training data and targets
                self.inputs = self.inputs_full[chosen_indices]
                self.targets = self.targets_full[chosen_indices]

                y_predict = self.sigmoid(np.matmul(self.inputs, self.weights))

                gradient = -gradient_scale * np.matmul(self.inputs.T,
                                                       self.targets - y_predict)

                self.weights -= self.eta * gradient
                tmp_cost.append(self.cross_entropy(self.inputs_full, self.targets_full))

            self.accuracies.append(self.accuracy(self.inputs_full, self.targets_full))
            self.cost_function.append(tmp_cost)

    def fit(self, x, y):
        """
        Fit the weights to inputs.
        """
        if self.intercept:
            self.inputs_full = np.c_[np.ones(x.shape[0]), x]
        else:
            self.inputs_full = x

        self.targets_full = y
        self.n_inputs = self.inputs_full.shape[0]
        self.n_features = self.inputs_full.shape[1]
        self.n_iter = self.n_inputs // self.batch_size

        # Initialize weights using OLS regression
        # ols = OrdinaryLeastSquares()
        # ols.fit_coefficients(self.inputs_full, self.targets_full)
        # self.weights = ols.coeff

        # Initializing weights randomly
        self.weights = np.random.randn(self.n_features)

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

        term = np.matmul(input, self.weights)
        ce = -np.sum(targets * term - np.log(1 + np.exp(term)))

        return ce

    def accuracy(self, x, y, threshold = 0.5):
        """
        Evaluate the accuracy of the model based on the number of correctly
        labeled classes divided by the number of classes in total.
        """
        y_predict = self.predict(x)

        check = []
        for el in y_predict:
            if el >= threshold:
                check.append(1)
            else:
                check.append(0)

        correct = np.array(check, dtype=int) - y
        count = 0
        for el in correct:
            if el == 0: count += 1
        score = count / correct.size

        return score