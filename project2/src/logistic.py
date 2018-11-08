import numpy as np
from tqdm import tqdm
from ols import OrdinaryLeastSquares


class LogisticRegression():
    """
    Perform logistic regression  on a data set y, to fit a set of
    weights, beta.
    """

    def __init__(self, learning_rate = 5e-4, intercept = True, num_iter = 1.5e2):
        """

        :param learning_rate: How fast should we learn?
        :param intercept: boolean - Specify if a constant should be added
        :param num_iter: number of iterations
        """

        self.learning_rate = learning_rate
        self.intercept = intercept
        self.num_iter = int(num_iter)
        self.x = None
        self.y = None
        self.weights = None

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
        Use gradient descent to optimize the weights.
        Calculate the gradient of the cost function (cross entropy).
        :return: gradient
        TODO: Add some tolerance condition for when to stop.
        """

        for i in tqdm(range(self.num_iter)):
            y_predict = self.make_prediction(self.x)

            # Dividing gradient by number of states to prevent overflow/nan.
            # TODO: Figure out why this happens?
            gradient = -np.dot(self.x.T, self.y - y_predict)/self.x.shape[0]

            self.weights += self.learning_rate * gradient

    def fit(self, x, y):
        """
        :param x: x values that generated the data
        :param y: true values for x
        """

        # Initialize x based on intercept boolean
        if self.intercept:
            self.x = np.c_[np.ones(x.shape[0]), x]
        else:
            self.x = x
        self.y = y

        # Initialize weights using OLS regression
        print("Initializing weights...")
        ols = OrdinaryLeastSquares()
        ols.fit_coefficients(self.x, self.y)
        self.weights = ols.coeff
        #self.weights = np.random.uniform(1e-4, 0.1, self.x.shape[1])

        # Gradient descent to optimize weights
        print("Performing gradient descent...")
        self.gradient_descent()

        return self.weights

    def make_prediction(self, x):
        """
        Generate a set of probabilites for some new input x.
        :param x: Input values to predict new data for
        :return: predicted values
        """
        while x.shape != self.x.shape:
            x = np.c_[np.ones(x.shape[0]), x]

        y_predict = np.dot(x, self.weights)
        y_predict = self.sigmoid(y_predict)

        return y_predict

    def cross_entropy(self):
        """
        Calculate the cross-entropy (negative log likelihood).
        :return: cross-entropy
        """

        y_predict = self.make_prediction(self.x)
        ce = -np.sum(self.y * y_predict - np.log(1 + np.exp(y_predict)))

        return ce

    def accuracy(self, x, y, threshold = 0.5):
        """
        Evaluate the accuracy of the model based on the number of correctly
        labeled classes divided by the number of classes in total.
        """

        y_predict = self.make_prediction(x)
        check = []
        for el in y_predict:
            if el >= threshold:
                check.append(1)
            else:
                check.append(0)

        correct = np.array(check) - y
        score = (correct.size - np.count_nonzero(correct))/correct.size

        return score