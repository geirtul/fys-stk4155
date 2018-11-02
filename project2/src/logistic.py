import numpy as np
from analysis import Analysis


class LogisticRegression(Analysis):

    def __init__(self):
        """
        Perform logistic regression  on a data set y.
        """

        self.x = None
        self.y = None
        self.coeff = None

    def fit_coefficients(self, x, y):
        """
        make s a fit of probabilities.

        :param x: x values that generated the outcome
        :param y: the dataset to fit
        """
        self.x = x
        self.y = y

        # Regression
        self.coeff = np.linalg.pinv(x.T @ x) @ x.T @ y

    def make_prediction(self, x):
        """
        Use the trained model to predict values given an input x.

        :param x: Input values to predict new data for
        :return: predicted values
        """

        y_predict = x @ self.coeff

        return y_predict
