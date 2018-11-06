import numpy as np
from analysis import Analysis


class LogisticRegression(Analysis):

    def __init__(self):
        """
        Perform logistic regression  on a data set y.
        """

        self.x = None
        self.beta = None

    def fit_coefficients(self, x, beta):
        """
        makes a fit of probabilities.

        :param x: x values that generated the outcome
        :param beta: the set of weights to optimize
        """
        self.x = x
        self.beta = beta

        # Regression

    def make_prediction(self, x):
        """

        :param x: Input values to predict new data for
        :return: predicted values
        """

        #y_predict = x @ self.coeff

        return y_predict
