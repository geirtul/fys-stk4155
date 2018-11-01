import numpy as np
from analysis import Analysis


class OrdinaryLeastSquares(Analysis):

    def __init__(self):
        """
        Perform regression using the ordinary least squares method on a
        data set y.
        Sets up the matrix X in the matrix equation y = X*Beta
        and performs linear regression to find the best coefficients.
        """

        self.x = None
        self.y = None
        self.coeff = None

    def fit_coefficients(self, x, y):
        """
        Makes a linear fit of the data.

        :param x: x values that generated the outcome
        :param y: the dataset to fit
        """
        self.x = x
        self.y = y

        # Regression
        self.coeff = np.linalg.inv(x.T @ x) @ x.T @ y

    def make_prediction(self, x):
        """
        Use the trained model to predict values given an input x.

        :param x: Input values to predict new data for
        :return: predicted values
        """

        y_predict = x @ self.coeff

        return y_predict
