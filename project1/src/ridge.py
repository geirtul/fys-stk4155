import numpy as np
from analysis import Analysis


class RidgeRegression(Analysis):

    def __init__(self):
        
        """
        Perform regression using the ridge method on a data set y.
        Sets up the matrix X in the matrix equation y = X*Beta
        and performs linear regression to find the best coefficients.
        """

        self.x = None
        self.y = None
        self.coeff = None

    def fit_coefficients(self, x, y, lmb=0):
        """
        Makes a linear fit of the data given the input x.

        :param x: x values that generated the dataset
        :param y: the dataset we will fit
        :param lmb: float, shrinkage lmb=0 makes the model equal to ols
        """
        self.x = x
        self.y = y

        # Regression
        I = np.eye(len(x[1]))

        self.coeff = (np.linalg.pinv(x.T @ x + lmb * I) @ x.T @ y)

    def make_prediction(self, x):
        """
        Use the trained model to predict values given an input x.

        :param x: Input values to predict new data for
        :return: predicted values
        """

        y_predict = x @ self.coeff

        return y_predict
