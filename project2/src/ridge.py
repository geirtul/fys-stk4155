import numpy as np


class RidgeRegression():

    def __init__(self, lmb):
        
        """
        Perform regression using the ridge method on a dataset y,
        with a polynomial of degree m.
        TODO: Add better description of the method, or link to it.
        
        :param lmb: float, shrinkage lmb=0 makes the model equal to ols
        """

        self.x = None
        self.y = None
        self.coeff = None
        self.lmb = lmb

    def fit_coefficients(self, x, y):
        """
        Fits the polynomial coefficients beta to the matrix
        of polynomial features.

        :param x: x values that generated the outcome
        :param y: data corresponding to x
        """

        self.x = x
        self.y = y

        # Regression
        I = np.eye(len(x[1]))

        self.coeff = (np.linalg.pinv(x.T @ x + self.lmb * I) @ x.T @ y)

    def make_prediction(self, x):
        """
        Makes a model prediction
        Returns prediction together with x and y values for plotting.

        :param x: x values to generate data values for.
        :returns: predicted data
        """
        y_predict = x @ self.coeff

        return y_predict

    def mean_squared_error(self, x, y):
        """
        Evaluate the mean squared error of the output generated
        by Ordinary Least Squares regressions.

        :param x:   x-values for data to calculate mean_squared error on.
        :param y:   true values for x
        :return:    returns mean squared error for the fit compared with true
                    values.
        """

        y_predict = self.make_prediction(x)
        mse = np.mean(np.square(y - y_predict))

        return mse

    def r2_score(self, x, y):
        """
        Evaluates the R2 score for the fitted model.

        :param x:   input values x
        :param y:   true values for x
        :return:    r2 score
        """
        y_predict = self.make_prediction(x)

        y_mean = np.mean(y)
        upper_sum = np.sum(np.square(y - y_predict))
        lower_sum = np.sum(np.square(y - y_mean))
        r2score = 1 - upper_sum / lower_sum
        return r2score
