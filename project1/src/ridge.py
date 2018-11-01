import numpy as np


class RidgeRegression:

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

        self.coeff = (np.linalg.inv(x.T @ x + lmb * I) @ x.T @ y)

    def make_prediction(self, x):
        """
        Use the trained model to predict values given an input x.

        :param x: Input values to predict new data for
        :return: predicted values
        """

        y_predict = x @ self.coeff

        return y_predict

    def mean_squared_error(self, x, y):
        """
        Evaluate the mean squared error of the output generated
        by Ordinary Least Squares regressions.

        :param x: x-values for data to calculate mean_squared error on.
        :param y: true values for x
        :return: returns mean squared error for the fit compared with true values.
        """

        y_predict = self.make_prediction(x)
        mse = np.mean(np.square(y - y_predict))

        return mse

    def r2_score(self, x, y):
        """
        Evaluates the R2 score for the fitted model.

        :param x: input values x
        :param y: true values for x
        :return: r2 score
        """
        y_predict = self.make_prediction(x)

        y_mean = np.mean(y)
        upper_sum = np.sum(np.square(y - y_predict))
        lower_sum = np.sum(np.square(y - y_mean))
        r2score = 1 - upper_sum / lower_sum
        return r2score
