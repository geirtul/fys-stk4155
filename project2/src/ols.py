import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class OrdinaryLeastSquares():

    def __init__(self):
        """
        Perform linear regression using the Ordinary Least Squares method
        on a dataset y, with a polynomial of degree m.
        The PolynomialFeatures module from scikit learn sets up the
        vandermonde matrix such that in the matrix equation X*beta = y,
        beta is the coefficient vector,
        and X contains the polynomial expressions.
        returns x and y values for plotting along with the predicted y values
        from the model.

        Sets up the matrix X in the matrix equation y = X*Beta
        and performs regression
        """

        self.x = None
        self.y = None
        self.poly_degree = None
        self.poly = None
        self.coeff = None

    def fit_coefficients(self, x, y, poly_degree):
        """
        Fits the polynomial coefficients beta to the matrix
        of polynomial features.

        :param x: x values that generated the outcome
        :param y: the dataset to fit
        :param poly_degree: Degree of the polynomial to fit
        """
        self.x = x
        self.poly_degree = poly_degree
        self.poly = PolynomialFeatures(poly_degree)
        self.y = y

        # Regression
        X = self.poly.fit_transform(self.x)
        self.coeff = np.linalg.inv(X.T @ X) @ X.T @ y

    def make_prediction(self, x):
        """
        Use the trained model to predict values given an input x_in.


        :param x: Input values to predict new data for
        :return: predicted values
        """

        X = self.poly.fit_transform(x)
        y_predict = X @ self.coeff

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