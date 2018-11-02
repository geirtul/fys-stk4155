import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class RidgeRegression(Analysis):

    def __init__(self):
        
        """
        Perform regression using the ridge method
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
        self.poly_degree = None
        self.y = None
        self.coeff = None
        self.poly = None
        self.predicted_outcome = None
        self.lmb = None

    def fit_coefficients(self, x, y, poly_degree, lmb=0):
        """
        Fits the polynomial coefficients beta to the matrix
        of polynomial features.

        :param x: x values that generated the outcome
        :param y: data corresponding to x
        :param poly_degree: Degree of the polynomial to fit
        :param lmb: float, shrinkage lmb=0 makes the model equal to ols
        """

        self.x = x
        self.poly_degree = poly_degree
        self.poly = PolynomialFeatures(poly_degree)
        self.y = y
        self.lmb = lmb

        # Regression
        X = self.poly.fit_transform(self.x)

        I = np.eye(len(X[1]))

        self.coeff = (np.linalg.inv(X.T @ X + lmb * I) @ X.T @ y)

    def make_prediction(self, x):
        """
        Makes a model prediction
        Returns prediction together with x and y values for plotting.

        :param x: x values to generate data values for.
        :returns: predicted data
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
