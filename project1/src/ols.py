import numpy as np
from franke_function import FrankeFunction
from sklearn.preprocessing import PolynomialFeatures
from analysis import Analysis


class OrdinaryLeastSquares(Analysis):

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

        self.predictors = None
        self.poly_degree = None
        self.outcome = None
        self.beta = None
        self.poly = None
        self.predicted_outcome = None

    def fit_coefficients(self, predictors, outcome, poly_degree):
        """
        Fits the polynomial coefficients beta to the matrix
        of polynomial features.

        :param predictors: x,y, ... values that generated the outcome
        :param outcome: the dataset we will fit
        :param poly_degree: Degree of the polynomial to fit
        """
        self.predictors = predictors
        self.poly_degree = poly_degree
        self.poly = PolynomialFeatures(poly_degree)
        self.outcome = outcome

        # Regression
        X = self.poly.fit_transform(self.predictors)  # predictors values to design matrix
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ outcome

    def make_prediction(self, x_in):
        """
        Makes a model prediction
        Returns prediction together with x and y values for plotting.

        :param x_in: predictors to generate an outcome with
        """
        X = self.poly.fit_transform(x_in)
        self.predicted_outcome = X @ self.beta
        return self.predicted_outcome


if __name__ == "__main__":
    # Data from FrankeFunction
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    # Make predictor values a matrix with number of columns = number of predictors.
    # TODO: Need better input handling. Number of predictors shouldn't matter.
    predictors_input = np.c_[x.ravel(), y.ravel()]

    z = FrankeFunction(x, y).ravel()

    # Initialize the regression object and perform a fit
    ols = OrdinaryLeastSquares()
    ols.fit_coefficients(predictors_input, z, 5)
    z_predict = ols.make_prediction(predictors_input)

    print("MSE = ", ols.mean_squared_error())
    print("R2 score = ", ols.r2_score())

    ols.bootstrap()
