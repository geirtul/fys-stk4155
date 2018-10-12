import numpy as np
from franke_function import FrankeFunction
from sklearn.preprocessing import PolynomialFeatures
from analysis import Analysis


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

        self.predictors = None
        self.poly_degree = None
        self.outcome = None
        self.beta = None
        self.poly = None
        self.predicted_outcome = None
        self.lmb = None

    def fit_coefficients(self, predictors, outcome, poly_degree, lmb=0):
        """
        Fits the polynomial coefficients beta to the matrix
        of polynomial features.

        :param predictors: x,y, ... values that generated the outcome
        :param outcome: the dataset we will fit
        :param poly_degree: Degree of the polynomial to fit
        :param lmb: float, shrinkage lmb=0 makes the model equal to ols
        """
        self.predictors = predictors
        self.poly_degree = poly_degree
        self.poly = PolynomialFeatures(poly_degree)
        self.outcome = outcome
        self.lmb = lmb

        # Regression
        X = self.poly.fit_transform(self.predictors)

        I = np.eye(len(X[1]))

        self.beta = (np.linalg.inv(X.T @ X + lmb * I) @ X.T @ outcome)

    def make_prediction(self, x_in):
        """
        Makes a model prediction
        Returns prediction together with x and y values for plotting. 
        """
        X = self.poly.fit_transform(x_in)
        self.predicted_outcome = X @ self.beta

        return self.predicted_outcome


if __name__ == "__main__":
    # Data from FrankeFunction
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    z = FrankeFunction(x, y).ravel()

    # Make predictor values a matrix with
    # number of columns = number of predictors.
    predictors_input = np.c_[x.ravel(), y.ravel()]

    """
    noiseRange = 1
    noise = noiseRange*np.random.uniform(-0.5, 0.5, size = z.shape)
    z = z + noise
    """
    lmb_values = [0, 1e-4, 1e-3, 1e-2, 10, 1e2, 1e4]
    
    for lmb in lmb_values:
        print("Running Ridge Regression with lambda = {}\n".format(lmb))
        ridge = RidgeRegression() 
        ridge.fit_coefficients(predictors_input, z, 5, lmb)

        ridge.make_prediction(predictors_input)
        # ridge.bootstrap()
        # ridge.plotting_3d()
        mse = ridge.mean_squared_error()
        r2 = ridge.r2_score()
        print("MSE = {:f} for lmd = {:f}".format(mse, lmb))
        print("R2 = {:f} for lmd = {:f}".format(r2, lmb))
        print("===================================\n")
