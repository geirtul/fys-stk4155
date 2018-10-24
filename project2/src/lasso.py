import numpy as np
from franke_function import FrankeFunction 
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from analysis import Analysis
from sklearn.model_selection import train_test_split


class LassoRegression(Analysis):

    def __init__(self):
        """
        Perform regression using the Lasso method
        on a dataset y, with a polynomial of degree poly_degree.
        The PolynomialFeatures module from scikit learn sets up the 
        vandermonde matrix such that in the matrix equation X*beta = y, 
        beta is the coefficient vector, and X contains the polynomial
        expressions.

        The method uses scikit learn's Lasso regression methods.
        """

        self.predictors = None
        self.poly_degree = None
        self.outcome = None
        self.beta = None
        self.poly = None
        self.predicted_outcome = None
        self.alpha = None
        self.lasso_object = None

    def fit_coefficients(self, predictors, outcome, poly_degree, alpha=1.0):
        """
        Fits the polynomial coefficients beta to the matrix
        of polynomial features.

        :param predictors: x,y, ... values that generated the outcome
        :param outcome: the dataset we will fit
        :param poly_degree: Degree of the polynomial to fit
        :param alpha: float, Constant that multiplies the L1 term.
                        Defaults to 1.0. alpha = 0 is equivalent to
                        an ordinary least square regression.
        """

        self.poly_degree = poly_degree
        self.poly = PolynomialFeatures(poly_degree)
        self.outcome = outcome
        self.alpha = alpha

        # Regression
        # Input values to design matrix
        self.predictors = self.poly.fit_transform(predictors)
        
        self.lasso_object = linear_model.Lasso(alpha=self.alpha, max_iter=1e3)
        self.lasso_object.fit(self.predictors, self.outcome)

    def make_prediction(self, x_in, z_in):
        """
        Makes a model prediction.
        Demands both x and z- values in order to update the target values
        (self.outcome) for MSE calculations. This can probably be done better.
        """
        X = self.poly.fit_transform(x_in)
        self.predicted_outcome = self.lasso_object.predict(X)
        self.outcome = z_in

        return self.predicted_outcome
