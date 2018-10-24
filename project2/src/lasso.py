import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


class LassoRegression():

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

        self.x = None
        self.poly_degree = None
        self.y = None
        self.coeff = None
        self.poly = None
        self.alpha = None
        self.lasso_object = None

    def fit_coefficients(self, x, y, poly_degree, alpha=1.0):
        """
        Fits the polynomial coefficients beta to the matrix
        of polynomial features.

        :param x: input values that generated the dataset y
        :param y: the dataset corresponding to input values x
        :param poly_degree: Degree of the polynomial to fit
        :param alpha: float, Constant that multiplies the L1 term.
                        Defaults to 1.0. alpha = 0 is equivalent to
                        an ordinary least square regression.
        """

        self.poly_degree = poly_degree
        self.poly = PolynomialFeatures(poly_degree)
        self.y = y
        self.alpha = alpha

        # Regression
        # Input values to design matrix
        self.x = self.poly.fit_transform(x)
        
        self.lasso_object = linear_model.Lasso(alpha=self.alpha, max_iter=1e3)
        self.lasso_object.fit(self.x, self.y)

    def make_prediction(self, x):
        """
        Makes a model prediction.
        Demands both x and z- values in order to update the target values
        (self.outcome) for MSE calculations. This can probably be done better.

        :param x: input values to generate predicted data set for
        :returns: predicted values
        """
        X = self.poly.fit_transform(x)
        y_predict = self.lasso_object.predict(X)

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

    def r2_score(self):
        """
        Evaluate the R2 score of the model fitted to the dataset.

        :return:    Returns the mean squared error of the model compared with
                    dataset used for fitting.
        """

        outcome_mean = np.mean(self.y)
        upper_sum = np.sum(np.square(self.y - self.predicted_outcome))
        lower_sum = np.sum(np.square(self.y - outcome_mean))
        r2score = 1 - upper_sum / lower_sum
        return r2score
