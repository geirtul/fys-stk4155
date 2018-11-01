import numpy as np
from sklearn import linear_model
from analysis import Analysis


class LassoRegression(Analysis):

    def __init__(self):
        """
        Perform regression using the Lasso method on a
        data set y. The method is scikit learn's Lasso implementation
        put into the same framework as the OLD and Ridge methods here.
        """

        self.x = None
        self.y = None
        self.coeff = None
        self.alpha = None
        self.lasso_object = None

    def fit_coefficients(self, x, y, alpha=1.0):
        """
        Fits coefficients to the matrix.

        :param x: input values that generated the dataset y
        :param y: the dataset corresponding to input values x
        :param alpha: float, Constant that multiplies the L1 term.
                        Defaults to 1.0. alpha = 0 is equivalent to
                        an ordinary least square regression.
        """

        self.y = y
        self.alpha = alpha

        # Regression
        # Input values to design matrix
        
        self.lasso_object = linear_model.Lasso(alpha=self.alpha, max_iter=1e3)
        self.lasso_object.fit(x, y)

    def make_prediction(self, x):
        """
        Makes a model prediction.
        Demands both x and z- values in order to update the target values
        (self.outcome) for MSE calculations. This can probably be done better.

        :param x: input values to generate predicted data set for
        :returns: predicted values
        """

        y_predict = self.lasso_object.predict(x)

        return y_predict
