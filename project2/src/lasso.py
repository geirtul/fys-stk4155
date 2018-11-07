from sklearn import linear_model
from analysis import Analysis


class LassoRegression(Analysis):

    def __init__(self, alpha = 1.0):
        """
        Perform regression using the Lasso method on a
        data set y. The method is scikit learn's Lasso implementation
        put into the same framework as the OLD and Ridge methods here.
        
        :param alpha: float, Constant that multiplies the L1 term.
                        Defaults to 1.0. alpha = 0 is equivalent to
                        an ordinary least square regression.
        """

        self.x = None
        self.y = None
        self.coeff = None
        self.alpha = alpha
        self.lasso_object = linear_model.Lasso(alpha=self.alpha)

    def fit_coefficients(self, x, y):
        """
        Fits coefficients to the matrix.

        :param x: input values that generated the dataset y
        :param y: the dataset corresponding to input values x
        """

        self.y = y

        # Regression
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

    def mean_squared_error(self, x, y):
        """
        Evaluate the mean squared error of the output generated
        by regressions.

        :param x:   x-values for data to calculate mean_squared error on.
        :param y:   true values for x
        :return:    returns mean squared error for the fit compared with true
                    values.
        """

        y_predict = self.lasso_object.predict(x)
        mse = np.mean(np.square(y - y_predict))

        return mse

    def r2_score(self, x, y):
        """
        Evaluates the R2 score for the fitted model.
        Uses scikit's own built-in r2 score

        :param x:   input values x
        :param y:   true values for x
        :return:    r2 score
        """

        return self.lasso_object.score(x, y)
