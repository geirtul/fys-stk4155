import numpy as np


class Analysis:

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
    
    def bootstrap(self, x, y):
        raise NotImplemented
