import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


class Resampling:
    """
    Class containing resampling methods to be used with regression classes.
    """

    def bootstrap(self, num_bootstraps = 100, test_size = 0.1, datasize = 100):
        """
        Note:   This code is heavily inspired by the Piazza post
                "On Bias and Variance" from project 1.
                https://piazza.com/class/ji78s1cduul39a?cid=59

        Perform bootstrapping on the dataset the regression model has been
        fit to. It is assumed that the regression class has been initialized
        with whatever parameters necessary (alpha, lmb etc).

        :param num_bootstraps:  Number of random draws to make.
                                Defaults to 100.
        :param test_size:   Relative size of test_data compared to full data.
                            Defaults to 0.1.

        :return: y_predictions - predicted values for each resampled data set.
        """

        # Split the data the model was initially fit on into training and test
        # sets. Recommended test_size is around 0.1, so we'll stick to that.
        indices = np.arange(len(self.y))
        chosen_indices = np.random.choice(
                indices, size=datasize, replace=False)
        data_x = self.x[chosen_indices]
        data_y = self.y[chosen_indices]

        x_train, x_test, y_train, y_test = train_test_split(data_x,
                                                            data_y,
                                                            test_size=test_size)

        y_pred = np.empty((y_test.shape[0], num_bootstraps))
        print("Bootstrapping...")
        for i in tqdm(range(num_bootstraps)):
            x_re, y_re = resample(x_train, y_train)

            # Fit the model to the resampled data. Updates coefficients
            # so we can use make_prediction.
            self.fit_coefficients(x_re, y_re)
            y_pred[:, i] = self.make_prediction(x_test)

        # Analysis
        # Need y_test[:, None] for proper broadcasting of arrays.
        error = np.mean(np.mean((y_test[:, None] - y_pred) ** 2,
                                axis=1, keepdims=True))
        bias = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
        variance = np.mean(np.var(y_pred, axis=1, keepdims=True))

        #print('Error:', error)
        #print('Bias^2:', bias)
        #print('Var:', variance)
        #print(
        #    '{} >= {} + {} = {}'.format(error, bias, variance, bias + variance))

        return [error, bias, variance]
