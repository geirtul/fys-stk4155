import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


class Resampling:
    """
    Class containing resampling methods to be used with regression classes.
    """

    def bootstrap(self, num_bootstraps = 100, test_size = 0.2):
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
                            Defaults to 0.2.

        :return: y_predictions - predicted values for each resampled data set.
        """

        # Split the data the model was initially fit on into training and test
        # sets. Recommended test_size is around 0.2, so we'll stick to that.
        #indices = np.arange(len(self.y))

        x_train, x_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=test_size)

        y_pred = np.empty((y_test.shape[0], num_bootstraps))
        print("Bootstrapping...")
        coeffs = np.zeros((num_bootstraps, x_train.shape[1]))
        for i in tqdm(range(num_bootstraps)):
            x_re, y_re = resample(x_train, y_train)

            self.fit_coefficients(x_re, y_re)
            y_pred[:, i] = self.make_prediction(x_test)

            # Store coefficients
            coeffs[i] = self.coeff

        # Analysis
        # Need y_test[:, None] for proper broadcasting of arrays.
        error = np.mean(np.mean((y_test[:, None] - y_pred) ** 2,
                                axis=1, keepdims=True))
        bias = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
        variance = np.mean(np.var(y_pred, axis=1, keepdims=True))

        # Estimate the variance in the parameters beta
        coeff_variance = np.var(coeffs, axis=0)
        coeff_mean = np.mean(coeffs, axis=0)
        
        output = [error, bias, variance] + coeff_mean.tolist() + coeff_variance.tolist()

        return output
