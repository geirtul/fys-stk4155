import numpy as np
from numpy.random import randint, randn
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


class Resampling:
    """
    Class containing resampling methods to be used with regression classes.
    """

    def bootstrap(self, num_bootstraps = 1000, test_size = 0.1):
        """
        Perform bootstrapping on the dataset the regression model has been
        fit to. It is assumed that the regression class has been initialized
        with whatever parameters necessary (alpha, lmb etc).

        :param num_bootstraps:  Number of random draws to make.
                                Defaults to 1000.
        :param test_size:   Relative size of test_data compared to full data.
                            Defaults to 0.1.

        :return: y_predictions - predicticed values for each resampled data set.
        """

        # Split the data the model was initially fit on into training and test
        # sets. Recommended test_size is around 0.1, so we'll stick to that.

        x_train, x_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=test_size)
        
        y_predictions = np.empty((n_boostraps, y_test.shape[0]))
        for i in range(num_bootstraps):
            x_re, y_re = resample(x_train, y_train)

            # Fit the model to the resampled data. Updates coefficients
            # so we can use make_prediction.
            self.fit_coefficients(x_re, y_re)
            y_predictions[i] = self.make_prediction(x_test)

        # Analysis

        print("Bootstrap statistics:")
        print("{:^8s} | {:^8s} | {:^8s} | {:^8s}".format("original",
                                                         "bias",
                                                         "mean",
                                                         "std.err"))

        print("{:8f} | {:8f} | {:8f} | {:8f}".format(np.mean(y_test),
                                                     np.std(t_test),
                                                     np.mean(y_predictions),
                                                     np.std(y_predictions)))

        return y_predictions

