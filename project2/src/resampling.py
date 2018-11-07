import numpy as np
from numpy.random import randint, randn
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


class Resampling:
    """
    Class containing resampling methods to be used with regression classes.
    """

    def bootstrap(self, num_bootstraps):
        """
        Perform bootstrapping on the dataset the regression model has been
        fit to.
        """
        y_predictions = np.empty((y_test.shape[0], n_boostraps))

        # Split the data the model was initially fit on into training and test
        # sets. Recommended test_size is around 0.1, so we'll stick to that.

        x_train, x_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=0.1)

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
        print("{:8f} | {:8f} | {:8f} | {:8f}".format(np.mean(data),
                                                     np.std(data),
                                                     np.mean(t),
                                                     np.std(t)))

        return t

