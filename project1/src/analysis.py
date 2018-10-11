import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


class Analysis:

    def mean_squared_error(self):

        """
        Evaluate the mean squared error of the output generated
        by Ordinary Least Squares regressions.

        y               - the data on which OLS was performed on
        y_predicted     - the data the model spits out after regression
        """

        N = self.outcome.size
        mse = np.sum(np.square(self.outcome - self.predicted_outcome)) / N
        return mse

    def r2_score(self):
        """
        Evaluate the R2 score of the model fitted to the dataset.

        :return:    Returns the mean squared error of the model compared with the dataset
                    used for fitting.
        """

        outcome_mean = np.mean(self.outcome)
        upper_sum = np.sum(np.square(self.outcome - self.predicted_outcome))
        lower_sum = np.sum(np.square(self.outcome - outcome_mean))
        r2score = 1 - upper_sum / lower_sum
        return r2score

    def bootstrap(self):
        """
        Perform the chosen regression using bootstrapping.
        """

        def statistics(testdata, predicted_data):
            """
            Calculates useful statistics

            :param testdata: Data to compare the fitted data with
            :param predicted_data: Data predicted by the model
            :return: mean square error, bias, variance
            """
            # MSE = np.mean(np.mean((testdata - predicted_data)**2, axis 1, keepdims=True))
            # bias = np.mean(testdata)
            raise NotImplemented

        n_bootstraps = 100

        # Split into training and test sets
        x_train, x_test, data_train, data_test = train_test_split(self.X_vals, self.z, test_size=0.2)

        y_fits = np.empty((n_bootstraps, len(data_train)))
        for i in range(n_bootstraps):
            x_resampled, data_resampled = resample(x_train, data_train, replace=True)
            self.fitCoefficients(5, 2, data_resampled)
            newfit = self.makePrediction(x_test)
            y_fits[i] = np.sum(newfit, axis=1)
            # TODO: make this business right here function as it should.

        print("Bootstrap statistics:")
        print("{:^8s} | {:^8s} | {:^8s} | {:^8s}".format("original", "bias", "mean_fit", "std.err"))
        print("{:8f} | {:8f} | {:8f} | {:8f}".format( np.mean(self.z),
                                                np.std(self.z),
                                                np.mean(y_fits),
                                                np.std(y_fits)))

        print("\nBootstrap with scikit-learn:")

        return y_fits

    def plotting_3d(self, save=False):
        """
        Plots the modeled data side-by-side with the original dataset
        for comparison.
        """

        # Doing some reshaping business for plotting in 3D.
        x, y = self.predictors[:,0], self.predictors[:,1]
        z, z_predict = self.outcome, self.predicted_outcome
        N = int(np.sqrt(x.size))
        x = x.reshape((N,N))
        y = y.reshape((N,N))
        z = z.reshape((N,N))
        z_predict = z_predict.reshape((N,N))

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        surf1 = ax.plot_surface(x, y, z_predict,
                                cmap=cm.coolwarm,
                                linewidth=0,
                                antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf1, shrink=0.5, aspect=5)

        # Second subplot
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf2 = ax.plot_surface(x, y, z,
                                cmap=cm.coolwarm,
                                linewidth=0,
                                antialiased=False)
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf2, shrink=0.5, aspect=5)

        if save:
            plt.savefig("../report/figures/comparison.pdf", format='pdf')
        plt.show()


if __name__ == "__main__":
    print("Loaded Analysis.py")
