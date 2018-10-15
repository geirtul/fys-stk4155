import numpy as np
from time import time
from numpy.random import randint
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import cm
from random import random, seed
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

class Analysis:

    def mean_squared_error(self):

        """Evaluate the mean squared error of the output generated
        by Ordinary Least Squares regressions.

        y               - the data on which OLS was performed on
        y_predicted     - the data the model spits out after regression
        """

        num_datapoints = self.z.shape[0]
        mse = np.sum(np.square(self.z - self.z_predicted))/num_datapoints
        return mse

    def r2_score(self):

        """ Evaluate the R2 score function.

        y               - the data on which OLS was performed on
        y_predicted     - the data the model spits out after regression
        y_mean          -  mean value of y
        upper/lower_sum - numerator/denominator in R2 definition.
        """

        N = self.z.shape[0]
        print(self.z_predicted)
        z_mean = np.sum(self.z)/N
        upper_sum = np.sum(np.square(self.z - self.z_predicted))
        lower_sum = np.sum(np.square(self.z - z_mean))
        r2score = 1 - upper_sum/lower_sum
        return r2score

    def bootstrap(self):
        """
        Perform the chosen regression using bootstrapping.
        """
        N = len(self.z)
        n_bootstraps = 100

        # Split into training and test sets
        test_size = 0.2
        stop = int(np.floor(test_size*N))
        x_test = self.X_vals[0:stop, :]
        data_test = self.z[0:stop, :]
        #x_train = self.X_vals[test_size*N:,:]
        data_train = self.z[stop:,:]

        def resample(data):
            points = len(data)
            #x_indices = randint(0, points, x.shape)
            data_indices = randint(0,points,data.shape)
            return data[data_indices]

        
        y_fits = np.empty((len(data_train), n_bootstraps))
        for i in range(n_bootstraps):
            
            data_re = resample(data_train)
            self.fitCoefficients(5, 2, data_re)
            y_fits[i] = self.makePrediction()

        
        
        print("Mean y_fit = {}".format(np.mean(y_fits)))
        """
        #Some pretty printing we probably don't need.
        print("Bootstrap statistics:")
        print("{:^8s} | {:^8s} | {:^8s} | {:^8s}".format("original", "bias", "mean", "std.err"))
        print("{:8f} | {:8f} | {:8f} | {:8f}".format( np.mean(self.z),
                                                np.std(self.z),
                                                np.mean(t),
                                                np.std(t)))
        """
        return y_fits


    def plotting_3d(self, data, output, save=False):

        """ Plots the modeled data side-by-side with the original dataset
        for comparison.

        data    - list[x, y, z]
                  contains x, y meshgrid and solution from FrankeFunction.

        output  - list[x, y, z_predicted]
                  contains x, y meshgrid and predicted solution.
        """

        self.data = data
        self.output = output

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        surf1 = ax.plot_surface(self.output[0], self.output[1],
                                self.output[2], cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf1, shrink=0.5, aspect=5)

        # Second subplot
        ax = fig.add_subplot(1,2,2, projection='3d')
        surf2 = ax.plot_surface(self.data[0], self.data[1],
                                self.data[2], cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf2, shrink=0.5, aspect=5)

        if save:
            plt.savefig("../report/figures/comparison.pdf", format='pdf')
        plt.show()
