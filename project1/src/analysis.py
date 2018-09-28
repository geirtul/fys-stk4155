import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import cm
from random import random, seed
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def mean_squared_error(y, y_predicted):
    """
    Evaluate the mean squared error of the output generated
    by Ordinary Least Squares regressions.
    y : the data on which OLS was performed on
    y_predicted : the data the model spits out after regression
    """

    N = y.shape[0]
    mse = np.sum(np.square(y - y_predicted))/N
    return mse

def r2_score(y, y_predicted):
    """
    Evaluate the R2 score function.
    y : the data on which OLS was performed on
    y_predicted : the data the model spits out after regression
    y_mean : mean value of y
    upper/lower_sum : numerator/denominator in R2 definition.
    """
    N = y.shape[0]
    y_mean = np.sum(y)/N
    upper_sum = np.sum(np.square(y - y_predicted))
    lower_sum = np.sum(np.square(y - y_mean))
    r2score = 1 - upper_sum/lower_sum


def plotting_3d(data, output):
    """
    Plots the modeled data side-by-side with the original dataset
    for comparison.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    surf1 = ax.plot_surface(output[0], output[1], output[2], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf1, shrink=0.5, aspect=5)

    # Second subplot
    ax = fig.add_subplot(1,2,2, projection='3d')
    surf2 = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf2, shrink=0.5, aspect=5)

    plt.show()
