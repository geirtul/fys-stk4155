import numpy as np
import sys
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from lasso import LassoRegression
from ridge import RidgeRegression
from ols import OrdinaryLeastSquares
from franke_function import FrankeFunction

# Part a)
if sys.argv[1] == "a":

    # Set up coordinates and data for regression.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    # Reshape input to 1D arrays because that's just easier to deal with
    z = FrankeFunction(x, y).ravel()
    predictors_input = np.c_[x.ravel(), y.ravel()]

    degrees = [1, 2, 3, 4, 5]
    collected_data = []
    for degree in degrees:
        ols = OrdinaryLeastSquares()
        ols.fit_coefficients(predictors_input, z, degree)
        z_predict = ols.make_prediction(predictors_input, z)
        collected_data.append([degree, ols.mean_squared_error(), ols.r2_score()])

    # Output data to file as csv to be handled in plot/analysis script.
    with open("regression_data/{}.csv".format(sys.argv[1]), 'w') as outfile:
        outfile.write("degree,mse,r2\n")
        for val in collected_data:
            outfile.write("{},{},{}\n".format(val[0], val[1], val[2]))
        outfile.close()

    """
    print("{:12s} | {:12s} | {:12s}".format("Degree", "MSE", "R2 Score"))
    for val in collected_data:
        print("{:12} | {:12f} | {:12f}".format(val[0], val[1], val[2]))
    """

# Part b)
if sys.argv[1] == "b":

    # Set up coordinates and data for regression.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    # Reshape input to 1D arrays because that's just easier to deal with
    z = FrankeFunction(x, y).ravel()
    predictors_input = np.c_[x.ravel(), y.ravel()]

    degrees = [1, 2, 3, 4, 5]
    lmb_values = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    collected_data = []
    for degree in degrees:
        for lmb in lmb_values:
            ridge = RidgeRegression()
            ridge.fit_coefficients(predictors_input, z, degree, lmb)
            z_predict = ridge.make_prediction(predictors_input, z)
            collected_data.append([degree,
                                   lmb,
                                   ridge.mean_squared_error(),
                                   ridge.r2_score()])

    # Output data to file as csv to be handled in plot/analysis script.
    with open("regression_data/{}.csv".format(sys.argv[1]), 'w') as outfile:
        outfile.write("degree,lambda,mse,r2\n")
        for val in collected_data:
            outfile.write("{},{},{},{}\n".format(val[0], val[1], val[2], val[3]))
        outfile.close()

    """
    print("{:12s} | {:12s} | {:12s} | {:12s}".format("Degree",
                                                     "Lambda",
                                                     "MSE",
                                                     "R2 Score"))
    for val in collected_data:
        print("{:12} | {:12} | {:12f} | {:12f}".format(val[0],
                                                       val[1],
                                                       val[2],
                                                       val[3]))                                              
    """

# Part c)
if sys.argv[1] == "c":

    # Set up coordinates and data for regression.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    # Reshape input to 1D arrays because that's just easier to deal with
    z = FrankeFunction(x, y).ravel()
    predictors_input = np.c_[x.ravel(), y.ravel()]

    degrees = [1, 2, 3, 4, 5]
    alpha_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    collected_data = []
    for degree in degrees:
        for alpha in alpha_values:
            lasso = LassoRegression()
            lasso.fit_coefficients(predictors_input, z, degree, alpha=alpha)
            lasso.make_prediction(predictors_input, z)
            collected_data.append([degree,
                                   alpha,
                                   lasso.mean_squared_error(),
                                   lasso.r2_score()])

    # Output data to file as csv to be handled in plot/analysis script.
    with open("regression_data/{}.csv".format(sys.argv[1]), 'w') as outfile:
        outfile.write("degree,lambda,mse,r2\n")
        for val in collected_data:
            outfile.write("{},{},{},{}\n".format(val[0], val[1], val[2], val[3]))
        outfile.close()

    """
    print("{:12s} | {:12s} | {:12s} | {:12s}".format("Degree",
                                                     "Alpha",
                                                     "MSE",
                                                     "R2 Score"))
    for val in collected_data:
        print("{:12} | {:12} | {:12f} | {:12f}".format(val[0],
                                                       val[1],
                                                       val[2],
                                                       val[3]))
    """

# Part e)
#
if sys.argv[1] == "e_ols":

    # Set up coordinates and data for regression.
    x = np.arange(terrain_resized.shape[1])
    y = np.arange(terrain_resized.shape[0])
    x, y = np.meshgrid(x, y)
    predictors_input = np.c_[x.ravel(), y.ravel()]
    z = terrain_resized.ravel()

    degrees = [5, 6, 7, 8, 9, 10]
    collected_data = []
    for degree in degrees:
        ols = OrdinaryLeastSquares()
        ols.fit_coefficients(predictors_input, z, degree)
        z_predict = ols.make_prediction(predictors_input, z)
        collected_data.append([degree, ols.mean_squared_error(), ols.r2_score()])

    print("{:12s} | {:12s} | {:12s}".format("Degree", "MSE", "R2 Score"))
    for val in collected_data:
        print("{:12} | {:12f} | {:12f}".format(val[0], val[1], val[2]))


# ============================================================================
# THE FOLLOWING CODE USES THE REAL TERRAIN DATA
# ============================================================================

# Load the terrain data provided as example. This is the Møsvatn data.
terrain = imread('geodata/SRTM_data_Norway_2.tif')


def resize_terrain(data, x1, x2, y1, y2):
    """
    Resize the provided terrain data to a smaller sample for testing
    and or computational efficiency if needed.

    :return: Subset of the terrain data.
    """

    data_subset = data[x1:x2, y1:y2]
    return data_subset


terrain_resized = resize_terrain(terrain, 0, 1000, 0, 1000)

if sys.argv[1] == "e_ridge":

    # Set up coordinates and data for regression.
    x = np.arange(terrain_resized.shape[1])
    y = np.arange(terrain_resized.shape[0])
    x, y = np.meshgrid(x, y)
    predictors_input = np.c_[x.ravel(), y.ravel()]
    z = terrain_resized.ravel()

    # Define the degrees and lambda-values to loop over
    degrees = [5, 6, 7, 8, 9, 10]
    lmb_values = [0, 1e-1, 1, 2, 5, 10]
    collected_data = []

    # Loop over degrees and lambda values.
    for degree in degrees:
        for lmb in lmb_values:
            ridge = RidgeRegression()
            ridge.fit_coefficients(predictors_input, z, degree, lmb)
            z_predict = ridge.make_prediction(predictors_input, z)
            collected_data.append([degree,
                                   lmb,
                                   ridge.mean_squared_error(),
                                   ridge.r2_score()])

    # Print some data
    print("{:12s} | {:12s} | {:12s} | {:12s}".format("Degree",
                                                     "Lambda",
                                                     "MSE",
                                                     "R2 Score"))
    for val in collected_data:
        print("{:12} | {:12} | {:12f} | {:12f}".format(val[0],
                                                       val[1],
                                                       val[2],
                                                       val[3]))

   
"""
plt.figure()
plt.subplot(1, 2, 1)
plt.title('Terrain predicted by OLS model.')
plt.imshow(z_predict.reshape((len(y), len(x))), cmap=cm.coolwarm)
plt.xlabel('X')
plt.ylabel('Y')
plt.subplot(1, 2, 2)
plt.title('Terrain data.')
plt.imshow(terrain_resized, cmap=cm.coolwarm)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""

