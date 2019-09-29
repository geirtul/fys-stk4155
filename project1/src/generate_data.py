import numpy as np
import sys
from imageio import imread
from lasso import LassoRegression
from ridge import RidgeRegression
from ols import OrdinaryLeastSquares
from sklearn.preprocessing import PolynomialFeatures
from franke_function import FrankeFunction

# Part a)
# ============================================================================
if sys.argv[1] == "a":

    # Set up coordinates and data for regression.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    # Reshape input to 1D arrays because that's just easier to deal with
    z = FrankeFunction(x, y).ravel()
    predictors_input = np.c_[x.ravel(), y.ravel()]

    # Run the regression with varying degrees of stochastic noise added
    degrees = [1, 2, 3, 4, 5]
    noise_levels = [0, 0.1, 0.5, 1.0]
    collected_data = []
    bootstraps = []
    for degree in degrees:
        x = PolynomialFeatures(degree).fit_transform(predictors_input)
        for noise_level in noise_levels:
            noise = noise_level*np.random.normal(0, 1, z.shape)
            tmp_z = z + noise
            ols = OrdinaryLeastSquares()
            ols.fit_coefficients(x, tmp_z)
            bootstraps.append([degree, noise] + ols.bootstrap(num_bootstraps=10000))
            collected_data.append(
                    [degree,
                    noise_level,
                    ols.mean_squared_error(x, tmp_z), 
                    ols.r2_score(x, tmp_z)]
                    )


    # Output data to file as csv to be handled in plot/analysis script.
    with open("regression_data/{}.csv".format(sys.argv[1]), 'w') as outfile:
        outfile.write("degree,noise,mse,r2\n")
        for val in collected_data:
            outfile.write("{},{},{},{}\n".format(val[0], val[1], val[2], val[3]))
        outfile.close()
    with open("regression_data/{}_bs.csv".format(sys.argv[1]), 'w') as outfile:
        outfile.write("degree,noise,error,bias,variance,coeeff_variance\n")
        for val in bootstraps:
            outfile.write("{},{},{},{},{},{}\n".format(
                val[0], val[1], val[2], val[3], val[4], val[5]))
        outfile.close()

    """
    print("{:12s} | {:12s} | {:12s}".format("Degree", "MSE", "R2 Score"))
    for val in collected_data:
        print("{:12} | {:12f} | {:12f}".format(val[0], val[1], val[2]))
    """
# ============================================================================

# Part b)
# ============================================================================
if sys.argv[1] == "b":

    # Set up coordinates and data for regression.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    # Reshape input to 1D arrays because that's just easier to deal with
    z = FrankeFunction(x, y).ravel()
    predictors_input = np.c_[x.ravel(), y.ravel()]

    degrees = [1, 2, 3, 4, 5]
    noise_levels = [0, 0.1, 0.5, 1.0]
    lmb_values = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    collected_data = []
    for degree in degrees:
        x = PolynomialFeatures(degree).fit_transform(predictors_input)
        for noise_level in noise_levels:
            noise = noise_level*np.random.normal(0, 1, z.shape)
            tmp_z = z + noise
            for lmb in lmb_values:
                ridge = RidgeRegression(lmb)
                ridge.fit_coefficients(x, tmp_z)
                collected_data.append([degree,
                                       lmb,
                                       noise_level,
                                       ridge.mean_squared_error(x, tmp_z),
                                       ridge.r2_score(x, tmp_z)])

    # Output data to file as csv to be handled in plot/analysis script.
    with open("regression_data/{}.csv".format(sys.argv[1]), 'w') as outfile:
        outfile.write("degree,lambda,noise,mse,r2\n")
        for val in collected_data:
            outfile.write("{},{},{},{},{}\n".format(val[0], val[1], val[2], val[3], val[4]))
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
# ============================================================================

# Part c)
# ============================================================================
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
            lasso.predict(predictors_input, z)
            collected_data.append([degree,
                                   alpha,
                                   lasso.mean_squared_error(),
                                   lasso.r2_score()])

    # Output data to file as csv to be handled in plot/analysis script.
    with open("regression_data/{}.csv".format(sys.argv[1]), 'w') as outfile:
        outfile.write("degree,alpha,mse,r2\n")
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
# ============================================================================


# ============================================================================
# THE FOLLOWING CODE USES THE REAL TERRAIN DATA
# ============================================================================



def resize_terrain(data, x1, x2, y1, y2):
    """
    Resize the provided terrain data to a smaller sample for testing
    and or computational efficiency if needed.

    :return: Subset of the terrain data.
    """

    data_subset = data[x1:x2, y1:y2]
    return data_subset



# Part e)
# ============================================================================
if sys.argv[1] == "e_ols":
    # Load the terrain data provided as example. This is the Møsvatn data.
    terrain = imread('geodata/SRTM_data_Norway_2.tif')
    terrain_resized = resize_terrain(terrain, 0, 1000, 0, 1000)

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
        z_predict = ols.predict(predictors_input, z)
        collected_data.append([degree, ols.mean_squared_error(), ols.r2_score()])

    # Output data to file as csv to be handled in plot/analysis script.
    with open("regression_data/{}.csv".format(sys.argv[1]), 'w') as outfile:
        outfile.write("degree,mse,r2\n")
        for val in collected_data:
            outfile.write("{},{},{}\n".format(val[0], val[1], val[2]))
        outfile.close()

    """
    # Print data
    print("{:12s} | {:12s} | {:12s}".format("Degree", "MSE", "R2 Score"))
    for val in collected_data:
        print("{:12} | {:12f} | {:12f}".format(val[0], val[1], val[2]))
    """

if sys.argv[1] == "e_ridge":
    # Load the terrain data provided as example. This is the Møsvatn data.
    terrain = imread('geodata/SRTM_data_Norway_2.tif')
    terrain_resized = resize_terrain(terrain, 0, 1000, 0, 1000)

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
            z_predict = ridge.predict(predictors_input, z)
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
    # Print data
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

if sys.argv[1] == "e_lasso":
    # Load the terrain data provided as example. This is the Møsvatn data.
    terrain = imread('geodata/SRTM_data_Norway_2.tif')
    terrain_resized = resize_terrain(terrain, 0, 1000, 0, 1000)

    # Set up coordinates and data for regression.
    x = np.arange(terrain_resized.shape[1])
    y = np.arange(terrain_resized.shape[0])
    x, y = np.meshgrid(x, y)
    predictors_input = np.c_[x.ravel(), y.ravel()]
    z = terrain_resized.ravel()

    # Define the degrees and lambda-values to loop over
    degrees = [5, 6, 7, 8, 9, 10]
    alpha_values = [1e-5, 1e-4, 1e-3, 1e-2]
    collected_data = []

    # Loop over degrees and alpha values.
    for degree in degrees:
        for alpha in alpha_values:
            lasso = LassoRegression()
            lasso.fit_coefficients(predictors_input, z, degree, alpha=alpha)
            lasso.predict(predictors_input, z)
            collected_data.append([degree,
                                   alpha,
                                   lasso.mean_squared_error(),
                                   lasso.r2_score()])

    # Output data to file as csv to be handled in plot/analysis script.
    with open("regression_data/{}.csv".format(sys.argv[1]), 'w') as outfile:
        outfile.write("degree,alpha,mse,r2\n")
        for val in collected_data:
            outfile.write("{},{},{},{}\n".format(val[0], val[1], val[2], val[3]))
        outfile.close()

    """
    # Print data
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
