import numpy as np
from franke_function import FrankeFunction 
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import cm
from random import random, seed
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from analysis import plotting_3d


# Data from FrankeFunction
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)



# Function definitions
def ordinary_least_squares(z, m, predictors):

    """Perform linear regression using the Ordinary Least Squares method
    on a dataset y, with a polynomial of degree m.
    The PolynomialFeatures module from scikit learn sets up the vandermonde
    matrix such that in the matrix equation X*beta = y, 
    beta is the coefficient vector,
    and X contains the polynomial expressions.
    returns x and y values for plotting along with the predicted y values
    from the model.

    Sets up the matrix X in the matrix equation y = X*Beta
    and performs regression
    """

    # Setup
    num_datapoints = z.shape[0]
    X_vals = np.random.uniform(0, 1, (num_datapoints, predictors))
    X_vals = np.sort(X_vals, axis=0) # Sort the x-values
    poly = PolynomialFeatures(m)

    # Regression
    X = poly.fit_transform(X_vals) # Input values to design matrix
    #beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
    #z_predicted = X.dot(beta)
    
    beta = np.linalg.inv( X.T @ X ) @ X.T @ z
    z_predicted = X @ beta 


    # Output
    X_plot, Y_plot = np.meshgrid(X_vals[:,0], X_vals[:,1])
    return [X_plot, Y_plot, z_predicted]

data = [x,y,z]
output = ordinary_least_squares(z,5,2)
plotting_3d(data, output)

