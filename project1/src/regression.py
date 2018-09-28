import numpy as np
from franke_function import FrankeFunction 
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import cm
from random import random, seed
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


#making x and y
X = np.random.uniform(0,1, (20, 2))
X = np.sort(X, axis=0) #sort x values

from analysis import plotting_3d


# Data from FrankeFunction
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

x_plot, y_plot = np.meshgrid(x, y)

#setting up a fifth order polynomial and setting in for x an y. 
#Also addes a bias line of ones. 
poly = PolynomialFeatures(5)
fifthPoly = poly.fit_transform(X) 
print (poly.get_feature_names())

z = FrankeFunction(x, y)

beta = np.linalg.inv( fifthPoly.T @ fifthPoly ) @ fifthPoly.T @ z
z_approx = fifthPoly @ beta 


#plot franke_function

fig = plt.figure()
ax = fig.add_subplot(1,2,1, projection='3d')

surf1 = ax.plot_surface(x_plot, y_plot, z_approx, cmap=cm.coolwarm,
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




# Function definitions
def ordinary_least_squares(y, m, predictors):
    """
    Perform linear regression using the Ordinary Least Squares method
    on a dataset y, with a polynomial of degree m.
    The PolynomialFeatures module from scikit learn sets up the vandermonde
    matrix such that in the matrix equation X*beta = y, beta is the coefficient vector,
    and X contains the polynomial expressions.
    returns x and y values for plotting along with the predicted y values
    from the model.

    Sets up the matrix X in the matrix equation y = X*Beta
    and performs regression
    """

    # Setup
    num_datapoints = y.shape[0]
    X_vals = np.random.uniform(0, 1, (num_datapoints, predictors))
    X_vals = np.sort(X_vals, axis=0) # Sort the x-values
    poly = PolynomialFeatures(m)

    # Regression
    X = poly.fit_transform(X_vals) # Input values to design matrix
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    y_predicted = X.dot(beta)

    # Output
    X_plot, Y_plot = np.meshgrid(X_vals[:,0], X_vals[:,1])
    return [X_plot, Y_plot, y_predicted]

data = [x,y,z]
output = ordinary_least_squares(z,5,2)
plotting_3d(data, output, True)

