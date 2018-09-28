import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import cm
from random import random, seed
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

# Data from FrankeFunction
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x, y)

# Function definitions
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


# Setting up the matrix X in the matrix equation y = X*Beta + Eps
# and perform regression

def ordinary_least_squares(y, m):
    """
    Perform linear regression using th ordinary least squares method
    on a dataset y, with a polynomial of degree m.
    The PolynomialFeatures module from scikit learn sets up the vandermonde
    matrix such that in the matrix equation X*beta = y, beta is the coefficient vector,
    and X contains the polynomial expressions.
    returns x and y values for plotting along with the predicted y values
    from the model.
    """

    # Setup
    N = y.shape[0]
    X_vals = np.random.uniform(0, 1, (N,2))
    X_vals = np.sort(X_vals, axis=0) # Sort the x-values
    poly = PolynomialFeatures(m)

    # Regression
    X = poly.fit_transform(X_vals)
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    y_predicted = X.dot(beta)

    # Output
    X_plot, Y_plot = np.meshgrid(X_vals[:,0], X_vals[:,1])
    return [X_plot, Y_plot, y_predicted]


output = ordinary_least_squares(z,5)

# =================================================================
# Plotting
# Plots the least squares fit and the FrankeFunction next to
# each other for comparison.
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
