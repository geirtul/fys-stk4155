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

# Setting up the matrix X in the matrix equation y = X*Beta + Eps

X = np.random.uniform(0, 1, (20,2))
X = np.sort(X, axis=0) # Sort the x-values
poly = PolynomialFeatures(5)
X_vals = poly.fit_transform(X)

beta = np.linalg.inv(X_vals.T.dot(X_vals)).dot(X_vals.T).dot(z)

z_approx = X_vals.dot(beta)
X_plot, Y_plot = np.meshgrid(X[:,0], X[:,1])


# Plotting
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X_plot, Y_plot, z_approx, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
