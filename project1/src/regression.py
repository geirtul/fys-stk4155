import numpy as np
from franke_function import FrankeFunction 
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib import cm
from random import random, seed
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

#making x and y
X = np.random.uniform(0,1, (20, 2))
X = np.sort(X, axis=0) #sort x values

x = X[:,0]
y = X[:,1]

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



