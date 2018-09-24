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
print (X.shape)

x = X[:,0]
y = X[:,1]

x, y = np.meshgrid(x, y)

#setting up a fifth order polynomial and setting in for x an y. 
#Also addes a bias line of ones. 
poly = PolynomialFeatures(5)
fifthPoly = poly.fit_transform(X) 
# print (poly.get_feature_names())
print (fifthPoly.shape)

z = FrankeFunction(x, y)

beta = np.linalg.inv( X.T @ X ) @ X.T @ z 

print (beta.shape) 


#plot franke_function
fig = plt.figure()
ax = fig.gca(projection='3d')


surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()






