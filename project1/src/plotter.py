import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Part a)

if len(sys.argv) > 0:
    for arg in sys.argv[1:]:
        with open("regression_data/{}.csv".format(arg)) as infile:
            headers = infile.readline()
            infile.close()
        data = np.loadtxt("regression_data/{}.csv".format(arg),
                          delimiter=',',
                          skiprows=1)
        print(headers)
        print(data)


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
