import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Setting up the matrix X in the matrix equation y = X*Beta + Eps
X = np.arange(6).reshape(3,2)
poly = PolynomialFeatures(2)
print(poly.fit_transform(X))
