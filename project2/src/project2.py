import numpy as np
import warnings
import matplotlib.pyplot as plt

from ising_data import ising_energies, recast_to_regression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from ols import OrdinaryLeastSquares
from ridge import RidgeRegression
from lasso import LassoRegression

"""
This is the file that will run the numerical experiments.
Ising data setup from Mehta et. al. notebook on Ising linear regression.
"""

# Comment this to turn on warnings
warnings.filterwarnings('ignore')

# Define Ising model params
# system size

np.random.seed(12)
L = 40
num_states = 1000

# create N random Ising states, calculate energies and recast for regression.
states = np.random.choice([-1, 1], size=(num_states, L))
energies = ising_energies(states, L)
states = recast_to_regression(states)
data = [states, energies]

# Store errors for regression with sklearn and homemade regression.
train_errors_ols = []
test_errors_ols = []

# train_errors_ridge = []
# test_errors_ridge = []
#
# train_errors_lasso = []
# test_errors_lasso = []

homemade_train_errors_ols = []
homemade_test_errors_ols = []

# homemade_train_errors_ridge = []
# homemade_test_errors_ridge = []
#
# homemade_train_errors_lasso = []
# homemade_test_errors_lasso = []


# Split into training and test data sets
x_train, x_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.1)

# Set up scikit regression
ols = linear_model.LinearRegression()
ols.fit(x_train, y_train)
scikit_r2_train = ols.score(x_train, y_train)
scikit_r2_test = ols.score(x_test, y_test)

# ridge = linear_model.Ridge()
# lasso = linear_model.Lasso()

# Set up homemade regression
# homemade_ols = OrdinaryLeastSquares()
# homemade_ols.fit_coefficients(x_train, y_train, 5)
# homemade_r2_train = homemade_ols.r2_score(x_train, y_train)
# homemade_r2_test = homemade_ols.r2_score(x_test, y_test)

# Print some comparison values
print("Scikit results\nR2 train | R2 test")
print(scikit_r2_train," ", scikit_r2_test)
# print("Homemade results\nR2 train | R2 test")
# print(homemade_r2_train," ", homemade_r2_test)
