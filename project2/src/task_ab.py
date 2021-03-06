import numpy as np
import sys
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn

from ising_data import ising_energies, recast_to_regression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from ols import OrdinaryLeastSquares
from ridge import RidgeRegression
from lasso import LassoRegression
seaborn.set()


# Comment this to turn on warnings
warnings.filterwarnings('ignore')

# Define Ising model params
# system size

np.random.seed(12)
L = 40
num_states = 10000

# create N random Ising states, calculate energies and recast for regression.
states = np.random.choice([-1, 1], size=(num_states, L))
energies = ising_energies(states, L)
states = recast_to_regression(states)
data = [states, energies]

# Store errors for regression with sklearn and homemade regression.
train_errors_leastsq = []
test_errors_leastsq = []

train_errors_ridge = []
test_errors_ridge = []

train_errors_lasso = []
test_errors_lasso = []


# Split into training and test data sets
X_train, X_test, Y_train, Y_test = train_test_split(data[0],
                                                    data[1],
                                                    test_size=0.2)

print(X_train.shape)
X_train = X_train[:400, :]
Y_train = Y_train[:400]
# Run through the calculations and plot same as Mehta et. al for comparison.

# Set regularization strength values
lmbdas = (1e-4, 1e-2, 1e0, 1e4)

#Initialize coeffficients for ridge regression and Lasso
coefs_leastsq = []
coefs_ridge = []
coefs_lasso=[]

# Run regression for all models for multiple lambda/alpha values
fig, axarr = plt.subplots(nrows=len(lmbdas), ncols=3)
fig.subplots_adjust(hspace=0.5)

for i, lmbda in enumerate(lmbdas):
    print("Running regression for lambda = ", lmbda)
    # Initialize regression objects
    leastsq = OrdinaryLeastSquares()
    ridge = RidgeRegression(lmbda)
    lasso = LassoRegression(lmbda)
    # Ordinary Least Squares
    print("OLS...")
    leastsq.fit_coefficients(X_train, Y_train)  # fit model
    coefs_leastsq.append(leastsq.coeff)  # store weights
    # use the coefficient of determination R^2 as the performance of prediction.
    train_errors_leastsq.append(leastsq.r2_score(X_train, Y_train))
    test_errors_leastsq.append(leastsq.r2_score(X_test, Y_test))

    # apply Ridge regression
    print("Ridge...")
    ridge.fit_coefficients(X_train, Y_train)  # fit model
    coefs_ridge.append(ridge.coeff)  # store weights
    # use the coefficient of determination R^2 as the performance of prediction.
    train_errors_ridge.append(ridge.r2_score(X_train, Y_train))
    test_errors_ridge.append(ridge.r2_score(X_test, Y_test))

    # apply lasso regression
    print("Lasso...")
    lasso.fit_coefficients(X_train, Y_train)  # fit model
    coefs_lasso.append(lasso.coeff)  # store weights
    # use the coefficient of determination R^2 as the performance of prediction.
    train_errors_lasso.append(lasso.r2_score(X_train, Y_train))
    test_errors_lasso.append(lasso.r2_score(X_test, Y_test))

    # plot Ising interaction J
    J_leastsq = np.array(leastsq.coeff).reshape((L, L))
    J_ridge = np.array(ridge.coeff).reshape((L, L))
    J_lasso = np.array(lasso.coeff).reshape((L, L))

    cmap_args = dict(vmin=-1., vmax=1., cmap='seismic')



    axarr[i, 0].imshow(J_leastsq, **cmap_args)
    axarr[i, 0].set_title('$\\mathrm{OLS}$', fontsize=12)
    axarr[i, 0].tick_params(labelsize=16)

    axarr[i, 1].imshow(J_ridge, **cmap_args)
    axarr[i, 1].set_title('$\\mathrm{Ridge},\ \\lambda=%.4f$' % lmbda,
                       fontsize=12)
    axarr[i, 1].tick_params(labelsize=16)

    im = axarr[i, 2].imshow(J_lasso, **cmap_args)
    axarr[i, 2].set_title('$\\mathrm{LASSO},\ \\lambda=%.4f$' % lmbda,
                       fontsize=12)
    axarr[i, 2].tick_params(labelsize=16)

    divider = make_axes_locatable(axarr[i, 2])
    cax = divider.append_axes("right", size="5%", pad=0.5)
    cbar = fig.colorbar(im, cax=cax)

    cbar.ax.set_yticklabels(np.arange(-1.0, 1.0 + 0.25, 0.25), fontsize=12)
    cbar.set_label('$J_{i,j}$', labelpad=-40, y=1.25, x=1.25, fontsize=16, rotation=0)

    #fig.subplots_adjust(right=2.0)

plt.show()

# Plot our performance on both the training and test data
plt.semilogx(lmbdas, train_errors_leastsq, 'b',label='Train (OLS)')
plt.semilogx(lmbdas, test_errors_leastsq,'--b',label='Test (OLS)')
plt.semilogx(lmbdas, train_errors_ridge,'r',label='Train (Ridge)',linewidth=1)
plt.semilogx(lmbdas, test_errors_ridge,'--r',label='Test (Ridge)',linewidth=1)
plt.semilogx(lmbdas, train_errors_lasso, 'g',label='Train (LASSO)')
plt.semilogx(lmbdas, test_errors_lasso, '--g',label='Test (LASSO)')

fig = plt.gcf()
fig.set_size_inches(10.0, 6.0)

#plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
#           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left',fontsize=16)
plt.ylim([-0.01, 1.01])
plt.xlim([min(lmbdas), max(lmbdas)])
plt.xlabel(r'$\lambda$',fontsize=16)
plt.ylabel('Performance',fontsize=16)
plt.tick_params(labelsize=16)
plt.show()
