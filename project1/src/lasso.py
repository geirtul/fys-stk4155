import numpy as np
from numpy import mean
from franke_function import FrankeFunction 
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model 
from matplotlib import cm
from random import random, seed
from analysis import Analysis 
#import matplotlib.pyplot as plt
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from mpl_toolkits.mplot3d import Axes3D
#from analysis import plotting_3d
#include <.h>



class LassoRegression(Analysis):


    def __init__(self):
        
        """Perform regression using the ridge method
        on a dataset y, with a polynomial of degree m.
        The PolynomialFeatures module from scikit learn sets up the 
        vandermonde matrix such that in the matrix equation X*beta = y, 
        beta is the coefficient vector,
        and X contains the polynomial expressions.
        returns x and y values for plotting along with the predicted y values
        from the model.

        Sets up the matrix X in the matrix equation y = X*Beta
        and performs regression
        """

        self.beta = None 

        #TODO metodene må skrives om. 

    def fitCoefficients(self, m, numOfPredictors, z, lmb=0):
        
        """ fits beta to model

        n - int, degree of polynomial you want to fit
        numOfPredictors - int, number of predictors  
        z - vector, target data
        lmb - float, shrinkage lmb=0 makes the model equal to ols  
        
        """

        self.m = m
        self.predictors = numOfPredictors 
        self.z = z

       # Setup
        num_datapoints = z.shape[0]
        X_vals = np.random.uniform(0, 1, (num_datapoints, self.predictors))
        self.X_vals = np.sort(X_vals, axis=0) # Sort the x-values
        poly = PolynomialFeatures(m) 

        # Regression
        self.X = poly.fit_transform(X_vals) # Input values to design matrix
        
        self.lassoreg = linear_model.Lasso(alpha=lmb) 
        self.lassoreg.fit(self.X, self.z) 

        #self.beta = (np.linalg.inv(self.X.T @ self.X + lmb*I) 
        #           @ self.X.T @ self.z)


    def makePrediction(self):

        """Makes a model prediction
        Returns prediction together with x and y values for plotting. 
        """
        
        self.z_predicted = self.lassoreg.predict(self.X)
        
        # Output
        X_plot, Y_plot = np.meshgrid(self.X_vals[:,0], self.X_vals[:,1])
        return [X_plot, Y_plot, self.z_predicted]


if __name__ == "__main__":
    # Data from FrankeFunction
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x,y)

    z = FrankeFunction(x, y)
    noiseRange = 1
    noise = noiseRange*np.random.uniform(-0.5, 0.5, size = z.shape)
    #z  = z - z.mean(1)[:, np.newaxis]
    z = z + noise

    data = [x, y, z] 
    
    lmb_values = [0, 1e-4, 1e-3, 1e-2, 10, 1e2, 1e4]

    lasso = LassoRegression()
    lasso.fitCoefficients(5, 2, z, 0.1)

    output = lasso.makePrediction() 

    r2 = lasso.r2_score()
    print(r2)
    """
    for lmb in lmb_values:
        ridge = RidgeRegression() 
        ridge.fitCoefficients(5, 2, z, lmb)

        output = ridge.makePrediction()

        r2 = ridge.r2_score() 
        print("R2 = {:f} for lmd = {:f}".format(r2, lmb))
    """

    #ridge.plotting_3d(data, output)

