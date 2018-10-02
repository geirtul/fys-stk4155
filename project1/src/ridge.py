import numpy as np
from numpy import mean
from franke_function import FrankeFunction 
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import cm
from random import random, seed
from analysis import Analysis 
#import matplotlib.pyplot as plt
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from mpl_toolkits.mplot3d import Axes3D
#from analysis import plotting_3d
#include <.h>



class RidgeRegression(Analysis):


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

    def fitCoefficients(self, m, numOfPredictors, z):
        
        """ fits beta to model

        n - int, degree of polynomial you want to fit
        numOfPredictors - int, number of predictors  
        z - vector, target data
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

        #centering X?
        meanX = self.X.mean(1)
        centeredX = self.X - meanX[:, np.newaxis]
        self.centeredX = centeredX[:, 1:] #without intercept
        
        print (self.centeredX.shape)
        
        p = len(centeredX) 

        self.beta0 = self.z.mean(1) 
        #lmb_values = [1E-4, 1E-3, 1E-2, 10, 1E2, 1E4] #from lecture notes 
        #numValues = len(lmb_values)
        I = np.eye(p)

        #self.beta = np.zeros((p+1, numValues))

        
        #for i, lmb in enumerate(lmb_values):
        self.beta = (np.linalg.inv(self.centeredX.T @ self.centeredX + 1E-2*I) @ self.centeredX.T @ self.z)


    def makePrediction(self):

        """Makes a model prediction
        Returns prediction together with x and y values for plotting. 
        """
        
        self.z_predicted = self.centeredX @ self.beta + self.beta0


        # Output
        X_plot, Y_plot = np.meshgrid(self.X_vals[:,0], self.X_vals[:,1])
        return [X_plot, Y_plot, self.z_predicted]


if __name__ == "__main__":
    # Data from FrankeFunction
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x,y)

    z = FrankeFunction(x, y)

    data = [x, y, z] 

    ridge = RidgeRegression() 
    ridge.fitCoefficients(5, 2, z)

    output = ridge.makePrediction()

    #r2 = ridge.r2_score 


    ridge.plotting_3d(data, output)

