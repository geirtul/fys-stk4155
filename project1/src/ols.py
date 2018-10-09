import numpy as np
from franke_function import FrankeFunction
from sklearn.preprocessing import PolynomialFeatures
from analysis import Analysis


class OrdinaryLeastSquares(Analysis):

    def __init__(self):
        """
        Perform linear regression using the Ordinary Least Squares method
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
        """ 
        Fits polynomial coefficients beta for the model to the data.

        m - int, degree of polynomial you want to fit
        numOfPredictors - int, number of predictors
        z - vector, target data
        """
        self.m = m
        self.predictors = numOfPredictors
        self.z = z

        # Setup
        num_datapoints = z.shape[0]
        X_vals = np.random.uniform(0, 1, (num_datapoints, self.predictors))
        self.X_vals = np.sort(X_vals, axis=0)  # Sort the x-values
        poly = PolynomialFeatures(m)

        # Regression
        self.X = poly.fit_transform(X_vals)  # Input values to design matrix
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.z

    def makePrediction(self):
        """
        Makes a model prediction
        Returns prediction together with x and y values for plotting.
        """
        self.z_predicted = self.X @ self.beta

        # Output
        X_plot, Y_plot = np.meshgrid(self.X_vals[:, 0], self.X_vals[:, 1])
        return [X_plot, Y_plot, self.z_predicted]


if __name__ == "__main__":
    # Data from FrankeFunction
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    z = FrankeFunction(x, y)

    data = [x, y, z]

    ols = OrdinaryLeastSquares()
    ols.fitCoefficients(5, 2, z)

    output = ols.makePrediction()

    r2 = ols.r2_score
    ols.bootstrap()

    # ols.plotting_3d(data, output)
