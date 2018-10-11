import numpy as np
from franke_function import FrankeFunction 
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from analysis import Analysis
from sklearn.model_selection import train_test_split


class LassoRegression(Analysis):

    def __init__(self):
        """
        Perform regression using the Lasso method
        on a dataset y, with a polynomial of degree poly_degree.
        The PolynomialFeatures module from scikit learn sets up the 
        vandermonde matrix such that in the matrix equation X*beta = y, 
        beta is the coefficient vector, and X contains the polynomial expressions.

        The method uses scikit learn's Lasso regression methods.
        """

        self.predictors = None
        self.poly_degree = None
        self.outcome = None
        self.beta = None
        self.poly = None
        self.predicted_outcome = None
        self.alpha = None
        self.lasso_object = None

    def fit_coefficients(self, predictors, outcome, poly_degree, alpha=1.0):
        """
        Fits the polynomial coefficients beta to the matrix
        of polynomial features.

        :param predictors: x,y, ... values that generated the outcome
        :param outcome: the dataset we will fit
        :param poly_degree: Degree of the polynomial to fit
        :param alpha: float, Constant that multiplies the L1 term.
                        Defaults to 1.0. alpha = 0 is equivalent to
                        an ordinary least square regression.
        """

        self.predictors = predictors
        self.poly_degree = poly_degree
        self.poly = PolynomialFeatures(poly_degree)
        self.outcome = outcome
        self.alpha = alpha

        # Regression
        X = self.poly.fit_transform(self.predictors)  # Input values to design matrix
        
        self.lasso_object = linear_model.Lasso(alpha=self.alpha)
        self.lasso_object.fit(X, self.outcome)

    def make_prediction(self, x_in):
        """
        Makes a model prediction
        """
        X = self.poly.fit_transform(x_in)
        self.predicted_outcome = self.lasso_object.predict(X)

        return self.predicted_outcome


if __name__ == "__main__":
    # Data from FrankeFunction
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    z = FrankeFunction(x, y).ravel()
    # Make predictor values a matrix with number of columns = number of predictors.
    # TODO: Need better input handling. Number of predictors shouldn't matter.
    predictors_input = np.c_[x.ravel(), y.ravel()]

    x_train, x_test, data_train, data_test = train_test_split(predictors_input, z, test_size=0.2)

    lasso = linear_model.Lasso(alpha=1.0)
    lasso.fit(x_train, data_train)
    pred1 = lasso.predict(x_test)

    print("MSE = ", np.mean(np.square(data_test - pred1) / len(data_test)))
    print("R2: ", lasso.score(pred1, data_test))

    """
    noiseRange = 10
    noise = noiseRange*np.random.uniform(-0.5, 0.5, size = z.shape)
    z  = z - z.mean(1)[:, np.newaxis]
    z = z + noise


    alpha_values = [1e-2, 1, 10]

    lasso = LassoRegression()
    lasso.fit_coefficients(x_train, data_train, 5, 0.1)
    print("MSE = ", np.mean(np.square(data_test - lasso.make_prediction(x_test))/len(data_test)))
    r2 = lasso.lasso_object.score(lasso.lasso_object.predict(x_test), data_test)
    print("R2 = ".format(r2))

    for alpha in alpha_values:
        print("Running Lasso Regression with alpha = {}\n".format(alpha))
        lasso = LassoRegression() 
        lasso.fit_coefficients(predictors_input, z, 5, alpha)

        output = lasso.make_prediction(predictors_input)
        mse = lasso.mean_squared_error()
        r2 = lasso.r2_score()
        print("MSE = {:f} for alpha = {:f}".format(r2, alpha))
        print("R2 = {:f} for alpha = {:f}".format(r2, alpha))
        print("===================================\n")
    """



