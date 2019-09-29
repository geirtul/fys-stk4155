import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Helper functions
def determine_best_mse(data, headers, type):
    """
    Figure out which data is the "better fit for" ridge and lasso regressions.
    Basically finds which lambda or alpha provides the lowest mean squared
    error.
    """
    if type == 'ridge':
        lmb_index = headers.index('lambda')
        mse_index = headers.index('mse')
        lmb_values = np.unique(data[:, lmb_index])
        mean_mse = []
        for lmb in lmb_values:
            current_indices = np.where(data[:, lmb_index] == lmb)
            mean_mse.append([lmb, np.mean(data[current_indices, mse_index])])
        best_lmb = mean_mse[np.argmin(np.array(mean_mse))]
        best_indices = np.where(data[:, lmb_index] == best_lmb[0])
        newdata = data[best_indices]
        return newdata

    elif type == 'lasso':
        alpha_index = headers.index('alpha')
        mse_index = headers.index('mse')
        alpha_values = np.unique(data[:, alpha_index])
        mean_mse = []
        for alpha in alpha_values:
            current_indices = np.where(data[:, alpha_index] == alpha)
            mean_mse.append([alpha, np.mean(data[current_indices, mse_index])])
        best_alpha = mean_mse[np.argmin(np.array(mean_mse))]
        best_indices = np.where(data[:, alpha_index] == best_alpha[0])
        newdata = data[best_indices]
        return newdata
    else:
        return 0

# Plotting functions


def plot_mse_vs_complexity(data, headers, model):
    """
    :param data: data as imported from csv files
    :param headers: headers provided with data
    :param model: string, type of regression. 'ols', 'ridge', or 'lasso'.
    :return: Nothing. Outputs plots and .tex to file.
    """

    # Get columns for mse, degrees, and noise
    mse_index = headers.index("mse")
    degree_index = headers.index("degree")
    noise_index = headers.index("noise")

    mse_vals = data[:, mse_index]
    degree_vals = data[:, degree_index]
    noise_vals = data[:, noise_index]

    # Ridge plots
    if model == 'ridge':
        param_index = headers.index('lambda')
        label = model + ", lambda = {}".format(data[0, param_index])
        noise_levels = np.unique(data[:, noise_index])
        for noise in noise_levels:
            n_idx = np.where(data[:,noise_index] == noise)[0]
            for 
            plt.plot(degree_vals[n_idx], mse_vals[n_idx], 'o-', label=label)

    # Lasso plots
    elif model == 'lasso':
        param_index = headers.index('alpha')
        label = model + ", alpha = {}".format(data[0, param_index])
        noise_levels = np.unique(data[:, noise_index])
        for noise in noise_levels:
            n_idx = np.where(data[:,noise_index] == noise)[0]
            plt.plot(degree_vals[n_idx], mse_vals[n_idx], 'o-', label=label)

    # OLS plots
    else:
        noise_levels = np.unique(data[:, noise_index])
        for noise in noise_levels:
            label = r'$\eta=$'+str(noise)
            n_idx = np.where(data[:,noise_index] == noise)[0]
            plt.plot(degree_vals[n_idx], mse_vals[n_idx], 'o-', label=label)





# Various plotting.
if len(sys.argv) > 0:
    for arg in sys.argv[1:]:
        with open("regression_data/{}.csv".format(arg)) as infile:
            headers = infile.readline().strip("\n\r").split(',')
            infile.close()
        data = np.loadtxt("regression_data/{}.csv".format(arg),
                          delimiter=',',
                          skiprows=1)

        if arg == "a":
            plot_mse_vs_complexity(data, headers, 'ols')

        elif arg == "b":
            newdata = determine_best_mse(data, headers, 'ridge')
            plot_mse_vs_complexity(newdata, headers, 'ridge')

        elif arg == "c":
            newdata = determine_best_mse(data, headers, 'lasso')
            plot_mse_vs_complexity(newdata, headers, 'lasso')
        elif arg == "e_ols":
            plot_mse_vs_complexity(data, headers, 'ols')
        elif arg == "e_ridge":
            newdata = determine_best_mse(data, headers, 'ridge')
            plot_mse_vs_complexity(newdata, headers, 'ridge')
        elif arg == "e_lasso":
            newdata = determine_best_mse(data, headers, 'lasso')
            plot_mse_vs_complexity(newdata, headers, 'lasso')


    plt.title('Mean Squared Error vs. Complexity of model')
    plt.xlabel('Degree of fitted polynomial')
    plt.ylabel('Mean squared error')
    plt.legend()
    figname = "mse_vs_complexity_{}.pdf".format(sys.argv[1])
    plt.savefig(figname, format='pdf')
    plt.show()
