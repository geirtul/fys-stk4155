import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def plot_coeffs_nonoise(task, degree):
    """ Plot the coefficients with confidence interval based on
    variances from bootstrap resampling. Plots without noise

    :param task: which subtask of project.
    :param degree: degree of polynomial that was fit, can be list of degrees
    """


    # Regression file contains: noise_level, error, bias, variance, coeffs, var_coeff
    DATA_PATH="regression_data/"

    # If degree is a list, create subplots instead of just one
    if isinstance(degree, list):
        fig, ax = plt.subplots(len(degree), 1, figsize=(12,12))
        for i in range(len(degree)):
            fname = "{}_bootstrap_d{}.npy".format(task, degree[i])
            data = np.load(DATA_PATH+fname)
            
            num_coeff = int(len(data[0,4:])/2)
            coeff = data[0, 4:4+num_coeff]
            var_coeff = data[0, 4+num_coeff:]
            ci_coeff = np.sqrt(var_coeff)*1.96 # 95% Confidence Interval
            label = "deg(P) = {}".format(degree[i])

            # Generate x-axis ticks
            x_ticks = [r'$\beta_{%d}$'%i for i in range(num_coeff)]
            ax[i].errorbar(
                np.arange(num_coeff),
                y=coeff, 
                yerr=ci_coeff,
                label=label,
                fmt='o',
                markersize=4,
                linewidth=1,
                capsize=5,
                capthick=1,
                ecolor="black",
                )
            ax[i].set_xticks(np.arange(num_coeff))
            ax[i].set_xticklabels(x_ticks)
            ax[i].legend()

        fig.suptitle(r'Coefficients $\beta$ for deg(P) $\in$ {1,2,3,4,5}')
        plt.xlabel("Coefficient")
        plt.show()

    else:
        fname = "regression_data/{}_bootstrap_d{}.npy".format(task, degree)

        data = np.load(fname)
        num_coeff = int(len(data[0,4:])/2)

        coeff = data[0, 4:4+num_coeff]
        var_coeff = data[0, 4+num_coeff:]
        ci_coeff = np.sqrt(var_coeff)*1.96 # 95% Confidence Interval
        label = "deg(P) = {}".format(degree)

        # Generate x-axis ticks
        x_ticks = [r'$\beta_{%d}$'%i for i in range(num_coeff)]

        plt.errorbar(
                np.arange(num_coeff),
                y=coeff, 
                yerr=ci_coeff,
                label=label,
                fmt='o',
                markersize=4,
                linewidth=1,
                capsize=5,
                capthick=1,
                ecolor="black",
                )
        plt.xlabel("Coefficient")
        plt.ylabel("Value of Coefficient")
        plt.xticks(np.arange(num_coeff), x_ticks)
        plt.legend()
        plt.show()

def plot_coeffs_noise(task, degree):
    """ Plot the coefficients with confidence interval based on
    variances from bootstrap resampling. Plots with noise level for
    one specific degree.

    :param task: which subtask of project.
    :param degree: degree of polynomial that was fit
    """


    # Regression file contains: noise_level, error, bias, variance, coeffs, var_coeff

    # If degree is a list, create subplots instead of just one
    DATA_PATH="regression_data/"
    fname = "{}_bootstrap_d{}.npy".format(task, degree)
    data = np.load(DATA_PATH+fname)
    noise_levels = np.unique(data[:,0])
    fig, ax = plt.subplots(len(noise_levels), 1, figsize=(12,12))
    for i in range(len(noise_levels)):
        tmp_data = data[np.where(data[:,0] == noise_levels[i])]
        num_coeff = int(len(tmp_data[0,4:])/2)

        coeff = tmp_data[0, 4:4+num_coeff]
        var_coeff = tmp_data[0, 4+num_coeff:]
        ci_coeff = np.sqrt(var_coeff)*1.96 # 95% Confidence Interval
        label = "noise = {}".format(noise_levels[i])

        # Generate x-axis ticks
        x_ticks = [r'$\beta_{%d}$'%j for j in range(num_coeff)]
        ax[i].errorbar(
            np.arange(num_coeff),
            y=coeff, 
            yerr=ci_coeff,
            label=label,
            fmt='o',
            markersize=4,
            linewidth=1,
            capsize=5,
            capthick=1,
            ecolor="black",
            )
        ax[i].set_xticks(np.arange(num_coeff))
        ax[i].set_xticklabels(x_ticks)
        ax[i].legend()

    fig.suptitle(r'Coefficients $\beta$ for deg(P) = {}, with noise'.format(degree))
    plt.xlabel("Coefficient")
    plt.show()

def plot_bias_variance(task, degree):
    """ Plot the bias and variance results from bootstrapping
    as a function of polynomial degree.

    TODO: Implement two y-axes from
    https://matplotlib.org/gallery/api/two_scales.html
    :param task: which subtask of project.
    :param degree: list of degrees
    """


    DATA_PATH="regression_data/"
    # Regression file contains: noise_level, error, bias, variance, coeffs, var_coeff

    plot_data = {}

    # Extract data from the files and store them sorted by noise_level
    for i in range(len(degree)):
        fname = "{}_bootstrap_d{}.npy".format(task, degree[i])
        data = np.load(DATA_PATH+fname)
    
        noise_levels = np.unique(data[:,0])
        for j in range(len(noise_levels)):
            tmp_data = data[np.where(data[:,0] == noise_levels[j])]
            if noise_levels[j] not in plot_data.keys():
                plot_data[noise_levels[j]] = {}
                plot_data[noise_levels[j]][degree[i]] = tmp_data[0,2:4]
            else:
                plot_data[noise_levels[j]][degree[i]] = tmp_data[0,2:4]

    for i, noise_level in enumerate(sorted(plot_data.keys())):
        fig, ax = plt.subplots()
        bias = []
        variance = []
        for deg in sorted(plot_data[noise_level].keys()):
            bias.append(plot_data[noise_level][deg][0])
            variance.append(plot_data[noise_level][deg][1])

        color = 'tab:red'
        #ax[i].set_xlabel('Degree of polynomial')
        #ax[i].set_ylabel('Bias', color=color)
        #ax[i].plot(
        #        np.arange(len(bias)), 
        #        bias, 
        #        label="bias", 
        #        color=color)
        #ax[i].tick_params(axis='y', labelcolor=color)

        ## instantiate a second axes that shares the same x-axis
        #ax2 = ax[i].twinx()          
        
        ax.set_xlabel('Degree of polynomial')
        ax.set_ylabel('Bias', color=color)
        ax.plot(
                np.arange(len(bias)), 
                bias, 
                label="bias", 
                color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_xticks(np.arange(len(degree)))
        ax.set_xticklabels(np.arange(1,len(degree)+1))

        # instantiate a second axes that shares the same x-axis
        ax2 = ax.twinx()          
        color = 'tab:blue'
        # we already handled the x-label with ax1
        ax2.set_ylabel('Variance', color=color)          
        ax2.plot(
                np.arange(len(variance)), 
                variance, 
                label="variance", 
                color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title("Noise level = " + str(noise_level))
        plt.show()


def plot_mse(task, degree):
    """ Plot the MSE results from bootstrapping
    as a function of polynomial degree and added noise.

    :param task: which subtask of project.
    :param degree: list of degrees
    """


    DATA_PATH="regression_data/"
    # Regression file contains: noise_level, error, bias, variance, coeffs, var_coeff

    plot_data = {}

    # Extract data from the files and store them sorted by noise_level
    for i in range(len(degree)):
        fname = "{}_bootstrap_d{}.npy".format(task, degree[i])
        data = np.load(DATA_PATH+fname)
        noise_levels = np.unique(data[:,0])
        for j in range(len(noise_levels)):
            tmp_data = data[np.where(data[:,0] == noise_levels[j])]
            if noise_levels[j] not in plot_data.keys():
                plot_data[noise_levels[j]] = np.zeros(len(degree))
                plot_data[noise_levels[j]][i] = tmp_data[0, 1]
            else:
                plot_data[noise_levels[j]][i] = tmp_data[0, 1]

    fig, ax = plt.subplots(len(plot_data.keys()), 1, figsize=(12,12), sharex=True)
    noise_levels = sorted(plot_data.keys())
    for i, key in enumerate(sorted(plot_data.keys())):
        label = "noise = {}".format(key)
        ax[i].plot(
                np.arange(len(degree)), 
                plot_data[key],
                'o--',
                label=label,
                )
        ax[i].set_xticks(range(5))
        ax[i].set_xticklabels(range(1, 6, 1))
        ax[i].set_ylabel("MSE")
        ax[i].legend()
    plt.xlabel("Degree of fitted polynomial")
    plt.show()
            

# Output plots for each subtask in the project
if len(sys.argv) > 0:
    for arg in sys.argv[1:]:
        if arg == "a":
            plot_coeffs_nonoise(arg, [1,2,3,4,5])
            plot_coeffs_noise(arg, 3)
        if arg == "test":
            plot_bias_variance("a", [1,2,3,4,5])
            #plot_mse("a", [1,2,3,4,5])

