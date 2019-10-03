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
            
            num_coeff = int(len(data[0,8:])/2)
            coeff = data[0, 8:8+num_coeff]
            var_coeff = data[0, 8+num_coeff:]
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
        fig.text(0.06, 0.5, 'Coefficient value', ha='center', va='center', rotation='vertical')
        plt.xlabel("Coefficient")
        plt.show()

    else:
        fname = "regression_data/{}_bootstrap_d{}.npy".format(task, degree)

        data = np.load(fname)
        num_coeff = int(len(data[0,8:])/2)

        coeff = data[0, 8:8+num_coeff]
        var_coeff = data[0, 8+num_coeff:]
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
        num_coeff = int(len(tmp_data[0,8:])/2)

        coeff = tmp_data[0, 8:8+num_coeff]
        var_coeff = tmp_data[0, 8+num_coeff:]
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
    fig.text(0.06, 0.5, 'Coefficient value', ha='center', va='center', rotation='vertical')
    plt.xlabel("Coefficient")
    plt.show()

def plot_errors(task, degree):
    """ Plot e_in and e_out results from bootstrapping
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
                plot_data[noise_levels[j]][degree[i]] = tmp_data[0,1:3]
            else:
                plot_data[noise_levels[j]][degree[i]] = tmp_data[0,1:3]

    for i, noise_level in enumerate(sorted(plot_data.keys())):
        fig, ax = plt.subplots()
        e_in = []
        e_out = []
        for deg in sorted(plot_data[noise_level].keys()):
            e_in.append(plot_data[noise_level][deg][0])
            e_out.append(plot_data[noise_level][deg][1])

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
        ax.set_ylabel('e_in', color=color)
        ax.plot(
                np.arange(len(e_in)), 
                e_in, 
                'o--',
                label="e_in", 
                color=color,
                )
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_xticks(np.arange(len(degree)))
        ax.set_xticklabels(np.arange(1,len(degree)+1))

        # instantiate a second axes that shares the same x-axis
        ax2 = ax.twinx()          
        color = 'tab:blue'
        # we already handled the x-label with ax1
        ax2.set_ylabel('e_out', color=color)          
        ax2.plot(
                np.arange(len(e_out)), 
                e_out, 
                'o--',
                label="e_out", 
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
                plot_data[noise_levels[j]] = np.zeros((len(degree),2))
                plot_data[noise_levels[j]][i] = tmp_data[0, 2:4]
            else:
                plot_data[noise_levels[j]][i] = tmp_data[0, 2:4]

    fig, ax = plt.subplots(len(plot_data.keys()), 1, figsize=(12,12), sharex=True)
    noise_levels = sorted(plot_data.keys())
    for i, key in enumerate(sorted(plot_data.keys())):
        label = "noise = {}".format(key)
        var_error=plot_data[key][:,1]
        ci_error = np.sqrt(var_error)*1.96 # 95% Confidence Interval
        ax[i].errorbar(
                np.arange(len(degree)), 
                plot_data[key][:,0],
                yerr=ci_error,
                label=label,
                fmt='o',
                markersize=4,
                linewidth=1,
                capsize=5,
                capthick=1,
                ecolor="black",
                )
        ax[i].set_xticks(range(5))
        ax[i].set_xticklabels(range(1, 6, 1))
        ax[i].legend()
    fig.suptitle("Mean squared error as a function of model complexity")
    fig.text(0.06, 0.5, 'Mean Squared Error', ha='center', va='center', rotation='vertical')
    plt.xlabel("Degree of fitted polynomial")
    plt.show()


def plot_r2(task, degree):
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
                plot_data[noise_levels[j]] = np.zeros((len(degree),2))
                plot_data[noise_levels[j]][i] = tmp_data[0, 4:6]
            else:
                plot_data[noise_levels[j]][i] = tmp_data[0, 4:6]

    fig, ax = plt.subplots(len(plot_data.keys()), 1, figsize=(12,12), sharex=True)
    noise_levels = sorted(plot_data.keys())
    for i, key in enumerate(sorted(plot_data.keys())):
        label = "R2, noise = {}".format(key)
        var_r2 = plot_data[key][:,1]
        ci_r2 = np.sqrt(var_r2)*1.96 # 95% Confidence Interval
        ax[i].errorbar(
                np.arange(len(degree)), 
                plot_data[key][:,0],
                yerr=ci_r2,
                label=label,
                fmt='o',
                markersize=4,
                linewidth=1,
                capsize=5,
                capthick=1,
                ecolor="black",
                )
        ax[i].set_xticks(range(5))
        ax[i].set_xticklabels(range(1, 6, 1))
        ax[i].legend()
    fig.suptitle("R2 Score as a function of model complexity")
    fig.text(0.06, 0.5, 'R2 Score', ha='center', va='center', rotation='vertical')
    plt.xlabel("Degree of fitted polynomial")
    plt.show()

# Output plots for each subtask in the project
if len(sys.argv) > 0:
    for arg in sys.argv[1:]:
        if arg == "a":
            plot_coeffs_nonoise(arg, [1,2,3,4,5])
            plot_coeffs_noise(arg, 3)
        if arg == "test":
            #plot_errors("a", [1,2,3,4,5])
            plot_mse("a", [1,2,3,4,5])
            plot_r2("a", [1,2,3,4,5])

