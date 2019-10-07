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
    FIGURE_PATH = "../report/figures/"
    
    if task == "ols":
        # If degree is a list, create subplots instead of just one
        if isinstance(degree, list):
            fig, ax = plt.subplots(len(degree), 1, figsize=(10,10))
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

            fig.suptitle(r'OLS Coefficients $\beta$ for deg(P) $\in$ {1,2,3,4,5}')
            fig.text(0.06, 0.5, 'Coefficient value', ha='center', va='center', rotation='vertical')
            plt.xlabel("Coefficient")
            
            #plt.show()
            figname = "coeffs_vs_complexity_{}.pdf".format(task)
            plt.savefig(FIGURE_PATH+figname, format="pdf")

    # With ridge we have four lines per lambda, where the first line has
    # no noise added, and noise increases with each line
    if task == "ridge":
        
        # Load all the datafiles into a dict
        all_data = {}
        for i in range(len(degree)):
            fname = "{}_bootstrap_d{}.npy".format(task, degree[i])
            data = np.load(DATA_PATH+fname)
            all_data[degree[i]] = data

        # Get number of lambdas
        lmbdas = np.unique(all_data[degree[0]][:,0])

        for j in range(len(lmbdas)):
            fig, ax = plt.subplots(len(degree), 1, figsize=(10,10))
            for i in range(len(degree)):
                data = all_data[degree[i]]
                    
                # Get indices where data for the current lmbda is located.
                tmp_data = data[np.where(data[:,0] == lmbdas[j])[0]]
                num_coeff = int(len(tmp_data[0,8:])/2)
                coeff = tmp_data[0, 8:8+num_coeff]
                var_coeff = tmp_data[0, 8+num_coeff:]
                ci_coeff = np.sqrt(var_coeff)*1.96 # 95% Confidence Interval
                label = 'deg(P) = {}'.format(degree[i])

                # Generate x-axis ticks
                x_ticks = [r'$\beta_{%d}$'%k for k in range(num_coeff)]
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

            fig.suptitle(r'Ridge Coefficients $\beta$ for deg(P) = 1,2,3,4,5 and $\lambda = ${}'.format(lmbdas[j]))
            fig.text(0.06, 0.5, 'Coefficient value', ha='center', va='center', rotation='vertical')
            fig.tight_layout()
            plt.xlabel("Coefficient")
        
            #plt.show()
            figname = "coeffs_vs_complexity_{}_lmd_{}.pdf".format(task, lmbdas[j])
            plt.savefig(FIGURE_PATH+figname, format="pdf")

    if task == "lasso":
        
        # Load all the datafiles into a dict
        all_data = {}
        for i in range(len(degree)):
            fname = "{}_bootstrap_d{}.npy".format(task, degree[i])
            data = np.load(DATA_PATH+fname)
            all_data[degree[i]] = data

        # Get the alpha values
        alphas = np.unique(all_data[degree[0]][:,0])

        for j in range(len(alphas)):
            fig, ax = plt.subplots(len(degree), 1, figsize=(10,10))
            for i in range(len(degree)):
                data = all_data[degree[i]]
                    
                # Get indices where data for the current lmbda is located.
                tmp_data = data[np.where(data[:,0] == alphas[j])[0]]
                num_coeff = int(len(tmp_data[0,8:])/2)
                coeff = tmp_data[0, 8:8+num_coeff]
                var_coeff = tmp_data[0, 8+num_coeff:]
                ci_coeff = np.sqrt(var_coeff)*1.96 # 95% Confidence Interval
                label = 'deg(P) = {}'.format(degree[i])

                # Generate x-axis ticks
                x_ticks = [r'$\beta_{%d}$'%k for k in range(num_coeff)]
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

            fig.suptitle(r'Lasso Coefficients $\beta$ for deg(P) = 1,2,3,4,5 and $\alpha = ${}'.format(alphas[j]))
            fig.text(0.06, 0.5, 'Coefficient value', ha='center', va='center', rotation='vertical')
            fig.tight_layout()
            plt.xlabel("Coefficient")
        
            #plt.show()
            figname = "coeffs_vs_complexity_{}_a_{}.pdf".format(task, alphas[j])
            plt.savefig(FIGURE_PATH+figname, format="pdf")

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
    FIGURE_PATH = "../report/figures/"

    if task == "ols":
        fname = "{}_bootstrap_d{}.npy".format(task, degree)
        data = np.load(DATA_PATH+fname)
        noise_levels = np.unique(data[:,0])
        fig, ax = plt.subplots(len(noise_levels), 1, figsize=(10,10))
        for i in range(len(noise_levels)):
            tmp_data = data[np.where(data[:,0] == noise_levels[i])]
            num_coeff = int(len(tmp_data[0,8:])/2)

            coeff = tmp_data[0, 8:8+num_coeff]
            var_coeff = tmp_data[0, 8+num_coeff:]
            ci_coeff = np.sqrt(var_coeff)*1.96 # 95% Confidence Interval
            label = "Noise = {}".format(noise_levels[i])

            # Generate x-axis ticks
            x_ticks = [r'$\beta_{%d}$'%j for j in range(num_coeff)]
            ax[i].errorbar(
                np.arange(num_coeff),
                y=coeff, 
                yerr=ci_coeff,
                fmt='o',
                label=label,
                markersize=4,
                linewidth=1,
                capsize=5,
                capthick=1,
                ecolor="black",
                )
            ax[i].set_xticks(np.arange(num_coeff))
            ax[i].set_xticklabels(x_ticks)
            ax[i].legend()

        fig.suptitle(r'Coefficients $\beta$ for deg(P) = {}, with added stochastic noise'.format(degree))
        fig.text(0.06, 0.5, 'Coefficient value', ha='center', va='center', rotation='vertical')
        plt.xlabel("Coefficient")
        
        #plt.show()
        figname = "best_coeff_vs_noise_{}.pdf".format(task)
        plt.savefig(FIGURE_PATH+figname, format="pdf")

    # With ridge we have four lines per lambda, where the first line has
    # no noise added, and noise increases with each line
    if task == "ridge":
        
        # Load all the datafiles into a dict
        all_data = {}
        for i in range(len(degree)):
            fname = "{}_bootstrap_d{}.npy".format(task, degree[i])
            data = np.load(DATA_PATH+fname)
            all_data[degree[i]] = data

        # Get number of lambdas.
        lmbdas = np.unique(all_data[degree[0]][:,0])
        
        # There are 5 levels of noise for each lambda. Since we've already
        # plotted the no-noise plots, we only plot those with added noise.
        noise_levels = [0, 0.1, 0.2, 0.5, 1.0]
        for k in range(1, len(noise_levels)):
            for j in range(len(lmbdas)):
                fig, ax = plt.subplots(len(degree), 1, figsize=(10,10))
                for i in range(len(degree)):
                    data = all_data[degree[i]]
                        
                    # Get indices where data for the current lmbda is located.
                    tmp_data = data[np.where(data[:,0] == lmbdas[j])[0]]
                    num_coeff = int(len(tmp_data[k,8:])/2)
                    coeff = tmp_data[k, 8:8+num_coeff]
                    var_coeff = tmp_data[k, 8+num_coeff:]
                    ci_coeff = np.sqrt(var_coeff)*1.96 # 95% Confidence Interval
                    label = 'deg(P) = {}'.format(degree[i])

                    # Generate x-axis ticks
                    x_ticks = [r'$\beta_{%d}$'%tick for tick in range(num_coeff)]
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

                fig.suptitle(r'Ridge Coefficients $\beta$ for deg(P) = 1,2,3,4,5 and $\lambda = ${}, and noise = {}'.format(lmbdas[j], noise_levels[k]))
                fig.text(0.06, 0.5, 'Coefficient value', ha='center', va='center', rotation='vertical')
                plt.xlabel("Coefficient")
            
                #plt.show()
                figname = "coeffs_vs_complexity_noise_{}_lmd_{}_noise_{}.pdf".format(task, lmbdas[j], noise_levels[k])
                plt.savefig(FIGURE_PATH+figname, format="pdf")

    if task == "lasso":
        
        # Load all the datafiles into a dict
        all_data = {}
        for i in range(len(degree)):
            fname = "{}_bootstrap_d{}.npy".format(task, degree[i])
            data = np.load(DATA_PATH+fname)
            all_data[degree[i]] = data

        # Get number of lambdas.
        alphas = np.unique(all_data[degree[0]][:,0])
        
        # There are 5 levels of noise for each lambda. Since we've already
        # plotted the no-noise plots, we only plot those with added noise.
        noise_levels = [0, 0.1, 0.2, 0.5, 1.0]
        for k in range(1, len(noise_levels)):
            for j in range(len(alphas)):
                fig, ax = plt.subplots(len(degree), 1, figsize=(10,10))
                for i in range(len(degree)):
                    data = all_data[degree[i]]
                        
                    # Get indices where data for the current lmbda is located.
                    tmp_data = data[np.where(data[:,0] == alphas[j])[0]]
                    num_coeff = int(len(tmp_data[k,8:])/2)
                    coeff = tmp_data[k, 8:8+num_coeff]
                    var_coeff = tmp_data[k, 8+num_coeff:]
                    ci_coeff = np.sqrt(var_coeff)*1.96 # 95% Confidence Interval
                    label = 'deg(P) = {}'.format(degree[i])

                    # Generate x-axis ticks
                    x_ticks = [r'$\beta_{%d}$'%tick for tick in range(num_coeff)]
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

                fig.suptitle(r'Lasso Coefficients $\beta$ for deg(P) = 1,2,3,4,5 and $\lambda = ${}, and noise = {}'.format(alphas[j], noise_levels[k]))
                fig.text(0.06, 0.5, 'Coefficient value', ha='center', va='center', rotation='vertical')
                plt.xlabel("Coefficient")
            
                #plt.show()
                figname = "coeffs_vs_complexity_noise_{}_a_{}_noise_{}.pdf".format(task, alphas[j], noise_levels[k])
                plt.savefig(FIGURE_PATH+figname, format="pdf")


def plot_errors(task, degree):
    """ Plot e_in and e_out results from bootstrapping
    as a function of polynomial degree.

    :param task: which subtask of project.
    :param degree: list of degrees
    """


    DATA_PATH="regression_data/"
    FIGURE_PATH = "../report/figures/"
    # Regression file contains: noise_level, error, bias, variance, coeffs, var_coeff

    if task == "ols":
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

        # Plotting
        fig, ax = plt.subplots(len(degree), 1, figsize=(10,10), sharex=True)
        for i, noise_level in enumerate(sorted(plot_data.keys())):
            e_in = []
            e_out = []
            for deg in sorted(plot_data[noise_level].keys()):
                e_in.append(plot_data[noise_level][deg][0])
                e_out.append(plot_data[noise_level][deg][1])

            color = 'tab:red'
            ax[i].set_ylabel('e_in', color=color)
            ax[i].plot(
                    np.arange(len(e_in)), 
                    e_in, 
                    '^--',
                    color=color,
                    )
            ax[i].tick_params(axis='y', labelcolor=color)
            ax[i].set_xticks(np.arange(len(degree)))
            ax[i].set_xticklabels(np.arange(1,len(degree)+1))
            ax[i].set_title("Noise = {}".format(noise_level))

            # instantiate a second axes that shares the same x-axis
            ax2 = ax[i].twinx()          
            color = 'tab:blue'
            # we already handled the x-label with ax1
            ax2.set_ylabel('e_out', color=color)          
            ax2.plot(
                    np.arange(len(e_out)), 
                    e_out, 
                    'o--',
                    color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        fig.suptitle("In-sample error and out-of-sample error as a function of model complexity")
        plt.xlabel('Degree of polynomial')
        
        #plt.show()

        figname = "errors_vs_complexity_{}.pdf".format(task)
        plt.savefig(FIGURE_PATH+figname, format="pdf")

    if task == "ridge":
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

        # Plotting
        fig, ax = plt.subplots(len(degree), 1, figsize=(10,10), sharex=True)
        for i, noise_level in enumerate(sorted(plot_data.keys())):
            e_in = []
            e_out = []
            for deg in sorted(plot_data[noise_level].keys()):
                e_in.append(plot_data[noise_level][deg][0])
                e_out.append(plot_data[noise_level][deg][1])

            color = 'tab:red'
            ax[i].set_ylabel('e_in', color=color)
            ax[i].plot(
                    np.arange(len(e_in)), 
                    e_in, 
                    '^--',
                    color=color,
                    )
            ax[i].tick_params(axis='y', labelcolor=color)
            ax[i].set_xticks(np.arange(len(degree)))
            ax[i].set_xticklabels(np.arange(1,len(degree)+1))
            ax[i].set_title("Noise = {}".format(noise_level))

            # instantiate a second axes that shares the same x-axis
            ax2 = ax[i].twinx()          
            color = 'tab:blue'
            # we already handled the x-label with ax1
            ax2.set_ylabel('e_out', color=color)          
            ax2.plot(
                    np.arange(len(e_out)), 
                    e_out, 
                    'o--',
                    color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        fig.suptitle("In-sample error and out-of-sample error as a function of model complexity")
        plt.xlabel('Degree of polynomial')
        
        #plt.show()

        figname = "errors_vs_complexity_{}.pdf".format(task)
        plt.savefig(FIGURE_PATH+figname, format="pdf")

def plot_mse(task, degree):
    """ Plot the MSE results from bootstrapping
    as a function of polynomial degree and added noise.

    :param task: which subtask of project.
    :param degree: list of degrees
    """


    DATA_PATH="regression_data/"
    FIGURE_PATH = "../report/figures/"
    # Regression file contains: noise_level, error, bias, variance, coeffs, var_coeff

    if task == "ols":
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

        fig, ax = plt.subplots(len(plot_data.keys()), 1, figsize=(10,10), sharex=True)
        noise_levels = sorted(plot_data.keys())
        for i, key in enumerate(sorted(plot_data.keys())):
            label = "Noise = {}".format(key)
            var_error=plot_data[key][:,1]
            ci_error = np.sqrt(var_error)*1.96 # 95% Confidence Interval
            ax[i].errorbar(
                    np.arange(len(degree)), 
                    plot_data[key][:,0],
                    yerr=ci_error,
                    fmt='o',
                    markersize=4,
                    linewidth=1,
                    capsize=5,
                    capthick=1,
                    ecolor="black",
                    )
            ax[i].set_xticks(range(5))
            ax[i].set_xticklabels(range(1, 6, 1))
            ax[i].set_title(label)
        fig.suptitle("Mean squared error as a function of model complexity")
        fig.text(0.06, 0.5, 'Mean Squared Error', ha='center', va='center', rotation='vertical')
        plt.xlabel("Degree of fitted polynomial")
        
        #plt.show()
        figname = "mse_vs_complexity_{}.pdf".format(task)
        plt.savefig(FIGURE_PATH+figname, format="pdf")

    if task == "ridge":
        
        # Load all the datafiles into a dict
        all_data = {}
        for i in range(len(degree)):
            fname = "{}_bootstrap_d{}.npy".format(task, degree[i])
            data = np.load(DATA_PATH+fname)
            all_data[degree[i]] = data

        # Get number of lambdas.
        lmbdas = np.unique(all_data[degree[0]][:,0])

        noise_levels = [0, 0.1, 0.2, 0.5, 1.0]
        plot_data = np.zeros((len(lmbdas), len(degree), 2))
       
        for noise_idx in range(len(noise_levels)):
            for i in range(len(lmbdas)):
                for j in range(len(degree)):
                    data = all_data[degree[j]]
                    data = data[np.where(data[:,0] == lmbdas[i])]
                    error = data[noise_idx, 2]
                    var_error = data[noise_idx, 3]
                    ci_error = np.sqrt(var_error)*1.96 # 95% Confidence Interval
                    plot_data[i,j,0] = error
                    plot_data[i,j,1] = ci_error

            # Skip plotting lmbda = 0 
            fig, ax = plt.subplots(len(lmbdas), 1, figsize=(10,10))
            for j in range(len(lmbdas)):
                label = r'$\lambda$ = {}'.format(lmbdas[j])

                # Generate x-axis ticks
                #x_ticks = [r'$\beta_{%d}$'%tick for tick in range(num_coeff)]
                errors = plot_data[j,:,0]
                ci_errors = plot_data[j,:,1]
                ax[j].errorbar(
                    np.arange(len(degree)),
                    y=errors, 
                    yerr=ci_errors,
                    label=label,
                    fmt='o',
                    markersize=4,
                    linewidth=1,
                    capsize=5,
                    capthick=1,
                    ecolor="black",
                    )
                ax[j].set_xticks(np.arange(len(degree)))
                ax[j].set_xticklabels(np.arange(1, len(degree)+1))
                ax[j].legend()

            fig.suptitle(r'Ridge MSE as a function of model complexity deg(P) with added noise = {}'.format(noise_levels[noise_idx]))
            fig.text(0.06, 0.5, 'Mean Squared Error', ha='center', va='center', rotation='vertical')
            plt.xlabel("Degree of polynomial")
        
            #plt.show()
            figname = "coeffs_vs_complexity_noise_{}_lmd_{}_noise_{}.pdf".format(task, lmbdas[j], noise_levels[noise_idx])
            plt.savefig(FIGURE_PATH+figname, format="pdf")
    
    if task == "lasso":
        
        # Load all the datafiles into a dict
        all_data = {}
        for i in range(len(degree)):
            fname = "{}_bootstrap_d{}.npy".format(task, degree[i])
            data = np.load(DATA_PATH+fname)
            all_data[degree[i]] = data

        # Get number of lambdas.
        alphas = np.unique(all_data[degree[0]][:,0])

        plot_data = np.zeros((len(alphas), len(degree), 2))
        
        noise_levels = [0, 0.1, 0.2, 0.5, 1.0]
        for noise_idx in range(len(noise_levels)):
            for i in range(len(alphas)):
                for j in range(len(degree)):
                    data = all_data[degree[j]]
                    data = data[np.where(data[:,0] == alphas[i])]
                    error = data[noise_idx,2]
                    var_error = data[noise_idx,3]
                    ci_error = np.sqrt(var_error)*1.96 # 95% Confidence Interval
                    plot_data[i,j,0] = error
                    plot_data[i,j,1] = ci_error

            # Skip plotting lmbda = 0 
            fig, ax = plt.subplots(len(alphas), 1, figsize=(10,10))
            for j in range(len(alphas)):
                label = r'$\alpha$ = {}'.format(alphas[j])

                # Generate x-axis ticks
                #x_ticks = [r'$\beta_{%d}$'%tick for tick in range(num_coeff)]
                errors = plot_data[j,:,0]
                ci_errors = plot_data[j,:,1]
                ax[j].errorbar(
                    np.arange(len(degree)),
                    y=errors, 
                    yerr=ci_errors,
                    label=label,
                    fmt='o',
                    markersize=4,
                    linewidth=1,
                    capsize=5,
                    capthick=1,
                    ecolor="black",
                    )
                ax[j].set_xticks(np.arange(len(degree)))
                ax[j].set_xticklabels(np.arange(1, len(degree)+1))
                ax[j].legend()

            fig.suptitle(r'Lasso MSE as a function of model complexity deg(P), with added noise = {}'.format(noise_levels[noise_idx]))
            fig.text(0.06, 0.5, 'Mean Squared Error', ha='center', va='center', rotation='vertical')
            plt.xlabel("Degree of polynomial")
        
            #plt.show()
            figname = "coeffs_vs_complexity_noise_{}_a_{}_noise_{}.pdf".format(task, alphas[j], noise_levels[noise_idx])
            plt.savefig(FIGURE_PATH+figname, format="pdf")

def plot_r2(task, degree):
    """ Plot the R2 results from bootstrapping
    as a function of polynomial degree and added noise.

    :param task: which subtask of project.
    :param degree: list of degrees
    """


    DATA_PATH = "regression_data/"
    FIGURE_PATH = "../report/figures/"
    # Regression file contains: noise_level, error, bias, variance, coeffs, var_coeff

    if task == "ols":
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

        fig, ax = plt.subplots(len(plot_data.keys()), 1, figsize=(10,10), sharex=True)
        noise_levels = sorted(plot_data.keys())
        for i, key in enumerate(sorted(plot_data.keys())):
            label = "Noise = {}".format(key)
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
            ax[i].set_title(label)
        fig.suptitle("R2 Score as a function of model complexity")
        fig.text(0.06, 0.5, 'R2 Score', ha='center', va='center', rotation='vertical')
        plt.xlabel("Degree of fitted polynomial")
        
        figname = "r2_vs_complexity_{}.pdf".format(task)
        #plt.show()
        plt.savefig(FIGURE_PATH+figname, format="pdf")

    if task == "ridge":
        
        # Load all the datafiles into a dict
        all_data = {}
        for i in range(len(degree)):
            fname = "{}_bootstrap_d{}.npy".format(task, degree[i])
            data = np.load(DATA_PATH+fname)
            all_data[degree[i]] = data

        # Get number of lambdas.
        lmbdas = np.unique(all_data[degree[0]][:,0])

        plot_data = np.zeros((len(lmbdas), len(degree), 2))
        
        noise_levels = [0, 0.1, 0.2, 0.5, 1.0]
        for noise_idx in range(len(noise_levels)):
            for i in range(len(lmbdas)):
                for j in range(len(degree)):
                    data = all_data[degree[j]]
                    data = data[np.where(data[:,0] == lmbdas[i])]
                    r2 = data[noise_idx, 4]
                    var_r2 = data[noise_idx, 5]
                    ci_r2 = np.sqrt(var_r2)*1.96 # 95% Confidence Interval
                    plot_data[i,j,0] = r2
                    plot_data[i,j,1] = ci_r2

            # Skip plotting lmbda = 0 
            fig, ax = plt.subplots(len(lmbdas), 1, figsize=(10,10))
            for j in range(len(lmbdas)):
                label = r'$\lambda$ = {}'.format(lmbdas[j])

                # Generate x-axis ticks
                #x_ticks = [r'$\beta_{%d}$'%tick for tick in range(num_coeff)]
                r2 = plot_data[j,:,0]
                ci_r2 = plot_data[j,:,1]
                ax[j].errorbar(
                    np.arange(len(degree)),
                    y=r2, 
                    yerr=ci_r2,
                    label=label,
                    fmt='o',
                    markersize=4,
                    linewidth=1,
                    capsize=5,
                    capthick=1,
                    ecolor="black",
                    )
                ax[j].set_xticks(np.arange(len(degree)))
                ax[j].set_xticklabels(np.arange(1,len(degree)+1))
                ax[j].legend()

            fig.suptitle(r'Ridge R2 Score as a function of model complexity deg(P) with added noise = {}'.format(noise_levels[noise_idx]))
            fig.text(0.06, 0.5, 'R2 Score', ha='center', va='center', rotation='vertical')
            plt.xlabel("Degree of polynomial")
        
            #plt.show()
            figname = "coeffs_vs_complexity_noise_{}_lmd_{}_noise_{}.pdf".format(task, lmbdas[j], noise_levels[noise_idx])
            plt.savefig(FIGURE_PATH+figname, format="pdf")
    
    if task == "lasso":
        
        # Load all the datafiles into a dict
        all_data = {}
        for i in range(len(degree)):
            fname = "{}_bootstrap_d{}.npy".format(task, degree[i])
            data = np.load(DATA_PATH+fname)
            all_data[degree[i]] = data

        # Get number of alphas.
        alphas = np.unique(all_data[degree[0]][:,0])

        plot_data = np.zeros((len(alphas), len(degree), 2))        
        noise_levels = [0, 0.1, 0.2, 0.5, 1.0]
        for noise_idx in range(len(noise_levels)):
            for i in range(len(alphas)):
                for j in range(len(degree)):
                    data = all_data[degree[j]]
                    data = data[np.where(data[:,0] == alphas[i])]
                    r2 = data[noise_idx, 4]
                    var_r2 = data[noise_idx, 5]
                    ci_r2 = np.sqrt(var_r2)*1.96 # 95% Confidence Interval
                    plot_data[i,j,0] = r2
                    plot_data[i,j,1] = ci_r2

            # Skip plotting lmbda = 0 
            fig, ax = plt.subplots(len(alphas), 1, figsize=(10,10))
            for j in range(len(alphas)):
                label = r'$\alpha$ = {}'.format(alphas[j])

                # Generate x-axis ticks
                #x_ticks = [r'$\beta_{%d}$'%tick for tick in range(num_coeff)]
                r2 = plot_data[j,:,0]
                ci_r2 = plot_data[j,:,1]
                ax[j].errorbar(
                    np.arange(len(degree)),
                    y=r2, 
                    yerr=ci_r2,
                    label=label,
                    fmt='o',
                    markersize=4,
                    linewidth=1,
                    capsize=5,
                    capthick=1,
                    ecolor="black",
                    )
                ax[j].set_xticks(np.arange(len(degree)))
                ax[j].set_xticklabels(np.arange(1, len(degree)+1))
                ax[j].legend()

            fig.suptitle(r'Lasso R2 Score as a function of model complexity deg(P) with added noise = {}'.format(noise_levels[noise_idx]))
            fig.text(0.06, 0.5, 'R2 Score', ha='center', va='center', rotation='vertical')
            plt.xlabel("Degree of polynomial")
        
            #plt.show()
            figname = "coeffs_vs_complexity_noise_{}_lmd_{}_noise_{}.pdf".format(task, alphas[j], noise_levels[noise_idx])
            plt.savefig(FIGURE_PATH+figname, format="pdf")

def get_best_scores(task, degree):
    """ Get the best r2 and mse scores for a regression method.
    For all levels of noise.
    """

    DATA_PATH = "regression_data/"
    FIGURE_PATH = "../report/figures/"
    # Regression file contains: noise_level, error, bias, variance, coeffs, var_coeff
    # Load all the datafiles into a dict
    all_data = {}
    for i in range(len(degree)):
        fname = "{}_bootstrap_d{}.npy".format(task, degree[i])
        data = np.load(DATA_PATH+fname)
        all_data[degree[i]] = data

    if task == "ols":
        # shape is (num_degrees, num_noise_levels, mse and r2)
        noise_levels = np.unique(all_data[degree[0]][:,0])
        data = np.zeros((len(degree), len(noise_levels), 2))
        for i in range(len(degree)):
            for j in range(len(noise_levels)):
                data[i, j, 0] = all_data[degree[i]][j,2]
                data[i, j, 1] = all_data[degree[i]][j,4]

        best_mse = np.array([100, 100, 100])
        best_r2 = np.array([0, 100, 100])
                
        for i in range(len(degree)):
            for j in range(len(noise_levels)):
                if data[i,j,0] < best_mse[0]:
                    best_mse = [data[i, j, 0], degree[i], noise_levels[j]]
                if data[i,j,1] > best_r2[0]:
                    best_r2 = [data[i, j, 1], degree[i], noise_levels[j]]
        print("OLS: [best_mse, deg(P), noise_level]")
        print("Best MSE for OLS: ", best_mse)
        print("Best R2 for OLS: ", best_r2)

    if task == "ridge":
        # shape is (num_degrees, num_noise_levels, mse and r2)
        lmbdas = np.unique(all_data[degree[0]][:,0])

        # Hardcoding the noise levels for ridge and lasso
        noise_levels = np.array([0, 0.1, 0.2, 0.5, 1.0])
        data = np.zeros((len(degree), len(lmbdas), len(noise_levels), 2))
        for i in range(len(degree)):
            for j in range(len(lmbdas)):
                lmbda_idx = np.where(all_data[degree[i]][:,0] == lmbdas[j])[0]
                for k in range(len(noise_levels)):
                    data[i, j, k, 0] = all_data[degree[i]][lmbda_idx][k, 2]
                    data[i, j, k, 1] = all_data[degree[i]][lmbda_idx][k, 4]

        best_mse = np.array([100, 100, 100, 100])
        best_r2 = np.array([0, 100, 100, 100])
                
        for i in range(len(degree)):
            for j in range(len(noise_levels)):
                for k in range(len(noise_levels)):
                    if data[i,j,k,0] < best_mse[0]:
                        best_mse = [data[i, j, k, 0], degree[i], lmbdas[j], noise_levels[k]]
                    if data[i,j,k,1] > best_r2[0]:
                        best_r2 = [data[i, j, k, 1], degree[i], lmbdas[j], noise_levels[k]]

        print("Ridge: [best_mse, deg(P), lambda, noise_level]")
        print("Best MSE for Ridge: ", best_mse)
        print("Best R2 for Ridge: ", best_r2)

    if task == "lasso":
        # shape is (num_degrees, num_noise_levels, mse and r2)
        alphas = np.unique(all_data[degree[0]][:,0])

        # Hardcoding the noise levels for ridge and lasso
        noise_levels = np.array([0, 0.1, 0.2, 0.5, 1.0])
        data = np.zeros((len(degree), len(alphas), len(noise_levels), 2))
        for i in range(len(degree)):
            for j in range(len(alphas)):
                alpha_idx = np.where(all_data[degree[i]][:,0] == alphas[j])[0]
                for k in range(len(noise_levels)):
                    data[i, j, k, 0] = all_data[degree[i]][alpha_idx][k, 2]
                    data[i, j, k, 1] = all_data[degree[i]][alpha_idx][k, 4]

        best_mse = np.array([100, 100, 100, 100])
        best_r2 = np.array([0, 100, 100, 100])
                
        for i in range(len(degree)):
            for j in range(len(alphas)):
                for k in range(len(noise_levels)):
                    if data[i,j,k,0] < best_mse[0]:
                        best_mse = [data[i, j, k, 0], degree[i], alphas[j], noise_levels[k]]
                    if data[i,j,k,1] > best_r2[0]:
                        best_r2 = [data[i, j, k, 1], degree[i], alphas[j], noise_levels[k]]

        print("Ridge: [best_mse, deg(P), alpha, noise_level]")
        print("Best MSE for Lasso: ", best_mse)
        print("Best R2 for Lasso: ", best_r2)


# Output plots for each subtask in the project
if len(sys.argv) > 0:
    for arg in sys.argv[1:]:
        if arg == "ols":
            plot_r2(arg, [1,2,3,4,5])
            plot_mse(arg, [1,2,3,4,5])
            plot_errors(arg, [1,2,3,4,5])
            plot_coeffs_nonoise(arg, [1,2,3,4,5])
            plot_coeffs_noise(arg, 5)

        # Noise column is substituted for lambda and alpha for ridge and lasso
        # For each value of lmb/alpha there are four rows of data, corresponding to noise level
        if arg == "ridge":
            plot_r2(arg, [1,2,3,4,5])
            plot_mse(arg, [1,2,3,4,5])
            plot_coeffs_nonoise(arg, [1,2,3,4,5])
            plot_coeffs_noise(arg, [1,2,3,4,5])
        
        if arg == "lasso":
            plot_r2(arg, [1,2,3,4,5])
            plot_mse(arg, [1,2,3,4,5])
            plot_coeffs_nonoise(arg, [1,2,3,4,5])
            plot_coeffs_noise(arg, [1,2,3,4,5])

        if arg == "scores":
            get_best_scores("ols", [1,2,3,4,5])
            get_best_scores("ridge", [1,2,3,4,5])
            get_best_scores("lasso", [1,2,3,4,5])
