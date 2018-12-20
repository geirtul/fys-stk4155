import numpy as np
import matplotlib.pyplot as plt

def cumulative_gain_chart(targets, probabilities,filename, desired_class = 1):
    """
    Plot the cumulative gain chart for predicted probabilites for a binary
    classifier.

    :param targets: targets for the inputs for prediction
    :param probabilities: predicted probabilites from inputs
    :param filename: output filename for plot
    :param desired_class: which class to produce the plot for.
                          default = 1 assumed response is the desired class.
    """
    # Start by stacking the targets and probabilities
    vals = np.stack((targets, probabilities[:,0]), axis=-1)
    for i in range(1, probabilities.shape[1]):
        vals = np.c_[vals, probabilities[:,i]]

    # Then sort the values such that the probabilities of desired_class
    # are in descending order. desired_class+1 since col 0 is targets.
    vals_sorted = vals[vals[:,desired_class+1].argsort()[::-1]]
    # Cumulative sum arrays
    cumulative_sums = np.cumsum(vals_sorted[:,0])
    events = np.zeros(vals_sorted.shape[0])

    # Get the amount of targets that are in the desired class, and make
    # array for ideal case (cumulative_ideal)
    n_defaults = len(targets[np.where(targets==desired_class)])
    events[:n_defaults] += 1
    cumulative_ideal = np.cumsum(events)

    # Randomized case for comparison
    cumulative_random = np.cumsum(np.random.choice(targets,
                                                   size=len(targets),
                                                   replace=False))

    # Plotting
    plt.plot(range(len(vals_sorted)), cumulative_sums, label='Model')
    plt.plot(range(len(vals_sorted)), cumulative_ideal, label='Ideal classifier')
    plt.plot(range(len(vals_sorted)), cumulative_random, label='Random')
    plt.legend()
    plt.xlabel("Total samples")
    plt.ylabel("Cumulative sum of responses")
    plt.tight_layout()
    plt.savefig(filename + "_cumul.pdf", format="pdf")
