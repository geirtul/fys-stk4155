import numpy as np
from numpy.random import randint, randn
from time import time

def bootstrap(data, num_draws):
    """
    Perform bootstrapping on a dataset.
    """
    t = np.zeros(num_draws)
    N = len(data)
    t0 = time()


    # Non-parametric bootstrap
    for i in range(num_draws):
        t[i] = np.mean(data[randint(0,N,N)])
    t1 = time()
    # Analysis

    print("Runtime: {:f} sec".format(t1-t0))
    print("Bootstrap statistics:")
    print("{:^8s} | {:^8s} | {:^8s} | {:^8s}".format("original", "bias", "mean", "std.err"))
    print("{:8f} | {:8f} | {:8f} | {:8f}".format( np.mean(data),
                                            np.std(data),
                                            np.mean(t),
                                            np.std(t)))

    return t

if __name__ == "__main__":
    mu, sigma = 10, 2
    datapoints = 10000
    x = mu + sigma*randn(datapoints)
    t = bootstrap(x, datapoints)
