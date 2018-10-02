import numpy as np
from time import time

# Bootstrap algorithm
def bootstrap(data, num_draws):
    """
    Resample the data using the bootstrap algorithm.
    Based on the example found in lecture notes.

    param: data         - the data to perform resampling on
    param: num_draws    - how many times to draw sample datasets

    """
    drawn_samples = []
    N = len(data)
    time_start = time()

    # non-parametric bootstrap
    for i in range(num_draws):
        # Sample datasets are drawn by generating N sets of N indices.
        # Data on those indices are then extracted using data[indices]
        # because arrays are amazing that way.

        indices = np.random.randint(0,N,N)
        drawn_samples.append(np.array(data[indices]))

    # analysis
    print("Runtime: %g sec" % (time()-time_start));
    print("Bootstrap Statistics :")
    print("mean(data) | std(data)'bias' | mean(mean_values) | std(mean_values)")
    print("%8g | %8g | %14g | %15g" % (np.mean(data),
                                np.std(data),
                                np.mean(drawn_samples),
                                np.std(drawn_samples)))
    return drawn_samples

data = np.array([1,2,3,4,5,6,7,8,9,10])
bootstrap(data, 100)
