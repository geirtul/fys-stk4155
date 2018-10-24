import numpy as np
import warnings
from ising_data import ising_energies, recast_to_regression

"""
This is the file that will run the numerical experiments.
Ising data setup from Mehta et. al. notebook on Ising linear regression.
"""

# Comment this to turn on warnings
warnings.filterwarnings('ignore')

# Define Ising model params
# system size

np.random.seed(12)
L = 40
num_states = 1000

# create N random Ising states, calculate energies and recast for regression.
states = np.random.choice([-1, 1], size=(num_states, L))
energies = ising_energies(states, L)
states = recast_to_regression(states)
data = [states, energies]

