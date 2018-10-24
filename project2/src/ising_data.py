import numpy as np


def ising_energies(states, L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    Ising model params example:

    np.random.seed(12)
    L = 40
    num_states = 1000
    states = np.random.choice([-1, 1], size=(num_states, L))

    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)

    return E


def recast_to_regression(states):
    """
    Recast a set of Ising states to a form to which linear regression can be applied.
    See notebook 'NB_CVI-linreg_ising.ipynb' by Mehta et. al.

    :param states: The random states as initialized by np.random.choice
    :return: recast states
    """
    states_recast = np.einsum('...i,...j->...ij', states, states)
    shape = states_recast.shape
    states_recast = states_recast.reshape(shape[0], shape[1]*shape[2])
    return states_recast
