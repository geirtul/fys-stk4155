# This file will read in data and start the mlp network.

# Uncomment below to allow outputting confusion matrix as latex.
# import tabulate # latex tables
# from bmatrix import bmatrix #latex matrices

import numpy as np
import mlp
import warnings
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split

# Comment this to turn on warnings
warnings.filterwarnings('ignore')

# Shuffle random seed generator
np.random.seed()

# Ising model parameters
# ==================================================
L=40 # linear system size
J=-1.0 # Ising interaction
T=np.linspace(0.25,4.0,16) # set of temperatures
T_c=2.26 # Onsager critical temperature in the TD limit
# system size

# define ML parameters
num_classes=2


# Set up datasets before running the network.
# ==================================================
# Data files contains 16*10000 samples taken in
# T = np.arange(0.25,4.0001,0.25)
# Pickle imports the data as a 1D array, compressed bits.
# Decompress array and reshape for convenience
# Also map 0 state to -1 (Ising variable can take values +/-1)

print("Importing data...")

path_to_data = '../data/IsingData/'
file_name_data = "Ising2DFM_reSample_L40_T=All.pkl"
file_name_labels = "Ising2DFM_reSample_L40_T=All_labels.pkl"

data = pickle.load(open(path_to_data+file_name_data,'rb'))
data = np.unpackbits(data).reshape(-1, 1600)
data = data.astype('int')
data[np.where(data == 0)] = -1
labels = pickle.load(open(path_to_data+file_name_labels,'rb'))

# Divide data into ordered, critical and disordered
X_ordered=data[:70000, :]
Y_ordered=labels[:70000]

X_critical=data[70000:100000, :]
Y_critical=labels[70000:100000]

X_disordered=data[100000:, :]
Y_disordered=labels[100000:]

# We want to pick samples from ordered or disordered, so we
# combine these.
X=np.concatenate((X_ordered, X_disordered))
Y=np.concatenate((Y_ordered, Y_disordered))

# Pick random data points from ordered and disordered states
# to create the training and test sets
train_to_test_ratio=0.9 # training samples
X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=train_to_test_ratio)

# Full data set
# X = np.concatenate((X_critical,X))
# Y = np.concatenate((Y_critical,Y))
# print('X_train shape:', X_train.shape)
# print('Y_train shape:', Y_train.shape)
# print()
# print(X_train.shape[0], 'train samples')
# print(X_critical.shape[0], 'critical samples')
# print(X_test.shape[0], 'test samples')


# Try networks with different number of hidden nodes:
print("Starting network...")
num_hidden = [6, 8, 12]
iterations = 100
for hidden in num_hidden:
    # Initialize the network:
    net = mlp.mlp()

    # Run training:
    net.train(X_train, Y_train, X_test, Y_test, hidden)

    # Check how well the network performed:
    print("Validation accuracy = {}".format(net.validation_accuracy[-1]))
    #percentage, conf = net.confusion(X_test, Y_test)

    # Plotting and tex
    plt.plot(range(len(net.accuracy_vals)), net.accuracy_vals, label=str(hidden)+"train")
    plt.plot(range(len(net.validation_accuracy)), net.validation_accuracy, label=str(hidden)+"test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # print(bmatrix(conf) + "\n")

plt.show()
