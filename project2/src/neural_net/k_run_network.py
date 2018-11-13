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
L = 40  # linear system size
J = -1.0  # Ising interaction
T = np.linspace(0.25, 4.0, 16)  # set of temperatures
T_c = 2.26  # Onsager critical temperature in the TD limit
# system size

# define ML parameters
num_classes = 2

# Set up datasets before running the network.
# ==================================================
# Data files contains 16*10000 samples taken in
# T = np.arange(0.25,4.0001,0.25)
# Pickle imports the data as a 1D array, compressed bits.
# Decompress array and reshape for convenience
# Also map 0 state to -1 (Ising variable can take values +/-1)

path_to_data = '../data/IsingData/'
file_name_data = "Ising2DFM_reSample_L40_T=All.pkl"
file_name_labels = "Ising2DFM_reSample_L40_T=All_labels.pkl"

data = pickle.load(open(path_to_data + file_name_data, 'rb'))
data = np.unpackbits(data).reshape(-1, 1600)
data = data.astype('int')
data[np.where(data == 0)] = -1
labels = pickle.load(open(path_to_data + file_name_labels, 'rb'))

# Divide data into ordered, critical and disordered
X_ordered = data[:70000, :]
Y_ordered = labels[:70000]

X_critical = data[70000:100000, :]
Y_critical = labels[70000:100000]

X_disordered = data[100000:, :]
Y_disordered = labels[100000:]

# We want to pick samples from ordered or disordered, so we
# combine these.
X = np.concatenate((X_ordered, X_disordered))
Y = np.concatenate((Y_ordered, Y_disordered))

# Pick random data points from ordered and disordered states
# to create the training and test sets
train_to_test_ratio = 0.95  # training samples
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=train_to_test_ratio)

# Full data set
# X = np.concatenate((X_critical,X))
# Y = np.concatenate((Y_critical,Y))
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print()
print(X_train.shape[0], 'train samples')
print(X_critical.shape[0], 'critical samples')
print(X_test.shape[0], 'test samples')

# =========== Divide data in k segments.============
# TODO: This needs to be generalized and made into a function,
# TODO: Also needs to be changed to fit the new variable names.
k = 10

# Determine size of intervals, reusing data rather than discarding
nodes_per_segment = int(np.ceil(np.shape(movements)[0] / k))
training_segments = []
target_segments = []
for i in range(k):

    # Set up segments as long as (i+1)*nodes_per_segment
    # does not cause indexerror
    if (i + 1) * nodes_per_segment < movements.shape[0]:
        training_segments.append(
            movements[i * nodes_per_segment:(i + 1) * nodes_per_segment, 0:40])
        target_segments.append(
            target[i * nodes_per_segment:(i + 1) * nodes_per_segment])
    else:
        # Handle index error, reiterating some of the first input nodes.
        # This should probably pick random input nodes to fill the last segment.
        repeat_index = (i + 1) * nodes_per_segment - movements.shape[0]
        training_segments.append(np.array(
            list(movements[i * nodes_per_segment:, 0:40]) + list(
                movements[:repeat_index, 0:40])))
        target_segments.append(np.array(
            list(target[i * nodes_per_segment:]) + list(target[:repeat_index])))

# Now we can iterate over indices in training_segments list
# to perform the training.
correctness = []

# What happens in this for-loop is not elegant, but it works.
# Try-except blocks handle the cases where the indexing overshoots
# the length of the arrays.
for i in range(len(training_segments)):
    if i == 0:
        train = np.concatenate(training_segments[i:i + k - 3])
        train_targets = np.concatenate(target_segments[i:i + k - 3])
        valid = training_segments[i + k - 2]
        valid_targets = target_segments[i + k - 2]
        test = training_segments[i + k - 1]
        test_targets = target_segments[i + k - 1]
    else:
        train1 = np.concatenate(training_segments[i:])
        stop = len(training_segments) - (i + k - 3)
        try:
            train2 = np.concatenate(training_segments[:stop])
        except ValueError:
            train2 = training_segments[stop]

        train = np.concatenate((train1, train2))
        train_targets1 = np.concatenate(target_segments[i:])
        try:
            train_targets2 = np.concatenate(target_segments[:stop])
        except ValueError:
            train_targets2 = target_segments[stop]

        train_targets = np.concatenate((train_targets1, train_targets2))

        valid = training_segments[stop + 1]
        valid_targets = target_segments[stop + 1]

        test = training_segments[stop + 2]
        test_targets = target_segments[stop + 2]

    # Check that shapes are correct
    if train.shape[0] != train_targets.shape[0]:
        print("Shapes not equal in iter", i, ", train: ", train.shape,
              " targets: ", train_targets.shape)
        exit(1)

    # Set up networks and run training ++
    net = mlp.mlp(8)
    net.train(train, train_targets, valid, valid_targets)
    percent, conf = net.confusion(test, test_targets, out=False)
    correctness.append(percent)

    # === Uncomment for plotting errors ===
    # plt.plot(range(len(net.error_squared)), net.error_squared)
    # plt.xlabel("Epochs")
    # plt.ylabel("Error squared")
# plt.show()

print("Percentage correct for each fold:")
for percent in correctness:
    print(percent)
print("Average percentage correct= ", np.mean(np.array(correctness)))
print("Standard deviation = ", np.std(np.array(correctness)))
