import numpy as np
import warnings
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from logistic import LogisticRegression

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

# Set up datasets before regression begins.
# ==================================================
# Data files contains 16*10000 samples taken in 
# T = np.arange(0.25,4.0001,0.25)
# Pickle imports the data as a 1D array, compressed bits.
# Decompress array and reshape for convenience
# Also map 0 state to -1 (Ising variable can take values +/-1)

path_to_data = 'data/IsingData/'
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
train_size = 0.8  # training samples
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=train_size)

limit = int(len(X_train)*0.2)

# Full data set
# X = np.concatenate((X_critical,X))
# Y = np.concatenate((Y_critical,Y))
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print()
print(X_train.shape[0], 'train samples')
print(X_critical.shape[0], 'critical samples')
print(X_test.shape[0], 'test samples')

# ==================================================
# Regression
# ==================================================
# We apply logistic regression to the states to optimize
# a set of weights that minimize the cross-entropy to produce
# the best model for the data.
print('\nPerforming logistic regression:')
print('======================================================================')
etas = np.logspace(-3, -2, 2)
intercept = True
epochs = 1e1
batch_size = 100
accuracies = []
costs = []
for eta in etas:
    logistic = LogisticRegression(eta, intercept, epochs, batch_size)
    logistic.fit(X_train[:limit, :], Y_train[:limit])
    accuracy_train = logistic.accuracy(X_train, Y_train)
    accuracy_test = logistic.accuracy(X_test, Y_test)
    accuracy_critical = logistic.accuracy(X_critical, Y_critical)
    accuracies.append(logistic.accuracies)
    costs.append(logistic.cost_function)
    print("Eta = {}".format(eta))
    print('Accuracy train = {}'.format(accuracy_train))
    print('Accuracy test = {}'.format(accuracy_test))
    print('Accuracy critical = {}'.format(accuracy_critical))

# Plot cost function changes in each epoch
xvals = range(len(costs[0][0]))
for i, cost in enumerate(costs[0]):
    plt.subplot(5, 2, i+1)
    plt.plot(xvals, cost, label='epoch '+str(i))
    plt.title(r'$\eta$ = 0.001')
    plt.xlabel('iteration')
    plt.ylabel('cost function')
    plt.legend()
plt.show()

for i, cost in enumerate(costs[1]):
    plt.subplot(5, 2, i+1)
    plt.plot(xvals, cost, label='epoch '+str(i))
    plt.title(r'$\eta$ = 0.01')
    plt.xlabel('iteration')
    plt.ylabel('cost function')
    plt.legend()
plt.show()

for acc, eta in zip(accuracies, etas):
    plt.plot(range(len(acc)), acc, label=r'$\eta$ = '+str(eta))

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()