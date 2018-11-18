import numpy as np
from time import time
from neural_net import NeuralNet
import warnings
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

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

data = pickle.load(open(path_to_data+file_name_data, 'rb'))
data = np.unpackbits(data).reshape(-1, 1600)
data = data.astype('int')
data[np.where(data == 0)] = -1
labels = pickle.load(open(path_to_data+file_name_labels, 'rb'))

# Divide data into ordered, critical and disordered
X_ordered=data[:70000, :]
Y_ordered=labels[:70000]

X_critical=data[70000:100000, :]
Y_critical=labels[70000:100000]

X_disordered=data[100000:, :]
Y_disordered=labels[100000:]

# We want to pick samples from ordered or disordered, so we
# combine these.
X = np.concatenate((X_ordered, X_disordered))
Y = np.concatenate((Y_ordered, Y_disordered))

# Pick random data points from ordered and disordered states
# to create the training and test sets
train_size = 0.8  # training samples
X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=train_size, shuffle=True)

# Full data set
# X = np.concatenate((X_critical,X))
# Y = np.concatenate((Y_critical,Y))
# print('X_train shape:', X_train.shape)
# print('Y_train shape:', Y_train.shape)
# print('X_test shape:', X_test.shape)
# print('Y_test shape:', Y_test.shape)
# print()
# print(X_train.shape[0], 'train samples')
# print(X_critical.shape[0], 'critical samples')
# print(X_test.shape[0], 'test samples')

limit = int(len(X_train)*0.2)
#n_hidden1 = 50
#n_hidden2 = 50
#epochs = 5
#batch_size = 100
##etas = np.logspace(-4, 0, 5)
#eta = 0.01
#lmda = 0.01
#accuracies = []
#
#
#net = NeuralNet(X_train[:limit, :], Y_train[:limit], X_test, Y_test,
#                n_hidden1, n_hidden2, epochs, eta, batch_size, lmda)
#net.train()
#print("Accuracy on critical set = ", net.accuracy(X_critical, Y_critical))
#plotlabel = r'$\eta$ = {}'.format(eta)
#plt.plot(range(epochs), net.accuracies, 'x-', label=plotlabel)

# for eta in etas:
#    print("Running network with eta = {}".format(eta))
#    net = NeuralNet(X_train, Y_train, X_test, Y_test,
#                    n_hidden1, n_hidden2, epochs, eta, batch_size)
#    net.train()
#    accuracies.append([net.accuracies[-1],
#                           net.accuracy(X_test, Y_test),
#                           net.accuracy(X_critical, Y_critical)])
#    plotlabel = r'$\eta$ = {}'.format(eta)
#    plt.plot(range(epochs), net.accuracies, 'x-', label=plotlabel)

#plt.title("Test set accuracy")
#plt.xlabel("Epochs")
#plt.ylabel("Accuracy score")
#plt.legend()
#plt.show()

# Grid search for optimal parameters

n_hidden1 = 50
n_hidden2 = 50
epochs = 10
batch_size = 100
etas = np.logspace(-5, 0, 6)
lmdas = np.logspace(-5, 0, 6)

stored_models = np.zeros((len(etas), len(lmdas)), dtype=object)

n_eta = len(etas)
n_lmda = len(lmdas)
n_iters = n_eta*n_lmda
for i, eta in enumerate(etas):
    for j, lmda in enumerate(lmdas):
        if j == 0:
            t0 = time()
        net = NeuralNet(X_train, Y_train, X_test, Y_test,
                  n_hidden1, n_hidden2, epochs, eta, batch_size, lmda)
        net.train()
        stored_models[i, j] = net
        if j == 0:
            t1 = time()
            print("Time for one iteration = {}s".format(t1-t0))
            print("Estimated total time = {}m".format((t1-t0)*n_iters/60))
        #print("Eta: {} | Lambda: {} | Accuracy: {}".format(eta, lmd, acc))

# Plotting the  grid search.

train_accuracy = np.zeros((len(etas), len(lmdas)))
test_accuracy = np.zeros((len(etas), len(lmdas)))

for i in range(len(etas)):
    for j in range(len(lmdas)):
        net = stored_models[i][j]

        train_accuracy[i][j] = net.accuracies[-1]
        test_accuracy[i][j] = net.accuracy(X_test, Y_test)

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()
