import numpy as np
from time import time
from tqdm import tqdm
import sys
from neural_net import NeuralNet
import warnings
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import seaborn as sns
import sys
sns.set()

# Comment this to turn on warnings
warnings.filterwarnings('ignore')

# Shuffle random seed generator
np.random.seed()
# ============================================================================
# Ising model and data import
# ============================================================================

# Ising model parameters
# ======================================
L = 40  # linear system size
J = -1.0  # Ising interaction
T = np.linspace(0.25, 4.0, 16)  # set of temperatures
T_c = 2.26  # Onsager critical temperature in the TD limit
# system size

# Set up datasets before running the network.
# ======================================
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

# Shape the targets to (n_samples, n_outputs)
n_outputs = 1
Y_train = Y_train.reshape((len(Y_train), n_outputs))
Y_test = Y_test.reshape((len(Y_test), n_outputs))
Y_critical = Y_critical.reshape((len(Y_critical), n_outputs))

# Full data set
# X = np.concatenate((X_critical,X))
# Y = np.concatenate((Y_critical,Y))

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)
print()
print(X_train.shape[0], 'train samples')
print(X_critical.shape[0], 'critical samples')
print(X_test.shape[0], 'test samples')

# ============================================================================
# Running the network
# ============================================================================
# For faster testing runs, restrict training sets like X_train[:limit, :]


if sys.argv[1] == "regular":
    # Parameters
    limit = int(len(X_train) * 0.1)
    n_layers = 2
    n_nodes = (100, 50)
    n_classes = 1
    epochs = 3
    batch_size = 100
    #eta = 1e-2
    #lmda = 1e-4
    etas = [1e-3, 1e-2]   # Optimal params based on grid search
    lmdas = [1e-5, 1e-4]  # ^

    # # Run one net and print accuracies
    # net = NeuralNet(X_train, Y_train, X_test, Y_test,
    #                 n_layers, n_nodes, n_classes, epochs, eta, batch_size, lmda)
    # net.train()
    # print("Accuracy on training set = ", net.accuracy(X_train, Y_train))
    # print("Accuracy on test set = ", net.accuracy(X_test, Y_test))
    # print("Accuracy on critical set = ", net.accuracy(X_critical, Y_critical))


    # Test accuracy for multiple paramteres.
    for eta in etas:
        for lmda in lmdas:
            print("Running network with eta = {} and lambda = {}".format(eta, lmda))
            net = NeuralNet(X_train, Y_train, X_test, Y_test,
                            n_layers, n_nodes, n_classes, epochs, 
                            eta, batch_size, lmda)
            net.train()
            acc_train = net.accuracy(X_train, Y_train)
            acc_test = net.accuracy(X_test, Y_test)
            acc_crit = net.accuracy(X_critical, Y_critical)
            print("Accuracy on training set = ", acc_train )
            print("Accuracy on test set = ", acc_test)
            print("Accuracy on critical set = ", acc_crit)

# Grid search for optimal parameters
# =====================================
if sys.argv[1] == "gridsearch":
    # Parameters
    limit = int(len(X_train) * 0.1)
    n_layers = 2
    n_nodes = (100, 50)
    n_classes = 1
    epochs = 10
    batch_size = 100
    etas = np.logspace(-5, 1, 7)
    lmdas = np.logspace(-5, 1, 7)

    stored_models = np.zeros((len(etas), len(lmdas)), dtype=object)

    # Train and store all the models
    for i, eta in enumerate(etas):
        print("Eta {} = {}".format(i, etas[i]))
        for j, lmda in tqdm(enumerate(lmdas)):
            net = NeuralNet(X_train[:limit, :], Y_train[:limit], X_test, Y_test,
                            n_layers, n_nodes, n_classes, epochs,
                            eta, batch_size, lmda)
            net.train()
            stored_models[i, j] = net


    # Plotting the  grid search.
    print("Checking accuracy and plotting-..")
    train_accuracy = np.zeros((len(etas), len(lmdas)))
    test_accuracy = np.zeros((len(etas), len(lmdas)))
    critical_accuracy = np.zeros((len(etas), len(lmdas)))

    for i in range(len(etas)):
        for j in range(len(lmdas)):
            net = stored_models[i][j]

            train_accuracy[i][j] = net.accuracy(X_train[:limit, :], Y_train[:limit])
            test_accuracy[i][j] = net.accuracy(X_test, Y_test)
            critical_accuracy[i][j] = net.accuracy(X_critical, Y_critical)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    ax.set_xticklabels(lmdas)
    ax.set_yticklabels(etas)
    figname = "net_acc_train_{}.pdf".format(n_layers)
    plt.savefig(figname, format="pdf", pad_inches=0.0)
    #plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    ax.set_xticklabels(lmdas)
    ax.set_yticklabels(etas)
    figname = "net_acc_test_{}.pdf".format(n_layers)
    plt.savefig(figname, format="pdf", pad_inches=0.0)
    #plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(critical_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Critical Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    ax.set_xticklabels(lmdas)
    ax.set_yticklabels(etas)
    figname = "net_acc_crit_{}.pdf".format(n_layers)
    plt.savefig(figname, format="pdf", pad_inches=0.0)
    #plt.show()
