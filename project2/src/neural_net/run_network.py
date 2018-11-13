import numpy as np
from mlp import MLP
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
        X, Y, train_size=train_size)

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


def accuracy_score(test_targets, prediction):
    return np.sum(test_targets == prediction) / len(test_targets)


# n_hidden = 50
# n_classes = 2
# epochs = 100
# batch_size = 100
# eta = 0.001
# lmd = 1.0

# Try networks with different number of hidden nodes:
#nodes, eta, lmd, acc,
# 20 0.0001 1.0 0.8312307692307692



# Grid search for optimal parameters
# nodes = [5, 10, 15, 20, 30, 40, 50, 100, 150, 200]
# times = [3.9, 4.06, 4.18, 4.15, 4.55, 4.53, 4.70, 6.22, 7.30, 8.26]
# plt.plot(nodes, times)
# plt.show()
# exit(1)

n_hidden = [10, 20, 50]
n_classes = 2
epochs = 20
batch_size = 100
etas = np.logspace(-5, 1, 7)
lmds = np.logspace(-5, 1, 7)

limit = int(len(X_train)*0.2)
net_accuracies = []

for hidden in n_hidden:
    net = MLP(X_train, Y_train, X_test, Y_test, hidden, n_classes, epochs, batch_size, etas[1], 0.0)
    net.train()
    #prob = net.predict(X_test)
    #acc = accuracy_score(Y_test, prob)
    #print("\neta, lmd, acc,\n{}, {}, {}".format(etas[1], lmds[2], acc))
    net_accuracies.append(net.accuracies_test)

for acc, n in zip(net_accuracies, n_hidden):
    plt.plot(range(epochs), acc, label=str(n)+" nodes")
plt.xlabel("Epochs")
plt.ylabel("Accuracy score")
plt.title(r"$\eta$ = {}, $\lambda$ = {}".format(etas[1], 0.0))
plt.legend()
plt.show()


# stored_models = np.zeros((len(etas), len(lmds)), dtype=object)
#
# for i, eta in enumerate(etas):
#     for j, lmd in enumerate(lmds):
#         net = MLP(X_train[:limit, :], Y_train[:limit], X_test, Y_test, n_hidden, n_classes, epochs, batch_size, eta, lmd)
#         net.train()
#         stored_models[i, j] = net
#         test_prob = net.predict(X_test)
#         acc = accuracy_score(Y_test, test_prob)
#         #print("Eta: {} | Lambda: {} | Accuracy: {}".format(eta, lmd, acc))
#
# # Plotting the  grid search.
#
# train_accuracy = np.zeros((len(etas), len(lmds)))
# test_accuracy = np.zeros((len(etas), len(lmds)))
#
# for i in range(len(etas)):
#     for j in range(len(lmds)):
#         net = stored_models[i][j]
#
#         train_pred = net.predict(X_train)
#         test_pred = net.predict(X_test)
#
#         train_accuracy[i][j] = accuracy_score(Y_train, train_pred)
#         test_accuracy[i][j] = accuracy_score(Y_test, test_pred)
#
# fig, ax = plt.subplots(figsize=(10, 10))
# sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
# ax.set_title("Training Accuracy")
# ax.set_ylabel("$\eta$")
# ax.set_xlabel("$\lambda$")
# plt.show()
#
# fig, ax = plt.subplots(figsize=(10, 10))
# sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
# ax.set_title("Test Accuracy")
# ax.set_ylabel("$\eta$")
# ax.set_xlabel("$\lambda$")
# plt.show()