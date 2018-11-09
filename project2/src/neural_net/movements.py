# This file will read in data and start the mlp network.

#import tabulate # latex tables
#from bmatrix import bmatrix #latex matrices
import numpy as np
import mlp
import matplotlib.pyplot as plt

filename = 'data/movements_day1-3.dat'

movements = np.loadtxt(filename,delimiter='\t')

# Subtract arithmetic mean for each sensor. We only care about how it varies:
movements[:,:40] = movements[:,:40] - movements[:,:40].mean(axis=0)

# Find maximum absolute value:
imax = np.concatenate(  ( movements.max(axis=0) * np.ones((1,41)) ,
                          np.abs( movements.min(axis=0) * np.ones((1,41)) ) ),
                          axis=0 ).max(axis=0)

# Divide by imax, values should now be between -1,1
movements[:,:40] = movements[:,:40]/imax[:40]

# Generate target vectors for all inputs 2 -> [0,1,0,0,0,0,0,0]
target = np.zeros((np.shape(movements)[0],8));
for x in range(1,9):
    indices = np.where(movements[:,40]==x)
    target[indices,x-1] = 1

# Randomly order the data
order = list(range(np.shape(movements)[0]))
np.random.shuffle(order)
movements = movements[order,:]
target = target[order,:]

# Split data into 3 sets

# Training updates the weights of the network and thus improves the network
train = movements[::2,0:40]
train_targets = target[::2]

# Validation checks how well the network is performing and when to stop
valid = movements[1::4,0:40]
valid_targets = target[1::4]

# Test data is used to evaluate how good the completely trained network is.
test = movements[3::4,0:40]
test_targets = target[3::4]

# Try networks with different number of hidden nodes:
hiddens = [6,8,12]
iterations = 100
for hidden in hiddens:

    # Initialize the network:
    net = mlp.mlp(train, train_targets, hidden)

    # Run training:
    net.train(train,train_targets, valid, valid_targets)
    # NOTE: You can also call train method from here,
    #       and make train use earlystopping method.
    #       This is a matter of preference.

    # Check how well the network performed:
    percentage, conf = net.confusion(test,test_targets)

    # Plotting and tex
    plt.plot(range(len(net.error_squared)), net.error_squared)
    plt.plot(range(len(net.validation_error)), net.validation_error)
    plt.xlabel("Epochs")
    plt.ylabel("Error squared")

    #print(bmatrix(conf) + "\n")




#plt.show()
