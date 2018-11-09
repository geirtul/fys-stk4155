#!/usr/bin/env Python3
'''
    This file will read in data and start your mlp network.
    You can leave this file mostly untouched and do your
    mlp implementation in mlp.py.
'''
# Feel free to use numpy in your MLP if you like to.
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

#=========== Divide data in k segments.===========0
k = 10

# Determine size of intervals, reusing data rather than discarding
nodes_per_segment = int(np.ceil(np.shape(movements)[0]/k))
training_segments = []
target_segments = []
for i in range(k):
    
    # Set up segments as long as (i+1)*nodes_per_segment does not cause indexerror
    if (i+1)*nodes_per_segment < movements.shape[0]:
        training_segments.append(movements[i*nodes_per_segment:(i+1)*nodes_per_segment,0:40])
        target_segments.append(target[i*nodes_per_segment:(i+1)*nodes_per_segment])
    else:
        # Handle index error, reiterating some of the first input nodes.
        # This should probably pick random input nodes to fill the last segment.
        repeat_index = (i+1)*nodes_per_segment - movements.shape[0]
        training_segments.append(np.array(list(movements[i*nodes_per_segment:,0:40])+list(movements[:repeat_index,0:40])))
        target_segments.append(np.array(list(target[i*nodes_per_segment:])+list(target[:repeat_index])))

# Now we can iterate over indices in training_segments list to perform the training.
correctness = []

# What happens in this for-loop is not elegant, but it works.
# Try-except blocks handle the cases where the indexing overshoots
# the length of the arrays.
for i in range(len(training_segments)):
    if i == 0:
        train = np.concatenate(training_segments[i:i+k-3])
        train_targets = np.concatenate(target_segments[i:i+k-3])
        valid = training_segments[i+k-2]
        valid_targets = target_segments[i+k-2]
        test = training_segments[i+k-1]
        test_targets = target_segments[i+k-1]
    else:
        train1 = np.concatenate(training_segments[i:])
        stop = len(training_segments)-(i+k-3)
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

        valid = training_segments[stop+1]
        valid_targets = target_segments[stop+1]

        test = training_segments[stop+2]
        test_targets = target_segments[stop+2]

    
    # Check that shapes are correct
    if train.shape[0] != train_targets.shape[0]:
        print("Shapes not equal in iter",i,", train: ", train.shape, " targets: ", train_targets.shape)
        exit(1)

    # Set up networks and run training ++
    net = mlp.mlp(train,train_targets, 8)
    net.train(train,train_targets,valid,valid_targets)
    percent, conf = net.confusion(test,test_targets, out=False)
    correctness.append(percent)
    
    # === Uncommend for plotting errors ===
    #plt.plot(range(len(net.error_squared)), net.error_squared)
    #plt.xlabel("Epochs")
    #plt.ylabel("Error squared")
#plt.show()

print("Percentage correct for each fold:")
for percent in correctness:
    print(percent)
print("Average percentage correct= ", np.mean(np.array(correctness)))
print("Standard deviation = ", np.std(np.array(correctness)))


