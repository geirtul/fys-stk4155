import numpy as np


class mlp:
    def __init__(self):
        """
        Initialize the network. This MLP network has one hidden layer.
        """
        self.beta = 1
        self.eta = 0.1
        self.momentum = 0.0
        self.weights1 = None
        self.weights2 = None

        # errors
        self.accuracy_vals = []
        self.validation_accuracy = []

    def early_stopping(self, valid, validtargets, counter):
        """
        Stops the training phase of the network if two squared-error points
        are too close in value. The nature of the test is due to the error
        being seemingly strictly decreasing, but reaching plateaus now and then.
        We want to let the algorithm past the first plateau, but not necessarily
        the next.

        :param valid: validation set inputs
        :param validtargets: validation set outputs
        :param counter: counts number of epochs
        :return: Boolean, defines if training should stop.
        """

        eps = 1e-5  # Threshold value for squared error
        # Run validation set forward
        valid_output = self.forward(valid)[1]

        # Compute error in validation
        new_validation_accuracy = self.accuracy(valid_output, validtargets)

        # new_validation_accuracy = np.mean(
        #    0.5 * np.sum(np.square(valid_output - validtargets), axis=1))
        self.validation_accuracy.append(new_validation_accuracy)

        # Check if error has stopped changing.
        if counter > 10:
            check = abs(self.validation_accuracy[counter]
                        - self.validation_accuracy[counter - 10])
            if check > eps:
                return False
            else:
                return True

    def train(self, inputs, targets, valid, validtargets, nhidden):
        """
        Trains the network. Runs the backwards phase of the training algorithm
        to adjust the weights.

        :param inputs: training inputs
        :param targets: training targets
        :param valid: validation inputs
        :param validtargets: validation targets
        :param nhidden: number of hidden nodes in the network
        :return:
        """
        # Initialize randomized weights, and activations
        self.weights1 = np.random.uniform(-1 / np.sqrt(inputs.shape[1]),
                                          1 / np.sqrt(inputs.shape[1]),
                                          (inputs.shape[1], nhidden))

        # Handle target arrays with shape (N, )
        if len(targets.shape) == 1:
            targets = targets.reshape(len(targets), 1)
            validtargets = validtargets.reshape(len(validtargets), 1)

        self.weights2 = np.random.uniform(-1 / np.sqrt(inputs.shape[1]),
                                          1 / np.sqrt(inputs.shape[1]),
                                          (nhidden, targets.shape[1]))

        # Add bias
        np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        # errors
        counter = 0
        while counter >= 0:
            act_hidden, act_output = self.forward(inputs)
            self.backwards(inputs, act_hidden, act_output, targets)

            # Run earlystopping, break loop if necessary.
            if self.early_stopping(valid, validtargets, counter):
                break
            counter += 1

    def forward(self, inputs):
        """
        Feed inputs forward through the network.
        :param inputs: Training inputs.
        :return: Output from the network.
        """

        def g_hidden(inputs, weights):
            """
            Get activations in hidden layers based on sigmoid function.

            :param inputs: inputs to the network
            :param weights: weights in the network
            :return: sigmoid-based activations
            """

            activations = np.dot(inputs, weights)
            return 1 / (1 + np.exp(-self.beta * activations))

        def g_output(inputs, weights):
            """
            Get activations based on linear function.
            :param inputs: inputs to the network
            :param weights: weights in the network
            :return: linear-based activations for the output
            """

            activations = np.dot(inputs, weights)
            return activations

        act_hidden = g_hidden(inputs, self.weights1)

        # Using g_hidden for output aswell because the linear function causes
        # overflow. Likely, a restriction on the linear function has been
        # forgotten.
        # TODO: Check the g_output function to see if it can be implemented.
        act_output = g_hidden(act_hidden, self.weights2)

        return act_hidden, act_output

    def backwards(self, inputs, act_hidden, act_output, targets):
        """
        Backwards propagation of error, with updating weights accordingly.
        From 4.2.1 Marsland.

        :param inputs: inputs to the network
        :param act_hidden: activations from the hidden layers
        :param act_output: activations in output
        :param targets: target values for the supervised learning
        :return:
        """

        delta_o = (targets - act_output) * act_output * (1.0 - act_output)
        delta_h = act_hidden * (1.0 - act_hidden) * (
            np.dot(delta_o, np.transpose(self.weights2)))

        # Update weights
        updatew1 = self.eta * (np.dot(inputs.T, delta_h[:, :]))
        updatew2 = self.eta * (np.dot(act_hidden.T, delta_o))

        self.weights1 += updatew1
        self.weights2 += updatew2

        # Check current accuracy
        current_accuracy = self.accuracy(act_output, targets)
        self.accuracy_vals.append(current_accuracy)

    def accuracy(self, output, targets):
        """
        Evaluate the accuracy of the model based on the number of correctly
        labeled classes divided by the number of classes in total.
        """
        correct = output - targets

        count = 0
        for el in correct:
            if el == 0:
                count += 1
        score = count / correct.size

        return score

    def confusion(self, inputs, targets, out=True):
        """
        Calculates and prints confusion matrix for the network,
        and includes the percentage of correct classifications.

        :param inputs: inputs to the network
        :param targets: target values for the supervised learning
        :param out: Boolean, print success rate or not.
        :return: percentage of correct classifications, confusion matrix
        """

        output = self.forward(inputs)[1]
        conf = np.outer(np.transpose(output), targets)
        correct = np.matrix.trace(conf)
        total = np.sum(conf)

        # Just prettier printing of confusion matrix
        np.set_printoptions(suppress=True, precision=0)
        percentage = 100 * correct / total

        # Only print confusion matrix and percentages if we want to.
        if out == True:
            print(conf)
            print("Percentage correct = {}%".format(percentage))
        return percentage, conf
