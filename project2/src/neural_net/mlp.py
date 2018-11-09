import numpy as np


class mlp:
    def __init__(self, inputs, targets, nhidden):
        """
        Initialize the network.

        :param inputs:
        :param targets:
        :param nhidden: Number of hidden layers in the network.
        """
        self.beta = 1
        self.eta = 0.1
        self.momentum = 0.0

        # Initialize randomized weights, and activations
        self.weights1 = np.random.uniform(-1 / np.sqrt(inputs.shape[1]),
                                          1 / np.sqrt(inputs.shape[1]),
                                          (inputs.shape[1], nhidden))
        self.weights2 = np.random.uniform(-1 / np.sqrt(inputs.shape[1]),
                                          1 / np.sqrt(inputs.shape[1]),
                                          (nhidden, targets.shape[1]))

        # Add bias
        np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        # errors
        self.error_squared = []
        self.validation_error = []

    def earlystopping(self, valid, validtargets, counter):
        """
        Stops the training phase of the network if two squared-error points
        are too close in value. The nature of the test is due to the error
        being seemingly strictly decreasing, but reaching plateaus now and then.
        We want to let the algorithm past the first plateau, but not necessarily
        the next.

        :param valid:
        :param validtargets:
        :param counter:
        :return: Boolean, defines if training should stop.
        """

        eps = 1e-3  # Threshold value for squared error
        # Run validation set forward
        valid_output = self.forward(valid)[1]

        # Compute error in validation
        new_valid_error = np.mean(
            0.5 * np.sum(np.square(valid_output - validtargets), axis=1))
        self.validation_error.append(new_valid_error)

        # Check if error has stopped changing.
        if counter > 10:
            check = abs(self.validation_error[counter]
                        - self.validation_error[counter - 10])
            if check > eps:
                olderr = new_valid_error
                return False
            else:
                return True

    def train(self, inputs, targets, valid, validtargets):
        """
        Trains the network. Runs the backwards phase of the training algorithm
        to adjust the weights.

        :param inputs:
        :param targets:
        :param valid:
        :param validtargets:
        :return:
        """

        # errors
        counter = 0
        while counter >= 0:
            act_hidden, act_output = self.forward(inputs)
            self.backwards(inputs, act_hidden, act_output, targets)

            # Run earlystopping, break loop if necessary.
            if self.earlystopping(valid, validtargets, counter):
                break

            counter += 1

    def forward(self, inputs):
        """
        Feed inputs forward through the network.
        :param inputs: The inputs to feed forward.
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
            raise NotImplemented

        act_hidden = g_hidden(inputs, self.weights1)
        # Using g_hidden for output aswell because the linear function causes
        # overflow. Likely, a restriction on the linear function has been
        # forgotten.
        # TODO: Check the g_output function to see if it can be implemented.
        act_output = g_hidden(act_hidden, self.weights2)

        return act_output

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

        # updatew1 = np.zeros((np.shape(self.weights1)))
        # updatew2 = np.zeros((np.shape(self.weights2)))

        updatew1 = self.eta * (np.dot(inputs.T, delta_h[:, :]))
        updatew2 = self.eta * (np.dot(act_hidden.T, delta_o))
        self.weights1 += updatew1
        self.weights2 += updatew2
        self.error_squared.append(
            np.mean(0.5 * np.sum(np.square(act_output - targets), axis=1)))

    def confusion(self, inputs, targets, out=True):
        """
        Calculates and prints confusion matrix for the network,
        and includes the percentage of correct classifications.

        :param inputs: inputs to the network
        :param targets: target values for the supervised learning
        :param out: Boolean, print success rate or not.
        :return: percentage of correct classifications, confusion matrix
        """

        output = self.forward(inputs)
        conf = np.dot(np.transpose(output), targets)
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
