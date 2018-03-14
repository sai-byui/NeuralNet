from Neuron import Neuron
import random
from math import sqrt
import time


class Network:
    def __init__(self, topology, num_inputs):
        self.topology = topology  # For copying during the training process
        self.layers = []
        for index, layer in enumerate(topology):
            self.layers.append([])
            for neuron in range(layer):
                self.layers[index].append(Neuron())
        self.num_inputs = num_inputs
        self.__init_weights__()
        self.temperature = 1.0

    def __init_weights__(self):
        random.seed(time.time())
        for i in range(len(self.layers)):
            for j in self.layers[i]:
                if i == 0:
                    for net_input in range(self.num_inputs):
                        j.weights.append([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)])
                else:
                    for neuron in range(len(self.layers[i - 1])):
                        j.weights.append([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)])

    def run_network(self, inputs):
        """Runs the network through a SINGLE set of inputs (one data point) and returns the result. No training."""
        for i in range(len(self.layers)):
            if i == 0:
                for j in self.layers[i]:
                    for index, net_input in enumerate(inputs):
                        j.enqueue(net_input * j.weights[index][0] + j.weights[index][1])
            else:
                for index, neuron in enumerate(self.layers[i - 1]):
                    neuron.calc_output()
                    for j in self.layers[i]:
                        j.enqueue(neuron.output * j.weights[index][0] + j.weights[index][1])
        output = []
        for out_neuron in self.layers[len(self.layers) - 1]:
            out_neuron.calc_output()
            output.append(out_neuron.output)
        return output

    def train_network(self, inputs, expected):
        random.seed(time.time())
        changed = False
        """Runs the network over multiple data points, and iterates one generation of a semi-genetic algorithm."""
        output = []
        for data_index, data_point in enumerate(inputs):  # For each set of inputs
            output.append(self.run_network(inputs[data_index]))

        # Error calculated this way so that larger individual errors impact the end result more
        original_error = 0
        original_max_error = 0
        for index1, data_point in enumerate(output):
            for index2, out in enumerate(data_point):
                original_error += (out - expected[index1][index2]) ** 2.0
                if (out - expected[index1][index2]) ** 2 > original_max_error:
                    original_max_error = (out - expected[index1][index2]) ** 2

        original_error = sqrt(original_error)
        original_max_error = sqrt(original_max_error)
        original_weights = self.__get_weights__()

        for i in range(1000):
            if i % 100 == 0:
                print("Starting the " + str(i) + "th run.")
            for j in range(random.randint(1, 1000)):
                l_index = random.randint(0, len(self.layers) - 1)
                n_index = random.randint(0, len(self.layers[l_index]) - 1)
                w_index = random.randint(0, len(self.layers[l_index][n_index].weights) - 1)
                w_delta = self.temperature * random.uniform(-1, 1)
                index = random.randint(0, 1)
                # self.layers[l_index][n_index].weights[w_index] = \
                #     max(-1.0, min(self.layers[l_index][n_index].weights[w_index] + w_delta, 1.0))
                self.layers[l_index][n_index].weights[w_index][index] = \
                    self.layers[l_index][n_index].weights[w_index][index] + w_delta

            output = []
            for data_index, data_point in enumerate(inputs):  # For each set of inputs
                output.append(self.run_network(inputs[data_index]))

            # Error calculated this way so that larger individual errors impact the end result more
            error = 0
            for index1, data_point in enumerate(output):
                for index2, out in enumerate(data_point):
                    error += (out - expected[index1][index2]) ** 2.0

            error = sqrt(error)

            if error < original_error:
                original_weights = self.__get_weights__()
                original_error = error
                changed = True
            else:
                self.__set_weights__(original_weights)

        if not changed:
            self.temperature *= 0.99
            print("Setting temperature to " + str(self.temperature))
        return [original_error, original_max_error]

    def __get_weights__(self):
        weights = []
        for l_index, layer in enumerate(self.layers):
            weights.append([])
            for n_index, neuron in enumerate(self.layers[l_index]):
                weights[l_index].append([])
                for w_index, weight in enumerate(self.layers[l_index][n_index].weights):
                    weights[l_index][n_index].append([])
                    for index, weight_type in enumerate(weight):
                        weights[l_index][n_index][w_index].append(weight_type)
        return weights

    def __set_weights__(self, weights):
        for l_index, layer in enumerate(self.layers):
            for n_index, neuron in enumerate(self.layers[l_index]):
                for w_index, weight in enumerate(self.layers[l_index][n_index].weights):
                    for index, weight_type in enumerate(weight):
                        self.layers[l_index][n_index].weights[w_index][index] = \
                            weights[l_index][n_index][w_index][index]
