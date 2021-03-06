from Neuron import Neuron
import random
from math import sqrt, e, pi, cos, sin
import time


class Network:
    """Network class. Contains all the code needed to describe a simple feedforward network. 
    Layers, neurons, weights, etc. Contains functions for training the network."""

    def __init__(self, topology):
        self.topology = topology  # For copying during the training process
        self.layers = []
        for index, layer in enumerate(topology):
            self.layers.append([])
            for neuron in range(layer):
                self.layers[index].append(Neuron())
        print("There are " + str(len(self.layers)) + " Layers")
        self.__init_weights__()
        self.temperature = 1.0
        self.training_rate = 0.01

    def __init_weights__(self):
        random.seed(time.time())
        for i in range(len(self.layers)):
            for j in self.layers[i]:
                if i == 0:
                    for net_input in range(len(self.layers[0])):
                        j.weights.append([1, 0])
                else:
                    for neuron in range(len(self.layers[i - 1])):
                        j.weights.append([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)])

    def run_network(self, inputs):
        """Runs the network through a SINGLE set of inputs (one data point) and returns the result. No training."""

        if len(inputs) != len(self.layers[0]):
            raise Exception("Trying to run data with " + str(len(inputs)) + " inputs through network with " +
                            str(len(self.layers[0])) + " inputs!")

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

    def train_network_gen_alg(self, inputs, expected):
        """Trains the neural network with a pseudo-genetic algorithm. Returns a list: 
        [error metric, max error in any output]"""
        random.seed(time.time())
        changed = False
        """Runs the network over multiple data points, and iterates one generation of a semi-genetic algorithm."""
        output = []
        for data_point in inputs:  # For each set of inputs
            output.append(self.run_network(data_point))

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
            for data_point in inputs:  # For each set of inputs
                output.append(self.run_network(data_point))

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

    def train_network_backprop(self, inputs, expected):
        """Trains the network using backpropagation. Not currently functional. Returns error in the same format as 
        train_network_gen_alg"""
        random.seed(time.time())
        error = []
        for i in range(len(self.layers[len(self.layers) - 1])):
            error.append([])

        for layer in self.layers:
            for neuron in layer:
                neuron.reset_backprop_vars()

        for data_point_index, data_point in enumerate(inputs):
            results = self.run_network(data_point)
            for out_neuron_index, out_neuron in enumerate(self.layers[len(self.layers) - 1]):
                error[out_neuron_index].append(expected[data_point_index][out_neuron_index] - results[out_neuron_index])

        for out_index, output in enumerate(error):
            for error_point in output:
                self.layers[len(self.layers) - 1][out_index].request_delta((error_point * abs(error_point)) *
                                                                           self.training_rate)

        for layer_index in range(len(self.layers)):
            layer_index = len(self.layers) - 1 - layer_index
            for neuron_index, neuron in enumerate(self.layers[layer_index]):
                if layer_index == 0:  # first layer
                    for w_index, weight in enumerate(neuron.weights):
                        # Change weight - use average value for that input instead of neuron average activation
                        total_input = 0.0
                        for data_point in inputs:
                            total_input += data_point[w_index]
                        average_input = total_input / len(inputs)
                        d_weight = neuron.average_requested_delta
                        d_weight /= average_input
                        d_weight /= (e ** (-neuron.average_input) + 1) ** 2
                        d_weight /= len(neuron.weights)
                        neuron.weights[w_index][0] += d_weight
                        if neuron.weights[w_index][0] > 1.0:
                            neuron.weights[w_index][0] = 1.0
                        if neuron.weights[w_index][0] < -1.0:
                            neuron.weights[w_index][0] = -1.0

                        # Change bias
                        d_bias = neuron.average_requested_delta
                        d_bias /= e ** neuron.average_input
                        d_bias *= (e ** neuron.average_input + 1) ** 2
                        d_bias /= len(neuron.weights)
                        neuron.weights[w_index][1] += d_bias
                        if neuron.weights[w_index][1] > 1.0:
                            neuron.weights[w_index][1] = 1.0
                        if neuron.weights[w_index][1] < -1.0:
                            neuron.weights[w_index][1] = -1.0
                else:  # any other layer
                    for w_index, weight in enumerate(neuron.weights):
                        # Request delta from previous neuron
                        d_previous_neuron = neuron.average_requested_delta
                        if weight[0] == 0:
                            print(weight)
                        d_previous_neuron /= weight[0]
                        d_previous_neuron /= (e ** (-neuron.average_input) + 1) ** 2
                        d_previous_neuron /= len(neuron.weights)
                        self.layers[layer_index - 1][w_index].request_delta(d_previous_neuron)

                        # Change weight
                        d_weight = neuron.average_requested_delta
                        d_weight /= self.layers[layer_index - 1][w_index].average_output
                        d_weight /= (e ** (-neuron.average_input) + 1) ** 2
                        d_weight /= len(neuron.weights)
                        neuron.weights[w_index][0] += d_weight
                        if neuron.weights[w_index][0] > 1.0:
                            neuron.weights[w_index][0] = 1.0
                        if neuron.weights[w_index][0] < -1.0:
                            neuron.weights[w_index][0] = -1.0

                        # Change bias
                        d_bias = neuron.average_requested_delta
                        d_bias /= (e ** (-neuron.average_input) + 1) ** 2
                        d_bias /= len(neuron.weights)
                        neuron.weights[w_index][1] += d_bias
                        if neuron.weights[w_index][1] > 1.0:
                            neuron.weights[w_index][1] = 1.0
                        if neuron.weights[w_index][1] < -1.0:
                            neuron.weights[w_index][1] = -1.0

        error = []
        for i in range(len(self.layers[len(self.layers) - 1])):
            error.append([])

        for data_point_index, data_point in enumerate(inputs):
            results = self.run_network(data_point)
            for out_neuron_index, out_neuron in enumerate(self.layers[len(self.layers) - 1]):
                error[out_neuron_index].append(expected[data_point_index][out_neuron_index] - results[out_neuron_index])

        total_error = 0.0
        max_error = 0.0
        for error_point in error:
            for output in error_point:
                total_error += output ** 2
                if abs(output) > max_error:
                    max_error = abs(output)
        total_error = sqrt(total_error)
        return[total_error, max_error]

    def train_network_gen_alg_mark2(self, inputs, expected):
        """Trains network using a different variation of a genetic algorithm."""

        theta = pi / 2
        population = []
        for i in range(500):
            population.append(Network(self.topology))
            weights = self.__get_weights__()
            population[i].__set_weights__(weights)
            for l_index, layer in enumerate(population[i].layers):
                for n_index, neuron in enumerate(population[i].layers[l_index]):
                    for w_index, weight in enumerate(population[i].layers[l_index][n_index].weights):
                        population[i].layers[l_index][n_index].weights[w_index][0] += \
                            random.uniform(-self.training_rate, self.training_rate)
                        population[i].layers[l_index][n_index].weights[w_index][1] += \
                            random.uniform(-self.training_rate, self.training_rate)

        while theta > 0:
            print("Theta: " + str(theta))
            target = self.get_target_from_theta(theta)
            ranked_list = []
            largest_avg_distance = 0
            largest_fitness = 0
            for net1 in population:
                total_distance = 0
                count = 0
                for net2 in population:
                    total_distance += net1.get_distance(net2)
                    count += 1
                net1.avg_distance = total_distance / count
                if net1.avg_distance > largest_avg_distance:
                    largest_avg_distance = net1.avg_distance

                error = []
                for i in range(len(net1.layers[len(net1.layers) - 1])):
                    error.append([])

                for data_point_index, data_point in enumerate(inputs):
                    results = net1.run_network(data_point)
                    for out_neuron_index, out_neuron in enumerate(net1.layers[len(net1.layers) - 1]):
                        error[out_neuron_index].append(
                            expected[data_point_index][out_neuron_index] - results[out_neuron_index])

                total_error = 0.0
                for error_point in error:
                    for output in error_point:
                        total_error += output ** 2
                total_error = sqrt(total_error)
                net1.fitness = 1/total_error
                if net1.fitness > largest_fitness:
                    largest_fitness = net1.fitness

            for net in population:
                net.normalized_avg_distance = net.avg_distance / largest_avg_distance
                net.normalized_fitness = net.fitness / largest_fitness

                net.distance_to_target = sqrt((net.normalized_fitness - target[0]) ** 2.0 +
                                              (net.normalized_avg_distance - target[1]) ** 2.0)

                if not ranked_list:
                    ranked_list.append(net)
                else:
                    for index, i in enumerate(ranked_list):
                        if net.distance_to_target <= i.distance_to_target:
                            ranked_list.insert(index, net)
                            break
                    else:
                        ranked_list.append(net)

            new_list = []
            for i in range(250):
                new_list.append(ranked_list[i])
            population = new_list

            for net in population:
                new_net = Network(net.topology)
                new_net.__set_weights__(net.__get_weights__())
                for l_index, layer in enumerate(new_net.layers):
                    for n_index, neuron in enumerate(new_net.layers[l_index]):
                        for w_index, weight in enumerate(new_net.layers[l_index][n_index].weights):
                            new_net.layers[l_index][n_index].weights[w_index][0] += random.uniform(
                                -self.training_rate, self.training_rate)
                            new_net.layers[l_index][n_index].weights[w_index][1] += random.uniform(
                                -self.training_rate, self.training_rate)

            theta -= 0.01

        best_fitness = 0
        best_net = None
        for net in population:
            if net.fitness:
                if net.fitness > best_fitness:
                    best_fitness = net.fitness
                    best_net = net

        self.__set_weights__(best_net.__get_weights__())
        return best_fitness

    def train_adversarial(self, fitness_callback):
        """Trains network using evolutionary algorithm and fitness instead of error. 
        Uses a callback so the user can determine how fitness is assigned."""

        # Create new network to be an adversary
        net2 = Network(self.topology)
        # Make the new network's weights only slightly different from the current network
        net2.__set_weights__(self.__get_weights__())
        for l_index, layer in enumerate(net2.layers):
            for n_index, neuron in enumerate(net2.layers[l_index]):
                for w_index, weight in enumerate(net2.layers[l_index][n_index].weights):
                    net2.layers[l_index][n_index].weights[w_index][0] += \
                        random.uniform(-self.training_rate, self.training_rate)
                    net2.layers[l_index][n_index].weights[w_index][1] += \
                        random.uniform(-self.training_rate, self.training_rate)
        # get the fitnesses by calling the fitness_callback with both networks as parameters
        fitness = fitness_callback(self, net2)
        # the fitness should have two numbers - like [2.5, 6.2] - the first for the current network's fitness,
        # the second for the new network. raise exception otherwise.
        if len(fitness) != 2:
            raise(Exception("ERROR: Fitness must be a list of two numbers!"))
        try:
            fitness[0] * fitness[1]
        except:
            raise(Exception("ERROR: Fitness must be a list of two numbers!"))
        # if the new network's fitness is higher, use its weights to replace those of the current network
        if fitness[1] > fitness[1]:
            self.__set_weights__(net2.__get_weights__())

    @staticmethod
    def get_target_from_theta(theta):
        """Used in train_network_gen_alg_mark2 to determine the target - whether the training should
        currently focus on diversity or fitness."""
        return [cos(theta), sin(theta)]

    def get_distance(self, net):
        """Gets the euclidean distance between two networks in n-dimensional space."""

        squares = 0
        for l_index, layer in enumerate(self.layers):
            for n_index, neuron in enumerate(layer):
                for w_index, weight in enumerate(neuron.weights):
                    squares += (weight[0] - net.layers[l_index][n_index].weights[w_index][0]) ** 2.0
                    squares += (weight[1] - net.layers[l_index][n_index].weights[w_index][1]) ** 2.0
        return sqrt(squares)

    def __get_weights__(self):
        weights = []
        for l_index, layer in enumerate(self.layers):
            weights.append([])
            for n_index, neuron in enumerate(layer):
                weights[l_index].append([])
                for w_index, weight in enumerate(neuron.weights):
                    weights[l_index][n_index].append([])
                    for weight_type in weight:
                        weights[l_index][n_index][w_index].append(weight_type)
        return weights

    def __set_weights__(self, weights):
        for l_index, layer in enumerate(self.layers):
            for n_index, neuron in enumerate(layer):
                for w_index, weight in enumerate(neuron.weights):
                    for index, weight_type in enumerate(weight):
                        weight[index] = \
                            weights[l_index][n_index][w_index][index]
