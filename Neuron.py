from math import e


class Neuron:
    """Neuron class, used for describing a neural network."""
    def __init__(self):
        self.weights = []
        self.inputs = []
        self.output = None

        # Variables specifically used for backpropagation
        self.num_runs = 0
        self.total_output = 0.0
        self.average_output = 0.0
        self.total_requested_delta = 0.0
        self.num_requests = 0
        self.average_requested_delta = 0.0
        self.total_inputs = 0.0
        self.average_input = 0.0

    def enqueue(self, input_amt):
        """Adds an input to the queue, to be processed by calc_output"""
        self.inputs.append(input_amt)

    def calc_output(self):
        """Uses a sigmoid function, clamping it between 0 and 1 exclusive"""
        total = 0.0
        for i in self.inputs:
            total += i
            self.total_inputs += i
        self.inputs = []
        self.output = (1 / (1 + e ** (-total)))
        self.num_runs += 1
        self.total_output += self.output
        self.average_output = self.total_output / self.num_runs
        self.average_input = self.total_inputs / self.num_runs

    def reset_backprop_vars(self):
        """Resets the variables associated with backpropagation"""
        self.num_runs = 0
        self.total_output = 0.0
        self.average_output = 0.0
        self.total_requested_delta = 0.0
        self.num_requests = 0
        self.average_requested_delta = 0.0
        self.total_inputs = 0.0
        self.average_input = 0.0

    def request_delta(self, delta):
        """Requested from the network that this neuron change its output by a certain amount.
        Requests are later averaged, calculus is applied, etc."""
        self.total_requested_delta += delta
        self.num_requests += 1
        self.average_requested_delta = self.total_requested_delta / self.num_requests
