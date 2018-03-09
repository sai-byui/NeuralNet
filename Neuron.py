from math import e


class Neuron:
    def __init__(self):
        self.weights = []
        self.inputs = []
        self.output = None

    def enqueue(self, input_amt):
        self.inputs.append(input_amt)

    def calc_output(self):
        """Uses a sigmoid function, clamping it between 0 and 1 exclusive"""
        total = 0.0
        for i in self.inputs:
            total += i
        self.inputs = []
        self.output = (e ** total) / (1 + e ** total)
