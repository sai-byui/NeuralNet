class TrainingData:
    """TrainingData class, used to enable easier input of training and testing data for a neural network"""
    def __init__(self):
        self.__data_points_list__ = []

    def append(self, data_point):
        """Adds a DataPoint to the list."""
        if not isinstance(data_point, DataPoint):
            raise Exception("Can only append objects of type DataPoint to TrainingData.")
        self.__data_points_list__.append(data_point)

    def get_inputs(self):
        """Returns only the inputs from each data point."""
        data_inputs = []
        for data_point in self.__data_points_list__:
            data_inputs.append(data_point.inputs)
        return data_inputs

    def get_expected(self):
        """Returns only the expected output values from each data point."""
        data_expected = []
        for data_point in self.__data_points_list__:
            data_expected.append(data_point.expected)
        return data_expected


class DataPoint:
    """Simple structure to hold the inputs and associated expected outputs of a single piece of data."""
    def __init__(self, inputs, expected_outputs):
        self.inputs = inputs
        self.expected = expected_outputs
