class TrainingData:
    def __init__(self):
        self.data_points_list = []

    def append(self, data_point):
        if not isinstance(data_point, DataPoint):
            raise Exception("Can only append objects of type DataPoint to TrainingData.")
        self.data_points_list.append(data_point)

    def get_inputs(self):
        data_inputs = []
        for data_point in self.data_points_list:
            data_inputs.append(data_point.inputs)
        return data_inputs

    def get_expected(self):
        data_expected = []
        for data_point in self.data_points_list:
            data_expected.append(data_point.expected)
        return data_expected


class DataPoint:
    def __init__(self, inputs, expected_outputs):
        self.inputs = inputs
        self.expected = expected_outputs
