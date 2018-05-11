from Network import Network
from TrainingData import TrainingData, DataPoint

print("Creating network.")
net = Network([2, 8, 8, 4])

training_data = TrainingData()
#                     Inputs: 0 and 1  Expected outputs:and, or, nand, xor
training_data.append(DataPoint([0, 0], [0, 0, 1, 0]))
training_data.append(DataPoint([0, 1], [0, 1, 1, 1]))
training_data.append(DataPoint([1, 0], [0, 1, 1, 1]))
training_data.append(DataPoint([1, 1], [1, 1, 0, 0]))

print("Training network...")

current_error = net.train_network_gen_alg(training_data.get_inputs(), training_data.get_expected())
print("Current network error: " + str(current_error))
i = 0
while current_error[1] > 0.001:  # * len(training_data[1]) * len(training_data[1][0]):
    i += 1
    prev_error = current_error
    current_error = net.train_network_gen_alg(training_data.get_inputs(), training_data.get_expected())
    if i == 1:
        print()
        print("100 more iterations...")
        print("Current network error: " + str(current_error[0]))
        print("Current max error: " + str(current_error[1]))
        print("Temperature: " + str(net.temperature))
        i = 0

for data_input in training_data.get_inputs():
    print(net.run_network(data_input))
