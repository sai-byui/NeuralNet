from Network import Network

print("Creating network.")
net = Network([[11], [11], [11], [11], [11], [11], [11], [11], [11], [11], [4]], 2)
training_data = [[[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 0, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1], [1, 1, 0, 1]]]
print("Training network...")
current_error = net.train_network(training_data[0], training_data[1])
print("Current network error: " + str(current_error))
while current_error > 0.001 * len(training_data[1]) * len(training_data[1][0]):
    print()
    print("Training network...")
    current_error = net.train_network(training_data[0], training_data[1])
    print("Current network error: " + str(current_error))

for data_point in training_data[0]:
    print(net.run_network(data_point))
