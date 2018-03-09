from Network import Network

print("Creating network.")
net = Network([[11], [10], [9], [8], [7], [6], [5], [4], [3], [2], [1]], 3)
training_data = [[[0, 0, 0], [0.5, 0.5, 0.5]], [[0.5], [0.2]]]
print("Training network...")
current_error = 0
for index, data in enumerate(training_data[0]):
    current_error += net.train_network([training_data[0][index]], [training_data[1][index]])
print("Current network error: " + str(current_error))
while current_error > 0:
    print("Current network result: " + str(net.run_network([0, 0, 0])) + str(net.run_network([0.5, 0.5, 0.5])))
    print()
    print("Training network...")
    for index, data in enumerate(training_data[0]):
        current_error = net.train_network([training_data[0][index]], [training_data[1][index]])
    print("Current network error: " + str(current_error))
