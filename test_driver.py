from Network import Network

print("Creating network.")
net = Network([4,4,4,4,4, 1], 2)
# training_data = [[[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 0, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1], [1, 1, 0, 1]]]
print("Training network...")

training_data = [[[0,0],[0,1],[1,0],[1,1]],[[0],[1],[1],[1]]]
current_error = net.train_network_mark2(training_data[0], training_data[1])
print("Current network error: " + str(current_error))
i = 0
while current_error[1] > 0.2:  # * len(training_data[1]) * len(training_data[1][0]):
    i += 1
    # print("Training network...")
    prev_error = current_error
    current_error = net.train_network_mark2(training_data[0], training_data[1])
    if current_error[0] >= prev_error[0] and current_error[1] >= prev_error[1]:
        net.training_rate *= 0.9999
    if i == 2:
        print()
        print("100 more iterations...")
        print("Current network error: " + str(current_error[0]))
        print("Current max error: " + str(current_error[1]))
        print("Training rate: " + str(net.training_rate))
        i = 0

for data_point in training_data[0]:
    print(net.run_network(data_point))
