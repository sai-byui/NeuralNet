from Network import Network

print("Creating network.")
net = Network([[11], [11], [11], [11], [11], [11], [11], [11], [11], [11], [3]], 2)
training_data = [[[0,0], [0,1], [1,0],[1,1]], [[0,0,0],[0,1,1],[0,1,1],[1,1,0]]]
print("Training network...")
current_error = net.train_network(training_data[0], training_data[1])
print("Current network error: " + str(current_error))
while current_error > 0.001:
    #print("Current network result: " + str(net.run_network([0.2, 0.1, 0.7])) + str(net.run_network([0.5, 0.5, 0.5])))
    print()
    print("Training network...")
    current_error = net.train_network(training_data[0], training_data[1])
    print("Current network error: " + str(current_error))

#print("End network result: " + str(net.run_network([0.2, 0.1, 0.7])) + str(net.run_network([0.5, 0.5, 0.5])))
