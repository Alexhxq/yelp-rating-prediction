import numpy as np
from activation_functions import sigmoid_function

class Instance:
	
	def __init_(self, feature, output):
		self.feature = feature
		self.output = output

training_data = [Instance([0,0], [1]), Instance([1,1], [0])]

n_input = int(sys.argv[1])
n_output = int(sys.argv[2])
n_hidden_layer = int(sys.argv[3])
n_hidden_node = int(sys.argv[4])

function = [sigmoid_function]*n_hidden_layer + [sigmoid_function]

network = neuralNet(n_input, n_output, n_hidden_layer, n_hidden_node, function)

network.backpropagate(training_data, error_limit=1e-4, learning_rate=0.3, momentum_factor=0.9)

network.save_to_file("trained_configuration")

# predict the result
for instance in training_data:
	print network.update(np.array([instance.feature]), instance.output)
