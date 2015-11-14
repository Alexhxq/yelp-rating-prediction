import numpy as np
import itertools
import collections
import math
import sys

class neuralNet:

	def __init__(n_input, n_output, n_hidden_layer, n_hidden_node, function):
		self.n_input = n_input
		self.n_output = n_output
		self.n_hidden_layer = n_hidden_layer
		self.n_hidden_node = n_hidden_node
		self.function = function
		self.weight = []

		# input weight
		w = np.random.uniform((self.n_input+1)*self.n_hidden_node).reshape(self.n_hidden_node, (self.n_input+1))
		self.weight += w
		
		# hidden weight
		for i xrange(self.n_hidden_layer-1):
			w = np.random.uniform((self.n_hidden_node+1)*n_hidden_node).reshape(self.n_hidden_node, (self.n_hidden_node+1))
			self.weight += w

		# output weight
		w = np.random.uniform(self.n_output*(n_hidden_node+1)).reshape(self.n_output, (self.n_hidden_node+1))
		self.weight += w

	def addOne(data):¬
		return np.hstack(np.ones(data.shape[0],1), data)¬

	def backpropagate(training_data, error_limit=1e-3, learning_rate=0.2, momentum_factor=0.8):
		def addOne(data):
			return np.hstack(np.ones(data.shape[0],1), data)

		training_feature = np.array([instance.feature for instance in training_data])
		training_target = np.array([instance.output for instance in training_data])

		MSE = sys.maxsize
		momentum = collections.defaultdict(int)
		
		epoch = 0
		while MSE > error_limit:
			epoch += 1

			# compute the feed-forward result
			input_layers = self.update(training_data, trace=True)
			output_layer = input_layers[-1]
			
			error = training_target - output_layer
			delta = error
			MSE = np.mean(np.power(error, 2))

			loop = itertools.izip(
										xrange((len(self.weight)-1,-1,-1)),
										reversed(self.weight),
										reversed(input_layers[:-1])
								)

			for i, weight_value, output_value in loop:

				dW = learning_rate * np.dot(delta, addOne(output_value).T) + momentum_factor*momentum[i]
				
				weight_output = np.dot(output_value, weight_value)
				delta = np.multiply(weight_output, self.function[i-1](output_value, derivative=True))

				momentum[i] = dW
				self.weight[i] += dW

			print "* current network error (MSE):", MSE

		print "* Converged to error bound (%.4g) with MSE = %.4g." % ( ERROR_LIMIT, MSE )
		print "* Trained for %d epochs." % epoch

	def update(self, training_data, trace=True):
		input_layers = [training_data]
		
		for i, weight_layer in enumerate(self.weight):
			output = np.dot(output, weight_layer.T)
			output = self.function[i](output)
			input_layers.append(output)

		return input_layers

	def save_to_file(self, filename = "network.pkl" ):
		import cPickle
    """
    This save method pickles the parameters of the current network into a 
    binary file for persistant storage.
    """
		with open( filename , 'wb') as file:
			store_dict = {
				"n_inputs"             : self.n_inputs,
				"n_outputs"            : self.n_outputs,
				"n_hiddens"            : self.n_hiddens,
				"n_hidden_layers"      : self.n_hidden_layers,
				"activation_functions" : self.activation_functions,
				"n_weights"            : self.n_weights,
				"weights"              : self.weights
			}
			cPickle.dump( store_dict, file, 2 )
	#end