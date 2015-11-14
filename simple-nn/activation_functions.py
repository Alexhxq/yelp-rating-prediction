from scipy.special import expit
import numpy as np

def sigmoid_function(data, derivative=False):
	data = np.clip(data, -500, 500)
	data = expit(data)

	if derivative:
		return np.multipy(data, 1-data)
	else:
		return data