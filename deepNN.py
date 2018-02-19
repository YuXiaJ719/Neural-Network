import numpy as np

def initializ_parameters(layers_dims, seed=18):
	np.random.seed(seed)
	parameters = {}
	n = len(layers_dims)

	for i in range(1, n):
		parameters["W" + str(i)] = np.random.randn((layers_dims[i], layers_dims[i - 1])) * 0.01
		parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))

	return parameters

def sigmoid(Z):
	cache = Z
	A =  1 / (1 + np.exp(-Z))

	return A, cache

def relu(Z):
	A = np.maximum(0, Z)
	cache = Z

	return A, cache

def sigmoid_backward(dA, cache):
	Z = cache
	a = 1 / (1 + np.exp(-Z))
	dZ = dA * a *(1 - a)

	assert(dZ.shape == Z.shape)

	return dZ

def relu_backward(dA, cache):
	Z = cache
	dZ = np.array(dA, copy = True)
	dZ[Z <= 0] = 0

	assert(Z.shape == dZ.shape)

	return dZ
	
def linear_forward(W, A, b):
	Z = np.dot(W, A) + b

	assert(Z.shape == (W.shape[0], A.shape[1]))

	cache = (A, W, b)

	return Z, cache

def linear_activation_forward(A_prev, W, b, activation):

	if activation == "sigmoid":
		Z, linear_cache = linear_forward(W, A_prev, b)
		A, activation_cache = sigmoid(Z)

	elif activation == "relu":
		Z, linear_cache = linear_forward(W, A_prev, b)
		A, activation_cache = relu(Z)

	cache = (linear_cache, activation_cache)

	return A, cache

def deep_model_forward(X, parameters):

	caches = []
	A = X
	L = len(parameters) // 2

	for i in range(1, L):
		A_prev = A
		A, cache = linear_activation_forward(parameters["W" + str(i)], A_prev, parameters["b" + str(i)], activation = "relu")
		caches.append(cache) # ((W, A, b), Z)

	AL, cache = linear_activation_forward(parameters["W" + str(L)], A_prev, parameters["b" + str(L)], activation = "sigmoid")
	caches.append(cache)

	return AL, caches

def compute_cost(AL, Y):

	m = Y.shape[1]
	cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m

	cost = np.squeeze(cost)

	return cost

def linear_backward(dZ, cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = np.dot(dZ, A_prev.T) / m
	db = np.sum(dZ, axis = 1, keepdims = True) / m
	dA_prev = np.dot(W.T, dZ)

	return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
	linear_cache, activation_cache = cache

	if activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

	elif activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)


	return dA_prev, dW, db


def deep_model_backward(AL, Y, caches):
	grads = {}
	L = len(caches)
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)

	dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	current_cache = caches[L-1]
	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

	for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 

    for i in range(L):
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * grads["dW" + str(i + 1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]

    return parameters
