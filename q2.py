from mnist import MNIST
import numpy as np
import numpy.linalg as LA
from random import shuffle
from random import sample
from random import randrange

'''
1. Loading MNIST data
'''
def load_mnist():

	def sort_by_target(mnist):
		reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
		reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
		mnist.data[:60000] = mnist.data[reorder_train]
		mnist.target[:60000] = mnist.target[reorder_train]
		mnist.data[60000:] = mnist.data[reorder_test + 60000]
		mnist.target[60000:] = mnist.target[reorder_test + 60000]
		

	try:
		from sklearn.datasets import fetch_openml
		mnist = fetch_openml('mnist_784', version=1, cache=True)
		mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
		sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
		return mnist

	except ImportError:
		from sklearn.datasets import fetch_mldata
		mnist = fetch_mldata('MNIST original')


def vf(inp):
	vector = np.zeros((10,1))
	vector[int(inp)] = 1/1.0
	return vector

normalizing_factor = 255/1.0
flatten = (784, 1)
select = 60000

mnist = load_mnist()
mnist.data = mnist.data/normalizing_factor

train_images, train_labels = [], []
test_images, test_labels = [], []

# training data_set

for ti in mnist.data[:select]:
	train_images.append(np.reshape(ti, flatten))

for tl in mnist.target[:select]:
        train_labels.append(vf(tl))
		
train_dataset = [(im, lb) for im,lb in zip(train_images, train_labels)]

# testing data_set
for ti in mnist.data[select:]:
        test_images.append(np.reshape(ti, flatten))

for tl in mnist.target[select:]:
	test_labels.append(vf(tl))

test_dataset = [(i, l) for i, l in zip(test_images, test_labels)]
shuffle(train_dataset)


'''
2. Perform k means clustering
'''

def clusters_creation(centroids_input):
	clusters = []
	for _ in range(len(centroids_input)): clusters.append([])
	
	for (a,b) in data:
		least_d, ci = float('inf'), 0
		index = 0
		for index in range(len(centroids_input)):
			d = LA.norm(a - centroids_input[index])
			if (d < least_d): ci, least_d = index, d
		
		clusters[ci].append((a,b))
	
	return clusters
	
def centroids_modification(clusters_input):
	modified_centroids = []
	for j in range(len(clusters_input)):
		c_sum = clusters_input[j][0][0].copy()
		for k in range(1, len(clusters_input[j])): c_sum += clusters_input[j][k][0]
		modified_centroids.append(c_sum * (1.0 / len(clusters_input[j])))
		
	return modified_centroids

def check_break_condition(dc):
	return np.isnan(dc)

	
# training data
data = train_dataset[:1000].copy()

# initializing centroids
centroids = []
choices = sample(data, 16)
for choice in choices: centroids.append(choice[0])	

#initializing clusters
clusters = clusters_creation(centroids)
	
old = 0
for i in range(1, 100):
	stale, diff = centroids, []
	centroids = centroids_modification(clusters)
	clusters = clusters_creation(centroids)
	for i in range(len(centroids)): diff.append(LA.norm(stale[i] - centroids[i])) 
	
	md = max(diff)
	dc = abs((md-old)/np.mean([old,md])) * 100
	
	if check_break_condition(dc):break

	
'''
3. k-fold cross validation
'''

layers = [len(centroids), 10]

# Initialize the weights
weights = []
for dim_row, dim_col in zip(layers[:-1], layers[1:]):
	weight_matrix = []
	for i in range(dim_col):
		row = []
		for j in range(dim_row):
			row.append(np.random.randn())
		weight_matrix.append(row)
	weights.append(np.asarray(weight_matrix))
	
# Initialize the biases
biases = []
for dim_row in layers[1:]:
	bias_vector = []
	for j in range(dim_row):
		row = []
		row.append(np.random.randn())
		bias_vector.append(row)
	biases.append(np.asarray(bias_vector))


def helper_ff(a):
	temp_arr = []
	for b, c in zip(centroids, clusters): temp_arr.append([gaussian(a, b, c)])
	return np.dot(weights[0], ( np.reshape(np.asarray(temp_arr), (len(centroids), 1)) )) + biases[0]
	
def compute_acc(data_input):
	temp_arr = []
	for elem in data_input:
		i,l = elem[0], elem[1]
		a, b = np.argmax(helper_ff(i)), np.argmax(l)
		temp_arr.append( (a,b) )
	
	r = 0
	for t in temp_arr: r += (int(t[0] == t[1]))
	return r
	

def gaussian(a, b, c):
	s = 0
	for _ in range(len(c)): s += 1.0 * (LA.norm(a - b))
	s *= len(c)
	return np.exp(-( 2 * pow(s, 2) )*(pow(LA.norm(a - b), 2)))
	
	
def helper_backprop(img, lb):
	def init_grad(input):
		output = []
		for i in range(len(input)): output.append( np.zeros(input[i].shape) )
		return output
		
	def network_opt_helper(input):
		return np.dot(weights[0], input) + biases[0]
	
	def calc_nr_err(hd_nr_err, f):
		prod = np.dot(weights[-f+1].transpose(), hd_nr_err)
		error_p_one, error_p_two = 1.0/(1.0 + np.exp(-actv[-f])), 1.0 - (1.0/(1.0 + np.exp(-actv[-f])))		
		return prod * ( error_p_one * error_p_two )

		
	grad_for_weights, grad_for_biases = init_grad(weights), init_grad(biases)

	# feed-forward
	actv, trnsf = [], []

	temp_gsn_array = []
	iterator = zip(centroids, clusters)
	for m, c in iterator: temp_gsn_array.append([gaussian(img, m, c)])
	
	np_array = np.asarray(temp_gsn_array)
	new_shape = (len(centroids), 1)
	hdn_opt = np.reshape(np_array, new_shape)
	ntwrk_opt = network_opt_helper(hdn_opt)
	
	actv = [img]
	trnsf = [hdn_opt, ntwrk_opt]
	
	# backprop -- compute error on last neuron
	prev_err = trnsf[-1] - lb
	grad_for_biases[-1] = prev_err
	
	param_a, param_b = prev_err, trnsf[-2]
	grad_for_weights[-1] = np.dot(param_a, param_b.transpose())

	# correct the other neurons in the hidden layers based on previous calculations
	hd_nr_err = prev_err
	for f in range(2, len(layers)):
		hd_nr_err = calc_nr_err(hd_nr_err, f)
		grad_for_biases[-f], grad_for_weights[-f] = hd_nr_err, np.dot(hd_nr_err, trnsf[-f-1].transpose())

	return (grad_for_weights, grad_for_biases)

def grad_desc(input_dataset_for_grad_desc):
	for i in range(10):
		print("Epoch num: " + str(i+1))
		
		# to avoid over-fitting, we will shuffle the training-set
		shuffle(input_dataset_for_grad_desc)
		for j in range(len(input_dataset_for_grad_desc)):
			# update weights
			weight_gradients, bias_gradients = helper_backprop(input_dataset_for_grad_desc[j][0], input_dataset_for_grad_desc[j][1])

			for k in range(len(weights)): weights[k] -= 0.1 * weight_gradients[k] # update weights
			for k in range(len(biases)): biases[k] -= 0.1 * bias_gradients[k] # update biases
		
		print("Completed epoch num: " + str(i+1) + " and the accuracy is: ")
		print('%.3f%%' % ((compute_acc(input_dataset_for_grad_desc)/len(input_dataset_for_grad_desc))*100))
		print("\n")

data = train_dataset[:1000].copy()
# split data into k folds
folds, folds_count  = [], 5
for i in range(folds_count):
	fold = []
	while len(fold) < int(len(data)/folds_count):
		index = randrange(len(data))
		fold.append(data.pop(index))
	folds.append(fold)


acc = []
for i in range(folds_count):
	tf = []
	
	# create training_set
	for m in range(folds_count):
		if (i != m): tf.append(folds[m])
			
	print("\nFold being tested: " + str(i+1))
	print("--------------------")
	tf = sum(tf,[])
	
	grad_desc(tf) #compute gradient descent
	
	# testing the network
	accuracy = compute_acc(folds[i])/len(folds[i])
	acc.append(accuracy*100)
	print("Fold: " + str(i+1) )
	print("Accuracy: ", end = "")
	print("%.3f%%" % acc[i])
	
