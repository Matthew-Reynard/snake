'''
NOTE:

The following error states that my CPU is able to run TF faster, but I have to build it from source

If my NN takes too long to train, I can either upgrade my build of TF or buy a new Computer with a more powerful GPU, or at least with an NVIDIA GPU that is compatible with tensorflow-gpu

ERROR:
I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA

'''

import tensorflow as tf 

#importing mnist data set. Large set of numbers
from tensorflow.examples.tutorials.mnist import input_data
'''

This is the code for my first NN.
Code from sentdex (YouTube) TensorFlow tutorial - https://www.youtube.com/watch?v=BhpvH5DuVu8

input > weight > hl1 (activation function) > weights > hl2 > weights > output

compare output to intended output (label) > cost function
optimization functions (optimizer) > minimise cost (AdamOptimizer... SGD (Stochatis Gradient Descent), AdaGrad)

backpropagation

Feed forward + backprop = epoch (We will do 10 epochs in this example)

'''
#one_hot means from 0-9 (10 output classes): 0 = [1,0,0,0,0,0,0,0,0,0] and 5 =[0,0,0,0,0,1,0,0,0,0]
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#Number of hidden layer nodes
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#Number of classes
n_classes = 10
#manipulate the weights of 100 inputs at a time
batch_size = 100

#28 x 28 = 784 (images in mnist data set)

#height x width
#Dont have to have those -> [None, 784] -> values here, but TF will throw an error if you give it something different. Otherwise without the values, it wont throw an error.
#So its just safety incase something of not that shape is given in there
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

# we have no code that says modify weights - TF just does it

def neural_network_model(data):
	
	# will create an array (tensor) of your weights - initialized to random values
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))} 

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					  'biases': tf.Variable(tf.random_normal([n_classes]))}

	# (input data * weights) + biases
	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y)) # depricated function tf.nn.softmax_cross_entropy_with_logits()
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction,labels = y))
	optimizer = tf.train.AdamOptimizer().minimize(cost) # AdamOptimizer has a parameter learning rate (defaults to 0.001)

	#cycles of feed forward + backprop
	hm_epochs = 10

	#Session can start running
	with tf.Session() as sess:
		#sess.run(tf.initialize_all_variables()) #depricated
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			# _ variable is shorthand for "varible we dont care about"
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size) # lots of helper functions like this in TF
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c

			print('Epoch', epoch, 'completed out of', hm_epochs, '. Loss:', epoch_loss)

		#tf.argmax returns the max value in the array
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)

'''
Done in 104 lines of code with a ton of comments
TF is a very high level language
'''

'''
Lexicon
'''
