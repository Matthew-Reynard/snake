'''

This is using the code for my first NN.
Code from sentdex (YouTube) TensorFlow tutorial - https://www.youtube.com/watch?v=BhpvH5DuVu8

input > weight > hl1 (activation function) > weights > hl2 > weights > output

NOT FOR Q LEARNING
compare output to intended output (label) > cost function - NOT FOR Q LEARNING
optimization functions (optimizer) > minimise cost (AdamOptimizer... SGD (Stochatis Gradient Descent), AdaGrad)

backpropagation - but where?

Feed forward + backprop = epoch (We will do 10 epochs in this example)

'''

import numpy as np
import tensorflow as tf

from Environment_for_DQN import Environment

# First we need our environment form Environment_for_DQN.py
env = Environment()

# Then we need our data
input_data = env.input_data()

# Number of nodes at input layer
input_nodes = 400
# Number of hidden layer nodes - 3 Hidden layers => 5 layers in total
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# Number of actions - Up, down, left, right
n_actions = 4

# manipulate the weights of 100 inputs at a time
batch_size = 100

# 10 x 10 x 4 = 400 (input nodes)

# Dont have to have those -> [None, 400] -> values here, 
# but TF will throw an error if you give it something different.
# Otherwise without the values, it wont throw an error.
# So it's just safety incase something of not that shape is given in there
x = tf.placeholder('float', [None, input_nodes])
y = tf.placeholder('float')


def createModel(data):

	# will create an array (tensor) of your weights - initialized to random values
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([input_nodes, n_nodes_hl1])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))} 

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_actions])),
					  'biases': tf.Variable(tf.random_normal([n_actions]))}

	# (input data * weights) + biases
	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

	return output

def deep_q_learning(x):

	Q_values = createModel(x)
	
	# Need to alter the code below to do Deep Q Learning


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
	


def main():
	print("Running the Deep Q-Learning Model")

	deep_q_learning(x)


if __name__ == '__main__':
	main()