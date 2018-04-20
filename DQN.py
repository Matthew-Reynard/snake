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
# env = Environment()

# Then we need our data
# input_data = env.input_data()

# Number of nodes at input layer
n_input_nodes = 300

# Number of hidden layer nodes - 3 Hidden layers => 5 layers in total
n_nodes_hl1 = 200
# n_nodes_hl2 = 500
# n_nodes_hl3 = 500

# Number of actions - Up, down, left, right
n_actions = 4

# manipulate the weights of 100 inputs at a time
batch_size = 100

# 10 x 10 x 4 = 400 (input nodes)

# Dont have to have those -> [None, 400] -> values here, 
# but TF will throw an error if you give it something different.
# Otherwise without the values, it wont throw an error.
# So it's just safety incase something of not that shape is given in there

x = tf.placeholder(tf.float32, [1, n_input_nodes])

# x = tf.placeholder(tf.float32, [n_input_nodes, 1]) # Dont understand the None

# x = tf.placeholder(tf.float32, shape=(n_input_nodes, n_input_nodes))

# y = tf.placeholder(tf.float32, [n_actions, None])

# z = tf.matmul(x,x)

def createModel(data):

	# The structure of the model:

	# hidden_1_layer is simply a python dictionary (right?)
	# will create an array (tensor) of your weights - initialized to random values
	hidden_1_layer = {'weights': tf.Variable(tf.random_uniform([n_input_nodes, n_nodes_hl1], minval=0, maxval=None)),
					  'biases': tf.Variable(tf.random_uniform([n_nodes_hl1], minval=0, maxval=1))}

	# hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
	# 				  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))} 

	# hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
	# 				  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights': tf.Variable(tf.random_uniform([n_nodes_hl1, n_actions], minval=0, maxval=None)),
					'biases': tf.Variable(tf.random_uniform([n_actions], minval=0, maxval=1))}

	# The formula of whats happening below:
	# (input data * weights) + biases

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) # Activation function (ReLU)

	# l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	# l2 = tf.nn.relu(l2)

	# l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	# l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l1,output_layer['weights']), output_layer['biases'])

	# Normalize the output using the L2-Norm method
	output = tf.nn.l2_normalize(output)

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
	

# Main function
def main():

	print("Running the Deep Q-Learning Model")

	RENDER_TO_SCREEN = True

	# has to have a grid_size of 10 for this current NN
	env = Environment(wrap = True, grid_size = 10, rate = 1000, max_time = 100, tail = False)
	
	if RENDER_TO_SCREEN:
		env.prerender()

	# env.reset()

	# my_input = env.state_vector()


	# The Model
	# Q_values = createModel(x)

	# print(env.state_vector().shape)

	# print(" ")

	epsilon = 0.0

	# x1 = tf.placeholder(tf.float32, [n_input_nodes, 1])
	# x2 = tf.placeholder(tf.float32, [1, 1])

	# a = tf.placeholder(tf.float32)

	# z = tf.matmul(x1,x2)

	#Session can start running
	# with tf.Session() as sess:

		#sess.run(tf.initialize_all_variables()) #depricated
		# sess.run(tf.global_variables_initializer())

		# r1 = np.random.rand(n_input_nodes, 1)
		# r2 = np.random.rand(1, 1)

		# a = tf.argmax(z,1)

		# m = sess.run(z, feed_dict={x1: env.state_vector(), x2: r2})
		# q = sess.run(Q_values, feed_dict={x: env.state_vector()})

		# print(m)
		# print(q)

		# print("size:", sess.run(tf.shape(z)))
		# print("size:", sess.run(tf.shape(q)))

		# print("index of max:", sess.run(tf.argmax(m, axis=0)))
		# print("index of max:", sess.run(tf.argmax(q, axis=1)))

		# a = tf.argmax(z,1)

		# print(sess.run(a))

		# print(tf.equal(a,a))


	# Testing my DQN model with random values
	for episode in range(10):
		state, info = env.reset()
		done = False

		# Create a new model - this creates different weights and biases for each episode
		Q_values = createModel(x)

		# Open a new session with each new episode - NOT SURE WHY I'M CHOOSING TO DO THIS
		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())

			while not done:
				if RENDER_TO_SCREEN:
					env.render()

				# Retrieve the Q values from the NN
				q = sess.run(Q_values, feed_dict={x: env.state_vector()})
				print(q)

				if np.random.rand() <= epsilon:
					action = env.sample()
				else:
					# action is the max value of the Q values (output vector of NN)
					action = sess.run(tf.argmax(q, axis=1))
					# action = np.argmax(Q[env.state_index(state)])

				new_state, reward, done, info = env.step(action)

				# Q[env.state_index(state), action] += alpha * (reward + gamma * np.max(Q[env.state_index(new_state)]) - Q[env.state_index(state), action])

				state = new_state

			if episode % 1 == 0:
				print("Episode:", episode, "   Score:", info["score"])


if __name__ == '__main__':
	main()