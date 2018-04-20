'''
Simple SnakeAI Game with a Deep Q Network

INFO:
@author: Matthew Reynard
@year: 2018

DESCRIPTION:
This is using the code for my first NN.
Code from sentdex (YouTube) TensorFlow tutorial - https://www.youtube.com/watch?v=BhpvH5DuVu8

input > weight > hl1 (activation function) > weights > hl2 (activation function) > weights > output

--- NOT FOR Q LEARNING ---
compare output to intended output (label) > cost function - NOT FOR Q LEARNING
optimization functions (optimizer) > minimise cost (AdamOptimizer... SGD (Stochatis Gradient Descent), AdaGrad)

backpropagation - but where?

Feed forward + backprop = epoch (We will do 10 epochs in this example)

TO DO LIST:
- Learn how to train the RL Model


NOTES:
- Random weights and biases are used for the baseline

BUGS:
- None... yet

'''

import numpy as np
import tensorflow as tf
from Environment_for_DQN import Environment

# Number of nodes at input layer
n_input_nodes = 300

# Number of hidden layer nodes - 3 Hidden layers => 5 layers in total
n_nodes_hl1 = 200
# n_nodes_hl2 = 200
# n_nodes_hl3 = 200

# Number of actions - Up, down, left, right
n_actions = 4

# manipulate the weights of 100 inputs at a time - NOT SURE HOW TO UPDATE WEIGHTS FOR RL
batch_size = 100

# 10 x 10 x 4 = 400 (input nodes)

# Dont have to have those -> [None, 400] -> values here, 
# but TF will throw an error if you give it something different.
# Otherwise without the values, it wont throw an error.
# So it's just safety incase something of not that shape is given in there

x = tf.placeholder(tf.float32, [1, n_input_nodes])

# x = tf.placeholder(tf.float32, [n_input_nodes, None]) # Dont understand the None

# y = tf.placeholder(tf.float32, [n_actions, None])

# This creates the NN model
# Parameters:
# Input layer: (300)	
# Hidden layer: (200)
# Output layer (4)

# Full Matrix diagram:
# Input_Layer(1,300) x Weights1(300,200) -> Hidden_Layer(1,200) x Weights2(200,4) -> Output_Layer(1,4)

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

# Main function
def main():

	# Testing
	print("Running the Deep Q-Learning Model")

	# Decide whether or not to render to the screen or not
	RENDER_TO_SCREEN = True

	# First we need our environment form Environment_for_DQN.py
	# has to have a grid_size of 10 for this current NN
	env = Environment(wrap = True, grid_size = 10, rate = 1000, max_time = 100, tail = False)
	
	if RENDER_TO_SCREEN:
		env.prerender()

	# Not used yet
	alpha = 0.15  # Learning rate, i.e. which fraction of the Q values should be updated
	gamma = 0.99  # Discount factor, i.e. to which extent the algorithm considers possible future rewards

	epsilon = 0.0  # Probability to choose random action instead of best action

	# Random tests
	# x1 = tf.placeholder(tf.float32, [n_input_nodes, 1])
	# x2 = tf.placeholder(tf.float32, [1, 1])
	# z = tf.matmul(x1,x2)

	# with tf.Session() as sess:

		# sess.run(tf.global_variables_initializer())

		# r1 = np.random.rand(n_input_nodes, 1)
		# r2 = np.random.rand(1, 1)

		# m = sess.run(z, feed_dict={x1: env.state_vector(), x2: r2})
		# print(m)
		# print("size:", sess.run(tf.shape(z)))
		# print("index of max:", sess.run(tf.argmax(m, axis=0)))


	# Testing my DQN model with random values
	for episode in range(10):
		state, info = env.reset()
		done = False

		# Create a new model - this creates different weights and biases for each episode
		Q_values = createModel(x)

		# Open a new session with each new episode - NOT SURE WHY I'M CHOOSING TO DO THIS
		# Session can start running
		with tf.Session() as sess:

			# Need this for something - won't work without it
			sess.run(tf.global_variables_initializer())

			while not done:
				if RENDER_TO_SCREEN:
					env.render()

				# Retrieve the Q values from the NN
				q = sess.run(Q_values, feed_dict={x: env.state_vector()})
				print(q) # DEBUGGING

				# Deciding one which action to take
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