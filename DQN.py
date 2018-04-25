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
- Not learning at all

'''

import numpy as np
import tensorflow as tf
from Environment_for_DQN import Environment

# Global Variables
MODEL_PATH_SAVE = "./tmp/model_2.ckpt"
MODEL_PATH_LOAD = "./tmp/model_2.ckpt"

LOGDIR = "./tmp/demo/3"

GRID_SIZE = 5


# Number of nodes at input layer
n_input_nodes = (GRID_SIZE**2)*3

# Number of hidden layer nodes - 3 Hidden layers => 5 layers in total
n_nodes_hl1 = 50
# n_nodes_hl2 = 200
# n_nodes_hl3 = 200

# Number of actions - Up, down, left, right
n_actions = 4

# manipulate the weights of 100 inputs at a time - NOT SURE HOW TO UPDATE WEIGHTS FOR RL
# batch_size = 100

# 10 x 10 x 4 = 400 (input nodes)

# Dont have to have those -> [None, 400] -> values here, 
# but TF will throw an error if you give it something different.
# Otherwise without the values, it wont throw an error.
# So it's just safety incase something of not that shape is given in there

# input
x = tf.placeholder(tf.float32, [1, n_input_nodes], name="Input")

# output
y = tf.placeholder(tf.float32, [1, n_actions], name="Output")

# Q_values2 = createModel(x)

# Error function
# error = tf.reduce_mean(tf.square(Q_values - y))

# Optimizer
# optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(error)

# the next action value vector
# y_prime = tf.reduce_max(y, axis=1)

# action at time t - made a global variable for run() function
# action_t = tf.argmax(y, axis=1)

# x = tf.placeholder(tf.float32, [n_input_nodes, None]) # None means an arbitary size

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

	# hidden_1_layer is simply a python dictionary
	# will create an array (tensor) of your weights - initialized to random values

	# hidden_1_layer = {'weights': tf.Variable(tf.random_uniform([n_input_nodes, n_nodes_hl1], minval=0, maxval=None)),
	# 				  'biases': tf.Variable(tf.random_uniform([n_nodes_hl1], minval=0, maxval=1))}

	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_input_nodes, n_nodes_hl1]), name = "Weights1"),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]), name = "Biases1")}

	# hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
	# 				  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))} 

	# hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
	# 				  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	# output_layer = {'weights': tf.Variable(tf.random_uniform([n_nodes_hl1, n_actions], minval=0, maxval=None)),
					# 'biases': tf.Variable(tf.random_uniform([n_actions], minval=0, maxval=1))}
	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_actions]), name = "Weights2"),
					'biases': tf.Variable(tf.random_normal([n_actions]), name = "Biases2")}

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
	# output = tf.nn.l2_normalize(output)

	return output
	# return tf.transpose(output)

# Train function
def train():

	# Testing
	print("Training the Deep Q-Learning Model")

	# Decide whether or not to render to the screen or not
	RENDER_TO_SCREEN = False

	# True - Load model from modelpath_load; False - Initialise random weights
	USE_SAVED_MODEL_FILE = False 

	# First we need our environment form Environment_for_DQN.py
	# has to have a grid_size of 10 for this current NN
	env = Environment(wrap = False, grid_size = GRID_SIZE, rate = 0, max_time = 20, tail = False)
	
	if RENDER_TO_SCREEN:
		env.prerender()

	# Hyper-parameters
	alpha = 0.5  # Learning rate, i.e. which fraction of the Q values should be updated
	
	gamma = 0.99  # Discount factor, i.e. to which extent the algorithm considers possible future rewards

	epsilon = 0.1  # Probability to choose random action instead of best action

	# Create NN model
	with tf.name_scope('Model'):
		Q_values = createModel(x)

	# Error / Loss function 
	# Not sure why its reduce_mean, it reduces the [1,4] tensor to a scalar of the mean value
	with tf.name_scope('Error'):
		e1 = tf.subtract(y, Q_values)
		e2 = tf.square(e1)
		error = tf.reduce_max(e2)
		# error = tf.square(tf.subtract(y, Q_values))

	# Gradient descent optimizer - minimizes error/loss function
	with tf.name_scope('Optimizer'):
		optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(error)
		# optimizer = tf.train.AdamOptimizer(alpha).minimize(error)

	# The next states action-value [1,4] tensor, reduced to a scalar of the max value
	with tf.name_scope('y_prime_max'):
		y_prime_max = tf.reduce_max(y, axis=1)

	# Action at time t, the index of the max value in the action-value tensor (Made a global variable)

	action_t = tf.argmax(y, axis=1)

	avg_time = 0
	avg_score = 0
	# avg_error = 0

	print_episode = 1000
	total_episodes = 10000

	# Saving model capabilities
	saver = tf.train.Saver()

	# Initialising all variables (weights and biases)
	model = tf.global_variables_initializer()

	# Tensorboard capabilties
	writer = tf.summary.FileWriter("./tmp/demo/2")

	# Session can start running
	with tf.Session() as sess:

		# Restore the model, to keep training
		if USE_SAVED_MODEL_FILE:
			saver.restore(sess, MODEL_PATH_LOAD)
			print("Model restored.")

		sess.run(model)

		writer.add_graph(sess.graph)

		# variables_names = [v.name for v in tf.trainable_variables()]
		# values = sess.run(variables_names)
		# for k, v in zip(variables_names, values):
		#     print("Variable: ", k)
		#     print("Shape: ", v.shape)
		#     print(v)

		# Testing my DQN model with random values
		for episode in range(total_episodes):
			state, info = env.reset()
			done = False

			# Test for an Epsilon linear function - start at 0.9 random, 
			# and for 30% of the episodes, decrease down to 0.1 random
			epsilon = (-0.9 / (0.6*total_episodes)) * episode + 1
			if epsilon < 0.1: 
				epsilon = 0.1

			while not done:
				if RENDER_TO_SCREEN:
					env.render()

				# One Hot representation of the current state
				state_vector = env.state_vector()

				# Retrieve the Q values from the NN in vector form
				Q_vector = sess.run(Q_values, feed_dict={x: state_vector})
				# print("Qvector",Q_vector) # DEBUGGING

				# Deciding one which action to take
				if np.random.rand() <= epsilon:
					action = env.sample()
				else:
					# "action" is the max value of the Q values (output vector of NN)
					action = sess.run(action_t, feed_dict={y:Q_vector})

				# Update environment with by performing action
				new_state, reward, done, info = env.step(action)

				state = new_state

				# Gathering our now current states action-value vector
				new_state_vector = env.state_vector()
				y_prime = sess.run(Q_values, feed_dict={x: new_state_vector})

				# Equation for training
				maxq = sess.run(y_prime_max, feed_dict={y:y_prime})

				Q_vector[:,action] = reward + (gamma * maxq)

				_, e = sess.run([optimizer, error], feed_dict={x:state_vector, y:Q_vector})
				# _ = sess.run(optimizer, feed_dict={x: state_vector, y: Q_vector})
				# e = sess.run(error,feed_dict={x:state_vector, y:Q_vector})
				# sess.run(optimizer)
				
				# DEBUGGING
				# print("action:",action)
				# print("y_prime:", y_prime)
				# print("max q value:", maxq)
				# print("new Q_vector:", Q_vector)
				# print("error tensor:", e)

				if done:
					avg_time += info["time"]
					avg_score += info["score"]
					# avg_error += e

			if episode % print_episode == 0 and episode != 0:
				# print("Episode:", episode, "   Score:", info["score"])
				print("Episode:", episode, "   time:", avg_time/print_episode, "   score:", avg_score/print_episode, "    Error", e)
				# print("error tensor:", e)
				avg_time = 0
				avg_score = 0
				# avg_error = 0

		save_path = saver.save(sess, MODEL_PATH_SAVE)
		print("Model saved in path: %s" % save_path)


# Run the model in the game environment
def run():
	# Testing
	print("Running the Deep Q-Learning Model")

	# Decide whether or not to render to the screen or not
	RENDER_TO_SCREEN = True

	# First we need our environment form Environment_for_DQN.py
	# has to have a grid_size of 10 for this current NN
	env = Environment(wrap = False, grid_size = GRID_SIZE, rate = 100, max_time = 20, tail = False)
	
	if RENDER_TO_SCREEN:
		env.prerender()

	epsilon = 0.005  # Probability to choose random action instead of best action

	# Create NN model
	Q_values = createModel(x)

	action_t = tf.argmax(y, axis=1)

	avg_time = 0
	avg_score = 0
	got_food = 0

	print_episode = 10
	total_episodes = 100

	# Saving model
	saver = tf.train.Saver()

	# Session can start running
	with tf.Session() as sess:

		saver.restore(sess, MODEL_PATH_LOAD)
		print("Model restored.")

		# sess.run(tf.global_variables_initializer())

		# Testing my DQN model with random values
		for episode in range(total_episodes):
			state, info = env.reset()
			done = False

			while not done:
				if RENDER_TO_SCREEN:
					env.render()

				# One Hot representation of the current state
				state_vector = env.state_vector()

				# Retrieve the Q values from the NN in vector form
				Q_vector = sess.run(Q_values, feed_dict={x: state_vector})
				# print(Q_vector) # DEBUGGING

				# Deciding one which action to take
				if np.random.rand() <= epsilon:
					action = env.sample()
				else:
					# action is the max value of the Q values (output vector of NN)
					action = sess.run(action_t, feed_dict={y:Q_vector})
					# action = sess.run(tf.argmax(Q_vector, axis=1))
					# action = np.argmax(Q[env.state_index(state)])

				# Update environment with by performing action
				new_state, reward, done, info = env.step(action)

				# Q[env.state_index(state), action] += alpha * (reward + gamma * np.max(Q[env.state_index(new_state)]) - Q[env.state_index(state), action])

				state = new_state

				if reward == 100:
					got_food += 1

				if done:
					avg_time += info["time"]
					avg_score += info["score"]


			if episode % print_episode == 0 and episode != 0:
				# print("Episode:", episode, "   Score:", info["score"])
				print("Episode:", episode, "   time:", avg_time/print_episode, "   score:", avg_score/print_episode, "    Got food", got_food, "times")
				avg_time = 0
				avg_score = 0
				got_food = 0

# Play the game
def play():
	print("Playing the game")

	env = Environment(wrap = True, grid_size = 5, rate = 100, max_time = 20, tail = False)

	# env.play()

	# env.prerender()
	env.reset()

	print(env.state_vector())



if __name__ == '__main__':
	
	train()
	
	# run()

	# play()