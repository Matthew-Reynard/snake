'''
Simple SnakeAI Game with a Convolutional Neural Network (CNN) and Deep Q Network

INFO:
@author: Matthew Reynard
@year: 2018

DESCRIPTION:
This is using the code for my first CNN.
Code from sentdex (YouTube) TensorFlow tutorial - https://www.youtube.com/watch?v=mynJtLhhcXk


TO DO LIST:
- Optimize hyper-parameters
- Optimise the random initialisations of the weights and biases


NOTES:
- Random weights and biases are used for the baseline


BUGS - or just things that I can't get working:
- Saving and restoring model successfully using tf.Saver()
- 

'''

import numpy as np
import tensorflow as tf
from Environment_for_DQN import Environment
import matplotlib.pyplot as plt # not used yet
import time # Used to measure how long training takes

# GLOBAL VARIABLES
# Paths
MODEL_PATH_SAVE = "./tmp/model/model_4.ckpt"
MODEL_PATH_LOAD = "./tmp/model/model_4.ckpt"

# Not text files. Some of these arrays are 4D which can't be neatly written to a text file.
W_conv1_textfile_path_save = "./CNN_variables/W_conv1.npy"
b_conv1_textfile_path_save = "./CNN_variables/b_conv1.npy"

W_conv2_textfile_path_save = "./CNN_variables/W_conv2.npy"
b_conv2_textfile_path_save = "./CNN_variables/b_conv2.npy"

W_fc_textfile_path_save = "./CNN_variables/W_fc.npy"
b_fc_textfile_path_save = "./CNN_variables/b_fc.npy"

W_out_textfile_path_save = "./CNN_variables/W_out.npy"
b_out_textfile_path_save = "./CNN_variables/b_out.npy"

# This is for viewing the model and summaries in Tensorboard
LOGDIR = "./tmp/log/4"

# Parameters
GRID_SIZE = 8
SEED = 1
WRAP = False
TAIL = True

# Number of hidden layers, nodes, channels, etc. 
if TAIL:
	n_input_channels = 3
else:
	n_input_channels = 2

# these still need to be added to the code
n_out_channels_conv1 = 16
n_out_channels_conv2 = 32
n_out_fc = 256

size_filter1 = 3
size_filter2 = 3



# Number of actions - Up, down, left, right
n_actions = 4

# input - shape is included to minimise unseen errors
x = tf.placeholder(tf.float32, [n_input_channels, GRID_SIZE, GRID_SIZE], name="Input")

# output
y = tf.placeholder(tf.float32, [1, n_actions], name="Output")


# 2D convolution
def conv2d(x, W, name = None):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding ="SAME", name = name)


# Max pooling
def maxpool2d(x, name = None):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name = name)


# Input -> Conv layer & max pooling -> Conv layer & max pooling -> Fully connected -> Output (Action-value)
def createDeepModel(data, load_variables = False):

	# Model structure
	if load_variables:
		W_conv1 = np.load(W_conv1_textfile_path_save).astype('float32')
		W_conv2 = np.load(W_conv2_textfile_path_save).astype('float32')
		W_fc = np.load(W_fc_textfile_path_save).astype('float32')
		W_out = np.load(W_out_textfile_path_save).astype('float32')

		b_conv1 = np.load(b_conv1_textfile_path_save).astype('float32')
		b_conv2 = np.load(b_conv2_textfile_path_save).astype('float32')
		b_fc = np.load(b_fc_textfile_path_save).astype('float32')
		b_out = np.load(b_out_textfile_path_save).astype('float32')

		weights = {'W_conv1':tf.Variable(W_conv1, name = 'W_conv1'),
			   	   'W_conv2':tf.Variable(W_conv2, name = 'W_conv2'),
			   	   'W_fc':tf.Variable(W_fc, name = 'W_fc'),
			   	   'W_out':tf.Variable(W_out, name = 'W_out')}

		biases = {'b_conv1':tf.Variable(b_conv1, name = 'b_conv1'),
			   	  'b_conv2':tf.Variable(b_conv2, name = 'b_conv2'),
			   	  'b_fc':tf.Variable(b_fc, name = 'b_fc'),
			   	  'b_out':tf.Variable(b_out, name = 'b_out')}
	
	else:

		
		'''
		Different Initialisations for the weights (the biases can be initialised a constant)

		tf.random_normal([3, 3, n_input_channels, 16], mean=0.0, stddev=1.0, seed=SEED)
		tf.random_uniform([3, 3, n_input_channels, 16], minval=-1, maxval=1, seed=SEED)
		tf.truncated_normal([3, 3, n_input_channels, 16], mean=0, stddev=1.0, seed=SEED)

		use xavier initialiser
		use tf.get_variable()

		'''
		
		weights = {'W_conv1':tf.Variable(tf.random_normal([3, 3, n_input_channels, 16]), name = 'W_conv1'),
			   	   'W_conv2':tf.Variable(tf.random_normal([3, 3, 16, 32]), name = 'W_conv2'),
			   	   'W_fc':tf.Variable(tf.random_normal([2*2*32, 256]), name = 'W_fc'),
			   	   'W_out':tf.Variable(tf.random_normal([256, n_actions]), name = 'W_out')}

		biases = {'b_conv1':tf.Variable(tf.constant(0.1, shape=[16]), name = 'b_conv1'),
			   	  'b_conv2':tf.Variable(tf.constant(0.1, shape=[32]), name = 'b_conv2'),
			   	  'b_fc':tf.Variable(tf.constant(0.1, shape=[256]), name = 'b_fc'),
			   	  'b_out':tf.Variable(tf.constant(0.1, shape=[n_actions]), name = 'b_out')}

	# Model operations
	x = tf.reshape(data, shape=[-1, GRID_SIZE, GRID_SIZE, n_input_channels])

	conv1 = conv2d(x, weights['W_conv1'], name = 'conv1')
	conv1 = maxpool2d(conv1, name = 'max_pool1')

	conv2 = conv2d(conv1, weights['W_conv2'], name = 'conv2')
	conv2 = maxpool2d(conv2, name = 'max_pool2')

	fc = tf.reshape(conv2,[-1, 2*2*32])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

	output = tf.matmul(fc, weights['W_out']) + biases['b_out']
	output = tf.nn.l2_normalize(output)

	return output, weights, biases


# Train Deep Model function
def trainDeepModel(load = False):

	# Used to see how long model takes to train - model needs to be optimized!
	start_time = time.time()

	print("\n ---- Training the Deep Neural Network ----- \n")

	# Decide whether or not to render to the screen or not
	RENDER_TO_SCREEN = False

	# True - Load model from modelpath_load; False - Initialise random weights
	USE_SAVED_MODEL_FILE = False 

	# First we need our environment form Environment_for_DQN.py
	# has to have a grid_size of 10 for this current NN
	env = Environment(wrap = WRAP, grid_size = GRID_SIZE, rate = 0, max_time = 120, tail = TAIL)
	
	if RENDER_TO_SCREEN:
		env.prerender()

	# Hyper-parameters
	alpha = 0.01  # Learning rate, i.e. which fraction of the Q values should be updated
	gamma = 0.99  # Discount factor, i.e. to which extent the algorithm considers possible future rewards
	epsilon = 0.1  # Probability to choose random action instead of best action

	epsilon_function = True
	epsilon_start = 0.1
	epsilon_end = 0.05
	epsilon_percentage = 0.9 # in decimal

	alpha_function = False
	alpha_start = 0.01
	alpha_end = 0.003
	alpha_percentage = 0.9 # in decimal

	# Create NN model
	with tf.name_scope('Model'):
		Q_values, weights, biases  = createDeepModel(x, load_variables = load)

	# Error / Loss function 
	# reduce_max -> it reduces the [1,4] tensor to a scalar of the max value
	with tf.name_scope('Error'):

		# test
		error = tf.losses.mean_squared_error(labels=Q_values, predictions=y)

		# error = tf.reduce_max(tf.sqrt(tf.square(tf.subtract(Q_values, y))), axis=1) # Doesn't work!
		# error = tf.reduce_max(tf.square(tf.subtract(Q_values, y)), axis=1)
		# error = tf.reduce_max(tf.square(Q_values - y), axis=1)
	
	tf.summary.scalar('error', tf.squeeze(error))

	# Gradient descent optimizer - minimizes error/loss function
	with tf.name_scope('Optimizer'):
		optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(error)
		# optimizer = tf.train.AdamOptimizer(alpha).minimize(error)

	# The next states action-value [1,4] tensor, reduced to a scalar of the max value
	with tf.name_scope('Max_y_prime'):
		y_prime_max = tf.reduce_max(y, axis=1)

	# Action at time t, the index of the max value in the action-value tensor (Made a global variable)
	with tf.name_scope('Max_action'):
		action_t = tf.argmax(y, axis=1)

	avg_time = 0
	avg_score = 0
	avg_error = 0

	# error plot
	# errors = []

	print_episode = 1000
	total_episodes = 200000

	# Saving model capabilities
	saver = tf.train.Saver()

	# Initialising all variables (weights and biases)
	init = tf.global_variables_initializer()

	# Adds a summary graph of the error over time
	merged_summary = tf.summary.merge_all()

	# Tensorboard capabilties
	writer = tf.summary.FileWriter(LOGDIR)

	# Session can start running
	with tf.Session() as sess:

		# Restore the model, to keep training
		if USE_SAVED_MODEL_FILE:
			saver.restore(sess, MODEL_PATH_LOAD)
			print("Model restored.")

		# Initialize global variables
		sess.run(init)

		# Tensorboard graph
		writer.add_graph(sess.graph)

		print("\nProgram took {0:.4f} seconds to initialise\n".format(time.time()-start_time))
		start_time = time.time()

		# Testing my DQN model with random values
		for episode in range(total_episodes):
			state, info = env.reset()
			done = False

			# Linear function for alpha
			if alpha_function:
				alpha = (-alpha_start / (alpha_percentage*total_episodes)) * episode + (alpha_start+alpha_end)
				if alpha < alpha_end: 
					alpha = alpha_end

			# Linear function for epsilon
			if epsilon_function:
				epsilon = (-(epsilon_start-epsilon_end)/ (epsilon_percentage*total_episodes)) * episode + (epsilon_start)
				if epsilon < epsilon_end: 
					epsilon = epsilon_end

			while not done:
				if RENDER_TO_SCREEN:
					env.render()

				# One Hot representation of the current state
				state_vector = env.state_vector_3D()

				# Retrieve the Q values from the NN in vector form
				Q_vector = sess.run(Q_values, feed_dict={x: state_vector})
				# print("Qvector", Q_vector) # DEBUGGING

				# Deciding one which action to take
				if np.random.rand() <= epsilon:
					action = env.sample()
				else:
					# "action" is the max value of the Q values (output vector of NN)
					action = sess.run(action_t, feed_dict={y: Q_vector})

				# Update environment with by performing action
				new_state, reward, done, info = env.step(action)

				state = new_state

				# if final state of the episode
				if done:
					Q_vector[:,action] = reward
					# print("Reward:", reward)
				else:
					# Gathering the now current state's action-value vector
					new_state_vector = env.state_vector_3D()
					y_prime = sess.run(Q_values, feed_dict={x: new_state_vector})

					# Equation for training
					maxq = sess.run(y_prime_max, feed_dict={y: y_prime})

					# RL Equation
					Q_vector[:,action] = reward + (gamma * maxq)

				_, e = sess.run([optimizer, error], feed_dict={x: state_vector, y: Q_vector})
				# _ = sess.run(optimizer, feed_dict={x: state_vector, y: Q_vector})
				# e = sess.run(error,feed_dict={x:state_vector, y:Q_vector})
				# sess.run(optimizer)
				
				# DEBUGGING
				# print("action:", action)
				# print("y_prime:", y_prime)
				# print("max q value:", maxq)
				# print("new Q_vector:", Q_vector)
				# print("error tensor:", e)

				# add to the error list, to show the plot at the end of training - RAM OVERLOAD!!!
				# errors.append(e)

				if done:
					avg_time += info["time"]
					avg_score += info["score"]
					avg_error += e

			if (episode % print_episode == 0 and episode != 0) or (episode == total_episodes-1):
				current_time = time.time()-start_time
				print("Ep:", episode, 
					"\tavg t:", avg_time/print_episode, 
					"\tavg score:", avg_score/print_episode, 
					"\tErr {0:.3f}".format(avg_error/print_episode), 
					"\tepsilon {0:.2f}".format(epsilon), 
					"\ttime {0:.0f}m {1:.2f}s".format(current_time/60, current_time%60))
				avg_time = 0
				avg_score = 0
				avg_error = 0

				# Save the model's weights and biases to .npy files (can't save 4D array to text file)
				W_conv1 = np.array(sess.run(weights['W_conv1']))
				W_conv2 = np.array(sess.run(weights['W_conv2']))
				W_fc = np.array(sess.run(weights['W_fc']))
				W_out = np.array(sess.run(weights['W_out']))

				b_conv1 = np.array(sess.run(biases['b_conv1']))
				b_conv2 = np.array(sess.run(biases['b_conv2']))
				b_fc = np.array(sess.run(biases['b_fc']))
				b_out = np.array(sess.run(biases['b_out']))

				np.save(W_conv1_textfile_path_save, W_conv1.astype(np.float32))
				np.save(W_conv2_textfile_path_save, W_conv2.astype(np.float32))
				np.save(W_fc_textfile_path_save, W_fc.astype(np.float32))
				np.save(W_out_textfile_path_save, W_out.astype(np.float32))

				np.save(b_conv1_textfile_path_save, b_conv1.astype(np.float32))
				np.save(b_conv2_textfile_path_save, b_conv2.astype(np.float32))
				np.save(b_fc_textfile_path_save, b_fc.astype(np.float32))
				np.save(b_out_textfile_path_save, b_out.astype(np.float32))

				s = sess.run(merged_summary, feed_dict={x: state_vector, y: Q_vector})
				writer.add_summary(s, episode)

		save_path = saver.save(sess, MODEL_PATH_SAVE)
		print("Model saved in path: %s" % save_path)

	# plt.plot([np.mean(errors[i:i+500]) for i in range(len(errors) - 500)])
	# plt.savefig("./Images/errors.png")
	# plt.show()


# Running the deep model
def runDeepModel():

	# Testing
	print("\n ---- Running the Deep Neural Network ----- \n")

	# Decide whether or not to render to the screen or not
	RENDER_TO_SCREEN = True

	# True - Load model from modelpath_load; False - Initialise random weights
	USE_SAVED_MODEL_FILE = False 

	# First we need our environment form Environment_for_DQN.py
	# has to have a grid_size of 10 for this current NN
	env = Environment(wrap = WRAP, grid_size = GRID_SIZE, rate = 90, max_time = 1200, tail = TAIL)
	
	if RENDER_TO_SCREEN:
		env.prerender()

	# Hyper-parameters
	alpha = 0.01  # Learning rate, i.e. which fraction of the Q values should be updated
	gamma = 0.99  # Discount factor, i.e. to which extent the algorithm considers possible future rewards
	
	epsilon = 0.01  # Probability to choose random action instead of best action

	# Create NN model
	with tf.name_scope('Model'):
		Q_values, weights, biases  = createDeepModel(x, load_variables = True)

	# Error / Loss function 
	# Not sure why its reduce_mean, it reduces the [1,4] tensor to a scalar of the mean value
	with tf.name_scope('Error'):
		# e1 = tf.subtract(y, Q_values)
		# e2 = tf.square(e1)
		# error = tf.reduce_mean(e2, axis=1)

		# test
		error = tf.losses.mean_squared_error(labels=Q_values, predictions=y)

		# error = tf.reduce_max(tf.sqrt(tf.square(tf.subtract(Q_values, y))), axis=1)
		# error = tf.reduce_max(tf.square(tf.subtract(Q_values, y)), axis=1)
		# error = tf.reduce_max(tf.square(Q_values - y), axis=1)

	# Gradient descent optimizer - minimizes error/loss function
	with tf.name_scope('Optimizer'):
		optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(error)
		# optimizer = tf.train.AdamOptimizer(alpha).minimize(error)

	# The next states action-value [1,4] tensor, reduced to a scalar of the max value
	with tf.name_scope('Max_y_prime'):
		y_prime_max = tf.reduce_max(y, axis=1)

	# Action at time t, the index of the max value in the action-value tensor (Made a global variable)
	with tf.name_scope('Max_action'):
		action_t = tf.argmax(y, axis=1)

	avg_time = 0
	avg_score = 0
	avg_error = 0

	print_episode = 1
	total_episodes = 10

	# Saving model capabilities
	saver = tf.train.Saver()

	# Initialising all variables (weights and biases)
	init = tf.global_variables_initializer()

	# Session can start running
	with tf.Session() as sess:

		# Restore the model, to keep training
		if USE_SAVED_MODEL_FILE:
			saver.restore(sess, MODEL_PATH_LOAD)
			print("Model restored.")

		sess.run(init)

		# Testing my DQN model with random values
		for episode in range(total_episodes):
			state, info = env.reset()
			done = False

			while not done:
				if RENDER_TO_SCREEN:
					env.render()

				# One Hot representation of the current state
				state_vector = env.state_vector_3D()

				# Retrieve the Q values from the NN in vector form
				Q_vector = sess.run(Q_values, feed_dict={x: state_vector})
				# print("Qvector",Q_vector) # DEBUGGING

				# Deciding one which action to take
				if np.random.rand() <= epsilon:
					action = env.sample()
				else:
					# "action" is the max value of the Q values (output vector of NN)
					action = sess.run(action_t, feed_dict={y: Q_vector})

				# Update environment with by performing action
				new_state, reward, done, info = env.step(action)

				state = new_state

				if done:
					avg_time += info["time"]
					avg_score += info["score"]

			if episode % print_episode == 0 and episode != 0:
				print("Ep:", episode, "   avg t:", avg_time/print_episode, "   avg score:", avg_score/print_episode)
				avg_time = 0
				avg_score = 0


# Play the game
def play():
	print("\n ----- Playing the game -----\n")

	env = Environment(wrap = WRAP, grid_size = GRID_SIZE, rate = 100, tail = TAIL)

	env.play()

	# env.prerender()
	
	# env.reset()

	# print(env.state_vector_3D())
	
	# env.render()


# Choose the appropriate function to run - Need to find a better more user friendly way to implement this
if __name__ == '__main__':
	
	# --- Deep Neural Network with CNN --- #

	trainDeepModel(load = True)

	# runDeepModel()
	

	# --- Just for fun --- #

	# play()