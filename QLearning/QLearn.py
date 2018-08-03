'''
Simple SnakeAI Game with basic Q Learning

INFO:
@author: Matthew Reynard
@year: 2018

DESCRIPTION:
QLearn.py is a a simple implementation of a Q learning algorithm with a lookup table of Q values
You can change the code in the main() function to either train(), run(), or play() 
- with the latter being just for fun and to explore the world and see whats happening and
what the computer is learning to do

TO DO LIST:
- Comment all the code
- Make a more user friendly method to change between the train, run and play methods
- Add a score
-

'''

import numpy as np 
from SnakeGame import Environment

Q_textfile_path_load = "./Data/Q_test.txt"
Q_textfile_path_save = "./Data/Q_test.txt"

# dimensions: (states, actions)
def Qmatrix(x, env):
	if x == 0:
		Q = np.zeros((env.number_of_states(), env.number_of_actions()))
	elif x == 1:
		np.random.seed(0) # To ensure the results can be recreated
		Q = np.random.rand(env.number_of_states(), env.number_of_actions()) 
	elif x == 2:
		Q = np.loadtxt(Q_textfile_path_load, dtype='float', delimiter=" ")
	return Q


# Training function
def train():

	RENDER_TO_SCREEN = False

	# rate should be 0 when not rendering, else it will lengthen training time unnecessarily
	env = Environment(wrap = False, grid_size = 8, rate = 0, max_time = 50)

	if RENDER_TO_SCREEN:
		env.prerender()

	Q = Qmatrix(0, env) # 0 - zeros, 1 - random, 2 - textfile

	alpha = 0.15  # Learning rate, i.e. which fraction of the Q values should be updated
	gamma = 0.99  # Discount factor, i.e. to which extent the algorithm considers possible future rewards
	epsilon = 0.1  # Probability to choose random action instead of best action

	epsilon_function = True
	epsilon_start = 0.5
	epsilon_end = 0.01
	epsilon_percentage = 0.5 # in decimal

	# Test for an Epsilon linear function
	# y = mx + c
	# y = (0.9 / 20% of total episode)*x + 1
	# if epsilon <= 0.1, make epsilon = 0.1
	
	avg_time = 0
	avg_score = 0

	print_episode = 100
	total_episodes = 1000

	for episode in range(total_episodes):
		# Reset the environment
		state, info = env.reset()
		done = False

		# Epsilon linear function
		if epsilon_function:
			epsilon = (-(epsilon_start-epsilon_end)/ (epsilon_percentage*total_episodes)) * episode + (epsilon_start)
			if epsilon < epsilon_end: 
				epsilon = epsilon_end	

		while not done:

			# Testing with try except in loop
			# Might not be the best implementation of training and ensuring saving Q to a .txt file
			try:
				if RENDER_TO_SCREEN:
					env.render()

				if np.random.rand() <= epsilon:
					action = env.sample()
				else:
					action = np.argmax(Q[env.state_index(state)])

				new_state, reward, done, info = env.step(action)

				Q[env.state_index(state), action] += alpha * (reward + gamma * np.max(Q[env.state_index(new_state)]) - Q[env.state_index(state), action])

				state = new_state

				if done:
					avg_time += info["time"]
					avg_score += info["score"]

			except KeyboardInterrupt as e:
				# Test to see if I can write the Q file during runtime
				np.savetxt(Q_textfile_path_save, Q.astype(np.float), fmt='%f', delimiter = " ")
				print("Saved Q matrix to text file")
				raise e


		if (episode % print_episode == 0 and episode != 0) or (episode == total_episodes-1):
			print("Episode:", episode, "   time:", avg_time/print_episode, "   score:", avg_score/print_episode)
			np.savetxt(Q_textfile_path_save, Q.astype(np.float), fmt='%f', delimiter = " ")
			avg_time = 0
			avg_score = 0

	# This doesn't need to be here
	# np.savetxt(Q_textfile_path_save, Q.astype(np.float), fmt='%f', delimiter = " ")
	print("Simulation finished. \nSaved Q matrix to text file at:", Q_textfile_path_save)


# Testing function
def run():

	RENDER_TO_SCREEN = True

	env = Environment(wrap = False, grid_size = 8, rate = 80, max_time = 100, tail = False)

	if RENDER_TO_SCREEN:
		env.prerender()

	Q = Qmatrix(2, env) # 0 - zeros, 1 - random, 2 - textfile

	# Minimise the overfitting during testing
	epsilon = 0.005

	# Testing for a certain amount of episodes
	for episode in range(10):
		state, info = env.reset()
		done = False

		while not done:
			if RENDER_TO_SCREEN:
				env.render()

			if np.random.rand() <= epsilon:
				action = env.sample()
			else:
				action = np.argmax(Q[env.state_index(state)])

			new_state, reward, done, info = env.step(action)

			# Q[env.state_index(state), action] += alpha * (reward + gamma * np.max(Q[env.state_index(new_state)]) - Q[env.state_index(state), action])

			state = new_state

		if episode % 1 == 0:
			print("Episode:", episode, "   Score:", info["score"])


# Play the game yourself :)
def play():

	env = Environment(wrap = True, grid_size = 10, rate = 100, tail = False)

	env.play()

	# Few random tests

	# env.prerender()
	# env.reset()
	# print(env.state_vector())
	# env.render()


if __name__ == '__main__':

	# CHOOSE 1 OF THE 3

	# train() 

	run()

	# play()
