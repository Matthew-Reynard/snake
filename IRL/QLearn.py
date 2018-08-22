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
import pandas as pd
import csv
import sys
import time
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
	env = Environment(wrap = False, grid_size = 10, rate = 0, max_time = 50, obstacles = True)

	if RENDER_TO_SCREEN:
		env.prerender()

	Q = Qmatrix(1, env) # 0 - zeros, 1 - random, 2 - textfile

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
	total_episodes = 10000

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

	env = Environment(wrap = False, grid_size = 10, rate = 80, max_time = 1000, tail = False, obstacles = False)

	if RENDER_TO_SCREEN:
		env.prerender()

	Q = Qmatrix(2, env) # 0 - zeros, 1 - random, 2 - textfile

	# Minimise the overfitting during testing
	epsilon = 0.005

	# Testing for a certain amount of episodes
	for episode in range(10):
		# env.food.seed(1)
		# np.random.seed(1)
		state, info = env.reset()
		done = False


		while not done:
			if RENDER_TO_SCREEN:
				env.render()

			if np.random.rand() <= epsilon:
				action = env.sample()
			else:
				action = np.argmax(Q[env.state_index(state)])

			# print(state)

			new_state, reward, done, info = env.step(action)

			# Q[env.state_index(state), action] += alpha * (reward + gamma * np.max(Q[env.state_index(new_state)]) - Q[env.state_index(state), action])

			state = new_state

		if episode % 1 == 0:
			print("Episode:", episode, "   Score:", info["score"])


# Play the game yourself :)
def play():

	env = Environment(wrap = False, grid_size = 10, rate = 100, tail = True, obstacles = False)

	env.play()

	# env.replay("./Data/log_file.csv")

	# Few random tests

	# env.prerender()
	# env.reset()
	# print(env.state_vector())
	# env.render()


# Inverse RL
def irl_train():

	csv_file_path = "./Data/log_file.csv"

	RENDER_TO_SCREEN = True

	# rate should be 0 when not rendering, else it will lengthen training time unnecessarily
	env = Environment(wrap = False, grid_size = 10, rate = 80, tail = True, obstacles = False)

	if RENDER_TO_SCREEN:
		env.prerender()

	Q = Qmatrix(0, env) # 0 - zeros, 1 - random, 2 - textfile

	alpha = 0.5  # Learning rate, i.e. which fraction of the Q values should be updated
	gamma = 0.99  # Discount factor, i.e. to which extent the algorithm considers possible future rewards
	epsilon = 0.0  # Probability to choose random action instead of best action

	epsilon_function = False
	epsilon_start = 0.5
	epsilon_end = 0.01
	epsilon_percentage = 0.5 # in decimal
	
	avg_time = 0
	avg_score = 0

	episode = 0

	df = pd.read_csv(csv_file_path)
	print(df)

	GAME_OVER = False
	GAME_ON = True

	t = 3

	time_stamps = df["TIME_STAMP"].values

	snake_Xs = df["SNAKE_X"].values
	snake_Ys = df["SNAKE_Y"].values

	food_Xs = df["FOOD_X"].values
	food_Ys = df["FOOD_Y"].values

	my_actions = df["INPUT_BUTTON"].values

	# print(len(time_stamps))
	# print(len(my_actions))

	my_index = 0

	# with open("./Data/log_file.csv", 'r') as f:
	#     self.log_file = csv.reader(f, delimiter=",") #default delimiter is a comma


	#     for row in self.log_file:
	#         print(", ".join(row))

	while GAME_ON:

		# env.food.seed(1)
		# np.random.seed(1)

		# needs to be changed to ignore the SCALE
		state, info = env.irl_reset(snake_Xs[my_index], snake_Ys[my_index], food_Xs[my_index], food_Ys[my_index])
		last_action = 3

		print(state)

		my_index = my_index + 1 #BUG: TODO: FIX MY_INDEX

		if RENDER_TO_SCREEN:
			env.countdown = True
		else:
			env.countdown = False

		while not GAME_OVER:

			try:
				if RENDER_TO_SCREEN:
					env.render()

				action = last_action

				# print(self.time)
				if my_index < len(my_actions):
					print(time_stamps[my_index])
					if env.time == time_stamps[my_index]:
						action = my_actions[my_index]
						last_action = action
						my_index = my_index + 1
						print("action",action)

				if env.countdown:
					text = env.font.render(str(t), True, (255, 255, 255))
					env.display.blit(text,(0,0))
					env.pg.display.update()
					time.sleep(0.5)
					t =  t - 1
					if t == 0:
						t = 3
						env.countdown = False
				else:
					# print(action)
					new_state, reward, GAME_OVER, info = env.step(action, action_space = 4)
					# print(reward)

					if reward > 0:
						env.food.irl_make(food_Xs[my_index], food_Ys[my_index], env.SCALE)
						new_state = env.get_state()

					Q[env.state_index(state), action] += alpha * (reward + gamma * np.max(Q[env.state_index(new_state)]) - Q[env.state_index(state), action])

					state = new_state

		        # For the snake to look like it ate the food, render needs to be last
		        # Next piece of code if very BAD programming
				if GAME_OVER:
					# print("Game Over")
					avg_time += info["time"]
					avg_score += info["score"]
					my_index = my_index + 2
		            # self.render()

			except KeyboardInterrupt as e:
				# Test to see if I can write the Q file during runtime
				np.savetxt(Q_textfile_path_save, Q.astype(np.float), fmt='%f', delimiter = " ")
				print("Saved Q matrix to text file")
				raise e

		while GAME_OVER:

			# print(my_index)
			# print(len(my_actions)-1)

			if my_index >= len(my_actions)-1:
				print("Total Episodes: ", episode)
				env.end()
			# elif time_stamps[my_index] == -1:
			else:
				GAME_OVER = False
				# my_index = my_index + 1
				episode = episode + 1
				np.savetxt(Q_textfile_path_save, Q.astype(np.float), fmt='%f', delimiter = " ")
				avg_time = 0
				avg_score = 0

	print("Total Episodes: ", episode)
	env.end()


if __name__ == '__main__':

	# CHOOSE 1 OF THE 3

	# train() 

	# run()

	# play()

	irl_train()
