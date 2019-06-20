'''
Simple SnakeAI Game with basic Q Learning lookup table

INFO:
@author: Matthew Reynard
@year: 2018

DESCRIPTION:
QLearn.py is a simple implementation of a Q learning algorithm with a lookup table of Q values
You can change the code in the main() function to either train(), run(), or play() 
- with the latter being just for fun and to explore the world and see whats happening and
what the computer is learning to do

TO DO LIST:
- Comment all the code
- Make a more user friendly method to change between the train, run and play methods
- Add a score

'''

import numpy as np 
from Snake_Environment import Environment

'''
Takes information from a library, from Snake_Environment
'''

Q_textfile_path_load = "./Data/QLearning/Q_test.txt"
Q_textfile_path_save = "./Data/QLearning/Q_test.txt"
'''
Saving data and loading data I think?
'''
GRID_SIZE = 6


# Dimensions: (states, actions)
def Qmatrix(x, env):
	'''
	STarts a function called Qmatrix
	'''
	if x == 0:
		Q = np.zeros((env.number_of_states(), env.number_of_actions()))
	elif x == 1:
		np.random.seed(0) # To ensure the results can be recreated
		Q = np.random.rand(env.number_of_states(), env.number_of_actions()) 
	elif x == 2:
		Q = np.loadtxt(Q_textfile_path_load, dtype='float', delimiter=" ")
	return Q
	'''
	Starts a function: if x is 0 then it has the number of states ad the number of actions in the environment. If x is 1 it randomly generates 0 to then allow the results to happen again. If x is 2 then it loads data from the Q_textfile_path_load above it then saves the data type as a float. It then retruns Q
	'''


# Training function
def train():
	'''
	Starts a function called Train
	'''

	RENDER_TO_SCREEN = False
	# RENDER_TO_SCREEN = True

	# Setting up the environment
	env = Environment(wrap = False, 
					  grid_size = GRID_SIZE, 
					  rate = 80, 
					  max_time = 100,
					  tail = False, 
					  food_count = 1, 
					  obstacle_count = 0, 
					  multiplier_count = 0, 
					  map_path = None,
					  action_space = 5) 
	'''
	Sets the state of environemnt to equal the above grid size given, sets the speed the snake moves, the max time it runs for, if there is a tail or not, the amount of food spawned, the amount of obstacles spaned, if  there is a specific path to be taken.
	'''

	if RENDER_TO_SCREEN:
		env.prerender()

	Q = Qmatrix(1, env) # 0 - zeros, 1 - random, 2 - textfile

	alpha = 0.15  # Learning rate, i.e. which fraction of the Q values should be updated
	gamma = 0.99  # Discount factor, i.e. to which extent the algorithm considers possible future rewards
	epsilon = 0.1  # Probability to choose random action instead of best action

	'''
	Sets variables with values because of reasons stated above
	'''

	epsilon_function = True
	epsilon_start = 0.8
	epsilon_end = 0.05
	epsilon_percentage = 0.6 # in decimal
	
	avg_time = 0
	avg_score = 0

	print_episode = 1000
	total_episodes = 10000

	'''
	Sets values to variables
	'''

	for episode in range(total_episodes): # Takes an episode and if it is in range of the total episodes it proceeds
		# Reset the environment
		state, info = env.reset() # Resets environment state
		done = False 

		# Epsilon linear function
		if epsilon_function:
			epsilon = (-(epsilon_start-epsilon_end)/ (epsilon_percentage*total_episodes)) * episode + (epsilon_start) # minuses ep_start from ep_end dived by ep_percentage times by total_episodes then timesed by epsidoe the added to ep_start
			if epsilon < epsilon_end: 
				epsilon = epsilon_end	
				#checks to see if ep is less than ep_end and if it is it makes ep = to ep_end

		while not done:

			# If cancelled, Q lookup table is still saved
			try:
				if RENDER_TO_SCREEN:
					env.render()

				if np.random.rand() <= epsilon:
					action = env.sample_action() 
				else:
					action = np.argmax(Q[env.state_index(state)]) 
#checks if new random no. is less than or = to ep, else it does a new action.
				new_state, reward, done, info = env.step(action)

				# print(state)

				Q[env.state_index(state), action] += alpha * (reward + gamma * np.max(Q[env.state_index(new_state)]) - Q[env.state_index(state), action])

				state = new_state #assigns new value to state

				if done:
					avg_time += info["time"]
					avg_score += info["score"] # adds time nd score to the score counter and prints it out

			except KeyboardInterrupt as e:
				# Test to see if I can write the Q file during runtime
				np.savetxt(Q_textfile_path_save, Q.astype(np.float), fmt='%f', delimiter = " ")
				print("Saved Q matrix to text file")
				raise e
				#try and except work togther, this ecept is anyresponse that doesnt fall into the try section.


		if (episode % print_episode == 0 and episode != 0) or (episode == total_episodes-1): #tests to see if ep mod print_ep = 0 or if ep == total_ep-1, then if it does it proceeds
			print("Episode:", episode, 
				"\tavg t: {0:.3f}".format(avg_time/print_episode), 
				"\tavg score: {0:.3f}".format(avg_score/print_episode), 
				"\tepsilon {0:.3f}".format(epsilon)) #prints out episodes, score, time
			np.savetxt(Q_textfile_path_save, Q.astype(np.float), fmt='%f', delimiter = " ")
			avg_time = 0
			avg_score = 0 #resets time and score to 0

	# This doesn't need to be here
	# np.savetxt(Q_textfile_path_save, Q.astype(np.float), fmt='%f', delimiter = " ")
	print("Simulation finished. \nSaved Q matrix to text file at:", Q_textfile_path_save)


# Testing function
def run():
	'''

	'''

	RENDER_TO_SCREEN = True

	# Setting up the environment
	env = Environment(wrap = False, 
					  grid_size = GRID_SIZE, 
					  rate = 80, 
					  max_time = 100,
					  tail = False, 
					  food_count = 1, 
					  obstacle_count = 0, 
					  multiplier_count = 0, 
					  map_path = None,
					  action_space = 5) #sets up the environment

	if RENDER_TO_SCREEN:
		env.prerender()

	Q = Qmatrix(2, env) # 0 - zeros, 1 - random, 2 - textfile

	# Minimise the overfitting during testing
	epsilon = 0.005

	# Testing for a certain amount of episodes
	for episode in range(10):
		state, info = env.reset()
		done = False #if epsidoe is in the range of 10 it resets the environment unfo

		while not done:
			if RENDER_TO_SCREEN:
				env.render()

			if np.random.rand() <= epsilon: 
				action = env.sample_action() #if a random numpy is less than or = to epsilon then it does an action
			else:
				action = np.argmax(Q[env.state_index(state)]) #else it does a different action

			new_state, reward, done, info = env.step(action)

			# Q[env.state_index(state), action] += alpha * (reward + gamma * np.max(Q[env.state_index(new_state)]) - Q[env.state_index(state), action])

			state = new_state #gives state the value of new_state

		if episode % 1 == 0:
			print("Episode:", episode, 
				"\tScore:", info["score"],
				"\tTime:", info["time"]) #prints out episode, score and time


# Play the game yourself :)
def play():

	env = Environment(wrap = True, 
					  grid_size = 10, 
					  rate = 100, 
					  tail = True, 
					  action_space = 3,
					  food_count = 1,
					  obstacle_count = 0,
					  multiplier_count = 0, 
					  map_path = None) #sets up envirnnment

	env.play() #allows user to play


if __name__ == '__main__':

	# train() 

	# run()

	play()

	#switch to play, train or run
