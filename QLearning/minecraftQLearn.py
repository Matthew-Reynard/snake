import socket
import time
import sys
import numpy as np
# from _thread import *
from SnakeGame import Environment


GRID_SIZE = 8

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
def train(s):

	RENDER_TO_SCREEN = False

	# rate should be 0 when not rendering, else it will lengthen training time unnecessarily
	env = Environment(wrap = False, grid_size = 8, rate = 0, max_time = 50)

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
	total_episodes = 1000

	for episode in range(total_episodes):
		# Reset the environment
		# state, info = env.reset()
		done = False
		first_action = True
		action_count = 0

		# Epsilon linear function
		if epsilon_function:
			epsilon = (-(epsilon_start-epsilon_end)/ (epsilon_percentage*total_episodes)) * episode + (epsilon_start)
			if epsilon < epsilon_end: 
				epsilon = epsilon_end	

		while not done:

			# Testing with try except in loop
			# Might not be the best implementation of training and ensuring saving Q to a .txt file
			try:
				# print("waiting for recv...")
				# data = input("Send (q to Quit): ")
				# s.send(str.encode("p\n"))
				r = s.recv(1024)
				if r != None:
					x = r.decode("utf-8")
					# print(x) #to skip line
					x_cleaned = x[3:-1] #Need to find a better implementation
					a = x_cleaned.split(", ")
					# print("Cleaned msg: ", a) #raw bytes received

					if a[0] != "close":

						if a[0] == "done":
							done = True
							print("\nEpisode done, #", episode)
							
							state = np.zeros(4) # Needs to change to incorporate dead states
							reward = int(a[1])
							# print("reward = ", reward, "\n")

						else:
							state = np.zeros(4)
							for i in range(4):
								state[i] = float(a[i])

							# print("State = ", state)

							reward = int(a[4])
							# print("reward = ", reward, "\n")


							if np.random.rand() <= epsilon:
								action = np.random.randint(0,4)
							else:
								action = np.argmax(Q[state_index(state)])

							# print("Action = ", action)
						
							s.send(str.encode(str(action) + "\n"))

							if first_action:
								action_count = action_count + 1

								if action_count >= 2:
									first_action = False


							# Q[env.state_index(state), action] += alpha * (reward + gamma * np.max(Q[env.state_index(new_state)]) - Q[env.state_index(state), action])

						# TRAINING PART
						if not first_action:

							if reward == 10:
								avg_score = avg_score+1

							Q[state_index(prev_state), prev_action] += alpha * (reward + gamma * np.max(Q[state_index(state)]) - Q[state_index(prev_state), prev_action])


						# save the previous state
						prev_state = state
						prev_action = action

					else:
						s.close()
						print("Socket has been closed")
						connected = False


			except KeyboardInterrupt as e:
				# Test to see if I can write the Q file during runtime
				np.savetxt(Q_textfile_path_save, Q.astype(np.float), fmt='%f', delimiter = " ")
				print("Saved Q matrix to text file")

				s.close()
				print("Socket has been closed")
				raise e


		if (episode % print_episode == 0 and episode != 0) or (episode == total_episodes-1):
			print("Episode:", episode, "   time:", avg_time/print_episode, "   score:", avg_score/print_episode, "   epsilon:", epsilon)
			np.savetxt(Q_textfile_path_save, Q.astype(np.float), fmt='%f', delimiter = " ")
			avg_time = 0
			avg_score = 0

	# This doesn't need to be here
	# np.savetxt(Q_textfile_path_save, Q.astype(np.float), fmt='%f', delimiter = " ")
	print("Simulation finished. \nSaved Q matrix to text file at:", Q_textfile_path_save)


# Testing function
def run(s):

	RENDER_TO_SCREEN = False

	env = Environment(wrap = False, grid_size = 8, rate = 80, max_time = 100, tail = False)

	Q = Qmatrix(2, env) # 0 - zeros, 1 - random, 2 - textfile

	# Minimise the overfitting during testing
	epsilon = 0.005

	# Testing for a certain amount of episodes
	for episode in range(10):
		# state, info = env.reset()
		done = False

		while not done:

			try:
				print("waiting for recv...")
				# data = input("Send (q to Quit): ")
				# s.send(str.encode("p\n"))
				r = s.recv(1024)
				if r != None:
					x = r.decode("utf-8")
					# print(x) #to skip line
					x_cleaned = x[3:-1] #Need to find a better implementation
					a = x_cleaned.split(", ")

					if a[0] != "close":

						if a[0] == "done":
							done = True
							print("Episode done")

						else:
							state = np.zeros(4)
							for i in range(4):
								state[i] = float(a[i])
								state[i] = int(state[i])

							# print(state)

							if np.random.rand() <= epsilon:
								action = np.random.randint(0,4)
							else:
								action = np.argmax(Q[state_index(state)])

							# print("Action = ", action)
						
							s.send(str.encode(str(action) + "\n"))

					else:
						s.close()
						print("Socket has been closed")
						connected = False

			# To force close the connection
			except KeyboardInterrupt as e:
				s.close()
				print("Socket has been closed")
				raise e
				# connected = False
		

		if episode % 1 == 0:
			print("Episode:", episode, "   Score:")


# Play the game yourself :)
def play():

	env = Environment(wrap = True, grid_size = 10, rate = 100, tail = False)

	env.play()


def state_index(state_array):
    return int((GRID_SIZE**3)*state_array[0]+(GRID_SIZE**2)*state_array[1]+(GRID_SIZE**1)*state_array[2]+(GRID_SIZE**0)*state_array[3])


if __name__ == '__main__':

	# AF_INET => IPv4 address, SOCK_STREAM => TCP
	# SOCK_DGRAM => UDP (User Datagram Protocol)
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP connection

	print(s)

	# server = "127.0.0.1"
	server = "localhost"
	host = ""
	# server = "www.matthewreynard.com"
	port = 5555

	# connected = True

	s.connect((server, port))

	iteration = 0

	print("Connected...")

	# train(s) 

	run(s)





	# while connected:

	# 	# time.sleep(1)

	# 	iteration=iteration+1

	# 	try:
	# 		# print("waiting for recv...")
	# 		# data = input("Send (q to Quit): ")
	# 		# s.send(str.encode("p\n"))
	# 		r = s.recv(1024)
	# 		if r != None:
	# 			x = r.decode("utf-8")
	# 			# print(x) #to skip line
	# 			x_cleaned = x[3:-1] #Need to find a better implementation
	# 			a = x_cleaned.split(", ")

	# 			if a[0] != "close":

	# 				state = np.zeros(4)
	# 				for i in range(4):
	# 					state[i] = float(a[i])

	# 				if(iteration%10 == 0):
	# 					print(state, iteration)

	# 				action = np.argmax(Q[state_index(state)])

	# 				# print("Action = ", action)
				
	# 				s.send(str.encode(str(action) + "\n"))

	# 			else:
	# 				s.close()
	# 				print("Socket has been closed")
	# 				connected = False

	# 	# To force close the connection
	# 	except KeyboardInterrupt as e:
	# 		s.close()
	# 		print("Socket has been closed")
	# 		raise e
	# 		connected = False



