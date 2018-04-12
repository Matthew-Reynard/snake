import numpy as np 
from SnakeGame import Environment

# dimensions: (states, actions)
def Qmatrix(x, env):
	if x == 0:
		Q = np.zeros((env.number_of_states(), env.number_of_actions()))
	elif x == 1:
		Q = np.random.rand(height*width, no_of_actions) 
	elif x == 2:
		Q = np.loadtxt("Q3.txt", dtype='float', delimiter=" ")
	return Q

def run():

	RENDER_TO_SCREEN = True

	env = Environment(True, rate = 100, render = RENDER_TO_SCREEN)

	if RENDER_TO_SCREEN:
		env.prerender()

	# env.reset()

	Q = Qmatrix(2, env) # 0 - zeros, 1 - random, 2 - textfile

	epsilon = 0.001

	for epoch in range(100):
		state, mytime = env.reset()
		done = False

		if epoch % 10 == 0:
			print(epoch)

		while not done:
			if RENDER_TO_SCREEN:
				env.render()

			if np.random.rand() <= epsilon:
				action = env.sample()
			else:
				action = np.argmax(Q[env.state_index(state)])

			new_state, reward, done, myTime = env.step(action)

			# Q[env.state_index(state), action] += alpha * (reward + gamma * np.max(Q[env.state_index(new_state)]) - Q[env.state_index(state), action])

			state = new_state

			if myTime == 50:
				done = True


def main():

	RENDER_TO_SCREEN = False

	env = Environment(True, rate = 10, render = RENDER_TO_SCREEN)

	if RENDER_TO_SCREEN:
		env.prerender()

	# env.reset()

	Q = Qmatrix(2, env) # 0 - zeros, 1 - random, 2 - textfile

	alpha = 0.2  # learning rate, i.e. which fraction of the Q values should be updated
	gamma = 0.99  # discount factor, i.e. to which extent the algorithm considers possible future rewards
	epsilon = 0.1  # probability to choose random action instead of best action
	
	avg_time = 0

	print_episode = 10000

	for episode in range(10000000):
		state, mytime = env.reset()
		done = False

		if episode % print_episode == 0:
			print(episode,"Avg Time:",avg_time/print_episode)
			avg_time = 0

		while not done:
			if RENDER_TO_SCREEN:
				env.render()

			if np.random.rand() <= epsilon:
				action = env.sample()
			else:
				action = np.argmax(Q[env.state_index(state)])

			new_state, reward, done, myTime = env.step(action)

			Q[env.state_index(state), action] += alpha * (reward + gamma * np.max(Q[env.state_index(new_state)]) - Q[env.state_index(state), action])

			state = new_state

			if myTime == 50:
				done = True

			if done:
				avg_time += myTime

	np.savetxt("Q3.txt", Q.astype(np.float), fmt='%f', delimiter = " ")


if __name__ == '__main__':
	# main()
	run()

