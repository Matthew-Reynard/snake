import os
import numpy as np
from model import DeepQNetwork, Agent
from Snake_Environment import Environment

if __name__ == '__main__':

	GRID_SIZE = 6
	LOCAL_GRID_SIZE = 9 # Has to be an odd number (I think...)
	SEED = 1
	WRAP = False
	TAIL = True
	FOOD_COUNT = 1
	OBSTACLE_COUNT = 0
	# MAP_PATH = "./Maps/Grid{}/map3.txt".format(GRID_SIZE)
	MAP_PATH = None
	
	env = Environment(wrap = WRAP, 
					  grid_size = GRID_SIZE, 
					  rate = 100, 
					  tail = TAIL, 
					  food_count = FOOD_COUNT,
					  obstacle_count = OBSTACLE_COUNT,
					  action_space = 3,
					  map_path = MAP_PATH)

	brain = Agent(gamma = 0.99, epsilon = 1.0, alpha = 0.001, maxMemorySize = 5000, replace = None)

	# env.play()

	# env.prerender()

	while brain.memCntr < brain.memSize:
		obs, _ = env.reset()
		observation = env.local_state_vector_3D()
		done = False
		while not done:
			action = env.sample_action()
			observation_, reward, done, info = env.step(action)
			observation_ = env.local_state_vector_3D()
			if done:
				reward = -100
			brain.storeTransition(observation, action, reward, observation_)

			observation = observation_
	print("Done initialising memory")

	scores = []
	epsHistory = []
	numGames = 10000
	batch_size = 16

	avg_score = 0
	avg_loss = 0


	try:
		brain.load_model("../results/my_model.pth")
	except Exception:
		print('Could not load model')
		quit()


	for i in range(numGames):
		epsHistory.append(brain.EPSILON)
		done = False
		obs, _ = env.reset()
		observation = env.local_state_vector_3D()
		score = 0
		lastAction = 0

		while not done:
			action = brain.chooseAction(observation)

			observation_, reward, done, info = env.step(action)
			observation_ = env.local_state_vector_3D()
			score += reward

			# if done:
				# reward = -100

			brain.storeTransition(observation, action, reward, observation_)

			observation = observation_
			loss = brain.learn(batch_size)
			lastAction = action
			# env.render()

		avg_score += info["score"]
		# print(loss)


		if i%100 == 0 and not i==0 or i == numGames-1:
			print("Game", i+1, "\tepsilon: %.4f" %brain.EPSILON,"\tavg score", avg_score/100)
			brain.save_model("./results/my_model{}.pth".format(i+1))
			print("avg loss:", avg_loss/100)
			avg_loss = 0
			avg_score = 0

		scores.append(score)
		# print("score:", score)

	brain.save_model("./results/my_model.pth")

