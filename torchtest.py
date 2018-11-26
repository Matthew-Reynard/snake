import os
import numpy as np
from model import DeepQNetwork, Agent
from Snake_Environment import Environment

if __name__ == '__main__':

	TRAIN = True
	RENDER = False
	start_eps = (0.5 if TRAIN else 0.05)

	GRID_SIZE = 5
	LOCAL_GRID_SIZE = 9 # Has to be an odd number (I think...)
	SEED = 1
	WRAP = False
	TAIL = False
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
					  multiplier_count = 0,
					  action_space = 5,
					  map_path = MAP_PATH)

	brain = Agent(gamma = 0.99, epsilon = start_eps, alpha = 0.001, maxMemorySize = 5000, replace = None)

	# env.play()

	if RENDER: env.prerender()

	while brain.memCntr < brain.memSize:
		obs, _ = env.reset()
		observation = env.state_vector_3D()
		done = False
		while not done:
			action = env.sample_action()
			observation_, reward, done, info = env.step(action)
			observation_ = env.state_vector_3D()
			if done:
				reward = -1
			brain.storeTransition(observation, action, reward, observation_)

			observation = observation_
	print("Done initialising memory")

	scores = []
	epsHistory = []
	numGames = 100000
	batch_size = 16

	avg_score = 0
	avg_loss = 0

	# try:
	# 	brain.load_model("./Models/Torch/my_model.pth")
	# except Exception:
	# 	print('Could not load model')
	# 	quit()


	for i in range(numGames):
		epsHistory.append(brain.EPSILON)
		done = False
		obs, _ = env.reset()
		observation = env.state_vector_3D()
		score = 0
		lastAction = 0

		while not done:
			action = brain.chooseAction(observation)

			observation_, reward, done, info = env.step(action)
			
			observation_ = env.state_vector_3D()
			score += reward

			brain.storeTransition(observation, action, reward, observation_)

			observation = observation_
			if TRAIN: loss = brain.learn(batch_size)
			lastAction = action
			if RENDER: env.render()

		avg_score += info["score"]
		if TRAIN: avg_loss += loss.item()


		if i%100 == 0 and not i==0 or i == numGames-1:
			print("Game", i, 
				"\tepsilon: %.4f" %brain.EPSILON,
				"\tavg score", avg_score/100,
				"avg loss:", avg_loss/100)
			brain.save_model("./Models/Torch2/my_model{}.pth".format(i))
			avg_loss = 0
			avg_score = 0

		scores.append(score)
		# print("score:", score)

	brain.save_model("./Models/Torch2/my_model.pth")

