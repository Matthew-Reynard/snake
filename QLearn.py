import numpy as np 
import matplotlib.pyplot as plt 

width = 30
height = 20
no_of_actions = 5

Q = np.random.rand(height*width, no_of_actions) # dimensions: (states, actions)

# Q.size = 600 x 4

alpha = 0.3  # learning rate, i.e. which fraction of the Q values should be updated
gamma = 0.9  # discount factor, i.e. to which extent the algorithm considers possible future rewards
epsilon = 0.2  # probability to choose random action instead of best action

#Dont understand
def state_index(state):
    return state[0] * state[1]

history = []

verbose = False # no printing to the screen

game.p()

for epoch in range(100):
	state = game
