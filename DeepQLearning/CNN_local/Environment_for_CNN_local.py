'''

Simple SnakeAI Game 

@author: Matthew Reynard
@year: 2018

Purpose: Use Tensorflow to create a RL environment to train this snake to beat the game, i.e. fill the grid
Start Date: 5 March 2018
Due Date: Before June

DEPENDING ON THE WINDOW SIZE
1200-1 number comes from 800/20 = 40; 600/20 = 30; 40*30 = 1200 grid blocks; subtract one for the "head"

BUGS:
To increase the FPS, from 10 to 120, the player is able to press multiple buttons before the snake is updated, mkaing the nsake turn a full 180.

NOTES:
Action space with a tail can be 3 - forward, left, right
Action space without a tail can be 4 - up, down, left, right

This is because with 4 actions and a tail, it is possible to go backwards and over your tail
This could just end the game (but that wont work well.. i think)

Also, when having 3 actions, the game needs to know what forward means, and at this point, with just
a head and a food, its doesn't 

'''

import numpy as np
import pygame
from snakeAI import Snake
from foodAI import Food
import sys
import math # Only used for infinity game time

# import csv
# import pandas as pd
# import matplotlib.pyplot as plt 

# Trying to use a DQN with TF instead of the normal Q Learning
# import tensorflow as tf

class Environment:

    # Initialise the Game Environment with default values
    def __init__(self, wrap = False, grid_size = 10, rate = 100, max_time = math.inf, tail = False):

        # self.FPS = 120 # NOT USED YET
        self.UPDATE_RATE = rate
        self.SCALE = 20 # scale of the snake body 20x20 pixels
        self.GRID_SIZE = grid_size
        self.LOCAL_GRID_SIZE = 9
        self.ENABLE_WRAP = wrap
        self.ENABLE_TAIL = tail
        
        self.DISPLAY_WIDTH = self.GRID_SIZE * self.SCALE
        self.DISPLAY_HEIGHT = self.GRID_SIZE * self.SCALE

        # Maximum timesteps before an episode is stopped
        self.MAX_TIME_PER_EPISODE = max_time

        # Create and Initialise Snake 
        self.snake = Snake()

        # Create Food 
        self.food = Food()

        self.score = 0
        self.time = 0
        self.state = np.zeros(4)

        self.display = None
        self.bg = None
        self.clock = None

    # If you want to render the game to the screen, you will have to prerender
    # in order to load the textures (images)
    def prerender(self):
        pygame.init()

        self.display = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        pygame.display.set_caption('SnakeAI')
        self.clock = pygame.time.Clock()

        # Creates a visual Snake 
        self.snake.create(pygame)

        # Creates visual Food 
        self.food.create(pygame)

        # Creates the grid background
        self.bg = pygame.image.load("../../Images/Grid20.png").convert()

    # Resets environment
    def reset(self):

        # Reset the score to 0
        self.score = 0

        # Starting at random positions
        self.snake.x = np.random.randint(0,self.GRID_SIZE) * self.SCALE
        self.snake.y = np.random.randint(0,self.GRID_SIZE) * self.SCALE

        # Starting at the same spot
        # self.snake.x = 8 * self.SCALE
        # self.snake.y = 8 * self.SCALE

        # Initialise the movement to the right
        self.snake.dx = 1
        self.snake.dy = 0

        # Update the head position of the snake
        self.snake.box[0] = (self.snake.x, self.snake.y)

        # Create a piece of food that is not within the snake
        self.food.make(self.GRID_SIZE, self.SCALE, self.snake)

        # Reset snakes tail
        self.snake.tail_length = 0
        self.snake.box.clear()
        self.snake.box = [(self.snake.x, self.snake.y)] 

        # Fill the state array with the snake and food coordinates on the grid
        self.state[0] = int(self.snake.x / self.SCALE)
        self.state[1] = int(self.snake.y / self.SCALE)
        self.state[2] = int(self.food.x / self.SCALE)
        self.state[3] = int(self.food.y / self.SCALE)

        # Reset the time
        self.time = 0

        # A dictionary of information that can be useful
        info = {"time":self.time, "score":self.score}

        return self.state, info


    # Renders ONLY the CURRENT state
    def render(self):

        # Mainly to close the window and stop program when it's running
        action = self.controls()

        # Set the window caption to the score
        pygame.display.set_caption("Score: " + str(self.score))

        # Draw the background, the snake and the food
        self.display.blit(self.bg, (0, 0))
        self.food.draw(self.display)
        self.snake.draw(self.display)

        # Update the pygame display
        pygame.display.update()

        # print(clock.get_rawtime())
        # clock.tick(self.FPS) # FPS setting

        # Adds a delay to the the game when rendering so that humans can watch
        pygame.time.delay(self.UPDATE_RATE)

        # Testing
        return action

    # Ending the Game -  This has to be at the end of the code, 
    # as the exit button on the pygame window doesn't work (not implemented yet)
    def end(self):
        pygame.quit()
        quit()
        sys.exit(0) # safe backup  


    # If the snake goes out of the screen bounds, wrap it around
    def wrap(self):
        if self.snake.x > self.DISPLAY_WIDTH - self.SCALE:
            self.snake.x = 0;
        if self.snake.x < 0:
            self.snake.x = self.DISPLAY_WIDTH - self.SCALE;
        if self.snake.y > self.DISPLAY_HEIGHT - self.SCALE:
            self.snake.y = 0;
        if self.snake.y < 0:
            self.snake.y = self.DISPLAY_HEIGHT - self.SCALE;


    # Step through the game, one state at a time.
    # Return the reward, the new_state, whether its reached_food or not, and the time
    def step(self, action, action_space = 4):
        self.time += 1

        # Test
        # self.score += 1

        # If the snake has reached the food
        reached_food = False

        # If the episode is finished - after a certain amount of timesteps or it crashed
        done = False

        # Initialze to -1 for every time step - to find the fastest route (can be a more negative reward)
        # reward = -1
        reward = 0.05

        # Update the position of the snake head and tail
        self.snake.update(self.SCALE, action, action_space)

        if self.ENABLE_WRAP:
            self.wrap()
        else:
            if self.snake.x > self.DISPLAY_WIDTH - self.SCALE:
                reward = -1 # very negative reward, to ensure that it never crashes into the side
                done = True 
            if self.snake.x < 0:
                reward = -1
                done = True
            if self.snake.y > self.DISPLAY_HEIGHT - self.SCALE:
                reward = -1
                done = True
            if self.snake.y < 0:
                reward = -1
                done = True

        # Update the snakes tail positions (from back to front)
        if self.snake.tail_length > 0:
            for i in range(self.snake.tail_length, 0, -1):
                self.snake.box[i] = self.snake.box[i-1]

        # Update the head position of the snake for the draw method
        self.snake.box[0] = (self.snake.x, self.snake.y)

        # The snake can only start crashing into itself after the tail length it greater than 3
        if self.snake.tail_length >= 1:
            # print(snake.tail_length) # DEBUGGING
            for i in range(1, self.snake.tail_length + 1):
                if(self.snake.box[0] == (self.snake.box[i])):
                    # print("Crashed") # DEBUGGING
                    reward = -1
                    done = True

        # Checking if the snake has reached the food
        reached_food = ((self.snake.x, self.snake.y) == (self.food.x, self.food.y)) 

        # Reward: Including the distance between them
        # if reward == 0:
        #     reward = ((self.GRID_SIZE**2) / np.sqrt(((self.snake.x/self.SCALE-self.food.x/self.SCALE)**2 + (self.snake.y/self.SCALE-self.food.y/self.SCALE)**2) + 1)**2)/(self.GRID_SIZE**2)
            # print(reward) 
        
        # If the snake reaches the food, increment score (and increase snake tail length)
        if reached_food:
            self.score += 1 # Increment score

            # CHOOSE 1 OF THE 2 BELOW:

            # Create a piece of food that is not within the snake
            self.food.make(self.GRID_SIZE, self.SCALE, self.snake)
            # Test for one food item at a time
            # done = True 

            # Can't implement tail with Q learning algorithm
            if self.ENABLE_TAIL:
                self.snake.tail_length += 1
                self.snake.box.append((self.snake.x, self.snake.y)) #adds a rectangle variable to snake.box array(list)

            # Reward functions
            reward = 10
            # reward = 100 / (np.sqrt((self.snake.x-self.food.x)**2 + (self.snake.y-self.food.y)**2) + 1) # Including the distance between them
            # reward = 1000 * self.score
            # reward = 1000 / self.time # Including the time in the reward function

        # If the episode takes longer than the max time, it ends
        if self.time == self.MAX_TIME_PER_EPISODE:
            done = True

        # Create the new_state array/list
        new_state = np.zeros(4) # Does this increase programming time dramatically?
        if not done:
            new_state[0] = int(self.snake.x / self.SCALE)
            new_state[1] = int(self.snake.y / self.SCALE)
            new_state[2] = int(self.food.x / self.SCALE)
            new_state[3] = int(self.food.y / self.SCALE)

        # A dictionary of information that can be useful
        info = {"time":self.time, "score":self.score}

        return new_state, reward, done, info

    # Given the state array, return the index of that state as an integer
    # Used for the Qlearning lookup table
    def state_index(self, state_array):
        return int((self.GRID_SIZE**3)*state_array[0]+(self.GRID_SIZE**2)*state_array[1]+(self.GRID_SIZE**1)*state_array[2]+(self.GRID_SIZE**0)*state_array[3])


    # Random action generator
    def sample(self):
        # Can't use this with a tail, else it will have a double chance of doing nothing
        action = np.random.randint(0,4) 
        # action = np.random.randint(0,3)
        return action


    # Set the environment to a particular state
    def set_state(self, state):
        self.state = state


    # Number of states with just the head and food
    def number_of_states(self):
        return (self.GRID_SIZE**2)*((self.GRID_SIZE**2))


    # Number of actions that can be taken
    def number_of_actions(self):

        # 'forward', left, right
        return 3

        # up, down, left, right
        # return 4

    # The state represented as a onehot 1D vector
    def state_vector(self):

        if self.ENABLE_TAIL:
           # (rows, columns)
            state = np.zeros(((self.GRID_SIZE**2), 4))
            
            # Probabily very inefficient - TODO, find a better implementation
            # This is for the HEAD, TAIL, FOOD and EMPTY - [H, T, F, E]
            for i in range(self.GRID_SIZE): # rows
                for j in range(self.GRID_SIZE): # columns
                    if ((self.snake.x/self.SCALE) == j and (self.snake.y/self.SCALE) == i):
                        state[i*self.GRID_SIZE + j] = [1,0,0,0]
                        # print("Snake:", i*self.GRID_SIZE+j)
                    elif ((self.food.x/self.SCALE) == j and (self.food.y/self.SCALE) == i):
                        state[i*self.GRID_SIZE + j] = [0,0,1,0]
                        # print("Food:", i*self.GRID_SIZE+j)
                    # elif ((self.food.x/self.SCALE) == j and (self.food.y/self.SCALE) == i):
                    #     state[i*self.GRID_SIZE+j] = [0,0,1,0]
                        # print("Food:", i*self.GRID_SIZE+j)
                    else:
                        state[i*self.GRID_SIZE + j] = [0,0,0,1]

            # Adding the snakes tail to the state vector
            for i in range(1, self.snake.tail_length + 1):
                state[int((self.snake.box[i][1]/self.SCALE)*self.GRID_SIZE + (self.snake.box[i][0]/self.SCALE))] = [0,1,0,0]

            # Flatten the vector to a 1 dimensional vector for the input layer to the NN
            state = state.flatten()

            state = state.reshape(1,(self.GRID_SIZE**2)*4)

        else:
            # (rows, columns)
            state = np.zeros(((self.GRID_SIZE**2), 3))
            
            # Probabily very inefficient - TODO, find a better implementation
            # This is for the HEAD and the FOOD and EMPTY, need to add a column for a TAIL later [H, T, F, E]
            for i in range(self.GRID_SIZE): # rows
                for j in range(self.GRID_SIZE): # columns
                    if ((self.snake.x/self.SCALE) == j and (self.snake.y/self.SCALE) == i):
                        state[i*self.GRID_SIZE+j] = [1,0,0]
                        # print("Snake:", i*self.GRID_SIZE+j)
                    elif ((self.food.x/self.SCALE) == j and (self.food.y/self.SCALE) == i):
                        state[i*self.GRID_SIZE+j] = [0,1,0]
                        # print("Food:", i*self.GRID_SIZE+j)
                    else:
                        state[i*self.GRID_SIZE+j] = [0,0,1]

            # Flatten the vector to a 1 dimensional vector for the input layer to the NN
            state = state.flatten()

            state = state.reshape(1,(self.GRID_SIZE**2)*3)

        # state = np.transpose(state)

        return state

    def state_vector_3D(self):

        if self.ENABLE_TAIL:
            state = np.zeros((3, self.GRID_SIZE, self.GRID_SIZE))

            state[0, int(self.snake.y/self.SCALE), int(self.snake.x/self.SCALE)] = 1

            state[1, int(self.food.y/self.SCALE), int(self.food.x/self.SCALE)] = 1

            for i in range(1, self.snake.tail_length + 1):
                state[2, int(self.snake.box[i][1]/self.SCALE), int(self.snake.box[i][0]/self.SCALE)] = 1

        else:
            state = np.zeros((2, self.GRID_SIZE, self.GRID_SIZE))

            state[0, int(self.snake.y/self.SCALE), int(self.snake.x/self.SCALE)] = 1

            state[1, int(self.food.y/self.SCALE), int(self.food.x/self.SCALE)] = 1

        return state

    def local_state_vector_3D(self): #(3,9,9)

        sx = int(self.snake.x/self.SCALE)
        sy = int(self.snake.y/self.SCALE)

        if self.ENABLE_TAIL:
            state = np.zeros((3, self.LOCAL_GRID_SIZE, self.LOCAL_GRID_SIZE))

            # Agent
            state[0, 4, 4] = 1

            # Food
            x_prime = 4+int(self.food.x/self.SCALE)-int(self.snake.x/self.SCALE)
            y_prime = 4+int(self.food.y/self.SCALE)-int(self.snake.y/self.SCALE)

            if x_prime < 9 and x_prime >= 0 and y_prime < 9 and y_prime >= 0:
                state[1, y_prime, x_prime] = 1


            # Obstacles

            # Tail
            for i in range(1, self.snake.tail_length + 1):
                
                x_prime_tail = 4+int(self.snake.box[i][0]/self.SCALE)-int(self.snake.x/self.SCALE)
                y_prime_tail = 4+int(self.snake.box[i][1]/self.SCALE)-int(self.snake.y/self.SCALE)

                if x_prime_tail < 9 and x_prime_tail >= 0 and y_prime_tail < 9 and y_prime_tail >= 0:
                    state[2, y_prime_tail, x_prime_tail] = 1

            # Walls
            for j in range(0, self.LOCAL_GRID_SIZE):
                for i in range(0, self.LOCAL_GRID_SIZE):

                    x_prime_wall = 4-sx
                    y_prime_wall = 4-sy

                    if i < x_prime_wall or j < y_prime_wall:
                        state[2, j, i] = 1

                    x_prime_wall = 4+(9-sx)-1
                    y_prime_wall = 4+(9-sy)-1

                    if i > x_prime_wall or j > y_prime_wall:
                        state[2, j, i] = 1


        else:
            state = np.zeros((2, self.GRID_SIZE, self.GRID_SIZE))

            state[0, int(self.snake.y/self.SCALE), int(self.snake.x/self.SCALE)] = 1

            state[1, int(self.food.y/self.SCALE), int(self.food.x/self.SCALE)] = 1

        return state


    # Defines all the players controls during the game
    def controls(self):

        GAME_OVER = False # NOT IMPLEMENTED YET

        action = 0 # Do nothing as default

        for event in pygame.event.get():
            # print(event) # DEBUGGING

            if event.type == pygame.QUIT:
                self.end()

            if event.type == pygame.KEYDOWN:
                # print(event.key) #DEBUGGING

                # In order to stop training and still save the Q txt file
                if (event.key == pygame.K_q):
                    self.end()

                # Moving left
                if (event.key == pygame.K_LEFT or event.key == pygame.K_a) and GAME_OVER == False:
                    
                    # Moving up or down
                    if self.snake.dy != 0:
                        # moving down
                        if self.snake.dy == 1:
                            action = 2 # turn right
                        # moving up
                        elif self.snake.dy == -1:
                            action = 1 # turn left

                    # Moving left or right
                    elif self.snake.dx != 0:
                        action = 0

                    # log_file.writerow([pygame.time.get_ticks(), str(snake.x), str(snake.y), "A"])

                # Moving right
                elif (event.key == pygame.K_RIGHT or event.key == pygame.K_d) and GAME_OVER == False:
                    
                    # Moving up or down
                    if self.snake.dy != 0:
                        # moving down
                        if self.snake.dy == 1:
                            action = 1 # turn left
                        # moving up
                        elif self.snake.dy == -1:
                            action = 2 # turn right

                    # Moving left or right
                    elif self.snake.dx != 0:
                        action = 0

                    # log_file.writerow([pygame.time.get_ticks(), str(snake.x), str(snake.y), "D"])


                # Moving up
                elif (event.key == pygame.K_UP or event.key == pygame.K_w) and GAME_OVER == False:
                    
                    # Moving up or down
                    if self.snake.dy != 0:
                        action = 0

                    # Moving left or right
                    elif self.snake.dx != 0:
                        # moving left
                        if self.snake.dx == -1:
                            action = 2 # turn right
                        # moving right
                        elif self.snake.dx == 1:
                            action = 1 # turn left

                    # log_file.writerow([pygame.time.get_ticks(), str(snake.x), str(snake.y), "W"])


                # Moving down
                elif (event.key == pygame.K_DOWN or event.key == pygame.K_s) and GAME_OVER == False:
                    
                    # Moving up or down
                    if self.snake.dy != 0:
                        action = 0

                    # Moving left or right
                    elif self.snake.dx != 0:
                        # moving left
                        if self.snake.dx == -1:
                            action = 1 # turn left
                        # moving right
                        elif self.snake.dx == 1:
                            action = 2 # turn right

                    # log_file.writerow([pygame.time.get_ticks(), str(snake.x), str(snake.y), "S"])

        return action

    # Lets you simply play the game
    # Need to implement a log file to record the game, to attempt a IRL Algorithm
    def play(self):

        GAME_OVER = False

        self.prerender()

        self.reset()

        while not GAME_OVER:

            # print(self.local_state_vector_3D()) # DEBUGGING

            action = self.render()

            # When the snake touches the food, game ends
            # action_space has to be 3 for the players controls, 
            # because they know that the snake can't go backwards
            s, r, GAME_OVER, i = self.step(action, action_space = 3)

            # For the snake to look like it ate the food, render needs to be last
            # Next piece of code if very BAD programming
            if GAME_OVER:
                print("Game Over")
                # self.render()

        self.end()

# If I run this file by accident :P
if __name__ == "__main__":

    print("This file does not have a main method")
