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
from obstacleAI import Obstacle
import sys
import math # Only used for infinity game time
import time

import csv
import pandas as pd
# import matplotlib.pyplot as plt 

# Trying to use a DQN with TF instead of the normal Q Learning
# import tensorflow as tf

class Environment:

    # Initialise the Game Environment with default values
    def __init__(self, wrap = False, grid_size = 10, rate = 100, max_time = math.inf, tail = False, obstacles = True):

        # self.FPS = 120 # NOT USED YET
        self.UPDATE_RATE = rate
        self.SCALE = 20 # scale of the snake body 20x20 pixels
        self.GRID_SIZE = grid_size
        self.ENABLE_WRAP = wrap
        self.ENABLE_TAIL = tail
        self.ENABLE_OBSTACLES = obstacles
        
        self.DISPLAY_WIDTH = self.GRID_SIZE * self.SCALE
        self.DISPLAY_HEIGHT = self.GRID_SIZE * self.SCALE

        # Maximum timesteps before an episode is stopped
        self.MAX_TIME_PER_EPISODE = max_time

        # Create and Initialise Snake 
        self.snake = Snake()

        # Create Food 
        self.food = Food()

        # Create Obstacles
        # self.no_of_obstacles = 5
        # self.obstacle = [Obstacle() for i in range(self.no_of_obstacles)]

        if self.ENABLE_OBSTACLES:
            self.obstacle = Obstacle(20)
            self.obstacle.make(self.GRID_SIZE, self.SCALE)
        else:
            self.obstacle = None

        self.score = 0
        self.time = 0
        self.state = np.zeros(4)

        self.display = None
        self.font = None
        self.bg = None
        self.clock = None

        self.log_file = None
        self.countdown = True

        self.pg = None

    # If you want to render the game to the screen, you will have to prerender
    # in order to load the textures (images)
    def prerender(self):
        pygame.init()

        self.pg = pygame

        self.display = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        pygame.display.set_caption('SnakeAI')
        self.clock = pygame.time.Clock()

        pygame.font.init()
        self.font = pygame.font.SysFont('Comic Sans MS', 30)

        # Creates a visual Snake 
        self.snake.create(pygame)

        # Creates visual Food 
        self.food.create(pygame)

        # Creates visual Obstacles
        # if self.ENABLE_OBSTACLES:
        #     for i in range(self.no_of_obstacles):
        #         self.obstacle[i].create(pygame)

        if self.ENABLE_OBSTACLES:
            self.obstacle.create(pygame)

        # if self.ENABLE_OBSTACLES:
        #     self.obstacle.make(self.GRID_SIZE, self.SCALE)

        # Creates the grid background
        self.bg = pygame.image.load("./Images/Grid8_with_bounds.png").convert()

    # Reset game
    def reset(self):

        # Reset the score to 0
        self.score = 0

        # if self.ENABLE_OBSTACLES:
        #     self.obstacle.make(self.GRID_SIZE, self.SCALE)

        # Starting at random positions
        # self.snake.x = np.random.randint(0,self.GRID_SIZE) * self.SCALE
        # self.snake.y = np.random.randint(0,self.GRID_SIZE) * self.SCALE
        if self.ENABLE_OBSTACLES:
            made = False
            while not made:
                self.snake.x = np.random.randint(1,self.GRID_SIZE-1) * self.SCALE
                self.snake.y = np.random.randint(1,self.GRID_SIZE-1) * self.SCALE

                if self.ENABLE_OBSTACLES:
                    for i in range(0, self.obstacle.array_length):
                        if (self.snake.x, self.snake.y) == (self.obstacle.array[i][0], self.obstacle.array[i][1]):
                            made = False
                        else:
                            made = True
        else:
            self.snake.x = np.random.randint(1,self.GRID_SIZE-1) * self.SCALE
            self.snake.y = np.random.randint(1,self.GRID_SIZE-1) * self.SCALE

        # Starting at the same spot
        # self.snake.x = 2 * self.SCALE
        # self.snake.y = 2 * self.SCALE

        # Initialise the movement to the right
        self.snake.dx = 1
        self.snake.dy = 0

        # Update the head position of the snake
        self.snake.box[0] = (self.snake.x, self.snake.y)

        # Create a piece of food that is not within the snake
        # if self.ENABLE_OBSTACLES:
        #     for i in range(self.no_of_obstacles):
        #         self.obstacle[i].make(self.GRID_SIZE, self.SCALE, self.snake)

        # Create a piece of food that is not within the snake
        self.food.make(self.GRID_SIZE, self.SCALE, self.snake, self.ENABLE_OBSTACLES, self.obstacle)

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


    # Reset game
    def irl_reset(self, snake_x, snake_y, food_x, food_y):

        # Reset the score to 0
        self.score = 0

        # if self.ENABLE_OBSTACLES:
        #     self.obstacle.make(self.GRID_SIZE, self.SCALE)

        # Starting at random positions
        # self.snake.x = np.random.randint(0,self.GRID_SIZE) * self.SCALE
        # self.snake.y = np.random.randint(0,self.GRID_SIZE) * self.SCALE
        if self.ENABLE_OBSTACLES:
            made = False
            while not made:
                self.snake.x = np.random.randint(1,self.GRID_SIZE-1) * self.SCALE
                self.snake.y = np.random.randint(1,self.GRID_SIZE-1) * self.SCALE

                if self.ENABLE_OBSTACLES:
                    for i in range(0, self.obstacle.array_length):
                        if (self.snake.x, self.snake.y) == (self.obstacle.array[i][0], self.obstacle.array[i][1]):
                            made = False
                        else:
                            made = True
        else:
            self.snake.x = snake_x * self.SCALE
            self.snake.y = snake_y * self.SCALE

        # Starting at the same spot
        # self.snake.x = 2 * self.SCALE
        # self.snake.y = 2 * self.SCALE

        # Initialise the movement to the right
        self.snake.dx = 1
        self.snake.dy = 0

        # Update the head position of the snake
        self.snake.box[0] = (self.snake.x, self.snake.y)

        # Create a piece of food that is not within the snake
        # if self.ENABLE_OBSTACLES:
        #     for i in range(self.no_of_obstacles):
        #         self.obstacle[i].make(self.GRID_SIZE, self.SCALE, self.snake)

        # Create a piece of food that is not within the snake
        self.food.irl_make(food_x, food_y, self.SCALE)

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

        if self.ENABLE_OBSTACLES:
            # for i in range(self.no_of_obstacles):
            #     self.obstacle[i].draw(self.display)
            self.obstacle.draw(self.display)

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

        hit_obstacle = False

        # If the episode is finished - after a certain amount of timesteps or it crashed
        done = False

        # Initialze to -1 for every time step - to find the fastest route (can be a more negative reward)
        reward = -1

        # Update the position of the snake head and tail
        self.snake.update(self.SCALE, action, action_space)

        if self.ENABLE_WRAP:
            self.wrap()
        else:
            if self.snake.x > self.DISPLAY_WIDTH - (self.SCALE*2):
                reward = -10 # very negative reward, to ensure that it never crashes into the side
                done = True 
            if self.snake.x < (self.SCALE*1):
                reward = -10
                done = True
            if self.snake.y > self.DISPLAY_HEIGHT - (self.SCALE*2):
                reward = -10
                done = True
            if self.snake.y < (self.SCALE*1):
                reward = -10
                done = True

        if self.ENABLE_OBSTACLES:
            for i in range(self.obstacle.array_length):
                hit_obstacle = ((self.snake.x, self.snake.y) == (self.obstacle.array[i][0], self.obstacle.array[i][1]))

                if hit_obstacle:
                    # print("dead")
                    done = True
                    reward = -10

        # Update the snakes tail positions (from back to front)
        if self.snake.tail_length > 0:
            for i in range(self.snake.tail_length, 0, -1):
                self.snake.box[i] = self.snake.box[i-1]

        # Update the head position of the snake for the draw method
        self.snake.box[0] = (self.snake.x, self.snake.y)

        # The snake can only start crashing into itself after the tail length it greater than 3
        if self.snake.tail_length >= 3:
            # print(snake.tail_length) # DEBUGGING
            for i in range(1, self.snake.tail_length + 1):
                if(self.snake.box[0] == (self.snake.box[i])):
                    # done = True
                    #DEBUGGING
                    print("Crashed")
                    #print("Try again?")

        # Checking if the snake has reached the food
        reached_food = ((self.snake.x, self.snake.y) == (self.food.x, self.food.y)) 

        # Reward: Including the distance between them
        # reward = 100 / (np.sqrt((self.snake.x-self.food.x)**2 + (self.snake.y-self.food.y)**2) + 1)**2 
        
        # If the snake reaches the food, increment score (and increase snake tail length)
        if reached_food:
            self.score += 1 # Increment score

            # CHOOSE 1 OF THE 2 BELOW:

            # Create a piece of food that is not within the snake
            self.food.make(self.GRID_SIZE, self.SCALE, self.snake, self.ENABLE_OBSTACLES, self.obstacle)
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

    def get_state(self):
        state = np.zeros(4) # Does this increase programming time dramatically?
        state[0] = int(self.snake.x / self.SCALE)
        state[1] = int(self.snake.y / self.SCALE)
        state[2] = int(self.food.x / self.SCALE)
        state[3] = int(self.food.y / self.SCALE)

        return state

    # Number of states with just the head and food
    def number_of_states(self):
        return (self.GRID_SIZE**2)*((self.GRID_SIZE**2))


    # Number of actions that can be taken
    def number_of_actions(self):

        # forward, left, right
        # return 3

        # up, down, left, right
        return 4

    # The state represented at a onehot 1D vector
    def state_vector(self):
        # (rows, columns)
        state = np.zeros(((self.GRID_SIZE**2),3))
        
        # Probabily very inefficient - TODO, find a better implementation
        # This is for the HEAD and the FOOD and EMPTY, need to add a column for a TAIL later [H, T, F, E]
        for i in range(self.GRID_SIZE): # rows
            for j in range(self.GRID_SIZE): # columns
                if ((self.snake.x/self.SCALE) == j and (self.snake.y/self.SCALE) == i):
                    state[i*self.GRID_SIZE+j] = [1,0,0]
                    # print(i*self.GRID_SIZE+j)
                elif ((self.food.x/self.SCALE) == j and (self.food.y/self.SCALE) == i):
                    state[i*self.GRID_SIZE+j] = [0,1,0]
                    # print(i*self.GRID_SIZE+j)
                else:
                    state[i*self.GRID_SIZE+j] = [0,0,1]

        # Flatten the vector to a 1 dimensional vector for the input layer to the NN
        state = state.flatten()

        # Reshape into a column vector instead of a row vector - NOT SURE HOW
        # state = state.reshape(-1,1)

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
                        
                        if not self.countdown:
                            self.log_file.writerow([self.time, str(self.snake.x/self.SCALE), str(self.snake.y/self.SCALE), str(self.food.x/self.SCALE), str(self.food.y/self.SCALE), "2", str(self.score)])

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
                        
                        if not self.countdown:
                            self.log_file.writerow([self.time, str(self.snake.x/self.SCALE), str(self.snake.y/self.SCALE), str(self.food.x/self.SCALE), str(self.food.y/self.SCALE), "3", str(self.score)])

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
                        
                        if not self.countdown:
                            self.log_file.writerow([self.time, str(self.snake.x/self.SCALE), str(self.snake.y/self.SCALE), str(self.food.x/self.SCALE), str(self.food.y/self.SCALE), "0", str(self.score)])

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
                        
                        if not self.countdown:
                            self.log_file.writerow([self.time, str(self.snake.x/self.SCALE), str(self.snake.y/self.SCALE), str(self.food.x/self.SCALE), str(self.food.y/self.SCALE), "1", str(self.score)])

                    # log_file.writerow([pygame.time.get_ticks(), str(snake.x), str(snake.y), "S"])

        return action

    # Lets you simply play the game
    # Need to implement a log file to record the game, to attempt a IRL Algorithm
    def play(self):

        GAME_OVER = False
        GAME_ON = True

        t = 3

        self.prerender()

        with open("./Data/log_file.csv", 'w') as f:
            self.log_file = csv.writer(f, delimiter=",") #default delimiter is a comma
            self.log_file.writerow(["TIME_STAMP", "SNAKE_X", "SNAKE_Y", "FOOD_X", "FOOD_Y", "INPUT_BUTTON", "SCORE"])

            while GAME_ON:

                # self.food.seed(1)
                # np.random.seed(1)

                self.reset()

                self.log_file.writerow(["-1", str(self.snake.x/self.SCALE), str(self.snake.y/self.SCALE), str(self.food.x/self.SCALE), str(self.food.y/self.SCALE), "-1", str(self.score)])

                while not GAME_OVER:

                    action = self.render()

                    if self.countdown:
                        text = self.font.render(str(t), True, (255, 255, 255))
                        self.display.blit(text,(0,0))
                        pygame.display.update()
                        time.sleep(0.5)
                        t =  t - 1
                        if t == 0:
                            t = 3
                            self.countdown = False
                    else:
                        # When the snake touches the food, game ends
                        # action_space has to be 3 for the players controls, 
                        # because they know that the snake can't go backwards
                        s, r, GAME_OVER, i = self.step(action, action_space = 3)

                    # For the snake to look like it ate the food, render needs to be last
                    # Next piece of code if very BAD programming
                    if GAME_OVER:
                        print("Game Over")
                        self.log_file.writerow([self.time, str(self.snake.x/self.SCALE), str(self.snake.y/self.SCALE), str(self.food.x/self.SCALE), str(self.food.y/self.SCALE), "-1", str(self.score)])
                        # self.render()

                while GAME_OVER:
                    for event in pygame.event.get():
                        # print(event) # DEBUGGING

                        if event.type == pygame.QUIT:
                            self.end()

                        if event.type == pygame.KEYDOWN:
                            # In order to stop training and still save the Q txt file
                            if (event.key == pygame.K_q):
                                self.end()

                            if (event.key == pygame.K_SPACE):
                                GAME_OVER = False;
                                self.countdown = True
                                # self.log_file.writerow(["TIME_STAMP", "SNAKE_X", "SNAKE_Y", "FOOD_X", "FOOD_Y", "INPUT_BUTTON", "SCORE"])

        self.end()


    # Replay game from csv file
    def replay(self, csv_file_path):

        df = pd.read_csv(csv_file_path)
        print(df)

        GAME_OVER = False
        GAME_ON = True

        t = 3

        self.prerender()

        time_stamps = df["TIME_STAMP"].values

        my_actions = df["INPUT_BUTTON"].values

        print(len(time_stamps))
        print(len(my_actions))

        my_index = 1

        # with open("./Data/log_file.csv", 'r') as f:
        #     self.log_file = csv.reader(f, delimiter=",") #default delimiter is a comma


        #     for row in self.log_file:
        #         print(", ".join(row))

        while GAME_ON:

            # self.food.seed(0)
            # np.random.seed(0)

            self.reset()

            while not GAME_OVER:

                action = self.render()
                action = -1

                # print(self.time)
                if my_index < len(my_actions):
                    if self.time == time_stamps[my_index]:
                        action = my_actions[my_index]
                        my_index = my_index + 1

                if self.countdown:
                    text = self.font.render(str(t), True, (255, 255, 255))
                    self.display.blit(text,(0,0))
                    pygame.display.update()
                    time.sleep(0.5)
                    t =  t - 1
                    if t == 0:
                        t = 3
                        self.countdown = False
                else:
                    # When the snake touches the food, game ends
                    # action_space has to be 3 for the players controls, 
                    # because they know that the snake can't go backwards
                    s, r, GAME_OVER, i = self.step(action, action_space = 4)

                # For the snake to look like it ate the food, render needs to be last
                # Next piece of code if very BAD programming
                if GAME_OVER:
                    print("Game Over")
                    # self.render()

            while GAME_OVER:

                if my_index == len(my_actions)-1:
                    self.end()
                elif time_stamps[my_index] == -1:
                    GAME_OVER = False;
                    self.countdown = True
                    my_index = my_index + 1

                for event in pygame.event.get():
                    # print(event) # DEBUGGING

                    if event.type == pygame.QUIT:
                        self.end()

                    if event.type == pygame.KEYDOWN:
                        # In order to stop training and still save the Q txt file
                        if (event.key == pygame.K_q):
                            self.end()

                        if (event.key == pygame.K_SPACE):
                            GAME_OVER = False;
                            self.countdown = True
                            my_index = my_index + 1

        self.end()



# If I run this file by accident :P
if __name__ == "__main__":

    print("This file does not have a main method")

