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

'''

import numpy as np
import pygame
from snakeAI import Snake
from foodAI import Food
import sys

# import csv
# import pandas as pd
# import matplotlib.pyplot as plt 

# Trying to use a DQN with TF instead of the normal Q Learning
# import tensorflow as tf

class Environment:

    # Initialise the Game Environment
    def __init__(self, wrap, grid_size = 10, rate = 100, render = False):

        # self.FPS = 120 # NOT USED YET
        self.UPDATE_RATE = rate
        self.SCALE = 20 # scale of the snake body 20x20 pixels  
        self.state = 0
        self.GRID_SIZE = grid_size
        self.ENABLE_WRAP = wrap
        self.RENDER = render
        
        # self.score = 0 # NOT IMPLEMENTED YET

        self.DISPLAY_WIDTH = self.GRID_SIZE * self.SCALE
        self.DISPLAY_HEIGHT = self.GRID_SIZE * self.SCALE

        # Create and Initialise Snake 
        self.snake = Snake()

        # Create Food 
        self.food = Food()

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

        self.bg = pygame.image.load("./Images/Grid10.png").convert()

    def reset(self):
        # Starting at random positions
        self.snake.x = np.random.randint(0,10) * self.SCALE
        self.snake.y = np.random.randint(0,10) * self.SCALE

        # Starting at the same spot
        # self.snake.x = 3 * self.SCALE
        # self.snake.y = 3 * self.SCALE

        self.snake.dx = 1
        self.snake.dy = 0

        # Update the head position of the snake
        if self.RENDER:
            self.snake.box[0] = (self.snake.x, self.snake.y)

        self.food.make(self.GRID_SIZE, self.SCALE, self.snake)

        self.state[0] = int(self.snake.x / self.SCALE)
        self.state[1] = int(self.snake.y / self.SCALE)
        self.state[2] = int(self.food.x / self.SCALE)
        self.state[3] = int(self.food.y / self.SCALE)

        self.time = 0

        return self.state, self.time


    # Renders ONLY the CURRENT state
    def render(self):
        
        score = 0 # NOT IMPLEMENTED YET

        # Mainly to close the window and stop program when it's running
        action = self.controls()

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


    #If the snake goes out of the screen bounds, wrap it around
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
    def step(self, action):
        self.time += 1

        reached_food = False

        # Initialze to -1 for every time step - find the fastest route
        reward = -1

        self.snake.update(self.SCALE, action)

        if self.ENABLE_WRAP:
            self.wrap()
        else:
            if self.snake.x > self.DISPLAY_WIDTH - self.SCALE:
                reward = -10
                reached_food = True
            if self.snake.x < 0:
                reward = -10
                reached_food = True
            if self.snake.y > self.DISPLAY_HEIGHT - self.SCALE:
                reward = -10
                reached_food = True
            if self.snake.y < 0:
                reward = -10
                reached_food = True

        # Update the head position of the snake
        if self.RENDER:
            self.snake.box[0] = (self.snake.x, self.snake.y)

        reached_food = ((self.snake.x, self.snake.y) == (self.food.x, self.food.y)) #if snake reached food
        
        if reached_food:
            reward = 1000 / self.time

        new_state = np.zeros(4)
        new_state[0] = int(self.snake.x / self.SCALE)
        new_state[1] = int(self.snake.y / self.SCALE)
        new_state[2] = int(self.food.x / self.SCALE)
        new_state[3] = int(self.food.y / self.SCALE)

        info = self.time

        return new_state, reward, reached_food, info

    # Given the state array, return the index of that state as an integer
    def state_index(self, state_array):
        return int((self.GRID_SIZE**3)*state_array[0]+(self.GRID_SIZE**2)*state_array[1]+(self.GRID_SIZE**1)*state_array[2]+(self.GRID_SIZE**0)*state_array[3])


        #random action generator
    def sample(self):
        action = np.random.randint(0,3)
        return action


    def set_state(self, state):
        self.state = state


    # Number of states with just the head and food
    def number_of_states(self):
        return (self.GRID_SIZE**2)*((self.GRID_SIZE**2))


    # Number of actions that can be taken
    def number_of_actions(self):
        return 3


    # If the snake eats the food, increment score and increase snake tail length
    # NOT IMPLEMENTED YET
    def eatsFood(self):

        if ((self.snake.x, self.snake.y) == (self.food.x, self.food.y)):

            # score += 1 # NOT IMPLEMENTED YET
            food.make(self.GRID_SIZE, self.SCALE, self.snake)

            # CAN'T IMPLEMENT TAIL WITH Q LEARNING ALGORITHM

            # snake.tail_length += 1
            # snake.box.append(snake.tail_img.get_rect()) #adds a rectangle variable to snake.box array(list)

            # Create the new tail section by the head, and let it get updated tp the end with snake.update()
            # if snake.dx > 0: #RIGHT
            #     snake.box[snake.tail_length].topleft = (snake.x, snake.y)

            # if snake.dx < 0:#LEFT
            #     snake.box[snake.tail_length].topleft = (snake.x, snake.y)

            # if snake.dy > 0:#DOWN
            #     snake.box[snake.tail_length].topleft = (snake.x, snake.y)

            # if snake.dy < 0:#UP
            #     snake.box[snake.tail_length].topleft = (snake.x, snake.y)


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

            action = self.render()

            # When the snake touches the food, game ends
            s, r, GAME_OVER, i = self.step(action)

        self.end()


if __name__ == "__main__":

    print("This file does not have a main method")



