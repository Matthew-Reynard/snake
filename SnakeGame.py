'''

Simple Snake Game

@author: Matthew Reynard
@year: 2018

Purpose: Use Tensorflow to create a RL environment to train this snake to beat the game, i.e. have 1200-1 tail sections
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

# Trying to use DQN with TF instead of the normal Q Learning
# import tensorflow as tf

class Environment:

    def __init__(self, wrap, grid_size = 10, rate = 100, render = False):
        # self.FPS = 120
        self.UPDATE_RATE = rate
        self.SCALE = 20 # scale of the snake body 20x20 pixels  
        self.state = 0
        self.GRID_SIZE = grid_size
        self.ENABLE_WRAP = wrap
        self.RENDER = render
        # self.score = 0

        self.DISPLAY_WIDTH = self.GRID_SIZE*self.SCALE
        self.DISPLAY_HEIGHT = self.GRID_SIZE*self.SCALE

        # pygame.init()

        #Create and Initialise Snake 
        self.snake = Snake()

        #Create Food 
        self.food = Food()

        self.time = 0

        self.state = np.zeros(4)

        self.display = None
        self.bg = None
        self.clock = None


    def reset(self):
        self.snake.x = 3 * self.SCALE
        self.snake.y = 3 * self.SCALE
        self.snake.dx = 1
        self.snake.dy = 0

        # Update the head position of the snake
        if self.RENDER:
            self.snake.box[0].topleft = (self.snake.x, self.snake.y)

        self.food.make(self.GRID_SIZE, self.SCALE, self.snake)

        self.state[0] = int(self.snake.x / self.SCALE)
        self.state[1] = int(self.snake.y / self.SCALE)
        self.state[2] = int(self.food.x / self.SCALE)
        self.state[3] = int(self.food.y / self.SCALE)

        self.time = 0

        return self.state, self.time

    # number of states with just the head and food
    def number_of_states(self):
        return (self.GRID_SIZE**2)*((self.GRID_SIZE**2))

    # number of actions
    def number_of_actions(self):
        return 3

    # if you want to render the game to the screen, you will have to prerender
    # in order to load the textures (images)
    def prerender(self):
        pygame.init()

        self.display = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        pygame.display.set_caption('SnakeAI')
        self.clock = pygame.time.Clock()

        # Create and Initialise Snake 
        self.snake.create(pygame)

        # Create Food 
        self.food.create(pygame)

        self.bg = pygame.image.load("./Images/Grid10.png").convert()

    # Renders current state
    def render(self):
        
        score = 0 #NOT IMPLEMENTED YET

        # Draw the background, the snake and the food
        self.display.blit(self.bg, (0, 0))
        self.food.draw(self.display)
        self.snake.draw(self.display)

        pygame.display.update()

        # print(clock.get_rawtime())
        # clock.tick(self.FPS) # FPS setting

        pygame.time.delay(self.UPDATE_RATE)

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

    # Given the state array, return the index of that state as an integer
    def state_index(self, state_array):
        return int((self.GRID_SIZE**3)*state_array[0]+(self.GRID_SIZE**2)*state_array[1]+(self.GRID_SIZE**1)*state_array[2]+(self.GRID_SIZE**0)*state_array[3])

    # Step through the game, one state at a time.
    # Return the reward, the new_state, whether its reached_food or not, and the time
    def step(self, action):
        self.time += 1

        reached_food = False

        # initialze to -1 for every time step - find the fastest route
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

        #Update the head position of the snake
        if self.RENDER:
            self.snake.box[0].topleft = (self.snake.x, self.snake.y)

        reached_food = ((self.snake.x, self.snake.y) == (self.food.x, self.food.y)) #if snake reached food
        
        if reached_food:
            reward = 100

        new_state = np.zeros(4)
        new_state[0] = int(self.snake.x / self.SCALE)
        new_state[1] = int(self.snake.y / self.SCALE)
        new_state[2] = int(self.food.x / self.SCALE)
        new_state[3] = int(self.food.y / self.SCALE)

        info = self.time

        return new_state, reward, reached_food, info

        #random action generator
    def sample(self):
        action = np.random.randint(0,3)
        return action

    def set_state(self, state):
        self.state = state

    #If the red mask overlaps the food mask, increment score and increase snake tail length
    def eatsFood(self, snake):

        offset_x, offset_y = (food.rect.left - snake.box[0].left), (food.rect.top - snake.box[0].top)

        if (snake.head_mask.overlap(food.mask, (offset_x, offset_y)) != None):

            score += 1
            food.make(self.GRID_SIZE, self.SCALE, snake)

            #CAN'T IMPLEMENT TAIL WITH QLEARNING

            # snake.tail_length += 1
            # snake.box.append(snake.tail_img.get_rect()) #adds a rectangle variable to snake.box array(list)

            #Create the new tail section by the head, and let it get updated tp the end with snake.update()
            # if snake.dx > 0: #RIGHT
            #     snake.box[snake.tail_length].topleft = (snake.x, snake.y)

            # if snake.dx < 0:#LEFT
            #     snake.box[snake.tail_length].topleft = (snake.x, snake.y)

            # if snake.dy > 0:#DOWN
            #     snake.box[snake.tail_length].topleft = (snake.x, snake.y)

            # if snake.dy < 0:#UP
            #     snake.box[snake.tail_length].topleft = (snake.x, snake.y)


if __name__ == "__main__":
    print("this does nothing")



