'''
Simple SnakeAI Game 

@author: Matthew Reynard
@year: 2018

Purpose: Use Tensorflow to create a RL environment to train this snake to beat the game, i.e. fill the grid
Start Date: 5 March 2018
Due Date: Dec 2018

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
import contextlib
with contextlib.redirect_stdout(None):
    import pygame # Allows pygame to import without printing the pygame welcome message
from snakeAI import Snake
from foodAI import Food
from obstacleAI import Obstacle
from multiplierAI import Multiplier
import sys
import math # Used for infinity game time

# import csv
# import pandas as pd
# import matplotlib.pyplot as plt 

class Environment:

    def __init__(self, wrap = False, grid_size = 10, rate = 100, max_time = math.inf, tail = False, action_space = 3, food_count = 1, obstacle_count = 0, multiplier_count = 3, map_path = None):
        """
        Initialise the Game Environment with default values
        """

        #self.FPS = 120 # NOT USED YET
        self.UPDATE_RATE = rate
        self.SCALE = 20 # Scale of the snake body 20x20 pixels
        self.GRID_SIZE = grid_size
        self.LOCAL_GRID_SIZE = 9 # Has to be an odd number
        self.ENABLE_WRAP = wrap
        if not self.ENABLE_WRAP: 
            self.GRID_SIZE += 2
        self.ENABLE_TAIL = tail
        #self.ENABLE_OBSTACLES = obstacles
        self.ACTION_SPACE = action_space
        self.MAP_PATH = map_path
        
        self.DISPLAY_WIDTH = self.GRID_SIZE * self.SCALE
        self.DISPLAY_HEIGHT = self.GRID_SIZE * self.SCALE

        # Maximum timesteps before an episode is stopped
        self.MAX_TIME_PER_EPISODE = max_time

        # Create and Initialise Snake 
        self.snake = Snake()

        # Create Food
        self.NUM_OF_FOOD = food_count
        self.food = Food(self.NUM_OF_FOOD)

        # Create Obstacles
        self.NUM_OF_OBSTACLES = obstacle_count
        self.obstacle = Obstacle(self.NUM_OF_OBSTACLES)

        # Create Multipliers
        self.NUM_OF_MULTIPLIERS = multiplier_count
        self.multiplier = Multiplier(self.NUM_OF_MULTIPLIERS)

        self.score = 0
        self.time = 0
        self.state = None

        self.display = None
        self.bg = None
        self.clock = None
        self.font = None

        self.steps = 0

        # Used putting food and obstacles on the grid
        self.grid = []

        for j in range(self.GRID_SIZE):
            for i in range(self.GRID_SIZE):
                self.grid.append((i*self.SCALE, j*self.SCALE))


    def prerender(self):
        """
        If you want to render the game to the screen, you will have to prerender
        Load textures / images
        """

        pygame.init()
        pygame.key.set_repeat()

        self.display = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        pygame.display.set_caption('SnakeAI')
        self.clock = pygame.time.Clock()

        pygame.font.init()
        self.font = pygame.font.SysFont('Default', 32, bold=False)

        # Creates a visual Snake 
        self.snake.create(pygame)

        # Creates visual Food 
        self.food.create(pygame)

        # Creates visual Obstacles 
        self.obstacle.create(pygame)

        # Creates visual Multipliers 
        self.multiplier.create(pygame)

        # Creates the grid background
        self.bg = pygame.image.load("./Images/Grid50.png").convert()


    def reset(self):
        """Reset the environment"""

        self.steps = 0

        # Reset the score to 0
        self.score = 0

        # Positions on the grid that 
        disallowed = []

        # self.obstacle.array.clear()
        if self.MAP_PATH != None:
            self.obstacle.reset_map(self.GRID_SIZE, self.MAP_PATH, self.ENABLE_WRAP)
            if not self.ENABLE_WRAP:
                # pass
                self.obstacle.create_border(self.GRID_SIZE, self.SCALE)
        else:
            # Create obstacles at random positions
            # self.obstacle.reset(self.grid, disallowed)
            if not self.ENABLE_WRAP:
                self.obstacle.create_border(self.GRID_SIZE, self.SCALE)

        [disallowed.append(grid_pos) for grid_pos in self.obstacle.array]

        # Starting at random positions
        # self.snake.x = np.random.randint(0,self.GRID_SIZE) * self.SCALE
        # self.snake.y = np.random.randint(0,self.GRID_SIZE) * self.SCALE

        # self.snake.history.clear()
        # self.snake.history.append((self.snake.x, self.snake.y))

        # Starting at the same spot
        # self.snake.x = 8 * self.SCALE
        # self.snake.y = 8 * self.SCALE

        self.snake.reset(self.grid, disallowed)

        # Initialise the movement to the right
        self.snake.dx = 1
        self.snake.dy = 0

        # self.snake.pos = (self.snake.x, self.snake.y)

        disallowed.append(self.snake.pos)

        # Create obstacles at random positions
        # self.obstacle.reset(self.grid, disallowed)

        # [disallowed.append(grid_pos) for grid_pos in self.obstacle.array]

        # Create a piece of food that is not within the snake
        self.food.reset(self.grid, disallowed)
        # self.food.make_within_range(self.GRID_SIZE, self.SCALE, self.snake)

        [disallowed.append(grid_pos) for grid_pos in self.food.array]

        self.multiplier.reset(self.grid, disallowed)

        # Reset snakes tail
        self.snake.reset_tail(0)

        # Fill the state array with the snake and food coordinates on the grid
        self.state = self.state_array()

        # Reset the time
        self.time = 0

        # A dictionary of information that can be useful
        info = {"time":self.time, "score":self.score}

        return self.state, info


    def render(self):
        """Renders ONLY the CURRENT state"""

        # Mainly to close the window and stop program when it's running
        action = self.controls()

        # Set the window caption to the score
        pygame.display.set_caption("Score: " + str(self.score))

        # Draw the background, the snake and the food
        self.display.blit(self.bg, (0, 0))
        self.obstacle.draw(self.display)
        self.food.draw(self.display)
        self.multiplier.draw(self.display)
        self.snake.draw(self.display)

        # Text on screen
        text = self.font.render("Score: "+str(int(self.score)), True, (240, 240, 240, 0))
        self.display.blit(text,(10,0))
        text = self.font.render("Multiplier: "+str(self.snake.score_multiplier)+"x", True, (240, 240, 240, 0))
        self.display.blit(text,(150,0))

        # Update the pygame display
        pygame.display.update()

        # print(pygame.display.get_surface().get_size())

        # pygame.display.get_surface().lock()
        # p = pygame.PixelArray(pygame.display.get_surface())
        # p = pygame.surfarray.array3d(pygame.display.get_surface())
        # print("PIXELS:", p.shape)
        # pygame.display.get_surface().unlock()

        # print(clock.get_rawtime())
        # clock.tick(self.FPS) # FPS setting

        # Adds a delay to the the game when rendering so that humans can watch
        pygame.time.delay(self.UPDATE_RATE)

        # Testing
        return action


    def end(self):
        """
        Ending the Game -  This has to be at the end of the code
        Clean way to end the pygame env
        with a few backups...
        """

        pygame.quit()
        quit()
        sys.exit(0) # safe backup  


    def wrap(self):
        """ If the snake goes out the screen bounds, wrap it around"""

        if self.snake.x > self.DISPLAY_WIDTH - self.SCALE:
            self.snake.x = 0
        if self.snake.x < 0:
            self.snake.x = self.DISPLAY_WIDTH - self.SCALE
        if self.snake.y > self.DISPLAY_HEIGHT - self.SCALE:
            self.snake.y = 0
        if self.snake.y < 0:
            self.snake.y = self.DISPLAY_HEIGHT - self.SCALE
        self.snake.pos = (self.snake.x, self.snake.y)


    def step(self, action):
        """
        Step through the game, one state at a time.
        Return the reward, the new_state, whether its reached_food or not, and the time
        """

        self.steps += 1

        # Rewards:
        reward_out_of_bounds = -10
        reward_hitting_tail = -10
        reward_each_time_step = -0.1
        reward_reaching_food = 10

        # Increment time step
        self.time += 1

        # If the snake has reached the food
        reached_food = False

        hit_obstacle = False

        # If the episode is finished - after a certain amount of timesteps or it crashed
        done = False

        # Initialze to -1 for every time step - to find the fastest route (can be a more negative reward)
        # reward = -1
        reward = -0.01

        # Update the position of the snake head and tail
        self.snake.update(self.SCALE, action, self.ACTION_SPACE)

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

        for i in range(self.obstacle.array_length):
            hit_obstacle = (self.snake.pos == self.obstacle.array[i])

            if hit_obstacle:
                done = True
                reward = -1

        # Update the snakes tail positions (from back to front)
        if self.steps <= self.snake.start_tail_length:
            for i in range(self.steps, 0, -1):
                # print(i, len(self.snake.box))
                self.snake.box[i] = self.snake.box[i-1]

        else:
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

        # Make the most recent history have the most negative rewards
        # decay = (1+reward_each_time_step)/self.snake.history_size
        # for i in range(len(self.snake.history) - 1):
        #     # print(-1+(decay*i))
        #     if ((self.snake.x, self.snake.y) == (self.snake.history[-i-2][0], self.snake.history[-i-2][1])):
        #         reward = -1+(decay*i)
        #         break

        # Checking if the snake has reached the food
        reached_food, eaten_food = self.food.eat(self.snake)

        # Checking if the snake has reached the multiplier
        reached_multiplier, eaten_multiplier = self.multiplier.eat(self.snake)

        if reached_multiplier:
            # After every 3 multipliers are gathered, increase the score multiplier
            if self.multiplier.amount_eaten % 3 == 0 and self.multiplier.amount_eaten != 0:
                self.snake.score_multiplier += 1

            disallowed = []
            [disallowed.append(grid_pos) for grid_pos in self.obstacle.array]
            [disallowed.append(grid_pos) for grid_pos in self.food.array]
            self.multiplier.make(self.grid, self.snake, disallowed, index = eaten_multiplier)

        # if self.steps % 10 == 0:
        #     self.snake.score_multiplier = self.steps / 10

        # Reward: Including the distance between them
        # if reward == 0:
        #     reward = ((self.GRID_SIZE**2) / np.sqrt(((self.snake.x/self.SCALE-self.food.x/self.SCALE)**2 + (self.snake.y/self.SCALE-self.food.y/self.SCALE)**2) + 1)**2)/(self.GRID_SIZE**2)
            # print(reward) 

        
        # If the snake reaches the food, increment score (and increase snake tail length)
        if reached_food:
            self.score += 1*self.snake.score_multiplier # Increment score

            # Create a piece of food that is not within the snake
            if self.score + self.NUM_OF_FOOD <= self.GRID_SIZE**2:
                disallowed = []
                [disallowed.append(grid_pos) for grid_pos in self.obstacle.array]
                [disallowed.append(grid_pos) for grid_pos in self.multiplier.array]
                self.food.make(self.grid, self.snake, disallowed, index = eaten_food)

            # Test for one food item at a time
            # done = True 

            # Can't implement tail with Q learning algorithm
            if self.ENABLE_TAIL:
                self.snake.tail_length += 1
                self.snake.box.append((self.snake.x, self.snake.y)) #adds a rectangle variable to snake.box array(list)

            # Reward functions
            reward = 1
            # reward = 100 / (np.sqrt((self.snake.x-self.food.x)**2 + (self.snake.y-self.food.y)**2) + 1) # Including the distance between them
            # reward = 1000 * self.score
            # reward = 1000 / self.time # Including the time in the reward function

        # If the episode takes longer than the max time, it ends
        if self.time == self.MAX_TIME_PER_EPISODE:
            done = True

        # Get the new_state
        new_state = self.state_array()

        # A dictionary of information that may be useful
        info = {"time": self.time, "score": self.score}

        return new_state, reward, done, info


    def state_index(self, state_array):
        """
        Given the state array, return the index of that state as an integer
        Used for the Qlearning lookup table
        """
        return int((self.GRID_SIZE**3)*state_array[0]+(self.GRID_SIZE**2)*state_array[1]+(self.GRID_SIZE**1)*state_array[2]+(self.GRID_SIZE**0)*state_array[3])


    def sample_action(self):
        """
        Return a random action

        Can't use action space 4 with a tail, else it will have a double chance of doing nothing
        or crash into itself
        """
        return np.random.randint(0, self.ACTION_SPACE) 


    def set_state(self, state):
        """Set the state of the game environment"""
        self.state = state


    def number_of_states(self):
        """
        Return the number of states with just the snake head and 1 food

        Used for Q Learning look up table
        """
        return (self.GRID_SIZE**2)*((self.GRID_SIZE**2))


    def number_of_actions(self):
        """
        Return the number of possible actions 

        Used for Q Learning look up table

        Options:
        > 'forward' (i.e. do nothing), left, right
        > up, down, left, right
        """
        return self.ACTION_SPACE


    def state_array(self):
        """
        The state represented as an array or snake positions and food positions

        Used for Q learning
        """
        new_state = np.zeros(4) 

        new_state[0] = int(self.snake.x / self.SCALE)
        new_state[1] = int(self.snake.y / self.SCALE)
        new_state[2] = int(self.food.x / self.SCALE)
        new_state[3] = int(self.food.y / self.SCALE)

        return new_state


    def state_vector(self):
        """
        The state represented as a onehot 1D vector

        Used for the feed forward NN
        """

        if self.ENABLE_TAIL:
           # (rows, columns)
            state = np.zeros((self.GRID_SIZE**2, 4))
            
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
            state = np.zeros((self.GRID_SIZE**2, 3))
            
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
        """
        State as a 3D vector of the whole map for the CNN
        """
        print(int(self.snake.y/self.SCALE), int(self.snake.x/self.SCALE))

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


    def local_state_vector_3D(self): 
        """
        State as a 3D vector of the local area around the snake

        Shape = (3,9,9)
        """

        #s = snake
        sx = int(self.snake.x/self.SCALE)
        sy = int(self.snake.y/self.SCALE)

        # state = np.zeros((3, self.LOCAL_GRID_SIZE, self.LOCAL_GRID_SIZE))
        state = np.zeros((4, self.LOCAL_GRID_SIZE, self.LOCAL_GRID_SIZE)) # When using history

        # Agent
        local_pos = int((self.LOCAL_GRID_SIZE-1)/2)
        state[0, local_pos, local_pos] = 1

        # History
        decay = 0.05

        for i in range(len(self.snake.history)-1):
            x_prime = local_pos+int(self.snake.history[-i-2][0]/self.SCALE)-int(self.snake.x/self.SCALE)
            y_prime = local_pos+int(self.snake.history[-i-2][1]/self.SCALE)-int(self.snake.y/self.SCALE)

            if x_prime < self.LOCAL_GRID_SIZE and x_prime >= 0 and y_prime < self.LOCAL_GRID_SIZE and y_prime >= 0:
                if 1-decay*i >= 0 and state[3, y_prime, x_prime] == 0:
                    state[3, y_prime, x_prime] = 1-decay*i
                # else:
                    # state[3, y_prime, x_prime] = 0

        # Food
        for i in range(self.NUM_OF_FOOD):
            x_prime_food = local_pos+int(self.food.array[i][0]/self.SCALE)-int(self.snake.x/self.SCALE)
            y_prime_food = local_pos+int(self.food.array[i][1]/self.SCALE)-int(self.snake.y/self.SCALE)

            if x_prime_food < self.LOCAL_GRID_SIZE and x_prime_food >= 0 and y_prime_food < self.LOCAL_GRID_SIZE and y_prime_food >= 0:
                state[1, y_prime_food, x_prime_food] = 1


        # Obstacles

        # Tail
        for i in range(1, self.snake.tail_length + 1):
            
            x_prime_tail = local_pos+int(self.snake.box[i][0]/self.SCALE)-int(self.snake.x/self.SCALE)
            y_prime_tail = local_pos+int(self.snake.box[i][1]/self.SCALE)-int(self.snake.y/self.SCALE)

            if x_prime_tail < self.LOCAL_GRID_SIZE and x_prime_tail >= 0 and y_prime_tail < self.LOCAL_GRID_SIZE and y_prime_tail >= 0:
                state[2, y_prime_tail, x_prime_tail] = 1

        # Walls
        for j in range(0, self.LOCAL_GRID_SIZE):
            for i in range(0, self.LOCAL_GRID_SIZE):

                x_prime_wall = local_pos-sx
                y_prime_wall = local_pos-sy

                if i < x_prime_wall or j < y_prime_wall:
                    state[2, j, i] = 1

                x_prime_wall = local_pos+(self.GRID_SIZE-sx)-1
                y_prime_wall = local_pos+(self.GRID_SIZE-sy)-1

                if i > x_prime_wall or j > y_prime_wall:
                    state[2, j, i] = 1

        return state


    def pixels(self): 
        """
        Returns the pixels in a (GRID*20, GRID*20, 3) size array/
    
        Unfortunatly it has to render in order to gather the pixels
        """
        return pygame.surfarray.array3d(pygame.display.get_surface())


    def controls(self):
        """Defines all the players controls during the game"""

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


    def play(self):
        """ 
        Lets you simply play the game

        Useful for debugging and testing out the environment
        """

        GAME_OVER = False

        self.prerender()

        self.reset()

        while not GAME_OVER:

            # print(self.local_state_vector_3D()) # DEBUGGING
            # print(self.snake.history)

            action = self.render()

            # print(self.pixels().shape)

            # When the snake touches the food, game ends
            # action_space has to be 3 for the players controls, 
            # because they know that the snake can't go backwards
            s, r, GAME_OVER, i = self.step(action)

            # For the snake to look like it ate the food, render needs to be last
            # Next piece of code if very BAD programming
            if GAME_OVER:
                print("Game Over")
                # self.render()

        self.end()


# If I run this file by accident :P
if __name__ == "__main__":

    print("This file does not have a main method")
