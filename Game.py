'''

Simple Snake Game. FIRST ATTEMPT!

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
from snake import Snake
from food import Food
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt 

#Trying to use DQN with TF instead of the normal Q Learning
import tensorflow as tf

pygame.init()

pygame.font.init() # you have to call this at the start, if you want to use this module.
myfont = pygame.font.SysFont('arial', 30)
gameOverFont = pygame.font.SysFont('arial', 14)

#print(pygame.font.get_fonts())

#Window size

GRID_SIZE = 10
SCALE = 20 # Block/Grid size (in pixels)

DISPLAY_WIDTH = GRID_SIZE * SCALE
DISPLAY_HEIGHT = GRID_SIZE * SCALE
# DISPLAY_WIDTH = 800
# DISPLAY_HEIGHT = 600

#Wrap allows snake to wrap around and not crash into the edge of the screen
ENABLE_WRAP = True

#Not sure what this does
GAME_QUIT = False

#Various RGB colours
white = (255,255,255)
black = (0,0,0)
red = (255,0,0)

FPS = 120 # If not set, game tries to max out FPS (last test was 1200 FPS)
score = 0
MOVEMENT_SPEED = 1 # Meaning 1 block per timestep

UPDATE_RATE = 50 # lower the faster

gameDisplay = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption('SnakeAI')
clock = pygame.time.Clock()

bg = pygame.image.load("./Images/Grid10.png").convert()

#Create and Initialise Snake 
snake = Snake(pygame)

#Create Food 
food = Food(pygame)

#pygame.mouse.set_visible(False)

#If the snake goes out of the screen bounds, wrap it around
def wrap(x,y):
    if snake.x > DISPLAY_WIDTH - SCALE:
        snake.x = 0;
    if snake.x < 0:
        snake.x = DISPLAY_WIDTH - SCALE;
    if snake.y > DISPLAY_HEIGHT - SCALE:
        snake.y = 0;
    if snake.y < 0:
        snake.y = DISPLAY_HEIGHT - SCALE;

#If the red mask overlaps the food mask, increment score and increase snake tail length
def eatsFood(score):

    offset_x, offset_y = (food.rect.left - snake.box[0].left), (food.rect.top - snake.box[0].top)

    if (snake.head_mask.overlap(food.mask, (offset_x, offset_y)) != None):

        score += 1
        food.make(GRID_SIZE, SCALE, snake)

        snake.tail_length += 1
        snake.box.append(snake.tail_img.get_rect()) #adds a rectangle variable to snake.box array(list)

        #Create the new tail section by the head, and let it get updated tp the end with snake.update()
        if snake.dx > 0: #RIGHT
            snake.box[snake.tail_length].topleft = (snake.x, snake.y)

        if snake.dx < 0:#LEFT
            snake.box[snake.tail_length].topleft = (snake.x, snake.y)

        if snake.dy > 0:#DOWN
            snake.box[snake.tail_length].topleft = (snake.x, snake.y)

        if snake.dy < 0:#UP
            snake.box[snake.tail_length].topleft = (snake.x, snake.y)

    return score


#Defines all the Players controls during the game
def controls(GAME_OVER, log_file):
    for event in pygame.event.get():
        #print(event) # DEBUGGING
        if event.type == pygame.QUIT:
            log_file.writerow(["---END---"])
            GAME_QUIT = True
            pygame.quit()
            quit()
            sys.exit(0) # safe backup

        if event.type == pygame.KEYDOWN:
            #print(event.key) #DEBUGGING

            #Moving left
            if (event.key == pygame.K_LEFT or event.key == pygame.K_a) and snake.dx == 0 and GAME_OVER == False:
                snake.move = 1
                # snake.dx = -MOVEMENT_SPEED
                # if snake.dy == 1:
                #     snake.head_img = pygame.transform.rotate(snake.head_img, -90)
                # else:
                #     snake.head_img = pygame.transform.rotate(snake.head_img, 90)
                # snake.dy = 0
                # # log_file.write("A\n")
                # log_file.writerow([pygame.time.get_ticks(), str(snake.x), str(snake.y), "A"])

            #Moving right
            elif (event.key == pygame.K_RIGHT or event.key == pygame.K_d) and snake.dx == 0 and GAME_OVER == False:
                snake.move = 2
                # snake.dx = MOVEMENT_SPEED
                # if snake.dy == 1:
                #     snake.head_img = pygame.transform.rotate(snake.head_img, 90)
                # else:
                #     snake.head_img = pygame.transform.rotate(snake.head_img, -90)
                # snake.dy = 0
                # #log_file.write("D\n")
                # log_file.writerow([pygame.time.get_ticks(), str(snake.x), str(snake.y), "D"])


            #Moving up
            elif (event.key == pygame.K_UP or event.key == pygame.K_w) and snake.dy == 0 and GAME_OVER == False:
                snake.move = 3
                # snake.dy = -MOVEMENT_SPEED
                # if snake.dx == 1:
                #     snake.head_img = pygame.transform.rotate(snake.head_img, 90)
                # else:
                #     snake.head_img = pygame.transform.rotate(snake.head_img, -90)
                # snake.dx = 0
                # #log_file.write("W\n")
                # log_file.writerow([pygame.time.get_ticks(), str(snake.x), str(snake.y), "W"])


            #Moving down
            elif (event.key == pygame.K_DOWN or event.key == pygame.K_s) and snake.dy == 0 and GAME_OVER == False:
                snake.move = 4
                # snake.dy = MOVEMENT_SPEED
                # if snake.dx == 1:
                #     snake.head_img = pygame.transform.rotate(snake.head_img, -90)
                # else:
                #     snake.head_img = pygame.transform.rotate(snake.head_img, 90)
                # snake.dx = 0
                # #log_file.write("S\n")
                # log_file.writerow([pygame.time.get_ticks(), str(snake.x), str(snake.y), "S"])


            #DEBUGGING - add to the snakes tail
            elif event.key == pygame.K_f and GAME_OVER == False: 
                #score += 1
                #score_text = myfont.render("Score: " + str(score), True, white)  # anti aliasing

                snake.tail_length += 1
                snake.box.append(snake.tail_img.get_rect()) #adds a rectangle variable to snake.box array(list)
                snake.box[snake.tail_length].topleft = (snake.x, snake.y)

            #GAME OVER: Press space to try play again
            elif (event.key == pygame.K_SPACE) and GAME_OVER == True:
                GAME_OVER = False

                # log_file.writerow(["---NEW GAME--"])
                # log_file.writerow(["TIME_STAMP", "SCORE", "SNAKE_X", "SNAKE_Y", "INPUT_BUTTON", "FOOD_X", "FOOD_Y"])

                # if snake.dx == 1 and snake.dy == 0:
                #     #print("going R")
                #     pass
                # elif snake.dx == -1 and snake.dy == 0:
                #     #print("going L")
                #     snake.head_img = pygame.transform.rotate(snake.head_img, 180)
                # elif snake.dy == 1 and snake.dx == 0:
                #     #print("going D")
                #     snake.head_img = pygame.transform.rotate(snake.head_img, 90)
                # elif snake.dy == -1 and snake.dx == 0:
                #    #print("going U")
                #     snake.head_img = pygame.transform.rotate(snake.head_img, -90)

                snake.dx = MOVEMENT_SPEED # Initilise the snake moving to the left
                snake.dy = 0

                #print(snake.dx, snake.dy)
                snake.tail_length = 0
                snake.box.clear()

                snake.box = [(0,0)] # Not sure if this is neccessary

                snake.box[0] = snake.head_img.get_rect()
                snake.box[0].topleft = (0, 0)
                food.make(GRID_SIZE, SCALE, snake)

            elif (event.key == pygame.K_q) and GAME_OVER == True:
                # log_file.writerow(["---END---"])
                GAME_QUIT = True # Not working
                pygame.quit()
                quit()
                sys.exit(0) # safe backup   

            break

    return GAME_OVER


#Main game loop
def gameLoop(GAME_QUIT, log_file, start_time):

    #initialize starting variable to ensure that the players can retry after a GAME OVER
    score = 0
    snake.dx = MOVEMENT_SPEED # Initilise the snake moving to the left
    snake.dy = 0
    snake.x = 4*SCALE # Snake in top left corner
    snake.y = 4*SCALE
    GAME_OVER = False
    YOU_WIN = False

    #print("In game loop") # DEBUGGING

    while not GAME_QUIT:
        #print("Within the game loop") # DEBUGGING

        #print(pygame.time.get_ticks())
        #Players controls while in the game loop
        GAME_OVER = controls(GAME_OVER, log_file)

        if GAME_OVER:
            #Game over screen
            score = gameOver(score) 
        elif YOU_WIN:
            score = win(score)
        else:
            if snake.tail_length == (GRID_SIZE * GRID_SIZE) - 2:
                YOU_WIN = True

            #In game screen
            # FPS_text = myfont.render("FPS: %0.3f" % clock.get_fps(), True, white)  # anti aliasing lowers performance
            
            #Checks if the snake has eaten the food
            score = eatsFood(score)
            # score_text = myfont.render("Score: " + str(score), True, white)

            #Update the snakes positioning
            #print("Start time:", start_time, "        Time elapsed:" , start_time - pygame.time.get_ticks())
            if pygame.time.get_ticks() - start_time >= UPDATE_RATE:
                snake.update(SCALE, pygame, log_file, food, score)
                start_time = pygame.time.get_ticks()
            
            #Update the snakes tail positions (from back to front)
            if snake.box[0].topleft != (snake.x, snake.y): # if the snake has moved

                if snake.tail_length > 0:
                    for i in range(snake.tail_length, 0, -1):
                        #print(i)
                        snake.box[i].topleft = snake.box[i-1].topleft

                if ENABLE_WRAP:
                    wrap(snake.x, snake.y)
                else:
                    if snake.x > DISPLAY_WIDTH - SCALE:
                        GAME_OVER=True
                    if snake.x < 0:
                        GAME_OVER=True
                    if snake.y > DISPLAY_HEIGHT - SCALE:
                        GAME_OVER=True
                    if snake.y < 0:
                        GAME_OVER=True

                #Update the head position of the snake
                snake.box[0].topleft = (snake.x, snake.y)

                #If you think about it, the snake can only start crashing into itself after the tail length it greater than 3
                if snake.tail_length >= 3:
                    # print(snake.tail_length)
                    for i in range(1, snake.tail_length + 1):
                        if(snake.box[0].colliderect(snake.box[i])):
                            GAME_OVER = True
                            #DEBUGGING
                            #print("Game Over")
                            #print("Try again?")

            gameDisplay.fill(black)  #set background
            gameDisplay.blit(bg, (0, 0))
            pygame.display.set_caption("Score: " + str(score))
            # gameDisplay.blit(FPS_text, (0, 0))
            # gameDisplay.blit(score_text, (DISPLAY_WIDTH - 150, 0))

            food.draw(gameDisplay)
            snake.draw(gameDisplay)
            
        pygame.display.update()
        #print(clock.get_rawtime())
        clock.tick(FPS) # FPS setting
        #print(GAME_OVER)

    #Outside of Main Game While Loop i.e if GAME_QUIT == True
    pygame.quit()
    quit()
    sys.exit(0) # safe backup   
        

def gameOver(score):

    gameDisplay.fill(black)  #set background
    game_over_text = myfont.render("GAME OVER", True, white)
    try_again_text = gameOverFont.render("Press SPACE to Try Again", True, white)
    quit_text = gameOverFont.render("Press Q to Quit", True, white)

    gameDisplay.blit(game_over_text, ((int)(DISPLAY_WIDTH*0.1), (int)(DISPLAY_HEIGHT*0.2))) # 120, 200
    gameDisplay.blit(try_again_text, ((int)(DISPLAY_WIDTH*0.1), (int)(DISPLAY_HEIGHT*0.55))) #235 400
    gameDisplay.blit(quit_text,  ((int)(DISPLAY_WIDTH*0.1), (int)(DISPLAY_HEIGHT*0.65))) #300 ,450

    score = 0
    snake.x = 4*20 # 4 * SCALE
    snake.y = 4*20
    #snake.dx = MOVEMENT_SPEEDs
    #snake.dy = 0
    return score

def win(score):

    gameDisplay.fill(black)  #set background
    game_over_text = myfont.render("YOU WIN", True, white)
    try_again_text = gameOverFont.render("Press SPACE to Try Again", True, white)
    quit_text = gameOverFont.render("Press Q to Quit", True, white)

    gameDisplay.blit(game_over_text, ((int)(DISPLAY_WIDTH*0.1), (int)(DISPLAY_HEIGHT*0.2))) # 120, 200
    gameDisplay.blit(try_again_text, ((int)(DISPLAY_WIDTH*0.1), (int)(DISPLAY_HEIGHT*0.55))) #235 400
    gameDisplay.blit(quit_text,  ((int)(DISPLAY_WIDTH*0.1), (int)(DISPLAY_HEIGHT*0.65))) #300 ,450

    score = 0
    snake.x = 4*20 # 4 * SCALE
    snake.y = 4*20
    #snake.dx = MOVEMENT_SPEEDs
    #snake.dy = 0
    return score

def state_index(state):
    # print(state[0], state[1], state[2], state[3])
    # return int( ( 10*state[0] + state[1]+1 )*( 10*state[2] + state[3]+1 ) )-1
    return int(1000*state[0]+100*state[1]+10*state[2]+state[3])

def QLearn():
    width = int(DISPLAY_WIDTH/SCALE) #6
    height = int(DISPLAY_HEIGHT/SCALE) #6
    no_of_actions = 4

    # reset the Q matrix to a bunch of random numbers
    # Q = np.random.rand((height*width)*((height*width)), no_of_actions) # dimensions: (states, actions)

    # use the previous runs Q matrix
    Q = np.loadtxt("q.txt", dtype='float', delimiter=" ")

    # Reseting the Q matrix to zeroes
    # Q = np.zeros(((height*width*height*width), no_of_actions)) # dimensions: (states, actions)

    # Q.size = 600 x 4


    # Hyper parameters
    alpha = 0.4  # learning rate, i.e. which fraction of the Q values should be updated
    gamma = 0.99  # discount factor, i.e. to which extent the algorithm considers possible future rewards
    epsilon = 0.4  # probability to choose random action instead of best action


    history = []

    verbose = False # no printing to the screen

    myTime = 0
    reward = 0

    last_time = 0

    for epoch in range(100000):
        state, myTime = reset(myTime)
        done = False
        food.make(GRID_SIZE, SCALE, snake)
        timestep = 0

        if epoch % 1000 == 0:
            print(epoch, "avg timestep:", last_time/1000)
            last_time = 0

        while not done:
            if verbose: print('State {} (index {}),'.format(state, state_index(state)))

            #Choosing a random action or choosing the best action
            if np.random.rand() <= epsilon:
                timestep += 1
                action = np.random.randint(no_of_actions) # 4 actions: left, right, up, down
                if verbose: print('Choosing random action {},'.format(action))
            else:
                timestep += 1
                # print(state_index(state))
                action = np.argmax(Q[state_index(state)])
                if verbose: print('Choosing best action {},'.format(action))

            new_state, reward, done, myTime = step(action, myTime, state)
            if verbose: print('reward {}'.format(reward))

            Q[state_index(state), action] += alpha * (reward + gamma * np.max(Q[state_index(new_state)]) - Q[state_index(state), action])
            
            state = new_state

            if timestep == 500:
                # print("Ended at timestep: ",timestep)
                done = True

            if done:
                last_time += timestep

            # gameDisplay.fill(white)  #set background
            # gameDisplay.blit(snake.head_img, (snake.x, snake.y))
            # # food.draw(gameDisplay)
            # pygame.display.update()
            # #print(clock.get_rawtime())
            # clock.tick() # FPS setting

        # history.append(myTime)
        if verbose: print('Finished in {} steps'.format(myTime))
        if verbose: print("")

    np.savetxt("q.txt", Q.astype(np.float), fmt='%f', delimiter = " ")
    # np.savetxt("h.txt", history.astype(float), fmt='%f', delimiter = " ")

    # plt.plot(history)
    # plt.xlabel('Epoch')
    # plt.ylabel('Time steps to solve maze')
    # plt.ylim(0, 40000)
    # plt.show()

    # for i in range(len(Q)):
    #     print(Q[i])

    # while True:
    #     for event in pygame.event.get():
    #     #print(event) # DEBUGGING
    #         if event.type == pygame.QUIT:
    #             # GAME_QUIT = True
    #             pygame.quit()
    #             quit()
    #             sys.exit(0) # safe backup

def reset(myTime):

    snake.x = 4*20 # Snake in top left corner
    snake.y = 4*20
    snake.dx = MOVEMENT_SPEED
    snake.dy = 0
    # score = 0
    myTime = 0

    state = np.zeros(4)
    state[0] = int(snake.x / SCALE)
    state[1] = int(snake.y / SCALE)
    state[2] = int(food.x / SCALE)
    state[3] = int(food.y / SCALE)

    return state, myTime

def step(action, time, state):
    time += 1
    # new_player = (snake.dx, snake.dy)
    reached_food = 0

    # initialze to -1 for every time step - find the fastest route
    reward = -1

    #LEFT
    if action == 0: 
        snake.dx = 1
        snake.dy = 0

    #RIGHT
    elif action == 1:
        snake.dx = -1
        snake.dy = 0

    #UP
    elif action == 2:
        snake.dx = 0
        snake.dy = -1

    #DOWN
    elif action == 3:
        snake.dx = 0
        snake.dy = 1

    snake.x += snake.dx * SCALE
    snake.y += snake.dy * SCALE

    if ENABLE_WRAP:
        wrap(snake.x, snake.y)
    else:
        if snake.x > DISPLAY_WIDTH - SCALE:
            reward = -1
            reached_food = True
        if snake.x < 0:
            reward = -1
            reached_food = True
        if snake.y > DISPLAY_HEIGHT - SCALE:
            reward = -1
            reached_food = True
        if snake.y < 0:
            reward = -1
            reached_food = True
    
    # if snake.x >= 0 and snake.x < DISPLAY_WIDTH and snake.y >= 0 and snake.y < DISPLAY_HEIGHT:
    #     reached_food = ((snake.x,snake.y)==(food.x, food.y))
    #     reward = 100*reached_food / time
    # else:
    #     reward = -1

    # if the snake has reached the food -> done = True
    

    # if reward != -1:
    #     reached_food = ((snake.x,snake.y)==(food.x, food.y))
    #     reward = 100*reached_food / time
    #     state[0] = int(snake.x / SCALE)
    #     state[1] = int(snake.y / SCALE)
    #     state[2] = int(food.x / SCALE)
    #     state[3] = int(food.y / SCALE)

    reached_food = ((snake.x,snake.y) == (food.x, food.y)) #if snake reached food
    
    if reached_food:
        reward = 100

    state[0] = int(snake.x / SCALE)
    state[1] = int(snake.y / SCALE)
    state[2] = int(food.x / SCALE)
    state[3] = int(food.y / SCALE)

    return state, reward, reached_food, time

def ai_controls(action, snake):

    #Moving left
    if (action==0) and snake.dx == 0:
        snake.move = 1

    #Moving right
    elif (action==1) and snake.dx == 0:
        snake.move = 2

    #Moving up
    elif (action==2) and snake.dy == 0:
        snake.move = 3

    #Moving down
    elif (action==3) and snake.dy == 0:
        snake.move = 4

    for event in pygame.event.get():
        #print(event) # DEBUGGING
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
            sys.exit(0) # safe backup

def Qrun():
    width = int(DISPLAY_WIDTH/SCALE) #6
    height = int(DISPLAY_HEIGHT/SCALE) #6

    # Q = np.random.rand((height*width)*(height*width), no_of_actions) # dimensions: (states, actions)
    Q = np.loadtxt("q.txt", dtype='float', delimiter=" ")

    for epoch in range(100):
        done = False
        timestep = 0

        snake.x = 4*20 # Snake in top left corner
        snake.y = 4*20
        snake.dx = MOVEMENT_SPEED
        snake.dy = 0
        state = np.zeros(4)

        food.make(GRID_SIZE, SCALE, snake)


        if epoch % 10 == 0:
            print(epoch)

        start_time = pygame.time.get_ticks() 

        while not done:

            # timestep = 0

            state[0] = int(snake.x / SCALE)
            state[1] = int(snake.y / SCALE)
            state[2] = int(food.x / SCALE)
            state[3] = int(food.y / SCALE)

            # print(snake.x, snake.y)

            if np.random.rand() <= 0.0001:
                action = np.random.randint(4) # 4 actions: left, right, up, down
            else:
                # print(state_index(state))
                action = np.argmax(Q[state_index(state)])

            ai_controls(action, snake)

            if pygame.time.get_ticks() - start_time >= UPDATE_RATE:
                print(timestep)
                timestep += 1
                snake.update(SCALE, pygame, food, score)
                start_time = pygame.time.get_ticks()

            if ENABLE_WRAP:
                wrap(snake.x, snake.y)
            else:
                if snake.x > DISPLAY_WIDTH - SCALE:
                    done = True
                if snake.x < 0:
                    done = True
                if snake.y > DISPLAY_HEIGHT - SCALE:
                    done = True
                if snake.y < 0:
                    done = True
                    
            snake.box[0].topleft = (snake.x, snake.y)

            gameDisplay.blit(bg, (0, 0))
            # gameDisplay.fill(black)  #set background
            # gameDisplay.blit(FPS_text, (0, 0))
            # gameDisplay.blit(score_text, (DISPLAY_WIDTH - 150, 0))

            food.draw(gameDisplay)
            snake.draw(gameDisplay)

            pygame.display.set_caption("Moves: " + str(timestep))
            
            pygame.display.update()
            #print(clock.get_rawtime())
            clock.tick(FPS) # FPS setting
            #print(GAME_OVER)

            # timestep += 1
            # print(timestep)

            if timestep == 40:
                # timestep = 0
                done = True

            if not done:
                done = ((snake.x,snake.y)==(food.x, food.y))

            if done:
                print("timestep", timestep)


if __name__ == "__main__":
    snake.box[0].topleft = (snake.x, snake.y)
    food.make(GRID_SIZE, SCALE, snake)

    GAME_QUIT = False

    QLearn()

    # start_time = pygame.time.get_ticks() 

    # Qrun()

    ###############################################

    #Visualising the panda of the last game

    # df = pd.read_csv("./log_file.csv")
    # print(df)

    #Infinite game loop, until user presses exit

    #Context manager - opens and closes for you
    # with open("./log_file.csv", 'w') as f:
    #     csv_writer = csv.writer(f, delimiter=",") #default delimiter is a comma
    #     csv_writer.writerow(["TIME_STAMP", "SCORE", "SNAKE_X", "SNAKE_Y", "INPUT_BUTTON", "FOOD_X", "FOOD_Y"])

    #     start_time = pygame.time.get_ticks() 
    #     while True:       
    #         gameLoop(GAME_QUIT, csv_writer, start_time)

    ###############################################
    #fall back safety if something goes wrong
    pygame.quit()
    quit()



