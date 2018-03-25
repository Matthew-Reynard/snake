'''

Simple Snake Game

@author: Matthew Reynard
@year: 2018

Purpose: Use Tensorflow to create a RL environment to train this snake to beat the game, i.e. have 1200-1 tail sections
Start Date: 5 March 2018
Due Date: Before June

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

pygame.init()

pygame.font.init() # you have to call this at the start, if you want to use this module.
myfont = pygame.font.SysFont('arial', 30)
gameOverFont = pygame.font.SysFont('arial', 96)

#print(pygame.font.get_fonts())

#Window size
DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 400
# DISPLAY_WIDTH = 800
# DISPLAY_HEIGHT = 600

#Wrap allows snake to wrap around and not crash into the edge of the screen
ENABLE_WRAP = True
GAME_QUIT = False

#Various RGB colours
white = (255,255,255)
black = (0,0,0)
red = (255,0,0)

FPS = 120 # If not set, game tries to max out FPS (last test was 1200 FPS)
score = 0
MOVEMENT_SPEED = 1 # Meaning 1 block per timestep
SCALE = 20 # Block/Grid size (in pixels)
UPDATE_RATE = 80 # lower the faster

gameDisplay = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption('My Game')
clock = pygame.time.Clock()

#Create and Initialise Snake 
snake = Snake(pygame)

#Create Food 
food = Food(pygame)

#pygame.mouse.set_visible(False)

#If the snake goes out of the screen bounds, wrap it around
def wrap(x,y):
    if snake.x > DISPLAY_WIDTH - 20:
        snake.x = 0;
    if snake.x < 0:
        snake.x = DISPLAY_WIDTH - 20;
    if snake.y > DISPLAY_HEIGHT - 20:
        snake.y = 0;
    if snake.y < 0:
        snake.y = DISPLAY_HEIGHT - 20;

#If the red mask overlaps the food mask, increment score and increase snake tail length
def eatsFood(score):

    offset_x, offset_y = (food.rect.left - snake.box[0].left), (food.rect.top - snake.box[0].top)

    if (snake.head_mask.overlap(food.mask, (offset_x, offset_y)) != None):

        score += 1
        food.make(DISPLAY_HEIGHT, DISPLAY_WIDTH, SCALE, snake)

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

                log_file.writerow(["---NEW GAME--"])
                log_file.writerow(["TIME_STAMP", "SCORE", "SNAKE_X", "SNAKE_Y", "INPUT_BUTTON", "FOOD_X", "FOOD_Y"])

                if snake.dx == 1 and snake.dy == 0:
                    #print("going R")
                    pass
                elif snake.dx == -1 and snake.dy == 0:
                    #print("going L")
                    snake.head_img = pygame.transform.rotate(snake.head_img, 180)
                elif snake.dy == 1 and snake.dx == 0:
                    #print("going D")
                    snake.head_img = pygame.transform.rotate(snake.head_img, 90)
                elif snake.dy == -1 and snake.dx == 0:
                   #print("going U")
                    snake.head_img = pygame.transform.rotate(snake.head_img, -90)

                snake.dx = MOVEMENT_SPEED # Initilise the snake moving to the left
                snake.dy = 0

                #print(snake.dx, snake.dy)
                snake.tail_length = 0
                snake.box.clear()

                snake.box = [(0,0)] # Not sure if this is neccessary

                snake.box[0] = snake.head_img.get_rect()
                snake.box[0].topleft = (0, 0)
                food.make(DISPLAY_HEIGHT, DISPLAY_WIDTH, SCALE, snake)

            elif (event.key == pygame.K_q) and GAME_OVER == True:
                log_file.writerow(["---END---"])
                GAME_QUIT = True # Not working
                pygame.quit()
                quit()
                sys.exit(0) # safe backup   

        #NOT USED
        #if event.type == pygame.KEYUP:
            #if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                #dx = 0
            #if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                #dy = 0

            #break is ot make sure that while the game is "paused" or "waiting" for the clock fps cycle to finish (since its a low fps and the player can easily push multiple
            #commands each frame) the game only reads in the first command each frame (can make it the last, buts thats an effort)
            break

    return GAME_OVER


#Main game loop
def gameLoop(GAME_QUIT, log_file, start_time):

    #initialize starting variable to ensure that the players can retry after a GAME OVER
    score = 0
    snake.dx = MOVEMENT_SPEED # Initilise the snake moving to the left
    snake.dy = 0
    snake.x = 0 # Snake in top left corner
    snake.y = 0
    GAME_OVER = False

    #print("In game loop") # DEBUGGING

    while not GAME_QUIT:
        #print("Within the game loop") # DEBUGGING

        #print(pygame.time.get_ticks())
        #Players controls while in the game loop
        GAME_OVER = controls(GAME_OVER, log_file)

        if GAME_OVER:
            #Game over screen
            score = gameOver(score) 
        else:
            #In game screen
            FPS_text = myfont.render("FPS: %0.3f" % clock.get_fps(), True, white)  # anti aliasing lowers performance
            
            #Checks if the snake has eaten the food
            score = eatsFood(score)
            score_text = myfont.render("Score: " + str(score), True, white)

            #Update the snakes positioning
            #print("Start time:", start_time, "        Time elapsed:" , start_time - pygame.time.get_ticks())
            if pygame.time.get_ticks() - start_time>= UPDATE_RATE:
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
                    if snake.x > DISPLAY_WIDTH - 20:
                        GAME_OVER=True
                    if snake.x < 0:
                        GAME_OVER=True
                    if snake.y > DISPLAY_HEIGHT - 20:
                        GAME_OVER=True
                    if snake.y < 0:
                        GAME_OVER=True

                #Update the head position of the snake
                snake.box[0].topleft = (snake.x, snake.y)

                #If you think about it, the snake can only start crashing into itself after the tail length it greater than 3
                if snake.tail_length >= 3:
                    for i in range(1, snake.tail_length):
                        if(snake.box[0].colliderect(snake.box[i])):
                            GAME_OVER = True
                            #DEBUGGING
                            #print("Game Over")
                            #print("Try again?")

            gameDisplay.fill(black)  #set background
            gameDisplay.blit(FPS_text, (0, 0))
            gameDisplay.blit(score_text, (DISPLAY_WIDTH - 150, 0))

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
    game_over_text = gameOverFont.render("GAME OVER", True, white)
    try_again_text = myfont.render("Press SPACE to Try Again", True, white)
    quit_text = myfont.render("Press Q to Quit", True, white)

    gameDisplay.blit(game_over_text, ((int)(DISPLAY_WIDTH*0.1), (int)(DISPLAY_HEIGHT*0.2))) # 120, 200
    gameDisplay.blit(try_again_text, ((int)(DISPLAY_WIDTH*0.25), (int)(DISPLAY_HEIGHT*0.55))) #235 400
    gameDisplay.blit(quit_text,  ((int)(DISPLAY_WIDTH*0.35), (int)(DISPLAY_HEIGHT*0.65))) #300 ,450

    score = 0
    snake.x = 0
    snake.y = 0
    #snake.dx = MOVEMENT_SPEEDs
    #snake.dy = 0
    return score
    

snake.box[0].topleft = (snake.x, snake.y)
food.make(DISPLAY_HEIGHT, DISPLAY_WIDTH, SCALE, snake)

GAME_QUIT = False

###############################################

#Visualising the panda of the last game
df = pd.read_csv("./log_file.csv")
print(df)

#Infinite game loop, until user presses exit

#Context manager - opens and closes for you
with open("./log_file.csv", 'w') as f:
    csv_writer = csv.writer(f, delimiter=",") #default delimiter is a comma
    csv_writer.writerow(["TIME_STAMP", "SCORE", "SNAKE_X", "SNAKE_Y", "INPUT_BUTTON", "FOOD_X", "FOOD_Y"])

    start_time = pygame.time.get_ticks() 
    while True:       
        gameLoop(GAME_QUIT, csv_writer, start_time)

###############################################
#fall back safety if something goes wrong
pygame.quit()
quit()



