#######################################################
#
# Simple Snake Game
#
# Author: Matthew
# Purpose: Use Tensorflow to create a RL environment to train this snake to beat the game, i.e. have 1200-1 tail sections
# Start Date: 5 March 2018
# Due Date: Before June
#
#
# 1200-1 number comes from 800/20 = 40; 600/20 = 30; 40*30 = 1200 grid blocks; subtract one for the "head"
#######################################################
import numpy as np
import pygame
from Tail import Snake
import random
import sys

pygame.init()

pygame.font.init() # you have to call this at the start, if you want to use this module.
myfont = pygame.font.SysFont('arial', 30)
gameOverFont = pygame.font.SysFont('arial', 96)

#print(pygame.font.get_fonts())

#Window size
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

#Wrap allows snake to wrap around and not crash into the edge of the screen
ENABLE_WRAP = True
GAME_QUIT = False

#Various RGB colours
white = (255,255,255)
black = (0,0,0)
red = (255,0,0)

FPS = 15 # changing the speed of the snake
score = 0
MOVEMENT_SPEED = 1
SCALE = 20 # Block/Grid size (in pixels)

#Create and Initialise Snake 
snake = Snake(0,0,0,0)
#print(snake.x)

gameDisplay = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption('My Game')
clock = pygame.time.Clock()

#Starting position of the snake (top left corner)
snake.tail = [(DISPLAY_WIDTH * 0.0, DISPLAY_HEIGHT * 0.0)]

snake.head_img = pygame.image.load("./Images/Snake_Head.png").convert()
snake.head_img.set_colorkey(white) # sets white to alpha
snake.head_img = pygame.transform.scale(snake.head_img, (20, 20)) # scales it down from a 50x50 image to 20x20
snake.head_mask = pygame.mask.from_surface(snake.head_img) # creates a mask
snake.head_img = pygame.transform.flip(snake.head_img, False, True) #
snake.head_img = pygame.transform.rotate(snake.head_img, 90) # Start facing right

snake.tail_img = pygame.image.load("./Images/Snake_Tail.png").convert()
snake.tail_img.set_colorkey(white)
snake.tail_img = pygame.transform.scale(snake.tail_img, (20, 20))

blue_dot = pygame.image.load("./Images/BlueDot.png").convert()
blue_dot.set_colorkey(white)
blue_dot = pygame.transform.scale(blue_dot, (20, 20))
blue_mask = pygame.mask.from_surface(blue_dot)

snake.box = [(0,0)]
snake.box[0] = snake.head_img.get_rect()

blue_rect = blue_dot.get_rect()

#Load a food item into the screen at a random location
#Needs to be updated to not load onto the snake - NB
def makeFood():
    made = False
    rows = DISPLAY_HEIGHT / SCALE
    cols = DISPLAY_WIDTH / SCALE

    while not made:
        myRow = random.randint(0, rows-1)
        myCol = random.randint(0, cols-1)
        #print(myRow * SCALE, myCol * SCALE) # DEBUGGING

        #DEBUGGING
        #for n in snake.tail:
        #    print("x: " + str(n[0]) + "   y: " + str(n[1])) 

        blue_rect.topleft = (myCol * SCALE, myRow * SCALE)
        made = True # used for the updated function - NOT YET IMPLEMENTED

#pygame.mouse.set_visible(False)

#Draws the snake
def DrawSnake():
    #Draw head
    #gameDisplay.blit(snake.head_img, (snake.x, snake.y))
    gameDisplay.blit(snake.head_img, snake.box[0].topleft)
    #Draw tail
    if snake.tail_length > 0:
        for i in range(1, snake.tail_length + 1):
            gameDisplay.blit(snake.tail_img, snake.box[i].topleft)
            #print(i) #DEBUGGING

#Draw the food
def DrawBlueDot():
    gameDisplay.blit(blue_dot, blue_rect.topleft)

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

#Update the snakes position
def gameUpdate():
    snake.x += snake.dx*SCALE
    snake.y += snake.dy*SCALE


#If the red mask overlaps the blue mask, increment score and increase snake tail length
def eatsFood(score):

    offset_x, offset_y = (blue_rect.left - snake.box[0].left), (blue_rect.top - snake.box[0].top)

    if (snake.head_mask.overlap(blue_mask, (offset_x, offset_y)) != None):

        score += 1
        makeFood()

        snake.tail_length += 1
        snake.box.append(snake.tail_img.get_rect()) #adds a rectangle variable to snake.box array(list)

        if snake.dx > 0: #RIGHT
            snake.box[snake.tail_length].topleft = tuple(np.subtract(snake.box[snake.tail_length - 1].topleft, (20, 0)))

        if snake.dx < 0:#LEFT
            snake.box[snake.tail_length].topleft = tuple(np.subtract(snake.box[snake.tail_length - 1].topleft, (-20, 0)))

        if snake.dy > 0:#DOWN
            snake.box[snake.tail_length].topleft = tuple(np.subtract(snake.box[snake.tail_length - 1].topleft, (0, 20)))

        if snake.dy < 0:#UP
            snake.box[snake.tail_length].topleft = tuple(np.subtract(snake.box[snake.tail_length - 1].topleft, (0, -20)))

    return score


#Defines all the Players controls during the game
def controls(GAME_OVER):
    for event in pygame.event.get():
        #print(event) # DEBUGGING
        if event.type == pygame.QUIT:
            GAME_QUIT = True
            pygame.quit()
            quit()
            sys.exit(0) # safe backup

        if event.type == pygame.KEYDOWN:
            #print(event.key) #DEBUGGING

            #Moving left
            if (event.key == pygame.K_LEFT or event.key == pygame.K_a) and snake.dx == 0 and GAME_OVER == False:
                snake.dx = -MOVEMENT_SPEED
                if snake.dy == 1:
                    snake.head_img = pygame.transform.rotate(snake.head_img, -90)
                else:
                    snake.head_img = pygame.transform.rotate(snake.head_img, 90)
                snake.dy = 0

            #Moving right
            elif (event.key == pygame.K_RIGHT or event.key == pygame.K_d) and snake.dx == 0 and GAME_OVER == False:
                snake.dx = MOVEMENT_SPEED
                if snake.dy == 1:
                    snake.head_img = pygame.transform.rotate(snake.head_img, 90)
                else:
                    snake.head_img = pygame.transform.rotate(snake.head_img, -90)
                snake.dy = 0

            #Moving up
            elif (event.key == pygame.K_UP or event.key == pygame.K_w) and snake.dy == 0 and GAME_OVER == False:
                snake.dy = -MOVEMENT_SPEED
                if snake.dx == 1:
                    snake.head_img = pygame.transform.rotate(snake.head_img, 90)
                else:
                    snake.head_img = pygame.transform.rotate(snake.head_img, -90)
                snake.dx = 0

            #Moving down
            elif (event.key == pygame.K_DOWN or event.key == pygame.K_s) and snake.dy == 0 and GAME_OVER == False:
                snake.dy = MOVEMENT_SPEED
                if snake.dx == 1:
                    snake.head_img = pygame.transform.rotate(snake.head_img, -90)
                else:
                    snake.head_img = pygame.transform.rotate(snake.head_img, 90)
                snake.dx = 0

            #DEBUGGING - add to the snakes tail
            elif event.key == pygame.K_f and GAME_OVER == False: 
                #score += 1
                #score_text = myfont.render("Score: " + str(score), True, white)  # anti aliasing

                snake.tail_length += 1
                snake.box.append(snake.tail_img.get_rect()) #adds a rectangle variable to snake.box array(list)
                snake.box[snake.tail_length].topleft = tuple(np.subtract(snake.box[snake.tail_length - 1].topleft, (20, 0)))

            #GAME OVER: Press space to try play again
            elif (event.key == pygame.K_SPACE) and GAME_OVER == True:
                GAME_OVER = False

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
                makeFood()

            elif (event.key == pygame.K_q) and GAME_OVER == True:
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
def gameLoop(GAME_QUIT):

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

        #Players controls while in the game loop
        GAME_OVER = controls(GAME_OVER)

        #blue_rect.topleft = pygame.mouse.get_pos()

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
            #gameUpdate()
            snake.update(SCALE)
            
            #Update the snakes tail positions (from back to front)
            if snake.box[0].topleft != (snake.x, snake.y): # if the snake has moved

                if snake.tail_length > 0:
                    for i in range(snake.tail_length, 0, -1):
                        #print(i)
                        snake.box[i].topleft = snake.box[i-1].topleft

                if ENABLE_WRAP:
                    wrap(snake.x, snake.y)

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

            DrawBlueDot()
            DrawSnake()
            
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

    gameDisplay.blit(game_over_text, (120, 200))
    gameDisplay.blit(try_again_text, (235, 400))
    gameDisplay.blit(quit_text, (300, 450))

    score = 0
    snake.x = 0
    snake.y = 0
    #snake.dx = MOVEMENT_SPEED
    #snake.dy = 0
    return score
    

snake.box[0].topleft = (snake.x, snake.y)
makeFood()

GAME_QUIT = False
###############################################

#Infinite game loop, until user presses exit
while True:       
    gameLoop(GAME_QUIT)

###############################################
#fall back safety if something goes wrong
pygame.quit()
quit()



