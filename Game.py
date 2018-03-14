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
gameOverFont = pygame.font.SysFont('arial', 64)

#print(pygame.font.get_fonts())

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

ENABLE_WRAP = True
GAME_QUIT = False

white = (255,255,255)
black = (0,0,0)
red = (255,0,0)

FPS = 15 # changing the speed of the snake
score = 0
MOVEMENT_SPEED = 1
SCALE = 20

#Create and Initialise Snake 
snake = Snake(0,0,0,0)
#print(snake.x)

gameDisplay = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption('My Game')
clock = pygame.time.Clock()

#Starting position of the snake (top left corner)
snake.tail = [(DISPLAY_WIDTH * 0.0, DISPLAY_HEIGHT * 0.0)]

snake.head_img = pygame.image.load("./Images/Snake_Head.png").convert()
snake.head_img.set_colorkey(white)
snake.head_img = pygame.transform.scale(snake.head_img, (20, 20))
snake.head_mask = pygame.mask.from_surface(snake.head_img)

print("Red Mask is")
print(snake.head_mask)

snake.tail_img = pygame.image.load("./Images/Snake_Tail.png").convert()
snake.tail_img.set_colorkey(white)
snake.tail_img = pygame.transform.scale(snake.tail_img, (20, 20))
snake.mask = [pygame.mask.from_surface(snake.tail_img)]

blue_dot = pygame.image.load("./Images/BlueDot.png").convert()
blue_dot.set_colorkey(white)
blue_dot = pygame.transform.scale(blue_dot, (20, 20))
blue_mask = pygame.mask.from_surface(blue_dot)

#
snake.box = [(0,0)]
snake.box[0] = snake.head_img.get_rect()

blue_rect = blue_dot.get_rect()

def makeFood():
    made = False
    rows = DISPLAY_HEIGHT / SCALE
    cols = DISPLAY_WIDTH / SCALE

    while not made:
        myRow = random.randint(0, rows-1)
        myCol = random.randint(0, cols-1)
        #print(myRow * SCALE, myCol * SCALE)
        for n in snake.tail:
            print("x: " + str(n[0]) + "   y: " + str(n[1])) 

        blue_rect.topleft = (myCol * SCALE, myRow * SCALE)
        made = True


#red_rect.append((100,100))
#dots.append(red_rect[1])
#red_rect[1] = red_dot.get_rect()
#red_rect[1].topleft = (100,100)


#pygame.mouse.set_visible(False)

#dots.append((2,3))
#print(dots[1][1])

#Draw the snake
def DrawSnake():
    #Draw head
    gameDisplay.blit(snake.head_img, (snake.x, snake.y))
    #Draw tail
    if len(snake.tail) > 1:
        for i in range(len(snake.tail) - 1):
            gameDisplay.blit(snake.tail_img, snake.box[i].topleft)
            print(i)

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
        score_text = myfont.render("Score: " + str(score), True, white)  # anti aliasing
        makeFood()

        snake.tail_length += 1
        snake.tail_box.append(snake.tail_img.get_rect())
        snake.tail_box[snake.tail_length-1].topleft = (0,0)

        if snake.dx > 0: #RIGHT
            snake.box.append(snake.tail_img.get_rect())
            snake.tail.append(snake.box[len(snake.tail)-1])
            snake.box[len(snake.tail) - 1].topleft = tuple(np.subtract(snake.box[len(snake.tail) - 2].topleft, (20, 0)))

        if snake.dx < 0:#LEFT
            snake.box.append(snake.tail_img.get_rect())
            snake.tail.append(snake.box[len(snake.tail) - 1])
            snake.box[len(snake.tail) - 1].topleft = tuple(np.subtract(snake.box[len(snake.tail) - 2].topleft, (-20, 0)))

        if snake.dy > 0:#DOWN
            snake.box.append(snake.tail_img.get_rect())
            snake.tail.append(snake.box[len(snake.tail)-1])
            snake.box[len(snake.tail) - 1].topleft = tuple(np.subtract(snake.box[len(snake.tail) - 2].topleft, (0, 20)))

        if snake.dy < 0:#UP
            snake.box.append(snake.tail_img.get_rect())
            snake.tail.append(snake.box[len(snake.tail)-1])
            snake.box[len(snake.tail) - 1].topleft = tuple(np.subtract(snake.box[len(snake.tail) - 2].topleft, (0, -20)))


#Defines all the Players controls during the game
def controls(GAME_OVER):
    for event in pygame.event.get():
        #print(event)
        if event.type == pygame.QUIT:
            GAME_QUIT = True
            pygame.quit()
            quit()
            sys.exit(0) # safe backup

        if event.type == pygame.KEYDOWN:
            #print(event.key)

            #Moving left
            if (event.key == pygame.K_LEFT or event.key == pygame.K_a) and snake.dx == 0 and GAME_OVER == False:
                snake.dx = -MOVEMENT_SPEED
                snake.dy = 0

            #Moving right
            elif (event.key == pygame.K_RIGHT or event.key == pygame.K_d) and snake.dx == 0 and GAME_OVER == False:
                snake.dx = MOVEMENT_SPEED
                snake.dy = 0

            #Moving up
            elif (event.key == pygame.K_UP or event.key == pygame.K_w) and snake.dy == 0 and GAME_OVER == False:
                snake.dy = -MOVEMENT_SPEED
                snake.dx = 0

            #Moving down
            elif (event.key == pygame.K_DOWN or event.key == pygame.K_s) and snake.dy == 0 and GAME_OVER == False:
                snake.dy = MOVEMENT_SPEED
                snake.dx = 0

            #DEBUGGING - add to the snakes tail
            elif event.key == pygame.K_f and GAME_OVER == False: 
                snake.box.append(snake.tail_img.get_rect())
                snake.tail.append(snake.box[len(snake.tail)-1])
                snake.box[len(snake.tail) - 1].topleft = tuple(np.subtract(snake.box[len(snake.tail) - 2].topleft, (20, 0)))

            #GAME OVER: Press space to try play again
            elif (event.key == pygame.K_SPACE) and GAME_OVER == True:
                GAME_OVER = False

                snake.box.clear()
                snake.tail.clear()

                snake.box = [(0,0)]
                snake.tail = [(0,0)]

                snake.box[0] = snake.head_img.get_rect()
                print(snake.head_img.get_rect())
                snake.box[0].topleft = (snake.tail[0][0], snake.tail[0][1])
                makeFood()

        #if event.type == pygame.KEYUP:
            #if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                #dx = 0
            #if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                #dy = 0
            break

    return GAME_OVER

def gameLoop(GAME_QUIT):
    score = 0
    snake.dx = MOVEMENT_SPEED
    snake.dy = 0
    snake.x = 0
    snake.y = 0
    GAME_OVER = False

    print("In game loop")

    while not GAME_QUIT:
        #print("within the game loop")
        GAME_OVER = controls(GAME_OVER)

        #blue_rect.topleft = pygame.mouse.get_pos()

        if GAME_OVER:
            gameOver() 
        else:
            FPS_text = myfont.render("FPS: %0.3f" % clock.get_fps(), True, white)  # anti aliasing lowers performance
            score_text = myfont.render("Score: " + str(score), True, white)
            
            eatsFood(score)

            #gameUpdate()
            snake.update(SCALE)
            
            if snake.box[0].topleft != (snake.x, snake.y):

                if len(snake.tail) >= 1:
                    for x in range(len(snake.tail) - 1, 0, -1):
                        #print(x)
                        snake.box[x].topleft = snake.box[x-1].topleft

                if ENABLE_WRAP:
                    wrap(snake.x, snake.y)

                snake.box[0].topleft = (snake.x, snake.y)

            #if (red_mask.overlap(snake.mask[0], (0, 0)) != None):

                if len(snake.tail) > 1:
                    for i in range(1, len(snake.tail)-1, 1):
                        #print(range(1, len(snake.tail)-1, 1))
                        if(snake.box[0].colliderect(snake.box[i])):
                            GAME_OVER = True
                            print("Game Over")
                            print("Try again?")
                            #GAME_OVER = gameOverLoop()

            #DUPLICATE CODE FOR DEBUGGING                
            if False:
                if snake.tail_length > 1:
                    for i in range(snake.tail_length - 1, 0, -1):
                        #print(x)
                        snake.tail_box[i].topleft = snake.tail_box[i-1].topleft

                if ENABLE_WRAP:
                    wrap(snake.x, snake.y)

                snake.tail_box[0].topleft = (snake.x, snake.y)

                snake.update(SCALE)

                snake.head_box.topleft = (snake.x, snake.y)

                if len(snake.tail_length) > 1:
                    for i in range(1, snake.tail_length-1, 1):
                        #print(range(1, len(snake.tail)-1, 1))
                        if(snake.box[0].colliderect(snake.box[i])):
                            GAME_OVER = True
                            print("Game Over")
                            print("Try again?")
                            #GAME_OVER = gameOverLoop()

            gameDisplay.fill(black)  #set background
            gameDisplay.blit(FPS_text, (0, 0))
            gameDisplay.blit(score_text, (DISPLAY_WIDTH -150, 0))

            DrawBlueDot()
            DrawSnake()
            
        pygame.display.update()
        #print(clock.get_rawtime())
        clock.tick(FPS) # FPS setting
        #print(GAME_OVER)
                
        

def gameOver():

    gameDisplay.fill(black)  #set background
    game_over_text = gameOverFont.render("GAME OVER", True, white)
    try_again_text = myfont.render("Press SPACE to try again", True, white)
    gameDisplay.blit(game_over_text, (200, 200))
    gameDisplay.blit(try_again_text, (200, 400))

    snake.dx = MOVEMENT_SPEED
    snake.dy = 0
    snake.x = 0
    snake.y = 0

    #del snake.mask[:]

snake.box[0].topleft = (snake.tail[0][0], snake.tail[0][1])
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



