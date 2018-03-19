



import random

class Food:

    def __init__(self, pygame):
        self.x = 0
        self.y = 0
        white = (255,255,255)
        self.dot = pygame.image.load("./Images/BlueDot.png").convert()
        self.dot.set_colorkey(white)
        self.dot = pygame.transform.scale(self.dot, (20, 20))
        self.mask = pygame.mask.from_surface(self.dot)

        self.rect = self.dot.get_rect()

#Load a food item into the screen at a random location
#Needs to be updated to not load onto the snake - NB
    def make(self, height, width, scale, snake):
        made = False
        rows = height / scale
        cols = width / scale

        while not made:
            myRow = random.randint(0, rows-1)
            myCol = random.randint(0, cols-1)
            #print(myRow * SCALE, myCol * SCALE) # DEBUGGING
            self.rect.topleft = (myCol * scale, myRow * scale)

            for i in range(0, snake.tail_length + 1):
                if self.rect.topleft == snake.box[i].topleft:
                    made = False
                    break
                else:
                    self.x = myCol * scale
                    self.y = myRow * scale
                    made = True # the food is not within the snakes body
    #Draw the food
    def draw(self, display):
        display.blit(self.dot, self.rect.topleft)
