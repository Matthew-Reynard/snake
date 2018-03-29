



import random

class Food:

    def __init__(self, pygame):
        self.x = 0
        self.y = 0
        white = (255,255,255)
        self.dot = pygame.image.load("./Images/BlueDot.png").convert()
        self.dot.set_colorkey(white)
        self.dot = pygame.transform.scale(self.dot, (50, 50))
        self.mask = pygame.mask.from_surface(self.dot)

        self.rect = self.dot.get_rect()

#Load a food item into the screen at a random location
#Needs to be updated to not load onto the snake - NB
    def make(self, height, width, scale, snake):

        #So that it can be the same random numbers
        # random.seed(0)
        made = False

        rows = height / scale
        cols = width / scale

       
        # myRow = rows/2
        # myCol = cols/2

        while not made:
            myRow = random.randint(0, 5)
            myCol = random.randint(0, 5)

            if (snake.x, snake.y) != (myCol, myRow):
                made = True

        # print(myRow,myCol)
            #print(myRow * SCALE, myCol * SCALE) # DEBUGGING
        self.rect.topleft = (myCol * scale, myRow * scale)

        
        self.x = myCol * scale
        self.y = myRow * scale
    #Draw the food
    def draw(self, display):
        display.blit(self.dot, self.rect.topleft)
