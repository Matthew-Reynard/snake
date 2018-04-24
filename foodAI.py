# Maybe can use numpy.random instead of this
import random

class Food:

    def __init__(self):
        self.x = 0
        self.y = 0

        self.food_img = None

        # self.mask = None

        # self.rect = None

        self.pos = (self.x, self.y)

    # Create the Pygame sections of the food to render it
    def create(self, pygame):

        # PYGAME STUFF

        white = (255,255,255)
        self.food_img = pygame.image.load("./Images/Food.png").convert()
        self.food_img.set_colorkey(white)

        # self.mask = pygame.mask.from_surface(self.food_img)

        # If the image isn't 20x20 pixels
        # self.food_img = pygame.transform.scale(self.food_img, (20, 20))

        # self.rect = self.food_img.get_rect()

    #Load a food item into the screen at a random location
    def make(self, grid_size, scale, snake):
        made = False
        rows = grid_size
        cols = grid_size

        while not made:
            # myRow = random.randint(0, rows-1)
            # myCol = random.randint(0, cols-1)

            # Making the food only in one position - Test 1
            myRow = 6
            myCol = 6

            # Making the food only in one of three positions - Test 2
            # r = random.randint(0,2)
            # if r == 0:
            #     myRow = 1
            #     myCol = 9
            # if r == 1:
            #     myRow = 8
            #     myCol = 8
            # if r == 2:
            #     myRow = 9
            #     myCol = 1

            self.pos = (myCol * scale, myRow * scale) # multiplying by scale

            for i in range(0, snake.tail_length + 1):
                # print("making food")
                # Need to change this to the whole body of the snake
                # if self.pos == snake.box[i]:
                if self.pos == (snake.x, snake.y):
                    made = False # the food IS within the snakes body
                    break
                else:
                    self.x = myCol * scale
                    self.y = myRow * scale
                    made = True # the food IS NOT within the snakes body

    #Draw the food
    def draw(self, display):
        display.blit(self.food_img, self.pos)
