import random

class Obstacle:

    def __init__(self, size):
        self.x = 0
        self.y = 0
        
        self.obstacle_img = None

        self.pos = (self.x, self.y)

        self.array = [(self.x, self.y)]

        self.array_length = size

       


    def create(self, pygame):

        # PYGAME STUFF

        white = (255,255,255)
        self.obstacle_img = pygame.image.load("./Images/Obstacle.png").convert()
        self.obstacle_img.set_colorkey(white)

       
    def make(self, grid_size, scale):

        random.seed(10)
        self.array[0] = (random.randint(1,grid_size-2)*scale, random.randint(1,grid_size-2)*scale)

        for i in range(1, self.array_length):
            self.array.append((random.randint(1,grid_size-2)*scale, random.randint(1,grid_size-2)*scale))


        # made = False
        # rows = grid_size
        # cols = grid_size

        # if num == 0:
        #     random.seed(0)

        # while not made:
        #     myRow = random.randint(1, rows-2)
        #     myCol = random.randint(1, cols-2)

        #     # Making the food only in one position - Test 1
        #     # myRow = 3
        #     # myCol = 3

        #     # Making the food only in one of three positions - Test 2
        #     # r = random.randint(0,2)
        #     # if r == 0:
        #     #     myRow = 1
        #     #     myCol = 5
        #     # if r == 1:
        #     #     myRow = 6
        #     #     myCol = 6
        #     # if r == 2:
        #     #     myRow = 5
        #     #     myCol = 1

        #     self.pos = (myCol * scale, myRow * scale) # multiplying by scale

        #     for i in range(0, snake.tail_length + 1):
        #         # print("making food")
        #         # Need to change this to the whole body of the snake
        #         if self.pos == snake.box[i]:
        #         # if self.pos == (snake.x, snake.y):
        #             made = False # the food IS within the snakes body
        #             break
        #         else:
        #             self.x = myCol * scale
        #             self.y = myRow * scale
        #             made = True # the food IS NOT within the snakes body

    def draw(self, display):

        for i in range(self.array_length):
            display.blit(self.obstacle_img, self.array[i])
        # display.blit(self.obstacle_img, self.pos)