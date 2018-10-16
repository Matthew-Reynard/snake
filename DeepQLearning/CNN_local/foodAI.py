# Maybe can use numpy.random instead of this
import random

class Food:

    def __init__(self, number = 1):
        self.x = 0
        self.y = 0

        self.food_img = None

        # self.mask = None

        # self.rect = None

        self.pos = (self.x, self.y)

        self.array = [(self.x, self.y)]
        self.amount = number

    # Create the Pygame sections of the food to render it
    def create(self, pygame):

        # PYGAME STUFF

        white = (255,255,255)
        self.food_img = pygame.image.load("../../Images/Food.png").convert()
        self.food_img.set_colorkey(white)

        # self.mask = pygame.mask.from_surface(self.food_img)

        # If the image isn't 20x20 pixels
        # self.food_img = pygame.transform.scale(self.food_img, (20, 20))

        # self.rect = self.food_img.get_rect()

    def reset(self, grid_size, scale, snake):
        self.array.clear()

        for i in range(self.amount):
            made = False
            while not made:
                different = True
                myRow = random.randint(0, grid_size-1)
                myCol = random.randint(0, grid_size-1)

                self.pos = (myCol * scale, myRow * scale) # multiplying by scale

                if i > 0:
                    for n in range(len(self.array)):
                        # print(n)
                        # print(n,len(self.array))
                        if self.pos == self.array[n]:
                            # print(self.pos, self.array[n])
                            different = False
                            break

                if different:
                    for k in range(0, snake.tail_length + 1):
                        # print("making food")
                        # Need to change this to the whole body of the snake
                        if self.pos == snake.box[k]:
                        # if self.pos == (snake.x, snake.y):
                            made = False # the food IS within the snakes body
                            break
                        else:
                            self.x = self.pos[0]
                            self.y = self.pos[1]
                            self.array.append((self.x, self.y))
                            made = True # the food IS NOT within the snakes body

        # print(self.array)


    # Load a food item into the screen at a random location
    def make(self, grid_size, scale, snake, index = 0):
        made = False
        while not made:
            different = True
            myRow = random.randint(0, grid_size-1)
            myCol = random.randint(0, grid_size-1)

            # Making the food only in one position - Test 1
            # myRow = 3
            # myCol = 3

            # Making the food only in one of three positions - Test 2
            # r = random.randint(0,2)
            # if r == 0:
            #     myRow = 1
            #     myCol = 5
            # if r == 1:
            #     myRow = 6
            #     myCol = 6
            # if r == 2:
            #     myRow = 5
            #     myCol = 1

            self.pos = (myCol * scale, myRow * scale) # multiplying by scale

            for i in range(self.amount):
                if i != index:
                    if self.pos == self.array[i]:
                        different = False
                        break

            if different:
                for i in range(0, snake.tail_length + 1):
                    # print("making food")
                    # Need to change this to the whole body of the snake
                    if self.pos == snake.box[i]:
                    # if self.pos == (snake.x, snake.y):
                        made = False # the food IS within the snakes body
                        break
                    else:
                        self.x = myCol * scale
                        self.y = myRow * scale
                        self.array[index] = (self.x, self.y)
                        made = True # the food IS NOT within the snakes body
            else:
                pass
                # print("not different")

    # make a piece of food within the local grid
    def make_within_range(self, grid_size, scale, snake, local_grid_size = 3):
        made = False
        rows = grid_size
        cols = grid_size

        while not made:
            myRow = random.randint(0, rows-1)
            myCol = random.randint(0, cols-1)

            self.pos = (myCol * scale, myRow * scale) # multiplying by scale

            for i in range(0, snake.tail_length + 1):
                # print("making food")
                # Need to change this to the whole body of the snake
                if self.pos == snake.box[i]:
                    made = False # the food IS within the snakes body
                    break
                elif abs(snake.x - self.pos[0])/scale <= local_grid_size and abs(snake.y - self.pos[1])/scale <= local_grid_size:
                    self.x = myCol * scale
                    self.y = myRow * scale
                    made = True # the food IS NOT within the snakes body and within range

    #Draw the food
    def draw(self, display):
        for i in range(self.amount):
            display.blit(self.food_img, self.array[i])
