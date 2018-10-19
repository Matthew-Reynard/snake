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

    def reset(self, grid, disallowed):
        self.array.clear()

        # Make a copy of the grid
        allowed = grid[:]

        [allowed.remove(pos) for pos in disallowed]

        for i in range(self.amount):
            new_pos = random.choice((allowed))
            self.array.append(new_pos)
            allowed.remove(new_pos)


    # Load a food item into the screen at a random location
    def make(self, grid, snake, disallowed, index = 0):
        
        # Make a copy of the grid
        allowed = grid[:]

        [disallowed.append(grid_pos) for grid_pos in snake.box[1:]]

        [disallowed.append(grid_pos) for grid_pos in self.array]

        [allowed.remove(pos) for pos in disallowed]

        self.array[index] = random.choice((allowed))

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
