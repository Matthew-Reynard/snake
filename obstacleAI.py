import random
import csv

class Obstacle:

    def __init__(self, size):
        
        self.obstacle_img = None

        # Positions array
        self.array = []

        self.array_length = size
    
    def create(self, pygame):
        """Set the image for the obstacles"""

        white = (255,255,255)
        self.obstacle_img = pygame.image.load("./Images/Obstacle.png").convert()
        self.obstacle_img.set_colorkey(white)


    def reset(self, grid, disallowed):
        """Create all the obstacles, not in the disallowed positions"""
        self.array.clear()

        # If I want the obstacles in the same location every episode
        # random.seed(10)

        # Make a copy of the grid
        allowed = grid[:]

        [allowed.remove(pos) for pos in disallowed]

        for i in range(self.array_length):
            new_pos = random.choice((allowed))
            self.array.append(new_pos)
            allowed.remove(new_pos)

    def reset_map(self, grid_size, map_path):
        self.array.clear()

        map1 = []
        
        # Read the map in from the text file
        with open(map_path, 'r') as csvfile:
            matrixreader = csv.reader(csvfile, delimiter=' ')
            for row in matrixreader:
                map1.append(row)

        num = 0

        for j in range(grid_size):
            for i in range(grid_size):
                if map1[j][i] == '1':
                    self.array.append((i*20,j*20))
                    num = num + 1

        self.array_length = num


    def draw(self, display):
        """Display all the obstacles"""

        for i in range(self.array_length):
            display.blit(self.obstacle_img, self.array[i])