import random
import csv

class Obstacle:

    def __init__(self, size):
        
        self.obstacle_img = None

        # Positions array
        self.array = []

        self.array_length = size

        self.random_objects = []
    
    def create(self, pygame):
        """Set the image for the obstacles"""

        white = (255,255,255)
        self.obstacle_img = pygame.image.load("./Images/Obstacle.png").convert()
        self.obstacle_img.set_colorkey(white)

        for i in range(8):
            self.random_objects.append(pygame.image.load("./Images/Object{}.png".format(i+1)).convert())
            # self.random_objects[i].set_colorkey(white)


    def reset(self, grid, disallowed, num_of_obstacles):
        """Create all the obstacles, not in the disallowed positions"""
        # self.array.clear()
        random_array = []

        # If I want the obstacles in the same location every episode
        # random.seed(10)

        # Make a copy of the grid
        allowed = grid[:]

        [allowed.remove(pos) for pos in disallowed]

        for i in range(num_of_obstacles):
            new_pos = random.choice((allowed))
            self.array.append(new_pos)
            random_array.append(new_pos)
            allowed.remove(new_pos)

        self.array_length = self.array_length + num_of_obstacles

        return random_array

    def reset_map(self, grid_size, map_path, wrap):
        # self.array.clear()

        map1 = []
        
        # Read the map in from the text file
        with open(map_path, 'r') as csvfile:
            matrixreader = csv.reader(csvfile, delimiter=' ')
            if not wrap:
                row = []
                for i in range(grid_size):
                    row.insert(i,"0")
                map1.append(row)
                
                for row in matrixreader:
                    row.insert(grid_size, "0")
                    row.insert(0, "0")
                    map1.append(row)
                
                row = []
                for i in range(grid_size):
                    row.insert(i,"0")
                map1.append(row)
            else:
                for row in matrixreader:
                    map1.append(row)

        num = 0

        for j in range(grid_size):
            for i in range(grid_size):
                if map1[j][i] == '1':
                    self.array.append((i*20,j*20))
                    num +=1

        self.array_length = num

    def create_border(self, grid_size, scale):
        """Create all the obstacles, not in the disallowed positions"""
        # self.array.clear()

        for j in range(grid_size):
            for i in range(grid_size):
                if (i == 0 or i == grid_size-1) or (j == 0 or j == grid_size-1):
                    self.array.append((i*scale, j*scale))
                    self.array_length += 1

        # print(self.array)


    def draw(self, display):
        """Display all the obstacles"""
        # random.seed(0)

        for i in range(self.array_length):
            # random.seed(0)
            # n = random.randint(0,7)
            display.blit(self.obstacle_img, self.array[i])
            # display.blit(self.random_objects[n], self.array[i])
