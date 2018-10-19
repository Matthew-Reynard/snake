import random

class Obstacle:

    def __init__(self, size):
        
        self.obstacle_img = None

        # Positions array
        self.array = []

        self.array_length = size
    
    def create(self, pygame):
        """Set the image for the obstacles"""

        white = (255,255,255)
        self.obstacle_img = pygame.image.load("../../Images/Obstacle.png").convert()
        self.obstacle_img.set_colorkey(white)


    def reset(self, grid, disallowed):
        """Create all the obstacles, not in the disallowed positions"""

        # Make a copy of the grid
        allowed = grid[:]

        [allowed.remove(pos) for pos in disallowed]

        for i in range(self.array_length):
            new_pos = random.choice((allowed))
            self.array.append(new_pos)
            allowed.remove(new_pos)


    def draw(self, display):
        """Display all the obstacles"""

        for i in range(self.array_length):
            display.blit(self.obstacle_img, self.array[i])