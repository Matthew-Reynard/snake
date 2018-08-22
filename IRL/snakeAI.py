class Snake:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.dx = 0
        self.dy = 0
        
        self.head_img = None

        # self.head_mask = None

        self.tail_img = None

        # List of position tuples (x,y) for the snakes body 
        # self.box[0] => the head of the snake
        self.box = [(self.x, self.y)] 

        self.tail_length = 0 # basically len(self.box) - 1


    def create(self, pygame):

        # PYGAME STUFF

        white = (255,255,255)
        self.head_img = pygame.image.load("../Images/Head.png").convert()
        self.head_img.set_colorkey(white) # sets white to alpha

        self.tail_img = pygame.image.load("../Images/Tail.png").convert()
        self.tail_img.set_colorkey(white) # change white to alpha 

        # self.head_mask = pygame.mask.from_surface(self.head_img) # creates a mask

        # self.head_img = pygame.transform.flip(self.head_img, False, True) #
        # self.head_img = pygame.transform.rotate(self.head_img, 90) # Start facing right

        # If the images arent 20x20 pixels
        # self.head_img = pygame.transform.scale(self.head_img, (20, 20)) # scales it down from a 50x50 image to 20x20
        # self.tail_img = pygame.transform.scale(self.tail_img, (20, 20))

        # self.box = [(0,0)] # List of Rectangles tuples (x,y,width,height) for the snakes tail --> The main one
        # self.box[0] = self.head_img.get_rect()


    def update(self, scale, action, action_space):

        # ACTION SPACE OF 4 - Up, Down, Left, Right
        # This is for when just using the head and food

        # if action_space == 4:
        #     # Up
        #     if action == 0:
        #         if self.dy == 1:
        #             pass
        #         else:
        #             self.dy = -1 # move up
        #             self.dx = 0

        #     # Down
        #     elif action == 1:
        #         if self.dy == -1:
        #             pass
        #         else:
        #             self.dy = 1 # move down
        #             self.dx = 0

        #     # Left
        #     elif action == 2:
        #         if self.dx == 1:
        #             pass
        #         else:
        #             self.dx = -1 # move left
        #             self.dy = 0

        #     # Right
        #     elif action == 3:
        #         if self.dx == -1:
        #             pass
        #         else:
        #             self.dx = 1 # move right
        #             self.dy = 0

        if action_space == 4:
            # Up
            if action == 0:
                self.dy = -1 # move up
                self.dx = 0

            # Down
            elif action == 1:
                self.dy = 1 # move down
                self.dx = 0

            # Left
            elif action == 2:
                self.dx = -1 # move left
                self.dy = 0

            # Right
            elif action == 3:
                self.dx = 1 # move right
                self.dy = 0


        # ACTION SPACE OF 3 - Forward, Left, Right
        # This is for when using the tail
        # NOTE: when using the tail, the game knows what "forward" means, as the tail is behind

        if action_space == 3:
            # Moving forward / Do nothing
            if action == 0:
                pass

            # Moving left
            elif action == 1:

                # moving up or down
                if self.dy != 0:
                    # moving down
                    if self.dy == 1:
                        self.dx = 1 # move right
                    # moving up
                    elif self.dy == -1:
                        self.dx = -1 # move left
                    self.dy = 0

                # moving left or right
                elif self.dx != 0:
                    # moving left
                    if self.dx == -1:
                        self.dy = 1 # move down
                    # moving right
                    elif self.dx == 1:
                        self.dy = -1 # move up
                    self.dx = 0


            # Moving right
            elif action == 2:

                # moving up or down
                if self.dy != 0:
                    # moving down
                    if self.dy == 1:
                        self.dx = -1 # move left
                    # moving up
                    elif self.dy == -1:
                        self.dx = 1 # move right
                    self.dy = 0

                # moving left or right
                elif self.dx != 0:
                    # moving left
                    if self.dx == -1:
                        self.dy = -1 # move up
                    # moving right
                    elif self.dx == 1:
                        self.dy = 1 # move down
                    self.dx = 0
            
        # Updating positions using velocity
        self.x += self.dx * scale
        self.y += self.dy * scale

        # # Update the head position of the snake
        # self.box[0] = (self.x, self.y)

        # # Update the snakes tail positions (from back to front)
        # if self.tail_length > 0:
        #     for i in range(self.tail_length, 0, -1):
        #         self.box[i] = self.box[i-1]

    def draw(self, display):

        # Draw tail
        if self.tail_length > 0:
            for i in range(1, self.tail_length + 1):
                display.blit(self.tail_img, self.box[i])

        # Draw head after tail     
        display.blit(self.head_img, self.box[0])