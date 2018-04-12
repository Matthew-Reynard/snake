class Snake:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.dx = 0
        self.dy = 0
        
        self.head_img = None

        self.head_mask = None

        self.tail_img = None

        self.box = [(0,0)] # List of Rectangles tuples (x,y,width,height) for the snakes tail --> The main one
        # self.box[0] = self.head_img.get_rect()

        self.tail_length = 0 # basically len(snake.box) - 1

        # PYGAME STUFF

        # white = (255,255,255)
        # self.head_img = pygame.image.load("./Images/Head.png").convert()
        # self.head_img.set_colorkey(white) # sets white to alpha

        # self.head_mask = pygame.mask.from_surface(self.head_img) # creates a mask

        # # self.head_img = pygame.transform.flip(self.head_img, False, True) #
        # # self.head_img = pygame.transform.rotate(self.head_img, 90) # Start facing right

        # self.tail_img = pygame.image.load("./Images/Tail.png").convert()
        # self.tail_img.set_colorkey(white) # change white to alpha 

        # # If the images arent 20x20 pixels
        # # self.head_img = pygame.transform.scale(self.head_img, (20, 20)) # scales it down from a 50x50 image to 20x20
        # # self.tail_img = pygame.transform.scale(self.tail_img, (20, 20))

        # # self.box = [(0,0)] # List of Rectangles tuples (x,y,width,height) for the snakes tail --> The main one
        # self.box[0] = self.head_img.get_rect()


    def create(self, pygame):

        white = (255,255,255)
        self.head_img = pygame.image.load("./Images/Head.png").convert()
        self.head_img.set_colorkey(white) # sets white to alpha

        self.head_mask = pygame.mask.from_surface(self.head_img) # creates a mask

        # self.head_img = pygame.transform.flip(self.head_img, False, True) #
        # self.head_img = pygame.transform.rotate(self.head_img, 90) # Start facing right

        self.tail_img = pygame.image.load("./Images/Tail.png").convert()
        self.tail_img.set_colorkey(white) # change white to alpha 

        # If the images arent 20x20 pixels
        # self.head_img = pygame.transform.scale(self.head_img, (20, 20)) # scales it down from a 50x50 image to 20x20
        # self.tail_img = pygame.transform.scale(self.tail_img, (20, 20))

        # self.box = [(0,0)] # List of Rectangles tuples (x,y,width,height) for the snakes tail --> The main one
        self.box[0] = self.head_img.get_rect()




    # def update(self, scale, pygame, log_file, food, score):
    def update(self, scale, action):

        #Do nothing
        if action == 0:
            pass

        #Moving left
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


        #Moving right
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
            
        # print(self.x, self.y)
        self.x += self.dx * scale
        self.y += self.dy * scale
        # print(self.x, self.y)

    def draw(self, display):

        #Draw tail
        if self.tail_length > 0:
            for i in range(1, self.tail_length + 1):
                display.blit(self.tail_img, self.box[i].topleft)

        # Draw head after tail    
        # print("draw",self.box[0].topleft)    
        display.blit(self.head_img, self.box[0].topleft)