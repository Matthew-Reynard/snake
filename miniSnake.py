class Snake:

    def __init__(self, pygame):
        self.x = 0
        self.y = 0
        self.dx = 0
        self.dy = 0

        #TEST
        self.frame = 0
        self.right = True # goes right from the begineing
        self.left = False
        
        white = (255,255,255)
        self.head_img = pygame.image.load("./Images/Snake_Head.png").convert()
        self.head_img.set_colorkey(white) # sets white to alpha
        self.head_img = pygame.transform.scale(self.head_img, (50, 50)) # scales it down from a 50x50 image to 20x20
        self.head_mask = pygame.mask.from_surface(self.head_img) # creates a mask
        self.head_img = pygame.transform.flip(self.head_img, False, True) #
        self.head_img = pygame.transform.rotate(self.head_img, 90) # Start facing right

        self.tail_img = pygame.image.load("./Images/Snake_Tail.png").convert()
        self.tail_img.set_colorkey(white)
        self.tail_img = pygame.transform.scale(self.tail_img, (50, 50))

        self.box = [(0,0)] # List of Rectangles tuples (x,y,width,height) for the snakes tail --> The main one
        self.box[0] = self.head_img.get_rect()

        self.tail_length = 0 # basically len(snake.box) - 1

        self.SPEED = 1 # Meaning 1 block per timestep
        self.move = 0; #1 - Left; 2 - Right; 3 - Up; 4 - Down


    def update(self, scale, pygame, food, score):

        # TEST
        # To win the game
        # print(self.frame)
        
        # self.frame += 1

        # if self.right:
        #     if self.frame == 19:
        #         self.frame = -1
        #     if self.frame == -1:
        #         self.move = 4
        #     elif self.frame == 0:
        #         self.move = 1
        #         self.right = False
        #         self.left = True
        #     else:
        #         self.move = 0

        # elif self.left:
        #     if self.frame == 19:
        #         self.frame = -1
        #     if self.frame == -1:
        #         self.move = 4
        #     elif self.frame == 0:
        #         self.move = 2
        #         self.right = True
        #         self.left = False
        #     else:
        #         self.move = 0

        # else:
        #     print("Why am I here...")

        

        # self.x += self.dx * scale
        # self.y += self.dy * scale

        #Moving left
        if self.move == 1:
            self.dx = -self.SPEED
            if self.dy == 1:
                self.head_img = pygame.transform.rotate(self.head_img, -90)
            else:
                self.head_img = pygame.transform.rotate(self.head_img, 90)
            self.dy = 0
            self.move = 0

        #Moving right
        elif self.move == 2:
            self.dx = self.SPEED
            if self.dy == 1:
                self.head_img = pygame.transform.rotate(self.head_img, 90)
            else:
                self.head_img = pygame.transform.rotate(self.head_img, -90)
            self.dy = 0
            self.move = 0


        #Moving up
        elif self.move == 3:
            self.dy = -self.SPEED
            if self.dx == 1:
                self.head_img = pygame.transform.rotate(self.head_img, 90)
            else:
                self.head_img = pygame.transform.rotate(self.head_img, -90)
            self.dx = 0
            self.move = 0


        #Moving down
        elif self.move == 4:
            self.dy = self.SPEED
            if self.dx == 1:
                self.head_img = pygame.transform.rotate(self.head_img, -90)
            else:
                self.head_img = pygame.transform.rotate(self.head_img, 90)
            self.dx = 0
            self.move = 0

        self.x += self.dx * scale
        self.y += self.dy * scale

    def draw(self,display):

        #Draw tail
        if self.tail_length > 0:
            for i in range(1, self.tail_length + 1):
                display.blit(self.tail_img, self.box[i].topleft)

        # Draw head after tail        
        display.blit(self.head_img, self.box[0].topleft)