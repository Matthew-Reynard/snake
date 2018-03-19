class Snake:

    def __init__(self, pygame):
        self.x = 0
        self.y = 0
        self.dx = 0
        self.dy = 0
        
        white = (255,255,255)
        self.head_img = pygame.image.load("./Images/Snake_Head.png").convert()
        self.head_img.set_colorkey(white) # sets white to alpha
        self.head_img = pygame.transform.scale(self.head_img, (20, 20)) # scales it down from a 50x50 image to 20x20
        self.head_mask = pygame.mask.from_surface(self.head_img) # creates a mask
        self.head_img = pygame.transform.flip(self.head_img, False, True) #
        self.head_img = pygame.transform.rotate(self.head_img, 90) # Start facing right

        self.tail_img = pygame.image.load("./Images/Snake_Tail.png").convert()
        self.tail_img.set_colorkey(white)
        self.tail_img = pygame.transform.scale(self.tail_img, (20, 20))

        self.box = [(0,0)] # List of Rectangles tuples (x,y,width,height) for the snakes tail --> The main one
        self.box[0] = self.head_img.get_rect()

        self.tail_length = 0 # basically len(snake.box) - 1

        self.SPEED = 1 # Meaning 1 block per timestep
        self.move = 0; #1 - Left; 2 - Right; 3 - Up; 4 - Down


    def update(self, scale, pygame, log_file, food, score):
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
            #log_file.write("A\n")
            log_file.writerow([pygame.time.get_ticks(), str(score), str(self.x), str(self.y), "A", str(food.x), str(food.y)])

        #Moving right
        elif self.move == 2:
            self.dx = self.SPEED
            if self.dy == 1:
                self.head_img = pygame.transform.rotate(self.head_img, 90)
            else:
                self.head_img = pygame.transform.rotate(self.head_img, -90)
            self.dy = 0
            self.move = 0
            #log_file.write("D\n")
            log_file.writerow([pygame.time.get_ticks(), str(score), str(self.x), str(self.y), "D", str(food.x), str(food.y)])


        #Moving up
        elif self.move == 3:
            self.dy = -self.SPEED
            if self.dx == 1:
                self.head_img = pygame.transform.rotate(self.head_img, 90)
            else:
                self.head_img = pygame.transform.rotate(self.head_img, -90)
            self.dx = 0
            self.move = 0
            #log_file.write("W\n")
            log_file.writerow([pygame.time.get_ticks(), str(score), str(self.x), str(self.y), "W", str(food.x), str(food.y)])


        #Moving down
        elif self.move == 4:
            self.dy = self.SPEED
            if self.dx == 1:
                self.head_img = pygame.transform.rotate(self.head_img, -90)
            else:
                self.head_img = pygame.transform.rotate(self.head_img, 90)
            self.dx = 0
            self.move = 0
            #log_file.write("S\n")
            log_file.writerow([pygame.time.get_ticks(), str(score), str(self.x), str(self.y), "S", str(food.x), str(food.y)])

        self.x += self.dx * scale
        self.y += self.dy * scale

    def draw(self,display):

        #Draw tail
        if self.tail_length > 0:
            for i in range(1, self.tail_length + 1):
                display.blit(self.tail_img, self.box[i].topleft)

        # Draw head after tail        
        display.blit(self.head_img, self.box[0].topleft)