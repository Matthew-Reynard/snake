class Snake:

    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        
        self.head_img = None
        self.head_mask = None # used for collision with food
        self.tail_img = None

        self.box = [] # List of Rectangles tuples (x,y,width,height) for the snakes tail --> The main one
        self.tail_length = 0 # basically len(snake.box) - 1

    def update(self, scale):
        self.x += self.dx * scale
        self.y += self.dy * scale
        #DEBUGGING
        #print("SNAKE POSITION UPDATED")
        #print("dx = " + str(self.dx))
        #print("dy = " + str(self.dy))
        #print(" ")

