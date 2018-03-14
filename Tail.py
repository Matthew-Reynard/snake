class Snake:

    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.head_img = None
        self.head_mask = None
        self.head_box = None
        self.tail_img = None
        self.tail_box = [] # List of Rectangles tuples (x,y,width,height) for the snakes tail
        self.tail = [] # Tail positions
        self.mask = []
        self.box = [] # Tail rectangles used for collision
        self.tail_length = 0

    def update(self, scale):
        self.x += self.dx * scale
        self.y += self.dy * scale
        #print("SNAKE POSITION UPDATED")
        #print("dx = " + str(self.dx))
        #print("dy = " + str(self.dy))
        #print(" ")

