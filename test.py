# Python
# import tensorflow as tf
import numpy as np
# from SnakeGame import Environment
import random


# #Creates tf constants. Not a normal Python variable
# hello = tf.constant('Hello, TensorFlow!')
# a = tf.constant(5)
# b = tf.constant(6)

# #tf.mul, tf.sub was updated/changed to tf.multiply, tf.subtract, etc.
# c = tf.multiply(a,b)

# #Prints out a Tensor - not the variable or answer 30
# #print(c)

# #sess = tf.Session()
# #print(sess.run(c))
# #sess.close()

# #Closes the Session automatically when your out of the "with" block
# with tf.Session() as sess:
# 	print(sess.run(hello))
# 	output = print(sess.run(c))

# #Can access this vaairble as it was a python variable created in the Session
# print(output)

# a = -0

# if a != -1:
# 	print("hi")
# else:
# 	print("no")

# Q = np.zeros((10*10*10*10, 4))

# print(Q.shape)

# class A:

# 	def __init__(self, a, b = 5):
# 		self.a = a
# 		self.b = b


# if __name__ == '__main__':
# 	x = A(2)
# 	print(x.a, x.b)

# env = Environment(True)

# for i in range(10):
# 	for j in range(10):
# 		for k in range(10):
# 			for l in range(10):
# 				s = [i,j,k,l]
# 				print(env.state_index(s))


# env.prerender()

# env.reset()

# done = False

# while not done:
# 	env.render()
# 	state, reward, done, time = env.step(env.sample(0))

# print(time)

# env.end()

# info = {"a":2, "b": 3}

# print(info["a"])

# total_episodes = 100
# epsilon = 1

# for episode in range(total_episodes):
# 	epsilon = (-0.9 / (0.2*total_episodes)) * episode + 1
# 	if epsilon < 0.1: 
# 		epsilon = 0.1
# 	print(epsilon)

# print(np.sqrt((0-9)**2 + (0-9)**2))

# import tensorflow as tf
# import numpy as np

# # x and y are placeholders for our training data
# x = tf.placeholder("float")
# y = tf.placeholder("float")
# # w is the variable storing our values. It is initialised with starting "guesses"
# # w[0] is the "a" in our equation, w[1] is the "b"
# w = tf.Variable([1.0, 2.0], name="w")
# # Our model of y = a*x + b
# y_model = tf.multiply(x, w[0]) + w[1]

# # Our error is defined as the square of the differences
# error = tf.square(y - y_model)
# # The Gradient Descent Optimizer does the heavy lifting
# train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

# # Normal TensorFlow - initialize values, create a session and run the model
# model = tf.global_variables_initializer()

# writer = tf.summary.FileWriter("./tmp/demo/1")

# with tf.Session() as session:

# 	session.run(model)

# 	writer.add_graph(session.graph)
	
# 	for i in range(1000):
# 		x_value = np.random.rand()
# 		y_value = x_value * 2 + 6
# 		session.run(train_op, feed_dict={x: x_value, y: y_value})

# 	w_value = session.run(w)
# 	print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))

# import time

# start = time.time()

# for i in range(1000000):

# 	t = 5000

# 	a = 0.23
# 	b = 1+2453
# 	afd = 1+2453
# 	agf = 1+245
# 	grea = 1+25
# 	argg= 1+246
# 	arge = 1+263

# 	if i % 100000 == 0:
# 		print("Time taken: {0:.0f}m {1:.2f}s".format(((t)/60), (t%60)))

# x = "\x00\t1203.6526"

# print(x[2:-2])

# print(2.345 % 1)

# action = np.random.randint(0,4)

# # action[0]

# print(action)

# folder = "CNN_VARIABLES"

# path = "CNN/" + folder + "/Data1"

# print(path)
# a = 1
# a = a + 1

# print(a)

Q = np.zeros((500000, 5))

np.savetxt("./tmp/irl.txt", Q.astype(np.float), fmt='%.2f', delimiter = " ")


# Q = np.loadtxt(Q_textfile_path_load, dtype='float', delimiter=" ")