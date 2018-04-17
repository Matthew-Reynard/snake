# Python
# import tensorflow as tf
import numpy as np
# from SnakeGame import Environment


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

print(np.sqrt((0-9)**2 + (0-9)**2))