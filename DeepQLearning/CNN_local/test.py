import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import random

GRID_SIZE = 9

n_input_channels = 3

n_out_channels_conv1 = 16
n_out_channels_conv2 = 32
n_out_fc = 256

filter1_size = 3
filter2_size = 3

# Number of actions - Up, down, left, right
n_actions = 4


x = tf.placeholder(tf.float32, [n_input_channels, GRID_SIZE, GRID_SIZE], name="Input")

# output
y = tf.placeholder(tf.float32, [1, n_actions], name="Output")

def conv2d(x, W, name = None):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding ="VALID", name = name)


# Max pooling
def maxpool2d(x, name = None):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='VALID', name = name)

def createDeepModel(data):

	weights = {'W_conv1':tf.Variable(tf.truncated_normal([3, 3, n_input_channels, 16], mean=0, stddev=1.0, seed=0), name = 'W_conv1'),
			   	   'W_conv2':tf.Variable(tf.truncated_normal([3, 3, 16, 32], mean=0, stddev=1.0, seed=1), name = 'W_conv2'),
			   	   'W_fc':tf.Variable(tf.truncated_normal([4*4*32, 256], mean=0, stddev=1.0, seed=2), name = 'W_fc'),
			   	   'W_out':tf.Variable(tf.truncated_normal([256, n_actions], mean=0, stddev=1.0, seed=3), name = 'W_out')}

	biases = {'b_conv1':tf.Variable(tf.constant(0.1, shape=[16]), name = 'b_conv1'),
		   	  'b_conv2':tf.Variable(tf.constant(0.1, shape=[32]), name = 'b_conv2'),
		   	  'b_fc':tf.Variable(tf.constant(0.1, shape=[256]), name = 'b_fc'),
		   	  'b_out':tf.Variable(tf.constant(0.1, shape=[n_actions]), name = 'b_out')}

	# Model operations
	x = tf.reshape(data, shape=[-1, GRID_SIZE, GRID_SIZE, n_input_channels])

	conv1 = conv2d(x, weights['W_conv1'], name = 'conv1')
	print("Conv1 shape: ", conv1.shape)
	# conv1 = maxpool2d(conv1, name = 'max_pool1')
	print("Conv1 shape: ", conv1.shape)

	print()

	conv2 = conv2d(conv1, weights['W_conv2'], name = 'conv2')
	print("Conv2 shape: ", conv2.shape)
	conv2 = maxpool2d(conv2, name = 'max_pool2')
	print("Conv2 shape: ", conv2.shape)

	fc = tf.reshape(conv2,[-1, 4*4*32])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

	output = tf.matmul(fc, weights['W_out']) + biases['b_out']
	output = tf.nn.l2_normalize(output)

	return output

class Tau():
    def __init__(self, field1, field2, field3):
        self.field1 = field1
        self.field2 = field2
        self.field3 = field3


if __name__ == '__main__':


	# allowed = list((range(2), range(2)))
	allowed = []

	for j in range(3):
		for i in range(3):
			allowed.append((i,j))

	disallowed = [(1,0)]

	a = [(0,0),(2,0)]

	[disallowed.append(b) for b in a[3:]]

	# allowed.remove([pos for pos in disallowed])

	[allowed.remove(pos) for pos in disallowed]

	# print(np.asarray(allowed))

	for i in range(10):
		print(random.choice((allowed)))

	# total_episodes = 100 

	# scale_factor = 100

	# start = 0.7
	# end = 0.5
	# range_ = 0.5


	# x = np.arange(0,total_episodes)
	# y = np.zeros(total_episodes)
	# y[0] = start
	# print(x.size)

	# print(np.exp(-np.log(((2+1)/10)*100)))

	# for i in range(total_episodes):
	# 	# print(i, x[i])
	# 	# y[i] = np.exp(-np.log((((x[i]+1)/total_episodes)*scale_factor))-np.log(1/(start-end))) + end
	# 	y[i] = np.exp(-np.log(x[i]+1))*10

	# # y = np.exp(-np.log((x/total_episodes)*scale_factor)-np.log(1/(start-end))) + end

	# # print(np.exp(-np.log(100)-np.log(1/(start-end))) + end)
	# # y = 1/np.power(1.01,x) + end


	# # plt.plot([np.mean(errors[i:i+500]) for i in range(len(errors) - 500)])
	# plt.plot(x,y/10)
	# plt.savefig("./e.png")
	# plt.show()