import numpy as np
import tensorflow as tf
import math

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

	# for i in range(10000):
	# 	if i%60<10:
	# 		if math.floor((i/60)%60)<10:
	# 			print("\ttime {0:.0f}:0{1:.0f}:0{2:.0f}".format(math.floor((i/60)/60), math.floor((i/60)%60), i%60))
	# 		else:
	# 			print("\ttime {0:.0f}:{1:.0f}:0{2:.0f}".format(math.floor((i/60)/60), math.floor((i/60)%60), i%60))
	# 	else:
	# 		if math.floor((i/60)%60)<10:
	# 			print("\ttime {0:.0f}:0{1:.0f}:{2:.0f}".format(math.floor((i/60)/60), math.floor((i/60)%60), i%60))
	# 		else:
	# 			print("\ttime {0:.0f}:{1:.0f}:{2:.0f}".format(math.floor((i/60)/60), math.floor((i/60)%60), i%60))

	a = [Tau(1,2,3), Tau(4,5,6)]

	a.append(Tau(np.zeros((3,9,9)),1,9))


	for i in range(1000):
		print(a[0].field3)
		print("len: ",len(a))

		if len(a)<100:
			a.append(Tau(np.zeros((3,9,9)),4,9))
		else:
			a.pop(0)
			a.append(Tau(np.zeros((3,9,9)),4,i))


	# Q = createDeepModel(x)

	# init = tf.global_variables_initializer()

	# with tf.Session() as sess:
	# 	# Initialize global variables
	# 	sess.run(init)

	# 	state_vector = np.array([[[1, 2, 3, 4, 5, 6, 7, 8], 
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8], 
	# 							[1, 2, 3, 4, 5, 6, 7, 8]],

	# 							[[1, 2, 3, 4, 5, 6, 7, 8], 
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8], 
	# 							[1, 2, 3, 4, 5, 6, 7, 8]],

	# 							[[1, 2, 3, 4, 5, 6, 7, 8], 
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8], 
	# 							[1, 2, 3, 4, 5, 6, 7, 8]],
	# 							])

	# 	state_vector_9 = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9], 
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9], 
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9]],

	# 							[[1, 2, 3, 4, 5, 6, 7, 8, 9], 
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9], 
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9]],

	# 							[[1, 2, 3, 4, 5, 6, 7, 8, 9], 
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9],  
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9], 
	# 							[1, 2, 3, 4, 5, 6, 7, 8, 9]],
	# 							])


	# 	print(state_vector.shape)

	# 	Q_vector = sess.run(Q, feed_dict={x: state_vector_9})

	# 	print(Q_vector)

