import numpy as np 
import random

# Q = np.zeros((1296,4))

# print(Q)

# np.savetxt("q.txt", Q.astype(np.float), fmt='%f', delimiter = " ")

# R = np.loadtxt("q.txt", dtype='float', delimiter=" ")

# print(R)

SEED = 0

# random.seed(SEED)

for i in range(5):
	print(np.random.rand(3,4))


for a in range(5):
	for b in range(5):
		for c in range(5):
			for d in range(5):
 				print(int( ( 6*a + b+1 )*( 6*c + d+1 ) )-1)