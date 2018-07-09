import socket
import time
import sys
import numpy as np
# from _thread import *


GRID_SIZE = 10

def state_index(state_array):
    return int((GRID_SIZE**3)*state_array[0]+(GRID_SIZE**2)*state_array[1]+(GRID_SIZE**1)*state_array[2]+(GRID_SIZE**0)*state_array[3])


# AF_INET => IPv4 address, SOCK_STREAM => TCP
# SOCK_DGRAM => UDP (User Datagram Protocol)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP connection

Q_textfile_path_load = "./QLearning/Data/Q_10x10_no_wrap.txt"

print(s)

# Q learning in Minecraft
Q = np.loadtxt(Q_textfile_path_load, dtype='float', delimiter=" ")

# server = "127.0.0.1"
server = "localhost"
host = ""
# server = "www.matthewreynard.com"
port = 5555

# CLIENT

connected = True

s.connect((server, port))

iteration = 0

while connected:

	time.sleep(5)

	iteration=iteration+1

	try:
		# data = input("Send (q to Quit): ")
		s.send(str.encode("p\n"))
		r = s.recv(1024)
		if r != None:
			x = r.decode("utf-8")
			print(x)
			x_cleaned = x[3:-1] #Need to find a better implementation
			a = x_cleaned.split(", ")

			if a[0] != "close":

				state = np.zeros(4)
				for i in range(4):
					state[i] = float(a[i])

				print(state, iteration)

				action = np.argmax(Q[state_index(state)])

				print("Action = ", action)
			
				s.send(str.encode(str(action) + "\n"))

			else:
				s.close()
				print("Socket has been closed")
				connected = False

	# To force close the connection
	except KeyboardInterrupt as e:
		s.close()
		print("Socket has been closed")
		raise e
		connected = False