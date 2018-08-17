import socket
import time
import sys
import numpy as np
# from _thread import *


# AF_INET => IPv4 address, SOCK_STREAM => TCP
# SOCK_DGRAM => UDP (User Datagram Protocol)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP connection

print(s)

# server = "127.0.0.1"
server = "localhost"
host = ""
# server = "www.matthewreynard.com"
port = 5555

connected = False

s.connect((server, port))

iteration = 0

s.close()

while connected:

	# time.sleep(20)

	iteration=iteration+1

	try:
		# data = input("Send (q to Quit): ")
		# s.send(str.encode("p\n"))
		# print("recv")
		r = s.recv(1024)
		s.send(str.encode(str(iteration) + "\n"))
		# if r != None:
		# 	x = r.decode("utf-8")
		# 	# print(x)
		# 	x_cleaned = x[3:-1] #Need to find a better implementation
		# 	a = x_cleaned.split(", ")

		# 	if a[0] == "0":
			
		# 		s.send(str.encode(str(iteration) + "\n"))

			# else:
			# 	s.close()
			# 	print("Socket has been closed")
			# 	connected = False

	# To force close the connection
	except KeyboardInterrupt as e:
		s.close()
		print("Socket has been closed")
		raise e
		connected = False

	# data = input("Send (q to Quit): ")
	# if (data != "Q" and data != "q"):
	# 	if(data=="p"):
	# 		s.send(str.encode(data + "\n"))
	# 		print("P was sent")
	# 		# while True:
	# 		r = s.recv(1024)

	# 		# state = "Server message: " + r.decode("utf-8")
	# 			# if not r:
	# 			# 	break
	# 		# print(state)

	# 		x = r.decode("utf-8")

	# 		x_cleaned = x[3:-1]

	# 		a = x_cleaned.split(", ")

	# 		state = np.zeros(4)

	# 		for i in range(4):
	# 			state[i] = float(a[i])

	# 		print(state)

	# 		# new_state[0] = float(x[2:])

	# 		# for i in range(len(x)):
	# 		# 	print(i, x[i])

	# 		action = np.argmax(Q[state_index(state)])

	# 		print("Action = ", action)

	# 		s.send(str.encode(str(action) + "\n"))

	# 		# np.savetxt("./tmp/State.txt", new_state.astype(np.float), fmt='%f', delimiter = " ")
	# 		# np.savetxt("./tmp/State.txt", state)	
	# 	else:
	# 		s.send(str.encode(data + "\n"))
	# else:
	# 	s.send(str.encode(data + "\n"))
	# 	s.close()
	# 	break



# server_ip = socket.gethostbyname(server)
# print(server_ip)

# request = "GET / HTTP/1.1\nHost: " + server + "\n\n"

# s.connect((server, port))
# s.send(request.encode()) #encoding into byte format

# result = s.recv(1024)

# print(result)

# def pscan(port):
# 	try:
# 		s.connect((server,port))
# 		return True
# 	except:
# 		return False


# for x in range (1, 10000):
# 	if pscan(x):
# 		print("Port",x,"is open")
# 	else:
# 		print("Port",x,"is closed")

# try:
# 	s.bind((host, port))
# except socket.error as e:
# 	print(str(e))

# s.listen(5)

# print("Waiting for connection...")

# def threaded_client(conn):
# 	conn.send(str.encode("Welcome, enter a number:\n"))

# 	while True:
# 		data = conn.recv(1024)
# 		reply = "Server output: "+ data.decode("utf-8")
# 		if not data:
# 			break
# 		conn.sendall(str.encode(reply))

# 	conn.close()


# while True:

# 	conn, addr = s.accept()
# 	print("connected to: "+addr[0]+":"+str(addr[1]))

# 	start_new_thread(threaded_client,(conn,))

