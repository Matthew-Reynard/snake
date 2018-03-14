# Python
import tensorflow as tf

#Creates tf constants. Not a normal Python variable
hello = tf.constant('Hello, TensorFlow!')
a = tf.constant(5)
b = tf.constant(6)

#tf.mul, tf.sub was updated/changed to tf.multiply, tf.subtract, etc.
c = tf.multiply(a,b)

#Prints out a Tensor - not the variable or answer 30
#print(c)

#sess = tf.Session()
#print(sess.run(c))
#sess.close()

#Closes the Session automatically when your out of the "with" block
with tf.Session() as sess:
	print(sess.run(hello))
	output = print(sess.run(c))

#Can access this vaairble as it was a python variable created in the Session
print(output)