#lab4_queuerunner.py
import tensorflow as tf 
import numpy as np 

filename_queue=tf.train.string_input_producer(
	['lab4_dat_1.csv'], name="filename_queue")
reader= tf.TextLineReader()
key, val= reader.read(filename_queue)

#Defualt values for empty columns and for type casting for decoded result
record_defaults=[[0.],[0.],[0.],[0.],]
xy=tf.decode_csv(val, record_defaults=record_defaults)

#x,y batch
train_x_batch, train_y_batch=\
	tf.train.batch([xy[0:-1],xy[-1:]], batch_size=10)

sess=tf.Session()

#placeholder
X=tf.placeholder(tf.float32, shape=[None,3])
Y=tf.placeholder(tf.float32, shape=[None,1])
W=tf.Variable(tf.random_normal([3,1]))
b=tf.Variable(tf.random_normal([1]))

#hypothesis, cost
hyp=tf.matmul(X,W)+b
cost=tf.reduce_mean(tf.square(hyp-Y))

#optimizer, minimizing
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

#graph launcher
sess=tf.Session()
#initializing vars in graph
sess.run(tf.global_variables_initializer())

#coordinator starts queuerunner 
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
	x_batch, y_batch=sess.run([train_x_batch, train_y_batch])
	cost_val, hyp_val, _=sess.run([cost,hyp,train], feed_dict={X:x_batch, Y:y_batch})
	if step%500==1: 
		print(step, "cost: ", cost_val, "\tprediction\t", hyp_val)

coord.request_stop()
coord.join(threads)
#coordinator ends queueing

