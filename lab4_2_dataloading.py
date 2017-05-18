#loading csv using np

import tensorflow as tf 
import numpy as np 

#fetch x and y as matrices or vectors
xy=np.loadtxt("lab4_dat_1.csv",delimiter=",",dtype=np.float32)
x_data=xy[:, 0:-1]
y_data=xy[:, [-1]]
#check slicing with py syntax


#check the shape
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

#placeholders
X=tf.placeholder(tf.float32, shape=[None,3])
Y=tf.placeholder(tf.float32, shape=[None,1])

W=tf.Variable(tf.random_normal([3,1,]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

#hypothesis
hyp=tf.matmul(X,W)+b

#cost and optimizer
cost=tf.reduce_mean(tf.square(hyp-Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

#session
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#feed dict and run
for step in range(2001):
    cost_val, hy_val, _ =sess.run(
                [cost,hyp,train], feed_dict={X: x_data, Y: y_data})
    if step%10==0:
        print(step,"cost=", cost_val, "\nprediction: \n", hy_val)

#ask my score
print("ur score will be, ", sess.run(hyp, feed_dict={X:[[100,70,101]]}))
print("others\' score will be, ", sess.run(hyp, feed_dict={X:[[60,70,110],[90,100,80],]}))
