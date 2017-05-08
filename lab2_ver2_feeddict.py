#lab2 : version 2
# w/o declaring variables x, y just feed them later with tf.placeholder / feed_dict

import tensorflow as tf
#x_train=[1,2,3,]
#y_train=[2,3,4,]
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

W= tf.Variable(tf.random_normal([1]), name="weight")
b= tf.Variable(tf.random_normal([1]), name="bias")

# Variable is node. Not like usual variables we are using in programming language
# After dfning shape of the graph and give variable node values

#hypothesis= x_train* W + b
hypothesis= X* W + b

#cost: (1/n)*sum(h(xi)-yi)**2
#cost = tf.reduce_mean(tf.square(hypothesis-y_train))
cost = tf.reduce_mean(tf.square(hypothesis-Y))

#GD
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01)
train= optimizer.minimize(cost)

#run session in order to start optimizining (learning)
sess=tf.Session()
sess.run(tf.global_variables_initializer()) #global var initializing is important!


#line fitting
for step in range(2001):
    cost_val, W_val, b_val, _= \
            sess.run([cost,W,b,train], feed_dict={X: [1,2,3,4,5,], Y:[2.1,3.1,4.1,5.1,6.1]})
    if (step % 20 == 0):
        print(step, cost_val, W_val, b_val)
