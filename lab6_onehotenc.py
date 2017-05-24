#lab6-1 softmax, argmax
import numpy as np 
import tensorflow as tf 

xy=np.loadtxt("lab6_1_onehotenc.csv", delimiter=",",dtype=np.float32)
n_col=len(xy[0,:])
n_row=len(xy[:,0])

x_data=xy[:,:-3]
y_data=xy[:,-3:]

x_len=len(x_data[0,:])
y_len=len(y_data[0,:])


X=tf.placeholder("float", [None, x_len])
Y=tf.placeholder("float", [None, y_len])

W=tf.Variable(tf.random_normal([x_len,y_len]),name="weight")
b=tf.Variable(tf.random_normal([y_len]), name="bias")

hyp=tf.nn.softmax(tf.matmul(X,W)+b)
cost=-tf.reduce_sum(Y*tf.log(hyp))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if epoch%10==0:
            print("epoch= ",epoch, "\ncost= ",sess.run(cost, feed_dict={X:x_data, Y:y_data}))
    q=sess.run(hyp, feed_dict={X:[[1,2,3,4],[23,5,3,6],[1,2,5,3]]})
    print(q, sess.run(tf.arg_max(q,1)))

