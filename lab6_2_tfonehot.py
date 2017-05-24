#lab6-2 onehot rank extension and flattening
import tensorflow as tf 
import numpy as np 

xy=np.loadtxt("lab6_2_zoo.csv", delimiter=",", dtype=np.float32)
x_data=xy[:,:-1]
y_data=xy[:,[-1]]



n_feat=len(x_data[0])
n_class=max(y_data.flatten())

X=tf.placeholder("float", [None, n_feat])
Y=tf.placeholder("float", [None, 1])

Y_one_hot=tf.one_hot(Y,n_class)
Y_one_hot=tf.reshape(Y_one_hot, [-1, n_class])
#I don't fully understood one

W=tf.Variable(tf.random_normal([n_feat,n_class]), name="weight")
b=tf.Variable(tf.random_normal([n_class,1]),name="bias")

logits=tf.matmul(X,W)+b
hyp=tf.nn.softmax(logits)

cost_i= tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost=tf.reduce_mean(cost_i)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

pred= tf.argmax(hyp,1)
correct= tf.equal(pred,tf.argmax(Y_one_hot,1))
acc=tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step %100==0:
            loss, accu = sess.run([cost,acc], feed_dict={X:x_data, Y:y_data})
            print(step, loss, accu)
            