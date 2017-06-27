#lab6-2 onehot rank extension and flattening
import tensorflow as tf 
import numpy as np 

xy=np.loadtxt("lab6_2_zoo.csv", delimiter=",", dtype=np.float32)
x_data=xy[:,:-1]
y_data=xy[:,[-1]] #bracketing the each element of [:,-1]

'''
original:
n_feat=16
n_classes =7

(101, 16) (101, 1)
one_hot Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
reshape Tensor("Reshape:0", shape=(?, 7), dtype=float32)
'''

n_feat=len(x_data[0]) 
n_class=int(max(y_data.flatten())+1)

X=tf.placeholder("float", [None, n_feat])
Y=tf.placeholder("int64", [None, 1])

Y_one_hot=tf.one_hot(Y,n_class) #Y elements become indices
Y_one_hot=tf.reshape(Y_one_hot, [-1, n_class]) #n_class is max(all Y elements)
#I don't fully understood one

W=tf.Variable(tf.random_normal([n_feat,n_class]), name="weight")
b=tf.Variable(tf.random_normal([1, n_class]),name="bias")
b1=tf.Variable(tf.random_normal([n_class,1]),name="bias")


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
    print(sess.run(b))
    print(sess.run(b1))
    for step in range(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step %100==0:
            loss, accu = sess.run([cost,acc], feed_dict={X:x_data, Y:y_data})
            print(step, loss, accu)
            