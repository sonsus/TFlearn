#lab9_1 XOR convnet!
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
tf.set_random_seed(777)  


Xdat = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
Ydat = np.array([[0],[1], [1], [0]], dtype=np.float32)

X=tf.placeholder(dtype=tf.float32,shape=[None,2])
Y=tf.placeholder(dtype=tf.float32,shape=[None,1])

W1 = tf.Variable(tf.random_normal([2,2]))
b1 = tf.Variable(tf.random_normal([2]))
hyp1=tf.matmul(X,W1)+b1
layer1=tf.nn.sigmoid(hyp1) 

W= tf.Variable(tf.random_normal([2,1]))
b = tf.Variable(tf.random_normal([1]))
hyp=tf.sigmoid(tf.matmul(layer1,W)+b)

cost=-tf.reduce_mean(Y*tf.log(hyp)+(1-Y)*tf.log(1-hyp))
opt= tf.train.GradientDescentOptimizer(learning_rate =0.1).minimize(cost)

pred= tf.cast(hyp>0.5, dtype=tf.float32)
acc= tf.reduce_mean(tf.cast(tf.equal(Y, pred),dtype=tf.float32))

epoch=5000

accuracy_list=[]
cost_list=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(epoch):
        o,c,a,p=sess.run([opt,cost,acc,pred], feed_dict={X:Xdat, Y:Ydat})
        accuracy_list.append(a)
        cost_list.append(c)
        if step%100==0:
            print("step=%s"%step)
            print("cost=%s"%c)
            print("accuracy=%s"%a)
            print("prediction=%s"%p)


plt.plot(range(epoch),cost_list)
plt.show()

plt.plot(range(epoch),accuracy_list)
plt.show()