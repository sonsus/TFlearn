#lab9_2_tensorboard: adding tensorboard to the code


'''
How to use TensorBoard

1 from TF graph choose tensors to log
2 merge all summaries
3 create writer and add graph
4 run summary merge and add_summary 
5 launch TensorBoard
'''

#lab9_1 XOR convnet!
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
tf.set_random_seed(777)  


Xdat = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
Ydat = np.array([[0],[1], [1], [0]], dtype=np.float32)

X=tf.placeholder(dtype=tf.float32,shape=[None,2])
Y=tf.placeholder(dtype=tf.float32,shape=[None,1])

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2,2]))
    b1 = tf.Variable(tf.random_normal([2]))
    hyp1=tf.matmul(X,W1)+b1
    layer1=tf.nn.sigmoid(hyp1) 

    W1_hist= tf.summary.histogram("W1, input layer", W1)
    b1_hist= tf.summary.histogram("b1, input layer", b1)
    layer1hist=tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2") as scope:
    W= tf.Variable(tf.random_normal([2,1]))
    b = tf.Variable(tf.random_normal([1]))
    hyp=tf.sigmoid(tf.matmul(layer1,W)+b)

    Whist=tf.summary.histogram("W, outputlayer", W)
    bhist=tf.summary.histogram("b, outputlayer", b)
    hyphist=tf.summary.histogram("layer2", hyp)

with tf.name_scope("cost") as scope:
    cost=-tf.reduce_mean(Y*tf.log(hyp)+(1-Y)*tf.log(1-hyp))
    costsumm= tf.summary.histogram("cost summary", cost)
    
with tf.name_scope("train") as scope:
    opt= tf.train.GradientDescentOptimizer(learning_rate =0.1).minimize(cost)


pred= tf.cast(hyp>0.5, dtype=tf.float32)
acc= tf.reduce_mean(tf.cast(tf.equal(Y, pred),dtype=tf.float32))

epoch=5000

accuracy_list=[]
cost_list=[]

with tf.Session() as sess:
    summary= tf.summary.merge_all()
    writer=tf.summary.FileWriter("./lab9_xor")
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    for step in range(epoch):
        o,c,a,p,s=sess.run([opt,cost,acc,pred,summary], feed_dict={X:Xdat, Y:Ydat})
        accuracy_list.append(a)
        cost_list.append(c)
        writer.add_summary(s, global_step=step)
        if step%100==0:
            print("step=%s"%step)
            print("cost=%s"%c)
            print("accuracy=%s"%a)
            print("prediction=%s"%p)


plt.plot(range(epoch),cost_list)
plt.show()

plt.plot(range(epoch),accuracy_list)
plt.show()