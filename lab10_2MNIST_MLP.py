#lab10-2 mnist with MLP, xavier, dropout
#tensorboard
import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist= input_data.read_data_sets("MNIST_data/", one_hot=True)

learn_rate=0.001
epoch=15
batch_size=100

X=tf.placeholder(tf.float32,[None, 784])
Y=tf.placeholder(tf.float32,[None, 10])
keep_prob=tf.placeholder(tf.float32)
'''
input layer: 784x128
hidden layer: 128x128
output layer: 128x10
'''

with tf.name_scope("input_layer") as scope:
    W1=tf.get_variable("W1", [784,128], 
        initializer=tf.contrib.layers.xavier_initializer())
    b1=tf.Variable(tf.random_normal([128]))
    L1=tf.nn.relu(tf.matmul(X,W1)+b1)
    L1=tf.nn.dropout(L1, keep_prob=keep_prob)

    w1_hist=tf.summary.histogram("W1", W1)
    b1_hist=tf.summary.histogram("b1", b1)
    l1_hist=tf.summary.histogram("L1", L1)

with tf.name_scope("layer2") as scope:
    W2=tf.get_variable("W2", [128,128], 
        initializer=tf.contrib.layers.xavier_initializer())
    b2=tf.Variable(tf.random_normal([128]))
    L2=tf.nn.relu(tf.matmul(L1,W2)+b2)
    L2=tf.nn.dropout(L2, keep_prob=keep_prob)

    w2_hist=tf.summary.histogram("W2", W2)
    b2_hist=tf.summary.histogram("b2", b2)
    l2_hist=tf.summary.histogram("L2", L2)

with tf.name_scope("layer3") as scope:
    W3=tf.get_variable("W3", [128,128], 
        initializer=tf.contrib.layers.xavier_initializer())
    b3=tf.Variable(tf.random_normal([128]))
    L3=tf.nn.relu(tf.matmul(L2,W3)+b3)
    L3=tf.nn.dropout(L3, keep_prob=keep_prob)

    w3_hist=tf.summary.histogram("W3", W3)
    b3_hist=tf.summary.histogram("b3", b3)
    l3_hist=tf.summary.histogram("L3", L3)

with tf.name_scope("layer4") as scope:
    W4=tf.get_variable("W4", [128,128], 
        initializer=tf.contrib.layers.xavier_initializer())
    b4=tf.Variable(tf.random_normal([128]))
    L4=tf.nn.relu(tf.matmul(L3,W4)+b4)
    L4=tf.nn.dropout(L4, keep_prob=keep_prob)

    w4_hist=tf.summary.histogram("W4", W4)
    b4_hist=tf.summary.histogram("b4", b4)
    l4_hist=tf.summary.histogram("L4", L4)

with tf.name_scope("layer5") as scope:
    W5=tf.get_variable("W5", [128,128], 
        initializer=tf.contrib.layers.xavier_initializer())
    b5=tf.Variable(tf.random_normal([128]))
    L5=tf.nn.relu(tf.matmul(L4,W5)+b5)
    L5=tf.nn.dropout(L5, keep_prob=keep_prob)

    w5_hist=tf.summary.histogram("W5", W5)
    b5_hist=tf.summary.histogram("b5", b5)
    l5_hist=tf.summary.histogram("L5", L5)

with tf.name_scope("output_layer") as scope:
    W6=tf.get_variable("W6", [128,10], 
        initializer=tf.contrib.layers.xavier_initializer())
    b6=tf.Variable(tf.random_normal([10]))
    hyp=tf.matmul(L5,W6)+b6
    #the last layer mustnt have dropout even for training.

    w6_hist=tf.summary.histogram("W6", W6)
    b6_hist=tf.summary.histogram("b6", b6)
#   l6_hist=tf.summary.histogram("L6", L6)

with tf.name_scope("cost") as scope:
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=hyp, labels=Y ))
    costsumm= tf.summary.histogram("cost summary", cost)
with tf.name_scope("train") as scope:
    train=tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)




with tf.Session() as sess: #tf.Session.close() is a requirement-pair.
    #tensorboard
    summary= tf.summary.merge_all()
    writer=tf.summary.FileWriter("./lab10_skills")
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for j in range(total_batch):
            batch_xs, batch_ys= mnist.train.next_batch(batch_size)
            feed_dict= {X:batch_xs, Y:batch_ys, keep_prob:0.7} 
            s, c, _=sess.run([summary, cost, train], feed_dict=feed_dict)
            avg_cost += c/total_batch
        writer.add_summary(s, global_step=epoch*total_batch) #tensorboard plotting
            
        print("EPOCH=%04d"%(i+1), "avgCOST={:.9f}".format(avg_cost))
    print("Learning all done")


    #test the trained model
    #this should be done w/o closing the session
    isCorrect= tf.equal(tf.argmax(hyp,1), tf.argmax(Y,1))
    accuracy= tf.reduce_mean(tf.cast(isCorrect, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
    r=random.randint(0, mnist.test.num_examples -1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Pred:  ", sess.run(tf.argmax(hyp, 1), 
        feed_dict={X: mnist.test.images[r:r+1], keep_prob:1}))