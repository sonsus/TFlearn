#lab 10 MNIST higher_lv api

from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf
import random as r 

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learn_rate=0.01
n_epoch=15
batch_size= 100
keep_rate=0.7

X=tf.placeholder(tf.float32,[None, 784])
Y=tf.placeholder(tf.float32,[None,10])
trainmode=tf.placeholder(tf.bool, name="trainmode") # what is this?

h_out_dim=256
fin_out_dim=10

xavier_init = tf.contrib.layers.xavier_initializer()
bn_params={
    "is_training"           : trainmode,
    "decay"                 : 0.9,
    "updates_collections"   : None
}#batch_norm parameters?

#arg_scope for avoiding duplicative codes with different var names
with arg_scope([fully_connected],
                activation_fn=tf.nn.relu,
                weights_initializer=xavier_init,
                biases_initializer=None,
                normalizer_fn=batch_norm,
                normalizer_params=bn_params
                ):
    h1=fully_connected(X, h_out_dim, scope="h1")
    h1_d=dropout(h1, keep_rate, is_training=trainmode)
    h2=fully_connected(h1, h_out_dim, scope="h2")
    h2_d=dropout(h2, keep_rate, is_training=trainmode)
    h3=fully_connected(h2, h_out_dim, scope="h3")
    h3_d=dropout(h3, keep_rate, is_training=trainmode)
    h4=fully_connected(h3, h_out_dim, scope="h4")
    h4_d=dropout(h4, keep_rate, is_training=trainmode)
    h5=fully_connected(h4, h_out_dim, scope="h5")
    h5_d=dropout(h5, keep_rate, is_training=trainmode)
    h6=fully_connected(h5, h_out_dim, scope="h6")
    h6_d=dropout(h6, keep_rate, is_training=trainmode)
    h7=fully_connected(h6, h_out_dim, scope="h7")
    h7_d=dropout(h7, keep_rate, is_training=trainmode)
    hyp=fully_connected(h7_d, fin_out_dim, activation_fn=None, scope="hypothesis")


with tf.name_scope("cost") as scope:
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                        logits=hyp, labels=Y))
    tf.summary.scalar("cost", cost)
with tf.name_scope("train") as scope:
    optimizer=tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)

sess= tf.Session()
sess.run(tf.global_variables_initializer())

summary=tf.summary.merge_all()
writer=tf.summary.FileWriter("./TB_scalar")
writer.add_graph(sess.graph)

for epoch in range(n_epoch):
    avg_cost=0
    total_batch=int(mnist.train.num_examples/batch_size)
    for step in range(total_batch):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        train_feed= {X: batch_x, Y: batch_y, trainmode: True}
        cost_feed= {X: batch_x, Y: batch_y, trainmode: False}
        train=sess.run(optimizer,feed_dict=train_feed)
        c=sess.run(cost, feed_dict=cost_feed)
        s=sess.run(summary, feed_dict=train_feed)
        avg_cost += c/total_batch
        writer.add_summary(s, global_step=step+epoch*total_batch)
    print("Epoch={:>5}, cost={:>.10}".formats(epoch+1, avg_cost)) #look at this pretty primitive regex. I might want this for later
print("Learning Done!")

correct_prediction = tf.equal(tf.argmax(hyp, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels, trainmode: False}))

r=r.randint(0,mnist.test.num_examples -1)
print("label=", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
print("pred=", sess.run(tf.argmax(hyp,1), feed_dict= {X: mnist.test.images[r:r+1], trainmode:False}))

sess.close()