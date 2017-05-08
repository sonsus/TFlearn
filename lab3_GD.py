#lab3 : gradient descent
#cost function plot as iteration proceeds
import tensorflow as tf
import matplotlib.pyplot as plt

x_data=[0,1,2,3]
y_data=[0,1,2,3]

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
W=tf.Variable(tf.random_normal([1]), name="weight")

hyp=X*W
cost=tf.reduce_mean(tf.square(hyp-Y))

#GD
learning_rate=0.1
gradient=tf.reduce_mean((W*X-Y)*X)
descent=W-learning_rate*gradient
update=W.assign(descent)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

plot_y=[]
plot_x=[]

for epoch in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data })
    print(epoch, sess.run(cost,feed_dict={ X: x_data, Y: y_data }), sess.run(W)) # cost node needs feed_dict as its input for running
    #make lists for plt input
    plot_y.append(sess.run(cost, feed_dict={X: x_data, Y: y_data }))
    plot_x.append(epoch)

plt.plot(plot_x,plot_y)
plt.show()