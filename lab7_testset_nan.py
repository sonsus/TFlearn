# Lab 7 Learning rate and Evaluation

import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

logf=open("log.txt","w")
logf.write("for lab 7\n\n")

learning_rate=float(input("learning rate (1e-8~0.1), float: "))
logf.write("learningrate=%s\n"%learning_rate)
epoch=int(input("epoch in int: "))
logf.write("epoch=%s\n"%epoch)
stamp=int(input("how often do you want to see the vals? int: "))

#multivariable classification with given 3-classed one-hot vectors as a tagging 

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]


# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]


X=tf.placeholder("float", [None,3])
Y=tf.placeholder("float", [None,3])

W=tf.Variable(tf.random_normal([3,3]))
b=tf.Variable(tf.random_normal([1,3]))

hyp= tf.nn.softmax(tf.matmul(X,W)+b)

cost= tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hyp),axis=1))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

pred=tf.arg_max(hyp, 1)
is_correct=tf.equal(pred,tf.arg_max(Y,1))
acc=tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(epoch):
        c,p,a,w,_=sess.run( [cost, pred, acc, W, optimizer], feed_dict={X: x_data,Y:y_data} )
        if step%stamp==0:
            logf.write("\nstep: {step} \ncost={c}\npred={p}\nacc={a}\nw={w}\n".format(step=step, c=c, p=p, a=a, w=w))
            print("\nstep: {step} \ncost={c}\npred={p}\nacc={a}\nw={w}".format(step=step, c=c, p=p, a=a, w=w))
    #test
    print("Prediction:%s"%sess.run(pred, feed_dict={X:x_test}))
    logf.write("Prediction:%s\n"%sess.run(pred, feed_dict={X:x_test})) 
    print("accuracy eval:%s"%sess.run(acc, feed_dict={X:x_test, Y:y_test}))
    logf.write("accuracy eval:%s\n"%sess.run(acc, feed_dict={X:x_test, Y:y_test}))

logf.close()


