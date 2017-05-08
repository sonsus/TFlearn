#lab2 : linear regression

'''
tf.tree architecture

tf.something(tf.onemore(tf.constant(a)))
const(a) is a leaf and tf.something is root 
'''

import tensorflow as tf
x_train=[1,2,3,]
y_train=[2,3,4,]

W= tf.Variable(tf.random_normal([1]), name="weight")
b= tf.Variable(tf.random_normal([1]), name="bias")

# Variable is node. Not like usual variables we are using in programming language
# After dfning shape of the graph and give variable node values

hypothesis= x_train* W + b

#cost: (1/n)*sum(h(xi)-yi)**2
cost = tf.reduce_mean(tf.square(hypothesis-y_train))

#GD
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01)
train= optimizer.minimize(cost)

#run session in order to start optimizining (learning)
sess=tf.Session()
sess.run(tf.global_variables_initializer()) #global var initializing is important!


#line fitting
for step in range(300000):
    sess.run(train)
    if (step % 20 == 0):
        print(step, sess.run(cost), sess.run(W), sess.run(b))


'''
.
.
.
216140 5.19596e-11 [ 1.00000858] [ 0.99998128]
216160 5.19596e-11 [ 1.00000858] [ 0.99998128]
216180 5.19596e-11 [ 1.00000858] [ 0.99998128]
216200 5.19596e-11 [ 1.00000858] [ 0.99998128]
.
.
.
'''
#why is it converged to slightly biased value? Shouldnt it converged to W==1, cost==0 and b==1? 