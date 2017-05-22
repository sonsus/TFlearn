#matrices are the chosen ones.,
import tensorflow as tf 
import numpy as np
tf.set_random_seed(777) #for reproducibility

xy=np.loadtxt("lab4_dat_1.csv", delimiter=",", dtype=np.float32)
x_data=xy[:,:3]
y_data=xy[:,[3]]

#placeholders for a tensor that will always be fed.
X=tf.placeholder(tf.float32 , shape=[None,3])
Y=tf.placeholder(tf.float32 , shape=[None,1])
#[None,1] means unlimited rows and 3 columns
W=tf.Variable(tf.random_normal([3,1]), name="weight")
b=tf.Variable(tf.random_normal([1]), name="bias")

#cost function
hypothesis=tf.matmul(X,W)+b
cost=tf.reduce_mean(tf.square(hypothesis-Y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)


#launch the grapgh
sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _=sess.run([cost,hypothesis,train], feed_dict={X:x_data, Y:y_data})
    if step%10==0:
        print(step, "cost=", cost_val, "\nprediction\n", hy_val)

print("your score: ", sess.run(hypothesis, 
						feed_dict={X: [[100,70,101]]}))

print("others' scores: ", sess.run(hypothesis, 
						feed_dict={X: [[60,70,110],[90,100,80]]}))  
