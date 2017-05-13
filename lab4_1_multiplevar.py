#feeding multivariate data to dfn hypothesis with matrices

import tensorflow as tf 
x1_data=[100.,200.,154.,342.,424.,]
x2_data=[121.,232.,343.,454.,565.,]
x3_data=[232.,343.,454.,565.,121.,]
y=[152.,185.,180.,196.,142.,]

#placeholders
x1=tf.placeholder(tf.float32)
x2=tf.placeholder(tf.float32)
x3=tf.placeholder(tf.float32)
_y=tf.placeholder(tf.float32)

#hypothesis weight dfn
#5x3 vars, 5x1 output --> 3x1 weight matrix required
w1=tf.Variable(tf.random_normal([1]), name="weight1")
w2=tf.Variable(tf.random_normal([1]), name="weight2")
w3=tf.Variable(tf.random_normal([1]), name="weight3")
b=tf.Variable(tf.random_normal([1]), name="bias")

hypo=w1*x1+w2*x2+w3*x3+b

#cost function
cost=tf.reduce_mean(tf.square(hypo-_y))
#minimize
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

#launch the graph in a session
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(2001):
	cost_val, hy_val, _ = sess.run([cost,hypo,train], 
							feed_dict= {x1:x1_data, x2:x2_data,x3:x3_data,_y:y})
	if epoch%10 ==0:
		print(epoch, "cost=", cost_val,"\n prediction:\n",hy_val)
