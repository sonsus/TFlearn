#lab3 plus: sometimes you might want to massage gradient value

import tensorflow as tf
x_train=[1,2,3,]
y_train=[2,3,4,]

W= tf.Variable(tf.constant(5.)) #constant has no attribute name thus name="weight" is improper

hypothesis= x_train* W
cost = tf.reduce_mean(tf.square(hypothesis-y_train))


#GD: if you want to manually edit gradient value at each moment
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01)

#handwritten gradient and computed gradient
gradient=tf.reduce_mean((hypothesis-y_train)*x_train)*2
gvs=optimizer.compute_gradients(cost)                   #compute_gradient returns [gradient, W]
apply_gradients=optimizer.apply_gradients(gvs)


#run session in order to start optimizining (learning)
sess=tf.Session()
sess.run(tf.global_variables_initializer()) #global var initializing is important!


#line fitting
for step in range(100):
    print(step, sess.run([gradient, W, gvs])) 
    sess.run(apply_gradients)


'''output

95 [0.0030230682, 1.4288954, [(0.0030230284, 1.4288954)]]
96 [0.0027409394, 1.4288651, [(0.0027409196, 1.4288651)]]
97 [0.00248456, 1.4288377, [(0.00248456, 1.4288377)]]
98 [0.002253135, 1.4288129, [(0.0022531152, 1.4288129)]]
99 [0.002043565, 1.4287903, [(0.0020435452, 1.4287903)]]


'''