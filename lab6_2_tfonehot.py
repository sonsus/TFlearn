#lab6-2 onehot rank extension and flattening
import tensorflow as tf 
import numpy as np 

xy=np.loadtxt("lab6_2_zoo.csv", delimiter=",", dtype=np.float32)
x_data=xy[:,:-1]
y_data=xy[:,[-1]]



n_feat=len(x_data[0])
n_class=max(y_data.flatten())

X=tf.placeholder("float", [None, n_feat])
Y=tf.placeholder("float", [None, 1])

Y_one_hot=tf.one_hot([Y,n_class])
Y_one_hot=tf.reshape(Y_one_hot, [-1, n_class])

W=tf.Variable(tf.random_normal([n_feat,n_class]), name="weight")
b=tf.Variable(tf.random_normal([n_class,1]),name="bias")
