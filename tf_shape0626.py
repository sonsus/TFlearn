#shape of the matrices

import tensorflow as tf
import numpy as np 

a=np.ones([7,1])
b=np.ones([7])

sa=tf.shape(a)
sb=tf.shape(b)

print(a)
print(b)
print(sa, sb)