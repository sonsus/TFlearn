#matrices are the chosen ones.,
import tensorflow as tf 

x_data=[
[73.,80.,75.,],
[93.,88.,93.,],
[89.,91.,90.,],
[96.,98.,100.,],
[73.,66.,70.,],
]
y_data=[
[152.],[185.],[180.],[196.],[142.],
]

#placeholders for a tensor that will always be fed.
X=tf.placeholder(tf.float32 , shape=None)
Y=tf.placeholder(tf.float32 , shape=None)
