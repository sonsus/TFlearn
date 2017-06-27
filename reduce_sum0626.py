import tensorflow as tf

#for 2D matrix: 0=rowwise sum, 1=columnwise sum
#for 3D matrix: if 2D matrices are stacked and lets call its height directed vector as a plum
#               0=plumwise sum, 1=columnwise sum, 2=rowwise sum

no_axis=tf.reduce_sum([[1,2,3,4],[23,5,3,6],[1,2,5,3]])
axis_0=tf.reduce_sum([[1,2,3,4],[23,5,3,6],[1,2,5,3]],0)
axis_0_=tf.reduce_sum([[1,2,3,4],[23,5,3,6],[1,2,5,3]],0,True)
axis_1=tf.reduce_sum([[1,2,3,4],[23,5,3,6],[1,2,5,3]],1)
axis_1_=tf.reduce_sum([[1,2,3,4],[23,5,3,6],[1,2,5,3]],1,True)
a=tf.reduce_sum([[[1,2,3,4],
                       [23,5,3,6],
                       [1,2,5,3]],
                      [[1,2,3,4],
                       [23,5,3,6],
                        [1,2,5,3]],
                      [[1,2,3,4],
                       [23,5,3,6],
                        [1,2,5,3]]],0)
b=tf.reduce_sum([[[1,2,3,4],
                       [23,5,3,6],
                       [1,2,5,3]],
                      [[1,2,3,4],
                       [23,5,3,6],
                        [1,2,5,3]],
                      [[1,2,3,4],
                       [23,5,3,6],
                        [1,2,5,3]]],1)
c=tf.reduce_sum([[[1,2,3,4],
                       [23,5,3,6],
                       [1,2,5,3]],
                      [[1,2,3,4],
                       [23,5,3,6],
                        [1,2,5,3]],
                      [[1,2,3,4],
                       [23,5,3,6],
                        [1,2,5,3]]],2)


with tf.Session() as sess:
    for mats in [no_axis, axis_0, axis_1, axis_0_, axis_1,a,b,c]:
        res=sess.run(mats)
        print(res)
