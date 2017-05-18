#loading csv using np
import tensorflow as tf 
import numpy as np 

#fetch x and y as matrices or vectors
xy=np.loadtxt("lab4_dat_1.csv",delimiter=",",dtype=np.float32)
x_data=xy[: 0:-1]
y_data=xy[: [-1]]
#check slicing with py syntax


#check the shape
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)
