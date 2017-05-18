#numpy slicing adv

import numpy as np 

b= np.array([[1,2,3,4,],[5,6,7,8,],[1,2,3,4,],[5,6,7,8,]])

print(b[:,1])
#for all rows, get 1st column == [2,6,2,6,]
print(b[-1])
#take the last row == [5,6,7,8,]
#which is last sublist element of the array list
print(b[-1,:])
#same as above
print(b[-1,...])
#same as above
print(b[0:2, :])
#row 0~1, all columns as a matrix that is...
#[[1 2 3 4]
# [5 6 7 8]]