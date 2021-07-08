import numpy as np

A = np.array([[1,2,3], [4,5,6]])
# B = A.reshape((3,2))
A.resize((3,2))
print(A)