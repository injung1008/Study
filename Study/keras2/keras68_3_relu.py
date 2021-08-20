import numpy as np 
import matplotlib.pyplot as plt 

#relu - 0보다 작은건 다 0 으로 유지하고 0보다 큰건 그상태로유지 
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

#과제 
#elu, selu, reaky relu ... 68_3_2, 3,4,로 만들것 