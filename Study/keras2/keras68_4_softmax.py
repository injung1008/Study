import numpy as np 
import matplotlib.pyplot as plt 

#relu - 0보다 작은건 다 0 으로 유지하고 0보다 큰건 그상태로유지 
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(1,5)
y = softmax(x)

ratio = y 


plt.pie(ratio, labels=y, shadow=True, startangle=90)
plt.show()
