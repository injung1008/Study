import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

#1. data 
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. model 
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

print(model.weights)
# [[-0.41164476,  1.0387052 ,  0.7915262 ]], =(3,) =array([0., 0., 0.], 
# array([[-0.79694754, -0.6735836 ],
#        [ 0.8802494 , -0.04078317],
#        [ 0.07294631, -0.2592914 ]], shape=(2,) numpy=array([0., 0.], 
# array([[1.0760957],
#        [0.5769205]], 

# model.summary()
print(model.trainable_weights)
print(len(model.weights))
print(len(model.trainable_weights))
