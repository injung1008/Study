#1. 데이터
#같은 값의 데이터가 있어도 상관이 없다 데이터의 shape이 중요한것
import numpy as np
x = np.array([range(100), range(301, 401), range(1,101), range(100), range(401,501)])
x = np.transpose(x) #(100,5)

y = np.array([range(711,811), range(101, 201)])
y = np.transpose(y) #(100,2)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_shape=(5,)))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(2))

model.summary()


