import numpy as np
import matplotlib.pyplot as plt

#1. 데이터 
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) - 얘는 4차원을 받는데 3차원이라서 차원을 늘려야한다 
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)



x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

from sklearn.preprocessing import StandardScaler, MinMaxScaler ,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# #scaler = MinMaxScaler()
scaler = StandardScaler()
# #scaler = MaxAbsScaler()
# #scaler = RobustScaler()
# scaler = QuantileTransformer()
# #scaler = PowerTransformer()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_test)

#차원 늘려주기 
x_train = x_train.reshape(60000, 784,1) 
x_test = x_test.reshape(10000, 784, 1) 


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray() #(60000, 10)
y_test = encoder.transform(y_test).toarray() #(10000, 10)
#print(y_test.shape)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(64, 2,input_shape=(784,1)))
model.add(Flatten())
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(10, activation='softmax')) # 왜 시그모이드를 사용할까? 

model.compile(loss='categorical_crossentropy',optimizer='adam',
                    metrics=['accuracy'])



model.fit(x_train, y_train, epochs=10, verbose=1,
batch_size=300)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('metrix : ', loss[1] )


#metrix :  0.8108000159263611