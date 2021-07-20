import numpy as np
import matplotlib.pyplot as plt

#1. 데이터 
from tensorflow.keras.datasets import cifar10
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

# #전처리 

# print(np.unique(y_train)) #[0 1 2 3 4 5 6 7 8 9]
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()

encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray() #(60000, 10)
y_test = encoder.transform(y_test).toarray() #(10000, 10)
#print(y_test.shape)

#데이터 전처리 

#컨볼루션 데이터는 4차원으로 만들어줬지만, 표준화는 2차원 데이터만 가능하기떄문에
# #reshape을 통해서 다시 데이터차원 줄여주기 

x_train = x_train.reshape(50000*3, 32*32)
x_test = x_test.reshape(10000*3, 32*32)
#print(x_test)

from sklearn.preprocessing import StandardScaler, MinMaxScaler ,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
#scaler = MinMaxScaler()
#scaler = StandardScaler()
# #scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler = QuantileTransformer()
# #scaler = PowerTransformer()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_test)

#차원 늘려주기 
x_train = x_train.reshape(50000, 1024, 3) 
x_test = x_test.reshape(10000, 1024, 3) 

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(units=10, activation='relu', input_shape=(1024,3)))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',
                    metrics=['accuracy'])



model.fit(x_train, y_train, epochs=50, verbose=1,
batch_size=100)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('metrix : ', loss[1] )
