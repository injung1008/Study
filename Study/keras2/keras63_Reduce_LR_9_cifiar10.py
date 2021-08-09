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
from tensorflow.keras.layers import Dense, LSTM, Flatten, Conv1D

model = Sequential()
model.add(Conv1D(64, 2,input_shape=(1024,3)))
model.add(Flatten())
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))
model.add(Dense(10, activation='softmax'))


from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.1)
model.compile(loss='mse',optimizer=optimizer,
                    metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import time
start_time = time.time()
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)
#es는 조건이맞으면 끝나고, reduce_lr도 조건 맞으면 끝나지만, factor=0.5 로 해놓으면 감소가 없으면 러닝 레이트가 0.5 만큼 줄어든다.

hist = model.fit(x_train, y_train, epochs=300, verbose=1,validation_split=0.2,
batch_size=1, callbacks=[es, reduce_lr])


end_time = time.time() - start_time

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('metrix : ', loss[1] )
#metrix :  0.19470000267028809