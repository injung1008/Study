

import numpy as np
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(442, 10) (442,)

#2 모델구성

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size = 0.7, shuffle=True, random_state=9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
#scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape)#(309, 10)
# print(x_test.shape)#(133, 10)
# print(y_test.shape)#(133,)


x_train = x_train.reshape(309, 10, 1) 
x_test = x_test.reshape(133, 10, 1) 


# # # model 구성 


model = Sequential()
model.add(Conv1D(64, kernel_size=2 ,input_shape=(10,1)))
model.add(Flatten())
model.add(Dense(62))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')


model.fit(x_train, y_train, epochs=200, verbose=1, batch_size=30, validation_split=0.2)


loss = model.evaluate(x_test, y_test)
print('loss : ', loss )

y_predict = model.predict(x_test)
print('y_predict : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)
#r2 :  0.5092737230365898

#r2 :  0.5887508089105278