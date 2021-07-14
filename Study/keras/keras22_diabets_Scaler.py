# 실습 diabets
#1. loss와 r2로 평가를함 
#3. Minmax 와 standard 결과를 명시 

import numpy as np
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(442, 10) (442,)

#2 모델구성

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size = 0.7, shuffle=True, random_state=9)

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
scaler = PowerTransformer()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# model 구성 
model = Sequential()
model.add(Dense(1, input_dim=10))
model.add(Dense(10, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, verbose= 1, batch_size=1, validation_split=0.2, shuffle=True)

#모델 평가 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss )
#예측 
y_predict = model.predict(x_test)
print('y_predict : ', y_predict)
#r2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 : ' , r2)


# mimax  r2 :  0.6090073606596966 /loss :  2160.73193359375
# StandardScaler r2 =  0.5791065826591884 / loss :  2325.971923828125
# MaxAbsScaler - r2 = 0.5834101807753327
# RobustScaler - r2 =  0.4703624805730322
# QuantileTransformer - r2 =  0.5069239339048849
# PowerTransformer - r2 = 0.5573247698314562
