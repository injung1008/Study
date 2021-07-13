import numpy as np
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(442, 10) (442,)

print(datasets.feature_names)

print(datasets.DESCR)

print(y[:30])
print(np.min(y), np.max(y))
#2 모델구성

#3. 컴파일 훈련

#4. 평가, 예측 
# # 완료 하시오 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle=True, random_state=66)

# model 구성 

input1 = Input(shape=(10,)) #인풋레이어 구성
dense1 = Dense(3)(input1) #히든 레이어 구성 
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3) 

model = Model(inputs=input1, outputs=output1)
model.summary()


model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, verbose= 1, batch_size=3, validation_split=0.2, shuffle=True)


loss = model.evaluate(x_test, y_test)
print('loss : ', loss )

y_predict = model.predict(x_test)
print('y_predict : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 : ' , r2)

