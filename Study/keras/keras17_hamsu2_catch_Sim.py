from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred = [6] #6에 대한 예측값 

input1 = Input(shape=(1,))
dense1 = Dense(3)(input1)
dense2 = Dense(8)(dense1)
dense3 = Dense(2)(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs = input1, outputs=output1)

model.compile(loss='mse', optimizer='adam')

model.fit(x,y, epochs=1500, batch_size=4)

loss = model.evaluate(x,y)
print('loss : ', loss)

y_predict = model.predict(x)
print('x_pred 의 예측값 : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print(r2)

#과제 2 
#R2를 0.9 올려라 ! 