from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred = [6] #6에 대한 예측값 

model = Sequential()
model.add(Dense(1, input_dim=2))

model.compile(loss='mse', optimizer='adam')

model.fit(x,y, epochs=16000, batch_size=1)

loss = model.evaluate(x,y)
print('loss : ', loss)

y_predict = model.predict(x)
print('x_pred 의 예측값 : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print(r2)

#과제 2 
#R2를 0.9 올려라 ! 