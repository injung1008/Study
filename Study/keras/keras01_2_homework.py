from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred = [6] #6에 대한 예측값 

model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss='mse', optimizer='adam')

model.fit(x,y, epochs=1600, batch_size=1)

loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict(x_pred)
print('x_pred 의 예측값 : ', result)

