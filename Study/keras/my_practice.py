from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([1,2,3])
y = np.array([1,2,3])

model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(lose = 'mse', optimizer='adam')

model.fit(x,y, epochs=100, batch_size=1)

loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict([10])
print('result : ', result)