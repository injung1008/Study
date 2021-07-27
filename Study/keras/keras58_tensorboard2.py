#
from tensorflow.keras.models import Sequential # 모델에는 두가지가 있음 순차형/함수형 
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow import keras
import tensorflow as tf

from datetime import datetime

# Deep 인공신경망의 layer의 깊이 / layer는 node로 이루어져 있다.


#1. 데이터 전처리(특기)
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 2, 4, 3, 5, 6, 7, 8, 9, 10]
x_pred = [6]

x = np.array(x)
y = np.array(y)

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1)) # 예시 그림을 모델링 한 것

# Define the Keras TensorBoard callback.
logdir="./_save/_gragh/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1)

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, callbacks=[tensorboard_callback], validation_split=0.2)


#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict(x_pred) 
print('6의 예측값 : ', result)


'''
tensorboard --logdir=./_save/_gragh
'''

