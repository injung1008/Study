import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN , LSTM, Bidirectional
from datetime import datetime
import tensorflow as tf
'''
tensorboard --logdir=./logs/fit/ 
'''


#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape) # (4, 3) (4,)

x = x.reshape(4,3,1) #@ 피쳐 몇 개씩 자르는지 = 1 
#@ (batch_size, timesteps, feature) 위에꺼 모양나타내느거임 4 = batch_size 3 =timesteps


#2. 모델 
model = Sequential()
# model.add(SimpleRNN(units = 10, activation='relu', input_shape=(3,1))) 아래랑 같은  표현  timesteps, feature

#model.add(Bidirectional(LSTM(units = 10, activation='relu', input_length=3, input_dim=1))) #이게 안되는 이유? 
model.add(LSTM(units = 10, activation='relu', input_length=3, input_dim=1, return_sequences=True)) 
model.add(Bidirectional(LSTM(units = 10, activation='relu')))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))
model.summary()

exit()
'''
lstm*2 bid빼고 돌려보면 앎
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 10)             480
_________________________________________________________________
bidirectional (Bidirectional (None, 20)                1680
_________________________________________________________________
dense (Dense)                (None, 16)                336
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 136
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 18
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 3
=================================================================
Total params: 2,653
Trainable params: 2,653
Non-trainable params: 0
_________________________________________________________________
'''
# Define the Keras TensorBoard callback.
logdir="logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x,y, epochs=500, batch_size=1, callbacks=[tensorboard_callback])
'''
Total params = recurrent_weights + input_weights + biases
(num_units*num_units)+(num_features*num_units) + (1*num_units)
(num_features + num_units)* num_units + num_units

(input + bias) *  output + * output *output
=3 * ( input + bias + output) * output
'''



#4. 평가, 예측
x_input = np.array([[5], [6], [7]]).reshape(1,3,1)
results = model.predict(x_input)
print(results) 

# [[8.628703]]
# [[8.019494]]