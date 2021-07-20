import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. data
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
#[1,2,3] = 한개의 timesteps 이고 이걸 몇개의 feature로 짜를것인지는 선택의 몫
y = np.array([4,5,6,7])

print(x.shape, y.shape) #(4, 3) (4,)

#rnn은 3차원을 받기 때문에 reshape로 feature 1을 추가해서 3차원 만들어줌 
x = x.reshape(4,3,1)  #3차원 -> (batch_size , timesteps, feature)


# 모델 구성
model = Sequential()

#model.add(LSTM(units=5, activation='relu', input_shape=(3,1)))
model.add(GRU(units=10, activation='relu', input_shape=(3,1)))

model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()
'''
#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=250, batch_size=1)

#4. 평가 예측 
x_input = np.array([5,6,7]).reshape(1,3,1)
results = model.predict(x_input)
print(results)
'''

