import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. data
x = np.array([[1,2],[2,3],[3,4],[4,5],[5,6]])
#[1,2,3] = 한개의 timesteps 이고 이걸 몇개의 feature로 짜를것인지는 선택의 몫
y = np.array([3,4,5,6,7])

print(x.shape, y.shape) #(5, 2) (5,)

#rnn은 3차원을 받기 때문에 reshape로 feature 1을 추가해서 3차원 만들어줌 
x = x.reshape(5,2,1)  #3차원 -> (batch_size , timesteps, feature)
# 4 = 전체 데이터 

# 모델 구성
model = Sequential()
model.add(SimpleRNN(units=10, activation='relu', input_shape=(2,1))) 
#input_shape=(3,1) 앞에 전체 데이터 무시 (행무시 = 4 )
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()