import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

#1. data
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
                [5,6,7],[6,7,8],[7,8,9],[8,9,10],
                [9,10,11],[10,11,12],
                [20,30,40],[30,40,50],[40,50,60]])
#[1,2,3] = 한개의 timesteps 이고 이걸 몇개의 feature로 짜를것인지는 선택의 몫
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70]).reshape(1,3,1)

#예상값도 차원도 늘려줘야함 현재 2차원 
print(x.shape, y.shape) #(13, 3) (13,)


# #rnn은 3차원을 받기 때문에 reshape로 feature 1을 추가해서 3차원 만들어줌 
x = x.reshape(13,3,1)  #3차원 -> (batch_size , timesteps, feature)
#x = x.reshpae(x.shape[0], x.shape[1],1) 이렇게 하면 매번 안바꿔도됨 


# # 모델 구성
model = Sequential()
#LSTM 은 인풋은 3차원이지만 아웃풋으로 2차원을 보내기때문에 dense와 연결이 가능했다 하지만 lstm을 두개 연결하려면 
# 어떻게 해야할까? -> return_sequences=True 
model.add(LSTM(units=10, activation='relu', input_shape=(3,1), return_sequences=True))
model.add(LSTM(units=7, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

'''
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 10)             480
_________________________________________________________________
lstm_1 (LSTM)                (None, 7)                 504
_________________________________________________________________
dense (Dense)                (None, 5)                 40
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 6
'''


# #컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=250, batch_size=1)

# #4. 평가 예측 
# #x_input = np.array([5,6,7]).reshape(1,3,1)
results = model.predict(x_predict)
print(results)

# 결과값이 80 근접하게 튜닝하시오 
# [[81.4717]]
# [[78.458984]]
# [[80.669304]] 훈련 250
# [[80.528755]]