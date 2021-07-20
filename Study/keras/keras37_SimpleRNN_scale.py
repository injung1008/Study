import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM,GRU

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
#LSTM 파라미터 = simpleRNN * 4 (lstm 안에들어있는 연산 게이트가 simpe RNN 보다 4배 더있다 )
model.add(SimpleRNN(units=32, activation='relu', input_shape=(3,1)))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#model.summary()


# #컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=250, batch_size=1)

# #4. 평가 예측 
# #x_input = np.array([5,6,7]).reshape(1,3,1)
results = model.predict(x_predict)
print(results)

# # 결과값이 80 근접하게 튜닝하시오 
#[[79.93112]]
#[[80.48328]]
#[[80.203896]]
#[[80.03594]]