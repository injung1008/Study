from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터 

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2.모델 구성  
model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(8))
model.add(Dense(1))


#3. 컴파일 훈련 
model.compile(loss='mse', optimizer='adam')

model.fit(x,y, epochs=2000, batch_size=1) 
# -> 이과정에서 가중치가 생성이 된다 

#4. 평가, 예측 
loss = model.evaluate(x, y)
print('loss : ', loss)

#원래는 훈련데이터와 평가데이터는 달라야한다  훈련 50%/ 검증 30% 테스트 20% 
result = model.predict([6])
print('4의 예측값 : ', result)

