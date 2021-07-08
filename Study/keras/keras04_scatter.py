from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
#1. 데이터 

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,7,9,3,8,11])


#2.모델 구성  
model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(1))


#3. 컴파일 훈련 
model.compile(loss='mse', optimizer='adam')

model.fit(x,y, epochs=1000, batch_size=1) 
# -> 이과정에서 가중치가 생성이 된다 

#4. 평가, 예측 
loss = model.evaluate(x, y)
print('loss : ', loss)

#원래는 훈련데이터와 평가데이터는 달라야한다  훈련 50%/ 검증 30% 테스트 20% 
# result = model.predict([12])
# print('12의 예측값 : ', result)

y_result = model.predict(x) # x데이터 전체의 데이터 예측값이 나온다  (x,y)가 나오면서 이어지며 예측값을 이 나오게 된다 


plt.scatter(x,y)
plt.plot(x,y_result, color='red')
plt.show()