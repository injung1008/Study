from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
#1. 데이터 

#훈련 50%/ 검증 30% 테스트 20% 

#훈련용
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
#평가용
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])


#2.모델 구성  
model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(1))


#3. 컴파일 훈련 
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=1) 
# -> 이 과정에서 가중치가 생성이 된다 

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([11]) 
print(y_predict)

# result = model.predict([12])
# print('12의 예측값 : ', result)
# plt.scatter(x,y)
# plt.plot(x,y_result, color='red')
# plt.show()