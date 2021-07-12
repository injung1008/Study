from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터 

#훈련 50%/ 검증 30% 테스트 20% 

#훈련용
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

#train_test)spilt으로 나누기 



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state=66)

print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(x_val)
print(y_val)

#2.모델 구성  
model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(1))


# 3. 컴파일 훈련 
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, verbose= 1, batch_size=1, validation_split=0.3, shuffle=True)

# -> 이 과정에서 가중치가 생성이 된다 

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([11]) 
print(y_predict)
