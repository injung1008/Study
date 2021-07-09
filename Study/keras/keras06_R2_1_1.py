from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#1. 데이터 

#훈련 50%/ 검증 30% 테스트 20% 

# #훈련용
x = np.array(range(100))
y = np.array(range(1,101))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle=True, random_state=66)
print(x_test)
print(x_train)


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

y_predict = model.predict(x_test) 
print(y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print(r2)