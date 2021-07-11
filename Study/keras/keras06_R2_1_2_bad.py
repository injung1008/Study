#R2를 음수가 아닌 0.5이하로 만들어라 
#2. 데이터 건들지 않기
#3. 레이어는 인풋 아웃풋 포함 6개 이상
#4. batch_size = 1
#5. eppoch = 100이상
#6. 히든레이어의 노드는 10개 이상 1000개 이하.
#7. train 70% 

#1. 데이터 

#훈련 50%/ 검증 30% 테스트 20% 

# #훈련용
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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle=True, random_state=100)
print(x_test)
print(x_train)


#2.모델 구성  
model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(1000, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))

#3. 컴파일 훈련 
model.compile(loss='mse', optimizer='adam')


model.fit(x_train, y_train, epochs=200, batch_size=1)
# -> 이 과정에서 가중치가 생성이 된다 

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) 
print(y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print(r2)