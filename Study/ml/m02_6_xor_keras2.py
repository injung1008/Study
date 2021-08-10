from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 (xor gate)
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]
# and gate  둘중에 하나라도 0 이면 0 이고 둘다 1일경우 1이다 
# or gate  둘중에 하나라도 1이면 1이다 둘다 0일 경우만 0 
# xor gate 둘이 같으면 0,0 = 0  / 1,1, = 0  이고 둘이 다를경우 0,1 =1 이다 


#2. model 
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

import tensorflow as tf
#4. 평가, 예측 
y_predict = model.predict(x_data)
y_predict = np.round_(y_predict)


print(x_data, "의 예측 결과 : \n ", y_predict)


#[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측 결과 :  [0 1 1 0]

results = model.evaluate(x_data, y_data)
print('model.score : ', results[1])

#roud로 반올림 처리 해줘서 맞춰줘야한다 

acc = accuracy_score(y_data, y_predict)
print("accuracy_score :", acc)

# loss: 0.5562 - acc: 0.5333
# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측 결과 : 
#   [[0.]
#  [1.]
#  [0.]
#  [0.]]
# 1/1 [==============================] - ETA: 0s - loss: 0.1/1 [==============================] - 0s 104ms/step - loss: 0.4614 - acc: 0.7500
# model.score :  0.75
# accuracy_score : 0.75