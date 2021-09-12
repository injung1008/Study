import numpy as np

#1. data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일 , 훈련
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer = Adam(lr=0.001)
#lr = learning rate
# 러닝레이트와 에폭이 적당해야 골짜기 까지 값이 도달 할 수 있다  

optimizer = SGD(lr=0.0001,momentum=0.0,decay=0.0,nesterov=False)
# SGD = 확률적 경사하강법 lr이높으면 nan값이 나옴  0.0027057426050305367 결과물 :  [[10.945305]]
# optimizer = Adagrad() #loss :  1.1757640550058568e-06 결과물 :  [[10.997675]]

# optimizer = RMSprop() #loss :  0.001249517546966672 결과물 :  [[11.009884]]
# optimizer = Adadelta() #loss :  8.612983703613281 결과물 :  [[5.7320623]]
# optimizer = Adamax()  #loss :  1.708831476321393e-08 결과물 :  [[11.0000305]]
# optimizer = Nadam() #loss :  1.5989907353741728e-07 결과물 :  [[10.9993515]]

#일정한 변화가 없으면 값을 줄여라 

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x,y,epochs=100, batch_size=1)

#4. 평가 예측
loss, mse = model.evaluate(x,y)
y_pred = model.predict([11])

print('loss : ', loss, '결과물 : ', y_pred)

#lr = 0.001 - 모델이 간단하기 때문에 lr이 작을 수록 결과값이 좋다 & 적당한 에폭 
#- mse: 4.6896e-14
#loss :  4.6895819882540254e-14 결과물 :  [[11.]]
