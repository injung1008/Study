import numpy as np
from numpy import array 
from tensorflow.keras.layers import Dense,Input, SimpleRNN, LSTM,GRU
from tensorflow.keras.models import Sequential, Model

#1. data

x1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
                [5,6,7],[6,7,8],[7,8,9],[8,9,10],
                [9,10,11],[10,11,12],
                [20,30,40],[30,40,50],[40,50,60]])

x2 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
                [50,60,70],[60,70,80],[70,80,90],[80,90,100],
                [90,100,110],[100,110,120],
                [2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = array([55,65,75]).reshape(1,3,1)
x2_predict = array([65,75,85]).reshape(1,3,1)

print(x1.shape, x2.shape,y.shape) #(13, 3) (13, 3) (13,)

x1 = x1.reshape(13,3,1)  #3차원 -> (batch_size , timesteps, feature)
x2 = x2.reshape(13,3,1)  

#2-1 모델1 구성 
input1 = Input(shape=(3,1))
dense1 = LSTM(units=10, activation='relu')(input1) #히든 레이어 구성 
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)

#2-2 모델2 구성 
input2 = Input(shape=(3,1))
dense11 = LSTM(units=10, activation='relu')(input2)
dense12 = Dense(10, activation='relu')(dense11)
dense13 = Dense(10, activation='relu')(dense12)
dense14 = Dense(10, activation='relu')(dense13)
output2 = Dense(1)(dense14)


from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)
#concatenate, Concatenate 소문자와 대문자의 차이는 소문자 = 메소드, 대문자- 클래스
model = Model(inputs=[input1,input2] , outputs=last_output)

model.summary()

#3. 컴파일 , 훈련
#metrics=['mae', 'mse'] => 두개도 가능 
model.compile(loss= 'mse', optimizer='adam')
# x =3 개, y= 1개 이기땜시
model.fit([x1,x2], y, epochs=200, batch_size=1, verbose=1 )

#평가예측

result = model.predict([x1_predict,x2_predict])
print('result : ', result)
