from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten ,MaxPooling2D


model = Sequential()                         #(N,10,10,1)
model.add(Conv2D(10, kernel_size=(2,2), 
     padding='same', input_shape=(10, 10, 1)))  #(N,10,10,10) 
model.add(Conv2D(20, (2,2)))                #(N,9,9,20)
model.add(Conv2D(30, (2,2)))                #(N,8,8,30)
model.add(MaxPooling2D()) #쉐이프를 반으로줄인다 #(N,4,4,30)
#model.add(Flatten())                              #(N,480) -> 4x4x30 
# MaxPooling2D = 크기는 줄었지만 연산은 하지 않는다 하지만 데이터 손실이 있을수 있어서 잘 판단해야함              
model.add(Conv2D(15, (2,2)))                 # #(N,3,3,15)  
model.add(Flatten())                         #(N,135)        
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid')) # 왜 시그모이드를 사용할까? 
model.summary()