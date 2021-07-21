import numpy as np
import matplotlib.pyplot as plt

#1. 데이터 
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) - 얘는 4차원을 받는데 3차원이라서 차원을 늘려야한다 
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,) 

# #전처리

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

#scaler = StandardScaler()
scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray() #(60000, 10)
y_test = encoder.transform(y_test).toarray() #(10000, 10)
print(y_test.shape)





# # reshape = 3차원 데이터 4차원으로 늘리기 데이터의 내용물과 순서가 바뀌면안된다

# # #2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape,LSTM

model = Sequential()
# model.add(Conv2D(10, kernel_size=(2,2), padding='same', input_shape=(28,28,1)))
# model.add(Flatten())                                              #(N,180)
model.add(Dense(10, activation='relu', input_shape=(28,28)))
model.add(Flatten()) #(N,280)
model.add(Dense(784))     #(N,784)      
model.add(Reshape((28,28,1)))   #(N,28,28,1)
model.add(Conv2D(64, kernel_size=(2,2)))
model.add(Conv2D(64, kernel_size=(2,2)))                                
model.add(Conv2D(64, kernel_size=(2,2)))  
model.add(Reshape((40000,1)))   #(N,25,25,64)로 내려 왔기 때문에 25 * 25* 64 
model.add(LSTM(units=10, activation='relu'))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(9))
model.add(Dense(10, activation='softmax'))
model.summary()

# #3. 컴파일 훈련 metrics = 'acc'
model.compile(loss='categorical_crossentropy',optimizer='adam',
                    metrics=['accuracy'])



model.fit(x_train, y_train, epochs=10, verbose=1,
batch_size=300)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('metrix : ', loss[1] )



# #acc로만 판단  metrix :  0.9440000057220459