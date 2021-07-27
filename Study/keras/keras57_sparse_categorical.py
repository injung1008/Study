import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D


# 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# 전처리
x_train = x_train.reshape(60000, 784)  
x_test = x_test.reshape(10000, 784)
y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(10000,1)

print(type(y_train))

# from sklearn.preprocessing import OneHotEncoder
# en = OneHotEncoder(sparse=False)
# y_train = en.fit_transform(y_train)
# y_test = en.fit_transform(y_test)


print(type(y_train))
from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer, RobustScaler
scaler =MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(60000, 28, 28, 1) # 컨글루션은 4차원 데이터를 받기 때문에 reshape
x_test = x_test.reshape(10000, 28, 28, 1)


# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = PowerTransformer()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) 
# x_test = scaler.transform(x_test)

#2. 모델
model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', input_shape=(28,28, 1)))
model.add(Conv2D(80, (2,2), activation='relu'))  
model.add(Conv2D(60, (2,2), activation='relu'))  
model.add(MaxPool2D())               
model.add(Flatten()) 
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))



#3 컴파일, 훈련 metrics=['acc']
#@ 원핫인코딩 안해도됨 개꿀
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=3) # 

model.fit(x_train,y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es])


#4. 평가, 예측 predict할 필요는 없다datetime A combination of a date and a time. Attributes: ()
loss = model.evaluate(x_test,y_test) 
print('loss : ', loss[0])
print('acc : ', loss[1])

# acc로만 판단해보자

# loss :  0.06797253340482712
# acc :  0.9839000105857849