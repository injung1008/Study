import numpy as np
import matplotlib.pyplot as plt

#1. 데이터 
from tensorflow.keras.datasets import cifar100
(x_train, y_train),(x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

# #전처리 

print(np.unique(y_train)) #[0- 100]
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#print(y_test.shape)

#데이터 전처리 

#컨볼루션 데이터는 4차원으로 만들어줬지만, 표준화는 2차원 데이터만 가능하기떄문에
# #reshape을 통해서 다시 데이터차원 줄여주기 

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
#print(x_test)

from sklearn.preprocessing import StandardScaler, MinMaxScaler ,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = MaxAbsScaler()
#scaler = RobustScaler()
# scaler = QuantileTransformer()
# #scaler = PowerTransformer()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#print(x_test)

#3차원 으로 늘려주기 
x_train = x_train.reshape(50000, 32*32, 3) 
x_test = x_test.reshape(10000, 32*32, 3) 

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling1D,Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Dense(100, input_shape=(32*32,3))) #데이터 차원 3차원으로 맞추기                                       
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(GlobalAveragePooling1D())
#3차원 데이터로 진행할 경우 1D는 가능하다 
model.add(Dense(100, activation='softmax')) 

model.compile(loss='categorical_crossentropy',optimizer='adam',
                    metrics=['accuracy'])



model.fit(x_train, y_train, epochs=50, verbose=1, validation_split=0.02
,batch_size=100)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('metrix : ', loss[1] )

#accuracy: 0.2709