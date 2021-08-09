# overfit  극복 
# 1. 전체 훈련 데이터의 양이 많을 수록 과적합이 준다 
# 2. nomalization (정규화 - 데이터 압축 min-max 와 비슷한 ) 
# 3. dropout  레이어의 노드를 속아내는것 


#전처리 

import numpy as np
import matplotlib.pyplot as plt

#1. 데이터 
from tensorflow.keras.datasets import cifar100
(x_train, y_train),(x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

# #전처리 
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(np.unique(y_train)) #[0- 100]
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder()
# encoder.fit(y_train)
# y_train = encoder.transform(y_train).toarray() 
# y_test = encoder.transform(y_test).toarray() 
# #print(y_test.shape)

#데이터 전처리 

#컨볼루션 데이터는 4차원으로 만들어줬지만, 표준화는 2차원 데이터만 가능하기떄문에
# #reshape을 통해서 다시 데이터차원 줄여주기 

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
#print(x_test)

from sklearn.preprocessing import StandardScaler, MinMaxScaler ,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
scaler = QuantileTransformer()
# #scaler = PowerTransformer()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
#! 두개를 한번에 진행 할 때 fit_transform사용 하지만 fit은 항상 훈련데이터만 사용
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#print(x_test)

#차원 늘려주기 
x_train = x_train.reshape(50000, 32,32, 3) 
x_test = x_test.reshape(10000, 32,32, 3) 

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D 

model = Sequential()
model.add(Conv2D(156, kernel_size=(2,2), padding='valid',activation='relu', input_shape=(32,32,3)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())

# model.add(Flatten())        
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax')) 


from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.1)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,
                    metrics=['acc'])
#지정하는건 컴파일이지만 실행하는건 fit이다 ->  일정한 값이 적용되지 않으면 러닝레이트를 줄이는것은 callbacks

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import time
start_time = time.time()
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)
#es는 조건이맞으면 끝나고, reduce_lr도 조건 맞으면 끝나지만, factor=0.5 로 해놓으면 감소가 없으면 러닝 레이트가 0.5 만큼 줄어든다.

hist = model.fit(x_train, y_train, epochs=300, verbose=1,validation_split=0.2,
batch_size=512, callbacks=[es, reduce_lr])


end_time = time.time() - start_time

loss = model.evaluate(x_test, y_test)
print("===============================================")
print("걸린시간 : ", end_time)
print('loss : ', loss)
print('acc : ', loss[1] )

import matplotlib.pyplot as plt
#1. 
plt.subplot(2,1,1) #그림을 2개를 그려라 1행1열
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

#2.
plt.subplot(2,1,2) #그림을 2개를 그려라 1행1열
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_loss'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

#accuracy: 0.2709


'''
걸린시간 :  696.760805606=======>......] - ETA: 0s -842                      ========>.....] - ETA: 0s -
loss :  [2.48221874237060=========>....] - ETA: 0s -55, 0.3792000114917755]  ==========>...] - ETA: 0s -
acc :  0.3792000114917755============>.] - ETA: 0s -        
'''