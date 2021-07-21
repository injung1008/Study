


#전처리 

import numpy as np
import matplotlib.pyplot as plt

#1. 데이터 
from tensorflow.keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


# #전처리 
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)
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
x_train = x_train.reshape(60000, 28,28, 1) 
x_test = x_test.reshape(10000, 28,28, 1) 

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout 

# model = Sequential()
# model.add(Conv2D(156, kernel_size=(2,2), padding='valid',activation='relu', input_shape=(28,28,1)))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D())
# model.add(Conv2D(128, (2,2), padding='valid', activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(64, (2,2), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())        
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax')) 

# model.save('./_save/keras45_1_save_model.h5')


#저장해놓은 모델 불러오기 
from tensorflow.keras.models import load_model
model = load_model('./_save/keras45_1_save_model.h5')
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',
                    metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping

import time
start_time = time.time()

es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)


hist = model.fit(x_train, y_train, epochs=2, verbose=1,validation_split=0.2,
batch_size=300, callbacks=[es])

end_time = time.time() - start_time

loss = model.evaluate(x_test, y_test)
print("===============================================")
print("걸린시간 : ", end_time)
print('loss : ', loss)
print('acc : ', loss[1] )


#시각화
import matplotlib.pyplot as plt
# #1. 
# plt.subplot(2,1,1) #그림을 2개를 그려라 1행1열
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# #2.
# plt.subplot(2,1,2) #그림을 2개를 그려라 1행1열
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_loss'])
# plt.grid()
# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc'])

# plt.show()

# accuracy: 0.2709


'''

걸린시간 :  232.18859720230103      
loss :  [0.041484419256448746, 0.9897000193595886]
acc :  0.9897000193595886

걸린시간 :  11.047605752944946   
loss :  [0.08793385326862335, 0.9743000268936157]
acc :  0.9743000268936157        
'''