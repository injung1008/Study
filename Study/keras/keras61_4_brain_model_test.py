import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# LOAD LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#넘파이, 판다스, MATPLOT은 기본으로 항상

from sklearn.model_selection import train_test_split
# train.test 셋을 쉽게 분리하기 위해서

from keras.utils.np_utils import to_categorical
# cnn을 통해 최종적으로 결과를 받으면 라벨수만큼의 각각의 확률값으로 반환된다. 결과값을 받기 편하게 하기위한 함수
from keras.models import Sequential
# 케라스 모델구성기본 함수
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
# 케라스에서 필요한 레이어들 간편하게 쓸수 있다.
from keras.preprocessing.image import ImageDataGenerator
# 이미지를 조금 변화해줌으로써 성능을 올릴수 있다. 그랜드 마스터 Chris Deotte 의 25 Million Images! [0.99757] MNIST 커널에서 참고했다.(그

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
# 콜벡 모델이 어떤 기준으로 돌다가 멈추고 저장하고 하는것들을 설정해줄수 있다.
import warnings
warnings.filterwarnings('ignore')
# 지저분하게 워닝뜨는걸 막아준다.

x_train = np.load('D:\Study\_npy\k59_3_train_x.npy')
y_train = np.load('D:\Study\_npy\k59_3_train_y.npy')

x_test = np.load('D:\Study\_npy\k59_3_test_x.npy')
y_test = np.load('D:\Study\_npy\k59_3_test_y.npy')


fig = plt.figure(figsize=(10,10))

for i in range(10):
    i += 1
    plt.subplot(2,5,i)
    plt.imshow(x_train[i])
    plt.axis('off')
plt.show()


# model = Sequential()

# model.add(Conv2D(32,kernel_size=3,activation= 'relu', input_shape = (150,150,3) ))
# model.add(BatchNormalization())
# model.add(Conv2D(32,kernel_size=3,activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(32,kernel_size=5,activation = 'relu', padding='same',strides=2))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# model.add(Conv2D(64,kernel_size=3,activation= 'relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(64,kernel_size=3,activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(64,kernel_size=5,activation = 'relu', padding='same',strides=2))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# model.add(Conv2D(12,kernel_size=4,activation= 'relu'))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Dropout(0.8))
# model.add(Dense(1,activation='sigmoid'))
# model.summary()

# model.compile(optimizer='adam', loss = "binary_crossentropy", metrics=['acc'])

# # 콜벡은 이렇게 선언해서 callbacks에 담아놓자
# earlyStopping = EarlyStopping(patience=10, verbose=0)
# reduce_lr_loss = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=0)
# tqdm = TqdmCallback(verbose=0) #진행율 표시해준다.(없으면 답답하다)
# callbacks = [earlyStopping, reduce_lr_loss, tqdm]

# hist = model.fit(x_train,y_train,
#                               epochs = 10,
#                               steps_per_epoch = 30,
#                               validation_split=0.2,
#                               callbacks=callbacks,
#                               verbose=0)


# print('train_acc:{0:.5f} , val_acc:{1:.5f}'.format(max(hist.history['acc']),max(hist.history['val_acc'])))


# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']


# e_loss = model.evaluate(x_test, y_test)
# print('e_loss : ', e_loss )

# #print('acc 전체 : ', acc)

# print('acc : ', acc[-1])
# print('loss : ', loss)
# print('val acc : ', val_acc)
# print('val loss : ', val_loss)

# import tensorflow as tf


# temp = model.predict(x_test)
# print('원본 : ', temp)
# temp = tf.argmax(temp, axis=1)
# # temp = pd.DataFrame(temp)
# # print('예측값 : ', temp)
# print('원래값 : ',y_test[:5])






# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.title('Accuracy', fontsize=14)
# plt.xlabel('Epoch', fontsize=14)
# plt.ylabel('Accuracy',fontsize=14)
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
