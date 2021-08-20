#실습 cifar10 완성 

from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.applications import VGG16, VGG19


from tensorflow.keras.datasets import cifar100
(x_train, y_train),(x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)



#데이터 전처리 

#컨볼루션 데이터는 4차원으로 만들어줬지만, 표준화는 2차원 데이터만 가능하기떄문에
# #reshape을 통해서 다시 데이터차원 줄여주기 

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
#print(x_test)

from sklearn.preprocessing import StandardScaler, MinMaxScaler ,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MinMaxScaler()
scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_test)

#차원 늘려주기 
x_train = x_train.reshape(50000, 32,32, 3) 
x_test = x_test.reshape(10000, 32,32, 3) 

import tensorflow as tf 

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D 

#vgg는 이미지분류 모델이기 때문에 바로 3차원 받아버림 
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
vgg16.trainable=False #가중치를 동결한다 
model = Sequential()
model.add(vgg16)
# model.add(Flatten())
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
model.add(global_average_layer)
# model.add(Dense(125, activation='relu'))
# model.add(Dense(65, activation='relu'))
model.add(Dense(100, activation='softmax'))
model.summary()



# print(len(model.weights)) 
# #26(바이어스하나 레이어 하나 2개 * 13) -> 30 (트레이너블 파람은 0이 되고 26에서 레이어 2개를 늘려서 2*2 )
# print(len(model.trainable_weights)) #0 -> 4
from tensorflow.keras.optimizers import Adam
import time
from tensorflow.keras.callbacks import EarlyStopping
optimizer = Adam(lr=0.01)
model.compile(loss='sparse_categorical_crossentropy'
,optimizer= optimizer, metrics=['accuracy'])

start_time = time.time()
es = EarlyStopping(monitor='val_acc', patience=5, mode='max', verbose=3) # 

model.fit(x_train,y_train, epochs=100, batch_size=100,
 validation_split=0.2, callbacks=[es])

end_time = time.time() - start_time
print(end_time)


loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('metrix : ', loss[1] )
